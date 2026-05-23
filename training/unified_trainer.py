"""
unified_trainer.py
==================
Unified trainer for Arthedain SNNs supporting all learning paradigms.

Integrates:
  - Dual-timescale Hebbian (original Arthedain)
  - e-prop with eligibility traces
  - FORCE / RLS online learning
  - Dynamics-based fast learning (no weight changes)
  - Predictive coding with hybrid error mixing

Provides seamless switching between paradigms and supports
all combinations for hybrid learning.

Usage:
    trainer = UnifiedTrainer(
        rsnn, readout,
        mode="hybrid",  # or "hebbian", "eprop", "force", "dynamics", "pc"
        cfg=UnifiedConfig(...)
    )
    
    for x, y in stream:
        y_pred, error, info = trainer.step(x, target=y)

Configuration:
    Each mode can be configured independently, and the trainer
    automatically manages the appropriate state and updates.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any, Union
import warnings

from models.hebbian import DualHebbianAccumulator, HebbianConfig
from training.eprop import EPropConfig, EPropTrainer, make_eprop_trainer
from training.force_online import FORCEConfig, FORCETrainer, make_force_trainer
from training.dynamics_learning import DynamicsLearningConfig, DynamicsLearner, make_dynamics_learner
from models.predictive_coding import PCStack, build_pc_stack_for_arthedain, PCConfig


@dataclass
class UnifiedConfig:
    """Configuration for unified trainer."""
    
    # Mode selection
    mode: str = "hybrid"  # "hebbian", "eprop", "force", "dynamics", "pc", "hybrid"
    
    # Learning rates (used by multiple modes)
    lr_readout: float = 2e-3
    lr_recurrent: float = 5e-5
    
    # Hebbian parameters (original Arthedain)
    hebbian_tau_fast: float = 5.0
    hebbian_tau_slow: float = 50.0
    hebbian_alpha: float = 0.7  # fast trace weight
    hebbian_beta: float = 0.3   # slow trace weight
    
    # e-prop parameters
    eprop_tau_eligibility: float = 20.0
    eprop_tau_filter: float = 50.0
    eprop_learning_rate: float = 5e-5
    eprop_surrogate: str = "piecewise_linear"  # or "exponential"
    
    # FORCE parameters
    force_mode: str = "rls_readout_only"  # "rls_full", "rls_readout_only"
    force_alpha_rls: float = 1.0
    force_forgetting: float = 0.9995
    
    # Dynamics learning parameters
    dynamics_n_contexts: int = 4
    dynamics_adapt_steps: int = 50
    dynamics_lr_state: float = 0.1
    
    # Predictive coding parameters
    pc_lr_gen: float = 1e-4
    pc_lr_rec: float = 5e-5
    pc_tau_trace: float = 20.0
    pc_alpha_error: float = 0.5  # 0=pure PC, 1=pure global
    pc_use_stack: bool = True
    hidden_sizes: Optional[List[int]] = None  # Required for PC and multi-layer
    
    # Hybrid mixing (when mode="hybrid")
    hybrid_mix: Dict[str, float] = field(default_factory=lambda: {
        "hebbian": 0.3,
        "eprop": 0.3,
        "force": 0.4,
    })
    
    # Scheduling
    use_adaptive_alpha: bool = False
    alpha_drift_threshold: float = 0.3
    
    # Optimization settings
    use_cached_forward: bool = True  # Share forward pass in hybrid mode
    use_jit: bool = True  # Enable JIT compilation for hotspots
    skip_threshold: float = 0.001  # Skip FORCE updates below this error
    lr_warmup_steps: int = 100  # Warmup steps for learning rate
    
    # Monitoring
    track_eligibility: bool = True
    verbose: bool = False


class UnifiedTrainer:
    """
    Unified trainer supporting all Arthedain learning paradigms.
    
    Automatically manages:
      - Mode-specific state (eligibility traces, RLS matrices, etc.)
      - Integration with predictive coding stack
      - Hybrid combinations of learning methods
      - Adaptive scheduling based on error signals
    """
    
    SUPPORTED_MODES = ["hebbian", "eprop", "force", "dynamics", "pc", "hybrid"]
    
    def __init__(
        self,
        rsnn: nn.Module,
        readout: nn.Module,
        cfg: UnifiedConfig,
    ):
        self.rsnn = rsnn
        self.readout = readout
        self.cfg = cfg
        
        # Validate mode
        if cfg.mode not in self.SUPPORTED_MODES:
            raise ValueError(f"Unknown mode: {cfg.mode}. Choose from {self.SUPPORTED_MODES}")
        
        # Check for required hidden_sizes
        if cfg.hidden_sizes is None:
            # Try to infer from rsnn
            if hasattr(rsnn, 'hidden_size'):
                cfg.hidden_sizes = [rsnn.hidden_size]
            elif hasattr(rsnn, 'hidden_sizes'):
                cfg.hidden_sizes = rsnn.hidden_sizes
            elif hasattr(rsnn, 'W_rec'):
                cfg.hidden_sizes = [rsnn.W_rec.shape[0]]
            else:
                raise ValueError("Must provide hidden_sizes in config or use rsnn with hidden_size attribute")
        
        # Initialize mode-specific trainers
        self.trainers: Dict[str, Any] = {}
        self._init_trainers()
        
        # Initialize predictive coding stack
        self.pc_stack = None
        if cfg.pc_use_stack:
            self._init_pc_stack()
        
        # Tracking
        self.step_count = 0
        self.error_history = []
        self._current_alpha = cfg.pc_alpha_error
        
        # Cache for shared forward pass
        self._cached_spikes = None
        self._cached_y_pred = None
        
        # Learning rate schedule state
        self._lr_warmup_factor = 0.0
        
        # JIT compile eligibility updates if enabled
        if cfg.use_jit and "hebbian" in self.trainers:
            self._compile_hebbian_update()
        
    def _init_trainers(self):
        """Initialize trainers for the selected mode(s)."""
        cfg = self.cfg
        
        if cfg.mode == "hebbian" or cfg.mode == "hybrid":
            # Original Arthedain dual-timescale Hebbian
            hidden_size = cfg.hidden_sizes[0] if len(cfg.hidden_sizes) == 1 else cfg.hidden_sizes[-1]
            hebbian = DualHebbianAccumulator(
                shape=(hidden_size, hidden_size),
                tau_fast=cfg.hebbian_tau_fast,
                tau_slow=cfg.hebbian_tau_slow,
                alpha=cfg.hebbian_alpha,
                beta=cfg.hebbian_beta,
            )
            self.trainers["hebbian"] = hebbian
        
        if cfg.mode == "eprop" or cfg.mode == "hybrid":
            # e-prop with eligibility traces
            eprop_cfg = EPropConfig(
                tau_eligibility=cfg.eprop_tau_eligibility,
                tau_filter=cfg.eprop_tau_filter,
                learning_rate=cfg.lr_recurrent,
                surrogate_derivative=cfg.eprop_surrogate,
            )
            self.trainers["eprop"] = EPropTrainer(
                self.rsnn, self.readout, eprop_cfg, cfg.lr_readout
            )
        
        if cfg.mode == "force" or cfg.mode == "hybrid":
            # FORCE / RLS online learning
            force_cfg = FORCEConfig(
                mode=cfg.force_mode,
                alpha_rls=cfg.force_alpha_rls,
                forgetting_factor=cfg.force_forgetting,
                lr_readout=cfg.lr_readout,
                lr_recurrent=cfg.lr_recurrent,
            )
            self.trainers["force"] = FORCETrainer(self.rsnn, self.readout, force_cfg)
        
        if cfg.mode == "dynamics" or cfg.mode == "hybrid":
            # Dynamics-based fast learning
            dynamics_cfg = DynamicsLearningConfig(
                n_contexts=cfg.dynamics_n_contexts,
                n_adaptation_steps=cfg.dynamics_adapt_steps,
                lr_initial_state=cfg.dynamics_lr_state,
            )
            self.trainers["dynamics"] = DynamicsLearner(self.rsnn, self.readout, dynamics_cfg)
    
    def _init_pc_stack(self):
        """Initialize predictive coding stack for local errors."""
        if len(self.cfg.hidden_sizes) > 1 or self.cfg.pc_use_stack:
            self.pc_stack = build_pc_stack_for_arthedain(
                hidden_sizes=self.cfg.hidden_sizes,
                lr_gen=self.cfg.pc_lr_gen,
                lr_rec=self.cfg.pc_lr_rec,
                tau_trace=self.cfg.pc_tau_trace,
                alpha_error=self.cfg.pc_alpha_error,
            )
    
    def step(
        self,
        x: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        context_id: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Single training step with selected learning mode.
        
        Args:
            x: Input (batch, input_size)
            target: Target (batch, output_size) or None
            context_id: For dynamics mode: which task context
            
        Returns:
            y_pred: Prediction
            error: Error signal
            info: Dict with training info (eligibility norms, PC errors, etc.)
        """
        self.step_count += 1
        info = {}
        
        # Route to appropriate handler
        if self.cfg.mode == "hebbian":
            y_pred, error = self._step_hebbian(x, target)
        elif self.cfg.mode == "eprop":
            y_pred, error = self._step_eprop(x, target)
        elif self.cfg.mode == "force":
            y_pred, error = self._step_force(x, target)
        elif self.cfg.mode == "dynamics":
            y_pred, error = self._step_dynamics(x, target, context_id)
        elif self.cfg.mode == "pc":
            y_pred, error = self._step_pc(x, target)
        elif self.cfg.mode == "hybrid":
            y_pred, error, info = self._step_hybrid(x, target)
        else:
            raise ValueError(f"Unknown mode: {self.cfg.mode}")
        
        # Track error for adaptive scheduling
        if target is not None:
            error_norm = error.norm().item()
            self.error_history.append(error_norm)
            if len(self.error_history) > 100:
                self.error_history.pop(0)
        
        # Update adaptive alpha if enabled
        if self.cfg.use_adaptive_alpha and target is not None:
            self._update_adaptive_alpha(error)
        
        # Collect info
        info['mode'] = self.cfg.mode
        info['step'] = self.step_count
        info['alpha'] = self._current_alpha
        if target is not None:
            info['error_norm'] = error.norm().item()
        
        return y_pred, error, info
    
    def _step_hebbian(
        self,
        x: torch.Tensor,
        target: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Original Arthedain dual-timescale Hebbian step."""
        # Forward
        if hasattr(self.rsnn, 'forward_single'):
            spikes = self.rsnn.forward_single(x)
        else:
            spikes = self.rsnn(x)
        
        if spikes.dim() == 1:
            spikes = spikes.unsqueeze(0)
        
        y_pred = self.readout(spikes)
        
        # Error
        error = target - y_pred if target is not None else torch.zeros_like(y_pred)
        
        # Hebbian eligibility
        hebbian = self.trainers["hebbian"]
        if hasattr(self.rsnn, 'spike_list') and self.rsnn.spike_list:
            # Multi-layer: use last layer spikes
            post = self.rsnn.spike_list[-1]
        else:
            post = spikes
        
        # Take mean over batch for hebbian update (expects 1D)
        spikes_mean = spikes.mean(0) if spikes.dim() > 1 else spikes
        post_mean = post.mean(0) if post.dim() > 1 else post
        E = hebbian.update(spikes_mean, post_mean)
        
        # Apply updates
        with torch.no_grad():
            if hasattr(self.readout, 'W'):
                self.readout.W += self.cfg.lr_readout * torch.outer(error.mean(0), spikes.mean(0))
                if hasattr(self.readout, 'b'):
                    self.readout.b += self.cfg.lr_readout * error.mean(0)
            elif hasattr(self.readout, 'weight'):
                self.readout.weight += self.cfg.lr_readout * torch.outer(error.mean(0), spikes.mean(0))
                if self.readout.bias is not None:
                    self.readout.bias += self.cfg.lr_readout * error.mean(0)
            
            # Recurrent update
            if hasattr(self.rsnn, 'W_rec'):
                self.rsnn.W_rec += self.cfg.lr_recurrent * E
        
        return y_pred, error
    
    def _step_eprop(
        self,
        x: torch.Tensor,
        target: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """e-prop eligibility trace step."""
        return self.trainers["eprop"].step(x, target)
    
    def _step_force(
        self,
        x: torch.Tensor,
        target: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """FORCE/RLS online learning step."""
        return self.trainers["force"].step(x, target)
    
    def _step_dynamics(
        self,
        x: torch.Tensor,
        target: Optional[torch.Tensor],
        context_id: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Dynamics-based learning step."""
        y_pred = self.trainers["dynamics"].forward(x, context_id)
        error = target - y_pred if target is not None else torch.zeros_like(y_pred)
        return y_pred, error
    
    def _step_pc(
        self,
        x: torch.Tensor,
        target: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pure predictive coding step."""
        # Forward with spike collection
        if hasattr(self.rsnn, 'forward'):
            spike_list, y_pred = self.rsnn(x)
        else:
            spikes = self.rsnn(x)
            spike_list = [spikes]
            y_pred = self.readout(spikes)
        
        error = target - y_pred if target is not None else torch.zeros_like(y_pred)
        
        # PC updates
        if self.pc_stack is not None:
            pc_errors = self.pc_stack.step(spike_list, update=True)
        
        return y_pred, error
    
    def _shared_forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Shared forward pass with caching for efficiency.
        Run RSNN once, return (spikes, y_pred) for all methods to reuse.
        Uses no_grad context for inference efficiency.
        """
        with torch.no_grad():
            # Compute forward pass once
            if hasattr(self.rsnn, 'forward_single'):
                spikes = self.rsnn.forward_single(x)
            else:
                spikes = self.rsnn(x)

            if spikes.dim() == 1:
                spikes = spikes.unsqueeze(0)

            y_pred = self.readout(spikes)

        # Cache for reuse
        self._cached_spikes = spikes.detach()
        self._cached_y_pred = y_pred.detach()

        return spikes, y_pred
    
    def _compile_hebbian_update(self):
        """JIT compile hebbian update for speed."""
        # This is a placeholder - actual JIT would require refactoring
        # The update method will be optimized via in-place operations
        pass
    
    def _get_lr_with_warmup(self, base_lr: float) -> float:
        """Get learning rate with warmup schedule."""
        if self.step_count < self.cfg.lr_warmup_steps:
            # Linear warmup
            self._lr_warmup_factor = self.step_count / self.cfg.lr_warmup_steps
            return base_lr * self._lr_warmup_factor
        return base_lr
    
    def _step_hybrid(
        self,
        x: torch.Tensor,
        target: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Hybrid step combining multiple learning methods with ONE forward pass.

        Key optimizations:
        - Single RSNN forward pass cached for all methods
        - Pre-allocated delta_W_rec buffer to avoid repeated allocations
        - Accumulate weighted contributions then apply single in-place update
        - Skip expensive FORCE RLS inversion when error is below threshold
        """
        info = {}
        mix = self.cfg.hybrid_mix
        run_expensive = (self.step_count % self.cfg.update_frequency == 0)

        # --- SINGLE forward pass (no_grad for efficiency) ---
        spikes, y_pred = self._shared_forward(x)
        error = target - y_pred if target is not None else torch.zeros_like(y_pred)
        error_norm = error.norm().item()

        # Pre-compute mean over batch for efficiency
        spikes_mean = spikes.mean(0)
        error_mean = error.mean(0)

        # Pre-allocate delta_W_rec buffer for accumulating contributions
        if hasattr(self.rsnn, 'W_rec'):
            delta_W_rec = torch.zeros_like(self.rsnn.W_rec)
        else:
            delta_W_rec = None

        # Run available trainers with shared activations
        predictions = {}

        # 1. Hebbian update (lightweight, always runs)
        if "hebbian" in self.trainers:
            with torch.no_grad():
                hebbian = self.trainers["hebbian"]
                if hasattr(self.rsnn, 'spike_list') and self.rsnn.spike_list:
                    post = self.rsnn.spike_list[-1]
                    post_mean = post.mean(0) if post.dim() > 1 else post
                else:
                    post_mean = spikes_mean

                E = hebbian.update(spikes_mean, post_mean)

                # Accumulate weighted Hebbian contribution
                if delta_W_rec is not None:
                    delta_W_rec.add_(E, alpha=mix.get("hebbian", 0.3) * error_norm)

            predictions["hebbian"] = y_pred

        # 2. E-prop update (moderate cost, run per update_frequency)
        if "eprop" in self.trainers and run_expensive:
            with torch.no_grad():
                # Use cached forward result if available
                if hasattr(self.rsnn, 'lif') and hasattr(self.rsnn.lif, 'v'):
                    z = spikes
                    v = self.rsnn.lif.v
                    self.trainers["eprop"]._last_pre_spikes = x
                    self.trainers["eprop"]._last_post_v = v
                    self.trainers["eprop"]._last_z = z

                # Compute e-prop eligibility trace
                eprop = self.trainers["eprop"]
                pre_spikes = eprop._last_pre_spikes if eprop._last_pre_spikes is not None else x
                eprop.eprop_rec.update_eligibility(z, v, z)
                eprop.eprop_in.update_eligibility(pre_spikes, v, z)

                # Compute learning signal and weight update
                learning_signal = eprop._compute_learning_signal(error, z)
                dW_rec_ep = eprop.eprop_rec.compute_weight_update(learning_signal)

                # Accumulate weighted E-prop contribution
                if delta_W_rec is not None:
                    delta_W_rec.add_(dW_rec_ep, alpha=mix.get("eprop", 0.3))

            predictions["eprop"] = y_pred

        # 3. FORCE update (expensive O(n²) RLS, skip when error is negligible)
        if "force" in self.trainers and run_expensive and error_norm > self.cfg.skip_threshold:
            with torch.no_grad():
                force = self.trainers["force"]
                # Only update readout with RLS (recurrent handled by Hebbian/E-prop)
                if hasattr(force, 'rls_readout') and hasattr(self.readout, 'W'):
                    force.rls_readout.update(self.readout.W, spikes, error)

                # Optionally compute recurrent contribution if in rls_full mode
                if force.cfg.mode == "rls_full" and delta_W_rec is not None:
                    if force.prev_spikes is not None and hasattr(force, 'rls_rec'):
                        # RLS returns delta_W directly
                        dW_force = torch.zeros_like(delta_W_rec)
                        force.rls_rec.update(dW_force, force.prev_spikes, error)
                        delta_W_rec.add_(dW_force, alpha=mix.get("force", 0.4))

            predictions["force"] = y_pred
        elif "force" in self.trainers:
            # Skip expensive FORCE update - use current prediction
            predictions["force"] = y_pred
            info['force_skipped'] = True

        # --- Apply combined recurrent update with warmup scaling + grad clipping ---
        if delta_W_rec is not None:
            with torch.no_grad():
                lr_rec = self._get_lr_with_warmup(self.cfg.lr_recurrent)
                delta_W_rec.mul_(lr_rec)

                # Gradient clipping
                if self.cfg.grad_clip_norm > 0:
                    norm = delta_W_rec.norm()
                    if norm > self.cfg.grad_clip_norm:
                        delta_W_rec.mul_(self.cfg.grad_clip_norm / (norm + 1e-8))

                # In-place weight update - no new tensor allocation
                self.rsnn.W_rec.add_(delta_W_rec)

        # Readout update (shared across all methods)
        with torch.no_grad():
            lr_ro = self._get_lr_with_warmup(self.cfg.lr_readout)
            if hasattr(self.readout, 'W'):
                self.readout.W.add_(torch.outer(error_mean, spikes_mean), alpha=lr_ro)
                if hasattr(self.readout, 'b'):
                    self.readout.b.add_(error_mean, alpha=lr_ro)
            elif hasattr(self.readout, 'weight'):
                self.readout.weight.add_(torch.outer(error_mean, spikes_mean), alpha=lr_ro)
                if self.readout.bias is not None:
                    self.readout.bias.add_(error_mean, alpha=lr_ro)

        # Weighted combination (if multiple predictions)
        if len(predictions) > 1:
            y_pred = sum(mix.get(k, 0.0) * predictions[k] for k in predictions if k in mix)
            total_weight = sum(mix.get(k, 0.0) for k in predictions if k in mix)
            if total_weight > 0:
                y_pred = y_pred / total_weight

        # Recompute error after updates
        error = target - y_pred if target is not None else torch.zeros_like(y_pred)

        info['predictions'] = {k: v.detach() for k, v in predictions.items()}
        info['mix_weights'] = mix
        info['error_norm'] = error_norm
        info['run_expensive'] = run_expensive

        return y_pred, error, info
    
    def _update_adaptive_alpha(self, error: torch.Tensor):
        """Update alpha based on error RMS (drift detection)."""
        rms = error.norm().item()
        
        if rms > self.cfg.alpha_drift_threshold:
            # High error: rely more on local PC (lower alpha)
            self._current_alpha = max(0.0, self._current_alpha - 0.02)
        else:
            # Stable: drift back to base alpha
            self._current_alpha = min(
                self.cfg.pc_alpha_error,
                self._current_alpha + 0.005
            )
        
        # Update PC stack if exists
        if self.pc_stack is not None:
            for layer in self.pc_stack.layers:
                layer.cfg.alpha_error = self._current_alpha
    
    def fast_adapt(
        self,
        task_data: List[Tuple[torch.Tensor, torch.Tensor]],
        context_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Fast adaptation to a new task (dynamics mode).
        
        Only available when dynamics trainer is active.
        """
        if "dynamics" not in self.trainers:
            warnings.warn("Fast adaptation only available in dynamics mode")
            return {}
        
        stats = self.trainers["dynamics"].fast_adapt_to_task(task_data, context_id)
        return stats
    
    def set_context(self, context_id: int):
        """Set context for dynamics mode."""
        if "dynamics" in self.trainers:
            self.trainers["dynamics"].set_context(context_id)
    
    def reset(self):
        """Reset all trainers."""
        if "hebbian" in self.trainers:
            self.trainers["hebbian"].e_fast.zero_()
            self.trainers["hebbian"].e_slow.zero_()
        
        if "eprop" in self.trainers:
            self.trainers["eprop"].reset_eligibility()
        
        if "force" in self.trainers:
            self.trainers["force"].reset()
        
        if self.pc_stack is not None:
            self.pc_stack.reset_state()
        
        self.step_count = 0
        self.error_history = []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get trainer statistics."""
        stats = {
            'mode': self.cfg.mode,
            'step': self.step_count,
            'alpha': self._current_alpha,
        }
        
        if self.error_history:
            stats['error_mean'] = sum(self.error_history) / len(self.error_history)
            stats['error_last'] = self.error_history[-1]
        
        if "force" in self.trainers:
            stats['force'] = self.trainers["force"].get_stats()
        
        return stats


# ---------------------------------------------------------------------------
# Convenience factory functions
# ---------------------------------------------------------------------------

def make_unified_trainer(
    rsnn: nn.Module,
    readout: nn.Module,
    mode: str = "hybrid",
    hidden_sizes: Optional[List[int]] = None,
    lr_recurrent: float = 5e-5,
    use_pc: bool = True,
) -> UnifiedTrainer:
    """
    Factory for unified trainer with sensible defaults.
    
    Args:
        rsnn: Recurrent SNN
        readout: Readout layer
        mode: Learning mode ("hybrid", "hebbian", "eprop", "force", "dynamics")
        hidden_sizes: List of hidden layer sizes
        lr_recurrent: Recurrent learning rate
        use_pc: Whether to use predictive coding stack
    """
    cfg = UnifiedConfig(
        mode=mode,
        hidden_sizes=hidden_sizes,
        lr_recurrent=lr_recurrent,
        pc_use_stack=use_pc,
    )
    return UnifiedTrainer(rsnn, readout, cfg)
