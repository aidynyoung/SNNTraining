"""
eprop.py
========
Eligibility Propagation (e-prop) for online learning in recurrent SNNs.

Implements eligibility trace-based credit assignment without BPTT,
following the Meta-SpikePropamine approach and e-prop variants.

Key insight: Instead of backpropagating through time, e-prop uses
local eligibility traces that combine:
  - Immediate eligibility: ∂z_j(t)/∂W_ij  (local gradient at spike time)
  - Filtered eligibility trace: decayed accumulation over time

This gives constant-memory training with O(P) complexity.

References
----------
- Meta-SpikePropamine: https://pmc.ncbi.nlm.nih.gov/articles/PMC10213417/
- e-prop: Bellec et al. 2020, https://arxiv.org/abs/1901.09049
- Symmetric e-prop: random feedback alignment variant
"""

from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple, Callable


@dataclass
class EPropConfig:
    """Configuration for e-prop eligibility trace learning."""
    tau_eligibility: float = 20.0      # ms - eligibility trace decay
    tau_filter: float = 50.0            # ms - filter for learning signal
    learning_rate: float = 5e-5         # base learning rate
    # Neftci et al. 2019: window=0.3 optimal for temporal tasks
    surrogate_derivative: str = "piecewise_linear"  # "piecewise_linear" | "exponential" | "arctangent"
    surrogate_window: float = 0.3       # narrowed from 0.5 (Neftci et al. 2019)
    alpha_symmetric: float = 0.0         # 0=e-prop, 1=symmetric e-prop, 0.5=mix
    use_random_feedback: bool = False    # use random B instead of W^T
    feedback_scale: float = 1.0          # scale for random feedback weights

    # ALIF support (Bellec et al. 2020)
    use_alif: bool = False              # enable ALIF adaptation trace in eligibility
    rho: float = 0.96                   # adaptation decay (must match ALIFLayer)
    beta_a: float = 0.07                # adaptation strength (must match ALIFLayer)

    # Adaptive eligibility parameters
    adaptive_tau: bool = False           # Enable adaptive tau based on spike rate
    tau_min: float = 10.0               # Minimum eligibility time constant
    tau_max: float = 50.0               # Maximum eligibility time constant
    target_spike_rate: float = 0.1      # Target spike rate for adaptation

    # Gradient coherence (Hao et al. 2026, arXiv:2410.07547)
    # Aligns surrogate gradient with true gradient via forward Euler approx
    use_gradient_coherence: bool = False
    coherence_lambda: float = 0.1       # Regularisation strength
    coherence_tau: float = 5.0          # Timescale for true gradient estimate


@torch.jit.script

def piecewise_linear_surrogate_jit(v: torch.Tensor, v_th: float = 1.0, 
                                    window: float = 0.5) -> torch.Tensor:
    """JIT-compiled piecewise linear surrogate derivative."""
    return ((v >= v_th - window) & (v <= v_th + window)).float() / (2 * window)


def piecewise_linear_surrogate(v: torch.Tensor, v_th: float = 1.0, 
                                window: float = 0.5) -> torch.Tensor:
    """
    Piecewise linear surrogate derivative for spike function.
    
    Returns derivative of spike w.r.t. membrane potential.
    Non-zero only within window around threshold.
    """
    return piecewise_linear_surrogate_jit(v, v_th, window)


@torch.jit.script
def exponential_surrogate_jit(v: torch.Tensor, v_th: float = 1.0,
                              beta: float = 10.0) -> torch.Tensor:
    """JIT-compiled exponential surrogate derivative."""
    return beta * torch.exp(-beta * torch.abs(v - v_th))


def exponential_surrogate(v: torch.Tensor, v_th: float = 1.0,
                          beta: float = 10.0) -> torch.Tensor:
    """
    Exponential surrogate derivative.
    d/dv sigma(v - v_th) ≈ beta * exp(-beta * |v - v_th|)
    """
    return exponential_surrogate_jit(v, v_th, beta)


class EPropAccumulator(nn.Module):
    """
    Eligibility trace accumulator for e-prop learning.
    
    Maintains eligibility traces e_ij(t) for each synapse,
    updated locally without backpropagation through time.
    
    Eligibility trace dynamics:
        e_ij(t) = decay * e_ij(t-1) + dz_j/dW_ij
        
    where dz_j/dW_ij is the immediate eligibility from surrogate gradient.
    """
    
    def __init__(
        self,
        shape: Tuple[int, int],  # (n_post, n_pre)
        cfg: EPropConfig,
        v_th: float = 1.0,
    ):
        super().__init__()
        self.cfg = cfg
        self.shape = shape
        self.n_post, self.n_pre = shape
        self.v_th = v_th
        
        # Eligibility traces: E(t) with shape (n_post, n_pre)
        self.register_buffer("eligibility", torch.zeros(shape))
        
        # Decay coefficient for eligibility trace
        self.register_buffer("decay_elig", 
                           torch.tensor(1.0 - 1.0 / cfg.tau_eligibility))
        
        # Filter coefficient for learning signal
        self.register_buffer("decay_filter",
                           torch.tensor(1.0 - 1.0 / cfg.tau_filter))
        
        # Random feedback weights for e-prop variant (if enabled)
        if cfg.use_random_feedback:
            self.register_buffer("B", torch.randn(shape) * cfg.feedback_scale)
        else:
            self.B = None
            
        # Surrogate derivative function (window narrowed to 0.3 per Neftci 2019)
        w = cfg.surrogate_window
        if cfg.surrogate_derivative == "exponential":
            self.surrogate_fn = lambda v: exponential_surrogate(v, v_th)
        elif cfg.surrogate_derivative == "arctangent":
            # Fang et al. 2021 (ICCV) — smoothest gradient landscape
            self.surrogate_fn = lambda v: (
                1.0 / (math.pi * w * (1.0 + ((v - v_th) / w).pow(2)))
            )
        else:
            self.surrogate_fn = lambda v: piecewise_linear_surrogate(v, v_th, w)

        # ALIF adaptation eligibility trace (Bellec et al. 2020, Eq. 9)
        # e_a_ij[t] = ρ·e_a_ij[t-1] + ψ_j[t]·z_j[t-1]
        if cfg.use_alif:
            self.register_buffer("e_adapt", torch.zeros(shape))

        # Gradient coherence (Hao et al. 2026)
        if cfg.use_gradient_coherence:
            # Running estimate of true gradient via forward Euler
            self.register_buffer("true_grad_estimate", torch.zeros(shape))
            # Decay for true gradient estimate
            self.register_buffer("decay_coherence",
                               torch.tensor(1.0 - 1.0 / cfg.coherence_tau))
    
    def compute_immediate_eligibility(

        self,
        pre_spikes: torch.Tensor,    # (batch, n_pre)
        post_v: torch.Tensor,          # (batch, n_post) - membrane potential
        z_post: torch.Tensor,          # (batch, n_post) - spike output
    ) -> torch.Tensor:
        """
        Compute immediate eligibility ∂z_j/∂W_ij using surrogate gradient.
        
        For LIF: dz_j/dW_ij ≈ h'(v_j - v_th) * z_i(t) * 1/tau_m
        
        Returns tensor of shape (batch, n_post, n_pre)
        """
        # Surrogate derivative of spike w.r.t. membrane potential
        h_prime = self.surrogate_fn(post_v)  # (batch, n_post)
        
        # Immediate eligibility: outer product of h'(v) and pre-synaptic spikes
        # Shape: (batch, n_post, 1) * (batch, 1, n_pre) -> (batch, n_post, n_pre)
        immediate = h_prime.unsqueeze(2) * pre_spikes.unsqueeze(1)
        
        return immediate
    
    def update_eligibility(
        self,
        pre_spikes: torch.Tensor,    # (batch, n_pre)
        post_v: torch.Tensor,        # (batch, n_post)
        z_post: torch.Tensor,        # (batch, n_post)
    ) -> torch.Tensor:
        """
        Update eligibility traces with new timestep.
        Uses in-place operations for memory efficiency.
        With adaptive tau: adjusts decay based on observed spike rate.
        
        Returns the current eligibility traces after update.
        """
        # Compute immediate eligibility
        immediate = self.compute_immediate_eligibility(pre_spikes, post_v, z_post)
        
        # Update eligibility trace with in-place operations: E(t) = decay * E(t-1) + immediate
        # Mean over batch
        immediate_mean = immediate.mean(0)  # (n_post, n_pre)
        
        # Adaptive tau: adjust decay based on spike rate
        if self.cfg.adaptive_tau:
            spike_rate = z_post.float().mean().item()
            # Higher spike rate → shorter tau (faster decay) to prevent saturation
            # Lower spike rate → longer tau (slower decay) to maintain traces
            target_rate = self.cfg.target_spike_rate
            if spike_rate > target_rate * 1.5:
                # Too many spikes - reduce tau (increase decay)
                tau = max(self.cfg.tau_min, self.cfg.tau_eligibility * 0.9)
            elif spike_rate < target_rate * 0.5:
                # Too few spikes - increase tau (decrease decay)
                tau = min(self.cfg.tau_max, self.cfg.tau_eligibility * 1.1)
            else:
                tau = self.cfg.tau_eligibility
            
            # Update decay factor (as scalar, not buffer for JIT compatibility)
            decay_elig = 1.0 - 1.0 / tau
        else:
            decay_elig = self.decay_elig.item()
        
        self.eligibility.mul_(decay_elig).add_(immediate_mean)

        # Gradient coherence (Hao et al. 2026): align surrogate with true gradient
        if self.cfg.use_gradient_coherence:
            # Forward Euler approximation of true gradient:
            #   g_true(t) ≈ (z(t) - z(t-1)) / dt  — how spikes actually change
            #   g_surr(t) = ψ(t) · pre(t)          — surrogate gradient
            # Coherence loss: L_coherence = λ · ||g_surr - g_true||²
            # We apply this as a correction to the eligibility trace
            with torch.no_grad():
                # True gradient estimate: how pre-synaptic activity affects
                # post-synaptic spike probability (forward Euler)
                # g_true ≈ (z_post - β·z_post_prev) / (1 - β)  — de-meaned
                z_prev = getattr(self, '_z_prev', None)
                if z_prev is not None:
                    # Spike probability change (forward difference)
                    z_flat = z_post.float().mean(0)  # (n_post,)
                    z_prev_flat = z_prev.float().mean(0)
                    dz = z_flat - z_prev_flat
                    
                    # True gradient: outer product of dz and pre-spikes
                    pre_flat = pre_spikes.float().mean(0)  # (n_pre,)
                    g_true = dz.unsqueeze(1) * pre_flat.unsqueeze(0)  # (n_post, n_pre)
                    
                    # Surrogate gradient (immediate eligibility)
                    g_surr = immediate_mean
                    
                    # Coherence correction: pull surrogate toward true gradient
                    # This is the gradient of L_coherence w.r.t. eligibility
                    coherence_correction = self.cfg.coherence_lambda * (g_surr - g_true)
                    
                    # Apply correction to eligibility trace
                    self.eligibility.sub_(coherence_correction * (1 - decay_elig))
                
                # Store current z for next step
                self._z_prev = z_post.detach().clone()
        
        return self.eligibility
    
    def compute_weight_update(

        self,
        learning_signal: torch.Tensor,  # (batch, n_post) - L_j(t)
    ) -> torch.Tensor:
        """
        Compute weight update using eligibility traces and learning signal.
        
        e-prop update rule:
            ΔW_ij = η * L_j(t) * e_ij(t)
            
        where L_j(t) is the learning signal (error or reward).
        """
        # Learning signal mean over batch
        L = learning_signal.mean(0)  # (n_post,)
        
        # Three-factor rule: L_j * e_ij
        # (n_post, 1) * (n_post, n_pre) -> (n_post, n_pre)
        delta_W = self.cfg.learning_rate * L.unsqueeze(1) * self.eligibility
        
        return delta_W
    
    def reset(self):
        """Reset eligibility traces (between episodes)."""
        self.eligibility.zero_()


class EPropTrainer:
    """
    Trainer implementing e-prop for online learning in recurrent SNNs.
    
    Combines eligibility traces with local learning signals to enable
    constant-memory training without BPTT.
    
    Usage:
        trainer = EPropTrainer(rsnn, readout, eprop_config)
        for x, target in stream:
            y_pred, error = trainer.step(x, target)
    """
    
    def __init__(
        self,
        rsnn: nn.Module,
        readout: nn.Module,
        cfg: EPropConfig,
        lr_readout: float = 2e-3,
    ):
        self.rsnn = rsnn
        self.readout = readout
        self.cfg = cfg
        self.lr_readout = lr_readout
        
        # Create eligibility accumulators for recurrent and input weights
        hidden_size = getattr(rsnn, 'hidden_size', rsnn.W_rec.shape[0])
        input_size = getattr(rsnn, 'input_size', rsnn.W_in.shape[1])
        
        self.eprop_rec = EPropAccumulator(
            shape=(hidden_size, hidden_size),
            cfg=cfg,
        )
        self.eprop_in = EPropAccumulator(
            shape=(hidden_size, input_size),
            cfg=cfg,
        )
        
        # Store for surrogate gradient computation
        self._last_pre_spikes = None
        self._last_post_v = None
        self._last_z = None
    
    def step(
        self,
        x: torch.Tensor,
        target: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        One training step using e-prop.
        
        Args:
            x: Input tensor (batch, input_size)
            target: Target tensor (batch, output_size) or None
            
        Returns:
            y_pred: Prediction (batch, output_size)
            error: Error signal (batch, output_size)
        """
        # Forward pass through RSNN
        # Need to capture membrane potential for surrogate gradient
        z, v = self._forward_rsnn_with_potential(x)
        
        # Readout
        y_pred = self.readout(z)
        
        # Compute error
        if target is not None:
            error = target - y_pred
        else:
            error = torch.zeros_like(y_pred)
        
        # Compute learning signal L_j(t) for hidden neurons
        # L_j = Σ_k B_jk * error_k where B is feedback weights
        learning_signal = self._compute_learning_signal(error, z)
        
        # Update eligibility traces
        pre_spikes = self._last_pre_spikes if self._last_pre_spikes is not None else x
        self.eprop_rec.update_eligibility(z, v, z)
        self.eprop_in.update_eligibility(pre_spikes, v, z)
        
        # Compute weight updates
        dW_rec = self.eprop_rec.compute_weight_update(learning_signal)
        dW_in = self.eprop_in.compute_weight_update(learning_signal)
        
        # Apply updates
        with torch.no_grad():
            if hasattr(self.rsnn, 'W_rec'):
                self.rsnn.W_rec += dW_rec
            if hasattr(self.rsnn, 'W_in'):
                self.rsnn.W_in += dW_in
            
            # Update readout with delta rule
            if hasattr(self.readout, 'W'):
                self.readout.W += self.lr_readout * torch.outer(error.mean(0), z.mean(0))
                if hasattr(self.readout, 'b'):
                    self.readout.b += self.lr_readout * error.mean(0)
        
        # Track error for convergence monitoring
        if not hasattr(self, '_error_history'):
            self._error_history = []
        self._error_history.append(float(error.abs().mean().item()))
        if len(self._error_history) > 2000:
            self._error_history = self._error_history[-1000:]

        return y_pred, error

    def _forward_rsnn_with_potential(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass capturing both spikes and membrane potential.
        
        Returns:
            z: Spike output
            v: Membrane potential (for surrogate gradient)
        """
        # Store input for input weight eligibility
        self._last_pre_spikes = x.clone() if hasattr(x, 'clone') else x
        
        # If RSNN exposes membrane potential, use it
        if hasattr(self.rsnn, 'lif') and hasattr(self.rsnn.lif, 'v'):
            # Standard forward
            z = self.rsnn(x)
            v = self.rsnn.lif.v.clone()
        else:
            # Fallback: approximate with spike history
            z = self.rsnn(x)
            # Approximate membrane as decayed spike accumulation
            v = z.clone()  # Simplified - real impl would track actual v
        
        self._last_z = z
        self._last_post_v = v
        return z, v
    
    def _compute_learning_signal(
        self,
        error: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute learning signal L_j for each hidden neuron.
        
        L_j = Σ_k B_jk * error_k
        
        where B is either W_out^T (symmetric) or random feedback weights.
        """
        if self.cfg.use_random_feedback and self.eprop_rec.B is not None:
            # Random feedback alignment
            # L = B @ error
            if error.dim() == 1:
                L = self.eprop_rec.B[:, :error.shape[0]] @ error
            else:
                # Batch: (batch, n_post) = (batch, n_error) @ B^T
                L = error @ self.eprop_rec.B[:, :error.shape[1]].T
        else:
            # Symmetric feedback: use readout weights
            if hasattr(self.readout, 'W'):
                W_out = self.readout.W  # (output, hidden)
                if error.dim() == 1:
                    L = W_out.T @ error  # (hidden,)
                else:
                    # Batch: (batch, hidden) = (batch, output) @ W_out
                    L = error @ W_out
            else:
                # Fallback: use error directly broadcast
                L = torch.zeros(z.shape[0], z.shape[1] if z.dim() > 1 else z.shape[0])
        
        return L
    
    def reset_eligibility(self):
        """Reset all eligibility traces."""
        self.eprop_rec.reset()
        self.eprop_in.reset()

    def convergence_report(self) -> Dict:
        """
        Return convergence diagnostics for the current training run.

        Monitors:
          - Recent error magnitude (is learning happening?)
          - Error trend (improving / stable / diverging?)
          - Step count

        Useful for: automated early stopping, hyperparameter tuning,
        deployment readiness checks.

        Returns:
            Dict with 'steps', 'recent_error', 'trend', 'converged'.
        """
        if not hasattr(self, '_error_history'):
            self._error_history: list = []

        recent_n  = min(50, len(self._error_history))
        older_n   = min(100, len(self._error_history))
        if recent_n == 0:
            return {"steps": 0, "recent_error": 0.0, "trend": "unknown", "converged": False}

        recent_err = sum(self._error_history[-recent_n:]) / recent_n
        older_err  = (sum(self._error_history[-older_n:-recent_n]) / max(older_n - recent_n, 1)
                      if older_n > recent_n else recent_err)

        if recent_err < older_err * 0.95:
            trend = "improving"
        elif recent_err > older_err * 1.05:
            trend = "diverging"
        else:
            trend = "stable"

        return {
            "steps":        len(self._error_history),
            "recent_error": round(recent_err, 6),
            "trend":        trend,
            "converged":    trend == "stable" and recent_err < 0.05,
        }


# ---------------------------------------------------------------------------
# Convenience factory functions
# ---------------------------------------------------------------------------

def make_eprop_trainer(
    rsnn: nn.Module,
    readout: nn.Module,
    tau_eligibility: float = 20.0,
    tau_filter: float = 50.0,
    learning_rate: float = 5e-5,
    lr_readout: float = 2e-3,
    use_symmetric_feedback: bool = True,
) -> EPropTrainer:
    """
    Factory function to create an e-prop trainer with sensible defaults.
    
    Args:
        rsnn: Recurrent SNN model
        readout: Readout layer
        tau_eligibility: Eligibility trace time constant
        tau_filter: Filter time constant for learning signal
        learning_rate: Learning rate for recurrent weights
        lr_readout: Learning rate for readout weights
        use_symmetric_feedback: Use W^T feedback (True) or random (False)
    """
    cfg = EPropConfig(
        tau_eligibility=tau_eligibility,
        tau_filter=tau_filter,
        learning_rate=learning_rate,
        surrogate_derivative="piecewise_linear",
        alpha_symmetric=1.0 if use_symmetric_feedback else 0.0,
        use_random_feedback=not use_symmetric_feedback,
    )
    return EPropTrainer(rsnn, readout, cfg, lr_readout)
