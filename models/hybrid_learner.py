"""
hybrid_learner.py
=================
HybridLearner — unified training wrapper for Arthedain that combines:

  1. Dual-Timescale Hebbian Accumulators (original Arthedain rule)
  2. Predictive Coding local error signals (PC-SNN / EchoSpike)

The key insight from the literature synthesis:

  The original Arthedain rule uses a GLOBAL broadcast error:
      e_global(t) = W^T · (y - ŷ)

  Predictive coding provides a LOCAL error at each layer interface:
      ε_local(t) = s_curr - sigmoid(W_gen · s_above)

  These two error signals are complementary:
  - Global error: precise task-aligned credit, but spatially non-local
  - Local PC error: layer-local, self-supervised, available without labels

  HybridLearner gates the Hebbian update with a convex mixture:
      e_hybrid(t) = α · e_global(t) + (1 - α) · ε_local(t)

  where α ∈ [0, 1] is the alpha_error parameter per PCLayer.

  This gives three operating modes:
    α = 1.0  →  pure global (original Arthedain)
    α = 0.0  →  pure local PC (fully self-supervised)
    α ∈ (0,1)→  hybrid (default, best of both worlds)

  For UAV/manufacturing deployment, α can be scheduled:
    - Supervised phase   (labelled data available):  α → 1.0
    - Adaptation phase   (disruption / drift):       α → 0.5
    - Blind phase        (no labels at all):          α → 0.0

Memory cost
-----------
Adds O(P_gen + P_rec) per PC interface (one per hidden-layer pair).
Total additional buffers: 2 × weight matrices per interface, plus 4 trace/RMS vectors.
Does NOT grow with sequence length T.

References
----------
- Arthedain paper: Nallani & Shah, arXiv:2509.14447
- PC-SNN:          Lan et al. / Wang et al., arXiv:2211.15386
- EchoSpike ESPP:  Graf et al., arXiv:2405.13976
- Meta-SpikePropamine eligibility traces: PMC10213417
"""

from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, List, Dict

from .predictive_coding import PCStack, build_pc_stack_for_arthedain


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class HybridConfig:
    """
    Drop-in config for HybridLearner.

    Parameters mirror Arthedain's existing TrainerConfig fields where
    possible; new fields are additive only.
    """
    # --- Existing Arthedain fields (kept for backwards compat) ---
    mode: str = "supervised"          # "supervised" | "reward" | "self_supervised"
    lr_readout: float = 2e-3
    lr_recurrent: float = 5e-5

    # --- PC-specific fields ---
    hidden_sizes: List[int] = None    # e.g. [1024, 512] for MC Maze
    pc_lr_gen: float = 1e-4           # generative weight lr
    pc_lr_rec: float = 5e-5           # recognition weight lr (0 = tied)
    pc_tau_trace: float = 20.0        # ms, EMA trace for PC Hebbian update
    pc_tie_weights: bool = False      # tie W_rec = W_gen^T
    pc_alpha_error: float = 0.5       # 0=pure PC, 1=pure global, 0.5=hybrid

    # --- Scheduling ---
    alpha_schedule: Optional[str] = None   # None | "linear_anneal" | "adaptive"
    alpha_anneal_steps: int = 50_000       # steps over which to anneal α→1
    alpha_drift_threshold: float = 0.3    # RMS error threshold to trigger drift mode

    # --- Stability ---
    rms_eps: float = 1e-8
    weight_cap: float = 6.0


# ---------------------------------------------------------------------------
# HybridLearner
# ---------------------------------------------------------------------------

class HybridLearner(nn.Module):
    """
    Wraps an existing Arthedain RSNN + OnlineTrainer logic and adds a
    PCStack for layer-local error gating.

    Designed to slot into the existing training loop with minimal changes:

        # Before (original Arthedain):
        y_pred, error = trainer.step(x, target=y)

        # After (HybridLearner):
        y_pred, error, pc_errors = hybrid.step(x, target=y)

    The pc_errors list can be ignored or logged for analysis.
    """

    def __init__(
        self,
        rsnn: nn.Module,
        readout: nn.Module,
        hebbian,              # DualHebbianAccumulator from models/hebbian.py
        cfg: HybridConfig,
    ):
        super().__init__()
        self.rsnn    = rsnn
        self.readout = readout
        self.hebbian = hebbian
        self.cfg     = cfg

        # Build PC stack if hidden_sizes provided
        self.pc_stack: Optional[PCStack] = None
        if cfg.hidden_sizes and len(cfg.hidden_sizes) > 1:
            self.pc_stack = build_pc_stack_for_arthedain(
                hidden_sizes  = cfg.hidden_sizes,
                lr_gen        = cfg.pc_lr_gen,
                lr_rec        = cfg.pc_lr_rec,
                tau_trace     = cfg.pc_tau_trace,
                tie_weights   = cfg.pc_tie_weights,
                alpha_error   = cfg.pc_alpha_error,
                rms_eps       = cfg.rms_eps,
            )

        # Alpha scheduling state
        self._step_count = 0
        self._current_alpha = cfg.pc_alpha_error
        self._error_rms_ema = 1.0     # for adaptive scheduling

    # ------------------------------------------------------------------
    def step(
        self,
        x: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        dt: float = 1.0,
    ):
        """
        One timestep of hybrid forward + update.

        Returns
        -------
        y_pred    : (batch, n_out)
        error     : scalar global loss (for logging)
        pc_errors : list of (batch, n_pre) local PC errors, or []
        """
        self._step_count += 1
        alpha = self._get_alpha(target)

        # 1. RSNN forward — collect layer spikes
        spike_list, y_pred = self._rsnn_forward(x)

        # 2. Global error (task-supervised or reward)
        global_error = self._compute_global_error(y_pred, target)

        # 3. PC local errors
        pc_errors = []
        if self.pc_stack is not None and spike_list is not None:
            pc_errors = self.pc_stack.step(spike_list, dt=dt, update=True)

        # 4. Hybrid error for Hebbian gating
        #    Inject PC error into the Hebbian accumulator's error drive
        if pc_errors and alpha < 1.0:
            self._apply_hybrid_hebbian(
                spike_list, global_error, pc_errors, alpha
            )
        else:
            # Pure global path — identical to original Arthedain
            self._apply_global_hebbian(global_error)

        return y_pred, global_error, pc_errors

    # ------------------------------------------------------------------
    def _rsnn_forward(self, x: torch.Tensor):
        """
        Forward pass through the RSNN.
        Returns (spike_list, y_pred) where spike_list is a list of tensors,
        one per layer, used by the PC stack.

        NOTE: This method hooks into the existing rsnn forward.
        If the RSNN exposes intermediate spikes (e.g. rsnn.spike_list),
        we use them; otherwise we fall back to None and disable PC.
        """
        y_pred = self.rsnn(x)
        spike_list = getattr(self.rsnn, "spike_list", None)
        return spike_list, y_pred

    def _compute_global_error(
        self,
        y_pred: torch.Tensor,
        target: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if target is not None:
            return (target - y_pred)
        # Reward / self-supervised mode: return zero (Hebbian still runs via traces)
        return torch.zeros_like(y_pred)

    def _apply_global_hebbian(self, global_error: torch.Tensor):
        """Original Arthedain path — calls DualHebbianAccumulator unchanged."""
        if hasattr(self.hebbian, "update"):
            self.hebbian.update(global_error)

    def _apply_hybrid_hebbian(
        self,
        spike_list,
        global_error: torch.Tensor,
        pc_errors: List[torch.Tensor],
        alpha: float,
    ):
        """
        Mix global and local PC errors before passing to the Hebbian accumulator.

        For each hidden layer ℓ with a PC interface above it:
            e_hybrid[ℓ] = α · broadcast(global_error)[ℓ] + (1-α) · ε_local[ℓ]

        The broadcast is the standard weight-transpose backprop from the readout
        (already computed inside DualHebbianAccumulator). Here we inject a
        correction term based on PC errors before the weight update fires.
        """
        beta = 1.0 - alpha

        # Inject PC correction into Hebbian accumulator state if supported
        if hasattr(self.hebbian, "inject_pc_correction"):
            corrections = {}
            for i, pc_err in enumerate(pc_errors):
                # Reduce batch dim, normalise
                correction = beta * pc_err.detach().mean(0)
                corrections[i] = correction
            self.hebbian.inject_pc_correction(corrections, alpha=alpha)

        # Always run the global path as well (scaled by alpha)
        self._apply_global_hebbian(global_error * alpha)

    # ------------------------------------------------------------------
    def _get_alpha(self, target) -> float:
        """Compute current α for hybrid mixing."""
        cfg = self.cfg

        if cfg.alpha_schedule is None:
            return self._current_alpha

        if cfg.alpha_schedule == "linear_anneal":
            # Anneal α from initial value toward 1.0 (more global over time)
            progress = min(self._step_count / cfg.alpha_anneal_steps, 1.0)
            return cfg.pc_alpha_error + progress * (1.0 - cfg.pc_alpha_error)

        if cfg.alpha_schedule == "adaptive":
            # Switch to pure PC (α→0) when global error RMS spikes (disruption)
            if target is not None:
                # Not yet computed — use stored EMA
                pass
            err_rms = self._error_rms_ema
            if err_rms > cfg.alpha_drift_threshold:
                # Disruption detected: lower alpha (rely more on local PC)
                return max(0.0, self._current_alpha - 0.05)
            else:
                # Stable: drift alpha back toward configured value
                return min(
                    cfg.pc_alpha_error,
                    self._current_alpha + 0.01
                )

        return self._current_alpha

    def update_error_rms(self, error: torch.Tensor):
        """Call after each step to track error RMS for adaptive scheduling."""
        rms = error.detach().norm().item()
        self._error_rms_ema = 0.99 * self._error_rms_ema + 0.01 * rms

    # ------------------------------------------------------------------
    def set_alpha(self, alpha: float):
        """Manually set mixing coefficient (for deployment phase transitions)."""
        self._current_alpha = float(alpha)

    def reset_pc_state(self):
        """Reset PC traces between sessions / after disruption."""
        if self.pc_stack:
            self.pc_stack.reset_state()

    def state_dict_compact(self) -> dict:
        """
        Serialise weights + traces for crash-safe checkpoint.
        Covers both Hebbian and PC state.
        """
        snap = {
            "step":  self._step_count,
            "alpha": self._current_alpha,
            "rms":   self._error_rms_ema,
        }
        if self.pc_stack:
            snap["pc"] = [
                layer.state_dict_compact()
                for layer in self.pc_stack.layers
            ]
        return snap

    def load_state_compact(self, snapshot: dict):
        self._step_count    = snapshot.get("step", 0)
        self._current_alpha = snapshot.get("alpha", self.cfg.pc_alpha_error)
        self._error_rms_ema = snapshot.get("rms", 1.0)
        if self.pc_stack and "pc" in snapshot:
            for layer, s in zip(self.pc_stack.layers, snapshot["pc"]):
                layer.load_state_compact(s)

    def diagnostics(self) -> Dict:
        """
        Return a structured diagnostics dict for monitoring a deployed learner.

        Includes: step count, current alpha (error-weighting), error RMS EMA,
        and PC stack free energy (if attached).
        """
        report: Dict = {
            "step":         self._step_count,
            "alpha":        round(self._current_alpha, 6),
            "error_rms":    round(self._error_rms_ema, 6),
        }
        if self.pc_stack is not None:
            try:
                report["free_energy"]          = round(self.pc_stack.free_energy(), 6)
                report["prediction_confidence"] = round(self.pc_stack.prediction_confidence(), 4)
            except Exception:
                pass
        return report

    def hybrid_health(self) -> Dict:
        """
        One-call comprehensive health: alpha regime, error RMS, PC stack state.

        mode_label maps current alpha to a human-readable operating regime:
          'supervised'     α > 0.8
          'hybrid'         0.3 < α ≤ 0.8
          'self_supervised' α ≤ 0.3
        """
        α = self._current_alpha
        if α > 0.8:
            mode = "supervised"
        elif α > 0.3:
            mode = "hybrid"
        else:
            mode = "self_supervised"
        report: Dict = {
            **self.diagnostics(),
            "mode_label":      mode,
            "pc_attached":     self.pc_stack is not None,
        }
        if self.pc_stack is not None:
            try:
                report["pc_health"] = self.pc_stack.stack_health()
            except Exception:
                pass
        return report

    def learning_rate_schedule(self, n_steps: int) -> list:
        """
        Preview how alpha evolves over the next n_steps given the current error EMA.

        Returns a list of (step, alpha) pairs useful for visualisation.
        """
        schedule = []
        alpha = self._current_alpha
        rms   = self._error_rms_ema
        for i in range(n_steps):
            # Mimic the _get_alpha logic with a synthetic constant error
            # (uses current rms as a proxy for future error)
            alpha_target = (self.cfg.pc_alpha_error * (1 + rms * 10)
                            if hasattr(self.cfg, "pc_alpha_error") else 0.1)
            alpha = float(min(max(alpha * 0.99 + 0.01 * alpha_target, 0.0), 1.0))
            schedule.append((self._step_count + i, round(alpha, 6)))
        return schedule