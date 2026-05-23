"""
update_rules_pc.py
==================
Additional update rules for Arthedain that leverage predictive coding errors.

These are designed to be added to training/update_rules.py (or imported
alongside it). Each rule is a drop-in swappable replacement / extension
of the existing delta rule in OnlineTrainer.

New rules
---------
1. PCHebbianRule        — Hebbian update gated by PC local error (α-mix)
2. ESPPRule             — EchoSpike Predictive Plasticity contrastive update
3. AdaptiveAlphaRule    — Wrapper that auto-schedules α based on error RMS

All rules follow the same interface as the existing update_rules.py:
    rule.update(pre, post, error, **kwargs) → ΔW

References
----------
- EchoSpike ESPP: Graf et al., arXiv:2405.13976
- PC-SNN:         Lan/Wang et al., arXiv:2211.15386
- e-prop / Meta-SpikePropamine: PMC10213417
"""

from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# 1. PCHebbianRule
# ---------------------------------------------------------------------------

@dataclass
class PCHebbianConfig:
    lr: float = 1e-4
    alpha: float = 0.5          # 0=pure PC error, 1=pure global error
    rms_eps: float = 1e-8
    rms_alpha: float = 0.99
    weight_cap: float = 6.0


class PCHebbianRule(nn.Module):
    """
    Three-factor Hebbian update using a hybrid error signal.

    ΔW = η · RMS( α·e_global + (1-α)·e_local ) ⊗ pre

    Compatible with DualHebbianAccumulator: the output ΔW is passed
    directly into the accumulator's update() method.
    """

    def __init__(self, cfg: PCHebbianConfig):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("_rms", torch.tensor(1.0))

    def update(
        self,
        pre:       torch.Tensor,   # (batch, n_pre)
        post_sens: torch.Tensor,   # (batch, n_post)  — d_LIF surrogate
        e_global:  torch.Tensor,   # (batch, n_post)
        e_local:   Optional[torch.Tensor] = None,  # (batch, n_post), PC error
    ) -> torch.Tensor:
        """Returns ΔW (n_post × n_pre), ready to add to weight matrix."""
        cfg = self.cfg

        if e_local is not None:
            e_hybrid = cfg.alpha * e_global + (1.0 - cfg.alpha) * e_local
        else:
            e_hybrid = e_global

        # RMS normalise
        self._rms.mul_(cfg.rms_alpha).add_(
            (1 - cfg.rms_alpha) * e_hybrid.detach().pow(2).mean()
        )
        e_norm = e_hybrid / (self._rms.sqrt() + cfg.rms_eps)

        # Three-factor: (error ⊙ post_sensitivity) ⊗ pre
        driver = (e_norm * post_sens).detach()         # (batch, n_post)
        pre_d  = pre.detach()                          # (batch, n_pre)
        dW = cfg.lr * (driver.T @ pre_d) / max(pre.shape[0], 1)  # (n_post, n_pre)

        return dW


# ---------------------------------------------------------------------------
# 2. ESPPRule — EchoSpike Predictive Plasticity contrastive update
# ---------------------------------------------------------------------------

@dataclass
class ESPPConfig:
    lr: float = 5e-5
    tau_pos: float = 20.0     # ms — trace for "positive" (same-class) phase
    tau_neg: float = 5.0      # ms — trace for "negative" (contrastive) phase
    rms_eps: float = 1e-8
    rms_alpha: float = 0.99


class ESPPRule(nn.Module):
    """
    EchoSpike Predictive Plasticity contrastive update.

    Implements the two-phase ESPP contrastive rule:
      Phase 1 (positive):  network sees input x, records spike correlations
      Phase 2 (negative):  network sees perturbed / predicted x̃, records correlations

      ΔW = η · (C_pos - C_neg)

    where C_pos, C_neg are EMA spike correlation matrices.

    In the Arthedain streaming context (no explicit phases), we approximate:
      C_pos ← current spike-spike correlation (real input)
      C_neg ← predicted spike correlation (from PC generative model)

    This is computed at each timestep without storing full history.

    Reference: Graf et al. 2024, arXiv:2405.13976, Section 3.
    """

    def __init__(self, shape: tuple, cfg: ESPPConfig):
        """
        shape: (n_post, n_pre) — weight matrix shape
        """
        super().__init__()
        self.cfg = cfg
        n_post, n_pre = shape

        self.register_buffer("C_pos", torch.zeros(n_post, n_pre))
        self.register_buffer("C_neg", torch.zeros(n_post, n_pre))
        self.register_buffer("_rms",  torch.tensor(1.0))

        lp = 1.0 - 1.0 / cfg.tau_pos
        ln = 1.0 - 1.0 / cfg.tau_neg
        self.register_buffer("_lam_pos", torch.tensor(lp))
        self.register_buffer("_lam_neg", torch.tensor(ln))

    def update(
        self,
        s_real:      torch.Tensor,   # (batch, n_post) — actual post spikes
        s_pred:      torch.Tensor,   # (batch, n_post) — PC-generated prediction
        pre:         torch.Tensor,   # (batch, n_pre)
    ) -> torch.Tensor:
        """Returns ΔW (n_post × n_pre)."""
        pre_d     = pre.detach()
        s_real_d  = s_real.detach()
        s_pred_d  = s_pred.detach()

        # Update EMA correlation matrices
        C_pos_new = (s_real_d.T @ pre_d) / max(s_real.shape[0], 1)
        C_neg_new = (s_pred_d.T @ pre_d) / max(s_pred.shape[0], 1)

        self.C_pos.mul_(self._lam_pos).add_((1 - self._lam_pos) * C_pos_new)
        self.C_neg.mul_(self._lam_neg).add_((1 - self._lam_neg) * C_neg_new)

        contrastive = self.C_pos - self.C_neg

        # RMS normalise
        self._rms.mul_(0.99).add_(0.01 * contrastive.pow(2).mean())
        dW = self.cfg.lr * contrastive / (self._rms.sqrt() + self.cfg.rms_eps)

        return dW

    def reset(self):
        self.C_pos.zero_()
        self.C_neg.zero_()


# ---------------------------------------------------------------------------
# 3. AdaptiveAlphaRule — wrapper with automatic α scheduling
# ---------------------------------------------------------------------------

class AdaptiveAlphaRule:
    """
    Wraps any rule that accepts an `alpha` parameter and automatically
    adjusts it based on error RMS (for manufacturing/UAV drift detection).

    Logic:
      - Compute 1-step error RMS
      - Update EMA of error RMS
      - If EMA > drift_threshold: shift α toward 0 (trust local PC more)
      - If EMA < stable_threshold: shift α toward pc_alpha_base (mixed)

    This operationalises the deployment recommendation from the previous
    analysis: α is scheduled dynamically rather than fixed.
    """

    def __init__(
        self,
        rule,
        pc_alpha_base: float = 0.5,
        drift_threshold: float = 0.3,
        stable_threshold: float = 0.1,
        rms_alpha: float = 0.99,
    ):
        self.rule             = rule
        self.pc_alpha_base    = pc_alpha_base
        self.drift_threshold  = drift_threshold
        self.stable_threshold = stable_threshold
        self.rms_alpha        = rms_alpha
        self._err_rms_ema     = 0.1
        self._current_alpha   = pc_alpha_base

    def update(self, error: torch.Tensor, **kwargs):
        # Update error RMS EMA
        rms = error.detach().norm().item()
        self._err_rms_ema = (
            self.rms_alpha * self._err_rms_ema
            + (1 - self.rms_alpha) * rms
        )

        # Schedule α
        if self._err_rms_ema > self.drift_threshold:
            # Disruption: rely more on local PC
            self._current_alpha = max(0.0, self._current_alpha - 0.02)
        elif self._err_rms_ema < self.stable_threshold:
            # Stable: return to base mixture
            self._current_alpha = min(
                self.pc_alpha_base,
                self._current_alpha + 0.005,
            )

        # Delegate to wrapped rule with updated alpha
        if hasattr(self.rule, "cfg"):
            self.rule.cfg.alpha = self._current_alpha
        return self.rule.update(**kwargs)

    @property
    def current_alpha(self) -> float:
        return self._current_alpha

    @property
    def error_rms(self) -> float:
        return self._err_rms_ema