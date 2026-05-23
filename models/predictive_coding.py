"""
predictive_coding.py
====================
Predictive Coding layer for Arthedain SNNs.

Implements a layer-local predictive coding unit compatible with the existing
dual-timescale Hebbian accumulator. Each PCLayer maintains:

  - A top-down generative weight matrix  W_gen  (layer ℓ+1 → ℓ)
  - A bottom-up recognition weight matrix W_rec  (layer ℓ → ℓ+1)  [shared W^T or independent]
  - Explicit error neurons  e⁺(t), e⁻(t)  (push-pull signed encoding)
  - An EMA spike-trace for Hebbian plasticity on the generative path

Derivation sketch (follows PC-SNN / EchoSpike):
  μ(t)   = σ( W_gen · s_above(t) )    # top-down prediction of current layer
  ε(t)   = s_curr(t) - μ(t)           # prediction error (signed)
  e⁺(t)  = ReLU( ε(t))                # positive error neurons
  e⁻(t)  = ReLU(-ε(t))                # negative error neurons

  ΔW_gen += η · (e⁺ - e⁻) ⊗ s_above  # local generative update
  ΔW_rec += η · s_above ⊗ (e⁺ - e⁻)  # optional recognition update

  The signed error ε(t) is also returned to the caller so the
  HybridLearner can gate the Hebbian trace with it instead of (or in
  addition to) the global broadcast error from the readout layer.

Memory: O(P_gen + P_rec) — constant in T.
No backward graph is constructed.

References
----------
- PC-SNN: Lan et al. 2022 / Wang et al. 2025, arXiv:2211.15386
- EchoSpike (ESPP): Graf et al. 2024, arXiv:2405.13976
- SNN-PC: Frontiers Comp. Neurosci. 2024, doi:10.3389/fncom.2024.1338280
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class PCConfig:
    """Configuration for a single PCLayer."""
    n_pre: int                     # dimensionality of current layer (ℓ)
    n_post: int                    # dimensionality of layer above (ℓ+1)
    tau_trace: float = 20.0        # ms — EMA decay for spike trace used in PC update
    lr_gen: float = 1e-4           # learning rate for W_gen
    lr_rec: float = 5e-5           # learning rate for W_rec (0 = frozen / tied)
    tie_weights: bool = False      # if True, W_rec = W_gen^T (weight symmetry)
    alpha_error: float = 0.5       # mixture: 0 = pure PC error, 1 = pure global error
    rms_eps: float = 1e-8          # epsilon for RMS normalisation
    rms_alpha: float = 0.99        # EMA coefficient for running RMS


class PCLayer(nn.Module):
    """
    One predictive coding layer.

    Forward inputs
    --------------
    s_curr  : (batch, n_pre)   — spike output of this layer at time t
    s_above : (batch, n_post)  — spike output of layer above at time t
    update  : bool             — whether to apply weight update this timestep

    Forward outputs
    ---------------
    error   : (batch, n_pre)   — signed prediction error ε(t)
    e_plus  : (batch, n_pre)   — positive error neurons
    e_minus : (batch, n_pre)   — negative error neurons
    """

    def __init__(self, cfg: PCConfig):
        super().__init__()
        self.cfg = cfg

        # Generative (top-down) weights: map layer ℓ+1 → ℓ
        self.W_gen = nn.Parameter(
            torch.randn(cfg.n_pre, cfg.n_post) * 0.01,
            requires_grad=False,          # updated by local rule, not autograd
        )

        if not cfg.tie_weights:
            # Independent recognition (bottom-up) weights: ℓ → ℓ+1
            self.W_rec = nn.Parameter(
                torch.randn(cfg.n_post, cfg.n_pre) * 0.01,
                requires_grad=False,
            )
        # else W_rec = W_gen.T on the fly

        # EMA spike traces for Hebbian update
        self.register_buffer("trace_curr",  torch.zeros(cfg.n_pre))
        self.register_buffer("trace_above", torch.zeros(cfg.n_post))

        # Running RMS for normalisation
        self.register_buffer("rms_gen", torch.ones(cfg.n_pre))
        self.register_buffer("rms_rec", torch.ones(cfg.n_post))

        # Decay coefficient (per-timestep)
        self._lambda = None          # set lazily from dt

    # ------------------------------------------------------------------
    def _update_traces(self, s_curr: torch.Tensor, s_above: torch.Tensor):
        """EMA spike trace — approximates NMDA-like synaptic integration."""
        if self._lambda is None:
            self._lambda = 1.0 - 1.0 / self.cfg.tau_trace   # dt assumed =1
        lam = self._lambda
        # Detach to avoid graph construction
        self.trace_curr  = lam * self.trace_curr  + (1 - lam) * s_curr.detach().mean(0)
        self.trace_above = lam * self.trace_above + (1 - lam) * s_above.detach().mean(0)

    def _rms_norm(self, v: torch.Tensor, ema: torch.Tensor, key: str) -> torch.Tensor:
        """Normalise v using a running EMA of its squared magnitude."""
        ema.mul_(self.cfg.rms_alpha).add_(
            (1 - self.cfg.rms_alpha) * v.detach().pow(2).mean(0)
        )
        return v / (ema.sqrt() + self.cfg.rms_eps)

    # ------------------------------------------------------------------
    def forward(
        self,
        s_curr:  torch.Tensor,
        s_above: torch.Tensor,
        dt: float = 1.0,
        update: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        self._lambda = 1.0 - dt / self.cfg.tau_trace

        # 1. Top-down prediction: μ = sigmoid( W_gen · s_above )
        #    Using sigmoid so μ ∈ (0,1) — same range as LIF spike outputs
        mu = torch.sigmoid(s_above @ self.W_gen.T)   # (batch, n_pre)

        # 2. Prediction error
        error  = s_curr - mu                          # (batch, n_pre)
        e_plus  = torch.relu( error)
        e_minus = torch.relu(-error)

        if update:
            self._update_traces(s_curr, s_above)
            self._apply_pc_update(e_plus, e_minus)

        return error, e_plus, e_minus

    # ------------------------------------------------------------------
    def _apply_pc_update(
        self,
        e_plus:  torch.Tensor,
        e_minus: torch.Tensor,
    ):
        """
        Local Hebbian update on generative and (optionally) recognition weights.

        ΔW_gen ∝  (e⁺ - e⁻)  ⊗  trace_above
        ΔW_rec ∝  trace_above ⊗  (e⁺ - e⁻)   [if not tied]
        """
        signed_err = (e_plus - e_minus).detach().mean(0)   # (n_pre,)

        # Normalise
        signed_err_norm = self._rms_norm(
            signed_err.unsqueeze(0), self.rms_gen, "gen"
        ).squeeze(0)

        # ΔW_gen: outer product of signed error (n_pre) × trace_above (n_post)
        dW_gen = self.cfg.lr_gen * torch.outer(signed_err_norm, self.trace_above)
        self.W_gen.data.add_(dW_gen)

        if not self.cfg.tie_weights:
            trace_err_norm = self._rms_norm(
                self.trace_above.unsqueeze(0), self.rms_rec, "rec"
            ).squeeze(0)
            dW_rec = self.cfg.lr_rec * torch.outer(trace_err_norm, signed_err_norm)
            self.W_rec.data.add_(dW_rec)

        # Weight cap (matches Arthedain per-neuron projection, Section III-D)
        _cap_weights(self.W_gen, cap=6.0)
        if not self.cfg.tie_weights:
            _cap_weights(self.W_rec, cap=6.0)

    # ------------------------------------------------------------------
    def get_recognition_weights(self) -> torch.Tensor:
        """Return W_rec, honouring the tie_weights flag."""
        if self.cfg.tie_weights:
            return self.W_gen.T
        return self.W_rec

    def reset_state(self):
        """Zero traces between episodes / on disruption."""
        self.trace_curr.zero_()
        self.trace_above.zero_()

    def state_dict_compact(self) -> dict:
        """Minimal serialisable snapshot (weights + traces, no grad info)."""
        out = {"W_gen": self.W_gen.data.clone()}
        if not self.cfg.tie_weights:
            out["W_rec"] = self.W_rec.data.clone()
        out["trace_curr"]  = self.trace_curr.clone()
        out["trace_above"] = self.trace_above.clone()
        out["rms_gen"] = self.rms_gen.clone()
        out["rms_rec"] = self.rms_rec.clone()
        return out

    def load_state_compact(self, snapshot: dict):
        self.W_gen.data.copy_(snapshot["W_gen"])
        if not self.cfg.tie_weights and "W_rec" in snapshot:
            self.W_rec.data.copy_(snapshot["W_rec"])
        self.trace_curr.copy_(snapshot["trace_curr"])
        self.trace_above.copy_(snapshot["trace_above"])
        self.rms_gen.copy_(snapshot["rms_gen"])
        self.rms_rec.copy_(snapshot["rms_rec"])

    def layer_health(self) -> dict:
        """
        Diagnostic snapshot: weight norms, trace activity, saturation.
        W_gen saturation > 0.1 → weights hitting the cap frequently (lr too high).
        """
        w_norms = self.W_gen.data.norm(dim=1)
        saturated_frac = float((w_norms > 5.5).float().mean().item())
        return {
            "W_gen_mean_norm":    round(float(w_norms.mean().item()), 4),
            "W_gen_max_norm":     round(float(w_norms.max().item()), 4),
            "W_gen_saturated":    round(saturated_frac, 4),
            "trace_curr_mean":    round(float(self.trace_curr.mean().item()), 4),
            "trace_above_mean":   round(float(self.trace_above.mean().item()), 4),
            "rms_gen_mean":       round(float(self.rms_gen.mean().item()), 4),
            "alpha_error":        self.cfg.alpha_error,
        }


# ---------------------------------------------------------------------------
# Stack wrapper — attaches a PCLayer to every hidden-layer interface
# ---------------------------------------------------------------------------

class PCStack(nn.Module):
    """
    Attaches PCLayer instances between every consecutive pair of hidden layers
    in an existing Arthedain RSNN.

    Usage
    -----
        pc_stack = PCStack(hidden_sizes=[1024, 512], cfg_overrides={})
        # In OnlineTrainer.step(), after the forward pass:
        errors = pc_stack.step(spike_list, update=True)
        # errors[i] is ε at interface i (between layers i and i+1)
    """

    def __init__(
        self,
        hidden_sizes: list,
        cfg_overrides: Optional[dict] = None,
    ):
        super().__init__()
        cfg_overrides = cfg_overrides or {}
        self.layers = nn.ModuleList()
        for i in range(len(hidden_sizes) - 1):
            cfg = PCConfig(
                n_pre  = hidden_sizes[i],
                n_post = hidden_sizes[i + 1],
                **cfg_overrides,
            )
            self.layers.append(PCLayer(cfg))

    def step(
        self,
        spike_list: list,        # [s_layer0, s_layer1, ..., s_layerN]
        dt: float = 1.0,
        update: bool = True,
    ) -> list:
        """
        Run all PC interfaces for one timestep.

        spike_list must have at least len(self.layers)+1 elements.
        Returns list of signed errors, one per interface.
        """
        errors = []
        for i, pc in enumerate(self.layers):
            err, _, _ = pc(
                s_curr  = spike_list[i],
                s_above = spike_list[i + 1],
                dt      = dt,
                update  = update,
            )
            errors.append(err)
        return errors

    def reset_state(self):
        for layer in self.layers:
            layer.reset_state()

    def free_energy(self) -> float:
        """
        Estimate total free energy across the PC stack.

        Free energy = Σ_layers (prediction_error² / noise_variance)

        In predictive processing (Friston 2010), free energy is the objective
        the brain minimises during perception.  For Arthedain, this serves as:
          - Learning signal: high F → update weights more aggressively
          - Anomaly indicator: F >> baseline → unexpected observation
          - Convergence metric: decreasing F → the model is learning

        Returns:
            Scalar free energy estimate ∈ [0, ∞)
        """
        total_F = 0.0
        for layer in self.layers:
            # Prediction error: difference between current and predicted state
            if hasattr(layer, '_pe_buf') and layer._pe_buf:
                pe_sq = sum(float((pe**2).mean().item()) for pe in layer._pe_buf)
                total_F += pe_sq / max(len(layer._pe_buf), 1)
        return total_F

    def stack_health(self) -> dict:
        """Per-layer diagnostics + aggregate free energy."""
        report = {
            "n_layers":     len(self.layers),
            "free_energy":  round(self.free_energy(), 6),
        }
        for i, layer in enumerate(self.layers):
            report[f"layer_{i}"] = layer.layer_health()
        return report

    def prediction_confidence(self) -> float:
        """
        Confidence based on recent prediction errors.

        Confidence ∈ [0, 1]: 1 = perfectly confident (zero error),
        0 = completely uncertain.

        Returns:
            Confidence estimate from the bottom PC layer.
        """
        if not self.layers:
            return 0.0
        layer = self.layers[0]
        if hasattr(layer, '_pe_buf') and layer._pe_buf:
            recent_errors = [float((pe**2).mean().sqrt().item()) for pe in layer._pe_buf[-10:]]
            mean_rms = sum(recent_errors) / max(len(recent_errors), 1)
            # Map RMS error to confidence: confidence = exp(-rms_error)
            import math as _math
            return _math.exp(-mean_rms)
        return 0.5   # uninformed prior


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _cap_weights(W: nn.Parameter, cap: float = 6.0):
    """
    Per-neuron L2 weight projection (bitshift-implementable on hardware).
    Matches Arthedain Section III-D weight cap.
    """
    norms = W.data.norm(dim=1, keepdim=True).clamp(min=1e-8)
    mask  = (norms > cap).float()
    W.data.mul_(1 - mask + mask * cap / norms)


def build_pc_stack_for_arthedain(hidden_sizes: list, **kwargs) -> PCStack:
    """
    Convenience constructor.  Example:

        pc = build_pc_stack_for_arthedain(
            hidden_sizes=[1024, 512],
            lr_gen=1e-4,
            lr_rec=5e-5,
            alpha_error=0.5,
        )
    """
    return PCStack(hidden_sizes=hidden_sizes, cfg_overrides=kwargs)