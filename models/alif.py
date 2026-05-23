"""
models/alif.py — Adaptive Leaky Integrate-and-Fire neuron layer.

Implements the ALIF neuron from Bellec et al. (2020) "A solution to the
learning dilemma for recurrent networks of spiking neurons", Nature Comms.

Each neuron has an adaptation variable a(t) that raises the firing
threshold after each spike, creating intrinsic working memory within
the neuron (τ_a ≈ 25–250 ms depending on ρ).

This is the neuron model required for the full e-prop algorithm. Without
the adaptation variable the eligibility trace cannot propagate credit over
timescales longer than τ_eligibility (~20 ms).

Dynamics
--------
    a[t]   = ρ · a[t−1] + (1 − ρ) · z[t−1]     adaptation (normalised)
    A[t]   = v_th + β_a · a[t]                   adaptive threshold
    v[t]   = β · v[t−1] + I[t] − z[t−1] · v_th  membrane (reset-by-subtraction)
    z[t]   = 1 if v[t] ≥ A[t] else 0             spike
    ψ[t]   = γ · max(0, 1 − |v[t]−A[t]| / v_th) / v_th   pseudo-derivative

Parameters (Bellec 2020 Table S1)
----------------------------------
    rho    : adaptation decay ∈ [0.90, 0.99] (ρ=0.96 → τ_a≈25 ms at dt=1 ms)
    beta_a : adaptation strength ∈ [0.07, 0.2]
    gamma  : pseudo-derivative scaling ∈ [0.3, 0.5]
"""

import math
import torch
from dataclasses import dataclass
from typing import Optional, Tuple, Union


@dataclass
class ALIFConfig:
    size:        int   = 128
    tau:         float = 20.0    # membrane time constant (ms)
    v_th:        float = 1.0     # base spike threshold
    v_reset:     float = 0.0     # reset voltage (subtraction: v → v − v_th)
    refractory:  int   = 2       # absolute refractory period (steps)
    rho:         float = 0.96    # adaptation decay (τ_a = −dt/ln(ρ))
    beta_a:      float = 0.07    # adaptation strength (raises threshold)
    gamma:       float = 0.3     # pseudo-derivative scaling
    dt:          float = 1.0
    device:      Optional[str] = None
    # Threshold modulation for test-time distribution shift (Zhao et al. 2026)
    # Adjusts the BASE threshold (v_th_0) using an EMA of mean membrane potential.
    # Acts on top of ALIF's biological adaptation (which still operates normally).
    enable_threshold_adaptation: bool  = False
    threshold_adaptation_rate:   float = 0.01   # γ: how fast threshold tracks v̄
    threshold_momentum:          float = 0.99   # EMA decay for running mean of v


class ALIFLayer:
    """
    Adaptive LIF neuron layer compatible with DualHebbian and EPropLearner.

    Exposes the same step() interface as LIFLayer plus an optional
    return_pseudo_deriv flag required by ALIFEPropLearner.

    State tensors
    -------------
    v   : (size,) — membrane potential
    a   : (size,) — adaptation variable (≥ 0)
    refr: (size,) — refractory counter (counts down)
    """

    def __init__(
        self,
        size=None,
        config: Optional[ALIFConfig] = None,
        **kwargs,
    ) -> None:
        # Accept config as first positional arg (mirrors LIFLayer convention)
        if isinstance(size, ALIFConfig):
            config = size
            size = None
        if config is None:
            config = ALIFConfig(size=size or 128, **kwargs)
        self.cfg = config
        self.size = config.size
        self.device = torch.device(
            config.device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # Precompute membrane decay β = exp(−dt/τ)
        self.beta: float = math.exp(-config.dt / config.tau)

        # Threshold modulation state (Zhao et al. 2026)
        self._v_th_0: float = config.v_th           # nominal base threshold
        self._v_running_mean: Optional[torch.Tensor] = None

        self._reset_state()

    def _reset_state(self) -> None:
        d = self.device
        self.v    = torch.zeros(self.size, device=d)
        self.a    = torch.zeros(self.size, device=d)   # adaptation
        self.refr = torch.zeros(self.size, device=d)   # refractory counter
        self.spike_hist: list = []

    def reset(self, reset_threshold: bool = False) -> None:
        """Reset all state (call between independent sequences).

        Args:
            reset_threshold: If True, restore base threshold to nominal v_th_0
                             and clear the running mean.  Set False (default)
                             to let the adapted threshold persist across sequences
                             during continuous deployment.
        """
        self._reset_state()
        if reset_threshold:
            self._v_th_0 = self.cfg.v_th
            self._v_running_mean = None

    # ------------------------------------------------------------------
    # Core step
    # ------------------------------------------------------------------

    def step(
        self,
        input_current: torch.Tensor,
        return_pseudo_deriv: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        One ALIf timestep.

        Parameters
        ----------
        input_current       : (size,) — pre-computed synaptic input
        return_pseudo_deriv : if True, also return ψ(t) for ALIF e-prop

        Returns
        -------
        spikes              : (size,) binary
        pseudo_deriv        : (size,) ψ(t)   [only if return_pseudo_deriv=True]
        """
        input_current = input_current.to(self.device)
        cfg = self.cfg

        # Refractory countdown
        self.refr = (self.refr - 1).clamp(min=0)
        not_refr  = (self.refr == 0).float()

        # Adaptation update (independent of spikes at this step)
        # a[t] = ρ·a[t−1] + (1−ρ)·z[t−1]   (prev spike already in v reset)
        # We update a AFTER using prev_z — prev_z is reflected in the
        # refractory counter entering this step, so we approximate:
        # a[t] = ρ·a[t-1] + (1-ρ)·z[t-1] where z[t-1] = fired at last step
        # (tracked implicitly via a small approximation: update a before v)
        self.a = cfg.rho * self.a   # decay; spike contribution added below

        # Threshold modulation: update base threshold via EMA of membrane potential
        # (Zhao et al. 2026, arXiv:2505.05375) — acts on v_th_0, not on adaptation
        if cfg.enable_threshold_adaptation:
            v_mean = self.v.mean()
            if self._v_running_mean is None:
                self._v_running_mean = v_mean.clone()
            else:
                self._v_running_mean = (
                    cfg.threshold_momentum * self._v_running_mean
                    + (1.0 - cfg.threshold_momentum) * v_mean
                )
            self._v_th_0 = (
                cfg.v_th
                + cfg.threshold_adaptation_rate
                * (self._v_running_mean.item() - cfg.v_th)
            )

        # Adaptive threshold = shifted base + biological adaptation
        A = self._v_th_0 + cfg.beta_a * self.a   # (size,)

        # Membrane dynamics (reset-by-subtraction: v stays near threshold)
        # v[t] = β·v[t−1] + I[t]
        self.v = self.beta * self.v + input_current

        # Spike detection (gated by refractory)
        spikes = ((self.v >= A) & (not_refr > 0.5)).float()

        # Pseudo-derivative ψ(t) = γ/v_th · max(0, 1 − |v−A|/v_th)
        psi = (cfg.gamma / cfg.v_th) * (
            1.0 - (self.v - A).abs() / cfg.v_th
        ).clamp(min=0.0)

        # Reset: subtract v_th from voltage of spiking neurons (soft reset)
        self.v = self.v - spikes * cfg.v_th

        # Update adaptation for neurons that fired
        self.a = self.a + (1.0 - cfg.rho) * spikes   # a[t] += (1-ρ)·z[t]

        # Refractory counter
        self.refr = torch.where(spikes > 0.5,
                                torch.full_like(self.refr, cfg.refractory),
                                self.refr)

        # History (capped at 1000 steps)
        self.spike_hist.append(spikes.clone())
        if len(self.spike_hist) > 1000:
            self.spike_hist.pop(0)

        if return_pseudo_deriv:
            return spikes, psi
        return spikes

    def get_firing_rates(self, window: int = 100) -> torch.Tensor:
        if not self.spike_hist:
            return torch.zeros(self.size, device=self.device)
        w = min(window, len(self.spike_hist))
        return torch.stack(self.spike_hist[-w:]).mean(dim=0)

    def get_adaptive_threshold(self) -> torch.Tensor:
        """Return current adaptive threshold A = v_th + β_a·a."""
        return self.cfg.v_th + self.cfg.beta_a * self.a

    def __repr__(self) -> str:
        c = self.cfg
        return (f"ALIFLayer(size={c.size}, τ={c.tau}, ρ={c.rho}, "
                f"β_a={c.beta_a})")
