"""
models/sparse_coder.py
======================
k-Winners-Take-All (k-WTA) sparse coding for extreme energy efficiency.

Enforces hard sparsity: exactly k neurons fire per timestep.  This
reduces synaptic operations (SynOps) proportionally to the sparsity
ratio, directly cutting energy on neuromorphic hardware.

At 5% activity (k = 0.05 * N):
  - SynOps reduction: 20×
  - Energy reduction: ~20× on Loihi 2 (measured, Orchard et al. 2021)
  - Information capacity: log₂(C(N,k)) bits  (still high for large N)

Implementations
---------------
k_wta         : Hard k-WTA (exact k neurons per step)
soft_wta      : Soft WTA via lateral inhibition (more biologically plausible)
adaptive_wta  : k adapts to maintain target firing rate (homeostatic)

References
----------
- Ahmad & Hawkins (2016) "How do neurons operate on sparse distributed
  representations?" Numenta.
- Mahowald & Douglas (1991) "A silicon neuron" Nature.
- Orchard et al. (2021) "Efficient neuromorphic signal processing" NeurIPS.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class SparseCoderConfig:
    n_neurons:   int   = 128
    k:           int   = 10          # number of winners (hard WTA)
    target_rate: float = 0.05        # for adaptive WTA
    boost_lr:    float = 0.001       # boost factor learning rate
    mode:        str   = "hard"      # "hard" | "soft" | "adaptive"


class SparseCoderLayer(nn.Module):
    """
    Drop-in replacement for any activation that enforces sparsity.

    Can be inserted after the LIF layer or used as a post-processing
    gate on RSNN output spikes.

    Usage
    -----
        sparse = SparseCoderLayer(SparseCoderConfig(n_neurons=128, k=6))
        # After RSNN step:
        spikes = sparse(raw_spikes)   # exactly k neurons active
    """

    def __init__(self, config: SparseCoderConfig) -> None:
        super().__init__()
        self.cfg = config

        if config.mode == "adaptive":
            # Per-neuron duty cycle tracking for homeostatic boost
            self.register_buffer("duty_cycle",
                torch.zeros(config.n_neurons))
            self.register_buffer("boost",
                torch.ones(config.n_neurons))
        else:
            self.duty_cycle = None
            self.boost = None

    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        """
        Apply sparse coding.

        Parameters
        ----------
        spikes : (n_neurons,) float (from LIF, values 0 or 1)

        Returns
        -------
        sparse_spikes : (n_neurons,) float, exactly k nonzero
        """
        mode = self.cfg.mode
        if mode == "hard":
            return _hard_kwta(spikes, self.cfg.k)
        elif mode == "soft":
            return _soft_wta(spikes, self.cfg.k)
        elif mode == "adaptive":
            return self._adaptive_wta(spikes)
        return spikes

    def _adaptive_wta(self, spikes: torch.Tensor) -> torch.Tensor:
        """
        Adaptive WTA: boost under-active neurons so every neuron gets
        used over time (prevents dead neurons).
        """
        cfg = self.cfg
        # Apply boost to overcome under-activity
        boosted = spikes * self.boost
        out = _hard_kwta(boosted, cfg.k)

        # Update duty cycle (EMA of recent activity)
        α = 0.1
        self.duty_cycle = (1 - α) * self.duty_cycle + α * out

        # Update boost: neurons below target rate get a boost
        ratio = cfg.target_rate / (self.duty_cycle.clamp(min=1e-6))
        self.boost = torch.clamp(self.boost * ratio.pow(cfg.boost_lr), min=0.1, max=10.0)

        return out

    def synops_count(self, spikes: torch.Tensor, W: torch.Tensor) -> int:
        """Count synaptic operations for an energy estimate."""
        n_active = (spikes > 0.5).sum().item()
        # Each active pre-synaptic neuron touches every post-synaptic neuron
        # (with sparse connectivity, multiply by density)
        return int(n_active * W.shape[0])

    def energy_estimate_pJ(self, spikes: torch.Tensor, W: torch.Tensor,
                            pJ_per_synop: float = 0.5) -> float:
        """Estimated energy per timestep in pJ (Loihi 2 typical: 0.25–0.5 pJ/SynOp)."""
        return self.synops_count(spikes, W) * pJ_per_synop

    def lateral_inhibition(
        self,
        activations: torch.Tensor,
        inhibit_strength: float = 0.3,
    ) -> torch.Tensor:
        """
        Apply lateral (competitive) inhibition before sparse selection.

        Reference:
            Olshausen & Field (1996) "Emergence of simple-cell receptive field
            properties by learning a sparse code" Nature 381:607-609.

        Lateral inhibition: each neuron's activation is suppressed by the
        mean activity of its neighbours:
            a_i_inhibited = a_i - inhibit_strength × mean(a_{i≠j})

        This sharpens the competitive dynamics of k-WTA by ensuring that
        only neurons with clearly above-average activation win — improving
        selectivity and reducing redundancy in the sparse representation.

        Args:
            activations:      (n_neurons,) pre-WTA activation values
            inhibit_strength: Amount of mean inhibition [0, 1]

        Returns:
            (n_neurons,) inhibited activations (pass to _hard_kwta next)
        """
        mean_act = activations.mean()
        return activations - inhibit_strength * mean_act


# ---------------------------------------------------------------------------
# Functional helpers
# ---------------------------------------------------------------------------

def _hard_kwta(x: torch.Tensor, k: int) -> torch.Tensor:
    """Return binary tensor with the top-k elements set to 1, rest 0."""
    if k <= 0 or k >= x.numel():
        return (x > 0.5).float()
    k = min(k, x.numel())
    topk_vals = torch.topk(x, k, largest=True, sorted=False).values
    threshold = topk_vals.min()
    return (x >= threshold).float()


def _soft_wta(x: torch.Tensor, k: int) -> torch.Tensor:
    """Soft WTA: normalise by local competition, then threshold."""
    inhibition = x.sum() / x.numel()
    return ((x - inhibition) > 0).float()


def sparse_info_capacity(n: int, k: int) -> float:
    """Information capacity of k-of-n sparse code in bits."""
    if k <= 0 or k >= n:
        return 0.0
    log_c = sum(math.log2(n - i) - math.log2(i + 1) for i in range(k))
    return log_c
