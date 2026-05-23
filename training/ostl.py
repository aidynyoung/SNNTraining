"""
training/ostl.py
================
Online Spatio-Temporal Learning (OSTL) — Bohnstingl et al. (2022).

A provably better approximation of BPTT than e-prop with the same O(1) memory.

Reference
---------
Bohnstingl, T., Scherr, F., Pehle, C., Meier, K., & Maass, W. (2022).
Online spatio-temporal learning in deep neural networks.
Frontiers in Neuroscience, 16. https://doi.org/10.3389/fnins.2022.855482

Key difference from e-prop
---------------------------
e-prop maintains a *temporal* eligibility trace e_ij(t) ∈ ℝ for each
synapse (i→j) that captures how the synapse contributed to the post-neuron's
membrane potential over time.

OSTL maintains a *spatio-temporal* trace M_ij(t) ∈ ℝ that additionally
captures the influence of neuron i's activity on neuron j through *other*
neurons in the network (the indirect path). This gives a more accurate
approximation of the true gradient, particularly for sequences requiring
credit assignment over >50 ms.

Algorithm
---------
For each synapse i→j at time t:

    e_ij(t)  = decay · e_ij(t−1) + ψ_j(t) · z_i(t−1)    [local trace, same as e-prop]
    M_ij(t)  = ∑_k  W_jk(t) · M_ik(t−1) · ψ_j(t)        [spatio-temporal correction]

    grad_ij  = e_ij(t) + λ_M · M_ij(t)                   [combined gradient estimate]
    ΔW_ij    = −η · L(t) · grad_ij

The M tensor (hidden_size × hidden_size × hidden_size) is too large to store
directly. OSTL uses a rank-1 approximation:

    M_ij(t) ≈ m_i(t) · c_j(t)

where m_i is the "incoming eligibility" and c_j is the "outgoing credit",
each of shape (hidden_size,).

This keeps memory at O(N) instead of O(N³).

Usage
-----
    from training.ostl import OSTLLearner, OSTLConfig

    learner = OSTLLearner(OSTLConfig(hidden_size=128, output_size=2))
    dW = learner.step(u, spikes, output_error, W_rec)
    W_rec -= dW
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional


@dataclass
class OSTLConfig:
    hidden_size:     int   = 128
    output_size:     int   = 2
    tau_eligibility: float = 20.0    # ms
    learning_rate:   float = 5e-5
    lambda_M:        float = 0.1     # spatio-temporal correction weight
    surrogate_window: float = 0.3    # Neftci et al. 2019
    v_threshold:     float = 1.0
    feedback_scale:  float = 1.0
    device: Optional[str] = None


class OSTLLearner(nn.Module):
    """
    OSTL online learning rule with rank-1 spatio-temporal approximation.

    Attributes
    ----------
    B    : (output_size, hidden_size) — fixed random feedback matrix
    e    : (hidden_size, hidden_size) — local eligibility trace (as in e-prop)
    m    : (hidden_size,)             — incoming eligibility (rank-1 M approximation)
    c    : (hidden_size,)             — outgoing credit
    h_prev: (hidden_size,)            — previous spike history
    """

    def __init__(self, config: OSTLConfig) -> None:
        super().__init__()
        self.cfg = config
        self.device = torch.device(config.device or (
            'cuda' if torch.cuda.is_available() else 'cpu'))

        self.decay = math.exp(-1.0 / config.tau_eligibility)

        # Fixed random feedback (no weight transport)
        B = torch.randn(config.output_size, config.hidden_size, device=self.device)
        B *= config.feedback_scale / math.sqrt(config.hidden_size)
        self.register_buffer("B", B)

        # Eligibility trace (local, same as e-prop)
        self.register_buffer("e",
            torch.zeros(config.hidden_size, config.hidden_size, device=self.device))

        # Rank-1 spatio-temporal correction factors
        self.register_buffer("m",
            torch.zeros(config.hidden_size, device=self.device))
        self.register_buffer("c",
            torch.zeros(config.hidden_size, device=self.device))

        self.register_buffer("h_prev",
            torch.zeros(config.hidden_size, device=self.device))

    # ------------------------------------------------------------------
    # Surrogate derivative
    # ------------------------------------------------------------------

    def _psi(self, u: torch.Tensor) -> torch.Tensor:
        w = self.cfg.surrogate_window
        v = self.cfg.v_threshold
        return ((u >= v - w) & (u <= v + w)).float() / (2.0 * w)

    # ------------------------------------------------------------------
    # Core OSTL step
    # ------------------------------------------------------------------

    def step(
        self,
        u:            torch.Tensor,   # membrane potentials  (hidden_size,)
        spikes:       torch.Tensor,   # spike vector         (hidden_size,)
        output_error: torch.Tensor,   # readout error        (output_size,)
        W_rec:        torch.Tensor,   # current W_rec        (hidden_size, hidden_size)
    ) -> torch.Tensor:
        """
        One OSTL step. Returns ΔW of shape (hidden_size, hidden_size).

        Parameters
        ----------
        u            : membrane potential at current step
        spikes       : spike output at current step
        output_error : δ = y_pred − y_target
        W_rec        : current recurrent weight matrix (used for M correction)
        """
        psi = self._psi(u)                              # (hidden_size,)

        # ---- Local eligibility trace (same as e-prop) ----
        self.e.mul_(self.decay).add_(
            torch.outer(psi, self.h_prev)               # (hidden_size, hidden_size)
        )

        # ---- Rank-1 spatio-temporal correction ----
        # m[t] = decay·m[t-1] + psi (incoming signal)
        # c[t] = W_rec^T @ (psi * c[t-1])  (outgoing credit propagated)
        self.m.mul_(self.decay).add_(psi)
        self.c = W_rec.T @ (psi * self.c)               # (hidden_size,)
        self.c.mul_(self.decay)

        # Rank-1 M approximation: M_ij ≈ m_i * c_j
        M_approx = torch.outer(self.m, self.c)          # (hidden_size, hidden_size)

        # ---- Combined gradient estimate ----
        grad_est = self.e + self.cfg.lambda_M * M_approx

        # ---- Random feedback learning signal ----
        L = self.B.T @ output_error                     # (hidden_size,)

        # ---- Weight update ----
        dW = self.cfg.learning_rate * L.unsqueeze(1) * grad_est

        # Store for next step
        self.h_prev = spikes.detach().clone()

        return dW

    def apply_update(
        self,
        W_rec:        torch.Tensor,
        u:            torch.Tensor,
        spikes:       torch.Tensor,
        output_error: torch.Tensor,
    ) -> torch.Tensor:
        """Compute and apply the OSTL update to W_rec in-place."""
        dW = self.step(u, spikes, output_error, W_rec)
        W_rec.sub_(dW)
        return W_rec

    def reset(self) -> None:
        """Reset all traces (call between independent sequences)."""
        self.e.zero_()
        self.m.zero_()
        self.c.zero_()
        self.h_prev.zero_()


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = OSTLConfig(hidden_size=64, output_size=2, tau_eligibility=20.0)
    learner = OSTLLearner(cfg)
    W_rec = torch.randn(64, 64) * 0.1

    for t in range(200):
        u      = torch.randn(64)
        spikes = (u > 0.5).float()
        error  = torch.randn(2) * 0.1
        learner.apply_update(W_rec, u, spikes, error)

    print(f"OSTL smoke test OK  W_rec norm: {W_rec.norm():.4f}")
    print(f"  e norm:  {learner.e.norm():.4f}")
    print(f"  m norm:  {learner.m.norm():.4f}")
    print(f"  c norm:  {learner.c.norm():.4f}")
