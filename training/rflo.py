"""
rflo.py
=======
Random Feedback Local Online (RFLO) learning rule.

Murray (2019) "Local online learning in recurrent networks with random
feedback." eLife 8:e43299.

Key idea: replace the exact weight-transpose feedback (used in e-prop and
BPTT) with a fixed random matrix B. The learning signal L_j(t) is a random
projection of the output error rather than W^T * error. This eliminates
weight transport — each synapse only needs local pre/post information plus a
broadcast scalar — while retaining competitive performance.

Algorithm per timestep t
------------------------
1. Surrogate derivative:  phi_j = f'(u_j(t))     # at postsynaptic neuron j
2. Eligibility trace:     e_ij += -e_ij/tau + phi_j * h_i(t-1)
3. Random learning signal: L_j = sum_k B_kj * delta_k(t)
4. Weight update:         dW_ij = eta * L_j * e_ij

Compared with e-prop
---------------------
e-prop uses   L_j = sum_k W_kj * delta_k    (exact, requires weight transport)
RFLO uses     L_j = sum_k B_kj * delta_k    (approximate, fully local)

References
----------
- Murray (2019): https://elifesciences.org/articles/43299
- Roth et al. (2019) Kernel RNNs: https://arxiv.org/abs/1906.02027
- Bellec et al. (2020) e-prop: https://www.nature.com/articles/s41467-020-17236-y
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class RFLOConfig:
    """Configuration for RFLO learning rule."""
    hidden_size: int = 128
    output_size: int = 2
    tau_eligibility: float = 20.0       # ms — eligibility trace decay
    learning_rate: float = 5e-5         # weight update step size
    feedback_scale: float = 1.0         # scale applied to random B at init
    surrogate_window: float = 0.5       # half-width of piecewise-linear surrogate
    v_threshold: float = 1.0            # LIF spike threshold (matches LIFLayer)
    dt: float = 1.0                     # simulation timestep
    device: Optional[str] = None


class RFLOLearner(nn.Module):
    """
    RFLO online learner for a single recurrent layer.

    Attributes
    ----------
    B : (output_size, hidden_size)
        Fixed random feedback matrix drawn once at construction.
    e : (hidden_size, hidden_size)
        Eligibility trace matrix: e[j, i] connects pre i → post j.
    decay : float
        Per-step decay factor for the eligibility trace.
    """

    def __init__(self, config: RFLOConfig) -> None:
        super().__init__()
        self.cfg = config
        self.device = torch.device(config.device or (
            'cuda' if torch.cuda.is_available() else 'cpu'))

        self.decay: float = math.exp(-config.dt / config.tau_eligibility)

        # Fixed random feedback matrix (never updated)
        B_raw = torch.randn(config.output_size, config.hidden_size,
                            device=self.device)
        B_raw *= config.feedback_scale / math.sqrt(config.hidden_size)
        self.register_buffer("B", B_raw)

        # Eligibility trace [n_post × n_pre] — matches W_rec shape
        self.register_buffer(
            "e", torch.zeros(config.hidden_size, config.hidden_size,
                             device=self.device))

        # Previous hidden state (pre-synaptic activity)
        self.register_buffer(
            "h_prev", torch.zeros(config.hidden_size, device=self.device))

    # ------------------------------------------------------------------
    # Surrogate derivative
    # ------------------------------------------------------------------

    def _surrogate(self, u: torch.Tensor) -> torch.Tensor:
        """Piecewise-linear surrogate derivative of the Heaviside spike function."""
        w = self.cfg.surrogate_window
        v_th = self.cfg.v_threshold
        return ((u >= v_th - w) & (u <= v_th + w)).float() / (2.0 * w)

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------

    def step(
        self,
        u: torch.Tensor,           # membrane potentials   (hidden_size,)
        spikes: torch.Tensor,      # spike output          (hidden_size,)
        output_error: torch.Tensor # readout error signal  (output_size,)
    ) -> torch.Tensor:
        """
        One RFLO timestep: update eligibility trace and return ΔW_rec.

        Parameters
        ----------
        u : membrane potential vector at current step, shape (hidden_size,)
        spikes : spike vector at current step (not used directly — kept for
                 API symmetry with e-prop), shape (hidden_size,)
        output_error : δ = y_pred − y_target at current step, (output_size,)

        Returns
        -------
        dW : weight update matrix (hidden_size, hidden_size) — same shape as
             W_rec, ready to be added with a minus sign for gradient descent
             or plus sign for Hebbian-style rules.
        """
        phi = self._surrogate(u)                        # (hidden_size,)

        # Eligibility trace update (vectorised outer product)
        # e_ij = decay * e_ij + phi_j * h_i(t-1)
        self.e.mul_(self.decay).add_(
            torch.outer(phi, self.h_prev)               # (hidden_size, hidden_size)
        )

        # Random learning signal: L_j = sum_k B_kj * delta_k
        # B: (output_size, hidden_size)  →  B^T @ error: (hidden_size,)
        L = self.B.T @ output_error                     # (hidden_size,)

        # Weight update: dW_ij = eta * L_j * e_ij
        dW = self.cfg.learning_rate * L.unsqueeze(1) * self.e   # (hidden_size, hidden_size)

        # Store current spikes as next step's pre-synaptic history
        self.h_prev = spikes.detach().clone()

        return dW

    def apply_update(self, W_rec: torch.Tensor, output_error: torch.Tensor,
                     u: torch.Tensor, spikes: torch.Tensor) -> torch.Tensor:
        """Convenience wrapper: compute dW and apply it to W_rec in-place."""
        dW = self.step(u, spikes, output_error)
        W_rec.sub_(dW)   # gradient descent convention
        return W_rec

    def reset(self) -> None:
        """Reset eligibility trace and hidden state (call between episodes)."""
        self.e.zero_()
        self.h_prev.zero_()


# ---------------------------------------------------------------------------
# Comparison baseline: exact feedback (e-prop style, for ablation)
# ---------------------------------------------------------------------------

class ExactFeedbackLearner(RFLOLearner):
    """
    Same as RFLOLearner but uses W_rec^T as the feedback matrix (exact).

    Pass the current W_rec to step() via the extra w_rec kwarg; the
    learning signal becomes L = W_rec^T @ error (weight transport).
    Only useful as a strict upper-bound baseline in ablation studies.
    """

    def step(  # type: ignore[override]
        self,
        u: torch.Tensor,
        spikes: torch.Tensor,
        output_error: torch.Tensor,
        w_out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        w_out : (output_size, hidden_size) — exact output weight matrix.
                L = w_out^T @ output_error  (weight transport).
                Falls back to random B when None.
        """
        phi = self._surrogate(u)
        self.e.mul_(self.decay).add_(torch.outer(phi, self.h_prev))

        if w_out is not None:
            L = w_out.T @ output_error    # (hidden_size,)
        else:
            L = self.B.T @ output_error

        dW = self.cfg.learning_rate * L.unsqueeze(1) * self.e
        self.h_prev = spikes.detach().clone()
        return dW


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = RFLOConfig(hidden_size=64, output_size=2, tau_eligibility=20.0,
                     learning_rate=1e-4)
    learner = RFLOLearner(cfg)
    W_rec = torch.randn(64, 64) * 0.1

    for t in range(100):
        u = torch.randn(64)
        spikes = (u > 0.5).float()
        error = torch.randn(2) * 0.1
        dW = learner.step(u, spikes, error)

    print(f"RFLO smoke test OK — dW norm: {dW.norm():.4f}")
    print(f"Eligibility trace norm: {learner.e.norm():.4f}")
