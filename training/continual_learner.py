"""
training/continual_learner.py
==============================
Continual learning for SNNTraining SNNs — learn new tasks without
catastrophic forgetting.

Critical for field deployment: a drone payload that can recognize a new
threat signature without forgetting the original ones.

Implements two complementary approaches:

  1. Elastic Weight Consolidation (EWC) — Kirkpatrick et al. (2017)
     Adds a Fisher-information-weighted penalty to prevent important
     weights from moving too far from their task-A values.

  2. Synaptic Intelligence (SI) — Zenke et al. (2017)
     Accumulates per-synapse path-integral importance during learning,
     protecting important synapses online (no explicit Fisher computation).

Both are O(N) in memory, compatible with online learning, and require
no replay buffer or stored task data.

References
----------
- Kirkpatrick, J. et al. (2017). Overcoming catastrophic forgetting in
  neural networks. PNAS, 114(13), 3521–3526.
- Zenke, F., Poole, B., & Ganguli, S. (2017). Continual learning through
  synaptic intelligence. ICML.
- Huszár, F. (2018). Note on the quadratic penalties in EWC. arXiv.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch


@dataclass
class ContinualConfig:
    method:       str   = "ewc"       # "ewc" | "si" | "both"
    ewc_lambda:   float = 100.0       # EWC penalty strength
    si_xi:        float = 0.1         # SI damping constant
    si_epsilon:   float = 0.1         # SI stability constant
    fisher_n:     int   = 200         # samples for Fisher estimation (EWC)
    n_tasks:      int   = 10          # max tasks to remember


class EWCRegulariser:
    """
    Elastic Weight Consolidation.

    After each task, computes the Fisher information matrix (diagonal
    approximation) and adds an L2 penalty toward task parameters,
    weighted by Fisher importance.

    Penalty = λ/2 * Σ_i F_i * (θ_i - θ*_i)²

    This directly translates to: important weights for old tasks move
    only when new task gradients are strong enough to overcome the penalty.
    """

    def __init__(self, config: ContinualConfig) -> None:
        self.cfg = config
        self.task_params: List[Dict[str, torch.Tensor]] = []   # θ* per task
        self.fisher_diag: List[Dict[str, torch.Tensor]] = []   # F_i per task

    def compute_fisher(
        self,
        rsnn,
        readout,
        data_stream,            # iterable of (x, y) pairs for the just-completed task
        n_samples: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Estimate diagonal Fisher information via squared gradients.

        F_i = E[(∂ log p(y|x,θ) / ∂θ_i)²]
        """
        n = n_samples or self.cfg.fisher_n
        fisher: Dict[str, torch.Tensor] = {}

        # We use the squared output gradient as a proxy for Fisher
        for name, p in [("W_rec", rsnn.W_rec), ("W_out", readout.W)]:
            fisher[name] = torch.zeros_like(p.data)

        count = 0
        for x, y in data_stream:
            if count >= n:
                break
            # Forward pass
            spikes = rsnn.forward(x)
            pred   = readout.forward(spikes)
            loss   = (pred - y.to(pred.device)).pow(2).mean()

            # Manual gradient (since RSNN isn't nn.Module with autograd)
            error  = (pred - y.to(pred.device))
            # Approximate Fisher: outer product of gradient
            grad_W_out = torch.outer(error, spikes).detach()
            fisher["W_out"] += grad_W_out.pow(2) / n

            # For W_rec, use spike outer product as proxy
            grad_W_rec = torch.outer(rsnn.prev_spikes, spikes).detach()
            fisher["W_rec"] += grad_W_rec.pow(2) / n

            count += 1

        return fisher

    def consolidate(
        self,
        rsnn,
        readout,
        data_stream,
    ) -> None:
        """
        Consolidate current task parameters.
        Call after finishing training on each task.
        """
        if len(self.task_params) >= self.cfg.n_tasks:
            self.task_params.pop(0)
            self.fisher_diag.pop(0)

        # Save current parameters
        params = {
            "W_rec": rsnn.W_rec.detach().clone(),
            "W_out": readout.W.detach().clone(),
        }
        fisher = self.compute_fisher(rsnn, readout, data_stream)

        self.task_params.append(params)
        self.fisher_diag.append(fisher)

    def penalty(self, rsnn, readout) -> torch.Tensor:
        """
        Compute EWC penalty term (add to training loss / subtract from update).

        Returns a scalar penalty tensor.
        """
        penalty = torch.tensor(0.0)
        for params, fisher in zip(self.task_params, self.fisher_diag):
            for name, θ_star in params.items():
                F    = fisher[name].to(θ_star.device)
                θ    = rsnn.W_rec if name == "W_rec" else readout.W
                diff = (θ.detach() - θ_star).pow(2)
                penalty = penalty + (F * diff).sum()

        return self.cfg.ewc_lambda / 2.0 * penalty

    def apply_gradient_penalty(self, rsnn, readout, lr: float) -> None:
        """
        Apply EWC penalty as a direct weight regularisation step.
        Modifies W_rec and W_out in-place (no autograd required).
        """
        for params, fisher in zip(self.task_params, self.fisher_diag):
            F_rec  = fisher["W_rec"].to(rsnn.W_rec.device)
            θ_rec  = params["W_rec"].to(rsnn.W_rec.device)
            grad_rec = F_rec * (rsnn.W_rec.detach() - θ_rec)
            rsnn.W_rec.data -= lr * self.cfg.ewc_lambda * grad_rec

            F_out  = fisher["W_out"].to(readout.W.device)
            θ_out  = params["W_out"].to(readout.W.device)
            grad_out = F_out * (readout.W.detach() - θ_out)
            readout.W.data -= lr * self.cfg.ewc_lambda * grad_out


class SynapticIntelligence:
    """
    Synaptic Intelligence (Zenke et al. 2017).

    Accumulates per-synapse importance Ω_i online during learning:

        Ω_i += -(∂L/∂θ_i) · Δθ_i / (Δθ_i² + ε)

    This requires no Fisher computation or replay buffer — importance
    accumulates as a by-product of normal online training.
    """

    def __init__(self, config: ContinualConfig) -> None:
        self.cfg = config
        self.omega_rec:  Optional[torch.Tensor] = None   # importance
        self.omega_out:  Optional[torch.Tensor] = None
        self._w_rec_prev: Optional[torch.Tensor] = None  # params at task start
        self._w_out_prev: Optional[torch.Tensor] = None
        self._w_rec_task: Optional[torch.Tensor] = None  # task-A params
        self._w_out_task: Optional[torch.Tensor] = None

    def init_task(self, rsnn, readout) -> None:
        """Call at the start of each new task."""
        n_rec = rsnn.W_rec.numel()
        n_out = readout.W.numel()
        if self.omega_rec is None:
            self.omega_rec = torch.zeros_like(rsnn.W_rec)
            self.omega_out = torch.zeros_like(readout.W)
        self._w_rec_prev = rsnn.W_rec.detach().clone()
        self._w_out_prev = readout.W.detach().clone()
        self._w_rec_task = rsnn.W_rec.detach().clone()
        self._w_out_task = readout.W.detach().clone()

    def update_importance(self, rsnn, readout, error: torch.Tensor, spikes: torch.Tensor) -> None:
        """
        Update importance after each training step.
        Call immediately after the weight update.
        """
        if self._w_rec_prev is None:
            return

        ε = self.cfg.si_epsilon
        # Approximate gradient contribution for W_rec (Hebbian proxy)
        # ∂L/∂W_rec ≈ error_norm * outer(prev_spikes, spikes)
        err_norm   = float(error.norm().item())
        delta_rec  = rsnn.W_rec.detach() - self._w_rec_prev
        importance = err_norm * delta_rec.abs() / (delta_rec.pow(2) + ε)
        self.omega_rec = self.omega_rec + importance.clamp(min=0)
        self._w_rec_prev = rsnn.W_rec.detach().clone()

        delta_out  = readout.W.detach() - self._w_out_prev
        importance_out = err_norm * delta_out.abs() / (delta_out.pow(2) + ε)
        self.omega_out = self.omega_out + importance_out.clamp(min=0)
        self._w_out_prev = readout.W.detach().clone()

    def apply_penalty(self, rsnn, readout, lr: float) -> None:
        """
        Apply SI regularisation penalty.
        Modifies W_rec and W_out in-place.
        """
        if self.omega_rec is None or self._w_rec_task is None:
            return
        ξ = self.cfg.si_xi

        θ_rec  = self._w_rec_task.to(rsnn.W_rec.device)
        Ω_rec  = self.omega_rec.to(rsnn.W_rec.device)
        grad_r = Ω_rec * (rsnn.W_rec.detach() - θ_rec)
        rsnn.W_rec.data -= lr * ξ * grad_r

        θ_out  = self._w_out_task.to(readout.W.device)
        Ω_out  = self.omega_out.to(readout.W.device)
        grad_o = Ω_out * (readout.W.detach() - θ_out)
        readout.W.data -= lr * ξ * grad_o


class ContinualLearner:
    """
    Combined EWC + SI continual learner.

    Wraps both methods and provides a unified API:
        learner.step(rsnn, readout, x, y, spikes, error)
        learner.finish_task(rsnn, readout, data_stream)
    """

    def __init__(self, config: Optional[ContinualConfig] = None) -> None:
        self.cfg     = config or ContinualConfig()
        self.ewc     = EWCRegulariser(self.cfg)
        self.si      = SynapticIntelligence(self.cfg)
        self.n_tasks = 0

    def begin_task(self, rsnn, readout) -> None:
        """Call at the start of each new task."""
        self.si.init_task(rsnn, readout)
        self.n_tasks += 1

    def step(self, rsnn, readout, spikes, error, lr: float) -> None:
        """
        Apply continual learning regularisation after a normal training step.
        Call this AFTER the normal weight update.
        """
        method = self.cfg.method
        if method in ("ewc", "both") and self.n_tasks > 1:
            self.ewc.apply_gradient_penalty(rsnn, readout, lr)
        if method in ("si", "both"):
            self.si.update_importance(rsnn, readout, error, spikes)
            if self.n_tasks > 1:
                self.si.apply_penalty(rsnn, readout, lr)

    def finish_task(self, rsnn, readout, data_stream) -> None:
        """Consolidate current task. Call after training completes on a task."""
        if self.cfg.method in ("ewc", "both"):
            self.ewc.consolidate(rsnn, readout, data_stream)
