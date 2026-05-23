"""
Modular learning rules for Arthedain.
Can be swapped into the trainer independently.
"""

import torch
from typing import Optional
from dataclasses import dataclass


@dataclass
class AdaptiveLRConfig:
    """Configuration for adaptive learning rate scheduler."""
    base_lr: float = 1e-3
    warmup_steps: int = 100
    min_lr: float = 1e-6
    max_lr: float = 1e-2
    decay_factor: float = 0.95
    decay_every: int = 1000
    grad_norm_threshold: float = 10.0  # Reduce LR if grad norm exceeds this


class AdaptiveLRScheduler:
    """
    Adaptive learning rate scheduler with warmup and decay.

    Features:
    - Linear warmup for first N steps
    - Exponential decay every N steps
    - Automatic LR reduction on gradient explosion
    - Per-method learning rate tracking
    """
    def __init__(self, config: AdaptiveLRConfig):
        self.cfg = config
        self.step_count = 0
        self.grad_norm_history = []
        self.current_lr = config.base_lr

    def get_lr(self, grad_norm: Optional[float] = None) -> float:
        """Get current learning rate with warmup and decay."""
        self.step_count += 1

        # Warmup phase
        if self.step_count <= self.cfg.warmup_steps:
            return self.cfg.base_lr * (self.step_count / self.cfg.warmup_steps)

        # Exponential decay
        decay_steps = self.step_count // self.cfg.decay_every
        lr = self.cfg.base_lr * (self.cfg.decay_factor ** decay_steps)

        # Gradient-based adaptation
        if grad_norm is not None:
            self.grad_norm_history.append(grad_norm)
            if len(self.grad_norm_history) > 100:
                self.grad_norm_history.pop(0)

            # Reduce LR if gradient norm is exploding
            if grad_norm > self.cfg.grad_norm_threshold:
                lr *= 0.5

        self.current_lr = max(self.cfg.min_lr, min(self.cfg.max_lr, lr))
        return self.current_lr

    def reset(self):
        """Reset scheduler state."""
        self.step_count = 0
        self.grad_norm_history.clear()
        self.current_lr = self.cfg.base_lr


def delta_rule(W: torch.Tensor, error: torch.Tensor, pre: torch.Tensor, lr: float):
    """Standard supervised delta rule. ΔW = lr * error ⊗ pre"""
    W += lr * torch.outer(error, pre)


def rstdp(
    W: torch.Tensor,
    eligibility: torch.Tensor,
    reward: float,
    baseline: float,
    lr: float,
):
    """
    Reward-modulated STDP.
    ΔW = lr * (R - b) * E
    """
    W += lr * (reward - baseline) * eligibility


def bcm_rule(
    W: torch.Tensor,
    pre: torch.Tensor,
    post: torch.Tensor,
    threshold: torch.Tensor,
    lr: float,
):
    """
    Bienenstock-Cooper-Munro rule.
    ΔW = lr * post * (post - θ) * pre
    Θ slides with mean activity → selectivity.
    """
    phi = post * (post - threshold)
    W += lr * torch.outer(phi, pre)


def oja_rule(W: torch.Tensor, pre: torch.Tensor, post: torch.Tensor, lr: float):
    """
    Oja's rule — Hebbian with weight normalization.
    ΔW = lr * (post ⊗ pre - post² * W)
    """
    W += lr * (torch.outer(post, pre) - (post ** 2).unsqueeze(1) * W)
