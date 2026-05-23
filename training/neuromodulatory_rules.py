"""
neuromodulatory_rules.py
========================
Neuromodulatory eligibility trace learning rules.

Based on Meta-SpikePropamine (Schmidgall & Hays, 2023):
- Three-factor learning: pre-synaptic, post-synaptic, and neuromodulatory signal
- Eligibility traces accumulate Hebbian correlations locally
- Neuromodulator gates when learning occurs (dopamine-like signal)

This extends SNNTraining's dual-timescale Hebbian with a global third factor
to enable reward-based and error-based learning.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class NeuromodulatoryConfig:
    """Configuration for neuromodulatory eligibility learning."""
    
    # Eligibility trace decay (γ in paper)
    # Higher = longer-lasting traces (slower decay)
    eligibility_decay: float = 0.95
    
    # Trace accumulation rate (α in paper)
    trace_lr: float = 1.0
    
    # Weight soft bounds
    w_max: float = 1.0
    w_min: float = 0.0
    
    # Learning rate for actual weight updates
    weight_lr: float = 1e-3
    
    # Device
    device: Optional[str] = None


class NeuromodulatoryEligibilityTrace(nn.Module):
    """
    Neuromodulatory eligibility trace learning rule.
    
    Implements three-factor learning from Meta-SpikePropamine:
    1. Pre-synaptic activity creates trace
    2. Post-synaptic activity + eligibility → weight change candidate
    3. Neuromodulatory signal gates actual weight change
    
    The neuromodulator M(t) can be:
    - Reward signal (for RL)
    - Prediction error (for supervised learning)
    - Error signal from readout (as in e-prop/FORCE)
    
    Weight update: ΔW = M(t) * eligibility(t)
    
    This is more biologically plausible than pure Hebbian and enables
    reward-based learning scenarios.
    """
    
    def __init__(
        self,
        shape: Tuple[int, int],
        cfg: Optional[NeuromodulatoryConfig] = None,
        device: str = "cpu",
    ):
        super().__init__()
        self.shape = shape  # (n_post, n_pre)
        self.cfg = cfg or NeuromodulatoryConfig()
        self.device = device
        
        # Eligibility trace buffer E(t)
        # Accumulates Hebbian correlations: E += pre * post
        # Then decays: E *= γ
        self.register_buffer("eligibility", torch.zeros(shape, device=device))
        
        # Pre-synaptic trace x_pre(t) - for pair-based STDP
        self.register_buffer("x_pre", torch.zeros(shape[1], device=device))
        
        # Post-synaptic trace x_post(t)
        self.register_buffer("x_post", torch.zeros(shape[0], device=device))
        
        # Running statistics for normalization
        self.register_buffer("e_mean", torch.tensor(0.0, device=device))
        self.register_buffer("e_var", torch.tensor(1.0, device=device))
        self.e_count = 0
        
    def compute_eligibility(
        self,
        pre_spikes: torch.Tensor,  # (n_pre,)
        post_spikes: torch.Tensor,  # (n_post,)
    ) -> torch.Tensor:
        """
        Compute new eligibility from spike pair.
        
        This is the Hebbian part: E += pre * post
        The eligibility accumulates spike correlations.
        """
        # Update pre and post traces (synaptic activity traces)
        # x_pre(t) = γ * x_pre(t-1) + α * pre_spike
        self.x_pre.mul_(self.cfg.eligibility_decay).add_(pre_spikes, alpha=self.cfg.trace_lr)
        self.x_post.mul_(self.cfg.eligibility_decay).add_(post_spikes, alpha=self.cfg.trace_lr)
        
        # Pair-based STDP eligibility:
        # When post spikes, add pre-trace to eligibility (LTP)
        # When pre spikes, subtract post-trace from eligibility (LTD)
        
        # LTP: post spike → potentiate by pre-trace
        ltp = torch.outer(post_spikes, self.x_pre)  # (n_post, n_pre)
        
        # LTD: pre spike → depress by post-trace
        ltd = torch.outer(self.x_post, pre_spikes)  # (n_post, n_pre)
        
        # Combined eligibility update
        delta_e = ltp - ltd
        
        # Accumulate into eligibility trace with decay
        self.eligibility.mul_(self.cfg.eligibility_decay).add_(delta_e)
        
        # Update statistics for normalization
        self.e_count += 1
        if self.e_count % 100 == 0:
            self.e_mean = self.eligibility.mean()
            self.e_var = self.eligibility.var() + 1e-8
        
        return self.eligibility
    
    def apply_neuromodulator(
        self,
        W: torch.Tensor,
        neuromodulator: torch.Tensor,
        weight_lr: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Apply neuromodulatory signal to produce weight change.
        
        Args:
            W: Weight matrix to update
            neuromodulator: Global third factor M(t)
                - Can be scalar (broadcast to all synapses)
                - Can be per-neuron
                - Can be per-synapse (full matrix)
            weight_lr: Override learning rate
            
        Returns:
            delta_W: Weight change for logging/analysis
        """
        lr = weight_lr or self.cfg.weight_lr
        
        # Normalize eligibility for stability
        e_normalized = self.eligibility / (torch.sqrt(self.e_var) + 1e-8)
        
        # Three-factor learning: ΔW = M(t) * eligibility(t)
        # The neuromodulator gates when and how much learning occurs
        if neuromodulator.dim() == 0:
            # Scalar neuromodulator - broadcast to all synapses
            delta_W = lr * neuromodulator * e_normalized
        elif neuromodulator.dim() == 1:
            if neuromodulator.shape[0] == self.shape[0]:
                # Per-post-neuron neuromodulator
                delta_W = lr * neuromodulator.unsqueeze(1) * e_normalized
            elif neuromodulator.shape[0] == self.shape[1]:
                # Per-pre-neuron neuromodulator
                delta_W = lr * neuromodulator.unsqueeze(0) * e_normalized
            else:
                raise ValueError(f"Neuromodulator shape {neuromodulator.shape} incompatible")
        else:
            # Full matrix neuromodulator
            delta_W = lr * neuromodulator * e_normalized
        
        # Apply soft weight bounds (multiplicative weight dependence)
        # LTP scaled by (W_max - W), LTD scaled by (W - W_min)
        w_range = self.cfg.w_max - self.cfg.w_min
        if w_range > 0:
            # Soft bounds: scale update by proximity to bounds
            ltp_mask = (delta_W > 0).float()
            ltd_mask = (delta_W < 0).float()
            
            # Scale LTP by remaining room to w_max
            ltp_scale = (self.cfg.w_max - W) / w_range
            # Scale LTD by room down to w_min  
            ltd_scale = (W - self.cfg.w_min) / w_range
            
            delta_W = delta_W * (ltp_mask * ltp_scale + ltd_mask * ltd_scale)
        
        # Apply update
        W.add_(delta_W)
        
        # Hard clamp for safety
        W.clamp_(self.cfg.w_min, self.cfg.w_max)
        
        return delta_W
    
    def update(
        self,
        W: torch.Tensor,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        neuromodulator: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full update: compute eligibility and apply neuromodulator.
        
        Args:
            W: Weight matrix (modified in-place)
            pre_spikes: Pre-synaptic spikes
            post_spikes: Post-synaptic spikes
            neuromodulator: Global third factor
            
        Returns:
            (eligibility, delta_W) for monitoring
        """
        E = self.compute_eligibility(pre_spikes, post_spikes)
        delta_W = self.apply_neuromodulator(W, neuromodulator)
        return E, delta_W
    
    def reset(self):
        """Reset all traces."""
        self.eligibility.zero_()
        self.x_pre.zero_()
        self.x_post.zero_()
        self.e_mean.zero_()
        self.e_var.fill_(1.0)
        self.e_count = 0
    
    def get_stats(self) -> dict:
        """Get trace statistics."""
        return {
            "eligibility_mean": self.eligibility.mean().item(),
            "eligibility_std": self.eligibility.std().item(),
            "eligibility_max": self.eligibility.abs().max().item(),
            "x_pre_mean": self.x_pre.mean().item(),
            "x_post_mean": self.x_post.mean().item(),
        }


class RewardModulatedSTDP(NeuromodulatoryEligibilityTrace):
    """
    Reward-modulated STDP (R-STDP) for reinforcement learning.
    
    The neuromodulator is the reward signal R(t):
    - Positive reward → potentiate active synapses
    - Negative reward → depress active synapses
    
    This enables learning from sparse reward signals,
    similar to dopamine-based learning in the brain.
    """
    
    def __init__(
        self,
        shape: Tuple[int, int],
        cfg: Optional[NeuromodulatoryConfig] = None,
        device: str = "cpu",
        baseline_reward: float = 0.0,
    ):
        super().__init__(shape, cfg, device)
        self.baseline_reward = baseline_reward
        
        # Running average of reward for baseline subtraction
        self.register_buffer("reward_ema", torch.tensor(baseline_reward, device=device))
        self.reward_alpha = 0.01  # EMA decay
    
    def update_with_reward(
        self,
        W: torch.Tensor,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        reward: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update with reward signal.
        
        Args:
            W: Weight matrix
            pre_spikes: Pre-synaptic activity
            post_spikes: Post-synaptic activity
            reward: Scalar reward signal (can be positive or negative)
        """
        # Update reward baseline
        if reward.dim() == 0:
            self.reward_ema = (1 - self.reward_alpha) * self.reward_ema + self.reward_alpha * reward
        
        # Reward prediction error (advantage)
        # Positive when reward > expected, negative when reward < expected
        advantage = reward - self.reward_ema
        
        # Use advantage as neuromodulator
        return self.update(W, pre_spikes, post_spikes, advantage)


class ErrorModulatedSTDP(NeuromodulatoryEligibilityTrace):
    """
    Error-modulated STDP for supervised learning.
    
    The neuromodulator is the prediction error:
    M(t) = target(t) - prediction(t)
    
    This is similar to FORCE training but with eligibility traces
    and biological-style weight updates.
    """
    
    def __init__(
        self,
        shape: Tuple[int, int],
        cfg: Optional[NeuromodulatoryConfig] = None,
        device: str = "cpu",
    ):
        super().__init__(shape, cfg, device)
    
    def update_with_error(
        self,
        W: torch.Tensor,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        error: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update with prediction error.
        
        Args:
            W: Weight matrix
            pre_spikes: Pre-synaptic activity
            post_spikes: Post-synaptic activity
            error: Prediction error (target - prediction)
                   Can be scalar or per-output-neuron
        """
        # Use error directly as neuromodulator
        # Positive error → increase weight (reduce error)
        # Negative error → decrease weight (reduce error)
        return self.update(W, pre_spikes, post_spikes, error)


# -----------------------------------------------------------------------------
# Meta-learning wrapper for plasticity optimization
# -----------------------------------------------------------------------------

@dataclass
class MetaPlasticityConfig:
    """Configuration for meta-learning plasticity parameters."""
    
    # What to meta-learn
    learn_eligibility_decay: bool = True
    learn_trace_lr: bool = True
    learn_weight_lr: bool = True
    
    # Meta-learning rate
    meta_lr: float = 1e-4
    
    # Inner loop steps per meta-update
    inner_steps: int = 10


class MetaLearnablePlasticity(nn.Module):
    """
    Meta-learning wrapper for plasticity parameters.
    
    The outer loop optimizes the plasticity rule parameters
    (eligibility decay, trace learning rate, etc.) while the
    inner loop uses these parameters for actual learning.
    
    This implements the Meta-SpikePropamine concept: learning
    how to learn by optimizing the learning algorithm itself.
    """
    
    def __init__(
        self,
        base_plasticity: NeuromodulatoryEligibilityTrace,
        cfg: MetaPlasticityConfig,
    ):
        super().__init__()
        self.base = base_plasticity
        self.cfg = cfg
        
        # Learnable plasticity parameters (meta-parameters)
        if cfg.learn_eligibility_decay:
            # Initialize as logit for [0,1] constraint
            self.eligibility_decay_logit = nn.Parameter(
                torch.tensor(1.5)  # ~0.95 after sigmoid
            )
        
        if cfg.learn_trace_lr:
            self.trace_lr = nn.Parameter(torch.tensor(1.0))
        
        if cfg.learn_weight_lr:
            self.weight_lr_log = nn.Parameter(torch.log(torch.tensor(1e-3)))
    
    def get_plasticity_params(self) -> dict:
        """Get current plasticity parameters (with constraints)."""
        params = {}
        
        if self.cfg.learn_eligibility_decay:
            # Sigmoid to keep in [0, 1]
            params['eligibility_decay'] = torch.sigmoid(self.eligibility_decay_logit)
        else:
            params['eligibility_decay'] = self.base.cfg.eligibility_decay
        
        if cfg.learn_trace_lr:
            params['trace_lr'] = torch.nn.functional.softplus(self.trace_lr)
        else:
            params['trace_lr'] = self.base.cfg.trace_lr
        
        if self.cfg.learn_weight_lr:
            params['weight_lr'] = torch.exp(self.weight_lr_log)
        else:
            params['weight_lr'] = self.base.cfg.weight_lr
        
        return params
    
    def inner_loop_update(
        self,
        W: torch.Tensor,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        neuromodulator: torch.Tensor,
    ):
        """Single inner loop update with current plasticity parameters."""
        params = self.get_plasticity_params()
        
        # Temporarily set base plasticity parameters
        old_decay = self.base.cfg.eligibility_decay
        old_trace_lr = self.base.cfg.trace_lr
        old_weight_lr = self.base.cfg.weight_lr
        
        self.base.cfg.eligibility_decay = params['eligibility_decay'].item()
        self.base.cfg.trace_lr = params['trace_lr'].item()
        self.base.cfg.weight_lr = params['weight_lr'].item()
        
        # Perform update
        E, delta_W = self.base.update(W, pre_spikes, post_spikes, neuromodulator)
        
        # Restore original parameters
        self.base.cfg.eligibility_decay = old_decay
        self.base.cfg.trace_lr = old_trace_lr
        self.base.cfg.weight_lr = old_weight_lr
        
        return E, delta_W
