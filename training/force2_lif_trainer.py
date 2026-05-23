"""
force2_lif_trainer.py
=====================
LIF-based FORCE2 trainer - spiking neuron implementation.

Integrates Arthedain's LIF neurons with FORCE training:
- Spiking dynamics (LIF with refractory period)
- Filtered spike trains for readout (exponential filtering)
- RLS applied to filtered spikes
- Chaotic initialization from paper
- Multi-timescale synaptic currents

This matches the paper's implementation and should achieve
correlations >0.95 on oscillator tasks.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

from models.lif import LIFLayer, LIFConfig
from models.force_enhanced import (
    ChaoticInitConfig, ChaoticInitializer,
    MultiTimescaleSynapseConfig, MultiTimescaleSynapses,
    PatternGenerator,
)


@dataclass
class FORCE2LIFConfig:
    """Configuration for LIF-based FORCE2 trainer."""
    
    # Network architecture
    n_neurons: int = 1000
    n_outputs: int = 1
    
    # LIF parameters
    lif_tau: float = 20.0          # Membrane time constant (ms)
    lif_v_th: float = 1.0          # Spike threshold
    lif_refractory: int = 2        # Refractory period (timesteps)
    
    # Chaotic initialization
    chaotic_cfg: Optional[ChaoticInitConfig] = None
    
    # Spike filtering for readout (exponential filter)
    # The paper uses filtered spikes: r(t) = alpha*r(t-1) + (1-alpha)*s(t)
    filter_tau: float = 20.0       # Filter time constant (ms)
    
    # Multi-timescale synaptic currents (to LIF)
    multi_tau_cfg: Optional[MultiTimescaleSynapseConfig] = None
    
    # RLS parameters
    alpha_rls: float = 1.0
    forgetting_factor: float = 0.9995
    regularization: float = 1e-6
    
    # Training
    train_readout: bool = True
    train_recurrent: bool = False
    
    # Skip logic
    skip_below_error: float = 0.001


class FilteredSpikeBuffer:
    """
    Exponentially filtered spike buffer.
    
    The paper uses filtered spike trains for the readout:
        r(t) = alpha * r(t-1) + (1-alpha) * s(t)
    
    where s(t) is the binary spike and r(t) is the filtered rate.
    This provides smoother dynamics for the RLS algorithm.
    """
    
    def __init__(self, n_neurons: int, filter_tau: float, device: str = "cpu"):
        self.n_neurons = n_neurons
        self.filter_tau = filter_tau
        self.device = device
        
        # Filter decay factor
        self.alpha = 1.0 - 1.0 / filter_tau
        
        # Filtered spike train (the "rate")
        self.register_buffer("r", torch.zeros(n_neurons, device=device))
    
    def register_buffer(self, name: str, tensor: torch.Tensor):
        """Simulate nn.Module buffer registration."""
        setattr(self, name, tensor)
    
    def update(self, spikes: torch.Tensor) -> torch.Tensor:
        """
        Update filtered spike train with new spikes.
        
        Args:
            spikes: Binary spike tensor (n_neurons,)
            
        Returns:
            Filtered rate r(t)
        """
        # r(t) = alpha * r(t-1) + (1-alpha) * s(t)
        self.r = self.alpha * self.r + (1.0 - self.alpha) * spikes
        return self.r
    
    def reset(self):
        """Reset filtered buffer."""
        self.r.zero_()


class FORCE2LIFTrainer(nn.Module):
    """
    LIF-based FORCE2 trainer using Arthedain's spiking neurons.
    
    This implements the full spiking network from Nicola & Clopath 2017:
    - LIF neurons with refractory periods
    - Chaotic recurrent connectivity
    - Multi-timescale synaptic currents
    - Filtered spike trains for readout
    - RLS on filtered spikes
    
    The filtered spike train r(t) is what the readout sees and what RLS
    uses for weight updates, not the raw binary spikes s(t).
    """
    
    def __init__(self, cfg: Optional[FORCE2LIFConfig] = None, device: str = "cpu"):
        super().__init__()
        self.cfg = cfg or FORCE2LIFConfig()
        self.device = device
        
        n = self.cfg.n_neurons
        n_out = self.cfg.n_outputs
        
        # LIF neuron layer
        lif_cfg = LIFConfig(
            size=n,
            tau=self.cfg.lif_tau,
            v_th=self.cfg.lif_v_th,
            refractory=self.cfg.lif_refractory,
            device=device,
        )
        self.lif = LIFLayer(lif_cfg)
        
        # Chaotic recurrent weights
        chaotic_init = ChaoticInitializer(self.cfg.chaotic_cfg)
        self.register_buffer(
            "W_rec",
            chaotic_init.initialize(n, device)
        )
        self.initial_spectral_radius = chaotic_init.compute_spectral_radius(self.W_rec)
        
        # Input weights (for external input)
        self.W_in = nn.Parameter(torch.randn(n, 1, device=device) * 0.5)
        
        # Feedback weights (readout back to network - crucial for FORCE)
        # This creates the recurrent loop: spikes -> readout -> network input
        fb_scale = 1.0 / np.sqrt(n_out)
        self.W_fb = nn.Parameter(torch.randn(n, n_out, device=device) * fb_scale)
        
        # Bias current to maintain spontaneous activity
        self.register_buffer("bias", torch.ones(n, device=device) * 2.0)
        
        # Multi-timescale synaptic currents
        if self.cfg.multi_tau_cfg:
            self.synapses = MultiTimescaleSynapses(n, self.cfg.multi_tau_cfg, device)
        else:
            self.synapses = None
        
        # Filtered spike buffer (for readout and RLS)
        self.filter_buffer = FilteredSpikeBuffer(
            n_neurons=n,
            filter_tau=self.cfg.filter_tau,
            device=device,
        )
        
        # Initialize with small random activity to kickstart dynamics
        self.filter_buffer.r = torch.rand(n, device=device) * 0.1
        
        # Readout weights (trained with RLS)
        # Initialize with small values to allow learning
        self.W_out = nn.Parameter(
            torch.randn(n_out, n, device=device) * 0.001,
            requires_grad=False,
        )
        
        # RLS state
        self._init_rls_state()
        
        # Tracking
        self.error_history: List[float] = []
        self.step_count = 0
        self.output_history: List[torch.Tensor] = []
        
    def _init_rls_state(self):
        """Initialize Recursive Least Squares state."""
        cfg = self.cfg
        n = cfg.n_neurons
        n_out = cfg.n_outputs
        
        # P matrices (one per output)
        self.P = [
            cfg.alpha_rls * torch.eye(n, device=self.device)
            for _ in range(n_out)
        ]
        
        self.lambda_rls = cfg.forgetting_factor
    
    def step(self, x: torch.Tensor, target: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through spiking network.
        
        Args:
            x: External input (batch, 1) or (1,)
            target: Target output for teacher forcing during training
            
        Returns:
            Output prediction from readout
        """
        # Handle input
        if x.dim() == 0:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 1:
            x = x.unsqueeze(1)
        
        # Compute input current
        I_in = torch.matmul(self.W_in, x.T).squeeze(-1)  # (n_neurons,)
        
        # Compute recurrent current
        r_filtered = self.filter_buffer.r
        
        if self.synapses:
            s_syn = self.synapses.step(r_filtered)
            I_rec = torch.matmul(self.W_rec, s_syn)
        else:
            I_rec = torch.matmul(self.W_rec, r_filtered)
        
        # Teacher forcing: during training, feed back the TARGET, not the output
        # This is the key to FORCE - it "forces" the network to follow the target
        if target is not None:
            # Training mode: use target for feedback (teacher forcing)
            if target.dim() == 0:
                target = target.unsqueeze(0)
            # W_fb is (n_neurons, n_outputs), target is (n_outputs,)
            I_fb = torch.matmul(self.W_fb, target)
        elif len(self.output_history) > 0:
            # Test mode: use actual output
            prev_output = self.output_history[-1]
            I_fb = torch.matmul(self.W_fb, prev_output)
        else:
            I_fb = torch.zeros_like(self.bias)
        
        # Total current to LIF
        I_total = I_in + I_rec + I_fb + self.bias
        
        # Run LIF step (spiking dynamics)
        spikes = self.lif.step(I_total)  # Binary spikes s(t)
        
        # Filter spikes for readout
        r_filtered = self.filter_buffer.update(spikes)  # Continuous r(t)
        
        # Readout (linear combination of filtered spikes)
        output = torch.matmul(self.W_out, r_filtered)
        
        # Store
        self.output_history.append(output.detach().clone())
        if len(self.output_history) > 1000:
            self.output_history.pop(0)
        
        self.step_count += 1
        
        return output
    
    def update(self, target: torch.Tensor, error_weight: float = 1.0) -> float:
        """
        Update readout weights using RLS on filtered spikes.
        
        This is the key: RLS uses the filtered spike train r(t), not
        the binary spikes s(t). This provides smoother gradients and
        better convergence.
        """
        if not self.cfg.train_readout or len(self.output_history) == 0:
            return 0.0
        
        y_pred = self.output_history[-1]
        
        # Handle dimensions
        if target.dim() == 0:
            target = target.unsqueeze(0)
        if y_pred.dim() == 0:
            y_pred = y_pred.unsqueeze(0)
        
        # Compute error
        error = target - y_pred
        error_norm = error.norm().item()
        
        # Track
        self.error_history.append(error_norm)
        if len(self.error_history) > 100:
            self.error_history.pop(0)
        
        # Skip small errors
        if error_norm < self.cfg.skip_below_error:
            return error_norm
        
        # RLS update using FILTERED spikes
        r_filtered = self.filter_buffer.r  # This is r(t), not s(t)
        
        for k in range(self.cfg.n_outputs):
            e_k = error[k] if error.dim() > 0 else error
            
            P_k = self.P[k]
            
            # Gain: g = P @ r / (lambda + r^T @ P @ r)
            denom = self.lambda_rls + torch.dot(r_filtered, P_k @ r_filtered) + self.cfg.regularization
            g = (P_k @ r_filtered) / denom
            
            # Update P
            self.P[k] = (P_k - torch.outer(g, r_filtered @ P_k)) / self.lambda_rls
            
            # Update weights: W[k] += e_k * g
            self.W_out.data[k] = self.W_out.data[k] + e_k * g * error_weight
        
        return error_norm
    
    def train_step(self, x: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Combined forward + update with teacher forcing."""
        # Pass target to step for teacher forcing (key for FORCE)
        y_pred = self.step(x, target=target)
        error_norm = self.update(target)
        return y_pred, error_norm
    
    def reset_state(self):
        """Reset network state (LIF, filters, history)."""
        self.lif.reset()
        self.filter_buffer.reset()
        if self.synapses:
            self.synapses.reset()
        self.output_history.clear()
    
    def reset_rls(self):
        """Reset RLS state."""
        self._init_rls_state()
    
    def get_stats(self) -> Dict[str, float]:
        """Get training statistics."""
        stats = {
            "step": self.step_count,
            "spectral_radius": self.initial_spectral_radius,
        }
        
        if self.error_history:
            stats["error_mean"] = np.mean(self.error_history[-100:])
            stats["error_last"] = self.error_history[-1]
        
        # Firing rate
        rates = self.lif.get_firing_rates(window=100)
        stats["mean_firing_rate"] = rates.mean().item()
        
        # RLS uncertainty
        if hasattr(self, 'P'):
            stats["rls_uncertainty"] = np.mean([p.trace().item() for p in self.P])
        
        return stats
    
    def train_on_pattern(
        self,
        pattern: torch.Tensor,
        input_signal: Optional[torch.Tensor] = None,
        n_epochs: int = 1,
    ) -> List[float]:
        """
        Train on target pattern.
        
        Args:
            pattern: Target (n_steps, n_outputs) or (n_steps,)
            input_signal: Optional input
            n_epochs: Training epochs
            
        Returns:
            List of error norms
        """
        if pattern.dim() == 1:
            pattern = pattern.unsqueeze(1)
        
        n_steps = pattern.shape[0]
        errors = []
        
        for epoch in range(n_epochs):
            self.reset_state()
            
            for t in range(n_steps):
                if input_signal is not None:
                    x = input_signal[t] if input_signal.dim() > 1 else input_signal[t].unsqueeze(0)
                else:
                    x = torch.zeros(1, device=self.device)
                
                _, error = self.train_step(x, pattern[t])
                errors.append(error)
        
        return errors


# -----------------------------------------------------------------------------
# Factory Functions
# -----------------------------------------------------------------------------

def make_lif_force_trainer_for_oscillator(
    freq: float,
    n_neurons: int = 800,
    device: str = "cpu",
) -> FORCE2LIFTrainer:
    """
    Create LIF-based FORCE2 trainer for oscillator learning.
    
    Optimized for learning simple sinusoidal oscillators.
    """
    # Original alpha_rls works well - larger networks need stronger feedback
    alpha_rls = 1.0
    
    cfg = FORCE2LIFConfig(
        n_neurons=n_neurons,
        n_outputs=1,
        lif_tau=20.0,
        lif_refractory=2,
        chaotic_cfg=ChaoticInitConfig(target_radius=1.5),
        filter_tau=10.0,  # Faster filter for oscillators
        multi_tau_cfg=MultiTimescaleSynapseConfig(
            tau_fast=2.0,
            tau_slow=30.0,
            alpha_fast=0.7,
            alpha_slow=0.3,
        ),
        alpha_rls=alpha_rls,  # Smaller for larger networks
        forgetting_factor=0.998,  # Slightly faster forgetting for adaptation
        train_readout=True,
    )
    
    return FORCE2LIFTrainer(cfg, device)


def make_lif_force_trainer_for_chaos(
    n_neurons: int = 2000,
    n_outputs: int = 3,
    device: str = "cpu",
) -> FORCE2LIFTrainer:
    """
    Create LIF-based FORCE2 trainer for chaotic attractors.
    
    Larger network with higher spectral radius for complex dynamics.
    """
    cfg = FORCE2LIFConfig(
        n_neurons=n_neurons,
        n_outputs=n_outputs,
        lif_tau=20.0,
        lif_refractory=2,
        chaotic_cfg=ChaoticInitConfig(
            target_radius=1.8,
            connectivity_p=0.15,
        ),
        filter_tau=30.0,
        multi_tau_cfg=MultiTimescaleSynapseConfig(
            tau_fast=5.0,
            tau_slow=100.0,
            tau_ultra=300.0,
            alpha_fast=0.4,
            alpha_slow=0.4,
            alpha_ultra=0.2,
        ),
        alpha_rls=2.0,
        forgetting_factor=0.9995,
        train_readout=True,
    )
    
    return FORCE2LIFTrainer(cfg, device)


def make_lif_force_trainer_full(
    n_neurons: int = 1000,
    n_outputs: int = 1,
    device: str = "cpu",
) -> FORCE2LIFTrainer:
    """Fully configurable LIF-based FORCE2 trainer."""
    cfg = FORCE2LIFConfig(
        n_neurons=n_neurons,
        n_outputs=n_outputs,
    )
    return FORCE2LIFTrainer(cfg, device)
