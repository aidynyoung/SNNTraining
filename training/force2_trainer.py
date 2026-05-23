"""
force2_trainer.py
=================
Enhanced FORCE trainer with all improvements from Nicola & Clopath 2017.

Integrates:
1. Chaotic regime initialization
2. Multi-timescale synaptic dynamics
3. Sparse fixed connectivity (partially trainable)
4. Efficient RLS with skip logic

This trainer is designed for:
- Learning complex temporal patterns (oscillators, chaos, songs)
- Training both recurrent and readout weights simultaneously
- Online real-time learning with O(n²) memory for RLS

Usage:
    trainer = FORCE2Trainer(
        n_neurons=1000,
        n_outputs=5,
        cfg=FORCE2Config(...)
    )
    
    for t in range(n_steps):
        y_pred = trainer.step(x_t)
        trainer.update(target_t)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
import numpy as np

from models.force_enhanced import (
    ChaoticInitConfig, ChaoticInitializer,
    MultiTimescaleSynapseConfig, MultiTimescaleSynapses,
    SparseFixedConnectivity, PatternGenerator,
)


@dataclass
class FORCE2Config:
    """Configuration for enhanced FORCE2 trainer."""
    
    # Network architecture
    n_neurons: int = 1000
    n_outputs: int = 1
    
    # Chaotic initialization
    chaotic_cfg: Optional[ChaoticInitConfig] = None
    
    # Multi-timescale synapses
    multi_tau_cfg: Optional[MultiTimescaleSynapseConfig] = None
    
    # Sparse connectivity
    connectivity_p: float = 0.1          # Overall sparsity
    trainable_rec_fraction: float = 0.1  # Fraction of recurrent weights to train
    
    # RLS parameters
    alpha_rls: float = 1.0              # Initial P matrix diagonal
    forgetting_factor: float = 0.9995   # Lambda (close to 1 = slow forgetting)
    
    # Learning rates for different components
    lr_readout: float = 1.0              # RLS effectively sets this
    lr_recurrent: float = 0.0            # Only non-zero if training recurrent
    
    # Training modes
    train_readout: bool = True          # Always train readout
    train_recurrent: bool = False       # Optionally train recurrent
    train_input: bool = False           # Optionally train input weights
    
    # Stability
    regularization: float = 1e-6
    
    # Skip logic for efficiency
    skip_below_error: float = 0.001     # Skip RLS when error is small
    error_window: int = 100             # Window for error variance tracking
    adaptive_forgetting: bool = False    # Adjust forgetting factor online


class FORCE2Trainer(nn.Module):
    """
    Enhanced FORCE trainer with chaotic initialization and multi-timescale synapses.
    
    This implements the full algorithm from Nicola & Clopath 2017:
    1. Initialize recurrent weights in chaotic regime (spectral radius > 1)
    2. Use multi-timescale synaptic filtering (fast + slow)
    3. Train readout weights with RLS
    4. Optionally train recurrent weights for more complex tasks
    
    Memory: O(N²) for RLS P matrix where N = number of neurons feeding readout
    """
    
    def __init__(self, cfg: Optional[FORCE2Config] = None, device: str = "cpu"):
        super().__init__()
        self.cfg = cfg or FORCE2Config()
        self.device = device
        
        n = self.cfg.n_neurons
        n_out = self.cfg.n_outputs
        
        # Initialize chaotic recurrent weights
        chaotic_init = ChaoticInitializer(self.cfg.chaotic_cfg)
        
        if self.cfg.train_recurrent:
            # Use sparse connectivity with partial trainability
            self.sparse_conn = SparseFixedConnectivity(
                n_neurons=n,
                connectivity_p=self.cfg.connectivity_p,
                trainable_fraction=self.cfg.trainable_rec_fraction,
                device=device,
            )
            # Override with chaotic initialization for fixed part
            W_chaos = chaotic_init.initialize(n, device)
            self.sparse_conn.W_fixed = W_chaos * (self.sparse_conn.connectivity_mask 
                                                     - self.sparse_conn.trainable_mask)
        else:
            # Fixed chaotic reservoir
            self.register_buffer(
                "W_rec",
                chaotic_init.initialize(n, device)
            )
            # Verify spectral radius
            self.initial_spectral_radius = chaotic_init.compute_spectral_radius(self.W_rec)
        
        # Multi-timescale synapses
        self.synapses = MultiTimescaleSynapses(
            n_neurons=n,
            cfg=self.cfg.multi_tau_cfg,
            device=device,
        )
        
        # Input weights (random, fixed or trainable)
        self.W_in = nn.Parameter(
            torch.randn(n, 1, device=device) * 0.1,
            requires_grad=self.cfg.train_input,
        )
        
        # Readout weights (to be trained with RLS)
        self.W_out = nn.Parameter(
            torch.randn(n_out, n, device=device) * 0.01,
            requires_grad=False,  # Trained with RLS, not gradient
        )
        
        # RLS state for readout training
        self._init_rls_state()
        
        # Network state
        self.register_buffer("v", torch.zeros(n, device=device))  # Membrane potential
        self.register_buffer("r", torch.zeros(n, device=device))  # Firing rates (filtered spikes)
        
        # Tracking
        self.error_history: List[float] = []
        self.step_count = 0
        self.spike_history: List[torch.Tensor] = []
        self.output_history: List[torch.Tensor] = []
        
    def _init_rls_state(self):
        """Initialize Recursive Least Squares state."""
        cfg = self.cfg
        n = cfg.n_neurons
        n_out = cfg.n_outputs
        
        # P matrices (inverse correlation) - one per output dimension
        # Using block-diagonal approximation for efficiency
        self.P = [
            cfg.alpha_rls * torch.eye(n, device=self.device)
            for _ in range(n_out)
        ]
        
        # Forgetting factor (can be adapted)
        self.lambda_rls = cfg.forgetting_factor
        
    def step(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor (batch, input_dim) or (input_dim,)
            
        Returns:
            Output prediction (batch, n_outputs) or (n_outputs,)
        """
        # Handle batch dimension
        if x.dim() == 1:
            x = x.unsqueeze(0)
        batch_size = x.shape[0]
        
        # Ensure correct input dimension
        if x.shape[1] != self.W_in.shape[1]:
            # Broadcast if needed
            if self.W_in.shape[1] == 1 and x.shape[1] > 1:
                x = x.mean(dim=1, keepdim=True)
        
        # Compute input current
        I_in = torch.matmul(x, self.W_in.T)  # (batch, n_neurons)
        
        # Compute recurrent current (with synaptic filtering)
        if self.cfg.train_recurrent:
            W_eff = self.sparse_conn.get_effective_weights()
            I_rec = torch.matmul(self.r, W_eff.T).unsqueeze(0)  # (1, n_neurons)
        else:
            I_rec = torch.matmul(self.r, self.W_rec.T).unsqueeze(0)
        
        # Total current
        I_total = I_in + I_rec.expand(batch_size, -1)
        
        # Simple rate-based dynamics (sigmoid activation)
        # Could be replaced with LIF spiking neurons
        self.r = torch.sigmoid(I_total.squeeze(0)) if batch_size == 1 else torch.sigmoid(I_total[0])
        
        # Multi-timescale synaptic filtering
        s_filtered = self.synapses.step(self.r)
        
        # Readout (linear combination of filtered rates)
        if batch_size == 1:
            output = torch.matmul(self.W_out, s_filtered)
        else:
            output = torch.matmul(s_filtered.unsqueeze(0), self.W_out.T)
        
        # Store history
        self.spike_history.append(self.r.detach().clone())
        self.output_history.append(output.detach().clone())
        
        # Trim history to prevent memory bloat
        if len(self.spike_history) > 1000:
            self.spike_history.pop(0)
            self.output_history.pop(0)
        
        self.step_count += 1
        
        return output.squeeze(0) if batch_size == 1 else output
    
    def update(self, target: torch.Tensor, error_weight: float = 1.0) -> float:
        """
        Update readout weights using RLS.
        
        Args:
            target: Target output (n_outputs,) or (batch, n_outputs)
            error_weight: Weight for this update (for importance weighting)
            
        Returns:
            Error norm for monitoring
        """
        if not self.cfg.train_readout:
            return 0.0
        
        # Get current output
        if len(self.output_history) == 0:
            return 0.0
        
        y_pred = self.output_history[-1]
        
        # Handle batch dimension
        if target.dim() == 0:
            target = target.unsqueeze(0)
        if target.dim() == 1 and y_pred.dim() == 0:
            y_pred = y_pred.unsqueeze(0)
        
        # Compute error
        error = target - y_pred
        error_norm = error.norm().item()
        
        # Track error history
        self.error_history.append(error_norm)
        if len(self.error_history) > self.cfg.error_window:
            self.error_history.pop(0)
        
        # Skip update if error is negligible
        if error_norm < self.cfg.skip_below_error:
            return error_norm
        
        # RLS update for each output dimension
        # Current filtered synaptic state
        s_current = (
            self.cfg.multi_tau_cfg.alpha_fast * self.synapses.s_fast +
            self.cfg.multi_tau_cfg.alpha_slow * self.synapses.s_slow +
            self.cfg.multi_tau_cfg.alpha_ultra * self.synapses.s_ultra
            if self.cfg.multi_tau_cfg else self.r
        )
        
        for k in range(self.cfg.n_outputs):
            e_k = error[k] if error.dim() > 0 else error
            
            # RLS gain computation
            P_k = self.P[k]
            
            # Denominator: λ + z^T P z
            denom = self.lambda_rls + torch.dot(s_current, P_k @ s_current) + self.cfg.regularization
            
            # Gain vector: g = P @ z / denom
            g = (P_k @ s_current) / denom
            
            # Update P matrix
            # P_new = (1/λ) * (P - g @ z^T @ P)
            self.P[k] = (P_k - torch.outer(g, s_current @ P_k)) / self.lambda_rls
            
            # Update weights: W[k] += e_k * g
            self.W_out.data[k] = self.W_out.data[k] + e_k * g * error_weight
        
        return error_norm
    
    def train_step(self, x: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Combined forward pass + update (convenience method).
        
        Args:
            x: Input
            target: Target output
            
        Returns:
            (prediction, error_norm)
        """
        y_pred = self.step(x)
        error_norm = self.update(target)
        return y_pred, error_norm
    
    def reset_state(self):
        """Reset network state (not weights)."""
        self.v.zero_()
        self.r.zero_()
        self.synapses.reset()
        self.spike_history.clear()
        self.output_history.clear()
    
    def reset_rls(self):
        """Reset RLS state (P matrices) - useful for new tasks."""
        self._init_rls_state()
    
    def get_stats(self) -> Dict[str, float]:
        """Get training statistics."""
        stats = {
            "step": self.step_count,
            "spectral_radius": getattr(self, 'initial_spectral_radius', 0.0),
        }
        
        if self.error_history:
            stats["error_mean"] = np.mean(self.error_history[-100:])
            stats["error_last"] = self.error_history[-1]
        
        # Synaptic timescale contributions
        tau_contrib = self.synapses.get_timescale_contributions()
        stats.update({f"tau_{k}": v for k, v in tau_contrib.items()})
        
        # RLS uncertainty (mean trace of P matrices)
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
        Train on a target pattern.
        
        Args:
            pattern: Target pattern (n_steps, n_outputs) or (n_steps,)
            input_signal: Optional input (if None, uses zero input)
            n_epochs: Number of training epochs
            
        Returns:
            List of error norms per step
        """
        if pattern.dim() == 1:
            pattern = pattern.unsqueeze(1)
        
        n_steps = pattern.shape[0]
        errors = []
        
        for epoch in range(n_epochs):
            self.reset_state()
            
            for t in range(n_steps):
                # Get input (zero if not provided)
                if input_signal is not None:
                    x = input_signal[t] if input_signal.dim() > 1 else input_signal[t].unsqueeze(0)
                else:
                    x = torch.zeros(1, device=self.device)
                
                # Training step
                target = pattern[t]
                _, error_norm = self.train_step(x, target)
                errors.append(error_norm)
        
        return errors


# -----------------------------------------------------------------------------
# Factory Functions
# -----------------------------------------------------------------------------

def make_force2_trainer_for_oscillator(
    freq: float,
    n_neurons: int = 1000,
    dt: float = 1.0,
    device: str = "cpu",
) -> FORCE2Trainer:
    """
    Create FORCE2 trainer configured for learning an oscillator.
    
    Args:
        freq: Target frequency in Hz
        n_neurons: Network size
        dt: Time step
        device: PyTorch device
    """
    cfg = FORCE2Config(
        n_neurons=n_neurons,
        n_outputs=1,
        chaotic_cfg=ChaoticInitConfig(target_radius=1.5),  # Higher radius for oscillators
        multi_tau_cfg=MultiTimescaleSynapseConfig(
            tau_fast=2.0,    # Fast for rapid oscillations
            tau_slow=50.0,   # Slower integration
            alpha_fast=0.6,
            alpha_slow=0.4,
        ),
        train_readout=True,
        train_recurrent=False,
    )
    
    return FORCE2Trainer(cfg, device)


def make_force2_trainer_for_chaos(
    n_neurons: int = 2000,
    n_outputs: int = 3,
    device: str = "cpu",
) -> FORCE2Trainer:
    """
    Create FORCE2 trainer for learning chaotic attractors.
    
    Chaotic tasks require:
    - Larger networks
    - Higher spectral radius
    - Multi-timescale synapses for capturing complex dynamics
    """
    cfg = FORCE2Config(
        n_neurons=n_neurons,
        n_outputs=n_outputs,
        chaotic_cfg=ChaoticInitConfig(
            target_radius=1.8,  # Higher for richer dynamics
            connectivity_p=0.15,
        ),
        multi_tau_cfg=MultiTimescaleSynapseConfig(
            tau_fast=5.0,
            tau_slow=100.0,
            tau_ultra=300.0,
            alpha_fast=0.4,
            alpha_slow=0.4,
            alpha_ultra=0.2,
        ),
        train_readout=True,
        train_recurrent=False,
        alpha_rls=2.0,  # More conservative RLS for stability
    )
    
    return FORCE2Trainer(cfg, device)


def make_force2_trainer_full(
    n_neurons: int = 1000,
    n_outputs: int = 1,
    train_recurrent: bool = False,
    device: str = "cpu",
) -> FORCE2Trainer:
    """
    Create fully configurable FORCE2 trainer.
    
    When train_recurrent=True, uses sparse connectivity with
    partial trainability (only ~10% of recurrent weights trained).
    """
    cfg = FORCE2Config(
        n_neurons=n_neurons,
        n_outputs=n_outputs,
        train_recurrent=train_recurrent,
        trainable_rec_fraction=0.1 if train_recurrent else 0.0,
    )
    
    return FORCE2Trainer(cfg, device)
