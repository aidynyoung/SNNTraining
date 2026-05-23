"""
force_online.py
===============
FORCE (First-Order Reduced and Controlled Error) and related online learning
methods for recurrent SNNs.

Implements recursive least squares (RLS) style online learning that enables
real-time training of recurrent networks with linear memory complexity.

Key algorithms:
1. FORCE: Sussillo & Abbott 2009 adaptation for SNNs
2. Online RLS: Recursive weight updates with exponential forgetting
3. Linear-memory online learning: Constant memory per weight

These methods are particularly effective for:
- Real-time BCI decoding
- Fast adaptation to new tasks
- Streaming data with concept drift

References
----------
- FORCE: https://www.nature.com/articles/s41467-017-01827-3
- Linear-Memory Online Learning: https://www.nature.com/articles/s41467-026-68453-w
- Sussillo & Abbott 2009: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2756108/
"""

from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple, Dict


@dataclass
class FORCEConfig:
    """Configuration for FORCE online learning."""
    # RLS parameters
    alpha_rls: float = 1.0              # Initial P matrix diagonal value (inverse covariance)
    forgetting_factor: float = 0.9995  # Lambda: close to 1.0 for slow forgetting

    # Learning rates (used when not in pure RLS mode)
    lr_readout: float = 2e-3
    lr_recurrent: float = 5e-5

    # Algorithm selection
    mode: str = "rls_full"               # "rls_full", "rls_readout_only", "online_gradient"

    # Regularization
    regularization: float = 1e-6         # L2 regularization for stability

    # Sparsity
    target_sparsity: float = 0.1         # Target connection sparsity
    enforce_sparsity: bool = False       # Whether to enforce sparse connectivity

    # Adaptive forgetting and skip logic
    adaptive_forgetting: bool = False     # Enable adaptive forgetting based on error variance
    min_alpha_rls: float = 0.5            # Minimum alpha (inverse covariance scale)
    max_alpha_rls: float = 2.0            # Maximum alpha
    skip_below_error: float = 0.001       # Skip RLS update when error norm below this
    error_window: int = 100               # Window for computing error variance


class RecursiveLeastSquares:
    """
    Recursive Least Squares (RLS) online learner.
    
    Maintains an inverse correlation matrix P and updates weights recursively:
        P(t) = (1/λ) * [P(t-1) - g(t) * z(t)^T * P(t-1)]
        W(t) = W(t-1) + g(t) * error(t)
        
    where g(t) is the gain vector and λ is the forgetting factor.
    
    Memory: O(n^2) for P matrix, where n is the number of weights.
    """
    
    def __init__(
        self,
        n_out: int,
        n_in: int,
        alpha: float = 1.0,
        forgetting_factor: float = 0.9995,
        regularization: float = 1e-6,
    ):
        self.n_out = n_out
        self.n_in = n_in
        self.alpha = alpha
        self.forgetting_factor = forgetting_factor
        self.reg = regularization
        
        # Inverse correlation matrix P: (n_out * n_in, n_out * n_in)
        # For efficiency, we use block-diagonal approximation per output neuron
        # Each output neuron has its own P matrix of shape (n_in, n_in)
        self.P = [alpha * torch.eye(n_in) for _ in range(n_out)]
        
    def update(
        self,
        W: torch.Tensor,       # (n_out, n_in) - current weights (modified in-place)
        z: torch.Tensor,       # (batch, n_in) or (n_in,) - input features
        error: torch.Tensor,   # (batch, n_out) or (n_out,) - prediction error
    ) -> torch.Tensor:
        """
        Update weights using RLS.
        
        Args:
            W: Weight matrix to update (modified in-place)
            z: Input features (pre-synaptic activity)
            error: Prediction error (target - prediction)
            
        Returns:
            delta_W: Weight change for logging/analysis
        """
        # Ensure batch dimensions
        if z.dim() == 1:
            z = z.unsqueeze(0)
        if error.dim() == 1:
            error = error.unsqueeze(0)
        
        batch_size = z.shape[0]
        delta_W = torch.zeros_like(W)
        
        # RLS update per output neuron
        for k in range(self.n_out):
            # Collect inputs and errors for this output
            z_k = z  # (batch, n_in)
            e_k = error[:, k]  # (batch,)
            
            # Mean over batch for online learning
            z_mean = z_k.mean(0)  # (n_in,)
            e_mean = e_k.mean()   # scalar
            
            # RLS gain: g = P @ z / (λ + z^T @ P @ z)
            P_k = self.P[k]  # (n_in, n_in)
            
            # Denominator: λ + z^T @ P @ z
            denom = self.forgetting_factor + z_mean @ P_k @ z_mean + self.reg
            
            # Gain vector
            g = (P_k @ z_mean) / denom  # (n_in,)
            
            # Update P matrix
            # P_new = (1/λ) * (P - g @ z^T @ P)
            self.P[k] = (P_k - torch.outer(g, z_mean @ P_k)) / self.forgetting_factor
            
            # Update weights for this output
            delta_W[k] = g * e_mean
            W[k] = W[k] + delta_W[k]
        
        return delta_W
    
    def reset(self):
        """Reset P matrices (e.g., for new task)."""
        self.P = [self.alpha * torch.eye(self.n_in) for _ in range(self.n_out)]


class LinearMemoryOnlineLearner:
    """
    Linear-memory online learning for recurrent SNNs.
    
    Based on "Linear-Memory Online Learning for Spiking Neural Networks" (Nature 2026).
    
    Key insight: Instead of storing all past activations (O(T) memory),
    maintain sufficient statistics that grow as O(P) with parameters,
    not O(T) with sequence length.
    
    Uses covariance matrix approximation for efficient online updates.
    """
    
    def __init__(
        self,
        weight_shape: Tuple[int, int],
        learning_rate: float = 5e-5,
        gamma: float = 0.99,        # Covariance decay factor
    ):
        self.n_out, self.n_in = weight_shape
        self.lr = learning_rate
        self.gamma = gamma
        
        # Sufficient statistics (linear memory)
        # C: running covariance of pre-synaptic activity
        # b: running correlation of pre with error signal
        self.register_buffer("C", torch.zeros(self.n_in, self.n_in))
        self.register_buffer("b", torch.zeros(self.n_out, self.n_in))
        
        # Running mean for centered covariance
        self.register_buffer("z_mean", torch.zeros(self.n_in))
        self.register_buffer("count", torch.tensor(0.0))
        
    def update_statistics(
        self,
        z: torch.Tensor,       # (batch, n_in) - pre-synaptic activity
        error: torch.Tensor,   # (batch, n_out) - error signal
    ):
        """
        Update sufficient statistics with new data.
        
        C_new = γ*C + (1-γ)*z^T @ z
        b_new = γ*b + (1-γ)*error^T @ z
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)
        if error.dim() == 1:
            error = error.unsqueeze(0)
        
        # Update count and mean
        batch_size = z.shape[0]
        self.count += batch_size
        
        # Centered update
        z_batch_mean = z.mean(0)
        self.z_mean = self.gamma * self.z_mean + (1 - self.gamma) * z_batch_mean
        z_centered = z - self.z_mean
        
        # Update covariance: C = γ*C + (1-γ)*z^T @ z / batch
        C_update = (z_centered.T @ z_centered) / batch_size
        self.C = self.gamma * self.C + (1 - self.gamma) * C_update
        
        # Update correlation: b = γ*b + (1-γ)*error^T @ z / batch
        b_update = (error.T @ z_centered) / batch_size  # (n_out, n_in)
        self.b = self.gamma * self.b + (1 - self.gamma) * b_update
    
    def compute_update(self) -> torch.Tensor:
        """
        Compute weight update from sufficient statistics.
        
        Uses pseudo-inverse of covariance for optimal update direction.
        
        ΔW = lr * b @ C^+
        """
        # Regularized pseudo-inverse
        C_reg = self.C + 1e-4 * torch.eye(self.n_in, device=self.C.device)
        
        # Solve C^T @ ΔW^T = b^T for each output
        # Or equivalently: ΔW = b @ C^{-1}
        try:
            C_inv = torch.linalg.inv(C_reg)
            delta_W = self.lr * self.b @ C_inv
        except:
            # Fallback to simple gradient if inversion fails
            delta_W = self.lr * self.b
        
        return delta_W
    
    def reset(self):
        """Reset sufficient statistics."""
        self.C.zero_()
        self.b.zero_()
        self.z_mean.zero_()
        self.count.zero_()


class FORCETrainer:
    """
    FORCE-style online trainer for recurrent SNNs.
    
    Combines RLS for readout weights with optional online learning
    for recurrent weights. Designed for real-time BCI decoding.
    
    Supports three modes:
    - "rls_full": RLS for both readout and recurrent weights
    - "rls_readout_only": RLS only for readout (faster, stable)
    - "online_gradient": Traditional online gradient with eligibility traces
    """
    
    def __init__(
        self,
        rsnn: nn.Module,
        readout: nn.Module,
        cfg: FORCEConfig,
    ):
        self.rsnn = rsnn
        self.readout = readout
        self.cfg = cfg
        
        # Get dimensions
        self.hidden_size = getattr(rsnn, 'hidden_size', rsnn.W_rec.shape[0])
        self.input_size = getattr(rsnn, 'input_size', rsnn.W_in.shape[1])
        
        if hasattr(readout, 'W'):
            self.output_size = readout.W.shape[0]
        else:
            self.output_size = readout.out_features if hasattr(readout, 'out_features') else 2
        
        # Initialize RLS for readout
        if cfg.mode in ["rls_full", "rls_readout_only"]:
            self.rls_readout = RecursiveLeastSquares(
                n_out=self.output_size,
                n_in=self.hidden_size,
                alpha=cfg.alpha_rls,
                forgetting_factor=cfg.forgetting_factor,
                regularization=cfg.regularization,
            )
            
            # RLS for recurrent weights (if full mode)
            if cfg.mode == "rls_full":
                self.rls_rec = RecursiveLeastSquares(
                    n_out=self.hidden_size,
                    n_in=self.hidden_size,
                    alpha=cfg.alpha_rls,
                    forgetting_factor=cfg.forgetting_factor,
                    regularization=cfg.regularization,
                )
                self.rls_in = RecursiveLeastSquares(
                    n_out=self.hidden_size,
                    n_in=self.input_size,
                    alpha=cfg.alpha_rls,
                    forgetting_factor=cfg.forgetting_factor,
                    regularization=cfg.regularization,
                )
            else:
                self.rls_rec = None
                self.rls_in = None
        
        # Linear memory learner for recurrent weights (if gradient mode)
        if cfg.mode == "online_gradient":
            self.lm_rec = LinearMemoryOnlineLearner(
                weight_shape=(self.hidden_size, self.hidden_size),
                learning_rate=cfg.lr_recurrent,
            )
            self.lm_in = LinearMemoryOnlineLearner(
                weight_shape=(self.hidden_size, self.input_size),
                learning_rate=cfg.lr_recurrent,
            )
        else:
            self.lm_rec = None
            self.lm_in = None
        
        # Store previous states for recurrent updates
        self.prev_spikes = None
        
        # Error history for adaptive forgetting
        self.error_history = []
        self.step_count = 0
        
    def step(
        self,
        x: torch.Tensor,
        target: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        One training step using FORCE/RLS.
        
        Args:
            x: Input (batch, input_size)
            target: Target (batch, output_size) or None
            
        Returns:
            y_pred: Prediction (batch, output_size)
            error: Error signal (batch, output_size)
        """
        # Forward pass
        spikes = self.rsnn(x)
        y_pred = self.readout(spikes)
        
        # Compute error
        if target is not None:
            error = target - y_pred
        else:
            error = torch.zeros_like(y_pred)
        
        # Update weights based on mode
        with torch.no_grad():
            self._update_weights(x, spikes, error)
        
        # Store for next step
        self.prev_spikes = spikes.detach()
        
        return y_pred, error
    
    def _update_weights(
        self,
        x: torch.Tensor,
        spikes: torch.Tensor,
        error: torch.Tensor,
    ):
        """Update all weights based on selected mode."""
        self.step_count += 1
        
        # Track error for adaptive forgetting
        error_norm = error.norm().item()
        self.error_history.append(error_norm)
        if len(self.error_history) > self.cfg.error_window:
            self.error_history.pop(0)
        
        # Skip RLS update if error is negligible (saves O(n²) computation)
        if error_norm < self.cfg.skip_below_error:
            return
        
        # Adaptive forgetting: adjust alpha_rls based on error variance
        if self.cfg.adaptive_forgetting and len(self.error_history) >= 10:
            error_variance = torch.tensor(self.error_history).var().item()
            
            # High variance = unstable → increase alpha (more regularization)
            # Low variance = stable → decrease alpha (faster adaptation)
            if error_variance > 1.0:
                new_alpha = min(self.cfg.max_alpha_rls, self.cfg.alpha_rls * 1.1)
            elif error_variance < 0.1:
                new_alpha = max(self.cfg.min_alpha_rls, self.cfg.alpha_rls * 0.95)
            else:
                new_alpha = self.cfg.alpha_rls
            
            # Update alpha in all RLS learners
            if hasattr(self, 'rls_readout'):
                self.rls_readout.alpha = new_alpha
            if hasattr(self, 'rls_rec') and self.rls_rec:
                self.rls_rec.alpha = new_alpha
            if hasattr(self, 'rls_in') and self.rls_in:
                self.rls_in.alpha = new_alpha
        
        # Readout update (always RLS if available)
        if hasattr(self.readout, 'W'):
            self.rls_readout.update(self.readout.W, spikes, error)
        
        # Recurrent updates based on mode
        if self.cfg.mode == "rls_full":
            # RLS for recurrent weights
            if self.prev_spikes is not None:
                self.rls_rec.update(self.rsnn.W_rec, self.prev_spikes, error)
            self.rls_in.update(self.rsnn.W_in, x, error)
            
        elif self.cfg.mode == "online_gradient":
            # Linear-memory online learning for recurrent
            if self.prev_spikes is not None:
                self.lm_rec.update_statistics(self.prev_spikes, error)
                dW_rec = self.lm_rec.compute_update()
                self.rsnn.W_rec += dW_rec
            
            self.lm_in.update_statistics(x, error)
            dW_in = self.lm_in.compute_update()
            self.rsnn.W_in += dW_in
    
    def reset(self):
        """Reset all learners (between episodes/tasks)."""
        if hasattr(self, 'rls_readout'):
            self.rls_readout.reset()
        if hasattr(self, 'rls_rec') and self.rls_rec:
            self.rls_rec.reset()
        if hasattr(self, 'rls_in') and self.rls_in:
            self.rls_in.reset()
        if hasattr(self, 'lm_rec') and self.lm_rec:
            self.lm_rec.reset()
        if hasattr(self, 'lm_in') and self.lm_in:
            self.lm_in.reset()
        self.prev_spikes = None
    
    def get_stats(self) -> Dict[str, float]:
        """Get training statistics."""
        stats = {}
        if hasattr(self, 'rls_readout'):
            # Trace of P matrix as measure of uncertainty
            stats['readout_uncertainty'] = sum(p.trace().item() for p in self.rls_readout.P) / len(self.rls_readout.P)
        return stats


# ---------------------------------------------------------------------------
# Convenience factory functions
# ---------------------------------------------------------------------------

def make_force_trainer(
    rsnn: nn.Module,
    readout: nn.Module,
    mode: str = "rls_readout_only",
    alpha_rls: float = 1.0,
    forgetting_factor: float = 0.9995,
    lr_recurrent: float = 5e-5,
) -> FORCETrainer:
    """
    Factory function for FORCE trainer.
    
    Recommended modes:
    - "rls_readout_only": Fast, stable, good for BCI (default)
    - "rls_full": Full RLS including recurrent weights (slower but powerful)
    - "online_gradient": Linear-memory gradient-based (resource constrained)
    """
    cfg = FORCEConfig(
        mode=mode,
        alpha_rls=alpha_rls,
        forgetting_factor=forgetting_factor,
        lr_recurrent=lr_recurrent,
    )
    return FORCETrainer(rsnn, readout, cfg)
