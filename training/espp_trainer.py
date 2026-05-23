"""
espp_trainer.py
===============
EchoSpike Predictive Plasticity (ESPP) trainer.

Based on Graf et al. 2024: "EchoSpike Predictive Plasticity: 
An Online Local Learning Rule for Spiking Neural Networks"

Key features:
- Self-supervised local learning (no backpropagation)
- Echo-based prediction: uses previous sample's activity as prediction
- Contrastive learning: same label → similar, different label → different
- Intrinsic spike rate regularization via negative feedback loop
- Online updates at every timestep with O(1) memory

The core idea:
1. For "fixation" (same label as previous): maximize similarity between
   current activity and previous sample's echo
2. For "saccade" (different label): minimize similarity (contrastive)
3. Adaptive threshold c(y) creates intrinsic regularization
"""

from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
import numpy as np

from models.lif import LIFLayer, LIFConfig


@dataclass
class ESPPConfig:
    """Configuration for ESPP learning rule."""
    
    # Network architecture
    n_neurons: int = 1000
    n_outputs: int = 10  # Number of classes
    
    # LIF parameters
    lif_tau: float = 20.0
    lif_v_th: float = 1.0
    lif_refractory: int = 2
    
    # ESPP-specific thresholds (adaptive)
    # c(y=1) for fixation (same label) - positive threshold
    c_fixation: float = 0.5
    # c(y=-1) for saccade (different label) - negative threshold
    c_saccade: float = -0.5
    
    # Learning rate
    lr: float = 1e-3
    
    # Surrogate gradient parameters
    surrogate_scale: float = 1.0
    
    # Spike rate target (for regularization monitoring)
    target_spike_rate: float = 0.1
    
    device: str = "cpu"


class ESPPActivityBuffer:
    """
    Buffer to store and normalize previous sample's activity (the "echo").
    
    This is the key innovation of ESPP - instead of using a separate
    prediction weight matrix, it uses the previous sample's accumulated
    spike activity as the prediction signal.
    """
    
    def __init__(self, size: int, device: str = "cpu"):
        self.size = size
        self.device = device
        
        # Accumulated spikes from previous sample
        self.s_prev = torch.zeros(size, device=device)
        self.n_tot = 0  # Total spikes in previous sample
        
        # Running accumulator for current sample
        self.s_current = torch.zeros(size, device=device)
        self.n_current = 0
        
        # Normalized previous activity (the "echo")
        self.s_bar_prev = torch.zeros(size, device=device)
    
    def accumulate(self, spikes: torch.Tensor):
        """Accumulate spikes for current sample."""
        self.s_current += spikes
        self.n_current += spikes.sum().item()
    
    def finalize_sample(self) -> torch.Tensor:
        """
        Finalize current sample and prepare for next.
        
        Returns normalized previous activity (echo) for learning.
        """
        # Store as previous
        self.s_prev = self.s_current.clone()
        self.n_tot = self.n_current
        
        # Normalize: s̄_prev = s_prev / (n_tot / N + ε)
        # where N is number of neurons
        if self.n_tot > 0:
            normalization = self.n_tot / self.size + 1e-8
            self.s_bar_prev = self.s_prev / normalization
        else:
            self.s_bar_prev.zero_()
        
        # Reset current accumulator
        self.s_current.zero_()
        self.n_current = 0
        
        return self.s_bar_prev
    
    def reset(self):
        """Reset all buffers."""
        self.s_prev.zero_()
        self.s_current.zero_()
        self.s_bar_prev.zero_()
        self.n_tot = 0
        self.n_current = 0


class ESPPTrainer(nn.Module):
    """
    EchoSpike Predictive Plasticity trainer.
    
    Implements self-supervised local learning for SNNs where:
    - Weight updates are local in time and space
    - No backpropagation through time needed
    - O(1) memory per neuron (constant, not scaling with timesteps)
    
    The learning rule:
    ΔW = -lr * ∂L/∂W
    
    where the loss L depends on:
    - Current spike activity s(t)
    - Previous sample's echo s̄_prev
    - Label comparison y ∈ {+1 (same), -1 (different)}
    """
    
    def __init__(self, cfg: Optional[ESPPConfig] = None, device: str = "cpu"):
        super().__init__()
        self.cfg = cfg or ESPPConfig()
        self.device = device
        
        n = self.cfg.n_neurons
        n_out = self.cfg.n_outputs
        
        # LIF layer for hidden neurons
        lif_cfg = LIFConfig(
            size=n,
            tau=self.cfg.lif_tau,
            v_th=self.cfg.lif_v_th,
            refractory=self.cfg.lif_refractory,
            device=device,
        )
        self.lif = LIFLayer(lif_cfg)
        
        # Input weights
        self.W_in = nn.Parameter(torch.randn(n, 1, device=device) * 0.1)
        
        # Recurrent weights (main plastic weights)
        self.W_rec = nn.Parameter(torch.randn(n, n, device=device) * 0.01)
        
        # Readout weights
        self.W_out = nn.Parameter(torch.randn(n_out, n, device=device) * 0.01)
        
        # ESPP activity buffer (the "echo")
        self.activity_buffer = ESPPActivityBuffer(n, device)
        
        # Spike traces for surrogate gradient
        self.register_buffer("spike_trace", torch.zeros(n, device=device))
        self.trace_decay = 0.9
        
        # Current sample label (for detecting saccade/fixation)
        self.prev_label: Optional[int] = None
        
        # Tracking
        self.step_count = 0
        self.sample_count = 0
        
    def surrogate_gradient(self, spikes: torch.Tensor) -> torch.Tensor:
        """
        Surrogate gradient for backpropagation through spikes.
        
        Uses simple rectangular function:
        ∂spike/∂v = 1 if |v - v_th| < scale, else 0
        """
        # Get membrane potential from LIF
        v = self.lif.v
        v_th = self.cfg.lif_v_th
        scale = self.cfg.surrogate_scale
        
        # Rectangular surrogate
        grad = ((v - v_th).abs() < scale).float()
        return grad
    
    def compute_espp_loss(
        self,
        spikes: torch.Tensor,
        s_bar_prev: torch.Tensor,
        y: int,  # +1 for fixation, -1 for saccade
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute ESPP loss and dL indicator.
        
        L = max(0, y * (s(t) · s̄_prev - c(y)))
        
        where c(y) is the adaptive threshold.
        
        Returns:
            loss: The ESPP loss value
            dL: Binary indicator (1 if loss > 0, else 0)
        """
        # Similarity score: dot product of current spikes and previous echo
        similarity = torch.dot(spikes, s_bar_prev)
        
        # Adaptive threshold based on y
        c = self.cfg.c_fixation if y == 1 else self.cfg.c_saccade
        
        # ESPP loss (hinge loss style)
        # For fixation (y=1): want similarity > c_fixation
        # For saccade (y=-1): want similarity < c_saccade
        loss_value = y * (similarity - c)
        loss = torch.clamp(loss_value, min=0.0)
        
        # dL indicator: 1 if loss is positive (need to update), 0 otherwise
        dL = (loss > 0).float()
        
        return loss, dL
    
    def compute_weight_update(
        self,
        spikes: torch.Tensor,
        s_bar_prev: torch.Tensor,
        dL: torch.Tensor,
        y: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute ESPP weight update for recurrent weights.
        
        The gradient (from paper):
        ∂L/∂W_ij = -y * dL * σ'(v_i) * (s_j - s̄_prev_j * Σs_k * W_ik / n_tot)
        
        For simplicity, we use the core ESPP update:
        ΔW ∝ -y * dL * spike_trace * (current_input - echo_feedback)
        """
        n = self.cfg.n_neurons
        
        # Surrogate gradient
        surr_grad = self.surrogate_gradient(spikes)
        
        # Current input to each neuron
        # For recurrent: input = W_rec @ spikes
        current_input = torch.matmul(self.W_rec, spikes)
        
        # Echo feedback (prediction from previous sample)
        echo_feedback = s_bar_prev
        
        # Core ESPP update
        # ΔW_ij ∝ -y * dL * surr_grad_i * (input_j - echo_j)
        # This is a simplified but effective form
        
        # Expand for outer product
        surr_grad_exp = surr_grad.unsqueeze(1)  # (n, 1)
        input_diff = (current_input - echo_feedback).unsqueeze(0)  # (1, n)
        
        # Gradient (negative of update direction)
        grad_W = -y * dL * surr_grad_exp * input_diff
        
        # Learning rate applied
        delta_W = self.cfg.lr * grad_W
        
        return delta_W, surr_grad
    
    def step(
        self,
        x: torch.Tensor,
        label: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass and ESPP update.
        
        Args:
            x: Input (batch, 1) or (1,)
            label: Current sample label (optional, for fixation/saccade detection)
            
        Returns:
            spikes: Hidden layer spikes
            output: Readout output
            loss: ESPP loss for monitoring
        """
        # Handle input
        if x.dim() == 0:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 1:
            x = x.unsqueeze(1)
        
        # Compute current
        I_in = torch.matmul(self.W_in, x.T).squeeze(-1)
        I_rec = torch.matmul(self.W_rec, self.lif.get_firing_rates(10))
        I_total = I_in + I_rec
        
        # LIF step
        spikes = self.lif.step(I_total)
        
        # Update spike trace
        self.spike_trace = self.trace_decay * self.spike_trace + spikes
        
        # Accumulate for ESPP
        self.activity_buffer.accumulate(spikes)
        
        # ESPP update (if we have previous sample)
        loss = torch.tensor(0.0, device=self.device)
        if self.prev_label is not None and label is not None:
            # Determine fixation (y=1) or saccade (y=-1)
            y = 1 if label == self.prev_label else -1
            
            # Get previous sample's echo
            s_bar_prev = self.activity_buffer.s_bar_prev
            
            # Compute ESPP loss
            loss, dL = self.compute_espp_loss(spikes, s_bar_prev, y)
            
            # Compute and apply weight update
            if dL > 0:
                delta_W, _ = self.compute_weight_update(
                    spikes, s_bar_prev, dL, y
                )
                self.W_rec.data += delta_W
        
        # Readout
        output = torch.matmul(self.W_out, spikes)
        
        self.step_count += 1
        
        return spikes, output, loss
    
    def end_sample(self, label: int):
        """
        Call at end of each sample to finalize echo buffer.
        
        Args:
            label: Label of the sample that just ended
        """
        self.activity_buffer.finalize_sample()
        self.prev_label = label
        self.sample_count += 1
    
    def reset_state(self):
        """Reset LIF and activity buffer for new sequence."""
        self.lif.reset()
        self.activity_buffer.reset()
        self.spike_trace.zero_()
        self.prev_label = None
    
    def get_stats(self) -> Dict[str, float]:
        """Get training statistics."""
        stats = {
            "step": self.step_count,
            "sample": self.sample_count,
        }
        
        # Spike rate
        rates = self.lif.get_firing_rates(window=100)
        stats["mean_spike_rate"] = rates.mean().item()
        
        # Activity buffer stats
        stats["echo_norm"] = self.activity_buffer.s_bar_prev.norm().item()
        stats["prev_spike_count"] = self.activity_buffer.n_tot
        
        return stats


# -----------------------------------------------------------------------------
# Classification layer trainers (for ESPP)
# -----------------------------------------------------------------------------

class ESPPClassifier:
    """
    Low-cost classifier for ESPP output layer.
    
    From paper Section IV: Can use either gradient descent or closed-form.
    The closed-form is particularly efficient for few-shot learning.
    """
    
    def __init__(
        self,
        n_features: int,
        n_classes: int,
        method: str = "gradient",  # "gradient" or "closed_form"
        lr: float = 1e-3,
        device: str = "cpu",
    ):
        self.n_features = n_features
        self.n_classes = n_classes
        self.method = method
        self.lr = lr
        self.device = device
        
        # Classifier weights
        self.W = torch.randn(n_classes, n_features, device=device) * 0.01
        
        # For closed-form: accumulate statistics
        self.class_means: Dict[int, torch.Tensor] = {}
        self.class_counts: Dict[int, int] = {}
    
    def gradient_update(
        self,
        features: torch.Tensor,
        label: int,
    ) -> torch.Tensor:
        """Simple gradient descent update for classifier."""
        # Softmax cross-entropy gradient
        logits = torch.matmul(self.W, features)
        
        # One-hot target
        target = torch.zeros(self.n_classes, device=self.device)
        target[label] = 1.0
        
        # Softmax
        exp_logits = torch.exp(logits - logits.max())
        probs = exp_logits / exp_logits.sum()
        
        # Gradient: (probs - target) outer features
        grad = torch.outer(probs - target, features)
        
        # Update
        self.W -= self.lr * grad
        
        return probs
    
    def closed_form_update(
        self,
        features: torch.Tensor,
        label: int,
    ):
        """
        Closed-form update (prototype-based).
        
        Each class is represented by mean feature vector.
        Classification by nearest prototype.
        """
        # Accumulate class statistics
        if label not in self.class_means:
            self.class_means[label] = torch.zeros(self.n_features, device=self.device)
            self.class_counts[label] = 0
        
        # Running mean update
        n = self.class_counts[label]
        self.class_means[label] = (n * self.class_means[label] + features) / (n + 1)
        self.class_counts[label] += 1
        
        # Set weights to class prototypes
        for i in range(self.n_classes):
            if i in self.class_means:
                self.W[i] = self.class_means[i]
    
    def predict(self, features: torch.Tensor) -> int:
        """Predict class label."""
        logits = torch.matmul(self.W, features)
        return logits.argmax().item()
    
    def update(
        self,
        features: torch.Tensor,
        label: int,
    ) -> torch.Tensor:
        """Update based on selected method."""
        if self.method == "gradient":
            return self.gradient_update(features, label)
        else:
            return self.closed_form_update(features, label)


# -----------------------------------------------------------------------------
# Factory functions
# -----------------------------------------------------------------------------

def make_espp_trainer(
    n_neurons: int = 1000,
    n_classes: int = 10,
    device: str = "cpu",
) -> ESPPTrainer:
    """Create ESPP trainer with default config."""
    cfg = ESPPConfig(
        n_neurons=n_neurons,
        n_outputs=n_classes,
        c_fixation=0.5,
        c_saccade=-0.5,
        lr=1e-3,
        device=device,
    )
    return ESPPTrainer(cfg, device)
