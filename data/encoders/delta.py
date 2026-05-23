"""
Delta Encoder
=============
Fires when signal changes by threshold (event-based encoding).

Ideal for: vibration/acoustic sensors (fault detection), 
           current/voltage signals (motor health monitoring)
"""

import torch
from typing import Optional


class DeltaEncoder:
    """
    Delta encoder: fires when |signal - last_signal| > threshold.
    
    Produces spike counts proportional to magnitude of change.
    """
    
    def __init__(
        self,
        n_channels: int,
        threshold: float = 0.1,
        scale: float = 1.0,
        device: Optional[str] = None
    ):
        """
        Args:
            n_channels: Number of input signal channels
            threshold: Minimum delta to trigger spike (in signal units)
            scale: Scaling factor for spike count magnitude
            device: PyTorch device
        """
        self.n_channels = n_channels
        self.threshold = threshold
        self.scale = scale
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # State: last observed value per channel
        self.last_value = torch.zeros(n_channels, device=self.device)
        self.initialized = False
        
    def encode(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Encode signal delta into spike counts.
        
        Args:
            signal: Tensor of shape (n_channels,) with current signal values
            
        Returns:
            Spike counts of shape (n_channels,)
        """
        signal = signal.to(self.device)
        
        # First call: initialize and return zeros
        if not self.initialized:
            self.last_value = signal.clone()
            self.initialized = True
            return torch.zeros_like(signal)
        
        # Compute delta
        delta = signal - self.last_value
        abs_delta = delta.abs()
        
        # Fire if delta exceeds threshold
        spike_counts = (abs_delta / self.threshold).floor().clamp(min=0, max=10) * self.scale
        spike_counts = spike_counts * (abs_delta >= self.threshold).float()
        
        # Update state
        self.last_value = signal.clone()
        
        return spike_counts
    
    def reset(self):
        """Reset encoder state."""
        self.last_value.zero_()
        self.initialized = False


class AdaptiveDeltaEncoder(DeltaEncoder):
    """
    Delta encoder with adaptive threshold based on recent signal variance.
    
    Automatically adjusts threshold to handle varying noise levels.
    """
    
    def __init__(
        self,
        n_channels: int,
        base_threshold: float = 0.1,
        adaptation_rate: float = 0.01,
        window_size: int = 100,
        device: Optional[str] = None
    ):
        super().__init__(n_channels, base_threshold, device=device)
        self.base_threshold = base_threshold
        self.adaptation_rate = adaptation_rate
        self.window_size = window_size
        
        # Running statistics for adaptive threshold
        self.delta_history = []
        self.current_threshold = torch.full((n_channels,), base_threshold, device=self.device)
        
    def encode(self, signal: torch.Tensor) -> torch.Tensor:
        """Encode with adaptive threshold."""
        signal = signal.to(self.device)
        
        if not self.initialized:
            self.last_value = signal.clone()
            self.initialized = True
            return torch.zeros_like(signal)
        
        delta = signal - self.last_value
        abs_delta = delta.abs()
        
        # Update history for adaptation
        self.delta_history.append(abs_delta.cpu())
        if len(self.delta_history) > self.window_size:
            self.delta_history.pop(0)
        
        # Adapt threshold based on recent delta variance
        if len(self.delta_history) >= 10:
            recent_deltas = torch.stack(self.delta_history[-10:])
            variance = recent_deltas.var(dim=0).to(self.device)
            # Threshold = base + k * sqrt(variance)
            target_threshold = self.base_threshold + 2.0 * variance.sqrt()
            self.current_threshold = (1 - self.adaptation_rate) * self.current_threshold + \
                                   self.adaptation_rate * target_threshold
        
        # Fire with adaptive threshold
        spike_counts = (abs_delta / self.current_threshold).floor().clamp(min=0, max=10)
        spike_counts = spike_counts * (abs_delta >= self.current_threshold).float()
        
        self.last_value = signal.clone()
        return spike_counts
