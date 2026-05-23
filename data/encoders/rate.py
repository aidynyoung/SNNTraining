"""
Rate Encoder
============
Sliding window spike count encoding.

Ideal for: continuous sensor monitoring where sustained 
           elevated values should produce sustained firing.
"""

import torch
from typing import Optional
from collections import deque


class RateEncoder:
    """
    Rate encoder: sliding window counts → spike rate.
    
    Accumulates signal over a window and emits spike counts
    proportional to accumulated magnitude.
    """
    
    def __init__(
        self,
        n_channels: int,
        window_size: int = 10,
        threshold: float = 0.5,
        max_spikes: int = 5,
        device: Optional[str] = None
    ):
        """
        Args:
            n_channels: Number of input channels
            window_size: Sliding window size in timesteps
            threshold: Signal magnitude required per spike
            max_spikes: Maximum spikes per timestep per channel
            device: PyTorch device
        """
        self.n_channels = n_channels
        self.window_size = window_size
        self.threshold = threshold
        self.max_spikes = max_spikes
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Ring buffer for sliding window
        self.buffer = deque(maxlen=window_size)
        
    def encode(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Encode signal into spike counts via sliding window.
        
        Args:
            signal: Current signal values (n_channels,)
            
        Returns:
            Spike counts (n_channels,)
        """
        signal = signal.to(self.device)
        abs_signal = signal.abs()
        
        # Add to buffer
        self.buffer.append(abs_signal)
        
        # Compute accumulated magnitude in window
        if len(self.buffer) > 0:
            window_sum = torch.stack(list(self.buffer)).sum(dim=0)
            # Spike count proportional to accumulated magnitude
            spike_counts = (window_sum / (self.threshold * self.window_size)).floor()
            spike_counts = spike_counts.clamp(min=0, max=self.max_spikes)
        else:
            spike_counts = torch.zeros_like(signal)
        
        return spike_counts
    
    def reset(self):
        """Reset buffer."""
        self.buffer.clear()


class ThresholdCrossingEncoder(RateEncoder):
    """
    Fires when signal crosses positive/negative thresholds.
    
    Binary spike encoding for threshold-crossing events.
    """
    
    def __init__(
        self,
        n_channels: int,
        positive_threshold: float = 0.5,
        negative_threshold: float = -0.5,
        device: Optional[str] = None
    ):
        super().__init__(n_channels, window_size=1, device=device)
        self.positive_threshold = positive_threshold
        self.negative_threshold = negative_threshold
        
        self.last_signal = None
        
    def encode(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Encode threshold crossings as spikes.
        
        Returns +1 for positive crossing, -1 for negative crossing (as spike count).
        """
        signal = signal.to(self.device)
        
        if self.last_signal is None:
            self.last_signal = signal.clone()
            return torch.zeros_like(signal)
        
        # Detect crossings
        pos_crossing = (signal >= self.positive_threshold) & \
                       (self.last_signal < self.positive_threshold)
        neg_crossing = (signal <= self.negative_threshold) & \
                       (self.last_signal > self.negative_threshold)
        
        # Encode as spike count (positive = +1, negative = -1 mapped to 1)
        spikes = pos_crossing.float() + neg_crossing.float()
        
        self.last_signal = signal.clone()
        return spikes
