"""
Time-To-First-Spike (TTFS) Encoder
==================================
Temporal coding: stronger values → earlier spikes.

Ideal for: low-latency encoding where signal magnitude 
           should be represented by spike timing.
"""

import torch
from typing import Optional, List


class TTFSEncoder:
    """
    Time-to-first-spike encoder.
    
    Maps signal magnitude to spike time within a fixed window.
    Stronger signals → earlier spikes (lower timesteps).
    """
    
    def __init__(
        self,
        n_channels: int,
        time_window: int = 20,
        min_spike_time: int = 0,
        max_spike_time: Optional[int] = None,
        device: Optional[str] = None
    ):
        """
        Args:
            n_channels: Number of input channels
            time_window: Total encoding window in timesteps
            min_spike_time: Earliest possible spike time
            max_spike_time: Latest possible spike time (default: time_window-1)
            device: PyTorch device
        """
        self.n_channels = n_channels
        self.time_window = time_window
        self.min_spike_time = min_spike_time
        self.max_spike_time = max_spike_time or (time_window - 1)
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Track pending spikes and their scheduled times
        self.pending_spikes: List[torch.Tensor] = []
        self.current_timestep = 0
        
    def encode(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Encode signal into spike schedule, return current timestep's spikes.
        
        Args:
            signal: Current signal values (n_channels,)
            
        Returns:
            Spike counts for current timestep (n_channels,)
        """
        signal = signal.to(self.device)
        
        # Normalize signal to [0, 1]
        signal_min = signal.min()
        signal_range = (signal.max() - signal_min).clamp(min=1e-6)
        normalized = (signal - signal_min) / signal_range
        
        # Map to spike times: stronger = earlier (inverted)
        # 1.0 (max) -> min_spike_time, 0.0 (min) -> max_spike_time
        spike_times = self.max_spike_time - \
                      (normalized * (self.max_spike_time - self.min_spike_time)).long()
        spike_times = spike_times.clamp(self.min_spike_time, self.max_spike_time)
        
        # Schedule spikes
        spikes_now = torch.zeros(self.n_channels, device=self.device)
        for i in range(self.n_channels):
            if spike_times[i] == self.current_timestep % self.time_window:
                spikes_now[i] = 1.0
        
        self.current_timestep += 1
        return spikes_now
    
    def reset(self):
        """Reset encoder state."""
        self.pending_spikes = []
        self.current_timestep = 0


class PopulationTTFS:
    """
    Population TTFS encoding with channel tiling.
    
    Each input channel is represented by a population of neurons
    with different preferred values, enabling richer temporal codes.
    """
    
    def __init__(
        self,
        n_channels: int,
        neurons_per_channel: int = 10,
        time_window: int = 20,
        value_range: tuple = (-1.0, 1.0),
        device: Optional[str] = None
    ):
        """
        Args:
            n_channels: Number of input channels
            neurons_per_channel: Population size per channel
            time_window: Encoding window
            value_range: (min, max) of expected signal values
            device: PyTorch device
        """
        self.n_channels = n_channels
        self.neurons_per_channel = neurons_per_channel
        self.time_window = time_window
        self.value_range = value_range
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Preferred values for each neuron in population (Gaussian receptive fields)
        min_val, max_val = value_range
        self.preferred = torch.linspace(min_val, max_val, neurons_per_channel, device=self.device)
        
        # Receptive field width
        self.rf_width = (max_val - min_val) / (neurons_per_channel - 1) if neurons_per_channel > 1 else 1.0
        
        self.current_timestep = 0
        
    def encode(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Encode into population spike pattern.
        
        Returns:
            Spikes of shape (n_channels * neurons_per_channel,)
        """
        signal = signal.to(self.device)
        
        # Compute response of each population neuron
        # Gaussian response based on distance to preferred value
        expanded_signal = signal.unsqueeze(1)  # (n_channels, 1)
        expanded_preferred = self.preferred.unsqueeze(0)  # (1, neurons_per_channel)
        
        distance = (expanded_signal - expanded_preferred).abs()
        responses = torch.exp(-(distance ** 2) / (2 * self.rf_width ** 2))
        
        # Map response strength to spike time (stronger response = earlier spike)
        spike_times = ((1.0 - responses) * (self.time_window - 1)).long()
        
        # Determine current spikes
        spikes = (spike_times == (self.current_timestep % self.time_window)).float()
        
        # Flatten to (n_channels * neurons_per_channel,)
        spikes_flat = spikes.reshape(-1)
        
        self.current_timestep += 1
        return spikes_flat
    
    def reset(self):
        """Reset encoder state."""
        self.current_timestep = 0
    
    @property
    def output_size(self) -> int:
        """Total output dimension after population encoding."""
        return self.n_channels * self.neurons_per_channel
