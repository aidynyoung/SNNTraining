"""
Bin Width Adapter
=================
Ring buffer-based adapter for receiving high-frequency raw input
while maintaining the network's trained resolution.

Accumulates spikes or raw signals and bins them at the target rate.
"""

import torch
from typing import Optional, Iterator
from collections import deque


class BinWidthAdapter:
    """
    Accumulates high-frequency input and bins to target network resolution.
    
    Allows the network to run at its trained bin width (e.g., 50ms) while
    receiving input at higher frequencies (e.g., 1kHz for UAV sensors).
    """
    
    def __init__(
        self,
        input_dim: int,
        source_bin_ms: float,      # Source sampling rate in ms
        target_bin_ms: float,      # Target network bin width in ms
        ring_buffer_size: int = 1000,
        device: Optional[str] = None
    ):
        """
        Args:
            input_dim: Input dimension
            source_bin_ms: Source sampling period in milliseconds
            target_bin_ms: Target bin width in milliseconds
            ring_buffer_size: Size of ring buffer for accumulation
            device: PyTorch device
        """
        self.input_dim = input_dim
        self.source_bin_ms = source_bin_ms
        self.target_bin_ms = target_bin_ms
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Compute accumulation ratio
        self.accumulation_ratio = int(target_bin_ms / source_bin_ms)
        if self.accumulation_ratio < 1:
            raise ValueError(f"Target bin ({target_bin_ms}ms) must be >= source bin ({source_bin_ms}ms)")
        
        # Ring buffer for accumulating inputs
        self.ring_buffer = deque(maxlen=ring_buffer_size)
        self.accumulated_count = 0
        
        # Current accumulation buffer
        self.current_accumulation = torch.zeros(input_dim, device=self.device)
        
    def add(self, input_vector: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Add a high-frequency input sample.
        
        Args:
            input_vector: Input sample (input_dim,)
            
        Returns:
            Binned output when target bin is complete, else None
        """
        input_vector = input_vector.to(self.device)
        
        # Accumulate
        self.current_accumulation += input_vector
        self.accumulated_count += 1
        
        # Check if target bin is complete
        if self.accumulated_count >= self.accumulation_ratio:
            # Return binned result
            result = self.current_accumulation.clone()
            
            # Store in ring buffer (for potential re-binning)
            self.ring_buffer.append(result.cpu())
            
            # Reset accumulator
            self.current_accumulation.zero_()
            self.accumulated_count = 0
            
            return result
        
        return None
    
    def get_binned_stream(self, stream: Iterator[torch.Tensor]) -> Iterator[torch.Tensor]:
        """
        Wrap an iterator to yield binned outputs.
        
        Args:
            stream: Iterator yielding input vectors
            
        Yields:
            Binned outputs at target rate
        """
        for sample in stream:
            result = self.add(sample)
            if result is not None:
                yield result
    
    def get_buffered_data(self, n_bins: int) -> Optional[torch.Tensor]:
        """
        Retrieve last n bins from ring buffer.
        
        Args:
            n_bins: Number of recent bins to retrieve
            
        Returns:
            Tensor of shape (n_bins, input_dim) or None if insufficient data
        """
        if len(self.ring_buffer) < n_bins:
            return None
        
        recent = list(self.ring_buffer)[-n_bins:]
        return torch.stack(recent).to(self.device)
    
    def reset(self):
        """Reset adapter state."""
        self.ring_buffer.clear()
        self.current_accumulation.zero_()
        self.accumulated_count = 0
    
    @property
    def effective_bin_width_ms(self) -> float:
        """Actual bin width being used."""
        return self.target_bin_ms


class AdaptiveBinAdapter(BinWidthAdapter):
    """
    Bin adapter with adaptive bin width based on signal activity.
    
    Dynamically adjusts effective bin width for burst detection.
    """
    
    def __init__(
        self,
        input_dim: int,
        source_bin_ms: float,
        target_bin_ms: float,
        min_bin_ms: float = 10.0,
        max_bin_ms: float = 100.0,
        activity_threshold: float = 5.0,
        device: Optional[str] = None
    ):
        super().__init__(input_dim, source_bin_ms, target_bin_ms, device=device)
        self.min_bin_ms = min_bin_ms
        self.max_bin_ms = max_bin_ms
        self.activity_threshold = activity_threshold
        
        # Activity tracking
        self.recent_activity = deque(maxlen=10)
        self.current_target_ratio = self.accumulation_ratio
        
    def add(self, input_vector: torch.Tensor) -> Optional[torch.Tensor]:
        """Add sample with adaptive bin width."""
        input_vector = input_vector.to(self.device)
        
        # Track activity level
        activity = input_vector.abs().sum().item()
        self.recent_activity.append(activity)
        
        # Adapt bin width based on activity
        if len(self.recent_activity) >= 5:
            mean_activity = sum(self.recent_activity) / len(self.recent_activity)
            
            if mean_activity > self.activity_threshold:
                # High activity: shorten bin width for finer temporal resolution
                target_ms = max(self.min_bin_ms, self.target_bin_ms * 0.5)
            else:
                # Low activity: lengthen bin width for efficiency
                target_ms = min(self.max_bin_ms, self.target_bin_ms * 1.5)
            
            self.current_target_ratio = int(target_ms / self.source_bin_ms)
        
        # Accumulate with adapted ratio
        self.current_accumulation += input_vector
        self.accumulated_count += 1
        
        if self.accumulated_count >= self.current_target_ratio:
            result = self.current_accumulation.clone()
            self.ring_buffer.append(result.cpu())
            self.current_accumulation.zero_()
            self.accumulated_count = 0
            return result
        
        return None
