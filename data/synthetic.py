import torch
import numpy as np


def generate_stream(T=1000, input_size=20, noise=0.1):
    """Generate synthetic data stream for SNN training.
    
    Args:
        T: Number of timesteps
        input_size: Dimension of input features
        noise: Noise level for target generation
    """
    freq = 50.0  # Default frequency
    for t in range(T):
        x = torch.rand(input_size) * 2.0  # Scale up for stronger drive
        # Sinusoidal target with noise
        y = torch.tensor([
            torch.sin(torch.tensor(t / freq)) + noise * torch.randn(1),
            torch.cos(torch.tensor(t / freq)) + noise * torch.randn(1)
        ]).squeeze()
        yield x, y


def bci_velocity_stream(T=1000, input_size=100, noise=0.1, seed=42):
    """Generate BCI velocity-like synthetic data stream with realistic neural tuning curves.
    
    Real neural data has tuning curves where neurons fire preferentially for specific
    movement directions. This generator produces sparse, event-driven spike patterns
    that match real Indy dataset statistics.
    
    The relationship between neural activity and movement is noisy and indirect,
    making it a challenging decoding task.
    
    Args:
        T: Number of timesteps
        input_size: Dimension of input features (number of neurons)
        noise: Noise level for target generation
        seed: Random seed for reproducibility
        
    Yields:
        Tuple of (input, target) tensors - (spike_counts [input_size], velocity [2])
    """
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    
    # Generate preferred directions for each neuron (uniformly distributed)
    preferred_dirs = np.linspace(0, 2 * np.pi, input_size, endpoint=False)
    
    # Add random offset to preferred directions for more realistic heterogeneity
    dir_offset = rng.randn(input_size) * 0.3
    preferred_dirs = preferred_dirs + dir_offset
    
    # Velocity trajectory: smooth circular motion with noise
    freq = 30.0
    velocity_scale = 0.8
    
    # Tuning curve parameters - more realistic (less direct encoding)
    base_rate = 10.0      # Base firing rate (Hz)
    peak_rate = 60.0      # Peak firing rate (Hz) - reduced for less direct encoding
    spike_prob = 0.02     # Lower bin probability for sparser spikes
    
    for t in range(T):
        # Generate smooth velocity trajectory with significant noise
        phase = 2 * np.pi * t / freq
        vel_x = np.sin(phase) * velocity_scale + noise * rng.randn()
        vel_y = np.cos(phase) * velocity_scale * 0.75 + noise * rng.randn()
        
        # Current movement direction and speed
        speed = np.sqrt(vel_x**2 + vel_y**2)
        direction = np.arctan2(vel_y, vel_x)
        
        # Compute tuning curve with speed modulation
        # Real neurons often have speed-dependent firing
        cosine_response = np.maximum(0, np.cos(preferred_dirs - direction))
        speed_modulation = np.clip(speed / velocity_scale, 0, 1.5)
        firing_rates = base_rate + (peak_rate - base_rate) * cosine_response * speed_modulation
        
        # Add Poisson-like noise to firing rates
        firing_rates = firing_rates + rng.randn(input_size) * 5.0
        firing_rates = np.maximum(0, firing_rates)  # Can't have negative rates
        
        # Convert firing rates to spike probabilities per bin (5ms bin)
        # Clip to valid range [0, 1] to avoid NaN errors
        spike_probs = np.clip(firing_rates * spike_prob, 0.0, 1.0)
        
        # Generate sparse binary spikes (Bernoulli process)
        spikes = rng.binomial(1, spike_probs)
        
        # Convert to tensors
        x = torch.tensor(spikes, dtype=torch.float32)
        y = torch.tensor([vel_x, vel_y], dtype=torch.float32)
        
        yield x, y


def supply_chain_stream(T=1000, input_size=50, noise=0.05, n_outputs=2):
    """Generate supply chain-like synthetic data stream.
    
    Args:
        T: Number of timesteps
        input_size: Dimension of input features
        noise: Noise level for target generation
        
    Yields:
        Tuple of (input, target) tensors
    """
    for t in range(T):
        x = torch.rand(input_size) * 2.0
        import math
        weekly_cycle = math.sin(2 * math.pi * t / 168)
        trend = 0.01 * t
        # Generate n_outputs demand signals with phase offsets
        y = torch.tensor([
            1.0 + weekly_cycle * math.cos(i * 2 * math.pi / n_outputs)
            + trend + noise * torch.randn(1).item()
            for i in range(n_outputs)
        ])
        yield x, y
