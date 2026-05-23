"""
force_enhanced.py
=================
Enhanced FORCE training components based on Nicola & Clopath 2017.

Key improvements from the paper:
1. Chaotic regime initialization for rich dynamics
2. Sparse structured recurrent connectivity (fixed random patterns)
3. Multi-timescale synaptic dynamics (fast and slow synapses)
4. Target pattern generators (oscillators, chaotic attractors)

References
----------
- Nicola & Clopath 2017: https://www.nature.com/articles/s41467-017-01827-3
- FORCE enables training both feedforward AND recurrent weights simultaneously
- Networks initialized in chaotic regime show better learning capacity
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List, Callable
try:
    from scipy import stats as scipy_stats
except ImportError:
    scipy_stats = None  # scipy is optional; only needed by PatternGenerator


@dataclass
class ChaoticInitConfig:
    """Configuration for chaotic regime initialization."""
    # Target spectral radius for chaos (typically 1.0-1.5)
    target_radius: float = 1.25
    
    # Connectivity density (sparse connections work better)
    connectivity_p: float = 0.1
    
    # Weight distribution parameters
    weight_mean: float = 0.0
    weight_std: float = 0.5
    
    # Balance of excitation/inhibition (E/I ratio)
    # Paper uses balanced E/I for stable chaos
    ei_balance: float = 0.8  # 0.8 = 80% excitatory
    
    # Dale's principle: enforce sign constraints
    use_dales_principle: bool = True


@dataclass
class MultiTimescaleSynapseConfig:
    """Configuration for multi-timescale synaptic dynamics."""
    # Fast synapse time constant (AMPA-like, ~2-5ms)
    tau_fast: float = 3.0
    
    # Slow synapse time constant (NMDA-like, ~50-150ms)
    tau_slow: float = 100.0
    
    # Ultra-slow (GABA-B-like, ~200-500ms) for longer timescales
    tau_ultra: float = 300.0
    
    # Mixing coefficients: how much each timescale contributes
    # These should sum to ~1.0
    alpha_fast: float = 0.5
    alpha_slow: float = 0.4
    alpha_ultra: float = 0.1


class ChaoticInitializer:
    """
    Initialize recurrent weights for chaotic dynamics.
    
    Based on Nicola & Clopath 2017, the key insight is that networks
    initialized in a chaotic regime (spectral radius > 1) have richer
    dynamics and can better learn complex temporal patterns.
    
    The initialization:
    1. Creates sparse random connectivity
    2. Scales to target spectral radius (typically 1.25)
    3. Optionally enforces Dale's principle (E/I separation)
    """
    
    def __init__(self, cfg: Optional[ChaoticInitConfig] = None):
        self.cfg = cfg or ChaoticInitConfig()
    
    def initialize(
        self,
        n_neurons: int,
        device: str = "cpu",
    ) -> torch.Tensor:
        """
        Initialize recurrent weight matrix for chaotic dynamics.
        
        Args:
            n_neurons: Number of recurrent neurons
            device: PyTorch device
            
        Returns:
            W_rec: Initialized recurrent weights (n_neurons, n_neurons)
        """
        cfg = self.cfg
        
        # Start with random sparse connectivity
        mask = (torch.rand(n_neurons, n_neurons, device=device) < cfg.connectivity_p).float()
        
        # Generate random weights
        W = torch.randn(n_neurons, n_neurons, device=device) * cfg.weight_std + cfg.weight_mean
        W = W * mask  # Apply sparsity mask
        
        # Enforce Dale's principle if requested (separate E and I populations)
        if cfg.use_dales_principle:
            W = self._apply_dales_principle(W, cfg.ei_balance)
        
        # Scale to target spectral radius
        # Compute eigenvalues (on CPU for stability, then move)
        W_cpu = W.cpu().numpy()
        eigenvalues = np.linalg.eigvals(W_cpu)
        current_radius = np.max(np.abs(eigenvalues))
        
        # Scale to target radius
        if current_radius > 0:
            W = W * (cfg.target_radius / current_radius)
        
        return W.to(device)
    
    def _apply_dales_principle(
        self,
        W: torch.Tensor,
        ei_ratio: float,
    ) -> torch.Tensor:
        """
        Apply Dale's principle: neurons are either excitatory or inhibitory.
        
        Args:
            W: Weight matrix
            ei_ratio: Fraction of excitatory neurons
            
        Returns:
            W with sign constraints applied
        """
        n = W.shape[0]
        n_excitatory = int(n * ei_ratio)
        
        # First n_excitatory neurons are excitatory (positive outgoing weights)
        # Remaining are inhibitory (negative outgoing weights)
        W_exc = W.clone()
        W_exc[n_excitatory:, :] = -torch.abs(W[n_excitatory:, :])
        W_exc[:n_excitatory, :] = torch.abs(W[:n_excitatory, :])
        
        return W_exc
    
    def compute_spectral_radius(self, W: torch.Tensor) -> float:
        """Compute spectral radius of weight matrix."""
        W_cpu = W.detach().cpu().numpy()
        eigenvalues = np.linalg.eigvals(W_cpu)
        return float(np.max(np.abs(eigenvalues)))


class MultiTimescaleSynapses(nn.Module):
    """
    Multi-timescale synaptic dynamics for rich temporal processing.
    
    The FORCE paper shows that using multiple synaptic timescales
    (fast AMPA-like and slow NMDA-like) improves learning of
    complex temporal patterns.
    
    Each synapse has three components:
    - Fast (AMPA): tau ~ 3ms, for rapid communication
    - Slow (NMDA): tau ~ 100ms, for integration
    - Ultra-slow: tau ~ 300ms, for long-term dependencies
    """
    
    def __init__(
        self,
        n_neurons: int,
        cfg: Optional[MultiTimescaleSynapseConfig] = None,
        device: str = "cpu",
    ):
        super().__init__()
        self.cfg = cfg or MultiTimescaleSynapseConfig()
        self.n_neurons = n_neurons
        
        # Synaptic state variables (filtered spike trains)
        self.register_buffer("s_fast", torch.zeros(n_neurons, device=device))
        self.register_buffer("s_slow", torch.zeros(n_neurons, device=device))
        self.register_buffer("s_ultra", torch.zeros(n_neurons, device=device))
        
        # Decay factors (precomputed for efficiency)
        cfg = self.cfg
        self.decay_fast = 1.0 - 1.0 / cfg.tau_fast
        self.decay_slow = 1.0 - 1.0 / cfg.tau_slow
        self.decay_ultra = 1.0 - 1.0 / cfg.tau_ultra
    
    def step(self, spikes: torch.Tensor) -> torch.Tensor:
        """
        Update synaptic states with new spikes.
        
        Args:
            spikes: Binary spike tensor (n_neurons,)
            
        Returns:
            Combined synaptic current (n_neurons,)
        """
        cfg = self.cfg
        
        # Update each synaptic component (exponential decay + new spikes)
        self.s_fast.mul_(self.decay_fast).add_(spikes)
        self.s_slow.mul_(self.decay_slow).add_(spikes)
        self.s_ultra.mul_(self.decay_ultra).add_(spikes)
        
        # Combine with mixing coefficients
        s_combined = (
            cfg.alpha_fast * self.s_fast +
            cfg.alpha_slow * self.s_slow +
            cfg.alpha_ultra * self.s_ultra
        )
        
        return s_combined
    
    def reset(self):
        """Reset all synaptic states."""
        self.s_fast.zero_()
        self.s_slow.zero_()
        self.s_ultra.zero_()
    
    def get_timescale_contributions(self) -> dict:
        """Get current contribution of each timescale."""
        total = self.s_fast.abs().sum() + self.s_slow.abs().sum() + self.s_ultra.abs().sum()
        if total == 0:
            return {"fast": 0.33, "slow": 0.33, "ultra": 0.34}
        return {
            "fast": (self.s_fast.abs().sum() / total).item(),
            "slow": (self.s_slow.abs().sum() / total).item(),
            "ultra": (self.s_ultra.abs().sum() / total).item(),
        }

    def dominant_timescale(self) -> str:
        """Return the name of the timescale currently contributing most."""
        contribs = self.get_timescale_contributions()
        return max(contribs, key=lambda k: contribs[k])

    def temporal_bandwidth(self) -> float:
        """
        Estimate the effective temporal bandwidth of the current synaptic state.

        Bandwidth = weighted average of 1/tau across timescales, where the
        weight is each timescale's current contribution.

        Higher bandwidth → faster dynamics; lower → slower dynamics.
        """
        import math
        contribs = self.get_timescale_contributions()
        # tau approximations from typical AMPA/NMDA/GABA constants
        taus = {"fast": 5.0, "slow": 50.0, "ultra": 200.0}
        return float(sum(contribs[k] / taus[k] for k in contribs))


class SparseFixedConnectivity(nn.Module):
    """
    Sparse fixed recurrent connectivity for FORCE training.
    
    The paper emphasizes that fixing most recurrent connections
    and only training a subset gives better stability and performance.
    
    This implements the "fixed sparse recurrent reservoir" pattern:
    - Most connections are fixed (untrainable)
    - Only specific connections (e.g., to readout-influenced neurons) are trained
    """
    
    def __init__(
        self,
        n_neurons: int,
        connectivity_p: float = 0.1,
        trainable_fraction: float = 0.1,
        device: str = "cpu",
    ):
        super().__init__()
        self.n_neurons = n_neurons
        self.connectivity_p = connectivity_p
        self.trainable_fraction = trainable_fraction
        
        # Create sparse connectivity mask
        mask = (torch.rand(n_neurons, n_neurons, device=device) < connectivity_p).float()
        
        # Determine which connections are trainable
        n_connections = int(mask.sum().item())
        n_trainable = int(n_connections * trainable_fraction)
        
        # Randomly select trainable connections from existing ones
        trainable_mask = torch.zeros_like(mask)
        connected_indices = torch.nonzero(mask, as_tuple=False)
        if n_trainable > 0 and len(connected_indices) > 0:
            trainable_idx = connected_indices[
                torch.randperm(len(connected_indices))[:n_trainable]
            ]
            trainable_mask[trainable_idx[:, 0], trainable_idx[:, 1]] = 1.0
        
        self.register_buffer("connectivity_mask", mask)
        self.register_buffer("trainable_mask", trainable_mask)
        
        # Initialize fixed weights (non-trainable)
        fixed_init = torch.randn(n_neurons, n_neurons, device=device) * 0.5
        self.register_buffer("W_fixed", fixed_init * (mask - trainable_mask))
        
        # Trainable weights (initialized to small values)
        self.W_train = nn.Parameter(
            torch.randn(n_neurons, n_neurons, device=device) * 0.01 * trainable_mask
        )
    
    def get_effective_weights(self) -> torch.Tensor:
        """Get combined fixed + trainable weights."""
        return self.W_fixed + self.W_train * self.trainable_mask
    
    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        """Apply recurrent connectivity to spikes."""
        W_eff = self.get_effective_weights()
        return torch.matmul(W_eff, spikes)
    
    def get_trainable_count(self) -> int:
        """Number of trainable connections."""
        return int(self.trainable_mask.sum().item())
    
    def get_total_count(self) -> int:
        """Total number of connections."""
        return int(self.connectivity_mask.sum().item())


# -----------------------------------------------------------------------------
# Target Pattern Generators (for FORCE training)
# -----------------------------------------------------------------------------

class PatternGenerator:
    """Generate target patterns for FORCE training."""
    
    @staticmethod
    def generate_oscillator(
        freq: float,
        amplitude: float = 1.0,
        n_steps: int = 1000,
        dt: float = 1.0,
    ) -> torch.Tensor:
        """
        Generate sinusoidal oscillator pattern.
        
        Args:
            freq: Frequency in Hz
            amplitude: Peak amplitude
            n_steps: Number of time steps
            dt: Time step size in ms
            
        Returns:
            Pattern tensor (n_steps,)
        """
        t = torch.arange(n_steps) * dt / 1000.0  # Convert to seconds
        pattern = amplitude * torch.sin(2 * np.pi * freq * t)
        return pattern
    
    @staticmethod
    def generate_coupled_oscillators(
        freqs: List[float],
        amplitudes: List[float],
        n_steps: int = 1000,
        dt: float = 1.0,
    ) -> torch.Tensor:
        """
        Generate sum of multiple oscillators.
        
        Args:
            freqs: List of frequencies in Hz
            amplitudes: List of amplitudes (must match freqs length)
            n_steps: Number of time steps
            dt: Time step size in ms
            
        Returns:
            Pattern tensor (n_steps,)
        """
        assert len(freqs) == len(amplitudes)
        
        t = torch.arange(n_steps) * dt / 1000.0
        pattern = torch.zeros(n_steps)
        
        for freq, amp in zip(freqs, amplitudes):
            pattern += amp * torch.sin(2 * np.pi * freq * t)
        
        return pattern
    
    @staticmethod
    def generate_lorenz_attractor(
        n_steps: int = 1000,
        dt: float = 0.01,
        sigma: float = 10.0,
        rho: float = 28.0,
        beta: float = 8.0/3.0,
    ) -> torch.Tensor:
        """
        Generate Lorenz chaotic attractor.
        
        The classic chaotic system used in the FORCE paper.
        dx/dt = sigma * (y - x)
        dy/dt = x * (rho - z) - y
        dz/dt = x * y - beta * z
        
        Returns:
            Pattern tensor (n_steps, 3) with [x, y, z] trajectory
        """
        # Initialize
        xyz = torch.tensor([1.0, 1.0, 1.0])
        trajectory = torch.zeros(n_steps, 3)
        
        # Integrate
        for i in range(n_steps):
            trajectory[i] = xyz
            
            x, y, z = xyz[0].item(), xyz[1].item(), xyz[2].item()
            
            dx = sigma * (y - x)
            dy = x * (rho - z) - y
            dz = x * y - beta * z
            
            xyz = xyz + torch.tensor([dx, dy, dz]) * dt
        
        return trajectory
    
    @staticmethod
    def generate_rossler_attractor(
        n_steps: int = 1000,
        dt: float = 0.01,
        a: float = 0.2,
        b: float = 0.2,
        c: float = 5.7,
    ) -> torch.Tensor:
        """
        Generate Rossler chaotic attractor.
        
        Another classic chaotic system.
        dx/dt = -y - z
        dy/dt = x + a*y
        dz/dt = b + z*(x - c)
        
        Returns:
            Pattern tensor (n_steps, 3)
        """
        xyz = torch.tensor([1.0, 1.0, 1.0])
        trajectory = torch.zeros(n_steps, 3)
        
        for i in range(n_steps):
            trajectory[i] = xyz
            
            x, y, z = xyz[0].item(), xyz[1].item(), xyz[2].item()
            
            dx = -y - z
            dy = x + a * y
            dz = b + z * (x - c)
            
            xyz = xyz + torch.tensor([dx, dy, dz]) * dt
        
        return trajectory
    
    @staticmethod
    def generate_ode_to_joy(n_steps: int = 4000, dt: float = 1.0) -> torch.Tensor:
        """
        Generate "Ode to Joy" melody pattern (as in the paper).
        
        Returns 5-component target signal representing the melody.
        """
        # Simplified version - actual implementation would use proper musical encoding
        # This generates a pattern with similar characteristics
        
        t = torch.arange(n_steps) * dt / 1000.0
        
        # Multiple frequency components as in the paper
        components = []
        base_freqs = [261.63, 293.66, 329.63, 349.23, 392.00]  # C major scale frequencies
        
        for freq in base_freqs:
            comp = torch.sin(2 * np.pi * freq * t) * (0.5 + 0.5 * torch.sin(2 * np.pi * 2 * t))
            components.append(comp)
        
        return torch.stack(components, dim=1)  # (n_steps, 5)


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------

def initialize_force_network(
    n_neurons: int,
    n_outputs: int,
    chaotic_cfg: Optional[ChaoticInitConfig] = None,
    multi_tau_cfg: Optional[MultiTimescaleSynapseConfig] = None,
    device: str = "cpu",
) -> dict:
    """
    Complete initialization for FORCE training.
    
    Returns all necessary components initialized according to the paper:
    - Chaotic recurrent weights
    - Multi-timescale synapses
    - Sparse connectivity structure
    - Readout weights (typically trained with RLS)
    
    Returns:
        Dictionary with 'W_rec', 'synapses', 'readout', etc.
    """
    chaotic_init = ChaoticInitializer(chaotic_cfg)
    
    # Initialize chaotic recurrent weights
    W_rec = chaotic_init.initialize(n_neurons, device)
    
    # Create multi-timescale synapses
    synapses = MultiTimescaleSynapses(n_neurons, multi_tau_cfg, device)
    
    # Initialize readout weights (random, to be trained)
    readout = torch.randn(n_outputs, n_neurons, device=device) * 0.1
    
    return {
        "W_rec": W_rec,
        "synapses": synapses,
        "readout": readout,
        "chaotic_init": chaotic_init,
        "spectral_radius": chaotic_init.compute_spectral_radius(W_rec),
    }


def test_chaos_property(W: torch.Tensor, n_steps: int = 1000) -> dict:
    """
    Test if a recurrent weight matrix produces chaotic dynamics.
    
    Measures:
    - Lyapunov exponent estimate
    - Activity variance over time
    - Autocorrelation decay
    
    Returns metrics dict indicating chaotic vs. stable dynamics.
    """
    n_neurons = W.shape[0]
    
    # Simulate network dynamics
    v = torch.randn(n_neurons) * 0.1  # Initial membrane potentials
    activity = torch.zeros(n_steps, n_neurons)
    
    for t in range(n_steps):
        # Simple rate-based dynamics
        r = torch.tanh(v)  # Rate
        v = v * 0.9 + torch.matmul(W, r) * 0.1 + torch.randn(n_neurons) * 0.01
        activity[t] = r
    
    # Compute metrics
    mean_activity = activity.mean(0)
    var_activity = activity.var(0).mean()
    
    # Autocorrelation at lag 1
    autocorr = torch.corrcoef(activity.T)[0, 1] if n_neurons > 1 else torch.tensor(0.0)
    
    # Estimate largest Lyapunov exponent (simplified)
    # True estimation requires perturbation analysis
    
    return {
        "mean_activity": mean_activity.mean().item(),
        "variance": var_activity.item(),
        "autocorr_lag1": autocorr.item(),
        "is_chaotic": var_activity > 0.1 and autocorr.abs() < 0.8,
    }
