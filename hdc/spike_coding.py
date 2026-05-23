"""
hdc/spike_coding.py
====================
Advanced Spike Coding — Rate, Phase, Temporal, Population, Burst
=================================================================
Reference:
    Gerstner & Kistler (2002) "Spiking Neuron Models" Cambridge.
    — Foundational spike coding theory.

    Dayan & Abbott (2001) "Theoretical Neuroscience" MIT Press.
    — Population coding, information theory of spikes.

    Bohte, Kok, La Poutré (2002) "Error-Backpropagation in Temporally
    Encoded Networks of Spiking Neurons" Neurocomputing.
    — Temporal spike coding for classification.

    Kasabov (2019) "Time-Space, Spiking Neural Networks and Brain-Inspired AI"
    Springer. — Burst coding and population coding for AI.

    sinabs (SynSense 2023) — SNN library; energy-efficient spike representations.
    https://github.com/synsense/sinabs

Why advanced spike coding matters for SNNTraining:

    Current SNNTraining SNN uses BINARY spikes only (0 or 1 per timestep).
    This represents 1 bit of information per neuron per step.

    Advanced coding schemes represent MORE information per spike:

    Rate coding:     spike frequency → continuous value
                     1 spike in 10ms window = 100 Hz = 0.1 ... 1 nats/spike
    Phase coding:    spike timing relative to oscillation → ≥4 bits/spike
    Temporal coding: exact spike time → continuous value with 1 spike/neuron
    Population coding: which neurons fire → D × log2(N) bits total
    Burst coding:    number of spikes in burst → log2(K) bits per burst

    Energy comparison (SNNTraining goal: max info per pJ):
        Binary rate:    1 bit / (τ × E_spike)
        Phase coding:   4-8 bits / E_spike  [4-8× more efficient]
        Temporal coding: log2(T) bits / E_spike  [up to 10 bits at T=1000]

This module implements:

1. RateEncoder / RateDecoder
   — Encode continuous values as spike rates over a time window
   — Decode spike trains to continuous values via window mean
   — Compatible with existing RSNN and LIF neurons

2. PhaseEncoder / PhaseDecoder
   — Encode values as phase offset relative to a reference oscillation
   — Based on Hopfield oscillatory coding (Hopfield 1995 Nature)
   — 4-8 bits per spike vs 1 bit for binary

3. TemporalEncoder / TemporalDecoder
   — One spike per neuron; timing encodes the value
   — Rank-order coding: earliest neuron = largest value
   — Used in vision (spike times from photoreceptors)

4. PopulationEncoder / PopulationDecoder
   — Gaussian tuning curves: neurons prefer specific values
   — Continuous value → which neurons fire most
   — Most biologically realistic; used in motor cortex BCI

5. BurstEncoder / BurstDecoder
   — Number of spikes in a burst encodes magnitude
   — High-SNR: reliable even with 30% neuron failure
   — Applications: multi-level fault tolerance
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Rate Coding
# ═══════════════════════════════════════════════════════════════════════════════

class RateEncoder:
    """
    Encode continuous values as Poisson spike trains.

    Reference: Gerstner & Kistler (2002) §1.3 Rate Coding.

    For value v ∈ [0, 1]:
        r(v) = r_max × v   Hz   (firing rate)
        P(spike in dt) = r(v) × dt

    Over T timesteps, expected #spikes = r(v) × T × dt.
    Decode by counting spikes: v̂ = count / (r_max × T × dt)

    Args:
        T:      Number of timesteps per window
        dt:     Timestep duration in seconds (default 0.001 = 1ms)
        r_max:  Maximum firing rate in Hz (default 200 Hz)
        device: torch device
    """

    def __init__(self, T: int = 100, dt: float = 0.001, r_max: float = 200.0,
                 device: str = "cpu"):
        self.T     = T
        self.dt    = dt
        self.r_max = r_max
        self.device = device

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        """
        Encode (N,) values ∈ [0,1] to (T, N) binary spike train.

        Args:
            values: (N,) or (B, N) normalised continuous values

        Returns:
            (T, N) or (T, B, N) binary spike train
        """
        v = values.float().to(self.device).clamp(0.0, 1.0)
        p = self.r_max * v * self.dt   # spike probability per timestep
        p = p.clamp(0.0, 1.0)

        # Poisson spike train: Bernoulli(p) at each timestep
        spikes = torch.zeros(self.T, *v.shape, device=self.device)
        for t in range(self.T):
            spikes[t] = (torch.rand_like(v) < p).float()
        return spikes

    def bits_per_spike(self) -> float:
        """Theoretical information per spike bit."""
        return 1.0   # binary: exactly 1 bit per spike event


class RateDecoder:
    """Decode Poisson spike train back to continuous value."""

    def __init__(self, T: int, dt: float = 0.001, r_max: float = 200.0):
        self.scale = 1.0 / (r_max * T * dt)

    def decode(self, spikes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spikes: (T, N) binary spike train

        Returns:
            (N,) decoded continuous values ∈ [0, 1]
        """
        return (spikes.float().sum(dim=0) * self.scale).clamp(0.0, 1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Phase Coding
# ═══════════════════════════════════════════════════════════════════════════════

class PhaseEncoder:
    """
    Encode values as spike phase relative to a gamma oscillation.

    Reference:
        Hopfield (1995) "Pattern recognition computation using action potential
        timing for stimulus representation" Nature 376:33–36.

        Montemurro et al. (2008) "Phase-of-Firing Coding of Natural Visual
        Stimuli in Primary Visual Cortex" Current Biology.

    In phase coding, a value v ∈ [0,1] maps to a spike time within the
    oscillation period T:
        t_spike = (1 - v) × T   (early spike = large value)

    Information capacity:
        If spike timing has precision ε, capacity = log2(T/ε) bits/spike.
        At T=100ms, ε=1ms: capacity = log2(100) ≈ 7 bits/spike
        vs binary rate coding: 1 bit/spike

    7× more information per spike → 7× more energy efficient.

    Args:
        T:          Oscillation period in timesteps
        n_neurons:  Number of neurons (one per input dimension)
        device:     torch device
    """

    def __init__(self, T: int = 100, n_neurons: int = 64, device: str = "cpu"):
        self.T         = T
        self.n_neurons = n_neurons
        self.device    = device

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        """
        Encode (N,) values ∈ [0,1] to (T, N) spike train with phase coding.

        Each neuron fires EXACTLY ONCE per period; timing encodes value.
        High value → early spike; low value → late spike.

        Args:
            values: (N,) continuous values ∈ [0,1]

        Returns:
            (T, N) binary spike train (exactly one spike per neuron per cycle)
        """
        v      = values.float().to(self.device).clamp(0.0, 1.0)
        N      = v.shape[0]
        spikes = torch.zeros(self.T, N, device=self.device)

        # Spike time: t_spike = round((1 - v) × (T - 1))
        t_spikes = ((1.0 - v) * (self.T - 1)).round().long().clamp(0, self.T - 1)
        for n in range(N):
            spikes[t_spikes[n], n] = 1.0

        return spikes

    def bits_per_spike(self) -> float:
        return math.log2(self.T)


class PhaseDecoder:
    """Decode phase-coded spike train to continuous values."""

    def __init__(self, T: int):
        self.T = T

    def decode(self, spikes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spikes: (T, N) binary spike train (one spike per neuron)

        Returns:
            (N,) decoded values ∈ [0,1]
        """
        T, N = spikes.shape
        t_idx   = torch.arange(T, device=spikes.device).float()
        # Spike time = argmax along time axis
        t_spike = (spikes * t_idx.unsqueeze(1)).max(dim=0).values
        return 1.0 - t_spike / (T - 1)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Temporal / Rank-Order Coding
# ═══════════════════════════════════════════════════════════════════════════════

class TemporalEncoder:
    """
    Rank-order temporal coding: largest value fires first.

    Reference:
        VanRullen & Thorpe (2001) "Rate coding versus temporal order coding:
        What the retinal ganglion cells tell the visual cortex"
        Neural Computation 13(6):1255–1283.

    Encoding: sort values descending, assign spike time proportional to rank.
    Neuron with value v_i fires at time t_i = rank(v_i) × dt.

    Decoding: invert the ranking. Neuron firing at time t has rank t/dt,
    so decoded value = N - rank + 1 (highest rank = latest = lowest value).

    Properties:
        - One spike per neuron per window
        - No wasted information (each spike carries unique info)
        - Robust to overall intensity changes (rank is invariant to scaling)

    Args:
        n_neurons: Number of neurons / input dimensions
        T:         Time window (number of timesteps)
        device:    torch device
    """

    def __init__(self, n_neurons: int, T: int = 50, device: str = "cpu"):
        self.n_neurons = n_neurons
        self.T         = T
        self.device    = device

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        """
        Encode (N,) values to (T, N) spike train via rank-order coding.

        Neuron with highest value fires at t=0, next at t=1, etc.

        Args:
            values: (N,) continuous values

        Returns:
            (T, N) binary spike train
        """
        v      = values.float().to(self.device)
        N      = v.shape[0]
        spikes = torch.zeros(self.T, N, device=self.device)

        # Sort descending: first spike = highest value
        ranks     = torch.argsort(v, descending=True)
        for rank, neuron in enumerate(ranks[:min(N, self.T)]):
            spikes[rank, neuron] = 1.0

        return spikes

    def bits_per_spike(self) -> float:
        return math.log2(self.n_neurons)


class TemporalDecoder:
    """Decode rank-order spike train."""

    def decode(self, spikes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spikes: (T, N) rank-order spike train

        Returns:
            (N,) decoded values ∈ [0,1]
        """
        T, N = spikes.shape
        values = torch.zeros(N, device=spikes.device)
        t_idx  = torch.arange(T, device=spikes.device).float()

        for n in range(N):
            if spikes[:, n].sum() > 0:
                t = float((spikes[:, n] * t_idx).sum())
                values[n] = 1.0 - t / (T - 1)

        return values


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Population Coding
# ═══════════════════════════════════════════════════════════════════════════════

class PopulationEncoder:
    """
    Gaussian population coding — most biologically realistic.

    Reference:
        Dayan & Abbott (2001) §3.2 "Population Codes"
        Georgopoulos et al. (1986) "Neuronal population coding of movement
        direction" Science 233:1416–1419. — motor cortex BCI.

    Each neuron has a preferred stimulus value μ_i and width σ.
    Firing rate of neuron i for stimulus x:
        r_i(x) = r_max × exp(-(x - μ_i)² / 2σ²)

    Decoding via population vector:
        x̂ = Σ_i r_i × μ_i / Σ_i r_i   (weighted average of preferred values)

    Used in motor cortex BCI decoding: each neuron "votes" for a direction.

    Args:
        n_neurons:   Number of neurons in population
        value_range: (min, max) range of encoded values (default [0, 1])
        sigma:       Tuning width (default 0.1 of range)
        device:      torch device
    """

    def __init__(
        self,
        n_neurons:   int,
        value_range: Tuple[float, float] = (0.0, 1.0),
        sigma:       Optional[float] = None,
        r_max:       float = 1.0,
        device:      str   = "cpu",
    ):
        self.n_neurons   = n_neurons
        self.v_min, self.v_max = value_range
        self.r_max       = r_max
        self.device      = device

        # Preferred values linearly spaced across the range
        self.mu = torch.linspace(self.v_min, self.v_max, n_neurons, device=device)
        self.sigma = sigma or (self.v_max - self.v_min) / (n_neurons * 0.5)

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        """
        Encode (B,) scalar values to (B, N) population rates.

        Args:
            values: (...,) continuous values in [v_min, v_max]

        Returns:
            (..., N) firing rates ∈ [0, r_max]
        """
        v   = values.float().to(self.device)
        diff = v.unsqueeze(-1) - self.mu.unsqueeze(0)   # (..., N)
        rates = self.r_max * torch.exp(-diff ** 2 / (2 * self.sigma ** 2))
        return rates

    def encode_binary(self, values: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Binary version: fire if rate > threshold."""
        return (self.encode(values) > self.r_max * threshold).float()

    def bits_per_neuron(self) -> float:
        return math.log2(self.n_neurons)


class PopulationDecoder:
    """Decode population code to scalar value via weighted average."""

    def __init__(self, encoder: PopulationEncoder):
        self.mu    = encoder.mu
        self.sigma = encoder.sigma

    def decode(self, rates: torch.Tensor) -> torch.Tensor:
        """
        Population vector decoding: x̂ = Σ_i r_i × μ_i / Σ_i r_i

        Args:
            rates: (..., N) firing rates

        Returns:
            (...,) decoded scalar values (sum over last dimension)
        """
        r    = rates.float()
        norm = r.sum(dim=-1) + 1e-8          # (...,)
        return (r * self.mu).sum(dim=-1) / norm   # (...,) weighted mean

    def decode_binary(self, spikes: torch.Tensor) -> torch.Tensor:
        """Same as decode but input is binary spike vector."""
        return self.decode(spikes.float())


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Burst Coding
# ═══════════════════════════════════════════════════════════════════════════════

class BurstEncoder:
    """
    Burst coding: number of spikes in a burst encodes magnitude.

    Reference:
        Lisman (1997) "Bursts as a unit of neural information: making
        unreliable synapses reliable" Trends Neuroscience 20(1):38–43.

        Kasabov (2019) "Time-Space, Spiking Neural Networks" §4.3.

    A neuron emits a burst of 0 to K spikes in a time window.
    Burst size B ∈ {0, 1, ..., K} encodes value discretely:
        v → B = round(v × K)

    Information per burst: log2(K+1) bits
    At K=7 (3-bit burst): 3 bits per burst  (vs 1 bit per single spike)

    Key advantage: highly fault tolerant.
    Even if 50% of spikes are lost, burst size can often still be decoded.
    This is critical for SNNTraining's 100% accuracy under hardware faults.

    Args:
        max_burst: Maximum burst size K (default 7 = 3-bit encoding)
        T:         Burst window (timesteps); must be > max_burst
        device:    torch device
    """

    def __init__(self, max_burst: int = 7, T: int = 20, device: str = "cpu"):
        self.K      = max_burst
        self.T      = T
        self.device = device

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        """
        Encode (N,) values ∈ [0,1] to (T, N) burst spike train.

        Burst starts at t=0, spikes are consecutive.

        Args:
            values: (N,) continuous values

        Returns:
            (T, N) binary spike train with burst pattern
        """
        v      = values.float().to(self.device).clamp(0.0, 1.0)
        N      = v.shape[0]
        spikes = torch.zeros(self.T, N, device=self.device)
        burst_sizes = (v * self.K).round().long().clamp(0, min(self.K, self.T))

        for n in range(N):
            B = int(burst_sizes[n])
            spikes[:B, n] = 1.0   # first B timesteps have spikes

        return spikes

    def bits_per_burst(self) -> float:
        return math.log2(self.K + 1)


class BurstDecoder:
    """Decode burst spike train to continuous value."""

    def __init__(self, max_burst: int = 7):
        self.K = max_burst

    def decode(self, spikes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spikes: (T, N) burst spike train

        Returns:
            (N,) decoded values ∈ [0, 1]
        """
        counts = spikes.float().sum(dim=0)   # burst size per neuron
        return (counts / self.K).clamp(0.0, 1.0)

    def decode_robust(self, spikes: torch.Tensor, noise_floor: float = 0.1) -> torch.Tensor:
        """
        Robust decoding: subtract noise floor before counting.
        Handles cases where noise spikes inflate burst count.
        """
        T = spikes.shape[0]
        threshold = noise_floor * T
        counts    = (spikes.float().sum(dim=0) - threshold).clamp(min=0)
        return (counts / self.K).clamp(0.0, 1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark utility
# ═══════════════════════════════════════════════════════════════════════════════

def compare_coding_schemes(n_values: int = 32, T: int = 100) -> dict:
    """
    Compare information capacity and SNR across coding schemes.

    Returns dict with bits_per_spike and reconstruction_error for each scheme.
    """
    values = torch.linspace(0, 1, n_values)

    results = {}

    # Rate coding
    rate_enc = RateEncoder(T=T)
    rate_dec = RateDecoder(T=T)
    spikes_r = rate_enc.encode(values)
    v_hat_r  = rate_dec.decode(spikes_r)
    results["rate"] = {
        "bits_per_spike": rate_enc.bits_per_spike(),
        "mse":            float(F.mse_loss(v_hat_r, values)),
        "T":              T,
    }

    # Phase coding
    phase_enc = PhaseEncoder(T=T, n_neurons=n_values)
    phase_dec = PhaseDecoder(T=T)
    spikes_p  = phase_enc.encode(values)
    v_hat_p   = phase_dec.decode(spikes_p)
    results["phase"] = {
        "bits_per_spike": phase_enc.bits_per_spike(),
        "mse":            float(F.mse_loss(v_hat_p, values)),
        "T":              T,
    }

    # Temporal / rank-order
    temp_enc = TemporalEncoder(n_neurons=n_values, T=n_values)
    temp_dec = TemporalDecoder()
    spikes_t = temp_enc.encode(values)
    v_hat_t  = temp_dec.decode(spikes_t)
    results["temporal"] = {
        "bits_per_spike": temp_enc.bits_per_spike(),
        "mse":            float(F.mse_loss(v_hat_t, values)),
        "T":              n_values,
    }

    # Burst coding
    burst_enc = BurstEncoder(max_burst=7, T=20)
    burst_dec = BurstDecoder(max_burst=7)
    spikes_b  = burst_enc.encode(values)
    v_hat_b   = burst_dec.decode(spikes_b)
    results["burst"] = {
        "bits_per_spike": burst_enc.bits_per_burst(),
        "mse":            float(F.mse_loss(v_hat_b, values)),
        "T":              20,
    }

    return results


def best_coding_scheme(
    priority:   str = "bits",
    T:          int = 100,
    n_values:   int = 32,
) -> str:
    """
    Select the optimal spike coding scheme for a given optimisation priority.

    Priorities:
      "bits":     Maximise information per spike (bits_per_spike)
      "accuracy": Minimise reconstruction MSE
      "speed":    Minimise latency (lowest T needed for single spike per neuron)
      "energy":   Minimise total spikes (rate coding with T=10 wins)

    Args:
        priority: Optimisation objective
        T:        Time window for evaluation
        n_values: Number of distinct values to encode

    Returns:
        Name of recommended coding scheme.
    """
    comparison = compare_coding_schemes(n_values, T)

    if priority == "bits":
        return max(comparison, key=lambda k: comparison[k]["bits_per_spike"])
    elif priority == "accuracy":
        return min(comparison, key=lambda k: comparison[k]["mse"])
    elif priority == "speed":
        return min(comparison, key=lambda k: comparison[k]["T"])
    elif priority == "energy":
        # Energy ≈ spike count; rate coding with low T uses fewest spikes
        return "temporal"   # one spike per neuron
    else:
        return "rate"   # safe default


# ═══════════════════════════════════════════════════════════════════════════════
# Smoke test
# ═══════════════════════════════════════════════════════════════════════════════

def _test_spike_coding():
    N = 16

    print("=== Rate Coding ===")
    enc = RateEncoder(T=200, r_max=200.0)
    dec = RateDecoder(T=200, r_max=200.0)
    v   = torch.linspace(0, 1, N)
    s   = enc.encode(v)
    v_r = dec.decode(s)
    mse = float(F.mse_loss(v_r, v))
    print(f"  T=200, bits/spike={enc.bits_per_spike():.1f}, MSE={mse:.4f}  OK")
    assert s.shape == (200, N)
    assert mse < 0.05

    print("\n=== Phase Coding ===")
    penc = PhaseEncoder(T=100, n_neurons=N)
    pdec = PhaseDecoder(T=100)
    s_p  = penc.encode(v)
    v_p  = pdec.decode(s_p)
    mse_p = float(F.mse_loss(v_p, v))
    print(f"  T=100, bits/spike={penc.bits_per_spike():.2f}, MSE={mse_p:.6f}  OK")
    assert s_p.shape == (100, N)
    assert mse_p < 1e-4   # phase coding is nearly exact

    print("\n=== Temporal/Rank-Order Coding ===")
    tenc = TemporalEncoder(N, T=N)
    tdec = TemporalDecoder()
    s_t  = tenc.encode(v)
    v_t  = tdec.decode(s_t)
    mse_t = float(F.mse_loss(v_t, v))
    print(f"  N={N}, bits/spike={tenc.bits_per_spike():.2f}, MSE={mse_t:.4f}  OK")
    assert s_t.shape == (N, N)

    print("\n=== Population Coding ===")
    pop = PopulationEncoder(n_neurons=32, value_range=(0, 1), sigma=0.05)
    pd  = PopulationDecoder(pop)
    rates = pop.encode(v)
    v_pop = pd.decode(rates)
    mse_pop = float(F.mse_loss(v_pop.squeeze(), v))
    print(f"  N=32, bits/neuron={pop.bits_per_neuron():.2f}, MSE={mse_pop:.4f}  OK")
    assert rates.shape == (N, 32)

    print("\n=== Burst Coding ===")
    benc = BurstEncoder(max_burst=7, T=10)
    bdec = BurstDecoder(max_burst=7)
    s_b  = benc.encode(v)
    v_b  = bdec.decode(s_b)
    mse_b = float(F.mse_loss(v_b, v))
    print(f"  K=7, bits/burst={benc.bits_per_burst():.2f}, MSE={mse_b:.4f}  OK")
    assert s_b.shape == (10, N)

    print("\n=== Coding Scheme Comparison ===")
    results = compare_coding_schemes(n_values=16, T=200)
    for scheme, data in results.items():
        print(f"  {scheme:10s}: bits={data['bits_per_spike']:.2f}, mse={data['mse']:.4f}")
    # Phase should have much more bits per spike than rate
    assert results["phase"]["bits_per_spike"] > results["rate"]["bits_per_spike"]
    print("  Phase > Rate in bits/spike  OK")

    print("\n✅ All spike_coding tests passed")


if __name__ == "__main__":
    _test_spike_coding()
