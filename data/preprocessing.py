"""
Preprocessing Utilities
=======================
Spike encoding, binning, normalization, and signal processing.
"""

import torch
import math
from typing import Iterator, Tuple


def rate_encode(signal: torch.Tensor, T: int, max_rate: float = 1.0) -> torch.Tensor:
    """
    Rate encoding: continuous signal → spike train [T x n].
    Each timestep fires with probability proportional to |signal|.
    """
    n = signal.shape[0]
    rates = (signal.abs() / signal.abs().max().clamp(min=1e-6)) * max_rate
    return (torch.rand(T, n) < rates.unsqueeze(0)).float()


def temporal_encode(signal: torch.Tensor, T: int) -> torch.Tensor:
    """
    Time-to-first-spike encoding.
    Stronger values → earlier spikes.
    Returns [T x n] binary spike train.
    """
    n = signal.shape[0]
    norm = (signal - signal.min()) / (signal.max() - signal.min()).clamp(min=1e-6)
    spike_times = ((1.0 - norm) * (T - 1)).long()
    spikes = torch.zeros(T, n)
    for i, t in enumerate(spike_times):
        spikes[t.item(), i] = 1.0
    return spikes


def bin_spikes(spike_train: torch.Tensor, bin_size: int) -> torch.Tensor:
    """
    Bin a spike train [T x n] into [T//bin_size x n].
    """
    T, n = spike_train.shape
    T_new = T // bin_size
    return spike_train[:T_new * bin_size].reshape(T_new, bin_size, n).sum(dim=1)


def zscore(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Z-score normalize along time axis [T x n]."""
    mu = x.mean(0, keepdim=True)
    std = x.std(0, keepdim=True).clamp(min=eps)
    return (x - mu) / std


def smooth(signal: torch.Tensor, tau: float = 10.0) -> torch.Tensor:
    """
    Exponential smoothing of signal [T x n].
    """
    decay = math.exp(-1.0 / tau)
    out = torch.zeros_like(signal)
    state = signal[0].clone()
    for t in range(len(signal)):
        state = decay * state + (1 - decay) * signal[t]
        out[t] = state
    return out
