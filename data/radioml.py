"""
data/radioml.py
================
RadioML 2016.10A dataset loader for RF signal classification.

RadioML 2016.10A is the standard benchmark for RF modulation classification —
11 modulation types (AM-DSB, AM-SSB, WBFM, BPSK, QPSK, 8PSK, QAM16, QAM64,
CPFSK, GFSK, PAM4) at SNR levels from -20 to +18 dB.

This is DeepSig's free public dataset, used to validate that SNNTraining's SNN
pipeline classifies real-world RF signals — not just synthetic patterns.

IQT relevance
-------------
  Defense ISR requires RF signal classification from real sensor data.
  "Validated on synthetic data only" is insufficient for TRL 5.
  RadioML provides the publicly available validation step.

Download
--------
  Dataset: https://www.deepsig.ai/datasets (RadioML 2016.10A, 55MB)
  Or via: http://opendata.deepsig.ai/datasets/2016.10/RML2016.10a.tar.bz2

  Auto-download is attempted; manual download also supported.

Usage
-----
    from data.radioml import RadioMLDataset, load_radioml

    # With auto-download
    train, test = load_radioml(snr_min=0, n_per_class=500)

    # Manual path
    train, test = load_radioml(path="data/RML2016.10a_dict.pkl")

    # As spike encoder
    encoder = RadioMLSpikeEncoder(n_neurons=64)
    spikes = encoder.encode(iq_sample)   # (T, 64) spike train

Paper reference
---------------
    O'Shea, T., & West, N. (2016). Radio machine learning dataset generation
    with GNU radio. Proceedings of the GNU Radio Conference, 1(1).
    https://pubs.gnuradio.org/index.php/grcon/article/view/11
"""

from __future__ import annotations

import os
import pickle
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

DATA_DIR = Path(__file__).parent / "radioml"
DATASET_URL = "http://opendata.deepsig.ai/datasets/2016.10/RML2016.10a_dict.pkl"
DATASET_FILENAME = "RML2016.10a_dict.pkl"

MODULATIONS = [
    "AM-DSB", "AM-SSB", "WBFM",
    "BPSK",   "QPSK",   "8PSK",
    "QAM16",  "QAM64",
    "CPFSK",  "GFSK",   "PAM4",
]
MOD_TO_IDX = {m: i for i, m in enumerate(MODULATIONS)}
SNR_LEVELS  = list(range(-20, 19, 2))   # -20 to +18 dB step 2


# ── Dataset class ──────────────────────────────────────────────────────────────

class RadioMLDataset(Dataset):
    """
    RadioML 2016.10A dataset.

    Each sample is a complex IQ time series (2, 128) — real and imaginary
    components — for one of 11 modulation types.

    Attributes
    ----------
    samples : (N, 2, 128) float32 IQ signals
    labels  : (N,) int64 modulation class indices
    snrs    : (N,) int SNR values
    """

    def __init__(
        self,
        data: Dict,
        snr_min: int = -20,
        snr_max: int = 18,
        n_per_class: Optional[int] = None,
        seed: int = 42,
    ):
        rng = np.random.RandomState(seed)
        samples, labels, snrs = [], [], []

        for (mod, snr), iq_array in data.items():
            if mod not in MOD_TO_IDX:
                continue
            if not (snr_min <= snr <= snr_max):
                continue
            idx = MOD_TO_IDX[mod]
            N = iq_array.shape[0]
            chosen = rng.permutation(N)[:n_per_class] if n_per_class else np.arange(N)
            for i in chosen:
                samples.append(iq_array[i])
                labels.append(idx)
                snrs.append(snr)

        self.samples = torch.tensor(np.stack(samples), dtype=torch.float32)
        self.labels  = torch.tensor(labels,            dtype=torch.long)
        self.snrs    = torch.tensor(snrs,              dtype=torch.int32)
        self.n_classes = len(MODULATIONS)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx], self.labels[idx]

    def per_snr_accuracy(self, preds: torch.Tensor) -> Dict[int, float]:
        """Compute per-SNR accuracy given prediction tensor."""
        results = {}
        for snr_val in SNR_LEVELS:
            mask = self.snrs == snr_val
            if mask.sum() == 0:
                continue
            correct = (preds[mask] == self.labels[mask]).float().mean().item()
            results[snr_val] = correct
        return results


# ── Auto-download ──────────────────────────────────────────────────────────────

def _download_radioml(dest: Path) -> Optional[Path]:
    """Attempt auto-download. Returns path on success, None on failure."""
    dest.mkdir(parents=True, exist_ok=True)
    pkl_path = dest / DATASET_FILENAME
    if pkl_path.exists():
        return pkl_path

    print(f"  Downloading RadioML 2016.10A from {DATASET_URL}…")
    print(f"  Destination: {pkl_path}")
    try:
        urllib.request.urlretrieve(
            DATASET_URL, str(pkl_path),
            reporthook=lambda b, bs, ts: print(
                f"\r  {min(b*bs, ts)/1e6:.1f} / {ts/1e6:.1f} MB", end="", flush=True
            ) if ts > 0 else None,
        )
        print()
        return pkl_path
    except Exception as e:
        print(f"\n  Download failed: {e}")
        print(f"  Manual download: {DATASET_URL}")
        return None


def load_radioml(
    path: Optional[str] = None,
    snr_min: int = 0,
    snr_max: int = 18,
    n_per_class: Optional[int] = 500,
    train_frac: float = 0.8,
    seed: int = 42,
) -> Tuple[Optional[RadioMLDataset], Optional[RadioMLDataset]]:
    """
    Load RadioML 2016.10A and return (train_dataset, test_dataset).

    Args:
        path:         Path to RML2016.10a_dict.pkl (auto-download if None)
        snr_min:      Minimum SNR to include (dB)
        snr_max:      Maximum SNR to include (dB)
        n_per_class:  Samples per (modulation, SNR) pair (None = all)
        train_frac:   Fraction for training
        seed:         Random seed for split

    Returns:
        (train_dataset, test_dataset) or (None, None) if data unavailable.
    """
    # Resolve path
    if path is None:
        pkl_path = _download_radioml(DATA_DIR)
    else:
        pkl_path = Path(path)
        if not pkl_path.exists():
            print(f"  RadioML file not found: {pkl_path}")
            return None, None

    if pkl_path is None or not pkl_path.exists():
        print("  RadioML dataset unavailable. Skipping.")
        return None, None

    print(f"  Loading RadioML from {pkl_path}…")
    with open(pkl_path, "rb") as f:
        try:
            data = pickle.load(f, encoding="latin1")
        except Exception:
            data = pickle.load(f)

    full = RadioMLDataset(data, snr_min=snr_min, snr_max=snr_max,
                          n_per_class=n_per_class, seed=seed)

    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(full))
    n_train = int(len(full) * train_frac)
    train_idx = indices[:n_train]
    test_idx  = indices[n_train:]

    class Subset(Dataset):
        def __init__(self, ds, idx):
            self.ds = ds; self.idx = idx
            self.labels = ds.labels[idx]
            self.snrs   = ds.snrs[idx]
            self.n_classes = ds.n_classes
        def __len__(self): return len(self.idx)
        def __getitem__(self, i):
            return self.ds.samples[self.idx[i]], self.ds.labels[self.idx[i]]

    return Subset(full, train_idx), Subset(full, test_idx)


# ── Spike encoder ──────────────────────────────────────────────────────────────

class RadioMLSpikeEncoder:
    """
    Converts IQ samples to spike trains for SNNTraining SNN processing.

    Encoding:
      1. Compute instantaneous amplitude and phase from I, Q channels
      2. Divide the feature space into n_neurons threshold bands
      3. A neuron fires when its band's value exceeds its threshold

    This converts continuous RF features into binary spike events that
    the SNN can process event-driven.
    """

    def __init__(
        self,
        n_neurons: int = 64,
        threshold_scale: float = 1.0,
        seed: int = 0,
    ):
        self.n_neurons = n_neurons
        self.threshold_scale = threshold_scale
        rng = np.random.RandomState(seed)
        # Thresholds uniformly distributed over [-2, 2] (normalised IQ range)
        self.thresholds = torch.tensor(
            rng.uniform(-2, 2, (n_neurons,)), dtype=torch.float32
        )

    def encode(self, iq: torch.Tensor) -> torch.Tensor:
        """
        Encode one IQ sample to a spike train.

        Args:
            iq: (2, 128) IQ tensor [real, imag]

        Returns:
            (128, n_neurons) spike train — binary
        """
        I, Q = iq[0], iq[1]          # (128,) each
        amplitude = torch.sqrt(I**2 + Q**2)   # instantaneous amplitude
        phase = torch.atan2(Q, I) / torch.pi  # normalised phase [-1, 1]

        # Stack to (128, 2) features
        features = torch.stack([amplitude, phase], dim=1)   # (128, 2)

        # Expand to (128, n_neurons) by comparing each sample to neuron thresholds
        # Each neuron fires when amplitude > threshold (rate-coded)
        spikes = (amplitude.unsqueeze(1) > self.thresholds.unsqueeze(0)).float()
        return spikes  # (128, n_neurons)

    def encode_batch(self, iq_batch: torch.Tensor) -> torch.Tensor:
        """Encode a batch of IQ samples.

        Args:
            iq_batch: (N, 2, 128)

        Returns:
            (N, 128, n_neurons) spike trains
        """
        return torch.stack([self.encode(iq_batch[i]) for i in range(len(iq_batch))])


# ── Quick validation ───────────────────────────────────────────────────────────

def validate_radioml_available() -> bool:
    """Return True if RadioML dataset is present locally."""
    return (DATA_DIR / DATASET_FILENAME).exists()


def radioml_summary() -> str:
    """Return a one-line summary of dataset availability."""
    if validate_radioml_available():
        return f"RadioML 2016.10A available at {DATA_DIR / DATASET_FILENAME}"
    return (
        f"RadioML 2016.10A not found. "
        f"Download from {DATASET_URL} → {DATA_DIR / DATASET_FILENAME}"
    )


if __name__ == "__main__":
    print(radioml_summary())
    if validate_radioml_available():
        train, test = load_radioml(snr_min=0, n_per_class=100)
        if train:
            print(f"Train: {len(train)} samples, {train.n_classes} classes")
            print(f"Test:  {len(test)} samples")
            x, y = train[0]
            print(f"Sample shape: {x.shape}, label: {MODULATIONS[y]}")
            enc = RadioMLSpikeEncoder(n_neurons=64)
            spikes = enc.encode(x)
            print(f"Spike train shape: {spikes.shape} (T=128 × N=64 neurons)")
            print(f"Mean spike rate: {spikes.mean():.3f}")
