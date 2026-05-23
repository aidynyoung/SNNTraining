"""
Data Loaders
============
Interfaces for real datasets:
  - Indy BCI (Sabes lab / CRCNS)       — NLB reaching, 0.81 Pearson R
  - SHD (Spiking Heidelberg Digits)    — zenodo.org/record/4906925, auto-downloaded
  - MC Maze (NeurIPS 2021 challenge)

All return generators compatible with OnlineTrainer.

Dependencies:
  pip install scipy h5py        # for Indy + SHD
  pip install nlb-tools         # for MC Maze
"""

import gzip
import os
import shutil
import urllib.request
import torch
from pathlib import Path
from typing import Iterator, Tuple, Optional, List


# ---------------------------------------------------------------------------
# SHD dataset — Spiking Heidelberg Digits
# Zenodo record 4906925: https://zenodo.org/record/4906925
# ---------------------------------------------------------------------------

SHD_URLS = {
    "train": "https://compneuro.net/datasets/shd_train.h5.gz",
    "test":  "https://compneuro.net/datasets/shd_test.h5.gz",
}
SHD_N_INPUTS  = 700
SHD_N_CLASSES = 20
SHD_SEQ_BINS  = 250       # 1-second recordings at 250 Hz (4 ms bins)


def _download_shd(split: str = "train", data_dir: str = "data/shd") -> Path:
    """Download and decompress SHD HDF5 file if not present."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    h5_path = data_dir / f"shd_{split}.h5"
    if h5_path.exists():
        return h5_path

    gz_path = data_dir / f"shd_{split}.h5.gz"
    url = SHD_URLS[split]
    print(f"Downloading SHD {split} set from {url} ...")
    urllib.request.urlretrieve(url, gz_path)
    print(f"Decompressing {gz_path} ...")
    with gzip.open(gz_path, "rb") as f_in, open(h5_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    gz_path.unlink()
    print(f"SHD {split} ready → {h5_path}")
    return h5_path


def load_shd(
    split:    str = "train",
    data_dir: str = "data/shd",
    seq_bins: int = SHD_SEQ_BINS,
    n_inputs: int = SHD_N_INPUTS,
) -> Iterator[Tuple[torch.Tensor, int]]:
    """
    Load the Spiking Heidelberg Digits dataset.

    Auto-downloads from Zenodo on first call.

    Yields
    ------
    spikes : (seq_bins, n_inputs) binary float32
    label  : int  ∈ [0, 19]
    """
    try:
        import h5py
    except ImportError:
        raise ImportError("pip install h5py")

    h5_path = _download_shd(split, data_dir)
    with h5py.File(h5_path, "r") as f:
        labels  = f["labels"][:]
        times   = f["spikes"]["times"]
        units   = f["spikes"]["units"]

        for i in range(len(labels)):
            ts   = times[i]
            us   = units[i]
            spikes = torch.zeros(seq_bins, n_inputs)
            # Each event: time in [0,1], unit index in [0, n_inputs)
            t_bins = (ts * seq_bins).astype(int).clip(0, seq_bins - 1)
            u_idx  = us.astype(int).clip(0, n_inputs - 1)
            for t_b, u in zip(t_bins, u_idx):
                spikes[t_b, u] = 1.0
            yield spikes, int(labels[i])


def load_indy(
    path: str = None,
    mat_path: str = None,
    bin_size_ms: int = 5,
    normalize: bool = True,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Indy reaching dataset (Sabes lab).
    Yields (spikes [n_channels], velocity [2]) per bin.

    Args:
        path: path to .mat file
        bin_size_ms: spike binning window
        normalize: z-score spike counts
    """
    try:
        import scipy.io as sio
        import numpy as np
    except ImportError:
        raise ImportError("pip install scipy numpy")

    resolved = mat_path or path
    if resolved is None:
        raise ValueError("Provide path= or mat_path= to the Indy .mat file")
    data = sio.loadmat(resolved)
    spikes_raw = data["spikes"]      # [T, n_channels]
    velocity = data["cursor_vel"]    # [T, 2]

    spikes = torch.tensor(spikes_raw, dtype=torch.float32)
    vel = torch.tensor(velocity, dtype=torch.float32)

    if normalize:
        mu = spikes.mean(0, keepdim=True)
        std = spikes.std(0, keepdim=True).clamp(min=1e-6)
        spikes = (spikes - mu) / std

    for t in range(len(spikes)):
        yield spikes[t], vel[t]


def load_mc_maze(
    split: str = "train",
    bin_size_ms: int = 5,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    """
    MC Maze dataset via nlb-tools.
    Yields (spikes [n_neurons], velocity [2]) per bin.

    Args:
        split: 'train' or 'val'
    """
    try:
        from nlb_tools.nwb_interface import NWBDataset
        import numpy as np
    except ImportError:
        raise ImportError("pip install nlb-tools")

    print("[load_mc_maze] Loading NWB dataset...")
    # Dataset path must be set in env or passed explicitly
    dataset = NWBDataset("mc_maze")
    data = dataset.make_trial_data(align_field="move_onset_time", bin_size=bin_size_ms)

    spikes = torch.tensor(data["spikes"], dtype=torch.float32)
    vel = torch.tensor(data["hand_vel"], dtype=torch.float32)

    for t in range(len(spikes)):
        yield spikes[t], vel[t]


# ---------------------------------------------------------------------------
# Icarus soil telemetry encoder
# Converts per-cell analyte dict → spike tensor for Arthedain RSNN inference.
# Used by icarus-dashboard/api/simulation.py via OnlineAnomalyDetector.
# ---------------------------------------------------------------------------

_SOIL_RANGES = {
    "n":                  (0.05, 0.50),
    "soc":                (0.30, 5.00),
    "moisture":           (0.00, 1.00),
    "contamination_risk": (0.00, 1.00),
}
_SOIL_ORDER = ["n", "soc", "moisture", "contamination_risk"]


def encode_soil_reading(
    analytes: dict,
    n_neurons_per_analyte: int = 10,
    T: int = 20,
) -> torch.Tensor:
    """
    Gaussian population coding of soil analytes → Poisson spike tensor.

    Args:
        analytes: dict with keys n, soc, moisture, contamination_risk
        n_neurons_per_analyte: neurons per analyte (default 10 → 40 total inputs)
        T: number of timesteps per cell reading

    Returns:
        (T, 4 * n_neurons_per_analyte) float32 spike tensor
    """
    sigma = 0.15
    centers = torch.linspace(0.0, 1.0, n_neurons_per_analyte)

    rates = []
    for key in _SOIL_ORDER:
        lo, hi = _SOIL_RANGES[key]
        val = float(max(0.0, min(1.0, (analytes[key] - lo) / (hi - lo))))
        r = torch.exp(-0.5 * ((torch.tensor(val) - centers) / sigma) ** 2)
        rates.append(r)

    rates_all = torch.cat(rates)                                  # (40,)
    return (torch.rand(T, len(rates_all)) < rates_all).float()   # (T, 40)


def get_stream(name: str = "bci_velocity", **kwargs):
    """
    Factory for all data streams. Use in experiments.
    """
    from data.synthetic import bci_velocity_stream, supply_chain_stream, mc_maze_mock

    registry = {
        "bci_velocity": bci_velocity_stream,
        "supply_chain": supply_chain_stream,
        "mc_maze_mock": mc_maze_mock,
    }
    if name not in registry:
        raise ValueError(f"Unknown stream '{name}'. Options: {list(registry.keys())}")
    return registry[name](**kwargs)
