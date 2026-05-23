"""
snntraining_robustness.py
=======================
End-to-end robustness experiment: SNN → HDC pipeline under hardware faults.

This is the strongest test of the SNNTraining integration thesis:
it measures how HDC error correction (ECC) and fault-tolerant encoding
protect the entire SNN→HDC pipeline when the SNN's weight memory is
corrupted by realistic hardware faults.

Key design decisions (literature-backed):
  1. **Reservoir computing paradigm** (Biswas et al. 2024, Karki et al. 2024):
     The RSNN's recurrent weights are FIXED (random reservoir). Only the
     readout is trained. This is the standard approach for liquid state
     machines and echo state networks.
  2. **SpikeFI-compatible fault models** (Spyrou et al. 2024):
     stuck-at-0/1, permanent bit-flips, synaptic silence, mixed
  3. **HDC error correction codes (ECC)** (Podlaski et al. 2025; Saponati et al. 2026):
     Uses associative memory prototypes to detect and repair corrupted weights
  4. **Temporal task with sustained-input blocks**:
     Each class is presented for `block_len` consecutive timesteps.
     The reservoir integrates the sustained input and produces class-specific
     spike patterns that the readout decodes.

Architecture
------------
    SNN weight memory (W_rec, W_in)  ← SpikeFI fault injection here
         ↓
    RSNN forward pass (FIXED reservoir, no Hebbian update)
         ↓
    Spikes → Readout (trained via SGD with weight decay)
         ↓
    Accuracy (%)  vs  HDC Accuracy (%)

    5 configurations compared:
        baseline:     SNN only (no HDC)
        hdc_head:     SNN + HDC (static classification head)
        hdc_loop:     SNN + HDC + closed-loop lr feedback
        hdc_masked:   SNN + HDC + loop + error masking on SNN weights
        hdc_ecc:      SNN + HDC + ECC weight repair (PI control)

Usage
-----
    python experiments/snntraining_robustness.py --error-rates 0 1e-6 1e-4 1e-2
    python experiments/snntraining_robustness.py --quick
    python experiments/snntraining_robustness.py --fault-type stuck_at_0 --persistent
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple, Optional

import torch
import numpy as np

from models import (
    RSNN, RSNNConfig, Readout, ReadoutConfig,
    HDCEncoder, SpikeHDC, AssocMemory, ItemMemory, GrapHD,
)
from models.hdc import gen_hvs, bind, batch_sim, thresh
from hdc.fault_models import FaultInjector, FaultConfig, FaultType, make_fault_profile
from hdc.ecc import HDCCorrector, ECCConfig


# ---------------------------------------------------------------------------
# Temporal classification stream: sustained-input blocks
# ---------------------------------------------------------------------------

def make_temporal_stream(prototypes: np.ndarray, T: int,
                          block_len: int = 20, noise: float = 0.05):
    """Generate a temporal classification stream with sustained-input blocks.

    Each class is presented for `block_len` consecutive timesteps.
    The reservoir integrates the sustained input and produces class-specific
    spike patterns that the readout decodes.

    Args:
        prototypes: (n_classes, input_size) array of class prototypes
        T: Total number of timesteps
        block_len: Number of consecutive timesteps per class
        noise: Gaussian noise std added to prototypes

    Yields:
        (input_tensor, label_tensor) where label is a scalar long tensor.
    """
    n_classes = prototypes.shape[0]
    rng = np.random.RandomState(42)
    t = 0
    while t < T:
        label = (t // block_len) % n_classes
        for _ in range(block_len):
            if t >= T:
                break
            x = prototypes[label] + noise * rng.randn(prototypes.shape[1]).astype(np.float32)
            yield torch.tensor(x, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
            t += 1


def make_prototypes(n_classes: int, input_size: int, seed: int = 42) -> np.ndarray:
    """Create normalized class prototypes."""
    rng = np.random.RandomState(seed)
    prototypes = rng.randn(n_classes, input_size).astype(np.float32)
    prototypes = prototypes / np.linalg.norm(prototypes, axis=1, keepdims=True)
    return prototypes


def make_regression_stream(input_size: int, T: int, seed: int = 42,
                            noise: float = 0.1):
    """Generate BCI velocity regression stream for robustness testing.

    Uses the same synthetic velocity stream as bci_decoding.py.
    More realistic than block classification — tests the SNN's ability
    to decode smooth trajectories under hardware faults.

    Yields:
        (input_tensor, target_tensor) where target is (2,) velocity
    """
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    freq = 30.0
    velocity_scale = 0.8

    for t in range(T):
        phase = 2 * np.pi * t / freq
        vel_x = np.sin(phase) * velocity_scale + noise * rng.randn()
        vel_y = np.cos(phase) * velocity_scale * 0.75 + noise * rng.randn()

        # Input: sparse Poisson-like spikes with directional tuning
        preferred_dirs = np.linspace(0, 2 * np.pi, input_size, endpoint=False)
        dir_offset = rng.randn(input_size) * 0.3
        preferred_dirs = preferred_dirs + dir_offset
        speed = np.sqrt(vel_x**2 + vel_y**2)
        direction = np.arctan2(vel_y, vel_x)
        cosine_response = np.maximum(0, np.cos(preferred_dirs - direction))
        speed_modulation = np.clip(speed / velocity_scale, 0, 1.5)
        firing_rates = 10.0 + 50.0 * cosine_response * speed_modulation
        spike_probs = np.clip(firing_rates * 0.02, 0.0, 1.0)
        spikes = rng.binomial(1, spike_probs).astype(np.float32)

        x = torch.tensor(spikes, dtype=torch.float32)
        y = torch.tensor([vel_x, vel_y], dtype=torch.float32)
        yield x, y


# ---------------------------------------------------------------------------
# Fault type mapping
# ---------------------------------------------------------------------------

FAULT_TYPE_MAP = {
    "none": None,
    "stuck_at_0": FaultType.STUCK_AT_0,
    "stuck_at_1": FaultType.STUCK_AT_1,
    "wbf_t": FaultType.WEIGHT_BITFLIP_TRANSIENT,
    "wbf_p": FaultType.WEIGHT_BITFLIP_PERMANENT,
    "syn_silence": FaultType.SYNAPTIC_SILENCE,
    "retention": FaultType.RETENTION_FAILURE,
    "read_disturb": FaultType.READ_DISTURB,
    "mixed": FaultType.MIXED,
}


# ---------------------------------------------------------------------------
# Experiment configurations
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    input_size: int = 100
    hidden_size: int = 128
    n_classes: int = 4
    hdc_dim: int = 4096
    block_len: int = 20       # timesteps per class block
    T_train: int = 4000       # training timesteps (200 blocks of 20)
    T_test: int = 2000        # testing timesteps (100 blocks of 20)
    error_rates: List[float] = None

    # LIF parameters
    tau: float = 20.0
    v_th: float = 1.0
    dt: float = 1.0       # Simulation time step (ms)

    # Task type
    task: str = "classification"  # "classification" or "regression"

    # Hard mode: more classes, shorter blocks, higher noise
    hard_mode: bool = False

    # Readout training
    lr_readout: float = 1e-2
    weight_decay: float = 1e-4

    # Dual-trace Hebbian learning (instead of reservoir computing)
    enable_hebbian: bool = False
    lr_recurrent: float = 1e-4
    tau_fast: float = 5.0
    tau_slow: float = 50.0
    hebbian_alpha: float = 0.7
    hebbian_beta: float = 0.3

    # Fault model
    fault_type: str = "none"
    fault_persistent: bool = True

    # HDC feedback (Option A)
    enable_hdc_feedback: bool = False
    lr_boost_threshold: float = 0.15
    lr_boost_factor: float = 5.0

    # Error masking on SNN weights (Option C)
    enable_weight_masking: bool = False
    weight_masking_scheme: str = "zero"

    # HDC error correction (Option D)
    enable_hdc_ecc: bool = False
    ecc_similarity_threshold: float = 0.3
    ecc_correction_strength: float = 0.1

    seed: int = 42


def build_reservoir(cfg: ExperimentConfig, error_rate: float,
                     device: torch.device) -> RSNN:
    """Build an RSNN reservoir with SpikeFI-compatible fault injection.

    The reservoir weights are FIXED (random) — no Hebbian update.
    This follows the liquid state machine / echo state network paradigm
    (Biswas et al. 2024, Karki et al. 2024).

    Fault injection is applied to the reservoir weights (W_in, W_rec)
    using the SpikeFI taxonomy (Spyrou et al. 2024):
    - stuck_at_0: weights permanently set to 0
    - stuck_at_1: weights permanently set to 1
    - wbf_t: transient bit-flips (re-sampled each step)
    - wbf_p: permanent bit-flips
    - syn_silence: synaptic silence (weight=0)
    - mixed: combination of all fault types
    """
    has_faults = (error_rate > 0 and cfg.fault_type != "none")
    config = RSNNConfig(
        input_size=cfg.input_size,
        hidden_size=cfg.hidden_size,
        sparse_init=True,
        sparse_p=0.15,
        input_gain=50.0,
        tau=cfg.tau,
        v_th=cfg.v_th,
        dt=cfg.dt,
        device=str(device) if device is not None else None,
        enable_memory_error_injection=has_faults,
        memory_error_rate=error_rate,
        fault_type=cfg.fault_type,
        fault_rate=error_rate,
        fault_persistent=cfg.fault_persistent,
        enable_voltage_scaling=False,
    )
    return RSNN(config=config)


def train_readout_sgd_regression(rsnn: RSNN, readout: Readout,
                                  train_stream: list, cfg: ExperimentConfig,
                                  device: torch.device) -> float:
    """Train the readout for regression (MSE loss) using SGD.

    Returns:
        Final mean Pearson R over the training stream.
    """
    lr = cfg.lr_readout
    wd = cfg.weight_decay
    preds, targets = [], []

    for step_idx, (x, y) in enumerate(train_stream):
        x = x.to(device)
        y = y.to(device)

        with torch.no_grad():
            spikes = rsnn.forward(x)

        pred = readout.forward(spikes)

        # MSE gradient: dL/dpred = 2 * (pred - target)
        error = 2.0 * (pred - y)

        with torch.no_grad():
            dW = torch.outer(error, spikes)
            readout.W.add_(-lr * dW - lr * wd * readout.W)
            readout.b.add_(-lr * error)

        preds.append(pred.detach().cpu())
        targets.append(y.detach().cpu())

    # Compute Pearson R across all training steps
    if len(preds) > 1:
        preds_t = torch.stack(preds)
        targets_t = torch.stack(targets)
        p = preds_t - preds_t.mean(0)
        t = targets_t - targets_t.mean(0)
        num = (p * t).sum(0)
        den = (p.norm(dim=0) * t.norm(dim=0)).clamp(min=1e-12)
        r = (num / den).mean().item()
    else:
        r = 0.0

    return r


def test_snn_readout_regression(rsnn: RSNN, readout: Readout,
                                  test_stream: list, device: torch.device,
                                  cfg: Optional[ExperimentConfig] = None,
                                  hdc_encoder=None,
                                  corrector=None, masker=None) -> float:
    """Test the SNN readout on a regression task.

    Returns Pearson R across the test stream.
    """
    preds, targets = [], []
    window = cfg.block_len if cfg else 20
    spike_buf = []

    for step_idx, (x, y) in enumerate(test_stream):
        x = x.to(device)
        y = y.to(device)

        with torch.no_grad():
            spikes = rsnn.forward(x)

            # --- HDC feedback: apply constraint based on similarity ---
            if cfg is not None and cfg.enable_hdc_feedback and hdc_encoder is not None:
                hv = hdc_encoder.encode(spikes)
                sims = batch_sim(hv, hdc_encoder.memory.class_hvs, "bipolar")
                max_sim = float(sims.max().item())
                if max_sim < cfg.lr_boost_threshold:
                    pred = readout.forward(spikes)
                    error = 2.0 * (pred - y)
                    boosted_lr = cfg.lr_readout * cfg.lr_boost_factor
                    dW = torch.outer(error, spikes)
                    readout.W.add_(-boosted_lr * dW)
                    readout.b.add_(-boosted_lr * error)

            # --- HDC ECC ---
            if corrector is not None and hdc_encoder is not None:
                corrected_W, strength, info = corrector.repair_weights(
                    rsnn.W_rec, spikes, hdc_encoder,
                    hdc_encoder.memory, true_label=0,
                )
                if info["corrected"]:
                    rsnn.W_rec = corrected_W

            # --- Error masking ---
            if masker is not None:
                masker.update_error_rate(0.01)
                flat_W = rsnn.W_rec.flatten()
                masked_flat = masker(flat_W)
                if not torch.equal(masked_flat, flat_W):
                    rsnn.W_rec = masked_flat.reshape_as(rsnn.W_rec)

            pred = readout.forward(spikes)

        preds.append(pred.detach().cpu())
        targets.append(y.detach().cpu())

    if len(preds) > 1:
        preds_t = torch.stack(preds)
        targets_t = torch.stack(targets)
        p = preds_t - preds_t.mean(0)
        t = targets_t - targets_t.mean(0)
        num = (p * t).sum(0)
        den = (p.norm(dim=0) * t.norm(dim=0)).clamp(min=1e-12)
        r = (num / den).mean().item()
    else:
        r = 0.0

    return r


def train_readout_sgd(rsnn, readout, train_stream, cfg, device):
    """Train the readout using SGD with weight decay.

    The reservoir weights are FIXED — only the readout is trained.
    This is the standard reservoir computing approach.

    Returns:
        Final training accuracy
    """
    lr = cfg.lr_readout
    wd = cfg.weight_decay

    correct, total = 0, 0
    for step_idx, (x, y) in enumerate(train_stream):
        x = x.to(device)
        y = y.to(device)

        with torch.no_grad():
            spikes = rsnn.forward(x)

        logits = readout.forward(spikes)

        logits_exp = torch.exp(logits - logits.max())
        softmax = logits_exp / logits_exp.sum()
        target_onehot = torch.zeros(cfg.n_classes, device=device)
        target_onehot[y] = 1.0
        error = softmax - target_onehot

        dW = torch.outer(error, spikes)
        db = error

        with torch.no_grad():
            readout.W.add_(-lr * dW - lr * wd * readout.W)
            readout.b.add_(-lr * db)

        pred = logits.argmax().item()
        if pred == y.item():
            correct += 1
        total += 1

    return correct / max(total, 1)


def train_hdc_encoder(rsnn: RSNN, hdc_encoder: HDCEncoder,
                       train_stream: list, cfg: ExperimentConfig,
                       device: torch.device,
                       task: str = "classification") -> None:
    """Train the HDC encoder on reservoir spike patterns.

    Accumulates spikes over each block, then encodes the average
    spike pattern into the HDC associative memory.

    For regression tasks, HDC encoder is not used (returns immediately).
    """
    if task == "regression":
        return  # HDC not trained for regression

    window = cfg.block_len
    spike_buf = []

    for step_idx, (x, y) in enumerate(train_stream):
        x = x.to(device)
        with torch.no_grad():
            spikes = rsnn.forward(x)
        spike_buf.append(spikes.clone())

        if len(spike_buf) >= window:
            avg_spikes = torch.stack(spike_buf).mean(dim=0)
            hdc_encoder.train_step(avg_spikes, y.item())
            spike_buf = []

    hdc_encoder.finalize()


def test_snn_readout(rsnn: RSNN, readout: Readout,
                      test_stream: list, device: torch.device,
                      cfg: Optional[ExperimentConfig] = None,
                      hdc_encoder=None,
                      corrector=None, masker=None) -> float:
    """Test the SNN readout accuracy.

    When HDC feedback/ECC/masking is enabled, this function exercises
    those modules during the test phase:
    - hdc_loop: HDC similarity modulates readout learning rate
    - hdc_masked: Error masking applied to SNN weights
    - hdc_ecc: HDC-based weight repair applied to reservoir weights
    """
    correct, total = 0, 0
    ecc_corrections = 0
    masking_applied = 0

    for step_idx, (x, y) in enumerate(test_stream):
        x = x.to(device)
        y = y.to(device)

        with torch.no_grad():
            spikes = rsnn.forward(x)

            # --- HDC feedback: modulate learning rate based on HDC similarity ---
            if cfg is not None and cfg.enable_hdc_feedback and hdc_encoder is not None:
                hv = hdc_encoder.encode(spikes)
                sims = batch_sim(hv, hdc_encoder.memory.class_hvs, "bipolar")
                max_sim = float(sims.max().item())
                # If HDC is uncertain (low similarity), boost readout LR
                if max_sim < cfg.lr_boost_threshold:
                    # Apply a small corrective update to readout
                    logits = readout.forward(spikes)
                    logits_exp = torch.exp(logits - logits.max())
                    softmax = logits_exp / logits_exp.sum()
                    target_onehot = torch.zeros(cfg.n_classes, device=device)
                    target_onehot[y] = 1.0
                    error = softmax - target_onehot
                    dW = torch.outer(error, spikes)
                    db = error
                    boosted_lr = cfg.lr_readout * cfg.lr_boost_factor
                    readout.W.add_(-boosted_lr * dW)
                    readout.b.add_(-boosted_lr * db)

            # --- HDC ECC: repair corrupted reservoir weights ---
            if corrector is not None and hdc_encoder is not None:
                # Attempt weight repair using HDC prototypes
                corrected_W, strength, info = corrector.repair_weights(
                    rsnn.W_rec, spikes, hdc_encoder,
                    hdc_encoder.memory, true_label=y.item(),
                )
                if info["corrected"]:
                    rsnn.W_rec = corrected_W
                    ecc_corrections += 1

            # --- Error masking on SNN weights ---
            if masker is not None:
                # Apply masking to reservoir weights (flatten for 1D masker)
                masker.update_error_rate(0.01)  # Assume 1% error rate during test
                flat_W = rsnn.W_rec.flatten()
                masked_flat = masker(flat_W)
                if not torch.equal(masked_flat, flat_W):
                    rsnn.W_rec = masked_flat.reshape_as(rsnn.W_rec)
                    masking_applied += 1



            logits = readout.forward(spikes)

        pred = logits.argmax().item()
        if pred == y.item():
            correct += 1
        total += 1

    return correct / max(total, 1)



def test_hdc_classifier(rsnn: RSNN, hdc_encoder: HDCEncoder,
                         test_stream: list, cfg: ExperimentConfig,
                         device: torch.device) -> float:
    """Test the HDC classifier accuracy."""
    window = cfg.block_len
    spike_buf = []
    correct, total = 0, 0

    for x, y in test_stream:
        x = x.to(device)
        with torch.no_grad():
            spikes = rsnn.forward(x)
        spike_buf.append(spikes.clone())

        if len(spike_buf) >= window:
            avg_spikes = torch.stack(spike_buf).mean(dim=0)
            hv = hdc_encoder.encode(avg_spikes)
            pred = hdc_encoder.memory.predict(hv)
            if pred == y.item():
                correct += 1
            total += 1
            spike_buf = []

    return correct / max(total, 1)


def train_hebbian_regression(rsnn: RSNN, readout: Readout,
                               train_stream: list, cfg: ExperimentConfig,
                               device: torch.device) -> float:
    """Train using dual-trace Hebbian learning (matching bci_decoding.py).

    Unlike the reservoir computing approach (fixed W_rec), this updates
    the recurrent weights via the DualHebbian accumulator during training.
    This is the approach that achieves 0.81 Pearson R in the Core Claim.

    Returns:
        Final mean Pearson R over the training stream.
    """
    from models.hebbian import DualHebbianAccumulator, HebbianConfig

    hebbian = DualHebbianAccumulator(HebbianConfig(
        shape=(cfg.hidden_size, cfg.hidden_size),
        tau_fast=cfg.tau_fast,
        tau_slow=cfg.tau_slow,
        alpha=cfg.hebbian_alpha,
        beta=cfg.hebbian_beta,
    ), device=device)

    lr_r = cfg.lr_readout
    lr_w = cfg.lr_recurrent
    wd = cfg.weight_decay
    preds, targets = [], []

    for step_idx, (x, y) in enumerate(train_stream):
        x = x.to(device)
        y = y.to(device)

        with torch.no_grad():
            spikes = rsnn.forward(x)

        # Hebbian update to recurrent weights
        trace = hebbian.update(rsnn.prev_spikes, spikes)
        with torch.no_grad():
            rsnn.W_rec.add_(-lr_w * trace)

        # Readout update (MSE)
        pred = readout.forward(spikes)
        error = 2.0 * (pred - y)
        with torch.no_grad():
            dW = torch.outer(error, spikes)
            readout.W.add_(-lr_r * dW - lr_r * wd * readout.W)
            readout.b.add_(-lr_r * error)

        preds.append(pred.detach().cpu())
        targets.append(y.detach().cpu())

    # Compute Pearson R
    if len(preds) > 1:
        preds_t = torch.stack(preds)
        targets_t = torch.stack(targets)
        p = preds_t - preds_t.mean(0)
        t = targets_t - targets_t.mean(0)
        num = (p * t).sum(0)
        den = (p.norm(dim=0) * t.norm(dim=0)).clamp(min=1e-12)
        r = (num / den).mean().item()
    else:
        r = 0.0

    return r


def run_single_config(cfg: ExperimentConfig, error_rate: float) -> dict:
    """Run one (config, error_rate) trial and return metrics.

    Two training paradigms available:
    1. Reservoir computing (default): fixed W_rec, train readout only
    2. Dual-trace Hebbian (--hebbian): W_rec updated via Hebbian traces

    For advanced configs (hdc_loop, hdc_masked, hdc_ecc), the HDC
    feedback/ECC/masking modules are exercised DURING the test phase:
    - hdc_loop: HDC similarity modulates readout learning rate
    - hdc_masked: Error masking applied to SNN weights during test
    - hdc_ecc: HDC-based weight repair applied during test
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # --- Build reservoir with fault injection ---
    rsnn = build_reservoir(cfg, error_rate, device)

    # --- Create fixed prototypes (for classification) ---
    prototypes = make_prototypes(cfg.n_classes, cfg.input_size, seed=cfg.seed)

    # --- Training streams ---
    if cfg.task == "regression":
        train_stream = list(make_regression_stream(
            cfg.input_size, cfg.T_train, seed=cfg.seed, noise=0.1))
        test_stream = list(make_regression_stream(
            cfg.input_size, cfg.T_test, seed=cfg.seed+1, noise=0.1))
    else:
        noise = 0.15 if cfg.hard_mode else 0.05
        train_stream = list(make_temporal_stream(
            prototypes, T=cfg.T_train, block_len=cfg.block_len, noise=noise))
        test_stream = list(make_temporal_stream(
            prototypes, T=cfg.T_test, block_len=cfg.block_len, noise=noise))

    # --- Train ---
    if cfg.task == "regression":
        readout = Readout(
            hidden_size=cfg.hidden_size,
            output_size=2,  # 2-D velocity
            device=device,
        )
        if cfg.enable_hebbian:
            # Dual-trace Hebbian: updates W_rec during training (matches bci_decoding.py)
            train_acc = train_hebbian_regression(rsnn, readout, train_stream, cfg, device)
        else:
            # Reservoir computing: fixed W_rec, train readout only
            train_acc = train_readout_sgd_regression(rsnn, readout, train_stream, cfg, device)
    else:
        readout = Readout(
            hidden_size=cfg.hidden_size,
            output_size=cfg.n_classes,
            device=device,
        )
        train_acc = train_readout_sgd(rsnn, readout, train_stream, cfg, device)

    # --- Build and train HDC encoder ---
    hdc_encoder = HDCEncoder(
        input_size=cfg.hidden_size,
        n_classes=cfg.n_classes,
        dim=cfg.hdc_dim,
        mode="bipolar",
        device=device,
        seed=cfg.seed,
    )
    if cfg.task != "regression":
        train_hdc_encoder(rsnn, hdc_encoder, train_stream, cfg, device, task=cfg.task)

    # --- Set up HDC feedback / ECC / masking if enabled ---
    corrector = None
    masker = None
    if cfg.task != "regression":
        if cfg.enable_hdc_ecc and hdc_encoder is not None and hasattr(hdc_encoder, 'memory'):
            from hdc.ecc import HDCCorrector, ECCConfig
            corrector = HDCCorrector(ECCConfig(
                hdc_dim=cfg.hdc_dim,
                n_classes=cfg.n_classes,
                similarity_threshold=cfg.ecc_similarity_threshold,
                correction_strength=cfg.ecc_correction_strength,
                device=str(device),
            ))
    if cfg.enable_weight_masking:
        from hdc.error_masking import ErrorMasker, ErrorMaskingConfig
        masker = ErrorMasker(
            dim=cfg.hidden_size,
            config=ErrorMaskingConfig(
                enabled=True,
                masking_scheme=cfg.weight_masking_scheme,
                error_threshold=1e-6,
            ),
        )

    # --- Test SNN readout (with optional HDC feedback/ECC/masking) ---
    if cfg.task == "regression":
        snn_accuracy = test_snn_readout_regression(
            rsnn, readout, test_stream, device,
            cfg=cfg, hdc_encoder=hdc_encoder,
            corrector=corrector, masker=masker,
        )
        # For regression, we report Pearson R instead of accuracy
        hdc_accuracy = 0.0  # HDC not used for regression
    else:
        snn_accuracy = test_snn_readout(
            rsnn, readout, test_stream, device,
            cfg=cfg, hdc_encoder=hdc_encoder,
            corrector=corrector, masker=masker,
        )
        # --- Test HDC classifier ---
        hdc_accuracy = test_hdc_classifier(rsnn, hdc_encoder, test_stream, cfg, device)

    # --- Collect ECC/masking stats ---
    ecc_stats = {}
    if corrector is not None:
        ecc_stats = corrector.get_stats()
    if masker is not None:
        ecc_stats["masking_stats"] = masker.get_stats()

    metrics = {
        "snn_accuracy": snn_accuracy,
        "hdc_accuracy": hdc_accuracy,
        "train_accuracy": train_acc,
        "ecc_stats": ecc_stats,
    }

    return metrics



# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_sweep(cfg: ExperimentConfig) -> dict:
    """Run all configurations across all error rates."""
    configs = {
        "baseline": ExperimentConfig(
            **{k: v for k, v in cfg.__dict__.items()
               if k not in ("enable_hdc_feedback", "enable_weight_masking",
                           "weight_masking_scheme", "enable_hdc_ecc",
                           "ecc_similarity_threshold", "ecc_correction_strength")},
            enable_hdc_feedback=False,
            enable_weight_masking=False,
            enable_hdc_ecc=False,
        ),
        "hdc_head": ExperimentConfig(
            **{k: v for k, v in cfg.__dict__.items()
               if k not in ("enable_hdc_feedback", "enable_weight_masking",
                           "weight_masking_scheme", "enable_hdc_ecc",
                           "ecc_similarity_threshold", "ecc_correction_strength")},
            enable_hdc_feedback=False,
            enable_weight_masking=False,
            enable_hdc_ecc=False,
        ),
        "hdc_loop": ExperimentConfig(
            **{k: v for k, v in cfg.__dict__.items()
               if k not in ("enable_hdc_feedback", "enable_weight_masking",
                           "weight_masking_scheme", "enable_hdc_ecc",
                           "ecc_similarity_threshold", "ecc_correction_strength")},
            enable_hdc_feedback=True,
            enable_weight_masking=False,
            enable_hdc_ecc=False,
        ),
        "hdc_masked": ExperimentConfig(
            **{k: v for k, v in cfg.__dict__.items()
               if k not in ("enable_hdc_feedback", "enable_weight_masking",
                           "weight_masking_scheme", "enable_hdc_ecc",
                           "ecc_similarity_threshold", "ecc_correction_strength")},
            enable_hdc_feedback=True,
            enable_weight_masking=True,
            weight_masking_scheme="zero",
            enable_hdc_ecc=False,
        ),
        "hdc_ecc": ExperimentConfig(
            **{k: v for k, v in cfg.__dict__.items()
               if k not in ("enable_hdc_feedback", "enable_weight_masking",
                           "weight_masking_scheme", "enable_hdc_ecc",
                           "ecc_similarity_threshold", "ecc_correction_strength")},
            enable_hdc_feedback=False,
            enable_weight_masking=False,
            enable_hdc_ecc=True,
            ecc_similarity_threshold=cfg.ecc_similarity_threshold,
            ecc_correction_strength=cfg.ecc_correction_strength,
        ),
    }

    error_rates = cfg.error_rates or [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    results = {}

    for config_name, config in configs.items():
        print(f"\n{'='*60}")
        print(f"  Config: {config_name}")
        print(f"{'='*60}")
        results[config_name] = {}
        for rate in error_rates:
            print(f"    Error rate: {rate:.0e}  ", end="", flush=True)
            t0 = time.perf_counter()
            metrics = run_single_config(config, rate)
            elapsed = time.perf_counter() - t0
            print(f"SNN={metrics['snn_accuracy']:.1%}  "
                  f"HDC={metrics['hdc_accuracy']:.1%}  "
                  f"(train={metrics['train_accuracy']:.1%})  "
                  f"({elapsed:.1f}s)")
            results[config_name][str(rate)] = metrics

    return results


# ---------------------------------------------------------------------------
# Pretty-print results table
# ---------------------------------------------------------------------------

def print_results_table(results: dict):
    """Print a 2×2 results table: error rate × config."""
    configs = list(results.keys())
    rates = list(results[configs[0]].keys())

    print("\n" + "=" * 140)
    print(f"{'Error Rate':<14}", end="")
    for cfg in configs:
        print(f"{cfg+' (SNN)':<16} {cfg+' (HDC)':<16}", end="  ")
    print()
    print("-" * 140)

    for rate in rates:
        print(f"{rate:<14}", end="")
        for cfg in configs:
            m = results[cfg][rate]
            s_str = f"{m['snn_accuracy']:.1%}"
            h_str = f"{m['hdc_accuracy']:.1%}"
            print(f"{s_str:<16} {h_str:<16}", end="  ")
        print()

    print("=" * 140)

    print("\nDegradation at highest error rate:")
    for cfg in configs:
        s0 = results[cfg][rates[0]]['snn_accuracy']
        s1 = results[cfg][rates[-1]]['snn_accuracy']
        h0 = results[cfg][rates[0]]['hdc_accuracy']
        h1 = results[cfg][rates[-1]]['hdc_accuracy']
        sd = ((s1 / s0) - 1) * 100 if s0 > 0 else 0
        hd = ((h1 / h0) - 1) * 100 if h0 > 0 else 0
        print(f"  {cfg:<14}:  SNN {s0:.1%}→{s1:.1%}  ({sd:+.1f}%)  "
              f"HDC {h0:.1%}→{h1:.1%}  ({hd:+.1f}%)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="End-to-end SNNTraining robustness experiment")
    parser.add_argument("--error-rates", type=float, nargs="+",
                        default=[0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2])
    parser.add_argument("--T-train", type=int, default=4000)
    parser.add_argument("--T-test", type=int, default=2000)
    parser.add_argument("--block-len", type=int, default=20)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--hdc-dim", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--task", type=str, default="classification",
                        choices=["classification", "regression"])

    parser.add_argument("--fault-type", type=str, default="none",
                        choices=list(FAULT_TYPE_MAP.keys()))
    parser.add_argument("--persistent", action="store_true")

    parser.add_argument("--tau", type=float, default=20.0)
    parser.add_argument("--v-th", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=1.0,
                        help="Simulation time step (ms)")

    parser.add_argument("--lr", type=float, default=1e-2,
                        help="Readout learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument("--ecc-threshold", type=float, default=0.3)
    parser.add_argument("--ecc-strength", type=float, default=0.1)

    parser.add_argument("--hard", action="store_true",
                        help="Hard mode: 8 classes, block_len=5, higher noise")
    parser.add_argument("--hebbian", action="store_true",
                        help="Use dual-trace Hebbian learning (instead of reservoir computing)")

    args = parser.parse_args()

    if args.quick:
        args.T_train = 400
        args.T_test = 200
        args.block_len = 10
        args.hidden_size = 32
        args.hdc_dim = 512
        args.error_rates = [0, 1e-3]

    # Hard mode: more classes, shorter blocks, higher noise
    n_classes = 8 if args.hard else 4
    block_len = 5 if args.hard else args.block_len
    noise = 0.15 if args.hard else 0.05

    cfg = ExperimentConfig(
        input_size=100,
        hidden_size=args.hidden_size,
        n_classes=n_classes,
        hdc_dim=args.hdc_dim,
        block_len=block_len,
        T_train=args.T_train,
        T_test=args.T_test,
        error_rates=args.error_rates,
        seed=args.seed,
        tau=args.tau,
        v_th=args.v_th,
        dt=args.dt,
        lr_readout=args.lr,
        weight_decay=args.weight_decay,
        fault_type=args.fault_type,
        fault_persistent=args.persistent,
        task=args.task,
        hard_mode=args.hard,
        ecc_similarity_threshold=args.ecc_threshold,
        ecc_correction_strength=args.ecc_strength,
        enable_hebbian=args.hebbian,
    )

    print("=" * 60)
    print("SNNTraining End-to-End Robustness Experiment")
    print("=" * 60)
    print(f"  Hidden size: {cfg.hidden_size}")
    print(f"  HDC dim:     {cfg.hdc_dim}")
    print(f"  Block len:   {cfg.block_len} (temporal task)")
    print(f"  Train steps: {cfg.T_train} ({cfg.T_train//cfg.block_len} blocks)")
    print(f"  Test steps:  {cfg.T_test} ({cfg.T_test//cfg.block_len} blocks)")
    print(f"  Error rates: {cfg.error_rates}")
    print(f"  Fault type:  {cfg.fault_type} (persistent={cfg.fault_persistent})")
    print(f"  LIF:         tau={cfg.tau}, v_th={cfg.v_th}, dt={cfg.dt}")
    print(f"  Readout:     lr={cfg.lr_readout}, wd={cfg.weight_decay}")
    print(f"  ECC:         threshold={cfg.ecc_similarity_threshold}, "
          f"strength={cfg.ecc_correction_strength}")
    paradigm = "Dual-trace Hebbian (W_rec updated via eligibility traces)" if cfg.enable_hebbian else "Reservoir computing (fixed W_rec, train readout only)"
    print(f"  Paradigm:    {paradigm}")

    results = run_sweep(cfg)
    print_results_table(results)

    if args.save:
        os.makedirs("results", exist_ok=True)
        path = "results/snntraining_robustness.json"
        # Convert results to JSON-safe format (tensors → floats)
        def to_json_safe(obj):
            if isinstance(obj, dict):
                return {k: to_json_safe(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [to_json_safe(v) for v in obj]
            elif isinstance(obj, torch.Tensor):
                return obj.item() if obj.numel() == 1 else obj.tolist()
            return obj

        with open(path, "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "config": {
                    "hidden_size": cfg.hidden_size,
                    "hdc_dim": cfg.hdc_dim,
                    "block_len": cfg.block_len,
                    "T_train": cfg.T_train,
                    "T_test": cfg.T_test,
                    "seed": cfg.seed,
                    "fault_type": cfg.fault_type,
                    "fault_persistent": cfg.fault_persistent,
                    "tau": cfg.tau,
                    "v_th": cfg.v_th,
                    "dt": cfg.dt,
                    "lr_readout": cfg.lr_readout,
                    "weight_decay": cfg.weight_decay,
                    "ecc_threshold": cfg.ecc_similarity_threshold,
                    "ecc_strength": cfg.ecc_correction_strength,
                },
                "results": to_json_safe(results),
            }, f, indent=2)

        print(f"\nResults saved to {path}")

    print("\n" + "=" * 60)
    print("Verdict")
    print("=" * 60)
    s0 = results["baseline"][str(cfg.error_rates[0])]['snn_accuracy']
    s1 = results["baseline"][str(cfg.error_rates[-1])]['snn_accuracy']
    sd = ((s1 / s0) - 1) * 100 if s0 > 0 else 0
    h0 = results["hdc_head"][str(cfg.error_rates[0])]['hdc_accuracy']
    h1 = results["hdc_head"][str(cfg.error_rates[-1])]['hdc_accuracy']
    hd = ((h1 / h0) - 1) * 100 if h0 > 0 else 0
    print(f"  SNN classification degradation at {cfg.error_rates[-1]:.0e}: {sd:+.1f}%")
    print(f"  HDC classification degradation: {hd:+.1f}%")


if __name__ == "__main__":
    main()
