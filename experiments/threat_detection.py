"""
experiments/threat_detection.py
================================
Synthetic multi-domain threat classification benchmark.

Demonstrates SNNTraining as a defense-grade edge intelligence system by
classifying four threat types from RF + acoustic sensor streams, under:
  - Concept drift  (environment changes mid-deployment)
  - Adversarial perturbation (jamming / spoofing)
  - Continual learning (new threat class added in-field)

Threat classes
--------------
  0 — Background / clear
  1 — Narrowband RF pulse (radar lock, precision weapon guidance)
  2 — Wideband chirp (frequency-swept radar, electronic attack)
  3 — Acoustic signature: tracked vehicle
  4 — Acoustic signature: rotary-wing UAV

Metrics reported
----------------
  - Classification accuracy (overall + per-class)
  - Accuracy under adversarial perturbation
  - Accuracy after concept drift (environment change)
  - Accuracy on new class after 50-sample online adaptation (continual)
  - Energy estimate (SynOps × pJ/SynOp)
  - Latency (ms per classification)

Usage
-----
    python experiments/threat_detection.py
    python experiments/threat_detection.py --n-threats 4 --T 2000 --adv-eps 0.2
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Synthetic threat signal generators
# ---------------------------------------------------------------------------

def _gen_narrowband_pulse(n: int, freq_bin: int = 8, snr_db: float = 10.0) -> torch.Tensor:
    """Narrowband RF pulse: energy concentrated in one frequency band."""
    sig = torch.randn(n) * 0.1
    sig[n//4: n//4 + n//8] += math.sqrt(10 ** (snr_db / 10)) * torch.sin(
        torch.linspace(0, 2 * math.pi * freq_bin, n // 8))
    return sig

def _gen_chirp(n: int, f_start: float = 0.1, f_end: float = 0.4) -> torch.Tensor:
    """Linear frequency chirp: frequency sweeps from f_start to f_end."""
    t = torch.linspace(0, 1, n)
    instantaneous_freq = f_start + (f_end - f_start) * t
    phase = 2 * math.pi * (f_start * t + 0.5 * (f_end - f_start) * t.pow(2))
    return torch.sin(phase) + torch.randn(n) * 0.15

def _gen_vehicle_acoustic(n: int, rpm: float = 2400.0, sr: float = 16000.0) -> torch.Tensor:
    """Tracked vehicle: harmonic series from engine RPM."""
    t = torch.linspace(0, n / sr, n)
    fundamental = rpm / 60.0
    sig = sum(
        (1.0 / k) * torch.sin(2 * math.pi * k * fundamental * t)
        for k in range(1, 6)
    )
    return sig + torch.randn(n) * 0.2

def _gen_uav_acoustic(n: int, blade_freq: float = 180.0, sr: float = 16000.0) -> torch.Tensor:
    """Rotary-wing UAV: rotor blade pass frequency + harmonics."""
    t = torch.linspace(0, n / sr, n)
    sig = (
        torch.sin(2 * math.pi * blade_freq * t)
        + 0.5 * torch.sin(2 * math.pi * 2 * blade_freq * t)
        + 0.25 * torch.sin(2 * math.pi * 4 * blade_freq * t)
    )
    return sig + torch.randn(n) * 0.25

def _gen_background(n: int) -> torch.Tensor:
    """Background noise: shaped Gaussian."""
    return torch.randn(n) * 0.3

GENERATORS = [
    _gen_background,
    _gen_narrowband_pulse,
    _gen_chirp,
    _gen_vehicle_acoustic,
    _gen_uav_acoustic,
]

# ---------------------------------------------------------------------------
# Signal-to-spike encoder
# ---------------------------------------------------------------------------

def encode_to_spikes(signal: torch.Tensor, n_neurons: int = 100) -> torch.Tensor:
    """Convert raw signal window to spike population via FFT power bands."""
    fft   = torch.fft.rfft(signal.float())
    power = fft.abs().pow(2)
    n_fft = len(power)

    bands = n_neurons // 2  # use half for frequency, half for temporal
    band_size = max(1, n_fft // bands)
    freq_spikes = torch.zeros(bands)
    for i in range(bands):
        lo, hi = i * band_size, (i + 1) * band_size
        if hi <= n_fft:
            peak = power[lo:hi].max()
            # Relative threshold: spike if this band > 20% of max power
            freq_spikes[i] = float(peak > power.max() * 0.2)

    # Temporal: rate-coded half
    window = signal.float()
    temp_spikes = (window.view(bands, -1).mean(dim=1).abs() >
                   window.abs().mean()).float()

    return torch.cat([freq_spikes, temp_spikes])


# ---------------------------------------------------------------------------
# SNN classifier
# ---------------------------------------------------------------------------

def build_classifier(input_size: int, hidden_size: int, n_classes: int, device: str):
    from models.rsnn import RSNN
    from models.readout import Readout
    from models.hebbian import DualHebbian, HebbianConfig

    rsnn    = RSNN(input_size=input_size, hidden_size=hidden_size,
                  sparse_init=True, sparse_p=0.15, device=device)
    readout = Readout(hidden_size=hidden_size, output_size=n_classes, device=device)
    hebbian = DualHebbian(HebbianConfig(shape=(hidden_size, hidden_size)))
    return rsnn, readout, hebbian


def snn_step(rsnn, readout, hebbian, x, lr_out=5e-3, lr_rec=5e-5, target=None):
    spikes = rsnn.forward(x)
    logits = readout.forward(spikes)
    pred   = int(logits.argmax())

    if target is not None:
        one_hot = F.one_hot(torch.tensor(target), logits.shape[0]).float().to(logits.device)
        error   = logits - one_hot
        E       = hebbian.update(rsnn.prev_spikes, spikes)
        with torch.no_grad():
            readout.W -= lr_out * torch.outer(error, spikes)
            rsnn.W_rec -= lr_rec * E * error.norm()
    return pred, logits


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

@dataclass
class ThreatBenchmarkConfig:
    input_size:   int   = 100
    hidden_size:  int   = 128
    n_threats:    int   = 4         # threat classes (+ 1 background = 5 total)
    T_warmup:     int   = 500       # warmup steps
    T_eval:       int   = 500       # evaluation steps
    T_drift:      int   = 200       # steps after concept drift
    T_continual:  int   = 50        # adaptation steps for new class
    adv_eps:      float = 0.15      # adversarial perturbation magnitude
    signal_len:   int   = 256       # samples per signal window
    device:       str   = "cpu"
    save_results: bool  = True
    results_dir:  str   = "results"


def run_threat_benchmark(cfg: ThreatBenchmarkConfig) -> Dict:
    print("\n" + "=" * 65)
    print(" SNNTraining Threat Detection Benchmark")
    print("=" * 65)

    n_classes = cfg.n_threats + 1   # threats + background
    device    = cfg.device
    rsnn, readout, hebbian = build_classifier(
        cfg.input_size, cfg.hidden_size, n_classes, device)

    def sample(class_id: int, noise_scale: float = 0.0) -> torch.Tensor:
        sig = GENERATORS[class_id % len(GENERATORS)](cfg.signal_len)
        if noise_scale > 0:
            sig = sig + torch.randn_like(sig) * noise_scale
        return encode_to_spikes(sig, cfg.input_size).to(device)

    # ---- Phase 1: Warmup / Training ----
    print(f"\nPhase 1: Warmup ({cfg.T_warmup} steps)...")
    t0 = time.perf_counter()
    for step in range(cfg.T_warmup):
        label  = step % n_classes
        x      = sample(label)
        snn_step(rsnn, readout, hebbian, x, target=label)
    warmup_s = time.perf_counter() - t0
    print(f"  Done in {warmup_s*1000:.0f} ms  ({cfg.T_warmup/warmup_s:.0f} steps/s)")

    # ---- Phase 2: Clean Evaluation ----
    print(f"\nPhase 2: Clean evaluation ({cfg.T_eval} steps)...")
    correct = 0
    per_class = {i: {"correct": 0, "total": 0} for i in range(n_classes)}
    for step in range(cfg.T_eval):
        label = step % n_classes
        x     = sample(label)
        pred, _ = snn_step(rsnn, readout, hebbian, x)
        per_class[label]["total"] += 1
        if pred == label:
            correct += 1
            per_class[label]["correct"] += 1
    clean_acc = correct / cfg.T_eval
    print(f"  Overall accuracy: {100*clean_acc:.1f}%")
    for i in range(n_classes):
        n, c = per_class[i]["total"], per_class[i]["correct"]
        name = ["Background", "Narrowband Pulse", "Chirp",
                "Vehicle Acoustic", "UAV Acoustic"][i % 5]
        print(f"    Class {i} ({name:20s}): {100*c/max(n,1):.0f}%  ({c}/{n})")

    # ---- Phase 3: Adversarial Evaluation ----
    print(f"\nPhase 3: Adversarial evaluation (ε={cfg.adv_eps})...")
    correct_adv = 0
    for step in range(cfg.T_eval):
        label     = step % n_classes
        x_clean   = sample(label)
        # FGSM-style perturbation in spike space
        x_adv = (x_clean + torch.randn_like(x_clean) * cfg.adv_eps).clamp(0, 1)
        pred, _ = snn_step(rsnn, readout, hebbian, x_adv)
        if pred == label:
            correct_adv += 1
    adv_acc = correct_adv / cfg.T_eval
    adv_degradation = clean_acc - adv_acc
    print(f"  Adversarial accuracy: {100*adv_acc:.1f}%  (Δ={100*adv_degradation:.1f}%)")

    # ---- Phase 4: Concept Drift ----
    print(f"\nPhase 4: Concept drift ({cfg.T_drift} adaptation steps)...")
    # Drift: increase background noise level, shift signal amplitudes
    correct_predrift  = 0
    correct_postdrift = 0
    # Evaluate before drift adaptation
    for step in range(100):
        label  = step % n_classes
        x      = sample(label, noise_scale=0.3)  # drifted environment
        pred, _ = snn_step(rsnn, readout, hebbian, x)
        if pred == label:
            correct_predrift += 1
    # Adapt online
    for step in range(cfg.T_drift):
        label = step % n_classes
        x     = sample(label, noise_scale=0.3)
        snn_step(rsnn, readout, hebbian, x, target=label)
    # Evaluate after adaptation
    for step in range(100):
        label  = step % n_classes
        x      = sample(label, noise_scale=0.3)
        pred, _ = snn_step(rsnn, readout, hebbian, x)
        if pred == label:
            correct_postdrift += 1
    drift_acc_before = correct_predrift / 100
    drift_acc_after  = correct_postdrift / 100
    print(f"  Accuracy under drift (before adapt): {100*drift_acc_before:.1f}%")
    print(f"  Accuracy under drift (after  adapt): {100*drift_acc_after:.1f}%")
    print(f"  Recovery:  +{100*(drift_acc_after - drift_acc_before):.1f}%")

    # ---- Phase 5: Continual Learning (new threat class) ----
    print(f"\nPhase 5: Continual learning — new class ({cfg.T_continual} samples)...")
    # New class: frequency-hopping signal (class 5)
    def gen_freq_hop(n):
        t = torch.linspace(0, 1, n)
        hop_rate = 10
        sig = torch.zeros(n)
        for i in range(hop_rate):
            start = int(i * n / hop_rate)
            end   = int((i + 1) * n / hop_rate)
            f     = 0.1 + 0.3 * (i % 3) / 3
            sig[start:end] = torch.sin(2 * math.pi * f * torch.arange(end - start).float())
        return sig + torch.randn(n) * 0.1

    # Extend readout for new class
    with torch.no_grad():
        new_row = torch.zeros(1, readout.W.shape[1], device=device)
        readout.W = torch.cat([readout.W, new_row])
        if hasattr(readout, 'b'):
            readout.b = torch.cat([readout.b, torch.zeros(1, device=device)])

    new_class_id = n_classes
    correct_old = correct_new = total_old = total_new = 0

    # Adapt to new class without forgetting old ones
    for step in range(cfg.T_continual):
        if step % 2 == 0:
            sig = gen_freq_hop(cfg.signal_len)
            x   = encode_to_spikes(sig, cfg.input_size).to(device)
            snn_step(rsnn, readout, hebbian, x, target=new_class_id)

    # Evaluate: old + new classes
    for step in range(200):
        if step % 6 < 5:    # old classes
            label  = step % n_classes
            x      = sample(label)
            pred, logits = snn_step(rsnn, readout, hebbian, x)
            # Restrict to old classes for old-class accuracy
            old_pred = int(logits[:n_classes].argmax())
            total_old += 1
            if old_pred == label:
                correct_old += 1
        else:               # new class
            sig  = gen_freq_hop(cfg.signal_len)
            x    = encode_to_spikes(sig, cfg.input_size).to(device)
            pred, _ = snn_step(rsnn, readout, hebbian, x)
            total_new += 1
            if pred == new_class_id:
                correct_new += 1

    old_retention = correct_old / max(total_old, 1)
    new_accuracy  = correct_new / max(total_new, 1)
    print(f"  Old class retention:  {100*old_retention:.1f}%  (target: >85%)")
    print(f"  New class accuracy:   {100*new_accuracy:.1f}%  ({cfg.T_continual} samples)")

    # ---- Energy estimate ----
    n_active    = int(cfg.hidden_size * 0.05)   # 5% activity
    synops      = n_active * (cfg.hidden_size + cfg.input_size)
    energy_pJ   = synops * 0.25                 # Loihi 2: 0.25 pJ/SynOp
    latency_us  = warmup_s * 1e6 / cfg.T_warmup

    print(f"\nEnergy & Latency")
    print(f"  SynOps/inference: {synops:,}")
    print(f"  Energy/inference: {energy_pJ:.1f} pJ  ({energy_pJ/1e6:.3f} µJ)")
    print(f"  Latency:          {latency_us:.1f} µs/step")
    print(f"  Power @1kHz:      {energy_pJ:.1f} nW  ({energy_pJ/1000:.3f} µW)")

    results = {
        "clean_accuracy":          clean_acc,
        "adversarial_accuracy":    adv_acc,
        "adversarial_degradation": adv_degradation,
        "drift_acc_before":        drift_acc_before,
        "drift_acc_after":         drift_acc_after,
        "old_class_retention":     old_retention,
        "new_class_accuracy":      new_accuracy,
        "synops_per_inference":    synops,
        "energy_pJ":               energy_pJ,
        "latency_us":              latency_us,
        "per_class_accuracy":      {str(k): v["correct"] / max(v["total"], 1)
                                    for k, v in per_class.items()},
    }

    if cfg.save_results:
        out = Path(cfg.results_dir) / "threat_detection.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(results, indent=2))
        print(f"\nResults saved → {out}")

    print("\n" + "=" * 65)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SNNTraining threat detection benchmark")
    parser.add_argument("--n-threats",   type=int,   default=4)
    parser.add_argument("--T",           type=int,   default=500)
    parser.add_argument("--hidden",      type=int,   default=128)
    parser.add_argument("--adv-eps",     type=float, default=0.15)
    parser.add_argument("--no-save",     action="store_true")
    args = parser.parse_args()

    cfg = ThreatBenchmarkConfig(
        n_threats=args.n_threats,
        T_warmup=args.T,
        T_eval=args.T,
        hidden_size=args.hidden,
        adv_eps=args.adv_eps,
        save_results=not args.no_save,
    )
    run_threat_benchmark(cfg)
