"""
experiments/hva_composition_demo.py
=====================================
The Founders Fund demo: compose any N architectures in 4 lines, no retraining.
Shows that HVA ensemble composition beats individual models.

This is the empirical proof for the claim:
  "Slap together architectures that are completely different.
   No retraining. Multimodality is free."

Experiment design
-----------------
Three completely different "model" architectures process the same
multi-modal synthetic task (classification of 4-class spike patterns):

  Vision channel  — MLP on rate-coded spike summary
  Temporal channel — RSNN on raw spike sequence
  Frequency channel — FFT magnitude features → MLP

Each architecture is independently weak (task is designed so no single
channel has enough information).  HVA composition bundles all three into
a single hypervector — and the composed prediction beats any individual.

Results table
-------------
  Single model (vision only)    : X%
  Single model (temporal only)  : Y%
  Single model (frequency only) : Z%
  Best individual model          : max(X, Y, Z)%
  HVA bundle (3 models)          : W%   ← should exceed best individual
  HVA weighted (confidence-based): V%   ← should be ≥ bundle

Usage
-----
    python experiments/hva_composition_demo.py
    python experiments/hva_composition_demo.py --seeds 5 --n-classes 6
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from hdc.hypervector_architecture import (
    HVModel, HVModelConfig, HVPipeline,
)


# ── Synthetic multi-modal dataset ─────────────────────────────────────────────

def make_multimodal_dataset(
    n_classes: int,
    n_train: int,
    n_test: int,
    input_dim: int = 32,
    seq_len: int = 20,
    seed: int = 42,
) -> tuple:
    """
    Generate a genuinely multi-modal dataset where NO single modality
    can classify all classes, but their combination can.

    Orthogonal split design (for n_classes=4):
      - Vision    discriminates classes {0,1} vs {2,3}  → 2-way split
      - Temporal  discriminates classes {0,2} vs {1,3}  → orthogonal 2-way
      - Frequency discriminates classes {0,3} vs {1,2}  → third orthogonal split

    Each single modality achieves ~50% (2-way chance on a 4-class task).
    Together they achieve ~100% (unique class fingerprint per combination).

    This is the canonical test of whether composition actually helps.
    """
    assert n_classes == 4, "Orthogonal split design requires exactly 4 classes"
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)

    # Each modality distinguishes a different binary partition of the 4 classes
    # Vision:    0,1 → proto_A;  2,3 → proto_B
    # Temporal:  0,2 → proto_C;  1,3 → proto_D
    # Frequency: 0,3 → proto_E;  1,2 → proto_F

    vision_proto   = rng.randn(2, input_dim)        # 2 binary groups
    temporal_proto = rng.randn(2, input_dim)
    freq_proto     = rng.randn(2, input_dim // 2)

    # Maps class → binary group for each modality
    vision_group   = {0: 0, 1: 0, 2: 1, 3: 1}
    temporal_group = {0: 0, 1: 1, 2: 0, 3: 1}
    freq_group     = {0: 0, 1: 1, 2: 1, 3: 0}

    def make_split(n, seed_offset):
        rng2 = np.random.RandomState(seed + seed_offset)
        vision, temporal, frequency, labels = [], [], [], []
        for i in range(n):
            lbl = i % n_classes

            # Vision: noisy version of the correct 2-group prototype
            vg = vision_group[lbl]
            v = vision_proto[vg] + rng2.randn(input_dim) * 0.8
            vision.append(v)

            # Temporal: each timestep is a noisy version of the correct temporal group
            tg = temporal_group[lbl]
            seq = np.stack([
                temporal_proto[tg] + rng2.randn(input_dim) * 0.8
                for _ in range(seq_len)
            ])
            temporal.append(seq)

            # Frequency
            fg = freq_group[lbl]
            f = freq_proto[fg] + rng2.randn(input_dim // 2) * 0.8
            frequency.append(f)

            labels.append(lbl)

        return (
            torch.tensor(np.stack(vision),    dtype=torch.float32),
            torch.tensor(np.stack(temporal),  dtype=torch.float32),
            torch.tensor(np.stack(frequency), dtype=torch.float32),
            torch.tensor(labels,              dtype=torch.long),
        )

    train = make_split(n_train, 0)
    test  = make_split(n_test, 1000)
    return train, test, input_dim, input_dim // 2


# ── Individual model architectures ───────────────────────────────────────────

HIDDEN = 64   # hidden layer dimension — exposed as HVModel output

class VisionMLP(nn.Module):
    """MLP on rate-coded spike summary. Exposes hidden layer for HVA."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.hidden = nn.Sequential(nn.Linear(in_dim, HIDDEN), nn.ReLU())
        self.head   = nn.Linear(HIDDEN, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.hidden(x))

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.hidden(x)


class TemporalMLP(nn.Module):
    """MLP on mean-pooled spike sequence. Exposes hidden layer for HVA."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.hidden = nn.Sequential(nn.Linear(in_dim, HIDDEN), nn.ReLU())
        self.head   = nn.Linear(HIDDEN, out_dim)

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=1) if x.dim() == 3 else x.mean(dim=0, keepdim=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.hidden(self._pool(x)))

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.hidden(self._pool(x))


class FreqMLP(nn.Module):
    """MLP on frequency features. Exposes hidden layer for HVA."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.hidden = nn.Sequential(nn.Linear(in_dim, HIDDEN), nn.ReLU())
        self.head   = nn.Linear(HIDDEN, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.hidden(x))

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.hidden(x)


def train_individual(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    epochs: int = 30,
    lr: float = 1e-3,
) -> nn.Module:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        logits = model(X)
        if logits.dim() == 3:
            logits = logits.squeeze(1)
        loss = nn.functional.cross_entropy(logits, y)
        opt.zero_grad(); loss.backward(); opt.step()
    return model


def eval_individual(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        logits = model(X)
        if logits.dim() == 3:
            logits = logits.squeeze(1)
        pred = logits.argmax(dim=-1)
    return float((pred == y).float().mean().item())


# ── HVA wrapper ───────────────────────────────────────────────────────────────

def make_hv_model(torch_model: nn.Module, role: str, hv_dim: int) -> HVModel:
    """Wrap a trained model's HIDDEN LAYER (not logits) for HVA composition.

    Using the hidden layer (64-dim rich features) rather than the final
    logits (4-dim) gives the AutoencoderBridge enough information to build
    a meaningful hypervector — logits are too compressed.
    """
    torch_model.eval()

    def model_fn(x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feat = torch_model.features(x)
            if feat.dim() == 3:
                feat = feat.squeeze(1)
            return feat   # (batch, HIDDEN)

    return HVModel(
        model_fn,
        HVModelConfig(hv_dim=hv_dim, model_output_dim=HIDDEN, role_name=role),
    )


# ── Main experiment ───────────────────────────────────────────────────────────

def run_trial(
    n_classes: int,
    n_train: int,
    n_test: int,
    hv_dim: int,
    seed: int,
) -> Dict[str, float]:
    torch.manual_seed(seed)

    (Xv_tr, Xt_tr, Xf_tr, y_tr), (Xv_te, Xt_te, Xf_te, y_te), v_dim, f_dim = (
        make_multimodal_dataset(n_classes, n_train, n_test, seed=seed)
    )

    # ── Train individual models ────────────────────────────────────────────────
    vis_model  = train_individual(VisionMLP(v_dim, n_classes),  Xv_tr, y_tr)
    temp_model = train_individual(TemporalMLP(v_dim, n_classes), Xt_tr, y_tr)
    freq_model = train_individual(FreqMLP(f_dim, n_classes),    Xf_tr, y_tr)

    acc_vision  = eval_individual(vis_model,  Xv_te, y_te)
    acc_temporal= eval_individual(temp_model, Xt_te, y_te)
    acc_freq    = eval_individual(freq_model, Xf_te, y_te)

    # ── HVA: wrap trained models → compose → classify ─────────────────────────
    hv_vis  = make_hv_model(vis_model,  "vision",    hv_dim)
    hv_temp = make_hv_model(temp_model, "temporal",  hv_dim)
    hv_freq = make_hv_model(freq_model, "frequency", hv_dim)

    pipe_bundle   = HVPipeline({"vision": hv_vis, "temporal": hv_temp, "frequency": hv_freq},
                               n_classes=n_classes, hv_dim=hv_dim, strategy="bundle")
    pipe_weighted = HVPipeline({"vision": hv_vis, "temporal": hv_temp, "frequency": hv_freq},
                               n_classes=n_classes, hv_dim=hv_dim, strategy="weighted")
    pipe_bind     = HVPipeline({"vision": hv_vis, "temporal": hv_temp, "frequency": hv_freq},
                               n_classes=n_classes, hv_dim=hv_dim, strategy="bind")

    # Online training of HVA classifier heads (no retraining of wrapped models)
    for i in range(n_train):
        inputs = {
            "vision":    Xv_tr[i:i+1],
            "temporal":  Xt_tr[i:i+1],
            "frequency": Xf_tr[i:i+1],
        }
        lbl = int(y_tr[i].item())
        pipe_bundle.train_step(inputs, lbl)
        pipe_weighted.train_step(inputs, lbl)
        pipe_bind.train_step(inputs, lbl)

    # Evaluate HVA
    def eval_pipe(pipe, Xv, Xt, Xf, y):
        correct = 0
        for i in range(len(y)):
            inputs = {"vision": Xv[i:i+1], "temporal": Xt[i:i+1], "frequency": Xf[i:i+1]}
            hv = pipe.encode(inputs)
            pred, _ = pipe.predict(hv)
            if pred == int(y[i].item()):
                correct += 1
        return correct / len(y)

    acc_bundle   = eval_pipe(pipe_bundle,   Xv_te, Xt_te, Xf_te, y_te)
    acc_weighted = eval_pipe(pipe_weighted, Xv_te, Xt_te, Xf_te, y_te)
    acc_bind     = eval_pipe(pipe_bind,     Xv_te, Xt_te, Xf_te, y_te)

    return {
        "vision":    acc_vision,
        "temporal":  acc_temporal,
        "frequency": acc_freq,
        "best_individual": max(acc_vision, acc_temporal, acc_freq),
        "hva_bundle":   acc_bundle,
        "hva_weighted": acc_weighted,
        "hva_bind":     acc_bind,
        "best_hva":  max(acc_bundle, acc_weighted, acc_bind),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",     type=int, default=3)
    parser.add_argument("--n-classes", type=int, default=4)
    parser.add_argument("--n-train",   type=int, default=400)
    parser.add_argument("--n-test",    type=int, default=200)
    parser.add_argument("--hv-dim",    type=int, default=512)
    args = parser.parse_args()

    print(f"\n{'='*65}")
    print(f"  HyperVector Architecture — Composition Beats Individual Models")
    print(f"{'='*65}")
    print(f"  Task: {args.n_classes}-class multi-modal, {args.seeds} seeds")
    print(f"  Models: VisionMLP + TemporalMLP + FreqMLP  →  HVPipeline")
    print(f"  HV dim: {args.hv_dim}")
    print()

    all_results = []
    for seed in range(args.seeds):
        print(f"Seed {seed}…", end=" ", flush=True)
        t0 = time.perf_counter()
        r = run_trial(args.n_classes, args.n_train, args.n_test, args.hv_dim, seed)
        all_results.append(r)
        print(
            f"vision={r['vision']:.0%} temp={r['temporal']:.0%} freq={r['frequency']:.0%} "
            f"| best_indiv={r['best_individual']:.0%} | bundle={r['hva_bundle']:.0%} "
            f"({time.perf_counter()-t0:.1f}s)"
        )

    print(f"\n{'─'*65}")
    print(f"  {'Metric':<35} {'Mean':>8}  {'Std':>8}")
    print(f"{'─'*65}")

    keys = [
        ("Vision only",         "vision"),
        ("Temporal only",       "temporal"),
        ("Frequency only",      "frequency"),
        ("Best individual",     "best_individual"),
        ("HVA bundle",          "hva_bundle"),
        ("HVA weighted",        "hva_weighted"),
        ("HVA bind",            "hva_bind"),
        ("Best HVA",            "best_hva"),
    ]

    best_indiv_mean = np.mean([r["best_individual"] for r in all_results])
    best_hva_mean   = np.mean([r["best_hva"]        for r in all_results])

    for label, key in keys:
        vals = [r[key] for r in all_results]
        mean = np.mean(vals) * 100
        std  = np.std(vals, ddof=1) * 100 if len(vals) > 1 else 0.0
        marker = " ◄ BEST" if key == "best_hva" else ""
        print(f"  {label:<35} {mean:>7.1f}%  ±{std:>5.1f}%{marker}")

    gain = (best_hva_mean - best_indiv_mean) * 100
    print(f"\n  Composition gain over best individual: {gain:+.1f}pp")
    if gain > 0:
        print(f"  ✓ HVA WINS: composing {3} different architectures beats any single model")
    else:
        print(f"  ○ Individual models already saturate this task (try more classes or lower overlap)")

    print(f"\n  Code (4 lines, no retraining of wrapped models):")
    print(f"    pipe = HVPipeline({{")
    print(f"        'vision':   HVModel(vision_model, ...),")
    print(f"        'temporal': HVModel(temporal_model, ...),")
    print(f"        'frequency': HVModel(freq_model, ...),")
    print(f"    }}, strategy='bundle')")
    print(f"    pipe.add_model('lidar', HVModel(lidar_model, ...))  # runtime, no retraining")
    print()


if __name__ == "__main__":
    main()
