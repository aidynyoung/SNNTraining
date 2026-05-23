"""
experiments/compare_classifiers.py
====================================
M-model (HolographicInferenceModel) vs AdaptiveHD (prototype classifier).

Directly compares the two HDC classification architectures on the same
encoder, data, and training budget so the tradeoffs are clear.

Architecture comparison:
    HolographicInferenceModel:
        - ONE superposed model vector M = Σ bind(encode(x), class_hv)
        - Inference: XOR(query, M) → nearest class HV
        - Bidirectional: reconstruct(class) → stereotypical input
        - Provenance: find most similar stored training examples
        - Storage: O(dim) regardless of n_classes or n_examples

    AdaptiveHDClassifier (RefineHD, Verges Boncompte 2025):
        - K prototype vectors, one per class
        - Inference: argmax sim(encode(query), P_c)
        - Online pull/push updates per example
        - Storage: O(K × dim)

Usage:
    python experiments/compare_classifiers.py
    python experiments/compare_classifiers.py --n-classes 10 --n-train 200
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch

from hdc.inference_model import HolographicInferenceModel, FixedThresholdEncoder
from hdc.resonator import AdaptiveHDClassifier
from hdc.concentration import DIM_CANONICAL, theoretical_std


def make_encoder(feat_dim: int, dim: int, seed: int = 42) -> FixedThresholdEncoder:
    """Random-projection encoder with fixed threshold calibration."""
    torch.manual_seed(seed)
    W = torch.randn(feat_dim, dim) / (dim ** 0.5)

    def raw(x: torch.Tensor) -> torch.Tensor:
        return x.float() @ W

    enc = FixedThresholdEncoder(raw, dim=dim)
    return enc


def run_comparison(
    n_classes: int = 5,
    n_train: int = 50,
    n_test: int = 20,
    feat_dim: int = 64,
    dim: int = DIM_CANONICAL,
    class_sep: float = 3.0,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    torch.manual_seed(seed)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Classifier Comparison")
        print(f"  dim={dim}  classes={n_classes}  train/class={n_train}  sep={class_sep}")
        print(f"{'='*60}")

    # Build encoder and calibrate from balanced data
    enc = make_encoder(feat_dim, dim, seed=seed)
    calib = torch.cat([
        torch.randn(30, feat_dim) + c * class_sep
        for c in range(n_classes)
    ])
    enc.calibrate(calib)

    # Generate train / test sets
    def gen(c, n):
        return [torch.randn(feat_dim) + c * class_sep for _ in range(n)]

    train_data = {c: gen(c, n_train) for c in range(n_classes)}
    test_data  = {c: gen(c, n_test)  for c in range(n_classes)}

    # ── HolographicInferenceModel ──────────────────────────────────────────────
    # encoder=None: pass pre-encoded binary HVs directly to avoid double-encoding
    t0 = time.perf_counter()
    m_model = HolographicInferenceModel(dim=dim, n_classes=n_classes, encoder=None, seed=seed)
    for c in range(n_classes):
        for x in train_data[c]:
            m_model.train(enc(x), c)
    m_model.finalize()
    m_train_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    m_correct = 0
    m_z_correct, m_z_wrong = [], []
    for c in range(n_classes):
        for x in test_data[c]:
            pred, stats = m_model.classify(enc(x))
            if pred == c:
                m_correct += 1
                m_z_correct.append(stats["z_score"])
            else:
                m_z_wrong.append(stats["z_score"])
    m_test_s = time.perf_counter() - t0
    m_acc = m_correct / (n_classes * n_test)

    # ── AdaptiveHDClassifier ───────────────────────────────────────────────────
    t0 = time.perf_counter()
    p_model = AdaptiveHDClassifier(
        n_features=dim, n_classes=n_classes, dim=dim,
        mode="binary", learning_rate=0.1, seed=seed,
    )
    for c in range(n_classes):
        for x in train_data[c]:
            hv = enc(x)
            p_model.train_step(hv, c, predict_first=False)
    p_train_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    p_correct = 0
    for c in range(n_classes):
        for x in test_data[c]:
            hv = enc(x)
            pred, _ = p_model.predict(hv)
            if pred == c:
                p_correct += 1
    p_test_s = time.perf_counter() - t0
    p_acc = p_correct / (n_classes * n_test)

    # ── Unique M-model capabilities ────────────────────────────────────────────
    recon_better = sum(
        m_model.reconstruction_similarity(c, enc(test_data[c][0])) > 0.5
        for c in range(n_classes)
    )

    total = n_classes * n_test
    if verbose:
        sigma = theoretical_std(dim)
        print(f"\n  {'Metric':<35} {'M-model':>12} {'AdaptiveHD':>12}")
        print(f"  {'-'*59}")
        print(f"  {'Accuracy':<35} {m_acc:>11.1%} {p_acc:>11.1%}")
        print(f"  {'Train time (s)':<35} {m_train_s:>12.3f} {p_train_s:>12.3f}")
        print(f"  {'Test time (s)':<35} {m_test_s:>12.3f} {p_test_s:>12.3f}")
        print(f"  {'Storage (# HVs)':<35} {'1':>12} {str(n_classes):>12}")
        print(f"  {'Avg Z (correct preds)':<35} {sum(m_z_correct)/max(len(m_z_correct),1):>11.1f}σ {'N/A':>12}")
        print(f"  {'Avg Z (wrong preds)':<35}  {sum(m_z_wrong)/max(len(m_z_wrong),1):>10.1f}σ {'N/A':>12}")
        print(f"  {'Z separation':<35} {sum(m_z_correct)/max(len(m_z_correct),1)-sum(m_z_wrong)/max(len(m_z_wrong),1):>11.1f}σ {'N/A':>12}")
        print(f"\n  M-model unique capabilities:")
        print(f"    Reconstruction better than random: {recon_better}/{n_classes} classes")
        prov = m_model.provenance(enc(test_data[0][0]), top_k=5)
        prov_class0 = sum(1 for e in prov if e['class_idx'] == 0)
        print(f"    Provenance: class-0 in top-5 = {prov_class0}/5")
        print(f"    Null distribution 3σ cutoff:  H/dim < {0.5 - 3*sigma:.4f}")

    result = {
        "m_model_accuracy": m_acc,
        "adaptive_hd_accuracy": p_acc,
        "m_model_train_s": m_train_s,
        "adaptive_hd_train_s": p_train_s,
        "m_model_z_separation": (
            sum(m_z_correct) / max(len(m_z_correct), 1)
            - sum(m_z_wrong) / max(len(m_z_wrong), 1)
        ),
        "reconstruction_classes_better_than_random": recon_better,
        "dim": dim,
        "n_classes": n_classes,
        "n_train": n_train,
        "class_separation": class_sep,
    }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-classes", type=int, default=5)
    parser.add_argument("--n-train", type=int, default=50)
    parser.add_argument("--n-test", type=int, default=20)
    parser.add_argument("--feat-dim", type=int, default=64)
    parser.add_argument("--dim", type=int, default=DIM_CANONICAL)
    parser.add_argument("--sep", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    result = run_comparison(
        n_classes=args.n_classes,
        n_train=args.n_train,
        n_test=args.n_test,
        feat_dim=args.feat_dim,
        dim=args.dim,
        class_sep=args.sep,
        seed=args.seed,
        verbose=True,
    )

    if args.save:
        out = Path("results/compare_classifiers.json")
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(result, indent=2))
        print(f"\n  Results saved → {out}")
