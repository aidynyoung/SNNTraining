"""
Ablation Studies
================
Compares variants:
  1. Dual-timescale Hebbian (full model)
  2. Single fast trace only
  3. Single slow trace only
  4. Pure delta rule (no eligibility traces)

Metric: Pearson R on BCI velocity decoding.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models import RSNN, RSNNConfig, DualHebbianAccumulator, HebbianConfig, Readout, ReadoutConfig
from training import OnlineTrainer, TrainerConfig
from data.synthetic import bci_velocity_stream


def pearson_r(pred, target):
    p = pred - pred.mean(0)
    t = target - target.mean(0)
    return ((p * t).sum(0) / (p.norm(0) * t.norm(0)).clamp(min=1e-6)).mean().item()


def run_variant(name: str, alpha: float, beta: float, T=1000, seed=42):
    torch.manual_seed(seed)
    INPUT, HIDDEN, OUTPUT = 100, 128, 2

    rsnn = RSNN(RSNNConfig(INPUT, HIDDEN))
    readout = Readout(ReadoutConfig(HIDDEN, OUTPUT))
    hebbian = DualHebbianAccumulator(HebbianConfig(
        shape=(HIDDEN, HIDDEN), alpha=alpha, beta=beta
    ))
    trainer = OnlineTrainer(rsnn, readout, hebbian, TrainerConfig(lr_readout=2e-3))

    preds, targets = [], []
    for x, y in bci_velocity_stream(T=T, input_size=INPUT, seed=seed):
        yp, _ = trainer.step(x, target=y)
        preds.append(yp.detach())
        targets.append(y)

    r = pearson_r(torch.stack(preds), torch.stack(targets))
    print(f"  {name:35s}  R = {r:.4f}")
    return r


def run():
    print("\nABLATION STUDY — SNNTraining Dual-Timescale Hebbian\n" + "=" * 55)

    variants = [
        ("Dual-timescale (α=0.7, β=0.3)",  0.7, 0.3),
        ("Fast only (α=1.0, β=0.0)",        1.0, 0.0),
        ("Slow only (α=0.0, β=1.0)",        0.0, 1.0),
        ("Equal weight (α=0.5, β=0.5)",     0.5, 0.5),
    ]

    results = {}
    for name, alpha, beta in variants:
        r = run_variant(name, alpha, beta)
        results[name] = r

    # Plot
    names = list(results.keys())
    values = list(results.values())
    colors = ["#2563eb" if i == 0 else "#94a3b8" for i in range(len(names))]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.barh(names, values, color=colors, height=0.5)
    ax.set_xlim(0, max(values) * 1.3)
    ax.set_xlabel("Pearson R")
    ax.set_title("Ablation: Trace Weighting", fontweight="bold")
    for bar, val in zip(bars, values):
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=9)
    plt.tight_layout()

    os.makedirs("results/plots", exist_ok=True)
    plt.savefig("results/plots/ablations.png", dpi=150)
    print("\nSaved → results/plots/ablations.png")
    plt.close()

    return results


if __name__ == "__main__":
    run()
