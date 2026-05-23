"""
Supply Chain Experiment
=======================
Arthedain's industrial IoT differentiator.

Non-stationary event stream decoding with concept drift —
the ground-truth sensor→demand mapping shifts slowly over time.
This stress-tests online adaptation in a way BCI benchmarks don't.

Use case:
  Edge SNN deployed at a warehouse or agricultural site.
  Sensor events (arrivals, environmental readings, machine states)
  → real-time demand / yield forecast.
  The underlying process drifts (seasonality, equipment wear, supply shocks).
  Model must adapt without retraining.

Run:
    python experiments/supply_chain.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models import RSNN, RSNNConfig, DualHebbianAccumulator, HebbianConfig, Readout, ReadoutConfig
from training import OnlineTrainer, TrainerConfig
from data.synthetic import supply_chain_stream


def pearson_r(pred: torch.Tensor, target: torch.Tensor) -> float:
    p = pred - pred.mean(0)
    t = target - target.mean(0)
    return ((p * t).sum(0) / (p.norm(0) * t.norm(0)).clamp(min=1e-6)).mean().item()


def run_variant(drift_rate: float, label: str, T: int = 3000, seed: int = 0):
    torch.manual_seed(seed)

    INPUT_SIZE = 50
    HIDDEN_SIZE = 96
    OUTPUT_SIZE = 3

    rsnn = RSNN(RSNNConfig(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, sparse_p=0.12))
    readout = Readout(ReadoutConfig(HIDDEN_SIZE, OUTPUT_SIZE, mode="smoothed"))
    hebbian = DualHebbianAccumulator(HebbianConfig(
        shape=(HIDDEN_SIZE, HIDDEN_SIZE),
        tau_fast=5.0,
        tau_slow=50.0,
        alpha=0.7,
        beta=0.3,
    ))
    trainer = OnlineTrainer(
        rsnn, readout, hebbian,
        TrainerConfig(lr_readout=3e-3, lr_recurrent=1e-4),
    )

    preds, targets, losses = [], [], []

    for x, y in supply_chain_stream(T=T, input_size=INPUT_SIZE, n_outputs=OUTPUT_SIZE,
                                     drift_rate=drift_rate, seed=seed):
        y_pred, error = trainer.step(x, target=y)
        preds.append(y_pred.detach())
        targets.append(y)
        losses.append(error.pow(2).mean().item())

    preds_t = torch.stack(preds)
    targets_t = torch.stack(targets)
    r = pearson_r(preds_t, targets_t)
    print(f"  {label:40s}  R={r:.4f}  final_loss={losses[-1]:.4f}")
    return preds_t, targets_t, losses, r


def run():
    print("\nARTHEDAIN — Supply Chain / Industrial IoT Experiment")
    print("=" * 60)
    print("Testing online adaptation under concept drift\n")

    variants = [
        (0.000, "No drift       (stationary baseline)"),
        (0.001, "Slow drift     (seasonal shift)"),
        (0.005, "Medium drift   (equipment degradation)"),
        (0.015, "Fast drift     (supply shock)"),
    ]

    results = {}
    all_losses = {}

    for drift_rate, label in variants:
        preds, targets, losses, r = run_variant(drift_rate, label)
        results[label] = r
        all_losses[label] = losses

    # ── Plot 1: loss curves under drift ──
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    axes = axes.flatten()
    colors = ["#2563eb", "#16a34a", "#d97706", "#dc2626"]

    for ax, (drift_rate, label), color in zip(axes, variants, colors):
        losses = all_losses[label]
        # smooth for readability
        window = 50
        smoothed = [
            sum(losses[max(0, i-window):i+1]) / min(i+1, window)
            for i in range(len(losses))
        ]
        ax.plot(smoothed, color=color, linewidth=1.2)
        ax.set_title(label.strip(), fontsize=10, fontweight="bold")
        ax.set_ylabel("MSE loss")
        ax.set_xlabel("Timestep")
        ax.spines[["top", "right"]].set_visible(False)
        r = results[label]
        ax.text(0.97, 0.92, f"R={r:.3f}", transform=ax.transAxes,
                ha="right", fontsize=9, color=color, fontweight="bold")

    fig.suptitle("Arthedain — Online Adaptation Under Concept Drift", fontweight="bold", fontsize=13)
    plt.tight_layout()
    os.makedirs("results/plots", exist_ok=True)
    plt.savefig("results/plots/supply_chain_drift.png", dpi=150)
    print("\nSaved → results/plots/supply_chain_drift.png")
    plt.close()

    # ── Plot 2: prediction vs target, slow drift case ──
    _, (drift_rate, label) = next(
        (i, v) for i, v in enumerate(variants) if v[0] == 0.001
    ), variants[1]
    preds, targets, _, _ = run_variant(0.001, "slow drift", T=3000)

    fig, axes = plt.subplots(3, 1, figsize=(13, 7), sharex=True)
    output_labels = ["demand channel 0", "demand channel 1", "demand channel 2"]
    for i, ax in enumerate(axes):
        ax.plot(targets[:, i].numpy(), label="target", alpha=0.75, linewidth=0.8)
        ax.plot(preds[:, i].numpy(), label="predicted", alpha=0.75, linewidth=0.8)
        ax.set_ylabel(output_labels[i], fontsize=9)
        if i == 0:
            ax.legend(loc="upper right", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlabel("Timestep")
    fig.suptitle("Supply Chain Decoding — Slow Drift (drift_rate=0.001)", fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/plots/supply_chain_prediction.png", dpi=150)
    print("Saved → results/plots/supply_chain_prediction.png")
    plt.close()

    # ── Summary ──
    print("\nSummary — Pearson R by drift severity:")
    print("-" * 50)
    for label, r in results.items():
        bar = "█" * int(r * 30) if r > 0 else ""
        print(f"  {r:.3f}  {bar}  {label.strip()}")

    return results


if __name__ == "__main__":
    run()
