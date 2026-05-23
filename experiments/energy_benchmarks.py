"""
Energy Benchmark
================
Estimates synaptic operations (SynOps) as proxy for energy consumption.
Compares Arthedain SNN vs dense ANN equivalent.

Methodology
-----------
SynOps (Synaptic Operations) serve as a proxy for energy consumption because:
1. Memory access dominates energy in neuromorphic hardware (~90% of total)
2. Each multiply-accumulate (MAC) requires weight fetch + computation
3. Sparse event-driven computation skips inactive synapses entirely

SNN Baseline (Arthedain):
- Sparse spike activity (~5% neurons active per timestep)
- Only active neurons trigger synaptic lookups
- SynOps = active_neurons × fan_in per timestep
- Measured from actual spike logs during BCI decoding

ANN Baseline (Dense Equivalent):
- Every neuron computes every timestep (no sparsity)
- All weights accessed regardless of activity
- SynOps = hidden_size × (input_size + hidden_size + output_size) per timestep
- Equivalent to standard LSTM/GRU or dense feedforward

Energy Model:
- Dynamic energy ∝ SynOps × energy_per_mac
- SNN: ~5% activity → ~20× fewer SynOps
- Additional savings: no backprop storage, O(1) memory vs O(T)

Validation Approach:
These are simulation-based estimates. For hardware-validated numbers:
1. Synthesize with Vivado (Xilinx) or Quartus (Intel)
2. Run post-implementation power analysis with activity factors
3. Compare against actual current measurements on development board

References:
- Davies et al. (2018) Loihi paper: synaptic ops as energy proxy
- Qiao et al. (2020) FPGA SNN surveys: activity-dependent power models
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from models import RSNN, RSNNConfig
from data.synthetic import bci_velocity_stream


def count_synops(spikes_log: list[torch.Tensor], W_in, W_rec) -> dict:
    """Count synaptic operations from spike log."""
    total_in_ops = 0
    total_rec_ops = 0

    for spikes in spikes_log:
        n_active = spikes.sum().item()
        total_in_ops += n_active * W_in.shape[1]     # fan-in per active neuron
        total_rec_ops += n_active * W_rec.shape[1]

    T = len(spikes_log)
    return {
        "T": T,
        "mean_active_neurons": sum(s.sum().item() for s in spikes_log) / T,
        "total_synops": total_in_ops + total_rec_ops,
        "synops_per_step": (total_in_ops + total_rec_ops) / T,
        "sparsity": 1.0 - (sum(s.sum().item() for s in spikes_log) / (T * spikes_log[0].shape[0])),
    }


def ann_synops_estimate(input_size, hidden_size, output_size, T):
    """Dense ANN: every neuron fires every step."""
    return {
        "T": T,
        "total_synops": T * (input_size * hidden_size + hidden_size * hidden_size + hidden_size * output_size),
        "synops_per_step": input_size * hidden_size + hidden_size * hidden_size + hidden_size * output_size,
        "sparsity": 0.0,
    }


def run(T: int = 1000):
    INPUT_SIZE = 100
    HIDDEN_SIZE = 128
    OUTPUT_SIZE = 2

    rsnn = RSNN(RSNNConfig(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, sparse_p=0.15))

    spike_log = []
    for x, _ in bci_velocity_stream(T=T, input_size=INPUT_SIZE):
        spikes = rsnn.forward(x)
        spike_log.append(spikes.detach().clone())

    snn_stats = count_synops(spike_log, rsnn.W_in, rsnn.W_rec)
    ann_stats = ann_synops_estimate(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, T)

    ratio = ann_stats["synops_per_step"] / max(snn_stats["synops_per_step"], 1)

    print("=" * 50)
    print("ARTHEDAIN ENERGY BENCHMARK")
    print("=" * 50)
    print(f"  Hidden size:        {HIDDEN_SIZE}")
    print(f"  Timesteps:          {T}")
    print(f"\nSNN (Arthedain):")
    print(f"  Mean active neurons: {snn_stats['mean_active_neurons']:.2f} / {HIDDEN_SIZE}")
    print(f"  Sparsity:            {snn_stats['sparsity']*100:.1f}%")
    print(f"  SynOps/step:         {snn_stats['synops_per_step']:.0f}")
    print(f"\nDense ANN (equivalent):")
    print(f"  SynOps/step:         {ann_stats['synops_per_step']:.0f}")
    print(f"\nEnergy reduction (est.): {ratio:.1f}×")

    # Bar chart
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(
        ["Dense ANN", "Arthedain SNN"],
        [ann_stats["synops_per_step"], snn_stats["synops_per_step"]],
        color=["#888", "#2563eb"],
        width=0.4,
    )
    ax.set_ylabel("SynOps / timestep")
    ax.set_title(f"Estimated Energy — {ratio:.1f}× reduction", fontweight="bold")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
                f"{bar.get_height():.0f}", ha="center", fontsize=10)
    plt.tight_layout()
    os.makedirs("results/plots", exist_ok=True)
    plt.savefig("results/plots/energy_benchmark.png", dpi=150)
    print("\nSaved → results/plots/energy_benchmark.png")
    plt.close()

    return ratio


if __name__ == "__main__":
    run()
