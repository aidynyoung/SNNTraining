"""
benchmark_energy.py — SynOp counting and energy estimation for Arthedain.

Compares estimated energy per inference across:
  - Arthedain (SNN with sparse recurrent weights)
  - Equivalent MLP (same hidden size)
  - Reference Transformer (tiny, for scale comparison)

Energy estimates use the Horowitz (ISSCC 2014) table:
  - 45nm CMOS: 1 MAC = ~4.6 pJ (INT8), ~0.9 pJ per SynOp (spike-driven)
  - SNN advantage: event-driven (only active neurons consume energy)
  - Sparsity factor: fraction of neurons firing per timestep

Usage:
    python experiments/benchmark_energy.py
    python experiments/benchmark_energy.py --hidden 128 --steps 5000
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import argparse
import torch
import numpy as np
from models import RSNN, RSNNConfig
from models.hdc import HDCEncoder


# Energy model constants (Horowitz ISSCC 2014, 45nm CMOS)
# -------------------------------------------------------
# INT8 MAC:  4.6 pJ  (multiply-accumulate, digital)
# INT8 ADD:  0.4 pJ
# SRAM read: 5.0 pJ  (per 64-bit word)
# DRAM read: 640 pJ (per 64-bit word) — not used for edge inference
# SNN SynOp: 0.9 pJ (spike-driven, event-based; ~5x lower than MAC because
#                     spiking multiplication is a gated pass-through)
# Source: Horowitz, M. "1.1 Computing's energy problem (and what we can do
#         about it)," ISSCC 2014, Table 1.

ENERGY_MAC_PJ = 4.6      # pJ per INT8 multiply-accumulate
ENERGY_SYNOP_PJ = 0.9    # pJ per synaptic operation (spike-driven)

# Energy for standard nonlinearity (ReLU / LIF fire)
ENERGY_ACTIVATION_PJ = 0.4  # pJ per activation (comparison + pass-through)


def estimate_arthedain_energy(rsnn: RSNN, n_steps: int) -> dict:
    """Estimate Arthedain energy consumption from SynOp counts.

    1. Count total SynOps across n_steps
    2. Compute average SynOps per inference
    3. Estimate pJ per inference

    Returns:
        dict with synops, energy_pj, and breakdown
    """
    if rsnn.total_inferences == 0:
        return {"synops_per_inference": 0, "energy_pj_per_inference": 0.0, "error": "no inferences"}

    total_synops = rsnn.total_synops
    total_infs = rsnn.total_inferences
    synops_per_inf = total_synops / total_infs

    # Energy: SynOps + LIF activation
    # LIF: each neuron checks threshold every timestep (one ADD + compare)
    lif_energy = rsnn.hidden_size * ENERGY_ACTIVATION_PJ  # per timestep, all neurons
    synops_energy = synops_per_inf * ENERGY_SYNOP_PJ

    # Readout: linear layer (output_size * hidden_size MACs)
    # Assume reading a linear layer is a MAC per connection
    readout_energy = 2 * rsnn.hidden_size * ENERGY_MAC_PJ  # 2 output dims

    total_energy_pj = synops_energy + lif_energy + readout_energy
    total_energy_nj = total_energy_pj / 1000.0  # pJ -> nJ

    return {
        "total_synops": int(total_synops),
        "total_inferences": total_infs,
        "synops_per_inference": float(f"{synops_per_inf:.1f}"),
        "mean_spike_fraction": float(f"{rsnn.prev_spikes.mean().item():.4f}"),
        "lif_energy_pj": float(f"{lif_energy:.2f}"),
        "synops_energy_pj": float(f"{synops_energy:.2f}"),
        "readout_energy_pj": float(f"{readout_energy:.2f}"),
        "total_energy_pj_per_inference": float(f"{total_energy_pj:.2f}"),
        "total_energy_nj_per_inference": float(f"{total_energy_nj:.4f}"),
    }


def estimate_mlp_energy(input_size: int, hidden_size: int, output_size: int) -> dict:
    """Estimate energy for an equivalent MLP: input->hidden(ReLU)->output.

    MLP processes ALL activations — no sparsity advantage.
    Energy = n_macs * ENERGY_MAC_PJ + n_activations * ENERGY_ACTIVATION_PJ
    """
    # Layer 1: input_size * hidden_size MACs
    macs_layer1 = input_size * hidden_size
    # Layer 2: hidden_size * output_size MACs
    macs_layer2 = hidden_size * output_size
    total_macs = macs_layer1 + macs_layer2

    # Activations: hidden ReLU + output (no activation on output)
    total_activations = hidden_size

    energy_macs = total_macs * ENERGY_MAC_PJ
    energy_activations = total_activations * ENERGY_ACTIVATION_PJ
    total_energy_pj = energy_macs + energy_activations
    total_energy_nj = total_energy_pj / 1000.0

    return {
        "architecture": f"MLP({input_size}->{hidden_size}->{output_size})",
        "total_macs": total_macs,
        "energy_macs_pj": float(f"{energy_macs:.2f}"),
        "energy_activations_pj": float(f"{energy_activations:.2f}"),
        "total_energy_pj_per_inference": float(f"{total_energy_pj:.2f}"),
        "total_energy_nj_per_inference": float(f"{total_energy_nj:.4f}"),
    }


def estimate_transformer_energy(input_size: int, hidden_size: int, n_heads: int = 2) -> dict:
    """Estimate energy for a tiny reference transformer (single-layer).

    Approximate MACs for one transformer layer:
      - Self-attention: 2 * seq_len * d_model^2  (Q,K,V projections + output)
      - FFN: 2 * seq_len * d_model * d_ff  (d_ff = 4 * d_model typical)
    Using seq_len=1 for per-timestep comparison (autoregressive).
    """
    d_model = hidden_size
    d_ff = 4 * d_model
    seq_len = 1  # per-timestep

    # QKV projections: 3 * d_model^2
    # Output projection: d_model^2
    attn_macs = 4 * seq_len * d_model * d_model

    # FFN: 2 * d_model * d_ff
    ffn_macs = 2 * seq_len * d_model * d_ff

    total_macs = attn_macs + ffn_macs
    energy_macs = total_macs * ENERGY_MAC_PJ
    total_energy_pj = energy_macs
    total_energy_nj = total_energy_pj / 1000.0

    return {
        "architecture": f"Transformer(d_model={d_model}, d_ff={d_ff}, heads={n_heads})",
        "total_macs": total_macs,
        "energy_macs_pj": float(f"{energy_macs:.2f}"),
        "total_energy_pj_per_inference": float(f"{total_energy_pj:.2f}"),
        "total_energy_nj_per_inference": float(f"{total_energy_nj:.4f}"),
    }


def run_arthedain_benchmark(input_size: int, hidden_size: int,
                             n_steps: int, device: torch.device) -> dict:
    """Run Arthedain on a synthetic stream and count SynOps."""
    rsnn = RSNN(config=RSNNConfig(
        input_size=input_size,
        hidden_size=hidden_size,
        sparse_init=True,
        sparse_p=0.15,
        input_gain=50.0,
        tau=20.0,
        v_th=1.0,
        device=str(device) if device is not None else None,
    ))

    # Run for n_steps on synthetic data
    torch.manual_seed(42)
    for t in range(n_steps):
        x = torch.randn(input_size, device=device) * 0.5
        _ = rsnn.forward(x)

    energy = estimate_arthedain_energy(rsnn, n_steps)
    energy["architecture"] = f"Arthedain(hidden={hidden_size}, sparse=15%)"
    return energy


def main():
    parser = argparse.ArgumentParser(description="Energy benchmark for Arthedain")
    parser.add_argument("--input-size", type=int, default=100)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=5000)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("Arthedain Energy Benchmark")
    print("=" * 70)
    print(f"Input size:  {args.input_size}")
    print(f"Hidden size: {args.hidden_size}")
    print(f"Steps:       {args.steps}")
    print(f"Device:      {device}")
    print(f"Technology:  45nm CMOS (Horowitz ISSCC 2014)")
    print()

    # 1. Arthedain
    print("--- Arthedain (SNN with sparse LIF reservoir) ---")
    ae = run_arthedain_benchmark(
        args.input_size, args.hidden_size, args.steps, device)
    for k, v in ae.items():
        print(f"  {k}: {v}")
    print()

    # 2. Equivalent MLP
    print("--- Equivalent MLP ---")
    me = estimate_mlp_energy(args.input_size, args.hidden_size, 2)
    for k, v in me.items():
        print(f"  {k}: {v}")
    print()

    # 3. Tiny Transformer
    print("--- Reference Transformer (tiny, 1-layer) ---")
    xf = estimate_transformer_energy(args.input_size, args.hidden_size)
    for k, v in xf.items():
        print(f"  {k}: {v}")
    print()

    # 4. Comparison
    print("--- Comparison ---")
    ae_nj = float(ae["total_energy_nj_per_inference"])
    mlp_nj = float(me["total_energy_nj_per_inference"])
    xfmr_nj = float(xf["total_energy_nj_per_inference"])

    print(f"  Arthedain:        {ae_nj:.4f} nJ/inference")
    print(f"  MLP:              {mlp_nj:.4f} nJ/inference")
    print(f"  Transformer:      {xfmr_nj:.4f} nJ/inference")

    if ae_nj > 0:
        print(f"\n  Arthedain vs MLP:         {mlp_nj/ae_nj:.1f}x lower energy")
        print(f"  Arthedain vs Transformer: {xfmr_nj/ae_nj:.1f}x lower energy")

    # Check synops per inference for brief
    synops_per_inf = float(ae["synops_per_inference"])
    total_macs_mlp = int(me["total_macs"])
    total_macs_xfmr = int(xf["total_macs"])
    spike_frac = float(ae["mean_spike_fraction"])

    print(f"\n--- Key numbers for the brief ---")
    print(f"  SynOps/inference:        {synops_per_inf:.0f}")
    print(f"  MLP MACs/inference:      {total_macs_mlp}")
    print(f"  Transformer MACs/inference: {total_macs_xfmr}")
    print(f"  Mean spike fraction:     {spike_frac:.2%}")
    print(f"  SynOp vs MAC ratio:      {ENERGY_SYNOP_PJ/ENERGY_MAC_PJ:.2f}x energy per op")


if __name__ == "__main__":
    main()
