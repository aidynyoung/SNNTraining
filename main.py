"""
SNNTraining: Online FORCE/RLS-style Training for BCI Velocity Decoding

This script implements online learning for recurrent spiking neural networks (RSNN)
using a dual-timescale Hebbian plasticity rule combined with recursive least squares
(RLS)-style error correction. The approach is designed for edge deployment in brain-
computer interfaces (BCIs) where real-time adaptation is required without backpropagation.

Architecture:
- RSNN: Recurrent SNN with LIF neurons and sparse spectral initialization
- DualHebbian: Fast (~100ms) and slow (~700ms) eligibility traces for local learning
- Readout: Linear decoder with optional exponential smoothing (EMA)
- Trainer: Online FORCE/RLS-style update combining Hebbian plasticity with error feedback

The training loop processes synthetic BCI velocity data (population spike trains -> 2D
velocity) in a single online pass—no epochs, no replay buffer, O(1) memory.

References:
- DePasquale et al. (2018): FULL-FORCE method for training SNNs
- Nicola & Clopath (2017): SuperSpike—online learning with eligibility traces
- Bellec et al. (2020): E-prop—online learning in RSNNs
"""

import torch
import matplotlib.pyplot as plt
import numpy as np

from models.rsnn import RSNN
from models.readout import Readout
from models.hebbian import DualHebbianAccumulator, HebbianConfig
from training.online_trainer import OnlineTrainer
from data.synthetic import generate_stream
from utils import load_config, get_device, print_config_summary


def main():
    # Load configuration
    config = load_config()
    print_config_summary(config)

    # Extract parameters
    device = get_device()
    print(f"Using device: {device}")

    input_size = config['model']['input_size']
    hidden_size = config['model']['hidden_size']
    output_size = config['model']['output_size']
    lr_readout = config['training']['lr_readout']
    lr_recurrent = config['training']['lr_recurrent']
    timesteps = config['data']['T']

    # model
    sparse_init = config['model'].get('sparse_init', False)
    sparse_p = config['model'].get('sparse_p', 0.15)
    rsnn = RSNN(
        input_size, hidden_size, device=device,
        sparse_init=sparse_init, sparse_p=sparse_p
    )

    readout_mode = config['readout'].get('mode', 'direct')
    readout_smooth_tau = config['readout'].get('smooth_tau', 5.0)
    readout = Readout(
        hidden_size, output_size, device=device,
        mode=readout_mode, smooth_tau=readout_smooth_tau
    )
    hebbian = DualHebbianAccumulator(HebbianConfig(
        shape=(hidden_size, hidden_size),
        tau_fast=config['hebbian']['tau_fast'],
        tau_slow=config['hebbian']['tau_slow'],
        alpha=config['hebbian']['alpha'],
        beta=config['hebbian']['beta'],
    ), device=device)

    # Trainer combines dual-timescale Hebbian (local, O(1) memory) with RLS-style
    # error feedback for online weight updates. This is the core FORCE-like algorithm:
    # 1. RSNN generates spikes via LIF dynamics
    # 2. DualHebbian accumulates fast/slow eligibility traces at each synapse
    # 3. Readout produces prediction via linear decode
    # 4. Error signal modulates Hebbian traces for weight updates
    trainer = OnlineTrainer(
        rsnn, readout, hebbian,
        lr_readout=lr_readout, lr_recurrent=lr_recurrent, device=device
    )

    # Online training loop: single pass through streaming data
    # No epochs, no backprop through time, O(1) memory per timestep
    preds = []
    targets = []
    losses = []

    for i, (x, y) in enumerate(generate_stream(timesteps, input_size, config['data']['noise'])):
        # Single forward pass + local weight update
        y_pred, error = trainer.step(x, y)
        
        loss = torch.mean(error**2)
        losses.append(loss.item())

        preds.append(y_pred.detach().cpu().numpy())
        targets.append(y.cpu().numpy())
        
        if i % config['training']['log_every'] == 0:
            print(f"Step {i}: Loss = {loss:.4f}")

    # Convert lists to tensors efficiently
    preds = torch.tensor(np.array(preds))
    targets = torch.tensor(np.array(targets))

    # plot results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Trajectory tracking
    ax1.plot(preds[:,0], label="pred x", alpha=0.8)
    ax1.plot(targets[:,0], label="true x", alpha=0.8)
    ax1.set_ylabel("Position X")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Training loss
    ax2.plot(losses, label="Training Loss", color='red', alpha=0.8)
    ax2.set_xlabel("Timestep")
    ax2.set_ylabel("MSE Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Online SNN Decoding (Dual-Timescale Hebbian)")
    plt.tight_layout()
    plt.show()

    print(f"Final Loss: {losses[-1]:.4f}")
    print(f"Average Loss (last 100 steps): {sum(losses[-100:])/100:.4f}")


if __name__ == "__main__":
    main()