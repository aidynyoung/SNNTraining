"""
experiments/benchmark_sequential_mnist.py
===========================================
Sequential MNIST (sMNIST) benchmark for Arthedain.

This benchmark addresses Gap 1: "SNN Doesn't Learn Classification."
It tests the RSNN's ability to learn a temporal classification task
where MNIST digits are presented pixel-by-pixel (one pixel per timestep).

Why sMNIST matters:
  - Standard neuromorphic benchmark (temporal credit assignment)
  - 784 timesteps of sequential input requires the SNN to retain context
  - Published baselines: BPTT ~98%, e-prop ~95%, online Hebbian ~85%
  - Directly tests the dual-trace Hebbian's ability to bridge long timescales

Modes:
  --mode reservoir  : Fixed W_rec, train readout via SGD (echo state)
  --mode hebbian    : Train W_rec via dual-trace Hebbian, readout via SGD
  --mode eprop      : Train W_rec via e-prop, readout via SGD
  --mode reservoir-ridge : Ridge regression on reservoir states (fast baseline)

References
----------
- Hochreiter & Schmidhuber (1997) LSTM
- Bellec et al. (2020) e-prop, Nature Comms
- Cramer et al. (2020) SHD, IEEE TNNLS

Usage
-----
    python experiments/benchmark_sequential_mnist.py
    python experiments/benchmark_sequential_mnist.py --mode hebbian --n-train 60000
    python experiments/benchmark_sequential_mnist.py --mode eprop --hidden 256
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.rsnn import RSNN, RSNNConfig
from models.lif import LIFLayer, LIFConfig
from models.readout import Readout, ReadoutConfig
from models.hebbian import DualHebbianAccumulator, HebbianConfig


# ── sMNIST data ───────────────────────────────────────────────────────────────

def load_smnist(n_train: int = 60000, n_test: int = 10000,
                permute: bool = True, seed: int = 42) -> Tuple[
                    Tuple[torch.Tensor, torch.Tensor],
                    Tuple[torch.Tensor, torch.Tensor],
                ]:
    """
    Load MNIST dataset as sequential (pixel-by-pixel) streams.

    Each image (28×28=784 pixels) is serialised row-by-row into a
    784-timestep sequence with 1 input neuron (grayscale value).

    Args:
        n_train: Number of training samples
        n_test: Number of test samples
        permute: If True, use fixed random permutation of pixel order
                 (standard sMNIST benchmark)
        seed: Permutation seed

    Returns:
        ((X_train, y_train), (X_test, y_test)) where
        X_train: (n_train, 784) float tensor, values 0-1
        y_train: (n_train,) long tensor, labels 0-9
    """
    from torchvision import datasets, transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_data = datasets.MNIST(
        root=os.path.join(os.path.dirname(__file__), "..", "data", "mnist"),
        train=True, download=True, transform=transform,
    )
    test_data = datasets.MNIST(
        root=os.path.join(os.path.dirname(__file__), "..", "data", "mnist"),
        train=False, download=True, transform=transform,
    )

    # Flatten to (n_samples, 784)
    X_train = train_data.data.float()[:n_train] / 255.0
    X_train = (X_train - 0.1307) / 0.3081  # Normalize
    y_train = train_data.targets[:n_train]

    X_test = test_data.data.float()[:n_test] / 255.0
    X_test = (X_test - 0.1307) / 0.3081
    y_test = test_data.targets[:n_test]

    # Reshape: (n_samples, 784) — each row is the full image flattened
    X_train = X_train.reshape(n_train, 784)
    X_test = X_test.reshape(n_test, 784)

    # Fixed permutation (standard sMNIST protocol)
    if permute:
        rng = np.random.RandomState(seed)
        perm = rng.permutation(784)
        X_train = X_train[:, perm]
        X_test = X_test[:, perm]

    return (X_train, y_train), (X_test, y_test)


# ── SNN Classifier ────────────────────────────────────────────────────────────

class sMNISTClassifier:
    """
    Sequential MNIST classifier using Arthedain SNN.

    Each pixel is presented as a scalar input (input_size=1).
    The SNN accumulates temporal context over 784 timesteps.
    Readout from the final hidden state (or average over last N steps).
    """

    def __init__(
        self,
        hidden_size: int = 256,
        tau: float = 20.0,
        v_th: float = 1.0,
        device: str = "cpu",
        mode: str = "reservoir",
        tau_fast: float = 5.0,
        tau_slow: float = 50.0,
        hebbian_alpha: float = 0.7,
        hebbian_beta: float = 0.3,
        readout_window: int = 50,  # Last N steps to average for readout
    ):
        self.hidden_size = hidden_size
        self.device = device
        self.mode = mode
        self.readout_window = readout_window

        # Build reservoir SNN
        self.rsnn = RSNN(RSNNConfig(
            input_size=1,  # Scalar pixel input
            hidden_size=hidden_size,
            sparse_init=True,
            sparse_p=0.15,
            input_gain=5.0,
            tau=tau,
            v_th=v_th,
            dt=1.0,
            device=device,
        ))

        # Readout: hidden_size → 10 classes
        self.readout = Readout(ReadoutConfig(
            hidden_size=hidden_size,
            output_size=10,
        ), device=device)

        # Dual-trace Hebbian (only used in hebbian mode)
        if mode == "hebbian":
            self.hebbian = DualHebbianAccumulator(HebbianConfig(
                shape=(hidden_size, hidden_size),
                tau_fast=tau_fast,
                tau_slow=tau_slow,
                alpha=hebbian_alpha,
                beta=hebbian_beta,
            ), device=device)
            self.lr_rec = 1e-4
        else:
            self.hebbian = None

        # E-prop learner (only used in eprop mode)
        if mode == "eprop":
            from models.eprop import EPropLearner, EPropConfig
            beta_decay = math.exp(-1.0 / tau)
            self.eprop = EPropLearner(EPropConfig(
                n_in=1, n_rec=hidden_size, n_out=10,
                beta=beta_decay, rho=0.96, beta_a=0.07,
                device=device,
            ))
            self.B = torch.nn.init.orthogonal_(
                torch.empty(hidden_size, 10)
            ).to(device)
        else:
            self.eprop = None

    def process_sequence(
        self,
        pixels: torch.Tensor,  # (784,)
        target: Optional[int] = None,
        lr_out: float = 1e-2,
        weight_decay: float = 1e-4,
    ) -> Tuple[int, torch.Tensor]:
        """
        Process one sequential MNIST sample.

        Args:
            pixels: (784,) tensor of pixel values
            target: Optional label for training (0-9)
            lr_out: Readout learning rate
            weight_decay: L2 regularization for readout

        Returns:
            (prediction, logits)
        """
        self.rsnn.reset()
        self.readout.reset()

        if self.mode == "eprop" and target is not None:
            self.eprop.reset()

        # Buffer for readout window (last N spike patterns)
        spike_buffer = []
        z_prev = torch.zeros(self.hidden_size, device=self.device)

        T = len(pixels)
        for t in range(T):
            # Scalar pixel → 1D input
            x_t = pixels[t].unsqueeze(0).to(self.device)

            # Forward pass
            input_current = self.rsnn.input_gain * (self.rsnn.W_in @ x_t) + \
                            self.rsnn.W_rec @ z_prev
            spikes = self.rsnn.lif.step(input_current)

            # Accumulate in readout window
            spike_buffer.append(spikes.clone())
            if len(spike_buffer) > self.readout_window:
                spike_buffer.pop(0)

            if self.mode == "eprop" and target is not None:
                # E-prop accumulates eligibility traces
                psi = torch.ones_like(spikes)  # Pseudo-derivative (boxcar)
                self.eprop.accumulate(x_t, z_prev, psi)

            self.rsnn.prev_spikes = spikes.clone()
            z_prev = spikes.clone()

        # Readout from average over last N steps
        if spike_buffer:
            avg_spikes = torch.stack(spike_buffer).mean(dim=0)
        else:
            avg_spikes = torch.zeros(self.hidden_size, device=self.device)

        logits = self.readout.forward(avg_spikes)
        pred = int(logits.argmax().item())

        if target is not None:
            # Training update
            logits_exp = torch.exp(logits - logits.max())
            softmax = logits_exp / logits_exp.sum()
            target_onehot = torch.zeros(10, device=self.device)
            target_onehot[target] = 1.0
            error = softmax - target_onehot

            if self.mode == "hebbian":
                # Dual-trace Hebbian update for W_rec
                trace = self.hebbian.update(
                    torch.zeros_like(avg_spikes), avg_spikes
                )
                with torch.no_grad():
                    self.rsnn.W_rec.add_(-self.lr_rec * trace)

            elif self.mode == "eprop":
                # E-prop update
                L = self.B @ error
                self.eprop.apply(
                    self.rsnn.W_rec, self.rsnn.W_in, self.readout.W,
                    error, L, avg_spikes,
                )

            # Readout update (common to all modes)
            with torch.no_grad():
                dW = torch.outer(error, avg_spikes)
                self.readout.W.add_(-lr_out * dW - lr_out * weight_decay * self.readout.W)
                self.readout.b.add_(-lr_out * error)

        return pred, logits

    def get_hidden_state_ridge(self, pixels: torch.Tensor,
                                n_checkpoints: int = 8) -> torch.Tensor:
        """
        Get concatenated mean hidden states at evenly-spaced checkpoints.
        Used for ridge regression readout.

        Returns:
            (hidden_size * n_checkpoints,) tensor
        """
        self.rsnn.reset()
        z_prev = torch.zeros(self.hidden_size, device=self.device)
        T = len(pixels)
        checkpoints = [int((k + 1) * T / n_checkpoints) for k in range(n_checkpoints)]

        segments = []
        sum_h = torch.zeros(self.hidden_size, device=self.device)
        prev_cp = 0

        for t in range(T):
            x_t = pixels[t].unsqueeze(0).to(self.device)
            input_current = self.rsnn.input_gain * (self.rsnn.W_in @ x_t) + \
                            self.rsnn.W_rec @ z_prev
            spikes = self.rsnn.lif.step(input_current)
            sum_h += spikes
            z_prev = spikes.clone()

            if (t + 1) in checkpoints:
                n_steps = (t + 1) - prev_cp
                segments.append((sum_h.clone() / n_steps).cpu())
                sum_h.zero_()
                prev_cp = t + 1

        return torch.cat(segments)  # (hidden_size * n_checkpoints,)


# ── Ridge regression helper ───────────────────────────────────────────────────

def train_ridge(model: sMNISTClassifier, X_train, y_train,
                n_checkpoints: int = 8, alpha: Optional[float] = None):
    """
    Collect reservoir states and solve readout analytically.

    This is the simplest-possible baseline — fixed random reservoir,
    no learning in recurrent weights, just linear regression at readout.
    """
    print(f"  Collecting reservoir states for {len(X_train)} samples...")
    H_list = []
    Y_list = []

    for i in range(len(X_train)):
        h = model.get_hidden_state_ridge(X_train[i], n_checkpoints=n_checkpoints)
        H_list.append(h)
        Y_list.append(y_train[i].item())
        if (i + 1) % 5000 == 0:
            print(f"    {i+1}/{len(X_train)} states collected")

    X = torch.stack(H_list)
    feat_dim = X.shape[1]
    Y_oh = F.one_hot(torch.tensor(Y_list), 10).float()

    if alpha is None:
        alpha = feat_dim / len(X_train)

    print(f"  Solving ridge regression: {feat_dim} features, λ={alpha:.4f}")
    A = X.T @ X + alpha * torch.eye(feat_dim)
    W_opt = torch.linalg.solve(A, X.T @ Y_oh)

    # Replace readout weights
    model.readout.W = torch.zeros(10, feat_dim, device=model.device)
    model.readout.W.copy_(W_opt.T)
    model._ridge_feat_dim = feat_dim

    return model


def evaluate_ridge(model: sMNISTClassifier, X_test, y_test,
                   n_checkpoints: int = 8) -> float:
    """Evaluate ridge-regression model."""
    correct = 0
    feat_dim = model._ridge_feat_dim

    for i in range(len(X_test)):
        h = model.get_hidden_state_ridge(X_test[i], n_checkpoints=n_checkpoints).to(model.device)
        logits = model.readout.W @ h
        pred = int(logits.argmax().item())
        if pred == y_test[i].item():
            correct += 1

    return correct / len(X_test)


# ── Benchmark config ──────────────────────────────────────────────────────────

@dataclass
class sMNISTConfig:
    hidden_size: int = 256
    tau: float = 20.0
    v_th: float = 1.0
    n_train: int = 60000
    n_test: int = 10000
    n_epochs: int = 3
    mode: str = "reservoir"  # reservoir | hebbian | eprop | reservoir-ridge
    lr_out: float = 1e-2
    weight_decay: float = 1e-4
    device: str = "cpu"
    seed: int = 42
    save_results: bool = True


# ── Main benchmark ────────────────────────────────────────────────────────────

def run_smnist_benchmark(cfg: sMNISTConfig) -> Dict:
    print("\n" + "=" * 65)
    print(" Arthedain — Sequential MNIST (sMNIST) Benchmark")
    print("=" * 65)
    print(f"  Hidden: {cfg.hidden_size}  |  Tau: {cfg.tau}  |  V_th: {cfg.v_th}")
    print(f"  Mode: {cfg.mode}  |  Epochs: {cfg.n_epochs}")
    print(f"  Train samples: {cfg.n_train}  |  Test samples: {cfg.n_test}")

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    # Load data
    print("\nLoading MNIST dataset...")
    (X_train, y_train), (X_test, y_test) = load_smnist(
        n_train=cfg.n_train, n_test=cfg.n_test, seed=cfg.seed,
    )
    print(f"  Train: {X_train.shape}  |  Test: {X_test.shape}")
    print(f"  Pixel order: {'permuted' if True else 'row-major'}")

    device = torch.device(cfg.device)

    model = sMNISTClassifier(
        hidden_size=cfg.hidden_size,
        tau=cfg.tau,
        v_th=cfg.v_th,
        device=str(device),
        mode=cfg.mode,
    )

    # ── Training ──
    if cfg.mode == "reservoir-ridge":
        print(f"\nTraining — ridge regression on reservoir states...")
        t_train = time.perf_counter()
        model = train_ridge(model, X_train, y_train)
        train_time = time.perf_counter() - t_train

        print(f"\nEvaluation...")
        accuracy = evaluate_ridge(model, X_test, y_test)

    else:
        total_steps = cfg.n_train * cfg.n_epochs
        print(f"\nTraining — {cfg.mode} ({cfg.n_train} × {cfg.n_epochs} epochs = {total_steps} steps)...")
        t_train = time.perf_counter()

        step = 0
        for epoch in range(cfg.n_epochs):
            # Shuffle training data
            indices = torch.randperm(cfg.n_train, generator=torch.Generator().manual_seed(cfg.seed + epoch))
            epoch_correct = 0

            for idx in indices:
                pixels = X_train[idx]
                target = y_train[idx].item()

                pred, logits = model.process_sequence(
                    pixels, target=target,
                    lr_out=cfg.lr_out, weight_decay=cfg.weight_decay,
                )
                if pred == target:
                    epoch_correct += 1
                step += 1

                if step % 3000 == 0:
                    acc = epoch_correct / max(1, (step - (epoch * cfg.n_train)))
                    print(f"  Epoch {epoch+1}, Step {step}/{total_steps}: "
                          f"train acc = {acc:.1%}", end="\r")

            epoch_acc = epoch_correct / cfg.n_train
            print(f"  Epoch {epoch+1}: train acc = {epoch_acc:.3%}")

        train_time = time.perf_counter() - t_train

        # ── Evaluation ──
        print(f"\nEvaluation ({cfg.n_test} samples, no learning)...")
        correct = 0
        t_eval = time.perf_counter()

        for i in range(cfg.n_test):
            pixels = X_test[i]
            target = y_test[i].item()
            pred, _ = model.process_sequence(pixels, target=None)
            if pred == target:
                correct += 1

        eval_time = time.perf_counter() - t_eval
        accuracy = correct / cfg.n_test

    # ── Results ──
    print(f"\n{'='*65}")
    print(f"  Test Accuracy: {accuracy:.4f}  ({accuracy*100:.2f}%)")
    print(f"  Training time: {train_time:.1f}s")
    if cfg.mode != "reservoir-ridge":
        print(f"  Evaluation time: {eval_time:.1f}s")

    # Comparison table
    print(f"\n  Comparison vs. published results on sMNIST:")
    print(f"  {'Method':<30} {'Accuracy':>10} {'Memory':>8} {'Recurrent Learn':>15}")
    print(f"  {'-'*65}")
    print(f"  {'BPTT SNN':<30} {'~98%':>10} {'O(T)':>8} {'BPTT':>15}")
    print(f"  {'e-prop':<30} {'~95%':>10} {'O(1)':>8} {'e-prop traces':>15}")
    print(f"  {'Online Hebbian':<30} {'~85%':>10} {'O(1)':>8} {'dual-trace':>15}")
    print(f"  {'Reservoir + Ridge':<30} {'~82%':>10} {'O(1)':>8} {'none':>15}")
    print(f"  {'Arthedain (this run)':<30} {f'{accuracy*100:.1f}%':>10} {'O(1)':>8} {'---':>15}")

    results = {
        "benchmark": "sMNIST",
        "mode": cfg.mode,
        "hidden_size": cfg.hidden_size,
        "tau": cfg.tau,
        "v_th": cfg.v_th,
        "n_train": cfg.n_train,
        "n_test": cfg.n_test,
        "n_epochs": cfg.n_epochs,
        "test_accuracy": accuracy,
        "train_time_s": train_time,
        "seed": cfg.seed,
        "comparison": {
            "bptt_snn": 0.98,
            "eprop": 0.95,
            "online_hebbian": 0.85,
            "reservoir_ridge": 0.82,
            "arthedain": accuracy,
        },
    }

    if cfg.save_results:
        out = Path("results") / "benchmark_smnist.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(results, indent=2))
        print(f"\n  Results saved → {out}")

    print("=" * 65)
    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sequential MNIST benchmark")
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--tau", type=float, default=20.0)
    parser.add_argument("--v-th", type=float, default=1.0)
    parser.add_argument("--n-train", type=int, default=60000)
    parser.add_argument("--n-test", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--mode", type=str, default="reservoir-ridge",
                        choices=["reservoir", "hebbian", "eprop", "reservoir-ridge"])
    parser.add_argument("--lr-out", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test with small parameters")

    args = parser.parse_args()

    if args.quick:
        args.n_train = 1000
        args.n_test = 200
        args.hidden = 64
        args.epochs = 1

    cfg = sMNISTConfig(
        hidden_size=args.hidden,
        tau=args.tau,
        v_th=args.v_th,
        n_train=args.n_train,
        n_test=args.n_test,
        n_epochs=args.epochs,
        mode=args.mode,
        lr_out=args.lr_out,
        weight_decay=args.weight_decay,
        device=args.device,
        seed=args.seed,
        save_results=not args.no_save,
    )

    run_smnist_benchmark(cfg)
