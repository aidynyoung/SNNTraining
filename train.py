"""
train.py — train a spiking neural network on synthetic or real data.

Usage:
    python train.py                          # synthetic 4-class (default)
    python train.py --task shd              # SHD neuromorphic (requires h5py)
    python train.py --task bci              # BCI velocity (requires scipy + data)
    python train.py --method refine         # prototype refinement
    python train.py --method fallback       # SNN readout fallback
    python train.py --hidden 256 --dim 4096 # larger model
    python train.py --config configs/default.yaml
"""

import argparse
import time
import torch
import yaml
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Train a spiking neural network")
    p.add_argument("--task",    default="synthetic", choices=["synthetic", "shd", "bci"])
    p.add_argument("--method",  default="bundle",    choices=["bundle", "refine", "fallback", "lehdc"])
    p.add_argument("--hidden",  type=int, default=128)
    p.add_argument("--dim",     type=int, default=4096)
    p.add_argument("--classes", type=int, default=4)
    p.add_argument("--epochs",  type=int, default=5)
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--config",  type=str, default=None)
    p.add_argument("--device",  type=str, default=None)
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def make_synthetic_data(n_classes: int, n_samples: int = 200, input_size: int = 20):
    """Generate linearly separable synthetic spike data."""
    torch.manual_seed(42)
    X, y = [], []
    for c in range(n_classes):
        mu = torch.zeros(input_size)
        mu[c * (input_size // n_classes):(c + 1) * (input_size // n_classes)] = 1.0
        X.append(mu.unsqueeze(0).repeat(n_samples // n_classes, 1) + 0.1 * torch.randn(n_samples // n_classes, input_size))
        y.extend([c] * (n_samples // n_classes))
    return torch.cat(X), torch.tensor(y)


def train_bundle(pipe, X_train, y_train):
    """Single-pass prototype bundling — fastest, no gradients."""
    print("  Method: prototype bundling (single pass)")
    t0 = time.time()
    for x, label in zip(X_train, y_train):
        pipe.train_step(x, int(label.item()))
    pipe.finalize()
    print(f"  Trained in {1000*(time.time()-t0):.1f}ms")


def train_fallback(pipe, X_train, y_train):
    """Prototype bundling + parallel LMS readout training."""
    print("  Method: prototype bundling + SNN fallback readout")
    t0 = time.time()
    for x, label in zip(X_train, y_train):
        pipe.train_step(x, int(label.item()))
    pipe.finalize()
    print(f"  Trained in {1000*(time.time()-t0):.1f}ms")


def evaluate(pipe, X_test, y_test) -> float:
    correct = 0
    for x, label in zip(X_test, y_test):
        pipe.reset()
        pred, conf, _ = pipe.hdc_infer()
        # push through snn_step first
        for _ in range(pipe.cfg.window):
            pipe.snn_step(x)
        pred, conf, _ = pipe.hdc_infer()
        correct += int(pred == label.item())
    return correct / len(y_test)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    if args.config:
        cfg_dict = load_config(args.config)
        for k, v in cfg_dict.items():
            if hasattr(args, k):
                setattr(args, k, v)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nSNNTraining")
    print(f"  task={args.task}  method={args.method}  hidden={args.hidden}  dim={args.dim}")
    print(f"  device={device}  seed={args.seed}\n")

    # --- Data ---
    if args.task == "synthetic":
        input_size = args.classes * 5
        X, y = make_synthetic_data(args.classes, n_samples=400, input_size=input_size)
        split = int(0.8 * len(X))
        X_train, y_train = X[:split], y[:split]
        X_test,  y_test  = X[split:], y[split:]
        print(f"  Synthetic data: {len(X_train)} train / {len(X_test)} test, {input_size} features")

    elif args.task == "shd":
        try:
            from data.loaders import load_shd
            X_train, y_train, X_test, y_test = load_shd()
            input_size = X_train.shape[-1]
            args.classes = int(y_train.max().item()) + 1
        except Exception as e:
            print(f"  SHD load failed: {e}")
            print("  Install h5py and download from https://zenkelab.org/resources/spiking-heidelberg-datasets/")
            return

    elif args.task == "bci":
        try:
            from data.loaders import load_bci_indy
            X_train, y_train, X_test, y_test = load_bci_indy()
            input_size = X_train.shape[-1]
        except Exception as e:
            print(f"  BCI load failed: {e}")
            print("  Install scipy and place data/indy/indy_2016-10-05_1.mat (CRCNS pmd-1)")
            return
    else:
        raise ValueError(args.task)

    # --- Model ---
    from models.snn_hdc_pipeline import SNNHDCPipeline, PipelineConfig

    pipe_cfg = PipelineConfig(
        input_size=input_size if args.task == "synthetic" else X_train.shape[-1],
        hidden_size=args.hidden,
        n_classes=args.classes,
        hdc_dim=args.dim,
        use_snn_fallback=(args.method == "fallback"),
        gate_threshold=0.4,
        fallback_lr=0.02,
        device=device,
    )
    pipe = SNNHDCPipeline(pipe_cfg)
    n_params = pipe.rsnn.W_in.numel() + pipe.rsnn.W_rec.numel()
    print(f"  RSNN params: {n_params:,}  |  HDC dim: {args.dim}")

    # --- Train ---
    print()
    if args.method in ("bundle", "fallback"):
        train_fn = train_fallback if args.method == "fallback" else train_bundle
        train_fn(pipe, X_train, y_train)

    elif args.method == "refine":
        from hdc.continual_hdc import ClassMeanHDCClassifier
        print("  Method: class-mean init + RefineHD (Harun & Kanan 2025)")
        clf = ClassMeanHDCClassifier(dim=args.dim, n_classes=args.classes)
        t0 = time.time()
        clf.fit(X_train.numpy(), y_train.numpy())
        clf.refine(X_train.numpy(), y_train.numpy(), epochs=args.epochs)
        print(f"  Refined in {1000*(time.time()-t0):.1f}ms")
        acc = clf.score(X_test.numpy(), y_test.numpy())
        print(f"\n  Test accuracy: {acc:.1%}")
        return

    elif args.method == "lehdc":
        from hdc.adaptive_encoder import LeHDCEncoder
        print("  Method: LeHDC gradient training with STE binarization")
        enc = LeHDCEncoder(dim=args.dim, input_dim=input_size)
        t0 = time.time()
        enc.train_supervised(X_train.numpy(), y_train.numpy(), epochs=args.epochs, lr=1e-3)
        print(f"  Trained in {1000*(time.time()-t0):.1f}ms")
        return

    # --- Evaluate ---
    print()
    correct = 0
    for x, label in zip(X_test, y_test):
        pipe.reset()
        for _ in range(pipe.cfg.window):
            pipe.snn_step(x.to(device))
        pred, conf, _ = pipe.hdc_infer()
        correct += int(pred == label.item())
    acc = correct / len(y_test)

    summary = pipe.pipeline_summary()
    print(f"  Test accuracy : {acc:.1%}")
    print(f"  Mean similarity: {summary['last_sim']:.3f}")
    print(f"  SNN firing rate: {summary['snn_firing_rate']:.3f}")
    if pipe_cfg.use_snn_fallback:
        print(f"  HDC route: {summary['route_hdc']}  Fallback route: {summary['route_fallback']}")

    health = pipe.rsnn.network_health()
    print(f"\n  RSNN health: spectral_radius={health['spectral_radius']}  "
          f"sparsity={health['sparsity']}  edge_of_chaos={health['edge_of_chaos']}")

    # Save checkpoint
    out = Path("checkpoints")
    out.mkdir(exist_ok=True)
    ckpt = out / f"{args.task}_{args.method}_acc{acc:.2f}.pt"
    torch.save({
        "assoc_mem": pipe.assoc_mem.class_hvs,
        "config": pipe_cfg,
        "accuracy": acc,
    }, ckpt)
    print(f"\n  Saved checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
