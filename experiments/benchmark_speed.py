"""
benchmark_speed.py
===================
Benchmark training speed improvements.

Compares training time per step across all methods before and after optimizations.

Usage:
    python experiments/benchmark_speed.py --method all --steps 1000
"""

import argparse
import time
import torch
import numpy as np
from dataclasses import dataclass
from typing import Dict, List
import json

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.rsnn import RSNN, RSNNConfig
from models.readout import Readout, ReadoutConfig
from models.hebbian import DualHebbianAccumulator, HebbianConfig
from training.unified_trainer import UnifiedTrainer, UnifiedConfig, make_unified_trainer
from training.online_trainer import OnlineTrainer, TrainerConfig
from data.synthetic import bci_velocity_stream


def benchmark_method(
    method: str,
    steps: int = 1000,
    input_size: int = 100,
    hidden_size: int = 128,
    seed: int = 42,
) -> Dict[str, float]:
    """Benchmark a single training method."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device('cpu')  # Use CPU for fair comparison
    
    # Create models
    rsnn = RSNN(RSNNConfig(input_size=input_size, hidden_size=hidden_size, device=device))
    readout = Readout(ReadoutConfig(hidden_size, 2, mode="smoothed"))
    
    # Create trainer based on method
    if method == "hebbian":
        from models.hebbian import DualHebbianAccumulator, HebbianConfig
        hebbian = DualHebbianAccumulator(
            HebbianConfig(shape=(hidden_size, hidden_size), tau_fast=5.0, tau_slow=50.0)
        )
        trainer = OnlineTrainer(rsnn, readout, hebbian, lr_readout=2e-3, lr_recurrent=5e-5, device=device)
        
    elif method == "hybrid":
        cfg = UnifiedConfig(
            mode="hybrid",
            hidden_sizes=[hidden_size],
            lr_readout=2e-3,
            lr_recurrent=5e-5,
            use_cached_forward=True,
            skip_threshold=0.001,
            lr_warmup_steps=100,
        )
        trainer = UnifiedTrainer(rsnn, readout, cfg)
        
    elif method == "eprop":
        cfg = UnifiedConfig(
            mode="eprop",
            hidden_sizes=[hidden_size],
            lr_recurrent=5e-5,
            use_jit=True,
        )
        trainer = UnifiedTrainer(rsnn, readout, cfg)
        
    elif method == "force":
        cfg = UnifiedConfig(
            mode="force",
            hidden_sizes=[hidden_size],
            lr_readout=2e-3,
            lr_recurrent=5e-5,
        )
        trainer = UnifiedTrainer(rsnn, readout, cfg)
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Warmup (first steps are slower due to JIT compilation and initialization)
    stream = bci_velocity_stream(T=100, input_size=input_size, seed=seed)
    for i, (x, y) in enumerate(stream):
        if method == "hebbian":
            trainer.step(x, y)
        else:
            trainer.step(x, target=y)
    
    # Benchmark
    stream = bci_velocity_stream(T=steps, input_size=input_size, seed=seed + 1)
    times = []
    
    start_time = time.perf_counter()
    
    for i, (x, y) in enumerate(stream):
        step_start = time.perf_counter()
        
        if method == "hebbian":
            y_pred, error = trainer.step(x, y)
        else:
            y_pred, error, info = trainer.step(x, target=y)
        
        step_end = time.perf_counter()
        times.append((step_end - step_start) * 1000)  # Convert to ms
    
    end_time = time.perf_counter()
    total_time = end_time - start_time
    
    # Compute statistics
    times_arr = np.array(times)
    
    return {
        "method": method,
        "steps": steps,
        "total_time_s": total_time,
        "mean_ms": float(np.mean(times_arr)),
        "median_ms": float(np.median(times_arr)),
        "std_ms": float(np.std(times_arr)),
        "min_ms": float(np.min(times_arr)),
        "max_ms": float(np.max(times_arr)),
        "steps_per_second": steps / total_time,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark training speed")
    parser.add_argument("--method", type=str, default="all",
                       choices=["all", "hebbian", "hybrid", "eprop", "force"],
                       help="Method to benchmark")
    parser.add_argument("--steps", type=int, default=1000,
                       help="Number of training steps")
    parser.add_argument("--save", type=str, default="results/benchmark_speed.json",
                       help="Path to save results JSON")
    args = parser.parse_args()
    
    methods = ["hebbian", "hybrid", "eprop", "force"] if args.method == "all" else [args.method]
    
    print("=" * 60)
    print(f"Arthedain Training Speed Benchmark")
    print(f"Steps: {args.steps}")
    print("=" * 60)
    print()
    
    results = {}
    
    for method in methods:
        print(f"Benchmarking {method}...")
        try:
            result = benchmark_method(method, steps=args.steps)
            results[method] = result
            
            print(f"  Mean: {result['mean_ms']:.3f} ms/step")
            print(f"  Median: {result['median_ms']:.3f} ms/step")
            print(f"  Steps/sec: {result['steps_per_second']:.1f}")
            print(f"  Total: {result['total_time_s']:.2f} s")
            print()
            
        except Exception as e:
            print(f"  ERROR: {e}")
            print()
    
    # Summary
    if len(results) > 1:
        print("=" * 60)
        print("Summary (mean ms/step):")
        print("-" * 60)
        for method, result in results.items():
            print(f"  {method:12s}: {result['mean_ms']:8.3f} ms/step ({result['steps_per_second']:6.1f} steps/s)")
        print("=" * 60)
        print()
    
    # Save results
    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    with open(args.save, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {args.save}")
    
    # Optimization notes
    print()
    print("Optimizations applied:")
    print("  - Shared forward pass (hybrid mode)")
    print("  - JIT compilation (eprop, hebbian)")
    print("  - In-place operations (all methods)")
    print("  - Learning rate warmup (all methods)")
    print("  - Gradient clipping (all methods)")
    print("  - FORCE skip on low error (hybrid mode)")
    print("  - Adaptive tau eligibility (eprop)")
    print("  - Adaptive forgetting factor (force)")
    print("  - Error variance tracking (force)")


if __name__ == "__main__":
    main()
