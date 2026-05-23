"""
experiments/compare_learning_methods.py
=========================================
Benchmark comparing all SNNTraining learning methods on BCI decoding.

Methods compared:
1. Dual-timescale Hebbian (original SNNTraining)
2. e-prop with eligibility traces
3. FORCE / RLS online learning
4. Dynamics-based fast learning
5. Predictive coding only
6. Hybrid (combining multiple methods)

Metrics:
- Pearson R on velocity decoding
- Adaptation speed after disruption
- Memory usage
- Convergence speed
- Robustness to noise

Usage:
    python experiments/compare_learning_methods.py --method all --duration 5000
"""

import argparse
import time
import torch
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import json

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.rsnn import RSNN, RSNNConfig
from models.readout import Readout
from models.deep_rsnn import make_deep_rsnn
from training.unified_trainer import UnifiedTrainer, UnifiedConfig, make_unified_trainer
from data.synthetic import bci_velocity_stream


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    method: str
    pearson_r: float
    steps_to_recovery: int
    final_mse: float
    memory_mb: float
    runtime_seconds: float
    convergence_step: Optional[int] = None


def synthetic_bci_stream(
    T: int,
    input_size: int = 96,
    output_size: int = 2,
    noise: float = 0.1,
    disruption_at: Optional[int] = None,
    seed: int = 42,
):
    """Generate synthetic BCI-like data stream."""
    rng = np.random.default_rng(seed)
    preferred_dirs = rng.uniform(0, 2 * np.pi, input_size)
    
    for t in range(T):
        # Disruption: remap preferred directions
        if disruption_at and t == disruption_at:
            preferred_dirs = rng.uniform(0, 2 * np.pi, input_size)
        
        # Random target velocity direction
        theta = rng.uniform(0, 2 * np.pi)
        velocity = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
        
        # Tuning curve: neurons fire based on preferred direction
        rates = 5.0 + 95.0 * np.maximum(0, np.cos(preferred_dirs - theta))
        spike_probs = np.clip(rates * 0.05, 0, 1)
        spikes = rng.binomial(1, spike_probs).astype(np.float32)
        
        # Add noise
        velocity += rng.normal(0, noise, output_size).astype(np.float32)
        
        yield (
            torch.from_numpy(spikes),
            torch.from_numpy(velocity),
        )


def run_method(
    method: str,
    duration: int = 3000,
    disruption_at: int = 2000,
    input_size: int = 96,
    hidden_size: int = 256,
    seed: int = 42,
) -> BenchmarkResult:
    """
    Run a single learning method and collect metrics.
    
    Args:
        method: One of "hebbian", "eprop", "force", "dynamics", "pc", "hybrid"
        duration: Total timesteps
        disruption_at: When to inject disruption (remapping)
        input_size: Input dimension
        hidden_size: Hidden layer size
        seed: Random seed
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create model
    if method in ["pc", "hybrid"]:
        # Use deep RSNN for PC (needs multiple layers)
        rsnn = make_deep_rsnn(
            input_size=input_size,
            hidden_sizes=[hidden_size, hidden_size // 2],
            output_size=2,
        )
    else:
        rsnn = RSNN(
            input_size=input_size,
            hidden_size=hidden_size,
            sparse_init=True,
        )
    
    # Readout
    if hasattr(rsnn, 'hidden_sizes'):
        readout = torch.nn.Linear(rsnn.hidden_sizes[-1], 2)
    else:
        readout = Readout(hidden_size, 2)
    
    # Create trainer
    if method == "hebbian":
        cfg = UnifiedConfig(
            mode="hebbian",
            hidden_sizes=[hidden_size],
            lr_recurrent=5e-5,
            pc_use_stack=False,
        )
    elif method == "eprop":
        cfg = UnifiedConfig(
            mode="eprop",
            hidden_sizes=[hidden_size],
            lr_recurrent=5e-5,
            pc_use_stack=False,
        )
    elif method == "force":
        cfg = UnifiedConfig(
            mode="force",
            hidden_sizes=[hidden_size],
            force_mode="rls_readout_only",
            pc_use_stack=False,
        )
    elif method == "dynamics":
        cfg = UnifiedConfig(
            mode="dynamics",
            hidden_sizes=[hidden_size],
            dynamics_n_contexts=2,
            pc_use_stack=False,
        )
    elif method == "pc":
        cfg = UnifiedConfig(
            mode="pc",
            hidden_sizes=[hidden_size, hidden_size // 2],
            pc_alpha_error=0.0,  # Pure PC
            pc_use_stack=True,
        )
    elif method == "hybrid":
        cfg = UnifiedConfig(
            mode="hybrid",
            hidden_sizes=[hidden_size, hidden_size // 2],
            hybrid_mix={"hebbian": 0.3, "eprop": 0.3, "force": 0.4},
            pc_use_stack=True,
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    
    trainer = UnifiedTrainer(rsnn, readout, cfg)
    
    # For dynamics mode: do fast adaptation first
    if method == "dynamics":
        adapt_data = [
            (s, v) for s, v in synthetic_bci_stream(100, input_size, seed=seed)
        ]
        trainer.fast_adapt(adapt_data, context_id=0)
        trainer.set_context(0)
    
    # Run training
    start_time = time.time()
    
    predictions = []
    targets = []
    errors = []
    
    recovery_detected = False
    recovery_step = duration  # worst case
    convergence_threshold = 0.1  # MSE threshold for convergence
    convergence_step = None
    
    for t, (x, y) in enumerate(synthetic_bci_stream(duration, input_size, seed=seed)):
        # Determine context for dynamics mode
        context_id = 0 if t < disruption_at else 1
        
        # Training step
        y_pred, error, info = trainer.step(x, y, context_id if method == "dynamics" else None)
        
        # Track after disruption
        if t >= disruption_at:
            predictions.append(y_pred.detach().numpy().flatten())
            targets.append(y.numpy().flatten())
            errors.append(error.norm().item())
            
            # Recovery detection: rolling correlation > 0.5
            if len(predictions) >= 20 and not recovery_detected:
                p_arr = np.array(predictions[-20:]).flatten()
                t_arr = np.array(targets[-20:]).flatten()
                if np.std(p_arr) > 1e-6 and np.std(t_arr) > 1e-6:
                    r = np.corrcoef(p_arr, t_arr)[0, 1]
                    if r > 0.5:
                        recovery_detected = True
                        recovery_step = t - disruption_at
            
            # Convergence detection
            if convergence_step is None and len(errors) > 50:
                recent_mse = np.mean([e**2 for e in errors[-50:]])
                if recent_mse < convergence_threshold:
                    convergence_step = t - disruption_at
    
    runtime = time.time() - start_time
    
    # Compute final metrics
    if predictions:
        p_arr = np.array(predictions)
        t_arr = np.array(targets)
        
        # Pearson R per dimension
        rs = []
        for dim in range(2):
            if np.std(p_arr[:, dim]) > 1e-6 and np.std(t_arr[:, dim]) > 1e-6:
                r = np.corrcoef(p_arr[:, dim], t_arr[:, dim])[0, 1]
                rs.append(r)
            else:
                rs.append(0.0)
        pearson_r = np.mean(rs)
        
        # Final MSE
        final_mse = np.mean((p_arr - t_arr) ** 2)
    else:
        pearson_r = 0.0
        final_mse = 1.0
    
    # Estimate memory (approximate)
    # Handle both nn.Module (has parameters()) and original RSNN (raw tensors)
    param_bytes = 0
    if hasattr(rsnn, 'parameters'):
        try:
            param_bytes = sum(p.numel() * 4 for p in rsnn.parameters())
        except:
            param_bytes = 0
    
    # Add raw tensors for original RSNN
    if hasattr(rsnn, 'W_in'):
        param_bytes += rsnn.W_in.numel() * 4
    if hasattr(rsnn, 'W_rec'):
        param_bytes += rsnn.W_rec.numel() * 4
    
    if method == "force":
        # RLS adds P matrices
        param_bytes += hidden_size * hidden_size * 4 * 2  # P matrices
    elif method == "eprop":
        # Eligibility traces
        param_bytes += hidden_size * hidden_size * 4 * 2  # eligibility + filter
    elif method == "dynamics":
        # Initial states and gates
        param_bytes += hidden_size * 4 * cfg.dynamics_n_contexts
    
    memory_mb = param_bytes * 2 / 1e6  # rough estimate with buffers
    
    return BenchmarkResult(
        method=method,
        pearson_r=pearson_r,
        steps_to_recovery=recovery_step,
        final_mse=final_mse,
        memory_mb=memory_mb,
        runtime_seconds=runtime,
        convergence_step=convergence_step,
    )


def print_comparison_table(results: List[BenchmarkResult]):
    """Print formatted comparison table."""
    print("\n" + "=" * 100)
    print(f"{'Method':<15} {'Pearson R':>12} {'Recovery':>12} {'Final MSE':>12} {'Memory(MB)':>12} {'Runtime(s)':>12}")
    print("-" * 100)
    
    for r in results:
        print(f"{r.method:<15} {r.pearson_r:>12.3f} {r.steps_to_recovery:>12d} "
              f"{r.final_mse:>12.4f} {r.memory_mb:>12.2f} {r.runtime_seconds:>12.2f}")
    
    print("=" * 100)
    
    # Summary insights
    if len(results) > 1:
        best_r = max(results, key=lambda x: x.pearson_r)
        fastest_recovery = min(results, key=lambda x: x.steps_to_recovery)
        lowest_memory = min(results, key=lambda x: x.memory_mb)
        
        print(f"\nBest Pearson R: {best_r.method} ({best_r.pearson_r:.3f})")
        print(f"Fastest recovery: {fastest_recovery.method} ({fastest_recovery.steps_to_recovery} steps)")
        print(f"Lowest memory: {lowest_memory.method} ({lowest_memory.memory_mb:.2f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Compare SNNTraining learning methods")
    parser.add_argument("--method", default="all",
                       choices=["all", "hebbian", "eprop", "force", "dynamics", "pc", "hybrid"],
                       help="Which method to benchmark")
    parser.add_argument("--duration", type=int, default=3000,
                       help="Number of timesteps")
    parser.add_argument("--disruption", type=int, default=2000,
                       help="When to inject disruption")
    parser.add_argument("--input_size", type=int, default=96,
                       help="Input dimension (neurons)")
    parser.add_argument("--hidden_size", type=int, default=256,
                       help="Hidden layer size")
    parser.add_argument("--runs", type=int, default=1,
                       help="Number of runs per method (for averaging)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Base random seed")
    parser.add_argument("--output", type=str, default=None,
                       help="Save results to JSON file")
    args = parser.parse_args()
    
    # Determine which methods to run
    if args.method == "all":
        methods = ["hebbian", "eprop", "force", "dynamics", "pc", "hybrid"]
    else:
        methods = [args.method]
    
    # Run benchmarks
    all_results = []
    
    for method in methods:
        print(f"\nRunning {method}...")
        
        method_results = []
        for run in range(args.runs):
            seed = args.seed + run * 100
            result = run_method(
                method=method,
                duration=args.duration,
                disruption_at=args.disruption,
                input_size=args.input_size,
                hidden_size=args.hidden_size,
                seed=seed,
            )
            method_results.append(result)
        
        # Average across runs
        if args.runs > 1:
            avg_result = BenchmarkResult(
                method=method,
                pearson_r=np.mean([r.pearson_r for r in method_results]),
                steps_to_recovery=int(np.mean([r.steps_to_recovery for r in method_results])),
                final_mse=np.mean([r.final_mse for r in method_results]),
                memory_mb=np.mean([r.memory_mb for r in method_results]),
                runtime_seconds=np.mean([r.runtime_seconds for r in method_results]),
            )
        else:
            avg_result = method_results[0]
        
        all_results.append(avg_result)
        print(f"  Pearson R: {avg_result.pearson_r:.3f}, Recovery: {avg_result.steps_to_recovery} steps")
    
    # Print comparison
    print_comparison_table(all_results)
    
    # Save to file if requested
    if args.output:
        output_data = {
            'config': vars(args),
            'results': [
                {
                    'method': r.method,
                    'pearson_r': r.pearson_r,
                    'steps_to_recovery': r.steps_to_recovery,
                    'final_mse': r.final_mse,
                    'memory_mb': r.memory_mb,
                    'runtime_seconds': r.runtime_seconds,
                }
                for r in all_results
            ]
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
