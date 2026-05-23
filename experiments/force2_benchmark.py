"""
force2_benchmark.py
===================
Benchmark FORCE2 implementation against Nicola & Clopath 2017 paper.

Tests:
1. Simple oscillator (Figure 2a)
2. Coupled oscillators (Figure 2b)
3. Chaotic attractor (Figure 2c/d)
4. Ode to Joy pattern (Figure 3)

Metrics compared to paper:
- R² correlation between target and learned output
- Training convergence time
- Spectral radius effect on performance
- Multi-timescale synapse contribution

Usage:
    python experiments/force2_benchmark.py --test all --save_results
"""

from __future__ import annotations

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
import time
import argparse
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.force_enhanced import (
    PatternGenerator, ChaoticInitConfig, MultiTimescaleSynapseConfig,
    test_chaos_property
)
# Use LIF-based FORCE2 for spiking neurons (achieves paper's reported correlations)
from training.force2_lif_trainer import (
    FORCE2LIFTrainer, FORCE2LIFConfig,
    make_lif_force_trainer_for_oscillator, make_lif_force_trainer_for_chaos,
)


@dataclass
class BenchmarkResult:
    """Results from a single benchmark test."""
    test_name: str
    target_correlation: float
    mse_final: float
    training_time_ms: float
    n_steps: int
    spectral_radius: float
    convergence_step: Optional[int] = None
    extra_metrics: Dict = None
    
    def to_dict(self) -> dict:
        def to_float(v):
            if isinstance(v, (np.float32, np.float64)):
                return float(v)
            if isinstance(v, torch.Tensor):
                return v.item() if v.numel() == 1 else v.tolist()
            return v
        
        result = {
            "test_name": self.test_name,
            "target_correlation": to_float(self.target_correlation),
            "mse_final": to_float(self.mse_final),
            "training_time_ms": to_float(self.training_time_ms),
            "n_steps": self.n_steps,
            "spectral_radius": to_float(self.spectral_radius),
            "convergence_step": self.convergence_step,
        }
        if self.extra_metrics:
            result.update({k: to_float(v) for k, v in self.extra_metrics.items()})
        return result


class FORCEBenchmark:
    """Benchmark suite for FORCE2 trainer."""
    
    def __init__(self, device: str = "cpu", verbose: bool = True):
        self.device = device
        self.verbose = verbose
        self.results: List[BenchmarkResult] = []
    
    def log(self, msg: str):
        """Print if verbose."""
        if self.verbose:
            print(f"[FORCE2] {msg}")
    
    def test_simple_oscillator(
        self,
        freq: float = 2.0,
        n_neurons: int = 800,
        n_steps: int = 3000,
        n_train_steps: int = 2000,  # More training steps for better learning
    ) -> BenchmarkResult:
        """
        Test 1: Simple sinusoidal oscillator (Figure 2a from paper).
        
        Paper result: Networks with spectral radius > 1 can learn simple oscillators.
        """
        self.log(f"Testing simple oscillator (f={freq}Hz)...")
        
        # Generate target
        pattern = PatternGenerator.generate_oscillator(
            freq=freq, amplitude=1.0, n_steps=n_steps, dt=1.0
        ).to(self.device)
        
        # Create LIF-based trainer (spiking neurons for paper-level accuracy)
        trainer = make_lif_force_trainer_for_oscillator(
            freq=freq, n_neurons=n_neurons, device=self.device
        )
        
        start_time = time.time()
        
        # Train
        errors = []
        outputs = []
        
        trainer.reset_state()
        for t in range(n_train_steps):
            x = torch.zeros(1, device=self.device)  # No input
            y_pred, error = trainer.train_step(x, pattern[t])
            errors.append(error)
            outputs.append(y_pred.item())
        
        # Test (no training)
        test_outputs = []
        for t in range(n_train_steps, n_steps):
            x = torch.zeros(1, device=self.device)
            y_pred = trainer.step(x)
            test_outputs.append(y_pred.item())
        
        training_time = (time.time() - start_time) * 1000
        
        # Compute correlation
        all_outputs = outputs + test_outputs
        target_segment = pattern[:len(all_outputs)].cpu().numpy()
        output_arr = np.array(all_outputs)
        
        correlation = np.corrcoef(target_segment, output_arr)[0, 1]
        mse = np.mean((target_segment - output_arr) ** 2)
        
        # Find convergence (error < threshold)
        threshold = 0.1
        convergence = None
        for i, e in enumerate(errors):
            if e < threshold and all(e2 < threshold for e2 in errors[i:i+50] if i+50 < len(errors)):
                convergence = i
                break
        
        # Get firing rate stats
        stats = trainer.get_stats()
        
        result = BenchmarkResult(
            test_name=f"simple_oscillator_{freq}Hz",
            target_correlation=correlation,
            mse_final=mse,
            training_time_ms=training_time,
            n_steps=n_steps,
            spectral_radius=trainer.initial_spectral_radius,
            convergence_step=convergence,
            extra_metrics={
                "mean_firing_rate": stats.get("mean_firing_rate", 0.0),
            },
        )
        
        self.log(f"  Correlation: {correlation:.4f}, MSE: {mse:.6f}, Time: {training_time:.1f}ms")
        
        # Store for plotting
        self._last_outputs = all_outputs
        self._last_pattern = pattern.cpu().numpy()
        
        return result
    
    def test_coupled_oscillators(
        self,
        freqs: List[float] = [1.0, 3.0, 5.0],
        n_neurons: int = 1000,
        n_steps: int = 3000,
    ) -> BenchmarkResult:
        """
        Test 2: Sum of multiple oscillators (Figure 2b from paper).
        
        Tests network capacity for complex periodic patterns.
        """
        self.log(f"Testing coupled oscillators (freqs={freqs})...")
        
        pattern = PatternGenerator.generate_coupled_oscillators(
            freqs=freqs,
            amplitudes=[1.0, 0.5, 0.25],
            n_steps=n_steps,
            dt=1.0,
        ).to(self.device)
        
        # Use LIF-based trainer with higher spectral radius for complex pattern
        cfg = FORCE2LIFConfig(
            n_neurons=n_neurons,
            n_outputs=1,
            chaotic_cfg=ChaoticInitConfig(target_radius=1.6),
            train_readout=True,
        )
        trainer = FORCE2LIFTrainer(cfg, self.device)
        
        start_time = time.time()
        
        # Train/test split
        n_train = n_steps // 2
        
        trainer.reset_state()
        outputs = []
        
        for t in range(n_steps):
            x = torch.zeros(1, device=self.device)
            if t < n_train:
                y_pred, _ = trainer.train_step(x, pattern[t])
            else:
                y_pred = trainer.step(x)
            outputs.append(y_pred.item())
        
        training_time = (time.time() - start_time) * 1000
        
        # Evaluate on test set
        target_test = pattern[n_train:].cpu().numpy()
        output_test = np.array(outputs[n_train:])
        
        correlation = np.corrcoef(target_test, output_test)[0, 1]
        mse = np.mean((target_test - output_test) ** 2)
        
        result = BenchmarkResult(
            test_name="coupled_oscillators",
            target_correlation=correlation,
            mse_final=mse,
            training_time_ms=training_time,
            n_steps=n_steps,
            spectral_radius=trainer.initial_spectral_radius,
        )
        
        self.log(f"  Correlation: {correlation:.4f}, MSE: {mse:.6f}")
        
        return result
    
    def test_lorenz_attractor(
        self,
        n_neurons: int = 2000,
        n_steps: int = 5000,
    ) -> BenchmarkResult:
        """
        Test 3: Lorenz chaotic attractor (Figure 2c/d from paper).
        
        The classic test for FORCE - learning chaotic dynamics.
        """
        self.log("Testing Lorenz chaotic attractor...")
        
        # Generate target
        trajectory = PatternGenerator.generate_lorenz_attractor(
            n_steps=n_steps, dt=0.01
        )
        
        # Use x and y coordinates as targets (2D output)
        pattern = trajectory[:, :2].to(self.device)  # (n_steps, 2)
        
        # Create LIF-based trainer for chaotic dynamics
        trainer = make_lif_force_trainer_for_chaos(
            n_neurons=n_neurons,
            n_outputs=2,
            device=self.device,
        )
        
        start_time = time.time()
        
        n_train = n_steps // 2
        trainer.reset_state()
        
        outputs = []
        
        for t in range(n_steps):
            x = torch.zeros(1, device=self.device)
            if t < n_train:
                y_pred, _ = trainer.train_step(x, pattern[t])
            else:
                y_pred = trainer.step(x)
            outputs.append(y_pred.cpu().numpy())
        
        training_time = (time.time() - start_time) * 1000
        
        # Evaluate
        output_arr = np.array(outputs[n_train:])
        target_test = pattern[n_train:].cpu().numpy()
        
        # Correlation for each dimension
        corr_x = np.corrcoef(target_test[:, 0], output_arr[:, 0])[0, 1]
        corr_y = np.corrcoef(target_test[:, 1], output_arr[:, 1])[0, 1]
        correlation = (corr_x + corr_y) / 2
        
        mse = np.mean((target_test - output_arr) ** 2)
        
        result = BenchmarkResult(
            test_name="lorenz_attractor",
            target_correlation=correlation,
            mse_final=mse,
            training_time_ms=training_time,
            n_steps=n_steps,
            spectral_radius=trainer.initial_spectral_radius,
            extra_metrics={
                "corr_x": corr_x,
                "corr_y": corr_y,
                "synapse_fast_contrib": trainer.synapses.get_timescale_contributions()["fast"],
            },
        )
        
        self.log(f"  Correlation: {correlation:.4f} (x: {corr_x:.4f}, y: {corr_y:.4f})")
        self.log(f"  MSE: {mse:.6f}")
        
        return result
    
    def test_ode_to_joy(
        self,
        n_neurons: int = 1500,
        n_steps: int = 4000,
    ) -> BenchmarkResult:
        """
        Test 4: "Ode to Joy" melody (Figure 3 from paper).
        
        The iconic benchmark - learning a complex 5-component musical pattern.
        """
        self.log("Testing Ode to Joy pattern...")
        
        pattern = PatternGenerator.generate_ode_to_joy(n_steps=n_steps).to(self.device)
        n_outputs = pattern.shape[1]  # Should be 5
        
        # Create LIF-based trainer
        cfg = FORCE2LIFConfig(
            n_neurons=n_neurons,
            n_outputs=n_outputs,
            chaotic_cfg=ChaoticInitConfig(target_radius=1.5),
            multi_tau_cfg=MultiTimescaleSynapseConfig(
                tau_fast=3.0,
                tau_slow=80.0,
                alpha_fast=0.5,
                alpha_slow=0.5,
            ),
            train_readout=True,
            alpha_rls=1.5,
        )
        trainer = FORCE2LIFTrainer(cfg, self.device)
        
        start_time = time.time()
        
        n_train = n_steps // 2
        trainer.reset_state()
        
        outputs = []
        
        for t in range(n_steps):
            x = torch.zeros(1, device=self.device)
            if t < n_train:
                y_pred, _ = trainer.train_step(x, pattern[t])
            else:
                y_pred = trainer.step(x)
            outputs.append(y_pred.cpu().numpy())
        
        training_time = (time.time() - start_time) * 1000
        
        # Evaluate
        output_arr = np.array(outputs[n_train:])
        target_test = pattern[n_train:].cpu().numpy()
        
        # Mean correlation across all 5 components
        correlations = []
        for i in range(n_outputs):
            corr = np.corrcoef(target_test[:, i], output_arr[:, i])[0, 1]
            correlations.append(corr)
        
        mean_correlation = np.mean(correlations)
        mse = np.mean((target_test - output_arr) ** 2)
        
        result = BenchmarkResult(
            test_name="ode_to_joy",
            target_correlation=mean_correlation,
            mse_final=mse,
            training_time_ms=training_time,
            n_steps=n_steps,
            spectral_radius=trainer.initial_spectral_radius,
            extra_metrics={
                f"corr_component_{i}": c for i, c in enumerate(correlations)
            },
        )
        
        self.log(f"  Mean Correlation: {mean_correlation:.4f}")
        self.log(f"  Component correlations: {[f'{c:.3f}' for c in correlations]}")
        self.log(f"  MSE: {mse:.6f}")
        
        return result
    
    def test_spectral_radius_sweep(
        self,
        radii: List[float] = [0.8, 1.0, 1.25, 1.5, 1.8, 2.0],
    ) -> List[BenchmarkResult]:
        """
        Test effect of spectral radius on learning performance.
        
        Paper insight: Radius > 1 (chaotic regime) is crucial for complex tasks.
        """
        self.log(f"Testing spectral radius sweep: {radii}")
        
        pattern = PatternGenerator.generate_oscillator(
            freq=2.0, n_steps=1000
        ).to(self.device)
        
        results = []
        
        for radius in radii:
            self.log(f"  Testing radius={radius}...")
            
            cfg = FORCE2LIFConfig(
                n_neurons=500,
                n_outputs=1,
                chaotic_cfg=ChaoticInitConfig(target_radius=radius),
                train_readout=True,
            )
            trainer = FORCE2LIFTrainer(cfg, self.device)
            
            trainer.reset_state()
            errors = []
            
            for t in range(500):
                x = torch.zeros(1, device=self.device)
                _, error = trainer.train_step(x, pattern[t])
                errors.append(error)
            
            # Test correlation
            outputs = []
            for t in range(500, 1000):
                x = torch.zeros(1, device=self.device)
                y_pred = trainer.step(x)
                outputs.append(y_pred.item())
            
            target_test = pattern[500:].cpu().numpy()
            output_test = np.array(outputs)
            
            correlation = np.corrcoef(target_test, output_test)[0, 1]
            
            # Test if actually chaotic
            is_chaotic = test_chaos_property(
                trainer.W_rec if hasattr(trainer, 'W_rec') else trainer.sparse_conn.W_fixed
            )["is_chaotic"]
            
            result = BenchmarkResult(
                test_name=f"radius_sweep_{radius}",
                target_correlation=correlation,
                mse_final=np.mean(errors[-100:]),
                training_time_ms=0,
                n_steps=1000,
                spectral_radius=radius,
                extra_metrics={"actual_chaos": is_chaotic},
            )
            results.append(result)
            
            self.log(f"    Correlation: {correlation:.4f}, Chaotic: {is_chaotic}")
        
        return results
    
    def run_all_tests(self) -> Dict[str, any]:
        """Run complete benchmark suite."""
        self.log("=" * 60)
        self.log("FORCE2 Benchmark Suite (Nicola & Clopath 2017)")
        self.log("=" * 60)
        
        # Run tests
        self.results = []
        
        try:
            r1 = self.test_simple_oscillator()
            self.results.append(r1)
        except Exception as e:
            self.log(f"Oscillator test failed: {e}")
        
        try:
            r2 = self.test_coupled_oscillators()
            self.results.append(r2)
        except Exception as e:
            self.log(f"Coupled oscillators test failed: {e}")
        
        try:
            r3 = self.test_lorenz_attractor()
            self.results.append(r3)
        except Exception as e:
            self.log(f"Lorenz test failed: {e}")
        
        try:
            r4 = self.test_ode_to_joy()
            self.results.append(r4)
        except Exception as e:
            self.log(f"Ode to Joy test failed: {e}")
        
        try:
            rs = self.test_spectral_radius_sweep()
            self.results.extend(rs)
        except Exception as e:
            self.log(f"Radius sweep test failed: {e}")
        
        # Summary
        self.log("=" * 60)
        self.log("SUMMARY")
        self.log("=" * 60)
        
        summary = {
            "overall_mean_correlation": np.mean([r.target_correlation for r in self.results if "sweep" not in r.test_name]),
            "tests": [r.to_dict() for r in self.results],
        }
        
        for r in self.results:
            if "sweep" not in r.test_name:
                status = "✓" if r.target_correlation > 0.8 else "⚠" if r.target_correlation > 0.5 else "✗"
                self.log(f"{status} {r.test_name}: r={r.target_correlation:.4f}, MSE={r.mse_final:.6f}")
        
        self.log(f"Overall mean correlation: {summary['overall_mean_correlation']:.4f}")
        
        return summary
    
    def save_results(self, path: str):
        """Save results to JSON."""
        results_dict = {
            "benchmark": "FORCE2_Nicola_Clopath_2017",
            "results": [r.to_dict() for r in self.results],
        }
        
        with open(path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        self.log(f"Results saved to {path}")


def main():
    parser = argparse.ArgumentParser(description="FORCE2 Benchmark")
    parser.add_argument("--test", type=str, default="all",
                       choices=["all", "oscillator", "coupled", "lorenz", "ode", "sweep"])
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--save_results", type=str, default=None,
                       help="Path to save JSON results")
    parser.add_argument("--save_plot", type=str, default=None,
                       help="Path to save plot")
    parser.add_argument("--quiet", action="store_true", help="Reduce output")
    
    args = parser.parse_args()
    
    # Create benchmark
    benchmark = FORCEBenchmark(device=args.device, verbose=not args.quiet)
    
    # Run specified test
    if args.test == "all":
        results = benchmark.run_all_tests()
    elif args.test == "oscillator":
        r = benchmark.test_simple_oscillator()
        results = {"tests": [r.to_dict()]}
        benchmark.results = [r]
    elif args.test == "coupled":
        r = benchmark.test_coupled_oscillators()
        results = {"tests": [r.to_dict()]}
        benchmark.results = [r]
    elif args.test == "lorenz":
        r = benchmark.test_lorenz_attractor()
        results = {"tests": [r.to_dict()]}
        benchmark.results = [r]
    elif args.test == "ode":
        r = benchmark.test_ode_to_joy()
        results = {"tests": [r.to_dict()]}
        benchmark.results = [r]
    elif args.test == "sweep":
        rs = benchmark.test_spectral_radius_sweep()
        results = {"tests": [r.to_dict() for r in rs]}
        benchmark.results = rs
    
    # Save if requested
    if args.save_results:
        benchmark.save_results(args.save_results)
    
    # Print summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    for r in benchmark.results:
        print(f"{r.test_name:30s} | r={r.target_correlation:.4f} | MSE={r.mse_final:.6f}")
    
    return results


if __name__ == "__main__":
    main()
