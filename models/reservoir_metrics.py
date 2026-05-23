"""
models/reservoir_metrics.py
============================
Quantify the computational quality of the Arthedain reservoir.

Three core metrics from Liquid State Machine (LSM) theory that IQT
evaluators will ask about:

  Memory Capacity (MC)    — how many past inputs can the reservoir recall?
                            Ideal: MC ≈ N (hidden_size). Current: measure it.

  Kernel Quality (KQ)     — do different inputs produce separable states?
                            Ideal: KQ = 1.0. Measures linear separability.

  Generalization Rank (GR) — stability of the separation under noise.
                              High GR → robust in contested environments.

  Lyapunov Exponent (λ)   — is the reservoir at the edge of chaos?
                              λ ≈ 0 → edge of chaos (optimal for computation)
                              λ > 0 → chaotic, unstable
                              λ < 0 → ordered, low memory

References
----------
- Maass, W., Natschläger, T., & Markram, H. (2002). Real-time computing
  without stable states: A new framework. Neural Computation, 14(11).
- Jaeger, H. (2001). The echo state approach to analysing and training
  recurrent neural networks. GMD Report 148.
- Legenstein & Maass (2007). Edge of chaos and prediction of computational
  performance for neural circuit models. Neural Networks, 20(3).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class ReservoirMetricsConfig:
    n_test_inputs:  int   = 50     # random inputs to test separability
    mc_max_lag:     int   = 50     # max lag for memory capacity (ms)
    noise_std:      float = 0.1    # noise for generalization rank
    lyapunov_steps: int   = 100    # steps for Lyapunov estimation


class ReservoirMetrics:
    """
    Compute LSM quality metrics for an RSNN instance.

    Usage
    -----
        from models.reservoir_metrics import ReservoirMetrics
        from models.rsnn import RSNN

        rsnn = RSNN(input_size=100, hidden_size=128)
        metrics = ReservoirMetrics()
        report = metrics.evaluate(rsnn, input_size=100)
        print(report)
    """

    def __init__(self, config: Optional[ReservoirMetricsConfig] = None) -> None:
        self.cfg = config or ReservoirMetricsConfig()

    def evaluate(self, rsnn, input_size: int = 100) -> Dict[str, float]:
        """
        Run all metrics on an RSNN and return a report dict.

        Parameters
        ----------
        rsnn       : RSNN instance
        input_size : dimension of input vector

        Returns
        -------
        dict with: memory_capacity, kernel_quality, generalization_rank,
                   lyapunov_exponent, spectral_radius
        """
        report = {}
        report["memory_capacity"]    = self.memory_capacity(rsnn, input_size)
        report["kernel_quality"]     = self.kernel_quality(rsnn, input_size)
        report["generalization_rank"]= self.generalization_rank(rsnn, input_size)
        report["lyapunov_exponent"]  = self.lyapunov_exponent(rsnn, input_size)
        report["spectral_radius"]    = self._spectral_radius(rsnn)
        return report

    # ------------------------------------------------------------------
    # Memory Capacity (Jaeger 2001)
    # ------------------------------------------------------------------

    def memory_capacity(self, rsnn, input_size: int) -> float:
        """
        Estimate memory capacity: how many past inputs can be recalled?

        Drives the RSNN with random i.i.d. inputs, trains a linear readout
        to reconstruct each past input x[t-k] from the current state,
        and sums the R² values across lags.

        Returns MC ∈ [0, hidden_size].
        """
        device = rsnn.device
        cfg    = self.cfg
        T      = cfg.mc_max_lag * 3    # run long enough to fill lags
        inputs = torch.randn(T, input_size, device=device) * 0.5

        # Collect states
        rsnn.reset()
        states = []
        for t in range(T):
            spikes = rsnn.forward(inputs[t])
            states.append(spikes.detach())
        S = torch.stack(states)   # (T, N)

        # For each lag k, fit linear readout x[t-k] ~ W_out @ state[t]
        mc = 0.0
        for k in range(1, cfg.mc_max_lag + 1):
            if k >= T:
                break
            X = S[k:].cpu()           # (T-k, N)
            y = inputs[:T-k].cpu()    # (T-k, input_size)
            # Least squares: W = (X^T X)^{-1} X^T y
            try:
                W, _, _, _ = torch.linalg.lstsq(X, y)
                y_pred = X @ W
                ss_res = (y - y_pred).pow(2).sum()
                ss_tot = (y - y.mean(0)).pow(2).sum()
                r2 = max(0.0, float(1 - ss_res / (ss_tot + 1e-12)))
                mc += r2
            except Exception:
                pass

        rsnn.reset()
        return mc

    # ------------------------------------------------------------------
    # Kernel Quality (Maass et al. 2002)
    # ------------------------------------------------------------------

    def kernel_quality(self, rsnn, input_size: int) -> float:
        """
        Kernel quality: can different inputs be separated by their states?

        Generates n_test_inputs random inputs, drives the reservoir,
        collects final states.  KQ = rank(state_matrix) / n_neurons.

        KQ ≈ 1.0 → all states are linearly independent (ideal).
        KQ << 1.0 → states collapse to a low-dimensional subspace.
        """
        cfg    = self.cfg
        device = rsnn.device
        warmup = 50

        state_list = []
        for _ in range(cfg.n_test_inputs):
            rsnn.reset()
            x_stream = torch.randn(warmup + 1, input_size, device=device) * 0.5
            for t in range(warmup):
                rsnn.forward(x_stream[t])
            spikes = rsnn.forward(x_stream[-1])
            state_list.append(spikes.detach().cpu())

        S = torch.stack(state_list)   # (n_test, N)
        # Normalise
        S = S - S.mean(0)
        if S.std() < 1e-9:
            return 0.0
        S = S / S.std()

        # Effective rank via singular values
        sv = torch.linalg.svdvals(S)
        sv = sv[sv > 1e-6]
        effective_rank = float(sv.sum().pow(2) / sv.pow(2).sum())

        rsnn.reset()
        return min(1.0, effective_rank / S.shape[1])

    # ------------------------------------------------------------------
    # Generalization Rank (Legenstein & Maass 2007)
    # ------------------------------------------------------------------

    def generalization_rank(self, rsnn, input_size: int) -> float:
        """
        Generalization rank: stability of states under input noise.

        Runs the same input twice — once clean, once with Gaussian noise.
        GR = 1 - mean_cosine_distance(clean_states, noisy_states).

        GR → 1.0 means noisy inputs produce similar states (robust).
        GR → 0.0 means the reservoir is noise-sensitive (brittle).
        """
        cfg    = self.cfg
        device = rsnn.device
        T      = 30

        clean_states = []
        noisy_states = []
        inputs = torch.randn(T, input_size, device=device) * 0.5

        for use_noise in [False, True]:
            rsnn.reset()
            for t in range(T):
                x = inputs[t]
                if use_noise:
                    x = x + torch.randn_like(x) * cfg.noise_std
                s = rsnn.forward(x)
                if use_noise:
                    noisy_states.append(s.detach())
                else:
                    clean_states.append(s.detach())

        clean = torch.stack(clean_states)   # (T, N)
        noisy = torch.stack(noisy_states)

        cosine_dists = 1 - torch.nn.functional.cosine_similarity(clean, noisy)
        gr = float(1 - cosine_dists.mean().clamp(0, 1))

        rsnn.reset()
        return gr

    # ------------------------------------------------------------------
    # Lyapunov Exponent
    # ------------------------------------------------------------------

    def lyapunov_exponent(self, rsnn, input_size: int) -> float:
        """
        Estimate the maximal Lyapunov exponent of the reservoir.

        Runs two nearby trajectories (perturbed by ε in initial state)
        and measures divergence rate.

        λ ≈ 0 → edge of chaos (optimal for computation)
        λ > 0 → chaotic
        λ < 0 → ordered/stable
        """
        cfg    = self.cfg
        device = rsnn.device
        T      = cfg.lyapunov_steps
        ε      = 1e-4

        inputs = torch.randn(T, input_size, device=device) * 0.3

        # Trajectory 1
        rsnn.reset()
        traj1 = []
        for t in range(T):
            s = rsnn.forward(inputs[t])
            traj1.append(s.detach().cpu())

        # Trajectory 2 (perturbed initial state)
        rsnn.reset()
        rsnn.lif.v += torch.randn_like(rsnn.lif.v) * ε
        traj2 = []
        for t in range(T):
            s = rsnn.forward(inputs[t])
            traj2.append(s.detach().cpu())

        divergences = [
            (t1 - t2).norm().item() / (ε + 1e-16)
            for t1, t2 in zip(traj1[1:], traj2[1:])
        ]
        divergences = [d for d in divergences if d > 0]
        if not divergences:
            return 0.0

        log_divs = [math.log(d) for d in divergences]
        λ = sum(log_divs) / len(log_divs) / T

        rsnn.reset()
        return λ

    # ------------------------------------------------------------------
    # Spectral radius
    # ------------------------------------------------------------------

    def _spectral_radius(self, rsnn) -> float:
        try:
            ev = torch.linalg.eigvals(rsnn.W_rec)
            return float(ev.abs().max().item())
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # Report formatter
    # ------------------------------------------------------------------

    @staticmethod
    def format_report(report: Dict[str, float]) -> str:
        mc  = report.get("memory_capacity",    0)
        kq  = report.get("kernel_quality",     0)
        gr  = report.get("generalization_rank",0)
        λ   = report.get("lyapunov_exponent",  0)
        sr  = report.get("spectral_radius",    0)

        edge = "✓ edge-of-chaos" if abs(λ) < 0.1 else ("⚠ chaotic" if λ > 0.1 else "⚠ ordered")

        return (
            f"Reservoir Quality Report\n"
            f"  Memory Capacity:     {mc:6.2f}   (target: ~{int(sr*10)})\n"
            f"  Kernel Quality:      {kq:6.3f}   (target: >0.7)\n"
            f"  Generalization Rank: {gr:6.3f}   (target: >0.85)\n"
            f"  Lyapunov Exponent:   {λ:+7.4f}  {edge}\n"
            f"  Spectral Radius:     {sr:6.3f}   (target: 0.97–1.00)\n"
        )

    @staticmethod
    def overall_score(report: Dict[str, float]) -> float:
        """
        Compute a single 0–1 quality score from a metrics report.

        Weights:
          memory_capacity   (normalised by target ≈ sr × 10)   → 0.3
          kernel_quality    (target > 0.7)                      → 0.3
          generalization_rank (target > 0.85)                   → 0.2
          lyapunov proximity (|λ| < 0.1 → edge-of-chaos)       → 0.2

        Returns:
            Scalar ∈ [0, 1]; > 0.7 is good; > 0.85 is excellent.
        """
        sr  = report.get("spectral_radius",     0.97)
        mc  = report.get("memory_capacity",     0.0)
        kq  = report.get("kernel_quality",      0.0)
        gr  = report.get("generalization_rank", 0.0)
        lam = report.get("lyapunov_exponent",   0.0)

        mc_target  = max(sr * 10, 1.0)
        mc_score   = min(1.0, mc / mc_target)
        kq_score   = min(1.0, kq / 0.7)
        gr_score   = min(1.0, gr / 0.85)
        lam_score  = max(0.0, 1.0 - abs(lam) / 0.5)   # best at λ≈0

        return float(
            0.3 * mc_score
            + 0.3 * kq_score
            + 0.2 * gr_score
            + 0.2 * lam_score
        )
