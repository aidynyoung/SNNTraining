"""
models/anomaly_detector.py
==========================
Online anomaly detector for the Arthedain SNN.

Threat detection in the field requires the network to flag inputs that
deviate from the learned distribution — without cloud connectivity or
retraining.  This module provides three complementary anomaly signals:

  1. Prediction error magnitude  — high when the SNN is surprised
  2. Reservoir state novelty     — Mahalanobis distance from running mean
  3. Spike rate deviation        — firing rates outside learned bounds

Combined into a single anomaly score with statistical confidence bounds
(Chebyshev inequality for distribution-free guarantees).

Usage
-----
    from models.anomaly_detector import AnomalyDetector, AnomalyConfig
    detector = AnomalyDetector(AnomalyConfig(hidden_size=128))

    # Warmup phase (learn normal distribution)
    for x in normal_stream:
        spikes, pred = snn_step(x)
        detector.update(spikes, pred, target)

    # Deployment phase
    for x in test_stream:
        spikes, pred = snn_step(x)
        score, alert, detail = detector.score(spikes, pred, target)
        if alert:
            act_on_threat(score, detail)

References
----------
- Pimentel et al. (2014) "A review of novelty detection" Signal Processing.
- Mahalanobis, P.C. (1936) "On the generalised distance in statistics."
- Chebyshev inequality for distribution-free confidence bounds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch


@dataclass
class AnomalyConfig:
    hidden_size:       int   = 128
    warmup_steps:      int   = 500      # steps before anomaly detection activates
    pred_error_weight: float = 0.4      # weight for prediction error signal
    novelty_weight:    float = 0.4      # weight for reservoir novelty signal
    rate_weight:       float = 0.2      # weight for spike rate deviation
    alert_threshold:   float = 3.0      # z-score threshold for alert
    ema_alpha:         float = 0.01     # EMA decay for running statistics
    confidence_k:      float = 2.0      # Chebyshev k (≥k σ has prob ≤ 1/k²)


@dataclass
class AnomalyReport:
    score:          float
    alert:          bool
    z_score:        float
    pred_err_contrib:   float
    novelty_contrib:    float
    rate_contrib:       float
    confidence_bound:   float    # Chebyshev upper bound on false-alarm rate


class AnomalyDetector:
    """
    Online anomaly detector using prediction error + reservoir novelty.

    Maintains running statistics (mean, variance) of the anomaly score
    during warmup, then flags deviations in deployment.
    """

    def __init__(self, config: Optional[AnomalyConfig] = None) -> None:
        self.cfg = config or AnomalyConfig()
        n = self.cfg.hidden_size
        α = self.cfg.ema_alpha

        # Running statistics (Welford-style EMA)
        self._step = 0
        self._score_mean  = 0.0
        self._score_var   = 1.0
        self._score_m2    = 0.0     # for Welford variance

        # Reservoir state running mean/covariance (diagonal)
        self._state_mean  = torch.zeros(n)
        self._state_var   = torch.ones(n)

        # Spike rate running mean
        self._rate_mean   = torch.full((n,), 0.1)
        self._rate_var    = torch.ones(n) * 0.01

        # Prediction error running stats
        self._err_mean    = 0.0
        self._err_var     = 1.0

    # ------------------------------------------------------------------
    # Signal computation
    # ------------------------------------------------------------------

    def _pred_error_signal(self, pred: torch.Tensor, target: Optional[torch.Tensor]) -> float:
        if target is None:
            return float(pred.norm().item())
        return float((pred - target.to(pred.device)).pow(2).mean().item())

    def _novelty_signal(self, spikes: torch.Tensor) -> float:
        """Mahalanobis distance of current state from learned mean."""
        diff = spikes.float() - self._state_mean.to(spikes.device)
        std  = self._state_var.to(spikes.device).clamp(min=1e-8).sqrt()
        return float((diff / std).pow(2).mean().sqrt().item())

    def _rate_signal(self, spikes: torch.Tensor) -> float:
        """Deviation of current firing rate from learned mean."""
        rate = spikes.float().mean().item()
        mean = float(self._rate_mean.mean())
        std  = float(self._rate_var.mean().sqrt().clamp(min=1e-8))
        return abs(rate - mean) / std

    # ------------------------------------------------------------------
    # Update (call during warmup and optionally deployment)
    # ------------------------------------------------------------------

    def update(
        self,
        spikes:  torch.Tensor,                  # (hidden_size,)
        pred:    torch.Tensor,                  # (output_size,)
        target:  Optional[torch.Tensor] = None, # (output_size,)
    ) -> None:
        """Update running statistics with new observation."""
        α = self.cfg.ema_alpha
        s = spikes.float()

        # Reservoir state mean and variance
        self._state_mean = (1 - α) * self._state_mean + α * s.cpu()
        diff_sq = (s.cpu() - self._state_mean).pow(2)
        self._state_var  = (1 - α) * self._state_var  + α * diff_sq

        # Spike rate
        rate = s.mean().item()
        self._rate_mean = (1 - α) * self._rate_mean + α * rate
        self._rate_var  = (1 - α) * self._rate_var  + α * (rate - float(self._rate_mean.mean()))**2

        # Prediction error
        err = self._pred_error_signal(pred, target)
        self._err_mean = (1 - α) * self._err_mean + α * err
        self._err_var  = max(1e-8, (1 - α) * self._err_var + α * (err - self._err_mean)**2)

        # Composite score statistics (Welford)
        raw = self._raw_score(spikes, pred, target)
        self._step += 1
        n = self._step
        delta = raw - self._score_mean
        self._score_mean += delta / n
        delta2 = raw - self._score_mean
        self._score_m2   += delta * delta2
        self._score_var  = max(1e-8, self._score_m2 / max(n - 1, 1))

    # ------------------------------------------------------------------
    # Score computation
    # ------------------------------------------------------------------

    def _raw_score(
        self,
        spikes: torch.Tensor,
        pred:   torch.Tensor,
        target: Optional[torch.Tensor],
    ) -> float:
        cfg = self.cfg
        s_pred    = self._pred_error_signal(pred, target)
        s_novelty = self._novelty_signal(spikes)
        s_rate    = self._rate_signal(spikes)
        return (cfg.pred_error_weight * s_pred
                + cfg.novelty_weight   * s_novelty
                + cfg.rate_weight      * s_rate)

    def score(
        self,
        spikes:  torch.Tensor,
        pred:    torch.Tensor,
        target:  Optional[torch.Tensor] = None,
    ) -> AnomalyReport:
        """
        Compute anomaly score and alert status.

        Returns
        -------
        AnomalyReport with score, alert flag, z-score, and contributions
        """
        cfg = self.cfg
        raw = self._raw_score(spikes, pred, target)

        std     = math.sqrt(self._score_var)
        z_score = (raw - self._score_mean) / max(std, 1e-8)

        # Alert if z > threshold AND past warmup
        alert = (z_score > cfg.alert_threshold) and (self._step >= cfg.warmup_steps)

        # Chebyshev upper bound on false-alarm rate: P(|X - μ| ≥ k·σ) ≤ 1/k²
        k = max(z_score, 1.0)
        confidence_bound = 1.0 / (k * k)

        return AnomalyReport(
            score=raw,
            alert=alert,
            z_score=z_score,
            pred_err_contrib=cfg.pred_error_weight * self._pred_error_signal(pred, target),
            novelty_contrib =cfg.novelty_weight    * self._novelty_signal(spikes),
            rate_contrib    =cfg.rate_weight        * self._rate_signal(spikes),
            confidence_bound=confidence_bound,
        )

    def is_warmed_up(self) -> bool:
        return self._step >= self.cfg.warmup_steps

    def reset_stats(self) -> None:
        """Reset running statistics (new deployment environment)."""
        self._step       = 0
        self._score_mean = 0.0
        self._score_var  = 1.0
        self._score_m2   = 0.0
        n = self.cfg.hidden_size
        self._state_mean = torch.zeros(n)
        self._state_var  = torch.ones(n)


# ---------------------------------------------------------------------------
# Icarus pipeline: sequence-level anomaly detector
# ---------------------------------------------------------------------------

class OnlineAnomalyDetector:
    """
    Stateful online anomaly detector for streaming soil telemetry.

    Processes one spike sequence per grid cell; scores by Mahalanobis distance
    of the time-averaged hidden state from a Welford rolling mean.

    Reservoir weights are frozen at init — no learning, O(1) memory,
    deterministic latency. Compatible with FPGA deployment path.

    Usage:
        from models.anomaly_detector import OnlineAnomalyDetector
        from data.loaders import encode_soil_reading

        detector = OnlineAnomalyDetector()
        for cell in drone_survey:
            spikes = encode_soil_reading(cell)
            score  = detector.process(spikes)   # 0 = normal, >2 = anomalous
    """

    INPUT_SIZE: int = 40   # 4 analytes × 10 population-coded neurons
    T: int = 20            # spike train timesteps per cell reading

    def __init__(self, hidden_size: int = 256, device: str = "cpu") -> None:
        import os, sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from models.rsnn import RSNN
        from models.alif import ALIFLayer, ALIFConfig

        self.device = device
        self.hidden_size = hidden_size

        self.rsnn = RSNN(
            input_size=self.INPUT_SIZE,
            hidden_size=hidden_size,
            sparse_init=True,
            sparse_p=0.12,
            device=device,
        )
        alif_cfg = ALIFConfig(size=hidden_size, tau=20.0, rho=0.96, beta_a=0.07)
        self.alif = ALIFLayer(alif_cfg)

        # Welford running statistics
        self._n:    int          = 0
        self._mean: torch.Tensor = torch.zeros(hidden_size)
        self._M2:   torch.Tensor = torch.zeros(hidden_size)

    def process(self, spike_seq: torch.Tensor) -> float:
        """
        Process one cell's spike sequence; return anomaly score ≥ 0.

        Score interpretation:
          < 1.5   normal
          1.5–2.5 elevated (review)
          > 2.5   anomalous (contamination / depletion hotspot)

        Returns 0.0 during the first 5 readings (burn-in).
        """
        self.rsnn.reset()
        self.alif.reset()

        sum_h = torch.zeros(self.hidden_size)
        for t in range(spike_seq.shape[0]):
            x_t = spike_seq[t].to(self.device)
            ic = (self.rsnn.input_gain * (self.rsnn.W_in @ x_t)
                  + self.rsnn.W_rec @ self.rsnn.prev_spikes)
            sp, _ = self.alif.step(ic, return_pseudo_deriv=True)
            self.rsnn.prev_spikes = sp.clone()
            sum_h += sp.cpu()

        h = sum_h / spike_seq.shape[0]

        if self._n < 5:
            score = 0.0
        else:
            var   = (self._M2 / max(self._n - 1, 1)).clamp(min=1e-8)
            score = float(((h - self._mean).pow(2) / var).mean().sqrt())

        # Welford update
        self._n   += 1
        delta      = h - self._mean
        self._mean = self._mean + delta / self._n
        self._M2   = self._M2   + delta * (h - self._mean)

        return round(score, 4)

    def score_history(self) -> List[float]:
        """Return all scores logged so far (requires _score_history to be populated)."""
        return list(getattr(self, "_score_history", []))

    def process_tracked(self, spike_seq: torch.Tensor) -> float:
        """
        Like process() but also appends score to internal history for trend analysis.
        """
        score = self.process(spike_seq)
        if not hasattr(self, "_score_history"):
            self._score_history: List[float] = []
        self._score_history.append(score)
        return score

    def adaptive_threshold(
        self,
        percentile: float = 97.0,
        window:     int   = 100,
    ) -> float:
        """
        Compute a data-driven anomaly threshold from recent score history.

        Instead of hard-coded thresholds (< 1.5 normal, > 2.5 anomalous),
        use the empirical percentile of the last `window` scores.

        Args:
            percentile: e.g. 97.0 = flag top 3% as anomalies
            window:     How many recent scores to use

        Returns:
            Adaptive threshold value.
        """
        history = getattr(self, "_score_history", [])
        if not history:
            return 2.0   # safe default
        import math
        recent  = history[-window:]
        sorted_s = sorted(recent)
        idx      = max(0, min(len(sorted_s) - 1, int(len(sorted_s) * percentile / 100.0)))
        return float(sorted_s[idx])

    def is_anomalous(
        self,
        spike_seq: torch.Tensor,
        percentile: float = 97.0,
    ) -> Tuple[bool, float]:
        """
        Convenience: process + adaptive threshold in one call.

        Returns (is_anomaly, score).
        """
        score  = self.process_tracked(spike_seq)
        thresh = self.adaptive_threshold(percentile)
        return score > thresh, score

    def reset(self) -> None:
        """Reset statistics — call when starting a new field survey."""
        self._n = 0
        self._mean.zero_()
        self._M2.zero_()
        if hasattr(self, "_score_history"):
            self._score_history.clear()

    def detector_health(self, percentile: float = 97.0, window: int = 100) -> dict:
        """
        One-call health report: score stats, threshold, anomaly rate.

        anomaly_rate > 0.1 → model is flagging >10% of inputs (too sensitive or genuine drift).
        n_samples < 50     → warm-up phase: threshold not yet reliable.
        """
        history = getattr(self, "_score_history", [])
        n = len(history)
        thresh = self.adaptive_threshold(percentile, window)
        if n == 0:
            return {"n_samples": 0, "status": "warming_up", "threshold": thresh}
        recent = history[-window:]
        mean_score = sum(recent) / len(recent)
        max_score  = max(recent)
        anomaly_rate = sum(1 for s in recent if s > thresh) / max(len(recent), 1)
        return {
            "n_samples":     n,
            "mean_score":    round(mean_score, 4),
            "max_score":     round(max_score, 4),
            "threshold":     round(thresh, 4),
            "anomaly_rate":  round(anomaly_rate, 4),
            "warmed_up":     n >= 50,
            "status":        "anomalous" if anomaly_rate > 0.1 else "nominal",
        }
