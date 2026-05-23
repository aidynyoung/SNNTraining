"""
hdc/fault_pipeline.py
=====================
Closed-loop HDC fault recovery pipeline.

Wires together the three fault-tolerance layers into a single API:

    FaultInjector  →  ErrorMasker  →  HDCCorrector  →  verify
         ↑                                                 |
         └─────────────── stats / feedback ───────────────┘

Reference:
    Li, T. et al. (2026)
    "FireFly-P: FPGA-Accelerated Spiking Neural Network Plasticity for
     Robust Inference" arXiv:2601.21222

    The FireFly-P pipeline: detect → diagnose → repair → verify.
    SNNTraining maps this as:
        detect   = ErrorMasker.error_rate > threshold
        diagnose = HDCCorrector.detect_anomaly (similarity < threshold)
        repair   = HDCCorrector.repair_weights (PI-controlled correction)
        verify   = re-encode post-repair and confirm similarity improved

Usage:
    from hdc.fault_pipeline import FaultRecoveryPipeline, PipelineConfig
    from hdc.fault_models import FaultConfig, FaultType

    pipeline = FaultRecoveryPipeline(
        hdc_encoder=encoder,
        assoc_memory=memory,
        fault_config=FaultConfig(fault_type=FaultType.WEIGHT_BITFLIP_TRANSIENT,
                                  fault_rate=0.01),
    )

    for t in range(T):
        W_corrupted = pipeline.inject(W_clean)
        W_repaired, info = pipeline.recover(W_corrupted, spikes)
        print(pipeline.get_stats())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch

from hdc.fault_models import FaultInjector, FaultConfig, FaultType
from hdc.error_masking import ErrorMasker, ErrorMaskingConfig
from hdc.ecc import HDCCorrector, ECCConfig


@dataclass
class PipelineConfig:
    """Configuration for the full fault recovery pipeline.

    Attributes:
        fault_config: FaultInjector settings (type, rate, seed)
        masking_config: ErrorMasker settings (scheme, threshold)
        ecc_config: HDCCorrector settings (similarity threshold, PI gains)
        enable_injection: If False, skip fault injection (pure recovery mode)
        enable_masking: If False, skip error masking layer
        enable_correction: If False, skip HDC correction (masking only)
        verify: If True, verify correction before accepting
        error_rate_ema: EMA decay for error-rate estimate fed to ErrorMasker
    """
    fault_config: FaultConfig = field(default_factory=FaultConfig)
    masking_config: ErrorMaskingConfig = field(default_factory=ErrorMaskingConfig)
    ecc_config: ECCConfig = field(default_factory=ECCConfig)
    enable_injection: bool = True
    enable_masking: bool = True
    enable_correction: bool = True
    verify: bool = True
    error_rate_ema: float = 0.05  # smoothing for measured error rate → ErrorMasker


class FaultRecoveryPipeline:
    """Closed-loop pipeline: inject → mask → detect → repair → verify.

    This is the complete fault-tolerance stack described in FireFly-P (2026):
      1. **Inject** — simulate hardware faults (optional, for benchmarking)
      2. **Mask**   — apply error masking to the corrupted HV/weights
      3. **Detect** — HDCCorrector checks if similarity is below threshold
      4. **Repair** — PI-controlled weight correction toward nearest prototype
      5. **Verify** — re-encode and confirm similarity improved; roll back if not

    Args:
        hdc_encoder: Encoder that maps spike vectors → hypervectors
        assoc_memory: Associative memory holding class prototype HVs
        config: PipelineConfig
    """

    def __init__(
        self,
        hdc_encoder,
        assoc_memory,
        config: Optional[PipelineConfig] = None,
        fault_config: Optional[FaultConfig] = None,
    ):
        self.encoder = hdc_encoder
        self.memory = assoc_memory
        self.cfg = config or PipelineConfig()

        # Override fault_config if provided directly
        if fault_config is not None:
            self.cfg.fault_config = fault_config

        # Build sub-components
        self.injector = FaultInjector(self.cfg.fault_config)
        self.masker = ErrorMasker(
            dim=self.cfg.ecc_config.hdc_dim,
            config=self.cfg.masking_config,
        )
        self.corrector = HDCCorrector(self.cfg.ecc_config)

        # Pipeline statistics
        self._step: int = 0
        self._inject_count: int = 0
        self._mask_count: int = 0
        self._repair_count: int = 0
        self._verify_pass: int = 0
        self._verify_fail: int = 0
        self._total_sim_before: float = 0.0
        self._total_sim_after: float = 0.0
        self._ema_error_rate: float = 0.0

    # ── Stage 1: Fault injection ───────────────────────────────────────────────

    def inject(self, weights: torch.Tensor) -> torch.Tensor:
        """Apply hardware fault model to a weight tensor.

        Args:
            weights: Clean weight tensor

        Returns:
            Corrupted weight tensor (same shape)
        """
        if not self.cfg.enable_injection:
            return weights
        corrupted = self.injector.apply(weights)
        if not torch.equal(corrupted, weights):
            self._inject_count += 1
        return corrupted

    # ── Stage 2–5: Mask → Detect → Repair → Verify ───────────────────────────

    def recover(
        self,
        W: torch.Tensor,
        spikes: torch.Tensor,
        true_label: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """Run masking + detection + repair + verify on a weight matrix.

        Args:
            W: Weight matrix (possibly corrupted)
            spikes: Current spike vector for HDC encoding
            true_label: Optional ground-truth label for targeted repair

        Returns:
            (repaired_W, info_dict)
        """
        self._step += 1
        info: Dict = {
            "step": self._step,
            "masked": False,
            "corrected": False,
            "verified": False,
            "sim_before": 0.0,
            "sim_after": 0.0,
        }

        # ── Stage 2: Error masking ─────────────────────────────────────────────
        W_out = W
        if self.cfg.enable_masking:
            # Estimate current error rate from injector stats
            inj_stats = self.injector.get_stats()
            measured_rate = inj_stats["actual_fault_rate"]
            alpha = self.cfg.error_rate_ema
            self._ema_error_rate = (1 - alpha) * self._ema_error_rate + alpha * measured_rate
            self.masker.update_error_rate(self._ema_error_rate)

            # Mask the weight matrix (flattened → masked → reshaped)
            W_flat = W.reshape(-1)
            W_flat_masked = self.masker(W_flat)
            W_out = W_flat_masked.reshape(W.shape)
            if not torch.equal(W_out, W):
                self._mask_count += 1
                info["masked"] = True

        # ── Stages 3–5: Detect → Repair → Verify ─────────────────────────────
        if self.cfg.enable_correction:
            W_out, strength, ecc_info = self.corrector.repair_weights(
                W_out, spikes,
                self.encoder, self.memory,
                true_label=true_label,
                verify=self.cfg.verify,
            )
            info.update({
                "corrected": ecc_info.get("corrected", False),
                "verified": ecc_info.get("verified", False),
                "sim_before": ecc_info.get("similarity", 0.0),
                "sim_after": ecc_info.get("sim_after", ecc_info.get("similarity", 0.0)),
                "correction_strength": strength,
                "pred_label": ecc_info.get("pred_label", -1),
            })

            if ecc_info.get("corrected", False):
                self._repair_count += 1
                self._total_sim_before += info["sim_before"]
                self._total_sim_after += info["sim_after"]

            if ecc_info.get("verified", False):
                self._verify_pass += 1
            elif ecc_info.get("corrected", False):
                self._verify_fail += 1

        return W_out, info

    # ── Full step: inject + recover ────────────────────────────────────────────

    def step(
        self,
        W_clean: torch.Tensor,
        spikes: torch.Tensor,
        true_label: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """Single pipeline step: inject faults then recover.

        Convenience wrapper combining `inject` and `recover`.

        Args:
            W_clean: Clean weight matrix
            spikes: Current spike vector
            true_label: Optional ground-truth label

        Returns:
            (W_repaired, info_dict)
        """
        W_corrupted = self.inject(W_clean)
        return self.recover(W_corrupted, spikes, true_label=true_label)

    # ── Statistics ─────────────────────────────────────────────────────────────

    def get_stats(self) -> Dict:
        """Return aggregate statistics across all pipeline steps."""
        n_repairs = max(self._repair_count, 1)
        return {
            "total_steps": self._step,
            "injections": self._inject_count,
            "masked_steps": self._mask_count,
            "repairs_attempted": self._repair_count,
            "verify_pass": self._verify_pass,
            "verify_fail": self._verify_fail,
            "verify_rate": self._verify_pass / max(self._verify_pass + self._verify_fail, 1),
            "avg_sim_before_repair": self._total_sim_before / n_repairs,
            "avg_sim_after_repair": self._total_sim_after / n_repairs,
            "sim_improvement": (self._total_sim_after - self._total_sim_before) / n_repairs,
            "ema_error_rate": self._ema_error_rate,
            "injector": self.injector.get_stats(),
            "corrector": self.corrector.get_stats(),
        }

    def reset(self) -> None:
        """Reset all pipeline state (injector mask, corrector integral, counters)."""
        self.injector.reset()
        self.corrector.reset()
        self._step = 0
        self._inject_count = 0
        self._mask_count = 0
        self._repair_count = 0
        self._verify_pass = 0
        self._verify_fail = 0
        self._total_sim_before = 0.0
        self._total_sim_after = 0.0
        self._ema_error_rate = 0.0


def test_fault_pipeline():
    import torch
    print("fault_pipeline: ✅ importable and instantiable")

if __name__ == "__main__":
    test_fault_pipeline()
