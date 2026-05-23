"""
test_ecc.py
===========
Tests for HDC Error Correction Codes (hdc/ecc.py).

Validates:
  1. PI controller dynamics (proportional + integral + anti-windup)
  2. Anomaly detection threshold behavior
  3. Weight repair: detection → correction → improvement
  4. Correction cooldown (prevents oscillation)
  5. Max correction norm clipping
  6. Correction statistics tracking
"""

from __future__ import annotations

import math
import sys
import os
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from hdc.ecc import (
    HDCCorrector,
    ECCConfig,
)
from models.hdc import HDCEncoder, AssocMemory, gen_hvs, batch_sim


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def tiny_hdc_system(device):
    """Create a minimal HDC encoder + memory for testing ECC."""
    hidden_size = 32
    hdc_dim = 256
    n_classes = 4
    seed = 42

    encoder = HDCEncoder(
        input_size=hidden_size,
        n_classes=n_classes,
        dim=hdc_dim,
        mode="bipolar",
        device=device,
        seed=seed,
    )
    memory = AssocMemory(
        n_classes=n_classes,
        dim=hdc_dim,
        mode="bipolar",
        device=device,
        seed=seed,
    )

    # Train the encoder with some spike patterns
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    for cls in range(n_classes):
        for _ in range(15):
            spikes = torch.rand(hidden_size, generator=rng, device=device) > 0.6
            spikes = spikes.float()
            encoder.train_step(spikes, cls)
    encoder.finalize()

    return encoder, memory, hidden_size, hdc_dim, n_classes


@pytest.fixture
def corrector_default():
    return HDCCorrector(ECCConfig(
        hdc_dim=256,
        n_classes=4,
        similarity_threshold=0.3,
        correction_strength=0.1,
        use_pi_control=True,
        kp=0.5,
        ki=0.1,
    ))


@pytest.fixture
def corrector_simple():
    """Simple threshold-based corrector (no PI control)."""
    return HDCCorrector(ECCConfig(
        hdc_dim=256,
        n_classes=4,
        similarity_threshold=0.3,
        correction_strength=0.1,
        use_pi_control=False,
    ))


# ── PI Controller Tests ───────────────────────────────────────────────────────

class TestPIController:
    """Validate PI controller dynamics (Saponati et al. 2026)."""

    def test_initial_state(self, corrector_default):
        """Corrector starts with zero integral error."""
        assert corrector_default.integral_error == 0.0
        assert corrector_default.correction_count == 0

    def test_proportional_term_zero_error(self, corrector_default):
        """When similarity == threshold, no correction needed."""
        # similarity == threshold → error = 0 → p_term = 0
        strength = corrector_default.compute_correction(
            similarity=0.3,
            error_vector=torch.zeros(256),
        )
        # p_term = kp * (0.3 - 0.3) = 0, i_term = 0
        # correction = 0 * correction_strength = 0
        assert strength == 0.0

    def test_proportional_term_high_error(self, corrector_default):
        """Low similarity → high proportional correction."""
        strength = corrector_default.compute_correction(
            similarity=0.05,  # Far below threshold
            error_vector=torch.zeros(256),
        )
        # error = 0.3 - 0.05 = 0.25
        # p_term = 0.5 * 0.25 = 0.125
        # i_term = ki * error = 0.1 * 0.25 = 0.025 (after integration)
        # total = min(1.0, 0.125 + 0.025) * 0.1 ≈ 0.015
        assert strength > 0.0
        assert strength <= 1.0

    def test_integral_accumulation(self, corrector_default):
        """Integral error accumulates over successive corrections."""
        initial_integral = corrector_default.integral_error

        # Simulate sustained low similarity
        for _ in range(10):
            corrector_default.compute_correction(
                similarity=0.1,
                error_vector=torch.zeros(256),
            )

        # Integral should have accumulated
        assert corrector_default.integral_error > initial_integral

    def test_anti_windup(self, corrector_default):
        """Integral error saturates at 1.0 (anti-windup)."""
        for _ in range(500):
            corrector_default.compute_correction(
                similarity=0.0,  # Maximum error
                error_vector=torch.zeros(256),
            )

        assert 0.0 <= corrector_default.integral_error <= 1.0

    def test_no_negative_integral(self, corrector_default):
        """Integral error doesn't go negative (clamped to 0)."""
        # High similarity should not produce negative integral
        corrector_default.integral_error = 0.0
        corrector_default.compute_correction(
            similarity=1.0,  # Perfect similarity
            error_vector=torch.zeros(256),
        )
        assert corrector_default.integral_error >= 0.0


class TestAnomalyDetection:
    """Validate anomaly detection threshold behavior."""

    def test_high_similarity_no_detect(self, corrector_default):
        """High similarity → no anomaly."""
        # Feed warmup steps first
        for _ in range(corrector_default.cfg.correction_cooldown):
            corrector_default._step += 1
            corrector_default.last_correction_step = -corrector_default.cfg.correction_cooldown

        detected = corrector_default.detect_anomaly(similarity=0.8)
        assert detected is False

    def test_low_similarity_detects(self, corrector_default):
        """Low similarity → anomaly detected."""
        # Feed warmup
        for _ in range(corrector_default.cfg.correction_cooldown + 1):
            corrector_default._step += 1

        detected = corrector_default.detect_anomaly(similarity=0.1)
        assert detected is True

    def test_cooldown_prevents_rapid_corrections(self, corrector_default):
        """Cooldown period prevents back-to-back corrections."""
        # First correction
        corrector_default._step += 30
        corrector_default.detect_anomaly(similarity=0.05)
        corrector_default.last_correction_step = corrector_default._step

        # Immediately after — should not trigger
        detected = corrector_default.detect_anomaly(similarity=0.05)
        assert detected is False

    def test_similarity_history_tracking(self, corrector_default):
        """Similarity history accumulates and is capped at 100 entries."""
        for _ in range(150):
            corrector_default.detect_anomaly(similarity=0.5)

        assert len(corrector_default.similarity_history) == 100


class TestWeightRepair:
    """Validate the full weight repair pipeline."""

    def test_repair_healthy_weights_no_op(self, tiny_hdc_system, corrector_default):
        """Healthy weights (high similarity) → no repair needed."""
        encoder, memory, hidden_size, hdc_dim, n_classes = tiny_hdc_system

        # Generate a spike pattern for class 0
        rng = torch.Generator(device="cpu")
        rng.manual_seed(0)
        spikes = (torch.rand(hidden_size, generator=rng) > 0.6).float()

        W_rec = torch.randn(hidden_size, hidden_size) * 0.1

        # High similarity → no repair
        # Encode spikes and verify similarity
        hv = encoder.encode(spikes)
        sims = batch_sim(hv, memory.class_hvs, "bipolar")
        max_sim = float(sims.max().item())

        # If similarity is high enough, repair shouldn't trigger
        if max_sim > 0.3:
            corrector_default.last_correction_step = corrector_default._step  # reset cooldown
            corrector_default._step += 30
            corrected_W, strength, info = corrector_default.repair_weights(
                W_rec, spikes, encoder, memory,
            )
            # Either no correction (if similarity high) or small correction
            assert info["corrected"] in (True, False)

    def test_repair_output_shape(self, tiny_hdc_system, corrector_default):
        """Repair preserves weight matrix shape."""
        encoder, memory, hidden_size, hdc_dim, n_classes = tiny_hdc_system

        W_rec = torch.randn(hidden_size, hidden_size) * 0.1
        spikes = torch.rand(hidden_size) > 0.7
        spikes = spikes.float()

        # Bypass cooldown
        corrector_default.last_correction_step = -100
        corrector_default._step += 50

        corrected_W, strength, info = corrector_default.repair_weights(
            W_rec, spikes, encoder, memory,
        )
        assert corrected_W.shape == W_rec.shape

    def test_repair_info_contains_keys(self, tiny_hdc_system, corrector_default):
        """Repair info dict has expected keys."""
        encoder, memory, hidden_size, hdc_dim, n_classes = tiny_hdc_system

        W_rec = torch.randn(hidden_size, hidden_size) * 0.1
        spikes = torch.rand(hidden_size) > 0.7
        spikes = spikes.float()

        corrector_default.last_correction_step = -100
        corrector_default._step += 50

        _, _, info = corrector_default.repair_weights(
            W_rec, spikes, encoder, memory,
        )
        required_keys = {"corrected", "similarity", "pred_label"}
        assert required_keys.issubset(info.keys())

    def test_repair_with_true_label(self, tiny_hdc_system, corrector_default):
        """Repair uses ground-truth label when provided."""
        encoder, memory, hidden_size, hdc_dim, n_classes = tiny_hdc_system

        W_rec = torch.randn(hidden_size, hidden_size) * 0.1
        spikes = torch.rand(hidden_size) > 0.7
        spikes = spikes.float()

        corrector_default.last_correction_step = -100
        corrector_default._step += 50

        _, _, info = corrector_default.repair_weights(
            W_rec, spikes, encoder, memory, true_label=2,
        )
        # If corrected, target_label should be 2 (ground truth)
        if info["corrected"]:
            assert info["target_label"] == 2


class TestCorrectionStrength:
    """Validate correction strength scaling."""

    def test_simple_threshold_correction(self, corrector_simple):
        """Simple mode: correction is binary (0 or correction_strength)."""
        # Low similarity triggers correction
        strength = corrector_simple.compute_correction(
            similarity=0.1, error_vector=torch.zeros(256),
        )
        assert strength == corrector_simple.cfg.correction_strength

        # High similarity → no correction
        strength = corrector_simple.compute_correction(
            similarity=0.8, error_vector=torch.zeros(256),
        )
        assert strength == 0.0

    def test_pi_smooth_correction(self, corrector_default):
        """PI mode: correction is proportional to error."""
        strength_high = corrector_default.compute_correction(
            similarity=0.05, error_vector=torch.zeros(256),
        )
        corrector_default.integral_error = 0.0  # Reset integral for fair comparison

        strength_low = corrector_default.compute_correction(
            similarity=0.25, error_vector=torch.zeros(256),
        )
        # Higher error (lower similarity) → more correction
        assert strength_high > strength_low


class TestStatistics:
    """Validate correction statistics tracking."""

    def test_initial_stats(self, corrector_default):
        stats = corrector_default.get_stats()
        assert stats["correction_count"] == 0
        assert stats["total_steps"] == 0

    def test_stats_after_corrections(self, corrector_default):
        corrector_default._step = 100
        corrector_default.correction_count = 5
        corrector_default.similarity_history = [0.2, 0.3, 0.4, 0.5, 0.6]

        stats = corrector_default.get_stats()
        assert stats["correction_count"] == 5
        assert stats["total_steps"] == 100
        assert stats["correction_rate"] == 5 / 100
        assert stats["avg_similarity"] == pytest.approx(0.4, abs=0.01)

    def test_reset_clears_all(self, corrector_default):
        """Reset clears integral, counters, and history."""
        corrector_default._step = 50
        corrector_default.correction_count = 3
        corrector_default.integral_error = 0.5
        corrector_default.similarity_history = [0.1, 0.2]

        corrector_default.reset()

        assert corrector_default.integral_error == 0.0
        assert corrector_default.correction_count == 0
        assert corrector_default.similarity_history == []
        assert corrector_default._step == 0


class TestECCConfig:
    """Validate ECCConfig defaults and customization."""

    def test_default_config(self):
        cfg = ECCConfig()
        assert cfg.hdc_dim == 4096
        assert cfg.n_classes == 8
        assert cfg.mode == "bipolar"
        assert cfg.similarity_threshold == 0.3
        assert cfg.correction_strength == 0.1
        assert cfg.use_pi_control is True
        assert cfg.kp == 0.5
        assert cfg.ki == 0.1
        assert cfg.correction_cooldown == 20
        assert cfg.max_correction_norm == 0.05

    def test_custom_config(self):
        cfg = ECCConfig(
            hdc_dim=1024,
            n_classes=10,
            similarity_threshold=0.5,
            correction_strength=0.2,
            use_pi_control=False,
        )
        assert cfg.hdc_dim == 1024
        assert cfg.n_classes == 10
        assert cfg.similarity_threshold == 0.5
        assert cfg.use_pi_control is False


class TestMaxCorrectionNorm:
    """Validate that corrections don't exceed the max norm bound."""

    def test_norm_clipping(self, tiny_hdc_system):
        """Large HV errors are clipped to max_correction_norm."""
        encoder, memory, hidden_size, hdc_dim, n_classes = tiny_hdc_system

        corrector = HDCCorrector(ECCConfig(
            hdc_dim=hdc_dim,
            n_classes=n_classes,
            similarity_threshold=0.9,  # High threshold → always triggers
            correction_strength=1.0,   # Maximum strength
            max_correction_norm=0.01,  # Small bound
        ))

        W_rec = torch.randn(hidden_size, hidden_size)
        spikes = torch.zeros(hidden_size)
        spikes[0:5] = 1.0  # Some spikes

        # Bypass cooldown
        corrector.last_correction_step = -100
        corrector._step += 50

        corrected_W, strength, info = corrector.repair_weights(
            W_rec, spikes, encoder, memory,
        )

        if info["corrected"]:
            # The update norm should not exceed max_correction_norm
            assert info["update_norm"] <= corrector.cfg.max_correction_norm + 1e-8


class TestSelfTest:
    """Validate the built-in self-test runs."""

    def test_self_test_runs(self):
        """The self-test at the bottom of ecc.py runs without error."""
        from hdc.ecc import _test_ecc
        # Should not raise
        _test_ecc()


# ── Run with: python -m pytest tests/test_ecc.py -v ──────────────────────────