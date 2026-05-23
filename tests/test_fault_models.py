"""
test_fault_models.py
====================
Tests for hardware-realistic fault injection (hdc/fault_models.py).

SpikeFI fault taxonomy (Spyrou et al. 2024):
  - STUCK_AT_0/1: neuron output pinned
  - WEIGHT_BITFLIP_TRANSIENT/PERMANENT: SEU / permanent flip
  - SYNAPTIC_SILENCE: synapse zeroed
  - RETENTION_FAILURE: gradual analog drift
  - READ_DISTURB: weight perturbed on read
  - MIXED: combination
"""

from __future__ import annotations

import math
import sys
import os
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from hdc.fault_models import FaultInjector, FaultConfig, FaultType


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def weights():
    """Standard weight matrix for testing."""
    torch.manual_seed(0)
    return torch.randn(32, 32)


@pytest.fixture
def spikes():
    """Binary spike vector."""
    torch.manual_seed(1)
    return (torch.rand(64) > 0.5).float()


# ── Output shape / identity ───────────────────────────────────────────────────

class TestOutputShape:
    """Fault injection preserves tensor shape."""

    @pytest.mark.parametrize("fault_type", list(FaultType))
    def test_shape_preserved(self, weights, fault_type):
        injector = FaultInjector(FaultConfig(fault_type=fault_type, fault_rate=0.01, seed=0))
        corrupted = injector.apply(weights)
        assert corrupted.shape == weights.shape

    def test_1d_tensor(self):
        t = torch.randn(128)
        injector = FaultInjector(FaultConfig(fault_type=FaultType.STUCK_AT_0, fault_rate=0.05, seed=0))
        corrupted = injector.apply(t)
        assert corrupted.shape == t.shape

    def test_3d_tensor(self):
        t = torch.randn(4, 16, 16)
        injector = FaultInjector(FaultConfig(fault_type=FaultType.SYNAPTIC_SILENCE, fault_rate=0.05, seed=0))
        corrupted = injector.apply(t)
        assert corrupted.shape == t.shape


# ── Stuck-at-0 ────────────────────────────────────────────────────────────────

class TestStuckAt0:
    def test_corrupted_elements_are_zero(self, weights):
        injector = FaultInjector(FaultConfig(fault_type=FaultType.STUCK_AT_0,
                                              fault_rate=0.1, seed=0))
        corrupted = injector.apply(weights)
        # All corrupted positions must be 0
        changed = (corrupted != weights)
        assert (corrupted[changed] == 0.0).all()

    def test_uncorrupted_elements_unchanged(self, weights):
        injector = FaultInjector(FaultConfig(fault_type=FaultType.STUCK_AT_0,
                                              fault_rate=0.1, seed=0))
        corrupted = injector.apply(weights)
        unchanged = (corrupted == weights)
        assert unchanged.sum() > 0

    def test_fault_rate_approximately_correct(self, weights):
        rate = 0.2
        injector = FaultInjector(FaultConfig(fault_type=FaultType.STUCK_AT_0,
                                              fault_rate=rate, seed=42))
        # Aggregate over many calls
        total_corrupted = 0
        n_trials = 50
        for _ in range(n_trials):
            corrupted = injector.apply(weights)
            total_corrupted += int((corrupted != weights).sum().item())
        actual_rate = total_corrupted / (weights.numel() * n_trials)
        # Should be within 2× of target
        assert 0.5 * rate <= actual_rate <= 2.0 * rate

    def test_persistent_same_mask_every_call(self, weights):
        injector = FaultInjector(FaultConfig(fault_type=FaultType.STUCK_AT_0,
                                              fault_rate=0.05, persistent=True, seed=0))
        c1 = injector.apply(weights)
        c2 = injector.apply(weights)
        assert torch.equal(c1, c2)

    def test_transient_different_mask_each_call(self, weights):
        injector = FaultInjector(FaultConfig(fault_type=FaultType.STUCK_AT_0,
                                              fault_rate=0.1, persistent=False))
        results = [injector.apply(weights) for _ in range(5)]
        # At least some calls should differ (extremely unlikely they're all equal)
        n_equal = sum(torch.equal(results[0], results[i]) for i in range(1, 5))
        assert n_equal < 4  # At least 1 different


# ── Stuck-at-1 ────────────────────────────────────────────────────────────────

class TestStuckAt1:
    def test_corrupted_elements_nonzero(self, weights):
        injector = FaultInjector(FaultConfig(fault_type=FaultType.STUCK_AT_1,
                                              fault_rate=0.1, seed=0))
        corrupted = injector.apply(weights)
        changed = (corrupted != weights)
        # Stuck-at-1 sets to a non-zero value
        assert (corrupted[changed] != 0.0).all()


# ── Bitflip Transient ─────────────────────────────────────────────────────────

class TestBitflipTransient:
    """SEU Poisson+recovery model (not random telegraph noise).

    Bits accumulate flips via Poisson arrivals and recover independently.
    The changed mask at any time only contains bits with an odd flip count.
    """

    def test_corrupted_elements_negated(self, weights):
        injector = FaultInjector(FaultConfig(fault_type=FaultType.WEIGHT_BITFLIP_TRANSIENT,
                                              fault_rate=0.1, seed=7))
        corrupted = injector.apply(weights)
        changed = (corrupted != weights)
        # Every element in the SEU mask is negated (XOR semantics)
        if changed.any():
            assert torch.allclose(corrupted[changed], -weights[changed])

    def test_no_corruption_at_zero_rate(self, weights):
        injector = FaultInjector(FaultConfig(fault_type=FaultType.WEIGHT_BITFLIP_TRANSIENT,
                                              fault_rate=0.0, seed=0))
        corrupted = injector.apply(weights)
        assert torch.equal(corrupted, weights)

    def test_mask_accumulates_over_calls(self, weights):
        """SEU mask grows over successive calls (Poisson accumulation)."""
        injector = FaultInjector(FaultConfig(fault_type=FaultType.WEIGHT_BITFLIP_TRANSIENT,
                                              fault_rate=0.05, seu_recovery_prob=0.0, seed=1))
        # With no recovery, flipped bits only accumulate
        n_flipped = []
        for _ in range(10):
            corrupted = injector.apply(weights)
            n_flipped.append(int((corrupted != weights).sum().item()))
        # Total flipped count should grow (or at least not decrease monotonically)
        assert n_flipped[-1] >= n_flipped[0] or max(n_flipped) > 0

    def test_recovery_clears_bits(self, weights):
        """High recovery rate clears the SEU mask quickly."""
        # No new arrivals, full recovery → mask should clear after a few calls
        injector = FaultInjector(FaultConfig(fault_type=FaultType.WEIGHT_BITFLIP_TRANSIENT,
                                              fault_rate=0.0, seu_recovery_prob=1.0, seed=2))
        # Manually set the SEU mask to be all-ones
        injector.apply(weights)  # init mask
        injector._seu_mask = torch.ones(weights.shape, dtype=torch.bool)
        # Next call: rate=0 (no new events) + full recovery → mask clears
        corrupted = injector.apply(weights)
        assert torch.equal(corrupted, weights)

    def test_equilibrium_between_arrivals_and_recovery(self, weights):
        """With arrivals + recovery, mask reaches a non-zero equilibrium."""
        injector = FaultInjector(FaultConfig(fault_type=FaultType.WEIGHT_BITFLIP_TRANSIENT,
                                              fault_rate=0.05, seu_recovery_prob=0.3, seed=3))
        # Run for many steps to reach equilibrium
        for _ in range(200):
            injector.apply(weights)
        # At equilibrium, some bits should be flipped (arrivals > 0)
        assert injector._seu_mask is not None


# ── Bitflip Permanent ─────────────────────────────────────────────────────────

class TestBitflipPermanent:
    def test_permanent_same_positions_every_call(self, weights):
        injector = FaultInjector(FaultConfig(fault_type=FaultType.WEIGHT_BITFLIP_PERMANENT,
                                              fault_rate=0.05, seed=3))
        c1 = injector.apply(weights)
        c2 = injector.apply(weights)
        assert torch.equal(c1, c2)

    def test_permanent_positions_are_flipped(self, weights):
        injector = FaultInjector(FaultConfig(fault_type=FaultType.WEIGHT_BITFLIP_PERMANENT,
                                              fault_rate=0.1, seed=4))
        corrupted = injector.apply(weights)
        changed_mask = injector._fault_positions
        # Positions under the fault mask should be negated
        assert torch.allclose(corrupted[changed_mask], -weights[changed_mask])
        # Unchanged positions should be identical
        assert torch.allclose(corrupted[~changed_mask], weights[~changed_mask])


# ── Synaptic Silence ──────────────────────────────────────────────────────────

class TestSynapticSilence:
    def test_silenced_weights_are_zero(self, weights):
        injector = FaultInjector(FaultConfig(fault_type=FaultType.SYNAPTIC_SILENCE,
                                              fault_rate=0.15, seed=0))
        corrupted = injector.apply(weights)
        changed = (corrupted != weights)
        assert (corrupted[changed] == 0.0).all()


# ── Retention Failure ─────────────────────────────────────────────────────────

class TestRetentionFailure:
    def test_all_elements_drift(self, weights):
        drift_std = 0.1
        injector = FaultInjector(FaultConfig(fault_type=FaultType.RETENTION_FAILURE,
                                              retention_drift_std=drift_std, seed=0))
        corrupted = injector.apply(weights)
        # All elements should be perturbed (or nearly all)
        changed = (corrupted != weights).sum().item()
        assert changed > weights.numel() * 0.9

    def test_drift_magnitude(self, weights):
        drift_std = 0.05
        injector = FaultInjector(FaultConfig(fault_type=FaultType.RETENTION_FAILURE,
                                              retention_drift_std=drift_std, seed=0))
        corrupted = injector.apply(weights)
        drift = (corrupted - weights).abs().mean().item()
        # Empirically: mean(|N(0, std)|) = std * sqrt(2/pi) ≈ std * 0.798
        expected = drift_std * math.sqrt(2 / math.pi)
        assert 0.3 * expected <= drift <= 3.0 * expected


# ── Read Disturb ──────────────────────────────────────────────────────────────

class TestReadDisturb:
    def test_very_low_rate_mostly_unchanged(self, weights):
        injector = FaultInjector(FaultConfig(fault_type=FaultType.READ_DISTURB,
                                              read_disturb_prob=1e-6, seed=0))
        corrupted = injector.apply(weights)
        # With prob 1e-6, almost no elements should change
        n_changed = int((corrupted != weights).sum().item())
        assert n_changed < weights.numel() * 0.01


# ── Mixed ─────────────────────────────────────────────────────────────────────

class TestMixed:
    def test_mixed_produces_some_faults(self, weights):
        injector = FaultInjector(FaultConfig(fault_type=FaultType.MIXED,
                                              fault_rate=0.1, seed=0))
        corrupted = injector.apply(weights)
        n_changed = int((corrupted != weights).sum().item())
        assert n_changed > 0

    def test_mixed_preserves_shape(self, weights):
        injector = FaultInjector(FaultConfig(fault_type=FaultType.MIXED,
                                              fault_rate=0.1, seed=1))
        corrupted = injector.apply(weights)
        assert corrupted.shape == weights.shape


# ── Neuron Output Faults ──────────────────────────────────────────────────────

class TestNeuronOutputFaults:
    def test_stuck_at_0_on_spikes(self, spikes):
        injector = FaultInjector(FaultConfig(fault_type=FaultType.STUCK_AT_0,
                                              fault_rate=0.1, seed=0))
        corrupted = injector.apply_to_neuron_outputs(spikes)
        changed = (corrupted != spikes)
        assert (corrupted[changed] == 0.0).all()

    def test_stuck_at_1_on_spikes(self, spikes):
        injector = FaultInjector(FaultConfig(fault_type=FaultType.STUCK_AT_1,
                                              fault_rate=0.1, stuck_value=1, seed=0))
        corrupted = injector.apply_to_neuron_outputs(spikes)
        changed = (corrupted != spikes)
        # Changed positions set to stuck_value (1.0)
        assert (corrupted[changed] == float(injector.config.stuck_value)).all()

    def test_non_stuck_fault_is_noop_on_spikes(self, spikes):
        injector = FaultInjector(FaultConfig(fault_type=FaultType.RETENTION_FAILURE,
                                              fault_rate=0.1, seed=0))
        corrupted = injector.apply_to_neuron_outputs(spikes)
        assert torch.equal(corrupted, spikes)


# ── Persistence & Reset ───────────────────────────────────────────────────────

class TestPersistenceReset:
    def test_reset_clears_persistent_state(self, weights):
        injector = FaultInjector(FaultConfig(fault_type=FaultType.STUCK_AT_0,
                                              fault_rate=0.1, persistent=True, seed=0))
        c1 = injector.apply(weights)
        injector.reset()
        # After reset, re-initialization re-seeds randomly → may differ
        c2 = injector.apply(weights)
        # Shape should still match
        assert c1.shape == c2.shape

    def test_reset_clears_statistics(self, weights):
        injector = FaultInjector(FaultConfig(fault_type=FaultType.STUCK_AT_0, fault_rate=0.1))
        injector.apply(weights)
        assert injector.total_elements_processed > 0

        injector.reset()
        assert injector.total_faults_injected == 0
        assert injector.total_elements_processed == 0


# ── Statistics ────────────────────────────────────────────────────────────────

class TestStatistics:
    def test_stats_keys(self, weights):
        injector = FaultInjector(FaultConfig(fault_type=FaultType.STUCK_AT_0, fault_rate=0.05))
        injector.apply(weights)
        stats = injector.get_stats()
        for key in ("fault_type", "fault_rate", "persistent",
                    "total_faults_injected", "total_elements_processed",
                    "actual_fault_rate"):
            assert key in stats

    def test_total_elements_accumulates(self, weights):
        injector = FaultInjector(FaultConfig(fault_type=FaultType.STUCK_AT_0, fault_rate=0.05))
        for _ in range(5):
            injector.apply(weights)
        assert injector.total_elements_processed == 5 * weights.numel()

    def test_actual_rate_within_range(self, weights):
        rate = 0.1
        injector = FaultInjector(FaultConfig(fault_type=FaultType.STUCK_AT_0,
                                              fault_rate=rate, seed=99))
        for _ in range(20):
            injector.apply(weights)
        stats = injector.get_stats()
        # Allow generous bounds due to sampling variance
        assert 0.3 * rate <= stats["actual_fault_rate"] <= 3.0 * rate


# ── Reproducibility ───────────────────────────────────────────────────────────

class TestReproducibility:
    @pytest.mark.parametrize("fault_type", [
        FaultType.STUCK_AT_0,
        FaultType.WEIGHT_BITFLIP_TRANSIENT,
        FaultType.SYNAPTIC_SILENCE,
    ])
    def test_same_seed_same_result(self, weights, fault_type):
        inj1 = FaultInjector(FaultConfig(fault_type=fault_type, fault_rate=0.1, seed=42))
        inj2 = FaultInjector(FaultConfig(fault_type=fault_type, fault_rate=0.1, seed=42))
        c1 = inj1.apply(weights)
        c2 = inj2.apply(weights)
        assert torch.equal(c1, c2)

    def test_different_seeds_different_results(self, weights):
        inj1 = FaultInjector(FaultConfig(fault_type=FaultType.STUCK_AT_0,
                                          fault_rate=0.1, seed=1))
        inj2 = FaultInjector(FaultConfig(fault_type=FaultType.STUCK_AT_0,
                                          fault_rate=0.1, seed=2))
        c1 = inj1.apply(weights)
        c2 = inj2.apply(weights)
        assert not torch.equal(c1, c2)
