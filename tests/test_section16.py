"""
tests/test_section16.py
========================
Tests for Section 1.6 HDC modules — the gaps identified in the audit.

Covers:
  hdc/ecc.py              — HDCCorrector, PI control, weight repair
  hdc/fault_models.py     — FaultInjector, all SpikeFI fault types
  hdc/cleanup_memory.py   — ItemMemory, CleanupMemory, release
  hdc/memristive_crossbar.py — MemristiveCrossbar, age(), retention loss
  hdc/autoencoder_bridge.py  — AutoencoderBridge, CrossModalBinding, weighted_fusion
  hdc/resonator.py           — dual memory (Teeters), D2H-AD (Ghajari)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import torch
from hdc.ecc import HDCCorrector, ECCConfig
from hdc.fault_models import FaultInjector, FaultConfig, FaultType
from hdc.cleanup_memory import ItemMemory, CleanupMemory, CleanupConfig
from hdc.memristive_crossbar import MemristiveCrossbar, CrossbarConfig
from hdc.autoencoder_bridge import (
    AutoencoderBridge, BridgeConfig, MultimodalFusion, CrossModalBinding
)
from hdc.resonator import AdaptiveHDClassifier

torch.manual_seed(42)


# ── HDCCorrector / ECC ────────────────────────────────────────────────────────

def test_ecc_detect_anomaly_below_threshold():
    corrector = HDCCorrector(ECCConfig(similarity_threshold=0.7, correction_cooldown=0))
    assert corrector.detect_anomaly(0.5) is True

def test_ecc_detect_anomaly_above_threshold():
    corrector = HDCCorrector(ECCConfig(similarity_threshold=0.7, correction_cooldown=0))
    assert corrector.detect_anomaly(0.9) is False

def test_ecc_detect_anomaly_fires_when_below_threshold():
    """detect_anomaly returns True whenever similarity < threshold (no repair called)."""
    corrector = HDCCorrector(ECCConfig(similarity_threshold=0.7, correction_cooldown=3))
    # last_correction_step starts at -cooldown; steps_since always >= cooldown
    # until repair_weights is called and resets last_correction_step
    assert corrector.detect_anomaly(0.3) is True
    assert corrector.detect_anomaly(0.3) is True   # still triggers — no repair happened

def test_ecc_pi_control_returns_scalar():
    corrector = HDCCorrector(ECCConfig(use_pi_control=True, kp=0.5, ki=0.1,
                                       correction_cooldown=0))
    error = torch.randn(128)
    strength = corrector.compute_correction(0.4, error)
    assert isinstance(strength, float)
    assert 0.0 <= strength <= 1.0

def test_ecc_similarity_history_grows():
    """detect_anomaly should accumulate a similarity history."""
    corrector = HDCCorrector(ECCConfig(similarity_threshold=0.7, correction_cooldown=0))
    for s in [0.9, 0.8, 0.5, 0.6]:
        corrector.detect_anomaly(s)
    assert len(corrector.similarity_history) == 4

def test_ecc_detect_no_anomaly_high_similarity():
    """detect_anomaly must return False when similarity is above threshold."""
    corrector = HDCCorrector(ECCConfig(similarity_threshold=0.5, correction_cooldown=0))
    assert corrector.detect_anomaly(0.95) is False


# ── FaultInjector ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("fault_type", [
    FaultType.STUCK_AT_0,
    FaultType.STUCK_AT_1,
    FaultType.WEIGHT_BITFLIP_TRANSIENT,
    FaultType.WEIGHT_BITFLIP_PERMANENT,
    FaultType.SYNAPTIC_SILENCE,
])
def test_fault_injector_output_shape(fault_type):
    injector = FaultInjector(FaultConfig(fault_type=fault_type, fault_rate=0.1, seed=0))
    W = torch.randn(16, 16)
    corrupted = injector.apply(W)
    assert corrupted.shape == W.shape

def test_stuck_at_0_zeros_some_elements():
    injector = FaultInjector(FaultConfig(
        fault_type=FaultType.STUCK_AT_0, fault_rate=0.5, seed=0))
    W = torch.ones(100)
    corrupted = injector.apply(W)
    assert (corrupted == 0.0).any()

def test_stuck_at_1_ones_some_elements():
    injector = FaultInjector(FaultConfig(
        fault_type=FaultType.STUCK_AT_1, fault_rate=0.5, seed=0))
    W = torch.zeros(100)
    corrupted = injector.apply(W)
    assert (corrupted == 1.0).any()

def test_persistent_fault_reproducible():
    """Persistent faults must corrupt the same positions every call."""
    injector = FaultInjector(FaultConfig(
        fault_type=FaultType.STUCK_AT_0, fault_rate=0.2, persistent=True, seed=42))
    W = torch.ones(50)
    c1 = injector.apply(W)
    c2 = injector.apply(W)
    assert torch.equal(c1, c2)

def test_transient_fault_varies():
    """Transient faults re-sample positions each call — outputs must differ."""
    injector = FaultInjector(FaultConfig(
        fault_type=FaultType.WEIGHT_BITFLIP_TRANSIENT, fault_rate=0.5, persistent=False))
    # Use non-zero weights so bit-flips produce measurably different values
    W = torch.ones(200)
    c1 = injector.apply(W)
    c2 = injector.apply(W)
    # With 50% flip rate on 200 elements the masks almost certainly differ
    assert not torch.equal(c1, c2)

def test_fault_rate_zero_no_change():
    injector = FaultInjector(FaultConfig(
        fault_type=FaultType.STUCK_AT_0, fault_rate=0.0, seed=0))
    W = torch.ones(50)
    corrupted = injector.apply(W)
    assert torch.equal(corrupted, W)

def test_mixed_fault_applies():
    injector = FaultInjector(FaultConfig(fault_type=FaultType.MIXED, fault_rate=0.3, seed=0))
    W = torch.randn(64)
    corrupted = injector.apply(W)
    assert not torch.equal(corrupted, W)


# ── ItemMemory / CleanupMemory ─────────────────────────────────────────────────

def test_item_memory_add_and_get():
    im = ItemMemory(dim=64)
    hv = torch.randint(0, 2, (64,)).float()
    im.add("cat", hv)
    assert "cat" in im
    stored = im.get("cat")
    assert stored is not None and stored.shape == (64,)

def test_item_memory_similarity_nearest():
    im = ItemMemory(dim=128)
    hvs = torch.randint(0, 2, (4, 128)).float()
    labels = ["a", "b", "c", "d"]
    im.add_batch(labels, hvs)
    result = im.nearest(hvs[2])
    assert result is not None
    assert result[0] == "c"
    assert result[1] > 0.9

def test_item_memory_top_k():
    im = ItemMemory(dim=64)
    for i in range(5):
        im.add(str(i), torch.randint(0, 2, (64,)).float())
    results = im.similarity(torch.randint(0, 2, (64,)).float(), top_k=3)
    assert len(results) == 3

def test_cleanup_memory_finds_noisy():
    im = ItemMemory(dim=100)
    hv = torch.randint(0, 2, (100,)).float()
    im.add("dog", hv)
    cm = CleanupMemory(im, CleanupConfig(dim=100, similarity_threshold=0.6))
    # 15% noise
    noisy = hv.clone()
    flip = torch.rand(100) < 0.15
    noisy[flip] = 1.0 - noisy[flip]
    result = cm.cleanup(noisy)
    assert result is not None
    label, _, sim = result
    assert label == "dog" and sim > 0.6

def test_cleanup_memory_rejects_dissimilar():
    im = ItemMemory(dim=64)
    im.add("x", torch.zeros(64))
    cm = CleanupMemory(im, CleanupConfig(dim=64, similarity_threshold=0.9))
    totally_different = torch.ones(64)
    assert cm.cleanup(totally_different) is None

def test_release_finds_components():
    im = ItemMemory(dim=64)
    hv_a = torch.randint(0, 2, (64,)).float()
    hv_b = torch.randint(0, 2, (64,)).float()
    im.add("a", hv_a)
    im.add("b", hv_b)
    cm = CleanupMemory(im, CleanupConfig(dim=64, similarity_threshold=0.3))
    bound = ((hv_a > 0) != (hv_b > 0)).float()
    components = cm.release(bound, max_components=2)
    assert len(components) >= 1


# ── MemristiveCrossbar ────────────────────────────────────────────────────────

def test_crossbar_bind_shape():
    xbar = MemristiveCrossbar(CrossbarConfig(rows=64, cols=8))
    a = torch.randint(0, 2, (64,)).float()
    b = torch.randint(0, 2, (64,)).float()
    result = xbar.bind(a, b)
    assert result.shape == (64,)
    assert set(result.unique().tolist()).issubset({0.0, 1.0})

def test_crossbar_bundle_shape():
    xbar = MemristiveCrossbar(CrossbarConfig(rows=64, cols=8))
    hvs = [torch.randint(0, 2, (64,)).float() for _ in range(3)]
    result = xbar.bundle(hvs)
    assert result.shape == (64,)

def test_crossbar_similarity_search():
    xbar = MemristiveCrossbar(CrossbarConfig(rows=64, cols=4))
    hvs = torch.randint(0, 2, (4, 64)).float()
    xbar.program(hvs, labels=["w", "x", "y", "z"])
    results = xbar.similarity_search(hvs[0], top_k=2)
    assert len(results) == 2
    assert results[0][0] == 0   # self should be nearest

def test_crossbar_energy_tracked():
    xbar = MemristiveCrossbar(CrossbarConfig(rows=32, cols=4))
    hvs = torch.randint(0, 2, (4, 32)).float()
    xbar.program(hvs)
    assert xbar._energy_total_pJ > 0.0

def test_crossbar_age_reduces_conductance():
    xbar = MemristiveCrossbar(CrossbarConfig(rows=32, cols=4, retention_loss=0.1))
    hvs = torch.ones(4, 32)   # all high conductance
    xbar.program(hvs)
    conductance_before = xbar._conductance.mean().item()
    stats = xbar.age(hours=10.0)
    conductance_after = xbar._conductance.mean().item()
    assert conductance_after < conductance_before
    assert "bit_errors" in stats

def test_crossbar_age_zero_hours_no_change():
    xbar = MemristiveCrossbar(CrossbarConfig(rows=32, cols=4, retention_loss=0.1))
    xbar.program(torch.ones(4, 32))
    before = xbar._conductance.clone()
    xbar.age(hours=0.0)
    assert torch.allclose(xbar._conductance, before, atol=1e-5)


# ── AutoencoderBridge / CrossModalBinding ─────────────────────────────────────

def test_bridge_encode_shape():
    bridge = AutoencoderBridge(BridgeConfig(input_dim=32, hdc_dim=128, encoding_layers=2))
    x = torch.randn(4, 32)
    hv = bridge.encode(x)
    assert hv.shape == (4, 128)

def test_bridge_encode_binary():
    bridge = AutoencoderBridge(BridgeConfig(input_dim=16, hdc_dim=64, encoding_layers=2))
    hv = bridge.encode(torch.randn(2, 16))
    assert set(hv.flatten().unique().tolist()).issubset({0.0, 1.0})

def test_bridge_decode_shape():
    bridge = AutoencoderBridge(BridgeConfig(input_dim=16, hdc_dim=64, encoding_layers=2))
    hv = torch.randint(0, 2, (2, 64)).float()
    recon = bridge.decode(hv)
    assert recon.shape == (2, 16)

def test_weighted_fusion_zero_weights():
    """weighted_fusion with all-zero weights must not raise."""
    fusion = MultimodalFusion(hdc_dim=64)
    hv1 = torch.randint(0, 2, (64,)).float()
    hv2 = torch.randint(0, 2, (64,)).float()
    result = fusion.weighted_fusion([hv1, hv2], [0.0, 0.0])
    assert result.shape == (64,)

def test_cross_modal_binding_encode_shape():
    binding = CrossModalBinding(hdc_dim=64, modalities=["a", "b"])
    hv_a = torch.randint(0, 2, (64,)).float()
    hv_b = torch.randint(0, 2, (64,)).float()
    joint = binding.encode({"a": hv_a, "b": hv_b})
    assert joint.shape == (64,)

def test_cross_modal_decode_approximate_recovery():
    """After encoding and decoding, result should be similar to original."""
    binding = CrossModalBinding(hdc_dim=256, modalities=["x"])
    hv_x = torch.randint(0, 2, (256,)).float()
    joint = binding.encode({"x": hv_x})
    recovered = binding.decode(joint, "x")
    # Decoding single-modality joint should be identical to the original
    # (no other modalities to interfere)
    sim = 1.0 - ((hv_x > 0) != (recovered > 0)).float().mean().item()
    assert sim == pytest.approx(1.0, abs=0.01)


# ── AdaptiveHDClassifier — Teeters + Ghajari extensions ──────────────────────

def test_dual_memory_predict_dual_shape():
    clf = AdaptiveHDClassifier(n_features=4, n_classes=3, dim=64, seed=0)
    clf.enable_dual_memory()
    x = torch.randn(4)
    for _ in range(10):
        clf.train_step(x, label=0)
    pred, sims = clf.predict_dual(x)
    assert 0 <= pred < 3
    assert sims.shape == (3,)

def test_dual_memory_st_lt_diverge_after_training():
    """After training, ST and LT prototypes should be distinct."""
    clf = AdaptiveHDClassifier(n_features=4, n_classes=2, dim=64, seed=0)
    clf.enable_dual_memory(consolidation_steps=5)
    for i in range(30):
        clf.train_step(torch.randn(4), label=i % 2)
    assert not torch.equal(clf.st_hvs, clf.lt_hvs)

def test_d2h_anomaly_score_range():
    clf = AdaptiveHDClassifier(n_features=4, n_classes=2, dim=64, seed=0)
    clf.enable_anomaly_detection(percentile=90.0, warmup_steps=10)
    for i in range(15):
        clf.update_anomaly_threshold(torch.randn(4))
    score, _ = clf.anomaly_score(torch.randn(4))
    assert 0.0 <= score <= 1.0

def test_d2h_threshold_set_after_warmup():
    clf = AdaptiveHDClassifier(n_features=4, n_classes=2, dim=64, seed=0)
    clf.enable_anomaly_detection(percentile=95.0, warmup_steps=5)
    for i in range(5):
        clf.update_anomaly_threshold(torch.randn(4))
    assert clf._d2h_threshold is not None
