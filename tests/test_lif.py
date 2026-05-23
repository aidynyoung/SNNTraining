import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import pytest
from models.lif import LIFLayer, LIFConfig


def make_lif(size=16):
    return LIFLayer(LIFConfig(size=size))


def test_lif_output_shape():
    lif = make_lif(32)
    current = torch.randn(32)
    spikes = lif.step(current)
    assert spikes.shape == (32,)


def test_lif_spikes_binary():
    lif = make_lif(64)
    for _ in range(20):
        spikes = lif.step(torch.randn(64) * 2.0)
    assert set(spikes.unique().tolist()).issubset({0.0, 1.0})


def test_lif_reset_on_spike():
    """Neurons that spike should have v == v_reset immediately after."""
    cfg = LIFConfig(size=4, v_th=0.5, v_reset=0.0, refractory=0)
    lif = LIFLayer(cfg)
    # Force large input to guarantee spikes
    spikes = lif.step(torch.tensor([10.0, 10.0, 10.0, 10.0]))
    assert spikes.sum() > 0
    spiked = spikes.bool()
    assert (lif.v[spiked] == cfg.v_reset).all()


def test_lif_refractory_period():
    """Neurons in refractory period should not fire."""
    cfg = LIFConfig(size=2, v_th=0.5, v_reset=0.0, refractory=5)
    lif = LIFLayer(cfg)
    # Force spike
    lif.step(torch.tensor([10.0, 10.0]))
    # Next 5 steps: large input but should not spike (refractory)
    for _ in range(5):
        spikes = lif.step(torch.tensor([10.0, 10.0]))
        assert spikes.sum() == 0, "Neurons fired during refractory period"


def test_lif_firing_rates():
    lif = make_lif(8)
    for _ in range(200):
        lif.step(torch.ones(8) * 2.0)
    rates = lif.get_firing_rates(window=100)
    assert rates.shape == (8,)
    assert (rates >= 0).all() and (rates <= 1).all()


def test_lif_reset_clears_state():
    lif = make_lif(16)
    for _ in range(50):
        lif.step(torch.randn(16))
    lif.reset()
    assert lif.v.sum() == 0.0
    assert len(lif.spike_hist) == 0


# ── Threshold adaptation (Zhao et al. 2026) ───────────────────────────────────

def test_threshold_adaptation_enabled():
    cfg = LIFConfig(size=8, enable_threshold_adaptation=True,
                    threshold_adaptation_rate=0.1, threshold_momentum=0.9)
    lif = LIFLayer(config=cfg)
    assert lif.enable_threshold_adaptation is True
    initial = lif.v_th
    for _ in range(100):
        lif.step(torch.ones(8) * 3.0)
    assert lif.v_th != initial, "Threshold should drift under sustained input"

def test_threshold_adaptation_disabled_unchanged():
    lif = LIFLayer(LIFConfig(size=8, v_th=1.0, enable_threshold_adaptation=False))
    for _ in range(100):
        lif.step(torch.ones(8) * 3.0)
    assert lif.v_th == pytest.approx(1.0)

def test_threshold_reset_restores_nominal():
    cfg = LIFConfig(size=4, v_th=1.0, enable_threshold_adaptation=True,
                    threshold_adaptation_rate=0.2, threshold_momentum=0.8)
    lif = LIFLayer(config=cfg)
    for _ in range(50):
        lif.step(torch.ones(4) * 5.0)
    assert lif.v_th != pytest.approx(1.0)
    lif.reset(reset_threshold=True)
    assert lif.v_th == pytest.approx(1.0)
    assert lif._v_running_mean is None

def test_threshold_persists_across_reset_without_flag():
    """Without reset_threshold=True, adapted threshold should persist."""
    cfg = LIFConfig(size=4, v_th=1.0, enable_threshold_adaptation=True,
                    threshold_adaptation_rate=0.2, threshold_momentum=0.8)
    lif = LIFLayer(config=cfg)
    for _ in range(50):
        lif.step(torch.ones(4) * 5.0)
    adapted = lif.v_th
    lif.reset(reset_threshold=False)
    assert lif.v_th == pytest.approx(adapted)
