"""Unit tests for OracleDefense poison detection."""

import torch
import pytest
from training.oracle_defense import OracleDefense, DefenseConfig


DIM = 64
N_CLASSES = 3


@pytest.fixture
def warmed_guard():
    guard = OracleDefense(DefenseConfig(
        n_classes=N_CLASSES, hdc_dim=DIM, warmup_samples=5,
        sim_thresh=0.05, z_thresh=2.5,
    ))
    # Seed with clean class-distinguishable HVs
    for cls in range(N_CLASSES):
        for _ in range(8):
            hv = torch.zeros(DIM)
            hv[cls * (DIM // N_CLASSES):(cls + 1) * (DIM // N_CLASSES)] = 1.0
            hv = hv / hv.norm()
            guard.update(hv, cls)
    return guard


def test_clean_sample_passes(warmed_guard):
    guard = warmed_guard
    hv = torch.zeros(DIM)
    hv[0:DIM // N_CLASSES] = 1.0
    hv = hv / hv.norm()
    verdict = guard.check(hv, 0)
    assert verdict.clean, f"Expected clean, got: {verdict}"


def test_wrong_label_suspected(warmed_guard):
    guard = warmed_guard
    # HV clearly belongs to class 0 but claimed to be class 2
    hv = torch.zeros(DIM)
    hv[0:DIM // N_CLASSES] = 1.0
    hv = hv / hv.norm()
    verdict = guard.check(hv, 2)
    assert not verdict.clean, f"Expected suspect, got: {verdict}"


def test_verdict_fields_populated(warmed_guard):
    guard = warmed_guard
    hv = torch.randn(DIM)
    hv = hv / hv.norm()
    verdict = guard.check(hv, 0)
    assert 0 <= verdict.claimed_label < N_CLASSES
    assert isinstance(verdict.sim_to_claimed, float)
    assert isinstance(verdict.z_score, float)


def test_update_increments_count(warmed_guard):
    guard = warmed_guard
    before = guard.counts[0].item()
    hv = torch.randn(DIM)
    guard.update(hv / hv.norm(), 0)
    assert guard.counts[0].item() == before + 1
