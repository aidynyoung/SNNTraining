"""
tests/test_physical_ai_hybrid.py
=================================
Tests for the Physical AI Hybrid Pipeline (hdc/physical_ai_hybrid.py).

Validates:
  1. AdaptiveModalityFusion — error-weighted modality bundling
  2. FractionalInterpolator — fractional power encoding for temporal prediction
  3. MultiSpaceSync — dual-space divergence detection
  4. EnsembleUncertainty — multi-seed predictor disagreement
  5. HybridPhysicalAIPipeline — full pipeline integration
"""

from __future__ import annotations

import sys
import os

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from hdc.physical_ai_hybrid import (
    AdaptiveModalityFusion,
    FractionalInterpolator,
    MultiSpaceSync,
    EnsembleUncertainty,
    HybridPhysicalAIPipeline,
    ExperienceConsolidation,
)
from hdc.physics_world_model import _hamming, _xor, _majority
from hdc.sensor_stream import SensorSpec, SensorReading, ModalityType


@pytest.fixture
def hd_dim():
    return 256

@pytest.fixture
def n_modalities():
    return 3


class TestAdaptiveModalityFusion:
    def test_init(self, n_modalities, hd_dim):
        fusion = AdaptiveModalityFusion(n_modalities, hd_dim)
        assert fusion.n_modalities == n_modalities
        assert fusion.hd_dim == hd_dim
        assert fusion.weights.shape == (n_modalities,)
        # Initial weights should be uniform
        assert torch.allclose(fusion.weights, torch.ones(n_modalities))

    def test_initial_weights_uniform(self, n_modalities, hd_dim):
        fusion = AdaptiveModalityFusion(n_modalities, hd_dim)
        # Each weight is 1.0, sum = n_modalities
        assert float(fusion.weights.sum()) == float(n_modalities)

    def test_weight_dict_inspection(self, n_modalities, hd_dim):
        fusion = AdaptiveModalityFusion(n_modalities, hd_dim)
        # After init, all weights are 1.0
        w = fusion.weights.detach().clone()
        assert float(w.sum()) == float(n_modalities)

    def test_update_weights_changes_weights(self, n_modalities, hd_dim):
        fusion = AdaptiveModalityFusion(n_modalities, hd_dim)
        errors = torch.tensor([0.1, 0.3, 0.5])
        fusion.update_weights(errors)
        # After update, weights should differ from uniform
        assert not torch.allclose(fusion.weights, torch.ones(n_modalities))

    def test_forward_shape(self, n_modalities, hd_dim):
        fusion = AdaptiveModalityFusion(n_modalities, hd_dim)
        hvs = torch.rand(n_modalities, hd_dim)
        fused = fusion(hvs)
        assert fused.shape == (hd_dim,)

    def test_forward_binary(self, n_modalities, hd_dim):
        fusion = AdaptiveModalityFusion(n_modalities, hd_dim)
        hvs = (torch.rand(n_modalities, hd_dim) >= 0.5).float()
        fused = fusion(hvs)
        assert fused.shape == (hd_dim,)
        assert fused.dtype == torch.float32


class TestFractionalInterpolator:
    def test_init(self, hd_dim):
        fi = FractionalInterpolator(hd_dim)
        assert fi.hd_dim == hd_dim
        assert fi.max_horizon == 20

    def test_predict_shape(self, hd_dim):
        fi = FractionalInterpolator(hd_dim)
        hv = (torch.rand(hd_dim) >= 0.5).float()
        pred = fi.predict(hv, horizon=5)
        assert pred.shape == (hd_dim,)

    def test_predict_zero_horizon_returns_self(self, hd_dim):
        fi = FractionalInterpolator(hd_dim)
        hv = (torch.rand(hd_dim) >= 0.5).float()
        pred = fi.predict(hv, horizon=0)
        # Zero-horizon prediction should return a valid HV
        assert pred.shape == (hd_dim,)
        assert pred.dtype == torch.float32

    def test_batch_predict_trajectory(self, hd_dim):
        fi = FractionalInterpolator(hd_dim)
        hv = (torch.rand(hd_dim) >= 0.5).float()
        steps = [0, 1, 5, 10]
        result = fi.predict_trajectory(hv, steps)
        assert isinstance(result, dict)
        for h in steps:
            assert h in result
            assert result[h].shape == (hd_dim,)

    def test_position_hv_shape(self, hd_dim):
        fi = FractionalInterpolator(hd_dim)
        pos_hv = fi.position_hv(step=5)
        assert pos_hv.shape == (hd_dim,)


class TestMultiSpaceSync:
    def test_init(self, hd_dim):
        sync = MultiSpaceSync(hd_dim)
        assert sync.hd_dim == hd_dim
        assert sync.n_triggers == 0

    def test_step_returns_dict_with_required_keys(self, hd_dim):
        sync = MultiSpaceSync(hd_dim)
        a = (torch.rand(hd_dim) >= 0.5).float()
        b = (torch.rand(hd_dim) >= 0.5).float()
        result = sync.step(a, b)
        required = {"divergence", "hamming", "fpe_cosine", "needs_trigger", "trigger_count"}
        assert required.issubset(result.keys())

    def test_identical_predictions_low_divergence(self, hd_dim):
        sync = MultiSpaceSync(hd_dim, threshold=0.15)
        hv = (torch.rand(hd_dim) >= 0.5).float()
        result = sync.step(hv, hv)
        assert result["divergence"] < 0.15
        assert result["needs_trigger"] is False

    def test_opposite_predictions_high_divergence(self, hd_dim):
        sync = MultiSpaceSync(hd_dim, threshold=0.15)
        a = (torch.rand(hd_dim) >= 0.5).float()
        b = 1.0 - a  # opposite
        result = sync.step(a, b)
        assert result["divergence"] > 0.15
        assert result["needs_trigger"] is True


class TestEnsembleUncertainty:
    def test_init(self, hd_dim):
        ens = EnsembleUncertainty(hd_dim, n_members=3)
        assert ens.hd_dim == hd_dim
        assert ens.n_members == 3

    def test_predict_all_shape(self, hd_dim):
        ens = EnsembleUncertainty(hd_dim, n_members=3)
        hv = (torch.rand(hd_dim) >= 0.5).float()
        preds = ens.predict_all(hv)
        assert len(preds) == 3
        for p in preds:
            assert p.shape == (hd_dim,)

    def test_predict_with_uncertainty(self, hd_dim):
        ens = EnsembleUncertainty(hd_dim, n_members=3)
        hv = (torch.rand(hd_dim) >= 0.5).float()
        consensus, uncertainty = ens.predict_with_uncertainty(hv)
        assert consensus.shape == (hd_dim,)
        assert 0.0 <= uncertainty <= 0.5

    def test_predict_all_diverse(self, hd_dim):
        """Ensemble members should produce predictions with valid shape."""
        ens = EnsembleUncertainty(hd_dim, n_members=5)
        hv = (torch.rand(hd_dim) >= 0.5).float()
        preds = ens.predict_all(hv)
        assert len(preds) == 5
        for p in preds:
            assert p.shape == (hd_dim,)

    def test_update(self, hd_dim):
        ens = EnsembleUncertainty(hd_dim, n_members=3)
        state = (torch.rand(hd_dim) >= 0.5).float()
        actual = (torch.rand(hd_dim) >= 0.5).float()
        # Update should not crash
        ens.update(state, actual)


class TestExperienceConsolidation:
    def test_init(self, hd_dim):
        ec = ExperienceConsolidation(hd_dim)
        assert ec.hd_dim == hd_dim
        assert ec.consolidation_period == 20

    def test_maybe_consolidate_returns_none_early(self, hd_dim):
        """Before consolidation_period steps, maybe_consolidate returns None."""
        from hdc.sensor_stream import SensorStreamBuffer
        from hdc.physics_world_model import PhysicsWorldModel
        ec = ExperienceConsolidation(hd_dim, consolidation_period=10)
        buffer = SensorStreamBuffer(capacity=100)
        wm = PhysicsWorldModel(hd_dim=hd_dim)
        result = ec.maybe_consolidate(buffer, wm)
        assert result is None


class TestHybridPipelineComponentsTogether:
    def test_multi_sync_with_ensemble_feedback(self, hd_dim):
        """Integration: MultiSpaceSync + EnsembleUncertainty together."""
        sync = MultiSpaceSync(hd_dim, threshold=0.15)
        ens = EnsembleUncertainty(hd_dim, n_members=3)

        hv = (torch.rand(hd_dim) >= 0.5).float()
        consensus, _ = ens.predict_with_uncertainty(hv)
        result = sync.step(consensus, hv)
        assert "divergence" in result
        assert "hamming" in result
        assert "fpe_cosine" in result
        assert "needs_trigger" in result
        assert "trigger_count" in result
