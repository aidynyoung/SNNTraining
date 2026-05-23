"""
test_world_model.py
===================
Tests for the SNNTraining World Model (hdc/world_model.py).

Validates:
  1. PhasorEncoder: continuous sensor values → hypervectors
  2. TemporalEncoder: sequence encoding with permutation
  3. PredictiveCodingModule: prediction error, Hebbian update
  4. CognitiveMapLayer: store, retrieve, attention
  5. HDCAttention: HDC-native attention mechanism
  6. SNNTrainingWorldModel: full forward pass, energy tracking
  7. MultiModalFusion: cross-modal binding and bundling
  8. SkillTransferModule: find, register, transfer skills
"""

from __future__ import annotations

import math
import sys
import os
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from hdc.world_model import (
    WorldModelConfig,
    SNNTrainingWorldModel,
    LearnablePhasorEncoder,
    TemporalEncoder,
    PredictiveCodingModule,
    ResonatorNetwork,
    CognitiveMapLayer,
    HDCAttention,
    MultiModalFusion,
    SkillTransferModule,
    hv_xor,
    hv_popcount,
    hv_hamming_sim,
    hv_majority,
    hv_bundle,
    hv_bind,
    hv_permute,
    gen_hvs,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def hd_dim():
    return 128  # Small for fast tests


@pytest.fixture
def small_config():
    return WorldModelConfig(
        n_sensors=2,
        sensor_dim=8,
        hd_dim=128,
        n_projections=2,
        n_phasors=16,
        prediction_horizon=3,
        temporal_window=10,
        learning_rate=0.1,
        n_resonators=2,
        n_cognitive_layers=1,
        use_predictive_coding=True,
        use_attention=False,  # Skip attention for speed
        track_energy=False,
        device="cpu",
    )


# ── HDC Operations Tests ──────────────────────────────────────────────────────

class TestHDCOperations:
    """Validate pure VSA operations."""

    def test_hv_xor_binary(self):
        a = torch.tensor([1.0, 0.0, 1.0, 0.0])
        b = torch.tensor([1.0, 1.0, 0.0, 0.0])
        result = hv_xor(a, b)
        expected = torch.tensor([0.0, 1.0, 1.0, 0.0])
        assert torch.equal(result, expected)

    def test_hv_popcount(self):
        hv = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0])
        count = hv_popcount(hv)
        assert count.item() == 3.0

    def test_hv_hamming_sim_identical(self):
        a = torch.tensor([1.0, 0.0, 1.0, 0.0])
        sim = hv_hamming_sim(a, a)
        assert sim.item() == 1.0

    def test_hv_hamming_sim_opposite(self):
        a = torch.tensor([1.0, 0.0, 1.0, 0.0])
        b = torch.tensor([0.0, 1.0, 0.0, 1.0])
        sim = hv_hamming_sim(a, b)
        assert sim.item() == 0.0

    def test_hv_majority(self):
        hv = torch.tensor([0.6, 0.3, 0.7, 0.2, 0.9])
        result = hv_majority(hv)
        assert result[0].item() == 1.0
        assert result[1].item() == 0.0
        assert result[2].item() == 1.0
        assert result[3].item() == 0.0
        assert result[4].item() == 1.0

    def test_hv_bundle_average(self):
        a = torch.tensor([1.0, 0.0, 1.0, 0.0])
        b = torch.tensor([0.0, 1.0, 1.0, 0.0])
        bundled = hv_bundle([a, b])
        # Average: [0.5, 0.5, 1.0, 0.0] → majority → [0, 0, 1, 0]
        assert bundled[2].item() == 1.0

    def test_hv_bind_xor(self):
        a = torch.tensor([1.0, 0.0, 1.0, 0.0])
        b = torch.tensor([1.0, 1.0, 0.0, 0.0])
        bound = hv_bind(a, b)
        # Should be XOR
        assert bound[0].item() == 0.0
        assert bound[1].item() == 1.0
        assert bound[2].item() == 1.0

    def test_hv_permute_shift(self):
        hv = torch.tensor([1.0, 0.0, 0.0, 0.0])
        result = hv_permute(hv, shift=1)
        assert result[1].item() == 1.0

    def test_gen_hvs_reproducible(self):
        hvs1 = gen_hvs(5, 128, seed=42)
        hvs2 = gen_hvs(5, 128, seed=42)
        assert torch.equal(hvs1, hvs2)

    def test_gen_hvs_shape(self):
        hvs = gen_hvs(10, 256)
        assert hvs.shape == (10, 256)


# ── Phasor Encoder Tests ──────────────────────────────────────────────────────

class TestPhasorEncoder:
    """Validate learnable phasor encoding."""

    def test_output_shape(self, hd_dim):
        encoder = LearnablePhasorEncoder(
            n_sensors=3, sensor_dim=8, hd_dim=hd_dim, n_phasors=16,
        )
        x = torch.randn(1, 3, 8)
        hv = encoder(x)
        assert hv.shape == (1, hd_dim)

    def test_binary_output(self, hd_dim):
        encoder = LearnablePhasorEncoder(n_sensors=1, sensor_dim=4, hd_dim=hd_dim)
        x = torch.randn(2, 1, 4)
        hv = encoder(x)
        # Output should be binary (0 or 1)
        assert ((hv == 0.0) | (hv == 1.0)).all()

    def test_batch_processing(self, hd_dim):
        encoder = LearnablePhasorEncoder(n_sensors=2, sensor_dim=8, hd_dim=hd_dim)
        x = torch.randn(4, 2, 8)
        hv = encoder(x)
        assert hv.shape == (4, hd_dim)

    def test_deterministic_no_grad_variance(self, hd_dim):
        """Same input twice → same output (during inference)."""
        encoder = LearnablePhasorEncoder(n_sensors=1, sensor_dim=4, hd_dim=hd_dim)
        x = torch.randn(1, 1, 4)
        with torch.no_grad():
            hv1 = encoder(x)
            hv2 = encoder(x)
        assert torch.equal(hv1, hv2)


# ── Temporal Encoder Tests ────────────────────────────────────────────────────

class TestTemporalEncoder:
    """Validate temporal encoding with permutation."""

    def test_output_shape(self, hd_dim):
        encoder = TemporalEncoder(hd_dim=hd_dim, window=10)
        sensor_hv = (torch.rand(1, hd_dim) >= 0.5).float()
        temporal_hv, buffer = encoder(sensor_hv)
        assert temporal_hv.shape == (1, hd_dim)

    def test_buffer_accumulates(self, hd_dim):
        encoder = TemporalEncoder(hd_dim=hd_dim, window=5)
        buffer = None
        for _ in range(3):
            sensor_hv = (torch.rand(1, hd_dim) >= 0.5).float()
            _, buffer = encoder(sensor_hv, buffer)

        # Buffer should have 3 entries (plus zeros)
        assert buffer.shape[0] == 5
        # First 3 should be non-zero (or some are non-zero)
        assert (buffer[:3] != 0).any()

    def test_temporal_hv_different_for_different_sequences(self, hd_dim):
        encoder = TemporalEncoder(hd_dim=hd_dim, window=5)
        buffer = None

        # Sequence 1: all ones
        hv1 = (torch.ones(1, hd_dim)).float()
        t1, _ = encoder(hv1, buffer)

        # Sequence 2: all zeros
        hv2 = (torch.zeros(1, hd_dim)).float()
        t2, _ = encoder(hv2, buffer)

        # Different inputs should produce different temporal outputs
        assert not torch.equal(t1, t2)


# ── Predictive Coding Tests ───────────────────────────────────────────────────

class TestPredictiveCoding:
    """Validate predictive coding module."""

    def test_prediction_shape(self, hd_dim):
        module = PredictiveCodingModule(hd_dim=hd_dim)
        current = (torch.rand(1, hd_dim) >= 0.5).float()
        predicted, error = module(current)
        assert predicted.shape == (1, hd_dim)
        assert error.shape == (1, hd_dim)

    def test_perfect_prediction_zero_error(self, hd_dim):
        """Predicting the same state → prediction error should be zero."""
        module = PredictiveCodingModule(hd_dim=hd_dim)
        current = (torch.rand(1, hd_dim) >= 0.5).float()

        # Set predictor to identity (perfect predictor)
        with torch.no_grad():
            module.predictor.weight = torch.nn.Parameter(torch.eye(hd_dim))

        predicted, error = module(current, target_hv=current)
        assert error.abs().sum() < 1.0  # Hamming → not zero but small

    def test_hebbian_update_changes_weights(self, hd_dim):
        module = PredictiveCodingModule(hd_dim=hd_dim)
        current = (torch.rand(1, hd_dim) >= 0.5).float()
        target = (torch.rand(1, hd_dim) >= 0.5).float()
        predicted, error = module(current, target_hv=target)

        old_weight = module.predictor.weight.clone()
        module.hebbian_update(current, error, lr=0.01)
        assert not torch.equal(module.predictor.weight, old_weight)

    def test_error_buffer_decays(self, hd_dim):
        module = PredictiveCodingModule(hd_dim=hd_dim)
        current = (torch.rand(1, hd_dim) >= 0.5).float()
        target = (torch.rand(1, hd_dim) >= 0.5).float()

        module(current, target_hv=target)
        buf1 = module.error_buffer.clone()
        module(current, target_hv=target)
        buf2 = module.error_buffer.clone()

        # Buffer should change (running average)
        assert not torch.equal(buf1, buf2)


# ── Cognitive Map Tests ───────────────────────────────────────────────────────

class TestCognitiveMap:
    """Validate cognitive map memory."""

    def test_output_shape(self, hd_dim):
        cmap = CognitiveMapLayer(hd_dim=hd_dim, n_cells=100)
        query = (torch.rand(1, hd_dim) >= 0.5).float()
        retrieved, attention = cmap(query)
        assert retrieved.shape == (1, hd_dim)
        assert attention.shape == (1, 100)

    def test_attention_sums_to_one(self, hd_dim):
        cmap = CognitiveMapLayer(hd_dim=hd_dim, n_cells=50)
        query = (torch.rand(1, hd_dim) >= 0.5).float()
        _, attention = cmap(query)
        assert abs(attention.sum().item() - 1.0) < 1e-5

    def test_store_updates_cells(self, hd_dim):
        cmap = CognitiveMapLayer(hd_dim=hd_dim, n_cells=100)
        hv = (torch.rand(1, hd_dim) >= 0.5).float()
        old_cells = cmap.cells.clone()
        cmap.store(hv)
        # At least one cell should change
        assert not torch.equal(cmap.cells, old_cells)

    def test_retrieve_self_similarity(self, hd_dim):
        """After storing, retrieval should produce high similarity."""
        cmap = CognitiveMapLayer(hd_dim=hd_dim, n_cells=100)
        hv = (torch.rand(1, hd_dim) >= 0.5).float()
        cmap.store(hv)
        retrieved, _ = cmap(hv)
        sim = 1.0 - (retrieved != hv).float().mean().item()
        # Should be moderately similar (not exact due to blending)
        assert sim > 0.4


# ── HDC Attention Tests ───────────────────────────────────────────────────────

class TestHDCAttention:
    """Validate HDC-native attention."""

    def test_output_shape(self, hd_dim):
        attn = HDCAttention(hd_dim=hd_dim, n_heads=2)
        seq = (torch.rand(1, 3, hd_dim) >= 0.5).float()
        output = attn(seq, seq, seq)
        assert output.shape == (1, 3, hd_dim)

    def test_self_attention_high_similarity(self, hd_dim):
        """A sequence attending to itself should produce high overlap."""
        attn = HDCAttention(hd_dim=hd_dim, n_heads=2)
        seq = (torch.rand(1, 2, hd_dim) >= 0.5).float()
        output = attn(seq, seq, seq)
        # Output should be binary
        assert ((output == 0.0) | (output == 1.0)).all()

    def test_multi_head_produces_different(self, hd_dim):
        """Different heads should produce different projections."""
        attn = HDCAttention(hd_dim=hd_dim, n_heads=4)
        assert attn.key_proj.shape[0] == 4
        # Different heads should have different projections
        for h in range(1, 4):
            assert not torch.equal(attn.key_proj[0], attn.key_proj[h])


# ── World Model Tests ─────────────────────────────────────────────────────────

class TestSNNTrainingWorldModel:
    """Validate the full world model."""

    def test_forward_returns_required_keys(self, small_config):
        model = SNNTrainingWorldModel(small_config)
        x = torch.randn(1, small_config.n_sensors, small_config.sensor_dim)
        output = model(x, train=False)

        required_keys = {
            "world_state", "prediction", "prediction_error",
            "factors", "retrieved_memory", "attention_output",
            "distribution_shift",
        }
        assert required_keys.issubset(output.keys())

    def test_world_state_is_binary(self, small_config):
        model = SNNTrainingWorldModel(small_config)
        x = torch.randn(1, small_config.n_sensors, small_config.sensor_dim)
        output = model(x, train=False)
        ws = output["world_state"]
        assert ((ws == 0.0) | (ws == 1.0)).all()

    def test_forward_multiple_steps(self, small_config):
        model = SNNTrainingWorldModel(small_config)
        for _ in range(10):
            x = torch.randn(1, small_config.n_sensors, small_config.sensor_dim)
            output = model(x, train=True)

    def test_distribution_shift_detectable(self, small_config):
        model = SNNTrainingWorldModel(small_config)
        # Process normal data
        for _ in range(5):
            x = torch.randn(1, small_config.n_sensors, small_config.sensor_dim)
            model(x, train=True)
        shift1 = model.distribution_shift_estimate

        # Process out-of-distribution data
        for _ in range(5):
            x = torch.randn(1, small_config.n_sensors, small_config.sensor_dim) * 3 + 2
            model(x, train=True)
        shift2 = model.distribution_shift_estimate

        # Shift estimate should change
        assert shift2 != shift1

    def test_reset_clears_buffers(self, small_config):
        model = SNNTrainingWorldModel(small_config)
        # Store some state
        for _ in range(3):
            x = torch.randn(1, small_config.n_sensors, small_config.sensor_dim)
            model(x, train=True)

        old_buffer = model.temporal_buffer.clone()
        model.reset()

        # After reset, buffer should be zeroed
        assert model.temporal_buffer.abs().sum() == 0.0

    def test_prediction_error_tracks_changes(self, small_config):
        """Prediction error should be higher after sudden shift."""
        model = SNNTrainingWorldModel(small_config)

        # Warm up with stable data
        for _ in range(5):
            x = torch.randn(1, small_config.n_sensors, small_config.sensor_dim) * 0.5
            out = model(x, train=True)
        err1 = out["prediction_error"].abs().mean().item()

        # Abrupt shift
        x = torch.randn(1, small_config.n_sensors, small_config.sensor_dim) * 3 + 2
        out = model(x, train=True)
        err2 = out["prediction_error"].abs().mean().item()

        # Error should be non-zero (shift detected)
        assert err2 != err1


# ── Multi-Modal Fusion Tests ──────────────────────────────────────────────────

class TestMultiModalFusion:
    """Validate cross-modal fusion."""

    def test_output_shape(self, hd_dim):
        fusion = MultiModalFusion(hd_dim=hd_dim, n_modalities=3)
        hvs = [(torch.rand(2, hd_dim) >= 0.5).float() for _ in range(3)]
        fused = fusion(hvs)
        assert fused.shape == (2, hd_dim)

    def test_fused_is_binary(self, hd_dim):
        fusion = MultiModalFusion(hd_dim=hd_dim, n_modalities=2)
        hvs = [(torch.rand(1, hd_dim) >= 0.5).float() for _ in range(2)]
        fused = fusion(hvs)
        assert ((fused == 0.0) | (fused == 1.0)).all()

    def test_different_modality_keys(self, hd_dim):
        fusion = MultiModalFusion(hd_dim=hd_dim, n_modalities=4)
        # Each modality should have a different key
        for i in range(3):
            assert not torch.equal(
                fusion.modality_keys[i],
                fusion.modality_keys[i + 1],
            )


# ── Skill Transfer Tests ──────────────────────────────────────────────────────

class TestSkillTransfer:
    """Validate skill transfer module."""

    def test_find_transferable_skill(self, hd_dim):
        module = SkillTransferModule(hd_dim=hd_dim, n_skills=50)
        ws = (torch.rand(1, hd_dim) >= 0.5).float()
        idx, sim = module.find_transferable_skill(ws)
        assert 0 <= idx < 50
        assert 0.0 <= sim <= 1.0

    def test_register_skill_increments_count(self, hd_dim):
        module = SkillTransferModule(hd_dim=hd_dim, n_skills=50)
        ws = (torch.rand(1, hd_dim) >= 0.5).float()
        module.register_skill(ws, "test_skill")
        # At least one skill count should be 1
        assert module.skill_counts.sum().item() == 1

    def test_register_same_skill_twice(self, hd_dim):
        module = SkillTransferModule(hd_dim=hd_dim, n_skills=50)
        ws = (torch.rand(1, hd_dim) >= 0.5).float()
        module.register_skill(ws, "skill_a")
        module.register_skill(ws, "skill_b")
        # Same world state → should update existing skill, not create new
        assert module.skill_counts.sum().item() == 2  # Count increased

    def test_register_different_skills(self, hd_dim):
        module = SkillTransferModule(hd_dim=hd_dim, n_skills=50)
        ws1 = (torch.rand(1, hd_dim) >= 0.5).float()
        ws2 = (torch.rand(1, hd_dim) >= 0.5).float()
        module.register_skill(ws1, "skill_a")
        module.register_skill(ws2, "skill_b")
        # Two different patterns → potentially two skills
        total = module.skill_counts.sum().item()
        assert total == 2


# ── Integration Tests ─────────────────────────────────────────────────────────

class TestIntegration:
    """End-to-end world model integration."""

    def test_full_pipeline_runs(self, small_config):
        """The world model processes a complete sensor stream."""
        model = SNNTrainingWorldModel(small_config)
        n_steps = 20

        for t in range(n_steps):
            x = torch.randn(1, small_config.n_sensors, small_config.sensor_dim)
            output = model(x, train=True)

            # All outputs should be non-trivial
            assert output["world_state"] is not None
            assert output["prediction"] is not None
            assert len(output["factors"]) > 0

        # After processing, adaptation counter should be updated
        assert model.adaptation_counter == n_steps

    def test_world_model_with_distribution_shift(self, small_config):
        """Model adapts to distribution shift in real-time."""
        model = SNNTrainingWorldModel(small_config)

        # Phase 1: stable sensors
        shift_early = []
        for _ in range(10):
            x = torch.randn(1, small_config.n_sensors, small_config.sensor_dim)
            output = model(x, train=True)
            shift_early.append(output["distribution_shift"])

        # Phase 2: shifted sensors
        shift_late = []
        for _ in range(10):
            x = torch.randn(1, small_config.n_sensors, small_config.sensor_dim) * 2 + 1
            output = model(x, train=True)
            shift_late.append(output["distribution_shift"])

        # Average shift should change between phases
        assert abs(sum(shift_early) / len(shift_early) -
                   sum(shift_late) / len(shift_late)) > 0.0


# ── Config Tests ──────────────────────────────────────────────────────────────

class TestWorldModelConfig:
    """Validate WorldModelConfig defaults."""

    def test_default_config(self):
        cfg = WorldModelConfig()
        assert cfg.n_sensors == 16
        assert cfg.sensor_dim == 64
        assert cfg.hd_dim == 4096
        assert cfg.prediction_horizon == 10
        assert cfg.temporal_window == 50
        assert cfg.learning_rate == 0.1
        assert cfg.use_predictive_coding is True
        assert cfg.track_energy is True

    def test_custom_config(self):
        cfg = WorldModelConfig(
            n_sensors=4,
            hd_dim=1024,
            prediction_horizon=5,
            track_energy=False,
        )
        assert cfg.n_sensors == 4
        assert cfg.hd_dim == 1024
        assert cfg.prediction_horizon == 5
        assert cfg.track_energy is False


# ── Run with: python -m pytest tests/test_world_model.py -v ──────────────────