"""
test_hypervector_architecture.py
=================================
Tests for HyperVector Architecture (hdc/hypervector_architecture.py).

Validates:
  1. HVModel wrapping — any callable → binary hypervector
  2. HVComposer composition strategies (bundle, bind, weighted)
  3. HVPipeline: encode, predict, train_step, add/remove at runtime
  4. Graceful degradation when a modality is missing
  5. Energy tracking
  6. AutoencoderBridge encode/decode cycle
"""

from __future__ import annotations

import sys
import os
import pytest
import torch
import torch.nn as nn
from typing import Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from hdc.hypervector_architecture import (
    HVModel,
    HVModelConfig,
    HVComposer,
    HVComposerConfig,
    HVPipeline,
    AutoencoderBridge,
    BridgeConfig,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_linear_model(in_dim: int, out_dim: int) -> nn.Module:
    return nn.Linear(in_dim, out_dim, bias=False)


def make_hv_model(in_dim: int = 64, out_dim: int = 32,
                  hv_dim: int = 256, role: str = "test") -> HVModel:
    base = make_linear_model(in_dim, out_dim)
    cfg = HVModelConfig(hv_dim=hv_dim, model_output_dim=out_dim,
                        bridge_hidden=64, bridge_layers=1, role_name=role)
    return HVModel(base, config=cfg)


# ── HVModel ───────────────────────────────────────────────────────────────────

class TestHVModel:
    def test_output_shape(self):
        hvm = make_hv_model(in_dim=64, out_dim=32, hv_dim=256)
        x = torch.randn(4, 64)
        hv = hvm(x)
        assert hv.shape == (4, 256)

    def test_output_is_binary(self):
        hvm = make_hv_model(in_dim=64, out_dim=32, hv_dim=256)
        x = torch.randn(4, 64)
        hv = hvm(x)
        unique = hv.unique()
        assert set(unique.tolist()).issubset({0.0, 1.0})

    def test_wraps_plain_callable(self):
        """Non-nn.Module callables should also work."""
        def simple_fn(x):
            return x[:, :32]  # just slice

        hvm = HVModel(simple_fn,
                      config=HVModelConfig(hv_dim=128, model_output_dim=32,
                                           bridge_hidden=32, bridge_layers=1))
        x = torch.randn(2, 64)
        hv = hvm(x)
        assert hv.shape == (2, 128)

    def test_bypass_bridge(self):
        """With bypass_bridge=True, model output is binarized directly."""
        model = nn.Linear(256, 256, bias=False)
        hvm = HVModel(model,
                      config=HVModelConfig(hv_dim=256, model_output_dim=256),
                      bypass_bridge=True)
        x = torch.randn(3, 256)
        hv = hvm(x)
        assert hv.shape == (3, 256)
        unique = hv.unique()
        assert set(unique.tolist()).issubset({0.0, 1.0})

    def test_role_name_stored(self):
        hvm = make_hv_model(role="vision")
        assert hvm.role_name == "vision"

    def test_energy_tracking(self):
        hvm = make_hv_model(hv_dim=256)
        assert hvm.energy_pJ == 0.0
        hvm(torch.randn(2, 64))
        # 2 * 256 bits * 0.1 pJ = 51.2 pJ
        assert hvm.energy_pJ > 0.0

    def test_forward_deterministic_given_same_input(self):
        hvm = make_hv_model(hv_dim=128)
        x = torch.randn(3, 64)
        hv1 = hvm(x)
        hv2 = hvm(x)
        assert torch.equal(hv1, hv2)


# ── HVComposer ────────────────────────────────────────────────────────────────

class TestHVComposer:
    @pytest.fixture
    def two_models(self):
        return [
            make_hv_model(hv_dim=128, role="a"),
            make_hv_model(hv_dim=128, role="b"),
        ]

    @pytest.fixture
    def bundle_composer(self, two_models):
        cfg = HVComposerConfig(hv_dim=128, strategy="bundle", n_classes=4)
        return HVComposer(two_models, config=cfg)

    @pytest.fixture
    def bind_composer(self, two_models):
        cfg = HVComposerConfig(hv_dim=128, strategy="bind", n_classes=4)
        return HVComposer(two_models, config=cfg)

    def test_compose_output_shape(self, bundle_composer):
        # compose always returns (D,) — batches are averaged/squeezed internally
        hvs = [torch.rand(2, 128).round() for _ in range(2)]
        out = bundle_composer.compose(hvs)
        assert out.shape == (128,)

    def test_bundle_output_binary(self, bundle_composer):
        hvs = [(torch.rand(2, 128) > 0.5).float() for _ in range(2)]
        out = bundle_composer.compose(hvs)
        assert set(out.unique().tolist()).issubset({0.0, 1.0})

    def test_bind_output_shape(self, bind_composer):
        hvs = [(torch.rand(2, 128) > 0.5).float() for _ in range(2)]
        out = bind_composer.compose(hvs)
        assert out.shape == (128,)

    def test_forward_produces_joint_hv(self, bundle_composer, two_models):
        inputs = [torch.randn(2, 64) for _ in two_models]
        out = bundle_composer(inputs)
        assert out.shape == (128,)

    def test_single_hv_compose(self):
        """Compose with a single 1D HV returns that HV as-is."""
        m = make_hv_model(hv_dim=128, role="solo")
        cfg = HVComposerConfig(hv_dim=128, strategy="bundle", n_classes=4)
        composer = HVComposer([m], config=cfg)
        hv = (torch.rand(128) > 0.5).float()  # 1D
        out = composer.compose([hv])
        assert out.shape == (128,)

    def test_add_model_increases_count(self, bundle_composer):
        initial_count = len(bundle_composer.hv_models)
        new_model = make_hv_model(hv_dim=128, role="new")
        bundle_composer.add_model(new_model)
        assert len(bundle_composer.hv_models) == initial_count + 1

    def test_remove_model_decreases_count(self, bundle_composer):
        initial_count = len(bundle_composer.hv_models)
        bundle_composer.remove_model("a")
        assert len(bundle_composer.hv_models) == initial_count - 1

    def test_remove_nonexistent_model_returns_false(self, bundle_composer):
        assert bundle_composer.remove_model("nonexistent") is False


# ── HVPipeline ────────────────────────────────────────────────────────────────

class TestHVPipeline:
    @pytest.fixture
    def pipeline(self):
        models = {
            "vision": make_hv_model(in_dim=64, out_dim=32, hv_dim=256, role="vision"),
            "audio": make_hv_model(in_dim=48, out_dim=16, hv_dim=256, role="audio"),
        }
        return HVPipeline(models=models, n_classes=5, hv_dim=256, strategy="bundle")

    @pytest.fixture
    def sample_inputs(self):
        return {
            "vision": torch.randn(1, 64),
            "audio": torch.randn(1, 48),
        }

    def test_encode_output_shape(self, pipeline, sample_inputs):
        hv = pipeline.encode(sample_inputs)
        assert hv.shape == (256,)

    def test_encode_binary(self, pipeline, sample_inputs):
        hv = pipeline.encode(sample_inputs)
        assert set(hv.unique().tolist()).issubset({0.0, 1.0})

    def test_predict_returns_valid_class(self, pipeline, sample_inputs):
        # Train first to get non-trivial prototypes
        for cls in range(5):
            inp = {k: torch.randn(1, v.shape[1])
                   for k, v in [("vision", torch.empty(1, 64)),
                                 ("audio", torch.empty(1, 48))]}
            inp = {"vision": torch.randn(1, 64), "audio": torch.randn(1, 48)}
            pipeline.train_step(inp, cls)

        hv = pipeline.encode(sample_inputs)
        pred, logits = pipeline.predict(hv)
        assert 0 <= pred < 5
        assert logits.shape[0] == 5

    def test_train_step_no_error(self, pipeline, sample_inputs):
        # Should not raise
        pipeline.train_step(sample_inputs, label=2)

    def test_n_models(self, pipeline):
        assert pipeline.n_models == 2

    def test_roles(self, pipeline):
        assert set(pipeline.roles) == {"vision", "audio"}

    def test_add_model_at_runtime(self, pipeline):
        new_model = make_hv_model(in_dim=32, out_dim=16, hv_dim=256, role="sensor")
        pipeline.add_model("sensor", new_model)
        assert pipeline.n_models == 3
        assert "sensor" in pipeline.roles

    def test_add_then_encode_with_new_modality(self, pipeline, sample_inputs):
        new_model = make_hv_model(in_dim=32, out_dim=16, hv_dim=256, role="sensor")
        pipeline.add_model("sensor", new_model)
        inputs_extended = {**sample_inputs, "sensor": torch.randn(1, 32)}
        hv = pipeline.encode(inputs_extended)
        assert hv.shape == (256,)

    def test_remove_model_at_runtime(self, pipeline):
        removed = pipeline.remove_model("audio")
        assert removed is True
        assert pipeline.n_models == 1
        assert "audio" not in pipeline.roles

    def test_encode_with_missing_modality(self, pipeline):
        """Pipeline should handle missing modalities gracefully."""
        partial_inputs = {"vision": torch.randn(1, 64)}  # no "audio"
        hv = pipeline.encode(partial_inputs)
        assert hv.shape == (256,)

    def test_remove_nonexistent_model(self, pipeline):
        assert pipeline.remove_model("nonexistent") is False


# ── AutoencoderBridge ─────────────────────────────────────────────────────────

class TestAutoencoderBridge:
    @pytest.fixture
    def bridge(self):
        cfg = BridgeConfig(input_dim=64, hdc_dim=256, hidden_dim=128,
                           encoding_layers=2, device="cpu")
        return AutoencoderBridge(cfg)

    def test_encode_output_shape(self, bridge):
        x = torch.randn(4, 64)
        hv = bridge.encode(x)
        assert hv.shape == (4, 256)

    def test_encode_binary(self, bridge):
        x = torch.randn(4, 64)
        hv = bridge.encode(x)
        assert set(hv.unique().tolist()).issubset({0.0, 1.0})

    def test_decode_output_shape(self, bridge):
        hv = (torch.rand(4, 256) > 0.5).float()
        recon = bridge.decode(hv)
        assert recon.shape == (4, 64)

    def test_roundtrip_compresses_then_reconstructs(self, bridge):
        """Encode then decode returns something in the right domain."""
        x = torch.randn(4, 64)
        hv = bridge.encode(x)
        recon = bridge.decode(hv)
        assert recon.shape == x.shape


# ── Runtime Composition (Integration) ────────────────────────────────────────

class TestRuntimeComposition:
    """Multi-step: build pipeline, train, add model, re-predict."""

    def test_pipeline_workflow(self):
        hv_dim = 128
        n_classes = 3
        torch.manual_seed(42)

        models = {
            "a": make_hv_model(in_dim=32, out_dim=16, hv_dim=hv_dim, role="a"),
            "b": make_hv_model(in_dim=32, out_dim=16, hv_dim=hv_dim, role="b"),
        }
        pipe = HVPipeline(models=models, n_classes=n_classes,
                           hv_dim=hv_dim, strategy="bundle")

        # Train on some data
        for cls in range(n_classes):
            for _ in range(5):
                inp = {"a": torch.randn(1, 32), "b": torch.randn(1, 32)}
                pipe.train_step(inp, cls)

        # Predict before adding new model
        test_inp = {"a": torch.randn(1, 32), "b": torch.randn(1, 32)}
        hv1 = pipe.encode(test_inp)
        pred1, _ = pipe.predict(hv1)
        assert 0 <= pred1 < n_classes

        # Add a third model at runtime — no retraining
        pipe.add_model("c", make_hv_model(in_dim=16, out_dim=8, hv_dim=hv_dim, role="c"))

        # Pipeline still works with two modalities (c missing)
        hv2 = pipe.encode(test_inp)
        pred2, _ = pipe.predict(hv2)
        assert 0 <= pred2 < n_classes

        # Remove model "b" — graceful degradation
        pipe.remove_model("b")
        assert "b" not in pipe.roles
        assert pipe.n_models == 2

        # Should still encode with remaining models
        inp_partial = {"a": torch.randn(1, 32)}
        hv3 = pipe.encode(inp_partial)
        assert hv3.shape == (hv_dim,)
