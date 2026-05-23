"""
tests/test_hva.py
=================
Tests for the HyperVector Architecture (HVA) — hdc/hypervector_architecture.py

Covers: HVModel, HVPrototypeHead, HVComposer, HVPipeline, HVScaler.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import torch
from hdc.hypervector_architecture import (
    HVModel, HVModelConfig,
    HVPrototypeHead,
    HVComposer, HVComposerConfig,
    HVPipeline,
    HVScaler,
)

torch.manual_seed(42)
D, N_CLS = 256, 4


# ── Helpers ────────────────────────────────────────────────────────────────────

def _toy_model(out_dim: int):
    """Returns a simple linear callable for use as a wrapped model."""
    W = torch.randn(out_dim, 32)
    def fn(x):
        return x @ W.T
    return fn


def _make_pipe(strategy="bundle") -> HVPipeline:
    m1 = HVModel(_toy_model(64), HVModelConfig(hv_dim=D, model_output_dim=64, role_name="a"))
    m2 = HVModel(_toy_model(48), HVModelConfig(hv_dim=D, model_output_dim=48, role_name="b"))
    return HVPipeline({"a": m1, "b": m2}, n_classes=N_CLS, hv_dim=D, strategy=strategy)


# ── HVModel ────────────────────────────────────────────────────────────────────

def test_hv_model_output_shape():
    model = HVModel(_toy_model(64), HVModelConfig(hv_dim=D, model_output_dim=64))
    out = model(torch.randn(1, 32))
    assert out.shape[-1] == D

def test_hv_model_output_binary():
    model = HVModel(_toy_model(64), HVModelConfig(hv_dim=D, model_output_dim=64))
    out = model(torch.randn(1, 32))
    assert set(out.flatten().unique().tolist()).issubset({0.0, 1.0})

def test_hv_model_bypass_bridge():
    """bypass_bridge=True: callable must already produce D-dim output."""
    def passthrough(x):
        return (x > 0).float()
    model = HVModel(passthrough, HVModelConfig(hv_dim=D, model_output_dim=D), bypass_bridge=True)
    out = model(torch.randn(1, D))
    assert out.shape[-1] == D
    assert set(out.flatten().unique().tolist()).issubset({0.0, 1.0})

def test_hv_model_energy_tracked():
    model = HVModel(_toy_model(64), HVModelConfig(hv_dim=D, model_output_dim=64))
    assert model.energy_pJ == 0.0
    model(torch.randn(1, 32))
    assert model.energy_pJ > 0.0


# ── HVPrototypeHead ────────────────────────────────────────────────────────────

def test_prototype_head_predict_shape():
    head = HVPrototypeHead(n_classes=N_CLS, hv_dim=D)
    hv = torch.randint(0, 2, (D,)).float()
    pred, sims = head.predict(hv)
    assert 0 <= pred < N_CLS
    assert sims.shape == (N_CLS,)

def test_prototype_head_train_updates_counts():
    head = HVPrototypeHead(n_classes=N_CLS, hv_dim=D)
    hv = torch.randint(0, 2, (D,)).float()
    head.train_step(hv, label=0)
    assert head.counts[0].item() == 1.0

def test_prototype_head_prototypes_binary_after_train():
    head = HVPrototypeHead(n_classes=N_CLS, hv_dim=D)
    for i in range(20):
        hv = torch.randint(0, 2, (D,)).float()
        head.train_step(hv, label=i % N_CLS)
    assert set(head.prototypes.flatten().unique().tolist()).issubset({0.0, 1.0})

def test_prototype_head_handles_batch_hv():
    """Head must accept (1, D) batched HV without error."""
    head = HVPrototypeHead(n_classes=N_CLS, hv_dim=D)
    hv = torch.randint(0, 2, (1, D)).float()
    pred, sims = head.predict(hv)
    assert 0 <= pred < N_CLS

def test_prototype_head_anomaly_detection():
    head = HVPrototypeHead(n_classes=N_CLS, hv_dim=D)
    head.enable_anomaly_detection(percentile=90.0, warmup_steps=10)
    for _ in range(15):
        hv = torch.randint(0, 2, (D,)).float()
        head.train_step(hv, 0)
        score, _ = head.anomaly_score(hv)
    assert head._d2h_threshold is not None
    assert 0.0 <= score <= 1.0


# ── HVComposer ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("strategy", ["bundle", "bind", "weighted", "sequential"])
def test_composer_strategies(strategy):
    m1 = HVModel(_toy_model(64), HVModelConfig(hv_dim=D, model_output_dim=64, role_name="x"))
    m2 = HVModel(_toy_model(48), HVModelConfig(hv_dim=D, model_output_dim=48, role_name="y"))
    cfg = HVComposerConfig(hv_dim=D, strategy=strategy, n_classes=N_CLS)
    comp = HVComposer([m1, m2], config=cfg)
    hvs = [m1(torch.randn(1, 32)), m2(torch.randn(1, 32))]
    result = comp.compose(hvs)
    assert result.shape == (D,)
    assert set(result.unique().tolist()).issubset({0.0, 1.0})

def test_composer_add_remove_model():
    m1 = HVModel(_toy_model(64), HVModelConfig(hv_dim=D, model_output_dim=64, role_name="x"))
    cfg = HVComposerConfig(hv_dim=D, n_classes=N_CLS)
    comp = HVComposer([m1], config=cfg)
    assert len(comp.hv_models) == 1

    m2 = HVModel(_toy_model(32), HVModelConfig(hv_dim=D, model_output_dim=32, role_name="y"))
    comp.add_model(m2)
    assert len(comp.hv_models) == 2

    removed = comp.remove_model("x")
    assert removed
    assert len(comp.hv_models) == 1

def test_composer_energy():
    m1 = HVModel(_toy_model(64), HVModelConfig(hv_dim=D, model_output_dim=64, role_name="x"))
    cfg = HVComposerConfig(hv_dim=D, n_classes=N_CLS)
    comp = HVComposer([m1], config=cfg)
    comp.forward([torch.randn(1, 32)])
    assert comp.total_energy_pJ > 0.0


# ── HVPipeline ─────────────────────────────────────────────────────────────────

def test_pipeline_encode_shape():
    pipe = _make_pipe()
    joint = pipe.encode({"a": torch.randn(1, 32), "b": torch.randn(1, 32)})
    assert joint.shape == (D,)

def test_pipeline_predict_valid_class():
    pipe = _make_pipe()
    for i in range(20):
        pipe.train_step({"a": torch.randn(1, 32), "b": torch.randn(1, 32)}, label=i % N_CLS)
    joint = pipe.encode({"a": torch.randn(1, 32), "b": torch.randn(1, 32)})
    pred, sims = pipe.predict(joint)
    assert 0 <= pred < N_CLS

def test_pipeline_add_model_runtime():
    pipe = _make_pipe()
    assert pipe.n_models == 2
    m3 = HVModel(_toy_model(16), HVModelConfig(hv_dim=D, model_output_dim=16, role_name="c"))
    pipe.add_model("c", m3)
    assert pipe.n_models == 3
    assert "c" in pipe.roles

def test_pipeline_remove_model_graceful():
    pipe = _make_pipe()
    pipe.remove_model("a")
    assert pipe.n_models == 1
    # Still encodes with remaining model
    joint = pipe.encode({"b": torch.randn(1, 32)})
    assert joint.shape == (D,)

def test_pipeline_missing_modality_skipped():
    """Encoding with only one modality present should not raise."""
    pipe = _make_pipe()
    joint = pipe.encode({"a": torch.randn(1, 32)})  # "b" missing
    assert joint.shape == (D,)

def test_pipeline_roles_tracked():
    pipe = _make_pipe()
    assert set(pipe.roles) == {"a", "b"}


# ── HVScaler ──────────────────────────────────────────────────────────────────

def test_scaler_upgrade_level1():
    scaler = HVScaler(_toy_model(64), output_dim=64, n_classes=N_CLS, hv_dim=D)
    pipe = scaler.upgrade(role="base")
    assert scaler.level == 1
    assert pipe is not None
    assert "base" in pipe.roles

def test_scaler_add_peer_level2():
    scaler = HVScaler(_toy_model(64), output_dim=64, n_classes=N_CLS, hv_dim=D)
    scaler.upgrade(role="base")
    scaler.add_peer(_toy_model(32), output_dim=32, role="peer")
    assert scaler.level >= 2
    assert "peer" in scaler.pipeline.roles
