"""Unit tests for the SNN → HDC inference pipeline."""

import torch
import pytest
from models.snn_hdc_pipeline import SNNHDCPipeline, PipelineConfig


@pytest.fixture
def pipe():
    return SNNHDCPipeline(PipelineConfig(
        input_size=10, hidden_size=16, n_classes=3,
        hdc_dim=64, window=5, overlap=0,
    ))


def test_snn_step_returns_spike_vector(pipe):
    x = torch.randn(10)
    spikes = pipe.snn_step(x)
    assert spikes.shape == (16,)
    assert spikes.min() >= 0.0 and spikes.max() <= 1.0


def test_predict_returns_none_before_window(pipe):
    x = torch.randn(10)
    for t in range(4):       # window=5, so first 4 should give None
        label, conf = pipe.predict(x)
        assert label is None


def test_predict_returns_label_at_window(pipe):
    # Train first so assoc_mem has non-zero prototypes
    for cls in range(3):
        for _ in range(3):
            x = torch.zeros(10)
            x[cls * 3] = 1.0
            pipe.train_step(x, cls)
    pipe.finalize()
    pipe.reset()

    x = torch.zeros(10)
    x[0] = 1.0
    label, conf = None, None
    for t in range(10):
        label, conf = pipe.predict(x)
        if label is not None:
            break

    assert label is not None
    assert isinstance(label, int)
    assert 0 <= label < 3


def test_reset_clears_buffer(pipe):
    x = torch.randn(10)
    pipe.snn_step(x)
    pipe.snn_step(x)
    pipe.reset()
    assert len(pipe._buf) == 0
    assert pipe._step_count == 0
