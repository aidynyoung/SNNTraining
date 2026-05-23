"""Unit tests for the RFLO learning rule."""

import torch
import pytest
from training.rflo import RFLOLearner, ExactFeedbackLearner, RFLOConfig


@pytest.fixture
def cfg():
    return RFLOConfig(hidden_size=16, output_size=2, tau_eligibility=10.0,
                      learning_rate=1e-3)


def test_rflo_dw_shape(cfg):
    learner = RFLOLearner(cfg)
    u = torch.randn(16)
    spikes = (u > 0).float()
    error = torch.randn(2)
    dW = learner.step(u, spikes, error)
    assert dW.shape == (16, 16)


def test_rflo_eligibility_trace_nonzero_after_spikes(cfg):
    learner = RFLOLearner(cfg)
    u = torch.ones(16)      # all above threshold → all spike
    spikes = torch.ones(16)
    error = torch.ones(2)
    learner.step(u, spikes, error)
    # After one step with all-ones, h_prev = ones; next step should populate e
    u2 = torch.ones(16)
    learner.step(u2, torch.ones(16), error)
    assert learner.e.abs().sum() > 0


def test_rflo_reset_clears_state(cfg):
    learner = RFLOLearner(cfg)
    for _ in range(5):
        learner.step(torch.randn(16), torch.randint(0, 2, (16,)).float(),
                     torch.randn(2))
    learner.reset()
    assert learner.e.sum() == 0.0
    assert learner.h_prev.sum() == 0.0


def test_exact_feedback_uses_w_out(cfg):
    learner = ExactFeedbackLearner(cfg)
    # w_out: (output_size, hidden_size) = (2, 16)
    W_out = torch.randn(cfg.output_size, cfg.hidden_size)
    u = torch.randn(16)
    spikes = (u > 0).float()
    error = torch.randn(2)
    dW = learner.step(u, spikes, error, w_out=W_out)
    assert dW.shape == (16, 16)
