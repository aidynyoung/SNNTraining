"""
Unit tests for the dual-timescale Hebbian accumulator.

Covers:
  - Fast and slow trace decay correctness
  - Weight-update sign convention
  - batch_update matches step-by-step update
  - End-to-end decoding on tiny synthetic data
"""

import math
import pytest
import torch
from models.hebbian import DualHebbian, DualHebbianAccumulator, HebbianConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_hebb():
    cfg = HebbianConfig(shape=(4, 4), tau_fast=5.0, tau_slow=50.0,
                        alpha=0.7, beta=0.3)
    return DualHebbian(cfg)


# ---------------------------------------------------------------------------
# 1. Eligibility-trace decay
# ---------------------------------------------------------------------------

class TestDecay:
    def test_fast_trace_decays_without_spikes(self, small_hebb):
        """After seeding a trace, zero-spike steps should decay it geometrically."""
        hebb = small_hebb
        pre  = torch.ones(4)
        post = torch.ones(4)
        hebb.update(pre, post)          # seed the trace

        decay_fast = hebb.decay_fast
        initial = hebb.e_fast.clone()

        hebb.update(torch.zeros(4), torch.zeros(4))   # no spikes
        expected = initial * decay_fast
        assert torch.allclose(hebb.e_fast, expected, atol=1e-6)

    def test_slow_trace_decays_slower_than_fast(self, small_hebb):
        """tau_slow > tau_fast → slow trace retains more than fast trace."""
        hebb = small_hebb
        pre = post = torch.ones(4)
        hebb.update(pre, post)

        # Many zero-spike steps
        for _ in range(20):
            hebb.update(torch.zeros(4), torch.zeros(4))

        # After 20 zero steps, slow should be larger than fast
        assert hebb.e_slow.abs().mean() > hebb.e_fast.abs().mean()

    def test_decay_constants_match_tau(self):
        """Decay factors must equal 1 - 1/tau (analytic formula used in code)."""
        cfg = HebbianConfig(shape=(2, 2), tau_fast=10.0, tau_slow=100.0)
        h = DualHebbian(cfg)
        assert abs(h.decay_fast - (1.0 - 1.0 / 10.0)) < 1e-7
        assert abs(h.decay_slow - (1.0 - 1.0 / 100.0)) < 1e-7


# ---------------------------------------------------------------------------
# 2. Weight-update sign
# ---------------------------------------------------------------------------

class TestUpdateSign:
    def test_correlated_spikes_produce_positive_trace(self, small_hebb):
        """When pre and post both fire, E should be positive."""
        hebb = small_hebb
        E = hebb.update(torch.ones(4), torch.ones(4))
        assert E.min() >= 0.0, "Correlated spikes must yield non-negative E"

    def test_uncorrelated_gives_zero(self, small_hebb):
        """If pre fires but post does not, outer product is zero; only decay."""
        hebb = small_hebb
        E = hebb.update(torch.ones(4), torch.zeros(4))
        # Fresh traces, so E should be zero (outer product = 0, traces start at 0)
        assert torch.allclose(E, torch.zeros(4, 4), atol=1e-7)

    def test_combined_trace_is_weighted_sum(self, small_hebb):
        """E = alpha * e_fast + beta * e_slow."""
        hebb = small_hebb
        pre = post = torch.eye(4)[0]   # only neuron 0 fires
        E = hebb.update(pre, post)
        expected = hebb.cfg.alpha * hebb.e_fast + hebb.cfg.beta * hebb.e_slow
        assert torch.allclose(E, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# 3. batch_update matches step-by-step
# ---------------------------------------------------------------------------

class TestBatchUpdate:
    def test_batch_matches_sequential(self):
        """batch_update over T steps must equal T sequential update() calls."""
        T = 10
        cfg = HebbianConfig(shape=(8, 8), tau_fast=5.0, tau_slow=50.0,
                            alpha=0.7, beta=0.3)

        # Two identical-state Hebbian modules
        hebb_seq   = DualHebbian(cfg)
        hebb_batch = DualHebbian(cfg)

        pre_seq  = torch.randint(0, 2, (T, 8)).float()
        post_seq = torch.randint(0, 2, (T, 8)).float()

        # Sequential
        for t in range(T):
            E_seq = hebb_seq.update(pre_seq[t], post_seq[t])

        # Batch
        E_batch = hebb_batch.batch_update(pre_seq, post_seq)

        assert torch.allclose(E_seq, E_batch, atol=1e-4), \
            f"max diff = {(E_seq - E_batch).abs().max():.6f}"

    def test_batch_update_shapes(self):
        cfg = HebbianConfig(shape=(6, 4))
        hebb = DualHebbian(cfg)
        T = 5
        E = hebb.batch_update(
            torch.randint(0, 2, (T, 4)).float(),
            torch.randint(0, 2, (T, 6)).float(),
        )
        assert E.shape == (6, 4)


# ---------------------------------------------------------------------------
# 4. DualHebbianAccumulator alias
# ---------------------------------------------------------------------------

def test_accumulator_alias():
    cfg = HebbianConfig(shape=(4, 4))
    acc = DualHebbianAccumulator(cfg)
    E = acc.update(torch.ones(4), torch.ones(4))
    assert E.shape == (4, 4)
    acc.reset()
    assert acc._impl.e_fast.sum() == 0.0


# ---------------------------------------------------------------------------
# 5. End-to-end: tiny synthetic decoding
# ---------------------------------------------------------------------------

def test_e2e_decoding_improves():
    """
    After 200 steps on a trivial linear stream, MSE should drop.
    Not a tight bound — just confirms the loop runs and learning occurs.
    """
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from models.rsnn import RSNN
    from models.readout import Readout

    torch.manual_seed(42)
    rsnn    = RSNN(input_size=20, hidden_size=64)
    hebb    = DualHebbian(HebbianConfig(shape=(64, 64)))
    readout = Readout(hidden_size=64, output_size=1)

    lr_out = 1e-2
    mse_early, mse_late = [], []

    for step in range(500):
        # Binary input; target = fraction of active neurons in first half
        x = torch.randint(0, 2, (20,)).float()
        target = torch.tensor([x[:10].mean().item()])  # in [0, 1]

        spikes  = rsnn.forward(x)
        pred    = readout.forward(spikes)
        error   = pred - target

        with torch.no_grad():
            readout.W -= lr_out * torch.outer(error, spikes)

        mse = float((error ** 2).mean().item())
        if 50 <= step < 150:
            mse_early.append(mse)
        if step >= 400:
            mse_late.append(mse)

    avg_early = sum(mse_early) / max(len(mse_early), 1)
    avg_late  = sum(mse_late)  / max(len(mse_late), 1)
    assert avg_late < avg_early, \
        f"MSE did not decrease: early={avg_early:.4f}, late={avg_late:.4f}"
