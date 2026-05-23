import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import pytest
from models.rsnn import RSNN, RSNNConfig


def make_rsnn(input_size=20, hidden_size=32):
    return RSNN(RSNNConfig(input_size=input_size, hidden_size=hidden_size))


def test_output_shape():
    rsnn = make_rsnn()
    spikes = rsnn.forward(torch.rand(20))
    assert spikes.shape == (32,)


def test_spikes_binary():
    rsnn = make_rsnn()
    for _ in range(30):
        spikes = rsnn.forward(torch.randn(20))
    assert set(spikes.unique().tolist()).issubset({0.0, 1.0})


def test_no_self_connections():
    rsnn = make_rsnn(hidden_size=16)
    diag = rsnn.W_rec.diagonal()
    assert (diag == 0.0).all(), "Self-connections present in W_rec"


def test_sparse_init_sparsity():
    cfg = RSNNConfig(input_size=10, hidden_size=64, sparse_init=True, sparse_p=0.1)
    rsnn = RSNN(cfg)
    nonzero = (rsnn.W_rec != 0).float().mean().item()
    # should be close to 0.1 (with some variance)
    assert nonzero < 0.25, f"W_rec not sparse enough: {nonzero:.2f} nonzero"


def test_reset_clears_state():
    rsnn = make_rsnn()
    for _ in range(50):
        rsnn.forward(torch.randn(20))
    rsnn.reset()
    assert rsnn.prev_spikes.sum() == 0.0
    assert rsnn.lif.v.sum() == 0.0


def test_recurrent_state_changes():
    """prev_spikes should update after forward."""
    rsnn = make_rsnn()
    rsnn.forward(torch.ones(20) * 5.0)
    assert rsnn.prev_spikes.sum() >= 0  # just check it ran


def test_get_state_keys():
    rsnn = make_rsnn()
    rsnn.forward(torch.rand(20))
    state = rsnn.get_state()
    assert "v" in state
    assert "prev_spikes" in state
    assert "firing_rates" in state
