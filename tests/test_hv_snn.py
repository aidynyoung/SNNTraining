"""
tests/test_hv_snn.py
=====================
Tests for models/hv_snn.py — SpikingHVLayer and SpikingHVNetwork.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import torch
from models.hv_snn import (
    SpikingHVLayer, SpikingHVConfig,
    SpikingHVNetwork, SpikingHVNetworkConfig,
    _bundle_active, _hamming_sim_vec,
)

torch.manual_seed(42)
N, D = 32, 128


# ── Helpers ────────────────────────────────────────────────────────────────────

def _layer() -> SpikingHVLayer:
    return SpikingHVLayer(SpikingHVConfig(n_neurons=N, hv_dim=D, seed=0))

def _network() -> SpikingHVNetwork:
    return SpikingHVNetwork(SpikingHVNetworkConfig(
        input_size=16, n_neurons=N, hv_dim=D, seed=0
    ))


# ── _bundle_active ─────────────────────────────────────────────────────────────

def test_bundle_active_no_spikes():
    basis = torch.randint(0, 2, (N, D)).float()
    mask = torch.zeros(N, dtype=torch.bool)
    result = _bundle_active(basis, mask)
    assert result.shape == (D,)
    assert result.sum() == 0.0

def test_bundle_active_all_spikes():
    basis = torch.randint(0, 2, (N, D)).float()
    mask = torch.ones(N, dtype=torch.bool)
    result = _bundle_active(basis, mask)
    assert result.shape == (D,)
    assert set(result.unique().tolist()).issubset({0.0, 1.0})

def test_bundle_active_single_spike():
    basis = torch.eye(D)[:N]   # orthogonal basis
    mask = torch.zeros(N, dtype=torch.bool)
    mask[0] = True
    result = _bundle_active(basis, mask)
    # Single active neuron → result should equal its basis HV
    assert torch.equal(result, (basis[0] > 0.5).float())


# ── _hamming_sim_vec ───────────────────────────────────────────────────────────

def test_hamming_sim_identical():
    hv = torch.randint(0, 2, (D,)).float()
    matrix = hv.unsqueeze(0).expand(4, -1).clone()
    sims = _hamming_sim_vec(hv, matrix)
    assert sims.shape == (4,)
    assert (sims > 0.99).all()

def test_hamming_sim_orthogonal():
    hv = torch.zeros(D)
    hv[:D//2] = 1.0
    other = torch.zeros(D)
    other[D//2:] = 1.0
    sims = _hamming_sim_vec(hv, other.unsqueeze(0))
    # Complement → similarity should be 0
    assert float(sims[0]) == pytest.approx(0.0, abs=0.01)

def test_hamming_sim_range():
    hv = torch.randint(0, 2, (D,)).float()
    matrix = torch.randint(0, 2, (8, D)).float()
    sims = _hamming_sim_vec(hv, matrix)
    assert (sims >= 0.0).all() and (sims <= 1.0).all()


# ── SpikingHVLayer ─────────────────────────────────────────────────────────────

def test_layer_output_shapes():
    layer = _layer()
    current = torch.randn(N) * 3.0
    spikes, state_hv, seq_hv = layer.step(current)
    assert spikes.shape == (N,)
    assert state_hv.shape == (D,)
    assert seq_hv.shape == (D,)

def test_layer_spikes_binary():
    layer = _layer()
    spikes, _, _ = layer.step(torch.randn(N) * 3.0)
    assert set(spikes.unique().tolist()).issubset({0.0, 1.0})

def test_layer_state_hv_binary():
    layer = _layer()
    _, state_hv, _ = layer.step(torch.randn(N) * 3.0)
    assert set(state_hv.unique().tolist()).issubset({0.0, 1.0})

def test_layer_seq_hv_evolves():
    layer = _layer()
    seq_hvs = []
    # Use very large current to guarantee spikes, then alternating zero
    # so the seq_hv is driven then permuted — must produce different values
    for i in range(6):
        current = torch.ones(N) * 10.0 if i % 2 == 0 else torch.zeros(N)
        _, _, seq_hv = layer.step(current)
        seq_hvs.append(seq_hv.clone())
    # After at least one spike step, seq_hv must be non-zero
    assert any(s.sum() > 0 for s in seq_hvs), "seq_hv never populated"
    # And not all identical
    assert not all(torch.equal(seq_hvs[0], s) for s in seq_hvs[1:])

def test_layer_reset_clears_seq():
    layer = _layer()
    for _ in range(10):
        layer.step(torch.randn(N) * 3.0)
    layer.reset()
    assert layer._seq_hv.sum() == 0.0
    assert layer._step == 0

def test_layer_basis_shape():
    layer = _layer()
    assert layer.basis.shape == (N, D)
    assert set(layer.basis.flatten().unique().tolist()).issubset({0.0, 1.0})

def test_layer_state_dim_property():
    layer = _layer()
    assert layer.state_dim == D


# ── SpikingHVNetwork ───────────────────────────────────────────────────────────

def test_network_step_output_shapes():
    net = _network()
    spikes, state_hv, seq_hv = net.step(torch.randn(16))
    assert spikes.shape == (N,)
    assert state_hv.shape == (D,)
    assert seq_hv.shape == (D,)

def test_network_run_sequence():
    net = _network()
    seq = torch.randint(0, 2, (50, 16)).float()
    state_hv, seq_hv = net.run_sequence(seq)
    assert state_hv.shape == (D,)
    assert seq_hv.shape == (D,)

def test_network_no_w_rec():
    """SpikingHVNetwork must not have a W_rec attribute."""
    net = _network()
    assert not hasattr(net, "W_rec")

def test_network_only_win_parameters():
    """n_parameters counts W_in elements; no W_rec exists."""
    net = _network()
    # W_in is a fixed buffer (reservoir paradigm), not an nn.Parameter
    assert not hasattr(net, "W_rec"), "SpikingHVNetwork must not have W_rec"
    # n_parameters reports W_in size = N × input_size
    assert net.n_parameters == net.cfg.n_neurons * net.cfg.input_size

def test_network_reset_clears_state():
    net = _network()
    seq = torch.randint(0, 2, (20, 16)).float()
    net.run_sequence(seq)
    net.reset()
    assert net._state_hv.sum() == 0.0
    assert net._seq_hv.sum() == 0.0

def test_network_as_hv_model_callable():
    net = _network()
    fn = net.as_hv_model()
    seq = torch.randint(0, 2, (20, 16)).float()
    out = fn(seq)
    assert out.shape == (1, D)
    assert set(out.flatten().unique().tolist()).issubset({0.0, 1.0})

def test_network_integrates_with_hv_pipeline():
    from hdc.hypervector_architecture import HVModel, HVModelConfig, HVPipeline
    net = _network()
    hv_snn = HVModel(
        net.as_hv_model(),
        HVModelConfig(hv_dim=D, model_output_dim=D, role_name="spike"),
        bypass_bridge=True,
    )
    pipe = HVPipeline({"spike": hv_snn}, n_classes=3, hv_dim=D)
    seq = torch.randint(0, 2, (20, 16)).float()
    joint = pipe.encode({"spike": seq})
    assert joint.shape == (D,)
