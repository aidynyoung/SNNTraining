"""
tests/test_causal_discovery.py
================================
Tests for HDCCausalVariable, CausalSignatureGraph, OnlineCausalDiscovery.
"""
import pytest
import torch
from hdc.causal_discovery import (
    HDCCausalVariable,
    CausalSignatureGraph,
    OnlineCausalDiscovery,
    _gen_hv,
)

D = 256


def _hv(seed):
    return _gen_hv(D, seed=seed)


# ── HDCCausalVariable ─────────────────────────────────────────────────────────

class TestHDCCausalVariable:
    def test_has_data_false_initially(self):
        var = HDCCausalVariable("X", D)
        assert not var.has_data

    def test_has_data_after_two_obs(self):
        var = HDCCausalVariable("X", D)
        var.observe(_hv(0))
        assert not var.has_data  # needs 2 obs
        var.observe(_hv(1))
        assert var.has_data

    def test_transition_hv_shape(self):
        var = HDCCausalVariable("X", D)
        for i in range(5):
            var.observe(_hv(i))
        assert var.transition_hv.shape == (D,)

    def test_transition_hv_changes_with_obs(self):
        var = HDCCausalVariable("X", D)
        for i in range(3):
            var.observe(_hv(i))
        t0 = var.transition_hv.clone()
        # Feed many diverse observations so the EMA shifts enough to differ
        for i in range(50):
            var.observe(_hv(100 + i))
        assert not torch.equal(t0, var.transition_hv)

    def test_n_obs_increments(self):
        var = HDCCausalVariable("X", D)
        for i in range(7):
            var.observe(_hv(i))
        assert var._n_obs == 7

    def test_reset(self):
        var = HDCCausalVariable("X", D)
        for i in range(5):
            var.observe(_hv(i))
        var.reset()
        assert var._n_obs == 0
        assert not var.has_data


# ── CausalSignatureGraph ──────────────────────────────────────────────────────

class TestCausalSignatureGraph:
    def setup_method(self):
        self.g = CausalSignatureGraph(D, min_obs=5, threshold=0.001)
        self.g.register_variable("X")
        self.g.register_variable("Y")
        self.g.register_variable("W")

    def _feed_observations(self, n=30):
        base_X = _hv(10)
        base_Y = _hv(20)
        base_W = _hv(30)
        from hdc.physics_world_model import _xor, _majority
        for t in range(n):
            x_hv = _majority(base_X.float() + 0.05 * _hv(t).float())
            y_hv = _majority((0.8 * _xor(x_hv, base_Y) + 0.2 * base_Y).float())
            w_hv = base_W.clone()
            self.g.observe("X", x_hv)
            self.g.observe("Y", y_hv)
            self.g.observe("W", w_hv)

    def test_register_variable(self):
        assert "X" in self.g._variables
        assert "Y" in self.g._variables

    def test_observe_updates_variable(self):
        self.g.observe("X", _hv(0))
        self.g.observe("X", _hv(1))
        assert self.g._variables["X"].has_data

    def test_observe_all(self):
        obs = {"X": _hv(0), "Y": _hv(1), "W": _hv(2)}
        self.g.observe_all(obs)
        self.g.observe_all(obs)
        assert self.g._variables["X"].has_data

    def test_causal_score_returns_float(self):
        self._feed_observations()
        score = self.g.causal_score("X", "Y")
        assert isinstance(score, float)

    def test_causal_score_range(self):
        self._feed_observations()
        score = self.g.causal_score("X", "Y")
        assert -1.0 <= score <= 1.0

    def test_discover_edges_returns_list(self):
        self._feed_observations()
        edges = self.g.discover_edges()
        assert isinstance(edges, list)

    def test_causal_graph_returns_dict(self):
        self._feed_observations()
        graph = self.g.causal_graph()
        assert isinstance(graph, dict)
        assert "X" in graph

    def test_causal_parents_returns_list(self):
        self._feed_observations()
        parents = self.g.causal_parents("Y")
        assert isinstance(parents, list)

    def test_add_known_pair_builds_proto(self):
        self._feed_observations(30)
        self.g.add_known_pair("X", "Y")
        assert self.g._causal_proto is not None

    def test_intervention_hv_shape(self):
        self._feed_observations()
        iv = self.g.intervention_hv("X", "Y")
        if iv is not None:
            assert iv.shape == (D,)


# ── OnlineCausalDiscovery ─────────────────────────────────────────────────────

class TestOnlineCausalDiscovery:
    def setup_method(self):
        self.cd = OnlineCausalDiscovery(
            D,
            variables=["X", "Y", "Z", "W"],
            min_obs=5,
            threshold=0.001,
        )

    def _run(self, n=30):
        from hdc.physics_world_model import _xor, _majority
        base = {v: _hv(i) for i, v in enumerate(["X", "Y", "Z", "W"])}
        for t in range(n):
            obs = {}
            obs["X"] = _majority(base["X"].float() + 0.05 * _hv(t).float())
            obs["Y"] = _majority((0.8 * _xor(obs["X"], base["Y"]) + 0.2 * base["Y"]).float())
            obs["Z"] = _majority((0.7 * _xor(obs["Y"], base["Z"]) + 0.3 * base["Z"]).float())
            obs["W"] = base["W"].clone()
            self.cd.step(obs)

    def test_n_variables(self):
        assert self.cd.n_variables == 4

    def test_step_increments_timesteps(self):
        self.cd.step({"X": _hv(0), "Y": _hv(1), "Z": _hv(2), "W": _hv(3)})
        assert self.cd._n_timesteps == 1

    def test_causal_summary_returns_dict(self):
        self._run()
        summary = self.cd.causal_summary()
        assert isinstance(summary, dict)

    def test_what_causes_returns_list(self):
        self._run()
        parents = self.cd.what_causes("Y")
        assert isinstance(parents, list)

    def test_what_does_returns_list(self):
        self._run()
        effects = self.cd.what_does("X")
        assert isinstance(effects, list)

    def test_stability_in_range(self):
        self._run()
        stab = self.cd.stability()
        assert 0.0 <= stab <= 1.0

    def test_causal_chain_returns_path_or_none(self):
        self._run()
        chain = self.cd.causal_chain("X", "Z", max_depth=4)
        assert chain is None or isinstance(chain, list)

    def test_register_known_cause_no_error(self):
        self._run()
        self.cd.register_known_cause("X", "Y")  # should not raise
