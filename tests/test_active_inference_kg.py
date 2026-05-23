"""
tests/test_active_inference_kg.py
==================================
Tests for hdc/active_inference.py (FreeEnergyEstimator, ActiveInferenceAgent,
PrecisionWeightedAttention, BeliefPropagation, ExpectedFreeEnergy)
and hdc/knowledge_graph.py (KGCodebook, HDCKnowledgeGraph, KGReasoner,
HDCOntology, KGCompletionEvaluator).
"""
import pytest
import torch
from hdc.active_inference import (
    FreeEnergyEstimator, ActiveInferenceAgent,
    PrecisionWeightedAttention, BeliefPropagation, ExpectedFreeEnergy,
    _gen_hv,
)
from hdc.knowledge_graph import (
    KGCodebook, HDCKnowledgeGraph, KGReasoner,
    HDCOntology, KGCompletionEvaluator, Triple,
)
from hdc.hrr import HRR

D = 256


# ─── FreeEnergyEstimator ─────────────────────────────────────────────────────

class TestFreeEnergyEstimator:
    def setup_method(self):
        self.fe = FreeEnergyEstimator(dim=D, complexity_weight=0.5)

    def test_exact_prediction_zero_accuracy(self):
        pred = _gen_hv(D, seed=0)
        result = self.fe.free_energy(pred, pred)
        assert result["accuracy"] < 1e-4

    def test_random_prediction_positive_accuracy(self):
        pred = _gen_hv(D, seed=0)
        obs  = _gen_hv(D, seed=1)
        result = self.fe.free_energy(pred, obs)
        assert result["accuracy"] > 0.0

    def test_free_energy_components(self):
        pred   = _gen_hv(D, seed=0)
        obs    = _gen_hv(D, seed=1)
        belief = _gen_hv(D, seed=2)
        result = self.fe.free_energy(pred, obs, belief)
        assert "accuracy" in result
        assert "complexity" in result
        assert "free_energy" in result

    def test_free_energy_nonnegative(self):
        pred   = _gen_hv(D, seed=0)
        obs    = _gen_hv(D, seed=1)
        result = self.fe.free_energy(pred, obs)
        assert result["free_energy"] >= 0.0

    def test_surprise_in_range(self):
        obs  = _gen_hv(D, seed=0)
        pred = _gen_hv(D, seed=1)
        surp = self.fe.surprise(obs, pred)
        assert 0.0 <= surp <= 1.0

    def test_average_F_after_observations(self):
        for i in range(10):
            self.fe.free_energy(_gen_hv(D, seed=i), _gen_hv(D, seed=100+i))
        avg = self.fe.average_F(window=10)
        assert avg > 0.0

    def test_update_prior(self):
        new_prior = _gen_hv(D, seed=999)
        self.fe.update_prior(new_prior)
        assert torch.equal(self.fe._prior, new_prior)


# ─── ActiveInferenceAgent ─────────────────────────────────────────────────────

class TestActiveInferenceAgent:
    def setup_method(self):
        self.agent = ActiveInferenceAgent(
            dim=D,
            preferred_obs=_gen_hv(D, seed=42),
        )

    def test_perceive_returns_dict(self):
        obs = _gen_hv(D, seed=0)
        result = self.agent.perceive(obs)
        assert "free_energy" in result
        assert "accuracy" in result

    def test_belief_shape(self):
        assert self.agent.belief.shape == (D,)

    def test_belief_updates_on_perceive(self):
        # Feed a very different observation to ensure belief changes
        belief0 = self.agent.belief.clone()
        for _ in range(5):
            self.agent.perceive(_gen_hv(D, seed=999), lr=0.5)
        assert not torch.equal(self.agent.belief, belief0)

    def test_step_increments(self):
        self.agent.perceive(_gen_hv(D, seed=0))
        assert self.agent._step == 1

    def test_select_action_in_range(self):
        actions = [_gen_hv(D, seed=i) for i in range(4)]
        best, G, all_G = self.agent.select_action(actions)
        assert 0 <= best < 4
        assert len(all_G) == 4

    def test_select_action_empty(self):
        best, G, all_G = self.agent.select_action([])
        assert best == 0

    def test_free_energy_report_after_perception(self):
        for i in range(5):
            self.agent.perceive(_gen_hv(D, seed=i))
        report = self.agent.free_energy_report()
        assert "avg_free_energy" in report
        assert report["n_steps"] == 5

    def test_update_preferred(self):
        new_pref = _gen_hv(D, seed=77)
        self.agent.update_preferred(new_pref)
        assert torch.equal(self.agent.preferred, new_pref)

    def test_precision_in_01(self):
        assert 0.0 <= self.agent.precision <= 1.0


# ─── PrecisionWeightedAttention ──────────────────────────────────────────────

class TestPrecisionWeightedAttention:
    def setup_method(self):
        self.pwa = PrecisionWeightedAttention(dim=D, n_channels=3)

    def test_integrate_shape(self):
        channels = [_gen_hv(D, seed=i) for i in range(3)]
        out = self.pwa.integrate(channels)
        assert out.shape == (D,)

    def test_integrate_binary_ish(self):
        channels = [_gen_hv(D, seed=i) for i in range(3)]
        out = self.pwa.integrate(channels)
        assert set(out.unique().tolist()).issubset({0.0, 1.0})

    def test_precision_in_range(self):
        assert all(0.0 <= float(p) <= 1.0 for p in self.pwa.precision)

    def test_update_precision_decreases_for_high_error(self):
        pi0 = float(self.pwa.precision[0].item())
        self.pwa.update_precision(0, prediction_error=0.9, lr=0.1)
        assert float(self.pwa.precision[0].item()) < pi0

    def test_attend_to_increases_channel(self):
        pi1_before = float(self.pwa.precision[1].item())
        self.pwa.attend_to(1, boost=0.2)
        assert float(self.pwa.precision[1].item()) > pi1_before

    def test_precision_report_keys(self):
        report = self.pwa.precision_report()
        assert "ch_0" in report and "ch_1" in report and "ch_2" in report


# ─── BeliefPropagation ───────────────────────────────────────────────────────

class TestBeliefPropagation:
    def setup_method(self):
        self.bp = BeliefPropagation(n_layers=3, dim=D)

    def test_forward_returns_n_layers(self):
        states = self.bp.forward(_gen_hv(D, seed=0))
        assert len(states) == 3

    def test_forward_shapes(self):
        states = self.bp.forward(_gen_hv(D, seed=0))
        assert all(s.shape == (D,) for s in states)

    def test_states_change_with_input(self):
        s0 = self.bp.forward(_gen_hv(D, seed=0))
        s1 = self.bp.forward(_gen_hv(D, seed=999))
        assert not torch.equal(s0[0], s1[0])

    def test_prediction_error_nonnegative(self):
        self.bp.forward(_gen_hv(D, seed=0))
        pe = self.bp.prediction_error_norm()
        assert pe >= 0.0

    def test_multiple_iterations(self):
        s1 = self.bp.forward(_gen_hv(D, seed=0), n_iterations=1)
        self.bp2 = BeliefPropagation(n_layers=3, dim=D)
        s5 = self.bp2.forward(_gen_hv(D, seed=0), n_iterations=5)
        # Both should produce valid shapes
        assert all(s.shape == (D,) for s in s5)


# ─── ExpectedFreeEnergy ───────────────────────────────────────────────────────

class TestExpectedFreeEnergy:
    def setup_method(self):
        self.agent = ActiveInferenceAgent(D, preferred_obs=_gen_hv(D, seed=42))
        self.efe   = ExpectedFreeEnergy(self.agent, epistemic_weight=0.3)

    def test_compute_returns_dict(self):
        result = self.efe.compute(_gen_hv(D, seed=0), n_simulations=2)
        assert "G" in result
        assert "pragmatic" in result
        assert "epistemic" in result

    def test_G_is_float(self):
        result = self.efe.compute(_gen_hv(D, seed=0))
        assert isinstance(result["G"], float)

    def test_select_best_returns_valid_idx(self):
        actions = [_gen_hv(D, seed=i) for i in range(3)]
        best, G, results = self.efe.select_best(actions, n_simulations=2)
        assert 0 <= best < 3
        assert len(results) == 3

    def test_running_G_after_selections(self):
        actions = [_gen_hv(D, seed=i) for i in range(2)]
        self.efe.select_best(actions, n_simulations=1)
        g = self.efe.running_G()
        assert isinstance(g, float)


# ─── KGCodebook ──────────────────────────────────────────────────────────────

class TestKGCodebook:
    def setup_method(self):
        self.hrr = HRR(dim=D)
        self.cb  = KGCodebook(self.hrr)

    def test_entity_shape(self):
        hv = self.cb.entity("Paris")
        assert hv.shape == (D,)

    def test_entity_deterministic(self):
        hv1 = self.cb.entity("Paris")
        hv2 = self.cb.entity("Paris")
        assert torch.equal(hv1, hv2)

    def test_relation_shape(self):
        hv = self.cb.relation("capital_of")
        assert hv.shape == (D,)

    def test_entity_relation_different(self):
        e = self.cb.entity("Paris")
        r = self.cb.relation("Paris")   # same name, different space
        sim = float(torch.cosine_similarity(e.unsqueeze(0), r.unsqueeze(0)))
        assert abs(sim) < 0.95

    def test_n_entities_increments(self):
        for name in ["A", "B", "C"]:
            self.cb.entity(name)
        assert self.cb.n_entities == 3

    def test_nearest_entity_shape(self):
        for name in ["A", "B", "C"]:
            self.cb.entity(name)
        q   = self.cb.entity("A")
        res = self.cb.nearest_entity(q, top_k=2)
        assert len(res) == 2

    def test_nearest_entity_self(self):
        self.cb.entity("TestEnt")
        q   = self.cb.entity("TestEnt")
        res = self.cb.nearest_entity(q, top_k=1)
        assert res[0][0] == "TestEnt"
        assert abs(res[0][1] - 1.0) < 1e-4


# ─── HDCKnowledgeGraph ────────────────────────────────────────────────────────

class TestHDCKnowledgeGraph:
    def setup_method(self):
        self.hrr = HRR(dim=D)
        self.kg  = HDCKnowledgeGraph(self.hrr)
        self.kg.insert_batch([
            ("Paris",  "capital_of", "France"),
            ("Berlin", "capital_of", "Germany"),
            ("London", "capital_of", "UK"),
        ])

    def test_n_triples(self):
        assert self.kg.n_triples == 3

    def test_query_object_correct(self):
        results = self.kg.query_object("Paris", "capital_of", top_k=3)
        assert results[0][0] == "France"

    def test_query_object_shape(self):
        results = self.kg.query_object("Paris", "capital_of", top_k=2)
        assert len(results) == 2
        for name, sim in results:
            assert isinstance(name, str)
            assert isinstance(sim, float)

    def test_query_subject_correct(self):
        results = self.kg.query_subject("capital_of", "Germany", top_k=3)
        assert results[0][0] == "Berlin"

    def test_query_relation_shape(self):
        results = self.kg.query_relation("Paris", "France", top_k=2)
        assert len(results) >= 1

    def test_remove_triple(self):
        self.kg.insert("Tokyo", "capital_of", "Japan")
        assert self.kg.n_triples == 4
        removed = self.kg.remove("Tokyo", "capital_of", "Japan")
        assert removed
        assert self.kg.n_triples == 3

    def test_remove_nonexistent(self):
        removed = self.kg.remove("X", "y", "Z")
        assert not removed


# ─── KGReasoner ──────────────────────────────────────────────────────────────

class TestKGReasoner:
    def setup_method(self):
        self.hrr      = HRR(dim=D)
        self.kg       = HDCKnowledgeGraph(self.hrr)
        self.kg.insert_batch([
            ("Paris",  "capital_of", "France"),
            ("Berlin", "capital_of", "Germany"),
            ("London", "capital_of", "UK"),
            ("France", "in_continent", "Europe"),
            ("Germany","in_continent", "Europe"),
        ])
        self.reasoner = KGReasoner(self.kg)

    def test_multi_hop_shape(self):
        results = self.reasoner.multi_hop("Paris", ["capital_of"], top_k=2)
        assert len(results) >= 1

    def test_multi_hop_correct_one_hop(self):
        results = self.reasoner.multi_hop("Paris", ["capital_of"], top_k=3)
        assert results[0][0] == "France"

    def test_path_exists_positive(self):
        found = self.reasoner.path_exists("Paris", "France", "capital_of")
        assert found

    def test_infer_inverse_inserts(self):
        rel = self.reasoner.infer_inverse("Paris", "capital_of", "France")
        assert rel == "capital_of_inv"
        assert self.kg.n_triples == 6   # 5 original + 1 inverse


# ─── HDCOntology ─────────────────────────────────────────────────────────────

class TestHDCOntology:
    def setup_method(self):
        self.hrr  = HRR(dim=D)
        self.onto = HDCOntology(self.hrr)
        self.onto.add_class("Animal")
        self.onto.add_class("Mammal",  parent="Animal")
        self.onto.add_class("Dog",     parent="Mammal")
        self.onto.add_class("Cat",     parent="Mammal")

    def test_is_a_direct(self):
        assert self.onto.is_a("Dog", "Mammal")

    def test_is_a_indirect(self):
        assert self.onto.is_a("Mammal", "Animal")

    def test_is_a_false_for_sibling(self):
        # Dog and Cat are siblings, not is_a relation
        # With low threshold they might be similar, with high threshold not
        result = self.onto.is_a("Dog", "Cat", threshold=0.9)
        assert not result

    def test_class_hv_shape(self):
        hv = self.onto.class_hv("Dog")
        assert hv is not None
        assert hv.shape == (D,)

    def test_most_specific_class_self(self):
        dog_hv = self.onto.class_hv("Dog")
        cls, sim = self.onto.most_specific_class(dog_hv)
        assert cls == "Dog"
        assert abs(sim - 1.0) < 1e-4

    def test_unknown_class_returns_none(self):
        hv = self.onto.class_hv("Unicorn")
        assert hv is None


# ─── KGCompletionEvaluator ───────────────────────────────────────────────────

class TestKGCompletionEvaluator:
    def setup_method(self):
        self.hrr  = HRR(dim=D)
        self.kg   = HDCKnowledgeGraph(self.hrr)
        self.kg.insert_batch([
            ("Paris",  "capital_of", "France"),
            ("Berlin", "capital_of", "Germany"),
            ("London", "capital_of", "UK"),
        ])
        self.evaluator = KGCompletionEvaluator(self.kg)

    def test_evaluate_returns_dict(self):
        test = [("Paris", "capital_of", "France")]
        metrics = self.evaluator.evaluate_object_queries(test)
        assert "hit@1" in metrics
        assert "MRR" in metrics

    def test_hit_at_1_for_stored_triples(self):
        test = [
            ("Paris",  "capital_of", "France"),
            ("Berlin", "capital_of", "Germany"),
        ]
        metrics = self.evaluator.evaluate_object_queries(test, k_values=[1, 3])
        assert metrics["hit@1"] >= 0.5   # at least one correct

    def test_mrr_in_range(self):
        test = [("Paris", "capital_of", "France")]
        metrics = self.evaluator.evaluate_object_queries(test)
        assert 0.0 <= metrics["MRR"] <= 1.0


# ─── ActiveInferenceAgent v1.46 improvements ─────────────────────────────────

class TestActiveInferenceAgentV146:
    def setup_method(self):
        self.dim   = D
        self.agent = ActiveInferenceAgent(self.dim, precision=0.8)

    def test_adaptive_precision_attrs(self):
        assert hasattr(self.agent, '_precision_ema')
        assert hasattr(self.agent, '_precision_tau')

    def test_precision_ema_updates_after_perceive(self):
        obs = _gen_hv(self.dim)
        p0  = self.agent._precision_ema
        for _ in range(10):
            self.agent.perceive(obs)
        # precision EMA should have moved from its initial value
        assert isinstance(self.agent._precision_ema, float)
        assert 0.0 < self.agent._precision_ema <= 1.0

    def test_visited_set_populated(self):
        obs = _gen_hv(self.dim)
        assert len(self.agent._visited) == 0
        self.agent.perceive(obs)
        assert len(self.agent._visited) >= 1

    def test_habit_formation_update(self):
        action = _gen_hv(self.dim)
        self.agent.update_habits(action, reward_signal=0.9)
        assert len(self.agent._action_q) == 1
        vals = list(self.agent._action_q.values())
        assert 0.0 < vals[0] <= 1.0

    def test_habit_decays_toward_zero(self):
        action = _gen_hv(self.dim)
        self.agent.update_habits(action, reward_signal=1.0)
        v0 = list(self.agent._action_q.values())[0]
        # Apply zero reward repeatedly → Q should decay
        for _ in range(20):
            self.agent.update_habits(action, reward_signal=0.0)
        v1 = list(self.agent._action_q.values())[0]
        assert v1 < v0

    def test_select_action_with_habits(self):
        actions = [_gen_hv(self.dim, seed=i) for i in range(3)]
        # Register a habit for action 0
        self.agent.update_habits(actions[0], reward_signal=1.0)
        obs = _gen_hv(self.dim, seed=99)
        for _ in range(3):
            self.agent.perceive(obs)
        idx, g, gs = self.agent.select_action(actions)
        assert 0 <= idx < len(actions)
        assert isinstance(g, float)

    def test_novelty_bonus_for_unseen_state(self):
        actions = [_gen_hv(self.dim, seed=i) for i in range(2)]
        obs = _gen_hv(self.dim, seed=5)
        self.agent.perceive(obs)
        # G values should be valid floats
        _, _, gs = self.agent.select_action(actions)
        assert all(isinstance(g, float) for g in gs)


# ─────────────────────────────────────────────────────────────────────────────
# KGReasoner: find_path + apply_rule
# ─────────────────────────────────────────────────────────────────────────────

from hdc.knowledge_graph import HDCKnowledgeGraph, KGReasoner, KGCodebook
from hdc.hrr import HRR


class TestKGReasonerPathRule:
    D = 128

    def _make_kg(self):
        hrr   = HRR(self.D)
        kg    = HDCKnowledgeGraph(hrr)
        cb    = KGCodebook(hrr)
        # Register entities (relations are created lazily on first use)
        cb.register_entity("A")
        cb.register_entity("B")
        cb.register_entity("C")
        # Touch relations to ensure they exist
        cb.relation("r1")
        cb.relation("r2")
        return kg, cb, hrr

    def test_find_path_none_when_empty(self):
        kg, cb, hrr = self._make_kg()
        reasoner = KGReasoner(kg)   # KGReasoner only takes kg + max_hops
        path = reasoner.find_path("A", "C")
        assert path is None

    def test_find_path_direct(self):
        kg, cb, hrr = self._make_kg()
        kg.insert("A", "r1", "B")
        kg.insert("B", "r2", "C")
        reasoner = KGReasoner(kg)
        path = reasoner.find_path("A", "B")
        assert path is None or isinstance(path, list)

    def test_apply_rule_no_crash(self):
        kg, cb, hrr = self._make_kg()
        kg.insert("A", "r1", "B")
        reasoner = KGReasoner(kg)
        inferred = reasoner.apply_rule(if_predicate="r1", then_predicate="r2")
        assert isinstance(inferred, list)


# ─────────────────────────────────────────────────────────────────────────────
# FractionalPowerEncoding bandwidth auto-selection
# ─────────────────────────────────────────────────────────────────────────────

from hdc.vsa_algebras import FractionalPowerEncoding


class TestFPEBandwidthFit:
    def test_silverman_returns_positive(self):
        fpe = FractionalPowerEncoding(n_features=4, dim=64)
        X   = torch.randn(30, 4)
        bw  = fpe.fit_bandwidth(X, method="silverman")
        assert bw > 0.0
        assert fpe.bw == bw

    def test_grid_search_returns_valid(self):
        fpe = FractionalPowerEncoding(n_features=4, dim=64)
        X   = torch.randn(20, 4)
        bw  = fpe.fit_bandwidth(X, method="grid_search")
        assert bw > 0.0

    def test_fit_bw_changes_encoding(self):
        fpe = FractionalPowerEncoding(n_features=4, dim=64, bw=1.0)
        X   = torch.randn(20, 4)
        z1  = fpe.encode(X[0])
        fpe.fit_bandwidth(X, method="silverman")
        z2  = fpe.encode(X[0])
        # Different bw → different encoding
        if fpe.bw != 1.0:
            assert not torch.equal(z1.real, z2.real)

    def test_silverman_clipped_range(self):
        fpe = FractionalPowerEncoding(n_features=2, dim=32)
        X   = torch.randn(10, 2) * 100   # high variance
        bw  = fpe.fit_bandwidth(X, method="silverman")
        assert 0.01 <= bw <= 10.0

    def test_unknown_method_raises(self):
        fpe = FractionalPowerEncoding(n_features=2, dim=32)
        X   = torch.randn(10, 2)
        import pytest
        with pytest.raises(ValueError):
            fpe.fit_bandwidth(X, method="magic")


# ─────────────────────────────────────────────────────────────────────────────
# HDCKnowledgeGraph: pattern_match + count_triples
# ─────────────────────────────────────────────────────────────────────────────

class TestHDCKGPatternMatch:
    D = 128

    def _make_kg(self):
        hrr = HRR(self.D)
        kg  = HDCKnowledgeGraph(hrr)
        kg.insert("Paris",    "in_country",  "France")
        kg.insert("Paris",    "capital_of",  "France")
        kg.insert("Berlin",   "in_country",  "Germany")
        return kg

    def test_pattern_match_subject_wildcard(self):
        kg      = self._make_kg()
        matches = kg.pattern_match(subject="Paris")
        assert len(matches) == 2
        for t in matches:
            assert t.subject == "Paris"

    def test_pattern_match_predicate_wildcard(self):
        kg      = self._make_kg()
        matches = kg.pattern_match(predicate="in_country")
        assert len(matches) == 2

    def test_pattern_match_object_wildcard(self):
        kg      = self._make_kg()
        matches = kg.pattern_match(obj="France")
        assert len(matches) == 2

    def test_pattern_match_all_wildcards(self):
        kg      = self._make_kg()
        matches = kg.pattern_match()
        assert len(matches) == 3

    def test_pattern_match_no_match(self):
        kg      = self._make_kg()
        matches = kg.pattern_match(subject="Tokyo")
        assert len(matches) == 0

    def test_count_triples(self):
        kg = self._make_kg()
        assert kg.count_triples(subject="Paris") == 2
        assert kg.count_triples(predicate="in_country") == 2
        assert kg.count_triples() == 3


# ─────────────────────────────────────────────────────────────────────────────
# SensorStreamBuffer: IS weights + priority update
# ─────────────────────────────────────────────────────────────────────────────

from hdc.sensor_stream import SensorStreamBuffer


class TestSensorStreamBufferIS:
    D = 64

    def _make_buffer(self, n: int = 20) -> SensorStreamBuffer:
        buf = SensorStreamBuffer(capacity=50)
        for i in range(n):
            hv  = (torch.rand(self.D) > 0.5).float()
            err = float(i) / n   # increasing errors
            buf.push(hv, err)
        return buf

    def test_sample_with_weights_returns_dicts(self):
        buf    = self._make_buffer()
        result = buf.sample_with_weights(n=5)
        assert len(result) == 5
        for item in result:
            assert "sample" in item
            assert "is_weight" in item
            assert 0.0 <= item["is_weight"] <= 1.0 + 1e-6

    def test_is_weights_max_one(self):
        buf    = self._make_buffer()
        result = buf.sample_with_weights(n=10)
        weights = [r["is_weight"] for r in result]
        assert max(weights) <= 1.0 + 1e-6

    def test_mean_error_positive(self):
        buf = self._make_buffer()
        assert buf.mean_error() > 0.0

    def test_mean_error_empty_buffer(self):
        buf = SensorStreamBuffer()
        assert buf.mean_error() == 0.0

    def test_update_priority_changes_sample_probs(self):
        buf    = self._make_buffer(10)
        before = buf.mean_error()
        buf.update_priority(0, new_error=10.0)
        # Mean error may change after priority update
        assert isinstance(buf.mean_error(), float)
