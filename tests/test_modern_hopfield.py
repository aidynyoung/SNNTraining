"""
tests/test_modern_hopfield.py
==============================
Tests for ModernHopfieldHDC, ModernHopfieldAttention,
HopfieldHDCMemoryBank, and AssociativeReasoningHopfield.
"""
import pytest
import torch
from hdc.modern_hopfield import (
    ModernHopfieldHDC,
    ModernHopfieldAttention,
    HopfieldHDCMemoryBank,
    AssociativeReasoningHopfield,
    _to_bipolar,
    _to_binary,
)

D = 256


def _hv(seed):
    g = torch.Generator(); g.manual_seed(seed)
    return (torch.rand(D, generator=g) >= 0.5).float()


# ── ModernHopfieldHDC ─────────────────────────────────────────────────────────

class TestModernHopfieldHDC:
    def test_store_and_retrieve_exact(self):
        mh = ModernHopfieldHDC(D, beta=8.0)
        p = _hv(0)
        mh.store(p, label="p0")
        retrieved, lbl, _ = mh.retrieve(p)
        sim = float((retrieved == p).float().mean())
        assert sim > 0.95, f"Should retrieve exact pattern, got sim={sim}"
        assert lbl == "p0"

    def test_store_multiple_and_retrieve(self):
        mh = ModernHopfieldHDC(D, beta=8.0)
        patterns = [_hv(i) for i in range(10)]
        for i, p in enumerate(patterns):
            mh.store(p, label=f"p{i}")
        assert mh.n_patterns == 10

        # Each pattern should be retrievable
        for i, p in enumerate(patterns):
            _, lbl, _ = mh.retrieve(p)
            assert lbl == f"p{i}"

    def test_noisy_retrieval(self):
        mh = ModernHopfieldHDC(D, beta=8.0)
        p = _hv(42)
        mh.store(p, label="clean")

        noisy = p.clone()
        flip = torch.rand(D) < 0.15  # 15% bit flip
        noisy[flip] = 1.0 - noisy[flip]

        retrieved, lbl, _ = mh.retrieve(noisy)
        sim = float((retrieved == p).float().mean())
        assert sim > 0.85, f"Should denoise 15% noise, got sim={sim}"

    def test_capacity_estimate(self):
        mh = ModernHopfieldHDC(D, beta=5.0)
        for i in range(20):
            mh.store(_hv(i))
        cap = mh.capacity_estimate()
        assert cap["n_stored"] == 20
        assert cap["practical_limit"] > cap["n_stored"]
        assert "load_fraction" in cap

    def test_nearest_k(self):
        mh = ModernHopfieldHDC(D, beta=5.0)
        for i in range(10):
            mh.store(_hv(i), label=f"p{i}")
        results = mh.nearest_k(_hv(0), k=3)
        assert len(results) == 3
        assert all(0.0 <= sim <= 1.0 for _, _, sim in results)
        # First result should be the exact match
        assert results[0][1] == "p0"

    def test_batch_retrieve(self):
        mh = ModernHopfieldHDC(D, beta=5.0)
        for i in range(5):
            mh.store(_hv(i), label=f"p{i}")
        queries = torch.stack([_hv(i) for i in range(3)])
        retrieved, labels = mh.retrieve_batch(queries)
        assert retrieved.shape == (3, D)
        assert len(labels) == 3

    def test_prune_lru(self):
        mh = ModernHopfieldHDC(D, beta=5.0)
        for i in range(20):
            mh.store(_hv(i))
        mh._access_count[0] = 100  # mark as frequently accessed
        mh.prune_lru(keep_top=10)
        assert mh.n_patterns == 10

    def test_reset(self):
        mh = ModernHopfieldHDC(D)
        for i in range(5):
            mh.store(_hv(i))
        mh.reset()
        assert mh.n_patterns == 0

    def test_utility_functions(self):
        x = torch.tensor([0.0, 1.0, 0.0, 1.0])
        bip = _to_bipolar(x)
        assert bip.tolist() == [-1.0, 1.0, -1.0, 1.0]
        back = _to_binary(bip)
        assert back.tolist() == [0.0, 1.0, 0.0, 1.0]


# ── ModernHopfieldAttention ───────────────────────────────────────────────────

class TestModernHopfieldAttention:
    def test_attend_single(self):
        attn = ModernHopfieldAttention(D, beta=5.0)
        for i in range(5):
            attn.register(_hv(i), _hv(100 + i))
        out, weights = attn.attend(_hv(2))
        assert out.shape == (D,)
        assert weights.shape == (5,)
        assert abs(float(weights.sum()) - 1.0) < 1e-4

    def test_attend_batch(self):
        attn = ModernHopfieldAttention(D, beta=5.0)
        for i in range(4):
            attn.register(_hv(i), _hv(100 + i))
        queries = torch.stack([_hv(i) for i in range(3)])
        out, weights = attn.batch_attend(queries)
        assert out.shape == (3, D)
        assert weights.shape == (3, 4)

    def test_no_keys(self):
        attn = ModernHopfieldAttention(D, beta=5.0)
        q = _hv(0)
        out, weights = attn.attend(q)
        assert out.shape == (D,)

    def test_n_registered(self):
        attn = ModernHopfieldAttention(D)
        for i in range(7):
            attn.register(_hv(i), _hv(100 + i))
        assert attn.n_registered == 7


# ── HopfieldHDCMemoryBank ─────────────────────────────────────────────────────

class TestHopfieldHDCMemoryBank:
    def test_store_within_capacity(self):
        bank = HopfieldHDCMemoryBank(D, episodic_capacity=10)
        for i in range(8):
            bank.store(_hv(i), label=f"item_{i}")
        assert bank.episodic.n_patterns == 8

    def test_evicts_at_capacity(self):
        bank = HopfieldHDCMemoryBank(D, episodic_capacity=5)
        for i in range(10):
            bank.store(_hv(i), label=f"item_{i}")
        assert bank.episodic.n_patterns <= 5

    def test_consolidate_promotes_frequent(self):
        bank = HopfieldHDCMemoryBank(D, episodic_capacity=20, consolidate_at=3)
        for i in range(5):
            bank.store(_hv(i))
        bank.episodic._access_count[0] = 5
        bank.episodic._access_count[1] = 4
        bank.consolidate()
        assert bank.semantic.n_patterns == 2

    def test_retrieve_returns_source(self):
        bank = HopfieldHDCMemoryBank(D, episodic_capacity=20)
        for i in range(5):
            bank.store(_hv(i), label=f"item_{i}")
        hv, lbl, source = bank.retrieve(_hv(2))
        assert source in ("episodic", "semantic", "empty")
        assert hv.shape == (D,)

    def test_stats(self):
        bank = HopfieldHDCMemoryBank(D)
        bank.store(_hv(0))
        stats = bank.stats()
        assert "episodic_n" in stats and "semantic_n" in stats


# ── AssociativeReasoningHopfield ──────────────────────────────────────────────

class TestAssociativeReasoningHopfield:
    def setup_method(self):
        self.ar = AssociativeReasoningHopfield(D, beta=5.0)
        for i in range(8):
            self.ar.store_transition(_hv(i), _hv(100 + i), _hv(200 + i))

    def test_n_transitions(self):
        assert self.ar.n_transitions == 8

    def test_query_outcome_shape(self):
        outcome, conf = self.ar.query_outcome(_hv(3), _hv(103))
        assert outcome.shape == (D,)
        assert 0.0 <= conf <= 1.0

    def test_query_cause_shape(self):
        (state, action), conf = self.ar.query_cause(_hv(205))
        assert state.shape == (D,)
        assert action.shape == (D,)
        assert 0.0 <= conf <= 1.0

    def test_query_action_shape(self):
        action, conf = self.ar.query_action(_hv(2), _hv(202))
        assert action.shape == (D,)
        assert 0.0 <= conf <= 1.0

    def test_high_conf_for_stored_transitions(self):
        # Query with exact stored (state, action) should give high confidence
        _, conf = self.ar.query_outcome(_hv(4), _hv(104))
        assert conf > 0.5, f"Expected high conf for stored transition, got {conf}"


# ── ModernHopfieldHDC.approximate_retrieve ───────────────────────────────────

class TestApproximateRetrieve:
    def test_returns_triple(self):
        mh = ModernHopfieldHDC(D, beta=6.0)
        for i in range(10):
            mh.store(_hv(i), str(i))
        result, lbl, sim = mh.approximate_retrieve(_hv(0), n_candidates=5)
        assert result.shape == (D,)
        assert 0.0 <= sim <= 1.0

    def test_approximate_matches_exact_small_set(self):
        mh = ModernHopfieldHDC(D, beta=6.0)
        for i in range(5):   # small set → approx = exact
            mh.store(_hv(i), str(i))
        q = _hv(0)
        exact_r, exact_l, _ = mh.retrieve(q)
        approx_r, approx_l, approx_sim = mh.approximate_retrieve(q, n_candidates=20)
        # With 5 patterns and n_candidates=20, should be the same result
        assert approx_l == exact_l

    def test_large_set_no_crash(self):
        mh = ModernHopfieldHDC(D, beta=4.0)
        for i in range(50):
            mh.store(_hv(i), str(i))
        result, lbl, sim = mh.approximate_retrieve(_hv(0), n_candidates=10)
        assert result.shape == (D,)
        assert isinstance(lbl, str)

    def test_empty_memory_returns_query(self):
        mh = ModernHopfieldHDC(D)
        q = _hv(7)
        result, lbl, sim = mh.approximate_retrieve(q)
        assert torch.equal(result, q)
        assert lbl is None


# ── EWCRegularizer task-specific consolidation ───────────────────────────────

from models.hebbian import EWCRegularizer


class TestEWCTaskSpecific:
    def test_consolidate_task_stores_data(self):
        ewc = EWCRegularizer(lambda_ewc=100.0, fisher_samples=20)
        W   = torch.randn(2, 16)
        for _ in range(20):
            ewc.accumulate(torch.rand(16), torch.rand(2))
        ewc.consolidate_task("task_A", W)
        assert ewc.n_consolidated_tasks() >= 1

    def test_multi_task_penalty_shape(self):
        ewc = EWCRegularizer(lambda_ewc=100.0, fisher_samples=10)
        W   = torch.randn(2, 16)
        for _ in range(10):
            ewc.accumulate(torch.rand(16), torch.rand(2))
        ewc.consolidate_task("task_A", W)
        ewc.consolidate_task("task_B", W + 0.1)
        grad = ewc.multi_task_penalty_grad(W)
        assert grad.shape == W.shape

    def test_multi_task_penalty_zero_at_star(self):
        ewc = EWCRegularizer(lambda_ewc=100.0, fisher_samples=10)
        W   = torch.randn(2, 16)
        for _ in range(10):
            ewc.accumulate(torch.rand(16), torch.rand(2))
        ewc.consolidate_task("task_A", W)
        # At W*, penalty grad should be near zero
        grad = ewc.multi_task_penalty_grad(W)
        assert float(grad.abs().mean().item()) < 1e-4

    def test_reset_clears_task_stars(self):
        ewc = EWCRegularizer(lambda_ewc=100.0)
        W   = torch.randn(2, 16)
        for _ in range(5):
            ewc.accumulate(torch.rand(16), torch.rand(2))
        ewc.consolidate_task("task_A", W)
        ewc.reset()
        assert ewc.n_consolidated_tasks() == 0


# ── OnlineCausalDiscovery.granger_edges + causal_lag_estimate ────────────────

from hdc.causal_discovery import OnlineCausalDiscovery


class TestOnlineCausalDiscoveryGranger:
    def test_granger_edges_list(self):
        ocd = OnlineCausalDiscovery(dim=64)
        ocd.step({"A": (torch.rand(64) > 0.5).float(),
                   "B": (torch.rand(64) > 0.5).float()})
        for _ in range(25):
            ocd.step({"A": (torch.rand(64) > 0.5).float(),
                       "B": (torch.rand(64) > 0.5).float()})
        edges = ocd.granger_edges()
        assert isinstance(edges, list)

    def test_granger_edges_scores_nonneg(self):
        ocd = OnlineCausalDiscovery(dim=64)
        for _ in range(25):
            ocd.step({"A": (torch.rand(64) > 0.5).float(),
                       "B": (torch.rand(64) > 0.5).float()})
        for cause, effect, score in ocd.granger_edges(threshold=0.0):
            assert score >= 0.0

    def test_causal_lag_estimate_returns_int(self):
        ocd = OnlineCausalDiscovery(dim=64)
        for _ in range(25):
            ocd.step({"X": (torch.rand(64) > 0.5).float(),
                       "Y": (torch.rand(64) > 0.5).float()})
        lag = ocd.causal_lag_estimate("X", "Y", max_lag=3)
        assert isinstance(lag, int)
        assert 0 <= lag <= 3

    def test_causal_lag_missing_variable(self):
        ocd = OnlineCausalDiscovery(dim=64)
        lag = ocd.causal_lag_estimate("X", "Z", max_lag=3)
        assert lag == 0


# ── HierarchicalInContextHDC.adapt_from_feedback + task_quality ──────────────

from hdc.in_context_hdc import HierarchicalInContextHDC


def _gen_hv_icl(seed, D=128):
    g = torch.Generator(); g.manual_seed(seed)
    return (torch.rand(D, generator=g) >= 0.5).float()


class TestHierarchicalICLAdaptation:
    def test_adapt_from_feedback_no_crash(self):
        icl = HierarchicalInContextHDC(128)
        examples = [
            (_gen_hv_icl(i), _gen_hv_icl(i + 100), f"class_{i % 3}")
            for i in range(9)
        ]
        icl.add_task("task1", examples)
        # Adapt from one correct example
        q = _gen_hv_icl(0)
        icl.adapt_from_feedback(q, true_label="class_0", task_name="task1")

    def test_task_quality_returns_dict(self):
        icl = HierarchicalInContextHDC(128)
        examples = [
            (_gen_hv_icl(i), _gen_hv_icl(i + 100), f"c{i % 2}")
            for i in range(6)
        ]
        icl.add_task("task1", examples)
        quality = icl.task_quality("task1")
        assert "intra_sim" in quality
        assert "n_examples" in quality

    def test_task_quality_unknown_task(self):
        icl = HierarchicalInContextHDC(128)
        quality = icl.task_quality("nonexistent")
        assert quality["n_examples"] == 0
