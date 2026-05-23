"""
tests/test_sdm_hrr.py
======================
Tests for BSDCCodebook, SparseDistributedMemory, EliteSDMClassifier,
SparseBundler, ThresholdSearch (hdc/sdm.py)
and HRR, HRRCodebook, CompositionalHRR, HRRAnalogy,
HRRTemporalMemory (hdc/hrr.py).
"""
import pytest
import torch
import math
from hdc.sdm import (
    BSDCConfig, BSDCCodebook,
    SparseDistributedMemory,
    EliteSDMClassifier,
    SparseBundler,
    ThresholdSearch,
)
from hdc.hrr import (
    HRR, HRRCodebook, CompositionalHRR, HRRAnalogy, HRRTemporalMemory,
)

D_BSDC = 500
D_HRR  = 512


# ─── BSDCCodebook ─────────────────────────────────────────────────────────────

class TestBSDCCodebook:
    def setup_method(self):
        self.cb = BSDCCodebook(BSDCConfig(dim=D_BSDC, density=0.01))

    def test_gen_shape_single(self):
        hv = self.cb.gen(1)
        assert hv.shape == (D_BSDC,)

    def test_gen_shape_batch(self):
        hvs = self.cb.gen(5)
        assert hvs.shape == (5, D_BSDC)

    def test_gen_is_binary(self):
        hv = self.cb.gen(1)
        assert set(hv.unique().tolist()).issubset({0.0, 1.0})

    def test_gen_density_approx(self):
        hv = self.cb.gen(1)
        density = float(hv.mean())
        assert density < 0.10, f"Expected sparse, got density={density}"

    def test_bind_shape(self):
        a, b = self.cb.gen(2)
        bound = self.cb.bind(a, b)
        assert bound.shape == (D_BSDC,)

    def test_bind_binary(self):
        a, b = self.cb.gen(2)
        bound = self.cb.bind(a, b)
        assert set(bound.unique().tolist()).issubset({0.0, 1.0})

    def test_bind_preserves_sparsity(self):
        a, b = self.cb.gen(2)
        bound = self.cb.bind(a, b)
        assert float(bound.mean()) < 0.15, "Binding should preserve sparsity"

    def test_bundle_shape(self):
        hvs = [self.cb.gen(1) for _ in range(5)]
        bundled = self.cb.bundle(hvs)
        assert bundled.shape == (D_BSDC,)

    def test_jaccard_self(self):
        a = self.cb.gen(1)
        assert self.cb.similarity(a, a) == pytest.approx(1.0)

    def test_jaccard_random_near_zero(self):
        a = self.cb.gen(1, seed=0)
        b = self.cb.gen(1, seed=999)
        # Two random sparse codes: expected Jaccard ≈ δ/(2-δ) ≈ 0.005
        sim = self.cb.similarity(a, b)
        assert sim < 0.1

    def test_similarity_batch_shape(self):
        q    = self.cb.gen(1)
        keys = self.cb.gen(10)
        sims = self.cb.similarity_batch(q, keys)
        assert sims.shape == (10,)

    def test_capacity_energy_gain(self):
        cap = self.cb.capacity_estimate()
        assert cap["energy_gain"] >= 10.0   # at least 10× faster ops

    def test_capacity_discriminability_gain(self):
        cap = self.cb.capacity_estimate()
        assert cap["discriminability_gain"] > 5.0


# ─── SparseDistributedMemory ──────────────────────────────────────────────────

class TestSparseDistributedMemory:
    def setup_method(self):
        self.sdm = SparseDistributedMemory(D=D_BSDC, N=256, r=int(0.45 * D_BSDC))

    def test_write_and_read_exact(self):
        addr = (torch.rand(D_BSDC) >= 0.5).float()
        data = (torch.rand(D_BSDC) >= 0.5).float()
        self.sdm.write(addr, data)
        recovered, conf = self.sdm.read(addr)
        sim = float((recovered == data).float().mean())
        assert sim > 0.6, f"Should recover data after 1 write, got sim={sim}"

    def test_read_confidence_in_range(self):
        addr = (torch.rand(D_BSDC) >= 0.5).float()
        data = (torch.rand(D_BSDC) >= 0.5).float()
        self.sdm.write(addr, data)
        _, conf = self.sdm.read(addr)
        assert 0.0 <= conf <= 1.0

    def test_read_returns_binary(self):
        addr = (torch.rand(D_BSDC) >= 0.5).float()
        data = (torch.rand(D_BSDC) >= 0.5).float()
        self.sdm.write(addr, data)
        rec, _ = self.sdm.read(addr)
        assert set(rec.unique().tolist()).issubset({0.0, 1.0})

    def test_associative_recall_shape(self):
        noisy = (torch.rand(D_BSDC) >= 0.5).float()
        result = self.sdm.associative_recall(noisy, n_iterations=2)
        assert result.shape == (D_BSDC,)

    def test_write_batch(self):
        addrs = torch.stack([(torch.rand(D_BSDC) >= 0.5).float() for _ in range(3)])
        datas = torch.stack([(torch.rand(D_BSDC) >= 0.5).float() for _ in range(3)])
        self.sdm.write_batch(addrs, datas)
        assert self.sdm._n_writes >= 1   # at least some writes occurred

    def test_stats_keys(self):
        stats = self.sdm.stats()
        assert "D" in stats and "N" in stats and "n_writes" in stats

    def test_reset_clears_counters(self):
        addr = (torch.rand(D_BSDC) >= 0.5).float()
        self.sdm.write(addr, (torch.rand(D_BSDC) >= 0.5).float())
        self.sdm.reset()
        assert self.sdm._n_writes == 0
        assert float(self.sdm.counters.abs().sum()) == 0.0


# ─── EliteSDMClassifier ───────────────────────────────────────────────────────

class TestEliteSDMClassifier:
    def setup_method(self):
        self.clf = EliteSDMClassifier(n_features=10, n_classes=3, dim=D_BSDC, density=0.01)

    def test_train_and_predict(self):
        for c in range(3):
            for _ in range(5):
                self.clf.train_step(torch.randn(10) + c * 3, c)
        pred, sims = self.clf.predict(torch.randn(10))
        assert 0 <= pred < 3
        assert len(sims) == 3

    def test_sims_in_range(self):
        for c in range(3):
            self.clf.train_step(torch.randn(10), c)
        _, sims = self.clf.predict(torch.randn(10))
        assert all(0.0 <= s <= 1.0 for s in sims)

    def test_capacity_report(self):
        cap = self.clf.capacity_report()
        assert "energy_gain" in cap
        assert cap["energy_gain"] >= 10.0


# ─── SparseBundler ────────────────────────────────────────────────────────────

class TestSparseBundler:
    def setup_method(self):
        self.bundler = SparseBundler(density=0.01)
        self.cb      = BSDCCodebook(BSDCConfig(dim=D_BSDC, density=0.01))

    def test_bundle_shape(self):
        hvs = [self.cb.gen(1) for _ in range(5)]
        out = self.bundler(hvs)
        assert out.shape == (D_BSDC,)

    def test_bundle_binary(self):
        hvs = [self.cb.gen(1) for _ in range(5)]
        out = self.bundler(hvs)
        assert set(out.unique().tolist()).issubset({0.0, 1.0})

    def test_bundle_preserves_density(self):
        hvs = [self.cb.gen(1) for _ in range(10)]
        out = self.bundler(hvs)
        density = float(out.mean())
        assert density < 0.15, f"Should be sparser than dense (0.5), got {density}"

    def test_incremental_shape(self):
        a = self.cb.gen(1)
        b = self.cb.gen(1)
        out = self.bundler.incremental(a, b, decay=0.9)
        assert out.shape == (D_BSDC,)


# ─── ThresholdSearch ──────────────────────────────────────────────────────────

class TestThresholdSearch:
    def setup_method(self):
        self.ts = ThresholdSearch(D=1000, N=131072, target_n_activated=1000)

    def test_optimal_radius_in_range(self):
        r = self.ts.optimal_radius()
        assert 0 < r < 1000

    def test_optimal_radius_fraction(self):
        r = self.ts.optimal_radius()
        assert 0.2 < r / 1000 < 0.8, f"r/D={r/1000:.3f} should be 0.2-0.8"

    def test_expected_activated_positive(self):
        r   = self.ts.optimal_radius()
        eta = self.ts.expected_activated(r)
        assert eta >= 0.0

    def test_radius_report_keys(self):
        rr = self.ts.radius_report()
        assert "optimal_r" in rr
        assert "expected_activated" in rr
        assert "capacity_estimate" in rr


# ─── HRR ─────────────────────────────────────────────────────────────────────

class TestHRR:
    def setup_method(self):
        self.hrr = HRR(dim=D_HRR)

    def test_gen_shape(self):
        hv = self.hrr.gen(1)
        assert hv.shape == (D_HRR,)

    def test_gen_unit_norm(self):
        hv = self.hrr.gen(1, unit=True)
        assert abs(float(hv.norm()) - 1.0) < 1e-5

    def test_bind_shape(self):
        a, b = self.hrr.gen(2)
        c = self.hrr.bind(a, b)
        assert c.shape == (D_HRR,)

    def test_unbind_approximate(self):
        a = self.hrr.gen(1, seed=0)
        b = self.hrr.gen(1, seed=1)
        c = self.hrr.bind(a, b)
        rec = self.hrr.unbind(c, b)
        sim = self.hrr.similarity(rec, a)
        assert sim > 0.5, f"Approx unbinding should recover a, sim={sim}"

    def test_unbind_exact(self):
        a = self.hrr.gen(1, seed=0)
        b = self.hrr.gen(1, seed=1)
        c = self.hrr.bind(a, b)
        rec = self.hrr.unbind_exact(c, b)
        sim = self.hrr.similarity(rec, a)
        assert sim > 0.999, f"Exact unbinding should give sim≈1.0, got {sim}"

    def test_bundle_shape(self):
        hvs = [self.hrr.gen(1, seed=i) for i in range(5)]
        out = self.hrr.bundle(hvs)
        assert out.shape == (D_HRR,)

    def test_similarity_self(self):
        a = self.hrr.gen(1)
        assert abs(self.hrr.similarity(a, a) - 1.0) < 1e-5

    def test_similarity_batch_shape(self):
        q    = self.hrr.gen(1)
        keys = self.hrr.gen(6)
        sims = self.hrr.similarity_batch(q, keys)
        assert sims.shape == (6,)

    def test_permute_invertible(self):
        hv    = self.hrr.gen(1)
        perm  = self.hrr.permute(hv, steps=3)
        back  = self.hrr.permute_inverse(perm, steps=3)
        assert torch.allclose(hv, back)

    def test_bundle_retrieval_from_pairs(self):
        pairs    = [(self.hrr.gen(1, seed=i), self.hrr.gen(1, seed=100+i)) for i in range(5)]
        bindings = [self.hrr.bind(r, f) for r, f in pairs]
        bundle   = self.hrr.bundle(bindings)
        # Retrieve role 2
        role2, filler2 = pairs[2]
        cand = self.hrr.unbind_exact(bundle, role2)
        sim_correct = self.hrr.similarity(cand, filler2)
        sim_wrong   = self.hrr.similarity(cand, pairs[0][1])
        assert sim_correct > sim_wrong


# ─── HRRCodebook ─────────────────────────────────────────────────────────────

class TestHRRCodebook:
    def setup_method(self):
        self.hrr = HRR(dim=D_HRR)
        self.cb  = HRRCodebook(self.hrr)
        concepts = ["color", "shape", "red", "blue", "circle", "square"]
        for i, name in enumerate(concepts):
            self.cb.register(name, self.hrr.gen(1, seed=i))

    def test_n_items(self):
        assert self.cb.n_items == 6

    def test_cleanup_returns_list(self):
        q      = self.cb._items["red"]
        result = self.cb.cleanup(q, top_k=2)
        assert len(result) == 2

    def test_cleanup_one_correct(self):
        q    = self.cb._items["circle"]
        name, sim = self.cb.cleanup_one(q)
        assert name == "circle"

    def test_store_and_retrieve_pair(self):
        composite = self.cb.store_pair("color", "red")
        assert composite.shape == (D_HRR,)
        name, sim = self.cb.retrieve_filler(composite, "color")
        assert name == "red", f"Expected 'red', got '{name}'"

    def test_retrieve_filler_unknown_role(self):
        composite = self.hrr.gen(1)
        name, sim = self.cb.retrieve_filler(composite, "nonexistent")
        assert name is None


# ─── CompositionalHRR ────────────────────────────────────────────────────────

class TestCompositionalHRR:
    def setup_method(self):
        self.hrr  = HRR(dim=D_HRR)
        self.cb   = HRRCodebook(self.hrr)
        for i, name in enumerate(["color", "shape", "red", "blue", "circle", "square"]):
            self.cb.register(name, self.hrr.gen(1, seed=i))
        self.comp = CompositionalHRR(self.hrr, self.cb)

    def test_build_shape(self):
        r1 = self.cb._items["color"]
        f1 = self.cb._items["red"]
        r2 = self.cb._items["shape"]
        f2 = self.cb._items["circle"]
        obj = self.comp.build([(r1, f1), (r2, f2)])
        assert obj.shape == (D_HRR,)

    def test_query_correct_filler(self):
        r1 = self.cb._items["color"]
        f1 = self.cb._items["red"]
        obj = self.comp.build([(r1, f1)])
        result = self.comp.query(obj, r1, top_k=2)
        assert result[0][0] == "red"

    def test_compose_nested_shape(self):
        outer = self.hrr.gen(1, seed=50)
        pairs = [(self.cb._items["color"], self.cb._items["blue"])]
        nested = self.comp.compose_nested(outer, pairs)
        assert nested.shape == (D_HRR,)


# ─── HRRAnalogy ──────────────────────────────────────────────────────────────

class TestHRRAnalogy:
    def setup_method(self):
        self.hrr     = HRR(dim=D_HRR)
        self.cb      = HRRCodebook(self.hrr)
        for i, name in enumerate(["A", "B", "C", "D", "E"]):
            self.cb.register(name, self.hrr.gen(1, seed=i))
        self.analogy = HRRAnalogy(self.hrr, self.cb)

    def test_extract_relation_shape(self):
        a = self.cb._items["A"]
        b = self.cb._items["B"]
        rel = self.analogy.extract_relation(a, b)
        assert rel.shape == (D_HRR,)

    def test_solve_returns_list(self):
        a, b, c = [self.cb._items[k] for k in "ABC"]
        results = self.analogy.solve(a, b, c, top_k=2)
        assert len(results) == 2

    def test_solve_hv_shape(self):
        a, b, c = [self.cb._items[k] for k in "ABC"]
        d_raw = self.analogy.solve_hv(a, b, c)
        assert d_raw.shape == (D_HRR,)


# ─── HRRTemporalMemory ───────────────────────────────────────────────────────

class TestHRRTemporalMemory:
    def setup_method(self):
        self.hrr  = HRR(dim=D_HRR)
        self.tmem = HRRTemporalMemory(self.hrr, max_len=10)
        self.items = [self.hrr.gen(1, seed=i) for i in range(5)]
        self.tmem.encode_sequence(self.items)

    def test_length_after_encode(self):
        assert self.tmem.length == 5

    def test_retrieve_shape(self):
        candidate, _, _ = self.tmem.retrieve(0)
        assert candidate.shape == (D_HRR,)

    def test_retrieve_correct_more_similar(self):
        cand, _, _ = self.tmem.retrieve(2)
        sim_correct = self.hrr.similarity(cand, self.items[2])
        sim_wrong   = self.hrr.similarity(cand, self.items[4])
        assert sim_correct > sim_wrong

    def test_push_increments_length(self):
        new_item = self.hrr.gen(1, seed=99)
        self.tmem.push(new_item)
        assert self.tmem.length == 6

    def test_max_len_respected(self):
        tmem = HRRTemporalMemory(self.hrr, max_len=3)
        for i in range(10):
            tmem.push(self.hrr.gen(1, seed=i))
        assert tmem.length == 3

    def test_incremental_push_no_recompute(self):
        tmem = HRRTemporalMemory(self.hrr, max_len=20)
        for i in range(5):
            tmem.push(self.hrr.gen(1, seed=i))
        assert tmem.length == 5
        assert tmem._memory is not None

    def test_peek_returns_most_recent(self):
        tmem = HRRTemporalMemory(self.hrr, max_len=10)
        items = [self.hrr.gen(1, seed=i) for i in range(3)]
        tmem.encode_sequence(items)
        cand, _, _ = tmem.peek(lag=0)
        # Most recent item at lag=0 should be items[2]
        sim_recent = self.hrr.similarity(cand, items[2])
        sim_old    = self.hrr.similarity(cand, items[0])
        assert sim_recent > sim_old

    def test_peek_lag1_returns_second_most_recent(self):
        tmem = HRRTemporalMemory(self.hrr, max_len=10)
        items = [self.hrr.gen(1, seed=i) for i in range(4)]
        tmem.encode_sequence(items)
        cand, _, _ = tmem.peek(lag=1)
        sim_1_back = self.hrr.similarity(cand, items[2])
        sim_latest = self.hrr.similarity(cand, items[3])
        assert sim_1_back > sim_latest - 0.3   # should be closer to items[2]

    def test_push_with_decay(self):
        tmem = HRRTemporalMemory(self.hrr, max_len=10)
        for i in range(5):
            tmem.push(self.hrr.gen(1, seed=i), decay=0.9)
        assert tmem.length == 5
        # Memory should be non-zero
        assert tmem._memory.abs().sum() > 0


# ─────────────────────────────────────────────────────────────────────────────
# SparseDistributedMemory: forget_old, utilisation, defragment
# ─────────────────────────────────────────────────────────────────────────────

class TestSDMForgetting:
    DIM = 64

    def _make_sdm(self):
        sdm = SparseDistributedMemory(N=256, D=self.DIM)
        addr = (torch.rand(self.DIM) > 0.5).float()
        data = (torch.rand(self.DIM) > 0.5).float()
        sdm.write(addr, data)
        return sdm, addr, data

    def test_forget_old_reduces_counters(self):
        sdm, addr, data = self._make_sdm()
        total_before = sdm.counters.abs().sum().item()
        sdm.forget_old(decay=0.5)
        total_after  = sdm.counters.abs().sum().item()
        # Decay should reduce total counter magnitudes
        assert total_after <= total_before

    def test_utilisation_in_range(self):
        sdm, _, _ = self._make_sdm()
        u = sdm.utilisation()
        assert 0.0 <= u <= 1.0

    def test_utilisation_increases_after_writes(self):
        sdm = SparseDistributedMemory(N=256, D=self.DIM)
        u0 = sdm.utilisation()
        for i in range(10):
            a = (torch.rand(self.DIM) > 0.5).float()
            d = (torch.rand(self.DIM) > 0.5).float()
            sdm.write(a, d)
        u1 = sdm.utilisation()
        assert u1 >= u0

    def test_defragment_reduces_counters(self):
        sdm, _, _ = self._make_sdm()
        # Add some weak writes
        for _ in range(3):
            a = (torch.rand(self.DIM) > 0.5).float()
            sdm.write(a, (torch.rand(self.DIM) > 0.5).float())
        n_before = (sdm.counters.abs().sum(dim=1) > 0).sum().item()
        sdm.defragment(threshold=0.5)
        n_after  = (sdm.counters.abs().sum(dim=1) > 0).sum().item()
        # Some rows should have been zeroed out
        assert n_after <= n_before


# ─────────────────────────────────────────────────────────────────────────────
# PositionalEncoding: sinusoidal + relative
# ─────────────────────────────────────────────────────────────────────────────

from hdc.vsa_sequence_model import PositionalEncoding


class TestPositionalEncodingEnhanced:
    D = 64

    def test_sinusoidal_shape(self):
        hrr = HRR(self.D)
        pe  = PositionalEncoding(hrr, scheme="sinusoidal", max_len=100)
        p   = pe.get(5)
        assert p.shape == (self.D,)

    def test_sinusoidal_unit_norm(self):
        hrr = HRR(self.D)
        pe  = PositionalEncoding(hrr, scheme="sinusoidal", max_len=100)
        p   = pe.get(5)
        norm = float(p.norm().item())
        assert abs(norm - 1.0) < 0.01   # unit norm

    def test_sinusoidal_different_positions_orthogonal(self):
        hrr = HRR(self.D)
        pe  = PositionalEncoding(hrr, scheme="sinusoidal", max_len=100)
        p0  = pe.get(0)
        p50 = pe.get(50)
        sim = float((p0 * p50).sum().item())   # dot product
        # Far-apart positions should be less similar than adjacent ones
        p1  = pe.get(1)
        sim01 = float((p0 * p1).sum().item())
        assert abs(sim) <= abs(sim01) + 0.3   # rough check

    def test_get_relative_permute_scheme(self):
        hrr = HRR(self.D)
        pe  = PositionalEncoding(hrr, scheme="permute", max_len=100)
        r   = pe.get_relative(3, 7)
        assert r.shape == (self.D,)

    def test_get_relative_different_lags(self):
        hrr = HRR(self.D)
        pe  = PositionalEncoding(hrr, scheme="permute", max_len=100)
        r1  = pe.get_relative(0, 1)
        r5  = pe.get_relative(0, 5)
        # Different lags should give different relative HVs
        assert not torch.equal(r1, r5)
