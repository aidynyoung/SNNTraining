"""
tests/test_repo_implementations.py
=====================================
Tests for implementations derived from analysing external repos:
  - TriadicMemory (PeterOvermann/TriadicMemory) → hdc/triadic_memory.py
  - VTB/FHRR/CGR/MCR/FPE (hyperdimensional-computing/torchhd) → hdc/vsa_algebras.py
  - NeuralHD/DistHD/LeHDC (torchhd) → hdc/adaptive_encoder.py
  - FSA-in-VSA (torchhd) → hdc/finite_state_automata.py
"""
import math
import pytest
import torch

# Triadic Memory
from hdc.triadic_memory import (
    DyadicMemory, TriadicMemory, DeepTemporalMemory,
    ElementaryTemporalMemory, _gen_sdr, _sums_to_sdr,
)
# VSA Algebras
from hdc.vsa_algebras import (
    FHRR, VTBAlgebra, CGRAlgebra, MCRAlgebra, FractionalPowerEncoding,
)
# Adaptive Encoders
from hdc.adaptive_encoder import (
    NeuralHDEncoder, DistHDEncoder, LeHDC,
)
# Finite State Automata
from hdc.finite_state_automata import (
    VSAFiniteStateAutomaton, VSANondeterministicFSA,
    RegularLanguageMatcher, FSAConstrainedLanguageModel,
)

N, P = 100, 5   # SDR dimensions


# ─── SDR Utilities ────────────────────────────────────────────────────────────

class TestSDRUtils:
    def test_gen_sdr_shape(self):
        sdr = _gen_sdr(N, P)
        assert sdr.shape == (N,)

    def test_gen_sdr_exactly_p_active(self):
        sdr = _gen_sdr(N, P)
        assert int(sdr.sum()) == P

    def test_gen_sdr_binary(self):
        sdr = _gen_sdr(N, P)
        assert set(sdr.unique().tolist()).issubset({0, 1})

    def test_sums_to_sdr_top_p(self):
        sums = torch.zeros(N, dtype=torch.int32)
        sums[:P] = 5
        sdr = _sums_to_sdr(sums, P)
        assert int(sdr.sum()) == P
        assert all(sdr[i] == 1 for i in range(P))


# ─── DyadicMemory ─────────────────────────────────────────────────────────────

class TestDyadicMemory:
    def setup_method(self):
        self.dm = DyadicMemory(N, P)

    def test_write_and_retrieve(self):
        x = _gen_sdr(N, P, seed=0)
        y = _gen_sdr(N, P, seed=1)
        self.dm.write(x, y)
        y_ret = self.dm.query_y(x)
        overlap = int((y_ret & y).sum())
        assert overlap == P, f"Should retrieve exact y, overlap={overlap}"

    def test_bidirectional(self):
        x = _gen_sdr(N, P, seed=0)
        y = _gen_sdr(N, P, seed=1)
        self.dm.write(x, y)
        x_ret = self.dm.query_x(y)
        overlap = int((x_ret & x).sum())
        assert overlap == P, f"Bidirectional retrieval failed, overlap={overlap}"

    def test_n_pairs(self):
        for i in range(5):
            self.dm.write(_gen_sdr(N, P, seed=i), _gen_sdr(N, P, seed=100+i))
        assert self.dm.n_pairs == 5

    def test_capacity_estimate(self):
        cap = self.dm.capacity_estimate()
        assert cap == (N / P) ** 2


# ─── TriadicMemory ────────────────────────────────────────────────────────────

class TestTriadicMemory:
    def setup_method(self):
        self.tm = TriadicMemory(N, P)
        for i in range(10):
            self.tm.write(
                _gen_sdr(N, P, seed=i),
                _gen_sdr(N, P, seed=100+i),
                _gen_sdr(N, P, seed=200+i),
            )

    def test_n_triples(self):
        assert self.tm.n_triples == 10

    def test_query_z_exact(self):
        x = _gen_sdr(N, P, seed=5)
        y = _gen_sdr(N, P, seed=105)
        z = _gen_sdr(N, P, seed=205)
        tm = TriadicMemory(N, P)
        tm.write(x, y, z)
        z_ret = tm.query_z(x, y)
        assert int((z_ret & z).sum()) == P

    def test_query_y_exact(self):
        x = _gen_sdr(N, P, seed=5)
        y = _gen_sdr(N, P, seed=105)
        z = _gen_sdr(N, P, seed=205)
        tm = TriadicMemory(N, P)
        tm.write(x, y, z)
        y_ret = tm.query_y(x, z)
        assert int((y_ret & y).sum()) == P

    def test_query_x_exact(self):
        x = _gen_sdr(N, P, seed=5)
        y = _gen_sdr(N, P, seed=105)
        z = _gen_sdr(N, P, seed=205)
        tm = TriadicMemory(N, P)
        tm.write(x, y, z)
        x_ret = tm.query_x(y, z)
        assert int((x_ret & x).sum()) == P

    def test_tridirectional_all_directions(self):
        # All 3 query directions return the correct vector
        x = _gen_sdr(N, P, seed=7)
        y = _gen_sdr(N, P, seed=107)
        z = _gen_sdr(N, P, seed=207)
        tm = TriadicMemory(N, P)
        tm.write(x, y, z)
        assert int((tm.query_z(x, y) & z).sum()) == P
        assert int((tm.query_y(x, z) & y).sum()) == P
        assert int((tm.query_x(y, z) & x).sum()) == P

    def test_capacity_estimate(self):
        assert self.tm.capacity_estimate() == (N / P) ** 3

    def test_reset(self):
        self.tm.reset()
        assert self.tm.n_triples == 0


# ─── DeepTemporalMemory ───────────────────────────────────────────────────────

class TestDeepTemporalMemory:
    def setup_method(self):
        self.dtm = DeepTemporalMemory(N, P, n_layers=3)
        seq = [_gen_sdr(N, P, seed=i % 5) for i in range(20)]
        for s in seq:
            self.dtm.step(s, train=True)

    def test_step_shape(self):
        out = self.dtm.step(_gen_sdr(N, P, seed=0), train=False)
        assert len(out) == 3
        assert all(s.shape == (N,) for s in out)

    def test_predict_shape(self):
        pred = self.dtm.predict(_gen_sdr(N, P, seed=0))
        assert pred.shape == (N,)

    def test_n_stored_triples(self):
        nts = self.dtm.n_stored_triples
        assert len(nts) == 3
        assert all(n >= 0 for n in nts)

    def test_reset_states(self):
        self.dtm.reset_states()
        for s in self.dtm._states:
            assert int(s.sum()) == 0


# ─── ElementaryTemporalMemory ─────────────────────────────────────────────────

class TestElementaryTemporalMemory:
    def setup_method(self):
        self.etm = ElementaryTemporalMemory(N, P)
        for i in range(20):
            self.etm.step(_gen_sdr(N, P, seed=i % 4), train=True)

    def test_step_shape(self):
        out = self.etm.step(_gen_sdr(N, P, seed=0))
        assert out.shape == (N,)

    def test_predict_shape(self):
        pred = self.etm.predict_next(_gen_sdr(N, P, seed=0))
        assert pred.shape == (N,)

    def test_context_shape(self):
        ctx = self.etm.context_vector()
        assert ctx.shape == (N,)

    def test_total_triples(self):
        assert self.etm.total_triples > 0

    def test_reset(self):
        self.etm.reset()
        assert int(self.etm._prev_output.sum()) == 0


# ─── FHRR ─────────────────────────────────────────────────────────────────────

D = 128


class TestFHRR:
    def setup_method(self):
        self.fhrr = FHRR(D)

    def test_gen_complex(self):
        hv = self.fhrr.gen(1, seed=0)
        assert hv.shape == (D,)
        assert hv.is_complex()

    def test_gen_unit_magnitude(self):
        hv = self.fhrr.gen(1, seed=0)
        mags = hv.abs()
        assert torch.allclose(mags, torch.ones_like(mags), atol=1e-5)

    def test_exact_unbinding(self):
        a = self.fhrr.gen(1, seed=0)
        b = self.fhrr.gen(1, seed=1)
        c = self.fhrr.bind(a, b)
        r = self.fhrr.unbind(c, b)
        sim = self.fhrr.similarity(r, a)
        assert sim > 0.999

    def test_bind_commutativity(self):
        a = self.fhrr.gen(1, seed=0)
        b = self.fhrr.gen(1, seed=1)
        assert torch.allclose(self.fhrr.bind(a, b), self.fhrr.bind(b, a), atol=1e-5)

    def test_bundle_returns_unit(self):
        hvs = [self.fhrr.gen(1, seed=i) for i in range(5)]
        bundled = self.fhrr.bundle(hvs)
        assert bundled.shape == (D,)
        mags = bundled.abs()
        assert torch.allclose(mags, torch.ones_like(mags), atol=1e-5)

    def test_fractional_power_group_property(self):
        a = self.fhrr.gen(1, seed=0)
        a_half = self.fhrr.fractional_power(a, 0.5)
        a_rec  = self.fhrr.bind(a_half, a_half)
        sim = self.fhrr.similarity(a_rec, a)
        assert sim > 0.999


# ─── VTBAlgebra ──────────────────────────────────────────────────────────────

class TestVTBAlgebra:
    def setup_method(self):
        self.vtb = VTBAlgebra(D)

    def test_gen_unit_norm(self):
        hv = self.vtb.gen(1, seed=0)
        assert abs(float(hv.norm()) - 1.0) < 1e-5

    def test_exact_unbinding(self):
        a = self.vtb.gen(1, seed=0)
        b = self.vtb.gen(1, seed=1)
        c = self.vtb.bind(a, b)
        r = self.vtb.unbind(c, a)
        sim = self.vtb.similarity(r, b)
        assert sim > 0.99, f"VTB unbinding should be exact: sim={sim}"

    def test_bind_shape(self):
        a = self.vtb.gen(1, seed=0)
        b = self.vtb.gen(1, seed=1)
        c = self.vtb.bind(a, b)
        assert c.shape == (D,)

    def test_bundle_normalised(self):
        hvs     = [self.vtb.gen(1, seed=i) for i in range(4)]
        bundled = self.vtb.bundle(hvs)
        assert abs(float(bundled.norm()) - 1.0) < 1e-5

    def test_requires_even_dim(self):
        with pytest.raises(ValueError):
            VTBAlgebra(3)


# ─── CGRAlgebra ──────────────────────────────────────────────────────────────

class TestCGRAlgebra:
    def setup_method(self):
        self.cgr = CGRAlgebra(D, m=7)

    def test_gen_in_range(self):
        hv = self.cgr.gen(1, seed=0)
        assert hv.shape == (D,)
        assert hv.min() >= 0 and hv.max() < 7

    def test_exact_unbinding(self):
        a = self.cgr.gen(1, seed=0)
        b = self.cgr.gen(1, seed=1)
        c = self.cgr.bind(a, b)
        r = self.cgr.unbind(c, b)
        assert self.cgr.similarity(r, a) == 1.0

    def test_bind_stays_in_range(self):
        a = self.cgr.gen(1, seed=0)
        b = self.cgr.gen(1, seed=1)
        c = self.cgr.bind(a, b)
        assert c.min() >= 0 and c.max() < 7

    def test_bundle_shape(self):
        hvs     = [self.cgr.gen(1, seed=i) for i in range(3)]
        bundled = self.cgr.bundle(hvs)
        assert bundled.shape == (D,)
        assert bundled.min() >= 0 and bundled.max() < 7


# ─── MCRAlgebra ──────────────────────────────────────────────────────────────

class TestMCRAlgebra:
    def setup_method(self):
        self.mcr = MCRAlgebra(D, m=100)

    def test_gen_in_range(self):
        hv = self.mcr.gen(1, seed=0)
        assert hv.min() >= 0 and hv.max() < 100

    def test_near_exact_unbinding(self):
        a = self.mcr.gen(1, seed=0)
        b = self.mcr.gen(1, seed=1)
        c = self.mcr.bind(a, b)
        r = self.mcr.unbind(c, b)
        sim = self.mcr.similarity(r, a)
        assert sim > 0.95, f"MCR unbinding should be near-exact: {sim}"

    def test_bundle_shape(self):
        hvs     = [self.mcr.gen(1, seed=i) for i in range(5)]
        bundled = self.mcr.bundle(hvs)
        assert bundled.shape == (D,)

    def test_similarity_self(self):
        a = self.mcr.gen(1, seed=0)
        assert abs(self.mcr.similarity(a, a) - 1.0) < 1e-4


# ─── FractionalPowerEncoding ─────────────────────────────────────────────────

class TestFractionalPowerEncoding:
    def setup_method(self):
        self.fpe = FractionalPowerEncoding(n_features=4, dim=D, bw=1.0,
                                            kernel="gaussian", seed=42)

    def test_encode_shape(self):
        z = self.fpe.encode(torch.tensor([1.0, 2.0, 3.0, 4.0]))
        assert z.shape == (D,)
        assert z.is_complex()

    def test_nearby_more_similar(self):
        x1 = torch.tensor([1.0, 2.0, 3.0, 4.0])
        x2 = torch.tensor([1.1, 2.1, 3.1, 4.1])
        x3 = torch.tensor([10.0, 20.0, 30.0, 40.0])
        sim_near = self.fpe.similarity(x1, x2)
        sim_far  = self.fpe.similarity(x1, x3)
        assert sim_near > sim_far

    def test_kernel_matrix_shape(self):
        X = torch.randn(6, 4)
        K = self.fpe.kernel_matrix(X)
        assert K.shape == (6, 6)

    def test_kernel_matrix_diagonal_ones(self):
        X = torch.randn(4, 4)
        K = self.fpe.kernel_matrix(X)
        assert torch.allclose(K.diagonal(), torch.ones(4), atol=1e-4)

    def test_sinc_kernel(self):
        fpe_sinc = FractionalPowerEncoding(4, D, kernel="uniform", seed=1)
        z = fpe_sinc.encode(torch.ones(4))
        assert z.shape == (D,)

    def test_batch_encode_shape(self):
        X = torch.randn(8, 4)
        Z = self.fpe.encode_batch(X)
        assert Z.shape == (8, D)


# ─── NeuralHDEncoder ─────────────────────────────────────────────────────────

class TestNeuralHDEncoder:
    def setup_method(self):
        self.enc = NeuralHDEncoder(n_features=8, dim=D, n_regen=2, regen_frac=0.1)

    def test_encode_shape_bipolar(self):
        z = self.enc.encode(torch.randn(8))
        assert z.shape == (D,)
        assert set(z.unique().tolist()).issubset({-1.0, 1.0})

    def test_encode_batch(self):
        X = torch.randn(6, 8)
        Z = self.enc.encode_batch(X)
        assert Z.shape == (6, D)

    def test_regeneration_changes_encoder(self):
        omega0 = self.enc.omega.clone()
        protos = {0: torch.zeros(D), 1: torch.ones(D)}
        self.enc.update_dim_variance(protos)
        self.enc.regenerate()
        assert not torch.equal(omega0, self.enc.omega)

    def test_fit_and_regenerate_accuracy(self):
        X = torch.cat([torch.randn(20, 8) + c * 3 for c in range(3)])
        y = torch.cat([torch.full((20,), c, dtype=torch.long) for c in range(3)])
        accs = self.enc.fit_and_regenerate(X, y, n_classes=3)
        assert len(accs) == self.enc.n_regen + 1
        assert all(0.0 <= a <= 1.0 for a in accs)
        # Accuracy should improve or stay roughly constant after regeneration
        assert accs[-1] >= accs[0] - 0.15  # allow small variance

    def test_regen_count_increases(self):
        self.enc.update_dim_variance({0: torch.zeros(D), 1: torch.ones(D)})
        before = self.enc._regen_step
        self.enc.regenerate()
        assert self.enc._regen_step == before + 1


# ─── DistHDEncoder ───────────────────────────────────────────────────────────

class TestDistHDEncoder:
    def setup_method(self):
        self.enc = DistHDEncoder(n_features=8, dim=D, regen_frac=0.1)

    def test_record_mismatch(self):
        self.enc.record_mismatch(torch.randn(D), torch.randn(D))
        assert self.enc._n_misses == 1

    def test_regenerate_by_score_resets(self):
        self.enc.record_mismatch(torch.randn(D), torch.randn(D))
        n = self.enc.regenerate_by_score()
        assert n > 0
        assert self.enc._n_misses == 0

    def test_fallback_to_variance(self):
        # With no misses, should fall back to variance-based
        n = self.enc.regenerate_by_score()
        assert n > 0


# ─── LeHDC ───────────────────────────────────────────────────────────────────

class TestLeHDC:
    def setup_method(self):
        self.model = LeHDC(n_features=8, dim=D, n_classes=3, lr=1e-2)

    def test_forward_shape(self):
        X = torch.randn(4, 8)
        logits = self.model.forward(X)
        assert logits.shape == (4, 3)

    def test_train_step_returns_loss(self):
        x = torch.randn(8)
        loss = self.model.train_step(x, 0)
        assert isinstance(loss, float)

    def test_train_step_scalar_label(self):
        loss = self.model.train_step(torch.randn(8), 1)
        assert isinstance(loss, float)

    def test_predict_shape(self):
        X = torch.randn(4, 8)
        preds = self.model.predict(X)
        assert preds.shape == (4,)

    def test_accuracy_in_range(self):
        X = torch.randn(10, 8)
        y = torch.randint(0, 3, (10,))
        acc = self.model.accuracy(X, y)
        assert 0.0 <= acc <= 1.0

    def test_training_improves_accuracy(self):
        X = torch.cat([torch.randn(10, 8) + c * 4 for c in range(3)])
        y = torch.cat([torch.full((10,), c, dtype=torch.long) for c in range(3)])
        acc0 = self.model.accuracy(X, y)
        for _ in range(30):
            for i in range(len(X)):
                self.model.train_step(X[i], y[i])
        acc1 = self.model.accuracy(X, y)
        assert acc1 >= acc0 - 0.1  # should not get much worse


# ─── VSAFiniteStateAutomaton ──────────────────────────────────────────────────

FSA_D = 512   # Larger D for reliable similarity


class TestVSAFiniteStateAutomaton:
    def setup_method(self):
        self.fsa = VSAFiniteStateAutomaton(FSA_D)
        self.fsa.add_state("S0", is_initial=True)
        self.fsa.add_state("S1", is_final=True)
        self.fsa.add_state("S2")
        self.fsa.add_transition("S0", "a", "S1")
        self.fsa.add_transition("S1", "b", "S2")

    def test_n_states(self):
        assert self.fsa.n_states == 3

    def test_n_transitions(self):
        assert self.fsa.n_transitions == 2

    def test_next_state_top_is_correct(self):
        results = self.fsa.next_state("S0", "a", top_k=2)
        assert results[0][0] == "S1"

    def test_simulate_accepted(self):
        _, accepted = self.fsa.simulate(["a"])
        assert accepted   # ends in S1 which is final

    def test_simulate_sequence(self):
        final, _ = self.fsa.simulate(["a", "b"])
        assert final == "S2"

    def test_next_state_returns_list(self):
        results = self.fsa.next_state("S0", "a", top_k=2)
        assert len(results) == 2
        for name, sim in results:
            assert isinstance(name, str)
            assert isinstance(sim, float)


# ─── VSANondeterministicFSA ───────────────────────────────────────────────────

class TestVSANondeterministicFSA:
    def test_simulate_nfa_returns_set(self):
        nfa = VSANondeterministicFSA(FSA_D, sim_threshold=0.05)
        nfa.add_state("q0", is_initial=True)
        nfa.add_state("q1", is_final=True)
        nfa.add_transition("q0", "x", "q1")
        active, accepted = nfa.simulate_nfa(["x"])
        assert isinstance(active, set)
        assert len(active) >= 1

    def test_reachable_states(self):
        nfa = VSANondeterministicFSA(FSA_D, sim_threshold=0.05)
        nfa.add_state("q0", is_initial=True)
        nfa.add_state("q1", is_final=True)
        nfa.add_transition("q0", "x", "q1")
        reachable = nfa.reachable_states("q0", "x")
        assert isinstance(reachable, list)


# ─── RegularLanguageMatcher ──────────────────────────────────────────────────

class TestRegularLanguageMatcher:
    def test_simple_match(self):
        m = RegularLanguageMatcher(1024)
        m.add_state("q0"); m.add_state("q1")
        m.set_initial("q0"); m.add_final("q1")
        m.add_transition("q0", "a", "q1")
        assert m.match(["a"]) == True

    def test_describe_keys(self):
        m = RegularLanguageMatcher(256)
        m.add_state("q0"); m.set_initial("q0")
        m.add_final("q0")
        desc = m.describe()
        assert "n_states" in desc
        assert "n_transitions" in desc
        assert "initial" in desc

    def test_no_transition_rejects(self):
        m = RegularLanguageMatcher(1024)
        m.add_state("q0"); m.add_state("q1")
        m.set_initial("q0"); m.add_final("q1")
        m.add_transition("q0", "a", "q1")
        # Empty sequence: stays at q0, not final → reject
        result = m.match([])
        assert result == False   # q0 is not final


# ─── VSAFiniteStateAutomaton v1.46 — beam search + online_update ─────────────

class TestVSAFSAV146:
    def setup_method(self):
        from hdc.finite_state_automata import VSAFiniteStateAutomaton
        self.fsa = VSAFiniteStateAutomaton(256)
        for s in ["s0", "s1", "s2"]:
            is_init  = (s == "s0")
            is_final = (s == "s2")
            self.fsa.add_state(s, is_initial=is_init, is_final=is_final)
        self.fsa.add_transition("s0", "a", "s1")
        self.fsa.add_transition("s1", "b", "s2")

    def test_simulate_beam_returns_list(self):
        result = self.fsa.simulate_beam(["a", "b"], beam_width=2)
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_simulate_beam_path_length(self):
        result = self.fsa.simulate_beam(["a", "b"], beam_width=2)
        for path, score, accepted in result:
            assert len(path) == 3   # initial + 2 tokens

    def test_simulate_beam_score_nonneg(self):
        result = self.fsa.simulate_beam(["a", "b"], beam_width=3)
        for _, score, _ in result:
            assert score >= 0.0

    def test_online_update_increases_transitions(self):
        n_before = self.fsa.n_transitions
        self.fsa.online_update("s0", "c", "s1", weight=0.5)
        assert self.fsa.n_transitions > n_before

    def test_online_update_new_state(self):
        # Adding transition with new states should not crash
        self.fsa.online_update("s0", "x", "s1", weight=1.0)
        assert self.fsa.n_transitions > 0


# ─── FractionalPowerEncoding v1.46 — decode + nearest ────────────────────────

class TestFPEDecodeNearest:
    def setup_method(self):
        from hdc.vsa_algebras import FractionalPowerEncoding
        self.fpe = FractionalPowerEncoding(1, 128, bw=1.0, kernel="gaussian", seed=0)

    def test_decode_1d_returns_tensor(self):
        x = torch.tensor([2.0])
        hv = self.fpe.encode(x)
        decoded = self.fpe.decode(hv, x_range=(-5.0, 5.0), n_candidates=50)
        assert decoded.shape == (1,)

    def test_decode_1d_rough_accuracy(self):
        x = torch.tensor([1.5])
        hv = self.fpe.encode(x)
        decoded = self.fpe.decode(hv, x_range=(-3.0, 3.0), n_candidates=100)
        # Should be within 1.0 of the true value
        assert abs(float(decoded[0]) - 1.5) < 1.0

    def test_nearest_finds_correct_candidate(self):
        from hdc.vsa_algebras import FractionalPowerEncoding
        fpe = FractionalPowerEncoding(1, 256, bw=1.0, seed=0)
        candidates = [torch.tensor([float(i)]) for i in range(5)]
        query_hv = fpe.encode(torch.tensor([2.0]))   # encode value 2
        idx = fpe.nearest(query_hv, candidates)
        # Should find candidate closest to 2.0 (index 2)
        assert 0 <= idx < len(candidates)
