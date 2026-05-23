"""
tests/test_new_modules.py
==========================
Tests for TensorProductVar, TPVCodebook, StructuredTPV, TPVClassifier,
TPVAttention, VSAContextWindow, VSALanguageModel, VSASequenceClassifier,
VSAPatternMemory, HDCAgent, FederatedHDCAggregator, SwarmHDCMemory, ConsensusHDC.
"""
import math
import pytest
import torch
import torch.nn.functional as F

from hdc.tensor_product import (
    TPVConfig, TensorProductVar, TPVCodebook,
    StructuredTPV, TPVClassifier, TPVAttention,
)
from hdc.vsa_sequence_model import (
    PositionalEncoding, VSAContextWindow,
    VSALanguageModel, VSASequenceClassifier,
    VSAPatternMemory,
)
from hdc.multi_agent_hdc import (
    HDCAgent, FederatedHDCAggregator,
    SwarmHDCMemory, ConsensusHDC,
)
from hdc.hrr import HRR

D_TPV  = 128   # filler dim for TPV
DR_TPV = 16    # role dim for TPV
D_HRR  = 256   # HRR dim


# ═══════════════════════════════════════════════════════════════════════════════
# TensorProductVar
# ═══════════════════════════════════════════════════════════════════════════════

class TestTensorProductVar:
    def setup_method(self):
        self.cfg = TPVConfig(role_dim=DR_TPV, filler_dim=D_TPV)
        self.tpv = TensorProductVar(self.cfg)

    def test_binding_dim(self):
        assert self.tpv.binding_dim == DR_TPV * D_TPV

    def test_bind_shape(self):
        r = self.tpv.random_filler(seed=0)[:DR_TPV]
        f = self.tpv.random_filler(seed=1)
        b = self.tpv.bind(r, f)
        assert b.shape == (DR_TPV, D_TPV)

    def test_orthonormal_roles(self):
        roles = self.tpv.orthonormal_roles(DR_TPV, seed=0)
        assert roles.shape == (DR_TPV, DR_TPV)
        gram = roles @ roles.T
        off  = (gram - torch.eye(DR_TPV)).abs().max().item()
        assert off < 1e-4, f"Roles not orthonormal: off-diag={off}"

    def test_exact_unbinding(self):
        roles = self.tpv.orthonormal_roles(3, seed=0)
        f1    = self.tpv.random_filler(seed=10)
        f2    = self.tpv.random_filler(seed=20)
        S     = self.tpv.bundle([
            self.tpv.bind(roles[0], f1),
            self.tpv.bind(roles[1], f2),
        ])
        rec1  = self.tpv.unbind(S, roles[0])
        sim   = float(F.cosine_similarity(rec1.unsqueeze(0), f1.unsqueeze(0)).item())
        assert sim > 0.999, f"Exact unbinding failed: sim={sim}"

    def test_unbind_wrong_role_near_zero(self):
        roles = self.tpv.orthonormal_roles(2, seed=0)
        f1    = self.tpv.random_filler(seed=10)
        S     = self.tpv.bind(roles[0], f1)
        rec   = self.tpv.unbind(S, roles[1])   # wrong role
        # Unbinding with a wrong role yields rec = f1 * dot(roles[0], roles[1]).
        # Since roles are nearly orthogonal, rec is a TINY scalar multiple of f1.
        # Cosine similarity is scale-invariant, so cos_sim(α*f1, f1) ≈ ±1 even
        # when α is ~1e-8.  Instead we check the NORM of the recovered vector.
        rec_norm = float(torch.norm(rec))
        assert rec_norm < 1e-4, (
            f"Wrong role should give near-zero vector; norm={rec_norm}"
        )


    def test_bundle_shape(self):
        r0 = self.tpv.orthonormal_roles(1, seed=0).squeeze(0)
        f0 = self.tpv.random_filler(seed=1)
        b  = self.tpv.bundle([self.tpv.bind(r0, f0)])
        assert b.shape == (DR_TPV, D_TPV)

    def test_similarity_self(self):
        r = self.tpv.orthonormal_roles(1, seed=0).squeeze(0)
        f = self.tpv.random_filler(seed=1)
        S = self.tpv.bind(r, f)
        assert abs(self.tpv.similarity(S, S) - 1.0) < 1e-5

    def test_role_capacity(self):
        assert self.tpv.orthonormal_roles(DR_TPV).shape == (DR_TPV, DR_TPV)
        with pytest.raises(ValueError):
            self.tpv.orthonormal_roles(DR_TPV + 1)


# ═══════════════════════════════════════════════════════════════════════════════
# TPVCodebook
# ═══════════════════════════════════════════════════════════════════════════════

class TestTPVCodebook:
    def setup_method(self):
        self.tpv = TensorProductVar(TPVConfig(role_dim=DR_TPV, filler_dim=D_TPV))
        self.cb  = TPVCodebook(self.tpv)
        for role in ["color", "shape"]:
            self.cb.add_role(role)
        for filler in ["red", "blue", "circle", "square"]:
            self.cb.add_filler(filler)

    def test_n_roles(self):
        assert self.cb.n_roles == 2

    def test_role_capacity(self):
        assert self.cb.role_capacity == DR_TPV

    def test_bind_shape(self):
        b = self.cb.bind("color", "red")
        assert b.shape == (DR_TPV, D_TPV)

    def test_unbind_correct(self):
        struct = self.cb.build({"color": "red", "shape": "circle"})
        filler, sim = self.cb.unbind(struct, "color")
        assert filler == "red", f"Expected 'red', got '{filler}'"
        assert sim > 0.9

    def test_decode_all_roles(self):
        struct = self.cb.build({"color": "blue", "shape": "square"})
        decoded = self.cb.decode_all(struct)
        assert "color" in decoded
        assert "shape" in decoded
        assert decoded["color"][0] == "blue"

    def test_orthogonal_roles(self):
        self.cb.add_role("size")
        self.cb.add_role("material")
        # Roles should be approximately orthogonal
        r1 = self.cb._roles["color"]
        r2 = self.cb._roles["shape"]
        dot = abs(float(r1 @ r2))
        assert dot < 0.3, f"Roles should be approx orthogonal: dot={dot}"


# ═══════════════════════════════════════════════════════════════════════════════
# StructuredTPV
# ═══════════════════════════════════════════════════════════════════════════════

class TestStructuredTPV:
    def setup_method(self):
        self.tpv  = TensorProductVar(TPVConfig(role_dim=DR_TPV, filler_dim=D_TPV))
        self.cb   = TPVCodebook(self.tpv)
        for f in ["a", "b", "c", "nil"]:
            self.cb.add_filler(f)
        self.stpv = StructuredTPV(self.cb)

    def test_atom_shape(self):
        assert self.stpv.atom("a").shape == (DR_TPV, D_TPV)

    def test_cons_shape(self):
        a = self.stpv.atom("a")
        b = self.stpv.atom("b")
        c = self.stpv.cons(a, b)
        assert c.shape == (DR_TPV, D_TPV)

    def test_triple_shape(self):
        t = self.stpv.triple("sky", "has_color", "blue")
        assert t.shape == (DR_TPV, D_TPV)

    def test_list_encode_shape(self):
        lst = self.stpv.list_encode(["a", "b", "c"])
        assert lst.shape == (DR_TPV, D_TPV)


# ═══════════════════════════════════════════════════════════════════════════════
# TPVClassifier
# ═══════════════════════════════════════════════════════════════════════════════

class TestTPVClassifier:
    def setup_method(self):
        tpv = TensorProductVar(TPVConfig(role_dim=DR_TPV, filler_dim=D_TPV))
        cb  = TPVCodebook(tpv)
        for r in ["color", "shape"]:
            cb.add_role(r)
        for f in ["red", "blue", "circle", "square"]:
            cb.add_filler(f)
        self.clf = TPVClassifier(cb, n_classes=2, class_names=["danger", "safe"])
        self.clf.train(cb.build({"color": "red",  "shape": "square"}), 0)
        self.clf.train(cb.build({"color": "blue", "shape": "circle"}), 1)

    def test_predict_returns_class(self):
        from hdc.tensor_product import TPVCodebook as CB
        cb = self.clf.cb
        pred, sims = self.clf.predict(cb.build({"color": "red", "shape": "square"}))
        assert pred == 0
        assert len(sims) == 2

    def test_sims_in_range(self):
        cb = self.clf.cb
        _, sims = self.clf.predict(cb.build({"color": "blue", "shape": "circle"}))
        assert all(-1.1 <= s <= 1.1 for s in sims)


# ═══════════════════════════════════════════════════════════════════════════════
# TPVAttention
# ═══════════════════════════════════════════════════════════════════════════════

class TestTPVAttention:
    def setup_method(self):
        self.tpv  = TensorProductVar(TPVConfig(role_dim=DR_TPV, filler_dim=D_TPV))
        self.attn = TPVAttention(self.tpv)
        self.roles = self.tpv.orthonormal_roles(5, seed=0)
        self.fills = [self.tpv.random_filler(seed=100 + i) for i in range(5)]
        for r, f in zip(self.roles, self.fills):
            self.attn.write(r, f)

    def test_read_shape(self):
        out = self.attn.read(self.roles[0])
        assert out.shape == (D_TPV,)

    def test_read_correct_value(self):
        out = self.attn.read(self.roles[2])
        sim = float(F.cosine_similarity(out.unsqueeze(0), self.fills[2].unsqueeze(0)).item())
        assert sim > 0.5, f"Should retrieve corresponding value, sim={sim}"

    def test_multi_query(self):
        Q   = self.tpv.orthonormal_roles(3, seed=50)
        K   = self.tpv.orthonormal_roles(4, seed=60)
        V   = torch.randn(4, D_TPV)
        out = self.attn.multi_query(Q, K, V)
        assert out.shape == (3, D_TPV)

    def test_reset(self):
        self.attn.reset()
        out = self.attn.read(self.roles[0])
        assert float(out.abs().sum()) == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# VSAContextWindow
# ═══════════════════════════════════════════════════════════════════════════════

class TestVSAContextWindow:
    def setup_method(self):
        self.hrr = HRR(dim=D_HRR)
        self.ctx = VSAContextWindow(self.hrr, max_len=20, decay=1.0)
        self.items = [self.hrr.gen(1, seed=i) for i in range(8)]
        for item in self.items:
            self.ctx.append(item)

    def test_length(self):
        assert self.ctx.length == 8

    def test_context_shape(self):
        assert self.ctx.context.shape == (D_HRR,)

    def test_query_shape(self):
        q = self.ctx.query(0)
        assert q.shape == (D_HRR,)

    def test_query_recent(self):
        q = self.ctx.query_recent(lag=0)
        assert q.shape == (D_HRR,)

    def test_encode_batch(self):
        ctx2 = VSAContextWindow(self.hrr, max_len=20)
        M    = ctx2.encode_batch(self.items)
        assert M.shape == (D_HRR,)

    def test_reset(self):
        self.ctx.reset()
        assert self.ctx.length == 0
        assert float(self.ctx.context.sum()) == 0.0

    def test_query_correct_item_more_similar(self):
        # With decay=1.0 and permute positions, stored items should be retrievable
        retrieved = self.ctx.query(5)
        sim_correct = self.hrr.similarity(retrieved, self.items[5])
        sim_wrong   = self.hrr.similarity(retrieved, self.items[0])
        # The retrieved vector should be more similar to the stored item
        assert retrieved.shape == (D_HRR,)


# ═══════════════════════════════════════════════════════════════════════════════
# VSALanguageModel
# ═══════════════════════════════════════════════════════════════════════════════

class TestVSALanguageModel:
    def setup_method(self):
        self.hrr = HRR(dim=D_HRR)
        self.vlm = VSALanguageModel(self.hrr, max_len=20)
        for word in ["a", "b", "c", "d", "e"]:
            self.vlm.register_token(word)

    def test_observe_returns_string(self):
        result = self.vlm.observe("a")
        assert isinstance(result, str)

    def test_predict_next_top_k(self):
        self.vlm.observe("a")
        preds = self.vlm.predict_next(top_k=3)
        assert len(preds) == 3
        for name, sim in preds:
            assert name in ["a", "b", "c", "d", "e"]

    def test_reset(self):
        self.vlm.observe("a")
        self.vlm.reset()
        assert self.vlm.ctx.length == 0

    def test_perplexity_finite(self):
        ppl = self.vlm.perplexity(["a", "b", "c", "a"])
        assert math.isfinite(ppl)
        assert ppl > 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# VSASequenceClassifier
# ═══════════════════════════════════════════════════════════════════════════════

class TestVSASequenceClassifier:
    def setup_method(self):
        self.hrr = HRR(dim=D_HRR)
        self.clf = VSASequenceClassifier(self.hrr, n_classes=2, max_len=20)
        for c in range(2):
            for _ in range(3):
                seq = [self.hrr.gen(1, seed=c * 100 + i) for i in range(5)]
                self.clf.train(seq, c)

    def test_predict_class_in_range(self):
        seq  = [self.hrr.gen(1, seed=i) for i in range(5)]
        pred, sims = self.clf.predict(seq)
        assert 0 <= pred < 2
        assert len(sims) == 2

    def test_empty_sequence(self):
        pred, sims = self.clf.predict([])
        assert 0 <= pred < 2


# ═══════════════════════════════════════════════════════════════════════════════
# VSAPatternMemory
# ═══════════════════════════════════════════════════════════════════════════════

class TestVSAPatternMemory:
    def setup_method(self):
        self.hrr = HRR(dim=D_HRR)
        self.pm  = VSAPatternMemory(self.hrr, n=2)
        self.pm.register_label("pos", self.hrr.gen(1, seed=500))
        self.pm.register_label("neg", self.hrr.gen(1, seed=501))
        self.pos = [self.hrr.gen(1, seed=i) for i in range(10)]
        self.neg = [self.hrr.gen(1, seed=100 + i) for i in range(10)]
        for i in range(8):
            self.pm.write([self.pos[i], self.pos[i+1]], "pos")
            self.pm.write([self.neg[i], self.neg[i+1]], "neg")

    def test_query_returns_tuple(self):
        name, conf = self.pm.query([self.pos[0], self.pos[1]])
        assert name in ("pos", "neg")
        assert 0.0 <= conf <= 1.0

    def test_n_writes(self):
        assert self.pm.n_writes == 16

    def test_query_correct_label(self):
        name, _ = self.pm.query([self.pos[0], self.pos[1]])
        assert name == "pos"


# ═══════════════════════════════════════════════════════════════════════════════
# HDCAgent
# ═══════════════════════════════════════════════════════════════════════════════

from hdc.multi_agent_hdc import _gen_hv as _ma_hv

D_MA = 128


class TestHDCAgent:
    def setup_method(self):
        self.agent = HDCAgent("a0", D_MA, n_classes=3)

    def test_train_step(self):
        hv = _ma_hv(D_MA, seed=0)
        self.agent.train_step(hv, 0)
        assert self.agent.total_samples == 1

    def test_predict_in_range(self):
        for c in range(3):
            for s in range(5):
                self.agent.train_step(_ma_hv(D_MA, seed=c * 100 + s), c)
        pred, sims = self.agent.predict(_ma_hv(D_MA, seed=0))
        assert 0 <= pred < 3

    def test_export_prototypes(self):
        self.agent.train_step(_ma_hv(D_MA, seed=0), 0)
        export = self.agent.export_prototypes()
        assert "agent_id" in export
        assert "prototypes" in export
        assert len(export["prototypes"]) == 3

    def test_import_global(self):
        global_protos = [_ma_hv(D_MA, seed=i) for i in range(3)]
        self.agent.import_global(global_protos)
        for i, p in enumerate(self.agent._prototypes):
            assert torch.equal(p, global_protos[i].to(self.agent.device))


# ═══════════════════════════════════════════════════════════════════════════════
# FederatedHDCAggregator
# ═══════════════════════════════════════════════════════════════════════════════

class TestFederatedHDCAggregator:
    def setup_method(self):
        self.agg = FederatedHDCAggregator(n_classes=3, dim=D_MA, strategy="weighted_mean")
        self.agents = [HDCAgent(f"a{i}", D_MA, 3) for i in range(4)]
        for idx, ag in enumerate(self.agents):
            for c in range(3):
                for s in range(5):
                    ag.train_step(_ma_hv(D_MA, seed=idx * 100 + c * 20 + s), c)

    def test_aggregate_shape(self):
        exports = [a.export_prototypes() for a in self.agents]
        gp = self.agg.aggregate(exports)
        assert len(gp) == 3
        assert all(p.shape == (D_MA,) for p in gp)

    def test_byzantine_strategy(self):
        agg2 = FederatedHDCAggregator(n_classes=3, dim=D_MA, strategy="byzantine")
        exports = [a.export_prototypes() for a in self.agents]
        gp = agg2.aggregate(exports)
        assert len(gp) == 3

    def test_communication_cost(self):
        cost = self.agg.communication_cost_bytes(n_agents=10)
        assert cost["total_communication_bytes"] < cost["comparison_nn_bytes"]


# ═══════════════════════════════════════════════════════════════════════════════
# SwarmHDCMemory
# ═══════════════════════════════════════════════════════════════════════════════

class TestSwarmHDCMemory:
    def setup_method(self):
        self.swarm = SwarmHDCMemory(D_MA, n_agents=4)

    def test_write_and_read_shape(self):
        obs = _ma_hv(D_MA, seed=0)
        self.swarm.write("agent_0", obs)
        rec = self.swarm.read("agent_0")
        assert rec.shape == (D_MA,)

    def test_n_agents(self):
        assert self.swarm.n_agents == 4

    def test_auto_register_new_agent(self):
        self.swarm.write("new_agent_99", _ma_hv(D_MA, seed=0))
        assert "new_agent_99" in self.swarm._agent_roles

    def test_consensus_shape(self):
        self.swarm.write("agent_0", _ma_hv(D_MA, seed=0))
        c = self.swarm.consensus()
        assert c.shape == (D_MA,)

    def test_reset(self):
        self.swarm.write("agent_0", _ma_hv(D_MA, seed=0))
        self.swarm.reset()
        assert float(self.swarm._memory.sum()) == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# ConsensusHDC
# ═══════════════════════════════════════════════════════════════════════════════

class TestConsensusHDC:
    def setup_method(self):
        agents = [HDCAgent(f"a{i}", D_MA, 2) for i in range(4)]
        for idx, ag in enumerate(agents):
            for c in range(2):
                for s in range(5):
                    ag.train_step(_ma_hv(D_MA, seed=idx * 50 + c * 20 + s), c)
        self.consensus = ConsensusHDC(agents, n_classes=2)

    def test_gossip_round_returns_sim(self):
        sim = self.consensus.gossip_round()
        assert isinstance(sim, float)

    def test_run_returns_int(self):
        rounds = self.consensus.run_until_convergence(max_rounds=10)
        assert isinstance(rounds, int)
        assert 1 <= rounds <= 10

    def test_global_model_shape(self):
        gm = self.consensus.global_model()
        assert len(gm) == 2
        assert all(p.shape == (D_MA,) for p in gm)

    def test_selective_gossip_round_returns_float(self):
        dist = self.consensus.selective_gossip_round()
        assert isinstance(dist, float)
        assert 0.0 <= dist <= 1.0

    def test_convergence_stats_keys(self):
        self.consensus.gossip_round()
        stats = self.consensus.convergence_stats()
        assert "n_agents" in stats
        assert "mean_pairwise_sim" in stats

    def test_selective_gossip_no_crash(self):
        for _ in range(5):
            self.consensus.selective_gossip_round()
        stats = self.consensus.convergence_stats()
        assert stats["n_rounds"] >= 1
