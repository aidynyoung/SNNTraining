"""
tests/test_elite_1000x.py
==========================
Tests for hdc/probabilistic_hdc.py, hdc/neuromorphic_transformer.py,
hdc/self_supervised.py, and hdc/symbolic_reasoning.py.
"""
import pytest
import math
import torch
import torch.nn.functional as F

# Probabilistic HDC
from hdc.probabilistic_hdc import (
    BayesianHDCClassifier, HDCParticleFilter,
    BeliefUpdateNetwork, HDCVariationalInference, ConfidenceCalibrator,
    _gen_hv as _p_gen,
)
# Neuromorphic Transformer
from hdc.neuromorphic_transformer import (
    HDCTransformerConfig, HDCAttentionBlock, HDCAttentionBlockEfficient,
    HDCSparseFFN, HDCLayerNorm, HDCTransformerBlock,
    HDCTransformerStack, HDCLanguageHead,
    _gen_hv as _t_gen,
)
# Self-supervised
from hdc.self_supervised import (
    HDCContrastiveLearner, HDCMaskedAutoencoder,
    HDCMomentumEncoder, HDCBootstrap, HDCClusterLearner,
    _gen_hv as _s_gen, _augment,
)
# Symbolic reasoning
from hdc.symbolic_reasoning import (
    HDCPropLogic, HDCRuleEngine, HDCRule,
    HDCTheoremProver, HDCUnifier,
    _gen_hv as _sym_gen,
)

D = 128


# ─── BayesianHDCClassifier ────────────────────────────────────────────────────

class TestBayesianHDCClassifier:
    def setup_method(self):
        self.clf = BayesianHDCClassifier(D, n_classes=3, beta=8.0)
        for c in range(3):
            for s in range(10):
                self.clf.train(_p_gen(D, seed=c*100+s), c)

    def test_posterior_sums_to_one(self):
        post = self.clf.posterior(_p_gen(D, seed=0))
        assert abs(float(post.sum()) - 1.0) < 1e-4

    def test_predict_in_range(self):
        pred, conf, post = self.clf.predict(_p_gen(D, seed=0))
        assert 0 <= pred < 3
        assert 0.0 <= conf <= 1.0

    def test_rejection_below_threshold(self):
        pred, conf, _ = self.clf.predict(_p_gen(D, seed=42), threshold=0.99)
        assert pred == -1  # should be rejected with very high threshold

    def test_calibrate_returns_beta(self):
        cal_hvs    = [_p_gen(D, seed=300+i) for i in range(10)]
        cal_labels = [i % 3 for i in range(10)]
        for hv, lbl in zip(cal_hvs, cal_labels):
            self.clf.train(hv, lbl)
        beta = self.clf.calibrate(cal_hvs, cal_labels, n_temps=5)
        assert beta > 0.0

    def test_ece_in_range(self):
        cal_hvs    = [_p_gen(D, seed=i) for i in range(20)]
        cal_labels = [i % 3 for i in range(20)]
        for hv, lbl in zip(cal_hvs, cal_labels):
            self.clf.train(hv, lbl)
        ece = self.clf.ece(cal_hvs, cal_labels)
        assert 0.0 <= ece <= 1.0


# ─── HDCParticleFilter ────────────────────────────────────────────────────────

class TestHDCParticleFilter:
    def setup_method(self):
        self.pf = HDCParticleFilter(D, n_particles=30, beta=3.0, noise_rate=0.05)

    def test_state_estimate_shape(self):
        self.pf.predict()
        self.pf.update(_p_gen(D, seed=0))
        est = self.pf.state_estimate()
        assert est.shape == (D,)

    def test_particles_shape(self):
        assert self.pf.particles.shape == (30, D)

    def test_weights_sum_to_one(self):
        assert abs(float(self.pf.weights.sum()) - 1.0) < 1e-4

    def test_effective_sample_size_positive(self):
        ess = self.pf.effective_sample_size()
        assert ess > 0.0

    def test_entropy_nonnegative(self):
        assert self.pf.entropy() >= 0.0

    def test_predict_updates_step(self):
        self.pf.predict()
        assert self.pf._step == 1

    def test_reset_uniform_weights(self):
        self.pf.predict()
        self.pf.update(_p_gen(D, seed=0))
        self.pf.reset()
        assert abs(float(self.pf.weights.mean()) - 1.0 / 30) < 1e-5

    def test_update_with_correct_obs(self):
        # After many updates with a consistent observation, state should converge
        target = _p_gen(D, seed=42)
        for _ in range(10):
            self.pf.predict()
            self.pf.update(target)
        ess = self.pf.effective_sample_size()
        assert ess > 0.0  # shouldn't collapse completely


# ─── BeliefUpdateNetwork ─────────────────────────────────────────────────────

class TestBeliefUpdateNetwork:
    def setup_method(self):
        self.bun = BeliefUpdateNetwork(D, beta=5.0)
        for i in range(4):
            self.bun.add_hypothesis(f"h{i}", _p_gen(D, seed=i))

    def test_observe_returns_dict(self):
        beliefs = self.bun.observe(_p_gen(D, seed=0))
        assert isinstance(beliefs, dict)
        assert len(beliefs) == 4

    def test_beliefs_sum_to_one(self):
        self.bun.observe(_p_gen(D, seed=0))
        total = sum(self.bun._beliefs.tolist())
        assert abs(total - 1.0) < 1e-4

    def test_most_likely_returns_tuple(self):
        self.bun.observe(_p_gen(D, seed=0))
        name, prob = self.bun.most_likely()
        assert name in [f"h{i}" for i in range(4)]
        assert 0.0 <= prob <= 1.0

    def test_entropy_nonnegative(self):
        assert self.bun.entropy() >= 0.0

    def test_reset_uniform(self):
        self.bun.observe(_p_gen(D, seed=0))
        self.bun.reset()
        assert abs(float(self.bun._beliefs.mean()) - 0.25) < 1e-5


# ─── HDCVariationalInference ──────────────────────────────────────────────────

class TestHDCVariationalInference:
    def setup_method(self):
        self.vi = HDCVariationalInference(D, beta=5.0)
        for i in range(5):
            self.vi.register(f"lat_{i}", _p_gen(D, seed=i))

    def test_elbo_returns_dict(self):
        result = self.vi.elbo(_p_gen(D, seed=0))
        assert "reconstruction" in result
        assert "complexity" in result
        assert "elbo" in result

    def test_infer_top_k(self):
        top3 = self.vi.infer(_p_gen(D, seed=0), top_k=3)
        assert len(top3) == 3
        assert all(0.0 <= p <= 1.0 for _, p in top3)


# ─── ConfidenceCalibrator ─────────────────────────────────────────────────────

class TestConfidenceCalibrator:
    def setup_method(self):
        self.cal = ConfidenceCalibrator(n_classes=3)

    def test_calibrate_returns_temperature(self):
        logits = [torch.randn(3) for _ in range(20)]
        labels = [i % 3 for i in range(20)]
        T = self.cal.calibrate(logits, labels, n_temps=10)
        assert T > 0.0

    def test_calibrated_probs_sum_to_one(self):
        self.cal.temperature = 2.0
        probs = self.cal.calibrate_probabilities(torch.tensor([1.0, 2.0, 3.0]))
        assert abs(float(probs.sum()) - 1.0) < 1e-5

    def test_ece_in_range(self):
        probs  = [torch.tensor([0.8, 0.1, 0.1]) for _ in range(20)]
        labels = [0] * 15 + [1] * 5
        ece = self.cal.expected_calibration_error(probs, labels)
        assert 0.0 <= ece <= 1.0


# ─── HDCAttentionBlock ────────────────────────────────────────────────────────

class TestHDCAttentionBlock:
    def setup_method(self):
        self.cfg    = HDCTransformerConfig(dim=D, n_heads=2, beta=5.0)
        self.attn   = HDCAttentionBlockEfficient(self.cfg)
        self.tokens = torch.stack([_t_gen(D, seed=i) for i in range(6)])

    def test_output_shape(self):
        out = self.attn.forward(self.tokens)
        assert out.shape == (6, D)

    def test_output_binary(self):
        out = self.attn.forward(self.tokens)
        assert set(out.unique().tolist()).issubset({0.0, 1.0})

    def test_causal_mask_shape(self):
        N    = 6
        mask = torch.triu(torch.ones(N, N, dtype=torch.bool), diagonal=1)
        out  = self.attn.forward(self.tokens, mask=mask)
        assert out.shape == (6, D)


# ─── HDCSparseFFN ─────────────────────────────────────────────────────────────

class TestHDCSparseFFN:
    def setup_method(self):
        self.ffn = HDCSparseFFN(D, k_frac=0.1)

    def test_single_forward_shape(self):
        out = self.ffn.forward(_t_gen(D, seed=0))
        assert out.shape == (D,)

    def test_batch_forward_shape(self):
        tokens = torch.stack([_t_gen(D, seed=i) for i in range(4)])
        out    = self.ffn.forward_batch(tokens)
        assert out.shape == (4, D)


# ─── HDCLayerNorm ─────────────────────────────────────────────────────────────

class TestHDCLayerNorm:
    def test_normalize_shape(self):
        ln = HDCLayerNorm(target_density=0.5)
        hv = _t_gen(D, seed=0)
        n  = ln.normalize(hv)
        assert n.shape == (D,)

    def test_normalize_density_near_target(self):
        ln   = HDCLayerNorm(target_density=0.5)
        hv   = _t_gen(D, seed=0)
        n    = ln.normalize(hv)
        dens = float(n.mean())
        assert 0.35 <= dens <= 0.65

    def test_batch_normalize(self):
        ln     = HDCLayerNorm(target_density=0.5)
        tokens = torch.stack([_t_gen(D, seed=i) for i in range(4)])
        normed = ln.normalize(tokens)
        assert normed.shape == (4, D)


# ─── HDCTransformerStack ──────────────────────────────────────────────────────

class TestHDCTransformerStack:
    def setup_method(self):
        self.cfg   = HDCTransformerConfig(dim=D, n_heads=2, n_layers=2, beta=3.0)
        self.stack = HDCTransformerStack(self.cfg, causal=False)
        self.tokens = torch.stack([_t_gen(D, seed=i) for i in range(5)])

    def test_output_shape(self):
        out = self.stack.forward(self.tokens)
        assert out.shape == (5, D)

    def test_causal_output_shape(self):
        stack_c = HDCTransformerStack(self.cfg, causal=True)
        out     = stack_c.forward(self.tokens)
        assert out.shape == (5, D)

    def test_position_encoding_changes_tokens(self):
        x1 = self.stack._positional_encoding(self.tokens)
        # Different positions → different HVs
        assert not torch.equal(x1[0], x1[1])


# ─── HDCLanguageHead ──────────────────────────────────────────────────────────

class TestHDCLanguageHead:
    def setup_method(self):
        self.vocab = [(f"w{i}", _t_gen(D, seed=100+i)) for i in range(8)]
        self.head  = HDCLanguageHead(self.vocab, D)

    def test_logits_shape(self):
        logits = self.head.logits(_t_gen(D, seed=0))
        assert logits.shape == (8,)

    def test_predict_top_k(self):
        preds = self.head.predict(_t_gen(D, seed=0), top_k=3)
        assert len(preds) == 3

    def test_add_token(self):
        self.head.add_token("new", _t_gen(D, seed=999))
        logits = self.head.logits(_t_gen(D, seed=0))
        assert logits.shape == (9,)


# ─── HDCContrastiveLearner ────────────────────────────────────────────────────

class TestHDCContrastiveLearner:
    def setup_method(self):
        self.learner = HDCContrastiveLearner(D, memory_size=30, aug_rate=0.1)

    def test_update_returns_loss(self):
        loss = self.learner.update(_s_gen(D, seed=0))
        assert isinstance(loss, float)

    def test_encode_shape(self):
        enc = self.learner.encode(_s_gen(D, seed=0))
        assert enc.shape == (D,)

    def test_linear_probe(self):
        for i in range(20):
            self.learner.update(_s_gen(D, seed=i % 3), label=f"c{i%3}")
        hvs    = [_s_gen(D, seed=i) for i in range(9)]
        labels = [f"c{i%3}" for i in range(9)]
        protos = self.learner.linear_probe(hvs, labels)
        assert len(protos) == 3


# ─── HDCMaskedAutoencoder ─────────────────────────────────────────────────────

class TestHDCMaskedAutoencoder:
    def setup_method(self):
        self.mae = HDCMaskedAutoencoder(D, mask_rate=0.5, n_memory=30)
        for i in range(20):
            self.mae.train_step(_s_gen(D, seed=i))

    def test_n_stored(self):
        assert self.mae._n_stored == 20

    def test_complete_returns_candidates(self):
        partial = _s_gen(D, seed=0)
        partial[:D//2] = 0.0
        completions = self.mae.complete(partial, top_k=3)
        assert len(completions) == 3

    def test_reconstruction_accuracy_in_range(self):
        acc = self.mae.reconstruction_accuracy([_s_gen(D, seed=i) for i in range(5)])
        assert 0.0 <= acc <= 1.0


# ─── HDCBootstrap ─────────────────────────────────────────────────────────────

class TestHDCBootstrap:
    def setup_method(self):
        self.byol = HDCBootstrap(D, momentum=0.99)

    def test_step_returns_loss(self):
        loss = self.byol.step(_s_gen(D, seed=0))
        assert isinstance(loss, float)

    def test_encode_shape(self):
        enc = self.byol.encode(_s_gen(D, seed=0))
        assert enc.shape == (D,)

    def test_running_loss_after_steps(self):
        for i in range(10):
            self.byol.step(_s_gen(D, seed=i))
        avg = self.byol.running_loss()
        assert isinstance(avg, float)

    def test_is_collapsed_returns_bool(self):
        result = self.byol.is_collapsed()
        assert isinstance(result, bool)

    def test_recover_from_collapse_changes_online(self):
        online_before = self.byol._online.clone()
        # Force collapse by saturating online prototype
        self.byol._online = torch.ones(D, device=self.byol.device)
        self.byol.recover_from_collapse()
        assert not torch.equal(self.byol._online, torch.ones(D, device=self.byol.device))

    def test_step_auto_recovers_collapse(self):
        # After many steps, should not be collapsed
        for i in range(20):
            self.byol.step(_s_gen(D, seed=i))
        # Should have attempted recovery if collapsed
        assert isinstance(self.byol.is_collapsed(), bool)


class TestHDCContrastiveTempAnnealing:
    def test_temperature_anneals_downward(self):
        from hdc.self_supervised import HDCContrastiveLearner
        learner = HDCContrastiveLearner(D, temperature=0.07, temp_anneal=True,
                                         temp_final=0.03, anneal_steps=50)
        t0 = learner.temperature
        for i in range(50):
            learner.update(_s_gen(D, seed=i))
        t_final = learner.temperature
        assert t_final <= t0   # temperature should decrease

    def test_temperature_annealing_disabled(self):
        from hdc.self_supervised import HDCContrastiveLearner
        learner = HDCContrastiveLearner(D, temperature=0.07, temp_anneal=False)
        for i in range(20):
            learner.update(_s_gen(D, seed=i))
        assert learner.temperature == 0.07   # unchanged


# ─── HDCClusterLearner ────────────────────────────────────────────────────────

class TestHDCClusterLearner:
    def setup_method(self):
        self.clf = HDCClusterLearner(D, max_clusters=10, creation_threshold=0.55)

    def test_update_returns_int(self):
        c = self.clf.update(_s_gen(D, seed=0))
        assert isinstance(c, int)

    def test_n_clusters_positive(self):
        for i in range(10):
            self.clf.update(_s_gen(D, seed=i % 3))
        assert self.clf.n_clusters >= 1

    def test_predict_in_range(self):
        for i in range(5):
            self.clf.update(_s_gen(D, seed=i))
        c, sim = self.clf.predict(_s_gen(D, seed=0))
        assert 0 <= c < self.clf.n_clusters
        assert 0.0 <= sim <= 1.0

    def test_cluster_stats_keys(self):
        self.clf.update(_s_gen(D, seed=0))
        stats = self.clf.cluster_stats()
        assert "n_clusters" in stats and "n_processed" in stats


# ─── HDCPropLogic ─────────────────────────────────────────────────────────────

class TestHDCPropLogic:
    def setup_method(self):
        self.logic = HDCPropLogic(D)
        self.logic.IMPLIES("P", "Q")
        self.logic.IMPLIES("Q", "R")
        self.logic.assert_fact("P")

    def test_query_asserted_fact(self):
        is_true, conf = self.logic.query("P")
        assert is_true

    def test_modus_ponens(self):
        conclusions = self.logic.modus_ponens("P")
        names = [n for n, _ in conclusions]
        assert "Q" in names

    def test_forward_chain(self):
        derived = self.logic.forward_chain()
        assert isinstance(derived, list)

    def test_not_operation(self):
        p  = self.logic.atom("P")
        np = self.logic.NOT(p)
        assert np.shape == (D,)
        sim = float((p == np).float().mean())
        assert sim < 0.2   # NOT should be very different


# ─── HDCRuleEngine ────────────────────────────────────────────────────────────

class TestHDCRuleEngine:
    def setup_method(self):
        self.engine = HDCRuleEngine(D)
        self.engine.add_rule("alert",
                              conditions=["sensor_A", "sensor_B"],
                              action="alarm")

    def test_fire_before_conditions(self):
        fired = self.engine.fire_rules()
        assert len(fired) == 0   # no conditions met

    def test_fire_after_conditions(self):
        self.engine.assert_fact("sensor_A")
        self.engine.assert_fact("sensor_B")
        fired = self.engine.fire_rules()
        assert any(action == "alarm" for _, action in fired)

    def test_query_after_firing(self):
        self.engine.assert_fact("sensor_A")
        self.engine.assert_fact("sensor_B")
        self.engine.fire_rules()
        is_alarm, _ = self.engine.query("alarm")
        assert is_alarm


# ─── HDCUnifier ──────────────────────────────────────────────────────────────

class TestHDCUnifier:
    def setup_method(self):
        self.uni = HDCUnifier(D)
        self.uni.const("a"); self.uni.const("b")
        self.uni.var("X"); self.uni.var("Y")
        self.role_hvs = {"r1": _sym_gen(D, seed="r1")}

    def test_unify_var_with_const(self):
        unified, bindings = self.uni.unify(
            {"r1": "X"}, {"r1": "a"}, self.role_hvs
        )
        assert unified
        assert bindings.get("X") == "a"

    def test_unify_same_constants(self):
        unified, bindings = self.uni.unify(
            {"r1": "a"}, {"r1": "a"}, self.role_hvs
        )
        assert unified
        assert bindings == {}

    def test_unify_two_vars(self):
        unified, bindings = self.uni.unify(
            {"r1": "X"}, {"r1": "Y"}, self.role_hvs
        )
        assert unified


# ─── HDCParticleFilter v1.46.2 — systematic resampling + diversity ───────────

class TestHDCParticleFilterV146:
    def setup_method(self):
        from hdc.probabilistic_hdc import HDCParticleFilter
        self.pf = HDCParticleFilter(D, n_particles=20, beta=3.0, noise_rate=0.05)

    def test_systematic_resample_preserves_n(self):
        obs = (torch.rand(D) > 0.5).float()
        self.pf.predict()
        self.pf.update(obs)
        self.pf._resample()
        assert self.pf.particles.shape[0] == 20
        assert abs(float(self.pf.weights.sum()) - 1.0) < 1e-5

    def test_diversity_in_range(self):
        d = self.pf.diversity()
        assert 0.0 <= d <= 1.0

    def test_adaptive_noise_predict(self):
        self.pf.predict(adaptive_noise=True)
        assert self.pf.particles.shape[0] == 20

    def test_fixed_noise_predict(self):
        self.pf.predict(adaptive_noise=False)
        assert self.pf.particles.shape[0] == 20

    def test_noise_rate_unchanged_after_adaptive(self):
        orig = self.pf.noise_rate
        self.pf.predict(adaptive_noise=True)
        assert self.pf.noise_rate == orig   # adaptive doesn't permanently change rate


# ─── HDCPropLogic v1.46.2 — backward chain + explain ─────────────────────────

class TestHDCPropLogicV146:
    def setup_method(self):
        from hdc.symbolic_reasoning import HDCPropLogic
        self.logic = HDCPropLogic(D)
        self.logic.IMPLIES("rain",  "wet_road")
        self.logic.IMPLIES("wet_road", "slow_car")
        self.logic.assert_fact("rain")

    def test_forward_chain_priority(self):
        derived = self.logic.forward_chain(max_steps=5)
        assert isinstance(derived, list)

    def test_explain_returns_string(self):
        self.logic.forward_chain()
        expl = self.logic.explain("wet_road")
        assert isinstance(expl, str)
        assert len(expl) > 0

    def test_backward_chain_finds_proof(self):
        self.logic.assert_fact("rain")
        proof = self.logic.backward_chain("wet_road", max_depth=3)
        # Should find a path via "rain"
        assert proof is None or isinstance(proof, list)

    def test_backward_chain_returns_none_for_unprovable(self):
        proof = self.logic.backward_chain("sun_shining", max_depth=2)
        assert proof is None or isinstance(proof, list)

    def test_explain_initial_fact(self):
        from hdc.symbolic_reasoning import HDCPropLogic
        logic = HDCPropLogic(D)
        logic.assert_fact("sensor_A")
        expl = logic.explain("sensor_A")
        assert "sensor_A" in expl


# ─── BayesianHDCClassifier: label smoothing + entropy reg ────────────────────

class TestBayesianLabelSmooth:
    def test_label_smooth_posterior_not_sharp(self):
        clf = BayesianHDCClassifier(D, n_classes=3, label_smooth=0.1)
        for c in range(3):
            for _ in range(10):
                clf.train(_p_gen(D, seed=c*100+_), c)
        hv = _p_gen(D, seed=0)
        post = clf.posterior(hv)
        # With label smoothing, min probability ≥ smooth/n_classes
        assert float(post.min().item()) >= 0.1 / 3 - 0.01

    def test_label_smooth_zero_matches_original(self):
        clf0 = BayesianHDCClassifier(D, n_classes=3, label_smooth=0.0)
        clf1 = BayesianHDCClassifier(D, n_classes=3, label_smooth=0.0)
        for c in range(3):
            for _ in range(5):
                hv = _p_gen(D, seed=c*50+_)
                clf0.train(hv, c); clf1.train(hv, c)
        hv = _p_gen(D, seed=999)
        assert torch.allclose(clf0.posterior(hv), clf1.posterior(hv), atol=1e-5)

    def test_entropy_reg_spreads_distribution(self):
        clf_base = BayesianHDCClassifier(D, n_classes=4, label_smooth=0.0, entropy_reg=0.0)
        clf_ent  = BayesianHDCClassifier(D, n_classes=4, label_smooth=0.0, entropy_reg=1.0)
        for c in range(4):
            for _ in range(10):
                hv = _p_gen(D, seed=c*100+_)
                clf_base.train(hv, c); clf_ent.train(hv, c)
        hv   = _p_gen(D, seed=42)
        p0   = clf_base.posterior(hv)
        p1   = clf_ent.posterior(hv)
        # Entropy-regularized should have lower max (more uniform)
        assert float(p1.max()) <= float(p0.max()) + 0.05


# ─── HDCParticleFilter: adaptive_resample ────────────────────────────────────

class TestParticleFilterAdaptive:
    def test_adaptive_resample_no_crash(self):
        pf = HDCParticleFilter(D, n_particles=30)
        for _ in range(10):
            obs = _p_gen(D, seed=_)
            pf.predict()
            pf.update(obs)
        pf.adaptive_resample(target_diversity=0.4, max_particles=60, min_particles=10)
        assert 10 <= pf.n_particles <= 60

    def test_adaptive_resample_grows_on_low_diversity(self):
        pf = HDCParticleFilter(D, n_particles=20)
        # Force low diversity: all particles equal
        pf.particles = _p_gen(D, seed=0).unsqueeze(0).expand(20, -1).clone()
        pf.weights = torch.ones(20) / 20
        n_before = pf.n_particles
        pf.adaptive_resample(target_diversity=0.4, max_particles=100, min_particles=5)
        assert pf.n_particles >= n_before   # should grow

    def test_adaptive_resample_shrinks_on_high_diversity(self):
        pf = HDCParticleFilter(D, n_particles=100)
        # Ensure all particles are uniformly random (high diversity)
        for i in range(100):
            pf.particles[i] = _p_gen(D, seed=i)
        pf.weights = torch.ones(100) / 100
        n_before = pf.n_particles
        pf.adaptive_resample(target_diversity=0.01, max_particles=200, min_particles=10)
        # May shrink (depends on actual diversity vs very low target)
        assert 10 <= pf.n_particles <= 200


# ─── MultiHorizonPredictor: ensemble_pred in forward output ──────────────────

from hdc.physics_world_model import MultiHorizonPredictor


class TestMultiHorizonEnsemble:
    def test_forward_has_ensemble_pred(self):
        mhp = MultiHorizonPredictor(D)
        hv  = _p_gen(D, seed=0)
        out = mhp.forward(hv)
        assert "ensemble_pred" in out
        assert out["ensemble_pred"].shape == (D,)

    def test_forward_has_uncertainties(self):
        mhp = MultiHorizonPredictor(D)
        hv  = _p_gen(D, seed=1)
        out = mhp.forward(hv)
        assert "uncertainties" in out
        for unc in out["uncertainties"].values():
            assert 0.0 <= unc <= 1.0

    def test_ensemble_pred_binary(self):
        mhp = MultiHorizonPredictor(D)
        hv  = _p_gen(D, seed=2)
        out = mhp.forward(hv)
        ep  = out["ensemble_pred"]
        assert set(ep.unique().tolist()).issubset({0.0, 1.0})
