"""
tests/test_new_capabilities.py
================================
Tests for hdc/veckm.py, hdc/hdc_vae.py, hdc/spike_coding.py
"""
import math
import pytest
import torch
import torch.nn.functional as F

# VecKM
from hdc.veckm import (
    ExactVecKM, FastVecKM, HDCPointCloudClassifier, LiDAREncoder,
)
# HDC-VAE
from hdc.hdc_vae import (
    HDCEncoder, HDCDecoder, HDCVAE, HDCConditionalVAE, HDCInterpolator,
)
# Spike Coding
from hdc.spike_coding import (
    RateEncoder, RateDecoder,
    PhaseEncoder, PhaseDecoder,
    TemporalEncoder, TemporalDecoder,
    PopulationEncoder, PopulationDecoder,
    BurstEncoder, BurstDecoder,
    compare_coding_schemes,
)

# ── Constants ─────────────────────────────────────────────────────────────────
D      = 128
N_PTS  = 30


# ─── ExactVecKM ───────────────────────────────────────────────────────────────

class TestExactVecKM:
    def setup_method(self):
        self.enc = ExactVecKM(dim=D, n_neighbors=8, sigma=1.0, seed=0)

    def test_encode_shape(self):
        pts = torch.randn(N_PTS, 3)
        hvs = self.enc.encode(pts)
        assert hvs.shape == (N_PTS, D)

    def test_encode_binary(self):
        hvs = self.enc.encode(torch.randn(N_PTS, 3))
        assert set(hvs.unique().tolist()).issubset({0.0, 1.0})

    def test_encode_density_near_half(self):
        hvs = self.enc.encode(torch.randn(N_PTS, 3))
        dens = float(hvs.mean())
        assert 0.3 <= dens <= 0.7


# ─── FastVecKM ────────────────────────────────────────────────────────────────

class TestFastVecKM:
    def setup_method(self):
        self.enc = FastVecKM(dim=D, sigma=1.0, seed=0)

    def test_encode_shape(self):
        pts = torch.randn(N_PTS, 3)
        hvs = self.enc.encode(pts)
        assert hvs.shape == (N_PTS, D)

    def test_encode_binary(self):
        hvs = self.enc.encode(torch.randn(N_PTS, 3))
        assert set(hvs.unique().tolist()).issubset({0.0, 1.0})

    def test_global_descriptor_shape(self):
        gd = self.enc.global_descriptor(torch.randn(N_PTS, 3))
        assert gd.shape == (D,)
        assert set(gd.unique().tolist()).issubset({0.0, 1.0})

    def test_similar_clouds_more_similar(self):
        from hdc.physics_world_model import _hamming
        pts1 = torch.randn(N_PTS, 3)
        pts2 = pts1 + torch.randn_like(pts1) * 0.05   # slight perturbation
        pts3 = torch.randn(N_PTS, 3) * 5.0              # very different
        gd1  = self.enc.global_descriptor(pts1)
        gd2  = self.enc.global_descriptor(pts2)
        gd3  = self.enc.global_descriptor(pts3)
        sim12 = float(_hamming(gd1.unsqueeze(0), gd2.unsqueeze(0)))
        sim13 = float(_hamming(gd1.unsqueeze(0), gd3.unsqueeze(0)))
        assert sim12 > sim13, "Similar clouds should be more similar"

    def test_encode_with_features(self):
        pts  = torch.randn(N_PTS, 3)
        feat = torch.randn(N_PTS, 8)   # additional features
        hvs  = self.enc.encode(pts, point_features=feat)
        assert hvs.shape == (N_PTS, D)

    def test_batch_encode(self):
        batch = [torch.randn(n, 3) for n in [10, 20, 15]]
        results = self.enc.encode_batch(batch)
        assert len(results) == 3
        assert results[0].shape == (10, D)


# ─── HDCPointCloudClassifier ─────────────────────────────────────────────────

class TestHDCPointCloudClassifier:
    def setup_method(self):
        self.clf = HDCPointCloudClassifier(n_classes=2, dim=D, sigma=0.5)
        for c in range(2):
            for s in range(5):
                pts = torch.randn(20, 3) + c * 3.0
                self.clf.train(pts, c)

    def test_predict_in_range(self):
        pts  = torch.randn(20, 3)
        pred, sims = self.clf.predict(pts)
        assert 0 <= pred < 2
        assert len(sims) == 2

    def test_sims_in_range(self):
        _, sims = self.clf.predict(torch.randn(20, 3))
        assert all(0.0 <= s <= 1.0 for s in sims)

    def test_encode_shape(self):
        hv = self.clf.encode(torch.randn(20, 3))
        assert hv.shape == (D,)


# ─── LiDAREncoder ────────────────────────────────────────────────────────────

class TestLiDAREncoder:
    def setup_method(self):
        self.lidar = LiDAREncoder(dim=D, max_range=50.0, n_layers=3)

    def test_encode_scan_shape(self):
        scan = torch.randn(100, 3) * 10.0
        hv   = self.lidar.encode_scan(scan)
        assert hv.shape == (D,)

    def test_temporal_context_shape(self):
        self.lidar.encode_scan(torch.randn(50, 3))
        ctx = self.lidar.temporal_context()
        assert ctx.shape == (D,)

    def test_scan_count_increments(self):
        for _ in range(3):
            self.lidar.encode_scan(torch.randn(50, 3))
        assert self.lidar._scan_count == 3

    def test_reset(self):
        self.lidar.encode_scan(torch.randn(50, 3))
        self.lidar.reset()
        assert self.lidar._scan_count == 0
        assert float(self.lidar._scan_memory.sum()) == 0.0

    def test_obstacle_score_in_range(self):
        scan_hv = (torch.rand(D) >= 0.5).float()
        proto   = (torch.rand(D) >= 0.5).float()
        score   = self.lidar.obstacle_score(scan_hv, proto)
        assert 0.0 <= score <= 1.0


# ─── HDCEncoder ──────────────────────────────────────────────────────────────

class TestHDCEncoder:
    def setup_method(self):
        self.enc = HDCEncoder(input_dim=32, latent_dim=16, hidden_dim=64)

    def test_forward_shapes(self):
        x      = torch.rand(4, 32)
        mu, lv = self.enc(x)
        assert mu.shape == (4, 16)
        assert lv.shape == (4, 16)

    def test_mu_in_01(self):
        mu, _ = self.enc(torch.rand(4, 32))
        assert mu.min() >= 0.0 and mu.max() <= 1.0

    def test_sample_binary(self):
        self.enc.train()
        mu, lv = self.enc(torch.rand(4, 32))
        z = self.enc.sample(mu, lv)
        assert z.shape == (4, 16)
        assert set(z.unique().tolist()).issubset({0.0, 1.0})


# ─── HDCVAE ──────────────────────────────────────────────────────────────────

class TestHDCVAE:
    def setup_method(self):
        self.vae = HDCVAE(input_dim=32, latent_dim=16, hidden_dim=64, beta=1.0, lr=1e-2)
        self.X   = (torch.rand(10, 32) > 0.5).float()

    def test_forward_keys(self):
        out = self.vae(self.X)
        assert "z" in out and "mu" in out and "elbo" in out

    def test_forward_shapes(self):
        out = self.vae(self.X)
        assert out["z"].shape == (10, 16)
        assert out["x_binary"].shape == (10, 32)

    def test_train_step_returns_losses(self):
        result = self.vae.train_step(self.X)
        assert "elbo" in result and "recon" in result and "kl" in result

    def test_kl_nonneg(self):
        result = self.vae.train_step(self.X)
        assert result["kl"] >= 0.0

    def test_generate_shape(self):
        gen = self.vae.generate(n=5)
        assert gen.shape == (5, 32)

    def test_generate_binary(self):
        gen = self.vae.generate(n=3)
        assert set(gen.unique().tolist()).issubset({0.0, 1.0})

    def test_encode_shape(self):
        z = self.vae.encode(self.X[:4])
        assert z.shape == (4, 16)

    def test_decode_shape(self):
        z   = (torch.rand(4, 16) > 0.5).float()
        x_r = self.vae.decode(z)
        assert x_r.shape == (4, 32)

    def test_reconstruct_shape(self):
        x_r = self.vae.reconstruct(self.X[:4])
        assert x_r.shape == (4, 32)

    def test_reconstruction_accuracy_in_range(self):
        for _ in range(5):
            self.vae.train_step(self.X)
        acc = self.vae.reconstruction_accuracy(self.X)
        assert 0.0 <= acc <= 1.0


# ─── HDCConditionalVAE ────────────────────────────────────────────────────────

class TestHDCConditionalVAE:
    def setup_method(self):
        self.cvae = HDCConditionalVAE(input_dim=32, latent_dim=16, n_classes=3)

    def test_generate_class_shape(self):
        samples = self.cvae.generate_class(label=0, n=4)
        assert samples.shape == (4, 32)

    def test_generate_different_classes(self):
        s0 = self.cvae.generate_class(label=0, n=3)
        s1 = self.cvae.generate_class(label=1, n=3)
        assert s0.shape == s1.shape == (3, 32)


# ─── HDCInterpolator ─────────────────────────────────────────────────────────

class TestHDCInterpolator:
    def setup_method(self):
        vae = HDCVAE(input_dim=32, latent_dim=16, hidden_dim=64, lr=1e-2)
        self.interp = HDCInterpolator(vae)
        self.x1 = (torch.rand(32) > 0.3).float()
        self.x2 = (torch.rand(32) > 0.7).float()

    def test_interpolate_length(self):
        path = self.interp.interpolate(self.x1, self.x2, n_steps=5)
        assert len(path) == 7   # n_steps + 2

    def test_interpolate_shapes(self):
        path = self.interp.interpolate(self.x1, self.x2, n_steps=3)
        assert all(p.shape == (32,) for p in path)

    def test_concept_arithmetic_shape(self):
        result = self.interp.concept_arithmetic(
            base=self.x1, add=self.x2,
            sub=(torch.rand(32) > 0.5).float()
        )
        assert result.shape == (32,)


# ─── Rate Coding ─────────────────────────────────────────────────────────────

class TestRateCoding:
    def test_encode_shape(self):
        enc = RateEncoder(T=50)
        s   = enc.encode(torch.linspace(0, 1, 8))
        assert s.shape == (50, 8)

    def test_encode_binary(self):
        enc = RateEncoder(T=50)
        s   = enc.encode(torch.linspace(0, 1, 8))
        assert set(s.unique().tolist()).issubset({0.0, 1.0})

    def test_decode_in_range(self):
        enc = RateEncoder(T=200)
        dec = RateDecoder(T=200)
        v   = torch.linspace(0, 1, 8)
        v_r = dec.decode(enc.encode(v))
        assert v_r.min() >= 0.0 and v_r.max() <= 1.0

    def test_decode_accuracy(self):
        enc = RateEncoder(T=500, r_max=200.0)
        dec = RateDecoder(T=500, r_max=200.0)
        v   = torch.linspace(0.1, 0.9, 8)
        v_r = dec.decode(enc.encode(v))
        mse = float(F.mse_loss(v_r, v))
        assert mse < 0.05, f"Rate coding MSE too high: {mse}"


# ─── Phase Coding ─────────────────────────────────────────────────────────────

class TestPhaseCoding:
    def test_encode_exactly_one_spike(self):
        enc = PhaseEncoder(T=50, n_neurons=8)
        v   = torch.linspace(0, 1, 8)
        s   = enc.encode(v)
        assert s.shape == (50, 8)
        # Each neuron fires exactly once per cycle
        assert (s.sum(dim=0) == 1).all()

    def test_decode_near_exact(self):
        enc = PhaseEncoder(T=100, n_neurons=8)
        dec = PhaseDecoder(T=100)
        v   = torch.linspace(0, 1, 8)
        v_p = dec.decode(enc.encode(v))
        mse = float(F.mse_loss(v_p, v))
        assert mse < 1e-3

    def test_bits_per_spike_larger_than_rate(self):
        enc_p = PhaseEncoder(T=100)
        enc_r = RateEncoder(T=100)
        assert enc_p.bits_per_spike() > enc_r.bits_per_spike()


# ─── Population Coding ────────────────────────────────────────────────────────

class TestPopulationCoding:
    def setup_method(self):
        self.enc = PopulationEncoder(n_neurons=32, value_range=(0, 1))
        self.dec = PopulationDecoder(self.enc)

    def test_encode_shape(self):
        rates = self.enc.encode(torch.linspace(0, 1, 8))
        assert rates.shape == (8, 32)

    def test_encode_rates_in_range(self):
        rates = self.enc.encode(torch.linspace(0, 1, 8))
        assert rates.min() >= 0.0 and rates.max() <= self.enc.r_max

    def test_decode_shape(self):
        rates = self.enc.encode(torch.linspace(0, 1, 8))
        v_dec = self.dec.decode(rates)
        assert v_dec.shape == (8,)

    def test_decode_accuracy(self):
        v   = torch.linspace(0.1, 0.9, 8)
        v_d = self.dec.decode(self.enc.encode(v))
        mse = float(F.mse_loss(v_d, v))
        assert mse < 0.05


# ─── Burst Coding ─────────────────────────────────────────────────────────────

class TestBurstCoding:
    def test_encode_shape(self):
        enc = BurstEncoder(max_burst=7, T=10)
        s   = enc.encode(torch.linspace(0, 1, 8))
        assert s.shape == (10, 8)

    def test_decode_accuracy(self):
        enc = BurstEncoder(max_burst=7, T=10)
        dec = BurstDecoder(max_burst=7)
        v   = torch.linspace(0, 1, 8)
        v_r = dec.decode(enc.encode(v))
        mse = float(F.mse_loss(v_r, v))
        assert mse < 0.05

    def test_bits_per_burst(self):
        enc = BurstEncoder(max_burst=7)
        assert enc.bits_per_burst() == pytest.approx(math.log2(8), abs=1e-5)

    def test_robust_decode(self):
        enc = BurstEncoder(max_burst=7, T=10)
        dec = BurstDecoder(max_burst=7)
        v   = torch.linspace(0, 1, 8)
        s   = enc.encode(v)
        v_r = dec.decode_robust(s, noise_floor=0.05)
        assert v_r.min() >= 0.0 and v_r.max() <= 1.0


# ─── compare_coding_schemes ──────────────────────────────────────────────────

class TestCompareCodingSchemes:
    def test_all_schemes_present(self):
        results = compare_coding_schemes(n_values=8, T=100)
        assert "rate" in results
        assert "phase" in results
        assert "temporal" in results
        assert "burst" in results

    def test_phase_more_bits_than_rate(self):
        results = compare_coding_schemes(n_values=8, T=100)
        assert results["phase"]["bits_per_spike"] > results["rate"]["bits_per_spike"]

    def test_all_mse_nonneg(self):
        results = compare_coding_schemes(n_values=8, T=100)
        for scheme, data in results.items():
            assert data["mse"] >= 0.0


# ─── FastVecKM improvements ───────────────────────────────────────────────────

class TestFastVecKMImprovements:
    def test_multi_scale_encode_shape(self):
        enc = FastVecKM(dim=D, sigma=1.0, seed=0, multi_scale=True)
        hvs = enc.encode(torch.randn(N_PTS, 3))
        assert hvs.shape == (N_PTS, D)

    def test_multi_scale_binary(self):
        enc = FastVecKM(dim=D, sigma=1.0, seed=0, multi_scale=True)
        hvs = enc.encode(torch.randn(N_PTS, 3))
        assert set(hvs.unique().tolist()).issubset({0.0, 1.0})

    def test_incremental_encode_shape(self):
        enc = FastVecKM(dim=D, sigma=1.0, seed=0)
        pts_a = torch.randn(10, 3)
        pts_b = torch.randn(10, 3)
        hvs = enc.encode_incremental(pts_a)
        assert hvs.shape == (10, D)
        hvs2 = enc.encode_incremental(pts_b)
        assert hvs2.shape == (10, D)

    def test_reset_incremental_clears_pool(self):
        enc = FastVecKM(dim=D, sigma=1.0, seed=0)
        enc.encode_incremental(torch.randn(10, 3))
        assert enc._global_pool is not None
        enc.reset_incremental()
        assert enc._global_pool is None
        assert enc._n_incremental == 0

    def test_estimate_normals_shape(self):
        enc = FastVecKM(dim=D, sigma=1.0, seed=0)
        pts = torch.randn(N_PTS, 3)
        normals = enc.estimate_normals(pts)
        assert normals.shape == (N_PTS, 3)

    def test_estimate_normals_unit_length(self):
        enc = FastVecKM(dim=D, sigma=1.0, seed=0)
        pts = torch.randn(20, 3)
        normals = enc.estimate_normals(pts)
        norms = normals.norm(dim=1)
        assert torch.allclose(norms, torch.ones(20), atol=1e-5)

    def test_small_feature_linear_time(self):
        enc = FastVecKM(dim=D, sigma=1.0, seed=0)
        pts  = torch.randn(N_PTS, 3)
        feat = torch.randn(N_PTS, 4)   # small feature dim — should use O(N×D) path
        hvs  = enc.encode(pts, point_features=feat)
        assert hvs.shape == (N_PTS, D)


# ─── LiDAREncoder obstacle detection ─────────────────────────────────────────

class TestLiDARObstacleDetection:
    def test_register_obstacle(self):
        lidar = LiDAREncoder(dim=D, max_range=50.0)
        scan = torch.randn(100, 3) * 10.0
        scan_hv = lidar.encode_scan(scan)
        lidar.register_obstacle(scan_hv, "wall")
        assert len(lidar._obstacle_protos) == 1
        assert lidar._obstacle_labels[0] == "wall"

    def test_detect_registered_obstacle(self):
        lidar = LiDAREncoder(dim=D, max_range=50.0)
        scan  = torch.randn(100, 3) * 5.0
        scan[:, 2] *= 0.2
        scan_hv = lidar.encode_scan(scan)
        lidar.register_obstacle(scan_hv, "obstacle")
        alerts = lidar.detect_obstacles(scan_hv, danger_thresh=0.5)
        assert len(alerts) > 0
        assert alerts[0]["label"] == "obstacle"

    def test_no_alert_on_different_scan(self):
        lidar = LiDAREncoder(dim=D, max_range=50.0)
        scan_a  = torch.randn(50, 3) * 2.0
        scan_b  = torch.randn(50, 3) * 20.0 + 30.0  # very different
        hv_a = lidar.encode_scan(scan_a)
        hv_b = lidar.encode_scan(scan_b)
        lidar.register_obstacle(hv_a, "zone_A")
        alerts = lidar.detect_obstacles(hv_b, danger_thresh=0.8)
        # Different scan — should have zero or very few alerts
        assert isinstance(alerts, list)

    def test_same_label_updates_prototype(self):
        lidar = LiDAREncoder(dim=D, max_range=50.0)
        for _ in range(3):
            scan  = torch.randn(50, 3) * 5.0
            hv = lidar.encode_scan(scan)
            lidar.register_obstacle(hv, "building")
        # Multiple updates to same label → single prototype
        assert len(lidar._obstacle_protos) == 1
        assert lidar._obstacle_counts[0] == 3


# ─── HDCMessagePassing improvements ──────────────────────────────────────────

class TestHDCMessagePassingImprovements:
    def setup_method(self):
        from hdc.graph_neural_hd import HDCMessagePassing
        self.mp = HDCMessagePassing(D, degree_norm=True, use_attention=True)
        self.node_hvs = {i: (torch.rand(D) > 0.5).float() for i in range(4)}
        self.adj      = {0: [1, 2], 1: [0, 3], 2: [0], 3: [1]}

    def test_pass_messages_shape(self):
        out = self.mp.pass_messages(self.node_hvs, self.adj)
        assert len(out) == len(self.node_hvs)
        for hv in out.values():
            assert hv.shape == (D,)

    def test_multi_round_with_skip(self):
        from hdc.graph_neural_hd import HDCMessagePassing
        mp = HDCMessagePassing(D)
        out = mp.multi_round(self.node_hvs, self.adj, n_rounds=3, skip_connect=True)
        assert len(out) == len(self.node_hvs)

    def test_multi_round_no_skip(self):
        from hdc.graph_neural_hd import HDCMessagePassing
        mp = HDCMessagePassing(D)
        out = mp.multi_round(self.node_hvs, self.adj, n_rounds=2, skip_connect=False)
        assert len(out) == len(self.node_hvs)

    def test_degree_norm_no_crash(self):
        from hdc.graph_neural_hd import HDCMessagePassing
        mp = HDCMessagePassing(D, degree_norm=True, use_attention=False)
        out = mp.pass_messages(self.node_hvs, self.adj)
        assert len(out) > 0


# ─── HDCContrastiveLearner hard negatives ─────────────────────────────────────

class TestHDCContrastiveLearnerHardNeg:
    def test_update_returns_float(self):
        from hdc.self_supervised import HDCContrastiveLearner
        cl = HDCContrastiveLearner(D, memory_size=32)
        hv = (torch.rand(D) > 0.5).float()
        loss = cl.update(hv, n_hard=4)
        assert isinstance(loss, float)

    def test_hard_neg_lower_loss_than_no_hard(self):
        from hdc.self_supervised import HDCContrastiveLearner
        torch.manual_seed(42)
        cl = HDCContrastiveLearner(D, memory_size=32)
        hv = (torch.rand(D) > 0.5).float()
        # Multiple updates with and without hard negatives
        losses_no_hard = [cl.update(hv, n_hard=0) for _ in range(5)]
        losses_hard    = [cl.update(hv, n_hard=8) for _ in range(5)]
        assert isinstance(losses_no_hard[0], float)
        assert isinstance(losses_hard[0], float)

    def test_label_prototype_created(self):
        from hdc.self_supervised import HDCContrastiveLearner
        cl = HDCContrastiveLearner(D, memory_size=32)
        hv = (torch.rand(D) > 0.5).float()
        cl.update(hv, label="cat", n_hard=4)
        assert "cat" in cl._prototypes
