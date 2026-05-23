"""
tests/test_manifold_decoder.py
================================
Tests for OnlinePCA, NeuralManifoldDecoder,
LatentDynamicsModel, PopulationActivityEncoder.
"""
import math
import pytest
import torch
from models.manifold_decoder import (
    OnlinePCA,
    NeuralManifoldDecoder,
    LatentDynamicsModel,
    PopulationActivityEncoder,
)

N = 32    # neurons
K = 2     # output size
k = 8     # manifold dims


# ─── OnlinePCA ───────────────────────────────────────────────────────────────

class TestOnlinePCA:
    def setup_method(self):
        self.pca = OnlinePCA(n_input=N, n_components=k, lr=0.01)

    def test_update_returns_projection(self):
        z = self.pca.update(torch.randn(N))
        assert z.shape == (k,)

    def test_project_shape(self):
        for _ in range(10):
            self.pca.update(torch.randn(N))
        z = self.pca.project(torch.randn(N))
        assert z.shape == (k,)

    def test_project_batch(self):
        for _ in range(10):
            self.pca.update(torch.randn(N))
        X  = torch.randn(5, N)
        zs = self.pca.project(X)
        assert zs.shape == (5, k)

    def test_mean_shifts_with_data(self):
        mean0 = self.pca._mean.clone()
        for _ in range(20):
            self.pca.update(torch.ones(N) * 5.0)
        # Mean should shift toward 5.0
        assert float(self.pca._mean.mean()) > 1.0

    def test_evr_shape(self):
        for _ in range(20):
            self.pca.update(torch.randn(N))
        evr = self.pca.explained_variance_ratio()
        assert evr.shape == (k,)

    def test_evr_sums_to_one(self):
        for _ in range(20):
            self.pca.update(torch.randn(N))
        evr = self.pca.explained_variance_ratio()
        assert abs(float(evr.sum()) - 1.0) < 1e-4

    def test_components_shape(self):
        assert self.pca.W.shape == (k, N)

    def test_reset(self):
        for _ in range(10):
            self.pca.update(torch.randn(N))
        self.pca.reset()
        assert self.pca._n_seen == 0
        assert float(self.pca._mean.sum()) == 0.0


# ─── NeuralManifoldDecoder ────────────────────────────────────────────────────

class TestNeuralManifoldDecoder:
    def setup_method(self):
        self.mfd = NeuralManifoldDecoder(
            n_neurons=N, output_size=K, n_components=k
        )

    def test_step_shape(self):
        spikes = (torch.rand(N) > 0.8).float()
        pred   = self.mfd.step(spikes)
        assert pred.shape == (K,)

    def test_forward_shape(self):
        # Need some PCA data first
        for _ in range(5):
            self.mfd.step((torch.rand(N) > 0.8).float())
        pred = self.mfd.forward((torch.rand(N) > 0.8).float())
        assert pred.shape == (K,)

    def test_update_returns_dict(self):
        spikes = (torch.rand(N) > 0.8).float()
        self.mfd.step(spikes)
        info = self.mfd.update(torch.randn(K) * 0.1)
        assert "eff_lr" in info
        assert "denom" in info

    def test_eff_lr_positive(self):
        spikes = (torch.rand(N) > 0.8).float()
        self.mfd.step(spikes)
        info = self.mfd.update(torch.randn(K))
        assert info["eff_lr"] > 0.0

    def test_weights_change_after_update(self):
        W0 = self.mfd.W.clone()
        for _ in range(10):
            s = (torch.rand(N) > 0.8).float()
            self.mfd.step(s)
            self.mfd.update(torch.ones(K))   # large constant error drives updates
        # After 10 steps with error=1, W should have changed
        assert not torch.allclose(self.mfd.W, W0, atol=1e-6)

    def test_manifold_report(self):
        for _ in range(10):
            self.mfd.step((torch.rand(N) > 0.8).float())
        rep = self.mfd.manifold_report()
        assert "n_components" in rep
        assert rep["n_components"] == k
        assert rep["n_seen"] >= 10

    def test_reset(self):
        for _ in range(5):
            self.mfd.step((torch.rand(N) > 0.8).float())
        self.mfd.reset(reset_pca=True)   # full reset including PCA
        assert float(self.mfd.pca._mean.abs().sum()) == 0.0


# ─── LatentDynamicsModel ─────────────────────────────────────────────────────

class TestLatentDynamicsModel:
    def setup_method(self):
        self.lds = LatentDynamicsModel(n_latent=k, output_size=K)

    def test_observe_shape(self):
        z    = torch.randn(k)
        z_sm = self.lds.observe(z)
        assert z_sm.shape == (k,)

    def test_observe_with_y(self):
        z    = torch.randn(k)
        y    = torch.randn(K)
        z_sm = self.lds.observe(z, y)
        assert z_sm.shape == (k,)

    def test_decode_shape(self):
        z     = torch.randn(k)
        z_sm  = self.lds.observe(z)
        vel   = self.lds.decode(z_sm)
        assert vel.shape == (K,)

    def test_observe_changes_state(self):
        z0 = self.lds.z_hat.clone()
        for _ in range(5):
            self.lds.observe(torch.randn(k))
        assert not torch.equal(self.lds.z_hat, z0)

    def test_a_stays_stable(self):
        for _ in range(60):
            self.lds.observe(torch.randn(k), torch.randn(K))
        eigs = torch.linalg.eigvals(self.lds.A).abs()
        assert float(eigs.max()) <= 1.0 + 1e-5

    def test_reset(self):
        for _ in range(5):
            self.lds.observe(torch.randn(k))
        self.lds.reset()
        assert float(self.lds.z_hat.sum()) == 0.0


# ─── PopulationActivityEncoder ───────────────────────────────────────────────

class TestPopulationActivityEncoder:
    def setup_method(self):
        self.enc = PopulationActivityEncoder(n_neurons=N, window=5)

    def test_feature_dim(self):
        assert self.enc.feature_dim == 3 * N

    def test_encode_shape(self):
        feat = self.enc.encode((torch.rand(N) > 0.8).float())
        assert feat.shape == (3 * N,)

    def test_encode_with_correlations(self):
        enc_c = PopulationActivityEncoder(n_neurons=8, window=3, use_corr=True)
        feat  = enc_c.encode((torch.rand(8) > 0.5).float())
        expected = 3 * 8 + 8 * 7 // 2
        assert feat.shape == (expected,)

    def test_mean_rate_updates_over_window(self):
        # Feed constant signal, mean rate should stabilize
        for _ in range(10):
            self.enc.encode(torch.ones(N))
        feat  = self.enc.encode(torch.ones(N))
        mean_rate = feat[:N]
        assert float(mean_rate.mean()) > 0.5  # all neurons fire → high mean

    def test_temporal_deriv_captures_change(self):
        # First step: spikes
        self.enc.encode(torch.ones(N))
        # Second step: no spikes → derivative should be negative
        feat2 = self.enc.encode(torch.zeros(N))
        temp_deriv = feat2[2*N:]   # last N features
        assert float(temp_deriv.mean()) < 0.0

    def test_reset(self):
        for _ in range(5):
            self.enc.encode((torch.rand(N) > 0.5).float())
        self.enc.reset()
        assert float(self.enc._buf.sum()) == 0.0
        assert float(self.enc._prev.sum()) == 0.0


# ─── Integration: EliteSNNTrainingModel with 0.95-tier ─────────────────────────

class TestEliteModelWith95Tier:
    def test_alif_manifold_wired(self):
        from arthedain.model import EliteSNNTrainingModel, SNNTrainingConfig
        cfg = SNNTrainingConfig(input_size=10, hidden_size=N, output_size=K)
        m   = EliteSNNTrainingModel(config=cfg, wiener_lags=3, manifold_k=k,
                                   use_alif=True, use_manifold=True, use_pop_enc=True)
        for i in range(20):
            x = torch.rand(10)
            p = m.step(x)
            t = torch.tensor([math.sin(i * 0.1), math.cos(i * 0.1)])
            m.track(p, t)
            m.update(p.detach() - t)
        r = m.pearson_r()
        assert isinstance(r, float)
        assert "ALIF" in repr(m)
        assert "Manifold" in repr(m)

    def test_tier95_in_repr(self):
        from arthedain.model import EliteSNNTrainingModel, SNNTrainingConfig
        cfg = SNNTrainingConfig(input_size=8, hidden_size=16, output_size=2)
        m   = EliteSNNTrainingModel(config=cfg, manifold_k=4, wiener_lags=2)
        r   = repr(m)
        assert "ALIF" in r or "Manifold" in r
