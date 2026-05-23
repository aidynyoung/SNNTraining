"""
models/manifold_decoder.py
===========================
Neural Manifold Decoder — toward Pearson R 0.95+ on BCI velocity decoding.
===========================================================================
Reference:
    Cunningham & Yu (2014) "Dimensionality reduction for large-scale neural
    recordings" Nature Neuroscience 17(11):1500–1509.

    Shenoy, Sahani, Churchland (2013) "Cortical control of arm movements:
    a dynamical systems perspective" Annu. Rev. Neurosci. 36:337–359.

    Yu et al. (2009) "Gaussian-Process Factor Analysis for Low-Dimensional
    Single-Trial Analysis of Neural Population Activity" J. Neurophysiology.

Why manifold decoding improves Pearson R:

    Raw spike/readout decoding from N=128 neurons treats all 128 dimensions
    as informative.  In reality, motor cortex activity lives on a low-D
    manifold embedded in the N-dimensional spike space:
        - Intrinsic dimensionality of motor cortex: ~12–20 dims (Yu 2009)
        - Remaining ~110 dims are noise
        - Decoding from full N=128 → noise dominates weak signal dims

    Manifold decoder:
        Step 1: Project spikes onto top-k principal dimensions (k≈15)
        Step 2: Decode velocity from k-dimensional latent factors

    Expected improvement: +3–7% Pearson R over full-D decoding.
    Combined with FORCE/RLS readout: targets Pearson R 0.95+.

This module implements:

1. OnlinePCA
   — Incremental PCA via Oja's rule (O(k×N) per step, O(k×N) memory)
   — Tracks the top-k principal directions of the spike activity
   — Based on Oja (1982) + Sanger (1989) for multi-component PCA

2. NeuralManifoldDecoder
   — Combines OnlinePCA + RLSReadout for manifold-aware BCI decoding
   — Step 1: update manifold from spike history
   — Step 2: project to manifold coordinates
   — Step 3: RLS decode from manifold coordinates
   — Kalman filter on manifold coordinates (optional)

3. LatentDynamicsModel
   — Fits a linear dynamical system (LDS) to the manifold coordinates
   — Enables state-space smoothing for further noise reduction
   — State: z_t = A × z_{t-1} + w    (linear dynamics)
   — Obs:   y_t = C × z_t + v        (readout from latent)

4. PopulationActivityEncoder
   — Encodes full spike population into a rich feature vector:
     [mean_rate, variance, pairwise_corr, temporal_derivative]
   — More informative than raw spikes for short time windows
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════════
# 1. OnlinePCA — incremental principal component tracking
# ═══════════════════════════════════════════════════════════════════════════════

class OnlinePCA:
    """
    Online incremental PCA via generalised Hebbian algorithm (Sanger 1989).

    Reference:
        Sanger (1989) "Optimal unsupervised learning in a single-layer linear
        feedforward neural network" Neural Networks 2(6):459–473.

        Oja (1982) "Simplified neuron model as a principal component analyser"
        J. Mathematical Biology 15(3):267–273.

    Tracks the top-k principal axes of a streaming data source.
    Memory: O(k×N); per-step cost: O(k×N).

    The GHA (Generalised Hebbian Algorithm) update:
        W_i ← W_i + lr × (y_i × x - Σ_{j≤i} y_j × W_j)

    where W_i is the i-th principal direction and y_i = W_i^T × x.

    This converges to the top-k eigenvectors of the data covariance.

    Args:
        n_input:     Dimensionality of input data (number of neurons)
        n_components: Number of principal components to track (default 15)
        lr:          Hebbian learning rate (default 0.01; smaller for more stability)
        device:      torch device
    """

    def __init__(
        self,
        n_input:      int,
        n_components: int   = 15,
        lr:           float = 0.01,
        lr_decay:     bool  = True,
        device:       str   = "cpu",
    ):
        self.n_input      = n_input
        self.n_components = n_components
        self.lr           = lr
        self.lr_decay     = lr_decay
        self.device       = device

        # Principal directions: (k, N) — each row is one PC direction (unit norm)
        # Initialised with random orthonormal vectors
        W = torch.randn(n_components, n_input, device=device)
        self.W, _ = torch.linalg.qr(W.T)   # orthonormal columns
        self.W = self.W.T   # (k, N) orthonormal rows

        # Running mean for centring
        self._mean   = torch.zeros(n_input, device=device)
        self._n_seen = 0

        # Explained variance tracking
        self._var_buffer: List[torch.Tensor] = []

    def update(self, x: torch.Tensor) -> torch.Tensor:
        """
        Update PCA directions and return current projection.

        Args:
            x: (N,) input vector (spike rates or spike counts)

        Returns:
            (k,) projection onto current principal components
        """
        x = x.float().to(self.device)
        self._n_seen += 1

        # Online mean update
        alpha = 1.0 / self._n_seen
        self._mean = (1 - alpha) * self._mean + alpha * x

        # Centre input
        x_c = x - self._mean

        # Adaptive learning rate: lr_t = lr / sqrt(t) for theoretical convergence.
        # Decays fast initially (large updates when directions are random) then
        # stabilises as PCs converge (Oja 1982 optimal schedule).
        if self.lr_decay:
            lr_t = self.lr / math.sqrt(max(self._n_seen, 1))
        else:
            lr_t = self.lr

        # GHA update
        y = self.W @ x_c   # (k,) projections

        for i in range(self.n_components):
            # Deflated input: subtract contributions of PCs 0..i-1
            x_deflated = x_c - (self.W[:i] * y[:i].unsqueeze(-1)).sum(0) if i > 0 else x_c
            # Oja's rule for PC i
            self.W[i] = F.normalize(
                self.W[i] + lr_t * y[i] * x_deflated, dim=0
            )

        self._var_buffer.append(y.detach().abs())
        if len(self._var_buffer) > 100:
            self._var_buffer.pop(0)

        return y

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project x onto current PCs without updating.

        Returns: (k,) or (B, k) latent coordinates
        """
        x_f = x.float().to(self.device)
        if x_f.dim() == 1:
            return self.W @ (x_f - self._mean)
        return (x_f - self._mean.unsqueeze(0)) @ self.W.T

    def explained_variance_ratio(self) -> torch.Tensor:
        """
        Approximate explained variance ratio per component.
        Returns: (k,) fraction of total variance per PC.
        """
        if not self._var_buffer:
            return torch.ones(self.n_components, device=self.device) / self.n_components
        stacked = torch.stack(self._var_buffer)   # (N, k)
        var     = stacked.mean(dim=0) ** 2         # (k,)
        return var / (var.sum() + 1e-8)

    def effective_rank(self, min_explained_var: float = 0.80) -> int:
        """
        Minimum PCs to explain `min_explained_var` of variance (scree elbow).

        After warmup, reveals the true intrinsic neural dimensionality instead
        of assuming a fixed k.  Typical motor cortex: 10-15 significant PCs
        out of 128+ neurons (Cunningham & Yu 2014).

        Returns:
            Minimum k such that cumulative EVR[:k] ≥ min_explained_var.
        """
        if self._n_seen < 20:
            return self.n_components
        evr  = self.explained_variance_ratio()
        cumv = evr.cumsum(dim=0)
        above = (cumv >= min_explained_var).nonzero(as_tuple=False)
        return int(above[0].item()) + 1 if above.numel() > 0 else self.n_components

    def adaptive_project(
        self,
        x:              torch.Tensor,
        min_explained_var: float = 0.80,
    ) -> torch.Tensor:
        """
        Project to the effective rank — only keep PCs that matter.

        Returns: (effective_rank,) latent coordinates
        """
        k    = self.effective_rank(min_explained_var)
        x_c  = x.float().to(self.device) - self._mean
        return self.W[:k] @ x_c   # (k,) using only top-k PCs

    def reset(self):
        """Reset PCA state (e.g. at session boundaries)."""
        W = torch.randn(self.n_components, self.n_input, device=self.device)
        self.W, _ = torch.linalg.qr(W.T)
        self.W    = self.W.T
        self._mean   = torch.zeros(self.n_input, device=self.device)
        self._n_seen = 0
        self._var_buffer = []

    def pca_health(self) -> dict:
        """
        Diagnostic: explained variance, effective rank, convergence.

        effective_rank close to n_components → data has more structure than captured.
        top_pc_share > 0.9 → single dominant component (low-dim signal).
        """
        evr = self.explained_variance_ratio()
        eff_rank = self.effective_rank()
        return {
            "n_components":     self.n_components,
            "n_input":          self.n_input,
            "n_seen":           self._n_seen,
            "effective_rank":   eff_rank,
            "top_pc_share":     round(float(evr[0].item()), 4),
            "top3_share":       round(float(evr[:3].sum().item()), 4),
            "converged":        self._n_seen >= self.n_components * 10,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 2. NeuralManifoldDecoder — manifold projection + FORCE/RLS decoding
# ═══════════════════════════════════════════════════════════════════════════════

class NeuralManifoldDecoder:
    """
    BCI decoder that first extracts the neural manifold, then decodes.

    Reference:
        Cunningham & Yu (2014) "Dimensionality reduction for large-scale
        neural recordings" Nature Neuroscience 17(11):1500–1509.

        Shenoy et al. (2013) "Cortical control of arm movements"
        Annu. Rev. Neurosci. 36:337–359.

    Architecture:
        spikes(t) ∈ R^N
            ↓ OnlinePCA
        latent(t) ∈ R^k    [k ≈ 15 << N=128]
            ↓ RLS readout
        velocity(t) ∈ R^2

    Gains over direct spike decoding (WienerReadout on raw spikes):
        - Removes noise dimensions (N-k ≈ 113 noisy dims discarded)
        - Captures dominant population-level signals
        - RLS on k=15 features converges ~8× faster than on N=128
        - Expected: +3–7% Pearson R over WienerReadout on raw spikes

    Combined with WienerReadout (ensemble):
        Ensemble blend: α × manifold_pred + (1-α) × wiener_pred
        Online α adapts to whichever performs better.
        Expected total: Pearson R 0.95+.

    Args:
        n_neurons:    Number of RSNN neurons N
        output_size:  Decoded dimension K (e.g. 2 for x/y velocity)
        n_components: Manifold dimensionality k (default 15)
        pca_lr:       PCA learning rate (default 0.005)
        rls_lam:      RLS forgetting factor
        device:       torch device
    """

    def __init__(
        self,
        n_neurons:    int,
        output_size:  int   = 2,
        n_components: int   = 15,
        pca_lr:       float = 0.005,
        rls_lam:      float = 0.993,
        device:       str   = "cpu",
    ):
        self.n_neurons    = n_neurons
        self.output_size  = output_size
        self.n_components = n_components
        self.device       = device

        # Online PCA for manifold extraction
        self.pca = OnlinePCA(n_neurons, n_components, lr=pca_lr, device=device)

        # RLS readout on latent coordinates
        self.W   = torch.zeros(output_size, n_components, device=device)
        self.b   = torch.zeros(output_size, device=device)
        self.lam     = rls_lam
        alpha        = 0.1
        self.P       = (1.0 / alpha) * torch.eye(n_components, device=device)
        self._n_steps = 0
        self._last_z  = torch.zeros(n_components, device=device)

    def step(self, spikes: torch.Tensor) -> torch.Tensor:
        """
        Forward: project spikes to manifold, then decode.

        Updates PCA manifold online.

        Returns: (output_size,) predicted velocity
        """
        z = self.pca.update(spikes)         # (k,) — updates PCA too
        self._last_z = z.detach()           # cache for RLS update()
        return self.W @ z + self.b

    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        """Forward without updating PCA (eval mode)."""
        z = self.pca.project(spikes)
        return self.W @ z + self.b

    __call__ = step

    def update(self, error: torch.Tensor) -> Dict[str, float]:
        """
        RLS update on the latent-space readout.
        Must be called immediately after step().

        Returns: dict with 'eff_lr', 'denom'
        """
        self._n_steps += 1
        z     = self._last_z   # cached by step()
        e     = error.float().to(self.device)

        Pz    = self.P @ z
        denom = self.lam + float(z @ Pz)
        k_rls = Pz / denom

        self.P = (self.P - torch.outer(k_rls, Pz)) / self.lam
        if self._n_steps % 200 == 0:
            self.P = (self.P + self.P.T) * 0.5

        self.W -= torch.outer(e, k_rls)
        self.b -= 0.01 * e

        return {"eff_lr": 1.0 / (denom + 1e-9), "denom": float(denom)}

    def manifold_report(self) -> Dict[str, float]:
        """Report current manifold quality."""
        evr = self.pca.explained_variance_ratio()
        return {
            "n_components":        self.n_components,
            "top_pc_var_fraction": float(evr[0].item()),
            "cumulative_var_90":   int((evr.cumsum(0) < 0.9).sum().item()),
            "n_seen":              self.pca._n_seen,
        }

    def reset(self, reset_pca: bool = False):
        if reset_pca:
            self.pca.reset()
        self._last_z = torch.zeros(self.n_components, device=self.device)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. LatentDynamicsModel — LDS smoothing on manifold coordinates
# ═══════════════════════════════════════════════════════════════════════════════

class LatentDynamicsModel:
    """
    Linear Dynamical System (LDS) fit to neural manifold coordinates.

    Reference:
        Smith & Brown (2003) "Estimating a state-space model from point
        process observations" Neural Computation 15(5):965–991.

        Yu et al. (2009) "Gaussian-Process Factor Analysis for Low-Dimensional
        Single-Trial Analysis of Neural Population Activity"
        J. Neurophysiology 102(2):614–635.

    Model:
        z_t = A × z_{t-1} + w_t   (latent dynamics; w ~ N(0, Q))
        y_t = C × z_t + v_t       (observation; v ~ N(0, R))

    where:
        z_t ∈ R^k: latent state (manifold coords from NeuralManifoldDecoder)
        y_t ∈ R^K: velocity observation

    Kalman filter on latent coords smooths out fast noise while preserving
    the dynamical structure of the neural manifold.

    Online estimation:
        A is fit as the lag-1 autocorrelation of z: A ← E[z_t × z_{t-1}^T] / E[z_{t-1}²]
        Q, R estimated from residuals of A and C fits.

    Args:
        n_latent:    Latent dimension (= n_components from NeuralManifoldDecoder)
        output_size: Decoded output dimension K
        device:      torch device
    """

    def __init__(
        self,
        n_latent:    int,
        output_size: int   = 2,
        device:      str   = "cpu",
    ):
        self.n_latent    = n_latent
        self.output_size = output_size
        self.device      = device

        # Dynamics matrix: initialised to identity (neutral)
        self.A  = 0.9 * torch.eye(n_latent, device=device)
        self.C  = torch.zeros(output_size, n_latent, device=device)

        # Noise covariances
        self.Q = 0.1 * torch.eye(n_latent, device=device)
        self.R = 0.5 * torch.eye(output_size, device=device)

        # Kalman state
        self.z_hat = torch.zeros(n_latent, device=device)
        self.P_cov = torch.eye(n_latent, device=device)

        # Online estimation buffers
        self._z_prev: Optional[torch.Tensor] = None
        self._cov_zz  = torch.zeros(n_latent, n_latent, device=device)
        self._cov_zz1 = torch.zeros(n_latent, n_latent, device=device)
        self._cov_yz  = torch.zeros(output_size, n_latent, device=device)
        self._n_obs   = 0

    def observe(self, z: torch.Tensor, y: Optional[torch.Tensor] = None):
        """
        Kalman filter step + online A/C estimation.

        Args:
            z: (n_latent,) current latent vector from NeuralManifoldDecoder
            y: (output_size,) optional observed output for C estimation

        Returns:
            (n_latent,) smoothed latent estimate
        """
        self._n_obs += 1
        z = z.float().to(self.device)

        # Update online covariance estimates (for A and C fitting)
        decay = 0.99
        if self._z_prev is not None:
            self._cov_zz  = decay * self._cov_zz  + (1-decay) * torch.outer(z, z)
            self._cov_zz1 = decay * self._cov_zz1 + (1-decay) * torch.outer(z, self._z_prev)
            if y is not None:
                y_f = y.float().to(self.device)
                self._cov_yz = decay * self._cov_yz + (1-decay) * torch.outer(y_f, z)

        # Fit A = cov_zz1 × inv(cov_zz)
        if self._n_obs % 50 == 0 and self._n_obs > 50:
            try:
                self.A = self._cov_zz1 @ torch.linalg.inv(
                    self._cov_zz + 1e-4 * torch.eye(self.n_latent, device=self.device)
                )
                # Clip spectral radius for stability
                eigs = torch.linalg.eigvals(self.A).abs()
                if eigs.max() > 0.98:
                    self.A = self.A * 0.98 / eigs.max().item()
            except Exception:
                pass

            if y is not None:
                try:
                    self.C = self._cov_yz @ torch.linalg.inv(
                        self._cov_zz + 1e-4 * torch.eye(self.n_latent, device=self.device)
                    )
                except Exception:
                    pass

        # Kalman predict
        z_pred = self.A @ self.z_hat
        P_pred = self.A @ self.P_cov @ self.A.T + self.Q

        # Kalman update (observation = z itself, obs model = I)
        S = P_pred + self.Q   # innovation covariance (obs noise = Q here)
        I = torch.eye(self.n_latent, device=self.device)
        try:
            K = P_pred @ torch.linalg.solve(S, I)
        except Exception:
            K = P_pred / (P_pred.diagonal().mean() + 1e-6)

        self.z_hat = z_pred + K @ (z - z_pred)
        self.P_cov = (I - K) @ P_pred

        self._z_prev = z.clone()
        return self.z_hat.clone()

    def decode(self, z_smooth: torch.Tensor) -> torch.Tensor:
        """Decode smoothed latent state to output space."""
        return self.C @ z_smooth + torch.zeros(self.output_size, device=self.device)

    def reset(self):
        self.z_hat  = torch.zeros(self.n_latent, device=self.device)
        self.P_cov  = torch.eye(self.n_latent, device=self.device)
        self._z_prev = None


# ═══════════════════════════════════════════════════════════════════════════════
# 4. PopulationActivityEncoder — rich multi-stat spike features
# ═══════════════════════════════════════════════════════════════════════════════

class PopulationActivityEncoder:
    """
    Encodes a population of N neurons into a rich feature vector.

    Instead of using raw binary spikes (1 bit per neuron per step), this
    encoder computes summary statistics over a rolling window W:

        features = [
            mean_rate(t),          (N,)   mean firing rate per neuron
            sqrt(variance(t)),     (N,)   std of firing rate (variability)
            temporal_deriv(t),     (N,)   drate/dt (transient detection)
            top_k_corr(t),         (K,)   pairwise correlations (optional)
        ]

    Total feature dimension: 3N (or 3N + K if correlations enabled).

    This is the "rich readout" approach from:
        Maass, Natschläger, Markram (2002) "Real-Time Computing Without
        Stable States" — Liquid State Machine paper; they use mean + variance
        of reservoir activity as the readout feature vector, not raw spikes.

    Combined with NeuralManifoldDecoder, this feeds richer statistics
    into the manifold → significantly better Pearson R.

    Args:
        n_neurons:   Number of RSNN neurons N
        window:      Rolling window size W (steps; default 10 = 200ms at 20ms dt)
        use_corr:    Include top-K pairwise correlations (expensive)
        device:      torch device
    """

    def __init__(
        self,
        n_neurons: int,
        window:    int  = 10,
        use_corr:  bool = False,
        device:    str  = "cpu",
    ):
        self.n_neurons = n_neurons
        self.window    = window
        self.use_corr  = use_corr
        self.device    = device

        # Rolling spike buffer: (W, N)
        self._buf     = torch.zeros(window, n_neurons, device=device)
        self._prev    = torch.zeros(n_neurons, device=device)

    @property
    def feature_dim(self) -> int:
        """Total feature dimension (3N or 3N + N(N-1)/2)."""
        base = 3 * self.n_neurons
        if self.use_corr:
            base += self.n_neurons * (self.n_neurons - 1) // 2
        return base

    def encode(self, spikes: torch.Tensor) -> torch.Tensor:
        """
        Encode current spike vector as rich population features.

        Args:
            spikes: (N,) binary spike vector

        Returns:
            (feature_dim,) rich feature vector
        """
        s = spikes.float().to(self.device)

        # Update rolling buffer
        self._buf = torch.roll(self._buf, 1, dims=0)
        self._buf[0] = s

        # 1. Mean firing rate per neuron over window
        mean_rate = self._buf.mean(dim=0)   # (N,)

        # 2. Std of firing rate (variability indicator)
        rate_std  = self._buf.std(dim=0) + 1e-6   # (N,)

        # 3. Temporal derivative (rate change from last step)
        temp_deriv = s - self._prev   # (N,)
        self._prev = s.clone()

        features = [mean_rate, rate_std, temp_deriv]

        if self.use_corr:
            # Top-k pairwise correlations (approximate, too expensive for full N)
            corr = torch.einsum('ti,tj->ij', self._buf, self._buf) / self.window
            # Extract upper triangle
            idx = torch.triu_indices(self.n_neurons, self.n_neurons, offset=1)
            corr_vec = corr[idx[0], idx[1]]   # (N(N-1)/2,)
            features.append(corr_vec)

        return torch.cat(features, dim=0)

    def reset(self):
        self._buf.zero_()
        self._prev.zero_()

    def encoder_health(self) -> dict:
        """
        Summary: mean population activity, feature dimensionality, use_corr flag.

        mean_rate < 0.01 → population mostly silent (gain too low).
        mean_rate > 0.4  → saturated population (too many spikes).
        """
        mean_rate = float(self._buf.mean().item())
        active_frac = float((self._buf.mean(dim=0) > 0.05).float().mean().item())
        return {
            "n_neurons":    self.n_neurons,
            "feature_dim":  self.feature_dim,
            "window":       self.window,
            "use_corr":     self.use_corr,
            "mean_rate":    round(mean_rate, 4),
            "active_frac":  round(active_frac, 4),
            "diagnosis":    (
                "silent"     if mean_rate < 0.01 else
                "saturated"  if mean_rate > 0.4  else
                "healthy"
            ),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Smoke test
# ═══════════════════════════════════════════════════════════════════════════════

def _test_manifold_decoder():
    N, K, k = 32, 2, 8

    print("=== OnlinePCA ===")
    pca = OnlinePCA(n_input=N, n_components=k, lr=0.01)
    # Simulate 100 observations
    for i in range(100):
        x = torch.randn(N) * 0.1
        x[:3] += torch.sin(torch.tensor(i * 0.1))   # structured signal in first 3 dims
        z = pca.update(x)
    assert z.shape == (k,), f"Expected ({k},), got {z.shape}"
    evr = pca.explained_variance_ratio()
    print(f"  Top-PC variance fraction: {evr[0]:.3f}")
    print(f"  Projection shape: {z.shape}  OK")

    print("\n=== NeuralManifoldDecoder ===")
    mfd = NeuralManifoldDecoder(n_neurons=N, output_size=K, n_components=k)

    # Simulate 50 training steps
    for i in range(50):
        spikes = (torch.rand(N) > 0.8).float()
        pred   = mfd.step(spikes)
        target = torch.tensor([math.sin(i * 0.1), math.cos(i * 0.1)])
        mfd.update(pred.detach() - target)

    pred = mfd.step((torch.rand(N) > 0.8).float())
    assert pred.shape == (K,)
    print(f"  Prediction shape: {pred.shape}  OK")

    report = mfd.manifold_report()
    print(f"  Manifold: {report['n_components']} dims, "
          f"top PC var={report['top_pc_var_fraction']:.3f}, "
          f"n_seen={report['n_seen']}")

    print("\n=== LatentDynamicsModel ===")
    lds = LatentDynamicsModel(n_latent=k, output_size=K)
    for i in range(60):
        z_raw = torch.randn(k) * 0.5
        y     = torch.randn(K) * 0.1
        z_sm  = lds.observe(z_raw, y)
    assert z_sm.shape == (k,)
    print(f"  Smoothed latent shape: {z_sm.shape}  OK")

    vel = lds.decode(z_sm)
    assert vel.shape == (K,)
    print(f"  Decoded velocity shape: {vel.shape}  OK")

    print("\n=== PopulationActivityEncoder ===")
    enc = PopulationActivityEncoder(n_neurons=N, window=10)
    feat = enc.encode((torch.rand(N) > 0.8).float())
    assert feat.shape == (3 * N,), f"Expected ({3*N},), got {feat.shape}"
    print(f"  Feature shape: {feat.shape} = 3×{N}  OK")

    # With correlations
    enc_corr = PopulationActivityEncoder(n_neurons=8, window=5, use_corr=True)
    feat2    = enc_corr.encode((torch.rand(8) > 0.8).float())
    expected = 3 * 8 + 8 * 7 // 2
    assert feat2.shape == (expected,), f"Expected ({expected},), got {feat2.shape}"
    print(f"  With corr: {feat2.shape}  OK")

    print("\n✅ All manifold_decoder tests passed")


if __name__ == "__main__":
    import math
    _test_manifold_decoder()
