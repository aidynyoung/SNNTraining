"""
Vector Function Architecture (VFA)
====================================
Frady, Kleyko, Kymn, Olshausen & Sommer (2021)
"Computing on Functions Using Randomized Vector Representations"
arXiv:2109.03429 — Intel Neuromorphic Computing Lab / UC Berkeley RCTN

Generalises VSA to function spaces via the formal connection to kernel methods
(Bochner's theorem, 1932).

Core principle (Theorem, §3.2 of paper):
  Encode a continuous value x ∈ ℝ as z(x) = exp(iωx) for random ω ~ p(ω).
  The inner product of two encodings approximates a kernel:
    <z(x₁), z(x₂)>_n ≈ K(x₁ - x₂)
  where K is the Fourier transform of p(ω) (Eq. 5, Bochner's theorem).

  Key kernels by choosing p(ω):
    Gaussian:   ω ~ N(0,σ²)    → K(d) = exp(-d²σ²/2)   [RBF kernel]
    Laplacian:  ω ~ Cauchy(0,λ) → K(d) = exp(-|d|/λ)
    Periodic:   ω = fixed freq  → K(d) = cos(ω₀d)

  Compatibility with VSA binding (Def 2, Eq. 10):
    z(x₁ + x₂) = z(x₁) ⊙ z(x₂)  [FHRR complex multiply]
  This means binding = translation in the encoded domain.

Implementations:

1. KernelEncoder — KLPE (Kernel Learning Phase Encoding) using FHRR
   Encodes scalars/vectors to complex HVs with specified kernel.
   Directly extends FractionalBinding to support multiple kernel types.

2. KernelHDCRegressor — Kernel regression in HV space (VFA learning)
   Encodes training data {(x_i, y_i)}, computes function HV F = Σ y_i z(x_i).
   Prediction: f(x) = Re(<z(x), F>) / n — purely HDC, no matrix inversion.
   Equivalent to kernel ridge regression at the zero-regularisation limit.

3. SpatialHDCEncoder — Multi-dimensional spatial position encoding
   Encodes 2D/3D positions as product of per-axis FHRR HVs:
     z(px, py) = z_x(px) ⊙ z_y(py)  [binding = tensor product of kernels]
   Gives a 2D Gaussian kernel: K(p₁, p₂) = K_x(px₁-px₂) × K_y(py₁-py₂).
   Directly useful for spatial world models and cognitive maps.

4. GaborHDCEncoder — Gabor wavelet encoding for image-like signals
   Encodes (position, orientation, frequency) tuples as HVs.
   Simulates V1-like receptive fields in HV space.

References:
  Frady et al. 2021 (arXiv:2109.03429) — VFA framework
  FractionalBinding (hdc/minirocket_hdc.py) — FHRR FPE implementation
  FractionalInterpolator (hdc/physics_world_model.py) — temporal position
  RecursiveBindingEncoder (hdc/sequence_vsa.py) — sequence positions
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from hdc.hdc_glue import hv_batch_sim


# ═══════════════════════════════════════════════════════════════════════════════
# 1. KernelEncoder — KLPE with specified kernel type
# ═══════════════════════════════════════════════════════════════════════════════

class KernelEncoder:
    """
    Kernel Learning Phase Encoding (KLPE) via random Fourier features.

    Encodes a scalar x as z(x) = exp(iωx) for random frequencies ω ~ p(ω).
    The inner product <z(x₁), z(x₂)> approximates the kernel K(x₁-x₂).

    Bochner's theorem (Eq. 5 of Frady et al. 2021):
      K(x-y) = ∫ p(ω) exp(iω(x-y)) dω = E_p[exp(iωx) exp(-iωy)]
    So z(x) = (exp(iω₁x), ..., exp(iωₙx)) for n sampled ωⱼ ~ p(ω),
    and <z(x), z(y)>_n → K(x-y) as n → ∞.

    Supported kernels:
      'gaussian':  ω ~ N(0, σ²)      → K(d) = exp(-σ²d²/2)
      'laplacian': ω ~ Cauchy(0, λ)  → K(d) = exp(-|d|/λ)
      'periodic':  ω = ω₀ (fixed)    → K(d) = cos(ω₀ d)

    The compatibility with FHRR binding (Def 2, VFA):
      z(x+y) = z(x) ⊙ z(y)  (element-wise complex product)
    means binding corresponds to translation in the encoded domain.

    Args:
        hd_dim: n — number of random frequencies (HV dimension)
        kernel: 'gaussian' | 'laplacian' | 'periodic'
        bandwidth: σ for Gaussian, λ for Laplacian, ω₀ for periodic
        seed: Random seed
    """

    def __init__(
        self,
        hd_dim: int = 4096,
        kernel: str = "gaussian",
        bandwidth: float = 1.0,
        seed: Optional[int] = None,
    ):
        self.hd_dim = hd_dim
        self.kernel = kernel
        self.bandwidth = bandwidth

        g = torch.Generator()
        if seed is not None:
            g.manual_seed(seed)

        # Sample random frequencies ω ~ p(ω) for the chosen kernel
        if kernel == "gaussian":
            # ω ~ N(0, σ²), σ = bandwidth
            self._omegas = torch.randn(hd_dim, generator=g) * bandwidth
        elif kernel == "laplacian":
            # ω ~ Cauchy(0, 1/λ), λ = bandwidth
            # Cauchy = Normal / Normal (ratio distribution)
            n1 = torch.randn(hd_dim, generator=g)
            n2 = torch.randn(hd_dim, generator=g)
            self._omegas = (n1 / n2.abs().clamp(min=1e-8)) / bandwidth
        elif kernel == "periodic":
            # Fixed frequency ω₀ = bandwidth, all dims get the same
            self._omegas = torch.full((hd_dim,), bandwidth)
        else:
            raise ValueError(f"Unknown kernel: {kernel}")

    def encode(self, x: float) -> torch.Tensor:
        """
        Encode scalar x as complex FHRR HV z(x) = exp(iωx).

        Args:
            x: Scalar value to encode

        Returns:
            (hd_dim,) complex64 HV with unit-magnitude components
        """
        angles = self._omegas * x           # (hd_dim,) real
        return torch.exp(1j * angles)       # (hd_dim,) complex, |z_k| = 1

    def encode_batch(self, X: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of scalars.

        Args:
            X: (N,) tensor of scalar values

        Returns:
            (N, hd_dim) complex64 tensor
        """
        # X: (N,), omegas: (D,) → outer product: (N, D)
        angles = X.unsqueeze(-1) * self._omegas.unsqueeze(0)  # (N, D)
        return torch.exp(1j * angles)

    def encode_vector(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode a d-dimensional vector by binding per-dimension encodings.

        z(x₁, ..., x_d) = z(x₁) ⊙ z(x₂) ⊙ ... ⊙ z(x_d)
        where ⊙ is element-wise complex multiply (FHRR binding).

        Each dimension gets its own slice of omegas.

        Args:
            x: (d,) vector

        Returns:
            (hd_dim,) complex HV
        """
        d = x.shape[0]
        # Assign D/d omegas to each input dimension
        chunk = self.hd_dim // d
        result = torch.ones(self.hd_dim, dtype=torch.complex64)
        for k in range(d):
            start = k * chunk
            end = min(start + chunk, self.hd_dim)
            angles = self._omegas[start:end] * x[k].item()
            result[start:end] = torch.exp(1j * angles)
        return result

    def kernel_value(self, x1: float, x2: float) -> float:
        """
        Estimate K(x₁ - x₂) from the random feature approximation.

        Returns:
            Real part of <z(x₁), z(x₂)>_n / n
        """
        z1 = self.encode(x1)
        z2 = self.encode(x2)
        return float((z1 * z2.conj()).real.mean())

    def theoretical_kernel(self, d: float) -> float:
        """
        Exact kernel value K(d) for distance d (ground truth).

        Gaussian: K(d) = exp(-σ²d²/2)
        Laplacian: K(d) = exp(-|d|/λ) approximately
        Periodic: K(d) = cos(ω₀d)
        """
        if self.kernel == "gaussian":
            return float(math.exp(-self.bandwidth**2 * d**2 / 2))
        elif self.kernel == "laplacian":
            return float(math.exp(-abs(d) / self.bandwidth))
        elif self.kernel == "periodic":
            return float(math.cos(self.bandwidth * d))
        return 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# 2. KernelHDCRegressor — VFA-based kernel regression
# ═══════════════════════════════════════════════════════════════════════════════

class KernelHDCRegressor:
    """
    Kernel regression in HV space using VFA.

    Fits a function f: ℝ → ℝ from training data {(x_i, y_i)}.
    The function is represented as a HV:
        F = (1/n) Σᵢ yᵢ z(xᵢ)

    Prediction at query x:
        f(x) ≈ Re(<z(x), F>) = Re(z(x)^H F)
             = Re((1/n) Σᵢ yᵢ <z(x), z(xᵢ)>)
             = (1/n) Σᵢ yᵢ K(x - xᵢ)    [kernel Nadaraya-Watson regression]

    This is equivalent to kernel regression with the kernel specified by
    the encoder, but computed purely via HDC inner products — no explicit
    kernel matrix, no O(n²) computation.

    Multi-output: encode each x_i separately, accumulate per-output HVs.

    Args:
        encoder: KernelEncoder to use
    """

    def __init__(self, encoder: KernelEncoder):
        self.encoder = encoder
        self._function_hv: Optional[torch.Tensor] = None
        self._n = 0

    def fit(self, X: torch.Tensor, y: torch.Tensor):
        """
        Fit by accumulating y_i × z(x_i) into the function HV.

        Args:
            X: (N,) training inputs (scalars)
            y: (N,) training targets
        """
        N = X.shape[0]
        Z = self.encoder.encode_batch(X)   # (N, D) complex

        # F = (1/N) Σ yᵢ z(xᵢ)
        y_c = y.float().unsqueeze(-1).to(torch.complex64)   # (N, 1)
        self._function_hv = (y_c * Z).mean(dim=0)           # (D,) complex
        self._n = N

    def predict(self, x: float) -> float:
        """
        Predict f(x) ≈ Re(<z(x), F>).

        Args:
            x: Query scalar

        Returns:
            Predicted function value
        """
        assert self._function_hv is not None, "Call fit() first"
        z_x = self.encoder.encode(x)               # (D,) complex
        inner = (z_x.conj() * self._function_hv).real.mean()
        return float(inner)

    def predict_batch(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predict for a batch of query points.

        Args:
            X: (M,) query scalars

        Returns:
            (M,) predicted values
        """
        assert self._function_hv is not None
        Z_q = self.encoder.encode_batch(X)          # (M, D) complex
        F = self._function_hv.unsqueeze(0)           # (1, D) complex
        preds = (Z_q.conj() * F).real.mean(dim=-1)  # (M,)
        return preds


# ═══════════════════════════════════════════════════════════════════════════════
# 3. SpatialHDCEncoder — multi-dim spatial position encoding
# ═══════════════════════════════════════════════════════════════════════════════

class SpatialHDCEncoder:
    """
    2D/3D spatial position encoding via FHRR binding of per-axis encoders.

    Encodes position (px, py) as:
        z(px, py) = z_x(px) ⊙ z_y(py)    [element-wise complex multiply]

    This induces a 2D kernel as the product of per-axis kernels:
        K((p₁,q₁), (p₂,q₂)) = K_x(p₁-p₂) × K_y(q₁-q₂)

    For Gaussian encoders: K = 2D Gaussian (isotropic if σ_x = σ_y).

    Applications:
      - Cognitive map encoding (spatial world models)
      - Place cell representation in HDC
      - Image feature encoding

    Args:
        hd_dim: D — HV dimension shared across all spatial axes
        dims: Number of spatial dimensions (2 or 3)
        kernel: Kernel type for each axis
        bandwidth: Kernel bandwidth per axis (scalar or list)
        seed: Random seed
    """

    def __init__(
        self,
        hd_dim: int = 4096,
        dims: int = 2,
        kernel: str = "gaussian",
        bandwidth: float = 1.0,
        seed: Optional[int] = None,
    ):
        self.hd_dim = hd_dim
        self.dims = dims

        # Each spatial axis gets its own KernelEncoder (different omegas)
        self._axis_encoders = [
            KernelEncoder(
                hd_dim=hd_dim,
                kernel=kernel,
                bandwidth=bandwidth,
                seed=(seed or 0) + i * 1000,
            )
            for i in range(dims)
        ]

    def encode(self, position: torch.Tensor) -> torch.Tensor:
        """
        Encode a spatial position as a complex HV.

        Args:
            position: (dims,) position tensor

        Returns:
            (hd_dim,) complex HV
        """
        assert len(position) == self.dims
        result = torch.ones(self.hd_dim, dtype=torch.complex64)
        for k, enc in enumerate(self._axis_encoders):
            z_k = enc.encode(float(position[k]))
            result = result * z_k        # FHRR bind (multiply phasors)
        return result

    def encode_batch(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of positions.

        Args:
            positions: (N, dims) position matrix

        Returns:
            (N, hd_dim) complex HV matrix
        """
        N = positions.shape[0]
        result = torch.ones(N, self.hd_dim, dtype=torch.complex64)
        for k, enc in enumerate(self._axis_encoders):
            z_k = enc.encode_batch(positions[:, k])   # (N, D)
            result = result * z_k
        return result

    def kernel_value(
        self,
        pos1: torch.Tensor,
        pos2: torch.Tensor,
    ) -> float:
        """
        2D kernel K(pos1, pos2) = product of per-axis kernels.

        Args:
            pos1, pos2: (dims,) position tensors

        Returns:
            Kernel value ∈ [0, 1]
        """
        z1 = self.encode(pos1)
        z2 = self.encode(pos2)
        return float((z1 * z2.conj()).real.mean())

    def similarity_field(
        self,
        query: torch.Tensor,
        grid_points: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute similarity of query position to all grid points.

        Args:
            query: (dims,) query position
            grid_points: (N, dims) grid positions

        Returns:
            (N,) similarity values ∈ [0, 1]
        """
        z_query = self.encode(query)                    # (D,)
        Z_grid  = self.encode_batch(grid_points)        # (N, D)
        # Inner product similarity
        sims = (Z_grid * z_query.conj().unsqueeze(0)).real.mean(dim=-1)  # (N,)
        return sims


# ═══════════════════════════════════════════════════════════════════════════════
# 4. GaborHDCEncoder — Gabor wavelet encoding (V1-like)
# ═══════════════════════════════════════════════════════════════════════════════

class GaborHDCEncoder:
    """
    Gabor wavelet encoding for image-like signals in HV space.

    Encodes (position, orientation, spatial_freq) tuples as HVs:
        z(x, y, θ, f) = z_pos(x,y) ⊙ z_ori(θ) ⊙ z_freq(f)

    The spatial component uses a 2D Gaussian kernel (SpatialHDCEncoder).
    Orientation and frequency are encoded with periodic/Gaussian kernels.

    This mimics V1 simple cell receptive fields in HV space — each
    neuron is tuned to a position, orientation, and spatial frequency.

    Applications:
      - Image feature extraction
      - Texture classification
      - Optical flow encoding

    Args:
        hd_dim: HV dimension
        spatial_bandwidth: σ for spatial Gaussian kernel
        orientation_bandwidth: σ for orientation Gaussian kernel
        seed: Random seed
    """

    def __init__(
        self,
        hd_dim: int = 4096,
        spatial_bandwidth: float = 0.1,     # σ_spatial
        orientation_bandwidth: float = 0.5,  # σ_θ
        frequency_bandwidth: float = 1.0,    # σ_f
        seed: int = 0,
    ):
        self.hd_dim = hd_dim

        self.spatial_enc = SpatialHDCEncoder(
            hd_dim, dims=2, kernel="gaussian",
            bandwidth=spatial_bandwidth, seed=seed,
        )
        self.orientation_enc = KernelEncoder(
            hd_dim, kernel="periodic",
            bandwidth=orientation_bandwidth, seed=seed+1,
        )
        self.frequency_enc = KernelEncoder(
            hd_dim, kernel="gaussian",
            bandwidth=frequency_bandwidth, seed=seed+2,
        )

    def encode(
        self,
        x: float,
        y: float,
        theta: float = 0.0,
        freq: float = 1.0,
    ) -> torch.Tensor:
        """
        Encode a Gabor receptive field as a complex HV.

        Args:
            x, y: Spatial position
            theta: Preferred orientation (radians)
            freq: Preferred spatial frequency

        Returns:
            (hd_dim,) complex HV
        """
        z_pos   = self.spatial_enc.encode(torch.tensor([x, y]))
        z_theta = self.orientation_enc.encode(theta)
        z_freq  = self.frequency_enc.encode(freq)
        return z_pos * z_theta * z_freq   # FHRR binding

    def response(
        self,
        image_features: List[Tuple[float, float, float]],
        x: float,
        y: float,
        theta: float,
        freq: float,
    ) -> float:
        """
        Compute Gabor filter response at (x,y,θ,f) to a set of image features.

        image_features: List of (x_i, y_i, intensity_i) tuples
        Returns: Real-valued response
        """
        z_filter = self.encode(x, y, theta, freq)
        response = 0.0
        for xi, yi, intensity in image_features:
            z_feat = self.encode(xi, yi, theta, freq)
            response += intensity * float((z_filter * z_feat.conj()).real.mean())
        return response

    def feature_map(
        self,
        positions: List[Tuple[float, float]],
        theta:     float = 0.0,
        freq:      float = 1.0,
    ) -> torch.Tensor:
        """
        Compute a feature map HV by bundling all position encodings.

        This is the holographic representation of a V1 orientation column:
        all spatial positions at the same (θ, f) bundled into one HV.

        Args:
            positions: List of (x, y) positions to encode
            theta:     Orientation
            freq:      Spatial frequency

        Returns:
            (hd_dim,) complex bundle HV representing all positions.
        """
        hvs = [self.encode(x, y, theta, freq) for x, y in positions]
        if not hvs:
            return torch.zeros(self.hd_dim, dtype=torch.cfloat)
        return torch.stack(hvs).mean(dim=0)

    def encode_batch(
        self,
        positions: List[Tuple[float, float, float, float]],
    ) -> torch.Tensor:
        """
        Batch encode: List of (x, y, theta, freq) tuples → (N, hd_dim) complex.
        """
        hvs = [self.encode(x, y, t, f) for x, y, t, f in positions]
        return torch.stack(hvs) if hvs else torch.zeros(0, self.hd_dim, dtype=torch.cfloat)


# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_kernel_encoder():
    print("=" * 60)
    print("Testing KernelEncoder (VFA, Frady/Kleyko/Sommer 2021)")
    print("=" * 60)

    torch.manual_seed(42)
    D = 10000

    for kernel_type in ["gaussian", "laplacian", "periodic"]:
        enc = KernelEncoder(hd_dim=D, kernel=kernel_type, bandwidth=1.0, seed=0)

        # Verify kernel approximation: K̂(d) ≈ K(d) for various d
        test_ds = [0.0, 0.5, 1.0, 2.0]
        print(f"\n  {kernel_type} kernel:")
        for d in test_ds:
            estimated = enc.kernel_value(0.0, d)
            theoretical = enc.theoretical_kernel(d)
            error = abs(estimated - theoretical)
            print(f"    K̂({d}) = {estimated:.4f}  K({d}) = {theoretical:.4f}  err={error:.4f}")
            # At D=10000, error should be small (< 0.05)
            assert error < 0.1, f"Kernel approximation too poor: err={error:.4f}"

        # FHRR binding: z(x+y) = z(x) ⊙ z(y)
        z1 = enc.encode(1.5)
        z2 = enc.encode(0.5)
        z_sum = enc.encode(2.0)
        z_bound = z1 * z2
        # z1 * z2 should ≈ z(2.0)
        agreement = float((z_bound.conj() * z_sum).real.mean())
        print(f"    FHRR binding: z(1.5)⊙z(0.5)≈z(2.0): {agreement:.4f}  (want ≈ 1.0)")
        assert agreement > 0.95, f"FHRR binding violated: {agreement}"

    print("\n  ✅ KernelEncoder OK")


def test_kernel_regressor():
    print("=" * 60)
    print("Testing KernelHDCRegressor (VFA kernel regression)")
    print("=" * 60)

    torch.manual_seed(7)
    D = 5000
    enc = KernelEncoder(hd_dim=D, kernel="gaussian", bandwidth=2.0, seed=1)
    reg = KernelHDCRegressor(enc)

    # Fit: f(x) = sin(x)
    X_train = torch.linspace(-math.pi, math.pi, 50)
    y_train = torch.sin(X_train)
    reg.fit(X_train, y_train)

    # Predict at test points
    X_test = torch.linspace(-math.pi, math.pi, 20)
    y_pred = reg.predict_batch(X_test)
    y_true = torch.sin(X_test)

    mse = float(((y_pred - y_true)**2).mean())
    print(f"  MSE(sin): {mse:.6f}")
    print(f"  Sample: pred={y_pred[:3].tolist()} true={y_true[:3].tolist()}")
    assert mse < 0.5, f"MSE too high: {mse}"

    print("  ✅ KernelHDCRegressor OK")


def test_spatial_encoder():
    print("=" * 60)
    print("Testing SpatialHDCEncoder (2D Gaussian kernel)")
    print("=" * 60)

    torch.manual_seed(0)
    enc = SpatialHDCEncoder(hd_dim=5000, dims=2, kernel="gaussian",
                             bandwidth=1.0, seed=42)

    # Nearby positions should be more similar than distant ones
    p0 = torch.tensor([0.0, 0.0])
    p1 = torch.tensor([0.1, 0.1])  # close
    p2 = torch.tensor([3.0, 3.0])  # far

    k_near = enc.kernel_value(p0, p1)
    k_far  = enc.kernel_value(p0, p2)
    print(f"  K(close): {k_near:.4f}  K(far): {k_far:.4f}")
    assert k_near > k_far, "Closer points should have higher kernel"

    # 2D similarity field
    grid = torch.tensor([[x, y] for x in [-1.,0.,1.] for y in [-1.,0.,1.]])
    query = torch.tensor([0.0, 0.0])
    sims = enc.similarity_field(query, grid)
    center_idx = 4   # [0,0] is index 4 in the 3×3 grid
    print(f"  Similarity field at center: {float(sims[center_idx]):.4f}  (want highest)")
    assert float(sims[center_idx]) == float(sims.max()), "Center should have max similarity"

    print("  ✅ SpatialHDCEncoder OK")


def test_gabor_encoder():
    print("=" * 60)
    print("Testing GaborHDCEncoder (V1-like encoding)")
    print("=" * 60)

    torch.manual_seed(3)
    enc = GaborHDCEncoder(hd_dim=3000, spatial_bandwidth=0.2, seed=0)

    # Encode and verify shapes
    z = enc.encode(x=0.5, y=0.5, theta=0.0, freq=1.0)
    print(f"  Gabor HV shape: {z.shape}")
    assert z.shape == (3000,)
    assert z.dtype == torch.complex64

    # Two Gabor filters with same orientation and position should be similar
    z1 = enc.encode(x=0.5, y=0.5, theta=0.0, freq=1.0)
    z2 = enc.encode(x=0.5, y=0.5, theta=0.0, freq=1.0)
    z3 = enc.encode(x=5.0, y=5.0, theta=1.5, freq=5.0)   # very different

    sim_same = float((z1 * z2.conj()).real.mean())
    sim_diff = float((z1 * z3.conj()).real.mean())
    print(f"  sim(same Gabor) = {sim_same:.4f}  sim(diff Gabor) = {sim_diff:.4f}")
    assert sim_same > sim_diff, "Same Gabor should be more similar"

    print("  ✅ GaborHDCEncoder OK")


if __name__ == "__main__":
    test_kernel_encoder()
    print()
    test_kernel_regressor()
    print()
    test_spatial_encoder()
    print()
    test_gabor_encoder()
    print()
    print("=== All VFA tests passed ===")
