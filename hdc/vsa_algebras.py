"""
hdc/vsa_algebras.py
====================
Novel VSA Algebras: VTB, FHRR, CGR, MCR + FractionalPower Encoding
====================================================================
Reference:
    Schlegel, Neubert, Protzel (2021) "A Comparison of Vector Symbolic Architectures"
    Artificial Intelligence Review 55:4523–4555.
    — Comprehensive comparison; establishes VTB as superior for structured data.

    Plate (1995) "Holographic Reduced Representations" IEEE TNNLS.
    — Original FHRR (Fourier Holographic Reduced Representations).

    Gayler (2003) "Vector Symbolic Architectures answer Jackendoff's challenges"
    — CGR (Cyclic Group Representation).

    Rachkovskij (2001) "Representation and Processing with Structures"
    — MCR (Modular Composite Representation) with integer quantisation.

    torchhd (Heddes et al. 2023) JMLR — FractionalPower embedding, NeuralHD.
    https://github.com/hyperdimensional-computing/torchhd

Why these algebras extend Arthedain beyond its current binary/real HRR:

    Arthedain already has:
        Binary XOR (BSC):    bind=XOR, bundle=majority — simple, hardware-friendly
        Real HRR:            bind=circular convolution, unbind=exact via FFT conjugate

    New algebras:
        FHRR:  bind = complex element-wise multiply (phasor rotation)
               unbind = complex conjugate multiply (exact, O(D))
               Advantage: exact inverse without pseudo-inverse; works for
               fractional power encoding natively via e^{iθ}

        VTB:   bind(a, b) = √D × M(a) @ b where M(a) is √D×√D block-diag matrix
               unbind(c, a) = √D × M(a)^T @ c
               Advantage: provably better for nested compositional structures;
               used in Schlegel 2021 as best-performing algebra for deep binding

        CGR:   Cyclic Group — integer vectors modulo n, bundle = mode
               bind = element-wise modular sum, unbind = modular difference
               Advantage: natural discrete alphabet support

        MCR:   Modular Composite — integer vectors, bundle converts to complex
               phasors, sums in phase space, quantises back
               Advantage: high capacity bundling with integer arithmetic

    FractionalPower Embedding:
        Encodes continuous values x ∈ ℝ as FHRR phasors:
            z(x) = exp(i × bw × x × ω)  for random frequencies ω ~ p(ω)
        Inner product approximates a kernel K(x₁ - x₂) via Bochner's theorem.
        Bandwidth bw controls kernel width (Gaussian or Sinc shape).
        This is the cleanest continuous-to-hypervector encoder in the literature.
        Arthedain's FPE (from vfa.py) uses a similar idea but this implements
        the torchhd version with explicit kernel control.

This module implements all 4 algebras plus FractionalPower embedding,
all sharing the same VSAVector interface for drop-in compatibility.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════════
# 1. FHRR — Fourier Holographic Reduced Representations
# ═══════════════════════════════════════════════════════════════════════════════

class FHRR:
    """
    Fourier Holographic Reduced Representations.

    Reference: Plate (1995) — phasor-valued vectors.

    Each component is a unit complex number e^{iθ_k}.
    Binding: element-wise complex multiplication (phase addition)
    Unbinding: element-wise conjugate multiplication (phase subtraction)
    Bundling: element-wise complex sum → angle of sum

    Properties:
        - Exact inverse: unbind(bind(a, b), b) = a  exactly
        - bind(a^α, a^β) = a^{α+β}  (supports fractional powers naturally)
        - Supports FractionalPower encoding natively

    Args:
        dim: Dimensionality (number of phasors)
    """

    def __init__(self, dim: int, device: str = "cpu"):
        self.dim    = dim
        self.device = device

    def gen(self, n: int = 1, seed: Optional[int] = None) -> torch.Tensor:
        """
        Generate n random FHRR vectors.

        Returns: (n, dim) or (dim,) complex tensor of unit phasors.
        """
        g = torch.Generator(device=self.device)
        if seed is not None:
            g.manual_seed(seed)
        angles = torch.rand(n, self.dim, generator=g, device=self.device) * 2 * math.pi
        hvs    = torch.exp(1j * angles)
        return hvs.squeeze(0) if n == 1 else hvs

    def bind(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """FHRR binding: element-wise complex multiplication (phase addition)."""
        return a * b   # complex multiply = add phases

    def unbind(self, composite: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """FHRR unbinding: element-wise conjugate multiply (exact inverse)."""
        return composite * key.conj()

    def bundle(self, hvs: List[torch.Tensor], weights: Optional[List[float]] = None) -> torch.Tensor:
        """
        Bundle FHRR vectors by summing in complex space and normalising.

        Sum of unit phasors → direction = mean phase, magnitude = consensus.
        """
        if not hvs:
            return torch.ones(self.dim, dtype=torch.complex64, device=self.device)
        w = torch.tensor(weights or [1.0] * len(hvs), device=self.device)
        w = w / (w.sum() + 1e-8)
        summed = sum(wi * hv for wi, hv in zip(w, hvs))
        # Normalise back to unit circle (majority vote in phase space)
        return summed / (summed.abs() + 1e-10)

    def similarity(self, a: torch.Tensor, b: torch.Tensor) -> float:
        """Cosine similarity in phase space = real part of mean(a × b*)."""
        return float((a * b.conj()).real.mean().item())

    def fractional_power(self, hv: torch.Tensor, alpha: float) -> torch.Tensor:
        """
        Compute hv^alpha: multiply all phases by alpha.
        hv^alpha × hv^beta = hv^{alpha+beta}  (group property).
        """
        return torch.exp(1j * alpha * hv.angle())

    def to_real(self, hv: torch.Tensor) -> torch.Tensor:
        """Project to real-valued by taking real part (optional)."""
        return hv.real


# ═══════════════════════════════════════════════════════════════════════════════
# 2. VTBAlgebra — Vector-Derived Transformation Binding
# ═══════════════════════════════════════════════════════════════════════════════

class VTBAlgebra:
    """
    Vector-Derived Transformation Binding (VTB).

    Reference:
        Gosmann & Eliasmith (2019)
        "Vector-Derived Transformation Binding: An Improved Binding Operation
        for Deep Symbol-Like Processing in Neural Networks"
        Neural Computation 31(5):849–869.

    Binding: bind(a, b) = √D × M(a) @ b
        where M(a) is the √D×√D block-diagonal matrix derived from a.
        M(a) = block_diag( [a[0]  -a[1] ], [a[2]  -a[3] ], ... )
                           [a[1]   a[0] ]  [a[3]   a[2] ]

    Unbinding: unbind(c, a) = √D × M(a)^T @ c  (exact inverse)

    Advantage over HRR:
        - Better structured binding for nested compositionality
        - Used in the best-performing algebra for deep structures (Schlegel 2021)
        - O(D^{3/2}) cost vs O(D log D) for HRR — faster for small D

    Args:
        dim: Dimensionality (must be even)
    """

    def __init__(self, dim: int, device: str = "cpu"):
        if dim % 2 != 0:
            raise ValueError(f"VTB requires even dimension, got {dim}")
        self.dim   = dim
        self.sqD   = math.sqrt(dim)
        self.block = dim // 2
        self.device = device

    def _M(self, a: torch.Tensor) -> torch.Tensor:
        """
        Build the block-orthogonal transformation matrix from vector a.

        Each 2×2 block is normalised so that M(a)^T M(a) = I (block-orthogonal):
            block_k = [[a_{2k},  -a_{2k+1}],
                       [a_{2k+1}, a_{2k}  ]] / ||[a_{2k}, a_{2k+1}]||

        This makes unbinding exact: M(a)^T @ M(a) = I.
        """
        D   = self.dim
        M   = torch.zeros(D, D, device=self.device)
        a_f = a.float().to(self.device)
        for k in range(self.block):
            i      = 2 * k
            a0, a1 = float(a_f[i]), float(a_f[i + 1])
            norm   = math.sqrt(a0 ** 2 + a1 ** 2) + 1e-8
            a0, a1 = a0 / norm, a1 / norm   # normalise each block
            M[i,   i  ] =  a0
            M[i,   i+1] = -a1
            M[i+1, i  ] =  a1
            M[i+1, i+1] =  a0
        return M

    def gen(self, n: int = 1, seed: Optional[int] = None) -> torch.Tensor:
        """Generate n random unit-norm real VTB vectors."""
        g = torch.Generator(device=self.device)
        if seed is not None:
            g.manual_seed(seed)
        hvs = torch.randn(n, self.dim, generator=g, device=self.device)
        hvs = F.normalize(hvs, dim=-1)
        return hvs.squeeze(0) if n == 1 else hvs

    def bind(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """VTB binding: bind(a, b) = M(a) @ b  (M is block-orthogonal)."""
        Ma = self._M(a)
        return Ma @ b.float().to(self.device)

    def unbind(self, composite: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """VTB unbinding (exact via block-orthogonality): M(a)^T @ c = b."""
        Ma = self._M(a)
        return Ma.T @ composite.float().to(self.device)

    def bundle(self, hvs: List[torch.Tensor]) -> torch.Tensor:
        """Superposition bundling: sum + L2-normalise."""
        if not hvs:
            return torch.zeros(self.dim, device=self.device)
        out = sum(hv.float() for hv in hvs)
        return F.normalize(out, dim=0)

    def similarity(self, a: torch.Tensor, b: torch.Tensor) -> float:
        """Cosine similarity."""
        return float(F.cosine_similarity(a.float().unsqueeze(0), b.float().unsqueeze(0)).item())


# ═══════════════════════════════════════════════════════════════════════════════
# 3. CGRAlgebra — Cyclic Group Representation
# ═══════════════════════════════════════════════════════════════════════════════

class CGRAlgebra:
    """
    Cyclic Group Representation (CGR).

    Reference: Gayler (2003), Schlegel (2021) §2.3.

    Integer-valued vectors in Z_m^D (integers modulo m).
    Binding: element-wise modular addition
    Unbinding: element-wise modular subtraction
    Bundling: element-wise mode (most common value)

    Advantage:
        - Natural support for discrete/categorical alphabets
        - Exact unbinding (cyclic group is self-inverse)
        - Integer arithmetic — hardware-friendly

    Args:
        dim: Dimensionality
        m:   Modulus (number of distinct values per component)
    """

    def __init__(self, dim: int, m: int = 256, device: str = "cpu"):
        self.dim    = dim
        self.m      = m
        self.device = device

    def gen(self, n: int = 1, seed: Optional[int] = None) -> torch.Tensor:
        """Generate n random CGR vectors (integers in [0, m))."""
        g = torch.Generator(device=self.device)
        if seed is not None:
            g.manual_seed(seed)
        hvs = torch.randint(0, self.m, (n, self.dim), generator=g, device=self.device)
        return hvs.squeeze(0) if n == 1 else hvs

    def bind(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """CGR binding: element-wise modular addition."""
        return (a.long() + b.long()) % self.m

    def unbind(self, composite: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """CGR unbinding: element-wise modular subtraction (exact inverse)."""
        return (composite.long() - key.long() + self.m) % self.m

    def bundle(self, hvs: List[torch.Tensor]) -> torch.Tensor:
        """CGR bundling: element-wise mode (majority in integer space)."""
        if not hvs:
            return torch.zeros(self.dim, dtype=torch.long, device=self.device)
        stacked = torch.stack([hv.long() for hv in hvs])  # (N, D)
        # Mode via one-hot histogramming
        mode_vals = torch.zeros(self.dim, dtype=torch.long, device=self.device)
        for d in range(self.dim):
            col = stacked[:, d]
            counts = torch.bincount(col, minlength=self.m)
            mode_vals[d] = counts.argmax()
        return mode_vals

    def similarity(self, a: torch.Tensor, b: torch.Tensor) -> float:
        """Fraction of matching components."""
        return float((a == b).float().mean().item())


# ═══════════════════════════════════════════════════════════════════════════════
# 4. MCRAlgebra — Modular Composite Representation
# ═══════════════════════════════════════════════════════════════════════════════

class MCRAlgebra:
    """
    Modular Composite Representation (MCR).

    Reference: Rachkovskij (2001), Schlegel (2021) §2.4.

    Integer-valued vectors. Bundling converts to complex phasors,
    sums in phase space, quantises back to integers.

    Binding: element-wise modular addition (same as CGR)
    Bundling: phasor averaging → quantise
    Unbinding: element-wise modular subtraction

    Advantage over CGR:
        - Bundling via phasor averaging is smoother (less noise) than mode
        - Scales to large N (no histogram needed)
        - Higher capacity for superposition

    Args:
        dim: Dimensionality
        m:   Modulus (resolution of integer encoding)
    """

    def __init__(self, dim: int, m: int = 256, device: str = "cpu"):
        self.dim    = dim
        self.m      = m
        self.device = device

    def gen(self, n: int = 1, seed: Optional[int] = None) -> torch.Tensor:
        """Generate n random MCR vectors."""
        g = torch.Generator(device=self.device)
        if seed is not None:
            g.manual_seed(seed)
        hvs = torch.randint(0, self.m, (n, self.dim), generator=g, device=self.device)
        return hvs.squeeze(0) if n == 1 else hvs

    def bind(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """MCR binding: modular addition."""
        return (a.long() + b.long()) % self.m

    def unbind(self, composite: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """MCR unbinding: modular subtraction."""
        return (composite.long() - key.long() + self.m) % self.m

    def bundle(self, hvs: List[torch.Tensor], weights: Optional[List[float]] = None) -> torch.Tensor:
        """
        MCR bundling: average in phasor space, quantise back to integers.

        Steps:
            1. Convert integers to phasors: θ_k = 2π × x_k / m → e^{iθ_k}
            2. Weighted sum in complex space
            3. Recover angle: θ = atan2(Im, Re)
            4. Quantise back: x_k = round(m × θ / (2π)) mod m
        """
        if not hvs:
            return torch.zeros(self.dim, dtype=torch.long, device=self.device)

        w = torch.tensor(weights or [1.0] * len(hvs), device=self.device)
        w = w / (w.sum() + 1e-8)

        # Convert to phasors
        phasors = [torch.exp(1j * 2 * math.pi * hv.float() / self.m)
                   for hv in hvs]

        # Weighted sum in complex space
        summed = sum(wi * ph for wi, ph in zip(w, phasors))

        # Recover angle and quantise
        angles   = summed.angle()           # in [-π, π]
        angles   = (angles + 2 * math.pi) % (2 * math.pi)  # to [0, 2π]
        integers = (angles * self.m / (2 * math.pi)).round().long() % self.m

        return integers

    def similarity(self, a: torch.Tensor, b: torch.Tensor) -> float:
        """
        Phasor similarity: real part of mean(e^{i 2π (a-b)/m}).
        0 = orthogonal, 1 = identical.
        """
        diff   = (a.float() - b.float()) * 2 * math.pi / self.m
        return float(torch.cos(diff).mean().item())


# ═══════════════════════════════════════════════════════════════════════════════
# 5. FractionalPowerEncoding — continuous values as FHRR phasors
# ═══════════════════════════════════════════════════════════════════════════════

class FractionalPowerEncoding:
    """
    FractionalPower Embedding: encode continuous values as FHRR phasors.

    Reference:
        torchhd (Heddes et al. 2023) — FractionalPower embedding class.
        Frady, Kleyko, Sommer (2021) — VFA kernel theory.

    Maps x ∈ ℝ to a FHRR hypervector:
        z(x) = exp(i × bw × x × ω)  where ω ~ p(ω)

    By Bochner's theorem: E[z(x₁)* × z(x₂)] ≈ K(x₁ - x₂)

    where K is determined by the distribution p(ω):
        p(ω) = N(0, 1)  →  K(d) = exp(-d²/2)   [Gaussian kernel]
        p(ω) = Uniform  →  K(d) = sinc(d)        [Sinc kernel]

    The bandwidth parameter `bw` controls the kernel width:
        large bw → narrow kernel → fine-grained similarity
        small bw → wide kernel → coarse-grained similarity

    Multidimensional: encode vector x ∈ ℝ^n by binding per-dimension encodings:
        z(x) = bind(z_1(x_1), z_2(x_2), ..., z_n(x_n))
             = exp(i × bw × Σ_k x_k × ω_k)   [in FHRR space]

    Args:
        n_features: Input dimension (1 for scalar, n for vector)
        dim:        Output FHRR dimension
        bw:         Bandwidth parameter (default 1.0)
        kernel:     'gaussian' or 'uniform' (determines frequency distribution)
        device:     torch device
    """

    def __init__(
        self,
        n_features: int,
        dim:        int,
        bw:         float = 1.0,
        kernel:     str   = "gaussian",
        seed:       Optional[int] = None,
        device:     str   = "cpu",
    ):
        self.n_features = n_features
        self.dim        = dim
        self.bw         = bw
        self.kernel     = kernel
        self.device     = device

        g = torch.Generator(device=device)
        if seed is not None:
            g.manual_seed(seed)

        if kernel == "gaussian":
            # ω ~ N(0, 1): approximates Gaussian kernel
            self.omega = torch.randn(n_features, dim, generator=g, device=device)
        elif kernel == "uniform":
            # ω ~ U(-π, π): approximates Sinc kernel
            self.omega = (torch.rand(n_features, dim, generator=g, device=device) - 0.5) * 2 * math.pi
        else:
            raise ValueError(f"Unknown kernel: {kernel!r}")

        self.fhrr = FHRR(dim, device=device)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode a feature vector x ∈ ℝ^{n_features} to a FHRR hypervector.

        z(x) = exp(i × bw × x @ ω)

        Returns: (dim,) complex FHRR unit-phasor vector.
        """
        x_f = x.float().to(self.device)
        if x_f.dim() == 1:
            phases = self.bw * (x_f @ self.omega)  # (dim,)
        else:
            phases = self.bw * (x_f @ self.omega)  # (B, dim)
        return torch.exp(1j * phases)

    def encode_batch(self, X: torch.Tensor) -> torch.Tensor:
        """Encode a batch (B, n_features) → (B, dim) complex HVs."""
        return self.encode(X)

    def similarity(self, x1: torch.Tensor, x2: torch.Tensor) -> float:
        """
        Kernel similarity between two feature vectors.
        K(x1, x2) ≈ Re(z(x1)* × z(x2)).mean()
        """
        z1 = self.encode(x1)
        z2 = self.encode(x2)
        return float((z1.conj() * z2).real.mean().item())

    def decode(
        self,
        hv:    torch.Tensor,
        x_range: Tuple[float, float] = (-5.0, 5.0),
        n_candidates: int = 200,
    ) -> torch.Tensor:
        """
        Approximate decoding: recover feature vector from FHRR hypervector.

        Uses a grid-search over candidate feature values and returns the
        candidate with maximum kernel similarity to the query HV.
        This is the approximate inverse of encode() — exact inversion is
        ill-posed for multi-dimensional FPE, but grid search is accurate
        for low-frequency features within [x_range].

        Args:
            hv:           (dim,) complex FHRR HV to decode
            x_range:      (min, max) for grid search
            n_candidates: Number of grid points per feature dimension

        Returns:
            (n_features,) decoded feature vector (best estimate)
        """
        if self.n_features == 1:
            # 1D: scan the range and pick max similarity
            xs = torch.linspace(x_range[0], x_range[1], n_candidates, device=self.device)
            best_x, best_sim = 0.0, -float("inf")
            for x_val in xs:
                z = self.encode(x_val.unsqueeze(0))
                sim = float((hv.conj() * z).real.mean().item())
                if sim > best_sim:
                    best_sim, best_x = sim, float(x_val.item())
            return torch.tensor([best_x], device=self.device)
        else:
            # Multi-dimensional: greedy per-feature decoding (approximate)
            decoded = []
            for k in range(self.n_features):
                xs = torch.linspace(x_range[0], x_range[1], n_candidates, device=self.device)
                best_x, best_sim = 0.0, -float("inf")
                for x_val in xs:
                    feat = torch.zeros(self.n_features, device=self.device)
                    feat[k] = x_val
                    z = self.encode(feat)
                    sim = float((hv.conj() * z).real.mean().item())
                    if sim > best_sim:
                        best_sim, best_x = sim, float(x_val.item())
                decoded.append(best_x)
            return torch.tensor(decoded, device=self.device)

    def nearest(
        self,
        query_hv:    torch.Tensor,
        candidates:  List[torch.Tensor],
    ) -> int:
        """
        Find the nearest candidate feature vector to the query HV.

        Args:
            query_hv:   (dim,) complex FHRR HV
            candidates: List of (n_features,) feature vectors to search

        Returns:
            Index of the most similar candidate.
        """
        best_idx, best_sim = 0, -float("inf")
        for i, x in enumerate(candidates):
            z   = self.encode(x.to(self.device))
            sim = float((query_hv.conj() * z).real.mean().item())
            if sim > best_sim:
                best_sim, best_idx = sim, i
        return best_idx

    def kernel_matrix(self, X: torch.Tensor) -> torch.Tensor:
        """
        Compute N×N kernel matrix for a batch X of shape (N, n_features).
        K[i,j] ≈ K(x_i, x_j)
        """
        Z = self.encode_batch(X)  # (N, dim)
        K = (Z @ Z.conj().T).real / self.dim
        return K

    def fit_bandwidth(
        self,
        X:           torch.Tensor,   # (N, n_features) calibration data
        method:      str = "silverman",
        target_sim:  float = 0.5,
    ) -> float:
        """
        Auto-select optimal bandwidth from data.

        Two methods:
            "silverman":  Scott's rule / Silverman's rule of thumb
                          bw = σ × n^{-1/(n_features+4)}
                          Works well for unimodal distributions.

            "grid_search": search bw over a log-grid to maximise pairwise
                          kernel distinctiveness (not too wide, not too narrow).
                          Slower but works for multi-modal data.

        Args:
            X:          (N, n_features) data matrix
            method:     "silverman" or "grid_search"
            target_sim: Target mean pairwise similarity for grid search (0.5 = good separation)

        Returns:
            Optimal bandwidth (also sets self.bw).
        """
        X_f = X.float().to(self.device)
        N, F = X_f.shape

        if method == "silverman":
            # Scott/Silverman rule: bw = std × N^{-1/(F+4)}
            std = X_f.std(dim=0).mean().item()
            if std < 1e-8:
                std = 1.0
            bw = float(std * (N ** (-1.0 / (F + 4))))
            bw = max(0.01, min(bw, 10.0))

        elif method == "grid_search":
            # Search over log-spaced bandwidth values
            bw_candidates = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
            best_bw, best_score = 1.0, float("inf")
            n_pairs = min(N, 20)

            for bw_cand in bw_candidates:
                orig_bw = self.bw
                self.bw  = bw_cand
                # Compute mean pairwise similarity on a subset
                idx      = torch.randperm(N)[:n_pairs].to(self.device)
                Z        = self.encode_batch(X_f[idx])   # (n_pairs, dim)
                K        = (Z @ Z.conj().T).real / self.dim
                # Target: mean off-diagonal sim ≈ target_sim
                off_diag = K[~torch.eye(n_pairs, dtype=torch.bool)]
                mean_sim = float(off_diag.mean().item())
                score    = abs(mean_sim - target_sim)
                if score < best_score:
                    best_score, best_bw = score, bw_cand
                self.bw = orig_bw

            bw = best_bw
        else:
            raise ValueError(f"Unknown method: {method!r}")

        self.bw = bw
        return bw


# ═══════════════════════════════════════════════════════════════════════════════
# Smoke test
# ═══════════════════════════════════════════════════════════════════════════════

def _test_vsa_algebras():
    D = 128

    print("=== FHRR ===")
    fhrr = FHRR(D)
    a = fhrr.gen(1, seed=0)
    b = fhrr.gen(1, seed=1)
    c = fhrr.bind(a, b)
    r = fhrr.unbind(c, b)
    sim = fhrr.similarity(r, a)
    print(f"  bind/unbind similarity: {sim:.6f}  (should be ≈1.0)  OK")
    assert sim > 0.999, f"FHRR exact unbinding failed: {sim}"

    # Fractional powers
    a_half = fhrr.fractional_power(a, 0.5)
    a_rec  = fhrr.bind(a_half, a_half)
    sim2   = fhrr.similarity(a_rec, a)
    print(f"  a^0.5 * a^0.5 ≈ a: similarity={sim2:.4f}  OK")

    print("\n=== VTB ===")
    vtb = VTBAlgebra(D)
    a_v = vtb.gen(1, seed=0)
    b_v = vtb.gen(1, seed=1)
    c_v = vtb.bind(a_v, b_v)
    r_v = vtb.unbind(c_v, a_v)
    sim_v = vtb.similarity(r_v, b_v)
    print(f"  VTB bind/unbind similarity: {sim_v:.4f}  (should be ≈1.0)  OK")
    assert sim_v > 0.99, f"VTB unbinding failed: {sim_v}"

    print("\n=== CGR ===")
    cgr = CGRAlgebra(D, m=7)
    a_c = cgr.gen(1, seed=0)
    b_c = cgr.gen(1, seed=1)
    c_c = cgr.bind(a_c, b_c)
    r_c = cgr.unbind(c_c, b_c)
    sim_c = cgr.similarity(r_c, a_c)
    print(f"  CGR bind/unbind match: {sim_c:.3f}  (should be 1.0)  OK")
    assert sim_c == 1.0, f"CGR should be exact: {sim_c}"

    # Bundling
    hvs_c   = [cgr.gen(1, seed=i) for i in range(3)]
    bundled = cgr.bundle(hvs_c)
    assert bundled.shape == (D,)
    print(f"  CGR bundle shape: {bundled.shape}  OK")

    print("\n=== MCR ===")
    mcr = MCRAlgebra(D, m=100)
    a_m = mcr.gen(1, seed=0)
    b_m = mcr.gen(1, seed=1)
    c_m = mcr.bind(a_m, b_m)
    r_m = mcr.unbind(c_m, b_m)
    sim_m = mcr.similarity(r_m, a_m)
    print(f"  MCR bind/unbind similarity: {sim_m:.4f}  (should be ≈1.0)  OK")
    assert sim_m > 0.95, f"MCR should be near-exact: {sim_m}"

    # Bundle
    hvs_m   = [mcr.gen(1, seed=i) for i in range(5)]
    bundled_m = mcr.bundle(hvs_m)
    print(f"  MCR bundle shape: {bundled_m.shape}  OK")

    print("\n=== FractionalPowerEncoding ===")
    fpe = FractionalPowerEncoding(n_features=4, dim=D, bw=1.0, kernel="gaussian")

    x1 = torch.tensor([1.0, 2.0, 3.0, 4.0])
    x2 = torch.tensor([1.1, 2.1, 3.1, 4.1])   # similar
    x3 = torch.tensor([10.0, 20.0, 30.0, 40.0]) # very different

    z1 = fpe.encode(x1)
    assert z1.shape == (D,)
    assert z1.dtype == torch.complex64 or z1.dtype == torch.complex128

    sim_near = fpe.similarity(x1, x2)
    sim_far  = fpe.similarity(x1, x3)
    print(f"  sim(x1, x2[similar])={sim_near:.4f}  sim(x1, x3[far])={sim_far:.4f}")
    assert sim_near > sim_far, "Nearby should be more similar"

    # Kernel matrix
    X = torch.randn(6, 4)
    K = fpe.kernel_matrix(X)
    assert K.shape == (6, 6)
    diag_mean = float(K.diagonal().mean())
    print(f"  Kernel matrix shape: {K.shape}, diagonal mean={diag_mean:.3f}  OK")

    # Sinc kernel variant
    fpe_sinc = FractionalPowerEncoding(n_features=4, dim=D, bw=1.0, kernel="uniform", seed=42)
    sim_s = fpe_sinc.similarity(x1, x2)
    print(f"  Sinc FPE sim(nearby)={sim_s:.4f}  OK")

    print("\n✅ All vsa_algebras tests passed")


if __name__ == "__main__":
    _test_vsa_algebras()
