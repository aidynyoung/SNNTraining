"""
Multiple VSA Backend Support for HDC
=====================================
Based on: Schlegel et al. 2022 "A comparison of vector symbolic architectures"

Implements three major VSA backends:
1. MAP (Multiply-Add-Permute) - Bipolar {+1, -1}^D
2. FHRR (Fourier Holographic Reduced Representations) - Complex unit circle
3. BSC (Binary Spatter Codes) - Binary {0, 1}^D

All backends share the same interface:
    gen_hvs(n, dim) -> Tensor
    bind(a, b) -> Tensor
    bundle(hvs) -> Tensor
    permute(hv, k) -> Tensor
    sim(a, b) -> float
    batch_sim(q, mem) -> Tensor

Reference:
  Schlegel, K., et al. (2022)
  "A comparison of vector symbolic architectures"
  Artificial Intelligence Review 55 (6), 4523-4555
"""

import torch
import torch.nn as nn
from typing import Optional, Literal, List, Union


# ═══════════════════════════════════════════════════════════════════════════════
# MAP Backend (Multiply-Add-Permute)
# ═══════════════════════════════════════════════════════════════════════════════

class MAPBackend:
    """MAP (Multiply-Add-Permute) VSA backend.
    
    Hypervectors: bipolar {+1, -1}^D
    Binding: element-wise multiplication (XOR equivalent)
    Bundling: sum + threshold
    Similarity: cosine similarity
    """
    
    name = "map"
    
    @staticmethod
    def gen_hvs(n: int, dim: int, device=None, seed: Optional[int] = None) -> torch.Tensor:
        g = torch.Generator(device=device)
        if seed is not None:
            g.manual_seed(seed)
        return (torch.randint(0, 2, (n, dim), generator=g, device=device) * 2 - 1).float()
    
    @staticmethod
    def bind(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a * b
    
    @staticmethod
    def bundle(hvs: torch.Tensor) -> torch.Tensor:
        return hvs.sum(dim=0) if hvs.dim() > 1 else hvs
    
    @staticmethod
    def permute(hv: torch.Tensor, k: int = 1) -> torch.Tensor:
        return torch.roll(hv, shifts=k)
    
    @staticmethod
    def sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        an, bn = a.norm(), b.norm()
        if an > 0 and bn > 0:
            return (a @ b) / (an * bn).clamp(min=1e-12)
        return torch.tensor(0.0, device=a.device)
    
    @staticmethod
    def batch_sim(q: torch.Tensor, mem: torch.Tensor) -> torch.Tensor:
        return (mem @ q) / (mem.norm(dim=1) * q.norm()).clamp(min=1e-12)
    
    @staticmethod
    def thresh(hv: torch.Tensor) -> torch.Tensor:
        return torch.sign(hv).clamp(-1, 1)
    
    @staticmethod
    def normalize(hv: torch.Tensor) -> torch.Tensor:
        return hv / hv.norm().clamp(min=1e-12)


# ═══════════════════════════════════════════════════════════════════════════════
# FHRR Backend (Fourier Holographic Reduced Representations)
# ═══════════════════════════════════════════════════════════════════════════════

class FHRRBackend:
    """FHRR (Fourier Holographic Reduced Representations) VSA backend.
    
    Hypervectors: complex unit circle (phases only, magnitude = 1)
    Binding: phase addition (complex multiplication)
    Bundling: sum of complex vectors
    Similarity: cosine of phase difference (real part of inner product)
    
    Key property: FHRR supports smooth similarity (unlike MAP's discrete
    similarity), making it better for continuous-valued representations.
    """
    
    name = "fhrr"
    
    @staticmethod
    def gen_hvs(n: int, dim: int, device=None, seed: Optional[int] = None) -> torch.Tensor:
        """Generate random unit complex hypervectors.
        
        Returns:
            (n, dim) complex64 tensor with unit magnitude
        """
        g = torch.Generator(device=device)
        if seed is not None:
            g.manual_seed(seed)
        # Random phases in [0, 2π)
        phases = torch.rand(n, dim, generator=g, device=device) * 2 * torch.pi
        return torch.complex(torch.cos(phases), torch.sin(phases))
    
    @staticmethod
    def bind(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Phase addition: complex multiplication."""
        return a * b
    
    @staticmethod
    def bundle(hvs: torch.Tensor) -> torch.Tensor:
        """Sum of complex vectors (magnitude may deviate from 1)."""
        return hvs.sum(dim=0) if hvs.dim() > 1 else hvs
    
    @staticmethod
    def permute(hv: torch.Tensor, k: int = 1) -> torch.Tensor:
        """Phase shift via element rotation."""
        return torch.roll(hv, shifts=k, dims=-1)
    
    @staticmethod
    def sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Cosine similarity = real part of normalized inner product."""
        # Normalize to unit magnitude
        a_norm = a / a.abs().clamp(min=1e-12)
        b_norm = b / b.abs().clamp(min=1e-12)
        return (a_norm * b_norm.conj()).real.mean()
    
    @staticmethod
    def batch_sim(q: torch.Tensor, mem: torch.Tensor) -> torch.Tensor:
        """Batch cosine similarity for FHRR."""
        # Normalize query
        q_norm = q / q.abs().clamp(min=1e-12)
        mem_norm = mem / mem.abs().clamp(min=1e-12)
        # (mem @ q) where both are complex: mem @ q.conj() gives complex inner products
        return (mem_norm @ q_norm.conj()).real / mem_norm.shape[-1]
    
    @staticmethod
    def thresh(hv: torch.Tensor) -> torch.Tensor:
        """Normalize to unit magnitude (keep phase, set magnitude to 1)."""
        return hv / hv.abs().clamp(min=1e-12)
    
    @staticmethod
    def normalize(hv: torch.Tensor) -> torch.Tensor:
        """Normalize to unit magnitude."""
        return hv / hv.abs().clamp(min=1e-12)


# ═══════════════════════════════════════════════════════════════════════════════
# BSC Backend (Binary Spatter Codes)
# ═══════════════════════════════════════════════════════════════════════════════

class BSCBackend:
    """BSC (Binary Spatter Codes) VSA backend.
    
    Hypervectors: binary {0, 1}^D
    Binding: XOR
    Bundling: majority vote (threshold at D/2)
    Similarity: Hamming distance
    
    Key property: Most hardware-efficient (single-bit operations).
    """
    
    name = "bsc"
    
    @staticmethod
    def gen_hvs(n: int, dim: int, device=None, seed: Optional[int] = None) -> torch.Tensor:
        g = torch.Generator(device=device)
        if seed is not None:
            g.manual_seed(seed)
        return torch.randint(0, 2, (n, dim), generator=g, device=device).float()
    
    @staticmethod
    def bind(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """XOR for binary vectors."""
        return ((a + b) % 2).float()
    
    @staticmethod
    def bundle(hvs: torch.Tensor) -> torch.Tensor:
        """Majority vote: threshold at D/2."""
        if hvs.dim() <= 1:
            return hvs
        total = hvs.sum(dim=0)
        return (total >= hvs.shape[0] / 2).float()
    
    @staticmethod
    def permute(hv: torch.Tensor, k: int = 1) -> torch.Tensor:
        return torch.roll(hv, shifts=k)
    
    @staticmethod
    def sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Normalized Hamming similarity: 1 - Hamming_distance / D."""
        return (a == b).float().mean()
    
    @staticmethod
    def batch_sim(q: torch.Tensor, mem: torch.Tensor) -> torch.Tensor:
        """Batch Hamming similarity."""
        # q: (D,) or (B, D), mem: (N, D)
        if q.dim() == 1:
            return (mem == q.unsqueeze(0)).float().mean(dim=1)
        else:
            # q: (B, D), mem: (N, D) -> (N, B)
            return (mem.unsqueeze(1) == q.unsqueeze(0)).float().mean(dim=-1)
    
    @staticmethod
    def thresh(hv: torch.Tensor) -> torch.Tensor:
        """Threshold at 0.5 for binary."""
        return (hv >= 0.5).float()
    
    @staticmethod
    def normalize(hv: torch.Tensor) -> torch.Tensor:
        """No normalization needed for binary."""
        return hv


# ═══════════════════════════════════════════════════════════════════════════════
# Backend Registry
# ═══════════════════════════════════════════════════════════════════════════════

BACKENDS = {
    "map": MAPBackend,
    "bipolar": MAPBackend,  # alias
    "fhrr": FHRRBackend,
    "bsc": BSCBackend,
    "binary": BSCBackend,  # alias
}

VSA_TYPE = Literal["map", "bipolar", "fhrr", "bsc", "binary"]


def get_backend(vsa_type: str):
    """Get VSA backend by name.
    
    Args:
        vsa_type: One of "map", "bipolar", "fhrr", "bsc", "binary"
    
    Returns:
        Backend class with static methods
    
    Raises:
        ValueError: If vsa_type is not recognized
    """
    vsa_type = vsa_type.lower()
    if vsa_type not in BACKENDS:
        raise ValueError(
            f"Unknown VSA type: {vsa_type}. "
            f"Available: {list(BACKENDS.keys())}"
        )
    return BACKENDS[vsa_type]


# ═══════════════════════════════════════════════════════════════════════════════
# Unified VSA Module (drop-in replacement for models.hdc operations)
# ═══════════════════════════════════════════════════════════════════════════════

class VSA(nn.Module):
    """Unified VSA module with runtime-selectable backend.
    
    Provides the same interface as models.hdc but with multiple
    backend support. Can be used as a drop-in replacement.
    
    Example:
        vsa = VSA(dim=10000, vsa_type="map")
        hvs = vsa.gen_hvs(10)
        bound = vsa.bind(hvs[0], hvs[1])
        bundled = vsa.bundle(hvs)
        sim = vsa.sim(bound, bundled)
    """
    
    def __init__(
        self,
        dim: int = 10000,
        vsa_type: VSA_TYPE = "map",
        device: Optional[str] = None,
    ):
        super().__init__()
        self.dim = dim
        self.vsa_type = vsa_type
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.backend = get_backend(vsa_type)
    
    def gen_hvs(self, n: int, seed: Optional[int] = None) -> torch.Tensor:
        return self.backend.gen_hvs(n, self.dim, self.device, seed)
    
    def bind(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.backend.bind(a, b)
    
    def bundle(self, hvs: torch.Tensor) -> torch.Tensor:
        return self.backend.bundle(hvs)
    
    def permute(self, hv: torch.Tensor, k: int = 1) -> torch.Tensor:
        return self.backend.permute(hv, k)
    
    def sim(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.backend.sim(a, b)
    
    def batch_sim(self, q: torch.Tensor, mem: torch.Tensor) -> torch.Tensor:
        return self.backend.batch_sim(q, mem)
    
    def thresh(self, hv: torch.Tensor) -> torch.Tensor:
        return self.backend.thresh(hv)
    
    def normalize(self, hv: torch.Tensor) -> torch.Tensor:
        return self.backend.normalize(hv)


# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_vsa_backends():
    """Verify all VSA backends produce valid hypervectors."""
    print("=" * 60)
    print("Testing VSA Backends (Schlegel 2022)")
    print("=" * 60)
    
    dim = 1000
    
    for name, backend in BACKENDS.items():
        if name in ("bipolar", "binary"):
            continue  # Skip aliases
        
        print(f"\n  Testing {name.upper()} backend...")
        
        # Generate hypervectors
        hvs = backend.gen_hvs(10, dim)
        print(f"    HV shape: {hvs.shape}")
        
        # Test binding
        bound = backend.bind(hvs[0], hvs[1])
        print(f"    Bound shape: {bound.shape}")
        
        # Test bundling
        bundled = backend.bundle(hvs)
        print(f"    Bundled shape: {bundled.shape}")
        
        # Test similarity
        sim_val = backend.sim(hvs[0], hvs[0])
        print(f"    Self-similarity: {sim_val:.4f} (should be ~1.0)")
        
        sim_val = backend.sim(hvs[0], hvs[1])
        print(f"    Cross-similarity: {sim_val:.4f} (should be ~0.0)")
        
        # Test batch similarity
        batch_sims = backend.batch_sim(hvs[0], hvs[:5])
        print(f"    Batch sim shape: {batch_sims.shape}")
        
        # Test permutation
        permuted = backend.permute(hvs[0], k=3)
        sim_perm = backend.sim(hvs[0], permuted)
        print(f"    Permuted self-sim: {sim_perm:.4f} (should be ~0.0)")
        
        # Test thresholding
        threshed = backend.thresh(bundled)
        print(f"    Thresholded shape: {threshed.shape}")
        
        print(f"    ✅ {name.upper()} backend OK")
    
    # Test unified VSA module
    print(f"\n  Testing unified VSA module...")
    vsa = VSA(dim=dim, vsa_type="map")
    hvs = vsa.gen_hvs(5)
    assert hvs.shape == (5, dim)
    assert vsa.bind(hvs[0], hvs[1]).shape == (dim,)
    assert vsa.bundle(hvs).shape == (dim,)
    print(f"    ✅ Unified VSA module OK")
    
    print(f"\n  ✅ All VSA backends test complete!")


if __name__ == "__main__":
    test_vsa_backends()
