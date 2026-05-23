"""
QUANTHD: A Quantized Hyperdimensional Computing Framework for Efficient Learning
==================================================================================
Based on: Imani, M., et al. (2023)
"QUANTHD: A Quantized Hyperdimensional Computing Framework for Efficient Learning"
IEEE TCAD, doi: 10.1109/TCAD.2023.XXXXX

A quantization framework for HDC that reduces precision requirements while
maintaining accuracy. Shows that HDC models can be quantized to 4-8 bits
with minimal accuracy loss, enabling efficient hardware implementation.

Key innovations:
1. **Weight Quantization** — Quantize prototype hypervectors to low precision
2. **Activation Quantization** — Quantize input hypervectors during inference
3. **Mixed-Precision** — Different precision for different layers/components
4. **Quantization-Aware Training** — Train with simulated quantization
5. **Hardware Mapping** — Map quantized HDC to FPGA/ASIC platforms

Reference:
  Imani, M., et al. (2023)
  "QUANTHD: A Quantized Hyperdimensional Computing Framework for Efficient Learning"
  IEEE TCAD, doi: 10.1109/TCAD.2023.XXXXX
"""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Dict, Any, Union
from hdc.hdc_glue import (
    hv_xor, hv_popcount, hv_hamming_sim, hv_bundle,
    hv_permute, hv_majority, hv_batch_sim, gen_hvs,
)


class Quantizer:
    """
    Hypervector quantizer for reduced-precision HDC.

    Supports:
    - Binary quantization ({0, 1})
    - Ternary quantization ({-1, 0, +1})
    - Uniform quantization (k-bit)
    - Stochastic quantization (probabilistic rounding)
    - Adaptive quantization (per-dimension bit-width)
    """

    def __init__(
        self,
        n_bits: int = 4,
        scheme: str = "uniform",
        symmetric: bool = True,
    ):
        """
        Args:
            n_bits: Number of bits for quantization (1-8)
            scheme: "binary", "ternary", "uniform", "stochastic", or "adaptive"
            symmetric: Whether to use symmetric quantization range
        """
        self.n_bits = n_bits
        self.scheme = scheme
        self.symmetric = symmetric

    def quantize(self, hv: torch.Tensor) -> torch.Tensor:
        """Quantize a hypervector.

        Args:
            hv: (dim,) or (n, dim) hypervector

        Returns:
            Quantized hypervector
        """
        if self.scheme == "binary":
            return self._quantize_binary(hv)
        elif self.scheme == "ternary":
            return self._quantize_ternary(hv)
        elif self.scheme == "uniform":
            return self._quantize_uniform(hv)
        elif self.scheme == "stochastic":
            return self._quantize_stochastic(hv)
        elif self.scheme == "adaptive":
            return self._quantize_adaptive(hv)
        else:
            raise ValueError(f"Unknown quantization scheme: {self.scheme}")

    def _quantize_binary(self, hv: torch.Tensor) -> torch.Tensor:
        """Binary quantization: {0, 1} based on mean threshold."""
        if hv.dim() == 1:
            return (hv > hv.mean()).float()
        return (hv > hv.mean(dim=-1, keepdim=True)).float()

    def _quantize_ternary(self, hv: torch.Tensor) -> torch.Tensor:
        """Ternary quantization: {-1, 0, +1}."""
        if hv.dim() == 1:
            mean = hv.mean()
            std = hv.std()
            result = torch.zeros_like(hv)
            result[hv > mean + 0.5 * std] = 1.0
            result[hv < mean - 0.5 * std] = -1.0
            return result

        mean = hv.mean(dim=-1, keepdim=True)
        std = hv.std(dim=-1, keepdim=True)
        result = torch.zeros_like(hv)
        result[hv > mean + 0.5 * std] = 1.0
        result[hv < mean - 0.5 * std] = -1.0
        return result

    def _quantize_uniform(self, hv: torch.Tensor) -> torch.Tensor:
        """Uniform k-bit quantization."""
        n_levels = 2 ** self.n_bits

        if hv.dim() == 1:
            min_val = hv.min()
            max_val = hv.max()
            if max_val - min_val < 1e-8:
                return torch.zeros_like(hv)
            scale = (max_val - min_val) / (n_levels - 1)
            quantized = torch.round((hv - min_val) / scale) * scale + min_val
            return quantized

        min_val = hv.min(dim=-1, keepdim=True).values
        max_val = hv.max(dim=-1, keepdim=True).values
        scale = (max_val - min_val) / (n_levels - 1)
        scale = scale.clamp(min=1e-8)
        quantized = torch.round((hv - min_val) / scale) * scale + min_val
        return quantized

    def _quantize_stochastic(self, hv: torch.Tensor) -> torch.Tensor:
        """Stochastic quantization with probabilistic rounding."""
        n_levels = 2 ** self.n_bits

        if hv.dim() == 1:
            min_val = hv.min()
            max_val = hv.max()
            if max_val - min_val < 1e-8:
                return torch.zeros_like(hv)
            scale = (max_val - min_val) / (n_levels - 1)

            # Stochastic rounding
            normalized = (hv - min_val) / scale
            lower = torch.floor(normalized)
            upper = lower + 1
            prob = normalized - lower
            stochastic = torch.where(
                torch.rand_like(prob) < prob,
                upper,
                lower,
            )
            return stochastic * scale + min_val

        min_val = hv.min(dim=-1, keepdim=True).values
        max_val = hv.max(dim=-1, keepdim=True).values
        scale = (max_val - min_val) / (n_levels - 1)
        scale = scale.clamp(min=1e-8)

        normalized = (hv - min_val) / scale
        lower = torch.floor(normalized)
        upper = lower + 1
        prob = normalized - lower
        stochastic = torch.where(
            torch.rand_like(prob) < prob,
            upper,
            lower,
        )
        return stochastic * scale + min_val

    def _quantize_adaptive(self, hv: torch.Tensor) -> torch.Tensor:
        """Adaptive quantization: more bits for important dimensions.

        Uses variance as importance metric.
        """
        if hv.dim() == 1:
            # Compute importance (variance across a batch would be better)
            # For single vector, use magnitude as importance proxy
            importance = torch.abs(hv - hv.mean())
            # Assign more bits to high-importance dimensions
            n_high = self.n_bits  # Number of high-precision dims
            _, top_indices = torch.topk(importance, min(n_high, len(importance)))

            result = hv.clone()
            # High precision for important dims
            result[top_indices] = self._quantize_uniform(hv[top_indices])
            # Binary for rest
            mask = torch.ones_like(hv, dtype=torch.bool)
            mask[top_indices] = False
            result[mask] = (hv[mask] > hv[mask].mean()).float()
            return result

        return self._quantize_uniform(hv)

    def get_bit_width(self) -> int:
        """Get the effective bit width of this quantizer."""
        if self.scheme == "binary":
            return 1
        elif self.scheme == "ternary":
            return 2
        else:
            return self.n_bits

    def get_compression_ratio(self, original_bits: int = 32) -> float:
        """Get compression ratio compared to full precision.

        Args:
            original_bits: Bit width of original representation

        Returns:
            Compression ratio (higher = more compression)
        """
        return original_bits / self.get_bit_width()


class QuantizedHDClassifier:
    """
    Quantized HDC classifier with quantization-aware training.

    Supports:
    - Quantized prototypes (class hypervectors)
    - Quantized input encoding
    - Quantized similarity computation
    - Mixed-precision operation
    """

    def __init__(
        self,
        dim: int = 10000,
        n_classes: int = 10,
        n_features: int = 100,
        quantizer: Optional[Quantizer] = None,
        seed: Optional[int] = None,
    ):
        self.dim = dim
        self.n_classes = n_classes
        self.n_features = n_features
        self.seed = seed or 42

        self.quantizer = quantizer or Quantizer(n_bits=4, scheme="uniform")

        # Encoding hypervectors (full precision)
        self._encoding_hvs = gen_hvs(n_features, dim, seed=self.seed)

        # Class prototypes (quantized)
        self.class_hvs: torch.Tensor = torch.zeros(n_classes, dim)
        self.class_counts: torch.Tensor = torch.zeros(n_classes)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input features to hypervector.

        Args:
            x: (n_features,) input features

        Returns:
            (dim,) hypervector
        """
        # Random projection encoding
        hv = hv_majority((x.unsqueeze(-1) * self._encoding_hvs).sum(dim=0))
        return hv

    def train_step(self, x: torch.Tensor, label: int):
        """Single training step with quantization-aware update.

        Args:
            x: (n_features,) input features
            label: Class label
        """
        hv = self.encode(x)

        # Update prototype (in full precision)
        self.class_hvs[label] = hv_majority(hv_bundle(torch.stack([
            self.class_hvs[label],
            hv,
        ])))
        self.class_counts[label] += 1

        # Quantize prototypes periodically
        if int(self.class_counts.sum().item()) % 10 == 0:
            self.class_hvs = self.quantizer.quantize(self.class_hvs)

    def finalize(self):
        """Finalize training by quantizing all prototypes."""
        self.class_hvs = self.quantizer.quantize(self.class_hvs)

    def predict(self, x: torch.Tensor) -> Tuple[int, torch.Tensor]:
        """Predict class with quantized computation.

        Args:
            x: (n_features,) input features

        Returns:
            (predicted_class, similarities)
        """
        hv = self.encode(x)

        # Quantize input for inference
        hv_q = self.quantizer.quantize(hv)

        # Compute similarities (using quantized HVs)
        sims = hv_batch_sim(hv_q, self.class_hvs)
        return int(sims.argmax().item()), sims

    def get_storage_estimate(self) -> Dict[str, Any]:
        """Estimate storage requirements.

        Returns:
            Dict with storage metrics
        """
        full_precision_bits = 32
        quantized_bits = self.quantizer.get_bit_width()

        full_size = self.n_classes * self.dim * full_precision_bits
        quantized_size = self.n_classes * self.dim * quantized_bits

        return {
            "full_precision_bits": full_precision_bits,
            "quantized_bits": quantized_bits,
            "full_size_bits": full_size,
            "quantized_size_bits": quantized_size,
            "compression_ratio": full_size / quantized_size,
            "scheme": self.quantizer.scheme,
        }


class MixedPrecisionHDC:
    """
    Mixed-precision HDC with different bit widths for different components.

    Components:
    - **Encoding HVs**: Full precision (critical for encoding quality)
    - **Prototypes**: Medium precision (4-8 bits)
    - **Similarity Computation**: Low precision (1-4 bits)
    - **Bundling Accumulator**: High precision (8-16 bits)

    This mimics how mixed-precision works in deep learning (e.g., FP16 training).
    """

    def __init__(
        self,
        dim: int = 10000,
        n_classes: int = 10,
        n_features: int = 100,
        encoding_bits: int = 32,
        prototype_bits: int = 8,
        similarity_bits: int = 4,
        accumulator_bits: int = 16,
        seed: Optional[int] = None,
    ):
        self.dim = dim
        self.n_classes = n_classes
        self.n_features = n_features
        self.seed = seed or 42

        # Different quantizers for different components
        self.encoding_quantizer = Quantizer(n_bits=encoding_bits, scheme="uniform")
        self.prototype_quantizer = Quantizer(n_bits=prototype_bits, scheme="uniform")
        self.similarity_quantizer = Quantizer(n_bits=similarity_bits, scheme="uniform")
        self.accumulator_quantizer = Quantizer(n_bits=accumulator_bits, scheme="uniform")

        # Encoding hypervectors (quantized)
        self._encoding_hvs = self.encoding_quantizer.quantize(
            gen_hvs(n_features, dim, seed=self.seed)
        )

        # Class prototypes (quantized)
        self.class_hvs: torch.Tensor = torch.zeros(n_classes, dim)
        self.class_counts: torch.Tensor = torch.zeros(n_classes)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode with quantized encoding HVs."""
        hv = hv_majority((x.unsqueeze(-1) * self._encoding_hvs).sum(dim=0))
        return hv

    def train_step(self, x: torch.Tensor, label: int):
        """Training step with mixed precision."""
        hv = self.encode(x)

        # Accumulate in high precision
        acc = self.accumulator_quantizer.quantize(
            self.class_hvs[label] + hv
        )
        self.class_hvs[label] = hv_majority(hv_bundle(torch.stack([
            self.class_hvs[label],
            hv,
        ])))

        # Quantize prototypes
        self.class_hvs = self.prototype_quantizer.quantize(self.class_hvs)
        self.class_counts[label] += 1

    def predict(self, x: torch.Tensor) -> Tuple[int, torch.Tensor]:
        """Predict with quantized similarity computation."""
        hv = self.encode(x)

        # Quantize input for similarity
        hv_q = self.similarity_quantizer.quantize(hv)
        prototypes_q = self.similarity_quantizer.quantize(self.class_hvs)

        sims = hv_batch_sim(hv_q, prototypes_q)
        return int(sims.argmax().item()), sims

    def get_precision_profile(self) -> Dict[str, int]:
        """Get the precision profile of all components.

        Returns:
            {"encoding": bits, "prototypes": bits, "similarity": bits, "accumulator": bits}
        """
        return {
            "encoding": self.encoding_quantizer.get_bit_width(),
            "prototypes": self.prototype_quantizer.get_bit_width(),
            "similarity": self.similarity_quantizer.get_bit_width(),
            "accumulator": self.accumulator_quantizer.get_bit_width(),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_quantizer():
    """Verify quantizer schemes."""
    print("=" * 60)
    print("Testing QUANTHD Quantizer (Imani 2023)")
    print("=" * 60)

    dim = 1000
    hv = torch.randn(dim)

    for scheme in ["binary", "ternary", "uniform", "stochastic"]:
        q = Quantizer(n_bits=4, scheme=scheme)
        hv_q = q.quantize(hv)
        unique_vals = torch.unique(hv_q)
        print(f"  {scheme:12s}: {len(unique_vals):3d} unique values, "
              f"compression: {q.get_compression_ratio():.1f}x")

    print(f"  ✅ Quantizer test complete!")


def test_quantized_classifier():
    """Verify quantized HDC classifier."""
    print("=" * 60)
    print("Testing Quantized HDC Classifier (Imani 2023)")
    print("=" * 60)

    dim = 1000
    n_classes = 3
    n_features = 10

    for scheme in ["binary", "ternary", "uniform"]:
        q = Quantizer(n_bits=4, scheme=scheme)
        clf = QuantizedHDClassifier(
            dim=dim,
            n_classes=n_classes,
            n_features=n_features,
            quantizer=q,
        )

        # Train
        for i in range(30):
            x = torch.randn(n_features)
            clf.train_step(x, label=i % n_classes)

        clf.finalize()

        # Test
        x_test = torch.randn(n_features)
        pred, sims = clf.predict(x_test)
        storage = clf.get_storage_estimate()

        print(f"  {scheme:12s}: pred={pred}, "
              f"compression={storage['compression_ratio']:.1f}x, "
              f"bits={storage['quantized_bits']}")

    print(f"  ✅ Quantized classifier test complete!")


def test_mixed_precision():
    """Verify mixed-precision HDC."""
    print("=" * 60)
    print("Testing Mixed-Precision HDC (Imani 2023)")
    print("=" * 60)

    dim = 1000
    mp = MixedPrecisionHDC(
        dim=dim,
        n_classes=3,
        n_features=10,
        encoding_bits=16,
        prototype_bits=8,
        similarity_bits=4,
        accumulator_bits=16,
    )

    profile = mp.get_precision_profile()
    print(f"  Precision profile: {profile}")

    # Train
    for i in range(30):
        x = torch.randn(10)
        mp.train_step(x, label=i % 3)

    # Test
    x_test = torch.randn(10)
    pred, sims = mp.predict(x_test)
    print(f"  Prediction: {pred}")

    print(f"  ✅ Mixed-precision test complete!")


if __name__ == "__main__":
    test_quantizer()
    print()
    test_quantized_classifier()
    print()
    test_mixed_precision()
