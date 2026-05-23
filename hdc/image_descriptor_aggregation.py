"""
Hyperdimensional Computing as a Framework for Systematic Aggregation of Image Descriptors
==========================================================================================
Based on: Neubert, P., & Schubert, S. (2021)
"Hyperdimensional computing as a framework for systematic aggregation of image descriptors"
Pattern Recognition Letters, 147, 80-87. DOI: 10.1016/j.patrec.2021.04.003

Key contributions:

1. **Systematic Aggregation Framework** — HDC provides a principled way to
   aggregate multiple image descriptors (SIFT, SURF, ORB, etc.) into a single
   representation without retraining.

2. **Descriptor Fusion** — Different feature types (color, texture, shape, keypoints)
   are encoded as hypervectors and bundled/bound to form a unified representation.

3. **Hierarchical Aggregation** — Local descriptors → patch HVs → image HV,
   preserving spatial information through permutation.

4. **Place Recognition** — The aggregated HVs are used for visual place recognition,
   achieving competitive results with learned methods while being fully unsupervised.

Reference:
  Neubert, P., & Schubert, S. (2021)
  "Hyperdimensional computing as a framework for systematic aggregation
   of image descriptors"
  Pattern Recognition Letters, 147, 80-87
"""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Dict, Any, Callable
from hdc.hdc_glue import (
    hv_xor, hv_popcount, hv_hamming_sim, hv_bundle,
    hv_permute, hv_majority, hv_batch_sim, gen_hvs,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Section II: Descriptor Encoding
# ═══════════════════════════════════════════════════════════════════════════════

class DescriptorEncoder:
    """
    Encodes individual image descriptors into hypervectors.

    Supports multiple descriptor types:
    - SIFT: 128-dim float vectors → hypervectors
    - Color histograms: n-bin histograms → hypervectors
    - Texture features: Gabor/LBP responses → hypervectors
    - Keypoint locations: (x, y, scale, orientation) → hypervectors
    - Deep features: CNN layer activations → hypervectors

    Each descriptor type uses a different random projection, ensuring
    that different feature types are quasi-orthogonal in HD space.
    """

    def __init__(
        self,
        dim: int = 10000,
        descriptor_types: Optional[List[str]] = None,
        seed: Optional[int] = None,
    ):
        """
        Args:
            dim: Hypervector dimensionality
            descriptor_types: List of descriptor type names
            seed: Random seed
        """
        self.dim = dim
        self.seed = seed or 42

        if descriptor_types is None:
            descriptor_types = ["sift", "color", "texture", "keypoint", "deep"]

        self.descriptor_types = descriptor_types

        # Generate type-specific projection matrices
        self.projections: Dict[str, torch.Tensor] = {}
        for i, dtype in enumerate(descriptor_types):
            self.projections[dtype] = gen_hvs(
                dim, dim, seed=self.seed + i * 100
            )

    def encode_descriptor(
        self,
        descriptor: torch.Tensor,
        descriptor_type: str = "sift",
    ) -> torch.Tensor:
        """Encode a single descriptor vector into a hypervector.

        Uses random projection: hv = sign(projection @ descriptor)

        Args:
            descriptor: (n_features,) descriptor vector
            descriptor_type: Type of descriptor

        Returns:
            (dim,) binary hypervector
        """
        if descriptor_type not in self.projections:
            raise ValueError(f"Unknown descriptor type: {descriptor_type}")

        # Random projection
        proj = self.projections[descriptor_type]
        n_features = descriptor.shape[0]

        # Use first n_features dimensions of projection
        hv_raw = proj[:, :n_features] @ descriptor
        return (hv_raw > 0).float()

    def encode_descriptor_set(
        self,
        descriptors: torch.Tensor,
        descriptor_type: str = "sift",
        aggregation: str = "bundle",
    ) -> torch.Tensor:
        """Encode a set of descriptors into a single hypervector.

        Args:
            descriptors: (n_descriptors, n_features) descriptor vectors
            descriptor_type: Type of descriptor
            aggregation: "bundle" or "sum"

        Returns:
            (dim,) aggregated hypervector
        """
        if descriptors.shape[0] == 0:
            return torch.zeros(self.dim)

        hvs = []
        for i in range(descriptors.shape[0]):
            hv = self.encode_descriptor(descriptors[i], descriptor_type)
            hvs.append(hv)

        if aggregation == "bundle":
            aggregated = hv_bundle(torch.stack(hvs))
            return hv_majority(aggregated)
        else:
            return torch.stack(hvs).sum(dim=0)


# ═══════════════════════════════════════════════════════════════════════════════
# Section III: Hierarchical Aggregation
# ═══════════════════════════════════════════════════════════════════════════════

class HierarchicalImageEncoder:
    """
    Hierarchical aggregation of image descriptors (Neubert & Schubert 2021).

    Encodes images at multiple levels:
    1. **Local level**: Individual descriptors → local HVs
    2. **Patch level**: Spatially-localized groups → patch HVs
    3. **Image level**: All patches → image HV

    Spatial information is preserved by permuting patch HVs based on
    their position in the image grid.

    This is the VSA analog of a convolutional neural network's
    hierarchical feature extraction.
    """

    def __init__(
        self,
        dim: int = 10000,
        n_patches_x: int = 4,
        n_patches_y: int = 4,
        seed: Optional[int] = None,
    ):
        """
        Args:
            dim: Hypervector dimensionality
            n_patches_x: Number of horizontal patches
            n_patches_y: Number of vertical patches
            seed: Random seed
        """
        self.dim = dim
        self.n_patches_x = n_patches_x
        self.n_patches_y = n_patches_y
        self.seed = seed or 42

        # Position hypervectors for each patch
        self.position_hvs: Dict[Tuple[int, int], torch.Tensor] = {}
        for px in range(n_patches_x):
            for py in range(n_patches_y):
                pos_seed = self.seed + px * n_patches_y + py
                self.position_hvs[(px, py)] = gen_hvs(
                    1, dim, seed=pos_seed
                ).squeeze(0)

        # Descriptor encoder
        self.descriptor_encoder = DescriptorEncoder(dim=dim, seed=seed)

    def encode_patch(
        self,
        descriptors: Dict[str, torch.Tensor],
        patch_x: int,
        patch_y: int,
    ) -> torch.Tensor:
        """Encode a single image patch.

        Args:
            descriptors: {descriptor_type: (n, n_features) descriptors}
            patch_x: Patch X position
            patch_y: Patch Y position

        Returns:
            (dim,) patch hypervector
        """
        # Encode each descriptor type
        type_hvs = []
        for dtype, descs in descriptors.items():
            if descs.shape[0] > 0:
                hv = self.descriptor_encoder.encode_descriptor_set(descs, dtype)
                type_hvs.append(hv)

        if not type_hvs:
            return torch.zeros(self.dim)

        # Bundle descriptor types
        patch_hv = hv_bundle(torch.stack(type_hvs))
        patch_hv = hv_majority(patch_hv)

        # Bind with position
        pos_hv = self.position_hvs.get((patch_x, patch_y))
        if pos_hv is not None:
            patch_hv = hv_xor(patch_hv, pos_hv)

        return patch_hv

    def encode_image(
        self,
        patches: Dict[Tuple[int, int], Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """Encode a full image from its patches.

        Args:
            patches: {(px, py): {descriptor_type: descriptors}}

        Returns:
            (dim,) image hypervector
        """
        patch_hvs = []
        for (px, py), descriptors in patches.items():
            patch_hv = self.encode_patch(descriptors, px, py)
            patch_hvs.append(patch_hv)

        if not patch_hvs:
            return torch.zeros(self.dim)

        # Bundle all patches
        image_hv = hv_bundle(torch.stack(patch_hvs))
        return hv_majority(image_hv)

    def encode_image_from_features(
        self,
        features: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode an image from a flat feature map.

        Args:
            features: (n_features,) or (h, w, c) feature tensor
            positions: Optional (n_features, 2) position coordinates

        Returns:
            (dim,) image hypervector
        """
        if features.dim() == 3:
            # (h, w, c) → flatten with position info
            h, w, c = features.shape
            flat_features = features.reshape(-1, c)
            if positions is None:
                # Create grid positions
                ys, xs = torch.meshgrid(
                    torch.arange(h), torch.arange(w), indexing="ij"
                )
                positions = torch.stack([xs.flatten(), ys.flatten()], dim=-1)
        else:
            flat_features = features.unsqueeze(0) if features.dim() == 1 else features
            if positions is None:
                positions = torch.zeros(flat_features.shape[0], 2)

        # Encode each feature with its position
        feature_hvs = []
        for i in range(flat_features.shape[0]):
            hv = self.descriptor_encoder.encode_descriptor(
                flat_features[i], "deep"
            )
            # Bind with position
            px = int(positions[i, 0].item())
            py = int(positions[i, 1].item())
            pos_key = (px % self.n_patches_x, py % self.n_patches_y)
            if pos_key in self.position_hvs:
                hv = hv_xor(hv, self.position_hvs[pos_key])
            feature_hvs.append(hv)

        if not feature_hvs:
            return torch.zeros(self.dim)

        image_hv = hv_bundle(torch.stack(feature_hvs))
        return hv_majority(image_hv)


# ═══════════════════════════════════════════════════════════════════════════════
# Section IV: Multi-Descriptor Fusion
# ═══════════════════════════════════════════════════════════════════════════════

class MultiDescriptorFusion:
    """
    Fuses multiple descriptor types into a unified representation.

    Key insight (Neubert & Schubert 2021, Section 3):
    Different descriptor types capture complementary information.
    HDC allows principled fusion through bundling and binding.

    Fusion strategies:
    1. **Bundle fusion**: hv = bundle(hv_sift, hv_color, hv_texture)
    2. **Weighted fusion**: hv = bundle(w1*hv_sift, w2*hv_color, ...)
    3. **Hierarchical fusion**: First fuse within type, then across types
    4. **Attention fusion**: Weight descriptors by relevance
    """

    def __init__(
        self,
        dim: int = 10000,
        descriptor_types: Optional[List[str]] = None,
        fusion_strategy: str = "bundle",
        seed: Optional[int] = None,
    ):
        """
        Args:
            dim: Hypervector dimensionality
            descriptor_types: List of descriptor type names
            fusion_strategy: "bundle", "weighted", "hierarchical", or "attention"
            seed: Random seed
        """
        self.dim = dim
        self.seed = seed or 42

        if descriptor_types is None:
            descriptor_types = ["sift", "color", "texture", "keypoint", "deep"]

        self.descriptor_types = descriptor_types
        self.fusion_strategy = fusion_strategy

        # Type-specific hypervectors for binding
        self.type_hvs: Dict[str, torch.Tensor] = {}
        for i, dtype in enumerate(descriptor_types):
            self.type_hvs[dtype] = gen_hvs(1, dim, seed=self.seed + i).squeeze(0)

        # Weights for weighted fusion
        self.weights: Dict[str, float] = {d: 1.0 for d in descriptor_types}

    def set_weights(self, weights: Dict[str, float]):
        """Set fusion weights for each descriptor type.

        Args:
            weights: {descriptor_type: weight}
        """
        for dtype, w in weights.items():
            if dtype in self.weights:
                self.weights[dtype] = w

    def fuse(
        self,
        descriptor_hvs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Fuse multiple descriptor hypervectors.

        Args:
            descriptor_hvs: {descriptor_type: (dim,) hypervector}

        Returns:
            (dim,) fused hypervector
        """
        if not descriptor_hvs:
            return torch.zeros(self.dim)

        if self.fusion_strategy == "bundle":
            return self._bundle_fusion(descriptor_hvs)
        elif self.fusion_strategy == "weighted":
            return self._weighted_fusion(descriptor_hvs)
        elif self.fusion_strategy == "hierarchical":
            return self._hierarchical_fusion(descriptor_hvs)
        elif self.fusion_strategy == "attention":
            return self._attention_fusion(descriptor_hvs)
        else:
            return self._bundle_fusion(descriptor_hvs)

    def _bundle_fusion(self, descriptor_hvs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Simple bundling of all descriptor types."""
        hvs = [hv for hv in descriptor_hvs.values()]
        fused = hv_bundle(torch.stack(hvs))
        return hv_majority(fused)

    def _weighted_fusion(self, descriptor_hvs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Weighted bundling of descriptor types."""
        hvs = []
        for dtype, hv in descriptor_hvs.items():
            w = self.weights.get(dtype, 1.0)
            hvs.append(hv * w)

        fused = hv_bundle(torch.stack(hvs))
        return hv_majority(fused)

    def _hierarchical_fusion(self, descriptor_hvs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Hierarchical fusion: bind each type with type HV, then bundle."""
        bound_hvs = []
        for dtype, hv in descriptor_hvs.items():
            type_hv = self.type_hvs.get(dtype)
            if type_hv is not None:
                bound_hvs.append(hv_xor(hv, type_hv))
            else:
                bound_hvs.append(hv)

        fused = hv_bundle(torch.stack(bound_hvs))
        return hv_majority(fused)

    def _attention_fusion(self, descriptor_hvs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Attention-based fusion: weight by similarity to query."""
        # Use the first descriptor as query (or average)
        hvs_list = list(descriptor_hvs.values())
        if not hvs_list:
            return torch.zeros(self.dim)

        query = hv_bundle(torch.stack(hvs_list))
        query = hv_majority(query)

        # Compute attention weights
        attention_hvs = []
        attention_weights = []

        for dtype, hv in descriptor_hvs.items():
            sim = float(hv_hamming_sim(query, hv))
            attention_hvs.append(hv)
            attention_weights.append(sim)

        # Softmax weights
        weights = torch.softmax(torch.tensor(attention_weights), dim=0)

        # Weighted fusion
        weighted_hvs = [hv * w.item() for hv, w in zip(attention_hvs, weights)]
        fused = hv_bundle(torch.stack(weighted_hvs))
        return hv_majority(fused)


# ═══════════════════════════════════════════════════════════════════════════════
# Section V: Visual Place Recognition
# ═══════════════════════════════════════════════════════════════════════════════

class VisualPlaceRecognizer:
    """
    Visual place recognition using aggregated HDC descriptors.

    Based on Neubert & Schubert 2021, this system:
    1. Encodes images into hypervectors using hierarchical aggregation
    2. Stores place HVs in an associative memory
    3. Recognizes places by nearest-neighbor search in HD space
    4. Supports incremental learning (add new places without retraining)

    Key advantage: fully unsupervised, works with any descriptor type,
    and supports online learning.
    """

    def __init__(
        self,
        dim: int = 10000,
        n_patches_x: int = 4,
        n_patches_y: int = 4,
        fusion_strategy: str = "bundle",
        seed: Optional[int] = None,
    ):
        """
        Args:
            dim: Hypervector dimensionality
            n_patches_x: Number of horizontal patches
            n_patches_y: Number of vertical patches
            fusion_strategy: Descriptor fusion strategy
            seed: Random seed
        """
        self.dim = dim
        self.seed = seed or 42

        self.image_encoder = HierarchicalImageEncoder(
            dim=dim,
            n_patches_x=n_patches_x,
            n_patches_y=n_patches_y,
            seed=seed,
        )

        self.fusion = MultiDescriptorFusion(
            dim=dim,
            fusion_strategy=fusion_strategy,
            seed=seed,
        )

        # Place memory
        self.place_hvs: Dict[str, torch.Tensor] = {}
        self.place_counts: Dict[str, int] = {}

    def add_place(
        self,
        place_id: str,
        image_hv: torch.Tensor,
    ):
        """Add or update a place in memory.

        Args:
            place_id: Unique place identifier
            image_hv: (dim,) image hypervector
        """
        if place_id in self.place_hvs:
            # Update existing place (incremental learning)
            count = self.place_counts[place_id]
            self.place_hvs[place_id] = (
                (self.place_hvs[place_id] * count + image_hv) / (count + 1)
            )
            self.place_counts[place_id] += 1
        else:
            self.place_hvs[place_id] = image_hv.clone()
            self.place_counts[place_id] = 1

    def recognize(
        self,
        query_hv: torch.Tensor,
        top_k: int = 1,
    ) -> List[Tuple[str, float]]:
        """Recognize a place from a query hypervector.

        Args:
            query_hv: (dim,) query image hypervector
            top_k: Number of top matches

        Returns:
            List of (place_id, similarity) tuples
        """
        if not self.place_hvs:
            return []

        results = []
        for place_id, place_hv in self.place_hvs.items():
            sim = float(hv_hamming_sim(query_hv, place_hv))
            results.append((place_id, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def recognize_from_features(
        self,
        features: Dict[str, Any],
        top_k: int = 1,
    ) -> List[Tuple[str, float]]:
        """Recognize a place from raw features.

        Args:
            features: Dict with descriptor data
            top_k: Number of top matches

        Returns:
            List of (place_id, similarity) tuples
        """
        # Encode features into hypervector
        if "patches" in features:
            image_hv = self.image_encoder.encode_image(features["patches"])
        elif "features" in features:
            image_hv = self.image_encoder.encode_image_from_features(
                features["features"],
                features.get("positions"),
            )
        else:
            raise ValueError("Features must contain 'patches' or 'features'")

        return self.recognize(image_hv, top_k)


# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_descriptor_encoder():
    """Verify descriptor encoding."""
    print("=" * 60)
    print("Testing Descriptor Encoding (Neubert & Schubert 2021)")
    print("=" * 60)

    dim = 1000
    encoder = DescriptorEncoder(dim=dim)

    # Encode a SIFT-like descriptor
    sift = torch.randn(128)
    hv = encoder.encode_descriptor(sift, "sift")
    print(f"  SIFT descriptor → HV shape: {hv.shape}")
    print(f"  HV density: {float(hv.mean().item()):.4f}")

    # Encode a set of descriptors
    sifts = torch.randn(10, 128)
    hv_set = encoder.encode_descriptor_set(sifts, "sift")
    print(f"  SIFT set → HV shape: {hv_set.shape}")

    print(f"  ✅ Descriptor encoding test complete!")


def test_hierarchical_encoder():
    """Verify hierarchical image encoding."""
    print("=" * 60)
    print("Testing Hierarchical Image Encoding (Neubert & Schubert 2021)")
    print("=" * 60)

    dim = 1000
    encoder = HierarchicalImageEncoder(dim=dim, n_patches_x=2, n_patches_y=2)

    # Create synthetic patches
    patches = {}
    for px in range(2):
        for py in range(2):
            patches[(px, py)] = {
                "sift": torch.randn(5, 128),
                "color": torch.randn(3, 32),
            }

    image_hv = encoder.encode_image(patches)
    print(f"  Image HV shape: {image_hv.shape}")

    # Similar images should have higher similarity
    patches2 = {}
    for px in range(2):
        for py in range(2):
            patches2[(px, py)] = {
                "sift": torch.randn(5, 128),
                "color": torch.randn(3, 32),
            }
    image_hv2 = encoder.encode_image(patches2)

    sim = float(hv_hamming_sim(image_hv, image_hv2))
    print(f"  Similarity between different images: {sim:.4f}")

    print(f"  ✅ Hierarchical encoding test complete!")


def test_multi_descriptor_fusion():
    """Verify multi-descriptor fusion."""
    print("=" * 60)
    print("Testing Multi-Descriptor Fusion (Neubert & Schubert 2021)")
    print("=" * 60)

    dim = 1000
    fusion = MultiDescriptorFusion(dim=dim)

    # Create descriptor HVs
    descriptor_hvs = {
        "sift": gen_hvs(1, dim, seed=1).squeeze(0),
        "color": gen_hvs(1, dim, seed=2).squeeze(0),
        "texture": gen_hvs(1, dim, seed=3).squeeze(0),
    }

    for strategy in ["bundle", "weighted", "hierarchical", "attention"]:
        fusion.fusion_strategy = strategy
        fused = fusion.fuse(descriptor_hvs)
        print(f"  {strategy} fusion → HV shape: {fused.shape}")

    print(f"  ✅ Multi-descriptor fusion test complete!")


if __name__ == "__main__":
    test_descriptor_encoder()
    print()
    test_hierarchical_encoder()
    print()
    test_multi_descriptor_fusion()
