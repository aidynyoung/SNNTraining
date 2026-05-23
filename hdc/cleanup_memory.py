"""
hdc/cleanup_memory.py
======================
Cleanup Memory / Item Memory for Hyperdimensional Computing.

Implements the token selection mechanism from:
    Teeters et al. (2023) "On separating long- and short-term memories
    in hyperdimensional computing" Frontiers in Neuroscience, 16.
    doi:10.3389/fnins.2022.867568

Key insight: In LLMs, tokens are selected via softmax over a vocabulary.
In HDC, this is handled by Cleanup Memory or Item Memory (IM). When a noisy
or combined hypervector is generated, the system performs a "release" operation
to project it back onto the nearest known basis hypervector in the codebook.

This module provides:
- ItemMemory: Stores basis hypervectors (the "vocabulary" of HDC)
- CleanupMemory: Projects noisy/composite HVs back to nearest basis vectors
- Release operation: Decomposes a bound/bundled HV into its nearest components
- Codebook: The set of all known basis hypervectors

Usage:
    from hdc.cleanup_memory import CleanupMemory, ItemMemory

    im = ItemMemory(dim=10000)
    im.add("cat", cat_hv)
    im.add("dog", dog_hv)

    cm = CleanupMemory(im)
    result = cm.cleanup(noisy_hv)       # Project to nearest basis
    tokens = cm.release(composite_hv)   # Decompose into components
"""

import torch
import torch.nn.functional as F
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Set

logger = logging.getLogger(__name__)


@dataclass
class CleanupConfig:
    """Configuration for cleanup memory."""
    dim: int = 10000              # Hypervector dimension
    similarity_threshold: float = 0.55  # Minimum similarity for match
    max_candidates: int = 10      # Max candidates for release operation
    use_hamming: bool = True      # Use Hamming vs cosine similarity
    device: str = "cpu"


class ItemMemory:
    """
    Item Memory (IM) — the "vocabulary" of HDC.

    Stores basis hypervectors that represent atomic symbols/concepts.
    This is HDC's equivalent of a token embedding table.

    Attributes:
        dim: Hypervector dimension
        items: Dict mapping label → hypervector
    """

    def __init__(self, dim: int = 10000):
        self.dim = dim
        self.items: Dict[str, torch.Tensor] = {}
        self._item_vectors: Optional[torch.Tensor] = None
        self._item_labels: List[str] = []

    def add(self, label: str, hv: torch.Tensor) -> None:
        """Add a basis hypervector to the item memory."""
        assert hv.shape == (self.dim,), f"Expected ({self.dim},), got {hv.shape}"
        self.items[label] = (hv > 0).float()
        self._rebuild_index()

    def add_batch(self, labels: List[str], hvs: torch.Tensor) -> None:
        """Add multiple basis hypervectors at once."""
        assert hvs.shape[1] == self.dim, f"Expected dim={self.dim}, got {hvs.shape[1]}"
        for label, hv in zip(labels, hvs):
            self.items[label] = (hv > 0).float()
        self._rebuild_index()

    def get(self, label: str) -> Optional[torch.Tensor]:
        """Retrieve a basis hypervector by label."""
        return self.items.get(label)

    def remove(self, label: str) -> None:
        """Remove a basis hypervector."""
        self.items.pop(label, None)
        self._rebuild_index()

    def _rebuild_index(self):
        """Rebuild the tensor index for fast similarity search."""
        if not self.items:
            self._item_vectors = None
            self._item_labels = []
            return
        self._item_labels = list(self.items.keys())
        self._item_vectors = torch.stack(
            [self.items[lbl] for lbl in self._item_labels]
        )

    def similarity(
        self, hv: torch.Tensor, top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Compute similarity of hv against all stored items.

        Args:
            hv: (dim,) query hypervector
            top_k: Number of top matches

        Returns:
            List of (label, similarity) tuples
        """
        if self._item_vectors is None:
            return []
        hv_bin = (hv > 0).float()
        n = self._item_vectors.shape[0]
        hamming = (self._item_vectors != hv_bin.unsqueeze(0)).sum(dim=1).float()
        similarities = 1.0 - (hamming / self.dim)
        top_vals, top_idxs = similarities.topk(min(top_k, n))
        return [(self._item_labels[int(idx)], float(top_vals[i]))
                for i, idx in enumerate(top_idxs)]

    def nearest(self, hv: torch.Tensor) -> Optional[Tuple[str, float]]:
        """Find the single nearest item."""
        results = self.similarity(hv, top_k=1)
        return results[0] if results else None

    def __len__(self) -> int:
        return len(self.items)

    def __contains__(self, label: str) -> bool:
        return label in self.items

    def __repr__(self) -> str:
        return f"ItemMemory(dim={self.dim}, items={len(self.items)})"


class CleanupMemory:
    """
    Cleanup Memory — projects noisy/composite hypervectors back to basis.

    This is HDC's equivalent of the softmax token selection in LLMs.
    When a noisy or combined hypervector is generated, the cleanup memory
    performs a "release" operation to project it back onto the nearest
    known basis hypervector in the codebook.

    Supports:
    - cleanup: Project a noisy HV to the nearest basis HV
    - release: Decompose a bound/bundled HV into its components
    - iterative_cleanup: Repeated cleanup for multi-step reasoning
    """

    def __init__(
        self,
        item_memory: ItemMemory,
        config: Optional[CleanupConfig] = None,
    ):
        self.im = item_memory
        self.config = config or CleanupConfig(dim=item_memory.dim)

    def cleanup(self, hv: torch.Tensor) -> Optional[Tuple[str, torch.Tensor, float]]:
        """
        Project a noisy hypervector to the nearest basis hypervector.

        This is the core cleanup operation: given a noisy or degraded HV,
        find the closest basis HV in the item memory and return it.

        Args:
            hv: (dim,) noisy hypervector

        Returns:
            (label, clean_hv, similarity) or None if below threshold
        """
        result = self.im.nearest(hv)
        if result is None:
            return None
        label, sim = result
        if sim < self.config.similarity_threshold:
            return None
        clean_hv = self.im.get(label)
        return (label, clean_hv, sim)

    def release(
        self, hv: torch.Tensor, max_components: int = 3
    ) -> List[Tuple[str, float]]:
        """
        Release operation: decompose a bound/bundled HV into components.

        Iteratively finds the nearest basis HV, subtracts it, and repeats.
        This is how HDC "unpacks" a composite representation.

        Args:
            hv: (dim,) composite hypervector (bound or bundled)
            max_components: Maximum number of components to extract

        Returns:
            List of (label, confidence) tuples
        """
        residual = hv.clone()
        components = []

        for _ in range(max_components):
            result = self.im.nearest(residual)
            if result is None:
                break
            label, sim = result
            if sim < self.config.similarity_threshold:
                break

            components.append((label, sim))

            # Subtract the found component from residual
            # For binary HVs: residual XOR found = remove its influence
            found_hv = self.im.get(label)
            if found_hv is not None:
                residual = (residual > 0) != (found_hv > 0)
                residual = residual.float()

        return components

    def iterative_cleanup(
        self, hv: torch.Tensor, max_iterations: int = 5
    ) -> torch.Tensor:
        """
        Iterative cleanup: repeatedly clean up a hypervector.

        Useful for multi-step reasoning where each cleanup step
        refines the representation.

        Args:
            hv: (dim,) hypervector to clean
            max_iterations: Maximum cleanup iterations

        Returns:
            (dim,) cleaned hypervector
        """
        current = hv.clone()
        for _ in range(max_iterations):
            result = self.cleanup(current)
            if result is None:
                break
            _, clean_hv, sim = result
            if sim > 0.95:  # Already very close
                break
            # Blend: move toward the clean version
            current = (current + clean_hv) / 2.0
            current = (current > 0.5).float()
        return current

    def batch_cleanup(
        self, hvs: torch.Tensor
    ) -> List[Optional[Tuple[str, torch.Tensor, float]]]:
        """Clean up multiple hypervectors in batch."""
        return [self.cleanup(hv) for hv in hvs]

    def soft_cleanup(
        self,
        hv:          torch.Tensor,
        temperature: float = 5.0,
        top_k:       int   = 5,
    ) -> List[Tuple[str, float]]:
        """
        Soft (probabilistic) cleanup: return a distribution over items.

        Instead of hard nearest-neighbour, returns a Boltzmann distribution
        over the top-k items. Useful when the input is genuinely ambiguous
        between multiple concepts.

        P(item_i | hv) ∝ exp(β × sim(hv, item_i))

        Args:
            hv:          (dim,) query hypervector
            temperature: Sharpness of Boltzmann distribution (higher = sharper)
            top_k:       Number of candidates to return

        Returns:
            List of (label, probability) sorted descending. Probabilities sum ≤ 1.
        """
        if not self.im._labels:
            return []

        # Compute similarities to all items
        labels    = list(self.im._labels.keys())
        hvs_stack = torch.stack([self.im._hvs[l] for l in labels])
        sims      = self.im.similarity(hv, hvs_stack)   # (N,)

        # Top-k selection
        k        = min(top_k, len(labels))
        topk_sim, topk_idx = sims.topk(k)

        # Boltzmann normalisation
        log_probs = temperature * topk_sim
        probs     = F.softmax(log_probs, dim=0)

        return [(labels[int(idx)], float(p)) for idx, p in zip(topk_idx, probs)]

    def __repr__(self) -> str:
        return (
            f"CleanupMemory(items={len(self.im)}, "
            f"threshold={self.config.similarity_threshold})"
        )


# ── Test ──────────────────────────────────────────────────────────────────────

def test_cleanup_memory():
    """Verify cleanup memory operations."""
    torch.manual_seed(42)
    dim = 100

    # Create item memory with basis vectors
    im = ItemMemory(dim=dim)
    labels = ["cat", "dog", "bird", "fish", "tree"]
    hvs = torch.randint(0, 2, (len(labels), dim)).float()
    for label, hv in zip(labels, hvs):
        im.add(label, hv)

    cm = CleanupMemory(im)

    # Test cleanup: noisy version of "cat"
    noisy_cat = hvs[0].clone()
    noise_mask = torch.rand(dim) < 0.2  # 20% bit flips
    noisy_cat[noise_mask] = 1.0 - noisy_cat[noise_mask]
    result = cm.cleanup(noisy_cat)
    assert result is not None, "Cleanup should find match"
    label, clean_hv, sim = result
    assert label == "cat", f"Expected 'cat', got '{label}'"
    assert sim > 0.7, f"Similarity too low: {sim}"

    # Test release: decompose bound representation
    bound_hv = (hvs[0] > 0) != (hvs[1] > 0)  # cat XOR dog
    bound_hv = bound_hv.float()
    components = cm.release(bound_hv, max_components=2)
    assert len(components) >= 1, "Release should find components"

    # Test iterative cleanup
    cleaned = cm.iterative_cleanup(noisy_cat)
    assert cleaned.shape == (dim,), f"Shape: {cleaned.shape}"

    # Test similarity search
    sims = im.similarity(hvs[0], top_k=3)
    assert len(sims) == 3, f"Top-k: {len(sims)}"
    assert sims[0][0] == "cat", f"Best match should be 'cat', got {sims[0]}"

    print(f"  ItemMemory: {im}")
    print(f"  Cleanup: {label} (sim={sim:.3f})")
    print(f"  Release components: {components}")
    print(f"  Similarity search: {sims}")
    print("  ✓ All cleanup memory tests pass")


if __name__ == "__main__":
    test_cleanup_memory()
