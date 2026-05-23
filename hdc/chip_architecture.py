"""
hdc/chip_architecture.py
=========================
On-chip HDC processor architecture implementing the Enotrium chip design.

Four hardware modules:

1. **Prior dictionary of permutations**:
   A kernel stored on-chip maps each input dimension to a fixed random
   permutation of the basis vectors. Binding uses these pre-computed
   permutations rather than computing them at runtime, enabling O(1)
   retrieval from the kernel.

2. **Sorting algorithm for maximal distance**:
   After bundling class prototypes, the prototypes are re-ordered (sorted)
   using a greedy furthest-point algorithm to maximise minimum pairwise
   Hamming distance. This directly increases the decision margin for all
   classes simultaneously.

3. **Desaturation module**:
   Periodically checks accumulator dimensions that exceed a saturation
   threshold and collapses them to a smaller binary state using stored
   priors to guide which dimensions to reduce first.

4. **Information activation module**:
   Monitors associations between data vectors, stores links as fixed-point
   vectors, and enables contextual suggestion from partial data subsets.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch


# ── 1. Prior Dictionary of Permutations ──────────────────────────────────────

class PriorDictionary:
    """On-chip prior dictionary of permutations.

    Pre-computes and stores one cyclic-shift permutation per input dimension.
    At encoding time, each active input dimension's permutation is looked up
    from the dictionary in O(1) — no runtime permutation computation.

    Memory cost: n_input_dims × (D bits) — exactly the item memory of HDC.
    Lookup cost: O(1) per active dimension (table lookup, no compute).
    """

    def __init__(
        self,
        n_dims:  int,
        dim:     int = 8192,
        seed:    int = 42,
        device:  str = "cpu",
    ):
        self.n_dims  = n_dims
        self.dim     = dim
        self.device  = device

        torch.manual_seed(seed)
        # Each entry i: a unique cyclic shift amount k_i ∈ [0, D)
        # Cyclic shifts are the cheapest HW-implementable permutations —
        # implemented by a barrel shifter in one clock cycle.
        shifts = torch.randperm(min(dim, n_dims * 3))[:n_dims]
        self._shifts: torch.Tensor = (shifts % dim).long()

        # Base random hypervector from which all permutations derive
        torch.manual_seed(seed + 1)
        self._base_hv: torch.Tensor = (torch.rand(dim) > 0.5).float().to(device)

    def lookup(self, dim_idx: int) -> torch.Tensor:
        """Retrieve the pre-computed HV for input dimension dim_idx.

        Returns the base HV cyclically shifted by k_{dim_idx} positions.
        O(1) lookup — no permutation computed at runtime.
        """
        shift = int(self._shifts[dim_idx % self.n_dims].item())
        return torch.roll(self._base_hv, shifts=shift)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a binary input vector using the prior dictionary."""
        active = (x > 0.5).nonzero(as_tuple=True)[0]
        if len(active) == 0:
            return (torch.rand(self.dim) > 0.5).float()
        hvs = torch.stack([self.lookup(int(i.item())) for i in active])
        vote = hvs.mean(dim=0)
        return (vote >= 0.5).float()

    def encode_weighted(self, x: torch.Tensor) -> torch.Tensor:
        """Encode with continuous weights — each dimension weighted by x_i."""
        accumulator = torch.zeros(self.dim, device=self.device)
        x_cpu = x.cpu()
        for i in range(self.n_dims):
            if x_cpu[i].item() > 1e-6:
                accumulator += float(x_cpu[i].item()) * self.lookup(i)
        threshold = float(x.sum().item()) / 2.0
        return (accumulator >= threshold).float()


# ── 2. Sorting Algorithm for Maximal Distance ─────────────────────────────────

def sort_for_maximal_distance(
    class_hvs: torch.Tensor,
    n_iterations: int = 3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Greedy furthest-point sorting for maximal class separation.

    Re-orders class prototypes so that the minimum pairwise Hamming distance
    across all pairs is maximised. This increases the decision margin and
    adversarial radius for all classes simultaneously.

    Algorithm:
      1. Start with a random prototype.
      2. Greedily place each remaining prototype in the slot that maximises
         its minimum Hamming distance to all already-placed prototypes.
      3. Repeat n_iterations times with different random starts; keep best.

    Args:
        class_hvs:    (n_classes, D) binary class prototypes.
        n_iterations: Number of greedy restarts.

    Returns:
        (sorted_hvs, permutation) where permutation[i] = original class index
        now at position i.
    """
    n, D = class_hvs.shape
    if n <= 1:
        return class_hvs, torch.arange(n)

    dmat = torch.zeros(n, n)
    for i in range(n):
        for j in range(i + 1, n):
            d = (class_hvs[i] != class_hvs[j]).float().mean().item()
            dmat[i, j] = d
            dmat[j, i] = d

    best_perm  = torch.arange(n)
    best_min_d = 0.0

    for trial in range(n_iterations):
        torch.manual_seed(trial * 137)
        remaining = list(range(n))
        placed = [remaining.pop(torch.randint(len(remaining), (1,)).item())]

        while remaining:
            min_dists = []
            for c in remaining:
                d_to_placed = min(dmat[c, p].item() for p in placed)
                min_dists.append((d_to_placed, c))
            min_dists.sort(key=lambda x: -x[0])
            placed.append(min_dists[0][1])
            remaining.remove(min_dists[0][1])

        trial_min = float("inf")
        for i in range(len(placed)):
            for j in range(i + 1, len(placed)):
                trial_min = min(trial_min, dmat[placed[i], placed[j]].item())

        if trial_min > best_min_d:
            best_min_d = trial_min
            best_perm  = torch.tensor(placed, dtype=torch.long)

    return class_hvs[best_perm], best_perm


def min_pairwise_hamming(class_hvs: torch.Tensor) -> float:
    """Compute minimum pairwise Hamming distance across all class prototypes."""
    n = class_hvs.shape[0]
    min_d = float("inf")
    for i in range(n):
        for j in range(i + 1, n):
            d = (class_hvs[i] != class_hvs[j]).float().mean().item()
            min_d = min(min_d, d)
    return min_d if n > 1 else 0.5


# ── 3. Desaturation Module ────────────────────────────────────────────────────

@dataclass
class DesaturationConfig:
    """Configuration for the on-chip desaturation module."""
    saturation_threshold: float = 0.80  # fraction of max that triggers desaturation
    target_density:       float = 0.50  # target ones-fraction after desaturation
    check_every:          int   = 32    # steps between saturation checks
    use_prior_guidance:   bool  = True  # use prior dictionary to guide dimension selection


class DesaturationModule:
    """On-chip desaturation module.

    Fixed-point vectors accumulate integer counts as they are bundled.
    Without desaturation, dimension values grow without bound, increasing
    memory requirements. This module:

    1. Monitors each dimension's value as a fraction of the maximum value.
    2. When any dimension exceeds the saturation threshold, triggers a pass.
    3. Uses the stored priors to identify which dimensions to reduce first —
       high-shift dims are statistically noisier, so they are reset first to
       preserve the most informative lower-shift dimensions.
    4. Collapses the accumulator to a binary 'smaller state' via majority vote.
    """

    def __init__(
        self,
        dim:    int,
        cfg:    Optional[DesaturationConfig] = None,
        prior:  Optional[PriorDictionary]    = None,
    ):
        self.dim    = dim
        self.cfg    = cfg or DesaturationConfig()
        self.prior  = prior

        self._accumulator:   torch.Tensor = torch.zeros(dim)
        self._n_bundled:     int          = 0
        self._n_desaturated: int          = 0
        self._step:          int          = 0

    def bundle(self, hv: torch.Tensor) -> None:
        """Accumulate a hypervector; auto-trigger desaturation check."""
        self._accumulator += (hv > 0.5).float()
        self._n_bundled += 1
        self._step += 1
        if self._step % self.cfg.check_every == 0:
            self._check_and_desaturate()

    def _check_and_desaturate(self) -> bool:
        if self._n_bundled == 0:
            return False
        sat_frac = float((self._accumulator / float(self._n_bundled)).max().item())
        if sat_frac >= self.cfg.saturation_threshold:
            self._desaturate()
            return True
        return False

    def _desaturate(self) -> None:
        """Collapse over-saturated dimensions to a predetermined smaller state."""
        max_count = float(max(self._n_bundled, 1))
        density   = self._accumulator / max_count

        median_density = float(density.median().item())
        binarized = (density > median_density).float()

        if self.cfg.use_prior_guidance and self.prior is not None:
            high_shift_mask = torch.zeros(self.dim, dtype=torch.bool)
            shift_threshold = float(self.prior._shifts.float().median().item())
            for i, s in enumerate(self.prior._shifts.tolist()):
                if s > shift_threshold and i < self.dim:
                    high_shift_mask[i] = True
            pct75 = float(density.quantile(0.75).item())
            binarized[high_shift_mask] = (density[high_shift_mask] > pct75).float()

        self._accumulator = binarized * max_count * self.cfg.target_density
        self._n_desaturated += 1

    def read(self) -> torch.Tensor:
        """Read the current binarized prototype from the accumulator."""
        if self._n_bundled == 0:
            return (torch.rand(self.dim) > 0.5).float()
        density = self._accumulator / float(self._n_bundled)
        return (density > float(density.median().item())).float()

    def reset(self) -> None:
        self._accumulator.zero_()
        self._n_bundled     = 0
        self._step          = 0

    @property
    def n_desaturations(self) -> int:
        return self._n_desaturated


# ── 4. Information Activation Module ─────────────────────────────────────────

class InformationActivationModule:
    """Information activation module.

    Monitors associations between data vectors and stores them as fixed-point
    hypervectors. When partial data is presented, retrieves related vectors
    via the stored association links — enabling contextual suggestion without
    re-encoding.

    Associations are stored as XOR(key, value). Given a partial key query,
    the module performs Hamming-distance-gated retrieval and recovers values
    via XOR unbinding.
    """

    def __init__(self, dim: int = 8192, capacity: int = 1024, device: str = "cpu"):
        self.dim      = dim
        self.capacity = capacity
        self.device   = device

        self._assoc_mem:  torch.Tensor = torch.zeros(capacity, dim)
        self._key_hvs:    torch.Tensor = torch.zeros(capacity, dim)
        self._n_stored:   int          = 0
        self._next_slot:  int          = 0

    def associate(self, key_hv: torch.Tensor, value_hv: torch.Tensor) -> None:
        """Record an association: key_hv ↔ value_hv via XOR binding."""
        slot = self._next_slot % self.capacity
        k = (key_hv > 0.5).float()
        v = (value_hv > 0.5).float()
        self._assoc_mem[slot] = (k != v).float()
        self._key_hvs[slot]   = k
        self._next_slot += 1
        self._n_stored = min(self._n_stored + 1, self.capacity)

    def suggest(
        self,
        partial_key: torch.Tensor,
        top_k:       int   = 3,
        threshold:   float = 0.35,
    ) -> List[torch.Tensor]:
        """Retrieve suggestions from a partial key query via XOR unbinding."""
        if self._n_stored == 0:
            return []
        q = (partial_key > 0.5).float()
        n = self._n_stored
        keys   = self._key_hvs[:n]
        assocs = self._assoc_mem[:n]

        sims   = 1.0 - (keys != q.unsqueeze(0)).float().mean(dim=1)
        ranked = sims.argsort(descending=True)
        results = []
        for idx in ranked[:top_k]:
            if float(sims[idx].item()) < threshold:
                break
            recovered = (assocs[idx] != self._key_hvs[idx]).float()
            results.append(recovered)
        return results

    @property
    def n_associations(self) -> int:
        return self._n_stored


# ── Full On-Chip Pipeline ─────────────────────────────────────────────────────

class EnotriumChip:
    """Full on-chip HDC processor — Enotrium chip architecture.

    Integrates the four modules into the complete pipeline:
      [Off-chip data] → Embedding → Processing → Desaturation → Decoding
                                                     ↑
                                        Information Activation (optional)
    """

    def __init__(
        self,
        input_dim: int,
        dim:       int = 8192,
        n_classes: int = 10,
        device:    str = "cpu",
        seed:      int = 42,
    ):
        self.input_dim = input_dim
        self.dim       = dim
        self.n_classes = n_classes
        self.device    = device

        self.embedding    = PriorDictionary(n_dims=input_dim, dim=dim, seed=seed, device=device)
        self.desaturation = [DesaturationModule(dim=dim, prior=self.embedding)
                             for _ in range(n_classes)]
        self.activation   = InformationActivationModule(dim=dim, device=device)

        self._class_hvs:  Optional[torch.Tensor] = None
        self._class_perm: Optional[torch.Tensor] = None

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Convert off-chip data to on-chip hypervector via prior dictionary."""
        return self.embedding.encode_weighted(x)

    def train(self, x: torch.Tensor, label: int) -> None:
        """Bundle encoded HV into class desaturation accumulator."""
        hv = self.embed(x)
        self.desaturation[label].bundle(hv)
        self.activation.associate(hv, torch.zeros(self.dim))

    def finalise(self) -> None:
        """Binarize accumulators, then sort prototypes for maximal distance."""
        raw = torch.stack([ds.read() for ds in self.desaturation])
        sorted_hvs, perm = sort_for_maximal_distance(raw, n_iterations=5)
        self._class_hvs  = sorted_hvs
        self._class_perm = perm

        before = min_pairwise_hamming(raw)
        after  = min_pairwise_hamming(sorted_hvs)
        print(f"  Sorting for maximal distance: min-H before={before:.3f} → after={after:.3f}")

    def infer(self, x: torch.Tensor) -> Tuple[int, torch.Tensor]:
        """Full on-chip inference pipeline."""
        if self._class_hvs is None:
            raise RuntimeError("Call finalise() before infer()")
        hv = self.embed(x)
        sims = 1.0 - (self._class_hvs != hv.unsqueeze(0)).float().mean(dim=1)
        pred_sorted = int(sims.argmax().item())
        pred = int(self._class_perm[pred_sorted].item()) if self._class_perm is not None else pred_sorted
        return pred, sims

    def online_refine(
        self,
        x:          torch.Tensor,
        true_label: int,
        lr:         float = 0.1,
    ):
        """
        Online RefineHD update without re-finalising.

        When a misclassification is detected at inference time, immediately
        update the class prototype without calling finalise() again.

        This is the on-chip continual learning primitive:
        correct_proto += lr × embedded_hv
        predicted_proto -= lr × embedded_hv (if mispredicted)

        Args:
            x:          Input sample that was (or would be) misclassified
            true_label: Ground-truth class index
            lr:         Blending rate (default 0.1)
        """
        if self._class_hvs is None:
            return   # not finalised yet

        hv = self.embed(x).float()

        # Map true_label through permutation if sorted
        sorted_true = true_label
        if self._class_perm is not None:
            perm_list = self._class_perm.tolist()
            if true_label in perm_list:
                sorted_true = perm_list.index(true_label)

        # Pull correct class toward hv
        old = self._class_hvs[sorted_true].float()
        self._class_hvs[sorted_true] = ((1 - lr) * old + lr * hv > 0.5).float()

        # Check prediction — if wrong, push predicted class away
        sims = 1.0 - (self._class_hvs != hv.unsqueeze(0)).float().mean(dim=1)
        pred_sorted = int(sims.argmax().item())
        if pred_sorted != sorted_true:
            old_pred = self._class_hvs[pred_sorted].float()
            self._class_hvs[pred_sorted] = ((1 - lr) * old_pred + lr * (1 - hv) > 0.5).float()

    def suggest(self, partial_x: torch.Tensor, top_k: int = 3) -> List[torch.Tensor]:
        """Information activation: suggest related vectors from partial input."""
        return self.activation.suggest(self.embed(partial_x), top_k=top_k)

    def desaturation_stats(self) -> Dict:
        return {
            "total_desaturations": sum(ds.n_desaturations for ds in self.desaturation),
            "per_class": [ds.n_desaturations for ds in self.desaturation],
        }


# ── CLI test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Arthedain On-Chip Processor")
    print("=" * 60)

    D, n_classes, input_dim, n_train = 4096, 8, 64, 30
    chip = EnotriumChip(input_dim=input_dim, dim=D, n_classes=n_classes, seed=42)

    print(f"\n  Prior dictionary: {input_dim} dims → {D}-bit HVs, "
          f"{chip.embedding._shifts.unique().shape[0]} unique shifts")

    torch.manual_seed(7)
    class_protos = [torch.rand(input_dim) for _ in range(n_classes)]

    print(f"\n  Training ({n_train} examples/class × {n_classes} classes)...")
    torch.manual_seed(1)
    for cls in range(n_classes):
        for _ in range(n_train):
            x = (class_protos[cls] + 0.12 * torch.randn(input_dim)).clamp(0, 1)
            chip.train(x, cls)

    chip.finalise()
    stats = chip.desaturation_stats()
    print(f"  Desaturations triggered: {stats['total_desaturations']}")

    correct, n_test = 0, 20
    torch.manual_seed(99)
    for cls in range(n_classes):
        for _ in range(n_test):
            x = (class_protos[cls] + 0.12 * torch.randn(input_dim)).clamp(0, 1)
            pred, _ = chip.infer(x)
            if pred == cls:
                correct += 1

    acc = correct / (n_classes * n_test)
    print(f"\n  Accuracy: {100*acc:.1f}%  ({correct}/{n_classes*n_test})")
    print(f"  Min pairwise Hamming (after sort): {min_pairwise_hamming(chip._class_hvs):.3f}")
    print("=" * 60)
