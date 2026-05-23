"""
Cellular Automata Rule 90 for HDC Memory Reduction
====================================================
Kleyko, Frady & Sommer (2020) "Cellular Automata Can Reduce Memory
Requirements of Collective-State Computing." arXiv:2010.03585

The Problem: Large HDC systems need to store N×D random binary basis vectors
(N neurons × D dimensions) — up to gigabytes for large N and D.

The Solution: Elementary Cellular Automaton Rule 90 (CA90) enables a
space-time tradeoff: store only N short seeds (N×N_seed bits) and expand
them on-the-fly to D-dimensional vectors by running CA90 for T steps.

CA90 rule (Fig. 1 of paper):
  next_state[i] = state[i-1] XOR state[i+1]   (wrapping at boundaries)
  = XOR of left and right neighbors

Key properties of CA90 (from §III of paper):
  1. For seed length N (grid cells), the randomization period is:
     - N = 2^j: period = 2^(j-2) - 1
     - N odd prime: period = (2^N - 1) (Mersenne prime; very long)
     - N even not 2^j: period = Π_N / 2
  2. Cyclic shift invariance: if seed b = Sh(a, i), then CA90(b, t) = Sh(CA90(a,t), i)
  3. Expanded representations from different seeds have ~0.5 Hamming distance
     (same as random i.i.d. binary vectors) after sufficient steps.

Storage tradeoff:
  Standard:  N × D bits  (all basis vectors stored)
  CA90:      N × N_seed bits + T computation steps  (N_seed << D)
  Savings:   factor of D/N_seed × (per-lookup compute overhead)

Practical implementation:
  - Each basis vector is identified by a short seed of N_seed bits
  - To get basis vector i: start CA90 from seed_i, run for T steps, return state
  - T = D - N_seed steps (expanding N_seed → D)
  - The final D-bit state IS the basis vector

Applications:
  - CA90ItemMemory: replace standard item memory with CA90-expanded vectors
  - CA90SparseMemory: combine with FlyHash for double savings
  - CA90Reservoir: use CA90 expansion for reservoir computing (RC)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from hdc.hdc_glue import hv_batch_sim, gen_hvs


# ═══════════════════════════════════════════════════════════════════════════════
# 1. CA90 Core Operations
# ═══════════════════════════════════════════════════════════════════════════════

def ca90_step(state: torch.Tensor) -> torch.Tensor:
    """
    One step of CA Rule 90 with periodic boundary conditions.

    Rule 90: next[i] = state[i-1] XOR state[i+1]
    Periodic: state[-1] = state[N-1], state[N] = state[0]

    Args:
        state: (N,) or (B, N) binary tensor (0/1)

    Returns:
        Next state, same shape as input
    """
    left  = torch.roll(state, shifts=1,  dims=-1)  # state[i-1]
    right = torch.roll(state, shifts=-1, dims=-1)   # state[i+1]
    return (left != right).float()                  # XOR


def ca90_run(seed: torch.Tensor, n_steps: int) -> torch.Tensor:
    """
    Run CA90 for n_steps from an initial seed.

    Args:
        seed: (N,) binary tensor (initial state)
        n_steps: Number of CA90 steps to apply

    Returns:
        (N,) binary tensor after n_steps
    """
    state = seed.float()
    for _ in range(n_steps):
        state = ca90_step(state)
    return state


def ca90_expand(seed: torch.Tensor, target_dim: int) -> torch.Tensor:
    """
    Expand a short seed to a full-dimensional HV using CA90.

    The expansion works by unrolling the CA90 dynamics:
    - Start from seed of length N_seed
    - Each step, the CA evolves to a new N_seed-dimensional state
    - Concatenate consecutive states to build the full D-dim HV

    Strategy: run CA90 ceil(D / N_seed) steps, concatenate intermediate states.

    Args:
        seed: (N_seed,) binary seed tensor
        target_dim: D — desired output dimensionality

    Returns:
        (target_dim,) binary HV expanded from seed
    """
    N = seed.shape[0]
    n_full_steps = math.ceil(target_dim / N)

    states = [seed.float()]
    state = seed.float()
    for _ in range(n_full_steps):
        state = ca90_step(state)
        states.append(state)

    # Concatenate all states (including initial) → trim to target_dim
    expanded = torch.cat(states, dim=0)[:target_dim]
    return expanded


def ca90_randomization_period(N: int) -> int:
    """
    Compute the randomization period of CA90 for a given grid size N.

    From Theorem in Kleyko 2020 §III:
      N = 2^j:            period = 2^(j-2) - 1
      N odd:              period ≈ 2^N - 1  (depends on primality)
      N even, not 2^j:    period = Π_N / 2  (Π_N is the periodic cycle)

    Returns an estimate (exact computation requires CA simulation for N even).
    """
    if N <= 1:
        return 1

    # Check if N is a power of 2
    if (N & (N - 1)) == 0:
        j = N.bit_length() - 1
        return max(1, 2 ** (j - 2) - 1)

    # N odd: long period (approximately 2^N - 1 for prime N)
    if N % 2 == 1:
        return 2 ** N - 1   # estimate (exact for prime N)

    # N even but not power of 2: use simulation to find period
    state = torch.zeros(N)
    state[0] = 1.0   # single active cell
    initial = state.clone()
    for t in range(1, 2 * N * N + 1):
        state = ca90_step(state)
        if torch.allclose(state, initial):
            return t
    return 2 * N * N   # fallback if not found


# ═══════════════════════════════════════════════════════════════════════════════
# 2. CA90ItemMemory — space-efficient basis vector storage
# ═══════════════════════════════════════════════════════════════════════════════

class CA90ItemMemory:
    """
    Item memory that stores seeds and expands to full HVs via CA90.

    Space-time tradeoff (Kleyko 2020, §IV-B):
      Standard:  N × D bits stored
      CA90:      N × N_seed bits stored + CA90 expansion at lookup time
      Savings:   factor of D / N_seed in storage (e.g., 100× for N_seed=100, D=10000)

    CA90 expansion is cheap (N_seed bit operations per step × T steps), making
    this practical for edge devices with tight memory but available compute.

    Recommended settings (from paper §III):
      N_seed = 37 (prime, period ≈ 2^37 ≈ 137B — far exceeds D=10000 needs)
      n_steps ≈ ceil(D / N_seed) for full expansion

    Args:
        item_dim: D — dimensionality of the expanded HVs
        seed_dim: N_seed — length of stored seeds (should be odd prime for long period)
        seed: Random seed for initialising item seeds
    """

    def __init__(
        self,
        item_dim: int = 10000,
        seed_dim: int = 37,    # prime → very long randomization period
        seed: Optional[int] = None,
    ):
        self.item_dim = item_dim
        self.seed_dim = seed_dim

        g = torch.Generator()
        if seed is not None:
            g.manual_seed(seed)
        self._g = g

        self._seeds: Dict[str, torch.Tensor] = {}
        self._n = 0

        # Theoretical randomization period
        self._period = ca90_randomization_period(seed_dim)

    def add(self, name: str, seed: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Register an item and return its seed (not the full HV).

        Args:
            name: Item name
            seed: Optional (seed_dim,) binary seed (generated if None)

        Returns:
            (seed_dim,) binary seed tensor
        """
        if seed is None:
            seed = (torch.rand(self.seed_dim, generator=self._g) < 0.5).float()
        self._seeds[name] = seed
        self._n += 1
        return seed

    def get(self, name: str) -> Optional[torch.Tensor]:
        """
        Retrieve the full-dimensional HV for an item by expanding its seed.

        This is the space-time tradeoff: compute the HV on demand rather
        than storing it.

        Returns:
            (item_dim,) binary HV, or None if item not found
        """
        if name not in self._seeds:
            return None
        return ca90_expand(self._seeds[name], self.item_dim)

    def get_batch(self, names: List[str]) -> torch.Tensor:
        """Return (len(names), item_dim) matrix of expanded HVs."""
        hvs = []
        for name in names:
            hv = self.get(name)
            if hv is not None:
                hvs.append(hv)
        return torch.stack(hvs) if hvs else torch.zeros(0, self.item_dim)

    def nearest(
        self,
        query_hv: torch.Tensor,
        top_k: int = 1,
    ) -> List[Tuple[str, float]]:
        """
        Find nearest stored items via CA90-expanded Hamming similarity.

        Expands each seed on demand — memory-efficient but compute-proportional.
        """
        names = list(self._seeds.keys())
        results = []
        for name in names:
            hv = self.get(name)
            sim = float(hv_batch_sim(query_hv, hv.unsqueeze(0))[0])
            results.append((name, sim))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    @property
    def memory_bytes(self) -> int:
        """Bytes used for seed storage (not full HVs)."""
        return self._n * math.ceil(self.seed_dim / 8)

    @property
    def standard_memory_bytes(self) -> int:
        """Bytes that standard item memory would use."""
        return self._n * math.ceil(self.item_dim / 8)

    @property
    def memory_reduction(self) -> float:
        """Factor of memory saved vs standard item memory."""
        if self.memory_bytes == 0:
            return 1.0
        return self.standard_memory_bytes / self.memory_bytes


# ═══════════════════════════════════════════════════════════════════════════════
# 3. CA90HDCClassifier — full classifier with CA90 basis vectors
# ═══════════════════════════════════════════════════════════════════════════════

class CA90HDCClassifier:
    """
    HDC classifier using CA90-expanded basis vectors.

    Standard HDC classifier stores n_features × hd_dim random basis HVs.
    CA90HDCClassifier stores only n_features × seed_dim seeds and expands
    them on demand, achieving a factor of hd_dim/seed_dim memory reduction.

    For n_features=100, hd_dim=10000, seed_dim=37:
      Standard: 100 × 10000 / 8 = 125 KB for basis vectors
      CA90:     100 × 37 / 8 ≈ 0.46 KB for seeds → 270× memory reduction

    Args:
        n_features: Number of input features
        n_classes: Number of output classes
        hd_dim: HV dimensionality
        seed_dim: CA90 seed length
        seed: Random seed
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        hd_dim: int = 10000,
        seed_dim: int = 37,
        seed: Optional[int] = None,
    ):
        self.n_features = n_features
        self.n_classes = n_classes
        self.hd_dim = hd_dim

        # Feature basis: CA90 item memory
        self.feature_mem = CA90ItemMemory(hd_dim, seed_dim, seed=seed)
        for i in range(n_features):
            self.feature_mem.add(f"f{i}")

        # Class prototype accumulators
        self._accums = torch.zeros(n_classes, hd_dim)
        self._counts = torch.zeros(n_classes)
        self._prototypes: Optional[torch.Tensor] = None

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input vector to HV via majority-bundle of feature HVs.

        Active features (x_i > 0.5) contribute their CA90-expanded HV.
        """
        active_mask = (x > 0.5)
        if not active_mask.any():
            return (torch.rand(self.hd_dim) < 0.5).float()

        components = []
        bipolar_hvs = []
        for i in range(self.n_features):
            if active_mask[i].item():
                hv = self.feature_mem.get(f"f{i}")
                if hv is not None:
                    bipolar_hvs.append(2.0 * hv - 1.0)

        if not bipolar_hvs:
            return (torch.rand(self.hd_dim) < 0.5).float()

        weighted_sum = torch.stack(bipolar_hvs).sum(dim=0)
        return (weighted_sum > 0).float()

    def train_step(self, x: torch.Tensor, label: int):
        """Accumulate encoded HV into class prototype."""
        hv = self._encode(x)
        self._accums[label] += hv.float()
        self._counts[label] += 1

    def finalize(self):
        """Binarise class prototypes."""
        counts = self._counts.clamp(min=1).unsqueeze(-1)
        self._prototypes = (self._accums / counts > 0.5).float()

    def predict(self, x: torch.Tensor) -> Tuple[int, float]:
        """Predict class via Hamming similarity."""
        assert self._prototypes is not None
        hv = self._encode(x)
        sims = hv_batch_sim(hv, self._prototypes)
        pred = int(sims.argmax().item())
        return pred, float(sims[pred])

    def accuracy(self, X: torch.Tensor, y: torch.Tensor) -> float:
        correct = sum(1 for i in range(X.shape[0])
                      if self.predict(X[i])[0] == int(y[i].item()))
        return correct / X.shape[0]

    def online_refine(self, x: torch.Tensor, label: int, lr: float = 0.1):
        """
        Online RefineHD update for the CA90 classifier.

        After finalization, incrementally refine prototypes from new labelled
        samples without storing full basis matrices:
          wrong prediction: pull correct prototype, push predicted one
          correct prediction: mild reinforcement

        Args:
            x:     (n_features,) input
            label: True class label
            lr:    Blending rate
        """
        if self._prototypes is None:
            self.finalize()

        hv   = self._encode(x).float()
        sims = torch.stack([
            1.0 - (hv != self._prototypes[c].float()).float().mean()
            for c in range(self.n_classes)
        ])
        pred = int(sims.argmax().item())

        with torch.no_grad():
            if pred != label:
                self._prototypes[label] = (
                    (1 - lr) * self._prototypes[label].float() + lr * hv > 0.5
                ).float()
                self._prototypes[pred] = (
                    (1 - lr) * self._prototypes[pred].float() + lr * (1 - hv) > 0.5
                ).float()

    def memory_report(self) -> Dict:
        standard_kb = self.feature_mem.standard_memory_bytes / 1024
        ca90_kb = self.feature_mem.memory_bytes / 1024
        return {
            "n_features": self.n_features,
            "hd_dim": self.hd_dim,
            "seed_dim": self.feature_mem.seed_dim,
            "standard_basis_kb": round(standard_kb, 2),
            "ca90_seeds_kb": round(ca90_kb, 3),
            "memory_reduction": round(self.feature_mem.memory_reduction, 0),
            "ca90_period": self.feature_mem._period,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_ca90_properties():
    print("=" * 60)
    print("Testing CA90 core properties (Kleyko/Frady/Sommer 2020)")
    print("=" * 60)

    # Verify CA90 rule
    state = torch.tensor([1., 0., 1., 0., 0.])
    next_s = ca90_step(state)
    # next[0] = state[4] XOR state[1] = 0 XOR 0 = 0
    # next[1] = state[0] XOR state[2] = 1 XOR 1 = 0
    # next[2] = state[1] XOR state[3] = 0 XOR 0 = 0
    # next[3] = state[2] XOR state[4] = 1 XOR 0 = 1
    # next[4] = state[3] XOR state[0] = 0 XOR 1 = 1
    print(f"  CA90 step: {state.int().tolist()} → {next_s.int().tolist()}")

    # Randomization: expanded seed should have ~0.5 Hamming distance to random
    torch.manual_seed(0)
    seed_a = (torch.rand(37) < 0.5).float()
    seed_b = (torch.rand(37) < 0.5).float()

    hv_a = ca90_expand(seed_a, 10000)
    hv_b = ca90_expand(seed_b, 10000)
    density_a = float(hv_a.mean())
    sim_ab    = float(hv_batch_sim(hv_a, hv_b.unsqueeze(0))[0])

    print(f"  Expansion density (want ≈ 0.5): {density_a:.4f}")
    print(f"  Sim between different expansions: {sim_ab:.4f}  (want ≈ 0.5)")
    assert 0.45 < density_a < 0.55, f"CA90 expansion density off: {density_a}"
    assert 0.45 < sim_ab < 0.55, f"CA90 cross-sim off: {sim_ab}"

    # Cyclic shift invariance: CA90(Sh(a, i), t) = Sh(CA90(a, t), i)
    seed_shifted = torch.roll(seed_a, 3)
    hv_a_shifted = ca90_expand(seed_shifted, 10000)
    hv_a_then_shifted = torch.roll(hv_a, 3)
    sim_shift = float(hv_batch_sim(hv_a_shifted, hv_a_then_shifted.unsqueeze(0))[0])
    print(f"  Cyclic shift invariance: sim = {sim_shift:.4f}  (want ≈ 1.0)")
    assert sim_shift > 0.95, "CA90 shift invariance violated"

    # Randomization period
    period_37 = ca90_randomization_period(37)
    period_32 = ca90_randomization_period(32)
    print(f"  Period (N=37 odd prime): {period_37:,}  (want huge)")
    print(f"  Period (N=32 = 2^5):    {period_32:,}  (want 7 = 2^3 - 1)")
    assert period_32 == 7, f"N=32 period should be 7: {period_32}"

    print("  ✅ CA90 properties OK")


def test_ca90_item_memory():
    print("=" * 60)
    print("Testing CA90ItemMemory (space-time tradeoff)")
    print("=" * 60)

    torch.manual_seed(42)
    mem = CA90ItemMemory(item_dim=5000, seed_dim=37, seed=0)

    # Add 50 items
    for i in range(50):
        mem.add(f"item_{i}")

    # Retrieve and verify properties
    hv_0 = mem.get("item_0")
    hv_1 = mem.get("item_1")
    assert hv_0 is not None and hv_0.shape == (5000,)

    # Different items should be near-orthogonal
    sim = float(hv_batch_sim(hv_0, hv_1.unsqueeze(0))[0])
    print(f"  Hamming sim between items: {sim:.4f}  (want ≈ 0.5)")
    assert 0.4 < sim < 0.6

    # Memory report
    report = mem.memory_report() if hasattr(mem, 'memory_report') else {}
    std_kb = mem.standard_memory_bytes / 1024
    ca90_kb = mem.memory_bytes / 1024
    reduction = mem.memory_reduction
    print(f"  Standard basis: {std_kb:.1f}KB")
    print(f"  CA90 seeds:     {ca90_kb:.3f}KB")
    print(f"  Memory savings: {reduction:.0f}×")
    assert reduction > 50, f"Should have large savings: {reduction:.0f}×"

    # Nearest neighbor search
    results = mem.nearest(hv_0, top_k=1)
    print(f"  Nearest to item_0: {results[0][0]}, sim={results[0][1]:.4f}")
    assert results[0][0] == "item_0"

    print("  ✅ CA90ItemMemory OK")


def test_ca90_classifier():
    print("=" * 60)
    print("Testing CA90HDCClassifier (memory-efficient HDC)")
    print("=" * 60)

    torch.manual_seed(99)
    n_features, n_classes = 20, 4
    clf = CA90HDCClassifier(n_features, n_classes, hd_dim=3000, seed_dim=37, seed=7)

    # Binary cluster data
    X, y = [], []
    for c in range(n_classes):
        base = torch.zeros(n_features)
        base[c * 5:(c + 1) * 5] = 1.0
        for _ in range(20):
            x = base.clone()
            mask = torch.rand(n_features) < 0.1
            x[mask] = 1.0 - x[mask]
            X.append(x); y.append(c)
    X = torch.stack(X); y = torch.tensor(y)

    for i in range(X.shape[0]):
        clf.train_step(X[i], int(y[i]))
    clf.finalize()

    acc = clf.accuracy(X, y)
    report = clf.memory_report()

    print(f"  Accuracy: {acc:.1%}")
    print(f"  Memory report: {report}")
    assert acc > 0.5, f"Accuracy too low: {acc:.1%}"
    assert report["memory_reduction"] > 50

    print("  ✅ CA90HDCClassifier OK")


if __name__ == "__main__":
    test_ca90_properties()
    print()
    test_ca90_item_memory()
    print()
    test_ca90_classifier()
    print()
    print("=== All CA90-HDC tests passed ===")
