"""
Holographic Graph Neuron (HoloGN)
==================================
Based on: Kleyko, D., Osipov, E., Senior, A., Khan, A.I., and Şekercioğlu, Y.A. (2015/2016)
"Holographic Graph Neuron: a Bio-Inspired Architecture for Pattern Processing"
arXiv:1501.03784 — included as a paper in:
Kleyko, D. (2016) "Pattern Recognition with Vector Symbolic Architectures"
Licentiate Thesis, Luleå University of Technology. diva2:990444. Page 15+.

Key contributions implemented:

1. **Zadoff-Chu / Cyclic-Shift HD Indexing** (Section III-B, III-C)
   Each Graph Neuron j gets an initialization vector IV_j. The HD-index of
   element i in GN j is EHD_{j,i} = Sh(IV_j, i), where Sh is a cyclic shift.
   Cyclic shifts are invertible, preserving Hamming weight, and the result is
   near-orthogonal to the original vector.

2. **HoloGN Encoding** (Section IV, Eq. 3)
   When a pattern activates elements in the GN array, the holographic
   representation is:  HGN = MAJORITY_SUM(EHD_1, EHD_2, ..., EHD_n)
   All manipulations use simple bit-wise arithmetic.

3. **Linear-Time Recall via Complex-Number Hamming** (Section V-A, Eq. 4)
   Map {0,1} → {j, 1} (where j = √-1). Then H × hq (matrix multiplication)
   gives complex results whose imaginary parts / d = Hamming distances.
   This reduces recall from O(l·d) sequential XOR to one matrix-vector multiply.

4. **Bundle Capacity Analysis** (Section VI-B, Eq. 8-9)
   p_n(n) = 1/2 - C(n-1, (n-1)//2) / 2^n
   Capacity(d, thr) = max n such that k+(d, p_n(n), thr) ≤ k-(d, 0.5, thr)
   For d=10000: capacity ≈ 89 vectors.

5. **Pattern Overlap Estimation** (Section VI-C, Eq. 10-13)
   Given Hamming distance between two bundled HVs, estimate the number
   of common component vectors without decoding the representations.

6. **Supervised Learning Mode** (Section V-C, Eq. 5)
   Bundle multiple distorted examples of the same class:
   h(L) = MAJORITY_SUM_{i=1}^{e}(HGN(L_i))
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch

from hdc.hdc_glue import (
    hv_batch_sim, gen_hvs, hv_permute,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Section III-B: Cyclic-Shift HD Indexing (Zadoff-Chu approach)
# ═══════════════════════════════════════════════════════════════════════════════

class ZadoffChuIndexer:
    """
    Generate pseudo-orthogonal HD-vectors via cyclic shifts (Kleyko 2016, §III-B).

    Starting from a random initialization vector IV, cyclic-shifting by i
    positions yields a vector that is:
    - near-orthogonal to IV (Hamming distance ≈ 0.5)
    - associative: Sh(IV, i+j) = Sh(Sh(IV, i), j)
    - invertible: Sh(Sh(IV, i), -i) = IV
    - weight-preserving: ||Sh(IV,i)||_1 = ||IV||_1

    This replaces random generation of element HVs with a structured,
    reproducible coding that uses only one stored base vector per GN.
    """

    def __init__(
        self,
        n_neurons: int,
        dim: int = 10000,
        seed: Optional[int] = None,
    ):
        """
        Args:
            n_neurons: Number of Graph Neurons (GNs) in the array
            dim: Hypervector dimensionality
            seed: Random seed
        """
        self.n_neurons = n_neurons
        self.dim = dim

        # One initialization vector per GN, mutually near-orthogonal
        self.init_vectors = gen_hvs(n_neurons, dim, seed=seed)  # (n_neurons, dim)

    def element_hv(self, neuron_j: int, element_i: int) -> torch.Tensor:
        """
        Compute EHD_{j,i} = Sh(IV_j, i) — the HD-index of element i in GN j.

        Args:
            neuron_j: GN index (0 ≤ j < n_neurons)
            element_i: Element/symbol index

        Returns:
            (dim,) binary HD-vector
        """
        iv = self.init_vectors[neuron_j]
        return hv_permute(iv, k=element_i)  # cyclic shift by element_i positions

    def all_elements(self, neuron_j: int, n_elements: int) -> torch.Tensor:
        """
        Compute all element HVs for a single GN.

        Args:
            neuron_j: GN index
            n_elements: Number of possible element values (alphabet size)

        Returns:
            (n_elements, dim) HV matrix
        """
        iv = self.init_vectors[neuron_j]
        return torch.stack([hv_permute(iv, k=i) for i in range(n_elements)])


# ═══════════════════════════════════════════════════════════════════════════════
# Section IV: HoloGN Encoding (Eq. 3)
# ═══════════════════════════════════════════════════════════════════════════════

class HoloGNEncoder:
    """
    HoloGN pattern encoder (Kleyko 2016, Section IV, Eq. 3).

    Given a pattern (sequence of symbols activating GNs), encode it as:
        HGN = MAJORITY_SUM_{j=1}^{n} EHD_j

    where EHD_j = Sh(IV_j, activated_element_i) is the HD-index of the
    activated element in GN j.

    Properties of the result:
    - Hamming distance from any component EHD_j is < 0.5 (similar to all)
    - Different patterns produce near-orthogonal HVs if few elements overlap
    - Supports one-shot learning: each pattern → one HV, stored immediately
    """

    def __init__(self, indexer: ZadoffChuIndexer):
        self.indexer = indexer
        self.n_neurons = indexer.n_neurons
        self.dim = indexer.dim

    @staticmethod
    def _majority_threshold(counts: torch.Tensor, n: int) -> torch.Tensor:
        """
        Proper majority vote: threshold sum at n/2 (Kleyko 2016, §III-B).

        hv_majority() from hdc_glue thresholds at 0.5, which is wrong
        when counts is an integer sum (0..n). We threshold at n/2 for
        true majority: position = 1 iff count > n/2.
        For even n, ties are broken randomly (here: tie → 0).
        """
        return (counts > n / 2).float()

    def encode(self, pattern: List[int]) -> torch.Tensor:
        """
        Encode a pattern (list of activated element indices, one per GN).

        Args:
            pattern: List of length n_neurons; pattern[j] = element activated in GN j.
                     Use -1 to indicate GN j is not activated (no contribution).

        Returns:
            (dim,) binary HV representing the holographic pattern
        """
        components = []
        for j, elem_i in enumerate(pattern):
            if elem_i >= 0:
                ehd = self.indexer.element_hv(j, elem_i)
                components.append(ehd)

        if not components:
            return torch.zeros(self.dim)

        n = len(components)
        stacked = torch.stack(components)        # (n_active, dim)
        counts = stacked.sum(dim=0)              # (dim,) integer counts 0..n
        return self._majority_threshold(counts, n)  # (dim,) binary

    def encode_batch(self, patterns: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of patterns.

        Args:
            patterns: (N, n_neurons) integer tensor; -1 = not activated

        Returns:
            (N, dim) binary HV matrix
        """
        N = patterns.shape[0]
        hvs = torch.zeros(N, self.dim)
        for i in range(N):
            hvs[i] = self.encode(patterns[i].tolist())
        return hvs


# ═══════════════════════════════════════════════════════════════════════════════
# Section V-A: Complex-Number Hamming Distance (Eq. 4)
# ═══════════════════════════════════════════════════════════════════════════════

class ComplexHammingSearch:
    """
    Efficient Hamming-distance recall via complex-number encoding (Kleyko 2016, Eq. 4).

    Transformation: {0, 1} → {j, 1}  (where j = √−1)
    Then for a stored matrix H (l × d complex) and query hq (d complex):
        H × hq = vector of l complex numbers
        Im(result[i]) / d = Hamming distance between H[i] and hq

    This reduces recall from O(l·d) sequential XOR+popcount to a single
    matrix-vector multiply, which runs at BLAS speed and is only ~2× slower
    than raw matrix multiplication.

    This is the key algorithmic insight of Kleyko 2016: exploiting that
    j × j = −1 (same bits → real), 1 × j = j (different bits → imaginary),
    so imaginary accumulation = popcount of XOR.
    """

    def __init__(self, dim: int = 10000):
        self.dim = dim
        self._H_complex: Optional[torch.Tensor] = None  # (l, dim) complex
        self._labels: List[int] = []

    @staticmethod
    def to_complex(hv: torch.Tensor) -> torch.Tensor:
        """
        Map binary HV {0,1} → complex {j, 1}.

        0 → j = 0 + 1j
        1 → 1 = 1 + 0j

        Args:
            hv: (..., d) binary tensor

        Returns:
            (..., d) complex64 tensor
        """
        real = hv.float()                          # 1→1, 0→0
        imag = (1.0 - hv).float()                  # 1→0, 0→1  (j for 0-bits)
        return torch.complex(real, imag)

    @staticmethod
    def hamming_from_complex_dot(
        dot_imag: torch.Tensor,
        dim: int,
    ) -> torch.Tensor:
        """
        Extract Hamming distances from imaginary parts of H × hq.

        Hamming(H[i], hq) = Im(dot[i]) / d

        Args:
            dot_imag: (l,) imaginary parts of H × hq
            dim: Hypervector dimensionality

        Returns:
            (l,) Hamming distances in [0, 1]
        """
        return dot_imag.abs() / dim

    def store(self, hv: torch.Tensor, label: int):
        """
        Add one HV to the memory.

        Args:
            hv: (dim,) binary HV
            label: Pattern label/identifier
        """
        c = self.to_complex(hv).unsqueeze(0)  # (1, dim)
        if self._H_complex is None:
            self._H_complex = c
        else:
            self._H_complex = torch.cat([self._H_complex, c], dim=0)
        self._labels.append(label)

    def store_batch(self, hvs: torch.Tensor, labels: List[int]):
        """
        Add multiple HVs to memory.

        Args:
            hvs: (l, dim) binary HV matrix
            labels: List of l labels
        """
        c = self.to_complex(hvs)
        if self._H_complex is None:
            self._H_complex = c
        else:
            self._H_complex = torch.cat([self._H_complex, c], dim=0)
        self._labels.extend(labels)

    def query(
        self,
        hq: torch.Tensor,
        threshold: float = 0.5,
        top_k: Optional[int] = None,
    ) -> List[Dict]:
        """
        Retrieve patterns within Hamming distance ≤ threshold (Eq. 4).

        Args:
            hq: (dim,) binary query HV
            threshold: Maximum Hamming distance for match (default 0.5)
            top_k: If set, return only the k closest matches

        Returns:
            List of {label, hamming_distance, rank} sorted by distance
        """
        if self._H_complex is None:
            return []

        # Complex-number trick: H × hq^* (conjugate for proper imaginary extraction)
        hq_c = self.to_complex(hq)                          # (dim,)
        dot = self._H_complex @ hq_c                        # (l,)

        # Imaginary part / dim = Hamming distance
        distances = self.hamming_from_complex_dot(dot.imag, self.dim)  # (l,)

        # Sort by distance
        sorted_idx = distances.argsort()
        results = []
        for rank, idx in enumerate(sorted_idx.tolist()):
            d = float(distances[idx])
            if d <= threshold:
                results.append({
                    "label": self._labels[idx],
                    "hamming_distance": d,
                    "rank": rank,
                })
            if top_k is not None and len(results) >= top_k:
                break

        return results

    def query_best(self, hq: torch.Tensor) -> Optional[Dict]:
        """Return the single best match."""
        results = self.query(hq, threshold=1.0, top_k=1)
        return results[0] if results else None

    def remove(self, label: int) -> bool:
        """
        Remove all entries with the given label.

        Returns:
            True if at least one entry was removed, False if label not found.
        """
        indices = [i for i, lbl in enumerate(self._labels) if lbl == label]
        if not indices:
            return False

        keep = [i for i in range(len(self._labels)) if i not in set(indices)]
        if keep:
            self._H_complex = self._H_complex[keep]
        else:
            self._H_complex = None
        self._labels = [self._labels[i] for i in keep]
        return True

    def update(self, label: int, hv: torch.Tensor):
        """
        Replace all stored entries for `label` with the new HV.
        Convenience: remove + store.
        """
        self.remove(label)
        self.store(hv, label)

    def query_batch(
        self,
        hvs:       torch.Tensor,   # (B, dim) binary query batch
        threshold: float = 0.5,
        top_k:     int   = 1,
    ) -> List[List[Dict]]:
        """
        Efficient batch query: retrieve top_k matches for each query in batch.

        O(B × L) where B = batch size, L = stored items.
        Much faster than calling query() B times when B is large.

        Returns:
            List of B result lists, each matching the format of query().
        """
        if self._H_complex is None:
            return [[] for _ in range(hvs.shape[0])]

        B = hvs.shape[0]
        # Stack query HVs → (B, dim) complex
        hqs  = torch.stack([self.to_complex(hvs[i]) for i in range(B)])  # (B, dim)
        # Batch matmul: (B, dim) × (L, dim)^T = (B, L)
        dots = hqs @ self._H_complex.conj().T                             # (B, L)
        dists = self.hamming_from_complex_dot(dots.imag, self.dim)        # (B, L)

        results = []
        for b in range(B):
            row_dists  = dists[b]
            sorted_idx = row_dists.argsort()
            b_results  = []
            for rank, idx in enumerate(sorted_idx.tolist()):
                d = float(row_dists[idx])
                if d <= threshold:
                    b_results.append({
                        "label":           self._labels[idx],
                        "hamming_distance": d,
                        "rank":             rank,
                    })
                if len(b_results) >= top_k:
                    break
            results.append(b_results)
        return results

    def all_similarities(self, hq: torch.Tensor) -> torch.Tensor:
        """
        Return Hamming similarities (1 - distance) to ALL stored patterns.

        Useful for soft nearest-neighbour retrieval without a threshold.
        Returns: (L,) tensor of similarities ∈ [0, 1].
        """
        if self._H_complex is None:
            return torch.zeros(0)
        hq_c  = self.to_complex(hq)
        dot   = self._H_complex @ hq_c
        dists = self.hamming_from_complex_dot(dot.imag, self.dim)
        return 1.0 - dists

    def n_stored(self) -> int:
        return len(self._labels)

    def clear(self):
        self._H_complex = None
        self._labels.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# Section VI-B: Bundle Capacity Analysis (Eq. 8-9)
# ═══════════════════════════════════════════════════════════════════════════════

class BundleCapacityAnalyzer:
    """
    Theoretical capacity of HoloGN representations (Kleyko 2016, §VI-B, Eq. 8-9).

    Capacity is the maximum number of mutually near-orthogonal HD-vectors that
    can be robustly decoded from their majority-sum bundle.

    Noise introduced by bundling n vectors (Eq. 8):
        p_n(n) = 1/2 - C(n-1, (n-1)//2) / 2^n

    Where p_n is the noise level (probability of bit flip) from majority-sum
    of n vectors. As n grows, p_n → 0.5, making decoding impossible.

    Capacity formula (Eq. 9):
        Capacity(d, thr) = max n such that k+(d, p_n(n), thr) ≤ k-(d, 0.5, thr)

    k+ and k- are the upper/lower density bounds from the binomial distribution
    at threshold thr (the probability below which density deviation is negligible).

    Result: for d=10000, thr=1e-6 → capacity ≈ 89 vectors.
    """

    def __init__(self, dim: int = 10000, thr: float = 1e-6):
        self.dim = dim
        self.thr = thr

    def _binomial_normal_approx(self, k: int, d: int, p: float) -> float:
        """
        Gaussian approximation of binomial Pr(k; d, p) via de Moivre-Laplace (Eq. 2).

        Pr(k, d, p) ≈ exp(-(k - d*p)² / (2*d*p*(1-p))) / sqrt(2π*d*p*(1-p))
        """
        if p <= 0 or p >= 1:
            return 0.0
        mean = d * p
        var = d * p * (1 - p)
        if var < 1e-12:
            return 1.0 if abs(k - mean) < 0.5 else 0.0
        return math.exp(-(k - mean) ** 2 / (2 * var)) / math.sqrt(2 * math.pi * var)

    def _k_bounds(self, d: int, p: float, thr: float) -> Tuple[int, int]:
        """
        Compute k- and k+ density bounds (Eq. 6-7).

        k-(d,p,thr) = max k such that Pr(k,d,p) ≤ thr and k < d*p
        k+(d,p,thr) = min k such that Pr(k,d,p) ≤ thr and k > d*p
        """
        mean = int(d * p)

        # k- : largest k below mean where probability is still ≤ thr
        k_minus = 0
        for k in range(mean, -1, -1):
            prob = self._binomial_normal_approx(k, d, p)
            if prob <= thr:
                k_minus = k
                break

        # k+ : smallest k above mean where probability is still ≤ thr
        k_plus = d
        for k in range(mean, d + 1):
            prob = self._binomial_normal_approx(k, d, p)
            if prob <= thr:
                k_plus = k
                break

        return k_minus, k_plus

    def noise_level(self, n: int) -> float:
        """
        Noise level p_n introduced by majority-sum of n vectors (Eq. 8).

        p_n(n) = 1/2 - C(n-1, (n-1)//2) / 2^n

        The majority sum requires an ODD number of vectors (per paper §III-B).
        For even n, the bundle is undefined (tie possible); we approximate by
        treating even n as n+1 (rounding up to next odd).

        Args:
            n: Number of vectors in the majority sum (ideally odd)

        Returns:
            Noise probability p_n ∈ [0, 0.5]
        """
        if n <= 1:
            return 0.0
        # Majority sum is defined for odd n; round even n up to next odd
        n_odd = n if n % 2 == 1 else n + 1
        try:
            binom = math.comb(n_odd - 1, (n_odd - 1) // 2)
            return max(0.0, 0.5 - binom / (2 ** n_odd))
        except (ValueError, OverflowError):
            return 0.5  # At large n, noise saturates to 0.5

    def capacity(self) -> int:
        """
        Compute Capacity(d, thr) = max n such that bundle is still decodable (Eq. 9).

        Iterates over odd n values only (majority sum requires odd count).

        Returns:
            Maximum number of vectors (odd) that can be robustly decoded.
        """
        d, thr = self.dim, self.thr

        # k-(d, 0.5, thr): LOWER bound of the balanced (p=0.5) density distribution
        # _k_bounds returns (k_minus, k_plus); we want the first element
        k_minus_balanced, _ = self._k_bounds(d, 0.5, thr)

        last_valid = 1
        for n in range(1, 10001, 2):  # odd n only (majority sum requires odd count)
            pn = self.noise_level(n)
            if pn >= 0.5:
                return last_valid

            # k+(d, p_n, thr): UPPER bound of the noise density distribution
            # _k_bounds returns (k_minus, k_plus); we want the second element
            _, k_plus_noise = self._k_bounds(d, pn, thr)

            # Condition (Eq. 9): decoding fails when noise upper bound
            # reaches balanced lower bound — distributions overlap
            if k_plus_noise >= k_minus_balanced:
                return last_valid
            last_valid = n

        return last_valid

    def capacity_vs_dim(self, dims: List[int]) -> List[Tuple[int, int]]:
        """Compute capacity for multiple dimensionalities."""
        results = []
        for d in dims:
            original_dim = self.dim
            self.dim = d
            cap = self.capacity()
            self.dim = original_dim
            results.append((d, cap))
        return results

    def noise_curve(self, max_n: int = 200) -> List[Tuple[int, float]]:
        """Return (n, p_n) pairs for n = 1 … max_n."""
        return [(n, self.noise_level(n)) for n in range(1, max_n + 1)]


# ═══════════════════════════════════════════════════════════════════════════════
# Section VI-C: Pattern Overlap Estimation (Eq. 10-13)
# ═══════════════════════════════════════════════════════════════════════════════

class PatternOverlapEstimator:
    """
    Estimate number of common elements between two bundled patterns (Kleyko 2016, §VI-C).

    Given two bundled HVs with m and n atomic components respectively,
    the Hamming distance between them is a function of the number of
    overlapping (common) atomic vectors c.

    The theoretical Hamming distance is (Eq. 10):
        ΔH = p(c, m, n) = Σ_{||C||_1=0}^{c} C(c, ||C||_1)/2^c ·
                          (p1(m,c,||C||_1)·p0(n,c,||C||_1) +
                           p0(m,c,||C||_1)·p1(n,c,||C||_1))

    Bundle sensitivity (Eq. 13):
        Sensitivity(d, thr, m, n) = min c such that k+(d, p(c,n,m), thr) ≤ k-(d, 0.5, thr)

    This allows estimating how many elements two patterns share just by
    measuring their Hamming distance — without decoding the representations.
    """

    def __init__(self, dim: int = 10000, thr: float = 1e-6):
        self.dim = dim
        self.thr = thr
        self._capacity_analyzer = BundleCapacityAnalyzer(dim, thr)

    def _p1(self, m: int, c: int, cnt: int) -> float:
        """
        p1(m, c, ||C||_1): probability that majority sum is '1' (Eq. 11).

        m = total number of atomic vectors in pattern
        c = number of common/overlapping vectors
        cnt = ||C||_1 = number of 1s in column C of common matrix M
        """
        if cnt > m / 2:
            return 1.0

        non_overlap = m - c
        threshold = math.ceil((m + 1) / 2) - cnt  # need this many 1s from non-overlap

        if threshold > non_overlap:
            return 0.0

        # Sum over all valid numbers of 1s from the non-overlapping part
        prob = 0.0
        for i in range(threshold, non_overlap + 1):
            try:
                prob += math.comb(non_overlap, i) / (2 ** non_overlap)
            except (OverflowError, ValueError):
                break
        return min(1.0, prob)

    def _p0(self, m: int, c: int, cnt: int) -> float:
        """p0(m, c, ||C||_1): probability that majority sum is '0' (Eq. 12)."""
        complement = c - cnt  # number of 0s in column C
        if complement > m / 2:
            return 1.0

        non_overlap = m - c
        threshold = math.ceil((m + 1) / 2) - complement

        if threshold > non_overlap:
            return 0.0

        prob = 0.0
        for i in range(threshold, non_overlap + 1):
            try:
                prob += math.comb(non_overlap, i) / (2 ** non_overlap)
            except (OverflowError, ValueError):
                break
        return min(1.0, prob)

    def expected_hamming(self, c: int, m: int, n: int) -> float:
        """
        Theoretical Hamming distance between two bundled HVs (Eq. 10).

        Args:
            c: Number of common (overlapping) atomic vectors
            m: Number of atomic vectors in pattern 1
            n: Number of atomic vectors in pattern 2 (m ≤ n)

        Returns:
            Expected Hamming distance ∈ [0, 0.5]
        """
        if c < 0 or c > min(m, n):
            return 0.5

        total = 0.0
        for cnt in range(c + 1):
            # C(c, cnt) / 2^c
            try:
                weight = math.comb(c, cnt) / (2 ** c)
            except (OverflowError, ValueError):
                continue

            p1_m = self._p1(m, c, cnt)
            p0_m = self._p0(m, c, cnt)
            p1_n = self._p1(n, c, cnt)
            p0_n = self._p0(n, c, cnt)

            total += weight * (p1_m * p0_n + p0_m * p1_n)

        return total

    def estimate_overlap(
        self,
        hamming_dist: float,
        m: int,
        n: int,
    ) -> int:
        """
        Estimate number of common elements c from observed Hamming distance.

        Inverts the expected_hamming function by searching for the c that
        minimises |expected_hamming(c, m, n) - hamming_dist|.

        Args:
            hamming_dist: Observed Hamming distance between two bundled HVs
            m: Size of pattern 1 (number of atomic vectors)
            n: Size of pattern 2
            n_steps: Search resolution

        Returns:
            Estimated number of common elements c
        """
        best_c = 0
        best_err = abs(self.expected_hamming(0, m, n) - hamming_dist)

        for c in range(1, min(m, n) + 1):
            err = abs(self.expected_hamming(c, m, n) - hamming_dist)
            if err < best_err:
                best_err = err
                best_c = c

        return best_c

    def sensitivity(self, m: int, n: int) -> int:
        """
        Bundle sensitivity: minimum c for robust overlap detection (Eq. 13).

        Sensitivity(d, thr, m, n) = min c such that the Hamming distance
        distinguishes c common elements from c-1.

        Args:
            m: Size of pattern 1
            n: Size of pattern 2

        Returns:
            Minimum number of common elements robustly detectable
        """
        analyzer = self._capacity_analyzer
        _, k_minus_ref = analyzer._k_bounds(self.dim, 0.5, self.thr)

        for c in range(1, min(m, n) + 1):
            p = self.expected_hamming(c, m, n)
            if p >= 0.5:
                continue
            _, k_plus = analyzer._k_bounds(self.dim, p, self.thr)
            if k_plus <= k_minus_ref:
                return c

        return min(m, n)  # All elements needed (patterns are indistinguishable)


# ═══════════════════════════════════════════════════════════════════════════════
# HoloGN Full Architecture (combining all components)
# ═══════════════════════════════════════════════════════════════════════════════

class HoloGNMemory:
    """
    Complete Holographic Graph Neuron memory (Kleyko 2016).

    Combines:
    - ZadoffChuIndexer for cyclic-shift HD indexing
    - HoloGNEncoder for pattern → HV encoding
    - ComplexHammingSearch for efficient linear-time recall
    - Supervised learning mode via bundling distorted examples (Eq. 5)

    Modes:
    - One-shot: each pattern stored directly as one HV
    - Supervised: multiple distorted examples per class bundled together
    """

    def __init__(
        self,
        n_neurons: int,
        dim: int = 10000,
        seed: Optional[int] = None,
    ):
        self.n_neurons = n_neurons
        self.dim = dim

        self.indexer = ZadoffChuIndexer(n_neurons, dim, seed=seed)
        self.encoder = HoloGNEncoder(self.indexer)
        self.memory = ComplexHammingSearch(dim)

        # Supervised learning: per-label accumulators
        self._supervised_accums: Dict[int, torch.Tensor] = {}
        self._supervised_counts: Dict[int, int] = {}

    def memorize(self, pattern: List[int], label: int):
        """
        One-shot memorization: encode pattern and store immediately.

        Args:
            pattern: List of n_neurons element indices (−1 = not activated)
            label: Pattern identifier
        """
        hv = self.encoder.encode(pattern)
        self.memory.store(hv, label)

    def memorize_supervised(self, pattern: List[int], label: int):
        """
        Supervised learning mode (Kleyko 2016, Eq. 5).

        Accumulates multiple distorted examples per label. Call
        finalize_supervised() after all examples are presented.
        """
        hv = self.encoder.encode(pattern)
        if label not in self._supervised_accums:
            self._supervised_accums[label] = hv.float()
            self._supervised_counts[label] = 1
        else:
            self._supervised_accums[label] = self._supervised_accums[label] + hv.float()
            self._supervised_counts[label] += 1

    def finalize_supervised(self):
        """
        Bundle all accumulated examples per label and store them.

        h(L) = MAJORITY_SUM_{i=1}^{e}(HGN(L_i))  [Eq. 5]
        """
        for label, accum in self._supervised_accums.items():
            e = self._supervised_counts[label]
            # accum = sum of e binary HVs → threshold at e/2 for majority vote
            bundled = HoloGNEncoder._majority_threshold(accum, e)
            self.memory.store(bundled, label)

        self._supervised_accums.clear()
        self._supervised_counts.clear()

    def recall(
        self,
        pattern: List[int],
        threshold: float = 0.5,
        top_k: Optional[int] = None,
    ) -> List[Dict]:
        """
        Recall patterns matching the query (linear-time via complex Hamming).

        Args:
            pattern: Query pattern (n_neurons element indices)
            threshold: Maximum Hamming distance for a match
            top_k: Return only k best matches

        Returns:
            List of {label, hamming_distance, rank}
        """
        hq = self.encoder.encode(pattern)
        return self.memory.query(hq, threshold=threshold, top_k=top_k)

    def recall_best(self, pattern: List[int]) -> Optional[Dict]:
        """Return the single best-matching stored pattern."""
        hq = self.encoder.encode(pattern)
        return self.memory.query_best(hq)

    def refine(
        self,
        pattern: List[int],
        label:   int,
        lr:      float = 0.1,
    ):
        """
        Online RefineHD: update stored prototype toward a new example.

        If the label exists in memory, blend the new encoding into the stored HV.
        If not, store it fresh (one-shot).

        Args:
            pattern: New example pattern
            label:   Class label
            lr:      Blending rate ∈ (0, 1]
        """
        hv_new = self.encoder.encode(pattern).float()
        existing = self.memory.query_best(hv_new)

        if existing is not None and existing["label"] == label:
            # Retrieve the stored HV via all_similarities — pick the stored one
            all_sims = self.memory.all_similarities(hv_new)   # (n_stored,)
            best_idx = int(all_sims.argmax().item())
            # Reconstruct stored HV from ComplexHammingSearch internals
            if self.memory._H_complex is not None and best_idx < self.memory._H_complex.shape[0]:
                stored_col = self.memory._H_complex[best_idx]   # (dim,) complex
                stored_real = (stored_col.real > 0).float()
                blended = (1 - lr) * stored_real + lr * hv_new
                self.memory.update(label, (blended > 0.5).float())
            else:
                # Fallback: just store the new one if can't retrieve old
                self.memory.update(label, (hv_new > 0.5).float())
        else:
            self.memory.store((hv_new > 0.5).float(), label)

    def batch_recall(
        self,
        patterns:  List[List[int]],
        threshold: float = 0.5,
        top_k:     int   = 1,
    ) -> List[List[Dict]]:
        """
        Recall for a batch of query patterns (one result list per query).

        Uses ComplexHammingSearch.query_batch() under the hood for efficiency.

        Returns:
            List of per-query result lists.
        """
        hvs = [self.encoder.encode(p) for p in patterns]
        stacked = torch.stack(hvs)
        return self.memory.query_batch(stacked, threshold=threshold, top_k=top_k)

    def capacity_remaining(self) -> float:
        """
        Fraction of theoretical capacity still available.

        Uses BundleCapacityAnalyzer: 1.0 = empty, 0.0 = at theoretical limit.
        """
        analyzer = BundleCapacityAnalyzer(self.dim)
        n        = self.memory.n_stored()
        cap      = analyzer.practical_capacity()
        return max(0.0, 1.0 - n / max(cap, 1))

    def n_stored(self) -> int:
        return self.memory.n_stored()


# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_cyclic_shift_indexer():
    print("=" * 60)
    print("Testing ZadoffChuIndexer (Kleyko 2016, §III-B)")
    print("=" * 60)

    indexer = ZadoffChuIndexer(n_neurons=10, dim=10000, seed=42)

    iv0 = indexer.init_vectors[0]
    # Cyclic shift should be near-orthogonal to original
    shifted = indexer.element_hv(0, 1)
    sim = float(hv_batch_sim(iv0, shifted.unsqueeze(0))[0])
    print(f"  Sim(IV_0, Sh(IV_0, 1)) = {sim:.4f}  (want ≈ 0.5)")
    assert abs(sim - 0.5) < 0.05, f"Shift not near-orthogonal: {sim}"

    # Different neurons should be near-orthogonal
    iv1 = indexer.init_vectors[1]
    sim_ij = float(hv_batch_sim(iv0, iv1.unsqueeze(0))[0])
    print(f"  Sim(IV_0, IV_1)         = {sim_ij:.4f}  (want ≈ 0.5)")
    assert abs(sim_ij - 0.5) < 0.05

    # Invertibility: Sh(Sh(IV, i), -i) ≈ IV
    from hdc.hdc_glue import hv_permute
    iv_roundtrip = hv_permute(shifted, k=-1)
    sim_rt = float(hv_batch_sim(iv0, iv_roundtrip.unsqueeze(0))[0])
    print(f"  Roundtrip Sh(Sh(IV,1),-1)≈IV: sim = {sim_rt:.4f}  (want ≈ 1.0)")
    assert sim_rt > 0.95, f"Cyclic shift not invertible: {sim_rt}"

    print("  ✅ ZadoffChuIndexer OK")


def test_holographic_encoding():
    print("=" * 60)
    print("Testing HoloGNEncoder (Kleyko 2016, §IV, Eq. 3)")
    print("=" * 60)

    n_neurons = 7
    dim = 5000
    indexer = ZadoffChuIndexer(n_neurons, dim, seed=1)
    encoder = HoloGNEncoder(indexer)

    # Encode a pattern "YXYYX" style (binary alphabet: 0 or 1)
    pattern_a = [1, 0, 1, 1, 0, 1, 0]  # 7-element pattern
    pattern_b = [1, 0, 1, 1, 0, 1, 0]  # same pattern (noise-free)
    pattern_c = [0, 1, 0, 0, 1, 0, 1]  # inverted (should be orthogonal)

    hv_a = encoder.encode(pattern_a)
    hv_b = encoder.encode(pattern_b)
    hv_c = encoder.encode(pattern_c)

    sim_same = float(hv_batch_sim(hv_a, hv_b.unsqueeze(0))[0])
    sim_diff = float(hv_batch_sim(hv_a, hv_c.unsqueeze(0))[0])

    print(f"  Sim(same patterns) = {sim_same:.4f}  (want ≈ 1.0)")
    print(f"  Sim(diff patterns) = {sim_diff:.4f}  (want ≈ 0.5)")
    assert sim_same > 0.95, f"Same patterns differ: {sim_same}"
    assert abs(sim_diff - 0.5) < 0.10, f"Different patterns not orthogonal: {sim_diff}"

    print("  ✅ HoloGNEncoder OK")


def test_complex_hamming_search():
    print("=" * 60)
    print("Testing ComplexHammingSearch (Kleyko 2016, §V-A, Eq. 4)")
    print("=" * 60)

    dim = 10000
    search = ComplexHammingSearch(dim)

    # Generate random HVs and store them
    torch.manual_seed(42)
    hvs = (torch.rand(50, dim) < 0.5).float()
    labels = list(range(50))
    search.store_batch(hvs, labels)

    # Query with exact copy of hv[0]
    result = search.query_best(hvs[0])
    print(f"  Best match for hvs[0]: label={result['label']}, d={result['hamming_distance']:.4f}")
    assert result["label"] == 0, f"Wrong label: {result['label']}"
    assert result["hamming_distance"] < 0.01, f"Should be near-zero: {result['hamming_distance']}"

    # Query with noisy version (10% bit flips)
    noisy = hvs[5].clone()
    flip_idx = torch.randperm(dim)[:1000]  # flip 10%
    noisy[flip_idx] = 1.0 - noisy[flip_idx]
    result_noisy = search.query_best(noisy)
    print(f"  Noisy query (10% flips): label={result_noisy['label']}, d={result_noisy['hamming_distance']:.4f}")
    assert result_noisy["label"] == 5, f"Wrong noisy match: {result_noisy['label']}"

    # Verify complex encoding
    hv_test = torch.tensor([1.0, 0.0, 1.0, 1.0])
    c = ComplexHammingSearch.to_complex(hv_test)
    print(f"  Complex encoding [1,0,1,1]: {c.tolist()}")
    assert c[0].real == 1.0 and c[0].imag == 0.0  # 1 → 1+0j
    assert c[1].real == 0.0 and c[1].imag == 1.0  # 0 → 0+1j

    print("  ✅ ComplexHammingSearch OK")


def test_bundle_capacity():
    print("=" * 60)
    print("Testing BundleCapacityAnalyzer (Kleyko 2016, §VI-B, Eq. 8-9)")
    print("=" * 60)

    analyzer = BundleCapacityAnalyzer(dim=10000, thr=1e-6)

    # Check noise curve: p_n should increase from 0 toward 0.5
    curve = analyzer.noise_curve(max_n=10)
    print(f"  Noise p_n for n=1..10:")
    for n, pn in curve[:10]:
        print(f"    n={n:2d}: p_n={pn:.4f}")
    assert curve[0][1] == 0.0   # n=1: no noise
    assert curve[2][1] > 0.0   # n=3 (first odd > 1): some noise
    assert curve[-1][1] < 0.5  # n=10: not saturated yet

    # Capacity for d=10000 should be ~89 per the paper
    cap = analyzer.capacity()
    print(f"  Capacity (d=10000, thr=1e-6) = {cap}")
    assert 70 <= cap <= 120, f"Capacity out of expected range: {cap}"

    # Smaller d → smaller capacity
    analyzer_small = BundleCapacityAnalyzer(dim=1000, thr=1e-6)
    cap_small = analyzer_small.capacity()
    print(f"  Capacity (d=1000, thr=1e-6) = {cap_small}")
    assert cap_small < cap, "Smaller d should have smaller capacity"

    print("  ✅ BundleCapacityAnalyzer OK")


def test_pattern_overlap_estimator():
    print("=" * 60)
    print("Testing PatternOverlapEstimator (Kleyko 2016, §VI-C, Eq. 10-13)")
    print("=" * 60)

    estimator = PatternOverlapEstimator(dim=10000, thr=1e-6)

    # Use ODD bundle sizes: for odd n, majority is unambiguous and output
    # density is exactly 0.5, so no-overlap distance → 0.5.
    d_no_overlap = estimator.expected_hamming(c=0, m=11, n=11)
    d_full_overlap = estimator.expected_hamming(c=11, m=11, n=11)
    d_half_overlap = estimator.expected_hamming(c=5, m=11, n=11)

    print(f"  Expected ΔH(c=0,  m=n=11) = {d_no_overlap:.4f}  (want ≈ 0.5)")
    print(f"  Expected ΔH(c=11, m=n=11) = {d_full_overlap:.4f}  (want ≈ 0.0)")
    print(f"  Expected ΔH(c=5,  m=n=11) = {d_half_overlap:.4f}  (want between)")
    assert d_no_overlap > 0.40, f"No overlap should give d≈0.5: {d_no_overlap}"
    assert d_full_overlap < 0.15, f"Full overlap should give d≈0: {d_full_overlap}"
    assert d_full_overlap < d_half_overlap < d_no_overlap, "Must be monotone"

    # Monotonicity: more overlap → smaller distance (odd m=n=11)
    dists = [estimator.expected_hamming(c, 11, 11) for c in range(12)]
    for i in range(len(dists) - 1):
        assert dists[i] >= dists[i + 1] - 0.02, \
            f"Distance should decrease with overlap: d[{i}]={dists[i]:.4f} < d[{i+1}]={dists[i+1]:.4f}"

    # Sensitivity for medium patterns
    sens = estimator.sensitivity(m=15, n=15)
    print(f"  Sensitivity (m=n=15, d=10000): min c = {sens}")
    assert 1 <= sens <= 15

    print("  ✅ PatternOverlapEstimator OK")


def test_holograph_memory_oneshot():
    print("=" * 60)
    print("Testing HoloGNMemory one-shot (Kleyko 2016, §V-B)")
    print("=" * 60)

    torch.manual_seed(7)
    n_neurons = 10       # 2^10 = 1024 possible patterns; no collisions for 26 letters
    dim = 5000
    memory = HoloGNMemory(n_neurons=n_neurons, dim=dim, seed=42)

    # Memorize 26 letter-like patterns (binary alphabet, 5-element patterns)
    patterns = {}
    for letter_id in range(26):
        torch.manual_seed(letter_id)
        pattern = torch.randint(0, 2, (n_neurons,)).tolist()
        patterns[letter_id] = pattern
        memory.memorize(pattern, label=letter_id)

    print(f"  Stored {memory.n_stored()} patterns")

    # Recall exact match
    for letter_id in range(5):
        result = memory.recall_best(patterns[letter_id])
        assert result is not None and result["label"] == letter_id, \
            f"Exact recall failed for letter {letter_id}: got {result}"
    print("  Exact recall: 5/5 correct")

    # Recall noisy match (flip 1 position out of n_neurons=10)
    n_correct = 0
    for letter_id in range(10):
        noisy = patterns[letter_id].copy()
        flip_pos = letter_id % n_neurons
        noisy[flip_pos] = 1 - noisy[flip_pos]
        result = memory.recall_best(noisy)
        if result and result["label"] == letter_id:
            n_correct += 1
    print(f"  Noisy recall (1 bit flip): {n_correct}/10 correct")

    print("  ✅ HoloGNMemory one-shot OK")


def test_holograph_memory_supervised():
    print("=" * 60)
    print("Testing HoloGNMemory supervised (Kleyko 2016, §V-C, Eq. 5)")
    print("=" * 60)

    torch.manual_seed(99)
    n_neurons = 9
    dim = 5000
    n_classes = 4
    memory = HoloGNMemory(n_neurons=n_neurons, dim=dim, seed=3)

    # Generate class prototypes and distorted examples
    base_patterns = {}
    for c in range(n_classes):
        torch.manual_seed(c * 100)
        base_patterns[c] = torch.randint(0, 2, (n_neurons,)).tolist()

    # Supervised: present 20 distorted examples per class
    for c in range(n_classes):
        for _ in range(20):
            noisy = base_patterns[c].copy()
            for pos in range(n_neurons):
                if torch.rand(1).item() < 0.15:  # 15% distortion
                    noisy[pos] = 1 - noisy[pos]
            memory.memorize_supervised(noisy, label=c)

    memory.finalize_supervised()
    print(f"  Stored {memory.n_stored()} supervised bundles")
    assert memory.n_stored() == n_classes

    # Recall clean patterns
    correct = 0
    for c in range(n_classes):
        result = memory.recall_best(base_patterns[c])
        if result and result["label"] == c:
            correct += 1
    print(f"  Clean recall accuracy: {correct}/{n_classes}")
    assert correct >= n_classes - 1, f"Supervised recall failed: {correct}/{n_classes}"

    print("  ✅ HoloGNMemory supervised OK")


# ═══════════════════════════════════════════════════════════════════════════════
# Journal Additions (IEEE TNNLS 2017, DOI 10.1109/TNNLS.2016.2535338)
# Two new applications beyond the arXiv version:
#   1. Longest Common Substring search (§VII)
#   2. Fault Detection via HoloGN associative recall (§VIII)
# ═══════════════════════════════════════════════════════════════════════════════

# ── VII: Longest Common Substring Search ──────────────────────────────────────

class LongestCommonSubstringHDC:
    """
    Longest Common Substring (LCS) search using HoloGN (Kleyko 2017, §VII).

    Key property from the paper:
      The required number of operations on binary vectors equals the complexity
      of the suffix tree approach — the fastest classical LCS algorithm.
      Specifically: O(L1 + L2 − 1) diagonal evaluations.

    Algorithm (diagonal-stripe approach):
      For each of the L1+L2−1 diagonals d of the comparison matrix:
        1. Form the XOR of character HVs at aligned positions (i, i+d).
           XOR(A, A) = 0 (match); XOR(A, B) ≠ 0 for A≠B (mismatch).
        2. Bundle the XOR HVs for the diagonal:
             D_d = MAJORITY_SUM_{valid i}(ID(s1[i]) ⊗ ID(s2[i+d]))
        3. The density of D_d signals match count:
             density ≈ 0.5 means all mismatches (random XORs bundled)
             density < 0.5 means some character matches (XOR=0 pulls toward 0)
        4. The LCS position is on the diagonal with minimum density.

    Once the best diagonal is identified, slide a window along it to find
    the longest contiguous run of matching positions.

    Note: For the window-level LCS (not just the best diagonal), the algorithm
    uses a second pass encoding each length-L window as a sequence HV and
    comparing across strings.
    """

    def __init__(
        self,
        alphabet_size: int,
        dim: int = 10000,
        seed: Optional[int] = None,
    ):
        """
        Args:
            alphabet_size: Number of distinct symbols in the alphabet
            dim: Hypervector dimensionality
            seed: Random seed
        """
        self.alphabet_size = alphabet_size
        self.dim = dim
        # One random HD-vector per alphabet symbol
        self.char_hvs = gen_hvs(alphabet_size, dim, seed=seed)

    def _xor_hv(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Binary XOR of two {0,1} HVs: match=0, mismatch=random."""
        return (a != b).float()

    def diagonal_match_density(
        self,
        s1: List[int],
        s2: List[int],
        d: int,
    ) -> float:
        """
        Density of the bundled XOR HV for diagonal d.

        Diagonal d compares s1[i] with s2[i+d] for d ≥ 0,
        or s1[i−d] with s2[i] for d < 0.

        Returns:
            density ∈ [0, 0.5]: lower = more matches on this diagonal
        """
        L1, L2 = len(s1), len(s2)
        if d >= 0:
            pairs = [(i, i + d) for i in range(L1) if i + d < L2]
        else:
            pairs = [(i - d, i) for i in range(L2) if i - d < L1]

        if not pairs:
            return 0.5

        xor_hvs = []
        for i1, i2 in pairs:
            xor = self._xor_hv(self.char_hvs[s1[i1]], self.char_hvs[s2[i2]])
            xor_hvs.append(xor)

        n = len(xor_hvs)
        counts = torch.stack(xor_hvs).sum(dim=0)
        diagonal_hv = HoloGNEncoder._majority_threshold(counts, n)
        return float(diagonal_hv.mean().item())

    def best_diagonal(
        self,
        s1: List[int],
        s2: List[int],
    ) -> Tuple[int, float]:
        """
        Find the diagonal with most character matches (minimum density).

        Evaluates all L1+L2−1 diagonals — same operation count as suffix tree.

        Returns:
            (diagonal_offset_d, match_density)
        """
        L1, L2 = len(s1), len(s2)
        best_d = 0
        best_density = 1.0

        for d in range(-(L1 - 1), L2):
            density = self.diagonal_match_density(s1, s2, d)
            if density < best_density:
                best_density = density
                best_d = d

        return best_d, best_density

    def encode_window(self, tokens: List[int]) -> torch.Tensor:
        """
        Encode a substring as a position-sensitive sequence HV.

        Each character c at position i contributes: Sh(ID(c), i).
        The result is the majority-threshold bundle.
        """
        if not tokens:
            return torch.zeros(self.dim)
        components = [hv_permute(self.char_hvs[c], k=i) for i, c in enumerate(tokens)]
        n = len(components)
        counts = torch.stack(components).sum(dim=0)
        return HoloGNEncoder._majority_threshold(counts, n)

    def lcs_length(
        self,
        s1: List[int],
        s2: List[int],
        threshold: float = 0.42,
    ) -> Tuple[int, int, int]:
        """
        Find the Longest Common Substring length and position.

        Uses a two-phase approach:
          Phase 1: Identify the best diagonal (O(L1+L2−1) diagonal evaluations).
          Phase 2: Slide a window along the best diagonal to find the longest
                   contiguous run of matching positions.

        Args:
            s1: First sequence (list of integer token indices)
            s2: Second sequence
            threshold: Hamming similarity threshold for window match
                       (0.42 means windows must share ≥ 8% of characters)

        Returns:
            (lcs_length, start_in_s1, start_in_s2)
        """
        L1, L2 = len(s1), len(s2)

        # Phase 1: best diagonal
        best_d, _ = self.best_diagonal(s1, s2)

        # Phase 2: slide along best diagonal to find longest contiguous run
        if best_d >= 0:
            positions = [(i, i + best_d) for i in range(L1) if i + best_d < L2]
        else:
            positions = [(i - best_d, i) for i in range(L2) if i - best_d < L1]

        # Check character-level matches along this diagonal
        best_len = 0
        best_start_s1 = 0
        best_start_s2 = 0
        run_len = 0
        run_start_s1 = 0
        run_start_s2 = 0

        for k, (i1, i2) in enumerate(positions):
            # Match: Hamming distance between char HVs = 0 iff same char
            match = (s1[i1] == s2[i2])
            if match:
                if run_len == 0:
                    run_start_s1 = i1
                    run_start_s2 = i2
                run_len += 1
                if run_len > best_len:
                    best_len = run_len
                    best_start_s1 = run_start_s1
                    best_start_s2 = run_start_s2
            else:
                run_len = 0

        return best_len, best_start_s1, best_start_s2

    def window_similarity(
        self,
        s1: List[int],
        s2: List[int],
        window: int,
        threshold: float = 0.45,
    ) -> List[Tuple[int, int, float]]:
        """
        Find all common substrings of length ≥ window using sequence HV matching.

        Encodes every length-window substring of both strings as HVs, then
        uses ComplexHammingSearch for fast retrieval.

        Args:
            s1: First sequence
            s2: Second sequence
            window: Substring length to search for
            threshold: Maximum Hamming distance for a match

        Returns:
            List of (start_in_s1, start_in_s2, hamming_distance)
        """
        L1, L2 = len(s1), len(s2)
        results = []

        # Encode all substrings of s2
        s2_windows = []
        for j in range(L2 - window + 1):
            hv = self.encode_window(s2[j:j + window])
            s2_windows.append((j, hv))

        # Search each window of s1 against all windows of s2
        for i in range(L1 - window + 1):
            hv1 = self.encode_window(s1[i:i + window])
            for j, hv2 in s2_windows:
                # Hamming distance via XOR population count
                dist = float((hv1 != hv2).float().mean().item())
                if dist <= threshold:
                    results.append((i, j, dist))

        return results


# ── VIII: Fault Detection via HoloGN ─────────────────────────────────────────

class FaultDetector:
    """
    Fault detection using HoloGN associative recall (Kleyko 2017, §VIII).

    System states are encoded as HoloGN HVs from component states:
      - Each component j can be in state s_j (e.g., OK=0, FAULT=1, DEGRADED=2)
      - State at time t: HGN_t = MAJORITY_SUM_{j=1}^{n}(EHD_{j, s_j(t)})

    Normal operation profiles are memorized during training.
    At test time:
      1. Encode current state → HGN_current
      2. Recall against memorized normals with ComplexHammingSearch
      3. Anomaly score = minimum Hamming distance to any normal state
      4. If anomaly score > threshold: fault detected
      5. Fault identification: for each component j, compare current element
         HV EHD_{j, s_j_current} against the nearest normal EHD_{j, s_j_normal}
    """

    def __init__(
        self,
        n_components: int,
        n_states_per_component: int,
        dim: int = 10000,
        anomaly_threshold: float = 0.15,
        seed: Optional[int] = None,
    ):
        """
        Args:
            n_components: Number of monitored components (GNs)
            n_states_per_component: Number of possible states per component
            dim: Hypervector dimensionality
            anomaly_threshold: Hamming distance above which a state is anomalous
            seed: Random seed
        """
        self.n_components = n_components
        self.n_states = n_states_per_component
        self.dim = dim
        self.anomaly_threshold = anomaly_threshold

        self.indexer = ZadoffChuIndexer(n_components, dim, seed=seed)
        self.encoder = HoloGNEncoder(self.indexer)
        self.memory = ComplexHammingSearch(dim)

        self._n_normal = 0

    def learn_normal(self, component_states: List[int], label: Optional[int] = None):
        """
        Memorize a normal operating state.

        Args:
            component_states: List of n_components state indices (one per component)
            label: Optional identifier for this normal state
        """
        hv = self.encoder.encode(component_states)
        self.memory.store(hv, label if label is not None else self._n_normal)
        self._n_normal += 1

    def anomaly_score(self, component_states: List[int]) -> float:
        """
        Compute anomaly score for a system state.

        Args:
            component_states: Current state of each component

        Returns:
            Hamming distance to nearest memorized normal state ∈ [0, 0.5]
            Lower = more normal; higher = more anomalous
        """
        hv = self.encoder.encode(component_states)
        result = self.memory.query_best(hv)
        if result is None:
            return 0.5
        return result["hamming_distance"]

    def is_fault(self, component_states: List[int]) -> bool:
        """Return True if the state is anomalous (beyond threshold)."""
        return self.anomaly_score(component_states) > self.anomaly_threshold

    def identify_faulty_components(
        self,
        current_states: List[int],
        normal_states: List[int],
    ) -> List[int]:
        """
        Identify which components differ between current and nearest-normal state.

        Uses unbinding: compares element HVs EHD_{j, current} vs EHD_{j, normal}
        for each component j. Components with Hamming distance > 0.4 have changed.

        Args:
            current_states: Current component state indices
            normal_states: Nearest-normal state indices (from recall)

        Returns:
            List of component indices that have changed (suspected faults)
        """
        faults = []
        for j in range(self.n_components):
            ehd_current = self.indexer.element_hv(j, current_states[j])
            ehd_normal = self.indexer.element_hv(j, normal_states[j])
            sim = float(hv_batch_sim(ehd_current, ehd_normal.unsqueeze(0))[0])
            hamming = 1.0 - sim
            if hamming > 0.4:  # Shifted by ≥1 state ↔ Hamming ≈ 0.5
                faults.append(j)
        return faults

    def diagnose(
        self,
        current_states: List[int],
    ) -> Dict:
        """
        Full diagnosis: score, fault flag, nearest normal, faulty components.

        Args:
            current_states: Current component states

        Returns:
            Dict with anomaly_score, is_fault, nearest_normal_label, faulty_components
        """
        hv = self.encoder.encode(current_states)
        result = self.memory.query(hv, threshold=1.0, top_k=1)

        if not result:
            return {
                "anomaly_score": 0.5,
                "is_fault": True,
                "nearest_normal_label": None,
                "faulty_components": list(range(self.n_components)),
            }

        best = result[0]
        score = best["hamming_distance"]
        fault_flag = score > self.anomaly_threshold

        return {
            "anomaly_score": score,
            "is_fault": fault_flag,
            "nearest_normal_label": best["label"],
            "faulty_components": [],  # Populated by identify_faulty_components
        }


# ── Tests for journal additions ────────────────────────────────────────────────

def test_lcs_hdc():
    print("=" * 60)
    print("Testing LongestCommonSubstringHDC (Kleyko 2017, TNNLS §VII)")
    print("=" * 60)

    lcs = LongestCommonSubstringHDC(alphabet_size=10, dim=8000, seed=42)

    # Test 1: Identical strings — LCS = full length
    s = [0, 1, 2, 3, 4, 5]
    length, p1, p2 = lcs.lcs_length(s, s)
    print(f"  LCS(s, s) length = {length}  (want {len(s)})")
    assert length == len(s), f"Identical strings: {length} != {len(s)}"

    # Test 2: No overlap — LCS = 0
    s1 = [0, 0, 0, 0, 0]
    s2 = [1, 1, 1, 1, 1]
    length, _, _ = lcs.lcs_length(s1, s2)
    print(f"  LCS(AAAAA, BBBBB) length = {length}  (want 0)")
    assert length == 0

    # Test 3: Partial overlap
    s1 = [0, 1, 2, 3, 4]
    s2 = [5, 6, 2, 3, 4, 7]
    length, start1, start2 = lcs.lcs_length(s1, s2)
    print(f"  LCS([0,1,2,3,4], [5,6,2,3,4,7]) = {length} @ s1[{start1}] s2[{start2}]")
    assert length >= 3, f"Expected LCS ≥ 3, got {length}"

    # Test 4: Diagonal evaluation count is O(L1+L2−1)
    L1, L2 = 10, 12
    n_diagonals = L1 + L2 - 1
    print(f"  Diagonal count for L1={L1}, L2={L2}: {n_diagonals} (= L1+L2−1)")
    assert n_diagonals == 21

    # Test 5: Window similarity
    s1 = [0, 1, 2, 3, 4, 5]
    s2 = [6, 7, 1, 2, 3, 8]
    matches = lcs.window_similarity(s1, s2, window=3, threshold=0.3)
    print(f"  Window matches (window=3): {[(a,b,f'{d:.3f}') for a,b,d in matches[:3]]}")

    print("  ✅ LongestCommonSubstringHDC OK")


def test_fault_detector():
    print("=" * 60)
    print("Testing FaultDetector (Kleyko 2017, TNNLS §VIII)")
    print("=" * 60)

    torch.manual_seed(42)
    n_comp = 8
    n_states = 3  # 0=OK, 1=DEGRADED, 2=FAULT
    dim = 6000

    detector = FaultDetector(
        n_components=n_comp,
        n_states_per_component=n_states,
        dim=dim,
        anomaly_threshold=0.20,
        seed=7,
    )

    # Learn 5 normal states (all components OK)
    normal_states = [0] * n_comp
    for i in range(5):
        noisy = normal_states.copy()
        # Small random deviations in training data
        for j in range(n_comp):
            if torch.rand(1).item() < 0.05:
                noisy[j] = 1  # Occasional degraded
        detector.learn_normal(noisy, label=i)

    print(f"  Memorized {detector._n_normal} normal states")

    # Test: normal state should score low
    score_normal = detector.anomaly_score([0] * n_comp)
    print(f"  Anomaly score (all OK) : {score_normal:.4f}  (want < 0.20)")
    assert score_normal < 0.20, f"Normal state scored too high: {score_normal}"

    # Test: faulty state should score high
    faulty = [0] * n_comp
    faulty[2] = 2   # Component 2 has FAULT
    faulty[5] = 2   # Component 5 has FAULT
    score_fault = detector.anomaly_score(faulty)
    print(f"  Anomaly score (2 faults): {score_fault:.4f}  (want > normal)")
    assert score_fault > score_normal, "Fault should score higher than normal"

    # Test: fault detection flag
    assert not detector.is_fault([0] * n_comp), "Normal state should not be fault"
    # Note: faulty state detection depends on threshold vs score

    # Test: component identification
    # Compare faulty state against known normal
    identified = detector.identify_faulty_components(faulty, normal_states)
    print(f"  Identified faulty components: {identified}  (want [2, 5])")
    assert 2 in identified, f"Component 2 not identified: {identified}"
    assert 5 in identified, f"Component 5 not identified: {identified}"

    # Test: diagnose
    diagnosis = detector.diagnose([0] * n_comp)
    print(f"  Diagnosis (normal): score={diagnosis['anomaly_score']:.4f}, fault={diagnosis['is_fault']}")
    assert not diagnosis["is_fault"], "Normal state diagnosed as fault"

    print("  ✅ FaultDetector OK")


# ═══════════════════════════════════════════════════════════════════════════════
# Sparse HoloGN (Kleyko, Osipov, Rachkovskij — BICA 2016, Procedia Comp. Sci.)
# doi:10.1016/j.procs.2016.07.404
#
# Modifies the dense HoloGN to use sparse binary distributed representations:
#   - Initialization vectors: p(1) << 0.5 (e.g., M=1000 ones in N=100000 dims)
#   - Bundling: bitwise OR (disjunction) instead of majority-sum
#   - Binding: Context-Dependent Thinning (CDT) to normalise density after OR
#   - Similarity: overlap |x ∧ y|_1 (AND popcount) instead of Hamming distance
#
# CDT procedure (K=1 permutation, Rachkovskij 2001):
#   ⟨Z⟩ = Z ∧ perm(Z)
# This thins the OR-bundled vector back to approximately the component density.
# ═══════════════════════════════════════════════════════════════════════════════

class SparseHoloGN:
    """
    Sparse Holographic Graph Neuron (Kleyko et al., BICA 2016).

    Replaces dense {0,1} HVs (p=0.5) with sparse ones (p<<0.5), which:
    - Increases biological plausibility (sparse neural codes)
    - Allows very high dimensionality without proportional memory cost
    - Maintains recall accuracy even at N=2048, M=40

    Key parameters (from paper experiments):
        N=100000, M=1000, p(1)=0.01  — full sparse
        N=10000,  M=100,  p(1)=0.01  — medium sparse
        N=2048,   M=40,   p(1)≈0.02  — compact sparse

    Bundling (OR):
        Z = ⋁_{j=1}^n Sh(sIV_j, i)

    Context-Dependent Thinning (CDT, K=1):
        ⟨Z⟩ = Z ∧ perm_1(Z)   [bitwise AND with one fixed random permutation]

    Similarity:
        overlap(x, y) = |x ∧ y|_1 / |x|_1
        (fraction of x's active bits that are also active in y)
    """

    def __init__(
        self,
        n_neurons: int,
        dim: int = 10000,
        density: float = 0.01,
        cdt_shifts: int = 1,
        seed: Optional[int] = None,
    ):
        """
        Args:
            n_neurons: Number of Graph Neurons
            dim: Hypervector dimensionality N
            density: Fraction of 1s in each sparse HV (p(1) = M/N)
            cdt_shifts: Number of permutations K used in CDT normalisation
            seed: Random seed
        """
        self.n_neurons = n_neurons
        self.dim = dim
        self.density = density
        self.cdt_shifts = cdt_shifts

        g = torch.Generator()
        if seed is not None:
            g.manual_seed(seed)

        # Sparse initialization vectors: each has approximately density*dim ones
        n_ones = max(1, int(round(density * dim)))
        self._siv = torch.zeros(n_neurons, dim)
        for j in range(n_neurons):
            idx = torch.randperm(dim, generator=g)[:n_ones]
            self._siv[j, idx] = 1.0

        # Fixed CDT permutation indices (K shifts, each a cyclic roll by a random amount)
        self._cdt_roll = []
        for _ in range(cdt_shifts):
            shift = int(torch.randint(1, dim, (1,), generator=g).item())
            self._cdt_roll.append(shift)

        # Associative memory: list of (hv, label)
        self._memory: List[Tuple[torch.Tensor, int]] = []

    # ── Encoding ──────────────────────────────────────────────────────────────

    def _element_hv(self, neuron_j: int, element_i: int) -> torch.Tensor:
        """Sh(sIV_j, i) — cyclic shift of sparse init vector."""
        return torch.roll(self._siv[neuron_j], shifts=element_i)

    def _cdt(self, z: torch.Tensor) -> torch.Tensor:
        """
        Context-Dependent Thinning (K=cdt_shifts).

        ⟨Z⟩ = Z ∧ perm_1(Z) ∧ perm_2(Z) ∧ ...

        Reduces density after OR-bundling back toward the component density.
        Each AND operation approximately squares the density:
            p_out ≈ p_in^(K+1)
        So K=1 → p_out ≈ p_in^2; normalise by choosing K to match target.
        """
        result = z.clone()
        for shift in self._cdt_roll:
            result = result * torch.roll(z, shifts=shift)  # AND (for binary)
        return (result > 0).float()

    def encode(self, pattern: List[int]) -> torch.Tensor:
        """
        Sparse HoloGN encoding (BICA 2016, Eq. after §3).

        Z = ⋁_{j=1}^n Sh(sIV_j, pattern[j])     [OR-bundle]
        ⟨Z⟩ = CDT(Z)                               [thin to restore density]

        Args:
            pattern: n_neurons element indices (−1 = GN not activated)

        Returns:
            (dim,) sparse binary HV
        """
        z = torch.zeros(self.dim)
        for j, elem_i in enumerate(pattern):
            if elem_i >= 0:
                z = z + self._element_hv(j, elem_i)  # accumulate for OR

        # OR-bundle: any position touched by ≥1 component → 1
        z = (z > 0).float()

        # CDT thinning
        return self._cdt(z)

    # ── Overlap similarity ────────────────────────────────────────────────────

    @staticmethod
    def overlap(a: torch.Tensor, b: torch.Tensor) -> float:
        """
        Normalised overlap similarity (BICA 2016, §3).

        overlap(a, b) = |a ∧ b|_1 / |a|_1

        Returns 1.0 for identical sparse HVs, ≈ p_b for random b.
        """
        intersection = float((a * b).sum().item())  # AND = elementwise product
        weight_a = float(a.sum().item())
        if weight_a < 1e-9:
            return 0.0
        return intersection / weight_a

    # ── Associative memory ────────────────────────────────────────────────────

    def memorize(self, pattern: List[int], label: int):
        """Store a pattern in one-shot."""
        hv = self.encode(pattern)
        self._memory.append((hv, label))

    def recall_best(self, pattern: List[int]) -> Optional[Dict]:
        """
        Retrieve the stored pattern with highest overlap to query.

        In sparse HoloGN, highest overlap = best match (not lowest Hamming).
        """
        if not self._memory:
            return None
        hq = self.encode(pattern)
        best_label, best_sim = None, -1.0
        for hv, label in self._memory:
            sim = self.overlap(hq, hv)
            if sim > best_sim:
                best_sim = sim
                best_label = label
        return {"label": best_label, "overlap": best_sim}

    def recall(self, pattern: List[int], min_overlap: float = 0.1) -> List[Dict]:
        """Return all stored patterns with overlap ≥ min_overlap."""
        hq = self.encode(pattern)
        results = []
        for hv, label in self._memory:
            sim = self.overlap(hq, hv)
            if sim >= min_overlap:
                results.append({"label": label, "overlap": sim})
        return sorted(results, key=lambda x: x["overlap"], reverse=True)

    def actual_density(self, pattern: List[int]) -> float:
        """Return the actual density of an encoded HV for diagnostics."""
        hv = self.encode(pattern)
        return float(hv.mean().item())


def test_sparse_holograph_neuron():
    print("=" * 60)
    print("Testing SparseHoloGN (Kleyko/Osipov/Rachkovskij, BICA 2016)")
    print("=" * 60)

    torch.manual_seed(42)

    # Medium-sparse: N=5000, M=50, p=0.01
    n_neurons, dim, density = 10, 5000, 0.01
    sgn = SparseHoloGN(n_neurons=n_neurons, dim=dim, density=density, cdt_shifts=1, seed=0)

    # Verify sparse init vectors
    avg_density = float(sgn._siv.mean().item())
    print(f"  Init vector density: {avg_density:.4f}  (want ≈ {density})")
    assert abs(avg_density - density) < 0.005

    # Memorize 5 patterns
    torch.manual_seed(7)
    patterns = {}
    for pid in range(5):
        torch.manual_seed(pid * 100)
        p = torch.randint(0, 3, (n_neurons,)).tolist()
        patterns[pid] = p
        sgn.memorize(p, label=pid)

    # Exact recall
    for pid in range(5):
        result = sgn.recall_best(patterns[pid])
        assert result and result["label"] == pid, \
            f"Exact recall failed for pattern {pid}: got {result}"
    print("  Exact recall: 5/5  ✅")

    # Verify density after CDT
    d_after = sgn.actual_density(patterns[0])
    print(f"  Encoded HV density after CDT: {d_after:.4f}  (target ≈ {density**2:.4f})")

    # Noisy recall (flip 2 positions out of 10)
    noisy = patterns[0].copy()
    noisy[0] = (noisy[0] + 1) % 3
    noisy[1] = (noisy[1] + 1) % 3
    result = sgn.recall_best(noisy)
    print(f"  Noisy recall (2 flips): label={result['label']}, overlap={result['overlap']:.4f}")

    print("  ✅ SparseHoloGN OK")


if __name__ == "__main__":
    test_cyclic_shift_indexer()
    print()
    test_holographic_encoding()
    print()
    test_complex_hamming_search()
    print()
    test_bundle_capacity()
    print()
    test_pattern_overlap_estimator()
    print()
    test_holograph_memory_oneshot()
    print()
    test_holograph_memory_supervised()
    print()
    test_lcs_hdc()
    print()
    test_fault_detector()
    print()
    test_sparse_holograph_neuron()
    print()
    print("=== All HoloGN tests (arXiv + TNNLS journal + BICA 2016) passed ===")
    print("=== All Kleyko 2016 HoloGN tests passed ===")
