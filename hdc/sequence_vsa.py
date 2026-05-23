"""
Sequence VSA: Cambridge Test, Recursive Binding, Modular Composite Representations
====================================================================================
Three complementary improvements for sequence representation quality and
memory efficiency, grounded in Kleyko's research program:

1. **CambridgeTestVSA** — Recognizing permuted words with VSA.
   Kleyko, Osipov, Gayler (2016) "Recognizing permuted words with vector
   symbolic architectures: A Cambridge test for machines." Procedia Computer
   Science — Paper IV in Kleyko's 2016 licentiate thesis (diva2:990444).

   The Cambridge Effect: humans can read scrambled text as long as first and
   last letters are correct ("Aoccdrnig to a rscheearch..."). This paper tests
   whether VSA can do the same — and why it can: position-bound character HVs
   contribute most to the word HV based on their frequency, so outer letters
   (fixed) dominate the Hamming similarity.

2. **RecursiveBindingEncoder** — Similarity-preserving FHRR sequence HVs.
   Rachkovskij & Kleyko (2022) "Recursive Binding for Similarity-Preserving
   Hypervector Representations of Sequences." arXiv:2201.11691.

   Standard n-gram binding loses position similarity information: symbols at
   nearby positions should produce similar HVs, but naive binding treats each
   position as independent. Recursive binding fixes this:
     a_i = ea ⊙ (pos^i + pos^{i+1} + ... + pos^{i+R-1})  [Eq. 2]
   where pos is a unit-phasor HV, R is the similarity radius, and ⊙ is
   FHRR complex binding. The overlap between a_i and a_{i+j} decays
   gracefully: sim = (R - |j|) / R for |j| < R, 0 otherwise.

   Achieves: 0.84 Pearson correlation with human word priming experiments
   (vs 0.73 for spatial coding, 0.75 for kernel UOB).

3. **MCRBackend** — Modular Composite Representation.
   Angioli, Kymn, Rosato, Loufi, Olivieri, Kleyko (2025) "Efficient
   Hyperdimensional Computing with Modular Composite Representations."
   arXiv:2511.09708.

   MCR uses integer HVs with modular arithmetic (modulus r):
     bind(h, u):   c_i = (h_i + u_i) mod r    [Eq. from §II]
     unbind(h, u): c_i = (h_i - u_i) mod r
     similarity:   δ = Σ min((h-u)%r, (u-h)%r) / D  [Eq. 1]
     bundle:       v_i = Σ exp(2πj/r × h_i^(k))  →  round to [0,r)  [§II]

   Key result: MCR-3 (r=8, 3 bits/dim) matches BSC accuracy at 4× less
   memory and 3× faster inference. MCR-4 (r=16) matches FHRR at 4× less
   memory than float32.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from hdc.hdc_glue import hv_batch_sim, gen_hvs, hv_permute, hv_majority, hv_xor


# ═══════════════════════════════════════════════════════════════════════════════
# 1. CambridgeTestVSA — Kleyko, Osipov, Gayler (2016)
# ═══════════════════════════════════════════════════════════════════════════════

class CambridgeTestVSA:
    """
    Recognize permuted words using Binary Spatter Codes (BSC).

    Kleyko, Osipov, Gayler (2016): Paper IV of the 2016 licentiate thesis
    (diva2:990444, page 15+ of Part I summary).

    Word encoding:
        word_hv = MAJORITY_SUM_{i=0}^{L-1}( char_hv[word[i]] ⊗ pos_hv[i] )
      where:
        char_hv[c]: random HV for character c
        pos_hv[i]:  cyclic shift Sh(IV, i) — same Zadoff-Chu approach as HoloGN
        ⊗:          XOR binding (BSC)

    Cambridge scrambling:
        Fixed first and last letter, middle letters randomly permuted.
        Prediction: sim(word_hv, scrambled_hv) > threshold → recognized.

    Why it works (from the paper):
        The word HV is a majority-sum of L position-bound HVs. After bundling,
        the first and last position contributions dominate in the sense that
        they are unique (no other word has the same first/last character), while
        middle characters have higher confusability. This statistical argument
        holds in BSC because of the concentration-of-measure property.

    Edge effect (db option):
        Doubling the first and last character HVs explicitly weights them more,
        significantly improving recognition:
            word_hv = MAJORITY_SUM( 2×char[0]⊗pos[0], char[1]⊗pos[1], ...,
                                    char[L-2]⊗pos[L-2], 2×char[L-1]⊗pos[L-1] )

    Args:
        alphabet: Set of characters to encode
        hd_dim: Hypervector dimensionality
        double_boundary: If True, weight first/last characters twice (db option)
        recognition_threshold: Hamming similarity for a positive recognition
        seed: Random seed
    """

    def __init__(
        self,
        alphabet: str = "abcdefghijklmnopqrstuvwxyz",
        hd_dim: int = 10000,
        double_boundary: bool = True,
        recognition_threshold: float = 0.65,
        seed: int = 42,
    ):
        self.hd_dim = hd_dim
        self.double_boundary = double_boundary
        self.recognition_threshold = recognition_threshold

        # Character HVs (one per alphabet symbol)
        g = torch.Generator()
        g.manual_seed(seed)
        char_hvs = {}
        for i, c in enumerate(alphabet):
            char_hvs[c] = (torch.rand(hd_dim, generator=g) < 0.5).float()
        # Add space
        char_hvs[' '] = (torch.rand(hd_dim, generator=g) < 0.5).float()
        self._char_hvs = char_hvs

        # Position HVs: IV shifted by i (Zadoff-Chu approach)
        self._iv = (torch.rand(hd_dim, generator=g) < 0.5).float()

    def _pos_hv(self, i: int) -> torch.Tensor:
        """Position i HV = Sh(IV, i)."""
        return hv_permute(self._iv, k=i)

    def _char_hv(self, c: str) -> torch.Tensor:
        """Character HV — returns random HV for unseen characters."""
        if c not in self._char_hvs:
            seed = hash(c) & 0x7FFFFFFF
            g = torch.Generator(); g.manual_seed(seed)
            self._char_hvs[c] = (torch.rand(self.hd_dim, generator=g) < 0.5).float()
        return self._char_hvs[c]

    def encode_word(self, word: str) -> torch.Tensor:
        """
        Encode a word as a BSC HV (Eq. 3 from Kleyko 2016).

        Each character is XOR-bound with its position HV, then all are bundled.
        With double_boundary=True, first and last characters are doubled.

        Args:
            word: String to encode

        Returns:
            (hd_dim,) binary word HV
        """
        L = len(word)
        if L == 0:
            return torch.zeros(self.hd_dim)

        components = []
        for i, c in enumerate(word.lower()):
            chv = self._char_hv(c)
            phv = self._pos_hv(i)
            bound = hv_xor(chv, phv)

            # Edge effect: double first and last letter weight
            if self.double_boundary and (i == 0 or i == L - 1):
                components.append(bound)
                components.append(bound)  # double contribution
            else:
                components.append(bound)

        stacked = torch.stack(components)
        return hv_majority(stacked.float().mean(dim=0))

    def scramble(self, word: str, seed: Optional[int] = None) -> str:
        """
        Cambridge-scramble a word: fix first and last, shuffle middle.

        Args:
            word: Original word
            seed: Random seed for reproducible scrambling

        Returns:
            Scrambled string
        """
        if len(word) <= 3:
            return word   # too short to scramble meaningfully

        middle = list(word[1:-1])
        if seed is not None:
            random.seed(seed)
        random.shuffle(middle)
        return word[0] + ''.join(middle) + word[-1]

    def recognize(self, query: str, target: str) -> Tuple[bool, float]:
        """
        Test whether query is recognized as target (or vice versa).

        Args:
            query: String to test (may be scrambled)
            target: Reference word

        Returns:
            (recognized: bool, similarity: float)
        """
        hv_query  = self.encode_word(query)
        hv_target = self.encode_word(target)
        sim = float(hv_batch_sim(hv_query, hv_target.unsqueeze(0))[0])
        return sim >= self.recognition_threshold, sim

    def cambridge_test(
        self,
        words: List[str],
        n_trials: int = 10,
    ) -> Dict:
        """
        Run the Cambridge test on a list of words.

        For each word: scramble n_trials times, measure recognition rate.
        Compare to control: recognition of scrambled word vs. a different word.

        Returns:
            Dict with recognition_rate, false_positive_rate, accuracy
        """
        correct = 0
        false_positives = 0
        total = 0

        for word in words:
            if len(word) < 3:
                continue
            hv_target = self.encode_word(word)

            for trial in range(n_trials):
                scrambled = self.scramble(word, seed=trial * 1000 + len(word))
                hv_scrambled = self.encode_word(scrambled)
                sim = float(hv_batch_sim(hv_scrambled, hv_target.unsqueeze(0))[0])

                if sim >= self.recognition_threshold:
                    correct += 1
                total += 1

                # False positive: scrambled vs wrong word
                wrong_word = words[(words.index(word) + 1) % len(words)]
                hv_wrong = self.encode_word(wrong_word)
                sim_wrong = float(hv_batch_sim(hv_scrambled, hv_wrong.unsqueeze(0))[0])
                if sim_wrong >= self.recognition_threshold:
                    false_positives += 1

        recognition_rate = correct / max(total, 1)
        fp_rate = false_positives / max(total, 1)

        return {
            "recognition_rate": recognition_rate,
            "false_positive_rate": fp_rate,
            "accuracy": recognition_rate - fp_rate,
            "n_words": len(words),
            "n_trials": n_trials,
            "double_boundary": self.double_boundary,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 2. RecursiveBindingEncoder — Rachkovskij & Kleyko (2022)
# ═══════════════════════════════════════════════════════════════════════════════

class RecursiveBindingEncoder:
    """
    Similarity-preserving sequence HVs via recursive FHRR binding.

    Rachkovskij & Kleyko (2022) arXiv:2201.11691.

    Uses Fourier Holographic Reduced Representations (FHRR):
      - HVs are complex-valued: each component is exp(iθ), θ ∈ [0, 2π)
      - Binding ⊙: component-wise complex multiplication (phase addition)
      - Superposition +: component-wise complex addition

    Position HV with similarity radius R (Eq. 2):
        a_i = ea ⊙ (pos^i + pos^{i+1} + ... + pos^{i+R-1})
      where ea is the atomic complex HV for symbol a, and pos is a fixed
      unit-phasor HV (pos_j = exp(iφ_j) for random phases φ_j).

    Similarity property (key result):
        sim(a_i, a_{i+j}) = (R - |j|) / R   for |j| < R
                           = 0               for |j| ≥ R
    This gives GRADED positional similarity — nearby positions are similar,
    far positions are orthogonal. Exactly what the Cambridge Effect requires.

    Equivariance: shifting the whole sequence by j shifts all HV positions by j.
      Tj(F(a_i)) = pos^j ⊙ a_i = a_{i+j} = F(a_{i+j}) = F(Sj(a_i))   [Eq. 3]

    Args:
        alphabet_size: Number of distinct symbols
        hd_dim: HV dimensionality D
        similarity_radius: R — controls position similarity range
        seed: Random seed
    """

    def __init__(
        self,
        alphabet_size: int = 26,
        hd_dim: int = 10000,
        similarity_radius: int = 2,
        seed: int = 42,
    ):
        self.hd_dim = hd_dim
        self.R = similarity_radius

        g = torch.Generator()
        g.manual_seed(seed)

        # Atomic complex HVs for each symbol: ea_j = exp(i * θ_j)
        phases = torch.rand(alphabet_size, hd_dim, generator=g) * 2 * math.pi
        self._ea = torch.exp(1j * phases)                # (alphabet_size, D)

        # Position phasor: pos_j = exp(i * φ_j)
        pos_phases = torch.rand(hd_dim, generator=g) * 2 * math.pi
        self._pos = torch.exp(1j * pos_phases)            # (D,)

        # Precompute pos^k for k = 0 .. max_len + R
        self._max_pos = 200
        pos_powers = [torch.ones(hd_dim, dtype=torch.complex64)]
        for k in range(1, self._max_pos + self.R + 1):
            pos_powers.append(pos_powers[-1] * self._pos)
        self._pos_powers = pos_powers

    def _position_hv(self, i: int) -> torch.complex64:
        """
        Position HV for position i with similarity radius R (Eq. 2).

        = pos^i + pos^{i+1} + ... + pos^{i+R-1}
        """
        result = sum(self._pos_powers[i + k] for k in range(self.R))
        return result

    def encode_symbol(self, symbol_idx: int, position: int) -> torch.Tensor:
        """
        Encode symbol at position i: a_i = ea ⊙ position_hv(i).

        Returns complex (D,) HV.
        """
        ea = self._ea[symbol_idx]           # (D,) complex
        phv = self._position_hv(position)   # (D,) complex
        return ea * phv                     # element-wise complex multiply (FHRR bind)

    def encode_sequence(
        self,
        sequence: List[int],
        double_boundary: bool = False,
    ) -> torch.Tensor:
        """
        Encode a sequence of symbol indices as a composite HV.

        word_hv = Σ_i a_i  (complex superposition, then binarize for BSC compat)

        Args:
            sequence: List of integer symbol indices
            double_boundary: If True, weight first/last symbols twice

        Returns:
            (D,) real-valued HV (cosine-similarity compatible)
        """
        if not sequence:
            return torch.zeros(self.hd_dim)

        L = len(sequence)
        composite = torch.zeros(self.hd_dim, dtype=torch.complex64)

        for i, sym in enumerate(sequence):
            hv_i = self.encode_symbol(sym, i)
            weight = 2.0 if double_boundary and (i == 0 or i == L - 1) else 1.0
            composite += weight * hv_i

        # Return real part (cosine similarity works on real part of FHRR)
        return composite.real

    def similarity(self, seq_a: List[int], seq_b: List[int]) -> float:
        """
        Cosine similarity between two encoded sequences.

        Uses the real-valued composite HVs.
        """
        hv_a = self.encode_sequence(seq_a)
        hv_b = self.encode_sequence(seq_b)
        return float(F.cosine_similarity(hv_a.unsqueeze(0), hv_b.unsqueeze(0)).item())

    def position_similarity(self, pos_a: int, pos_b: int) -> float:
        """
        Theoretical similarity between positions a and b.

        sim(a_i, a_{i+j}) = (R - |j|) / R for |j| < R, else 0.
        (Same symbol assumed for comparison.)
        """
        j = abs(pos_a - pos_b)
        if j < self.R:
            return (self.R - j) / self.R
        return 0.0

    def cambridge_test_fhrr(
        self,
        words: List[str],
        char_to_idx: Dict[str, int],
        n_trials: int = 10,
    ) -> Dict:
        """
        Cambridge test using FHRR recursive binding.

        Args:
            words: List of test words
            char_to_idx: Mapping character → symbol index
            n_trials: Scrambling trials per word

        Returns:
            Dict with Pearson correlation and recognition metrics
        """
        results = []
        for word in words:
            if len(word) < 4:
                continue
            seq = [char_to_idx.get(c, 0) for c in word.lower()]
            hv_orig = self.encode_sequence(seq, double_boundary=True)

            # Scramble middle
            for trial in range(n_trials):
                random.seed(trial)
                middle = seq[1:-1]
                random.shuffle(middle)
                scrambled = [seq[0]] + middle + [seq[-1]]
                hv_scr = self.encode_sequence(scrambled, double_boundary=True)
                sim = float(F.cosine_similarity(hv_orig.unsqueeze(0), hv_scr.unsqueeze(0)))
                results.append({"word": word, "trial": trial, "sim": sim})

        mean_sim = sum(r["sim"] for r in results) / max(len(results), 1)
        return {
            "n_tests": len(results),
            "mean_similarity": mean_sim,
            "recognized": sum(1 for r in results if r["sim"] > 0.5) / max(len(results), 1),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 3. MCRBackend — Angioli, Kymn, Rosato, Loufi, Olivieri, Kleyko (2025)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MCRConfig:
    """Configuration for Modular Composite Representation."""
    dim: int = 2500          # HV dimensionality (4× smaller than BSC at D=10000)
    modulus: int = 8         # r — number of discrete values (MCR-3: r=8, 3 bits/dim)
    seed: Optional[int] = 42


class MCRVector:
    """
    Single Modular Composite Representation hypervector.

    Integer values in [0, r) with modular arithmetic (Angioli et al. 2025).

    Operations (from §II of the paper):
      bind(h, u):   c_i = (h_i + u_i) mod r        [modular sum]
      unbind(h, u): c_i = (h_i - u_i) mod r        [modular subtraction]
      similarity:   δ(h,u) = Σ min((h-u)%r, (u-h)%r) / D   [Eq. 1]
      bundle:       Project to unit circle, sum complex, project back [§II]

    Memory: log2(r) bits/component vs 1 bit/component for BSC.
      MCR-3 (r=8):  3 bits/dim  ↔  4× less memory than BSC at same accuracy
      MCR-4 (r=16): 4 bits/dim  ↔  match FHRR (float32) at 8× less memory
    """

    def __init__(self, data: torch.Tensor, modulus: int):
        self.data = data.long() % modulus    # ensure [0, r)
        self.modulus = modulus
        self.dim = data.shape[0]

    @classmethod
    def random(cls, dim: int, modulus: int, seed: Optional[int] = None) -> "MCRVector":
        """Generate a random MCR HV."""
        g = torch.Generator()
        if seed is not None:
            g.manual_seed(seed)
        data = torch.randint(0, modulus, (dim,), generator=g)
        return cls(data, modulus)

    def bind(self, other: "MCRVector") -> "MCRVector":
        """c = (h + u) mod r."""
        assert self.modulus == other.modulus and self.dim == other.dim
        return MCRVector((self.data + other.data) % self.modulus, self.modulus)

    def unbind(self, other: "MCRVector") -> "MCRVector":
        """c = (h - u) mod r."""
        assert self.modulus == other.modulus and self.dim == other.dim
        return MCRVector((self.data - other.data) % self.modulus, self.modulus)

    def similarity(self, other: "MCRVector") -> float:
        """
        Modular Manhattan distance normalised to [0, 1] (Eq. 1).

        sim = 1 - δ(h,u) / (D × r/4)
        where δ = Σ min((h-u)%r, (u-h)%r) / D
        and r/4 is the max possible δ for uniform random vectors.
        """
        diff = (self.data - other.data) % self.modulus
        mirror = (other.data - self.data) % self.modulus
        dist = float(torch.min(diff, mirror).float().mean().item())
        # Normalise: max distance for uniform random = r/4
        max_dist = self.modulus / 4
        return 1.0 - dist / max_dist

    @classmethod
    def bundle(cls, vectors: List["MCRVector"]) -> "MCRVector":
        """
        Bundle (superposition) of MCR HVs via circular mean (§II).

        Each component is projected to exp(2πi/r × h), the complex
        values are summed, and the result is projected back to [0, r).
        """
        assert vectors, "Cannot bundle empty list"
        r = vectors[0].modulus
        D = vectors[0].dim

        # Project to unit circle in complex plane
        complex_sum = torch.zeros(D, dtype=torch.complex64)
        for v in vectors:
            angles = (2 * math.pi / r) * v.data.float()
            complex_sum += torch.exp(1j * angles)

        # Project back: angle → integer
        angles_out = torch.angle(complex_sum) % (2 * math.pi)
        bundled = torch.round(angles_out * r / (2 * math.pi)) % r
        return cls(bundled.long(), r)

    def to_binary(self) -> torch.Tensor:
        """
        Convert MCR HV to binary BSC by thresholding at r/2.

        Provides a bridge to binary HDC operations.
        """
        return (self.data >= self.modulus // 2).float()

    @property
    def bits_per_dim(self) -> float:
        return math.log2(self.modulus)

    def memory_bytes(self) -> int:
        return math.ceil(self.dim * self.bits_per_dim / 8)


class MCRCodebook:
    """
    Item memory using MCR HVs.

    Stores symbol→MCR HV mappings and supports fast similarity lookup.
    For an alphabet of size n with dim D and modulus r:
        Memory: n × D × ceil(log2(r)) bits   (vs n × D bits for BSC)

    At r=8 and D=2500: same capacity as BSC at D=10000 with same memory.
    """

    def __init__(self, config: MCRConfig):
        self.cfg = config
        self._hvs: Dict[str, MCRVector] = {}
        self._n = 0

    def add(self, name: str, hv: Optional[MCRVector] = None) -> MCRVector:
        """Add a symbol to the codebook (generates random HV if none given)."""
        if hv is None:
            hv = MCRVector.random(self.cfg.dim, self.cfg.modulus,
                                  seed=(self.cfg.seed or 0) + self._n)
        self._hvs[name] = hv
        self._n += 1
        return hv

    def get(self, name: str) -> Optional[MCRVector]:
        return self._hvs.get(name)

    def nearest(self, query: MCRVector, top_k: int = 1) -> List[Tuple[str, float]]:
        """Find the nearest HVs in the codebook."""
        results = []
        for name, hv in self._hvs.items():
            sim = query.similarity(hv)
            results.append((name, sim))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def information_rate(
        self,
        accuracy: float,
    ) -> Dict[str, float]:
        """
        Compute information rate metrics (Eqs. 8-11 from the paper).

        Args:
            accuracy: Decoding accuracy ∈ [0, 1]

        Returns:
            Dict with Isymb, Itot, Idim, Ibit
        """
        d = len(self._hvs)  # codebook size
        D = self.cfg.dim
        b = math.ceil(math.log2(self.cfg.modulus))

        if accuracy <= 0 or accuracy >= 1 or d <= 1:
            return {"Isymb": 0, "Idim": 0, "Ibit": 0}

        # Eq. 8: information per symbol
        Isymb = (accuracy * math.log2(d * accuracy)
                 + (1 - accuracy) * math.log2((d / (d - 1)) * (1 - accuracy)))

        Idim = Isymb / D       # Eq. 10: information per HV component
        Ibit = Isymb / (D * b)  # Eq. 11: information per storage bit

        return {
            "Isymb_bits": round(Isymb, 4),
            "Idim": round(Idim, 6),
            "Ibit": round(Ibit, 6),
            "bits_per_dim": b,
            "memory_bytes": self._n * self.cfg.dim * b // 8,
            "bsc_equivalent_memory_bytes": self._n * self.cfg.dim // 8,
            "memory_reduction": self.cfg.dim // b // self.cfg.dim if b > 1 else 1,
        }


class MCRClassifier:
    """
    HDC classifier using MCR HVs for 4× memory efficiency over BSC.

    One-shot training: for each class, bundle training sample HVs.
    Inference: find class prototype with highest MCR similarity.

    At r=8 (MCR-3) and D=2500:
      - Same accuracy as BSC at D=10000
      - Memory: 2500 × 3 bits vs 10000 × 1 bit = same bits, 4× fewer dims
      - Faster inference: 4× fewer operations
    """

    def __init__(self, config: MCRConfig):
        self.cfg = config
        self._prototypes: Dict[int, List[MCRVector]] = {}
        self._class_hvs: Dict[int, MCRVector] = {}

    def train_step(self, hv: MCRVector, label: int):
        """Accumulate HV into class prototype."""
        if label not in self._prototypes:
            self._prototypes[label] = []
        self._prototypes[label].append(hv)

    def finalize(self):
        """Bundle accumulated HVs into class prototypes."""
        for label, hvs in self._prototypes.items():
            self._class_hvs[label] = MCRVector.bundle(hvs)

    def predict(self, hv: MCRVector) -> Tuple[int, float]:
        """Predict class by nearest prototype."""
        best_label = 0
        best_sim = -1.0
        for label, proto in self._class_hvs.items():
            sim = hv.similarity(proto)
            if sim > best_sim:
                best_sim = sim
                best_label = label
        return best_label, best_sim

    def predict_batch(
        self,
        hvs: List[MCRVector],
    ) -> List[Tuple[int, float]]:
        """Predict class for a list of MCRVectors."""
        return [self.predict(hv) for hv in hvs]

    def online_update(
        self,
        hv:    MCRVector,
        label: int,
        lr:    float = 0.1,
    ):
        """
        Online RefineHD update: add one labelled MCRVector.

        If the label is already known, blend the new HV into the prototype.
        If not, create a new class.

        Args:
            hv:    MCRVector for the new example
            label: True class label
            lr:    Blending rate
        """
        if label not in self._class_hvs:
            self._class_hvs[label] = hv
            return

        old  = self._class_hvs[label]
        # Blend: (1-lr) × old + lr × new (element-wise in MCR space)
        blended_data = (
            (1 - lr) * old.data.float() + lr * hv.data.float()
        ) % old.modulus
        self._class_hvs[label] = MCRVector(
            blended_data.long(), old.modulus, old.device
        )

    def accuracy(
        self,
        hvs:    List[MCRVector],
        labels: List[int],
    ) -> float:
        """Compute accuracy over a list of (hv, label) pairs."""
        correct = sum(
            1 for hv, lbl in zip(hvs, labels)
            if self.predict(hv)[0] == lbl
        )
        return correct / max(len(hvs), 1)


# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_cambridge_vsa():
    print("=" * 60)
    print("Testing CambridgeTestVSA (Kleyko/Osipov/Gayler 2016)")
    print("=" * 60)

    vsa = CambridgeTestVSA(hd_dim=10000, double_boundary=True, seed=42)

    # Test encoding: same word = sim 1.0
    hv1 = vsa.encode_word("hello")
    hv2 = vsa.encode_word("hello")
    sim_same = float(hv_batch_sim(hv1, hv2.unsqueeze(0))[0])
    print(f"  sim(hello, hello) = {sim_same:.4f}  (want 1.0)")
    assert sim_same > 0.99

    # Different words = lower similarity
    hv3 = vsa.encode_word("world")
    sim_diff = float(hv_batch_sim(hv1, hv3.unsqueeze(0))[0])
    print(f"  sim(hello, world) = {sim_diff:.4f}  (want << 1.0)")
    assert sim_diff < 0.9

    # Cambridge test: scramble middle letters
    recognized, sim_scr = vsa.recognize("hlleo", "hello")  # scrambled 'hello'
    print(f"  sim(hlleo→hello) = {sim_scr:.4f}  recognized={recognized}")

    # Full Cambridge test on vocabulary
    words = ["computer", "cambridge", "reading", "language", "problem",
             "science", "research", "university", "understanding"]
    result = vsa.cambridge_test(words, n_trials=5)
    print(f"  Cambridge test: recognition={result['recognition_rate']:.3f} "
          f"FP={result['false_positive_rate']:.3f} "
          f"accuracy={result['accuracy']:.3f}")
    assert result['recognition_rate'] > 0.3, "Should recognise some scrambled words"

    # Without double_boundary: should be worse
    vsa_nodb = CambridgeTestVSA(hd_dim=10000, double_boundary=False, seed=42)
    result_nodb = vsa_nodb.cambridge_test(words, n_trials=5)
    print(f"  Without db:      recognition={result_nodb['recognition_rate']:.3f} "
          f"accuracy={result_nodb['accuracy']:.3f}")

    print("  ✅ CambridgeTestVSA OK")


def test_recursive_binding():
    print("=" * 60)
    print("Testing RecursiveBindingEncoder (Rachkovskij & Kleyko 2022)")
    print("=" * 60)

    enc = RecursiveBindingEncoder(alphabet_size=26, hd_dim=5000,
                                   similarity_radius=2, seed=0)

    # Theoretical position similarity
    for j in range(4):
        theory = enc.position_similarity(0, j)
        print(f"  pos_sim(0, {j}) theory = {theory:.3f}  (R=2: expect {max(0, 2-j)/2:.3f})")

    # Sequence encoding
    hello = [7, 4, 11, 11, 14]     # h=7, e=4, l=11, l=11, o=14
    hlleo = [7, 11, 11, 4, 14]     # scrambled: hlleo (correct first/last)

    hv_hello = enc.encode_sequence(hello, double_boundary=True)
    hv_hlleo = enc.encode_sequence(hlleo, double_boundary=True)
    hv_world = enc.encode_sequence([22, 14, 17, 11, 3], double_boundary=True)  # world

    sim_scr  = float(F.cosine_similarity(hv_hello.unsqueeze(0), hv_hlleo.unsqueeze(0)))
    sim_diff = float(F.cosine_similarity(hv_hello.unsqueeze(0), hv_world.unsqueeze(0)))
    print(f"  sim(hello, hlleo) = {sim_scr:.4f}  (scrambled: want high)")
    print(f"  sim(hello, world) = {sim_diff:.4f}  (different: want lower)")
    assert sim_scr > sim_diff, "Scrambled should be more similar than different word"

    print("  ✅ RecursiveBindingEncoder OK")


def test_mcr_backend():
    print("=" * 60)
    print("Testing MCRBackend (Angioli, Kymn, Kleyko 2025)")
    print("=" * 60)

    cfg = MCRConfig(dim=2500, modulus=8, seed=42)

    # Basic operations
    h = MCRVector.random(cfg.dim, cfg.modulus, seed=0)
    u = MCRVector.random(cfg.dim, cfg.modulus, seed=1)

    # Self-similarity = 1.0
    sim_self = h.similarity(h)
    print(f"  sim(h, h) = {sim_self:.4f}  (want 1.0)")
    assert sim_self > 0.99

    # MCR similarity: random vectors have sim ≈ 0 (NOT 0.5 like BSC).
    # MCR distances: E[δ] = r/4 = max expected → normalised sim → 0.
    # This is correct: MCR is maximally discriminative (no bias toward 0.5).
    sim_rand = h.similarity(u)
    print(f"  sim(h, random_u) = {sim_rand:.4f}  (MCR: random→0, identical→1)")
    assert sim_rand < 0.2, f"Random MCR vectors should have low similarity: {sim_rand}"

    # Bind then unbind = identity
    bound   = h.bind(u)
    unbound = bound.unbind(u)
    sim_rt  = h.similarity(unbound)
    print(f"  sim(h, bind(h,u).unbind(u)) = {sim_rt:.4f}  (want 1.0)")
    assert sim_rt > 0.99

    # Bundle
    hvs = [MCRVector.random(cfg.dim, cfg.modulus, seed=i) for i in range(5)]
    bundled = MCRVector.bundle(hvs)
    # Bundled should be similar to each component (majority property)
    sims = [bundled.similarity(v) for v in hvs]
    print(f"  Bundle sim to components: {[round(s, 3) for s in sims]}")

    # Memory efficiency
    print(f"  MCR-3 (r=8, D=2500): {h.bits_per_dim:.0f} bits/dim, "
          f"{h.memory_bytes():,} bytes/HV")
    bsc_equiv_bytes = cfg.dim // 8
    print(f"  BSC equivalent (D=2500): {bsc_equiv_bytes:,} bytes/HV "
          f"(MCR stores 3× more info per byte)")

    # Classifier test
    clf = MCRClassifier(cfg)
    codebook = MCRCodebook(cfg)

    torch.manual_seed(42)
    n_classes = 4
    class_protos = [MCRVector.random(cfg.dim, cfg.modulus, seed=100+c) for c in range(n_classes)]

    # Train
    for c in range(n_classes):
        for _ in range(10):
            # Noisy version of class proto
            noise_data = (class_protos[c].data + torch.randint(-1, 2, (cfg.dim,))) % cfg.modulus
            clf.train_step(MCRVector(noise_data, cfg.modulus), label=c)
    clf.finalize()

    # Test
    correct = 0
    for c in range(n_classes):
        noise_data = (class_protos[c].data + torch.randint(-1, 2, (cfg.dim,))) % cfg.modulus
        pred, _ = clf.predict(MCRVector(noise_data, cfg.modulus))
        if pred == c:
            correct += 1
    acc = correct / n_classes
    print(f"  Classifier accuracy: {acc:.1%}  ({n_classes} classes, noisy protos)")
    assert acc > 0.5

    print("  ✅ MCRBackend OK")


if __name__ == "__main__":
    test_cambridge_vsa()
    print()
    test_recursive_binding()
    print()
    test_mcr_backend()
    print()
    print("=== All sequence_vsa tests passed ===")
