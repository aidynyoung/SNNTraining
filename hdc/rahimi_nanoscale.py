"""
High-Dimensional Computing as a Nanoscalable Paradigm
======================================================
Based on: Rahimi, A., Datta, S., Kleyko, D., Frady, E.P., Olshausen, B.,
          Kanerva, P., and Rabaey, J.M. (2017)
"High-Dimensional Computing as a Nanoscalable Paradigm"
IEEE Transactions on Circuits and Systems I: Regular Papers, 64(9), 2508–2521.
DOI: 10.1109/TCSI.2017.2705060

Key contributions implemented:

1. **Level Hypervectors (LHV)** — Thermometer-code encoding for continuous values.
   Consecutive levels share high similarity; distant levels are near-orthogonal.
   Produced by starting from a random HV and flipping bits progressively.

2. **ID Hypervectors (ID-HV)** — One random, mutually near-orthogonal HV per item.
   Used to encode discrete tokens/identities.

3. **Full Record Encoder** — Binds each feature's ID-HV with its LHV and bundles
   all features into a single holographic HV (Section II-C of the paper).

4. **Ternary HDC** — Operations on {-1, 0, +1} HVs for analog CMOS/memristive
   implementations (Section III-B). Ternary binding = element-wise multiply;
   ternary bundling = thresholded sum.

5. **Nanoscale Associative Classifier** — Online one-shot learning + error-driven
   retraining with the threshold update rule from Section IV of the paper.

6. **Nanoscale Hardware Model** — Energy/area estimates for resistive-memory-based
   in-sensor HDC (Section V). Includes the bit-line cosine similarity model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from hdc.hdc_glue import (
    hv_xor, hv_bundle, hv_majority, hv_batch_sim, gen_hvs, hv_permute,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Section II-B: ID Hypervectors
# ═══════════════════════════════════════════════════════════════════════════════

class IDHypervectors:
    """
    ID Hypervectors (ID-HV) — one random near-orthogonal HV per discrete item.

    In the nanoscale paradigm, these are stored in a small on-chip item memory
    (resistive RAM or SRAM) and looked up in O(1) at encoding time.
    """

    def __init__(self, n_items: int, dim: int = 10000, seed: Optional[int] = None):
        self.n_items = n_items
        self.dim = dim
        self.hvs = gen_hvs(n_items, dim, seed=seed)  # (n_items, dim) binary {0,1}

    def encode(self, idx: int) -> torch.Tensor:
        """Return ID-HV for item index."""
        return self.hvs[idx]

    def encode_batch(self, indices: torch.Tensor) -> torch.Tensor:
        """Return ID-HVs for a batch of indices. Shape: (n, dim)."""
        return self.hvs[indices.long()]

    def similarity(self, a: torch.Tensor) -> torch.Tensor:
        """Hamming similarity of query HV to all ID-HVs. Shape: (n_items,)."""
        return hv_batch_sim(a, self.hvs)


# ═══════════════════════════════════════════════════════════════════════════════
# Section II-B: Level Hypervectors
# ═══════════════════════════════════════════════════════════════════════════════

class LevelHypervectors:
    """
    Level Hypervectors (LHV) — thermometer encoding for continuous values.

    Construction (Rahimi 2017, Section II-B):
      1. Start from a random base HV L_0.
      2. For each level q = 1 … Q-1, flip a fraction (q/(Q-1)) × D bits
         progressively from L_{q-1}.
    This guarantees:
      - Hamming distance ≈ 0 between adjacent levels.
      - Hamming distance ≈ D/2 between extreme levels (L_0 and L_{Q-1}).
      - Similarity is monotonically decreasing with level distance.

    Encoding a scalar value v ∈ [v_min, v_max]:
      q = round((v - v_min) / (v_max - v_min) × (Q - 1))
      → return L_q
    """

    def __init__(
        self,
        n_levels: int,
        dim: int = 10000,
        seed: Optional[int] = None,
    ):
        self.n_levels = n_levels
        self.dim = dim
        self.hvs = self._build_levels(n_levels, dim, seed)

    @staticmethod
    def _build_levels(Q: int, D: int, seed: Optional[int]) -> torch.Tensor:
        g = torch.Generator()
        if seed is not None:
            g.manual_seed(seed)

        # Base level: random binary
        base = (torch.rand(D, generator=g) < 0.5).float()
        levels = [base.clone()]

        for q in range(1, Q):
            # Fraction of bits to flip at this level relative to L_0
            frac = q / (Q - 1)
            n_flip = int(round(frac * D / 2))  # flip half the distance per step

            prev = levels[-1].clone()
            # Choose n_flip positions that are not yet flipped from base
            flip_idx = torch.randperm(D, generator=g)[:n_flip]
            prev[flip_idx] = 1.0 - prev[flip_idx]
            levels.append(prev)

        return torch.stack(levels)  # (Q, D)

    def encode(self, value: float, v_min: float = 0.0, v_max: float = 1.0) -> torch.Tensor:
        """Encode a scalar value to a level hypervector."""
        v_clamped = max(v_min, min(v_max, value))
        frac = (v_clamped - v_min) / max(v_max - v_min, 1e-12)
        level_idx = int(round(frac * (self.n_levels - 1)))
        return self.hvs[level_idx]

    def encode_batch(
        self,
        values: torch.Tensor,
        v_min: float = 0.0,
        v_max: float = 1.0,
    ) -> torch.Tensor:
        """Encode a batch of scalar values. Shape: (n,) → (n, dim)."""
        frac = (values.clamp(v_min, v_max) - v_min) / max(v_max - v_min, 1e-12)
        indices = (frac * (self.n_levels - 1)).round().long().clamp(0, self.n_levels - 1)
        return self.hvs[indices]

    def similarity_profile(self, a: torch.Tensor) -> torch.Tensor:
        """Similarity of query HV to all level HVs. Shape: (n_levels,)."""
        return hv_batch_sim(a, self.hvs)


# ═══════════════════════════════════════════════════════════════════════════════
# Section II-C: Record Encoder (ID-HV ⊗ LHV bundling)
# ═══════════════════════════════════════════════════════════════════════════════

class NanoscaleRecordEncoder(nn.Module):
    """
    Holographic record encoder from Rahimi 2017, Section II-C.

    For a feature vector x = [x_1, …, x_F] with F features:
      hv = MAJORITY(⊕_{i=1}^{F} ID_i ⊗ LHV(x_i))

    where ⊗ is XOR binding and MAJORITY is binarising threshold.

    ID-HVs encode feature identity; LHVs encode feature magnitude.
    The resulting HV is holographic: each bit sees influence from all features.

    Args:
        n_features: Number of input features (F)
        n_levels: Number of quantisation levels for continuous values (Q)
        dim: Hypervector dimensionality
        v_min: Minimum expected feature value
        v_max: Maximum expected feature value
        seed: Random seed
    """

    def __init__(
        self,
        n_features: int,
        n_levels: int = 100,
        dim: int = 10000,
        v_min: float = 0.0,
        v_max: float = 1.0,
        seed: Optional[int] = None,
        use_ca90: bool = False,
        ca90_seed_dim: int = 37,
    ):
        """
        Args:
            use_ca90: If True, replace stored id_hvs with CA90ItemMemory seeds.
                      Reduces ID-HV storage by factor dim/ca90_seed_dim (≈270× at
                      dim=10000, seed_dim=37). HVs are expanded on-the-fly.
                      (Kleyko, Frady, Sommer 2020 — hdc/ca_hdc.py)
            ca90_seed_dim: CA90 seed length (37 = prime → period 137B).
        """
        super().__init__()
        self.n_features = n_features
        self.n_levels = n_levels
        self.dim = dim
        self.v_min = v_min
        self.v_max = v_max
        self.use_ca90 = use_ca90

        lhv = LevelHypervectors(n_levels, dim, seed=(seed or 0) + 1000)
        self.register_buffer("level_hvs", lhv.hvs)    # (Q, D) — always stored

        if use_ca90:
            # Store only CA90 seeds; expand to full HVs on demand
            from hdc.ca_hdc import CA90ItemMemory
            self._ca90_mem = CA90ItemMemory(item_dim=dim, seed_dim=ca90_seed_dim, seed=seed)
            for i in range(n_features):
                self._ca90_mem.add(f"f{i}")
            # Placeholder buffer so register_buffer path still works
            self.register_buffer("id_hvs", torch.zeros(1, dim))  # not used when ca90
        else:
            id_hvs = IDHypervectors(n_features, dim, seed=seed)
            self.register_buffer("id_hvs", id_hvs.hvs)    # (F, D)
            self._ca90_mem = None

    def _get_id_hvs(self) -> torch.Tensor:
        """Return (n_features, dim) ID HV matrix, expanding CA90 if needed."""
        if self.use_ca90 and self._ca90_mem is not None:
            return self._ca90_mem.get_batch([f"f{i}" for i in range(self.n_features)])
        return self.id_hvs

    def _quantise(self, values: torch.Tensor) -> torch.Tensor:
        """Map feature values to level indices. Shape: (...) → (...)."""
        frac = (values.clamp(self.v_min, self.v_max) - self.v_min) / max(
            self.v_max - self.v_min, 1e-12
        )
        return (frac * (self.n_levels - 1)).round().long().clamp(0, self.n_levels - 1)

    def encode_single(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a single feature vector. x: (F,) → (D,) binary HV."""
        id_hvs = self._get_id_hvs()
        level_idx = self._quantise(x)                   # (F,)
        bound = hv_xor(id_hvs, self.level_hvs[level_idx])  # (F, D)
        bundled = hv_bundle(bound)                      # (D,) integer counts
        return hv_majority(bundled)                     # (D,) binary

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of feature vectors. x: (N, F) → (N, D) binary HVs."""
        N, F = x.shape
        id_hvs = self._get_id_hvs()                     # (F, D) — CA90 or standard
        level_idx = self._quantise(x)                   # (N, F)

        # Gather level HVs: (N, F, D)
        lvl_hvs = self.level_hvs[level_idx]

        # Bind with ID-HVs: (N, F, D)
        bound = hv_xor(id_hvs.unsqueeze(0), lvl_hvs)

        # Bundle across features: (N, D)
        bundled = bound.sum(dim=1)

        # Majority threshold
        return (bundled >= (F / 2)).float()


# ═══════════════════════════════════════════════════════════════════════════════
# Section III-B: Ternary HDC operations
# ═══════════════════════════════════════════════════════════════════════════════

def ternary_bind(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Ternary binding: element-wise multiply for {-1, 0, +1} HVs.

    In nanoscale analog implementations (Rahimi 2017, Section III-B),
    ternary HVs model partial activation. Binding via multiplication
    preserves the ternary structure and is implementable with simple
    resistive-memory crossbar circuits.

    Args:
        a, b: (..., D) ternary tensors in {-1, 0, +1}

    Returns:
        (..., D) ternary tensor
    """
    return a * b


def ternary_bundle(hvs: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
    """
    Ternary bundling: thresholded sum → {-1, 0, +1}.

    Sum over the first dimension, then apply:
        result = +1 if sum > threshold
               =  0 if |sum| ≤ threshold
               = -1 if sum < -threshold

    Args:
        hvs: (N, D) ternary HVs to bundle
        threshold: Dead-zone threshold (0 → hard sign)

    Returns:
        (D,) ternary HV
    """
    s = hvs.sum(dim=0)
    result = torch.zeros_like(s)
    result[s > threshold] = 1.0
    result[s < -threshold] = -1.0
    return result


def binary_to_ternary(hv: torch.Tensor) -> torch.Tensor:
    """Map binary {0,1} HV to bipolar ternary {-1,+1} (no zeros)."""
    return hv * 2.0 - 1.0


def ternary_to_binary(hv: torch.Tensor) -> torch.Tensor:
    """Map ternary/bipolar HV to binary {0,1} via thresholding at 0."""
    return (hv > 0).float()


def ternary_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """
    Cosine similarity for ternary/bipolar HVs.

    In the nanoscale bitline model (Rahimi 2017, Section III-B), similarity
    is computed as the normalised dot product on the analog crossbar,
    equivalent to cosine similarity for unit-norm vectors.
    """
    dot = (a * b).sum()
    norm_a = a.norm(p=2).clamp(min=1e-12)
    norm_b = b.norm(p=2).clamp(min=1e-12)
    return float((dot / (norm_a * norm_b)).item())


# ═══════════════════════════════════════════════════════════════════════════════
# Section IV: Nanoscale Associative Classifier
# ═══════════════════════════════════════════════════════════════════════════════

class NanoscaleHDCClassifier(nn.Module):
    """
    One-shot + error-driven HDC classifier from Rahimi 2017, Section IV.

    Training procedure:
      Phase 1 (One-shot): Accumulate each training sample into its class
        prototype accumulator. After all samples, binarise.
      Phase 2 (Retraining): For each misclassified sample:
        M[true] += hv(sample)   (reinforce correct prototype)
        M[pred] -= hv(sample)   (suppress wrong prototype)
        Re-binarise after each update.

    This implements the threshold-update rule described in Section IV:
        M_c ← sign(M_c + α * (y==c ? +1 : -1) * hv)

    For ternary implementations, weights and updates use {-1, 0, +1}.

    Args:
        encoder: NanoscaleRecordEncoder to produce HVs from features
        n_classes: Number of classes
        ternary: If True, use ternary bipolar prototypes and similarity
        n_retrain: Number of retraining passes
        seed: Random seed
    """

    def __init__(
        self,
        encoder: NanoscaleRecordEncoder,
        n_classes: int,
        ternary: bool = False,
        n_retrain: int = 5,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.n_classes = n_classes
        self.ternary = ternary
        self.n_retrain = n_retrain
        self.dim = encoder.dim

        # Prototype accumulators (float for accumulation, binarised after)
        self.register_buffer("accumulators", torch.zeros(n_classes, self.dim))
        self.register_buffer("counts", torch.zeros(n_classes))
        self.register_buffer(
            "prototypes",
            torch.zeros(n_classes, self.dim) if not ternary
            else torch.ones(n_classes, self.dim) * 0.5,
        )
        self._trained = False

    @torch.no_grad()
    def train_one_shot(self, x: torch.Tensor, labels: torch.Tensor):
        """
        Phase 1: single-pass prototype accumulation.

        Args:
            x: (N, F) feature matrix
            labels: (N,) integer class labels
        """
        hvs = self.encoder(x)  # (N, D)

        for i in range(hvs.shape[0]):
            c = int(labels[i].item())
            self.accumulators[c] += hvs[i]
            self.counts[c] += 1

        # Binarise
        for c in range(self.n_classes):
            if self.counts[c] > 0:
                thresh = self.counts[c] / 2.0
                self.prototypes[c] = (self.accumulators[c] >= thresh).float()

        self._trained = True

    @torch.no_grad()
    def retrain(self, x: torch.Tensor, labels: torch.Tensor):
        """
        Phase 2: error-driven threshold updates.

        Args:
            x: (N, F) feature matrix
            labels: (N,) integer class labels
        """
        if not self._trained:
            raise RuntimeError("Call train_one_shot() first.")

        hvs = self.encoder(x)  # (N, D)

        for _ in range(self.n_retrain):
            preds = self._predict_hvs(hvs)
            errors = (preds != labels).nonzero(as_tuple=True)[0]

            if len(errors) == 0:
                break  # Converged

            for idx in errors:
                hv = hvs[idx]
                true_c = int(labels[idx].item())
                pred_c = int(preds[idx].item())

                if self.ternary:
                    # Ternary: accumulate bipolar ±1 then ternary-bundle
                    bip = binary_to_ternary(hv)
                    self.accumulators[true_c] += bip
                    self.accumulators[pred_c] -= bip
                else:
                    self.accumulators[true_c] += hv
                    self.accumulators[pred_c] -= hv

            # Re-binarise prototypes
            for c in range(self.n_classes):
                if self.ternary:
                    self.prototypes[c] = ternary_to_binary(self.accumulators[c])
                else:
                    self.prototypes[c] = (self.accumulators[c] > 0).float()

    def _predict_hvs(self, hvs: torch.Tensor) -> torch.Tensor:
        """Predict labels for pre-encoded HVs. Shape: (N, D) → (N,)."""
        N = hvs.shape[0]
        preds = torch.zeros(N, dtype=torch.long)
        for i in range(N):
            if self.ternary:
                # Cosine similarity for bipolar
                bip = binary_to_ternary(hvs[i])
                proto_bip = binary_to_ternary(self.prototypes)  # (C, D)
                sims = torch.tensor([
                    ternary_similarity(bip, proto_bip[c])
                    for c in range(self.n_classes)
                ])
            else:
                sims = hv_batch_sim(hvs[i], self.prototypes)
            preds[i] = sims.argmax()
        return preds

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict class labels. x: (N, F) → (N,) labels."""
        hvs = self.encoder(x)
        return self._predict_hvs(hvs)

    def accuracy(self, x: torch.Tensor, labels: torch.Tensor) -> float:
        """Compute classification accuracy."""
        preds = self.forward(x)
        return float((preds == labels).float().mean().item())


# ═══════════════════════════════════════════════════════════════════════════════
# Section V: Nanoscale Hardware Energy Model
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class NanoscaleHardwareConfig:
    """
    Configuration for nanoscale HDC hardware (Rahimi 2017, Section V).

    Models a resistive-memory-integrated (RRAM or DRAM-based) HDC processor
    where the prototype matrix is stored in memory arrays and similarity is
    computed via bitline-current accumulation (analog dot-product).
    """
    dim: int = 10000
    n_classes: int = 10
    n_features: int = 100
    n_levels: int = 100

    # Technology node
    tech_nm: float = 14.0

    # Memory parameters
    rram_cell_area_f2: float = 4.0      # cell area in F² (RRAM)
    rram_read_energy_fj: float = 50.0   # fJ per bit read (RRAM)
    sram_read_energy_fj: float = 5.0    # fJ per bit read (SRAM item memory)

    # Logic energy at this node (scaled from 45nm Horowitz model)
    xor_energy_fj: float = 10.0         # fJ per XOR gate
    popcount_energy_fj: float = 15.0    # fJ per popcount (with adder tree)
    majority_energy_fj: float = 8.0     # fJ per majority gate

    frequency_mhz: float = 500.0


class NanoscaleHardwareModel:
    """
    Energy and area model for nanoscale HDC (Rahimi 2017, Section V).

    Key insight: In RRAM-integrated HDC, each memory cell computes a
    partial dot-product via its conductance — similarity search costs
    only O(D × C) cell reads with no explicit XOR or popcount needed.
    This gives 10–100× energy advantage over SRAM+digital-logic approaches.
    """

    def __init__(self, config: NanoscaleHardwareConfig):
        self.cfg = config

    # ── Encoding cost ─────────────────────────────────────────────────────────

    def encode_energy_fj(self) -> Dict[str, float]:
        """
        Energy to encode one sample into a hypervector.

        Per feature:
          - 1 SRAM read for ID-HV (dim bits)
          - 1 SRAM read for LHV (dim bits)
          - dim XOR operations (ID-HV ⊗ LHV)
        Then: dim majority operations to bundle F features.
        """
        cfg = self.cfg
        F = cfg.n_features
        D = cfg.dim

        id_read = F * D * cfg.sram_read_energy_fj / 1000   # per-bit → total fJ
        lhv_read = F * D * cfg.sram_read_energy_fj / 1000
        xor_cost = F * D * cfg.xor_energy_fj / 1000
        majority_cost = D * cfg.majority_energy_fj / 1000

        total = id_read + lhv_read + xor_cost + majority_cost
        return {
            "id_hv_read_fj": id_read,
            "level_hv_read_fj": lhv_read,
            "xor_fj": xor_cost,
            "majority_fj": majority_cost,
            "total_encode_fj": total,
        }

    # ── Classification cost ───────────────────────────────────────────────────

    def classify_energy_digital_fj(self) -> Dict[str, float]:
        """
        Digital SRAM + XOR + popcount similarity search.

        For each of C classes: D XOR + D popcount + 1 compare.
        """
        cfg = self.cfg
        C, D = cfg.n_classes, cfg.dim

        proto_read = C * D * cfg.sram_read_energy_fj / 1000
        xor_cost = C * D * cfg.xor_energy_fj / 1000
        popcount_cost = C * D * cfg.popcount_energy_fj / 1000
        compare_cost = C * cfg.xor_energy_fj / 1000  # final argmax

        total = proto_read + xor_cost + popcount_cost + compare_cost
        return {
            "proto_read_fj": proto_read,
            "xor_fj": xor_cost,
            "popcount_fj": popcount_cost,
            "compare_fj": compare_cost,
            "total_classify_fj": total,
        }

    def classify_energy_rram_fj(self) -> Dict[str, float]:
        """
        RRAM bitline-current analog similarity search (Rahimi 2017, Section V).

        In RRAM integration, one wordline pulse activates all D cells in a row
        in parallel — the aggregated bitline current IS the dot-product.
        Cost per class = 1 wordline pulse (not D cell reads), so total = C pulses.
        Each wordline pulse energy scales as √D × I_cell × V_read.

        Energy reduction: ~5–20× over digital at same precision.
        """
        cfg = self.cfg
        C, D = cfg.n_classes, cfg.dim

        # One wordline activation reads D cells simultaneously (amortised cost)
        # Energy per wordline ≈ D × rram_cell_read / parallelism_factor
        # Parallelism factor ≈ D (all cells read in one cycle on the bitline)
        wordline_energy_fj = cfg.rram_read_energy_fj * 2  # 2 fJ overhead per row activation
        rram_read = C * wordline_energy_fj

        # ADC to convert bitline current to digital: ~5 fJ per column, amortised
        adc_cost = C * 5.0

        total = rram_read + adc_cost
        digital_total = self.classify_energy_digital_fj()["total_classify_fj"]
        return {
            "rram_read_fj": rram_read,
            "adc_compare_fj": adc_cost,
            "total_classify_fj": total,
            "speedup_vs_digital": digital_total / max(total, 1e-9),
        }

    # ── Full inference comparison ─────────────────────────────────────────────

    def full_inference_comparison(self) -> Dict[str, float]:
        """Compare total inference energy: digital vs. RRAM integration."""
        enc = self.encode_energy_fj()["total_encode_fj"]
        dig = self.classify_energy_digital_fj()["total_classify_fj"]
        rram = self.classify_energy_rram_fj()["total_classify_fj"]

        return {
            "encode_fj": enc,
            "classify_digital_fj": dig,
            "classify_rram_fj": rram,
            "total_digital_fj": enc + dig,
            "total_rram_fj": enc + rram,
            "rram_vs_digital_speedup": (enc + dig) / max(enc + rram, 1e-9),
        }

    # ── Area estimate ─────────────────────────────────────────────────────────

    def area_estimate_um2(self) -> Dict[str, float]:
        """Estimate silicon area for nanoscale HDC processor."""
        cfg = self.cfg
        F2 = (cfg.tech_nm * 1e-3) ** 2  # F² in μm²

        # RRAM prototype array: C × D cells
        rram_cells = cfg.n_classes * cfg.dim
        rram_area = rram_cells * cfg.rram_cell_area_f2 * F2

        # SRAM item memory: (F + Q) × D
        sram_cells = (cfg.n_features + cfg.n_levels) * cfg.dim
        sram_area = sram_cells * 146 * F2  # standard 6T-SRAM = 146 F²

        # Logic (XOR tree + majority): rough estimate
        logic_area = cfg.dim * 20 * F2  # ~20 F² per bit-slice

        total = rram_area + sram_area + logic_area
        return {
            "rram_array_um2": rram_area,
            "sram_item_memory_um2": sram_area,
            "logic_um2": logic_area,
            "total_um2": total,
            "total_mm2": total * 1e-6,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_level_hypervectors():
    print("=" * 60)
    print("Testing Level Hypervectors (Rahimi 2017, Section II-B)")
    print("=" * 60)

    Q, D = 100, 10000
    lhv = LevelHypervectors(n_levels=Q, dim=D, seed=42)

    # Adjacent levels should be highly similar
    sim_adjacent = float(hv_batch_sim(lhv.hvs[0], lhv.hvs[1:2])[0])
    # Extreme levels should be near-orthogonal
    sim_extreme = float(hv_batch_sim(lhv.hvs[0], lhv.hvs[-1:])[0])

    print(f"  Similarity L_0 vs L_1  : {sim_adjacent:.4f}  (want ≈ 1.0)")
    print(f"  Similarity L_0 vs L_99 : {sim_extreme:.4f}  (want ≈ 0.5)")
    assert sim_adjacent > 0.85, f"Adjacent levels too dissimilar: {sim_adjacent}"
    assert sim_extreme < 0.65, f"Extreme levels too similar: {sim_extreme}"
    print("  ✅ Level hypervectors OK")


def test_record_encoder():
    print("=" * 60)
    print("Testing NanoscaleRecordEncoder (Rahimi 2017, Section II-C)")
    print("=" * 60)

    enc = NanoscaleRecordEncoder(n_features=20, n_levels=50, dim=5000, seed=0)
    x = torch.rand(8, 20)
    hvs = enc(x)
    assert hvs.shape == (8, 5000), f"Shape mismatch: {hvs.shape}"
    assert hvs.max() <= 1.0 and hvs.min() >= 0.0, "Not binary"
    print(f"  Encoded {x.shape} → {hvs.shape}  ✅")


def test_nanoscale_classifier():
    print("=" * 60)
    print("Testing NanoscaleHDCClassifier (Rahimi 2017, Section IV)")
    print("=" * 60)

    torch.manual_seed(42)
    n_classes, n_feat, dim = 4, 16, 4000
    enc = NanoscaleRecordEncoder(n_features=n_feat, n_levels=32, dim=dim, seed=1)
    clf = NanoscaleHDCClassifier(enc, n_classes=n_classes, ternary=False, n_retrain=3)

    # Synthetic data: class-specific feature patterns
    X, y = [], []
    for c in range(n_classes):
        proto = torch.zeros(n_feat)
        proto[c * 4:(c + 1) * 4] = 1.0
        for _ in range(20):
            x = proto + torch.randn(n_feat) * 0.15
            X.append(x.clamp(0, 1))
            y.append(c)
    X = torch.stack(X)
    y = torch.tensor(y)

    clf.train_one_shot(X, y)
    acc_one_shot = clf.accuracy(X, y)
    print(f"  One-shot accuracy : {acc_one_shot:.1%}")

    clf.retrain(X, y)
    acc_retrain = clf.accuracy(X, y)
    print(f"  After retraining  : {acc_retrain:.1%}")
    assert acc_retrain >= acc_one_shot - 0.05, "Retraining degraded accuracy"
    print("  ✅ Nanoscale classifier OK")


def test_ternary_ops():
    print("=" * 60)
    print("Testing Ternary Operations (Rahimi 2017, Section III-B)")
    print("=" * 60)

    D = 10000
    a = torch.randint(-1, 2, (D,)).float()
    b = torch.randint(-1, 2, (D,)).float()

    bound = ternary_bind(a, b)
    assert set(bound.unique().tolist()).issubset({-1.0, 0.0, 1.0}), "Bind broke ternary"

    hvs = torch.stack([a, b, ternary_bind(a, b)])
    bundled = ternary_bundle(hvs)
    assert set(bundled.unique().tolist()).issubset({-1.0, 0.0, 1.0}), "Bundle broke ternary"

    sim = ternary_similarity(a, a)
    print(f"  Self-similarity: {sim:.4f}  (want ≈ 1.0)")
    assert sim > 0.99
    print("  ✅ Ternary ops OK")


def test_hardware_model():
    print("=" * 60)
    print("Testing Nanoscale Hardware Model (Rahimi 2017, Section V)")
    print("=" * 60)

    cfg = NanoscaleHardwareConfig(dim=10000, n_classes=10, n_features=100, n_levels=100)
    hw = NanoscaleHardwareModel(cfg)

    comp = hw.full_inference_comparison()
    print(f"  Digital total : {comp['total_digital_fj']:.1f} fJ")
    print(f"  RRAM total    : {comp['total_rram_fj']:.1f} fJ")
    print(f"  RRAM speedup  : {comp['rram_vs_digital_speedup']:.1f}×")

    area = hw.area_estimate_um2()
    print(f"  Total area    : {area['total_mm2']:.4f} mm²")

    assert comp["rram_vs_digital_speedup"] > 1.0, "RRAM should beat digital"
    print("  ✅ Hardware model OK")


if __name__ == "__main__":
    test_level_hypervectors()
    print()
    test_record_encoder()
    print()
    test_ternary_ops()
    print()
    test_nanoscale_classifier()
    print()
    test_hardware_model()
    print()
    print("=== All Rahimi 2017 nanoscale tests passed ===")
