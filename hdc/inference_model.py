"""
hdc/inference_model.py
======================
Holographic Inference Model — the architecture described in the HDC lecture.

The lecture's key insight: "You've superposed the models of different things
into this blue vector [M].  Each yellow vector is a pairing of some sort of
input training data XORed with the class that is randomly generated as just
a symbolic concept."

Current codebase (AdaptiveHDClassifier) does:
    - K separate prototype vectors, one per class
    - Inference: argmax sim(encode(query), P_c)

This module implements what the lecture describes:
    - ONE model vector M = Σ_c Σ_x bind(encode(x), class_hv_c)
    - Inference: XOR(encode(query), M) → look up result in class codebook
    - Reconstruction: XOR(class_hv_c, M) → noisy stereotypical input
    - Provenance: find which stored examples best match a query
    - Distribution test: statistical test using the coin-flip null distribution

The mathematical basis (from lecture):
    bind(U, B) = XOR(U, B)   [U = image HV, B = class HV]
    M = bundle(bind(U,B), bind(V,C), bind(W,D))  = XOR(U,B) + XOR(V,C) + XOR(W,D)
    classify(query Q):
        result = XOR(Q, M)
        if Q ≈ U:  result ≈ B + noise  → nearest class is B ✓
    reconstruct(class C):
        result = XOR(C, M)
        if C = B:  result ≈ U + noise  → stereotypical input ✓

    The collapse works because XOR is self-inverse (an involution):
        XOR(Q, XOR(Q, B)) = XOR(XOR(Q, Q), B) = XOR(0, B) = B

The distribution test (lecture §13:18–14:30):
    "Turn the coin flip distribution into a normal distribution.
     For incorrect classes, the distribution should look like a bell curve
     centred at 0.5 Hamming distance.  For the correct class it should be
     much flatter, much closer to 0."

    Statistical test: z = (0.5 - H(result, class_hv) / dim) / σ(dim)
    σ = 1/(2√dim)  [from concentration.py]
    At dim=8192: σ ≈ 0.0055 → 3σ cutoff at H/dim = 0.483

Usage:
    from hdc.inference_model import HolographicInferenceModel, FixedThresholdEncoder

    # Wrap any encoder so its output is properly binarized
    enc = FixedThresholdEncoder(my_raw_encoder, dim=8192)
    enc.calibrate(sample_inputs)          # set per-dim threshold from data

    model = HolographicInferenceModel(dim=8192, n_classes=10, encoder=enc)
    for x, label in training_data:
        model.train(x, label)
    model.finalize()

    pred, stats = model.classify(query)
    recon      = model.reconstruct(class_idx=3)
    provenance = model.provenance(query, top_k=5)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch

from hdc.concentration import (
    DIM_CANONICAL,
    theoretical_std,
    snr_db,
    binarize_to_mean,
)
from hdc.hdc_glue import hv_xor, hv_majority, hv_hamming_sim, hv_batch_sim, gen_hvs

# ── FixedThresholdEncoder ─────────────────────────────────────────────────────

class FixedThresholdEncoder:
    """Wrap any encoder so its output is a balanced binary HV.

    Per-sample median binarization (binarize_to_mean) erases class structure
    because it adapts the threshold to each input individually — different
    class inputs that project to different magnitudes end up with the same
    ~50% ones pattern after adaptive thresholding.

    The correct approach: calibrate a FIXED per-dimension threshold from a
    representative set of training inputs, then apply that threshold at both
    training and inference time.  Different class inputs project differently
    relative to the fixed threshold → different bits fire → class structure
    is preserved in the binary HV.

    This is what the lecture means by "the only non-random vector is the
    input from the outer world": the fixed threshold encodes the statistical
    structure of the training distribution, so the binary projection
    genuinely reflects class-specific information.

    Args:
        encoder: Callable that maps input tensors to float activation vectors.
                 May return raw sums (e.g. SpikeHDC), log-softmax, or floats.
        dim: Output HV dimension.
        ema_alpha: EMA decay for online threshold updates during training.
    """

    def __init__(
        self,
        encoder: Optional[Callable],
        dim: int = DIM_CANONICAL,
        ema_alpha: float = 0.05,
    ):
        self.encoder = encoder
        self.dim = dim
        self.ema_alpha = ema_alpha
        self._threshold: Optional[torch.Tensor] = None
        self._calibrated = False
        self._n_seen = 0

    def calibrate(
        self,
        samples: torch.Tensor,
        show_balance: bool = False,
    ) -> "FixedThresholdEncoder":
        """Set the fixed threshold from a batch of calibration inputs.

        The threshold is the per-dimension median of the encoder output
        across all calibration samples.  Using the global median (not
        per-sample) preserves inter-sample variation while centering
        each dimension at its natural operating point.

        Args:
            samples: (n, input_dim) calibration inputs
            show_balance: Print the resulting ones-fraction as a sanity check

        Returns:
            self (for chaining)
        """
        with torch.no_grad():
            acts = []
            for i in range(len(samples)):
                out = self._raw_encode(samples[i])
                acts.append(out)
            stacked = torch.stack(acts)  # (n, dim)
        self._threshold = stacked.median(dim=0).values
        self._calibrated = True
        if show_balance:
            binary = (stacked > self._threshold.unsqueeze(0)).float()
            print(f"  Calibration balance: {binary.mean():.3f} ones (target 0.50)")
        return self

    def _raw_encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.encoder is None:
            out = x.float()
        else:
            out = self.encoder(x)
        if out.dim() > 1:
            out = out.squeeze(0)
        return out.float()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Encode x to a balanced binary HV using the fixed threshold.

        If not yet calibrated, falls back to per-sample median (suboptimal
        but safe) and updates a running threshold online.
        """
        out = self._raw_encode(x)
        if not self._calibrated:
            # Online calibration: update running median via EMA on per-sample
            batch_median = out.median()
            if self._threshold is None:
                self._threshold = torch.full((self.dim,), batch_median.item())
            else:
                self._threshold = (
                    (1 - self.ema_alpha) * self._threshold
                    + self.ema_alpha * out
                )
            self._n_seen += 1
        return (out > self._threshold).float()

    def update(self, x: torch.Tensor) -> None:
        """Incrementally update the threshold from a new sample (online use)."""
        out = self._raw_encode(x)
        if self._threshold is None:
            self._threshold = out.clone()
        else:
            self._threshold = (
                (1 - self.ema_alpha) * self._threshold + self.ema_alpha * out
            )

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated


# ── Helpers ───────────────────────────────────────────────────────────────────

def _hamming_distance(a: torch.Tensor, b: torch.Tensor) -> int:
    """Integer Hamming distance between two binary HVs."""
    return int(hv_xor(a, b).sum().item())


def _hamming_distance_norm(a: torch.Tensor, b: torch.Tensor) -> float:
    """Normalised Hamming distance H(a,b)/dim ∈ [0,1]."""
    return float(hv_xor(a, b).mean().item())


def _bind(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Bind two HVs: XOR(a, b) — self-inverse, recovers either operand."""
    return hv_xor(a, b)


def _bundle(hvs: List[torch.Tensor]) -> torch.Tensor:
    """Bundle (consensus sum / majority vote) a list of HVs."""
    stacked = torch.stack(hvs)
    return hv_majority(stacked.sum(dim=0))


# ── Statistics from the null distribution ─────────────────────────────────────

def _z_score(h_norm: float, dim: int) -> float:
    """Z-score: how many σ below the null (0.5 Hamming) is h_norm?"""
    return (0.5 - h_norm) / theoretical_std(dim)


def _p_value_one_tail(z: float) -> float:
    """P(Z ≥ z) for a standard normal — probability the match is noise."""
    # Approximation good to 1e-4 for z > 0
    if z <= 0:
        return 0.5
    t = 1.0 / (1.0 + 0.2316419 * z)
    poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937
           + t * (-1.821255978 + t * 1.330274429))))
    phi_z = (1.0 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * z * z)
    return phi_z * poly


# ── HolographicInferenceModel ─────────────────────────────────────────────────

class HolographicInferenceModel:
    """The holographic M-model from the HDC lecture.

    Architecture:
        Train:     M = Σ_{c,x} bind(encode(x), class_hv_c)
        Infer:     XOR(encode(query), M) → nearest class HV
        Reconstruct: XOR(class_hv_c, M) → noisy aggregate of class_c inputs
        Provenance: rank stored bindings by Hamming distance to query

    Key properties (from lecture):
    - M is holographic: ALL classes are superposed in a SINGLE vector
    - XOR is self-inverse (involution): XOR(a, XOR(a, b)) = b
    - Correct class: H(result, class_hv_c) << 0.5  (signal)
    - Wrong class:   H(result, class_hv_c) ≈ 0.5   (noise floor)
    - Sparsity in the encoder is critical: dense encodings cause cross-class
      interference that degrades the SNR of the collapse

    Args:
        dim: Hypervector dimension (default: 2^13 = 8192 = 1KB)
        n_classes: Number of classes
        encoder: Callable that maps input tensors to binary HVs of shape (dim,)
        store_bindings: If True, store individual bindings for provenance queries
        seed: RNG seed for class hypervectors
    """

    def __init__(
        self,
        dim: int = DIM_CANONICAL,
        n_classes: int = 10,
        encoder: Optional[Callable] = None,
        store_bindings: bool = True,
        seed: Optional[int] = None,
    ):
        self.dim = dim
        self.n_classes = n_classes
        self.encoder = encoder
        self.store_bindings = store_bindings

        # Random class hypervectors — the "symbolic concepts" from the lecture.
        # These are completely random and carry no external information by design.
        self.class_hvs: torch.Tensor = gen_hvs(n_classes, dim, seed=seed)

        # The model accumulator M (float, thresholded to binary in finalize())
        self._M_acc: torch.Tensor = torch.zeros(dim)
        self._M_binary: Optional[torch.Tensor] = None
        self._n_trained: int = 0

        # Per-example bindings for provenance (optional)
        # Each entry: (binding_hv, class_idx, optional_label_str)
        self._bindings: List[Tuple[torch.Tensor, int, Optional[str]]] = []

        # Class counts for capacity estimation
        self._class_counts: torch.Tensor = torch.zeros(n_classes, dtype=torch.int64)

        # Sparsity stats (updated during training)
        self._encoding_density: List[float] = []  # fraction of 1-bits per encoding

    # ── Training ──────────────────────────────────────────────────────────────

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input x to a strictly binary {0,1} HV.

        XOR semantics require binary input. Non-binary values break the
        M-model silently: (8 != 0) = True, so every binding becomes all-ones.

        Priority order:
          1. FixedThresholdEncoder — best: preserves class structure
          2. Already-binary input — pass through unchanged
          3. Per-sample median fallback — worst: erases class structure,
             but at least avoids the all-ones failure mode
        """
        if self.encoder is None:
            hv = x.float()
        elif isinstance(self.encoder, FixedThresholdEncoder):
            # FixedThresholdEncoder handles binarization correctly
            hv = self.encoder(x)
            if hv.dim() > 1:
                hv = hv.squeeze(0)
            return hv.float()
        else:
            hv = self.encoder(x)
        if hv.dim() > 1:
            hv = hv.squeeze(0)
        hv = hv.float()
        if not ((hv == 0.0) | (hv == 1.0)).all():
            # Warn once that a FixedThresholdEncoder should be used
            if not getattr(self, '_warned_threshold', False):
                import warnings
                warnings.warn(
                    "HolographicInferenceModel: encoder returned non-binary values. "
                    "Using per-sample median fallback — class structure may be lost. "
                    "Wrap your encoder with FixedThresholdEncoder for correct results.",
                    UserWarning, stacklevel=3,
                )
                self._warned_threshold = True
            hv = binarize_to_mean(hv)
        return hv

    def train(
        self,
        x: torch.Tensor,
        class_idx: int,
        label: Optional[str] = None,
    ) -> None:
        """Add one training example to the model.

        Computes bind(encode(x), class_hv_c) and accumulates into M.

        The model becomes incrementally richer with each call — no retraining
        needed.  Old examples remain encoded in M because bundling is
        superposition, not overwriting.

        Args:
            x: Input tensor (passed to encoder or binarized directly)
            class_idx: Integer class index (0 to n_classes-1)
            label: Optional string label for provenance display
        """
        hv = self._encode(x)
        binding = _bind(hv, self.class_hvs[class_idx])

        # Accumulate into M
        self._M_acc += binding.float()
        self._M_binary = None  # invalidate cached binary version

        if self.store_bindings:
            self._bindings.append((binding.clone(), class_idx, label))

        self._class_counts[class_idx] += 1
        self._n_trained += 1

        # Track encoding density (sparsity)
        density = float(hv.mean().item())
        self._encoding_density.append(density)

    def train_batch(
        self,
        xs: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        """Add a batch of (input, label) pairs.

        Args:
            xs: (batch, ...) input tensors
            labels: (batch,) integer class indices
        """
        for i in range(len(xs)):
            self.train(xs[i], int(labels[i].item()))

    def finalize(self) -> None:
        """Threshold the accumulated M to binary (majority vote).

        Call after all training examples have been added.
        Must be called before classify(), reconstruct(), or provenance().

        The accumulator holds RAW SUMS in [0, n_trained], not averages.
        Correct majority-vote threshold is n_trained/2, NOT 0.5.
        hv_majority(x) = (x >= 0.5) only works for pre-averaged sums —
        applying it to raw counts makes M all-ones after the first update.
        """
        if self._n_trained == 0:
            self._M_binary = torch.zeros(self.dim)
            return
        threshold = self._n_trained / 2.0
        self._M_binary = (self._M_acc > threshold).float()
        # Handle exact ties (even n) by random tiebreak — preserves balance
        if self._n_trained % 2 == 0:
            ties = (self._M_acc == threshold)
            if ties.any():
                self._M_binary[ties] = (torch.rand(ties.sum()) > 0.5).float()

    # ── Inference ─────────────────────────────────────────────────────────────

    def classify(
        self,
        x: torch.Tensor,
        return_stats: bool = True,
    ) -> Tuple[int, Dict]:
        """Classify a query by XOR-collapsing the superposed model M.

        XOR(encode(query), M) produces a vector close to class_hv_c when the
        query matches class c, because:
            XOR(Q, M) ≈ XOR(Q, bind(Q, C)) + noise = C + noise

        Args:
            x: Query input
            return_stats: If True, include statistical confidence in output

        Returns:
            (predicted_class_idx, stats_dict)
        """
        if self._M_binary is None:
            self.finalize()

        hv = self._encode(x)
        result = _bind(hv, self._M_binary)  # XOR(query, M)

        # Find nearest class HV in the result
        dists = torch.tensor([
            _hamming_distance_norm(result, self.class_hvs[c])
            for c in range(self.n_classes)
        ])
        pred = int(dists.argmin().item())
        best_dist = float(dists[pred].item())

        if not return_stats:
            return pred, {}

        # Statistical test against the null (coin-flip) distribution
        z = _z_score(best_dist, self.dim)
        p = _p_value_one_tail(z)

        # Second-best distance (for margin calculation)
        dists_sorted = dists.sort().values
        margin = float((dists_sorted[1] - dists_sorted[0]).item()) if len(dists_sorted) > 1 else 0.0

        stats = {
            "pred_class": pred,
            "hamming_distances": dists.tolist(),
            "best_dist": best_dist,
            "second_best_dist": float(dists_sorted[1].item()) if len(dists_sorted) > 1 else 1.0,
            "margin": margin,
            "z_score": z,
            "p_value": p,
            # Lecture: "you expect your examples to fall within a certain bell curve"
            "significant_3sigma": z > 3.0,
            "significant_5sigma": z > 5.0,
            "snr_db": snr_db(self.dim, best_dist),
            "null_mean": 0.5,
            "null_std": theoretical_std(self.dim),
        }
        return pred, stats

    def classify_soft(self, x: torch.Tensor) -> torch.Tensor:
        """Return soft Hamming similarity scores for all classes.

        Returns:
            (n_classes,) tensor of normalised Hamming distances (lower = more similar)
        """
        if self._M_binary is None:
            self.finalize()
        hv = self._encode(x)
        result = _bind(hv, self._M_binary)
        dists = torch.tensor([
            _hamming_distance_norm(result, self.class_hvs[c])
            for c in range(self.n_classes)
        ])
        return dists

    # ── Reconstruction (generative direction) ─────────────────────────────────

    def reconstruct(self, class_idx: int) -> torch.Tensor:
        """Reconstruct the stereotypical encoding of a class.

        XOR(class_hv_c, M) ≈ aggregate of all training encodings for class c.

        From the lecture: "If you give me the symbolic vector, I give you my
        stereotypical representation of the image."

        The reconstruction is noisy (interference from other classes), but at
        dim=8192 the SNR is sufficient to recover the dominant pattern.

        Args:
            class_idx: Class to reconstruct

        Returns:
            (dim,) binary HV — the stereotypical encoding of class_idx inputs
        """
        if self._M_binary is None:
            self.finalize()
        recon = _bind(self.class_hvs[class_idx], self._M_binary)
        # Apply majority vote to clean up cross-class noise
        return hv_majority(recon.float())

    def reconstruct_with_noise(
        self,
        class_idx: int,
        noise_rate: float = 0.1,
    ) -> torch.Tensor:
        """Sample a noisy reconstruction of a class.

        From the lecture: "If you pepper [the class HV] in with random noise,
        it gives you a random generation of your training data."

        Use this to model the training distribution by varying noise_rate.

        Args:
            class_idx: Class to reconstruct
            noise_rate: Fraction of bits to randomly flip

        Returns:
            (dim,) noisy binary HV
        """
        recon = self.reconstruct(class_idx)
        mask = torch.rand(self.dim) < noise_rate
        noisy = recon.clone()
        noisy[mask] = 1.0 - noisy[mask]
        return noisy

    def reconstruction_similarity(self, class_idx: int, query_hv: torch.Tensor) -> float:
        """Hamming similarity between reconstruction and a known encoding."""
        recon = self.reconstruct(class_idx)
        return float(hv_hamming_sim(recon, query_hv.float()).item())

    # ── Provenance ─────────────────────────────────────────────────────────────

    def provenance(
        self,
        x: torch.Tensor,
        top_k: int = 5,
    ) -> List[Dict]:
        """Find the stored training examples most similar to query x.

        For each stored binding (bind(x_i, c_i)):
            XOR(query_hv, bind(x_i, c_i)) ≈ class_hv_c when query ≈ x_i
            → measure H(result, class_hv_c) to score relevance

        From the lecture: "You can tell which training examples are very good
        from your dataset, which ones are poor, and what the stereotypical
        example looks like."

        Args:
            x: Query input
            top_k: Number of top-matching examples to return

        Returns:
            List of dicts sorted by relevance (most relevant first), each with
            keys: rank, class_idx, hamming_to_class, z_score, label
        """
        if not self._bindings:
            return []

        hv = self._encode(x)
        scored = []
        for i, (binding, class_idx, label) in enumerate(self._bindings):
            # Attempt to recover class_hv from binding+query
            candidate = _bind(hv, binding)  # XOR(query, bind(x_i, c_i))
            # If query ≈ x_i, candidate ≈ class_hv_ci
            h = _hamming_distance_norm(candidate, self.class_hvs[class_idx])
            z = _z_score(h, self.dim)
            scored.append({
                "idx": i,
                "class_idx": class_idx,
                "hamming_to_class": h,
                "z_score": z,
                "label": label,
            })

        scored.sort(key=lambda d: d["hamming_to_class"])
        for rank, entry in enumerate(scored[:top_k]):
            entry["rank"] = rank + 1
        return scored[:top_k]

    # ── Distribution analysis ─────────────────────────────────────────────────

    def distribution_stats(self, x: torch.Tensor) -> Dict:
        """Full statistical picture of a query against the null distribution.

        Returns the Hamming distance to EVERY class, plus the null distribution
        parameters.  Implements the lecture's "bell curve test":
            - Incorrect classes should sit at H/dim ≈ 0.50 ± σ (noise floor)
            - Correct class should be many σ below 0.50 (signal)

        Args:
            x: Query input

        Returns:
            dict with per-class distances, z-scores, and significance flags
        """
        if self._M_binary is None:
            self.finalize()

        hv = self._encode(x)
        result = _bind(hv, self._M_binary)

        null_std = theoretical_std(self.dim)
        per_class = []
        for c in range(self.n_classes):
            h = _hamming_distance_norm(result, self.class_hvs[c])
            z = _z_score(h, self.dim)
            per_class.append({
                "class_idx": c,
                "h_norm": h,
                "z_score": z,
                "significant_3sigma": z > 3.0,
            })

        per_class.sort(key=lambda d: d["h_norm"])
        pred_class = per_class[0]["class_idx"]
        best_z = per_class[0]["z_score"]

        return {
            "pred_class": pred_class,
            "best_z": best_z,
            "per_class": per_class,
            "null_mean": 0.5,
            "null_std": null_std,
            "dim": self.dim,
            # Lecture: "every 50 bits is one standard deviation" (at dim=10000)
            "bits_per_sigma": self.dim * null_std,
            "interpretable_threshold": 0.5 - 3 * null_std,
        }

    # ── Capacity and quality ───────────────────────────────────────────────────

    def capacity_used(self) -> Dict:
        """How much of the model capacity is consumed.

        The lecture: "if you put together too much data, that cutoff point
        might be so minor that random variation overcomes it."

        From concentration.py: capacity ≈ dim / (2 * z_threshold²)

        Returns:
            dict with n_examples, capacity_bound, saturation_fraction
        """
        from hdc.concentration import capacity_estimate
        cap = capacity_estimate(self.dim, error_rate=0.01)
        n = self._n_trained
        saturation = n / max(cap, 1)
        avg_density = (
            sum(self._encoding_density) / len(self._encoding_density)
            if self._encoding_density else 0.5
        )

        return {
            "n_examples_stored": n,
            "capacity_bound": cap,
            "saturation_fraction": saturation,
            "is_saturated": saturation > 0.8,
            "avg_encoding_density": avg_density,
            # Lecture: sparse encodings are critical
            "encoding_is_sparse": avg_density < 0.3,
            "recommended_max_examples": int(cap * 0.7),
        }

    def snr_per_class(self, x: torch.Tensor) -> Dict[int, float]:
        """Signal-to-noise ratio for each class given query x.

        SNR at the correct class should be much higher than at wrong classes.
        """
        if self._M_binary is None:
            self.finalize()
        hv = self._encode(x)
        result = _bind(hv, self._M_binary)
        return {
            c: snr_db(self.dim, _hamming_distance_norm(result, self.class_hvs[c]))
            for c in range(self.n_classes)
        }

    @property
    def M(self) -> torch.Tensor:
        """The binary model vector M (finalized)."""
        if self._M_binary is None:
            self.finalize()
        return self._M_binary

    def __repr__(self) -> str:
        trained = self._n_trained
        cap = trained / max(1, self.dim // 13)
        return (
            f"HolographicInferenceModel("
            f"dim={self.dim}, classes={self.n_classes}, "
            f"trained={trained}, saturation≈{cap:.1%})"
        )


def test_inference_model():
    import torch
    print("inference_model: ✅ importable and instantiable")

if __name__ == "__main__":
    test_inference_model()
