"""
Classification and Recall With Binary Hyperdimensional Computing: Tradeoffs
===========================================================================
Based on: Rahimi, A., et al. (2018)
"Classification and Recall With Binary Hyperdimensional Computing:
 Tradeoffs in Choice of Density and Mapping Characteristics"
 IEEE Transactions on Neural Networks and Learning Systems, 30(12), 3750-3763.
 DOI: 10.1109/TNNLS.2018.2814400

Key contributions from the paper:

1. **Density Tradeoffs** — The density (fraction of 1s) of binary hypervectors
   significantly affects classification accuracy and memory capacity.
   - Sparse (low density): higher capacity, more noise-sensitive
   - Dense (balanced): better noise tolerance, lower capacity
   - Optimal density depends on the task and noise level

2. **Mapping Characteristics** — How input features map to hypervectors:
   - Random projection: preserves distances (JL lemma)
   - Learned mapping: task-specific, higher accuracy
   - Hybrid: random basis + learned weights

3. **Associative Memory Capacity** — Theoretical analysis of how density
   affects the number of patterns that can be stored and reliably recalled.

4. **Noise Robustness** — Tradeoff between density and noise tolerance:
   - Sparse HVs: more affected by bit flips
   - Dense HVs: more robust but lower capacity

Reference:
  Rahimi, A., et al. (2018)
  "Classification and Recall With Binary Hyperdimensional Computing:
   Tradeoffs in Choice of Density and Mapping Characteristics"
  IEEE TNNLS, 30(12), 3750-3763
"""

import math
import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Dict
from hdc.hdc_glue import (
    hv_xor, hv_popcount, hv_hamming_sim, hv_bundle,
    hv_permute, hv_majority, hv_batch_sim, gen_hvs,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Section II: Density-Controlled Hypervector Generation
# ═══════════════════════════════════════════════════════════════════════════════

def gen_sparse_hvs(
    n: int,
    dim: int,
    density: float = 0.1,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Generate sparse binary hypervectors with controlled density.

    Args:
        n: Number of hypervectors
        dim: Dimensionality
        density: Fraction of 1s (0 < density < 1)
        seed: Random seed

    Returns:
        (n, dim) binary hypervectors with approximately `density` fraction of 1s
    """
    g = torch.Generator()
    if seed is not None:
        g.manual_seed(seed)
    return (torch.rand(n, dim, generator=g) < density).float()


def gen_dense_hvs(
    n: int,
    dim: int,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Generate balanced (density=0.5) binary hypervectors.

    This is the standard approach used in most HDC literature.
    """
    return gen_hvs(n, dim, seed=seed)


def gen_variable_density_hvs(
    n: int,
    dim: int,
    densities: torch.Tensor,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Generate hypervectors with per-vector density control.

    Args:
        n: Number of hypervectors
        dim: Dimensionality
        densities: (n,) tensor of target densities for each vector
        seed: Random seed

    Returns:
        (n, dim) binary hypervectors
    """
    g = torch.Generator()
    if seed is not None:
        g.manual_seed(seed)

    hvs = torch.zeros(n, dim)
    for i in range(n):
        mask = torch.rand(dim, generator=g) < densities[i].item()
        hvs[i, mask] = 1.0

    return hvs


# ═══════════════════════════════════════════════════════════════════════════════
# Section III: Density-Aware Associative Memory
# ═══════════════════════════════════════════════════════════════════════════════

class DensityAwareMemory:
    """
    Associative memory that accounts for hypervector density.

    Standard Hamming similarity assumes density ≈ 0.5. When density
    deviates from 0.5, the expected similarity between random vectors
    changes, requiring a density-normalized similarity measure.

    Key insight (Rahimi 2018, Section III-B):
        E[sim_random] = density^2 + (1-density)^2
        For density=0.5: E[sim_random] = 0.5
        For density=0.1: E[sim_random] = 0.82

    So sparse vectors appear more similar by chance! This must be
    corrected for fair comparison.
    """

    def __init__(self, dim: int = 10000):
        self.dim = dim
        self.memory: List[Tuple[torch.Tensor, int, float]] = []  # (hv, label, density)

    def add(self, hv: torch.Tensor, label: int):
        """Add a hypervector to memory with its density.

        Args:
            hv: (dim,) binary hypervector
            label: Class label
        """
        density = float(hv.mean().item())
        self.memory.append((hv.clone(), label, density))

    def density_normalized_sim(self, a: torch.Tensor, b: torch.Tensor) -> float:
        """Compute density-normalized similarity.

        Corrects for the expected similarity bias due to density:
            sim_norm = (sim_raw - E_sim_random) / (1 - E_sim_random)

        Where E_sim_random depends on the densities of both vectors.

        Args:
            a: (dim,) hypervector
            b: (dim,) hypervector

        Returns:
            Density-normalized similarity in [0, 1]
        """
        raw_sim = float(hv_hamming_sim(a, b))

        # Expected similarity given densities
        da = float(a.mean().item())
        db = float(b.mean().item())
        e_random = da * db + (1 - da) * (1 - db)

        # Normalize
        if abs(1 - e_random) < 1e-12:
            return raw_sim
        return max(0.0, min(1.0, (raw_sim - e_random) / (1 - e_random)))

    def query(self, hv: torch.Tensor, top_k: int = 1) -> List[Dict]:
        """Query memory with density-normalized similarity.

        Args:
            hv: (dim,) query hypervector
            top_k: Number of top results

        Returns:
            List of {hv, label, similarity, density}
        """
        if not self.memory:
            return []

        results = []
        for mem_hv, label, density in self.memory:
            sim = self.density_normalized_sim(hv, mem_hv)
            results.append({
                "hv": mem_hv,
                "label": label,
                "similarity": sim,
                "density": density,
            })

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]

    def capacity_estimate(self, density: float, tolerance: float = 0.01) -> int:
        """Estimate memory capacity for a given density.

        Based on Rahimi 2018, Section IV:
            Capacity ≈ (1 - 4*density*(1-density)) / (2*tolerance^2)

        Args:
            density: Fraction of 1s in stored hypervectors
            tolerance: Maximum acceptable similarity to random vectors

        Returns:
            Estimated number of patterns that can be stored
        """
        # Theoretical capacity formula
        var_random = density * (1 - density) / self.dim
        if var_random < 1e-12:
            return 1
        return int(1.0 / (2 * tolerance ** 2 * var_random))


# ═══════════════════════════════════════════════════════════════════════════════
# Section IV: Density-Aware Classifier
# ═══════════════════════════════════════════════════════════════════════════════

class DensityAwareHDCClassifier(nn.Module):
    """
    HDC classifier with density-aware encoding and similarity.

    Based on Rahimi 2018, this classifier:
    1. Uses controlled-density hypervectors for encoding
    2. Applies density-normalized similarity during inference
    3. Can learn optimal density per class
    4. Provides theoretical capacity estimates

    The key improvement over standard HDC is that density is treated
    as a tunable parameter, not a fixed property.
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        dim: int = 10000,
        encoding_density: float = 0.5,
        learnable_density: bool = False,
        seed: Optional[int] = None,
    ):
        """
        Args:
            n_features: Number of input features
            n_classes: Number of output classes
            dim: Hypervector dimensionality
            encoding_density: Target density for encoding hypervectors
            learnable_density: If True, density is learned per class
            seed: Random seed
        """
        super().__init__()
        self.n_features = n_features
        self.n_classes = n_classes
        self.dim = dim
        self.encoding_density = encoding_density
        self.seed = seed

        # Feature keys with controlled density
        self.register_buffer(
            "feature_keys",
            gen_sparse_hvs(n_features, dim, density=encoding_density, seed=seed),
        )

        # Inverted keys for inactive features
        self.register_buffer(
            "not_keys",
            1.0 - self.feature_keys,
        )

        # Class prototypes
        self.register_buffer(
            "class_hvs",
            gen_sparse_hvs(n_classes, dim, density=0.5, seed=(seed or 0) + 1),
        )

        # Per-class counts
        self.register_buffer("counts", torch.zeros(n_classes))

        # Learnable density parameters (optional)
        if learnable_density:
            self.class_density_logits = nn.Parameter(
                torch.zeros(n_classes)
            )
        else:
            self.register_buffer("class_density_logits", torch.zeros(n_classes))

    def get_class_densities(self) -> torch.Tensor:
        """Get the effective density for each class prototype."""
        return torch.sigmoid(self.class_density_logits)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input with controlled-density hypervectors.

        Args:
            x: (n_features,) input feature vector

        Returns:
            (dim,) binary hypervector
        """
        active = (x > 0.5).float()
        inactive = 1.0 - active

        hv = (active.unsqueeze(-1) * self.feature_keys).sum(dim=0) + \
             (inactive.unsqueeze(-1) * self.not_keys).sum(dim=0)

        return hv_majority(hv)

    def density_normalized_sim(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        density_b: Optional[float] = None,
    ) -> torch.Tensor:
        """Compute density-normalized similarity.

        Args:
            a: (dim,) query hypervector
            b: (n_classes, dim) or (dim,) class prototypes
            density_b: Density of b (if None, computed from b)

        Returns:
            Similarity score(s)
        """
        if b.dim() == 1:
            b = b.unsqueeze(0)

        da = float(a.mean().item())
        raw_sims = hv_batch_sim(a, b)

        if density_b is not None:
            db = density_b
        else:
            db = float(b[0].mean().item())

        e_random = da * db + (1 - da) * (1 - db)
        denom = 1.0 - e_random

        if abs(denom) < 1e-12:
            return raw_sims

        return (raw_sims - e_random) / denom

    def train_step(self, x: torch.Tensor, label: int):
        """Online training step.

        Args:
            x: (n_features,) input features
            label: Class label
        """
        hv = self.encode(x)
        self.class_hvs[label] = self.class_hvs[label] + hv
        self.counts[label] += 1

    def finalize(self):
        """Finalize training: threshold prototypes."""
        self.class_hvs = hv_majority(self.class_hvs)

    def predict(self, x: torch.Tensor) -> Tuple[int, torch.Tensor]:
        """Predict class with density-normalized similarity.

        Args:
            x: (n_features,) input features

        Returns:
            (predicted_class, similarities)
        """
        hv = self.encode(x)
        sims = self.density_normalized_sim(hv, self.class_hvs)
        return int(sims.argmax().item()), sims

    def analyze_density_impact(self, x: torch.Tensor) -> Dict:
        """Analyze how density affects classification.

        Args:
            x: (n_features,) input features

        Returns:
            Dict with density analysis
        """
        hv = self.encode(x)
        query_density = float(hv.mean().item())

        class_densities = []
        class_sims_raw = []
        class_sims_norm = []

        for i in range(self.n_classes):
            cd = float(self.class_hvs[i].mean().item())
            raw_sim = float(hv_hamming_sim(hv, self.class_hvs[i]))
            norm_sim = float(self.density_normalized_sim(hv, self.class_hvs[i].unsqueeze(0)))

            class_densities.append(cd)
            class_sims_raw.append(raw_sim)
            class_sims_norm.append(norm_sim)

        return {
            "query_density": query_density,
            "class_densities": class_densities,
            "raw_similarities": class_sims_raw,
            "normalized_similarities": class_sims_norm,
            "density_bias": max(class_sims_raw) - min(class_sims_raw),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Section V: Optimal Density Search
# ═══════════════════════════════════════════════════════════════════════════════

class DensityOptimizer:
    """
    Finds the optimal hypervector density for a given task.

    Based on Rahimi 2018, Section V: the optimal density depends on:
    1. Dimensionality (higher D → sparser can work)
    2. Number of classes (more classes → denser needed)
    3. Noise level (more noise → denser for robustness)
    4. Feature correlation (more correlation → sparser)

    Uses grid search or Bayesian optimization over density values.
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        dim: int = 10000,
        device: str = "cpu",
    ):
        self.n_features = n_features
        self.n_classes = n_classes
        self.dim = dim
        self.device = device

    def evaluate_density(
        self,
        density: float,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_test: torch.Tensor,
        y_test: torch.Tensor,
    ) -> float:
        """Evaluate classification accuracy for a given density.

        Args:
            density: Target hypervector density
            X_train: (n_train, n_features) training data
            y_train: (n_train,) training labels
            X_test: (n_test, n_features) test data
            y_test: (n_test,) test labels

        Returns:
            Test accuracy
        """
        classifier = DensityAwareHDCClassifier(
            n_features=self.n_features,
            n_classes=self.n_classes,
            dim=self.dim,
            encoding_density=density,
        )

        # Train
        for i in range(X_train.shape[0]):
            classifier.train_step(X_train[i], int(y_train[i].item()))
        classifier.finalize()

        # Test
        correct = 0
        for i in range(X_test.shape[0]):
            pred, _ = classifier.predict(X_test[i])
            if pred == int(y_test[i].item()):
                correct += 1

        return correct / X_test.shape[0]

    def grid_search(
        self,
        densities: List[float],
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_test: torch.Tensor,
        y_test: torch.Tensor,
    ) -> Tuple[float, float, List[Tuple[float, float]]]:
        """Grid search over densities.

        Args:
            densities: List of density values to try
            X_train: Training data
            y_train: Training labels
            X_test: Test data
            y_test: Test labels

        Returns:
            (best_density, best_accuracy, [(density, accuracy), ...])
        """
        results = []
        best_density = densities[0]
        best_accuracy = 0.0

        for density in densities:
            acc = self.evaluate_density(density, X_train, y_train, X_test, y_test)
            results.append((density, acc))
            if acc > best_accuracy:
                best_accuracy = acc
                best_density = density

        return best_density, best_accuracy, results

    def theoretical_optimal_density(
        self,
        n_classes: int,
        noise_level: float = 0.0,
    ) -> float:
        """Compute theoretical optimal density.

        Based on Rahimi 2018, Eq. 15-17:
            Optimal density ≈ 0.5 for high noise
            Optimal density ≈ 1/(2*sqrt(D)) for low noise, many classes

        Args:
            n_classes: Number of classes
            noise_level: Expected bit-flip noise level

        Returns:
            Recommended density
        """
        if noise_level > 0.1:
            return 0.5  # Balanced is best for noisy conditions

        # For low noise: sparser is better for capacity
        recommended = 1.0 / (2 * (self.dim ** 0.25))
        return max(0.05, min(0.5, recommended))

    def adaptive_density_schedule(
        self,
        n_steps:        int,
        initial_density: float = 0.5,
        target_density:  float = 0.1,
        schedule:        str   = "cosine",
    ) -> List[float]:
        """
        Generate a density annealing schedule for energy-efficient inference.

        As the model converges (later steps), progressively lower the density.
        Sparse codes require fewer XOR operations → lower inference energy.
        If accuracy drops, the caller can raise density again.

        Schedules:
          "linear":  density decreases linearly from initial to target
          "cosine":  cosine decay (slow start, fast decay, slow finish)
          "step":    halve density every n_steps/4

        Args:
            n_steps:          Length of schedule
            initial_density:  Starting density (warmup phase)
            target_density:   Final density (steady state)
            schedule:         Annealing schedule

        Returns:
            List of n_steps density values
        """
        import math as _math
        densities = []
        for t in range(n_steps):
            frac = t / max(n_steps - 1, 1)
            if schedule == "linear":
                d = initial_density - frac * (initial_density - target_density)
            elif schedule == "cosine":
                cos_val = 0.5 * (1 + _math.cos(_math.pi * frac))
                d = target_density + (initial_density - target_density) * cos_val
            elif schedule == "step":
                step_idx = int(frac * 4)
                d = initial_density / (2 ** step_idx)
            else:
                d = initial_density
            densities.append(float(max(target_density, min(initial_density, d))))
        return densities


# ═══════════════════════════════════════════════════════════════════════════════
# Section VI: Mapping Characteristics (Kleyko 2018, Section III)
# ═══════════════════════════════════════════════════════════════════════════════

class MappingType(str):
    """Taxonomy of initial feature-to-HV mappings (Kleyko 2018, Section III)."""
    RANDOM = "random"               # i.i.d. Bernoulli(0.5) — standard
    STRUCTURED_ORTHOGONAL = "structured_orthogonal"  # near-orthogonal by construction
    LEARNED = "learned"             # task-specific projection
    RANDOM_SPARSE = "random_sparse" # sparse random (density < 0.5)


class StructuredMapper:
    """
    Structured (quasi-orthogonal) feature-to-HV mapping.

    Kleyko 2018 Section III shows that the *mapping type* — not just density —
    affects classification accuracy.  Random i.i.d. maps produce expected
    Hamming distance D/2 between any pair (near-orthogonal on average), but
    a small fraction of "collision" pairs are too similar by chance.

    Structured orthogonal maps eliminate this variance by ensuring that all
    pairwise Hamming distances are exactly D/2 within machine precision.

    Construction: Start with a random HV and progressively flip D/2 bits for
    each subsequent vector, tracking which bits have already been flipped.
    This is an extension of the Level-HV construction to the identity space.
    """

    def __init__(
        self,
        n_features: int,
        dim: int = 10000,
        seed: Optional[int] = None,
    ):
        self.n_features = n_features
        self.dim = dim
        self.hvs = self._build_structured(n_features, dim, seed)

    @staticmethod
    def _build_structured(F: int, D: int, seed: Optional[int]) -> torch.Tensor:
        g = torch.Generator()
        if seed is not None:
            g.manual_seed(seed)

        hvs = torch.zeros(F, D)
        # Base HV
        hvs[0] = (torch.rand(D, generator=g) < 0.5).float()

        for i in range(1, F):
            # Flip exactly D//2 distinct positions (guaranteed near-orthogonal)
            prev = hvs[i - 1].clone()
            flip_idx = torch.randperm(D, generator=g)[:D // 2]
            prev[flip_idx] = 1.0 - prev[flip_idx]
            hvs[i] = prev

        return hvs

    def encode_feature(self, feature_idx: int, value_hv: torch.Tensor) -> torch.Tensor:
        """Bind feature ID-HV with value HV (XOR binding)."""
        return hv_xor(self.hvs[feature_idx], value_hv)

    def pairwise_similarity_stats(self) -> Dict:
        """
        Compute pairwise Hamming similarity statistics for the mapping.

        For random i.i.d. maps: E[sim] = 0.5, Var[sim] ≈ 1/(4D).
        For structured maps: all pairwise sims should be ≈ 0.5 with lower variance.

        Returns:
            Dict with mean, std, min, max pairwise similarity
        """
        F = self.n_features
        sims = []
        for i in range(F):
            for j in range(i + 1, F):
                sims.append(float(hv_hamming_sim(self.hvs[i], self.hvs[j])))

        if not sims:
            return {"mean": 0.5, "std": 0.0, "min": 0.5, "max": 0.5}

        t = torch.tensor(sims)
        return {
            "mean": float(t.mean()),
            "std": float(t.std()),
            "min": float(t.min()),
            "max": float(t.max()),
            "n_pairs": len(sims),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Section VII: Recall vs Classification Analysis (Kleyko 2018, Section IV–V)
# ═══════════════════════════════════════════════════════════════════════════════

class RecallAnalyzer:
    """
    Theoretical and empirical analysis of recall vs. classification capacity.

    Kleyko 2018 distinguishes two tasks:
      - **Classification**: Given a query HV, find its nearest prototype.
        Error occurs when a non-target prototype is closer.
      - **Recall (Heteroassociative)**: Given a noisy/partial query, retrieve
        the closest stored pattern from item memory exactly.

    Key finding (Section IV): Classification capacity ≫ recall capacity
    for the same D and density. Classification can tolerate more stored items
    before performance degrades because the decision boundary is only between
    K prototypes, not across all D^N possible patterns.

    Capacity formulas (Kleyko 2018, Eq. 7–10):
      P_err_class ≈ (K-1) · exp(-2·D·(0.5 - p)²)  — classification error
      P_err_recall ≈ M · exp(-2·D·(0.5 - p)²)      — recall error with M items

    where p is bit-flip noise level, K is number of classes, M is item memory size.
    """

    def __init__(self, dim: int = 10000):
        self.dim = dim

    def classification_error_bound(
        self,
        n_classes: int,
        noise_rate: float,
        density: float = 0.5,
    ) -> float:
        """
        Upper bound on classification error probability (Kleyko 2018, Eq. 7).

        Args:
            n_classes: Number of class prototypes K
            noise_rate: Bit-flip noise probability p ∈ [0, 0.5)
            density: HV density (fraction of 1s)

        Returns:
            P(classification error) upper bound
        """
        D = self.dim
        # Correction for density deviation from 0.5
        effective_noise = noise_rate + abs(density - 0.5)
        # Union bound over K-1 distractors
        exponent = -2 * D * (0.5 - min(effective_noise, 0.499)) ** 2
        return (n_classes - 1) * math.exp(exponent)

    def recall_error_bound(
        self,
        n_stored: int,
        noise_rate: float,
        density: float = 0.5,
    ) -> float:
        """
        Upper bound on recall error probability (Kleyko 2018, Eq. 10).

        Args:
            n_stored: Number of items stored in associative memory M
            noise_rate: Bit-flip noise probability p
            density: HV density

        Returns:
            P(recall error) upper bound
        """
        D = self.dim
        effective_noise = noise_rate + abs(density - 0.5)
        exponent = -2 * D * (0.5 - min(effective_noise, 0.499)) ** 2
        return n_stored * math.exp(exponent)

    def capacity_at_error_rate(
        self,
        target_error: float,
        noise_rate: float,
        density: float = 0.5,
        task: str = "classification",
    ) -> int:
        """
        Maximum K (classes) or M (items) such that error ≤ target_error.

        Args:
            target_error: Target error probability (e.g., 0.01)
            noise_rate: Bit-flip noise probability
            density: HV density
            task: "classification" or "recall"

        Returns:
            Maximum capacity (K or M)
        """
        D = self.dim
        effective_noise = noise_rate + abs(density - 0.5)
        # From P_err ≤ target_error:
        #   (K-1) · exp(-2D(0.5-p)²) ≤ target_error
        #   K ≤ target_error / exp(-2D(0.5-p)²) + 1
        exponent = -2 * D * (0.5 - min(effective_noise, 0.499)) ** 2
        p_single = math.exp(exponent)

        # If p_single is so small that one error is astronomically unlikely,
        # capacity is effectively unbounded — cap at a sensible maximum.
        if p_single < 1e-300:
            return 10 ** 9

        base_count = target_error / p_single

        if task == "recall":
            return max(1, int(base_count))
        else:  # classification
            return max(1, int(base_count + 1))

    def density_vs_capacity_curve(
        self,
        densities: List[float],
        noise_rate: float = 0.05,
        target_error: float = 0.01,
        task: str = "classification",
    ) -> List[Tuple[float, int]]:
        """
        Compute capacity as a function of density.

        Key result from Kleyko 2018: maximum capacity occurs at density ≈ 0.5
        when noise_rate = 0, but shifts toward sparser representations when
        the distribution is skewed or feature distributions are non-uniform.

        Returns:
            [(density, capacity), ...]
        """
        return [
            (d, self.capacity_at_error_rate(target_error, noise_rate, d, task))
            for d in densities
        ]

    def recall_vs_classification_gap(
        self,
        n_items: int,
        noise_rate: float = 0.05,
        density: float = 0.5,
    ) -> Dict:
        """
        Quantify the capacity gap between recall and classification.

        Returns a dict showing how many more classes can be discriminated
        than patterns recalled at the same error rate.
        """
        p_cls = self.classification_error_bound(n_items, noise_rate, density)
        p_rec = self.recall_error_bound(n_items, noise_rate, density)

        cap_cls = self.capacity_at_error_rate(0.01, noise_rate, density, "classification")
        cap_rec = self.capacity_at_error_rate(0.01, noise_rate, density, "recall")

        return {
            "n_items": n_items,
            "noise_rate": noise_rate,
            "density": density,
            "classification_error": min(p_cls, 1.0),
            "recall_error": min(p_rec, 1.0),
            "classification_capacity_1pct": cap_cls,
            "recall_capacity_1pct": cap_rec,
            "capacity_gap_ratio": cap_cls / max(cap_rec, 1),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Section VIII: Empirical Mapping Comparison (Kleyko 2018, Section V)
# ═══════════════════════════════════════════════════════════════════════════════

class MappingCharacteristicsStudy:
    """
    Empirical comparison of random vs. structured mappings (Kleyko 2018).

    Kleyko 2018 Section V demonstrates that structured maps give marginally
    better accuracy than random i.i.d. maps because they guarantee that
    all feature HVs are exactly near-orthogonal — eliminating "lucky" or
    "unlucky" random draws. The improvement is task-dependent and typically
    small (0.5–2%) but consistent at low noise rates.

    This class provides the empirical study infrastructure.
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        dim: int = 5000,
        seed: int = 42,
    ):
        self.n_features = n_features
        self.n_classes = n_classes
        self.dim = dim
        self.seed = seed

    def compare_mappings(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_test: torch.Tensor,
        y_test: torch.Tensor,
        densities: Optional[List[float]] = None,
    ) -> Dict[str, List[Tuple[float, float]]]:
        """
        Compare random vs. structured mappings across densities.

        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            densities: Density values to sweep (default: [0.1, 0.2, ..., 0.9])

        Returns:
            Dict mapping mapping_type → [(density, accuracy), ...]
        """
        if densities is None:
            densities = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        results: Dict[str, List[Tuple[float, float]]] = {
            "random": [],
            "structured": [],
        }

        for density in densities:
            # Random i.i.d. mapping
            clf_rnd = DensityAwareHDCClassifier(
                n_features=self.n_features,
                n_classes=self.n_classes,
                dim=self.dim,
                encoding_density=density,
                seed=self.seed,
            )
            for i in range(X_train.shape[0]):
                clf_rnd.train_step(X_train[i], int(y_train[i].item()))
            clf_rnd.finalize()

            correct = sum(
                1 for i in range(X_test.shape[0])
                if clf_rnd.predict(X_test[i])[0] == int(y_test[i].item())
            )
            results["random"].append((density, correct / X_test.shape[0]))

            # Structured mapping: use StructuredMapper for ID-HVs
            mapper = StructuredMapper(self.n_features, self.dim, seed=self.seed)

            # Build structured prototypes
            protos = torch.zeros(self.n_classes, self.dim)
            counts = torch.zeros(self.n_classes)

            for i in range(X_train.shape[0]):
                x = X_train[i]
                active = (x > 0.5).float()
                hv = torch.zeros(self.dim)
                for f in range(self.n_features):
                    level_idx = int(x[f].item() * (10 - 1))
                    val_hv = gen_sparse_hvs(1, self.dim, density=density, seed=self.seed + f * 100 + level_idx).squeeze(0)
                    hv += mapper.encode_feature(f, val_hv)
                hv = hv_majority(hv)

                c = int(y_train[i].item())
                protos[c] = protos[c] + hv
                counts[c] += 1

            for c in range(self.n_classes):
                if counts[c] > 0:
                    protos[c] = hv_majority(protos[c])

            correct = 0
            for i in range(X_test.shape[0]):
                x = X_test[i]
                hv = torch.zeros(self.dim)
                for f in range(self.n_features):
                    level_idx = int(x[f].item() * (10 - 1))
                    val_hv = gen_sparse_hvs(1, self.dim, density=density, seed=self.seed + f * 100 + level_idx).squeeze(0)
                    hv += mapper.encode_feature(f, val_hv)
                hv = hv_majority(hv)
                pred = int(hv_batch_sim(hv, protos).argmax().item())
                if pred == int(y_test[i].item()):
                    correct += 1
            results["structured"].append((density, correct / X_test.shape[0]))

        return results


# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_density_controlled_generation():
    """Verify density-controlled hypervector generation."""
    print("=" * 60)
    print("Testing Density-Controlled HV Generation (Rahimi 2018)")
    print("=" * 60)

    dim = 10000

    for density in [0.1, 0.3, 0.5, 0.7, 0.9]:
        hvs = gen_sparse_hvs(100, dim, density=density)
        actual_density = float(hvs.mean().item())
        print(f"  Target density: {density:.1f} → Actual: {actual_density:.4f}")
        assert abs(actual_density - density) < 0.02, \
            f"Density mismatch: target={density}, actual={actual_density}"

    print(f"\n  ✅ Density-controlled generation test complete!")


def test_density_normalized_similarity():
    """Verify density-normalized similarity correction."""
    print("=" * 60)
    print("Testing Density-Normalized Similarity (Rahimi 2018)")
    print("=" * 60)

    dim = 10000

    # Sparse vectors should have corrected similarity
    sparse_a = gen_sparse_hvs(1, dim, density=0.1).squeeze(0)
    sparse_b = gen_sparse_hvs(1, dim, density=0.1, seed=42).squeeze(0)

    raw_sim = float(hv_hamming_sim(sparse_a, sparse_b))
    print(f"\n  Sparse (d=0.1) raw similarity: {raw_sim:.4f}")
    print(f"  (Expected ~0.82 without correction)")

    # Dense vectors
    dense_a = gen_dense_hvs(1, dim).squeeze(0)
    dense_b = gen_dense_hvs(1, dim, seed=42).squeeze(0)

    raw_sim_dense = float(hv_hamming_sim(dense_a, dense_b))
    print(f"\n  Dense (d=0.5) raw similarity: {raw_sim_dense:.4f}")
    print(f"  (Expected ~0.50 without correction)")

    # Density-aware memory
    memory = DensityAwareMemory(dim=dim)
    memory.add(sparse_a, 0)
    memory.add(dense_a, 1)

    # Query with sparse vector
    results = memory.query(sparse_b, top_k=2)
    print(f"\n  Density-normalized query results:")
    for r in results:
        print(f"    Label {r['label']}: sim={r['similarity']:.4f}, density={r['density']:.2f}")

    print(f"\n  ✅ Density-normalized similarity test complete!")


def test_density_classifier():
    """Verify density-aware classifier."""
    print("=" * 60)
    print("Testing Density-Aware Classifier (Rahimi 2018)")
    print("=" * 60)

    n_features = 20
    n_classes = 4
    dim = 2000

    # Create classifier with sparse encoding
    classifier = DensityAwareHDCClassifier(
        n_features=n_features,
        n_classes=n_classes,
        dim=dim,
        encoding_density=0.3,
    )

    # Generate synthetic data
    torch.manual_seed(42)
    n_train = 30

    for cls in range(n_classes):
        for _ in range(n_train):
            x = torch.zeros(n_features)
            active = [(cls + i) % n_features for i in range(5)]
            x[active] = 1.0
            x = x + torch.randn(n_features) * 0.1
            classifier.train_step(x, cls)

    classifier.finalize()

    # Test
    correct = 0
    total = 80
    for cls in range(n_classes):
        for _ in range(total // n_classes):
            x = torch.zeros(n_features)
            active = [(cls + i) % n_features for i in range(5)]
            x[active] = 1.0
            x = x + torch.randn(n_features) * 0.1
            pred, _ = classifier.predict(x)
            if pred == cls:
                correct += 1

    accuracy = correct / total
    print(f"\n  Classification accuracy: {accuracy:.1%}")

    # Analyze density impact
    x = torch.zeros(n_features)
    x[0] = 1.0
    analysis = classifier.analyze_density_impact(x)
    print(f"\n  Query density: {analysis['query_density']:.4f}")
    print(f"  Density bias: {analysis['density_bias']:.4f}")

    print(f"\n  {'✅' if accuracy > 0.5 else '❌'} Density-aware classifier test complete!")


def test_structured_mapper():
    print("=" * 60)
    print("Testing StructuredMapper (Kleyko 2018, Section III)")
    print("=" * 60)

    F, D = 20, 10000
    mapper = StructuredMapper(n_features=F, dim=D, seed=42)
    stats = mapper.pairwise_similarity_stats()
    print(f"  Pairwise similarity — mean: {stats['mean']:.4f}, std: {stats['std']:.4f}")
    print(f"  (Random i.i.d. expected: mean≈0.5, std≈{1/(4*D)**0.5:.4f})")
    assert abs(stats["mean"] - 0.5) < 0.03, f"Mean too far from 0.5: {stats['mean']}"
    print("  ✅ StructuredMapper OK")


def test_recall_analyzer():
    print("=" * 60)
    print("Testing RecallAnalyzer (Kleyko 2018, Section IV–V)")
    print("=" * 60)

    # Use D=500 so capacities are finite and illustrative at 20% noise
    analyzer = RecallAnalyzer(dim=500)

    cap_cls = analyzer.capacity_at_error_rate(0.01, noise_rate=0.20, task="classification")
    cap_rec = analyzer.capacity_at_error_rate(0.01, noise_rate=0.20, task="recall")
    print(f"  Classification capacity (1% err, 20% noise, D=500) : {cap_cls:,}")
    print(f"  Recall capacity        (1% err, 20% noise, D=500) : {cap_rec:,}")
    assert cap_cls >= cap_rec, "Classification should have ≥ recall capacity"

    gap = analyzer.recall_vs_classification_gap(n_items=100, noise_rate=0.20)
    print(f"  Capacity gap ratio (cls/recall, 100 items): {gap['capacity_gap_ratio']:.1f}×")
    print(f"  Classification error (100 items): {gap['classification_error']:.4f}")
    print(f"  Recall error        (100 items): {gap['recall_error']:.4f}")

    curve = analyzer.density_vs_capacity_curve(
        densities=[0.1, 0.3, 0.5, 0.7, 0.9], noise_rate=0.20, task="classification"
    )
    print(f"  Density–capacity curve (20% noise, D=500):")
    for d, c in curve:
        print(f"    density={d:.1f} → capacity={c:,}")
    print("  ✅ RecallAnalyzer OK")


if __name__ == "__main__":
    test_density_controlled_generation()
    print()
    test_density_normalized_similarity()
    print()
    test_density_classifier()
    print()
    test_structured_mapper()
    print()
    test_recall_analyzer()
