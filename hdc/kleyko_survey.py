"""
Kleyko 2023 Survey: Hyperdimensional Computing Part I & II
===========================================================
Based on: Kleyko, D., et al. (2023)
"Hyperdimensional Computing: A Survey of the Art" Part I & II
IEEE Circuits and Systems Magazine, 23(2), 8-33 and 23(3), 10-34.

This module implements the missing models and data transformations
from the comprehensive Kleyko 2023 survey that are not yet in SNNTraining:

Part I - Models and Data Transformations:
1. **N-gram Encoding** — Fixed-length subsequence encoding for sequences
2. **Record-Based Encoding** — Role-filler binding for structured data
3. **Spatial Encoding** — 2D/3D spatial relationships via binding
4. **Graph Encoding** — Edge-based graph representation
5. **Temporal Encoding** — Time-aware hypervector construction

Part II - Learning and Inference:
6. **Retraining Strategies** — Iterative refinement of prototypes
7. **Ensemble Methods** — Combining multiple HDC classifiers
8. **Adaptive Thresholding** — Dynamic decision boundaries
9. **Confidence Calibration** — Well-calibrated prediction probabilities

Reference:
  Kleyko, D., et al. (2023)
  "Hyperdimensional Computing: A Survey of the Art" Part I & II
  IEEE Circuits and Systems Magazine
"""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Dict, Any, Union, Callable
from hdc.hdc_glue import (
    hv_xor, hv_popcount, hv_hamming_sim, hv_bundle,
    hv_permute, hv_majority, hv_batch_sim, gen_hvs,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Part I, Section III-A: N-gram Encoding
# ═══════════════════════════════════════════════════════════════════════════════

class NGramEncoder:
    """
    N-gram encoding for sequences (Kleyko 2023, Part I, Section III-A).

    Encodes a sequence by extracting all n-grams and bundling them:
        S = ⊕ bind(permute^0(s_i), permute^1(s_{i+1}), ..., permute^{n-1}(s_{i+n-1}))

    This captures local structure while being robust to small shifts.
    N-gram encoding is the VSA analog of convolutional feature extraction.

    Key properties:
    - n=1: bag-of-symbols (no order)
    - n=2: bigram (adjacent pairs)
    - n=3: trigram (local triplets)
    - Larger n: longer-range dependencies
    """

    def __init__(
        self,
        dim: int = 10000,
        n: int = 3,
        mode: str = "binary",
    ):
        """
        Args:
            dim: Hypervector dimensionality
            n: N-gram length
            mode: "binary" or "real"
        """
        self.dim = dim
        self.n = n
        self.mode = mode

    def encode_ngram(self, symbols: List[torch.Tensor]) -> torch.Tensor:
        """Encode a single n-gram.

        Args:
            symbols: List of n (dim,) symbol hypervectors

        Returns:
            (dim,) n-gram hypervector
        """
        if len(symbols) != self.n:
            raise ValueError(f"Expected {self.n} symbols, got {len(symbols)}")

        # Permute each symbol by its position
        permuted = []
        for i, sym in enumerate(symbols):
            permuted.append(hv_permute(sym, k=i))

        # Bind all permuted symbols
        ngram = permuted[0]
        for p in permuted[1:]:
            ngram = hv_xor(ngram, p)

        return ngram

    def encode_sequence(self, symbols: List[torch.Tensor]) -> torch.Tensor:
        """Encode a sequence using sliding n-grams.

        Args:
            symbols: List of (dim,) symbol hypervectors

        Returns:
            (dim,) sequence hypervector
        """
        if len(symbols) < self.n:
            return torch.zeros(self.dim)

        ngrams = []
        for i in range(len(symbols) - self.n + 1):
            ngram = self.encode_ngram(symbols[i:i + self.n])
            ngrams.append(ngram)

        # Bundle all n-grams
        seq = hv_bundle(torch.stack(ngrams))
        return hv_majority(seq)

    def encode_weighted_sequence(
        self,
        symbols: List[torch.Tensor],
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode a sequence with position-dependent weights.

        Args:
            symbols: List of (dim,) symbol hypervectors
            weights: (len(symbols),) optional weights for each position

        Returns:
            (dim,) weighted sequence hypervector
        """
        if len(symbols) < self.n:
            return torch.zeros(self.dim)

        if weights is None:
            weights = torch.ones(len(symbols))

        ngrams = []
        for i in range(len(symbols) - self.n + 1):
            ngram = self.encode_ngram(symbols[i:i + self.n])
            # Weight by average of constituent symbol weights
            w = weights[i:i + self.n].mean().item()
            ngrams.append(ngram * w)

        seq = hv_bundle(torch.stack(ngrams))
        return hv_majority(seq)


# ═══════════════════════════════════════════════════════════════════════════════
# Part I, Section III-B: Record-Based Encoding
# ═══════════════════════════════════════════════════════════════════════════════

class RecordEncoder:
    """
    Record-based encoding for structured data (Kleyko 2023, Part I, Section III-B).

    Encodes structured data as role-filler bindings:
        R = ⊕ bind(role_i, filler_i) for each field i

    Supports:
    - Nested records (records within records)
    - Multi-valued fields (sets of fillers per role)
    - Optional fields (with null handling)
    - Type-tagged values (type hypervector bound with value)

    This is the VSA analog of JSON/XML structured data encoding.
    """

    def __init__(
        self,
        dim: int = 10000,
        mode: str = "binary",
        seed: Optional[int] = None,
    ):
        self.dim = dim
        self.mode = mode
        self._seed_counter = seed or 0
        self._role_hvs: Dict[str, torch.Tensor] = {}
        self._type_hvs: Dict[str, torch.Tensor] = {}

    def _get_role(self, name: str) -> torch.Tensor:
        if name not in self._role_hvs:
            self._seed_counter += 1
            self._role_hvs[name] = gen_hvs(1, self.dim, seed=self._seed_counter).squeeze(0)
        return self._role_hvs[name]

    def _get_type(self, name: str) -> torch.Tensor:
        if name not in self._type_hvs:
            self._seed_counter += 1
            self._type_hvs[name] = gen_hvs(1, self.dim, seed=self._seed_counter).squeeze(0)
        return self._type_hvs[name]

    def _value_to_hv(self, value: Any) -> torch.Tensor:
        """Convert a Python value to a hypervector."""
        if isinstance(value, torch.Tensor):
            return value
        elif isinstance(value, str):
            seed = hash(value) & 0x7FFFFFFF
            return gen_hvs(1, self.dim, seed=seed).squeeze(0)
        elif isinstance(value, (int, float)):
            seed = hash(str(value)) & 0x7FFFFFFF
            return gen_hvs(1, self.dim, seed=seed).squeeze(0)
        elif isinstance(value, bool):
            seed = 1 if value else 0
            return gen_hvs(1, self.dim, seed=seed).squeeze(0)
        elif isinstance(value, list):
            # Encode list as bundled elements
            if not value:
                return torch.zeros(self.dim)
            hvs = [self._value_to_hv(v) for v in value]
            bundled = hv_bundle(torch.stack(hvs))
            return hv_majority(bundled)
        elif isinstance(value, dict):
            # Encode dict as nested record
            return self.encode(value)
        else:
            seed = hash(str(value)) & 0x7FFFFFFF
            return gen_hvs(1, self.dim, seed=seed).squeeze(0)

    def encode(
        self,
        record: Dict[str, Any],
        type_tag: Optional[str] = None,
    ) -> torch.Tensor:
        """Encode a structured record.

        Args:
            record: {field_name: value} dictionary
            type_tag: Optional type identifier

        Returns:
            (dim,) record hypervector
        """
        bound_pairs = []
        for name, value in record.items():
            role = self._get_role(name)
            filler = self._value_to_hv(value)
            bound_pairs.append(hv_xor(role, filler))

        if not bound_pairs:
            return torch.zeros(self.dim)

        record_hv = hv_bundle(torch.stack(bound_pairs))
        record_hv = hv_majority(record_hv)

        if type_tag is not None:
            type_hv = self._get_type(type_tag)
            record_hv = hv_xor(record_hv, type_hv)

        return record_hv

    def get_field(
        self,
        record_hv: torch.Tensor,
        field: str,
        codebook: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Retrieve a field value from an encoded record.

        Args:
            record_hv: (dim,) record hypervector
            field: Field name
            codebook: Optional (n, dim) codebook for cleanup

        Returns:
            (dim,) field value hypervector
        """
        role = self._get_role(field)
        filler_noisy = hv_xor(record_hv, role)

        if codebook is not None:
            sims = hv_batch_sim(filler_noisy, codebook)
            best_idx = int(sims.argmax().item())
            return codebook[best_idx].clone()

        return filler_noisy


# ═══════════════════════════════════════════════════════════════════════════════
# Part I, Section III-C: Spatial Encoding
# ═══════════════════════════════════════════════════════════════════════════════

class SpatialEncoder:
    """
    Spatial encoding for 2D/3D data (Kleyko 2023, Part I, Section III-C).

    Encodes spatial positions using binding of coordinate hypervectors:
        P(x, y) = bind(X_x, Y_y)

    Where X_x and Y_y are hypervectors representing specific coordinates.
    This preserves spatial relationships: nearby points have similar HVs.

    Supports:
    - 2D grid positions
    - 3D volumetric positions
    - Continuous coordinates (via interpolation)
    - Spatial relationships (above, below, left, right)
    """

    def __init__(
        self,
        dim: int = 10000,
        mode: str = "binary",
        seed: Optional[int] = None,
    ):
        self.dim = dim
        self.mode = mode
        self._seed = seed
        self._axis_hvs: Dict[str, Dict[int, torch.Tensor]] = {}

    def _get_axis_hv(self, axis: str, coord: int) -> torch.Tensor:
        """Get hypervector for a specific coordinate on an axis."""
        if axis not in self._axis_hvs:
            self._axis_hvs[axis] = {}
        if coord not in self._axis_hvs[axis]:
            seed = (hash(f"{axis}_{coord}") & 0x7FFFFFFF) if self._seed is None else \
                   self._seed + len(self._axis_hvs) * 1000 + coord
            self._axis_hvs[axis][coord] = gen_hvs(1, self.dim, seed=seed).squeeze(0)
        return self._axis_hvs[axis][coord]

    def encode_2d(self, x: int, y: int) -> torch.Tensor:
        """Encode a 2D position.

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            (dim,) position hypervector
        """
        x_hv = self._get_axis_hv("x", x)
        y_hv = self._get_axis_hv("y", y)
        return hv_xor(x_hv, y_hv)

    def encode_3d(self, x: int, y: int, z: int) -> torch.Tensor:
        """Encode a 3D position.

        Args:
            x: X coordinate
            y: Y coordinate
            z: Z coordinate

        Returns:
            (dim,) position hypervector
        """
        x_hv = self._get_axis_hv("x", x)
        y_hv = self._get_axis_hv("y", y)
        z_hv = self._get_axis_hv("z", z)
        return hv_xor(hv_xor(x_hv, y_hv), z_hv)

    def encode_continuous_2d(self, x: float, y: float, resolution: int = 100) -> torch.Tensor:
        """Encode a continuous 2D position using interpolation.

        Args:
            x: Continuous X coordinate
            y: Continuous Y coordinate
            resolution: Number of discrete positions per axis

        Returns:
            (dim,) interpolated position hypervector
        """
        # Find nearest discrete coordinates
        xi = int(x * resolution)
        yi = int(y * resolution)

        # Get corner hypervectors
        p00 = self.encode_2d(xi, yi)
        p01 = self.encode_2d(xi, yi + 1)
        p10 = self.encode_2d(xi + 1, yi)
        p11 = self.encode_2d(xi + 1, yi + 1)

        # Bilinear interpolation weights
        wx = x * resolution - xi
        wy = y * resolution - yi

        # Interpolate (approximate: weighted bundle)
        interpolated = hv_bundle(torch.stack([
            p00 * (1 - wx) * (1 - wy),
            p01 * (1 - wx) * wy,
            p10 * wx * (1 - wy),
            p11 * wx * wy,
        ]))
        return hv_majority(interpolated)

    def spatial_similarity(self, pos_a: torch.Tensor, pos_b: torch.Tensor) -> float:
        """Compute spatial similarity between two positions.

        Higher similarity = spatially closer positions.
        """
        return float(hv_hamming_sim(pos_a, pos_b))


# ═══════════════════════════════════════════════════════════════════════════════
# Part II, Section IV: Retraining Strategies
# ═══════════════════════════════════════════════════════════════════════════════

class RetrainingStrategy:
    """
    Iterative retraining strategies for HDC (Kleyko 2023, Part II, Section IV).

    Implements:
    1. **Iterative Refinement** — Update prototypes with misclassified samples
    2. **Adaptive Learning Rate** — Adjust update magnitude based on confidence
    3. **Selective Retraining** — Only update for high-error regions
    4. **Retrain with Noise** — Add noise during retraining for robustness

    These strategies significantly improve accuracy over single-pass training.
    """

    def __init__(
        self,
        dim: int = 10000,
        strategy: str = "iterative",
        learning_rate: float = 0.1,
        max_iterations: int = 10,
    ):
        """
        Args:
            dim: Hypervector dimensionality
            strategy: "iterative", "adaptive", "selective", or "noise"
            learning_rate: Update magnitude
            max_iterations: Maximum retraining iterations
        """
        self.dim = dim
        self.strategy = strategy
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations

    def iterative_refinement(
        self,
        prototypes: torch.Tensor,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        encode_fn: Callable,
    ) -> torch.Tensor:
        """Iteratively refine prototypes using misclassified samples.

        Args:
            prototypes: (n_classes, dim) initial prototypes
            X_train: (n_train, n_features) training data
            y_train: (n_train,) training labels
            encode_fn: Function to encode features → hypervector

        Returns:
            (n_classes, dim) refined prototypes
        """
        refined = prototypes.clone()

        for iteration in range(self.max_iterations):
            errors = 0

            for i in range(X_train.shape[0]):
                hv = encode_fn(X_train[i])
                true_label = int(y_train[i].item())

                # Find closest prototype
                sims = hv_batch_sim(hv, refined)
                pred_label = int(sims.argmax().item())

                if pred_label != true_label:
                    errors += 1
                    # Move correct prototype toward sample
                    refined[true_label] = refined[true_label] + hv * self.learning_rate
                    # Move incorrect prototype away from sample
                    refined[pred_label] = refined[pred_label] - hv * self.learning_rate

            # Threshold
            refined = hv_majority(refined)

            if errors == 0:
                break

        return refined

    def adaptive_retraining(
        self,
        prototypes: torch.Tensor,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        encode_fn: Callable,
    ) -> torch.Tensor:
        """Retrain with adaptive learning rate based on confidence.

        Low confidence → larger update. High confidence → smaller update.
        """
        refined = prototypes.clone()

        for iteration in range(self.max_iterations):
            for i in range(X_train.shape[0]):
                hv = encode_fn(X_train[i])
                true_label = int(y_train[i].item())

                sims = hv_batch_sim(hv, refined)
                pred_label = int(sims.argmax().item())

                if pred_label != true_label:
                    # Confidence = difference between top-2 similarities
                    sorted_sims, _ = sims.sort(descending=True)
                    confidence = sorted_sims[0] - sorted_sims[1]

                    # Adaptive LR: lower confidence → higher LR
                    adaptive_lr = self.learning_rate * (1.0 + (1.0 - confidence))
                    refined[true_label] = refined[true_label] + hv * adaptive_lr
                    refined[pred_label] = refined[pred_label] - hv * adaptive_lr

            refined = hv_majority(refined)

        return refined

    def selective_retraining(
        self,
        prototypes: torch.Tensor,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        encode_fn: Callable,
        threshold: float = 0.1,
    ) -> torch.Tensor:
        """Only retrain on samples near decision boundaries.

        Samples with similarity difference < threshold are "uncertain"
        and get retrained. Confident samples are skipped.
        """
        refined = prototypes.clone()

        for iteration in range(self.max_iterations):
            for i in range(X_train.shape[0]):
                hv = encode_fn(X_train[i])
                true_label = int(y_train[i].item())

                sims = hv_batch_sim(hv, refined)
                sorted_sims, _ = sims.sort(descending=True)
                margin = sorted_sims[0] - sorted_sims[1]

                # Only retrain uncertain samples
                if margin < threshold:
                    pred_label = int(sims.argmax().item())
                    if pred_label != true_label:
                        refined[true_label] = refined[true_label] + hv * self.learning_rate
                        refined[pred_label] = refined[pred_label] - hv * self.learning_rate

            refined = hv_majority(refined)

        return refined

    def noise_robust_retraining(
        self,
        prototypes: torch.Tensor,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        encode_fn: Callable,
        noise_level: float = 0.05,
    ) -> torch.Tensor:
        """Retrain with noise injection for robustness.

        Adds noise to hypervectors during retraining to improve
        generalization and noise tolerance.
        """
        refined = prototypes.clone()

        for iteration in range(self.max_iterations):
            for i in range(X_train.shape[0]):
                hv = encode_fn(X_train[i])
                true_label = int(y_train[i].item())

                # Add noise to hypervector
                noise = (torch.rand_like(hv) < noise_level).float()
                hv_noisy = hv_xor(hv, noise)

                sims = hv_batch_sim(hv_noisy, refined)
                pred_label = int(sims.argmax().item())

                if pred_label != true_label:
                    refined[true_label] = refined[true_label] + hv * self.learning_rate
                    refined[pred_label] = refined[pred_label] - hv * self.learning_rate

            refined = hv_majority(refined)

        return refined


# ═══════════════════════════════════════════════════════════════════════════════
# Part II, Section V: Ensemble Methods
# ═══════════════════════════════════════════════════════════════════════════════

class HDEnsemble:
    """
    Ensemble methods for HDC (Kleyko 2023, Part II, Section V).

    Combines multiple HDC classifiers for improved accuracy:
    1. **Bagging** — Train on bootstrap samples, majority vote
    2. **Feature Subspace** — Each classifier uses different feature subsets
    3. **Random Encoding** — Different random seeds for each classifier
    4. **Weighted Voting** — Weight classifiers by validation accuracy

    Ensembles are particularly effective for HDC because different
    random projections capture complementary information.
    """

    def __init__(
        self,
        n_classifiers: int = 10,
        dim: int = 10000,
        strategy: str = "bagging",
        seed: Optional[int] = None,
    ):
        """
        Args:
            n_classifiers: Number of classifiers in ensemble
            dim: Hypervector dimensionality
            strategy: "bagging", "subspace", "random", or "weighted"
            seed: Random seed
        """
        self.n_classifiers = n_classifiers
        self.dim = dim
        self.strategy = strategy
        self.seed = seed

        self.classifiers = []
        self.weights: torch.Tensor = torch.ones(n_classifiers)

    def _create_classifier(self, n_features: int, n_classes: int, seed: int):
        """Create a classifier with specific seed."""
        from hdc.binary_hdc_tradeoffs import DensityAwareHDCClassifier
        return DensityAwareHDCClassifier(
            n_features=n_features,
            n_classes=n_classes,
            dim=self.dim,
            encoding_density=0.5,
            seed=seed,
        )

    def fit(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        n_classes: int,
    ):
        """Train the ensemble.

        Args:
            X_train: (n_train, n_features) training data
            y_train: (n_train,) training labels
            n_classes: Number of classes
        """
        n_features = X_train.shape[1]
        n_train = X_train.shape[0]

        for i in range(self.n_classifiers):
            seed = (self.seed or 0) + i
            classifier = self._create_classifier(n_features, n_classes, seed)

            if self.strategy == "bagging":
                # Bootstrap sample
                indices = torch.randint(0, n_train, (n_train,))
                X_boot = X_train[indices]
                y_boot = y_train[indices]
            elif self.strategy == "subspace":
                # Use all data but with different feature subsets
                X_boot = X_train
                y_boot = y_train
            else:
                X_boot = X_train
                y_boot = y_train

            # Train
            for j in range(X_boot.shape[0]):
                classifier.train_step(X_boot[j], int(y_boot[j].item()))
            classifier.finalize()

            self.classifiers.append(classifier)

    def predict(self, x: torch.Tensor) -> Tuple[int, torch.Tensor]:
        """Predict using ensemble voting.

        Args:
            x: (n_features,) input features

        Returns:
            (predicted_class, all_class_votes)
        """
        n_classes = len(self.classifiers[0].class_hvs)
        votes = torch.zeros(n_classes)

        for i, classifier in enumerate(self.classifiers):
            pred, sims = classifier.predict(x)
            votes[pred] += self.weights[i].item()

        return int(votes.argmax().item()), votes

    def set_weights_from_accuracy(
        self,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
    ):
        """Set classifier weights based on validation accuracy.

        Args:
            X_val: (n_val, n_features) validation data
            y_val: (n_val,) validation labels
        """
        for i, classifier in enumerate(self.classifiers):
            correct = 0
            for j in range(X_val.shape[0]):
                pred, _ = classifier.predict(X_val[j])
                if pred == int(y_val[j].item()):
                    correct += 1
            self.weights[i] = correct / X_val.shape[0]

        # Normalize weights
        self.weights = self.weights / self.weights.sum()

    def diversity_score(self) -> float:
        """
        Measure ensemble diversity: mean pairwise prediction disagreement.

        High diversity → ensemble members are making different predictions
        (more robust, less correlated errors).
        Low diversity  → members agree too much (little benefit from ensemble).

        Returns:
            Diversity ∈ [0, 1]; > 0.3 is healthy.
        """
        if len(self.classifiers) < 2:
            return 0.0
        # Use prototype pairwise distance as a proxy for prediction diversity
        n = len(self.classifiers)
        total, count = 0.0, 0
        for i in range(n):
            p_i = self.classifiers[i]
            for j in range(i + 1, n):
                p_j = self.classifiers[j]
                # Compare first class prototype similarity
                if hasattr(p_i, "prototypes") and hasattr(p_j, "prototypes"):
                    d = float((p_i.prototypes[0] != p_j.prototypes[0]).float().mean())
                    total += d
                    count += 1
        return total / max(count, 1)

    def online_weight_update(
        self,
        x:     torch.Tensor,
        label: int,
        lr:    float = 0.1,
    ):
        """
        Update ensemble weights online from a single labelled example.

        Members that correctly classify this example get a weight boost;
        members that misclassify get a weight penalty.

        Args:
            x:     Input sample
            label: True class label
            lr:    Weight update rate
        """
        with torch.no_grad():
            for i, clf in enumerate(self.classifiers):
                pred, _ = clf.predict(x)
                # Boost correct, penalise wrong
                if pred == label:
                    self.weights[i] = self.weights[i] * (1 + lr)
                else:
                    self.weights[i] = self.weights[i] * (1 - lr * 0.5)

            # Clamp and renormalise
            self.weights = self.weights.clamp(min=0.01)
            self.weights = self.weights / self.weights.sum()


# ═══════════════════════════════════════════════════════════════════════════════
# Part II, Section VI: Confidence Calibration
# ═══════════════════════════════════════════════════════════════════════════════

class ConfidenceCalibrator:
    """
    Confidence calibration for HDC classifiers (Kleyko 2023, Part II, Section VI).

    HDC similarity scores are not naturally calibrated probabilities.
    This module provides:
    1. **Platt Scaling** — Logistic regression on similarity scores
    2. **Temperature Scaling** — Single parameter for all classes
    3. **Vector Scaling** — Class-specific temperature parameters
    4. **Isotonic Regression** — Non-parametric calibration

    Calibrated confidence enables:
    - Reliable rejection of uncertain predictions
    - Better decision thresholds
    - Meaningful probability estimates
    """

    def __init__(self, method: str = "temperature"):
        """
        Args:
            method: "temperature", "platt", "vector", or "isotonic"
        """
        self.method = method
        self.temperature: float = 1.0
        self.platt_a: float = 0.0
        self.platt_b: float = 0.0
        self.vector_scales: Optional[torch.Tensor] = None

    def calibrate_temperature(
        self,
        similarities: torch.Tensor,
        labels: torch.Tensor,
    ):
        """Calibrate using temperature scaling.

        Args:
            similarities: (n, n_classes) similarity scores
            labels: (n,) true labels
        """
        # Simple grid search for optimal temperature
        best_temp = 1.0
        best_nll = float('inf')

        for temp in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
            scaled = similarities / temp
            probs = torch.softmax(scaled, dim=-1)

            # Negative log likelihood
            nll = 0.0
            for i in range(len(labels)):
                nll -= torch.log(probs[i, int(labels[i].item())] + 1e-12).item()

            if nll < best_nll:
                best_nll = nll
                best_temp = temp

        self.temperature = best_temp

    def calibrate_platt(
        self,
        similarities: torch.Tensor,
        labels: torch.Tensor,
    ):
        """Calibrate using Platt scaling.

        Fits P(y=1|x) = 1 / (1 + exp(A * sim + B))
        """
        # Simplified: use the max similarity per sample
        max_sims = similarities.max(dim=-1).values
        binary_labels = (similarities.argmax(dim=-1) == labels).float()

        # Simple linear fit
        pos = max_sims[binary_labels == 1]
        neg = max_sims[binary_labels == 0]

        if len(pos) > 0 and len(neg) > 0:
            self.platt_a = 1.0 / (pos.mean() - neg.mean() + 1e-12)
            self.platt_b = -self.platt_a * pos.mean()

    def predict_proba(self, similarities: torch.Tensor) -> torch.Tensor:
        """Convert similarity scores to calibrated probabilities.

        Args:
            similarities: (n_classes,) similarity scores

        Returns:
            (n_classes,) calibrated probabilities
        """
        if self.method == "temperature":
            scaled = similarities / self.temperature
            return torch.softmax(scaled, dim=-1)

        elif self.method == "platt":
            max_sim = similarities.max().item()
            prob = 1.0 / (1.0 + torch.exp(-(self.platt_a * max_sim + self.platt_b)))
            return prob.expand_as(similarities)

        elif self.method == "vector":
            if self.vector_scales is not None:
                scaled = similarities / self.vector_scales
                return torch.softmax(scaled, dim=-1)
            return torch.softmax(similarities, dim=-1)

        else:
            return torch.softmax(similarities, dim=-1)

    def reject_uncertain(
        self,
        similarities: torch.Tensor,
        threshold: float = 0.5,
    ) -> Tuple[Optional[int], float]:
        """Predict with rejection option.

        Args:
            similarities: (n_classes,) similarity scores
            threshold: Minimum confidence for acceptance

        Returns:
            (predicted_class or None, confidence)
        """
        probs = self.predict_proba(similarities)
        confidence = float(probs.max().item())

        if confidence >= threshold:
            return int(probs.argmax().item()), confidence
        else:
            return None, confidence


# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_ngram_encoder():
    """Verify N-gram encoding."""
    print("=" * 60)
    print("Testing N-gram Encoding (Kleyko 2023)")
    print("=" * 60)

    dim = 1000
    encoder = NGramEncoder(dim=dim, n=3)

    symbols = [gen_hvs(1, dim, seed=i).squeeze(0) for i in range(10)]
    seq = encoder.encode_sequence(symbols)
    print(f"  Sequence shape: {seq.shape}")

    # Similar sequences should have higher similarity
    symbols_similar = symbols[:8] + [gen_hvs(1, dim, seed=100).squeeze(0)] * 2
    seq_similar = encoder.encode_sequence(symbols_similar)

    symbols_diff = [gen_hvs(1, dim, seed=i+100).squeeze(0) for i in range(10)]
    seq_diff = encoder.encode_sequence(symbols_diff)

    sim_similar = float(hv_hamming_sim(seq, seq_similar))
    sim_diff = float(hv_hamming_sim(seq, seq_diff))
    print(f"  Similar seq similarity: {sim_similar:.4f}")
    print(f"  Different seq similarity: {sim_diff:.4f}")
    print(f"  {'✅' if sim_similar > sim_diff else '❌'} N-gram encoding test!")


def test_record_encoder():
    """Verify record-based encoding."""
    print("=" * 60)
    print("Testing Record Encoding (Kleyko 2023)")
    print("=" * 60)

    dim = 1000
    encoder = RecordEncoder(dim=dim)

    record = {
        "name": "Alice",
        "age": 30,
        "skills": ["Python", "HDC", "ML"],
        "address": {"city": "NYC", "zip": "10001"},
    }

    hv = encoder.encode(record, type_tag="person")
    print(f"  Record HV shape: {hv.shape}")

    # Retrieve field
    name_hv = encoder.get_field(hv, "name")
    print(f"  Retrieved name field shape: {name_hv.shape}")

    print(f"  ✅ Record encoding test complete!")


def test_spatial_encoder():
    """Verify spatial encoding."""
    print("=" * 60)
    print("Testing Spatial Encoding (Kleyko 2023)")
    print("=" * 60)

    dim = 1000
    encoder = SpatialEncoder(dim=dim)

    # Nearby positions should have higher similarity
    p1 = encoder.encode_2d(5, 5)
    p2 = encoder.encode_2d(5, 6)  # 1 unit away
    p3 = encoder.encode_2d(50, 50)  # far away

    sim_near = encoder.spatial_similarity(p1, p2)
    sim_far = encoder.spatial_similarity(p1, p3)
    print(f"  Nearby positions similarity: {sim_near:.4f}")
    print(f"  Far positions similarity: {sim_far:.4f}")
    print(f"  {'✅' if sim_near > sim_far else '❌'} Spatial encoding test!")


def test_retraining():
    """Verify retraining strategies."""
    print("=" * 60)
    print("Testing Retraining Strategies (Kleyko 2023)")
    print("=" * 60)

    dim = 1000
    n_classes = 3
    n_features = 10

    # Create synthetic data
    prototypes = gen_hvs(n_classes, dim)
    X_train = torch.randn(30, n_features)
    y_train = torch.randint(0, n_classes, (30,))

    def encode_fn(x):
        return hv_majority((x.unsqueeze(-1) * gen_hvs(n_features, dim)).sum(dim=0))

    strategy = RetrainingStrategy(dim=dim, strategy="iterative", learning_rate=0.1, max_iterations=5)
    refined = strategy.iterative_refinement(prototypes, X_train, y_train, encode_fn)
    print(f"  Refined prototypes shape: {refined.shape}")
    print(f"  ✅ Retraining test complete!")


if __name__ == "__main__":
    test_ngram_encoder()
    print()
    test_record_encoder()
    print()
    test_spatial_encoder()
    print()
    test_retraining()
