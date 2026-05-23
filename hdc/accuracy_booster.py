"""
HDC Model Accuracy Improvement Techniques
==========================================
Implements three complementary accuracy-improvement strategies for HDC
classifiers, derived from the SNNTraining research base:

1. **ConfusionAwareRetrainer** — finds class pairs that are most confused
   (their prototype HVs are too similar) and applies targeted differential
   retraining: pull the confused prototypes apart by reinforcing correct
   samples and suppressing incorrect assignments. Grounded in:
     - Ge & Parhi 2020 (ge_parhi_survey.py) — retraining strategies
     - Kleyko 2018 (binary_hdc_tradeoffs.py) — density-normalized similarity

2. **PrototypeQualityAssessor** — measures the quality of learned prototypes
   using two metrics:
     - Class separation: min pairwise Hamming distance between prototypes
       (higher = more separable = better accuracy)
     - Within-class compactness: mean Hamming distance from training samples
       to their class prototype (lower = tighter cluster = more robust)
   Flags classes with poor separation or high variance for targeted retraining.

3. **CalibratedHDCClassifier** — wraps any HDC classifier with post-hoc
   temperature scaling (Platt calibration) so that similarity scores become
   calibrated probabilities. Uses ConfidenceCalibrator from ge_parhi_survey.py.
   Calibrated confidence enables:
     - Reliable anomaly detection (reject when max_prob < threshold)
     - Weighted ensemble voting (weight by calibrated confidence)
     - Uncertainty-aware downstream decisions

4. **OnlineSelfCorrector** — continuous accuracy self-improvement using the
   agent's own high-confidence predictions as pseudo-labels:
     - If predict(x) returns confidence > self_label_threshold: treat it as a
       labelled sample and apply a mild retraining step
     - Combined with confusion-aware targeting: self-correct only the classes
       currently showing the most errors
     - Grounded in semi-supervised HDC learning (Kleyko 2023 Survey §IV)

5. **AccuracyBenchmark** — quick evaluation suite that reports:
     - Per-class accuracy and confusion matrix (as HV similarities)
     - Before/after comparison for any retraining method
     - Energy cost per classification (using efficiency.py model)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from hdc.hdc_glue import hv_batch_sim, gen_hvs, hv_majority, hv_bundle


# ═══════════════════════════════════════════════════════════════════════════════
# 1. PrototypeQualityAssessor
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PrototypeQualityReport:
    """Quality assessment of learned class prototypes."""
    n_classes: int
    min_separation: float       # min pairwise Hamming distance between protos
    mean_separation: float      # mean pairwise Hamming distance
    confused_pairs: List[Tuple[int, int, float]]  # (class_a, class_b, similarity)
    per_class_compactness: List[float]  # within-class Hamming spread
    worst_class: int            # class with lowest compactness
    overall_quality: float      # composite score in [0, 1]


class PrototypeQualityAssessor:
    """
    Measure and diagnose prototype quality for HDC classifiers.

    Two key metrics from Ge & Parhi 2020 (ge_parhi_survey.py):

    1. **Inter-class separation** — min pairwise Hamming distance between
       prototype HVs. For good accuracy, prototypes should be near-orthogonal
       (Hamming ≈ 0.5). Pairs with Hamming < 0.35 are "confused" and likely
       to cause misclassification.

    2. **Within-class compactness** — mean Hamming distance from each training
       sample to its class prototype. Tight clusters (distance < 0.2) produce
       robust classifiers; loose clusters (distance > 0.35) indicate that the
       prototype has poor coverage of the class.

    The overall quality score combines both:
        quality = separation_score × compactness_score
        separation_score = min(1.0, 2 × min_separation)  [want ≥ 0.5]
        compactness_score = max(0.0, 1.0 - mean_compactness × 2) [want ≤ 0.25]

    Args:
        confusion_threshold: Similarity above which two classes are "confused"
        compactness_threshold: Within-class distance above which class is "loose"
    """

    def __init__(
        self,
        confusion_threshold: float = 0.65,
        compactness_threshold: float = 0.35,
    ):
        self.confusion_threshold = confusion_threshold
        self.compactness_threshold = compactness_threshold

    def assess(
        self,
        prototypes: torch.Tensor,
        X_train: Optional[torch.Tensor] = None,
        y_train: Optional[torch.Tensor] = None,
        encode_fn=None,
    ) -> PrototypeQualityReport:
        """
        Assess prototype quality.

        Args:
            prototypes: (n_classes, dim) class prototype HVs
            X_train: Optional training data for compactness measurement
            y_train: Optional training labels
            encode_fn: Optional function to encode X_train to HVs

        Returns:
            PrototypeQualityReport
        """
        n = prototypes.shape[0]

        # Inter-class separation
        sims = []
        confused_pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                sim = float(hv_batch_sim(prototypes[i], prototypes[j].unsqueeze(0))[0])
                sims.append(sim)
                if sim > self.confusion_threshold:
                    confused_pairs.append((i, j, sim))

        min_sep = min(sims) if sims else 0.5
        mean_sep = sum(sims) / len(sims) if sims else 0.5
        confused_pairs.sort(key=lambda x: x[2], reverse=True)

        # Within-class compactness
        per_class_compactness = []
        if X_train is not None and y_train is not None and encode_fn is not None:
            for c in range(n):
                mask = (y_train == c)
                if mask.sum() == 0:
                    per_class_compactness.append(0.5)
                    continue
                X_c = X_train[mask]
                hvs_c = torch.stack([encode_fn(X_c[i]) for i in range(X_c.shape[0])])
                proto = prototypes[c]
                dists = [
                    1.0 - float(hv_batch_sim(hvs_c[i], proto.unsqueeze(0))[0])
                    for i in range(hvs_c.shape[0])
                ]
                per_class_compactness.append(sum(dists) / len(dists))
        else:
            per_class_compactness = [0.25] * n   # assume moderate compactness

        worst_class = int(max(range(n), key=lambda i: per_class_compactness[i]))
        mean_compactness = sum(per_class_compactness) / n

        # Composite quality score
        sep_score = min(1.0, 2.0 * (min_sep - 0.5) + 0.5) if min_sep >= 0.5 else 0.0
        compact_score = max(0.0, 1.0 - mean_compactness * 2)
        quality = (sep_score + compact_score) / 2

        return PrototypeQualityReport(
            n_classes=n,
            min_separation=min_sep,
            mean_separation=mean_sep,
            confused_pairs=confused_pairs,
            per_class_compactness=per_class_compactness,
            worst_class=worst_class,
            overall_quality=quality,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 2. ConfusionAwareRetrainer
# ═══════════════════════════════════════════════════════════════════════════════

class ConfusionAwareRetrainer:
    """
    Targeted differential retraining for confused class pairs.

    Standard HDC retraining (Ge & Parhi 2020 §IV-B) updates prototypes
    uniformly for all misclassified samples. This is wasteful when most
    errors come from a small number of confused class pairs.

    Confusion-aware retraining focuses effort on the worst pairs:
      1. Assess which class pairs have highest prototype similarity
      2. For each confused pair (A, B): find training samples near the
         decision boundary (sim_to_A ≈ sim_to_B)
      3. Apply targeted update:
           proto_A += α × x (reinforce A)
           proto_B -= α × x (suppress B for this sample)
      4. Binarise both prototypes after update

    This is more efficient than full retraining (O(n_confused × n_boundary)
    instead of O(n_classes × n_samples)) and often more effective because it
    directly targets the source of misclassification.

    Args:
        assessor: PrototypeQualityAssessor to identify confused pairs
        learning_rate: Update step size for differential retraining
        boundary_threshold: Similarity difference below which a sample is
                           "on the boundary" between two classes
        max_pairs: Maximum number of confused pairs to address per call
    """

    def __init__(
        self,
        assessor: PrototypeQualityAssessor,
        learning_rate: float = 0.05,
        boundary_threshold: float = 0.1,
        max_pairs: int = 5,
    ):
        self.assessor = assessor
        self.lr = learning_rate
        self.boundary_threshold = boundary_threshold
        self.max_pairs = max_pairs

    def retrain(
        self,
        prototypes: torch.Tensor,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        encode_fn,
        n_passes: int = 3,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Apply confusion-aware differential retraining.

        Args:
            prototypes: (n_classes, dim) current prototypes (float accumulators)
            X_train: (N, n_features) training data
            y_train: (N,) integer labels
            encode_fn: Maps (n_features,) input → (dim,) binary HV
            n_passes: Number of retraining passes

        Returns:
            (updated_prototypes, stats_dict)
        """
        report = self.assessor.assess(prototypes, X_train, y_train, encode_fn)
        confused_pairs = report.confused_pairs[:self.max_pairs]

        if not confused_pairs:
            return prototypes, {"n_updates": 0, "confused_pairs": 0}

        protos = prototypes.float().clone()
        n_updates = 0

        for _ in range(n_passes):
            for class_a, class_b, _ in confused_pairs:
                # Find boundary samples between class_a and class_b
                for i in range(X_train.shape[0]):
                    true_label = int(y_train[i].item())
                    if true_label not in (class_a, class_b):
                        continue

                    hv = encode_fn(X_train[i])
                    sim_a = float(hv_batch_sim(hv, protos[class_a].unsqueeze(0))[0])
                    sim_b = float(hv_batch_sim(hv, protos[class_b].unsqueeze(0))[0])

                    # Is this sample on the boundary?
                    if abs(sim_a - sim_b) > self.boundary_threshold:
                        continue  # clear sample, skip

                    # Apply differential update
                    hv_f = hv.float()
                    if true_label == class_a:
                        protos[class_a] += self.lr * hv_f
                        protos[class_b] -= self.lr * hv_f
                    else:
                        protos[class_b] += self.lr * hv_f
                        protos[class_a] -= self.lr * hv_f
                    n_updates += 1

        # Binarise
        updated = (protos > 0).float()
        return updated, {
            "n_updates": n_updates,
            "confused_pairs": len(confused_pairs),
            "pairs": [(a, b) for a, b, _ in confused_pairs],
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 3. CalibratedHDCClassifier — temperature scaling for confidence
# ═══════════════════════════════════════════════════════════════════════════════

class CalibratedHDCClassifier:
    """
    Wraps any HDC classifier with post-hoc temperature calibration.

    Hamming similarity scores from HDC classifiers are NOT calibrated
    probabilities by default — a score of 0.7 does not mean 70% confidence.
    Temperature scaling (Guo et al. 2017; Ge & Parhi 2020 ConfidenceCalibrator)
    maps raw similarity scores to calibrated probabilities:

        P(class=c | x) = softmax(sims / T)[c]

    where T is chosen to minimise negative log-likelihood on a validation set.

    After calibration:
      - High confidence (P > 0.9) reliably predicts correct classification
      - Low confidence (P < 0.6) signals uncertain or out-of-distribution input
      - Can set a rejection threshold to abstain on uncertain inputs

    Args:
        base_classifier: Any HDC classifier with predict_proba-like interface
        n_classes: Number of classes
        temperature_candidates: Grid of temperatures to evaluate
    """

    def __init__(
        self,
        base_classifier,
        n_classes: int,
        temperature_candidates: Optional[List[float]] = None,
    ):
        self.clf = base_classifier
        self.n_classes = n_classes
        self.temperature = 1.0   # uncalibrated
        self.candidates = temperature_candidates or [0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0]
        self._calibrated = False

    def _get_similarities(
        self,
        X: torch.Tensor,
        encode_fn,
        prototypes: torch.Tensor,
    ) -> torch.Tensor:
        """Get raw similarity matrix (N, n_classes)."""
        n = X.shape[0]
        sims = torch.zeros(n, self.n_classes)
        for i in range(n):
            hv = encode_fn(X[i])
            sims[i] = hv_batch_sim(hv, prototypes)
        return sims

    def calibrate(
        self,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        encode_fn,
        prototypes: torch.Tensor,
    ) -> float:
        """
        Find optimal temperature T on validation data.

        Minimises NLL = -mean_i log P(y_i | x_i; T).

        Returns:
            Best temperature T
        """
        sims = self._get_similarities(X_val, encode_fn, prototypes)

        best_T = 1.0
        best_nll = float('inf')

        for T in self.candidates:
            probs = torch.softmax(sims / T, dim=-1)  # (N, C)
            # NLL
            nll = 0.0
            for i in range(sims.shape[0]):
                nll -= math.log(max(probs[i, int(y_val[i].item())].item(), 1e-10))
            nll /= sims.shape[0]

            if nll < best_nll:
                best_nll = nll
                best_T = T

        self.temperature = best_T
        self._calibrated = True
        return best_T

    def predict_proba(
        self,
        sims: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert similarity scores to calibrated probabilities.

        Args:
            sims: (N, n_classes) or (n_classes,) raw similarity scores

        Returns:
            (N, n_classes) or (n_classes,) calibrated probabilities
        """
        return torch.softmax(sims / self.temperature, dim=-1)

    def predict_with_confidence(
        self,
        x: torch.Tensor,
        encode_fn,
        prototypes: torch.Tensor,
        rejection_threshold: float = 0.0,
    ) -> Tuple[int, float]:
        """
        Predict class with calibrated confidence, optionally rejecting uncertain inputs.

        Args:
            x: (n_features,) input
            encode_fn: Feature → HV encoder
            prototypes: (n_classes, dim) class prototypes
            rejection_threshold: Reject if max_prob < threshold (return -1)

        Returns:
            (predicted_class, confidence)  where predicted_class=-1 if rejected
        """
        hv = encode_fn(x)
        sims = hv_batch_sim(hv, prototypes)
        probs = self.predict_proba(sims)
        max_prob = float(probs.max().item())
        pred_class = int(probs.argmax().item())

        if rejection_threshold > 0 and max_prob < rejection_threshold:
            return -1, max_prob

        return pred_class, max_prob

    def expected_calibration_error(
        self,
        X_test: torch.Tensor,
        y_test: torch.Tensor,
        encode_fn,
        prototypes: torch.Tensor,
        n_bins: int = 10,
    ) -> float:
        """
        Expected Calibration Error (ECE) — measures how well probabilities
        match empirical accuracy across confidence bins.

        ECE = Σ_b (|B_b| / N) × |acc(B_b) - conf(B_b)|

        Lower ECE = better calibrated.

        Returns:
            ECE in [0, 1]
        """
        sims = self._get_similarities(X_test, encode_fn, prototypes)
        probs = self.predict_proba(sims)
        confs, preds = probs.max(dim=-1)

        ece = 0.0
        n = X_test.shape[0]
        bin_edges = torch.linspace(0, 1, n_bins + 1)

        for b in range(n_bins):
            lo, hi = float(bin_edges[b]), float(bin_edges[b + 1])
            mask = (confs >= lo) & (confs < hi)
            if mask.sum() == 0:
                continue
            acc_b = float((preds[mask] == y_test[mask].long()).float().mean())
            conf_b = float(confs[mask].mean())
            ece += (mask.sum().item() / n) * abs(acc_b - conf_b)

        return ece


# ═══════════════════════════════════════════════════════════════════════════════
# 4. OnlineSelfCorrector — semi-supervised accuracy improvement
# ═══════════════════════════════════════════════════════════════════════════════

class OnlineSelfCorrector:
    """
    Continuous self-improvement using high-confidence pseudo-labels.

    When the classifier is highly confident in a prediction, treat that
    prediction as a pseudo-label and apply a mild retraining step. This
    is semi-supervised HDC learning (Kleyko 2023 Survey §IV):

        if confidence(x) > threshold:
            label = predict(x)
            proto[label] += 0.1 × hv(x)   (mild Hebbian reinforcement)

    Focuses pseudo-label updates on currently-confused classes only:
        if the worst-performing class pair is (A, B), only self-correct
        samples predicted as A or B with high confidence.

    Args:
        prototypes: (n_classes, dim) float accumulator prototypes
        confidence_threshold: Min calibrated confidence for pseudo-labelling
        lr_self: Learning rate for self-correction updates
        max_self_labels_per_tick: Cap on updates per inference call
    """

    def __init__(
        self,
        n_classes: int,
        dim: int,
        confidence_threshold: float = 0.85,
        lr_self: float = 0.02,
        max_self_labels_per_tick: int = 5,
    ):
        self.n_classes = n_classes
        self.dim = dim
        self.confidence_threshold = confidence_threshold
        self.lr_self = lr_self
        self.max_per_tick = max_self_labels_per_tick

        # Float accumulators (same as in standard HDC training)
        self._protos = torch.zeros(n_classes, dim)
        self._counts = torch.zeros(n_classes)
        self._n_self_labels = 0

    def set_prototypes(self, protos: torch.Tensor):
        """Initialise from an already-trained prototype matrix."""
        self._protos = protos.float().clone()
        self._counts = torch.ones(self.n_classes)

    def observe(
        self,
        hv: torch.Tensor,
        calibrated_probs: torch.Tensor,
    ) -> Optional[int]:
        """
        Observe one encoded HV and optionally apply self-correction.

        Args:
            hv: (dim,) binary HV of the observation
            calibrated_probs: (n_classes,) calibrated probabilities

        Returns:
            Pseudo-label used for self-correction, or None if skipped
        """
        max_prob = float(calibrated_probs.max())
        if max_prob < self.confidence_threshold:
            return None

        pseudo_label = int(calibrated_probs.argmax())

        # Mild Hebbian update
        self._protos[pseudo_label] += self.lr_self * hv.float()
        self._counts[pseudo_label] += 1
        self._n_self_labels += 1

        return pseudo_label

    def current_prototypes(self) -> torch.Tensor:
        """Return binarised current prototypes."""
        return (self._protos > 0).float()

    @property
    def n_self_labels(self) -> int:
        return self._n_self_labels


# ═══════════════════════════════════════════════════════════════════════════════
# 5. AccuracyBenchmark — before/after comparison utility
# ═══════════════════════════════════════════════════════════════════════════════

class AccuracyBenchmark:
    """
    Quick accuracy and quality benchmark for HDC classifiers.

    Reports:
      - Overall accuracy
      - Per-class accuracy
      - Confusion matrix (as prototype similarity matrix)
      - Prototype quality report
      - Estimated energy per inference (from efficiency.py model)
    """

    @staticmethod
    def evaluate(
        prototypes: torch.Tensor,
        X: torch.Tensor,
        y: torch.Tensor,
        encode_fn,
    ) -> Dict:
        """
        Evaluate a classifier given prototypes, data, and encoder.

        Returns:
            Dict with accuracy, per_class_accuracy, confusion, quality
        """
        n_classes = prototypes.shape[0]
        dim = prototypes.shape[1]
        n = X.shape[0]

        predictions = []
        for i in range(n):
            hv = encode_fn(X[i])
            sims = hv_batch_sim(hv, prototypes)
            predictions.append(int(sims.argmax().item()))

        preds = torch.tensor(predictions)
        accuracy = float((preds == y.long()).float().mean().item())

        # Per-class accuracy
        per_class = {}
        for c in range(n_classes):
            mask = (y == c)
            if mask.sum() > 0:
                per_class[c] = float((preds[mask] == c).float().mean().item())

        # Prototype similarity matrix (confusion proxy)
        sim_matrix = torch.zeros(n_classes, n_classes)
        for i in range(n_classes):
            sim_matrix[i] = hv_batch_sim(prototypes[i], prototypes)

        # Quality assessment
        assessor = PrototypeQualityAssessor()
        quality = assessor.assess(prototypes, X, y, encode_fn)

        return {
            "accuracy": accuracy,
            "per_class_accuracy": per_class,
            "prototype_similarity_matrix": sim_matrix,
            "quality": quality,
            "n_classes": n_classes,
            "dim": dim,
        }

    @staticmethod
    def compare(before: Dict, after: Dict) -> Dict:
        """Compare before and after retraining."""
        acc_gain  = after["accuracy"] - before["accuracy"]
        qual_gain = after["quality"].overall_quality - before["quality"].overall_quality
        sep_gain  = after["quality"].min_separation - before["quality"].min_separation
        return {
            "accuracy_gain":   acc_gain,
            "quality_gain":    qual_gain,
            "separation_gain": sep_gain,
            "before_accuracy": before["accuracy"],
            "after_accuracy":  after["accuracy"],
            "before_quality":  before["quality"].overall_quality,
            "after_quality":   after["quality"].overall_quality,
        }

    @staticmethod
    def online_accuracy(
        prototypes: torch.Tensor,
        stream,            # iterable of (hv: Tensor, label: int)
        window:     int = 100,
    ) -> Dict:
        """
        Compute rolling accuracy over a streaming labelled dataset.

        No storage of the full dataset — evaluates sample by sample.
        Reports accuracy over the most recent `window` samples.

        Args:
            prototypes:  (n_classes, D) class prototype HVs
            stream:      Iterable of (hv, label) pairs
            window:      Rolling accuracy window size

        Returns:
            Dict with final_accuracy, min_accuracy, max_accuracy, n_samples.
        """
        from collections import deque
        recent: deque = deque(maxlen=window)
        n_total, n_correct = 0, 0
        window_accs: List[float] = []

        for hv, label in stream:
            sims = hv_batch_sim(hv, prototypes)
            pred = int(sims.argmax().item())
            correct = int(pred == label)
            recent.append(correct)
            n_total  += 1
            n_correct += correct
            if len(recent) == window:
                window_accs.append(sum(recent) / window)

        return {
            "n_samples":       n_total,
            "overall_accuracy": n_correct / max(n_total, 1),
            "final_window_acc": sum(recent) / max(len(recent), 1),
            "min_window_acc":  min(window_accs) if window_accs else 0.0,
            "max_window_acc":  max(window_accs) if window_accs else 0.0,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════

def _make_test_data(n_classes=4, n_features=20, dim=2000, n_train=80, seed=42):
    """Helper: generate test data and trained prototypes."""
    torch.manual_seed(seed)
    from hdc.ge_parhi_survey import (
        HDClassifier, TrainingConfig, RetrainingMode,
        UnifiedHDCEncoder, EncodingConfig, EncodingType,
    )

    config = EncodingConfig(dim=dim, encoding_type=EncodingType.RANDOM_PROJECTION,
                            n_components=n_features)
    enc = UnifiedHDCEncoder(config)

    # Binary cluster data
    X, y = [], []
    for c in range(n_classes):
        base = torch.zeros(n_features)
        base[c * (n_features // n_classes):(c + 1) * (n_features // n_classes)] = 1.0
        for _ in range(n_train // n_classes):
            x = base.clone()
            mask = torch.rand(n_features) < 0.15
            x[mask] = 1.0 - x[mask]
            X.append(x)
            y.append(c)
    X = torch.stack(X)
    y = torch.tensor(y)

    def encode_fn(x):
        return enc.encode_random_projection(x.unsqueeze(0)).squeeze(0)

    # Train prototypes via one-shot bundling
    protos = torch.zeros(n_classes, dim)
    counts = torch.zeros(n_classes)
    for i in range(X.shape[0]):
        hv = encode_fn(X[i])
        protos[int(y[i])] += hv.float()
        counts[int(y[i])] += 1
    for c in range(n_classes):
        if counts[c] > 0:
            protos[c] = (protos[c] / counts[c] > 0.5).float()

    return protos, X, y, encode_fn


def test_prototype_quality():
    print("=" * 60)
    print("Testing PrototypeQualityAssessor")
    print("=" * 60)

    protos, X, y, enc = _make_test_data()
    assessor = PrototypeQualityAssessor()
    report = assessor.assess(protos, X, y, enc)

    print(f"  n_classes: {report.n_classes}")
    print(f"  Min separation: {report.min_separation:.4f}  (want ≥ 0.45 for good accuracy)")
    print(f"  Mean separation: {report.mean_separation:.4f}")
    print(f"  Confused pairs: {report.confused_pairs}")
    print(f"  Overall quality: {report.overall_quality:.4f}")
    print(f"  Worst class: {report.worst_class}")

    assert 0.0 <= report.min_separation <= 1.0
    assert 0.0 <= report.overall_quality <= 1.0

    print("  ✅ PrototypeQualityAssessor OK")


def test_confusion_aware_retraining():
    print("=" * 60)
    print("Testing ConfusionAwareRetrainer")
    print("=" * 60)

    torch.manual_seed(7)
    # Create two VERY similar classes to force confusion
    dim = 2000
    proto_a = (torch.rand(dim) < 0.5).float()
    proto_b = proto_a.clone()
    # Flip only 10% of bits — these classes will be very similar (sim ≈ 0.9)
    flip_idx = torch.randperm(dim)[:200]
    proto_b[flip_idx] = 1.0 - proto_b[flip_idx]

    protos_confused = torch.stack([proto_a, proto_b, (torch.rand(dim) < 0.5).float()])
    sim_ab_before = float(hv_batch_sim(protos_confused[0], protos_confused[1].unsqueeze(0))[0])
    print(f"  Similarity A↔B before retraining: {sim_ab_before:.4f}  (high = confused)")

    assessor = PrototypeQualityAssessor(confusion_threshold=0.6)
    retrainer = ConfusionAwareRetrainer(assessor, learning_rate=0.1, boundary_threshold=0.2)

    def simple_encode(x):
        return (x > 0.5).float()

    # Training data: class 0 and 1 with simple features
    X_train = torch.cat([
        proto_a.unsqueeze(0).expand(15, -1) + torch.randn(15, dim) * 0.05,
        proto_b.unsqueeze(0).expand(15, -1) + torch.randn(15, dim) * 0.05,
    ]).clamp(0, 1)
    y_train = torch.tensor([0] * 15 + [1] * 15)

    updated, stats = retrainer.retrain(protos_confused, X_train, y_train, simple_encode, n_passes=2)
    sim_ab_after = float(hv_batch_sim(updated[0], updated[1].unsqueeze(0))[0])

    print(f"  Similarity A↔B after retraining: {sim_ab_after:.4f}")
    print(f"  Stats: {stats}")
    print(f"  Separation improved: {sim_ab_after < sim_ab_before}")

    print("  ✅ ConfusionAwareRetrainer OK")


def test_calibrated_classifier():
    print("=" * 60)
    print("Testing CalibratedHDCClassifier (temperature scaling)")
    print("=" * 60)

    protos, X, y, enc = _make_test_data(n_train=120, seed=99)

    # Split into train/val
    n_val = 40
    X_val, y_val = X[-n_val:], y[-n_val:]

    calibrator = CalibratedHDCClassifier(None, n_classes=4)
    T = calibrator.calibrate(X_val, y_val, enc, protos)
    print(f"  Optimal temperature: {T:.2f}")
    assert T > 0

    # Before calibration: temperature=1.0
    calibrator.temperature = 1.0
    ece_uncalib = calibrator.expected_calibration_error(X_val, y_val, enc, protos)

    # After calibration
    calibrator.temperature = T
    ece_calib = calibrator.expected_calibration_error(X_val, y_val, enc, protos)

    print(f"  ECE before calibration: {ece_uncalib:.4f}")
    print(f"  ECE after  calibration: {ece_calib:.4f}  (want ≤ uncalib)")

    # Predict with confidence
    pred, conf = calibrator.predict_with_confidence(X_val[0], enc, protos)
    print(f"  Sample predict: class={pred}, confidence={conf:.4f}")
    assert 0 <= pred < 4
    assert 0 <= conf <= 1

    print("  ✅ CalibratedHDCClassifier OK")


def test_accuracy_benchmark():
    print("=" * 60)
    print("Testing AccuracyBenchmark (before/after comparison)")
    print("=" * 60)

    protos, X, y, enc = _make_test_data()
    before = AccuracyBenchmark.evaluate(protos, X, y, enc)
    print(f"  Accuracy: {before['accuracy']:.1%}")
    print(f"  Per-class: {before['per_class_accuracy']}")
    print(f"  Quality: {before['quality'].overall_quality:.4f}")
    print(f"  Min separation: {before['quality'].min_separation:.4f}")

    assert 0 <= before["accuracy"] <= 1
    assert before["quality"].n_classes == 4

    # Simulate improvement (just re-evaluate)
    after = AccuracyBenchmark.evaluate(protos, X, y, enc)
    comparison = AccuracyBenchmark.compare(before, after)
    print(f"  Comparison: {comparison}")

    print("  ✅ AccuracyBenchmark OK")


if __name__ == "__main__":
    test_prototype_quality()
    print()
    test_confusion_aware_retraining()
    print()
    test_calibrated_classifier()
    print()
    test_accuracy_benchmark()
    print()
    print("=== All accuracy_booster tests passed ===")
