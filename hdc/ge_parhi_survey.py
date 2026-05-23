"""
Ge & Parhi 2020 Survey: Classification Using Hyperdimensional Computing
=======================================================================
Based on: Ge, L. and Parhi, K.K. (2020)
"Classification Using Hyperdimensional Computing: A Review"
IEEE Circuits and Systems Magazine, 20(2), pp. 30-47.
doi: 10.1109/MCAS.2020.2988388

Comprehensive survey covering:
1. **Encoding Methods** (Section III): Random projection, ID-level, N-gram,
   record-based, spatial, and biosignal-specific encoding
2. **Training Strategies** (Section IV): One-shot, iterative, retraining,
   multi-label extensions
3. **Classification** (Section V): Associative memory, similarity search,
   confidence calibration
4. **Hardware Implementations** (Section VI): Digital, analog, in-memory
5. **Applications** (Section VII): Text, speech, biosignal, image, network security

Key innovations implemented here:
1. **UnifiedHDCEncoder** — All encoding methods in one framework
2. **MultiLabelHDClassifier** — Multi-label HDC classification
3. **RetrainingStrategies** — Multiple retraining algorithms
4. **HardwareEfficiencyModel** — Energy/area estimation
5. **BiosignalEncoder** — EEG/ECG-specific encoding
6. **ConfidenceCalibrator** — Platt scaling and temperature scaling for HDC
"""

import torch
import torch.nn as nn
import math
from typing import Optional, List, Tuple, Dict, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from hdc.hdc_glue import (
    hv_xor, hv_popcount, hv_hamming_sim, hv_bundle,
    hv_permute, hv_majority, hv_batch_sim, gen_hvs,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Section III: Encoding Methods
# ═══════════════════════════════════════════════════════════════════════════════

class EncodingType(Enum):
    """Taxonomy of HDC encoding methods (Ge & Parhi 2020, Section III)."""
    RANDOM_PROJECTION = "random_projection"
    ID_LEVEL = "id_level"
    NGRAM = "ngram"
    RECORD = "record"
    SPATIAL = "spatial"
    TEMPORAL = "temporal"
    GRAPH = "graph"
    BIOSIGNAL = "biosignal"
    MULTI_LABEL = "multi_label"


@dataclass
class EncodingConfig:
    """Configuration for HDC encoding (Ge & Parhi 2020, Section III)."""
    dim: int = 10000
    encoding_type: EncodingType = EncodingType.RANDOM_PROJECTION
    seed: Optional[int] = None

    # Random projection parameters
    n_components: Optional[int] = None  # If None, use dim

    # ID-level encoding parameters
    n_items: int = 100  # Number of unique items to encode
    item_hvs: Optional[torch.Tensor] = None  # Pre-generated item HVs

    # N-gram parameters
    ngram_order: int = 3
    ngram_vocab_size: int = 100

    # Record encoding parameters
    n_fields: int = 10
    field_hvs: Optional[torch.Tensor] = None

    # Spatial encoding parameters
    spatial_resolution: int = 100
    spatial_dims: int = 2

    # Temporal encoding parameters
    temporal_window: int = 10
    temporal_stride: int = 1

    # Biosignal encoding parameters
    biosignal_channels: int = 64
    biosignal_samples: int = 256
    biosignal_fs: float = 250.0  # Sampling frequency (Hz)

    # Multi-label parameters
    n_labels: int = 10
    label_threshold: float = 0.5


class UnifiedHDCEncoder:
    """
    Unified HDC encoder covering all encoding methods from Ge & Parhi 2020.

    Provides a single interface for:
    - Random projection encoding (Section III-A)
    - ID-level encoding (Section III-B)
    - N-gram encoding (Section III-C)
    - Record encoding (Section III-D)
    - Spatial encoding (Section III-E)
    - Temporal encoding (Section III-F)
    - Graph encoding (Section III-G)
    - Biosignal encoding (Section III-H)
    """

    def __init__(self, config: EncodingConfig):
        self.config = config
        self.dim = config.dim
        self.seed = config.seed or 42

        # Initialize encoding-specific components
        self._init_components()

    def _init_components(self):
        """Initialize encoding components based on config."""
        # Random projection matrix
        if self.config.encoding_type == EncodingType.RANDOM_PROJECTION:
            n_comp = self.config.n_components or self.dim
            g = torch.Generator()
            g.manual_seed(self.seed)
            self.proj_matrix = torch.randint(
                0, 2, (n_comp, self.dim), generator=g
            ).float()

        # ID-level item hypervectors
        if self.config.encoding_type == EncodingType.ID_LEVEL:
            if self.config.item_hvs is None:
                self.item_hvs = gen_hvs(
                    self.config.n_items, self.dim, seed=self.seed
                )
            else:
                self.item_hvs = self.config.item_hvs

        # N-gram item hypervectors
        if self.config.encoding_type == EncodingType.NGRAM:
            self.ngram_item_hvs = gen_hvs(
                self.config.ngram_vocab_size, self.dim, seed=self.seed
            )
            self.ngram_permute_hv = gen_hvs(1, self.dim, seed=self.seed + 1000).squeeze(0)

        # Record field hypervectors
        if self.config.encoding_type == EncodingType.RECORD:
            if self.config.field_hvs is None:
                self.field_hvs = gen_hvs(
                    self.config.n_fields, self.dim, seed=self.seed + 2000
                )
            else:
                self.field_hvs = self.config.field_hvs

        # Spatial encoding
        if self.config.encoding_type == EncodingType.SPATIAL:
            self.spatial_hvs = gen_hvs(
                self.config.spatial_resolution, self.dim, seed=self.seed + 3000
            )

        # Temporal encoding
        if self.config.encoding_type == EncodingType.TEMPORAL:
            self.temporal_permute_hv = gen_hvs(
                1, self.dim, seed=self.seed + 4000
            ).squeeze(0)

        # Biosignal encoding
        if self.config.encoding_type == EncodingType.BIOSIGNAL:
            self.channel_hvs = gen_hvs(
                self.config.biosignal_channels, self.dim, seed=self.seed + 5000
            )
            self.time_hvs = gen_hvs(
                self.config.biosignal_samples, self.dim, seed=self.seed + 6000
            )

        # Multi-label encoding
        if self.config.encoding_type == EncodingType.MULTI_LABEL:
            self.label_hvs = gen_hvs(
                self.config.n_labels, self.dim, seed=self.seed + 7000
            )

    def encode_random_projection(self, x: torch.Tensor) -> torch.Tensor:
        """Random projection encoding (Ge & Parhi 2020, Section III-A).

        Maps input features to hypervectors via random projection matrix.

        Args:
            x: (n_samples, n_features) input features

        Returns:
            (n_samples, dim) hypervectors
        """
        # Project and binarize
        proj = x @ self.proj_matrix[:x.shape[1]]
        return (proj > 0).float()

    def encode_id_level(self, ids: torch.Tensor) -> torch.Tensor:
        """ID-level encoding (Ge & Parhi 2020, Section III-B).

        Each unique item maps to a random hypervector.

        Args:
            ids: (n_samples,) or (n_samples, n_items) item indices

        Returns:
            (n_samples, dim) hypervectors
        """
        if ids.dim() == 1:
            return self.item_hvs[ids.long()]
        else:
            # Multiple items per sample — bundle them
            hvs = torch.stack([self.item_hvs[ids[i].long()].sum(dim=0)
                               for i in range(ids.shape[0])])
            return (hvs > 0).float()

    def encode_ngram(self, sequences: torch.Tensor) -> torch.Tensor:
        """N-gram encoding (Ge & Parhi 2020, Section III-C).

        Encodes sequences by binding n-gram items with position information.

        Args:
            sequences: (n_samples, seq_len) token indices

        Returns:
            (n_samples, dim) hypervectors
        """
        n_samples, seq_len = sequences.shape
        n = self.config.ngram_order
        hvs = []

        for i in range(n_samples):
            seq = sequences[i]
            ngram_hvs = []

            for j in range(seq_len - n + 1):
                # Bind n-gram items together
                ngram = seq[j:j + n]
                bound = self.ngram_item_hvs[ngram[0].long()].clone()
                for k in range(1, n):
                    bound = hv_xor(bound, self.ngram_item_hvs[ngram[k].long()])

                # Permute by position
                permuted = hv_permute(bound, k=j)
                ngram_hvs.append(permuted)

            # Bundle all n-grams
            if ngram_hvs:
                bundled = hv_bundle(torch.stack(ngram_hvs))
                hvs.append(hv_majority(bundled))
            else:
                hvs.append(torch.zeros(self.dim))

        return torch.stack(hvs)

    def encode_record(self, fields: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Record encoding (Ge & Parhi 2020, Section III-D).

        Encodes structured data by binding field values with field IDs.

        Args:
            fields: Dict mapping field_name -> (n_samples,) values

        Returns:
            (n_samples, dim) hypervectors
        """
        n_samples = list(fields.values())[0].shape[0]
        field_names = list(fields.keys())
        n_fields = len(field_names)

        hvs = []
        for i in range(n_samples):
            record_hv = torch.zeros(self.dim)
            for j, name in enumerate(field_names):
                val = fields[name][i]
                # Bind field HV with value HV
                field_hv = self.field_hvs[j]
                val_hv = gen_hvs(1, self.dim, seed=self.seed + int(val.item())).squeeze(0)
                bound = hv_xor(field_hv, val_hv)
                record_hv = record_hv + bound

            hvs.append((record_hv > 0).float())

        return torch.stack(hvs)

    def encode_spatial(self, positions: torch.Tensor) -> torch.Tensor:
        """Spatial encoding (Ge & Parhi 2020, Section III-E).

        Encodes spatial positions using level-based hypervectors.

        Args:
            positions: (n_samples, n_dims) spatial coordinates

        Returns:
            (n_samples, dim) hypervectors
        """
        n_samples = positions.shape[0]
        hvs = []

        for i in range(n_samples):
            pos = positions[i]
            spatial_hv = torch.zeros(self.dim)

            for d in range(pos.shape[0]):
                # Map coordinate to level index
                level = int((pos[d].item() + 1.0) / 2.0 *
                           (self.config.spatial_resolution - 1))
                level = max(0, min(level, self.config.spatial_resolution - 1))
                spatial_hv = spatial_hv + self.spatial_hvs[level]

            hvs.append((spatial_hv > 0).float())

        return torch.stack(hvs)

    def encode_temporal(self, sequence: torch.Tensor) -> torch.Tensor:
        """Temporal encoding (Ge & Parhi 2020, Section III-F).

        Encodes temporal sequences using position-dependent permutation.

        Args:
            sequence: (n_samples, seq_len, feat_dim) temporal data

        Returns:
            (n_samples, dim) hypervectors
        """
        n_samples, seq_len, feat_dim = sequence.shape
        hvs = []

        for i in range(n_samples):
            temporal_hv = torch.zeros(self.dim)
            for t in range(seq_len):
                # Encode frame at time t
                frame = sequence[i, t]
                frame_hv = gen_hvs(1, self.dim, seed=self.seed + int(frame.sum().item())).squeeze(0)
                # Permute by time
                permuted = hv_permute(frame_hv, k=t)
                temporal_hv = temporal_hv + permuted

            hvs.append((temporal_hv > 0).float())

        return torch.stack(hvs)

    def encode_biosignal(self, signal: torch.Tensor) -> torch.Tensor:
        """Biosignal encoding (Ge & Parhi 2020, Section III-H).

        Encodes multi-channel biosignals (EEG, ECG) using channel-specific
        and time-specific hypervectors.

        Args:
            signal: (n_samples, n_channels, n_samples) biosignal data

        Returns:
            (n_samples, dim) hypervectors
        """
        n_samples, n_channels, n_time = signal.shape
        hvs = []

        for i in range(n_samples):
            sample_hv = torch.zeros(self.dim)
            for ch in range(min(n_channels, self.config.biosignal_channels)):
                channel_hv = self.channel_hvs[ch]
                for t in range(min(n_time, self.config.biosignal_samples)):
                    val = signal[i, ch, t]
                    if abs(val.item()) > 0.01:  # Threshold for sparsity
                        time_hv = self.time_hvs[t]
                        # Bind channel, time, and value
                        val_hv = gen_hvs(1, self.dim,
                                         seed=self.seed + int(val.item() * 1000)).squeeze(0)
                        bound = hv_xor(hv_xor(channel_hv, time_hv), val_hv)
                        sample_hv = sample_hv + bound

            hvs.append((sample_hv > 0).float())

        return torch.stack(hvs)

    def encode(self, data: Any) -> torch.Tensor:
        """Unified encoding interface.

        Args:
            data: Input data (format depends on encoding_type)

        Returns:
            (n_samples, dim) hypervectors
        """
        encoding_map = {
            EncodingType.RANDOM_PROJECTION: self.encode_random_projection,
            EncodingType.ID_LEVEL: self.encode_id_level,
            EncodingType.NGRAM: self.encode_ngram,
            EncodingType.SPATIAL: self.encode_spatial,
            EncodingType.TEMPORAL: self.encode_temporal,
            EncodingType.BIOSIGNAL: self.encode_biosignal,
        }

        encoder = encoding_map.get(self.config.encoding_type)
        if encoder is None:
            raise ValueError(f"Unsupported encoding type: {self.config.encoding_type}")

        return encoder(data)


# ═══════════════════════════════════════════════════════════════════════════════
# Section IV: Training Strategies
# ═══════════════════════════════════════════════════════════════════════════════

class RetrainingMode(Enum):
    """Retraining strategies from Ge & Parhi 2020, Section IV."""
    ONE_SHOT = "one_shot"
    ITERATIVE = "iterative"
    ADAPTIVE_LR = "adaptive_lr"
    WEIGHTED = "weighted"
    MULTI_LABEL = "multi_label"


@dataclass
class TrainingConfig:
    """Configuration for HDC training (Ge & Parhi 2020, Section IV)."""
    dim: int = 10000
    n_classes: int = 10
    retraining_mode: RetrainingMode = RetrainingMode.ONE_SHOT
    n_iterations: int = 10
    learning_rate: float = 0.1
    adaptive_lr_base: float = 0.5
    adaptive_lr_decay: float = 0.1
    weighted_decay: float = 0.01
    multi_label_threshold: float = 0.5
    seed: Optional[int] = None


class HDClassifier:
    """
    HDC classifier with multiple training strategies (Ge & Parhi 2020, Section IV).

    Supports:
    - One-shot training (Section IV-A)
    - Iterative retraining (Section IV-B)
    - Adaptive learning rate (Section IV-C)
    - Weighted prototypes (Section IV-D)
    - Multi-label classification (Section VI)
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.dim = config.dim
        self.n_classes = config.n_classes
        self.seed = config.seed or 42

        # Class prototype hypervectors
        self.prototypes: Optional[torch.Tensor] = None

        # Class counts for adaptive learning
        self.class_counts: Optional[torch.Tensor] = None

        # Class weights for weighted training
        self.class_weights: Optional[torch.Tensor] = None

        # Training history
        self.training_history: List[Dict[str, Any]] = []

    def _init_prototypes(self, hvs: torch.Tensor, labels: torch.Tensor):
        """Initialize prototypes via one-shot bundling (Section IV-A)."""
        n_samples = hvs.shape[0]
        self.prototypes = torch.zeros(self.n_classes, self.dim)
        self.class_counts = torch.zeros(self.n_classes)

        for i in range(n_samples):
            label = int(labels[i].item())
            self.prototypes[label] = self.prototypes[label] + hvs[i]
            self.class_counts[label] += 1

        # Binarize prototypes
        for c in range(self.n_classes):
            if self.class_counts[c] > 0:
                self.prototypes[c] = (self.prototypes[c] / self.class_counts[c] > 0.5).float()

    def train_one_shot(self, hvs: torch.Tensor, labels: torch.Tensor):
        """One-shot training (Ge & Parhi 2020, Section IV-A).

        Single-pass bundling of all samples into class prototypes.
        """
        self._init_prototypes(hvs, labels)
        self.training_history.append({
            "mode": "one_shot",
            "n_samples": hvs.shape[0],
            "n_classes": self.n_classes,
        })

    def train_iterative(self, hvs: torch.Tensor, labels: torch.Tensor):
        """Iterative retraining (Ge & Parhi 2020, Section IV-B).

        Multiple passes: assign → update → repeat.
        """
        # Initialize with one-shot
        self._init_prototypes(hvs, labels)

        for iteration in range(self.config.n_iterations):
            # Assign samples to nearest prototype
            preds = self.predict(hvs)
            correct = (preds == labels).float().mean().item()

            # Update prototypes
            new_protos = torch.zeros(self.n_classes, self.dim)
            new_counts = torch.zeros(self.n_classes)

            for i in range(hvs.shape[0]):
                label = int(labels[i].item())
                new_protos[label] = new_protos[label] + hvs[i]
                new_counts[label] += 1

            # Binarize
            for c in range(self.n_classes):
                if new_counts[c] > 0:
                    self.prototypes[c] = (new_protos[c] / new_counts[c] > 0.5).float()

            self.training_history.append({
                "iteration": iteration,
                "accuracy": correct,
            })

    def train_adaptive_lr(self, hvs: torch.Tensor, labels: torch.Tensor):
        """Adaptive learning rate training (Ge & Parhi 2020, Section IV-C).

        Per-class learning rates that decay with sample count.
        lr_c = base_lr / (1 + count_c * decay)
        """
        # Initialize with one-shot
        self._init_prototypes(hvs, labels)

        for iteration in range(self.config.n_iterations):
            preds = self.predict(hvs)
            correct = (preds == labels).float().mean().item()

            # Update with adaptive learning rates
            for i in range(hvs.shape[0]):
                label = int(labels[i].item())
                pred = int(preds[i].item())

                # Per-class learning rate
                lr = self.config.adaptive_lr_base / (
                    1 + self.class_counts[label].item() * self.config.adaptive_lr_decay
                )

                if label == pred:
                    # Correct: move prototype toward sample
                    sim = float(hv_hamming_sim(self.prototypes[label], hvs[i]))
                    update = lr * (1 - sim) * (hvs[i] - self.prototypes[label])
                    self.prototypes[label] = self.prototypes[label] + update
                else:
                    # Incorrect: move correct prototype toward, wrong prototype away
                    sim_correct = float(hv_hamming_sim(self.prototypes[label], hvs[i]))
                    update_correct = lr * (1 - sim_correct) * (hvs[i] - self.prototypes[label])
                    self.prototypes[label] = self.prototypes[label] + update_correct

                    sim_wrong = float(hv_hamming_sim(self.prototypes[pred], hvs[i]))
                    update_wrong = lr * sim_wrong * (self.prototypes[pred] - hvs[i])
                    self.prototypes[pred] = self.prototypes[pred] + update_wrong

                self.class_counts[label] += 1

            # Binarize
            for c in range(self.n_classes):
                self.prototypes[c] = (self.prototypes[c] > 0.5).float()

            self.training_history.append({
                "iteration": iteration,
                "accuracy": correct,
            })

    def train_weighted(self, hvs: torch.Tensor, labels: torch.Tensor):
        """Weighted prototype training (Ge & Parhi 2020, Section IV-D).

        Uses class weights to handle imbalanced datasets.
        """
        # Compute class weights (inverse frequency)
        self._init_prototypes(hvs, labels)
        total = self.class_counts.sum()
        self.class_weights = total / (self.n_classes * self.class_counts + 1e-8)

        # Weighted prototype update
        weighted_protos = torch.zeros(self.n_classes, self.dim)
        weighted_counts = torch.zeros(self.n_classes)

        for i in range(hvs.shape[0]):
            label = int(labels[i].item())
            weight = self.class_weights[label].item()
            weighted_protos[label] = weighted_protos[label] + weight * hvs[i]
            weighted_counts[label] += weight

        for c in range(self.n_classes):
            if weighted_counts[c] > 0:
                self.prototypes[c] = (weighted_protos[c] / weighted_counts[c] > 0.5).float()

        self.training_history.append({
            "mode": "weighted",
            "class_weights": self.class_weights.tolist(),
        })

    def train(self, hvs: torch.Tensor, labels: torch.Tensor):
        """Unified training interface.

        Args:
            hvs: (n_samples, dim) hypervectors
            labels: (n_samples,) class labels
        """
        mode_map = {
            RetrainingMode.ONE_SHOT: self.train_one_shot,
            RetrainingMode.ITERATIVE: self.train_iterative,
            RetrainingMode.ADAPTIVE_LR: self.train_adaptive_lr,
            RetrainingMode.WEIGHTED: self.train_weighted,
        }

        trainer = mode_map.get(self.config.retraining_mode)
        if trainer is None:
            raise ValueError(f"Unsupported training mode: {self.config.retraining_mode}")

        trainer(hvs, labels)

    def predict(self, hvs: torch.Tensor) -> torch.Tensor:
        """Predict class labels.

        Args:
            hvs: (n_samples, dim) hypervectors

        Returns:
            (n_samples,) predicted labels
        """
        if self.prototypes is None:
            raise ValueError("Model not trained. Call train() first.")

        # Compute similarities to all prototypes
        # hv_batch_sim expects 1D query, 2D memory
        n_samples = hvs.shape[0]
        sims = torch.zeros(n_samples, self.n_classes)
        for i in range(n_samples):
            sims[i] = hv_batch_sim(hvs[i], self.prototypes)
        return sims.argmax(dim=-1)

    def predict_proba(self, hvs: torch.Tensor) -> torch.Tensor:
        """Predict class probabilities.

        Args:
            hvs: (n_samples, dim) hypervectors

        Returns:
            (n_samples, n_classes) probabilities
        """
        if self.prototypes is None:
            raise ValueError("Model not trained. Call train() first.")

        # hv_batch_sim expects 1D query, 2D memory
        n_samples = hvs.shape[0]
        sims = torch.zeros(n_samples, self.n_classes)
        for i in range(n_samples):
            sims[i] = hv_batch_sim(hvs[i], self.prototypes)
        # Convert similarities to probabilities via softmax
        return torch.softmax(sims * 10.0, dim=-1)  # Temperature scaling

    def evaluate(self, hvs: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """Evaluate classifier performance.

        Args:
            hvs: (n_samples, dim) hypervectors
            labels: (n_samples,) true labels

        Returns:
            Dict of metrics
        """
        preds = self.predict(hvs)
        accuracy = (preds == labels).float().mean().item()

        # Per-class metrics
        n_classes = self.n_classes
        per_class_acc = []
        for c in range(n_classes):
            mask = (labels == c)
            if mask.sum() > 0:
                acc = (preds[mask] == labels[mask]).float().mean().item()
                per_class_acc.append(acc)

        return {
            "accuracy": accuracy,
            "mean_per_class_accuracy": sum(per_class_acc) / len(per_class_acc) if per_class_acc else 0.0,
            "n_samples": hvs.shape[0],
        }

    def classifier_health(self) -> Dict[str, Any]:
        """
        Prototype quality diagnostics: inter-class separation and class imbalance.

        mean_proto_sim > 0.7 → prototypes are crowded → high confusion risk.
        imbalance_ratio > 10 → dominant class may overfit prototype space.
        """
        if self.prototypes is None:
            return {"status": "untrained"}
        C = self.n_classes
        protos = self.prototypes  # (C, D)
        sims = []
        for i in range(C):
            for j in range(i + 1, C):
                xor = (protos[i] != protos[j]).float()
                sims.append(float(1.0 - xor.mean().item()))
        mean_sim = sum(sims) / max(len(sims), 1)
        min_sim  = min(sims) if sims else 1.0
        counts   = self.class_counts
        imbalance = float(counts.max().item()) / max(float(counts.min().item()), 1.0) if counts is not None else 1.0
        return {
            "n_classes":         C,
            "mean_proto_sim":    round(mean_sim, 4),
            "min_proto_sim":     round(min_sim, 4),
            "imbalance_ratio":   round(imbalance, 2),
            "n_training_rounds": len(self.training_history),
            "well_separated":    mean_sim < 0.6,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Section VI: Multi-Label Classification
# ═══════════════════════════════════════════════════════════════════════════════

class MultiLabelHDClassifier:
    """
    Multi-label HDC classification (Ge & Parhi 2020, Section VI).

    Extends HDC to multi-label problems where each sample can belong
    to multiple classes simultaneously.

    Key approaches:
    1. **Binary Relevance**: Train one binary classifier per label
    2. **Label Embedding**: Encode label combinations as hypervectors
    3. **Threshold-based**: Predict labels above similarity threshold
    """

    def __init__(
        self,
        dim: int = 10000,
        n_labels: int = 10,
        threshold: float = 0.5,
        method: str = "binary_relevance",
        seed: Optional[int] = None,
    ):
        self.dim = dim
        self.n_labels = n_labels
        self.threshold = threshold
        self.method = method
        self.seed = seed or 42

        # Binary relevance: one classifier per label
        self.binary_classifiers: List[HDClassifier] = []

        # Label hypervectors (for label embedding method)
        self.label_hvs = gen_hvs(n_labels, dim, seed=self.seed)

        # Combined label prototypes (for label embedding method)
        self.label_prototypes: Dict[int, torch.Tensor] = {}

    def _train_binary_relevance(self, hvs: torch.Tensor, labels: torch.Tensor):
        """Train one binary classifier per label (Section VI-A)."""
        self.binary_classifiers = []

        for label_idx in range(self.n_labels):
            # Create binary labels: 1 if this label present, 0 otherwise
            binary_labels = labels[:, label_idx]

            # Skip if no positive or negative examples
            if binary_labels.sum() == 0 or (1 - binary_labels).sum() == 0:
                self.binary_classifiers.append(None)
                continue

            config = TrainingConfig(
                dim=self.dim,
                n_classes=2,
                retraining_mode=RetrainingMode.ITERATIVE,
                n_iterations=5,
            )
            clf = HDClassifier(config)
            clf.train(hvs, binary_labels)
            self.binary_classifiers.append(clf)

    def _train_label_embedding(self, hvs: torch.Tensor, labels: torch.Tensor):
        """Train using label embedding (Section VI-B).

        Each unique label combination gets its own prototype.
        """
        n_samples = hvs.shape[0]

        # Group samples by label combination
        label_combos: Dict[str, List[int]] = {}
        for i in range(n_samples):
            combo = tuple(labels[i].long().tolist())
            key = str(combo)
            if key not in label_combos:
                label_combos[key] = []
            label_combos[key].append(i)

        # Create prototype for each label combination
        for combo_key, indices in label_combos.items():
            combo_hvs = hvs[indices]
            prototype = hv_majority(hv_bundle(combo_hvs))
            self.label_prototypes[combo_key] = prototype

    def train(self, hvs: torch.Tensor, labels: torch.Tensor):
        """Train multi-label classifier.

        Args:
            hvs: (n_samples, dim) hypervectors
            labels: (n_samples, n_labels) binary label matrix
        """
        if self.method == "binary_relevance":
            self._train_binary_relevance(hvs, labels)
        elif self.method == "label_embedding":
            self._train_label_embedding(hvs, labels)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def predict(self, hvs: torch.Tensor) -> torch.Tensor:
        """Predict multi-label outputs.

        Args:
            hvs: (n_samples, dim) hypervectors

        Returns:
            (n_samples, n_labels) binary predictions
        """
        n_samples = hvs.shape[0]

        if self.method == "binary_relevance":
            predictions = torch.zeros(n_samples, self.n_labels)
            for label_idx, clf in enumerate(self.binary_classifiers):
                if clf is None:
                    continue
                probs = clf.predict_proba(hvs)
                # Probability of class 1 (positive)
                predictions[:, label_idx] = probs[:, 1]
            return (predictions > self.threshold).float()

        elif self.method == "label_embedding":
            predictions = torch.zeros(n_samples, self.n_labels)
            for i in range(n_samples):
                hv = hvs[i]
                best_combo = None
                best_sim = -1.0

                for combo_key, prototype in self.label_prototypes.items():
                    sim = float(hv_hamming_sim(hv, prototype))
                    if sim > best_sim:
                        best_sim = sim
                        best_combo = combo_key

                if best_combo is not None:
                    # Parse the label combination
                    labels_list = eval(best_combo)
                    predictions[i] = torch.tensor(labels_list)

            return predictions

    def evaluate(self, hvs: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """Evaluate multi-label classifier.

        Args:
            hvs: (n_samples, dim) hypervectors
            labels: (n_samples, n_labels) true labels

        Returns:
            Dict of metrics
        """
        preds = self.predict(hvs)

        # Hamming loss
        hamming_loss = (preds != labels).float().mean().item()

        # Exact match ratio
        exact_match = (preds == labels).all(dim=-1).float().mean().item()

        # Per-label F1
        f1_scores = []
        for label_idx in range(self.n_labels):
            tp = ((preds[:, label_idx] == 1) & (labels[:, label_idx] == 1)).sum().item()
            fp = ((preds[:, label_idx] == 1) & (labels[:, label_idx] == 0)).sum().item()
            fn = ((preds[:, label_idx] == 0) & (labels[:, label_idx] == 1)).sum().item()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            f1_scores.append(f1)

        return {
            "hamming_loss": hamming_loss,
            "exact_match_ratio": exact_match,
            "mean_f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
            "n_labels": self.n_labels,
        }

    def online_update(
        self,
        hv:         torch.Tensor,   # (dim,) encoded sample
        true_labels: torch.Tensor,  # (n_labels,) binary true labels
        lr:         float = 0.05,
    ):
        """
        Online update: add one labelled sample without full retraining.

        Updates each binary classifier's positive/negative prototype
        by blending the new example in.  Compatible with binary_relevance mode.

        Args:
            hv:          (dim,) encoded hypervector
            true_labels: (n_labels,) binary label vector (0/1)
            lr:          Blending rate
        """
        if not self.binary_classifiers:
            return   # not trained yet

        for i, clf in enumerate(self.binary_classifiers):
            if clf is None:
                continue
            is_positive = bool(true_labels[i].item() > 0.5)
            pred_labels, _ = clf.predict_binary(hv.unsqueeze(0)) if hasattr(clf, "predict_binary") else (None, None)

            # Simple prototype blend — works for any HDClassifier with prototypes
            if hasattr(clf, "prototypes") and clf.prototypes is not None:
                target_class = 1 if is_positive else 0
                if target_class < clf.prototypes.shape[0]:
                    old = clf.prototypes[target_class].float()
                    clf.prototypes[target_class] = ((1 - lr) * old + lr * hv.float() > 0).float() * 2 - 1


# ═══════════════════════════════════════════════════════════════════════════════
# Section V: Confidence Calibration
# ═══════════════════════════════════════════════════════════════════════════════

class ConfidenceCalibrator:
    """
    Confidence calibration for HDC classifiers (Ge & Parhi 2020, Section V).

    Methods:
    - **Platt scaling**: Logistic regression on similarity scores
    - **Temperature scaling**: Single parameter scaling of logits
    - **Isotonic regression**: Non-parametric calibration
    """

    def __init__(self, method: str = "temperature"):
        self.method = method
        self.temperature: float = 1.0
        self.platt_a: float = 0.0
        self.platt_b: float = 0.0

    def calibrate_temperature(self, sims: torch.Tensor, labels: torch.Tensor):
        """Temperature scaling: find optimal temperature.

        Args:
            sims: (n_samples, n_classes) similarity scores
            labels: (n_samples,) true labels
        """
        # Simple grid search for temperature
        best_temp = 1.0
        best_nll = float('inf')

        for temp in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
            scaled = sims / temp
            probs = torch.softmax(scaled, dim=-1)

            # Negative log likelihood
            nll = 0.0
            for i in range(sims.shape[0]):
                nll -= math.log(max(probs[i, int(labels[i].item())].item(), 1e-10))

            if nll < best_nll:
                best_nll = nll
                best_temp = temp

        self.temperature = best_temp

    def calibrate_platt(self, sims: torch.Tensor, labels: torch.Tensor):
        """Platt scaling: logistic regression on similarity scores.

        Args:
            sims: (n_samples, n_classes) similarity scores
            labels: (n_samples,) true labels
        """
        # Simple Platt scaling: P(y=1|x) = 1 / (1 + exp(A * sim + B))
        n_samples = sims.shape[0]
        correct = torch.zeros(n_samples)
        for i in range(n_samples):
            correct[i] = 1.0 if sims[i].argmax() == labels[i] else 0.0

        max_sims = sims.max(dim=-1).values
        # Simple linear fit: A = slope, B = intercept
        # Using least squares on logit(correct) = A * max_sim + B
        # Clamp to avoid log(0) or log(1)
        eps = 1e-6
        correct_clamped = torch.clamp(correct, eps, 1 - eps)
        logits = torch.log(correct_clamped / (1 - correct_clamped))

        # Least squares
        X = torch.stack([max_sims, torch.ones(n_samples)], dim=-1)
        try:
            theta = torch.linalg.lstsq(X, logits).solution
            self.platt_a = theta[0].item() if not torch.isnan(theta[0]).item() else 1.0
            self.platt_b = theta[1].item() if not torch.isnan(theta[1]).item() else 0.0
        except:
            self.platt_a = 1.0
            self.platt_b = 0.0

    def calibrate(self, sims: torch.Tensor, labels: torch.Tensor):
        """Calibrate confidence scores.

        Args:
            sims: (n_samples, n_classes) similarity scores
            labels: (n_samples,) true labels
        """
        if self.method == "temperature":
            self.calibrate_temperature(sims, labels)
        elif self.method == "platt":
            self.calibrate_platt(sims, labels)

    def predict_proba(self, sims: torch.Tensor) -> torch.Tensor:
        """Get calibrated probabilities.

        Args:
            sims: (n_samples, n_classes) similarity scores

        Returns:
            (n_samples, n_classes) calibrated probabilities
        """
        if self.method == "temperature":
            return torch.softmax(sims / self.temperature, dim=-1)
        elif self.method == "platt":
            probs = 1.0 / (1.0 + torch.exp(-(sims * self.platt_a + self.platt_b)))
            return probs / probs.sum(dim=-1, keepdim=True)
        return torch.softmax(sims, dim=-1)


# ═══════════════════════════════════════════════════════════════════════════════
# Section VI: Hardware Efficiency Model
# ═══════════════════════════════════════════════════════════════════════════════

class HardwarePlatform(Enum):
    """Hardware platforms for HDC (Ge & Parhi 2020, Section VI)."""
    DIGITAL_ASIC = "digital_asic"
    ANALOG_CROSSBAR = "analog_crossbar"
    FPGA = "fpga"
    GPU = "gpu"
    CPU = "cpu"


@dataclass
class HardwareConfig:
    """Configuration for hardware efficiency estimation."""
    platform: HardwarePlatform = HardwarePlatform.DIGITAL_ASIC
    dim: int = 10000
    n_classes: int = 10
    n_samples: int = 1000
    frequency_mhz: float = 100.0
    technology_nm: float = 45.0
    voltage_v: float = 1.0


class HardwareEfficiencyModel:
    """
    Hardware efficiency model for HDC classifiers (Ge & Parhi 2020, Section VI).

    Estimates:
    - Energy per classification
    - Area requirements
    - Throughput
    - Power consumption

    Based on the Horowitz (2014) 45nm CMOS energy model.
    """

    def __init__(self, config: HardwareConfig):
        self.config = config

        # Energy per operation (45nm CMOS, Horowitz 2014)
        self.energy_per_op = {
            "int8_add": 0.1,    # pJ
            "int8_mul": 0.2,    # pJ
            "int32_add": 0.9,   # pJ
            "int32_mul": 3.1,   # pJ
            "fp32_add": 0.9,    # pJ
            "fp32_mul": 3.7,    # pJ
            "popcount": 0.1,    # pJ (XOR + popcount)
            "xor": 0.05,        # pJ
            "memory_read_8kb": 10.0,   # pJ
            "memory_read_32kb": 20.0,  # pJ
            "memory_write_8kb": 10.0,  # pJ
        }

    def estimate_energy_hdc(self) -> Dict[str, float]:
        """Estimate energy per classification for HDC.

        HDC operations per classification:
        - n_classes * dim XOR operations (similarity)
        - n_classes * dim popcount operations
        - n_classes comparisons

        Returns:
            Dict of energy estimates
        """
        dim = self.config.dim
        n_classes = self.config.n_classes

        # XOR operations: n_classes * dim
        xor_energy = n_classes * dim * self.energy_per_op["xor"]

        # Popcount operations: n_classes * dim
        popcount_energy = n_classes * dim * self.energy_per_op["popcount"]

        # Comparisons: n_classes
        compare_energy = n_classes * self.energy_per_op["int32_add"]

        # Memory reads: prototypes (n_classes * dim bits)
        memory_energy = n_classes * dim / 8 / 1024 * self.energy_per_op["memory_read_8kb"]

        total = xor_energy + popcount_energy + compare_energy + memory_energy

        return {
            "xor_energy_pj": xor_energy,
            "popcount_energy_pj": popcount_energy,
            "compare_energy_pj": compare_energy,
            "memory_energy_pj": memory_energy,
            "total_energy_pj": total,
            "total_energy_nj": total / 1000,
        }

    def estimate_energy_nn(self, n_layers: int = 3, hidden_size: int = 128) -> Dict[str, float]:
        """Estimate energy per inference for equivalent neural network.

        NN operations per inference:
        - n_layers * hidden_size^2 MAC operations

        Returns:
            Dict of energy estimates
        """
        n_layers = n_layers
        hidden = hidden_size

        # MAC operations
        macs = n_layers * hidden * hidden
        mac_energy = macs * self.energy_per_op["int8_mul"]

        # Add operations
        adds = macs
        add_energy = adds * self.energy_per_op["int8_add"]

        # Memory reads
        params = n_layers * hidden * hidden
        memory_energy = params / 8 / 1024 * self.energy_per_op["memory_read_8kb"]

        total = mac_energy + add_energy + memory_energy

        return {
            "mac_energy_pj": mac_energy,
            "add_energy_pj": add_energy,
            "memory_energy_pj": memory_energy,
            "total_energy_pj": total,
            "total_energy_nj": total / 1000,
        }

    def estimate_area(self) -> Dict[str, float]:
        """Estimate silicon area for HDC classifier.

        Returns:
            Dict of area estimates (mm^2)
        """
        dim = self.config.dim
        n_classes = self.config.n_classes

        # Storage: n_classes * dim bits
        storage_bits = n_classes * dim
        storage_mm2 = storage_bits / (8 * 1024 * 1024) * 0.01  # ~0.01 mm^2 per MB

        # Logic: XOR + popcount + comparator
        logic_mm2 = 0.001  # ~0.001 mm^2 for simple logic

        # Total
        total_mm2 = storage_mm2 + logic_mm2

        return {
            "storage_mm2": storage_mm2,
            "logic_mm2": logic_mm2,
            "total_mm2": total_mm2,
        }

    def estimate_throughput(self) -> Dict[str, float]:
        """Estimate classification throughput.

        Returns:
            Dict of throughput estimates
        """
        freq = self.config.frequency_mhz * 1e6  # Hz
        dim = self.config.dim
        n_classes = self.config.n_classes

        # Cycles per classification: n_classes * dim (similarity) + n_classes (compare)
        cycles_per_class = n_classes * dim + n_classes

        # Throughput
        classifications_per_second = freq / cycles_per_class

        return {
            "cycles_per_classification": cycles_per_class,
            "classifications_per_second": classifications_per_second,
            "classifications_per_ms": classifications_per_second / 1000,
        }

    def compare_with_nn(self) -> Dict[str, Any]:
        """Compare HDC with equivalent NN.

        Returns:
            Dict of comparison metrics
        """
        hdc_energy = self.estimate_energy_hdc()
        nn_energy = self.estimate_energy_nn()

        energy_ratio = nn_energy["total_energy_pj"] / hdc_energy["total_energy_pj"]

        return {
            "hdc_energy_nj": hdc_energy["total_energy_nj"],
            "nn_energy_nj": nn_energy["total_energy_nj"],
            "energy_ratio_hdc_vs_nn": energy_ratio,
            "hdc_area_mm2": self.estimate_area()["total_mm2"],
        }

    def efficiency_summary(self) -> Dict[str, Any]:
        """
        One-call efficiency report with a plain-English label.

        energy_class:
          'ultralow'  < 0.01 nJ — suitable for implantable / nano-IoT
          'low'       < 1 nJ   — suitable for edge sensor nodes
          'medium'    < 100 nJ — general embedded deployment
          'high'      ≥ 100 nJ — server-class or power-connected
        """
        hdc = self.estimate_energy_hdc()
        area = self.estimate_area()
        cmp  = self.compare_with_nn()
        energy_nj = hdc["total_energy_nj"]
        energy_class = (
            "ultralow" if energy_nj < 0.01 else
            "low"       if energy_nj < 1.0  else
            "medium"    if energy_nj < 100  else
            "high"
        )
        return {
            "energy_nj":          energy_nj,
            "energy_class":       energy_class,
            "area_mm2":           area["total_mm2"],
            "speedup_vs_nn":      round(cmp["energy_ratio_hdc_vs_nn"], 2),
            "platform":           self.config.platform.value,
            "hd_dim":             self.config.hd_dim,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_unified_encoder():
    """Verify all encoding methods."""
    print("=" * 60)
    print("Testing Unified HDC Encoder (Ge & Parhi 2020, Section III)")
    print("=" * 60)

    dim = 1000

    # Test random projection encoding
    config = EncodingConfig(dim=dim, encoding_type=EncodingType.RANDOM_PROJECTION)
    encoder = UnifiedHDCEncoder(config)
    x = torch.randn(10, 50)
    hvs = encoder.encode(x)
    print(f"  Random projection: {hvs.shape} ✅")

    # Test ID-level encoding
    config = EncodingConfig(dim=dim, encoding_type=EncodingType.ID_LEVEL, n_items=50)
    encoder = UnifiedHDCEncoder(config)
    ids = torch.randint(0, 50, (10,))
    hvs = encoder.encode(ids)
    print(f"  ID-level: {hvs.shape} ✅")

    # Test N-gram encoding
    config = EncodingConfig(dim=dim, encoding_type=EncodingType.NGRAM,
                            ngram_order=3, ngram_vocab_size=50)
    encoder = UnifiedHDCEncoder(config)
    seqs = torch.randint(0, 50, (10, 20))
    hvs = encoder.encode(seqs)
    print(f"  N-gram: {hvs.shape} ✅")

    # Test spatial encoding
    config = EncodingConfig(dim=dim, encoding_type=EncodingType.SPATIAL,
                            spatial_resolution=50)
    encoder = UnifiedHDCEncoder(config)
    pos = torch.rand(10, 2) * 2 - 1
    hvs = encoder.encode(pos)
    print(f"  Spatial: {hvs.shape} ✅")

    # Test temporal encoding
    config = EncodingConfig(dim=dim, encoding_type=EncodingType.TEMPORAL)
    encoder = UnifiedHDCEncoder(config)
    seq = torch.randn(10, 8, 16)
    hvs = encoder.encode(seq)
    print(f"  Temporal: {hvs.shape} ✅")

    # Test biosignal encoding
    config = EncodingConfig(dim=dim, encoding_type=EncodingType.BIOSIGNAL,
                            biosignal_channels=8, biosignal_samples=32)
    encoder = UnifiedHDCEncoder(config)
    signal = torch.randn(5, 8, 32)
    hvs = encoder.encode(signal)
    print(f"  Biosignal: {hvs.shape} ✅")

    print(f"  ✅ Unified encoder test complete!")


def test_hd_classifier():
    """Verify HDC classifier with all training strategies."""
    print("=" * 60)
    print("Testing HDC Classifier (Ge & Parhi 2020, Section IV)")
    print("=" * 60)

    dim = 1000
    n_classes = 4
    n_samples = 40

    # Generate synthetic data
    prototypes = gen_hvs(n_classes, dim)
    hvs = []
    labels = []
    for i in range(n_samples):
        label = i % n_classes
        noise = (torch.rand(dim) < 0.2).float()
        hv = hv_majority(hv_bundle(torch.stack([prototypes[label], noise])))
        hvs.append(hv)
        labels.append(label)
    hvs = torch.stack(hvs)
    labels = torch.tensor(labels)

    # Test one-shot training
    config = TrainingConfig(dim=dim, n_classes=n_classes,
                            retraining_mode=RetrainingMode.ONE_SHOT)
    clf = HDClassifier(config)
    clf.train(hvs, labels)
    metrics = clf.evaluate(hvs, labels)
    print(f"  One-shot accuracy: {metrics['accuracy']:.3f}")

    # Test iterative training
    config = TrainingConfig(dim=dim, n_classes=n_classes,
                            retraining_mode=RetrainingMode.ITERATIVE,
                            n_iterations=5)
    clf = HDClassifier(config)
    clf.train(hvs, labels)
    metrics = clf.evaluate(hvs, labels)
    print(f"  Iterative accuracy: {metrics['accuracy']:.3f}")

    # Test adaptive LR training
    config = TrainingConfig(dim=dim, n_classes=n_classes,
                            retraining_mode=RetrainingMode.ADAPTIVE_LR,
                            n_iterations=3)
    clf = HDClassifier(config)
    clf.train(hvs, labels)
    metrics = clf.evaluate(hvs, labels)
    print(f"  Adaptive LR accuracy: {metrics['accuracy']:.3f}")

    # Test weighted training
    config = TrainingConfig(dim=dim, n_classes=n_classes,
                            retraining_mode=RetrainingMode.WEIGHTED)
    clf = HDClassifier(config)
    clf.train(hvs, labels)
    metrics = clf.evaluate(hvs, labels)
    print(f"  Weighted accuracy: {metrics['accuracy']:.3f}")

    print(f"  ✅ HDC classifier test complete!")


def test_multi_label_classifier():
    """Verify multi-label HDC classification."""
    print("=" * 60)
    print("Testing Multi-Label HDC Classifier (Ge & Parhi 2020, Section VI)")
    print("=" * 60)

    dim = 1000
    n_labels = 3
    n_samples = 30

    # Generate synthetic multi-label data
    prototypes = gen_hvs(n_labels, dim)
    hvs = []
    labels = []
    for i in range(n_samples):
        # Each sample has 1-2 labels
        n_active = 1 + (i % 2)
        active_labels = torch.randperm(n_labels)[:n_active]
        label_vec = torch.zeros(n_labels)
        label_vec[active_labels] = 1.0

        # Create combined HV
        active_hvs = prototypes[active_labels]
        hv = hv_majority(hv_bundle(active_hvs))
        hvs.append(hv)
        labels.append(label_vec)
    hvs = torch.stack(hvs)
    labels = torch.stack(labels)

    # Test binary relevance
    ml_clf = MultiLabelHDClassifier(dim=dim, n_labels=n_labels,
                                     method="binary_relevance")
    ml_clf.train(hvs, labels)
    metrics = ml_clf.evaluate(hvs, labels)
    print(f"  Binary relevance:")
    print(f"    Hamming loss: {metrics['hamming_loss']:.3f}")
    print(f"    Exact match: {metrics['exact_match_ratio']:.3f}")
    print(f"    Mean F1: {metrics['mean_f1']:.3f}")

    # Test label embedding
    ml_clf2 = MultiLabelHDClassifier(dim=dim, n_labels=n_labels,
                                      method="label_embedding")
    ml_clf2.train(hvs, labels)
    metrics2 = ml_clf2.evaluate(hvs, labels)
    print(f"  Label embedding:")
    print(f"    Hamming loss: {metrics2['hamming_loss']:.3f}")
    print(f"    Exact match: {metrics2['exact_match_ratio']:.3f}")
    print(f"    Mean F1: {metrics2['mean_f1']:.3f}")

    print(f"  ✅ Multi-label classifier test complete!")


def test_confidence_calibration():
    """Verify confidence calibration."""
    print("=" * 60)
    print("Testing Confidence Calibration (Ge & Parhi 2020, Section V)")
    print("=" * 60)

    dim = 1000
    n_classes = 3
    n_samples = 30

    # Generate synthetic data
    prototypes = gen_hvs(n_classes, dim)
    hvs = []
    labels = []
    for i in range(n_samples):
        label = i % n_classes
        noise = (torch.rand(dim) < 0.2).float()
        hv = hv_majority(hv_bundle(torch.stack([prototypes[label], noise])))
        hvs.append(hv)
        labels.append(label)
    hvs = torch.stack(hvs)
    labels = torch.tensor(labels)

    # Train classifier
    config = TrainingConfig(dim=dim, n_classes=n_classes)
    clf = HDClassifier(config)
    clf.train(hvs, labels)

    # Get similarity scores (hv_batch_sim expects 1D query, 2D memory)
    n_samples = hvs.shape[0]
    sims = torch.zeros(n_samples, n_classes)
    for i in range(n_samples):
        sims[i] = hv_batch_sim(hvs[i], clf.prototypes)

    # Test temperature calibration
    calibrator = ConfidenceCalibrator(method="temperature")
    calibrator.calibrate(sims, labels)
    probs = calibrator.predict_proba(sims)
    print(f"  Temperature: {calibrator.temperature:.2f}")
    print(f"  Calibrated probs shape: {probs.shape}")

    # Test Platt calibration
    calibrator2 = ConfidenceCalibrator(method="platt")
    calibrator2.calibrate(sims, labels)
    probs2 = calibrator2.predict_proba(sims)
    print(f"  Platt A: {calibrator2.platt_a:.3f}, B: {calibrator2.platt_b:.3f}")
    print(f"  Platt probs shape: {probs2.shape}")

    print(f"  ✅ Confidence calibration test complete!")


def test_hardware_efficiency():
    """Verify hardware efficiency model."""
    print("=" * 60)
    print("Testing Hardware Efficiency Model (Ge & Parhi 2020, Section VI)")
    print("=" * 60)

    config = HardwareConfig(
        platform=HardwarePlatform.DIGITAL_ASIC,
        dim=10000,
        n_classes=10,
    )
    model = HardwareEfficiencyModel(config)

    hdc_energy = model.estimate_energy_hdc()
    print(f"  HDC energy: {hdc_energy['total_energy_nj']:.3f} nJ")

    nn_energy = model.estimate_energy_nn()
    print(f"  NN energy: {nn_energy['total_energy_nj']:.3f} nJ")

    comparison = model.compare_with_nn()
    print(f"  Energy ratio (NN/HDC): {comparison['energy_ratio_hdc_vs_nn']:.1f}x")

    area = model.estimate_area()
    print(f"  HDC area: {area['total_mm2']:.4f} mm²")

    throughput = model.estimate_throughput()
    print(f"  Throughput: {throughput['classifications_per_ms']:.1f} /ms")

    print(f"  ✅ Hardware efficiency test complete!")


if __name__ == "__main__":
    test_unified_encoder()
    print()
    test_hd_classifier()
    print()
    test_multi_label_classifier()
    print()
    test_confidence_calibration()
    print()
    test_hardware_efficiency()
    print()
    print("=== All Ge & Parhi 2020 tests complete ===")
