"""
NeuroBench: Benchmarking Neuromorphic Computing Algorithms and Systems
=======================================================================
Based on: Yik, J., Van den Berghe, K., den Blanken, D., et al. (2025)
"The NeuroBench Framework for Benchmarking Neuromorphic Computing
Algorithms and Systems"
Nature Communications, 16, Article 1545.
DOI: 10.1038/s41467-025-56739-4

Implements the dual-track evaluation architecture:

**Algorithm Track** (hardware-independent):
  Evaluates correctness and computational complexity without tying
  results to a specific hardware platform. Metrics:
    - accuracy           : task-specific correctness score
    - activation_sparsity: fraction of zero activations (crucial for SNN)
    - synaptic_operations: effective multiply-accumulates (MACs/SOPs)
    - memory_footprint   : bytes required for model parameters
    - parameter_count    : number of trainable parameters

**System Track** (hardware-dependent):
  Evaluates fully-deployed systems with standardised measurements:
    - total_energy_j     : joules per inference
    - latency_s          : seconds per inference
    - throughput_sps     : samples per second
    - power_w            : average power draw

Both tracks require *paired* correctness + efficiency metrics so that
raw efficiency claims cannot be made without reporting task performance.

Usage:
    evaluator = NeuroBenchEvaluator(model, task="hdc_classification")
    algo_metrics = evaluator.algorithm_track(X_test, y_test)
    # For hardware-deployed systems:
    sys_metrics = evaluator.system_track(energy_j=..., latency_s=..., ...)
    report = evaluator.report(algo_metrics, sys_metrics)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn


# ═══════════════════════════════════════════════════════════════════════════════
# Algorithm Track Metrics
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class AlgorithmTrackMetrics:
    """
    NeuroBench Algorithm Track output (NeuroBench 2025, Section 3.1).

    All metrics are hardware-independent and must always be reported
    paired with a correctness metric (accuracy / error rate).
    """
    # Correctness
    accuracy: float = 0.0
    n_correct: int = 0
    n_total: int = 0

    # Computational complexity
    synaptic_operations: int = 0        # MAC-equivalent ops per inference
    activation_sparsity: float = 0.0    # fraction of zero activations ∈ [0,1]

    # Model footprint
    parameter_count: int = 0            # trainable parameters
    memory_footprint_bytes: int = 0     # bytes for all parameters

    # Task metadata
    task: str = ""
    model_name: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "model_name": self.model_name,
            "accuracy": self.accuracy,
            "n_correct": self.n_correct,
            "n_total": self.n_total,
            "synaptic_operations": self.synaptic_operations,
            "activation_sparsity": self.activation_sparsity,
            "parameter_count": self.parameter_count,
            "memory_footprint_bytes": self.memory_footprint_bytes,
        }


@dataclass
class SystemTrackMetrics:
    """
    NeuroBench System Track output (NeuroBench 2025, Section 3.2).

    Requires actual hardware measurements. Must always be accompanied
    by Algorithm Track correctness to prevent unconstrained efficiency claims.
    """
    # Paired correctness (required)
    accuracy: float = 0.0

    # Hardware measurements
    total_energy_j: float = 0.0         # joules per inference
    latency_s: float = 0.0              # seconds per inference
    throughput_sps: float = 0.0         # samples per second
    power_w: float = 0.0                # average power (W)

    # Platform description
    platform: str = ""
    chip: str = ""
    frequency_hz: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "total_energy_j": self.total_energy_j,
            "latency_s": self.latency_s,
            "throughput_sps": self.throughput_sps,
            "power_w": self.power_w,
            "energy_per_sample_uj": self.total_energy_j * 1e6,
            "platform": self.platform,
            "chip": self.chip,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Synaptic Operations Counter
# ═══════════════════════════════════════════════════════════════════════════════

class SynapticOpCounter:
    """
    Counts synaptic operations (SOPs) for HDC/SNN models.

    NeuroBench distinguishes:
      - Effective MACs (for binary/ternary networks where zeros are skipped)
      - Synaptic Operations (SOPs) for spike-based models: each spike
        contributes one SOP per fan-out synapse

    For HDC classifiers:
      - Encoding: F × D XOR + D majority ops → F × D + D SOPs
      - Classification: C × D Hamming distance ops → C × D SOPs
      - Activation sparsity: fraction of zero bits in intermediate HVs
    """

    def __init__(self):
        self._ops: List[int] = []
        self._activations: List[torch.Tensor] = []
        self._hooks: List[Any] = []

    def count_hdc_encoding(self, n_features: int, dim: int) -> int:
        """SOPs for record encoding: F XOR + 1 majority per dimension."""
        return n_features * dim + dim

    def count_hdc_classification(self, n_classes: int, dim: int) -> int:
        """SOPs for Hamming-distance similarity search over C prototypes."""
        return n_classes * dim

    def count_hdc_full(self, n_features: int, dim: int, n_classes: int) -> int:
        """Total SOPs for one HDC inference."""
        return self.count_hdc_encoding(n_features, dim) + \
               self.count_hdc_classification(n_classes, dim)

    def count_nn_layer(self, in_features: int, out_features: int) -> int:
        """MACs for a linear layer."""
        return in_features * out_features

    def activation_sparsity_from_hvs(self, hvs: torch.Tensor) -> float:
        """
        Compute activation sparsity from binary hypervectors.

        For binary HDC: sparsity = fraction of 0 bits.
        For SNN: sparsity = fraction of non-spiking neurons over time.

        Args:
            hvs: (..., D) binary tensor
        Returns:
            Sparsity ∈ [0, 1]  (0 = dense, 1 = all zeros)
        """
        return float((hvs == 0).float().mean().item())

    def hook_nn_module(self, module: nn.Module):
        """
        Attach forward hooks to count MACs and track activations.

        Call remove_hooks() after evaluation.
        """
        def _make_hook(m):
            def hook(mod, inp, out):
                if isinstance(mod, nn.Linear):
                    ops = mod.in_features * mod.out_features
                    if inp[0].dim() > 1:
                        ops *= inp[0].shape[0]
                    self._ops.append(ops)
                if isinstance(out, torch.Tensor):
                    self._activations.append(out.detach().cpu())
            return hook

        for m in module.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                self._hooks.append(m.register_forward_hook(_make_hook(m)))

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def total_ops(self) -> int:
        return sum(self._ops)

    def total_sparsity(self) -> float:
        if not self._activations:
            return 0.0
        all_acts = torch.cat([a.flatten() for a in self._activations])
        return float((all_acts == 0).float().mean().item())

    def reset(self):
        self._ops.clear()
        self._activations.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# Model Footprint Inspector
# ═══════════════════════════════════════════════════════════════════════════════

class ModelFootprint:
    """Compute parameter count and memory footprint for any PyTorch module."""

    @staticmethod
    def parameter_count(model: nn.Module, trainable_only: bool = True) -> int:
        """Count parameters in a PyTorch model."""
        if trainable_only:
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        return sum(p.numel() for p in model.parameters())

    @staticmethod
    def memory_footprint_bytes(model: nn.Module) -> int:
        """Memory in bytes for all parameters (including buffers)."""
        param_bytes = sum(
            p.numel() * p.element_size() for p in model.parameters()
        )
        buffer_bytes = sum(
            b.numel() * b.element_size() for b in model.buffers()
        )
        return param_bytes + buffer_bytes

    @staticmethod
    def hdc_prototype_footprint(n_classes: int, dim: int, bits_per_element: int = 1) -> int:
        """
        Memory footprint for HDC class prototypes.

        Binary prototypes: n_classes × dim bits, packed into bytes.
        Float prototypes: n_classes × dim × 4 bytes.
        """
        if bits_per_element == 1:
            return math.ceil(n_classes * dim / 8)
        return n_classes * dim * (bits_per_element // 8)

    @staticmethod
    def hdc_item_memory_footprint(n_items: int, dim: int, bits_per_element: int = 1) -> int:
        """Memory footprint for ID-HV / LHV item memory."""
        if bits_per_element == 1:
            return math.ceil(n_items * dim / 8)
        return n_items * dim * (bits_per_element // 8)


# ═══════════════════════════════════════════════════════════════════════════════
# NeuroBench Evaluator
# ═══════════════════════════════════════════════════════════════════════════════

class NeuroBenchEvaluator:
    """
    Unified NeuroBench evaluation interface (NeuroBench 2025).

    Supports:
    - PyTorch nn.Module models (via hooks)
    - HDC classifiers with explicit operation counting
    - Any callable model with a predict() method

    Usage:
        evaluator = NeuroBenchEvaluator(
            model=clf,
            model_name="NanoscaleHDC",
            task="hdc_classification",
            hdc_config={"n_features": 20, "dim": 10000, "n_classes": 4},
        )
        algo = evaluator.algorithm_track(X_test, y_test)
        report = evaluator.report(algo)
    """

    def __init__(
        self,
        model: Any,
        model_name: str = "model",
        task: str = "classification",
        hdc_config: Optional[Dict] = None,
    ):
        """
        Args:
            model: Callable model (nn.Module, HDC classifier, or any predict())
            model_name: Descriptive name for reporting
            task: Task name string
            hdc_config: For HDC models — dict with keys:
                        n_features, dim, n_classes (enables SOP counting)
        """
        self.model = model
        self.model_name = model_name
        self.task = task
        self.hdc_config = hdc_config or {}
        self.op_counter = SynapticOpCounter()
        self.footprint = ModelFootprint()

    def _predict(self, X: torch.Tensor) -> torch.Tensor:
        """Run model inference and return predicted labels."""
        if isinstance(self.model, nn.Module):
            self.model.eval()
            with torch.no_grad():
                out = self.model(X)
                if out.dim() > 1:
                    return out.argmax(dim=-1)
                return out
        elif hasattr(self.model, "predict"):
            return self.model.predict(X)
        elif hasattr(self.model, "forward"):
            return self.model.forward(X)
        else:
            return self.model(X)

    def algorithm_track(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        intermediate_hvs: Optional[torch.Tensor] = None,
    ) -> AlgorithmTrackMetrics:
        """
        Run Algorithm Track evaluation.

        Args:
            X: (N, ...) input samples
            y: (N,) integer ground-truth labels
            intermediate_hvs: Optional (N, D) encoded hypervectors for
                              sparsity measurement (HDC-specific)

        Returns:
            AlgorithmTrackMetrics
        """
        N = X.shape[0]

        # Attach hooks if nn.Module
        is_nn = isinstance(self.model, nn.Module)
        if is_nn:
            self.op_counter.reset()
            self.op_counter.hook_nn_module(self.model)

        # Run inference
        preds = self._predict(X)

        if is_nn:
            self.op_counter.remove_hooks()

        # Correctness
        if isinstance(preds, torch.Tensor):
            correct = int((preds == y).sum().item())
        else:
            correct = int(sum(p == t for p, t in zip(preds, y.tolist())))
        accuracy = correct / N

        # Synaptic operations
        # For HDC nn.Module models that store weights as buffers (not Parameters),
        # hook-based counting yields 0; fall back to explicit hdc_config accounting.
        nn_sops = self.op_counter.total_ops() if is_nn else 0
        if nn_sops > 0:
            sops = nn_sops
            sparsity = self.op_counter.total_sparsity()
        elif self.hdc_config:
            cfg = self.hdc_config
            sops = self.op_counter.count_hdc_full(
                cfg.get("n_features", 1),
                cfg.get("dim", 10000),
                cfg.get("n_classes", 2),
            )
            if intermediate_hvs is not None:
                sparsity = self.op_counter.activation_sparsity_from_hvs(intermediate_hvs)
            else:
                sparsity = 0.0
        else:
            sops = 0
            sparsity = 0.0

        # Memory footprint
        # memory_footprint_bytes() counts both parameters AND buffers, so it
        # correctly handles HDC nn.Module models that use register_buffer.
        # parameter_count() only counts nn.Parameter — for buffer-only HDC models
        # use sum of buffer elements as the effective "parameter count".
        if is_nn:
            mem = self.footprint.memory_footprint_bytes(self.model)
            params = self.footprint.parameter_count(self.model, trainable_only=False)
            if params == 0:
                # Buffer-only HDC model: count buffer elements as logical parameters
                params = sum(b.numel() for b in self.model.buffers())
        elif self.hdc_config:
            cfg = self.hdc_config
            dim = cfg.get("dim", 10000)
            n_classes = cfg.get("n_classes", 2)
            n_feat = cfg.get("n_features", 1)
            n_levels = cfg.get("n_levels", 100)
            proto_mem = self.footprint.hdc_prototype_footprint(n_classes, dim)
            item_mem = self.footprint.hdc_item_memory_footprint(n_feat + n_levels, dim)
            mem = proto_mem + item_mem
            params = n_classes * dim + (n_feat + n_levels) * dim
        else:
            params = 0
            mem = 0

        return AlgorithmTrackMetrics(
            accuracy=accuracy,
            n_correct=correct,
            n_total=N,
            synaptic_operations=sops,
            activation_sparsity=sparsity,
            parameter_count=params,
            memory_footprint_bytes=mem,
            task=self.task,
            model_name=self.model_name,
        )

    def system_track(
        self,
        accuracy: float,
        total_energy_j: float,
        latency_s: float,
        power_w: float = 0.0,
        platform: str = "",
        chip: str = "",
        frequency_hz: float = 0.0,
    ) -> SystemTrackMetrics:
        """
        Record System Track hardware measurements.

        Args:
            accuracy: Correctness metric (MUST match algorithm track)
            total_energy_j: Energy per inference in joules
            latency_s: Latency per inference in seconds
            power_w: Average power draw in watts
            platform: Hardware platform description
            chip: Chip/board name
            frequency_hz: Clock frequency

        Returns:
            SystemTrackMetrics
        """
        throughput = 1.0 / max(latency_s, 1e-12)
        if power_w == 0.0 and latency_s > 0:
            power_w = total_energy_j / latency_s

        return SystemTrackMetrics(
            accuracy=accuracy,
            total_energy_j=total_energy_j,
            latency_s=latency_s,
            throughput_sps=throughput,
            power_w=power_w,
            platform=platform,
            chip=chip,
            frequency_hz=frequency_hz,
        )

    def measure_latency(
        self,
        X: torch.Tensor,
        n_warmup: int = 5,
        n_repeat: int = 50,
    ) -> Tuple[float, float]:
        """
        CPU/GPU wall-clock latency measurement.

        Args:
            X: Single sample or batch for timing
            n_warmup: Warm-up iterations (not timed)
            n_repeat: Timed iterations

        Returns:
            (mean_latency_s, std_latency_s)
        """
        for _ in range(n_warmup):
            self._predict(X)

        times = []
        for _ in range(n_repeat):
            t0 = time.perf_counter()
            self._predict(X)
            t1 = time.perf_counter()
            times.append(t1 - t0)

        times_t = torch.tensor(times)
        return float(times_t.mean().item()), float(times_t.std().item())

    def report(
        self,
        algo: AlgorithmTrackMetrics,
        sys: Optional[SystemTrackMetrics] = None,
    ) -> Dict[str, Any]:
        """
        Build NeuroBench-style structured report.

        Mirrors the NeuroBench output format from the paper's supplementary
        materials, with paired correctness + efficiency metrics.
        """
        r: Dict[str, Any] = {
            "neurobench_version": "2025",
            "algorithm_track": algo.to_dict(),
        }

        if sys is not None:
            # Verify correctness consistency (NeuroBench requirement)
            delta = abs(algo.accuracy - sys.accuracy)
            if delta > 0.01:
                import warnings
                warnings.warn(
                    f"Algorithm and system track accuracies differ by {delta:.3f}. "
                    "NeuroBench requires paired metrics from the same evaluation."
                )
            r["system_track"] = sys.to_dict()
            r["system_track"]["accuracy_delta_algo_vs_sys"] = delta

        # Composite efficiency score: accuracy / log10(ops + 1)
        if algo.synaptic_operations > 0:
            efficiency_score = algo.accuracy / math.log10(algo.synaptic_operations + 1)
        else:
            efficiency_score = 0.0
        r["composite_efficiency_score"] = round(efficiency_score, 6)

        return r


# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark Suite
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BenchmarkTask:
    """Definition of a NeuroBench benchmark task."""
    name: str
    description: str
    metric: str = "accuracy"
    n_samples: int = 1000
    generate_data: Optional[Callable] = None


class BenchmarkSuite:
    """
    Collection of standardised NeuroBench tasks for HDC models.

    Provides reproducible synthetic benchmarks analogous to the real-world
    tasks in NeuroBench (gesture recognition, keyword spotting, etc.) but
    executable without external datasets.

    Tasks:
      - hdc_classification: N-class HD classification with Gaussian clusters
      - hdc_temporal: Temporal sequence classification (n-gram encoding)
      - hdc_spatial: 2-D spatial pattern recognition
      - hdc_multiclass: Increasing number of classes stress test
    """

    TASKS = [
        BenchmarkTask(
            name="hdc_classification",
            description="N-class binary HDC classification with Gaussian clusters",
            metric="accuracy",
        ),
        BenchmarkTask(
            name="hdc_temporal",
            description="Sequence classification using n-gram encoding",
            metric="accuracy",
        ),
        BenchmarkTask(
            name="hdc_spatial",
            description="2-D spatial pattern recognition",
            metric="accuracy",
        ),
        BenchmarkTask(
            name="hdc_scalability",
            description="Accuracy vs. dimensionality sweep",
            metric="accuracy_at_dims",
        ),
    ]

    @staticmethod
    def generate_classification_data(
        n_classes: int = 4,
        n_features: int = 20,
        n_train: int = 200,
        n_test: int = 100,
        noise: float = 0.2,
        seed: int = 42,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Synthetic Gaussian cluster data for classification benchmark.

        Returns:
            X_train, y_train, X_test, y_test — all normalised to [0, 1]
        """
        torch.manual_seed(seed)
        protos = torch.rand(n_classes, n_features)

        def make_split(n):
            X, y = [], []
            per_class = n // n_classes
            for c in range(n_classes):
                samples = protos[c].unsqueeze(0).expand(per_class, -1)
                samples = (samples + torch.randn(per_class, n_features) * noise).clamp(0, 1)
                X.append(samples)
                y.extend([c] * per_class)
            return torch.cat(X), torch.tensor(y)

        X_tr, y_tr = make_split(n_train)
        X_te, y_te = make_split(n_test)
        return X_tr, y_tr, X_te, y_te

    @staticmethod
    def generate_temporal_data(
        n_classes: int = 4,
        seq_len: int = 20,
        vocab_size: int = 30,
        n_train: int = 200,
        n_test: int = 100,
        seed: int = 42,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Synthetic token sequences for n-gram temporal benchmark.

        Each class has a characteristic n-gram pattern embedded in noise.
        Returns integer token sequences: (N, seq_len).
        """
        torch.manual_seed(seed)
        class_tokens = [
            torch.randint(0, vocab_size // n_classes, (3,)) + c * (vocab_size // n_classes)
            for c in range(n_classes)
        ]

        def make_split(n):
            X, y = [], []
            per_class = n // n_classes
            for c in range(n_classes):
                for _ in range(per_class):
                    seq = torch.randint(0, vocab_size, (seq_len,))
                    insert_pos = torch.randint(0, seq_len - 3, (1,)).item()
                    seq[insert_pos:insert_pos + 3] = class_tokens[c]
                    X.append(seq)
                    y.append(c)
            return torch.stack(X), torch.tensor(y)

        return (*make_split(n_train), *make_split(n_test))

    @classmethod
    def run_all(
        cls,
        evaluator: NeuroBenchEvaluator,
        X_test: torch.Tensor,
        y_test: torch.Tensor,
    ) -> Dict[str, AlgorithmTrackMetrics]:
        """Run algorithm track evaluation across all applicable tasks."""
        return {
            "hdc_classification": evaluator.algorithm_track(X_test, y_test)
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Baseline Comparisons
# ═══════════════════════════════════════════════════════════════════════════════

class BaselineComparator:
    """
    Compare HDC models against NN baselines (NeuroBench dual-track).

    NeuroBench requires neuromorphic results to be compared against
    conventional DNN baselines with equivalent correctness budgets.
    """

    @staticmethod
    def nn_ops_for_accuracy(
        target_accuracy: float,
        n_features: int,
        n_classes: int,
        hidden_sizes: Optional[List[int]] = None,
    ) -> int:
        """
        Estimate MACs for a fully-connected NN achieving target accuracy.

        Uses a simple scaling heuristic: higher accuracy requires larger
        networks (more MACs).

        Args:
            target_accuracy: Desired accuracy ∈ [0, 1]
            n_features: Input feature count
            n_classes: Output class count
            hidden_sizes: Layer sizes (auto-estimated if None)

        Returns:
            Estimated MACs per inference
        """
        if hidden_sizes is None:
            # Rough heuristic: accuracy ~0.9 → 2 layers × 128
            scale = max(1, int(target_accuracy * 10))
            hidden = 32 * scale
            hidden_sizes = [hidden, hidden]

        macs = n_features * hidden_sizes[0]
        for i in range(len(hidden_sizes) - 1):
            macs += hidden_sizes[i] * hidden_sizes[i + 1]
        macs += hidden_sizes[-1] * n_classes
        return macs

    @staticmethod
    def efficiency_ratio(
        hdc_metrics: AlgorithmTrackMetrics,
        nn_macs: int,
    ) -> Dict[str, float]:
        """
        Compute HDC efficiency ratio vs NN baseline.

        Returns ops reduction and accuracy parity.
        """
        ops_ratio = nn_macs / max(hdc_metrics.synaptic_operations, 1)
        return {
            "hdc_accuracy": hdc_metrics.accuracy,
            "hdc_sops": hdc_metrics.synaptic_operations,
            "nn_macs_estimated": nn_macs,
            "ops_reduction_ratio": ops_ratio,
            "hdc_memory_bytes": hdc_metrics.memory_footprint_bytes,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_op_counter():
    print("=" * 60)
    print("Testing SynapticOpCounter (NeuroBench 2025, Section 3.1)")
    print("=" * 60)

    counter = SynapticOpCounter()

    enc_ops = counter.count_hdc_encoding(n_features=20, dim=10000)
    cls_ops = counter.count_hdc_classification(n_classes=4, dim=10000)
    total = counter.count_hdc_full(20, 10000, 4)

    print(f"  Encoding ops : {enc_ops:,}")
    print(f"  Classify ops : {cls_ops:,}")
    print(f"  Total ops    : {total:,}")
    assert total == enc_ops + cls_ops

    hvs = (torch.rand(100, 10000) < 0.5).float()
    sparsity = counter.activation_sparsity_from_hvs(hvs)
    print(f"  Sparsity (d=0.5 HVs): {sparsity:.4f}  (want ≈ 0.5)")
    assert 0.45 < sparsity < 0.55

    print("  ✅ SynapticOpCounter OK")


def test_model_footprint():
    print("=" * 60)
    print("Testing ModelFootprint (NeuroBench 2025)")
    print("=" * 60)

    proto_bytes = ModelFootprint.hdc_prototype_footprint(n_classes=10, dim=10000)
    item_bytes = ModelFootprint.hdc_item_memory_footprint(n_items=200, dim=10000)
    print(f"  Prototype memory (10 classes, D=10000) : {proto_bytes:,} bytes = {proto_bytes/1024:.1f} KB")
    print(f"  Item memory (200 items, D=10000)       : {item_bytes:,} bytes = {item_bytes/1024:.1f} KB")
    assert proto_bytes == math.ceil(10 * 10000 / 8)
    print("  ✅ ModelFootprint OK")


def test_evaluator_hdc():
    print("=" * 60)
    print("Testing NeuroBenchEvaluator with HDC model (NeuroBench 2025)")
    print("=" * 60)

    from hdc.rahimi_nanoscale import NanoscaleRecordEncoder, NanoscaleHDCClassifier

    torch.manual_seed(0)
    n_classes, n_feat, dim = 3, 12, 3000
    enc = NanoscaleRecordEncoder(n_features=n_feat, n_levels=32, dim=dim, seed=2)
    clf = NanoscaleHDCClassifier(enc, n_classes=n_classes, n_retrain=2)

    X_tr, y_tr, X_te, y_te = BenchmarkSuite.generate_classification_data(
        n_classes=n_classes, n_features=n_feat, n_train=120, n_test=60, seed=7
    )

    clf.train_one_shot(X_tr, y_tr)
    clf.retrain(X_tr, y_tr)

    evaluator = NeuroBenchEvaluator(
        model=clf,
        model_name="NanoscaleHDC",
        task="hdc_classification",
        hdc_config={"n_features": n_feat, "dim": dim, "n_classes": n_classes, "n_levels": 32},
    )

    algo = evaluator.algorithm_track(X_te, y_te)
    print(f"  Accuracy          : {algo.accuracy:.1%}")
    print(f"  Synaptic ops      : {algo.synaptic_operations:,}")
    print(f"  Memory footprint  : {algo.memory_footprint_bytes:,} bytes")
    print(f"  Parameter count   : {algo.parameter_count:,}")

    # Synthetic system track
    sys = evaluator.system_track(
        accuracy=algo.accuracy,
        total_energy_j=1e-6,
        latency_s=1e-4,
        platform="SNNTraining Simulator",
    )

    report = evaluator.report(algo, sys)
    print(f"  Efficiency score  : {report['composite_efficiency_score']:.6f}")
    assert "algorithm_track" in report
    assert "system_track" in report
    print("  ✅ Evaluator OK")


def test_benchmark_suite():
    print("=" * 60)
    print("Testing BenchmarkSuite (NeuroBench 2025)")
    print("=" * 60)

    X_tr, y_tr, X_te, y_te = BenchmarkSuite.generate_classification_data(
        n_classes=4, n_features=16, n_train=160, n_test=80, seed=99
    )
    print(f"  Classification data: train {X_tr.shape}, test {X_te.shape}")

    seqs_tr, sy_tr, seqs_te, sy_te = BenchmarkSuite.generate_temporal_data(
        n_classes=3, seq_len=15, vocab_size=20, n_train=90, n_test=45, seed=11
    )
    print(f"  Temporal data: train {seqs_tr.shape}, test {seqs_te.shape}")

    # Verify label balance
    for c in range(4):
        assert (y_tr == c).sum() > 0, f"Class {c} missing from training"
    print("  ✅ BenchmarkSuite OK")


def test_baseline_comparator():
    print("=" * 60)
    print("Testing BaselineComparator (NeuroBench 2025)")
    print("=" * 60)

    nn_macs = BaselineComparator.nn_ops_for_accuracy(0.9, n_features=20, n_classes=4)
    print(f"  NN MACs for 90% accuracy: {nn_macs:,}")

    dummy_algo = AlgorithmTrackMetrics(
        accuracy=0.88,
        synaptic_operations=204_000,
        memory_footprint_bytes=12_500,
    )
    ratio = BaselineComparator.efficiency_ratio(dummy_algo, nn_macs)
    print(f"  Ops reduction vs NN : {ratio['ops_reduction_ratio']:.1f}×")
    print(f"  HDC memory          : {ratio['hdc_memory_bytes']:,} bytes")
    print("  ✅ BaselineComparator OK")


if __name__ == "__main__":
    test_op_counter()
    print()
    test_model_footprint()
    print()
    test_evaluator_hdc()
    print()
    test_benchmark_suite()
    print()
    test_baseline_comparator()
    print()
    print("=== All NeuroBench 2025 tests passed ===")
