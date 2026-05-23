"""
Arthedain Benchmark Suite
==========================
Compares SelfImprovementLoop against standard ML baselines on real tasks.
Also provides the UCR time-series and CSV data adapters.
"""

from __future__ import annotations

import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════════
# Data Adapters
# ═══════════════════════════════════════════════════════════════════════════════

def load_ucr(path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load a UCR time-series dataset from a .tsv or .txt file.

    UCR format: first column = class label (integer or float), remaining
    columns = time-series values. No header row.

    Download datasets from: https://www.timeseriesclassification.com/

    Args:
        path: Path to UCR .tsv/.txt file

    Returns:
        (X, y) — (N, T) float time series, (N,) long labels (0-indexed)
    """
    rows, labels = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            try:
                label = float(parts[0])
                values = [float(v) for v in parts[1:] if v not in ('NaN', 'nan', '')]
                if values:
                    labels.append(label)
                    rows.append(values)
            except (ValueError, IndexError):
                continue   # skip header or malformed lines

    if not rows:
        raise ValueError(f"No valid data found in {path}")

    X = torch.tensor(rows, dtype=torch.float32)
    y_raw = torch.tensor(labels, dtype=torch.float32)
    # Map labels to 0-indexed integers
    unique = y_raw.unique()
    label_map = {float(v): i for i, v in enumerate(unique)}
    y = torch.tensor([label_map[float(v)] for v in y_raw], dtype=torch.long)
    return X, y


def load_csv(path: str, label_col: int = -1, delimiter: str = ",") -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load a CSV file into (X, y) tensors.

    Args:
        path: Path to CSV file
        label_col: Column index of the label (-1 = last column)
        delimiter: Field delimiter

    Returns:
        (X, y) — float feature matrix and long label vector
    """
    rows, labels = [], []
    with open(path) as f:
        reader = csv.reader(f, delimiter=delimiter)
        for row in reader:
            if not row or row[0].startswith('#'):
                continue
            try:
                values = [float(v) for v in row]
            except ValueError:
                continue   # skip header
            if label_col == -1:
                labels.append(values[-1])
                rows.append(values[:-1])
            else:
                labels.append(values[label_col])
                rows.append(values[:label_col] + values[label_col+1:])

    X = torch.tensor(rows, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    return X, y


def generate_synthetic_benchmark(
    n_classes: int = 4,
    n_features: int = 20,
    n_train: int = 200,
    n_test: int = 80,
    noise: float = 0.3,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate a Gaussian cluster classification benchmark.

    Returns:
        X_train, y_train, X_test, y_test
    """
    torch.manual_seed(seed)
    protos = torch.randn(n_classes, n_features) * 2.0

    def _make(n, noise_seed):
        torch.manual_seed(noise_seed)
        X, y = [], []
        per = n // n_classes
        for c in range(n_classes):
            samples = protos[c].unsqueeze(0).expand(per, -1)
            X.append(samples + torch.randn(per, n_features) * noise)
            y.extend([c] * per)
        return torch.cat(X), torch.tensor(y)

    return (*_make(n_train, seed+1), *_make(n_test, seed+2))


def generate_temporal_benchmark(
    n_classes: int = 3,
    seq_len: int = 50,
    n_train: int = 150,
    n_test: int = 60,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate a temporal pattern classification task.

    Each class has a characteristic frequency component embedded in noise.
    """
    torch.manual_seed(seed)
    freqs = [0.1 * (c + 1) for c in range(n_classes)]
    t = torch.linspace(0, 2 * math.pi, seq_len)

    def _make(n, noise_seed):
        torch.manual_seed(noise_seed)
        X, y = [], []
        per = n // n_classes
        for c in range(n_classes):
            signal = torch.sin(freqs[c] * t).unsqueeze(0).expand(per, -1)
            X.append(signal + torch.randn(per, seq_len) * 0.5)
            y.extend([c] * per)
        return torch.cat(X), torch.tensor(y)

    return (*_make(n_train, seed+10), *_make(n_test, seed+20))


# ═══════════════════════════════════════════════════════════════════════════════
# Baselines
# ═══════════════════════════════════════════════════════════════════════════════

class LogisticRegressionBaseline:
    """Closed-form logistic regression via gradient descent (50 steps)."""

    def __init__(self, n_features: int, n_classes: int, lr: float = 0.1):
        self.W = torch.zeros(n_classes, n_features)
        self.b = torch.zeros(n_classes)
        self.lr = lr

    def fit(self, X: torch.Tensor, y: torch.Tensor, n_epochs: int = 100):
        for _ in range(n_epochs):
            logits = X @ self.W.T + self.b
            probs = F.softmax(logits, dim=-1)
            n = X.shape[0]
            one_hot = torch.zeros(n, probs.shape[1])
            one_hot.scatter_(1, y.long().unsqueeze(1), 1.0)
            grad_logits = (probs - one_hot) / n
            self.W -= self.lr * (grad_logits.T @ X)
            self.b -= self.lr * grad_logits.sum(dim=0)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        return (X @ self.W.T + self.b).argmax(dim=-1)

    def accuracy(self, X: torch.Tensor, y: torch.Tensor) -> float:
        return float((self.predict(X) == y.long()).float().mean())


class HDCBaseline:
    """Standard one-shot HDC classifier (no retraining, no self-improvement)."""

    def __init__(self, n_features: int, n_classes: int, dim: int = 4096, seed: int = 42):
        from hdc.hdc_glue import gen_hvs
        self.dim = dim
        self.n_classes = n_classes
        self.feature_hvs = gen_hvs(n_features, dim, seed=seed)
        self.prototypes = torch.zeros(n_classes, dim)
        self.counts = torch.zeros(n_classes)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        active = (x > x.mean()).float()
        bipolar = 2.0 * self.feature_hvs - 1.0
        weighted = (active.unsqueeze(-1) * bipolar).sum(dim=0)
        return (weighted > 0).float()

    def fit(self, X: torch.Tensor, y: torch.Tensor):
        for i in range(X.shape[0]):
            hv = self._encode(X[i])
            c = int(y[i].item())
            self.prototypes[c] += hv.float()
            self.counts[c] += 1
        for c in range(self.n_classes):
            if self.counts[c] > 0:
                self.prototypes[c] = (self.prototypes[c] / self.counts[c] > 0.5).float()

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        from hdc.hdc_glue import hv_batch_sim
        preds = []
        for i in range(X.shape[0]):
            hv = self._encode(X[i])
            preds.append(int(hv_batch_sim(hv, self.prototypes).argmax()))
        return torch.tensor(preds)

    def accuracy(self, X: torch.Tensor, y: torch.Tensor) -> float:
        return float((self.predict(X) == y.long()).float().mean())


# ═══════════════════════════════════════════════════════════════════════════════
# Arthedain Agent Wrapper for Benchmarking
# ═══════════════════════════════════════════════════════════════════════════════

class ArthedainBenchmarkWrapper:
    """Wraps SelfImprovementLoop for classification benchmarking."""

    def __init__(self, n_features: int, n_classes: int, dim: int = 1000, seed: int = 42):
        from hdc.sensor_stream import SensorSpec, ModalityType, SensorReading
        from hdc.physical_ai_hybrid import HybridPhysicalAIPipeline, ActionCandidate
        from hdc.world_context import ContextualWorldModel
        from hdc.planner import SelfImprovementLoop, AutoCalibrator
        from hdc.physics_world_model import ActionCandidate as AC

        self.n_classes = n_classes
        self.dim = dim

        specs = [SensorSpec("x", ModalityType.TIME_SERIES, raw_dim=n_features, hd_dim=dim, seed=seed)]
        base = HybridPhysicalAIPipeline(specs, hd_dim=dim, n_ensemble=3, consolidation_period=50)
        world = ContextualWorldModel(base, pattern_window=4, pattern_stride=2)
        self.agent = SelfImprovementLoop(world, beam_width=2, planning_horizon=2,
                                          min_causal_for_planning=30, lr_base=0.01)

        # Class prototype accumulators (for classification head)
        self._proto_accum = torch.zeros(n_classes, dim)
        self._proto_counts = torch.zeros(n_classes)
        self._prototypes: Optional[torch.Tensor] = None

    def _to_reading(self, x: torch.Tensor, t: float):
        from hdc.sensor_stream import SensorReading
        return SensorReading(timestamp=t, data={"x": x.unsqueeze(0)})

    def fit(self, X: torch.Tensor, y: torch.Tensor, n_passes: int = 1):
        from hdc.hdc_glue import hv_batch_sim
        for _ in range(n_passes):
            perm = torch.randperm(X.shape[0])
            for i, idx in enumerate(perm.tolist()):
                result = self.agent.tick(
                    self._to_reading(X[idx], float(i))
                )
                sensor_hv = result["sensor_hv"]
                c = int(y[idx].item())
                self._proto_accum[c] += sensor_hv.float()
                self._proto_counts[c] += 1

        # Binarise prototypes
        counts = self._proto_counts.clamp(min=1).unsqueeze(-1)
        self._prototypes = (self._proto_accum / counts > 0.5).float()

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        assert self._prototypes is not None
        from hdc.hdc_glue import hv_batch_sim
        preds = []
        for i in range(X.shape[0]):
            result = self.agent.tick(self._to_reading(X[i], float(i)), candidate_actions=None)
            hv = result["sensor_hv"]
            sims = hv_batch_sim(hv, self._prototypes)
            preds.append(int(sims.argmax()))
        return torch.tensor(preds)

    def accuracy(self, X: torch.Tensor, y: torch.Tensor) -> float:
        return float((self.predict(X) == y.long()).float().mean())


# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark Runner
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BenchmarkResult:
    method: str
    accuracy: float
    train_time_ms: float
    inference_time_ms: float
    n_train: int
    n_test: int
    n_features: int
    n_classes: int


def run_benchmark(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    dim_hdc: int = 1000,
    seed: int = 42,
) -> List[BenchmarkResult]:
    """
    Run all methods and return comparison results.

    Methods:
      - Logistic Regression (sklearn-free, pure PyTorch)
      - HDC Baseline (one-shot, no retraining)
      - Arthedain (self-improving agent)
    """
    n_f = X_train.shape[1]
    n_c = int(y_train.max().item()) + 1
    n_tr = X_train.shape[0]
    n_te = X_test.shape[0]
    results = []

    # 1. Logistic Regression
    t0 = time.perf_counter()
    lr = LogisticRegressionBaseline(n_f, n_c, lr=0.05)
    lr.fit(X_train, y_train, n_epochs=200)
    t_train = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    acc_lr = lr.accuracy(X_test, y_test)
    t_inf = (time.perf_counter() - t0) * 1000 / n_te

    results.append(BenchmarkResult("LogisticRegression", acc_lr, t_train, t_inf,
                                    n_tr, n_te, n_f, n_c))

    # 2. HDC Baseline
    t0 = time.perf_counter()
    hdc_clf = HDCBaseline(n_f, n_c, dim=dim_hdc, seed=seed)
    hdc_clf.fit(X_train, y_train)
    t_train = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    acc_hdc = hdc_clf.accuracy(X_test, y_test)
    t_inf = (time.perf_counter() - t0) * 1000 / n_te

    results.append(BenchmarkResult("HDC_Baseline", acc_hdc, t_train, t_inf,
                                    n_tr, n_te, n_f, n_c))

    # 3. Arthedain
    t0 = time.perf_counter()
    art = ArthedainBenchmarkWrapper(n_f, n_c, dim=dim_hdc, seed=seed)
    art.fit(X_train, y_train, n_passes=1)
    t_train = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    acc_art = art.accuracy(X_test, y_test)
    t_inf = (time.perf_counter() - t0) * 1000 / n_te

    results.append(BenchmarkResult("Arthedain_SelfImproving", acc_art, t_train, t_inf,
                                    n_tr, n_te, n_f, n_c))

    return results


def print_benchmark_table(results: List[BenchmarkResult]):
    """Print a formatted comparison table."""
    print(f"\n{'Method':<28} {'Accuracy':>9} {'Train(ms)':>10} {'Infer(ms)':>10}")
    print("-" * 62)
    for r in results:
        print(f"  {r.method:<26} {r.accuracy:>8.1%} {r.train_time_ms:>10.0f} {r.inference_time_ms:>10.2f}")
    print()
    best = max(results, key=lambda x: x.accuracy)
    print(f"  Best: {best.method} ({best.accuracy:.1%})")


def test_benchmark():
    print("=" * 60)
    print("Running Arthedain Benchmark vs Baselines")
    print("=" * 60)

    torch.manual_seed(42)
    X_tr, y_tr, X_te, y_te = generate_synthetic_benchmark(
        n_classes=4, n_features=16, n_train=120, n_test=60, noise=0.4
    )
    print(f"  Dataset: {X_tr.shape[0]} train, {X_te.shape[0]} test, "
          f"{X_tr.shape[1]} features, {int(y_tr.max())+1} classes")

    results = run_benchmark(X_tr, y_tr, X_te, y_te, dim_hdc=800)
    print_benchmark_table(results)

    assert any(r.accuracy > 0.4 for r in results), "All methods failed"
    print("  ✅ Benchmark OK")


def test_persistence():
    print("=" * 60)
    print("Testing Agent Persistence (save / load)")
    print("=" * 60)

    import tempfile, os
    from hdc.sensor_stream import SensorSpec, SensorReading, ModalityType
    from hdc.physical_ai_hybrid import HybridPhysicalAIPipeline, ActionCandidate
    from hdc.world_context import ContextualWorldModel
    from hdc.planner import SelfImprovementLoop
    from hdc.persistence import save_agent, load_agent_state

    torch.manual_seed(0)
    dim = 500
    specs = [SensorSpec("s", ModalityType.SCALAR, raw_dim=1, hd_dim=dim, seed=0)]
    base = HybridPhysicalAIPipeline(specs, hd_dim=dim, n_ensemble=2, consolidation_period=50)
    world = ContextualWorldModel(base, pattern_window=4, pattern_stride=3)
    agent = SelfImprovementLoop(world, beam_width=2, planning_horizon=2,
                                  min_causal_for_planning=10, lr_base=0.01)

    # Train for 20 ticks
    for t in range(20):
        r = SensorReading(float(t), {"s": torch.tensor([float(t % 4) / 4])})
        agent.tick(r)

    orig_tick = agent._tick
    orig_n_trans = agent.world.causal_graph.n_transitions
    orig_protos = len(agent.world.pipeline.world_model.action_evaluator._safe_prototypes)

    # Save
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        ckpt_path = f.name
    save_agent(agent, ckpt_path)
    ckpt_size = os.path.getsize(ckpt_path) // 1024

    # Build fresh agent and restore
    base2 = HybridPhysicalAIPipeline(specs, hd_dim=dim, n_ensemble=2, consolidation_period=50)
    world2 = ContextualWorldModel(base2, pattern_window=4, pattern_stride=3)
    agent2 = SelfImprovementLoop(world2, beam_width=2, planning_horizon=2,
                                   min_causal_for_planning=10, lr_base=0.01)
    load_agent_state(agent2, ckpt_path)
    os.unlink(ckpt_path)

    print(f"  Checkpoint size: {ckpt_size}KB")
    print(f"  Tick restored:   {agent2._tick}  (want {orig_tick})")
    print(f"  Causal graph:    {agent2.world.causal_graph.n_transitions} transitions")
    assert agent2._tick == orig_tick, f"Tick mismatch: {agent2._tick} != {orig_tick}"

    print("  ✅ Persistence OK")


if __name__ == "__main__":
    test_persistence()
    print()
    test_benchmark()
    print()
    print("=== All benchmark tests passed ===")
