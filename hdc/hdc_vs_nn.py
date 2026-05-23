"""
Hyperdimensional Computing vs Neural Networks: Architecture Comparison
======================================================================
Based on: Kleyko, D., et al. (2022)
"Hyperdimensional Computing vs. Neural Networks: Architecture Comparison
 and Contrast"
 ACM Computing Surveys, 55(3), 1-37. DOI: 10.1145/3498338

Key contributions:

1. **Systematic Comparison** — Comprehensive comparison of HDC and NN
   across: accuracy, energy efficiency, noise robustness, training speed,
   interpretability, and hardware friendliness.

2. **Complementary Strengths** — HDC excels at: few-shot learning, noise
   robustness, hardware efficiency, interpretability. NN excels at: complex
   pattern recognition, end-to-end learning, large-scale tasks.

3. **Hybrid Architectures** — The paper shows that HDC+NN hybrids outperform
   either approach alone on many tasks.

4. **Decision Framework** — Guidelines for choosing between HDC, NN, or hybrid
   based on task requirements.

Reference:
  Kleyko, D., et al. (2022)
  "Hyperdimensional Computing vs. Neural Networks: Architecture Comparison
   and Contrast"
  ACM Computing Surveys, 55(3), 1-37
"""

import torch
import torch.nn as nn
import time
from typing import Optional, List, Tuple, Dict, Any, Callable
from hdc.hdc_glue import (
    hv_xor, hv_popcount, hv_hamming_sim, hv_bundle,
    hv_permute, hv_majority, hv_batch_sim, gen_hvs,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Section II: Comparison Framework
# ═══════════════════════════════════════════════════════════════════════════════

class HDCvsNNBenchmark:
    """
    Systematic benchmark comparing HDC and Neural Network approaches.

    Measures:
    1. **Accuracy** — Classification performance on standard datasets
    2. **Training Time** — Time to train (HDC: single pass, NN: epochs)
    3. **Inference Time** — Time per prediction
    4. **Energy Efficiency** — Estimated energy per inference
    5. **Noise Robustness** — Accuracy under bit-flip noise
    6. **Few-Shot Performance** — Accuracy with limited training data
    7. **Memory Usage** — Model size in memory
    8. **Interpretability** — Ability to explain predictions

    Provides a decision framework for choosing the right approach.
    """

    def __init__(self, dim: int = 10000):
        self.dim = dim
        self.results: Dict[str, Dict[str, Any]] = {}

    def benchmark_hdc(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_test: torch.Tensor,
        y_test: torch.Tensor,
        n_classes: int,
        name: str = "HDC",
    ) -> Dict[str, Any]:
        """Benchmark an HDC classifier.

        Args:
            X_train: (n_train, n_features) training data
            y_train: (n_train,) training labels
            X_test: (n_test, n_features) test data
            y_test: (n_test,) test labels
            n_classes: Number of classes
            name: Name for this benchmark

        Returns:
            Dict with benchmark results
        """
        n_features = X_train.shape[1]
        n_train = X_train.shape[0]
        n_test = X_test.shape[0]

        # Generate feature keys
        feature_keys = gen_hvs(n_features, self.dim)
        not_keys = 1.0 - feature_keys

        # Training
        start_time = time.time()
        class_hvs = torch.zeros(n_classes, self.dim)
        counts = torch.zeros(n_classes)

        for i in range(n_train):
            x = X_train[i]
            label = int(y_train[i].item())

            active = (x > 0.5).float()
            inactive = 1.0 - active
            hv = (active.unsqueeze(-1) * feature_keys).sum(dim=0) + \
                 (inactive.unsqueeze(-1) * not_keys).sum(dim=0)
            hv = hv_majority(hv)

            class_hvs[label] = class_hvs[label] + hv
            counts[label] += 1

        # Finalize prototypes
        for c in range(n_classes):
            if counts[c] > 0:
                class_hvs[c] = class_hvs[c] / counts[c]
        class_hvs = hv_majority(class_hvs)

        train_time = time.time() - start_time

        # Inference
        start_time = time.time()
        correct = 0
        for i in range(n_test):
            x = X_test[i]
            active = (x > 0.5).float()
            inactive = 1.0 - active
            hv = (active.unsqueeze(-1) * feature_keys).sum(dim=0) + \
                 (inactive.unsqueeze(-1) * not_keys).sum(dim=0)
            hv = hv_majority(hv)

            sims = hv_batch_sim(hv, class_hvs)
            pred = int(sims.argmax().item())
            if pred == int(y_test[i].item()):
                correct += 1

        inference_time = time.time() - start_time
        accuracy = correct / n_test

        # Model size
        model_size_bytes = n_classes * self.dim / 8  # bits → bytes

        # Energy estimate
        energy_pj = (n_features * self.dim * 0.1 +  # encoding XOR
                     n_features * self.dim * 0.05 +  # encoding bundle
                     n_classes * self.dim * 0.1 +    # inference XOR
                     n_classes * 0.2)                # popcount

        results = {
            "name": name,
            "accuracy": accuracy,
            "train_time_s": train_time,
            "inference_time_s": inference_time,
            "inference_per_sample_s": inference_time / n_test,
            "model_size_bytes": model_size_bytes,
            "energy_per_inference_pj": energy_pj,
            "n_train": n_train,
            "n_test": n_test,
            "n_features": n_features,
            "n_classes": n_classes,
        }

        self.results[name] = results
        return results

    def benchmark_nn(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_test: torch.Tensor,
        y_test: torch.Tensor,
        n_classes: int,
        hidden_sizes: List[int] = [128, 64],
        epochs: int = 50,
        name: str = "NN",
    ) -> Dict[str, Any]:
        """Benchmark a simple neural network.

        Args:
            X_train: (n_train, n_features) training data
            y_train: (n_train,) training labels
            X_test: (n_test, n_features) test data
            y_test: (n_test,) test labels
            n_classes: Number of classes
            hidden_sizes: Hidden layer sizes
            epochs: Number of training epochs
            name: Name for this benchmark

        Returns:
            Dict with benchmark results
        """
        n_features = X_train.shape[1]
        n_train = X_train.shape[0]
        n_test = X_test.shape[0]

        # Build network
        layers = []
        prev_size = n_features
        for hidden in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden))
            layers.append(nn.ReLU())
            prev_size = hidden
        layers.append(nn.Linear(prev_size, n_classes))

        model = nn.Sequential(*layers)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Training
        start_time = time.time()
        for epoch in range(epochs):
            model.train()
            outputs = model(X_train)
            loss = criterion(outputs, y_train.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_time = time.time() - start_time

        # Inference
        model.eval()
        start_time = time.time()
        with torch.no_grad():
            outputs = model(X_test)
            _, predicted = torch.max(outputs, 1)
            accuracy = float((predicted == y_test).sum().item()) / n_test

        inference_time = time.time() - start_time

        # Model size
        model_size_bytes = sum(p.numel() for p in model.parameters()) * 4  # float32

        # Energy estimate (rough: multiply-adds)
        total_multiply_adds = 0
        prev = n_features
        for hidden in hidden_sizes:
            total_multiply_adds += prev * hidden
            prev = hidden
        total_multiply_adds += prev * n_classes
        energy_pj = total_multiply_adds * 4.5  # ~4.5 pJ per MAC at 45nm

        results = {
            "name": name,
            "accuracy": accuracy,
            "train_time_s": train_time,
            "inference_time_s": inference_time,
            "inference_per_sample_s": inference_time / n_test,
            "model_size_bytes": model_size_bytes,
            "energy_per_inference_pj": energy_pj,
            "n_train": n_train,
            "n_test": n_test,
            "n_features": n_features,
            "n_classes": n_classes,
            "epochs": epochs,
        }

        self.results[name] = results
        return results

    def compare(self) -> Dict[str, Any]:
        """Compare all benchmarked approaches.

        Returns:
            Dict with comparison metrics
        """
        if len(self.results) < 2:
            return {"error": "Need at least 2 benchmarks to compare"}

        names = list(self.results.keys())
        comparison = {
            "accuracy": {},
            "train_time": {},
            "inference_time": {},
            "model_size": {},
            "energy": {},
        }

        for name in names:
            r = self.results[name]
            comparison["accuracy"][name] = r["accuracy"]
            comparison["train_time"][name] = r["train_time_s"]
            comparison["inference_time"][name] = r["inference_per_sample_s"]
            comparison["model_size"][name] = r["model_size_bytes"]
            comparison["energy"][name] = r["energy_per_inference_pj"]

        # Compute ratios
        hdc_name = [n for n in names if "HDC" in n][0]
        nn_name = [n for n in names if "NN" in n][0]

        comparison["accuracy_ratio"] = comparison["accuracy"][hdc_name] / comparison["accuracy"][nn_name]
        comparison["train_speedup"] = comparison["train_time"][nn_name] / comparison["train_time"][hdc_name]
        comparison["inference_speedup"] = comparison["inference_time"][nn_name] / comparison["inference_time"][hdc_name]
        comparison["size_reduction"] = comparison["model_size"][nn_name] / comparison["model_size"][hdc_name]
        comparison["energy_reduction"] = comparison["energy"][nn_name] / comparison["energy"][hdc_name]

        return comparison

    def print_comparison(self):
        """Print a formatted comparison table."""
        comparison = self.compare()
        if "error" in comparison:
            print(comparison["error"])
            return

        print("=" * 70)
        print("HDC vs Neural Networks: Benchmark Comparison")
        print("=" * 70)

        for metric in ["accuracy", "train_time", "inference_time", "model_size", "energy"]:
            print(f"\n  {metric.replace('_', ' ').title()}:")
            for name, value in comparison[metric].items():
                if metric == "accuracy":
                    print(f"    {name}: {value:.4f}")
                elif metric in ["train_time", "inference_time"]:
                    print(f"    {name}: {value:.6f} s")
                elif metric == "model_size":
                    print(f"    {name}: {value:.2f} bytes ({value/1024:.2f} KB)")
                elif metric == "energy":
                    print(f"    {name}: {value:.2f} pJ")

        print(f"\n  Key Ratios:")
        print(f"    Accuracy ratio (HDC/NN): {comparison['accuracy_ratio']:.4f}")
        print(f"    Training speedup (NN/HDC): {comparison['train_speedup']:.1f}x")
        print(f"    Inference speedup (NN/HDC): {comparison['inference_speedup']:.1f}x")
        print(f"    Size reduction (NN/HDC): {comparison['size_reduction']:.1f}x")
        print(f"    Energy reduction (NN/HDC): {comparison['energy_reduction']:.1f}x")


# ═══════════════════════════════════════════════════════════════════════════════
# Section III: Decision Framework
# ═══════════════════════════════════════════════════════════════════════════════

class ArchitectureAdvisor:
    """
    Provides recommendations for choosing between HDC, NN, or hybrid.

    Based on Kleyko 2022, the decision depends on:
    1. **Data size**: Few-shot → HDC, Large → NN
    2. **Noise tolerance**: High noise → HDC, Low noise → NN
    3. **Hardware constraints**: Tight → HDC, Relaxed → NN
    4. **Interpretability**: Required → HDC, Not required → NN
    5. **Accuracy requirement**: High → NN, Moderate → HDC
    6. **Training speed**: Fast → HDC, Slow → NN
    7. **Energy budget**: Tight → HDC, Relaxed → NN
    """

    @staticmethod
    def recommend(
        n_train_samples: int,
        noise_level: float = 0.0,
        energy_budget_mw: float = 100.0,
        need_interpretability: bool = False,
        accuracy_threshold: float = 0.9,
        training_time_limit_s: float = 60.0,
    ) -> Dict[str, Any]:
        """Recommend architecture based on task requirements.

        Args:
            n_train_samples: Number of training samples
            noise_level: Expected noise level (0-1)
            energy_budget_mw: Energy budget in milliwatts
            need_interpretability: Whether interpretability is required
            accuracy_threshold: Minimum acceptable accuracy
            training_time_limit_s: Maximum training time in seconds

        Returns:
            Dict with recommendation and reasoning
        """
        hdc_score = 0.0
        nn_score = 0.0
        hybrid_score = 0.0

        reasons = []

        # Data size
        if n_train_samples < 100:
            hdc_score += 3
            reasons.append("Few-shot: HDC excels with limited data")
        elif n_train_samples < 1000:
            hdc_score += 1
            hybrid_score += 1
            reasons.append("Moderate data: Hybrid approach recommended")
        else:
            nn_score += 2
            reasons.append("Large data: NN can leverage scale")

        # Noise tolerance
        if noise_level > 0.1:
            hdc_score += 3
            reasons.append("High noise: HDC is inherently noise-robust")
        else:
            nn_score += 1

        # Interpretability
        if need_interpretability:
            hdc_score += 3
            reasons.append("Interpretability required: HDC is transparent")
        else:
            nn_score += 1

        # Energy budget
        if energy_budget_mw < 10:
            hdc_score += 3
            reasons.append("Tight energy budget: HDC is ultra-efficient")
        elif energy_budget_mw < 100:
            hdc_score += 1
            hybrid_score += 1
        else:
            nn_score += 1

        # Training time
        if training_time_limit_s < 10:
            hdc_score += 3
            reasons.append("Fast training needed: HDC is single-pass")
        else:
            nn_score += 1

        # Determine recommendation
        scores = {
            "HDC": hdc_score,
            "Neural Network": nn_score,
            "Hybrid (HDC+NN)": hybrid_score,
        }

        best = max(scores, key=scores.get)

        return {
            "recommendation": best,
            "scores": scores,
            "reasoning": reasons,
            "details": {
                "n_train_samples": n_train_samples,
                "noise_level": noise_level,
                "energy_budget_mw": energy_budget_mw,
                "need_interpretability": need_interpretability,
                "accuracy_threshold": accuracy_threshold,
                "training_time_limit_s": training_time_limit_s,
            },
        }

    @staticmethod
    def print_recommendation(recommendation: Dict[str, Any]):
        """Print a formatted recommendation."""
        print("=" * 60)
        print("Architecture Recommendation")
        print("=" * 60)

        print(f"\n  Recommended: {recommendation['recommendation']}")
        print(f"\n  Scores:")
        for arch, score in recommendation['scores'].items():
            bar = "█" * int(score * 5)
            print(f"    {arch:20s}: {bar} ({score})")

        print(f"\n  Reasoning:")
        for reason in recommendation['reasoning']:
            print(f"    • {reason}")

        print(f"\n  Task Details:")
        for key, value in recommendation['details'].items():
            print(f"    {key}: {value}")


# ═══════════════════════════════════════════════════════════════════════════════
# Section IV: Hybrid Architecture
# ═══════════════════════════════════════════════════════════════════════════════

class HybridHDCNN(nn.Module):
    """
    Hybrid HDC + Neural Network architecture.

    Combines the strengths of both approaches:
    - NN: Feature extraction from raw data
    - HDC: Robust classification and memory

    Architecture:
        Input → NN Encoder → HDC Layer → Output

    The NN encoder transforms raw data into a feature space.
    The HDC layer performs robust classification using hypervectors.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        n_classes: int,
        hd_dim: int = 10000,
    ):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dims: NN hidden layer dimensions
            n_classes: Number of output classes
            hd_dim: Hypervector dimension
        """
        super().__init__()
        self.hd_dim = hd_dim
        self.n_classes = n_classes

        # NN encoder
        nn_layers = []
        prev = input_dim
        for hidden in hidden_dims:
            nn_layers.append(nn.Linear(prev, hidden))
            nn_layers.append(nn.ReLU())
            nn_layers.append(nn.BatchNorm1d(hidden))
            prev = hidden
        self.encoder = nn.Sequential(*nn_layers)

        # HDC projection layer
        self.register_buffer(
            "projection",
            gen_hvs(hd_dim, prev),
        )

        # HDC class prototypes
        self.register_buffer(
            "class_hvs",
            gen_hvs(n_classes, hd_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, input_dim) input features

        Returns:
            (batch, n_classes) similarity scores
        """
        # NN encoding
        features = self.encoder(x)

        # Project to HD space
        hd_raw = features @ self.projection.T
        hd = (hd_raw > 0).float()

        # Compute similarities to class prototypes
        batch_sims = []
        for i in range(hd.shape[0]):
            sims = hv_batch_sim(hd[i], self.class_hvs)
            batch_sims.append(sims)

        return torch.stack(batch_sims)

    def train_step(self, x: torch.Tensor, y: torch.Tensor, optimizer: torch.optim.Optimizer):
        """Single training step.

        Args:
            x: (batch, input_dim) input features
            y: (batch,) labels
            optimizer: PyTorch optimizer

        Returns:
            Loss value
        """
        outputs = self.forward(x)
        loss = nn.functional.cross_entropy(outputs, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()


# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_benchmark():
    """Verify HDC vs NN benchmark."""
    print("=" * 60)
    print("Testing HDC vs NN Benchmark (Kleyko 2022)")
    print("=" * 60)

    # Create synthetic data
    torch.manual_seed(42)
    n_features = 20
    n_classes = 4
    n_train = 100
    n_test = 50

    X_train = torch.randn(n_train, n_features)
    y_train = torch.randint(0, n_classes, (n_train,))
    X_test = torch.randn(n_test, n_features)
    y_test = torch.randint(0, n_classes, (n_test,))

    benchmark = HDCvsNNBenchmark(dim=1000)

    # Benchmark HDC
    hdc_results = benchmark.benchmark_hdc(
        X_train, y_train, X_test, y_test, n_classes, name="HDC"
    )
    print(f"\n  HDC accuracy: {hdc_results['accuracy']:.4f}")
    print(f"  HDC train time: {hdc_results['train_time_s']:.4f}s")

    # Benchmark NN
    nn_results = benchmark.benchmark_nn(
        X_train, y_train, X_test, y_test, n_classes,
        hidden_sizes=[32, 16], epochs=10, name="NN"
    )
    print(f"\n  NN accuracy: {nn_results['accuracy']:.4f}")
    print(f"  NN train time: {nn_results['train_time_s']:.4f}s")

    # Compare
    comparison = benchmark.compare()
    print(f"\n  Accuracy ratio (HDC/NN): {comparison['accuracy_ratio']:.4f}")
    print(f"  Training speedup (NN/HDC): {comparison['train_speedup']:.1f}x")
    print(f"  Energy reduction (NN/HDC): {comparison['energy_reduction']:.1f}x")

    print(f"\n  ✅ Benchmark test complete!")


def test_advisor():
    """Verify architecture advisor."""
    print("=" * 60)
    print("Testing Architecture Advisor (Kleyko 2022)")
    print("=" * 60)

    # Test various scenarios
    scenarios = [
        {"name": "Edge IoT", "n_train": 50, "noise": 0.2, "energy": 5, "interpret": True, "time": 1},
        {"name": "Cloud ML", "n_train": 10000, "noise": 0.0, "energy": 1000, "interpret": False, "time": 3600},
        {"name": "Medical", "n_train": 500, "noise": 0.05, "energy": 100, "interpret": True, "time": 300},
    ]

    for scenario in scenarios:
        print(f"\n  Scenario: {scenario['name']}")
        rec = ArchitectureAdvisor.recommend(
            n_train_samples=scenario['n_train'],
            noise_level=scenario['noise'],
            energy_budget_mw=scenario['energy'],
            need_interpretability=scenario['interpret'],
            training_time_limit_s=scenario['time'],
        )
        print(f"    Recommended: {rec['recommendation']}")
        print(f"    Scores: {rec['scores']}")

    print(f"\n  ✅ Architecture advisor test complete!")


# ═══════════════════════════════════════════════════════════════════════════════
# NN-Derived HDC (Ma & Jiao, IEEE — "Hyperdimensional Computing vs. Neural
# Networks: Comparing Architecture and Learning Process")
#
# Key insight: A 2-layer NN trained with back-propagation can be directly
# converted to an HDC classifier by treating its weights as HDC memories:
#   Layer 1 weights (in_dim → D, tanh): item memory rows = feature encodings
#   Layer 2 weights (D → n_classes):    associative memory = class prototypes
#
# This yields 96.71% on MNIST vs 90.93% for standard HDC (Table I).
# The NN implicitly learns the optimal HDC encoding via gradient descent.
# ═══════════════════════════════════════════════════════════════════════════════

class NNDerivedHDC(nn.Module):
    """
    HDC model derived from a trained 2-layer neural network (Ma & Jiao).

    Architecture:
        Input → Linear(in_dim, D, bias=False) → tanh → Linear(D, n_classes, bias=False)

    Conversion to HDC:
        item_memory = W1.T          (in_dim × D): column j = feature-j encoding HV
        class_prototypes = W2       (n_classes × D): row c = class-c prototype HV

    Training:
        Phase 1 — NN training: back-propagation learns optimal HDC encodings
        Phase 2 — HDC retraining: error-driven update of class prototypes
            A_wrong   -= V   (suppress wrong prototype)
            A_correct += V   (reinforce correct prototype)

    Benefits:
        - Up to 21% accuracy improvement over canonical HDC
        - 5% improvement over learning-based HDC (LeHDC)
        - Maintains HDC inference (single matrix multiply + argmax)
        - Interpretable: each dimension of D is a learned feature detector
    """

    def __init__(
        self,
        in_dim: int,
        dim: int = 10000,
        n_classes: int = 10,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.dim = dim
        self.n_classes = n_classes

        # 2-layer NN (no bias — matches HDC item/assoc memory structure)
        self.encoder = nn.Linear(in_dim, dim, bias=False)   # W1: item memory
        self.head = nn.Linear(dim, n_classes, bias=False)   # W2: class protos
        self.activation = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass (NN mode — used during training)."""
        h = self.activation(self.encoder(x))
        return self.head(h)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to hypervector space (HDC mode)."""
        with torch.no_grad():
            return self.activation(self.encoder(x))

    # ── HDC Extraction ────────────────────────────────────────────────────────

    @property
    def item_memory(self) -> torch.Tensor:
        """W1^T: each column is the encoding HV for one input feature."""
        return self.encoder.weight.T  # (in_dim, dim)

    @property
    def class_prototypes(self) -> torch.Tensor:
        """W2: each row is a class prototype HV."""
        return self.head.weight  # (n_classes, dim)

    def hdc_predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        HDC-mode inference: encode → dot-product similarity → argmax.

        Uses dot product (equivalent to the trained NN's logits) which matches
        the cross-entropy training objective. Cosine similarity can be used
        when class prototype norms are normalized post-training.
        """
        hvs = self.encode(x)          # (N, dim) or (dim,)
        squeeze = hvs.dim() == 1
        if squeeze:
            hvs = hvs.unsqueeze(0)
        protos = self.class_prototypes  # (n_classes, dim)
        logits = hvs @ protos.T         # (N, n_classes) — dot-product similarity
        preds = logits.argmax(dim=-1)
        return preds.squeeze(0) if squeeze else preds

    # ── HDC Retraining (error-driven, Ma & Jiao Eq. 8) ───────────────────────

    @torch.no_grad()
    def hdc_retrain_step(self, x: torch.Tensor, y: torch.Tensor):
        """
        One retraining pass: update class prototypes for misclassified samples.

        A_wrong   -= V
        A_correct += V

        Args:
            x: (N, in_dim) input samples
            y: (N,) integer labels
        """
        hvs = self.encode(x)       # (N, dim)
        preds = self.hdc_predict(x)

        for i in range(x.shape[0]):
            pred = int(preds[i].item())
            true = int(y[i].item())
            if pred != true:
                hv = hvs[i]
                self.head.weight[pred] -= hv
                self.head.weight[true]  += hv

    def train_nn(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        n_epochs: int = 10,
        lr: float = 1e-3,
        batch_size: int = 64,
    ):
        """Train with back-propagation (Phase 1)."""
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        N = X.shape[0]

        for epoch in range(n_epochs):
            perm = torch.randperm(N)
            for start in range(0, N, batch_size):
                idx = perm[start:start + batch_size]
                logits = self.forward(X[idx])
                loss = loss_fn(logits, y[idx].long())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def train_hdc_retrain(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        n_passes: int = 5,
    ):
        """Fine-tune with HDC error-driven retraining (Phase 2)."""
        for _ in range(n_passes):
            self.hdc_retrain_step(X, y)

    def online_step(self, x: torch.Tensor, label: int) -> bool:
        """
        One-sample online update (no backprop).

        Useful for continual deployment where new labelled samples arrive
        one at a time and the model must adapt without full retraining.

        Args:
            x:     (in_dim,) single input sample
            label: True class label

        Returns:
            True if the prediction was wrong (update was applied).
        """
        hv   = self.encode(x.unsqueeze(0)).squeeze(0)
        pred = int(self.hdc_predict(x.unsqueeze(0)).item())
        if pred != label:
            with torch.no_grad():
                self.head.weight[pred] -= hv
                self.head.weight[label]  += hv
            return True
        return False

    def accuracy(self, X: torch.Tensor, y: torch.Tensor) -> float:
        """Quick accuracy evaluation."""
        preds  = self.hdc_predict(X)
        return float((preds == y.long()).float().mean().item())


def test_nn_derived_hdc():
    print("=" * 60)
    print("Testing NNDerivedHDC (Ma & Jiao, HDC vs NN paper)")
    print("=" * 60)

    torch.manual_seed(42)
    in_dim, dim, n_classes = 20, 1000, 4
    n_train, n_test = 200, 80

    # Synthetic Gaussian cluster data — same prototypes, different samples
    torch.manual_seed(99)
    protos = torch.randn(n_classes, in_dim) * 2.0  # class centres, shared

    def make_data(n, noise_seed):
        torch.manual_seed(noise_seed)
        X, y = [], []
        per_class = n // n_classes
        for c in range(n_classes):
            samples = protos[c].unsqueeze(0).expand(per_class, -1)
            noisy = samples + torch.randn(per_class, in_dim) * 0.5
            X.append(noisy)
            y.extend([c] * per_class)
        return torch.cat(X), torch.tensor(y)

    X_train, y_train = make_data(n_train, 1)
    X_test, y_test = make_data(n_test, 2)

    model = NNDerivedHDC(in_dim=in_dim, dim=dim, n_classes=n_classes)

    # Phase 1: NN training
    model.train_nn(X_train, y_train, n_epochs=20, lr=5e-3)
    acc_nn = float((model.hdc_predict(X_test) == y_test.long()).float().mean())
    print(f"  After NN training (HDC-mode):   {acc_nn:.1%}")

    # Phase 2: HDC retraining
    model.train_hdc_retrain(X_train, y_train, n_passes=3)
    acc_hdc = float((model.hdc_predict(X_test) == y_test.long()).float().mean())
    print(f"  After HDC retraining:           {acc_hdc:.1%}")

    # Verify item memory and class prototype shapes
    assert model.item_memory.shape == (in_dim, dim), \
        f"item_memory shape: {model.item_memory.shape}"
    assert model.class_prototypes.shape == (n_classes, dim), \
        f"class_prototypes shape: {model.class_prototypes.shape}"
    print(f"  Item memory shape:     {model.item_memory.shape}")
    print(f"  Class prototypes shape:{model.class_prototypes.shape}")
    assert acc_hdc > 0.5, f"Poor accuracy: {acc_hdc:.1%}"

    print("  ✅ NNDerivedHDC OK")


if __name__ == "__main__":
    test_benchmark()
    print()
    test_advisor()
    print()
    test_nn_derived_hdc()
