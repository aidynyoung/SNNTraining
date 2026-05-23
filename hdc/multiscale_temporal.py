"""
Multi-Scale Temporal Encoding for HDC
======================================
Based on: Schlegel et al. 2025 "Structured temporal representation in
time series classification with ROCKETs and hyperdimensional computing"

Key insight: Encode spike windows at multiple timescales and bundle
the multi-scale representations for richer temporal context. This is
analogous to how ROCKET uses multiple kernel lengths, but implemented
entirely in hyperdimensional space.

Reference:
  Schlegel, K., et al. (2025)
  "Structured temporal representation in time series classification
   with ROCKETs and hyperdimensional computing"
"""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple
from models.hdc import gen_hvs, bind, bundle, sim, thresh, batch_sim


class TemporalConvolutionHD(nn.Module):
    """
    Temporal convolution in hyperdimensional space.

    Applies a sliding window over a sequence of hypervectors and
    bundles each window into a single "moment" hypervector. The
    window size determines the temporal resolution.

    This is the HDC analog of 1D convolution: instead of learning
    kernel weights, we use bundling (sum) to aggregate temporal
    context.
    """

    def __init__(
        self,
        dim: int = 10000,
        mode: str = "bipolar",
        window_size: int = 5,
        stride: int = 1,
    ):
        super().__init__()
        self.dim = dim
        self.mode = mode
        self.window_size = window_size
        self.stride = stride

    def forward(self, seq_hvs: torch.Tensor) -> torch.Tensor:
        """Apply temporal convolution to a sequence of hypervectors.

        Args:
            seq_hvs: (T, dim) sequence of hypervectors

        Returns:
            (T_out, dim) convolved hypervectors, where
            T_out = max(1, (T - window_size) // stride + 1)
        """
        T = seq_hvs.shape[0]
        if T < self.window_size:
            # Pad by repeating the last frame
            padding = self.window_size - T
            seq_hvs = torch.cat([seq_hvs, seq_hvs[-1:].repeat(padding, 1)], dim=0)
            T = seq_hvs.shape[0]

        T_out = max(1, (T - self.window_size) // self.stride + 1)
        outputs = []

        for t in range(0, T - self.window_size + 1, self.stride):
            window = seq_hvs[t : t + self.window_size]  # (window_size, dim)
            bundled = bundle(window)  # (dim,)

            if self.mode == "bipolar":
                bundled = thresh(bundled)

            outputs.append(bundled)

        stacked = torch.stack(outputs) if outputs else seq_hvs[:1]
        # Bundle all windows into a single (dim,) hypervector
        result = bundle(stacked)
        if self.mode == "bipolar":
            result = thresh(result)
        return result


class MultiScaleTemporalEncoder(nn.Module):
    """
    Multi-scale temporal encoder for spike trains.

    Encodes a spike train at multiple temporal resolutions and
    bundles the representations into a single rich hypervector.

    Architecture:
        Spike train (T, N)
            ├── Scale 1 (fast): window=5, stride=1 → TemporalConvolutionHD
            ├── Scale 2 (medium): window=20, stride=5 → TemporalConvolutionHD
            └── Scale 3 (slow): window=50, stride=10 → TemporalConvolutionHD
                            ↓
                    Bundle all scales → Multi-scale HV

    This is inspired by Schlegel 2025's ROCKET-HD approach, where
    multiple kernel lengths capture different temporal dynamics.
    """

    def __init__(
        self,
        n_neurons: int,
        dim: int = 10000,
        mode: str = "bipolar",
        n_levels: int = 13,
        scales: Optional[List[Tuple[int, int]]] = None,
        device: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.n_neurons = n_neurons
        self.dim = dim
        self.mode = mode
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Default scales: (window_size, stride)
        if scales is None:
            scales = [
                (5, 1),    # Fast: fine-grained temporal resolution
                (20, 5),   # Medium: moderate temporal context
                (50, 10),  # Slow: broad temporal context
            ]
        self.scales = scales

        # Per-neuron hypervector keys
        self.register_buffer(
            "neuron_keys",
            gen_hvs(n_neurons, dim, mode, self.device, seed),
        )

        # Level hypervectors for scalar encoding
        self.register_buffer(
            "level_hvs",
            gen_hvs(n_levels, dim, mode, self.device, seed + 1 if seed else None),
        )
        self.n_levels = n_levels

        # Temporal convolution modules for each scale
        self.convs = nn.ModuleList([
            TemporalConvolutionHD(dim=dim, mode=mode, window_size=w, stride=s)
            for w, s in scales
        ])

        # Learnable scale weights (which temporal scale is most informative)
        self.scale_log_weights = nn.Parameter(torch.zeros(len(scales)))

    def encode_frame(self, frame: torch.Tensor) -> torch.Tensor:
        """Encode a single time frame of spike activity.

        Args:
            frame: (n_neurons,) spike activity at one time step

        Returns:
            (dim,) hypervector encoding the frame
        """
        mn, mx = frame.min().item(), frame.max().item()
        if mx - mn < 1e-6:
            mx = mn + 1.0

        hvs = []
        for i in range(self.n_neurons):
            value = frame[i].item()
            value = max(0.0, min(1.0, (value - mn) / (mx - mn)))
            level_idx = min(int(value * (self.n_levels - 1)), self.n_levels - 1)

            neuron_hv = self.neuron_keys[i]
            level_hv = self.level_hvs[level_idx]
            hvs.append(bind(neuron_hv, level_hv, self.mode))

        stacked = torch.stack(hvs)  # (n_neurons, dim)
        bundled = bundle(stacked)  # (dim,)

        if self.mode == "bipolar":
            bundled = thresh(bundled)

        return bundled

    def encode_sequence(self, spike_train: torch.Tensor) -> torch.Tensor:
        """Encode a full spike train into per-frame hypervectors.

        Args:
            spike_train: (T, n_neurons) or (n_neurons,) spike train

        Returns:
            (T, dim) sequence of frame hypervectors
        """
        if spike_train.dim() == 1:
            spike_train = spike_train.unsqueeze(0)  # (n_neurons,) → (1, n_neurons)
        frames = []
        for t in range(spike_train.shape[0]):
            frames.append(self.encode_frame(spike_train[t]))
        return torch.stack(frames)

    def forward(self, spike_train: torch.Tensor) -> torch.Tensor:
        """Encode spike train at multiple temporal scales.

        Args:
            spike_train: (T, n_neurons) spike train

        Returns:
            (dim,) multi-scale hypervector
        """
        # Encode to per-frame hypervectors
        seq_hvs = self.encode_sequence(spike_train)  # (T, dim)

        # Apply each temporal scale
        scale_hvs = []
        for conv in self.convs:
            convolved = conv(seq_hvs)  # (T_out, dim)
            # Bundle all windows at this scale
            scale_bundle = bundle(convolved)  # (dim,)
            if self.mode == "bipolar":
                scale_bundle = thresh(scale_bundle)
            scale_hvs.append(scale_bundle)

        # Weighted superposition of scales
        scale_weights = torch.softmax(self.scale_log_weights, dim=0)
        stacked = torch.stack(scale_hvs)  # (n_scales, dim)
        weighted = stacked * scale_weights.unsqueeze(-1)
        result = weighted.sum(dim=0)  # (dim,)

        if self.mode == "bipolar":
            result = thresh(result)

        return result

    def get_scale_importance(self) -> torch.Tensor:
        """Return learned importance weights for each temporal scale."""
        return torch.softmax(self.scale_log_weights, dim=0).detach().cpu()


class MultiScaleHDCClassifier(nn.Module):
    """
    Complete multi-scale HDC classifier for spike trains.

    Combines:
    1. Multi-scale temporal encoding (Schlegel 2025)
    2. Weighted associative memory (Schlegel 2024)
    3. One-shot learning via bundling

    This is the recommended replacement for SpikeHDC when working
    with temporal spike train data.
    """

    def __init__(
        self,
        n_neurons: int,
        n_classes: int,
        dim: int = 10000,
        mode: str = "bipolar",
        n_levels: int = 13,
        scales: Optional[List[Tuple[int, int]]] = None,
        device: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.n_neurons = n_neurons
        self.n_classes = n_classes
        self.dim = dim
        self.mode = mode
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Multi-scale temporal encoder
        self.encoder = MultiScaleTemporalEncoder(
            n_neurons=n_neurons,
            dim=dim,
            mode=mode,
            n_levels=n_levels,
            scales=scales,
            device=self.device,
            seed=seed,
        )

        # Class hypervectors (associative memory)
        self.register_buffer(
            "class_hvs",
            gen_hvs(n_classes, dim, mode, self.device, seed + 2 if seed else None),
        )
        self.register_buffer("counts", torch.zeros(n_classes, device=self.device))

    def train_step(self, spike_train: torch.Tensor, label: int):
        """Add one training example.

        Args:
            spike_train: (T, n_neurons) spike train
            label: class index
        """
        hv = self.encoder(spike_train)
        self.class_hvs[label] = self.class_hvs[label] + hv
        self.counts[label] += 1

    def finalize(self):
        """Normalize class hypervectors after training."""
        if self.mode == "bipolar":
            self.class_hvs = thresh(self.class_hvs)
        elif self.mode == "binary":
            self.class_hvs = (
                self.class_hvs >= self.class_hvs.mean(dim=1, keepdim=True)
            ).float()
        else:
            self.class_hvs = self.class_hvs / self.class_hvs.norm(
                dim=1, keepdim=True
            ).clamp(min=1e-12)

    def predict(self, spike_train: torch.Tensor) -> int:
        """Predict class for a spike train.

        Args:
            spike_train: (T, n_neurons) spike train

        Returns:
            predicted class index
        """
        hv = self.encoder(spike_train)
        similarities = batch_sim(hv, self.class_hvs, self.mode)
        return int(similarities.argmax().item())

    def train_step_refine(
        self,
        spike_train: torch.Tensor,
        label: int,
        lr: float = 0.5,
    ):
        """
        Online RefineHD update: pull correct prototype, push wrong prototype.

        Reference: Vergés Boncompte (2025) §4.2 RefineHD algorithm.

        If prediction is wrong: update both correct and predicted class prototypes
        to reduce the error (push/pull update). If prediction is correct: small
        reinforcement update to tighten the correct class cluster.

        Args:
            spike_train: (T, n_neurons) spike train
            label:       True class index
            lr:          Learning rate (default 0.5 for HDC bundling scale)
        """
        hv = self.encoder(spike_train).to(self.class_hvs.device)
        sims    = batch_sim(hv, self.class_hvs, self.mode)
        pred    = int(sims.argmax().item())

        with torch.no_grad():
            if pred != label:
                # Pull correct prototype toward hv
                self.class_hvs[label]  = self.class_hvs[label]  + lr * hv
                # Push wrong prototype away from hv
                self.class_hvs[pred]   = self.class_hvs[pred]   - lr * hv
            else:
                # Reinforce: mild pull
                self.class_hvs[label]  = self.class_hvs[label]  + 0.1 * lr * hv

    def predict_with_confidence(
        self,
        spike_train: torch.Tensor,
    ) -> Tuple[int, float, float]:
        """
        Predict class with confidence and margin scores.

        Returns:
            (predicted_label, top_similarity, margin)
            margin = top_sim - second_top_sim (higher = more confident)
        """
        hv   = self.encoder(spike_train)
        sims = batch_sim(hv, self.class_hvs, self.mode)
        topk = sims.topk(min(2, self.n_classes))
        pred = int(topk.indices[0].item())
        top_sim = float(topk.values[0].item())
        margin  = float((topk.values[0] - topk.values[1]).item()) if len(topk.values) > 1 else 1.0
        return pred, top_sim, margin

    def forward(self, spike_train: torch.Tensor) -> torch.Tensor:
        """Return class similarities for a spike train.

        Args:
            spike_train: (T, n_neurons) spike train

        Returns:
            (n_classes,) similarity scores
        """
        hv = self.encoder(spike_train)
        return batch_sim(hv, self.class_hvs, self.mode)


# ── Tests ────────────────────────────────────────────────────────────────────
def test_multiscale_temporal():
    """Verify multi-scale temporal encoding captures temporal structure."""
    print("=" * 60)
    print("Testing Multi-Scale Temporal Encoding")
    print("=" * 60)

    dim = 2000
    n_neurons = 16
    n_classes = 3
    T = 100

    encoder = MultiScaleTemporalEncoder(
        n_neurons=n_neurons, dim=dim, scales=[(5, 1), (20, 5), (50, 10)]
    )
    classifier = MultiScaleHDCClassifier(
        n_neurons=n_neurons, n_classes=n_classes, dim=dim
    )

    # Generate synthetic spike trains with temporal structure
    # Class 0: early activity (first 30 timesteps)
    # Class 1: middle activity (timesteps 30-60)
    # Class 2: late activity (last 40 timesteps)
    torch.manual_seed(42)
    n_train = 20

    for cls in range(n_classes):
        for _ in range(n_train):
            spike_train = torch.randn(T, n_neurons) * 0.1
            if cls == 0:
                spike_train[:30] += 0.5
            elif cls == 1:
                spike_train[30:60] += 0.5
            else:
                spike_train[60:] += 0.5
            spike_train = torch.sigmoid(spike_train)

            classifier.train_step(spike_train, cls)

    classifier.finalize()

    # Test prediction
    correct = 0
    total = 60
    for cls in range(n_classes):
        for _ in range(total // n_classes):
            spike_train = torch.randn(T, n_neurons) * 0.1
            if cls == 0:
                spike_train[:30] += 0.5
            elif cls == 1:
                spike_train[30:60] += 0.5
            else:
                spike_train[60:] += 0.5
            spike_train = torch.sigmoid(spike_train)

            pred = classifier.predict(spike_train)
            if pred == cls:
                correct += 1

    accuracy = correct / total
    print(f"\n  Prediction accuracy: {accuracy:.1%}")

    # Check scale importance
    scale_importance = encoder.get_scale_importance()
    print(f"\n  Scale importance:")
    for i, (w, s) in enumerate(encoder.scales):
        print(f"    Scale {i} (window={w}, stride={s}): {scale_importance[i]:.4f}")

    print(f"\n  {'✅' if accuracy > 0.5 else '❌'} Multi-scale temporal test complete!")


if __name__ == "__main__":
    test_multiscale_temporal()
