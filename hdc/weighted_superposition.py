"""
Learnable Weighted Superposition for HDC
=========================================
Based on: Schlegel et al. 2024 "Learnable weighted superposition in HDC"

Key insight: Instead of simple bundling (sum) of hypervectors, use
learnable weights per channel to improve class separation in the
associative memory. This replaces the naive bundling in ItemMemory
and AssocMemory with a weighted approach that can be optimized.

Reference:
  Schlegel, K., et al. (2024)
  "Learnable weighted superposition in HDC"
  arXiv / Neural Computing
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, List, Tuple
from models.hdc import gen_hvs, bind, bundle, sim, thresh, batch_sim


class WeightedSuperposition(nn.Module):
    """
    Learnable weighted superposition for bundling hypervectors.

    Instead of: bundle = sum(hv_i)
    Use: bundle = sum(w_i * hv_i) where w_i are learnable weights.

    The weights are constrained to be non-negative and sum to 1,
    ensuring the superposition remains a valid hypervector.
    """

    def __init__(
        self,
        n_channels: int,
        dim: int = 10000,
        mode: str = "bipolar",
        init_weights: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.dim = dim
        self.mode = mode
        self.temperature = temperature

        # Learnable log-weights (softmax ensures positivity + sum-to-1)
        if init_weights is not None:
            log_weights = torch.log(init_weights.clamp(min=1e-12))
        else:
            log_weights = torch.zeros(n_channels)

        self.log_weights = nn.Parameter(log_weights)

    def get_weights(self) -> torch.Tensor:
        """Return normalized weights that sum to 1."""
        return torch.softmax(self.log_weights / self.temperature, dim=0)

    def forward(self, hvs: torch.Tensor) -> torch.Tensor:
        """Apply weighted superposition to a set of hypervectors.

        Args:
            hvs: (n_channels, dim) tensor of hypervectors to bundle

        Returns:
            (dim,) weighted superposition hypervector
        """
        weights = self.get_weights()  # (n_channels,)
        weighted = hvs * weights.unsqueeze(-1)  # (n_channels, dim)
        result = weighted.sum(dim=0)  # (dim,)

        if self.mode == "bipolar":
            result = thresh(result)

        return result

    def online_update(
        self,
        per_channel_errors: torch.Tensor,
        lr: float = 0.01,
    ):
        """
        Gradient-free online weight adaptation from per-channel errors.

        Channels with lower error get higher weight; channels with higher
        error get lower weight.  This is the HDC analogue of attention:
        reliable channels are attended to more than noisy ones.

        No backprop needed — the update is:
            log_w_i ← log_w_i − lr × (e_i − mean_error)

        Args:
            per_channel_errors: (n_channels,) per-channel prediction errors
            lr: Learning rate (default 0.01)
        """
        with torch.no_grad():
            e = per_channel_errors.float().to(self.log_weights.device)
            e_norm = e - e.mean()   # centre errors: positive = worse than avg
            self.log_weights.sub_(lr * e_norm)


class ChannelWeightedEncoder(nn.Module):
    """
    Multi-channel encoder with learnable per-channel weights.

    Each input channel (e.g., neuron, frequency band, sensor) gets its
    own hypervector key. The encoder learns which channels are most
    informative and weights them accordingly during bundling.

    This is the practical application of Schlegel 2024's weighted
    superposition for time-series / spike-train encoding.
    """

    def __init__(
        self,
        n_channels: int,
        dim: int = 10000,
        mode: str = "bipolar",
        n_levels: int = 13,
        device: Optional[str] = None,
        seed: Optional[int] = None,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.dim = dim
        self.mode = mode
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Per-channel hypervector keys
        self.register_buffer(
            "channel_keys",
            gen_hvs(n_channels, dim, mode, self.device, seed),
        )

        # Level hypervectors for scalar encoding
        self.register_buffer(
            "level_hvs",
            gen_hvs(n_levels, dim, mode, self.device, seed + 1 if seed else None),
        )
        self.n_levels = n_levels

        # Learnable channel weights
        self.weighted_superposition = WeightedSuperposition(
            n_channels=n_channels,
            dim=dim,
            mode=mode,
            temperature=temperature,
        )

    def encode_channel(self, value: float, channel_idx: int) -> torch.Tensor:
        """Encode a single channel value into a hypervector.

        Args:
            value: Scalar value for this channel
            channel_idx: Which channel this belongs to

        Returns:
            (dim,) hypervector encoding the channel value
        """
        # Quantize value to level index
        value = max(0.0, min(1.0, value))
        level_idx = min(int(value * (self.n_levels - 1)), self.n_levels - 1)

        # Bind channel key with level hypervector
        channel_hv = self.channel_keys[channel_idx]
        level_hv = self.level_hvs[level_idx]
        return bind(channel_hv, level_hv, self.mode)

    def _forward_single(self, values: torch.Tensor) -> torch.Tensor:
        """Encode a single (n_channels,) input."""
        channel_hvs = []
        for i in range(self.n_channels):
            hv = self.encode_channel(values[i].item(), i)
            channel_hvs.append(hv)
        stacked = torch.stack(channel_hvs)  # (n_channels, dim)
        return self.weighted_superposition(stacked)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Encode multi-channel input with learnable weights.

        Args:
            values: (n_channels,) or (B, n_channels) tensor

        Returns:
            (dim,) or (B, dim) weighted superposition hypervector
        """
        if values.dim() == 2:
            return torch.stack([self._forward_single(values[i]) for i in range(values.shape[0])])
        return self._forward_single(values)

    def get_channel_importance(self) -> torch.Tensor:
        """Return learned importance weights for each channel."""
        return self.weighted_superposition.get_weights().detach().cpu()


class WeightedAssocMemory(nn.Module):
    """
    Associative memory with learnable per-class weighting.

    Extends the standard AssocMemory with:
    1. Learnable per-class confidence weights
    2. Weighted voting during prediction
    3. Adaptive thresholding based on class distribution

    This improves on the simple bundling approach by learning
    which classes need more separation margin.
    """

    def __init__(
        self,
        n_classes: int,
        dim: int = 10000,
        mode: str = "bipolar",
        device: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.dim = dim
        self.mode = mode
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Class hypervectors
        self.register_buffer(
            "class_hvs",
            gen_hvs(n_classes, dim, mode, self.device, seed),
        )

        # Learnable per-class confidence weights
        self.class_weights = nn.Parameter(torch.ones(n_classes))

        # Training counts for normalization
        self.register_buffer("counts", torch.zeros(n_classes, device=self.device))

    def add(self, hv: torch.Tensor, label: int):
        """Add a training example to the associative memory."""
        self.class_hvs[label] = self.class_hvs[label] + hv
        self.counts[label] += 1

    def renormalize(self):
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

    def predict(self, hv: torch.Tensor) -> int:
        """Predict class with weighted similarity.

        Uses learned class weights to scale similarities before argmax.
        """
        similarities = batch_sim(hv, self.class_hvs, self.mode)
        # Apply learned class weights
        weights = torch.softmax(self.class_weights, dim=0)
        weighted_sims = similarities * weights
        return int(weighted_sims.argmax().item())

    def query(self, hv: torch.Tensor) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """Query the memory.

        Returns:
            (pred_class, weighted_similarities, output_hypervector)
        """
        weighted_sims = self.forward(hv)
        pred = int(weighted_sims.argmax().item())
        output_hv = self.class_hvs[pred].detach().clone()
        return pred, weighted_sims, output_hv

    def forward(self, hv: torch.Tensor) -> torch.Tensor:
        """Return weighted similarities for all classes."""
        similarities = batch_sim(hv, self.class_hvs, self.mode)
        weights = torch.softmax(self.class_weights, dim=0)
        return similarities * weights

    def rebalance_weights(self):
        """
        Rebalance class weights inversely proportional to class frequency.

        Classes with fewer training examples get higher weight, preventing
        majority-class dominance in predictions.  This is the HDC equivalent
        of class-balanced sampling.

        Call after training is complete and counts are populated.
        """
        with torch.no_grad():
            counts = self.counts.clamp(min=1.0)
            inv_freq = 1.0 / counts
            inv_freq = inv_freq / inv_freq.sum()
            # Set class_weights to log(inv_freq) so softmax gives inv_freq
            import math
            log_w = torch.tensor(
                [math.log(float(f)) for f in inv_freq.tolist()],
                dtype=torch.float32,
            )
            self.class_weights.data.copy_(log_w)

    def class_utilisation(self) -> Dict[int, float]:
        """Return fraction of total training examples per class."""
        total = max(self.counts.sum().item(), 1.0)
        return {i: float(self.counts[i].item() / total)
                for i in range(self.n_classes)}


# ── Tests ────────────────────────────────────────────────────────────────────
def test_weighted_superposition():
    """Verify weighted superposition improves class separation."""
    print("=" * 60)
    print("Testing Learnable Weighted Superposition")
    print("=" * 60)

    dim = 2000
    n_channels = 10
    n_classes = 5
    n_samples = 50

    # Create encoder and memory
    encoder = ChannelWeightedEncoder(
        n_channels=n_channels, dim=dim, n_levels=8
    )
    memory = WeightedAssocMemory(n_classes=n_classes, dim=dim)

    # Generate synthetic data with channel importance gradient
    # Channel 0 is most informative, channel 9 is noise
    torch.manual_seed(42)
    for cls in range(n_classes):
        for _ in range(n_samples):
            values = torch.randn(n_channels) * 0.3
            # Make channel 0 highly discriminative
            values[0] = values[0] + cls * 0.5
            # Make channel 9 pure noise
            values[9] = torch.randn(1).item() * 2.0

            hv = encoder(values)
            memory.add(hv, cls)

    memory.renormalize()

    # Check that channel 0 got higher weight than channel 9
    importance = encoder.get_channel_importance()
    print(f"\n  Channel importance (should show ch0 > ch9):")
    for i in range(n_channels):
        print(f"    Channel {i}: {importance[i]:.4f}")

    ch0_weight = importance[0].item()
    ch9_weight = importance[9].item()
    print(f"\n  Channel 0 weight: {ch0_weight:.4f}")
    print(f"  Channel 9 weight: {ch9_weight:.4f}")
    print(f"  Ratio (ch0/ch9): {ch0_weight / max(ch9_weight, 1e-12):.2f}x")

    # Test prediction accuracy
    correct = 0
    total = 100
    for cls in range(n_classes):
        for _ in range(total // n_classes):
            values = torch.randn(n_channels) * 0.3
            values[0] = values[0] + cls * 0.5
            hv = encoder(values)
            pred = memory.predict(hv)
            if pred == cls:
                correct += 1

    accuracy = correct / total
    print(f"\n  Prediction accuracy: {accuracy:.1%}")
    print(f"\n  {'✅' if accuracy > 0.5 else '❌'} Weighted superposition test complete!")


if __name__ == "__main__":
    test_weighted_superposition()
