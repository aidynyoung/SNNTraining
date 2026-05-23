"""
Multivariate Time Series Analysis for Driving Style Classification Using HDC
=============================================================================
Based on: Kang, M., et al. (2021)
"Multivariate Time Series Analysis for Driving Style Classification Using
 Hyperdimensional Computing"
 IEEE Internet of Things Journal, 9(14), 12568-12581.
 DOI: 10.1109/JIOT.2021.3138912

Key contributions:

1. **Multivariate Time Series Encoding** — Multiple sensor channels (speed,
   acceleration, steering angle, etc.) are encoded into a single hypervector
   that preserves temporal and cross-channel relationships.

2. **Channel-Specific Encoding** — Each sensor channel gets its own encoding
   hypervector, allowing the model to learn channel-specific patterns.

3. **Temporal Window Encoding** — Time windows are encoded using permutation
   to preserve temporal order within each window.

4. **Driving Style Classification** — The system classifies driving styles
   (aggressive, normal, cautious) from multivariate sensor data.

Reference:
  Kang, M., et al. (2021)
  "Multivariate Time Series Analysis for Driving Style Classification
   Using Hyperdimensional Computing"
  IEEE Internet of Things Journal, 9(14), 12568-12581
"""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Dict, Any
from hdc.hdc_glue import (
    hv_xor, hv_popcount, hv_hamming_sim, hv_bundle,
    hv_permute, hv_majority, hv_batch_sim, gen_hvs,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Section II: Channel-Specific Encoding
# ═══════════════════════════════════════════════════════════════════════════════

class ChannelEncoder:
    """
    Encodes individual sensor channels into hypervectors.

    Each channel (e.g., speed, acceleration, steering angle) gets:
    1. A unique channel hypervector (for channel identity)
    2. A quantization codebook (for value encoding)
    3. A temporal permutation scheme (for time ordering)

    The encoding for a single channel at time t is:
        hv_channel(t) = bind(channel_id, quantize(value(t)), permute^t(base))

    This preserves both the channel identity and the temporal structure.
    """

    def __init__(
        self,
        dim: int = 10000,
        n_channels: int = 6,
        n_quantization_levels: int = 32,
        seed: Optional[int] = None,
    ):
        """
        Args:
            dim: Hypervector dimensionality
            n_channels: Number of sensor channels
            n_quantization_levels: Number of discrete levels per channel
            seed: Random seed
        """
        self.dim = dim
        self.n_channels = n_channels
        self.n_quantization_levels = n_quantization_levels
        self.seed = seed or 42

        # Channel identity hypervectors
        self.channel_hvs = gen_hvs(n_channels, dim, seed=self.seed)

        # Quantization codebooks for each channel
        self.quantization_codebooks: List[torch.Tensor] = []
        for c in range(n_channels):
            codebook = gen_hvs(
                n_quantization_levels, dim, seed=self.seed + c * 100 + 1
            )
            self.quantization_codebooks.append(codebook)

        # Base hypervector for temporal permutation
        self.base_hv = gen_hvs(1, dim, seed=self.seed + 9999).squeeze(0)

    def quantize(self, value: float, channel: int) -> int:
        """Quantize a continuous value to a discrete level.

        Args:
            value: Continuous sensor value
            channel: Channel index

        Returns:
            Quantized level index
        """
        # Simple uniform quantization
        # In practice, use channel-specific min/max normalization
        normalized = (torch.tanh(torch.tensor(value)) + 1) / 2
        level = int((normalized * (self.n_quantization_levels - 1)).item())
        return max(0, min(self.n_quantization_levels - 1, level))

    def encode_channel_at_time(
        self,
        value: float,
        channel: int,
        time_idx: int,
    ) -> torch.Tensor:
        """Encode a single channel reading at a specific time.

        Args:
            value: Sensor value
            channel: Channel index
            time_idx: Time index (for permutation)

        Returns:
            (dim,) channel-time hypervector
        """
        # Quantize value
        level = self.quantize(value, channel)
        value_hv = self.quantization_codebooks[channel][level]

        # Bind with channel identity
        channel_hv = self.channel_hvs[channel]
        hv = hv_xor(channel_hv, value_hv)

        # Permute by time
        hv = hv_permute(hv, k=time_idx)

        return hv

    def encode_channel_sequence(
        self,
        values: torch.Tensor,
        channel: int,
    ) -> torch.Tensor:
        """Encode a sequence of values from one channel.

        Args:
            values: (n_timesteps,) sensor values
            channel: Channel index

        Returns:
            (dim,) channel sequence hypervector
        """
        if values.shape[0] == 0:
            return torch.zeros(self.dim)

        hvs = []
        for t in range(values.shape[0]):
            hv = self.encode_channel_at_time(
                float(values[t].item()), channel, t
            )
            hvs.append(hv)

        # Bundle all time steps
        seq_hv = hv_bundle(torch.stack(hvs))
        return hv_majority(seq_hv)


# ═══════════════════════════════════════════════════════════════════════════════
# Section III: Multivariate Time Series Encoder
# ═══════════════════════════════════════════════════════════════════════════════

class MultivariateTimeSeriesEncoder:
    """
    Encodes multivariate time series data into hypervectors.

    The encoding process:
    1. Each channel is encoded separately (preserving channel identity)
    2. Temporal order is preserved via permutation
    3. Channels are fused via bundling or binding

    For a multivariate time series X of shape (n_channels, n_timesteps):
        hv = ⊕ bind(channel_i, ⊕ permute^t(quantize(X[i, t])))

    This produces a single hypervector that captures:
    - Which channels had which values
    - When those values occurred
    - Cross-channel correlations
    """

    def __init__(
        self,
        dim: int = 10000,
        n_channels: int = 6,
        n_quantization_levels: int = 32,
        fusion: str = "bundle",
        seed: Optional[int] = None,
    ):
        """
        Args:
            dim: Hypervector dimensionality
            n_channels: Number of sensor channels
            n_quantization_levels: Number of discrete levels per channel
            fusion: "bundle" or "bind" for channel fusion
            seed: Random seed
        """
        self.dim = dim
        self.n_channels = n_channels
        self.fusion = fusion
        self.seed = seed or 42

        self.channel_encoder = ChannelEncoder(
            dim=dim,
            n_channels=n_channels,
            n_quantization_levels=n_quantization_levels,
            seed=seed,
        )

    def encode(
        self,
        data: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a multivariate time series.

        Args:
            data: (n_channels, n_timesteps) sensor data

        Returns:
            (dim,) multivariate time series hypervector
        """
        n_channels, n_timesteps = data.shape

        # Encode each channel
        channel_hvs = []
        for c in range(n_channels):
            channel_hv = self.channel_encoder.encode_channel_sequence(
                data[c], c
            )
            channel_hvs.append(channel_hv)

        if not channel_hvs:
            return torch.zeros(self.dim)

        # Fuse channels
        if self.fusion == "bind":
            # Binding: captures cross-channel correlations
            fused = channel_hvs[0]
            for hv in channel_hvs[1:]:
                fused = hv_xor(fused, hv)
        else:
            # Bundling: captures channel presence
            fused = hv_bundle(torch.stack(channel_hvs))
            fused = hv_majority(fused)

        return fused

    def encode_window(
        self,
        data: torch.Tensor,
        window_size: int = 10,
        stride: int = 5,
    ) -> torch.Tensor:
        """Encode using sliding windows.

        Each window is encoded separately, then windows are fused.
        This captures both local and global temporal patterns.

        Args:
            data: (n_channels, n_timesteps) sensor data
            window_size: Size of each temporal window
            stride: Stride between windows

        Returns:
            (dim,) windowed encoding hypervector
        """
        n_channels, n_timesteps = data.shape
        windows = []

        for start in range(0, n_timesteps - window_size + 1, stride):
            window_data = data[:, start:start + window_size]
            window_hv = self.encode(window_data)
            windows.append(window_hv)

        if not windows:
            return torch.zeros(self.dim)

        # Fuse windows
        fused = hv_bundle(torch.stack(windows))
        return hv_majority(fused)


# ═══════════════════════════════════════════════════════════════════════════════
# Section IV: Driving Style Classifier
# ═══════════════════════════════════════════════════════════════════════════════

class DrivingStyleClassifier:
    """
    Classifies driving styles from multivariate sensor data using HDC.

    Based on Kang 2021, this classifier:
    1. Encodes multivariate time series into hypervectors
    2. Stores class prototypes for each driving style
    3. Classifies by nearest-neighbor in HD space
    4. Supports incremental learning (add new trips without retraining)

    Driving styles:
    - Aggressive: rapid acceleration, hard braking, sharp turns
    - Normal: moderate driving patterns
    - Cautious: gentle acceleration, early braking, wide turns
    """

    # Typical sensor channels for driving data
    CHANNEL_NAMES = [
        "speed",           # Vehicle speed (km/h)
        "acceleration",    # Longitudinal acceleration (m/s²)
        "brake",           # Brake pedal position (%)
        "steering_angle",  # Steering wheel angle (degrees)
        "yaw_rate",        # Yaw rate (degrees/s)
        "engine_rpm",      # Engine RPM
    ]

    def __init__(
        self,
        dim: int = 10000,
        n_channels: int = 6,
        n_quantization_levels: int = 32,
        window_size: int = 10,
        stride: int = 5,
        seed: Optional[int] = None,
    ):
        """
        Args:
            dim: Hypervector dimensionality
            n_channels: Number of sensor channels
            n_quantization_levels: Number of discrete levels per channel
            window_size: Temporal window size
            stride: Window stride
            seed: Random seed
        """
        self.dim = dim
        self.seed = seed or 42

        self.encoder = MultivariateTimeSeriesEncoder(
            dim=dim,
            n_channels=n_channels,
            n_quantization_levels=n_quantization_levels,
            seed=seed,
        )

        self.window_size = window_size
        self.stride = stride

        # Class prototypes
        self.class_prototypes: Dict[str, torch.Tensor] = {}
        self.class_counts: Dict[str, int] = {}

    def add_trip(
        self,
        trip_data: torch.Tensor,
        style: str,
    ):
        """Add a labeled trip to the classifier.

        Args:
            trip_data: (n_channels, n_timesteps) sensor data
            style: Driving style label
        """
        hv = self.encoder.encode_window(
            trip_data, self.window_size, self.stride
        )

        if style in self.class_prototypes:
            count = self.class_counts[style]
            self.class_prototypes[style] = (
                (self.class_prototypes[style] * count + hv) / (count + 1)
            )
            self.class_counts[style] += 1
        else:
            self.class_prototypes[style] = hv.clone()
            self.class_counts[style] = 1

    def classify(
        self,
        trip_data: torch.Tensor,
    ) -> Tuple[str, float, Dict[str, float]]:
        """Classify a driving trip.

        Args:
            trip_data: (n_channels, n_timesteps) sensor data

        Returns:
            (predicted_style, confidence, {style: similarity})
        """
        hv = self.encoder.encode_window(
            trip_data, self.window_size, self.stride
        )

        similarities = {}
        for style, prototype in self.class_prototypes.items():
            sim = float(hv_hamming_sim(hv, prototype))
            similarities[style] = sim

        # Find best match
        best_style = max(similarities, key=similarities.get)
        confidence = similarities[best_style]

        return best_style, confidence, similarities

    def incremental_update(
        self,
        trip_data: torch.Tensor,
        true_style: str,
        lr: float = 0.1,
    ):
        """
        Online (incremental) update from a single labeled trip.

        Allows the classifier to adapt to new driving patterns without
        reprocessing the full historical dataset.  Uses a blending update:
            proto[style] = (1-lr) × proto[style] + lr × new_trip_hv

        Args:
            trip_data:  (n_channels, n_timesteps) sensor data
            true_style: Ground-truth driving style label
            lr:         Blending rate (0.1 = 10% influence from new trip)
        """
        from hdc.hdc_glue import hv_majority as _hm
        hv = self.encoder.encode_window(trip_data, self.window_size, self.stride)
        if true_style in self.class_prototypes:
            old = self.class_prototypes[true_style].float()
            blended = (1 - lr) * old + lr * hv.float()
            self.class_prototypes[true_style] = _hm(blended)
        else:
            self.add_trip(trip_data, true_style)

    def concept_drift_score(
        self,
        recent_trips: List[torch.Tensor],
        recent_labels: List[str],
        window: int = 10,
    ) -> float:
        """
        Measure concept drift: how much have recent classification errors increased?

        Reference:
            Gama et al. (2014) "A survey on concept drift adaptation"
            ACM Computing Surveys 46(4):44.

        Computes the difference between recent prediction accuracy and the
        long-run accuracy.  A positive score means recent accuracy is lower
        than expected → concept drift likely.

        Args:
            recent_trips:  Last N trips (n_channels, n_timesteps) each
            recent_labels: Ground-truth labels for recent trips
            window:        Recent window size for comparison

        Returns:
            Drift score ∈ [0, 1]; > 0.2 suggests significant drift.
        """
        if not self.class_prototypes or not recent_trips:
            return 0.0

        n = min(len(recent_trips), window)
        n_correct = 0
        for trip, true_label in zip(recent_trips[-n:], recent_labels[-n:]):
            pred_style, _, _ = self.classify(trip)
            n_correct += int(pred_style == true_label)

        recent_acc = n_correct / max(n, 1)

        # Expected accuracy at equal chance level
        expected_acc = 1.0 / max(len(self.class_prototypes), 1)

        # Drift score: how far below expected accuracy we are
        drift = max(0.0, expected_acc - recent_acc + 0.5 * (1.0 - recent_acc))
        return min(1.0, drift)

    def get_style_prototypes(self) -> Dict[str, torch.Tensor]:
        """Get the class prototypes.

        Returns:
            {style: prototype_hypervector}
        """
        return dict(self.class_prototypes)


# ═══════════════════════════════════════════════════════════════════════════════
# Section V: Feature Analysis
# ═══════════════════════════════════════════════════════════════════════════════

class DrivingFeatureAnalyzer:
    """
    Analyzes which sensor channels and time windows are most discriminative
    for driving style classification.

    Provides:
    1. Channel importance scores
    2. Temporal importance scores
    3. Cross-channel correlation analysis
    4. Style-specific pattern discovery
    """

    def __init__(
        self,
        dim: int = 10000,
        n_channels: int = 6,
        seed: Optional[int] = None,
    ):
        self.dim = dim
        self.n_channels = n_channels
        self.seed = seed or 42

        self.channel_encoder = ChannelEncoder(
            dim=dim, n_channels=n_channels, seed=seed
        )

    def channel_importance(
        self,
        classifier: DrivingStyleClassifier,
        trip_data: torch.Tensor,
    ) -> Dict[str, float]:
        """Compute importance of each channel for classification.

        Args:
            classifier: Trained DrivingStyleClassifier
            trip_data: (n_channels, n_timesteps) sensor data

        Returns:
            {channel_name: importance_score}
        """
        n_channels, n_timesteps = trip_data.shape
        base_style, _, _ = classifier.classify(trip_data)

        importances = {}
        for c in range(n_channels):
            # Perturb this channel
            perturbed = trip_data.clone()
            perturbed[c] = torch.randn(n_timesteps) * 0.5

            new_style, _, _ = classifier.classify(perturbed)
            # Importance = 1 if classification changed, else 0
            importances[classifier.CHANNEL_NAMES[c]] = 1.0 if new_style != base_style else 0.0

        return importances

    def temporal_importance(
        self,
        classifier: DrivingStyleClassifier,
        trip_data: torch.Tensor,
    ) -> torch.Tensor:
        """Compute importance of each time window.

        Args:
            classifier: Trained DrivingStyleClassifier
            trip_data: (n_channels, n_timesteps) sensor data

        Returns:
            (n_windows,) importance scores
        """
        n_channels, n_timesteps = trip_data.shape
        base_style, _, _ = classifier.classify(trip_data)

        n_windows = max(1, (n_timesteps - classifier.window_size) // classifier.stride + 1)
        importances = torch.zeros(n_windows)

        for w in range(n_windows):
            start = w * classifier.stride
            end = start + classifier.window_size

            # Zero out this window
            perturbed = trip_data.clone()
            perturbed[:, start:end] = 0

            new_style, _, _ = classifier.classify(perturbed)
            importances[w] = 1.0 if new_style != base_style else 0.0

        return importances


# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_channel_encoder():
    """Verify channel encoding."""
    print("=" * 60)
    print("Testing Channel Encoding (Kang 2021)")
    print("=" * 60)

    dim = 1000
    encoder = ChannelEncoder(dim=dim, n_channels=3)

    # Encode a single reading
    hv = encoder.encode_channel_at_time(0.5, channel=0, time_idx=0)
    print(f"  Single reading HV shape: {hv.shape}")

    # Encode a sequence
    values = torch.tensor([0.1, 0.5, 0.9, 0.3, 0.7])
    seq_hv = encoder.encode_channel_sequence(values, channel=0)
    print(f"  Channel sequence HV shape: {seq_hv.shape}")

    print(f"  ✅ Channel encoding test complete!")


def test_multivariate_encoder():
    """Verify multivariate time series encoding."""
    print("=" * 60)
    print("Testing Multivariate Time Series Encoding (Kang 2021)")
    print("=" * 60)

    dim = 1000
    encoder = MultivariateTimeSeriesEncoder(dim=dim, n_channels=3)

    # Create synthetic multivariate data
    data = torch.randn(3, 20)  # 3 channels, 20 timesteps
    hv = encoder.encode(data)
    print(f"  Multivariate HV shape: {hv.shape}")

    # Windowed encoding
    hv_windowed = encoder.encode_window(data, window_size=5, stride=2)
    print(f"  Windowed HV shape: {hv_windowed.shape}")

    print(f"  ✅ Multivariate encoding test complete!")


def test_driving_classifier():
    """Verify driving style classification."""
    print("=" * 60)
    print("Testing Driving Style Classifier (Kang 2021)")
    print("=" * 60)

    dim = 1000
    classifier = DrivingStyleClassifier(dim=dim, n_channels=3, window_size=5, stride=2)

    # Create synthetic driving data for each style
    torch.manual_seed(42)

    # Aggressive: high variance, rapid changes
    for _ in range(5):
        data = torch.randn(3, 20) * 2.0
        classifier.add_trip(data, "aggressive")

    # Normal: moderate variance
    for _ in range(5):
        data = torch.randn(3, 20) * 1.0
        classifier.add_trip(data, "normal")

    # Cautious: low variance, smooth
    for _ in range(5):
        data = torch.randn(3, 20) * 0.5
        classifier.add_trip(data, "cautious")

    # Test classification
    test_data = torch.randn(3, 20) * 1.5
    style, confidence, sims = classifier.classify(test_data)
    print(f"\n  Predicted style: {style}")
    print(f"  Confidence: {confidence:.4f}")
    print(f"  All similarities: {sims}")

    print(f"\n  ✅ Driving style classifier test complete!")


if __name__ == "__main__":
    test_channel_encoder()
    print()
    test_multivariate_encoder()
    print()
    test_driving_classifier()
