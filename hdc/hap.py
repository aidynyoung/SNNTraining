"""Hyperdimensional Active Perception (HAP) for SNNTraining.
=====================================================
Based on Section V-A of Amrouch et al. 2022:
"Learning Sensorimotor Control with Neuromorphic Sensors:
 Toward Hyperdimensional Active Perception"

Key insight: An autonomous agent's movements generate perception
in the form of event-camera "time slices". These are encoded into
binary hypervectors, aggregated into a sequence bundle, and bound
to action hypervectors for real-time sensorimotor learning.

Reference:
  Mitrokhin, Sutor, Fermuller, Aloimonos (2019)
  "Learning Sensorimotor Control with Neuromorphic Sensors"
  Science Robotics, vol. 4, no. 30
"""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple
from models.hdc import gen_hvs, bind, bundle, sim, thresh, batch_sim


class TimeSliceEncoder(nn.Module):
    """
    Encodes event-camera time slices into hypervectors.

    Each time slice is a 2D frame where pixels encode event activity
    over a short temporal window. The encoder maps intensity values
    at each (x, y) position to hypervectors via binding with position keys.

    Pipeline (from paper Section V-A):
        time_slice → encode each pixel (intensity + position keys)
        → bundle all pixel HVs → one slice HV
    """

    def __init__(
        self,
        height: int,
        width: int,
        dim: int = 10000,
        n_intensity_levels: int = 256,
        mode: str = "bipolar",
        device: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.height = height
        self.width = width
        self.dim = dim
        self.mode = mode
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Position keys: one hypervector per pixel
        n_positions = height * width
        self.register_buffer(
            "pos_keys",
            gen_hvs(n_positions, dim, mode, self.device, seed)
        )

        # Intensity keys: evenly spaced across intensity range
        self.register_buffer(
            "intensity_keys",
            gen_hvs(n_intensity_levels, dim, mode, self.device,
                    seed + 1 if seed is not None else None)
        )
        self.n_intensity_levels = n_intensity_levels

    def encode_slice(self, time_slice: torch.Tensor) -> torch.Tensor:
        """Encode a single time slice into a hypervector.

        Uses efficient vectorized operations instead of pixel-by-pixel loops.
        Based on Mitrokhin, Sutor et al. 2019 "Learning Sensorimotor Control
        with Neuromorphic Sensors" - proper time-slice encoding with
        position-intensity binding.

        Args:
            time_slice: (H, W) frame of event activity

        Returns:
            (dim,) hypervector
        """
        H, W = time_slice.shape
        # Normalize to [0, 1]
        mn, mx = time_slice.min(), time_slice.max()
        if mx - mn < 1e-8:
            normalized = torch.zeros_like(time_slice)
        else:
            normalized = (time_slice - mn) / (mx - mn + 1e-12)

        # Vectorized encoding: process all pixels at once
        flat = normalized.flatten()  # (H*W,)
        n_pixels = flat.shape[0]

        # Quantize intensities to level indices
        i_indices = (flat * (self.n_intensity_levels - 1)).long().clamp(
            0, self.n_intensity_levels - 1
        )

        # Bind position keys with intensity keys for all pixels
        # pixel_hv[i] = bind(pos_keys[i], intensity_keys[i_idx[i]])
        pos_hvs = self.pos_keys[:n_pixels]  # (n_pixels, dim)
        int_hvs = self.intensity_keys[i_indices]  # (n_pixels, dim)

        if self.mode == "bipolar":
            pixel_hvs = pos_hvs * int_hvs  # bind via element-wise multiply
        else:
            pixel_hvs = (pos_hvs + int_hvs) % 2  # binary XOR

        # Bundle all pixel HVs via sum
        hv = pixel_hvs.sum(dim=0)  # (dim,)

        if self.mode == "bipolar":
            hv = thresh(hv)
        return hv

    def encode_sequence(
        self,
        time_slices: torch.Tensor,
        window_size: int = 5,
    ) -> torch.Tensor:
        """Encode a sequence of time slices via bundling.

        Aggregates window_size slices into one "moment" hypervector,
        as described in the paper: bundling N consecutive slices
        captures a moment in time that the agent is perceiving.

        Args:
            time_slices: (T, H, W) sequence of time slices
            window_size: Number of slices to bundle together

        Returns:
            (num_windows, dim) sequence hypervectors
        """
        T = time_slices.shape[0]
        windows = []

        for t in range(0, T - window_size + 1, window_size):
            # Bundle window_size slices
            window_hv = torch.zeros(self.dim, device=self.device)
            for w in range(window_size):
                window_hv = window_hv + self.encode_slice(time_slices[t + w])

            if self.mode == "bipolar":
                window_hv = thresh(window_hv)
            windows.append(window_hv)

        return torch.stack(windows) if windows else torch.zeros(0, self.dim)

    def forward(self, time_slice: torch.Tensor) -> torch.Tensor:
        return self.encode_slice(time_slice)


class HyperdimensionalActivePerception(nn.Module):
    """
    HAP: Learns sensorimotor control by binding time-slice bundles
    to action (ego-motion) hypervectors.

    Training:
        For each (sequence_bundle, velocity) pair:
            M += bind(sequence_hv, velocity_hv)

    Inference:
        Given a new sequence_hv, probe M:
            recovered = M ⊗ sequence_hv
            predict closest velocity_hv

    From the paper: "The small size and rapid inference of the HAP model
    allows for multiple models to be formed."
    """

    def __init__(
        self,
        n_velocity_bins: int,
        vel_range: Tuple[float, float],
        dim: int = 10000,
        mode: str = "bipolar",
        device: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.n_velocity_bins = n_velocity_bins
        self.vel_range = vel_range
        self.dim = dim
        self.mode = mode
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Velocity hypervectors (one per discrete velocity bin)
        self.register_buffer(
            "velocity_hvs",
            gen_hvs(n_velocity_bins, dim, mode, self.device, seed)
        )

        # Memory hypervector (HIL — Hyperdimensional Inference Layer)
        self.register_buffer(
            "memory",
            torch.zeros(dim, device=torch.device(self.device))
        )
        self.register_buffer("count", torch.tensor(0, device=torch.device(self.device)))

    def discretize_velocity(self, velocity: torch.Tensor) -> int:
        """Map continuous 3D velocity to discrete bin index.

        Args:
            velocity: (3,) tensor [vx, vy, vz]

        Returns:
            bin index
        """
        # Use magnitude for discretization (simplified from paper)
        mag = velocity.norm().item()
        lo, hi = self.vel_range
        bin_idx = min(
            int((mag - lo) / (hi - lo + 1e-12) * self.n_velocity_bins),
            self.n_velocity_bins - 1
        )
        return max(0, bin_idx)

    def train_pair(self, sequence_hv: torch.Tensor, velocity: torch.Tensor):
        """Add one (sequence, velocity) association to memory.

        Args:
            sequence_hv: (dim,) bundled sequence hypervector
            velocity: (3,) velocity vector
        """
        vel_idx = self.discretize_velocity(velocity)
        vel_hv = self.velocity_hvs[vel_idx]

        association = bind(
            sequence_hv.to(self.device),
            vel_hv,
            self.mode
        )
        self.memory += association
        self.count += 1

    def train_batch(
        self,
        sequence_hvs: torch.Tensor,
        velocities: torch.Tensor,
    ):
        """Train on a batch of (sequence, velocity) pairs.

        Args:
            sequence_hvs: (B, dim) bundled sequence hypervectors
            velocities: (B, 3) velocity vectors
        """
        for i in range(sequence_hvs.shape[0]):
            self.train_pair(sequence_hvs[i], velocities[i])

    def predict(self, sequence_hv: torch.Tensor) -> Tuple[int, torch.Tensor]:
        """Predict velocity bin from sequence hypervector.

        Args:
            sequence_hv: (dim,) query hypervector

        Returns:
            (velocity_bin_idx, similarity_scores)
        """
        # Unbind: memory ⊗ query → velocity hv
        recovered = bind(
            self.memory,
            sequence_hv.to(self.device),
            self.mode
        )

        # Find closest velocity hypervector
        similarities = batch_sim(recovered, self.velocity_hvs, self.mode)
        return int(similarities.argmax().item()), similarities

    def get_velocity(self, vel_bin: int) -> torch.Tensor:
        """Get the representative velocity vector for a bin."""
        lo, hi = self.vel_range
        mag = lo + (hi - lo) * vel_bin / max(1, self.n_velocity_bins - 1)
        # Return unit vector scaled by magnitude (direction is learned)
        return torch.tensor([mag, 0.0, 0.0], device=self.device)

    def forward(
        self,
        sequence_hv: torch.Tensor,
    ) -> torch.Tensor:
        """Batch prediction.

        Args:
            sequence_hv: (B, dim) query hypervectors

        Returns:
            (B, 3) predicted velocity vectors
        """
        results = []
        for i in range(sequence_hv.shape[0]):
            vel_bin, _ = self.predict(sequence_hv[i])
            results.append(self.get_velocity(vel_bin))
        return torch.stack(results)


class MultiHAP(nn.Module):
    """
    Multiple HAP models for consensus-based sensorimotor control.

    As described in the paper: "the small size and rapid inference
    of the HAP model allows for multiple models to be formed."

    Multiple HAP instances can be trained on different sensor modalities
    or time windows, then their predictions are combined via consensus.
    """

    def __init__(
        self,
        n_models: int,
        n_velocity_bins: int,
        vel_range: Tuple[float, float],
        dim: int = 10000,
        mode: str = "bipolar",
        device: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.n_models = n_models

        self.models = nn.ModuleList([
            HyperdimensionalActivePerception(
                n_velocity_bins=n_velocity_bins,
                vel_range=vel_range,
                dim=dim,
                mode=mode,
                device=device,
                seed=seed + i if seed is not None else None,
            )
            for i in range(n_models)
        ])

    def train_all(
        self,
        sequence_hvs_list: List[torch.Tensor],
        velocities: torch.Tensor,
    ):
        """Train each model with its respective inputs.

        Args:
            sequence_hvs_list: List of (B, dim) per model
            velocities: (B, 3) ground truth
        """
        for model, hvs in zip(self.models, sequence_hvs_list):
            model.train_batch(hvs, velocities)

    def predict_consensus(
        self,
        sequence_hvs_list: List[torch.Tensor],
    ) -> torch.Tensor:
        """Consensus prediction from all models.

        Args:
            sequence_hvs_list: List of (B, dim) per model

        Returns:
            (B, 3) consensus velocity predictions
        """
        # Get predictions from all models
        all_preds = []
        for model, hvs in zip(self.models, sequence_hvs_list):
            all_preds.append(model(hvs))

        # Average predictions (simple consensus)
        stacked = torch.stack(all_preds)  # (n_models, B, 3)
        return stacked.mean(dim=0)


# ── Tests ────────────────────────────────────────────────────────────────────
def test_hap():
    """Verify HAP encoding and learning pipeline."""
    print("=" * 60)
    print("Testing Hyperdimensional Active Perception (HAP)")
    print("=" * 60)

    # Simulate event-camera time slices
    H, W = 32, 32  # Small for testing
    T = 20
    dim = 2000
    time_slices = torch.rand(T, H, W)  # Random event frames

    encoder = TimeSliceEncoder(
        height=H, width=W, dim=dim, n_intensity_levels=16
    )
    encoder.eval()

    # Test single slice encoding
    with torch.no_grad():
        slice_hv = encoder.encode_slice(time_slices[0])
    print(f"\n  Single slice HV shape: {slice_hv.shape}")
    print(f"  Non-zero dims: {(slice_hv != 0).sum().item()} / {dim}")

    # Test sequence encoding with window
    with torch.no_grad():
        seq_hvs = encoder.encode_sequence(time_slices, window_size=5)
    print(f"\n  Sequence HVs shape: {seq_hvs.shape}  (expected: [{T//5}, {dim}])")

    # Test HAP learning
    hap = HyperdimensionalActivePerception(
        n_velocity_bins=8,
        vel_range=(0.0, 10.0),
        dim=dim,
    )
    hap.eval()

    # Generate random velocities and train
    B = seq_hvs.shape[0]
    velocities = torch.rand(B, 3) * 5 + 2  # Random velocities in [2, 7]

    with torch.no_grad():
        hap.train_batch(seq_hvs, velocities)

    # Predict
    with torch.no_grad():
        vel_bin, sims = hap.predict(seq_hvs[0])
        pred_vel = hap(seq_hvs)

    print(f"\n  Predicted velocity bin: {vel_bin}")
    print(f"  Top-3 similarities: {sims.topk(min(3, len(sims))).values.tolist()}")
    print(f"  Batch predictions shape: {pred_vel.shape}")

    # Test MultiHAP
    multi = MultiHAP(n_models=3, n_velocity_bins=8, vel_range=(0.0, 10.0), dim=dim)
    multi.eval()

    with torch.no_grad():
        multi.train_all(
            [seq_hvs, seq_hvs, seq_hvs],  # Same inputs for testing
            velocities
        )
        consensus = multi.predict_consensus([seq_hvs, seq_hvs, seq_hvs])

    print(f"\n  Consensus predictions shape: {consensus.shape}")
    print(f"  Mean velocity magnitude: {consensus.norm(dim=1).mean():.2f}")

    print("\n✅ HAP test complete!")


if __name__ == "__main__":
    test_hap()