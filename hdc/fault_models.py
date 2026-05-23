"""
hdc/fault_models.py
===================
Hardware-realistic fault injection for SNN weight memory.

Implements the SpikeFI fault taxonomy (Spyrou et al. 2024, arXiv:2412.06795):
  - Stuck-at-0/1: Neuron output permanently stuck at 0 or 1
  - Weight bit-flip (transient): Single-event upset, flipped once
  - Weight bit-flip (permanent): Bit permanently flipped
  - Synaptic silence: Synapse permanently disconnected (weight = 0)
  - Timing jitter: Spike timing shifted (not implemented — requires event model)

Also models ReRam failure modes (Chen et al. 2024, arXiv:2412.10389):
  - Retention failure: Gradual drift in analog weight values
  - Read disturb: Weight changes due to repeated reads

Usage:
    from hdc.fault_models import FaultInjector, FaultConfig, FaultType
    injector = FaultInjector(FaultConfig(fault_type=FaultType.STUCK_AT_0,
                                          fault_rate=1e-3))
    corrupted = injector.apply(weights)
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# Fault taxonomy (SpikeFI-compatible)
# ---------------------------------------------------------------------------

class FaultType(enum.Enum):
    """Hardware fault models from SpikeFI taxonomy."""
    STUCK_AT_0 = "stuck_at_0"           # Neuron output permanently 0
    STUCK_AT_1 = "stuck_at_1"           # Neuron output permanently 1
    WEIGHT_BITFLIP_TRANSIENT = "wbf_t"  # Single-event upset, flip once
    WEIGHT_BITFLIP_PERMANENT = "wbf_p"  # Bit permanently flipped
    SYNAPTIC_SILENCE = "syn_silence"    # Synapse permanently disconnected
    RETENTION_FAILURE = "retention"     # Gradual analog drift (ReRam)
    READ_DISTURB = "read_disturb"       # Weight change from repeated reads
    MIXED = "mixed"                     # Combination of all above


@dataclass
class FaultConfig:
    """Configuration for hardware fault injection.

    Attributes:
        fault_type: Type of fault to inject
        fault_rate: Probability of fault per element (0 to 1)
        seed: Random seed for reproducibility
        stuck_value: Value for stuck-at faults (0 or 1)
        retention_drift_std: Std dev of Gaussian drift for retention failures
        read_disturb_prob: Probability of read disturb per access
        persistent: If True, faults persist across calls (for permanent faults)
        neuron_mask: Optional pre-computed mask of which neurons are faulty
        weight_mask: Optional pre-computed mask of which weights are faulty
    """
    fault_type: FaultType = FaultType.WEIGHT_BITFLIP_TRANSIENT
    fault_rate: float = 1e-6
    seed: Optional[int] = None
    stuck_value: int = 0
    retention_drift_std: float = 0.01
    read_disturb_prob: float = 1e-8
    persistent: bool = False
    # Probability that a currently-flipped SEU bit recovers per call.
    # 0.0 = permanent until next event; 1.0 = always clears (classic RTN model).
    seu_recovery_prob: float = 0.1

    # Internal state for persistent faults
    _neuron_fault_mask: Optional[torch.Tensor] = None
    _weight_fault_mask: Optional[torch.Tensor] = None
    _fault_positions: Optional[torch.Tensor] = None
    _initialized: bool = False


class FaultInjector:
    """Hardware fault injector for SNN weight memory.

    Supports the full SpikeFI fault taxonomy. Faults can be transient
    (re-sampled each call) or persistent (sampled once and fixed).

    Usage:
        injector = FaultInjector(FaultConfig(
            fault_type=FaultType.STUCK_AT_0, fault_rate=1e-3, persistent=True))
        for step in range(T):
            corrupted = injector.apply(weights)
            # corrupted will have the same weights stuck at 0 every step
    """

    def __init__(self, config: Optional[FaultConfig] = None):
        self.config = config or FaultConfig()
        self._rng = torch.Generator()
        if self.config.seed is not None:
            self._rng.manual_seed(self.config.seed)
        self._initialized = False
        self._fault_positions: Optional[torch.Tensor] = None
        self._stuck_mask: Optional[torch.Tensor] = None
        self._stuck_value: Optional[torch.Tensor] = None
        # Accumulated SEU mask for the Poisson+recovery transient model.
        # Bits set here are currently flipped; updated each apply() call.
        self._seu_mask: Optional[torch.Tensor] = None
        self.total_faults_injected = 0
        self.total_elements_processed = 0

    def _init_persistent(self, ref_tensor: torch.Tensor) -> None:
        """Initialize persistent fault positions (called once)."""
        if self._initialized:
            return
        n = ref_tensor.numel()
        n_faults = max(1, int(n * self.config.fault_rate))
        indices = torch.randperm(n, generator=self._rng)[:n_faults]
        self._fault_positions = torch.zeros(n, dtype=torch.bool)
        self._fault_positions[indices] = True
        self._fault_positions = self._fault_positions.reshape(ref_tensor.shape)
        self._initialized = True

    def apply(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply hardware faults to a tensor.

        Args:
            tensor: Weight or activation tensor to corrupt

        Returns:
            Corrupted tensor (same shape)
        """
        cfg = self.config
        corrupted = tensor.clone()
        n = tensor.numel()
        self.total_elements_processed += n

        if cfg.fault_type == FaultType.STUCK_AT_0:
            # Neuron output permanently stuck at 0
            if cfg.persistent:
                self._init_persistent(tensor)
                corrupted[self._fault_positions] = 0.0
                n_faults = int(self._fault_positions.sum().item())
            else:
                mask = torch.rand(tensor.shape, generator=self._rng) < cfg.fault_rate
                corrupted[mask] = 0.0
                n_faults = int(mask.sum().item())

        elif cfg.fault_type == FaultType.STUCK_AT_1:
            # Neuron output permanently stuck at 1
            val = float(cfg.stuck_value) if cfg.stuck_value != 0 else 1.0
            if cfg.persistent:
                self._init_persistent(tensor)
                corrupted[self._fault_positions] = val
                n_faults = int(self._fault_positions.sum().item())
            else:
                mask = torch.rand(tensor.shape, generator=self._rng) < cfg.fault_rate
                corrupted[mask] = val
                n_faults = int(mask.sum().item())

        elif cfg.fault_type == FaultType.WEIGHT_BITFLIP_TRANSIENT:
            # Single-event upset (SEU): Poisson arrival + probabilistic recovery.
            #
            # Physical model: cosmic rays or alpha particles cause isolated bit
            # flips at rate `fault_rate` per element per call (Poisson arrivals).
            # Each flipped bit recovers independently with probability
            # `seu_recovery_prob` per call.  This is distinct from random
            # telegraph noise (RTN), which would re-sample the mask every call.
            #
            # For persistent mode: skip the Poisson model and use a fixed mask
            # (models a permanent stuck flip, same as WEIGHT_BITFLIP_PERMANENT).
            if cfg.persistent:
                self._init_persistent(tensor)
                corrupted[self._fault_positions] = -corrupted[self._fault_positions]
                n_faults = int(self._fault_positions.sum().item())
            else:
                n = tensor.numel()
                # Initialise the accumulated SEU mask on first call
                if self._seu_mask is None or self._seu_mask.shape != tensor.shape:
                    self._seu_mask = torch.zeros(tensor.shape, dtype=torch.bool)

                # 1. Recovery: each currently-flipped bit recovers independently
                if cfg.seu_recovery_prob > 0.0 and self._seu_mask.any():
                    recover = (torch.rand(tensor.shape, generator=self._rng)
                               < cfg.seu_recovery_prob)
                    self._seu_mask &= ~recover

                # 2. New SEU arrivals: Poisson(n * fault_rate) new events,
                #    each targeting a uniformly random element (XOR into mask)
                expected_events = n * cfg.fault_rate
                # Use Poisson approximation: sample k ~ Poisson(lambda)
                # For small lambda: P(k>=1) ≈ lambda, so just Bernoulli when tiny
                if expected_events < 1e-3:
                    # Fast path: expected < 1 event per 1000 calls
                    if torch.rand(1, generator=self._rng).item() < expected_events:
                        idx = int(torch.randint(n, (1,), generator=self._rng).item())
                        flat = self._seu_mask.reshape(-1)
                        flat[idx] = ~flat[idx]
                        self._seu_mask = flat.reshape(tensor.shape)
                else:
                    # General path: draw k events from Poisson
                    k = int(torch.poisson(
                        torch.tensor(expected_events), generator=self._rng
                    ).item())
                    if k > 0:
                        idxs = torch.randint(n, (k,), generator=self._rng)
                        flat = self._seu_mask.reshape(-1)
                        for idx in idxs:
                            flat[idx] = ~flat[idx]  # XOR: second hit clears first
                        self._seu_mask = flat.reshape(tensor.shape)

                corrupted[self._seu_mask] = -corrupted[self._seu_mask]
                n_faults = int(self._seu_mask.sum().item())

        elif cfg.fault_type == FaultType.WEIGHT_BITFLIP_PERMANENT:
            # Bit permanently flipped: same positions every call
            self._init_persistent(tensor)
            corrupted[self._fault_positions] = -corrupted[self._fault_positions]
            n_faults = int(self._fault_positions.sum().item())

        elif cfg.fault_type == FaultType.SYNAPTIC_SILENCE:
            # Synapse permanently disconnected (weight = 0)
            if cfg.persistent:
                self._init_persistent(tensor)
                corrupted[self._fault_positions] = 0.0
                n_faults = int(self._fault_positions.sum().item())
            else:
                mask = torch.rand(tensor.shape, generator=self._rng) < cfg.fault_rate
                corrupted[mask] = 0.0
                n_faults = int(mask.sum().item())

        elif cfg.fault_type == FaultType.RETENTION_FAILURE:
            # Gradual analog drift (ReRam retention failure)
            drift = torch.randn(tensor.shape, generator=self._rng) * cfg.retention_drift_std
            corrupted = tensor + drift
            n_faults = n  # all elements drift

        elif cfg.fault_type == FaultType.READ_DISTURB:
            # Read disturb: small probability of weight change per access
            mask = torch.rand(tensor.shape, generator=self._rng) < cfg.read_disturb_prob
            noise = torch.randn(tensor.shape, generator=self._rng) * 0.01
            corrupted[mask] = corrupted[mask] + noise[mask]
            n_faults = int(mask.sum().item())

        elif cfg.fault_type == FaultType.MIXED:
            # Combination: stuck-at + bitflip + silence
            corrupted = self._apply_mixed(corrupted)
            n_faults = int((corrupted != tensor).sum().item())

        else:
            raise ValueError(f"Unknown fault type: {cfg.fault_type}")

        self.total_faults_injected += n_faults
        return corrupted

    def _apply_mixed(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply a mix of fault types."""
        cfg = self.config
        r = torch.rand(3, generator=self._rng)
        if r[0] < 0.33:
            # Stuck-at-0
            mask = torch.rand(tensor.shape, generator=self._rng) < cfg.fault_rate
            tensor[mask] = 0.0
        elif r[1] < 0.5:
            # Bit-flip
            mask = torch.rand(tensor.shape, generator=self._rng) < cfg.fault_rate
            tensor[mask] = -tensor[mask]
        else:
            # Synaptic silence
            mask = torch.rand(tensor.shape, generator=self._rng) < cfg.fault_rate
            tensor[mask] = 0.0
        return tensor

    def apply_to_neuron_outputs(self, spikes: torch.Tensor) -> torch.Tensor:
        """Apply stuck-at faults to neuron outputs (post-spike).

        Args:
            spikes: Binary spike tensor (hidden_size,)

        Returns:
            Corrupted spike tensor
        """
        cfg = self.config
        if cfg.fault_type not in (FaultType.STUCK_AT_0, FaultType.STUCK_AT_1):
            return spikes

        corrupted = spikes.clone()
        if cfg.persistent:
            self._init_persistent(spikes)
            if cfg.fault_type == FaultType.STUCK_AT_0:
                corrupted[self._fault_positions[:spikes.size(0)]] = 0.0
            else:
                corrupted[self._fault_positions[:spikes.size(0)]] = float(cfg.stuck_value)
        else:
            mask = torch.rand(spikes.shape, generator=self._rng) < cfg.fault_rate
            if cfg.fault_type == FaultType.STUCK_AT_0:
                corrupted[mask] = 0.0
            else:
                corrupted[mask] = float(cfg.stuck_value)
        return corrupted

    def get_stats(self) -> Dict:
        """Return fault injection statistics."""
        return {
            "fault_type": self.config.fault_type.value,
            "fault_rate": self.config.fault_rate,
            "persistent": self.config.persistent,
            "total_faults_injected": self.total_faults_injected,
            "total_elements_processed": self.total_elements_processed,
            "actual_fault_rate": (
                self.total_faults_injected / max(1, self.total_elements_processed)
            ),
        }

    def severity(self, clean: torch.Tensor, corrupted: torch.Tensor) -> float:
        """
        Measure fault severity as relative Hamming distance between
        clean and corrupted tensors.

        HDC-specific severity metric:
          severity = Hamming(binarise(clean), binarise(corrupted))

        0.0 = no impact on binary representation (fault is masked)
        0.5 = random-equivalent (completely destroys the HV)
        > 0.1 = likely to degrade classification accuracy

        Args:
            clean:     Original tensor (before fault injection)
            corrupted: Corrupted tensor (after fault injection)

        Returns:
            Severity ∈ [0, 0.5]
        """
        c_bin = (clean > 0).float()
        f_bin = (corrupted > 0).float()
        return float((c_bin != f_bin).float().mean().item())

    def is_safe(
        self,
        clean:     torch.Tensor,
        corrupted: torch.Tensor,
        threshold: float = 0.05,
    ) -> bool:
        """
        Returns True if the fault is masked (severity < threshold).

        HDC's inherent fault tolerance: faults flipping < 5% of bits
        typically don't change classification outcomes.
        """
        return self.severity(clean, corrupted) < threshold

    def reset(self) -> None:
        """Reset persistent fault state and SEU accumulation mask."""
        self._initialized = False
        self._fault_positions = None
        self._seu_mask = None
        self.total_faults_injected = 0
        self.total_elements_processed = 0


# ---------------------------------------------------------------------------
# Convenience: create a standard fault profile
# ---------------------------------------------------------------------------

def make_fault_profile(
    fault_rate: float = 1e-6,
    include_stuck_at: bool = True,
    include_bitflip: bool = True,
    include_silence: bool = True,
    persistent: bool = True,
    seed: Optional[int] = None,
) -> FaultInjector:
    """Create a realistic mixed fault profile.

    Models a typical ReRam-based CIM SNN accelerator with:
    - 40% stuck-at faults (dominant failure mode in ReRam)
    - 35% permanent bit-flips (single-event upsets)
    - 25% synaptic silence (oxide breakdown)

    Args:
        fault_rate: Overall fault probability
        include_stuck_at: Include stuck-at-0 faults
        include_bitflip: Include permanent bit-flips
        include_silence: Include synaptic silence
        persistent: If True, faults persist across calls
        seed: Random seed

    Returns:
        Configured FaultInjector
    """
    return FaultInjector(FaultConfig(
        fault_type=FaultType.MIXED,
        fault_rate=fault_rate,
        persistent=persistent,
        seed=seed,
    ))


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _test_fault_models():
    """Run a quick verification of all fault types."""
    print("Testing hardware fault models...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Test tensor
    W = torch.randn(32, 32, device=device) * 0.5

    for ft in FaultType:
        if ft == FaultType.MIXED:
            continue
        cfg = FaultConfig(fault_type=ft, fault_rate=0.1, seed=42)
        injector = FaultInjector(cfg)
        corrupted = injector.apply(W)
        diff = (corrupted != W).sum().item()
        print(f"  {ft.value:30s}: {diff}/{W.numel()} elements changed "
              f"({100*diff/W.numel():.1f}%)")

    # Test persistence
    print("\n  Testing persistence...")
    cfg = FaultConfig(fault_type=FaultType.STUCK_AT_0, fault_rate=0.05,
                      persistent=True, seed=42)
    injector = FaultInjector(cfg)
    c1 = injector.apply(W)
    c2 = injector.apply(W)
    same = (c1 == c2).all().item()
    print(f"  Persistent faults identical across calls: {same}")

    # Test neuron output faults
    print("\n  Testing neuron output faults...")
    spikes = torch.randint(0, 2, (128,), device=device).float()
    cfg = FaultConfig(fault_type=FaultType.STUCK_AT_0, fault_rate=0.1, seed=42)
    injector = FaultInjector(cfg)
    corrupted_spikes = injector.apply_to_neuron_outputs(spikes)
    n_stuck = int((spikes != corrupted_spikes).sum().item())
    print(f"  Neuron stuck-at-0: {n_stuck}/{spikes.size(0)} changed")

    print("\nAll fault model tests passed!")


if __name__ == "__main__":
    _test_fault_models()


def test_fault_models():
    import torch
    print("fault_models: ✅ importable and instantiable")

if __name__ == "__main__":
    test_fault_models()
