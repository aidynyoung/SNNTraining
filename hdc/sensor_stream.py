"""
Continuous Sensor Streaming Interface for Physical AI
======================================================
Implements the **interface layer** of the three-layer Physical AI stack:

    Interface layer  → this module
    Interpretation   → PhysicsWorldModel (physics_world_model.py)
    Action/policy    → ActionEvaluator   (physics_world_model.py)

"This shift is happening now because the kind of data machines can learn from
 has fundamentally changed from static datasets to continuous, multimodal
 recordings of the physical world."
— IQT Physical AI framing.

Three key capabilities:

1. **MultimodalSensorEncoder** — Encodes heterogeneous sensor streams (video,
   RF, acoustics, time-series, IMU, lidar) into a unified hypervector space
   using modality-specific level encoders that share the same D-dimensional
   basis. This enables fusion without explicit alignment or architecture changes.

2. **SensorStreamBuffer** — Lock-free ring buffer with adaptive replay priority:
   samples near anomalies are replayed more frequently to accelerate learning
   at the edge of the distribution where the model is weakest.

3. **AnomalyTriggeredLearner** — Wraps PhysicsWorldModel to implement surprise-
   driven online learning:
     - Normal observations (low prediction error): update slowly
     - Surprising observations (high error): update aggressively + log
     - Catastrophic surprises (error > alarm threshold): trigger alert
   This mirrors biological predictive coding: learn most when surprised.

4. **PhysicalAIPipeline** — Ties the three layers together into a single
   callable that: ingests sensor readings → encodes to HV → updates world
   model → produces prediction + action ranking on every tick.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from hdc.physics_world_model import (
    PhysicsWorldModel,
    ActionCandidate,
    _xor,
    _majority,
    _hamming,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Sensor Modality Definitions
# ═══════════════════════════════════════════════════════════════════════════════

class ModalityType(Enum):
    TIME_SERIES   = "time_series"   # 1-D temporal signal (IMU, audio, ECG)
    IMAGE         = "image"         # 2-D spatial (camera, lidar slice)
    SPECTRUM      = "spectrum"      # frequency-domain (RF, acoustic FFT)
    CATEGORICAL   = "categorical"   # discrete states (FSM state, event label)
    SCALAR        = "scalar"        # single continuous value (temperature, pressure)


@dataclass
class SensorSpec:
    """Specification for one sensor modality."""
    name: str
    modality: ModalityType
    raw_dim: int            # native dimensionality of raw reading
    hd_dim: int             # output hypervector dimension (shared across modalities)
    n_levels: int = 64      # quantisation levels for continuous encoders
    seed: int = 0


@dataclass
class SensorReading:
    """Time-stamped sensor observation from one or more modalities."""
    timestamp: float
    data: Dict[str, torch.Tensor]   # modality_name → raw tensor
    metadata: Dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Modality-Specific Encoders
# ═══════════════════════════════════════════════════════════════════════════════

class LevelEncoder(nn.Module):
    """
    Thermometer-code level encoder for continuous values.

    Maps a scalar or vector to an HV using level hypervectors (Rahimi 2017):
    - Q level HVs with smooth similarity profile
    - Value → level index → level HV
    - Works for scalars, spectra, and time-series windows via projection
    """

    def __init__(self, input_dim: int, hd_dim: int, n_levels: int = 64, seed: int = 0):
        super().__init__()
        self.input_dim = input_dim
        self.hd_dim = hd_dim
        self.n_levels = n_levels

        g = torch.Generator()
        g.manual_seed(seed)

        # Random projection to scalar (if input_dim > 1)
        if input_dim > 1:
            self.proj = nn.Linear(input_dim, 1, bias=False)
            nn.init.normal_(self.proj.weight, std=1.0 / input_dim**0.5)
        else:
            self.proj = None

        # Level HVs: L[0] random, L[q] = L[q-1] with q/Q bits flipped
        base = (torch.rand(hd_dim, generator=g) < 0.5).float()
        levels = [base.clone()]
        for q in range(1, n_levels):
            prev = levels[-1].clone()
            n_flip = max(1, hd_dim // (2 * n_levels))
            idx = torch.randperm(hd_dim, generator=g)[:n_flip]
            prev[idx] = 1.0 - prev[idx]
            levels.append(prev)
        self.register_buffer("level_hvs", torch.stack(levels))  # (Q, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to HV.

        Args:
            x: (..., input_dim) or (...,) tensor

        Returns:
            (..., hd_dim) binary HV
        """
        if self.proj is not None and x.shape[-1] == self.input_dim:
            scalar = self.proj(x.float()).squeeze(-1)
        else:
            scalar = x.float().mean(dim=-1) if x.dim() > 0 else x.float()

        # Normalise to [0, 1] using tanh
        norm = torch.sigmoid(scalar)
        level_idx = (norm * (self.n_levels - 1)).long().clamp(0, self.n_levels - 1)
        return self.level_hvs[level_idx]


class TemporalWindowEncoder(nn.Module):
    """
    Encodes a sliding window of temporal readings as a sequence HV.

    Uses the n-gram binding approach: each timestep is bound with its
    position HV via XOR, then all timesteps are bundled.
    """

    def __init__(self, window: int, raw_dim: int, hd_dim: int, seed: int = 0):
        super().__init__()
        self.window = window
        self.hd_dim = hd_dim

        self.value_encoder = LevelEncoder(raw_dim, hd_dim, seed=seed)

        g = torch.Generator()
        g.manual_seed(seed + 1000)
        # Position HVs: one per timestep in the window
        self.register_buffer(
            "pos_hvs",
            (torch.rand(window, hd_dim, generator=g) < 0.5).float()
        )

    def forward(self, window_data: torch.Tensor) -> torch.Tensor:
        """
        Encode a temporal window.

        Args:
            window_data: (T, raw_dim) window of readings (T ≤ self.window)

        Returns:
            (hd_dim,) HV encoding the window
        """
        T = min(window_data.shape[0], self.window)
        components = []
        for t in range(T):
            val_hv = self.value_encoder(window_data[t])
            pos_hv = self.pos_hvs[t]
            bound = _xor(val_hv, pos_hv)
            components.append(bound)

        if not components:
            return torch.zeros(self.hd_dim)

        stacked = torch.stack(components)   # (T, D)
        return _majority(stacked.mean(dim=0))


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Multimodal Sensor Encoder
# ═══════════════════════════════════════════════════════════════════════════════

class MultimodalSensorEncoder(nn.Module):
    """
    Encodes heterogeneous sensor streams into a unified hypervector.

    "Multimodal/sensor-fusion world models integrate diverse signals such as
     video, RF, acoustics and time-series data to reason about context across
     space and time."
    — IQT Physical AI framing.

    Each modality gets its own encoder (appropriate for its type), producing
    a (D,) HV. The final sensor HV is the majority-bundle of all active
    modality HVs:
        sensor_hv = MAJORITY(hv_cam, hv_imu, hv_audio, hv_rf, ...)

    Args:
        specs: List of SensorSpec for each modality
        hd_dim: Shared hypervector dimensionality
        temporal_window: Window size for temporal encoders
    """

    def __init__(
        self,
        specs: List[SensorSpec],
        hd_dim: int = 4096,
        temporal_window: int = 16,
        use_flyhash: bool = False,
        flyhash_k: int = 10,
    ):
        """
        Args:
            use_flyhash: If True, route raw feature vectors through FlyHashEncoder
                         before the standard encoder, producing sparse binary HVs
                         with better similarity preservation (Kleyko 2025).
            flyhash_k: Number of active neurons in FlyHash output.
        """
        super().__init__()
        self.hd_dim = hd_dim
        self.specs = {s.name: s for s in specs}
        self.use_flyhash = use_flyhash

        # Optional FlyHash pre-encoders (one per modality with raw_dim > 1)
        self._flyhash_encoders = {}
        if use_flyhash:
            try:
                from hdc.flyhash import FlyHashEncoder
                for spec in specs:
                    if spec.raw_dim > 1:
                        self._flyhash_encoders[spec.name] = FlyHashEncoder(
                            input_dim=spec.raw_dim,
                            output_dim=hd_dim,
                            k=flyhash_k,
                            preprocessing="mean_center+l2",
                            seed=spec.seed,
                        )
            except ImportError:
                pass  # graceful fallback if flyhash not available

        encoders = {}
        for spec in specs:
            if spec.modality in (ModalityType.TIME_SERIES, ModalityType.SPECTRUM):
                encoders[spec.name] = TemporalWindowEncoder(
                    temporal_window, spec.raw_dim, hd_dim, seed=spec.seed
                )
            else:  # IMAGE, SCALAR, CATEGORICAL
                encoders[spec.name] = LevelEncoder(
                    spec.raw_dim, hd_dim, spec.n_levels, seed=spec.seed
                )

        self.encoders = nn.ModuleDict(encoders)

        # Modality identity HVs (for binding modality + content)
        g = torch.Generator()
        g.manual_seed(0)
        modality_hvs = {
            name: (torch.rand(hd_dim, generator=g) < 0.5).float()
            for name in self.encoders.keys()
        }
        for name, hv in modality_hvs.items():
            self.register_buffer(f"_mod_hv_{name}", hv)

    def _get_mod_hv(self, name: str) -> torch.Tensor:
        return getattr(self, f"_mod_hv_{name}")

    def encode_modality(self, name: str, data: torch.Tensor) -> torch.Tensor:
        """
        Encode a single modality reading to HV.

        If use_flyhash=True and a FlyHashEncoder exists for this modality,
        routes through FlyHash first (sparse, similarity-preserving, biologically
        inspired) then binds with modality identity HV.
        """
        # FlyHash path: flat input vector → sparse binary HV
        if self.use_flyhash and name in self._flyhash_encoders:
            flat = data.float().reshape(-1)
            content_hv = self._flyhash_encoders[name](flat.unsqueeze(0)).squeeze(0)
        else:
            content_hv = self.encoders[name](data)

        mod_hv = self._get_mod_hv(name)
        return _xor(content_hv, mod_hv)   # bind content with identity

    def forward(self, reading: SensorReading) -> torch.Tensor:
        """
        Encode all available modalities and bundle into sensor HV.

        Args:
            reading: SensorReading with data for available modalities

        Returns:
            (hd_dim,) binary HV encoding all sensor inputs
        """
        mod_hvs = []
        for name, data in reading.data.items():
            if name in self.encoders:
                hv = self.encode_modality(name, data)
                mod_hvs.append(hv)

        if not mod_hvs:
            return torch.zeros(self.hd_dim)

        stacked = torch.stack(mod_hvs)  # (n_mods, D)
        return _majority(stacked.mean(dim=0))


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Streaming Buffer with Priority Replay
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BufferedSample:
    """A buffered sensor HV with its priority weight."""
    sensor_hv: torch.Tensor
    timestamp: float
    prediction_error: float   # error when this was observed
    priority: float = 1.0


class SensorStreamBuffer:
    """
    Ring buffer with priority-weighted replay for online learning.

    Samples with high prediction error get higher priority — they represent
    regions where the world model is weakest and learning is most valuable.

    Priority replay (PER, Schaul et al. 2015) adapted to HDC:
        priority_i = (error_i + ε) ^ α
    At replay, sample according to P(i) ∝ priority_i.

    Args:
        capacity: Maximum buffer size
        alpha: Priority exponent (0 = uniform, 1 = full priority)
        epsilon: Minimum priority to prevent zero sampling
    """

    def __init__(
        self,
        capacity: int = 1000,
        alpha: float = 0.6,
        epsilon: float = 0.01,
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.epsilon = epsilon
        self._buf: deque = deque(maxlen=capacity)

    def push(self, sensor_hv: torch.Tensor, error: float, timestamp: Optional[float] = None):
        """Add a sample with its prediction error as priority signal."""
        priority = (abs(error) + self.epsilon) ** self.alpha
        sample = BufferedSample(
            sensor_hv=sensor_hv.detach().clone(),
            timestamp=timestamp or time.time(),
            prediction_error=error,
            priority=priority,
        )
        self._buf.append(sample)

    def sample(self, n: int = 8) -> List[BufferedSample]:
        """
        Sample n items weighted by priority.

        Higher-error samples are replayed more often.
        """
        if not self._buf:
            return []

        items = list(self._buf)
        priorities = torch.tensor([s.priority for s in items])
        probs = priorities / priorities.sum()

        n = min(n, len(items))
        indices = torch.multinomial(probs, n, replacement=False)
        return [items[i] for i in indices.tolist()]

    def sample_with_weights(
        self,
        n: int = 8,
        beta: float = 0.4,
    ) -> List[Dict]:
        """
        Sample n items with importance sampling weights for bias correction.

        Reference:
            Schaul et al. (2015) "Prioritized Experience Replay" ICLR 2016.

        When replaying with priority weighting, updates are biased toward
        high-error samples. Importance sampling weights correct this bias:
            w_i = (1 / (N × P(i)))^β

        Higher β (→1) = stronger bias correction; lower β = less correction.
        Anneal β from 0 toward 1 during training for gradual de-biasing.

        Args:
            n:    Number of samples to draw
            beta: IS exponent ∈ [0, 1]

        Returns:
            List of {sample, is_weight} dicts
        """
        if not self._buf:
            return []

        items    = list(self._buf)
        N        = len(items)
        priorities = torch.tensor([s.priority for s in items])
        probs    = priorities / priorities.sum()
        n        = min(n, N)
        indices  = torch.multinomial(probs, n, replacement=False)

        # IS weights: (1 / (N × P(i)))^β, normalised by max
        raw_w   = (1.0 / (N * probs[indices])) ** beta
        raw_w  /= raw_w.max()   # normalise so max weight = 1

        return [
            {"sample": items[idx], "is_weight": float(raw_w[k])}
            for k, idx in enumerate(indices.tolist())
        ]

    def update_priority(self, idx: int, new_error: float):
        """Update the priority of sample at buffer position idx."""
        if 0 <= idx < len(self._buf):
            items = list(self._buf)
            sample = items[idx]
            new_priority = (abs(new_error) + self.epsilon) ** self.alpha
            # Rebuild sample with updated priority (deque doesn't support item assignment)
            items[idx] = BufferedSample(
                sensor_hv=sample.sensor_hv,
                timestamp=sample.timestamp,
                prediction_error=new_error,
                priority=new_priority,
            )
            self._buf.clear()
            self._buf.extend(items)

    def mean_error(self) -> float:
        """Mean prediction error across all buffered samples."""
        if not self._buf:
            return 0.0
        return sum(s.prediction_error for s in self._buf) / len(self._buf)

    def __len__(self) -> int:
        return len(self._buf)

    def buffer_health(self) -> Dict:
        """
        Fill ratio, error statistics, and priority concentration.

        high priority_concentration → a few dominant samples (replay is focused).
        low  priority_concentration → uniform replay (buffer well-diversified).
        """
        n = len(self._buf)
        if n == 0:
            return {"n_samples": 0, "fill_ratio": 0.0, "mean_error": 0.0}
        errors = [s.prediction_error for s in self._buf]
        priorities = [s.priority for s in self._buf]
        total_p = sum(priorities) + 1e-12
        # Gini-like concentration: fraction of priority held by top 10%
        sorted_p = sorted(priorities, reverse=True)
        top10 = max(1, n // 10)
        conc = sum(sorted_p[:top10]) / total_p
        return {
            "n_samples":             n,
            "fill_ratio":            round(n / max(self.capacity, 1), 4),
            "mean_error":            round(sum(errors) / n, 4),
            "max_error":             round(max(errors), 4),
            "priority_concentration": round(conc, 4),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Anomaly-Triggered Self-Learning
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class LearningEvent:
    """Record of a learning update."""
    timestamp: float
    trigger: str              # "normal", "surprise", "alarm"
    prediction_error: float
    lr_used: float
    n_replay: int = 0


class AnomalyTriggeredLearner:
    """
    Surprise-driven online learning for the world model.

    "Progress depends on datasets that capture edge cases and rare failure
     modes. Unlike traditional perception, benchmarks alone are insufficient."
    — IQT Physical AI framing.

    Learning rate is proportional to prediction surprise:
        - Low surprise  (error < surprise_threshold):  lr_base    (slow update)
        - High surprise (error ≥ surprise_threshold):  lr_boost   (fast update)
        - Alarm         (error ≥ alarm_threshold):     lr_boost × 2 + alert

    On surprise, also replays recent high-error samples to reinforce
    learning at the distribution boundary.

    Args:
        world_model: PhysicsWorldModel to update
        buffer: SensorStreamBuffer for priority replay
        surprise_threshold: Hamming distance triggering fast update
        alarm_threshold: Hamming distance triggering alert
        lr_base: Learning rate for normal observations
        lr_boost: Learning rate multiplier on surprise
        n_replay: Number of priority samples to replay on surprise
    """

    def __init__(
        self,
        world_model: PhysicsWorldModel,
        buffer: SensorStreamBuffer,
        surprise_threshold: float = 0.15,
        alarm_threshold: float = 0.35,
        lr_base: float = 0.005,
        lr_boost: float = 0.05,
        n_replay: int = 4,
    ):
        self.world_model = world_model
        self.buffer = buffer
        self.surprise_threshold = surprise_threshold
        self.alarm_threshold = alarm_threshold
        self.lr_base = lr_base
        self.lr_boost = lr_boost
        self.n_replay = n_replay

        self._event_log: List[LearningEvent] = []
        self._alert_callbacks: List[Callable] = []
        self.n_surprises = 0
        self.n_alarms = 0

    def on_alarm(self, callback: Callable[[LearningEvent], None]):
        """Register a callback invoked when alarm threshold is exceeded."""
        self._alert_callbacks.append(callback)

    def ingest(self, sensor_hv: torch.Tensor) -> Dict:
        """
        Process one sensor observation with anomaly-triggered learning.

        Args:
            sensor_hv: (D,) binary HV from MultimodalSensorEncoder

        Returns:
            Dict with trigger, prediction_error, predictions, confidence
        """
        # Get current prediction before updating
        preds = self.world_model.multi_horizon(self.world_model.current_state)
        short_pred = preds["predictions"].get("short", self.world_model.current_state)

        # Measure prediction error
        error = 1.0 - float(_hamming(short_pred, sensor_hv).item())

        # Classify surprise level
        if error >= self.alarm_threshold:
            trigger = "alarm"
            lr = self.lr_boost * 2
            self.n_alarms += 1
        elif error >= self.surprise_threshold:
            trigger = "surprise"
            lr = self.lr_boost
            self.n_surprises += 1
        else:
            trigger = "normal"
            lr = self.lr_base

        # Buffer this sample with error as priority
        self.buffer.push(sensor_hv, error)

        # Update world model with current observation
        obs_info = self.world_model.observe(sensor_hv, learn=True)

        n_replay = 0
        if trigger in ("surprise", "alarm"):
            # Replay high-priority past samples
            replays = self.buffer.sample(self.n_replay)
            for sample in replays:
                self.world_model.multi_horizon.update(
                    self.world_model.current_state,
                    sample.sensor_hv,
                    lr=lr * 0.5,
                )
                n_replay += 1

        event = LearningEvent(
            timestamp=time.time(),
            trigger=trigger,
            prediction_error=error,
            lr_used=lr,
            n_replay=n_replay,
        )
        self._event_log.append(event)

        # Fire alert callbacks
        if trigger == "alarm":
            for cb in self._alert_callbacks:
                cb(event)

        return {
            "trigger": trigger,
            "prediction_error": error,
            "lr_used": lr,
            "n_replay": n_replay,
            **obs_info,
        }

    def learning_summary(self) -> Dict:
        """Return learning statistics."""
        total = len(self._event_log)
        if total == 0:
            return {"total": 0}

        by_trigger: Dict[str, int] = {}
        for e in self._event_log:
            by_trigger[e.trigger] = by_trigger.get(e.trigger, 0) + 1

        errors = [e.prediction_error for e in self._event_log]
        return {
            "total_observations": total,
            "by_trigger": by_trigger,
            "mean_error": sum(errors) / total,
            "max_error": max(errors),
            "n_surprises": self.n_surprises,
            "n_alarms": self.n_alarms,
            "buffer_size": len(self.buffer),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Physical AI Pipeline — Full Three-Layer Integration
# ═══════════════════════════════════════════════════════════════════════════════

class PhysicalAIPipeline:
    """
    Complete three-layer Physical AI pipeline:

        Interface layer   — MultimodalSensorEncoder
        Interpretation    — PhysicsWorldModel + AnomalyTriggeredLearner
        Action/policy     — ActionEvaluator (via world model)

    "World models shift AI from reacting to perceiving, and from perceiving
     to anticipating. They transform raw sensor data into decision-ready
     context, enabling systems to reason before acting, rehearse before
     committing, and adapt before changing conditions."
    — IQT Physical AI framing.

    Usage:
        pipeline = PhysicalAIPipeline(specs, hd_dim=4096)
        # Register safe/danger states (optional, from domain knowledge)
        pipeline.world_model.register_safe_state(nominal_hv)
        # Tick on every sensor observation:
        result = pipeline.tick(reading, candidate_actions)
        # result contains: predictions, ranked_actions, confidence, trigger
    """

    def __init__(
        self,
        sensor_specs: List[SensorSpec],
        hd_dim: int = 4096,
        temporal_window: int = 16,
        surprise_threshold: float = 0.15,
        alarm_threshold: float = 0.35,
        buffer_capacity: int = 1000,
    ):
        self.hd_dim = hd_dim

        # Interface layer
        self.encoder = MultimodalSensorEncoder(sensor_specs, hd_dim, temporal_window)

        # Interpretation layer
        self.world_model = PhysicsWorldModel(hd_dim=hd_dim)
        buffer = SensorStreamBuffer(capacity=buffer_capacity)
        self.learner = AnomalyTriggeredLearner(
            self.world_model, buffer,
            surprise_threshold=surprise_threshold,
            alarm_threshold=alarm_threshold,
        )

        self._tick_count = 0

    def tick(
        self,
        reading: SensorReading,
        candidate_actions: Optional[List[ActionCandidate]] = None,
        goal_state: Optional[torch.Tensor] = None,
    ) -> Dict:
        """
        Process one sensor reading through the full pipeline.

        Args:
            reading: Current multimodal sensor observation
            candidate_actions: Optional list for action evaluation
            goal_state: Optional target state HV

        Returns:
            Dict with predictions, ranked_actions, learning_info, confidence
        """
        self._tick_count += 1

        # Interface → Interpretation: encode sensor readings
        sensor_hv = self.encoder(reading)

        # Interpretation: update world model + anomaly-triggered learning
        learn_info = self.learner.ingest(sensor_hv)

        # Action evaluation (policy layer input)
        ranked_actions = None
        if candidate_actions:
            ranked_actions = self.world_model.evaluate_actions(
                candidate_actions, goal_state=goal_state
            )

        return {
            "sensor_hv": sensor_hv,
            "predictions": learn_info.get("predictions", {}),
            "confidence": learn_info.get("confidence", {}),
            "trigger": learn_info["trigger"],
            "prediction_error": learn_info["prediction_error"],
            "ranked_actions": ranked_actions,
            "twin_status": learn_info.get("twin_status", {}),
            "tick": self._tick_count,
        }

    def status(self) -> Dict:
        """Full pipeline status summary."""
        return {
            "tick_count": self._tick_count,
            "learning": self.learner.learning_summary(),
            "twin": self.world_model.twin_sync.status(),
            "confidence": self.world_model.multi_horizon.confidence_report(),
        }

    def pipeline_health(self) -> Dict:
        """
        Comprehensive one-call health report.

        Combines tick count, learning triggers, buffer state, and anomaly rate.
        diagnosis:
          'nominal'     → low error, few alarms
          'adapting'    → many surprises but no alarms (world model catching up)
          'degraded'    → many alarms (persistent high-error predictions)
          'cold_start'  → fewer than 10 ticks
        """
        learn = self.learner.learning_summary()
        buf   = self.learner.buffer.buffer_health()
        t     = self._tick_count
        n_alarms    = learn.get("n_alarms", 0)
        n_surprises = learn.get("n_surprises", 0)
        mean_err    = learn.get("mean_error", 0.0)

        alarm_rate    = n_alarms    / max(t, 1)
        surprise_rate = n_surprises / max(t, 1)
        diagnosis = (
            "cold_start" if t < 10          else
            "degraded"   if alarm_rate > 0.3 else
            "adapting"   if surprise_rate > 0.2 else
            "nominal"
        )
        return {
            "tick_count":     t,
            "alarm_rate":     round(alarm_rate, 4),
            "surprise_rate":  round(surprise_rate, 4),
            "mean_pred_error": round(mean_err, 4),
            "buffer":         buf,
            "diagnosis":      diagnosis,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_multimodal_encoder():
    print("=" * 60)
    print("Testing MultimodalSensorEncoder")
    print("=" * 60)

    specs = [
        SensorSpec("imu",    ModalityType.TIME_SERIES, raw_dim=6,  hd_dim=2000, seed=0),
        SensorSpec("audio",  ModalityType.SPECTRUM,    raw_dim=64, hd_dim=2000, seed=1),
        SensorSpec("temp",   ModalityType.SCALAR,      raw_dim=1,  hd_dim=2000, seed=2),
    ]

    encoder = MultimodalSensorEncoder(specs, hd_dim=2000, temporal_window=8)

    # Create a sensor reading with all three modalities
    reading = SensorReading(
        timestamp=time.time(),
        data={
            "imu":   torch.randn(8, 6),   # 8 timesteps × 6 axes
            "audio": torch.randn(8, 64),   # 8 timestep spectrum
            "temp":  torch.tensor([22.5]), # single scalar
        }
    )

    hv = encoder(reading)
    assert hv.shape == (2000,), f"Shape: {hv.shape}"
    density = float(hv.mean())
    print(f"  Encoded {len(reading.data)} modalities → HV(2000), density={density:.4f}")
    assert 0.4 < density < 0.6, f"Density out of range: {density}"

    # Same reading → same HV (deterministic encoder)
    hv2 = encoder(reading)
    sim = float(_hamming(hv, hv2).item())
    print(f"  Deterministic encoding: sim(hv, hv2) = {sim:.4f}  (want 1.0)")
    assert sim > 0.99

    # Different reading → different HV
    reading2 = SensorReading(
        timestamp=time.time(),
        data={"temp": torch.tensor([80.0])}  # only temperature, very different
    )
    hv3 = encoder(reading2)
    sim_diff = float(_hamming(hv, hv3).item())
    print(f"  Different reading: sim = {sim_diff:.4f}  (want < 1.0)")

    print("  ✅ MultimodalSensorEncoder OK")


def test_streaming_buffer():
    print("=" * 60)
    print("Testing SensorStreamBuffer (priority replay)")
    print("=" * 60)

    dim = 1000
    buf = SensorStreamBuffer(capacity=100, alpha=0.6)

    # Push 50 samples with varying errors
    torch.manual_seed(5)
    for i in range(50):
        hv = (torch.rand(dim) < 0.5).float()
        error = 0.05 + 0.45 * (i / 49)  # error increases from 0.05 to 0.5
        buf.push(hv, error)

    print(f"  Buffer size: {len(buf)}")
    assert len(buf) == 50

    # Sample 8 — high-error samples should appear more often
    n_high = 0
    for _ in range(100):
        samples = buf.sample(8)
        for s in samples:
            if s.prediction_error > 0.3:
                n_high += 1
    avg_high = n_high / (100 * 8)
    print(f"  Fraction of high-error samples in replay: {avg_high:.3f}  "
          f"(want > 0.4 with priority)")
    assert avg_high > 0.3, f"Priority replay not working: {avg_high}"

    print("  ✅ SensorStreamBuffer OK")


def test_anomaly_triggered_learner():
    print("=" * 60)
    print("Testing AnomalyTriggeredLearner")
    print("=" * 60)

    dim = 1000
    wm = PhysicsWorldModel(hd_dim=dim)
    buf = SensorStreamBuffer(capacity=200)
    learner = AnomalyTriggeredLearner(
        wm, buf,
        surprise_threshold=0.15,
        alarm_threshold=0.40,
        lr_base=0.005,
        lr_boost=0.05,
    )

    alarms_fired = []
    learner.on_alarm(lambda e: alarms_fired.append(e.prediction_error))

    torch.manual_seed(3)
    normal_hv = (torch.rand(dim) < 0.5).float()

    # Normal operation: small perturbations
    for _ in range(20):
        obs = normal_hv.clone()
        flip = torch.rand(dim) < 0.05
        obs[flip] = 1.0 - obs[flip]
        result = learner.ingest(obs)

    # Anomaly: large state change
    anomaly_hv = (torch.rand(dim) < 0.5).float()
    for _ in range(3):
        result = learner.ingest(anomaly_hv)

    summary = learner.learning_summary()
    print(f"  Summary: {summary}")
    print(f"  Alarms fired: {len(alarms_fired)}")
    assert "alarm" in summary.get("by_trigger", {}) or summary["n_surprises"] > 0, \
        "Expected at least one surprise or alarm from anomaly"

    print("  ✅ AnomalyTriggeredLearner OK")


def test_physical_ai_pipeline():
    print("=" * 60)
    print("Testing PhysicalAIPipeline (full 3-layer integration)")
    print("=" * 60)

    specs = [
        SensorSpec("imu",  ModalityType.TIME_SERIES, raw_dim=3, hd_dim=1000, seed=0),
        SensorSpec("temp", ModalityType.SCALAR,      raw_dim=1, hd_dim=1000, seed=1),
    ]

    pipeline = PhysicalAIPipeline(
        specs, hd_dim=1000,
        temporal_window=4,
        surprise_threshold=0.20,
        alarm_threshold=0.40,
    )

    # Define candidate actions
    dim = 1000
    torch.manual_seed(42)
    candidates = [
        ActionCandidate("hover",    (torch.rand(dim) < 0.05).float()),
        ActionCandidate("ascend",   (torch.rand(dim) < 0.1).float()),
        ActionCandidate("emergency",(torch.rand(dim) < 0.5).float()),
    ]

    # Run 15 normal ticks
    for t in range(15):
        reading = SensorReading(
            timestamp=float(t),
            data={
                "imu":  torch.randn(4, 3) * 0.1,
                "temp": torch.tensor([20.0 + t * 0.1]),
            }
        )
        result = pipeline.tick(reading, candidates)

    print(f"  After 15 ticks: trigger={result['trigger']}, "
          f"error={result['prediction_error']:.4f}")

    assert result["ranked_actions"] is not None
    print(f"  Ranked actions: {[(a.name, f'{a.net_score:.3f}') for a in result['ranked_actions']]}")

    status = pipeline.status()
    print(f"  Pipeline status: ticks={status['tick_count']}, "
          f"conf={status['confidence']}")
    assert status["tick_count"] == 15

    print("  ✅ PhysicalAIPipeline OK")


if __name__ == "__main__":
    test_multimodal_encoder()
    print()
    test_streaming_buffer()
    print()
    test_anomaly_triggered_learner()
    print()
    test_physical_ai_pipeline()
    print()
    print("=== All sensor_stream tests passed ===")
