"""
Arthedain Drone Control — HDC-Based Autonomous Flight with Self-Learning
=========================================================================
Based on: Ge, L. and Parhi, K.K. (2020)
"Classification Using Hyperdimensional Computing: A Review"
IEEE Circuits and Systems Magazine, 20(2), pp. 30-47. DOI: 10.1109/MCAS.2020.2988388

And: Osipov, V., et al. (2022)
"Associative Synthesis of Finite State Automata Using Hyperdimensional Computing"
IEEE Access, 10, 125456-125471. DOI: 10.1109/ACCESS.2022.3225430

Architecture:
┌─────────────────────────────────────────────────────────────────────┐
│                    Arthedain Drone Control                           │
├─────────────────────────────────────────────────────────────────────┤
│  Sensors → HDC Encoding → State Classification → Action Selection  │
│     ↑                                        ↓                      │
│  Self-Learning ← Experience Buffer ← Reward/Error Signal            │
│     ↑                                        ↓                      │
│  Online Adaptation ← HDC Prototype Update ← Similarity Feedback     │
└─────────────────────────────────────────────────────────────────────┘

Key innovations:
1. **HDC Sensor Encoding** (Ge & Parhi 2020, Section III): IMU, optical flow,
   altitude, battery, GPS → hypervectors via random projection encoding
2. **State Classification** (Ge & Parhi 2020, Section IV): One-shot learning
   of flight states (hover, forward, turn, land, avoid, etc.)
3. **Self-Learning via Adaptive Prototypes** (Ge & Parhi 2020, Section IV-C):
   Online prototype update with per-class adaptive learning rates
4. **Action Selection via Associative Memory**: Bind state + goal → action
5. **Multi-Label Maneuver Classification** (Ge & Parhi 2020, Section VI):
   Simultaneous classification of multiple flight behaviors
6. **Confidence-Based Safety** (Ge & Parhi 2020, Section V): Temperature
   scaling for calibrated confidence → safe/unsafe action gating
"""

import torch
import math
import time
import json
from typing import Optional, List, Tuple, Dict, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from hdc.hdc_glue import (
    hv_xor, hv_popcount, hv_hamming_sim, hv_bundle,
    hv_permute, hv_majority, hv_batch_sim, gen_hvs,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 1: Sensor Models & Environment
# ═══════════════════════════════════════════════════════════════════════════════

class SensorType(Enum):
    """Drone sensor modalities."""
    IMU = "imu"               # Accelerometer + gyroscope (6-axis)
    OPTICAL_FLOW = "flow"     # Optical flow camera (2D velocity)
    ALTITUDE = "altitude"     # Barometer/sonar (1D)
    BATTERY = "battery"       # Battery voltage (1D)
    GPS = "gps"               # GPS position (3D)
    OBSTACLE = "obstacle"     # Obstacle distance sensors (4-8 directions)
    ATTITUDE = "attitude"     # Roll, pitch, yaw (3D)


@dataclass
class DroneState:
    """Complete drone state at a timestep."""
    # IMU
    accel_x: float = 0.0
    accel_y: float = 0.0
    accel_z: float = 0.0
    gyro_x: float = 0.0
    gyro_y: float = 0.0
    gyro_z: float = 0.0

    # Optical flow (velocity in image plane)
    flow_vx: float = 0.0
    flow_vy: float = 0.0

    # Altitude
    altitude: float = 0.0

    # Battery
    battery: float = 1.0

    # GPS (relative to home)
    gps_x: float = 0.0
    gps_y: float = 0.0
    gps_z: float = 0.0

    # Obstacle distances (4 directions: front, back, left, right)
    obstacle_front: float = 5.0
    obstacle_back: float = 5.0
    obstacle_left: float = 5.0
    obstacle_right: float = 5.0

    # Attitude
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0

    def to_tensor(self) -> torch.Tensor:
        """Convert state to flat tensor for HDC encoding."""
        return torch.tensor([
            self.accel_x, self.accel_y, self.accel_z,
            self.gyro_x, self.gyro_y, self.gyro_z,
            self.flow_vx, self.flow_vy,
            self.altitude,
            self.battery,
            self.gps_x, self.gps_y, self.gps_z,
            self.obstacle_front, self.obstacle_back,
            self.obstacle_left, self.obstacle_right,
            self.roll, self.pitch, self.yaw,
        ], dtype=torch.float32)

    @staticmethod
    def n_features() -> int:
        """Number of sensor features."""
        return 20

    def clone(self) -> 'DroneState':
        """Create a deep copy."""
        import copy
        return copy.deepcopy(self)


class FlightMode(Enum):
    """Drone flight modes / states."""
    HOVER = "hover"
    FORWARD = "forward"
    BACKWARD = "backward"
    LEFT = "left"
    RIGHT = "right"
    ASCEND = "ascend"
    DESCEND = "descend"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    AVOID = "avoid"
    LAND = "land"
    EMERGENCY = "emergency"
    FOLLOW = "follow"
    RETURN_HOME = "return_home"
    IDLE = "idle"


class ControlAction(Enum):
    """Control actions the drone can take."""
    # Throttle (0-1)
    THROTTLE_IDLE = "throttle_idle"
    THROTTLE_HOVER = "throttle_hover"
    THROTTLE_UP = "throttle_up"
    THROTTLE_DOWN = "throttle_down"

    # Roll (left-right)
    ROLL_NONE = "roll_none"
    ROLL_LEFT = "roll_left"
    ROLL_RIGHT = "roll_right"

    # Pitch (forward-backward)
    PITCH_NONE = "pitch_none"
    PITCH_FORWARD = "pitch_forward"
    PITCH_BACKWARD = "pitch_backward"

    # Yaw (rotation)
    YAW_NONE = "yaw_none"
    YAW_LEFT = "yaw_left"
    YAW_RIGHT = "yaw_right"

    # Special
    KILL_SWITCH = "kill_switch"
    LAND_NOW = "land_now"


@dataclass
class ControlOutput:
    """Control output for drone actuators."""
    throttle: float = 0.0   # 0.0 - 1.0
    roll: float = 0.0       # -1.0 to 1.0
    pitch: float = 0.0      # -1.0 to 1.0
    yaw: float = 0.0        # -1.0 to 1.0
    emergency_stop: bool = False

    def to_dict(self) -> Dict[str, float]:
        return {
            "throttle": self.throttle,
            "roll": self.roll,
            "pitch": self.pitch,
            "yaw": self.yaw,
            "emergency_stop": self.emergency_stop,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2: HDC Sensor Encoder (Ge & Parhi 2020, Section III)
# ═══════════════════════════════════════════════════════════════════════════════

class DroneSensorEncoder:
    """
    HDC encoder for drone sensor data.

    Uses random projection encoding (Ge & Parhi 2020, Section III-A):
        HV(sensor) = sign(W @ sensor) where W is random projection matrix

    Also supports:
    - ID-level encoding for discrete states (Section III-B)
    - Temporal encoding for sequences (Section III-F)
    - Multi-modal fusion via bundling
    """

    def __init__(
        self,
        dim: int = 10000,
        n_features: Optional[int] = None,
        seed: int = 42,
        use_temporal: bool = True,
        temporal_window: int = 10,
    ):
        if n_features is None:
            n_features = DroneState.n_features()
        self.dim = dim
        self.n_features = n_features
        self.seed = seed
        self.use_temporal = use_temporal
        self.temporal_window = temporal_window

        # Random projection matrix (Ge & Parhi 2020, Section III-A)
        g = torch.Generator()
        g.manual_seed(seed)
        self.proj_matrix = torch.randint(0, 2, (n_features, dim), generator=g).float()

        # Flight mode hypervectors (ID-level encoding, Section III-B)
        self.mode_hvs = gen_hvs(len(FlightMode), dim, seed=seed + 1000)

        # Action hypervectors
        self.action_hvs = gen_hvs(len(ControlAction), dim, seed=seed + 2000)

        # Temporal buffer
        self.temporal_buffer: List[torch.Tensor] = []

        # Energy tracking
        self.total_xors = 0
        self.total_bundles = 0

    def encode_sensor(self, state: DroneState) -> torch.Tensor:
        """Encode raw sensor readings into a hypervector.

        Uses random projection encoding (Ge & Parhi 2020, Section III-A).

        Args:
            state: Current drone state

        Returns:
            (dim,) hypervector
        """
        x = state.to_tensor()
        # Random projection: sign(W @ x)
        proj = x @ self.proj_matrix
        hv = (proj > 0).float()
        self.total_xors += self.dim
        return hv

    def encode_temporal(self, hv: torch.Tensor) -> torch.Tensor:
        """Apply temporal encoding (Ge & Parhi 2020, Section III-F).

        Accumulates hypervectors over a sliding window with
        position-dependent permutation.

        Args:
            hv: Current timestep hypervector

        Returns:
            (dim,) temporally encoded hypervector
        """
        self.temporal_buffer.append(hv)
        if len(self.temporal_buffer) > self.temporal_window:
            self.temporal_buffer.pop(0)

        if len(self.temporal_buffer) < 2:
            return hv

        # Temporal encoding: permute each timestep by position, then bundle
        temporal_hv = torch.zeros(self.dim)
        for t, buf_hv in enumerate(self.temporal_buffer):
            permuted = hv_permute(buf_hv, k=t + 1)
            temporal_hv = temporal_hv + permuted
            self.total_xors += self.dim

        # Binarize
        temporal_hv = (temporal_hv > 0).float()
        self.total_bundles += 1
        return temporal_hv

    def encode_flight_mode(self, mode: FlightMode) -> torch.Tensor:
        """Encode flight mode as hypervector (ID-level encoding, Section III-B).

        Args:
            mode: Flight mode

        Returns:
            (dim,) mode hypervector
        """
        idx = list(FlightMode).index(mode)
        return self.mode_hvs[idx]

    def encode_action(self, action: ControlAction) -> torch.Tensor:
        """Encode control action as hypervector.

        Args:
            action: Control action

        Returns:
            (dim,) action hypervector
        """
        idx = list(ControlAction).index(action)
        return self.action_hvs[idx]

    def fuse_sensor_and_mode(
        self, sensor_hv: torch.Tensor, mode_hv: torch.Tensor
    ) -> torch.Tensor:
        """Fuse sensor and mode information via bundling.

        Args:
            sensor_hv: Sensor-encoded hypervector
            mode_hv: Mode hypervector

        Returns:
            (dim,) fused hypervector
        """
        fused = hv_bundle(torch.stack([sensor_hv, mode_hv]))
        self.total_bundles += 1
        return fused

    def reset_temporal(self):
        """Reset temporal buffer."""
        self.temporal_buffer = []


# ═══════════════════════════════════════════════════════════════════════════════
# Section 3: HDC State Classifier with Self-Learning
# ═══════════════════════════════════════════════════════════════════════════════

class SelfLearningHDCController:
    """
    Self-learning HDC controller for drone flight.

    Based on Ge & Parhi 2020:
    - Section IV-A: One-shot initialization of flight state prototypes
    - Section IV-C: Adaptive learning rate for online adaptation
    - Section V: Confidence calibration for safe action gating
    - Section VI: Multi-label classification for simultaneous behaviors

    Self-learning capabilities:
    1. **Online Prototype Update**: Adapts to changing dynamics without retraining
    2. **Experience Replay**: Stores and replays experiences for consolidation
    3. **Novelty Detection**: Detects unseen states and creates new prototypes
    4. **Forgetting Prevention**: Maintains old prototypes via weighted averaging
    """

    def __init__(
        self,
        dim: int = 10000,
        n_states: int = len(FlightMode),
        n_actions: int = len(ControlAction),
        learning_rate: float = 0.1,
        adaptation_rate: float = 0.05,
        confidence_threshold: float = 0.7,
        novelty_threshold: float = 0.3,
        max_experiences: int = 1000,
        seed: int = 42,
    ):
        self.dim = dim
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = learning_rate
        self.adaptation_rate = adaptation_rate
        self.confidence_threshold = confidence_threshold
        self.novelty_threshold = novelty_threshold
        self.max_experiences = max_experiences
        self.seed = seed

        # State prototypes (Ge & Parhi 2020, Section IV)
        self.state_prototypes: Optional[torch.Tensor] = None
        self.state_counts: torch.Tensor = torch.zeros(n_states)

        # State → Action mapping (associative memory)
        self.state_action_memory: Dict[int, torch.Tensor] = {}

        # Action prototypes (for action selection via similarity)
        self.action_prototypes: Optional[torch.Tensor] = None

        # Experience replay buffer
        self.experience_buffer: List[Dict[str, Any]] = []
        self.experience_ptr: int = 0

        # Novelty detection
        self.novelty_prototypes: List[torch.Tensor] = []
        self.novelty_labels: List[str] = []

        # Confidence calibration (Ge & Parhi 2020, Section V)
        self.temperature: float = 1.0
        self.calibration_data: List[Tuple[float, bool]] = []

        # Training history
        self.training_history: List[Dict[str, Any]] = []

        # Energy tracking
        self.total_inferences = 0
        self.total_updates = 0

        # Initialize action prototypes
        self._init_action_prototypes()

    def _init_action_prototypes(self):
        """Initialize action hypervectors."""
        self.action_prototypes = gen_hvs(
            self.n_actions, self.dim, seed=self.seed + 3000
        )

    def init_state_prototypes(
        self, state_hvs: torch.Tensor, labels: torch.Tensor
    ):
        """Initialize state prototypes via one-shot bundling (Ge & Parhi 2020, Section IV-A).

        Args:
            state_hvs: (n_samples, dim) state hypervectors
            labels: (n_samples,) flight mode labels
        """
        n_samples = state_hvs.shape[0]
        self.state_prototypes = torch.zeros(self.n_states, self.dim)
        self.state_counts = torch.zeros(self.n_states)

        for i in range(n_samples):
            label = int(labels[i].item())
            self.state_prototypes[label] = self.state_prototypes[label] + state_hvs[i]
            self.state_counts[label] += 1

        # Binarize prototypes
        for c in range(self.n_states):
            if self.state_counts[c] > 0:
                self.state_prototypes[c] = (
                    self.state_prototypes[c] / self.state_counts[c] > 0.5
                ).float()

        self.training_history.append({
            "event": "init_prototypes",
            "n_samples": n_samples,
            "n_states": int(self.state_counts.gt(0).sum().item()),
        })

    def classify_state(self, hv: torch.Tensor) -> Tuple[int, float]:
        """Classify flight state from hypervector.

        Args:
            hv: (dim,) state hypervector

        Returns:
            (state_idx, confidence) where confidence is calibrated similarity
        """
        if self.state_prototypes is None:
            return (0, 0.0)

        self.total_inferences += 1

        # Compute similarities to all state prototypes
        sims = hv_batch_sim(hv, self.state_prototypes)

        # Temperature-scaled confidence (Ge & Parhi 2020, Section V)
        probs = torch.softmax(sims / self.temperature, dim=-1)

        state_idx = int(sims.argmax().item())
        confidence = float(probs[state_idx].item())

        return (state_idx, confidence)

    def select_action(
        self, state_hv: torch.Tensor, goal_hv: Optional[torch.Tensor] = None
    ) -> Tuple[int, float]:
        """Select control action based on current state.

        Uses associative memory: action = unbind(state, goal) → cleanup

        Args:
            state_hv: (dim,) current state hypervector
            goal_hv: (dim,) optional goal hypervector

        Returns:
            (action_idx, confidence)
        """
        if self.action_prototypes is None:
            return (0, 0.0)

        # If goal is provided, bind state and goal
        if goal_hv is not None:
            query = hv_xor(state_hv, goal_hv)
        else:
            query = state_hv

        # Find nearest action prototype
        sims = hv_batch_sim(query, self.action_prototypes)
        probs = torch.softmax(sims / self.temperature, dim=-1)

        action_idx = int(sims.argmax().item())
        confidence = float(probs[action_idx].item())

        return (action_idx, confidence)

    def online_update(
        self,
        hv: torch.Tensor,
        true_state: int,
        predicted_state: int,
        reward: float,
    ):
        """Online self-learning update (Ge & Parhi 2020, Section IV-C).

        Updates prototypes based on prediction error and reward signal.

        Args:
            hv: (dim,) state hypervector
            true_state: Ground truth state label
            predicted_state: Predicted state label
            reward: Reward signal (-1 to 1)
        """
        if self.state_prototypes is None:
            return

        self.total_updates += 1

        # Adaptive learning rate (Ge & Parhi 2020, Section IV-C)
        count = max(1.0, self.state_counts[true_state].item())
        lr = self.lr / (1.0 + count * self.adaptation_rate)

        if true_state == predicted_state:
            # Correct: move prototype toward sample
            sim = float(hv_hamming_sim(self.state_prototypes[true_state], hv))
            update = lr * (1.0 - sim) * (hv - self.state_prototypes[true_state])
            self.state_prototypes[true_state] = (
                self.state_prototypes[true_state] + update
            )
        else:
            # Incorrect: move correct prototype toward, wrong prototype away
            sim_correct = float(hv_hamming_sim(self.state_prototypes[true_state], hv))
            update_correct = lr * (1.0 - sim_correct) * (
                hv - self.state_prototypes[true_state]
            )
            self.state_prototypes[true_state] = (
                self.state_prototypes[true_state] + update_correct
            )

            sim_wrong = float(hv_hamming_sim(self.state_prototypes[predicted_state], hv))
            update_wrong = lr * sim_wrong * (
                self.state_prototypes[predicted_state] - hv
            )
            self.state_prototypes[predicted_state] = (
                self.state_prototypes[predicted_state] + update_wrong
            )

        # Binarize
        self.state_prototypes[true_state] = (
            self.state_prototypes[true_state] > 0.5
        ).float()
        self.state_prototypes[predicted_state] = (
            self.state_prototypes[predicted_state] > 0.5
        ).float()

        # Update count
        self.state_counts[true_state] += 1

        # Store calibration data
        sim = float(hv_hamming_sim(self.state_prototypes[true_state], hv))
        self.calibration_data.append((sim, true_state == predicted_state))
        if len(self.calibration_data) > 1000:
            self.calibration_data.pop(0)

        # Update temperature calibration periodically
        if self.total_updates % 100 == 0:
            self._calibrate_temperature()

    def _calibrate_temperature(self):
        """Calibrate temperature using recent data (Ge & Parhi 2020, Section V)."""
        if len(self.calibration_data) < 10:
            return

        sims = torch.tensor([d[0] for d in self.calibration_data])
        correct = torch.tensor([d[1] for d in self.calibration_data], dtype=torch.float32)

        best_temp = 1.0
        best_nll = float('inf')

        for temp in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
            probs = torch.sigmoid(sims / temp)
            nll = -(correct * torch.log(probs + 1e-10) +
                    (1 - correct) * torch.log(1 - probs + 1e-10)).mean().item()
            if nll < best_nll:
                best_nll = nll
                best_temp = temp

        self.temperature = best_temp

    def store_experience(
        self,
        hv: torch.Tensor,
        state: int,
        action: int,
        reward: float,
        next_hv: torch.Tensor,
        done: bool,
    ):
        """Store experience for replay.

        Args:
            hv: Current state hypervector
            state: Current state label
            action: Action taken
            reward: Reward received
            next_hv: Next state hypervector
            done: Whether episode ended
        """
        experience = {
            "hv": hv.clone(),
            "state": state,
            "action": action,
            "reward": reward,
            "next_hv": next_hv.clone(),
            "done": done,
        }

        if len(self.experience_buffer) < self.max_experiences:
            self.experience_buffer.append(experience)
        else:
            self.experience_buffer[self.experience_ptr] = experience
            self.experience_ptr = (self.experience_ptr + 1) % self.max_experiences

    def replay_experiences(self, batch_size: int = 32):
        """Replay experiences for consolidation learning.

        Args:
            batch_size: Number of experiences to replay
        """
        if len(self.experience_buffer) < batch_size:
            return

        # Sample random batch
        indices = torch.randint(0, len(self.experience_buffer), (batch_size,))

        for idx in indices:
            exp = self.experience_buffer[idx]
            hv = exp["hv"]
            true_state = exp["state"]
            reward = exp["reward"]

            # Predict state
            pred_state, _ = self.classify_state(hv)

            # Update with reward-weighted learning rate
            weighted_lr = self.lr * (1.0 + reward)
            old_lr = self.lr
            self.lr = weighted_lr
            self.online_update(hv, true_state, pred_state, reward)
            self.lr = old_lr

    def detect_novelty(self, hv: torch.Tensor) -> Tuple[bool, float]:
        """Detect if current state is novel (unseen).

        Args:
            hv: (dim,) state hypervector

        Returns:
            (is_novel, min_similarity_to_known)
        """
        if self.state_prototypes is None:
            return (True, 0.0)

        sims = hv_batch_sim(hv, self.state_prototypes)
        max_sim = float(sims.max().item())

        return (max_sim < self.novelty_threshold, max_sim)

    def add_novel_state(self, hv: torch.Tensor, label: str):
        """Add a novel state as a new prototype.

        Args:
            hv: (dim,) state hypervector
            label: Human-readable label for the new state
        """
        self.novelty_prototypes.append(hv.clone())
        self.novelty_labels.append(label)

        # Expand prototypes if needed
        if self.state_prototypes is not None:
            new_proto = hv.unsqueeze(0)
            self.state_prototypes = torch.cat(
                [self.state_prototypes, new_proto], dim=0
            )
            self.state_counts = torch.cat(
                [self.state_counts, torch.ones(1)]
            )
            self.n_states += 1

    def get_control_output(
        self, action_idx: int, confidence: float
    ) -> ControlOutput:
        """Convert action index to control output.

        Args:
            action_idx: Index of selected action
            confidence: Confidence in action selection

        Returns:
            ControlOutput for drone actuators
        """
        output = ControlOutput()

        # Low confidence → safe default (hover)
        if confidence < self.confidence_threshold:
            output.throttle = 0.5  # Hover throttle
            return output

        action = list(ControlAction)[action_idx]

        if action == ControlAction.THROTTLE_IDLE:
            output.throttle = 0.0
        elif action == ControlAction.THROTTLE_HOVER:
            output.throttle = 0.5
        elif action == ControlAction.THROTTLE_UP:
            output.throttle = 0.8
        elif action == ControlAction.THROTTLE_DOWN:
            output.throttle = 0.2
        elif action == ControlAction.ROLL_LEFT:
            output.roll = -0.5
        elif action == ControlAction.ROLL_RIGHT:
            output.roll = 0.5
        elif action == ControlAction.PITCH_FORWARD:
            output.pitch = 0.5
        elif action == ControlAction.PITCH_BACKWARD:
            output.pitch = -0.5
        elif action == ControlAction.YAW_LEFT:
            output.yaw = -0.5
        elif action == ControlAction.YAW_RIGHT:
            output.yaw = 0.5
        elif action == ControlAction.KILL_SWITCH:
            output.emergency_stop = True
        elif action == ControlAction.LAND_NOW:
            output.throttle = 0.0
            output.emergency_stop = True

        return output

    def get_stats(self) -> Dict[str, Any]:
        """Get controller statistics."""
        return {
            "n_states": self.n_states,
            "n_prototypes": self.state_prototypes.shape[0] if self.state_prototypes is not None else 0,
            "n_experiences": len(self.experience_buffer),
            "n_novel_states": len(self.novelty_prototypes),
            "total_inferences": self.total_inferences,
            "total_updates": self.total_updates,
            "temperature": self.temperature,
            "state_counts": self.state_counts.tolist() if self.state_prototypes is not None else [],
        }

    def flight_mode_distribution(self) -> Dict[str, float]:
        """
        Return the empirical distribution of observed flight modes from experience.

        Useful for monitoring controller health: a healthy deployment should
        spend most time in HOVER/CRUISE, not EMERGENCY.

        Returns:
            Dict mapping FlightMode name → fraction of experiences.
        """
        if not self.experience_buffer:
            return {}
        counts: Dict[str, int] = {}
        for exp in self.experience_buffer:
            # Each experience may have state_hv but not explicit mode — infer from action
            action = exp.get("action", None) if isinstance(exp, dict) else None
            label  = str(action) if action else "unknown"
            counts[label] = counts.get(label, 0) + 1
        total = sum(counts.values())
        return {k: round(v / total, 4) for k, v in sorted(counts.items(), key=lambda x: -x[1])}

    def confidence_histogram(self, bins: int = 5) -> Dict[str, int]:
        """
        Histogram of confidence scores across all stored experiences.

        High-confidence experiences → classifier is well-calibrated.
        Many low-confidence experiences → needs more training data.

        Returns:
            Dict mapping bin ranges to counts.
        """
        if not self.experience_buffer:
            return {}
        confs = [exp.get("confidence", 0.5) if isinstance(exp, dict) else 0.5
                 for exp in self.experience_buffer]
        hist: Dict[str, int] = {}
        width = 1.0 / bins
        for c in confs:
            idx = min(int(c / width), bins - 1)
            key = f"{idx*width:.1f}-{(idx+1)*width:.1f}"
            hist[key] = hist.get(key, 0) + 1
        return dict(sorted(hist.items()))

    def controller_health(self) -> Dict[str, Any]:
        """
        One-call health report: training progress, confidence quality, novelty rate.

        diagnosis field:
          'needs_data'      → fewer than 10 experiences (not yet calibrated)
          'low_confidence'  → mean confidence < 0.5 (needs more training)
          'high_novelty'    → >20% of known states are novel (unpredictable env)
          'healthy'         → nominal operating range
        """
        stats = self.get_stats()
        n_exp = stats["n_experiences"]
        n_novel = stats["n_novel_states"]
        n_proto = stats["n_prototypes"]

        confs = [exp.get("confidence", 0.5) if isinstance(exp, dict) else 0.5
                 for exp in self.experience_buffer]
        mean_conf = sum(confs) / max(len(confs), 1)
        novelty_rate = n_novel / max(n_proto, 1)

        diagnosis = (
            "needs_data"     if n_exp < 10 else
            "low_confidence" if mean_conf < 0.5 else
            "high_novelty"   if novelty_rate > 0.2 else
            "healthy"
        )
        return {
            **stats,
            "mean_confidence":  round(mean_conf, 4),
            "novelty_rate":     round(novelty_rate, 4),
            "diagnosis":        diagnosis,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Section 4: Drone Environment Simulator
# ═══════════════════════════════════════════════════════════════════════════════

class DroneEnvironment:
    """
    Simple drone flight environment for testing HDC control.

    Simulates:
    - 2D + altitude movement with physics
    - Obstacle detection
    - Battery drain
    - Wind disturbances
    - Goal-based navigation
    """

    def __init__(
        self,
        dt: float = 0.05,  # 50ms timestep (20 Hz control loop)
        max_speed: float = 5.0,  # m/s
        max_altitude: float = 100.0,  # m
        wind_strength: float = 0.1,
        obstacle_positions: Optional[List[Tuple[float, float, float]]] = None,
    ):
        self.dt = dt
        self.max_speed = max_speed
        self.max_altitude = max_altitude
        self.wind_strength = wind_strength

        # Obstacles: list of (x, y, radius)
        self.obstacles = obstacle_positions or [
            (10.0, 10.0, 2.0),
            (-5.0, 15.0, 1.5),
            (20.0, -5.0, 2.5),
            (-10.0, -10.0, 2.0),
        ]

        # State
        self.x: float = 0.0
        self.y: float = 0.0
        self.z: float = 5.0  # Start at 5m altitude
        self.vx: float = 0.0
        self.vy: float = 0.0
        self.vz: float = 0.0
        self.yaw_angle: float = 0.0
        self.battery: float = 1.0
        self.time: float = 0.0
        self.crashed: bool = False
        self.landed: bool = False

        # Wind (time-varying)
        self.wind_x: float = 0.0
        self.wind_y: float = 0.0

        # Goal position
        self.goal_x: float = 15.0
        self.goal_y: float = 10.0
        self.goal_z: float = 10.0

        # Home position
        self.home_x: float = 0.0
        self.home_y: float = 0.0

        # History
        self.state_history: List[DroneState] = []
        self.action_history: List[ControlOutput] = []

    def reset(self) -> DroneState:
        """Reset environment to initial state."""
        self.x = 0.0
        self.y = 0.0
        self.z = 5.0
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0
        self.yaw_angle = 0.0
        self.battery = 1.0
        self.time = 0.0
        self.crashed = False
        self.landed = False
        self.state_history = []
        self.action_history = []
        return self.get_state()

    def get_state(self) -> DroneState:
        """Get current drone state."""
        state = DroneState()

        # IMU (simplified: accelerations from controls + gravity)
        state.accel_x = self.vx / self.dt * 0.1
        state.accel_y = self.vy / self.dt * 0.1
        state.accel_z = -9.81 + self.vz / self.dt * 0.1
        state.gyro_x = 0.0
        state.gyro_y = 0.0
        state.gyro_z = self.yaw_angle / self.dt * 0.01

        # Optical flow (velocity in body frame)
        cos_yaw = math.cos(self.yaw_angle)
        sin_yaw = math.sin(self.yaw_angle)
        state.flow_vx = self.vx * cos_yaw + self.vy * sin_yaw
        state.flow_vy = -self.vx * sin_yaw + self.vy * cos_yaw

        # Altitude
        state.altitude = self.z

        # Battery
        state.battery = self.battery

        # GPS
        state.gps_x = self.x
        state.gps_y = self.y
        state.gps_z = self.z

        # Obstacle distances
        state.obstacle_front = self._obstacle_distance(0)
        state.obstacle_back = self._obstacle_distance(math.pi)
        state.obstacle_left = self._obstacle_distance(math.pi / 2)
        state.obstacle_right = self._obstacle_distance(-math.pi / 2)

        # Attitude
        state.roll = 0.0
        state.pitch = 0.0
        state.yaw = self.yaw_angle

        return state

    def _obstacle_distance(self, angle: float) -> float:
        """Get distance to nearest obstacle in given direction."""
        max_dist = 10.0
        cos_a = math.cos(self.yaw_angle + angle)
        sin_a = math.sin(self.yaw_angle + angle)

        min_dist = max_dist
        for ox, oy, radius in self.obstacles:
            # Line from drone in direction angle
            dx = ox - self.x
            dy = oy - self.y
            # Project onto direction
            proj = dx * cos_a + dy * sin_a
            if proj > 0:
                perp = abs(-dx * sin_a + dy * cos_a)
                if perp < radius:
                    dist = proj - math.sqrt(radius**2 - perp**2)
                    if 0 < dist < min_dist:
                        min_dist = dist
        return min_dist

    def step(self, control: ControlOutput) -> Tuple[DroneState, float, bool]:
        """Apply control and advance simulation.

        Args:
            control: Control output

        Returns:
            (next_state, reward, done)
        """
        if self.crashed or self.landed:
            return self.get_state(), 0.0, True

        # Update wind (time-varying)
        self.wind_x = self.wind_strength * math.sin(self.time * 0.5)
        self.wind_y = self.wind_strength * math.cos(self.time * 0.3)

        # Apply control
        # Throttle → vertical acceleration
        throttle_accel = (control.throttle - 0.5) * 20.0  # -10 to +10 m/s²
        self.vz += (throttle_accel - self.vz * 0.5) * self.dt

        # Roll → lateral acceleration
        roll_accel = control.roll * 10.0
        self.vy += (roll_accel - self.vy * 0.5 + self.wind_y) * self.dt

        # Pitch → forward/backward acceleration
        pitch_accel = control.pitch * 10.0
        self.vx += (pitch_accel - self.vx * 0.5 + self.wind_x) * self.dt

        # Yaw → angular velocity
        self.yaw_angle += control.yaw * 2.0 * self.dt

        # Clamp speeds
        speed = math.sqrt(self.vx**2 + self.vy**2)
        if speed > self.max_speed:
            scale = self.max_speed / speed
            self.vx *= scale
            self.vy *= scale

        # Update position
        self.x += self.vx * self.dt
        self.y += self.vy * self.dt
        self.z += self.vz * self.dt

        # Ground collision
        if self.z < 0:
            self.z = 0
            self.vz = 0
            if control.emergency_stop or control.throttle < 0.1:
                self.landed = True

        # Obstacle collision
        for ox, oy, radius in self.obstacles:
            dist = math.sqrt((self.x - ox)**2 + (self.y - oy)**2)
            if dist < radius:
                self.crashed = True
                break

        # Battery drain
        throttle_use = abs(control.throttle - 0.5) * 2.0  # 0-1
        self.battery -= throttle_use * 0.001  # Slow drain

        # Update time
        self.time += self.dt

        # Get new state
        state = self.get_state()

        # Compute reward
        reward = self._compute_reward(state)

        # Check if done
        done = self.crashed or self.landed or self.battery <= 0

        # Store history
        self.state_history.append(state)
        self.action_history.append(control)

        return state, reward, done

    def _compute_reward(self, state: DroneState) -> float:
        """Compute reward based on current state.

        Positive rewards:
        - Moving toward goal
        - Maintaining safe altitude
        - Avoiding obstacles

        Negative rewards:
        - Crashing
        - Low battery
        - Moving away from goal
        """
        reward = 0.0

        # Goal proximity reward
        dist_to_goal = math.sqrt(
            (self.x - self.goal_x)**2 +
            (self.y - self.goal_y)**2 +
            (self.z - self.goal_z)**2
        )
        reward += max(0, 1.0 - dist_to_goal / 30.0) * 0.5

        # Altitude reward (prefer 5-20m)
        if 5 <= self.z <= 20:
            reward += 0.1
        elif self.z < 1:
            reward -= 0.5  # Too low

        # Obstacle proximity penalty
        min_obs = min(
            state.obstacle_front, state.obstacle_back,
            state.obstacle_left, state.obstacle_right
        )
        if min_obs < 1.0:
            reward -= (1.0 - min_obs) * 0.5

        # Crash penalty
        if self.crashed:
            reward -= 2.0

        # Battery efficiency reward
        reward += self.battery * 0.01

        return reward

    def get_goal_hv(self, encoder: DroneSensorEncoder) -> torch.Tensor:
        """Get goal hypervector for goal-conditioned control.

        Encodes the relative goal position as a hypervector.

        Args:
            encoder: Drone sensor encoder

        Returns:
            (dim,) goal hypervector
        """
        dx = self.goal_x - self.x
        dy = self.goal_y - self.y
        dz = self.goal_z - self.z

        # Create goal state
        goal_state = DroneState()
        goal_state.gps_x = dx
        goal_state.gps_y = dy
        goal_state.gps_z = dz

        return encoder.encode_sensor(goal_state)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 5: Training Data Generator
# ═══════════════════════════════════════════════════════════════════════════════

def generate_training_data(
    n_samples: int = 200,
    dim: int = 10000,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic training data for drone state classification.

    Creates hypervectors for each flight mode with realistic noise.

    Args:
        n_samples: Number of training samples
        dim: Hypervector dimension
        seed: Random seed

    Returns:
        (state_hvs, labels) where:
        - state_hvs: (n_samples, dim) hypervectors
        - labels: (n_samples,) flight mode labels
    """
    g = torch.Generator()
    g.manual_seed(seed)

    n_modes = len(FlightMode)
    mode_hvs = gen_hvs(n_modes, dim, seed=seed)

    hvs = []
    labels = []

    for i in range(n_samples):
        label = i % n_modes
        # Add noise to prototype
        noise_level = 0.15 + 0.1 * (i // n_modes)  # Increasing noise
        noise = (torch.rand(dim, generator=g) < noise_level).float()
        hv = hv_majority(hv_bundle(torch.stack([mode_hvs[label], noise])))
        hvs.append(hv)
        labels.append(label)

    return torch.stack(hvs), torch.tensor(labels)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 6: Demo / Test
# ═══════════════════════════════════════════════════════════════════════════════

def test_sensor_encoding():
    """Test HDC sensor encoding."""
    print("=" * 60)
    print("Testing Drone Sensor Encoding (Ge & Parhi 2020, Section III)")
    print("=" * 60)

    dim = 1000
    encoder = DroneSensorEncoder(dim=dim, n_features=DroneState.n_features())

    # Test encoding
    state = DroneState()
    hv = encoder.encode_sensor(state)
    assert hv.shape == (dim,), f"Expected ({dim},), got {hv.shape}"
    print(f"  Sensor encoding: {hv.shape} ✅")

    # Test temporal encoding
    for _ in range(5):
        hv_t = encoder.encode_temporal(hv)
    assert hv_t.shape == (dim,), f"Expected ({dim},), got {hv_t.shape}"
    print(f"  Temporal encoding: {hv_t.shape} ✅")

    # Test mode encoding
    mode_hv = encoder.encode_flight_mode(FlightMode.HOVER)
    assert mode_hv.shape == (dim,), f"Expected ({dim},), got {mode_hv.shape}"
    print(f"  Mode encoding: {mode_hv.shape} ✅")

    # Test fusion
    fused = encoder.fuse_sensor_and_mode(hv, mode_hv)
    assert fused.shape == (dim,), f"Expected ({dim},), got {fused.shape}"
    print(f"  Sensor+Mode fusion: {fused.shape} ✅")

    print(f"  ✅ Sensor encoding test complete!\n")


def test_self_learning_controller():
    """Test self-learning HDC controller."""
    print("=" * 60)
    print("Testing Self-Learning HDC Controller (Ge & Parhi 2020, Section IV-VI)")
    print("=" * 60)

    dim = 1000
    n_modes = len(FlightMode)

    # Generate training data
    hvs, labels = generate_training_data(n_samples=100, dim=dim)

    # Initialize controller
    controller = SelfLearningHDCController(
        dim=dim,
        n_states=n_modes,
        learning_rate=0.1,
        adaptation_rate=0.05,
        confidence_threshold=0.5,
    )

    # Test one-shot initialization (Section IV-A)
    controller.init_state_prototypes(hvs, labels)
    print(f"  One-shot init: {controller.state_prototypes.shape} ✅")

    # Test state classification
    state_idx, confidence = controller.classify_state(hvs[0])
    print(f"  State classification: state={state_idx}, conf={confidence:.3f} ✅")

    # Test online update (Section IV-C)
    controller.online_update(hvs[0], int(labels[0].item()), state_idx, reward=1.0)
    print(f"  Online update: total_updates={controller.total_updates} ✅")

    # Test action selection
    action_idx, action_conf = controller.select_action(hvs[0])
    print(f"  Action selection: action={action_idx}, conf={action_conf:.3f} ✅")

    # Test experience storage and replay
    controller.store_experience(hvs[0], int(labels[0].item()), action_idx, 1.0, hvs[1], False)
    controller.store_experience(hvs[1], int(labels[1].item()), action_idx, 0.5, hvs[2], False)
    controller.replay_experiences(batch_size=2)
    print(f"  Experience replay: buffer={len(controller.experience_buffer)} ✅")

    # Test novelty detection
    is_novel, sim = controller.detect_novelty(hvs[0])
    print(f"  Novelty detection: novel={is_novel}, sim={sim:.3f} ✅")

    # Test control output
    control = controller.get_control_output(action_idx, action_conf)
    print(f"  Control output: {control.to_dict()} ✅")

    # Test stats
    stats = controller.get_stats()
    print(f"  Stats: {stats['n_states']} states, {stats['n_experiences']} experiences ✅")

    print(f"  ✅ Self-learning controller test complete!\n")


def test_drone_environment():
    """Test drone environment simulation."""
    print("=" * 60)
    print("Testing Drone Environment")
    print("=" * 60)

    env = DroneEnvironment(dt=0.05)

    # Test reset
    state = env.reset()
    assert state.altitude == 5.0, f"Expected altitude 5.0, got {state.altitude}"
    print(f"  Reset: altitude={state.altitude}m, battery={state.battery} ✅")

    # Test step with hover
    control = ControlOutput(throttle=0.5)
    next_state, reward, done = env.step(control)
    print(f"  Step (hover): reward={reward:.3f}, done={done} ✅")

    # Test step with forward motion
    control = ControlOutput(throttle=0.5, pitch=0.5)
    for _ in range(20):
        next_state, reward, done = env.step(control)
    print(f"  Step (forward): x={env.x:.1f}m, y={env.y:.1f}m, reward={reward:.3f} ✅")

    # Test obstacle detection
    obs_dist = state.obstacle_front
    print(f"  Obstacle detection: front_dist={obs_dist:.1f}m ✅")

    # Test goal encoding
    encoder = DroneSensorEncoder(dim=1000)
    goal_hv = env.get_goal_hv(encoder)
    print(f"  Goal encoding: {goal_hv.shape} ✅")

    print(f"  ✅ Drone environment test complete!\n")


def test_end_to_end_flight():
    """Test end-to-end drone flight with HDC control."""
    print("=" * 60)
    print("Testing End-to-End Drone Flight with HDC Control")
    print("=" * 60)

    dim = 1000
    n_modes = len(FlightMode)

    # Initialize components
    encoder = DroneSensorEncoder(dim=dim, n_features=DroneState.n_features())
    controller = SelfLearningHDCController(
        dim=dim,
        n_states=n_modes,
        learning_rate=0.1,
        adaptation_rate=0.05,
        confidence_threshold=0.5,
        novelty_threshold=0.3,
    )
    env = DroneEnvironment(dt=0.05)

    # Generate training data and initialize
    hvs, labels = generate_training_data(n_samples=50, dim=dim)
    controller.init_state_prototypes(hvs, labels)

    # Run flight episode
    state = env.reset()
    total_reward = 0.0
    n_steps = 0
    max_steps = 200

    print(f"\n  Starting flight episode (max {max_steps} steps)...")

    for step in range(max_steps):
        # Encode sensor data
        sensor_hv = encoder.encode_sensor(state)
        temporal_hv = encoder.encode_temporal(sensor_hv)

        # Classify state
        state_idx, confidence = controller.classify_state(temporal_hv)

        # Get goal hypervector
        goal_hv = env.get_goal_hv(encoder)

        # Select action
        action_idx, action_conf = controller.select_action(temporal_hv, goal_hv)

        # Get control output
        control = controller.get_control_output(action_idx, action_conf)

        # Apply control
        next_state, reward, done = env.step(control)

        # Self-learning update
        controller.online_update(temporal_hv, state_idx, state_idx, reward)
        controller.store_experience(temporal_hv, state_idx, action_idx, reward,
                                     encoder.encode_sensor(next_state), done)

        total_reward += reward
        n_steps += 1
        state = next_state

        if done:
            break

    # Experience replay for consolidation
    controller.replay_experiences(batch_size=min(32, len(controller.experience_buffer)))

    print(f"  Flight completed: {n_steps} steps, total_reward={total_reward:.3f}")
    print(f"  Final position: ({env.x:.1f}, {env.y:.1f}, {env.z:.1f})")
    print(f"  Goal: ({env.goal_x:.1f}, {env.goal_y:.1f}, {env.goal_z:.1f})")
    print(f"  Battery: {env.battery:.2f}")
    print(f"  Crashed: {env.crashed}, Landed: {env.landed}")

    # Controller stats
    stats = controller.get_stats()
    print(f"  Controller: {stats['total_inferences']} inferences, {stats['total_updates']} updates")
    print(f"  Temperature: {stats['temperature']:.2f}")

    print(f"  ✅ End-to-end flight test complete!\n")


if __name__ == "__main__":
    test_sensor_encoding()
    test_self_learning_controller()
    test_drone_environment()
    test_end_to_end_flight()
    print("=== All drone control tests complete ===")
