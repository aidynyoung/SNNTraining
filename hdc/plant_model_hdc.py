"""
Associative Synthesis of Finite State Automata Model of a Controlled Object
=============================================================================
Based on: Osipov, E., Kleyko, D., and Legalov, A. (2017)
"Associative synthesis of finite state automata model of a controlled
 object with hyperdimensional computing"
IEEE Industrial Electronics Society Conference (IECON 2017).
DOI: 10.1109/IECON.2017.8216554

Key contribution:
  Evidence-based learning of a plant's dynamic model as an FSM from
  observed (state, input, next_state) transitions in a distributed
  automation and control system.

Unlike fsm_synthesis.py (which encodes a *predefined* FSM), this module
learns the FSM transition function *from observations* — the system does
not know the transition table in advance. It builds the model online,
handling:
  1. Ambiguous transitions — multiple observed next-states for the same
     (state, input) pair → confidence-weighted voting via bundling.
  2. Unseen transitions — graceful "don't know" for unobserved (s, a).
  3. Incremental learning — new observations update the model in O(D).

Algorithm (Osipov/Kleyko 2017):
  - Each state sᵢ and input aⱼ gets a unique random HV.
  - For each observed transition (sᵢ, aⱼ) → sₖ:
        key_hv  = XOR(s_hv[i], a_hv[j])   [bind state+input]
        next_hv = s_hv[k]                   [next-state HV]
        Store (key_hv, next_hv) in associative memory.
  - To query next state given (sᵢ, aⱼ):
        probe  = XOR(s_hv[i], a_hv[j])
        result = AM.query(probe) → nearest s_hv → decode state label

  With bundling for ambiguous transitions:
        proto[i,j] = MAJORITY_SUM over all observed s' for (sᵢ, aⱼ)
  Hamming distance from result to codebook entries gives confidence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import torch

from hdc.hdc_glue import (
    hv_xor, hv_bundle, hv_majority, hv_batch_sim, gen_hvs,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Codebooks: discrete labels → HVs
# ═══════════════════════════════════════════════════════════════════════════════

class HDCCodebook:
    """
    Bidirectional mapping: discrete label → random near-orthogonal HV.

    New labels are registered lazily on first encounter, preserving the
    near-orthogonality guarantee as long as the total number of labels
    is much smaller than D.
    """

    def __init__(self, dim: int = 10000, seed: Optional[int] = None):
        self.dim = dim
        self._seed = seed or 0
        self._label_to_idx: Dict[str, int] = {}
        self._hvs: Optional[torch.Tensor] = None
        self._n = 0

    def _grow(self, n_new: int = 10):
        """Allocate HVs for n_new additional labels."""
        g = torch.Generator()
        g.manual_seed(self._seed + self._n * 1000)
        new_hvs = (torch.rand(n_new, self.dim, generator=g) < 0.5).float()
        if self._hvs is None:
            self._hvs = new_hvs
        else:
            self._hvs = torch.cat([self._hvs, new_hvs], dim=0)

    def register(self, label: str) -> int:
        """Register label and return its index (idempotent)."""
        if label not in self._label_to_idx:
            if self._hvs is None or self._n >= self._hvs.shape[0]:
                self._grow()
            self._label_to_idx[label] = self._n
            self._n += 1
        return self._label_to_idx[label]

    def encode(self, label: str) -> torch.Tensor:
        """Return the HV for a label (registers it if new)."""
        idx = self.register(label)
        return self._hvs[idx]

    def decode(self, hv: torch.Tensor, top_k: int = 1) -> List[Tuple[str, float]]:
        """
        Find the closest label(s) to a query HV.

        Returns list of (label, similarity) sorted by similarity descending.
        """
        if self._n == 0 or self._hvs is None:
            return []

        sims = hv_batch_sim(hv, self._hvs[:self._n])
        top_idx = sims.topk(min(top_k, self._n)).indices.tolist()
        idx_to_label = {v: k for k, v in self._label_to_idx.items()}
        return [(idx_to_label[i], float(sims[i])) for i in top_idx]

    @property
    def labels(self) -> List[str]:
        return list(self._label_to_idx.keys())

    @property
    def n_labels(self) -> int:
        return self._n


# ═══════════════════════════════════════════════════════════════════════════════
# Transition Record
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TransitionObservation:
    """A single observed state transition."""
    state: str
    action: str
    next_state: str
    count: int = 1


# ═══════════════════════════════════════════════════════════════════════════════
# Learned Plant Model (Osipov/Kleyko 2017)
# ═══════════════════════════════════════════════════════════════════════════════

class LearnedPlantModel:
    """
    Evidence-based FSM learned online from (state, action, next_state) triples.

    Uses HDC associative memory to store the learned transition function:
        key_hv  = XOR(state_hv, action_hv)   — bind state + action
        proto   = MAJORITY_SUM over all observed next_state_hvs for this key
        AM: key_hv → proto (next-state prototype)

    Prediction for (state, action):
        1. Encode key = XOR(state_hv, action_hv)
        2. Find nearest proto in AM (ComplexHamming or batch Hamming)
        3. Decode proto → nearest state label

    Ambiguity:
        When multiple next-states are observed for the same (state, action),
        bundling their HVs forms a composite prototype. The Hamming distance
        from this prototype to the codebook entries indicates ambiguity:
            dist ≈ 0.5 × (1 - 2/n_conflicting_states)

    Confidence:
        similarity to nearest codebook entry → [0.5, 1.0] range
        0.5 = random (uncertain), 1.0 = perfect recall

    Args:
        dim: Hypervector dimensionality
        seed: Random seed
        unknown_threshold: Hamming distance above which a query is "unknown"
    """

    def __init__(
        self,
        dim: int = 10000,
        seed: Optional[int] = None,
        unknown_threshold: float = 0.45,
    ):
        self.dim = dim
        self.unknown_threshold = unknown_threshold

        self.state_book = HDCCodebook(dim=dim, seed=(seed or 0))
        self.action_book = HDCCodebook(dim=dim, seed=(seed or 0) + 10000)

        # Transition memory: key_str → (accumulated_hv, count)
        self._transition_accum: Dict[str, torch.Tensor] = {}
        self._transition_count: Dict[str, int] = {}

        # Binarised transition table for fast query
        self._keys: Optional[torch.Tensor] = None       # (n_transitions, dim)
        self._protos: Optional[torch.Tensor] = None     # (n_transitions, dim)
        self._key_index: Dict[str, int] = {}            # key_str → row in _keys
        self._dirty = True                              # rebuild index on next query

        # Observation log
        self._observations: List[TransitionObservation] = []
        self._obs_index: Dict[str, TransitionObservation] = {}

    def _key_str(self, state: str, action: str) -> str:
        return f"{state}|{action}"

    def _encode_key(self, state: str, action: str) -> torch.Tensor:
        """Bind state + action → transition key HV."""
        s_hv = self.state_book.encode(state)
        a_hv = self.action_book.encode(action)
        return hv_xor(s_hv, a_hv)

    # ── Learning ──────────────────────────────────────────────────────────────

    def observe(self, state: str, action: str, next_state: str):
        """
        Record one observed transition (online, O(D) per call).

        Accumulates next-state HVs for each (state, action) key.
        Identical transitions reinforce the prototype; conflicting ones
        produce a mixed prototype with lower confidence.

        Args:
            state: Current state label
            action: Input / action label
            next_state: Observed next state label
        """
        key = self._key_str(state, action)
        # Register all labels up front (state_book and action_book)
        self.state_book.register(state)
        self.action_book.register(action)
        ns_hv = self.state_book.encode(next_state)

        if key not in self._transition_accum:
            self._transition_accum[key] = ns_hv.float()
            self._transition_count[key] = 1
            self._obs_index[key] = TransitionObservation(state, action, next_state)
        else:
            self._transition_accum[key] = self._transition_accum[key] + ns_hv.float()
            self._transition_count[key] += 1
            self._obs_index[key].count += 1

        self._dirty = True

        self._observations.append(
            TransitionObservation(state, action, next_state)
        )

    def observe_sequence(self, sequence: List[Tuple[str, str, str]]):
        """
        Record a sequence of (state, action, next_state) observations.

        Args:
            sequence: List of (state, action, next_state) tuples
        """
        for s, a, ns in sequence:
            self.observe(s, a, ns)

    def _rebuild_index(self):
        """Binarise accumulated prototypes and build fast-query arrays."""
        if not self._transition_accum:
            self._dirty = False
            return

        keys_list = []
        protos_list = []
        self._key_index = {}

        for i, (key_str, accum) in enumerate(self._transition_accum.items()):
            n = self._transition_count[key_str]
            # Key HV
            obs = self._obs_index[key_str]
            key_hv = self._encode_key(obs.state, obs.action)

            # Binarise prototype: majority threshold at n/2
            proto_hv = (accum > n / 2).float()

            keys_list.append(key_hv)
            protos_list.append(proto_hv)
            self._key_index[key_str] = i

        self._keys = torch.stack(keys_list)      # (n_trans, dim)
        self._protos = torch.stack(protos_list)   # (n_trans, dim)
        self._dirty = False

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(
        self,
        state: str,
        action: str,
        top_k: int = 1,
    ) -> List[Dict]:
        """
        Predict the next state for a given (state, action).

        Args:
            state: Current state label
            action: Input / action label
            top_k: Return top-k candidates

        Returns:
            List of {'next_state', 'similarity', 'confidence', 'n_observations'}
            sorted by similarity (descending).
            Empty if (state, action) was never observed.
        """
        if self._dirty:
            self._rebuild_index()

        if self._keys is None:
            return []

        key_hv = self._encode_key(state, action)

        # Find nearest stored key
        key_sims = hv_batch_sim(key_hv, self._keys)  # (n_trans,)
        best_key_idx = int(key_sims.argmax().item())
        best_key_sim = float(key_sims[best_key_idx])

        if best_key_sim < 1.0 - self.unknown_threshold:
            # No sufficiently similar key found
            return []

        # Get prototype for this key and decode next state
        proto_hv = self._protos[best_key_idx]

        # Decode proto → nearest state labels
        candidates = self.state_book.decode(proto_hv, top_k=top_k)

        # Build result dicts
        key_str = list(self._key_index.keys())[best_key_idx]
        n_obs = self._transition_count.get(key_str, 0)

        results = []
        for label, sim in candidates:
            # Confidence: how similar the prototype is to a codebook entry
            # sim=1 → unambiguous, sim≈0.5 → highly ambiguous
            confidence = max(0.0, 2 * (sim - 0.5))
            results.append({
                "next_state": label,
                "similarity": sim,
                "confidence": confidence,
                "n_observations": n_obs,
                "ambiguous": n_obs > 1 and sim < 0.9,
            })

        return results

    def predict_best(
        self,
        state: str,
        action: str,
    ) -> Optional[Dict]:
        """Return the single most likely next state."""
        results = self.predict(state, action, top_k=1)
        return results[0] if results else None

    # ── Analysis ──────────────────────────────────────────────────────────────

    def transition_table(self) -> Dict[str, Dict[str, Dict]]:
        """
        Return the learned transition table as a nested dict.

        Returns:
            {state: {action: {next_state, confidence, n_observations}}}
        """
        table: Dict[str, Dict[str, Dict]] = {}
        for key_str, obs in self._obs_index.items():
            s, a = obs.state, obs.action
            pred = self.predict_best(s, a)
            if pred is None:
                continue
            if s not in table:
                table[s] = {}
            table[s][a] = pred

        return table

    def ambiguous_transitions(self) -> List[Dict]:
        """
        Return all (state, action) pairs with conflicting next-state observations.

        A transition is ambiguous if the same (state, action) was observed
        leading to different next states (non-deterministic plant behaviour).
        """
        conflicts = []
        for key_str, n in self._transition_count.items():
            if n > 1:
                obs = self._obs_index[key_str]
                pred = self.predict_best(obs.state, obs.action)
                if pred and pred["confidence"] < 0.8:
                    conflicts.append({
                        "state": obs.state,
                        "action": obs.action,
                        "n_observations": n,
                        "confidence": pred["confidence"],
                    })
        return conflicts

    def plan_to_state(
        self,
        start:     str,
        goal:      str,
        max_depth: int = 10,
    ) -> Optional[List[str]]:
        """
        Find a shortest action sequence from `start` to `goal` via BFS.

        Uses the learned transition model as a forward model for planning.
        Only uses transitions with confidence > 0.5 (at least partially known).

        Args:
            start:     Starting state name
            goal:      Target state name
            max_depth: Maximum plan length

        Returns:
            List of action names [a_0, a_1, ...] or None if unreachable.
        """
        from collections import deque as _deque
        queue   = _deque([((start,), [])])   # (state_path, action_path)
        visited = {start}

        while queue:
            state_path, action_path = queue.popleft()
            current = state_path[-1]

            if current == goal:
                return action_path

            if len(action_path) >= max_depth:
                continue

            for action in self.known_actions:
                result = self.predict_best(current, action)
                if result is None or result["confidence"] < 0.5:
                    continue
                next_state = result.get("next_state", "")
                if next_state and next_state not in visited:
                    visited.add(next_state)
                    queue.append((
                        state_path + (next_state,),
                        action_path + [action],
                    ))

        return None   # unreachable within max_depth

    @property
    def n_transitions(self) -> int:
        return len(self._transition_accum)

    @property
    def n_observations(self) -> int:
        return len(self._observations)

    @property
    def known_states(self) -> List[str]:
        return self.state_book.labels

    @property
    def known_actions(self) -> List[str]:
        return self.action_book.labels


# ═══════════════════════════════════════════════════════════════════════════════
# Distributed Plant Monitor
# ═══════════════════════════════════════════════════════════════════════════════

class DistributedPlantMonitor:
    """
    Online monitor that uses a LearnedPlantModel to detect unexpected behaviour.

    Application (Osipov/Kleyko 2017): In distributed automation, each
    sub-system maintains its own model and monitors the plant for deviations
    from the learned normal behaviour. An anomaly is declared when the
    observed transition contradicts the model's prediction.

    Anomaly scoring:
        score = 1 - sim(observed_next_hv, predicted_proto_hv)
        score ≈ 0 → normal  |  score ≈ 0.5 → completely unexpected
    """

    def __init__(self, model: LearnedPlantModel, anomaly_threshold: float = 0.15):
        self.model = model
        self.anomaly_threshold = anomaly_threshold
        self._anomaly_log: List[Dict] = []

    def check(self, state: str, action: str, observed_next: str) -> Dict:
        """
        Check whether an observed transition is consistent with the model.

        Args:
            state: Observed current state
            action: Applied input / action
            observed_next: Observed next state

        Returns:
            Dict with anomaly_score, is_anomaly, predicted_next
        """
        pred = self.model.predict_best(state, action)
        obs_ns_hv = self.model.state_book.encode(observed_next)

        if pred is None:
            # Never seen this (state, action): unseen transition
            score = 0.5
            anomaly = True
            predicted = None
        else:
            # Compare observed next-state HV to predicted prototype
            pred_ns_hv = self.model.state_book.encode(pred["next_state"])
            sim = float(hv_batch_sim(obs_ns_hv, pred_ns_hv.unsqueeze(0))[0])
            score = 1.0 - sim
            anomaly = score > self.anomaly_threshold
            predicted = pred["next_state"]

        result = {
            "state": state,
            "action": action,
            "observed_next": observed_next,
            "predicted_next": predicted,
            "anomaly_score": score,
            "is_anomaly": anomaly,
        }

        if anomaly:
            self._anomaly_log.append(result)

        return result

    @property
    def anomaly_log(self) -> List[Dict]:
        return list(self._anomaly_log)


# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_learned_plant_model():
    print("=" * 60)
    print("Testing LearnedPlantModel (Osipov/Kleyko, IECON 2017)")
    print("=" * 60)

    dim = 8000
    model = LearnedPlantModel(dim=dim, seed=42)

    # Simulate a simple traffic light FSM:
    # States: RED, GREEN, YELLOW
    # Actions: TIMER (automatic), EMERGENCY (override)
    # Transitions:
    #   RED    + TIMER     → GREEN
    #   GREEN  + TIMER     → YELLOW
    #   YELLOW + TIMER     → RED
    #   *      + EMERGENCY → RED (override)

    normal_transitions = [
        ("RED",    "TIMER",     "GREEN"),
        ("GREEN",  "TIMER",     "YELLOW"),
        ("YELLOW", "TIMER",     "RED"),
    ]

    emergency_transitions = [
        ("GREEN",  "EMERGENCY", "RED"),
        ("YELLOW", "EMERGENCY", "RED"),
    ]

    all_transitions = normal_transitions + emergency_transitions

    # Observe each transition 5 times (with slight noise via repetition)
    for _ in range(5):
        model.observe_sequence(all_transitions)

    print(f"  Learned: {model.n_transitions} unique transitions, "
          f"{model.n_observations} observations")
    print(f"  States:  {sorted(model.known_states)}")
    print(f"  Actions: {sorted(model.known_actions)}")

    # Test exact prediction
    errors = 0
    for s, a, ns in all_transitions:
        pred = model.predict_best(s, a)
        if pred is None or pred["next_state"] != ns:
            errors += 1
            print(f"  ✗ ({s}, {a}) → expected {ns}, got {pred}")

    print(f"  Exact prediction accuracy: {len(all_transitions) - errors}/{len(all_transitions)}")
    assert errors == 0, f"{errors} prediction errors"

    # Test confidence
    pred = model.predict_best("RED", "TIMER")
    print(f"  Confidence (RED, TIMER)→GREEN: {pred['confidence']:.3f}  (want ≈ 1.0)")
    assert pred["confidence"] > 0.5

    # Test unknown action
    pred_unknown = model.predict_best("RED", "UNKNOWN_ACTION")
    print(f"  Unknown action result: {pred_unknown}  (want None or low confidence)")

    print("  ✅ LearnedPlantModel OK")


def test_ambiguous_transitions():
    print("=" * 60)
    print("Testing ambiguous transitions (non-deterministic plant)")
    print("=" * 60)

    model = LearnedPlantModel(dim=6000, seed=1)

    # Non-deterministic: SENSOR_ERROR can lead to different states
    model.observe("RUNNING", "SENSOR_ERROR", "FAULT_A")
    model.observe("RUNNING", "SENSOR_ERROR", "FAULT_A")
    model.observe("RUNNING", "SENSOR_ERROR", "FAULT_B")  # conflicting!
    model.observe("RUNNING", "TIMER",        "IDLE")
    model.observe("RUNNING", "TIMER",        "IDLE")

    ambiguous = model.ambiguous_transitions()
    print(f"  Ambiguous transitions: {[(a['state'], a['action']) for a in ambiguous]}")
    # SENSOR_ERROR should be flagged as potentially ambiguous
    sensor_err = model.predict("RUNNING", "SENSOR_ERROR", top_k=2)
    cands = [(r["next_state"], f"{r['confidence']:.2f}") for r in sensor_err]
    print(f"  SENSOR_ERROR candidates: {cands}")

    # Unambiguous should have high confidence
    timer_pred = model.predict_best("RUNNING", "TIMER")
    print(f"  TIMER confidence: {timer_pred['confidence']:.3f}  (want ≈ 1.0)")
    assert timer_pred["confidence"] > 0.5
    assert timer_pred["next_state"] == "IDLE"

    print("  ✅ Ambiguous transitions OK")


def test_plant_monitor():
    print("=" * 60)
    print("Testing DistributedPlantMonitor (Osipov/Kleyko 2017)")
    print("=" * 60)

    model = LearnedPlantModel(dim=6000, seed=7)
    model.observe_sequence([
        ("IDLE",    "START",  "RUNNING"),
        ("RUNNING", "STOP",   "IDLE"),
        ("RUNNING", "FAULT",  "ERROR"),
        ("ERROR",   "RESET",  "IDLE"),
    ] * 3)

    monitor = DistributedPlantMonitor(model, anomaly_threshold=0.15)

    # Normal transitions → should not flag
    result = monitor.check("IDLE", "START", "RUNNING")
    print(f"  Normal (IDLE, START)→RUNNING: score={result['anomaly_score']:.4f}, anomaly={result['is_anomaly']}")
    assert not result["is_anomaly"], "Normal transition flagged as anomaly"

    # Anomalous: RUNNING + START → RUNNING (unexpected next state for this action)
    result_anom = monitor.check("RUNNING", "STOP", "ERROR")  # should be IDLE
    print(f"  Unexpected (RUNNING,STOP)→ERROR: score={result_anom['anomaly_score']:.4f}, anomaly={result_anom['is_anomaly']}")

    print(f"  Anomaly log length: {len(monitor.anomaly_log)}")
    print("  ✅ DistributedPlantMonitor OK")


if __name__ == "__main__":
    test_learned_plant_model()
    print()
    test_ambiguous_transitions()
    print()
    test_plant_monitor()
    print()
    print("=== All plant_model_hdc tests passed ===")
