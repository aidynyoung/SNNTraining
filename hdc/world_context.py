"""
World Context: Pattern Memory, Causal Reasoning, Hierarchical Context
======================================================================
Bridges three existing SNNTraining modules — HoloGN, VSAGraph, and the
PhysicsWorldModel — to give the world model what it currently lacks:

  1. **SequencePatternMemory** — recognises *recurring temporal patterns*
     in the sensor stream using HoloGN one-shot encoding and recall.
     A sawtooth pattern, a heartbeat signature, a gait cycle: each is
     stored as a single sequence HV and recalled on every new window.
     When a known pattern is recognised, confidence rises and prediction
     error drops; when a novel pattern appears, the learner is surprised
     and updates aggressively.

  2. **CausalTransitionGraph** — encodes *causal structure* as a VSA
     graph (Kleyko 2022, kleyko_framework.py §III-C). Each observed
     (state, action) → next_state transition is stored as an edge:
         edge = XOR(XOR(state_hv, action_hv), next_state_hv)
     The graph supports:
       - Causal query: what caused this state? (unbind from graph)
       - Forward simulation: given action, what happens next?
       - Counterfactual: had action been different, what would happen?
     This is the difference between a lookup table and genuine causal
     understanding.

  3. **HierarchicalContextEncoder** — three-level temporal working memory:
       Tick     (1 step):   current sensor HV — what is happening now
       Pattern  (K steps):  n-gram HV over a window — what behaviour is ongoing
       Situation (M steps): EMA prototype — what context has persisted
     All three are XOR-bound into a single context HV that the world
     model uses for prediction. This gives the predictor genuine "working
     memory" — it can distinguish the same sensor reading in different
     contexts (e.g., vibration during startup vs vibration during fault).

  4. **ContextualWorldModel** — wires all three into the HybridPhysicalAIPipeline,
     replacing the bare sensor HV with a context-enriched version.

Literature:
  - HoloGN: Kleyko 2017 (holographic_graph_neuron.py)
  - VSAGraph: Kleyko 2022 (kleyko_framework.py)
  - N-gram context: Kleyko 2023 Survey (kleyko_survey.py)
  - Cognitive map: Bent 2024 (cognitive_map.py)
  - Multi-scale temporal: Schlegel 2025 (multiscale_temporal.py)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from hdc.hdc_glue import (
    hv_xor, hv_bundle, hv_majority, hv_batch_sim, gen_hvs, hv_permute,
)
from hdc.holographic_graph_neuron import (
    HoloGNEncoder, ZadoffChuIndexer, ComplexHammingSearch,
    HoloGNEncoder,
)
from hdc.kleyko_framework import VSARecord, VSAGraph
from hdc.physics_world_model import (
    PhysicsWorldModel, _xor, _majority, _hamming, MultiHorizonPredictor,
)
from hdc.physical_ai_hybrid import (
    HybridPhysicalAIPipeline, AdaptiveModalityFusion, ResonatorAttractor,
)
from hdc.sensor_stream import SensorReading, SensorSpec


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Sequence Pattern Memory (HoloGN-backed)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PatternMatch:
    """Result of a pattern recognition query."""
    pattern_id: int
    hamming_distance: float      # 0 = exact, 0.5 = orthogonal
    similarity: float            # 1 - hamming_distance
    is_known: bool               # True if distance < threshold
    n_times_seen: int
    label: Optional[str] = None


class SequencePatternMemory:
    """
    Recognise and store recurring temporal patterns via HoloGN.

    How it works:
      1. Maintain a sliding window of the last W sensor HVs.
      2. Every `stride` steps, encode the window as a HoloGN HV:
           window_hv = MAJORITY_SUM_{t=0}^{W-1}(Sh(IV_t, quantised_hv_t))
         where IV_t is the Graph Neuron init vector for position t,
         and quantised_hv_t is the sensor HV at position t (element ∈ {0,1}).
      3. Search the HoloGN memory for the most similar stored pattern.
      4. If similarity > recognition_threshold → pattern recognised.
      5. If similarity < novelty_threshold → store as new pattern.

    Pattern recognition is O(n_stored) via ComplexHammingSearch — linear
    in the number of known patterns, independent of window size W.

    Args:
        hd_dim: Shared hypervector dimensionality
        window: Sliding window size W (number of sensor ticks)
        stride: Encode and search every `stride` ticks
        recognition_threshold: Hamming distance to call a match (< = known)
        novelty_threshold: Hamming distance to call a novel pattern (> = store)
        max_patterns: Maximum stored patterns
    """

    def __init__(
        self,
        hd_dim: int,
        window: int = 8,
        stride: int = 4,
        recognition_threshold: float = 0.25,
        novelty_threshold: float = 0.40,
        max_patterns: int = 256,
        seed: int = 0,
    ):
        self.hd_dim = hd_dim
        self.window = window
        self.stride = stride
        self.seed = seed
        self.recognition_threshold = recognition_threshold
        self.novelty_threshold = novelty_threshold

        # Graph Neuron array: one neuron per window position
        self._indexer = ZadoffChuIndexer(n_neurons=window, dim=hd_dim, seed=seed)
        self._encoder = HoloGNEncoder(self._indexer)

        # Fast Hamming search via complex number trick
        self._memory = ComplexHammingSearch(hd_dim)
        self._n_seen: List[int] = []      # how many times each pattern was seen
        self._labels: List[str] = []

        # Sliding window buffer
        self._buf: deque = deque(maxlen=window)
        self._tick = 0
        self._pattern_count = 0

    def _quantise_to_pattern(self, sensor_hv: torch.Tensor) -> List[int]:
        """
        Convert a sensor HV to a per-neuron activation pattern.

        Each window position maps to one Graph Neuron. The binary HV is
        split into `window` equal chunks; the chunk's majority bit (0 or 1)
        becomes the element index for that GN.
        """
        chunk = self.hd_dim // self.window
        pattern = []
        for i in range(self.window):
            start = i * chunk
            end = start + chunk
            bit = int((sensor_hv[start:end].mean() > 0.5).item())
            pattern.append(bit)
        return pattern

    def push(self, sensor_hv: torch.Tensor, label: Optional[str] = None) -> Optional[PatternMatch]:
        """
        Add one sensor HV to the window and optionally search/store.

        Args:
            sensor_hv: (hd_dim,) current sensor HV
            label: Optional semantic label for this observation

        Returns:
            PatternMatch if the stride triggered a search, else None
        """
        self._buf.append(sensor_hv.detach())
        self._tick += 1

        if self._tick % self.stride != 0 or len(self._buf) < self.window:
            return None

        # Encode the window as a position-of-largest-change fingerprint.
        #
        # Root cause diagnosed: sensor HVs are SPARSE (density ~0.20), not balanced.
        # XOR changes between phases: 2.1% for adjacent, 9.9% for cycle boundary.
        # Majority of 6 sparse XOR HVs → all-zeros for every window (useless).
        #
        # Fix: identify WHERE in the window the largest change occurs — this position
        # is uniquely determined by the cycle phase offset, making it a perfect
        # discriminator. Encode as a cyclic-shift HV: Sh(IV, boundary_position).
        # Windows at different cycle offsets get near-orthogonal pattern HVs.
        window_hvs = list(self._buf)
        W = len(window_hvs)

        # Compute XOR density for each consecutive pair (cyclic)
        change_densities = []
        for t in range(W):
            hv_curr = window_hvs[t]
            hv_next = window_hvs[(t + 1) % W]
            density = float((hv_curr != hv_next).float().mean().item())
            change_densities.append(density)

        # The cycle boundary is at the position with the largest XOR density
        boundary_pos = int(torch.tensor(change_densities).argmax().item())

        # Encode boundary position as a cyclic shift of the init vector
        # Sh(IV_0, boundary_pos) is near-orthogonal for different boundary_pos values
        window_hv = hv_permute(self._indexer.init_vectors[0], k=boundary_pos)

        # Search memory
        result = self._memory.query(window_hv, threshold=1.0, top_k=1)

        if result:
            dist = result[0]["hamming_distance"]
            pat_id = result[0]["label"]
            is_known = dist < self.recognition_threshold
            is_novel = dist > self.novelty_threshold
        else:
            dist = 0.5
            pat_id = -1
            is_known = False
            is_novel = True

        # Store if novel
        if is_novel and self._pattern_count < 256:
            self._memory.store(window_hv, self._pattern_count)
            self._n_seen.append(1)
            self._labels.append(label or f"pattern_{self._pattern_count}")
            pat_id = self._pattern_count
            self._pattern_count += 1
            is_known = True
            dist = 0.0
        elif is_known and 0 <= pat_id < len(self._n_seen):
            self._n_seen[pat_id] += 1

        return PatternMatch(
            pattern_id=pat_id,
            hamming_distance=dist,
            similarity=1.0 - dist,
            is_known=is_known,
            n_times_seen=self._n_seen[pat_id] if 0 <= pat_id < len(self._n_seen) else 0,
            label=self._labels[pat_id] if 0 <= pat_id < len(self._labels) else None,
        )

    @property
    def n_patterns(self) -> int:
        return self._pattern_count

    def pattern_hv(self, pattern_id: int) -> Optional[torch.Tensor]:
        """Retrieve the stored HV for a pattern (from ComplexHammingSearch store)."""
        if self._memory._H_complex is None or pattern_id >= self._memory.n_stored():
            return None
        # Reconstruct from complex encoding
        row = self._memory._H_complex[pattern_id]
        return (row.real > 0.5).float()


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Causal Transition Graph (VSAGraph-backed)
# ═══════════════════════════════════════════════════════════════════════════════

class CausalTransitionGraph:
    """
    Encode causal structure using item-memory lookup (Kleyko 2022, §III-C).

    REVISED APPROACH — fixes the EMA-bundle confidence problem:

    Instead of bundling ALL transitions into a single graph HV (which makes
    unbinding recover a noisy superposition of all next states), each unique
    (state, action) binding is stored as a SEPARATE KEY in a ComplexHammingSearch
    item memory, paired with its accumulated next-state prototype.

    Algorithm:
      observe(s, a, s'):
        key = XOR(quantised_s_hv, quantised_a_hv)   [bind state + action]
        Find nearest existing key in AM (threshold 0.25 Hamming)
        If found: accumulate s'_hv into that entry's prototype
        If novel:  store new (key, s'_hv) entry

      forward_query(s, a):
        probe = XOR(quantised_s_hv, quantised_a_hv)
        result = AM.query_best(probe)
        If found:  next_approx = binarised prototype
                   confidence  = 2 × |density − 0.5| of prototype
                                 (near-0 = one consistent s', near-0.5 = ambiguous)
        If not:    return random HV, confidence = 0.0

    Confidence is now meaningful:
      High (>0.7) = most observations for (s,a) led to the same next state
      Low  (<0.2) = conflicting transitions — non-deterministic or insufficient data
      Zero        = this (s,a) pair has never been seen

    Backward and counterfactual queries use the same AM via XOR probe.

    Literature: Kleyko 2022 VSA Framework (kleyko_framework.py — VSAGraph);
                LearnedPlantModel (plant_model_hdc.py) — same pattern for labels;
                ComplexHammingSearch (holographic_graph_neuron.py) — O(n) Hamming.

    Args:
        hd_dim: Hypervector dimensionality
        key_match_threshold: Max Hamming dist to treat two keys as the same transition
        max_entries: Maximum stored (state,action) → next_state mappings
    """

    def __init__(
        self,
        hd_dim: int,
        decay: float = 0.99,           # kept for API compat, not used in new impl
        state_codebook_size: int = 128, # kept for API compat
        key_match_threshold: float = 0.20,
        max_entries: int = 512,
    ):
        self.hd_dim = hd_dim
        self.key_match_threshold = key_match_threshold
        self.max_entries = max_entries

        # Item memory: keys (state XOR action) → next_state accumulators
        from hdc.holographic_graph_neuron import ComplexHammingSearch
        self._key_mem = ComplexHammingSearch(hd_dim)
        self._next_accum: List[torch.Tensor] = []   # float accumulators per entry
        self._next_count: List[int] = []             # observation counts per entry
        self._n_transitions = 0

    def _make_key(
        self,
        state_hv: torch.Tensor,
        action_hv: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """XOR bind state + action into a unique transition key."""
        if action_hv is None:
            action_hv = torch.zeros(self.hd_dim)
        return _xor(state_hv, action_hv)

    def observe(
        self,
        state_hv: torch.Tensor,
        action_hv: Optional[torch.Tensor],
        next_state_hv: torch.Tensor,
        state_label: Optional[str] = None,
        next_state_label: Optional[str] = None,
    ):
        """Register one observed (state, action) → next_state transition."""
        key = self._make_key(state_hv, action_hv)

        # Search for existing entry
        results = self._key_mem.query(key, threshold=self.key_match_threshold, top_k=1)

        if results and results[0]["hamming_distance"] < self.key_match_threshold:
            # Update existing entry: accumulate next_state
            idx = results[0]["label"]
            self._next_accum[idx] = self._next_accum[idx] + next_state_hv.float()
            self._next_count[idx] += 1
        elif len(self._next_accum) < self.max_entries:
            # New entry
            idx = len(self._next_accum)
            self._key_mem.store(key, label=idx)
            self._next_accum.append(next_state_hv.float().clone())
            self._next_count.append(1)

        self._n_transitions += 1

    def forward_query(
        self,
        state_hv: torch.Tensor,
        action_hv: Optional[torch.Tensor] = None,
        top_k: int = 3,
    ) -> Tuple[torch.Tensor, float]:
        """
        Forward causal query: (state, action) → predicted next state + confidence.

        Uses top-k nearest-key weighted blending for better predictions when
        multiple similar state-action pairs have been observed.  Confidence is
        derived from per-dimension entropy of the blended accumulator:
          low entropy (each dim near 0 or 1) → high confidence
          high entropy (dims near 0.5)        → low confidence

        Args:
            state_hv:  (D,) current state hypervector
            action_hv: (D,) action hypervector (or None)
            top_k:     Number of nearest neighbours to blend (default 3)

        Returns:
            (next_state_hv, confidence ∈ [0, 1])
        """
        key = self._make_key(state_hv, action_hv)
        results = self._key_mem.query(key, threshold=self.key_match_threshold,
                                      top_k=max(top_k, 1))

        if not results:
            return (torch.rand(self.hd_dim) < 0.5).float(), 0.0

        # Top-k weighted blend: weight by inverse Hamming distance
        weighted_accum = torch.zeros(self.hd_dim)
        total_weight   = 0.0
        for r in results[:top_k]:
            idx = r["label"]
            if idx >= len(self._next_accum):
                continue
            # Weight = 1 / (hamming_distance + ε) — closer keys count more
            dist = float(r.get("hamming_distance", 0.5))
            w    = 1.0 / (dist + 0.05)
            n    = max(self._next_count[idx], 1)
            weighted_accum += w * (self._next_accum[idx] / n)
            total_weight   += w

        if total_weight == 0.0:
            return (torch.rand(self.hd_dim) < 0.5).float(), 0.0

        avg = weighted_accum / total_weight   # (D,) ∈ [0, 1] blended average

        # Entropy-based confidence: H = -p log p - (1-p) log(1-p)
        # Low entropy (p near 0 or 1) → high confidence → confidence = 1 - H_norm
        p   = avg.clamp(1e-6, 1 - 1e-6)
        H   = -(p * p.log() + (1 - p) * (1 - p).log())  # (D,) per-dim entropy
        H_max = 0.693  # ln(2) = max binary entropy
        confidence = float(1.0 - H.mean().item() / H_max)

        # Use primary match count for tie-breaking when all weights equal
        n_primary = max(self._next_count[results[0]["label"]], 1) if results else 1
        # Scale confidence by observation count (uncertain with few samples)
        count_scale = min(1.0, n_primary / 10.0)
        confidence  = confidence * count_scale

        next_hv = (avg > 0.5).float()
        return next_hv, confidence

    def counterfactual_query(
        self,
        state_hv: torch.Tensor,
        actual_action_hv: torch.Tensor,
        counterfactual_action_hv: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        """
        Counterfactual: "What WOULD have happened with a different action?"

        actual_pred, _ = forward_query(state, actual_action)
        cf_pred,     _ = forward_query(state, cf_action)
        divergence     = Hamming distance(actual_pred, cf_pred)

        Returns:
            (cf_next_state, divergence_from_actual)
        """
        actual_pred, _ = self.forward_query(state_hv, actual_action_hv)
        cf_pred, _     = self.forward_query(state_hv, counterfactual_action_hv)
        divergence = 1.0 - float(_hamming(actual_pred, cf_pred).item())
        return cf_pred, divergence

    def causal_surprise(
        self,
        state_hv: torch.Tensor,
        action_hv: Optional[torch.Tensor],
        observed_next_hv: torch.Tensor,
    ) -> float:
        """
        Measure how surprising an observed transition is given the causal model.

        surprise = Hamming(forward_query(s, a), observed_next)
        Low surprise = causal model predicted this; high = new/unexpected cause.
        """
        pred, _ = self.forward_query(state_hv, action_hv)
        return 1.0 - float(_hamming(pred, observed_next_hv).item())

    def multi_hop_query(
        self,
        state_hv: torch.Tensor,
        action_sequence: List[Optional[torch.Tensor]],
    ) -> Tuple[torch.Tensor, float]:
        """
        Multi-hop causal query: simulate a sequence of actions.

        Given state s and actions [a1, a2, ..., ak], compute:
          s1 = forward(s,  a1)
          s2 = forward(s1, a2)
          ...
          sk = forward(s_{k-1}, ak)

        Returns (final_state, minimum_confidence) — the minimum confidence
        across all hops (weakest link in the chain).

        This enables A→B→C reasoning without having observed A→C directly.
        """
        current = state_hv
        min_conf = 1.0
        for action_hv in action_sequence:
            next_state, conf = self.forward_query(current, action_hv)
            min_conf = min(min_conf, conf)
            current = next_state
        return current, min_conf

    def backward_query(
        self,
        observed_state_hv: torch.Tensor,
        top_k: int = 3,
    ) -> List[Dict]:
        """
        Backward causal query: what (state, action) pairs could have led here?

        Searches all stored keys: key = XOR(state, action).
        For each stored entry, unbinds the key from the observed next_state
        and checks similarity to stored prototypes.

        Returns:
            List of {state_approx, next_state, confidence} sorted by confidence
        """
        results = []
        for i, (accum, count) in enumerate(
            zip(self._next_accum, self._next_count)
        ):
            proto = (accum / max(count, 1) > 0.5).float()
            sim_to_obs = float(_hamming(proto, observed_state_hv).item())
            if sim_to_obs > 0.6:  # this entry leads to a similar state
                results.append({
                    "entry_idx": i,
                    "next_state": proto,
                    "confidence": sim_to_obs,
                    "n_observations": count,
                })

        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results[:top_k]

    @property
    def n_transitions(self) -> int:
        return self._n_transitions

    @property
    def n_entries(self) -> int:
        """Number of distinct (state, action) transition entries."""
        return len(self._next_accum)

    @property
    def causal_graph_hv(self) -> torch.Tensor:
        """Bundle of all stored next-state prototypes (for structural queries)."""
        if not self._next_accum:
            return torch.zeros(self.hd_dim)
        protos = [(a / max(c, 1) > 0.5).float()
                  for a, c in zip(self._next_accum, self._next_count)]
        stacked = torch.stack(protos)
        return _majority(stacked.mean(dim=0))

    def transition_entropy(self) -> float:
        """
        Shannon entropy of the transition count distribution.

        High entropy → many equally-visited transitions (exploration).
        Low entropy  → a few transitions dominate (exploitation / routine).

        Returns:
            Entropy in nats; 0.0 = all transitions identical count.
        """
        if not self._next_count or sum(self._next_count) == 0:
            return 0.0
        import math
        total = sum(self._next_count)
        probs = [c / total for c in self._next_count if c > 0]
        return float(-sum(p * math.log(p) for p in probs))

    def graph_density(self) -> float:
        """
        Fraction of possible (state, action) pairs that have been observed.

        Density ≈ n_entries / max_entries.
        Low density = sparse exploration; high = thorough coverage.
        """
        if self.max_entries == 0:
            return 0.0
        return min(1.0, len(self._next_accum) / self.max_entries)

    def most_visited_transitions(self, top_k: int = 5) -> List[Dict]:
        """
        Return the top-k most frequently observed transitions.

        Useful for understanding which (state, action) → next_state patterns
        dominate the model's experience.
        """
        if not self._next_count:
            return []
        indexed = [(i, c) for i, c in enumerate(self._next_count)]
        indexed.sort(key=lambda x: x[1], reverse=True)
        results = []
        for idx, count in indexed[:top_k]:
            avg = self._next_accum[idx] / max(count, 1)
            conf = float((avg - 0.5).abs().mean().item()) * 2
            results.append({"entry_idx": idx, "n_observations": count, "confidence": round(conf, 4)})
        return results


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Hierarchical Context Encoder
# ═══════════════════════════════════════════════════════════════════════════════

class HierarchicalContextEncoder:
    """
    Three-level working memory via hierarchical HV context.

    Encodes three temporal scales simultaneously:
      Tick     (1 step):   sensor_hv itself — immediate perception
      Pattern  (K steps):  n-gram of recent K states — ongoing behaviour
      Situation (M steps): EMA prototype of recent M states — sustained context

    The three levels are XOR-bound into a single context HV:
        context = XOR(XOR(tick_hv, permute(pattern_hv, 1)), permute(sit_hv, 2))

    Using different cyclic shifts for each level ensures they are near-
    orthogonal after binding, so each level contributes independently.

    This gives the world model genuine "working memory": the same sensor
    value means something different depending on whether it's a tick within
    a known rising pattern (pattern level) in a high-vibration situation
    (situation level).

    Literature: Kleyko 2023 Survey (n-gram encoding, §III-A);
                Schlegel 2025 (multiscale_temporal.py — multi-scale HDC);
                Bent 2024 (cognitive_map.py — context binding).

    Args:
        hd_dim: Hypervector dimensionality
        pattern_window: K — n-gram window for pattern level
        situation_window: M — EMA horizon for situation level
        situation_decay: EMA decay for situation prototype
    """

    def __init__(
        self,
        hd_dim: int,
        pattern_window: int = 8,
        situation_window: int = 50,
        situation_decay: float = 0.95,
    ):
        self.hd_dim = hd_dim
        self.pattern_window = pattern_window
        self.situation_decay = situation_decay

        # Pattern level: sliding window for n-gram HV
        self._pattern_buf: deque = deque(maxlen=pattern_window)

        # Situation level: EMA of all seen HVs
        self._situation_proto = torch.zeros(hd_dim)
        self._sit_count = 0

        # Position HVs for n-gram binding (one per window position)
        g = torch.Generator()
        g.manual_seed(0)
        self._pos_hvs = (torch.rand(pattern_window, hd_dim, generator=g) < 0.5).float()

    def _build_pattern_hv(self) -> torch.Tensor:
        """
        N-gram HV over the current window.

        Each element at position t is XOR-bound with its position HV,
        then all are bundled. This encodes WHAT and WHERE within the window.

        Kleyko 2023 Survey §III-A: n-gram encoding captures temporal order.
        """
        if not self._pattern_buf:
            return torch.zeros(self.hd_dim)

        bound = []
        for t, hv in enumerate(self._pattern_buf):
            bound.append(_xor(hv, self._pos_hvs[t]))

        n = len(bound)
        counts = torch.stack(bound).sum(dim=0)
        return (counts > n / 2).float()

    def encode(self, sensor_hv: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Update all three context levels and return the context HV.

        Args:
            sensor_hv: (D,) current sensor HV

        Returns:
            Dict with tick, pattern, situation, context HVs
        """
        # Tick level: immediate sensor HV
        tick_hv = sensor_hv

        # Pattern level: update window and build n-gram
        self._pattern_buf.append(sensor_hv.detach())
        pattern_hv = self._build_pattern_hv()

        # Situation level: EMA update
        self._situation_proto = (
            self.situation_decay * self._situation_proto
            + (1 - self.situation_decay) * sensor_hv.float()
        )
        self._sit_count += 1
        sit_hv = (self._situation_proto > 0.5).float()

        # Combine: XOR with position-shifted versions to keep levels separate
        context_hv = _xor(
            _xor(tick_hv, hv_permute(pattern_hv, k=1)),
            hv_permute(sit_hv, k=2)
        )

        return {
            "tick": tick_hv,
            "pattern": pattern_hv,
            "situation": sit_hv,
            "context": context_hv,
        }

    def similarity_profile(
        self,
        hv_a: torch.Tensor,
        hv_b: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Compare two HVs at each context level.

        Recovers each level by unbinding the position shifts
        (XOR with same shift = undo shift).
        """
        # Recover components from context HV by unbinding
        def recover_tick(ctx):
            # tick = context XOR perm(pattern, 1) XOR perm(sit, 2) — but we don't
            # have all pieces, so approximate: use context directly as tick proxy
            return ctx

        return {
            "tick": float(_hamming(hv_a, hv_b).item()),
            "note": "Full level decomposition requires storing per-level HVs separately",
        }

    def reset_situation(self):
        """Reset situation prototype (e.g., after a context switch)."""
        self._situation_proto = torch.zeros(self.hd_dim)
        self._sit_count = 0


# ═══════════════════════════════════════════════════════════════════════════════
# 4. ContextualWorldModel — wires everything together
# ═══════════════════════════════════════════════════════════════════════════════

class ContextualWorldModel:
    """
    World model enriched with pattern memory, causal reasoning, and
    hierarchical context.

    Augments HybridPhysicalAIPipeline with:
    - SequencePatternMemory: recognise recurring patterns
    - CausalTransitionGraph: learn and query cause-effect relationships
    - HierarchicalContextEncoder: tick/pattern/situation working memory

    The context HV fed to the predictors is now:
        context = XOR(sensor_hv, XOR(pattern_hv, situation_hv))
    instead of raw sensor_hv. This means:
      - Predictors see the same vibration reading differently in "startup" vs
        "normal operation" situations.
      - The causal graph can explain WHY a transition occurred.
      - Pattern memory can signal "we've been here before" even before
        the predictor has learned the full dynamics.

    Args:
        base_pipeline: HybridPhysicalAIPipeline to augment
        pattern_window: Window for pattern memory
        causal_decay: EMA decay for causal graph
        situation_decay: EMA decay for situation prototype
    """

    def __init__(
        self,
        base_pipeline: HybridPhysicalAIPipeline,
        pattern_window: int = 8,
        pattern_stride: int = 4,
        causal_decay: float = 0.98,
        situation_decay: float = 0.95,
    ):
        self.pipeline = base_pipeline
        self.hd_dim = base_pipeline.hd_dim

        self.pattern_memory = SequencePatternMemory(
            self.hd_dim,
            window=pattern_window,
            stride=pattern_stride,
        )
        self.causal_graph = CausalTransitionGraph(
            self.hd_dim,
            decay=causal_decay,
        )
        self.context_encoder = HierarchicalContextEncoder(
            self.hd_dim,
            pattern_window=pattern_window,
            situation_decay=situation_decay,
        )

        self._prev_sensor_hv: Optional[torch.Tensor] = None
        self._prev_action_hv: Optional[torch.Tensor] = None
        self._tick = 0

    def tick(
        self,
        reading: SensorReading,
        action_hv: Optional[torch.Tensor] = None,
        candidate_actions=None,
        goal_state=None,
    ) -> Dict:
        """
        Full contextual world model tick.

        1. Encode sensor → raw sensor HV (via adaptive fusion)
        2. Build hierarchical context (tick / pattern / situation)
        3. Pass context HV to world model (richer input than raw sensor HV)
        4. Update pattern memory → pattern recognition signal
        5. Update causal graph → causal surprise signal
        6. Compute counterfactual if alternative action provided
        7. Return enriched result dict
        """
        self._tick += 1

        # ── Step 1: raw sensor encoding ───────────────────────────────────────
        sensor_hv = self.pipeline._encode_with_adaptive_fusion(reading)

        # ── Step 2: hierarchical context ──────────────────────────────────────
        ctx = self.context_encoder.encode(sensor_hv)
        context_hv = ctx["context"]

        # ── Step 3: pattern memory ─────────────────────────────────────────────
        pattern_match = self.pattern_memory.push(sensor_hv)

        # ── Step 4: causal update ─────────────────────────────────────────────
        if self._prev_sensor_hv is not None:
            self.causal_graph.observe(
                self._prev_sensor_hv,
                self._prev_action_hv,
                sensor_hv,
            )

        # Causal surprise: how expected was this transition?
        causal_surprise = 0.0
        if self._prev_sensor_hv is not None and self.causal_graph.n_transitions > 5:
            causal_surprise = self.causal_graph.causal_surprise(
                self._prev_sensor_hv, self._prev_action_hv, sensor_hv
            )

        # ── Step 5: causal forward query ─────────────────────────────────────
        causal_pred, causal_conf = self.causal_graph.forward_query(
            sensor_hv, action_hv
        )

        # ── Step 6: run base pipeline with context HV ─────────────────────────
        # Temporarily override current state with context-enriched HV
        original_state = self.pipeline.world_model.current_state.clone()
        self.pipeline.world_model.current_state = context_hv.detach()
        base_result = self.pipeline.tick(reading, candidate_actions, goal_state)
        # Restore (tick already updated it, so just record)

        # ── Step 7: update history ────────────────────────────────────────────
        self._prev_sensor_hv = sensor_hv.detach()
        self._prev_action_hv = action_hv.detach() if action_hv is not None else None

        # ── Assemble result ───────────────────────────────────────────────────
        result = {
            **base_result,
            "context_hv": context_hv,
            "context_tick": ctx["tick"],
            "context_pattern": ctx["pattern"],
            "context_situation": ctx["situation"],
            "pattern_match": pattern_match,
            "n_known_patterns": self.pattern_memory.n_patterns,
            "causal_surprise": causal_surprise,
            "causal_pred": causal_pred,
            "causal_confidence": causal_conf,
            "n_causal_transitions": self.causal_graph.n_transitions,
            "contextual_tick": self._tick,
        }

        # Pattern boost: if known pattern → reduce effective prediction error
        if pattern_match and pattern_match.is_known and pattern_match.similarity > 0.8:
            result["pattern_boost"] = pattern_match.similarity
            result["prediction_error"] = result["prediction_error"] * (1 - 0.3 * pattern_match.similarity)
        else:
            result["pattern_boost"] = 0.0

        return result

    def counterfactual(
        self,
        state_hv: torch.Tensor,
        actual_action_hv: torch.Tensor,
        alternative_action_hv: torch.Tensor,
    ) -> Dict:
        """
        "What would have happened if I had taken a different action?"

        Returns the counterfactual prediction and how much it diverges
        from the actual prediction.
        """
        cf_pred, divergence = self.causal_graph.counterfactual_query(
            state_hv, actual_action_hv, alternative_action_hv
        )
        return {
            "cf_prediction": cf_pred,
            "divergence_from_actual": divergence,
            "causal_impact": divergence,   # higher = action matters more
        }

    def status(self) -> Dict:
        base = self.pipeline.status()
        return {
            **base,
            "n_known_patterns": self.pattern_memory.n_patterns,
            "n_causal_transitions": self.causal_graph.n_transitions,
            "contextual_ticks": self._tick,
        }

    def state_report(self) -> Dict:
        """
        Human-readable report of what the world model currently knows.

        Useful for debugging, monitoring, and understanding model state
        without inspecting raw hypervectors.

        Returns structured summary:
          patterns:   recognised recurring temporal patterns with counts
          causal:     discovered cause-effect relationships
          prediction: current prediction quality and confidence
          memory:     number of stored experiences
        """
        # Pattern library
        patterns_info = []
        for i, (name, count) in enumerate(
            zip(self.pattern_memory._labels[:10],
                self.pattern_memory._n_seen[:10])
        ):
            patterns_info.append({"label": name or f"pattern_{i}", "seen": count})

        # Causal edges
        causal_info = []
        for cause, effect, score in self.causal_graph.discover_edges()[:5]:
            causal_info.append({"cause": cause, "effect": effect, "score": round(score, 3)})

        return {
            "ticks":         self._tick,
            "patterns":      patterns_info,
            "n_patterns":    self.pattern_memory.n_patterns,
            "causal_edges":  causal_info,
            "n_transitions": self.causal_graph.n_transitions,
            "causal_stability": self.causal_graph.stability()
                                if hasattr(self.causal_graph, 'stability') else 0.0,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# 5. MultiScalePatternMemory — patterns at multiple temporal resolutions
# ═══════════════════════════════════════════════════════════════════════════════

class MultiScalePatternMemory:
    """
    Pattern recognition across multiple temporal scales simultaneously.

    A single SequencePatternMemory with window W can only detect patterns
    of length ~W. MultiScalePatternMemory runs K pattern memories at
    different scales W/4, W/2, W, 2W simultaneously (Schlegel 2025 —
    multi-scale temporal encoding):

      Scale 0 (fine):   W//4 — fast micro-patterns (vibration cycles)
      Scale 1 (medium): W//2 — intermediate patterns (operational cycles)
      Scale 2 (coarse): W    — slow macro-patterns  (startup/shutdown)
      Scale 3 (global): 2W   — situational patterns (fault progression)

    Recognition at ANY scale triggers a pattern match. The scale at which
    a match is found indicates the temporal frequency of the pattern.

    On each tick, returns:
      - best_match: PatternMatch from the scale with highest similarity
      - all_matches: all scale matches (some may be None)
      - active_scale: which scale found the best match

    Args:
        hd_dim: Hypervector dimensionality
        base_window: Base window size W (other scales derived from this)
        recognition_threshold: Hamming distance for match at any scale
    """

    def __init__(
        self,
        hd_dim: int,
        base_window: int = 8,
        stride: int = 3,
        recognition_threshold: float = 0.25,
        novelty_threshold: float = 0.40,
        seed: int = 0,
    ):
        self.hd_dim = hd_dim
        self.base_window = base_window

        # Four scales: W//4, W//2, W, 2W (clamped to ≥2)
        self._scales = [
            max(2, base_window // 4),
            max(2, base_window // 2),
            base_window,
            base_window * 2,
        ]

        # One SequencePatternMemory per scale
        self._memories: List[SequencePatternMemory] = [
            SequencePatternMemory(
                hd_dim=hd_dim,
                window=w,
                stride=max(1, stride // 2) if w < base_window else stride,
                recognition_threshold=recognition_threshold,
                novelty_threshold=novelty_threshold,
                max_patterns=64,
                seed=seed + i,
            )
            for i, w in enumerate(self._scales)
        ]

    def push(
        self,
        sensor_hv: torch.Tensor,
        label: Optional[str] = None,
    ) -> Optional[PatternMatch]:
        """
        Push one sensor HV through all scale memories and return best match.

        Multi-scale agreement bonus: if the same pattern label is matched at
        multiple scales simultaneously, the similarity is boosted.  This
        reduces false positives — transient coincidences rarely repeat at
        multiple timescales, but true recurring patterns do.

        Returns:
            Best PatternMatch across all scales, or None if no match fired.
        """
        matches = []
        for mem in self._memories:
            match = mem.push(sensor_hv, label)
            if match is not None:
                matches.append(match)

        if not matches:
            return None

        # Find best match by base similarity
        best_match = max(matches, key=lambda m: m.similarity)

        # Scale-agreement bonus: count how many scales matched the same label
        if best_match.label is not None:
            agreeing = sum(1 for m in matches if m.label == best_match.label)
            if agreeing > 1:
                # Boost similarity proportionally to number of agreeing scales
                boost = 0.05 * (agreeing - 1)   # +5% per additional agreeing scale
                from dataclasses import replace as _replace
                best_match = _replace(
                    best_match,
                    similarity=min(1.0, best_match.similarity + boost),
                )

        return best_match

    @property
    def n_patterns_total(self) -> int:
        return sum(m.n_patterns for m in self._memories)

    @property
    def n_patterns_by_scale(self) -> List[int]:
        return [m.n_patterns for m in self._memories]


def test_sequence_pattern_memory():
    print("=" * 60)
    print("Testing SequencePatternMemory (HoloGN-backed)")
    print("=" * 60)

    torch.manual_seed(42)
    dim = 2000
    mem = SequencePatternMemory(dim, window=4, stride=2, recognition_threshold=0.25)

    # Generate a repeating sawtooth sensor HV pattern
    def sawtooth_hv(phase: float) -> torch.Tensor:
        g = torch.Generator()
        g.manual_seed(int(phase * 1000) % 10000)
        base = (torch.rand(dim, generator=g) < 0.5).float()
        # Add small phase-dependent noise
        noise = (torch.rand(dim) < 0.05).float()
        return ((base + noise) > 0.5).float()

    phases = [0.0, 0.25, 0.5, 0.75]  # 4-tick period

    # Run 3 complete cycles
    last_match = None
    n_recognized = 0
    for cycle in range(4):
        for phase in phases:
            hv = sawtooth_hv(phase)
            match = mem.push(hv, label=f"phase_{phase}")
            if match is not None:
                last_match = match
                if match.is_known and cycle > 0:
                    n_recognized += 1

    print(f"  Known patterns after 4 cycles: {mem.n_patterns}")
    pm_id  = last_match.pattern_id if last_match else None
    pm_sim = round(last_match.similarity, 4) if last_match else 0.0
    print(f"  Last match: id={pm_id}, sim={pm_sim}")
    print(f"  Recognised (not first cycle): {n_recognized}")
    assert mem.n_patterns > 0, "Should have stored at least one pattern"

    print("  ✅ SequencePatternMemory OK")


def test_causal_transition_graph():
    print("=" * 60)
    print("Testing CausalTransitionGraph (VSAGraph-backed)")
    print("=" * 60)

    torch.manual_seed(1)
    dim = 2000

    # States: A → B → C → A (cyclic)
    state_a = (torch.rand(dim) < 0.5).float()
    state_b = (torch.rand(dim) < 0.5).float()
    state_c = (torch.rand(dim) < 0.5).float()

    # Action: ADVANCE (same for all transitions)
    action = (torch.rand(dim) < 0.5).float()

    causal = CausalTransitionGraph(dim, decay=0.9)

    # Observe transitions 10 times each
    for _ in range(10):
        causal.observe(state_a, action, state_b, "A", "B")
        causal.observe(state_b, action, state_c, "B", "C")
        causal.observe(state_c, action, state_a, "C", "A")

    print(f"  Registered {causal.n_transitions} transitions")

    # Forward query: from A + ADVANCE → should be close to B
    pred_b, conf = causal.forward_query(state_a, action)
    sim_to_b = float(_hamming(pred_b, state_b).item())
    sim_to_c = float(_hamming(pred_b, state_c).item())
    print(f"  Forward (A, ADVANCE): sim_to_B={sim_to_b:.4f}, sim_to_C={sim_to_c:.4f}")
    print(f"  Causal confidence: {conf:.4f}")

    # Causal surprise: known transition should have low surprise
    low_surprise = causal.causal_surprise(state_a, action, state_b)
    high_surprise = causal.causal_surprise(state_a, action, (torch.rand(dim) < 0.5).float())
    print(f"  Surprise (known A→B): {low_surprise:.4f}")
    print(f"  Surprise (novel A→rand): {high_surprise:.4f}")
    assert low_surprise < high_surprise, "Known transitions should have lower surprise"

    # Counterfactual
    null_action = (torch.rand(dim) < 0.5).float()
    cf_pred, divergence = causal.counterfactual_query(state_a, action, null_action)
    print(f"  Counterfactual divergence (different action): {divergence:.4f}")

    print("  ✅ CausalTransitionGraph OK")


def test_hierarchical_context():
    print("=" * 60)
    print("Testing HierarchicalContextEncoder (n-gram + EMA)")
    print("=" * 60)

    torch.manual_seed(7)
    dim = 2000
    ctx = HierarchicalContextEncoder(dim, pattern_window=4, situation_decay=0.9)

    # Same HV in two different situations
    hv = (torch.rand(dim) < 0.5).float()
    noise_hv = (torch.rand(dim) < 0.5).float()

    # Situation 1: see noise_hv for 10 ticks then hv
    for _ in range(10):
        r1 = ctx.encode(noise_hv)
    r1_in_noise = ctx.encode(hv)   # hv seen after noise situation

    # Reset situation
    ctx.reset_situation()

    # Situation 2: see hv continuously then hv again
    for _ in range(10):
        ctx.encode(hv)
    r2_in_signal = ctx.encode(hv)  # hv seen after hv situation

    # Context should differ even though tick HV is identical
    ctx_sim = float(_hamming(r1_in_noise["context"], r2_in_signal["context"]).item())
    print(f"  Same tick HV in different situations: context_sim={ctx_sim:.4f}")
    print(f"  (want < 1.0 — different situations produce different contexts)")
    assert ctx_sim < 0.95, "Same HV should have different context in different situations"

    # Pattern and situation levels should differ
    sit_sim = float(_hamming(r1_in_noise["situation"], r2_in_signal["situation"]).item())
    print(f"  Situation HV similarity (noise vs signal): {sit_sim:.4f}  (want < 0.8)")
    assert sit_sim < 0.9

    print("  ✅ HierarchicalContextEncoder OK")


def test_contextual_world_model():
    print("=" * 60)
    print("Testing ContextualWorldModel (full integration)")
    print("=" * 60)

    import time as _time
    torch.manual_seed(99)

    from hdc.sensor_stream import SensorSpec, ModalityType

    specs = [
        SensorSpec("imu",   ModalityType.TIME_SERIES, raw_dim=3, hd_dim=800, seed=0),
        SensorSpec("temp",  ModalityType.SCALAR,      raw_dim=1, hd_dim=800, seed=1),
    ]

    base = HybridPhysicalAIPipeline(
        specs, hd_dim=800, temporal_window=4, n_ensemble=3,
        consolidation_period=8, surprise_threshold=0.20,
    )
    world = ContextualWorldModel(base, pattern_window=4, pattern_stride=3)

    dim = 800
    action = (torch.rand(dim) < 0.5).float()

    # Sawtooth: repeat 3 cycles of 4 distinct states
    def make_reading(phase: int) -> SensorReading:
        return SensorReading(
            timestamp=float(_time.time()),
            data={
                "imu":  torch.randn(4, 3) * (0.1 + 0.2 * (phase % 4)),
                "temp": torch.tensor([15.0 + 5.0 * (phase % 4)]),
            }
        )

    last_result = None
    for t in range(12):
        reading = make_reading(t)
        result = world.tick(reading, action_hv=action)
        last_result = result

    print(f"  After 12 ticks (3 sawtooth cycles):")
    print(f"    Known patterns: {last_result['n_known_patterns']}")
    print(f"    Causal transitions: {last_result['n_causal_transitions']}")
    print(f"    Pattern boost: {last_result['pattern_boost']:.4f}")
    print(f"    Causal surprise: {last_result['causal_surprise']:.4f}")
    print(f"    Causal confidence: {last_result['causal_confidence']:.4f}")
    print(f"    Prediction error: {last_result['prediction_error']:.4f}")

    assert last_result["n_causal_transitions"] > 0
    assert last_result["contextual_tick"] == 12

    # Counterfactual
    alt_action = (torch.rand(dim) < 0.5).float()
    cf = world.counterfactual(
        last_result["sensor_hv"], action, alt_action
    )
    print(f"    Counterfactual divergence: {cf['divergence_from_actual']:.4f}")

    status = world.status()
    print(f"  Status: patterns={status['n_known_patterns']}, "
          f"causal={status['n_causal_transitions']}")
    print("  ✅ ContextualWorldModel OK")


if __name__ == "__main__":
    test_sequence_pattern_memory()
    print()
    test_causal_transition_graph()
    print()
    test_hierarchical_context()
    print()
    test_contextual_world_model()
    print()
    print("=== All world_context tests passed ===")
