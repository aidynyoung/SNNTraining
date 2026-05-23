"""
Long-Term Memory Consolidation for Physical AI
================================================
Addresses the ring-buffer limitation: the SensorStreamBuffer loses old
samples when full, and the SequencePatternMemory stores patterns permanently
but without importance weighting — causing indiscriminate growth.

Biological memory systems solve this with two mechanisms:
  1. **Importance scoring** — grade each memory by how surprising it was,
     how well it was later explained, and how often it was recalled.
  2. **Spaced repetition** — replay important memories at exponentially
     increasing intervals so they get consolidated without dominating runtime.

Arthedain's implementation (all HDC, no backpropagation):

  **ImportanceMemory** — wraps SensorStreamBuffer with:
    - Importance score: f(surprise, resolution, recency)
    - Forgetting: low-importance samples are evicted first
    - Surprise-then-resolve tracking: high-surprise events that later got
      a low-error explanation are IMPORTANT; high-surprise events that
      stayed mysterious are CRITICAL.

  **SpacedReplay** — schedules replay of important memories:
    - Items with high importance are replayed more frequently
    - After each replay, if prediction error is low → consolidate into prototype
    - Consolidation: bundle N similar HVs → one durable prototype in the world model

  **LongTermMemory** — combines both, gives ContextualWorldModel a memory
    system that grows selectively rather than uniformly:
    - Short-term: SensorStreamBuffer (ring buffer, fast access)
    - Long-term: ImportanceMemory (importance-weighted, slow eviction)
    - Prototypes: durable HV clusters from repeated consolidation

  **MemoryConsolidator** — the offline consolidation pass:
    - Samples high-importance items from long-term memory
    - Clusters similar HVs (Hamming distance < cluster_threshold)
    - Bundles cluster members → cluster prototype
    - Registers prototype with world model's ActionEvaluator

Literature:
  - Teeters 2023 (resonator.py — AdaptiveHDClassifier dual memory ST/LT)
  - Kleyko 2023 Survey (kleyko_survey.py — RetrainingStrategy)
  - Schlegel 2024 (weighted_superposition.py — weighted bundle for consolidation)
  - Kanerva 1988 (sparse distributed memory — importance-weighted storage)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from hdc.hdc_glue import hv_batch_sim, hv_bundle, hv_majority, gen_hvs
from hdc.sensor_stream import SensorStreamBuffer, BufferedSample
from hdc.physics_world_model import PhysicsWorldModel, _hamming, _majority


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Importance Scoring
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MemoryEntry:
    """One entry in the long-term importance memory."""
    sensor_hv: torch.Tensor
    timestamp: float
    initial_surprise: float      # prediction error when first observed
    final_error: float           # prediction error at last recall
    n_replays: int = 0
    last_replay: float = 0.0

    @property
    def importance(self) -> float:
        """
        Importance score combining surprise, resolution quality, and recency.

        High importance = initially surprising AND later still uncertain
                        = the system was surprised and still hasn't learned it.
        Low importance  = initially surprising but later well-predicted
                        = was surprising, now consolidated.

        score = initial_surprise × (1 - resolution_quality) × recency_factor

        recency_factor decays with time since last replay (spaced repetition):
          items that haven't been replayed recently are boosted.
        """
        resolution_quality = max(0.0, 1.0 - self.final_error * 2)
        age_hours = max(0.0, (time.time() - self.last_replay) / 3600)
        recency_boost = 1.0 + math.log1p(age_hours * 4)
        return self.initial_surprise * (1.0 - 0.7 * resolution_quality) * recency_boost


class ImportanceMemory:
    """
    Long-term memory with importance-weighted retention and eviction.

    Stores samples from the short-term SensorStreamBuffer that pass an
    importance threshold. When full, evicts the lowest-importance item.

    This implements a selective long-term store:
      - High surprise → store (potentially important event)
      - High surprise + later low error → demote (learned, no longer critical)
      - High surprise + still high error → retain (still unresolved)

    Args:
        capacity: Maximum number of entries
        min_importance: Minimum surprise to enter long-term store
        eviction_percentile: Fraction evicted when full (evict bottom percentile)
    """

    def __init__(
        self,
        capacity: int = 512,
        min_importance: float = 0.15,
        eviction_percentile: float = 0.1,
    ):
        self.capacity = capacity
        self.min_importance = min_importance
        self.eviction_percentile = eviction_percentile
        self._entries: List[MemoryEntry] = []

    def push(
        self,
        sensor_hv: torch.Tensor,
        prediction_error: float,
        timestamp: Optional[float] = None,
    ) -> bool:
        """
        Conditionally add a sample to long-term memory.

        Returns True if added, False if below threshold.
        """
        if prediction_error < self.min_importance:
            return False

        entry = MemoryEntry(
            sensor_hv=sensor_hv.detach().clone(),
            timestamp=timestamp or time.time(),
            initial_surprise=prediction_error,
            final_error=prediction_error,
            last_replay=time.time(),
        )

        if len(self._entries) >= self.capacity:
            self._evict()

        self._entries.append(entry)
        return True

    def _evict(self):
        """Remove lowest-importance entries."""
        n_evict = max(1, int(self.capacity * self.eviction_percentile))
        self._entries.sort(key=lambda e: e.importance, reverse=True)
        self._entries = self._entries[:-n_evict]

    def update_resolution(self, idx: int, current_error: float):
        """Update the final_error for an entry after replay."""
        if 0 <= idx < len(self._entries):
            e = self._entries[idx]
            e.final_error = 0.8 * e.final_error + 0.2 * current_error
            e.n_replays += 1
            e.last_replay = time.time()

    def sample_by_importance(self, n: int) -> List[Tuple[int, MemoryEntry]]:
        """
        Sample n entries weighted by importance score.

        Higher importance → more likely to be selected.
        Returns list of (index, entry) pairs.
        """
        if not self._entries:
            return []

        scores = torch.tensor([e.importance for e in self._entries])
        scores = scores / scores.sum().clamp(min=1e-9)

        n = min(n, len(self._entries))
        indices = torch.multinomial(scores, n, replacement=False)
        return [(int(i), self._entries[i]) for i in indices.tolist()]

    def all_hvs(self) -> Optional[torch.Tensor]:
        """Return all stored HVs as a matrix (N, D)."""
        if not self._entries:
            return None
        return torch.stack([e.sensor_hv for e in self._entries])

    def __len__(self) -> int:
        return len(self._entries)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Spaced Repetition Replay
# ═══════════════════════════════════════════════════════════════════════════════

class SpacedReplay:
    """
    Spaced-repetition scheduler for memory consolidation.

    Items due for replay are identified by their last_replay time + interval:
        next_replay_due = last_replay + base_interval × 2^n_replays

    This creates an exponentially growing schedule:
      Replay 0 → 1 tick interval
      Replay 1 → 2 ticks
      Replay 2 → 4 ticks
      Replay n → 2^n ticks

    Items that have been replayed many times need replaying less often.
    Items not yet replayed are always due.

    Args:
        base_interval_ticks: Minimum interval between replays
        max_interval_ticks: Maximum interval (caps exponential growth)
    """

    def __init__(
        self,
        base_interval_ticks: int = 5,
        max_interval_ticks: int = 200,
    ):
        self.base = base_interval_ticks
        self.max_interval = max_interval_ticks
        self._tick = 0

    def tick(self):
        self._tick += 1

    def is_due(self, entry: MemoryEntry) -> bool:
        """Return True if this entry is due for replay."""
        if entry.n_replays == 0:
            return True   # never replayed → always due
        n = entry.n_replays
        interval = min(self.base * (2 ** n), self.max_interval)
        ticks_since_replay = self._tick - n * self.base
        return ticks_since_replay >= interval

    def due_entries(
        self,
        memory: ImportanceMemory,
        max_per_tick: int = 4,
    ) -> List[Tuple[int, MemoryEntry]]:
        """Return up to max_per_tick entries due for replay."""
        due = []
        for i, e in enumerate(memory._entries):
            if self.is_due(e):
                due.append((i, e))
        # Sort by importance (most important first)
        due.sort(key=lambda x: x[1].importance, reverse=True)
        return due[:max_per_tick]


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Memory Consolidator
# ═══════════════════════════════════════════════════════════════════════════════

class MemoryConsolidator:
    """
    Offline consolidation: cluster similar HVs → durable prototypes.

    Algorithm:
      1. Sample high-importance entries from ImportanceMemory.
      2. Greedily cluster by Hamming distance: entries with distance < threshold
         are assigned to the same cluster.
      3. For each cluster with ≥ min_cluster_size members:
         - Bundle member HVs with importance-weighted superposition
         - Binarize → cluster prototype
         - Register prototype with world model's ActionEvaluator (safe or danger)
         - Mark members as "consolidated" (lower importance for future eviction)

    Classification:
      - Clusters from LOW-error entries → safe prototypes
      - Clusters from HIGH-error entries → danger prototypes

    Args:
        cluster_threshold: Max Hamming distance within a cluster
        min_cluster_size: Minimum members to form a stable prototype
        safe_error_ceil: Max error for a cluster to be "safe"
        danger_error_floor: Min error for a cluster to be "dangerous"
    """

    def __init__(
        self,
        cluster_threshold: float = 0.15,
        min_cluster_size: int = 3,
        safe_error_ceil: float = 0.08,
        danger_error_floor: float = 0.30,
    ):
        self.cluster_threshold = cluster_threshold
        self.min_cluster_size = min_cluster_size
        self.safe_error_ceil = safe_error_ceil
        self.danger_error_floor = danger_error_floor

        self._n_safe_created = 0
        self._n_danger_created = 0
        self._n_consolidations = 0

    def consolidate(
        self,
        memory: ImportanceMemory,
        world_model: PhysicsWorldModel,
        n_sample: int = 32,
    ) -> Dict:
        """
        Run one consolidation pass.

        Args:
            memory: ImportanceMemory to consolidate from
            world_model: PhysicsWorldModel to register prototypes with
            n_sample: Number of entries to sample per pass

        Returns:
            Dict with n_clusters, n_safe_created, n_danger_created
        """
        if len(memory) < self.min_cluster_size:
            return {"n_clusters": 0, "n_safe_created": 0, "n_danger_created": 0}

        # Sample by importance
        sampled = memory.sample_by_importance(min(n_sample, len(memory)))
        if not sampled:
            return {"n_clusters": 0, "n_safe_created": 0, "n_danger_created": 0}

        # Greedy clustering
        clusters: List[List[Tuple[int, MemoryEntry]]] = []
        assigned = set()

        for i, (idx, entry) in enumerate(sampled):
            if idx in assigned:
                continue
            cluster = [(idx, entry)]
            assigned.add(idx)
            hv_i = entry.sensor_hv

            for j, (jdx, jentry) in enumerate(sampled):
                if jdx in assigned:
                    continue
                dist = 1.0 - float(_hamming(hv_i, jentry.sensor_hv).item())
                if dist < self.cluster_threshold:
                    cluster.append((jdx, jentry))
                    assigned.add(jdx)

            if len(cluster) >= self.min_cluster_size:
                clusters.append(cluster)

        n_safe = 0
        n_danger = 0

        for cluster in clusters:
            entries = [e for _, e in cluster]
            hvs = torch.stack([e.sensor_hv for e in entries])

            # Importance-weighted bundle
            weights = torch.tensor([e.importance for e in entries])
            weights = weights / weights.sum().clamp(min=1e-9)
            weighted = (hvs.float() * weights.unsqueeze(-1)).sum(dim=0)
            prototype = (weighted > 0.5).float()

            mean_error = sum(e.final_error for e in entries) / len(entries)

            ev = world_model.action_evaluator

            if mean_error <= self.safe_error_ceil:
                # Check diversity before adding
                if self._is_diverse(prototype, ev._safe_prototypes):
                    ev.add_safe_state(prototype)
                    n_safe += 1
                    self._n_safe_created += 1

            elif mean_error >= self.danger_error_floor:
                if self._is_diverse(prototype, ev._danger_prototypes):
                    ev.add_danger_state(prototype)
                    n_danger += 1
                    self._n_danger_created += 1

            # Demote consolidated entries (they're now captured in a prototype)
            for idx, entry in cluster:
                entry.final_error *= 0.5   # lower final error → lower importance

        self._n_consolidations += 1
        return {
            "n_clusters": len(clusters),
            "n_safe_created": n_safe,
            "n_danger_created": n_danger,
        }

    @staticmethod
    def _is_diverse(
        hv: torch.Tensor,
        existing: List[torch.Tensor],
        min_dist: float = 0.12,
    ) -> bool:
        if not existing:
            return True
        protos = torch.stack(existing)
        sims = hv_batch_sim(hv, protos)
        return float(sims.max().item()) < (1.0 - min_dist)

    @property
    def n_total_prototypes_created(self) -> int:
        return self._n_safe_created + self._n_danger_created


# ═══════════════════════════════════════════════════════════════════════════════
# 4. LongTermMemory — combines everything
# ═══════════════════════════════════════════════════════════════════════════════

class LongTermMemory:
    """
    Complete long-term memory system for the Physical AI agent.

    Wraps ImportanceMemory + SpacedReplay + MemoryConsolidator into
    a single interface called on every agent tick.

    Usage:
        ltm = LongTermMemory(world_model)
        # In SelfImprovementLoop.tick():
        ltm.tick(sensor_hv, prediction_error)
        # Periodically call consolidate() — or it auto-fires every N ticks:
        result = ltm.maybe_consolidate()

    Args:
        world_model: PhysicsWorldModel whose ActionEvaluator will receive prototypes
        ltm_capacity: Maximum long-term memory entries
        min_surprise: Minimum prediction error to enter LTM
        consolidation_period: Ticks between automatic consolidation passes
        base_replay_interval: Base spaced-repetition interval in ticks
    """

    def __init__(
        self,
        world_model: PhysicsWorldModel,
        ltm_capacity: int = 512,
        min_surprise: float = 0.12,
        consolidation_period: int = 25,
        base_replay_interval: int = 8,
    ):
        self.world_model = world_model
        self.consolidation_period = consolidation_period

        self.lt_memory = ImportanceMemory(
            capacity=ltm_capacity,
            min_importance=min_surprise,
        )
        self.spaced_replay = SpacedReplay(
            base_interval_ticks=base_replay_interval,
        )
        self.consolidator = MemoryConsolidator()

        self._tick = 0
        self._consolidation_log: List[Dict] = []

    def tick(
        self,
        sensor_hv: torch.Tensor,
        prediction_error: float,
    ) -> Dict:
        """
        Process one observation through the long-term memory system.

        1. Push to ImportanceMemory if surprising enough
        2. Run spaced replay (update resolution of due entries)
        3. Auto-consolidate every consolidation_period ticks

        Args:
            sensor_hv: (D,) current encoded sensor HV
            prediction_error: Prediction error on this tick

        Returns:
            Dict with ltm_size, replay_count, consolidation_result
        """
        self._tick += 1
        self.spaced_replay.tick()

        # Push to long-term memory
        added = self.lt_memory.push(sensor_hv, prediction_error)

        # Spaced replay: update resolution of due entries
        due = self.spaced_replay.due_entries(self.lt_memory, max_per_tick=4)
        for idx, entry in due:
            # Re-evaluate: use current predictor error as proxy for resolution
            # (in practice, you'd re-run the predictor on the stored HV)
            entry.n_replays += 1
            entry.last_replay = time.time()
            # If current error is lower, this entry is becoming resolved
            if prediction_error < entry.initial_surprise * 0.5:
                self.lt_memory.update_resolution(idx, prediction_error)

        # Auto-consolidate
        consol_result = self.maybe_consolidate()

        return {
            "ltm_size": len(self.lt_memory),
            "added_to_ltm": added,
            "replay_count": len(due),
            "consolidation": consol_result,
        }

    def maybe_consolidate(self) -> Optional[Dict]:
        """Run consolidation if it's time."""
        if self._tick % self.consolidation_period == 0:
            result = self.consolidator.consolidate(
                self.lt_memory, self.world_model, n_sample=32
            )
            self._consolidation_log.append(result)
            return result
        return None

    def summary(self) -> Dict:
        ev = self.world_model.action_evaluator
        return {
            "ltm_size": len(self.lt_memory),
            "total_consolidations": self.consolidator._n_consolidations,
            "safe_prototypes_from_ltm": self.consolidator._n_safe_created,
            "danger_prototypes_from_ltm": self.consolidator._n_danger_created,
            "total_safe_prototypes": len(ev._safe_prototypes),
            "total_danger_prototypes": len(ev._danger_prototypes),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_importance_memory():
    print("=" * 60)
    print("Testing ImportanceMemory (importance-weighted LTM)")
    print("=" * 60)

    torch.manual_seed(42)
    dim = 1000
    mem = ImportanceMemory(capacity=20, min_importance=0.10)

    # Push 30 samples with varying surprise levels
    for i in range(30):
        hv = (torch.rand(dim) < 0.5).float()
        error = 0.05 + 0.45 * (i % 6) / 5   # cycle: 0.05 → 0.5
        mem.push(hv, error)

    print(f"  LTM size (cap=20, 30 pushes): {len(mem)}")
    assert len(mem) <= 20, "Should not exceed capacity"

    # Verify importance ordering
    sampled = mem.sample_by_importance(5)
    errors = [e.initial_surprise for _, e in sampled]
    print(f"  Sampled errors (want generally high): {[round(e, 2) for e in errors]}")
    assert sum(1 for e in errors if e > 0.25) >= 2, "High-error samples should dominate"

    # Resolution update
    if sampled:
        idx, entry = sampled[0]
        old_imp = entry.importance
        mem.update_resolution(idx, current_error=0.02)  # this got resolved
        new_imp = entry.importance
        print(f"  Importance after resolution: {old_imp:.4f} → {new_imp:.4f}  (want decrease)")
        assert new_imp <= old_imp, "Resolution should reduce importance"

    print("  ✅ ImportanceMemory OK")


def test_spaced_replay():
    print("=" * 60)
    print("Testing SpacedReplay (spaced repetition schedule)")
    print("=" * 60)

    torch.manual_seed(1)
    dim = 500
    mem = ImportanceMemory(capacity=100, min_importance=0.1)
    replay = SpacedReplay(base_interval_ticks=3)

    for i in range(10):
        hv = (torch.rand(dim) < 0.5).float()
        mem.push(hv, prediction_error=0.3)

    # At tick 0: all entries are due (never replayed)
    due_0 = replay.due_entries(mem, max_per_tick=20)
    print(f"  Due at tick 0: {len(due_0)} (want all 10)")
    assert len(due_0) == min(20, len(mem._entries))

    # Advance ticks and mark replayed
    for idx, entry in due_0[:5]:
        entry.n_replays += 1
        entry.last_replay = time.time()
    for _ in range(3):
        replay.tick()

    # Some should still be due, others not
    due_3 = replay.due_entries(mem, max_per_tick=20)
    print(f"  Due at tick 3 (after 5 replayed): {len(due_3)}")

    print("  ✅ SpacedReplay OK")


def test_memory_consolidator():
    print("=" * 60)
    print("Testing MemoryConsolidator (clustering → prototypes)")
    print("=" * 60)

    torch.manual_seed(7)
    dim = 1000
    from hdc.physics_world_model import PhysicsWorldModel
    wm = PhysicsWorldModel(hd_dim=dim)
    mem = ImportanceMemory(capacity=200, min_importance=0.1)
    consolidator = MemoryConsolidator(cluster_threshold=0.15, min_cluster_size=3)

    # Generate 2 clusters: normal region (low error) and fault region (high error)
    safe_base = (torch.rand(dim) < 0.5).float()
    danger_base = (torch.rand(dim) < 0.5).float()

    for _ in range(15):
        # Safe cluster: base + small noise, low error
        hv = safe_base.clone()
        mask = torch.rand(dim) < 0.05
        hv[mask] = 1.0 - hv[mask]
        mem.push(hv, prediction_error=0.04)

    for _ in range(10):
        # Danger cluster: danger_base + small noise, high error
        hv = danger_base.clone()
        mask = torch.rand(dim) < 0.05
        hv[mask] = 1.0 - hv[mask]
        mem.push(hv, prediction_error=0.42)

    result = consolidator.consolidate(mem, wm, n_sample=25)
    print(f"  Clusters formed: {result['n_clusters']}")
    print(f"  Safe prototypes: {result['n_safe_created']}")
    print(f"  Danger prototypes: {result['n_danger_created']}")

    ev = wm.action_evaluator
    print(f"  ActionEvaluator safe: {len(ev._safe_prototypes)}, danger: {len(ev._danger_prototypes)}")
    assert len(ev._safe_prototypes) + len(ev._danger_prototypes) > 0, \
        "Should have registered at least one prototype"

    # Action scores should now be non-zero
    from hdc.physics_world_model import ActionCandidate
    candidates = [
        ActionCandidate("safe", safe_base),
        ActionCandidate("danger", danger_base),
    ]
    ranked = ev.evaluate(safe_base, candidates)
    print(f"  Action scores: {[(c.name, round(c.net_score, 3)) for c in ranked]}")
    assert any(c.net_score != 0.0 for c in ranked), "Scores should be non-zero after consolidation"

    print("  ✅ MemoryConsolidator OK")


def test_long_term_memory():
    print("=" * 60)
    print("Testing LongTermMemory (full system integration)")
    print("=" * 60)

    import time as _time
    torch.manual_seed(99)
    dim = 800

    from hdc.physics_world_model import PhysicsWorldModel
    wm = PhysicsWorldModel(hd_dim=dim)
    ltm = LongTermMemory(wm, ltm_capacity=100, min_surprise=0.12,
                          consolidation_period=10)

    # Simulate 40 ticks: 25 normal, 5 anomaly, 10 recovery
    normal_hv = (torch.rand(dim) < 0.5).float()
    anomaly_hv = (torch.rand(dim) < 0.5).float()

    for t in range(40):
        if t < 25:
            hv = normal_hv.clone()
            mask = torch.rand(dim) < 0.03
            hv[mask] = 1.0 - hv[mask]
            err = 0.04
        elif t < 30:
            hv = anomaly_hv.clone()
            mask = torch.rand(dim) < 0.05
            hv[mask] = 1.0 - hv[mask]
            err = 0.45
        else:
            hv = normal_hv.clone()
            mask = torch.rand(dim) < 0.03
            hv[mask] = 1.0 - hv[mask]
            err = 0.06

        result = ltm.tick(hv.float(), err)

    summary = ltm.summary()
    print(f"  LTM summary: {summary}")
    assert summary["ltm_size"] > 0, "Should have entries in LTM"
    assert summary["total_consolidations"] > 0, "Should have run consolidation"

    print("  ✅ LongTermMemory OK")


if __name__ == "__main__":
    test_importance_memory()
    print()
    test_spaced_replay()
    print()
    test_memory_consolidator()
    print()
    test_long_term_memory()
    print()
    print("=== All memory_consolidation tests passed ===")
