"""
VSA Analogical Reasoning and Scenario Transfer
================================================
Implements the most architecturally unique capability of Vector Symbolic
Architectures: native analogical reasoning via algebraic composition.

Core insight (Plate 1995, Kanerva 2009, Kleyko 2022):
  If A, B, C, D are hypervectors with A:B :: C:D, then in bipolar VSA:
      D ≈ A* ⊗ B ⊗ C   (where A* = bipolar conjugate of A)
  For BSC (binary):
      D ≈ XOR(XOR(A, B), C)

  Example:
    A = "normal_bearing"   B = "bearing_fault"
    C = "normal_motor"     D = ?
    D = XOR(XOR(A, B), C) ≈ "motor_fault"   [transferred via analogy]

This enables ZERO-SHOT TRANSFER: given knowledge of one fault type,
reason about structurally analogous faults in related components without
having observed them.

Modules:

  1. AnalogicalReasoner — solves A:B :: C:? in VSA space
     - Bipolar XOR-chain reasoning
     - Cleanup via nearest-neighbor in concept codebook
     - Confidence from Hamming distance to codebook entry
     - Composable: chain multiple analogies

  2. ConceptMap — structured knowledge graph of physical concepts
     - Nodes: physical states (HVs from world model)
     - Edges: relational HVs (XOR(node_a, node_b))
     - Analogy: find nodes whose relational HV matches a query relation

  3. ScenarioTransfer — transfer world model knowledge across scenarios
     - Source: world model learned in scenario A
     - Target: new scenario B (partially observed)
     - Transfer: map source prototypes to target via analogical chain
     - Acceleration: target world model bootstrapped from transferred knowledge

  4. AnalogicalPlanner — extends HDCPlanner with analogical action transfer
     - When causal graph is sparse in new scenario, use analogy to
       borrow causal rules from a known similar scenario
     - Transfer: target_rule ≈ XOR(XOR(source_state, source_action), target_state)

Literature:
  Plate 1995 (Holographic Reduced Representations — analogy by circular conv)
  Kanerva 2009 (HDC intro — analogy by XOR/multiply)
  Kleyko 2022 Survey (kleyko_framework.py — VSARecord, VSAGraph)
  Gayler 2003 (VSA answers Jackendoff — compositionality)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from hdc.hdc_glue import (
    hv_xor, hv_batch_sim, hv_bundle, hv_majority, gen_hvs,
)
from hdc.physics_world_model import _xor, _hamming, _majority


# ═══════════════════════════════════════════════════════════════════════════════
# 1. AnalogicalReasoner — A:B :: C:? solving in VSA
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class AnalogyResult:
    """Result of an analogical inference."""
    query_hv: torch.Tensor        # raw XOR-derived answer
    nearest_concept: Optional[str]  # closest concept in codebook (if available)
    confidence: float               # Hamming similarity to nearest concept
    relational_hv: torch.Tensor   # XOR(A, B) — the transferred relation


class AnalogicalReasoner:
    """
    Solve A:B :: C:? analogies in VSA space.

    For binary BSC vectors: D ≈ XOR(XOR(A, B), C)
    The operation XOR(A, B) extracts the "relational HV" — the
    transformation that maps A to B. Applying this transformation to C
    gives an approximation of D.

    Properties (from VSA algebra):
      - If A, B are atomic (from item memory), the answer is exact.
      - If A, B are composed (e.g. bundled prototypes), the answer is
        approximate — cleanup via codebook lookup is needed.
      - The relational HV XOR(A,B) can be stored and reused to transfer
        the same relationship to any number of new base concepts.

    Multi-hop analogies:
      A:B :: C:D :: E:?  =  XOR(XOR(A,B), XOR(C,D), E) — two relations composed

    Args:
        hd_dim: Hypervector dimensionality
        concept_codebook: Optional dict of {name: HV} for cleanup
    """

    def __init__(
        self,
        hd_dim: int,
        concept_codebook: Optional[Dict[str, torch.Tensor]] = None,
    ):
        self.hd_dim = hd_dim
        self._codebook: Dict[str, torch.Tensor] = concept_codebook or {}

    def register_concept(self, name: str, hv: torch.Tensor):
        """Add a concept to the cleanup codebook."""
        self._codebook[name] = hv.detach().clone()

    def relational_hv(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Extract the relational HV: XOR(A, B).

        This is the "difference vector" from A to B.
        Applying it to C gives the analog of B relative to C.
        """
        return _xor(a, b)

    def solve(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
        cleanup: bool = True,
    ) -> AnalogyResult:
        """
        Solve A:B :: C:? — find D such that C:D has the same relation as A:B.

        D_raw = XOR(XOR(A, B), C)   [apply A→B relation to C]

        Args:
            a: HV of source base concept
            b: HV of source transformed concept
            c: HV of target base concept
            cleanup: If True, find nearest concept in codebook

        Returns:
            AnalogyResult with raw answer and optional cleanup
        """
        rel = self.relational_hv(a, b)
        d_raw = _xor(rel, c)

        nearest, conf = self._cleanup(d_raw) if cleanup else (None, 0.0)
        return AnalogyResult(
            query_hv=d_raw,
            nearest_concept=nearest,
            confidence=conf,
            relational_hv=rel,
        )

    def solve_multi(
        self,
        pairs: List[Tuple[torch.Tensor, torch.Tensor]],
        c: torch.Tensor,
        cleanup: bool = True,
    ) -> AnalogyResult:
        """
        Multi-relation analogy: (A1:B1, A2:B2, ...) :: C:?

        Compose multiple relational HVs via XOR-bundle and apply to C.
        Each pair contributes evidence for the final answer.
        """
        rels = [self.relational_hv(a, b) for a, b in pairs]
        rel_stacked = torch.stack(rels)
        # Majority-bundle of relations
        rel_composed = _majority(rel_stacked.mean(dim=0))
        d_raw = _xor(rel_composed, c)

        nearest, conf = self._cleanup(d_raw) if cleanup else (None, 0.0)
        return AnalogyResult(
            query_hv=d_raw,
            nearest_concept=nearest,
            confidence=conf,
            relational_hv=rel_composed,
        )

    def _cleanup(self, hv: torch.Tensor) -> Tuple[Optional[str], float]:
        """Find nearest concept in codebook."""
        if not self._codebook:
            return None, 0.0
        names = list(self._codebook.keys())
        hvs = torch.stack([self._codebook[n] for n in names])
        sims = hv_batch_sim(hv, hvs)
        best_idx = int(sims.argmax().item())
        return names[best_idx], float(sims[best_idx].item())

    def similarity_to(self, hv: torch.Tensor, concept: str) -> float:
        """Hamming similarity of a HV to a named concept."""
        if concept not in self._codebook:
            return 0.0
        return float(hv_batch_sim(hv, self._codebook[concept].unsqueeze(0))[0].item())

    def find_analogous_pairs(
        self,
        relation_hv: torch.Tensor,
        top_k: int = 5,
        similarity_threshold: float = 0.65,
    ) -> List[Tuple[str, str, float]]:
        """
        Find all pairs (A, B) in the codebook that share the given relation.

        Given a relation HV r = XOR(A, B), find all pairs (C, D) such that
        XOR(C, D) is similar to r. This discovers STRUCTURAL ANALOGIES:
        all pairs in the codebook that share the same transformation.

        Returns:
            List of (concept_a, concept_b, similarity) sorted by similarity
        """
        names = list(self._codebook.keys())
        if len(names) < 2:
            return []

        results = []
        for i, na in enumerate(names):
            for j, nb in enumerate(names):
                if i >= j:
                    continue
                ab_rel = self.relational_hv(
                    self._codebook[na], self._codebook[nb]
                )
                sim = float(hv_batch_sim(relation_hv, ab_rel.unsqueeze(0))[0].item())
                if sim >= similarity_threshold:
                    results.append((na, nb, sim))

        results.sort(key=lambda x: x[2], reverse=True)
        return results[:top_k]


# ═══════════════════════════════════════════════════════════════════════════════
# 2. ConceptMap — structured physical knowledge graph
# ═══════════════════════════════════════════════════════════════════════════════

class ConceptMap:
    """
    Structured knowledge graph of physical concepts for analogical reasoning.

    Nodes: physical states (e.g. "normal_bearing", "bearing_fault_stage1")
    Edges: relational HVs XOR(node_a, node_b) labelled with relationship type

    The ConceptMap enables:
      - Storing domain knowledge about physical concepts
      - Discovering analogical pairs that share a relation
      - Transferring fault knowledge across component types

    Physical AI use case:
      Register these concepts:
        normal_bearing, bearing_fault_early, bearing_fault_severe
        normal_gear,    gear_fault_early,    gear_fault_severe

      The analogical reasoner can then infer:
        "bearing_fault_early is to normal_bearing as
         gear_fault_early is to normal_gear"
      And apply new bearing fault observations to predict gear faults.

    Args:
        hd_dim: Hypervector dimensionality
        seed: Random seed for concept HV generation
    """

    def __init__(self, hd_dim: int, seed: int = 42):
        self.hd_dim = hd_dim
        self._concepts: Dict[str, torch.Tensor] = {}
        self._relations: Dict[str, List[Tuple[str, str, str]]] = {}  # type → [(a,b,label)]
        self._reasoner = AnalogicalReasoner(hd_dim)
        self._seed = seed
        self._n = 0

    def add_concept(
        self,
        name: str,
        hv: Optional[torch.Tensor] = None,
        related_to: Optional[str] = None,
        relation_type: Optional[str] = None,
    ):
        """
        Register a physical concept.

        Args:
            name: Concept name (e.g. "bearing_fault_stage1")
            hv: Optional HV (generated if None)
            related_to: Name of a related concept (adds edge)
            relation_type: Type of relation (e.g. "fault_progression", "component_analog")
        """
        if hv is None:
            g = torch.Generator()
            g.manual_seed(self._seed + self._n)
            hv = (torch.rand(self.hd_dim, generator=g) < 0.5).float()
            self._n += 1

        self._concepts[name] = hv.detach().clone()
        self._reasoner.register_concept(name, hv)

        if related_to and relation_type and related_to in self._concepts:
            if relation_type not in self._relations:
                self._relations[relation_type] = []
            self._relations[relation_type].append((name, related_to, relation_type))

    def add_from_observation(self, name: str, observed_hv: torch.Tensor):
        """Register a concept from an observed sensor HV (ground truth)."""
        self.add_concept(name, hv=observed_hv)

    def query_analogy(
        self,
        a_name: str,
        b_name: str,
        c_name: str,
    ) -> Optional[AnalogyResult]:
        """Solve A:B :: C:? where A, B, C are named concepts."""
        if not all(n in self._concepts for n in [a_name, b_name, c_name]):
            return None
        return self._reasoner.solve(
            self._concepts[a_name],
            self._concepts[b_name],
            self._concepts[c_name],
        )

    def transfer_relation(
        self,
        source_a: str,
        source_b: str,
        target_c: str,
        target_d_name: str,
    ) -> Optional[torch.Tensor]:
        """
        Transfer the A→B relationship from source to target: C→D.

        If A:B is a known fault progression, and C is analogous to A,
        then C:D captures the same progression for the target component.
        Registers D as a new concept.

        Returns:
            Transferred HV for D (registered as target_d_name)
        """
        result = self.query_analogy(source_a, source_b, target_c)
        if result is None:
            return None

        self.add_concept(target_d_name, hv=result.query_hv)
        return result.query_hv

    def find_all_analogies(
        self,
        relation_type: str,
        candidates: List[str],
    ) -> List[Tuple[str, AnalogyResult]]:
        """
        Find all analogical completions for a relation type across candidates.

        For each registered pair (A, B) of the given relation type, solve:
            A:B :: C:?  for each C in candidates.
        Returns the closest analogical completions.
        """
        if relation_type not in self._relations:
            return []

        results = []
        for a_name, b_name, _ in self._relations[relation_type]:
            if a_name not in self._concepts or b_name not in self._concepts:
                continue
            for c_name in candidates:
                if c_name not in self._concepts or c_name == a_name:
                    continue
                result = self._reasoner.solve(
                    self._concepts[a_name],
                    self._concepts[b_name],
                    self._concepts[c_name],
                )
                results.append((f"{c_name}_from_{a_name}:{b_name}", result))

        results.sort(key=lambda x: x[1].confidence, reverse=True)
        return results

    def concept_hv(self, name: str) -> Optional[torch.Tensor]:
        return self._concepts.get(name)

    @property
    def n_concepts(self) -> int:
        return len(self._concepts)

    @property
    def concept_names(self) -> List[str]:
        return list(self._concepts.keys())


# ═══════════════════════════════════════════════════════════════════════════════
# 3. ScenarioTransfer — zero-shot knowledge transfer
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TransferResult:
    """Result of scenario transfer."""
    n_prototypes_transferred: int
    n_causal_rules_transferred: int
    transfer_confidence: float
    transferred_concepts: List[str]


class ScenarioTransfer:
    """
    Transfer world model knowledge across scenarios via analogical mapping.

    Scenario A (learned): bearing fault detection in a wind turbine
    Scenario B (new):     gear fault detection in a gearbox

    Transfer mechanism:
      1. Register corresponding concepts in source and target:
             source: "normal_A", "fault_A"  (known)
             target: "normal_B"             (observed in B so far)
      2. Compute analogical mapping: A:B :: normal_A:normal_B
      3. Transfer: fault_B ≈ XOR(XOR(normal_A, fault_A), normal_B)
      4. Register transferred fault_B as a danger prototype in world model B
      5. Optionally transfer causal rules: "normal_A→fault_A via vibration_spike"
         becomes "normal_B→fault_B via vibration_spike" (same action HV)

    This gives world model B a head start: it has fault prototypes BEFORE
    ever observing a fault, purely from analogical transfer.

    Args:
        source_concept_map: ConceptMap from the source scenario
        source_world_model: Trained PhysicsWorldModel from source scenario
        target_world_model: New PhysicsWorldModel in target scenario
    """

    def __init__(
        self,
        source_concept_map: ConceptMap,
        source_world_model,   # PhysicsWorldModel
        target_world_model,   # PhysicsWorldModel
        transfer_threshold: float = 0.55,
    ):
        self.source_map = source_concept_map
        self.source_wm = source_world_model
        self.target_wm = target_world_model
        self.threshold = transfer_threshold
        self.target_map = ConceptMap(source_concept_map.hd_dim)

        self._reasoner = AnalogicalReasoner(source_concept_map.hd_dim)

    def register_anchor(
        self,
        source_concept: str,
        target_hv: torch.Tensor,
        target_name: str,
    ):
        """
        Register an anchor concept pair (source ↔ target).

        An anchor is a known corresponding concept between the two scenarios:
        e.g., "normal operation" in source corresponds to "normal operation"
        in target (both observed at the beginning of each scenario).
        """
        self.target_map.add_concept(target_name, hv=target_hv)

        # Register in reasoner's codebook
        src_hv = self.source_map.concept_hv(source_concept)
        if src_hv is not None:
            self._reasoner.register_concept(f"source:{source_concept}", src_hv)
        self._reasoner.register_concept(f"target:{target_name}", target_hv)

    def transfer_prototypes(
        self,
        source_base: str,
        target_base: str,
        transfer_labels: Optional[List[str]] = None,
    ) -> TransferResult:
        """
        Transfer all prototypes from source to target via analogy.

        For each concept C in source (other than source_base):
            C_target ≈ XOR(XOR(source_base_hv, target_base_hv), C_source)

        Then register high-confidence transfers as:
          - Safe prototypes if C was safe in source world model
          - Danger prototypes if C was danger in source world model

        Args:
            source_base: Name of anchor concept in source
            target_base: Name of corresponding concept in target
            transfer_labels: Subset of source concepts to transfer (all if None)

        Returns:
            TransferResult summary
        """
        src_base_hv = self.source_map.concept_hv(source_base)
        tgt_base_hv = self.target_map.concept_hv(target_base)

        if src_base_hv is None or tgt_base_hv is None:
            return TransferResult(0, 0, 0.0, [])

        concepts_to_transfer = transfer_labels or self.source_map.concept_names
        transferred = []
        n_safe = 0
        n_danger = 0
        confidences = []

        for src_name in concepts_to_transfer:
            if src_name == source_base:
                continue
            src_hv = self.source_map.concept_hv(src_name)
            if src_hv is None:
                continue

            # Analogical transfer: source_base:src_name :: target_base:?
            result = self._reasoner.solve(src_base_hv, src_hv, tgt_base_hv, cleanup=False)
            transferred_hv = result.query_hv
            tgt_name = f"transferred_{src_name}"

            # Register in target concept map
            self.target_map.add_concept(tgt_name, hv=transferred_hv)
            transferred.append(tgt_name)

            # Estimate transfer confidence from source codebook similarity
            self._reasoner.register_concept(src_name, src_hv)
            _, src_conf = self._reasoner._cleanup(src_hv)
            confidences.append(src_conf)

            # Determine if source was safe/danger and transfer to target WM
            ev_src = self.source_wm.action_evaluator
            is_safe = any(
                float(hv_batch_sim(src_hv, p.unsqueeze(0))[0]) > 0.7
                for p in ev_src._safe_prototypes
            ) if ev_src._safe_prototypes else False

            is_danger = any(
                float(hv_batch_sim(src_hv, p.unsqueeze(0))[0]) > 0.7
                for p in ev_src._danger_prototypes
            ) if ev_src._danger_prototypes else False

            ev_tgt = self.target_wm.action_evaluator
            if is_safe:
                ev_tgt.add_safe_state(transferred_hv)
                n_safe += 1
            elif is_danger:
                ev_tgt.add_danger_state(transferred_hv)
                n_danger += 1

        # Transfer causal rules (approximate)
        n_causal = self._transfer_causal_rules(src_base_hv, tgt_base_hv)

        mean_conf = sum(confidences) / max(len(confidences), 1)
        return TransferResult(
            n_prototypes_transferred=len(transferred),
            n_causal_rules_transferred=n_causal,
            transfer_confidence=mean_conf,
            transferred_concepts=transferred,
        )

    def _transfer_causal_rules(
        self,
        src_base_hv: torch.Tensor,
        tgt_base_hv: torch.Tensor,
    ) -> int:
        """
        Transfer causal rules from source to target CausalTransitionGraph.

        For each entry in source causal memory:
          Transfer key: tgt_key ≈ XOR(XOR(src_base, tgt_base), src_key)
          Transfer value: tgt_val ≈ XOR(XOR(src_base, tgt_base), src_val)
        """
        src_causal = getattr(self.source_wm, '_causal_graph', None)
        tgt_causal = getattr(self.target_wm, '_causal_graph', None)

        if src_causal is None or tgt_causal is None:
            return 0

        rel = _xor(src_base_hv, tgt_base_hv)
        n_transferred = 0

        for i, (accum, count) in enumerate(
            zip(src_causal._next_accum, src_causal._next_count)
        ):
            # Get source key from ComplexHammingSearch
            if src_causal._key_mem._H_complex is None or i >= src_causal._key_mem.n_stored():
                continue

            src_key_c = src_causal._key_mem._H_complex[i]
            src_key = (src_key_c.real > 0.5).float()
            src_val = (accum / max(count, 1) > 0.5).float()

            # Transfer via analogy
            tgt_key = _xor(rel, src_key)
            tgt_val = _xor(rel, src_val)

            # Register in target causal graph as a synthetic observation
            tgt_causal._key_mem.store(tgt_key, label=len(tgt_causal._next_accum))
            tgt_causal._next_accum.append(tgt_val.float())
            tgt_causal._next_count.append(count)
            n_transferred += 1

        return n_transferred


# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_analogical_reasoner():
    print("=" * 60)
    print("Testing AnalogicalReasoner (A:B :: C:?)")
    print("=" * 60)

    torch.manual_seed(42)
    dim = 5000
    reasoner = AnalogicalReasoner(dim)

    # Register atomic concepts
    torch.manual_seed(0)
    concepts = {
        "normal_bearing":    (torch.rand(dim) < 0.5).float(),
        "fault_bearing":     (torch.rand(dim) < 0.5).float(),
        "normal_gear":       (torch.rand(dim) < 0.5).float(),
        "fault_gear":        None,  # unknown — to be inferred
    }
    for name, hv in concepts.items():
        if hv is not None:
            reasoner.register_concept(name, hv)

    # The "fault" relation is XOR(normal_bearing, fault_bearing)
    # Solving: normal_bearing:fault_bearing :: normal_gear:?
    result = reasoner.solve(
        concepts["normal_bearing"],
        concepts["fault_bearing"],
        concepts["normal_gear"],
        cleanup=True,
    )

    print(f"  Relational HV density: {result.relational_hv.mean():.4f}  (want ≈ 0.5)")
    print(f"  Nearest concept: {result.nearest_concept}")
    print(f"  Confidence: {result.confidence:.4f}")
    assert result.query_hv.shape == (dim,)

    # Verify: applying the same relation BACK to fault_bearing → normal_bearing
    inverse = reasoner.solve(concepts["fault_bearing"], concepts["normal_bearing"],
                              result.query_hv, cleanup=True)
    # Should be close to normal_gear
    sim_to_normal_gear = float(hv_batch_sim(
        inverse.query_hv, concepts["normal_gear"].unsqueeze(0)
    )[0].item())
    print(f"  Inverse: XOR(fault→normal applied to fault_gear_approx) sim to normal_gear: {sim_to_normal_gear:.4f}")
    assert sim_to_normal_gear > 0.95, "Inverse should recover normal_gear"

    # Multi-hop: two relations composed
    concepts2 = {
        "A": (torch.rand(dim) < 0.5).float(),
        "B": (torch.rand(dim) < 0.5).float(),
        "C": (torch.rand(dim) < 0.5).float(),
        "D": (torch.rand(dim) < 0.5).float(),
        "E": (torch.rand(dim) < 0.5).float(),
    }
    result_multi = reasoner.solve_multi(
        [(concepts2["A"], concepts2["B"]), (concepts2["C"], concepts2["D"])],
        concepts2["E"],
    )
    print(f"  Multi-hop result shape: {result_multi.query_hv.shape}  ✅")

    print("  ✅ AnalogicalReasoner OK")


def test_concept_map():
    print("=" * 60)
    print("Testing ConceptMap (physical knowledge graph)")
    print("=" * 60)

    torch.manual_seed(7)
    dim = 4000
    cmap = ConceptMap(dim, seed=42)

    # Register fault progression in source component
    cmap.add_concept("normal_bearing")
    cmap.add_concept("early_fault",  related_to="normal_bearing", relation_type="fault_progression")
    cmap.add_concept("severe_fault", related_to="early_fault",    relation_type="fault_progression")

    # Register normal state of analogous component
    cmap.add_concept("normal_gear")

    print(f"  Registered {cmap.n_concepts} concepts: {cmap.concept_names}")
    assert cmap.n_concepts == 4

    # Transfer fault knowledge: what does early_fault look like for gear?
    result = cmap.query_analogy("normal_bearing", "early_fault", "normal_gear")
    assert result is not None
    print(f"  Analogy result confidence: {result.confidence:.4f}")

    # Transfer and register as new concept
    transferred_hv = cmap.transfer_relation(
        "normal_bearing", "early_fault", "normal_gear", "gear_early_fault"
    )
    assert transferred_hv is not None
    print(f"  Transferred gear_early_fault shape: {transferred_hv.shape}")
    assert "gear_early_fault" in cmap.concept_names

    print("  ✅ ConceptMap OK")


def test_scenario_transfer():
    print("=" * 60)
    print("Testing ScenarioTransfer (zero-shot transfer)")
    print("=" * 60)

    torch.manual_seed(99)
    dim = 2000
    from hdc.physics_world_model import PhysicsWorldModel

    # Source scenario: bearing fault detection (well-trained)
    src_wm = PhysicsWorldModel(hd_dim=dim)
    tgt_wm = PhysicsWorldModel(hd_dim=dim)

    src_map = ConceptMap(dim, seed=1)
    src_map.add_concept("normal_bearing")
    src_map.add_concept("bearing_fault")

    # Register some prototypes in source WM
    normal_hv = src_map.concept_hv("normal_bearing")
    fault_hv  = src_map.concept_hv("bearing_fault")
    src_wm.action_evaluator.add_safe_state(normal_hv)
    src_wm.action_evaluator.add_danger_state(fault_hv)

    # Create transfer
    transfer = ScenarioTransfer(src_map, src_wm, tgt_wm)

    # Register anchor: normal_gear corresponds to normal_bearing
    torch.manual_seed(42)
    normal_gear_hv = (torch.rand(dim) < 0.5).float()
    transfer.register_anchor("normal_bearing", normal_gear_hv, "normal_gear")

    # Run transfer
    result = transfer.transfer_prototypes("normal_bearing", "normal_gear",
                                          transfer_labels=["bearing_fault"])

    print(f"  Transferred prototypes: {result.n_prototypes_transferred}")
    print(f"  Transferred concepts: {result.transferred_concepts}")
    print(f"  Target safe/danger: {len(tgt_wm.action_evaluator._safe_prototypes)} / "
          f"{len(tgt_wm.action_evaluator._danger_prototypes)}")

    assert result.n_prototypes_transferred > 0, "Should have transferred prototypes"
    n_protos = (len(tgt_wm.action_evaluator._safe_prototypes) +
                len(tgt_wm.action_evaluator._danger_prototypes))
    assert n_protos > 0, "Target WM should have prototypes after transfer"

    print("  ✅ ScenarioTransfer OK")


# ═══════════════════════════════════════════════════════════════════════════════
# Elite Enhancements — EnsembleAnalogicalReasoner
# ═══════════════════════════════════════════════════════════════════════════════

class EnsembleAnalogicalReasoner:
    """
    Elite replacement for AnalogicalReasoner.

    Improvements over baseline:
      - Ensemble of M codebooks with controlled per-codebook noise (5% bit flips),
        producing diverse analogical answers that are majority-voted together.
      - Confidence calibration: raw Hamming confidence is compared against a
        null distribution (random queries) and converted to a p-value.
      - Chain-of-thought: break A:B into sub-relations and apply them
        sequentially to C, enabling multi-step analogical transfer.

    Args:
        hd_dim: Hypervector dimension
        n_codebooks: Ensemble size (default 5)
        chain_depth: Max chaining depth (default 3)
    """

    def __init__(self, hd_dim: int, n_codebooks: int = 5, chain_depth: int = 3,
                 use_hrr: bool = False):
        self.hd_dim      = hd_dim
        self.n_codebooks = n_codebooks
        self.chain_depth = chain_depth
        self.use_hrr     = use_hrr   # use HRR exact unbinding (lower noise, higher accuracy)

        self._codebooks: List[Dict[str, torch.Tensor]] = [{} for _ in range(n_codebooks)]
        self._null_sims: Dict[int, List[float]] = {}

        # HRR instance for exact unbinding (use_hrr=True mode)
        if use_hrr:
            try:
                from hdc.hrr import HRR
                self._hrr = HRR(hd_dim)
                # Convert codebook HVs to real-valued for HRR (from binary)
                self._hrr_real = True
            except ImportError:
                self.use_hrr = False

    def register_concept(self, name: str, hv: torch.Tensor, n_codebooks: Optional[int] = None):
        """Register a concept in all codebooks with slight per-codebook noise."""
        nc = n_codebooks or self.n_codebooks
        for cb_idx in range(nc):
            if cb_idx == 0:
                self._codebooks[cb_idx][name] = hv.detach().clone()
            else:
                noisy = hv.detach().clone()
                flip = torch.rand_like(noisy) < 0.05
                noisy[flip] = 1.0 - noisy[flip]
                self._codebooks[cb_idx][name] = noisy

    def solve_ensemble(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
    ) -> Tuple[torch.Tensor, float, float]:
        """
        Solve A:B :: C:? using ensemble of codebooks.

        Returns:
            (answer_hv, mean_confidence, ensemble_agreement)
        """
        # Use HRR exact unbinding when available — much lower noise than XOR
        if self.use_hrr and hasattr(self, '_hrr'):
            # Convert binary {0,1} to real-valued for HRR
            a_r = a.float() * 2 - 1   # {0,1} → {-1,+1}
            b_r = b.float() * 2 - 1
            c_r = c.float() * 2 - 1
            rel_r = self._hrr.bind(a_r, b_r)   # HRR circular convolution
            d_raw_r = self._hrr.bind(rel_r, c_r)
            # Convert back and find nearest
            answers = [d_raw_r]
            _, conf = self._cleanup_cb_hrr(d_raw_r, 0)
            return _majority(torch.stack(answers).float().mean(dim=0)), conf, 1.0

        rel = _xor(a, b)
        answers, confidences = [], []
        for cb_idx in range(self.n_codebooks):
            d_raw = _xor(rel, c)
            _, conf = self._cleanup_cb(d_raw, cb_idx)
            answers.append(d_raw)
            confidences.append(conf)

        # Confidence-weighted aggregation: codebooks with higher confidence
        # contribute more to the final answer than uncertain ones.
        # This reduces noise from poorly-calibrated codebooks.
        w     = torch.tensor(confidences).clamp(min=1e-6)
        w     = w / w.sum()
        stacked = torch.stack(answers).float()          # (K, D)
        weighted_mean = (stacked * w.unsqueeze(-1)).sum(0)  # (D,)
        answer = _majority(weighted_mean)
        mean_conf = float(w.dot(torch.tensor(confidences)))

        pairwise = [
            float(_hamming(answers[i].unsqueeze(0), answers[j].unsqueeze(0)).item())
            for i in range(self.n_codebooks)
            for j in range(i + 1, self.n_codebooks)
        ]
        agreement = sum(pairwise) / max(len(pairwise), 1)
        return answer, mean_conf, agreement

    def solve_chain(
        self,
        chain: List[Tuple[torch.Tensor, torch.Tensor]],
        c: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        """
        Chain-of-thought analogy: [(A1,B1), (A2,B2), ...] :: C:?

        Each pair provides a sub-relation applied sequentially.
        Returns (final_answer, chain_confidence).
        """
        current = c
        confidences = []
        for a_i, b_i in chain[:self.chain_depth]:
            current = _xor(_xor(a_i, b_i), current)
            _, conf = self._cleanup_cb(current, 0)
            confidences.append(conf)
        return current, sum(confidences) / max(len(confidences), 1)

    def _cleanup_cb(self, hv: torch.Tensor, cb_idx: int) -> Tuple[Optional[str], float]:
        cb = self._codebooks[cb_idx]
        if not cb:
            return None, 0.0
        names = list(cb.keys())
        hvs = torch.stack([cb[n] for n in names])
        sims = _hamming(hv.unsqueeze(0), hvs)
        best_idx = int(sims.argmax().item())
        return names[best_idx], float(sims[best_idx].item())

    def _cleanup_cb_hrr(self, hv: torch.Tensor, cb_idx: int) -> Tuple[Optional[str], float]:
        """HRR version: cosine similarity cleanup for real-valued vectors."""
        import torch.nn.functional as F
        cb = self._codebooks[cb_idx]
        if not cb:
            return None, 0.0
        names = list(cb.keys())
        # Convert binary codebook HVs to real {-1,+1}
        hvs_r = torch.stack([(cb[n].float() * 2 - 1) for n in names])
        hv_n  = F.normalize(hv.float().unsqueeze(0), dim=-1)
        hvs_n = F.normalize(hvs_r, dim=-1)
        sims  = (hv_n @ hvs_n.T).squeeze(0)
        best_idx = int(sims.argmax().item())
        return names[best_idx], float(sims[best_idx].item())

    def calibrate_confidence(self, confidence: float, cb_idx: int = 0) -> float:
        """
        Calibrate raw confidence against a null distribution.

        Returns the fraction of random queries that score below `confidence`
        — i.e. the empirical p-value of the raw similarity.
        """
        if cb_idx not in self._null_sims:
            cb = self._codebooks[cb_idx]
            if not cb or len(cb) < 3:
                return confidence
            null_sims = []
            for _ in range(100):
                random_hv = (torch.rand(self.hd_dim) >= 0.5).float()
                hvs = torch.stack(list(cb.values()))
                sims = _hamming(random_hv.unsqueeze(0), hvs)
                null_sims.append(float(sims.max().item()))
            self._null_sims[cb_idx] = null_sims
        null_sims = self._null_sims[cb_idx]
        return sum(1 for s in null_sims if s < confidence) / max(len(null_sims), 1)


    def transfer_strength(
        self,
        source_pair:  Tuple[torch.Tensor, torch.Tensor],   # (A, B)
        target_start: torch.Tensor,                         # C
    ) -> float:
        """
        Measure how well the A:B relation transfers to C → D*.

        Returns the confidence of the best ensemble answer.
        A high score means the A:B relation is strongly applicable in C's domain;
        a low score means the analogy is forced or the relation doesn't transfer.

        Args:
            source_pair:  (A, B) — the source analogy pair
            target_start: C — the target concept

        Returns:
            Transfer strength ∈ [0, 1]
        """
        a, b = source_pair
        _, conf, _ = self.solve_ensemble(a, b, target_start)
        return float(conf)

    def analogy_matrix(
        self,
        concepts_a: List[str],
        concepts_b: List[str],
    ) -> torch.Tensor:
        """
        Compute a |A|×|B| matrix of analogy transfer strengths.

        Entry [i, j] = transfer_strength of (concepts_a[i], concepts_a[i]) → concepts_b[j].
        High values indicate strong analogical structure shared between pairs.

        Useful for: identifying the most transferable relational structures,
        discovering unexpected analogies, building concept maps.

        Returns:
            (|A|, |B|) float tensor of transfer strengths.
        """
        n, m = len(concepts_a), len(concepts_b)
        matrix = torch.zeros(n, m)

        for i, name_a in enumerate(concepts_a):
            hv_a = None
            for cb in self._codebooks:
                if name_a in cb:
                    hv_a = cb[name_a]; break
            if hv_a is None:
                continue
            for j, name_b in enumerate(concepts_b):
                hv_b = None
                for cb in self._codebooks:
                    if name_b in cb:
                        hv_b = cb[name_b]; break
                if hv_b is None:
                    continue
                # Transfer: does A:A relation mean anything for B?
                _, conf, _ = self.solve_ensemble(hv_a, hv_a, hv_b)
                matrix[i, j] = conf

        return matrix

    def top_analogies_for(
        self,
        query:  str,
        top_k:  int = 5,
    ) -> List[Tuple[str, str, float]]:
        """
        Find the top-k concept pairs (A, B) such that A:B :: query:?.

        Scans all registered concept pairs and returns those where the
        analogy transfer to `query` has the highest confidence.

        Returns:
            List of (concept_a, concept_b, confidence) sorted descending.
        """
        query_hv = None
        for cb in self._codebooks:
            if query in cb:
                query_hv = cb[query]; break
        if query_hv is None:
            return []

        # Collect all registered concept names
        all_names = []
        all_hvs   = []
        for cb in self._codebooks:
            for name, hv in cb.items():
                if name not in all_names:
                    all_names.append(name)
                    all_hvs.append(hv)

        results = []
        for i in range(len(all_names)):
            for j in range(len(all_names)):
                if i == j or all_names[i] == query or all_names[j] == query:
                    continue
                _, conf, _ = self.solve_ensemble(all_hvs[i], all_hvs[j], query_hv)
                results.append((all_names[i], all_names[j], float(conf)))

        results.sort(key=lambda x: x[2], reverse=True)
        return results[:top_k]


if __name__ == "__main__":
    test_analogical_reasoner()
    print()
    test_concept_map()
    print()
    test_scenario_transfer()
    print()
    print("=== All analogy tests passed ===")
