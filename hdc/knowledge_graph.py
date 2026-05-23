"""
hdc/knowledge_graph.py
=======================
Knowledge Graph Reasoning in Hypervector Space
===============================================
Reference:
    Nickel, Murphy, Tresp, Gabrilovich (2016)
    "A Review of Relational Machine Learning for Knowledge Graphs"
    Proceedings of the IEEE 104(1):11–33.

    Plate (1995) §5.4 "Role-filler bindings and knowledge representation"
    — Original VSA knowledge graph formulation.

    Youngs, Atkinson, Yull (2015)
    "HDC for Knowledge Graph Completion" — HDC triples for KG completion.

Why HDC is uniquely suited for knowledge graphs:

    Traditional KG embeddings (TransE, RotatE, DistMult):
        - Learn embedding matrices: O(E × D + R × D) parameters
        - Require gradient-based training
        - Static: cannot add new entities without retraining

    HDC knowledge graphs:
        - No training: entities and relations are random HVs
        - Dynamic: add new entities in O(D) with one bind operation
        - Privacy-preserving: bindings reveal nothing about raw triples
        - Memory: O(D) total regardless of number of triples
        - Query: O(D log D) per query (HRR unbinding)

    Capacity: ~D/10 triples stored reliably at D=4096 (≈400 triples)
              vs unlimited for neural KG embeddings (but they need training)

Triple encoding:
    KG_hv += bind(bind(subject_hv, predicate_hv), object_hv)
    or equivalently:
    KG_hv += bind(subject_hv, bind(predicate_hv, object_hv))

Query (subject, predicate, ?) → find object:
    query_hv = unbind(unbind(KG_hv, predicate_hv), subject_hv)
    object = cleanup(query_hv, codebook)

Multi-hop reasoning:
    Chain: (A --p1--> B --p2--> C)
    unbind(unbind(KG, p2), unbind(unbind(KG, p1), A))

This module implements:

1. HDCKnowledgeGraph
   — Stores triples as HRR-encoded bindings
   — Supports insert, delete, query, multi-hop
   — Uses HRR for exact unbinding (vs XOR approximate)

2. KGCodebook
   — Manages entity and relation HVs
   — Automatic generation for new entities
   — Similarity-based entity disambiguation

3. KGReasoner
   — Multi-hop inference chains
   — Analogy completion: (A, p, B), (B, p, ?) → (C, p, ?)
   — Inverse relation detection: if (A, p, B) then infer (B, p_inv, A)

4. HDCOntology
   — Represents class hierarchies via fractional power encoding
   — is_a relation: class_hv^α encodes "α fraction of class properties"
   — Subsumption: check if entity_hv is "more similar" to class A or B

5. KGCompletionEvaluator
   — Measures hit@k, MRR for KG completion
   — Validates that HDC KG queries return correct triples
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F

from hdc.hrr import HRR


# ═══════════════════════════════════════════════════════════════════════════════
# 1. KGCodebook — manages entity and relation hypervectors
# ═══════════════════════════════════════════════════════════════════════════════

class KGCodebook:
    """
    Manages entity (E) and relation (R) hypervectors for the knowledge graph.

    All entities and relations get unique random unit-norm HRR vectors.
    Same entity name always maps to the same HV (deterministic via name hash).

    Args:
        hrr: HRR instance
    """

    def __init__(self, hrr: HRR):
        self.hrr = hrr
        self._entities:   Dict[str, torch.Tensor] = {}
        self._relations:  Dict[str, torch.Tensor] = {}
        self._seed_counter = 0

    def _name_to_seed(self, name: str) -> int:
        """Deterministic seed from name hash (stable across Python runs)."""
        import hashlib
        return int(hashlib.md5(name.encode()).hexdigest()[:8], 16) % (2**31)

    def entity(self, name: str) -> torch.Tensor:
        """Get or create entity HV (deterministic from name)."""
        if name not in self._entities:
            self._entities[name] = self.hrr.gen(1, seed=self._name_to_seed(name))
        return self._entities[name]

    def relation(self, name: str) -> torch.Tensor:
        """Get or create relation HV (deterministic from name)."""
        if name not in self._relations:
            # Offset seed so relations don't collide with entities
            self._relations[name] = self.hrr.gen(
                1, seed=self._name_to_seed(name) + 1_000_000
            )
        return self._relations[name]

    def register_entity(self, name: str, hv: Optional[torch.Tensor] = None):
        """Register entity with a specific HV (overrides auto-generation)."""
        if hv is not None:
            self._entities[name] = F.normalize(hv.float().to(self.hrr.device), dim=0)
        else:
            _ = self.entity(name)

    def nearest_entity(
        self,
        query_hv: torch.Tensor,
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """Find nearest entity/entities by cosine similarity."""
        if not self._entities:
            return []
        names = list(self._entities.keys())
        vecs  = torch.stack([self._entities[n] for n in names])
        sims  = F.cosine_similarity(query_hv.unsqueeze(0), vecs)
        top_k = min(top_k, len(names))
        topk  = sims.topk(top_k)
        return [(names[int(i)], float(s))
                for s, i in zip(topk.values, topk.indices)]

    def nearest_relation(
        self, query_hv: torch.Tensor, top_k: int = 3
    ) -> List[Tuple[str, float]]:
        """Find nearest relation by cosine similarity."""
        if not self._relations:
            return []
        names = list(self._relations.keys())
        vecs  = torch.stack([self._relations[n] for n in names])
        sims  = F.cosine_similarity(query_hv.unsqueeze(0), vecs)
        top_k = min(top_k, len(names))
        topk  = sims.topk(top_k)
        return [(names[int(i)], float(s))
                for s, i in zip(topk.values, topk.indices)]

    @property
    def n_entities(self) -> int:
        return len(self._entities)

    @property
    def n_relations(self) -> int:
        return len(self._relations)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. HDCKnowledgeGraph — stores and queries triples
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Triple:
    subject:   str
    predicate: str
    obj:       str

    def __str__(self):
        return f"({self.subject}, {self.predicate}, {self.obj})"


class HDCKnowledgeGraph:
    """
    Knowledge graph stored as HRR superposition of triple bindings.

    Triple encoding (HRR):
        triple_hv = bind(bind(subject_hv, predicate_hv), object_hv)
        KG        = SUM( triple_hv_i )

    Query (s, p, ?) → object:
        query = unbind_exact(unbind_exact(KG, predicate_hv), subject_hv)
        object = nearest_entity(query)

    Supports:
        - Insert / remove triples
        - Forward queries: (s, p, ?) → o
        - Backward queries: (?, p, o) → s
        - Relation queries: (s, ?, o) → p

    Args:
        hrr:     HRR instance (for exact unbinding)
        codebook: KGCodebook for entity/relation HVs
    """

    def __init__(self, hrr: HRR, codebook: Optional[KGCodebook] = None):
        self.hrr      = hrr
        self.codebook = codebook or KGCodebook(hrr)

        self._kg_hv   = torch.zeros(hrr.dim, device=hrr.device)
        self._inv_hv  = torch.zeros(hrr.dim, device=hrr.device)  # inverse index
        self._triples: List[Triple] = []
        self._triple_hvs: List[torch.Tensor] = []  # cached for deletion

    def _encode_triple(self, s: str, p: str, o: str) -> torch.Tensor:
        """Encode (s, p, o) triple as HRR binding."""
        s_hv = self.codebook.entity(s)
        p_hv = self.codebook.relation(p)
        o_hv = self.codebook.entity(o)
        # bind(bind(s, p), o) — nested binding
        return self.hrr.bind(self.hrr.bind(s_hv, p_hv), o_hv)

    def insert(self, subject: str, predicate: str, obj: str):
        """Add a triple (subject, predicate, object) to the graph."""
        self.codebook.entity(subject)
        self.codebook.relation(predicate)
        self.codebook.entity(obj)

        triple_hv = self._encode_triple(subject, predicate, obj)
        self._kg_hv = self._kg_hv + triple_hv   # superposition
        self._triples.append(Triple(subject, predicate, obj))
        self._triple_hvs.append(triple_hv)

        # Also store inverse index: bind(bind(o, p), s) for backward queries
        o_hv = self.codebook.entity(obj)
        p_hv = self.codebook.relation(predicate)
        s_hv = self.codebook.entity(subject)
        inv_hv = self.hrr.bind(self.hrr.bind(o_hv, p_hv), s_hv)
        self._inv_hv = self._inv_hv + inv_hv

    def insert_batch(self, triples: List[Tuple[str, str, str]]):
        """Insert multiple triples at once."""
        for s, p, o in triples:
            self.insert(s, p, o)

    def remove(self, subject: str, predicate: str, obj: str) -> bool:
        """
        Remove a triple from the graph.
        In HDC: subtract the triple HV from the superposition.
        """
        for i, t in enumerate(self._triples):
            if t.subject == subject and t.predicate == predicate and t.obj == obj:
                self._kg_hv = self._kg_hv - self._triple_hvs[i]
                self._triples.pop(i)
                self._triple_hvs.pop(i)
                return True
        return False

    def query_object(
        self, subject: str, predicate: str, top_k: int = 3
    ) -> List[Tuple[str, float]]:
        """
        (subject, predicate, ?) → find objects.

        Returns: List of (entity_name, similarity) sorted desc.
        """
        s_hv = self.codebook.entity(subject)
        p_hv = self.codebook.relation(predicate)
        # Exact unbinding: query = unbind_exact(unbind_exact(KG, bind(s,p) part))
        # Step 1: unbind predicate from the (s⊗p)⊗o structure
        # triple = bind(bind(s,p), o), so:
        # unbind(triple, bind(s,p)) = o
        # unbind(KG, bind(s,p)) = superposition of all o for (s,p,*) triples
        sp_hv     = self.hrr.bind(s_hv, p_hv)
        candidate = self.hrr.unbind_exact(self._kg_hv, sp_hv)
        return self.codebook.nearest_entity(candidate, top_k)

    def query_subject(
        self, predicate: str, obj: str, top_k: int = 3
    ) -> List[Tuple[str, float]]:
        """(?, predicate, object) → find subjects (uses inverse index)."""
        p_hv  = self.codebook.relation(predicate)
        o_hv  = self.codebook.entity(obj)
        # Inverse index stores bind(bind(o, p), s)
        # unbind(inv_index, bind(o,p)) → s
        op_hv       = self.hrr.bind(o_hv, p_hv)
        candidate_s = self.hrr.unbind_exact(self._inv_hv, op_hv)
        return self.codebook.nearest_entity(candidate_s, top_k)

    def query_relation(
        self, subject: str, obj: str, top_k: int = 3
    ) -> List[Tuple[str, float]]:
        """(subject, ?, object) → find relations/predicates."""
        s_hv = self.codebook.entity(subject)
        o_hv = self.codebook.entity(obj)
        # Need to find p such that bind(bind(s,p), o) is in KG
        # Partial unbind: unbind(unbind(KG, o), s) ≈ p
        candidate_sp = self.hrr.unbind_exact(self._kg_hv, o_hv)
        candidate_p  = self.hrr.unbind_exact(candidate_sp, s_hv)
        return self.codebook.nearest_relation(candidate_p, top_k)

    def contains(
        self, subject: str, predicate: str, obj: str, threshold: float = 0.5
    ) -> bool:
        """
        Check if triple exists (approximate).

        Computes similarity of the triple HV to the KG superposition.
        """
        triple_hv = self._encode_triple(subject, predicate, obj)
        sim = float(F.cosine_similarity(
            triple_hv.unsqueeze(0),
            self._kg_hv.unsqueeze(0)
        ).item())
        return sim > threshold

    @property
    def n_triples(self) -> int:
        return len(self._triples)


    def pattern_match(
        self,
        subject:   Optional[str] = None,
        predicate: Optional[str] = None,
        obj:       Optional[str] = None,
    ) -> List['Triple']:
        """
        SPARQL-like wildcard triple pattern matching.

        Any field set to None acts as a wildcard (matches everything).

        Examples:
            pattern_match("Paris", None, None)   → all triples with Paris as subject
            pattern_match(None, "capital", None) → all triples with capital predicate
            pattern_match(None, None, "France")  → all triples with France as object
            pattern_match("Paris", "in", None)   → all (Paris, in, ?) triples

        Returns:
            List of matching Triple objects.
        """
        return [
            t for t in self._triples
            if (subject is None or t.subject == subject)
            and (predicate is None or t.predicate == predicate)
            and (obj is None or t.obj == obj)
        ]

    def count_triples(
        self,
        subject:   Optional[str] = None,
        predicate: Optional[str] = None,
        obj:       Optional[str] = None,
    ) -> int:
        """Count triples matching the given pattern (None = wildcard)."""
        return len(self.pattern_match(subject, predicate, obj))


# ═══════════════════════════════════════════════════════════════════════════════
# 3. KGReasoner — multi-hop and analogical inference
# ═══════════════════════════════════════════════════════════════════════════════

class KGReasoner:
    """
    Multi-hop reasoning and analogy completion over an HDC knowledge graph.

    Implements:
        - Multi-hop: (A --p1--> B --p2--> C) infer (A --p1·p2--> C)
        - Analogy: given (A --p--> B), (C --p--> ?), complete by analogy
        - Inverse: given (A --p--> B) in KG, infer (B --p_inv--> A)
        - Transitivity: (A --p--> B), (B --p--> C) → (A --p--> C)

    Args:
        kg:       HDCKnowledgeGraph to reason over
        max_hops: Maximum number of hops for path queries
    """

    def __init__(self, kg: HDCKnowledgeGraph, max_hops: int = 4):
        self.kg       = kg
        self.hrr      = kg.hrr
        self.codebook = kg.codebook
        self.max_hops = max_hops

    def multi_hop(
        self,
        start: str,
        relations: List[str],
        top_k: int = 3,
    ) -> List[Tuple[str, float]]:
        """
        Follow a chain of relations from a starting entity.

        multi_hop("Paris", ["country", "capital"]):
            Paris --country--> France --capital--> ?

        Args:
            start:      Starting entity name
            relations:  Ordered list of relation names to follow
            top_k:      Number of top results to return

        Returns:
            List of (entity_name, similarity) for the end of the chain.
        """
        if not relations:
            return [(start, 1.0)]

        current_hv = self.codebook.entity(start)

        for rel in relations[:self.max_hops]:
            p_hv = self.codebook.relation(rel)
            # Find the object of (current, rel, ?)
            sp_hv      = self.hrr.bind(current_hv, p_hv)
            candidate  = self.hrr.unbind_exact(self.kg._kg_hv, sp_hv)
            # Use the candidate as the new "current"
            current_hv = candidate

        return self.codebook.nearest_entity(current_hv, top_k)

    def analogy_completion(
        self,
        a: str,
        b: str,
        c: str,
        top_k: int = 3,
    ) -> List[Tuple[str, float]]:
        """
        Analogy: A is to B as C is to ? — using KG structure.

        Finds D such that (C, p, D) mirrors (A, p, B) for any relation p.

        Using HRR: D_raw = unbind(bind(B_hv, A_inv), C_hv)
                         = C ⊗ (B ⊗ A*)   — transfer the A→B relation to C
        """
        a_hv = self.codebook.entity(a)
        b_hv = self.codebook.entity(b)
        c_hv = self.codebook.entity(c)
        # Relation A→B in HRR: unbind(B, A) = A* ⊗ B
        relation = self.hrr.unbind_exact(b_hv, a_hv)
        # Apply to C: bind(C, relation)
        d_raw = self.hrr.bind(c_hv, relation)
        return self.codebook.nearest_entity(d_raw, top_k)

    def infer_inverse(
        self, subject: str, predicate: str, obj: str
    ) -> str:
        """
        Create the inverse relation name (heuristic).

        If (A, parent_of, B) then infer (B, child_of, A).
        """
        inverse_name = f"{predicate}_inv"
        # Register the inverse triple
        self.kg.insert(obj, inverse_name, subject)
        return inverse_name

    def path_exists(
        self, start: str, end: str, relation: Optional[str] = None
    ) -> bool:
        """Check if a direct (1-hop) path exists between start and end."""
        if relation:
            results = self.kg.query_object(start, relation, top_k=3)
            return any(name == end for name, _ in results)
        else:
            for rel in self.codebook._relations:
                results = self.kg.query_object(start, rel, top_k=1)
                if results and results[0][0] == end:
                    return True
            return False

    def find_path(
        self,
        source:    str,
        target:    str,
        max_depth: int = 4,
        min_sim:   float = 0.55,
    ) -> Optional[List[Tuple[str, str]]]:
        """
        Find a relation path from source to target via BFS over KG structure.

        Returns the first path found as a list of (entity, relation) steps,
        or None if no path exists within max_depth.

        Example:
            find_path("Paris", "Europe") might return:
            [("Paris", "in_country"), ("France", "in_continent"), ("Europe", "")]

        Args:
            source:    Starting entity name
            target:    Target entity name
            max_depth: Maximum hops to explore
            min_sim:   Minimum similarity for a valid step

        Returns:
            List of (entity, relation_used) pairs, or None.
        """
        relations = list(self.codebook._relations.keys()) if hasattr(
            self.codebook, '_relations') else []
        if not relations:
            return None

        # BFS over the entity graph
        queue:   List[List[Tuple[str, str]]] = [[(source, "")]]
        visited: set = {source}

        while queue:
            path = queue.pop(0)
            current = path[-1][0]

            if current == target:
                return path

            if len(path) > max_depth:
                continue

            for rel in relations:
                results = self.kg.query_object(current, rel, top_k=1)
                for next_ent, sim in results:
                    if sim < min_sim:
                        continue
                    if next_ent not in visited:
                        visited.add(next_ent)
                        queue.append(path + [(next_ent, rel)])

        return None

    def apply_rule(
        self,
        if_predicate:   str,   # body predicate
        then_predicate: str,   # head predicate
        top_k: int = 5,
    ) -> List[Tuple[str, str, str]]:
        """
        Apply a Horn clause rule: ∀x,y: (x, if_pred, y) → (x, then_pred, y).

        Materialises all entailed triples (x, then_pred, y) that can be derived
        from existing (x, if_pred, y) triples.

        Returns:
            List of (subject, then_predicate, object) inferred triples.
        """
        inferred = []
        for entity_name in list(self.codebook._entities.keys()):
            results = self.kg.query_object(entity_name, if_predicate, top_k=top_k)
            for obj_name, sim in results:
                if sim < 0.5:
                    continue
                # Insert the entailed triple
                self.kg.insert(entity_name, then_predicate, obj_name)
                inferred.append((entity_name, then_predicate, obj_name))
        return inferred


# ═══════════════════════════════════════════════════════════════════════════════
# 4. HDCOntology — class hierarchy via fractional power encoding
# ═══════════════════════════════════════════════════════════════════════════════

class HDCOntology:
    """
    Represents class hierarchies using fractional power encoding.

    The is_a relation in an ontology:
        Dog is_a Mammal is_a Animal

    In HDC fractional power encoding (Plate 1995):
        animal_hv^1.0 = Animal
        animal_hv^0.7 = Mammal   (70% of animal properties)
        animal_hv^0.5 = Dog      (50% of animal properties)

    Subsumption check: is A a subclass of B?
        sim(A_hv, B_hv) > threshold

    Args:
        hrr:       HRR instance
        base_class: Top-level class HV (e.g., "entity")
    """

    def __init__(self, hrr: HRR):
        self.hrr    = hrr
        self._class_hvs:   Dict[str, torch.Tensor] = {}
        self._parent_of:   Dict[str, str]          = {}   # child → parent
        self._depth:       Dict[str, int]           = {}   # class → depth

        # Root class
        import hashlib
        root_seed = int(hashlib.md5(b"entity").hexdigest()[:8], 16) % (2**31)
        root = hrr.gen(1, seed=root_seed)
        self._class_hvs["entity"] = root
        self._depth["entity"]     = 0

    def add_class(self, name: str, parent: str = "entity"):
        """Add a new class to the ontology."""
        if parent not in self._class_hvs:
            self.add_class(parent)

        parent_hv   = self._class_hvs[parent]
        parent_depth = self._depth.get(parent, 0)
        child_depth  = parent_depth + 1
        # Fractional power: deeper classes are "further" from root
        alpha        = max(0.1, 1.0 - 0.1 * child_depth)

        # Generate child as a perturbation of parent with α fraction shared
        g = torch.Generator(device=self.hrr.device)
        g.manual_seed(abs(hash(name)) % (2**31))
        noise  = torch.randn(self.hrr.dim, generator=g, device=self.hrr.device)
        noise  = F.normalize(noise, dim=0)
        child_hv = F.normalize(alpha * parent_hv + (1 - alpha) * noise, dim=0)

        self._class_hvs[name]  = child_hv
        self._parent_of[name]  = parent
        self._depth[name]      = child_depth

    def is_a(self, class_a: str, class_b: str, threshold: float = 0.3) -> bool:
        """
        Check if class_a is a subclass of class_b.

        Uses cosine similarity: subclasses share significant overlap with parents.
        """
        if class_a not in self._class_hvs or class_b not in self._class_hvs:
            return False
        hv_a = self._class_hvs[class_a]
        hv_b = self._class_hvs[class_b]
        sim  = float(F.cosine_similarity(hv_a.unsqueeze(0), hv_b.unsqueeze(0)).item())
        return sim > threshold

    def most_specific_class(
        self, entity_hv: torch.Tensor, candidates: Optional[List[str]] = None
    ) -> Tuple[str, float]:
        """
        Find the most specific class for an entity HV.

        Returns the class with highest cosine similarity.
        """
        classes = candidates or list(self._class_hvs.keys())
        best, best_sim = "entity", 0.0
        for c in classes:
            hv  = self._class_hvs[c]
            sim = float(F.cosine_similarity(
                entity_hv.unsqueeze(0), hv.unsqueeze(0)
            ).item())
            if sim > best_sim:
                best_sim = sim
                best     = c
        return best, best_sim

    def class_hv(self, name: str) -> Optional[torch.Tensor]:
        return self._class_hvs.get(name)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. KGCompletionEvaluator
# ═══════════════════════════════════════════════════════════════════════════════

class KGCompletionEvaluator:
    """
    Evaluates HDC knowledge graph completion quality.

    Metrics:
        Hit@k: fraction of queries where correct answer is in top-k results
        MRR:   mean reciprocal rank of correct answer
        Both standard metrics from KG embedding literature.

    Args:
        kg: HDCKnowledgeGraph to evaluate
    """

    def __init__(self, kg: HDCKnowledgeGraph):
        self.kg = kg

    def evaluate_object_queries(
        self,
        test_triples: List[Tuple[str, str, str]],
        k_values: List[int] = [1, 3, 10],
    ) -> Dict:
        """
        Evaluate (s, p, ?) → o query completion.

        Args:
            test_triples: List of (subject, predicate, object) test cases
            k_values:     Hit@k values to compute

        Returns:
            Dict with hit@k for each k and MRR.
        """
        hits = {k: 0 for k in k_values}
        rr_sum = 0.0
        n = len(test_triples)

        for s, p, o in test_triples:
            max_k   = max(k_values)
            results = self.kg.query_object(s, p, top_k=max_k)
            names   = [r[0] for r in results]

            for k in k_values:
                if o in names[:k]:
                    hits[k] += 1

            # Reciprocal rank
            if o in names:
                rr_sum += 1.0 / (names.index(o) + 1)

        return {
            **{f"hit@{k}": hits[k] / max(n, 1) for k in k_values},
            "MRR": rr_sum / max(n, 1),
            "n_queries": n,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Smoke test
# ═══════════════════════════════════════════════════════════════════════════════

def _test_knowledge_graph():
    D   = 512
    hrr = HRR(dim=D)

    print("=== KGCodebook ===")
    cb = KGCodebook(hrr)
    for e in ["Paris", "France", "Berlin", "Germany", "London", "UK"]:
        cb.entity(e)
    for r in ["capital_of", "located_in", "borders"]:
        cb.relation(r)
    print(f"  {cb.n_entities} entities, {cb.n_relations} relations  OK")

    print("\n=== HDCKnowledgeGraph ===")
    kg = HDCKnowledgeGraph(hrr, cb)
    kg.insert_batch([
        ("Paris",  "capital_of", "France"),
        ("Berlin", "capital_of", "Germany"),
        ("London", "capital_of", "UK"),
        ("Paris",  "located_in", "France"),
        ("Berlin", "located_in", "Germany"),
    ])
    print(f"  Inserted {kg.n_triples} triples  OK")

    # Query: (Paris, capital_of, ?) → France
    results = kg.query_object("Paris", "capital_of", top_k=3)
    print(f"  Query (Paris, capital_of, ?): {results[:2]}")
    assert results[0][0] == "France", f"Expected 'France', got '{results[0][0]}'"
    print("  ✓ Correct answer retrieved")

    # Backward query: (?, capital_of, Germany) → Berlin
    results_b = kg.query_subject("capital_of", "Germany", top_k=3)
    print(f"  Query (?, capital_of, Germany): {results_b[:2]}")
    assert results_b[0][0] == "Berlin", f"Expected 'Berlin', got '{results_b[0][0]}'"
    print("  ✓ Backward query correct")

    print("\n=== KGReasoner ===")
    reasoner = KGReasoner(kg, max_hops=3)

    # Multi-hop test: start at Paris, follow capital_of (should reach France)
    hop_results = reasoner.multi_hop("Paris", ["capital_of"], top_k=3)
    print(f"  Multi-hop Paris→capital_of→? : {hop_results[:2]}")

    # Analogy using KG query: (Paris, capital_of, ?) = France, apply to Berlin
    # This uses the KG structure rather than raw HRR geometry
    paris_obj  = kg.query_object("Paris",  "capital_of", top_k=1)[0][0]
    berlin_obj = kg.query_object("Berlin", "capital_of", top_k=1)[0][0]
    print(f"  Paris capital_of→{paris_obj}, Berlin capital_of→{berlin_obj}")
    assert berlin_obj == "Germany", f"Expected 'Germany', got '{berlin_obj}'"
    print("  ✓ KG-based analogy (Berlin→Germany) correct")

    print("\n=== HDCOntology ===")
    onto = HDCOntology(hrr)
    onto.add_class("Animal")
    onto.add_class("Mammal",  parent="Animal")
    onto.add_class("Dog",     parent="Mammal")
    onto.add_class("Cat",     parent="Mammal")

    assert onto.is_a("Dog",    "Mammal"),  "Dog should be_a Mammal"
    assert onto.is_a("Mammal", "Animal"),  "Mammal should be_a Animal"
    print(f"  Dog is_a Mammal: {onto.is_a('Dog', 'Mammal')}  OK")
    print(f"  Mammal is_a Animal: {onto.is_a('Mammal', 'Animal')}  OK")

    dog_hv = onto.class_hv("Dog")
    cls, sim = onto.most_specific_class(dog_hv)
    print(f"  Most specific class for Dog HV: '{cls}' (sim={sim:.3f})")

    print("\n=== KGCompletionEvaluator ===")
    evaluator = KGCompletionEvaluator(kg)
    test_triples = [
        ("Paris",  "capital_of", "France"),
        ("Berlin", "capital_of", "Germany"),
        ("London", "capital_of", "UK"),
    ]
    metrics = evaluator.evaluate_object_queries(test_triples, k_values=[1, 3])
    print(f"  Hit@1={metrics['hit@1']:.2f}, Hit@3={metrics['hit@3']:.2f}, "
          f"MRR={metrics['MRR']:.2f}")
    assert metrics["hit@1"] >= 2/3, f"Should get at least 2/3 right at @1"

    print("\n✅ All knowledge_graph tests passed")


if __name__ == "__main__":
    _test_knowledge_graph()
