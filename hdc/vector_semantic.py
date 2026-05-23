"""
Vector Semantic Representations for Visual Place Recognition
=============================================================
Based on Sutor et al. research program (University of Maryland, 2018-2025):

Core papers:
1. Sutor, Summers-Stay, Aloimonos (2018) "A Computational Theory for Life-Long 
   Learning of Semantics" — arXiv:1806.10755
   - Semantic vectors that evolve over time without catastrophic forgetting
   - New knowledge is bound, not overwritten

2. Summers-Stay, Sutor, Li (2018) "Representing Sets as Summed Semantic Vectors" 
   — arXiv:1809.08823
   - Sets as summed semantic vectors
   - Set operations (union, intersection, membership) via VSA

3. Mitrokhin, Sutor et al. (2019) "Learning sensorimotor control with neuromorphic 
   sensors" — Science Robotics 4(30), eaaw6736
   - HAP: Hyperdimensional Active Perception
   - Time-slice encoding of visual events into hypervectors

4. Mitrokhin, Sutor et al. (2020) "Symbolic Representation and Learning With 
   Hyperdimensional Computing" — Frontiers in Robotics and AI 7, 63
   - HD-Glue: symbolic fusion via HDC consensus
   - HIL: Hyperdimensional Interactive Learning

5. Sutor et al. (2022) "Gluing Neural Networks Symbolically Through Hyperdimensional 
   Computing" — IJCNN 2022 / arXiv:2205.15534
   - Weighted consensus for multi-model fusion
   - Binary hypervectors for efficient edge deployment

6. Sutor et al. (2025) "Vector Semantic Representations as Descriptors for Visual 
   Place Recognition" — In preparation
   - VSA-based place recognition using semantic hypervectors
   - Place descriptors that preserve spatial relationships

Key insight: Vector Semantic Representations (VSR) use hypervectors as universal
descriptors that can represent any modality (visual, spatial, semantic) in a common
VSA space. Place recognition becomes a simple Hamming distance computation.

Architecture:
    Visual Features → Semantic Hypervector → Place Descriptor
    Place Descriptor ⊕ Context → Bound Representation
    Query → Hamming Similarity → Place Recognition

All operations are pure VSA: XOR, popcount, bundle, permute.
No neural networks, no backpropagation, no gradient descent.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Dict, Any
from hdc.hdc_glue import (
    hv_xor, hv_popcount, hv_hamming_sim, hv_bundle, 
    hv_permute, hv_majority, hv_batch_sim, gen_hvs,
    HolographicEncoder, ChimeraEngine
)


# ═══════════════════════════════════════════════════════════════════════════════
# Semantic Hypervector Encoder — Sutor's "semantic vectors that evolve"
# ═══════════════════════════════════════════════════════════════════════════════

class SemanticVectorEncoder(nn.Module):
    """
    Encodes any feature vector into a semantic hypervector.
    
    Based on Sutor et al. (2018) "A Computational Theory for Life-Long Learning 
    of Semantics":
    - Semantic vectors are learned from data to express semantic relationships
    - New knowledge is bound, not overwritten (no catastrophic forgetting)
    - VSA operations preserve semantic structure
    
    The encoder uses:
    1. Random basis hypervectors for each feature dimension
    2. Threshold-based encoding (active → key, inactive → flipped key)
    3. Majority vote bundling
    
    This is a pure VSA encoder. No neural networks, no backpropagation.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 10000,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Random basis hypervectors — one per input dimension
        self.register_buffer(
            "basis",
            gen_hvs(input_dim, output_dim, seed=seed),
        )
        
        # Inverted basis for inactive features
        self.register_buffer(
            "not_basis",
            1.0 - self.basis,
        )
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a feature vector into a semantic hypervector.
        
        For each feature i:
          if x[i] > 0.5: hv += basis[i]    (feature active)
          else:          hv += not_basis[i] (feature inactive)
        
        Then majority vote.
        
        Args:
            x: (input_dim,) feature vector
        
        Returns:
            (output_dim,) binary semantic hypervector
        """
        active = (x > 0.5).float()
        inactive = 1.0 - active
        
        hv = (active.unsqueeze(1) * self.basis).sum(dim=0) + \
             (inactive.unsqueeze(1) * self.not_basis).sum(dim=0)
        
        return hv_majority(hv)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            return self.encode(x)
        return torch.stack([self.encode(x[i]) for i in range(x.shape[0])])


# ═══════════════════════════════════════════════════════════════════════════════
# Place Descriptor — Sutor's "Vector Semantic Representations for VPR"
# ═══════════════════════════════════════════════════════════════════════════════

class PlaceDescriptor(nn.Module):
    """
    Place descriptor using Vector Semantic Representations.
    
    Based on Sutor et al. (2025) "Vector Semantic Representations as Descriptors 
    for Visual Place Recognition":
    - Visual features are encoded into semantic hypervectors
    - Multiple views of the same place are bundled into a single descriptor
    - Place descriptors can be compared via Hamming distance
    - Context (time, weather, season) can be bound to place descriptors
    
    The descriptor is a binary hypervector that represents a place.
    Similar places have similar hypervectors (high Hamming similarity).
    Different places have different hypervectors (low Hamming similarity).
    
    This enables:
    - Place recognition: query → Hamming similarity → nearest place
    - Place comparison: Hamming distance between descriptors
    - Place composition: bundle multiple place descriptors
    - Place-context binding: bind place with context hypervector
    
    All operations are pure VSA. No neural networks, no backpropagation.
    """
    
    def __init__(
        self,
        feature_dim: int,
        dim: int = 10000,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.dim = dim
        
        # Semantic encoder for visual features
        self.encoder = SemanticVectorEncoder(
            input_dim=feature_dim,
            output_dim=dim,
            seed=seed,
        )
        
        # Place memory: dict of {place_id: accumulated_hypervector}
        self.place_memory: Dict[int, torch.Tensor] = {}
        
        # Place counts for retroactive interference prevention
        self.place_counts: Dict[int, int] = {}
        
        # Context hypervectors (time, weather, season, etc.)
        self.context_hvs: Dict[str, torch.Tensor] = {}
        
        # Maximum count before saturation
        self.max_count = 1000
    
    def encode_view(self, features: torch.Tensor) -> torch.Tensor:
        """Encode a single view into a semantic hypervector.
        
        Args:
            features: (feature_dim,) visual features
        
        Returns:
            (dim,) semantic hypervector
        """
        return self.encoder.encode(features)
    
    def add_view(self, place_id: int, features: torch.Tensor):
        """Add a view to a place descriptor.
        
        Multiple views of the same place are bundled into a single descriptor.
        This is the VSA equivalent of "averaging" views.
        
        Args:
            place_id: Unique place identifier
            features: (feature_dim,) visual features
        """
        hv = self.encode_view(features)
        
        if place_id not in self.place_memory:
            self.place_memory[place_id] = hv.clone()
            self.place_counts[place_id] = 1
        else:
            count = self.place_counts[place_id]
            if count < self.max_count:
                self.place_memory[place_id] = self.place_memory[place_id] + hv
            else:
                # Running average: bounded influence
                self.place_memory[place_id] = (
                    self.place_memory[place_id] * (count / (count + 1)) + hv / (count + 1)
                )
            self.place_counts[place_id] += 1
    
    def finalize(self):
        """Finalize all place descriptors via majority vote."""
        for place_id in self.place_memory:
            self.place_memory[place_id] = hv_majority(self.place_memory[place_id])
    
    def get_descriptor(self, place_id: int) -> Optional[torch.Tensor]:
        """Get the descriptor for a place.
        
        Args:
            place_id: Unique place identifier
        
        Returns:
            (dim,) binary hypervector, or None if place not found
        """
        return self.place_memory.get(place_id)
    
    def query(self, features: torch.Tensor) -> Tuple[Optional[int], torch.Tensor, float]:
        """Query the place memory.
        
        Returns the nearest place and its similarity.
        
        Args:
            features: (feature_dim,) visual features
        
        Returns:
            (place_id, similarities, max_similarity)
        """
        if not self.place_memory:
            return None, torch.tensor([]), 0.0
        
        hv = self.encode_view(features)
        
        # Compute Hamming similarity to all places
        place_ids = list(self.place_memory.keys())
        prototypes = torch.stack([self.place_memory[pid] for pid in place_ids])
        
        similarities = hv_batch_sim(hv, prototypes)
        max_sim, max_idx = similarities.max(dim=0)
        
        return place_ids[max_idx.item()], similarities, max_sim.item()
    
    def bind_context(self, place_id: int, context_name: str, context_hv: torch.Tensor) -> torch.Tensor:
        """Bind a place descriptor with context.
        
        This enables context-dependent place recognition.
        E.g., same place in different seasons → different bound descriptors.
        
        Args:
            place_id: Unique place identifier
            context_name: Name of the context (e.g., "summer", "night")
            context_hv: (dim,) context hypervector
        
        Returns:
            (dim,) bound hypervector (place ⊕ context)
        """
        descriptor = self.get_descriptor(place_id)
        if descriptor is None:
            raise ValueError(f"Place {place_id} not found")
        
        self.context_hvs[context_name] = context_hv
        return hv_xor(descriptor, context_hv)
    
    def query_with_context(
        self, 
        features: torch.Tensor, 
        context_name: str
    ) -> Tuple[Optional[int], torch.Tensor, float]:
        """Query place memory with context binding.
        
        Args:
            features: (feature_dim,) visual features
            context_name: Name of the context
        
        Returns:
            (place_id, similarities, max_similarity)
        """
        if context_name not in self.context_hvs:
            return self.query(features)
        
        hv = self.encode_view(features)
        context_hv = self.context_hvs[context_name]
        
        # Bind query with context
        bound_hv = hv_xor(hv, context_hv)
        
        # Compare to context-bound place descriptors
        place_ids = list(self.place_memory.keys())
        prototypes = torch.stack([
            hv_xor(self.place_memory[pid], context_hv) 
            for pid in place_ids
        ])
        
        similarities = hv_batch_sim(bound_hv, prototypes)
        max_sim, max_idx = similarities.max(dim=0)
        
        return place_ids[max_idx.item()], similarities, max_sim.item()
    
    def place_similarity(self, place_a: int, place_b: int) -> float:
        """Compute similarity between two place descriptors.
        
        Args:
            place_a: First place ID
            place_b: Second place ID
        
        Returns:
            Hamming similarity between the two descriptors
        """
        desc_a = self.get_descriptor(place_a)
        desc_b = self.get_descriptor(place_b)
        
        if desc_a is None or desc_b is None:
            return 0.0
        
        return hv_hamming_sim(desc_a, desc_b).item()
    
    def bundle_places(self, place_ids: List[int]) -> torch.Tensor:
        """Bundle multiple place descriptors into a route descriptor.
        
        This enables route-level place recognition.
        A route is the bundle of all places along a path.
        
        Args:
            place_ids: List of place IDs to bundle
        
        Returns:
            (dim,) bundled hypervector
        """
        descriptors = []
        for pid in place_ids:
            desc = self.get_descriptor(pid)
            if desc is not None:
                descriptors.append(desc)
        
        if not descriptors:
            return torch.zeros(self.dim)
        
        bundled = hv_bundle(torch.stack(descriptors))
        return hv_majority(bundled)
    
    def size(self) -> int:
        """Number of places in memory."""
        return len(self.place_memory)


# ═══════════════════════════════════════════════════════════════════════════════
# Set Operations — Summers-Stay, Sutor (2018) "Representing Sets as Summed 
# Semantic Vectors"
# ═══════════════════════════════════════════════════════════════════════════════

class SemanticSet(nn.Module):
    """
    Set operations using Vector Semantic Representations.
    
    Based on Summers-Stay, Sutor, Li (2018) "Representing Sets as Summed 
    Semantic Vectors":
    - Sets are represented as summed semantic vectors
    - Set operations (union, intersection, membership) via VSA
    - The sum of element vectors preserves set membership information
    
    Key insight: The sum of element hypervectors is itself a hypervector
    that represents the set. Set operations become VSA operations.
    
    Operations:
    - Union: bundle(set_a, set_b) → majority vote
    - Intersection: approximate via similarity threshold
    - Membership: Hamming similarity to set descriptor
    - Subset: all elements of A are in B
    """
    
    def __init__(
        self,
        dim: int = 10000,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.dim = dim
        
        # Element hypervectors
        self.elements: Dict[str, torch.Tensor] = {}
        
        # Set descriptors: {set_name: bundled_hypervector}
        self.sets: Dict[str, torch.Tensor] = {}
        
        # Set element membership: {set_name: set_of_element_names}
        self.set_elements: Dict[str, set] = {}
        
        # Random generator for new elements
        self.seed = seed
        self.next_seed = seed if seed is not None else 42
    
    def add_element(self, name: str) -> torch.Tensor:
        """Add a new element to the universe.
        
        Each element gets a unique random hypervector.
        
        Args:
            name: Element name
        
        Returns:
            (dim,) element hypervector
        """
        if name not in self.elements:
            hv = gen_hvs(1, self.dim, seed=self.next_seed).squeeze(0)
            self.elements[name] = hv
            self.next_seed += 1
        return self.elements[name]
    
    def get_element(self, name: str) -> Optional[torch.Tensor]:
        """Get the hypervector for an element.
        
        Args:
            name: Element name
        
        Returns:
            (dim,) element hypervector, or None
        """
        return self.elements.get(name)
    
    def create_set(self, name: str, element_names: List[str]) -> torch.Tensor:
        """Create a set from a list of element names.
        
        The set descriptor is the bundled sum of its element hypervectors.
        
        Args:
            name: Set name
            element_names: List of element names in the set
        
        Returns:
            (dim,) set descriptor hypervector
        """
        # Ensure all elements exist
        hvs = []
        for ename in element_names:
            hv = self.add_element(ename)
            hvs.append(hv)
        
        # Bundle all element hypervectors
        if len(hvs) == 1:
            set_hv = hvs[0].clone()
        else:
            set_hv = hv_bundle(torch.stack(hvs))
            set_hv = hv_majority(set_hv)
        
        self.sets[name] = set_hv
        self.set_elements[name] = set(element_names)
        
        return set_hv
    
    def union(self, set_a: str, set_b: str, result_name: str) -> torch.Tensor:
        """Union of two sets.
        
        Union = bundle(set_a, set_b) → majority vote.
        
        Args:
            set_a: First set name
            set_b: Second set name
            result_name: Name for the result set
        
        Returns:
            (dim,) union descriptor hypervector
        """
        hv_a = self.sets.get(set_a)
        hv_b = self.sets.get(set_b)
        
        if hv_a is None or hv_b is None:
            raise ValueError(f"Set not found: {set_a if hv_a is None else set_b}")
        
        union_hv = hv_majority(hv_a + hv_b)
        
        self.sets[result_name] = union_hv
        elements_a = self.set_elements.get(set_a, set())
        elements_b = self.set_elements.get(set_b, set())
        self.set_elements[result_name] = elements_a | elements_b
        
        return union_hv
    
    def intersection(self, set_a: str, set_b: str, threshold: float = 0.6) -> List[str]:
        """Approximate intersection of two sets.
        
        Elements are in the intersection if they have high similarity
        to both set descriptors.
        
        Args:
            set_a: First set name
            set_b: Second set name
            threshold: Similarity threshold for membership
        
        Returns:
            List of element names in the intersection
        """
        hv_a = self.sets.get(set_a)
        hv_b = self.sets.get(set_b)
        
        if hv_a is None or hv_b is None:
            raise ValueError(f"Set not found")
        
        # Find elements that are similar to both sets
        intersection = []
        for ename, ehv in self.elements.items():
            sim_a = hv_hamming_sim(ehv, hv_a).item()
            sim_b = hv_hamming_sim(ehv, hv_b).item()
            if sim_a > threshold and sim_b > threshold:
                intersection.append(ename)
        
        return intersection
    
    def membership(self, element_name: str, set_name: str) -> float:
        """Check if an element is in a set.
        
        Returns the Hamming similarity between the element and set descriptor.
        Higher values indicate stronger membership.
        
        Args:
            element_name: Element name
            set_name: Set name
        
        Returns:
            Similarity score (0-1)
        """
        ehv = self.elements.get(element_name)
        shv = self.sets.get(set_name)
        
        if ehv is None or shv is None:
            return 0.0
        
        return hv_hamming_sim(ehv, shv).item()
    
    def is_subset(self, set_a: str, set_b: str, threshold: float = 0.6) -> bool:
        """Check if set_a is a subset of set_b.
        
        All elements of set_a must have high similarity to set_b.
        
        Args:
            set_a: Potential subset
            set_b: Potential superset
            threshold: Similarity threshold
        
        Returns:
            True if set_a ⊆ set_b
        """
        elements_a = self.set_elements.get(set_a, set())
        for ename in elements_a:
            sim = self.membership(ename, set_b)
            if sim < threshold:
                return False
        return True
    
    def get_set_descriptor(self, set_name: str) -> Optional[torch.Tensor]:
        """Get the descriptor for a set.
        
        Args:
            set_name: Set name
        
        Returns:
            (dim,) set descriptor hypervector
        """
        return self.sets.get(set_name)


# ═══════════════════════════════════════════════════════════════════════════════
# Knowledge Graph — Sutor, Summers-Stay, Aloimonos (2018) §3
# arXiv:1806.10755
#
# Vertices are concepts; edges are weighted co-occurrence counts.
# The graph drives the tension optimizer: connected nodes should have
# similar HVs, disconnected nodes should have dissimilar HVs.
# ═══════════════════════════════════════════════════════════════════════════════

class KnowledgeGraph:
    """Weighted co-occurrence graph over semantic concepts.

    Vertices are concepts, each with a binary hypervector.
    Edges are weighted by co-occurrence frequency observed in the input stream.

    After accumulating edges from a data stream, calling
    ``TensionOptimizer.minimize_tension(graph)`` refines the HV assignments
    so that related concepts become more similar and unrelated ones diverge.

    Reference:
        Sutor, Summers-Stay, Aloimonos (2018)
        "A Computational Theory for Life-Long Learning of Semantics"
        arXiv:1806.10755, §3 "Geometric Foundation"
    """

    def __init__(self, dim: int = 10000, seed: Optional[int] = None):
        self.dim = dim
        self._seed = seed
        self._counter = 0

        # concept → HV
        self.vertices: Dict[str, torch.Tensor] = {}
        # (concept_a, concept_b) → float weight
        self.edges: Dict[Tuple[str, str], float] = {}

    # ── vertex management ──────────────────────────────────────────────────────

    def _new_hv(self) -> torch.Tensor:
        seed = (self._seed + self._counter) if self._seed is not None else None
        hv = gen_hvs(1, self.dim, seed=seed).squeeze(0)
        self._counter += 1
        return hv

    def add_concept(self, name: str) -> torch.Tensor:
        """Add a concept vertex (no-op if already exists)."""
        if name not in self.vertices:
            self.vertices[name] = self._new_hv()
        return self.vertices[name]

    def get_hv(self, name: str) -> Optional[torch.Tensor]:
        return self.vertices.get(name)

    def set_hv(self, name: str, hv: torch.Tensor) -> None:
        """Replace a concept's HV (used by TensionOptimizer)."""
        self.vertices[name] = hv.clone()

    # ── edge management ────────────────────────────────────────────────────────

    def add_cooccurrence(self, a: str, b: str, weight: float = 1.0) -> None:
        """Record that concept a co-occurred with concept b.

        Undirected: weight stored on the canonical (min, max) key.
        Both concepts are auto-added if not already present.
        """
        self.add_concept(a)
        self.add_concept(b)
        key = (min(a, b), max(a, b))
        self.edges[key] = self.edges.get(key, 0.0) + weight

    def build_from_sequences(
        self,
        sequences: List[List[str]],
        window: int = 2,
    ) -> None:
        """Populate edges from co-occurrences within a sliding window.

        Args:
            sequences: List of token sequences (each a list of concept names)
            window: Co-occurrence window size (pairs within this distance)
        """
        for seq in sequences:
            for i, tok in enumerate(seq):
                self.add_concept(tok)
                for j in range(i + 1, min(i + window + 1, len(seq))):
                    self.add_cooccurrence(tok, seq[j], weight=1.0 / (j - i))

    def neighbours(self, concept: str) -> List[Tuple[str, float]]:
        """Return all neighbours of a concept with their edge weights."""
        result = []
        for (a, b), w in self.edges.items():
            if a == concept:
                result.append((b, w))
            elif b == concept:
                result.append((a, w))
        return result

    @property
    def n_vertices(self) -> int:
        return len(self.vertices)

    @property
    def n_edges(self) -> int:
        return len(self.edges)


# ═══════════════════════════════════════════════════════════════════════════════
# Tension Optimizer — Sutor (2018) §3.2–3.3
# arXiv:1806.10755
#
# Minimizes T(A) = Σ_{(i,k)∈E} [ W_ik · H(A_i, A_k)  +  1 / H(A_i, A_k)² ]
#
# The two terms are opposing forces:
#   Connective: W_ik · H  — pulls connected nodes together (wants small H)
#   Proximal:   1 / H²   — repels all pairs (prevents collapse)
#
# Equilibrium Hamming distance for an edge of weight W:
#   H* = (2 / W)^(1/3)
# Strong co-occurrence (large W) → smaller H → more similar HVs.
#
# Greedy bit-flip descent: for each concept vertex i and each bit j,
# compute ΔT from flipping bit j; keep if ΔT < 0 (reduces tension).
# ═══════════════════════════════════════════════════════════════════════════════

class TensionOptimizer:
    """Greedy tension-minimization for semantic vector placement.

    Iteratively flips bits in concept HVs to reduce the total tension
    of the knowledge graph, making co-occurring concepts more similar
    and non-co-occurring concepts more dissimilar.

    This is the core algorithmic contribution of Sutor et al. 2018.
    Without it, the vectors are random and carry no semantic information.

    Reference:
        Sutor, Summers-Stay, Aloimonos (2018)
        "A Computational Theory for Life-Long Learning of Semantics"
        arXiv:1806.10755, §3.3 "Tension Minimization"
    """

    def __init__(
        self,
        c_conn: float = 1.0,
        c_prox: float = 1.0,
        min_hamming: int = 1,
    ):
        """
        Args:
            c_conn: Connective force coefficient (attraction along edges)
            c_prox: Proximal force coefficient (repulsion from proximity)
            min_hamming: Floor for H to avoid division by zero
        """
        self.c_conn = c_conn
        self.c_prox = c_prox
        self.min_hamming = min_hamming

    # ── Tension calculation ────────────────────────────────────────────────────

    def _edge_tension(self, h: float, w: float) -> float:
        h = max(h, self.min_hamming)
        return self.c_conn * w * h + self.c_prox / (h * h)

    def compute_tension(self, graph: KnowledgeGraph) -> float:
        """Compute total tension T(A) across all edges.

        T(A) = Σ_{(i,k)∈E} [ c_conn · W_ik · H(A_i,A_k) + c_prox / H(A_i,A_k)² ]
        """
        T = 0.0
        for (a, b), w in graph.edges.items():
            hv_a = graph.vertices[a]
            hv_b = graph.vertices[b]
            h = float(hv_xor(hv_a, hv_b).sum().item())
            T += self._edge_tension(h, w)
        return T

    # ── Greedy bit-flip descent ────────────────────────────────────────────────

    def _delta_tension_flip_bit(
        self,
        hv_i: torch.Tensor,
        neighbours: List[Tuple[torch.Tensor, float]],
        bit_j: int,
    ) -> float:
        """Compute ΔT if bit j in hv_i is flipped.

        For each neighbour k with weight w:
            h_cur = H(hv_i, hv_k)
            δ = +1 if hv_i[j] == hv_k[j] (flip increases H)
                -1 if hv_i[j] != hv_k[j] (flip decreases H)
            ΔT_edge = T(h_cur + δ, w) - T(h_cur, w)
        """
        delta = 0.0
        for hv_k, w in neighbours:
            h_cur = float(hv_xor(hv_i, hv_k).sum().item())
            same = (hv_i[bit_j].item() == hv_k[bit_j].item())
            dh = 1 if same else -1
            h_new = max(self.min_hamming, h_cur + dh)
            h_cur = max(self.min_hamming, h_cur)
            delta += self._edge_tension(h_new, w) - self._edge_tension(h_cur, w)
        return delta

    def _delta_tension_vectorized(
        self,
        hv_i: torch.Tensor,
        neighbour_hvs: torch.Tensor,   # (n_nbr, dim)
        neighbour_ws: torch.Tensor,    # (n_nbr,)
    ) -> torch.Tensor:
        """Vectorized ΔT for all dim bits simultaneously.

        Returns:
            (dim,) tensor of ΔT values per bit flip
        """
        dim = hv_i.shape[0]
        # Hamming distances to each neighbour: (n_nbr,)
        xors = hv_xor(hv_i.unsqueeze(0), neighbour_hvs)  # (n_nbr, dim)
        h_cur = xors.sum(dim=-1).float().clamp(min=self.min_hamming)  # (n_nbr,)

        # For each bit j: same_bits[k,j] = (hv_i[j] == nbr_k[j])
        same_bits = (hv_i.unsqueeze(0) == neighbour_hvs)  # (n_nbr, dim) bool
        # dh[k,j] = +1 if same, -1 if different
        dh = torch.where(same_bits, torch.ones_like(same_bits, dtype=torch.float),
                         -torch.ones_like(same_bits, dtype=torch.float))  # (n_nbr, dim)

        # h_new[k,j] = clamp(h_cur[k] + dh[k,j], min=min_hamming)
        h_new = (h_cur.unsqueeze(1) + dh).clamp(min=float(self.min_hamming))  # (n_nbr, dim)

        # Tension delta per edge per bit:
        # ΔT[k,j] = c_conn*w[k]*dh[k,j] + c_prox*(1/h_new[k,j]^2 - 1/h_cur[k]^2)
        w = neighbour_ws.unsqueeze(1)  # (n_nbr, 1)
        dt = (self.c_conn * w * dh +
              self.c_prox * (1.0 / h_new.pow(2) - 1.0 / h_cur.unsqueeze(1).pow(2)))  # (n_nbr, dim)

        return dt.sum(dim=0)  # (dim,) — sum over all neighbours

    def minimize_tension(
        self,
        graph: KnowledgeGraph,
        n_iterations: int = 20,
        verbose: bool = False,
    ) -> List[float]:
        """Run greedy bit-flip descent until convergence or n_iterations.

        Each iteration sweeps all vertices and flips bits that reduce T.
        Returns the tension history (one value per iteration).

        Args:
            graph: KnowledgeGraph whose vertex HVs will be updated in-place
            n_iterations: Maximum number of full sweeps
            verbose: Print progress

        Returns:
            List of tension values (length = n_iterations)
        """
        tension_history = []

        for iteration in range(n_iterations):
            n_flips = 0

            for concept, hv_i in list(graph.vertices.items()):
                nbrs = graph.neighbours(concept)
                if not nbrs:
                    continue

                # Collect neighbour HVs and weights
                nbr_hvs = torch.stack([graph.vertices[n] for n, _ in nbrs])
                nbr_ws = torch.tensor([w for _, w in nbrs], dtype=torch.float32)

                # Vectorized ΔT for all bits
                delta_T = self._delta_tension_vectorized(hv_i, nbr_hvs, nbr_ws)

                # Flip all bits where ΔT < 0 (independently — greedy)
                flip_mask = (delta_T < 0)
                if flip_mask.any():
                    hv_new = hv_i.clone()
                    hv_new[flip_mask] = 1.0 - hv_new[flip_mask]
                    graph.set_hv(concept, hv_new)
                    n_flips += flip_mask.sum().item()

            T = self.compute_tension(graph)
            tension_history.append(T)

            if verbose:
                print(f"  iter {iteration+1:3d}: T={T:.4f}  flips={n_flips}")

            if n_flips == 0:
                break  # converged

        return tension_history


# ═══════════════════════════════════════════════════════════════════════════════
# Sequence Encoder — Sutor (2018) §2.3
# arXiv:1806.10755
#
# An ordered sequence [z₁, z₂, …, z_m] is encoded as:
#   S = Π^(m-1)z₁ ⊕ Π^(m-2)z₂ ⊕ … ⊕ Π^0 z_m
# where Π is a fixed random permutation (implemented as circular shift).
#
# Properties:
#   - Position-sensitive: swapping two elements gives a different HV
#   - Recoverable: given S and z₁…z_{k-1}, recover z_k via Π^{-(m-k)} and lookup
#   - Compositional: sub-sequences are similar to longer sequences
# ═══════════════════════════════════════════════════════════════════════════════

class SequenceEncoder:
    """Position-sensitive sequence encoding via iterated permutations.

    Encodes [z₁, z₂, …, z_m] as:
        S = Π^(m-1)z₁ ⊕ Π^(m-2)z₂ ⊕ … ⊕ z_m

    Each position gets a unique permuted copy of its HV, so order matters.
    XOR-bundling the shifted copies gives a single HV for the whole sequence.

    Reference:
        Sutor, Summers-Stay, Aloimonos (2018)
        "A Computational Theory for Life-Long Learning of Semantics"
        arXiv:1806.10755, §2.3 "Ordered Pairs and Sequences"
    """

    def __init__(self, shift_per_position: int = 1):
        """
        Args:
            shift_per_position: Circular-shift step size per position.
                                 Using 1 means position k gets k circular shifts.
        """
        self.shift = shift_per_position

    def encode_sequence(self, hvs: List[torch.Tensor]) -> torch.Tensor:
        """Encode a list of hypervectors as a positional sequence.

        Args:
            hvs: List of (dim,) hypervectors z₁…z_m (position 0 = first)

        Returns:
            (dim,) sequence hypervector S
        """
        if not hvs:
            raise ValueError("Empty sequence")
        m = len(hvs)
        dim = hvs[0].shape[0]
        acc = torch.zeros(dim, device=hvs[0].device)
        for k, hv in enumerate(hvs):
            # Position k from end: element at index k gets shift (m-1-k)
            n_shifts = (m - 1 - k) * self.shift
            shifted = torch.roll(hv, shifts=n_shifts, dims=-1)
            acc = acc + shifted
        # Majority vote (threshold at 0.5 for binary HVs)
        return (acc >= acc.shape[0] / 2 / m * m).float() if False else hv_majority(acc)

    def decode_position(
        self,
        seq_hv: torch.Tensor,
        position: int,
        seq_length: int,
        codebook: torch.Tensor,
    ) -> Tuple[int, float]:
        """Recover the element at a given position.

        Applies the inverse permutation for that position, then finds the
        nearest HV in the codebook via Hamming similarity.

        Args:
            seq_hv: (dim,) encoded sequence HV
            position: 0-indexed position to decode
            seq_length: Total length m of the original sequence
            codebook: (n_concepts, dim) matrix of all known concept HVs

        Returns:
            (concept_index, similarity)
        """
        n_shifts = (seq_length - 1 - position) * self.shift
        # Inverse shift: roll by -n_shifts
        unshifted = torch.roll(seq_hv, shifts=-n_shifts, dims=-1)
        # Nearest neighbour in codebook
        sims = hv_batch_sim(unshifted, codebook)
        best_idx = int(sims.argmax().item())
        return best_idx, float(sims[best_idx].item())

    def encode_ngram(self, hvs: List[torch.Tensor], n: int) -> torch.Tensor:
        """Bundle all n-gram sequence HVs from a list of element HVs.

        Extracts every consecutive n-gram and bundles their sequence HVs.

        Args:
            hvs: List of (dim,) element HVs
            n: n-gram length

        Returns:
            (dim,) bundled n-gram HV
        """
        ngrams = []
        for i in range(len(hvs) - n + 1):
            ngrams.append(self.encode_sequence(hvs[i:i + n]))
        if not ngrams:
            return hvs[0] if hvs else torch.zeros_like(hvs[0])
        return hv_majority(hv_bundle(torch.stack(ngrams)))


# ═══════════════════════════════════════════════════════════════════════════════
# Record Encoder — Sutor (2018) §2.2
# arXiv:1806.10755
#
# A record R with fields {r₁:v₁, r₂:v₂, …} is encoded as:
#   R = +c({ r₁⊕v₁,  r₂⊕v₂,  … })   (consensus sum of bound pairs)
#
# Decode field rᵢ value: R ⊕ rᵢ → lookup nearest in value codebook.
# Encode field rᵢ with value vᵢ: bind(field_hv, value_hv).
#
# This is the multi-modal fusion primitive: each modality is a field,
# its encoded output is the value. The record bundles all modalities.
# ═══════════════════════════════════════════════════════════════════════════════

class RecordEncoder:
    """Encode/decode structured records as single hypervectors.

    A record {field₁: value₁, …, fieldₙ: valueₙ} maps to:
        R = majority_vote({ XOR(field_hv_i, value_hv_i) })

    Fields and values are both hypervectors — field names are random HVs,
    values are the output of any encoder (sensor, SNN, LLM, etc.).

    Decoding: given R and field_hv_i, recover value_hv_i as R ⊕ field_hv_i
    then find nearest in the value codebook.

    This is the core multi-modal fusion operation: any number of modalities
    can be fused into one HV by adding their (field, value) binding pairs.

    Reference:
        Sutor, Summers-Stay, Aloimonos (2018)
        "A Computational Theory for Life-Long Learning of Semantics"
        arXiv:1806.10755, §2.2 "Records"
    """

    def __init__(self, dim: int = 10000, seed: Optional[int] = None):
        self.dim = dim
        self._seed = seed
        self._counter = 0
        # field_name → random basis HV
        self.field_hvs: Dict[str, torch.Tensor] = {}

    def _get_field_hv(self, field_name: str) -> torch.Tensor:
        if field_name not in self.field_hvs:
            seed = (self._seed + self._counter) if self._seed is not None else None
            hv = gen_hvs(1, self.dim, seed=seed).squeeze(0)
            self.field_hvs[field_name] = hv
            self._counter += 1
        return self.field_hvs[field_name]

    def encode_record(self, fields: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode a record of field→value HV pairs into one HV.

        R = majority_vote({ XOR(field_hv, value_hv) for each field })

        Args:
            fields: dict mapping field_name → (dim,) value hypervector

        Returns:
            (dim,) record hypervector R
        """
        bindings = []
        for fname, val_hv in fields.items():
            f_hv = self._get_field_hv(fname)
            bindings.append(hv_xor(f_hv, val_hv))
        if len(bindings) == 1:
            return bindings[0]
        return hv_majority(hv_bundle(torch.stack(bindings)))

    def decode_field(
        self,
        record_hv: torch.Tensor,
        field_name: str,
        value_codebook: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[int]]:
        """Decode the value for a given field.

        recovered = R ⊕ field_hv  (XOR is its own inverse)
        If value_codebook is given, also returns the nearest index.

        Args:
            record_hv: (dim,) record hypervector
            field_name: Name of the field to decode
            value_codebook: Optional (n, dim) codebook for nearest-neighbor lookup

        Returns:
            (recovered_value_hv, nearest_index_or_None)
        """
        f_hv = self._get_field_hv(field_name)
        recovered = hv_xor(record_hv, f_hv)
        if value_codebook is not None:
            sims = hv_batch_sim(recovered, value_codebook)
            return recovered, int(sims.argmax().item())
        return recovered, None

    def add_field(
        self,
        record_hv: torch.Tensor,
        field_name: str,
        value_hv: torch.Tensor,
    ) -> torch.Tensor:
        """Add a new field to an existing record (online update).

        Returns a new record HV with the additional field bound in.

        Args:
            record_hv: (dim,) existing record HV
            field_name: New field name
            value_hv: (dim,) value hypervector

        Returns:
            (dim,) updated record hypervector
        """
        f_hv = self._get_field_hv(field_name)
        new_binding = hv_xor(f_hv, value_hv)
        return hv_majority(record_hv + new_binding)


# ═══════════════════════════════════════════════════════════════════════════════
# Life-Long Semantic Learner — Sutor (2018) "A Computational Theory for
# Life-Long Learning of Semantics"
# ═══════════════════════════════════════════════════════════════════════════════

class LifeLongSemanticLearner(nn.Module):
    """
    Life-long learning of semantics using VSA with tension-based optimization.

    Based on Sutor, Summers-Stay, Aloimonos (2018) "A Computational Theory for
    Life-Long Learning of Semantics" arXiv:1806.10755:

    - Semantic vectors evolve over time without catastrophic forgetting
    - New knowledge is bound, not overwritten
    - Co-occurrence graph is built from streaming observations
    - Tension minimization refines HVs so related concepts are more similar

    The full pipeline (as described in the paper):
    1. observe(x, context) — encode x, bundle into memory
    2. observe_sequence(tokens) — add co-occurrences to knowledge graph
    3. optimize_semantics() — run tension minimization on the knowledge graph
    4. query(x) — similarity of x to current semantic memory

    Step 3 is what the original code was missing: without tension minimization
    the vectors are random projections, not semantic representations.

    All operations are pure VSA. No backpropagation, no gradient descent.
    """

    def __init__(
        self,
        input_dim: int,
        dim: int = 10000,
        seed: Optional[int] = None,
        cooccurrence_window: int = 2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.dim = dim
        self.cooccurrence_window = cooccurrence_window

        # Semantic encoder
        self.encoder = SemanticVectorEncoder(
            input_dim=input_dim,
            output_dim=dim,
            seed=seed,
        )

        # Semantic memory: accumulated hypervector of all observations
        self.register_buffer("semantic_memory", torch.zeros(dim))

        # Context-specific memories: {context_name: accumulated_hypervector}
        self.context_memories: Dict[str, torch.Tensor] = {}

        # Observation count
        self.register_buffer("observation_count", torch.tensor(0))

        # Context hypervectors
        self.context_hvs: Dict[str, torch.Tensor] = {}

        # Next context seed
        self.next_seed = seed if seed is not None else 42

        # Knowledge graph for tension-based semantic optimization (Sutor 2018 §3)
        self.knowledge_graph = KnowledgeGraph(dim=dim, seed=seed)
        self._tension_optimizer = TensionOptimizer()

        # Sequence encoder for ordered observations (Sutor 2018 §2.3)
        self.seq_encoder = SequenceEncoder()

        # Record encoder for multi-modal fusion (Sutor 2018 §2.2)
        self.record_enc = RecordEncoder(dim=dim, seed=(seed + 1) if seed is not None else None)

        # Track how many sequences have been observed
        self._seq_count: int = 0
    
    def observe(self, x: torch.Tensor, context: Optional[str] = None):
        """Observe a new data point and update semantic memory.
        
        The observation is encoded as a hypervector and bundled with
        the existing semantic memory. Old memories are preserved because
        bundling is superposition, not overwriting.
        
        Args:
            x: (input_dim,) observation
            context: Optional context name (e.g., "task_A", "session_1")
        """
        hv = self.encoder.encode(x)
        
        # If context is provided, bind observation with context
        if context is not None:
            if context not in self.context_hvs:
                ctx_hv = gen_hvs(1, self.dim, seed=self.next_seed).squeeze(0)
                self.context_hvs[context] = ctx_hv
                self.next_seed += 1
            
            ctx_hv = self.context_hvs[context]
            bound_hv = hv_xor(hv, ctx_hv)
            
            # Update context-specific memory
            if context not in self.context_memories:
                self.context_memories[context] = bound_hv.clone()
            else:
                self.context_memories[context] = self.context_memories[context] + bound_hv
        else:
            # Update global semantic memory
            self.semantic_memory = self.semantic_memory + hv
        
        self.observation_count += 1
    
    def finalize(self):
        """Finalize all memories via majority vote."""
        self.semantic_memory = hv_majority(self.semantic_memory)
        for ctx in self.context_memories:
            self.context_memories[ctx] = hv_majority(self.context_memories[ctx])
    
    def query(self, x: torch.Tensor, context: Optional[str] = None) -> float:
        """Query the semantic memory.
        
        Returns the similarity between the query and the accumulated memory.
        Higher values indicate the query is consistent with past observations.
        
        Args:
            x: (input_dim,) query observation
            context: Optional context name
        
        Returns:
            Similarity score (0-1)
        """
        hv = self.encoder.encode(x)
        
        if context is not None and context in self.context_memories:
            ctx_hv = self.context_hvs.get(context)
            if ctx_hv is not None:
                bound_hv = hv_xor(hv, ctx_hv)
                return hv_hamming_sim(bound_hv, self.context_memories[context]).item()
        
        return hv_hamming_sim(hv, self.semantic_memory).item()
    
    def similarity_between(self, x_a: torch.Tensor, x_b: torch.Tensor) -> float:
        """Compute semantic similarity between two observations.
        
        Args:
            x_a: (input_dim,) first observation
            x_b: (input_dim,) second observation
        
        Returns:
            Semantic similarity (0-1)
        """
        hv_a = self.encoder.encode(x_a)
        hv_b = self.encoder.encode(x_b)
        return hv_hamming_sim(hv_a, hv_b).item()
    
    def get_memory(self, context: Optional[str] = None) -> torch.Tensor:
        """Get the accumulated semantic memory."""
        if context is not None and context in self.context_memories:
            return self.context_memories[context]
        return self.semantic_memory

    # ── New: tension-based semantic methods (Sutor 2018 §3) ───────────────────

    def observe_sequence(self, tokens: List[str]) -> torch.Tensor:
        """Observe a token sequence and update the knowledge graph.

        Adds co-occurrence edges between tokens that appear within
        ``cooccurrence_window`` positions of each other.  Also encodes
        the sequence as a positional HV and bundles it into semantic memory.

        Args:
            tokens: Ordered list of concept/token names

        Returns:
            (dim,) sequence hypervector for this observation
        """
        # Update knowledge graph with sliding-window co-occurrences
        self.knowledge_graph.build_from_sequences([tokens], self.cooccurrence_window)

        # Encode sequence positionally and bundle into semantic memory
        hvs = [self.knowledge_graph.add_concept(t) for t in tokens]
        if len(hvs) > 1:
            seq_hv = self.seq_encoder.encode_sequence(hvs)
        else:
            seq_hv = hvs[0]
        self.semantic_memory = self.semantic_memory + seq_hv
        self.observation_count += 1
        self._seq_count += 1
        return seq_hv

    def optimize_semantics(
        self,
        n_iterations: int = 20,
        verbose: bool = False,
    ) -> List[float]:
        """Refine concept HVs via tension minimization.

        This is the core missing step from the original implementation.
        Runs ``TensionOptimizer.minimize_tension`` on the accumulated
        knowledge graph so that co-occurring concepts end up with similar
        HVs and rarely-co-occurring concepts diverge.

        Should be called periodically after a batch of ``observe_sequence``
        calls — analogous to a sleep/consolidation phase.

        Args:
            n_iterations: Maximum greedy-descent sweeps
            verbose: Print tension per iteration

        Returns:
            Tension history (list of floats, one per iteration)
        """
        if self.knowledge_graph.n_edges == 0:
            return []
        history = self._tension_optimizer.minimize_tension(
            self.knowledge_graph, n_iterations=n_iterations, verbose=verbose
        )
        # Rebuild semantic memory from the refined concept HVs
        if self.knowledge_graph.n_vertices > 0:
            all_hvs = list(self.knowledge_graph.vertices.values())
            self.semantic_memory = hv_majority(
                torch.stack(all_hvs).sum(dim=0)
            )
        return history

    def encode_multimodal(self, fields: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Fuse multiple modalities into one HV using record binding.

        Each modality is a named field with a value HV.  Returns a single
        record HV that encodes all modalities and can be stored in semantic
        memory or used as a query.

        Example::

            learner.encode_multimodal({
                "vision": vision_encoder(img),
                "audio":  audio_encoder(wav),
                "sensor": sensor_encoder(imu),
            })

        Args:
            fields: dict mapping modality_name → (dim,) value hypervector

        Returns:
            (dim,) fused record hypervector
        """
        return self.record_enc.encode_record(fields)

    def semantic_similarity(self, concept_a: str, concept_b: str) -> float:
        """Return Hamming similarity between two concept HVs in the graph.

        Before ``optimize_semantics()`` this reflects the random initialization.
        After optimization it reflects actual co-occurrence structure.

        Args:
            concept_a: First concept name
            concept_b: Second concept name

        Returns:
            Hamming similarity in [0, 1], or 0.0 if either concept is unknown
        """
        hv_a = self.knowledge_graph.get_hv(concept_a)
        hv_b = self.knowledge_graph.get_hv(concept_b)
        if hv_a is None or hv_b is None:
            return 0.0
        return hv_hamming_sim(hv_a, hv_b).item()

    def get_concept_hv(self, concept: str) -> Optional[torch.Tensor]:
        """Return the (possibly optimized) HV for a concept."""
        return self.knowledge_graph.get_hv(concept)

    def semantic_drift_score(
        self,
        concept:     str,
        recent_hvs:  List[torch.Tensor],
        n_recent:    int = 10,
    ) -> float:
        """
        Measure how much a concept's meaning has drifted recently.

        Compares the stored prototype HV to the mean of the most recent
        observations.  High drift score → the concept is undergoing semantic
        shift (e.g., a word's usage changed, a sensor's calibration drifted).

        Args:
            concept:     Concept name to check
            recent_hvs:  Recent observation HVs labelled as this concept
            n_recent:    How many recent observations to use

        Returns:
            Drift score ∈ [0, 0.5]; > 0.2 suggests meaningful drift.
        """
        proto = self.get_concept_hv(concept)
        if proto is None or not recent_hvs:
            return 0.0
        subset  = recent_hvs[-n_recent:]
        mean_hv = torch.stack([h.float() for h in subset]).mean(0)
        mean_bin = (mean_hv > 0.5).float()
        drift    = float((proto.float() != mean_bin).float().mean().item())
        return drift

    def stable_concepts(self, threshold: float = 0.15) -> List[str]:
        """
        Return concept names whose stored HVs are likely stable (low drift).

        A concept is stable if its HV has density close to 0.5 (balanced).
        Very sparse or very dense prototypes indicate degenerate representations
        that may have been corrupted by many updates.

        Args:
            threshold: Max allowed deviation from 0.5 density.

        Returns:
            List of stable concept names.
        """
        stable = []
        for name in self.knowledge_graph._concept_hvs:
            hv      = self.knowledge_graph.get_hv(name)
            if hv is None:
                continue
            density = float(hv.float().mean().item())
            if abs(density - 0.5) <= threshold:
                stable.append(name)
        return stable


# ═══════════════════════════════════════════════════════════════════════════════
# Visual Place Recognition Pipeline — Full VPR System
# ═══════════════════════════════════════════════════════════════════════════════

class VisualPlaceRecognition(nn.Module):
    """
    Complete Visual Place Recognition pipeline using Vector Semantic Representations.
    
    Based on Sutor et al. (2025) "Vector Semantic Representations as Descriptors 
    for Visual Place Recognition":
    
    Pipeline:
        Visual Features → Semantic Hypervector → Place Descriptor
        Place Descriptor ⊕ Context → Bound Representation
        Query → Hamming Similarity → Place Recognition
    
    Key properties:
    - **No neural networks**: pure VSA operations
    - **No backpropagation**: accumulation-only learning
    - **No catastrophic forgetting**: new places don't overwrite old ones
    - **Context-aware**: same place in different contexts → different descriptors
    - **Route recognition**: bundle of place descriptors along a path
    - **All operations are pure bitwise**: XOR + popcount only
    
    Args:
        feature_dim: Dimension of visual features
        dim: Hypervector dimension
        seed: Random seed
    """
    
    def __init__(
        self,
        feature_dim: int,
        dim: int = 10000,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.dim = dim
        
        # Place descriptor memory
        self.place_memory = PlaceDescriptor(
            feature_dim=feature_dim,
            dim=dim,
            seed=seed,
        )
        
        # Semantic set operations for place relationships
        self.semantic_sets = SemanticSet(
            dim=dim,
            seed=seed,
        )
        
        # Life-long semantic learner
        self.lifelong_learner = LifeLongSemanticLearner(
            input_dim=feature_dim,
            dim=dim,
            seed=seed,
        )
        
        # Track operations for energy estimation
        self.total_queries = 0
    
    def learn_place(self, place_id: int, features: torch.Tensor, context: Optional[str] = None):
        """Learn a new place or add a view to an existing place.
        
        Args:
            place_id: Unique place identifier
            features: (feature_dim,) visual features
            context: Optional context name
        """
        self.place_memory.add_view(place_id, features)
        self.lifelong_learner.observe(features, context)
    
    def finalize(self):
        """Finalize all memories."""
        self.place_memory.finalize()
        self.lifelong_learner.finalize()
    
    def recognize(self, features: torch.Tensor, context: Optional[str] = None) -> Tuple[Optional[int], float]:
        """Recognize a place from visual features.
        
        Args:
            features: (feature_dim,) visual features
            context: Optional context name
        
        Returns:
            (place_id, confidence)
        """
        self.total_queries += 1
        
        if context is not None:
            place_id, sims, confidence = self.place_memory.query_with_context(features, context)
        else:
            place_id, sims, confidence = self.place_memory.query(features)
        
        return place_id, confidence
    
    def compare_places(self, place_a: int, place_b: int) -> float:
        """Compare two places.
        
        Args:
            place_a: First place ID
            place_b: Second place ID
        
        Returns:
            Similarity score (0-1)
        """
        return self.place_memory.place_similarity(place_a, place_b)
    
    def create_route(self, route_name: str, place_ids: List[int]) -> torch.Tensor:
        """Create a route descriptor from a sequence of places.
        
        Args:
            route_name: Route name
            place_ids: List of place IDs along the route
        
        Returns:
            (dim,) route descriptor hypervector
        """
        route_hv = self.place_memory.bundle_places(place_ids)
        self.semantic_sets.create_set(route_name, [str(pid) for pid in place_ids])
        return route_hv
    
    def estimate_energy(self) -> Dict:
        """Estimate energy per recognition query.
        
        Energy model (45nm CMOS, Horowitz ISSCC 2014):
        - XOR: 0.1 pJ per bit
        - Popcount: 0.2 pJ per operation
        """
        if self.total_queries == 0:
            return {"error": "no queries"}
        
        ENERGY_XOR_PJ = 0.1
        ENERGY_POPCOUNT_PJ = 0.2
        ENERGY_BIT_ADD_PJ = 0.05
        
        n_places = self.place_memory.size()
        
        # Encoding: for each feature, XOR with basis, add to accumulator
        avg_active = self.feature_dim / 2.0
        encode_xor = avg_active * self.dim * ENERGY_XOR_PJ
        encode_add = self.feature_dim * self.dim * ENERGY_BIT_ADD_PJ
        
        # Recognition: XOR + popcount for each place
        recognition_xor = n_places * self.dim * ENERGY_XOR_PJ
        recognition_popcount = n_places * ENERGY_POPCOUNT_PJ
        
        total_energy_pj = encode_xor + encode_add + recognition_xor + recognition_popcount
        total_energy_nj = total_energy_pj / 1000.0
        
        return {
            "architecture": f"VPR(feature_dim={self.feature_dim}, dim={self.dim}, places={n_places})",
            "total_queries": self.total_queries,
            "encode_energy_pj": float(f"{encode_xor + encode_add:.2f}"),
            "recognition_energy_pj": float(f"{recognition_xor + recognition_popcount:.2f}"),
            "total_energy_pj_per_query": float(f"{total_energy_pj:.2f}"),
            "total_energy_nj_per_query": float(f"{total_energy_nj:.4f}"),
            "learning": "accumulation (no backpropagation)",
            "inference_ops": "XOR + popcount only",
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_semantic_vector_encoder():
    """Verify semantic vector encoder."""
    print("=" * 60)
    print("Testing Semantic Vector Encoder")
    print("=" * 60)
    
    input_dim = 10
    output_dim = 1000
    
    encoder = SemanticVectorEncoder(input_dim=input_dim, output_dim=output_dim)
    
    x = torch.zeros(input_dim)
    x[0] = 1.0
    x[3] = 1.0
    x[7] = 1.0
    
    hv = encoder.encode(x)
    print(f"\n  Input dim: {input_dim}")
    print(f"  Output dim: {output_dim}")
    print(f"  HV is binary: {((hv == 0) | (hv == 1)).all().item()}")
    
    # Test similarity preservation
    x1 = torch.zeros(input_dim)
    x1[0] = 1.0; x1[3] = 1.0; x1[7] = 1.0
    
    x2 = torch.zeros(input_dim)
    x2[0] = 1.0; x2[3] = 1.0; x2[8] = 1.0
    
    hv1 = encoder.encode(x1)
    hv2 = encoder.encode(x2)
    
    sim = hv_hamming_sim(hv1, hv2)
    print(f"  Similarity between similar inputs: {sim:.4f}")
    
    print(f"\n  ✅ Semantic vector encoder test complete!")


def test_place_descriptor():
    """Verify place descriptor memory."""
    print("=" * 60)
    print("Testing Place Descriptor Memory")
    print("=" * 60)
    
    feature_dim = 10
    dim = 1000
    
    memory = PlaceDescriptor(feature_dim=feature_dim, dim=dim)
    
    # Add views for 3 places
    torch.manual_seed(42)
    for place_id in range(3):
        for _ in range(10):
            x = torch.rand(feature_dim)
            memory.add_view(place_id, x)
    
    memory.finalize()
    
    print(f"\n  Places in memory: {memory.size()}")
    
    # Test query
    x_query = torch.rand(feature_dim)
    place_id, sims, confidence = memory.query(x_query)
    print(f"  Query result: place_id={place_id}, confidence={confidence:.4f}")
    
    # Test place similarity
    sim = memory.place_similarity(0, 1)
    print(f"  Similarity between place 0 and 1: {sim:.4f}")
    
    # Test context binding
    ctx_hv = gen_hvs(1, dim).squeeze(0)
    bound = memory.bind_context(0, "summer", ctx_hv)
    print(f"  Bound descriptor is binary: {((bound == 0) | (bound == 1)).all().item()}")
    
    print(f"\n  ✅ Place descriptor test complete!")


def test_semantic_set():
    """Verify semantic set operations."""
    print("=" * 60)
    print("Testing Semantic Set Operations")
    print("=" * 60)
    
    dim = 1000
    ss = SemanticSet(dim=dim)
    
    # Create sets
    ss.create_set("fruits", ["apple", "banana", "cherry"])
    ss.create_set("red_fruits", ["apple", "cherry", "strawberry"])
    
    # Test membership
    apple_in_fruits = ss.membership("apple", "fruits")
    print(f"\n  Apple in fruits: {apple_in_fruits:.4f}")
    
    dog_in_fruits = ss.membership("dog", "fruits")
    print(f"  Dog in fruits: {dog_in_fruits:.4f}")
    
    # Test union
    ss.union("fruits", "red_fruits", "all_fruits")
    print(f"  Union created: {'all_fruits' in ss.sets}")
    
    # Test intersection
    intersection = ss.intersection("fruits", "red_fruits", threshold=0.5)
    print(f"  Intersection: {intersection}")
    
    # Test subset
    subset = ss.is_subset("fruits", "all_fruits", threshold=0.5)
    print(f"  Fruits ⊆ all_fruits: {subset}")
    
    print(f"\n  ✅ Semantic set test complete!")


def test_lifelong_learner():
    """Verify life-long semantic learning."""
    print("=" * 60)
    print("Testing Life-Long Semantic Learner")
    print("=" * 60)
    
    input_dim = 10
    dim = 1000
    
    learner = LifeLongSemanticLearner(input_dim=input_dim, dim=dim)
    
    # Observe data from two contexts
    torch.manual_seed(42)
    for i in range(50):
        x = torch.rand(input_dim)
        learner.observe(x, context="context_A" if i < 25 else "context_B")
    
    learner.finalize()
    
    print(f"\n  Observations: {learner.observation_count.item()}")
    print(f"  Contexts: {list(learner.context_memories.keys())}")
    
    # Test query
    x_test = torch.rand(input_dim)
    sim_global = learner.query(x_test)
    sim_ctx_a = learner.query(x_test, context="context_A")
    sim_ctx_b = learner.query(x_test, context="context_B")
    
    print(f"  Global similarity: {sim_global:.4f}")
    print(f"  Context A similarity: {sim_ctx_a:.4f}")
    print(f"  Context B similarity: {sim_ctx_b:.4f}")
    
    # Test similarity between observations
    x1 = torch.rand(input_dim)
    x2 = torch.rand(input_dim)
    sim = learner.similarity_between(x1, x2)
    print(f"  Similarity between random observations: {sim:.4f}")
    
    print(f"\n  ✅ Life-long semantic learner test complete!")


def test_visual_place_recognition():
    """Verify complete VPR pipeline."""
    print("=" * 60)
    print("Testing Visual Place Recognition Pipeline")
    print("=" * 60)
    
    feature_dim = 10
    dim = 1000
    
    vpr = VisualPlaceRecognition(feature_dim=feature_dim, dim=dim)
    
    # Learn 3 places
    torch.manual_seed(42)
    for place_id in range(3):
        for _ in range(10):
            x = torch.rand(feature_dim)
            vpr.learn_place(place_id, x)
    
    vpr.finalize()
    
    print(f"\n  Places learned: {vpr.place_memory.size()}")
    
    # Test recognition
    x_test = torch.rand(feature_dim)
    place_id, confidence = vpr.recognize(x_test)
    print(f"  Recognition: place_id={place_id}, confidence={confidence:.4f}")
    
    # Test place comparison
    sim = vpr.compare_places(0, 1)
    print(f"  Similarity place 0 vs 1: {sim:.4f}")
    
    # Test route creation
    route_hv = vpr.create_route("route_1", [0, 1, 2])
    print(f"  Route descriptor is binary: {((route_hv == 0) | (route_hv == 1)).all().item()}")
    
    # Test energy estimation
    energy = vpr.estimate_energy()
    print(f"  Energy per query: {energy.get('total_energy_nj_per_query', 'N/A')} nJ")
    
    print(f"\n  ✅ Visual place recognition test complete!")


def test_vector_semantic():
    """Run all vector semantic tests."""
    print("\n" + "=" * 60)
    print("Vector Semantic Representations — Test Suite")
    print("=" * 60)
    
    test_semantic_vector_encoder()
    test_place_descriptor()
    test_semantic_set()
    test_lifelong_learner()
    test_visual_place_recognition()
    
    print("\n" + "=" * 60)
    print("All vector semantic tests passed! ✅")
    print("=" * 60)


if __name__ == "__main__":
    test_vector_semantic()
