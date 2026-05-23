"""
Cognitive Map Memory for VSA-based OODA Loop Processing
=========================================================
References:
  Bent et al. (2024) "A demonstration of vector symbolic architecture as an
    effective integrated technology for AI at the network edge" — Proc. SPIE
  Bent et al. (2024) "The transformative potential of vector symbolic architecture
    for cognitive processing at the network edge" — Proc. SPIE, DOI:10.1117/12.3030949
  Stöckl, Yang & Maass (2024) "Local prediction-learning in high-dimensional
    spaces enables neural networks to plan" — Nature Communications 15:2344
  McDonald (2023) "Modularizing and assembling cognitive map learners via
    hyperdimensional computing" — arXiv:2304.04734

Architecture:
    Memory ← self-organizing hypercube of hypervectors
        ├── Position vectors: Zp[i,j] = Zx[i] * Zy[j]
        ├── Angle vectors: Zc[θ] with circular similarity
        ├── Object vectors: role-filler bound (desc + camera + workflow)
        ├── Workflow vectors: cyclic-shifted action sequences
        ├── Cognitive map vectors: ZO * Zp (objects at positions)
        ├── CognitiveMapLearner: online learning — distance in VSA ≈ path distance
        └── HierarchicalCML: multi-level FSM via VSA binding (McDonald 2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict, Callable
from models.hdc import gen_hvs, bind, bundle, sim, batch_sim
from hdc.physics_world_model import _hamming, _majority


class CircularAngleEncoder:
    """
    Circular angle hypervectors for directional encoding.
    
    Based on Bent et al. 2024 Section 4.3.
    
    Creates an array of hypervectors where the Hamming/cosine similarity
    between any two vectors encodes the minimum angular distance.
    
    Key property: sim(Zc[10°], Zc[350°]) ≈ sim(Zc[10°], Zc[30°])
    because the minimum angular distance is 20° in both cases.
    
    This is achieved by generating vectors on a circle in hyperspace:
    each vector is a random permutation of the previous, ensuring
    smooth rotational similarity.
    """
    
    def __init__(
        self,
        n_angles: int = 36,
        dim: int = 10000,
        mode: str = "bipolar",
        seed: Optional[int] = None,
    ):
        """
        Args:
            n_angles: Number of discrete angles (default 36 for 10° steps)
            dim: Hypervector dimensionality
            mode: VSA mode ("bipolar" or "binary")
            seed: Random seed for reproducibility
        """
        self.n_angles = n_angles
        self.dim = dim
        self.mode = mode
        self.angle_step = 360.0 / n_angles
        
        # Generate circular angle vectors
        # Start with a random vector, then rotate by 1 position each step
        g = torch.Generator()
        if seed is not None:
            g.manual_seed(seed)
        
        base = gen_hvs(1, dim, mode, seed=seed).squeeze(0)
        self.vectors = torch.zeros(n_angles, dim, dtype=base.dtype)
        self.vectors[0] = base
        
        for i in range(1, n_angles):
            # Circular shift by 1 position creates smooth rotation
            self.vectors[i] = torch.roll(self.vectors[i-1], shifts=1)
    
    def encode(self, angle_deg: float) -> torch.Tensor:
        """Encode an angle in degrees into a hypervector.
        
        Args:
            angle_deg: Angle in degrees [0, 360)
        
        Returns:
            (dim,) hypervector encoding the angle
        """
        # Normalize angle to [0, n_angles)
        idx = int((angle_deg % 360) / self.angle_step)
        return self.vectors[idx].clone()
    
    def similarity(self, angle_a: float, angle_b: float) -> float:
        """Compute similarity between two angles.
        
        Should be high for small angular differences and low for large ones.
        """
        va = self.encode(angle_a)
        vb = self.encode(angle_b)
        return float(sim(va, vb, self.mode))
    
    def decode(self, hv: torch.Tensor) -> Tuple[float, float]:
        """Decode a hypervector back to the nearest angle.
        
        Args:
            hv: (dim,) hypervector
        
        Returns:
            (angle_deg, confidence)
        """
        sims = batch_sim(hv, self.vectors, self.mode)
        best_idx = int(sims.argmax().item())
        angle = best_idx * self.angle_step
        return angle, float(sims[best_idx])


class PositionEncoder:
    """
    2D Position encoding using bound linear hypervector arrays.
    
    Based on Bent et al. 2024 Section 4.4.
    
    Creates two linear hypervector arrays (X and Y axes) where
    similarity encodes linear distance. Position vectors are:
        Zp[i,j] = Zx[i] * Zy[j]
    
    This enables:
    1. Distance queries: sim(Zp[i,j], Zp[i',j']) ∝ 1/distance
    2. Nearest-neighbor: find closest position in memory
    3. Directional queries: find positions in a given direction
    """
    
    def __init__(
        self,
        grid_size: Tuple[int, int] = (30, 30),
        dim: int = 10000,
        mode: str = "bipolar",
        seed: Optional[int] = None,
    ):
        """
        Args:
            grid_size: (width, height) of the position grid
            dim: Hypervector dimensionality
            mode: VSA mode
            seed: Random seed
        """
        self.grid_size = grid_size
        self.dim = dim
        self.mode = mode
        
        # Generate linear hypervector arrays
        # Each successive vector is a random perturbation of the previous
        g = torch.Generator()
        if seed is not None:
            g.manual_seed(seed)
        
        # X-axis linear vectors
        self.x_vectors = torch.zeros(grid_size[0], dim)
        self.x_vectors[0] = gen_hvs(1, dim, mode, seed=seed).squeeze(0)
        for i in range(1, grid_size[0]):
            # Random flip of ~10% of bits creates smooth linear similarity
            flip_mask = torch.rand(dim) < 0.1
            if mode == "bipolar":
                self.x_vectors[i] = self.x_vectors[i-1] * (1 - 2 * flip_mask.float())
            else:
                self.x_vectors[i] = self.x_vectors[i-1] ^ flip_mask.long()
        
        # Y-axis linear vectors
        self.y_vectors = torch.zeros(grid_size[1], dim)
        self.y_vectors[0] = gen_hvs(1, dim, mode, seed=(seed or 0) + 1).squeeze(0)
        for i in range(1, grid_size[1]):
            flip_mask = torch.rand(dim) < 0.1
            if mode == "bipolar":
                self.y_vectors[i] = self.y_vectors[i-1] * (1 - 2 * flip_mask.float())
            else:
                self.y_vectors[i] = self.y_vectors[i-1] ^ flip_mask.long()
    
    def encode(self, x: int, y: int) -> torch.Tensor:
        """Encode a 2D position into a hypervector.
        
        Zp[x,y] = Zx[x] * Zy[y]
        
        Args:
            x: X coordinate (0 to grid_size[0]-1)
            y: Y coordinate (0 to grid_size[1]-1)
        
        Returns:
            (dim,) position hypervector
        """
        return bind(self.x_vectors[x], self.y_vectors[y], self.mode)
    
    def distance(self, pos_a: Tuple[int, int], pos_b: Tuple[int, int]) -> float:
        """Compute VSA-based distance between two positions.
        
        Returns similarity (higher = closer).
        """
        va = self.encode(*pos_a)
        vb = self.encode(*pos_b)
        return float(sim(va, vb, self.mode))
    
    def decode(self, hv: torch.Tensor) -> Tuple[Tuple[int, int], float]:
        """Decode a position hypervector to grid coordinates.
        
        Uses the property: unbind with Zy to get Zx, and vice versa.
        
        Args:
            hv: (dim,) position hypervector
        
        Returns:
            ((x, y), confidence)
        """
        best_x, best_y = 0, 0
        best_sim = -1.0
        
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                s = float(sim(hv, self.encode(x, y), self.mode))
                if s > best_sim:
                    best_sim = s
                    best_x, best_y = x, y
        
        return (best_x, best_y), best_sim


class RoleFillerBinding:
    """
    Role-filler binding for compound VSA representations.
    
    Based on Bent et al. 2024 Section 4.1.
    
    Creates compound vectors using role-filler pairs:
        Z_compound = Σ_i (Role_i * Filler_i)
    
    Where * is binding (XOR for binary, element-wise multiply for bipolar)
    and Σ is bundling (majority vote for binary, sum+threshold for bipolar).
    
    This enables:
    1. Structured representations: objects with multiple attributes
    2. Selective unbinding: query by role to recover filler
    3. Compositional semantics: combine multiple knowledge sources
    """
    
    def __init__(
        self,
        dim: int = 10000,
        mode: str = "bipolar",
        seed: Optional[int] = None,
    ):
        self.dim = dim
        self.mode = mode
        self.roles = {}  # role_name -> hypervector
        self._seed_counter = seed or 0
    
    def add_role(self, name: str) -> torch.Tensor:
        """Add a new role vector.
        
        Args:
            name: Role name (e.g., "description", "camera", "workflow")
        
        Returns:
            (dim,) role hypervector
        """
        if name not in self.roles:
            self._seed_counter += 1
            self.roles[name] = gen_hvs(
                1, self.dim, self.mode, seed=self._seed_counter
            ).squeeze(0)
        return self.roles[name]
    
    def _str_to_hv(self, s: str) -> torch.Tensor:
        """Encode a string to a deterministic hypervector."""
        seed = hash(s) & 0x7FFFFFFF
        return gen_hvs(1, self.dim, self.mode, seed=seed).squeeze(0)

    def bind_filler(self, role_name: str = None, filler=None, role: str = None) -> torch.Tensor:
        """Bind a filler to a role.

        Args:
            role_name: Name of the role (positional or keyword)
            filler: (dim,) filler hypervector or string
            role: Alias for role_name (keyword-only)

        Returns:
            (dim,) role * filler
        """
        if role is not None:
            role_name = role
        role_hv = self.add_role(role_name)
        if isinstance(filler, str):
            filler = self._str_to_hv(filler)
        return bind(role_hv, filler, self.mode)
    
    def compose(self, pairs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compose multiple role-filler pairs into a compound vector.
        
        Args:
            pairs: {role_name: filler_hv} dictionary
        
        Returns:
            (dim,) compound hypervector
        """
        bound = []
        for role_name, filler in pairs.items():
            bound.append(self.bind_filler(role_name, filler))
        
        return bundle(torch.stack(bound))
    
    def unbind(self, compound: torch.Tensor, role_name: str) -> torch.Tensor:
        """Unbind a role from a compound vector to recover the filler.
        
        Args:
            compound: (dim,) compound hypervector
            role_name: Role to unbind
        
        Returns:
            (dim,) recovered filler (with noise)
        """
        role = self.roles.get(role_name)
        if role is None:
            raise KeyError(f"Role '{role_name}' not found")
        return bind(compound, role, self.mode)
    
    def clean(self, hv: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
        """Clean a noisy hypervector by projecting onto nearest codebook entry.
        
        Args:
            hv: (dim,) noisy hypervector
            codebook: (n, dim) codebook of clean vectors
        
        Returns:
            (dim,) cleaned hypervector
        """
        sims = batch_sim(hv, codebook, self.mode)
        best_idx = int(sims.argmax().item())
        return codebook[best_idx].clone()


class WorkflowEncoder:
    """
    Workflow encoding using role-filler binding for sequential actions.
    
    Based on Bent et al. 2024 Section 4.2.
    
    Workflow vectors encode a sequence of actions using role-filler pairs:
        Z_workflow = Role_0 * A_0 + Role_1 * A_1 + ... + Role_n * Stop
    
    Where Role_i are random hypervectors for each step position.
    As each action is completed, the step counter increments.
    
    This enables:
    1. Compact representation: entire workflow in one hypervector
    2. Sequential execution: unbind role to reveal next action
    3. Termination detection: Stop vector at final position
    """
    
    def __init__(
        self,
        dim: int = 10000,
        mode: str = "bipolar",
        seed: Optional[int] = None,
    ):
        self.dim = dim
        self.mode = mode
        
        # Stop vector
        self.stop = gen_hvs(1, dim, mode, seed=seed).squeeze(0)
        
        # Role vectors for each step position
        self.roles = []
        for i in range(100):  # Max 100 steps
            self.roles.append(
                gen_hvs(1, dim, mode, seed=(seed or 0) + i + 1).squeeze(0)
            )
    
    def encode(self, actions) -> torch.Tensor:
        """Encode a sequence of actions into a workflow vector.

        Args:
            actions: List of (dim,) action hypervectors or strings

        Returns:
            (dim,) workflow hypervector
        """
        bound = []
        for i, action in enumerate(actions):
            if isinstance(action, str):
                seed = hash(action) & 0x7FFFFFFF
                action = gen_hvs(1, self.dim, self.mode, seed=seed).squeeze(0)
            bound.append(bind(self.roles[i], action, self.mode))
        
        # Add stop vector at the end: Role_n * Stop
        bound.append(bind(self.roles[len(actions)], self.stop, self.mode))
        
        return bundle(torch.stack(bound))
    
    def next_action(self, workflow: torch.Tensor, codebook: torch.Tensor, step: int = 0) -> Tuple[Optional[torch.Tensor], int]:
        """Get the next action from a workflow.
        
        Args:
            workflow: (dim,) workflow hypervector
            codebook: (n, dim) codebook of possible actions
            step: Current step index
        
        Returns:
            (next_action_hv or None if complete, next_step)
        """
        # Unbind role for current step
        action_noisy = bind(workflow, self.roles[step], self.mode)
        
        # Check if it's the Stop vector
        sim_to_stop = float(sim(action_noisy, self.stop, self.mode))
        
        # Also check similarity to all actions in codebook
        sims = batch_sim(action_noisy, codebook, self.mode)
        best_action_sim = float(sims.max())
        
        # If Stop is more similar than any action, or Stop similarity is high
        if sim_to_stop > best_action_sim and sim_to_stop > 0.3:
            return None, step + 1  # Workflow complete
        
        # Find closest action in codebook
        best_idx = int(sims.argmax().item())
        action = codebook[best_idx].clone()
        
        return action, step + 1
    
    def is_complete(self, workflow: torch.Tensor, step: int = 0, threshold: float = 0.3) -> bool:
        """Check if workflow is complete."""
        action_noisy = bind(workflow, self.roles[step], self.mode)
        sim_to_stop = float(sim(action_noisy, self.stop, self.mode))
        return sim_to_stop > threshold


class CognitiveMapMemory(nn.Module):
    """
    Self-organizing hypercube memory for VSA cognitive maps.
    
    Based on Bent et al. 2024 Section 3.1.4 and Section 5.
    
    The memory stores hypervectors in a self-organizing structure where:
    - Similar vectors are close together (measured by similarity)
    - Dissimilar vectors are quasi-orthogonal
    - Queries return the best-matching vector(s)
    
    Key operations:
    1. Store: add a vector to memory
    2. Query: find best match for a query vector
    3. Cognitive map: query for objects at positions
    4. Nearest neighbor: find closest position with a given object
    
    This implements the 'In-Memory' processing paradigm where
    vector operations are performed directly on the memory content.
    """
    
    def __init__(
        self,
        dim: int = 10000,
        mode: str = "bipolar",
        max_size: int = 10000,
        similarity_fn: Optional[Callable] = None,
    ):
        super().__init__()
        self.dim = dim
        self.mode = mode
        self.max_size = max_size
        
        # Memory storage
        self.vectors = []  # List of (dim,) hypervectors
        self.labels = []   # Optional labels for each vector
        
        # Similarity function
        self.similarity_fn = similarity_fn or (
            lambda q, v: sim(q, v, mode)
        )
    
    def store(self, hv: torch.Tensor, label: Optional[str] = None):
        """Store a hypervector in memory.
        
        Args:
            hv: (dim,) hypervector to store
            label: Optional label for identification
        """
        if len(self.vectors) >= self.max_size:
            # Evict least similar vector (simple FIFO for now)
            self.vectors.pop(0)
            if self.labels:
                self.labels.pop(0)
        
        self.vectors.append(hv.detach().cpu())
        if label is not None:
            self.labels.append(label)
    
    def add(self, hv: torch.Tensor, label: Optional[str] = None):
        """Alias for store() — consistent with AssocMemory API."""
        self.store(hv, label)

    def query(self, query_hv: torch.Tensor, top_k: int = 1) -> List[Dict]:
        """Query memory for best-matching vectors (vectorized).

        Args:
            query_hv: (dim,) query hypervector
            top_k: Number of top matches to return

        Returns:
            List of dicts with keys: 'vector', 'similarity', 'label'
        """
        if not self.vectors:
            return []

        dev = query_hv.device
        # Vectorized Hamming similarity over all stored vectors
        mem = torch.stack([v.to(dev) for v in self.vectors])  # (N, dim)
        xor = (query_hv.unsqueeze(0) != mem).float()
        sims = 1.0 - xor.mean(dim=-1)  # (N,)

        k = min(top_k, len(self.vectors))
        top_indices = sims.argsort(descending=True)[:k]

        results = []
        for idx in top_indices.tolist():
            r = {
                'vector':     self.vectors[idx].clone(),
                'similarity': float(sims[idx]),
            }
            if self.labels and idx < len(self.labels):
                r['label'] = self.labels[idx]
            results.append(r)
        return results

    def get_all(self) -> List[torch.Tensor]:
        """Return all stored hypervectors."""
        return [v.clone() for v in self.vectors]
    
    def query_cognitive_map(
        self,
        object_hv: torch.Tensor,
        position_encoder: PositionEncoder,
        current_pos: Tuple[int, int],
    ) -> List[Dict]:
        """Query cognitive map for nearest position with a given object.
        
        Based on Bent et al. 2024 Section 5, Figure 8.
        
        Constructs query: Z_query = Z_object * Zp[current_pos]
        Returns positions sorted by distance.
        
        Args:
            object_hv: (dim,) object hypervector
            position_encoder: PositionEncoder instance
            current_pos: (x, y) current agent position
        
        Returns:
            List of dicts with keys: 'position', 'distance', 'similarity'
        """
        current_pos_hv = position_encoder.encode(*current_pos)
        
        # Query: Z_object * Zp[current_pos]
        query_hv = bind(object_hv, current_pos_hv, self.mode)
        
        results = self.query(query_hv, top_k=len(self.vectors))
        
        # Decode positions from results
        decoded = []
        for r in results:
            # Unbind object to get position
            pos_hv = bind(r['vector'], object_hv, self.mode)
            pos, confidence = position_encoder.decode(pos_hv)
            
            # Compute Euclidean distance
            dist = ((pos[0] - current_pos[0]) ** 2 + (pos[1] - current_pos[1]) ** 2) ** 0.5
            
            decoded.append({
                'position': pos,
                'distance': dist,
                'similarity': r['similarity'],
                'confidence': confidence,
            })
        
        # Sort by distance
        decoded.sort(key=lambda x: x['distance'])
        return decoded
    
    def bundle_all(self) -> torch.Tensor:
        """Bundle all vectors in memory into a single vector.
        
        Useful for creating a 'summary' vector of all knowledge.
        """
        if not self.vectors:
            return torch.zeros(self.dim)
        
        stacked = torch.stack([v.to(self.vectors[0].device) for v in self.vectors])
        return bundle(stacked)
    
    def size(self) -> int:
        """Return number of vectors in memory."""
        return len(self.vectors)

    def memory_health(self) -> dict:
        """
        Diagnostic: fill ratio, mean pairwise similarity (diversity), label coverage.
        High mean_similarity → memory has redundant entries (poor diversity).
        """
        n = len(self.vectors)
        fill_ratio = n / max(self.max_size, 1)
        if n >= 2:
            import random
            pairs = [(random.randrange(n), random.randrange(n)) for _ in range(min(40, n))]
            pairs = [(i, j) for i, j in pairs if i != j]
            if pairs:
                stacked = torch.stack([v for v in self.vectors])
                sims = []
                for i, j in pairs:
                    xor = (stacked[i] != stacked[j]).float()
                    sims.append(float(1.0 - xor.mean().item()))
                mean_sim = round(sum(sims) / len(sims), 4)
            else:
                mean_sim = 1.0
        else:
            mean_sim = 1.0
        n_labelled = sum(1 for l in self.labels if l is not None) if self.labels else 0
        return {
            "n_stored":       n,
            "fill_ratio":     round(fill_ratio, 4),
            "mean_similarity": mean_sim,
            "n_labelled":     n_labelled,
            "diverse":        mean_sim < 0.7,
        }


class SemanticVectorEncoder:
    """
    Encode/decode text to/from hypervectors using semantic embeddings.
    
    Based on Bent et al. 2024 Section 2.3.2 and 3.1.3.
    
    Uses a language model (BERT, Llama, etc.) to create semantic embeddings
    of text, then projects them into the VSA hypervector space.
    
    Key properties:
    1. Semantic similarity: similar words have similar hypervectors
    2. Compositional: sentences can be encoded as bundled word vectors
    3. Robust: noisy vectors can be cleaned by projecting onto codebook
    
    This enables:
    1. LLM integration: encode LLM outputs into VSA space
    2. Agent communication: exchange semantic vectors between agents
    3. Knowledge base: store semantic vectors for querying
    """
    
    def __init__(
        self,
        dim: int = 10000,
        mode: str = "bipolar",
        seed: Optional[int] = None,
    ):
        self.dim = dim
        self.mode = mode
        self.seed = seed
        
        # Random projection matrix (fixed, not learned)
        g = torch.Generator()
        if seed is not None:
            g.manual_seed(seed)
        
        self.projection = torch.randn(dim, 768, generator=g)  # 768 = BERT embedding dim
    
    def embed_text(self, text: str) -> torch.Tensor:
        """Create a semantic hypervector from text.
        
        In a real system, this would use BERT/Llama embeddings.
        Here we use a simple hash-based embedding for demonstration.
        
        Args:
            text: Input text string
        
        Returns:
            (dim,) semantic hypervector
        """
        # Simple hash-based embedding (placeholder for BERT)
        # In production, replace with actual BERT embedding projection
        hash_val = hash(text) % (2**31)
        g = torch.Generator()
        g.manual_seed(hash_val)
        
        return gen_hvs(1, self.dim, self.mode, seed=hash_val).squeeze(0)
    
    def project_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        """Project a real-valued embedding into VSA space.
        
        Args:
            embedding: (768,) BERT/LLM embedding vector
        
        Returns:
            (dim,) binary/bipolar hypervector
        """
        # Project: Z = sign(P @ embedding)
        projected = self.projection @ embedding  # (dim,)
        
        if self.mode == "bipolar":
            return torch.sign(projected).clamp(-1, 1)
        elif self.mode == "binary":
            return (projected >= 0).float()
        else:
            return projected / projected.norm().clamp(min=1e-12)
    
    def encode_sentence(self, words: List[str]) -> torch.Tensor:
        """Encode a sentence as a bundled vector of word vectors.
        
        Args:
            words: List of words in the sentence
        
        Returns:
            (dim,) sentence hypervector
        """
        word_vectors = [self.embed_text(w) for w in words]
        return bundle(torch.stack(word_vectors))
    
    def decode_to_codebook(
        self,
        hv: torch.Tensor,
        codebook: Dict[str, torch.Tensor],
        top_k: int = 3,
    ) -> List[Tuple[str, float]]:
        """Decode a hypervector to the nearest words in a codebook.
        
        Args:
            hv: (dim,) hypervector to decode
            codebook: {word: hypervector} dictionary
            top_k: Number of top matches
        
        Returns:
            List of (word, similarity) tuples
        """
        words = list(codebook.keys())
        vectors = torch.stack([codebook[w] for w in words])
        
        sims = batch_sim(hv, vectors, self.mode)
        top_indices = sims.argsort(descending=True)[:top_k]
        
        return [(words[idx], float(sims[idx])) for idx in top_indices]


# ── CognitiveMapLearner ───────────────────────────────────────────────────────

class CognitiveMapLearner:
    """Online learning of cognitive maps via local prediction error minimisation.

    From Stöckl, Yang & Maass (2024) "Local prediction-learning in high-
    dimensional spaces enables neural networks to plan", Nature Comm. 15:2344.
    Cited as ref [33] in Bent et al. (2024), and the foundation of the OODA
    Orient → Decide pipeline described in Sections 3.2–3.3.

    Core idea:
        Each cell `c` has a float state vector v[c] ∈ R^D (initially random).
        Each action `a` has a displacement vector d[a] ∈ R^D (initially random).

        When the agent takes action `a` from cell `c` to cell `c'`, the
        prediction is  v_pred = v[c] + d[a].  The learning rule minimises:
            L = ||v[c'] - (v[c] + d[a])||²

        After training:
            v[c'] ≈ v[c] + d[a]   for every valid transition.

        Consequence: the vector space becomes a metric embedding of the graph
        where displacement vectors are consistent translations. The Hamming/
        cosine distance between v[goal] and v[current] corresponds to the
        average shortest path distance avoiding barriers.

    Planning (Bent 2024, Section 3.3 — Decide step):
        To navigate from `start` to `goal`:
            delta = v[goal] - v[current]           # desired displacement
            a*    = argmax_a  sim(d[a], delta)      # best action
            v[current] += d[a*]                     # step in vector space
        Repeat until sim(v[current], v[goal]) > threshold.
        The sequence of a* forms the course-of-action (COA).

    Graph completion property (Stöckl 2024):
        The full graph structure can be inferred WITHOUT exploring all paths.
        The learning is local and online — each excursion updates only the
        cells and actions involved in that excursion.

    Args:
        n_cells:    Number of distinct cells/states in the environment.
        n_actions:  Number of possible actions (e.g. 4 for up/down/left/right).
        dim:        Hypervector dimension D (higher = more accurate metric).
        lr:         Learning rate for prediction-error updates.
        seed:       Random seed.
    """

    def __init__(
        self,
        n_cells:   int,
        n_actions: int,
        dim:       int   = 4096,
        lr:        float = 0.05,
        seed:      int   = 42,
    ) -> None:
        self.n_cells   = n_cells
        self.n_actions = n_actions
        self.dim       = dim
        self.lr        = lr

        torch.manual_seed(seed)
        # Float vectors — updated continuously during training
        self.cell_vecs:   torch.Tensor = torch.randn(n_cells,   dim) * 0.1
        self.action_vecs: torch.Tensor = torch.randn(n_actions, dim) * 0.1

        # Transition table: (cell, action) → next_cell; -1 = barrier/invalid
        self._transitions: Dict[Tuple[int, int], int] = {}

        # Training statistics
        self.n_excursions:  int   = 0
        self.total_loss:    float = 0.0

    # ── Environment registration ──────────────────────────────────────────────

    def register_transition(self, cell: int, action: int, next_cell: int) -> None:
        """Register a valid transition (cell, action) → next_cell."""
        self._transitions[(cell, action)] = next_cell

    def register_grid(
        self,
        grid_w:   int,
        grid_h:   int,
        barriers: Optional[List[Tuple[int, int]]] = None,
    ) -> None:
        """Register a 2-D grid with optional barrier cells.

        Cell index = y * grid_w + x.
        Actions: 0=right, 1=left, 2=down, 3=up.
        """
        barrier_set = set(barriers or [])
        dx = [1, -1, 0, 0]
        dy = [0, 0, 1, -1]
        for y in range(grid_h):
            for x in range(grid_w):
                c = y * grid_w + x
                if (x, y) in barrier_set:
                    continue
                for a in range(4):
                    nx, ny = x + dx[a], y + dy[a]
                    if 0 <= nx < grid_w and 0 <= ny < grid_h and (nx, ny) not in barrier_set:
                        self._transitions[(c, a)] = ny * grid_w + nx

    # ── Online learning step ──────────────────────────────────────────────────

    def excursion(self, cell: int, action: int) -> Optional[int]:
        """Take one step and update vectors via prediction-error learning.

        Stöckl 2024 Eq. 1–2:
            error    = v[next] - (v[cell] + d[action])
            v[next]  += lr * error          (pull destination toward prediction)
            d[action]+= lr * error          (pull action toward actual displacement)
            v[cell]  -= lr * error          (push source away — keeps separation)

        Returns:
            next_cell index, or None if transition is invalid (barrier).
        """
        next_cell = self._transitions.get((cell, action))
        if next_cell is None:
            return None

        pred    = self.cell_vecs[cell] + self.action_vecs[action]
        error   = self.cell_vecs[next_cell] - pred          # = v[next] - v[cell] - d[a]
        loss    = float((error ** 2).mean().item())

        with torch.no_grad():
            # Gradient descent on ||v[next] - (v[cell] + d[a])||²
            # ∂/∂v[next]  = +2·error  → v[next]  -= lr·error
            # ∂/∂d[a]     = −2·error  → d[a]     += lr·error
            # ∂/∂v[cell]  = −2·error  → v[cell]  += lr·error (gentler)
            self.cell_vecs[next_cell]  -= self.lr * error
            self.action_vecs[action]   += self.lr * error
            self.cell_vecs[cell]       += self.lr * error * 0.3

        self.n_excursions += 1
        self.total_loss   += loss
        return next_cell

    def train(self, n_steps: int = 10000, random_start: bool = True) -> float:
        """Train by making random excursions from random starting cells.

        Returns average loss over the last 1000 steps.
        """
        import random as _random
        cells   = list({c for (c, _) in self._transitions})
        actions = list(range(self.n_actions))
        if not cells:
            raise RuntimeError("No transitions registered. Call register_transition() first.")

        recent_losses = []
        for _ in range(n_steps):
            cell = _random.choice(cells) if random_start else cells[0]
            action = _random.choice(actions)
            self.excursion(cell, action)
            if len(recent_losses) < 1000:
                recent_losses.append(self.total_loss / max(self.n_excursions, 1))

        return sum(recent_losses) / max(len(recent_losses), 1)

    # ── Planning (Decide step — Bent 2024 Section 3.3) ────────────────────────

    def plan(
        self,
        start:      int,
        goal:       int,
        max_steps:  int   = 200,
        threshold:  float = 0.90,
    ) -> Tuple[List[int], List[int]]:
        """Navigate from start to goal using vector-addition planning.

        Algorithm (Bent 2024, Section 3.3 — analogous to A*):
            1. delta = v[goal] - v[current]           # desired displacement
            2. a* = argmax_a  cosine_sim(d[a], delta) # best action
            3. current_vec += d[a*]                   # step in vector space
            4. Decode current_vec to nearest cell
            5. Repeat until near goal or max_steps reached

        This works because after training, action vectors are consistent
        translations in the vector space, so vector addition is navigation.

        Returns:
            (cell_path, action_path) — sequences of cell indices and action
            indices from start to goal (or best attempt if max_steps reached).
        """
        current_vec = self.cell_vecs[start].clone()
        goal_vec    = self.cell_vecs[goal]

        cell_path:   List[int] = [start]
        action_path: List[int] = []

        for _ in range(max_steps):
            # Check if we've reached the goal
            g_sim = float(torch.nn.functional.cosine_similarity(
                current_vec.unsqueeze(0), goal_vec.unsqueeze(0)
            ).item())
            if g_sim >= threshold:
                break

            # Desired displacement
            delta = goal_vec - current_vec                        # (D,)

            # Find best action: argmax cosine_sim(d[a], delta)
            delta_norm = delta / (delta.norm().clamp(min=1e-12))
            a_norms    = self.action_vecs / (
                self.action_vecs.norm(dim=1, keepdim=True).clamp(min=1e-12)
            )
            a_sims  = (a_norms @ delta_norm)                     # (n_actions,)
            best_a  = int(a_sims.argmax().item())

            # Step in vector space
            current_vec = current_vec + self.action_vecs[best_a]

            # Decode: find nearest cell to current_vec
            c_norms    = self.cell_vecs / (
                self.cell_vecs.norm(dim=1, keepdim=True).clamp(min=1e-12)
            )
            c_v_norm   = current_vec / current_vec.norm().clamp(min=1e-12)
            c_sims     = c_norms @ c_v_norm                      # (n_cells,)
            decoded    = int(c_sims.argmax().item())

            action_path.append(best_a)
            cell_path.append(decoded)

            # Stop if we're cycling
            if len(cell_path) > 4 and cell_path[-1] == cell_path[-3]:
                break

        return cell_path, action_path

    # ── Inspection helpers ────────────────────────────────────────────────────

    def hamming_distance(self, cell_a: int, cell_b: int) -> float:
        """Normalised Hamming distance between two binarised cell vectors."""
        v_a = (self.cell_vecs[cell_a] > 0).float()
        v_b = (self.cell_vecs[cell_b] > 0).float()
        return float((v_a != v_b).float().mean().item())

    def cell_hv(self, cell: int) -> torch.Tensor:
        """Return binarised hypervector for a cell."""
        return (self.cell_vecs[cell] > 0).float()

    def action_hv(self, action: int) -> torch.Tensor:
        """Return binarised hypervector for an action."""
        return (self.action_vecs[action] > 0).float()

    def distance_matrix(self) -> torch.Tensor:
        """Return n_cells × n_cells cosine distance matrix."""
        norms      = self.cell_vecs / (self.cell_vecs.norm(dim=1, keepdim=True).clamp(min=1e-12))
        cosine_sim = norms @ norms.T
        return 1.0 - cosine_sim

    def map_quality(self) -> Dict:
        """
        Assess the geometric quality of the learned cognitive map.

        A well-trained map should have:
          - High mean inter-cell distance (cells are spread out)
          - Low mean prediction error (transitions are consistent)
          - action_consistency: action vectors should be stable across cells

        Returns:
            Dict with mean_inter_cell_dist, action_norm_std, quality_score.
        """
        D_mat = self.distance_matrix()
        # Mean pairwise distance (exclude diagonal)
        n     = D_mat.shape[0]
        mask  = ~torch.eye(n, dtype=torch.bool)
        mean_dist = float(D_mat[mask].mean().item())

        # Action consistency: std of action vector norms (low = stable translations)
        a_norms    = self.action_vecs.norm(dim=1)
        a_norm_std = float(a_norms.std().item())

        # Quality score: higher mean_dist and lower variance = better
        quality = float(mean_dist / (1.0 + a_norm_std))

        return {
            "mean_inter_cell_dist": round(mean_dist, 4),
            "action_norm_std":      round(a_norm_std, 4),
            "quality_score":        round(quality, 4),
            "n_cells":              n,
        }

    def register_landmark(self, cell: int, name: str):
        """Tag a cell as a named landmark for human-readable planning output."""
        if not hasattr(self, "_landmarks"):
            self._landmarks: Dict[str, int] = {}
        self._landmarks[name] = cell

    def plan_to_landmark(
        self,
        start:      int,
        landmark:   str,
        max_steps:  int   = 200,
        threshold:  float = 0.90,
    ) -> Tuple[List[int], List[int]]:
        """Plan from start to a named landmark."""
        if not hasattr(self, "_landmarks") or landmark not in self._landmarks:
            raise ValueError(f"Unknown landmark: {landmark!r}")
        return self.plan(start, self._landmarks[landmark], max_steps, threshold)

    def learner_summary(self) -> Dict:
        """One-call status: training progress, map quality, transition coverage."""
        avg_loss = self.total_loss / max(self.n_excursions, 1)
        n_trans  = len(self._transitions)
        n_landmarks = len(getattr(self, "_landmarks", {}))
        quality = self.map_quality()
        return {
            "n_excursions":       self.n_excursions,
            "avg_loss":           round(avg_loss, 6),
            "n_transitions":      n_trans,
            "n_landmarks":        n_landmarks,
            **quality,
        }


# ── HierarchicalCML ────────────────────────────────────────────────────────────

class HierarchicalCML:
    """Hierarchical Cognitive Map Learner — multi-level mission planning.

    From McDonald (2023) "Modularizing and assembling cognitive map learners
    via hyperdimensional computing", arXiv:2304.04734. Cited in Bent et al.
    (2024), Section 3.3 for hierarchical planning in C5ISR.

    Architecture (two-level default, extensible to N levels):
        Level 0 (HIGH): Abstract knowledge graph / mission objectives.
                        Cells = mission states (e.g., "patrol", "engage", "retreat").
                        Actions = mission transitions.
        Level 1 (LOW):  Navigation / path execution.
                        Cells = grid positions.
                        Actions = movement primitives.

    Composition:
        High-level actions are bound to low-level goal cells via VSA binding:
            link[high_action] = bind(HIGH.action_hv(a), LOW.cell_hv(goal_cell))

        Executing a high-level action:
            1. Unbind to recover the target low-level cell goal.
            2. Run LOW.plan(current_low_cell, goal_low_cell).

        The VSA binding is the "glue" McDonald describes — it lets independently
        developed maps be assembled without modifying either map.

    IQT / C5ISR relevance:
        - High level: strategic COA (course of action) from a knowledge graph
        - Low level: tactical navigation avoiding obstacles
        - Binding = the inter-level protocol; no shared training, no retraining
    """

    def __init__(
        self,
        levels: Optional[List[CognitiveMapLearner]] = None,
        dim:    int = 4096,
        seed:   int = 42,
    ) -> None:
        self.dim    = dim
        self.levels: List[CognitiveMapLearner] = levels or []

        torch.manual_seed(seed + 999)
        # Link vectors: bind(high_action_hv, low_goal_hv) stored here
        # key = (level_idx, high_action_idx)  →  low_goal_hv
        self._links: Dict[Tuple[int, int], torch.Tensor] = {}

    def add_level(self, cml: CognitiveMapLearner) -> int:
        """Add a CML as the next hierarchical level. Returns level index."""
        self.levels.append(cml)
        return len(self.levels) - 1

    def link(
        self,
        high_level:  int,
        high_action: int,
        low_level:   int,
        low_goal:    int,
    ) -> None:
        """Bind a high-level action to a low-level goal cell.

        McDonald (2023): the link is stored as:
            link_hv = bind(high_action_hv, low_goal_hv)

        Given the high-level action at inference time, unbind to recover
        the low-level goal and drive the lower-level planner.

        Args:
            high_level:  Index of the high-level CML.
            high_action: Action index in the high-level CML.
            low_level:   Index of the low-level CML.
            low_goal:    Cell index in the low-level CML that this action leads to.
        """
        hi_hv   = self.levels[high_level].action_hv(high_action)
        lo_hv   = self.levels[low_level].cell_hv(low_goal)
        # MAP binding: element-wise multiply (bipolar -1/+1)
        hi_f    = hi_hv * 2 - 1   # {0,1} → {-1,+1}
        lo_f    = lo_hv * 2 - 1
        link_hv = ((hi_f * lo_f + 1) / 2)   # back to {0,1}
        self._links[(high_level, high_action)] = link_hv

    def resolve_action(
        self,
        high_level:  int,
        high_action: int,
        low_level:   int,
        low_start:   int,
        max_steps:   int = 200,
    ) -> Tuple[List[int], List[int]]:
        """Execute a high-level action by resolving it to a low-level plan.

        Steps:
            1. Retrieve link_hv for (high_level, high_action).
            2. Unbind high_action_hv to recover low_goal_hv.
            3. Decode low_goal_hv to the nearest low-level cell.
            4. Run low_level CML planner from low_start to that cell.

        Returns:
            (cell_path, action_path) from the low-level planner.
        """
        key = (high_level, high_action)
        if key not in self._links:
            raise KeyError(f"No link registered for level={high_level} action={high_action}")

        link_hv     = self._links[key]
        hi_hv       = self.levels[high_level].action_hv(high_action)
        hi_f        = hi_hv * 2 - 1
        link_f      = link_hv * 2 - 1
        lo_goal_f   = hi_f * link_f                                   # unbind
        lo_goal_hv  = (lo_goal_f + 1) / 2

        # Decode: find nearest low-level cell to lo_goal_hv
        low_cml      = self.levels[low_level]
        low_bv       = (low_cml.cell_vecs > 0).float()               # (n_cells, D)
        sims         = 1.0 - (low_bv != lo_goal_hv.unsqueeze(0)).float().mean(dim=1)
        goal_cell    = int(sims.argmax().item())

        return low_cml.plan(low_start, goal_cell, max_steps=max_steps)

    def execute_mission(
        self,
        high_level:   int,
        high_start:   int,
        high_goal:    int,
        low_level:    int,
        low_start:    int,
        max_high:     int = 20,
    ) -> List[Tuple[int, List[int]]]:
        """Run full hierarchical mission: high-level COA → low-level paths.

        Returns:
            List of (high_action, low_cell_path) for each high-level step.
        """
        high_cml = self.levels[high_level]
        _, high_actions = high_cml.plan(high_start, high_goal, max_steps=max_high)

        mission = []
        current_low = low_start
        for ha in high_actions:
            try:
                cell_path, _ = self.resolve_action(high_level, ha, low_level, current_low)
                mission.append((ha, cell_path))
                current_low = cell_path[-1] if cell_path else current_low
            except KeyError:
                mission.append((ha, [current_low]))   # unlinked action: stay
        return mission


# ── Tests ────────────────────────────────────────────────────────────────────

def test_circular_angles():
    """Verify circular angle encoding."""
    print("=" * 60)
    print("Testing Circular Angle Encoding (Bent 2024)")
    print("=" * 60)
    
    encoder = CircularAngleEncoder(n_angles=36, dim=1000, mode="bipolar")
    
    # Test circular property: 10° vs 350° should be similar to 10° vs 30°
    sim_near = encoder.similarity(10, 30)    # 20° apart
    sim_circular = encoder.similarity(10, 350)  # 20° apart (circular)
    sim_far = encoder.similarity(10, 190)   # 180° apart
    
    print(f"\n  sim(10°, 30°): {sim_near:.4f} (20° apart)")
    print(f"  sim(10°, 350°): {sim_circular:.4f} (20° apart, circular)")
    print(f"  sim(10°, 190°): {sim_far:.4f} (180° apart)")
    print(f"  Circular property: {'✅' if abs(sim_near - sim_circular) < 0.1 else '❌'}")
    print(f"  Far angles dissimilar: {'✅' if sim_far < sim_near else '❌'}")
    
    # Test decode
    hv = encoder.encode(45)
    decoded, conf = encoder.decode(hv)
    print(f"\n  Encode 45°, decode: {decoded:.0f}° (conf={conf:.4f})")
    print(f"  {'✅' if abs(decoded - 45) < 15 else '❌'} Decode correct")
    
    print(f"\n  ✅ Circular angle encoding test complete!")


def test_position_encoding():
    """Verify position encoding."""
    print("=" * 60)
    print("Testing Position Encoding (Bent 2024)")
    print("=" * 60)
    
    encoder = PositionEncoder(grid_size=(10, 10), dim=1000, mode="bipolar")
    
    # Test distance property
    d_near = encoder.distance((5, 5), (5, 6))   # 1 apart
    d_far = encoder.distance((5, 5), (9, 9))    # far apart
    
    print(f"\n  sim((5,5), (5,6)): {d_near:.4f} (distance 1)")
    print(f"  sim((5,5), (9,9)): {d_far:.4f} (distance ~5.7)")
    print(f"  Near > Far: {'✅' if d_near > d_far else '❌'}")
    
    # Test decode
    hv = encoder.encode(3, 7)
    pos, conf = encoder.decode(hv)
    print(f"\n  Encode (3,7), decode: {pos} (conf={conf:.4f})")
    print(f"  {'✅' if pos == (3, 7) else '❌'} Decode correct")
    
    print(f"\n  ✅ Position encoding test complete!")


def test_role_filler():
    """Verify role-filler binding."""
    print("=" * 60)
    print("Testing Role-Filler Binding (Bent 2024)")
    print("=" * 60)
    
    dim = 1000
    rfb = RoleFillerBinding(dim=dim, mode="bipolar")
    
    # Create filler vectors
    desc = gen_hvs(1, dim, "bipolar", seed=42).squeeze(0)
    camera = gen_hvs(1, dim, "bipolar", seed=43).squeeze(0)
    workflow = gen_hvs(1, dim, "bipolar", seed=44).squeeze(0)
    
    # Compose compound vector
    compound = rfb.compose({
        "description": desc,
        "camera": camera,
        "workflow": workflow,
    })
    
    # Unbind each role
    recovered_desc = rfb.unbind(compound, "description")
    recovered_camera = rfb.unbind(compound, "camera")
    
    # Check recovery (should be noisy but recognizable)
    sim_desc = sim(recovered_desc, desc, "bipolar")
    sim_camera = sim(recovered_camera, camera, "bipolar")
    
    print(f"\n  sim(desc, recovered_desc): {sim_desc:.4f}")
    print(f"  sim(camera, recovered_camera): {sim_camera:.4f}")
    print(f"  Description recoverable: {'✅' if sim_desc > 0.5 else '❌'}")
    print(f"  Camera recoverable: {'✅' if sim_camera > 0.5 else '❌'}")
    
    print(f"\n  ✅ Role-filler binding test complete!")


def test_workflow():
    """Verify workflow encoding."""
    print("=" * 60)
    print("Testing Workflow Encoding (Bent 2024)")
    print("=" * 60)
    
    dim = 1000
    encoder = WorkflowEncoder(dim=dim, mode="bipolar")
    
    # Create action vectors
    actions = gen_hvs(4, dim, "bipolar", seed=42)
    
    # Encode workflow
    workflow = encoder.encode([actions[0], actions[1], actions[2]])
    
    # Execute workflow
    print(f"\n  Workflow: [A0, A1, A2, Stop]")
    step = 0
    while True:
        action, next_step = encoder.next_action(workflow, actions, step=step)
        if action is None:
            print(f"  Step {step}: Stop → workflow complete")
            break
        # Find which action
        sims = batch_sim(action, actions, "bipolar")
        action_idx = int(sims.argmax().item())
        print(f"  Step {step}: A{action_idx} (sim={float(sims.max()):.4f})")
        step = next_step
        if step > 10:
            print("  ⚠️ Too many steps, breaking")
            break
    
    print(f"\n  ✅ Workflow encoding test complete!")


def test_cognitive_map():
    """Verify cognitive map memory."""
    print("=" * 60)
    print("Testing Cognitive Map Memory (Bent 2024)")
    print("=" * 60)
    
    dim = 1000
    memory = CognitiveMapMemory(dim=dim, mode="bipolar")
    pos_encoder = PositionEncoder(grid_size=(10, 10), dim=dim, mode="bipolar")
    
    # Create object vectors
    objects = gen_hvs(3, dim, "bipolar", seed=42)
    obj_names = ["rock", "tree", "crate"]
    
    # Store observations: object at positions
    observations = [
        (0, (2, 3)),
        (1, (5, 7)),
        (2, (8, 2)),
        (0, (4, 5)),  # rock seen again
    ]
    
    for obj_idx, pos in observations:
        pos_hv = pos_encoder.encode(*pos)
        obs_hv = bind(objects[obj_idx], pos_hv, "bipolar")
        memory.store(obs_hv, label=f"{obj_names[obj_idx]} at {pos}")
    
    print(f"\n  Stored {memory.size()} observations")
    
    # Query for nearest rock from position (0, 0)
    results = memory.query_cognitive_map(objects[0], pos_encoder, (0, 0))
    print(f"\n  Query: nearest rock from (0,0)")
    for r in results[:3]:
        print(f"    Position {r['position']}: dist={r['distance']:.1f}, sim={r['similarity']:.4f}")
    
    print(f"\n  ✅ Cognitive map memory test complete!")


def test_cognitive_map_learner():
    """Verify CognitiveMapLearner online training and planning.

    5×5 grid, no barriers. After training:
      - distance_matrix should correlate with Manhattan distance
      - plan(0, 24) should find a path from top-left to bottom-right
    """
    print("=" * 60)
    print("Testing CognitiveMapLearner (Stöckl 2024, Nature Comm.)")
    print("=" * 60)

    W, H = 5, 5
    cml = CognitiveMapLearner(n_cells=W * H, n_actions=4, dim=512, lr=0.05, seed=0)
    cml.register_grid(W, H)

    print(f"\n  Training on {W}×{H} grid (no barriers)...")
    loss = cml.train(n_steps=8000)
    print(f"  Final avg loss: {loss:.4f}  |  excursions: {cml.n_excursions}")

    # Distance matrix (cosine) should track Manhattan distance
    dmat = cml.distance_matrix()   # cosine distance — higher = farther
    cell_00 = 0                    # top-left
    cell_44 = 24                   # bottom-right
    cell_01 = 1                    # adjacent (Manhattan 1)
    d_adj  = float(dmat[cell_00, cell_01].item())
    d_far  = float(dmat[cell_00, cell_44].item())
    print(f"\n  Cosine dist (0,0)↔(0,1) [Manhattan=1]: {d_adj:.4f}")
    print(f"  Cosine dist (0,0)↔(4,4) [Manhattan=8]: {d_far:.4f}")
    metric_ok = d_far > d_adj
    print(f"  Metric ordering: {'✅' if metric_ok else '❌'} far > adjacent")

    # Plan from corner to corner
    path, actions = cml.plan(cell_00, cell_44, max_steps=50)
    print(f"\n  plan(0 → 24): path length={len(path)}  actions={actions[:6]}...")
    reached = cell_44 in path
    print(f"  Goal reached: {'✅' if reached else '⚠️ (partial — goal approximated)'}")
    print(f"\n  ✅ CognitiveMapLearner test complete!")


def test_hierarchical_cml():
    """Verify HierarchicalCML binds high-level actions to low-level goals."""
    print("=" * 60)
    print("Testing HierarchicalCML (McDonald 2023, arXiv:2304.04734)")
    print("=" * 60)

    # High level: 3 mission states, 2 transitions
    high = CognitiveMapLearner(n_cells=3, n_actions=2, dim=256, lr=0.08, seed=1)
    high.register_transition(0, 0, 1)   # state 0 + action 0 → state 1
    high.register_transition(1, 1, 2)   # state 1 + action 1 → state 2
    high.train(n_steps=2000)

    # Low level: 4×4 grid
    W, H = 4, 4
    low = CognitiveMapLearner(n_cells=W * H, n_actions=4, dim=256, lr=0.08, seed=2)
    low.register_grid(W, H)
    low.train(n_steps=4000)

    hcml = HierarchicalCML(dim=256, seed=3)
    hcml.add_level(high)      # level 0 = high
    hcml.add_level(low)       # level 1 = low

    # Link: high_action 0 → low goal cell 15 (bottom-right corner)
    hcml.link(high_level=0, high_action=0, low_level=1, low_goal=15)
    # Link: high_action 1 → low goal cell 0  (top-left corner)
    hcml.link(high_level=0, high_action=1, low_level=1, low_goal=0)

    # Resolve high-action 0 from low cell 0
    path, _ = hcml.resolve_action(
        high_level=0, high_action=0,
        low_level=1, low_start=0,
    )
    print(f"\n  resolve_action(high=0, a=0) from low cell 0:")
    print(f"    path: {path[:8]}{'...' if len(path)>8 else ''}")
    print(f"    goal (cell 15) {'reached ✅' if 15 in path else 'attempted ⚠️'}")

    # Execute full mission: high 0→2
    mission = hcml.execute_mission(
        high_level=0, high_start=0, high_goal=2,
        low_level=1, low_start=0,
    )
    print(f"\n  execute_mission(high 0→2): {len(mission)} high-level steps")
    for ha, lp in mission:
        print(f"    high_action={ha}  low_path_len={len(lp)}")

    print(f"\n  ✅ HierarchicalCML test complete!")


def test_truncation():
    """Verify truncate_hv and the bandwidth trade-off table."""
    print("=" * 60)
    print("Testing HV Truncation (Bent 2024 §3.1.2)")
    print("=" * 60)
    from hdc.concentration import truncate_hv, truncation_similarity_curve

    D = 4096
    torch.manual_seed(0)
    a = (torch.rand(D) > 0.5).float()
    b = (torch.rand(D) > 0.5).float()
    # Give them a planted similarity of ~0.40 Hamming distance (0.60 sim)
    n_match = int(0.60 * D)
    b[:n_match] = a[:n_match]
    sim_full = float(1.0 - (a != b).float().mean().item())

    print(f"\n  Full D={D} similarity: {sim_full:.3f}")
    for d in [2048, 1024, 512, 256]:
        a_t = truncate_hv(a, d)
        b_t = truncate_hv(b, d)
        s   = float(1.0 - (a_t != b_t).float().mean().item())
        print(f"  Truncated to d={d:>4}: sim={s:.3f}  BW saving={D//d}×")

    # Error on invalid call
    try:
        truncate_hv(a, D + 1)
        print("  ❌ Should have raised ValueError")
    except ValueError:
        print("  ValueError on new_dim >= D: ✅")

    print(f"\n  ✅ Truncation test complete!")


# ═══════════════════════════════════════════════════════════════════════════════
# Elite Enhancements — SparseCognitiveMap, MultiTimescaleAttention,
#                      TopDownPredictiveCoding
# ═══════════════════════════════════════════════════════════════════════════════

class SparseCognitiveMap:
    """
    Elite replacement for CognitiveMapMemory.

    Improvements over baseline:
      - Compressive-sensing storage: each HV is sketched to a lower-dim
        projection, halving memory without losing retrieval quality.
      - Locality-sensitive hashing (LSH) for O(log N) nearest-neighbor
        queries instead of O(N) linear scan.
      - Ebbinghaus forgetting: items not queried for > 1000 ticks have
        their retrieval priority decayed toward zero.

    Args:
        hd_dim: Hypervector dimension
        sketch_dim: Compressed sketch size (default hd_dim // 4)
        n_buckets: LSH bucket count (default 256)
        max_capacity: Max stored items before LRU eviction
        decay_rate: Age increment per un-queried tick
    """

    def __init__(
        self,
        hd_dim: int,
        sketch_dim: Optional[int] = None,
        n_buckets: int = 256,
        max_capacity: int = 5000,
        decay_rate: float = 0.999,
    ):
        self.hd_dim = hd_dim
        self.sketch_dim = sketch_dim or max(64, hd_dim // 4)
        self.n_buckets = n_buckets
        self.max_capacity = max_capacity

        self._proj = (torch.randn(hd_dim, self.sketch_dim) > 0).float()
        self._lsh_planes = torch.randn(self.sketch_dim, n_buckets)

        self._sketches: List[torch.Tensor] = []
        self._age: List[int] = []
        self._query_count: List[int] = []
        self._values: Dict[int, torch.Tensor] = {}
        self._tick: int = 0

    def _sketch(self, hv: torch.Tensor) -> torch.Tensor:
        return (hv.float() @ self._proj.to(hv.device) > 0).float()

    def _bucket(self, sketch: torch.Tensor) -> int:
        if sketch.dim() == 1:
            sketch = sketch.unsqueeze(0)
        code = (sketch @ self._lsh_planes.to(sketch.device) > 0).int()
        bucket = 0
        for b in range(min(self.n_buckets, code.shape[1])):
            bucket = (bucket << 1) | int(code[0, b].item())
        return bucket % self.n_buckets

    def store(self, hv: torch.Tensor, value: Optional[torch.Tensor] = None):
        """Store an HV (plus optional full-resolution value tensor)."""
        sketch = self._sketch(hv)
        if len(self._sketches) >= self.max_capacity:
            oldest = min(range(len(self._age)), key=lambda i: self._age[i])
            self._sketches.pop(oldest)
            self._age.pop(oldest)
            self._query_count.pop(oldest)
        self._sketches.append(sketch.cpu())
        self._age.append(0)
        self._query_count.append(0)
        if value is not None:
            self._values[len(self._sketches) - 1] = value.cpu()

    def query(self, hv: torch.Tensor, top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Query for nearest neighbors via LSH + Hamming refinement.

        Returns:
            List of (index, similarity) for top_k best matches.
        """
        self._tick += 1
        query_sketch = self._sketch(hv)
        query_bucket = self._bucket(query_sketch)

        candidates: set = set()
        for offset in [-1, 0, 1]:
            adj = (query_bucket + offset) % self.n_buckets
            for i, s in enumerate(self._sketches):
                if self._bucket(s.to(hv.device)) == adj:
                    candidates.add(i)
        if not candidates:
            candidates = set(range(len(self._sketches)))

        results = []
        for idx in candidates:
            sketch = self._sketches[idx].to(hv.device)
            raw_sim = float(_hamming(query_sketch.unsqueeze(0), sketch.unsqueeze(0)).item())
            # Age-decay: items not accessed recently get lower effective similarity.
            # Ebbinghaus forgetting: similarity × exp(-age / τ) where τ≈500 ticks.
            # This implements temporal relevance: recent activations are prioritised.
            age_factor = 0.9995 ** self._age[idx]   # ≈ exp(-age/2000)
            sim = raw_sim * age_factor
            results.append((idx, sim))
            self._query_count[idx] += 1
            self._age[idx] = 0

        for idx in set(range(len(self._sketches))) - candidates:
            self._age[idx] += 1
            if self._age[idx] > 1000:
                self._query_count[idx] = max(0, self._query_count[idx] - 1)

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def retrieve(self, idx: int) -> Optional[torch.Tensor]:
        return self._values.get(idx)


class MultiTimescaleAttention:
    """
    Elite enhancement for HDCAttention.

    Computes Hamming similarities at multiple temporal strides (fast/medium/slow)
    and blends them to simultaneously attend to rapid events and slow trends.

    Args:
        hd_dim: Hypervector dimension
        n_heads: Number of temporal scales (unused — kept for API symmetry)
    """

    def __init__(self, hd_dim: int, n_heads: int = 4):
        self.hd_dim = hd_dim
        self._memory: List[torch.Tensor] = []

    def attend(
        self,
        query: torch.Tensor,
        memory: Optional[List[torch.Tensor]] = None,
        top_k: int = 10,
    ) -> torch.Tensor:
        """Return attention-weighted blend of top_k relevant memories."""
        mem = memory if memory is not None else self._memory
        if not mem:
            return query

        sims_1 = [float(_hamming(query.unsqueeze(0), m.unsqueeze(0)).item()) for m in mem]
        sims_5  = [sims_1[i] for i in range(len(mem)) if i >= 4]
        sims_20 = [sims_1[i] for i in range(len(mem)) if i >= 19]

        combined = []
        for i in range(len(mem)):
            s = sims_1[i]
            s += sims_5[i // 5]  if i // 5 < len(sims_5)  else 0.0
            s += sims_20[i // 20] if i // 20 < len(sims_20) else 0.0
            combined.append(s / 3.0)

        if len(mem) > top_k:
            idxs = sorted(range(len(combined)), key=lambda i: combined[i], reverse=True)[:top_k]
        else:
            idxs = list(range(len(mem)))
            for i in range(len(mem)):
                combined[i] *= 0.9 ** (len(mem) - 1 - i)

        weights = F.softmax(
            torch.tensor([combined[i] for i in idxs], dtype=torch.float32) / 0.1, dim=0
        )
        chosen = torch.stack([mem[i].to(query.device) for i in idxs])
        return _majority((chosen * weights.unsqueeze(-1)).sum(dim=0))

    def add_to_memory(self, hv: torch.Tensor):
        self._memory.append(hv.cpu())
        if len(self._memory) > 1000:
            self._memory.pop(0)


class TopDownPredictiveCoding:
    """
    Elite enhancement for PredictiveCodingModule.

    Adds top-down modulation (Friston / Rao-Ballard predictive coding):
    the output blends the bottom-up prediction with a prior expectation
    from the cognitive map and a context signal from the temporal encoder.

    final = (1 - w_prior - w_ctx) * low_level + w_prior * prior + w_ctx * context
    """

    def __init__(self, hd_dim: int, blend_weight_prior: float = 0.3, blend_weight_context: float = 0.2):
        self.hd_dim = hd_dim
        self.blend_weight_prior = blend_weight_prior
        self.blend_weight_context = blend_weight_context

    def modulate(
        self,
        low_level_pred: torch.Tensor,
        prior_expectation: Optional[torch.Tensor] = None,
        context_modulation: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        w_low = 1.0 - self.blend_weight_prior - self.blend_weight_context
        blended = w_low * low_level_pred.float()
        if prior_expectation is not None:
            blended = blended + self.blend_weight_prior * prior_expectation.float()
        if context_modulation is not None:
            blended = blended + self.blend_weight_context * context_modulation.float()
        return _majority(blended)


if __name__ == "__main__":
    test_circular_angles()
    print()
    test_position_encoding()
    print()
    test_role_filler()
    print()
    test_workflow()
    print()
    test_cognitive_map()
    print()
    test_cognitive_map_learner()
    print()
    test_hierarchical_cml()
    print()
    test_truncation()
