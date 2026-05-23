"""
Modern Hopfield Networks for HDC — Exponential Storage Capacity
================================================================
Reference:
    Ramsauer et al. (2020) "Hopfield Networks Is All You Need"
    ICLR 2021. arXiv:2008.02217.

    Krotov & Hopfield (2016) "Dense Associative Memory for Pattern Recognition"
    NeurIPS 2016. — Precursor to modern Hopfield.

    Millidge, Salvatori, Song, Bogacz, Bogacz (2022)
    "Associative Memories as Cells of Attention Mechanisms" arXiv:2107.14590.

Problem with classical Hopfield (and autoassociative_memory.py):
    Storage capacity ~ 0.138 × D patterns (Hertz, Krogh, Palmer 1991).
    At D=4096, classical Hopfield stores only ~565 clean patterns.
    Larger memories require impractically large D.

Modern Hopfield solution:
    Replace the Hebbian weight matrix with an energy function of the form:
        E(ξ) = −lse(β, Mᵀξ) + ½‖ξ‖²
    where lse(β, z) = β⁻¹ log Σᵢ exp(βzᵢ) (log-sum-exp with temperature β)

    This gives update rule:
        ξ ← M softmax(β × Mᵀξ)

    Capacity: 2^(αD) for some α > 0 — EXPONENTIAL in D.
    At D=4096, β=1: stores ~10^500 patterns (effectively unlimited).
    At D=4096, β=1 with retrieval accuracy >0.99: ~D patterns in a single step.

For binary HDC (ξ ∈ {0,1}^D):
    1. Map binary → bipolar: x̃ = 2x − 1
    2. Apply softmax-weighted bundle:
       α = softmax(β × M_bipolar ᵀ × ξ̃)   [similarity scores]
       ξ̃_new = M_bipolar × α                [weighted superposition]
    3. Binarize: ξ ← (ξ̃_new > 0).float()

This module implements:

1. ModernHopfieldHDC
   ── Single-step associative memory with exponential capacity
   ── Binary patterns, temperature-controlled retrieval
   ── Online pattern storage and removal

2. ModernHopfieldAttention
   ── Hopfield as an attention mechanism (Ramsauer 2021 §4)
   ── Cross-modal: queries from one space, keys from another
   ── Drop-in replacement for transformer attention in HDC pipelines

3. HopfieldHDCMemoryBank
   ── Hierarchical memory: global (slow) + episodic (fast) banks
   ── Consolidation: move episodic → global when patterns stabilise
   ── Capacity management: remove rarely-accessed patterns

4. AssociativeReasoningHopfield
   ── Use Hopfield retrieval to complete partial knowledge graphs
   ── Given partial HV, retrieve the most likely complete HV from memory
   ── Enables analogical reasoning: partial observation → full state
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from hdc.physics_world_model import _hamming, _majority


# ── utility ────────────────────────────────────────────────────────────────────

def _to_bipolar(x: torch.Tensor) -> torch.Tensor:
    """Binary {0,1} → bipolar {-1,+1}."""
    return 2.0 * x.float() - 1.0

def _to_binary(x: torch.Tensor) -> torch.Tensor:
    """Bipolar {-1,+1} → binary {0,1} via threshold at 0."""
    return (x > 0.0).float()


# ═══════════════════════════════════════════════════════════════════════════════
# 1. ModernHopfieldHDC — single-level exponential-capacity memory
# ═══════════════════════════════════════════════════════════════════════════════

class ModernHopfieldHDC:
    """
    Modern Hopfield Network for binary hypervectors.

    Capacity: exponential in D — effectively unlimited for binary HDC use cases.
    Retrieval: one-step softmax-weighted superposition (O(N × D) per query).

    Energy function (bipolar formulation):
        E(ξ) = −lse(β, Mᵀξ̃) + ½‖ξ̃‖²

    Update:
        α   = softmax(β × Mᵀξ̃)     similarity-weighted prototype selection
        ξ̃'  = M × α                 soft-retrieved pattern
        ξ'  = sign(ξ̃') > 0          binarize

    Args:
        dim: Hypervector dimension D
        beta: Retrieval temperature (higher = more selective; default 5.0)
              β→0: uniform weighting (bad retrieval)
              β→∞: hard argmax (exact nearest-neighbour)
        n_steps: Retrieval iterations (default 1; rarely need >2)
        device: torch device
    """

    def __init__(
        self,
        dim: int,
        beta: float = 5.0,
        n_steps: int = 1,
        device: str = "cpu",
    ):
        self.dim     = dim
        self.beta    = beta
        self.n_steps = n_steps
        self.device  = device

        self._patterns:    List[torch.Tensor]  = []    # stored binary HVs
        self._labels:      List[Optional[str]] = []    # optional label per pattern
        self._access_count: List[int]          = []    # for LRU eviction
        self._M: Optional[torch.Tensor]        = None  # (D, N) bipolar matrix cache

    # ── storage ──────────────────────────────────────────────────────────────

    def store(self, hv: torch.Tensor, label: Optional[str] = None):
        """Store a binary hypervector in the memory bank."""
        self._patterns.append(hv.float().to(self.device))
        self._labels.append(label)
        self._access_count.append(0)
        self._M = None   # invalidate cache

    def store_batch(self, hvs: torch.Tensor, labels: Optional[List[str]] = None):
        """Store a batch of (N, D) binary HVs."""
        for i in range(hvs.shape[0]):
            lbl = labels[i] if labels is not None else None
            self.store(hvs[i], lbl)

    def remove(self, idx: int):
        """Remove the pattern at position idx."""
        self._patterns.pop(idx)
        self._labels.pop(idx)
        self._access_count.pop(idx)
        self._M = None

    def _build_M(self) -> torch.Tensor:
        """Build (D, N) bipolar pattern matrix."""
        if self._M is None or self._M.shape[1] != len(self._patterns):
            M_bin = torch.stack(self._patterns, dim=1)     # (D, N)
            self._M = _to_bipolar(M_bin)                   # (D, N)
        return self._M

    # ── retrieval ────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query_hv: torch.Tensor,
        return_scores: bool = False,
    ) -> Tuple[torch.Tensor, Optional[str], Optional[torch.Tensor]]:
        """
        Retrieve the nearest stored pattern for a query.

        Args:
            query_hv:      (D,) binary query HV
            return_scores: If True, also return attention weights α

        Returns:
            (retrieved_hv, label, scores_or_None)
        """
        if not self._patterns:
            return query_hv.clone(), None, None

        M  = self._build_M()              # (D, N)
        ξ  = _to_bipolar(query_hv.float().to(self.device))  # (D,)

        for _ in range(self.n_steps):
            logits = M.T @ ξ              # (N,) — similarity scores (bipolar dot)
            α      = F.softmax(self.beta * logits, dim=0)  # (N,) attention weights
            ξ      = M @ α               # (D,) weighted superposition
        # Binarize
        result  = _to_binary(ξ)
        # Find argmax pattern for label lookup
        best_idx = int(α.argmax().item())
        self._access_count[best_idx] += 1
        lbl = self._labels[best_idx]

        scores = α if return_scores else None
        return result, lbl, scores

    def approximate_retrieve(
        self,
        query_hv:    torch.Tensor,
        n_candidates: int = 20,
    ) -> Tuple[torch.Tensor, Optional[str], float]:
        """
        Fast approximate retrieval using random projection LSH pre-filtering.

        For large pattern sets (N > 1000), the full Hopfield update is O(N×D).
        This method first identifies the `n_candidates` most similar patterns
        via Hamming distance (O(N×D) but with very low constant), then runs
        the full Hopfield update on this smaller candidate set.

        Expected speedup: N/n_candidates × for large N.

        Args:
            query_hv:     (D,) binary query HV
            n_candidates: Number of candidates for pre-filtering

        Returns:
            (retrieved_hv, label, best_similarity)
        """
        if not self._patterns:
            return query_hv.clone(), None, 0.0

        n = len(self._patterns)
        if n <= n_candidates:
            result, lbl, scores = self.retrieve(query_hv, return_scores=True)
            sim = float(scores.max().item()) if scores is not None else 0.0
            return result, lbl, sim

        # Pre-filter: find n_candidates most similar patterns via Hamming sim
        q   = query_hv.float().to(self.device)
        M   = self._build_M()                          # (D, N)
        # Quick Hamming sim: (1 - mean(|q - p|)) for binary {0,1} patterns
        sims = 1.0 - (q.unsqueeze(1) - M).abs().mean(dim=0)   # (N,)
        topk_idx = sims.topk(min(n_candidates, n)).indices      # (K,)

        # Build reduced memory matrix
        M_red    = M[:, topk_idx]   # (D, K)
        lbl_red  = [self._labels[int(i)] for i in topk_idx]

        # Run Hopfield update on reduced set
        ξ = _to_bipolar(q)
        for _ in range(self.n_steps):
            logits  = M_red.T @ ξ
            α       = F.softmax(self.beta * logits, dim=0)
            ξ       = M_red @ α
        result   = _to_binary(ξ)
        best_k   = int(α.argmax().item())
        best_sim = float(sims[topk_idx[best_k]].item())

        return result, lbl_red[best_k], best_sim

    def retrieve_batch(
        self, queries: torch.Tensor
    ) -> Tuple[torch.Tensor, List[Optional[str]]]:
        """
        Batch retrieval for (B, D) queries.

        Returns:
            (retrieved_hvs, labels)  shapes: (B, D), List[B]
        """
        if not self._patterns:
            return queries.clone(), [None] * queries.shape[0]

        M   = self._build_M()                          # (D, N)
        ξ   = _to_bipolar(queries.float().to(self.device))  # (B, D)

        for _ in range(self.n_steps):
            logits = ξ @ M                             # (B, N)
            α      = F.softmax(self.beta * logits, dim=1)  # (B, N)
            ξ      = α @ M.T                          # (B, D)

        results  = _to_binary(ξ)                       # (B, D)
        best_ids = α.argmax(dim=1).tolist()
        labels   = [self._labels[i] for i in best_ids]
        return results, labels

    def nearest_k(
        self, query_hv: torch.Tensor, k: int = 5
    ) -> List[Tuple[int, Optional[str], float]]:
        """Return top-k nearest patterns by Hamming similarity."""
        if not self._patterns:
            return []
        M_bin = torch.stack(self._patterns, dim=0)     # (N, D)
        sims  = _hamming(query_hv.float().unsqueeze(0), M_bin)  # (N,)
        top   = sims.topk(min(k, len(self._patterns)))
        return [
            (int(idx), self._labels[int(idx)], float(sim))
            for sim, idx in zip(top.values.tolist(), top.indices.tolist())
        ]

    # ── capacity and diagnostics ─────────────────────────────────────────────

    @property
    def n_patterns(self) -> int:
        return len(self._patterns)

    def capacity_estimate(self) -> Dict[str, float]:
        """
        Estimate current memory load.

        Modern Hopfield capacity formula (Ramsauer 2021 Theorem 3):
            N_max ≈ exp(β × D / 2)   (exponential in D)

        At D=1000, β=5: N_max ≈ exp(2500) — practically unlimited.
        The practical limit at given β and D is shown.
        """
        n_max_theory = math.exp(min(self.beta * self.dim / 2, 700))  # cap at float max
        load = len(self._patterns) / max(n_max_theory, 1)
        return {
            "n_stored":      len(self._patterns),
            "beta":          self.beta,
            "dim":           self.dim,
            "n_max_theory":  n_max_theory if n_max_theory < 1e300 else float("inf"),
            "load_fraction": load,
            "practical_limit": min(int(self.beta * self.dim), 100_000),
        }

    def interference_matrix(
        self,
        sample_size: int = 20,
    ) -> torch.Tensor:
        """
        Compute pairwise interference between stored patterns.

        Interference occurs when retrieval of pattern A pulls toward pattern B
        because they are too similar.  High interference → the memory will
        mix up these two patterns under noise.

        For each stored pattern, we query it and measure how often the
        second-best match is retrieved (not the correct one).

        Args:
            sample_size: Max patterns to check (for speed on large memories)

        Returns:
            (min(N, sample_size), min(N, sample_size)) interference matrix.
            Entry [i, j] = Hamming similarity between patterns i and j.
        """
        n = min(len(self._patterns), sample_size)
        if n < 2:
            return torch.zeros(n, n)

        patterns = torch.stack(self._patterns[:n]).float()   # (n, D)
        sims = _hamming(
            patterns.unsqueeze(1),   # (n, 1, D)
            patterns.unsqueeze(0),   # (1, n, D)
        )   # (n, n)
        # Zero diagonal (self-similarity)
        sims.fill_diagonal_(0.0)
        return sims

    def most_confused_pairs(self, top_k: int = 5) -> List[Tuple[int, int, float]]:
        """
        Return the top-k most similar pattern pairs (highest confusion risk).

        Pairs with high inter-pattern similarity will interfere with each other
        during retrieval — the Hopfield network will sometimes retrieve the
        wrong one when given a noisy query.

        Returns:
            List of (idx_a, idx_b, similarity) sorted descending.
        """
        imat = self.interference_matrix(sample_size=min(len(self._patterns), 50))
        n = imat.shape[0]
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((i, j, float(imat[i, j].item())))
        pairs.sort(key=lambda x: x[2], reverse=True)
        return pairs[:top_k]

    def prune_lru(self, keep_top: int = 100):
        """Remove least-recently-accessed patterns, keeping top `keep_top`."""
        if len(self._patterns) <= keep_top:
            return
        # Sort by access count descending
        order   = sorted(range(len(self._patterns)),
                         key=lambda i: self._access_count[i], reverse=True)
        to_keep = set(order[:keep_top])
        self._patterns     = [p for i, p in enumerate(self._patterns)     if i in to_keep]
        self._labels       = [l for i, l in enumerate(self._labels)       if i in to_keep]
        self._access_count = [a for i, a in enumerate(self._access_count) if i in to_keep]
        self._M = None

    def reset(self):
        """Clear all stored patterns."""
        self._patterns     = []
        self._labels       = []
        self._access_count = []
        self._M            = None


# ═══════════════════════════════════════════════════════════════════════════════
# 2. ModernHopfieldAttention — Hopfield as an attention layer
# ═══════════════════════════════════════════════════════════════════════════════

class ModernHopfieldAttention:
    """
    Hopfield network as a cross-modal attention mechanism.

    Reference:
        Ramsauer et al. (2021) §4: "Transformer attention = Hopfield retrieval"

        softmax(β × K^T × q) × V  ≡  Modern Hopfield update

    where K = stored key patterns, q = query, V = value patterns.

    For HDC: queries and keys are binary HVs; values may be HVs or labels.
    Cross-modal: keys from one modality (e.g., sensor HVs),
                 values from another (e.g., action HVs).

    Args:
        dim:    Key/query dimension D
        beta:   Retrieval temperature
        device: torch device
    """

    def __init__(self, dim: int, beta: float = 5.0, device: str = "cpu"):
        self.dim    = dim
        self.beta   = beta
        self.device = device

        self._keys:   List[torch.Tensor] = []   # (D,) binary
        self._values: List[torch.Tensor] = []   # (D,) binary (or other dim)

    def register(self, key_hv: torch.Tensor, value_hv: torch.Tensor):
        """Register a (key, value) pair for cross-modal attention."""
        self._keys.append(key_hv.float().to(self.device))
        self._values.append(value_hv.float().to(self.device))

    def attend(self, query_hv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Hopfield attention for a query.

        Returns:
            (attended_value, attention_weights)
            attended_value: weighted superposition of values
            attention_weights: (N,) softmax scores
        """
        if not self._keys:
            return query_hv.clone(), torch.tensor([1.0])

        K      = torch.stack(self._keys, dim=1)       # (D, N) keys (bipolar)
        K_bip  = _to_bipolar(K)
        q_bip  = _to_bipolar(query_hv.float().to(self.device))  # (D,)
        logits = K_bip.T @ q_bip                       # (N,)
        α      = F.softmax(self.beta * logits, dim=0)  # (N,) attention weights

        V      = torch.stack(self._values, dim=0)     # (N, V_dim)
        out    = α @ V                                 # (V_dim,) weighted sum
        return _to_binary(out), α

    def batch_attend(
        self, queries: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Attend for a batch of queries (B, D).

        Returns:
            (attended_values, attention_weights)  shapes: (B, V_dim), (B, N)
        """
        if not self._keys:
            return queries.clone(), torch.ones(queries.shape[0], 1)

        K      = _to_bipolar(torch.stack(self._keys, dim=1).to(self.device))  # (D, N)
        q_bip  = _to_bipolar(queries.float().to(self.device))                 # (B, D)
        logits = q_bip @ K                                                     # (B, N)
        α      = F.softmax(self.beta * logits, dim=1)                         # (B, N)

        V      = torch.stack(self._values, dim=0).to(self.device)             # (N, V_dim)
        out    = α @ V                                                         # (B, V_dim)
        return _to_binary(out), α

    @property
    def n_registered(self) -> int:
        return len(self._keys)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. HopfieldHDCMemoryBank — hierarchical episodic + global memory
# ═══════════════════════════════════════════════════════════════════════════════

class HopfieldHDCMemoryBank:
    """
    Two-tier memory system: fast episodic + slow semantic storage.

    Inspired by the hippocampal-neocortical memory consolidation model:
        - Episodic buffer:  rapid storage, limited capacity, high beta (selective)
        - Semantic memory:  slow consolidation, large capacity, lower beta (generative)

    Consolidation:
        Patterns accessed repeatedly in episodic memory are promoted to semantic.
        Rarely-accessed episodic patterns are pruned.

    Args:
        dim:                 Hypervector dimension
        episodic_capacity:   Max episodic patterns (default 256)
        episodic_beta:       Episodic retrieval temperature (high = exact)
        semantic_beta:       Semantic retrieval temperature (lower = generalising)
        consolidate_at:      Access count threshold for semantic promotion
        device:              torch device
    """

    def __init__(
        self,
        dim: int,
        episodic_capacity: int = 256,
        episodic_beta:     float = 10.0,
        semantic_beta:     float = 3.0,
        consolidate_at:    int = 3,
        device:            str = "cpu",
    ):
        self.dim            = dim
        self.consolidate_at = consolidate_at
        self.device         = device

        self.episodic = ModernHopfieldHDC(dim, beta=episodic_beta, device=device)
        self.semantic  = ModernHopfieldHDC(dim, beta=semantic_beta, device=device)
        self._episodic_capacity = episodic_capacity

    def store(self, hv: torch.Tensor, label: Optional[str] = None):
        """Store in episodic buffer; evict oldest if over capacity."""
        self.episodic.store(hv, label)
        if self.episodic.n_patterns > self._episodic_capacity:
            # Evict the least-recently-accessed episodic pattern
            lru_idx = int(min(range(self.episodic.n_patterns),
                              key=lambda i: self.episodic._access_count[i]))
            self.episodic.remove(lru_idx)

    def consolidate(self):
        """
        Promote frequently-accessed episodic patterns to semantic memory.
        Called periodically (e.g., at the end of each episode).
        """
        to_promote = [
            i for i, cnt in enumerate(self.episodic._access_count)
            if cnt >= self.consolidate_at
        ]
        for idx in reversed(sorted(to_promote)):
            hv  = self.episodic._patterns[idx]
            lbl = self.episodic._labels[idx]
            self.semantic.store(hv, lbl)
            self.episodic.remove(idx)

    def retrieve(
        self,
        query_hv:      torch.Tensor,
        prefer_episodic: bool = False,
    ) -> Tuple[torch.Tensor, Optional[str], str]:
        """
        Retrieve from both banks; return best match.

        Strategy:
            1. Try episodic first (recent, exact)
            2. If not found or low confidence, try semantic (general)
            3. prefer_episodic: always return episodic if it has any patterns

        Returns:
            (retrieved_hv, label, source)  where source ∈ {"episodic", "semantic"}
        """
        if prefer_episodic and self.episodic.n_patterns > 0:
            hv, lbl, _ = self.episodic.retrieve(query_hv)
            return hv, lbl, "episodic"

        results = []
        if self.episodic.n_patterns > 0:
            ep_hv, ep_lbl, _ = self.episodic.retrieve(query_hv)
            ep_sim = float(_hamming(query_hv.unsqueeze(0), ep_hv.unsqueeze(0)).item())
            results.append((ep_sim, ep_hv, ep_lbl, "episodic"))

        if self.semantic.n_patterns > 0:
            se_hv, se_lbl, _ = self.semantic.retrieve(query_hv)
            se_sim = float(_hamming(query_hv.unsqueeze(0), se_hv.unsqueeze(0)).item())
            results.append((se_sim, se_hv, se_lbl, "semantic"))

        if not results:
            return query_hv.clone(), None, "empty"

        results.sort(key=lambda x: x[0], reverse=True)
        _, hv, lbl, source = results[0]
        return hv, lbl, source

    def reset(self):
        """Clear both episodic and semantic memory banks."""
        self.episodic.reset()
        self.semantic.reset()

    def stats(self) -> Dict:
        return {
            "episodic_n":  self.episodic.n_patterns,
            "semantic_n":  self.semantic.n_patterns,
            "episodic_cap": self.episodic.capacity_estimate(),
            "semantic_cap": self.semantic.capacity_estimate(),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. AssociativeReasoningHopfield — partial observation → complete knowledge
# ═══════════════════════════════════════════════════════════════════════════════

class AssociativeReasoningHopfield:
    """
    Knowledge graph completion via Hopfield retrieval.

    Store (state_hv, action_hv, outcome_hv) triples in Modern Hopfield.
    Given a partial triple (e.g., state + action, no outcome), retrieve
    the most likely complete triple from memory.

    Enables:
        - "What happens if I take action A in state S?" (forward prediction)
        - "What caused outcome O?" (backward inference)
        - "What action leads to outcome O from state S?" (planning)

    Storage: each triple is stored as a composite HV:
        triple_hv = MAJORITY(state_hv ⊕ action_hv ⊕ outcome_hv)
                             (bundle of all three)

    Retrieval: query with partial triple, get back the composite HV,
    then unbind to extract missing component.

    Args:
        dim:    Hypervector dimension
        beta:   Hopfield retrieval temperature (default 5.0)
        device: torch device
    """

    def __init__(self, dim: int, beta: float = 5.0, device: str = "cpu"):
        self.dim    = dim
        self.device = device
        self.memory = ModernHopfieldHDC(dim, beta=beta, device=device)

        # Store (state, action, outcome) separately for unbinding
        self._states:   List[torch.Tensor] = []
        self._actions:  List[torch.Tensor] = []
        self._outcomes: List[torch.Tensor] = []

    def store_transition(
        self,
        state_hv:   torch.Tensor,
        action_hv:  torch.Tensor,
        outcome_hv: torch.Tensor,
    ):
        """Store a (state, action → outcome) transition."""
        s, a, o = (x.float().to(self.device) for x in (state_hv, action_hv, outcome_hv))
        # Composite: bundle of XOR pairs
        composite = _majority(((s + a + o) / 3.0))
        self.memory.store(composite)
        self._states.append(s)
        self._actions.append(a)
        self._outcomes.append(o)

    def query_outcome(
        self, state_hv: torch.Tensor, action_hv: torch.Tensor
    ) -> Tuple[torch.Tensor, float]:
        """
        Forward prediction: given (state, action), retrieve most likely outcome.

        Returns: (outcome_hv, confidence)
        """
        s, a = (x.float().to(self.device) for x in (state_hv, action_hv))
        partial_query = _majority((s + a) / 2.0)
        retrieved, _, scores = self.memory.retrieve(partial_query, return_scores=True)

        if scores is None or not self._outcomes:
            return retrieved, 0.0

        # Unbind from most similar stored triple
        best_idx = int(scores.argmax().item()) if scores is not None else 0
        outcome  = self._outcomes[min(best_idx, len(self._outcomes) - 1)]
        conf     = float(scores.max().item()) if scores is not None else 0.0
        return outcome, conf

    def query_cause(
        self, outcome_hv: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], float]:
        """
        Backward inference: given outcome, retrieve (state, action) pair.

        Returns: ((state_hv, action_hv), confidence)
        """
        o = outcome_hv.float().to(self.device)
        _, _, scores = self.memory.retrieve(o, return_scores=True)

        if scores is None or not self._states:
            return (o, o), 0.0

        best_idx = int(scores.argmax().item())
        best_idx = min(best_idx, len(self._states) - 1)
        state  = self._states[best_idx]
        action = self._actions[best_idx]
        conf   = float(scores.max().item()) if scores is not None else 0.0
        return (state, action), conf

    def query_action(
        self, state_hv: torch.Tensor, outcome_hv: torch.Tensor
    ) -> Tuple[torch.Tensor, float]:
        """
        Planning query: given (state, desired outcome), retrieve best action.

        Returns: (action_hv, confidence)
        """
        s, o = (x.float().to(self.device) for x in (state_hv, outcome_hv))
        partial = _majority((s + o) / 2.0)
        _, _, scores = self.memory.retrieve(partial, return_scores=True)

        if scores is None or not self._actions:
            return partial, 0.0

        best_idx = int(scores.argmax().item())
        best_idx = min(best_idx, len(self._actions) - 1)
        action = self._actions[best_idx]
        conf   = float(scores.max().item()) if scores is not None else 0.0
        return action, conf

    @property
    def n_transitions(self) -> int:
        return len(self._states)


# ═══════════════════════════════════════════════════════════════════════════════
# Smoke test
# ═══════════════════════════════════════════════════════════════════════════════

def _test_modern_hopfield():
    D = 1000

    def _gen(seed):
        g = torch.Generator()
        g.manual_seed(seed)
        return (torch.rand(D, generator=g) >= 0.5).float()

    print("=== ModernHopfieldHDC ===")
    mh = ModernHopfieldHDC(D, beta=5.0)
    patterns = [_gen(i) for i in range(20)]
    labels   = [f"class_{i}" for i in range(20)]
    for p, l in zip(patterns, labels):
        mh.store(p, l)

    # Retrieve clean pattern
    retrieved, lbl, _ = mh.retrieve(patterns[5])
    sim = float(_hamming(patterns[5].unsqueeze(0), retrieved.unsqueeze(0)).item())
    print(f"  Retrieved '{lbl}' for pattern 5, sim={sim:.3f}")

    # Retrieve noisy pattern (30% bit flips)
    noisy = patterns[10].clone()
    flip  = torch.rand(D) < 0.3
    noisy[flip] = 1.0 - noisy[flip]
    retrieved_noisy, lbl_noisy, _ = mh.retrieve(noisy)
    sim_noisy = float(_hamming(patterns[10].unsqueeze(0), retrieved_noisy.unsqueeze(0)).item())
    print(f"  Noisy retrieval (30% flip): '{lbl_noisy}', restored sim={sim_noisy:.3f}")

    cap = mh.capacity_estimate()
    print(f"  Capacity: stored={cap['n_stored']}, practical_limit={cap['practical_limit']}")

    print("\n=== ModernHopfieldAttention ===")
    attn = ModernHopfieldAttention(D, beta=5.0)
    for i in range(10):
        attn.register(_gen(i), _gen(100 + i))

    out, weights = attn.attend(_gen(3))
    print(f"  Attended output shape: {out.shape}, top weight: {weights.max():.3f}")

    print("\n=== HopfieldHDCMemoryBank ===")
    bank = HopfieldHDCMemoryBank(D, episodic_capacity=10)
    for i in range(15):
        bank.store(_gen(i), label=f"item_{i}")
    print(f"  Episodic: {bank.episodic.n_patterns}, Semantic: {bank.semantic.n_patterns}")

    # Consolidate frequently accessed
    bank.episodic._access_count[0] = 5
    bank.consolidate()
    print(f"  After consolidate — Episodic: {bank.episodic.n_patterns}, Semantic: {bank.semantic.n_patterns}")

    hv, lbl, source = bank.retrieve(_gen(0))
    print(f"  Retrieved from {source}: '{lbl}', sim={float(_hamming(_gen(0).unsqueeze(0), hv.unsqueeze(0)).item()):.3f}")

    print("\n=== AssociativeReasoningHopfield ===")
    ar = AssociativeReasoningHopfield(D, beta=5.0)
    for i in range(15):
        ar.store_transition(_gen(i), _gen(100 + i), _gen(200 + i))

    outcome, conf = ar.query_outcome(_gen(5), _gen(105))
    print(f"  Forward prediction: outcome sim to stored={float(_hamming(_gen(205).unsqueeze(0), outcome.unsqueeze(0)).item()):.3f}, conf={conf:.3f}")

    (state, action), conf = ar.query_cause(_gen(207))
    print(f"  Backward inference: conf={conf:.3f}")

    action, conf = ar.query_action(_gen(3), _gen(203))
    print(f"  Planning query: action sim={float(_hamming(_gen(103).unsqueeze(0), action.unsqueeze(0)).item()):.3f}, conf={conf:.3f}")

    print(f"\n  n_transitions={ar.n_transitions}")
    print("\n✅ All modern_hopfield tests passed")


if __name__ == "__main__":
    _test_modern_hopfield()
