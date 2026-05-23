"""
hdc/attention.py
================
HDC-Native Attention — O(N·D) vs Transformer O(N²·D)
======================================================
Reference:
    Karunaratne et al. (2020) "In-Memory Hyperdimensional Computing"
    Nature Electronics 3:327–337.

    Ramsauer et al. (2021) "Hopfield Networks Is All You Need" ICLR.
    — Modern Hopfield = Transformer attention in HV space.

    Millidge et al. (2022) "Associative Memories as Cells of
    Attention Mechanisms" arXiv:2107.14590.

Transformer self-attention cost:
    - O(N²·d) per layer  (N = sequence length, d = embedding dim)
    - Memory: O(N²) attention matrix
    - At N=4096: 16M attention weights per head per layer

HDC attention cost:
    - O(N·D) per query  (D = HV dimension = fixed)
    - Memory: O(N·D) stored patterns
    - At N=4096, D=4096: 16M bits — but in binary, not float32
    - Actual memory: D/32 integers per query = 128× smaller than float

The key insight:
    Transformer softmax(QK^T/√d)V is mathematically identical to
    running a Modern Hopfield retrieval with β = 1/√d.

    For binary HDC: replace dot product with Hamming similarity.
    This gives the same "soft retrieval" behavior with XOR + popcount.

Modules:

1. HDCAttentionHead
   ── Single-head HDC attention: Q, K, V are all binary HVs
   ── Hamming similarity instead of dot product
   ── Temperature-controlled softmax over similarities

2. MultiHeadHDCAttention
   ── H independent attention heads → bundle outputs
   ── Each head uses a different random projection of the input
   ── No learned Q/K/V matrices — random projections suffice

3. HDCTransformerLayer
   ── Drop-in HDC replacement for a Transformer encoder layer
   ── Self-attention (HDC) + feed-forward (XOR + majority)
   ── Suitable for wrapping in a full HDC sequence model

4. HDCSequenceAttention
   ── Temporal sequence attention: attend to past N states
   ── Online: stream-compatible, O(D) per step
   ── Used in EliteSNNTrainingPipeline for temporal context

5. CrossModalHDCAttention
   ── Cross-modal: queries from one modality, keys/values from another
   ── Example: sensor HV queries, action HV keys/values
   ── Enables multi-modal reasoning without learned bridges
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from hdc.physics_world_model import _hamming, _majority, _xor


# ── Utilities ──────────────────────────────────────────────────────────────────

def _to_bipolar(x: torch.Tensor) -> torch.Tensor:
    return 2.0 * x.float() - 1.0

def _to_binary(x: torch.Tensor) -> torch.Tensor:
    return (x > 0.0).float()

def _gen_hv(dim: int, seed: Optional[int] = None, device: str = "cpu") -> torch.Tensor:
    g = torch.Generator(device=device)
    if seed is not None:
        g.manual_seed(seed)
    return (torch.rand(dim, generator=g, device=device) >= 0.5).float()


# ═══════════════════════════════════════════════════════════════════════════════
# 1. HDCAttentionHead — single-head binary attention
# ═══════════════════════════════════════════════════════════════════════════════

class HDCAttentionHead:
    """
    Single-head HDC attention over a sequence of binary hypervectors.

    Computes attention weights as Hamming similarity between a query HV
    and each key HV in the sequence, then returns a soft blend of values.

    Complexity:
        O(N × D) — N keys, D-dimensional HVs
        vs Transformer: O(N² × d) — but our D is fixed while N grows

    Args:
        dim:         HV dimension D
        temperature: Softmax temperature β (default 5.0)
                     β → ∞  : hard argmax (nearest-neighbour)
                     β → 0  : uniform weights (no attention)
        device:      torch device
    """

    def __init__(self, dim: int, temperature: float = 5.0, device: str = "cpu"):
        self.dim         = dim
        self.temperature = temperature
        self.device      = device

    def attend(
        self,
        query:  torch.Tensor,              # (D,) query HV
        keys:   torch.Tensor,              # (N, D) key HVs
        values: Optional[torch.Tensor] = None,  # (N, D) value HVs (defaults to keys)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        HDC attention: weight keys by Hamming similarity to query.

        Returns:
            (output_hv, attention_weights)
            output_hv: (D,) majority-vote blended value HV
            weights:   (N,) softmax attention weights
        """
        vals = values if values is not None else keys
        N    = keys.shape[0]

        if N == 0:
            return query.clone(), torch.zeros(0, device=self.device)

        # Compute Hamming similarity between query and each key
        sims = _hamming(query.unsqueeze(0), keys)                # (N,)

        # Soft selection via temperature-scaled softmax
        weights = F.softmax(self.temperature * sims, dim=0)      # (N,)

        # Weighted bundle: each value HV weighted by attention score
        # In binary HDC: weighted majority vote
        weighted = (vals.float() * weights.unsqueeze(-1)).sum(dim=0)  # (D,)
        output   = _majority(weighted / (weights.sum() + 1e-8))

        return output, weights

    def attend_batch(
        self,
        queries: torch.Tensor,             # (B, D)
        keys:    torch.Tensor,             # (N, D)
        values:  Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Batch attention for B queries over N keys.

        Returns:
            (output_hvs, attention_weights)  shapes: (B, D), (B, N)
        """
        vals = values if values is not None else keys
        B, D = queries.shape
        N    = keys.shape[0]

        if N == 0:
            return queries.clone(), torch.zeros(B, 0, device=self.device)

        # Pairwise Hamming similarities: (B, N)
        # _hamming operates on last dim; broadcast (B, 1, D) over (N, D)
        sims    = 1.0 - _xor(queries.unsqueeze(1), keys.unsqueeze(0)).mean(dim=-1)
        weights = F.softmax(self.temperature * sims, dim=1)      # (B, N)

        # Weighted sum of values
        output_f = weights @ vals.float()                         # (B, D)
        output   = _majority(output_f)
        return output, weights


# ═══════════════════════════════════════════════════════════════════════════════
# 2. MultiHeadHDCAttention — H-head HDC attention
# ═══════════════════════════════════════════════════════════════════════════════

class MultiHeadHDCAttention:
    """
    Multi-head HDC attention with random projections.

    Each head uses a different random projection of the input HV before
    computing attention.  Heads capture different aspects of the input,
    similar to how different filters in a CNN capture different features.

    The outputs of all heads are bundled (majority vote) into a single HV.

    Complexity: O(H × N × D)  —  H heads, N keys, D dimensions

    Args:
        dim:         HV dimension D
        n_heads:     Number of attention heads H (default 4)
        temperature: Softmax temperature per head
        device:      torch device
    """

    def __init__(
        self,
        dim:         int,
        n_heads:     int   = 4,
        temperature: float = 5.0,
        device:      str   = "cpu",
    ):
        self.dim         = dim
        self.n_heads     = n_heads
        self.temperature = temperature
        self.device      = device

        # Random projection masks for each head (different random seeds)
        # Each head gets a random subset of ~50% of the dimensions
        self._head_masks = [
            (torch.rand(dim, device=device) >= 0.5).float()
            for h in range(n_heads)
        ]
        self._heads = [
            HDCAttentionHead(dim, temperature, device)
            for _ in range(n_heads)
        ]

    def _project(self, hv: torch.Tensor, head: int) -> torch.Tensor:
        """Apply head-specific random projection (mask) to HV."""
        mask = self._head_masks[head]
        if hv.dim() == 1:
            return hv * mask
        return hv * mask.unsqueeze(0)

    def attend(
        self,
        query:  torch.Tensor,
        keys:   torch.Tensor,
        values: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Multi-head attention.

        Returns:
            (bundled_output, per_head_weights)
            bundled_output: (D,) majority vote of all head outputs
            per_head_weights: list of H × (N,) attention weight tensors
        """
        head_outputs  = []
        head_weights  = []

        for h in range(self.n_heads):
            q_h   = _majority(self._project(query, h))
            k_h   = _majority(self._project(keys, h)) if keys.dim() > 1 else keys
            out_h, w_h = self._heads[h].attend(q_h, k_h, values)
            head_outputs.append(out_h)
            head_weights.append(w_h)

        # Bundle all head outputs via majority vote
        bundled = _majority(torch.stack(head_outputs).float().mean(dim=0))
        return bundled, head_weights

    def attend_batch(
        self,
        queries: torch.Tensor,
        keys:    torch.Tensor,
        values:  Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Batch multi-head attention for (B, D) queries over (N, D) keys.

        Returns:
            (bundled_outputs, per_head_weights)  shape: (B, D)
        """
        head_outputs = []
        head_weights = []

        for h in range(self.n_heads):
            q_h   = _majority(self._project(queries, h))
            k_h   = _majority(self._project(keys, h))
            out_h, w_h = self._heads[h].attend_batch(q_h, k_h, values)
            head_outputs.append(out_h)
            head_weights.append(w_h)

        bundled = _majority(torch.stack(head_outputs, dim=0).float().mean(dim=0))
        return bundled, head_weights

    def head_diversity(
        self,
        query: torch.Tensor,
        keys:  torch.Tensor,
        values: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Measure output diversity across heads for a given query.
        Low mean_sim → heads are attending to different aspects (good).
        High mean_sim → redundant heads (wasted capacity).
        """
        head_outputs = []
        for h in range(self.n_heads):
            q_h = _majority(self._project(query, h))
            k_h = _majority(self._project(keys, h)) if keys.dim() > 1 else keys
            out_h, _ = self._heads[h].attend(q_h, k_h, values)
            head_outputs.append(out_h)
        sims = []
        for i in range(len(head_outputs)):
            for j in range(i + 1, len(head_outputs)):
                sims.append(float(_hamming(
                    head_outputs[i].unsqueeze(0), head_outputs[j].unsqueeze(0)
                ).item()))
        return {
            "mean_head_sim": round(sum(sims) / max(len(sims), 1), 4),
            "min_head_sim":  round(min(sims) if sims else 1.0, 4),
            "n_heads":       self.n_heads,
            "is_diverse":    (sum(sims) / max(len(sims), 1)) < 0.7,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 3. HDCTransformerLayer — drop-in Transformer encoder layer replacement
# ═══════════════════════════════════════════════════════════════════════════════

class HDCTransformerLayer:
    """
    HDC equivalent of a Transformer encoder layer.

    Architecture:
        x → MultiHeadHDCAttention(x, x, x)   [self-attention]
          → residual: bundle(x, attn_out)
          → FFN: XOR + majority                [feed-forward]
          → residual: bundle(residual, ffn_out)

    No learnable weights.  No backpropagation.  O(N·D) per sequence.

    Args:
        dim:      HV dimension D
        n_heads:  Number of attention heads
        device:   torch device
    """

    def __init__(self, dim: int, n_heads: int = 4, device: str = "cpu"):
        self.dim     = dim
        self.attn    = MultiHeadHDCAttention(dim, n_heads, device=device)
        self._ffn_hv = _gen_hv(dim, seed=42, device=device)  # fixed random FFN kernel

    def forward(
        self,
        sequence: torch.Tensor,         # (N, D) sequence of HVs
        query:    Optional[torch.Tensor] = None,  # (D,) if cross-attention
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process a sequence of HVs.

        Args:
            sequence: (N, D) input HV sequence
            query:    (D,) optional query HV. If None, uses last HV in sequence.

        Returns:
            (output_hv, attention_weights)
            output_hv: (D,) context-attended output
        """
        q = query if query is not None else sequence[-1]

        # Self-attention: query attends to all keys in sequence
        attn_out, head_ws = self.attn.attend(q, sequence, sequence)

        # Residual: bundle query with attention output
        residual = _majority((q.float() + attn_out.float()) / 2.0)

        # Feed-forward: XOR with fixed random kernel (adds information mixing)
        ffn_out = _majority((_xor(residual, self._ffn_hv).float() + residual.float()) / 2.0)

        # Combine head weights into a single (N,) summary
        if head_ws:
            combined_weights = torch.stack(head_ws).mean(dim=0)
        else:
            combined_weights = torch.zeros(sequence.shape[0])

        return ffn_out, combined_weights

    def forward_sequence(self, sequence: torch.Tensor) -> torch.Tensor:
        """
        Apply self-attention to produce (N, D) attended sequence HVs.
        Each position attends to all other positions.
        """
        outputs = []
        for i in range(sequence.shape[0]):
            q   = sequence[i]
            out, _ = self.forward(sequence, query=q)
            outputs.append(out)
        return torch.stack(outputs)

    def causal_forward_sequence(self, sequence: torch.Tensor) -> torch.Tensor:
        """
        Causal (autoregressive) self-attention: each position only attends
        to PREVIOUS positions (no future leakage).

        Equivalent to applying a causal mask in standard transformers.
        Essential for sequence generation and time-series forecasting where
        future information must not be used.

        Returns:
            (N, D) causally attended HVs.
        """
        outputs = []
        for i in range(sequence.shape[0]):
            q       = sequence[i]
            # Only attend to positions 0..i (inclusive)
            context = sequence[:i + 1]
            out, _  = self.attn.attend(q, context, context)
            # Residual + FFN (same as forward())
            res = _majority((q.float() + out.float()) / 2.0)
            ffn = _majority((_xor(res, self._ffn_hv).float() + res.float()) / 2.0)
            outputs.append(ffn)
        return torch.stack(outputs)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. HDCSequenceAttention — streaming temporal attention
# ═══════════════════════════════════════════════════════════════════════════════

class HDCSequenceAttention:
    """
    Online temporal attention over a rolling buffer of past HVs.

    Maintains a fixed-size buffer of the most recent N observations.
    At each step, computes multi-head attention of the current HV over
    the buffer — giving a context-aware summary of the recent past.

    Stream-compatible: O(D) per step after buffer fills.

    Use case in Physical AI:
        State(t) may be anomalous in isolation but normal in context.
        HDCSequenceAttention lets the system "remember" the last 50 steps
        and attend to the most relevant previous states when interpreting
        the current sensor reading.

    Args:
        dim:         HV dimension D
        buffer_size: Maximum past observations to attend over (default 50)
        n_heads:     Attention heads (default 4)
        temperature: Attention temperature
        device:      torch device
    """

    def __init__(
        self,
        dim:          int,
        buffer_size:  int   = 50,
        n_heads:      int   = 4,
        temperature:  float = 5.0,
        device:       str   = "cpu",
        recency_bias: float = 0.95,
    ):
        self.dim          = dim
        self.buffer_size  = buffer_size
        self.device       = device
        self.recency_bias = recency_bias   # decay per step back in buffer

        self._buffer: List[torch.Tensor] = []
        self._attn = MultiHeadHDCAttention(dim, n_heads, temperature, device)

    def step(self, hv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process one HV and return the context-attended output.

        Args:
            hv: (D,) current state HV

        Returns:
            (context_hv, attention_weights)
            context_hv: (D,) current HV in context of recent history
            weights:    (N,) attention over the buffer (or zeros if empty)
        """
        if not self._buffer:
            self._buffer.append(hv.to(self.device))
            return hv.clone(), torch.zeros(1, device=self.device)

        keys = torch.stack(self._buffer)   # (N, D)

        # Recency-biased keys: more recent entries are amplified.
        # bias_weights[i] = recency_bias^(N-1-i), so the most recent (i=N-1) = 1.0
        if self.recency_bias < 1.0:
            N = len(self._buffer)
            bias = torch.tensor(
                [self.recency_bias ** (N - 1 - i) for i in range(N)],
                device=self.device
            ).unsqueeze(-1)   # (N, 1)
            keys = keys * bias   # scale each key by its recency weight

        out, wt = self._attn.attend(hv, keys, keys)

        self._buffer.append(hv.to(self.device))
        if len(self._buffer) > self.buffer_size:
            self._buffer.pop(0)

        combined = torch.stack(wt).mean(dim=0) if wt else torch.zeros(len(self._buffer))
        return out, combined

    def reset(self):
        self._buffer = []

    @property
    def buffer_len(self) -> int:
        return len(self._buffer)

    def buffer_diversity(self) -> float:
        """
        Mean pairwise Hamming similarity within the buffer.
        Low → buffer contains diverse observations (temporally varied input).
        High → buffer is full of near-duplicate states (static environment).
        Samples up to 20 pairs to keep O(1).
        """
        n = len(self._buffer)
        if n < 2:
            return 1.0
        import random
        pairs = []
        indices = list(range(n))
        for _ in range(min(20, n * (n - 1) // 2)):
            i, j = random.sample(indices, 2)
            pairs.append((i, j))
        sims = [
            float(_hamming(
                self._buffer[i].unsqueeze(0), self._buffer[j].unsqueeze(0)
            ).item())
            for i, j in pairs
        ]
        return round(sum(sims) / len(sims), 4)

    def sequence_summary(self) -> Dict[str, float]:
        """Buffer fill level and diversity."""
        return {
            "buffer_len":      self.buffer_len,
            "buffer_capacity": self.buffer_size,
            "fill_ratio":      round(self.buffer_len / max(self.buffer_size, 1), 4),
            "buffer_diversity": self.buffer_diversity(),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 5. CrossModalHDCAttention — cross-modal retrieval
# ═══════════════════════════════════════════════════════════════════════════════

class CrossModalHDCAttention:
    """
    Cross-modal HDC attention for multi-sensor fusion.

    Queries come from one modality (e.g., vision HVs),
    keys/values from another (e.g., audio HVs).

    The attention mechanism finds the audio context most relevant
    to the current visual query — no explicit alignment or learned bridge.

    Applications:
        - Sensor fusion: IMU query → find relevant LIDAR context
        - Action selection: state HV query → find relevant past action HVs
        - Knowledge retrieval: observation HV → find relevant world-model HVs

    Args:
        dim:       HV dimension (shared across modalities)
        n_heads:   Attention heads
        max_keys:  Maximum stored keys per modality
        device:    torch device
    """

    def __init__(
        self,
        dim:      int,
        n_heads:  int = 4,
        max_keys: int = 200,
        device:   str = "cpu",
    ):
        self.dim      = dim
        self.max_keys = max_keys
        self._attn    = MultiHeadHDCAttention(dim, n_heads, device=device)

        self._modalities: Dict[str, Tuple[List[torch.Tensor], List[torch.Tensor]]] = {}
        # _modalities[name] = (keys, values)

    def register_modality(self, name: str):
        """Register a named modality."""
        if name not in self._modalities:
            self._modalities[name] = ([], [])

    def add_context(
        self,
        modality:  str,
        key_hv:    torch.Tensor,
        value_hv:  Optional[torch.Tensor] = None,
    ):
        """
        Add a key-value pair to a modality's context buffer.

        Args:
            modality:  Modality name
            key_hv:    (D,) key hypervector
            value_hv:  (D,) value HV (defaults to key if None)
        """
        self.register_modality(modality)
        keys, vals = self._modalities[modality]
        keys.append(key_hv.float().to(self.device if hasattr(self, 'device') else "cpu"))
        vals.append((value_hv if value_hv is not None else key_hv).float())
        if len(keys) > self.max_keys:
            keys.pop(0)
            vals.pop(0)

    def query(
        self,
        query_hv:         torch.Tensor,
        target_modality:  str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Cross-modal attention: query against a specific modality's context.

        Args:
            query_hv:        (D,) query from any modality
            target_modality: Name of the modality to attend over

        Returns:
            (attended_hv, attention_weights)
        """
        if target_modality not in self._modalities:
            return query_hv.clone(), torch.zeros(1)

        keys, vals = self._modalities[target_modality]
        if not keys:
            return query_hv.clone(), torch.zeros(1)

        K = torch.stack(keys)
        V = torch.stack(vals)
        out, wts = self._attn.attend(query_hv, K, V)
        combined = torch.stack(wts).mean(dim=0) if wts else torch.zeros(K.shape[0])
        return out, combined

    def query_all(
        self,
        query_hv: torch.Tensor,
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Query across all registered modalities.

        Returns:
            {modality_name: (attended_hv, weights)} for each modality.
        """
        return {
            name: self.query(query_hv, name)
            for name in self._modalities
        }

    def fuse(
        self,
        query_hv:    torch.Tensor,
        modalities:  Optional[List[str]] = None,
    ) -> torch.Tensor:
        """
        Fuse cross-modal attention outputs into a single HV.

        Attends to each modality and bundles the results.

        Returns:
            (D,) multi-modal context HV
        """
        targets  = modalities or list(self._modalities.keys())
        outputs  = []
        for name in targets:
            out, _ = self.query(query_hv, name)
            outputs.append(out)
        if not outputs:
            return query_hv.clone()
        return _majority(torch.stack(outputs).float().mean(dim=0))

    def compress_context(
        self,
        modality:   str,
        max_keep:   int = 50,
        threshold:  float = 0.65,
    ):
        """
        Remove redundant context entries from a modality buffer.

        Pairs of entries with Hamming similarity > threshold are merged
        (averaged), keeping only `max_keep` most diverse entries.

        This prevents the context buffer from growing with near-duplicate
        observations — which dilute attention signal without adding information.

        Args:
            modality:  Target modality to compress
            max_keep:  Maximum entries after compression
            threshold: Similarity above which two entries are considered redundant
        """
        if modality not in self._modalities:
            return
        keys, vals = self._modalities[modality]
        if len(keys) <= max_keep:
            return

        # Greedy deduplication: keep most diverse entries
        kept_keys, kept_vals = [keys[0]], [vals[0]]
        for k, v in zip(keys[1:], vals[1:]):
            if not kept_keys:
                kept_keys.append(k); kept_vals.append(v)
                continue
            recent_keys = torch.stack(kept_keys[-10:])   # compare to recent
            sims = 1.0 - (k.unsqueeze(0) != recent_keys).float().mean(dim=1)
            if float(sims.max().item()) < threshold:
                kept_keys.append(k)
                kept_vals.append(v)
            if len(kept_keys) >= max_keep:
                break

        self._modalities[modality] = (kept_keys[-max_keep:], kept_vals[-max_keep:])

    @property
    def modality_sizes(self) -> Dict[str, int]:
        return {name: len(keys) for name, (keys, _) in self._modalities.items()}

    def context_health(self) -> Dict:
        """
        Per-modality buffer fill, diversity, and compression recommendation.
        """
        report: Dict = {"n_modalities": len(self._modalities)}
        for name, (keys, _) in self._modalities.items():
            n = len(keys)
            if n < 2:
                div = 1.0
            else:
                import random
                pairs = [(random.randrange(n), random.randrange(n)) for _ in range(min(20, n))]
                pairs = [(i, j) for i, j in pairs if i != j]
                if pairs:
                    stacked = torch.stack(keys)
                    sims = [float(_hamming(stacked[i:i+1], stacked[j:j+1]).item()) for i, j in pairs]
                    div = round(sum(sims) / len(sims), 4)
                else:
                    div = 1.0
            report[name] = {
                "size":            n,
                "capacity":        self.max_keys,
                "fill_ratio":      round(n / max(self.max_keys, 1), 4),
                "mean_similarity": div,
                "needs_compress":  n > 0.9 * self.max_keys and div > 0.7,
            }
        return report


# ═══════════════════════════════════════════════════════════════════════════════
# Smoke test
# ═══════════════════════════════════════════════════════════════════════════════

def _test_hdc_attention():
    D = 256
    N = 20

    def _hv(s):
        g = torch.Generator(); g.manual_seed(s)
        return (torch.rand(D, generator=g) >= 0.5).float()

    print("=== HDCAttentionHead ===")
    head   = HDCAttentionHead(D, temperature=5.0)
    query  = _hv(0)
    keys   = torch.stack([_hv(i) for i in range(N)])
    out, w = head.attend(query, keys)
    assert out.shape == (D,)
    assert w.shape  == (N,)
    assert abs(float(w.sum()) - 1.0) < 1e-4
    print(f"  out={out.shape}, w_max={w.max():.3f}, w_sum={w.sum():.4f}  OK")

    # Batch
    queries = torch.stack([_hv(i) for i in range(4)])
    out_b, w_b = head.attend_batch(queries, keys)
    assert out_b.shape == (4, D)
    print(f"  batch out={out_b.shape}  OK")

    print("=== MultiHeadHDCAttention ===")
    mh = MultiHeadHDCAttention(D, n_heads=4, temperature=5.0)
    out, hw = mh.attend(query, keys)
    assert out.shape == (D,)
    assert len(hw) == 4
    print(f"  out={out.shape}, {len(hw)} head weights  OK")

    print("=== HDCTransformerLayer ===")
    layer = HDCTransformerLayer(D, n_heads=2)
    seq   = torch.stack([_hv(i) for i in range(10)])
    out, w = layer.forward(seq)
    assert out.shape == (D,)
    out_seq = layer.forward_sequence(seq)
    assert out_seq.shape == (10, D)
    print(f"  forward={out.shape}, forward_sequence={out_seq.shape}  OK")

    print("=== HDCSequenceAttention ===")
    sa = HDCSequenceAttention(D, buffer_size=20, n_heads=2)
    for i in range(25):  # fill buffer and overflow
        ctx, w = sa.step(_hv(i))
    assert ctx.shape == (D,)
    assert sa.buffer_len == 20  # capped at max
    print(f"  ctx={ctx.shape}, buffer={sa.buffer_len}  OK")

    print("=== CrossModalHDCAttention ===")
    cm = CrossModalHDCAttention(D, n_heads=2)
    for i in range(10):
        cm.add_context("sensor",  _hv(i),        _hv(100 + i))
        cm.add_context("action",  _hv(200 + i),  _hv(300 + i))
    out_s, w_s = cm.query(_hv(5), "sensor")
    assert out_s.shape == (D,)
    fused = cm.fuse(_hv(3))
    assert fused.shape == (D,)
    print(f"  sizes={cm.modality_sizes}, fused={fused.shape}  OK")

    print("\n✅ All hdc/attention tests passed")


if __name__ == "__main__":
    _test_hdc_attention()
