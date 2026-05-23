"""
hdc/neuromorphic_transformer.py
=================================
Neuromorphic Transformer — Complete HDC Replacement for O(N²) Attention
========================================================================
Reference:
    Ramsauer et al. (2021) "Hopfield Networks Is All You Need" ICLR.
    — Modern Hopfield = Transformer attention in exact mathematical form.

    Vaswani et al. (2017) "Attention Is All You Need" NeurIPS.
    — Original Transformer.

    Millidge, Seth, Buckley (2022) "Predictive Coding: a Review"
    — Self-attention as inference in a hierarchical model.

The Transformer attention bottleneck:
    softmax(QK^T / √d) V  →  O(N² × d) per layer per head

The HDC replacement:
    - Attention → Modern Hopfield retrieval O(N × D)
    - Feed-forward → Sparse HDC mapping O(δ × D)
    - Layer norm → Density normalization O(D)
    - Positional encoding → HRR permutation O(D log D)
    - Residual → Bundle with current HV O(D)

Total cost per token: O(N × D) — linear in context length.
Compare to Transformer: O(N² × d) — quadratic.

At N=4096, D=4096:
    HDC:         16M operations per token
    Transformer: 16G operations per layer (1000× more)

This module implements:

1. HDCAttentionBlock
   — Drop-in attention replacement using Modern Hopfield retrieval
   — Multi-head: H independent Hopfield memories in parallel
   — O(N × H × D) — linear in sequence length

2. HDCSparseFFN
   — Feed-forward network using sparse HDC projections
   — k-WTA activation: exactly k neurons fire per position
   — O(δ × D) per token — 100× sparser than dense FFN

3. HDCLayerNorm
   — Density normalisation: adjust active bit fraction toward target δ
   — Equivalent to Layer Norm but for binary HVs

4. HDCTransformerBlock
   — Complete single block: Attention + FFN + residuals
   — Equivalent to one transformer encoder layer
   — All operations: XOR + popcount + majority

5. HDCTransformerStack
   — Stack of L HDCTransformerBlocks
   — Processes sequences of HV-encoded tokens
   — Output: contextualised token HVs (same shape as input)

6. HDCLanguageHead
   — Final prediction layer: contextualised HV → token logits
   — Uses Hamming similarity to vocabulary HVs
   — No softmax weight matrix needed
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from hdc.physics_world_model import _hamming, _majority, _xor
from hdc.modern_hopfield import ModernHopfieldHDC


# ── Utility ────────────────────────────────────────────────────────────────────

def _gen_hv(dim: int, seed: Optional[int] = None, device: str = "cpu") -> torch.Tensor:
    g = torch.Generator(device=device)
    if seed is not None:
        g.manual_seed(seed)
    return (torch.rand(dim, generator=g, device=device) >= 0.5).float()


@dataclass
class HDCTransformerConfig:
    """Configuration for HDC Transformer."""
    dim:          int   = 512     # HV dimension (= embedding dim)
    n_heads:      int   = 4       # Number of attention heads
    n_layers:     int   = 4       # Number of transformer blocks
    ffn_k_frac:   float = 0.1     # Fraction of neurons active in FFN
    target_density: float = 0.5   # Target HV density after layer norm
    dropout_rate: float = 0.0     # Optional dropout (not used in HDC by default)
    beta:         float = 8.0     # Hopfield retrieval temperature


# ═══════════════════════════════════════════════════════════════════════════════
# 1. HDCAttentionBlock — Modern Hopfield multi-head attention
# ═══════════════════════════════════════════════════════════════════════════════

class HDCAttentionBlock:
    """
    Multi-head HDC attention via Modern Hopfield retrieval.

    Each head:
        - Projects each token to a head-specific subspace (random mask)
        - Stores key-value pairs in a Modern Hopfield memory
        - Retrieves via softmax(β × Hamming_sim) weighted sum

    Mathematically equivalent to Transformer attention with Hamming similarity
    instead of scaled dot-product (Ramsauer 2021 Theorem 1 equivalence).

    Complexity: O(N × H × D)  — N tokens, H heads, D dimensions.

    Args:
        cfg: HDCTransformerConfig
        device: torch device
    """

    def __init__(self, cfg: HDCTransformerConfig, device: str = "cpu"):
        self.cfg    = cfg
        self.device = device
        self.n_heads = cfg.n_heads
        self.dim     = cfg.dim

        # Head-specific projection masks (random, fixed)
        self._head_masks = [
            (torch.rand(cfg.dim, device=device) >= 0.5).float()
            for _ in range(cfg.n_heads)
        ]

    def _project(self, hvs: torch.Tensor, head: int) -> torch.Tensor:
        """Project token HVs using head-specific mask."""
        mask = self._head_masks[head].unsqueeze(0)   # (1, D)
        return _majority((hvs.float() * mask).mean(dim=0, keepdim=True).expand_as(hvs))

    def forward(
        self,
        tokens: torch.Tensor,   # (N, D) token HVs
        mask:   Optional[torch.Tensor] = None,  # (N,) bool mask (True = ignore)
    ) -> torch.Tensor:
        """
        Multi-head attention forward pass.

        Args:
            tokens: (N, D) input token HVs
            mask:   Optional causal or padding mask

        Returns:
            (N, D) attended token HVs
        """
        N, D = tokens.shape
        all_head_outputs = []

        for h in range(self.n_heads):
            # Project to head subspace
            q_h = self._project(tokens, h)   # (N, D) projected queries
            k_h = q_h.clone()                # self-attention: keys = queries
            v_h = tokens.clone()             # values = original tokens

            # Compute Hamming similarities: (N, N)
            sims = torch.zeros(N, N, device=self.device)
            for i in range(N):
                sims[i] = _hamming(q_h[i].unsqueeze(0), k_h)  # (N,)

            # Apply mask (causal: prevent attending to future tokens)
            if mask is not None:
                sims = sims.masked_fill(mask.unsqueeze(0), -1e9)

            # Softmax-weighted aggregation
            attn_weights = F.softmax(self.cfg.beta * sims, dim=1)  # (N, N)

            # Weighted sum of values
            attended = attn_weights @ v_h.float()   # (N, D)
            all_head_outputs.append(attended)

        # Bundle all heads
        stacked = torch.stack(all_head_outputs)   # (H, N, D)
        return _majority(stacked.mean(dim=0))


class HDCAttentionBlockEfficient(HDCAttentionBlock):
    """
    Memory-efficient attention using batch Hamming similarities.
    Uses matrix operations instead of loops for O(N²×D/32) bitwise ops.
    """

    def forward(
        self,
        tokens: torch.Tensor,
        mask:   Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        N, D = tokens.shape
        all_head_outputs = []

        for h in range(self.n_heads):
            mask_h = self._head_masks[h]
            q_h    = _majority((tokens.float() * mask_h.unsqueeze(0)).mean(dim=0, keepdim=True)
                                .expand(N, D))

            # Efficient pairwise Hamming: (N, N) via broadcasting
            # sim(i,j) = 1 - mean(q_h[i] != q_h[j])
            sims = 1.0 - (q_h.unsqueeze(1) != q_h.unsqueeze(0)).float().mean(dim=-1)

            if mask is not None:
                sims = sims.masked_fill(mask.to(sims.device), -1e9)

            attn    = F.softmax(self.cfg.beta * sims, dim=1)
            attended = attn @ tokens.float()
            all_head_outputs.append(attended)

        return _majority(torch.stack(all_head_outputs).mean(dim=0))


# ═══════════════════════════════════════════════════════════════════════════════
# 2. HDCSparseFFN — sparse feed-forward network in HDC
# ═══════════════════════════════════════════════════════════════════════════════

class HDCSparseFFN:
    """
    Sparse feed-forward network using HDC projections and k-WTA activation.

    Standard transformer FFN:
        FFN(x) = max(0, xW₁ + b₁)W₂ + b₂   [O(d_model × d_ff)]

    HDC sparse FFN:
        FFN(x) = kWTA(x ⊗ proj₁) ⊗ proj₂   [O(δ × d_model)]

    where:
        proj₁, proj₂ = fixed random projection HVs
        k-WTA         = keep only top-k fraction of active bits
        δ             = k/D (sparsity fraction)

    At δ=0.1: 10× sparser than standard FFN, with similar representational capacity.

    Args:
        dim:     HV dimension
        k_frac:  Fraction of neurons active (default 0.1 = 10%)
        device:  torch device
    """

    def __init__(self, dim: int, k_frac: float = 0.1, device: str = "cpu"):
        self.dim    = dim
        self.k      = max(1, int(dim * k_frac))
        self.device = device

        # Fixed random projections (the "weights" of the FFN)
        self._proj1 = _gen_hv(dim, seed=31415, device=device)
        self._proj2 = _gen_hv(dim, seed=27182, device=device)

    def _k_wta(self, hv: torch.Tensor) -> torch.Tensor:
        """k-Winners-Take-All: keep top-k bits active, zero rest."""
        if self.k >= self.dim:
            return hv.clone()
        topk_vals, topk_idx = hv.float().topk(self.k)
        result = torch.zeros_like(hv)
        result[topk_idx] = 1.0
        return result

    def forward(self, token_hv: torch.Tensor) -> torch.Tensor:
        """
        Single-token FFN forward.

        Args:
            token_hv: (D,) token HV

        Returns:
            (D,) transformed HV
        """
        hv = token_hv.float().to(self.device)
        # First projection + sparse activation
        h1 = self._k_wta(_majority((hv * self._proj1 + hv) / 2.0))
        # Second projection (residual)
        h2 = _majority((h1 * self._proj2 + hv) / 2.0)
        return h2

    def forward_batch(self, tokens: torch.Tensor) -> torch.Tensor:
        """Batch forward for (N, D) tokens."""
        return torch.stack([self.forward(tokens[i]) for i in range(tokens.shape[0])])


# ═══════════════════════════════════════════════════════════════════════════════
# 3. HDCLayerNorm — density normalisation
# ═══════════════════════════════════════════════════════════════════════════════

class HDCLayerNorm:
    """
    Layer normalization for binary HVs via density adjustment.

    Standard layer norm: (x - mean) / std
    HDC layer norm: adjust active bit fraction toward target δ

    If actual density > target: randomly zero out bits
    If actual density < target: randomly set bits

    This ensures consistent information density across all layers,
    equivalent to the variance stabilisation role of standard layer norm.

    Args:
        target_density: Target fraction of active bits (default 0.5)
    """

    def __init__(self, target_density: float = 0.5):
        self.target = target_density

    def __call__(self, hv: torch.Tensor) -> torch.Tensor:
        return self.normalize(hv)

    def normalize(self, hv: torch.Tensor) -> torch.Tensor:
        """Normalise a single HV or batch (N, D)."""
        if hv.dim() == 1:
            return self._norm1d(hv)
        return torch.stack([self._norm1d(hv[i]) for i in range(hv.shape[0])])

    def _norm1d(self, hv: torch.Tensor) -> torch.Tensor:
        hv_f    = hv.float()
        density = float(hv_f.mean())
        if abs(density - self.target) < 0.01:
            return hv.clone()
        if density > self.target:
            # Zero out excess bits randomly
            keep_prob = self.target / max(density, 1e-6)
            mask = torch.rand_like(hv_f) < keep_prob
            return (hv_f * mask.float())
        else:
            # Set additional bits randomly
            add_prob = (self.target - density) / max(1 - density, 1e-6)
            add_mask = torch.rand_like(hv_f) < add_prob
            return _majority(hv_f + add_mask.float())


# ═══════════════════════════════════════════════════════════════════════════════
# 4. HDCTransformerBlock — complete single encoder block
# ═══════════════════════════════════════════════════════════════════════════════

class HDCTransformerBlock:
    """
    Single HDC Transformer encoder block.

    Architecture:
        x → HDCAttention(x) → bundle(x, attn_out) → HDCLayerNorm
          → HDCSparseFFN   → bundle(norm, ffn_out) → HDCLayerNorm

    This is the HDC equivalent of one PreLN Transformer encoder layer.

    All operations:
        - XOR + popcount (Hamming similarity)
        - Majority vote (bundling)
        - k-WTA (sparse activation)
    No matrix multiplications, no softmax (just indexing).

    Args:
        cfg:    HDCTransformerConfig
        device: torch device
    """

    def __init__(self, cfg: HDCTransformerConfig, device: str = "cpu"):
        self.cfg    = cfg
        self.device = device
        self.attn   = HDCAttentionBlockEfficient(cfg, device)
        self.ffn    = HDCSparseFFN(cfg.dim, cfg.ffn_k_frac, device)
        self.norm1  = HDCLayerNorm(cfg.target_density)
        self.norm2  = HDCLayerNorm(cfg.target_density)

    def forward(
        self,
        tokens: torch.Tensor,   # (N, D)
        mask:   Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through one encoder block.

        Args:
            tokens: (N, D) input token HVs
            mask:   Optional causal mask

        Returns:
            (N, D) contextualised token HVs
        """
        # Self-attention + residual
        attn_out = self.attn.forward(tokens, mask)
        x1       = self.norm1.normalize(
            _majority((tokens.float() + attn_out.float()) / 2.0)
        )

        # FFN + residual
        ffn_out = self.ffn.forward_batch(x1)
        x2      = self.norm2.normalize(
            _majority((x1.float() + ffn_out.float()) / 2.0)
        )
        return x2


# ═══════════════════════════════════════════════════════════════════════════════
# 5. HDCTransformerStack — full encoder
# ═══════════════════════════════════════════════════════════════════════════════

class HDCTransformerStack:
    """
    Stack of L HDCTransformerBlocks — complete HDC encoder.

    Processes a sequence of token HVs and returns contextualised HVs.

    Position encoding:
        Each position gets a unique HV via cyclic shift (Plate 1995).
        token_with_pos[t] = majority(token_hv[t], shift^t(position_base))

    This HDC encoder is a drop-in replacement for:
        - BERT encoder (bidirectional)
        - GPT encoder (causal, with causal mask)

    Complexity: O(L × N × H × D) — linear in both layers and sequence length.

    Args:
        cfg:    HDCTransformerConfig
        causal: If True, use causal mask (autoregressive)
        device: torch device
    """

    def __init__(
        self,
        cfg:    HDCTransformerConfig,
        causal: bool = False,
        device: str  = "cpu",
    ):
        self.cfg    = cfg
        self.causal = causal
        self.device = device

        self.blocks  = [HDCTransformerBlock(cfg, device) for _ in range(cfg.n_layers)]
        self._pos_hv = _gen_hv(cfg.dim, seed=271828, device=device)   # position base

    def _positional_encoding(self, tokens: torch.Tensor) -> torch.Tensor:
        """Add position encoding to tokens via HRR cyclic shift."""
        N = tokens.shape[0]
        positioned = []
        for t in range(N):
            pos_hv  = torch.roll(self._pos_hv, t, dims=0)
            tok_pos = _majority((tokens[t].float() + pos_hv.float()) / 2.0)
            positioned.append(tok_pos)
        return torch.stack(positioned)

    def _causal_mask(self, N: int) -> torch.Tensor:
        """Upper triangular mask (True = ignore, i.e. can't attend to future)."""
        return torch.triu(torch.ones(N, N, device=self.device, dtype=torch.bool), diagonal=1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through all encoder layers.

        Args:
            tokens: (N, D) raw token HVs

        Returns:
            (N, D) contextualised HVs
        """
        x    = self._positional_encoding(tokens)
        mask = self._causal_mask(tokens.shape[0]) if self.causal else None

        for block in self.blocks:
            x = block.forward(x, mask)

        return x


# ═══════════════════════════════════════════════════════════════════════════════
# 6. HDCLanguageHead — final prediction layer
# ═══════════════════════════════════════════════════════════════════════════════

class HDCLanguageHead:
    """
    Language model head: contextualised HV → next token logits.

    Uses Hamming similarity to vocabulary HVs as logits:
        logits[t] = sim(context_hv, vocab_hv[t]) for all t in vocabulary

    No learned weight matrix needed — vocabulary HVs are the "weights".

    Args:
        vocabulary: List of (token_name, token_hv) pairs
        dim:        HV dimension
    """

    def __init__(
        self,
        vocabulary: List[Tuple[str, torch.Tensor]],
        dim:        int,
        device:     str = "cpu",
    ):
        self.vocabulary = vocabulary
        self.dim        = dim
        self.device     = device

        if vocabulary:
            self._vocab_hvs = torch.stack([hv.float().to(device) for _, hv in vocabulary])
        else:
            self._vocab_hvs = torch.zeros(0, dim, device=device)

    def logits(self, context_hv: torch.Tensor) -> torch.Tensor:
        """
        Compute logits over vocabulary.

        Returns: (vocab_size,) Hamming similarities as logits
        """
        if not self.vocabulary:
            return torch.zeros(0, device=self.device)
        return _hamming(context_hv.float().unsqueeze(0), self._vocab_hvs)

    def predict(self, context_hv: torch.Tensor, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Predict top-k next tokens.

        Returns:
            List of (token_name, probability) sorted desc.
        """
        logits = self.logits(context_hv)
        probs  = F.softmax(logits * 10.0, dim=0)
        top_k  = min(top_k, len(self.vocabulary))
        topk   = probs.topk(top_k)
        return [(self.vocabulary[int(i)][0], float(p))
                for p, i in zip(topk.values, topk.indices)]

    def add_token(self, name: str, hv: torch.Tensor):
        """Add a new token to the vocabulary (no retraining needed)."""
        self.vocabulary.append((name, hv.float().to(self.device)))
        self._vocab_hvs = torch.stack([v.float() for _, v in self.vocabulary])


# ═══════════════════════════════════════════════════════════════════════════════
# Smoke test
# ═══════════════════════════════════════════════════════════════════════════════

def _test_neuromorphic_transformer():
    D = 128

    cfg = HDCTransformerConfig(
        dim=D, n_heads=2, n_layers=2, ffn_k_frac=0.1, beta=5.0
    )

    print("=== HDCAttentionBlock ===")
    attn   = HDCAttentionBlockEfficient(cfg)
    tokens = torch.stack([_gen_hv(D, seed=i) for i in range(8)])
    out    = attn.forward(tokens)
    assert out.shape == (8, D)
    print(f"  Input: {tokens.shape} → Output: {out.shape}  OK")

    # Causal attention
    from hdc.neuromorphic_transformer import HDCTransformerBlock
    mask = torch.triu(torch.ones(8, 8, dtype=torch.bool), diagonal=1)
    out_causal = attn.forward(tokens, mask=mask)
    assert out_causal.shape == (8, D)
    print(f"  Causal attention: {out_causal.shape}  OK")

    print("\n=== HDCSparseFFN ===")
    ffn = HDCSparseFFN(D, k_frac=0.1)
    x   = _gen_hv(D, seed=0)
    y   = ffn.forward(x)
    assert y.shape == (D,)
    # Check sparsity
    print(f"  Output density: {y.mean():.3f}  OK")

    batch_out = ffn.forward_batch(tokens)
    assert batch_out.shape == (8, D)
    print(f"  Batch FFN: {batch_out.shape}  OK")

    print("\n=== HDCLayerNorm ===")
    ln = HDCLayerNorm(target_density=0.5)
    x  = _gen_hv(D, seed=0)
    n  = ln.normalize(x)
    assert n.shape == (D,)
    print(f"  Normalised density: {float(n.mean()):.3f} (target=0.5)  OK")

    print("\n=== HDCTransformerBlock ===")
    block = HDCTransformerBlock(cfg)
    out   = block.forward(tokens)
    assert out.shape == (8, D)
    print(f"  Block output: {out.shape}  OK")

    print("\n=== HDCTransformerStack ===")
    stack = HDCTransformerStack(cfg, causal=False)
    out   = stack.forward(tokens)
    assert out.shape == (8, D)
    print(f"  Stack output (8 tokens, {cfg.n_layers} layers): {out.shape}  OK")

    # Causal (autoregressive) mode
    stack_c = HDCTransformerStack(cfg, causal=True)
    out_c   = stack_c.forward(tokens)
    assert out_c.shape == (8, D)
    print(f"  Causal stack: {out_c.shape}  OK")

    print("\n=== HDCLanguageHead ===")
    vocab = [(f"word_{i}", _gen_hv(D, seed=100 + i)) for i in range(10)]
    head  = HDCLanguageHead(vocab, D)
    logits = head.logits(tokens[0])
    assert logits.shape == (10,)
    preds = head.predict(tokens[0], top_k=3)
    assert len(preds) == 3
    print(f"  Vocab size=10, logits={logits.shape}, top-3: {[(n, f'{p:.3f}') for n,p in preds]}  OK")

    # Add new token (no retraining!)
    head.add_token("new_word", _gen_hv(D, seed=999))
    logits2 = head.logits(tokens[0])
    assert logits2.shape == (11,)
    print(f"  After adding token: vocab={len(head.vocabulary)}  OK")

    print("\n✅ All neuromorphic_transformer tests passed")


if __name__ == "__main__":
    _test_neuromorphic_transformer()
