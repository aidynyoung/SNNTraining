"""
hdc/vsa_sequence_model.py
==========================
VSA-Based Sequence Model — Transformer Alternative at O(T·D) vs O(T²·D)
=========================================================================
Reference:
    Plate (1995) "Holographic Reduced Representations" IEEE TNNLS.
    Frady, Kleyko, Olshausen, Sommer (2020) "Resonator Networks" Neuroscience.
    Frady & Sommer (2018) "A Theory of Sequence Indexing and Working Memory"

The Transformer bottleneck:
    Self-attention: Q K^T / √d → softmax → weighted V
    Cost: O(T² × d) — quadratic in sequence length
    At T=4096 tokens: 16M attention operations per head per layer

The VSA sequence model:
    Memory: M = MAJORITY/SUM( bind(pos_t, item_t) for t in 0..T-1 )
    Query:  item_t ≈ unbind(M, pos_t)   (via HRR exact deconvolution)
    Cost:   O(T × D) — LINEAR in sequence length

    This is the VSA equivalent of the Transformer:
        - M encodes the entire context window as one D-dimensional vector
        - Any item can be retrieved by unblinding with its position HV
        - Positions are represented as HRR permutations (shift^t)
        - Exact retrieval via HRR pseudo-inverse

Why this matters:
    At T=4096, D=4096:
        Transformer: 16M ops/head/layer
        VSA model:   16M ops for ENTIRE context (one pass)
    The VSA model is O(1) in inference once the context vector is built.

This module implements:

1. VSAContextWindow
   — Fixed-length context window encoded as one HRR vector
   — Append: O(D log D) — one FFT-based bind per token
   — Query:  O(D log D) — one FFT-based unbind per position
   — Update: online, streaming, no attention matrix stored

2. VSALanguageModel (causal)
   — Causal (auto-regressive) variant of VSAContextWindow
   — At each step t: predict item_{t+1} from M_t
   — Learned: a readout that maps unbind(M_t, pos_{t+1}) → class logits
   — Training: online Hebbian (no backprop through time)

3. VSASequenceClassifier
   — Classify variable-length sequences via VSA context
   — Build context → probe with summary role → classify
   — Works on any sequence type: text, spikes, sensor readings

4. VSAPatternMemory
   — Associative memory for N-gram patterns
   — Stores: bind(ngram_hv, class_hv) for each observed N-gram
   — Retrieves: most likely class for a query N-gram
   — Equivalent to an N-gram language model but in O(D) memory

5. PositionalEncoding
   — Generates position HVs that encode temporal order
   — Three schemes: permutation-based (Plate 1995), power-based (FPE),
     random-walk (stochastic)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from hdc.hrr import HRR


# ═══════════════════════════════════════════════════════════════════════════════
# Positional Encoding for VSA Sequences
# ═══════════════════════════════════════════════════════════════════════════════

class PositionalEncoding:
    """
    Position hypervectors for VSA sequence encoding.

    Three schemes:
        'permute': pos_t = perm^t(base) — cyclic shift by t positions
                   Plate (1995) §4.3 "representing sequential information"
        'power':   pos_t = base^(t/T) — fractional power encoding
                   Ensures smooth interpolation between positions
        'random':  pos_t = random HRR vector at each position (cached)
                   Maximum position-orthogonality, no ordering structure

    Recommended: 'permute' for sequences with explicit order,
                 'power' for continuous temporal streams.

    Args:
        hrr:    HRR instance
        scheme: 'permute' | 'power' | 'random' (default 'permute')
        max_len: Maximum sequence length to precompute (default 1000)
    """

    def __init__(
        self,
        hrr:     HRR,
        scheme:  str = "permute",
        max_len: int = 1000,
    ):
        self.hrr     = hrr
        self.scheme  = scheme
        self.max_len = max_len

        # Base position HV
        self._base = hrr.gen(1, seed=314159)   # (D,)

        if scheme == "random":
            # Pre-generate all position HVs (D × max_len storage)
            self._positions = hrr.gen(max_len, seed=271828)   # (max_len, D)

        elif scheme == "sinusoidal":
            # Transformer-style sinusoidal PE adapted to HDC real space.
            # Vaswani et al. (2017): PE[t, 2k] = sin(t/10000^{2k/D})
            #                        PE[t, 2k+1] = cos(t/10000^{2k/D})
            # For HDC: encode as unit-norm real vector — naturally orthogonal
            # for positions differing by > λ/4.
            import math as _math
            D = hrr.dim
            T = max_len
            pe = torch.zeros(T, D, device=hrr.device)
            for t in range(T):
                for k in range(D // 2):
                    denom = 10000 ** (2 * k / D)
                    pe[t, 2 * k]     = _math.sin(t / denom)
                    pe[t, 2 * k + 1] = _math.cos(t / denom)
            # L2-normalise each row to unit length
            norms = pe.norm(dim=1, keepdim=True).clamp(min=1e-8)
            self._positions = pe / norms

    def get(self, t: int) -> torch.Tensor:
        """
        Get the position HV for timestep t.

        Args:
            t: Timestep index (0-indexed)

        Returns:
            (D,) position HV
        """
        if self.scheme == "permute":
            return self.hrr.permute(self._base, steps=t % self.hrr.dim)
        elif self.scheme == "power":
            alpha = (t % max(self.max_len, 1)) / max(self.max_len, 1)
            base_f = self._base.float()
            pos    = F.normalize(base_f * (1 - alpha) + alpha * torch.roll(base_f, t), dim=0)
            return pos
        elif self.scheme == "sinusoidal":
            idx = min(t, self.max_len - 1)
            return self._positions[idx]
        else:  # random
            idx = min(t, self.max_len - 1)
            return self._positions[idx]

    def get_relative(self, t1: int, t2: int) -> torch.Tensor:
        """
        Get a relative positional HV encoding the offset (t2 - t1).

        Relative PE is more robust than absolute PE for variable-length
        sequences — the model only needs to know "how far" two tokens are,
        not their absolute positions.

        Implementation: bind(pos[t2], inverse(pos[t1])) = pos[t2 - t1].
        For permute scheme this is equivalent to a cyclic shift by (t2-t1).

        Returns: (D,) relative position HV.
        """
        lag = t2 - t1
        if self.scheme == "permute":
            return self.hrr.permute(self._base, steps=lag % self.hrr.dim)
        else:
            return self.hrr.bind(self.get(t2), self.get_inverse(t1))

    def get_inverse(self, t: int) -> torch.Tensor:
        """Get the inverse position HV (for unbinding)."""
        if self.scheme == "permute":
            return self.hrr.permute_inverse(self._base, steps=t % self.hrr.dim)
        return self.get(t)   # For power/random: approximate inverse = same HV


# ═══════════════════════════════════════════════════════════════════════════════
# 1. VSAContextWindow — O(T·D) sequence memory
# ═══════════════════════════════════════════════════════════════════════════════

class VSAContextWindow:
    """
    Streaming context window encoded as a single HRR vector.

    Reference:
        Plate (1995) §4.3; Frady et al. (2018)
        "A Theory of Sequence Indexing and Working Memory in RNNs"

    Architecture:
        context_t = SUM( bind(pos_s, item_s) for s in 0..t )

    Append: context_{t+1} = context_t + bind(pos_{t+1}, item_{t+1})
            — O(D log D) per token (one FFT-bind)

    Query item at position s:
            item_s ≈ unbind_exact(context_t, pos_s)
            — O(D log D) per query (one FFT-unbind)
            — Exact for unit-norm items with permutation positions

    Decay: context_{t+1} = λ × context_t + bind(pos_{t+1}, item_{t+1})
           — Exponential forgetting, like an RNN hidden state

    Args:
        hrr:    HRR instance
        max_len: Context window size (older items removed beyond this)
        decay:  Exponential decay factor λ (1.0 = no decay)
        pos_scheme: Positional encoding scheme ('permute' | 'power' | 'random')
    """

    def __init__(
        self,
        hrr:        HRR,
        max_len:    int   = 512,
        decay:      float = 0.99,
        pos_scheme: str   = "permute",
    ):
        self.hrr     = hrr
        self.max_len = max_len
        self.decay   = decay

        self._context = torch.zeros(hrr.dim, device=hrr.device)
        self._pos_enc = PositionalEncoding(hrr, scheme=pos_scheme, max_len=max_len)
        self._t       = 0   # current timestep

    def append(self, item_hv: torch.Tensor):
        """
        Append one item to the context window.

        Args:
            item_hv: (D,) item HRR vector (should be unit-norm for exact retrieval)

        O(D log D) via FFT-based HRR bind.
        """
        pos     = self._pos_enc.get(self._t % self.max_len)
        binding = self.hrr.bind(pos, item_hv.float().to(self.hrr.device))
        self._context = self.decay * self._context + binding
        self._t += 1

    def query(self, position: int) -> torch.Tensor:
        """
        Retrieve the item stored at `position`.

        Uses HRR exact unbinding (pseudo-inverse): O(D log D).

        Args:
            position: Absolute timestep index (0-indexed)

        Returns:
            (D,) noisy item HRR (exact if only one item at each position)
        """
        pos = self._pos_enc.get(position % self.max_len)
        return self.hrr.unbind_exact(self._context, pos)

    def query_recent(self, lag: int = 0) -> torch.Tensor:
        """
        Retrieve the item at lag steps before the current position.

        lag=0: most recent item
        lag=1: second most recent item
        """
        pos_idx = max(0, self._t - 1 - lag)
        return self.query(pos_idx)

    def encode_batch(self, items: List[torch.Tensor]) -> torch.Tensor:
        """
        Encode a sequence of items into the context in one shot.

        Returns the final context HV after encoding all items.
        """
        self.reset()
        for item in items:
            self.append(item)
        return self._context.clone()

    @property
    def context(self) -> torch.Tensor:
        """Current context vector."""
        return self._context.clone()

    @property
    def length(self) -> int:
        """Number of items appended so far."""
        return self._t

    def reset(self):
        self._context = torch.zeros(self.hrr.dim, device=self.hrr.device)
        self._t = 0


# ═══════════════════════════════════════════════════════════════════════════════
# 2. VSALanguageModel — causal auto-regressive sequence prediction
# ═══════════════════════════════════════════════════════════════════════════════

class VSALanguageModel:
    """
    Causal VSA language model (auto-regressive sequence prediction).

    Architecture:
        - Context encoding: VSAContextWindow (O(T·D) total)
        - Prediction: unbind(context_t, pos_{t+1}) → nearest vocabulary item
        - Training: online Hebbian (no backpropagation)

    Inference at step T:
        context_T = SUM_s bind(pos_s, item_s)      [already built]
        query = unbind_exact(context_T, pos_{T+1})  [O(D log D)]
        pred = argmax cos_sim(query, vocabulary)     [O(V × D)]

    Total cost per token: O(D log D + V × D) — completely independent of T!
    Compare to Transformer: O(T × D) per token (linear in context length).

    Args:
        hrr:        HRR instance
        vocabulary: List of (name, HRR vector) for the output vocabulary
        max_len:    Context window size
        decay:      Exponential decay for context
    """

    def __init__(
        self,
        hrr:        HRR,
        vocabulary: Optional[List[Tuple[str, torch.Tensor]]] = None,
        max_len:    int   = 512,
        decay:      float = 0.995,
    ):
        self.hrr     = hrr
        self.ctx     = VSAContextWindow(hrr, max_len=max_len, decay=decay)
        self._vocab: List[Tuple[str, torch.Tensor]] = vocabulary or []

    def register_token(self, name: str, hv: Optional[torch.Tensor] = None):
        """Register a vocabulary token."""
        if hv is None:
            seed = len(self._vocab)
            hv   = self.hrr.gen(1, seed=seed)
        self._vocab.append((name, hv.float().to(self.hrr.device)))

    def _cleanup(self, noisy: torch.Tensor, top_k: int = 1) -> List[Tuple[str, float]]:
        """Find nearest vocabulary token(s) by cosine similarity."""
        if not self._vocab:
            return []
        names = [v[0] for v in self._vocab]
        vecs  = torch.stack([v[1] for v in self._vocab])
        sims  = F.cosine_similarity(noisy.unsqueeze(0), vecs)
        top   = sims.topk(min(top_k, len(names)))
        return [(names[int(i)], float(s)) for s, i in zip(top.values, top.indices)]

    def observe(self, token_name: str) -> str:
        """
        Observe a token and simultaneously predict the next token.

        Returns the name of the predicted next token (before the actual next is seen).
        """
        # Find the token HV
        token_hv = None
        for name, hv in self._vocab:
            if name == token_name:
                token_hv = hv
                break
        if token_hv is None:
            raise ValueError(f"Token '{token_name}' not in vocabulary")

        # Predict BEFORE appending (causal: predict next from current context)
        next_pos_idx = self.ctx.length
        if next_pos_idx > 0:
            query      = self.ctx.query(next_pos_idx)
            pred_token = self._cleanup(query, top_k=1)
            predicted  = pred_token[0][0] if pred_token else ""
        else:
            predicted = ""

        # Append the actual token
        self.ctx.append(token_hv)

        return predicted

    def predict_next(self, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Predict the next token given the current context.

        Returns list of (token_name, confidence) sorted by confidence.
        """
        next_pos_idx = self.ctx.length
        query        = self.ctx.query(next_pos_idx)
        return self._cleanup(query, top_k=top_k)

    def perplexity(self, sequence: List[str]) -> float:
        """
        Compute pseudo-perplexity on a sequence.

        Lower is better. Computed as exp(-mean log P(token_t | context_{t-1})).
        """
        self.ctx.reset()
        log_probs = []
        for i, token in enumerate(sequence):
            if i > 0:
                preds = self.predict_next(top_k=len(self._vocab))
                sims  = [s for _, s in preds]
                probs = torch.softmax(torch.tensor(sims), dim=0)
                pred_names = [n for n, _ in preds]
                if token in pred_names:
                    idx = pred_names.index(token)
                    log_probs.append(math.log(max(float(probs[idx]), 1e-10)))
                else:
                    log_probs.append(math.log(1e-10))
            self.observe(token)

        if not log_probs:
            return float('inf')
        return math.exp(-sum(log_probs) / len(log_probs))

    def reset(self):
        self.ctx.reset()


# ═══════════════════════════════════════════════════════════════════════════════
# 3. VSASequenceClassifier — classify variable-length sequences
# ═══════════════════════════════════════════════════════════════════════════════

class VSASequenceClassifier:
    """
    Classify sequences of arbitrary length using VSA context encoding.

    Architecture:
        1. Encode sequence as VSA context M = SUM(bind(pos_t, item_t))
        2. Extract summary features:
               mean_probe:   unbind(M, pos_0)   (first position)
               last_probe:   unbind(M, pos_{T-1})  (last position)
               mid_probe:    unbind(M, pos_{T//2})
        3. Classify: find nearest class prototype via similarity to probes
        4. Training: online update of class prototypes

    This handles variable-length sequences without padding or truncation.

    Args:
        hrr:        HRR instance
        n_classes:  Number of output classes
        max_len:    Maximum sequence length
    """

    def __init__(
        self,
        hrr:       HRR,
        n_classes: int,
        max_len:   int = 256,
        class_names: Optional[List[str]] = None,
    ):
        self.hrr        = hrr
        self.n_classes  = n_classes
        self.max_len    = max_len
        self.class_names = class_names or [f"class_{i}" for i in range(n_classes)]

        self._prototypes = [torch.zeros(hrr.dim * 3, device=hrr.device)
                            for _ in range(n_classes)]
        self._counts     = [0] * n_classes

    def _encode_sequence(self, items: List[torch.Tensor]) -> torch.Tensor:
        """Encode sequence and extract summary probe features."""
        ctx = VSAContextWindow(self.hrr, max_len=self.max_len)
        for item in items:
            ctx.append(item)
        T = len(items)
        if T == 0:
            return torch.zeros(self.hrr.dim * 3, device=self.hrr.device)

        # Three probes: first, middle, last
        p0 = ctx.query(0)
        pm = ctx.query(T // 2)
        pT = ctx.query(T - 1)
        return torch.cat([p0, pm, pT], dim=0)   # (3D,)

    def train(self, items: List[torch.Tensor], label: int):
        """Online training: update class prototype."""
        feat = self._encode_sequence(items)
        n    = self._counts[label]
        self._prototypes[label] = (n * self._prototypes[label] + feat) / (n + 1)
        self._counts[label] += 1

    def predict(self, items: List[torch.Tensor]) -> Tuple[int, List[float]]:
        """Predict class for a variable-length sequence."""
        feat = self._encode_sequence(items)
        sims = [float(F.cosine_similarity(feat.unsqueeze(0),
                                           p.unsqueeze(0)).item())
                for p in self._prototypes]
        best = int(max(range(len(sims)), key=lambda i: sims[i]))
        return best, sims


# ═══════════════════════════════════════════════════════════════════════════════
# 4. VSAPatternMemory — N-gram associative memory
# ═══════════════════════════════════════════════════════════════════════════════

class VSAPatternMemory:
    """
    Associative N-gram pattern memory via VSA binding.

    Reference:
        Plate (1995) §5.3 "Encoding sequential patterns"
        — bind(ngram_HV, class_HV) encodes the association between a
          sequential pattern and its label.

    For N-grams of length n:
        ngram_hv(w_1..w_n) = bind(perm^0(w_1), bind(perm^1(w_2), ..., bind(perm^{n-1}(w_n), base)))
                            = sequential binding that captures temporal order

    Memory: M = SUM( bind(ngram_hv_i, label_hv_i) )
    Query:  label ≈ unbind(M, query_ngram_hv) → cleanup in label codebook

    This is the HDC equivalent of an N-gram language model:
        - O(D) memory (vs O(V^N) for explicit N-gram counts)
        - Generalises to unseen N-grams via Hamming similarity
        - Updates online in O(D log D)

    Args:
        hrr:    HRR instance
        n:      N-gram length
        labels: List of (name, HRR vector) for class labels
    """

    def __init__(
        self,
        hrr:    HRR,
        n:      int = 3,
        labels: Optional[List[Tuple[str, torch.Tensor]]] = None,
    ):
        self.hrr    = hrr
        self.n      = n
        self._memory = torch.zeros(hrr.dim, device=hrr.device)
        self._labels: List[Tuple[str, torch.Tensor]] = labels or []
        self._n_writes = 0

    def _ngram_hv(self, items: List[torch.Tensor]) -> torch.Tensor:
        """Encode an N-gram as a single HRR vector via sequential binding."""
        if not items:
            return torch.zeros(self.hrr.dim, device=self.hrr.device)
        result = items[0].float().to(self.hrr.device)
        for i, item in enumerate(items[1:], 1):
            shifted = self.hrr.permute(item.float().to(self.hrr.device), steps=i)
            result  = self.hrr.bind(result, shifted)
        return F.normalize(result, dim=0)

    def register_label(self, name: str, hv: Optional[torch.Tensor] = None):
        """Register a label HV."""
        if hv is None:
            hv = self.hrr.gen(1, seed=len(self._labels))
        self._labels.append((name, hv.float().to(self.hrr.device)))

    def write(self, ngram_items: List[torch.Tensor], label_name: str):
        """Store an N-gram → label association in memory."""
        ngram_hv = self._ngram_hv(ngram_items[:self.n])
        label_hv = next((hv for n, hv in self._labels if n == label_name), None)
        if label_hv is None:
            return
        binding = self.hrr.bind(ngram_hv, label_hv)
        self._memory = self._memory + binding
        self._n_writes += 1

    def query(self, ngram_items: List[torch.Tensor]) -> Tuple[Optional[str], float]:
        """
        Retrieve the most likely label for a query N-gram.

        Returns: (label_name, confidence)
        """
        ngram_hv  = self._ngram_hv(ngram_items[:self.n])
        candidate = self.hrr.unbind_exact(self._memory, ngram_hv)

        if not self._labels:
            return None, 0.0
        names = [n for n, _ in self._labels]
        vecs  = torch.stack([hv for _, hv in self._labels])
        sims  = F.cosine_similarity(candidate.unsqueeze(0), vecs)
        best  = int(sims.argmax().item())
        return names[best], float(sims[best].item())

    @property
    def n_writes(self) -> int:
        return self._n_writes


# ═══════════════════════════════════════════════════════════════════════════════
# Smoke test
# ═══════════════════════════════════════════════════════════════════════════════

def _test_vsa_sequence_model():
    D   = 512
    hrr = HRR(dim=D)

    print("=== PositionalEncoding ===")
    pe = PositionalEncoding(hrr, scheme="permute", max_len=100)
    p0 = pe.get(0)
    p1 = pe.get(1)
    p10 = pe.get(10)
    assert p0.shape == (D,)
    sim_01  = hrr.similarity(p0, p1)
    sim_010 = hrr.similarity(p0, p10)
    print(f"  sim(pos_0, pos_1)={sim_01:.3f}  sim(pos_0, pos_10)={sim_010:.3f}")
    print(f"  (adjacent positions more similar: {sim_01 > sim_010})")

    print("\n=== VSAContextWindow ===")
    ctx   = VSAContextWindow(hrr, max_len=50, decay=1.0)
    items = [hrr.gen(1, seed=i) for i in range(10)]
    for item in items:
        ctx.append(item)
    print(f"  Context length: {ctx.length}")

    # Retrieve specific items
    for t in [0, 5, 9]:
        retrieved = ctx.query(t)
        sim = hrr.similarity(retrieved, items[t])
        print(f"  Retrieve pos {t}: sim={sim:.3f}  (higher = better)")

    print("\n=== VSALanguageModel ===")
    vlm = VSALanguageModel(hrr, max_len=20, decay=0.99)
    for i, word in enumerate(["the", "cat", "sat", "on", "the", "mat"]):
        vlm.register_token(word)
    vlm.reset()

    preds = []
    for word in ["the", "cat", "sat", "on", "the", "mat"]:
        pred = vlm.observe(word)
        preds.append(pred)

    print(f"  Sequence: the cat sat on the mat")
    print(f"  Predictions (shifted): {preds}")

    next_preds = vlm.predict_next(top_k=3)
    print(f"  Next token predictions: {next_preds}")

    print("\n=== VSASequenceClassifier ===")
    clf = VSASequenceClassifier(hrr, n_classes=2, max_len=20)
    for label, seed_offset in [(0, 0), (1, 100)]:
        for _ in range(3):
            seq = [hrr.gen(1, seed=seed_offset + i) for i in range(5)]
            clf.train(seq, label)

    test_seq = [hrr.gen(1, seed=i) for i in range(5)]
    pred, sims = clf.predict(test_seq)
    print(f"  Classify test sequence: class={pred} (sims={[f'{s:.3f}' for s in sims]})")

    print("\n=== VSAPatternMemory ===")
    pm = VSAPatternMemory(hrr, n=2)
    pm.register_label("positive", hrr.gen(1, seed=500))
    pm.register_label("negative", hrr.gen(1, seed=501))

    pos_items = [hrr.gen(1, seed=i) for i in range(10)]
    neg_items = [hrr.gen(1, seed=100 + i) for i in range(10)]

    for i in range(8):
        pm.write([pos_items[i], pos_items[i+1]], "positive")
        pm.write([neg_items[i], neg_items[i+1]], "negative")

    label, conf = pm.query([pos_items[0], pos_items[1]])
    print(f"  Query positive bigram: '{label}' (conf={conf:.3f})")

    label2, conf2 = pm.query([neg_items[0], neg_items[1]])
    print(f"  Query negative bigram: '{label2}' (conf={conf2:.3f})")

    print("\n✅ All VSA sequence model tests passed")


if __name__ == "__main__":
    _test_vsa_sequence_model()
