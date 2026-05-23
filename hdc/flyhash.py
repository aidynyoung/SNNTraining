"""
FlyHash: Similarity-Preserving Sparse Randomized Embeddings
============================================================
Kleyko & Rachkovskij (2025) "On Design Choices in Similarity-Preserving
Sparse Randomized Embeddings." arXiv:2501.14741

Implements the "Expand & Sparsify" principle observed in Drosophila melanogaster
olfactory system and mammalian cerebellum:
  1. Expand: project d-dim input to much larger D-dim space (D >> d)
  2. Sparsify: activate only the k most excited neurons (Winner-Take-All)

The resulting sparse binary embedding preserves similarity: inputs with high
cosine similarity produce sparse vectors with high overlap (Jaccard similarity).

Three key design choices studied in the paper:
  Preprocessing: original FlyHash | mean-center | L2-normalize | both
  RP matrix M:   sparse binary | Gaussian | bipolar {-1,+1}
  Sparsification: kWTA (top-k) | Block sparse (1 winner per block)

Key findings (Table I / Fig 1 of the paper):
  - Mean centering + L2 normalization gives best MAP across datasets
  - Block sparse codes use fewer bits than kWTA at same similarity quality
  - Gaussian M slightly better than sparse binary in some settings

Improvements over DenseToHV (hdc/physical_ai_hybrid.py):
  - DenseToHV: dense binary output (50% ones), Gaussian M, sign() activation
  - FlyHash: SPARSE binary output (k/D ones, typically 1-5%), sparse M, kWTA
  - Sparse output enables Jaccard similarity (more discriminative than Hamming)
  - Fewer active bits → faster AND-based similarity on sparse hardware
  - Biologically plausible (matches Drosophila circuit structure)

Reference implementation: github.com/facebookresearch/ParlAI (FlyHash)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════════
# 1. FlyHashEncoder — core Expand & Sparsify
# ═══════════════════════════════════════════════════════════════════════════════

class FlyHashEncoder(nn.Module):
    """
    FlyHash: biologically-inspired sparse randomized embedding.

    Algorithm (Kleyko & Rachkovskij 2025, Eq. 1-5):
      1. Preprocess:  x̃ = preprocess(x)
      2. Expand:      y = M @ x̃        (M is sparse binary, D × d)
      3. Sparsify:    z = kWTA(y, k)   or block-sparse(y, k)
      4. Binarize:    z_bin = (z > 0)

    The projection matrix M is sparse: each row has exactly s nonzero
    entries (connections from input to this output neuron). Small s (e.g.,
    s=10 out of d=128) → very fast sparse matrix-vector product.

    Similarity preservation (key property):
      If cos_sim(x, x') is high, then Jaccard(z_bin, z_bin') is high.
      Jaccard(a, b) = |a AND b| / |a OR b|

    Design choices supported:
      preprocessing: 'none' | 'mean_center' | 'l2' | 'mean_center+l2'
      sparsify: 'kwta' (top-k) | 'block' (1 per block)
      matrix: 'sparse_binary' | 'gaussian' | 'bipolar'

    Args:
        input_dim: d — dimensionality of input vectors
        output_dim: D — embedding dimension (typically 20k, 20× larger than d)
        k: Number of active neurons in output (sparsity = k/D)
        sparsify: Sparsification strategy
        matrix: RP matrix type
        preprocessing: Input preprocessing
        connections_per_neuron: s — connections per output neuron (for sparse M)
        seed: Random seed
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        k: int = 10,
        sparsify: str = "kwta",
        matrix: str = "sparse_binary",
        preprocessing: str = "mean_center+l2",
        connections_per_neuron: int = 10,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.k = k
        self.sparsify = sparsify
        self.preprocessing = preprocessing

        g = torch.Generator()
        if seed is not None:
            g.manual_seed(seed)

        # Build projection matrix M (D × d)
        M = self._build_matrix(input_dim, output_dim, matrix,
                               connections_per_neuron, g)
        self.register_buffer("M", M)

        # Dataset statistics for preprocessing (set via fit_preprocessing)
        self.register_buffer("data_mean", torch.zeros(input_dim))
        self._fitted = False

    def _build_matrix(
        self,
        d: int,
        D: int,
        matrix_type: str,
        s: int,
        g: torch.Generator,
    ) -> torch.Tensor:
        """Build projection matrix M (D × d)."""
        if matrix_type == "sparse_binary":
            # Each row has exactly s ones at random positions
            M = torch.zeros(D, d)
            for i in range(D):
                cols = torch.randperm(d, generator=g)[:s]
                M[i, cols] = 1.0
        elif matrix_type == "gaussian":
            M = torch.randn(D, d, generator=g) / math.sqrt(s)
        elif matrix_type == "bipolar":
            M = (torch.rand(D, d, generator=g) < 0.5).float() * 2 - 1
            # Sparsify: keep s connections per row
            mask = torch.zeros(D, d)
            for i in range(D):
                cols = torch.randperm(d, generator=g)[:s]
                mask[i, cols] = 1.0
            M = M * mask / math.sqrt(s)
        else:
            raise ValueError(f"Unknown matrix type: {matrix_type}")
        return M

    def fit_preprocessing(self, X: torch.Tensor):
        """
        Fit dataset statistics for preprocessing.

        Args:
            X: (N, d) dataset matrix
        """
        self.data_mean = X.mean(dim=0)
        self._fitted = True

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Apply selected preprocessing (Kleyko 2025, Eqs. 6-10)."""
        if self.preprocessing == "none":
            return x
        elif self.preprocessing == "mean_center":
            # Subtract dataset mean (Eq. 8)
            if self._fitted:
                return x - self.data_mean
            return x
        elif self.preprocessing == "l2":
            # L2 normalize (Eq. 9)
            return F.normalize(x, p=2, dim=-1)
        elif self.preprocessing in ("mean_center+l2", "both"):
            # Mean center then L2 normalize (Eq. 10)
            if self._fitted:
                x = x - self.data_mean
            return F.normalize(x, p=2, dim=-1)
        else:
            raise ValueError(f"Unknown preprocessing: {self.preprocessing}")

    def _kwta(self, y: torch.Tensor) -> torch.Tensor:
        """
        k-Winner-Take-All: keep top-k values, zero out rest (Eq. 3).

        Args:
            y: (..., D) pre-activation

        Returns:
            (..., D) sparse tensor with at most k nonzero per row
        """
        # Find top-k threshold
        topk_vals, _ = torch.topk(y, self.k, dim=-1)
        threshold = topk_vals[..., -1:]   # k-th largest value
        mask = (y >= threshold).float()
        return y * mask

    def _block_sparse(self, y: torch.Tensor) -> torch.Tensor:
        """
        Block-sparse: 1 winner per block of size D/k (Eq. 5).

        Divides output into k blocks of equal size, activates the maximum
        neuron in each block. More hardware-friendly than kWTA.

        Args:
            y: (..., D) pre-activation

        Returns:
            (..., D) tensor with exactly k nonzero (one per block)
        """
        D = y.shape[-1]
        block_size = D // self.k
        result = torch.zeros_like(y)

        for b in range(self.k):
            start = b * block_size
            end = min(start + block_size, D)
            block = y[..., start:end]
            max_idx = block.argmax(dim=-1, keepdim=True)
            result[..., start:end].scatter_(-1, max_idx, block.gather(-1, max_idx))

        return result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute sparse binary FlyHash embedding.

        Args:
            x: (..., input_dim) input vector(s)

        Returns:
            (..., output_dim) sparse binary embedding (k/D ones per vector)
        """
        # Step 1: Preprocess
        x_pre = self._preprocess(x.float())

        # Step 2: Expand via RP matrix (Eq. 1)
        y = x_pre @ self.M.T      # (..., D)

        # Step 3: Sparsify
        if self.sparsify == "kwta":
            z = self._kwta(y)
        elif self.sparsify == "block":
            z = self._block_sparse(y)
        else:
            raise ValueError(f"Unknown sparsify: {self.sparsify}")

        # Step 4: Binarize (Eq. 4)
        return (z > 0).float()

    def jaccard_similarity(self, a: torch.Tensor, b: torch.Tensor) -> float:
        """
        Jaccard similarity between two sparse binary embeddings.

        Jaccard(a, b) = |a AND b| / |a OR b|
        For sparse embeddings (k/D << 0.5), Jaccard is more discriminative
        than Hamming similarity.

        Args:
            a, b: (D,) binary sparse embeddings

        Returns:
            Jaccard similarity ∈ [0, 1]
        """
        intersection = (a * b).sum()
        union = ((a + b) > 0).float().sum()
        if union < 1:
            return 1.0
        return float(intersection / union)

    def mean_average_precision(
        self,
        X: torch.Tensor,
        K: int = 50,
    ) -> float:
        """
        Estimate MAP@K of the embeddings (evaluation metric from paper).

        MAP measures how well similarity in embedding space matches
        similarity in original space.

        Args:
            X: (N, d) dataset
            K: Number of neighbors to evaluate

        Returns:
            MAP@K ∈ [0, 1]
        """
        N = X.shape[0]
        embeddings = self.forward(X)   # (N, D)

        # Similarity in original space (cosine)
        X_norm = F.normalize(X.float(), p=2, dim=-1)
        orig_sims = X_norm @ X_norm.T   # (N, N)

        # Similarity in embedding space (Hamming-based for sparse)
        # Use intersection / k as proxy for Jaccard
        inter = embeddings @ embeddings.T      # (N, N)
        embed_sims = inter / max(self.k, 1)   # (N, N)

        # MAP@K
        K = min(K, N - 1)
        avg_precisions = []
        for i in range(min(100, N)):   # subsample for speed
            orig_rank = orig_sims[i].argsort(descending=True)[1:K+1]
            embed_rank = embed_sims[i].argsort(descending=True)[1:K+1]

            # Average precision
            hits = 0
            prec_sum = 0.0
            orig_set = set(orig_rank.tolist())
            for j, idx in enumerate(embed_rank.tolist()):
                if idx in orig_set:
                    hits += 1
                    prec_sum += hits / (j + 1)
            avg_prec = prec_sum / K
            avg_precisions.append(avg_prec)

        return sum(avg_precisions) / len(avg_precisions) if avg_precisions else 0.0

    @property
    def density(self) -> float:
        """Output density (fraction of active bits): k/D."""
        return self.k / self.output_dim

    @property
    def memory_bytes_per_vector(self) -> int:
        """Memory for one embedding (block sparse is more efficient)."""
        if self.sparsify == "block":
            # Block sparse: log2(D/k) bits per block × k blocks
            bits_per_block = math.ceil(math.log2(self.output_dim // self.k))
            return math.ceil(self.k * bits_per_block / 8)
        else:
            # kWTA binary: D/8 bytes
            return math.ceil(self.output_dim / 8)

    def encoder_report(self) -> Dict:
        """Summary of encoder configuration and capacity."""
        return {
            "input_dim":       self.input_dim,
            "output_dim":      self.output_dim,
            "sparsity_k":      self.k,
            "sparsity_target": round(self.k / max(self.output_dim, 1), 4),
            "memory_bytes":    self.memory_bytes_per_vector(),
            "density":         round(self.density(), 4),
            "mode":            self.mode,
        }

    def nearest_neighbours(
        self,
        query:   torch.Tensor,   # (input_dim,) or (D,) already encoded
        library: torch.Tensor,   # (N, D) encoded embeddings
        top_k:   int = 5,
        already_encoded: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Find top-k nearest neighbours in the FlyHash space.

        Uses Jaccard similarity (appropriate for sparse embeddings).
        Much faster than Hamming for sparse codes since non-zero positions
        are a small fraction of D.

        Args:
            query:           Input query (raw or encoded)
            library:         (N, D) encoded library
            top_k:           Number of neighbours to return
            already_encoded: True if query is already in FlyHash space

        Returns:
            (indices, similarities) both shape (min(top_k, N),)
        """
        if not already_encoded:
            q_enc = self.forward(query.unsqueeze(0)).squeeze(0)
        else:
            q_enc = query.float()

        lib = library.float()
        # Jaccard via matrix ops: intersection = q @ lib^T, union = |q| + |lib| - intersection
        intersection = (q_enc.unsqueeze(0) * lib).sum(dim=1)     # (N,)
        union        = (((q_enc.unsqueeze(0) + lib) > 0).float().sum(dim=1)
                        .clamp(min=1.0))                           # (N,)
        sims = intersection / union                                # (N,)

        k    = min(top_k, lib.shape[0])
        topk = sims.topk(k)
        return topk.indices, topk.values


# ═══════════════════════════════════════════════════════════════════════════════
# 2. AdaptiveFlyHash — auto-tune k and preprocessing
# ═══════════════════════════════════════════════════════════════════════════════

class AdaptiveFlyHash:
    """
    FlyHash with automatic design choice selection.

    Sweeps over k (sparsity level) and preprocessing options, selects the
    combination that maximises MAP on a validation subset.

    Args:
        input_dim: d
        output_dim: D
        k_candidates: List of k values to try
        preprocessing_options: Preprocessing strategies to try
        seed: Random seed
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        k_candidates: Optional[List[int]] = None,
        preprocessing_options: Optional[List[str]] = None,
        seed: int = 42,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.k_candidates = k_candidates or [5, 10, 20, 50]
        self.preprocessing_options = preprocessing_options or [
            "none", "l2", "mean_center+l2"
        ]
        self.seed = seed
        self.best_encoder: Optional[FlyHashEncoder] = None
        self.best_map = 0.0
        self.best_config: Dict = {}

    def fit(self, X_val: torch.Tensor, K_map: int = 50) -> FlyHashEncoder:
        """
        Sweep design choices and select best by MAP@K.

        Args:
            X_val: (N, d) validation dataset
            K_map: K for MAP evaluation

        Returns:
            Best FlyHashEncoder
        """
        best_map = 0.0
        best_enc = None
        best_cfg = {}

        for k in self.k_candidates:
            for prep in self.preprocessing_options:
                enc = FlyHashEncoder(
                    self.input_dim, self.output_dim, k=k,
                    preprocessing=prep, seed=self.seed,
                )
                enc.fit_preprocessing(X_val)
                map_score = enc.mean_average_precision(X_val, K=K_map)

                if map_score > best_map:
                    best_map = map_score
                    best_enc = enc
                    best_cfg = {"k": k, "preprocessing": prep, "map": map_score}

        self.best_encoder = best_enc
        self.best_map = best_map
        self.best_config = best_cfg
        return best_enc


# ═══════════════════════════════════════════════════════════════════════════════
# 3. FlyHashClassifier — HDC classifier using sparse FlyHash embeddings
# ═══════════════════════════════════════════════════════════════════════════════

class FlyHashClassifier:
    """
    HDC classifier using sparse FlyHash embeddings.

    Replaces the dense binary JL projection in standard HDC classifiers
    with sparse FlyHash embeddings. Benefits:
    - Sparser prototypes → faster similarity search (AND instead of XOR+popcount)
    - Better similarity preservation → higher accuracy
    - Biologically plausible

    Uses Jaccard similarity for nearest-prototype search.
    """

    def __init__(self, encoder: FlyHashEncoder, n_classes: int):
        self.encoder = encoder
        self.n_classes = n_classes
        self._prototypes: Optional[torch.Tensor] = None
        self._accums: Optional[torch.Tensor] = None
        self._counts: Optional[torch.Tensor] = None

    def _init_accums(self):
        D = self.encoder.output_dim
        self._accums = torch.zeros(self.n_classes, D)
        self._counts = torch.zeros(self.n_classes)

    def train_step(self, x: torch.Tensor, label: int):
        """One-shot training: accumulate embeddings per class."""
        if self._accums is None:
            self._init_accums()
        emb = self.encoder(x)
        self._accums[label] += emb
        self._counts[label] += 1

    def finalize(self):
        """Binarise accumulated class prototypes."""
        counts = self._counts.clamp(min=1).unsqueeze(-1)
        self._prototypes = (self._accums / counts > 0.5).float()

    def predict(self, x: torch.Tensor) -> Tuple[int, float]:
        """Predict class via Jaccard similarity to prototypes."""
        assert self._prototypes is not None, "Call finalize() first"
        emb = self.encoder(x)

        best_class, best_sim = 0, -1.0
        for c in range(self.n_classes):
            proto = self._prototypes[c]
            intersection = (emb * proto).sum()
            union = ((emb + proto) > 0).float().sum()
            sim = float(intersection / union.clamp(min=1))
            if sim > best_sim:
                best_sim = sim
                best_class = c

        return best_class, best_sim

    def accuracy(self, X: torch.Tensor, y: torch.Tensor) -> float:
        correct = sum(
            1 for i in range(X.shape[0])
            if self.predict(X[i])[0] == int(y[i].item())
        )
        return correct / X.shape[0]

    def classifier_health(self) -> Dict:
        """
        Prototype separation and training balance.

        Requires finalize() to have been called first.
        """
        if self._prototypes is None:
            return {"status": "not_finalized"}
        C = self._prototypes.shape[0]
        sims = []
        for i in range(C):
            for j in range(i + 1, C):
                a = self._prototypes[i]
                b = self._prototypes[j]
                intersection = (a * b).sum()
                union = ((a + b) > 0).float().sum()
                sims.append(float(intersection / union.clamp(min=1)))
        mean_sim = sum(sims) / max(len(sims), 1)
        min_sim  = min(sims) if sims else 1.0
        counts = self._counts.tolist() if self._counts is not None else []
        return {
            "n_classes":      C,
            "mean_jaccard_sim": round(mean_sim, 4),
            "min_jaccard_sim":  round(min_sim, 4),
            "class_counts":   [int(c) for c in counts],
            "well_separated": mean_sim < 0.5,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_flyhash_encoder():
    print("=" * 60)
    print("Testing FlyHashEncoder (Kleyko & Rachkovskij 2025)")
    print("=" * 60)

    torch.manual_seed(42)
    d, D, k = 128, 2560, 10   # D = 20k as in paper
    enc = FlyHashEncoder(d, D, k=k, preprocessing="mean_center+l2", seed=0)

    X = torch.randn(100, d)
    enc.fit_preprocessing(X)

    # Encode a batch
    embeddings = enc(X)
    assert embeddings.shape == (100, D)
    assert embeddings.max() <= 1.0 and embeddings.min() >= 0.0

    # Density check: should be ≈ k/D
    actual_density = float(embeddings.mean())
    expected_density = k / D
    print(f"  Density: actual={actual_density:.4f}  expected={expected_density:.4f}")
    assert abs(actual_density - expected_density) < 0.01

    # Similarity preservation: similar inputs → similar embeddings
    x_base = X[0]
    x_near = x_base + torch.randn(d) * 0.1   # small perturbation
    x_far  = torch.randn(d)                    # random, unrelated

    emb_base = enc(x_base)
    emb_near = enc(x_near)
    emb_far  = enc(x_far)

    jac_near = enc.jaccard_similarity(emb_base, emb_near)
    jac_far  = enc.jaccard_similarity(emb_base, emb_far)

    print(f"  Jaccard(similar): {jac_near:.4f}  Jaccard(unrelated): {jac_far:.4f}")
    print(f"  Similarity preserved: {jac_near > jac_far}")

    # Memory comparison
    kwta_bytes  = enc.memory_bytes_per_vector
    enc_block = FlyHashEncoder(d, D, k=k, sparsify="block", seed=0)
    block_bytes = enc_block.memory_bytes_per_vector
    dense_bytes = D // 8
    print(f"  Memory: kWTA={kwta_bytes}B  block={block_bytes}B  dense-binary={dense_bytes}B")
    assert block_bytes < kwta_bytes, "Block sparse should use less memory"

    print("  ✅ FlyHashEncoder OK")


def test_flyhash_classifier():
    print("=" * 60)
    print("Testing FlyHashClassifier")
    print("=" * 60)

    torch.manual_seed(7)
    d, D, k, n_classes = 64, 1280, 8, 4

    enc = FlyHashEncoder(d, D, k=k, preprocessing="l2", seed=1)
    clf = FlyHashClassifier(enc, n_classes)

    # Gaussian cluster data
    X, y = [], []
    protos = torch.randn(n_classes, d)
    for c in range(n_classes):
        for _ in range(20):
            x = protos[c] + torch.randn(d) * 0.3
            X.append(x); y.append(c)
    X = torch.stack(X); y = torch.tensor(y)
    enc.fit_preprocessing(X)

    for i in range(X.shape[0]):
        clf.train_step(X[i], int(y[i]))
    clf.finalize()

    acc = clf.accuracy(X, y)
    print(f"  Accuracy: {acc:.1%}  (n_classes={n_classes}, D={D}, k={k})")
    assert acc > 0.6, f"Accuracy too low: {acc:.1%}"

    print("  ✅ FlyHashClassifier OK")


if __name__ == "__main__":
    test_flyhash_encoder()
    print()
    test_flyhash_classifier()
    print()
    print("=== All FlyHash tests passed ===")
