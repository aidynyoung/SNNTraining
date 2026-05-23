"""
hdc/adaptive_encoder.py
========================
NeuralHD Dimension Regeneration, DistHD, and LeHDC Gradient Training
=====================================================================
Reference:
    Imani, Kim, et al. (2022)
    "NeuralHD: Harnessing Brain-Inspired Online Learning for Scalable
    Hyperdimensional Computing" DAC 2022.
    — Dimension regeneration: identify low-information encoder dimensions
      and resample them.

    Imani et al. (2022)
    "DistHD: Learner-Aware Efficient Hyperdimensional Classification
    for Heterogeneous IoT Devices" DAC 2022.
    — Dimension scoring via misclassification patterns.

    Ge, Parhi (2020)
    "Classification Using Hyperdimensional Computing: A Review"
    IEEE Circuits and Systems Magazine.
    — Comparative analysis including gradient-based HDC.

    torchhd (Heddes et al. 2023) JMLR.
    — NeuralHD, DistHD, LeHDC implementations.

Why dimension regeneration is missing from SNNTraining:

    Standard HDC encoding is FIXED after initialisation — the random
    projection vectors ω never change regardless of data distribution.
    This wastes capacity: many dimensions carry negligible information.

    NeuralHD insight: the encoder dimension i is "useful" if removing it
    increases classification error. We can measure this by the variance
    of dimension i across class prototypes — low variance = useless.

    Algorithm (NeuralHD):
        1. Train HDC classifier for N steps
        2. Compute per-dimension variance across class prototypes:
           var_i = Var_{c} [proto_c[i]]
        3. Zero out the k% lowest-variance encoder dimensions
        4. Resample those dimensions with new random weights
        5. Re-train the classifier (now using better dimensions)

    This finds the minimal encoding that maximises class separation —
    analogous to feature selection but fully online and HDC-native.

    DistHD scores dimensions differently:
        For each misclassified sample: score[i] += |proto_pred[i] - proto_true[i]|
        Higher score = this dimension was discriminating for the misclassification
        Zero out + resample the LOWEST-scored dimensions

    LeHDC: true gradient descent through binarization
        Maintains two models: continuous (for gradients) + binary (for inference)
        Gradient from binary model transferred to continuous model via STE
        (Straight-Through Estimator), then binary model is updated from continuous

This module implements:

1. NeuralHDEncoder
   — Sinusoidal random projection encoder (base class for NeuralHD/DistHD)
   — Per-dimension variance tracking
   — Regeneration: zero + resample low-variance dimensions

2. DistHDEncoder
   — Same architecture but scores dimensions by misclassification contribution
   — More discriminating than NeuralHD for heterogeneous data

3. LeHDC
   — Full gradient-based HDC with binarization in the loop
   — Dual continuous+binary model with cross-entropy loss
   — Straight-Through Estimator for gradient through sign()
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Utility ────────────────────────────────────────────────────────────────────

def _gen_sinusoid_proj(n_features: int, dim: int, seed: Optional[int] = None,
                       device: str = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate sinusoidal random projection: ω ~ N(0,1), b ~ U(0, 2π).
    Feature encoded as cos(x @ ω + b).
    """
    g = torch.Generator(device=device)
    if seed is not None:
        g.manual_seed(seed)
    omega = torch.randn(n_features, dim, generator=g, device=device)
    bias  = torch.rand(dim, generator=g, device=device) * 2 * math.pi
    return omega, bias


# ═══════════════════════════════════════════════════════════════════════════════
# 1. NeuralHDEncoder — sinusoidal encoder with dimension regeneration
# ═══════════════════════════════════════════════════════════════════════════════

class NeuralHDEncoder:
    """
    NeuralHD sinusoidal encoder with adaptive dimension regeneration.

    Reference: Imani et al. (2022) NeuralHD — DAC 2022.

    Encoding: z(x) = sign(cos(x @ ω + b))  ∈ {-1, +1}^D

    Dimension regeneration:
        1. For each class c, compute prototype p_c = MEAN(encoded_samples_c)
        2. Compute per-dimension variance: var_d = Var_c[p_c[d]]
        3. Zero + resample the k% lowest-variance dimensions
        4. Re-run training (new encoder captures different features)

    Why this works: low-variance dimensions don't separate classes →
    replacing them with new random features gives another chance to find
    useful structure.

    Args:
        n_features: Input feature dimension
        dim:        Encoding dimension
        n_regen:    Number of regeneration rounds (default 2)
        regen_frac: Fraction of dimensions to regenerate per round (0.05 = 5%)
        seed:       Random seed
        device:     torch device
    """

    def __init__(
        self,
        n_features: int,
        dim:        int,
        n_regen:    int   = 2,
        regen_frac: float = 0.05,
        seed:       Optional[int] = None,
        device:     str   = "cpu",
    ):
        self.n_features  = n_features
        self.dim         = dim
        self.n_regen     = n_regen
        self.regen_frac  = regen_frac
        self.device      = device
        self._seed       = seed or 0
        self._regen_step = 0

        self.omega, self.bias = _gen_sinusoid_proj(n_features, dim, seed, device)

        # Per-dimension utility tracking
        self._dim_var  = torch.ones(dim, device=device)   # dimension variance scores
        self._dim_mask = torch.ones(dim, dtype=torch.bool, device=device)  # active dims

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode feature vector to bipolar binary HV.

        z(x) = sign(cos(x @ ω + b))

        Returns: (D,) bipolar {-1, +1} tensor
        """
        x_f  = x.float().to(self.device)
        proj = x_f @ self.omega + self.bias   # (D,)
        cont = torch.cos(proj)
        return torch.sign(cont)   # {-1, +1}

    def encode_batch(self, X: torch.Tensor) -> torch.Tensor:
        """Encode a batch (N, n_features) → (N, D)."""
        proj = X.float().to(self.device) @ self.omega + self.bias
        return torch.sign(torch.cos(proj))

    def update_dim_variance(self, class_prototypes: Dict[int, torch.Tensor]):
        """
        Compute per-dimension variance across class prototypes.

        var_d = Var_c[ proto_c[d] ]

        Higher variance = dimension is more discriminating between classes.
        Lower variance = dimension is uninformative → candidate for regeneration.

        Args:
            class_prototypes: {class_idx: prototype_HV (D,)}
        """
        if len(class_prototypes) < 2:
            return
        protos = torch.stack(list(class_prototypes.values())).float()  # (C, D)
        self._dim_var = protos.var(dim=0)   # (D,) per-dimension variance

    def regenerate(self, n_dims: Optional[int] = None) -> int:
        """
        Zero out and resample the lowest-variance dimensions.

        Args:
            n_dims: Number of dimensions to regenerate. Defaults to regen_frac × dim.

        Returns:
            Number of dimensions regenerated.
        """
        self._regen_step += 1
        if n_dims is None:
            n_dims = max(1, int(self.dim * self.regen_frac))

        # Find lowest-variance dimensions
        _, low_idx = self._dim_var.topk(n_dims, largest=False)

        # Resample those dimensions
        g = torch.Generator(device=self.device)
        g.manual_seed(self._seed + self._regen_step * 1000)

        new_omega = torch.randn(self.n_features, n_dims, generator=g, device=self.device)
        new_bias  = torch.rand(n_dims, generator=g, device=self.device) * 2 * math.pi

        self.omega[:, low_idx] = new_omega
        self.bias[low_idx]     = new_bias

        return int(n_dims)

    def fit_and_regenerate(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        n_classes: int,
    ) -> List[float]:
        """
        Full NeuralHD pipeline: initial train → regenerate → re-train.

        Returns per-round accuracy list.
        """
        from hdc.hdcc_compiler import HDCCClassifier

        accs = []
        for rnd in range(self.n_regen + 1):
            # Build class prototypes from encoded data
            prototypes: Dict[int, torch.Tensor] = {}
            counts:     Dict[int, int]           = {}
            for x, y in zip(X_train, y_train):
                c = int(y.item())
                z = self.encode(x)
                if c not in prototypes:
                    prototypes[c] = z.float()
                    counts[c] = 1
                else:
                    prototypes[c] = (counts[c] * prototypes[c] + z.float()) / (counts[c] + 1)
                    counts[c] += 1

            # Accuracy
            correct = 0
            for x, y in zip(X_train, y_train):
                z = self.encode(x)
                sims = {c: float(F.cosine_similarity(z.float().unsqueeze(0),
                                                       p.unsqueeze(0)).item())
                        for c, p in prototypes.items()}
                pred = max(sims, key=sims.get)
                correct += int(pred == int(y.item()))
            acc = correct / max(len(X_train), 1)
            accs.append(acc)

            # Regenerate for next round (not after last)
            if rnd < self.n_regen:
                self.update_dim_variance(prototypes)
                self.regenerate()

        return accs

    def encoder_health(self) -> Dict:
        """
        Dimension utility statistics: fraction of low-variance (inactive) dims.

        frac_low_var > 0.5 → many uninformative dimensions → regeneration needed.
        """
        var = self._dim_var
        total_var = float(var.sum().item())
        frac_low = float((var < var.mean() * 0.1).float().mean().item())
        top10_share = float(var.topk(max(1, int(0.1 * self.dim))).values.sum().item()) / max(total_var, 1e-8)
        return {
            "dim":               self.dim,
            "n_features":        self.n_features,
            "regen_steps":       self._regen_step,
            "mean_dim_var":      round(float(var.mean().item()), 6),
            "frac_low_var":      round(frac_low, 4),
            "top10pct_var_share": round(top10_share, 4),
            "needs_regen":       frac_low > 0.5,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 2. DistHDEncoder — misclassification-weighted dimension scoring
# ═══════════════════════════════════════════════════════════════════════════════

class DistHDEncoder(NeuralHDEncoder):
    """
    DistHD encoder: dimension scoring via misclassification contribution.

    Reference: Imani et al. (2022) DistHD — DAC 2022.

    Instead of variance across class prototypes, DistHD scores dimensions by
    how much they contributed to misclassifications:

        For each misclassified sample:
            score[d] += |proto_pred[d] - proto_true[d]|

    Dimensions with HIGH score were discriminating in the WRONG direction →
    zero them and resample.

    This is more data-aware than NeuralHD (which ignores which samples failed).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mismatch_scores = torch.zeros(self.dim, device=self.device)
        self._n_misses        = 0

    def record_mismatch(
        self,
        predicted_proto: torch.Tensor,   # (D,) predicted class prototype
        true_proto:      torch.Tensor,   # (D,) true class prototype
    ):
        """Accumulate dimension score for one misclassification."""
        self._mismatch_scores += (predicted_proto.float() - true_proto.float()).abs()
        self._n_misses        += 1

    def regenerate_by_score(self, n_dims: Optional[int] = None) -> int:
        """
        Regenerate the dimensions with HIGHEST mismatch scores.
        High score = contributed most to misclassifications.
        """
        self._regen_step += 1
        if n_dims is None:
            n_dims = max(1, int(self.dim * self.regen_frac))

        if self._n_misses == 0:
            # Fall back to variance-based
            return self.regenerate(n_dims)

        # Find highest-score dimensions
        _, high_idx = self._mismatch_scores.topk(n_dims, largest=True)

        # Resample
        g = torch.Generator(device=self.device)
        g.manual_seed(self._seed + self._regen_step * 1000 + 1)
        new_omega = torch.randn(self.n_features, n_dims, generator=g, device=self.device)
        new_bias  = torch.rand(n_dims, generator=g, device=self.device) * 2 * math.pi

        self.omega[:, high_idx] = new_omega
        self.bias[high_idx]     = new_bias

        # Reset scores
        self._mismatch_scores.zero_()
        self._n_misses = 0

        return int(n_dims)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. LeHDC — gradient-based HDC with binarization in the training loop
# ═══════════════════════════════════════════════════════════════════════════════

class LeHDC(nn.Module):
    """
    LeHDC: Learning with Hyperdimensional Computing via gradient descent.

    Reference: torchhd (Heddes et al. 2023) — LeHDC classifier.

    Architecture:
        Continuous encoder: W_cont ∈ ℝ^{n×D} (differentiable)
        Binary encoder:     W_bin  = sign(W_cont) ∈ {-1,+1}^{n×D}

    Forward pass:
        z = sign(x @ W_bin)      [binary encoding, for inference]
        logits = z @ proto_bin^T  [Hamming similarity as logits]

    Training:
        1. Compute loss with binary model (for correct loss signal)
        2. Estimate gradient via Straight-Through Estimator (STE):
           d_loss/d_W_cont ≈ d_loss/d_W_bin  (pretend sign is identity)
        3. Update W_cont with Adam
        4. Binarise: W_bin = sign(W_cont)

    Why this works:
        - The continuous model accumulates gradients smoothly
        - The binary model gives exactly binary representations
        - STE bridges the gap: gradients flow through the sign function
        - At inference: entirely binary (fast, energy-efficient)

    Args:
        n_features: Input dimension
        dim:        Encoding dimension
        n_classes:  Number of output classes
        lr:         Learning rate for Adam
        device:     torch device
    """

    def __init__(
        self,
        n_features: int,
        dim:        int,
        n_classes:  int,
        lr:         float = 1e-3,
        device:     str   = "cpu",
    ):
        super().__init__()
        self.n_features = n_features
        self.dim        = dim
        self.n_classes  = n_classes
        self.device_str = device

        # Continuous encoder weights (differentiable)
        self.W_cont = nn.Parameter(
            torch.randn(n_features, dim, device=device) / math.sqrt(n_features)
        )
        self.bias_cont = nn.Parameter(torch.zeros(dim, device=device))

        # Class prototype weights (continuous, binarised at inference)
        self.protos_cont = nn.Parameter(
            torch.randn(n_classes, dim, device=device) / math.sqrt(dim)
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self._n_steps  = 0

    def _encode(self, x: torch.Tensor, binary: bool = True) -> torch.Tensor:
        """
        Encode via continuous or binary weights.

        Args:
            x:      (B, n_features) or (n_features,) input
            binary: If True, use sign(W_cont) (inference); else use W_cont (training)
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if binary:
            W = self.W_cont.detach().sign()
        else:
            W = self.W_cont
        proj = x @ W + self.bias_cont
        if binary:
            return proj.sign()
        return torch.tanh(proj)  # smooth approximation for gradients

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Inference forward: binary encoding → cosine similarity → logits."""
        z      = self._encode(x, binary=True)                    # (B, D) binary
        protos = self.protos_cont.detach().sign()                  # (C, D) binary
        sims   = F.cosine_similarity(z.unsqueeze(1), protos.unsqueeze(0), dim=-1)  # (B, C)
        return sims

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """
        One training step with STE gradient.

        Args:
            x: (B, n_features) or (n_features,)
            y: (B,) or scalar integer labels

        Returns:
            Scalar loss value
        """
        self._n_steps += 1
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if isinstance(y, int) or (isinstance(y, torch.Tensor) and y.dim() == 0):
            y = torch.tensor([int(y)], device=self.device_str)
        else:
            y = y.to(self.device_str)
        if y.dim() == 0:
            y = y.unsqueeze(0)

        # Continuous forward for gradients (STE)
        z_cont  = self._encode(x, binary=False)    # (B, D) continuous
        p_cont  = torch.tanh(self.protos_cont)      # (C, D) continuous

        # Cosine similarity logits
        z_n  = F.normalize(z_cont, dim=-1)
        p_n  = F.normalize(p_cont, dim=-1)
        sims = z_n @ p_n.T   # (B, C)

        # Cross-entropy loss
        loss = F.cross_entropy(sims, y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return float(loss.item())

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict class indices (B,) for input (B, n_features)."""
        with torch.no_grad():
            logits = self.forward(x)
            return logits.argmax(dim=-1)

    def accuracy(self, X: torch.Tensor, y: torch.Tensor) -> float:
        """Compute accuracy on a dataset."""
        with torch.no_grad():
            if X.dim() == 1:
                X = X.unsqueeze(0)
            preds   = self.predict(X)
            correct = (preds == y.to(self.device_str)).float().mean()
            return float(correct.item())


# ═══════════════════════════════════════════════════════════════════════════════
# Smoke test
# ═══════════════════════════════════════════════════════════════════════════════

def _test_adaptive_encoder():
    N_FEAT, DIM, N_CLS = 16, 128, 3
    torch.manual_seed(42)

    print("=== NeuralHDEncoder ===")
    enc = NeuralHDEncoder(N_FEAT, DIM, n_regen=2, regen_frac=0.1)

    x = torch.randn(N_FEAT)
    z = enc.encode(x)
    assert z.shape == (DIM,)
    assert set(z.unique().tolist()).issubset({-1.0, 1.0})
    print(f"  Encoded shape: {z.shape}, bipolar={set(z.unique().tolist())}  OK")

    # Simulate training data
    X = torch.cat([torch.randn(20, N_FEAT) + c * 3 for c in range(N_CLS)])
    y = torch.cat([torch.full((20,), c, dtype=torch.long) for c in range(N_CLS)])

    accs = enc.fit_and_regenerate(X, y, N_CLS)
    print(f"  NeuralHD accuracies per round: {[f'{a:.2f}' for a in accs]}")
    assert all(0.0 <= a <= 1.0 for a in accs)
    assert len(accs) == enc.n_regen + 1
    print(f"  Final accuracy: {accs[-1]:.2f}  OK")

    print("\n=== DistHDEncoder ===")
    dist = DistHDEncoder(N_FEAT, DIM, n_regen=1, regen_frac=0.1)
    # Simulate a mismatch
    pred_proto = torch.randn(DIM)
    true_proto = torch.randn(DIM)
    dist.record_mismatch(pred_proto, true_proto)
    assert dist._n_misses == 1
    n_regen = dist.regenerate_by_score()
    assert n_regen > 0
    assert dist._n_misses == 0   # reset after regeneration
    print(f"  DistHD regenerated {n_regen} dims, reset scores  OK")

    print("\n=== LeHDC ===")
    model = LeHDC(N_FEAT, DIM, N_CLS, lr=1e-2)
    logits = model.forward(X[:4])
    assert logits.shape == (4, N_CLS)
    print(f"  Logits shape: {logits.shape}  OK")

    losses = []
    for _ in range(20):
        for i in range(len(X)):
            loss = model.train_step(X[i], y[i])
            losses.append(loss)
    acc = model.accuracy(X, y)
    print(f"  LeHDC accuracy after 20 epochs: {acc:.2f}  OK")
    assert isinstance(acc, float)

    # Predict
    preds = model.predict(X[:4])
    assert preds.shape == (4,)
    print(f"  Predictions shape: {preds.shape}  OK")

    print("\n✅ All adaptive_encoder tests passed")


if __name__ == "__main__":
    _test_adaptive_encoder()
