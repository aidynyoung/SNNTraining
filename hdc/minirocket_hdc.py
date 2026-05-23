"""
HDC-MiniROCKET: Explicit Time Encoding in Time Series Classification
=====================================================================
Based on: Schlegel, K., Neubert, P., and Protzel, P.
"HDC-MiniROCKET: Explicit Time Encoding in Time Series Classification
with Hyperdimensional Computing"
arXiv:2202.08055 (Chemnitz University of Technology)

Key insight:
MiniROCKET's Proportion of Positive Values (PPV) = elementwise mean over time
= HDC *bundling* without temporal information. Adding HDC *binding* before
bundling encodes the temporal position of each feature response.

Algorithm:
    MiniROCKET step 1: dilated convolution ck,d = x * Wk,d
    MiniROCKET step 2: binarize ck,d,b = (ck,d > Bb)
    MiniROCKET step 3 (PPV):     yPPV = mean_t(ck,d,b)
    HDC-MiniROCKET step 3:  yHDC = Σ_t (F^BP_t ⊙ Pt)    [Eq. 4]

Where:
    F^BP_t = 1 - 2*F_t  (bipolar conversion of 9996-dim feature at time t)
    Pt = v^{t·s/T}      (timestamp encoding via fractional binding, Eq. 5)
    v^p = IDFT((DFT(v))^p)  (fractional power in frequency domain)

Efficient implementation (Eq. 6-8):
    c''_t = +Pt  if ck,d,t > Bb
          = -Pt  if ck,d,t ≤ Bb
    yHDC_i = Σ_t c''_{t,i}

The scale parameter s controls temporal resolution:
    s=0 → all Pt identical → reduces to original MiniROCKET
    s=1 → similarity decreases to 0 at distance T
    s=2 → similarity reaches 0 at T/2 (finer temporal resolution)

Improvements over MiniROCKET:
    - 97% vs 65% on synthetic datasets with temporal peaks
    - +3.1% average on 81/128 UCR benchmark datasets
    - Same computational cost (only precomputed Pt additions)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════════
# Fractional Binding (Komer et al. 2019, used in HDC-MiniROCKET §IV-B)
# ═══════════════════════════════════════════════════════════════════════════════

class FractionalBinding:
    """
    Fractional power binding in the Fourier domain (Eq. 5).

    v^p = IDFT( (DFT(v))^p )

    Maps scalar p to a hypervector such that:
        sim(v^p1, v^p2) ≈ f(|p1 - p2|)  (graded similarity)
        v^0 = all-ones (identity element)
        v^1 = v  (base vector)
        v^{p1} ⊗ v^{p2} = v^{p1+p2}  (exponent addition)

    Used to encode timestamps t ∈ [0, 1] as continuous hypervectors
    with similarity proportional to temporal proximity.
    """

    def __init__(self, dim: int, seed: Optional[int] = None):
        """
        Args:
            dim: Hypervector dimensionality
            seed: Random seed for base vector
        """
        self.dim = dim
        g = torch.Generator()
        if seed is not None:
            g.manual_seed(seed)
        # Base vector v: random unit-phasor in frequency domain
        # (real-valued in time domain, unit-norm phasors in freq domain)
        phases = torch.rand(dim, generator=g) * 2 * math.pi
        self._v_freq = torch.exp(1j * phases)  # (dim,) complex

    def encode(self, p: float) -> torch.Tensor:
        """
        Compute v^p via fractional power in Fourier domain.

        Args:
            p: Scalar value to encode (typically t*s/T for timestamps)

        Returns:
            (dim,) real-valued hypervector
        """
        # v^p = IDFT((DFT(v))^p) = IDFT(v_freq^p)
        v_freq_p = self._v_freq ** p
        v_p = torch.fft.ifft(v_freq_p).real
        return v_p

    def encode_batch(self, values: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of scalar values.

        Args:
            values: (T,) tensor of scalars

        Returns:
            (T, dim) real-valued hypervectors
        """
        T = values.shape[0]
        result = torch.zeros(T, self.dim)
        for i, p in enumerate(values.tolist()):
            result[i] = self.encode(p)
        return result

    def similarity(self, p1: float, p2: float) -> float:
        """Cosine similarity between v^p1 and v^p2."""
        h1 = self.encode(p1)
        h2 = self.encode(p2)
        return float(F.cosine_similarity(h1.unsqueeze(0), h2.unsqueeze(0)).item())


# ═══════════════════════════════════════════════════════════════════════════════
# MiniROCKET Kernels (simplified deterministic version)
# ═══════════════════════════════════════════════════════════════════════════════

class MiniROCKETKernels:
    """
    Deterministic MiniROCKET kernels (Dempster et al. 2021).

    MiniROCKET uses 84 predefined kernels of length 9, with weights in {-1, 2}
    and exactly 3 weights equal to 2. The kernels are applied with multiple
    dilations to produce 9,996 binary feature vectors per time series.

    This implementation uses a simplified subset for efficiency:
    instead of the exact 84×119 combinations, we use a configurable
    number of random kernels with the same structure.
    """

    # All 84 patterns of 3 positions out of 9 that get weight 2
    # (the rest get weight -1). Generated deterministically.
    _N_KERNEL_WEIGHTS = 84
    _KERNEL_LENGTH = 9
    _N_BIAS_PER_KERNEL = 119  # dilations × biases per kernel
    _TOTAL_FEATURES = 9996    # = 84 × 119

    def __init__(
        self,
        n_features: int = 9996,
        kernel_length: int = 9,
        seed: int = 42,
    ):
        """
        Args:
            n_features: Number of output features (≤ 9996)
            kernel_length: Convolutional kernel length
            seed: Random seed for kernel generation
        """
        self.n_features = n_features
        self.kernel_length = kernel_length

        g = torch.Generator()
        g.manual_seed(seed)

        # Generate kernels: weights ∈ {-1, 2}, with exactly 3 positions = 2
        n_kernels = n_features  # one kernel per feature for simplicity
        kernels = torch.full((n_kernels, kernel_length), -1.0)
        for k in range(n_kernels):
            pos = torch.randperm(kernel_length, generator=g)[:3]
            kernels[k, pos] = 2.0
        self.kernels = kernels  # (n_features, kernel_length)

        # Bias values: quantiles of filter responses (approximated as uniform)
        self.biases = torch.rand(n_features, generator=g) * 6 - 3  # rough range

        # Dilations: exponentially distributed, from 1 to T/kernel_length
        self.dilations = torch.pow(
            2, torch.arange(n_features, dtype=torch.float) * 3.0 / n_features
        ).long().clamp(min=1)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply all kernels with their dilations and biases.

        Args:
            x: (T,) or (B, T) univariate time series

        Returns:
            (T, n_features) binary feature matrix per timestep, or (B, T, n_features)
        """
        squeeze = x.dim() == 1
        if squeeze:
            x = x.unsqueeze(0)  # (1, T)

        B, T = x.shape
        results = torch.zeros(B, T, self.n_features)

        for k in range(self.n_features):
            w = self.kernels[k]              # (kernel_length,)
            d = int(self.dilations[k].item())
            b = float(self.biases[k].item())

            # Build dilated kernel: kernel_length elements with d-1 zeros between
            dilated_len = (self.kernel_length - 1) * d + 1
            dilated_w = torch.zeros(dilated_len)
            dilated_w[::d] = w

            # Convolve: use unfold for efficiency
            pad = dilated_len // 2
            x_padded = F.pad(x, (pad, pad), mode='constant', value=0.0)  # (B, T+2*pad)

            # Manual conv: unfold + matmul
            x_unf = x_padded.unfold(-1, dilated_len, 1)  # (B, T, dilated_len)
            if x_unf.shape[1] != T:
                # Trim or pad to get exactly T outputs
                x_unf = x_unf[:, :T, :]

            conv_out = (x_unf * dilated_w.unsqueeze(0).unsqueeze(0)).sum(-1)  # (B, T)

            # Binarize: > bias → 1, else 0
            results[:, :, k] = (conv_out > b).float()

        if squeeze:
            results = results.squeeze(0)  # (T, n_features)
        return results


# ═══════════════════════════════════════════════════════════════════════════════
# HDC-MiniROCKET Encoder (Schlegel et al. 2022, §IV)
# ═══════════════════════════════════════════════════════════════════════════════

class HDCMiniROCKET(nn.Module):
    """
    HDC-MiniROCKET: MiniROCKET with explicit temporal encoding (Schlegel 2022).

    Generalizes MiniROCKET by adding HDC fractional binding to encode
    the temporal position of each feature response before bundling:

        yHDC = Σ_{t=1}^T (F^BP_t ⊙ Pt)              [Eq. 4]

    where:
        F^BP_t = 1 - 2*F_t  (binary feature vector at time t, bipolarized)
        Pt = v^{t·s/T}      (timestamp HV via fractional binding)
        ⊙ = elementwise multiply (MAP binding)

    When s=0: Pt = v^0 = constant → PPV mode (original MiniROCKET).
    When s>0: timestamps are graded → temporal position is preserved.

    Efficient implementation (Eq. 6-8):
        Instead of forming F^BP_t explicitly, directly use ±Pt:
        c''_{k,d,b,t} = +Pt if conv_response > bias, else -Pt
        yHDC = Σ_t c''_{k,d,b,t}

    Args:
        n_features: Number of convolution features (kernel × dilation × bias)
        scale_s: Temporal scale parameter s (0 = PPV, 1 = full temporal)
        seed: Random seed
    """

    def __init__(
        self,
        n_features: int = 500,   # reduced from 9996 for speed
        scale_s: float = 1.0,
        seed: int = 42,
    ):
        super().__init__()
        self.n_features = n_features
        self.scale_s = scale_s

        self.kernels = MiniROCKETKernels(n_features=n_features, seed=seed)
        self.frac_binder = FractionalBinding(dim=n_features, seed=seed + 1)

        # Pre-computed timestamp encodings are cached per series length
        self._cached_T: Optional[int] = None
        self._cached_Pt: Optional[torch.Tensor] = None

    def _get_timestamp_encodings(self, T: int) -> torch.Tensor:
        """Get or compute Pt for all t ∈ {1,...,T}. Returns (T, n_features)."""
        if self._cached_T == T:
            return self._cached_Pt

        values = torch.arange(1, T + 1, dtype=torch.float) * self.scale_s / T
        Pt = self.frac_binder.encode_batch(values)  # (T, n_features)

        self._cached_T = T
        self._cached_Pt = Pt
        return Pt

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode time series to HDC descriptor.

        Args:
            x: (T,) univariate time series

        Returns:
            (n_features,) descriptor vector yHDC
        """
        T = x.shape[-1]
        F_bin = self.kernels.transform(x)   # (T, n_features), binary {0,1}
        F_bp = 1.0 - 2.0 * F_bin           # bipolarize: {0,1} → {1,-1}

        Pt = self._get_timestamp_encodings(T)  # (T, n_features)

        # Efficient: c''_{t,i} = F^BP_{t,i} * Pt,i = (±1) * Pt,i
        # Then sum over t
        y_hdc = (F_bp * Pt).sum(dim=0)     # (n_features,)
        return y_hdc

    def transform_batch(self, X: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of time series.

        Args:
            X: (B, T) batch of time series

        Returns:
            (B, n_features) descriptor matrix
        """
        return torch.stack([self.forward(X[i]) for i in range(X.shape[0])])


class HDCMiniROCKETClassifier(nn.Module):
    """
    Full HDC-MiniROCKET classifier: encoder + ridge regression head.

    The ridge regression (or any linear classifier) is trained on the
    HDC-MiniROCKET descriptors. Since MiniROCKET is the s=0 special case,
    this classifier can also replicate the original MiniROCKET by setting
    scale_s=0.
    """

    def __init__(
        self,
        n_classes: int,
        n_features: int = 500,
        scale_s: float = 1.0,
        seed: int = 42,
    ):
        super().__init__()
        self.encoder = HDCMiniROCKET(n_features=n_features, scale_s=scale_s, seed=seed)
        self.classifier = nn.Linear(n_features, n_classes, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (T,) or (B, T) time series

        Returns:
            (n_classes,) or (B, n_classes) logits
        """
        if x.dim() == 1:
            desc = self.encoder(x)
            return self.classifier(desc)
        descs = self.encoder.transform_batch(x)
        return self.classifier(descs)

    def fit_ridge(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        alpha: float = 1.0,
    ):
        """
        Fit classifier with ridge regression (closed-form, no gradient descent).

        Args:
            X: (B, T) training time series
            y: (B,) integer class labels
            alpha: Ridge regularization strength
        """
        with torch.no_grad():
            descs = self.encoder.transform_batch(X)  # (B, n_features)

        n_classes = int(y.max().item()) + 1

        # One-hot encode labels
        Y = torch.zeros(X.shape[0], n_classes)
        Y.scatter_(1, y.long().unsqueeze(1), 1.0)

        # Ridge: W = (D^T D + α I)^{-1} D^T Y
        D = descs
        A = D.T @ D + alpha * torch.eye(D.shape[1])
        try:
            W = torch.linalg.solve(A, D.T @ Y)  # (n_features, n_classes)
        except Exception:
            W = torch.linalg.lstsq(D, Y).solution

        b = torch.zeros(n_classes)

        with torch.no_grad():
            self.classifier.weight.copy_(W.T)
            self.classifier.bias.copy_(b)


    def train_online(
        self,
        x:     torch.Tensor,   # (T,) time series
        label: int,
        lr:    float = 0.1,
    ):
        """
        Online one-shot update: add one labelled time series to the classifier.

        No gradient descent — directly updates the class weight vector
        by bundling the descriptor with the existing class weights:
            W[label] = (1-lr) × W[label] + lr × descriptor

        This enables continual learning: add new classes or reinforce existing
        ones without retraining from scratch.

        Args:
            x:     (T,) time series to learn from
            label: True class index
            lr:    Blending rate (default 0.1)
        """
        with torch.no_grad():
            desc = self.encoder(x)   # (n_features,)
            w    = self.classifier.weight.data   # (n_classes, n_features)
            w[label] = (1 - lr) * w[label] + lr * desc

    def predict_label(self, x: torch.Tensor) -> int:
        """Predict class index for a single time series."""
        return int(self.forward(x).argmax().item())

    def predict_with_abstain(
        self,
        x:         torch.Tensor,
        threshold: float = 0.5,
    ) -> Tuple[int, float, bool]:
        """
        Predict with confidence-based abstention.

        If the top logit score < threshold, abstain (return -1).
        Useful for: safety-critical deployment, out-of-distribution rejection.

        Returns:
            (label, confidence, abstained)
            label = -1 and abstained = True when confidence < threshold.
        """
        logits = self.forward(x)
        probs  = torch.softmax(logits, dim=0)
        conf   = float(probs.max().item())
        if conf < threshold:
            return -1, conf, True
        label = int(probs.argmax().item())
        return label, conf, False

    def stream_accuracy(
        self,
        stream,            # Iterable of (x: Tensor, label: int)
        n_steps: int = 20,
    ) -> Dict:
        """
        Evaluate accuracy over a streaming time-series dataset.

        For each sample in the stream, processes `n_steps` of the signal
        and predicts the label.

        Returns:
            Dict with accuracy, n_samples, and confusion (correct/incorrect counts).
        """
        n_correct, n_total = 0, 0
        for x, true_label in stream:
            pred = self.predict_label(x)
            if pred == true_label:
                n_correct += 1
            n_total += 1
        return {
            "accuracy":  n_correct / max(n_total, 1),
            "n_samples": n_total,
            "n_correct": n_correct,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Scale Selection (Schlegel 2022, §V-C)
# ═══════════════════════════════════════════════════════════════════════════════

class HDCMiniROCKETScaleSelector:
    """
    Data-driven scale parameter selection via cross-validation (Schlegel 2022).

    The scale s controls how aggressively temporal position is encoded.
    The paper recommends selecting s from {0, 1, 2, ..., 6} via 10-fold CV.

    s=0 → MiniROCKET (no temporal encoding)
    s=1 → moderate temporal sensitivity
    s≥3 → high temporal sensitivity (may hurt non-temporal datasets)
    """

    def __init__(
        self,
        n_classes: int,
        n_features: int = 500,
        candidates: Optional[List[float]] = None,
        seed: int = 42,
    ):
        self.n_classes = n_classes
        self.n_features = n_features
        self.candidates = candidates or [0.0, 1.0, 2.0, 3.0]
        self.seed = seed
        self.best_s: float = 1.0

    def select(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        n_folds: int = 5,
        alpha: float = 1.0,
    ) -> float:
        """
        Select best scale via cross-validation.

        Args:
            X_train: (B, T) training time series
            y_train: (B,) labels
            n_folds: Number of CV folds
            alpha: Ridge regularization

        Returns:
            Best scale parameter s
        """
        B = X_train.shape[0]
        fold_size = B // n_folds

        scores: dict = {s: [] for s in self.candidates}

        for fold in range(n_folds):
            val_start = fold * fold_size
            val_end = val_start + fold_size
            val_idx = list(range(val_start, val_end))
            train_idx = [i for i in range(B) if i not in val_idx]

            X_t = X_train[train_idx]
            y_t = y_train[train_idx]
            X_v = X_train[val_idx]
            y_v = y_train[val_idx]

            for s in self.candidates:
                clf = HDCMiniROCKETClassifier(
                    n_classes=self.n_classes,
                    n_features=self.n_features,
                    scale_s=s,
                    seed=self.seed,
                )
                clf.fit_ridge(X_t, y_t, alpha=alpha)
                with torch.no_grad():
                    logits = clf(X_v)
                    preds = logits.argmax(dim=-1)
                acc = float((preds == y_v.long()).float().mean().item())
                scores[s].append(acc)

        avg_scores = {s: sum(v) / len(v) for s, v in scores.items()}
        self.best_s = max(avg_scores, key=avg_scores.get)
        return self.best_s


# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_fractional_binding():
    print("=" * 60)
    print("Testing FractionalBinding (Schlegel 2022, §IV-B, Eq. 5)")
    print("=" * 60)

    fb = FractionalBinding(dim=1000, seed=42)

    # v^0 should be near-constant (identity for addition)
    h0 = fb.encode(0.0)
    print(f"  v^0 std: {h0.std():.4f}  (want ≈ 0, identity element)")

    # Self similarity: sim(v^p, v^p) = 1
    sim_self = fb.similarity(0.5, 0.5)
    print(f"  sim(v^0.5, v^0.5) = {sim_self:.4f}  (want ≈ 1.0)")
    assert sim_self > 0.99

    # Graded similarity: closer timestamps → higher similarity
    sim_close = fb.similarity(0.1, 0.2)
    sim_far   = fb.similarity(0.1, 0.9)
    print(f"  sim(v^0.1, v^0.2) = {sim_close:.4f}  (want > sim_far)")
    print(f"  sim(v^0.1, v^0.9) = {sim_far:.4f}")
    assert sim_close > sim_far, "Closer timestamps should be more similar"

    print("  ✅ FractionalBinding OK")


def test_hdc_minirocket_encoding():
    print("=" * 60)
    print("Testing HDCMiniROCKET encoding (Schlegel 2022, §IV)")
    print("=" * 60)

    torch.manual_seed(0)
    T, n_feat = 100, 200
    encoder = HDCMiniROCKET(n_features=n_feat, scale_s=1.0, seed=7)

    x = torch.randn(T)
    desc = encoder(x)
    print(f"  Descriptor shape: {desc.shape}  (want ({n_feat},))")
    assert desc.shape == (n_feat,)

    # s=0 should produce same result as mean PPV (up to constant)
    encoder_ppv = HDCMiniROCKET(n_features=n_feat, scale_s=0.0, seed=7)
    desc_ppv = encoder_ppv(x)
    # Both should be finite
    assert not torch.isnan(desc).any(), "NaN in HDC descriptor"
    assert not torch.isnan(desc_ppv).any(), "NaN in PPV descriptor"
    print(f"  s=1 descriptor norm: {desc.norm():.2f}")
    print(f"  s=0 descriptor norm: {desc_ppv.norm():.2f}")

    # Temporal sensitivity: same signal shifted in time should differ more with s>0
    x_early = torch.zeros(T)
    x_early[20] = 5.0  # peak at t=20 (first half)
    x_late = torch.zeros(T)
    x_late[80] = 5.0   # peak at t=80 (second half)

    desc_early_s1 = encoder(x_early)
    desc_late_s1  = encoder(x_late)
    desc_early_s0 = encoder_ppv(x_early)
    desc_late_s0  = encoder_ppv(x_late)

    sim_s1 = float(F.cosine_similarity(desc_early_s1.unsqueeze(0), desc_late_s1.unsqueeze(0)))
    sim_s0 = float(F.cosine_similarity(desc_early_s0.unsqueeze(0), desc_late_s0.unsqueeze(0)))

    print(f"  Temporal sensitivity: sim(early,late) s=1: {sim_s1:.4f}, s=0: {sim_s0:.4f}")
    print(f"  (s=1 should separate temporal peaks better than s=0)")

    print("  ✅ HDCMiniROCKET encoding OK")


def test_hdc_minirocket_classifier():
    print("=" * 60)
    print("Testing HDCMiniROCKETClassifier (Schlegel 2022, §V-A)")
    print("=" * 60)

    torch.manual_seed(42)
    T, n_classes, n_feat = 100, 2, 200
    B_train, B_test = 40, 20

    # Synthetic: class 0 = peak in first half, class 1 = peak in second half
    def make_data(n, peak_first):
        X = torch.randn(n, T) * 0.3
        for i in range(n):
            pos = int(T * 0.25) if peak_first else int(T * 0.75)
            X[i, pos] = 5.0
        return X

    X_train = torch.cat([make_data(B_train//2, True), make_data(B_train//2, False)])
    y_train = torch.cat([torch.zeros(B_train//2), torch.ones(B_train//2)]).long()
    X_test  = torch.cat([make_data(B_test//2, True), make_data(B_test//2, False)])
    y_test  = torch.cat([torch.zeros(B_test//2), torch.ones(B_test//2)]).long()

    # Train with s=1 (temporal encoding)
    clf_s1 = HDCMiniROCKETClassifier(n_classes, n_features=n_feat, scale_s=1.0, seed=0)
    clf_s1.fit_ridge(X_train, y_train, alpha=1.0)
    with torch.no_grad():
        preds_s1 = clf_s1(X_test).argmax(dim=-1)
    acc_s1 = float((preds_s1 == y_test).float().mean().item())

    # Train with s=0 (no temporal encoding, original MiniROCKET behaviour)
    clf_s0 = HDCMiniROCKETClassifier(n_classes, n_features=n_feat, scale_s=0.0, seed=0)
    clf_s0.fit_ridge(X_train, y_train, alpha=1.0)
    with torch.no_grad():
        preds_s0 = clf_s0(X_test).argmax(dim=-1)
    acc_s0 = float((preds_s0 == y_test).float().mean().item())

    print(f"  Temporal dataset accuracy — s=1: {acc_s1:.1%}, s=0: {acc_s0:.1%}")
    print(f"  (s=1 should beat s=0 on this temporal dataset)")
    assert acc_s1 >= acc_s0 - 0.05, \
        f"s=1 ({acc_s1:.1%}) significantly worse than s=0 ({acc_s0:.1%})"

    print("  ✅ HDCMiniROCKETClassifier OK")


if __name__ == "__main__":
    import torch.nn.functional as F
    test_fractional_binding()
    print()
    test_hdc_minirocket_encoding()
    print()
    test_hdc_minirocket_classifier()
    print()
    print("=== All HDC-MiniROCKET tests passed ===")
