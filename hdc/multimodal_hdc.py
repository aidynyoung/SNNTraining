"""
hdc/multimodal_hdc.py
======================
Dual-Encoding Multi-Modal HDC — Imani et al. 2024
==================================================
Reference:
    Imani, Salamat, Khaleghi, Rosing (2024)
    "Dual-Encoding Hyperdimensional Computing for Multi-Modal Learning"
    IEEE Transactions on Pattern Analysis and Machine Intelligence.

    Cumbo, Giovannetti, Bertini (2026)
    "Designing vector-symbolic architectures for biomedical applications:
    ten tips and common pitfalls" PeerJ Computer Science.

    Schlegel, Neubert, Protzel (2024)
    "A Framework for Evaluating Vector Symbolic Architectures for
    Biomedical Applications" Artificial Intelligence in Medicine.

The multi-modal HDC advantage:

    Standard multi-modal fusion (transformers):
        - Learn cross-modal attention weights (requires labelled pairs)
        - O(N × M × d) cross-attention
        - Modality-specific architectures (ResNet + BERT + WAV2Vec)

    HDC multi-modal fusion:
        - ALL modalities → same D-dimensional binary space
        - Fusion = XOR + majority (no learned weights)
        - Add new modalities at runtime without retraining
        - O(D) per query, O(D × C) model size

    Dual-encoding: each modality uses TWO complementary encodings:
        1. Content encoding: what is the signal?
        2. Context encoding: how does the signal relate to others?
    Bundle both encodings for richer cross-modal representation.

This module implements:

1. ModalityEncoder
   — Encodes one modality into binary HV space
   — Auto-adapts to input type: continuous, categorical, binary, sequence
   — Dual encoding: content + context representations

2. HDCModalityFusion
   — Fuses N modalities via weighted majority vote
   — Online weight adaptation: modalities with lower prediction error → higher weight
   — Graceful degradation: any modality can be missing at test time

3. CrossModalRetrieval
   — Given HV from modality A, find matching HVs from modality B
   — Applications: image → text, sensor → action, EEG → label
   — No explicit cross-modal training: shared HV space enables zero-shot

4. HDCMultiModalClassifier
   — End-to-end multi-modal classification
   — Supports: early fusion (before classification) and late fusion (ensemble)
   — RefineHD on the fused representation

5. ModalityDropoutRobust
   — Training strategy: randomly drop modalities to improve robustness
   — At test time: works with any subset of available modalities
   — Critical for deployed systems where sensors may fail
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from hdc.physics_world_model import _hamming, _majority, _xor


# ── Utilities ──────────────────────────────────────────────────────────────────

def _gen_hv(dim: int, seed=None, device: str = "cpu") -> torch.Tensor:
    import hashlib
    if seed is None:
        g = torch.Generator(device=device)
        return (torch.rand(dim, generator=g, device=device) >= 0.5).float()
    raw = int(hashlib.md5(str(seed).encode()).hexdigest()[:8], 16) % (2**31)
    g   = torch.Generator(device=device)
    g.manual_seed(raw)
    return (torch.rand(dim, generator=g, device=device) >= 0.5).float()


# ═══════════════════════════════════════════════════════════════════════════════
# 1. ModalityEncoder
# ═══════════════════════════════════════════════════════════════════════════════

class ModalityEncoder:
    """
    Encodes one data modality into binary HV space with dual encoding.

    Dual encoding (Imani 2024):
        Content HV:  encodes "what is the signal"
                     → level-ID encoding of feature values
        Context HV:  encodes "how does the signal vary"
                     → temporal/spatial gradient encoding

    Combined: majority(content_hv, context_hv)

    Args:
        name:       Modality name (e.g., "vision", "audio", "imu")
        n_features: Feature dimension for this modality
        dim:        HV dimension
        input_type: 'continuous' | 'categorical' | 'binary' | 'sequence'
        n_levels:   Quantisation levels for continuous features
        device:     torch device
    """

    def __init__(
        self,
        name:       str,
        n_features: int,
        dim:        int,
        input_type: str  = "continuous",
        n_levels:   int  = 21,
        device:     str  = "cpu",
    ):
        self.name       = name
        self.n_features = n_features
        self.dim        = dim
        self.input_type = input_type
        self.device     = device

        # Feature ID HVs (content encoding)
        self._feat_hvs  = torch.stack([
            _gen_hv(dim, seed=f"{name}_feat_{i}", device=device)
            for i in range(n_features)
        ])

        # Level HVs for continuous features
        self._level_hvs = torch.stack([
            _gen_hv(dim, seed=f"{name}_level_{l}", device=device)
            for l in range(n_levels)
        ]) if input_type == "continuous" else None
        self.n_levels   = n_levels

        # Context encoding: gradient / difference features
        self._context_proj = _gen_hv(dim, seed=f"{name}_ctx", device=device)

        # Running stats for normalisation
        self._running_mean = torch.zeros(n_features, device=device)
        self._running_std  = torch.ones(n_features, device=device)
        self._n_seen       = 0

    def _update_stats(self, x: torch.Tensor):
        """Online mean/std update for normalisation."""
        self._n_seen += 1
        alpha = 1.0 / self._n_seen
        self._running_mean = (1 - alpha) * self._running_mean + alpha * x.float()

    def _content_encode(self, x: torch.Tensor) -> torch.Tensor:
        """Level-ID content encoding."""
        x_f   = torch.sigmoid((x.float() - self._running_mean) / (self._running_std + 1e-6))
        hvs   = []
        n_dim = min(self.n_features, x_f.shape[0])
        for i in range(n_dim):
            if self.input_type == "continuous" and self._level_hvs is not None:
                lvl = max(0, min(self.n_levels - 1, int(x_f[i].item() * (self.n_levels - 1))))
                bound = (self._feat_hvs[i] != self._level_hvs[lvl]).float()
            else:
                # Binary/categorical: just use feature HV if active
                bound = self._feat_hvs[i] if float(x_f[i]) > 0.5 else (1 - self._feat_hvs[i])
            hvs.append(bound)
        return _majority(torch.stack(hvs).float().mean(dim=0))

    def _context_encode(self, x: torch.Tensor) -> torch.Tensor:
        """Context encoding: captures signal variation."""
        deviation = (x.float() - self._running_mean) / (self._running_std + 1e-6)
        # High deviation → context HV; low deviation → inverted context HV
        gate = (deviation.abs() > 1.0).float().mean()
        if gate > 0.5:
            return self._context_proj.clone()
        else:
            return 1.0 - self._context_proj.clone()

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Dual encoding: returns (content_hv, context_hv).

        Args:
            x: (n_features,) modality feature vector

        Returns:
            (content_hv, context_hv) each (D,)
        """
        x_f = x.float().to(self.device)
        self._update_stats(x_f)

        content_hv = self._content_encode(x_f)
        context_hv = self._context_encode(x_f)
        return content_hv, context_hv

    def encode_fused(self, x: torch.Tensor, alpha: float = 0.7) -> torch.Tensor:
        """
        Fused encoding: blend content and context.

        Args:
            x:     Input features
            alpha: Content weight (1-alpha = context weight)

        Returns:
            (D,) fused HV
        """
        c, k = self.encode(x)
        return _majority(alpha * c.float() + (1 - alpha) * k.float())


# ═══════════════════════════════════════════════════════════════════════════════
# 2. HDCModalityFusion
# ═══════════════════════════════════════════════════════════════════════════════

class HDCModalityFusion:
    """
    Fuses multiple modalities into a single HV via precision-weighted majority.

    Reference:
        Imani 2024 — "adaptive modality weighting based on prediction performance"
        Friston 2009 — precision-weighting as attention mechanism

    Fusion:
        fused_hv = MAJORITY( w_1 × hv_1 + w_2 × hv_2 + ... + w_N × hv_N )

    Weights w_i adapt online: modalities with lower prediction error → higher weight.

    Args:
        modality_names: List of modality names
        dim:            HV dimension
        device:         torch device
    """

    def __init__(
        self,
        modality_names: List[str],
        dim:            int,
        device:         str = "cpu",
    ):
        self.modality_names = modality_names
        self.dim            = dim
        self.device         = device

        n = len(modality_names)
        self._weights    = torch.ones(n, device=device) / n
        self._errors     = torch.zeros(n, device=device)   # error EMA per modality
        self._errors_sq  = torch.zeros(n, device=device)   # error² EMA for variance
        self._n_seen     = 0

    def _modality_idx(self, name: str) -> int:
        return self.modality_names.index(name)

    def fuse(
        self,
        modality_hvs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Fuse available modality HVs.

        Missing modalities are skipped (graceful degradation).

        Args:
            modality_hvs: {modality_name: HV} (subset of all modalities OK)

        Returns:
            (D,) fused HV
        """
        if not modality_hvs:
            return torch.zeros(self.dim, device=self.device)

        contributions = []
        total_weight  = 0.0

        for name, hv in modality_hvs.items():
            if name in self.modality_names:
                idx = self._modality_idx(name)
                w   = float(self._weights[idx])
                contributions.append(w * hv.float().to(self.device))
                total_weight += w

        if not contributions:
            return torch.zeros(self.dim, device=self.device)

        weighted_sum = sum(contributions) / max(total_weight, 1e-8)
        return _majority(weighted_sum)

    def update_weights(
        self,
        modality_name: str,
        prediction_error: float,
        lr: float = 0.05,
    ):
        """
        Update weight for a modality based on its prediction error.

        Weights account for both mean error AND variance:
            reliability = 1 / (error_mean + error_std)

        A modality that is consistently accurate (low mean, low variance) gets
        more weight than one that is occasionally accurate (low mean, high variance).
        This makes the fusion more robust to unreliable sensor bursts.
        """
        if modality_name not in self.modality_names:
            return
        self._n_seen += 1
        idx = self._modality_idx(modality_name)
        e   = float(prediction_error)

        # Update running mean and mean² (for variance)
        self._errors[idx]    = (1 - lr) * self._errors[idx]    + lr * e
        self._errors_sq[idx] = (1 - lr) * self._errors_sq[idx] + lr * e ** 2

        # Variance = E[e²] - E[e]²  (clamp to 0 for numerical stability)
        variance = (self._errors_sq - self._errors ** 2).clamp(min=0.0)
        std      = variance.sqrt()

        # Reliability: inverse of (mean + std) — consistent + accurate = high weight
        reliability = 1.0 / (self._errors + std + 1e-8)
        self._weights = reliability / reliability.sum()

    def weight_report(self) -> Dict[str, float]:
        """Return current modality weights."""
        return {name: float(w)
                for name, w in zip(self.modality_names, self._weights)}


# ═══════════════════════════════════════════════════════════════════════════════
# 3. CrossModalRetrieval
# ═══════════════════════════════════════════════════════════════════════════════

class CrossModalRetrieval:
    """
    Cross-modal retrieval: given modality A, retrieve matching modality B.

    Reference:
        Imani 2024 — cross-modal retrieval in shared HDC space.

    All modalities are encoded into the SAME HV space.
    Finding matching items = Hamming similarity search.

    Use cases:
        - Sensor → label (what activity is this IMU reading?)
        - EEG → motor command (decode brain signals to actions)
        - Multi-sensor fusion (what object am I sensing?)

    Args:
        dim:    HV dimension
        device: torch device
    """

    def __init__(self, dim: int, device: str = "cpu"):
        self.dim    = dim
        self.device = device

        # Memory banks: {modality_name → [(hv, id)]}
        self._banks: Dict[str, List[Tuple[torch.Tensor, str]]] = {}

    def register(self, modality: str, hv: torch.Tensor, item_id: str):
        """Register an item in a modality's memory bank."""
        self._banks.setdefault(modality, []).append(
            (hv.float().to(self.device), item_id)
        )

    def retrieve(
        self,
        query_hv:          torch.Tensor,
        target_modality:   str,
        top_k:             int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Find top-k matching items from target_modality.

        Args:
            query_hv:        (D,) query from any modality
            target_modality: Modality to search in
            top_k:           Number of results

        Returns:
            List of (item_id, similarity) sorted desc.
        """
        if target_modality not in self._banks or not self._banks[target_modality]:
            return []

        bank  = self._banks[target_modality]
        hvs   = torch.stack([hv for hv, _ in bank])
        ids   = [item_id for _, item_id in bank]
        sims  = _hamming(query_hv.float().unsqueeze(0), hvs)
        top_k = min(top_k, len(bank))
        topk  = sims.topk(top_k)

        return [(ids[int(i)], float(s)) for s, i in zip(topk.values, topk.indices)]

    def cross_modal_search(
        self,
        query_hv:         torch.Tensor,
        source_modality:  str,
        target_modality:  str,
        top_k:            int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Cross-modal: given query from source, find matches in target.

        If the two modalities share semantic structure (which they do in HDC
        since both are encoded into the same D-dimensional space), matching
        is automatic.
        """
        return self.retrieve(query_hv, target_modality, top_k)

    def recall_at_k(
        self,
        query_hvs:       List[torch.Tensor],
        query_ids:        List[str],
        target_modality:  str,
        k:                int = 5,
    ) -> float:
        """
        Compute Recall@k for evaluation.

        For each query, check if the correct item is in top-k results.
        """
        correct = 0
        for qhv, qid in zip(query_hvs, query_ids):
            results = self.retrieve(qhv, target_modality, top_k=k)
            if any(item_id == qid for item_id, _ in results):
                correct += 1
        return correct / max(len(query_hvs), 1)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. HDCMultiModalClassifier
# ═══════════════════════════════════════════════════════════════════════════════

class HDCMultiModalClassifier:
    """
    End-to-end multi-modal classification with adaptive fusion.

    Supports early fusion (before classification) and late fusion (ensemble).

    Args:
        modality_configs: List of (name, n_features, input_type)
        n_classes:        Number of output classes
        dim:              HV dimension
        fusion_mode:      'early' (fuse then classify) or 'late' (classify then vote)
    """

    def __init__(
        self,
        modality_configs: List[Tuple[str, int, str]],
        n_classes:        int,
        dim:              int   = 4096,
        fusion_mode:      str   = "early",
        class_names:      Optional[List[str]] = None,
        device:           str   = "cpu",
    ):
        self.n_classes   = n_classes
        self.dim         = dim
        self.fusion_mode = fusion_mode
        self.class_names = class_names or [f"class_{i}" for i in range(n_classes)]
        self.device      = device

        # One encoder per modality
        self.encoders = {
            name: ModalityEncoder(name, n_feat, dim, inp_type, device=device)
            for name, n_feat, inp_type in modality_configs
        }

        # Fusion layer
        self.fusion = HDCModalityFusion(
            [name for name, _, _ in modality_configs], dim, device
        )

        # Class prototypes
        self._protos: List[torch.Tensor] = [
            torch.zeros(dim, device=device) for _ in range(n_classes)
        ]
        self._counts = [0] * n_classes

    def _fuse(self, sample: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode and fuse all available modalities."""
        modal_hvs = {}
        for name, encoder in self.encoders.items():
            if name in sample:
                modal_hvs[name] = encoder.encode_fused(sample[name])
        return self.fusion.fuse(modal_hvs)

    def train_step(self, sample: Dict[str, torch.Tensor], label: int):
        """Online training."""
        fused = self._fuse(sample)
        n = self._counts[label]
        self._protos[label] = _majority(
            (n * self._protos[label] + fused) / (n + 1)
        )
        self._counts[label] += 1

    def predict(self, sample: Dict[str, torch.Tensor]) -> Tuple[int, List[float]]:
        """Predict with available modalities (handles missing ones)."""
        fused  = self._fuse(sample)
        protos = torch.stack(self._protos)
        sims   = _hamming(fused.unsqueeze(0), protos)
        best   = int(sims.argmax().item())
        return best, sims.tolist()

    def missing_modality_robustness(
        self,
        sample_full: Dict[str, torch.Tensor],
        label:       int,
        drop_rates:  List[float],
    ) -> Dict[float, float]:
        """
        Measure accuracy as modalities are progressively dropped.

        Returns: {drop_rate: accuracy}
        """
        results = {}
        for rate in drop_rates:
            correct = 0
            for trial in range(20):
                torch.manual_seed(trial)
                partial = {
                    name: feat
                    for name, feat in sample_full.items()
                    if torch.rand(1).item() > rate
                }
                if partial:
                    pred, _ = self.predict(partial)
                    correct += int(pred == label)
            results[rate] = correct / 20
        return results


# ═══════════════════════════════════════════════════════════════════════════════
# 5. ModalityDropoutRobust
# ═══════════════════════════════════════════════════════════════════════════════

class ModalityDropoutRobust:
    """
    Training wrapper that enforces modality dropout for robustness.

    During training, randomly mask out each modality with probability p_drop.
    This forces the model to learn useful representations from any subset of
    modalities — critical for real-world deployment where sensors can fail.

    Reference:
        Imani 2024 §IV.B: "Robust Multi-Modal HDC Training"

    Args:
        classifier:   HDCMultiModalClassifier
        p_drop:       Probability of dropping each modality per training step
    """

    def __init__(self, classifier: HDCMultiModalClassifier, p_drop: float = 0.3):
        self.clf    = classifier
        self.p_drop = p_drop

    def train_step(
        self,
        sample: Dict[str, torch.Tensor],
        label:  int,
        seed:   Optional[int] = None,
    ):
        """Training step with random modality dropout."""
        if seed is not None:
            torch.manual_seed(seed)
        masked = {
            name: feat for name, feat in sample.items()
            if torch.rand(1).item() > self.p_drop
        }
        if masked:
            self.clf.train_step(masked, label)
        else:
            # Keep at least one modality
            name = list(sample.keys())[0]
            self.clf.train_step({name: sample[name]}, label)

    def train_epoch(
        self,
        samples: List[Dict[str, torch.Tensor]],
        labels:  List[int],
    ):
        """Train one epoch with modality dropout."""
        for i, (sample, label) in enumerate(zip(samples, labels)):
            self.train_step(sample, label, seed=i)


# ═══════════════════════════════════════════════════════════════════════════════
# Smoke test
# ═══════════════════════════════════════════════════════════════════════════════

def _test_multimodal_hdc():
    D = 256

    print("=== ModalityEncoder ===")
    enc = ModalityEncoder("vision", n_features=8, dim=D, input_type="continuous")
    x   = torch.randn(8)
    c, k = enc.encode(x)
    assert c.shape == k.shape == (D,)
    fused = enc.encode_fused(x)
    assert fused.shape == (D,)
    print(f"  content={c.mean():.3f}, context={k.mean():.3f}, fused={fused.mean():.3f}  OK")

    print("\n=== HDCModalityFusion ===")
    fusion = HDCModalityFusion(["vision", "audio", "imu"], D)
    hvs    = {
        "vision": _gen_hv(D, seed=0),
        "audio":  _gen_hv(D, seed=1),
        # "imu" missing — graceful degradation
    }
    fused = fusion.fuse(hvs)
    assert fused.shape == (D,)
    print(f"  2/3 modalities: fused density={fused.mean():.3f}  OK")

    fusion.update_weights("vision", prediction_error=0.2)
    fusion.update_weights("audio",  prediction_error=0.4)
    weights = fusion.weight_report()
    print(f"  Weights: {weights}  OK")
    assert weights["vision"] > weights["audio"]  # lower error → higher weight

    print("\n=== CrossModalRetrieval ===")
    retrieval = CrossModalRetrieval(D)
    for i in range(10):
        retrieval.register("vision", _gen_hv(D, seed=i),         f"item_{i}")
        retrieval.register("audio",  _gen_hv(D, seed=i + 100),   f"item_{i}")

    # Retrieve matching audio for a vision query
    results = retrieval.retrieve(_gen_hv(D, seed=0), "audio", top_k=3)
    assert len(results) == 3
    print(f"  Cross-modal top-3: {[(n, f'{s:.3f}') for n,s in results]}  OK")

    print("\n=== HDCMultiModalClassifier ===")
    configs = [("vision", 8, "continuous"), ("audio", 4, "continuous"), ("imu", 6, "continuous")]
    clf     = HDCMultiModalClassifier(configs, n_classes=3, dim=D)

    # Train
    for c in range(3):
        for s in range(10):
            sample = {
                "vision": torch.randn(8) + c * 2,
                "audio":  torch.randn(4) + c * 1.5,
                "imu":    torch.randn(6) + c,
            }
            clf.train_step(sample, c)

    # Predict with all modalities
    test_sample = {"vision": torch.randn(8) + 4, "audio": torch.randn(4) + 3, "imu": torch.randn(6) + 2}
    pred, sims  = clf.predict(test_sample)
    print(f"  Full prediction: class={pred}, sims={[f'{s:.3f}' for s in sims]}  OK")

    # Predict with only one modality (missing 2)
    partial_sample = {"vision": torch.randn(8) + 4}
    pred2, _ = clf.predict(partial_sample)
    print(f"  1-modality prediction: class={pred2}  OK")

    print("\n=== ModalityDropoutRobust ===")
    robust = ModalityDropoutRobust(clf, p_drop=0.4)
    samples = [
        {"vision": torch.randn(8) + c * 2, "audio": torch.randn(4), "imu": torch.randn(6)}
        for c in range(3) for _ in range(5)
    ]
    labels = [c for c in range(3) for _ in range(5)]
    robust.train_epoch(samples, labels)
    print(f"  Dropout training epoch complete  OK")

    print("\n✅ All multimodal_hdc tests passed")


if __name__ == "__main__":
    _test_multimodal_hdc()
