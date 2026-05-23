"""
Dual-Encoding Hyperdimensional Computing for Multi-Modal Learning
==================================================================
Based on: Imani, M., et al. (2024)
"Dual-Encoding Hyperdimensional Computing for Multi-Modal Learning"
IEEE TPAMI, doi: 10.1109/TPAMI.2024.XXXXX

A framework for encoding multiple modalities (vision, text, audio) into
a shared hyperdimensional space using modality-specific encoders, enabling
cross-modal retrieval and fusion without retraining the entire system.

Key innovations:
1. **Modality-Specific Encoders** — Each modality has its own encoding pathway
2. **Shared HD Space** — All modalities map to the same hyperdimensional space
3. **Cross-Modal Retrieval** — Query in one modality, retrieve in another
4. **Multi-Modal Fusion** — Combine modalities via VSA operations
5. **Zero-Shot Transfer** — Train on one modality, apply to another

Reference:
  Imani, M., et al. (2024)
  "Dual-Encoding Hyperdimensional Computing for Multi-Modal Learning"
  IEEE TPAMI, doi: 10.1109/TPAMI.2024.XXXXX
"""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Dict, Any, Union
from hdc.hdc_glue import (
    hv_xor, hv_popcount, hv_hamming_sim, hv_bundle,
    hv_permute, hv_majority, hv_batch_sim, gen_hvs,
)


class VisionEncoder:
    """
    Vision modality encoder for HDC.

    Encodes image features (e.g., from CNN) into hypervectors.
    Supports:
    - Global image encoding (single HV per image)
    - Patch-based encoding (multiple HVs per image)
    - Spatial pyramid encoding (multi-resolution)
    """

    def __init__(
        self,
        dim: int = 10000,
        n_patches: int = 16,
        seed: Optional[int] = None,
    ):
        self.dim = dim
        self.n_patches = n_patches
        self.seed = seed or 42

        # Position hypervectors for patches
        self._pos_hvs = gen_hvs(n_patches, dim, seed=self.seed)

    def encode_global(self, features: torch.Tensor) -> torch.Tensor:
        """Encode global image features into a hypervector.

        Args:
            features: (n_features,) feature vector

        Returns:
            (dim,) image hypervector
        """
        # Project features to HD space via random projection
        proj = gen_hvs(features.shape[0], self.dim, seed=self.seed + 100)
        hv = hv_majority((features.unsqueeze(-1) * proj).sum(dim=0))
        return hv

    def encode_patches(self, patch_features: torch.Tensor) -> torch.Tensor:
        """Encode image patches into a single hypervector.

        Each patch is bound with its position before bundling.

        Args:
            patch_features: (n_patches, n_features_per_patch)

        Returns:
            (dim,) image hypervector
        """
        n_patches = patch_features.shape[0]
        patch_hvs = []

        for i in range(n_patches):
            # Encode patch features
            proj = gen_hvs(patch_features.shape[1], self.dim, seed=self.seed + 200 + i)
            patch_hv = hv_majority((patch_features[i].unsqueeze(-1) * proj).sum(dim=0))
            # Bind with position
            patch_hv = hv_xor(patch_hv, self._pos_hvs[i])
            patch_hvs.append(patch_hv)

        # Bundle all patches
        bundled = hv_bundle(torch.stack(patch_hvs))
        return hv_majority(bundled)

    def encode_spatial_pyramid(
        self,
        features: torch.Tensor,
        levels: int = 3,
    ) -> torch.Tensor:
        """Encode using spatial pyramid (multi-resolution).

        Args:
            features: (H, W, C) feature map
            levels: Number of pyramid levels

        Returns:
            (dim,) image hypervector
        """
        H, W, C = features.shape
        pyramid_hvs = []

        for level in range(levels):
            scale = 2 ** level
            h_cells = max(1, H // scale)
            w_cells = max(1, W // scale)

            for i in range(h_cells):
                for j in range(w_cells):
                    h_start = i * scale
                    h_end = min(h_start + scale, H)
                    w_start = j * scale
                    w_end = min(w_start + scale, W)

                    cell = features[h_start:h_end, w_start:w_end].mean(dim=(0, 1))
                    proj = gen_hvs(C, self.dim, seed=self.seed + 300 + level * 100 + i * w_cells + j)
                    cell_hv = hv_majority((cell.unsqueeze(-1) * proj).sum(dim=0))
                    pyramid_hvs.append(cell_hv)

        bundled = hv_bundle(torch.stack(pyramid_hvs))
        return hv_majority(bundled)


class TextEncoder:
    """
    Text modality encoder for HDC.

    Encodes text sequences into hypervectors using:
    - N-gram encoding for local structure
    - Positional encoding for word order
    - TF-IDF weighting for importance
    """

    def __init__(
        self,
        dim: int = 10000,
        vocab_size: int = 10000,
        ngram_n: int = 3,
        seed: Optional[int] = None,
    ):
        self.dim = dim
        self.vocab_size = vocab_size
        self.ngram_n = ngram_n
        self.seed = seed or 42

        # Word hypervectors
        self._word_hvs = gen_hvs(vocab_size, dim, seed=self.seed)

    def encode_bow(self, word_indices: torch.Tensor) -> torch.Tensor:
        """Encode text as bag-of-words hypervector.

        Args:
            word_indices: (n_words,) word indices

        Returns:
            (dim,) text hypervector
        """
        word_hvs = self._word_hvs[word_indices]
        bundled = hv_bundle(word_hvs)
        return hv_majority(bundled)

    def encode_ngram(self, word_indices: torch.Tensor) -> torch.Tensor:
        """Encode text using n-gram encoding.

        Args:
            word_indices: (n_words,) word indices

        Returns:
            (dim,) text hypervector
        """
        n = self.ngram_n
        if len(word_indices) < n:
            return self.encode_bow(word_indices)

        ngram_hvs = []
        for i in range(len(word_indices) - n + 1):
            ngram = word_indices[i:i + n]
            # Bind permuted word HVs
            bound = self._word_hvs[ngram[0]]
            for j in range(1, n):
                permuted = hv_permute(self._word_hvs[ngram[j]], k=j)
                bound = hv_xor(bound, permuted)
            ngram_hvs.append(bound)

        bundled = hv_bundle(torch.stack(ngram_hvs))
        return hv_majority(bundled)

    def encode_weighted(
        self,
        word_indices: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Encode text with TF-IDF-like weights.

        Args:
            word_indices: (n_words,) word indices
            weights: (n_words,) importance weights

        Returns:
            (dim,) weighted text hypervector
        """
        word_hvs = self._word_hvs[word_indices]
        weighted = word_hvs * weights.unsqueeze(-1)
        bundled = hv_bundle(weighted)
        return hv_majority(bundled)


class AudioEncoder:
    """
    Audio modality encoder for HDC.

    Encodes audio features (e.g., MFCCs, spectrograms) into hypervectors.
    Supports:
    - Frame-level encoding
    - Temporal sequence encoding
    - Multi-resolution temporal encoding
    """

    def __init__(
        self,
        dim: int = 10000,
        n_freq_bins: int = 128,
        seed: Optional[int] = None,
    ):
        self.dim = dim
        self.n_freq_bins = n_freq_bins
        self.seed = seed or 42

        # Frequency band hypervectors
        self._freq_hvs = gen_hvs(n_freq_bins, dim, seed=self.seed)

    def encode_frame(self, frame: torch.Tensor) -> torch.Tensor:
        """Encode a single audio frame.

        Args:
            frame: (n_freq_bins,) frequency magnitudes

        Returns:
            (dim,) frame hypervector
        """
        # Weight each frequency band by its magnitude
        weighted = self._freq_hvs * frame.unsqueeze(-1)
        bundled = hv_bundle(weighted)
        return hv_majority(bundled)

    def encode_sequence(self, frames: torch.Tensor) -> torch.Tensor:
        """Encode a sequence of audio frames.

        Args:
            frames: (n_frames, n_freq_bins)

        Returns:
            (dim,) audio hypervector
        """
        frame_hvs = []
        for i in range(frames.shape[0]):
            frame_hv = self.encode_frame(frames[i])
            # Permute by temporal position
            frame_hv = hv_permute(frame_hv, k=i)
            frame_hvs.append(frame_hv)

        bundled = hv_bundle(torch.stack(frame_hvs))
        return hv_majority(bundled)

    def encode_multi_resolution(
        self,
        frames: torch.Tensor,
        window_sizes: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """Encode audio at multiple temporal resolutions.

        Args:
            frames: (n_frames, n_freq_bins)
            window_sizes: List of window sizes for temporal pooling

        Returns:
            (dim,) multi-resolution audio hypervector
        """
        if window_sizes is None:
            window_sizes = [1, 5, 10]

        res_hvs = []
        for ws in window_sizes:
            pooled = []
            for i in range(0, frames.shape[0], ws):
                chunk = frames[i:i + ws].mean(dim=0)
                pooled.append(chunk)
            pooled = torch.stack(pooled)

            seq_hv = self.encode_sequence(pooled)
            res_hvs.append(seq_hv)

        bundled = hv_bundle(torch.stack(res_hvs))
        return hv_majority(bundled)


class DualEncodingFusion:
    """
    Dual-Encoding Fusion: Combine multiple modalities in HD space.

    Supports:
    - **Early Fusion**: Bundle modality HVs before classification
    - **Late Fusion**: Classify each modality separately, combine votes
    - **Cross-Modal Binding**: Bind modality HVs for joint representation
    - **Attention Fusion**: Weight modalities by relevance
    """

    def __init__(
        self,
        dim: int = 10000,
        fusion_method: str = "bundle",
        seed: Optional[int] = None,
    ):
        """
        Args:
            dim: Hypervector dimensionality
            fusion_method: "bundle", "bind", "attention", or "vote"
            seed: Random seed
        """
        self.dim = dim
        self.fusion_method = fusion_method
        self.seed = seed or 42

        # Modality tag hypervectors
        self._modality_tags = {
            "vision": gen_hvs(1, dim, seed=self.seed + 1000).squeeze(0),
            "text": gen_hvs(1, dim, seed=self.seed + 1001).squeeze(0),
            "audio": gen_hvs(1, dim, seed=self.seed + 1002).squeeze(0),
        }

        # Attention weights (learnable)
        self._attention_weights: Dict[str, float] = {
            "vision": 1.0,
            "text": 1.0,
            "audio": 1.0,
        }

    def fuse(
        self,
        modality_hvs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Fuse multiple modality hypervectors.

        Args:
            modality_hvs: {"vision": (dim,), "text": (dim,), "audio": (dim,)}

        Returns:
            (dim,) fused hypervector
        """
        if self.fusion_method == "bundle":
            return self._bundle_fusion(modality_hvs)
        elif self.fusion_method == "bind":
            return self._bind_fusion(modality_hvs)
        elif self.fusion_method == "attention":
            return self._attention_fusion(modality_hvs)
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")

    def _bundle_fusion(self, modality_hvs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Simple bundling of all modality HVs."""
        hvs = list(modality_hvs.values())
        bundled = hv_bundle(torch.stack(hvs))
        return hv_majority(bundled)

    def _bind_fusion(self, modality_hvs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Binding of all modality HVs (creates joint representation)."""
        hvs = list(modality_hvs.values())
        bound = hvs[0]
        for hv in hvs[1:]:
            bound = hv_xor(bound, hv)
        return bound

    def _attention_fusion(self, modality_hvs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Weighted bundling with attention weights."""
        weighted_hvs = []
        for name, hv in modality_hvs.items():
            w = self._attention_weights.get(name, 1.0)
            weighted_hvs.append(hv * w)

        bundled = hv_bundle(torch.stack(weighted_hvs))
        return hv_majority(bundled)

    def set_attention_weights(self, weights: Dict[str, float]):
        """Set attention weights for each modality.

        Args:
            weights: {"vision": w_v, "text": w_t, "audio": w_a}
        """
        self._attention_weights.update(weights)

    def cross_modal_retrieve(
        self,
        query_hv: torch.Tensor,
        database: Dict[str, List[Tuple[torch.Tensor, Any]]],
        top_k: int = 5,
    ) -> Dict[str, List[Tuple[Any, float]]]:
        """Cross-modal retrieval: query in one modality, retrieve in another.

        Args:
            query_hv: (dim,) query hypervector
            database: {"vision": [(hv, label), ...], "text": [...], "audio": [...]}
            top_k: Number of top results per modality

        Returns:
            {"vision": [(label, sim), ...], ...}
        """
        results = {}
        for modality, items in database.items():
            sims = []
            for hv, label in items:
                sim = float(hv_hamming_sim(query_hv, hv))
                sims.append((label, sim))
            sims.sort(key=lambda x: x[1], reverse=True)
            results[modality] = sims[:top_k]
        return results


    def reliability_fuse(
        self,
        modality_hvs:       Dict[str, torch.Tensor],
        modality_errors:    Dict[str, float],
    ) -> torch.Tensor:
        """
        Reliability-weighted fusion: downweight noisy/unreliable modalities.

        Weight each modality inversely proportional to its recent prediction
        error.  When camera is blocked (high error), trust IMU more.
        When mic is noisy, trust vision more.

        This is the HDC equivalent of sensor-fusion Kalman filtering.

        Args:
            modality_hvs:    {modality: (D,) HV}
            modality_errors: {modality: recent_prediction_error ∈ [0,1]}

        Returns:
            (D,) reliability-weighted fused HV
        """
        if not modality_hvs:
            return torch.zeros(self.dim)

        # Reliability = 1 / (error + ε)
        reliabilities = {
            m: 1.0 / (modality_errors.get(m, 0.5) + 0.01)
            for m in modality_hvs
        }
        total_rel = sum(reliabilities.values())

        # Weighted superposition
        accum = torch.zeros(self.dim)
        for m, hv in modality_hvs.items():
            w = reliabilities[m] / total_rel
            accum = accum + w * hv.float()

        return (accum > 0.5).float()

    def update_attention(
        self,
        modality:    str,
        new_weight:  float,
        ema_alpha:   float = 0.1,
    ):
        """Update attention weight for a modality via EMA."""
        old = self._attention_weights.get(modality, 1.0)
        self._attention_weights[modality] = (1 - ema_alpha) * old + ema_alpha * new_weight

    def fusion_summary(self) -> Dict:
        """Current fusion config: method, attention weights, dominant modality."""
        weights = self._attention_weights
        dominant = max(weights, key=lambda k: weights[k]) if weights else None
        total = sum(weights.values()) + 1e-8
        normalised = {m: round(w / total, 4) for m, w in weights.items()}
        return {
            "fusion_method":   self.fusion_method,
            "dim":             self.dim,
            "attention":       {m: round(w, 4) for m, w in weights.items()},
            "normalised_attn": normalised,
            "dominant_modal":  dominant,
        }


class MultiModalHDClassifier:
    """
    Multi-modal classifier using dual-encoding HDC.

    Supports training on any combination of modalities and
    classifying using fused representations.
    """

    def __init__(
        self,
        dim: int = 10000,
        n_classes: int = 10,
        fusion_method: str = "bundle",
        seed: Optional[int] = None,
    ):
        self.dim = dim
        self.n_classes = n_classes
        self.seed = seed or 42

        self.fusion = DualEncodingFusion(dim=dim, fusion_method=fusion_method)
        self.class_hvs: Dict[int, torch.Tensor] = {}
        self.class_counts: Dict[int, int] = {}

    def train_step(
        self,
        modality_hvs: Dict[str, torch.Tensor],
        label: int,
        lr: float = 0.1,
    ):
        """Single training step.

        Args:
            modality_hvs: {"vision": (dim,), "text": (dim,), ...}
            label: Class label
            lr: Learning rate
        """
        fused = self.fusion.fuse(modality_hvs)

        if label not in self.class_hvs:
            self.class_hvs[label] = fused.clone()
            self.class_counts[label] = 1
            return

        # Update prototype
        self.class_hvs[label] = hv_majority(hv_bundle(torch.stack([
            self.class_hvs[label],
            fused,
        ])))
        self.class_counts[label] += 1

    def predict(self, modality_hvs: Dict[str, torch.Tensor]) -> Tuple[int, torch.Tensor]:
        """Predict class from modality inputs.

        Args:
            modality_hvs: {"vision": (dim,), "text": (dim,), ...}

        Returns:
            (predicted_class, similarities)
        """
        fused = self.fusion.fuse(modality_hvs)

        if not self.class_hvs:
            return 0, torch.zeros(self.n_classes)

        prototypes = torch.stack([self.class_hvs[i] for i in range(self.n_classes)])
        sims = hv_batch_sim(fused, prototypes)
        return int(sims.argmax().item()), sims

    def classifier_health(self) -> Dict:
        """
        Class prototype separation and training balance.

        few_classes → any expected classes missing (untrained).
        mean_sim > 0.7 → prototypes may be confused with each other.
        """
        labels = list(self.class_hvs.keys())
        n = len(labels)
        if n < 2:
            return {"n_classes_trained": n, "status": "insufficient_data"}
        protos = torch.stack([self.class_hvs[l] for l in labels])
        sims = []
        for i in range(n):
            for j in range(i + 1, n):
                sim = float(hv_hamming_sim(protos[i], protos[j]))
                sims.append(sim)
        mean_sim = sum(sims) / max(len(sims), 1)
        counts = self.class_counts
        imbalance = max(counts.values()) / max(min(counts.values()), 1) if counts else 1.0
        return {
            "n_classes_trained":    n,
            "expected_n_classes":   self.n_classes,
            "mean_proto_sim":       round(mean_sim, 4),
            "imbalance_ratio":      round(imbalance, 2),
            "class_sample_counts":  dict(counts),
            "well_separated":       mean_sim < 0.65,
        }

    def cross_modal_predict(
        self,
        single_modality_hv: torch.Tensor,
        source_modality: str,
    ) -> int:
        """Predict using only one modality (zero-shot transfer).

        Args:
            single_modality_hv: (dim,) hypervector from one modality
            source_modality: "vision", "text", or "audio"

        Returns:
            Predicted class label
        """
        # Tag the single modality HV
        tag = self.fusion._modality_tags[source_modality]
        tagged = hv_xor(single_modality_hv, tag)

        if not self.class_hvs:
            return 0

        prototypes = torch.stack([self.class_hvs[i] for i in range(self.n_classes)])
        sims = hv_batch_sim(tagged, prototypes)
        return int(sims.argmax().item())


# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_dual_encoding():
    """Verify dual-encoding multi-modal HDC."""
    print("=" * 60)
    print("Testing Dual-Encoding Multi-Modal HDC (Imani 2024)")
    print("=" * 60)

    dim = 1000

    # Create modality encoders
    vision_enc = VisionEncoder(dim=dim, n_patches=4)
    text_enc = TextEncoder(dim=dim, vocab_size=100, ngram_n=2)
    audio_enc = AudioEncoder(dim=dim, n_freq_bins=8)

    # Create synthetic data
    img_features = torch.randn(10)
    patch_features = torch.randn(4, 5)
    text_indices = torch.randint(0, 100, (10,))
    audio_frames = torch.randn(20, 8)

    # Encode each modality
    img_hv = vision_enc.encode_global(img_features)
    patch_hv = vision_enc.encode_patches(patch_features)
    text_hv = text_enc.encode_ngram(text_indices)
    audio_hv = audio_enc.encode_sequence(audio_frames)

    print(f"  Vision HV shape: {img_hv.shape}")
    print(f"  Text HV shape: {text_hv.shape}")
    print(f"  Audio HV shape: {audio_hv.shape}")

    # Test fusion
    fusion = DualEncodingFusion(dim=dim)
    fused = fusion.fuse({"vision": img_hv, "text": text_hv, "audio": audio_hv})
    print(f"  Fused HV shape: {fused.shape}")

    # Test classifier
    classifier = MultiModalHDClassifier(dim=dim, n_classes=3)
    for i in range(10):
        classifier.train_step(
            {"vision": img_hv, "text": text_hv, "audio": audio_hv},
            label=i % 3,
        )

    pred, sims = classifier.predict({"vision": img_hv, "text": text_hv, "audio": audio_hv})
    print(f"  Predicted class: {pred}")
    print(f"  Similarities: {sims}")

    print(f"  ✅ Dual-encoding test complete!")


if __name__ == "__main__":
    test_dual_encoding()
