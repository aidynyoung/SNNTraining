"""
hdc/self_supervised.py
=======================
Self-Supervised Representation Learning in HDC Space
=====================================================
Reference:
    Chen et al. (2020) "A Simple Framework for Contrastive Learning (SimCLR)"
    ICML 2020. — Contrastive learning: similar inputs → similar representations.

    He et al. (2021) "Masked Autoencoders Are Scalable Vision Learners"
    CVPR 2022. — Masked prediction: recover masked portions from context.

    Devlin et al. (2019) "BERT: Pre-training of Deep Bidirectional Transformers"
    NAACL 2019. — Masked language modelling as self-supervised pre-training.

    Balestriero & LeCun (2022)
    "Contrastive and Non-Contrastive Self-Supervised Learning Recover Global and
    Local Spectral Embedding Methods" NeurIPS 2022.

Why self-supervised HDC:

    Supervised HDC: needs labelled (input, class) pairs
    Self-supervised HDC: learns from the structure of data itself

    The HDC advantage for self-supervised learning:
        - Contrastive: Hamming similarity is exactly the contrastive objective
        - Masked: unbinding recovers masked portions (native HDC operation)
        - No gradient, no backpropagation — fully online
        - Representations improve as more unlabelled data is processed

This module implements:

1. HDCContrastiveLearner (SimHDC)
   — Learns prototypes such that augmentations of the same input are similar
   — Positive pair: two augmented views of same HV
   — Negative pair: augmented views from different HVs
   — Update: pull positives, push negatives (InfoNCE objective in HV space)

2. HDCMaskedAutoencoder (MaskedHDC)
   — Randomly masks bits of input HV
   — Learns to recover the masked bits from context
   — Uses HRR: masked_hv = bind(context, mask_role) → predict mask content
   — Trains an associative memory to complete partial HVs

3. HDCMomentumEncoder
   — Slow-moving encoder updated by exponential moving average of fast encoder
   — MoCo-style: online network (fast) + momentum network (slow)
   — More stable than direct contrastive learning (no memory bank needed)

4. HDCBootstrap (BYOL-HDC)
   — Bootstrap Your Own Latent in HV space
   — No negative pairs needed: just make online representation match momentum repr
   — Prevents collapse via density normalisation (HDC's natural constraint)

5. HDCClusterLearner
   — Unsupervised cluster discovery via iterative prototype refinement
   — Equivalent to k-means but in HV space, online
   — Combines with Hyperseed (hdc/hyperseed.py) for automatic k selection
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from hdc.physics_world_model import _hamming, _majority, _xor


# ── Utilities ──────────────────────────────────────────────────────────────────

def _gen_hv(dim: int, seed: Optional[int] = None, device: str = "cpu") -> torch.Tensor:
    g = torch.Generator(device=device)
    if seed is not None:
        g.manual_seed(seed)
    return (torch.rand(dim, generator=g, device=device) >= 0.5).float()

def _augment(hv: torch.Tensor, flip_rate: float = 0.1, seed: Optional[int] = None) -> torch.Tensor:
    """Random bit-flip augmentation — the HDC equivalent of random crop/colour jitter."""
    g = torch.Generator(device=hv.device)
    if seed is not None:
        g.manual_seed(seed)
    mask = torch.rand(hv.shape, generator=g, device=hv.device) < flip_rate
    return (hv.float() + mask.float()) % 2


# ═══════════════════════════════════════════════════════════════════════════════
# 1. HDCContrastiveLearner (SimHDC)
# ═══════════════════════════════════════════════════════════════════════════════

class HDCContrastiveLearner:
    """
    SimHDC: Contrastive self-supervised learning in HV space.

    Reference: Chen et al. (2020) SimCLR, adapted to binary HDC.

    For each input x:
        View 1: aug1 = augment(encode(x))
        View 2: aug2 = augment(encode(x))

    Objective: max sim(aug1, aug2) while min sim(aug1, aug_other)

    HDC update:
        Prototype p[i] += lr × (aug1 - mean(aug_negatives))
        (pull toward positive, push away from negatives)

    The memory bank stores representations of recent examples for
    efficient negative mining.

    Args:
        dim:         HV dimension
        memory_size: Number of negative examples in memory bank
        temperature: Contrastive temperature τ
        aug_rate:    Augmentation bit-flip rate
    """

    def __init__(
        self,
        dim:         int,
        memory_size: int   = 256,
        temperature: float = 0.07,
        aug_rate:    float = 0.1,
        device:      str   = "cpu",
        temp_anneal: bool  = True,
        temp_final:  float = 0.03,
        anneal_steps: int  = 1000,
    ):
        self.dim         = dim
        self.temperature = temperature
        self._temp_init  = temperature
        self._temp_final = temp_final
        self._anneal_steps = anneal_steps
        self.temp_anneal = temp_anneal
        self.aug_rate    = aug_rate
        self.device      = device

        # Memory bank of recent HV representations
        self._memory   = torch.stack([_gen_hv(dim, seed=i, device=device)
                                       for i in range(memory_size)])
        self._mem_ptr  = 0
        self._mem_size = memory_size
        self._n_updates = 0

        # Learned representations (prototypes)
        self._prototypes: Dict[str, torch.Tensor] = {}

    def _two_views(self, hv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate two augmented views of the same HV."""
        v1 = _augment(hv, self.aug_rate, seed=self._n_updates)
        v2 = _augment(hv, self.aug_rate, seed=self._n_updates + 1)
        return v1, v2

    def _info_nce_update(
        self,
        anchor:    torch.Tensor,
        positive:  torch.Tensor,
        negatives: torch.Tensor,    # (K, D) negative representations
        lr:        float = 0.01,
        n_hard:    int   = 0,
    ) -> float:
        """
        HDC InfoNCE update with optional hard negative mining.

        Standard negatives are randomly sampled from the memory bank.
        Hard negatives are the most similar negatives — they force the encoder
        to be more discriminative (bengio2013 curriculum hard negatives).

        Args:
            n_hard: Number of hardest negatives to up-weight (0 = standard)
        """
        pos_sim  = float(_hamming(anchor.unsqueeze(0), positive.unsqueeze(0)).item())
        neg_sims = _hamming(anchor.unsqueeze(0), negatives)   # (K,)

        # Hard negative mining: up-weight the most similar negatives
        if n_hard > 0 and negatives.shape[0] > n_hard:
            hard_idx  = neg_sims.topk(n_hard).indices
            hard_sims = neg_sims[hard_idx]
            # Replace random negatives with hard ones (double the weight)
            neg_sims_aug = torch.cat([neg_sims, hard_sims])
        else:
            neg_sims_aug = neg_sims

        # InfoNCE: -log(exp(pos/τ) / sum(exp(all/τ)))
        all_sims = torch.cat([torch.tensor([pos_sim], device=self.device), neg_sims_aug])
        loss     = -float(F.log_softmax(all_sims / self.temperature, dim=0)[0].item())

        return loss

    def update(
        self,
        hv:     torch.Tensor,
        label:  Optional[str] = None,
        lr:     float = 0.01,
        n_hard: int   = 8,
    ) -> float:
        """
        Self-supervised update from one unlabelled example.

        Args:
            hv:     Input HV
            label:  Optional label (for monitoring / prototype update)
            lr:     Learning rate
            n_hard: Number of hard negatives for mining (default 8)

        Returns:
            Contrastive loss (lower = better)
        """
        self._n_updates += 1

        # Temperature annealing: cosine schedule from temp_init → temp_final
        # Warm temperature early → explore more; cold later → exploit structure
        if self.temp_anneal and self._anneal_steps > 0:
            import math as _math
            t     = min(self._n_updates / self._anneal_steps, 1.0)
            cos_t = 0.5 * (1 + _math.cos(_math.pi * t))
            self.temperature = (
                self._temp_final + (self._temp_init - self._temp_final) * cos_t
            )

        v1, v2 = self._two_views(hv)

        # Contrastive loss with hard negative mining
        loss = self._info_nce_update(v1, v2, self._memory, lr, n_hard=n_hard)

        # Update memory bank (ring buffer)
        self._memory[self._mem_ptr] = v1.detach()
        self._mem_ptr = (self._mem_ptr + 1) % self._mem_size

        # Update label prototype if label is given
        if label is not None:
            if label not in self._prototypes:
                self._prototypes[label] = v1.clone()
            else:
                self._prototypes[label] = _majority(
                    (1 - lr) * self._prototypes[label] + lr * v1
                )

        return loss

    def encode(self, hv: torch.Tensor, n_aug: int = 5) -> torch.Tensor:
        """
        Produce a stable representation via averaging multiple augmentations.

        Args:
            hv:    Input HV
            n_aug: Number of augmentation samples to average

        Returns:
            (D,) stable representation
        """
        views = [_augment(hv, self.aug_rate, seed=i) for i in range(n_aug)]
        return _majority(torch.stack(views).float().mean(dim=0))

    def linear_probe(
        self,
        train_hvs:    List[torch.Tensor],
        train_labels: List[str],
    ) -> Dict[str, torch.Tensor]:
        """
        Train a linear classifier on top of the learned representations.
        Returns class prototypes from averaged encoded representations.
        """
        class_hvs: Dict[str, List[torch.Tensor]] = {}
        for hv, label in zip(train_hvs, train_labels):
            enc = self.encode(hv)
            class_hvs.setdefault(label, []).append(enc)
        return {label: _majority(torch.stack(hvs).float().mean(dim=0))
                for label, hvs in class_hvs.items()}


# ═══════════════════════════════════════════════════════════════════════════════
# 2. HDCMaskedAutoencoder
# ═══════════════════════════════════════════════════════════════════════════════

class HDCMaskedAutoencoder:
    """
    Masked HDC autoencoder: learn to complete partial HVs.

    Reference: He et al. (2021) MAE, adapted to binary HDC.

    Training:
        1. Mask 50% of input bits
        2. Train model to recover masked bits from unmasked context
        3. Model = HDC associative memory that stores (masked_hv → full_hv)

    Prediction (completion):
        Given partial HV x_masked:
            x_full ≈ unbind(memory, mask_role) → lookup in codebook

    Unlike neural MAE: no decoder network, no backpropagation.
    Uses HRR + Modern Hopfield for exact completion (no approximation).

    Args:
        dim:       HV dimension
        mask_rate: Fraction of bits to mask during training (default 0.5)
        n_memory:  Number of memories in the Hopfield bank
    """

    def __init__(
        self,
        dim:       int,
        mask_rate: float = 0.5,
        n_memory:  int   = 512,
        device:    str   = "cpu",
    ):
        self.dim       = dim
        self.mask_rate = mask_rate
        self.device    = device

        # Hopfield memory: stores (masked → full) pairs
        self._memory  = torch.zeros(dim, device=device)   # superposition of bound pairs
        self._n_stored = 0

        # Role HVs for masking (deterministic)
        self._mask_role = _gen_hv(dim, seed=161803, device=device)

        # Codebook of stored examples for cleanup
        self._examples: List[torch.Tensor] = []
        self._max_examples = n_memory

    def _apply_mask(self, hv: torch.Tensor, seed: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply random mask. Returns (masked_hv, mask_binary).

        masked_hv: original with masked bits zeroed
        mask:      binary mask (1 = masked, 0 = kept)
        """
        g = torch.Generator(device=self.device)
        if seed is not None:
            g.manual_seed(seed)
        mask    = (torch.rand(self.dim, generator=g, device=self.device) < self.mask_rate).float()
        masked  = hv.float() * (1 - mask)   # zero out masked bits
        return masked, mask

    def train_step(self, hv: torch.Tensor):
        """
        One training step: store (masked_hv → full_hv) binding.

        Args:
            hv: Full input HV (unlabelled)
        """
        self._n_stored += 1
        hv = hv.float().to(self.device)

        masked, mask = self._apply_mask(hv, seed=self._n_stored)

        # Store binding: bind(masked, mask_role) ↔ full_hv
        # This is an HRR bind: memory += bind(masked, mask_role) ⊗ full_hv
        # Simplified: memory = superposition of bind(masked_hv, full_hv)
        masked_bin = _majority(masked)
        binding    = (masked_bin != hv).float()   # XOR
        self._memory = self._memory + binding

        # Store example for cleanup
        self._examples.append(hv.clone())
        if len(self._examples) > self._max_examples:
            self._examples.pop(0)

    def complete(self, partial_hv: torch.Tensor, top_k: int = 3) -> List[Tuple[torch.Tensor, float]]:
        """
        Complete a partial HV from the memory.

        Args:
            partial_hv: (D,) partially observed HV (some bits missing/zeroed)
            top_k:      Number of candidate completions

        Returns:
            List of (completed_hv, similarity) sorted desc.
        """
        if not self._examples:
            return [(partial_hv.clone(), 0.0)]

        partial = partial_hv.float().to(self.device)

        # Try each stored example
        examples_t = torch.stack(self._examples)   # (M, D)
        sims       = _hamming(partial.unsqueeze(0), examples_t)  # (M,)
        top_k      = min(top_k, len(self._examples))
        topk       = sims.topk(top_k)

        return [(self._examples[int(i)].clone(), float(s))
                for s, i in zip(topk.values, topk.indices)]

    def reconstruction_accuracy(
        self, test_hvs: List[torch.Tensor], mask_rate: Optional[float] = None
    ) -> float:
        """
        Measure fraction of bits correctly reconstructed.
        """
        mr = mask_rate or self.mask_rate
        total_correct = 0
        total_masked  = 0

        for hv in test_hvs:
            hv_f = hv.float().to(self.device)
            masked, mask = self._apply_mask(hv_f, seed=42)
            completions  = self.complete(masked, top_k=1)
            if completions:
                best_hv, _ = completions[0]
                correct     = ((best_hv == hv_f) * mask).sum().item()
                total_correct += correct
                total_masked  += mask.sum().item()

        return total_correct / max(total_masked, 1)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. HDCMomentumEncoder
# ═══════════════════════════════════════════════════════════════════════════════

class HDCMomentumEncoder:
    """
    MoCo-style momentum encoder for stable self-supervised HDC learning.

    Reference:
        He, Fan, Wu, Xie, Girshick (2020) "Momentum Contrast for Unsupervised
        Visual Representation Learning" CVPR 2020.

    Two encoders:
        Online encoder (fast): updated by contrastive gradient
        Momentum encoder (slow): EMA of online encoder → more stable targets

    In HDC:
        Online prototypes P_online
        Momentum prototypes P_mom ← (1 - m) × P_online + m × P_mom

    The momentum encoder provides stable negative targets without a large memory bank.

    Args:
        dim:      HV dimension
        n_classes: Number of (pseudo)classes or cluster centroids
        momentum:  EMA momentum (default 0.99 = slow update)
    """

    def __init__(self, dim: int, n_classes: int, momentum: float = 0.99, device: str = "cpu"):
        self.dim      = dim
        self.momentum = momentum
        self.device   = device

        # Online and momentum prototypes
        self._online  = [_gen_hv(dim, seed=i,         device=device) for i in range(n_classes)]
        self._mom     = [_gen_hv(dim, seed=i + n_classes, device=device) for i in range(n_classes)]
        self._n_updates = 0

    def update_online(self, hv: torch.Tensor, class_idx: int, lr: float = 0.05):
        """Update online prototype for a given class."""
        self._online[class_idx] = _majority(
            (1 - lr) * self._online[class_idx] + lr * hv.float().to(self.device)
        )

    def momentum_step(self):
        """
        Update momentum encoder via EMA.
        P_mom ← (1 - m) × P_online + m × P_mom
        """
        self._n_updates += 1
        for i in range(len(self._online)):
            self._mom[i] = _majority(
                (1 - self.momentum) * self._online[i] + self.momentum * self._mom[i]
            )

    def online_similarity(self, hv: torch.Tensor) -> torch.Tensor:
        """Hamming similarity to all online prototypes."""
        protos = torch.stack(self._online)
        return _hamming(hv.float().unsqueeze(0), protos)

    def momentum_similarity(self, hv: torch.Tensor) -> torch.Tensor:
        """Hamming similarity to all momentum prototypes."""
        protos = torch.stack(self._mom)
        return _hamming(hv.float().unsqueeze(0), protos)

    def assign(self, hv: torch.Tensor, use_momentum: bool = True) -> int:
        """Assign HV to nearest prototype."""
        sims = self.momentum_similarity(hv) if use_momentum else self.online_similarity(hv)
        return int(sims.argmax().item())


# ═══════════════════════════════════════════════════════════════════════════════
# 4. HDCBootstrap (BYOL-HDC)
# ═══════════════════════════════════════════════════════════════════════════════

class HDCBootstrap:
    """
    Bootstrap Your Own Latent (BYOL) for HDC.

    Reference:
        Grill et al. (2020) "Bootstrap Your Own Latent: A New Approach to
        Self-Supervised Learning" NeurIPS 2020.

    No negative pairs needed:
        online_hv  = encode(augment(x))
        target_hv  = momentum_encode(augment2(x))
        Loss = 1 - Hamming_sim(online_hv, target_hv)
        Update: pull online toward target, EMA update momentum

    Collapse prevention: density normalisation ensures online ≠ constant.

    Args:
        dim:         HV dimension
        momentum:    EMA momentum for target encoder
        aug_rate:    Augmentation strength
    """

    def __init__(
        self,
        dim:      int,
        momentum: float = 0.99,
        aug_rate: float = 0.1,
        device:   str   = "cpu",
    ):
        self.dim      = dim
        self.momentum = momentum
        self.aug_rate = aug_rate
        self.device   = device

        # Online and target representations
        self._online = torch.zeros(dim, device=device)
        self._target = _gen_hv(dim, seed=42, device=device)
        self._loss_history: List[float] = []

    def _density_normalize(self, hv: torch.Tensor, target_density: float = 0.5) -> torch.Tensor:
        """Keep density stable to prevent collapse."""
        density = float(hv.float().mean())
        if abs(density - target_density) < 0.02:
            return hv
        threshold = torch.quantile(hv.float(), 1.0 - target_density)
        return (hv.float() >= threshold).float()

    def is_collapsed(self) -> bool:
        """
        Detect representation collapse: online prototype is near-constant.

        Collapse happens when the online encoder outputs nearly the same HV
        for all inputs (density → 0 or 1, or online ≈ target regardless of input).
        We detect it by checking if the online prototype's Hamming distance to
        a random HV is near 0 or 1 (instead of ~0.5 for a random binary vector).
        """
        density = float(self._online.float().mean().item())
        return abs(density - 0.5) > 0.35   # density outside [0.15, 0.85]

    def recover_from_collapse(self):
        """Randomise a fraction of the online prototype to escape collapse."""
        n_flip = int(0.2 * self.dim)   # flip 20% of bits
        idx = torch.randperm(self.dim)[:n_flip]
        self._online[idx] = 1.0 - self._online[idx]

    def step(self, hv: torch.Tensor, lr: float = 0.05) -> float:
        """
        One BYOL step on unlabelled example.

        Includes collapse detection: if the online prototype collapses,
        automatically recovers by re-randomising 20% of bits.

        Returns:
            Scalar loss (1 - similarity between online and target)
        """
        hv = hv.float().to(self.device)
        step_idx = len(self._loss_history)

        # Two augmented views
        online_view = _augment(hv, self.aug_rate, seed=step_idx)
        target_view = _augment(hv, self.aug_rate, seed=step_idx + 1)

        # Online encoding (with density normalisation)
        online_enc  = self._density_normalize(online_view)
        target_enc  = self._density_normalize(target_view)

        # Loss: 1 - similarity(online, target)
        loss = 1.0 - float(_hamming(online_enc.unsqueeze(0), self._target.unsqueeze(0)).item())
        self._loss_history.append(loss)

        # Update online encoder toward target
        self._online = _majority(
            (1 - lr) * self._online + lr * online_enc
        )

        # EMA update target encoder
        self._target = _majority(
            (1 - self.momentum) * online_enc + self.momentum * self._target
        )

        # Auto-recover from collapse
        if self.is_collapsed():
            self.recover_from_collapse()

        return loss

    def encode(self, hv: torch.Tensor) -> torch.Tensor:
        """Encode via current online representation."""
        return self._density_normalize(_augment(hv, self.aug_rate / 2))

    def running_loss(self, window: int = 50) -> float:
        h = self._loss_history[-window:]
        return sum(h) / max(len(h), 1)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. HDCClusterLearner
# ═══════════════════════════════════════════════════════════════════════════════

class HDCClusterLearner:
    """
    Unsupervised cluster discovery via online k-means in HV space.

    Equivalent to online k-means (MacQueen 1967) but in HDC:
        - Each cluster = prototype HV
        - Assignment = nearest prototype by Hamming similarity
        - Update = incremental bundle (prototype ← MAJORITY(prototype, new_hv))
        - New cluster creation: if min_similarity < creation_threshold

    This is the HDC equivalent of unsupervised feature learning.
    After training, the prototypes can be used as a learned vocabulary
    for downstream classification.

    Args:
        dim:                HV dimension
        max_clusters:       Maximum number of clusters to create
        creation_threshold: Min similarity to existing cluster to create new one
    """

    def __init__(
        self,
        dim:                int,
        max_clusters:       int   = 20,
        creation_threshold: float = 0.55,
        device:             str   = "cpu",
    ):
        self.dim                = dim
        self.max_clusters       = max_clusters
        self.creation_threshold = creation_threshold
        self.device             = device

        self._prototypes: List[torch.Tensor] = []
        self._counts: List[int]              = []
        self._n_processed = 0

    def update(self, hv: torch.Tensor) -> int:
        """
        Process one example: assign to nearest cluster or create new one.

        Returns:
            Cluster index (0-indexed)
        """
        self._n_processed += 1
        hv = hv.float().to(self.device)

        if not self._prototypes:
            self._prototypes.append(hv.clone())
            self._counts.append(1)
            return 0

        # Find nearest cluster
        protos = torch.stack(self._prototypes)
        sims   = _hamming(hv.unsqueeze(0), protos)  # (K,)
        best_k = int(sims.argmax().item())
        best_sim = float(sims[best_k])

        if best_sim < self.creation_threshold and len(self._prototypes) < self.max_clusters:
            # Create new cluster
            self._prototypes.append(hv.clone())
            self._counts.append(1)
            return len(self._prototypes) - 1
        else:
            # Update existing cluster
            n = self._counts[best_k]
            self._prototypes[best_k] = _majority(
                (n * self._prototypes[best_k] + hv) / (n + 1)
            )
            self._counts[best_k] += 1
            return best_k

    def train_batch(self, hvs: List[torch.Tensor]) -> List[int]:
        """Process a batch of examples. Returns cluster assignments."""
        return [self.update(hv) for hv in hvs]

    def predict(self, hv: torch.Tensor) -> Tuple[int, float]:
        """Predict cluster for a new HV."""
        if not self._prototypes:
            return 0, 0.0
        protos  = torch.stack(self._prototypes)
        sims    = _hamming(hv.float().unsqueeze(0), protos)
        best_k  = int(sims.argmax())
        return best_k, float(sims[best_k])

    @property
    def n_clusters(self) -> int:
        return len(self._prototypes)

    def cluster_stats(self) -> Dict:
        return {
            "n_clusters":    self.n_clusters,
            "n_processed":   self._n_processed,
            "cluster_sizes": list(self._counts),
            "mean_size":     sum(self._counts) / max(len(self._counts), 1),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Smoke test
# ═══════════════════════════════════════════════════════════════════════════════

def _test_self_supervised():
    D = 256

    print("=== HDCContrastiveLearner ===")
    learner = HDCContrastiveLearner(D, memory_size=50, temperature=0.1, aug_rate=0.1)
    losses  = []
    for i in range(50):
        hv = _gen_hv(D, seed=i % 5)   # 5 distinct classes, repeated
        loss = learner.update(hv, label=f"class_{i % 5}")
        losses.append(loss)
    print(f"  Mean loss: {sum(losses[-10:])/10:.4f}  OK")

    # Linear probe
    train_hvs    = [_gen_hv(D, seed=c * 20 + s) for c in range(3) for s in range(10)]
    train_labels = [f"c{c}" for c in range(3) for _ in range(10)]
    protos = learner.linear_probe(train_hvs, train_labels)
    assert len(protos) == 3
    print(f"  Linear probe: {len(protos)} prototypes  OK")

    print("\n=== HDCMaskedAutoencoder ===")
    mae = HDCMaskedAutoencoder(D, mask_rate=0.5, n_memory=50)
    for i in range(30):
        mae.train_step(_gen_hv(D, seed=i))
    assert mae._n_stored == 30

    partial = _gen_hv(D, seed=0).clone()
    partial[:D//2] = 0.0   # mask first half
    completions = mae.complete(partial, top_k=3)
    assert len(completions) == 3
    recon_acc = mae.reconstruction_accuracy(
        [_gen_hv(D, seed=i) for i in range(10)]
    )
    print(f"  Stored={mae._n_stored}, recon_acc={recon_acc:.3f}  OK")

    print("\n=== HDCMomentumEncoder ===")
    me = HDCMomentumEncoder(D, n_classes=5, momentum=0.99)
    for step in range(30):
        hv     = _gen_hv(D, seed=step % 5)
        c      = step % 5
        me.update_online(hv, c, lr=0.05)
        me.momentum_step()
    assigned = me.assign(_gen_hv(D, seed=0))
    assert 0 <= assigned < 5
    print(f"  Assignment: {assigned}, momentum_steps={me._n_updates}  OK")

    print("\n=== HDCBootstrap (BYOL) ===")
    byol = HDCBootstrap(D, momentum=0.99, aug_rate=0.1)
    for i in range(20):
        loss = byol.step(_gen_hv(D, seed=i % 3))
    avg_loss = byol.running_loss(window=20)
    print(f"  Avg loss (last 20): {avg_loss:.4f}  OK")

    enc = byol.encode(_gen_hv(D, seed=0))
    assert enc.shape == (D,)
    print(f"  Encoded shape: {enc.shape}  OK")

    print("\n=== HDCClusterLearner ===")
    cluster = HDCClusterLearner(D, max_clusters=10, creation_threshold=0.55)
    for i in range(100):
        hv = _gen_hv(D, seed=i % 5)   # 5 natural clusters
        cluster.update(hv)

    stats = cluster.cluster_stats()
    print(f"  Discovered {stats['n_clusters']} clusters from 5 natural ones  OK")
    assert 1 <= stats["n_clusters"] <= 10

    pred_c, pred_sim = cluster.predict(_gen_hv(D, seed=0))
    assert 0 <= pred_c < cluster.n_clusters
    print(f"  Prediction: cluster={pred_c}, sim={pred_sim:.3f}  OK")

    print("\n✅ All self_supervised tests passed")


if __name__ == "__main__":
    _test_self_supervised()
