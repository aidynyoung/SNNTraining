"""
hdc/hypervector_architecture.py
================================
HyperVector Architecture (HVA) — Compositional AI via Hypervectors.

The fundamental insight:
    Don't replace neural networks. Coexist with them.
    Autoencoders solve the translation problem for you.
    Multimodality is free once everything is a hypervector.
    Slap together architectures that are completely different.
    No retraining required.

The shift:
    Old paradigm: single model → vector → decision
    HVA paradigm: N models → N hypervectors → compose → decision

Why this works:
    - Any model (NN, SNN, HDC, transformer) maps to the same HV space
      via an AutoencoderBridge (or zero-shot projection)
    - Hypervectors use the same mathematical space regardless of source
    - Composition (bind, bundle) is free — no new parameters, no backprop
    - Old models don't go away; they become HV encoders
    - Add a new model: plug it in. No retraining. 10ms.
    - Remove a model: the others compensate (graceful degradation)
    - Category theory is the harmonizing point: morphisms preserve structure

Architecture::

    Input A (image)    Input B (audio)    Input C (sensor)
         ↓                   ↓                   ↓
    ResNet-18          Wav2Vec 2.0          Arthedain SNN
         ↓                   ↓                   ↓
    AutoBridge          AutoBridge          AutoBridge
         ↓                   ↓                   ↓
      HV_A (D)           HV_B (D)            HV_C (D)
         └───────────────────┴────────────────────┘
                             ↓
                   HVComposer (bundle/bind/morph)
                             ↓
                    HV_joint (D) ← the decision
                             ↓
              AdaptiveHDClassifier / D2H-AD / Readout

References:
    Ghajari (2026) D2H-AD — hybrid NN+HDC, wrong way = replace, right way = coexist
    Cumbo (2026)   Biomedical VSA — multimodality is free
    Rotam (2025)   Chrology — category theory as harmonizing point
    Karunaratne (2020) In-memory HDC — single-shot holographic lookup
    Teeters (2023) Long/short-term HDC memory
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from hdc.autoencoder_bridge import AutoencoderBridge, BridgeConfig, MultimodalFusion
from hdc.concentration import DIM_CANONICAL, binarize_to_mean as _binarize_to_mean
from hdc.physics_world_model import _hamming, _majority

logger = logging.getLogger(__name__)


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class HVModelConfig:
    """Configuration for a single HV-wrapped model."""
    hv_dim: int = 4096           # Shared hypervector dimension
    model_output_dim: int = 512  # Output dimension of the wrapped model
    bridge_hidden: int = 256     # AutoencoderBridge hidden dim
    bridge_layers: int = 2       # AutoencoderBridge depth
    role_name: str = "model"     # Modality/role label for CrossModalBinding
    freeze_bridge: bool = False  # If True, bridge weights are not trained
    device: str = "cpu"


@dataclass
class HVComposerConfig:
    """Configuration for the HV composition layer."""
    hv_dim: int = 4096
    strategy: str = "bundle"     # "bundle" | "bind" | "weighted" | "sequential"
    n_classes: int = 10          # For built-in AdaptiveHDClassifier head
    enable_anomaly: bool = False  # Attach D2H-AD anomaly head
    anomaly_percentile: float = 95.0
    device: str = "cpu"


# ── HVModel — wraps ANY model as a hypervector producer ───────────────────────

class HVModel(nn.Module):
    """
    Wraps any callable model with a bridge to hypervector space.

    The wrapped model can be:
    - A pretrained PyTorch nn.Module (ResNet, BERT, Wav2Vec …)
    - An Arthedain SNN (RSNN + Readout)
    - A pure HDC module (already outputs hypervectors)
    - Any callable that takes a tensor and returns a tensor

    The bridge (AutoencoderBridge) maps the model's output to a
    binary D-dimensional hypervector.  If the model already produces
    D-dimensional binary outputs, the bridge can be bypassed.

    Example::

        import torchvision.models as tv
        resnet = tv.resnet18(pretrained=True)
        resnet.fc = nn.Identity()   # strip classifier, keep 512-d features

        hv_resnet = HVModel(
            resnet,
            config=HVModelConfig(hv_dim=4096, model_output_dim=512, role_name="vision"),
        )
        hv = hv_resnet(image_batch)   # → (batch, 4096) binary hypervector
    """

    def __init__(
        self,
        model: Callable,
        config: Optional[HVModelConfig] = None,
        bridge: Optional[AutoencoderBridge] = None,
        bypass_bridge: bool = False,
    ):
        super().__init__()
        self.cfg = config or HVModelConfig()
        self.role_name = self.cfg.role_name
        self.bypass_bridge = bypass_bridge
        self._energy_xors: int = 0
        # Running median threshold for balanced binarization (bypass_bridge mode)
        self._running_mean: Optional[torch.Tensor] = None  # kept for compat
        self._running_threshold: Optional[torch.Tensor] = None

        # Wrap the model — if it's an nn.Module, register it so parameters move
        if isinstance(model, nn.Module):
            self.model = model
        else:
            # Non-nn.Module callable: wrap in a container so it's not lost
            self._model_fn = model
            self.model = None

        # Bridge: map model output dim → hv_dim
        if bypass_bridge:
            self.bridge = None
        else:
            if bridge is not None:
                self.bridge = bridge
            else:
                bridge_cfg = BridgeConfig(
                    input_dim=self.cfg.model_output_dim,
                    hdc_dim=self.cfg.hv_dim,
                    hidden_dim=self.cfg.bridge_hidden,
                    encoding_layers=self.cfg.bridge_layers,
                    device=self.cfg.device,
                )
                self.bridge = AutoencoderBridge(bridge_cfg)

        if self.cfg.freeze_bridge and self.bridge is not None:
            for p in self.bridge.encoder.parameters():
                p.requires_grad_(False)

    def _call_model(self, x: torch.Tensor) -> torch.Tensor:
        if self.model is not None:
            return self.model(x)
        return self._model_fn(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Produce a binary hypervector from raw input.

        Args:
            x: Input tensor (any shape the wrapped model accepts)

        Returns:
            (D,) or (batch, D) binary hypervector
        """
        out = self._call_model(x)

        # Flatten to 2D if needed (batch × features)
        if out.dim() > 2:
            out = out.flatten(start_dim=1)

        if self.bypass_bridge:
            # Threshold at the per-sample mean, not at zero.
            # This keeps produced HVs balanced (≈50% ones), placing them in the
            # same statistical space as the random coin-flip basis vectors.
            # Thresholding at zero only works for zero-centred activations
            # (not true for ReLU, sigmoid, or any shifted layer).
            # Use running median threshold for balanced binarization.
            # Median is computed per-sample then EMA-smoothed across batches.
            if out.dim() > 1:
                batch_median = out.detach().median(dim=0).values
                if self._running_threshold is None:
                    self._running_threshold = batch_median
                else:
                    self._running_threshold = 0.99 * self._running_threshold + 0.01 * batch_median
            hv = _binarize_to_mean(out, self._running_threshold)
        else:
            hv = self.bridge.encode(out)

        # Track approximate XOR energy: 0.1 pJ per bit
        self._energy_xors += int(hv.numel())
        return hv

    @property
    def energy_pJ(self) -> float:
        return self._energy_xors * 0.1

    def __repr__(self) -> str:
        model_name = type(self.model).__name__ if self.model else "fn"
        return (
            f"HVModel(role={self.role_name}, model={model_name}, "
            f"hv_dim={self.cfg.hv_dim}, bridge={'yes' if self.bridge else 'bypass'})"
        )


# ── HVPrototypeHead — RefineHD on pre-encoded hypervectors ───────────────────

class HVPrototypeHead(nn.Module):
    """
    Prototype classifier that operates directly on composed hypervectors.

    Unlike AdaptiveHDClassifier (which re-encodes raw features), this head
    receives a joint HV that is ALREADY in hypervector space and simply
    computes Hamming similarity to per-class prototype HVs.

    Learning: RefineHD (Vergés Boncompte 2025) — pull correct, push wrong,
    per-class adaptive learning rate.  No re-encoding, no backprop.

    Anomaly: D2H-AD (Ghajari 2026) — distance to nearest prototype.
    """

    def __init__(self, n_classes: int, hv_dim: int, lr: float = 0.1, device: str = "cpu"):
        super().__init__()
        self.n_classes = n_classes
        self.hv_dim = hv_dim
        self.lr = lr
        self.device = device

        # Random initial prototypes
        g = torch.Generator()
        g.manual_seed(0)
        proto = torch.randint(0, 2, (n_classes, hv_dim), generator=g).float()
        self.register_buffer("prototypes", proto)
        self.register_buffer("counts", torch.zeros(n_classes))

        # D2H-AD state
        self._d2h_active = False
        self._d2h_distances: list = []
        self._d2h_threshold: Optional[float] = None

    def _flatten(self, hv: torch.Tensor) -> torch.Tensor:
        """Ensure hv is 1-D: squeeze batch dim of size 1."""
        if hv.dim() == 2 and hv.shape[0] == 1:
            return hv.squeeze(0)
        if hv.dim() == 2 and hv.shape[0] > 1:
            return hv.mean(dim=0)
        return hv

    def _hamming_sim(self, hv: torch.Tensor) -> torch.Tensor:
        """Hamming similarity between hv (D,) and all prototypes (C, D)."""
        hv_b = (self._flatten(hv) > 0).float()             # (D,)
        proto_b = (self.prototypes > 0).float()             # (C, D)
        mismatches = (proto_b != hv_b.unsqueeze(0)).sum(dim=-1).float()  # (C,)
        return 1.0 - mismatches / self.hv_dim

    def predict(self, hv: torch.Tensor) -> Tuple[int, torch.Tensor]:
        """Return (class_idx, similarities) for a joint hypervector."""
        sims = self._hamming_sim(hv)
        return int(sims.argmax().item()), sims

    def train_step(self, hv: torch.Tensor, label: int) -> None:
        """RefineHD update: pull correct prototype, push wrong prototype."""
        with torch.no_grad():
            sims = self._hamming_sim(hv)
            pred = int(sims.argmax().item())

        count = self.counts[label].item()
        lr = self.lr / (1.0 + count * 0.1)
        hv_b = (hv > 0).float()

        if pred == label:
            pull = lr * (1.0 - float(sims[label]))
            self.prototypes[label] = (self.prototypes[label] + pull * hv_b)
        else:
            push = lr * (1.0 - float(sims[pred]))
            self.prototypes[pred] = (self.prototypes[pred] - push * hv_b)
            pull = lr * (1.0 - float(sims[label]))
            self.prototypes[label] = (self.prototypes[label] + pull * hv_b)

        # Binarise (threshold at 0.5)
        self.prototypes[label] = (self.prototypes[label] > 0.5).float()
        if pred != label:
            self.prototypes[pred] = (self.prototypes[pred] > 0.5).float()

        self.counts[label] += 1

    def enable_anomaly_detection(
        self, percentile: float = 95.0, warmup_steps: int = 200
    ) -> None:
        self._d2h_percentile = percentile
        self._d2h_warmup = warmup_steps
        self._d2h_active = True

    def anomaly_score(self, hv: torch.Tensor) -> Tuple[float, bool]:
        sims = self._hamming_sim(hv)
        score = 1.0 - float(sims.max().item())
        is_anomaly = (
            self._d2h_threshold is not None and score > self._d2h_threshold
        )
        if self._d2h_active:
            self._d2h_distances.append(score)
            if len(self._d2h_distances) >= getattr(self, "_d2h_warmup", 200):
                import numpy as np
                self._d2h_threshold = float(
                    np.percentile(self._d2h_distances, self._d2h_percentile)
                )
        return score, is_anomaly


# ── HVComposer — composes N hypervectors into one ─────────────────────────────

class HVComposer(nn.Module):
    """
    Composes hypervectors from multiple HVModels into a single decision HV.

    Composition strategies:
    - **bundle**: majority vote — models are parallel experts, equal weight
    - **bind**: role-filler — each model's HV is XOR'd with its role key,
                then bundled; lets you recover per-model contributions later
    - **weighted**: confidence-weighted bundling — uses per-model weights
    - **sequential**: apply a chain of HDC morphisms (permute → bind → bundle)

    Why compose in HV space instead of concatenating?
    - O(D) instead of O(N×D): composition doesn't grow the representation
    - Fault tolerant: drop one model → others compensate
    - No retraining: add a new HVModel, plug in, done
    - Category theoretically sound: bundle/bind are functors

    Example::

        composer = HVComposer(
            [hv_resnet, hv_snn, hv_gpt],
            config=HVComposerConfig(hv_dim=4096, strategy="bundle", n_classes=20),
        )
        joint_hv = composer.compose([vision_hv, audio_hv, text_hv])
        pred, sims = composer.classify(joint_hv)
    """

    def __init__(
        self,
        hv_models: List[HVModel],
        config: Optional[HVComposerConfig] = None,
    ):
        super().__init__()
        self.cfg = config or HVComposerConfig()
        self.hv_models = nn.ModuleList(hv_models)
        self._fusion = MultimodalFusion(hdc_dim=self.cfg.hv_dim)

        # Role keys for bind strategy: each model gets a fixed random role HV
        role_keys = torch.randint(0, 2, (len(hv_models), self.cfg.hv_dim)).float()
        self.register_buffer("role_keys", role_keys)

        # Per-model learnable weights (softmax-normalised during compose)
        self.model_weights = nn.Parameter(torch.ones(len(hv_models)))

        # Built-in classification head — RefineHD on pre-encoded HVs
        # Uses HVPrototypeHead (not AdaptiveHDClassifier which re-encodes features)
        self.classifier = HVPrototypeHead(
            n_classes=self.cfg.n_classes,
            hv_dim=self.cfg.hv_dim,
            device=self.cfg.device,
        )

        # D2H-AD anomaly head (Ghajari 2026)
        if self.cfg.enable_anomaly:
            self.classifier.enable_anomaly_detection(
                percentile=self.cfg.anomaly_percentile
            )

    def compose(self, hvs: List[torch.Tensor]) -> torch.Tensor:
        """
        Compose a list of hypervectors into one joint HV.

        Args:
            hvs: List of (D,) or (batch, D) hypervectors

        Returns:
            (D,) or (batch, D) composed hypervector
        """
        if not hvs:
            return torch.zeros(self.cfg.hv_dim)

        # Normalise all HVs to 1-D (D,) — squeeze batch dim of size 1
        def _flat(hv: torch.Tensor) -> torch.Tensor:
            if hv.dim() == 2 and hv.shape[0] == 1:
                return hv.squeeze(0)
            if hv.dim() == 2:
                return hv.mean(dim=0)
            return hv

        flat_hvs = [_flat(hv) for hv in hvs]
        strategy = self.cfg.strategy

        if strategy == "bundle":
            stacked = torch.stack([(hv > 0).float() for hv in flat_hvs])
            return (stacked.mean(dim=0) >= 0.5).float()

        elif strategy == "bind":
            bound = []
            for i, hv in enumerate(flat_hvs):
                role = self.role_keys[i]
                b = ((hv > 0) != (role > 0)).float()
                bound.append(b)
            stacked = torch.stack(bound)
            return (stacked.mean(dim=0) >= 0.5).float()

        elif strategy == "weighted":
            w = torch.softmax(self.model_weights, dim=0)
            stacked = torch.stack([(hv > 0).float() for hv in flat_hvs])
            weighted = (stacked * w.unsqueeze(-1)).sum(dim=0)
            return (weighted >= 0.5).float()

        elif strategy == "sequential":
            result = (flat_hvs[0] > 0).float()
            for i, hv in enumerate(flat_hvs[1:], 1):
                result = torch.roll(result, shifts=i, dims=-1)
                result = ((result > 0) != (hv > 0)).float()
            return result

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        """
        Full forward pass: inputs → HV models → compose → joint HV.

        Args:
            inputs: List of tensors, one per HVModel (can differ in shape)

        Returns:
            (D,) or (batch, D) joint hypervector — the decision
        """
        assert len(inputs) == len(self.hv_models), (
            f"Got {len(inputs)} inputs for {len(self.hv_models)} models"
        )
        hvs = [model(x) for model, x in zip(self.hv_models, inputs)]
        return self.compose(hvs)

    def classify(self, joint_hv: torch.Tensor) -> Tuple[int, torch.Tensor]:
        """Classify from a composed hypervector."""
        # AdaptiveHDClassifier expects (n_features,) not (hv_dim,)
        # For HVA, the "features" ARE the hypervector components
        return self.classifier.predict(joint_hv)

    def train_step(
        self,
        inputs: List[torch.Tensor],
        label: int,
    ) -> None:
        """
        Online one-shot training step — no backpropagation through models.

        The HVModels are frozen during HDC training.  Only the
        AdaptiveHDClassifier (RefineHD) updates its prototypes.
        """
        joint_hv = self.forward(inputs)
        self.classifier.train_step(joint_hv, label)

    def add_model(self, hv_model: HVModel) -> None:
        """
        Add a new HVModel to the ensemble at runtime — no retraining.

        The new model gets a random role key.  Its weight starts equal
        to existing models and adjusts via the per-model weight parameter.
        """
        self.hv_models.append(hv_model)
        n = len(self.hv_models)

        # Extend role keys
        new_role = torch.randint(0, 2, (1, self.cfg.hv_dim)).float()
        self.role_keys = torch.cat([self.role_keys, new_role], dim=0)

        # Extend model weights (new model starts with equal weight)
        new_w = self.model_weights.data.mean().unsqueeze(0)
        self.model_weights = nn.Parameter(
            torch.cat([self.model_weights.data, new_w], dim=0)
        )
        logger.info(f"Added HVModel '{hv_model.role_name}' → {n} models total")

    def remove_model(self, role_name: str) -> bool:
        """
        Remove an HVModel by role name at runtime — graceful degradation.

        Returns True if the model was found and removed.
        """
        for i, m in enumerate(self.hv_models):
            if m.role_name == role_name:
                self.hv_models = nn.ModuleList(
                    [m for j, m in enumerate(self.hv_models) if j != i]
                )
                self.role_keys = torch.cat(
                    [self.role_keys[:i], self.role_keys[i + 1:]], dim=0
                )
                idxs = list(range(len(self.model_weights)))
                idxs.pop(i)
                self.model_weights = nn.Parameter(
                    self.model_weights.data[idxs]
                )
                logger.info(f"Removed HVModel '{role_name}' → {len(self.hv_models)} models")
                return True
        return False

    @property
    def total_energy_pJ(self) -> float:
        return sum(m.energy_pJ for m in self.hv_models)

    def __repr__(self) -> str:
        roles = [m.role_name for m in self.hv_models]
        return (
            f"HVComposer(strategy={self.cfg.strategy}, "
            f"models={roles}, hv_dim={self.cfg.hv_dim})"
        )


# ── HVPipeline — end-to-end HyperVector Architecture ─────────────────────────

class HVPipeline:
    """
    End-to-end HyperVector Architecture pipeline.

    This is the top-level interface for the HVA paradigm.  It wires
    together arbitrarily many models from arbitrarily many modalities
    into a single hypervector decision — with no retraining.

    Example (three completely different architectures, no retraining)::

        from hdc.hypervector_architecture import HVPipeline, HVModel, HVModelConfig

        # Wrap any models
        hv_vision = HVModel(resnet, HVModelConfig(role_name="vision", model_output_dim=512))
        hv_audio  = HVModel(wav2vec, HVModelConfig(role_name="audio",  model_output_dim=768))
        hv_snn    = HVModel(rsnn,    HVModelConfig(role_name="spike",  model_output_dim=128))

        # Compose — zero retraining
        pipe = HVPipeline(
            models={"vision": hv_vision, "audio": hv_audio, "spike": hv_snn},
            n_classes=20,
            hv_dim=4096,
            strategy="bundle",
        )

        # Train (online, one sample at a time)
        for inputs, label in stream:
            pipe.train_step(inputs, label)

        # Infer
        joint_hv = pipe.encode({"vision": img, "audio": waveform, "spike": spikes})
        pred = pipe.predict(joint_hv)

        # Add a new model at runtime — no retraining
        hv_text = HVModel(bert, HVModelConfig(role_name="text", model_output_dim=768))
        pipe.add_model("text", hv_text)
    """

    def __init__(
        self,
        models: Dict[str, HVModel],
        n_classes: int = 10,
        hv_dim: int = 4096,
        strategy: str = "bundle",
        enable_anomaly: bool = False,
        device: str = "cpu",
    ):
        self.hv_dim = hv_dim
        self.device = device

        # Named HVModel registry
        self._registry: Dict[str, HVModel] = dict(models)

        # Build composer
        composer_cfg = HVComposerConfig(
            hv_dim=hv_dim,
            strategy=strategy,
            n_classes=n_classes,
            enable_anomaly=enable_anomaly,
            device=device,
        )
        self.composer = HVComposer(
            list(models.values()),
            config=composer_cfg,
        )

        # Role order must match composer's model order
        self._role_order: List[str] = list(models.keys())

    def encode(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode a dict of named inputs into a joint hypervector.

        Missing modalities are skipped — the composer uses whatever is present.
        """
        ordered = [inputs[role] for role in self._role_order if role in inputs]
        models  = [self._registry[role] for role in self._role_order if role in inputs]

        hvs = [m(x) for m, x in zip(models, ordered)]
        return self.composer.compose(hvs)

    def predict(self, joint_hv: torch.Tensor) -> Tuple[int, torch.Tensor]:
        """Classify from a joint hypervector."""
        return self.composer.classify(joint_hv)

    def train_step(
        self, inputs: Dict[str, torch.Tensor], label: int
    ) -> None:
        """Online training step — no backprop through wrapped models."""
        joint_hv = self.encode(inputs)
        self.composer.classifier.train_step(joint_hv, label)

    def add_model(self, role: str, hv_model: HVModel) -> None:
        """Add a new model at runtime — no retraining required."""
        self._registry[role] = hv_model
        self._role_order.append(role)
        self.composer.add_model(hv_model)

    def remove_model(self, role: str) -> bool:
        """Remove a model at runtime — graceful degradation."""
        if role in self._registry:
            del self._registry[role]
            self._role_order.remove(role)
            return self.composer.remove_model(role)
        return False

    def anomaly_score(self, inputs: Dict[str, torch.Tensor]) -> Tuple[float, bool]:
        """D2H-AD anomaly score from joint hypervector."""
        joint_hv = self.encode(inputs)
        return self.composer.classifier.anomaly_score(joint_hv)

    @property
    def n_models(self) -> int:
        return len(self._registry)

    @property
    def roles(self) -> List[str]:
        return list(self._registry)

    @property
    def total_energy_pJ(self) -> float:
        return self.composer.total_energy_pJ

    def encode_batch(
        self,
        batch_inputs: List[Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """
        Encode a batch of multi-modal inputs to joint HVs.

        Args:
            batch_inputs: List of B input dicts, each {role: tensor}

        Returns:
            (B, D) joint hypervector matrix
        """
        hvs = [self.encode(inp) for inp in batch_inputs]
        return torch.stack(hvs)

    def predict_batch(
        self,
        batch_inputs: List[Dict[str, torch.Tensor]],
    ) -> Tuple[List[int], torch.Tensor]:
        """
        Predict class labels for a batch of inputs.

        Returns:
            (labels_list, similarities (B, n_classes))
        """
        joint_hvs = self.encode_batch(batch_inputs)
        labels, all_sims = [], []
        for hv in joint_hvs:
            label, sims = self.predict(hv)
            labels.append(label)
            all_sims.append(sims)
        return labels, torch.stack(all_sims)

    def pipeline_report(self) -> Dict[str, Any]:
        """
        Return a structured report of the pipeline's current state.

        Includes: model count, energy estimate, classifier state.
        """
        return {
            "n_models":         self.n_models,
            "roles":            self.roles,
            "hv_dim":           self.hv_dim,
            "strategy":         self.composer.cfg.strategy,
            "n_classes":        self.composer.cfg.n_classes,
            "total_energy_pJ":  round(self.total_energy_pJ, 4),
            "classifier_counts": (
                self.composer.classifier.counts.tolist()
                if hasattr(self.composer.classifier, "counts")
                else []
            ),
        }

    def __repr__(self) -> str:
        return (
            f"HVPipeline(roles={self.roles}, strategy={self.composer.cfg.strategy}, "
            f"hv_dim={self.hv_dim}, n_classes={self.composer.cfg.n_classes})"
        )


# ── HVScaler — scale from single vector to hypervector architecture ───────────

class HVScaler:
    """
    Progressive scaling from single-vector to hypervector architecture.

    The scaling path (Karunaratne 2020, Cumbo 2026):
        Level 0: single model, vector output, softmax classifier
        Level 1: single HVModel, hypervector output, HDC classifier
        Level 2: 2-4 HVModels, composed HV, HDC classifier
        Level 3: N HVModels (any architectures), composed HV, full HVA

    The scaler wraps an existing model at Level 0 and provides a
    `.upgrade()` method to add HDC layers incrementally.
    """

    def __init__(self, base_model: Callable, output_dim: int, n_classes: int, hv_dim: int = 4096):
        self.base_model = base_model
        self.output_dim = output_dim
        self.n_classes = n_classes
        self.hv_dim = hv_dim
        self.level = 0
        self._pipeline: Optional[HVPipeline] = None

    def upgrade(self, role: str = "base") -> HVPipeline:
        """
        Upgrade the base model to Level 1 HVA.

        Wraps the existing model with an AutoencoderBridge and
        attaches an AdaptiveHDClassifier head.  No retraining of the
        base model.
        """
        cfg = HVModelConfig(
            hv_dim=self.hv_dim,
            model_output_dim=self.output_dim,
            role_name=role,
        )
        hv_model = HVModel(self.base_model, config=cfg)
        self._pipeline = HVPipeline(
            models={role: hv_model},
            n_classes=self.n_classes,
            hv_dim=self.hv_dim,
        )
        self.level = 1
        logger.info(f"Upgraded to HVA Level 1: {hv_model}")
        return self._pipeline

    def add_peer(self, model: Callable, output_dim: int, role: str) -> HVPipeline:
        """
        Add a peer model to an existing Level 1 pipeline, reaching Level 2+.

        The new model is wrapped and added to the composition ensemble.
        The existing AdaptiveHDClassifier continues to learn; no retraining.
        """
        assert self._pipeline is not None, "Call upgrade() first"
        cfg = HVModelConfig(hv_dim=self.hv_dim, model_output_dim=output_dim, role_name=role)
        hv_peer = HVModel(model, config=cfg)
        self._pipeline.add_model(role, hv_peer)
        self.level = max(self.level, 2)
        logger.info(f"Added peer model '{role}' → Level {self.level}")
        return self._pipeline

    @property
    def pipeline(self) -> Optional[HVPipeline]:
        return self._pipeline


# ── LayerBinarizer ────────────────────────────────────────────────────────────

class LayerBinarizer(nn.Module):
    """Convert any neural network layer's activations to a balanced binary HV.

    The user's insight: "Any NN can be converted to a binary vector. Layer each
    neuron as a position and it can become a vector. Process of convergence —
    neuron is activated or not."

    The key correctness requirement (from concentration.py):
    The resulting binary vector must have ≈50% ones to sit in the same
    statistical space as the random coin-flip basis.  Thresholding at zero
    (which ``HVModel(bypass_bridge=True)`` used to do) fails for:
      - ReLU outputs (always ≥ 0 → far more than 50% ones when dim is large)
      - Sigmoid/softmax outputs (centred around 0.5, not 0)
      - Any shifted distribution

    The correct threshold is the **running mean** of the activation distribution,
    updated online as new samples arrive.

    Usage::

        resnet = torchvision.models.resnet18(pretrained=True)
        binarizer = LayerBinarizer(
            model=resnet,
            layer_name="layer4",   # hook on this layer
            hv_dim=8192,           # canonical 2^13
        )
        img = torch.randn(1, 3, 224, 224)
        hv = binarizer(img)   # (1, 8192) balanced binary HV

    If ``hv_dim`` is larger than the layer's output dimension a random
    projection (fixed, not learned) expands the vector.  If smaller, it
    truncates — though truncation is not recommended as it loses capacity.

    Reference:
        Concentration of measure in binary HDC: hdc/concentration.py
        Sutor et al. 2018 arXiv:1806.10755
    """

    def __init__(
        self,
        model: nn.Module,
        layer_name: Optional[str] = None,
        hv_dim: int = DIM_CANONICAL,
        ema_alpha: float = 0.01,
        seed: Optional[int] = None,
    ):
        """
        Args:
            model: Any nn.Module
            layer_name: Name of the sub-module to hook.  If None, hooks the
                        final output of the full model.
            hv_dim: Target HV dimension (default: 2^13 = 8192)
            ema_alpha: EMA decay for running mean update (0.01 = slow, stable)
            seed: Seed for the random projection matrix (if needed)
        """
        super().__init__()
        self.model = model
        self.layer_name = layer_name
        self.hv_dim = hv_dim
        self.ema_alpha = ema_alpha

        self._hook_output: Optional[torch.Tensor] = None
        self._hook_handle = None
        self._running_threshold: Optional[torch.Tensor] = None
        self._n_updates: int = 0

        # Random projection (fixed, no grad) — used if layer_dim != hv_dim
        self._proj: Optional[torch.Tensor] = None
        self._seed = seed

        if layer_name is not None:
            self._register_hook(layer_name)

    def _binarize(self, act: torch.Tensor) -> torch.Tensor:
        return _binarize_to_mean(act, self._running_threshold)

    def _register_hook(self, layer_name: str) -> None:
        """Register a forward hook on the named layer."""
        target = dict(self.model.named_modules()).get(layer_name)
        if target is None:
            raise ValueError(
                f"Layer '{layer_name}' not found in model. "
                f"Available: {list(dict(self.model.named_modules()).keys())[:10]}…"
            )

        def _hook(module, input, output):
            self._hook_output = output.detach()

        self._hook_handle = target.register_forward_hook(_hook)

    def remove_hook(self) -> None:
        """Remove the forward hook (call when done to free memory)."""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def _get_projection(self, feature_dim: int, device: torch.device) -> torch.Tensor:
        """Get (or build) the random projection matrix."""
        if self._proj is not None and self._proj.shape == (feature_dim, self.hv_dim):
            return self._proj.to(device)
        g = torch.Generator(device=device)
        if self._seed is not None:
            g.manual_seed(self._seed)
        # Random ±1 projection (maintains distances better than Gaussian)
        proj = (torch.rand(feature_dim, self.hv_dim, generator=g, device=device) > 0.5).float() * 2 - 1
        self._proj = proj
        return proj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the model, extract the target layer's activations, binarize.

        Args:
            x: Input tensor for the wrapped model

        Returns:
            (batch, hv_dim) balanced binary hypervector
        """
        if self.layer_name is not None:
            self._hook_output = None
            _ = self.model(x)
            act = self._hook_output
            if act is None:
                raise RuntimeError(f"Hook on '{self.layer_name}' did not fire")
        else:
            act = self.model(x)

        # Flatten to (batch, features)
        if act.dim() > 2:
            act = act.flatten(start_dim=1)
        if act.dim() == 1:
            act = act.unsqueeze(0)

        batch, feat_dim = act.shape

        # Project to hv_dim if needed
        if feat_dim != self.hv_dim:
            proj = self._get_projection(feat_dim, act.device)
            act = act @ proj  # (batch, hv_dim)

        # Update running median threshold (per-dimension).
        # Median guarantees 50% ones for any activation distribution, including
        # asymmetric ones like ReLU (mean-based binarization fails for ReLU).
        batch_median = act.median(dim=0).values.detach()
        if self._running_threshold is None:
            self._running_threshold = batch_median
        else:
            α = self.ema_alpha
            self._running_threshold = (1 - α) * self._running_threshold + α * batch_median
        self._n_updates += 1

        # Binarize at running median threshold
        hv = _binarize_to_mean(act, self._running_threshold)
        return hv

    def balance_stats(self) -> Dict[str, float]:
        """Return statistics about HV balance from the running threshold."""
        if self._running_threshold is None:
            return {"status": "not_initialized"}
        # Estimate what fraction of activations are above the running threshold
        # We approximate from the threshold values themselves
        return {
            "n_updates": self._n_updates,
            "threshold_mean": float(self._running_threshold.mean().item()),
            "threshold_std": float(self._running_threshold.std().item()),
            "is_initialized": True,
        }

    @classmethod
    def wrap_final_layer(
        cls,
        model: nn.Module,
        hv_dim: int = DIM_CANONICAL,
        seed: Optional[int] = None,
    ) -> "LayerBinarizer":
        """Convenience: wrap the full model output (no hook, uses final output)."""
        return cls(model=model, layer_name=None, hv_dim=hv_dim, seed=seed)


# ── Demo ──────────────────────────────────────────────────────────────────────

def demo_hva():
    """
    Demonstrate the HyperVector Architecture with three toy models.

    Shows: different architectures, no retraining, runtime add/remove.
    """
    import torch
    torch.manual_seed(42)

    D = 512         # hypervector dimension (use 4096+ in production)
    N_CLASSES = 4

    # Three completely different "models" (toy lambdas here)
    def vision_model(x):     # Simulates ResNet feature extractor
        return torch.nn.functional.normalize(x @ torch.randn(64, 128), dim=-1)

    def sensor_model(x):     # Simulates SNN spike readout
        return (x > 0.5).float() @ torch.randn(32, 64).abs()

    def text_model(x):       # Simulates LLM last-hidden-state
        return torch.tanh(x @ torch.randn(16, 96))

    # Wrap each as an HVModel
    hv_vision = HVModel(
        vision_model,
        HVModelConfig(hv_dim=D, model_output_dim=128, role_name="vision"),
    )
    hv_sensor = HVModel(
        sensor_model,
        HVModelConfig(hv_dim=D, model_output_dim=64, role_name="sensor"),
    )
    hv_text = HVModel(
        text_model,
        HVModelConfig(hv_dim=D, model_output_dim=96, role_name="text"),
    )

    # Build pipeline — three architectures, one hypervector, no retraining
    pipe = HVPipeline(
        models={"vision": hv_vision, "sensor": hv_sensor, "text": hv_text},
        n_classes=N_CLASSES,
        hv_dim=D,
        strategy="bundle",
    )
    print(f"\nHVPipeline: {pipe}")

    # Online training (100 samples, no backprop)
    print("\nTraining (100 samples, no backprop)...")
    for i in range(100):
        label = i % N_CLASSES
        inputs = {
            "vision": torch.randn(1, 64),
            "sensor": torch.randn(1, 32),
            "text":   torch.randn(1, 16),
        }
        pipe.train_step(inputs, label)

    # Evaluate
    correct = 0
    for i in range(50):
        label = i % N_CLASSES
        inputs = {
            "vision": torch.randn(1, 64),
            "sensor": torch.randn(1, 32),
            "text":   torch.randn(1, 16),
        }
        hv = pipe.encode(inputs)
        pred, _ = pipe.predict(hv)
        correct += int(pred == label)
    print(f"  Accuracy (3 models): {correct}/50 = {correct/50:.0%}")

    # Add a fourth model AT RUNTIME — no retraining
    def lidar_model(x):
        return x @ torch.randn(8, 48)

    hv_lidar = HVModel(
        lidar_model,
        HVModelConfig(hv_dim=D, model_output_dim=48, role_name="lidar"),
    )
    pipe.add_model("lidar", hv_lidar)
    print(f"\nAdded 'lidar' model at runtime → {pipe.n_models} models, no retraining")

    # Remove sensor model — graceful degradation
    pipe.remove_model("sensor")
    print(f"Removed 'sensor' model → {pipe.n_models} models, still running")

    # HVScaler: upgrade any existing model to HVA incrementally
    print("\nHVScaler: upgrading existing model to HVA...")
    scaler = HVScaler(vision_model, output_dim=128, n_classes=N_CLASSES, hv_dim=D)
    lvl1 = scaler.upgrade(role="vision")
    lvl2 = scaler.add_peer(sensor_model, output_dim=64, role="sensor")
    print(f"  Level {scaler.level}: {lvl2}")

    print(f"\n  Total energy: {pipe.total_energy_pJ:.1f} pJ")
    print("  HVA demo complete ✓")


if __name__ == "__main__":
    demo_hva()


def test_hypervector_architecture():
    import torch, torch.nn as nn
    from hdc.hypervector_architecture import LayerBinarizer, HVScaler, HVPrototypeHead
    lb = LayerBinarizer(nn.Linear(32, 16))
    out = lb(torch.randn(4, 32))
    assert out.shape[0] == 4 and out.max() <= 1.0 and out.min() >= 0.0
    print(f"hypervector_architecture: ✅ LayerBinarizer {out.shape}")

# ═══════════════════════════════════════════════════════════════════════════════
# Elite Enhancements — EliteHVComposer
# ═══════════════════════════════════════════════════════════════════════════════

class EliteHVComposer:
    """
    Elite replacement for HVComposer.

    Improvements over baseline:
      - Per-model weights learned online via EMA: models that agree with the
        ensemble get higher weight; those that consistently diverge are
        down-weighted automatically (no manual tuning).
      - Health scores: call degrade(idx) when a sensor is known to fail;
        health feeds into effective weight so failing models are suppressed.
      - Cross-modal consistency check: cross_modal_confidence() returns the
        mean pairwise Hamming agreement over recent calls — low values
        indicate sensor disagreement or failure.

    Args:
        n_models: Number of models in ensemble
        hv_dim: Hypervector dimension
        ema_alpha: EMA decay for weight and consistency updates
        health_decay: Not currently used (health is set explicitly via degrade/recover)
    """

    def __init__(
        self,
        n_models: int,
        hv_dim: int,
        ema_alpha: float = 0.05,
        health_decay: float = 0.99,
    ):
        self.hv_dim = hv_dim
        self.ema_alpha = ema_alpha

        self.weights     = torch.ones(n_models) / n_models
        self.health      = torch.ones(n_models)
        self.consistency = torch.ones(n_models) * 0.5
        self._agreement_history: List[float] = []

    def compose(
        self,
        hvs: List[torch.Tensor],
        return_meta: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Compose HVs with health-weighted majority vote.

        Args:
            hvs: List of (D,) HVs from each model
            return_meta: If True return (composed, metadata_dict)
        """
        n = len(hvs)
        if n == 0:
            return torch.zeros(self.hv_dim)

        w = self.weights[:n].clone().to(hvs[0].device)
        w = w / (w.sum() + 1e-8)
        health = self.health[:n].to(hvs[0].device)
        effective_w = (w * health)
        effective_w = effective_w / (effective_w.sum() + 1e-8)

        stacked = torch.stack([(hv > 0).float() for hv in hvs])   # (n, D)
        weighted = (stacked * effective_w.unsqueeze(-1)).sum(dim=0)  # (D,)
        composed = _majority(weighted)

        if n >= 2:
            pairwise = [
                float(_hamming(hvs[i].unsqueeze(0), hvs[j].unsqueeze(0)).item())
                for i in range(n) for j in range(i + 1, n)
            ]
            mean_agreement = sum(pairwise) / len(pairwise)
        else:
            mean_agreement = 1.0
        self._agreement_history.append(mean_agreement)

        for i in range(n):
            ag_i = float(_hamming(hvs[i].unsqueeze(0), composed.unsqueeze(0)).item())
            self.consistency[i] = (1 - self.ema_alpha) * self.consistency[i] + self.ema_alpha * ag_i

        if return_meta:
            return composed, {
                "weights":               self.weights[:n].clone(),
                "health":                self.health[:n].clone(),
                "consistency":           self.consistency[:n].clone(),
                "cross_modal_agreement": mean_agreement,
            }
        return composed

    def update_weights(self, loss: Optional[torch.Tensor] = None):
        """Update per-model weights proportional to ensemble consistency."""
        self.weights = (1 - self.ema_alpha) * self.weights + self.ema_alpha * self.consistency
        self.weights = (self.weights / (self.weights.sum() + 1e-8)).clamp(min=0.01)

    def degrade(self, model_idx: int, penalty: float = 0.1):
        """Reduce health of a malfunctioning model."""
        self.health[model_idx] = max(0.01, self.health[model_idx] * (1.0 - penalty))

    def recover(self, model_idx: int, boost: float = 0.1):
        self.health[model_idx] = min(1.0, self.health[model_idx] + boost)

    def cross_modal_confidence(self) -> float:
        """Mean pairwise agreement over recent compose() calls."""
        if not self._agreement_history:
            return 1.0
        recent = self._agreement_history[-min(10, len(self._agreement_history)):]
        return sum(recent) / len(recent)


if __name__ == "__main__":
    test_hypervector_architecture()
