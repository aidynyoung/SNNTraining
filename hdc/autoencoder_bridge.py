"""
hdc/autoencoder_bridge.py
==========================
Autoencoder Bridge: NN → HDC translation layer.

Implements the hybrid architecture from:
    Ghajari (2026) "D2H-AD: A Hybrid Model Utilizing Hyperdimensional Computing
    for Advanced Anomaly Detection" IEEE Access, 14.

    Cumbo (2026) "Designing vector-symbolic architectures for biomedical
    applications" PeerJ Computer Science, 12. doi:10.7717/peerj-cs.3682

Key insight: The "wrong way" is trying to replace Neural Networks (NNs).
The emergent "right way" is a hybrid architecture where NNs act as encoders
and HDC acts as the reasoning/memory layer. Autoencoders can be used to
"solve" the translation from raw data to hyperdimensional space, allowing
different models to be combined into a single cohesive decision-making unit.

Multimodality is free: because hypervectors use the same mathematical space
regardless of source, a vision hypervector and a lidar hypervector can be
bound or bundled together without retraining the entire system.

This module provides:
- AutoencoderBridge: NN encoder → HDC hypervector translator
- MultimodalFusion: Fuse multiple modalities via binding/bundling
- HybridClassifier: NN encodes, HDC classifies (no backprop needed)
- CrossModalBinding: Bind hypervectors from different modalities

Usage:
    from hdc.autoencoder_bridge import AutoencoderBridge, MultimodalFusion

    bridge = AutoencoderBridge(input_dim=784, hdc_dim=10000)
    hv = bridge.encode(image_tensor)       # NN → HDC

    fusion = MultimodalFusion(hdc_dim=10000)
    fused = fusion.fuse(vision_hv, lidar_hv)  # Bind modalities
"""

import torch
import torch.nn as nn
import logging
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any

logger = logging.getLogger(__name__)


@dataclass
class BridgeConfig:
    """Configuration for the autoencoder bridge."""
    input_dim: int = 784           # Raw input dimension (e.g., 28×28 image)
    hdc_dim: int = 10000           # Target hypervector dimension
    hidden_dim: int = 512          # Bottleneck hidden dimension
    encoding_layers: int = 3       # Number of encoding layers
    activation: str = "relu"       # Activation function
    dropout: float = 0.1           # Dropout rate
    use_batch_norm: bool = True    # Use batch normalization
    learning_rate: float = 1e-3    # Learning rate for training
    device: str = "cpu"


class EncoderNN(nn.Module):
    """Neural network encoder that maps raw data to hyperdimensional space."""

    def __init__(self, config: BridgeConfig):
        super().__init__()
        self.config = config

        layers = []
        dims = [config.input_dim] + [config.hidden_dim] * (config.encoding_layers - 1)

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if config.use_batch_norm:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
            if config.activation == "relu":
                layers.append(nn.ReLU())
            elif config.activation == "tanh":
                layers.append(nn.Tanh())
            layers.append(nn.Dropout(config.dropout))

        # Final projection to HDC dimension
        layers.append(nn.Linear(dims[-1], config.hdc_dim))
        layers.append(nn.Tanh())  # Output in [-1, 1] for binarization

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode raw input to hyperdimensional space."""
        return self.net(x)


class DecoderNN(nn.Module):
    """Neural network decoder that reconstructs raw data from hypervector."""

    def __init__(self, config: BridgeConfig):
        super().__init__()
        self.config = config

        dims = [config.hdc_dim] + [config.hidden_dim] * (config.encoding_layers - 1)

        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if config.use_batch_norm:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.dropout))

        layers.append(nn.Linear(dims[-1], config.input_dim))
        layers.append(nn.Sigmoid())  # Output in [0, 1]

        self.net = nn.Sequential(*layers)

    def forward(self, hv: torch.Tensor) -> torch.Tensor:
        """Reconstruct raw data from hypervector."""
        return self.net(hv)


class AutoencoderBridge:
    """
    Autoencoder Bridge: NN → HDC translation layer.

    Trains an autoencoder to translate raw data into hyperdimensional space.
    The encoder maps inputs to HVs, the decoder reconstructs them.
    Once trained, only the encoder is used for inference.

    This "solves" the translation from raw data to hyperdimensional space,
    allowing different models to be combined into a single decision-making unit.
    """

    def __init__(self, config: Optional[BridgeConfig] = None):
        self.config = config or BridgeConfig()
        self.device = torch.device(self.config.device)

        self.encoder = EncoderNN(self.config).to(self.device)
        self.decoder = DecoderNN(self.config).to(self.device)
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=self.config.learning_rate,
        )
        self.criterion = nn.MSELoss()
        self.trained = False

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode raw input to a binary hypervector.

        Uses eval mode to avoid BatchNorm issues with single samples.

        Args:
            x: (batch, input_dim) or (input_dim,) raw input

        Returns:
            (batch, hdc_dim) or (hdc_dim,) binary hypervector
        """
        was_1d = x.dim() == 1
        if was_1d:
            x = x.unsqueeze(0)
        x = x.to(self.device)
        was_training = self.encoder.training
        self.encoder.eval()
        with torch.no_grad():
            hv = self.encoder(x)
            hv_bin = (hv > 0).float()
        if was_training:
            self.encoder.train()
        return hv_bin.squeeze(0) if was_1d else hv_bin

    def decode(self, hv: torch.Tensor) -> torch.Tensor:
        """
        Decode a hypervector back to raw input space.

        Args:
            hv: (batch, hdc_dim) or (hdc_dim,) hypervector

        Returns:
            (batch, input_dim) or (input_dim,) reconstruction
        """
        was_1d = hv.dim() == 1
        if was_1d:
            hv = hv.unsqueeze(0)
        hv = hv.to(self.device)
        with torch.no_grad():
            recon = self.decoder(hv)
        return recon.squeeze(0) if was_1d else recon

    def train_step(self, x: torch.Tensor) -> float:
        """
        Single training step.

        Args:
            x: (batch, input_dim) training batch

        Returns:
            Loss value
        """
        x = x.to(self.device)
        self.optimizer.zero_grad()

        hv = self.encoder(x)
        recon = self.decoder(hv)
        loss = self.criterion(recon, x)

        # Add HDC regularization: encourage binary-like activations
        hdc_reg = 0.01 * (hv.abs() - 1.0).pow(2).mean()
        loss = loss + hdc_reg

        loss.backward()
        self.optimizer.step()

        self.trained = True
        return float(loss)

    def fit(
        self,
        dataloader: torch.utils.data.DataLoader,
        epochs: int = 10,
        verbose: bool = True,
    ) -> List[float]:
        """
        Train the autoencoder bridge.

        Args:
            dataloader: DataLoader providing training batches
            epochs: Number of training epochs
            verbose: Print progress

        Returns:
            List of epoch losses
        """
        epoch_losses = []
        for epoch in range(epochs):
            total_loss = 0.0
            n_batches = 0
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                else:
                    x = batch
                loss = self.train_step(x)
                total_loss += loss
                n_batches += 1
            avg_loss = total_loss / max(n_batches, 1)
            epoch_losses.append(avg_loss)
            if verbose:
                logger.info(f"Epoch {epoch+1}/{epochs}  loss={avg_loss:.6f}")
        return epoch_losses

    def save(self, path: str):
        """Save bridge weights."""
        torch.save({
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "config": self.config,
        }, path)

    def load(self, path: str):
        """Load bridge weights."""
        checkpoint = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(checkpoint["encoder"])
        self.decoder.load_state_dict(checkpoint["decoder"])
        self.trained = True

    def reconstruction_error(self, x: torch.Tensor) -> float:
        """
        Measure reconstruction quality: MSE between input and auto-decoded output.

        Useful for:
          - Quality control: high error = the bridge hasn't seen this data type
          - Distribution shift: increasing error over time = input changed
          - Compression fidelity: measures information loss in the HV encoding

        Args:
            x: (input_dim,) or (B, input_dim) raw input

        Returns:
            Mean squared reconstruction error
        """
        hv  = self.encode(x)
        rec = self.decode(hv)
        if x.dim() == 1:
            return float(((rec - x.float().to(rec.device))**2).mean().item())
        return float(((rec - x.float().to(rec.device))**2).mean().item())

    def bridge_health(self, sample: Optional[torch.Tensor] = None) -> dict:
        """
        Health report: training status, weight norms, optional reconstruction error.

        If `sample` is provided, includes reconstruction error.
        enc_weight_norm > 100 → encoder may be saturating (lr too high or overfitting).
        """
        enc_norm = float(self.encoder.fc1.weight.data.norm().item()) if hasattr(self.encoder, "fc1") else None
        dec_norm = float(self.decoder.fc1.weight.data.norm().item()) if hasattr(self.decoder, "fc1") else None
        rec_err  = self.reconstruction_error(sample) if sample is not None else None
        return {
            "trained":         self.trained,
            "input_dim":       self.config.input_dim,
            "hdc_dim":         self.config.hdc_dim,
            "enc_weight_norm": round(enc_norm, 4) if enc_norm is not None else None,
            "dec_weight_norm": round(dec_norm, 4) if dec_norm is not None else None,
            "reconstruction_error": round(rec_err, 6) if rec_err is not None else None,
        }

    def __repr__(self) -> str:
        return (
            f"AutoencoderBridge({self.config.input_dim}→{self.config.hdc_dim}, "
            f"trained={self.trained})"
        )


class MultimodalFusion:
    """
    Multimodal Fusion: combine hypervectors from different modalities.

    Because hypervectors use the same mathematical space regardless of source,
    a vision hypervector and a lidar hypervector can be bound or bundled
    together without retraining the entire system.

    Fusion strategies:
    - bind: XOR-based binding (preserves individual identity)
    - bundle: Majority-based bundling (creates prototype)
    - weighted: Weighted combination (confidence-weighted)
    - sequential: Concatenate and project
    """

    def __init__(self, hdc_dim: int = 10000):
        self.hdc_dim = hdc_dim

    def bind(self, *hvs: torch.Tensor) -> torch.Tensor:
        """
        Bind multiple modality hypervectors via XOR.

        Binding preserves individual identity — each modality's contribution
        can be recovered via release operation.

        Args:
            *hvs: Variable number of (dim,) hypervectors

        Returns:
            (dim,) bound hypervector
        """
        if not hvs:
            return torch.zeros(self.hdc_dim)
        result = (hvs[0] > 0).float()
        for hv in hvs[1:]:
            result = (result > 0) != (hv > 0)
            result = result.float()
        return result

    def bundle(self, *hvs: torch.Tensor) -> torch.Tensor:
        """
        Bundle multiple modality hypervectors via majority.

        Bundling creates a prototype that captures shared structure.

        Args:
            *hvs: Variable number of (dim,) hypervectors

        Returns:
            (dim,) bundled hypervector
        """
        if not hvs:
            return torch.zeros(self.hdc_dim)
        stacked = torch.stack([(hv > 0).float() for hv in hvs])
        return (stacked.mean(dim=0) >= 0.5).float()

    def weighted_fusion(
        self, hvs: List[torch.Tensor], weights: List[float]
    ) -> torch.Tensor:
        """
        Weighted fusion: confidence-weighted combination.

        Args:
            hvs: List of (dim,) hypervectors
            weights: List of confidence weights (must sum to 1.0)

        Returns:
            (dim,) fused hypervector
        """
        assert len(hvs) == len(weights), "Mismatched lengths"
        total = sum(weights)
        if total == 0.0:
            return self.bundle(*hvs)
        w = torch.tensor(weights) / total
        stacked = torch.stack([(hv > 0).float() for hv in hvs])
        weighted = (stacked * w.unsqueeze(1)).sum(dim=0)
        return (weighted >= 0.5).float()

    def fuse(
        self,
        *modalities: Dict[str, Any],
        strategy: str = "bind",
    ) -> torch.Tensor:
        """
        Fuse multiple modalities using the specified strategy.

        Args:
            *modalities: Dicts with 'hv' key containing hypervectors
            strategy: 'bind', 'bundle', or 'weighted'

        Returns:
            (dim,) fused hypervector
        """
        hvs = [m["hv"] for m in modalities]
        if strategy == "bind":
            return self.bind(*hvs)
        elif strategy == "bundle":
            return self.bundle(*hvs)
        elif strategy == "weighted":
            weights = [m.get("weight", 1.0) for m in modalities]
            return self.weighted_fusion(hvs, weights)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")


class HybridClassifier:
    """
    Hybrid Classifier: NN encodes, HDC classifies.

    Implements the Ghajari (2026) paradigm:
    - NN acts as encoder (feature extractor)
    - HDC acts as reasoning/memory layer (classifier)
    - No backpropagation needed for HDC layer
    - One-shot or few-shot learning via bundling

    This is the "right way" — hybrid, not replacement.
    """

    def __init__(
        self,
        bridge: AutoencoderBridge,
        n_classes: int,
        hdc_dim: int = 10000,
    ):
        self.bridge = bridge
        self.n_classes = n_classes
        self.hdc_dim = hdc_dim

        # Class prototype hypervectors (one-shot learned via bundling)
        self.class_prototypes: Optional[torch.Tensor] = None
        self.class_counts: torch.Tensor = torch.zeros(n_classes)
        self.class_labels: List[str] = [f"class_{i}" for i in range(n_classes)]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode raw input to hypervector via NN bridge."""
        return self.bridge.encode(x)

    def learn_one_shot(
        self, x: torch.Tensor, label: int
    ) -> None:
        """
        One-shot learning: encode and bundle into class prototype.

        No gradient descent needed — just bundling (element-wise addition).

        Args:
            x: (input_dim,) raw input
            label: Class label (0 to n_classes-1)
        """
        hv = self.encode(x)
        if self.class_prototypes is None:
            self.class_prototypes = torch.zeros(self.n_classes, self.hdc_dim)

        # Bundle: running average of hypervectors for this class
        count = self.class_counts[label]
        self.class_prototypes[label] = (
            (self.class_prototypes[label] * count + hv) / (count + 1)
        )
        self.class_prototypes[label] = (self.class_prototypes[label] > 0.5).float()
        self.class_counts[label] += 1

    def learn_few_shot(
        self, samples: torch.Tensor, labels: torch.Tensor
    ) -> None:
        """
        Few-shot learning: bundle multiple samples into class prototypes.

        Args:
            samples: (n_samples, input_dim) raw inputs
            labels: (n_samples,) class labels
        """
        for x, lbl in zip(samples, labels):
            self.learn_one_shot(x, int(lbl))

    def predict(self, x: torch.Tensor) -> Tuple[int, float]:
        """
        Predict class using HDC similarity search.

        Args:
            x: (input_dim,) raw input

        Returns:
            (predicted_class, confidence)
        """
        if self.class_prototypes is None:
            return (0, 0.0)

        hv = self.encode(x)
        hv_bin = (hv > 0).float()

        # Hamming distance to each class prototype
        distances = (self.class_prototypes != hv_bin.unsqueeze(0)).sum(dim=1).float()
        similarities = 1.0 - (distances / self.hdc_dim)

        pred = int(similarities.argmax())
        confidence = float(similarities[pred])
        return (pred, confidence)

    def predict_batch(self, x: torch.Tensor) -> torch.Tensor:
        """Predict for a batch of inputs."""
        hvs = self.encode(x)
        hvs_bin = (hvs > 0).float()
        distances = (self.class_prototypes.unsqueeze(0) != hvs_bin.unsqueeze(1)).sum(dim=2)
        return distances.argmin(dim=1)

    def accuracy(self, samples: torch.Tensor, labels: torch.Tensor) -> float:
        """Compute classification accuracy."""
        preds = self.predict_batch(samples)
        correct = (preds == labels).sum().item()
        return correct / len(labels)

    def __repr__(self) -> str:
        return (
            f"HybridClassifier(NN→HDC, {self.n_classes} classes, "
            f"prototypes={'yes' if self.class_prototypes is not None else 'no'})"
        )


class CrossModalBinding:
    """
    Cross-modal binding via role-filler hypervectors.

    Implements the Cumbo (2026) "multimodality is free" principle:
    each modality gets a fixed random role hypervector; data HVs are
    bound (XOR) to their role before being bundled into a joint
    representation.  The role allows the fusion to be reversed — given
    the bundle, binding with the role again approximately recovers the
    original modality HV (up to noise from other modalities).

    Usage::
        binding = CrossModalBinding(hdc_dim=10000, modalities=["vision", "lidar"])
        joint = binding.encode({"vision": vision_hv, "lidar": lidar_hv})
        vision_approx = binding.decode(joint, "vision")
    """

    def __init__(self, hdc_dim: int = 10000, modalities: Optional[List[str]] = None):
        self.hdc_dim = hdc_dim
        self._roles: Dict[str, torch.Tensor] = {}
        for name in (modalities or []):
            self.add_modality(name)

    def add_modality(self, name: str, seed: Optional[int] = None) -> torch.Tensor:
        """Register a new modality and generate its role hypervector."""
        if name in self._roles:
            return self._roles[name]
        g = torch.Generator()
        if seed is not None:
            g.manual_seed(seed)
        role = torch.randint(0, 2, (self.hdc_dim,), generator=g).float()
        self._roles[name] = role
        return role

    def bind(self, name: str, hv: torch.Tensor) -> torch.Tensor:
        """Bind a modality HV to its role: role XOR hv."""
        if name not in self._roles:
            self.add_modality(name)
        role = self._roles[name].to(hv.device)
        return ((role > 0) != (hv > 0)).float()

    def encode(self, modality_hvs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode a dict of modality hypervectors into a single joint HV.

        Each modality HV is bound to its role, then all are bundled
        via majority vote.

        Args:
            modality_hvs: {name: (hdc_dim,) HV} for each modality

        Returns:
            (hdc_dim,) joint hypervector
        """
        bound = [self.bind(name, hv) for name, hv in modality_hvs.items()]
        stacked = torch.stack(bound)
        return (stacked.mean(dim=0) >= 0.5).float()

    def decode(self, joint_hv: torch.Tensor, name: str) -> torch.Tensor:
        """
        Approximately recover a modality HV from the joint representation.

        Binding the joint HV with the role again cancels the XOR and
        leaves a noisy version of the original modality HV.

        Args:
            joint_hv: (hdc_dim,) joint hypervector
            name: Modality to extract

        Returns:
            (hdc_dim,) approximate modality HV
        """
        return self.bind(name, joint_hv)

    def __repr__(self) -> str:
        return f"CrossModalBinding(dim={self.hdc_dim}, modalities={list(self._roles)})"


# ── Test ──────────────────────────────────────────────────────────────────────

def test_autoencoder_bridge():
    """Verify autoencoder bridge operations."""
    torch.manual_seed(42)
    dim = 100
    hdc_dim = 1000

    bridge = AutoencoderBridge(BridgeConfig(
        input_dim=dim, hdc_dim=hdc_dim, hidden_dim=64, encoding_layers=2
    ))

    # Test encode/decode (use batch of 2 to avoid BatchNorm issues)
    x = torch.randn(2, dim)
    hv = bridge.encode(x)
    assert hv.shape == (2, hdc_dim), f"HV shape: {hv.shape}"
    assert hv.eq(0).any() and hv.eq(1).any(), "HV should be binary"

    recon = bridge.decode(hv)
    assert recon.shape == (2, dim), f"Recon shape: {recon.shape}"

    # Test single input (eval mode)
    bridge.encoder.eval()
    x_single = torch.randn(dim)
    hv_single = bridge.encode(x_single)
    assert hv_single.shape == (hdc_dim,), f"Single HV shape: {hv_single.shape}"
    bridge.encoder.train()

    # Test multimodal fusion
    fusion = MultimodalFusion(hdc_dim=hdc_dim)
    hv1 = torch.randint(0, 2, (hdc_dim,)).float()
    hv2 = torch.randint(0, 2, (hdc_dim,)).float()

    bound = fusion.bind(hv1, hv2)
    assert bound.shape == (hdc_dim,), f"Bound shape: {bound.shape}"

    bundled = fusion.bundle(hv1, hv2)
    assert bundled.shape == (hdc_dim,), f"Bundled shape: {bundled.shape}"

    # Test hybrid classifier
    classifier = HybridClassifier(bridge, n_classes=3, hdc_dim=hdc_dim)

    # One-shot learning
    for i in range(3):
        sample = torch.randn(dim)
        classifier.learn_one_shot(sample, i)

    # Test prediction
    pred, conf = classifier.predict(torch.randn(dim))
    assert 0 <= pred < 3, f"Prediction out of range: {pred}"
    assert 0.0 <= conf <= 1.0, f"Confidence out of range: {conf}"

    print(f"  Bridge: {bridge}")
    print(f"  HV shape: {hv.shape}, binary: {bool(hv.eq(0).any() and hv.eq(1).any())}")
    print(f"  Fusion: bind={bound.shape}, bundle={bundled.shape}")
    print(f"  Classifier: {classifier}")
    print(f"  Prediction: class {pred}, confidence {conf:.3f}")
    print("  ✓ All autoencoder bridge tests pass")


if __name__ == "__main__":
    test_autoencoder_bridge()
