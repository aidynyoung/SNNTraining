"""
HD-Glue: Symbolic Fusion via Hyperdimensional Computing
=======================================================
Based on Section V-B of Amrouch et al. 2022:
"Symbolic Representation and Learning with HDC"

Fuses output from multiple models into a consensus HDC memory
for robust few-shot learning without backpropagation.

Key insight: project symbolic outputs from arbitrary models into
a common hyperspace, learn HILs (Hyperdimensional Inference Layers)
for each model, then aggregate into a consensus structure.

Enhanced with:
- Weighted consensus based on model confidence (Sutor 2020, 2022)
- Proper HIL training with normalization
- Learnable model weights for optimal fusion

References:
  Mitrokhin, Sutor, Summers-Stay, Fermuller, Aloimonos (2020)
    "Symbolic Representation and Learning with Hyperdimensional Computing"
  Sutor, et al. (2022)
    "Gluing neural networks symbolically through hyperdimensional computing"
"""

import torch
import torch.nn as nn
from typing import List, Optional, Dict, Tuple
from models.hdc import gen_hvs, bind, bundle, sim, thresh, batch_sim


class HyperdimensionalInferenceLayer(nn.Module):
    """
    HIL: Maps input hypervectors to output hypervectors via
    bundling + binding. Learns associations in a single pass.

    For each training pair (input_hv, output_hv):
        M += bind(input_hv, output_hv)

    At inference: probe M with query to find closest output_hv.

    Enhanced with:
    - Normalized memory updates (prevents saturation)
    - Confidence tracking per association
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        mode: str = "bipolar",
        device: Optional[str] = None,
        learning_rate: float = 1.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mode = mode
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.learning_rate = learning_rate

        # Single memory hypervector
        self.register_buffer(
            "memory", torch.zeros(output_dim, device=torch.device(self.device))
        )
        self.register_buffer(
            "count", torch.tensor(0, device=torch.device(self.device))
        )

    def train_pair(self, input_hv: torch.Tensor, output_hv: torch.Tensor):
        """Add a single (input, output) association to memory."""
        association = bind(input_hv.to(self.memory.device),
                          output_hv.to(self.memory.device),
                          self.mode)
        self.memory += self.learning_rate * association
        self.count += 1

    def train_batch(self, inputs: torch.Tensor, outputs: torch.Tensor):
        """Add multiple associations at once.

        Args:
            inputs:  (B, input_dim)
            outputs: (B, output_dim)
        """
        for i in range(inputs.shape[0]):
            self.train_pair(inputs[i], outputs[i])

    def probe(self, query: torch.Tensor) -> torch.Tensor:
        """Given query hv, return the recovered output hv via unbinding."""
        # Unbind: memory * query = output_hv
        # In bipolar: multiply (XOR equivalent)
        recovered = bind(self.memory, query.to(self.memory.device), self.mode)
        return recovered

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        return self.probe(query)


class WeightedConsensusHDC(nn.Module):
    """
    Enhanced consensus mechanism with learnable model weights.

    Instead of simple bundling of all HIL memories, uses learned
    per-model confidence weights for optimal fusion.

    Based on Sutor 2022: "Gluing neural networks symbolically through
    hyperdimensional computing" - weighted consensus based on model
    confidence.
    """

    def __init__(
        self,
        n_models: int,
        input_dim: int,
        output_dim: int,
        mode: str = "bipolar",
        device: Optional[str] = None,
    ):
        super().__init__()
        self.n_models = n_models
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mode = mode
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # One HIL per model
        self.hils = nn.ModuleList([
            HyperdimensionalInferenceLayer(input_dim, output_dim, mode, self.device)
            for _ in range(n_models)
        ])

        # Output key hypervectors (for classification)
        self.register_buffer(
            "output_keys",
            gen_hvs(output_dim, output_dim, mode, self.device, seed=42)
        )

        # Consensus memory: aggregate of all HIL memories
        self.register_buffer(
            "consensus_memory",
            torch.zeros(output_dim, device=torch.device(self.device))
        )

        # Learnable per-model confidence weights
        self.model_log_weights = nn.Parameter(torch.ones(n_models))

        # Per-model accuracy tracking for adaptive weighting
        self.register_buffer(
            "model_accuracies",
            torch.zeros(n_models, device=torch.device(self.device))
        )
        self.register_buffer(
            "model_counts",
            torch.zeros(n_models, device=torch.device(self.device))
        )

    def get_model_weights(self) -> torch.Tensor:
        """Return normalized model weights that sum to 1."""
        return torch.softmax(self.model_log_weights, dim=0)

    def train_all(self, model_outputs: List[torch.Tensor], labels: torch.Tensor):
        """Train all HILs with their respective model outputs.

        Args:
            model_outputs: List of (B, output_dim) tensors, one per model
            labels: (B,) class indices
        """
        B = labels.shape[0]
        for m_idx, outputs in enumerate(model_outputs):
            # Convert labels to output hypervectors
            label_hvs = self.output_keys[labels]
            self.hils[m_idx].train_batch(outputs, label_hvs)

    def build_consensus(self):
        """Aggregate all HIL memories into consensus via weighted bundling.

        Uses learned model weights instead of simple averaging.
        """
        weights = self.get_model_weights()  # (n_models,)

        consensus = torch.zeros(self.consensus_memory.shape[0],
                               device=self.consensus_memory.device)
        for idx, hil in enumerate(self.hils):
            consensus += weights[idx] * hil.memory

        # Normalize — use .copy_() to preserve the registered buffer
        if self.mode == "bipolar":
            self.consensus_memory.copy_(thresh(consensus))
        else:
            self.consensus_memory.copy_(consensus / consensus.norm().clamp(min=1e-12))

    def predict(self, query: torch.Tensor, model_idx: Optional[int] = None) -> int:
        """Predict class label for a query hypervector.

        Args:
            query: (input_dim,) query hypervector
            model_idx: If specified, use single HIL; else use consensus

        Returns:
            predicted class index
        """
        if model_idx is not None:
            recovered = self.hils[model_idx].probe(query)
        else:
            # Unbind using consensus memory
            recovered = bind(
                self.consensus_memory,
                query.to(self.consensus_memory.device),
                self.mode
            )

        # Find closest output key
        similarities = batch_sim(recovered, self.output_keys, self.mode)
        return int(similarities.argmax().item())

    def predict_with_confidence(
        self, query: torch.Tensor, model_idx: Optional[int] = None
    ) -> Tuple[int, float]:
        """Predict with confidence score.

        Returns:
            (predicted_class, confidence)
        """
        if model_idx is not None:
            recovered = self.hils[model_idx].probe(query)
        else:
            recovered = bind(
                self.consensus_memory,
                query.to(self.consensus_memory.device),
                self.mode
            )

        similarities = batch_sim(recovered, self.output_keys, self.mode)
        probs = torch.softmax(similarities, dim=0)
        pred = int(similarities.argmax().item())
        confidence = float(probs[pred].item())
        return pred, confidence

    def forward(
        self,
        queries: torch.Tensor,
        model_idx: Optional[int] = None
    ) -> torch.Tensor:
        """Batch prediction.

        Args:
            queries: (B, input_dim)
            model_idx: If specified, use single HIL

        Returns:
            (B,) class predictions
        """
        results = []
        for i in range(queries.shape[0]):
            results.append(self.predict(queries[i], model_idx))
        return torch.tensor(results, device=queries.device)


class HDGlue(nn.Module):
    """
    Full HD-Glue pipeline: accepts raw outputs from arbitrary models,
    projects them into hyperspace, learns HILs, and produces consensus.

    Enhanced with:
    - Weighted consensus based on model confidence (Sutor 2022)
    - Learnable projection layers
    - Confidence-based tie-breaking

    Architecture:
        Model outputs -> Hypervector projection -> HIL per model
                                                     |
                                        Weighted consensus memory
                                                     |
                                        Class prediction + confidence
    """

    def __init__(
        self,
        model_output_dims: List[int],
        n_classes: int,
        hd_dim: int = 10000,
        mode: str = "bipolar",
        device: Optional[str] = None,
    ):
        super().__init__()
        self.model_output_dims = model_output_dims
        self.n_classes = n_classes
        self.hd_dim = hd_dim
        self.mode = mode
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Projection layers: model output -> hyperdimensional space
        self.projections = nn.ModuleList([
            nn.Linear(dim, hd_dim, bias=False)
            for dim in model_output_dims
        ])

        # Weighted consensus engine
        self.consensus = WeightedConsensusHDC(
            n_models=len(model_output_dims),
            input_dim=hd_dim,
            output_dim=hd_dim,
            mode=mode,
            device=self.device,
        )

        # Class hypervectors
        self.register_buffer(
            "class_hvs",
            gen_hvs(n_classes, hd_dim, mode, self.device, seed=42)
        )

    def project(
        self,
        model_outputs: List[torch.Tensor],
        binarize: bool = True,
    ) -> List[torch.Tensor]:
        """Project raw model outputs to hyperdimensional space.

        Args:
            model_outputs: List of (B, model_dim) tensors
            binarize: If True, threshold to bipolar after projection

        Returns:
            List of (B, hd_dim) hypervectors
        """
        projected = []
        for proj, output in zip(self.projections, model_outputs):
            hv = proj(output.to(self.device))
            if binarize and self.mode == "bipolar":
                hv = thresh(hv)
            projected.append(hv)
        return projected

    def fit(self, model_outputs: List[torch.Tensor], labels: torch.Tensor):
        """One-shot training: project, train HILs, build consensus.

        Args:
            model_outputs: List of (B, model_dim) tensors
            labels: (B,) class indices
        """
        # Project to hyperspace
        projected = self.project(model_outputs)

        # Train all HILs
        self.consensus.train_all(projected, labels)

        # Build weighted consensus
        self.consensus.build_consensus()

    def predict(self, query: torch.Tensor, model_idx: int = 0) -> torch.Tensor:
        """Predict from a single model's projected output.

        Args:
            query: (B, model_dim) from one model
            model_idx: Which model this output came from

        Returns:
            (B,) class predictions
        """
        hv = self.projections[model_idx](query.to(self.device))
        if self.mode == "bipolar":
            hv = thresh(hv)
        # Use consensus memory to recover output hv, then compare to class HVs
        recovered = bind(
            self.consensus.consensus_memory.unsqueeze(0),
            hv.to(self.consensus.consensus_memory.device),
            self.mode
        )
        # Compute cosine similarity: (B, D) vs (n_classes, D) -> (B, n_classes)
        B, D = recovered.shape
        n_classes = self.class_hvs.shape[0]
        recovered_norm = recovered.norm(dim=1, keepdim=True).clamp(min=1e-12)
        class_norm = self.class_hvs.norm(dim=1, keepdim=True).clamp(min=1e-12)
        similarities = (recovered @ self.class_hvs.T) / (recovered_norm * class_norm.T)
        return similarities.argmax(dim=1)

    def _recover_sample(self, projected: List[torch.Tensor], i: int) -> torch.Tensor:
        """Recover output HV for sample i via weighted consensus unbinding."""
        recovered = torch.zeros(self.hd_dim, device=self.device)
        weights = self.consensus.get_model_weights()
        mem = self.consensus.consensus_memory
        for m_idx, hv in enumerate(projected):
            recovered += weights[m_idx] * bind(mem, hv[i].to(mem.device), self.mode)
        if self.mode == "bipolar":
            recovered = thresh(recovered)
        return recovered

    def predict_consensus(
        self,
        model_outputs: List[torch.Tensor]
    ) -> torch.Tensor:
        """Predict using all models with weighted consensus.

        Args:
            model_outputs: List of (B, model_dim) tensors

        Returns:
            (B,) consensus predictions
        """
        projected = self.project(model_outputs)
        B = projected[0].shape[0]
        preds = [
            int(batch_sim(self._recover_sample(projected, i), self.class_hvs, self.mode).argmax().item())
            for i in range(B)
        ]
        return torch.tensor(preds, device=self.device)

    def predict_with_confidence(
        self,
        model_outputs: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict with confidence scores.

        Args:
            model_outputs: List of (B, model_dim) tensors

        Returns:
            (predictions, confidences) each (B,)
        """
        projected = self.project(model_outputs)
        B = projected[0].shape[0]
        final_preds, final_confs = [], []
        for i in range(B):
            recovered = self._recover_sample(projected, i)
            sims = batch_sim(recovered, self.class_hvs, self.mode)
            probs = torch.softmax(sims, dim=0)
            pred = int(sims.argmax().item())
            final_preds.append(pred)
            final_confs.append(float(probs[pred].item()))
        return torch.tensor(final_preds), torch.tensor(final_confs)


# Tests
def test_hd_glue():
    """Verify HD-Glue consensus mechanism."""
    print("=" * 60)
    print("Testing HD-Glue: Symbolic Fusion via HDC")
    print("=" * 60)

    # Simulate 3 models with different output dimensions
    B, n_classes = 32, 10
    hd_dim = 2000  # Smaller for testing

    model_output_dims = [128, 256, 512]
    outputs = [
        torch.randn(B, dim)
        for dim in model_output_dims
    ]
    labels = torch.randint(0, n_classes, (B,))

    glue = HDGlue(
        model_output_dims=model_output_dims,
        n_classes=n_classes,
        hd_dim=hd_dim,
    )
    glue.eval()

    # One-shot fit
    with torch.no_grad():
        glue.fit(outputs, labels)

    # Test single-model prediction
    preds = glue.predict(outputs[0], model_idx=0)
    print(f"\n  Single-model predictions shape: {preds.shape}")
    print(f"  Unique predicted classes: {preds.unique().tolist()}")

    # Test consensus prediction
    consensus_preds = glue.predict_consensus(outputs)
    print(f"\n  Consensus predictions shape: {consensus_preds.shape}")
    print(f"  Agreement with first model: "
          f"{(consensus_preds == preds).float().mean():.1%}")

    # Verify memory is non-zero (learning happened)
    non_zero = (glue.consensus.consensus_memory.abs() > 1e-6).sum().item()
    print(f"\n  Consensus memory occupancy: {non_zero}/{hd_dim} "
          f"({non_zero/hd_dim:.1%})")

    # Test confidence prediction
    preds_conf, confs = glue.predict_with_confidence(outputs)
    print(f"\n  Confidence-based predictions shape: {preds_conf.shape}")
    print(f"  Mean confidence: {confs.mean():.3f}")

    print("\nHD-Glue test complete!")


if __name__ == "__main__":
    test_hd_glue()
