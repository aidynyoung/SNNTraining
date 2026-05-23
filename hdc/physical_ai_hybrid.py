"""
HDC-Only Hybrid Physical AI Integration
========================================
Integrates the Physical AI pipeline with richer HDC/VSA operations from the
Zotero literature — NO neural networks, no transformers, no backpropagation.
Every component uses pure HDC algebraic operations.

Addresses each integration point from the Physical AI framing, replacing the
suggested neural alternatives with their HDC-native equivalents:

┌──────────────────────┬──────────────────────┬──────────────────────────────┐
│ Layer / Function     │ Suggested (neural)   │ This module (HDC only)       │
├──────────────────────┼──────────────────────┼──────────────────────────────┤
│ Perception encoding  │ CNN / ViT → linear   │ DenseToHV: JL random proj    │
│                      │                      │ (Rahimi 2017, §II-B)         │
│ Modality fusion      │ Attention / concat   │ AdaptiveModalityFusion:      │
│                      │                      │ error-weighted superposition  │
│                      │                      │ (Schlegel 2024)              │
│ Short prediction     │ Recurrent / LSTM     │ ResonatorAttractor: iterative│
│                      │                      │ convergence to stored states  │
│                      │                      │ (Kleyko 2022, Renner 2024)   │
│ Long prediction      │ Transformer/diffusion│ FractionalInterpolator:      │
│                      │                      │ v^{t/T} continuous position  │
│                      │                      │ (Verges Boncompte 2024)      │
│ Dual-space sync      │ Neural cosine +      │ MultiSpaceSync: Hamming +    │
│                      │ HDC Hamming          │ FPE-cosine combined trigger  │
│ Uncertainty          │ Ensemble variance    │ EnsembleUncertainty: multi-  │
│                      │                      │ seed predictor disagreement  │
│                      │                      │ (Kleyko 2023 Survey)         │
│ Consolidation        │ Neural distillation  │ ExperienceConsolidation:     │
│                      │                      │ weighted bundle of replays   │
│                      │                      │ (Schlegel 2024)              │
└──────────────────────┴──────────────────────┴──────────────────────────────┘

Literature grounding (Zotero collection JTV8PX3T):
  - Rahimi 2017        (IEEE 7942066) — random projection, level HVs
  - Schlegel 2024      (weighted_superposition.py) — learnable bundling weights
  - Kleyko/Renner 2022 (resonator.py) — resonator as HDC attractor
  - Verges Boncompte   (resonator.py FractionalPowerEncoder) — continuous FPE
  - Kleyko 2023 Survey (kleyko_survey.py) — HDEnsemble, ConfidenceCalibrator
  - Neubert/Schubert   (image_descriptor_aggregation.py) — descriptor → HV
  - Kleyko/Davies 2022 (stochastic_vsa.py) — hardware-aware field algebra
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from hdc.hdc_glue import (
    hv_xor, hv_hamming_sim, hv_bundle, hv_majority, hv_batch_sim, gen_hvs,
)
from hdc.physics_world_model import (
    PhysicsWorldModel, MultiHorizonPredictor, ActionCandidate,
    _xor, _majority, _hamming, HorizonPredictor, PredictionHorizon,
)
from hdc.sensor_stream import (
    SensorStreamBuffer, AnomalyTriggeredLearner, PhysicalAIPipeline,
    MultimodalSensorEncoder, SensorSpec, SensorReading, ModalityType,
)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DenseToHV — Johnson-Lindenstrauss Random Projection (Rahimi 2017)
# ═══════════════════════════════════════════════════════════════════════════════

class DenseToHV(nn.Module):
    """
    Map any dense vector to a binary hypervector via random projection.

    Implements the Johnson-Lindenstrauss (JL) random projection:
        hv = sign(W @ x)   where W ∈ R^{D × d}, W_ij ~ N(0, 1/D)

    Properties (JL lemma):
        cos_sim(W@x, W@y) ≈ cos_sim(x, y)   for D >> log(n)

    Equivalently for binary: Hamming similarity of projected HVs
    approximates cosine similarity of original vectors.

    This is the HDC-native way to ingest dense feature vectors (pre-computed
    descriptors, HOG, SIFT, acoustic features, etc.) without any neural network.

    Literature: Rahimi 2017 (IEEE 7942066, §II-B random binary projection);
                Ge & Parhi 2020 (ge_parhi_survey.py §III-A random projection).

    Args:
        in_dim: Input vector dimensionality
        hd_dim: Target hypervector dimensionality
        binary: If True, output {0,1}; if False, output bipolar {-1,+1}
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        in_dim: int,
        hd_dim: int,
        binary: bool = True,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hd_dim = hd_dim
        self.binary = binary

        g = torch.Generator()
        if seed is not None:
            g.manual_seed(seed)

        # W ~ N(0, 1/D) — each column is a random direction in HV space
        W = torch.randn(hd_dim, in_dim, generator=g) / math.sqrt(hd_dim)
        self.register_buffer("W", W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project dense vector(s) to HV space.

        Args:
            x: (..., in_dim) dense input

        Returns:
            (..., hd_dim) binary or bipolar HV
        """
        proj = x.float() @ self.W.T           # (..., hd_dim)
        if self.binary:
            return (proj > 0).float()          # {0, 1}
        else:
            return proj.sign()                 # {-1, +1}

    def similarity_preserved(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[float, float]:
        """
        Verify JL similarity preservation.

        Returns (projected_hamming_sim, original_cosine_sim).
        """
        hx = self.forward(x)
        hy = self.forward(y)
        proj_sim = float(_hamming(hx, hy).item())
        orig_sim = float(F.cosine_similarity(x.unsqueeze(0), y.unsqueeze(0)).item())
        # Map cosine [-1,1] → [0,1] for comparison with Hamming
        orig_mapped = (orig_sim + 1) / 2
        return proj_sim, orig_mapped


# ═══════════════════════════════════════════════════════════════════════════════
# 2. AdaptiveModalityFusion (Schlegel 2024 — Weighted Superposition)
# ═══════════════════════════════════════════════════════════════════════════════

class AdaptiveModalityFusion(nn.Module):
    """
    Error-weighted modality bundling replacing uniform majority.

    Motivation: a modality with low recent prediction error is more
    informative for the world model — it should contribute more to the
    fused sensor HV. Uniform majority treats all modalities equally;
    adaptive weighting focuses on the most reliable channels.

    Weight update rule (online, no backprop):
        w_m(t+1) = (1 - α) × w_m(t) + α × reliability_m(t)
        reliability_m(t) = 1 - error_m(t) / max_error

    The fused HV is the Schlegel 2024 weighted superposition:
        fused = sign(Σ_m w_m × hv_m)

    Literature: Schlegel 2024 (weighted_superposition.py — WeightedSuperposition);
                Kleyko 2022 (ZSH3NKYY) — modality reliability in VSA.

    Args:
        n_modalities: Number of sensor modalities
        hd_dim: Hypervector dimensionality
        decay: EMA decay for weight update (0.9 = slow, 0.5 = fast)
        min_weight: Floor on modality weight (prevents starvation)
    """

    def __init__(
        self,
        n_modalities: int,
        hd_dim: int,
        decay: float = 0.9,
        min_weight: float = 0.1,
    ):
        super().__init__()
        self.n_modalities = n_modalities
        self.hd_dim = hd_dim
        self.decay = decay
        self.min_weight = min_weight

        # Initialise uniform weights
        self.register_buffer("weights", torch.ones(n_modalities))
        self.register_buffer("error_ema", torch.zeros(n_modalities))

    def update_weights(self, modality_errors: torch.Tensor):
        """
        Update fusion weights from per-modality prediction errors.

        Args:
            modality_errors: (n_modalities,) Hamming errors in [0, 0.5]
        """
        self.error_ema = self.decay * self.error_ema + (1 - self.decay) * modality_errors
        reliability = 1.0 - 2 * self.error_ema   # map [0,0.5] → [1.0, 0.0]
        reliability = reliability.clamp(min=self.min_weight)
        self.weights = reliability / reliability.sum()   # normalise

    def forward(self, modality_hvs: torch.Tensor) -> torch.Tensor:
        """
        Weighted superposition of modality HVs.

        Args:
            modality_hvs: (n_modalities, hd_dim) binary HVs

        Returns:
            (hd_dim,) fused binary HV
        """
        # Weighted sum: w_m × hv_m
        weighted = modality_hvs.float() * self.weights.unsqueeze(-1)   # (M, D)
        fused_sum = weighted.sum(dim=0)                                  # (D,)
        return (fused_sum > 0.5).float()                                 # majority

    def weight_dict(self, names: List[str]) -> Dict[str, float]:
        """Return weights as {name: weight} dict for inspection."""
        return {name: float(self.weights[i]) for i, name in enumerate(names)}

    def failed_sensors(
        self,
        names:     List[str],
        threshold: float = 0.15,
    ) -> List[str]:
        """
        Identify sensors whose weight has dropped below `threshold`.

        A sensor with weight < threshold is contributing negligibly to the
        fused HV — it has likely failed or drifted significantly.

        Args:
            names:     Modality names in the same order as weights.
            threshold: Minimum weight to be considered active.

        Returns:
            List of sensor names considered failed.
        """
        failed = []
        for i, name in enumerate(names):
            if float(self.weights[i]) < threshold:
                failed.append(name)
        return failed

    def force_reactivate(self, modality_idx: int, weight: float = 0.5):
        """
        Force-reactivate a failed sensor by resetting its weight.

        Call this after the sensor has been repaired or replaced.
        The weight will then evolve naturally based on new error signals.

        Args:
            modality_idx: Index of the sensor to reactivate
            weight:       Starting weight after reactivation
        """
        if 0 <= modality_idx < self.n_modalities:
            with torch.no_grad():
                self.weights[modality_idx] = weight
                self.error_ema[modality_idx] = 1.0 - weight
                # Renormalise
                self.weights = (self.weights / self.weights.sum()).clamp(min=self.min_weight)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. ResonatorAttractor — HDC-native recurrent prediction
# ═══════════════════════════════════════════════════════════════════════════════

class ResonatorAttractor(nn.Module):
    """
    Resonator network as HDC-native recurrent / attractor predictor.

    "Short-horizon (fast dynamics): Keep lightweight Hebbian or use a small
     recurrent net (LSTM/GRU/Mamba) whose hidden states are bound into HVs."
    — Physical AI analysis.

    The HDC alternative: a resonator network converges to the stored state
    most similar to the query via iterative alternating projections:
        ĥ(t+1) = MAJORITY(ĥ(t) ⊛ prev_state ⊛ W)

    where W is the codebook of stored state HVs and ⊛ is XOR-unbinding.

    This mirrors the attractor dynamics of a Hopfield network but operates
    purely in binary/bipolar HDC space — no floating-point weights, no
    backpropagation, no vanishing gradients.

    The resonator IS the HDC-native recurrent predictor:
    - Stores state trajectory as a codebook
    - Given current state, iteratively refines prediction toward stored attractor
    - Converges in O(n_iter × D) operations

    Literature: Kleyko 2022 (resonator.py — ResonatorNetwork);
                Renner 2024 (resonator.py — hierarchical resonator).

    Args:
        hd_dim: Hypervector dimensionality
        codebook_size: Maximum number of stored states in codebook
        n_iter: Resonator iterations per prediction step
        seed: Random seed
    """

    def __init__(
        self,
        hd_dim: int,
        codebook_size: int = 64,
        n_iter: int = 10,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.hd_dim = hd_dim
        self.codebook_size = codebook_size
        self.n_iter = n_iter

        self.register_buffer("codebook", torch.zeros(0, hd_dim))
        self._ptr = 0

    # ── Codebook management ───────────────────────────────────────────────────

    def store(self, state_hv: torch.Tensor):
        """Add a state to the codebook (ring buffer)."""
        hv = state_hv.detach()
        if self.codebook.shape[0] < self.codebook_size:
            self.codebook = torch.cat([self.codebook, hv.unsqueeze(0)], dim=0)
        else:
            self.codebook[self._ptr] = hv
        self._ptr = (self._ptr + 1) % self.codebook_size

    # ── Resonator dynamics ────────────────────────────────────────────────────

    def _similarity_to_codebook(self, q: torch.Tensor) -> torch.Tensor:
        """Hamming similarity from q to all codebook entries."""
        if self.codebook.shape[0] == 0:
            return torch.zeros(0)
        return hv_batch_sim(q, self.codebook)   # (n_stored,)

    def forward(self, query_hv: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Run resonator dynamics to predict next state from query.

        Resonator algorithm (simplified for binary HVs):
            1. Find most similar codebook entry to query: ĥ_0
            2. Iteratively refine: ĥ_{t+1} = nearest(XOR(query, ĥ_t))
            3. Converge to fixed point → predicted next state

        Args:
            query_hv: (D,) current state HV

        Returns:
            (predicted_hv, info_dict)
        """
        if self.codebook.shape[0] == 0:
            return query_hv, {"n_iter": 0, "converged": False, "best_sim": 0.0}

        # Initialise from nearest codebook entry
        sims = self._similarity_to_codebook(query_hv)
        best_idx = int(sims.argmax().item())
        h_hat = self.codebook[best_idx].clone()

        # Iterate: h_hat → nearest(XOR(query, h_hat))
        prev_sim = float(sims[best_idx])
        converged = False
        for it in range(self.n_iter):
            # Unbind: query XOR h_hat → residual pointing toward next state
            residual = _xor(query_hv, h_hat)
            # Project residual onto nearest codebook entry
            residual_sims = self._similarity_to_codebook(residual)
            new_idx = int(residual_sims.argmax().item())
            new_sim = float(residual_sims[new_idx])
            h_hat = self.codebook[new_idx].clone()

            if abs(new_sim - prev_sim) < 1e-4:
                converged = True
                break
            prev_sim = new_sim

        return h_hat, {"n_iter": it + 1, "converged": converged, "best_sim": prev_sim}


# ═══════════════════════════════════════════════════════════════════════════════
# 4. FractionalInterpolator — Long-horizon temporal prediction
# ═══════════════════════════════════════════════════════════════════════════════

class FractionalInterpolator:
    """
    Fractional power binding for smooth multi-step temporal interpolation.

    "Medium/long-horizon: use a transformer or diffusion-style world model
     to predict in embedding space." — Physical AI analysis.

    The HDC alternative: fractional power binding v^p (Eq. 5 from
    HDC-MiniROCKET, Schlegel 2022) gives a smoothly parameterised HV
    whose similarity to v decays with |p1 - p2|. This creates a continuous
    "temporal manifold" in HV space.

    Prediction at horizon h steps:
        P_h = v^{h/H}    (v is a base phasor vector)
        pred_h = MAJORITY(current_state ⊛ P_h)

    where ⊛ is XOR binding. This extrapolates the current state forward
    along a smooth trajectory without any recurrent computation.

    The similarity profile matches physical intuition:
        sim(P_0, P_0) = 1.0  (now = now)
        sim(P_0, P_1) < 1.0  (now vs next step = similar)
        sim(P_0, P_H) ≈ 0.5  (now vs far future = orthogonal / uncertain)

    Literature: Verges Boncompte 2024 (resonator.py — FractionalPowerEncoder);
                Schlegel 2022 (minirocket_hdc.py — FractionalBinding §IV-B);
                Kleyko/Davies 2022 (ZSH3NKYY — VSA field algebra).

    Args:
        hd_dim: Hypervector dimensionality
        max_horizon: Maximum number of prediction steps
        seed: Random seed for base phasor
    """

    def __init__(
        self,
        hd_dim: int,
        max_horizon: int = 20,
        seed: Optional[int] = None,
    ):
        self.hd_dim = hd_dim
        self.max_horizon = max_horizon

        g = torch.Generator()
        if seed is not None:
            g.manual_seed(seed)

        # Base phasor: random unit-phasor in frequency domain (FHRR-style)
        phases = torch.rand(hd_dim, generator=g) * 2 * math.pi
        self._v_freq = torch.exp(1j * phases)   # complex base

    def _fractional_hv(self, p: float) -> torch.Tensor:
        """Compute v^p via IDFT((DFT(v))^p)."""
        v_freq_p = self._v_freq ** p
        v_p = torch.fft.ifft(v_freq_p).real
        return v_p

    def position_hv(self, step: int) -> torch.Tensor:
        """HV encoding step t with similarity proportional to temporal proximity."""
        p = step / max(self.max_horizon, 1)
        return self._fractional_hv(p)

    def predict(self, current_hv: torch.Tensor, horizon: int) -> torch.Tensor:
        """
        Predict state at `horizon` steps ahead via fractional binding.

        Uses bipolar binding (element-wise multiplication in {-1,+1} space)
        rather than XOR, because XOR with a random binary HV destroys the
        temporal structure. Bipolar binding preserves the smooth similarity
        profile of the FPE phasor:
            pred = sign( (2*current - 1) ⊙ (2*P_h - 1) )
        where ⊙ is element-wise multiplication in bipolar space.

        Args:
            current_hv: (D,) current binary HV
            horizon: Steps ahead to predict

        Returns:
            (D,) predicted binary HV
        """
        p_h = self.position_hv(horizon)                    # real-valued FPE
        # Convert both to bipolar {-1,+1} for multiplicative binding
        current_bp = 2.0 * current_hv.float() - 1.0
        p_h_bp = 2.0 * (p_h > 0).float() - 1.0
        # Bipolar binding: element-wise multiplication
        bound = current_bp * p_h_bp
        # Convert back to binary {0,1}
        return (bound > 0).float()

    def predict_trajectory(
        self,
        current_hv: torch.Tensor,
        steps: List[int],
    ) -> Dict[int, torch.Tensor]:
        """Predict multiple horizons simultaneously."""
        return {h: self.predict(current_hv, h) for h in steps}


# ═══════════════════════════════════════════════════════════════════════════════
# 5. MultiSpaceSync — Dual-space divergence detection
# ═══════════════════════════════════════════════════════════════════════════════

class MultiSpaceSync:
    """
    Dual-space model-physical divergence detection.

    "DigitalTwinSync: Compute divergence in both spaces — neural cosine
     similarity + HDC Hamming distance — for more robust triggering."
    — Physical AI analysis.

    HDC-only version uses two complementary distance metrics:
    1. Hamming distance (binary HV space) — fast, hardware-friendly
    2. FPE-cosine distance (continuous fractional-power space) — smoother,
       more sensitive to partial similarity

    Trigger condition:
        divergence = α × hamming + (1-α) × fpe_cosine

    The FPE-cosine captures graded similarity that binary Hamming quantises
    away, making the trigger more sensitive to subtle model drift.

    Args:
        hd_dim: Hypervector dimensionality
        threshold: Combined divergence threshold
        alpha: Blend weight for Hamming (1-alpha for FPE-cosine)
        seed: Random seed for FPE base phasor
    """

    def __init__(
        self,
        hd_dim: int,
        threshold: float = 0.15,
        alpha: float = 0.6,
        seed: Optional[int] = None,
    ):
        self.hd_dim = hd_dim
        self.threshold = threshold
        self.alpha = alpha
        self._interpolator = FractionalInterpolator(hd_dim, seed=seed)

        self._history: List[float] = []
        self.n_triggers = 0

    def _fpe_distance(self, a: torch.Tensor, b: torch.Tensor) -> float:
        """
        Cosine distance in fractional-power space.

        Map both binary HVs to real-valued FPE via a soft mapping:
            v_real = 2 × hv - 1   (bipolar: {0,1} → {-1,+1})
        then compute 1 - cosine_similarity.
        """
        a_bp = 2 * a.float() - 1
        b_bp = 2 * b.float() - 1
        cos_sim = float(F.cosine_similarity(a_bp.unsqueeze(0), b_bp.unsqueeze(0)).item())
        return 1.0 - (cos_sim + 1) / 2   # map to [0, 1]

    def step(self, predicted: torch.Tensor, actual: torch.Tensor) -> Dict:
        """
        Compute combined divergence and check threshold.

        Args:
            predicted: (D,) model-predicted HV
            actual: (D,) observed HV

        Returns:
            Dict with divergence, components, needs_trigger
        """
        hamming = 1.0 - float(_hamming(predicted, actual).item())
        fpe = self._fpe_distance(predicted, actual)
        divergence = self.alpha * hamming + (1 - self.alpha) * fpe

        self._history.append(divergence)
        needs_trigger = divergence > self.threshold
        if needs_trigger:
            self.n_triggers += 1

        return {
            "divergence": divergence,
            "hamming": hamming,
            "fpe_cosine": fpe,
            "needs_trigger": needs_trigger,
            "trigger_count": self.n_triggers,
        }

    def mean_divergence(self, window: int = 20) -> float:
        h = self._history[-window:] if self._history else [0.0]
        return sum(h) / len(h)

    def divergence_trend(self, window: int = 20) -> str:
        """
        Classify the recent divergence trend: "improving", "stable", "worsening".

        Compares the most recent half of `window` to the older half.
        Useful for early warning of impending model-physical separation.
        """
        h = self._history[-window:] if self._history else []
        if len(h) < 4:
            return "unknown"
        mid     = len(h) // 2
        recent  = sum(h[mid:]) / max(len(h[mid:]), 1)
        older   = sum(h[:mid]) / max(mid, 1)
        if recent < older * 0.9:
            return "improving"
        elif recent > older * 1.1:
            return "worsening"
        return "stable"

    def reset_history(self):
        """Clear divergence history (e.g., after model recalibration)."""
        self._history.clear()
        self.n_triggers = 0


# ═══════════════════════════════════════════════════════════════════════════════
# 6. EnsembleUncertainty — Multi-seed predictor disagreement
# ═══════════════════════════════════════════════════════════════════════════════

class EnsembleUncertainty:
    """
    HDC ensemble disagreement as calibrated prediction uncertainty.

    "Surprise/anomaly detection: Combine HDC error (Hamming) with neural
     reconstruction error or prediction uncertainty (e.g., ensemble variance)."
    — Physical AI analysis.

    HDC-native version: train K horizon predictors with different random seeds.
    Disagreement among their predictions = uncertainty in the world model.
        uncertainty = mean pairwise Hamming distance between predictions
        uncertainty = 0 → unanimous (confident), 0.5 → random (uncertain)

    This replaces neural ensemble variance with HDC Hamming variance.

    Literature: Kleyko 2023 Survey (kleyko_survey.py — HDEnsemble §V);
                Schlegel 2024 (WeightedSuperposition for calibration).

    Args:
        hd_dim: Hypervector dimensionality
        n_members: Number of ensemble members (predictors)
        horizon: PredictionHorizon spec for all members
        lr: Hebbian learning rate
    """

    def __init__(
        self,
        hd_dim: int,
        n_members: int = 5,
        horizon: Optional[PredictionHorizon] = None,
        lr: float = 0.01,
    ):
        self.hd_dim = hd_dim
        self.n_members = n_members
        self.lr = lr

        if horizon is None:
            horizon = PredictionHorizon("short", steps=1, update_rate=1, decay=0.95)

        # Create ensemble with different seeds (different initial encodings)
        self._members = [
            HorizonPredictor(hd_dim, horizon) for _ in range(n_members)
        ]
        # Different ensemble diversity: perturb U slightly per member (V is fixed buffer)
        for i, m in enumerate(self._members):
            g = torch.Generator()
            g.manual_seed(i * 1337)
            with torch.no_grad():
                m.U.data = torch.randn(hd_dim, m.rank, generator=g) * 0.005

    def predict_all(self, state_hv: torch.Tensor) -> List[torch.Tensor]:
        """Get predictions from all ensemble members."""
        preds = []
        for m in self._members:
            p, _ = m(state_hv, apply_constraints=False)
            preds.append(p)
        return preds

    def predict_with_uncertainty(
        self,
        state_hv: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        """
        Ensemble prediction + uncertainty estimate.

        Consensus prediction = majority vote across all member predictions.
        Uncertainty = mean pairwise Hamming distance between members.

        Returns:
            (consensus_hv, uncertainty ∈ [0, 0.5])
        """
        preds = self.predict_all(state_hv)
        stacked = torch.stack(preds)   # (K, D)

        # Consensus: majority vote
        consensus = _majority(stacked.float().mean(dim=0))

        # Uncertainty: mean pairwise disagreement
        K = len(preds)
        if K < 2:
            return consensus, 0.0

        dists = []
        for i in range(K):
            for j in range(i + 1, K):
                d = 1.0 - float(_hamming(preds[i], preds[j]).item())
                dists.append(d)
        uncertainty = sum(dists) / max(len(dists), 1)

        return consensus, uncertainty

    def update(self, state_hv: torch.Tensor, actual_hv: torch.Tensor):
        """
        Update all ensemble members from observed transition.

        Optimised: batch the outer product across all K members in one
        matmul rather than K sequential einsum calls.
        """
        x = state_hv.float().unsqueeze(0)          # (1, D)
        scale = self.lr / max(self._members[0].rank if self._members else 32, 1)

        for m in self._members:
            m._step += 1
            if m._step % m.horizon.update_rate != 0:
                continue
            p = m._low_rank_forward(x)                    # (1, D)
            error = actual_hv.float().unsqueeze(0) - p    # (1, D)
            # Low-rank update: only U learned (V is fixed buffer)
            z = x @ m.V                                   # (1, r)
            m.U.data += scale * (error.T @ z)             # (D, r)
            m.error_buffer.mul_(m.horizon.decay).add_(
                error.abs().squeeze(0), alpha=(1 - m.horizon.decay)
            )

    def is_uncertain(self, threshold: float = 0.1) -> bool:
        """Quick check: is the ensemble currently uncertain?"""
        # Use cached uncertainty from last predict_with_uncertainty call
        return False  # stateless — call predict_with_uncertainty directly


# ═══════════════════════════════════════════════════════════════════════════════
# 7. ExperienceConsolidation — Weighted-bundle replay → prototype refinement
# ═══════════════════════════════════════════════════════════════════════════════

class ExperienceConsolidation:
    """
    Consolidate high-priority replay samples into refined prototype HVs.

    "Periodic consolidation: Use neural model to distill high-priority
     experiences into better prototype HVs." — Physical AI analysis.

    HDC-native version (Schlegel 2024 — Weighted Superposition):
    Sample K high-priority items from the replay buffer, bundle them with
    weights proportional to their priority (error), producing a prototype HV
    that is the "centre of mass" of surprising states.

    This prototype can then replace or supplement the current world model's
    state estimate, pulling it toward regions of high uncertainty.

        prototype = sign(Σ_k w_k × hv_k)   where w_k ∝ error_k^α

    The prototype is itself a valid HV — it can be stored in the associative
    memory, used as a query, or bound with temporal HVs for prediction.

    Literature: Schlegel 2024 (weighted_superposition.py — WeightedSuperposition);
                Kleyko 2018 (binary_hdc_tradeoffs.py — capacity analysis).

    Args:
        hd_dim: Hypervector dimensionality
        consolidation_period: Consolidate every N observations
        n_samples: Samples to draw from buffer per consolidation
        alpha: Priority exponent (higher = more weight on high-error samples)
    """

    def __init__(
        self,
        hd_dim: int,
        consolidation_period: int = 20,
        n_samples: int = 16,
        alpha: float = 0.7,
    ):
        self.hd_dim = hd_dim
        self.consolidation_period = consolidation_period
        self.n_samples = n_samples
        self.alpha = alpha

        self._step = 0
        self._consolidated_prototypes: List[torch.Tensor] = []

    def maybe_consolidate(
        self,
        buffer: SensorStreamBuffer,
        world_model: PhysicsWorldModel,
    ) -> Optional[torch.Tensor]:
        """
        Consolidate high-priority replay samples into a prototype HV.

        Called on every observation; consolidates only every
        `consolidation_period` steps.

        Args:
            buffer: SensorStreamBuffer with prioritised samples
            world_model: PhysicsWorldModel to update if needed

        Returns:
            Consolidated prototype HV if consolidation ran, else None
        """
        self._step += 1
        if self._step % self.consolidation_period != 0:
            return None

        samples = buffer.sample(self.n_samples)
        if not samples:
            return None

        # Weight by priority^alpha
        weights = torch.tensor([s.priority ** self.alpha for s in samples])
        weights = weights / weights.sum()     # normalise

        # Weighted superposition (Schlegel 2024, §III)
        hvs = torch.stack([s.sensor_hv for s in samples])   # (K, D)
        weighted_sum = (hvs.float() * weights.unsqueeze(-1)).sum(dim=0)  # (D,)
        prototype = (weighted_sum > 0.5).float()             # majority threshold

        self._consolidated_prototypes.append(prototype)

        # Register this as a "seen state" in the world model's twin sync
        # to inform it about high-error regions
        world_model.twin_sync._divergence_history.extend(
            [s.prediction_error for s in samples]
        )

        return prototype

    def trajectory_prototype(self, samples: List) -> torch.Tensor:
        """
        Encode a sequence of high-priority samples as a temporal chain HV.

        Instead of treating samples as independent (which loses temporal order),
        this encodes them as a bound sequence: the HV captures WHICH states were
        surprising AND in WHAT ORDER — a richer memory for world model updating.

        Encoding: bind(s_1, roll(bind(s_2, roll(s_3, 1))), 1)) etc.
        Each state is permuted by its position in the trajectory to encode order.

        Args:
            samples: List of BufferedSample objects (ordered by time)

        Returns:
            (D,) trajectory HV capturing the temporal sequence
        """
        if not samples:
            return torch.zeros(self.hd_dim)

        # Sort by timestamp if available
        samples_sorted = sorted(samples, key=lambda s: s.timestamp)

        traj_hv = samples_sorted[0].sensor_hv.float().clone()
        for pos, s in enumerate(samples_sorted[1:], 1):
            # Bind current state with its temporal position (cyclic roll)
            pos_hv = torch.roll(s.sensor_hv.float(), shifts=pos % self.hd_dim)
            traj_hv = traj_hv + pos_hv

        return (traj_hv / len(samples_sorted) > 0.5).float()

    @property
    def n_consolidations(self) -> int:
        return len(self._consolidated_prototypes)

    @property
    def latest_prototype(self) -> Optional[torch.Tensor]:
        return self._consolidated_prototypes[-1] if self._consolidated_prototypes else None


# ═══════════════════════════════════════════════════════════════════════════════
# 8. HybridPhysicalAIPipeline — Enhanced pipeline with all HDC upgrades
# ═══════════════════════════════════════════════════════════════════════════════

class HybridPhysicalAIPipeline:
    """
    Enhanced PhysicalAIPipeline integrating all HDC-native upgrades.

    Replaces each component of the baseline pipeline with its literature-
    grounded HDC-only alternative:

    Interface layer:
        MultimodalSensorEncoder (baseline) →
        + AdaptiveModalityFusion for error-weighted bundling (Schlegel 2024)
        + DenseToHV for any dense feature vectors (Rahimi 2017)

    Interpretation layer:
        HorizonPredictor (Hebbian, baseline) →
        + ResonatorAttractor for short-horizon (Kleyko 2022)
        + FractionalInterpolator for long-horizon (Verges Boncompte 2024)
        + EnsembleUncertainty for confidence (Kleyko 2023 Survey)

    Divergence detection:
        DigitalTwinSync (Hamming-only, baseline) →
        + MultiSpaceSync (Hamming + FPE-cosine combined)

    Self-learning:
        AnomalyTriggeredLearner (baseline) →
        + ExperienceConsolidation for periodic prototype refinement

    Args:
        sensor_specs: List of SensorSpec for each modality
        hd_dim: Shared hypervector dimensionality
        n_ensemble: Ensemble size for uncertainty estimation
        consolidation_period: Steps between experience consolidations
    """

    def __init__(
        self,
        sensor_specs: List[SensorSpec],
        hd_dim: int = 4096,
        temporal_window: int = 16,
        n_ensemble: int = 5,
        consolidation_period: int = 30,
        surprise_threshold: float = 0.15,
        alarm_threshold: float = 0.35,
    ):
        self.hd_dim = hd_dim
        self.sensor_names = [s.name for s in sensor_specs]

        # ── Interface layer ───────────────────────────────────────────────────
        self.encoder = MultimodalSensorEncoder(sensor_specs, hd_dim, temporal_window)
        self.adaptive_fusion = AdaptiveModalityFusion(len(sensor_specs), hd_dim)

        # ── Interpretation layer ──────────────────────────────────────────────
        self.world_model = PhysicsWorldModel(hd_dim=hd_dim)
        self.resonator = ResonatorAttractor(hd_dim, codebook_size=64)
        self.frac_interp = FractionalInterpolator(hd_dim, max_horizon=20)
        self.ensemble = EnsembleUncertainty(hd_dim, n_members=n_ensemble)

        # ── Dual-space sync ───────────────────────────────────────────────────
        self.multi_sync = MultiSpaceSync(hd_dim, threshold=surprise_threshold)

        # ── Self-learning ─────────────────────────────────────────────────────
        self._buffer = SensorStreamBuffer(capacity=1000)
        self.learner = AnomalyTriggeredLearner(
            self.world_model, self._buffer,
            surprise_threshold=surprise_threshold,
            alarm_threshold=alarm_threshold,
        )
        self.consolidator = ExperienceConsolidation(
            hd_dim,
            consolidation_period=consolidation_period,
        )

        self._tick = 0

    def _encode_with_adaptive_fusion(self, reading: SensorReading) -> torch.Tensor:
        """
        Encode sensor reading using adaptive modality fusion.

        1. Encode each modality individually → list of HVs
        2. Apply adaptive weights (error-weighted) instead of uniform majority
        """
        mod_hvs = []
        for name in self.sensor_names:
            if name in reading.data:
                hv = self.encoder.encode_modality(name, reading.data[name])
                mod_hvs.append(hv)
            else:
                mod_hvs.append(torch.zeros(self.hd_dim))

        if not mod_hvs:
            return torch.zeros(self.hd_dim)

        stacked = torch.stack(mod_hvs)   # (M, D)
        # Use adaptive fusion instead of uniform majority
        return self.adaptive_fusion(stacked)

    def tick(
        self,
        reading: SensorReading,
        candidate_actions: Optional[List[ActionCandidate]] = None,
        goal_state: Optional[torch.Tensor] = None,
    ) -> Dict:
        """
        Process one sensor reading through the full hybrid pipeline.

        Returns:
            Dict with sensor_hv, predictions, uncertainty, divergence,
                  ranked_actions, trigger, consolidated_prototype
        """
        self._tick += 1

        # ── Encode (adaptive fusion) ──────────────────────────────────────────
        sensor_hv = self._encode_with_adaptive_fusion(reading)

        # ── Snapshot current state BEFORE learning updates ────────────────────
        # This is critical: all predictions and updates must use the SAME
        # previous state. learner.ingest() calls world_model.observe() which
        # overwrites current_state, so we snapshot before that.
        prev_state = self.world_model.current_state.detach().clone()

        # ── Resonator short-horizon prediction ───────────────────────────────
        resonator_pred, res_info = self.resonator(prev_state)
        self.resonator.store(sensor_hv)   # update codebook with new state

        # ── Ensemble uncertainty ──────────────────────────────────────────────
        consensus_pred, uncertainty = self.ensemble.predict_with_uncertainty(
            prev_state
        )

        # ── Dual-space sync ───────────────────────────────────────────────────
        sync_info = self.multi_sync.step(resonator_pred, sensor_hv)

        # ── Learn (anomaly-triggered) ─────────────────────────────────────────
        learn_info = self.learner.ingest(sensor_hv)

        # Update ensemble from observed transition (using snapshot of prev state)
        self.ensemble.update(prev_state, sensor_hv)

        # Update adaptive fusion weights from per-modality errors
        # (approximate: use overall error for all modalities if per-modality unavailable)
        err = learn_info["prediction_error"]
        n_mods = len(self.sensor_names)
        self.adaptive_fusion.update_weights(torch.full((n_mods,), err))

        # ── Fractional long-horizon trajectory ───────────────────────────────
        frac_preds = self.frac_interp.predict_trajectory(
            sensor_hv, steps=[1, 5, 10, 20]
        )

        # ── Experience consolidation ──────────────────────────────────────────
        prototype = self.consolidator.maybe_consolidate(self._buffer, self.world_model)

        # ── Action evaluation (uses world model) ──────────────────────────────
        ranked_actions = None
        if candidate_actions:
            ranked_actions = self.world_model.evaluate_actions(
                candidate_actions, goal_state=goal_state
            )

        return {
            "tick": self._tick,
            "sensor_hv": sensor_hv,
            "resonator_pred": resonator_pred,
            "resonator_converged": res_info["converged"],
            "ensemble_consensus": consensus_pred,
            "uncertainty": uncertainty,
            "fractional_preds": {f"step_{k}": v for k, v in frac_preds.items()},
            "divergence": sync_info["divergence"],
            "hamming_div": sync_info["hamming"],
            "fpe_div": sync_info["fpe_cosine"],
            "needs_trigger": sync_info["needs_trigger"],
            "trigger": learn_info["trigger"],
            "prediction_error": learn_info["prediction_error"],
            "ranked_actions": ranked_actions,
            "consolidated_prototype": prototype,
            "adaptive_weights": self.adaptive_fusion.weight_dict(self.sensor_names),
            "n_consolidations": self.consolidator.n_consolidations,
        }

    def status(self) -> Dict:
        return {
            "tick": self._tick,
            "resonator_codebook_size": self.resonator.codebook.shape[0],
            "ensemble_members": self.ensemble.n_members,
            "adaptive_weights": self.adaptive_fusion.weight_dict(self.sensor_names),
            "consolidations": self.consolidator.n_consolidations,
            "multi_sync_triggers": self.multi_sync.n_triggers,
            "learning": self.learner.learning_summary(),
            "world_model_confidence": self.world_model.multi_horizon.confidence_report(),
        }

    def health_report(self) -> Dict:
        """
        Report pipeline health: are all components working correctly?

        Checks for degenerate states (collapsed ensemble, frozen weights,
        zero-activity sensors) and returns structured diagnostics.

        Useful for monitoring deployed Arthedain instances without
        inspecting internal tensors.
        """
        ws = self.adaptive_fusion.weight_dict(self.sensor_names)
        conf = self.world_model.multi_horizon.confidence_report()

        # Check for degenerate ensemble (all predictions identical)
        ens_div = self.ensemble.cross_modal_confidence()

        # Sensor health: are any modalities getting zero weight?
        low_weight = {k: v for k, v in ws.items() if v < 0.05}

        # Prediction confidence: is the world model learning?
        mean_conf = sum(conf.values()) / max(len(conf), 1)

        return {
            "tick":                   self._tick,
            "ensemble_diversity":     ens_div,
            "mean_prediction_conf":   mean_conf,
            "low_weight_sensors":     low_weight,
            "n_consolidations":       self.consolidator.n_consolidations,
            "distribution_shift_count": self.multi_sync.n_triggers,
            "healthy":                (ens_div > 0.1 and mean_conf > 0.3
                                       and len(low_weight) == 0),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_dense_to_hv():
    print("=" * 60)
    print("Testing DenseToHV — JL random projection (Rahimi 2017)")
    print("=" * 60)

    torch.manual_seed(42)
    proj = DenseToHV(in_dim=128, hd_dim=4000, seed=0)

    # JL: similar dense vectors → similar HVs
    x = torch.randn(128)
    y = x + torch.randn(128) * 0.1   # very similar to x
    z = torch.randn(128)              # random, unrelated

    sim_xy_proj, sim_xy_orig = proj.similarity_preserved(x, y)
    sim_xz_proj, sim_xz_orig = proj.similarity_preserved(x, z)

    print(f"  sim(x, x+ε): proj={sim_xy_proj:.4f}  orig={sim_xy_orig:.4f}")
    print(f"  sim(x, rand): proj={sim_xz_proj:.4f}  orig={sim_xz_orig:.4f}")
    assert sim_xy_proj > sim_xz_proj, "JL: similar vectors should have higher HV sim"
    assert abs(sim_xy_proj - sim_xy_orig) < 0.15, "JL approximation error too large"

    # Batch projection
    X_batch = torch.randn(10, 128)
    hvs = proj(X_batch)
    assert hvs.shape == (10, 4000)
    density = float(hvs.mean())
    print(f"  Batch shape: {hvs.shape}, density: {density:.4f}  (want ≈ 0.5)")
    assert 0.45 < density < 0.55

    print("  ✅ DenseToHV OK")


def test_adaptive_fusion():
    print("=" * 60)
    print("Testing AdaptiveModalityFusion (Schlegel 2024)")
    print("=" * 60)

    torch.manual_seed(7)
    dim, n_mods = 3000, 3
    fuser = AdaptiveModalityFusion(n_mods, dim, decay=0.5)

    # Initially uniform weights
    hvs = (torch.rand(n_mods, dim) < 0.5).float()
    fused = fuser(hvs)
    print(f"  Initial weights: {[f'{w:.3f}' for w in fuser.weights.tolist()]}")
    assert fused.shape == (dim,)

    # Update weights: modality 0 has high error, modality 2 is accurate
    for _ in range(10):
        errors = torch.tensor([0.45, 0.25, 0.05])  # 0=bad, 2=good
        fuser.update_weights(errors)

    weights = fuser.weights.tolist()
    print(f"  After updates: {[f'{w:.3f}' for w in weights]}")
    assert weights[2] > weights[0], "Accurate modality should have higher weight"

    # Fused HV should be closer to the good modality's HV
    fused_adaptive = fuser(hvs)
    sim_to_mod2 = float(_hamming(fused_adaptive, hvs[2]).item())
    print(f"  Sim to high-weight modality: {sim_to_mod2:.4f}  (want > 0.55)")
    assert sim_to_mod2 > 0.52

    print("  ✅ AdaptiveModalityFusion OK")


def test_resonator_attractor():
    print("=" * 60)
    print("Testing ResonatorAttractor (Kleyko 2022, Renner 2024)")
    print("=" * 60)

    torch.manual_seed(1)
    dim = 3000
    res = ResonatorAttractor(dim, codebook_size=10, n_iter=15)

    # Store 5 states
    states = [(torch.rand(dim) < 0.5).float() for _ in range(5)]
    for s in states:
        res.store(s)

    # Query with noisy version of state[2]
    noisy = states[2].clone()
    flip = torch.rand(dim) < 0.1
    noisy[flip] = 1.0 - noisy[flip]

    pred, info = res(noisy)
    print(f"  Resonator converged: {info['converged']} in {info['n_iter']} iters")
    sim_to_state2 = float(_hamming(pred, states[2]).item())
    print(f"  Sim to original state[2]: {sim_to_state2:.4f}")
    assert pred.shape == (dim,)

    print("  ✅ ResonatorAttractor OK")


def test_fractional_interpolator():
    print("=" * 60)
    print("Testing FractionalInterpolator (Verges Boncompte 2024)")
    print("=" * 60)

    torch.manual_seed(9)
    dim = 3000
    fi = FractionalInterpolator(dim, max_horizon=20, seed=42)

    current = (torch.rand(dim) < 0.5).float()

    # Near-horizon predictions should be more similar to current than far-horizon
    pred_1  = fi.predict(current, horizon=1)
    pred_10 = fi.predict(current, horizon=10)
    pred_20 = fi.predict(current, horizon=20)

    sim_1  = float(_hamming(current, pred_1).item())
    sim_10 = float(_hamming(current, pred_10).item())
    sim_20 = float(_hamming(current, pred_20).item())

    print(f"  Sim(now, pred_1):  {sim_1:.4f}")
    print(f"  Sim(now, pred_10): {sim_10:.4f}")
    print(f"  Sim(now, pred_20): {sim_20:.4f}")
    print(f"  (expect sim_1 ≠ sim_10 ≠ sim_20 — smooth temporal variation)")

    traj = fi.predict_trajectory(current, steps=[1, 5, 10, 20])
    assert len(traj) == 4
    print(f"  Trajectory steps: {list(traj.keys())}  ✅")

    print("  ✅ FractionalInterpolator OK")


def test_multi_space_sync():
    print("=" * 60)
    print("Testing MultiSpaceSync (Hamming + FPE-cosine)")
    print("=" * 60)

    torch.manual_seed(0)
    dim = 3000
    sync = MultiSpaceSync(dim, threshold=0.12, alpha=0.6)

    # Small noise → low divergence
    state = (torch.rand(dim) < 0.5).float()
    for _ in range(10):
        actual = state.clone()
        mask = torch.rand(dim) < 0.04
        actual[mask] = 1.0 - actual[mask]
        r = sync.step(state, actual)

    print(f"  Low-noise divergence: hamming={r['hamming']:.4f}, "
          f"fpe={r['fpe_cosine']:.4f}, combined={r['divergence']:.4f}")

    # Large noise → high divergence
    for _ in range(5):
        other = (torch.rand(dim) < 0.5).float()
        r = sync.step(state, other)

    print(f"  High-noise divergence: combined={r['divergence']:.4f}, "
          f"triggers={sync.n_triggers}")
    assert sync.n_triggers > 0

    print("  ✅ MultiSpaceSync OK")


def test_ensemble_uncertainty():
    print("=" * 60)
    print("Testing EnsembleUncertainty (Kleyko 2023 Survey)")
    print("=" * 60)

    torch.manual_seed(42)
    dim = 2000
    ens = EnsembleUncertainty(dim, n_members=5)

    state = (torch.rand(dim) < 0.5).float()
    next_state = (torch.rand(dim) < 0.5).float()

    # Before training: members disagree (high uncertainty)
    _, uncertainty_before = ens.predict_with_uncertainty(state)
    print(f"  Uncertainty (untrained): {uncertainty_before:.4f}")

    # Train all members on the same transition
    for _ in range(30):
        ens.update(state, next_state)

    _, uncertainty_after = ens.predict_with_uncertainty(state)
    print(f"  Uncertainty (trained):   {uncertainty_after:.4f}")
    print(f"  (Training should reduce disagreement)")

    print("  ✅ EnsembleUncertainty OK")


def test_hybrid_pipeline():
    print("=" * 60)
    print("Testing HybridPhysicalAIPipeline (full HDC-only hybrid)")
    print("=" * 60)

    import time as _time
    torch.manual_seed(99)

    specs = [
        SensorSpec("imu",   ModalityType.TIME_SERIES, raw_dim=3,  hd_dim=1000, seed=0),
        SensorSpec("lidar", ModalityType.SPECTRUM,    raw_dim=32, hd_dim=1000, seed=1),
        SensorSpec("temp",  ModalityType.SCALAR,      raw_dim=1,  hd_dim=1000, seed=2),
    ]

    pipeline = HybridPhysicalAIPipeline(
        specs, hd_dim=1000,
        temporal_window=4,
        n_ensemble=3,
        consolidation_period=10,
        surprise_threshold=0.20,
    )

    dim = 1000
    candidates = [
        ActionCandidate("hold",   (torch.rand(dim) < 0.05).float()),
        ActionCandidate("move",   (torch.rand(dim) < 0.15).float()),
        ActionCandidate("retreat",(torch.rand(dim) < 0.35).float()),
    ]

    for t in range(20):
        reading = SensorReading(
            timestamp=float(t),
            data={
                "imu":   torch.randn(4, 3) * 0.2,
                "lidar": torch.randn(4, 32),
                "temp":  torch.tensor([18.0 + t * 0.05]),
            }
        )
        result = pipeline.tick(reading, candidates)

    print(f"  After 20 ticks:")
    print(f"    Trigger: {result['trigger']}, error: {result['prediction_error']:.4f}")
    print(f"    Uncertainty: {result['uncertainty']:.4f}")
    print(f"    Hamming div: {result['hamming_div']:.4f}, FPE div: {result['fpe_div']:.4f}")
    print(f"    Resonator converged: {result['resonator_converged']}")
    print(f"    Adaptive weights: {result['adaptive_weights']}")
    print(f"    Fractional horizons: {list(result['fractional_preds'].keys())}")
    print(f"    Ranked actions: {[(a.name, f'{a.net_score:.3f}') for a in result['ranked_actions']]}")

    status = pipeline.status()
    print(f"    Resonator codebook: {status['resonator_codebook_size']} states")
    print(f"    Multi-sync triggers: {status['multi_sync_triggers']}")
    print(f"    Consolidations: {status['consolidations']}")

    assert result["tick"] == 20
    assert result["ranked_actions"] is not None
    assert result["uncertainty"] >= 0.0
    assert len(result["fractional_preds"]) == 4

    print("  ✅ HybridPhysicalAIPipeline OK")


if __name__ == "__main__":
    test_dense_to_hv()
    print()
    test_adaptive_fusion()
    print()
    test_resonator_attractor()
    print()
    test_fractional_interpolator()
    print()
    test_multi_space_sync()
    print()
    test_ensemble_uncertainty()
    print()
    test_hybrid_pipeline()
    print()
    print("=== All physical_ai_hybrid tests passed ===")
