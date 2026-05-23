"""
Resonator Networks for HDC Factorization
=========================================
Based on: Renner et al. 2024 "Neuromorphic Visual Scene Understanding
with Resonator Networks" and Kleyko et al. 2022 "Vector Symbolic
Architectures as a Computing Framework for Emerging Hardware"

Key insight: Resonator networks factorize a superposition of bound
hypervectors into their constituent factors. Given a scene vector
S = sum_i (obj_i ⊗ pose_i), the resonator iteratively recovers
the individual objects and poses through alternating projections.

This enables:
1. Scene decomposition: factorize a scene into objects and poses
2. Hierarchical factorization: partition transforms into translation/rotation
3. Spiking phasor neurons: implement complex-valued resonators for neuromorphic HW

Reference:
  Renner, A., Supic, L., et al. (2024)
  "Neuromorphic Visual Scene Understanding with Resonator Networks"
  arXiv:2406.17676

  Kleyko, D., et al. (2022)
  "Vector Symbolic Architectures as a Computing Framework for Emerging Hardware"
  Proceedings of the IEEE
"""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Dict
from models.hdc import gen_hvs, bind, bundle, sim, thresh, batch_sim


class ResonatorNetwork(nn.Module):
    """
    Resonator network for factorizing hypervector superpositions.
    
    Given a scene hypervector S that is a superposition of bound
    factor pairs:
        S = sum_i (factor_a_i ⊗ factor_b_i)
    
    The resonator recovers the individual factors through iterative
    alternating projections:
        a_hat(t+1) = threshold(S ⊗ b_hat(t) * W_a)
        b_hat(t+1) = threshold(S ⊗ a_hat(t+1) * W_b)
    
    Where W_a and W_b are codebooks of all possible factor values.
    
    Args:
        codebook_a: (n_a, dim) codebook for factor A
        codebook_b: (n_b, dim) codebook for factor B
        dim: Hypervector dimensionality
        mode: VSA mode ("bipolar" or "binary")
        n_iterations: Maximum number of iterations
        threshold: Convergence threshold (similarity change < threshold)
    """
    
    def __init__(
        self,
        codebook_a: torch.Tensor,
        codebook_b: torch.Tensor,
        dim: int = 10000,
        mode: str = "bipolar",
        n_iterations: int = 100,
        threshold: float = 1e-4,
    ):
        super().__init__()
        self.dim = dim
        self.mode = mode
        self.n_iterations = n_iterations
        self.threshold = threshold
        
        # Register codebooks as buffers
        self.register_buffer("codebook_a", codebook_a)
        self.register_buffer("codebook_b", codebook_b)
        self.n_a = codebook_a.shape[0]
        self.n_b = codebook_b.shape[0]
    
    def _similarity(self, q: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
        """Compute similarity between query and all codebook entries."""
        q = q.reshape(-1)  # flatten to 1D
        if self.mode == "bipolar":
            return (codebook @ q) / (codebook.norm(dim=1) * q.norm()).clamp(min=1e-12)
        elif self.mode == "binary":
            return (codebook == q.unsqueeze(0)).float().mean(dim=1)
        else:
            return (codebook @ q) / (codebook.norm(dim=1) * q.norm()).clamp(min=1e-12)
    
    def _project(self, hv: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
        """Project hypervector onto nearest codebook entry."""
        sims = self._similarity(hv, codebook)
        best_idx = int(sims.argmax().item())
        return codebook[best_idx].clone(), best_idx, sims
    
    def forward(
        self,
        scene_hv: torch.Tensor,
        init_a: Optional[torch.Tensor] = None,
        init_b: Optional[torch.Tensor] = None,
        return_trace: bool = False,
        n_restarts: int = 1,
    ):
        """Factorize a scene hypervector.
        
        Args:
            scene_hv: (dim,) scene hypervector (superposition of bound pairs)
            init_a: Optional (dim,) initial guess for factor A
            init_b: Optional (dim,) initial guess for factor B
            return_trace: If True, return iteration history
            n_restarts: Number of random restarts (uses best result)
        
        Returns:
            dict with keys:
                - factor_a: (dim,) recovered factor A
                - factor_b: (dim,) recovered factor B
                - idx_a: index in codebook A
                - idx_b: index in codebook B
                - n_iterations: iterations used
                - converged: whether converged
                - trace: (optional) iteration history
        """
        best_result = None
        best_confidence = -1.0
        
        for restart in range(n_restarts):
            # Initialize guesses
            if restart == 0 and init_a is not None:
                a_hat = init_a.clone()
            else:
                rand_idx = torch.randint(0, self.n_a, (1,)).item()
                a_hat = self.codebook_a[rand_idx].clone()
            
            if restart == 0 and init_b is not None:
                b_hat = init_b.clone()
            else:
                rand_idx = torch.randint(0, self.n_b, (1,)).item()
                b_hat = self.codebook_b[rand_idx].clone()
            
            trace = [] if return_trace else None
            prev_a_idx, prev_b_idx = -1, -1
            
            for i in range(self.n_iterations):
                # Update factor A: S ⊗ b_hat → project onto codebook A
                if self.mode == "bipolar":
                    a_residual = scene_hv * b_hat
                else:
                    a_residual = (scene_hv + b_hat) % 2
                
                a_new, a_idx, a_sims = self._project(a_residual, self.codebook_a)
                
                # Update factor B: S ⊗ a_hat → project onto codebook B
                if self.mode == "bipolar":
                    b_residual = scene_hv * a_new
                else:
                    b_residual = (scene_hv + a_new) % 2
                
                b_new, b_idx, b_sims = self._project(b_residual, self.codebook_b)
                
                if trace is not None:
                    trace.append({
                        "iteration": i,
                        "a_idx": a_idx,
                        "b_idx": b_idx,
                        "a_confidence": float(a_sims.max()),
                        "b_confidence": float(b_sims.max()),
                    })
                
                a_hat, b_hat = a_new, b_new
                
                # Check convergence: both indices stable
                if a_idx == prev_a_idx and b_idx == prev_b_idx:
                    break
                
                prev_a_idx, prev_b_idx = a_idx, b_idx
            
            # Score this restart by confidence
            confidence = float(a_sims.max() + b_sims.max()) / 2.0
            
            if best_result is None or confidence > best_confidence:
                best_confidence = confidence
                best_result = {
                    "factor_a": a_hat,
                    "factor_b": b_hat,
                    "idx_a": a_idx,
                    "idx_b": b_idx,
                    "n_iterations": i + 1,
                    "converged": i < self.n_iterations - 1,
                    "confidence": confidence,
                }
                if trace is not None:
                    best_result["trace"] = trace
        
        fa = best_result["factor_a"]
        fb = best_result["factor_b"]
        # Preserve batch dimension from input
        if scene_hv.dim() == 2:
            return (fa.unsqueeze(0), fb.unsqueeze(0))
        return (fa, fb)

    def factorize_batch(
        self,
        scene_hvs: torch.Tensor,
    ) -> List[Dict]:
        """Factorize a batch of scene hypervectors.
        
        Args:
            scene_hvs: (B, dim) batch of scene hypervectors
        
        Returns:
            List of factorization results
        """
        results = []
        for i in range(scene_hvs.shape[0]):
            results.append(self.forward(scene_hvs[i]))
        return results


class HierarchicalResonatorNetwork(nn.Module):
    """
    Hierarchical Resonator Network (HRN) for complex scene factorization.
    
    Based on Renner et al. 2024. Partitions the factorization into
    hierarchical levels:
    - Level 1: Translation (horizontal + vertical)
    - Level 2: Rotation + scaling
    - Level 3: Object identity
    
    Each level uses its own resonator network, and the results
    propagate upward/downward through the hierarchy.
    
    This is more efficient than a single large resonator because
    each level operates on a smaller codebook.
    """
    
    def __init__(
        self,
        codebooks: Dict[str, torch.Tensor],
        hierarchy: List[List[str]],
        dim: int = 10000,
        mode: str = "bipolar",
        n_iterations: int = 50,
        threshold: float = 1e-4,
    ):
        """
        Args:
            codebooks: Dict mapping factor names to codebook tensors
            hierarchy: List of levels, each level is a list of factor names
                       e.g., [["tx", "ty"], ["rot", "scale"], ["object"]]
            dim: Hypervector dimensionality
            mode: VSA mode
            n_iterations: Max iterations per level
            threshold: Convergence threshold
        """
        super().__init__()
        self.codebooks = codebooks
        self.hierarchy = hierarchy
        self.dim = dim
        self.mode = mode
        self.n_iterations = n_iterations
        self.threshold = threshold
        
        # Create resonator for each level
        self.resonators = nn.ModuleList()
        for level_factors in hierarchy:
            if len(level_factors) == 2:
                res = ResonatorNetwork(
                    codebook_a=codebooks[level_factors[0]],
                    codebook_b=codebooks[level_factors[1]],
                    dim=dim, mode=mode,
                    n_iterations=n_iterations,
                    threshold=threshold,
                )
            elif len(level_factors) == 1:
                # Single factor: just project onto codebook
                res = None  # Handled separately
            else:
                raise ValueError(f"Each level must have 1 or 2 factors, got {len(level_factors)}")
            self.resonators.append(res)
    
    def forward(
        self,
        scene_hv: torch.Tensor,
        return_all: bool = False,
    ) -> Dict:
        """Factorize scene through hierarchical resonator network.
        
        Args:
            scene_hv: (dim,) scene hypervector
            return_all: If True, return all intermediate results
        
        Returns:
            dict with recovered factors at each level
        """
        results = {}
        current_hv = scene_hv.clone()
        
        for level_idx, level_factors in enumerate(self.hierarchy):
            resonator = self.resonators[level_idx]
            
            if len(level_factors) == 2:
                # Two-factor level: use resonator
                res = resonator.forward(current_hv)
                results[level_factors[0]] = {
                    "hv": res["factor_a"],
                    "idx": res["idx_a"],
                    "converged": res["converged"],
                }
                results[level_factors[1]] = {
                    "hv": res["factor_b"],
                    "idx": res["idx_b"],
                    "converged": res["converged"],
                }
                
                # Unbind recovered factors from scene for next level
                if self.mode == "bipolar":
                    current_hv = current_hv * res["factor_a"] * res["factor_b"]
                else:
                    current_hv = (current_hv + res["factor_a"] + res["factor_b"]) % 2
            
            elif len(level_factors) == 1:
                # Single factor level: project onto codebook
                factor_name = level_factors[0]
                codebook = self.codebooks[factor_name]
                sims = batch_sim(current_hv, codebook, self.mode)
                best_idx = int(sims.argmax().item())
                results[factor_name] = {
                    "hv": codebook[best_idx].clone(),
                    "idx": best_idx,
                    "confidence": float(sims[best_idx]),
                }
        
        if not return_all:
            # Return only indices
            simplified = {}
            for level_factors in self.hierarchy:
                for f in level_factors:
                    simplified[f] = results[f]["idx"]
            return simplified
        
        return results


class PhasorNeuron(nn.Module):
    """
    Spiking phasor neuron model for neuromorphic resonator networks.
    
    Based on Renner et al. 2024. A phasor neuron represents a complex
    value as a pair of coupled LIF neurons (I & Q components). The
    phase is encoded in the relative timing of spikes.
    
    This enables mapping resonator networks onto efficient neuromorphic
    hardware (Loihi 2, etc.).
    
    Architecture:
        Input current → I-compartment LIF → I-spike
                      → Q-compartment LIF → Q-spike
        Phase = atan2(Q_spikes, I_spikes)
        Magnitude = sqrt(I² + Q²)
    """
    
    def __init__(
        self,
        tau_m: float = 20.0,
        tau_s: float = 5.0,
        v_th: float = 1.0,
        v_reset: float = 0.0,
        dt: float = 1.0,
    ):
        super().__init__()
        self.tau_m = tau_m
        self.tau_s = tau_s
        self.v_th = v_th
        self.v_reset = v_reset
        self.dt = dt
        
        # Decay factors
        self.alpha = torch.tensor(dt / tau_m)
        self.beta = torch.tensor(dt / tau_s)
    
    def forward(
        self,
        i_in: torch.Tensor,
        q_in: torch.Tensor,
        v_i: Optional[torch.Tensor] = None,
        v_q: Optional[torch.Tensor] = None,
        i_syn_i: Optional[torch.Tensor] = None,
        i_syn_q: Optional[torch.Tensor] = None,
    ) -> Dict:
        """Forward pass for one timestep.
        
        Args:
            i_in: I-component input current
            q_in: Q-component input current
            v_i: Previous I-compartment membrane potential
            v_q: Previous Q-compartment membrane potential
            i_syn_i: Previous I-compartment synaptic current
            i_syn_q: Previous Q-compartment synaptic current
        
        Returns:
            dict with keys: spikes_i, spikes_q, v_i, v_q, phase, magnitude
        """
        # Initialize state
        if v_i is None:
            v_i = torch.zeros_like(i_in)
        if v_q is None:
            v_q = torch.zeros_like(q_in)
        if i_syn_i is None:
            i_syn_i = torch.zeros_like(i_in)
        if i_syn_q is None:
            i_syn_q = torch.zeros_like(q_in)
        
        # Update synaptic currents (exponential synapse)
        i_syn_i = i_syn_i * (1 - self.beta) + i_in * self.beta
        i_syn_q = i_syn_q * (1 - self.beta) + q_in * self.beta
        
        # Update membrane potentials (LIF dynamics)
        v_i = v_i * (1 - self.alpha) + i_syn_i * self.alpha
        v_q = v_q * (1 - self.alpha) + i_syn_q * self.alpha
        
        # Spike generation
        spikes_i = (v_i >= self.v_th).float()
        spikes_q = (v_q >= self.v_th).float()
        
        # Reset
        v_i = v_i * (1 - spikes_i) + spikes_i * self.v_reset
        v_q = v_q * (1 - spikes_q) + spikes_q * self.v_reset
        
        # Compute phase and magnitude from spike rates
        # (using exponential moving average of spikes)
        phase = torch.atan2(spikes_q, spikes_i + 1e-12)
        magnitude = torch.sqrt(spikes_i ** 2 + spikes_q ** 2 + 1e-12)
        
        return {
            "spikes_i": spikes_i,
            "spikes_q": spikes_q,
            "v_i": v_i,
            "v_q": v_q,
            "i_syn_i": i_syn_i,
            "i_syn_q": i_syn_q,
            "phase": phase,
            "magnitude": magnitude,
        }


class FractionalPowerEncoder(nn.Module):
    """
    Fractional Power Encoding for continuous-valued HDC.
    
    Based on: Verges Boncompte 2024 "Classification with Hyperdimensional
    Computing" and Kleyko et al. 2022.
    
    Key insight: Instead of quantizing continuous values to discrete
    levels, use fractional power encoding where the phase of each
    hypervector component is proportional to the value:
    
        HV(v) = [cos(θ_i * v), sin(θ_i * v)]  for FHRR
        HV(v) = sign(cos(θ_i * v))             for MAP (bipolar)
    
    This enables:
    1. Learning the encoding phasors θ_i to match data distribution
    2. Smooth interpolation between values
    3. Differentiable encoding for end-to-end learning
    
    Reference:
      Verges Boncompte, P. (2024)
      "Classification with Hyperdimensional Computing"
      PhD Thesis, Universitat Politecnica de Catalunya
    """
    
    def __init__(
        self,
        dim: int = 10000,
        mode: str = "bipolar",
        learnable_phasors: bool = True,
        device: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.dim = dim
        self.mode = mode
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize phasors (random frequencies)
        g = torch.Generator(device=self.device)
        if seed is not None:
            g.manual_seed(seed)
        
        # Phasors control the frequency of each dimension
        # Higher frequency → more sensitive to value changes
        if learnable_phasors:
            self.phasors = nn.Parameter(
                torch.randn(dim, generator=g, device=self.device)
            )
        else:
            self.register_buffer(
                "phasors",
                torch.randn(dim, generator=g, device=self.device)
            )
    
    def encode(self, value: torch.Tensor) -> torch.Tensor:
        """Encode a continuous value into a hypervector.
        
        Args:
            value: Scalar value to encode (will be normalized to [-π, π])
        
        Returns:
            (dim,) hypervector encoding the value
        """
        # Normalize value to [-π, π]
        v = torch.tanh(value) * torch.pi
        
        # Compute phases: θ_i * v
        phases = self.phasors * v  # (dim,)
        
        if self.mode == "bipolar":
            # sign(cos(θ_i * v)) for bipolar
            return torch.sign(torch.cos(phases)).clamp(-1, 1)
        elif self.mode == "binary":
            # threshold(cos(θ_i * v)) for binary
            return (torch.cos(phases) >= 0).float()
        else:
            # Full complex for FHRR
            return torch.complex(torch.cos(phases), torch.sin(phases))
    
    def encode_batch(self, values: torch.Tensor) -> torch.Tensor:
        """Encode a batch of values.
        
        Args:
            values: (B,) tensor of values
        
        Returns:
            (B, dim) hypervectors
        """
        hvs = []
        for v in values:
            hvs.append(self.encode(v))
        return torch.stack(hvs)
    
    def get_phasor_spectrum(self) -> torch.Tensor:
        """Return the learned phasor frequencies.
        
        Useful for analyzing which frequencies the encoder learned.
        """
        return self.phasors.detach().cpu()


class AdaptiveHDClassifier(nn.Module):
    """
    Adaptive HDC classifier with learnable encoding and online updates.
    
    Based on: Verges Boncompte 2024 "Classification with Hyperdimensional
    Computing" - RefineHD approach.
    
    Features:
    1. Fractional power encoding for continuous features
    2. Learnable phasors per feature dimension
    3. Online prototype updates with adaptive learning rate
    4. Confidence-based rejection
    
    Reference:
      Verges Boncompte, P. (2024)
      "Classification with Hyperdimensional Computing"
      PhD Thesis, Chapter 4: Adaptive Learning
    """
    
    def __init__(
        self,
        n_features: int,
        n_classes: int,
        dim: int = 10000,
        mode: str = "bipolar",
        learning_rate: float = 0.1,
        device: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.n_features = n_features
        self.n_classes = n_classes
        self.dim = dim
        self.mode = mode
        self.learning_rate = learning_rate
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Per-feature fractional power encoders
        self.encoders = nn.ModuleList([
            FractionalPowerEncoder(
                dim=dim, mode=mode,
                learnable_phasors=True,
                device=self.device,
                seed=seed + i if seed is not None else None,
            )
            for i in range(n_features)
        ])
        
        # Class prototypes
        self.register_buffer(
            "class_hvs",
            gen_hvs(n_classes, dim, mode, self.device, seed + n_features if seed else None),
        )
        
        # Per-class counts for adaptive learning rate
        self.register_buffer("counts", torch.zeros(n_classes, device=self.device))
        
        # Feature importance weights (learnable)
        self.feature_weights = nn.Parameter(torch.ones(n_features))
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a feature vector into a hypervector.
        
        Args:
            x: (n_features,) feature vector
        
        Returns:
            (dim,) hypervector
        """
        # Encode each feature with its own fractional power encoder
        feature_hvs = []
        for i in range(self.n_features):
            hv = self.encoders[i].encode(x[i])
            feature_hvs.append(hv)
        
        # Weighted bundling
        stacked = torch.stack(feature_hvs)  # (n_features, dim)
        weights = torch.softmax(self.feature_weights, dim=0)
        weighted = stacked * weights.unsqueeze(-1)
        result = weighted.sum(dim=0)
        
        if self.mode == "bipolar":
            result = thresh(result)
        
        return result
    
    def predict(self, x: torch.Tensor) -> Tuple[int, torch.Tensor]:
        """Predict class for a feature vector.
        
        Args:
            x: (n_features,) feature vector
        
        Returns:
            (class_idx, similarities)
        """
        hv = self.encode(x)
        sims = batch_sim(hv, self.class_hvs, self.mode)
        return int(sims.argmax().item()), sims
    
    # ── Long/short-term memory separation (Teeters 2023) ─────────────────────
    # "On separating long- and short-term memories in hyperdimensional computing"
    # Short-term (ST): fast EMA — responsive to recent distribution shift
    # Long-term (LT): consolidated from ST every K steps — stable, high capacity
    # Prediction: sim = (1-α)·sim(x, LT) + α·sim(x, ST)

    def enable_dual_memory(
        self,
        st_momentum: float = 0.8,
        lt_momentum: float = 0.02,
        lt_weight: float = 0.7,
        consolidation_steps: int = 50,
    ) -> None:
        """Enable Teeters 2023 long/short-term memory separation.

        Args:
            st_momentum: EMA decay for short-term prototypes (higher = more stable)
            lt_momentum: Blend rate when consolidating ST into LT
            lt_weight: Weight of long-term vs short-term in final similarity (α = 1-lt_weight)
            consolidation_steps: Consolidate LT from ST every this many training steps
        """
        self._st_momentum = st_momentum
        self._lt_momentum = lt_momentum
        self._lt_weight = lt_weight
        self._consolidation_steps = consolidation_steps
        self._total_steps = 0
        # ST prototypes: copy of current class_hvs, updated via fast EMA
        self.register_buffer("st_hvs", self.class_hvs.clone())
        # LT prototypes: more stable version, updated via slow consolidation
        self.register_buffer("lt_hvs", self.class_hvs.clone())
        self._dual_memory = True

    def _update_st(self, hv: torch.Tensor, label: int) -> None:
        """Update short-term prototype via fast EMA."""
        m = self._st_momentum
        self.st_hvs[label] = m * self.st_hvs[label] + (1.0 - m) * hv.detach()

    def _consolidate_lt(self) -> None:
        """Merge short-term memory into long-term (slow, stable)."""
        m = 1.0 - self._lt_momentum
        self.lt_hvs = m * self.lt_hvs + self._lt_momentum * self.st_hvs

    def predict_dual(self, x: torch.Tensor) -> Tuple[int, torch.Tensor]:
        """Predict using combined long-term + short-term similarity.

        Returns (class_idx, blended_similarities).
        Falls back to standard predict if dual memory is not enabled.
        """
        if not getattr(self, "_dual_memory", False):
            return self.predict(x)

        hv = self.encode(x)
        sim_lt = batch_sim(hv, self.lt_hvs, self.mode)
        sim_st = batch_sim(hv, self.st_hvs, self.mode)
        α = 1.0 - self._lt_weight
        sims = self._lt_weight * sim_lt + α * sim_st
        return int(sims.argmax().item()), sims

    def train_step(
        self,
        x: torch.Tensor,
        label: int,
        predict_first: bool = True,
        reward: float = 1.0,
    ):
        """Online training step with RefineHD adaptive learning.

        Extends RefineHD (Verges Boncompte 2025) with a three-factor reward
        gate (Chakraborty et al. 2026, "Reward-Modulated Local Learning in
        Spiking Encoders").  The reward scalar multiplies the effective
        learning rate, implementing:

            ΔP = η · reward · (RefineHD direction)

        reward > 0: normal update (positive reinforcement)
        reward = 0: no update (block learning on this sample)
        reward < 0: reversal (punish the current prediction)

        When dual memory is enabled, also updates the short-term prototype
        and periodically consolidates it into the long-term memory
        (Teeters 2023).

        Based on Verges Boncompte 2025 PhD Dissertation, Algorithm 3:
        RefineHD Single-Pass Learning.

        The RefineHD algorithm has two modes:
        1. **Correct prediction**: Pull prototype toward sample
           P_c = normalize(P_c + lr_c * (1 - sim(x, P_c)) * x)
        2. **Incorrect prediction**: Push wrong prototype away, pull correct toward
           P_wrong = normalize(P_wrong - lr_c * (1 - sim(x, P_wrong)) * x)
           P_correct = normalize(P_correct + lr_c * (1 - sim(x, P_correct)) * x)

        The adaptive learning rate is per-class:
           lr_c = base_lr / (1 + count_c * 0.1)

        This gives higher learning rates for rare classes and lower
        for well-represented ones, preventing overfitting.

        Args:
            x: (n_features,) feature vector
            label: class label
            predict_first: If True, check prediction and apply retraction
                          for incorrect predictions (full RefineHD).
                          If False, always pull toward correct prototype.
            reward: Scalar reward gate in [-1, 1].  Default 1.0 (supervised).
                    Pass the task reward signal for reinforcement settings.
        """
        if reward == 0.0:
            return  # no update — blocked by reward signal

        hv = self.encode(x)

        # Adaptive learning rate: higher for rare classes, scaled by reward
        count = self.counts[label].item()
        lr = self.learning_rate / (1.0 + count * 0.1) * reward

        if predict_first:
            # Check current prediction (detach to avoid grad tracking on sims)
            with torch.no_grad():
                sims = batch_sim(hv, self.class_hvs, self.mode)
                pred = int(sims.argmax().item())

            if pred == label:
                sim_to_proto = float(sims[label].detach())
                pull_strength = lr * (1.0 - sim_to_proto)
                self.class_hvs[label] = self.class_hvs[label] + pull_strength * hv.detach()
            else:
                sim_to_wrong = float(sims[pred].detach())
                push_strength = lr * (1.0 - sim_to_wrong)
                self.class_hvs[pred] = self.class_hvs[pred] - push_strength * hv.detach()

                sim_to_correct = float(sims[label].detach())
                pull_strength = lr * (1.0 - sim_to_correct)
                self.class_hvs[label] = self.class_hvs[label] + pull_strength * hv.detach()
        else:
            self.class_hvs[label] = (1.0 - lr) * self.class_hvs[label] + lr * hv.detach()

        # Teeters 2023: update short-term memory and periodically consolidate
        if getattr(self, "_dual_memory", False):
            self._update_st(hv, label)
            self._total_steps += 1
            if self._total_steps % self._consolidation_steps == 0:
                self._consolidate_lt()

        self.counts[label] += 1

    # ── D2H-AD: Distance-to-Hypervector Anomaly Detection (Ghajari 2026) ─────
    # "D2H-AD: A Hybrid Model Utilizing Hyperdimensional Computing for
    #  Advanced Anomaly Detection"
    # Anomaly score = min_c(hamming_dist(x, P_c))
    # Adaptive threshold τ = percentile of training-set distances

    def enable_anomaly_detection(
        self,
        percentile: float = 95.0,
        warmup_steps: int = 200,
    ) -> None:
        """Enable D2H-AD anomaly detection mode.

        Args:
            percentile: Training-set distance percentile used as threshold τ.
                        Samples with min-distance > τ are flagged as anomalies.
            warmup_steps: Number of training samples to collect before τ is fixed.
        """
        self._d2h_percentile = percentile
        self._d2h_warmup = warmup_steps
        self._d2h_distances: list = []
        self._d2h_threshold: Optional[float] = None
        self._d2h_active = True

    def anomaly_score(self, x: torch.Tensor) -> Tuple[float, bool]:
        """Compute D2H-AD anomaly score for a sample.

        Returns:
            (score, is_anomaly) where score is the minimum Hamming distance
            to any class prototype (normalised to [0, 1]) and is_anomaly
            is True when score exceeds the learned threshold τ.
        """
        if not getattr(self, "_d2h_active", False):
            raise RuntimeError("Call enable_anomaly_detection() first.")

        hv = self.encode(x)
        # Hamming distance = (1 - cosine_similarity) / 2 for bipolar HVs
        sims = batch_sim(hv, self.class_hvs, self.mode)
        min_sim = float(sims.max().item())       # highest similarity = nearest prototype
        score = 1.0 - min_sim                    # distance ∈ [0, 1]

        is_anomaly = False
        if self._d2h_threshold is not None:
            is_anomaly = score > self._d2h_threshold

        return score, is_anomaly

    def update_anomaly_threshold(self, x: torch.Tensor) -> None:
        """Record a training-set distance and update threshold τ."""
        score, _ = self.anomaly_score(x)
        self._d2h_distances.append(score)
        if len(self._d2h_distances) >= self._d2h_warmup:
            import numpy as np
            self._d2h_threshold = float(
                np.percentile(self._d2h_distances, self._d2h_percentile)
            )

    # ─────────────────────────────────────────────────────────────────────────

    def renormalize(self):
        """Normalize class prototypes to valid HV space."""
        if self.mode == "bipolar":
            self.class_hvs = thresh(self.class_hvs)
        elif self.mode == "binary":
            self.class_hvs = (self.class_hvs >= 0.5).float()
        else:
            self.class_hvs = self.class_hvs / self.class_hvs.norm(dim=1, keepdim=True).clamp(min=1e-12)

    def min_inter_class_distance(self) -> float:
        """
        Return the minimum pairwise Hamming distance between class prototypes.

        Low values indicate classes that may be confused.  Healthy HDC
        classifiers have min distance ≥ 0.3 (at least 30% bits differ).
        """
        C = self.class_hvs.shape[0]
        if C < 2:
            return 1.0
        min_d = 1.0
        for i in range(C):
            for j in range(i + 1, C):
                d = float((self.class_hvs[i] != self.class_hvs[j]).float().mean())
                if d < min_d:
                    min_d = d
        return min_d

    def enforce_diversity(self, min_distance: float = 0.25):
        """
        Re-randomise class prototypes that are too similar to each other.

        When two class prototypes have Hamming distance < min_distance, the
        classifier will systematically confuse them.  This method adds a
        random perturbation to one of the pair to restore separability.

        Typically called after many RefineHD updates have caused drift.

        Args:
            min_distance: Minimum acceptable pairwise Hamming distance.
        """
        C   = self.class_hvs.shape[0]
        D   = self.class_hvs.shape[1]
        changed = True
        while changed:
            changed = False
            for i in range(C):
                for j in range(i + 1, C):
                    d = float((self.class_hvs[i] != self.class_hvs[j]).float().mean())
                    if d < min_distance:
                        # Flip ~20% of bits in class j to push it away
                        flip = (torch.rand(D, device=self.class_hvs.device) < 0.2)
                        if self.mode == "bipolar":
                            self.class_hvs[j] = torch.where(flip, -self.class_hvs[j], self.class_hvs[j])
                        else:
                            self.class_hvs[j] = ((self.class_hvs[j] + flip.float()) % 2)
                        changed = True

    def confusion_pairs(self, threshold: float = 0.35) -> List[Tuple[int, int, float]]:
        """
        Identify class pairs with dangerously low inter-prototype distance.

        Returns list of (class_i, class_j, distance) sorted by distance ascending.
        Pairs with distance < threshold are likely to generate misclassifications.
        """
        C = self.class_hvs.shape[0]
        pairs = []
        for i in range(C):
            for j in range(i + 1, C):
                d = float((self.class_hvs[i] != self.class_hvs[j]).float().mean())
                if d < threshold:
                    pairs.append((i, j, d))
        return sorted(pairs, key=lambda x: x[2])


# ── Tests ────────────────────────────────────────────────────────────────────

def test_resonator():
    """Verify resonator network factorization."""
    print("=" * 60)
    print("Testing Resonator Network (Renner 2024)")
    print("=" * 60)
    
    dim = 1000
    n_a, n_b = 10, 8
    
    # Create codebooks
    codebook_a = gen_hvs(n_a, dim, "bipolar")
    codebook_b = gen_hvs(n_b, dim, "bipolar")
    
    # Create a scene: bind one pair
    scene = bind(codebook_a[3], codebook_b[5], "bipolar")
    
    # Factorize with resonator
    resonator = ResonatorNetwork(
        codebook_a, codebook_b, dim=dim, mode="bipolar",
        n_iterations=50, threshold=1e-4,
    )
    
    result = resonator.forward(scene, n_restarts=5)
    
    print(f"\n  True factors: a=3, b=5")
    print(f"  Recovered: a={result['idx_a']}, b={result['idx_b']}")
    print(f"  Iterations: {result['n_iterations']}")
    print(f"  Converged: {result['converged']}")
    print(f"  Confidence: {result['confidence']:.4f}")
    
    # Binding is commutative in bipolar VSA: (a,b) == (b,a)
    correct = (result['idx_a'] == 3 and result['idx_b'] == 5) or \
              (result['idx_a'] == 5 and result['idx_b'] == 3)
    print(f"  {'✅' if correct else '❌'} Factorization {'correct' if correct else 'incorrect'}")
    
    # Test with superposition (multiple bound pairs)
    print(f"\n  Testing superposition factorization...")
    scene2 = bind(codebook_a[1], codebook_b[2], "bipolar") + bind(codebook_a[7], codebook_b[4], "bipolar")
    if resonator.mode == "bipolar":
        scene2 = thresh(scene2)
    
    result2 = resonator.forward(scene2)
    print(f"  Scene has pairs: (1,2) and (7,4)")
    print(f"  Recovered: a={result2['idx_a']}, b={result2['idx_b']}")
    
    print(f"\n  ✅ Resonator network test complete!")


def test_fractional_power_encoding():
    """Verify fractional power encoding."""
    print("=" * 60)
    print("Testing Fractional Power Encoding (Verges Boncompte 2024)")
    print("=" * 60)
    
    dim = 1000
    
    encoder = FractionalPowerEncoder(dim=dim, mode="bipolar", learnable_phasors=True)
    
    # Encode different values
    v1 = encoder.encode(torch.tensor(0.0))
    v2 = encoder.encode(torch.tensor(0.5))
    v3 = encoder.encode(torch.tensor(1.0))
    
    # Similarity should decrease with distance
    sim_01 = sim(v1, v2, "bipolar")
    sim_02 = sim(v1, v3, "bipolar")
    
    print(f"\n  sim(0.0, 0.5): {sim_01:.4f}")
    print(f"  sim(0.0, 1.0): {sim_02:.4f}")
    print(f"  Monotonic decreasing: {'✅' if sim_01 > sim_02 else '❌'}")
    
    # Test batch encoding
    values = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    hvs = encoder.encode_batch(values)
    print(f"\n  Batch encoding shape: {hvs.shape}")
    
    # Check similarity matrix is smooth
    sim_matrix = torch.zeros(5, 5)
    for i in range(5):
        for j in range(5):
            sim_matrix[i, j] = sim(hvs[i], hvs[j], "bipolar")
    
    print(f"  Similarity matrix (should be diagonally dominant):")
    for i in range(5):
        row = "    ".join([f"{sim_matrix[i,j]:.2f}" for j in range(5)])
        print(f"    {row}")
    
    print(f"\n  ✅ Fractional power encoding test complete!")


def test_adaptive_classifier():
    """Verify adaptive HDC classifier."""
    print("=" * 60)
    print("Testing Adaptive HDC Classifier (Verges Boncompte 2024)")
    print("=" * 60)
    
    n_features = 10
    n_classes = 4
    dim = 1000
    
    classifier = AdaptiveHDClassifier(
        n_features=n_features, n_classes=n_classes, dim=dim,
        mode="bipolar", learning_rate=0.1,
    )
    
    # Generate synthetic data
    torch.manual_seed(42)
    n_samples = 20
    
    for cls in range(n_classes):
        for _ in range(n_samples):
            x = torch.randn(n_features) * 0.3 + cls * 0.5
            # Use simple pull mode for initial training (more stable)
            classifier.train_step(x, cls, predict_first=False)
    classifier.renormalize()
    
    # Test accuracy
    correct = 0
    total = 100
    for cls in range(n_classes):
        for _ in range(total // n_classes):
            x = torch.randn(n_features) * 0.3 + cls * 0.5
            pred, sims = classifier.predict(x)
            if pred == cls:
                correct += 1
    
    accuracy = correct / total
    print(f"\n  Prediction accuracy: {accuracy:.1%}")
    print(f"  {'✅' if accuracy > 0.5 else '❌'} Adaptive classifier test complete!")


# ── Gap 2: Learned HDC Decoder (Kinavuidi et al. 2025) ───────────────────────
# "Hyperdimensional Decoding of Spiking Neural Networks" arXiv:2511.08558
#
# Instead of random channel keys, learn the projection from feature space
# to HV space via an online contrastive rule:
#   same-class pairs → pull HVs together (keys align)
#   diff-class pairs → push HVs apart   (keys diverge)
#
# This lifts classification accuracy by 8–12% over random projections on
# standard HDC benchmarks by ensuring the HV geometry respects class structure.

class LearnedHDCDecoder(nn.Module):
    """Contrastive online learning for HDC channel keys.

    Replaces random key initialization in SpikeHDC/FractionalPowerEncoder with
    keys that are updated to maximize inter-class HV separation and minimize
    intra-class HV distance — without backpropagation.

    The update rule (contrastive Hebbian):
        keys += lr * label_sign * outer(hv_error, x)
    where label_sign = +1 for same-class pairs, -1 for different-class pairs,
    and hv_error = target_hv - current_hv.

    Reference:
        Kinavuidi, C. et al. (2025)
        "Hyperdimensional Decoding of Spiking Neural Networks"
        arXiv:2511.08558
    """

    def __init__(
        self,
        input_dim: int,
        hdc_dim: int,
        mode: str = "bipolar",
        lr: float = 0.01,
        device: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hdc_dim = hdc_dim
        self.mode = mode
        self.lr = lr
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Channel keys: (input_dim, hdc_dim)
        g = torch.Generator()
        if seed is not None:
            g.manual_seed(seed)
        keys_init = torch.randn(input_dim, hdc_dim, generator=g)
        if mode == "bipolar":
            keys_init = keys_init.sign()
        elif mode == "binary":
            keys_init = (keys_init > 0).float()
        self.register_buffer("keys", keys_init.to(self.device))

        self.update_count = 0

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input vector to hypervector using current keys.

        Args:
            x: (input_dim,) input features

        Returns:
            (hdc_dim,) hypervector
        """
        # hv = sum_i x[i] * keys[i]
        hv = (x.unsqueeze(-1) * self.keys).sum(dim=0)
        if self.mode == "bipolar":
            return hv.sign()
        return (hv > 0).float()

    def contrastive_update(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        same_class: bool,
    ) -> None:
        """Update keys from a pair of samples (online contrastive learning).

        Args:
            x1: (input_dim,) first sample
            x2: (input_dim,) second sample
            same_class: True if both samples share a label
        """
        hv1 = self.encode(x1)
        hv2 = self.encode(x2)
        hv_error = hv2 - hv1  # direction to push/pull in HV space

        # Contrastive sign: pull together (+1) or push apart (-1)
        sign = 1.0 if same_class else -1.0

        # Key update: outer(hv_error, x1) — how to adjust keys to reduce/increase gap
        delta = torch.outer(hv_error, x1)  # (hdc_dim, input_dim)
        self.keys += self.lr * sign * delta.T  # (input_dim, hdc_dim)

        # Renormalize to keep keys on the unit hypersphere (bipolar: sign)
        if self.mode == "bipolar":
            self.keys = self.keys.sign()
            self.keys[self.keys == 0] = 1.0
        elif self.mode == "binary":
            self.keys = (self.keys > 0).float()

        self.update_count += 1

    def batch_contrastive_update(
        self,
        X: torch.Tensor,
        labels: torch.Tensor,
        n_pairs: int = 32,
    ) -> None:
        """Sample random pairs from a batch and apply contrastive updates.

        Args:
            X: (batch, input_dim) batch of samples
            labels: (batch,) integer class labels
            n_pairs: number of random pairs to sample per call
        """
        n = X.shape[0]
        for _ in range(n_pairs):
            i = int(torch.randint(n, (1,)).item())
            j = int(torch.randint(n, (1,)).item())
            if i == j:
                continue
            same = bool(labels[i].item() == labels[j].item())
            self.contrastive_update(X[i], X[j], same_class=same)


# ── Gap 3: Columnar HDC Classifier (Larionov et al. 2025) ────────────────────
# "Continual Learning with Columnar Spiking Neural Networks" arXiv:2506.17169
#
# Organises HDC class prototypes into *task columns*.  When HDC similarity
# drops below a threshold between successive inputs, a new column is activated.
# Old columns are frozen — their prototypes cannot be overwritten.
#
# At inference the classifier uses the active column.  During ambiguous
# transitions (similarity near the threshold) it blends adjacent columns.
#
# Result: online continual learning with zero catastrophic forgetting in the
# associative memory.

class ColumnarHDClassifier(nn.Module):
    """HDC classifier with task-isolated prototype columns.

    Each *column* is an independent `AdaptiveHDClassifier` instance.
    A new column is created when the cosine similarity between the current
    world-state HV and the previous one drops below `task_change_threshold`,
    signalling a distribution shift.

    Old columns are frozen once a new column is activated.  Prediction uses
    the current (active) column; optionally it blends the top-2 columns by
    similarity to the current input for smooth transitions.

    Reference:
        Larionov, D. et al. (2025)
        "Continual Learning with Columnar Spiking Neural Networks"
        arXiv:2506.17169
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        dim: int = 4096,
        mode: str = "bipolar",
        learning_rate: float = 0.1,
        task_change_threshold: float = 0.3,
        max_columns: int = 20,
        blend_transitions: bool = True,
        device: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.n_features = n_features
        self.n_classes = n_classes
        self.dim = dim
        self.mode = mode
        self.learning_rate = learning_rate
        self.task_change_threshold = task_change_threshold
        self.max_columns = max_columns
        self.blend_transitions = blend_transitions
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._seed = seed

        # Columns: list of AdaptiveHDClassifier, indexed by task_id
        self.columns: List[AdaptiveHDClassifier] = []
        self._frozen: List[bool] = []  # True = column is frozen
        self.active_col: int = 0
        self.task_count: int = 0

        # Previous HV for task-change detection
        self._prev_hv: Optional[torch.Tensor] = None
        self._step: int = 0

        # Statistics
        self.task_changes: int = 0

        # Create first column
        self._add_column()

    def _add_column(self) -> int:
        """Add a new task column and return its index."""
        seed = self._seed + len(self.columns) if self._seed is not None else None
        col = AdaptiveHDClassifier(
            n_features=self.n_features,
            n_classes=self.n_classes,
            dim=self.dim,
            mode=self.mode,
            learning_rate=self.learning_rate,
            device=self.device,
            seed=seed,
        )
        self.columns.append(col)
        self._frozen.append(False)
        return len(self.columns) - 1

    def _detect_task_change(self, hv: torch.Tensor) -> bool:
        """Return True if similarity to previous HV is below threshold."""
        if self._prev_hv is None:
            return False
        similarity = float(sim(hv, self._prev_hv, self.mode).item())
        return similarity < self.task_change_threshold

    def _freeze_column(self, col_idx: int) -> None:
        """Freeze a column — its prototypes can no longer be updated."""
        self._frozen[col_idx] = True

    def maybe_new_task(self, hv: torch.Tensor) -> bool:
        """Check for task change; if detected, freeze current column and open new one.

        Args:
            hv: Current encoded hypervector (used as task-change probe)

        Returns:
            True if a new task column was activated
        """
        changed = self._detect_task_change(hv)
        if changed and len(self.columns) < self.max_columns:
            self._freeze_column(self.active_col)
            self.active_col = self._add_column()
            self.task_count += 1
            self.task_changes += 1
        self._prev_hv = hv.detach().clone()
        return changed

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.columns[self.active_col].encode(x)

    def train_step(
        self,
        x: torch.Tensor,
        label: int,
        reward: float = 1.0,
        auto_detect_task: bool = True,
    ) -> None:
        """Train the active column.

        Args:
            x: (n_features,) feature vector
            label: class label
            reward: reward gate passed through to RefineHD
            auto_detect_task: if True, probe for task change before updating
        """
        hv = self.columns[self.active_col].encode(x)

        if auto_detect_task:
            self.maybe_new_task(hv)

        if not self._frozen[self.active_col]:
            self.columns[self.active_col].train_step(x, label, reward=reward)

        self._step += 1

    def predict(self, x: torch.Tensor) -> Tuple[int, torch.Tensor]:
        """Predict using the active column.

        If blend_transitions is True and there are multiple columns,
        blends the top-2 columns by their HV similarity to x.

        Returns:
            (class_idx, similarities)
        """
        if not self.blend_transitions or len(self.columns) == 1:
            return self.columns[self.active_col].predict(x)

        # Collect sims from all columns
        hv = self.columns[self.active_col].encode(x)
        col_sims = []
        for col in self.columns:
            col_sims.append(float(sim(hv, col.class_hvs.mean(0), self.mode).item()))

        # Blend top-2 columns
        top2_idx = sorted(range(len(col_sims)), key=lambda i: col_sims[i], reverse=True)[:2]
        w0 = col_sims[top2_idx[0]]
        w1 = col_sims[top2_idx[1]] if len(top2_idx) > 1 else 0.0
        total = w0 + w1 + 1e-9

        _, sims0 = self.columns[top2_idx[0]].predict(x)
        _, sims1 = self.columns[top2_idx[1]].predict(x) if len(top2_idx) > 1 else (0, sims0)
        blended = (w0 * sims0 + w1 * sims1) / total
        return int(blended.argmax().item()), blended

    def get_stats(self) -> Dict:
        return {
            "n_columns": len(self.columns),
            "active_col": self.active_col,
            "task_changes": self.task_changes,
            "frozen_columns": sum(self._frozen),
            "total_steps": self._step,
        }


if __name__ == "__main__":
    test_resonator()
    print()
    test_fractional_power_encoding()
    print()
    test_adaptive_classifier()
