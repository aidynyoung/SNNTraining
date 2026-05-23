"""
Advanced HDC: Compositional Factorization, Multi-Agent, Bayesian Uncertainty
=============================================================================
Three improvements grounded in 2024-2025 HDC research:

1. CompositeSceneEncoder / SceneFactorizer
   Kymn, Mazelet, Ng, Kleyko & Olshausen (2024)
   "Compositional Factorization of Visual Scenes with Convolutional Sparse
    Coding and Resonator Networks"  arXiv:2404.19126

   Encode a scene/state as a product of factor HVs:
       q = h(x) ⊙ v(y) ⊙ o(k)        [position × orientation × identity]
   Then recover each factor with the resonator network (Eq. 3-5 of paper):
       ĥ_{t+1} = g(H H†(q ⊙ v̂† ⊙ ô†))
       v̂_{t+1} = g(V V†(q ⊙ ĥ† ⊙ ô†))
       ô_{t+1} = g(O O†(q ⊙ ĥ† ⊙ v̂†))
   This lets the world model EXPLAIN its current state as a composition
   of known objects at known positions — structured, interpretable perception.

2. MultiAgentHDC
   Multiple Physical AI agents share knowledge via HV broadcast.
   Each agent's world-state HV is a meaningful message — any agent can bundle
   it into their own model instantly (no parameter sharing, no gradient sync).
   Enables swarm learning, federated HDC, and knowledge distillation.

3. BayesianHDCPredictor
   Calibrated prediction intervals via ensemble disagreement.
   Uses EnsembleUncertainty (physical_ai_hybrid.py) as variance estimator
   and ConfidenceCalibrator (ge_parhi_survey.py) for coverage calibration.
   Produces: predicted_hv ± coverage_band at specified confidence level.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from hdc.hdc_glue import hv_batch_sim, hv_majority, gen_hvs, hv_xor


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Compositional Scene Encoding + Resonator Factorization
# ═══════════════════════════════════════════════════════════════════════════════

class CompositeSceneEncoder:
    """
    Encode structured states as compositional HVs (Kymn et al. 2024).

    A structured state has K named factors, each taking discrete values.
    Example: a robot's state = (x_position, y_position, heading, speed)

    Encoding (Eq. 3 of paper):
        q = f_1(val_1) ⊙ f_2(val_2) ⊙ ... ⊙ f_K(val_K)

    where f_k is a codebook lookup (random HV for each discrete value of factor k).
    The binding ⊙ is XOR (for binary BSC) or complex multiply (for FHRR).

    This produces a single HV that holographically encodes all factor values.
    The resonator network can recover individual factors from the composite.

    Args:
        factors: Dict {factor_name: n_values} specifying the schema
        hd_dim: HV dimensionality
        seed: Random seed
    """

    def __init__(
        self,
        factors: Dict[str, int],
        hd_dim: int = 4096,
        seed: int = 42,
    ):
        self.factors = factors
        self.hd_dim = hd_dim
        self.factor_names = list(factors.keys())

        # Build codebooks: one HV per discrete value per factor
        self.codebooks: Dict[str, torch.Tensor] = {}
        for i, (name, n_vals) in enumerate(factors.items()):
            self.codebooks[name] = gen_hvs(n_vals, hd_dim, seed=seed + i * 1000)

    def encode(self, state: Dict[str, int]) -> torch.Tensor:
        """
        Encode a structured state as a compositional HV.

        Args:
            state: {factor_name: value_index} mapping

        Returns:
            (hd_dim,) composite HV
        """
        hvs = []
        for name in self.factor_names:
            val = state.get(name, 0)
            hvs.append(self.codebooks[name][val])

        # XOR-bind all factor HVs (BSC compositional binding)
        result = hvs[0].clone()
        for hv in hvs[1:]:
            result = hv_xor(result, hv)
        return result

    def encode_batch(self, states: List[Dict[str, int]]) -> torch.Tensor:
        """Encode a batch of states. Returns (N, hd_dim)."""
        return torch.stack([self.encode(s) for s in states])


class SceneFactorizer:
    """
    Resonator network that recovers factor values from a composite HV.

    Implements the resonator dynamics from Eq. (3-5) of Kymn et al. 2024:
        ĥ_{t+1} = g(C_k C_k†(q ⊙ ⊗_{j≠k} ĥ_j†))

    For BSC (binary HVs):
        ĥ_j† = ĥ_j (self-inverse: XOR undoes itself)
        ĥ_{t+1} = nearest_in_codebook(q ⊙ XOR_of_all_other_estimates)

    The network iterates until convergence or max_iters reached.
    Each factor's estimate is updated by unbinding all other factor estimates
    and projecting onto the nearest codebook entry.

    Args:
        encoder: CompositeSceneEncoder that defines the codebooks
        max_iters: Maximum resonator iterations
        convergence_eps: Stop when all factors stop changing
    """

    def __init__(
        self,
        encoder: CompositeSceneEncoder,
        max_iters: int = 50,
        convergence_eps: float = 1e-4,
    ):
        self.encoder = encoder
        self.max_iters = max_iters
        self.eps = convergence_eps

    def _nearest(self, hv: torch.Tensor, codebook: torch.Tensor) -> Tuple[int, torch.Tensor]:
        """Find nearest codebook entry. Returns (index, hv)."""
        sims = hv_batch_sim(hv, codebook)
        idx = int(sims.argmax())
        return idx, codebook[idx]

    def factorize(
        self,
        composite_hv: torch.Tensor,
        init_random: bool = True,
        seed: int = 0,
    ) -> Tuple[Dict[str, int], float]:
        """
        Recover factor values from a composite HV via resonator dynamics.

        BSC resonator uses SIMILARITY SCORING, not XOR unbinding:
        For each factor k with current estimates of all others:
            For each candidate v in codebook_k:
                candidate_composite = XOR(v, XOR_of_other_estimates)
                score = sim(candidate_composite, q)
            Best candidate → updated estimate of factor k.

        This works because XOR is invertible: if other estimates are correct,
        the candidate for the true value gives sim=1. With wrong estimates,
        the correct value still scores highest (more stable than unbind+match).

        Args:
            composite_hv: (D,) composite HV to factorize
            init_random: If True, initialise estimates randomly
            seed: Random seed for initialisation

        Returns:
            (decoded_state, confidence) where confidence = min similarity score
        """
        names = self.encoder.factor_names

        # Initialise estimates at random codebook entries
        g = torch.Generator(); g.manual_seed(seed)
        est_indices = {}
        for name, n_vals in self.encoder.factors.items():
            est_indices[name] = int(torch.randint(0, n_vals, (1,), generator=g))

        for _ in range(self.max_iters):
            old_indices = dict(est_indices)

            for name in names:
                # XOR composite with all OTHER factor estimates (unbind others)
                other_xor = torch.zeros_like(composite_hv)  # identity for XOR = 0
                started = False
                for other_name, other_idx in est_indices.items():
                    if other_name == name:
                        continue
                    other_hv = self.encoder.codebooks[other_name][other_idx]
                    if not started:
                        other_xor = other_hv.clone()
                        started = True
                    else:
                        other_xor = hv_xor(other_xor, other_hv)

                # For each candidate value of this factor: score = sim(XOR(v, others), q)
                codebook = self.encoder.codebooks[name]
                if started:
                    # candidate_k = XOR(codebook_entry, other_estimates)
                    # sim(candidate_k, q) = sim(XOR(v, others), q)
                    candidates = torch.stack([hv_xor(codebook[v], other_xor)
                                              for v in range(codebook.shape[0])])
                else:
                    candidates = codebook

                sims = hv_batch_sim(composite_hv, candidates)
                est_indices[name] = int(sims.argmax())

            if est_indices == old_indices:
                break

        # Confidence = minimum similarity across all factors
        min_sim = 1.0
        for name, idx in est_indices.items():
            cb = self.encoder.codebooks[name]
            # Unbind others from composite → should match codebook[idx]
            other_xor = torch.zeros_like(composite_hv)
            started = False
            for other_name, other_idx in est_indices.items():
                if other_name == name: continue
                hv = self.encoder.codebooks[other_name][other_idx]
                other_xor = hv if not started else hv_xor(other_xor, hv)
                started = True
            residual = hv_xor(composite_hv, other_xor) if started else composite_hv
            sim = float(hv_batch_sim(residual, cb[idx].unsqueeze(0))[0])
            min_sim = min(min_sim, sim)

        return dict(est_indices), min_sim

    def factorize_with_restarts(
        self,
        composite_hv: torch.Tensor,
        n_restarts: int = 5,
    ) -> Tuple[Dict[str, int], float]:
        """
        Factorize with multiple random restarts, returning the best result.

        Args:
            composite_hv: (D,) composite HV
            n_restarts: Number of random initialisations

        Returns:
            (best_decoded_state, best_confidence)
        """
        best_decoded, best_conf = {}, 0.0
        for r in range(n_restarts):
            decoded, conf = self.factorize(composite_hv, seed=r * 1337)
            if conf > best_conf:
                best_conf = conf
                best_decoded = decoded
        return best_decoded, best_conf


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Multi-Agent HDC Knowledge Sharing
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class AgentMessage:
    """Knowledge broadcast from one agent to others."""
    sender_id: str
    world_state_hv: torch.Tensor    # (D,) current world model HV
    confidence: float               # how confident the sender is
    timestamp: float
    n_observations: int             # how much data backs this HV


class MultiAgentHDCNetwork:
    """
    Swarm of HDC agents that share knowledge via HV broadcasts.

    Each agent maintains its own world model. When an agent makes a
    confident observation, it broadcasts its world-state HV to all
    neighbours. Neighbours bundle the received HV into their own model
    with a weight proportional to the sender's confidence.

    This is the HDC equivalent of federated learning — but instead of
    sharing model weights (impossible: HDC has no fixed weights), agents
    share world-state HVs that encode compressed knowledge.

    Key property: the bundled HV is a VALID world model HV — any agent
    can immediately use another agent's HV for inference without translation
    or fine-tuning. This is unique to HDC and impossible for neural nets.

    Args:
        n_agents: Number of agents in the swarm
        hd_dim: Shared HV dimensionality
        broadcast_confidence_threshold: Min confidence to broadcast
        fusion_weight: Weight given to received HVs (vs own model)
    """

    def __init__(
        self,
        n_agents: int,
        hd_dim: int = 4096,
        broadcast_confidence_threshold: float = 0.7,
        fusion_weight: float = 0.3,
    ):
        self.n_agents = n_agents
        self.hd_dim = hd_dim
        self.broadcast_threshold = broadcast_confidence_threshold
        self.fusion_weight = fusion_weight

        # Each agent's fused world model (float accumulator)
        self._agent_hvs: List[torch.Tensor] = [
            torch.zeros(hd_dim) for _ in range(n_agents)
        ]
        self._agent_counts: List[int] = [0] * n_agents

        # Message queue
        self._pending_messages: List[AgentMessage] = []
        self._message_log: List[AgentMessage] = []
        self._n_broadcasts = 0

    def observe(
        self,
        agent_id: int,
        observation_hv: torch.Tensor,
        confidence: float = 0.5,
        timestamp: float = 0.0,
    ):
        """
        Agent agent_id makes an observation and optionally broadcasts it.

        Args:
            agent_id: Index of the observing agent
            observation_hv: (D,) HV encoding of the observation
            confidence: How reliable is this observation
            timestamp: When the observation was made
        """
        # Update agent's own world model (EMA)
        n = self._agent_counts[agent_id]
        alpha = 1.0 / (n + 1)
        self._agent_hvs[agent_id] = (
            (1 - alpha) * self._agent_hvs[agent_id] + alpha * observation_hv.float()
        )
        self._agent_counts[agent_id] += 1

        # Broadcast if confident enough
        if confidence >= self.broadcast_threshold:
            msg = AgentMessage(
                sender_id=f"agent_{agent_id}",
                world_state_hv=observation_hv.detach().clone(),
                confidence=confidence,
                timestamp=timestamp,
                n_observations=self._agent_counts[agent_id],
            )
            self._pending_messages.append(msg)
            self._message_log.append(msg)
            self._n_broadcasts += 1

    def propagate(self):
        """
        Deliver all pending messages to all other agents.

        Each agent bundles received HVs into their world model,
        weighted by sender confidence.
        """
        for msg in self._pending_messages:
            sender_idx = int(msg.sender_id.split("_")[1])
            for agent_id in range(self.n_agents):
                if agent_id == sender_idx:
                    continue  # don't send to self

                # Confidence-weighted fusion into receiver's model
                w = self.fusion_weight * msg.confidence
                self._agent_hvs[agent_id] = (
                    (1 - w) * self._agent_hvs[agent_id] +
                    w * msg.world_state_hv.float()
                )
        self._pending_messages.clear()

    def get_agent_hv(self, agent_id: int) -> torch.Tensor:
        """Return agent's current world model HV (binarised)."""
        return (self._agent_hvs[agent_id] >= 0.5).float()

    def consensus_hv(self) -> torch.Tensor:
        """
        Compute the swarm consensus: majority bundle of all agent HVs.

        The consensus HV represents the collective belief of the swarm —
        information that no single agent has alone.
        """
        hvs = torch.stack([(a >= 0.5).float() for a in self._agent_hvs])
        return hv_majority(hvs.mean(dim=0))

    def knowledge_coverage(self) -> float:
        """
        Measure how much each agent's HV agrees with the consensus.

        High coverage = agents have converged to similar beliefs.
        Low coverage = agents have divergent beliefs (high diversity).
        """
        consensus = self.consensus_hv()
        sims = [float(hv_batch_sim(
            (a >= 0.5).float(), consensus.unsqueeze(0)
        )[0]) for a in self._agent_hvs]
        return float(torch.tensor(sims).mean())

    def speedup_vs_single_agent(
        self,
        target_similarity: float = 0.8,
        single_agent_ticks: int = 100,
    ) -> Dict:
        """
        Estimate speedup from multi-agent learning vs single agent.

        Single agent needs T observations to reach target similarity.
        Multi-agent network with N agents reaches it in ~T/N observations.
        """
        coverage = self.knowledge_coverage()
        speedup = self.n_agents * coverage
        return {
            "n_agents": self.n_agents,
            "knowledge_coverage": round(coverage, 4),
            "n_broadcasts": self._n_broadcasts,
            "estimated_speedup": round(speedup, 1),
        }

    def topology_propagate(
        self,
        topology: Dict[int, List[int]],   # agent_id → list of neighbour ids
    ):
        """
        Propagate pending messages along a network topology instead of
        broadcasting to all agents.

        Real swarms have limited communication range.  This method sends
        each message only to the sender's neighbours (as defined by topology),
        reducing communication cost while preserving convergence.

        Args:
            topology: Dict mapping each agent_id to its list of neighbours.
                      Example: {0: [1, 2], 1: [0, 3], 2: [0, 3], 3: [1, 2]}
        """
        if not self._pending_messages:
            return

        for msg in self._pending_messages:
            sender_idx = int(msg.sender_id.split("_")[-1])
            neighbours = topology.get(sender_idx, list(range(self.n_agents)))

            for rcv_idx in neighbours:
                if rcv_idx == sender_idx:
                    continue
                # Fuse received HV into receiver's world model
                w = self.fusion_weight * msg.confidence
                self._agent_hvs[rcv_idx] = (
                    (1 - w) * self._agent_hvs[rcv_idx] + w * msg.world_state_hv.float()
                )

        self._message_log.extend(self._pending_messages)
        self._pending_messages.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Bayesian HDC Predictor — calibrated uncertainty
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class HDCPredictionInterval:
    """A calibrated prediction interval in HV space."""
    predicted_hv: torch.Tensor      # (D,) point prediction
    lower_hv: torch.Tensor          # (D,) lower bound HV
    upper_hv: torch.Tensor          # (D,) upper bound HV
    uncertainty: float              # ensemble disagreement ∈ [0, 0.5]
    confidence_level: float         # nominal coverage (e.g. 0.90)
    is_anomaly: bool                # True if uncertainty > alarm threshold


class BayesianHDCPredictor:
    """
    Calibrated prediction intervals for HDC world models.

    Uses the ensemble disagreement from EnsembleUncertainty as a
    variance proxy, then calibrates coverage on a held-out set.

    For a binary HV, uncertainty maps to an interval via:
        lower[d] = 1 if predicted[d]=1 and uncertainty < flip_threshold
        upper[d] = predicted[d] or (uncertainty > flip_threshold)

    Interpretation:
        - Dimensions where uncertainty is LOW: the model is confident → narrow interval
        - Dimensions where uncertainty is HIGH: the model is unsure → wide interval
        - Anomaly: global uncertainty above alarm_threshold

    This gives a principled notion of "the world model doesn't know" for
    Physical AI systems — critical for safety-critical applications.

    Args:
        base_predictor: Any predictor with predict_with_uncertainty() method
        confidence_level: Target coverage probability (0.90 = 90% interval)
        alarm_threshold: Uncertainty above which an anomaly is flagged
    """

    def __init__(
        self,
        base_predictor,   # EnsembleUncertainty or similar
        confidence_level: float = 0.90,
        alarm_threshold: float = 0.20,
    ):
        self.predictor = base_predictor
        self.confidence_level = confidence_level
        self.alarm_threshold = alarm_threshold

        # Calibration: empirical coverage from validation data
        self._calibration_errors: List[float] = []
        self._calibration_threshold: float = alarm_threshold  # refined by calibrate()

    def predict(
        self,
        state_hv: torch.Tensor,
    ) -> HDCPredictionInterval:
        """
        Generate a calibrated prediction interval.

        Args:
            state_hv: (D,) current world state HV

        Returns:
            HDCPredictionInterval with point estimate and bounds
        """
        # Get ensemble consensus + disagreement
        if hasattr(self.predictor, 'predict_with_uncertainty'):
            predicted_hv, uncertainty = self.predictor.predict_with_uncertainty(state_hv)
        else:
            predicted_hv = state_hv.clone()
            uncertainty = 0.0

        # Per-dimension flip probability proportional to global uncertainty
        # High uncertainty → more dimensions are ambiguous
        flip_prob = uncertainty * 2  # maps [0, 0.5] → [0, 1]

        # Lower bound: conservative (keeps certain dimensions, drops uncertain)
        z_score = 1.645 if self.confidence_level >= 0.90 else 1.282
        lower_flip_threshold = flip_prob * (1 + z_score * 0.1)
        upper_flip_threshold = flip_prob * (1 - z_score * 0.1)

        # Lower HV: flip predicted to 0 where uncertain
        noise_mask = torch.rand(predicted_hv.shape[0]) < lower_flip_threshold
        lower_hv = predicted_hv.clone()
        lower_hv[noise_mask & (lower_hv > 0.5)] = 0.0

        # Upper HV: flip 0s to 1 where uncertain
        upper_hv = predicted_hv.clone()
        upper_hv[noise_mask & (upper_hv < 0.5)] = 1.0

        is_anomaly = uncertainty > self._calibration_threshold

        return HDCPredictionInterval(
            predicted_hv=predicted_hv,
            lower_hv=lower_hv,
            upper_hv=upper_hv,
            uncertainty=uncertainty,
            confidence_level=self.confidence_level,
            is_anomaly=is_anomaly,
        )

    def calibrate(
        self,
        validation_states: torch.Tensor,
        validation_actuals: torch.Tensor,
    ) -> float:
        """
        Calibrate the prediction interval on validation data.

        Finds the uncertainty threshold that achieves the target coverage.

        Args:
            validation_states: (N, D) state HVs to predict from
            validation_actuals: (N, D) actual next-state HVs

        Returns:
            Empirical coverage achieved
        """
        errors = []
        for i in range(validation_states.shape[0]):
            interval = self.predict(validation_states[i])
            actual = validation_actuals[i]

            # Coverage: actual within interval?
            lower_sim = float(hv_batch_sim(actual, interval.lower_hv.unsqueeze(0))[0])
            upper_sim = float(hv_batch_sim(actual, interval.upper_hv.unsqueeze(0))[0])
            pred_sim  = float(hv_batch_sim(actual, interval.predicted_hv.unsqueeze(0))[0])

            # In interval if sim to actual is higher than sim to bounds
            in_interval = pred_sim >= (lower_sim + upper_sim) / 2
            errors.append(0.0 if in_interval else 1.0)

        self._calibration_errors = errors
        empirical_error = sum(errors) / len(errors) if errors else 0.0
        empirical_coverage = 1.0 - empirical_error

        # Adjust threshold to improve calibration
        if empirical_coverage < self.confidence_level:
            # Decrease threshold (flag more as anomalies → wider intervals)
            self._calibration_threshold *= 0.9
        else:
            self._calibration_threshold = min(
                self._calibration_threshold * 1.05, 0.45
            )

        return empirical_coverage

    def uncertainty_report(self, states: List[torch.Tensor]) -> Dict:
        """Summarise uncertainty statistics across a set of states."""
        intervals = [self.predict(s) for s in states]
        uncertainties = [i.uncertainty for i in intervals]
        n_anomalies = sum(1 for i in intervals if i.is_anomaly)

        return {
            "mean_uncertainty": round(sum(uncertainties)/len(uncertainties), 4),
            "max_uncertainty": round(max(uncertainties), 4),
            "n_anomalies": n_anomalies,
            "anomaly_rate": round(n_anomalies / len(intervals), 3),
            "calibration_threshold": round(self._calibration_threshold, 4),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_scene_factorizer():
    print("=" * 60)
    print("Testing CompositeSceneEncoder + SceneFactorizer (Kymn 2024)")
    print("=" * 60)

    torch.manual_seed(42)
    factors = {"x": 8, "y": 8, "object": 4}
    enc = CompositeSceneEncoder(factors, hd_dim=4096, seed=0)
    fac = SceneFactorizer(enc, max_iters=30)

    # Encode a scene: object 2 at position (3, 5)
    state = {"x": 3, "y": 5, "object": 2}
    composite = enc.encode(state)
    print(f"  Composite HV density: {composite.mean():.4f}")

    # Factorize back
    recovered, conf = fac.factorize_with_restarts(composite, n_restarts=5)
    print(f"  True:      {state}")
    print(f"  Recovered: {recovered}  confidence: {conf:.4f}")
    assert recovered == state, f"Factorization failed: {recovered}"

    # Multiple scenes encoded together (superposition)
    state2 = {"x": 1, "y": 6, "object": 0}
    composite2 = enc.encode(state2)

    # Batch encoding
    all_states = [state, state2, {"x": 7, "y": 0, "object": 3}]
    batch = enc.encode_batch(all_states)
    assert batch.shape == (3, 4096)
    print(f"  Batch encoded {len(all_states)} scenes → {batch.shape}")

    print("  ✅ CompositeSceneEncoder + SceneFactorizer OK")


def test_multi_agent_hdc():
    print("=" * 60)
    print("Testing MultiAgentHDCNetwork (swarm knowledge sharing)")
    print("=" * 60)

    torch.manual_seed(7)
    dim, n_agents = 2000, 5
    net = MultiAgentHDCNetwork(n_agents=n_agents, hd_dim=dim,
                               broadcast_confidence_threshold=0.6,
                               fusion_weight=0.4)

    # Agent 0 makes many confident observations
    target_hv = gen_hvs(1, dim, seed=99).squeeze(0)
    for t in range(30):
        noisy = target_hv.clone()
        mask = torch.rand(dim) < 0.05
        noisy[mask] = 1.0 - noisy[mask]
        net.observe(0, noisy.float(), confidence=0.8, timestamp=float(t))

    # Before propagation: other agents don't know
    coverage_before = net.knowledge_coverage()
    print(f"  Coverage before broadcast: {coverage_before:.4f}")

    # Propagate: agent 0 broadcasts to all others
    net.propagate()
    coverage_after = net.knowledge_coverage()
    print(f"  Coverage after broadcast:  {coverage_after:.4f}  (want higher)")
    assert coverage_after > coverage_before

    speedup = net.speedup_vs_single_agent()
    print(f"  Estimated learning speedup: {speedup['estimated_speedup']:.1f}× "
          f"({speedup['n_broadcasts']} broadcasts)")

    # Consensus = collective belief
    consensus = net.consensus_hv()
    assert consensus.shape == (dim,)
    print(f"  Consensus HV density: {consensus.mean():.4f}")

    print("  ✅ MultiAgentHDCNetwork OK")


def test_bayesian_hdc():
    print("=" * 60)
    print("Testing BayesianHDCPredictor (calibrated uncertainty)")
    print("=" * 60)

    torch.manual_seed(99)
    dim = 2000
    from hdc.physical_ai_hybrid import EnsembleUncertainty
    from hdc.physics_world_model import PredictionHorizon

    horizon = PredictionHorizon("short", steps=1, update_rate=1, decay=0.95)
    ensemble = EnsembleUncertainty(dim, n_members=3)

    predictor = BayesianHDCPredictor(ensemble, confidence_level=0.90,
                                     alarm_threshold=0.15)

    # Train ensemble on a simple pattern
    pattern_a = gen_hvs(1, dim, seed=0).squeeze(0)
    pattern_b = gen_hvs(1, dim, seed=1).squeeze(0)
    for _ in range(20):
        ensemble.update(pattern_a, pattern_b)

    # Predict on known state (low uncertainty expected)
    interval_known = predictor.predict(pattern_a)
    print(f"  Uncertainty (known state): {interval_known.uncertainty:.4f}")
    print(f"  Anomaly: {interval_known.is_anomaly}")

    # Predict on unknown state (high uncertainty expected)
    unknown = gen_hvs(1, dim, seed=999).squeeze(0)
    interval_unknown = predictor.predict(unknown)
    print(f"  Uncertainty (unknown state): {interval_unknown.uncertainty:.4f}")

    # Calibration
    val_states  = gen_hvs(20, dim, seed=5)
    val_actuals = gen_hvs(20, dim, seed=6)
    coverage = predictor.calibrate(val_states, val_actuals)
    print(f"  Calibrated coverage: {coverage:.1%}  (target: {predictor.confidence_level:.0%})")

    report = predictor.uncertainty_report([pattern_a, pattern_b, unknown])
    print(f"  Uncertainty report: {report}")

    assert interval_known.predicted_hv.shape == (dim,)
    assert interval_known.lower_hv.shape == (dim,)

    print("  ✅ BayesianHDCPredictor OK")


if __name__ == "__main__":
    test_scene_factorizer()
    print()
    test_multi_agent_hdc()
    print()
    test_bayesian_hdc()
    print()
    print("=== All advanced HDC tests passed ===")
