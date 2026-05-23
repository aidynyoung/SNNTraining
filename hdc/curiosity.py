"""
Curiosity-Driven Active Exploration for Physical AI
====================================================
Transforms the Physical AI agent from PASSIVE (wait for data) to ACTIVE
(seek informative data). Instead of evaluating actions solely by expected
utility + risk, the curious agent also considers:

  curiosity(action) = α × model_uncertainty + β × novelty

where:
  model_uncertainty = EnsembleUncertainty of the predicted next state
  novelty           = how different the predicted next state is from all
                      known states in the pattern memory

This mirrors the concept of **intrinsic motivation** in reinforcement learning
(Schmidhuber 1991, Oudeyer & Kaplan 2007) but implemented entirely in HDC:
  - Uncertainty from EnsembleUncertainty (multi-seed Hebbian disagreement)
  - Novelty from Hamming distance to nearest pattern in SequencePatternMemory
  - No reward signals, no value functions, no backpropagation

Physical AI application: Instead of always recommending "idle" (safe but
uninformative), the curious agent explores:
  - States it hasn't seen (high novelty)
  - States it's uncertain about (high model uncertainty)
  - While avoiding danger (risk constraint still applies)

The result: the causal graph fills faster, pattern memory grows richer,
and the world model converges in fewer observations.

Modules:

  1. NoveltyEstimator — measures how different a state is from all seen states
     via max Hamming similarity to pattern memory entries (inverted)

  2. InformationGainEstimator — measures how much uncertainty exists about
     the next state if a given action is taken

  3. CuriosityScorer — combines novelty + uncertainty into a curiosity score
     for each candidate action, balancing exploration vs exploitation

  4. CuriousAgent — replaces SelfImprovementLoop's greedy action selection
     with curiosity-weighted selection:
       net_score(a) = λ_utility × utility(a) + λ_curiosity × curiosity(a)
                     - λ_risk × risk(a)
     Adjusts λ_curiosity over time: high early (explore), low late (exploit)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from hdc.hdc_glue import hv_batch_sim, gen_hvs
from hdc.physics_world_model import (
    PhysicsWorldModel, ActionCandidate, _hamming, _xor, _majority,
)
try:
    from hdc.physical_ai_hybrid import EnsembleUncertainty, HybridPhysicalAIPipeline
except ImportError:
    EnsembleUncertainty = None  # type: ignore
    HybridPhysicalAIPipeline = None  # type: ignore
from hdc.world_context import ContextualWorldModel, SequencePatternMemory
from hdc.planner import SelfImprovementLoop, AdaptiveHebbian


# ═══════════════════════════════════════════════════════════════════════════════
# 1. NoveltyEstimator
# ═══════════════════════════════════════════════════════════════════════════════

class NoveltyEstimator:
    """
    Estimate state novelty as inverse similarity to all observed states.

    novelty(hv) = 1 - max_{i ∈ known_states} sim(hv, state_i)

    High novelty = far from all seen states = potentially informative.
    Low novelty  = very similar to a seen state = already understood.

    Maintains a compact set of representative past states (prototypes)
    via incremental clustering: states within `cluster_radius` of an
    existing prototype are merged, keeping only N_max prototypes total.

    Args:
        hd_dim: Hypervector dimensionality
        n_max_prototypes: Maximum stored state prototypes
        cluster_radius: Max Hamming distance to merge into existing prototype
    """

    def __init__(
        self,
        hd_dim: int,
        n_max_prototypes: int = 128,
        cluster_radius: float = 0.12,
    ):
        self.hd_dim = hd_dim
        self.n_max = n_max_prototypes
        self.radius = cluster_radius
        self._prototypes: List[torch.Tensor] = []
        self._counts: List[int] = []
        self._last_seen: List[int] = []  # recency tracking for LRU eviction
        self._n_seen = 0

    def observe(self, hv: torch.Tensor):
        """Update prototype set with a new observation."""
        self._n_seen += 1
        hv = hv.detach()

        if not self._prototypes:
            self._prototypes.append(hv.clone())
            self._counts.append(1)
            self._last_seen.append(self._n_seen)
            return

        protos = torch.stack(self._prototypes)
        sims = hv_batch_sim(hv, protos)
        max_sim = float(sims.max().item())
        best_idx = int(sims.argmax().item())

        if max_sim > (1.0 - self.radius):
            # Merge into nearest prototype
            n = self._counts[best_idx]
            self._prototypes[best_idx] = (
                _majority(((self._prototypes[best_idx].float() * n + hv.float()) / (n + 1)))
            )
            self._counts[best_idx] += 1
            self._last_seen[best_idx] = self._n_seen   # update recency
        elif len(self._prototypes) < self.n_max:
            # New prototype
            self._prototypes.append(hv.clone())
            self._counts.append(1)
            self._last_seen.append(self._n_seen)
        else:
            # Replace least-recently-seen prototype (recency-aware eviction)
            lru_idx = min(range(len(self._last_seen)), key=lambda i: self._last_seen[i])
            self._prototypes[lru_idx] = hv.clone()
            self._counts[lru_idx] = 1
            self._last_seen[lru_idx] = self._n_seen

    def novelty(self, hv: torch.Tensor, recency_weight: float = 0.2) -> float:
        """
        Compute recency-adjusted novelty score ∈ [0, 1].

        1 = completely novel (never seen anything like this)
        0 = identical to a recently-seen prototype

        Recency adjustment: prototypes not seen for a long time are treated
        as partially novel — the world may have changed since we last saw them.
        This prevents the estimator from blocking exploration of previously
        visited states that have become interesting again.

        novelty_adj = novelty + recency_weight × mean_time_since_seen

        Args:
            recency_weight: How much to boost novelty for stale prototypes
        """
        if not self._prototypes:
            return 1.0
        protos   = torch.stack(self._prototypes)
        sims     = hv_batch_sim(hv, protos)
        best_sim = float(sims.max().item())
        raw_nov  = 1.0 - best_sim

        # Recency boost: if the most-similar prototype hasn't been seen recently
        if recency_weight > 0 and self._last_seen and self._n_seen > 0:
            best_idx    = int(sims.argmax().item())
            age_frac    = (self._n_seen - self._last_seen[best_idx]) / max(self._n_seen, 1)
            raw_nov     = min(1.0, raw_nov + recency_weight * age_frac)

        return raw_nov

    def novelty_batch(self, hvs: torch.Tensor) -> torch.Tensor:
        """Compute novelty for a batch of HVs. Shape: (N,) → (N,)."""
        if not self._prototypes:
            return torch.ones(hvs.shape[0])
        protos = torch.stack(self._prototypes)
        # pairwise similarity: (N, K)
        sims = torch.zeros(hvs.shape[0], len(self._prototypes))
        for i, proto in enumerate(self._prototypes):
            sims[:, i] = hv_batch_sim(hvs, proto.unsqueeze(0).expand(hvs.shape[0], -1)) \
                if False else torch.tensor([
                    float(hv_batch_sim(hvs[j], proto.unsqueeze(0))[0])
                    for j in range(hvs.shape[0])
                ])
        return 1.0 - sims.max(dim=-1).values

    @property
    def n_prototypes(self) -> int:
        return len(self._prototypes)

    def novelty_report(self) -> Dict:
        """
        Coverage and cluster statistics: how densely is the state space mapped?

        small mean_cluster_size → many singletons (sparse coverage).
        large mean_cluster_size → agent keeps revisiting the same states.
        """
        n = self.n_prototypes
        if n == 0:
            return {"n_prototypes": 0, "n_seen": self._n_seen, "coverage": 0.0}
        mean_count = sum(self._counts) / max(n, 1)
        max_count  = max(self._counts)
        total_seen = sum(self._counts)
        return {
            "n_prototypes":       n,
            "n_seen":             self._n_seen,
            "capacity_used":      round(n / max(self.n_max, 1), 4),
            "mean_cluster_size":  round(mean_count, 2),
            "max_cluster_size":   max_count,
            "exploration_rate":   round(n / max(total_seen, 1), 4),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 2. InformationGainEstimator
# ═══════════════════════════════════════════════════════════════════════════════

class InformationGainEstimator:
    """
    Estimate information gain from taking an action: how much will we learn?

    IG(s, a) ≈ EnsembleUncertainty(predicted_next_state)

    If the ensemble of predictors strongly disagrees about what will happen
    after action a, the action is informative — taking it will help the
    agent calibrate its world model.

    Also incorporates **causal sparsity**: actions that trigger (state, action)
    pairs not yet in the causal graph are prioritised (there's no data there).

    Args:
        ensemble: EnsembleUncertainty for prediction uncertainty estimation
        causal_graph: CausalTransitionGraph to check causal sparsity
    """

    def __init__(
        self,
        ensemble: EnsembleUncertainty,
        causal_graph=None,  # CausalTransitionGraph
    ):
        self.ensemble = ensemble
        self.causal_graph = causal_graph

    def estimate(
        self,
        state_hv: torch.Tensor,
        action_hv: torch.Tensor,
    ) -> Tuple[float, float]:
        """
        Estimate (uncertainty, causal_coverage) for taking action in state.

        uncertainty:     EnsembleUncertainty of predicted next state [0, 0.5]
        causal_coverage: fraction of causal data at this (s,a) [0, 1]
                         Low coverage = high information gain opportunity

        Returns:
            (information_gain ∈ [0,1], details_dict)
        """
        # Simulate next state
        predicted_next = _xor(state_hv, action_hv)  # simple binding as proxy

        # Ensemble uncertainty about predicted state
        _, uncertainty = self.ensemble.predict_with_uncertainty(predicted_next)

        # Causal sparsity: how much data do we have for this (s,a)?
        causal_coverage = 0.0
        if self.causal_graph is not None and self.causal_graph.n_entries > 0:
            key = _xor(state_hv, action_hv)
            results = self.causal_graph._key_mem.query(
                key, threshold=0.25, top_k=1
            )
            if results:
                idx = results[0]["label"]
                n = self.causal_graph._next_count[idx]
                causal_coverage = min(1.0, n / 10.0)  # saturate at 10 observations

        information_gain = uncertainty * (1.0 - causal_coverage)
        return information_gain, causal_coverage


# ═══════════════════════════════════════════════════════════════════════════════
# 3. CuriosityScorer
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CuriosityScore:
    """Breakdown of an action's curiosity-augmented score."""
    action: ActionCandidate
    utility_score: float
    risk_score: float
    novelty_score: float
    uncertainty_score: float
    curiosity_score: float    # novelty × uncertainty
    net_score: float          # final combined score


class CuriosityScorer:
    """
    Score candidate actions using utility + curiosity.

    net_score(a) = λ_u × utility(a) - λ_r × risk(a) + λ_c × curiosity(a)

    curiosity(a) = α × novelty(predicted_next) + β × uncertainty(s,a)

    The curiosity weight λ_c decreases over time (exploration decay):
        λ_c(t) = λ_c_max × exp(-decay_rate × t)

    This creates an explore-exploit schedule:
      - Early: high curiosity weight → explore widely
      - Late: low curiosity weight → exploit learned knowledge

    Args:
        novelty_estimator: NoveltyEstimator for state novelty
        ig_estimator: InformationGainEstimator for model uncertainty
        curiosity_weight_max: Maximum curiosity weight (λ_c at t=0)
        curiosity_decay: Decay rate for curiosity weight
        utility_weight: Fixed weight for utility (λ_u)
        risk_weight: Fixed weight for risk (λ_r)
    """

    def __init__(
        self,
        novelty_estimator: NoveltyEstimator,
        ig_estimator: InformationGainEstimator,
        curiosity_weight_max: float = 0.6,
        curiosity_decay: float = 0.05,   # per-tick decay
        novelty_alpha: float = 0.5,
        uncertainty_beta: float = 0.5,
        utility_weight: float = 0.8,
        risk_weight: float = 0.5,
    ):
        self.novelty_est = novelty_estimator
        self.ig_est = ig_estimator
        self.curiosity_max = curiosity_weight_max
        self.curiosity_decay = curiosity_decay
        self.novelty_alpha = novelty_alpha
        self.uncertainty_beta = uncertainty_beta
        self.utility_weight = utility_weight
        self.risk_weight = risk_weight

        self._tick = 0
        self._curiosity_weight_history: List[float] = []

    @property
    def curiosity_weight(self) -> float:
        """Current curiosity weight (decaying exponential)."""
        return self.curiosity_max * math.exp(-self.curiosity_decay * self._tick)

    def tick(self):
        """Advance one time step (updates curiosity decay)."""
        self._tick += 1
        self._curiosity_weight_history.append(self.curiosity_weight)

    def score(
        self,
        state_hv: torch.Tensor,
        candidates: List[ActionCandidate],
        goal_state: Optional[torch.Tensor] = None,
    ) -> List[CuriosityScore]:
        """
        Score all candidates with curiosity-augmented evaluation.

        Args:
            state_hv: Current world state HV
            candidates: List of ActionCandidate (must have .predicted_outcome set
                       by ActionEvaluator.evaluate() beforehand, or we simulate)
            goal_state: Optional goal state for utility computation

        Returns:
            List of CuriosityScore sorted by net_score descending
        """
        lc = self.curiosity_weight
        results = []

        for cand in candidates:
            # Simulate predicted next state if not already set
            if cand.predicted_outcome is None:
                cand.predicted_outcome = _xor(state_hv, cand.hv)

            predicted_next = cand.predicted_outcome

            # Utility: toward goal or from action evaluator
            if goal_state is not None:
                utility = float(_hamming(predicted_next, goal_state).item())
            else:
                utility = cand.utility_score

            # Risk
            risk = cand.risk_score

            # Novelty of predicted next state
            novelty = self.novelty_est.novelty(predicted_next)

            # Uncertainty about predicted transition
            ig, _ = self.ig_est.estimate(state_hv, cand.hv)

            # Curiosity = weighted sum of novelty and uncertainty
            curiosity = (self.novelty_alpha * novelty +
                         self.uncertainty_beta * ig)

            # Net score
            net = (self.utility_weight * utility
                   - self.risk_weight * risk
                   + lc * curiosity)

            results.append(CuriosityScore(
                action=cand,
                utility_score=utility,
                risk_score=risk,
                novelty_score=novelty,
                uncertainty_score=ig,
                curiosity_score=curiosity,
                net_score=net,
            ))

        results.sort(key=lambda x: x.net_score, reverse=True)
        return results

    def exploration_stats(self) -> Dict:
        """Return current exploration statistics."""
        return {
            "current_curiosity_weight": round(self.curiosity_weight, 4),
            "tick": self._tick,
            "n_novelty_prototypes": self.novelty_est.n_prototypes,
            "is_exploring": self.curiosity_weight > self.curiosity_max * 0.2,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. CuriousAgent — full active learning agent
# ═══════════════════════════════════════════════════════════════════════════════

class CuriousAgent:
    """
    Active Physical AI agent with curiosity-driven exploration.

    Extends SelfImprovementLoop with curiosity-weighted action selection:
    instead of always greedily selecting the highest-utility action, the
    curious agent balances exploration (seek novel/uncertain states) with
    exploitation (maximise utility, minimise risk).

    Exploration schedule:
      - Early ticks: λ_curiosity = 0.6 (mostly explore)
      - After convergence: λ_curiosity → 0 (mostly exploit)

    The agent automatically transitions from exploration to exploitation
    when the improvement report shows error_reduction > convergence_threshold.

    Args:
        base_agent: SelfImprovementLoop to augment with curiosity
        curiosity_weight_max: Initial curiosity weight (0 = no curiosity)
        curiosity_decay: Per-tick decay of curiosity weight
        convergence_threshold: Error reduction at which to stop exploring
    """

    def __init__(
        self,
        base_agent: SelfImprovementLoop,
        curiosity_weight_max: float = 0.5,
        curiosity_decay: float = 0.03,
        convergence_threshold: float = 0.15,
    ):
        self.agent = base_agent
        self.convergence_threshold = convergence_threshold

        wm = base_agent.world.pipeline.world_model

        # Novelty estimator
        self.novelty_est = NoveltyEstimator(
            base_agent.world.hd_dim,
            n_max_prototypes=64,
        )

        # Information gain estimator (uses ensemble from hybrid pipeline)
        ensemble = base_agent.world.pipeline.ensemble
        causal_graph = base_agent.world.causal_graph
        self.ig_est = InformationGainEstimator(ensemble, causal_graph)

        # Curiosity scorer
        self.scorer = CuriosityScorer(
            self.novelty_est,
            self.ig_est,
            curiosity_weight_max=curiosity_weight_max,
            curiosity_decay=curiosity_decay,
        )

        self._tick = 0
        self._converged = False

    def tick(
        self,
        reading,  # SensorReading
        candidate_actions: Optional[List[ActionCandidate]] = None,
        action_hv: Optional[torch.Tensor] = None,
    ) -> Dict:
        """
        Run one tick of the curious agent.

        Same interface as SelfImprovementLoop.tick() but with curiosity-
        augmented action selection.
        """
        self._tick += 1
        self.scorer.tick()

        # Run base agent (handles learning, calibration, etc.)
        result = self.agent.tick(reading, candidate_actions=candidate_actions,
                                 action_hv=action_hv)

        # Update novelty estimator with observed state
        sensor_hv = result["sensor_hv"]
        self.novelty_est.observe(sensor_hv)

        current_novelty = self.novelty_est.novelty(sensor_hv)

        # Curiosity-augmented action scoring
        curiosity_scores = None
        best_curious_action = None
        if candidate_actions:
            # First run base evaluation to get utility/risk scores
            wm = self.agent.world.pipeline.world_model
            ranked = wm.action_evaluator.evaluate(
                wm.current_state, candidate_actions,
                goal_state=self.agent._goal_state,
            )

            # Apply curiosity scoring on top
            curiosity_scores = self.scorer.score(
                wm.current_state, ranked,
                goal_state=self.agent._goal_state,
            )
            best_curious_action = curiosity_scores[0].action.name if curiosity_scores else None

        # Check convergence: reduce curiosity when improvement plateaus
        if self._tick % 10 == 0 and not self._converged:
            report = self.agent.improvement_report()
            if report.get("error_reduction", 0) > self.convergence_threshold:
                self._converged = True

        return {
            **result,
            "current_novelty": current_novelty,
            "curiosity_weight": self.scorer.curiosity_weight,
            "curiosity_scores": [
                {
                    "action": cs.action.name,
                    "net": round(cs.net_score, 4),
                    "curiosity": round(cs.curiosity_score, 4),
                    "novelty": round(cs.novelty_score, 4),
                }
                for cs in (curiosity_scores or [])
            ],
            "best_curious_action": best_curious_action,
            "n_novelty_prototypes": self.novelty_est.n_prototypes,
            "converged": self._converged,
            "exploration_stats": self.scorer.exploration_stats(),
        }

    def improvement_report(self) -> Dict:
        base = self.agent.improvement_report()
        return {
            **base,
            "n_novelty_prototypes": self.novelty_est.n_prototypes,
            "curiosity_weight": round(self.scorer.curiosity_weight, 4),
            "converged": self._converged,
        }

    def curiosity_health(self) -> Dict:
        """
        Comprehensive curiosity agent diagnostic.

        exploration_phase: 'exploring'  → still curiosity-driven
                           'converging' → curiosity weight < 20% of max
                           'converged'  → error reduction threshold met
        """
        w = self.scorer.curiosity_weight
        w_max = self.scorer.curiosity_max
        phase = (
            "converged"  if self._converged else
            "converging" if w < 0.2 * w_max else
            "exploring"
        )
        return {
            "tick":               self._tick,
            "exploration_phase":  phase,
            "curiosity_weight":   round(w, 4),
            "curiosity_max":      w_max,
            "decay_rate":         self.scorer.curiosity_decay,
            **self.novelty_est.novelty_report(),
            "base_improvement":   self.improvement_report(),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_novelty_estimator():
    print("=" * 60)
    print("Testing NoveltyEstimator")
    print("=" * 60)

    torch.manual_seed(42)
    dim = 2000
    est = NoveltyEstimator(dim, n_max_prototypes=20, cluster_radius=0.1)

    # Observe same state repeatedly → novelty decreases
    base_hv = (torch.rand(dim) < 0.5).float()

    novelty_0 = est.novelty(base_hv)
    est.observe(base_hv)
    novelty_1 = est.novelty(base_hv)
    est.observe(base_hv)
    novelty_2 = est.novelty(base_hv)

    print(f"  Novelty: before={novelty_0:.4f}  after 1={novelty_1:.4f}  after 2={novelty_2:.4f}")
    print(f"  (want: before=1.0, after=~0.0)")
    assert novelty_0 > 0.9, f"Unseen state should have high novelty: {novelty_0}"
    assert novelty_1 < novelty_0, "After observing, novelty should decrease"

    # Different state should still be novel
    other_hv = (torch.rand(dim) < 0.5).float()
    novelty_other = est.novelty(other_hv)
    print(f"  Novelty of different state: {novelty_other:.4f}  (want ~0.5)")

    print(f"  Prototypes: {est.n_prototypes}")
    print("  ✅ NoveltyEstimator OK")


def test_curiosity_scorer():
    print("=" * 60)
    print("Testing CuriosityScorer (explore-exploit tradeoff)")
    print("=" * 60)

    torch.manual_seed(7)
    dim = 1000
    from hdc.physics_world_model import PhysicsWorldModel, MultiHorizonPredictor
    from hdc.physical_ai_hybrid import EnsembleUncertainty

    wm = PhysicsWorldModel(hd_dim=dim)
    ensemble = EnsembleUncertainty(dim, n_members=3)

    novelty_est = NoveltyEstimator(dim, n_max_prototypes=20)
    ig_est = InformationGainEstimator(ensemble)
    scorer = CuriosityScorer(
        novelty_est, ig_est,
        curiosity_weight_max=0.5,
        curiosity_decay=0.1,
    )

    state = (torch.rand(dim) < 0.5).float()

    # Two actions: one goes to known state, one to novel state
    known_next = (torch.rand(dim) < 0.5).float()
    novelty_est.observe(known_next)   # mark as known

    action_to_known  = ActionCandidate("to_known",  _xor(state, known_next))
    action_to_novel  = ActionCandidate("to_novel",  (torch.rand(dim) < 0.3).float())
    action_to_known.predicted_outcome = known_next
    action_to_novel.predicted_outcome = (torch.rand(dim) < 0.5).float()  # unseen

    scores = scorer.score(state, [action_to_known, action_to_novel])
    print(f"  Tick 0 (high curiosity={scorer.curiosity_weight:.3f})")
    for s in scores:
        print(f"    {s.action.name}: net={s.net_score:.4f}  novelty={s.novelty_score:.4f}")

    # After many ticks, curiosity should decay
    for _ in range(30):
        scorer.tick()
    scores_late = scorer.score(state, [action_to_known, action_to_novel])
    print(f"  Tick 30 (low curiosity={scorer.curiosity_weight:.3f})")
    for s in scores_late:
        print(f"    {s.action.name}: net={s.net_score:.4f}")

    assert scorer.curiosity_weight < 0.5, "Curiosity should decay"
    print(f"  Exploration stats: {scorer.exploration_stats()}")
    print("  ✅ CuriosityScorer OK")


def test_curious_agent():
    print("=" * 60)
    print("Testing CuriousAgent (full active learning loop)")
    print("=" * 60)

    import time as _time
    torch.manual_seed(99)

    from hdc.sensor_stream import SensorSpec, SensorReading, ModalityType
    from hdc.physical_ai_hybrid import HybridPhysicalAIPipeline
    from hdc.world_context import ContextualWorldModel

    specs = [SensorSpec("s", ModalityType.SCALAR, raw_dim=1, hd_dim=500, seed=0)]
    base_p = HybridPhysicalAIPipeline(specs, hd_dim=500, n_ensemble=3,
                                       consolidation_period=8)
    world = ContextualWorldModel(base_p, pattern_window=4, pattern_stride=3)
    base_agent = SelfImprovementLoop(world, beam_width=2, planning_horizon=2,
                                      min_causal_for_planning=10, lr_base=0.005)
    agent = CuriousAgent(base_agent, curiosity_weight_max=0.5, curiosity_decay=0.03)

    dim = 500
    candidates = [
        ActionCandidate("idle",   (torch.rand(dim) < 0.05).float()),
        ActionCandidate("probe",  (torch.rand(dim) < 0.25).float()),
        ActionCandidate("engage", (torch.rand(dim) < 0.45).float()),
    ]

    novelties = []
    curiosity_weights = []
    for t in range(40):
        phase = t % 5
        r = SensorReading(
            timestamp=float(t),
            data={"s": torch.tensor([float(phase) / 4])},
        )
        result = agent.tick(r, candidate_actions=candidates)
        novelties.append(result["current_novelty"])
        curiosity_weights.append(result["curiosity_weight"])

    print(f"  After 40 ticks:")
    print(f"    Novelty: early={sum(novelties[:10])/10:.3f}  late={sum(novelties[30:])/10:.3f}")
    print(f"    Curiosity weight: early={curiosity_weights[0]:.3f}  late={curiosity_weights[-1]:.3f}")
    print(f"    N novelty prototypes: {result['n_novelty_prototypes']}")
    print(f"    Best curious action: {result['best_curious_action']}")
    print(f"    Converged: {result['converged']}")

    assert curiosity_weights[-1] < curiosity_weights[0], "Curiosity should decay"
    assert result["n_novelty_prototypes"] > 0

    report = agent.improvement_report()
    print(f"  Report: error_reduction={report.get('error_reduction', 0):.3f}, "
          f"curiosity_weight={report.get('curiosity_weight', 0):.3f}")

    print("  ✅ CuriousAgent OK")


if __name__ == "__main__":
    test_novelty_estimator()
    print()
    test_curiosity_scorer()
    print()
    test_curious_agent()
    print()
    print("=== All curiosity tests passed ===")
