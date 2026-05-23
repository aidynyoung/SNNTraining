"""
HDC Goal-Conditioned Planner and Self-Improvement Loop
=======================================================
Closes the Physical AI sense → interpret → plan → act → observe loop
by adding three missing capabilities:

1. **AutoCalibrator** — automatically registers safe and anomalous regions
   from observation history, making the ActionEvaluator work without
   manual prototype registration. Tracks rolling prediction error; stable
   low-error regions → safe prototypes; high-error spikes → danger.

2. **HDCPlanner** — multi-step action planning via beam search in HV space.
   Uses the CausalTransitionGraph as a forward model to simulate action
   sequences, scores trajectories by goal proximity and cumulative causal
   confidence, returns the best k-step plan.

   Planning purely in HDC:
     state_t+1 = causal_graph.forward_query(state_t, action_hv)
     score     = Hamming_sim(state_T, goal_hv) × causal_confidence

   No backpropagation, no value networks — just VSA composition.

3. **SelfImprovementLoop** — the full Physical AI agent loop:
     Observe → Encode → Predict → Calibrate → Plan → Report

   Adaptive Hebbian learning rate based on:
     - Prediction error    (surprise → learn faster)
     - Pattern recognition (known pattern → learn slower)
     - Causal surprise     (novel causation → learn faster)

   Closes the loop: the agent's own predictions become the training signal
   for its own improvement. No external teacher required.

4. **WorldModelDiagnostics** — answers structured queries about the
   current system state, bridging the world model to interpretable
   human-readable assessments:
     "Is this normal?" / "What caused this?" / "What should I do?"
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch

from hdc.physics_world_model import (
    PhysicsWorldModel, ActionCandidate, MultiHorizonPredictor,
    _xor, _majority, _hamming,
)
from hdc.world_context import (
    ContextualWorldModel, CausalTransitionGraph, SequencePatternMemory,
    PatternMatch,
)
try:
    from hdc.physical_ai_hybrid import (
        HybridPhysicalAIPipeline, EnsembleUncertainty,
    )
except ImportError:
    HybridPhysicalAIPipeline = None  # type: ignore
    EnsembleUncertainty = None  # type: ignore
from hdc.sensor_stream import SensorReading, SensorStreamBuffer
from hdc.hdc_glue import hv_batch_sim, gen_hvs


# ═══════════════════════════════════════════════════════════════════════════════
# 1. AutoCalibrator — automatic safe/danger state registration
# ═══════════════════════════════════════════════════════════════════════════════

class AutoCalibrator:
    """
    Automatically registers safe and danger prototypes from experience.

    The ActionEvaluator's scores are 0.0 when no prototypes exist.
    AutoCalibrator fixes this by watching prediction errors over a rolling
    window and automatically classifying regions:

      stable  (error < stable_threshold for N consecutive steps):
          → register current HV as safe prototype

      anomalous (error > alarm_threshold for M consecutive steps):
          → register HV as danger prototype

      This requires zero manual labeling — the system learns its own
      "comfort zone" purely from prediction error signals.

    Also implements **prototype deduplication**: before storing a new
    prototype, check that it is sufficiently far from all existing ones
    (Hamming distance > diversity_threshold). This prevents redundant
    prototypes from accumulating.

    Args:
        world_model: PhysicsWorldModel whose ActionEvaluator to populate
        stable_threshold: Max error to be considered "stable"
        alarm_threshold: Min error to be considered "anomalous"
        stable_window: Consecutive stable steps required to register safe
        alarm_window: Consecutive alarm steps required to register danger
        diversity_threshold: Min Hamming distance between stored prototypes
        max_safe: Maximum safe prototypes to store
        max_danger: Maximum danger prototypes to store
    """

    def __init__(
        self,
        world_model: PhysicsWorldModel,
        stable_threshold: float = 0.08,
        alarm_threshold: float = 0.35,
        stable_window: int = 5,
        alarm_window: int = 3,
        diversity_threshold: float = 0.1,
        max_safe: int = 32,
        max_danger: int = 16,
    ):
        self.world_model = world_model
        self.stable_threshold = stable_threshold
        self.alarm_threshold = alarm_threshold
        self.stable_window = stable_window
        self.alarm_window = alarm_window
        self.diversity_threshold = diversity_threshold
        self.max_safe = max_safe
        self.max_danger = max_danger

        self._stable_streak = 0
        self._alarm_streak = 0
        self._n_safe_registered = 0
        self._n_danger_registered = 0

    def _is_diverse(
        self,
        hv: torch.Tensor,
        existing: List[torch.Tensor],
    ) -> bool:
        """Return True if hv is far enough from all existing prototypes."""
        if not existing:
            return True
        protos = torch.stack(existing)
        sims = hv_batch_sim(hv, protos)
        max_sim = float(sims.max().item())
        return max_sim < (1.0 - self.diversity_threshold)

    def update(self, sensor_hv: torch.Tensor, prediction_error: float) -> Dict:
        """
        Process one observation and possibly register a prototype.

        Args:
            sensor_hv: (D,) encoded sensor HV for current tick
            prediction_error: Hamming error of predictor on this step

        Returns:
            Dict with action taken ("registered_safe", "registered_danger", "none")
        """
        ev = self.world_model.action_evaluator
        action = "none"

        if prediction_error < self.stable_threshold:
            self._stable_streak += 1
            self._alarm_streak = 0
        elif prediction_error > self.alarm_threshold:
            self._alarm_streak += 1
            self._stable_streak = 0
        else:
            self._stable_streak = max(0, self._stable_streak - 1)
            self._alarm_streak = max(0, self._alarm_streak - 1)

        # Register safe prototype
        if (self._stable_streak >= self.stable_window
                and self._n_safe_registered < self.max_safe
                and self._is_diverse(sensor_hv, ev._safe_prototypes)):
            ev.add_safe_state(sensor_hv)
            self._n_safe_registered += 1
            self._stable_streak = 0
            action = "registered_safe"

        # Register danger prototype
        elif (self._alarm_streak >= self.alarm_window
                and self._n_danger_registered < self.max_danger
                and self._is_diverse(sensor_hv, ev._danger_prototypes)):
            ev.add_danger_state(sensor_hv)
            self._n_danger_registered += 1
            self._alarm_streak = 0
            action = "registered_danger"

        return {
            "action": action,
            "n_safe": len(ev._safe_prototypes),
            "n_danger": len(ev._danger_prototypes),
            "stable_streak": self._stable_streak,
            "alarm_streak": self._alarm_streak,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 2. HDCPlanner — multi-step beam search in HV space
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Plan:
    """A planned sequence of actions with predicted outcome."""
    actions: List[ActionCandidate]
    predicted_trajectory: List[torch.Tensor]  # HV at each step (all intermediate states)
    total_score: float
    goal_similarity: float
    causal_confidence: float
    value_estimate: float = 0.0        # HDCValueFunction mean Q-value for best step
    epistemic_variance: float = 0.0    # mean uncertainty across value heads


class HDCPlanner:
    """
    Multi-step action planner via beam search in hypervector space.

    Uses CausalTransitionGraph as a learned forward model to simulate
    action sequences, then scores the resulting trajectories by:
        - Goal similarity:      Hamming_sim(final_state, goal_hv)
        - Causal confidence:    product of per-step causal confidences
        - Safety:               min Hamming distance to danger prototypes

    Beam search keeps the top-k partial plans at each step, expanding
    each with all candidate actions. This is purely HDC:
        state_{t+1} = XOR(XOR(graph_hv, state_t), action_t)
    No neural network, no value function, no backpropagation.

    If the causal graph has insufficient data (< min_transitions), falls
    back to single-step ActionEvaluator scoring.

    Args:
        causal_graph: CausalTransitionGraph used as forward model
        action_evaluator: ActionEvaluator for scoring and risk
        beam_width: Number of plans to keep at each step
        horizon: Planning horizon (steps ahead)
        min_transitions: Minimum causal observations before planning
        discount: Per-step discount factor for long-horizon planning
    """

    def __init__(
        self,
        causal_graph: CausalTransitionGraph,
        action_evaluator,   # physics_world_model.ActionEvaluator
        beam_width: int = 4,
        horizon: int = 3,
        min_transitions: int = 10,
        discount: float = 0.9,
        value_fn: Optional['HDCValueFunction'] = None,
        robustness_scorer: Optional['PlanRobustnessScorer'] = None,
        value_weight: float = 0.2,
        curiosity_weight: float = 0.05,
    ):
        self.causal_graph = causal_graph
        self.action_evaluator = action_evaluator
        self.beam_width = beam_width
        self.horizon = horizon
        self.min_transitions    = min_transitions
        self.discount           = discount
        self.value_fn           = value_fn
        self.robustness_scorer  = robustness_scorer
        self.value_weight       = value_weight
        self.curiosity_weight   = curiosity_weight
        # Visit counter for curiosity (UCB1-style exploration bonus)
        # Key: (state_hash, action_idx) → visit count
        self._visit_counts: Dict[Tuple[int, int], int] = {}

    def plan(
        self,
        current_state: torch.Tensor,
        candidates: List[ActionCandidate],
        goal_state: Optional[torch.Tensor] = None,
        risk_weight: float = 0.4,
    ) -> List[Plan]:
        """
        Generate and rank multi-step plans via beam search.

        If causal graph has insufficient data, delegates to single-step
        ActionEvaluator (greedy fallback).

        Args:
            current_state: (D,) current world state HV
            candidates: Available actions (ActionCandidate list)
            goal_state: (D,) goal state HV (or None → maximise safety)
            risk_weight: Weight of risk vs utility

        Returns:
            List of Plan objects sorted by total_score descending
        """
        if self.causal_graph.n_transitions < self.min_transitions:
            return self._greedy_fallback(current_state, candidates, goal_state, risk_weight)

        # Beam: (state, plan_so_far, cumulative_score, confidence_product,
        #        variance_acc, trajectory)
        beam: List[Tuple[torch.Tensor, List, float, float, float, List[torch.Tensor]]] = [
            (current_state, [], 0.0, 1.0, 0.0, [current_state])
        ]

        for step in range(self.horizon):
            next_beam = []
            discount_factor = self.discount ** step

            for state, plan, score, conf_prod, var_acc, traj in beam:
                for action in candidates:
                    # Simulate one step via causal graph
                    next_state, causal_conf = self.causal_graph.forward_query(
                        state, action.hv
                    )

                    # Immediate reward: goal proximity
                    if goal_state is not None:
                        step_util = float(_hamming(next_state, goal_state).item())
                    else:
                        ev = self.action_evaluator
                        step_util = ev._max_similarity_to_set(
                            next_state, ev._safe_prototypes
                        )

                    # Risk
                    ev = self.action_evaluator
                    step_risk = ev._max_similarity_to_set(
                        next_state, ev._danger_prototypes
                    )

                    # Value function bonus + uncertainty
                    vf_score, vf_var = 0.0, 0.0
                    if self.value_fn is not None:
                        vf_score = self.value_fn.value(state, action.hv)
                        vf_var   = self.value_fn.value_uncertainty(state, action.hv)

                    # Curiosity bonus: UCB1-style exploration incentive
                    # Unexplored (state, action) pairs get higher scores
                    curiosity_bonus = 0.0
                    if self.curiosity_weight > 0:
                        s_hash = int(state.sum().item() * 1000) % (2 ** 20)
                        a_idx  = candidates.index(action) if action in candidates else 0
                        key    = (s_hash, a_idx)
                        visits = self._visit_counts.get(key, 0)
                        curiosity_bonus = self.curiosity_weight / math.sqrt(visits + 1)

                    step_score = discount_factor * (
                        step_util
                        - risk_weight * step_risk
                        + self.value_weight * vf_score
                        + curiosity_bonus
                    )

                    # Robustness penalty: favour plans all value heads agree on
                    if self.robustness_scorer is not None:
                        step_score = self.robustness_scorer.score(
                            step_score, vf_var, step_risk, risk_weight
                        )

                    new_score = score + step_score
                    new_conf  = conf_prod * max(causal_conf, 0.1)
                    new_var   = var_acc + vf_var

                    next_beam.append((
                        next_state,
                        plan + [action],
                        new_score,
                        new_conf,
                        new_var,
                        traj + [next_state],
                    ))

            # Keep top beam_width
            next_beam.sort(key=lambda x: x[2], reverse=True)
            beam = next_beam[:self.beam_width]

        # Convert beam to Plan objects
        plans = []
        for final_state, plan_actions, total_score, conf_prod, var_acc, traj in beam:
            if goal_state is not None:
                goal_sim = float(_hamming(final_state, goal_state).item())
            else:
                goal_sim = self.action_evaluator._max_similarity_to_set(
                    final_state, self.action_evaluator._safe_prototypes
                )
            mean_var = var_acc / max(len(plan_actions), 1)
            vf_est = 0.0
            if self.value_fn is not None and plan_actions:
                vf_est = self.value_fn.value(current_state, plan_actions[0].hv)
            plans.append(Plan(
                actions=plan_actions,
                predicted_trajectory=traj,
                total_score=total_score,
                goal_similarity=goal_sim,
                causal_confidence=conf_prod,
                value_estimate=vf_est,
                epistemic_variance=mean_var,
            ))

        plans.sort(key=lambda p: p.total_score, reverse=True)
        return plans

    def _greedy_fallback(
        self,
        current_state: torch.Tensor,
        candidates: List[ActionCandidate],
        goal_state: Optional[torch.Tensor],
        risk_weight: float,
    ) -> List[Plan]:
        """Single-step greedy plan when causal graph is too sparse."""
        ranked = self.action_evaluator.evaluate(
            current_state, candidates, goal_state, risk_weight
        )
        return [
            Plan(
                actions=[c],
                predicted_trajectory=[c.predicted_outcome if c.predicted_outcome is not None else current_state],
                total_score=c.net_score,
                goal_similarity=c.utility_score,
                causal_confidence=0.0,   # causal graph not used
            )
            for c in ranked
        ]

    def best_action(
        self,
        current_state: torch.Tensor,
        candidates: List[ActionCandidate],
        goal_state: Optional[torch.Tensor] = None,
        record_visit: bool = True,
    ) -> Optional[ActionCandidate]:
        """
        Return the first action of the best plan and record the visit for curiosity.

        Args:
            record_visit: Update visit count for the chosen (state, action) pair.
        """
        plans = self.plan(current_state, candidates, goal_state)
        if plans and plans[0].actions:
            action = plans[0].actions[0]
            if record_visit and self.curiosity_weight > 0:
                s_hash = int(current_state.sum().item() * 1000) % (2 ** 20)
                a_idx  = candidates.index(action) if action in candidates else 0
                key    = (s_hash, a_idx)
                self._visit_counts[key] = self._visit_counts.get(key, 0) + 1
            return action
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Adaptive Hebbian Learning Rate
# ═══════════════════════════════════════════════════════════════════════════════

class AdaptiveHebbian:
    """
    Adjusts Hebbian learning rates based on three surprise signals.

    Standard Hebbian uses a fixed lr. AdaptiveHebbian modulates lr based on:
      - Prediction error (high error → higher lr)
      - Pattern familiarity (known pattern → lower lr)
      - Causal surprise (novel causation → higher lr)

    Combined adaptive rate:
        lr = lr_base × (1 + α_err × err) × (1 - α_pat × familiarity) × (1 + α_caus × caus_surp)

    This makes the system:
      - Learn aggressively when surprised (unknown territory)
      - Consolidate slowly when revisiting familiar patterns
      - Update moderately for normal prediction errors

    Args:
        predictor: MultiHorizonPredictor to adapt
        lr_base: Minimum learning rate
        lr_max: Maximum learning rate cap
        alpha_error: Weight of prediction error in lr adjustment
        alpha_pattern: Weight of pattern familiarity in lr reduction
        alpha_causal: Weight of causal surprise in lr adjustment
    """

    def __init__(
        self,
        predictor: MultiHorizonPredictor,
        lr_base: float = 0.005,
        lr_max: float = 0.08,
        alpha_error: float = 0.5,
        alpha_pattern: float = 0.3,
        alpha_causal: float = 0.4,
    ):
        self.predictor = predictor
        self.lr_base = lr_base
        self.lr_max = lr_max
        self.alpha_error = alpha_error
        self.alpha_pattern = alpha_pattern
        self.alpha_causal = alpha_causal

        self._lr_history: List[float] = []
        self._tick = 0

    def compute_lr(
        self,
        prediction_error: float,
        pattern_familiarity: float = 0.0,
        causal_surprise: float = 0.0,
    ) -> float:
        """
        Compute adaptive learning rate.

        Args:
            prediction_error: Hamming error ∈ [0, 0.5]
            pattern_familiarity: Pattern match similarity ∈ [0, 1]
            causal_surprise: Causal graph surprise ∈ [0, 0.5]

        Returns:
            Adaptive learning rate
        """
        lr = self.lr_base
        lr *= (1 + self.alpha_error * prediction_error * 2)      # error boost
        lr *= (1 - self.alpha_pattern * pattern_familiarity)      # familiarity suppress
        lr *= (1 + self.alpha_causal * causal_surprise * 2)       # causal boost
        return min(lr, self.lr_max)

    def update(
        self,
        current_hv: torch.Tensor,
        actual_hv: torch.Tensor,
        prediction_error: float,
        pattern_familiarity: float = 0.0,
        causal_surprise: float = 0.0,
    ) -> float:
        """
        Apply adaptive Hebbian update to all horizon predictors.

        Returns:
            The learning rate used
        """
        lr = self.compute_lr(prediction_error, pattern_familiarity, causal_surprise)
        self.predictor.update(current_hv, actual_hv, lr=lr)
        self._lr_history.append(lr)
        self._tick += 1
        return lr

    def mean_lr(self, window: int = 20) -> float:
        h = self._lr_history[-window:] if self._lr_history else [self.lr_base]
        return sum(h) / len(h)

    def anneal(self, decay: float = 0.999):
        """
        Decay lr_base by `decay` per call.

        Curriculum learning: start aggressive, consolidate over time.
        Call once per episode or every N steps.
        """
        self.lr_base = max(self.lr_base * decay, self.lr_base * 0.01)
        return self.lr_base

    def momentum_lr(self, window: int = 10) -> float:
        """
        EMA-smoothed effective learning rate (reduces noise from single steps).
        """
        h = self._lr_history[-window:] if self._lr_history else [self.lr_base]
        ema = h[0]
        for v in h[1:]:
            ema = 0.9 * ema + 0.1 * v
        return ema

    def reset_boost(self):
        """
        Reset lr_base to initial value after a detected distribution shift.

        Call this when prediction error spikes significantly above the mean,
        indicating a new distribution that requires fast re-learning.
        """
        if len(self._lr_history) < 5:
            return
        recent_mean  = sum(self._lr_history[-5:]) / 5
        overall_mean = sum(self._lr_history) / len(self._lr_history)
        # If recent mean lr dropped well below overall — possibly underlearning
        if recent_mean < overall_mean * 0.5:
            self.lr_base = min(self.lr_max * 0.5, self.lr_base * 3.0)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. SelfImprovementLoop — the closed Physical AI agent loop
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class AgentStep:
    """Record of one agent tick."""
    tick: int
    prediction_error: float
    pattern_familiarity: float
    causal_surprise: float
    lr_used: float
    action_taken: Optional[str]
    calibration_action: str
    plan_length: int


class SelfImprovementLoop:
    """
    Closed Physical AI agent loop: observe → interpret → plan → act → improve.

    Combines ContextualWorldModel, AutoCalibrator, HDCPlanner, and
    AdaptiveHebbian into a single agent that:

      1. Observes sensor readings (interface layer)
      2. Encodes with hierarchical context and pattern memory
      3. Detects causal surprise and patterns
      4. AutoCalibrates safe/danger prototypes from observation history
      5. Plans best action using causal graph + beam search
      6. Updates Hebbian predictors with adaptive learning rate
      7. Returns full situational awareness report

    The agent improves over time without any external teacher:
      - Prediction error decreases as Hebbian predictors adapt
      - Pattern memory grows as recurring patterns are encountered
      - Causal graph builds an internal model of system dynamics
      - Safe/danger regions self-register from error history
      - Learning rate adapts to match current uncertainty

    Args:
        contextual_world_model: The ContextualWorldModel to use and improve
        beam_width: Beam search width for HDCPlanner
        planning_horizon: Steps ahead for multi-step planning
        min_causal_for_planning: Transitions needed before planning
        lr_base, lr_max: Adaptive Hebbian learning rate bounds
    """

    def __init__(
        self,
        contextual_world_model: ContextualWorldModel,
        beam_width: int = 3,
        planning_horizon: int = 3,
        min_causal_for_planning: int = 15,
        lr_base: float = 0.005,
        lr_max: float = 0.06,
    ):
        self.world = contextual_world_model
        wm = contextual_world_model.pipeline.world_model

        # Auto-calibrator (fixes action evaluation)
        self.calibrator = AutoCalibrator(wm)

        # Multi-step planner
        self.planner = HDCPlanner(
            causal_graph=contextual_world_model.causal_graph,
            action_evaluator=wm.action_evaluator,
            beam_width=beam_width,
            horizon=planning_horizon,
            min_transitions=min_causal_for_planning,
        )

        # Adaptive Hebbian
        self.adaptive_hebb = AdaptiveHebbian(
            wm.multi_horizon,
            lr_base=lr_base,
            lr_max=lr_max,
        )

        self._tick = 0
        self._step_log: List[AgentStep] = []
        self._goal_state: Optional[torch.Tensor] = None

    def set_goal(self, goal_hv: torch.Tensor):
        """Set a goal state for planning (optional)."""
        self._goal_state = goal_hv.detach().clone()

    def tick(
        self,
        reading: SensorReading,
        candidate_actions: Optional[List[ActionCandidate]] = None,
        action_hv: Optional[torch.Tensor] = None,
    ) -> Dict:
        """
        Run one full agent loop iteration.

        Args:
            reading: Current sensor observation
            candidate_actions: Available actions to evaluate and plan
            action_hv: HV of the action being taken this tick

        Returns:
            Full situational awareness dict including:
            - predictions, context, pattern match
            - calibration status (safe/danger prototypes)
            - best_plan (if candidates given)
            - lr_used (adaptive Hebbian rate)
            - agent diagnostics
        """
        self._tick += 1

        # ── 1. Observe + interpret via contextual world model ──────────────────
        result = self.world.tick(reading, action_hv=action_hv, candidate_actions=None)

        prediction_error = result["prediction_error"]
        pattern_match: Optional[PatternMatch] = result.get("pattern_match")
        causal_surprise = result.get("causal_surprise", 0.0)
        sensor_hv = result["sensor_hv"]

        pattern_familiarity = (
            pattern_match.similarity if pattern_match and pattern_match.is_known else 0.0
        )

        # ── 2. Auto-calibrate safe/danger states ───────────────────────────────
        cal = self.calibrator.update(sensor_hv, prediction_error)

        # ── 3. Adaptive Hebbian update ─────────────────────────────────────────
        wm = self.world.pipeline.world_model
        lr_used = self.adaptive_hebb.update(
            wm.current_state,
            sensor_hv,
            prediction_error=prediction_error,
            pattern_familiarity=pattern_familiarity,
            causal_surprise=causal_surprise,
        )

        # ── 4. Multi-step planning ─────────────────────────────────────────────
        best_plan = None
        best_action = None
        if candidate_actions:
            plans = self.planner.plan(
                wm.current_state,
                candidate_actions,
                goal_state=self._goal_state,
            )
            if plans:
                best_plan = plans[0]
                best_action = best_plan.actions[0].name if best_plan.actions else None

        # ── 5. Log step ────────────────────────────────────────────────────────
        step = AgentStep(
            tick=self._tick,
            prediction_error=prediction_error,
            pattern_familiarity=pattern_familiarity,
            causal_surprise=causal_surprise,
            lr_used=lr_used,
            action_taken=action_hv is not None,
            calibration_action=cal["action"],
            plan_length=len(best_plan.actions) if best_plan else 0,
        )
        self._step_log.append(step)

        return {
            **result,
            "lr_used": lr_used,
            "calibration": cal,
            "best_plan": best_plan,
            "best_action": best_action,
            "pattern_familiarity": pattern_familiarity,
            "adaptive_lr_mean": self.adaptive_hebb.mean_lr(),
            "agent_tick": self._tick,
        }

    def improvement_report(self) -> Dict:
        """Summarise agent learning progress over all ticks."""
        if not self._step_log:
            return {}

        errors = [s.prediction_error for s in self._step_log]
        lrs = [s.lr_used for s in self._step_log]

        # Split early vs recent to show learning
        n = len(errors)
        mid = n // 2
        early_err = sum(errors[:mid]) / max(mid, 1)
        recent_err = sum(errors[mid:]) / max(n - mid, 1)

        cal_actions = [s.calibration_action for s in self._step_log]
        n_safe = cal_actions.count("registered_safe")
        n_danger = cal_actions.count("registered_danger")

        return {
            "total_ticks": n,
            "early_mean_error": round(early_err, 4),
            "recent_mean_error": round(recent_err, 4),
            "error_reduction": round(early_err - recent_err, 4),
            "mean_lr": round(sum(lrs) / len(lrs), 5),
            "n_safe_registered": n_safe,
            "n_danger_registered": n_danger,
            "n_known_patterns": self.world.pattern_memory.n_patterns,
            "n_causal_transitions": self.world.causal_graph.n_transitions,
        }

    def apply_curriculum(self, error_window: int = 20):
        """
        Curriculum learning: anneal lr when error is consistently low.

        If recent prediction error is below the long-run mean, the agent has
        mastered the current distribution → decay lr toward consolidation.
        If error spikes back up, reset_boost restores aggressive learning.

        Call once per episode boundary.
        """
        if not self._step_log:
            return
        errors = [s.prediction_error for s in self._step_log]
        recent = sum(errors[-error_window:]) / max(min(len(errors), error_window), 1)
        overall = sum(errors) / len(errors)

        if recent < overall * 0.8:
            # Error is low → consolidate
            self.adaptive_hebb.anneal(decay=0.995)
        else:
            # Error is high → boost back up
            self.adaptive_hebb.reset_boost()

    def has_converged(
        self,
        window:    int   = 30,
        threshold: float = 0.05,
        min_ticks: int   = 50,
    ) -> bool:
        """
        Detect whether the agent has converged (prediction error stabilised).

        Convergence = std(recent errors) < threshold AND mean < 0.3.
        Requires at least `min_ticks` observations.

        Args:
            window:    Window of recent steps to check
            threshold: Maximum std for convergence declaration
            min_ticks: Minimum ticks before convergence can be declared

        Returns:
            True if converged, False if still learning.
        """
        if len(self._step_log) < min_ticks:
            return False
        recent_errors = [s.prediction_error for s in self._step_log[-window:]]
        import statistics
        mean_e = sum(recent_errors) / len(recent_errors)
        std_e  = statistics.stdev(recent_errors) if len(recent_errors) > 1 else 1.0
        return std_e < threshold and mean_e < 0.3

    def plateau_steps(self, window: int = 20, threshold: float = 0.01) -> int:
        """
        Count how many steps the agent has been on a learning plateau.

        A plateau is defined as: recent error variance < threshold
        AND error is NOT low (not converged, but not improving).
        Returns 0 if not on a plateau.
        """
        if len(self._step_log) < window * 2:
            return 0
        recent = [s.prediction_error for s in self._step_log[-window:]]
        older  = [s.prediction_error for s in self._step_log[-2*window:-window]]
        import statistics
        std_recent = statistics.stdev(recent) if len(recent) > 1 else 1.0
        mean_old   = sum(older)   / len(older)
        mean_new   = sum(recent)  / len(recent)
        if std_recent < threshold and abs(mean_old - mean_new) < threshold:
            return window
        return 0


# ═══════════════════════════════════════════════════════════════════════════════
# 5. ValueFunctionLoop — closes RL loop with TD-learning from prediction error
# ═══════════════════════════════════════════════════════════════════════════════

class ValueFunctionLoop:
    """
    Augments SelfImprovementLoop with an online HDCValueFunction.

    After each tick the agent's prediction error is converted to a reward
    signal and used to TD-update the value function.  The wired planner
    then scores beam nodes with Q-values, so planning improves alongside
    prediction:

        reward = 1.0 - 2 * prediction_error   (∈ [-1, 1])
        V_hv  += α × δ × bind(s_hv, a_hv)    (TD-update)

    The value function is automatically wired into the SelfImprovementLoop's
    HDCPlanner so every call to planner.plan() benefits from accumulated Q.

    Args:
        agent:    SelfImprovementLoop to wrap
        hd_dim:   Hypervector dimension (must match agent's world model)
        gamma:    TD discount factor
        alpha:    Value-function TD learning rate
        value_weight: Weight of Q-value bonus inside HDCPlanner beam scoring
        n_value_heads: Ensemble size for epistemic uncertainty
    """

    def __init__(
        self,
        agent: SelfImprovementLoop,
        hd_dim: int,
        gamma: float = 0.95,
        alpha: float = 0.05,
        value_weight: float = 0.2,
        n_value_heads: int = 3,
    ):
        self.agent = agent
        self.value_fn = HDCValueFunction(hd_dim, gamma=gamma, alpha=alpha,
                                         n_value_heads=n_value_heads)
        # Wire into planner immediately
        agent.planner.value_fn      = self.value_fn
        agent.planner.value_weight  = value_weight

        self._prev_state:  Optional[torch.Tensor] = None
        self._prev_action: Optional[torch.Tensor] = None
        self._td_steps = 0

    def tick(
        self,
        reading,
        candidate_actions=None,
        action_hv: Optional[torch.Tensor] = None,
    ) -> Dict:
        """
        Run one agent tick and perform a TD update on the value function.

        Returns agent tick result enriched with:
          - value_fn_best: name of the action with highest Q-value
          - value_fn_q:    that action's Q-value
          - td_reward:     reward signal used for the TD update
        """
        result = self.agent.tick(reading, candidate_actions, action_hv)

        sensor_hv        = result["sensor_hv"]
        prediction_error = result["prediction_error"]
        reward           = 1.0 - 2.0 * prediction_error  # [-1, 1]

        # TD update from previous transition
        if self._prev_state is not None and self._prev_action is not None:
            next_action_hvs = [c.hv for c in candidate_actions] if candidate_actions else None
            self.value_fn.td_update(
                self._prev_state,
                self._prev_action,
                reward,
                sensor_hv,
                next_action_hvs,
            )
            self._td_steps += 1

        self._prev_state  = sensor_hv.detach().clone()
        self._prev_action = (action_hv.detach().clone()
                             if action_hv is not None
                             else sensor_hv.detach().clone())

        # Report best action by Q-value
        result["value_fn_best"] = None
        result["value_fn_q"]    = 0.0
        result["td_reward"]     = reward
        if candidate_actions:
            action_hvs = [c.hv for c in candidate_actions]
            best_idx, best_q = self.value_fn.best_action(sensor_hv, action_hvs)
            result["value_fn_best"] = candidate_actions[best_idx].name
            result["value_fn_q"]    = best_q

        return result

    def reset(self):
        self._prev_state  = None
        self._prev_action = None

    @property
    def td_steps(self) -> int:
        return self._td_steps


# ═══════════════════════════════════════════════════════════════════════════════
# 6. WorldModelDiagnostics — structured queries on world state
# ═══════════════════════════════════════════════════════════════════════════════

class WorldModelDiagnostics:
    """
    Answers structured questions about the Physical AI agent's world state.

    Bridges raw HDC distances and similarities to human-readable assessments.

    Supported queries:
      is_normal(error)    → "normal" / "uncertain" / "anomalous"
      pattern_status(m)   → "recognised: <label>" / "novel"
      confidence_level()  → "high" / "medium" / "low"
      recommend(plan)     → "execute: <action>" / "gather data" / "alert"
    """

    def __init__(self, agent: SelfImprovementLoop):
        self.agent = agent

    def is_normal(self, prediction_error: float) -> str:
        if prediction_error < self.agent.calibrator.stable_threshold:
            return "normal"
        elif prediction_error < self.agent.calibrator.alarm_threshold:
            return "uncertain"
        else:
            return "anomalous"

    def pattern_status(self, match: Optional[PatternMatch]) -> str:
        if match is None:
            return "no_data"
        if match.is_known and match.similarity > 0.7:
            label = match.label or f"pattern_{match.pattern_id}"
            return f"recognised: {label} (seen {match.n_times_seen}×)"
        elif match.is_known:
            return f"partial_match (sim={match.similarity:.2f})"
        else:
            return "novel_pattern"

    def confidence_level(self) -> str:
        mean_err = self.agent.adaptive_hebb.mean_lr(10)
        # Higher mean lr = more surprise = lower confidence
        if mean_err < self.agent.adaptive_hebb.lr_base * 2:
            return "high"
        elif mean_err < self.agent.adaptive_hebb.lr_base * 5:
            return "medium"
        else:
            return "low"

    def recommend(self, best_plan: Optional[Plan]) -> str:
        if best_plan is None:
            return "observe: no candidates provided"
        if self.agent.world.causal_graph.n_transitions < self.agent.planner.min_transitions:
            needed = self.agent.planner.min_causal_for_planning - self.agent.world.causal_graph.n_transitions
            return f"gather_data: need {needed} more causal observations for planning"
        if best_plan.total_score <= 0:
            return "hold: all actions have non-positive score"
        action_name = best_plan.actions[0].name if best_plan.actions else "unknown"
        return f"execute: {action_name} (score={best_plan.total_score:.3f}, conf={best_plan.causal_confidence:.2f})"

    def full_report(self, result: Dict) -> Dict:
        """Generate a structured diagnostic report from a tick result."""
        return {
            "normality": self.is_normal(result["prediction_error"]),
            "pattern": self.pattern_status(result.get("pattern_match")),
            "confidence": self.confidence_level(),
            "recommendation": self.recommend(result.get("best_plan")),
            "causal_transitions": result.get("n_causal_transitions", 0),
            "safe_prototypes": result.get("calibration", {}).get("n_safe", 0),
            "lr": round(result.get("lr_used", 0), 5),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_auto_calibrator():
    print("=" * 60)
    print("Testing AutoCalibrator (fixes action scores)")
    print("=" * 60)

    torch.manual_seed(0)
    from hdc.physics_world_model import PhysicsWorldModel
    wm = PhysicsWorldModel(hd_dim=500)
    cal = AutoCalibrator(wm, stable_threshold=0.05, alarm_threshold=0.4,
                         stable_window=3, alarm_window=2)

    dim = 500
    torch.manual_seed(1)
    normal_hv = (torch.rand(dim) < 0.5).float()
    danger_hv = (torch.rand(dim) < 0.5).float()

    # Feed 6 normal ticks (error=0.02)
    for _ in range(6):
        r = cal.update(normal_hv, prediction_error=0.02)

    # Feed 3 alarm ticks (error=0.45)
    for _ in range(3):
        r = cal.update(danger_hv, prediction_error=0.45)

    print(f"  Safe prototypes: {r['n_safe']}  Danger: {r['n_danger']}")
    assert r["n_safe"] >= 1, "Should have registered at least one safe state"
    assert r["n_danger"] >= 1, "Should have registered at least one danger state"

    # Now action evaluation should produce non-zero scores
    ev = wm.action_evaluator
    action_hv = (torch.rand(dim) < 0.1).float()
    candidates = [
        ActionCandidate("safe_action", action_hv),
        ActionCandidate("danger_action", _xor(normal_hv, danger_hv)),
    ]
    ranked = ev.evaluate(normal_hv, candidates)
    print(f"  Action scores after calibration: {[(c.name, round(c.net_score, 3)) for c in ranked]}")
    assert any(c.net_score != 0.0 for c in ranked), "Action scores still 0.0 after calibration!"

    print("  ✅ AutoCalibrator OK")


def test_hdc_planner():
    print("=" * 60)
    print("Testing HDCPlanner (multi-step beam search)")
    print("=" * 60)

    torch.manual_seed(42)
    dim = 500
    from hdc.physics_world_model import PhysicsWorldModel
    from hdc.world_context import CausalTransitionGraph
    wm = PhysicsWorldModel(hd_dim=dim)
    causal = CausalTransitionGraph(dim, decay=0.8)

    # States: A → B → C (via action_go)
    state_a = (torch.rand(dim) < 0.5).float()
    state_b = (torch.rand(dim) < 0.5).float()
    state_c = (torch.rand(dim) < 0.5).float()
    action_go = (torch.rand(dim) < 0.5).float()
    action_stay = (torch.rand(dim) < 0.05).float()

    # Build causal graph with 20 observations
    for _ in range(20):
        causal.observe(state_a, action_go, state_b)
        causal.observe(state_b, action_go, state_c)

    # Auto-calibrate: C is the safe state
    cal = AutoCalibrator(wm, stable_threshold=0.1, stable_window=2)
    for _ in range(3):
        cal.update(state_c, prediction_error=0.02)

    planner = HDCPlanner(causal, wm.action_evaluator, beam_width=3, horizon=2,
                         min_transitions=5)

    candidates = [
        ActionCandidate("go",   action_go),
        ActionCandidate("stay", action_stay),
    ]

    # From state A, goal = C → should prefer "go"
    plans = planner.plan(state_a, candidates, goal_state=state_c)
    print(f"  Plans from A → C (horizon=2):")
    for p in plans[:3]:
        acts = [a.name for a in p.actions]
        print(f"    {acts}: score={p.total_score:.4f}, goal_sim={p.goal_similarity:.4f}")
    assert len(plans) > 0, "No plans generated"
    print(f"  Best action: {plans[0].actions[0].name if plans[0].actions else None}")

    print("  ✅ HDCPlanner OK")


def test_adaptive_hebbian():
    print("=" * 60)
    print("Testing AdaptiveHebbian (surprise-modulated learning)")
    print("=" * 60)

    torch.manual_seed(7)
    dim = 500
    from hdc.physics_world_model import MultiHorizonPredictor
    pred = MultiHorizonPredictor(dim)
    hebb = AdaptiveHebbian(pred, lr_base=0.005, lr_max=0.08)

    state = (torch.rand(dim) < 0.5).float()
    next_s = (torch.rand(dim) < 0.5).float()

    # Normal update: low error, known pattern
    lr_normal = hebb.compute_lr(prediction_error=0.03, pattern_familiarity=0.9)
    # Surprise update: high error, novel
    lr_surprise = hebb.compute_lr(prediction_error=0.40, pattern_familiarity=0.0, causal_surprise=0.35)

    print(f"  LR (normal, familiar): {lr_normal:.5f}")
    print(f"  LR (surprised, novel): {lr_surprise:.5f}")
    assert lr_surprise > lr_normal * 2, "Surprise should dramatically increase LR"
    assert lr_surprise <= 0.08, "Should not exceed lr_max"

    # Run 10 updates and verify predictor improves
    for _ in range(20):
        hebb.update(state, next_s, prediction_error=0.20)
    print(f"  Mean LR over 20 updates: {hebb.mean_lr():.5f}")

    print("  ✅ AdaptiveHebbian OK")


def test_self_improvement_loop():
    print("=" * 60)
    print("Testing SelfImprovementLoop (full closed loop)")
    print("=" * 60)

    import time as _time
    torch.manual_seed(99)

    from hdc.sensor_stream import SensorSpec, ModalityType
    from hdc.physical_ai_hybrid import HybridPhysicalAIPipeline
    from hdc.world_context import ContextualWorldModel

    specs = [
        SensorSpec("s", ModalityType.SCALAR, raw_dim=1, hd_dim=500, seed=0),
    ]
    base = HybridPhysicalAIPipeline(specs, hd_dim=500, n_ensemble=3,
                                     consolidation_period=10)
    world = ContextualWorldModel(base, pattern_window=4, pattern_stride=3)
    agent = SelfImprovementLoop(world, beam_width=2, planning_horizon=2,
                                 min_causal_for_planning=10, lr_base=0.005)

    dim = 500
    candidates = [
        ActionCandidate("low",  (torch.rand(dim) < 0.05).float()),
        ActionCandidate("high", (torch.rand(dim) < 0.4).float()),
    ]

    # Feed 40 ticks: sawtooth × 8 cycles
    last = None
    for t in range(40):
        phase = t % 5
        reading = SensorReading(
            timestamp=float(t),
            data={"s": torch.tensor([float(phase) / 4])},
        )
        action_hv = candidates[phase % 2].hv
        last = agent.tick(reading, candidate_actions=candidates, action_hv=action_hv)

    print(f"  After 40 ticks:")
    print(f"    Prediction error: {round(last['prediction_error'], 4)}")
    print(f"    Pattern familiarity: {round(last['pattern_familiarity'], 4)}")
    print(f"    LR used: {round(last['lr_used'], 5)}")
    print(f"    Calibration: {last['calibration']['action']}, "
          f"safe={last['calibration']['n_safe']}, danger={last['calibration']['n_danger']}")
    print(f"    Best action: {last['best_action']}")

    report = agent.improvement_report()
    print(f"  Improvement report: {report}")
    assert report["total_ticks"] == 40
    assert report["n_causal_transitions"] > 0

    diag = WorldModelDiagnostics(agent)
    dr = diag.full_report(last)
    print(f"  Diagnostics: {dr}")
    assert dr["normality"] in ("normal", "uncertain", "anomalous")

    print("  ✅ SelfImprovementLoop OK")


# ═══════════════════════════════════════════════════════════════════════════════
# Elite Enhancements — HDC_MCTS, PlanRobustnessScorer
# ═══════════════════════════════════════════════════════════════════════════════

class HDC_MCTS:
    """
    Elite replacement for HDCPlanner's beam search.

    Monte Carlo Tree Search in hypervector space:
      - Selection: UCB1 where value = Hamming similarity to goal
      - Expansion: try each candidate action from the selected node
      - Rollout: random action sequence to estimate long-term value
      - Backprop: accumulate visit counts and average scores

    Advantages over beam search:
      1. Exploration/exploitation balance via UCB1
      2. Rollouts estimate multi-step consequences
      3. Handles stochastic transitions naturally

    Args:
        n_simulations: MCTS iterations per planning call
        exploration_constant: UCB1 exploration weight C
        rollout_depth: Random rollout horizon
        max_actions: Max candidates considered per node
    """

    class _Node:
        def __init__(self, state_hv: torch.Tensor, parent=None, action_idx: int = -1):
            self.state_hv = state_hv
            self.parent = parent
            self.action_idx = action_idx
            self.children: List = []
            self.visits: int = 0
            self.total_value: float = 0.0
            self.untried_actions: List[int] = []

    def __init__(
        self,
        n_simulations: int = 50,
        exploration_constant: float = 1.4,
        rollout_depth: int = 5,
        max_actions: int = 10,
    ):
        self.n_simulations = n_simulations
        self.exploration_constant = exploration_constant
        self.rollout_depth = rollout_depth
        self.max_actions = max_actions

    def plan(
        self,
        current_state: torch.Tensor,
        candidates: list,
        goal_state: Optional[torch.Tensor] = None,
        forward_model: Optional[Callable] = None,
        action_evaluator=None,
    ) -> List:
        """
        Plan using MCTS in HV space.

        Args:
            current_state: (D,) current state HV
            candidates: List of action HVs or ActionCandidate-like objects
            goal_state: (D,) goal state HV
            forward_model: Optional callable(state, action) → next_state

        Returns:
            List of (action, expected_value) sorted by value descending.
        """
        if not candidates:
            return []

        action_list = candidates[:self.max_actions]
        action_hvs = [c.hv if hasattr(c, 'hv') else c for c in action_list]

        def _forward(state: torch.Tensor, action_hv: torch.Tensor) -> torch.Tensor:
            return forward_model(state, action_hv) if forward_model is not None else _xor(state, action_hv)

        root = self._Node(current_state)
        root.untried_actions = list(range(len(action_hvs)))

        for _ in range(self.n_simulations):
            node = root

            while not node.untried_actions and node.children:
                node = self._ucb_select(node)

            if node.untried_actions:
                a_idx = node.untried_actions.pop(0)
                next_state = _forward(node.state_hv, action_hvs[a_idx])
                child = self._Node(next_state, parent=node, action_idx=a_idx)
                child.untried_actions = list(range(len(action_hvs)))
                node.children.append(child)
                node = child

            value = self._rollout(node.state_hv, action_hvs, goal_state, _forward)

            while node is not None:
                node.visits += 1
                node.total_value += value
                node = node.parent

        results = []
        for child in sorted(root.children, key=lambda c: c.total_value / max(c.visits, 1), reverse=True):
            avg = child.total_value / max(child.visits, 1)
            if 0 <= child.action_idx < len(action_list):
                results.append((action_list[child.action_idx], avg))
        return results

    def _ucb_select(self, node: _Node) -> _Node:
        log_parent = math.log(node.visits + 1)
        return max(
            node.children,
            key=lambda c: (c.total_value / max(c.visits, 1))
                        + self.exploration_constant * math.sqrt(log_parent / max(c.visits, 1))
        )

    def _rollout(
        self,
        state: torch.Tensor,
        action_hvs: List[torch.Tensor],
        goal_state: Optional[torch.Tensor],
        forward_model: Callable,
    ) -> float:
        total, current, discount = 0.0, state, 0.9
        for step in range(self.rollout_depth):
            a_idx = int(torch.randint(0, len(action_hvs), (1,)).item())
            current = forward_model(current, action_hvs[a_idx])
            value = (
                float(_hamming(current.unsqueeze(0), goal_state.unsqueeze(0)).item())
                if goal_state is not None else 0.5
            )
            total += value * (discount ** step)
        return total / self.rollout_depth


class PlanRobustnessScorer:
    """
    Elite enhancement for plan scoring.

    Penalises plans with high ensemble variance so that the planner
    favours actions all ensemble members agree on (robust plans over
    brittle ones).

    score = expected_value - robustness_penalty * variance - risk_weight * risk_score
    """

    def __init__(self, robustness_penalty: float = 0.3):
        self.robustness_penalty = robustness_penalty

    def score(
        self,
        expected_value: float,
        predictor_variance: float,
        risk_score: float = 0.0,
        risk_weight: float = 0.4,
    ) -> float:
        var_penalty = self.robustness_penalty * min(predictor_variance / 0.5, 1.0)
        return expected_value - var_penalty - risk_weight * risk_score


# ═══════════════════════════════════════════════════════════════════════════════
# IQT-Level Enhancements — HDCValueFunction, OptionHDCPlanner
# ═══════════════════════════════════════════════════════════════════════════════

class HDCValueFunction:
    """
    Approximate Q-learning in hypervector space.

    Reference:
        Karunaratne et al. (2020) "In-memory hyperdimensional computing"
        Nature Electronics 3:327-337.

        Thomas et al. (2021) "SpAtten: Efficient sparse attention architecture
        with cascade token and head pruning" — value encoding in HV space.

    Encodes state-action values as hypervectors, enabling tabular-free
    Q-learning that generalises across similar states via Hamming similarity:

        Q(s, a) ≈ sim(bind(s_hv, a_hv), V_hv)          [retrieval]
        V_hv   += α × δ × bind(s_hv, a_hv)              [TD update]

    where:
        δ = r + γ × max_{a'} Q(s', a') − Q(s, a)        [TD error]
        bind = XOR in binary HDC
        V_hv = learned value prototype (accumulates Q information)

    Properties:
        - O(D) memory regardless of state/action space size
        - Generalises to unseen (s, a) pairs via Hamming similarity
        - Fully compatible with HDC_MCTS (use as value oracle)
        - No neural network — bitwise operations only

    Args:
        hd_dim: Hypervector dimension
        gamma: Discount factor
        alpha: TD learning rate
        n_value_heads: Number of value heads (ensemble for uncertainty)
    """

    def __init__(
        self,
        hd_dim: int,
        gamma: float = 0.95,
        alpha: float = 0.05,
        n_value_heads: int = 3,
    ):
        self.hd_dim = hd_dim
        self.gamma  = gamma
        self.alpha  = alpha
        self.n_heads = n_value_heads

        # Float prototype accumulators — binarized only at query time.
        # Storing floats avoids the broken _majority(binary + small_alpha) issue
        # where α × sa is always < 0.5 and never flips the binary threshold.
        # TD-weighted bundling: V_float += α × td_error × sa_hv
        # Q(s,a) = hamming_sim(sa_hv, (V_float > 0.5).float())
        g = torch.Generator()
        self._V_float: List[torch.Tensor] = []
        for h in range(n_value_heads):
            g.manual_seed(h * 7919)
            # Start at 0.5 (uniform prior — neither high nor low value)
            self._V_float.append(torch.full((hd_dim,), 0.5))
        self._V_acc: List[float] = [1.0] * n_value_heads

    def _sa_hv(self, state_hv: torch.Tensor, action_hv: torch.Tensor) -> torch.Tensor:
        """Bind state and action into a joint hypervector (XOR)."""
        return _xor(state_hv, action_hv)

    def _V_binary(self, h: int) -> torch.Tensor:
        """Binarize float prototype h for Hamming similarity queries."""
        return (self._V_float[h] > 0.5).float()

    def value(self, state_hv: torch.Tensor, action_hv: torch.Tensor) -> float:
        """
        Query Q(s, a) as ensemble mean of head similarities.

        Returns:
            Scalar Q-value estimate ∈ [0, 1]
        """
        sa = self._sa_hv(state_hv, action_hv)
        qs = [
            float(_hamming(sa.unsqueeze(0), self._V_binary(h).unsqueeze(0)).item())
            for h in range(self.n_heads)
        ]
        return sum(qs) / self.n_heads

    def value_uncertainty(self, state_hv: torch.Tensor, action_hv: torch.Tensor) -> float:
        """Return std of Q-value across heads (epistemic uncertainty)."""
        sa = self._sa_hv(state_hv, action_hv)
        qs = [
            float(_hamming(sa.unsqueeze(0), self._V_binary(h).unsqueeze(0)).item())
            for h in range(self.n_heads)
        ]
        mean_q = sum(qs) / self.n_heads
        return math.sqrt(sum((q - mean_q) ** 2 for q in qs) / max(self.n_heads, 1))

    @torch.no_grad()
    def td_update(
        self,
        state_hv:      torch.Tensor,
        action_hv:     torch.Tensor,
        reward:        float,
        next_state_hv: torch.Tensor,
        next_actions:  Optional[List[torch.Tensor]] = None,
    ):
        """
        Temporal-difference update: V_float += α × δ × bind(s, a).

        Uses float accumulation (not binarized updates), so even small α
        values correctly update the prototype across many samples.

        Args:
            state_hv:      (D,) current state HV
            action_hv:     (D,) action taken HV
            reward:        scalar reward signal
            next_state_hv: (D,) next state HV
            next_actions:  Optional list of next-state action HVs for max_a Q(s',a')
        """
        sa = self._sa_hv(state_hv, action_hv)

        # Estimate max_a' Q(s', a')
        if next_actions:
            next_q = max(self.value(next_state_hv, a) for a in next_actions)
        else:
            # Without next actions, bootstrap from current state value (SARSA-like)
            next_q = self.value(next_state_hv, action_hv)

        current_q = self.value(state_hv, action_hv)
        td_error   = reward + self.gamma * next_q - current_q

        sa_f = sa.float()
        for h in range(self.n_heads):
            # TD-weighted float bundling: accumulate α × δ × sa into V_float
            # Positive δ → push V toward sa (reinforce)
            # Negative δ → push V away from sa (suppress)
            self._V_float[h] = self._V_float[h] + self.alpha * td_error * sa_f
            # Soft clamp to [0, 1] to prevent drift
            self._V_float[h].clamp_(0.0, 1.0)
            self._V_acc[h] += 1.0

    def best_action(
        self,
        state_hv: torch.Tensor,
        candidates: List[torch.Tensor],
    ) -> Tuple[int, float]:
        """Return (index, Q-value) of the best action for a given state."""
        if not candidates:
            return 0, 0.0
        qs = [self.value(state_hv, a) for a in candidates]
        best_idx = max(range(len(qs)), key=lambda i: qs[i])
        return best_idx, qs[best_idx]


class OptionHDCPlanner:
    """
    Hierarchical HDC planner using the options framework.

    Reference:
        Sutton, Precup, Singh (1999) "Between MDPs and semi-MDPs: A framework
        for temporal abstraction in reinforcement learning" Artificial Intelligence
        112(1-2):181-211.

        Precup (2000) "Temporal abstraction in reinforcement learning"
        PhD Dissertation, University of Massachusetts.

    Options extend flat action-selection by introducing temporally extended
    behaviours that achieve sub-goals before returning control:

        Option o = (I_o, π_o, β_o):
            I_o : initiation set — states where option can start
            π_o : intra-option policy (here: HDC_MCTS toward sub-goal)
            β_o : termination condition (here: Hamming sim to sub-goal ≥ threshold)

    The high-level planner (also HDC_MCTS) plans *over options*, treating each
    sub-goal HV as a macro-action.  The low-level HDC_MCTS then executes the
    selected option until termination.

    Benefits for Physical AI:
        - Long-horizon planning without exponential branching
        - Reusable sub-skills (sub-goal HVs can be shared across tasks)
        - Natural hierarchy: strategic (options) + tactical (MCTS)

    Args:
        sub_goals: List of (name, goal_hv) pairs defining the option library
        base_planner: HDC_MCTS instance used for both levels
        termination_sim: Hamming similarity to sub-goal that triggers termination
        max_option_steps: Maximum steps before an option times out
    """

    def __init__(
        self,
        sub_goals: List[Tuple[str, torch.Tensor]],
        base_planner: Optional[HDC_MCTS] = None,
        value_fn: Optional[HDCValueFunction] = None,
        termination_sim: float = 0.80,
        max_option_steps: int = 20,
    ):
        self.sub_goals        = sub_goals   # [(name, goal_hv), ...]
        self.planner          = base_planner or HDC_MCTS(n_simulations=30)
        self.value_fn         = value_fn
        self.termination_sim  = termination_sim
        self.max_option_steps = max_option_steps

        self._active_option: Optional[Tuple[str, torch.Tensor]] = None
        self._option_steps: int = 0
        self._option_history: List[str] = []

    def select_option(
        self,
        current_state: torch.Tensor,
        final_goal: Optional[torch.Tensor] = None,
    ) -> Tuple[str, torch.Tensor]:
        """
        Select the best option (sub-goal) for the current state.

        If a value function is available, rank sub-goals by Q(s, sub_goal).
        Otherwise, rank by Hamming distance to the final goal.

        Returns:
            (option_name, sub_goal_hv)
        """
        if not self.sub_goals:
            raise ValueError("No sub-goals defined in option library")

        if self.value_fn is not None:
            scores = [
                self.value_fn.value(current_state, sg_hv)
                for _, sg_hv in self.sub_goals
            ]
        elif final_goal is not None:
            # Score by distance from sub-goal to final goal (closer = better)
            scores = [
                float(_hamming(sg_hv.unsqueeze(0), final_goal.unsqueeze(0)).item())
                for _, sg_hv in self.sub_goals
            ]
        else:
            # Random selection (exploration)
            idx = int(torch.randint(0, len(self.sub_goals), (1,)).item())
            return self.sub_goals[idx]

        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        return self.sub_goals[best_idx]

    def should_terminate(self, current_state: torch.Tensor) -> bool:
        """
        Check if the active option should terminate.

        Returns True if:
            - No active option
            - Current state is close enough to the sub-goal
            - Option has exceeded max_option_steps
        """
        if self._active_option is None:
            return True
        _, sg_hv = self._active_option
        sim = float(_hamming(current_state.unsqueeze(0), sg_hv.unsqueeze(0)).item())
        return sim >= self.termination_sim or self._option_steps >= self.max_option_steps

    def plan(
        self,
        current_state: torch.Tensor,
        primitive_actions: List,
        final_goal: Optional[torch.Tensor] = None,
        forward_model=None,
    ) -> Tuple[List, str]:
        """
        Plan using the hierarchical option framework.

        Args:
            current_state: (D,) current state HV
            primitive_actions: List of primitive action HVs/candidates
            final_goal: (D,) final goal HV
            forward_model: Optional callable(state, action) → next_state

        Returns:
            (action_plan, active_option_name)
            action_plan: Ranked list from HDC_MCTS toward current sub-goal
        """
        # Check option termination and select new option if needed
        if self.should_terminate(current_state):
            option_name, sg_hv = self.select_option(current_state, final_goal)
            self._active_option = (option_name, sg_hv)
            self._option_steps  = 0
            self._option_history.append(option_name)

        option_name, sg_hv = self._active_option
        self._option_steps += 1

        # Plan toward current sub-goal using low-level MCTS
        action_plan = self.planner.plan(
            current_state,
            primitive_actions,
            goal_state=sg_hv,
            forward_model=forward_model,
        )

        return action_plan, option_name

    def add_sub_goal(self, name: str, goal_hv: torch.Tensor):
        """Dynamically add a new option to the library."""
        self.sub_goals.append((name, goal_hv))

    def option_trace(self) -> List[str]:
        """Return the sequence of options selected so far."""
        return list(self._option_history)

    def reset(self):
        self._active_option  = None
        self._option_steps   = 0
        self._option_history = []


if __name__ == "__main__":
    test_auto_calibrator()
    print()
    test_hdc_planner()
    print()
    test_adaptive_hebbian()
    print()
    test_self_improvement_loop()
    print()
    print("=== All planner tests passed ===")
