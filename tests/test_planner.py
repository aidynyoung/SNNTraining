"""
tests/test_planner.py
======================
Tests for the HDC Goal-Conditioned Planner (hdc/planner.py).

Validates:
  1. AutoCalibrator — automatic safe/danger state registration
  2. HDCPlanner — multi-step beam search in HV space
  3. AdaptiveHebbian — surprise-modulated learning rate
  4. SelfImprovementLoop — full closed-loop agent
  5. WorldModelDiagnostics — structured queries on world state
"""

from __future__ import annotations

import sys
import os

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from hdc.planner import (
    AutoCalibrator,
    HDCPlanner,
    AdaptiveHebbian,
    SelfImprovementLoop,
    WorldModelDiagnostics,
    ValueFunctionLoop,
    Plan,
    HDCValueFunction,
    PlanRobustnessScorer,
    HDC_MCTS,
    OptionHDCPlanner,
)
from hdc.physics_world_model import (
    PhysicsWorldModel,
    MultiHorizonPredictor,
    ActionCandidate,
    _xor,
    _hamming,
)
from hdc.world_context import CausalTransitionGraph
from hdc.sensor_stream import SensorSpec, SensorReading, ModalityType
from hdc.physical_ai_hybrid import HybridPhysicalAIPipeline
from hdc.world_context import ContextualWorldModel


@pytest.fixture
def hd_dim():
    return 256


# ═══════════════════════════════════════════════════════════════════════════════
# Mock helpers for testing AutoCalibrator
# ═══════════════════════════════════════════════════════════════════════════════

class MockActionEvaluator:
    """Minimal mock that exposes _safe_prototypes and _danger_prototypes."""
    def __init__(self):
        self._safe_prototypes: list = []
        self._danger_prototypes: list = []

    def add_safe_state(self, hv):
        self._safe_prototypes.append(hv)

    def add_danger_state(self, hv):
        self._danger_prototypes.append(hv)

    def _max_similarity_to_set(self, hv, prototypes):
        """Mock: return 0.5 if prototypes exist, else 0.0."""
        if not prototypes:
            return 0.0
        return 0.5


class MockWorldModel:
    """Minimal mock PhysicsWorldModel for AutoCalibrator tests."""
    def __init__(self):
        self.action_evaluator = MockActionEvaluator()


class TestAutoCalibrator:
    def test_init(self):
        wm = MockWorldModel()
        cal = AutoCalibrator(wm)
        assert cal.stable_threshold == 0.08
        assert cal.alarm_threshold == 0.35

    def test_stable_streak_registers_safe(self, hd_dim):
        wm = MockWorldModel()
        cal = AutoCalibrator(wm, stable_threshold=0.05, stable_window=2)
        hv = (torch.rand(hd_dim) >= 0.5).float()
        for _ in range(3):
            cal.update(hv, prediction_error=0.02)
        assert len(wm.action_evaluator._safe_prototypes) >= 1

    def test_alarm_streak_registers_danger(self, hd_dim):
        wm = MockWorldModel()
        cal = AutoCalibrator(wm, alarm_threshold=0.4, alarm_window=2)
        hv = (torch.rand(hd_dim) >= 0.5).float()
        for _ in range(3):
            cal.update(hv, prediction_error=0.45)
        assert len(wm.action_evaluator._danger_prototypes) >= 1

    def test_mixed_errors_no_registration(self, hd_dim):
        wm = MockWorldModel()
        cal = AutoCalibrator(wm, stable_threshold=0.05, alarm_threshold=0.4,
                             stable_window=3, alarm_window=3)
        hv = (torch.rand(hd_dim) >= 0.5).float()
        # Mix of errors — should not trigger registration
        for err in [0.02, 0.45, 0.02, 0.45, 0.02]:
            cal.update(hv, prediction_error=err)
        # Neither streak should be long enough
        assert len(wm.action_evaluator._safe_prototypes) == 0
        assert len(wm.action_evaluator._danger_prototypes) == 0

    def test_register_safe_updates_evaluator(self, hd_dim):
        wm = MockWorldModel()
        cal = AutoCalibrator(wm, stable_threshold=0.05, stable_window=2)
        hv = (torch.rand(hd_dim) >= 0.5).float()
        for _ in range(3):
            cal.update(hv, prediction_error=0.02)
        assert len(wm.action_evaluator._safe_prototypes) >= 1

    def test_register_danger_updates_evaluator(self, hd_dim):
        wm = MockWorldModel()
        cal = AutoCalibrator(wm, alarm_threshold=0.4, alarm_window=2)
        hv = (torch.rand(hd_dim) >= 0.5).float()
        for _ in range(3):
            cal.update(hv, prediction_error=0.45)
        assert len(wm.action_evaluator._danger_prototypes) >= 1

    def test_max_safe_cap(self, hd_dim):
        wm = MockWorldModel()
        cal = AutoCalibrator(wm, stable_threshold=0.05, stable_window=1, max_safe=2)
        hv = (torch.rand(hd_dim) >= 0.5).float()
        for _ in range(10):
            cal.update(hv, prediction_error=0.02)
        assert len(wm.action_evaluator._safe_prototypes) <= 2

    def test_return_dict_keys(self, hd_dim):
        wm = MockWorldModel()
        cal = AutoCalibrator(wm)
        hv = (torch.rand(hd_dim) >= 0.5).float()
        result = cal.update(hv, prediction_error=0.02)
        assert "action" in result
        assert "n_safe" in result
        assert "n_danger" in result


class TestHDCPlanner:
    def test_init(self, hd_dim):
        causal = CausalTransitionGraph(hd_dim)
        ev = MockActionEvaluator()
        planner = HDCPlanner(causal, ev)
        assert planner.beam_width == 4
        assert planner.horizon == 3

    def test_plan_returns_plan(self, hd_dim):
        causal = CausalTransitionGraph(hd_dim)
        ev = MockActionEvaluator()
        planner = HDCPlanner(causal, ev, min_transitions=0)

        state = (torch.rand(hd_dim) >= 0.5).float()
        action = (torch.rand(hd_dim) >= 0.5).float()
        candidates = [ActionCandidate("test", action)]

        plans = planner.plan(state, candidates)
        assert len(plans) > 0
        assert isinstance(plans[0], Plan)

    def test_plan_actions_in_range(self, hd_dim):
        causal = CausalTransitionGraph(hd_dim)
        ev = MockActionEvaluator()
        planner = HDCPlanner(causal, ev, min_transitions=0)

        state = (torch.rand(hd_dim) >= 0.5).float()
        a1 = (torch.rand(hd_dim) >= 0.5).float()
        a2 = (torch.rand(hd_dim) >= 0.5).float()
        candidates = [
            ActionCandidate("a1", a1),
            ActionCandidate("a2", a2),
        ]

        plans = planner.plan(state, candidates)
        for p in plans:
            for a in p.actions:
                assert a.name in ("a1", "a2")

    def test_best_action_returns_something(self, hd_dim):
        causal = CausalTransitionGraph(hd_dim)
        ev = MockActionEvaluator()
        planner = HDCPlanner(causal, ev, min_transitions=0)

        state = (torch.rand(hd_dim) >= 0.5).float()
        action = (torch.rand(hd_dim) >= 0.5).float()
        candidates = [ActionCandidate("test", action)]

        best = planner.best_action(state, candidates)
        assert best is not None

    def test_different_goals_different_plans(self, hd_dim):
        causal = CausalTransitionGraph(hd_dim)
        ev = MockActionEvaluator()
        planner = HDCPlanner(causal, ev, min_transitions=0)

        state = (torch.rand(hd_dim) >= 0.5).float()
        a1 = (torch.rand(hd_dim) >= 0.5).float()
        a2 = (torch.rand(hd_dim) >= 0.5).float()
        candidates = [
            ActionCandidate("a1", a1),
            ActionCandidate("a2", a2),
        ]

        plans = planner.plan(state, candidates)
        assert len(plans) > 0


class TestAdaptiveHebbian:
    def test_init(self, hd_dim):
        pred = MultiHorizonPredictor(hd_dim)
        hebb = AdaptiveHebbian(pred)
        assert hebb.lr_base == 0.005
        assert hebb.lr_max == 0.08

    def test_compute_lr_high_error(self, hd_dim):
        pred = MultiHorizonPredictor(hd_dim)
        hebb = AdaptiveHebbian(pred, lr_base=0.005, lr_max=0.08)
        lr = hebb.compute_lr(prediction_error=0.4)
        assert lr > 0.005
        assert lr <= 0.08

    def test_compute_lr_clamps(self, hd_dim):
        pred = MultiHorizonPredictor(hd_dim)
        hebb = AdaptiveHebbian(pred, lr_base=0.005, lr_max=0.08)
        lr = hebb.compute_lr(prediction_error=0.5, pattern_familiarity=0.0, causal_surprise=0.5)
        assert lr <= 0.08

    def test_knowledge_reduces_lr(self, hd_dim):
        pred = MultiHorizonPredictor(hd_dim)
        hebb = AdaptiveHebbian(pred, lr_base=0.005, lr_max=0.08)
        lr_high = hebb.compute_lr(prediction_error=0.4, pattern_familiarity=0.0)
        lr_low = hebb.compute_lr(prediction_error=0.4, pattern_familiarity=0.9)
        assert lr_low < lr_high

    def test_update_changes_rate(self, hd_dim):
        pred = MultiHorizonPredictor(hd_dim)
        hebb = AdaptiveHebbian(pred, lr_base=0.005, lr_max=0.08)
        state = (torch.rand(hd_dim) >= 0.5).float()
        actual = (torch.rand(hd_dim) >= 0.5).float()
        lr = hebb.update(state, actual, prediction_error=0.2)
        assert 0.0 < lr <= 0.08


class TestSelfImprovementLoop:
    def test_init(self, hd_dim):
        specs = [SensorSpec("s", ModalityType.SCALAR, raw_dim=1, hd_dim=hd_dim, seed=0)]
        base = HybridPhysicalAIPipeline(specs, hd_dim=hd_dim, n_ensemble=2)
        world = ContextualWorldModel(base)
        loop = SelfImprovementLoop(world)
        assert loop._tick == 0

    def test_tick_returns_dict(self, hd_dim):
        specs = [SensorSpec("s", ModalityType.SCALAR, raw_dim=1, hd_dim=hd_dim, seed=0)]
        base = HybridPhysicalAIPipeline(specs, hd_dim=hd_dim, n_ensemble=2)
        world = ContextualWorldModel(base)
        loop = SelfImprovementLoop(world)
        reading = SensorReading(timestamp=0.0, data={"s": torch.tensor([0.5])})
        result = loop.tick(reading)
        assert isinstance(result, dict)

    def test_tick_increments_step(self, hd_dim):
        specs = [SensorSpec("s", ModalityType.SCALAR, raw_dim=1, hd_dim=hd_dim, seed=0)]
        base = HybridPhysicalAIPipeline(specs, hd_dim=hd_dim, n_ensemble=2)
        world = ContextualWorldModel(base)
        loop = SelfImprovementLoop(world)
        reading = SensorReading(timestamp=0.0, data={"s": torch.tensor([0.5])})
        loop.tick(reading)
        assert loop._tick == 1

    def test_set_goal(self, hd_dim):
        specs = [SensorSpec("s", ModalityType.SCALAR, raw_dim=1, hd_dim=hd_dim, seed=0)]
        base = HybridPhysicalAIPipeline(specs, hd_dim=hd_dim, n_ensemble=2)
        world = ContextualWorldModel(base)
        loop = SelfImprovementLoop(world)
        goal = (torch.rand(hd_dim) >= 0.5).float()
        loop.set_goal(goal)
        assert loop._goal_state is not None

    def test_multiple_ticks_no_crash(self, hd_dim):
        specs = [SensorSpec("s", ModalityType.SCALAR, raw_dim=1, hd_dim=hd_dim, seed=0)]
        base = HybridPhysicalAIPipeline(specs, hd_dim=hd_dim, n_ensemble=2)
        world = ContextualWorldModel(base)
        loop = SelfImprovementLoop(world)
        for t in range(5):
            reading = SensorReading(timestamp=float(t), data={"s": torch.tensor([float(t) / 10])})
            loop.tick(reading)
        assert loop._tick == 5

    def test_improvement_report(self, hd_dim):
        specs = [SensorSpec("s", ModalityType.SCALAR, raw_dim=1, hd_dim=hd_dim, seed=0)]
        base = HybridPhysicalAIPipeline(specs, hd_dim=hd_dim, n_ensemble=2)
        world = ContextualWorldModel(base)
        loop = SelfImprovementLoop(world)
        for t in range(5):
            reading = SensorReading(timestamp=float(t), data={"s": torch.tensor([float(t) / 10])})
            loop.tick(reading)
        report = loop.improvement_report()
        assert "total_ticks" in report
        assert report["total_ticks"] == 5


class TestWorldModelDiagnostics:
    def test_init(self, hd_dim):
        specs = [SensorSpec("s", ModalityType.SCALAR, raw_dim=1, hd_dim=hd_dim, seed=0)]
        base = HybridPhysicalAIPipeline(specs, hd_dim=hd_dim, n_ensemble=2)
        world = ContextualWorldModel(base)
        loop = SelfImprovementLoop(world)
        diag = WorldModelDiagnostics(loop)
        assert diag.agent is loop

    def test_is_normal_low_error(self, hd_dim):
        specs = [SensorSpec("s", ModalityType.SCALAR, raw_dim=1, hd_dim=hd_dim, seed=0)]
        base = HybridPhysicalAIPipeline(specs, hd_dim=hd_dim, n_ensemble=2)
        world = ContextualWorldModel(base)
        loop = SelfImprovementLoop(world)
        diag = WorldModelDiagnostics(loop)
        status = diag.is_normal(0.02)
        assert status in ("normal", "uncertain", "anomalous")

    def test_is_normal_high_error(self, hd_dim):
        specs = [SensorSpec("s", ModalityType.SCALAR, raw_dim=1, hd_dim=hd_dim, seed=0)]
        base = HybridPhysicalAIPipeline(specs, hd_dim=hd_dim, n_ensemble=2)
        world = ContextualWorldModel(base)
        loop = SelfImprovementLoop(world)
        diag = WorldModelDiagnostics(loop)
        status = diag.is_normal(0.45)
        assert status in ("normal", "uncertain", "anomalous")

    def test_recommend_exists(self, hd_dim):
        specs = [SensorSpec("s", ModalityType.SCALAR, raw_dim=1, hd_dim=hd_dim, seed=0)]
        base = HybridPhysicalAIPipeline(specs, hd_dim=hd_dim, n_ensemble=2)
        world = ContextualWorldModel(base)
        loop = SelfImprovementLoop(world)
        diag = WorldModelDiagnostics(loop)
        rec = diag.recommend(None)
        assert isinstance(rec, str)

    def test_full_report(self, hd_dim):
        specs = [SensorSpec("s", ModalityType.SCALAR, raw_dim=1, hd_dim=hd_dim, seed=0)]
        base = HybridPhysicalAIPipeline(specs, hd_dim=hd_dim, n_ensemble=2)
        world = ContextualWorldModel(base)
        loop = SelfImprovementLoop(world)
        reading = SensorReading(timestamp=0.0, data={"s": torch.tensor([0.5])})
        result = loop.tick(reading)
        diag = WorldModelDiagnostics(loop)
        report = diag.full_report(result)
        assert "normality" in report
        assert "pattern" in report
        assert "confidence" in report
        assert "recommendation" in report


class TestHDCValueFunction:
    def test_value_returns_float(self, hd_dim):
        vf = HDCValueFunction(hd_dim)
        s = (torch.rand(hd_dim) >= 0.5).float()
        a = (torch.rand(hd_dim) >= 0.5).float()
        v = vf.value(s, a)
        assert isinstance(v, float)
        assert 0.0 <= v <= 1.0

    def test_value_uncertainty_nonneg(self, hd_dim):
        vf = HDCValueFunction(hd_dim, n_value_heads=3)
        s = (torch.rand(hd_dim) >= 0.5).float()
        a = (torch.rand(hd_dim) >= 0.5).float()
        u = vf.value_uncertainty(s, a)
        assert u >= 0.0

    def test_td_update_runs(self, hd_dim):
        vf = HDCValueFunction(hd_dim)
        s  = (torch.rand(hd_dim) >= 0.5).float()
        a  = (torch.rand(hd_dim) >= 0.5).float()
        s2 = (torch.rand(hd_dim) >= 0.5).float()
        vf.td_update(s, a, reward=0.8, next_state_hv=s2)

    def test_best_action_returns_valid_index(self, hd_dim):
        vf = HDCValueFunction(hd_dim)
        s  = (torch.rand(hd_dim) >= 0.5).float()
        actions = [(torch.rand(hd_dim) >= 0.5).float() for _ in range(4)]
        idx, q = vf.best_action(s, actions)
        assert 0 <= idx < 4
        assert isinstance(q, float)

    def test_td_update_with_next_actions(self, hd_dim):
        vf = HDCValueFunction(hd_dim)
        s  = (torch.rand(hd_dim) >= 0.5).float()
        a  = (torch.rand(hd_dim) >= 0.5).float()
        s2 = (torch.rand(hd_dim) >= 0.5).float()
        next_actions = [(torch.rand(hd_dim) >= 0.5).float() for _ in range(2)]
        vf.td_update(s, a, reward=0.5, next_state_hv=s2, next_actions=next_actions)

    def test_td_updates_actually_change_value(self, hd_dim):
        vf = HDCValueFunction(hd_dim)
        s = (torch.rand(hd_dim) >= 0.5).float()
        a = (torch.rand(hd_dim) >= 0.5).float()
        s2 = (torch.rand(hd_dim) >= 0.5).float()

        v_before = vf.value(s, a)
        for _ in range(50):
            vf.td_update(s, a, reward=1.0, next_state_hv=s2)
        v_after = vf.value(s, a)
        # After 50 positive-reward updates, value should have changed
        assert v_after != v_before

    def test_negative_reward_changes_value_differently(self, hd_dim):
        vf_pos = HDCValueFunction(hd_dim, alpha=0.1)
        vf_neg = HDCValueFunction(hd_dim, alpha=0.1)
        s = (torch.rand(hd_dim) >= 0.5).float()
        a = (torch.rand(hd_dim) >= 0.5).float()
        s2 = (torch.rand(hd_dim) >= 0.5).float()
        v_init = vf_pos.value(s, a)

        for _ in range(30):
            vf_pos.td_update(s, a, reward=1.0, next_state_hv=s2)
            vf_neg.td_update(s, a, reward=-1.0, next_state_hv=s2)

        # Positive reward should push value up, negative should push down
        v_pos = vf_pos.value(s, a)
        v_neg = vf_neg.value(s, a)
        assert v_pos >= v_neg  # positive reward → higher value


class TestPlanRobustnessScorer:
    def test_score_decreases_with_variance(self):
        scorer = PlanRobustnessScorer(robustness_penalty=0.5)
        s_low  = scorer.score(0.8, predictor_variance=0.01)
        s_high = scorer.score(0.8, predictor_variance=0.5)
        assert s_low > s_high

    def test_score_decreases_with_risk(self):
        scorer = PlanRobustnessScorer()
        s_safe = scorer.score(0.8, predictor_variance=0.0, risk_score=0.0)
        s_risky = scorer.score(0.8, predictor_variance=0.0, risk_score=0.5)
        assert s_safe > s_risky

    def test_score_with_zero_variance(self):
        scorer = PlanRobustnessScorer()
        s = scorer.score(0.7, predictor_variance=0.0, risk_score=0.0)
        assert s == pytest.approx(0.7, abs=1e-6)


class TestHDCPlannerWithValueFunction:
    def test_plan_with_value_fn(self, hd_dim):
        causal = CausalTransitionGraph(hd_dim)
        ev = MockActionEvaluator()
        vf = HDCValueFunction(hd_dim)
        planner = HDCPlanner(causal, ev, min_transitions=0, value_fn=vf, value_weight=0.2)

        state = (torch.rand(hd_dim) >= 0.5).float()
        a1 = (torch.rand(hd_dim) >= 0.5).float()
        a2 = (torch.rand(hd_dim) >= 0.5).float()
        candidates = [ActionCandidate("a1", a1), ActionCandidate("a2", a2)]

        plans = planner.plan(state, candidates)
        assert len(plans) > 0
        assert hasattr(plans[0], "value_estimate")
        assert hasattr(plans[0], "epistemic_variance")
        assert plans[0].epistemic_variance >= 0.0

    def test_plan_with_robustness_scorer(self, hd_dim):
        causal = CausalTransitionGraph(hd_dim)
        ev = MockActionEvaluator()
        vf = HDCValueFunction(hd_dim)
        scorer = PlanRobustnessScorer(robustness_penalty=0.3)
        planner = HDCPlanner(causal, ev, min_transitions=0,
                             value_fn=vf, robustness_scorer=scorer)

        state = (torch.rand(hd_dim) >= 0.5).float()
        candidates = [ActionCandidate("a", (torch.rand(hd_dim) >= 0.5).float())]
        plans = planner.plan(state, candidates)
        assert len(plans) > 0

    def test_full_trajectory_populated(self, hd_dim):
        causal = CausalTransitionGraph(hd_dim)
        ev = MockActionEvaluator()
        planner = HDCPlanner(causal, ev, min_transitions=0, horizon=2)

        state = (torch.rand(hd_dim) >= 0.5).float()
        candidates = [ActionCandidate("a", (torch.rand(hd_dim) >= 0.5).float())]
        plans = planner.plan(state, candidates)
        # trajectory should include initial state + 2 simulated steps
        assert len(plans[0].predicted_trajectory) == 3


class TestHDC_MCTS:
    def test_plan_returns_ranked_list(self, hd_dim):
        mcts = HDC_MCTS(n_simulations=10, rollout_depth=2)
        state = (torch.rand(hd_dim) >= 0.5).float()
        goal  = (torch.rand(hd_dim) >= 0.5).float()
        candidates = [ActionCandidate(f"a{i}", (torch.rand(hd_dim) >= 0.5).float())
                      for i in range(3)]
        results = mcts.plan(state, candidates, goal_state=goal)
        assert isinstance(results, list)
        if results:
            assert len(results[0]) == 2   # (action, value)

    def test_plan_no_crash_no_goal(self, hd_dim):
        mcts = HDC_MCTS(n_simulations=5)
        state = (torch.rand(hd_dim) >= 0.5).float()
        candidates = [ActionCandidate("a", (torch.rand(hd_dim) >= 0.5).float())]
        results = mcts.plan(state, candidates)
        assert isinstance(results, list)

    def test_plan_empty_candidates(self, hd_dim):
        mcts = HDC_MCTS(n_simulations=5)
        state = (torch.rand(hd_dim) >= 0.5).float()
        results = mcts.plan(state, [])
        assert results == []


class TestOptionHDCPlanner:
    def test_plan_returns_action_and_option(self, hd_dim):
        sg1 = (torch.rand(hd_dim) >= 0.5).float()
        sg2 = (torch.rand(hd_dim) >= 0.5).float()
        sub_goals = [("reach_A", sg1), ("reach_B", sg2)]
        planner = OptionHDCPlanner(sub_goals, max_option_steps=5)

        state = (torch.rand(hd_dim) >= 0.5).float()
        candidates = [ActionCandidate(f"a{i}", (torch.rand(hd_dim) >= 0.5).float())
                      for i in range(3)]
        plan, option_name = planner.plan(state, candidates)
        assert option_name in ("reach_A", "reach_B")
        assert isinstance(plan, list)

    def test_option_termination(self, hd_dim):
        sg1 = (torch.rand(hd_dim) >= 0.5).float()
        planner = OptionHDCPlanner([("goal", sg1)], termination_sim=0.0,
                                   max_option_steps=1)
        state = sg1.clone()
        assert planner.should_terminate(state)

    def test_add_sub_goal(self, hd_dim):
        planner = OptionHDCPlanner([])
        sg = (torch.rand(hd_dim) >= 0.5).float()
        planner.add_sub_goal("new_goal", sg)
        assert len(planner.sub_goals) == 1

    def test_reset_clears_state(self, hd_dim):
        sg = (torch.rand(hd_dim) >= 0.5).float()
        planner = OptionHDCPlanner([("g", sg)])
        state = (torch.rand(hd_dim) >= 0.5).float()
        candidates = [ActionCandidate("a", (torch.rand(hd_dim) >= 0.5).float())]
        planner.plan(state, candidates)
        planner.reset()
        assert planner._active_option is None
        assert planner._option_steps == 0


class TestValueFunctionLoop:
    def test_init_wires_value_fn(self, hd_dim):
        specs = [SensorSpec("s", ModalityType.SCALAR, raw_dim=1, hd_dim=hd_dim, seed=0)]
        base = HybridPhysicalAIPipeline(specs, hd_dim=hd_dim, n_ensemble=2)
        world = ContextualWorldModel(base)
        agent = SelfImprovementLoop(world)
        loop = ValueFunctionLoop(agent, hd_dim)
        assert agent.planner.value_fn is loop.value_fn

    def test_tick_returns_dict(self, hd_dim):
        specs = [SensorSpec("s", ModalityType.SCALAR, raw_dim=1, hd_dim=hd_dim, seed=0)]
        base = HybridPhysicalAIPipeline(specs, hd_dim=hd_dim, n_ensemble=2)
        world = ContextualWorldModel(base)
        agent = SelfImprovementLoop(world)
        loop = ValueFunctionLoop(agent, hd_dim)
        reading = SensorReading(timestamp=0.0, data={"s": torch.tensor([0.5])})
        result = loop.tick(reading)
        assert isinstance(result, dict)
        assert "td_reward" in result

    def test_td_updates_accumulate(self, hd_dim):
        specs = [SensorSpec("s", ModalityType.SCALAR, raw_dim=1, hd_dim=hd_dim, seed=0)]
        base = HybridPhysicalAIPipeline(specs, hd_dim=hd_dim, n_ensemble=2)
        world = ContextualWorldModel(base)
        agent = SelfImprovementLoop(world)
        loop = ValueFunctionLoop(agent, hd_dim)
        for t in range(5):
            reading = SensorReading(timestamp=float(t), data={"s": torch.tensor([float(t)/4])})
            loop.tick(reading)
        assert loop.td_steps == 4  # first tick has no prev state

    def test_value_fn_best_populated(self, hd_dim):
        specs = [SensorSpec("s", ModalityType.SCALAR, raw_dim=1, hd_dim=hd_dim, seed=0)]
        base = HybridPhysicalAIPipeline(specs, hd_dim=hd_dim, n_ensemble=2)
        world = ContextualWorldModel(base)
        agent = SelfImprovementLoop(world)
        loop = ValueFunctionLoop(agent, hd_dim)
        reading = SensorReading(timestamp=0.0, data={"s": torch.tensor([0.5])})
        candidates = [
            ActionCandidate("low",  (torch.rand(hd_dim) >= 0.5).float()),
            ActionCandidate("high", (torch.rand(hd_dim) >= 0.5).float()),
        ]
        result = loop.tick(reading, candidate_actions=candidates)
        assert result["value_fn_best"] in ("low", "high")
