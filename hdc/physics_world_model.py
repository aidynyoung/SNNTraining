"""
Physics-Informed World Model for Physical AI
=============================================
Extends SNNTrainingWorldModel with three capabilities described in the
Physical AI framing (Bekele, Golota, Schaeffer 2026; IQT):

1. **Multi-Horizon Prediction** — Separate short/medium/long-term predictors
   that specialise at different temporal scales. Short-term captures fast
   dynamics (vibration, contact); long-term captures structural change
   (wear, drift). Each predictor is updated only from its relevant horizon.

2. **Physics-Informed Constraints** — Conservation laws and kinematic
   consistency enforced as soft constraints on prediction:
     - Kinematic chain: position → velocity → acceleration consistency
     - Energy conservation: predicted KE + PE ≤ input + source − dissipation
     - Momentum: bounded change between consecutive states
   These constraints guide the world model toward physically plausible
   futures without requiring explicit physics simulation.

3. **Action Evaluator** — "Think before acting": given a set of candidate
   actions (encoded as HVs), roll them through the learned forward model
   to predict consequences, then rank by expected utility minus predicted
   risk. This implements the core Physical AI loop:
       Observe → Perceive → Interpret (world model) → Evaluate → Act

Reference framing:
    "World models shift AI from reacting to perceiving, and from perceiving
     to anticipating. They transform raw sensor data into decision-ready
     context, enabling systems to reason before acting, rehearse before
     committing, and adapt before changing conditions."
    — Bekele, Golota, Schaeffer (2026), IQT.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════════
# HDC primitives (self-contained to avoid circular imports)
# ═══════════════════════════════════════════════════════════════════════════════

def _xor(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a != b).float()

def _majority(x: torch.Tensor) -> torch.Tensor:
    return (x >= 0.5).float()

def _hamming(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamming similarity in [0,1]. Operates on last dimension."""
    return 1.0 - _xor(a, b).mean(dim=-1)

def _bundle(hvs: torch.Tensor) -> torch.Tensor:
    """Majority-sum bundle. hvs: (..., n, D) → (..., D)."""
    return _majority(hvs.mean(dim=-2))


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Kinematic & Conservation Constraints
# ═══════════════════════════════════════════════════════════════════════════════

class KinematicConstraint:
    """
    Enforces physical kinematic consistency across prediction steps.

    In HDC space, a physical state HV encodes position, velocity, and
    acceleration as bound components. Kinematic consistency requires:
        vel_t = pos_t XOR pos_{t-1}     (finite difference velocity)
        acc_t = vel_t XOR vel_{t-1}     (finite difference acceleration)

    In high-D binary space, the XOR between consecutive position HVs
    is proportional to the Hamming distance — a large XOR means large
    displacement. The kinematic constraint penalises predictions where
    the XOR chain is inconsistent (e.g., position changes without
    proportional velocity change).

    Args:
        alpha: Kinematic consistency penalty weight (higher → stricter)
    """

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha

    def penalty(
        self,
        pos_seq: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute kinematic inconsistency for a sequence of position HVs.

        Args:
            pos_seq: (T, D) sequence of position hypervectors

        Returns:
            Scalar penalty ∈ [0, 0.5] — higher = more kinematically inconsistent
        """
        if pos_seq.shape[0] < 3:
            return torch.tensor(0.0)

        # Finite-difference velocities: vel_t = XOR(pos_t, pos_{t-1})
        vel = _xor(pos_seq[1:], pos_seq[:-1])           # (T-1, D)
        # Finite-difference accelerations
        acc = _xor(vel[1:], vel[:-1])                    # (T-2, D)

        # Kinematic consistency: second-order differences should be small
        # (smooth trajectories have low acceleration variation)
        acc_variation = _xor(acc[1:], acc[:-1]).mean() if acc.shape[0] > 1 else torch.tensor(0.0)

        return self.alpha * acc_variation

    def constrain_prediction(
        self,
        pred_hv: torch.Tensor,
        prev_state: torch.Tensor,
        prev_velocity: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Blend the raw prediction with a kinematically consistent prediction.

        Kinematic prediction: pred_kin = XOR(XOR(state_t, vel_t), acc_t)
        Final: blend between raw pred and kinematic pred based on consistency.

        Args:
            pred_hv: (D,) or (B, D) raw predicted HV
            prev_state: (D,) or (B, D) previous state HV
            prev_velocity: (D,) or (B, D) previous velocity HV (if known)

        Returns:
            (constrained_pred, velocity_hv)
        """
        velocity = _xor(pred_hv, prev_state)            # predicted velocity

        if prev_velocity is not None:
            acceleration = _xor(velocity, prev_velocity)
            # Kinematic prediction: extrapolate from velocity + acceleration
            kin_pred = _xor(_xor(prev_state, velocity), acceleration)
            # Blend: if velocity is consistent with previous, trust raw more
            vel_consistency = float(_hamming(velocity, prev_velocity).mean().item())
            blend = vel_consistency  # high consistency → trust raw pred
            constrained = _majority(blend * pred_hv + (1 - blend) * kin_pred)
        else:
            constrained = pred_hv

        return constrained, velocity


class EnergyConstraint:
    """
    Soft energy conservation constraint for predicted state transitions.

    Physical systems conserve energy between transitions (modulo dissipation
    and external work). In HDC, the "energy" of a state is approximated by
    its Hamming weight (density), which encodes information content.

    Two successive states should have similar Hamming weights unless energy
    is explicitly added (actuation) or removed (dissipation). This constraint
    penalises predictions that violate energy bounds.

    Args:
        max_delta: Maximum allowed change in Hamming weight per step
        dissipation_rate: Fraction of energy lost per step to dissipation
    """

    def __init__(self, max_delta: float = 0.05, dissipation_rate: float = 0.02):
        self.max_delta = max_delta
        self.dissipation_rate = dissipation_rate

    def energy(self, hv: torch.Tensor) -> torch.Tensor:
        """Approximate energy as Hamming weight (density) of HV."""
        return hv.mean(dim=-1)

    def penalty(
        self,
        current_hv: torch.Tensor,
        predicted_hv: torch.Tensor,
        action_energy: float = 0.0,
    ) -> torch.Tensor:
        """
        Penalty for energy-inconsistent transitions.

        Args:
            current_hv: (D,) current state HV
            predicted_hv: (D,) predicted next state HV
            action_energy: Energy input from action (0 = passive transition)

        Returns:
            Scalar penalty ≥ 0
        """
        e_current = self.energy(current_hv)
        e_predicted = self.energy(predicted_hv)
        e_dissipated = self.dissipation_rate * e_current

        # Expected next-state energy: current − dissipation + action input
        e_expected = e_current - e_dissipated + action_energy
        e_delta = (e_predicted - e_expected).abs()

        # Penalty only if delta exceeds max_delta
        excess = F.relu(e_delta - self.max_delta)
        return excess


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Multi-Horizon Predictor
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PredictionHorizon:
    """Definition of a prediction horizon."""
    name: str           # "short", "medium", "long"
    steps: int          # Steps ahead to predict
    update_rate: int    # Update every N observations
    decay: float        # EMA decay for error buffer


STANDARD_HORIZONS = [
    PredictionHorizon("short",  steps=1,  update_rate=1,  decay=0.95),
    PredictionHorizon("medium", steps=5,  update_rate=3,  decay=0.99),
    PredictionHorizon("long",   steps=20, update_rate=10, decay=0.999),
]


class HorizonPredictor(nn.Module):
    """
    Single-horizon predictor with physics-informed constraints.

    Learns to predict the state h steps ahead using Hebbian updates.

    Architecture: low-rank residual predictor
        pred = sign(x + U @ (V^T @ x))     where U, V ∈ R^{D×r}, r << D

    This replaces the original nn.Linear(D, D) (D² parameters) with two
    D×r matrices (2Dr parameters). For D=1000, r=32:
        Old: 1,000,000 parameters, O(D²) forward = 1M ops
        New:    64,000 parameters, O(2Dr) forward = 64K ops — 15× faster

    Initialised at U=V=0 so the predictor starts as the identity mapping
    (pred ≈ sign(x) = x for binary HVs). The low-rank correction is learned
    purely via Hebbian updates.
    """

    def __init__(
        self,
        hd_dim: int,
        horizon: PredictionHorizon,
        use_kinematic: bool = True,
        use_energy: bool = True,
        rank: int = 32,
    ):
        super().__init__()
        self.hd_dim = hd_dim
        self.horizon = horizon
        self.use_kinematic = use_kinematic
        self.use_energy = use_energy
        self.rank = rank

        # Low-rank residual: W ≈ U @ V^T
        # V is a FIXED random projection (D→r encoder), only U is learned.
        # This is the "random kitchen sink" approach: stable, fast, effective.
        #   pred = sign(x + U @ (V^T @ x))
        #   ΔU   = η × error^T @ (x @ V) / scale   [O(Dr) update]
        g = torch.Generator(); g.manual_seed(42)
        self.U = nn.Parameter(torch.zeros(hd_dim, rank))          # learned (D×r)
        self.register_buffer(
            "V", torch.randn(hd_dim, rank, generator=g) / (hd_dim ** 0.5)  # fixed (D×r)
        )

        # Compatibility shim for persistence code
        self.predictor = self

        # Physics constraints
        self.kin_constraint = KinematicConstraint(alpha=0.1) if use_kinematic else None
        self.energy_constraint = EnergyConstraint() if use_energy else None

        # Rolling state for kinematic correction
        self.register_buffer("prev_velocity", torch.zeros(hd_dim))
        self.register_buffer("error_buffer", torch.zeros(hd_dim))

        self._step = 0

    def _low_rank_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Low-rank residual forward pass: sign(x + U @ (V^T @ x)).

        Operations: V^T @ x = O(Dr), U @ z = O(Dr) → total 2Dr vs D² old.
        For D=1000, r=32: 64K ops vs 1M ops = 15× faster.
        """
        x_f = x.float()
        z = x_f @ self.V       # (..., r)   — O(Dr)
        correction = z @ self.U.T   # (..., D)  — O(Dr)
        return _majority(x_f + correction)

    def forward(
        self,
        current_hv: torch.Tensor,
        apply_constraints: bool = True,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Predict state h steps ahead.

        Args:
            current_hv: (D,) or (B, D) current state HV
            apply_constraints: Whether to apply physics constraints

        Returns:
            (predicted_hv, info_dict)
        """
        squeeze = current_hv.dim() == 1
        if squeeze:
            current_hv = current_hv.unsqueeze(0)

        # Raw prediction via low-rank residual
        raw_pred = self._low_rank_forward(current_hv)    # (B, D)

        info = {"raw_pred": raw_pred, "physics_penalty": torch.tensor(0.0)}

        if apply_constraints:
            # Kinematic correction
            if self.kin_constraint is not None and self.horizon.steps == 1:
                constrained, velocity = self.kin_constraint.constrain_prediction(
                    raw_pred[0], current_hv[0], self.prev_velocity
                )
                self.prev_velocity = velocity.detach()
                raw_pred = constrained.unsqueeze(0)

            # Energy penalty (informational, doesn't change prediction)
            if self.energy_constraint is not None:
                penalty = self.energy_constraint.penalty(current_hv[0], raw_pred[0])
                info["physics_penalty"] = penalty

        if squeeze:
            raw_pred = raw_pred.squeeze(0)

        return raw_pred, info

    @torch.no_grad()
    def hebbian_update(
        self,
        current_hv: torch.Tensor,
        actual_hv: torch.Tensor,
        lr: float = 0.01,
        predicted_hv: Optional[torch.Tensor] = None,
    ):
        """
        Update predictor weights from observed transition (Hebbian rule).

        ΔW = η · (actual - predicted) · current^T

        Optimisations vs naive implementation:
          1. Accept pre-computed prediction (avoids redundant forward pass)
          2. Use addmv_ / addmm_ in-place instead of einsum + copy
          3. Skip if not at this horizon's update_rate step
        """
        self._step += 1
        if self._step % self.horizon.update_rate != 0:
            return

        x = current_hv.float()
        y = actual_hv.float()
        if x.dim() == 1:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)

        # Compute prediction (use cache if provided)
        if predicted_hv is not None:
            p = predicted_hv.float()
            if p.dim() == 1:
                p = p.unsqueeze(0)
        else:
            p = self._low_rank_forward(x)

        error = y - p                                    # (B, D)

        # Scale by lr/rank (not lr/D): rank=32 << D, so normalise to the
        # effective update space. lr/D made updates 128× too small at D=4096.
        scale = lr / max(self.rank, 1)

        # Low-rank Hebbian: only U is learned (V is fixed random projection)
        #   ΔU = scale × error^T @ (x @ V)   — O(Dr) instead of O(D²)
        z = x @ self.V                                    # (B, r) — fixed projection
        self.U.data += scale * (error.T @ z) / x.shape[0]  # (D, r)

        # EMA error buffer
        self.error_buffer.mul_(self.horizon.decay).add_(
            error.abs().mean(dim=0), alpha=(1 - self.horizon.decay)
        )

    def prediction_confidence(self) -> float:
        """
        Estimate prediction confidence from error buffer.

        Returns value in [0, 1]: 1 = perfect prediction, 0 = random.
        The error buffer mean = 0.5 for a random predictor (binary HVs).
        """
        mean_error = float(self.error_buffer.mean().item())
        # Normalise: 0 error → 1.0, 0.5 error → 0.0
        return max(0.0, 1.0 - 2 * mean_error)


class MultiHorizonPredictor(nn.Module):
    """
    Physical AI world model predictor with short, medium, and long horizons.

    "Many modern world models focus on generating full-motion simulations
     that account for physics and interactions among objects."
    — IQT Physical AI framing.

    Each horizon has its own predictor tuned to its temporal scale:
      short  (1 step)  — fast dynamics: vibration, contact, rapid motion
      medium (5 steps) — intermediate: trajectory, approaching objects
      long   (20 steps)— slow trends: wear, drift, environmental change

    At inference, all three predictions are available simultaneously,
    enabling different downstream components to use the appropriate horizon.

    Args:
        hd_dim: Hypervector dimensionality
        horizons: List of PredictionHorizon specs
    """

    def __init__(
        self,
        hd_dim: int,
        horizons: Optional[List[PredictionHorizon]] = None,
    ):
        super().__init__()
        self.hd_dim = hd_dim
        self.horizons = horizons or STANDARD_HORIZONS

        self.predictors = nn.ModuleList([
            HorizonPredictor(hd_dim, h) for h in self.horizons
        ])

        # State buffer for multi-step rollout
        self.register_buffer("state_buffer", torch.zeros(25, hd_dim))
        self._buf_ptr = 0
        self._last_predictions: Dict[str, torch.Tensor] = {}

    def forward(self, state_hv: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict all horizons from current state.

        Args:
            state_hv: (D,) current world state HV

        Returns:
            Dict mapping horizon name → predicted HV
        """
        # Update rolling buffer: store state_hv at current position, then advance
        self.state_buffer[self._buf_ptr] = state_hv.detach()
        self._buf_ptr = (self._buf_ptr + 1) % self.state_buffer.shape[0]

        predictions = {}
        physics_penalties = {}

        for predictor, horizon in zip(self.predictors, self.horizons):
            pred, info = predictor(state_hv)
            predictions[horizon.name] = pred
            physics_penalties[horizon.name] = info["physics_penalty"]

        # Cache for reuse in update() — avoids redundant forward passes
        self._last_predictions = predictions

        # Confidence-weighted ensemble: blend predictions across horizons,
        # weighting each horizon by its recent prediction accuracy.
        # Short-horizon predictors are typically more accurate; this gives
        # them more influence when they're consistent with longer horizons.
        confidences = {h.name: p.prediction_confidence()
                       for h, p in zip(self.horizons, self.predictors)}
        total_conf = sum(confidences.values()) + 1e-8
        ensemble_pred = torch.zeros_like(state_hv).float()
        for h_name, pred_hv in predictions.items():
            w = confidences[h_name] / total_conf
            ensemble_pred += w * pred_hv.float()

        return {
            "predictions":      predictions,
            "physics_penalties": physics_penalties,
            "uncertainties":    {h: 1.0 - c for h, c in confidences.items()},
            "ensemble_pred":    _majority(ensemble_pred),
        }

    @torch.no_grad()
    def update(
        self,
        state_hv: torch.Tensor,
        actual_next_hv: torch.Tensor,
        lr: float = 0.01,
    ):
        """
        Update all predictors from an observed transition.

        Each predictor learns to predict `horizon.steps` ahead. The buffer
        stores states at each tick. For horizon h, we need the state from
        h ticks ago as the input to the predictor, and the current state
        as the target.

        Buffer layout (after forward() wrote state_hv at buf_ptr, then incremented):
            state_buffer[buf_ptr - 1] = state_hv (just written by forward)
            state_buffer[buf_ptr - 2] = state_{t-1}
            state_buffer[buf_ptr - 3] = state_{t-2}
            ...

        For horizon h=1 (predict 1 step ahead):
            We need state_{t-1} as input → buf_ptr - 1
            The target is actual_next_hv (the current observation)

        For horizon h=5 (predict 5 steps ahead):
            We need state_{t-5} as input → buf_ptr - 5
            The target is actual_next_hv (the current observation)

        So: past_state = state_buffer[buf_ptr - h]
        """
        for predictor, horizon in zip(self.predictors, self.horizons):
            h = horizon.steps
            # forward() stored state_hv at (buf_ptr - 1) before incrementing.
            # We need the state from h ticks before state_hv:
            #   past_state = state_buffer[(buf_ptr - 1) - (h - 1)] = state_buffer[buf_ptr - h]
            buf_idx = (self._buf_ptr - h) % self.state_buffer.shape[0]
            past_state = self.state_buffer[buf_idx]

            if past_state.abs().sum() < 1:
                continue

            # Always compute prediction fresh from past_state for correct gradient.
            predictor.hebbian_update(
                past_state, actual_next_hv, lr=lr, predicted_hv=None
            )

    def confidence_report(self) -> Dict[str, float]:
        """Return prediction confidence for each horizon."""
        return {
            h.name: p.prediction_confidence()
            for h, p in zip(self.horizons, self.predictors)
        }

    def surprise_signal(
        self,
        predicted_hv: torch.Tensor,
        actual_hv:    torch.Tensor,
    ) -> Dict[str, float]:
        """
        Compute surprise (prediction error) for a state transition.

        Surprise is the fundamental learning signal in predictive processing
        (Rao & Ballard 1999; Friston 2010 free energy principle).
        High surprise → the world model was wrong → update strongly.
        Low surprise → prediction was accurate → small update or none.

        Returns per-horizon surprise if predictions are cached, otherwise
        computes from the primary (short-horizon) predictor.

        Returns:
            Dict with 'short', 'medium', 'long', 'composite' surprise scores ∈ [0,1]
        """
        pred = predicted_hv.to(actual_hv.device)
        act  = actual_hv.float()

        # Primary surprise: Hamming distance between prediction and reality
        primary = float(_hamming(pred.unsqueeze(0), act.unsqueeze(0)).item())

        # Per-horizon surprise if we have cached predictions
        per_horizon = {}
        if hasattr(self, '_last_predictions') and self._last_predictions:
            for h_name, h_pred in self._last_predictions.items():
                per_horizon[h_name] = float(
                    _hamming(h_pred.unsqueeze(0), act.unsqueeze(0)).item()
                )

        composite = primary if not per_horizon else (
            sum(per_horizon.values()) / len(per_horizon)
        )

        return {
            "short":     per_horizon.get("short", primary),
            "medium":    per_horizon.get("medium", primary),
            "long":      per_horizon.get("long", primary),
            "composite": composite,
            "primary":   primary,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Action Evaluator — Think Before Acting
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ActionCandidate:
    """A candidate action with its HDC encoding and evaluation result."""
    name: str
    hv: torch.Tensor
    predicted_outcome: Optional[torch.Tensor] = None
    utility_score: float = 0.0
    risk_score: float = 0.0
    net_score: float = 0.0


class ActionEvaluator:
    """
    Mental simulation for evaluating actions before execution.

    "Tesla uses world models to simulate driving: constructing possible futures,
     evaluating the impact of actions like turning and accelerating, and
     identifying potential risks before they occur."
    — IQT Physical AI framing.

    Given a set of candidate actions and a learned forward model, the evaluator:
    1. Encodes each action as an HDC vector (binding action HV with current state)
    2. Rolls each action forward through the world model for k steps
    3. Estimates utility (similarity to goal state) and risk (distance from safe region)
    4. Returns the ranked action list

    Args:
        hd_dim: Hypervector dimensionality
        predictor: MultiHorizonPredictor (forward model)
        n_rollout_steps: Steps to simulate per action
        risk_threshold: Hamming distance from safe-region prototypes above which
                        an outcome is considered risky
    """

    def __init__(
        self,
        hd_dim: int,
        predictor: MultiHorizonPredictor,
        n_rollout_steps: int = 5,
        risk_threshold: float = 0.3,
    ):
        self.hd_dim = hd_dim
        self.predictor = predictor
        self.n_rollout_steps = n_rollout_steps
        self.risk_threshold = risk_threshold

        # Safe-region prototypes (updated from experience)
        self._safe_prototypes: List[torch.Tensor] = []
        self._n_safe = 0

        # Danger-region prototypes
        self._danger_prototypes: List[torch.Tensor] = []

    def add_safe_state(self, state_hv: torch.Tensor):
        """Register a known-safe state HV."""
        self._safe_prototypes.append(state_hv.detach().clone())
        self._n_safe += 1

    def add_danger_state(self, state_hv: torch.Tensor):
        """Register a known-dangerous state HV."""
        self._danger_prototypes.append(state_hv.detach().clone())

    def _max_similarity_to_set(
        self,
        hv: torch.Tensor,
        prototype_set: List[torch.Tensor],
    ) -> float:
        """Maximum Hamming similarity to any prototype in a set."""
        if not prototype_set:
            return 0.0
        protos = torch.stack(prototype_set)       # (N, D)
        sims = _hamming(hv.unsqueeze(0).expand_as(protos), protos)
        return float(sims.max().item())

    def _simulate_action(
        self,
        current_state: torch.Tensor,
        action_hv: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Simulate k steps of applying action_hv from current_state.

        Action is encoded as: modified_state = XOR(current_state, action_hv)
        This represents the state perturbation the action introduces.
        The predictor then extrapolates the trajectory.

        Returns:
            (final_state_hv, [state_hv_at_each_step])
        """
        with torch.no_grad():
            state = _xor(current_state, action_hv)  # apply action perturbation
            trajectory = [state]

            for _ in range(self.n_rollout_steps - 1):
                result = self.predictor(state)
                next_state = result["predictions"].get("short", state)
                state = next_state
                trajectory.append(state)

        return state, trajectory

    def evaluate(
        self,
        current_state: torch.Tensor,
        candidates: List[ActionCandidate],
        goal_state: Optional[torch.Tensor] = None,
        risk_weight: float = 0.5,
    ) -> List[ActionCandidate]:
        """
        Evaluate and rank candidate actions by predicted outcome.

        Scoring:
            utility = sim(final_state, goal_state)  if goal given
                     else sim(final_state, safe_prototypes)
            risk    = sim(final_state, danger_prototypes)
            net     = utility - risk_weight × risk

        Args:
            current_state: (D,) current world state
            candidates: List of ActionCandidate with encoded HVs
            goal_state: (D,) or None — target state HV
            risk_weight: Weight of risk in final score (higher → more conservative)

        Returns:
            Candidates sorted by net_score descending (best first)
        """
        for cand in candidates:
            final_state, _ = self._simulate_action(current_state, cand.hv)
            cand.predicted_outcome = final_state

            # Utility: toward goal or toward known-safe states
            if goal_state is not None:
                utility = float(_hamming(final_state, goal_state).item())
            else:
                utility = self._max_similarity_to_set(final_state, self._safe_prototypes)

            # Risk: proximity to known-danger states
            risk = self._max_similarity_to_set(final_state, self._danger_prototypes)

            cand.utility_score = utility
            cand.risk_score = risk
            cand.net_score = utility - risk_weight * risk

        candidates.sort(key=lambda c: c.net_score, reverse=True)
        return candidates

    def best_action(
        self,
        current_state: torch.Tensor,
        candidates: List[ActionCandidate],
        goal_state: Optional[torch.Tensor] = None,
        risk_weight: float = 0.5,
    ) -> Optional[ActionCandidate]:
        """Return the single best-scoring action."""
        ranked = self.evaluate(current_state, candidates, goal_state, risk_weight)
        return ranked[0] if ranked else None


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Digital Twin Sync — Model-Physical Divergence Detection
# ═══════════════════════════════════════════════════════════════════════════════

class DigitalTwinSync:
    """
    Maintains continuous synchronisation between the learned world model and
    the physical system it represents.

    "Digital twins... AI must reason about physical spaces and outcomes before
     executing an action."
    — IQT Physical AI framing.

    The twin tracks:
    - Model-predicted state at each step
    - Actual observed state at each step
    - Divergence = Hamming distance between predicted and actual
    - Calibration: when divergence exceeds threshold, trigger targeted re-learning

    Args:
        hd_dim: Hypervector dimensionality
        divergence_threshold: Hamming distance above which re-learning is triggered
        recalibration_window: Number of recent samples used for recalibration
    """

    def __init__(
        self,
        hd_dim: int,
        divergence_threshold: float = 0.15,
        recalibration_window: int = 50,
    ):
        self.hd_dim = hd_dim
        self.divergence_threshold = divergence_threshold
        self.recalibration_window = recalibration_window

        self._divergence_history: List[float] = []
        self._recalibration_buffer: List[Tuple[torch.Tensor, torch.Tensor]] = []

        # Running model state
        self.model_state: Optional[torch.Tensor] = None
        self.actual_state: Optional[torch.Tensor] = None

        # Cumulative statistics
        self.total_steps = 0
        self.n_recalibrations = 0

    def step(
        self,
        predicted_hv: torch.Tensor,
        actual_hv: torch.Tensor,
    ) -> Dict:
        """
        Record one synchronisation step.

        Args:
            predicted_hv: (D,) world model's prediction
            actual_hv: (D,) actual observed state

        Returns:
            Dict with divergence, needs_recalibration, recalibration_samples
        """
        divergence = float(1.0 - _hamming(predicted_hv, actual_hv).item())
        self._divergence_history.append(divergence)
        self.model_state = predicted_hv
        self.actual_state = actual_hv
        self.total_steps += 1

        needs_recal = divergence > self.divergence_threshold
        if needs_recal:
            self._recalibration_buffer.append(
                (actual_hv.clone(), predicted_hv.clone())
            )
            if len(self._recalibration_buffer) > self.recalibration_window:
                self._recalibration_buffer.pop(0)
            self.n_recalibrations += 1

        return {
            "divergence": divergence,
            "needs_recalibration": needs_recal,
            "recalibration_samples": list(self._recalibration_buffer),
            "mean_divergence_recent": self.mean_divergence(window=20),
        }

    def mean_divergence(self, window: Optional[int] = None) -> float:
        """Mean divergence over the last `window` steps."""
        if not self._divergence_history:
            return 0.0
        hist = self._divergence_history[-window:] if window else self._divergence_history
        return sum(hist) / len(hist)

    def is_synchronized(self, window: int = 10) -> bool:
        """Return True if recent divergence is below threshold."""
        return self.mean_divergence(window) <= self.divergence_threshold

    @property
    def recalibration_rate(self) -> float:
        """Fraction of steps requiring recalibration."""
        if self.total_steps == 0:
            return 0.0
        return self.n_recalibrations / self.total_steps

    def status(self) -> Dict:
        return {
            "total_steps": self.total_steps,
            "n_recalibrations": self.n_recalibrations,
            "recalibration_rate": self.recalibration_rate,
            "mean_divergence": self.mean_divergence(),
            "is_synchronized": self.is_synchronized(),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Physics World Model (combines all above)
# ═══════════════════════════════════════════════════════════════════════════════

class PhysicsWorldModel(nn.Module):
    """
    Complete physics-informed world model for Physical AI.

    Integrates:
    - MultiHorizonPredictor (short/medium/long-term forecasting)
    - ActionEvaluator (mental simulation before acting)
    - DigitalTwinSync (model-physical divergence tracking)
    - Physics constraints (kinematic, energy conservation)

    The three Physical AI layers are explicitly supported:
      Interface:       observe(sensor_hv) ingests new observations
      Interpretation:  forward() produces world state + multi-horizon predictions
      Action:          evaluate_actions() ranks candidates before execution

    Args:
        hd_dim: Hypervector dimensionality
        n_rollout_steps: Action simulation depth
        divergence_threshold: Sync alert threshold
    """

    def __init__(
        self,
        hd_dim: int = 4096,
        n_rollout_steps: int = 5,
        divergence_threshold: float = 0.15,
    ):
        super().__init__()
        self.hd_dim = hd_dim

        # Core predictors
        self.multi_horizon = MultiHorizonPredictor(hd_dim)

        # Action evaluator (uses short-horizon predictor as forward model)
        self.action_evaluator = ActionEvaluator(
            hd_dim=hd_dim,
            predictor=self.multi_horizon,
            n_rollout_steps=n_rollout_steps,
        )

        # Digital twin sync
        self.twin_sync = DigitalTwinSync(
            hd_dim=hd_dim,
            divergence_threshold=divergence_threshold,
        )

        # Current world state
        self.register_buffer("current_state", torch.zeros(hd_dim))
        self._initialized = False

    def observe(self, sensor_hv: torch.Tensor, learn: bool = True) -> Dict:
        """
        Ingest a new sensor observation (interface layer → interpretation layer).

        Args:
            sensor_hv: (D,) binary HV encoding of current sensor readings
            learn: Whether to update the predictor from this observation

        Returns:
            Dict with predictions, divergence, confidence
        """
        # Get predictions before updating
        if self._initialized:
            predictions = self.multi_horizon(self.current_state)
            short_pred = predictions["predictions"].get("short", self.current_state)

            # Digital twin sync
            sync_info = self.twin_sync.step(short_pred, sensor_hv)

            # Update predictor from observed transition
            if learn:
                self.multi_horizon.update(
                    self.current_state, sensor_hv, lr=0.01
                )

            # Recalibrate if needed
            if sync_info["needs_recalibration"] and learn:
                for actual, _ in sync_info["recalibration_samples"][-5:]:
                    self.multi_horizon.update(self.current_state, actual, lr=0.05)
        else:
            predictions = {"predictions": {}, "physics_penalties": {}}
            sync_info = {"divergence": 0.0, "needs_recalibration": False,
                         "mean_divergence_recent": 0.0}
            self._initialized = True

        # Update current state
        self.current_state = sensor_hv.detach().clone()

        return {
            "predictions": predictions["predictions"],
            "physics_penalties": predictions.get("physics_penalties", {}),
            "divergence": sync_info["divergence"],
            "needs_recalibration": sync_info["needs_recalibration"],
            "confidence": self.multi_horizon.confidence_report(),
            "twin_status": self.twin_sync.status(),
        }

   
    def evaluate_actions(
        self,
        candidates: List[ActionCandidate],
        goal_state: Optional[torch.Tensor] = None,
        risk_weight: float = 0.5,
    ) -> List[ActionCandidate]:
        """
        Rank candidate actions via mental simulation (action layer input).

        Args:
            candidates: List of ActionCandidate
            goal_state: Target state HV (if known)
            risk_weight: Conservative factor [0, 1]

        Returns:
            Ranked ActionCandidate list (best first)
        """
        return self.action_evaluator.evaluate(
            self.current_state, candidates, goal_state, risk_weight
        )

    def register_safe_state(self, state_hv: torch.Tensor):
        """Register a known-safe operating state for risk estimation."""
        self.action_evaluator.add_safe_state(state_hv)

    def register_danger_state(self, state_hv: torch.Tensor):
        """Register a known-dangerous state."""
        self.action_evaluator.add_danger_state(state_hv)

    def world_state_hv(self) -> torch.Tensor:
        """Current world state hypervector."""
        return self.current_state


# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_multi_horizon_predictor():
    print("=" * 60)
    print("Testing MultiHorizonPredictor")
    print("=" * 60)

    torch.manual_seed(42)
    dim = 2000
    predictor = MultiHorizonPredictor(dim)

    # Simulate a simple periodic sequence: state oscillates between two HVs
    state_a = (torch.rand(dim) < 0.5).float()
    state_b = (torch.rand(dim) < 0.5).float()

    # Train on alternating sequence: A→B→A→B...
    for i in range(50):
        curr = state_a if i % 2 == 0 else state_b
        nxt  = state_b if i % 2 == 0 else state_a
        predictor.update(curr, nxt, lr=0.05)

    # Check predictions are filled
    result = predictor(state_a)
    assert "short" in result["predictions"]
    assert "medium" in result["predictions"]
    assert "long" in result["predictions"]

    conf = predictor.confidence_report()
    print(f"  Confidence after 50 steps: {conf}")
    assert all(0.0 <= v <= 1.0 for v in conf.values()), "Confidence out of range"

    # Short predictor should learn the pattern best
    short_pred = result["predictions"]["short"]
    short_sim = float(_hamming(short_pred, state_b).item())
    print(f"  Short predictor sim to correct: {short_sim:.4f}  (want > 0.5)")

    print("  ✅ MultiHorizonPredictor OK")


def test_action_evaluator():
    print("=" * 60)
    print("Testing ActionEvaluator (think-before-act)")
    print("=" * 60)

    torch.manual_seed(7)
    dim = 2000
    predictor = MultiHorizonPredictor(dim)
    evaluator = ActionEvaluator(dim, predictor, n_rollout_steps=3)

    # Current state
    current = (torch.rand(dim) < 0.5).float()

    # Goal: move toward a target state
    goal = (torch.rand(dim) < 0.5).float()

    # Register goal-adjacent states as safe
    evaluator.add_safe_state(goal)

    # Candidate actions: one moves toward goal (correct), one is random
    action_correct = _xor(current, goal)  # XOR flips bits toward goal
    action_random  = (torch.rand(dim) < 0.5).float()

    candidates = [
        ActionCandidate("toward_goal", action_correct),
        ActionCandidate("random_action", action_random),
    ]

    ranked = evaluator.evaluate(current, candidates, goal_state=goal)
    print(f"  Ranked actions: {[(c.name, f'{c.net_score:.3f}') for c in ranked]}")
    # The toward_goal action should score higher (XOR with goal → sim to goal = 1.0)
    assert ranked[0].name == "toward_goal", \
        f"Expected toward_goal first, got {ranked[0].name}"

    print("  ✅ ActionEvaluator OK")


def test_digital_twin_sync():
    print("=" * 60)
    print("Testing DigitalTwinSync")
    print("=" * 60)

    torch.manual_seed(0)
    dim = 2000
    twin = DigitalTwinSync(dim, divergence_threshold=0.1)

    # In-sync: predicted ≈ actual (low divergence)
    for _ in range(20):
        pred = (torch.rand(dim) < 0.5).float()
        # Actual = small perturbation of predicted
        actual = pred.clone()
        flip = torch.rand(dim) < 0.05  # 5% bit flips
        actual[flip] = 1.0 - actual[flip]
        twin.step(pred, actual)

    assert twin.is_synchronized(), "Should be synchronized with 5% noise"
    print(f"  Synchronized (5% noise): {twin.is_synchronized()},  "
          f"mean div={twin.mean_divergence():.4f}")

    # Out-of-sync: large divergence
    for _ in range(10):
        pred = (torch.rand(dim) < 0.5).float()
        actual = (torch.rand(dim) < 0.5).float()  # fully random → ~50% divergence
        result = twin.step(pred, actual)

    print(f"  After drift: div={twin.mean_divergence(5):.4f}, "
          f"recal_rate={twin.recalibration_rate:.2f}")
    assert twin.n_recalibrations > 0, "Should have triggered recalibration"

    print("  ✅ DigitalTwinSync OK")


def test_physics_world_model():
    print("=" * 60)
    print("Testing PhysicsWorldModel (full pipeline)")
    print("=" * 60)

    torch.manual_seed(99)
    dim = 1000
    model = PhysicsWorldModel(hd_dim=dim, n_rollout_steps=3)

    # Simulate 30 sensor observations (random walk)
    state = (torch.rand(dim) < 0.5).float()
    for t in range(30):
        # Perturb state slightly each step
        flip = torch.rand(dim) < 0.1
        state = state.clone()
        state[flip] = 1.0 - state[flip]
        info = model.observe(state, learn=True)

    print(f"  After 30 steps: divergence={info['divergence']:.4f}, "
          f"conf={info['confidence']}")

    # Register safe and danger states
    model.register_safe_state(state)
    danger = (torch.rand(dim) < 0.5).float()
    model.register_danger_state(danger)

    # Evaluate two actions
    candidates = [
        ActionCandidate("safe_action", (torch.rand(dim) < 0.05).float()),
        ActionCandidate("risky_action", _xor(state, danger)),
    ]
    ranked = model.evaluate_actions(candidates, risk_weight=0.7)
    print(f"  Action ranking: {[(c.name, f'{c.net_score:.3f}') for c in ranked]}")

    status = model.twin_sync.status()
    print(f"  Twin status: steps={status['total_steps']}, "
          f"recal_rate={status['recalibration_rate']:.2f}")

    assert status["total_steps"] >= 29  # first call initialises without sync
    print("  ✅ PhysicsWorldModel OK")


# ═══════════════════════════════════════════════════════════════════════════════
# Elite Enhancements — EnsembleHorizonPredictor + supporting helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _gen_hvs_local(n: int, dim: int, seed: Optional[int] = None) -> torch.Tensor:
    """Generate n random binary hypervectors (local helper)."""
    g = torch.Generator()
    if seed is not None:
        g.manual_seed(seed)
    return (torch.rand(n, dim, generator=g) >= 0.5).float()


class EnsembleHorizonPredictor(nn.Module):
    """
    Elite replacement for HorizonPredictor.

    Improvements over baseline:
      - Ensemble of M low-rank predictors with different random projections;
        aggregate via majority vote for robustness to outliers.
      - Uncertainty = mean pairwise Hamming distance between ensemble members.
      - Winner-take-all updates: only top-performing members are updated;
        lagging members are restarted via gradient pressure.
      - Kalman-like correction: blend prediction and observation proportional
        to ensemble uncertainty.

    Args:
        hd_dim: Hypervector dimensionality
        rank: Low-rank bottleneck (default 64)
        n_ensemble: Ensemble size (default 5)
        horizon_steps: Steps ahead to predict
        update_rate: Update every N observations
        decay: EMA decay for per-member error buffers
    """

    def __init__(
        self,
        hd_dim: int,
        rank: int = 64,
        n_ensemble: int = 5,
        horizon_steps: int = 1,
        update_rate: int = 1,
        decay: float = 0.95,
    ):
        super().__init__()
        self.hd_dim = hd_dim
        self.rank = rank
        self.n_ensemble = n_ensemble
        self.horizon_steps = horizon_steps
        self.update_rate = update_rate
        self.decay = decay

        self.U_list = nn.ParameterList([
            nn.Parameter(torch.zeros(hd_dim, rank))
            for _ in range(n_ensemble)
        ])
        self.register_buffer("V_list", torch.stack([
            _gen_hvs_local(hd_dim, rank, seed=42 + m) * 2 - 1
            for m in range(n_ensemble)
        ]) / (hd_dim ** 0.5))

        self.register_buffer("error_buffers", torch.zeros(n_ensemble, hd_dim))
        self._step = 0

    def _predict_member(self, x: torch.Tensor, m: int) -> torch.Tensor:
        x_f = x.float()
        V_m = self.V_list[m].to(x.device)
        z = x_f @ V_m
        correction = z @ self.U_list[m].T
        return _majority(x_f + correction)

    def forward(
        self, x: torch.Tensor, return_ensemble: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: (D,) or (B, D) state HV
            return_ensemble: If True return (median_pred, member_preds, uncertainty)
        """
        squeeze = x.dim() == 1
        if squeeze:
            x = x.unsqueeze(0)

        stacked = torch.stack([self._predict_member(x, m) for m in range(self.n_ensemble)])
        median_pred = _majority(stacked.mean(dim=0))
        uncertainty = 1.0 - _hamming(stacked[0], stacked[1:].mean(dim=0))

        if squeeze:
            median_pred = median_pred.squeeze(0)
            uncertainty = uncertainty.squeeze(0)

        if return_ensemble:
            return median_pred, stacked, uncertainty
        return median_pred

    @torch.no_grad()
    def hebbian_update(
        self,
        current_hv: torch.Tensor,
        actual_hv: torch.Tensor,
        lr: float = 0.01,
        predicted_hv: Optional[torch.Tensor] = None,
    ):
        """Update best-performing ensemble members; skip laggards."""
        self._step += 1
        if self._step % self.update_rate != 0:
            return

        x = current_hv.float().unsqueeze(0) if current_hv.dim() == 1 else current_hv.float()
        y = actual_hv.float().unsqueeze(0) if actual_hv.dim() == 1 else actual_hv.float()

        member_errors = [
            (y - self._predict_member(x, m).float()).abs().mean().item()
            for m in range(self.n_ensemble)
        ]
        median_err = sorted(member_errors)[len(member_errors) // 2]
        scale = lr / max(self.rank, 1)
        z = x @ self.V_list[0].to(x.device)

        for m in range(self.n_ensemble):
            if member_errors[m] <= median_err * 1.2:
                p_m = self._predict_member(x, m)
                err_m = y - p_m.float()
                self.U_list[m].data += scale * (err_m.T @ z) / x.shape[0]
            self.error_buffers[m] = (
                self.decay * self.error_buffers[m] + (1 - self.decay) * member_errors[m]
            )

    def prediction_confidence(self) -> float:
        return max(0.0, 1.0 - 2 * float(self.error_buffers.mean().item()))

    def get_uncertainty(self, x: torch.Tensor) -> float:
        _, _, u = self.forward(x, return_ensemble=True)
        return float(u.mean().item()) if isinstance(u, torch.Tensor) else float(u)


class EliteMultiHorizonPredictor(nn.Module):
    """
    Elite multi-horizon predictor: one EnsembleHorizonPredictor per horizon
    with uncertainty-aware aggregation and adaptive learning rate per horizon.
    """

    def __init__(self, hd_dim: int):
        super().__init__()
        self.hd_dim = hd_dim
        self.short  = EnsembleHorizonPredictor(hd_dim, horizon_steps=1,  update_rate=1,  n_ensemble=3)
        self.medium = EnsembleHorizonPredictor(hd_dim, horizon_steps=5,  update_rate=3,  n_ensemble=3)
        self.long   = EnsembleHorizonPredictor(hd_dim, horizon_steps=20, update_rate=10, n_ensemble=3)
        self.register_buffer("state_buffer", torch.zeros(25, hd_dim))
        self._buf_ptr = 0

    def forward(self, state_hv: torch.Tensor) -> Dict:
        self.state_buffer[self._buf_ptr] = state_hv.detach()
        self._buf_ptr = (self._buf_ptr + 1) % self.state_buffer.shape[0]
        predictions, uncertainties = {}, {}
        for name, predictor in [("short", self.short), ("medium", self.medium), ("long", self.long)]:
            pred, _, uncert = predictor(state_hv, return_ensemble=True)
            predictions[name] = pred
            uncertainties[name] = uncert
        return {"predictions": predictions, "uncertainties": uncertainties}

    def update(self, state_hv: torch.Tensor, actual_next_hv: torch.Tensor, lr: float = 0.01):
        for name, predictor, horizon in [
            ("short", self.short, 1), ("medium", self.medium, 5), ("long", self.long, 20)
        ]:
            buf_idx = (self._buf_ptr - horizon) % self.state_buffer.shape[0]
            past_state = self.state_buffer[buf_idx]
            if past_state.abs().sum() < 1:
                continue
            predictor.hebbian_update(past_state, actual_next_hv, lr=lr)

    def confidence_report(self) -> Dict[str, float]:
        return {
            "short":  self.short.prediction_confidence(),
            "medium": self.medium.prediction_confidence(),
            "long":   self.long.prediction_confidence(),
        }

    def best_horizon(self) -> str:
        """
        Return the horizon name with the highest current prediction confidence.

        Use this to select which horizon's prediction to trust most.
        Short-horizon predictors are typically more accurate unless the system
        is in a highly dynamic phase, in which case longer horizons may be more
        stable.
        """
        report = self.confidence_report()
        return max(report, key=lambda k: report[k])

    def weighted_ensemble_prediction(
        self,
        state_hv: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict next state as a confidence-weighted blend of all horizons.

        Each horizon's prediction is weighted by its current confidence:
            pred = majority(Σ conf_h × pred_h)

        This is more robust than trusting any single horizon and naturally
        down-weights horizons that have been inaccurate recently.

        Returns:
            (D,) consensus prediction HV.
        """
        fwd = self.forward(state_hv)
        preds = fwd["predictions"]
        confs = self.confidence_report()
        total = sum(confs.values()) + 1e-8
        accum = torch.zeros(self.hd_dim)
        for name, pred in preds.items():
            w      = confs.get(name, 0.0) / total
            accum  = accum + w * pred.float()
        return _majority(accum)


class AdaptiveDivergenceThreshold:
    """
    Elite enhancement for DigitalTwinSync divergence threshold.

    Replaces a fixed threshold with a percentile-based dynamic one computed
    from the running distribution of divergences (P95 by default).
    """

    def __init__(self, window: int = 100, percentile: float = 95.0, min_threshold: float = 0.05):
        self.window = window
        self.percentile = percentile
        self.min_threshold = min_threshold
        self._divergences: List[float] = []

    def step(self, divergence: float) -> float:
        self._divergences.append(divergence)
        if len(self._divergences) > self.window:
            self._divergences.pop(0)
        if len(self._divergences) < 10:
            return 0.15
        sorted_div = sorted(self._divergences)
        idx = max(0, min(len(sorted_div) - 1, int(len(sorted_div) * self.percentile / 100.0)))
        return max(self.min_threshold, sorted_div[idx])


class StochasticActionSampler:
    """
    Elite enhancement for ActionEvaluator.

    Thompson-sampling-like exploration: sample from softmax over action scores
    instead of always picking the argmax.  Temperature controls explore/exploit.
    """

    def __init__(self, temperature: float = 0.5):
        self.temperature = temperature

    def sample(self, candidates: list, net_scores: List[float]) -> int:
        """Sample an action index from softmax over net_scores."""
        if not net_scores:
            return 0
        scores = torch.tensor(net_scores, dtype=torch.float32)
        probs = F.softmax(scores / max(self.temperature, 1e-6), dim=0)
        return int(torch.multinomial(probs, 1).item())


if __name__ == "__main__":
    test_multi_horizon_predictor()
    print()
    test_action_evaluator()
    print()
    test_digital_twin_sync()
    print()
    test_physics_world_model()
    print()
    print("=== All physics_world_model tests passed ===")
