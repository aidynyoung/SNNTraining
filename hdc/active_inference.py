"""
hdc/active_inference.py
========================
Active Inference and Free Energy Principle in HDC Space
========================================================
Reference:
    Friston, K. (2010) "The free-energy principle: a unified brain theory?"
    Nature Reviews Neuroscience 11(2):127–138.

    Friston et al. (2017) "Active inference: a process theory"
    Neural Computation 29(1):1–49.

    Parr & Friston (2019) "Generalised free energy and active inference"
    Biological Cybernetics 113(5–6):495–513.

    Millidge, Seth, Buckley (2022) "Predictive Coding: a Theoretical and
    Experimental Review" arXiv:2107.12979.

Why Active Inference is the right theoretical grounding for SNNTraining:

    Standard reinforcement learning: R(s, a) → V(s) via Bellman backup
    Active inference: F(o, μ) → action that minimises expected free energy
                      where F = -log p(o|μ) + KL(q(μ) || p(μ))

    The free energy F has two terms:
        Accuracy:   -log p(o|μ) = prediction error (how wrong am I about observations?)
        Complexity: KL(q(μ) || p(μ)) = how much did I update my beliefs?

    Minimising F = good predictions (accuracy) + minimal belief revision (Occam's razor)

    Active inference unifies:
        - Perception: minimise F by updating beliefs μ
        - Action:     minimise expected F by choosing actions that lead to
                      states where F will be small (preferred states)
        - Learning:   minimise F averaged over time by updating the generative model

    HDC implementation:
        - Beliefs μ: encoded as HVs (the "internal model")
        - Observations o: encoded as HVs (sensor readings)
        - Prediction error: Hamming distance between predicted and actual HV
        - Expected free energy: estimated via ensemble predictor uncertainty
        - Action selection: choose action that minimises expected future Hamming dist

This module implements:

1. FreeEnergyEstimator
   — Computes free energy F from observed and predicted HVs
   — Accuracy term: Hamming distance between predicted and observed
   — Complexity term: Hamming distance between current and prior belief
   — Minimising F = improving predictions while staying close to prior

2. ActiveInferenceAgent
   — Perceives by minimising F (updates belief toward observation)
   — Acts by selecting actions that minimise expected future F
   — Learns by updating the generative model (world model) to reduce F

3. PrecisionWeightedAttention
   — Implements Friston's precision-weighting in HDC
   — High-precision channels override low-precision ones during perception
   — Precision = inverse variance = confidence in a signal

4. BeliefPropagation
   — Propagates beliefs across a generative model hierarchy
   — Bottom-up: prediction errors flow up
   — Top-down: predictions flow down
   — Equilibrium: posterior = best explanation of observations

5. ExpectedFreeEnergy
   — G = epistemic value (information gain about hidden states)
            + pragmatic value (closeness to preferred states)
   — Action selection: argmin_a G(a)
   — This is Active Inference's replacement for Q-learning
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from hdc.physics_world_model import _hamming, _majority, _xor


# ── Utilities ──────────────────────────────────────────────────────────────────

def _gen_hv(dim: int, seed: Optional[int] = None, device: str = "cpu") -> torch.Tensor:
    g = torch.Generator(device=device)
    if seed is not None:
        g.manual_seed(seed)
    return (torch.rand(dim, generator=g, device=device) >= 0.5).float()

def _kl_hamming(q: torch.Tensor, p: torch.Tensor) -> float:
    """
    Approximate KL divergence between two binary HVs treated as Bernoulli distributions.
    KL(q||p) = q·log(q/p) + (1-q)·log((1-q)/(1-p))
    """
    q_f = q.float().clamp(0.01, 0.99)
    p_f = p.float().clamp(0.01, 0.99)
    kl  = q_f * (q_f / p_f).log() + (1 - q_f) * ((1 - q_f) / (1 - p_f)).log()
    return float(kl.mean().item())


# ═══════════════════════════════════════════════════════════════════════════════
# 1. FreeEnergyEstimator
# ═══════════════════════════════════════════════════════════════════════════════

class FreeEnergyEstimator:
    """
    Computes variational free energy F from HDC representations.

    F = -log p(o|μ) + KL(q(μ) || p(μ))
      ≈ Hamming(predicted_o, observed_o) + Hamming(current_μ, prior_μ)

    where:
        μ       = current belief state (HV)
        o       = observation (HV)
        p(o|μ)  = generative model (predicted observation given belief)
        KL term = complexity penalty (how much we deviated from prior)

    Minimising F drives:
        - Accurate predictions: reduce Hamming(predicted, observed)
        - Parsimonious updates: stay close to prior belief

    Args:
        dim:              HV dimension
        complexity_weight: λ controls accuracy vs complexity tradeoff (default 0.5)
        device:           torch device
    """

    def __init__(self, dim: int, complexity_weight: float = 0.5, device: str = "cpu"):
        self.dim               = dim
        self.complexity_weight = complexity_weight
        self.device            = device

        # Prior belief: uniform random HV (maximum entropy prior)
        self._prior = _gen_hv(dim, seed=314159, device=device)
        self._F_history: List[float] = []

    def update_prior(self, new_prior: torch.Tensor):
        """Update prior from accumulated experience."""
        self._prior = new_prior.float().to(self.device)

    def accuracy(self, predicted_obs: torch.Tensor, actual_obs: torch.Tensor) -> float:
        """
        Accuracy term: -log p(o|μ) ≈ Hamming(predicted_o, actual_o).

        Lower is better (good prediction = low Hamming distance).
        """
        return 1.0 - float(_hamming(
            predicted_obs.unsqueeze(0), actual_obs.unsqueeze(0)
        ).item())

    def complexity(self, current_belief: torch.Tensor) -> float:
        """
        Complexity term: KL(q(μ) || p(μ)) ≈ Hamming(μ, μ_prior).

        Lower = smaller deviation from prior = Occam's razor.
        """
        return 1.0 - float(_hamming(
            current_belief.unsqueeze(0), self._prior.unsqueeze(0)
        ).item())

    def free_energy(
        self,
        predicted_obs: torch.Tensor,
        actual_obs:    torch.Tensor,
        current_belief: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Compute full variational free energy.

        Returns:
            Dict with 'accuracy', 'complexity', 'free_energy'
        """
        acc = self.accuracy(predicted_obs, actual_obs)
        cmp = self.complexity(current_belief) if current_belief is not None else 0.0
        F   = acc + self.complexity_weight * cmp
        self._F_history.append(F)
        return {
            "accuracy":    acc,
            "complexity":  cmp,
            "free_energy": F,
        }

    def surprise(self, actual_obs: torch.Tensor, prior_pred: torch.Tensor) -> float:
        """
        Surprisal: -log p(o) = how surprising was this observation?
        Approximated as Hamming(actual_obs, prior_prediction).
        """
        return 1.0 - float(_hamming(actual_obs.unsqueeze(0), prior_pred.unsqueeze(0)).item())

    def average_F(self, window: int = 50) -> float:
        """Running average of free energy (lower = model is improving)."""
        h = self._F_history[-window:]
        return sum(h) / max(len(h), 1)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. ActiveInferenceAgent
# ═══════════════════════════════════════════════════════════════════════════════

class ActiveInferenceAgent:
    """
    Active Inference agent that minimises free energy through perception and action.

    Perceive:  μ ← μ + ε × (observation - prediction)   [gradient descent on F]
    Act:       a* = argmin_a G(a)   [expected free energy minimisation]
    Learn:     update generative model to reduce F averaged over time

    In HDC:
        - Belief μ is a HV that encodes the current internal state
        - Perception updates μ toward the observed HV via bundle
        - Action selection: evaluate expected F for each candidate action,
          pick the action that leads to lowest expected F
        - Learning: update the prediction model (world model) to reduce F

    This is the principled alternative to reward maximisation:
        RL asks: "What actions maximise reward?"
        AIF asks: "What actions minimise surprise about the future?"

    Args:
        dim:             HV dimension
        generative_model: Callable(belief_hv, action_hv) → predicted_next_obs_hv
        preferred_obs:   (D,) HV encoding preferred/desired states (replaces reward)
        precision:       Precision (confidence) of the sensory channel [0,1]
        device:          torch device
    """

    def __init__(
        self,
        dim:              int,
        generative_model: Optional[Callable] = None,
        preferred_obs:    Optional[torch.Tensor] = None,
        precision:        float = 0.8,
        device:           str   = "cpu",
    ):
        self.dim       = dim
        self.model     = generative_model
        self.precision = precision
        self.device    = device

        # Internal belief state (prior: uniform random)
        self.belief = _gen_hv(dim, seed=0, device=device)

        # Preferred states (desired observations)
        self.preferred = (preferred_obs.float().to(device)
                          if preferred_obs is not None
                          else _gen_hv(dim, seed=42, device=device))

        self.fe_estimator = FreeEnergyEstimator(dim, device=device)
        self._step        = 0
        self._F_log: List[Dict[str, float]] = []

        # Adaptive precision: precision increases when F is consistently low
        # (model is reliable), decreases when F is high (model is surprised).
        # This implements the meta-Bayesian principle: confidence in the
        # sensory channel should track the model's prediction accuracy.
        self._precision_ema  = precision         # running estimate of reliability
        self._precision_tau  = 50.0              # time constant for adaptation

        # Habit formation: running Q-values per action hash → bias toward
        # actions that have reduced F in the past (exploitation prior).
        self._action_q: Dict[int, float] = {}    # action_hash → Q-value
        self._habit_strength = 0.3               # weight of habit vs G estimate
        self._habit_decay    = 0.95              # EMA decay for Q-values

        # Novelty memory: set of visited state hashes; actions leading to
        # already-visited states receive lower epistemic bonus.
        self._visited:   set  = set()
        self._visit_cap: int  = 512              # max states tracked

    # ── Perception ────────────────────────────────────────────────────────────

    def perceive(self, observation: torch.Tensor, lr: float = 0.1) -> Dict[str, float]:
        """
        Update belief to minimise free energy w.r.t. current observation.

        belief_new = precision × observation + (1 - precision) × prediction
        This is the HDC equivalent of Kalman filter update.

        Args:
            observation: (D,) observed HV
            lr:          Belief update learning rate

        Returns:
            Free energy components
        """
        self._step += 1
        obs = observation.float().to(self.device)

        # Prediction from current belief (via generative model or identity)
        predicted_obs = self._predict(self.belief)

        # Compute free energy
        fe = self.fe_estimator.free_energy(predicted_obs, obs, self.belief)
        self._F_log.append(fe)

        # ── Adaptive precision: track model reliability ──────────────────────
        # Higher accuracy → higher precision (trust the senses more)
        acc = fe["accuracy"]
        self._precision_ema = (
            (1 - 1.0 / self._precision_tau) * self._precision_ema
            + (1.0 / self._precision_tau) * acc
        )
        effective_precision = max(0.1, min(0.95, self._precision_ema))

        # Update belief: blend toward observation proportional to precision
        # High precision = strongly update toward observation (reliable sensor)
        # Low precision = rely more on prior / prediction (uncertain sensor)
        belief_update = _majority(
            effective_precision * obs + (1 - effective_precision) * predicted_obs
        )
        self.belief = _majority(
            (1 - lr) * self.belief + lr * belief_update
        )

        # Track visited states for novelty computation
        state_hash = int(self.belief[:32].cpu().numpy().tobytes().__hash__() % (2**31))
        if len(self._visited) < self._visit_cap:
            self._visited.add(state_hash)
        self._state_hash = state_hash

        return fe

    def _predict(self, belief: torch.Tensor) -> torch.Tensor:
        """Predict next observation from current belief via generative model."""
        if self.model is not None:
            try:
                return self.model(belief)
            except Exception:
                pass
        return belief.clone()   # identity model: predict no change

    # ── Action ────────────────────────────────────────────────────────────────

    def select_action(
        self,
        candidate_actions: List[torch.Tensor],
        horizon: int = 3,
    ) -> Tuple[int, float, List[float]]:
        """
        Select action that minimises Expected Free Energy G.

        G(a) = epistemic_value(a) + pragmatic_value(a)

        epistemic_value:  information gain about hidden states (exploration)
                          ≈ Hamming uncertainty of predicted next state
        pragmatic_value:  closeness to preferred states (exploitation)
                          ≈ -Hamming(predicted_next, preferred)

        Args:
            candidate_actions: List of (D,) action HVs
            horizon:           Look-ahead steps for G estimation

        Returns:
            (best_action_idx, best_G, all_G_values)
        """
        if not candidate_actions:
            return 0, 0.0, []

        G_values = []
        for action in candidate_actions:
            G = self._expected_free_energy(action, horizon)
            G_values.append(G)

        # Habit bias: blend G with negative Q-value (lower G = better)
        if self._action_q:
            biased = []
            for i, (G, action) in enumerate(zip(G_values, candidate_actions)):
                a_hash = int(action[:8].cpu().numpy().tobytes().__hash__() % (2**31))
                q = self._action_q.get(a_hash, 0.0)
                biased.append(G - self._habit_strength * q)
            best_idx = int(min(range(len(biased)), key=lambda i: biased[i]))
        else:
            best_idx = int(min(range(len(G_values)), key=lambda i: G_values[i]))

        return best_idx, G_values[best_idx], G_values

    def update_habits(self, action: torch.Tensor, reward_signal: float):
        """
        Update habit Q-value for an action based on observed reward.

        Call after each action-outcome observation with:
            reward_signal = 1 - prediction_error  (low error = good outcome)

        The habit forms a prior toward actions that historically reduced F.

        Args:
            action:        (D,) action HV that was taken
            reward_signal: Scalar in [0,1]; higher = action led to lower F
        """
        a_hash = int(action[:8].cpu().numpy().tobytes().__hash__() % (2**31))
        old_q  = self._action_q.get(a_hash, 0.0)
        self._action_q[a_hash] = (
            self._habit_decay * old_q + (1 - self._habit_decay) * reward_signal
        )

    def _expected_free_energy(self, action: torch.Tensor, horizon: int) -> float:
        """
        Estimate expected free energy G for a given action.

        G = pragmatic_cost(a) − novelty_bonus(a)

        pragmatic_cost: Hamming distance from predicted future to preferred state
                        (low = action leads toward goal)
        novelty_bonus:  1 - is_visited(predicted_state)
                        (high = action leads to unexplored territory)

        Combined: G = Hamming(predicted, preferred) − 0.3 × novelty
        The novelty bonus decays naturally as states get visited — the agent
        transitions from exploration to exploitation automatically.
        """
        # Simulate future belief after taking this action
        simulated_belief = self.belief.clone()
        for _ in range(horizon):
            if self.model is not None:
                try:
                    next_obs = self.model(simulated_belief, action)
                except TypeError:
                    next_obs = self.model(simulated_belief)
            else:
                next_obs = _majority(_xor(simulated_belief, action).float())
            simulated_belief = next_obs

        # Pragmatic value: how close is the simulated future to preferred state?
        pragmatic = float(_hamming(
            simulated_belief.unsqueeze(0), self.preferred.unsqueeze(0)
        ).item())

        # Novelty-aware epistemic bonus
        # If the predicted state hash is already visited → low novelty → less bonus
        pred_hash = int(simulated_belief[:32].cpu().numpy().tobytes().__hash__() % (2**31))
        novelty   = 0.0 if pred_hash in self._visited else 1.0

        # Information gain proxy: state entropy change (if state is very certain → less gain)
        density    = float(simulated_belief.mean().item())
        info_gain  = 4.0 * density * (1.0 - density)   # ∈ [0,1], peak at 0.5

        epistemic  = 0.7 * novelty + 0.3 * info_gain   # blend novelty + entropy

        # G = pragmatic distance - exploration bonus
        return pragmatic - 0.3 * epistemic

    # ── Learning ─────────────────────────────────────────────────────────────

    def update_preferred(self, new_preferred: torch.Tensor):
        """Update preferred states (task/goal change)."""
        self.preferred = new_preferred.float().to(self.device)

    def update_model(self, generative_model: Callable):
        """Update the generative model."""
        self.model = generative_model

    # ── Diagnostics ──────────────────────────────────────────────────────────

    def free_energy_report(self) -> Dict[str, float]:
        """Return current free energy statistics."""
        if not self._F_log:
            return {}
        recent = self._F_log[-20:]
        avg_F  = sum(r["free_energy"] for r in recent) / len(recent)
        avg_acc = sum(r["accuracy"] for r in recent) / len(recent)
        return {
            "avg_free_energy":  avg_F,
            "avg_accuracy":     avg_acc,
            "n_steps":          self._step,
            "belief_density":   float(self.belief.mean().item()),
        }

    def goal_proximity(self, goal_hv: torch.Tensor) -> float:
        """
        Measure how close the current belief is to a goal state.

        Returns Hamming similarity ∈ [0, 1] between current belief (binarised)
        and the goal HV.  Tracks progress toward a target state.

        Args:
            goal_hv: (D,) goal state HV

        Returns:
            Similarity ∈ [0, 1]; 1 = goal reached.
        """
        belief_bin = (self.belief > 0.5).float()
        return float(_hamming(
            belief_bin.unsqueeze(0), goal_hv.float().to(self.device).unsqueeze(0)
        ).item())

    def agent_summary(self) -> Dict:
        """
        One-call summary of the active inference agent's current state.

        Returns:
            Dict with F (free energy), accuracy, action_entropy, step, belief_density.
        """
        fr = self.free_energy_report()
        H_actions = float(-(
            self._action_q + 1e-10
        ).log().mean().item()) if self._action_q else 0.0

        return {
            **fr,
            "habit_entropy":  round(H_actions, 4),
            "n_habits":       len(self._action_q),
            "n_visited":      len(self._visited),
            "is_exploring":   H_actions > 1.0,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 3. PrecisionWeightedAttention
# ═══════════════════════════════════════════════════════════════════════════════

class PrecisionWeightedAttention:
    """
    Friston's precision-weighted sensory integration in HDC.

    Reference:
        Friston (2009) "Predictive coding under the free-energy principle"
        Philosophical Transactions B 364(1521):1211–1221.

        Feldman & Friston (2010) "Attention, uncertainty, and free-energy"
        Frontiers in Human Neuroscience 4:215.

    Precision = inverse variance = confidence in a signal.
    High precision → signal strongly influences belief updates.
    Low precision → signal is down-weighted (treated as noisy).

    In the predictive brain:
        - Attention = increasing precision on selected sensory channels
        - Hallucination = prediction with zero sensory precision
        - Sensory attenuation = decreasing precision of predicted signals

    HDC implementation:
        - Each modality has a precision weight π_i ∈ [0, 1]
        - Combined prediction: MAJORITY(π_i × channel_i for all i)
        - Precision updates via free energy gradient

    Args:
        dim:        HV dimension
        n_channels: Number of sensory channels
        device:     torch device
    """

    def __init__(self, dim: int, n_channels: int, device: str = "cpu"):
        self.dim        = dim
        self.n_channels = n_channels
        self.device     = device

        # Precision for each channel (initialised to 0.5 = neutral)
        self.precision = torch.full((n_channels,), 0.5, device=device)
        self._fe_per_channel: List[List[float]] = [[] for _ in range(n_channels)]

    def integrate(
        self,
        channels: List[torch.Tensor],   # List of (D,) HVs, one per channel
    ) -> torch.Tensor:
        """
        Precision-weighted integration of sensory channels.

        Channels with higher precision contribute more to the integrated HV.

        Args:
            channels: List of n_channels (D,) HVs

        Returns:
            (D,) precision-weighted integrated HV
        """
        if not channels:
            return torch.zeros(self.dim, device=self.device)

        weighted = sum(
            self.precision[i].item() * ch.float().to(self.device)
            for i, ch in enumerate(channels[:self.n_channels])
        )
        return _majority(weighted / (self.precision[:len(channels)].sum() + 1e-8))

    def update_precision(
        self,
        channel_idx: int,
        prediction_error: float,
        lr: float = 0.05,
    ):
        """
        Update precision based on prediction error.

        Higher prediction error → lower precision (signal is unreliable).
        π_i ← π_i - lr × prediction_error

        Args:
            channel_idx:      Index of the channel to update
            prediction_error: Normalised prediction error for this channel [0,1]
            lr:               Precision learning rate
        """
        pi_new = float(self.precision[channel_idx].item()) - lr * prediction_error
        self.precision[channel_idx] = max(0.01, min(0.99, pi_new))
        self._fe_per_channel[channel_idx].append(prediction_error)

    def attend_to(self, channel_idx: int, boost: float = 0.2):
        """
        Explicitly increase precision for a specific channel (attention).

        This implements the 'spotlight of attention' in the free-energy framework.
        """
        for i in range(self.n_channels):
            if i == channel_idx:
                self.precision[i] = min(0.99, float(self.precision[i]) + boost)
            else:
                self.precision[i] = max(0.01, float(self.precision[i]) - boost * 0.5)

    def precision_report(self) -> Dict[str, float]:
        return {f"ch_{i}": float(p) for i, p in enumerate(self.precision)}

    def most_attended(self) -> int:
        """Return the channel index with highest precision (peak attention)."""
        return int(self.precision.argmax().item())

    def precision_entropy(self) -> float:
        """
        Entropy of the precision distribution — low = focused attention, high = diffuse.
        Normalised to [0, 1] by dividing by log(n_channels).
        """
        p = self.precision.clamp(1e-6, 1.0 - 1e-6)
        p = p / p.sum()
        H = -float((p * p.log()).sum().item())
        return round(H / (torch.log(torch.tensor(float(self.n_channels))) + 1e-8).item(), 4)

    def attention_summary(self) -> Dict:
        """One-call attention state: peak channel, entropy, all precisions."""
        return {
            "peak_channel":     self.most_attended(),
            "peak_precision":   round(float(self.precision.max().item()), 4),
            "precision_entropy": self.precision_entropy(),
            **self.precision_report(),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. BeliefPropagation — hierarchical predictive coding
# ═══════════════════════════════════════════════════════════════════════════════

class BeliefPropagation:
    """
    Hierarchical belief propagation for multi-level predictive coding.

    Reference:
        Rao & Ballard (1999) "Predictive coding in the visual cortex:
        a functional interpretation of some extra-classical receptive-field effects"
        Nature Neuroscience 2(1):79–87.

        Friston (2008) "Hierarchical models in the brain"
        PLoS Computational Biology 4(11):e1000211.

    Architecture:
        Layer L (highest, most abstract):
            sends predictions down, receives errors up
        Layer L-1:
            receives top-down predictions, sends bottom-up errors
        ...
        Layer 0 (sensory):
            receives top-down predictions, emits prediction errors to L0

    Each layer:
        prediction_error_l = observation_l - top_down_prediction_l
        bottom_up_signal_l = f(prediction_error_l)  [to layer l+1]
        top_down_pred_l-1  = g(state_l)             [to layer l-1]

    Args:
        n_layers: Number of hierarchical levels
        dim:      HV dimension per level
    """

    def __init__(self, n_layers: int = 3, dim: int = 512, device: str = "cpu"):
        self.n_layers = n_layers
        self.dim      = dim
        self.device   = device

        # Belief states at each level (initially random)
        self.states = [_gen_hv(dim, seed=i, device=device) for i in range(n_layers)]

        # Prediction errors at each level (initially zero)
        self.errors = [torch.zeros(dim, device=device) for _ in range(n_layers)]

        # Top-down generative connections (random fixed projections)
        self._top_down = [
            _gen_hv(dim, seed=100 + i, device=device) for i in range(n_layers - 1)
        ]

        self._steps = 0

    def _top_down_predict(self, level: int) -> torch.Tensor:
        """Generate prediction for level `level` from level `level+1`."""
        if level >= self.n_layers - 1:
            return self.states[level].clone()
        # Top-down prediction: XOR of higher-level state with top-down weight
        return _majority(_xor(self.states[level + 1], self._top_down[level]).float())

    def forward(
        self,
        sensory_input: torch.Tensor,
        n_iterations:  int = 3,
    ) -> List[torch.Tensor]:
        """
        Run belief propagation to equilibrium.

        Args:
            sensory_input: (D,) observed HV at the bottom level
            n_iterations:  Number of update iterations (3 usually sufficient)

        Returns:
            List of n_layers belief states at convergence
        """
        self._steps += 1

        for _ in range(n_iterations):
            # Bottom-up pass: update errors from sensory to top
            self.errors[0] = _majority(
                (sensory_input.float() - self._top_down_predict(0)).abs()
            )
            for l in range(1, self.n_layers):
                if l < self.n_layers - 1:
                    self.errors[l] = _majority(
                        (self.states[l - 1].float() - self._top_down_predict(l)).abs()
                    )

            # Top-down pass: update states from errors
            # Bottom-level receives sensory input (strong signal)
            self.states[0] = _majority(
                0.7 * sensory_input.float() + 0.3 * self._top_down_predict(0).float()
            )
            # Higher levels update from bottom-up errors
            for l in range(1, self.n_layers):
                self.states[l] = _majority(
                    0.6 * self.states[l].float() + 0.4 * self.errors[l - 1].float()
                )

        return [s.clone() for s in self.states]

    def prediction_error_norm(self) -> float:
        """Total prediction error across all levels (convergence metric)."""
        return sum(float(e.float().mean()) for e in self.errors) / self.n_layers

    def level_similarity(self, level_a: int, level_b: int) -> float:
        """Hamming similarity between belief states at two levels."""
        a = self.states[level_a]
        b = self.states[level_b]
        return float(_hamming(a.unsqueeze(0), b.unsqueeze(0)).item())

    def hierarchy_report(self) -> Dict:
        """Error per level, total steps, and cross-level similarity."""
        per_level = {f"error_L{l}": round(float(self.errors[l].float().mean().item()), 4)
                     for l in range(self.n_layers)}
        sims = {}
        for la in range(self.n_layers):
            for lb in range(la + 1, self.n_layers):
                sims[f"sim_L{la}_L{lb}"] = round(self.level_similarity(la, lb), 4)
        return {
            "steps":          self._steps,
            "total_error":    round(self.prediction_error_norm(), 4),
            **per_level,
            **sims,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 5. ExpectedFreeEnergy — action selection criterion
# ═══════════════════════════════════════════════════════════════════════════════

class ExpectedFreeEnergy:
    """
    Expected Free Energy G for action selection.

    Reference:
        Friston et al. (2017) "Active inference: a process theory"
        Neural Computation 29(1):1–49. — §3.3 Expected free energy.

        Parr & Friston (2019) "Generalised free energy and active inference"
        Biological Cybernetics. — G = instrumental + epistemic value.

    G(a) = E_q[log q(s|a) - log p(s|a, o_preferred)]
         ≈ pragmatic_cost(a) - epistemic_bonus(a)

    where:
        pragmatic_cost = how far is the expected future from preferred states
        epistemic_bonus = information gain (drives exploration)

    Action selection: a* = argmin_a G(a)

    This is preferable to ε-greedy or UCB exploration because:
        1. Exploration is internally motivated (not random)
        2. The agent seeks informative states naturally
        3. No separate exploration rate parameter needed

    Args:
        agent: ActiveInferenceAgent with belief + preferred states
        epistemic_weight: λ controls exploration vs exploitation
    """

    def __init__(self, agent: ActiveInferenceAgent, epistemic_weight: float = 0.3):
        self.agent             = agent
        self.epistemic_weight  = epistemic_weight
        self._G_history: List[float] = []

    def compute(
        self,
        action: torch.Tensor,
        n_simulations: int = 5,
    ) -> Dict[str, float]:
        """
        Compute expected free energy for a single action.

        Monte-Carlo estimate over n_simulations future trajectories.

        Returns:
            Dict with 'pragmatic', 'epistemic', 'G'
        """
        pragma_vals = []
        epistem_vals = []

        for sim in range(n_simulations):
            # Simulate future state
            if self.agent.model is not None:
                try:
                    future = self.agent.model(self.agent.belief, action)
                except Exception:
                    future = _majority(_xor(self.agent.belief, action).float())
            else:
                future = _majority(_xor(self.agent.belief, action).float())

            # Pragmatic: distance to preferred
            pragma = 1.0 - float(_hamming(
                future.unsqueeze(0), self.agent.preferred.unsqueeze(0)
            ).item())

            # Epistemic: distance from current belief (information gain)
            epistem = 1.0 - float(_hamming(
                future.unsqueeze(0), self.agent.belief.unsqueeze(0)
            ).item())

            pragma_vals.append(pragma)
            epistem_vals.append(epistem)

        avg_pragma  = sum(pragma_vals) / n_simulations
        avg_epistem = sum(epistem_vals) / n_simulations
        G           = avg_pragma - self.epistemic_weight * avg_epistem

        self._G_history.append(G)
        return {
            "pragmatic": avg_pragma,
            "epistemic": avg_epistem,
            "G":         G,
        }

    def select_best(
        self,
        actions:       List[torch.Tensor],
        n_simulations: int = 5,
    ) -> Tuple[int, float, List[Dict]]:
        """
        Select action with minimum expected free energy.

        Returns:
            (best_idx, best_G, all_results)
        """
        results = [self.compute(a, n_simulations) for a in actions]
        best    = min(range(len(results)), key=lambda i: results[i]["G"])
        return best, results[best]["G"], results

    def running_G(self) -> float:
        """Running average of G (lower = better alignment with preferred states)."""
        h = self._G_history[-20:]
        return sum(h) / max(len(h), 1)

    def G_trend(self, window: int = 10) -> float:
        """
        Linear slope of G over the last `window` steps.
        Negative slope = agent is improving (G decreasing toward preferred states).
        """
        h = self._G_history[-window:]
        if len(h) < 2:
            return 0.0
        n = len(h)
        xs = list(range(n))
        mx, my = sum(xs) / n, sum(h) / n
        num = sum((x - mx) * (y - my) for x, y in zip(xs, h))
        den = sum((x - mx) ** 2 for x in xs) + 1e-8
        return round(num / den, 6)

    def EFE_summary(self) -> Dict:
        """Summary of EFE agent: running G, trend, epistemic weight."""
        return {
            "running_G":       round(self.running_G(), 4),
            "G_trend":         self.G_trend(),
            "epistemic_weight": self.epistemic_weight,
            "n_evaluations":   len(self._G_history),
            "is_improving":    self.G_trend() < 0,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Smoke test
# ═══════════════════════════════════════════════════════════════════════════════

def _test_active_inference():
    D = 256

    print("=== FreeEnergyEstimator ===")
    fe = FreeEnergyEstimator(dim=D, complexity_weight=0.5)
    pred   = _gen_hv(D, seed=0)
    obs    = _gen_hv(D, seed=1)
    belief = _gen_hv(D, seed=2)
    result = fe.free_energy(pred, obs, belief)
    assert "accuracy" in result and "free_energy" in result
    assert 0.0 <= result["free_energy"] <= 2.0
    print(f"  F={result['free_energy']:.3f} "
          f"(acc={result['accuracy']:.3f}, cmp={result['complexity']:.3f})  OK")

    # Exact prediction should have near-zero accuracy term
    exact = fe.free_energy(pred, pred, belief)
    assert exact["accuracy"] < 0.01
    print(f"  Exact prediction: acc={exact['accuracy']:.4f} ≈ 0  OK")

    print("\n=== ActiveInferenceAgent ===")
    preferred = _gen_hv(D, seed=99)
    agent = ActiveInferenceAgent(dim=D, preferred_obs=preferred)

    for i in range(10):
        obs = _gen_hv(D, seed=i)
        fe_result = agent.perceive(obs, lr=0.1)

    assert agent._step == 10
    report = agent.free_energy_report()
    print(f"  After 10 steps: F={report['avg_free_energy']:.3f}, "
          f"n_steps={report['n_steps']}  OK")

    # Action selection
    actions = [_gen_hv(D, seed=100 + i) for i in range(4)]
    best_idx, best_G, all_G = agent.select_action(actions, horizon=2)
    assert 0 <= best_idx < 4
    print(f"  Best action: {best_idx}, G={best_G:.4f}  OK")

    print("\n=== PrecisionWeightedAttention ===")
    pwa = PrecisionWeightedAttention(dim=D, n_channels=3)
    channels = [_gen_hv(D, seed=i) for i in range(3)]
    integrated = pwa.integrate(channels)
    assert integrated.shape == (D,)
    print(f"  Integrated shape: {integrated.shape}  OK")

    pwa.update_precision(0, prediction_error=0.3)
    pwa.attend_to(1, boost=0.2)
    print(f"  Precisions: {pwa.precision_report()}  OK")

    print("\n=== BeliefPropagation ===")
    bp  = BeliefPropagation(n_layers=3, dim=D)
    states = bp.forward(_gen_hv(D, seed=0), n_iterations=3)
    assert len(states) == 3
    assert all(s.shape == (D,) for s in states)
    pe = bp.prediction_error_norm()
    print(f"  PE after convergence: {pe:.4f}, states: {len(states)}  OK")

    print("\n=== ExpectedFreeEnergy ===")
    efe = ExpectedFreeEnergy(agent, epistemic_weight=0.3)
    actions = [_gen_hv(D, seed=i) for i in range(3)]
    best, G_best, results = efe.select_best(actions, n_simulations=3)
    assert 0 <= best < 3
    print(f"  Best action: {best}, G={G_best:.4f}  OK")

    print("\n✅ All active_inference tests passed")


if __name__ == "__main__":
    _test_active_inference()
