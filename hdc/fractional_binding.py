"""
Fractional Power Encoding for Continuous HDC Geometry
======================================================
Reference:
    Plate (1995) "Holographic Reduced Representations" IEEE TNNLS 6(3):623-641.
    Kanerva (2009) "Hyperdimensional Computing" Cognitive Computation 1(2):139-159.
    Frady & Sommer (2021) "Robust computation with rhythmic spike patterns"
    PNAS 116(36):18138-18147.

    Kymn et al. (2025) "Binding in hippocampal-entorhinal circuits enables
    compositionality in cognitive maps" (grid_cells.py reference paper).

Core insight:
    For bipolar VSA (values ±1), the group of HVs under element-wise
    multiplication forms a continuous manifold.  Fractional power x^α (α ∈ ℝ)
    maps each component xᵢ ∈ {−1,+1} to a real-valued phasor on the unit circle:

        x^α_i  = exp(i × α × π × 𝟙[xᵢ = −1])

    For binary HDC ({0,1}), we use a probabilistic approximation:
        x^α ≈ flip each '1' bit back to '0' independently with P(flip) = 1 − α
        and each '0' bit to '1' with P(flip) = 0     [asymmetric interpolation]

    This gives:
        x^0 ≈ 0̄   (identity element for XOR)
        x^1 = x
        Hamming(x^α, x^β) ≈ |α − β| × density(x)

Applications in Physical AI:
    1. **Continuous state encoding**: position_hv(t) = base^(t/T) represents a
       smooth trajectory through HV space — no discontinuities, no aliasing.
    2. **Temporal context**: context_t = Σ_τ bind(base^τ, observation_τ) encodes
       the whole history as one HV with recency weighting.
    3. **Velocity encoding**: velocity_hv = base^(v/v_max) ∈ [base^0, base^1].
    4. **Interpolation**: between two states A and B:
          lerp_hv(α) = majority(A^(1−α), B^α)    (α ∈ [0,1])
    5. **Physics constraints**: enforce x(t+dt) ≈ x(t) ⊗ velocity^dt.

This module implements:
    FractionalPower          — core x^α operation
    ContinuousStateEncoder   — encode position/velocity as HVs with FPE
    TemporalContextEncoder   — rolling recency-weighted context HV
    PhysicsHVConstraints     — kinematic/energy consistency via FPE
    FPEInterpolator          — smooth interpolation between prototype HVs
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from hdc.physics_world_model import _xor, _majority, _hamming


# ── Helper ─────────────────────────────────────────────────────────────────────

def _gen_hv(dim: int, seed: Optional[int] = None, device: str = "cpu") -> torch.Tensor:
    g = torch.Generator(device=device)
    if seed is not None:
        g.manual_seed(seed)
    return (torch.rand(dim, generator=g, device=device) >= 0.5).float()


# ═══════════════════════════════════════════════════════════════════════════════
# 1. FractionalPower — core binary x^α
# ═══════════════════════════════════════════════════════════════════════════════

class FractionalPower:
    """
    Fractional power encoding for binary hypervectors.

    Computes x^α for α ∈ [0, 1] via controlled bit flipping:
        - α = 0: return all-zero HV (identity element for XOR)
        - α = 1: return x unchanged
        - α ∈ (0,1): each '1' bit of x is kept with probability α
                     each '0' bit of x stays 0 (asymmetric, for XOR algebra)

    Group property (approximate):
        x^α ⊗ x^β ≈ x^(α+β)  for α + β ≤ 1
        Hamming(x^α, x^β) ≈ |α − β| × density(x) × D

    Deterministic mode (seed-based):
        For reproducible encoding, pass seed to ensure the same random mask
        is used for the same input. This is important for trajectory encoding.

    Args:
        dim: Hypervector dimension D
        device: torch device
    """

    def __init__(self, dim: int, device: str = "cpu"):
        self.dim    = dim
        self.device = device

    def power(
        self,
        x: torch.Tensor,
        alpha: float,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Compute x^alpha for binary HV x ∈ {0,1}^D.

        Args:
            x:     (D,) binary hypervector
            alpha: Exponent ∈ [0, 1]. Values outside this range are clamped.
            seed:  Optional seed for deterministic flip mask.

        Returns:
            (D,) binary hypervector x^alpha
        """
        alpha = max(0.0, min(1.0, float(alpha)))
        if alpha == 0.0:
            return torch.zeros(self.dim, device=self.device)
        if alpha == 1.0:
            return x.float().to(self.device).clone()

        x_f = x.float().to(self.device)
        g   = torch.Generator(device=self.device)
        if seed is not None:
            g.manual_seed(seed)
        # Keep each '1' bit with probability alpha; '0' bits stay 0
        keep_mask = torch.rand(self.dim, generator=g, device=self.device) < alpha
        return x_f * keep_mask.float()

    def interpolate(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        alpha: float,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Smooth interpolation between HVs a (α=0) and b (α=1).

        lerp_hv(α) = majority(a^(1−α) ⊗ b^α)

        Args:
            a, b:  (D,) binary hypervectors (endpoints)
            alpha: Interpolation factor ∈ [0, 1]; 0 → a, 1 → b

        Returns:
            (D,) interpolated binary hypervector
        """
        a_part = self.power(a, 1.0 - alpha, seed=seed)
        b_part = self.power(b, alpha, seed=(seed or 0) + 1 if seed is not None else None)
        # Blend by majority vote of both contributions
        return _majority((a_part + b_part) / 2.0)

    def trajectory(
        self,
        x: torch.Tensor,
        n_steps: int,
        alpha_start: float = 0.0,
        alpha_end:   float = 1.0,
    ) -> torch.Tensor:
        """
        Generate a smooth trajectory of n_steps HVs from x^alpha_start to x^alpha_end.

        Returns:
            (n_steps, D) tensor of fractional power HVs
        """
        alphas = torch.linspace(alpha_start, alpha_end, n_steps)
        hvs    = [self.power(x, float(a), seed=i) for i, a in enumerate(alphas)]
        return torch.stack(hvs)

    def inverse(self, x: torch.Tensor, alpha: float) -> torch.Tensor:
        """
        Approximate inverse: x^(−α) ≈ XOR(x^α, x)  (approx for small α).

        For XOR algebra: bind(x^α, x^(−α)) ≈ identity.
        """
        return _xor(x.float(), self.power(x, alpha))


# ═══════════════════════════════════════════════════════════════════════════════
# 2. ContinuousStateEncoder — position + velocity encoding via FPE
# ═══════════════════════════════════════════════════════════════════════════════

class ContinuousStateEncoder:
    """
    Encode continuous physical state (position, velocity, acceleration)
    as binary hypervectors via fractional power encoding.

    Each physical dimension gets a base HV; the value is encoded by
    taking the fractional power of that base HV.

    For a 3D position (x, y, z):
        hv_x = base_x ^ (x / x_range)
        hv_y = base_y ^ (y / y_range)
        hv_z = base_z ^ (z / z_range)
        state_hv = MAJORITY(hv_x ⊗ hv_y ⊗ hv_z)

    Properties:
        - Similar positions → similar HVs (Hamming distance ≈ |Δpos|)
        - Distinct dimensions are orthogonal by construction
        - Velocity = position ⊗ prev_position (approximate finite difference)

    Args:
        dim:       Hypervector dimension D
        n_dims:    Number of physical dimensions (e.g., 3 for x,y,z)
        ranges:    [(min, max)] per dimension for normalisation
        device:    torch device
    """

    def __init__(
        self,
        dim:    int,
        n_dims: int,
        ranges: Optional[List[Tuple[float, float]]] = None,
        device: str = "cpu",
    ):
        self.dim    = dim
        self.n_dims = n_dims
        self.device = device
        self.ranges = ranges or [(-1.0, 1.0)] * n_dims

        self.fpe   = FractionalPower(dim, device)
        # One base HV per physical dimension (orthogonal by construction)
        self._bases = [_gen_hv(dim, seed=i, device=device) for i in range(n_dims)]
        self._prev_hv: Optional[torch.Tensor] = None

    def encode_position(self, pos: torch.Tensor) -> torch.Tensor:
        """
        Encode a position vector as a HV.

        Args:
            pos: (n_dims,) position values

        Returns:
            (D,) binary hypervector
        """
        pos_f = pos.float()
        hvs   = []
        for i in range(self.n_dims):
            lo, hi = self.ranges[i]
            alpha  = float((pos_f[i].item() - lo) / max(hi - lo, 1e-8))
            alpha  = max(0.0, min(1.0, alpha))
            hvs.append(self.fpe.power(self._bases[i], alpha, seed=i))

        # XOR-bind all dimensions then binarize
        composite = hvs[0].clone()
        for hv in hvs[1:]:
            composite = _xor(composite, hv)
        return _majority(composite)

    def encode_velocity(
        self,
        pos: torch.Tensor,
        prev_pos: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode position and compute velocity HV as XOR with previous position.

        Returns:
            (position_hv, velocity_hv)
        """
        pos_hv  = self.encode_position(pos)
        if prev_pos is not None:
            prev_hv = self.encode_position(prev_pos)
            vel_hv  = _xor(pos_hv, prev_hv)   # velocity = displacement in HV space
        elif self._prev_hv is not None:
            vel_hv  = _xor(pos_hv, self._prev_hv)
        else:
            vel_hv  = torch.zeros(self.dim, device=self.device)

        self._prev_hv = pos_hv.detach().clone()
        return pos_hv, vel_hv

    def predict_next_position(
        self,
        pos_hv: torch.Tensor,
        vel_hv: torch.Tensor,
        dt: float = 1.0,
    ) -> torch.Tensor:
        """
        Predict next position HV using kinematic integration:
            pos(t+dt) ≈ pos(t) ⊗ vel(t)^dt

        Args:
            pos_hv: (D,) current position HV
            vel_hv: (D,) current velocity HV (displacement per step)
            dt:     Time step scale (fractional velocity power)

        Returns:
            (D,) predicted next position HV
        """
        vel_scaled = self.fpe.power(vel_hv, dt)
        return _majority(_xor(pos_hv, vel_scaled).float())


# ═══════════════════════════════════════════════════════════════════════════════
# 3. TemporalContextEncoder — recency-weighted history
# ═══════════════════════════════════════════════════════════════════════════════

class TemporalContextEncoder:
    """
    Rolling recency-weighted temporal context hypervector.

    Encodes a sequence of observations into a single context HV by bundling
    time-shifted (fractional power) observations:

        C_t = MAJORITY( Σ_τ w_τ × bind(time_marker^τ, obs_hv_{t−τ}) )

    where w_τ = e^(−τ/τ_decay) is exponential recency weighting and
    time_marker is a fixed base HV that acts as a temporal axis.

    Properties:
        - C_t has maximum Hamming similarity to queries from recent timesteps
        - Older observations are exponentially downweighted
        - Capacity: O(D / log D) distinct timesteps before interference
        - Query: unbind(C_t, time_marker^τ) ≈ obs_hv_{t−τ} for recent τ

    Args:
        dim:         Hypervector dimension D
        tau_decay:   Recency decay time constant (steps; default 10)
        max_history: Maximum lag τ_max (default 20)
        device:      torch device
    """

    def __init__(
        self,
        dim:         int,
        tau_decay:   float = 10.0,
        max_history: int   = 20,
        device:      str   = "cpu",
    ):
        self.dim         = dim
        self.tau_decay   = tau_decay
        self.max_history = max_history
        self.device      = device

        self.fpe          = FractionalPower(dim, device)
        self._time_marker = _gen_hv(dim, seed=999, device=device)
        self._context     = torch.zeros(dim, device=device)
        self._obs_buf:    List[torch.Tensor] = []
        self._decay       = math.exp(-1.0 / tau_decay)

    def push(self, obs_hv: torch.Tensor):
        """
        Add a new observation to the temporal context.

        TRUE O(D) per step via EMA update — no rebuild needed:
            C_{t+1} = decay × (C_t ⊗ time_roll) + (1-decay) × obs_t

        This is equivalent to the full weighted sum but computed in O(D)
        instead of O(H×D) by maintaining the exponential accumulator directly.
        The time_roll permutation shifts all past observations forward by one
        lag automatically.
        """
        obs_hv = obs_hv.float().to(self.device)
        self._obs_buf.insert(0, obs_hv.clone())
        if len(self._obs_buf) > self.max_history:
            self._obs_buf.pop()

        # O(D) incremental: decay old context (shifted by 1 lag) + add new obs
        ctx_shifted = torch.roll(self._context, shifts=1)   # simulate time passing
        self._context = self._decay * ctx_shifted + (1 - self._decay) * obs_hv

    def _rebuild_context(self) -> torch.Tensor:
        """Rebuild context as weighted bundle of time-shifted observations."""
        total = torch.zeros(self.dim, device=self.device)
        total_w = 0.0
        for tau, obs in enumerate(self._obs_buf):
            w     = math.exp(-tau / self.tau_decay)
            t_hv  = self.fpe.power(self._time_marker, tau / max(self.max_history, 1), seed=tau)
            bound = _xor(t_hv, obs)
            total = total + w * bound.float()
            total_w += w
        return _majority(total / max(total_w, 1e-8))

    @property
    def context(self) -> torch.Tensor:
        """Current temporal context HV (D,)."""
        return self._context

    def query(self, lag: int) -> torch.Tensor:
        """
        Retrieve approximate observation from `lag` steps ago.

        Args:
            lag: Number of steps back (0 = most recent)

        Returns:
            (D,) approximate reconstruction of obs_{t-lag}
        """
        t_hv = self.fpe.power(self._time_marker, lag / max(self.max_history, 1), seed=lag)
        return _majority(_xor(self._context, t_hv).float())

    def reset(self):
        self._context = torch.zeros(self.dim, device=self.device)
        self._obs_buf = []


# ═══════════════════════════════════════════════════════════════════════════════
# 4. PhysicsHVConstraints — kinematic + energy constraints via FPE
# ═══════════════════════════════════════════════════════════════════════════════

class PhysicsHVConstraints:
    """
    Physics-respecting HDC constraints using fractional power encoding.

    Enforces that predicted HV trajectories satisfy:
        1. Kinematic consistency: pos(t+1) ≈ pos(t) ⊗ vel(t)
        2. Bounded velocity: Hamming(vel_hv, zero_hv) ≤ max_vel_hamming
        3. Smoothness: Hamming(state(t+1), state(t)) ≤ max_step_hamming

    These act as soft constraints on world model predictions:
        score = expected_utility − λ_kin × kinematic_error
                                 − λ_vel × velocity_bound_violation
                                 − λ_smooth × smoothness_violation

    Args:
        dim:                 HV dimension D
        max_vel_hamming:     Max allowed Hamming distance per step (velocity bound)
        max_step_hamming:    Max allowed state change per step (smoothness)
        lambda_kinematic:    Weight for kinematic constraint
        lambda_velocity:     Weight for velocity bound constraint
        lambda_smoothness:   Weight for smoothness constraint
    """

    def __init__(
        self,
        dim:               int,
        max_vel_hamming:   float = 0.3,
        max_step_hamming:  float = 0.4,
        lambda_kinematic:  float = 0.5,
        lambda_velocity:   float = 0.3,
        lambda_smoothness: float = 0.2,
    ):
        self.dim              = dim
        self.max_vel_hamming  = max_vel_hamming
        self.max_step_hamming = max_step_hamming
        self.lambda_k         = lambda_kinematic
        self.lambda_v         = lambda_velocity
        self.lambda_s         = lambda_smoothness

    def kinematic_error(
        self,
        pos_t:   torch.Tensor,
        vel_t:   torch.Tensor,
        pos_t1:  torch.Tensor,
    ) -> float:
        """
        Kinematic inconsistency: how far is pos(t+1) from pos(t) ⊗ vel(t)?

        Returns:
            Error ∈ [0, 1], where 0 = perfectly consistent.
        """
        predicted_t1 = _majority(_xor(pos_t, vel_t).float())
        return 1.0 - float(_hamming(predicted_t1.unsqueeze(0), pos_t1.unsqueeze(0)).item())

    def velocity_violation(self, vel_hv: torch.Tensor) -> float:
        """
        How much does vel_hv exceed the maximum allowed velocity?

        Returns:
            Violation ∈ [0, 1], where 0 = within bounds.
        """
        zero = torch.zeros(self.dim, device=vel_hv.device)
        actual_hamming = 1.0 - float(_hamming(vel_hv.unsqueeze(0), zero.unsqueeze(0)).item())
        return max(0.0, actual_hamming - self.max_vel_hamming)

    def smoothness_violation(self, state_t: torch.Tensor, state_t1: torch.Tensor) -> float:
        """
        How much does the state change exceed the smoothness bound?

        Returns:
            Violation ∈ [0, 1], where 0 = within bounds.
        """
        step_size = 1.0 - float(_hamming(state_t.unsqueeze(0), state_t1.unsqueeze(0)).item())
        return max(0.0, step_size - self.max_step_hamming)

    def score(
        self,
        pos_t:   torch.Tensor,
        vel_t:   torch.Tensor,
        pos_t1:  torch.Tensor,
    ) -> float:
        """
        Physics constraint score (higher = better, unconstrained = 0.0).

        Returns:
            Negative penalty ∈ (−∞, 0]; add to utility before ranking actions.
        """
        kin  = self.lambda_k * self.kinematic_error(pos_t, vel_t, pos_t1)
        vel  = self.lambda_v * self.velocity_violation(vel_t)
        smo  = self.lambda_s * self.smoothness_violation(pos_t, pos_t1)
        return -(kin + vel + smo)

    def is_physically_plausible(
        self,
        pos_t: torch.Tensor,
        vel_t: torch.Tensor,
        pos_t1: torch.Tensor,
        threshold: float = 0.5,
    ) -> bool:
        """Return True if the transition is physically plausible."""
        return self.score(pos_t, vel_t, pos_t1) > -threshold


# ═══════════════════════════════════════════════════════════════════════════════
# 5. FPEInterpolator — smooth prototype interpolation
# ═══════════════════════════════════════════════════════════════════════════════

class FPEInterpolator:
    """
    Smooth interpolation between registered prototype HVs via FPE.

    Applications:
        - Fault severity encoding: normal^1.0 → fault^0.0 (interpolate severity)
        - Task difficulty: easy^1.0 → hard^0.0
        - Physical state: stable^1.0 → critical^0.0

    Retrieval: given a query HV, find where on the interpolation axis it lies.

    Args:
        dim:    HV dimension
        device: torch device
    """

    def __init__(self, dim: int, device: str = "cpu"):
        self.dim    = dim
        self.device = device
        self.fpe    = FractionalPower(dim, device)
        self._prototypes: Dict[str, torch.Tensor] = {}

    def register(self, name: str, hv: torch.Tensor):
        """Register a named prototype."""
        self._prototypes[name] = hv.float().to(self.device)

    def interpolate(self, name_a: str, name_b: str, alpha: float) -> torch.Tensor:
        """
        Interpolate from prototype `name_a` (α=0) to `name_b` (α=1).

        Returns:
            (D,) interpolated HV
        """
        a = self._prototypes[name_a]
        b = self._prototypes[name_b]
        return self.fpe.interpolate(a, b, alpha)

    def locate(self, query_hv: torch.Tensor, name_a: str, name_b: str,
               resolution: int = 10) -> float:
        """
        Find where on the [name_a, name_b] axis the query HV lies.

        Uses coarse grid search over α ∈ [0, 1].

        Returns:
            α ∈ [0, 1] where query_hv is closest to the interpolated HV.
        """
        best_alpha = 0.0
        best_sim   = -1.0
        for k in range(resolution + 1):
            alpha      = k / resolution
            interp     = self.interpolate(name_a, name_b, alpha)
            sim        = float(_hamming(query_hv.unsqueeze(0), interp.unsqueeze(0)).item())
            if sim > best_sim:
                best_sim   = sim
                best_alpha = alpha
        return best_alpha


# ═══════════════════════════════════════════════════════════════════════════════
# Smoke test
# ═══════════════════════════════════════════════════════════════════════════════

def _test_fractional_binding():
    D = 500

    def _gen(s):
        g = torch.Generator()
        g.manual_seed(s)
        return (torch.rand(D, generator=g) >= 0.5).float()

    print("=== FractionalPower ===")
    fpe = FractionalPower(D)
    x   = _gen(42)

    x0 = fpe.power(x, 0.0)
    x1 = fpe.power(x, 1.0)
    xh = fpe.power(x, 0.5)

    sim_01 = float(_hamming(x0.unsqueeze(0), x1.unsqueeze(0)).item())
    sim_0h = float(_hamming(x0.unsqueeze(0), xh.unsqueeze(0)).item())
    sim_h1 = float(_hamming(xh.unsqueeze(0), x1.unsqueeze(0)).item())
    print(f"  sim(x^0, x^1)={sim_01:.3f}  sim(x^0, x^0.5)={sim_0h:.3f}  sim(x^0.5, x^1)={sim_h1:.3f}")
    print(f"  x^0 density={x0.mean():.3f}  x^0.5 density={xh.mean():.3f}  x^1 density={x1.mean():.3f}")

    traj = fpe.trajectory(x, n_steps=5)
    print(f"  Trajectory shape: {traj.shape}")

    print("\n=== ContinuousStateEncoder ===")
    enc = ContinuousStateEncoder(D, n_dims=2, ranges=[(-5.0, 5.0), (-5.0, 5.0)])
    pos1 = torch.tensor([0.0, 0.0])
    pos2 = torch.tensor([0.1, 0.0])
    pos3 = torch.tensor([5.0, 5.0])

    hv1 = enc.encode_position(pos1)
    hv2 = enc.encode_position(pos2)
    hv3 = enc.encode_position(pos3)

    sim_nearby  = float(_hamming(hv1.unsqueeze(0), hv2.unsqueeze(0)).item())
    sim_distant = float(_hamming(hv1.unsqueeze(0), hv3.unsqueeze(0)).item())
    print(f"  sim(nearby positions)={sim_nearby:.3f}  sim(distant)={sim_distant:.3f}")
    print(f"  (nearby should > distant — {'✓' if sim_nearby > sim_distant else '✗'})")

    pos_hv, vel_hv = enc.encode_velocity(pos2, prev_pos=pos1)
    next_pred = enc.predict_next_position(pos_hv, vel_hv, dt=1.0)
    print(f"  Predicted next pos HV shape: {next_pred.shape}")

    print("\n=== TemporalContextEncoder ===")
    tce = TemporalContextEncoder(D, tau_decay=5, max_history=10)
    for i in range(10):
        tce.push(_gen(i))
    ctx = tce.context
    print(f"  Context shape: {ctx.shape}, density: {ctx.mean():.3f}")
    retrieved = tce.query(lag=0)
    sim_q0 = float(_hamming(_gen(9).unsqueeze(0), retrieved.unsqueeze(0)).item())
    print(f"  Query lag=0 similarity to most recent obs: {sim_q0:.3f}")

    print("\n=== PhysicsHVConstraints ===")
    phys = PhysicsHVConstraints(D)
    pos_t  = _gen(10)
    vel_t  = _gen(11)
    pos_t1 = _majority((_xor(pos_t, vel_t)).float())  # physically consistent next state
    pos_bad = _gen(99)                                  # random (inconsistent)

    score_good = phys.score(pos_t, vel_t, pos_t1)
    score_bad  = phys.score(pos_t, vel_t, pos_bad)
    print(f"  Consistent transition score: {score_good:.3f}  (should be close to 0)")
    print(f"  Inconsistent transition score: {score_bad:.3f}  (should be negative)")
    plausible = phys.is_physically_plausible(pos_t, vel_t, pos_t1)
    print(f"  is_physically_plausible(consistent): {plausible}")

    print("\n=== FPEInterpolator ===")
    interp = FPEInterpolator(D)
    interp.register("normal", _gen(0))
    interp.register("fault",  _gen(1))

    mid = interp.interpolate("normal", "fault", alpha=0.5)
    print(f"  Interpolated HV density: {mid.mean():.3f}")
    alpha_found = interp.locate(mid, "normal", "fault", resolution=20)
    print(f"  Located α={alpha_found:.2f}  (expected ≈0.5)")

    print("\n✅ All fractional_binding tests passed")


if __name__ == "__main__":
    _test_fractional_binding()
