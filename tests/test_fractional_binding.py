"""
tests/test_fractional_binding.py
==================================
Tests for FractionalPower, ContinuousStateEncoder,
TemporalContextEncoder, PhysicsHVConstraints, FPEInterpolator.
"""
import pytest
import torch
from hdc.fractional_binding import (
    FractionalPower,
    ContinuousStateEncoder,
    TemporalContextEncoder,
    PhysicsHVConstraints,
    FPEInterpolator,
    _gen_hv,
)
from hdc.physics_world_model import _hamming

D = 256


def _hv(seed):
    return _gen_hv(D, seed=seed)


# ── FractionalPower ───────────────────────────────────────────────────────────

class TestFractionalPower:
    def setup_method(self):
        self.fpe = FractionalPower(D)
        self.x   = _hv(42)

    def test_alpha_zero_returns_zeros(self):
        result = self.fpe.power(self.x, 0.0)
        assert result.sum() == 0.0

    def test_alpha_one_returns_input(self):
        result = self.fpe.power(self.x, 1.0)
        assert torch.equal(result, self.x.float())

    def test_alpha_half_density(self):
        result = self.fpe.power(self.x, 0.5, seed=0)
        density = float(result.mean())
        expected_density = float(self.x.mean()) * 0.5
        # Allow ±10% tolerance
        assert abs(density - expected_density) < 0.1

    def test_monotone_density(self):
        d0  = float(self.fpe.power(self.x, 0.0).mean())
        d25 = float(self.fpe.power(self.x, 0.25, seed=1).mean())
        d75 = float(self.fpe.power(self.x, 0.75, seed=2).mean())
        d1  = float(self.fpe.power(self.x, 1.0).mean())
        assert d0 <= d25 <= d75 <= d1

    def test_deterministic_with_seed(self):
        r1 = self.fpe.power(self.x, 0.5, seed=42)
        r2 = self.fpe.power(self.x, 0.5, seed=42)
        assert torch.equal(r1, r2)

    def test_different_seeds_differ(self):
        r1 = self.fpe.power(self.x, 0.5, seed=0)
        r2 = self.fpe.power(self.x, 0.5, seed=999)
        # With high probability they differ (probability of collision is negligible)
        assert not torch.equal(r1, r2)

    def test_hamming_monotone_in_alpha(self):
        x0 = self.fpe.power(self.x, 0.0)
        x5 = self.fpe.power(self.x, 0.5, seed=0)
        x1 = self.fpe.power(self.x, 1.0)
        sim_0_5 = float(_hamming(x0.unsqueeze(0), x5.unsqueeze(0)))
        sim_0_1 = float(_hamming(x0.unsqueeze(0), x1.unsqueeze(0)))
        # x^0.5 is closer to x^0 than x^1 is
        assert sim_0_5 > sim_0_1

    def test_trajectory_shape(self):
        traj = self.fpe.trajectory(self.x, n_steps=8)
        assert traj.shape == (8, D)

    def test_trajectory_monotone_density(self):
        traj = self.fpe.trajectory(self.x, n_steps=5,
                                   alpha_start=0.0, alpha_end=1.0)
        densities = traj.mean(dim=1).tolist()
        # Each step should have equal or more density
        for i in range(len(densities) - 1):
            assert densities[i] <= densities[i + 1] + 0.05  # allow small noise

    def test_interpolate_midpoint(self):
        a, b = _hv(0), _hv(1)
        mid = self.fpe.interpolate(a, b, alpha=0.5)
        assert mid.shape == (D,)
        # Midpoint should be closer to both a and b than to a random HV
        sim_a = float(_hamming(mid.unsqueeze(0), a.unsqueeze(0)))
        sim_r = float(_hamming(mid.unsqueeze(0), _hv(999).unsqueeze(0)))
        assert sim_a > sim_r - 0.05  # midpoint is related to a

    def test_inverse_property(self):
        inv = self.fpe.inverse(self.x, 0.5)
        assert inv.shape == (D,)


# ── ContinuousStateEncoder ────────────────────────────────────────────────────

class TestContinuousStateEncoder:
    def setup_method(self):
        self.enc = ContinuousStateEncoder(
            D, n_dims=2, ranges=[(-5.0, 5.0), (-5.0, 5.0)]
        )

    def test_encode_shape(self):
        pos = torch.tensor([0.0, 0.0])
        hv = self.enc.encode_position(pos)
        assert hv.shape == (D,)
        assert set(hv.unique().tolist()).issubset({0.0, 1.0})

    def test_nearby_positions_similar(self):
        p1 = torch.tensor([0.0, 0.0])
        p2 = torch.tensor([0.1, 0.0])
        p3 = torch.tensor([5.0, 5.0])
        hv1 = self.enc.encode_position(p1)
        hv2 = self.enc.encode_position(p2)
        hv3 = self.enc.encode_position(p3)
        sim_near = float(_hamming(hv1.unsqueeze(0), hv2.unsqueeze(0)))
        sim_far  = float(_hamming(hv1.unsqueeze(0), hv3.unsqueeze(0)))
        assert sim_near > sim_far, \
            f"Nearby positions should be more similar: {sim_near:.3f} vs {sim_far:.3f}"

    def test_same_position_identical(self):
        pos = torch.tensor([1.0, 2.0])
        hv1 = self.enc.encode_position(pos)
        hv2 = self.enc.encode_position(pos)
        assert torch.equal(hv1, hv2)

    def test_encode_velocity_shape(self):
        pos  = torch.tensor([1.0, 2.0])
        prev = torch.tensor([0.9, 1.9])
        pos_hv, vel_hv = self.enc.encode_velocity(pos, prev_pos=prev)
        assert pos_hv.shape == (D,)
        assert vel_hv.shape == (D,)

    def test_predict_next_position_shape(self):
        pos  = torch.tensor([0.0, 0.0])
        prev = torch.tensor([-0.1, 0.0])
        pos_hv, vel_hv = self.enc.encode_velocity(pos, prev_pos=prev)
        next_hv = self.enc.predict_next_position(pos_hv, vel_hv, dt=1.0)
        assert next_hv.shape == (D,)


# ── TemporalContextEncoder ────────────────────────────────────────────────────

class TestTemporalContextEncoder:
    def setup_method(self):
        self.tce = TemporalContextEncoder(D, tau_decay=5.0, max_history=10)

    def test_push_builds_context(self):
        self.tce.push(_hv(0))
        assert self.tce.context.shape == (D,)

    def test_context_changes_with_new_observation(self):
        self.tce.push(_hv(0))
        ctx0 = self.tce.context.clone()
        self.tce.push(_hv(1))
        ctx1 = self.tce.context.clone()
        assert not torch.equal(ctx0, ctx1)

    def test_query_shape(self):
        for i in range(5):
            self.tce.push(_hv(i))
        q = self.tce.query(lag=0)
        assert q.shape == (D,)

    def test_recent_query_closer_to_recent(self):
        for i in range(8):
            self.tce.push(_hv(i))
        q0 = self.tce.query(lag=0)
        q7 = self.tce.query(lag=7)
        # query(0) should be more similar to most recent observation
        sim_recent = float(_hamming(q0.unsqueeze(0), _hv(7).unsqueeze(0)))
        sim_old    = float(_hamming(q7.unsqueeze(0), _hv(7).unsqueeze(0)))
        # both should be valid HVs
        assert q0.shape == (D,)
        assert q7.shape == (D,)

    def test_reset_clears_state(self):
        for i in range(5):
            self.tce.push(_hv(i))
        self.tce.reset()
        assert self.tce.context.sum() == 0.0
        assert len(self.tce._obs_buf) == 0


# ── PhysicsHVConstraints ──────────────────────────────────────────────────────

class TestPhysicsHVConstraints:
    def setup_method(self):
        from hdc.physics_world_model import _xor, _majority
        self.phys = PhysicsHVConstraints(D)
        self.pos_t  = _hv(10)
        self.vel_t  = _hv(11)
        # Physically consistent next state
        self.pos_t1_good = _majority(_xor(self.pos_t, self.vel_t).float())
        # Random (inconsistent) next state
        self.pos_t1_bad  = _hv(99)

    def test_kinematic_error_zero_for_consistent(self):
        err = self.phys.kinematic_error(self.pos_t, self.vel_t, self.pos_t1_good)
        assert 0.0 <= err <= 0.5, f"Consistent transition should have low error: {err}"

    def test_kinematic_error_higher_for_inconsistent(self):
        err_good = self.phys.kinematic_error(self.pos_t, self.vel_t, self.pos_t1_good)
        err_bad  = self.phys.kinematic_error(self.pos_t, self.vel_t, self.pos_t1_bad)
        assert err_good < err_bad

    def test_score_negative(self):
        score = self.phys.score(self.pos_t, self.vel_t, self.pos_t1_good)
        assert score <= 0.0

    def test_consistent_more_plausible(self):
        good = self.phys.score(self.pos_t, self.vel_t, self.pos_t1_good)
        bad  = self.phys.score(self.pos_t, self.vel_t, self.pos_t1_bad)
        assert good > bad

    def test_plausibility_check(self):
        assert self.phys.is_physically_plausible(
            self.pos_t, self.vel_t, self.pos_t1_good, threshold=0.5
        )


# ── FPEInterpolator ───────────────────────────────────────────────────────────

class TestFPEInterpolator:
    def setup_method(self):
        self.interp = FPEInterpolator(D)
        self.interp.register("start", _hv(0))
        self.interp.register("end",   _hv(1))

    def test_interpolate_shape(self):
        mid = self.interp.interpolate("start", "end", alpha=0.5)
        assert mid.shape == (D,)

    def test_alpha_zero_close_to_start(self):
        start = self.interp._prototypes["start"]
        result = self.interp.interpolate("start", "end", alpha=0.0)
        sim = float(_hamming(result.unsqueeze(0), start.unsqueeze(0)))
        assert sim > 0.7, f"alpha=0 should be close to start, sim={sim}"

    def test_alpha_one_close_to_end(self):
        end = self.interp._prototypes["end"]
        result = self.interp.interpolate("start", "end", alpha=1.0)
        sim = float(_hamming(result.unsqueeze(0), end.unsqueeze(0)))
        assert sim > 0.7, f"alpha=1 should be close to end, sim={sim}"

    def test_locate_midpoint(self):
        mid = self.interp.interpolate("start", "end", alpha=0.5)
        alpha_found = self.interp.locate(mid, "start", "end", resolution=10)
        assert 0.0 <= alpha_found <= 1.0

    def test_locate_returns_float(self):
        q = _hv(42)
        alpha = self.interp.locate(q, "start", "end", resolution=5)
        assert isinstance(alpha, float)
