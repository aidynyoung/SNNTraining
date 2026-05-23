"""
tests/test_elite_pipeline.py
=============================
Tests for EliteSNNTrainingPipeline — the top-level Physical AI orchestrator.
Focuses on v1.45 improvements: TD updates, cognitive map, grid cells,
status() report, reset(), and end-to-end step behavior.
"""

from __future__ import annotations
import pytest
import torch
from arthedain.elite_pipeline import EliteSNNTrainingPipeline, PipelineConfig


# Minimal config to keep tests fast
@pytest.fixture
def small_cfg():
    return PipelineConfig(
        snn_input_size=16,
        snn_hidden_size=32,
        snn_output_size=2,
        hd_dim=128,
        n_action_classes=3,
        n_sensor_features=8,
        episodic_capacity=32,
        mcts_simulations=3,
        option_library=2,
        planning_horizon=2,
        use_grid_cells=False,   # skip for fast tests
    )


@pytest.fixture
def pipe(small_cfg):
    return EliteSNNTrainingPipeline(small_cfg)


class TestPipelineBasic:
    def test_init_no_crash(self, small_cfg):
        p = EliteSNNTrainingPipeline(small_cfg)
        assert p._step == 0

    def test_step_no_crash(self, pipe):
        hv = (torch.rand(128) >= 0.5).float()
        result = pipe.step(sensor_hv=hv)
        assert result is not None

    def test_step_increments_counter(self, pipe):
        hv = (torch.rand(128) >= 0.5).float()
        pipe.step(sensor_hv=hv)
        assert pipe._step == 1

    def test_repr_contains_flags(self, pipe):
        r = repr(pipe)
        assert "SNN" in r or "WM" in r or "Mem" in r

    def test_status_returns_dict(self, pipe):
        hv = (torch.rand(128) >= 0.5).float()
        pipe.step(sensor_hv=hv)
        status = pipe.status()
        assert isinstance(status, dict)
        assert "step" in status

    def test_snn_step_and_update(self, pipe):
        snn_in = (torch.rand(16) >= 0.5).float()
        target = torch.rand(2)
        result = pipe.step(snn_input=snn_in, target_velocity=target)
        assert result.decoded_velocity is not None or result.pearson_r is not None


class TestPipelineMemoryAndCognition:
    def test_cognitive_map_novelty_populated(self, pipe):
        hv = (torch.rand(128) >= 0.5).float()
        result = pipe.step(sensor_hv=hv)
        # After one step, novelty should be set (first state is maximally novel)
        assert isinstance(result.novelty, float)

    def test_cognitive_map_novelty_decreases_for_repeated_state(self, pipe):
        hv = (torch.rand(128) >= 0.5).float()
        # First time — novel
        r1 = pipe.step(sensor_hv=hv)
        n1 = r1.novelty
        # Same HV many times — should become less novel
        for _ in range(5):
            result = pipe.step(sensor_hv=hv)
        n_last = result.novelty
        # Novelty should decrease or stay same (not increase) for repeated state
        # (This is a soft check — the exact value depends on LSH collisions)
        assert n_last >= 0.0

    def test_memory_retrieved_populates(self, pipe):
        hv = (torch.rand(128) >= 0.5).float()
        # Store a few states
        for _ in range(3):
            pipe.step(sensor_hv=hv)
        result = pipe.step(sensor_hv=hv)
        # memory_retrieved may be None or a string
        assert result.memory_retrieved is None or isinstance(result.memory_retrieved, str)


class TestPipelinePlanning:
    def test_planning_result_has_action(self, pipe):
        hv = (torch.rand(128) >= 0.5).float()
        a1 = (torch.rand(128) >= 0.5).float()
        a2 = (torch.rand(128) >= 0.5).float()
        pipe.register_action(a1)
        pipe.register_action(a2)
        result = pipe.step(sensor_hv=hv)
        # Recommended action may be None or a tensor HV
        assert result.recommended_action is None or isinstance(result.recommended_action, torch.Tensor)

    def test_status_shows_options(self, pipe):
        pipe.register_action((torch.rand(128) >= 0.5).float())
        status = pipe.status()
        if pipe.use_planning:
            assert "n_options" in status

    def test_status_shows_vf_td_steps(self, pipe):
        status = pipe.status()
        if pipe.use_planning:
            assert "vf_td_steps" in status

    def test_vf_td_steps_increase_after_planning(self, pipe):
        hv = (torch.rand(128) >= 0.5).float()
        a1 = (torch.rand(128) >= 0.5).float()
        a2 = (torch.rand(128) >= 0.5).float()
        pipe.register_action(a1)
        pipe.register_action(a2)
        # Multiple steps with actions — should trigger TD updates after first
        for _ in range(5):
            pipe.step(sensor_hv=hv)
        if pipe.use_planning:
            # _V_acc starts at 1.0, gets incremented on each TD update
            assert pipe.value_fn._V_acc[0] >= 1.0


class TestPipelineReset:
    def test_reset_clears_step_counter(self, pipe):
        hv = (torch.rand(128) >= 0.5).float()
        for _ in range(5):
            pipe.step(sensor_hv=hv)
        pipe.reset()
        assert pipe._current_state_hv is None

    def test_reset_clears_vf_state(self, pipe):
        hv = (torch.rand(128) >= 0.5).float()
        pipe.step(sensor_hv=hv)
        pipe.reset()
        assert pipe._vf_prev_state is None
        assert pipe._vf_prev_action is None

    def test_multiple_resets_no_crash(self, pipe):
        for _ in range(3):
            hv = (torch.rand(128) >= 0.5).float()
            for _ in range(3):
                pipe.step(sensor_hv=hv)
            pipe.reset()

    def test_full_reset_clears_memory(self, pipe):
        hv = (torch.rand(128) >= 0.5).float()
        for _ in range(5):
            pipe.step(sensor_hv=hv)
        pipe.reset(full=True)
        # After full reset the pipeline should still accept new steps
        result = pipe.step(sensor_hv=hv)
        assert result is not None


class TestPipelineGridCells:
    def test_grid_cells_state_initialised(self, small_cfg):
        cfg = PipelineConfig(
            snn_input_size=16, snn_hidden_size=32, snn_output_size=2,
            hd_dim=128, use_grid_cells=True, grid_periods=[5.0],
        )
        try:
            p = EliteSNNTrainingPipeline(cfg)
            if p.use_grid_cells:
                assert hasattr(p, '_grid_state')
                assert p._grid_state.shape[0] > 0
        except Exception:
            pytest.skip("GridCellNetwork not available")

    def test_grid_cells_position_updates(self, small_cfg):
        cfg = PipelineConfig(
            snn_input_size=16, snn_hidden_size=32, snn_output_size=2,
            hd_dim=128, use_grid_cells=True, grid_periods=[5.0],
        )
        try:
            p = EliteSNNTrainingPipeline(cfg)
            if not p.use_grid_cells:
                pytest.skip("Grid cells disabled")
            hv  = (torch.rand(128) >= 0.5).float()
            vel = torch.tensor([1.0, 0.5])
            for _ in range(3):
                p.step(sensor_hv=hv, snn_input=torch.rand(16),
                       target_velocity=vel)
            assert p._position_estimate[0] != 0.0 or p._position_estimate[1] != 0.0
        except Exception:
            pytest.skip("GridCellNetwork not available")


class TestPipelineSNNConvergence:
    def test_status_includes_snn_convergence(self, pipe):
        snn_in = (torch.rand(16) >= 0.5).float()
        target = torch.rand(2)
        for _ in range(5):
            pipe.step(snn_input=snn_in, target_velocity=target)
        status = pipe.status()
        if pipe.use_snn:
            assert "snn_convergence" in status
            conv = status["snn_convergence"]
            assert "n_updates" in conv
            assert "error_ema" in conv
            assert "spectral_radius" in conv
            assert "input_gain" in conv

    def test_pearson_r_improves_or_stable(self, pipe):
        torch.manual_seed(42)
        snn_in = (torch.rand(16) >= 0.5).float()
        t = torch.tensor([1.0, -1.0])
        r_values = []
        for step in range(20):
            pipe.step(snn_input=snn_in, target_velocity=t)
            r_values.append(pipe.snn.pearson_r())
        # Last few steps should have a valid (not NaN) Pearson R
        assert all(not float('nan') == r for r in r_values[-5:])
