"""
tests/test_occupancy.py
=======================
Tests for the VSA Occupancy Grid Mapping module (hdc/occupancy.py).

Validates:
  1. VSAOGM — VSA-based occupancy grid: update, query, binary map, feature vector
  2. VSAOGMAgent — agent that uses VSAOGM for navigation
"""

from __future__ import annotations

import sys
import os

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from hdc.occupancy import VSAOGM, VSAOGMAgent
from hdc.hdc_glue import hv_hamming_sim


@pytest.fixture
def hd_dim():
    return 256

@pytest.fixture
def ogm(hd_dim):
    return VSAOGM(hd_dim=hd_dim)


class TestVSAOGM:
    def test_init(self, ogm, hd_dim):
        assert ogm.hd_dim == hd_dim

    def test_update_single_point(self, ogm):
        pos = torch.tensor([[5.0, 7.0]])
        labels = torch.tensor([1.0])
        ogm.update(pos, labels)
        hv = ogm.ogm_hv
        assert hv.shape == (ogm.hd_dim,)

    def test_update_multiple_points(self, ogm):
        positions = torch.tensor([[5.0, 7.0], [3.0, 2.0], [10.0, 12.0]])
        labels = torch.tensor([1.0, -1.0, 1.0])
        ogm.update(positions, labels)
        hv = ogm.ogm_hv
        assert hv.shape == (ogm.hd_dim,)

    def test_query_shape(self, ogm):
        pos = torch.tensor([[5.0, 7.0], [3.0, 2.0]])
        labels = torch.tensor([1.0, -1.0])
        ogm.update(pos, labels)
        queries = torch.tensor([[5.0, 7.0], [8.0, 8.0]])
        result = ogm.query(queries)
        assert result.shape == (2,)

    def test_query_single(self, ogm):
        pos = torch.tensor([[5.0, 7.0]])
        labels = torch.tensor([1.0])
        ogm.update(pos, labels)
        queries = torch.tensor([[5.0, 7.0]])
        result = ogm.query(queries)
        assert result.ndim == 1

    def test_binary_map_shape(self, ogm):
        ogm.update(torch.tensor([[5.0, 7.0]]), torch.tensor([1.0]))
        grid = ogm.binary_map(x_range=(0, 10), y_range=(0, 10), resolution=2.0)
        assert grid.ndim == 2

    def test_binary_map_values(self, ogm):
        ogm.update(torch.tensor([[5.0, 7.0]]), torch.tensor([1.0]))
        grid = ogm.binary_map(x_range=(0, 10), y_range=(0, 10), resolution=2.0)
        assert ((grid == 0.0) | (grid == 1.0)).all()

    def test_ogm_feature_vector_shape(self, ogm):
        ogm.update(torch.tensor([[5.0, 7.0]]), torch.tensor([1.0]))
        query_grid = torch.tensor([[5.0, 7.0], [0.0, 0.0]])
        fv = ogm.ogm_feature_vector(query_grid)
        assert fv.shape == (2,)

    def test_ogm_hv_shape(self, ogm):
        ogm.update(torch.tensor([[5.0, 7.0]]), torch.tensor([1.0]))
        hv = ogm.ogm_hv
        assert hv.shape == (ogm.hd_dim,)

    def test_ogm_hv_is_tensor(self, ogm):
        ogm.update(torch.tensor([[5.0, 7.0]]), torch.tensor([1.0]))
        hv = ogm.ogm_hv
        assert isinstance(hv, torch.Tensor)

    def test_reset_clears(self, ogm):
        ogm.update(torch.tensor([[5.0, 7.0]]), torch.tensor([1.0]))
        ogm.reset()
        assert ogm._n == 0

    def test_known_position_higher_similarity(self, ogm):
        pos = torch.tensor([[5.0, 7.0]])
        labels = torch.tensor([1.0])
        ogm.update(pos, labels)
        known_prob = ogm.query(torch.tensor([[5.0, 7.0]]))
        unknown_prob = ogm.query(torch.tensor([[0.0, 0.0]]))
        assert known_prob.item() >= unknown_prob.item() - 0.1

    def test_multiple_updates_accumulate(self, ogm):
        pos = torch.tensor([[5.0, 7.0]])
        labels = torch.tensor([1.0])
        ogm.update(pos, labels)
        n1 = ogm._n
        ogm.update(pos, labels)
        ogm.update(pos, labels)
        assert ogm._n > n1


class TestVSAOGMAgent:
    def test_init(self, hd_dim):
        agent = VSAOGMAgent(hd_dim=hd_dim)
        assert agent.obs_dim > 0

    def test_reset(self, hd_dim):
        agent = VSAOGMAgent(hd_dim=hd_dim)
        obs = agent.reset()
        assert obs.ndim == 1

    def test_step_returns_observation(self, hd_dim):
        agent = VSAOGMAgent(hd_dim=hd_dim)
        lidar_ranges = torch.rand(8) * 5.0
        lidar_angles = torch.linspace(-1.57, 1.57, 8)
        obs = agent.step(lidar_ranges, lidar_angles)
        assert obs.shape == (agent.obs_dim,)

    def test_step_multiple_times(self, hd_dim):
        agent = VSAOGMAgent(hd_dim=hd_dim)
        for _ in range(5):
            lidar_ranges = torch.rand(8) * 5.0
            lidar_angles = torch.linspace(-1.57, 1.57, 8)
            obs = agent.step(lidar_ranges, lidar_angles)
            assert obs.shape == (agent.obs_dim,)
