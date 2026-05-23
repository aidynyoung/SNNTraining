"""
tests/test_analogy.py
======================
Tests for VSA Analogical Reasoning (hdc/analogy.py).

Validates:
  1. AnalogicalReasoner — A:B :: C:? solving in VSA space
  2. ConceptMap — structured physical knowledge graph
  3. ScenarioTransfer — zero-shot knowledge transfer across scenarios
"""

from __future__ import annotations

import sys
import os

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from hdc.analogy import (
    AnalogicalReasoner,
    AnalogyResult,
    ConceptMap,
    ScenarioTransfer,
    TransferResult,
)
from hdc.physics_world_model import _xor, _hamming, _majority
from hdc.hdc_glue import hv_batch_sim


@pytest.fixture
def hd_dim():
    return 256


class TestAnalogicalReasoner:
    def test_init(self, hd_dim):
        reasoner = AnalogicalReasoner(hd_dim)
        assert reasoner.hd_dim == hd_dim

    def test_register_concept(self, hd_dim):
        reasoner = AnalogicalReasoner(hd_dim)
        hv = (torch.rand(hd_dim) >= 0.5).float()
        reasoner.register_concept("test", hv)
        assert "test" in reasoner._codebook

    def test_relational_hv(self, hd_dim):
        reasoner = AnalogicalReasoner(hd_dim)
        a = (torch.rand(hd_dim) >= 0.5).float()
        b = (torch.rand(hd_dim) >= 0.5).float()
        rel = reasoner.relational_hv(a, b)
        assert rel.shape == (hd_dim,)

    def test_solve_returns_analogy_result(self, hd_dim):
        reasoner = AnalogicalReasoner(hd_dim)
        a = (torch.rand(hd_dim) >= 0.5).float()
        b = (torch.rand(hd_dim) >= 0.5).float()
        c = (torch.rand(hd_dim) >= 0.5).float()
        result = reasoner.solve(a, b, c, cleanup=False)
        assert isinstance(result, AnalogyResult)
        assert result.query_hv.shape == (hd_dim,)

    def test_solve_with_cleanup(self, hd_dim):
        reasoner = AnalogicalReasoner(hd_dim)
        a = (torch.rand(hd_dim) >= 0.5).float()
        b = (torch.rand(hd_dim) >= 0.5).float()
        c = (torch.rand(hd_dim) >= 0.5).float()
        d = (torch.rand(hd_dim) >= 0.5).float()
        reasoner.register_concept("d", d)
        result = reasoner.solve(a, b, c, cleanup=True)
        assert isinstance(result, AnalogyResult)

    def test_solve_multi(self, hd_dim):
        reasoner = AnalogicalReasoner(hd_dim)
        a = (torch.rand(hd_dim) >= 0.5).float()
        b = (torch.rand(hd_dim) >= 0.5).float()
        c = (torch.rand(hd_dim) >= 0.5).float()
        d = (torch.rand(hd_dim) >= 0.5).float()
        e = (torch.rand(hd_dim) >= 0.5).float()
        result = reasoner.solve_multi([(a, b), (c, d)], e, cleanup=False)
        assert isinstance(result, AnalogyResult)

    def test_similarity_to(self, hd_dim):
        reasoner = AnalogicalReasoner(hd_dim)
        hv = (torch.rand(hd_dim) >= 0.5).float()
        reasoner.register_concept("test", hv)
        sim = reasoner.similarity_to(hv, "test")
        assert sim > 0.5

    def test_find_analogous_pairs(self, hd_dim):
        reasoner = AnalogicalReasoner(hd_dim)
        a = (torch.rand(hd_dim) >= 0.5).float()
        b = (torch.rand(hd_dim) >= 0.5).float()
        c = (torch.rand(hd_dim) >= 0.5).float()
        d = (torch.rand(hd_dim) >= 0.5).float()
        reasoner.register_concept("a", a)
        reasoner.register_concept("b", b)
        reasoner.register_concept("c", c)
        reasoner.register_concept("d", d)
        rel = reasoner.relational_hv(a, b)
        pairs = reasoner.find_analogous_pairs(rel, top_k=3, similarity_threshold=0.0)
        assert isinstance(pairs, list)


class TestConceptMap:
    def test_init(self, hd_dim):
        cmap = ConceptMap(hd_dim)
        assert cmap.hd_dim == hd_dim
        assert cmap.n_concepts == 0

    def test_add_concept(self, hd_dim):
        cmap = ConceptMap(hd_dim)
        cmap.add_concept("test")
        assert cmap.n_concepts == 1
        assert "test" in cmap.concept_names

    def test_add_from_observation(self, hd_dim):
        cmap = ConceptMap(hd_dim)
        hv = (torch.rand(hd_dim) >= 0.5).float()
        cmap.add_from_observation("observed", hv)
        assert cmap.n_concepts == 1

    def test_query_analogy(self, hd_dim):
        cmap = ConceptMap(hd_dim)
        cmap.add_concept("a")
        cmap.add_concept("b")
        cmap.add_concept("c")
        result = cmap.query_analogy("a", "b", "c")
        assert result is not None
        assert isinstance(result, AnalogyResult)

    def test_transfer_relation(self, hd_dim):
        cmap = ConceptMap(hd_dim)
        cmap.add_concept("a")
        cmap.add_concept("b")
        cmap.add_concept("c")
        result = cmap.transfer_relation("a", "b", "c", "d")
        assert result is not None
        assert result.shape == (hd_dim,)
        assert "d" in cmap.concept_names

    def test_find_all_analogies(self, hd_dim):
        cmap = ConceptMap(hd_dim)
        cmap.add_concept("a", related_to="b", relation_type="test_rel")
        cmap.add_concept("b")
        cmap.add_concept("c")
        results = cmap.find_all_analogies("test_rel", ["c"])
        assert isinstance(results, list)

    def test_concept_hv(self, hd_dim):
        cmap = ConceptMap(hd_dim)
        cmap.add_concept("test")
        hv = cmap.concept_hv("test")
        assert hv is not None
        assert hv.shape == (hd_dim,)

    def test_concept_hv_missing(self, hd_dim):
        cmap = ConceptMap(hd_dim)
        hv = cmap.concept_hv("nonexistent")
        assert hv is None


class TestScenarioTransfer:
    def test_init(self, hd_dim):
        from hdc.physics_world_model import PhysicsWorldModel
        src_map = ConceptMap(hd_dim)
        src_wm = PhysicsWorldModel(hd_dim=hd_dim)
        tgt_wm = PhysicsWorldModel(hd_dim=hd_dim)
        transfer = ScenarioTransfer(src_map, src_wm, tgt_wm)
        assert transfer.threshold == 0.55

    def test_register_anchor(self, hd_dim):
        from hdc.physics_world_model import PhysicsWorldModel
        src_map = ConceptMap(hd_dim)
        src_wm = PhysicsWorldModel(hd_dim=hd_dim)
        tgt_wm = PhysicsWorldModel(hd_dim=hd_dim)
        transfer = ScenarioTransfer(src_map, src_wm, tgt_wm)
        hv = (torch.rand(hd_dim) >= 0.5).float()
        transfer.register_anchor("normal_src", hv, "normal_tgt")
        assert "normal_tgt" in transfer.target_map.concept_names

    def test_transfer_prototypes(self, hd_dim):
        from hdc.physics_world_model import PhysicsWorldModel
        src_map = ConceptMap(hd_dim)
        src_wm = PhysicsWorldModel(hd_dim=hd_dim)
        tgt_wm = PhysicsWorldModel(hd_dim=hd_dim)
        transfer = ScenarioTransfer(src_map, src_wm, tgt_wm)
        # Register source concepts
        src_map.add_concept("normal_src")
        src_map.add_concept("fault_src")
        # Register anchor in target
        hv = (torch.rand(hd_dim) >= 0.5).float()
        transfer.register_anchor("normal_src", hv, "normal_tgt")
        # Transfer
        result = transfer.transfer_prototypes("normal_src", "normal_tgt")
        assert isinstance(result, TransferResult)

    def test_transfer_confidence(self, hd_dim):
        from hdc.physics_world_model import PhysicsWorldModel
        src_map = ConceptMap(hd_dim)
        src_wm = PhysicsWorldModel(hd_dim=hd_dim)
        tgt_wm = PhysicsWorldModel(hd_dim=hd_dim)
        transfer = ScenarioTransfer(src_map, src_wm, tgt_wm)
        src_map.add_concept("normal_src")
        src_map.add_concept("fault_src")
        hv = (torch.rand(hd_dim) >= 0.5).float()
        transfer.register_anchor("normal_src", hv, "normal_tgt")
        result = transfer.transfer_prototypes("normal_src", "normal_tgt")
        assert 0.0 <= result.transfer_confidence <= 1.0

    def test_transfer_causal_rules(self, hd_dim):
        from hdc.physics_world_model import PhysicsWorldModel
        src_map = ConceptMap(hd_dim)
        src_wm = PhysicsWorldModel(hd_dim=hd_dim)
        tgt_wm = PhysicsWorldModel(hd_dim=hd_dim)
        transfer = ScenarioTransfer(src_map, src_wm, tgt_wm)
        src_map.add_concept("normal_src")
        src_map.add_concept("fault_src")
        hv = (torch.rand(hd_dim) >= 0.5).float()
        transfer.register_anchor("normal_src", hv, "normal_tgt")
        result = transfer.transfer_prototypes("normal_src", "normal_tgt")
        assert result.n_causal_rules_transferred >= 0
