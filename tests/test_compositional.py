"""
tests/test_compositional.py
=============================
Tests for RoleFillerCodebook, CompositionalHDCClassifier,
StructuredAnalogy, CompositionalWorldModel.
"""
import pytest
import torch
from hdc.compositional import (
    RoleFillerCodebook,
    CompositionalHDCClassifier,
    StructuredAnalogy,
    CompositionalWorldModel,
)

D = 256


# ── RoleFillerCodebook ────────────────────────────────────────────────────────

class TestRoleFillerCodebook:
    def setup_method(self):
        self.cb = RoleFillerCodebook(D)
        self.cb.register_role("color")
        self.cb.register_role("shape")
        for c in ["red", "blue", "green"]:
            self.cb.register_filler("color", c)
        for s in ["square", "circle", "triangle"]:
            self.cb.register_filler("shape", s)

    def test_roles_registered(self):
        assert "color" in self.cb.roles
        assert "shape" in self.cb.roles
        assert self.cb.n_roles == 2

    def test_encode_slot_shape(self):
        slot_hv = self.cb.encode_slot("color", "red")
        assert slot_hv.shape == (D,)

    def test_encode_slot_binary(self):
        slot_hv = self.cb.encode_slot("color", "red")
        assert set(slot_hv.unique().tolist()).issubset({0.0, 1.0})

    def test_encode_object_shape(self):
        obj = {"color": "red", "shape": "circle"}
        hv = self.cb.encode_object(obj)
        assert hv.shape == (D,)

    def test_decode_filler_returns_tuple(self):
        hv = self.cb.encode_object({"color": "red", "shape": "square"})
        filler, sim = self.cb.decode_filler(hv, "color")
        assert isinstance(filler, str)
        assert 0.0 <= sim <= 1.0

    def test_decode_roundtrip_color(self):
        hv = self.cb.encode_object({"color": "blue", "shape": "circle"})
        color, sim = self.cb.decode_filler(hv, "color")
        assert color == "blue", f"Expected 'blue', got '{color}' (sim={sim:.3f})"

    def test_decode_roundtrip_shape(self):
        hv = self.cb.encode_object({"color": "green", "shape": "triangle"})
        shape, sim = self.cb.decode_filler(hv, "shape")
        assert shape == "triangle", f"Expected 'triangle', got '{shape}' (sim={sim:.3f})"

    def test_decode_all_returns_all_roles(self):
        hv  = self.cb.encode_object({"color": "red", "shape": "square"})
        dec = self.cb.decode_all(hv)
        assert "color" in dec
        assert "shape" in dec

    def test_different_objects_differ(self):
        hv1 = self.cb.encode_object({"color": "red",  "shape": "square"})
        hv2 = self.cb.encode_object({"color": "blue", "shape": "circle"})
        from hdc.physics_world_model import _hamming
        sim = float(_hamming(hv1.unsqueeze(0), hv2.unsqueeze(0)))
        assert sim < 0.9  # should not be identical

    def test_missing_role_raises(self):
        with pytest.raises(KeyError):
            self.cb.encode_slot("nonexistent_role", "value")

    def test_missing_filler_raises(self):
        with pytest.raises(KeyError):
            self.cb.encode_slot("color", "nonexistent_color")

    def test_auto_register_role_in_encode_filler(self):
        self.cb.register_filler("size", "large")
        assert "size" in self.cb.roles


# ── CompositionalHDCClassifier ────────────────────────────────────────────────

class TestCompositionalHDCClassifier:
    def setup_method(self):
        self.cb = RoleFillerCodebook(D)
        self.cb.register_role("color")
        self.cb.register_role("shape")
        for c in ["red", "blue"]:
            self.cb.register_filler("color", c)
        for s in ["square", "circle"]:
            self.cb.register_filler("shape", s)
        self.clf = CompositionalHDCClassifier(
            self.cb, n_classes=2, class_names=["danger", "safe"]
        )

    def test_train_step_no_error(self):
        self.clf.train_step({"color": "red", "shape": "square"}, label=0)
        self.clf.train_step({"color": "blue", "shape": "circle"}, label=1)

    def test_predict_returns_class_idx(self):
        self.clf.train_step({"color": "red", "shape": "square"}, label=0)
        self.clf.train_step({"color": "blue", "shape": "circle"}, label=1)
        idx, name, scores = self.clf.predict({"color": "red", "shape": "square"})
        assert 0 <= idx < 2
        assert name in ("danger", "safe")
        assert len(scores) == 2

    def test_zero_shot_red_is_danger(self):
        self.clf.train_step({"color": "red",  "shape": "square"}, label=0)
        self.clf.train_step({"color": "blue", "shape": "circle"}, label=1)
        # {red, circle} not in training — should predict "danger" (red)
        idx, name, scores = self.clf.predict({"color": "red", "shape": "circle"})
        assert name == "danger", \
            f"Zero-shot: red should pull toward danger, got {name}"

    def test_predict_hv(self):
        self.clf.train_step({"color": "red", "shape": "square"}, label=0)
        hv  = self.cb.encode_object({"color": "red", "shape": "square"})
        idx, name, scores = self.clf.predict_hv(hv)
        assert 0 <= idx < 2

    def test_decompose_and_explain_keys(self):
        self.clf.train_step({"color": "red", "shape": "square"}, label=0)
        explanation = self.clf.decompose_and_explain(
            {"color": "red", "shape": "circle"}
        )
        assert "predicted_class" in explanation
        assert "confidence" in explanation
        assert "role_contributions" in explanation


# ── StructuredAnalogy ─────────────────────────────────────────────────────────

class TestStructuredAnalogy:
    def setup_method(self):
        self.cb = RoleFillerCodebook(D)
        self.cb.register_role("color")
        self.cb.register_role("shape")
        self.cb.register_role("size")
        for c in ["red", "blue", "green"]:
            self.cb.register_filler("color", c)
        for s in ["square", "circle", "triangle"]:
            self.cb.register_filler("shape", s)
        for sz in ["large", "small"]:
            self.cb.register_filler("size", sz)
        self.analogy = StructuredAnalogy(self.cb)

    def test_solve_returns_tuple(self):
        A = {"color": "red",  "shape": "square", "size": "large"}
        B = {"color": "blue", "shape": "square", "size": "large"}
        C = {"color": "red",  "shape": "circle", "size": "small"}
        D_slots, D_hv, conf = self.analogy.solve(A, B, C)
        assert isinstance(D_slots, dict)
        assert D_hv.shape == (D,)
        assert 0.0 <= conf <= 1.0

    def test_color_transfer(self):
        # A:B changes only color red→blue; C has red → D should have blue
        A = {"color": "red",  "shape": "square", "size": "large"}
        B = {"color": "blue", "shape": "square", "size": "large"}
        C = {"color": "red",  "shape": "circle", "size": "small"}
        D_slots, _, _ = self.analogy.solve(A, B, C)
        assert D_slots.get("color") == "blue"

    def test_unchanged_roles_preserved(self):
        # Shape doesn't change in A:B, so D should keep C's shape
        A = {"color": "red",  "shape": "square", "size": "large"}
        B = {"color": "blue", "shape": "square", "size": "large"}
        C = {"color": "red",  "shape": "circle", "size": "small"}
        D_slots, _, _ = self.analogy.solve(A, B, C)
        assert D_slots.get("shape") == "circle"
        assert D_slots.get("size") == "small"

    def test_solve_hv(self):
        A = {"color": "red",  "shape": "square"}
        B = {"color": "blue", "shape": "square"}
        C = {"color": "red",  "shape": "circle"}
        A_hv = self.cb.encode_object(A)
        B_hv = self.cb.encode_object(B)
        C_hv = self.cb.encode_object(C)
        D_slots, D_hv, conf = self.analogy.solve_hv(A_hv, B_hv, C_hv)
        assert D_hv.shape == (D,)
        assert isinstance(D_slots, dict)

    def test_high_confidence_for_single_role_change(self):
        A = {"color": "red",  "shape": "square"}
        B = {"color": "blue", "shape": "square"}  # one change
        C = {"color": "red",  "shape": "circle"}
        _, _, conf = self.analogy.solve(A, B, C)
        assert conf > 0.7, f"Single-role analogy should have high confidence: {conf}"


# ── CompositionalWorldModel ───────────────────────────────────────────────────

class TestCompositionalWorldModel:
    def setup_method(self):
        self.cb = RoleFillerCodebook(D)
        self.cb.register_role("color")
        self.cb.register_role("position")
        for c in ["red", "blue"]:
            self.cb.register_filler("color", c)
        for p in ["left", "right", "center"]:
            self.cb.register_filler("position", p)
        self.wm = CompositionalWorldModel(self.cb, n_actions=3)

    def test_observe_transition(self):
        self.wm.observe_transition(
            prev_slots={"color": "red", "position": "left"},
            action_idx=0,
            next_slots={"color": "red", "position": "right"},
        )

    def test_predict_next_shape(self):
        # Need some training first
        for _ in range(5):
            self.wm.observe_transition(
                prev_slots={"color": "red", "position": "left"},
                action_idx=1,
                next_slots={"color": "red", "position": "right"},
            )
        next_slots, conf = self.wm.predict_next(
            {"color": "red", "position": "left"}, action_idx=1
        )
        assert isinstance(next_slots, dict)
        assert 0.0 <= conf <= 1.0

    def test_predict_no_data_returns_current(self):
        # Action 2 has no training data
        next_slots, conf = self.wm.predict_next(
            {"color": "red", "position": "left"}, action_idx=2
        )
        # Should return current state (no change)
        assert "color" in next_slots
        assert "position" in next_slots

    def test_invariant_roles(self):
        # action 0 = no-op: nothing changes
        for _ in range(5):
            self.wm.observe_transition(
                prev_slots={"color": "red", "position": "left"},
                action_idx=0,
                next_slots={"color": "red", "position": "left"},
            )
        inv = self.wm.invariant_roles(action_idx=0)
        assert isinstance(inv, list)
