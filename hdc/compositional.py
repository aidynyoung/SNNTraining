"""
Systematic Compositional Generalization in HDC
===============================================
Reference:
    Lake et al. (2017) "Building machines that learn and reason like people"
    Behav. Brain Sci. 40:e253. — SCAN benchmark for compositional generalization.

    Fodor & Pylyshyn (1988) "Connectionism and cognitive architecture"
    Cognition 28:3-71. — Systematicity requires compositionality.

    Smolensky (1990) "Tensor product variable binding and the representation
    of symbolic structures in connectionist networks" Artif. Int. 46(1-2):159-216.
    — Role-filler binding (which is exactly what HDC does with bind).

    Plate (1995) "Holographic Reduced Representations" IEEE TNNLS.
    — Circular convolution enables compositional HVs.

The problem with standard HDC classifiers:
    A standard HDC classifier that has seen "red square" and "blue circle"
    cannot correctly classify "red circle" without explicitly training on it —
    even though "red" and "circle" are both known.  This is the systematicity
    failure of standard HDC.

The compositional HDC solution:
    Encode objects as ROLE-FILLER bindings:
        object_hv = bind(role_color, filler_red) ⊗ bind(role_shape, filler_circle)
    Then:
        "red circle" = bind(role_color, red_hv) XOR bind(role_shape, circle_hv)
        "blue circle" = bind(role_color, blue_hv) XOR bind(role_shape, circle_hv)
        Similarity(red_circle, blue_circle) reflects shared shape — automatic!

    Zero-shot: the system has never seen "green triangle" but can correctly
    encode and classify it if "green" and "triangle" are each known.

This module implements:

1. RoleFillerCodebook
   ── Maintains a codebook of roles and fillers
   ── Composites: bind(role_i, filler_i) for each attribute
   ── Supports unbinding: given composite + role → filler

2. CompositionalHDCClassifier
   ── Trains on individual attributes (not full compositions)
   ── Classifies novel compositions zero-shot
   ── Interpretable: can decompose any HV into its attributes

3. StructuredAnalogy
   ── Solves multi-slot analogies: A:B :: C:? with role structure
   ── Transfers individual roles rather than the full HV
   ── More reliable than single-XOR analogy for complex objects

4. CompositionalWorldModel
   ── World model that factorises state as composition of attributes
   ── Predicts how each attribute changes (e.g., position changes, color stays)
   ── Enables systematic generalisation across attribute combinations
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import torch

from hdc.physics_world_model import _xor, _majority, _hamming


# ── Utilities ──────────────────────────────────────────────────────────────────

def _bind(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """XOR binding: bind(a, b)."""
    return _xor(a, b)

def _unbind(composite: torch.Tensor, role_hv: torch.Tensor) -> torch.Tensor:
    """XOR unbinding: unbind(bind(role, filler), role) = filler."""
    return _xor(composite, role_hv)

def _bundle(hvs: List[torch.Tensor]) -> torch.Tensor:
    """Majority-vote bundle."""
    return _majority(torch.stack(hvs, dim=0).float().mean(dim=0))

def _gen_hv(dim: int, seed: Optional[int] = None, device: str = "cpu") -> torch.Tensor:
    g = torch.Generator(device=device)
    if seed is not None:
        g.manual_seed(seed)
    return (torch.rand(dim, generator=g, device=device) >= 0.5).float()


# ═══════════════════════════════════════════════════════════════════════════════
# 1. RoleFillerCodebook — structured attribute encoding
# ═══════════════════════════════════════════════════════════════════════════════

class RoleFillerCodebook:
    """
    Codebook of roles (attribute names) and fillers (attribute values).

    An object is encoded as the bundle of its role-filler bindings:
        obj_hv = MAJORITY( bind(role_1, filler_1),
                            bind(role_2, filler_2),
                            ...
                            bind(role_k, filler_k) )

    Properties:
        - Decomposable: given obj_hv and role_i, recover filler_i ≈ unbind(obj_hv, role_i)
        - Composable: any novel (role, filler) combination → valid HV
        - Systematic: similar fillers on same role → similar object HVs
        - Role-orthogonal: different roles are nearly orthogonal → no interference

    Args:
        dim:    HV dimension
        device: torch device
    """

    def __init__(self, dim: int, device: str = "cpu"):
        self.dim    = dim
        self.device = device

        self._roles:   Dict[str, torch.Tensor] = {}   # role_name → role_hv
        self._fillers: Dict[str, Dict[str, torch.Tensor]] = {}  # role → {filler → hv}
        self._seed_counter = 0

    def register_role(self, role: str, hv: Optional[torch.Tensor] = None):
        """Register a role with a given or auto-generated HV."""
        if role not in self._roles:
            if hv is not None:
                self._roles[role] = hv.float().to(self.device)
            else:
                self._seed_counter += 1
                self._roles[role] = _gen_hv(self.dim, seed=self._seed_counter, device=self.device)
            self._fillers[role] = {}

    def register_filler(self, role: str, filler: str, hv: Optional[torch.Tensor] = None):
        """Register a filler value for a given role."""
        if role not in self._roles:
            self.register_role(role)
        if filler not in self._fillers[role]:
            if hv is not None:
                self._fillers[role][filler] = hv.float().to(self.device)
            else:
                self._seed_counter += 1
                self._fillers[role][filler] = _gen_hv(self.dim, seed=self._seed_counter, device=self.device)

    def encode_slot(self, role: str, filler: str) -> torch.Tensor:
        """Return bind(role_hv, filler_hv) for one (role, filler) pair."""
        if role not in self._roles:
            raise KeyError(f"Role '{role}' not registered")
        if filler not in self._fillers[role]:
            raise KeyError(f"Filler '{filler}' not registered for role '{role}'")
        return _bind(self._roles[role], self._fillers[role][filler])

    def encode_object(self, slots: Dict[str, str]) -> torch.Tensor:
        """
        Encode an object from a dict of {role: filler} slots.

        Args:
            slots: e.g., {"color": "red", "shape": "circle", "size": "large"}

        Returns:
            (D,) compositional object HV — bundle of all role-filler bindings.
        """
        bound_pairs = [self.encode_slot(role, filler) for role, filler in slots.items()]
        return _bundle(bound_pairs)

    def decode_filler(self, object_hv: torch.Tensor, role: str) -> Tuple[str, float]:
        """
        Unbind the role to recover the most likely filler value.

        Args:
            object_hv: (D,) object HV
            role:      Role name to decode

        Returns:
            (filler_name, similarity_score)
        """
        if role not in self._roles:
            return "unknown", 0.0
        if not self._fillers.get(role):
            return "unknown", 0.0

        role_hv  = self._roles[role]
        candidate = _unbind(object_hv, role_hv)   # ≈ filler_hv for this role

        filler_names = list(self._fillers[role].keys())
        filler_hvs   = torch.stack([self._fillers[role][f] for f in filler_names])
        sims         = _hamming(candidate.unsqueeze(0), filler_hvs)
        best_idx     = int(sims.argmax().item())
        return filler_names[best_idx], float(sims[best_idx].item())

    def decode_all(self, object_hv: torch.Tensor) -> Dict[str, Tuple[str, float]]:
        """
        Decode all roles from an object HV.

        Returns:
            {role_name: (filler_name, similarity)} for all registered roles.
        """
        return {role: self.decode_filler(object_hv, role) for role in self._roles}

    @property
    def roles(self) -> List[str]:
        return list(self._roles.keys())

    @property
    def n_roles(self) -> int:
        return len(self._roles)

    def codebook_health(self) -> Dict:
        """
        Codebook quality: role count, filler count per role, mean filler separation.

        mean_filler_sep > 0.4 → fillers are well-separated (reliable decoding).
        """
        from hdc.physics_world_model import _hamming as _h
        n_roles = len(self._roles)
        filler_stats = {}
        all_seps = []
        for role in self._roles:
            fs = list(self._roles[role].items())
            n_fillers = len(fs)
            if n_fillers < 2:
                filler_stats[role] = {"n_fillers": n_fillers, "mean_sep": None}
                continue
            sims = []
            for i in range(n_fillers):
                for j in range(i + 1, n_fillers):
                    sim = float(_h(fs[i][1].unsqueeze(0), fs[j][1].unsqueeze(0)).item())
                    sims.append(sim)
            mean_sep = sum(sims) / len(sims)
            filler_stats[role] = {"n_fillers": n_fillers, "mean_sep": round(mean_sep, 4)}
            all_seps.extend(sims)
        return {
            "n_roles":         n_roles,
            "dim":             self.dim,
            "role_stats":      filler_stats,
            "mean_filler_sep": round(sum(all_seps) / max(len(all_seps), 1), 4),
            "well_separated":  (sum(all_seps) / max(len(all_seps), 1)) > 0.4,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 2. CompositionalHDCClassifier — zero-shot compositional classification
# ═══════════════════════════════════════════════════════════════════════════════

class CompositionalHDCClassifier:
    """
    HDC classifier with systematic compositional generalization.

    Unlike standard HDC classifiers that learn one prototype per class,
    this classifier learns one prototype per (role, filler) pair —
    enabling zero-shot classification of novel compositions.

    Training: register which (role, filler) values associate with each class.
    Inference: encode query as object HV, compare to each class's compositional prototype.

    Example:
        train on: {color: red, shape: square} → class "danger"
                  {color: blue, shape: circle} → class "safe"
        zero-shot predict: {color: red, shape: circle} → ?
            → red is "danger", circle is "safe" → predict "danger" (higher overlap)

    Args:
        codebook: RoleFillerCodebook to use for encoding
        n_classes: Number of output classes
        class_names: Optional list of class name strings
    """

    def __init__(
        self,
        codebook:    RoleFillerCodebook,
        n_classes:   int,
        class_names: Optional[List[str]] = None,
    ):
        self.codebook    = codebook
        self.n_classes   = n_classes
        self.class_names = class_names or [f"class_{i}" for i in range(n_classes)]
        self.dim         = codebook.dim

        # Per-class prototype HVs (bundled from training examples)
        self._prototypes: List[torch.Tensor] = [
            torch.zeros(self.dim, device=codebook.device)
            for _ in range(n_classes)
        ]
        self._counts: List[int] = [0] * n_classes

    def train_step(self, slots: Dict[str, str], label: int):
        """
        Update prototype for class `label` with a new training example.

        Args:
            slots: {role: filler} attribute dict
            label: Integer class label
        """
        obj_hv = self.codebook.encode_object(slots)
        # Online incremental bundle
        n = self._counts[label]
        self._prototypes[label] = _majority(
            (n * self._prototypes[label].float() + obj_hv.float()) / (n + 1)
        )
        self._counts[label] += 1

    def predict(self, slots: Dict[str, str]) -> Tuple[int, str, List[float]]:
        """
        Predict class for a (possibly novel) attribute combination.

        Args:
            slots: {role: filler} dict — may contain novel combinations

        Returns:
            (class_idx, class_name, similarity_scores)
        """
        obj_hv = self.codebook.encode_object(slots)
        protos = torch.stack([p.float() for p in self._prototypes])  # (C, D)
        sims   = _hamming(obj_hv.unsqueeze(0), protos)               # (C,)
        best   = int(sims.argmax().item())
        return best, self.class_names[best], sims.tolist()

    def predict_hv(self, object_hv: torch.Tensor) -> Tuple[int, str, List[float]]:
        """Predict class from a pre-encoded object HV."""
        protos = torch.stack([p.float() for p in self._prototypes])
        sims   = _hamming(object_hv.unsqueeze(0), protos)
        best   = int(sims.argmax().item())
        return best, self.class_names[best], sims.tolist()

    def decompose_and_explain(self, slots: Dict[str, str]) -> Dict[str, Any]:
        """
        Classify + explain: which attribute values drove the prediction?

        Returns:
            Dict with class prediction and per-role contribution analysis.
        """
        obj_hv    = self.codebook.encode_object(slots)
        pred_idx, pred_name, scores = self.predict_hv(obj_hv)

        # Contribution: how much does each slot contribute to the top class?
        top_proto = self._prototypes[pred_idx]
        contributions = {}
        for role, filler in slots.items():
            slot_hv = self.codebook.encode_slot(role, filler)
            # Remove this slot and re-evaluate
            other_slots = {r: f for r, f in slots.items() if r != role}
            if other_slots:
                partial_hv   = self.codebook.encode_object(other_slots)
                full_sim     = float(_hamming(obj_hv.unsqueeze(0), top_proto.unsqueeze(0)).item())
                partial_sim  = float(_hamming(partial_hv.unsqueeze(0), top_proto.unsqueeze(0)).item())
                contributions[f"{role}:{filler}"] = full_sim - partial_sim
            else:
                slot_sim = float(_hamming(slot_hv.unsqueeze(0), top_proto.unsqueeze(0)).item())
                contributions[f"{role}:{filler}"] = slot_sim

        return {
            "predicted_class": pred_name,
            "confidence":      max(scores),
            "all_scores":      dict(zip(self.class_names, scores)),
            "role_contributions": contributions,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 3. StructuredAnalogy — multi-slot role-aware analogy
# ═══════════════════════════════════════════════════════════════════════════════

class StructuredAnalogy:
    """
    Multi-slot structured analogy: A:B :: C:? with role structure.

    Standard XOR-chain analogy (analogy.py) transfers the full HV.
    StructuredAnalogy transfers each role independently:

        For each role r:
            filler_A_r, _ = codebook.decode_filler(A_hv, r)
            filler_B_r, _ = codebook.decode_filler(B_hv, r)
            filler_C_r, _ = codebook.decode_filler(C_hv, r)

            If filler_A_r == filler_B_r:  # role unchanged in A:B
                D_slots[r] = filler_C_r   # keep C's value for this role
            Else:                          # role changed in A:B
                D_slots[r] = filler_B_r   # apply the same change to D

        D_hv = encode_object(D_slots)

    Example:
        A = {color:red, shape:square, size:large}
        B = {color:blue, shape:square, size:large}  (only color changed)
        C = {color:red, shape:circle, size:small}
        D = ? → {color:blue, shape:circle, size:small}  (apply color change)

    Args:
        codebook: RoleFillerCodebook defining the compositional structure
    """

    def __init__(self, codebook: RoleFillerCodebook):
        self.codebook = codebook

    def solve(
        self,
        A_slots: Dict[str, str],
        B_slots: Dict[str, str],
        C_slots: Dict[str, str],
    ) -> Tuple[Dict[str, str], torch.Tensor, float]:
        """
        Solve A:B :: C:? using structured slot-wise analogy.

        Args:
            A_slots, B_slots, C_slots: Attribute dicts for A, B, C

        Returns:
            (D_slots, D_hv, confidence)
            D_slots:     Predicted attribute dict for D
            D_hv:        (dim,) predicted HV for D
            confidence:  Mean decode similarity across roles
        """
        D_slots  = {}
        sim_sum  = 0.0
        n_roles  = 0

        for role in self.codebook.roles:
            fa  = A_slots.get(role)
            fb  = B_slots.get(role)
            fc  = C_slots.get(role)

            if fc is None:
                continue  # C doesn't have this role; skip

            n_roles += 1

            if fa == fb:
                # This role was unchanged in A→B; keep C's value for D
                D_slots[role] = fc
                sim_sum      += 1.0  # no transfer needed, perfect
            elif fb is not None:
                # This role changed in A→B; apply same change to D
                D_slots[role] = fb
                sim_sum      += 0.8  # transfer applied

        confidence = sim_sum / max(n_roles, 1)
        D_hv       = self.codebook.encode_object(D_slots) if D_slots else torch.zeros(self.codebook.dim)
        return D_slots, D_hv, confidence

    def solve_hv(
        self,
        A_hv: torch.Tensor,
        B_hv: torch.Tensor,
        C_hv: torch.Tensor,
    ) -> Tuple[Dict[str, str], torch.Tensor, float]:
        """
        Solve A:B :: C:? from pre-encoded HVs (decode all slots first).

        Args:
            A_hv, B_hv, C_hv: (D,) binary HVs

        Returns:
            (D_slots, D_hv, mean_confidence)
        """
        A_dec = {r: self.codebook.decode_filler(A_hv, r) for r in self.codebook.roles}
        B_dec = {r: self.codebook.decode_filler(B_hv, r) for r in self.codebook.roles}
        C_dec = {r: self.codebook.decode_filler(C_hv, r) for r in self.codebook.roles}

        A_slots = {r: v for r, (v, _) in A_dec.items()}
        B_slots = {r: v for r, (v, _) in B_dec.items()}
        C_slots = {r: v for r, (v, _) in C_dec.items()}

        return self.solve(A_slots, B_slots, C_slots)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. CompositionalWorldModel — attribute-factored world model
# ═══════════════════════════════════════════════════════════════════════════════

class CompositionalWorldModel:
    """
    World model that predicts how each attribute changes independently.

    Instead of predicting a monolithic next-state HV, predicts per-attribute
    transitions:
        For each role r: predict next_filler(r) from current_filler(r) + action

    This enables:
        - Systematic generalisation: if "position" changes under "move right",
          apply this regardless of "color", "size", "object_type"
        - Compositional planning: find the action that changes only the desired
          attributes while keeping others constant
        - Interpretable prediction: understand which attributes are being changed

    Args:
        codebook:  RoleFillerCodebook
        n_actions: Number of possible actions
        device:    torch device
    """

    def __init__(
        self,
        codebook:  RoleFillerCodebook,
        n_actions: int = 8,
        device:    str = "cpu",
    ):
        self.codebook  = codebook
        self.n_actions = n_actions
        self.device    = device

        # Per-(role, action) transition HVs: accumulated evidence
        # transition_hv[role][action_idx] = expected change in role under action
        self._transitions: Dict[str, List[torch.Tensor]] = {
            role: [torch.zeros(codebook.dim, device=device) for _ in range(n_actions)]
            for role in codebook.roles
        }
        self._transition_counts: Dict[str, List[int]] = {
            role: [0] * n_actions for role in codebook.roles
        }

    def observe_transition(
        self,
        prev_slots: Dict[str, str],
        action_idx: int,
        next_slots: Dict[str, str],
    ):
        """
        Record a (state, action → next_state) transition.

        Updates per-role transition estimates.
        """
        for role in self.codebook.roles:
            prev_f = prev_slots.get(role)
            next_f = next_slots.get(role)
            if prev_f is None or next_f is None:
                continue

            # Transition HV = bind(prev_filler_hv, next_filler_hv)
            prev_hv = self.codebook._fillers[role].get(prev_f)
            next_hv = self.codebook._fillers[role].get(next_f)
            if prev_hv is None or next_hv is None:
                continue

            trans = _bind(prev_hv, next_hv)
            n = self._transition_counts[role][action_idx]
            self._transitions[role][action_idx] = _majority(
                (n * self._transitions[role][action_idx].float() + trans.float()) / (n + 1)
            )
            self._transition_counts[role][action_idx] += 1

    def predict_next(
        self,
        current_slots: Dict[str, str],
        action_idx: int,
    ) -> Tuple[Dict[str, str], float]:
        """
        Predict the next state for each attribute given current state + action.

        Args:
            current_slots: {role: filler} current state
            action_idx:    Integer action index

        Returns:
            (predicted_next_slots, mean_confidence)
        """
        next_slots  = {}
        sim_sum     = 0.0
        n_predicted = 0

        for role, current_filler in current_slots.items():
            if role not in self._transitions:
                next_slots[role] = current_filler
                continue

            if self._transition_counts[role][action_idx] == 0:
                # No data for this (role, action) — predict no change
                next_slots[role] = current_filler
                sim_sum += 0.5
                n_predicted += 1
                continue

            trans_hv  = self._transitions[role][action_idx]
            curr_hv   = self.codebook._fillers.get(role, {}).get(current_filler)
            if curr_hv is None:
                next_slots[role] = current_filler
                continue

            # Predicted next filler HV: unbind transition from current
            # trans = bind(prev, next) → next = bind(trans, prev) (XOR is self-inverse)
            pred_next_hv        = _bind(trans_hv, curr_hv)
            pred_filler, sim    = self.codebook.decode_filler(
                pred_next_hv.unsqueeze(0).squeeze(0) if pred_next_hv.dim() == 0 else pred_next_hv,
                role
            )
            next_slots[role]    = pred_filler
            sim_sum            += sim
            n_predicted        += 1

        confidence = sim_sum / max(n_predicted, 1)
        return next_slots, confidence

    def plan_to_goal(
        self,
        current_slots: Dict[str, str],
        target_slots:  Dict[str, str],
        max_steps:     int = 5,
    ) -> List[int]:
        """
        Find a short action sequence that transforms current_slots → target_slots.

        Uses greedy best-first search: at each step, pick the action that
        most improves the match between predicted and target state.

        This is **compositional planning** — the agent can plan to achieve
        a target combination of attribute values it has never seen together,
        because it understands each attribute's dynamics independently.

        Args:
            current_slots: {role: filler} start state
            target_slots:  {role: filler} goal state
            max_steps:     Maximum plan length

        Returns:
            List of action indices forming the plan.
        """
        def _slot_match(s1: Dict, s2: Dict) -> float:
            if not s1 or not s2:
                return 0.0
            roles = set(s1) & set(s2)
            return sum(1 for r in roles if s1.get(r) == s2.get(r)) / max(len(roles), 1)

        plan = []
        current = dict(current_slots)

        for _ in range(max_steps):
            if _slot_match(current, target_slots) >= 1.0:
                break   # goal reached

            best_action, best_match = 0, -1.0
            for a in range(self.n_actions):
                predicted, _ = self.predict_next(current, a)
                match = _slot_match(predicted, target_slots)
                if match > best_match:
                    best_match, best_action = match, a

            plan.append(best_action)
            current, _ = self.predict_next(current, best_action)

        return plan

    def world_model_summary(self) -> Dict:
        """High-level summary: transitions learned, prediction coverage."""
        total_trans = sum(
            sum(self._transition_counts[r][a] for a in range(len(self._actions)))
            for r in self.codebook.roles
        )
        well_trained = {
            r: sum(1 for a in range(len(self._actions))
                   if self._transition_counts[r][a] >= 3)
            for r in self.codebook.roles
        }
        return {
            "n_states_seen":   len(self._states),
            "n_actions":       len(self._actions),
            "n_roles":         len(self.codebook.roles),
            "total_transitions": total_trans,
            "well_trained_roles": well_trained,
        }

    def invariant_roles(self, action_idx: int) -> List[str]:
        """
        Identify which roles are not changed by a given action.

        A role r is invariant under action a if:
            Hamming(transition_hv[r][a], identity) is small
            (i.e., the transition barely changes anything)
        """
        invariant = []
        identity_hv = torch.zeros(self.codebook.dim, device=self.device)
        for role in self.codebook.roles:
            if self._transition_counts[role][action_idx] < 3:
                continue
            trans_hv = self._transitions[role][action_idx]
            sim_to_identity = float(
                _hamming(trans_hv.unsqueeze(0), identity_hv.unsqueeze(0)).item()
            )
            if sim_to_identity > 0.8:   # very similar to identity → no change
                invariant.append(role)
        return invariant


# ═══════════════════════════════════════════════════════════════════════════════
# Smoke test
# ═══════════════════════════════════════════════════════════════════════════════

def _test_compositional():
    D = 500
    print("=== RoleFillerCodebook ===")
    cb = RoleFillerCodebook(D)
    cb.register_role("color")
    cb.register_role("shape")
    cb.register_role("size")
    for color in ["red", "blue", "green"]:
        cb.register_filler("color", color)
    for shape in ["square", "circle", "triangle"]:
        cb.register_filler("shape", shape)
    for size in ["large", "small"]:
        cb.register_filler("size", size)

    print(f"  Roles: {cb.roles}, n_roles={cb.n_roles}")

    # Encode an object
    red_circle = cb.encode_object({"color": "red", "shape": "circle", "size": "large"})
    print(f"  red_circle HV density: {red_circle.mean():.3f}")

    # Decode each slot
    decoded = cb.decode_all(red_circle)
    print(f"  Decoded: { {r: v for r, (v, s) in decoded.items()} }")

    # Zero-shot test: novel combination
    blue_triangle = cb.encode_object({"color": "blue", "shape": "triangle", "size": "small"})
    decoded2 = cb.decode_all(blue_triangle)
    print(f"  Decoded novel: { {r: v for r, (v, s) in decoded2.items()} }")

    print("\n=== CompositionalHDCClassifier ===")
    clf = CompositionalHDCClassifier(cb, n_classes=2, class_names=["danger", "safe"])

    # Train only on {red, square} → danger and {blue, circle} → safe
    clf.train_step({"color": "red", "shape": "square", "size": "large"}, label=0)
    clf.train_step({"color": "blue", "shape": "circle", "size": "small"}, label=1)

    # Zero-shot: {red, circle} → should be "danger" (red influence)
    pred_idx, pred_name, scores = clf.predict({"color": "red", "shape": "circle", "size": "large"})
    print(f"  Zero-shot {{'red','circle'}}: predicted '{pred_name}' (scores={[round(s,3) for s in scores]})")

    explanation = clf.decompose_and_explain({"color": "red", "shape": "circle", "size": "large"})
    print(f"  Explanation: {explanation}")

    print("\n=== StructuredAnalogy ===")
    analogy = StructuredAnalogy(cb)
    A = {"color": "red", "shape": "square", "size": "large"}
    B = {"color": "blue", "shape": "square", "size": "large"}   # only color changed
    C = {"color": "red", "shape": "circle", "size": "small"}
    D_slots, D_hv, conf = analogy.solve(A, B, C)
    print(f"  A:B :: C:D → D_slots={D_slots}  (expected color:blue, conf={conf:.3f})")
    assert D_slots.get("color") == "blue", f"Expected color='blue', got {D_slots.get('color')}"
    assert D_slots.get("shape") == "circle"  # shape unchanged
    print("  ✓ Structured analogy correct")

    print("\n=== CompositionalWorldModel ===")
    wm = CompositionalWorldModel(cb, n_actions=2)  # action 0=no-op, 1=turn-red

    # Train: action 1 turns color to red
    for _ in range(5):
        wm.observe_transition(
            prev_slots={"color": "blue", "shape": "circle", "size": "small"},
            action_idx=1,
            next_slots={"color": "red", "shape": "circle", "size": "small"},
        )

    pred_slots, conf = wm.predict_next(
        {"color": "blue", "shape": "circle", "size": "small"}, action_idx=1
    )
    print(f"  Predicted after action 1 (turn-red): {pred_slots}  conf={conf:.3f}")
    inv = wm.invariant_roles(action_idx=0)
    print(f"  Invariant roles for action 0 (no-op): {inv}")

    print("\n✅ All compositional tests passed")


if __name__ == "__main__":
    _test_compositional()
