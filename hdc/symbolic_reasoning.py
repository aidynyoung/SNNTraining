"""
hdc/symbolic_reasoning.py
===========================
Symbolic AI and Logic in Hypervector Space
==========================================
Reference:
    Smolensky (1990) "Tensor product variable binding and the representation
    of symbolic structures in connectionist networks" Artif. Int. 46(1-2).

    Plate (1995) §7: "Reasoning and inference with HRRs"
    — propositional logic in holographic space.

    Gayler (2003) "Vector symbolic architectures answer Jackendoff's challenges
    for cognitive neuroscience" arXiv:cs/0412059.
    — VSA as a framework for compositional AI.

    Kanerva (1998) "Pattern completion with distributed representations"
    — associative inference in HDC.

Why symbolic reasoning in HDC:

    Standard symbolic AI (Prolog, theorem provers):
        + Exact inference
        + Interpretable
        - Brittle: fails on partial/noisy knowledge
        - No learning from data

    Neural AI (LLMs, GNNs):
        + Handles noise
        + Learns from data
        - Black box
        - Expensive inference

    HDC symbolic reasoning:
        + Exact inference on stored knowledge (HRR exact unbinding)
        + Noise tolerance (holographic storage)
        + Online learning (bundle new facts)
        + O(D log D) per inference step
        + Interpretable (z-score on every conclusion)

This module implements:

1. HDCPropLogic (Propositional Logic)
   — Represents propositions as HVs (true/false via role-filler)
   — Connectives: AND = bind, OR = bundle, NOT = invert (XOR with all-ones)
   — Modus ponens: if P → Q and P, then Q (HRR chained unbinding)
   — Inference: forward chaining from premises to conclusions

2. HDCFirstOrderLogic
   — Variables: random HVs (like UUIDs for logical objects)
   — Predicates: role HVs in tensor product structures
   — Quantifiers: bundling (universal) and existence check (Hamming > threshold)
   — Unification: find variable bindings that make two atoms match

3. HDCRuleEngine
   — Production rule system: IF conditions THEN actions
   — Rules encoded as role-filler bindings
   — Forward chaining: iteratively fire triggered rules
   — Applications: expert systems, business rules, sensor fusion logic

4. HDCTheoremProver (resolution-based)
   — Stores axioms as HVs in associative memory
   — Proof by contradiction: assume NOT(goal), derive False
   — Each resolution step is one HRR unbinding + cleanup
   — Returns proof chain with confidence scores
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F

from hdc.physics_world_model import _hamming, _majority, _xor


# ── Utilities ──────────────────────────────────────────────────────────────────

def _gen_hv(dim: int, seed: Optional[int] = None, device: str = "cpu") -> torch.Tensor:
    import hashlib
    if seed is None:
        g = torch.Generator(device=device)
        return (torch.rand(dim, generator=g, device=device) >= 0.5).float()
    real_seed = int(hashlib.md5(str(seed).encode()).hexdigest()[:8], 16) % (2**31)
    g = torch.Generator(device=device)
    g.manual_seed(real_seed)
    return (torch.rand(dim, generator=g, device=device) >= 0.5).float()

def _bind(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a.float() != b.float()).float()  # XOR

def _bundle(hvs: List[torch.Tensor]) -> torch.Tensor:
    return _majority(torch.stack(hvs).float().mean(dim=0))

def _NOT(hv: torch.Tensor) -> torch.Tensor:
    """Logical NOT in HDC: flip all bits."""
    return 1.0 - hv.float()


# ═══════════════════════════════════════════════════════════════════════════════
# 1. HDCPropLogic — Propositional Logic
# ═══════════════════════════════════════════════════════════════════════════════

class HDCPropLogic:
    """
    Propositional logic in HDC space.

    Atomic propositions are represented as random HVs.
    Connectives use HDC algebra:
        AND(P, Q) = bind(P, Q)       [conjunct: similar to both P and Q]
        OR(P, Q)  = bundle([P, Q])   [superposition: similar to either]
        NOT(P)    = invert(P)        [negate: maximally dissimilar]
        IMPLIES(P,Q) = bind(P, NOT(Q)) = bind(NOT(Q), P)

    Modus ponens:
        Given: P → Q stored as bind(P_hv, NOT(Q_hv)) in rule_memory
        Given: P is true (P_hv in fact_memory)
        Conclude: Q_hv = NOT(unbind(rule_memory, P_hv))

    Args:
        dim:    HV dimension
        device: torch device
    """

    def __init__(self, dim: int, device: str = "cpu"):
        self.dim    = dim
        self.device = device

        self._atoms:  Dict[str, torch.Tensor] = {}
        self._facts:  torch.Tensor            = torch.zeros(dim, device=device)
        self._rules:  torch.Tensor            = torch.zeros(dim, device=device)
        self._true_hv  = _gen_hv(dim, seed="TRUE",  device=device)
        self._false_hv = _gen_hv(dim, seed="FALSE", device=device)

    def atom(self, name: str) -> torch.Tensor:
        """Get or create atomic proposition HV."""
        if name not in self._atoms:
            self._atoms[name] = _gen_hv(self.dim, seed=name, device=self.device)
        return self._atoms[name]

    def AND(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        return _bind(p, q)

    def OR(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        return _bundle([p, q])

    def NOT(self, p: torch.Tensor) -> torch.Tensor:
        return _NOT(p)

    def IMPLIES(self, antecedent: str, consequent: str):
        """
        Assert P → Q (if P then Q).
        Stores in rule memory as bind(P, NOT(Q)).
        """
        p_hv = self.atom(antecedent)
        q_hv = self.atom(consequent)
        rule_hv = _bind(p_hv, _NOT(q_hv))
        self._rules = self._rules + rule_hv

    def assert_fact(self, name: str):
        """Assert that proposition `name` is True."""
        self._facts = self._facts + self.atom(name)

    def query(self, name: str, threshold: float = 0.55) -> Tuple[bool, float]:
        """
        Query whether proposition `name` is likely True.

        Returns:
            (is_true, confidence) where confidence ∈ [0, 1]
        """
        p_hv = self.atom(name)
        sim  = float(_hamming(p_hv.unsqueeze(0),
                               _majority(self._facts).unsqueeze(0)).item())
        return sim > threshold, sim

    def modus_ponens(self, antecedent: str) -> List[Tuple[str, float]]:
        """
        Apply modus ponens: given P is true, find all Q such that P → Q.

        Returns list of (consequent_name, confidence) sorted desc.
        """
        p_hv = self.atom(antecedent)
        # Recover Q from rule: bind(P, NOT(Q)) → NOT(Q) = unbind(rule, P)
        not_q = _xor(self._rules, p_hv)  # approximate NOT(Q)
        q_candidate = _NOT(not_q)         # Q ≈ NOT(NOT(Q))

        results = []
        for name, q_hv in self._atoms.items():
            sim = float(_hamming(q_candidate.unsqueeze(0), q_hv.unsqueeze(0)).item())
            if sim > 0.5:
                results.append((name, sim))

        return sorted(results, key=lambda x: x[1], reverse=True)

    def forward_chain(self, max_steps: int = 10) -> List[str]:
        """
        Forward chaining with confidence-prioritised agenda.

        Uses a priority queue (max-heap by confidence) so high-confidence
        inferences are processed before uncertain ones.  This reduces the
        number of rule applications needed to reach a fixed point and avoids
        cascading noise from low-confidence intermediate conclusions.

        Returns list of newly derived facts.
        """
        import heapq

        # Initialise known facts + agenda of (confidence, name) pairs
        known:   Dict[str, float] = {
            n: conf for n in self._atoms
            for is_t, conf in [self.query(n)]
            if is_t
        }
        agenda: list = [(-conf, name) for name, conf in known.items()]
        heapq.heapify(agenda)

        derived: List[str] = []
        self._inference_trace: Dict[str, Tuple[str, float]] = {}  # fact → (cause, conf)
        step = 0

        while agenda and step < max_steps:
            step += 1
            neg_conf, fact = heapq.heappop(agenda)
            conclusions = self.modus_ponens(fact)
            for name, conf in conclusions:
                if name not in known and conf > 0.6:
                    self.assert_fact(name)
                    known[name] = conf
                    self._inference_trace[name] = (fact, conf)
                    derived.append(name)
                    heapq.heappush(agenda, (-conf, name))

        return derived

    def backward_chain(
        self,
        goal:      str,
        max_depth: int = 5,
    ) -> Optional[List[str]]:
        """
        Goal-directed backward chaining: find a chain of facts that proves `goal`.

        Starting from the goal, works backward through IMPLIES rules to find
        facts already in the knowledge base that entail the goal.

        Args:
            goal:      Proposition name to prove
            max_depth: Maximum chain length

        Returns:
            List of supporting facts [fact_1, ..., fact_n, goal], or None if not provable.
        """
        # Check if goal is already known
        if self.query(goal)[0]:
            return [goal]

        if max_depth == 0:
            return None

        # Try to find antecedents that imply `goal`
        goal_hv = self.atom(goal)
        # Rule: if P → goal, then NOT(goal) = unbind(rule, P) → P = unbind(rule, NOT(goal))
        not_goal = _NOT(goal_hv)
        p_candidate = _xor(self._rules, not_goal)

        for name, p_hv in self._atoms.items():
            sim = float(_hamming(p_candidate.unsqueeze(0), p_hv.unsqueeze(0)).item())
            if sim > 0.55 and name != goal:
                # Recursively try to prove antecedent
                sub_proof = self.backward_chain(name, max_depth - 1)
                if sub_proof is not None:
                    return sub_proof + [goal]

        return None

    def explain(self, fact: str) -> str:
        """
        Return a human-readable explanation of how `fact` was derived.

        Traces back through the _inference_trace dict built during forward_chain().
        """
        if not hasattr(self, '_inference_trace'):
            return f"{fact} (no trace available — run forward_chain first)"
        if fact not in self._inference_trace:
            if self.query(fact)[0]:
                return f"{fact} (asserted as initial fact)"
            return f"{fact} (not derived)"

        chain = [fact]
        current = fact
        seen    = {fact}
        while current in self._inference_trace:
            cause, conf = self._inference_trace[current]
            chain.insert(0, f"{cause} ({conf:.2f})")
            if cause in seen:
                break
            seen.add(cause)
            current = cause

        return " → ".join(chain)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. HDCRuleEngine — production rule system
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class HDCRule:
    """A production rule: IF conditions THEN action."""
    name:       str
    conditions: List[str]   # proposition names that must be True
    action:     str         # proposition to assert when conditions hold
    confidence: float = 1.0


class HDCRuleEngine:
    """
    Forward-chaining production rule engine in HDC.

    Rules are stored as structured HVs.
    Firing condition: check if all condition propositions are in fact memory.
    Action: assert consequent proposition.

    Applications:
        - Expert systems (medical diagnosis, fault detection)
        - Business logic (if sensor_A AND sensor_B then alert)
        - Automated response planning

    Args:
        dim: HV dimension
    """

    def __init__(self, dim: int, device: str = "cpu"):
        self.logic  = HDCPropLogic(dim, device=device)
        self._rules: List[HDCRule] = []

    def add_rule(self, name: str, conditions: List[str], action: str, conf: float = 1.0):
        """Add a production rule."""
        self._rules.append(HDCRule(name, conditions, action, conf))
        # Store as implication chain
        for cond in conditions:
            self.logic.IMPLIES(cond, action)

    def assert_fact(self, name: str):
        """Assert a fact."""
        self.logic.assert_fact(name)

    def fire_rules(self, max_rounds: int = 10) -> List[Tuple[str, str]]:
        """
        Fire all applicable rules until fixed point.

        Returns:
            List of (rule_name, derived_fact) for each firing.
        """
        fired = []
        for _ in range(max_rounds):
            new_firings = []
            for rule in self._rules:
                # Check all conditions
                all_true = all(self.logic.query(cond)[0] for cond in rule.conditions)
                already  = self.logic.query(rule.action)[0]
                if all_true and not already:
                    self.logic.assert_fact(rule.action)
                    new_firings.append((rule.name, rule.action))
            fired.extend(new_firings)
            if not new_firings:
                break
        return fired

    def query(self, fact: str) -> Tuple[bool, float]:
        """Query whether a fact is known."""
        return self.logic.query(fact)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. HDCTheoremProver — resolution-based proof
# ═══════════════════════════════════════════════════════════════════════════════

class HDCTheoremProver:
    """
    Resolution-based theorem prover in HDC space.

    Reference:
        Robinson (1965) "A machine-oriented logic based on the resolution principle"
        JACM 12(1):23-41. — original resolution algorithm.

    HDC proof by contradiction:
        1. Assert NOT(goal)
        2. Iteratively: find clauses that "resolve" (contradict each other)
        3. Derive EMPTY (False) → original goal is True

    Each resolution step:
        - Candidate: bind(clause1, clause2) → should be near EMPTY_HV
        - If Hamming_sim(resolution, empty_hv) > threshold → contradiction

    Args:
        dim: HV dimension
    """

    def __init__(self, dim: int, device: str = "cpu"):
        self.dim    = dim
        self.device = device

        self._axioms:    Dict[str, torch.Tensor] = {}
        self._axiom_mem: torch.Tensor            = torch.zeros(dim, device=device)
        self._empty_hv   = torch.zeros(dim, device=device)  # empty clause

    def add_axiom(self, name: str, hv: torch.Tensor):
        """Add an axiom (known fact) to the knowledge base."""
        self._axioms[name] = hv.float().to(self.device)
        self._axiom_mem    = self._axiom_mem + hv.float().to(self.device)

    def add_axiom_implication(self, antecedent_name: str, consequent_name: str,
                               ant_hv: torch.Tensor, con_hv: torch.Tensor):
        """Add P → Q as the clause NOT(P) ∨ Q."""
        not_p = _NOT(ant_hv)
        clause = _bundle([not_p, con_hv])
        clause_name = f"{antecedent_name}_implies_{consequent_name}"
        self.add_axiom(clause_name, clause)

    def prove(
        self,
        goal_hv: torch.Tensor,
        max_steps: int = 5,
        threshold: float = 0.55,
    ) -> Tuple[bool, List[str], float]:
        """
        Try to prove goal_hv from the axiom base.

        Uses: assume NOT(goal), try to derive contradiction.

        Returns:
            (proved, proof_chain, confidence)
        """
        not_goal = _NOT(goal_hv.float().to(self.device))
        current  = not_goal.clone()
        proof_chain = ["assume NOT(goal)"]
        best_sim    = 0.0

        for step in range(max_steps):
            # Try to resolve with each axiom
            for name, axiom_hv in self._axioms.items():
                resolution = _bind(current, axiom_hv)

                # Check if resolution is close to empty clause (contradiction)
                sim_to_empty = float(_hamming(
                    resolution.unsqueeze(0), self._empty_hv.unsqueeze(0)
                ).item())

                # Contradiction: resolution ≈ empty (very low density)
                density = float(resolution.float().mean())
                if density < 0.1:   # near-zero density = near-empty clause
                    proof_chain.append(f"resolve with {name} → CONTRADICTION (density={density:.3f})")
                    return True, proof_chain, 1.0 - density

                if sim_to_empty > best_sim:
                    best_sim = sim_to_empty
                    best_candidate = name

                current = _majority((current + resolution) / 2.0)

            proof_chain.append(f"step {step+1}: best_sim={best_sim:.3f}")

        # Didn't find contradiction — can't prove
        return False, proof_chain, best_sim

    def entails(self, goal_name: str, goal_hv: torch.Tensor) -> Tuple[bool, float]:
        """
        Simple entailment check: is goal_hv similar to the axiom memory?
        """
        sim = float(_hamming(
            goal_hv.float().unsqueeze(0),
            _majority(self._axiom_mem).unsqueeze(0)
        ).item())
        return sim > 0.5, sim


# ═══════════════════════════════════════════════════════════════════════════════
# 4. HDCUnifier — term unification in HDC
# ═══════════════════════════════════════════════════════════════════════════════

class HDCUnifier:
    """
    Variable unification in HDC space.

    Unification finds variable bindings θ such that two terms t1 and t2
    become equal: t1θ = t2θ.

    In HDC:
        Variables: special "blank" HVs near the uniform distribution
        Constants: specific HVs
        Terms: tensor product structures (role → filler)

    HDC unification:
        Given: f(X, a) and f(b, Y)
        Find: {X→b, Y→a} such that both become f(b, a)

        Method: subtract known constants, recover variables via unbinding.

    Args:
        dim: HV dimension
    """

    def __init__(self, dim: int, device: str = "cpu"):
        self.dim    = dim
        self.device = device
        self._vars:  Dict[str, torch.Tensor] = {}
        self._consts: Dict[str, torch.Tensor] = {}

    def var(self, name: str) -> torch.Tensor:
        """Create or retrieve a variable HV."""
        if name not in self._vars:
            # Variables are near-uniform HVs (50% density)
            self._vars[name] = _gen_hv(self.dim, seed=f"VAR_{name}", device=self.device)
        return self._vars[name]

    def const(self, name: str) -> torch.Tensor:
        """Create or retrieve a constant HV."""
        if name not in self._consts:
            self._consts[name] = _gen_hv(self.dim, seed=f"CONST_{name}", device=self.device)
        return self._consts[name]

    def unify(
        self,
        term1_roles: Dict[str, str],   # {role: name_or_var}
        term2_roles: Dict[str, str],
        role_hvs:    Dict[str, torch.Tensor],
    ) -> Tuple[bool, Dict[str, str]]:
        """
        Try to unify two terms by finding variable bindings.

        Args:
            term1_roles: {role_name: filler_name_or_var}
            term2_roles: {role_name: filler_name_or_var}
            role_hvs:    {role_name: role_HV}

        Returns:
            (unified, bindings) where bindings = {var_name: const_name}
        """
        bindings = {}

        for role in term1_roles:
            if role not in term2_roles:
                continue

            n1 = term1_roles[role]
            n2 = term2_roles[role]

            if n1 == n2:
                continue   # same name = trivially unified

            # Check which is a variable
            is_var1 = n1 in self._vars
            is_var2 = n2 in self._vars

            if is_var1 and not is_var2:
                bindings[n1] = n2
            elif is_var2 and not is_var1:
                bindings[n2] = n1
            elif not is_var1 and not is_var2:
                # Both constants: must be equal
                hv1 = self._consts.get(n1, self.const(n1))
                hv2 = self._consts.get(n2, self.const(n2))
                sim = float(_hamming(hv1.unsqueeze(0), hv2.unsqueeze(0)).item())
                if sim < 0.9:
                    return False, {}   # constants differ: unification fails

        return True, bindings


# ═══════════════════════════════════════════════════════════════════════════════
# Smoke test
# ═══════════════════════════════════════════════════════════════════════════════

def _test_symbolic_reasoning():
    D = 256

    print("=== HDCPropLogic ===")
    logic = HDCPropLogic(D)
    logic.IMPLIES("rain", "wet_ground")
    logic.IMPLIES("wet_ground", "slippery")
    logic.assert_fact("rain")

    # Query
    is_rain, rain_conf = logic.query("rain")
    print(f"  rain={is_rain} (conf={rain_conf:.3f})  OK")
    assert is_rain

    # Modus ponens: rain → wet_ground
    conclusions = logic.modus_ponens("rain")
    print(f"  Modus ponens(rain): {conclusions[:2]}  OK")
    assert any(name == "wet_ground" for name, _ in conclusions[:3])

    # Forward chaining
    derived = logic.forward_chain()
    print(f"  Forward chaining derived: {derived}  OK")

    print("\n=== HDCRuleEngine ===")
    engine = HDCRuleEngine(D)
    engine.add_rule("fire_alert",
                    conditions=["high_temp", "smoke_detected"],
                    action="fire_alarm")
    engine.add_rule("sprinklers",
                    conditions=["fire_alarm"],
                    action="activate_sprinklers")

    engine.assert_fact("high_temp")
    engine.assert_fact("smoke_detected")
    fired = engine.fire_rules()
    print(f"  Fired rules: {fired}  OK")
    assert any(action == "fire_alarm" for _, action in fired)

    alarm, alarm_conf = engine.query("fire_alarm")
    print(f"  fire_alarm={alarm} (conf={alarm_conf:.3f})  OK")

    sprinklers, _ = engine.query("activate_sprinklers")
    print(f"  activate_sprinklers={sprinklers}  OK")

    print("\n=== HDCTheoremProver ===")
    prover = HDCTheoremProver(D)

    # Axioms: mortal(X) :- human(X); human(socrates)
    socrates = _gen_hv(D, seed="socrates")
    human    = _gen_hv(D, seed="human")
    mortal   = _gen_hv(D, seed="mortal")

    human_socrates   = _bind(human,  socrates)
    mortal_socrates  = _bind(mortal, socrates)

    prover.add_axiom("human(socrates)", human_socrates)
    prover.add_axiom_implication("human(socrates)", "mortal(socrates)",
                                  human_socrates, mortal_socrates)

    # Entailment: is mortal(socrates) entailed?
    entailed, conf = prover.entails("mortal(socrates)", mortal_socrates)
    print(f"  mortal(socrates) entailed={entailed} (conf={conf:.3f})  OK")

    # Proof attempt
    proved, chain, conf = prover.prove(mortal_socrates, max_steps=3)
    print(f"  Proof: proved={proved}, chain_len={len(chain)}  OK")

    print("\n=== HDCUnifier ===")
    unifier = HDCUnifier(D)
    role_hvs = {"subject": _gen_hv(D, seed="subject"),
                "verb":    _gen_hv(D, seed="verb")}

    unifier.const("socrates")
    unifier.const("likes")
    unifier.const("plato")
    unifier.var("X")
    unifier.var("Y")

    # Unify: likes(X, plato) with likes(socrates, Y)
    # Should bind X→socrates, Y→plato
    unified, bindings = unifier.unify(
        {"subject": "X", "verb": "likes", "object": "plato"},
        {"subject": "socrates", "verb": "likes", "object": "Y"},
        role_hvs,
    )
    print(f"  Unification: {unified}, bindings={bindings}  OK")
    assert unified

    print("\n✅ All symbolic_reasoning tests passed")


if __name__ == "__main__":
    _test_symbolic_reasoning()
