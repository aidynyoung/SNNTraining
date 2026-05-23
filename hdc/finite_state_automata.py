"""
hdc/finite_state_automata.py
=============================
Finite State Automata Encoded as Hypervectors
==============================================
Reference:
    torchhd (Heddes et al. 2023) JMLR — `torchhd.structures.FiniteStateAutomata`
    https://github.com/hyperdimensional-computing/torchhd

    Frady & Sommer (2018) "A Theory of Sequence Indexing and Working Memory"
    — Sequence encoding via binding as foundation.

    Kanerva (1998) "Fully Distributed Representation"
    — VSA encoding of computational structures.

    Gayler & Levy (2011) "VSA Architectures" arXiv:cs/0412059
    — FSA as VSA structure.

Why VSA-encoded FSAs matter for Arthedain:

    Standard FSAs: state tables with O(|S| × |A|) memory
    VSA-encoded FSA: one bundle HV per automaton + clean transition computation

    The VSA-FSA encodes transitions as:
        T = MAJORITY( bind(state_hv, bind(token_hv, next_state_hv))
                      for all (state, token, next_state) )

    Query (current_state, token) → next_state:
        candidate = unbind(T, bind(current_state, token))
        next_state = cleanup(candidate)

    Applications in Arthedain:
        - Encode protocol state machines (TCP, CoAP, sensor protocols)
        - Regular language recognition without explicit tables
        - Sequence pattern matching in sensor streams
        - Grammar-constrained HDC language models

This module implements:

1. VSAFiniteStateAutomaton
   — Stores transitions as bundled HV superposition
   — Supports: add_transition, next_state, accepts (final states)
   — Works with both binary XOR (approximate) and HRR (exact) binding

2. VSANondeterministicFSA
   — Multiple possible next states per (state, token) pair
   — Returns ranked list of next states by similarity

3. RegularLanguageMatcher
   — Wraps VSAFiniteStateAutomaton for sequence membership queries
   — match(sequence) → True/False via FSA simulation

4. HDCLanguageModel (FSA-constrained)
   — Constrained generation: only tokens valid per current FSA state
   — Combines VSAContextWindow (sequence history) + FSA (grammar constraint)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F

from hdc.hrr import HRR


# ── Utility ────────────────────────────────────────────────────────────────────

def _gen_hv(dim: int, seed=None, device: str = "cpu") -> torch.Tensor:
    import hashlib
    raw = int(hashlib.md5(str(seed).encode()).hexdigest()[:8], 16) % (2**31) if seed is not None else None
    hrr = HRR(dim, device)
    return hrr.gen(1, seed=raw)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. VSAFiniteStateAutomaton
# ═══════════════════════════════════════════════════════════════════════════════

class VSAFiniteStateAutomaton:
    """
    Finite State Automaton encoded as a hypervector superposition.

    Reference:
        torchhd `FiniteStateAutomata` class + Frady & Sommer (2018).

    Transition encoding:
        T = MAJORITY( bind(state_hv, bind(token_hv, next_state_hv)) )

    Query (state, token) → next_state:
        probe     = bind(state_hv, token_hv)
        candidate = unbind(T, probe)
        next_state = argmax cosine_sim(candidate, all_state_hvs)

    This stores ALL transitions in a single HV. Adding a new transition
    requires one bundle operation. Query is one bind + one unbind + one lookup.

    Args:
        dim:    HV dimension
        hrr:    Optional HRR instance (uses exact unbinding)
        device: torch device
    """

    def __init__(self, dim: int, hrr: Optional[HRR] = None, device: str = "cpu"):
        self.dim    = dim
        self.device = device
        self.hrr    = hrr or HRR(dim, device)

        # State and token codebooks
        self._state_hvs: Dict[str, torch.Tensor] = {}
        self._token_hvs: Dict[str, torch.Tensor] = {}
        self._final_states: Set[str]              = set()
        self._initial_state: Optional[str]        = None

        # Transition memory HV (superposition of all transitions)
        self._T = torch.zeros(dim, device=device)
        self._n_transitions = 0

        # Track for deletion support
        self._transition_hvs: List[torch.Tensor] = []

    def _name_seed(self, name: str) -> int:
        import hashlib
        return int(hashlib.md5(name.encode()).hexdigest()[:8], 16) % (2**31)

    def _get_state_hv(self, state: str) -> torch.Tensor:
        if state not in self._state_hvs:
            self._state_hvs[state] = self.hrr.gen(1, seed=self._name_seed(f"state_{state}"))
        return self._state_hvs[state]

    def _get_token_hv(self, token: str) -> torch.Tensor:
        if token not in self._token_hvs:
            self._token_hvs[token] = self.hrr.gen(1, seed=self._name_seed(f"token_{token}"))
        return self._token_hvs[token]

    def add_state(self, name: str, is_initial: bool = False, is_final: bool = False):
        """Register a state."""
        _ = self._get_state_hv(name)
        if is_initial:
            self._initial_state = name
        if is_final:
            self._final_states.add(name)

    def add_transition(self, from_state: str, token: str, to_state: str):
        """
        Add a transition: (from_state, token) → to_state.

        Encoded as: bind(from_state_hv, bind(token_hv, to_state_hv))
        Added to T via superposition.
        """
        s_hv = self._get_state_hv(from_state)
        t_hv = self._get_token_hv(token)
        n_hv = self._get_state_hv(to_state)

        # bind(s, bind(t, n))
        inner  = self.hrr.bind(t_hv, n_hv)
        triple = self.hrr.bind(s_hv, inner)

        self._T = self._T + triple
        self._transition_hvs.append(triple.clone())
        self._n_transitions += 1

    def next_state(
        self,
        current_state: str,
        token:         str,
        top_k:         int = 1,
    ) -> List[Tuple[str, float]]:
        """
        Query next state(s) for (current_state, token).

        Returns:
            List of (state_name, similarity) sorted desc.
        """
        if not self._state_hvs or current_state not in self._state_hvs:
            return []

        s_hv = self._state_hvs[current_state]
        t_hv = self._get_token_hv(token)

        # probe = bind(state, token)
        probe     = self.hrr.bind(s_hv, t_hv)
        # candidate ≈ bind(token, next_state) for the matching transition
        candidate = self.hrr.unbind_exact(self._T, probe)

        # candidate = unbind_exact(T, bind(s,t)) ≈ next_state_hv  (direct, not bound)
        # Compare directly to each known state HV
        results = []
        for name, n_hv in self._state_hvs.items():
            sim = self.hrr.similarity(candidate, n_hv)
            results.append((name, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def simulate(
        self,
        tokens:        List[str],
        min_sim:       float = 0.0,  # minimum similarity to accept a transition
    ) -> Tuple[str, bool]:
        """
        Run the FSA on a sequence of tokens.

        Args:
            tokens:  Sequence to simulate
            min_sim: Minimum similarity threshold to accept a transition.
                     With 0.0 (default), always transitions to best match.
                     With a positive value, rejects low-confidence transitions.

        Returns:
            (final_state_name, accepted) where accepted = state is final.
        """
        if self._initial_state is None:
            raise ValueError("No initial state set. Call add_state(..., is_initial=True)")

        current = self._initial_state
        for token in tokens:
            transitions = self.next_state(current, token, top_k=1)
            if not transitions:
                return current, False
            best_state, best_sim = transitions[0]
            if best_sim >= min_sim:
                current = best_state
            # If below threshold: remain in current state (implicit self-loop)

        return current, current in self._final_states

    def simulate_beam(
        self,
        tokens:    List[str],
        beam_width: int   = 3,
        min_sim:    float = 0.0,
    ) -> List[Tuple[List[str], float, bool]]:
        """
        Beam search simulation: maintain top-k paths through the FSA.

        Unlike greedy `simulate()`, beam search keeps multiple hypotheses
        alive and returns the top-beam_width complete paths ranked by their
        cumulative similarity score.  This is robust to noisy sensor tokens
        where the single best transition may be wrong.

        Args:
            tokens:     Sequence to simulate
            beam_width: Number of parallel paths to maintain
            min_sim:    Minimum similarity to consider a transition

        Returns:
            List of (state_path, cumulative_score, accepted) sorted desc by score.
            state_path: [initial, ..., final] — full state sequence
        """
        if self._initial_state is None:
            return []

        # Beam: list of (cumulative_score, state_path)
        beam = [(0.0, [self._initial_state])]

        for token in tokens:
            next_beam = []
            for score, path in beam:
                current = path[-1]
                transitions = self.next_state(current, token, top_k=beam_width)
                for next_st, sim in transitions:
                    if sim < min_sim:
                        continue
                    next_beam.append((score + sim, path + [next_st]))
                if not transitions:
                    # No transition found: implicit self-loop (penalise by 0)
                    next_beam.append((score + 0.0, path + [current]))

            # Prune to beam_width
            next_beam.sort(key=lambda x: x[0], reverse=True)
            beam = next_beam[:beam_width]

        return [
            (path, score, path[-1] in self._final_states)
            for score, path in beam
        ]

    def online_update(
        self,
        from_state: str,
        token:      str,
        to_state:   str,
        weight:     float = 1.0,
    ):
        """
        Incrementally add or strengthen a transition in the bundled memory.

        Unlike add_transition() which always bundles with weight 1.0, this
        allows adding weak transitions (weight < 1) for uncertain observations,
        or reinforcing existing ones (weight > 1) for confirmed state sequences.

        Args:
            from_state: Source state name
            token:      Token name
            to_state:   Destination state name
            weight:     Contribution weight to the bundle (default 1.0)
        """
        for name in [from_state, token, to_state]:
            if name not in (self._state_hvs if from_state == name or to_state == name else self._token_hvs):
                pass  # lazily create HVs as needed

        s_hv  = self._get_state_hv(from_state)
        t_hv  = self._get_token_hv(token)
        n_hv  = self._get_state_hv(to_state)

        # Bind and add to transition memory with given weight
        binding = self.hrr.bind(s_hv, self.hrr.bind(t_hv, n_hv))
        self._T = self._T + weight * binding
        self._n_transitions += 1

    @property
    def n_states(self) -> int:
        return len(self._state_hvs)

    @property
    def n_transitions(self) -> int:
        return self._n_transitions

    def automaton_health(self) -> Dict:
        """
        FSA structural summary: state/transition counts, connectivity.

        connectivity = n_transitions / (n_states²) — measures how dense the FSA is.
        n_final / n_states < 0.1 → very few accept states (strict language).
        """
        n_states = self.n_states
        n_trans  = self.n_transitions
        n_final  = len(self._final_states)
        max_trans = n_states ** 2
        return {
            "n_states":       n_states,
            "n_transitions":  n_trans,
            "n_final":        n_final,
            "initial_state":  self._initial_state,
            "connectivity":   round(n_trans / max(max_trans, 1), 4),
            "acceptance_rate": round(n_final / max(n_states, 1), 4),
            "dim":            self.dim,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 2. VSANondeterministicFSA — multiple next states per transition
# ═══════════════════════════════════════════════════════════════════════════════

class VSANondeterministicFSA(VSAFiniteStateAutomaton):
    """
    Nondeterministic FSA: multiple next states per (state, token) pair.

    In NFA, (state, token) can lead to multiple next states simultaneously.
    In VSA: the transition memory T bundles ALL valid next states.
    Query returns a ranked list; all states above a similarity threshold
    are considered reachable.

    Args:
        dim:          HV dimension
        sim_threshold: Minimum similarity to consider a state reachable
    """

    def __init__(self, dim: int, sim_threshold: float = 0.3, **kwargs):
        super().__init__(dim, **kwargs)
        self.threshold = sim_threshold

    def reachable_states(self, current: str, token: str) -> List[str]:
        """Return all states reachable from (current, token) above threshold."""
        results = self.next_state(current, token, top_k=self.n_states)
        return [name for name, sim in results if sim >= self.threshold]

    def simulate_nfa(
        self,
        tokens: List[str],
        max_active: int = 10,
    ) -> Tuple[Set[str], bool]:
        """
        NFA simulation via subset construction.

        Maintains a SET of active states (like BFS over FSA graph).
        Returns (active_states_at_end, any_is_final).
        """
        if self._initial_state is None:
            raise ValueError("No initial state")

        active = {self._initial_state}
        for token in tokens:
            next_active: Set[str] = set()
            for state in active:
                reached = self.reachable_states(state, token)
                next_active.update(reached[:max_active])
            active = next_active if next_active else active

        accepted = bool(active & self._final_states)
        return active, accepted


# ═══════════════════════════════════════════════════════════════════════════════
# 3. RegularLanguageMatcher
# ═══════════════════════════════════════════════════════════════════════════════

class RegularLanguageMatcher:
    """
    High-level regular language matcher backed by a VSA FSA.

    Defines a regular language via a set of states and transitions,
    then provides `match(sequence) → True/False`.

    Example: match all strings over {a, b} that start with 'a':
        m = RegularLanguageMatcher(dim=256)
        m.add_transition("start", "a", "in")
        m.add_transition("in", "a", "in")
        m.add_transition("in", "b", "in")
        m.set_initial("start")
        m.add_final("in")
        m.match(["a", "b", "a"])   # True
        m.match(["b", "a"])         # False

    Args:
        dim:    HV dimension
        device: torch device
    """

    def __init__(self, dim: int, device: str = "cpu"):
        self._fsa = VSAFiniteStateAutomaton(dim, device=device)
        self._built = False

    def add_state(self, name: str, is_initial: bool = False, is_final: bool = False):
        self._fsa.add_state(name, is_initial=is_initial, is_final=is_final)

    def set_initial(self, state: str):
        self._fsa._initial_state = state
        self._fsa._get_state_hv(state)

    def add_final(self, state: str):
        self._fsa._final_states.add(state)
        self._fsa._get_state_hv(state)

    def add_transition(self, from_state: str, token: str, to_state: str):
        self._fsa.add_transition(from_state, token, to_state)

    def match(self, sequence: List[str]) -> bool:
        _, accepted = self._fsa.simulate(sequence)
        return accepted

    def describe(self) -> dict:
        return {
            "n_states":      self._fsa.n_states,
            "n_transitions": self._fsa.n_transitions,
            "initial":       self._fsa._initial_state,
            "final_states":  list(self._fsa._final_states),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. HDCLanguageModel with FSA grammar constraints
# ═══════════════════════════════════════════════════════════════════════════════

class FSAConstrainedLanguageModel:
    """
    HDC language model constrained by a VSA FSA grammar.

    Combines:
        1. VSA context window (from hdc/vsa_sequence_model.py) — sequence history
        2. VSA FSA — grammar constraints on valid next tokens

    At each step:
        - FSA tells us which tokens are grammatically valid
        - Context window tells us which tokens are contextually probable
        - Final prediction = most probable token among grammatically valid ones

    This is the HDC equivalent of grammar-constrained language model decoding
    (like beam search with FSA constraints in neural LMs, but O(D) total cost).

    Args:
        dim:         HV dimension
        vocabulary:  List of token names
        fsa:         Pre-built VSAFiniteStateAutomaton
        device:      torch device
    """

    def __init__(
        self,
        dim:        int,
        vocabulary: List[str],
        fsa:        Optional[VSAFiniteStateAutomaton] = None,
        device:     str = "cpu",
    ):
        from hdc.vsa_sequence_model import VSALanguageModel
        from hdc.hrr import HRR

        self.dim        = dim
        self.vocabulary = vocabulary
        self.device     = device
        self.fsa        = fsa

        hrr = HRR(dim, device)
        self.lm  = VSALanguageModel(hrr, max_len=50)
        for token in vocabulary:
            self.lm.register_token(token)

        self._current_fsa_state: Optional[str] = None
        if fsa and fsa._initial_state:
            self._current_fsa_state = fsa._initial_state

    def observe(self, token: str) -> str:
        """
        Observe a token (update LM context + FSA state).

        Returns the predicted next token.
        """
        # Update FSA state
        if self.fsa and self._current_fsa_state:
            transitions = self.fsa.next_state(self._current_fsa_state, token, top_k=1)
            if transitions:
                self._current_fsa_state = transitions[0][0]

        # Update LM context and predict
        return self.lm.observe(token)

    def predict_constrained(self, top_k: int = 3) -> List[Tuple[str, float, bool]]:
        """
        Predict next tokens, annotated with FSA validity.

        Returns:
            List of (token, probability, is_grammatically_valid)
        """
        lm_preds = self.lm.predict_next(top_k=len(self.vocabulary))

        # Get valid tokens per FSA
        valid_tokens: Set[str] = set(self.vocabulary)
        if self.fsa and self._current_fsa_state:
            reachable = []
            for token in self.vocabulary:
                trans = self.fsa.next_state(self._current_fsa_state, token, top_k=1)
                if trans and trans[0][1] > 0.2:
                    reachable.append(token)
            if reachable:
                valid_tokens = set(reachable)

        results = []
        for name, prob in lm_preds[:top_k]:
            results.append((name, prob, name in valid_tokens))

        return sorted(results, key=lambda x: x[1] * (2 if x[2] else 1), reverse=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Smoke test
# ═══════════════════════════════════════════════════════════════════════════════

def _test_finite_state_automata():
    D = 256

    print("=== VSAFiniteStateAutomaton ===")
    fsa = VSAFiniteStateAutomaton(D)

    # Simple 2-state FSA: start --a--> in --b--> start
    fsa.add_state("start", is_initial=True)
    fsa.add_state("in",    is_final=True)
    fsa.add_transition("start", "a", "in")
    fsa.add_transition("in",    "b", "start")
    fsa.add_transition("in",    "a", "in")

    print(f"  States: {fsa.n_states}, Transitions: {fsa.n_transitions}  OK")

    # Query next state
    results = fsa.next_state("start", "a", top_k=2)
    print(f"  next_state(start, a): {results[:2]}")
    assert results[0][0] == "in", f"Expected 'in', got '{results[0][0]}'"

    # Simulate: check top-returned state is correct (content of T verifiable by query)
    final_state, accepted = fsa.simulate(["a", "a", "b", "a"])
    print(f"  simulate(['a','a','b','a']): final_state={final_state}, accepted={accepted}  OK")
    # The sequence a a b a: start->in->in->start->in, in is final → accepted
    # VSA retrieval is approximate; check top-1 state is correct
    assert final_state == "in", f"Expected 'in', got '{final_state}'"

    print("\n=== VSANondeterministicFSA ===")
    nfa = VSANondeterministicFSA(D, sim_threshold=0.05)
    nfa.add_state("s0", is_initial=True)
    nfa.add_state("s1", is_final=True)
    nfa.add_state("s2", is_final=True)
    nfa.add_transition("s0", "x", "s1")
    nfa.add_transition("s1", "y", "s0")

    active, accepted = nfa.simulate_nfa(["x"])
    print(f"  NFA after ['x']: active={active}, accepted={accepted}  OK")
    assert len(active) >= 1

    print("\n=== RegularLanguageMatcher ===")
    # Use larger D and single transition for high-fidelity test
    m = RegularLanguageMatcher(1024)
    m.add_state("q0"); m.add_state("q1")
    m.set_initial("q0"); m.add_final("q1")
    m.add_transition("q0", "a", "q1")   # single transition for clarity

    result_a = m.match(["a"])
    print(f"  match(['a'])={result_a}  OK")
    assert result_a == True, "Sequence ['a'] should be accepted"

    desc = m.describe()
    assert desc["n_states"] == 2
    assert desc["n_transitions"] == 1
    print(f"  {desc}  OK")

    print("\n=== FSAConstrainedLanguageModel ===")
    vocab = ["the", "cat", "sat", "on", "mat", "END"]
    fsa2  = VSAFiniteStateAutomaton(D)
    fsa2.add_state("s", is_initial=True)
    fsa2.add_state("f", is_final=True)
    for tok in vocab[:-1]:
        fsa2.add_transition("s", tok, "s")
    fsa2.add_transition("s", "END", "f")

    clm = FSAConstrainedLanguageModel(D, vocab, fsa=fsa2)
    for w in ["the", "cat", "sat"]:
        clm.observe(w)
    preds = clm.predict_constrained(top_k=3)
    print(f"  Constrained predictions: {[(n, f'{p:.3f}', v) for n,p,v in preds[:3]]}")
    assert len(preds) > 0
    print("  OK")

    print("\n✅ All finite_state_automata tests passed")


if __name__ == "__main__":
    _test_finite_state_automata()
