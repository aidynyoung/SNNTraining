"""
Associative Synthesis of Finite State Automata Using Hyperdimensional Computing
================================================================================
Based on: Osipov, V., et al. (2022)
"Associative Synthesis of Finite State Automata Using Hyperdimensional Computing"
IEEE Access, 10, 125456-125471. DOI: 10.1109/ACCESS.2022.3225430

Key contributions:

1. **FSM Synthesis via HDC** — Finite state machines are synthesized using
   associative operations in hyperdimensional space. States and transitions
   are encoded as hypervectors, and state transitions are performed via
   similarity search.

2. **Associative State Machine** — The state machine is implemented as an
   associative memory: given current state and input, the next state is
   retrieved by unbinding and cleanup.

3. **Compositional FSM** — Multiple FSMs can be composed by bundling their
   transition hypervectors, enabling hierarchical state machines.

4. **Noise-Robust Execution** — The associative nature provides inherent
   robustness to noise in state representations.

Reference:
  Osipov, V., et al. (2022)
  "Associative Synthesis of Finite State Automata Using Hyperdimensional Computing"
  IEEE Access, 10, 125456-125471
"""

import torch
from typing import Optional, List, Tuple, Dict, Set, Any
from hdc.hdc_glue import (
    hv_xor, hv_popcount, hv_hamming_sim, hv_bundle,
    hv_permute, hv_majority, hv_batch_sim, gen_hvs,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Section II: HDC State Machine Components
# ═══════════════════════════════════════════════════════════════════════════════

class HDCStateMachine:
    """
    Finite state machine implemented using hyperdimensional computing.

    Key idea (Osipov 2022, Section III):
    Each state and input symbol is a hypervector. Transitions are encoded as:
        T(s, i) = bind(s, i, s') where s' = next_state(s, i)

    The transition function is stored as a bundled set of transition HVs:
        F = ⊕ bind(s_j, i_k, s'_l) for all transitions

    State transition is performed by:
        s' = unbind(F, bind(s, i)) → cleanup → nearest state

    This provides:
    - O(1) transition lookup (single XOR + similarity search)
    - Robustness to noise in state representations
    - Compositional FSM construction
    """

    def __init__(
        self,
        dim: int = 10000,
        seed: Optional[int] = None,
    ):
        """
        Args:
            dim: Hypervector dimensionality
            seed: Random seed
        """
        self.dim = dim
        self.seed = seed or 42

        # State and input hypervectors
        self._state_hvs: Dict[str, torch.Tensor] = {}
        self._input_hvs: Dict[str, torch.Tensor] = {}
        self._counter = 0

        # Transition function: bundled hypervector of all transitions
        self.transition_hv: torch.Tensor = torch.zeros(dim)

        # Track which transitions have been added
        self._transitions: Set[Tuple[str, str, str]] = set()

        # Codebooks for cleanup
        self._state_codebook: torch.Tensor = torch.zeros(0, dim)
        self._input_codebook: torch.Tensor = torch.zeros(0, dim)

    def _get_state_hv(self, name: str) -> torch.Tensor:
        """Get or create a state hypervector."""
        if name not in self._state_hvs:
            seed = hash(f"state_{name}") & 0x7FFFFFFF
            self._state_hvs[name] = gen_hvs(1, self.dim, seed=seed).squeeze(0)
            # Update codebook
            self._state_codebook = torch.cat([
                self._state_codebook,
                self._state_hvs[name].unsqueeze(0),
            ])
        return self._state_hvs[name]

    def _get_input_hv(self, name: str) -> torch.Tensor:
        """Get or create an input symbol hypervector."""
        if name not in self._input_hvs:
            seed = hash(f"input_{name}") & 0x7FFFFFFF
            self._input_hvs[name] = gen_hvs(1, self.dim, seed=seed).squeeze(0)
            # Update codebook
            self._input_codebook = torch.cat([
                self._input_codebook,
                self._input_hvs[name].unsqueeze(0),
            ])
        return self._input_hvs[name]

    def add_transition(
        self,
        current_state: str,
        input_symbol: str,
        next_state: str,
    ):
        """Add a state transition.

        Args:
            current_state: Current state name
            input_symbol: Input symbol name
            next_state: Next state name
        """
        s = self._get_state_hv(current_state)
        i = self._get_input_hv(input_symbol)
        s_next = self._get_state_hv(next_state)

        # Encode transition: bind(s, i, s_next)
        transition = hv_xor(hv_xor(s, i), s_next)

        # Bundle with existing transitions
        if torch.norm(self.transition_hv) == 0:
            self.transition_hv = transition
        else:
            self.transition_hv = hv_majority(
                hv_bundle(torch.stack([self.transition_hv, transition]))
            )

        self._transitions.add((current_state, input_symbol, next_state))

    def add_transitions(self, transitions: List[Tuple[str, str, str]]):
        """Add multiple transitions at once.

        Args:
            transitions: List of (current_state, input_symbol, next_state)
        """
        for t in transitions:
            self.add_transition(*t)

    def step(
        self,
        current_state: str,
        input_symbol: str,
    ) -> Tuple[str, float]:
        """Execute one state transition.

        Args:
            current_state: Current state name
            input_symbol: Input symbol

        Returns:
            (next_state_name, confidence)
        """
        s = self._get_state_hv(current_state)
        i = self._get_input_hv(input_symbol)

        # Query: bind(s, i) → find matching transition
        query = hv_xor(s, i)
        result = hv_xor(self.transition_hv, query)

        # Cleanup: find nearest state
        sims = hv_batch_sim(result, self._state_codebook)
        best_idx = int(sims.argmax().item())
        confidence = float(sims[best_idx].item())

        # Map index back to state name
        state_names = list(self._state_hvs.keys())
        next_state = state_names[best_idx]

        return next_state, confidence

    def run(
        self,
        initial_state: str,
        input_sequence: List[str],
        max_steps: int = 100,
    ) -> Tuple[List[str], List[float]]:
        """Run the state machine on an input sequence.

        Args:
            initial_state: Starting state name
            input_sequence: List of input symbols
            max_steps: Maximum number of steps

        Returns:
            (state_trace, confidence_trace)
        """
        state_trace = [initial_state]
        confidence_trace = [1.0]

        current = initial_state
        for i, inp in enumerate(input_sequence[:max_steps]):
            next_state, confidence = self.step(current, inp)
            state_trace.append(next_state)
            confidence_trace.append(confidence)
            current = next_state

        return state_trace, confidence_trace

    def get_transition_table(self) -> Dict[Tuple[str, str], str]:
        """Get the full transition table.

        Returns:
            {(state, input): next_state} dictionary
        """
        table = {}
        for s, i, s_next in self._transitions:
            table[(s, i)] = s_next
        return table

    def verify_transition(self, current_state: str, input_symbol: str, expected_next: str) -> bool:
        """Verify a specific transition.

        Args:
            current_state: Current state name
            input_symbol: Input symbol
            expected_next: Expected next state

        Returns:
            True if transition is correct
        """
        next_state, confidence = self.step(current_state, input_symbol)
        return next_state == expected_next


# ═══════════════════════════════════════════════════════════════════════════════
# Section III: Compositional FSM
# ═══════════════════════════════════════════════════════════════════════════════

class CompositionalFSM:
    """
    Compositional finite state machine using HDC.

    Multiple FSMs can be composed by:
    1. **Parallel composition**: Bundle transition HVs of both FSMs
    2. **Sequential composition**: Chain FSMs by binding output/input
    3. **Hierarchical composition**: Sub-FSMs as states of parent FSM

    This enables building complex state machines from simpler ones,
    analogous to hierarchical state machines in software engineering.
    """

    def __init__(
        self,
        dim: int = 10000,
        seed: Optional[int] = None,
    ):
        self.dim = dim
        self.seed = seed or 42
        self.sub_fsms: Dict[str, HDCStateMachine] = {}
        self.parent_fsm: Optional[HDCStateMachine] = None

    def add_sub_fsm(self, name: str, fsm: HDCStateMachine):
        """Add a sub-FSM.

        Args:
            name: Sub-FSM identifier
            fsm: HDCStateMachine instance
        """
        self.sub_fsms[name] = fsm

    def parallel_compose(self, fsm_a: HDCStateMachine, fsm_b: HDCStateMachine) -> HDCStateMachine:
        """Parallel composition of two FSMs.

        The composed FSM's state is the pair (state_a, state_b).
        Transitions happen simultaneously in both FSMs.

        Args:
            fsm_a: First FSM
            fsm_b: Second FSM

        Returns:
            Composed HDCStateMachine
        """
        composed = HDCStateMachine(dim=self.dim, seed=self.seed)

        # Get all transitions from both FSMs
        table_a = fsm_a.get_transition_table()
        table_b = fsm_b.get_transition_table()

        # Find common inputs
        inputs_a = set(i for (_, i) in table_a.keys())
        inputs_b = set(i for (_, i) in table_b.keys())
        common_inputs = inputs_a & inputs_b

        # Create composed transitions
        for inp in common_inputs:
            for (s_a, i_a), s_next_a in table_a.items():
                if i_a != inp:
                    continue
                for (s_b, i_b), s_next_b in table_b.items():
                    if i_b != inp:
                        continue
                    # Composed state: bind(s_a, s_b)
                    composed_state = f"{s_a}_{s_b}"
                    composed_next = f"{s_next_a}_{s_next_b}"
                    composed.add_transition(composed_state, inp, composed_next)

        return composed

    def sequential_compose(
        self,
        fsm_a: HDCStateMachine,
        fsm_b: HDCStateMachine,
        output_to_input: Dict[str, str],
    ) -> HDCStateMachine:
        """Sequential composition: output of A → input of B.

        Args:
            fsm_a: First FSM (produces outputs)
            fsm_b: Second FSM (consumes inputs)
            output_to_input: {fsm_a_output: fsm_b_input} mapping

        Returns:
            Composed HDCStateMachine
        """
        composed = HDCStateMachine(dim=self.dim, seed=self.seed)

        table_a = fsm_a.get_transition_table()
        table_b = fsm_b.get_transition_table()

        # For each transition in A, chain to B
        for (s_a, i_a), s_next_a in table_a.items():
            # Map A's next state to B's input
            if s_next_a in output_to_input:
                b_input = output_to_input[s_next_a]
                # Find B transitions for this input
                for (s_b, i_b), s_next_b in table_b.items():
                    if i_b == b_input:
                        composed_state = f"{s_a}_{s_b}"
                        composed_next = f"{s_next_a}_{s_next_b}"
                        composed.add_transition(composed_state, i_a, composed_next)

        return composed


# ═══════════════════════════════════════════════════════════════════════════════
# Section IV: Applications
# ═══════════════════════════════════════════════════════════════════════════════

class PatternRecognizerFSM:
    """
    Pattern recognition using HDC-based FSM.

    Recognizes patterns in sequential data by encoding the pattern
    as a state machine and running it on input sequences.

    Example: recognize "101" pattern in binary sequences:
        States: q0 (start), q1 (saw "1"), q2 (saw "10"), q3 (saw "101" = accept)
        Transitions:
            q0 --0--> q0, q0 --1--> q1
            q1 --0--> q2, q1 --1--> q1
            q2 --0--> q0, q2 --1--> q3
            q3 --0--> q0, q3 --1--> q1
    """

    def __init__(
        self,
        dim: int = 10000,
        seed: Optional[int] = None,
    ):
        self.dim = dim
        self.seed = seed or 42
        self.fsm = HDCStateMachine(dim=dim, seed=seed)
        self.accept_states: Set[str] = set()

    def build_from_pattern(self, pattern: str, alphabet: Set[str]):
        """Build a pattern-recognizing FSM.

        Uses the Knuth-Morris-Pratt prefix function to build
        the minimal DFA for pattern matching.

        Args:
            pattern: Pattern string to recognize
            alphabet: Set of input symbols
        """
        n = len(pattern)
        states = [f"q{i}" for i in range(n + 1)]

        # Build prefix function (KMP)
        pi = [0] * n
        for i in range(1, n):
            j = pi[i - 1]
            while j > 0 and pattern[i] != pattern[j]:
                j = pi[j - 1]
            if pattern[i] == pattern[j]:
                j += 1
            pi[i] = j

        # Build transitions
        for i, state in enumerate(states):
            for symbol in alphabet:
                if i < n and symbol == pattern[i]:
                    next_state = f"q{i + 1}"
                else:
                    # Fallback using prefix function
                    j = i
                    while j > 0 and (j >= n or symbol != pattern[j]):
                        j = pi[j - 1] if j > 0 else 0
                    if j < n and symbol == pattern[j]:
                        next_state = f"q{j + 1}"
                    else:
                        next_state = "q0"

                self.fsm.add_transition(state, symbol, next_state)

        # Accept state is the last state
        self.accept_states.add(f"q{n}")

    def recognize(self, sequence: List[str]) -> Tuple[bool, List[str]]:
        """Check if a sequence matches the pattern.

        Args:
            sequence: List of input symbols

        Returns:
            (matched, state_trace)
        """
        state_trace, _ = self.fsm.run("q0", sequence)
        final_state = state_trace[-1]
        return final_state in self.accept_states, state_trace

    def build_from_regex(self, regex_pattern: str):
        """Build FSM from a simple regex pattern.

        Supports: concatenation, | (or), * (Kleene star)

        Args:
            regex_pattern: Simple regex pattern
        """
        # Simplified: build for common patterns
        # Full regex → FSM conversion would use Thompson's construction
        alphabet = set(c for c in regex_pattern if c not in "()|*")

        if "|" in regex_pattern:
            # Union: (A|B)
            parts = regex_pattern.strip("()").split("|")
            for part in parts:
                self.build_from_pattern(part, alphabet)
        elif "*" in regex_pattern:
            # Kleene star: A*
            base = regex_pattern.replace("*", "")
            self.build_from_pattern(base, alphabet)
            # Add self-loop on accept
            for accept in self.accept_states:
                for sym in alphabet:
                    self.fsm.add_transition(accept, sym, accept)
        else:
            self.build_from_pattern(regex_pattern, alphabet)


# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_hdc_state_machine():
    """Verify HDC state machine operations."""
    print("=" * 60)
    print("Testing HDC State Machine (Osipov 2022)")
    print("=" * 60)

    dim = 1000
    fsm = HDCStateMachine(dim=dim)

    # Build a simple turnstile FSM
    # States: locked, unlocked
    # Inputs: coin, push
    # Transitions:
    #   locked --coin--> unlocked
    #   locked --push--> locked
    #   unlocked --coin--> unlocked
    #   unlocked --push--> locked
    fsm.add_transitions([
        ("locked", "coin", "unlocked"),
        ("locked", "push", "locked"),
        ("unlocked", "coin", "unlocked"),
        ("unlocked", "push", "locked"),
    ])

    # Test transitions
    print("\n  Testing turnstile FSM:")
    tests = [
        ("locked", "coin", "unlocked"),
        ("unlocked", "push", "locked"),
        ("locked", "push", "locked"),
        ("unlocked", "coin", "unlocked"),
    ]

    all_correct = True
    for s, i, expected in tests:
        next_state, confidence = fsm.step(s, i)
        correct = next_state == expected
        status = "✅" if correct else "❌"
        print(f"    {s} --{i}--> {next_state} (expected {expected}) [{status}] conf={confidence:.4f}")
        if not correct:
            all_correct = False

    # Test sequence
    print("\n  Running sequence: coin, push, coin, push")
    trace, confs = fsm.run("locked", ["coin", "push", "coin", "push"])
    print(f"    State trace: {' → '.join(trace)}")
    print(f"    Final state: {trace[-1]} (expected 'locked')")

    print(f"\n  {'✅' if all_correct else '❌'} HDC state machine test complete!")


def test_pattern_recognizer():
    """Verify pattern recognition FSM."""
    print("=" * 60)
    print("Testing Pattern Recognition FSM (Osipov 2022)")
    print("=" * 60)

    dim = 1000
    recognizer = PatternRecognizerFSM(dim=dim)
    recognizer.build_from_pattern("101", {"0", "1"})

    # Test sequences
    test_cases = [
        ("101", True),
        ("0101", True),
        ("1101", True),
        ("100", False),
        ("111", False),
        ("1010", True),
    ]

    all_correct = True
    for seq_str, expected in test_cases:
        seq = list(seq_str)
        matched, trace = recognizer.recognize(seq)
        correct = matched == expected
        status = "✅" if correct else "❌"
        print(f"  Pattern '{seq_str}': matched={matched} (expected {expected}) [{status}]")
        if not correct:
            all_correct = False

    print(f"\n  {'✅' if all_correct else '❌'} Pattern recognition test complete!")


if __name__ == "__main__":
    test_hdc_state_machine()
    print()
    test_pattern_recognizer()
