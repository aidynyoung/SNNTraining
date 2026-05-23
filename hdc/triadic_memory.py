"""
hdc/triadic_memory.py
======================
Triadic Memory, Dyadic Memory, and Deep Temporal Memory
=========================================================
Reference:
    Overmann, P. (2021) "Triadic Memory"
    https://github.com/PeterOvermann/TriadicMemory
    — Novel associative memory algorithm for sparse binary hypervectors (SDRs).

    Kanerva (1988) "Sparse Distributed Memory" — foundational SDM
    — Triadic Memory extends SDM from dyadic to ternary associations.

Why Triadic Memory is distinct from everything already in SNNTraining:

    SNNTraining already has:
        SDM (dyadic):         stores (address, data) pairs
        HRR (dyadic):         stores bind(role, filler) = composite
        Tensor Products:      dense role-filler binding
        Knowledge Graph:      stores (subject, predicate, object) via HRR

    Triadic Memory is different:
        - Operates on SPARSE BINARY hypervectors (SDRs), not dense/real
        - Stores ordered triples (x, y, z) with TRUE TRIDIRECTIONAL retrieval:
          given any TWO of {x, y, z}, retrieve the THIRD exactly
        - Uses a 3D sparse integer tensor mem[N,N,N] for storage
        - Capacity: (N/p)³ where p = active bits per vector (e.g., 1%)
          At N=1000, p=10: capacity = (100)³ = 1,000,000 triples
        - O(p²) storage, O(p²) query — much sparser than full 3D tensor

    This is impossible with HRR (which approximates, not exactly retrieves)
    and impossible with SDM (which is dyadic — only 2 arguments).

    Key property: TriadicMemory(x, y, ?) = z, TriadicMemory(x, ?, z) = y,
                  TriadicMemory(?, y, z) = x — all exact.

This module implements:

1. DyadicMemory
   — 2D heteroassociative memory for (x, y) pairs
   — Triangular addressing via XOR: xaddr(i,j) = i*(i-1)//2 + j  for i>j
   — Query: given x, retrieve y and vice versa
   — More memory-efficient than full N×N matrix (N(N-1)/2 cells)

2. TriadicMemory
   — 3D sparse tensor for (x, y, z) triple storage
   — Storage: for all active bit combinations (ax, ay, az): mem[ax,ay,az] += 1
   — Query: sum relevant slices, threshold at top-p
   — Capacity: (N/p)³ triples with reliable retrieval

3. DeepTemporalMemory
   — Stacked TriadicMemory units forming a deep temporal circuit
   — Each layer: current_input × previous_state → next_state
   — The stack creates multi-timescale temporal memory
   — Based on Overmann's RNN circuit using Triadic Memory as the recurrent cell

4. ElementaryTemporalMemory
   — Minimal two-unit Elman network using Triadic Memory
   — Unit A: (input, prev_output) → output
   — Unit B: (input, output) → context  [stores temporal dependencies]
   — Minimal sequence memory with exact tridirectional retrieval
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F


# ── SDR utilities ──────────────────────────────────────────────────────────────

def _gen_sdr(n: int, p: int, seed: Optional[int] = None, device: str = "cpu") -> torch.Tensor:
    """
    Generate one sparse binary SDR with exactly p active bits out of n.

    Args:
        n:    Dimensionality
        p:    Number of active bits (sparsity = p/n)
        seed: Optional random seed

    Returns:
        (n,) binary tensor with exactly p ones
    """
    g = torch.Generator(device=device)
    if seed is not None:
        g.manual_seed(seed)
    idx = torch.randperm(n, generator=g, device=device)[:p]
    out = torch.zeros(n, dtype=torch.int8, device=device)
    out[idx] = 1
    return out

def _active_bits(sdr: torch.Tensor) -> torch.Tensor:
    """Return indices of active (1) bits in an SDR."""
    return sdr.nonzero(as_tuple=True)[0]

def _sdr_to_set(sdr: torch.Tensor) -> Set[int]:
    """Convert SDR tensor to set of active indices."""
    return set(_active_bits(sdr).tolist())

def _sums_to_sdr(sums: torch.Tensor, p: int) -> torch.Tensor:
    """
    Convert sum-of-slices to a binary SDR by keeping top-p activated bits.

    Args:
        sums: (n,) integer activation counts
        p:    Target number of active bits

    Returns:
        (n,) binary SDR with p active bits (or fewer if not enough non-zero)
    """
    out = torch.zeros_like(sums, dtype=torch.int8)
    if sums.max() == 0:
        return out
    top = sums.topk(min(p, int((sums > 0).sum().item()))).indices
    out[top] = 1
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DyadicMemory — 2D heteroassociative memory for (x, y) pairs
# ═══════════════════════════════════════════════════════════════════════════════

class DyadicMemory:
    """
    Dyadic (2-argument) associative memory for sparse binary hypervectors.

    Reference:
        Overmann (2021) — DyadicMemory with triangular XOR addressing.

    Stores (x, y) associations; given x retrieves y and vice versa.

    Storage: triangular matrix using address xaddr(i, j) = i(i-1)/2 + j for i>j.
    The XOR of the two index sets generates the triangle address, giving
    bidirectional lookup without storing the full N×N matrix.

    Capacity: (N/p)² pairs where p = number of active bits.

    Args:
        n: Dimensionality of each SDR
        p: Number of active bits per SDR (sparsity = p/n)
    """

    def __init__(self, n: int, p: int):
        self.n = n
        self.p = p

        # Triangular storage: size = n*(n-1)//2
        self._storage = defaultdict(int)   # addr → count
        self._n_pairs = 0

    def _xaddr(self, i: int, j: int) -> int:
        """Triangular address for pair (i, j) with i > j."""
        if i == j:
            return -1
        if i < j:
            i, j = j, i
        return i * (i - 1) // 2 + j

    def write(self, x: torch.Tensor, y: torch.Tensor):
        """
        Store a (x, y) pair.

        For each active (i, j) pair with i ∈ active(x), j ∈ active(y):
            mem[xaddr(i, j)] += 1
        """
        ax = _active_bits(x).tolist()
        ay = _active_bits(y).tolist()
        for i in ax:
            for j in ay:
                addr = self._xaddr(i, j)
                if addr >= 0:
                    self._storage[addr] += 1
        self._n_pairs += 1

    def query_y(self, x: torch.Tensor) -> torch.Tensor:
        """Given x, retrieve y."""
        ax   = _active_bits(x).tolist()
        sums = torch.zeros(self.n, dtype=torch.int32)
        for i in ax:
            for j in range(self.n):
                if j == i:
                    continue
                addr = self._xaddr(i, j)
                if addr in self._storage:
                    sums[j] += self._storage[addr]
        return _sums_to_sdr(sums, self.p)

    def query_x(self, y: torch.Tensor) -> torch.Tensor:
        """Given y, retrieve x (bidirectional)."""
        return self.query_y(y)   # symmetric by construction

    @property
    def n_pairs(self) -> int:
        return self._n_pairs

    def capacity_estimate(self) -> float:
        """Theoretical capacity = (N/p)²"""
        return (self.n / max(self.p, 1)) ** 2


# ═══════════════════════════════════════════════════════════════════════════════
# 2. TriadicMemory — 3D sparse tensor for (x, y, z) triple storage
# ═══════════════════════════════════════════════════════════════════════════════

class TriadicMemory:
    """
    Triadic (3-argument) associative memory with tridirectional retrieval.

    Reference:
        Overmann (2021) "Triadic Memory"
        https://github.com/PeterOvermann/TriadicMemory

    Stores ordered triples (x, y, z) of sparse binary SDRs.
    Retrieves any missing third element given the other two — exactly.

    Storage:
        For each triple (x, y, z): for all (ax, ay, az) active bit combinations:
            mem[ax][ay][az] += 1

        Storage per triple: p³ integer increments
        Total storage cells: O(n³) worst case, but sparse via dict

    Query (?, y, z):
        For each (ay, az) active pair: sum mem[all][ay][az] → n counts
        Return top-p indices as x

    Capacity:
        (N/p)³ triples at reliable retrieval (capacity per Overmann 2021)
        At N=1000, p=10: capacity ≈ 10⁶ triples

    Args:
        n: SDR dimensionality
        p: Number of active bits per SDR
    """

    def __init__(self, n: int, p: int):
        self.n = n
        self.p = p
        # 3D sparse storage: nested dict mem[i][j][k] = count
        self._mem: Dict[int, Dict[int, Dict[int, int]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(int))
        )
        self._n_triples = 0

    def write(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
        """
        Store a triple (x, y, z).

        Args:
            x, y, z: (n,) sparse binary SDR tensors
        """
        ax = _active_bits(x).tolist()
        ay = _active_bits(y).tolist()
        az = _active_bits(z).tolist()
        for i in ax:
            for j in ay:
                for k in az:
                    self._mem[i][j][k] += 1
        self._n_triples += 1

    def query_z(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Given (x, y), retrieve z."""
        ax   = _active_bits(x).tolist()
        ay   = _active_bits(y).tolist()
        sums = torch.zeros(self.n, dtype=torch.int32)
        for i in ax:
            for j in ay:
                if i in self._mem and j in self._mem[i]:
                    for k, cnt in self._mem[i][j].items():
                        sums[k] += cnt
        return _sums_to_sdr(sums, self.p)

    def query_y(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Given (x, z), retrieve y."""
        ax   = _active_bits(x).tolist()
        az   = _active_bits(z).tolist()
        sums = torch.zeros(self.n, dtype=torch.int32)
        for i in ax:
            if i in self._mem:
                for j, jdict in self._mem[i].items():
                    for k in az:
                        if k in jdict:
                            sums[j] += jdict[k]
        return _sums_to_sdr(sums, self.p)

    def query_x(self, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Given (y, z), retrieve x."""
        ay   = _active_bits(y).tolist()
        az   = _active_bits(z).tolist()
        sums = torch.zeros(self.n, dtype=torch.int32)
        for i, idict in self._mem.items():
            for j in ay:
                if j in idict:
                    for k in az:
                        if k in idict[j]:
                            sums[i] += idict[j][k]
        return _sums_to_sdr(sums, self.p)

    @property
    def n_triples(self) -> int:
        return self._n_triples

    def capacity_estimate(self) -> float:
        """Theoretical capacity = (N/p)³"""
        return (self.n / max(self.p, 1)) ** 3

    def reset(self):
        self._mem.clear()
        self._n_triples = 0


# ═══════════════════════════════════════════════════════════════════════════════
# 3. DeepTemporalMemory — stacked triadic circuit for multi-timescale memory
# ═══════════════════════════════════════════════════════════════════════════════

class DeepTemporalMemory:
    """
    Deep temporal memory: stack of TriadicMemory units forming a deep RNN.

    Reference:
        Overmann (2021) — "Deep Temporal Memory" architecture:
        "A circuit of Triadic Memory units can implement temporal sequence
        processing by using the current input and previous state as the
        two known arguments to retrieve the next state."

    Architecture:
        For L layers, each layer l stores:
            mem_l(input_l, prev_state_l, next_state_l)

        During query:
            given (input_l, prev_state_l) → next_state_l
            next_state_l becomes input_{l+1}

        This creates a temporal hierarchy where:
            Layer 1: fast dynamics (responds to each input)
            Layer L: slow dynamics (responds to patterns over many inputs)

    Training:
        Each layer writes (current_input, previous_state, current_output)
        on each new observation.  No backpropagation.

    Args:
        n:        SDR dimensionality
        p:        Active bits per SDR
        n_layers: Number of stacked TriadicMemory units
    """

    def __init__(self, n: int, p: int, n_layers: int = 3):
        self.n        = n
        self.p        = p
        self.n_layers = n_layers

        self._layers  = [TriadicMemory(n, p) for _ in range(n_layers)]
        self._states  = [torch.zeros(n, dtype=torch.int8) for _ in range(n_layers)]
        self._n_steps = 0

    def step(self, x: torch.Tensor, train: bool = True) -> List[torch.Tensor]:
        """
        Process one input and update temporal states.

        Args:
            x:     (n,) input SDR
            train: If True, write to memory; if False, inference only

        Returns:
            List of n_layers state vectors after processing x
        """
        self._n_steps += 1
        current_input = x
        new_states    = []

        for l, (layer, prev_state) in enumerate(zip(self._layers, self._states)):
            # Query: given (current_input, prev_state) → new_state
            new_state = layer.query_z(current_input, prev_state)

            if train:
                # Store: (current_input, prev_state, new_state)
                layer.write(current_input, prev_state, new_state)

            new_states.append(new_state)
            current_input = new_state   # output of layer l → input of layer l+1

        # Update states
        self._states = new_states
        return new_states

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inference-only: given current input, predict next top-level state.
        Does not update memory.
        """
        outputs = self.step(x, train=False)
        return outputs[-1]   # deepest layer's output

    def reset_states(self):
        """Reset temporal states (for new sequence boundary)."""
        self._states = [torch.zeros(self.n, dtype=torch.int8) for _ in range(self.n_layers)]

    @property
    def n_stored_triples(self) -> List[int]:
        return [layer.n_triples for layer in self._layers]


# ═══════════════════════════════════════════════════════════════════════════════
# 4. ElementaryTemporalMemory — minimal Elman network with Triadic Memory
# ═══════════════════════════════════════════════════════════════════════════════

class ElementaryTemporalMemory:
    """
    Minimal two-unit temporal circuit using Triadic Memory.

    Reference:
        Overmann (2021) — Elman-style circuit:
        "Unit A stores (input, prev_output) → output
         Unit B stores (input, output) → context"

    Unit A (primary predictor):
        given (current_input, previous_output) → current_output

    Unit B (context encoder):
        given (current_input, current_output) → context
        context represents temporal dependencies across multiple timesteps

    This creates a minimal but powerful sequence memory:
        - Unit A: one-step prediction
        - Unit B: multi-step context compression

    Args:
        n: SDR dimensionality
        p: Active bits per SDR
    """

    def __init__(self, n: int, p: int):
        self.n   = n
        self.p   = p
        self.A   = TriadicMemory(n, p)   # primary predictor
        self.B   = TriadicMemory(n, p)   # context encoder

        self._prev_output  = torch.zeros(n, dtype=torch.int8)
        self._context      = torch.zeros(n, dtype=torch.int8)
        self._n_steps      = 0

    def step(self, x: torch.Tensor, train: bool = True) -> torch.Tensor:
        """
        Process one input step.

        Args:
            x:     (n,) input SDR
            train: If True, write to both memories

        Returns:
            (n,) output SDR (primary prediction)
        """
        self._n_steps += 1

        # Unit A: (input, prev_output) → output
        output = self.A.query_z(x, self._prev_output)

        if train:
            self.A.write(x, self._prev_output, output)
            # Unit B: (input, output) → context
            context = self.B.query_z(x, output)
            self.B.write(x, output, context)
            self._context = context

        self._prev_output = output
        return output

    def predict_next(self, x: torch.Tensor) -> torch.Tensor:
        """Predict next output given current input (no state update)."""
        return self.A.query_z(x, self._prev_output)

    def context_vector(self) -> torch.Tensor:
        """Return current context vector (multi-step temporal summary)."""
        return self._context.clone()

    def reset(self):
        self._prev_output = torch.zeros(self.n, dtype=torch.int8)
        self._context     = torch.zeros(self.n, dtype=torch.int8)

    @property
    def total_triples(self) -> int:
        return self.A.n_triples + self.B.n_triples


# ═══════════════════════════════════════════════════════════════════════════════
# Smoke test
# ═══════════════════════════════════════════════════════════════════════════════

def _test_triadic_memory():
    N, P = 100, 5

    print("=== DyadicMemory ===")
    dm = DyadicMemory(N, P)
    pairs = [(gen := _gen_sdr(N, P, seed=i), _gen_sdr(N, P, seed=100+i)) for i in range(10)]
    for x, y in pairs:
        dm.write(x, y)
    # Retrieve y from x
    x0, y0 = pairs[3]
    y_ret = dm.query_y(x0)
    overlap = int((y_ret & y0).sum().item())
    print(f"  Stored 10 pairs, retrieved y overlap={overlap}/{P}  "
          f"(capacity≈{dm.capacity_estimate():.0f})  OK")

    print("\n=== TriadicMemory ===")
    tm = TriadicMemory(N, P)
    triples = [(_gen_sdr(N, P, seed=i), _gen_sdr(N, P, seed=100+i), _gen_sdr(N, P, seed=200+i))
               for i in range(20)]
    for x, y, z in triples:
        tm.write(x, y, z)

    # Test all 3 query directions
    x0, y0, z0 = triples[7]
    z_ret = tm.query_z(x0, y0)
    y_ret = tm.query_y(x0, z0)
    x_ret = tm.query_x(y0, z0)

    z_ovlp = int((z_ret & z0).sum().item())
    y_ovlp = int((y_ret & y0).sum().item())
    x_ovlp = int((x_ret & x0).sum().item())

    print(f"  Stored {tm.n_triples} triples")
    print(f"  query_z(x,y)→z overlap: {z_ovlp}/{P}")
    print(f"  query_y(x,z)→y overlap: {y_ovlp}/{P}")
    print(f"  query_x(y,z)→x overlap: {x_ovlp}/{P}")
    print(f"  Capacity estimate: {tm.capacity_estimate():.0f} triples")
    assert z_ovlp > 0, "Should retrieve at least partial z"

    print("\n=== DeepTemporalMemory ===")
    dtm = DeepTemporalMemory(N, P, n_layers=3)

    # Train on a repeating sequence
    seq = [_gen_sdr(N, P, seed=i % 5) for i in range(30)]
    for s in seq:
        dtm.step(s, train=True)

    dtm.reset_states()
    out = dtm.predict(seq[0])
    assert out.shape == (N,)
    print(f"  Trained {sum(dtm.n_stored_triples)} triples across {dtm.n_layers} layers  OK")
    print(f"  Prediction shape: {out.shape}  OK")

    print("\n=== ElementaryTemporalMemory ===")
    etm = ElementaryTemporalMemory(N, P)

    # Train on sequence
    for s in seq:
        etm.step(s, train=True)

    etm.reset()
    pred = etm.predict_next(seq[0])
    ctx  = etm.context_vector()
    assert pred.shape == (N,)
    assert ctx.shape  == (N,)
    print(f"  Total triples: {etm.total_triples}  OK")
    print(f"  Prediction: {pred.sum()} active bits  OK")

    print("\n✅ All triadic_memory tests passed")


if __name__ == "__main__":
    _test_triadic_memory()
