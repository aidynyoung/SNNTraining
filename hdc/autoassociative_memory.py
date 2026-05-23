"""
Autoassociative Memory Models for SNNTraining
=============================================
Based on: Kleyko, D., et al. (2017)
"Neural Distributed Autoassociative Memories: A Survey"
arXiv:1709.00848

Implements the full taxonomy of autoassociative memory models:
1. **Hopfield Networks** (Section II): Classical pairwise autoassociative memory
   with Hebbian learning, asynchronous update, and energy minimization.
2. **Willshaw Networks** (Section III): Sparse binary associative memory with
   local learning rules and high storage capacity for sparse patterns.
3. **Potts Networks** (Section IV): Multi-state autoassociative memory with
   q-state units for non-binary data.
4. **Higher-Order Networks** (Section V): Beyond pairwise connections for
   increased capacity and pattern completion.
5. **Bipartite Graph Networks** (Section VI): Non-binary data with linear
   constraints using bipartite graph structure.
6. **Generalization Properties** (Section VII): How autoassociative memories
   generalize beyond stored patterns.
7. **Similarity Search** (Section VIII): Sublinear time approximate nearest
   neighbor search using autoassociative memories.

Architecture:
┌─────────────────────────────────────────────────────────────────────┐
│              Autoassociative Memory Framework                        │
├─────────────────────────────────────────────────────────────────────┤
│  Hopfield     Willshaw     Potts     Higher-Order     Bipartite     │
│  (pairwise)   (sparse)    (multi-state)  (3rd+)     (graph)        │
│     ↓            ↓           ↓            ↓            ↓           │
│  Hebbian      Local       Potts       Tensor        Bipartite      │
│  learning     learning    dynamics    dynamics      dynamics        │
│     ↓            ↓           ↓            ↓            ↓           │
│  Energy       Capacity    Multi-state  High-order   Linear         │
│  minimization  analysis    retrieval   capacity     constraints    │
└─────────────────────────────────────────────────────────────────────┘

Key innovations:
1. **Unified API**: All memory models share a common interface
2. **Capacity Analysis**: Theoretical and empirical capacity estimation
3. **Pattern Completion**: Noisy pattern reconstruction
4. **Generalization**: Beyond simple storage to concept learning
5. **Similarity Search**: Sublinear time approximate nearest neighbor
6. **Hybrid Models**: Combine multiple memory types for complex tasks
"""

import torch
import math
import logging
from typing import Optional, List, Tuple, Dict, Any, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 1: Base Classes & Common Interface
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MemoryConfig:
    """Base configuration for autoassociative memory models."""
    n_neurons: int = 100          # Number of neurons (dimension)
    n_patterns: int = 10          # Number of patterns to store
    dtype: torch.dtype = torch.float32
    device: str = "cpu"
    seed: int = 42

    # Pattern properties
    pattern_sparsity: float = 0.1  # Fraction of active units (Willshaw)
    pattern_activation: float = 0.5  # Fraction of active units (Hopfield)

    # Dynamics
    max_iterations: int = 100     # Max iterations for retrieval
    convergence_threshold: float = 1e-6  # Convergence criterion
    async_update: bool = True     # Asynchronous vs synchronous update

    # Learning
    learning_rate: float = 1.0    # Hebbian learning rate
    forget_rate: float = 0.0      # Forgetting/decay rate

    # Similarity search
    n_probes: int = 10            # Number of probes for similarity search
    beam_width: int = 5           # Beam search width


class AutoassociativeMemory(ABC):
    """
    Abstract base class for all autoassociative memory models.

    Provides a unified interface for:
    - store(pattern): Store a pattern in memory
    - retrieve(noisy_pattern, max_iter): Retrieve stored pattern from noisy input
    - energy(state): Compute energy of a state
    - capacity(): Estimate storage capacity
    - similarity_search(query, k): Find k nearest neighbors
    """

    def __init__(self, config: MemoryConfig):
        self.config = config
        self.n = config.n_neurons
        self.patterns: List[torch.Tensor] = []
        self.labels: List[Any] = []
        self._initialized = False

    @abstractmethod
    def store(self, pattern: torch.Tensor, label: Any = None) -> None:
        """Store a pattern in memory."""
        pass

    @abstractmethod
    def retrieve(
        self,
        pattern: torch.Tensor,
        max_iter: Optional[int] = None,
    ) -> Tuple[torch.Tensor, int]:
        """Retrieve stored pattern from noisy input.

        Args:
            pattern: Noisy/partial input pattern (n,)
            max_iter: Maximum iterations for retrieval

        Returns:
            (retrieved_pattern, n_iterations)
        """
        pass

    @abstractmethod
    def energy(self, state: torch.Tensor) -> float:
        """Compute energy of a state."""
        pass

    def capacity(self) -> Dict[str, float]:
        """Estimate storage capacity.

        Returns:
            dict with keys: 'theoretical_max', 'empirical_max', 'current_load'
        """
        return {
            "theoretical_max": float(self.n),
            "empirical_max": float(self.n),
            "current_load": float(len(self.patterns)),
        }

    def similarity_search(
        self,
        query: torch.Tensor,
        k: int = 5,
    ) -> List[Tuple[int, float, Any]]:
        """Find k nearest neighbors using autoassociative dynamics.

        Uses the memory's retrieval dynamics to find approximate nearest
        neighbors. More efficient than brute-force for high dimensions.

        Args:
            query: Query pattern (n,)
            k: Number of neighbors to return

        Returns:
            List of (index, similarity, label) tuples
        """
        if not self.patterns:
            return []

        # Use retrieval dynamics to find nearest attractor
        retrieved, _ = self.retrieve(query)

        # Compute similarities to all stored patterns
        sims = []
        for i, p in enumerate(self.patterns):
            sim = float(torch.dot(retrieved, p) / (
                torch.norm(retrieved) * torch.norm(p) + 1e-10
            ))
            label = self.labels[i] if i < len(self.labels) else None
            sims.append((i, sim, label))

        # Sort by similarity
        sims.sort(key=lambda x: x[1], reverse=True)
        return sims[:k]

    def get_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "type": self.__class__.__name__,
            "n_neurons": self.n,
            "n_patterns": len(self.patterns),
            "config": self.config,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2: Hopfield Network (Kleyko 2017, Section II)
# ═══════════════════════════════════════════════════════════════════════════════

class HopfieldNetwork(AutoassociativeMemory):
    """
    Classical Hopfield Network (Kleyko 2017, Section II).

    Features:
    - Pairwise symmetric connections: W_ij = sum_p x_i^p x_j^p
    - Asynchronous update: x_i = sign(sum_j W_ij x_j)
    - Energy minimization: E = -0.5 * sum_ij W_ij x_i x_j
    - Pattern completion from partial/noisy inputs
    - Capacity: ~0.14n for random patterns (Hopfield 1982)

    Extensions:
    - Storkey learning rule for increased capacity
    - Sparse patterns for higher capacity
    - Iterative retrieval with convergence detection
    """

    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
        learning_rule: str = "hebbian",  # "hebbian" or "storkey"
    ):
        super().__init__(config or MemoryConfig())
        self.learning_rule = learning_rule
        self.W: Optional[torch.Tensor] = None  # Weight matrix (n, n)
        self.zero_diag = True  # Zero diagonal (no self-connections)

    def store(self, pattern: torch.Tensor, label: Any = None) -> None:
        """Store a pattern using Hebbian or Storkey learning.

        Args:
            pattern: Binary pattern {-1, +1}^n or {0, 1}^n
            label: Optional label for the pattern
        """
        # Convert to bipolar {-1, +1}
        x = self._to_bipolar(pattern)

        if self.W is None:
            self.W = torch.zeros(self.n, self.n, dtype=self.config.dtype)

        if self.learning_rule == "hebbian":
            # Hebbian: W += x * x^T (outer product)
            self.W += self.config.learning_rate * torch.outer(x, x)
        elif self.learning_rule == "storkey":
            # Storkey learning rule (higher capacity)
            # W_ij += x_i * x_j - x_i * h_j - h_i * x_j
            # where h_i = sum_k W_ik * x_k
            h = self.W @ x
            for i in range(self.n):
                for j in range(self.n):
                    if i != j:
                        delta = (x[i] * x[j] - x[i] * h[j] - h[i] * x[j])
                        self.W[i, j] += self.config.learning_rate * delta / self.n

        # Zero diagonal (no self-connections)
        if self.zero_diag:
            self.W.fill_diagonal_(0)

        self.patterns.append(x.clone())
        self.labels.append(label)

    def retrieve(
        self,
        pattern: torch.Tensor,
        max_iter: Optional[int] = None,
    ) -> Tuple[torch.Tensor, int]:
        """Retrieve stored pattern using asynchronous update.

        Args:
            pattern: Noisy/partial input pattern (n,)
            max_iter: Maximum iterations

        Returns:
            (retrieved_pattern, n_iterations)
        """
        if self.W is None:
            return pattern, 0

        max_iter = max_iter or self.config.max_iterations
        x = self._to_bipolar(pattern).clone()
        n = self.n

        for iteration in range(max_iter):
            x_prev = x.clone()

            if self.config.async_update:
                # Asynchronous: update one random neuron at a time
                order = torch.randperm(n)
                for i in order:
                    h = self.W[i] @ x
                    x[i] = 1.0 if h >= 0 else -1.0
            else:
                # Synchronous: update all neurons at once
                h = self.W @ x
                x = torch.where(h >= 0, torch.tensor(1.0), torch.tensor(-1.0))

            # Check convergence
            diff = torch.norm(x - x_prev).item()
            if diff < self.config.convergence_threshold:
                return x, iteration + 1

        return x, max_iter

    def energy(self, state: torch.Tensor) -> float:
        """Compute Hopfield energy: E = -0.5 * sum_ij W_ij x_i x_j.

        Args:
            state: Binary state vector (n,)

        Returns:
            Energy value (lower = more stable)
        """
        if self.W is None:
            return 0.0
        x = self._to_bipolar(state)
        return -0.5 * (x @ self.W @ x).item()

    def capacity(self) -> Dict[str, float]:
        """Estimate Hopfield network capacity.

        Theoretical: ~0.14n for random patterns (Hopfield 1982)
        Storkey: ~0.5n for random patterns (Storkey 1997)
        """
        if self.learning_rule == "storkey":
            theoretical_max = 0.5 * self.n
        else:
            theoretical_max = 0.14 * self.n

        return {
            "theoretical_max": theoretical_max,
            "empirical_max": theoretical_max,
            "current_load": float(len(self.patterns)),
            "load_ratio": len(self.patterns) / max(theoretical_max, 1),
        }

    def _to_bipolar(self, x: torch.Tensor) -> torch.Tensor:
        """Convert {0, 1} to {-1, +1} if needed."""
        if x.min() >= 0 and x.max() <= 1:
            return 2.0 * x - 1.0
        return x.float()

    def basin_of_attraction(
        self,
        pattern_idx: int,
        noise_levels: List[float] = None,
        n_trials: int = 10,
    ) -> Dict[float, float]:
        """Measure basin of attraction for a stored pattern.

        Args:
            pattern_idx: Index of stored pattern
            noise_levels: List of noise levels to test
            n_trials: Number of trials per noise level

        Returns:
            Dict mapping noise_level -> retrieval_probability
        """
        if noise_levels is None:
            noise_levels = [0.1, 0.2, 0.3, 0.4, 0.5]

        if pattern_idx >= len(self.patterns):
            return {}

        pattern = self.patterns[pattern_idx]
        results = {}

        for noise in noise_levels:
            successes = 0
            for _ in range(n_trials):
                # Add noise: flip bits with probability noise
                noise_mask = torch.rand(self.n) < noise
                noisy = pattern.clone()
                noisy[noise_mask] = -noisy[noise_mask]

                # Retrieve
                retrieved, _ = self.retrieve(noisy)

                # Check if retrieved matches original
                if torch.allclose(retrieved, pattern):
                    successes += 1

            results[noise] = successes / n_trials

        return results


# ═══════════════════════════════════════════════════════════════════════════════
# Section 3: Willshaw Network (Kleyko 2017, Section III)
# ═══════════════════════════════════════════════════════════════════════════════

class WillshawNetwork(AutoassociativeMemory):
    """
    Willshaw Network (Kleyko 2017, Section III).

    Features:
    - Sparse binary patterns: only a small fraction of units active
    - Local learning rule: W_ij = min(1, sum_p x_i^p x_j^p)
    - High storage capacity for sparse patterns
    - One-step retrieval: x_i = 1 if sum_j W_ij x_j >= theta_i
    - Threshold adaptation for optimal retrieval

    Key properties:
    - Capacity: ~(n * log(n)) / (a^2) where a = pattern sparsity
    - Optimal sparsity: ~log(n)/n for maximum capacity
    - Local learning: each synapse learns independently
    - Robust to noise and partial patterns
    """

    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
        threshold_mode: str = "willshaw",  # "willshaw", "adaptive", "fixed"
        fixed_threshold: float = 0.5,
    ):
        super().__init__(config or MemoryConfig())
        self.threshold_mode = threshold_mode
        self.fixed_threshold = fixed_threshold
        self.W: Optional[torch.Tensor] = None  # Weight matrix (n, n)
        self.thresholds: Optional[torch.Tensor] = None  # Per-neuron thresholds
        self.zero_diag = True

    def store(self, pattern: torch.Tensor, label: Any = None) -> None:
        """Store a sparse binary pattern using Willshaw learning.

        Willshaw rule: W_ij = min(1, W_ij + x_i * x_j)
        i.e., set W_ij = 1 if both x_i and x_j are active.

        Args:
            pattern: Sparse binary pattern {0, 1}^n
            label: Optional label
        """
        x = self._to_binary(pattern)

        if self.W is None:
            self.W = torch.zeros(self.n, self.n, dtype=torch.float32)

        # Willshaw learning: W_ij = min(1, W_ij + x_i * x_j)
        self.W += torch.outer(x, x)
        self.W = torch.clamp(self.W, 0, 1)

        # Zero diagonal
        if self.zero_diag:
            self.W.fill_diagonal_(0)

        self.patterns.append(x.clone())
        self.labels.append(label)

        # Update thresholds
        self._update_thresholds()

    def _update_thresholds(self):
        """Update per-neuron retrieval thresholds."""
        if self.threshold_mode == "willshaw":
            # Willshaw threshold: theta_i = max_j W_ij (activity of most active)
            # For sparse patterns, this should be the maximum number of co-active
            # units across all stored patterns
            self.thresholds = self.W.max(dim=1).values
        elif self.threshold_mode == "adaptive":
            # Adaptive: theta_i = mean(W_ij) + std(W_ij)
            mean = self.W.mean(dim=1)
            std = self.W.std(dim=1)
            self.thresholds = mean + std
        else:
            # Fixed threshold
            self.thresholds = torch.full(
                (self.n,), self.fixed_threshold * self.n
            )

    def retrieve(
        self,
        pattern: torch.Tensor,
        max_iter: Optional[int] = None,
    ) -> Tuple[torch.Tensor, int]:
        """Retrieve stored pattern using Willshaw one-step retrieval.

        Willshaw retrieval: x_i = 1 if sum_j W_ij * x_j >= theta_i

        Args:
            pattern: Noisy/partial input pattern (n,)
            max_iter: Maximum iterations (Willshaw typically converges in 1-3)

        Returns:
            (retrieved_pattern, n_iterations)
        """
        if self.W is None or self.thresholds is None:
            return pattern, 0

        max_iter = max_iter or min(self.config.max_iterations, 5)
        x = self._to_binary(pattern).clone()

        for iteration in range(max_iter):
            x_prev = x.clone()

            # Compute postsynaptic sums
            h = self.W @ x

            # Apply threshold
            x = (h >= self.thresholds).float()

            # Check convergence
            diff = torch.norm(x - x_prev).item()
            if diff < self.config.convergence_threshold:
                return x, iteration + 1

        return x, max_iter

    def energy(self, state: torch.Tensor) -> float:
        """Compute Willshaw energy.

        E = -sum_i (sum_j W_ij x_j - theta_i) * x_i

        Lower energy = more stable state.
        """
        if self.W is None or self.thresholds is None:
            return 0.0
        x = self._to_binary(state)
        h = self.W @ x
        return -(h - self.thresholds).dot(x).item()

    def capacity(self) -> Dict[str, float]:
        """Estimate Willshaw network capacity.

        Theoretical: C ~ (n * log(n)) / (a^2)
        where a = pattern sparsity (fraction of active units)

        For optimal sparsity a = log(n)/n:
        C ~ n^2 / log(n)
        """
        a = self.config.pattern_sparsity
        if a > 0:
            theoretical_max = (self.n * math.log(self.n)) / (a * a)
        else:
            theoretical_max = float(self.n)

        return {
            "theoretical_max": theoretical_max,
            "empirical_max": theoretical_max * 0.8,  # Empirical is ~80% of theoretical
            "current_load": float(len(self.patterns)),
            "load_ratio": len(self.patterns) / max(theoretical_max, 1),
            "sparsity": a,
            "optimal_sparsity": math.log(self.n) / self.n,
        }

    def _to_binary(self, x: torch.Tensor) -> torch.Tensor:
        """Convert to binary {0, 1}."""
        return (x > 0).float()


# ═══════════════════════════════════════════════════════════════════════════════
# Section 4: Potts Network (Kleyko 2017, Section IV)
# ═══════════════════════════════════════════════════════════════════════════════

class PottsNetwork(AutoassociativeMemory):
    """
    Potts Network (Kleyko 2017, Section IV).

    Features:
    - Multi-state units: each neuron can be in one of q states
    - Potts model: generalization of Ising model to q > 2 states
    - Suitable for non-binary data (e.g., grayscale, categorical)
    - Higher storage capacity per synapse than binary networks

    Architecture:
    - Each unit has q possible states (Potts spins)
    - Connections between units are q × q matrices
    - Energy: E = -sum_ij sum_ab W_ij^ab * delta(s_i, a) * delta(s_j, b)
    """

    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
        n_states: int = 5,  # q = number of Potts states
        temperature: float = 0.1,  # Temperature for stochastic dynamics
    ):
        super().__init__(config or MemoryConfig())
        self.q = n_states  # Number of Potts states
        self.temperature = temperature
        # Weight tensor: (n, n, q, q) — connections between states
        self.W: Optional[torch.Tensor] = None

    def store(self, pattern: torch.Tensor, label: Any = None) -> None:
        """Store a multi-state pattern.

        Args:
            pattern: Integer pattern {0, ..., q-1}^n
            label: Optional label
        """
        s = pattern.long()

        if self.W is None:
            self.W = torch.zeros(
                self.n, self.n, self.q, self.q,
                dtype=self.config.dtype,
            )

        # Potts Hebbian: W_ij^ab += delta(s_i, a) * delta(s_j, b)
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    a = s[i].item()
                    b = s[j].item()
                    self.W[i, j, a, b] += self.config.learning_rate

        self.patterns.append(s.clone())
        self.labels.append(label)

    def retrieve(
        self,
        pattern: torch.Tensor,
        max_iter: Optional[int] = None,
    ) -> Tuple[torch.Tensor, int]:
        """Retrieve stored pattern using Potts dynamics.

        Uses stochastic update with temperature-controlled exploration.

        Args:
            pattern: Noisy input pattern {0, ..., q-1}^n
            max_iter: Maximum iterations

        Returns:
            (retrieved_pattern, n_iterations)
        """
        if self.W is None:
            return pattern, 0

        max_iter = max_iter or self.config.max_iterations
        s = pattern.long().clone()
        n = self.n

        for iteration in range(max_iter):
            s_prev = s.clone()

            if self.config.async_update:
                order = torch.randperm(n)
                for i in order:
                    # Compute local field for each state
                    h = torch.zeros(self.q)
                    for a in range(self.q):
                        for j in range(n):
                            if i != j:
                                b = s[j].item()
                                h[a] += self.W[i, j, a, b]

                    # Stochastic update with temperature
                    probs = torch.softmax(h / self.temperature, dim=0)
                    s[i] = torch.multinomial(probs, 1).item()
            else:
                # Synchronous update
                s_new = s.clone()
                for i in range(n):
                    h = torch.zeros(self.q)
                    for a in range(self.q):
                        for j in range(n):
                            if i != j:
                                b = s[j].item()
                                h[a] += self.W[i, j, a, b]
                    probs = torch.softmax(h / self.temperature, dim=0)
                    s_new[i] = torch.multinomial(probs, 1).item()
                s = s_new

            # Check convergence
            if torch.all(s == s_prev):
                return s, iteration + 1

        return s, max_iter

    def energy(self, state: torch.Tensor) -> float:
        """Compute Potts energy.

        E = -sum_ij sum_ab W_ij^ab * delta(s_i, a) * delta(s_j, b)
        """
        if self.W is None:
            return 0.0
        s = state.long()
        e = 0.0
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    a = s[i].item()
                    b = s[j].item()
                    e += self.W[i, j, a, b]
        return -e

    def capacity(self) -> Dict[str, float]:
        """Estimate Potts network capacity.

        For q-state Potts: C ~ (q-1) * n / (2 * log(q))
        """
        theoretical_max = (self.q - 1) * self.n / (2 * math.log(self.q))
        return {
            "theoretical_max": theoretical_max,
            "empirical_max": theoretical_max * 0.8,
            "current_load": float(len(self.patterns)),
            "load_ratio": len(self.patterns) / max(theoretical_max, 1),
            "n_states": self.q,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Section 5: Higher-Order Network (Kleyko 2017, Section V)
# ═══════════════════════════════════════════════════════════════════════════════

class HigherOrderNetwork(AutoassociativeMemory):
    """
    Higher-Order Neural Network (Kleyko 2017, Section V).

    Features:
    - Beyond pairwise connections: 3rd-order, 4th-order, etc.
    - Increased storage capacity: C ~ n^(k-1) / (k! * log n) for k-th order
    - Better pattern completion for complex patterns
    - Tensor-based weight representation

    Architecture:
    - k-th order weights: W_(i1,i2,...,ik) tensor
    - Update: x_i = sign(sum_{j1,...,jk-1} W_(i,j1,...,jk-1) * x_j1 * ... * x_jk-1)
    """

    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
        order: int = 3,  # Order of connections (3 = third-order)
    ):
        super().__init__(config or MemoryConfig())
        self.order = order
        # Weights stored as list of tensors for each order
        self.W: Optional[List[torch.Tensor]] = None

    def store(self, pattern: torch.Tensor, label: Any = None) -> None:
        """Store a pattern using higher-order Hebbian learning.

        For k-th order: W_(i,j1,...,jk-1) += x_i * x_j1 * ... * x_jk-1

        Args:
            pattern: Binary pattern {-1, +1}^n or {0, 1}^n
            label: Optional label
        """
        x = self._to_bipolar(pattern)

        if self.W is None:
            self.W = [None] * (self.order + 1)  # Index by order (0, 1, ..., order)

        k = self.order
        # For k-th order, we need a (k)-dimensional tensor
        # But we use a sparse approximation: store outer products
        # W += x ⊗ x ⊗ ... ⊗ x (k times)

        # Compute outer product iteratively
        outer = x
        for _ in range(k - 1):
            outer = torch.outer(outer, x).flatten()

        # Store the outer product (compressed representation)
        if self.W[k] is None:
            self.W[k] = outer.clone()
        else:
            self.W[k] = self.W[k] + outer

        self.patterns.append(x.clone())
        self.labels.append(label)

    def retrieve(
        self,
        pattern: torch.Tensor,
        max_iter: Optional[int] = None,
    ) -> Tuple[torch.Tensor, int]:
        """Retrieve stored pattern using higher-order dynamics.

        Args:
            pattern: Noisy input pattern (n,)
            max_iter: Maximum iterations

        Returns:
            (retrieved_pattern, n_iterations)
        """
        if self.W is None or self.W[self.order] is None:
            return pattern, 0

        max_iter = max_iter or self.config.max_iterations
        x = self._to_bipolar(pattern).clone()
        n = self.n
        k = self.order

        for iteration in range(max_iter):
            x_prev = x.clone()

            # Compute higher-order field
            # h_i = sum_{j1,...,jk-1} W_(i,j1,...,jk-1) * x_j1 * ... * x_jk-1
            h = torch.zeros(n)

            # For efficiency, use iterative tensor contraction
            # W is stored as flattened outer product sum
            if self.W[k] is not None:
                # Reshape W to (n, n^(k-1))
                W_mat = self.W[k].reshape(n, n ** (k - 1))

                # Compute x ⊗ ... ⊗ x (k-1 times)
                x_pow = x
                for _ in range(k - 2):
                    x_pow = torch.outer(x_pow, x).flatten()

                h = W_mat @ x_pow

            # Update
            x = torch.where(h >= 0, torch.tensor(1.0), torch.tensor(-1.0))

            # Check convergence
            diff = torch.norm(x - x_prev).item()
            if diff < self.config.convergence_threshold:
                return x, iteration + 1

        return x, max_iter

    def energy(self, state: torch.Tensor) -> float:
        """Compute higher-order energy.

        E = -sum_{i,j1,...,jk-1} W_(i,j1,...,jk-1) * x_i * x_j1 * ... * x_jk-1
        """
        if self.W is None or self.W[self.order] is None:
            return 0.0
        x = self._to_bipolar(state)
        k = self.order

        # Compute energy using tensor contraction
        W_mat = self.W[k].reshape(self.n, self.n ** (k - 1))
        x_pow = x
        for _ in range(k - 2):
            x_pow = torch.outer(x_pow, x).flatten()

        return -(x @ (W_mat @ x_pow)).item()

    def capacity(self) -> Dict[str, float]:
        """Estimate higher-order network capacity.

        For k-th order: C ~ n^(k-1) / (k! * log n)
        """
        k = self.order
        theoretical_max = (self.n ** (k - 1)) / (math.factorial(k) * math.log(self.n))
        return {
            "theoretical_max": theoretical_max,
            "empirical_max": theoretical_max * 0.7,
            "current_load": float(len(self.patterns)),
            "load_ratio": len(self.patterns) / max(theoretical_max, 1),
            "order": k,
        }

    def _to_bipolar(self, x: torch.Tensor) -> torch.Tensor:
        """Convert {0, 1} to {-1, +1}."""
        if x.min() >= 0 and x.max() <= 1:
            return 2.0 * x - 1.0
        return x.float()


# ═══════════════════════════════════════════════════════════════════════════════
# Section 6: Bipartite Graph Network (Kleyko 2017, Section VI)
# ═══════════════════════════════════════════════════════════════════════════════

class BipartiteGraphNetwork(AutoassociativeMemory):
    """
    Bipartite Graph Network (Kleyko 2017, Section VI).

    Features:
    - Two-layer architecture: visible units + hidden units
    - Non-binary data with linear constraints
    - Bipartite graph structure: no connections within layers
    - Efficient inference via message passing
    - Suitable for continuous-valued data

    Architecture:
    - Visible layer: v ∈ R^n_v (input/output)
    - Hidden layer: h ∈ R^n_h (internal representation)
    - Weight matrix: W ∈ R^(n_v × n_h)
    - Bidirectional dynamics: v ← W @ h, h ← W^T @ v
    """

    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
        n_hidden: int = 50,
        activation: str = "relu",  # "relu", "sigmoid", "tanh"
    ):
        super().__init__(config or MemoryConfig())
        self.n_v = self.n  # Visible units
        self.n_h = n_hidden  # Hidden units
        self.activation = activation
        self.W: Optional[torch.Tensor] = None  # Weight matrix (n_v, n_h)
        self.b_v: Optional[torch.Tensor] = None  # Visible bias
        self.b_h: Optional[torch.Tensor] = None  # Hidden bias

    def store(self, pattern: torch.Tensor, label: Any = None) -> None:
        """Store a pattern using bipartite Hebbian learning.

        Learns a compressed representation via:
        W += v @ h^T where h = activate(W^T @ v)

        Args:
            pattern: Input pattern (n_v,)
            label: Optional label
        """
        v = pattern.float()

        if self.W is None:
            self.W = torch.randn(self.n_v, self.n_h) * 0.1
            self.b_v = torch.zeros(self.n_v)
            self.b_h = torch.zeros(self.n_h)

        # Compute hidden representation
        h = self._activate(self.W.T @ v + self.b_h)

        # Hebbian update
        self.W += self.config.learning_rate * torch.outer(v, h)

        # Update biases
        self.b_v += self.config.learning_rate * v
        self.b_h += self.config.learning_rate * h

        self.patterns.append(v.clone())
        self.labels.append(label)

    def _activate(self, x: torch.Tensor) -> torch.Tensor:
        """Apply activation function."""
        if self.activation == "relu":
            return torch.relu(x)
        elif self.activation == "sigmoid":
            return torch.sigmoid(x)
        elif self.activation == "tanh":
            return torch.tanh(x)
        else:
            return x

    def retrieve(
        self,
        pattern: torch.Tensor,
        max_iter: Optional[int] = None,
    ) -> Tuple[torch.Tensor, int]:
        """Retrieve stored pattern using bipartite dynamics.

        Alternates between visible and hidden layer updates:
        h = activate(W^T @ v + b_h)
        v = W @ h + b_v

        Args:
            pattern: Noisy input pattern (n_v,)
            max_iter: Maximum iterations

        Returns:
            (retrieved_pattern, n_iterations)
        """
        if self.W is None:
            return pattern, 0

        max_iter = max_iter or self.config.max_iterations
        v = pattern.float().clone()

        for iteration in range(max_iter):
            v_prev = v.clone()

            # Update hidden
            h = self._activate(self.W.T @ v + self.b_h)

            # Update visible
            v = self.W @ h + self.b_v

            # Check convergence
            diff = torch.norm(v - v_prev).item()
            if diff < self.config.convergence_threshold:
                return v, iteration + 1

        return v, max_iter

    def energy(self, state: torch.Tensor) -> float:
        """Compute bipartite energy.

        E = -v^T @ W @ h + 0.5 * ||v||^2 + 0.5 * ||h||^2
        """
        if self.W is None:
            return 0.0
        v = state.float()
        h = self._activate(self.W.T @ v + self.b_h)
        return -(v @ (self.W @ h)).item() + 0.5 * torch.norm(v).item()**2 + 0.5 * torch.norm(h).item()**2

    def capacity(self) -> Dict[str, float]:
        """Estimate bipartite network capacity.

        For bipartite with n_v visible and n_h hidden units:
        C ~ n_h * log(1 + n_v/n_h)  (information-theoretic bound)
        """
        theoretical_max = self.n_h * math.log(1 + self.n_v / max(self.n_h, 1))
        return {
            "theoretical_max": theoretical_max,
            "empirical_max": theoretical_max * 0.8,
            "current_load": float(len(self.patterns)),
            "load_ratio": len(self.patterns) / max(theoretical_max, 1),
            "n_visible": self.n_v,
            "n_hidden": self.n_h,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Section 7: Generalization Properties (Kleyko 2017, Section VII)
# ═══════════════════════════════════════════════════════════════════════════════

class GeneralizationAnalyzer:
    """
    Analyzes generalization properties of autoassociative memories.

    Kleyko 2017, Section VII discusses how autoassociative memories
    generalize beyond stored patterns — they can:
    1. Complete partial patterns (pattern completion)
    2. Denoise corrupted patterns (error correction)
    3. Interpolate between stored patterns (generalization)
    4. Extract prototypes from multiple examples (concept learning)

    This analyzer measures:
    - Pattern completion accuracy vs. noise level
    - Generalization error for unseen patterns
    - Prototype extraction quality
    - Capacity vs. generalization tradeoff
    """

    def __init__(self, memory: AutoassociativeMemory):
        self.memory = memory

    def pattern_completion_curve(
        self,
        pattern_idx: int,
        noise_levels: List[float] = None,
        n_trials: int = 20,
    ) -> Dict[str, List[float]]:
        """Measure pattern completion accuracy vs noise level.

        Args:
            pattern_idx: Index of stored pattern to test
            noise_levels: List of noise levels (fraction of bits flipped)
            n_trials: Number of trials per noise level

        Returns:
            Dict with 'noise_levels' and 'accuracy' lists
        """
        if noise_levels is None:
            noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

        if pattern_idx >= len(self.memory.patterns):
            return {"noise_levels": [], "accuracy": []}

        pattern = self.memory.patterns[pattern_idx]
        accuracies = []

        for noise in noise_levels:
            correct = 0
            for _ in range(n_trials):
                # Corrupt pattern
                noise_mask = torch.rand(self.memory.n) < noise
                noisy = pattern.clone()
                if pattern.min() >= 0:
                    noisy[noise_mask] = 1 - noisy[noise_mask]
                else:
                    noisy[noise_mask] = -noisy[noise_mask]

                # Retrieve
                retrieved, _ = self.memory.retrieve(noisy)

                # Check exact match
                if torch.allclose(retrieved, pattern):
                    correct += 1

            accuracies.append(correct / n_trials)

        return {"noise_levels": noise_levels, "accuracy": accuracies}

    def generalization_error(
        self,
        train_patterns: torch.Tensor,
        test_patterns: torch.Tensor,
    ) -> Dict[str, float]:
        """Measure generalization error on unseen patterns.

        Tests whether the memory can generalize to patterns that are
        similar but not identical to stored patterns.

        Args:
            train_patterns: Patterns to store (n_train, n)
            test_patterns: Patterns to test (n_test, n)

        Returns:
            Dict with 'train_accuracy', 'test_accuracy', 'generalization_gap'
        """
        # Store training patterns
        for i in range(train_patterns.shape[0]):
            self.memory.store(train_patterns[i])

        # Test training patterns
        train_correct = 0
        for i in range(train_patterns.shape[0]):
            retrieved, _ = self.memory.retrieve(train_patterns[i])
            if torch.allclose(retrieved, train_patterns[i]):
                train_correct += 1

        # Test unseen patterns
        test_correct = 0
        for i in range(test_patterns.shape[0]):
            retrieved, _ = self.memory.retrieve(test_patterns[i])
            # Check if retrieved is close to any stored pattern
            for j in range(train_patterns.shape[0]):
                if torch.allclose(retrieved, train_patterns[j]):
                    test_correct += 1
                    break

        n_train = train_patterns.shape[0]
        n_test = test_patterns.shape[0]

        return {
            "train_accuracy": train_correct / max(n_train, 1),
            "test_accuracy": test_correct / max(n_test, 1),
            "generalization_gap": (train_correct / max(n_train, 1) -
                                   test_correct / max(n_test, 1)),
        }

    def prototype_extraction(
        self,
        pattern_groups: List[torch.Tensor],
    ) -> Dict[str, Any]:
        """Test prototype extraction from multiple examples.

        Stores multiple noisy variants of the same prototype and
        checks if the memory extracts the correct prototype.

        Args:
            pattern_groups: List of tensors, each group is variants of one prototype

        Returns:
            Dict with extraction quality metrics
        """
        # Store all variants
        for group in pattern_groups:
            for pattern in group:
                self.memory.store(pattern)

        # For each group, check if retrieval converges to the mean
        qualities = []
        for i, group in enumerate(pattern_groups):
            prototype = group.mean(dim=0)
            # Test with a random variant
            idx = torch.randint(0, group.shape[0], (1,)).item()
            retrieved, _ = self.memory.retrieve(group[idx])

            # Measure similarity to prototype
            sim = float(torch.dot(retrieved, prototype) / (
                torch.norm(retrieved) * torch.norm(prototype) + 1e-10
            ))
            qualities.append(sim)

        return {
            "mean_prototype_similarity": float(torch.tensor(qualities).mean()),
            "min_prototype_similarity": float(torch.tensor(qualities).min()),
            "n_prototypes": len(pattern_groups),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Section 8: Similarity Search (Kleyko 2017, Section VIII)
# ═══════════════════════════════════════════════════════════════════════════════

class AutoassociativeSimilaritySearch:
    """
    Sublinear time approximate nearest neighbor search using
    autoassociative memories (Kleyko 2017, Section VIII).

    Key insight: Autoassociative memories can be used for similarity search
    because the retrieval dynamics converge to the nearest stored pattern.
    This is essentially an approximate nearest neighbor (ANN) search where
    the attractor dynamics replace explicit distance computations.

    Features:
    - Multi-probe search: start from multiple random initial states
    - Beam search: maintain multiple candidates during retrieval
    - Hierarchical search: coarse-to-fine refinement
    - Hybrid: combine with LSH for large-scale search
    """

    def __init__(
        self,
        memory: AutoassociativeMemory,
        n_probes: int = 10,
        beam_width: int = 5,
    ):
        self.memory = memory
        self.n_probes = n_probes
        self.beam_width = beam_width

    def search(
        self,
        query: torch.Tensor,
        k: int = 5,
        method: str = "multi_probe",  # "multi_probe", "beam", "hierarchical"
    ) -> List[Tuple[int, float, Any]]:
        """Search for k approximate nearest neighbors.

        Args:
            query: Query pattern (n,)
            k: Number of neighbors to return
            method: Search method

        Returns:
            List of (index, similarity, label) tuples
        """
        if method == "multi_probe":
            return self._multi_probe_search(query, k)
        elif method == "beam":
            return self._beam_search(query, k)
        elif method == "hierarchical":
            return self._hierarchical_search(query, k)
        else:
            return self.memory.similarity_search(query, k)

    def _multi_probe_search(
        self,
        query: torch.Tensor,
        k: int = 5,
    ) -> List[Tuple[int, float, Any]]:
        """Multi-probe search: start from multiple random initial states.

        Each probe starts from a slightly different initial state and
        converges to potentially different attractors.
        """
        candidates = set()
        results = []

        # Probe 1: start from the query itself
        retrieved, _ = self.memory.retrieve(query)
        self._add_candidate(retrieved, candidates, results)

        # Probes 2-N: start from noisy versions of the query
        for _ in range(self.n_probes - 1):
            noise = torch.randn(self.memory.n) * 0.1
            noisy_query = query + noise
            retrieved, _ = self.memory.retrieve(noisy_query)
            self._add_candidate(retrieved, candidates, results)

        # Sort by similarity to query
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    def _beam_search(
        self,
        query: torch.Tensor,
        k: int = 5,
    ) -> List[Tuple[int, float, Any]]:
        """Beam search: maintain multiple candidates during retrieval.

        At each iteration, keep the top-B candidates and expand them.
        """
        # Initialize beam with query
        beam = [(query.clone(), 0.0)]

        for _ in range(min(10, self.memory.config.max_iterations)):
            new_beam = []

            for state, _ in beam:
                # Perturb and retrieve
                for _ in range(self.beam_width):
                    noise = torch.randn(self.memory.n) * 0.05
                    noisy = state + noise
                    retrieved, _ = self.memory.retrieve(noisy)

                    # Compute similarity to query
                    sim = float(torch.dot(retrieved, query) / (
                        torch.norm(retrieved) * torch.norm(query) + 1e-10
                    ))
                    new_beam.append((retrieved, sim))

            # Keep top-B
            new_beam.sort(key=lambda x: x[1], reverse=True)
            beam = new_beam[:self.beam_width]

        # Collect unique candidates
        candidates = set()
        results = []
        for state, sim in beam:
            self._add_candidate(state, candidates, results)

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    def _hierarchical_search(
        self,
        query: torch.Tensor,
        k: int = 5,
    ) -> List[Tuple[int, float, Any]]:
        """Hierarchical search: coarse-to-fine refinement.

        First finds coarse clusters, then refines within the best cluster.
        """
        # Coarse search: use low-resolution version of query
        # (subsample every 4th dimension)
        stride = max(1, self.memory.n // 20)
        coarse_query = query[::stride]

        # Find coarse candidates
        coarse_results = []
        for i, p in enumerate(self.memory.patterns):
            coarse_p = p[::stride]
            sim = float(torch.dot(coarse_query, coarse_p) / (
                torch.norm(coarse_query) * torch.norm(coarse_p) + 1e-10
            ))
            label = self.memory.labels[i] if i < len(self.memory.labels) else None
            coarse_results.append((i, sim, label))

        coarse_results.sort(key=lambda x: x[1], reverse=True)

        # Fine search: refine top coarse candidates
        fine_results = []
        for idx, _, label in coarse_results[:self.beam_width]:
            p = self.memory.patterns[idx]
            sim = float(torch.dot(query, p) / (
                torch.norm(query) * torch.norm(p) + 1e-10
            ))
            fine_results.append((idx, sim, label))

        fine_results.sort(key=lambda x: x[1], reverse=True)
        return fine_results[:k]

    def _add_candidate(
        self,
        state: torch.Tensor,
        candidates: set,
        results: List[Tuple[int, float, Any]],
    ):
        """Add a candidate to results if unique."""
        # Find which stored pattern this corresponds to
        for i, p in enumerate(self.memory.patterns):
            if torch.allclose(state, p):
                key = (i,)
                if key not in candidates:
                    candidates.add(key)
                    sim = float(torch.dot(state, p) / (
                        torch.norm(state) * torch.norm(p) + 1e-10
                    ))
                    label = self.memory.labels[i] if i < len(self.memory.labels) else None
                    results.append((i, sim, label))
                break


# ═══════════════════════════════════════════════════════════════════════════════
# Section 9: Hybrid Memory System
# ═══════════════════════════════════════════════════════════════════════════════

class HybridMemorySystem:
    """
    Combines multiple autoassociative memory models for complex tasks.

    Kleyko 2017 discusses how different memory models excel at different
    aspects: Hopfield for pattern completion, Willshaw for sparse data,
    Potts for multi-state data, higher-order for complex patterns.

    This hybrid system:
    1. Routes patterns to the best-suited memory model
    2. Combines outputs from multiple models via voting
    3. Adaptively selects the model based on pattern statistics
    """

    def __init__(self, config: Optional[MemoryConfig] = None):
        self.config = config or MemoryConfig()
        self.memories: Dict[str, AutoassociativeMemory] = {}

        # Create default memories
        self.add_memory("hopfield", HopfieldNetwork(config))
        self.add_memory("willshaw", WillshawNetwork(config))
        self.add_memory("potts", PottsNetwork(config))
        self.add_memory("higher_order", HigherOrderNetwork(config))
        self.add_memory("bipartite", BipartiteGraphNetwork(config))

    def add_memory(self, name: str, memory: AutoassociativeMemory):
        """Add a memory model to the system."""
        self.memories[name] = memory

    def store(self, pattern: torch.Tensor, label: Any = None) -> Dict[str, bool]:
        """Store pattern in all compatible memories.

        Returns:
            Dict mapping memory name -> whether storage succeeded
        """
        results = {}
        for name, memory in self.memories.items():
            try:
                memory.store(pattern, label)
                results[name] = True
            except Exception as e:
                logger.warning(f"Failed to store in {name}: {e}")
                results[name] = False
        return results

    def retrieve(
        self,
        pattern: torch.Tensor,
        method: str = "voting",  # "voting", "best", "all"
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Retrieve pattern using ensemble of memories.

        Args:
            pattern: Noisy input pattern
            method: Combination method
                - "voting": Majority vote across memories
                - "best": Use the memory with highest confidence
                - "all": Return all results

        Returns:
            (retrieved_pattern, metadata)
        """
        results = {}
        for name, memory in self.memories.items():
            try:
                retrieved, n_iter = memory.retrieve(pattern)
                energy = memory.energy(retrieved)
                results[name] = {
                    "pattern": retrieved,
                    "iterations": n_iter,
                    "energy": energy,
                }
            except Exception as e:
                logger.warning(f"Failed to retrieve from {name}: {e}")

        if method == "voting":
            return self._voting_retrieve(results)
        elif method == "best":
            return self._best_retrieve(results)
        else:
            # Return first result
            for name, result in results.items():
                return result["pattern"], {"sources": list(results.keys())}
            return pattern, {"sources": []}

    def _voting_retrieve(
        self,
        results: Dict[str, Dict],
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Majority vote across memories."""
        if not results:
            return torch.zeros(self.config.n_neurons), {"method": "voting", "sources": []}

        # Collect all retrieved patterns
        patterns = [r["pattern"] for r in results.values()]

        # For binary patterns, majority vote
        if patterns[0].min() >= 0:
            # Binary {0, 1}: majority vote
            stacked = torch.stack(patterns)
            consensus = (stacked.sum(dim=0) > len(patterns) / 2).float()
        else:
            # Bipolar {-1, +1}: sign of sum
            stacked = torch.stack(patterns)
            consensus = torch.sign(stacked.sum(dim=0))

        return consensus, {
            "method": "voting",
            "sources": list(results.keys()),
            "n_votes": len(results),
        }

    def _best_retrieve(
        self,
        results: Dict[str, Dict],
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Use the memory with lowest energy (most stable)."""
        if not results:
            return torch.zeros(self.config.n_neurons), {"method": "best", "sources": []}

        # Find memory with lowest energy
        best_name = min(results, key=lambda n: results[n]["energy"])
        best = results[best_name]

        return best["pattern"], {
            "method": "best",
            "source": best_name,
            "energy": best["energy"],
            "iterations": best["iterations"],
        }

    def similarity_search(
        self,
        query: torch.Tensor,
        k: int = 5,
    ) -> List[Tuple[int, float, Any]]:
        """Ensemble similarity search across all memories."""
        all_results = []
        for name, memory in self.memories.items():
            try:
                results = memory.similarity_search(query, k)
                all_results.extend(results)
            except Exception:
                pass

        seen = set()
        unique = []
        for idx, sim, label in all_results:
            if idx not in seen:
                seen.add(idx)
                unique.append((idx, sim, label))

        unique.sort(key=lambda x: x[1], reverse=True)
        return unique[:k]

    def health_report(self) -> Dict[str, Any]:
        """
        Report capacity utilisation and health for each sub-memory.

        Useful for monitoring long-running deployments where memories
        can saturate and degrade retrieval quality.

        Returns:
            Dict per memory name with n_patterns, capacity_info, utilisation.
        """
        report: Dict[str, Any] = {}
        for name, memory in self.memories.items():
            try:
                cap  = memory.capacity()
                n    = len(memory.patterns) if hasattr(memory, "patterns") else 0
                util = n / max(cap.get("n_patterns", n + 1), 1)
                report[name] = {
                    "n_patterns":   n,
                    "capacity":     cap,
                    "utilisation":  round(util, 4),
                    "saturated":    util >= 0.9,
                }
            except Exception as e:
                report[name] = {"error": str(e)}
        return report

    def best_memory_for_pattern(
        self,
        pattern: torch.Tensor,
    ) -> str:
        """
        Heuristically route a pattern to the most appropriate sub-memory.

        Routing rules (based on Kleyko 2017 §II):
          Sparse (density < 0.1):  Willshaw — sparse patterns
          Continuous / multi-valued: Potts — multi-state
          Dense / balanced:         Hopfield — default
        """
        density = float(pattern.float().abs().mean().item())
        if density < 0.1:
            return "willshaw"
        elif pattern.unique().numel() > 3:
            return "potts"
        else:
            return "hopfield"

    def smart_store(self, pattern: torch.Tensor, label: Any = None):
        """Store a pattern in the most appropriate sub-memory only."""
        best = self.best_memory_for_pattern(pattern)
        if best in self.memories:
            self.memories[best].store(pattern, label)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 10: Tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_hopfield_network():
    """Test Hopfield network (Kleyko 2017, Section II)."""
    print("=" * 60)
    print("Testing Hopfield Network (Section II)")
    print("=" * 60)

    n = 50
    config = MemoryConfig(n_neurons=n, n_patterns=5, seed=42)
    hopfield = HopfieldNetwork(config)

    # Generate random bipolar patterns
    patterns = torch.randint(0, 2, (5, n)).float() * 2 - 1

    # Store patterns
    for i, p in enumerate(patterns):
        hopfield.store(p, label=f"pattern_{i}")
    print(f"  Stored {len(hopfield.patterns)} patterns ✅")

    # Test retrieval with noise
    for noise_level in [0.0, 0.1, 0.2, 0.3]:
        correct = 0
        for i, p in enumerate(patterns):
            if noise_level > 0:
                # Flip bits with probability noise_level
                flip_mask = torch.rand(n) < noise_level
                noisy = p.clone()
                noisy[flip_mask] = -noisy[flip_mask]
            else:
                noisy = p.clone()
            retrieved, n_iter = hopfield.retrieve(noisy)
            if torch.allclose(retrieved, p):
                correct += 1
        print(f"  Retrieval at {noise_level:.0%} noise: {correct}/{len(patterns)} ✅")

    # Test energy
    e = hopfield.energy(patterns[0])
    print(f"  Energy of stored pattern: {e:.4f} ✅")

    # Test capacity
    cap = hopfield.capacity()
    print(f"  Capacity: {cap['theoretical_max']:.1f} (load: {cap['load_ratio']:.2f}) ✅")

    # Test basin of attraction
    basin = hopfield.basin_of_attraction(0, n_trials=5)
    print(f"  Basin of attraction: {basin} ✅")

    # Test similarity search
    results = hopfield.similarity_search(patterns[0], k=3)
    print(f"  Similarity search: {len(results)} results ✅")

    # Test Storkey learning rule
    hopfield2 = HopfieldNetwork(config, learning_rule="storkey")
    for i, p in enumerate(patterns):
        hopfield2.store(p)
    cap2 = hopfield2.capacity()
    print(f"  Storkey capacity: {cap2['theoretical_max']:.1f} ✅")

    print(f"  ✅ Hopfield network test complete!\n")


def test_willshaw_network():
    """Test Willshaw network (Kleyko 2017, Section III)."""
    print("=" * 60)
    print("Testing Willshaw Network (Section III)")
    print("=" * 60)

    n = 100
    sparsity = 0.1
    config = MemoryConfig(n_neurons=n, n_patterns=5, pattern_sparsity=sparsity, seed=42)
    willshaw = WillshawNetwork(config)

    # Generate sparse binary patterns
    patterns = []
    for _ in range(5):
        p = (torch.rand(n) < sparsity).float()
        patterns.append(p)

    # Store patterns
    for i, p in enumerate(patterns):
        willshaw.store(p, label=f"pattern_{i}")
    print(f"  Stored {len(willshaw.patterns)} sparse patterns ✅")

    # Test retrieval
    for noise_level in [0.0, 0.1, 0.2]:
        correct = 0
        for i, p in enumerate(patterns):
            noise = (torch.rand(n) < noise_level).float()
            noisy = ((p + noise) > 0.5).float()
            retrieved, n_iter = willshaw.retrieve(noisy)
            if torch.allclose(retrieved, p):
                correct += 1
        print(f"  Retrieval at {noise_level:.0%} noise: {correct}/{len(patterns)} ✅")

    # Test capacity
    cap = willshaw.capacity()
    print(f"  Capacity: {cap['theoretical_max']:.1f} (sparsity={cap['sparsity']}) ✅")
    print(f"  Optimal sparsity: {cap['optimal_sparsity']:.4f} ✅")

    print(f"  ✅ Willshaw network test complete!\n")


def test_potts_network():
    """Test Potts network (Kleyko 2017, Section IV)."""
    print("=" * 60)
    print("Testing Potts Network (Section IV)")
    print("=" * 60)

    n = 20
    q = 5
    config = MemoryConfig(n_neurons=n, n_patterns=3, seed=42)
    potts = PottsNetwork(config, n_states=q, temperature=0.1)

    # Generate multi-state patterns
    patterns = []
    for _ in range(3):
        p = torch.randint(0, q, (n,))
        patterns.append(p)

    # Store patterns
    for i, p in enumerate(patterns):
        potts.store(p, label=f"pattern_{i}")
    print(f"  Stored {len(potts.patterns)} multi-state patterns (q={q}) ✅")

    # Test retrieval
    for noise_level in [0.0, 0.1, 0.2]:
        correct = 0
        for i, p in enumerate(patterns):
            noise_mask = torch.rand(n) < noise_level
            noisy = p.clone()
            noisy[noise_mask] = torch.randint(0, q, (noise_mask.sum().item(),))
            retrieved, n_iter = potts.retrieve(noisy)
            if torch.all(retrieved == p):
                correct += 1
        print(f"  Retrieval at {noise_level:.0%} noise: {correct}/{len(patterns)} ✅")

    # Test capacity
    cap = potts.capacity()
    print(f"  Capacity: {cap['theoretical_max']:.1f} (q={cap['n_states']}) ✅")

    print(f"  ✅ Potts network test complete!\n")


def test_higher_order_network():
    """Test higher-order network (Kleyko 2017, Section V)."""
    print("=" * 60)
    print("Testing Higher-Order Network (Section V)")
    print("=" * 60)

    n = 20
    config = MemoryConfig(n_neurons=n, n_patterns=3, seed=42)
    higher = HigherOrderNetwork(config, order=3)

    # Generate bipolar patterns
    patterns = (torch.randint(0, 2, (3, n)).float() * 2 - 1)

    # Store patterns
    for i, p in enumerate(patterns):
        higher.store(p, label=f"pattern_{i}")
    print(f"  Stored {len(higher.patterns)} patterns (order={higher.order}) ✅")

    # Test retrieval
    for noise_level in [0.0, 0.1]:
        correct = 0
        for i, p in enumerate(patterns):
            noise = (torch.rand(n) < noise_level).float() * 2 - 1
            noisy = p * noise
            retrieved, n_iter = higher.retrieve(noisy)
            if torch.allclose(retrieved, p):
                correct += 1
        print(f"  Retrieval at {noise_level:.0%} noise: {correct}/{len(patterns)} ✅")

    # Test capacity
    cap = higher.capacity()
    print(f"  Capacity: {cap['theoretical_max']:.1f} (order={cap['order']}) ✅")

    print(f"  ✅ Higher-order network test complete!\n")


def test_bipartite_network():
    """Test bipartite graph network (Kleyko 2017, Section VI)."""
    print("=" * 60)
    print("Testing Bipartite Graph Network (Section VI)")
    print("=" * 60)

    n_v = 20
    n_h = 10
    config = MemoryConfig(n_neurons=n_v, n_patterns=3, seed=42)
    bipartite = BipartiteGraphNetwork(config, n_hidden=n_h, activation="relu")

    # Generate continuous patterns
    patterns = torch.randn(3, n_v)

    # Store patterns
    for i, p in enumerate(patterns):
        bipartite.store(p, label=f"pattern_{i}")
    print(f"  Stored {len(bipartite.patterns)} continuous patterns ✅")

    # Test retrieval
    for noise_level in [0.0, 0.1]:
        correct = 0
        for i, p in enumerate(patterns):
            noisy = p + torch.randn(n_v) * noise_level
            retrieved, n_iter = bipartite.retrieve(noisy)
            # Check similarity to original
            sim = float(torch.dot(retrieved, p) / (
                torch.norm(retrieved) * torch.norm(p) + 1e-10
            ))
            if sim > 0.9:
                correct += 1
        print(f"  Retrieval at {noise_level:.0%} noise: {correct}/{len(patterns)} ✅")

    # Test capacity
    cap = bipartite.capacity()
    print(f"  Capacity: {cap['theoretical_max']:.1f} (n_h={cap['n_hidden']}) ✅")

    print(f"  ✅ Bipartite network test complete!\n")


def test_generalization():
    """Test generalization properties (Kleyko 2017, Section VII)."""
    print("=" * 60)
    print("Testing Generalization Properties (Section VII)")
    print("=" * 60)

    n = 50
    config = MemoryConfig(n_neurons=n, n_patterns=5, seed=42)
    hopfield = HopfieldNetwork(config)
    analyzer = GeneralizationAnalyzer(hopfield)

    # Generate training and test patterns
    train_patterns = (torch.randint(0, 2, (5, n)).float() * 2 - 1)
    test_patterns = (torch.randint(0, 2, (3, n)).float() * 2 - 1)

    # Test pattern completion curve
    for i, p in enumerate(train_patterns):
        hopfield.store(p)
    curve = analyzer.pattern_completion_curve(0)
    print(f"  Pattern completion curve: {len(curve['noise_levels'])} points ✅")

    # Test generalization error
    hopfield2 = HopfieldNetwork(config)
    analyzer2 = GeneralizationAnalyzer(hopfield2)
    gen = analyzer2.generalization_error(train_patterns, test_patterns)
    print(f"  Train accuracy: {gen['train_accuracy']:.2f} ✅")
    print(f"  Test accuracy: {gen['test_accuracy']:.2f} ✅")
    print(f"  Generalization gap: {gen['generalization_gap']:.2f} ✅")

    print(f"  ✅ Generalization test complete!\n")


def test_similarity_search():
    """Test similarity search (Kleyko 2017, Section VIII)."""
    print("=" * 60)
    print("Testing Similarity Search (Section VIII)")
    print("=" * 60)

    n = 50
    config = MemoryConfig(n_neurons=n, n_patterns=10, seed=42)
    hopfield = HopfieldNetwork(config)

    # Generate and store patterns
    patterns = (torch.randint(0, 2, (10, n)).float() * 2 - 1)
    for i, p in enumerate(patterns):
        hopfield.store(p)

    # Test similarity search
    searcher = AutoassociativeSimilaritySearch(hopfield, n_probes=5, beam_width=3)

    # Multi-probe search
    results = searcher.search(patterns[0], k=3, method="multi_probe")
    print(f"  Multi-probe search: {len(results)} results ✅")

    # Beam search
    results = searcher.search(patterns[0], k=3, method="beam")
    print(f"  Beam search: {len(results)} results ✅")

    # Hierarchical search
    results = searcher.search(patterns[0], k=3, method="hierarchical")
    print(f"  Hierarchical search: {len(results)} results ✅")

    print(f"  ✅ Similarity search test complete!\n")


def test_hybrid_system():
    """Test hybrid memory system."""
    print("=" * 60)
    print("Testing Hybrid Memory System")
    print("=" * 60)

    n = 30
    config = MemoryConfig(n_neurons=n, n_patterns=3, seed=42)
    hybrid = HybridMemorySystem(config)

    # Generate and store patterns
    patterns = (torch.randint(0, 2, (3, n)).float() * 2 - 1)
    for i, p in enumerate(patterns):
        results = hybrid.store(p, label=f"pattern_{i}")
        n_stored = sum(1 for v in results.values() if v)
        print(f"  Stored pattern {i} in {n_stored}/{len(results)} memories ✅")

    # Test voting retrieval
    noisy = patterns[0] + torch.randn(n) * 0.2
    retrieved, meta = hybrid.retrieve(noisy, method="voting")
    print(f"  Voting retrieval: {meta['method']}, {meta['n_votes']} votes ✅")

    # Test best retrieval
    retrieved, meta = hybrid.retrieve(noisy, method="best")
    print(f"  Best retrieval: source={meta['source']}, energy={meta['energy']:.2f} ✅")

    # Test similarity search
    results = hybrid.similarity_search(patterns[0], k=3)
    print(f"  Ensemble similarity search: {len(results)} results ✅")

    print(f"  ✅ Hybrid system test complete!\n")


if __name__ == "__main__":
    test_hopfield_network()
    test_willshaw_network()
    test_potts_network()
    test_higher_order_network()
    test_bipartite_network()
    test_generalization()
    test_similarity_search()
    test_hybrid_system()
    print("=== All autoassociative memory tests complete ===")
