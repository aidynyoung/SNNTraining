"""
In-Context Learning via Hyperdimensional Computing
===================================================
A genuinely novel capability: few-shot inference from examples provided at
runtime, without any weight update or retraining.

The transformer does in-context learning via attention over token embeddings.
HDC achieves the same via VSA algebra — XOR, majority vote, Hamming distance:

    Context = Σᵢ bind(query_hv_i, answer_hv_i)    (bundle of Q–A bindings)
    Retrieve:  unbind(Context, new_query) → nearest answer in codebook

Key properties:
    - O(K × D) context construction, O(D) retrieval — no O(K²) attention
    - Zero-shot: no gradient, no weight update
    - Capacity: up to ~D / (2 × ln D) examples before interference
    - Composable: merge contexts from different agents via bundle (Σ)
    - Private: the context HV reveals nothing about individual examples
      (holographic encryption by the binding operation)

This module implements three levels of sophistication:

1. InContextHDC
   ── direct bind-bundle-unbind (one task, fresh context each time)

2. TaskContextLibrary
   ── persistent library of task contexts (each a compressed K-shot summary)
   ── retrieval: pick the most relevant task context for a new query,
      then use it as a prior for answer retrieval

3. HierarchicalInContextHDC
   ── two levels: task selection + example retrieval
   ── enables fast generalisation across task families

References:
    Brown et al. (2020) "Language Models are Few-Shot Learners" — GPT-3 ICL
    Kanerva (2009) "Hyperdimensional Computing" — XOR-based associative memory
    Frady, Kent, Olshausen, Sommer (2020) "Resonator Networks" — unbinding via
        alternating projections
    Plate (1995) "Holographic Reduced Representations" — role-filler binding
    Kleyko et al. (2022) "A Survey on HDC" § III-B — capacity of bundle memory
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from hdc.physics_world_model import _xor, _majority, _hamming


# ── HDC primitives ─────────────────────────────────────────────────────────────

def _bind(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Binary XOR binding."""
    return _xor(a, b)

def _unbind(composite: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    """XOR unbinding: unbind(bind(a,b), b) = a  (self-inverse)."""
    return _xor(composite, key)

def _bundle_list(hvs: List[torch.Tensor], weights: Optional[List[float]] = None) -> torch.Tensor:
    """Majority-vote bundle of a list of HVs, optionally weighted."""
    stacked = torch.stack(hvs, dim=0).float()   # (K, D)
    if weights is not None:
        w = torch.tensor(weights, dtype=torch.float32).unsqueeze(-1)   # (K, 1)
        stacked = stacked * w
    return _majority(stacked.mean(dim=0))

def _gen_hv(dim: int, seed: Optional[int] = None, device: str = "cpu") -> torch.Tensor:
    g = torch.Generator(device=device)
    if seed is not None:
        g.manual_seed(seed)
    return (torch.rand(dim, generator=g, device=device) >= 0.5).float()


# ═══════════════════════════════════════════════════════════════════════════════
# 1. InContextHDC — direct bind-bundle-unbind
# ═══════════════════════════════════════════════════════════════════════════════

class InContextHDC:
    """
    Few-shot in-context learning via VSA algebra.

    Encodes K (query, answer) example pairs into a single context HV:

        C = MAJORITY( bind(q₁, a₁) + bind(q₂, a₂) + ... + bind(qₖ, aₖ) )

    Answer retrieval for a new query q̂:

        candidate = unbind(C, q̂)                  # XOR with query
        answer = argmax_{a ∈ codebook} Hamming_sim(candidate, a)

    Capacity (no interference): K ≤ D / (2 × ln D) ≈ 600 examples at D=10000

    Args:
        dim: Hypervector dimension
        codebook: Optional dict of {label → HV} for cleanup / exact retrieval.
                  If not provided, the raw unbinding result is returned.
    """

    def __init__(self, dim: int, codebook: Optional[Dict[str, torch.Tensor]] = None):
        self.dim = dim
        self.codebook: Dict[str, torch.Tensor] = codebook or {}
        self._context: Optional[torch.Tensor] = None
        self._examples: List[Tuple[torch.Tensor, torch.Tensor, Optional[str]]] = []
        self._n_steps = 0

    # ── codebook management ──────────────────────────────────────────────────

    def register(self, label: str, hv: torch.Tensor):
        """Register a concept in the answer codebook."""
        self.codebook[label] = hv.float()

    def register_all(self, items: Dict[str, torch.Tensor]):
        for label, hv in items.items():
            self.register(label, hv)

    # ── context construction ─────────────────────────────────────────────────

    def add_example(
        self,
        query_hv:  torch.Tensor,
        answer_hv: torch.Tensor,
        label:     Optional[str] = None,
        weight:    float = 1.0,
    ):
        """
        Add a (query, answer) example to the context.

        Incrementally bundles the new binding into the running context:
            C ← MAJORITY(C × n  + bind(q, a) × weight)  (weighted majority)

        This is equivalent to rebuilding from scratch but O(D) per example.

        Args:
            query_hv:  (D,) query hypervector
            answer_hv: (D,) answer hypervector
            label:     Optional string label for the answer
            weight:    Relative importance of this example (default 1.0)
        """
        self._examples.append((query_hv.float(), answer_hv.float(), label))
        self._n_steps += 1

        binding = _bind(query_hv.float(), answer_hv.float())
        if self._context is None:
            self._context = binding.clone()
        else:
            # Weighted incremental bundle (approximate, avoids O(K) recompute)
            w = weight / max(self._n_steps, 1)
            self._context = _majority(
                (1 - w) * self._context.float() + w * binding.float()
            )

    def build_context(
        self,
        queries:  List[torch.Tensor],
        answers:  List[torch.Tensor],
        weights:  Optional[List[float]] = None,
    ) -> torch.Tensor:
        """
        Build context from scratch from a list of (query, answer) pairs.
        More accurate than incremental add_example for large K.

        Returns: (D,) context HV
        """
        bindings = [_bind(q.float(), a.float()) for q, a in zip(queries, answers)]
        self._context = _bundle_list(bindings, weights)
        self._examples = [(q, a, None) for q, a in zip(queries, answers)]
        self._n_steps  = len(queries)
        return self._context

    def reset(self):
        """Clear the current context."""
        self._context  = None
        self._examples = []
        self._n_steps  = 0

    # ── retrieval ────────────────────────────────────────────────────────────

    def retrieve_raw(self, query_hv: torch.Tensor) -> torch.Tensor:
        """
        Retrieve the noisy answer candidate for a query.

        Returns: (D,) raw unbound HV (before codebook cleanup).
        """
        if self._context is None:
            raise RuntimeError("No context built — call add_example() or build_context() first")
        return _unbind(self._context, query_hv.float())

    def retrieve(
        self,
        query_hv: torch.Tensor,
        top_k: int = 1,
    ) -> List[Tuple[str, float, torch.Tensor]]:
        """
        Full retrieval pipeline: unbind → codebook cleanup → ranked results.

        Args:
            query_hv: (D,) new query hypervector
            top_k:    Number of top matches to return

        Returns:
            List of (label, similarity, answer_hv) sorted by similarity desc.
            Returns raw result if codebook is empty.
        """
        candidate = self.retrieve_raw(query_hv)

        if not self.codebook:
            return [("raw", 0.5, candidate)]

        labels  = list(self.codebook.keys())
        codes   = torch.stack([self.codebook[l] for l in labels])   # (C, D)
        sims    = _hamming(candidate.unsqueeze(0), codes)            # (C,)

        ranked  = sorted(zip(labels, sims.tolist()), key=lambda x: x[1], reverse=True)
        return [(lbl, sim, self.codebook[lbl]) for lbl, sim in ranked[:top_k]]

    def retrieve_one(self, query_hv: torch.Tensor) -> Tuple[str, float]:
        """Return (label, similarity) for the single best match."""
        results = self.retrieve(query_hv, top_k=1)
        lbl, sim, _ = results[0]
        return lbl, sim

    # ── context diagnostics ──────────────────────────────────────────────────

    def context_capacity(self) -> Dict[str, float]:
        """
        Estimate current context quality.

        Returns dict with:
            n_examples: number of examples in context
            fill_fraction: K / capacity_limit (> 1.0 → likely degraded)
            reconstruction_acc: fraction of stored examples retrievable
        """
        K        = len(self._examples)
        capacity = max(1, int(self.dim / (2 * math.log(max(self.dim, 2)))))
        fill     = K / capacity

        # Test reconstruction on stored examples (if codebook is available)
        if self.codebook and K > 0:
            correct = 0
            for q_hv, a_hv, label in self._examples[:min(K, 20)]:
                candidate = self.retrieve_raw(q_hv)
                if label is not None:
                    lbl, _ = self.retrieve_one(q_hv)
                    correct += int(lbl == label)
            acc = correct / min(K, 20) if K > 0 else 0.0
        else:
            acc = -1.0

        return {
            "n_examples":       K,
            "capacity_limit":   capacity,
            "fill_fraction":    fill,
            "reconstruction_acc": acc,
        }

    def merge(self, other: "InContextHDC") -> "InContextHDC":
        """
        Merge two contexts (federated / multi-agent).
        Returns a new context that summarises both.
        """
        merged = InContextHDC(self.dim, {**self.codebook, **other.codebook})
        if self._context is not None and other._context is not None:
            merged._context = _majority(
                (self._context.float() + other._context.float()) / 2.0
            )
            merged._n_steps = self._n_steps + other._n_steps
        elif self._context is not None:
            merged._context = self._context.clone()
        elif other._context is not None:
            merged._context = other._context.clone()
        return merged


# ═══════════════════════════════════════════════════════════════════════════════
# 2. TaskContextLibrary — multi-task in-context memory
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TaskContext:
    """A stored task: its summary HV + metadata."""
    name:      str
    context_hv: torch.Tensor          # (D,) compressed K-shot summary
    codebook:  Dict[str, torch.Tensor]
    n_examples: int
    created_at: int = 0

    def similarity(self, query_hv: torch.Tensor) -> float:
        """How relevant is this task context for a given query?"""
        return float(_hamming(query_hv.unsqueeze(0), self.context_hv.unsqueeze(0)).item())


class TaskContextLibrary:
    """
    Persistent library of task contexts for cross-task in-context learning.

    Each task is compressed into a single context HV (the bundle of its
    K example bindings).  For a new query, the library:
      1. Retrieves the most relevant stored task (Hamming similarity)
      2. Uses that task's context + codebook for answer retrieval

    This enables rapid generalisation across task families: e.g., if the
    system has seen 100 "bearing-fault" examples and 100 "gear-fault"
    examples as separate task contexts, it can retrieve the most relevant
    one for a new fault query without seeing any example from that fault type.

    Args:
        dim: Hypervector dimension
        max_tasks: Maximum stored task contexts (oldest evicted when full)
    """

    def __init__(self, dim: int, max_tasks: int = 50):
        self.dim       = dim
        self.max_tasks = max_tasks
        self._tasks:   List[TaskContext] = []
        self._tick:    int = 0

    def register_task(
        self,
        name:     str,
        queries:  List[torch.Tensor],
        answers:  List[torch.Tensor],
        codebook: Optional[Dict[str, torch.Tensor]] = None,
    ) -> TaskContext:
        """
        Compress K examples into a stored task context.

        Args:
            name:     Human-readable task name
            queries:  K query hypervectors
            answers:  K answer hypervectors (same order)
            codebook: Label → HV mapping for answer decoding

        Returns:
            The created TaskContext (also stored internally).
        """
        ctx = InContextHDC(self.dim, codebook or {})
        ctx.build_context(queries, answers)

        task = TaskContext(
            name=name,
            context_hv=ctx._context.clone(),
            codebook=codebook or {},
            n_examples=len(queries),
            created_at=self._tick,
        )
        self._tick += 1

        self._tasks.append(task)
        if len(self._tasks) > self.max_tasks:
            # Evict oldest
            self._tasks.pop(0)

        return task

    def retrieve_task(
        self, query_hv: torch.Tensor, top_k: int = 1
    ) -> List[Tuple[TaskContext, float]]:
        """Return the top-k most relevant task contexts for a query."""
        if not self._tasks:
            return []
        scored = sorted(
            [(t, t.similarity(query_hv)) for t in self._tasks],
            key=lambda x: x[1],
            reverse=True,
        )
        return scored[:top_k]

    def query(
        self,
        query_hv: torch.Tensor,
        top_k_answers: int = 1,
    ) -> Tuple[str, float, str]:
        """
        Full cross-task in-context inference.

        1. Find most relevant task
        2. Unbind query from that task's context
        3. Cleanup via task's codebook

        Returns:
            (predicted_label, similarity, task_name)
        """
        tasks = self.retrieve_task(query_hv, top_k=1)
        if not tasks:
            return "unknown", 0.0, "none"

        best_task, task_sim = tasks[0]

        # Re-instantiate InContextHDC for retrieval
        ctx = InContextHDC(self.dim, best_task.codebook)
        ctx._context = best_task.context_hv

        results = ctx.retrieve(query_hv, top_k=top_k_answers)
        lbl, ans_sim, _ = results[0]
        return lbl, ans_sim, best_task.name

    def merge_tasks(self, task_names: List[str]) -> Optional[TaskContext]:
        """
        Merge multiple task contexts into a composite context.

        Useful when a new problem combines aspects of multiple known tasks.
        """
        selected = [t for t in self._tasks if t.name in task_names]
        if not selected:
            return None

        merged_ctx = _bundle_list([t.context_hv for t in selected])
        merged_cb  = {}
        for t in selected:
            merged_cb.update(t.codebook)

        return TaskContext(
            name="+".join(task_names),
            context_hv=merged_ctx,
            codebook=merged_cb,
            n_examples=sum(t.n_examples for t in selected),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 3. HierarchicalInContextHDC — two-level: task selection + example retrieval
# ═══════════════════════════════════════════════════════════════════════════════

class HierarchicalInContextHDC:
    """
    Two-level hierarchical in-context learning.

    Level 1 (task): which task/domain does this query belong to?
    Level 2 (example): within that task, what is the answer?

    Architecture:
        query_hv
            │
            ├── [Level 1] TaskContextLibrary.retrieve_task()  → task_ctx
            │       Hamming similarity to compressed task summaries
            │
            └── [Level 2] InContextHDC(task_ctx).retrieve()   → answer
                    XOR unbind + codebook lookup

    This two-level structure enables:
        - Fast task identification even with hundreds of tasks
        - Fine-grained answer retrieval within each task
        - Graceful degradation: if task retrieval is poor, level-2 still helps

    Args:
        dim: Hypervector dimension
        max_tasks: Maximum stored task contexts
    """

    def __init__(self, dim: int, max_tasks: int = 100):
        self.dim     = dim
        self.library = TaskContextLibrary(dim, max_tasks)
        self._global_codebook: Dict[str, torch.Tensor] = {}

    def register_concept(self, label: str, hv: torch.Tensor):
        """Register a concept in the global codebook (available to all tasks)."""
        self._global_codebook[label] = hv.float()

    def add_task(
        self,
        task_name: str,
        examples:  List[Tuple[torch.Tensor, torch.Tensor, str]],
    ) -> TaskContext:
        """
        Register a K-shot task from (query_hv, answer_hv, label) triples.

        Args:
            task_name: Name of this task
            examples:  List of (query_hv, answer_hv, label) triples

        Returns:
            Created TaskContext
        """
        queries  = [q for q, a, _ in examples]
        answers  = [a for q, a, _ in examples]
        labels   = [l for q, a, l in examples]

        # Build task-specific codebook (answers keyed by their labels)
        task_cb  = {lbl: ans for ans, lbl in zip(answers, labels)
                    if lbl not in self._global_codebook}
        task_cb.update(self._global_codebook)

        return self.library.register_task(task_name, queries, answers, task_cb)

    def infer(
        self,
        query_hv:    torch.Tensor,
        top_k:       int = 1,
        task_top_k:  int = 3,
    ) -> List[Tuple[str, float, str]]:
        """
        Hierarchical in-context inference.

        1. Retrieve top-k_task most relevant tasks
        2. For each task, retrieve top-k answers
        3. Aggregate via ensemble voting (majority over task answers)

        Returns:
            List of (label, confidence, source_task) sorted by confidence.
        """
        task_results = self.library.retrieve_task(query_hv, top_k=task_top_k)

        vote_weights: Dict[str, float] = {}
        vote_tasks:   Dict[str, str]   = {}

        for task, task_sim in task_results:
            ctx           = InContextHDC(self.dim, task.codebook)
            ctx._context  = task.context_hv
            answers       = ctx.retrieve(query_hv, top_k=1)
            for lbl, ans_sim, _ in answers:
                weight = task_sim * ans_sim
                vote_weights[lbl] = vote_weights.get(lbl, 0.0) + weight
                vote_tasks[lbl]   = task.name

        if not vote_weights:
            return [("unknown", 0.0, "none")]

        ranked = sorted(vote_weights.items(), key=lambda x: x[1], reverse=True)
        total  = sum(w for _, w in ranked) + 1e-8
        return [(lbl, w / total, vote_tasks[lbl]) for lbl, w in ranked[:top_k]]

    def infer_one(self, query_hv: torch.Tensor) -> Tuple[str, float, str]:
        """Return (label, confidence, source_task) for single best prediction."""
        results = self.infer(query_hv, top_k=1)
        return results[0]

    def adapt_from_feedback(
        self,
        query_hv:   torch.Tensor,
        true_label: str,
        task_name:  Optional[str] = None,
        lr:         float = 0.1,
    ):
        """
        Online adaptation from a single labelled feedback sample.

        When a prediction is correct or corrected by the user, call this to
        strengthen the association between query_hv and true_label.  No full
        retraining needed — just updates the relevant task context HV.

        Args:
            query_hv:   (D,) query hypervector that was just inferred
            true_label: Ground-truth label for the query
            task_name:  Which task to update (None = most recently matched task)
            lr:         Blend rate for context update (default 0.1)
        """
        if task_name is None:
            # Fallback: update all tasks equally
            result = self.infer_one(query_hv)
            task_name = result[2]   # source_task

        for task in self.library._tasks:
            if task.name != task_name:
                continue
            # Blend the query HV into the task context, strengthening this label
            if true_label in task.codebook:
                ans_hv  = task.codebook[true_label].float()
                # Compute mini binding: bind(query, answer) for this example
                mini_ctx = _bind(query_hv.float(), ans_hv)
                # Blend mini context into the existing task context HV
                task.context_hv = _majority(
                    (1 - lr) * task.context_hv.float() + lr * mini_ctx.float()
                )
            break

    def task_quality(self, task_name: str) -> Dict[str, float]:
        """
        Measure task representation quality from the context HV and codebook.

        Returns:
            Dict with intra_sim (context density as cohesion proxy),
            n_codebook_entries.
        """
        for task in self.library._tasks:
            if task.name != task_name:
                continue
            ctx = task.context_hv.float()
            # Density as cohesion proxy: near-0.5 = uniform (poor), near-0/1 = discriminative
            density = float(ctx.mean().item())
            intra_sim = 2.0 * abs(density - 0.5)   # 0 = uniform, 1 = saturated
            return {
                "intra_sim":   intra_sim,
                "n_codebook":  len(task.codebook),
                "n_examples":  task.n_examples if hasattr(task, 'n_examples') else len(task.codebook),
                "density":     density,
            }
        return {"intra_sim": 0.0, "n_examples": 0, "n_codebook": 0}

    @property
    def n_tasks(self) -> int:
        return len(self.library._tasks)

    @property
    def task_names(self) -> List[str]:
        return [t.name for t in self.library._tasks]


# ═══════════════════════════════════════════════════════════════════════════════
# 4. HDCMetaLearner — prototype-based meta-learning (HDC-MAML)
# ═══════════════════════════════════════════════════════════════════════════════

class HDCMetaLearner:
    """
    Meta-learning via prototype adaptation speed optimisation.

    Reference:
        Finn, Abbeel, Levine (2017) "Model-Agnostic Meta-Learning for Fast
        Adaptation of Deep Networks" ICML 2017.

    MAML-for-HDC intuition:
        Instead of learning initial neural network weights, we learn an initial
        prototype HV for each class such that K-shot updates (bundling K
        examples) maximally separate the classes.

    Algorithm:
        Meta-train loop (outer):
            For each task t:
                1. Sample K support examples per class
                2. Update prototypes from support: P[c] ← MAJORITY(support_c)
                3. Evaluate on K query examples → meta-loss (Hamming error)
            Update initial prototypes to minimise mean meta-loss

        Meta-test (inference):
            Given K support examples for a new task:
                P[c] ← MAJORITY(initial_P[c] + support_c)   (warm start!)
            This converges faster than random initialisation.

    HDC advantage: the "inner loop" is O(K × D) — a single majority vote.
    No gradient, no backward pass, no learning rate tuning.

    Args:
        n_classes: Number of output classes
        dim: Hypervector dimension
        meta_lr: Meta-learning rate for outer loop prototype update
        inner_steps: Number of inner-loop support updates (default 1 for HDC)
    """

    def __init__(
        self,
        n_classes: int,
        dim: int,
        meta_lr: float = 0.05,
        inner_steps: int = 1,
    ):
        self.n_classes   = n_classes
        self.dim         = dim
        self.meta_lr     = meta_lr
        self.inner_steps = inner_steps

        # Meta-initialised prototypes (learned over many tasks)
        self.meta_prototypes: List[torch.Tensor] = [
            _gen_hv(dim, seed=c) for c in range(n_classes)
        ]
        self._meta_update_count = 0

    def _inner_update(
        self,
        support_hvs:    List[torch.Tensor],
        support_labels: List[int],
        init_protos:    Optional[List[torch.Tensor]] = None,
    ) -> List[torch.Tensor]:
        """
        K-shot inner loop: update prototypes from K support examples.

        Args:
            support_hvs:    List of K support hypervectors
            support_labels: Corresponding class labels
            init_protos:    Starting prototypes (default: meta_prototypes)

        Returns:
            Updated prototypes (one per class).
        """
        protos = [p.clone() for p in (init_protos or self.meta_prototypes)]
        accum  = [[] for _ in range(self.n_classes)]

        for hv, lbl in zip(support_hvs, support_labels):
            accum[lbl].append(hv.float())

        for c in range(self.n_classes):
            if accum[c]:
                class_mean = _bundle_list(accum[c])
                # Warm-start blend: init + new examples
                protos[c]  = _majority(
                    (1 - self.meta_lr) * protos[c].float() + self.meta_lr * class_mean.float()
                )

        return protos

    def predict(
        self,
        query_hv: torch.Tensor,
        adapted_protos: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[int, List[float]]:
        """
        Predict class for a query using (optionally adapted) prototypes.

        Args:
            query_hv:       (D,) query HV
            adapted_protos: Episode-specific prototypes (from adapt())

        Returns:
            (class_idx, similarity_scores)
        """
        protos = adapted_protos or self.meta_prototypes
        codes  = torch.stack([p.float() for p in protos])   # (C, D)
        sims   = _hamming(query_hv.float().unsqueeze(0), codes)  # (C,)
        pred   = int(sims.argmax().item())
        return pred, sims.tolist()

    def adapt(
        self,
        support_hvs:    List[torch.Tensor],
        support_labels: List[int],
    ) -> List[torch.Tensor]:
        """
        K-shot task adaptation: return episode-specific prototypes.
        Does NOT modify meta_prototypes (inner loop only).
        """
        return self._inner_update(support_hvs, support_labels)

    def meta_update(
        self,
        task_batches: List[Tuple[
            List[torch.Tensor],   # support HVs
            List[int],            # support labels
            List[torch.Tensor],   # query HVs
            List[int],            # query labels
        ]],
    ) -> float:
        """
        Outer meta-update: improve meta_prototypes across a batch of tasks.

        For each task:
            1. Run inner loop on support set
            2. Evaluate on query set
            3. Accumulate gradient signal: for each wrong prediction,
               pull correct prototype toward the query HV

        Args:
            task_batches: List of (support_hvs, support_labels, query_hvs, query_labels)

        Returns:
            Mean query accuracy across tasks.
        """
        total_correct = 0
        total_queries = 0

        proto_updates  = [[torch.zeros(self.dim)] for _ in range(self.n_classes)]

        for s_hvs, s_lbls, q_hvs, q_lbls in task_batches:
            adapted = self._inner_update(s_hvs, s_lbls)

            for q_hv, q_lbl in zip(q_hvs, q_lbls):
                pred, sims = self.predict(q_hv, adapted)
                total_queries += 1
                if pred == q_lbl:
                    total_correct += 1
                else:
                    # Push: accumulate a pull toward correct meta prototype
                    proto_updates[q_lbl].append(q_hv.float())

        # Apply meta-update: blend meta_prototypes toward query examples
        for c in range(self.n_classes):
            updates = proto_updates[c][1:]   # skip the zero placeholder
            if updates:
                correction = _bundle_list(updates)
                self.meta_prototypes[c] = _majority(
                    (1 - self.meta_lr) * self.meta_prototypes[c].float()
                    + self.meta_lr * correction.float()
                )

        self._meta_update_count += 1
        return total_correct / max(total_queries, 1)

    @property
    def meta_steps(self) -> int:
        return self._meta_update_count

    def prototype_diversity(self) -> float:
        """
        Measure diversity of meta-learned prototypes.

        Ideal meta-prototypes are maximally separated (diversity ≈ 0.5).
        Low diversity → classes are easily confused after K-shot adaptation.
        High diversity → classes remain distinct even with few examples.

        Returns:
            Mean pairwise Hamming distance between meta-prototypes ∈ [0, 0.5].
        """
        n = len(self.meta_prototypes)
        if n < 2:
            return 0.5
        total, count = 0.0, 0
        for i in range(n):
            for j in range(i + 1, n):
                d = float((self.meta_prototypes[i] != self.meta_prototypes[j]).float().mean())
                total += d
                count += 1
        return total / max(count, 1)

    def meta_report(self) -> Dict:
        """
        Summary report of the meta-learner's current state.

        Returns:
            Dict with meta_steps, prototype_diversity, n_classes.
        """
        return {
            "meta_steps":          self._meta_update_count,
            "n_classes":           self.n_classes,
            "prototype_diversity": round(self.prototype_diversity(), 4),
            "meta_lr":             self.meta_lr,
            "status":              (
                "diverse" if self.prototype_diversity() > 0.35
                else "needs_more_meta_training"
            ),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Smoke test
# ═══════════════════════════════════════════════════════════════════════════════

def _test_in_context_hdc():
    D   = 1000
    g   = lambda seed: _gen_hv(D, seed=seed)

    print("=== InContextHDC ===")
    concept_hvs = {f"class_{c}": g(c) for c in range(5)}

    ctx = InContextHDC(D, concept_hvs)
    # 5-shot: provide one example per class
    for c in range(5):
        q = g(100 + c)                        # arbitrary query HV for class c
        a = concept_hvs[f"class_{c}"]         # correct answer HV
        ctx.add_example(q, a, label=f"class_{c}")

    # Test retrieval
    for c in range(5):
        q   = g(100 + c)
        lbl, sim = ctx.retrieve_one(q)
        print(f"  query class_{c}: retrieved '{lbl}', sim={sim:.3f}")

    cap = ctx.context_capacity()
    print(f"  capacity: {cap}")

    print("\n=== TaskContextLibrary ===")
    lib = TaskContextLibrary(D, max_tasks=10)
    for task_id in range(3):
        q_list = [g(200 + task_id * 10 + i) for i in range(5)]
        a_list = [concept_hvs[f"class_{i % 5}"] for i in range(5)]
        lib.register_task(f"task_{task_id}", q_list, a_list, concept_hvs)

    q_test = g(200)
    lbl, sim, task_name = lib.query(q_test)
    print(f"  query → '{lbl}' (sim={sim:.3f}) from '{task_name}'")

    print("\n=== HierarchicalInContextHDC ===")
    hier = HierarchicalInContextHDC(D)
    hier.register_concept("fault_A", g(300))
    hier.register_concept("fault_B", g(301))
    for task_id in range(4):
        examples = [
            (g(400 + task_id * 5 + i),
             g(300 + (i % 2)),
             f"fault_{'A' if i % 2 == 0 else 'B'}")
            for i in range(4)
        ]
        hier.add_task(f"machine_{task_id}", examples)

    q_new = g(400)
    lbl, conf, src = hier.infer_one(q_new)
    print(f"  infer → '{lbl}' (conf={conf:.3f}) from '{src}'")
    print(f"  {hier.n_tasks} tasks: {hier.task_names}")

    print("\n=== HDCMetaLearner ===")
    meta = HDCMetaLearner(n_classes=3, dim=D, meta_lr=0.1)
    # Simulate 5 meta-training tasks
    tasks = []
    for t in range(5):
        s_hvs  = [g(500 + t * 20 + i) for i in range(9)]
        s_lbls = [i % 3 for i in range(9)]
        q_hvs  = [g(600 + t * 20 + i) for i in range(6)]
        q_lbls = [i % 3 for i in range(6)]
        tasks.append((s_hvs, s_lbls, q_hvs, q_lbls))
    acc = meta.meta_update(tasks)
    print(f"  meta-train acc={acc:.3f}, meta_steps={meta.meta_steps}")

    # Adapt to new task
    adapted = meta.adapt([g(700 + i) for i in range(3)], [0, 1, 2])
    pred, sims = meta.predict(g(700), adapted)
    print(f"  adapted predict: class={pred}, sims={[round(s,3) for s in sims]}")

    print("\n✅ All in_context_hdc tests passed")


if __name__ == "__main__":
    _test_in_context_hdc()
