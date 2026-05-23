"""
tests/test_in_context_hdc.py
=============================
Tests for InContextHDC, TaskContextLibrary,
HierarchicalInContextHDC, and HDCMetaLearner.
"""
import pytest
import torch
from hdc.in_context_hdc import (
    InContextHDC,
    TaskContextLibrary,
    HierarchicalInContextHDC,
    HDCMetaLearner,
    _bind,
    _unbind,
    _bundle_list,
    _gen_hv,
)

D = 512


def _hv(seed):
    return _gen_hv(D, seed=seed)


# ── InContextHDC ──────────────────────────────────────────────────────────────

class TestInContextHDC:
    def setup_method(self):
        self.codebook = {f"class_{c}": _hv(c) for c in range(5)}
        self.ctx = InContextHDC(D, self.codebook)

    def test_add_example_builds_context(self):
        self.ctx.add_example(_hv(10), _hv(0), label="class_0")
        assert self.ctx._context is not None
        assert self.ctx._context.shape == (D,)

    def test_retrieve_raw_shape(self):
        self.ctx.add_example(_hv(10), _hv(0))
        raw = self.ctx.retrieve_raw(_hv(10))
        assert raw.shape == (D,)

    def test_retrieve_with_codebook(self):
        # Register 5-shot examples: each class_i gets query _hv(100+i)
        for c in range(5):
            self.ctx.add_example(_hv(100 + c), self.codebook[f"class_{c}"],
                                 label=f"class_{c}")
        # The exact query for class_0 should retrieve class_0 or nearby
        results = self.ctx.retrieve(_hv(100), top_k=3)
        assert len(results) == 3
        labels = [r[0] for r in results]
        assert "class_0" in labels

    def test_build_context_batch(self):
        queries  = [_hv(200 + c) for c in range(5)]
        answers  = [self.codebook[f"class_{c}"] for c in range(5)]
        ctx_hv   = self.ctx.build_context(queries, answers)
        assert ctx_hv.shape == (D,)

    def test_retrieve_one_returns_tuple(self):
        self.ctx.add_example(_hv(0), self.codebook["class_0"], label="class_0")
        lbl, sim = self.ctx.retrieve_one(_hv(0))
        assert isinstance(lbl, str)
        assert 0.0 <= sim <= 1.0

    def test_context_capacity(self):
        for c in range(5):
            self.ctx.add_example(_hv(c), self.codebook[f"class_{c}"],
                                 label=f"class_{c}")
        cap = self.ctx.context_capacity()
        assert cap["n_examples"] == 5
        assert cap["capacity_limit"] > 0
        assert "fill_fraction" in cap

    def test_reset_clears_context(self):
        self.ctx.add_example(_hv(0), _hv(1))
        self.ctx.reset()
        assert self.ctx._context is None
        assert self.ctx._n_steps == 0

    def test_merge_two_contexts(self):
        ctx_a = InContextHDC(D, self.codebook)
        ctx_b = InContextHDC(D, self.codebook)
        ctx_a.add_example(_hv(0), self.codebook["class_0"])
        ctx_b.add_example(_hv(1), self.codebook["class_1"])
        merged = ctx_a.merge(ctx_b)
        assert merged._context is not None

    def test_bind_unbind_identity(self):
        a, b = _hv(0), _hv(1)
        bound   = _bind(a, b)
        unbound = _unbind(bound, b)
        # XOR self-inverse: bind(bind(a,b),b) == a
        assert torch.equal(unbound, a)

    def test_bundle_list(self):
        hvs = [_hv(i) for i in range(5)]
        bundled = _bundle_list(hvs)
        assert bundled.shape == (D,)
        assert set(bundled.unique().tolist()).issubset({0.0, 1.0})


# ── TaskContextLibrary ────────────────────────────────────────────────────────

class TestTaskContextLibrary:
    def setup_method(self):
        self.codebook = {f"class_{c}": _hv(c) for c in range(5)}
        self.lib = TaskContextLibrary(D, max_tasks=10)

    def test_register_task(self):
        queries = [_hv(100 + i) for i in range(5)]
        answers = [_hv(i) for i in range(5)]
        task = self.lib.register_task("task_0", queries, answers, self.codebook)
        assert task.name == "task_0"
        assert task.n_examples == 5
        assert len(self.lib._tasks) == 1

    def test_retrieve_task(self):
        for t in range(3):
            q = [_hv(t * 10 + i) for i in range(4)]
            a = [_hv(i) for i in range(4)]
            self.lib.register_task(f"task_{t}", q, a, self.codebook)
        results = self.lib.retrieve_task(_hv(0), top_k=2)
        assert len(results) == 2
        assert all(0.0 <= sim <= 1.0 for _, sim in results)

    def test_query_returns_label_and_task(self):
        queries = [_hv(200 + i) for i in range(5)]
        answers = [self.codebook[f"class_{i}"] for i in range(5)]
        self.lib.register_task("my_task", queries, answers, self.codebook)
        lbl, sim, task_name = self.lib.query(_hv(200))
        assert isinstance(lbl, str)
        assert isinstance(task_name, str)

    def test_max_tasks_evicts_oldest(self):
        lib = TaskContextLibrary(D, max_tasks=3)
        for t in range(5):
            lib.register_task(f"task_{t}", [_hv(t)], [_hv(100 + t)])
        assert len(lib._tasks) <= 3

    def test_merge_tasks(self):
        q = [_hv(i) for i in range(3)]
        a = [_hv(100 + i) for i in range(3)]
        self.lib.register_task("t0", q, a)
        self.lib.register_task("t1", q, a)
        merged = self.lib.merge_tasks(["t0", "t1"])
        assert merged is not None
        assert merged.context_hv.shape == (D,)


# ── HierarchicalInContextHDC ──────────────────────────────────────────────────

class TestHierarchicalInContextHDC:
    def setup_method(self):
        self.hier = HierarchicalInContextHDC(D, max_tasks=20)
        # Register two concepts
        self.hier.register_concept("fault_A", _hv(300))
        self.hier.register_concept("fault_B", _hv(301))
        # Add tasks
        for t in range(3):
            examples = [
                (_hv(400 + t * 5 + i), _hv(300 + (i % 2)),
                 f"fault_{'A' if i % 2 == 0 else 'B'}")
                for i in range(4)
            ]
            self.hier.add_task(f"machine_{t}", examples)

    def test_n_tasks(self):
        assert self.hier.n_tasks == 3

    def test_task_names(self):
        assert "machine_0" in self.hier.task_names

    def test_infer_returns_list(self):
        results = self.hier.infer(_hv(400), top_k=2)
        assert len(results) >= 1
        lbl, conf, src = results[0]
        assert isinstance(lbl, str)
        assert 0.0 <= conf <= 1.0

    def test_infer_one(self):
        lbl, conf, src = self.hier.infer_one(_hv(400))
        assert isinstance(lbl, str)
        assert isinstance(src, str)


# ── HDCMetaLearner ────────────────────────────────────────────────────────────

class TestHDCMetaLearner:
    def setup_method(self):
        self.meta = HDCMetaLearner(n_classes=3, dim=D, meta_lr=0.1)

    def test_meta_prototypes_shape(self):
        assert len(self.meta.meta_prototypes) == 3
        for p in self.meta.meta_prototypes:
            assert p.shape == (D,)

    def test_adapt_returns_prototypes(self):
        support_hvs  = [_hv(i) for i in range(9)]
        support_lbls = [i % 3 for i in range(9)]
        adapted = self.meta.adapt(support_hvs, support_lbls)
        assert len(adapted) == 3
        for p in adapted:
            assert p.shape == (D,)

    def test_predict_returns_class_and_sims(self):
        adapted = self.meta.adapt([_hv(i) for i in range(3)], [0, 1, 2])
        pred, sims = self.meta.predict(_hv(0), adapted)
        assert 0 <= pred < 3
        assert len(sims) == 3
        assert all(0.0 <= s <= 1.0 for s in sims)

    def test_meta_update_returns_accuracy(self):
        tasks = []
        for t in range(4):
            s_hvs  = [_hv(t * 20 + i) for i in range(9)]
            s_lbls = [i % 3 for i in range(9)]
            q_hvs  = [_hv(t * 20 + 10 + i) for i in range(6)]
            q_lbls = [i % 3 for i in range(6)]
            tasks.append((s_hvs, s_lbls, q_hvs, q_lbls))
        acc = self.meta.meta_update(tasks)
        assert 0.0 <= acc <= 1.0

    def test_meta_steps_increments(self):
        tasks = [([_hv(i) for i in range(3)], [0,1,2],
                  [_hv(10+i) for i in range(3)], [0,1,2])]
        self.meta.meta_update(tasks)
        assert self.meta.meta_steps == 1
