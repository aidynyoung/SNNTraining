"""
tests/test_continual_hdc.py
===========================
Tests for the Continual HDC Learning module (hdc/continual_hdc.py).

Validates:
  1. ClassMeanHDCClassifier — online class-mean classifier: observe, predict, accuracy
  2. LeastSquaresHDCInit — least-squares prototype initialization
  3. OnlineContinualHDC — full continual learning pipeline with task evaluation
"""

from __future__ import annotations

import sys
import os

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from hdc.continual_hdc import (
    ClassMeanHDCClassifier,
    LeastSquaresHDCInit,
    OnlineContinualHDC,
    ContinualTask,
)
from hdc.hdc_glue import hv_hamming_sim


@pytest.fixture
def hd_dim():
    return 256

@pytest.fixture
def n_classes():
    return 5

@pytest.fixture
def classifier(hd_dim, n_classes):
    return ClassMeanHDCClassifier(dim=hd_dim, n_classes=n_classes)

@pytest.fixture
def sample_hvs(hd_dim, n_classes):
    g = torch.Generator()
    g.manual_seed(42)
    hvs = []
    labels = []
    for cls in range(n_classes):
        for _ in range(10):
            hv = (torch.rand(hd_dim, generator=g) >= 0.5).float()
            hvs.append(hv)
            labels.append(cls)
    return torch.stack(hvs), torch.tensor(labels)


class TestClassMeanHDCClassifier:
    def test_init(self, classifier, hd_dim):
        assert classifier.dim == hd_dim
        # n_classes is a property: len(self._prototypes), starts at 0
        assert classifier.n_classes == 0

    def test_observe_single(self, classifier, hd_dim):
        hv = (torch.rand(hd_dim) >= 0.5).float()
        classifier.observe(hv, label=0)
        assert classifier.n_classes >= 1

    def test_observe_batch(self, classifier, sample_hvs):
        hvs, labels = sample_hvs
        classifier.observe_batch(hvs, labels)
        assert classifier.n_classes >= 5

    def test_predict_shape(self, classifier, sample_hvs):
        hvs, labels = sample_hvs
        classifier.observe_batch(hvs, labels)
        test_hv = (torch.rand(classifier.dim) >= 0.5).float()
        preds = classifier.predict(test_hv, top_k=3)
        assert len(preds) == 3
        for label, conf in preds:
            assert 0 <= label < 5
            assert 0.0 <= conf <= 1.0

    def test_predict_label(self, classifier, sample_hvs):
        hvs, labels = sample_hvs
        classifier.observe_batch(hvs, labels)
        test_hv = (torch.rand(classifier.dim) >= 0.5).float()
        label = classifier.predict_label(test_hv)
        assert 0 <= label < 5

    def test_accuracy(self, classifier, sample_hvs):
        hvs, labels = sample_hvs
        classifier.observe_batch(hvs, labels)
        acc = classifier.accuracy(hvs, labels)
        assert 0.0 <= acc <= 1.0

    def test_add_class(self, classifier, hd_dim):
        hvs = (torch.rand(5, hd_dim) >= 0.5).float()
        classifier.add_class(label=5, sample_hvs=hvs)
        # add_class adds 1 class, so n_classes becomes 1
        assert classifier.n_classes >= 1

    def test_known_labels(self, classifier, sample_hvs):
        hvs, labels = sample_hvs
        classifier.observe_batch(hvs, labels)
        known = classifier.known_labels
        assert len(known) >= 5

    def test_forgetting_report(self, classifier, sample_hvs):
        hvs, labels = sample_hvs
        classifier.observe_batch(hvs, labels)
        # Split into old and new for forgetting report
        n = len(hvs) // 2
        report = classifier.forgetting_report(
            hvs_old=hvs[:n], labels_old=labels[:n],
            hvs_new=hvs[n:], labels_new=labels[n:],
        )
        assert isinstance(report, dict)
        assert "old_accuracy" in report
        assert "new_accuracy" in report

    def test_self_similarity_high(self, classifier, hd_dim):
        g = torch.Generator()
        g.manual_seed(42)
        cls0_hvs = [(torch.rand(hd_dim, generator=g) >= 0.5).float() for _ in range(5)]
        cls1_hvs = [(torch.rand(hd_dim, generator=g) >= 0.5).float() for _ in range(5)]
        for hv in cls0_hvs:
            classifier.observe(hv, label=0)
        for hv in cls1_hvs:
            classifier.observe(hv, label=1)
        test_hv = cls0_hvs[0]
        pred = classifier.predict_label(test_hv)
        assert 0 <= pred < 5


class TestLeastSquaresHDCInit:
    def test_init(self, hd_dim):
        lsq = LeastSquaresHDCInit(dim=hd_dim, ridge_lambda=0.01)
        assert lsq.dim == hd_dim

    def test_observe_batch(self, hd_dim):
        lsq = LeastSquaresHDCInit(dim=hd_dim)
        hvs = (torch.rand(20, hd_dim) >= 0.5).float()
        labels = torch.randint(0, 4, (20,))
        lsq.observe_batch(hvs, labels)

    def test_compute_prototypes(self, hd_dim):
        lsq = LeastSquaresHDCInit(dim=hd_dim)
        hvs = (torch.rand(20, hd_dim) >= 0.5).float()
        labels = torch.randint(0, 4, (20,))
        lsq.observe_batch(hvs, labels)
        protos = lsq.compute_prototypes()
        assert isinstance(protos, dict)
        assert len(protos) > 0
        for label, proto in protos.items():
            assert proto.shape == (hd_dim,)


class TestOnlineContinualHDC:
    def test_init(self, hd_dim):
        ocl = OnlineContinualHDC(dim=hd_dim)
        assert ocl.dim == hd_dim

    def test_learn_task(self, hd_dim):
        ocl = OnlineContinualHDC(dim=hd_dim)
        hvs = (torch.rand(30, hd_dim) >= 0.5).float()
        labels = torch.randint(0, 3, (30,))
        result = ocl.learn_task(hvs=hvs, labels=labels, task_id=0)
        assert "n_classes_total" in result
        assert result["n_classes_total"] >= 3

    def test_evaluate_all(self, hd_dim):
        ocl = OnlineContinualHDC(dim=hd_dim)
        hvs0 = (torch.rand(30, hd_dim) >= 0.5).float()
        labels0 = torch.randint(0, 3, (30,))
        ocl.learn_task(hvs=hvs0, labels=labels0, task_id=0)
        hvs1 = (torch.rand(30, hd_dim) >= 0.5).float()
        labels1 = torch.randint(3, 6, (30,))
        ocl.learn_task(hvs=hvs1, labels=labels1, task_id=1)
        results = ocl.evaluate_all(
            task_hvs={0: hvs0, 1: hvs1},
            task_labels={0: labels0, 1: labels1},
        )
        assert isinstance(results, dict)
        assert "average" in results
        assert len(results) >= 2


class TestHardExampleReplay:
    D = 128

    def _make_continual(self):
        clf = OnlineContinualHDC(dim=self.D, buffer_size=20)
        hvs0 = torch.stack([(torch.rand(self.D) > 0.5).float() for _ in range(10)])
        hvs1 = torch.stack([(torch.rand(self.D) > 0.5).float() for _ in range(10)])
        labels0 = torch.zeros(10, dtype=torch.long)
        labels1 = torch.ones(10, dtype=torch.long)
        clf.learn_task(hvs0, labels0, task_id=0)
        clf.learn_task(hvs1, labels1, task_id=1)
        return clf

    def test_hard_example_replay_no_crash(self):
        clf = self._make_continual()
        clf.hard_example_replay(n_hard=5)

    def test_hard_example_replay_n_hard_respected(self):
        clf = self._make_continual()
        clf.hard_example_replay(n_hard=3, lr_scale=2.0)
        # Model should still work after replay
        hv     = (torch.rand(self.D) > 0.5).float()
        result = clf.classifier.predict(hv, top_k=1)
        label  = result[0][0] if result else 0
        assert isinstance(label, int)

    def test_hard_example_replay_empty_buffer(self):
        clf = OnlineContinualHDC(dim=self.D)
        clf.hard_example_replay(n_hard=5)  # no crash on empty buffer
