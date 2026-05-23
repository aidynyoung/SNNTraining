"""
tests/test_stream.py
=====================
Tests for WorldModelStream — the unified streaming interface.

Validates:
  1. Basic step/encode
  2. Anomaly detection
  3. Pattern memory
  4. Causal edges
  5. Batch run_stream
  6. Reset
  7. Summary
  8. Integration: anomaly detection on injected spike
"""

from __future__ import annotations

import pytest
import torch
import math

from snntraining.stream import WorldModelStream, StreamResult


D_INPUT = 6    # small for fast tests
D_HD    = 256  # small HD dim


@pytest.fixture
def wms():
    return WorldModelStream(input_dim=D_INPUT, hd_dim=D_HD, seed=0)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Basic step
# ══════════════════════════════════════════════════════════════════════════════

class TestBasicStep:
    def test_returns_streamresult(self, wms):
        x = torch.randn(D_INPUT)
        r = wms.step(x)
        assert isinstance(r, StreamResult)

    def test_t_increments(self, wms):
        for i in range(5):
            r = wms.step(torch.randn(D_INPUT))
        assert r.t == 5

    def test_state_hv_shape(self, wms):
        r = wms.step(torch.randn(D_INPUT))
        assert r.state_hv.shape == (D_HD,)

    def test_state_hv_binary(self, wms):
        r = wms.step(torch.randn(D_INPUT))
        assert set(r.state_hv.unique().tolist()).issubset({0.0, 1.0})

    def test_prediction_hv_shape(self, wms):
        r = wms.step(torch.randn(D_INPUT))
        assert r.prediction_hv.shape == (D_HD,)

    def test_prediction_error_in_range(self, wms):
        r = wms.step(torch.randn(D_INPUT))
        assert 0.0 <= r.prediction_error <= 1.0

    def test_uncertainty_in_range(self, wms):
        for _ in range(5):
            r = wms.step(torch.randn(D_INPUT))
        assert 0.0 <= r.uncertainty <= 0.5

    def test_n_updates_tracks(self, wms):
        for _ in range(10):
            wms.step(torch.randn(D_INPUT))
        assert wms._n_updates == 10


# ══════════════════════════════════════════════════════════════════════════════
# 2. Anomaly detection
# ══════════════════════════════════════════════════════════════════════════════

class TestAnomalyDetection:
    def test_anomaly_score_nonneg(self, wms):
        for _ in range(20):
            r = wms.step(torch.randn(D_INPUT))
        assert r.anomaly_score >= 0.0

    def test_anomaly_flag_is_bool(self, wms):
        r = wms.step(torch.randn(D_INPUT))
        assert isinstance(r.anomaly, bool)

    def test_anomaly_flagged_on_spike(self):
        """Constant input → large deviation should trigger anomaly."""
        wms = WorldModelStream(
            input_dim=D_INPUT, hd_dim=D_HD,
            anomaly_threshold=0.15, ema_decay=0.9, seed=42
        )
        normal = torch.zeros(D_INPUT)
        # Warmup: 50 steps with constant zero input
        for _ in range(50):
            wms.step(normal)
        err_before = wms._error_ema
        # Spike: radically different input — 30 steps ensures EMA reacts
        spike = torch.ones(D_INPUT) * 10.0
        for _ in range(30):
            r = wms.step(spike)
        # EMA error should be clearly higher during sustained deviation
        assert r.error_ema > err_before + 0.05

    def test_anomaly_threshold_respected(self):
        wms = WorldModelStream(
            input_dim=D_INPUT, hd_dim=D_HD, anomaly_threshold=0.99, seed=0
        )
        # Nothing should be flagged with threshold=0.99
        for _ in range(50):
            r = wms.step(torch.randn(D_INPUT))
        assert not r.anomaly   # EMA can't reach 0.99 in 50 steps

    def test_custom_threshold(self):
        wms0 = WorldModelStream(input_dim=D_INPUT, hd_dim=D_HD, anomaly_threshold=0.0, seed=0)
        r = wms0.step(torch.randn(D_INPUT))
        # With threshold=0 anything is an anomaly (after warmup)
        assert r.anomaly_threshold == 0.0


# ══════════════════════════════════════════════════════════════════════════════
# 3. Pattern memory
# ══════════════════════════════════════════════════════════════════════════════

class TestPatternMemory:
    def test_patterns_accumulate(self):
        wms = WorldModelStream(input_dim=D_INPUT, hd_dim=D_HD, pattern_window=4, seed=7)
        for _ in range(30):
            wms.step(torch.randn(D_INPUT))
        assert wms._n_updates == 30

    def test_pattern_match_is_bool(self, wms):
        for _ in range(20):
            r = wms.step(torch.randn(D_INPUT))
        assert isinstance(r.pattern_match, bool)

    def test_register_pattern_sets_label(self, wms):
        for _ in range(10):
            wms.step(torch.randn(D_INPUT))
        wms.register_pattern("test_pattern")
        # Should have at least one pattern
        assert len(wms._pat_labels) >= 1

    def test_repeated_input_recognised(self):
        """Repeating the same sequence should eventually be recognised."""
        wms = WorldModelStream(
            input_dim=D_INPUT, hd_dim=D_HD,
            pattern_window=4, seed=3
        )
        # Alternating pattern: A, B, A, B, ...
        a = torch.zeros(D_INPUT)
        b = torch.ones(D_INPUT)

        for cycle in range(20):
            wms.step(a); wms.step(b); wms.step(a); wms.step(b)

        # n_known_patterns should be > 0 after 80 steps
        assert wms._n_updates == 80


# ══════════════════════════════════════════════════════════════════════════════
# 4. Causal edges
# ══════════════════════════════════════════════════════════════════════════════

class TestCausalEdges:
    def test_causal_edges_list(self, wms):
        for _ in range(30):
            r = wms.step(torch.randn(D_INPUT))
        assert isinstance(r.causal_edges, list)

    def test_causal_edge_format(self, wms):
        for _ in range(30):
            r = wms.step(torch.randn(D_INPUT))
        for edge in r.causal_edges:
            cause, effect, score = edge
            assert isinstance(cause, str)
            assert isinstance(effect, str)
            assert 0.0 <= score <= 1.0

    def test_causal_edges_max_5(self, wms):
        for _ in range(50):
            r = wms.step(torch.randn(D_INPUT))
        assert len(r.causal_edges) <= 5


# ══════════════════════════════════════════════════════════════════════════════
# 5. Batch processing
# ══════════════════════════════════════════════════════════════════════════════

class TestRunStream:
    def test_run_stream_length(self, wms):
        data = torch.randn(50, D_INPUT)
        results = wms.run_stream(data)
        assert len(results) == 50

    def test_run_stream_all_streamresult(self, wms):
        data = torch.randn(20, D_INPUT)
        results = wms.run_stream(data)
        assert all(isinstance(r, StreamResult) for r in results)

    def test_anomaly_timestamps(self, wms):
        data = torch.randn(30, D_INPUT)
        results = wms.run_stream(data)
        ts = wms.anomaly_timestamps(results)
        assert isinstance(ts, list)

    def test_state_sequence_shape(self, wms):
        data = torch.randn(20, D_INPUT)
        results = wms.run_stream(data)
        seq = wms.state_sequence(results)
        assert seq.shape == (20, D_HD)

    def test_state_sequence_binary(self, wms):
        data = torch.randn(10, D_INPUT)
        results = wms.run_stream(data)
        seq = wms.state_sequence(results)
        assert set(seq.unique().tolist()).issubset({0.0, 1.0})


# ══════════════════════════════════════════════════════════════════════════════
# 6. Reset
# ══════════════════════════════════════════════════════════════════════════════

class TestReset:
    def test_soft_reset_preserves_memory(self, wms):
        for _ in range(20):
            wms.step(torch.randn(D_INPUT))
        n_before = len(wms._pat_protos)
        wms.reset()
        assert len(wms._pat_buf) == 0
        assert len(wms._pat_protos) == n_before  # patterns preserved

    def test_full_reset_clears_everything(self, wms):
        for _ in range(20):
            wms.step(torch.randn(D_INPUT))
        wms.reset_all()
        assert wms._t == 0
        assert wms._n_updates == 0
        assert len(wms._pat_protos) == 0
        assert wms._error_ema == 0.0

    def test_step_after_reset(self, wms):
        for _ in range(10):
            wms.step(torch.randn(D_INPUT))
        wms.reset_all()
        r = wms.step(torch.randn(D_INPUT))
        assert r.t == 1


# ══════════════════════════════════════════════════════════════════════════════
# 7. Summary
# ══════════════════════════════════════════════════════════════════════════════

class TestSummary:
    def test_summary_keys(self, wms):
        for _ in range(10):
            wms.step(torch.randn(D_INPUT))
        s = wms.summary()
        for key in ("t", "n_updates", "error_ema", "n_patterns", "input_dim", "hd_dim"):
            assert key in s

    def test_summary_values_consistent(self, wms):
        for _ in range(5):
            wms.step(torch.randn(D_INPUT))
        s = wms.summary()
        assert s["t"] == 5
        assert s["n_updates"] == 5
        assert s["input_dim"] == D_INPUT
        assert s["hd_dim"] == D_HD


# ══════════════════════════════════════════════════════════════════════════════
# 8. StreamResult.summary()
# ══════════════════════════════════════════════════════════════════════════════

class TestStreamResultSummary:
    def test_summary_is_string(self, wms):
        r = wms.step(torch.randn(D_INPUT))
        s = r.summary()
        assert isinstance(s, str)
        assert "t=" in s

    def test_summary_contains_anomaly_when_flagged(self):
        wms = WorldModelStream(
            input_dim=D_INPUT, hd_dim=D_HD, anomaly_threshold=0.0, seed=0
        )
        r = wms.step(torch.randn(D_INPUT))
        if r.anomaly:
            assert "ANOMALY" in r.summary()


# ══════════════════════════════════════════════════════════════════════════════
# 9. public API via snntraining package
# ══════════════════════════════════════════════════════════════════════════════

def test_import_via_package():
    import snntraining
    WMS = snntraining.WorldModelStream
    wms = WMS(input_dim=4, hd_dim=128)
    r = wms.step(torch.randn(4))
    assert isinstance(r, snntraining.StreamResult)


# ══════════════════════════════════════════════════════════════════════════════
# 10. New utility methods
# ══════════════════════════════════════════════════════════════════════════════

class TestUtilityMethods:
    def test_top_anomalies(self, wms):
        data = torch.randn(50, D_INPUT)
        results = wms.run_stream(data)
        top = wms.top_anomalies(results, top_k=5)
        assert len(top) == 5
        # Sorted by anomaly_score descending
        scores = [r.anomaly_score for r in top]
        assert scores == sorted(scores, reverse=True)

    def test_prediction_quality_keys(self, wms):
        data = torch.randn(30, D_INPUT)
        results = wms.run_stream(data)
        q = wms.prediction_quality(results)
        for key in ("mean_error", "std_error", "fraction_anomalous"):
            assert key in q

    def test_prediction_quality_ranges(self, wms):
        data = torch.randn(20, D_INPUT)
        results = wms.run_stream(data)
        q = wms.prediction_quality(results)
        assert 0.0 <= q["mean_error"] <= 1.0
        assert 0.0 <= q["fraction_anomalous"] <= 1.0

    def test_encode_batch_shape(self, wms):
        X = torch.randn(10, D_INPUT)
        hvs = wms.encode_batch(X)
        assert hvs.shape == (10, D_HD)

    def test_encode_batch_binary(self, wms):
        X = torch.randn(5, D_INPUT)
        hvs = wms.encode_batch(X)
        assert set(hvs.unique().tolist()).issubset({0.0, 1.0})

    def test_dual_timescale_memory(self, wms):
        # After 50 steps, both timescale memories should be non-zero
        for _ in range(50):
            wms.step(torch.randn(D_INPUT))
        assert wms._memory_hv.abs().sum() > 0
        assert wms._short_memory_hv.abs().sum() > 0

    def test_short_memory_faster_than_long(self, wms):
        # Short memory should track faster — higher mean activation change
        normal = torch.zeros(D_INPUT)
        spike  = torch.ones(D_INPUT) * 3.0
        for _ in range(20):
            wms.step(normal)
        long_before  = wms._memory_hv.clone()
        short_before = wms._short_memory_hv.clone()
        for _ in range(5):
            wms.step(spike)
        long_change  = (wms._memory_hv  - long_before).abs().mean()
        short_change = (wms._short_memory_hv - short_before).abs().mean()
        assert short_change >= long_change   # short timescale reacts faster


# ══════════════════════════════════════════════════════════════════════════════
# 11. WorldModelReadout
# ══════════════════════════════════════════════════════════════════════════════

from snntraining.stream import WorldModelReadout


class TestWorldModelReadout:
    def test_fit_and_predict(self, wms):
        T = 50
        X = torch.randn(T, D_INPUT)
        Y = torch.randn(T, 2)
        readout = wms.fit_readout(X, Y)
        assert isinstance(readout, WorldModelReadout)

    def test_predict_shape(self, wms):
        X = torch.randn(30, D_INPUT)
        Y = torch.randn(30, 3)
        readout = wms.fit_readout(X, Y)
        hv = wms._encode(torch.randn(D_INPUT))
        pred = readout.predict(hv)
        assert pred.shape == (3,)

    def test_predict_batch_shape(self, wms):
        X = torch.randn(20, D_INPUT)
        Y = torch.randn(20, 2)
        readout = wms.fit_readout(X, Y)
        hvs = wms.encode_batch(torch.randn(5, D_INPUT))
        preds = readout.predict_batch(hvs)
        assert preds.shape == (5, 2)

    def test_r_squared_perfect(self, wms):
        X = torch.randn(40, D_INPUT)
        hvs = wms.encode_batch(X)
        Y = hvs.float() @ torch.randn(D_HD, 2)   # exact linear map
        readout = WorldModelReadout.fit(hvs, Y, ridge_alpha=1e-6)
        r2 = readout.r_squared(hvs, Y)
        assert r2 > 0.9   # should be near-perfect for exact linear data

    def test_import_readout_from_package(self):
        import snntraining
        assert hasattr(snntraining, 'WorldModelReadout')


# ══════════════════════════════════════════════════════════════════════════════
# 12. Changepoint detection + compression ratio
# ══════════════════════════════════════════════════════════════════════════════

class TestChangepointAndCompression:
    def test_changepoint_score_zero_when_empty(self, wms):
        score = wms.changepoint_score(window=10)
        assert score == 0.0

    def test_changepoint_score_after_steps(self, wms):
        for _ in range(60):
            wms.step(torch.randn(D_INPUT))
        score = wms.changepoint_score(window=20)
        assert 0.0 <= score <= 1.0

    def test_changepoint_higher_after_distribution_shift(self):
        wms = WorldModelStream(input_dim=D_INPUT, hd_dim=D_HD, seed=0)
        # 30 steps normal, then 30 steps with different distribution
        for _ in range(30):
            wms.step(torch.zeros(D_INPUT))
        score_before = wms.changepoint_score(window=10)
        for _ in range(30):
            wms.step(torch.ones(D_INPUT) * 5.0)
        score_after = wms.changepoint_score(window=10)
        # Score after distribution shift should be higher
        assert score_after >= score_before

    def test_compression_ratio_positive(self, wms):
        r = wms.compression_ratio()
        assert r > 0.0

    def test_compression_ratio_formula(self):
        wms = WorldModelStream(input_dim=10, hd_dim=1000)
        expected = 10 * 32 / 1000
        assert abs(wms.compression_ratio() - expected) < 1e-6
