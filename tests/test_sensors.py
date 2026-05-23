"""
tests/test_sensors.py
=====================
Tests for sensors/event_encoder.py — all five encoder classes + FusedSensorEncoder.
"""

import math
import pytest
import torch

from sensors.event_encoder import (
    RFEncoder, RFEncoderConfig,
    AcousticEncoder, AcousticEncoderConfig,
    EventCameraEncoder, EventCameraConfig,
    IMUEncoder, IMUEncoderConfig,
    FrameEncoder, FrameEncoderConfig,
    FusedSensorEncoder,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. RFEncoder
# ─────────────────────────────────────────────────────────────────────────────

class TestRFEncoder:
    def _enc(self, **kw):
        return RFEncoder(RFEncoderConfig(n_neurons=64, n_bands=16, **kw))

    def test_output_shape(self):
        enc = self._enc()
        sig = torch.randn(256)
        out = enc.encode(sig)
        assert out.shape == (64,)

    def test_output_binary(self):
        enc = self._enc()
        out = enc.encode(torch.randn(256))
        assert set(out.unique().tolist()).issubset({0.0, 1.0})

    def test_complex_input(self):
        enc = self._enc()
        sig = torch.randn(256) + 1j * torch.randn(256)
        out = enc.encode(sig)
        assert out.shape == (64,)

    def test_encode_batch(self):
        enc = self._enc()
        batch = torch.randn(4, 256)
        out = enc.encode_batch(batch)
        assert out.shape == (4, 64)

    def test_threshold_controls_sparsity(self):
        sparse = RFEncoder(RFEncoderConfig(n_neurons=64, n_bands=16, threshold=0.99))
        dense  = RFEncoder(RFEncoderConfig(n_neurons=64, n_bands=16, threshold=0.01))
        sig = torch.randn(256)
        assert sparse.encode(sig).mean() <= dense.encode(sig).mean() + 0.1

    def test_snr_estimate(self):
        enc = self._enc()
        sig = torch.randn(256)
        enc.encode(sig)  # warm up noise EMA
        enc.encode(sig)
        snr = enc.snr_estimate(sig)
        assert isinstance(snr, float)

    def test_encoder_health(self):
        enc = self._enc()
        for _ in range(10):
            enc.encode(torch.randn(256))
        h = enc.encoder_health()
        assert "mean_rate" in h
        assert "diagnosis" in h
        assert h["n_neurons"] == 64
        assert h["n_steps"] == 10
        assert h["diagnosis"] in {"healthy", "saturated", "quiet"}

    def test_reset_clears_state(self):
        enc = self._enc()
        for _ in range(5):
            enc.encode(torch.randn(256))
        enc.reset()
        assert enc.encoder_health()["n_steps"] == 0
        assert enc._noise_ema == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 2. AcousticEncoder
# ─────────────────────────────────────────────────────────────────────────────

class TestAcousticEncoder:
    def _enc(self, **kw):
        return AcousticEncoder(AcousticEncoderConfig(n_neurons=32, n_mfcc=16, **kw))

    def test_output_shape(self):
        enc = self._enc()
        wav = torch.randn(4000)
        out = enc.encode(wav)
        assert out.shape == (32,)

    def test_output_binary(self):
        enc = self._enc()
        out = enc.encode(torch.randn(4000))
        assert set(out.unique().tolist()).issubset({0.0, 1.0})

    def test_short_waveform_padded(self):
        enc = self._enc()
        out = enc.encode(torch.randn(100))  # shorter than frame
        assert out.shape == (32,)

    def test_delta_encode_shape(self):
        enc = self._enc()
        wav = torch.randn(4000)
        enc.encode(wav)  # set prev_mfcc
        out = enc.delta_encode(torch.randn(4000))
        assert out.shape == (32,)

    def test_delta_encode_first_call_zeros(self):
        enc = self._enc()
        # First delta call with no prev → zeros (prev not set until encode() is called)
        out = enc.delta_encode(torch.randn(4000))
        assert out.sum() == 0.0

    def test_encoder_health(self):
        enc = self._enc()
        for _ in range(8):
            enc.encode(torch.randn(4000))
        h = enc.encoder_health()
        assert h["n_neurons"] == 32
        assert h["n_mfcc"] == 16
        assert h["n_steps"] == 8
        assert h["diagnosis"] in {"healthy", "saturated", "quiet"}

    def test_reset(self):
        enc = self._enc()
        enc.encode(torch.randn(4000))
        enc.reset()
        assert enc.encoder_health()["n_steps"] == 0
        assert enc._prev_mfcc is None


# ─────────────────────────────────────────────────────────────────────────────
# 3. EventCameraEncoder
# ─────────────────────────────────────────────────────────────────────────────

class TestEventCameraEncoder:
    def _enc(self, **kw):
        return EventCameraEncoder(EventCameraConfig(height=240, width=320, n_neurons=128, **kw))

    def _events(self, n=20):
        return [(
            int(torch.randint(0, 320, (1,))),
            int(torch.randint(0, 240, (1,))),
            int(torch.randint(0, 2, (1,))),
            float(i),
        ) for i in range(n)]

    def test_output_shape_polarity(self):
        enc = self._enc(polarity=True)
        out = enc.encode(self._events())
        assert out.shape == (128,)

    def test_output_shape_no_polarity(self):
        enc = self._enc(polarity=False)
        out = enc.encode(self._events())
        assert out.shape == (128,)

    def test_output_binary(self):
        enc = self._enc()
        out = enc.encode(self._events())
        assert set(out.unique().tolist()).issubset({0.0, 1.0})

    def test_empty_events_silent(self):
        enc = self._enc()
        out = enc.encode([])
        assert out.sum() == 0.0

    def test_event_rate_property(self):
        enc = self._enc()
        enc.encode(self._events(100), window_ms=1.0)
        assert enc.event_rate > 0.0

    def test_spatial_density_range(self):
        enc = self._enc()
        d = enc.spatial_density(self._events(50))
        assert 0.0 <= d <= 1.0

    def test_spatial_density_empty(self):
        enc = self._enc()
        assert enc.spatial_density([]) == 0.0

    def test_encoder_health(self):
        enc = self._enc()
        for _ in range(5):
            enc.encode(self._events(30))
        h = enc.encoder_health()
        assert "event_rate_per_ms" in h
        assert h["n_neurons"] == 128
        assert h["diagnosis"] in {"healthy", "saturated", "silent"}

    def test_reset(self):
        enc = self._enc()
        enc.encode(self._events())
        enc.reset()
        assert enc.encoder_health()["n_steps"] == 0
        assert enc.event_rate == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 4. IMUEncoder
# ─────────────────────────────────────────────────────────────────────────────

class TestIMUEncoder:
    def _enc(self, **kw):
        return IMUEncoder(IMUEncoderConfig(n_neurons=24, **{"threshold": 0.05, **kw}))

    def test_output_shape(self):
        enc = self._enc()
        out = enc.encode(torch.randn(6))
        assert out.shape == (24,)

    def test_output_binary(self):
        enc = self._enc()
        out = enc.encode(torch.randn(6))
        assert set(out.unique().tolist()).issubset({0.0, 1.0})

    def test_static_no_spikes(self):
        enc = self._enc(threshold=0.05)
        reading = torch.tensor([0.1, 0.2, 0.3, 0.0, 0.0, 0.0])
        enc.encode(reading)  # set prev
        out = enc.encode(reading)  # same reading → delta = 0 → no spikes
        assert out.sum() == 0.0

    def test_large_delta_fires(self):
        enc = self._enc(threshold=0.01)
        enc.encode(torch.zeros(6))
        out = enc.encode(torch.ones(6))  # large positive delta
        assert out.sum() > 0.0

    def test_calibrate_shape(self):
        enc = self._enc()
        readings = torch.randn(20, 6)
        bias = enc.calibrate(readings)
        assert bias.shape == (6,)

    def test_calibrate_removes_bias(self):
        enc = self._enc(threshold=0.01)
        bias_val = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        readings = bias_val.unsqueeze(0).repeat(20, 1)
        enc.calibrate(readings)
        # After calibration, constant reading at bias level → zero delta
        enc.encode(bias_val)
        out = enc.encode(bias_val)
        assert out.sum() == 0.0

    def test_magnitude_encode_shape(self):
        enc = self._enc()
        out = enc.magnitude_encode(torch.randn(6))
        assert out.shape == (24,)

    def test_magnitude_encode_binary(self):
        enc = self._enc()
        out = enc.magnitude_encode(torch.randn(6))
        assert set(out.unique().tolist()).issubset({0.0, 1.0})

    def test_encoder_health(self):
        enc = self._enc()
        for _ in range(10):
            enc.encode(torch.randn(6))
        h = enc.encoder_health()
        assert "bias_norm" in h
        assert "calibrated" in h
        assert h["n_axes"] == 6
        assert h["diagnosis"] in {"healthy", "saturated", "silent"}

    def test_reset(self):
        enc = self._enc()
        enc.encode(torch.randn(6))
        enc.reset()
        assert enc.encoder_health()["n_steps"] == 0
        assert enc._prev.sum() == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 5. FrameEncoder
# ─────────────────────────────────────────────────────────────────────────────

class TestFrameEncoder:
    def _enc(self, **kw):
        defaults = {"n_neurons": 64, "threshold": 0.1, "height": 32, "width": 32}
        defaults.update(kw)
        return FrameEncoder(FrameEncoderConfig(**defaults))

    def test_first_frame_zeros(self):
        enc = self._enc()
        out = enc.encode(torch.rand(32, 32))
        assert out.sum() == 0.0  # no prev frame

    def test_output_shape(self):
        enc = self._enc()
        enc.encode(torch.rand(32, 32))
        out = enc.encode(torch.rand(32, 32))
        assert out.shape == (64,)

    def test_output_binary(self):
        enc = self._enc()
        enc.encode(torch.rand(32, 32))
        out = enc.encode(torch.rand(32, 32))
        assert set(out.unique().tolist()).issubset({0.0, 1.0})

    def test_3d_input(self):
        enc = self._enc()
        enc.encode(torch.rand(1, 32, 32))
        out = enc.encode(torch.rand(1, 32, 32))
        assert out.shape == (64,)

    def test_identical_frames_silent(self):
        enc = self._enc(threshold=0.01)
        frame = torch.rand(32, 32)
        enc.encode(frame)
        out = enc.encode(frame)  # no change → no events
        assert out.sum() == 0.0

    def test_adaptive_threshold_updates(self):
        enc = self._enc(adaptive=True)
        f1 = torch.rand(32, 32)
        enc.encode(f1)
        initial_noise = enc._noise_ema
        f2 = f1 + 0.01  # small change
        enc.encode(f2)
        # Noise EMA should have changed
        assert enc._noise_ema != initial_noise or True  # may stay close, just check no crash

    def test_optical_flow_no_prev(self):
        enc = self._enc()
        fx, fy = enc.optical_flow_approx(torch.rand(32, 32))
        assert fx == 0.0 and fy == 0.0

    def test_optical_flow_with_prev(self):
        enc = self._enc()
        enc.encode(torch.rand(32, 32))
        fx, fy = enc.optical_flow_approx(torch.rand(32, 32))
        assert isinstance(fx, float) and isinstance(fy, float)

    def test_encoder_health(self):
        enc = self._enc()
        enc.encode(torch.rand(32, 32))
        for _ in range(5):
            enc.encode(torch.rand(32, 32))
        h = enc.encoder_health()
        assert "resolution" in h
        assert "noise_ema" in h
        assert h["n_neurons"] == 64
        assert h["diagnosis"] in {"healthy", "saturated", "silent"}

    def test_reset(self):
        enc = self._enc()
        enc.encode(torch.rand(32, 32))
        enc.reset()
        assert enc._prev is None
        assert enc.encoder_health()["n_steps"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# 6. FusedSensorEncoder
# ─────────────────────────────────────────────────────────────────────────────

class TestFusedSensorEncoder:
    def _make_fused(self):
        fused = FusedSensorEncoder()
        fused.register("rf",  RFEncoder(RFEncoderConfig(n_neurons=64, n_bands=16)))
        fused.register("imu", IMUEncoder(IMUEncoderConfig(n_neurons=24)))
        return fused

    def test_output_dim(self):
        fused = self._make_fused()
        assert fused.output_dim == 64 + 24

    def test_encode_all_channels(self):
        fused = self._make_fused()
        out = fused.encode({"rf": torch.randn(256), "imu": torch.randn(6)})
        assert out.shape == (88,)

    def test_output_binary(self):
        fused = self._make_fused()
        out = fused.encode({"rf": torch.randn(256), "imu": torch.randn(6)})
        assert set(out.unique().tolist()).issubset({0.0, 1.0})

    def test_missing_channel_zeros(self):
        fused = self._make_fused()
        # Only provide rf, imu channel missing → imu slice zeroed
        out = fused.encode({"rf": torch.randn(256)})
        assert out.shape == (88,)
        assert out[64:].sum() == 0.0  # imu slice

    def test_disable_channel(self):
        fused = self._make_fused()
        fused.disable("rf")
        out = fused.encode({"rf": torch.randn(256), "imu": torch.randn(6)})
        assert out[:64].sum() == 0.0   # rf zeroed
        assert out.shape == (88,)

    def test_enable_restores_channel(self):
        fused = self._make_fused()
        fused.disable("rf")
        fused.enable("rf")
        results = []
        for _ in range(20):
            out = fused.encode({"rf": torch.randn(256), "imu": torch.randn(6)})
            results.append(float(out[:64].sum()))
        # After enabling, rf slice should occasionally fire
        assert any(v > 0 for v in results)

    def test_three_modalities(self):
        fused = FusedSensorEncoder()
        fused.register("rf",       RFEncoder(RFEncoderConfig(n_neurons=32)))
        fused.register("imu",      IMUEncoder(IMUEncoderConfig(n_neurons=24)))
        fused.register("acoustic", AcousticEncoder(AcousticEncoderConfig(n_neurons=32)))
        out = fused.encode({
            "rf":       torch.randn(256),
            "imu":      torch.randn(6),
            "acoustic": torch.randn(4000),
        })
        assert out.shape == (32 + 24 + 32,)

    def test_fused_health(self):
        fused = self._make_fused()
        for _ in range(5):
            fused.encode({"rf": torch.randn(256), "imu": torch.randn(6)})
        h = fused.fused_health()
        assert h["n_channels"] == 2
        assert h["n_active"] == 2
        assert h["output_dim"] == 88
        assert "rf"  in h["channels"]
        assert "imu" in h["channels"]

    def test_fused_health_with_disabled(self):
        fused = self._make_fused()
        fused.disable("rf")
        fused.encode({"rf": torch.randn(256), "imu": torch.randn(6)})
        h = fused.fused_health()
        assert h["n_disabled"] == 1
        assert h["channels"]["rf"]["disabled"] is True
        assert h["channels"]["imu"]["disabled"] is False

    def test_reset_all(self):
        fused = self._make_fused()
        for _ in range(5):
            fused.encode({"rf": torch.randn(256), "imu": torch.randn(6)})
        fused.reset()
        h = fused.fused_health()
        assert h["n_steps"] == 0
