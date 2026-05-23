"""
tests/test_event_hdc.py
=======================
Tests for the Event-based HDC encoding module (hdc/event_hdc.py).

Validates:
  1. DVSEvent — data class for DVS event
  2. EventHDCEncoder — encode individual and streams of events to HVs
  3. ContinuousTimeHDC — continuous-time state evolution with decay
  4. EventSNNHDCLoop — hybrid SNN+HDC event processing
"""

from __future__ import annotations

import sys
import os

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from hdc.event_hdc import (
    DVSEvent,
    EventHDCEncoder,
    ContinuousTimeHDC,
    EventSNNHDCLoop,
)
from hdc.hdc_glue import hv_hamming_sim


@pytest.fixture
def hd_dim():
    return 256

@pytest.fixture
def encoder(hd_dim):
    return EventHDCEncoder(width=8, height=8, hd_dim=hd_dim)

@pytest.fixture
def sample_event():
    return DVSEvent(x=3, y=5, t=0.001, p=1)

@pytest.fixture
def event_stream():
    events = []
    for t in range(20):
        events.append(DVSEvent(x=t % 8, y=(t * 3) % 8, t=t * 0.01, p=1 if t % 2 == 0 else -1))
    return events


class TestDVSEvent:
    def test_init(self, sample_event):
        assert sample_event.x == 3
        assert sample_event.y == 5
        assert sample_event.t == 0.001
        assert sample_event.p == 1

    def test_is_on(self, sample_event):
        assert sample_event.is_on is True

    def test_is_off(self):
        ev = DVSEvent(x=0, y=0, t=0.0, p=-1)
        assert ev.is_on is False


class TestEventHDCEncoder:
    def test_init(self, hd_dim):
        enc = EventHDCEncoder(width=8, height=8, hd_dim=hd_dim)
        assert enc.hd_dim == hd_dim

    def test_x_y_hv_shape(self, hd_dim):
        enc = EventHDCEncoder(width=8, height=8, hd_dim=hd_dim)
        assert enc._x_hv(3).shape == (hd_dim,)
        assert enc._y_hv(5).shape == (hd_dim,)

    def test_x_y_hv_deterministic(self, hd_dim):
        enc = EventHDCEncoder(width=8, height=8, hd_dim=hd_dim)
        assert torch.equal(enc._x_hv(3), enc._x_hv(3))

    def test_encode_event_shape(self, hd_dim, sample_event):
        enc = EventHDCEncoder(width=8, height=8, hd_dim=hd_dim)
        hv = enc.encode_event(sample_event)
        assert hv.shape == (hd_dim,)

    def test_encode_event_binary(self, hd_dim, sample_event):
        enc = EventHDCEncoder(width=8, height=8, hd_dim=hd_dim)
        hv = enc.encode_event(sample_event)
        assert ((hv == 0.0) | (hv == 1.0)).all()

    def test_different_locations_different(self, hd_dim):
        enc = EventHDCEncoder(width=8, height=8, hd_dim=hd_dim)
        ev1 = DVSEvent(x=0, y=0, t=0.0, p=1)
        ev2 = DVSEvent(x=7, y=7, t=0.0, p=1)
        hv1 = enc.encode_event(ev1)
        hv2 = enc.encode_event(ev2)
        sim = float(hv_hamming_sim(hv1, hv2))
        assert sim < 0.8

    def test_different_polarities_different(self, hd_dim):
        enc = EventHDCEncoder(width=8, height=8, hd_dim=hd_dim)
        ev_on = DVSEvent(x=3, y=5, t=0.0, p=1)
        ev_off = DVSEvent(x=3, y=5, t=0.0, p=-1)
        assert not torch.equal(enc.encode_event(ev_on), enc.encode_event(ev_off))

    def test_encode_stream_shape(self, hd_dim, event_stream):
        enc = EventHDCEncoder(width=8, height=8, hd_dim=hd_dim)
        hv = enc.encode_stream(event_stream)
        assert hv.shape == (hd_dim,)

    def test_encode_stream_return_all(self, hd_dim, event_stream):
        enc = EventHDCEncoder(width=8, height=8, hd_dim=hd_dim)
        hvs = enc.encode_stream(event_stream, return_all=True)
        assert hvs.shape == (len(event_stream), hd_dim)


class TestContinuousTimeHDC:
    def test_init(self, encoder):
        ct = ContinuousTimeHDC(encoder)
        assert ct.hd_dim == encoder.hd_dim

    def test_push_event_updates_state(self, encoder, sample_event):
        ct = ContinuousTimeHDC(encoder)
        old = ct.state.clone()
        ct.push_event(sample_event)
        assert not torch.equal(ct.state, old)

    def test_push_event_returns_hv(self, encoder, sample_event):
        ct = ContinuousTimeHDC(encoder)
        hv = ct.push_event(sample_event)
        assert hv.shape == (encoder.hd_dim,)
        assert ((hv == 0.0) | (hv == 1.0)).all()

    def test_state_is_binary(self, encoder):
        ct = ContinuousTimeHDC(encoder)
        assert ((ct.state == 0.0) | (ct.state == 1.0)).all()

    def test_push_stream(self, encoder, event_stream):
        ct = ContinuousTimeHDC(encoder)
        hvs = ct.push_stream(event_stream)
        assert len(hvs) == len(event_stream)

    def test_state_continuous_shape(self, encoder):
        ct = ContinuousTimeHDC(encoder)
        assert ct.state_continuous.shape == (encoder.hd_dim,)

    def test_reset(self, encoder):
        ct = ContinuousTimeHDC(encoder)
        ct.push_event(DVSEvent(x=4, y=4, t=0.1, p=1))
        ct.reset()
        assert ct.state.abs().sum() < 1e-6 or ct.state.sum() < ct.state.numel() * 0.6

    def test_on_state_update_callback(self, encoder, sample_event):
        ct = ContinuousTimeHDC(encoder)
        called = []
        def cb(hv, t):
            called.append((hv.clone(), t))
        ct.on_state_update(cb)
        ct.push_event(sample_event)
        assert len(called) > 0

    def test_decay_over_time(self, encoder):
        ct = ContinuousTimeHDC(encoder)
        ct.push_event(DVSEvent(x=4, y=4, t=0.0, p=1))
        s1 = ct.state.clone()
        ct.push_event(DVSEvent(x=4, y=4, t=1.0, p=1))
        s2 = ct.state.clone()
        assert not torch.equal(s1, s2)


class TestEventSNNHDCLoop:
    def test_init(self, encoder):
        ct = ContinuousTimeHDC(encoder)
        loop = EventSNNHDCLoop(ct)
        assert loop.current_state.shape == (encoder.hd_dim,)

    def test_push_event_returns_dict(self, encoder, sample_event):
        ct = ContinuousTimeHDC(encoder)
        loop = EventSNNHDCLoop(ct)
        result = loop.push_event(sample_event)
        assert isinstance(result, dict)

    def test_push_event_contains_state(self, encoder, sample_event):
        ct = ContinuousTimeHDC(encoder)
        loop = EventSNNHDCLoop(ct)
        result = loop.push_event(sample_event)
        assert "state_hv" in result

    def test_push_stream_async(self, encoder, event_stream):
        ct = ContinuousTimeHDC(encoder)
        loop = EventSNNHDCLoop(ct)
        results = loop.push_stream_async(event_stream)
        assert len(results) == len(event_stream)

    def test_current_state_shape(self, encoder):
        ct = ContinuousTimeHDC(encoder)
        loop = EventSNNHDCLoop(ct)
        assert loop.current_state.shape == (encoder.hd_dim,)
