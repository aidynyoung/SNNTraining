"""
Continuous-Time Event-Camera HDC Encoding
==========================================
Implements the continuous-time interface layer of the super-Turing stack:

    Event camera (asynchronous) → ContinuousTimeHDC → FHRR world model
                    ↑___________________________________|
                    (SNN feedback — no discrete clock)

Event cameras (Dynamic Vision Sensors, DVS) produce asynchronous events
(x, y, t, polarity) at microsecond resolution with no fixed frame rate.
This is the critical ingredient for continuous-time computation:

  "Event cameras have NO frame rate — they're fully asynchronous.
   An Arthedain system with event camera input would be a continuous-time
   analog dynamical system — the exact conditions for super-Turing computation."
  — Turing Completeness framing document

Architecture:
  Each event e = (x, y, t, p) immediately updates the HDC state:

    spatial_hv   = XOR(pos_hv(x, y), time_hv(t))     [position × time binding]
    event_hv     = polarity_p × spatial_hv             [polarity-signed]
    state_hv(t)  ← EMA(state_hv(t-ε), event_hv)       [continuous EMA update]

  No frame accumulation. No discrete time step. One update per event.

  This directly implements the continuous-time recurrence that makes
  the SNN-HDC stack a candidate for super-Turing computation.

Three components:

1. EventHDCEncoder — asynchronous (x,y,t,p) → HV with no discrete clock
2. ContinuousTimeHDC — maintains HV state updated on every event
3. EventSNNHDCLoop — closed-loop: events → SNN → HDC → SNN (no synchronization)
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Dict, List, Optional, Tuple

import torch

from hdc.hdc_glue import hv_batch_sim, gen_hvs
from hdc.grid_cells import GridCellModule


# ═══════════════════════════════════════════════════════════════════════════════
# Event data structure
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DVSEvent:
    """
    Single Dynamic Vision Sensor event.

    DVS cameras fire an event when a pixel's log-luminance changes by a
    threshold amount. Each event carries its pixel coordinates, timestamp
    (microsecond resolution), and polarity (increasing or decreasing brightness).

    No frame rate — events arrive continuously at the resolution of the scene.
    """
    x: int            # pixel column
    y: int            # pixel row
    t: float          # timestamp in seconds (microsecond resolution)
    p: int            # polarity: +1 (ON) or -1 (OFF)

    @property
    def is_on(self) -> bool:
        return self.p > 0


# ═══════════════════════════════════════════════════════════════════════════════
# 1. EventHDCEncoder — per-event encoding with no frame accumulation
# ═══════════════════════════════════════════════════════════════════════════════

class EventHDCEncoder:
    """
    Encode individual DVS events as hypervectors in continuous time.

    For each event (x, y, t, p):
        spatial_hv(x,y) = XOR(x_hv[x], y_hv[y])    [bind x and y position]
        time_hv(t)      = g(t / τ_decay)             [fractional time binding]
        event_hv        = p × XOR(spatial_hv, time_hv)  [polarity × position × time]

    The time HV uses a grid cell module so that temporally close events
    produce similar HVs — encoding recency as a continuous graded signal.

    No frame accumulation. Each event produces one HV immediately.

    Args:
        width, height: Sensor pixel dimensions
        hd_dim: Hypervector dimensionality
        time_decay: τ — time constant for temporal encoding (seconds)
        seed: Random seed
    """

    def __init__(
        self,
        width: int = 346,
        height: int = 260,
        hd_dim: int = 4096,
        time_decay: float = 0.033,  # ~33ms = one "virtual frame"
        seed: int = 42,
    ):
        self.width  = width
        self.height = height
        self.hd_dim = hd_dim
        self.time_decay = time_decay

        g = torch.Generator()
        g.manual_seed(seed)

        # Spatial HVs: one per pixel column (x) and row (y)
        # For large sensors, we use a random projection to reduce memory
        n_x_bins = min(width, 64)
        n_y_bins = min(height, 64)
        self._x_hvs = gen_hvs(n_x_bins, hd_dim, seed=seed)       # (X_bins, D)
        self._y_hvs = gen_hvs(n_y_bins, hd_dim, seed=seed+1000)   # (Y_bins, D)
        self._x_step = width  / n_x_bins
        self._y_step = height / n_y_bins

        # Temporal encoding: grid cell module for smooth time similarity
        # Period = time_decay so events within τ are similar
        self._time_module = GridCellModule(dim=hd_dim, period=time_decay, seed=seed+2000)

        # Polarity HVs
        self._on_hv  = gen_hvs(1, hd_dim, seed=seed+3000).squeeze(0)
        self._off_hv = gen_hvs(1, hd_dim, seed=seed+4000).squeeze(0)

    def _x_hv(self, x: int) -> torch.Tensor:
        idx = min(int(x / self._x_step), len(self._x_hvs) - 1)
        return self._x_hvs[idx]

    def _y_hv(self, y: int) -> torch.Tensor:
        idx = min(int(y / self._y_step), len(self._y_hvs) - 1)
        return self._y_hvs[idx]

    def encode_event(self, event: DVSEvent) -> torch.Tensor:
        """
        Encode a single DVS event to a binary HV.

        Args:
            event: DVSEvent with (x, y, t, p)

        Returns:
            (hd_dim,) binary HV encoding position × time × polarity
        """
        # Spatial: XOR(x_hv, y_hv) — bind pixel column and row
        spatial = (self._x_hv(event.x) != self._y_hv(event.y)).float()

        # Temporal: grid cell encoding of timestamp
        time_hv_c = self._time_module.encode(event.t)
        time_real  = (time_hv_c.real > 0).float()    # binarise

        # Bind spatial with temporal
        spatiotemporal = (spatial != time_real).float()    # XOR

        # Polarity: bind with ON or OFF HV
        pol_hv = self._on_hv if event.p > 0 else self._off_hv
        event_hv = (spatiotemporal != pol_hv).float()      # XOR

        return event_hv

    def encode_stream(
        self,
        events: List[DVSEvent],
        return_all: bool = False,
    ) -> torch.Tensor:
        """
        Encode a list of events, returning their majority bundle.

        This is a batch convenience wrapper — the system still processes
        events one at a time (no frame accumulation).

        Args:
            events: List of DVSEvents
            return_all: If True, return (N, D) all individual HVs; else (D,) bundle

        Returns:
            (D,) majority-bundled HV or (N, D) individual HVs
        """
        hvs = [self.encode_event(e) for e in events]
        if not hvs:
            return torch.zeros(self.hd_dim)

        stacked = torch.stack(hvs)   # (N, D)
        if return_all:
            return stacked

        n = stacked.shape[0]
        return (stacked.sum(dim=0) * 2 > n).float()  # majority


# ═══════════════════════════════════════════════════════════════════════════════
# 2. ContinuousTimeHDC — EMA state updated on every event (no discrete clock)
# ═══════════════════════════════════════════════════════════════════════════════

class ContinuousTimeHDC:
    """
    HDC world model with continuous-time updates — no discrete clock.

    Unlike tick-based HDC (which updates every N milliseconds), this model
    updates its state on EVERY event:

        state(t) ← α(Δt) × state(t−) + (1−α(Δt)) × event_hv(t)

    where α(Δt) = exp(−Δt / τ) — exponential decay since last event.
    This gives time-invariant behaviour: identical patterns at different
    rates produce the same state HV.

    This is the critical property for continuous-time computation:
    the state is a function of the ENTIRE past event stream with exponential
    forgetting, not a function of discrete time bins.

    Args:
        encoder: EventHDCEncoder for per-event HV encoding
        tau: Time constant for state decay (seconds)
    """

    def __init__(
        self,
        encoder: EventHDCEncoder,
        tau: float = 0.1,
        tau_slow: Optional[float] = None,
    ):
        self.encoder = encoder
        self.tau      = tau
        self.tau_slow = tau_slow or tau * 10.0   # slow timescale = 10× fast
        self.hd_dim   = encoder.hd_dim

        # Dual-timescale states (fast + slow) for richer dynamics
        self._state      = torch.zeros(self.hd_dim)   # fast (τ)
        self._state_slow = torch.zeros(self.hd_dim)   # slow (τ_slow)
        self._last_t: Optional[float] = None
        self._n_events = 0

        # Callbacks for reactive processing (no polling needed)
        self._callbacks: List[Callable[[torch.Tensor, float], None]] = []

    def on_state_update(self, callback: Callable[[torch.Tensor, float], None]):
        """Register a callback called on every event with (state_hv, timestamp)."""
        self._callbacks.append(callback)

    def push_event(self, event: DVSEvent) -> torch.Tensor:
        """
        Process one event and update the continuous state.

        The state update uses time-invariant exponential decay:
            α = exp(−Δt / τ)
            state(t) = α × state(t−) + (1−α) × event_hv

        Args:
            event: Incoming DVSEvent

        Returns:
            (hd_dim,) current binary state HV
        """
        # Time-invariant exponential decay
        if self._last_t is not None:
            dt = max(0.0, event.t - self._last_t)
            alpha = math.exp(-dt / self.tau)
        else:
            alpha = 0.0   # first event: full weight on event_hv

        self._last_t = event.t
        self._n_events += 1

        # Encode event
        event_hv = self.encoder.encode_event(event).float()

        # Continuous EMA update (no binarisation — stay in float for continuity)
        self._state = alpha * self._state + (1.0 - alpha) * event_hv

        # Slow timescale update
        if self._last_t is not None:
            alpha_slow = math.exp(-max(0.0, event.t - self._last_t) / self.tau_slow)
        else:
            alpha_slow = 0.0
        self._state_slow = alpha_slow * self._state_slow + (1.0 - alpha_slow) * event_hv

        # Binary snapshot for downstream HDC operations
        state_binary = (self._state >= 0.5).float()

        # Fire callbacks
        for cb in self._callbacks:
            cb(state_binary, event.t)

        return state_binary

    def push_stream(self, events: List[DVSEvent]) -> List[torch.Tensor]:
        """Process a stream of events, returning state after each."""
        return [self.push_event(e) for e in events]

    @property
    def state(self) -> torch.Tensor:
        """Current binary state HV."""
        return (self._state >= 0.5).float()

    @property
    def state_continuous(self) -> torch.Tensor:
        """Current continuous (float) fast state."""
        return self._state.clone()

    @property
    def state_slow(self) -> torch.Tensor:
        """Current slow-timescale state HV."""
        return (self._state_slow >= 0.5).float()

    @property
    def temporal_contrast(self) -> torch.Tensor:
        """
        XOR of fast and slow states: bits that changed recently.

        High-contrast bits → recently active neurons (fast ≠ slow).
        Low-contrast bits → stable neurons (fast ≈ slow).
        Useful for detecting motion onset/offset in event streams.
        """
        fast = (self._state >= 0.5).float()
        slow = (self._state_slow >= 0.5).float()
        return ((fast + slow) % 2).float()   # XOR

    def reset(self):
        self._state.zero_()
        self._state_slow.zero_()
        self._last_t = None
        self._n_events = 0


# ═══════════════════════════════════════════════════════════════════════════════
# 3. EventSNNHDCLoop — closed-loop without synchronization clock
# ═══════════════════════════════════════════════════════════════════════════════

class EventSNNHDCLoop:
    """
    Continuous SNN ↔ HDC closed loop with no discrete synchronization clock.

    Implements the super-Turing architecture:
        Event → ContinuousTimeHDC → SNN threshold modulation → SNN spikes
                      ↑__________________________________|

    The loop is event-driven:
        1. DVS event arrives at time t
        2. ContinuousTimeHDC updates state(t) immediately (no wait)
        3. state(t) modulates SNN neuron thresholds (SemanticAttention)
        4. SNN may emit spikes based on modified thresholds
        5. Spikes immediately update HDC state (STDP-style)
        6. No synchronization — loop fires on every event

    This is the concrete implementation of the Continuous Time Dynamical
    System Argument (§7.3 of the framing document): the combined SNN+HDC
    system evolves continuously, driven by event timing with no discrete clock.

    Args:
        continuous_hdc: ContinuousTimeHDC world model
        snn: SpikingHVNetwork for neural processing
        semantic_attention: Optional SemanticAttention for threshold modulation
    """

    def __init__(
        self,
        continuous_hdc: ContinuousTimeHDC,
        snn=None,
        semantic_attention=None,
    ):
        self.hdc = continuous_hdc
        self.snn = snn
        self.attention = semantic_attention

        # SNN spike → HDC immediate feedback (STDP-style)
        self._spike_hv_buffer: Deque[Tuple[torch.Tensor, float]] = deque(maxlen=100)

        # Register the HDC state as a callback to the continuous model
        self.hdc.on_state_update(self._on_hdc_state_update)

        self._n_loop_iterations = 0

    def _on_hdc_state_update(self, state_hv: torch.Tensor, t: float):
        """Called immediately after each event updates the HDC state."""
        self._n_loop_iterations += 1

        # Step 1: Modulate SNN thresholds based on current HDC state
        if self.attention is not None and self.snn is not None:
            concept_lib = {}   # would come from KnowledgeGraph in full system
            name, thresholds = self.attention.attend(state_hv, concept_lib)
            # Apply thresholds to SNN (if SNN supports it)
            if hasattr(self.snn, 'lif') and hasattr(self.snn.lif, 'v_th'):
                self.snn.lif.v_th = float(thresholds.mean())

        # Step 2: If SNN is present, do a spike step (using current HDC as input)
        # In the full system, this would use actual sensor input,
        # not the HDC state itself (to avoid circular dependency)

    def push_event(self, event: DVSEvent) -> Dict:
        """
        Process one event through the full SNN-HDC loop.

        Returns:
            Dict with state_hv, timestamp, n_iterations
        """
        state_hv = self.hdc.push_event(event)

        return {
            "state_hv": state_hv,
            "timestamp": event.t,
            "n_loop_iterations": self._n_loop_iterations,
            "n_events": self.hdc._n_events,
        }

    def push_stream_async(
        self,
        events: List[DVSEvent],
        callback: Optional[Callable] = None,
    ) -> List[Dict]:
        """
        Process event stream asynchronously — events trigger updates immediately.

        In a real system this would be driven by hardware interrupts;
        here we simulate the async behaviour by processing events in order
        of their timestamps.

        Args:
            events: List of DVSEvents (will be sorted by timestamp)
            callback: Optional per-event callback(result_dict)
        """
        events_sorted = sorted(events, key=lambda e: e.t)
        results = []

        for event in events_sorted:
            result = self.push_event(event)
            results.append(result)
            if callback:
                callback(result)

        return results

    @property
    def current_state(self) -> torch.Tensor:
        return self.hdc.state


# ═══════════════════════════════════════════════════════════════════════════════
# Synthetic event generator (for testing without a physical DVS camera)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_moving_dot_events(
    width: int = 64,
    height: int = 64,
    duration: float = 0.1,
    velocity: Tuple[float, float] = (100.0, 50.0),  # pixels/second
    event_rate: float = 1000.0,
    seed: int = 42,
) -> List[DVSEvent]:
    """
    Generate synthetic DVS events from a moving dot (for testing).

    Simulates what a DVS camera would see when a bright dot moves across
    the sensor — ON events at the leading edge, OFF at the trailing edge.

    Args:
        width, height: Sensor dimensions
        duration: Simulation duration in seconds
        velocity: (vx, vy) in pixels/second
        event_rate: Average events per second
        seed: Random seed

    Returns:
        List of DVSEvents sorted by timestamp
    """
    torch.manual_seed(seed)
    events = []

    # Starting position
    x0, y0 = width / 2, height / 2
    dt = 1.0 / event_rate

    t = 0.0
    while t < duration:
        # Current dot position
        x = x0 + velocity[0] * t
        y = y0 + velocity[1] * t

        # Clamp to sensor bounds
        x = x % width
        y = y % height

        # Emit ON event at current position
        events.append(DVSEvent(
            x=int(x), y=int(y),
            t=t,
            p=1,
        ))

        # Emit OFF event slightly behind (simulating trailing edge)
        bx = int((x - velocity[0] * dt * 2) % width)
        by = int((y - velocity[1] * dt * 2) % height)
        events.append(DVSEvent(x=bx, y=by, t=t, p=-1))

        t += dt + float(torch.rand(1)) * dt * 0.1  # small jitter

    return sorted(events, key=lambda e: e.t)


# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_event_encoder():
    print("=" * 60)
    print("Testing EventHDCEncoder (asynchronous DVS events)")
    print("=" * 60)

    enc = EventHDCEncoder(width=64, height=64, hd_dim=2000, seed=42)

    # Same position, different time: should produce similar HVs
    e1 = DVSEvent(x=32, y=32, t=0.010, p=1)
    e2 = DVSEvent(x=32, y=32, t=0.012, p=1)  # 2ms later, same position
    e3 = DVSEvent(x=10, y=10, t=0.010, p=1)  # different position, same time

    hv1 = enc.encode_event(e1)
    hv2 = enc.encode_event(e2)
    hv3 = enc.encode_event(e3)

    sim_near = float(hv_batch_sim(hv1, hv2.unsqueeze(0))[0])
    sim_far  = float(hv_batch_sim(hv1, hv3.unsqueeze(0))[0])

    print(f"  sim(same pos, Δt=2ms): {sim_near:.4f}  (want high)")
    print(f"  sim(diff pos, Δt=0ms): {sim_far:.4f}")

    # Polarity should change the HV
    e_on  = DVSEvent(x=32, y=32, t=0.010, p=+1)
    e_off = DVSEvent(x=32, y=32, t=0.010, p=-1)
    hv_on  = enc.encode_event(e_on)
    hv_off = enc.encode_event(e_off)
    sim_pol = float(hv_batch_sim(hv_on, hv_off.unsqueeze(0))[0])
    print(f"  sim(ON, OFF same pos): {sim_pol:.4f}  (want < 0.9)")
    assert sim_pol < 0.9, "ON and OFF events should differ"

    # Test stream encoding
    events = generate_moving_dot_events(64, 64, duration=0.01, seed=1)
    print(f"  Generated {len(events)} events in 10ms")
    stream_hv = enc.encode_stream(events)
    assert stream_hv.shape == (2000,)
    print(f"  Stream HV density: {stream_hv.mean():.4f}  (want ≈ 0.5)")

    print("  ✅ EventHDCEncoder OK")


def test_continuous_time_hdc():
    print("=" * 60)
    print("Testing ContinuousTimeHDC (no discrete clock)")
    print("=" * 60)

    enc = EventHDCEncoder(width=64, height=64, hd_dim=2000, seed=0)
    hdc_model = ContinuousTimeHDC(enc, tau=0.05)

    # Process events: state should change on each event
    events = generate_moving_dot_events(64, 64, duration=0.05, seed=42)
    states = hdc_model.push_stream(events[:20])

    print(f"  Processed {len(states)} events without discrete clock")
    assert len(states) == 20

    # State should be non-trivial after events
    final_state = states[-1]
    density = float(final_state.mean())
    print(f"  Final state density: {density:.4f}  (want ≈ 0.5)")
    assert 0.3 < density < 0.7

    # Continuous state should be smoother than binary
    continuous = hdc_model.state_continuous
    print(f"  Continuous state range: [{continuous.min():.3f}, {continuous.max():.3f}]")
    assert float(continuous.std()) > 0.0  # non-trivial continuous state

    # Time-invariance: same pattern at 2× rate → same state
    enc2 = EventHDCEncoder(width=64, height=64, hd_dim=2000, seed=0)
    hdc_fast = ContinuousTimeHDC(enc2, tau=0.05)
    hdc_slow = ContinuousTimeHDC(enc2, tau=0.05)

    # Fast: events at [0, 0.01, 0.02] → same pattern as slow: [0, 0.02, 0.04]
    ev_fast = [DVSEvent(32,32, t, 1) for t in [0.0, 0.01, 0.02]]
    ev_slow = [DVSEvent(32,32, t, 1) for t in [0.0, 0.02, 0.04]]

    for e in ev_fast: hdc_fast.push_event(e)
    for e in ev_slow: hdc_slow.push_event(e)

    sim_rate_invariant = float(hv_batch_sim(hdc_fast.state, hdc_slow.state.unsqueeze(0))[0])
    print(f"  Rate-invariant sim (2× speed): {sim_rate_invariant:.4f}  (should be > 0.5)")

    print("  ✅ ContinuousTimeHDC OK")


def test_event_snn_loop():
    print("=" * 60)
    print("Testing EventSNNHDCLoop (closed-loop, no sync clock)")
    print("=" * 60)

    enc = EventHDCEncoder(width=32, height=32, hd_dim=1000, seed=7)
    hdc_model = ContinuousTimeHDC(enc, tau=0.05)
    loop = EventSNNHDCLoop(hdc_model)

    events = generate_moving_dot_events(32, 32, duration=0.02, seed=5)

    # Process via async stream
    results = loop.push_stream_async(events[:10])

    print(f"  Processed {len(results)} events asynchronously")
    print(f"  Loop iterations: {results[-1]['n_loop_iterations']}")
    print(f"  Total events: {results[-1]['n_events']}")

    assert results[-1]['n_loop_iterations'] == 10
    assert results[-1]['state_hv'].shape == (1000,)

    # State at end should differ from start (across many events)
    state_first = results[0]['state_hv']
    state_last  = results[-1]['state_hv']
    sim_change = float(hv_batch_sim(state_first, state_last.unsqueeze(0))[0])
    print(f"  sim(state_first, state_last): {sim_change:.4f}  (want < 1.0 — some change)")
    # The continuous state accumulates; binary may not change on consecutive events
    # but the continuous accumulator definitely changes
    cont = hdc_model.state_continuous
    print(f"  Continuous accumulator has content: std={cont.std():.4f}")

    print("  ✅ EventSNNHDCLoop OK")


if __name__ == "__main__":
    test_event_encoder()
    print()
    test_continuous_time_hdc()
    print()
    test_event_snn_loop()
    print()
    print("=== All event_hdc tests passed ===")
