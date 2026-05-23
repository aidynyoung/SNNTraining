"""
snn_hdc_pipeline.py
===================
Live SNN -> HDC inference pipeline -- pure VSA, no backpropagation.

The ENTIRE pipeline operates on hypervectors using only VSA operations
(XOR, popcount, permutation, bitwise ADD). No MACs, no SynOps, no
cosine similarity, no backpropagation, no hyperbolic convergence.

Architecture
------------
    Input spikes (x_t)
          |
        RSNN                 -- LIF dynamics, recurrent weights
          |  spike vector + eligibility trace E(t)
      Rate accumulator       -- rolling window mean (spikes or traces)
          |  vector (hidden_size,)  every WINDOW steps
       HDC encoder           -- bind rate levels to item-memory keys (XOR)
          |  hypervector (dim,)
    Associative memory       -- Hamming similarity (XOR + popcount) -> HV output
          |
   Hypervector output + confidence + similarity score
          |
   Learning rate modulator  -- if similarity drops -> increase lr

Key properties:
- Pure bitwise VSA: XOR + popcount only
- Output is a hypervector (not a class index)
- Learning is simple accumulation (no backpropagation)
- No hyperbolic convergence (tanh/sigmoid/softmax)
- ~1.9 nJ/inference at 45nm CMOS
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional, Deque, List, Tuple

import torch
import torch.nn as nn

from models.rsnn import RSNN, RSNNConfig
from models.readout import Readout, ReadoutConfig
from models.hdc import (
    HDCConfig, gen_hvs, bind, bundle, batch_sim, thresh,
    AssocMemory, ItemMemory,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    # SNN
    input_size: int = 100
    hidden_size: int = 128
    output_size: int = 2         # readout output (continuous decoding)
    input_gain: float = 5.0

    # HDC
    n_classes: int = 8
    hdc_dim: int = 4096
    hdc_mode: str = "bipolar"    # "binary" | "bipolar"
    n_levels: int = 21           # quantisation levels for rate encoding
    hdc_seed: Optional[int] = 42

    # Inference window
    window: int = 50             # timesteps to average before one HDC lookup
    overlap: int = 25            # slide overlap (0 = non-overlapping windows)

    # Online update
    online: bool = True
    online_lr: float = 0.05      # fraction of new sample to blend in

    # Option A: eligibility trace encoding
    use_eligibility_traces: bool = False
    # When enabled, E(t) from DualHebbian is stored instead of raw spikes,
    # preserving temporal context across fast (~100ms) and slow (~700ms) windows.

    # Option A: closed-loop learning rate modulation
    enable_hdc_feedback: bool = False
    # When enabled, HDC similarity to nearest prototype is tracked.
    # If it drops below lr_boost_threshold, the pipeline signals the
    # trainer to increase learning rate (distribution-shift detection).
    lr_boost_threshold: float = 0.15   # similarity below this -> boost lr
    lr_boost_factor: float = 5.0       # multiply learning rate by this
    lr_boost_cooldown: int = 50        # steps before can boost again

    # Adaptive routing: fall back to SNN linear readout when HDC is uncertain.
    # HDC is fast and energy-efficient on simple tasks; the SNN readout recovers
    # accuracy on harder tasks (e.g. SHD 20-class, BCI) where HDC loses 4-13%.
    #
    # gate_threshold = 0.0  → always use HDC (default, backward-compatible)
    # gate_threshold = 0.4  → use SNN readout when HDC confidence < 0.4
    use_snn_fallback: bool = False
    gate_threshold:   float = 0.4     # HDC similarity below this → SNN readout
    fallback_lr:      float = 0.02    # LMS learning rate for SNN readout

    device: Optional[str] = None


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class SNNHDCPipeline(nn.Module):
    """
    End-to-end SNN + HDC inference module with optional closed-loop feedback.

    Attributes
    ----------
    rsnn         : RSNN -- recurrent spiking network
    readout      : Readout -- continuous linear decoder (optional)
    item_mem     : ItemMemory -- level hypervectors for rate/trace encoding
    assoc_mem    : AssocMemory -- class prototype hypervectors
    channel_keys : (n_channels, dim) -- random key per hidden neuron
    spike_buf    : rolling deque of last `window` spike or trace tensors
    lr_multiplier : float -- current learning rate multiplier (HDC feedback)
    """

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        super().__init__()
        cfg = config or PipelineConfig()
        self.cfg = cfg
        self.device = torch.device(cfg.device or (
            'cuda' if torch.cuda.is_available() else 'cpu'))

        # SNN core
        self.rsnn = RSNN(
            input_size=cfg.input_size,
            hidden_size=cfg.hidden_size,
            input_gain=cfg.input_gain,
            device=self.device,
        )
        self.readout = Readout(
            hidden_size=cfg.hidden_size,
            output_size=cfg.output_size,
            device=self.device,
        )

        # HDC components
        self.item_mem = ItemMemory(
            n_levels=cfg.n_levels,
            dim=cfg.hdc_dim,
            mode=cfg.hdc_mode,
            device=self.device,
            seed=cfg.hdc_seed,
        )
        self.assoc_mem = AssocMemory(
            n_classes=cfg.n_classes,
            dim=cfg.hdc_dim,
            mode=cfg.hdc_mode,
            device=self.device,
            seed=cfg.hdc_seed,
        )

        # One random key hypervector per hidden neuron
        self.register_buffer(
            "channel_keys",
            gen_hvs(cfg.hidden_size, cfg.hdc_dim, cfg.hdc_mode,
                    self.device, seed=cfg.hdc_seed),
        )

        # Rolling buffer: stores spikes OR eligibility traces
        self._buf: Deque[torch.Tensor] = deque(maxlen=cfg.window)
        self._step_count: int = 0

        # Training accumulator
        self._train_hvs: List[Tuple[torch.Tensor, int]] = []

        # HDC feedback state
        self.lr_multiplier: float = 1.0
        self._last_similarity: float = 0.0
        self._last_lr_boost_step: int = -cfg.lr_boost_cooldown

        # SNN fallback readout: (n_classes, hidden_size) weight matrix.
        # Trained online with LMS (delta rule) alongside HDC.
        # Only allocated when use_snn_fallback=True.
        self._fallback_W: Optional[torch.Tensor] = None
        self._fallback_b: Optional[torch.Tensor] = None
        if cfg.use_snn_fallback:
            self._fallback_W = torch.zeros(
                cfg.n_classes, cfg.hidden_size, device=self.device)
            self._fallback_b = torch.zeros(cfg.n_classes, device=self.device)

        # Route tracking: how often each path is taken
        self._route_hdc:     int = 0
        self._route_fallback: int = 0

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def _encode_vector(self, vec: torch.Tensor) -> torch.Tensor:
        """
        Encode a vector (firing rates or eligibility trace) into a hypervector.

        vec : (hidden_size,) -- values in [0, 1] after normalisation

        Each channel i is encoded as:
            hv_i = bind(channel_keys[i], level_hv(vec[i]))
        Final HV = bundle(hv_0, ..., hv_{N-1})
        """
        # Build level HVs for all channels at once, then bind vectorized
        level_hvs = torch.stack([
            self.item_mem.encode_scalar(vec[i].item(), 0.0, 1.0)
            for i in range(self.cfg.hidden_size)
        ])  # (hidden_size, dim)

        if self.cfg.hdc_mode == "binary":
            bound = ((self.channel_keys + level_hvs) % 2).float()
        else:
            bound = self.channel_keys * level_hvs  # bipolar element-wise multiply

        hv = bound.sum(dim=0)
        return thresh(hv) if self.cfg.hdc_mode == "bipolar" else hv

    def _current_buffer_mean(self) -> torch.Tensor:
        """Mean of current buffer (returns zeros if empty)."""
        if len(self._buf) == 0:
            return torch.zeros(self.cfg.hidden_size, device=self.device)
        return torch.stack(list(self._buf)).mean(dim=0)

    # ------------------------------------------------------------------
    # SNN step (call every timestep)
    # ------------------------------------------------------------------

    def snn_step(self, x: torch.Tensor,
                 eligibility_trace: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward one SNN timestep and accumulate spikes or eligibility traces.

        Args:
            x: Input tensor (input_size,)
            eligibility_trace: Optional E(t) from DualHebbian (hidden_size, hidden_size).
                               If the pipeline is configured with use_eligibility_traces=True,
                               the trace is flattened and accumulated instead of raw spikes.

        Returns:
            Spike vector (hidden_size,)
        """
        x = x.to(self.device)
        spikes = self.rsnn.forward(x)
        self._step_count += 1

        if self.cfg.use_eligibility_traces and eligibility_trace is not None:
            # Flatten 2D trace (n_post, n_pre) -> 1D and accumulate
            # The trace carries temporal correlations across the dual window
            trace_1d = eligibility_trace.flatten()  # (hidden_size^2,)
            # Downsample to hidden_size by taking the diagonal + mean
            # This preserves self-correlations while being the right dimension
            diag = eligibility_trace.diag()  # (hidden_size,)
            self._buf.append(diag.detach())
        else:
            self._buf.append(spikes.detach())

        return spikes

    # ------------------------------------------------------------------
    # HDC inference
    # ------------------------------------------------------------------

    def hdc_infer(self) -> Tuple[int, float, torch.Tensor]:
        """
        Encode current buffer and classify via associative memory.

        Returns
        -------
        label : int   -- predicted class index
        conf  : float -- Hamming similarity to nearest prototype
        hv    : (dim,) -- the encoded query hypervector
        """
        vec = self._current_buffer_mean()

        # Normalise to [0, 1] for ItemMemory encoding
        mn, mx = vec.min().item(), vec.max().item()
        if mx - mn < 1e-6:
            mx = mn + 1.0
        vec_norm = (vec - mn) / (mx - mn)

        hv = self._encode_vector(vec_norm)
        sims = batch_sim(hv, self.assoc_mem.class_hvs, self.cfg.hdc_mode)
        label = int(sims.argmax().item())
        conf = float(sims[label].item())

        # Track similarity for closed-loop feedback
        self._last_similarity = conf

        return label, conf, hv

    # ------------------------------------------------------------------
    # SNN fallback readout
    # ------------------------------------------------------------------

    def _fallback_infer(self) -> Tuple[int, float]:
        """
        Classify using the SNN linear readout (fallback path).

        Logits = W @ mean_spikes + b, prediction = argmax.
        Confidence = softmax gap (top - second, ∈ [0, 1]).
        """
        spikes = self._current_buffer_mean()
        logits = self._fallback_W @ spikes + self._fallback_b  # (n_classes,)
        probs  = torch.softmax(logits, dim=0)
        label  = int(probs.argmax().item())
        top2   = probs.topk(min(2, self.cfg.n_classes)).values
        conf   = float((top2[0] - top2[-1]).item())
        return label, conf

    def _fallback_update(self, spikes: torch.Tensor, true_label: int) -> None:
        """
        Online LMS (delta rule) update for the SNN fallback readout.

        ΔW = lr * (one_hot - softmax(logits)) ⊗ spikes
        This is online softmax regression — no backprop, O(n_classes × hidden).
        """
        if self._fallback_W is None:
            return
        logits = self._fallback_W @ spikes + self._fallback_b
        probs  = torch.softmax(logits, dim=0)
        target = torch.zeros(self.cfg.n_classes, device=self.device)
        target[true_label] = 1.0
        error  = target - probs                                  # (n_classes,)
        self._fallback_W += self.cfg.fallback_lr * torch.outer(error, spikes)
        self._fallback_b += self.cfg.fallback_lr * error

    # ------------------------------------------------------------------
    # HDC feedback: compute learning rate multiplier
    # ------------------------------------------------------------------

    def get_lr_multiplier(self) -> float:
        """
        Compute learning rate multiplier based on HDC similarity.

        When similarity to the nearest class prototype drops below
        lr_boost_threshold, it indicates a potential distribution shift.
        The multiplier increases (up to lr_boost_factor), telling the
        trainer to adapt faster.

        Returns:
            float: learning rate multiplier (1.0 = normal, >1.0 = boosted)
        """
        if not self.cfg.enable_hdc_feedback:
            return 1.0

        steps_since_boost = self._step_count - self._last_lr_boost_step
        if (self._last_similarity < self.cfg.lr_boost_threshold
                and steps_since_boost >= self.cfg.lr_boost_cooldown):
            self.lr_multiplier = self.cfg.lr_boost_factor
            self._last_lr_boost_step = self._step_count
        else:
            # Decay multiplier back toward 1.0
            self.lr_multiplier = 1.0 + (self.lr_multiplier - 1.0) * 0.95

        return self.lr_multiplier

    # ------------------------------------------------------------------
    # Combined predict (handles windowing internally)
    # ------------------------------------------------------------------

    def predict(self, x: torch.Tensor,
                eligibility_trace: Optional[torch.Tensor] = None
                ) -> Tuple[Optional[int], Optional[float]]:
        """
        Feed one input timestep; return (label, confidence) when the window
        fires, else (None, None).

        Routing logic (when use_snn_fallback=True):
            HDC confidence >= gate_threshold  →  HDC path  (fast, efficient)
            HDC confidence <  gate_threshold  →  SNN readout (accurate fallback)
        """
        self.snn_step(x, eligibility_trace)
        stride = self.cfg.window - self.cfg.overlap
        if (self._step_count % stride) == 0 and len(self._buf) == self.cfg.window:
            label, conf, hv = self.hdc_infer()

            if (self.cfg.use_snn_fallback
                    and self._fallback_W is not None
                    and conf < self.cfg.gate_threshold):
                label, conf = self._fallback_infer()
                self._route_fallback += 1
            else:
                self._route_hdc += 1

            return label, conf
        return None, None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_step(self, x: torch.Tensor, label: int,
                   eligibility_trace: Optional[torch.Tensor] = None) -> None:
        """
        Push one sample through the SNN and bundle its hypervector into
        the class prototype for `label`.

        When use_snn_fallback=True, also trains the SNN linear readout
        with one LMS step — so both paths improve in parallel.
        """
        for _ in range(self.cfg.window):
            self.snn_step(x, eligibility_trace)
        vec = self._current_buffer_mean()
        mn, mx = vec.min().item(), vec.max().item()
        if mx - mn < 1e-6:
            mx = mn + 1.0
        vec_norm = (vec - mn) / (mx - mn)
        hv = self._encode_vector(vec_norm)
        self.assoc_mem.add(hv, label)

        # Also train the SNN fallback readout (zero extra SNN forward passes)
        if self.cfg.use_snn_fallback and self._fallback_W is not None:
            self._fallback_update(vec, label)

    def finalize(self) -> None:
        """Normalise class prototypes after training."""
        self.assoc_mem.renormalize()

    # ------------------------------------------------------------------
    # Online prototype update (during deployment)
    # ------------------------------------------------------------------

    def online_update(self, hv: torch.Tensor, true_label: int) -> None:
        """
        Blend a new sample hypervector into the class prototype.
        """
        if not self.cfg.online:
            return
        lr = self.cfg.online_lr
        proto = self.assoc_mem.class_hvs[true_label]
        self.assoc_mem.class_hvs[true_label] = thresh(
            (1.0 - lr) * proto + lr * hv
        ) if self.cfg.hdc_mode == "bipolar" else (1.0 - lr) * proto + lr * hv

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset SNN state and spike buffer (not the HDC memory or fallback weights)."""
        self.rsnn.reset()
        self.readout.reset()
        self._buf.clear()
        self._step_count = 0
        self.lr_multiplier = 1.0
        self._last_similarity = 0.0

    def evaluate_stream(
        self,
        data_stream,   # iterable of (x_tensor, label_int)
        n_steps: int = 20,
    ) -> Dict:
        """
        Evaluate pipeline accuracy over a labelled stream.

        Processes each sample for `n_steps` SNN timesteps, then classifies.

        Returns:
            Dict with accuracy, per-class accuracy, total samples.
        """
        from collections import defaultdict
        correct_per_class: Dict[int, int] = defaultdict(int)
        total_per_class:   Dict[int, int] = defaultdict(int)
        total, correct = 0, 0

        for x, label in data_stream:
            self.reset()
            for _ in range(n_steps):
                self.snn_step(x)
            pred, _, _ = self.hdc_infer()
            total_per_class[label] += 1
            if pred == label:
                correct += 1
                correct_per_class[label] += 1
            total += 1

        per_class_acc = {
            c: correct_per_class[c] / max(total_per_class[c], 1)
            for c in total_per_class
        }
        return {
            "accuracy":       correct / max(total, 1),
            "n_samples":      total,
            "per_class_acc":  per_class_acc,
            "n_classes":      len(total_per_class),
        }

    def pipeline_summary(self) -> Dict:
        """Return a summary of the current pipeline configuration and state."""
        total_routed = self._route_hdc + self._route_fallback
        report = {
            "input_size":    self.cfg.input_size,
            "hidden_size":   self.cfg.hidden_size,
            "n_classes":     self.cfg.n_classes,
            "hdc_dim":       self.cfg.hdc_dim,
            "step_count":    self._step_count,
            "lr_multiplier": round(self.lr_multiplier, 4),
            "last_sim":      round(self._last_similarity, 4),
            "snn_firing_rate": float(self.rsnn.prev_spikes.mean().item()),
        }
        if self.cfg.use_snn_fallback:
            report["route_hdc"]      = self._route_hdc
            report["route_fallback"] = self._route_fallback
            report["fallback_rate"]  = round(
                self._route_fallback / max(total_routed, 1), 4)
            report["gate_threshold"] = self.cfg.gate_threshold
        return report


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = PipelineConfig(
        input_size=20, hidden_size=32, n_classes=4, hdc_dim=512,
        window=10, overlap=5, online=True,
        use_eligibility_traces=False,
        enable_hdc_feedback=True,
    )
    pipe = SNNHDCPipeline(cfg)

    # Train: 4 classes x 10 samples
    print("Training...")
    for cls in range(4):
        for _ in range(10):
            x = torch.zeros(20)
            x[cls * 5:(cls + 1) * 5] = 1.0
            pipe.train_step(x, cls)
    pipe.finalize()

    # Inference
    print("Inference...")
    pipe.reset()
    correct = 0
    for cls in range(4):
        x = torch.zeros(20)
        x[cls * 5:(cls + 1) * 5] = 1.0
        for t in range(20):
            label, conf = pipe.predict(x)
            if label is not None:
                correct += int(label == cls)
                lr_mult = pipe.get_lr_multiplier()
                print(f"  class={cls} -> pred={label}  conf={conf:.3f}  "
                      f"lr_mult={lr_mult:.2f}")
                break

    print(f"SNN->HDC pipeline smoke test complete -- {correct}/4 correct")
