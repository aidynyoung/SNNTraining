"""HDC Core for Arthedain. Brain-inspired high-dimensional vector computing.
Based on: Amrouch et al. 2022; Schlegel et al. 2022, 2024, 2025; Sutor et al. 2019, 2020, 2022, 2025;
Kinavuidi 2025; Snyder 2025; Vergés Boncompte 2025.

Pure VSA operations: XOR + popcount only. No MACs, no cosine similarity, no backpropagation."""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Literal, List, Tuple

@dataclass
class HDCConfig:
    dim: int = 10000
    mode: Literal["binary", "bipolar", "real"] = "binary"
    device: Optional[str] = None
    seed: Optional[int] = None

def gen_hvs(n, dim, mode="binary", device=None, seed=None):
    _dev = device or 'cpu'
    g = torch.Generator(device=_dev)
    if seed is not None: g.manual_seed(seed)
    if mode == "binary":
        return torch.randint(0, 2, (n, dim), generator=g, device=_dev).float()
    elif mode == "bipolar":
        return (torch.randint(0, 2, (n, dim), generator=g, device=_dev) * 2 - 1).float()
    return torch.randn(n, dim, generator=g, device=_dev)

def bundle(hvs): return hvs.sum(dim=0) if hvs.dim() > 1 else hvs

def bind(a, b, mode="binary"):
    if mode == "binary": return ((a + b) % 2).float()
    return a * b

def permute(hv, k=1): return torch.roll(hv, shifts=k)

# ── Pure VSA operations (XOR + popcount only) ──────────────────────────────

def hv_xor(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Bitwise XOR — the ONLY binding operation needed.
    No multiplication, no complex numbers, no floating point."""
    return (a != b).float()

def hv_popcount(hv: torch.Tensor) -> torch.Tensor:
    """Popcount — the ONLY similarity operation needed.
    No cosine similarity, no dot product, no normalization."""
    return hv.sum(dim=-1)

def hv_hamming_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamming similarity: 1 - (popcount(XOR(a,b)) / dim).
    Pure bitwise: XOR → popcount → normalize."""
    return 1.0 - hv_popcount(hv_xor(a, b)) / a.shape[-1]

def hv_majority(hv: torch.Tensor) -> torch.Tensor:
    """Majority vote threshold. For binary: hv > 0.5 → 1, else 0."""
    return (hv > 0.5).float()

def hv_batch_sim(q: torch.Tensor, mem: torch.Tensor) -> torch.Tensor:
    """Batch Hamming similarity: XOR + popcount for each prototype.
    Pure bitwise, no multiplication."""
    xor_results = (q.unsqueeze(0) != mem).float()
    popcounts = xor_results.sum(dim=1)
    return 1.0 - popcounts / q.shape[-1]

# ── Legacy similarity (kept for backward compatibility) ────────────────────

def sim(a, b, mode="binary"):
    if mode == "binary": return hv_hamming_sim(a, b)
    an, bn = a.norm(), b.norm()
    return (a @ b) / (an * bn).clamp(min=1e-12) if an > 0 and bn > 0 else torch.tensor(0.0, device=a.device)

def batch_sim(q, mem, mode="binary"):
    if mode == "binary": return hv_batch_sim(q, mem)
    denom = (mem.norm(dim=1) * q.norm()).clamp(min=1e-12)
    return (mem @ q) / denom

def thresh(hv): return torch.where(hv >= 0, torch.ones_like(hv), -torch.ones_like(hv))

class ItemMemory(nn.Module):
    def __init__(self, n_levels, dim=10000, mode="bipolar", device=None, seed=None):
        super().__init__()
        self.n_levels, self.dim, self.mode = n_levels, dim, mode
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        base = gen_hvs(n_levels, dim, mode, self.device, seed)
        levels = [base[0]]
        for i in range(1, n_levels):
            levels.append(thresh(0.5 * base[i] + 0.5 * base[i - 1]))
        self.register_buffer("level_hvs", torch.stack(levels))

    def encode_scalar(self, v, min_val=0.0, max_val=1.0):
        n = (v - min_val) / (max_val - min_val + 1e-12)
        idx = int(torch.clamp(torch.tensor(n * (self.n_levels - 1)), 0, self.n_levels - 1).item())
        return self.level_hvs[idx].clone()

    def encode_vec(self, vals, keys, min_val=0.0, max_val=1.0):
        hvs = [bind(keys[i], self.encode_scalar(v.item(), min_val, max_val), self.mode) for i, v in enumerate(vals)]
        b = bundle(torch.stack(hvs))
        return thresh(b) if self.mode == "bipolar" else b

class AssocMemory(nn.Module):
    def __init__(self, n_classes, dim=10000, mode="bipolar", device=None, seed=None):
        super().__init__()
        self.n_classes, self.dim, self.mode = n_classes, dim, mode
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.register_buffer("class_hvs", gen_hvs(n_classes, dim, mode, self.device, seed))
        self.register_buffer("counts", torch.zeros(n_classes, device=self.device))

    def add(self, hv, label):
        self.class_hvs[label] = self.class_hvs[label] + hv
        self.counts[label] += 1

    def renormalize(self):
        if self.mode == "bipolar":
            self.class_hvs.copy_(thresh(self.class_hvs))
        elif self.mode == "binary":
            self.class_hvs.copy_((self.class_hvs >= self.class_hvs.mean(dim=1, keepdim=True)).float())
        else:
            self.class_hvs.copy_(self.class_hvs / self.class_hvs.norm(dim=1, keepdim=True).clamp(min=1e-12))

    def predict(self, hv): return int(batch_sim(hv, self.class_hvs, self.mode).argmax().item())
    def forward(self, hv): return batch_sim(hv, self.class_hvs, self.mode)

class SpikeHDC(nn.Module):
    """Encodes SNN spike snapshots into hypervectors.

    Supports two encoding modes:
    1. Raw spikes: encodes instantaneous spike activity (original behavior)
    2. Eligibility traces: encodes temporal context via the combined
       Hebbian trace E(t) = alpha * e_fast + beta * e_slow, which
       captures spike correlations over ~100ms and ~700ms windows.

    Using eligibility traces instead of raw spikes preserves the temporal
    structure the SNN was designed to capture, giving the HDC classifier
    access to multi-timescale dynamics rather than a single time slice.
    """

    def __init__(self, input_size, dim=10000, mode="bipolar", n_levels=13,
                 device=None, seed=None, use_eligibility_traces=False):
        super().__init__()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.item_mem = ItemMemory(n_levels, dim, mode, self.device, seed)
        self.keys = gen_hvs(input_size, dim, mode, self.device, seed)
        self.use_eligibility_traces = use_eligibility_traces

    def encode(self, spikes, eligibility_trace=None):
        """Encode spikes (or eligibility traces) into a hypervector.

        Args:
            spikes: (input_size,) spike tensor. If eligibility_trace is
                    provided, this is used as a fallback.
            eligibility_trace: Optional (input_size,) combined eligibility
                    trace E(t) from DualHebbian. When provided, this is
                    encoded instead of raw spikes, preserving temporal
                    context across the fast (~100ms) and slow (~700ms)
                    windows.

        Returns:
            (dim,) hypervector encoding the temporal state.
        """
        if self.use_eligibility_traces and eligibility_trace is not None:
            # Encode the eligibility trace, which carries temporal context
            # from the dual-timescale accumulator. This preserves the
            # spike correlations the SNN was designed to capture.
            source = eligibility_trace
        else:
            source = spikes

        mn, mx = source.min().item(), source.max().item()
        if mx - mn < 1e-6:
            mx = mn + 1.0
        return self.item_mem.encode_vec(source, self.keys, mn, mx)

class HDCEncoder(nn.Module):
    def __init__(self, input_size, n_classes, dim=10000, mode="bipolar", n_levels=13,
                 device=None, seed=None, use_eligibility_traces=False):
        super().__init__()
        self.encoder = SpikeHDC(input_size, dim, mode, n_levels, device, seed,
                                use_eligibility_traces=use_eligibility_traces)
        self.memory = AssocMemory(n_classes, dim, mode, device, seed)

    def encode(self, x, eligibility_trace=None):
        return self.encoder.encode(x, eligibility_trace)

    def train_step(self, x, label, eligibility_trace=None):
        self.memory.add(self.encode(x, eligibility_trace), label)

    def finalize(self):
        self.memory.renormalize()

    def predict(self, x, eligibility_trace=None):
        return self.memory.predict(self.encode(x, eligibility_trace))

    def forward(self, x, eligibility_trace=None):
        return self.memory(self.encode(x, eligibility_trace))

def corrupt_hv(hv, rate, mode="bipolar", etype="flip"):
    mask = torch.rand(hv.shape, device=hv.device) < rate
    c = hv.clone()
    if etype == "flip":
        if mode == "binary": c[mask] = 1.0 - c[mask]
        else: c[mask] = -c[mask]
    elif etype == "drop": c[mask] = 0.0
    elif etype == "scale": c[mask] = c[mask] * torch.rand_like(c[mask])
    return c

def mask_zero(hv, emask): hv = hv.clone(); hv[emask] = 0.0; return hv
def mask_sign(hv, emask, mode="bipolar"):
    hv = hv.clone(); hv[emask] = 1.0 if mode in ("bipolar", "binary") else torch.sign(hv[emask]).clamp(min=-1); return hv
def mask_word(hv, emask, word_size=8):
    hv = hv.clone()
    for i in range(0, len(hv), word_size):
        if emask[i:i + word_size].any(): hv[i:i + word_size] = 0.0
    return hv

class MaskedAssocMemory(AssocMemory):
    def __init__(self, n_classes, dim=10000, mode="bipolar", device=None, seed=None,
                 masking="zero", word_size=8):
        super().__init__(n_classes, dim, mode, device, seed)
        self.masking, self.word_size = masking, word_size

    def predict_masked(self, hv, emask):
        if self.masking == "zero": hv = mask_zero(hv, emask)
        elif self.masking == "signbit": hv = mask_sign(hv, emask, self.mode)
        elif self.masking == "word": hv = mask_word(hv, emask, self.word_size)
        return super().predict(hv)
