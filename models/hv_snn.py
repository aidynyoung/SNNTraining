"""
models/hv_snn.py
=================
Spiking Neural Network where each spike is a hypervector.

The insight
-----------
Standard SNN:  neuron i fires  →  z_i = 1  (scalar)
HV-SNN:        neuron i fires  →  emits basis_i  (D-dim hypervector)

Network state = bundle of all firing neurons' basis HVs.

This collapses the network state from (N,) binary scalars to a single (D,)
hypervector at every timestep — regardless of N.  The state is natively
composable with the rest of the HyperVector Architecture.

Two levels of integration
--------------------------
SpikingHVLayer       — LIF dynamics unchanged; spikes map to HV space after firing.
                       Drop-in replacement for LIFLayer in any RSNN.
                       Returns (spikes, state_hv) every step.

SpikingHVNetwork     — Full HV-SNN.  Recurrent connections are replaced by
                       Hamming similarity between the current state HV and each
                       neuron's receptor HV.  No W_rec matrix.  No backprop.
                       O(N·D) per step vs O(N²) for a dense W_rec.

Why this matters
----------------
- O(D) state instead of O(N×T) for temporal processing
- No recurrent weight matrix for SpikingHVNetwork — HDC IS the recurrence
- State is already a hypervector → direct input to HVPipeline as a modality
- Temporal sequences encoded via permutation (n-gram in time)
- Online learning: RefineHD on state HVs, not eligibility traces

Connection to prior art
------------------------
- Mitrokhin, Sutor (2019) HAP: population of spike events → HV (spatial)
  Here: population of neurons → HV (identity-based, at each timestep)
- Karunaratne (2020) in-memory HDC: basis HVs for symbols; same principle
- Teeters (2023): CleanupMemory decodes the state HV back to symbols
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn

from models.lif import LIFLayer, LIFConfig


# ── Helpers ───────────────────────────────────────────────────────────────────

def _gen_basis(n: int, dim: int, seed: Optional[int] = None, device: str = "cpu") -> torch.Tensor:
    """Generate N random binary basis hypervectors (N, D)."""
    g = torch.Generator(device=device)
    if seed is not None:
        g.manual_seed(seed)
    return (torch.rand(n, dim, generator=g, device=device) >= 0.5).float()


def _bundle_active(basis: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor:
    """
    Bundle basis HVs for all active (firing) neurons.

    Args:
        basis:       (N, D) — basis hypervectors for all neurons
        active_mask: (N,)  bool — which neurons fired

    Returns:
        (D,) majority-vote bundle of active basis HVs,
        or zeros if no neuron fired.
    """
    if not active_mask.any():
        return torch.zeros(basis.shape[1], device=basis.device)
    active = basis[active_mask]          # (k, D)
    return (active.float().mean(dim=0) >= 0.5).float()


def _hamming_sim_vec(query: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    """
    Hamming similarity between a single HV and a matrix of HVs.

    Args:
        query:  (D,)   — query hypervector
        matrix: (N, D) — N hypervectors to compare against

    Returns:
        (N,) similarities ∈ [0, 1]
    """
    q = (query > 0.5).float()
    m = (matrix > 0.5).float()
    xor = (m != q.unsqueeze(0)).float()   # (N, D)
    return 1.0 - xor.sum(dim=1) / matrix.shape[1]


# ── SpikingHVLayer ─────────────────────────────────────────────────────────────

@dataclass
class SpikingHVConfig:
    n_neurons: int = 128
    hv_dim: int = 4096
    # LIF parameters
    tau: float = 20.0
    v_th: float = 1.0
    v_reset: float = 0.0
    refractory: int = 2
    dt: float = 1.0
    # Threshold adaptation (Zhao 2026)
    enable_threshold_adaptation: bool = False
    threshold_adaptation_rate: float = 0.01
    # Temporal HV accumulation
    temporal_window: int = 10     # steps to accumulate temporal HV
    seed: Optional[int] = None
    device: str = "cpu"


class SpikingHVLayer(nn.Module):
    """
    LIF neurons where each spike is a D-dim hypervector.

    Standard spiking:  z_i(t) = 1  (scalar, sent along weight i)
    HV spiking:        z_i(t) = 1  →  contributes basis_i to state_hv(t)

    The state HV at timestep t is the majority-vote bundle of all firing
    neurons' basis HVs.  It summarises WHO fired, not just how many.

    Temporal accumulation: state_hvs from the last `temporal_window` steps
    are n-gram encoded (permute + XOR) to form a temporal sequence HV that
    captures recent firing history in O(D) space.

    Example::

        layer = SpikingHVLayer(SpikingHVConfig(n_neurons=128, hv_dim=4096))
        for t in range(T):
            current = W_in @ x_t
            spikes, state_hv, seq_hv = layer.step(current)
            # spikes:   (128,) binary — standard LIF output
            # state_hv: (4096,) binary — who fired this step
            # seq_hv:   (4096,) binary — recent firing history
    """

    def __init__(self, config: Optional[SpikingHVConfig] = None, **kwargs):
        super().__init__()
        cfg = config or SpikingHVConfig(**kwargs)
        self.cfg = cfg

        # Standard LIF dynamics (unchanged)
        self.lif = LIFLayer(
            size=cfg.n_neurons,
            tau=cfg.tau,
            v_th=cfg.v_th,
            v_reset=cfg.v_reset,
            refractory=cfg.refractory,
            dt=cfg.dt,
            device=cfg.device,
            enable_threshold_adaptation=cfg.enable_threshold_adaptation,
            threshold_adaptation_rate=cfg.threshold_adaptation_rate,
        )

        # Random basis HV for each neuron — fixed, not trained
        basis = _gen_basis(cfg.n_neurons, cfg.hv_dim, cfg.seed, cfg.device)
        self.register_buffer("basis", basis)

        # Temporal accumulation state
        self._seq_hv = torch.zeros(cfg.hv_dim, device=cfg.device)
        self._step = 0

    def step(
        self, input_current: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        One timestep: LIF → spike → HV emission → temporal accumulation.

        Args:
            input_current: (n_neurons,) input current

        Returns:
            spikes:   (n_neurons,) binary — standard LIF output (unchanged)
            state_hv: (hv_dim,)   binary — bundle of firing neurons' basis HVs
            seq_hv:   (hv_dim,)   binary — n-gram encoded temporal sequence HV
        """
        # 1. Standard LIF dynamics → binary spikes
        spikes = self.lif.step(input_current)         # (N,) binary

        # 2. Map firing neurons → hypervector
        active = spikes > 0.5                          # (N,) bool
        state_hv = _bundle_active(self.basis, active)  # (D,) binary

        # 3. Temporal n-gram encoding: permute accumulator, XOR with state
        #    This encodes "state at position t" in sequence context
        #    seq_hv(t) = state_hv(t) XOR permute(seq_hv(t-1))
        #    → each position gets a unique permutation, order is preserved
        permuted = torch.roll(self._seq_hv, shifts=1, dims=0)
        if state_hv.any():
            self._seq_hv = ((permuted > 0.5) != (state_hv > 0.5)).float()
        else:
            self._seq_hv = permuted

        self._step += 1
        return spikes, state_hv, self._seq_hv.clone()

    def reset(self) -> None:
        self.lif.reset()
        self._seq_hv.zero_()
        self._step = 0

    @property
    def state_dim(self) -> int:
        return self.cfg.hv_dim

    def __repr__(self) -> str:
        return (
            f"SpikingHVLayer(n_neurons={self.cfg.n_neurons}, "
            f"hv_dim={self.cfg.hv_dim})"
        )


# ── SpikingHVNetwork ───────────────────────────────────────────────────────────

@dataclass
class SpikingHVNetworkConfig:
    input_size: int = 100
    n_neurons: int = 128
    hv_dim: int = 4096
    input_gain: float = 5.0
    # Recurrence via HV similarity — no W_rec matrix
    rec_gain: float = 2.0          # scale factor for HV-based recurrent input
    # LIF parameters
    tau: float = 20.0
    v_th: float = 1.0
    v_reset: float = 0.0
    refractory: int = 2
    # Threshold adaptation
    enable_threshold_adaptation: bool = False
    threshold_adaptation_rate: float = 0.01
    seed: Optional[int] = None
    device: str = "cpu"


class SpikingHVNetwork(nn.Module):
    """
    Full HV-SNN: recurrent connections are Hamming similarities, not weights.

    Architecture::

        x_t ─→ W_in ─→ input current
                            +
        state_hv(t-1) ─→ sim(state_hv, receptor_i) ─→ recurrent current
                            ↓
                       LIF neuron i
                            ↓
                       spike z_i(t)
                            ↓
                   bundle(basis_i : z_i = 1)
                            ↓
                       state_hv(t)    ← O(D), not O(N)

    Recurrent connections
    ---------------------
    Each neuron i has:
    - basis_i:    the HV it EMITS when it fires (output)
    - receptor_i: the HV pattern it RESPONDS to (input)

    The recurrent input to neuron i is:
        I_rec_i(t) = rec_gain × sim(state_hv(t-1), receptor_i)

    This replaces W_rec (N×N) with two basis matrices (2×N×D).
    For D=N this is the same parameter count. For D >> N (standard HDC)
    the recurrent computation is O(N·D) vs O(N²), with richer temporal
    dynamics from the high-dimensional state.

    No weight matrix for recurrence. No backprop. No eligibility traces.
    The HDC associative memory IS the recurrence.
    """

    def __init__(self, config: Optional[SpikingHVNetworkConfig] = None, **kwargs):
        super().__init__()
        cfg = config or SpikingHVNetworkConfig(**kwargs)
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        # Input weights (feedforward — still needed for non-spike inputs)
        g = torch.Generator()
        if cfg.seed is not None:
            g.manual_seed(cfg.seed)
        W_in = torch.nn.init.xavier_uniform_(
            torch.empty(cfg.n_neurons, cfg.input_size)
        ).to(self.device)
        self.register_buffer("W_in", W_in)

        # LIF dynamics
        self.lif = LIFLayer(
            size=cfg.n_neurons,
            tau=cfg.tau,
            v_th=cfg.v_th,
            v_reset=cfg.v_reset,
            refractory=cfg.refractory,
            device=cfg.device,
            enable_threshold_adaptation=cfg.enable_threshold_adaptation,
            threshold_adaptation_rate=cfg.threshold_adaptation_rate,
        )

        # Basis HVs: what each neuron EMITS when it fires
        basis = _gen_basis(cfg.n_neurons, cfg.hv_dim, cfg.seed, cfg.device)
        self.register_buffer("basis", basis)

        # Receptor HVs: what HV pattern each neuron RESPONDS to
        rec_seed = cfg.seed + 1 if cfg.seed is not None else None
        receptors = _gen_basis(cfg.n_neurons, cfg.hv_dim, rec_seed, cfg.device)
        self.register_buffer("receptors", receptors)

        # Current network state HV
        self._state_hv = torch.zeros(cfg.hv_dim, device=self.device)
        # Temporal sequence HV
        self._seq_hv = torch.zeros(cfg.hv_dim, device=self.device)

    def step(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        One timestep.

        Args:
            x: (input_size,) input

        Returns:
            spikes:   (n_neurons,) binary
            state_hv: (hv_dim,)   network state — who fired this step
            seq_hv:   (hv_dim,)   temporal sequence HV
        """
        x = x.to(self.device)

        # Feedforward input
        I_in = self.cfg.input_gain * (self.W_in @ x)   # (N,)

        # Recurrent input via HV similarity — replaces W_rec @ prev_spikes
        # sim(state_hv, receptor_i) ∈ [0,1] for each neuron i
        I_rec = self.cfg.rec_gain * _hamming_sim_vec(
            self._state_hv, self.receptors
        )                                                # (N,)

        # LIF step
        spikes = self.lif.step(I_in + I_rec)            # (N,) binary

        # Update state HV: bundle firing neurons' basis HVs
        active = spikes > 0.5
        self._state_hv = _bundle_active(self.basis, active)

        # Temporal n-gram HV: permute + XOR
        permuted = torch.roll(self._seq_hv, shifts=1, dims=0)
        if self._state_hv.any():
            self._seq_hv = ((permuted > 0.5) != (self._state_hv > 0.5)).float()
        else:
            self._seq_hv = permuted

        return spikes, self._state_hv.clone(), self._seq_hv.clone()

    def run_sequence(
        self, spike_sequence: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process a full spike sequence and return the final state and sequence HVs.

        Args:
            spike_sequence: (T, input_size) input spike train

        Returns:
            state_hv_final: (hv_dim,) state after last step
            seq_hv_final:   (hv_dim,) accumulated temporal sequence HV
        """
        T = spike_sequence.shape[0]
        for t in range(T):
            _, state_hv, seq_hv = self.step(spike_sequence[t])
        return state_hv, seq_hv

    def forward(self, x: torch.Tensor) -> dict:
        """
        Forward pass for nn.Module compatibility.

        Args:
            x: (B, T, input_size) or (T, input_size) spike input

        Returns:
            dict with state_hv (B,D), seq_hv (B,D), spikes (B,T,N)
        """
        squeeze = x.dim() == 2
        if squeeze:
            x = x.unsqueeze(0)
        B, T, _ = x.shape

        all_state, all_seq, all_spikes = [], [], []
        for b in range(B):
            self.reset()
            s_list, sq_list, sp_list = [], [], []
            for t in range(T):
                sp, sv, sqv = self.step(x[b, t])
                s_list.append(sv); sq_list.append(sqv); sp_list.append(sp)
            all_state.append(s_list[-1])
            all_seq.append(sq_list[-1])
            all_spikes.append(torch.stack(sp_list))

        result = {
            "state_hv": torch.stack(all_state),
            "seq_hv":   torch.stack(all_seq),
            "spikes":   torch.stack(all_spikes),
        }
        if squeeze:
            result = {k: v.squeeze(0) for k, v in result.items()}
        return result

    def reset(self) -> None:
        self.lif.reset()
        self._state_hv.zero_()
        self._seq_hv.zero_()

    def as_hv_model(self):
        """
        Return a callable suitable for wrapping with HVModel.

        The callable takes (T, input_size) spike sequence and returns
        (hv_dim,) sequence HV — ready for HVPipeline composition.

        Example::

            net = SpikingHVNetwork(cfg)
            hv_snn = HVModel(
                net.as_hv_model(),
                HVModelConfig(model_output_dim=cfg.hv_dim, bypass_bridge=True),
            )
        """
        def _fn(spike_sequence: torch.Tensor) -> torch.Tensor:
            self.reset()
            _, seq_hv = self.run_sequence(spike_sequence)
            return seq_hv.unsqueeze(0)   # (1, D) for HVModel batch convention

        return _fn

    def as_inference_encoder(self, use_seq_hv: bool = True):
        """Return a callable for use as the encoder in HolographicInferenceModel.

        The SpikingHVNetwork already produces a binary HV — no bridge or
        binarization needed.  This closes the spike-as-vector loop:

            SpikingHVNetwork  →  binary (hv_dim,) state/sequence HV
                              →  HolographicInferenceModel.train(hv, label)
                              →  HolographicInferenceModel.classify(hv)

        Args:
            use_seq_hv: If True (default), use the temporal sequence HV
                        (encodes spike *timing* via permutation n-grams).
                        If False, use the state HV (encodes *which* neurons
                        fired, not when).

        Returns:
            Callable (spike_sequence: (T, input_size)) → (hv_dim,) binary HV

        Example::

            net = SpikingHVNetwork(SpikingHVNetworkConfig(input_size=16, n_neurons=128, hv_dim=8192))
            model = HolographicInferenceModel(dim=8192, n_classes=5, encoder=net.as_inference_encoder())
            for spike_train, label in dataset:
                model.train(spike_train, label)
            model.finalize()
            pred, stats = model.classify(query_spikes)
        """
        def _enc(spike_sequence: torch.Tensor) -> torch.Tensor:
            self.reset()
            state_hv, seq_hv = self.run_sequence(spike_sequence)
            hv = seq_hv if use_seq_hv else state_hv
            # Already binary {0,1} — no binarization needed
            return hv.float()

        return _enc

    @property
    def n_parameters(self) -> int:
        """Count trainable parameters (only W_in — no W_rec)."""
        return self.W_in.numel()

    def sequence_similarity(
        self,
        seq_a: torch.Tensor,   # (T, input_size)
        seq_b: torch.Tensor,   # (T, input_size)
    ) -> float:
        """
        Compute the Hamming similarity between two sequences' state HVs.

        Runs both sequences through the network independently and compares
        the resulting state hypervectors.  Two sequences with similar
        temporal dynamics produce similar state HVs.

        Args:
            seq_a, seq_b: (T, input_size) spike sequences

        Returns:
            Hamming similarity ∈ [0, 1]
        """
        self.reset()
        out_a = self.run_sequence(seq_a)
        hv_a  = out_a[-1]   # use last state HV

        self.reset()
        out_b = self.run_sequence(seq_b)
        hv_b  = out_b[-1]

        return float(_hamming_sim_vec(hv_a, hv_b.unsqueeze(0))[0].item())

    def state_trajectory_hvs(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the sequence of state HVs as the input sequence progresses.

        Useful for visualising how the network's state evolves over time.

        Args:
            x: (T, input_size) input sequence

        Returns:
            (T, hv_dim) tensor of state HVs at each timestep
        """
        self.reset()
        hvs = []
        for t in range(x.shape[0]):
            _, state_hv, _ = self.step(x[t])
            hvs.append(state_hv)
        return torch.stack(hvs)   # (T, D)

    def __repr__(self) -> str:
        return (
            f"SpikingHVNetwork(input={self.cfg.input_size}, "
            f"n_neurons={self.cfg.n_neurons}, hv_dim={self.cfg.hv_dim}, "
            f"W_in={self.cfg.n_neurons}×{self.cfg.input_size} "
            f"[no W_rec — HV recurrence])"
        )


# ── Demo ──────────────────────────────────────────────────────────────────────

def demo_hv_snn():
    """Show spike-as-vector encoding and HVA integration."""
    torch.manual_seed(42)
    N, D, T = 64, 512, 100

    print("SpikingHVLayer — drop-in LIF replacement")
    layer = SpikingHVLayer(SpikingHVConfig(n_neurons=N, hv_dim=D))
    state_hvs = []
    for t in range(T):
        current = torch.randn(N) * 0.5
        spikes, state_hv, seq_hv = layer.step(current)
        state_hvs.append(state_hv)
    print(f"  spikes: {spikes.shape}  state_hv: {state_hv.shape}  seq_hv: {seq_hv.shape}")
    n_fired = sum(s.sum().item() for s in state_hvs)
    print(f"  {T} steps, {n_fired:.0f} total spikes → each mapped to {D}-dim HV")

    print("\nSpikingHVNetwork — no W_rec, HV recurrence")
    net = SpikingHVNetwork(SpikingHVNetworkConfig(
        input_size=32, n_neurons=N, hv_dim=D
    ))
    seq = torch.randint(0, 2, (T, 32)).float()
    state_hv, seq_hv = net.run_sequence(seq)
    print(f"  {T} steps → state_hv: {state_hv.shape}, seq_hv: {seq_hv.shape}")
    print(f"  Parameters: {net.n_parameters} (W_in only, no W_rec)")

    # Connect to HVPipeline
    print("\nIntegrating SpikingHVNetwork into HVPipeline")
    from hdc.hypervector_architecture import HVModel, HVModelConfig, HVPipeline

    hv_snn = HVModel(
        net.as_hv_model(),
        HVModelConfig(hv_dim=D, model_output_dim=D, role_name="spike"),
        bypass_bridge=True,   # output is already a D-dim hypervector
    )
    pipe = HVPipeline(
        models={"spike": hv_snn},
        n_classes=4, hv_dim=D, strategy="bundle",
    )
    joint = pipe.encode({"spike": seq})
    print(f"  Joint HV: {joint.shape} — SNN state natively in HVA pipeline")
    print("  HV-SNN demo complete ✓")


if __name__ == "__main__":
    demo_hv_snn()
