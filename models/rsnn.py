import torch
from dataclasses import dataclass
from typing import Optional, Union
from models.lif import LIFLayer

# HDC cross-coupling: voltage scaling and memory error protection for SNN weights
from hdc.voltage_scaling import VoltageScaler, VoltageScalingConfig
from hdc.memory_errors import MemoryErrorInjector, MemoryErrorConfig
from hdc.fault_models import FaultInjector, FaultConfig, FaultType
from hdc.ecc import HDCCorrector, ECCConfig

@dataclass
class RSNNConfig:
    """Configuration for Recurrent Spiking Neural Network."""
    input_size:  int   = 100
    hidden_size: int   = 128
    device:      Optional[str] = None
    input_gain:  float = 5.0
    sparse_init: bool  = False
    sparse_p:    float = 0.15
    # LIF neuron parameters
    tau:         float = 20.0   # Membrane time constant (higher = slower integration)
    v_th:        float = 1.0    # Spike threshold
    v_reset:     float = 0.0    # Reset voltage
    refractory:  int   = 2      # Refractory period (timesteps)
    dt:          float = 1.0    # Simulation time step (ms)
    # Heterogeneous time constants (Perez-Nieves et al. 2021, Nature Comm.)
    heterogeneous_tau: bool = False
    sigma_log_tau: float = 0.5
    # HDC robustness integration
    enable_voltage_scaling: bool = False
    enable_memory_error_injection: bool = False
    nominal_voltage: float = 0.8
    memory_error_rate: float = 1e-6
    # Fault model (SpikeFI-compatible)
    fault_type: str = "none"    # "none", "stuck_at_0", "stuck_at_1", "wbf_t", "wbf_p", "syn_silence", "mixed"
    fault_rate: float = 0.0
    fault_persistent: bool = True

class RSNN:
    """Recurrent Spiking Neural Network with LIF neurons.

    A sparse recurrent network using Leaky Integrate-and-Fire neurons for
    event-driven computation. Supports online learning via eligibility traces.

    Attributes:
        input_size: Number of input features
        hidden_size: Number of hidden recurrent neurons
        device: PyTorch device for computation
        W_in: Input weight matrix (hidden_size, input_size)
        W_rec: Recurrent weight matrix (hidden_size, hidden_size)
        input_gain: Multiplicative gain applied to input currents
        lif: LIFLayer instance for neuron dynamics
        prev_spikes: Spike vector from previous timestep
        spike_list: List of spikes (for PC stack integration)
    """

    def __init__(
        self,
        input_size: Optional[int] = None,
        hidden_size: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
        config: Optional[RSNNConfig] = None,
        sparse_init: bool = False,
        sparse_p: float = 0.15,
        input_gain: float = 5.0,
        use_dale: bool = False,
        exc_frac: float = 0.8,
        use_stp: bool = False,
        stp_U: float = 0.5,
        stp_tau_rec: float = 200.0,
        stp_tau_fac: float = 20.0,
    ) -> None:
        """Initialize RSNN.

        Args:
            input_size: Number of input features (required if config=None)
            hidden_size: Number of hidden neurons (required if config=None)
            device: PyTorch device for computation (overrides config.device)
            config: RSNNConfig dataclass (alternative to individual params)
            sparse_init: Whether to use sparse initialization for W_rec
            sparse_p: Sparsity probability for sparse initialization
            input_gain: Input current amplification factor

        Raises:
            ValueError: If neither config nor (input_size, hidden_size) provided
        """
        # Accept RSNNConfig as first positional argument
        if isinstance(input_size, RSNNConfig):
            config = input_size
            input_size = None

        if config is not None:
            input_size   = config.input_size
            hidden_size  = config.hidden_size
            device       = device or config.device
            input_gain   = config.input_gain
            sparse_init  = config.sparse_init
            sparse_p     = config.sparse_p

        if input_size is None or hidden_size is None:
            raise ValueError("Must provide either config or input_size + hidden_size")

        self.input_size: int = input_size
        self.hidden_size: int = hidden_size
        self.device: torch.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Input weights: Xavier uniform (feedforward scaling)
        self.W_in: torch.Tensor = torch.nn.init.xavier_uniform_(
            torch.empty(hidden_size, input_size)
        ).to(self.device)
        
        # Recurrent weights: orthogonal initialization for stability, then sparsify
        W_rec_init = torch.empty(hidden_size, hidden_size)
        if sparse_init:
            # Sparse Erdos-Renyi initialization with spectral normalization
            # Critical for SNN stability: controls recurrent eigenvalue spectrum
            mask = torch.rand(hidden_size, hidden_size) < sparse_p
            W_rec_init = torch.nn.init.orthogonal_(W_rec_init)
            W_rec_init = W_rec_init * mask.float()
            # Edge-of-chaos spectral radius = 0.98 (Jaeger & Haas 2004, Science)
            # Maximises temporal memory capacity for BCI velocity decoding
            with torch.no_grad():
                eigenvalues = torch.linalg.eigvals(W_rec_init)
                spectral_radius = eigenvalues.abs().max().item()
                if spectral_radius > 0.98:
                    W_rec_init = W_rec_init * (0.98 / spectral_radius)
        else:
            # Dense orthogonal initialization (good spectral properties)
            W_rec_init = torch.nn.init.orthogonal_(W_rec_init)
            
        # Zero diagonal — no self-connections (biologically realistic)
        W_rec_init.fill_diagonal_(0.0)
        self.W_rec: torch.Tensor = W_rec_init.to(self.device)

        self.input_gain: float = input_gain
        self._sparse_init: bool = sparse_init
        self._sparse_p: float = sparse_p
        # Per-neuron input gain (None until enable_per_neuron_gain() is called).
        # When set, overrides the scalar self.input_gain in forward() for
        # channel-selective amplification — useful when some input channels
        # carry more information than others (e.g. selective BCI electrodes).
        self.per_neuron_gain: Optional[torch.Tensor] = None

        # LIF neuron parameters from config
        self._tau = config.tau if config is not None else 20.0
        self._v_th = config.v_th if config is not None else 1.0
        self._v_reset = config.v_reset if config is not None else 0.0
        self._refractory = config.refractory if config is not None else 2
        self._dt = config.dt if config is not None else 1.0

        _het_tau   = config.heterogeneous_tau if config is not None else False
        _sigma_log = config.sigma_log_tau     if config is not None else 0.5
        self.lif: LIFLayer = LIFLayer(
            hidden_size,
            tau=self._tau,
            v_th=self._v_th,
            v_reset=self._v_reset,
            refractory=self._refractory,
            dt=self._dt,
            device=self.device,
            heterogeneous_tau=_het_tau,
            sigma_log_tau=_sigma_log,
        )
        self.prev_spikes: torch.Tensor = torch.zeros(hidden_size, device=self.device)
        self.spike_list: list[torch.Tensor] = []

        # SynOp (synaptic operation) counter for energy estimation
        self.total_synops: int = 0
        self.total_inferences: int = 0

        # Dale's law E/I mask (Brunel 2000; Perez-Nieves et al. 2021 NeurIPS)
        # 80% excitatory (+1), 20% inhibitory (−1) by default
        self.use_dale = use_dale
        if use_dale:
            n_exc = int(exc_frac * hidden_size)
            dale = torch.ones(hidden_size, device=self.device)
            dale[n_exc:] = -1.0
            self.dale_mask: torch.Tensor = dale
            # Enforce sign constraint on initial weights
            with torch.no_grad():
                self.W_rec = self.W_rec.abs() * self.dale_mask.unsqueeze(0)

        # Short-Term Plasticity — Tsodyks-Markram model
        # (Tsodyks & Markram 1997 PNAS; Zucker & Regehr 2002 Ann Rev Physiol)
        self.use_stp = use_stp
        if use_stp:
            import math
            self.stp_U       = stp_U
            self.stp_decay_x = math.exp(-1.0 / stp_tau_rec)  # resource recovery
            self.stp_decay_u = math.exp(-1.0 / stp_tau_fac)  # facilitation decay
            # State: u (use/facilitation), x (available resources)
            self.stp_u: torch.Tensor = torch.full((hidden_size,), stp_U,
                                                  device=self.device)
            self.stp_x: torch.Tensor = torch.ones(hidden_size, device=self.device)

        # HDC robustness: voltage scaling and memory error injection
        self.enable_voltage_scaling = False
        self.enable_memory_error_injection = False
        if config is not None:
            self.enable_voltage_scaling = config.enable_voltage_scaling
            self.enable_memory_error_injection = config.enable_memory_error_injection
        if self.enable_voltage_scaling:
            self.voltage_scaler = VoltageScaler(
                config=VoltageScalingConfig(nominal_voltage=config.nominal_voltage)
            )
        if self.enable_memory_error_injection:
            self.memory_error_injector = MemoryErrorInjector(
                config=MemoryErrorConfig(
                    error_rate=config.memory_error_rate,
                    enable_injection=True,
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through RSNN.
        
        Args:
            x: Input tensor of shape (input_size,)
            
        Returns:
            Spike tensor of shape (hidden_size,)
        """
        if x.dim() != 1:
            raise ValueError(f"Expected 1D input tensor, got {x.dim()}D")
        if x.size(0) != self.input_size:
            raise ValueError(f"Expected input size {self.input_size}, got {x.size(0)}")
            
        x = x.to(self.device)

        # Apply Dale's law sign constraint to W_rec before each step
        if self.use_dale:
            W_rec_eff = self.W_rec.abs() * self.dale_mask.unsqueeze(0)
        else:
            W_rec_eff = self.W_rec

        # HDC memory error injection: simulate hardware faults on weights
        if self.enable_memory_error_injection:
            W_rec_eff = self.memory_error_injector.inject(W_rec_eff)
            W_in_eff = self.memory_error_injector.inject(self.W_in)
        else:
            W_in_eff = self.W_in

        # Short-term plasticity: scale recurrent input by STP factor (ASE)
        if self.use_stp:
            # Decay u and x between spikes (continuous approximation, dt=1)
            self.stp_u = self.stp_U + (self.stp_u - self.stp_U) * self.stp_decay_u
            self.stp_x = 1.0 + (self.stp_x - 1.0) * self.stp_decay_x
            # On spike: update use and deplete resources
            fired = (self.prev_spikes > 0.5)
            self.stp_u = torch.where(fired,
                self.stp_u + self.stp_U * (1.0 - self.stp_u),
                self.stp_u)
            ase = self.stp_u * self.stp_x                    # amplitude of synaptic effect
            self.stp_x = torch.where(fired, self.stp_x - ase, self.stp_x)
            # Weight modulated recurrent input
            rec_current = W_rec_eff @ (self.prev_spikes * ase)
        else:
            rec_current = W_rec_eff @ self.prev_spikes

        gain = self.per_neuron_gain if self.per_neuron_gain is not None else self.input_gain
        input_current = gain * (W_in_eff @ x) + rec_current
        spikes = self.lif.step(input_current)
        self.prev_spikes = spikes.clone()
        # Expose spike_list for Predictive Coding stack integration
        self.spike_list = [spikes]

        # Count synaptic operations (SynOps) for energy estimation
        # A SynOp occurs for every non-zero pre-synaptic spike × connected synapse
        input_spike_count = int(x.count_nonzero().item() if x.count_nonzero().numel() > 0 else 0)
        rec_spike_count = int(self.prev_spikes.count_nonzero().item() if self.prev_spikes.count_nonzero().numel() > 0 else 0)
        # Input SynOps: each active input channel contributes to ALL hidden neurons
        # Recurrent SynOps: each active recurrent neuron propagates through ALL hidden neurons
        # But with sparse_init, recurrent weights may be sparse
        if self.use_stp:
            stp_scale = 1.0  # STP doesn't change the count, just the weight
        else:
            stp_scale = 1.0
        input_synops = input_spike_count * self.hidden_size
        # For recurrent: only count synapses where W_rec[j,i] != 0 (sparsity)
        if self._sparse_init:
            nz_per_row = int((self.W_rec != 0).float().sum(dim=1).mean().item())
            rec_synops = rec_spike_count * nz_per_row
        else:
            rec_synops = rec_spike_count * self.hidden_size
        self.total_synops += input_synops + rec_synops
        self.total_inferences += 1

        return spikes
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Callable wrapper for forward pass.

        Args:
            x: Input tensor of shape (input_size,)

        Returns:
            Spike tensor of shape (hidden_size,)
        """
        return self.forward(x)

    def reset(self) -> None:
        """Reset network state (previous spikes and membrane potentials)."""
        self.prev_spikes = torch.zeros(self.hidden_size, device=self.device)
        self.lif.v = torch.zeros(self.hidden_size, device=self.device)
        self.spike_list = []
        self.total_synops = 0
        self.total_inferences = 0
        if self.use_stp:
            self.stp_u = torch.full((self.hidden_size,), self.stp_U, device=self.device)
            self.stp_x = torch.ones(self.hidden_size, device=self.device)
        # Reset voltage scaler error tracking if active
        if self.enable_voltage_scaling:
            self.voltage_scaler.step()

    def get_state(self) -> dict:
        """Return a snapshot of current network state."""
        return {
            "v":            self.lif.v.clone(),
            "prev_spikes":  self.prev_spikes.clone(),
            "firing_rates": self.lif.get_firing_rates(),
        }

    # ── Elite maintenance methods ──────────────────────────────────────────────

    def homeostatic_scale(self, target_radius: float = 0.97) -> float:
        """
        Rescale W_rec to maintain target spectral radius.

        After online learning (BCM, FORCE, SuperSpike), recurrent weights can
        drift outside the edge-of-chaos regime. This call restores the spectral
        radius to `target_radius` without changing the weight structure.

        Frequency: call every ~100 steps or after each sequence boundary.

        Args:
            target_radius: Desired spectral radius (default 0.97)

        Returns:
            Current spectral radius before rescaling
        """
        with torch.no_grad():
            try:
                eigs = torch.linalg.eigvals(self.W_rec)
                current = float(eigs.abs().max().item())
            except Exception:
                return 1.0
            if current > 1e-6:
                self.W_rec.mul_(target_radius / current)
            self.W_rec.fill_diagonal_(0.0)
        return current

    def structural_plasticity(
        self,
        prune_fraction: float = 0.02,
        grow_fraction:  float = 0.01,
        grow_std:       float = 0.005,
    ):
        """
        Synaptic pruning + growth for network self-organisation.

        Reference:
            Torben-Nielsen & De Schutter (2014) "Context-aware modeling of
            neuronal morphology" Frontiers Neuroanatomy.

            Butz, Wörgötter, van Ooyen (2009) "Activity-dependent structural
            plasticity" Brain Research Reviews 60(2):287–305.

        Algorithm:
            1. Prune: zero out the weakest `prune_fraction` of synapses
            2. Grow:  randomly form new synapses at `grow_fraction` of
               currently-zero positions (initialised small)

        This keeps the network sparse while allowing topology change.
        Combined with homeostatic_scale(), prevents runaway weight growth.

        Args:
            prune_fraction: Fraction of weakest weights to prune (0.02 = 2%)
            grow_fraction:  Fraction of zero positions to reconnect (0.01 = 1%)
            grow_std:       Std of newly grown connection weights
        """
        with torch.no_grad():
            abs_w   = self.W_rec.abs()

            # 1. Prune weakest non-zero weights
            non_zero_mask = abs_w > 0
            if non_zero_mask.sum() > 0:
                threshold = abs_w[non_zero_mask].quantile(prune_fraction)
                prune     = non_zero_mask & (abs_w <= threshold)
                self.W_rec[prune] = 0.0

            # 2. Grow new synapses at zero positions
            zero_mask = self.W_rec == 0.0
            # Exclude diagonal (no self-connections)
            diag_mask = torch.eye(self.hidden_size, dtype=torch.bool, device=self.device)
            zero_mask[diag_mask] = False

            if zero_mask.sum() > 0:
                grow_prob = torch.full_like(self.W_rec, grow_fraction)
                grow_mask = zero_mask & (torch.rand_like(self.W_rec) < grow_prob)
                if grow_mask.sum() > 0:
                    new_weights = torch.randn(
                        int(grow_mask.sum().item()), device=self.device
                    ) * grow_std
                    # Enforce Dale's law on new synapses if active
                    if self.use_dale:
                        n_exc = int(0.8 * self.hidden_size)
                        signs = self.dale_mask.unsqueeze(1).expand_as(self.W_rec)
                        new_weights = new_weights.abs() * signs[grow_mask].sign()
                    self.W_rec[grow_mask] = new_weights

            self.W_rec.fill_diagonal_(0.0)

    def enable_per_neuron_gain(self):
        """
        Switch from global scalar input_gain to per-neuron gain vector.

        Initialises per_neuron_gain = input_gain × ones(hidden_size).
        Also initialises a per-neuron firing-rate EMA used for stable adaptation.
        """
        self.per_neuron_gain = torch.full(
            (self.hidden_size,), self.input_gain, device=self.device
        )
        # Per-neuron firing-rate EMA (τ=50 steps) — prevents instantaneous
        # binary spike (0/1) from driving the gain ratio to 100,000×
        self._rate_ema = torch.full(
            (self.hidden_size,), 0.1, device=self.device   # initialise at target rate
        )

    def adapt_input_gain(self, x: torch.Tensor, target_rate: float = 0.1):
        """
        Adjust input gain to maintain target mean firing rate.

        Per-neuron mode: uses a 50-step EMA of each neuron's firing rate so
        that a single silent step (spike=0) does NOT drive ratio→100,000 and
        saturate the gain at the ceiling.  The EMA gives a stable estimate of
        actual per-neuron activity before adjusting gain.

        Args:
            x:           Most recent input (reserved for future use)
            target_rate: Target mean firing rate (default 0.1 = 10%)
        """
        if self.per_neuron_gain is not None:
            # Update per-neuron firing-rate EMA (τ=50 → decay=0.98)
            if not hasattr(self, '_rate_ema'):
                self._rate_ema = torch.full(
                    (self.hidden_size,), target_rate, device=self.device
                )
            self._rate_ema = 0.98 * self._rate_ema + 0.02 * self.prev_spikes.float()
            # Ratio uses EMA rate (stable) not instantaneous binary spike
            smooth_rates = self._rate_ema.clamp(min=0.01)   # floor at 1% to prevent ÷0
            ratio = target_rate / smooth_rates
            # Soft update: move 1% toward the target gain each call
            self.per_neuron_gain = (
                0.99 * self.per_neuron_gain + 0.01 * self.per_neuron_gain * ratio
            ).clamp(0.5, 10.0)   # cap at 10 (was 20 — ceiling was too easy to hit)
        else:
            # Global scalar: soft adaptation using mean rate EMA
            if not hasattr(self, '_global_rate_ema'):
                self._global_rate_ema = target_rate
            self._global_rate_ema = (
                0.98 * self._global_rate_ema + 0.02 * float(self.prev_spikes.mean().item())
            )
            ratio = target_rate / max(self._global_rate_ema, 0.01)
            self.input_gain = float(
                0.99 * self.input_gain + 0.01 * self.input_gain * ratio
            )
            self.input_gain = max(0.5, min(10.0, self.input_gain))

    def spectral_radius(self) -> float:
        """Compute current spectral radius of W_rec."""
        try:
            eigs = torch.linalg.eigvals(self.W_rec)
            return float(eigs.abs().max().item())
        except Exception:
            return float("nan")

    def network_health(self) -> dict:
        """
        Comprehensive RSNN diagnostic: spectral radius, weight stats, sparsity.

        Edge-of-chaos criterion: spectral_radius ≈ 0.9–1.05.
        Sparsity < 0.1 → almost fully connected (potential over-parameterisation).
        """
        sr = self.spectral_radius()
        w  = self.W_rec.data
        sparsity = float((w.abs() < 1e-6).float().mean().item())
        w_in_norm = float(self.W_in.data.norm().item()) if hasattr(self, "W_in") else None
        state = self.get_state()
        mean_rate = float(state.get("prev_spikes", torch.zeros(1)).float().mean().item())
        return {
            "spectral_radius":  round(sr, 4),
            "w_rec_mean_abs":   round(float(w.abs().mean().item()), 6),
            "w_rec_max_abs":    round(float(w.abs().max().item()), 4),
            "sparsity":         round(sparsity, 4),
            "w_in_norm":        round(w_in_norm, 4) if w_in_norm is not None else None,
            "hidden_size":      self.hidden_size,
            "mean_spike_rate":  round(mean_rate, 4),
            "edge_of_chaos":    0.85 <= sr <= 1.05,
            "diagnosis":        (
                "chaotic"      if sr > 1.05 else
                "stable_edge"  if sr >= 0.85 else
                "damped"
            ),
        }
