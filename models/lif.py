import torch
from dataclasses import dataclass
from typing import Optional

@dataclass
class LIFConfig:
    size: int = 16
    tau: float = 20.0
    v_th: float = 1.0
    v_reset: float = 0.0
    refractory: int = 2
    device: Optional[str] = None
    # Threshold modulation for test-time adaptation (Zhao et al. 2026)
    # arXiv:2505.05375 — zero additional compute, handles distribution drift
    enable_threshold_adaptation: bool = False
    threshold_adaptation_rate: float = 0.01   # γ: how fast threshold tracks v̄
    threshold_momentum: float = 0.99          # EMA decay for running mean of v
    # Heterogeneous time constants (Perez-Nieves et al. 2021, Nature Comm.)
    # "Neural heterogeneity promotes robust learning"
    # Per-neuron tau sampled from Lognormal(log(tau), sigma_log_tau).
    # Fast neurons (small tau) → precise timing; slow neurons → temporal context.
    # Expected: +2–4% on sequential/temporal tasks vs. uniform tau.
    heterogeneous_tau: bool = False
    sigma_log_tau: float = 0.5   # std of log(tau); 0.5 → tau range ≈ [0.6×, 1.6×] median

class LIFLayer:
    """Leaky Integrate-and-Fire neuron layer."""

    def __init__(
        self,
        size=None,
        tau=20.0,
        v_th=1.0,
        v_reset=0.0,
        device=None,
        refractory=2,
        dt=1.0,
        config=None,
        enable_threshold_adaptation=False,
        threshold_adaptation_rate=0.01,
        threshold_momentum=0.99,
        heterogeneous_tau=False,
        sigma_log_tau=0.5,
    ):
        """Initialize LIF layer.
        
        Args:
            size: Number of neurons or LIFConfig object
            tau: Membrane time constant
            v_th: Spike threshold voltage
            v_reset: Reset voltage after spike
            device: PyTorch device for computation
            refractory: Refractory period
            dt: Time step
            config: LIFConfig object (alternative to individual params)
        """
        # Handle case where config is passed as first argument
        if isinstance(size, LIFConfig):
            config = size
            size = None
            
        if config is not None:
            size = config.size
            tau = config.tau
            v_th = config.v_th
            v_reset = config.v_reset
            refractory = config.refractory
            device = config.device
            enable_threshold_adaptation = config.enable_threshold_adaptation
            threshold_adaptation_rate = config.threshold_adaptation_rate
            threshold_momentum = config.threshold_momentum
            heterogeneous_tau = config.heterogeneous_tau
            sigma_log_tau = config.sigma_log_tau

        self.size = size
        # Heterogeneous tau (Perez-Nieves 2021): per-neuron time constants.
        # tau_i ~ Lognormal(log(tau), sigma_log_tau) — gives fast+slow neurons.
        if heterogeneous_tau and size is not None:
            import math as _math
            device_t = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
            log_mean = _math.log(float(tau))
            self.tau = torch.exp(
                torch.normal(log_mean, sigma_log_tau, size=(size,))
            ).to(device_t).clamp(min=1.0, max=500.0)
        else:
            self.tau = tau
        self.v_th = v_th
        self.v_reset = v_reset
        self.refractory = refractory
        self.dt = dt
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.v = torch.zeros(size, device=self.device)
        self.refractory_counter = torch.zeros(size, device=self.device)
        self.spike_hist = []

        # Threshold modulation (Zhao et al. 2026, arXiv:2505.05375)
        self.enable_threshold_adaptation = enable_threshold_adaptation
        self._th_adapt_rate = threshold_adaptation_rate
        self._th_momentum = threshold_momentum
        self._v_th_0 = float(v_th)              # nominal threshold (anchors adaptation)
        self._v_running_mean: Optional[torch.Tensor] = None  # EMA of membrane potential

    def step(self, input_current: torch.Tensor) -> torch.Tensor:
        """Process one timestep of input current.
        
        Args:
            input_current: Input current tensor of shape (size,)
            
        Returns:
            Spike tensor of shape (size,)
        """
        if input_current.dim() != 1:
            raise ValueError(f"Expected 1D input tensor, got {input_current.dim()}D")
        if input_current.size(0) != self.size:
            raise ValueError(f"Expected input size {self.size}, got {input_current.size(0)}")
            
        input_current = input_current.to(self.device)

        # Decrement refractory counters FIRST (before spike generation)
        self.refractory_counter = torch.clamp(self.refractory_counter - 1, min=0)

        # Mask for neurons NOT in refractory period
        not_refractory = (self.refractory_counter == 0).float()

        # Membrane potential dynamics: dv/dt = (-v + I) / tau
        # tau may be a scalar or a (size,) tensor (heterogeneous case).
        dv = ((-self.v + input_current) / self.tau) * self.dt * not_refractory
        self.v += dv

        # Threshold modulation for test-time adaptation (Zhao et al. 2026)
        # v_th[t] = v_th_0 + γ · (v̄[t] − v_th_0)
        # where v̄ is an EMA of membrane potential across neurons
        if self.enable_threshold_adaptation:
            v_mean = self.v.mean()
            if self._v_running_mean is None:
                self._v_running_mean = v_mean.clone()
            else:
                self._v_running_mean = (
                    self._th_momentum * self._v_running_mean
                    + (1.0 - self._th_momentum) * v_mean
                )
            self.v_th = (
                self._v_th_0
                + self._th_adapt_rate * (self._v_running_mean.item() - self._v_th_0)
            )

        # Spike generation (only if not in refractory period)
        spikes = (self.v >= self.v_th).float() * not_refractory
        
        # Set refractory counter to refractory+1 so the counter reaches 0
        # only AFTER exactly `refractory` blocked steps (off-by-one fix)
        self.refractory_counter[spikes > 0] = self.refractory + 1
        
        # Reset membrane potential after spikes
        self.v = torch.where(spikes > 0, torch.tensor(self.v_reset, device=self.device), self.v)
        
        # Store spike history (keep only last 1000 steps to prevent memory bloat)
        self.spike_hist.append(spikes.clone())
        if len(self.spike_hist) > 1000:
            self.spike_hist.pop(0)

        return spikes
    
    def reset(self, reset_threshold: bool = False) -> None:
        """Reset membrane potential and refractory counters.

        Args:
            reset_threshold: If True, also reset the adapted threshold and
                             running mean back to the nominal value.  Use
                             between *independent* trials; leave False when
                             the adapted threshold should persist across a
                             continuous deployment stream.
        """
        self.v = torch.zeros(self.size, device=self.device)
        self.refractory_counter = torch.zeros(self.size, device=self.device)
        self.spike_hist = []
        if reset_threshold and self.enable_threshold_adaptation:
            self.v_th = self._v_th_0
            self._v_running_mean = None
    
    def get_firing_rates(self, window: int = 100) -> torch.Tensor:
        """Calculate firing rates (Hz) over a window.
        
        Args:
            window: Window size for rate calculation
            
        Returns:
            Firing rates tensor (spikes per timestep)
        """
        if len(self.spike_hist) == 0:
            return torch.zeros(self.size, device=self.device)
        
        # Use actual window size (up to available history)
        actual_window = min(window, len(self.spike_hist))

        # Sum spikes over window
        recent = torch.stack(self.spike_hist[-actual_window:])
        return recent.sum(dim=0) / actual_window

    def population_synchrony(self, window: int = 10) -> float:
        """
        Measure synchrony: fraction of timesteps where multiple neurons spike together.

        High synchrony (close to 1) = neurons fire in lockstep (oscillatory).
        Low synchrony (close to 0) = neurons fire independently (asynchronous).

        Relevant for:
          - Oscillation detection (beta/gamma in motor cortex)
          - Epilepsy-like hyper-synchrony detection
          - Population coding vs rate coding distinction

        Returns:
            Synchrony ∈ [0, 1]
        """
        if len(self.spike_hist) < 2:
            return 0.0
        recent = torch.stack(self.spike_hist[-window:])   # (W, N)
        # Synchrony = mean(std of spike count per step / max_possible_std)
        counts_per_step = recent.sum(dim=1).float()   # (W,) spikes per timestep
        max_count = float(self.size)
        if max_count == 0:
            return 0.0
        # High count variance = synchronised (all fire together or all silent)
        sync = float(counts_per_step.std().item()) / max(max_count / 4.0, 1.0)
        return min(1.0, sync)

    def burst_score(self, isi_threshold: int = 3) -> torch.Tensor:
        """
        Per-neuron burst detection: fraction of spikes part of a burst.

        A burst is two or more spikes with inter-spike interval ≤ isi_threshold.

        Returns:
            (N,) burst score per neuron ∈ [0, 1]
        """
        if len(self.spike_hist) < 2:
            return torch.zeros(self.size, device=self.device)

        spikes = torch.stack(self.spike_hist).float()   # (T, N)
        T = spikes.shape[0]

        burst_counts = torch.zeros(self.size, device=self.device)
        spike_counts = spikes.sum(dim=0).clamp(min=1)

        for t in range(1, T):
            # Neuron bursts if it fired at both t-1 and t (ISI=1) or within threshold
            for lag in range(1, min(isi_threshold + 1, t + 1)):
                consecutive = (spikes[t] > 0.5) & (spikes[t - lag] > 0.5)
                burst_counts += consecutive.float()

        return (burst_counts / spike_counts).clamp(0, 1)

    def neuron_health(self, window: int = 50) -> dict:
        """
        One-call diagnostic: mean/max firing rate, synchrony, burst fraction.

        Thresholds:
          mean_rate > 0.3  → over-active (potential runaway)
          mean_rate < 0.01 → silent (dead neurons / gain too low)
          synchrony > 0.5  → oscillatory regime
        """
        rates = self.get_firing_rates(window)
        mean_rate = float(rates.mean().item())
        max_rate  = float(rates.max().item())
        synchrony = self.population_synchrony(window)
        bursts    = self.burst_score()
        mean_burst = float(bursts.mean().item())
        return {
            "mean_firing_rate": round(mean_rate, 4),
            "max_firing_rate":  round(max_rate, 4),
            "synchrony":        round(synchrony, 4),
            "mean_burst_score": round(mean_burst, 4),
            "n_silent":         int((rates < 0.01).sum().item()),
            "n_overactive":     int((rates > 0.3).sum().item()),
            "n_neurons":        self.size,
            "diagnosis":        (
                "over_active"  if mean_rate > 0.3  else
                "silent"       if mean_rate < 0.01 else
                "oscillatory"  if synchrony > 0.5  else
                "healthy"
            ),
        }
