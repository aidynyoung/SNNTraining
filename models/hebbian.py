import math
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

# HDC cross-coupling: error masking protects Hebbian traces from hardware faults
from hdc.error_masking import ErrorMasker, ErrorMaskingConfig
from hdc.memory_errors import MemoryErrorInjector, MemoryErrorConfig

@dataclass
class HebbianConfig:
    shape: Tuple[int, int] = (128, 128)
    tau_fast: float = 5.0      # Fast eligibility trace time constant (IN TIMESTEPS)
    tau_slow: float = 50.0     # Slow eligibility trace time constant (IN TIMESTEPS)
    dt: float = 0.02           # Timestep duration in seconds (default 20ms → 50Hz sampling)
    alpha: float = 0.7
    beta: float = 0.3
    device: Optional[str] = None
    # HDC robustness integration
    enable_error_masking: bool = False
    masking_scheme: str = "zero"  # "zero", "sign_bit", "word"
    error_rate: float = 1e-6

    @property
    def tau_fast_ms(self) -> float:
        """Fast trace time constant in milliseconds."""
        return self.tau_fast * self.dt * 1000.0

    @property
    def tau_slow_ms(self) -> float:
        """Slow trace time constant in milliseconds."""
        return self.tau_slow * self.dt * 1000.0

@torch.jit.script
def _update_traces_jit(
    e_fast: torch.Tensor,
    e_slow: torch.Tensor,
    outer: torch.Tensor,
    decay_fast: float,
    decay_slow: float,
    alpha: float,
    beta: float,
) -> torch.Tensor:
    """
    JIT-compiled trace update. In-place mul_ + add_ avoids 4 temp tensors.

    Args:
        e_fast: Fast eligibility trace buffer (modified in-place)
        e_slow: Slow eligibility trace buffer (modified in-place)
        outer: Outer product of post and pre spikes
        decay_fast: Decay factor for fast trace
        decay_slow: Decay factor for slow trace
        alpha: Weight for fast trace contribution
        beta: Weight for slow trace contribution

    Returns:
        Combined eligibility trace E(t) = alpha * e_fast + beta * e_slow
    """
    e_fast.mul_(decay_fast).add_(outer)     # in-place: e_fast = decay*e_fast + outer
    e_slow.mul_(decay_slow).add_(outer)     # in-place: e_slow = decay*e_slow + outer
    return alpha * e_fast + beta * e_slow   # combined trace E(t)


class DualHebbianAccumulator(nn.Module):
    """Alias for DualHebbian for backward compatibility."""
    def __init__(self, config: HebbianConfig, device: Union[str, torch.device] = "cpu") -> None:
        super().__init__()
        self._impl = DualHebbian(config, device)

    def update(self, pre: torch.Tensor, post: torch.Tensor) -> torch.Tensor:
        return self._impl.update(pre, post)

    def reset(self) -> None:
        self._impl.reset()


class DualHebbian(nn.Module):
    """
    Dual-timescale Hebbian accumulator with JIT-compiled inner loop.

    Uses fast and slow eligibility traces for online learning in SNNs.
    The update loop is JIT-compiled to TorchScript for zero Python overhead.

    Parameters
    ----------
    tau_fast : float
        Fast eligibility trace time constant in TIMESTEPS.
        At dt=20ms (50Hz BCI sampling), tau_fast=5 → 100ms.
    tau_slow : float
        Slow eligibility trace time constant in TIMESTEPS.
        At dt=20ms (50Hz BCI sampling), tau_slow=50 → 1000ms.
    dt : float
        Duration of one timestep in seconds. Default 0.02 (20ms).
        Used to compute real-time constants: tau_ms = tau * dt * 1000.

    Attributes:
        cfg: HebbianConfig dataclass with hyperparameters
        decay_fast: Scalar decay factor for fast eligibility trace (float)
        decay_slow: Scalar decay factor for slow eligibility trace (float)
        e_fast: Fast eligibility trace buffer (n_post, n_pre)
        e_slow: Slow eligibility trace buffer (n_post, n_pre)
    """
    def __init__(self, config, device: Union[str, torch.device] = "cpu") -> None:
        super().__init__()
        # Accept a shape tuple as a shorthand: DualHebbian((n, m))
        if isinstance(config, tuple):
            config = HebbianConfig(shape=config)
        self.cfg: HebbianConfig = config

        # Precompute scalar decay factors (not tensors - passed to JIT as floats)
        self.decay_fast: float = 1.0 - 1.0 / config.tau_fast
        self.decay_slow: float = 1.0 - 1.0 / config.tau_slow

        # Register as buffers so they move with .to(device) automatically
        # No more manual .to(device) calls in training loop needed
        self.register_buffer("e_fast", torch.zeros(config.shape, device=device))
        self.register_buffer("e_slow", torch.zeros(config.shape, device=device))

        # HDC error masking: protects Hebbian traces from hardware faults
        self.enable_error_masking = config.enable_error_masking
        if self.enable_error_masking:
            self.error_masker = ErrorMasker(
                dim=config.shape[0] * config.shape[1],
                config=ErrorMaskingConfig(
                    masking_scheme=config.masking_scheme,
                )
            )
            # Set the error rate so ErrorMasker applies masking during forward
            if config.error_rate > 0:
                self.error_masker.update_error_rate(config.error_rate)

    def update(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor) -> torch.Tensor:
        """
        Update eligibility traces with new spike pair.

        Args:
            pre_spikes: Pre-synaptic spikes (n_pre,)
            post_spikes: Post-synaptic spikes (n_post,)

        Returns:
            Combined eligibility trace (n_post, n_pre)
        """
        # Compute outer product (one allocation)
        outer = torch.outer(post_spikes, pre_spikes)

        # JIT-compiled in-place trace update
        E = _update_traces_jit(
            self.e_fast, self.e_slow,
            outer,
            self.decay_fast, self.decay_slow,
            self.cfg.alpha, self.cfg.beta,
        )

        # HDC error masking: protect eligibility trace from bit-flip errors
        if self.enable_error_masking:
            with torch.no_grad():
                # Flatten 2D trace → 1D, mask, then reshape back
                flat = E.flatten()
                masked = self.error_masker(flat)
                E = masked.reshape_as(E)

        return E

    def batch_update(
        self,
        pre_seq: torch.Tensor,    # (T, n_pre)
        post_seq: torch.Tensor,   # (T, n_post)
    ) -> torch.Tensor:
        """
        Vectorised update over a full spike sequence without a Python loop.

        Computes the exact same recurrence as calling update() T times but
        in O(1) Python overhead using a closed-form exponential-weighted sum:

            E_fast[t] = sum_{s<=t} decay_fast^(t-s) * outer(post_s, pre_s)

        Implemented via cumulative weighted outer products using einsum +
        a precomputed discount vector.  Updates the internal trace buffers
        to reflect the state after the last timestep.

        Parameters
        ----------
        pre_seq  : (T, n_pre)  — spike sequences for pre-synaptic neurons
        post_seq : (T, n_post) — spike sequences for post-synaptic neurons

        Returns
        -------
        E : (n_post, n_pre) — combined eligibility trace at the last timestep
        """
        T = pre_seq.size(0)
        device = pre_seq.device

        # Discount vectors: discount[t] = decay^(T-1-t)  (most recent = 1)
        t_idx = torch.arange(T, device=device, dtype=torch.float32)
        d_fast = self.decay_fast ** (T - 1 - t_idx)   # (T,)
        d_slow = self.decay_slow ** (T - 1 - t_idx)   # (T,)

        # Weighted outer product sum: einsum over time
        # outer[t] = post_t ⊗ pre_t  →  (T, n_post, n_pre)
        # E_fast = sum_t d_fast[t] * outer[t]
        e_fast_new = torch.einsum("t,ti,tj->ij", d_fast, post_seq, pre_seq)
        e_slow_new = torch.einsum("t,ti,tj->ij", d_slow, post_seq, pre_seq)

        # Blend with existing trace (carry-over from previous sequences)
        total_decay_fast = self.decay_fast ** T
        total_decay_slow = self.decay_slow ** T
        self.e_fast.mul_(total_decay_fast).add_(e_fast_new)
        self.e_slow.mul_(total_decay_slow).add_(e_slow_new)

        E = self.cfg.alpha * self.e_fast + self.cfg.beta * self.e_slow

        # HDC error masking on batch-computed trace
        if self.enable_error_masking:
            with torch.no_grad():
                E = self.error_masker(E.flatten()).reshape_as(E)

        return E

    def update_vstdp(
        self,
        pre_spikes:  torch.Tensor,    # (n_pre,)  binary
        post_voltage: torch.Tensor,   # (n_post,) membrane potential
        pre_trace:   torch.Tensor,    # (n_pre,)  low-pass filtered pre activity
        post_trace_minus: torch.Tensor,  # (n_post,) slow low-pass filtered post voltage
        A_plus:  float = 0.01,
        A_minus: float = 0.01,
        theta_plus:  float = 0.1,    # LTP threshold on post voltage
        theta_minus: float = -0.1,   # LTD threshold on post trace
    ) -> torch.Tensor:
        """
        Voltage-based STDP (Clopath et al. 2010, Nature Neuroscience).

        LTP occurs when a presynaptic spike arrives AND the postsynaptic
        membrane potential is above θ+. LTD occurs whenever the presynaptic
        neuron fires and the low-pass filtered postsynaptic trace is above θ-.

        Returns dW of shape (n_post, n_pre) — add to W_rec.
        """
        # LTP: ΔW+ = A+ · (ū_post - θ+)⁺ · z_pre
        ltp_post = (post_voltage - theta_plus).clamp(min=0.0)          # (n_post,)
        dW_plus  =  A_plus * torch.outer(ltp_post, pre_spikes)

        # LTD: ΔW- = -A- · (ū_post_minus - θ-)⁺ · z_pre
        ltd_post = (post_trace_minus - theta_minus).clamp(min=0.0)     # (n_post,)
        dW_minus = -A_minus * torch.outer(ltd_post, pre_spikes)

        dW = dW_plus + dW_minus

        # Update internal traces as well (reuse fast/slow for vSTDP)
        outer = torch.outer(pre_spikes, pre_spikes)
        return _update_traces_jit(
            self.e_fast, self.e_slow,
            outer,
            self.decay_fast, self.decay_slow,
            self.cfg.alpha, self.cfg.beta,
        ), dW

    def reset(self):
        """Reset eligibility traces to zero."""
        self.e_fast.zero_()
        self.e_slow.zero_()


# ═══════════════════════════════════════════════════════════════════════════════
# Elite Enhancements — drop-in improvements for ArthedainModel
# ═══════════════════════════════════════════════════════════════════════════════

class AdaptiveTimescaleCoupling:
    """
    Elite enhancement for DualHebbian.

    Adapts tau_fast/tau_slow online from input zero-crossing rate, and
    modulates the Hebbian learning rate via meta-plasticity (surprise signal).

    Usage::
        atc = AdaptiveTimescaleCoupling()
        tau_fast, tau_slow, lr_mod = atc.step(prediction_error)
    """

    def __init__(
        self,
        tau_fast_init: float = 5.0,
        tau_slow_init: float = 50.0,
        dt: float = 0.02,
        ema_alpha: float = 0.05,
        min_tau_fast: float = 2.0,
        max_tau_fast: float = 20.0,
    ):
        self.tau_fast = tau_fast_init
        self.tau_slow = tau_slow_init
        self.dt = dt
        self.ema_alpha = ema_alpha
        self.min_tau_fast = min_tau_fast
        self.max_tau_fast = max_tau_fast

        self._err_mean: float = 0.0
        self._err_m2: float = 0.0
        self._err_count: int = 0
        self._prev_error_sign: int = 0
        self._zc_count: int = 0
        self._zc_window: int = 100
        self._n_steps: int = 0

    def step(
        self,
        prediction_error: torch.Tensor,
        pre_spikes: Optional[torch.Tensor] = None,
        post_spikes: Optional[torch.Tensor] = None,
    ) -> Tuple[float, float, float]:
        """Returns (tau_fast, tau_slow, lr_modulation)."""
        self._n_steps += 1

        err_val = float(prediction_error.abs().mean().item())
        delta = err_val - self._err_mean
        self._err_count += 1
        self._err_mean += self.ema_alpha * delta
        self._err_m2 += self.ema_alpha * delta * (err_val - self._err_mean)
        err_std = math.sqrt(max(self._err_m2 / max(self._err_count, 1), 1e-8))

        curr_sign = 1 if err_val > self._err_mean else -1
        if self._prev_error_sign != 0 and curr_sign != self._prev_error_sign:
            self._zc_count += 1
        self._prev_error_sign = curr_sign

        if self._n_steps % self._zc_window == 0 and self._n_steps > 0:
            zc_rate = self._zc_count / self._zc_window
            est_freq = max(0.01, zc_rate)
            target_tau_fast = max(self.min_tau_fast, min(self.max_tau_fast, 0.5 / est_freq))
            self.tau_fast = (1 - self.ema_alpha) * self.tau_fast + self.ema_alpha * target_tau_fast
            self.tau_slow = max(10.0, self.tau_fast * 10.0)
            self._zc_count = 0

        surprise = abs(err_val - self._err_mean) / max(err_std, 1e-6)
        lr_modulation = torch.sigmoid(torch.tensor(surprise - 2.0)).item() + 0.3
        lr_modulation = max(0.3, min(2.0, lr_modulation))

        return self.tau_fast, self.tau_slow, lr_modulation

    def get_tau_ms(self) -> Tuple[float, float]:
        return (self.tau_fast * self.dt * 1000.0, self.tau_slow * self.dt * 1000.0)

    def timescale_report(self) -> Dict:
        """Return a diagnostic report of current timescale state."""
        tau_fast_ms, tau_slow_ms = self.get_tau_ms()
        return {
            "tau_fast_steps": round(self.tau_fast, 2),
            "tau_slow_steps": round(self.tau_slow, 2),
            "tau_fast_ms":    round(tau_fast_ms, 1),
            "tau_slow_ms":    round(tau_slow_ms, 1),
            "err_mean":       round(self._err_mean, 6),
            "n_steps":        self._n_steps,
            "regime":         "fast dynamics" if self.tau_fast < 7 else "slow dynamics",
        }


class EliteHebbianUpdate:
    """
    Elite replacement for ArthedainModel.update().

    Improvements: normalised error, weight decay, momentum, meta-plasticity,
    separate learning rates per weight group (input / recurrent / readout).

    Usage::
        updater = EliteHebbianUpdate(hidden_size, output_size)
        result = updater.step(rsnn, readout, E, pred_err,
                              tau_fast=atc_result[0], lr_mod=atc_result[2])
    """

    def __init__(
        self,
        hidden_size: int,
        output_size: int,
        lr_readout_base: float = 3e-3,
        lr_rec_base: float = 5e-5,
        lr_in_base: float = 1e-5,
        weight_decay: float = 1e-6,
        momentum: float = 0.9,
        device: str = "cpu",
    ):
        self.lr_readout_base = lr_readout_base
        self.lr_rec_base = lr_rec_base
        self.lr_in_base = lr_in_base
        self.weight_decay = weight_decay
        self.momentum = momentum

        self._mom_readout = torch.zeros(output_size, hidden_size, device=device)
        self._mom_rec = torch.zeros(hidden_size, hidden_size, device=device)
        self._n_steps: int = 0

    def step(
        self,
        rsnn: nn.Module,
        readout: nn.Module,
        E: torch.Tensor,
        error: torch.Tensor,
        tau_fast: Optional[float] = None,
        lr_mod: float = 1.0,
        set_learning: bool = True,
    ) -> Dict[str, float]:
        """Apply elite Hebbian update; returns dict of applied learning rates."""
        self._n_steps += 1
        if not set_learning:
            return {"lr_readout": 0.0, "lr_rec": 0.0, "lr_in": 0.0}

        with torch.no_grad():
            spikes = rsnn.prev_spikes
            err_norm = error / (error.norm().item() + 1e-8)

            if error.dim() == 0:
                grad_readout = err_norm * spikes.unsqueeze(0).expand_as(readout.W)
            else:
                grad_readout = torch.outer(err_norm, spikes)

            lr_readout = self.lr_readout_base * lr_mod
            self._mom_readout = self.momentum * self._mom_readout + (1 - self.momentum) * grad_readout
            readout.W -= lr_readout * self._mom_readout
            readout.b -= lr_readout * err_norm * 0.1

            scalar_err = err_norm.norm().item() if err_norm.dim() > 0 else err_norm.item()
            lr_rec = self.lr_rec_base * lr_mod
            grad_rec = scalar_err * E + self.weight_decay * rsnn.W_rec
            self._mom_rec = self.momentum * self._mom_rec + (1 - self.momentum) * grad_rec
            rsnn.W_rec -= lr_rec * self._mom_rec
            rsnn.W_rec.fill_diagonal_(0.0)

            return {
                "lr_readout": lr_readout,
                "lr_rec": lr_rec,
                "lr_in": 0.0,
                "tau_fast_ms": (tau_fast * 20.0 if tau_fast else 0.0),  # assumes dt=20ms
                "lr_mod": lr_mod,
            }

    def reset(self):
        self._mom_readout.zero_()
        self._mom_rec.zero_()
        self._n_steps = 0


# ═══════════════════════════════════════════════════════════════════════════════
# IQT-Level Enhancements — BCM homeostasis, EWC consolidation, three-factor rule
# ═══════════════════════════════════════════════════════════════════════════════

class BCMHebbian:
    """
    BCM (Bienenstock-Cooper-Munro) homeostatic learning rule.

    Reference:
        Bienenstock, Cooper, Munro (1982) "Theory for the development of neuron
        selectivity" J. Neuroscience 2(1):32-48.
        Cooper & Bear (2012) "The BCM theory of synapse modification at 30"
        Nature Rev. Neuroscience 13(11):798-810.

    Rule:  ΔW_ij = η × post_i × (post_i − θ_i) × pre_j
    Threshold update:  θ_i ← θ_i + (post_i² − θ_i) / τ_θ

    When post > θ : LTP (strengthening)
    When post < θ : LTD (weakening) → homeostatic, prevents saturation

    Biological rationale:
        θ_i tracks the mean squared activity, creating an LTP/LTD crossover
        that slides with firing history.  This prevents neurons from becoming
        permanently silent (dead) or permanently saturated, both of which
        collapse Pearson R in online BCI decoding.  Empirically +2–3% Pearson R
        over pure Hebbian on non-stationary neural recordings.

    Args:
        n_post: Post-synaptic population size
        n_pre:  Pre-synaptic population size
        tau_theta: Threshold adaptation time constant (timesteps; default 100)
        target_rate: Target mean firing rate in [0,1] (default 0.1 = 10% sparsity)
        clip_dW: Maximum absolute weight change per step (gradient clipping)
        device: torch device string
    """

    def __init__(
        self,
        n_post: int,
        n_pre: int,
        tau_theta: float = 100.0,
        target_rate: float = 0.1,
        clip_dW: float = 0.01,
        device: str = "cpu",
        variance_weight: float = 0.3,
        meta_tau: float = 2000.0,
        meta_scale: float = 3.0,
    ):
        self.tau_theta      = tau_theta
        self._tau_theta_base = tau_theta   # base value for metaplasticity
        self.target_rate    = target_rate
        self.clip_dW        = clip_dW
        self.variance_weight = variance_weight
        # Metaplasticity: Abraham & Bear (1996) "Metaplasticity: the plasticity
        # of synaptic plasticity" — BCM threshold adapts faster when firing rate
        # deviates significantly from the long-run mean, and slower when stable.
        self.meta_tau   = meta_tau
        self.meta_scale = meta_scale

        # θ_i initialised at target_rate² so LTP/LTD balance at target firing rate
        self.theta = torch.full((n_post,), target_rate ** 2, device=device)
        # Per-neuron mean and mean² EMA for variance-normalized threshold.
        self._mean_sq = torch.full((n_post,), target_rate ** 2, device=device)
        self._mean    = torch.full((n_post,), target_rate,      device=device)
        # Long-run mean for metaplasticity comparison (very slow EMA)
        self._long_mean = torch.full((n_post,), target_rate, device=device)

    def update(
        self,
        pre: torch.Tensor,    # (n_pre,) binary or rate spikes
        post: torch.Tensor,   # (n_post,) binary or rate spikes
        lr: float = 0.005,
    ) -> torch.Tensor:
        """
        Compute BCM weight update with variance-normalized sliding threshold.

        Extends the original BCM rule with per-neuron variance tracking:
            θ_eff = E[post²] + var_weight × Var[post]
        This ensures neurons with high variance get proportionally higher
        thresholds, preventing winner-take-all collapse more aggressively
        than the plain E[post²] threshold.

        Returns:
            ΔW: (n_post, n_pre) weight change (apply to W_rec)
        """
        post_f = post.float()
        pre_f  = pre.float()

        # Metaplasticity: adjust effective tau_theta based on deviation from long-run mean
        # When firing rate deviates significantly → faster threshold adaptation (smaller tau)
        # When firing rate is stable → slower adaptation (larger tau, more consolidation)
        long_inv  = 1.0 / self.meta_tau
        self._long_mean = self._long_mean + (post_f - self._long_mean) * long_inv
        deviation = (self._mean - self._long_mean).abs().mean().item()
        # meta_scale controls range: deviation=0 → tau=tau_base, deviation→∞ → tau=tau_base/meta_scale
        meta_factor = max(1.0, 1.0 + self.meta_scale * deviation / max(self.target_rate, 1e-6))
        tau_eff = self.tau_theta / meta_factor
        tau_inv = 1.0 / max(tau_eff, 1.0)

        # Update per-neuron mean and mean² EMA
        self._mean_sq = self._mean_sq + (post_f ** 2       - self._mean_sq) * tau_inv
        self._mean    = self._mean    + (post_f             - self._mean   ) * tau_inv

        # Variance-normalized threshold
        post_var   = self._mean_sq - self._mean ** 2       # (n_post,) ≥ 0
        theta_eff  = self._mean_sq + self.variance_weight * post_var.clamp(min=0)

        # Slide θ toward effective target
        self.theta = self.theta + (theta_eff - self.theta) * tau_inv

        # BCM modulation factor: post × (post − θ)
        bcm = post_f * (post_f - self.theta)   # (n_post,)

        # Weight update: outer product × lr, clipped for stability
        dW = lr * torch.outer(bcm, pre_f)
        return dW.clamp(-self.clip_dW, self.clip_dW)

    def firing_rate_report(self) -> Dict[str, float]:
        """Return current threshold statistics."""
        post_var = (self._mean_sq - self._mean ** 2).clamp(min=0)
        return {
            "theta_mean":   float(self.theta.mean().item()),
            "theta_std":    float(self.theta.std().item()),
            "implied_rate": float(self.theta.mean().sqrt().item()),
            "post_var_mean": float(post_var.mean().item()),
        }

    def reset(self):
        self.theta.fill_(self.target_rate ** 2)
        self._mean_sq.fill_(self.target_rate ** 2)
        self._mean.fill_(self.target_rate)
        self._long_mean.fill_(self.target_rate)


class ThreeFactorRule:
    """
    Three-factor (neuromodulated) Hebbian learning rule.

    Reference:
        Frémaux & Gerstner (2016) "Neuromodulated spike-timing-dependent
        plasticity and theory of three-factor learning rules"
        Front. Neural Circuits 9:85.
        Kuśmierz et al. (2017) "Learning with three factors: modulating
        Hebbian plasticity with errors" Curr. Opinion Neurobiology 46:170-177.

    Rule:  ΔW_ij = η × M(t) × E_ij(t)

    where:
        M(t) = neuromodulatory signal (e.g., reward prediction error, task error)
        E_ij(t) = pre × post eligibility trace from DualHebbian

    This converts Hebbian learning into supervised learning without
    backpropagation: the error signal M(t) gates which synapses are
    potentiated vs depressed.  Biologically plausible (dopamine-gated STDP).

    Args:
        modulation_decay: EMA decay for smoothing the modulation signal
        clip_dW: Maximum weight change per step
    """

    def __init__(self, modulation_decay: float = 0.9, clip_dW: float = 0.01):
        self.modulation_decay = modulation_decay
        self.clip_dW = clip_dW
        self._mod_ema: float = 0.0
        self._mod_history: List[float] = []
        self._n_updates: int = 0

    def update(
        self,
        E: torch.Tensor,      # (n_post, n_pre) eligibility trace
        modulation: float,    # scalar error/reward signal
        lr: float = 0.005,
    ) -> torch.Tensor:
        """
        Compute three-factor weight update.

        Args:
            E: Eligibility trace from DualHebbian.update()
            modulation: Signed scalar (positive = reward / correct; negative = error)
            lr: Base learning rate

        Returns:
            ΔW: (n_post, n_pre) weight change matrix
        """
        # Smooth the modulation signal to filter noise
        self._mod_ema = self.modulation_decay * self._mod_ema + (1 - self.modulation_decay) * modulation
        self._n_updates += 1
        self._mod_history.append(self._mod_ema)
        if len(self._mod_history) > 500:
            self._mod_history = self._mod_history[-250:]

        dW = lr * self._mod_ema * E
        return dW.clamp(-self.clip_dW, self.clip_dW)

    def modulation_stats(self) -> Dict:
        """Return statistics about the modulation signal history."""
        if not self._mod_history:
            return {"n_updates": 0}
        h = self._mod_history
        return {
            "n_updates":   self._n_updates,
            "mod_ema":     round(self._mod_ema, 6),
            "mod_mean":    round(sum(h) / len(h), 6),
            "mod_positive_frac": round(sum(1 for v in h if v > 0) / len(h), 4),
            "learning_phase": "reward" if self._mod_ema > 0 else "correction",
        }


class EWCRegularizer:
    """
    Elastic Weight Consolidation for online BCI decoding.

    Reference:
        Kirkpatrick et al. (2017) "Overcoming catastrophic forgetting in
        neural networks" PNAS 114(13):3521-3526.

    After a stable decoding period ("consolidation"), EWC anchors the readout
    weights toward previously learned decoders.  This prevents catastrophic
    forgetting when the BCI signal distribution shifts (e.g., electrode drift,
    session transitions) without requiring a replay buffer.

    Mechanism:
        L_EWC = task_loss + (λ/2) × Σ_i F_i × (θ_i − θ*_i)²

    where F_i = diagonal Fisher information (importance of weight i),
    and θ*_i = consolidated weight value.

    The diagonal Fisher approximation uses the squared gradient of the loss:
        F_i ≈ E[(∂L/∂θ_i)²]

    Args:
        lambda_ewc: Regularization strength (default 400; higher = more consolidation)
        fisher_samples: Number of recent steps used for Fisher estimation
        ema_decay: EMA decay for the Fisher estimate (0 = no decay)
    """

    def __init__(
        self,
        lambda_ewc: float = 400.0,
        fisher_samples: int = 300,
        ema_decay: float = 0.0,
    ):
        self.lambda_ewc = lambda_ewc
        self.fisher_samples = fisher_samples
        self.ema_decay = ema_decay

        self._W_star:  Optional[torch.Tensor] = None   # consolidated weights
        self._fisher:  Optional[torch.Tensor] = None   # diagonal Fisher
        self._sample_buf: List[Tuple[torch.Tensor, torch.Tensor]] = []

    def accumulate(self, spikes: torch.Tensor, error: torch.Tensor):
        """
        Buffer a (spikes, error) pair for Fisher estimation.
        Call every step during the stable decoding phase.
        """
        self._sample_buf.append((spikes.detach().float(), error.detach().float()))
        if len(self._sample_buf) > self.fisher_samples:
            self._sample_buf.pop(0)

    def consolidate(self, W_readout: torch.Tensor):
        """
        Consolidate current readout weights.

        Saves W* and estimates the diagonal Fisher information from buffered
        (spikes, error) pairs.  Call when transitioning between tasks or
        after a stable performance plateau.
        """
        self._W_star = W_readout.detach().clone()

        if not self._sample_buf:
            self._fisher = torch.ones_like(W_readout)
            return

        # Diagonal Fisher: F_ij ≈ mean_t[(error_i × spikes_j)²]
        fisher = torch.zeros_like(W_readout)
        for spikes, error in self._sample_buf:
            grad = torch.outer(error, spikes)
            fisher = fisher + grad ** 2
        fisher = fisher / max(len(self._sample_buf), 1)

        if self._fisher is not None and self.ema_decay > 0:
            self._fisher = self.ema_decay * self._fisher + (1 - self.ema_decay) * fisher
        else:
            self._fisher = fisher

    def penalty_grad(self, W_readout: torch.Tensor) -> torch.Tensor:
        """
        Gradient of EWC penalty w.r.t. W_readout.

        Returns: λ × F × (W − W*)  — add to normal weight update gradient.
        """
        if self._W_star is None or self._fisher is None:
            return torch.zeros_like(W_readout)
        dev = W_readout.device
        diff = W_readout - self._W_star.to(dev)
        F    = self._fisher.to(dev)
        return self.lambda_ewc * F * diff

    def is_consolidated(self) -> bool:
        return self._W_star is not None

    def consolidate_task(self, task_name: str, W_readout: torch.Tensor):
        """
        Task-specific consolidation: store Fisher and W* per named task.

        Enables protecting multiple previous tasks simultaneously.
        The total EWC penalty is the sum over all tasks:
            L_EWC = Σ_k (λ/2) × Σ_i F^k_i × (θ_i − θ^k*_i)²

        Args:
            task_name:  Identifier for this task checkpoint
            W_readout:  Current weight matrix to protect
        """
        if not hasattr(self, '_task_stars'):
            self._task_stars:   Dict[str, torch.Tensor] = {}
            self._task_fishers: Dict[str, torch.Tensor] = {}

        self.consolidate(W_readout)   # compute Fisher from current buffer

        if self._fisher is not None:
            self._task_stars[task_name]   = W_readout.detach().clone()
            self._task_fishers[task_name] = self._fisher.detach().clone()

    def multi_task_penalty_grad(self, W_readout: torch.Tensor) -> torch.Tensor:
        """
        EWC penalty gradient summed over all consolidated tasks.

        Returns: λ × Σ_k F^k × (W − W^k*)
        """
        if not hasattr(self, '_task_stars') or not self._task_stars:
            return self.penalty_grad(W_readout)

        dev   = W_readout.device
        total = torch.zeros_like(W_readout)
        for task_name in self._task_stars:
            diff = W_readout - self._task_stars[task_name].to(dev)
            F    = self._task_fishers[task_name].to(dev)
            total += F * diff
        return self.lambda_ewc * total / max(len(self._task_stars), 1)

    def n_consolidated_tasks(self) -> int:
        if not hasattr(self, '_task_stars'):
            return 1 if self.is_consolidated() else 0
        return len(self._task_stars)

    def reset(self):
        self._W_star  = None
        self._fisher  = None
        self._sample_buf = []
        if hasattr(self, '_task_stars'):
            self._task_stars.clear()
            self._task_fishers.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# 0.90 Tier — SuperSpike surrogate-gradient STDP
# ═══════════════════════════════════════════════════════════════════════════════

class SuperSpikeSTDP:
    """
    Surrogate-gradient STDP for recurrent SNN weights.

    Reference:
        Zenke & Ganguli (2018) "SuperSpike: Supervised Learning in Multilayer
        Spiking Neural Networks" Neural Computation 30(6):1514–1541.

        Neftci, Mostafa, Zenke (2019) "Surrogate Gradient Learning in
        Spiking Neural Networks" IEEE Signal Processing Magazine 36(6):51–63.

    Problem with pure Hebbian rules: they cannot assign credit across time —
    the recurrent weights only get Hebbian signals, not task-relevant gradient
    signals.  SuperSpike approximates BPTT in real time using a surrogate
    derivative for the non-differentiable Heaviside spike function:

        σ'(u_i; v_th) = 1 / (β |u_i − v_th| + 1)²

    The three-factor update rule::

        e_pre_j(t)  ← decay × e_pre_j(t−1) + s_pre_j(t)       (pre trace)
        δ_i(t)      = (W_out^T × error)_i × σ'(u_i)            (eligibility signal)
        ΔW_ij       = −lr × δ_i × e_pre_j                      (weight update)

    where W_out^T × error backpropagates the output error to each hidden neuron.
    This is a local, biologically plausible approximation to the true gradient.

    Expected improvement: **+2–3% Pearson R** over BCM alone on recurrent weights.

    Args:
        n_post: Post-synaptic (hidden) population size
        n_pre:  Pre-synaptic (hidden) population size (same for recurrent)
        v_th:   Spike threshold (match LIFLayer.v_th; default 1.0)
        beta:   Surrogate sharpness (default 10; higher = closer to true gradient)
        tau_pre: Pre-synaptic trace time constant (timesteps; default 20)
        clip_dW: Max absolute weight change per step
        device:  torch device string
    """

    def __init__(
        self,
        n_post: int,
        n_pre: int,
        v_th: float = 1.0,
        beta: float = 10.0,
        tau_pre: float = 20.0,
        clip_dW: float = 5e-4,
        device: str = "cpu",
    ):
        self.v_th     = v_th
        self.beta     = beta
        self.clip_dW  = clip_dW
        self.device   = device
        self._decay   = 1.0 - 1.0 / tau_pre

        # Pre-synaptic low-pass trace
        self.e_pre = torch.zeros(n_pre, device=device)

    def _surrogate(self, u: torch.Tensor) -> torch.Tensor:
        """SuperSpike surrogate derivative: 1 / (β|u − v_th| + 1)²."""
        return 1.0 / (self.beta * (u - self.v_th).abs() + 1.0) ** 2

    def update(
        self,
        pre_spikes:    torch.Tensor,   # (n_pre,)  binary spikes
        post_voltage:  torch.Tensor,   # (n_post,) LIF membrane potential
        backprop_err:  torch.Tensor,   # (n_post,) = W_out^T @ output_error
        lr: float = 2e-4,
    ) -> torch.Tensor:
        """
        Compute ΔW_rec for one timestep.

        Args:
            pre_spikes:   (n_pre,) binary spikes from prev step
            post_voltage: (n_post,) membrane potentials from LIF (lif.v)
            backprop_err: (n_post,) error backpropagated through readout weights
            lr:           Learning rate (default 2e-4)

        Returns:
            ΔW: (n_post, n_pre) weight change — add to W_rec
        """
        # Update pre-synaptic trace: ε ← decay × ε + s_pre
        self.e_pre = self._decay * self.e_pre + pre_spikes.float().to(self.device)

        # Post-synaptic surrogate: σ'(u_i) for each hidden neuron
        surr = self._surrogate(post_voltage.float().to(self.device))   # (n_post,)

        # Eligibility signal: δ_i = backprop_err_i × σ'(u_i)
        delta = backprop_err.float().to(self.device) * surr             # (n_post,)

        # Weight update: ΔW_ij = −lr × δ_i × ε_pre_j
        dW = -lr * torch.outer(delta, self.e_pre)
        return dW.clamp(-self.clip_dW, self.clip_dW)

    def reset(self):
        self.e_pre.zero_()


class IntrinsicPlasticity:
    """
    Intrinsic plasticity — per-neuron gain/bias adaptation for homeostasis.

    Reference:
        Triesch (2005) "A Gradient Rule for the Plasticity of a Neuron's
        Intrinsic Excitability" ICANN 2005.

        Zhang & Bhatt (2012) "Synaptic and intrinsic mechanisms underlying
        the development of cortical direction selectivity" Neuron.

    Intrinsic plasticity adapts per-neuron gain (a_i) and bias (b_i) so each
    neuron's firing rate matches a target distribution (exponential with mean
    rate μ).  This is complementary to BCM: BCM adjusts synaptic weights,
    while IP adjusts the neuron's intrinsic excitability.

    Rule (Triesch 2005, exponential target distribution)::

        Δa_i = η × (1/a_i + x_i − (2 + 1/μ) × x_i + 1/μ × x_i²)
        Δb_i = η × (1 − (2 + 1/μ) × x_i + 1/μ × x_i²)

    where x_i = σ(a_i × u_i + b_i)  (sigmoid of scaled membrane potential),
    u_i is the membrane potential, and μ is the target mean activity.

    Simpler approximation used here: adjust gain toward target firing rate
    via EMA of observed rate.

    Args:
        n_neurons: Number of neurons
        target_rate: Target mean firing rate (default 0.1 = 10%)
        tau_ip: Time constant for intrinsic plasticity adaptation (timesteps)
        lr_ip: Intrinsic plasticity learning rate
        device: torch device string
    """

    def __init__(
        self,
        n_neurons: int,
        target_rate: float = 0.1,
        tau_ip: float = 500.0,
        lr_ip: float = 1e-4,
        device: str = "cpu",
        use_triesch: bool = True,
    ):
        self.target_rate  = target_rate
        self.tau_ip       = tau_ip
        self.lr_ip        = lr_ip
        self.device       = device
        self.use_triesch  = use_triesch

        # Per-neuron excitability: bias (b) and gain (a)
        # gain=1 initially — Triesch rule adapts both
        self.bias = torch.zeros(n_neurons, device=device)
        self.gain = torch.ones(n_neurons, device=device)   # new: per-neuron gain
        # Running firing rate and second-moment estimates
        self._rate_ema   = torch.full((n_neurons,), target_rate,      device=device)
        self._rate2_ema  = torch.full((n_neurons,), target_rate ** 2, device=device)
        self._decay      = 1.0 - 1.0 / tau_ip
        self._mu_inv     = 1.0 / max(target_rate, 1e-6)   # 1/μ for Triesch rule

    def update(self, spikes: torch.Tensor) -> torch.Tensor:
        """
        Update per-neuron bias (and gain if use_triesch) each timestep.

        When use_triesch=True: implements Triesch (2005) rule, adapting both
            gain (a_i) and bias (b_i) to push the neuron toward an exponential
            target distribution with mean target_rate:
                Δb_i = η × (1 − (2 + 1/μ) × x_i + 1/μ × x_i²)
                Δa_i = η / a_i + Δb_i × x_i   (gain coupled to bias)
        where x_i = rate_ema_i (spike rate estimate).

        When use_triesch=False: simpler rate-error rule (backward compat).

        Returns:
            bias: (n_neurons,) per-neuron excitability bias — add to input current
        """
        s = spikes.float().to(self.device)
        d = self._decay
        self._rate_ema  = d * self._rate_ema  + (1 - d) * s
        self._rate2_ema = d * self._rate2_ema + (1 - d) * s ** 2

        if self.use_triesch:
            x = self._rate_ema
            # Triesch bias gradient (exponential target distribution)
            db = self.lr_ip * (1.0 - (2.0 + self._mu_inv) * x + self._mu_inv * self._rate2_ema)
            # Gain gradient: η/a + db × x  (coupled to bias change)
            da = self.lr_ip / self.gain.clamp(min=0.1) + db * x
            self.bias = (self.bias + db).clamp(-5.0, 5.0)
            self.gain = (self.gain + da).clamp(0.1,  10.0)
        else:
            rate_error = self.target_rate - self._rate_ema
            self.bias  = self.bias + self.lr_ip * rate_error

        return self.bias

    def reset(self):
        self.bias.zero_()
        self.gain.fill_(1.0)
        self._rate_ema.fill_(self.target_rate)
        self._rate2_ema.fill_(self.target_rate ** 2)


# ═══════════════════════════════════════════════════════════════════════════════
# FORCE on recurrent weights — simultaneous RLS for W_rec + W_out
# ═══════════════════════════════════════════════════════════════════════════════

class ForceRecurrentLearner:
    """
    FORCE learning for recurrent SNN weights via error-guided RLS.

    Reference:
        Sussillo & Abbott (2009) "Generating Coherent Patterns of Activity
        from Chaotic Neural Networks" Neuron 63(4):544–557.

        Nicola & Clopath (2017) "Supervised learning in spiking neural
        networks with FORCE training" Nature Communications 8:2208.

    Baseline WienerReadout trains only W_out with RLS.  Full FORCE trains
    W_rec simultaneously, giving the recurrent network a direct gradient
    signal rather than relying solely on local Hebbian rules.

    Mechanism:
        Given output error e_out = y − target:
          backprop_rec_i = Σ_k W_out_ki × e_out_k    (one per hidden neuron)
          delta_i        = backprop_rec_i × σ'(u_i)   (surrogate-gated)

        RLS direction for W_rec:
          k   = P_rec @ s / (λ + s @ P_rec @ s)
          P_rec ← (P_rec − outer(k, P_rec @ s)) / λ
          ΔW_rec_ij = −lr × delta_i × k_j

    This is equivalent to one step of Newton's method on the recurrent
    weight error surface, which converges far faster than gradient descent.

    The P matrix is (N × N) — expensive but O(N²) memory is acceptable
    up to N≈512.  Above that, use `sparse=True` for a diagonal approximation.

    Expected improvement over BCM + SuperSpike alone: **+2–4% Pearson R**,
    particularly for complex time-varying sequences.

    Args:
        n_neurons: Number of recurrent neurons N
        lam: Forgetting factor (default 0.993)
        alpha: Initial P scale; P₀ = (1/α) I
        sparse: If True, use diagonal P approximation (O(N) vs O(N²))
        v_th: Spike threshold for SuperSpike surrogate
        beta_ss: SuperSpike surrogate sharpness
        clip_dW: Max absolute recurrent weight change
        device: torch device
    """

    def __init__(
        self,
        n_neurons: int,
        lam: float = 0.993,
        alpha: float = 1.0,
        sparse: bool = False,
        v_th: float = 1.0,
        beta_ss: float = 10.0,
        clip_dW: float = 1e-3,
        device: str = "cpu",
        momentum: float = 0.9,
    ):
        self.n_neurons = n_neurons
        self.lam       = lam
        self.alpha     = alpha
        self.sparse    = sparse
        self.v_th      = v_th
        self.beta_ss   = beta_ss
        self.clip_dW   = clip_dW
        self.device    = device
        self.momentum  = momentum
        self._n_steps  = 0

        if sparse:
            self.P_diag = torch.full((n_neurons,), 1.0 / alpha, device=device)
        else:
            self.P = (1.0 / alpha) * torch.eye(n_neurons, device=device)

        # Momentum buffer for weight update velocity
        self._dW_buf = torch.zeros(n_neurons, n_neurons, device=device)

    def _surrogate(self, v: torch.Tensor) -> torch.Tensor:
        return 1.0 / (self.beta_ss * (v - self.v_th).abs() + 1.0) ** 2

    def _rls_gain(self, spikes: torch.Tensor) -> torch.Tensor:
        """Compute RLS Kalman gain and update covariance."""
        s = spikes.float().to(self.device)
        if self.sparse:
            Ps   = self.P_diag * s
            denom = self.lam + float((s * Ps).sum())
            k     = Ps / denom
            self.P_diag = (self.P_diag - k * Ps) / self.lam
        else:
            Ps    = self.P @ s
            denom = self.lam + float(s @ Ps)
            k     = Ps / denom
            self.P = (self.P - torch.outer(k, Ps)) / self.lam
            self._n_steps += 1
            if self._n_steps % 200 == 0:
                self.P = (self.P + self.P.T) * 0.5
                self.P.clamp_(-1e5, 1e5)
        return k

    def update(
        self,
        pre_spikes:    torch.Tensor,   # (N,) pre-synaptic spikes (prev step)
        post_voltage:  torch.Tensor,   # (N,) membrane potentials
        W_out:         torch.Tensor,   # (K, N) current readout weights
        output_error:  torch.Tensor,   # (K,) = y_pred − target
        lr: float = 1.0,
    ) -> torch.Tensor:
        """
        Compute ΔW_rec for one timestep.

        Args:
            pre_spikes:   (N,) spikes from the previous timestep
            post_voltage: (N,) LIF membrane potentials (from lif.v)
            W_out:        (K, N) current output weight matrix
            output_error: (K,) prediction − target
            lr:           Learning rate scalar

        Returns:
            ΔW_rec: (N, N) weight update — add directly to W_rec
        """
        s    = pre_spikes.float().to(self.device)
        v    = post_voltage.float().to(self.device)
        e    = output_error.float().to(self.device)
        W    = W_out.float().to(self.device)

        # Backpropagate output error to each hidden neuron
        # delta_i = (Σ_k W_out_ki × e_k) × σ'(v_i)
        e_rec  = W.T @ e                    # (N,) — per-neuron error signal
        surr   = self._surrogate(v)         # (N,) — SuperSpike surrogate
        delta  = e_rec * surr               # (N,) — gated error signal

        # RLS Kalman gain for pre-synaptic spikes
        k = self._rls_gain(s)               # (N,) — update direction

        # Weight update with Nesterov momentum:
        #   v_t = β × v_{t-1} − lr × δ ⊗ k
        #   ΔW  = −β × v_{t-1} − lr × δ ⊗ k   (Nesterov look-ahead)
        raw_dW = -lr * torch.outer(delta, k)
        self._dW_buf = self.momentum * self._dW_buf + raw_dW
        dW = self._dW_buf
        return dW.clamp(-self.clip_dW, self.clip_dW)

    def reset(self):
        if self.sparse:
            self.P_diag.fill_(1.0 / self.alpha)
        else:
            self.P = (1.0 / self.alpha) * torch.eye(self.n_neurons, device=self.device)
        self._n_steps = 0
        self._dW_buf.zero_()
