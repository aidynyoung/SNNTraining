import torch
from dataclasses import dataclass
from typing import Optional, Callable, Iterator, Tuple

from hdc.ecc import HDCCorrector, ECCConfig


@dataclass
class TrainerConfig:
    lr_readout:    float = 1e-3
    lr_recurrent:  float = 5e-5
    device:        Optional[str] = None
    warmup_steps:  int   = 100
    grad_clip_norm: float = 1.0
    mode:          str   = "supervised"   # "supervised" | "reward" | "self_supervised"
    log_every:     int   = 200

    # Spike rate regularisation (Zenke et al. 2017; Eshraghian et al. 2023)
    lambda_rate:   float = 0.01
    target_rate:   float = 0.10

    # Synaptic homeostasis (Turrigiano 2008 Cell)
    use_homeostasis: bool = True
    homeo_every:   int   = 200
    homeo_bound:   float = 0.05

    # Option A: HDC closed-loop feedback
    # When the pipeline detects a distribution shift (low HDC similarity),
    # it provides an lr_multiplier that scales the effective learning rate.
    enable_hdc_feedback: bool = False

    # Option B: HDC error correction (weight repair via ECC)
    # Uses the HDC associative memory to detect and correct corrupted weights.
    # Replaces the heuristic LR threshold with PI control (Saponati et al. 2026).
    enable_hdc_ecc: bool = False
    ecc_similarity_threshold: float = 0.3
    ecc_correction_strength: float = 0.1
    ecc_kp: float = 0.5   # Proportional gain
    ecc_ki: float = 0.1   # Integral gain


class OnlineTrainer:
    """Online streaming trainer for Arthedain SNN."""

    def __init__(self, rsnn, readout, hebbian,
                 lr_readout=1e-3, lr_recurrent=5e-5, device=None,
                 warmup_steps=100, grad_clip_norm=1.0,
                 config: Optional[TrainerConfig] = None):

        # Accept TrainerConfig as 4th positional arg
        if isinstance(lr_readout, TrainerConfig):
            config = lr_readout
            lr_readout = config.lr_readout

        cfg = config or TrainerConfig(
            lr_readout=lr_readout,
            lr_recurrent=lr_recurrent,
            warmup_steps=warmup_steps,
            grad_clip_norm=grad_clip_norm,
        )
        self.cfg          = cfg
        self.rsnn         = rsnn
        self.readout      = readout
        self.hebbian      = hebbian
        self.lr_readout   = cfg.lr_readout
        self.lr_recurrent = cfg.lr_recurrent
        self.warmup_steps = cfg.warmup_steps
        self.grad_clip_norm = cfg.grad_clip_norm
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.step_count = 0
        self._losses: list = []

        # HDC feedback: external lr multiplier (set by pipeline each step)
        self._lr_multiplier: float = 1.0

        # HDC error correction (ECC) for weight repair
        self._hdc_corrector: Optional[HDCCorrector] = None
        self._hdc_encoder = None
        self._assoc_memory = None
        self._ecc_info: dict = {}

    # ------------------------------------------------------------------
    # _step alias (tests use trainer._step)
    # ------------------------------------------------------------------

    @property
    def _step(self) -> int:
        return self.step_count

    # ------------------------------------------------------------------
    # HDC feedback: accept lr multiplier from pipeline
    # ------------------------------------------------------------------

    def set_lr_multiplier(self, mult: float) -> None:
        """Set learning rate multiplier from HDC similarity feedback."""
        self._lr_multiplier = max(0.1, mult)  # clamp to prevent 0

    # ------------------------------------------------------------------
    # HDC error correction setup
    # ------------------------------------------------------------------

    def setup_hdc_ecc(self, hdc_encoder, assoc_memory) -> None:
        """Configure HDC error correction for weight repair.

        Args:
            hdc_encoder: HDCEncoder instance (maps spikes → hypervectors)
            assoc_memory: AssocMemory instance (stores class prototypes)
        """
        self._hdc_encoder = hdc_encoder
        self._assoc_memory = assoc_memory
        # AssocMemory stores dim, n_classes, mode directly
        hdc_dim = assoc_memory.dim
        n_classes = assoc_memory.n_classes
        mode = assoc_memory.mode
        self._hdc_corrector = HDCCorrector(ECCConfig(
            hdc_dim=hdc_dim,
            n_classes=n_classes,
            mode=mode,
            similarity_threshold=self.cfg.ecc_similarity_threshold,
            correction_strength=self.cfg.ecc_correction_strength,
            use_pi_control=True,
            kp=self.cfg.ecc_kp,
            ki=self.cfg.ecc_ki,
            correction_cooldown=20,
            max_correction_norm=0.05,
        ))

    # ------------------------------------------------------------------
    # Learning rate warmup (with optional HDC multiplier)
    # ------------------------------------------------------------------

    def _get_lr_with_warmup(self, base_lr: float) -> float:
        lr = base_lr
        if self.step_count < self.warmup_steps:
            lr = base_lr * (self.step_count / max(self.warmup_steps, 1))
        if self.cfg.enable_hdc_feedback:
            lr *= self._lr_multiplier
        return lr

    # ------------------------------------------------------------------
    # Core step
    # ------------------------------------------------------------------

    def step(self, x, target=None, reward=None):
        """One online training step.

        Args:
            x      : input tensor
            target : supervised target (required in 'supervised' mode)
            reward : scalar reward (used in 'reward' mode)

        Returns:
            (y_pred, error)
        """
        self.step_count += 1

        x = x.to(self.device)
        spikes  = self.rsnn.forward(x)
        y_pred  = self.readout.forward(spikes)

        # HDC error correction: repair corrupted weights before learning update
        if (self.cfg.enable_hdc_ecc and self._hdc_corrector is not None
                and self._hdc_encoder is not None
                and self._assoc_memory is not None):
            true_label = None
            if target is not None:
                if target.dim() == 0:
                    true_label = int(target.item())
                elif target.dim() == 1 and target.size(0) == 1:
                    true_label = int(target.item())
                else:
                    # One-hot: convert to label
                    true_label = int(target.argmax().item())
            corrected_W, strength, info = self._hdc_corrector.repair_weights(
                self.rsnn.W_rec, spikes, self._hdc_encoder,
                self._assoc_memory, true_label=true_label)
            if info["corrected"]:
                self.rsnn.W_rec = corrected_W
            self._ecc_info = info

        # Compute error signal
        mode = getattr(self.cfg, 'mode', 'supervised')
        if mode == 'reward' or reward is not None:
            r = float(reward) if reward is not None else 0.0
            baseline = getattr(self, '_reward_baseline', 0.0)
            error = y_pred * (baseline - r)           # REINFORCE-style
            self._reward_baseline = 0.9 * baseline + 0.1 * r
        elif target is not None:
            target = target.to(self.device)
            # Use softmax + cross-entropy gradient for classification
            # y_pred_softmax = softmax(y_pred)
            # error = y_pred_softmax - target  (cross-entropy gradient)
            y_pred_exp = torch.exp(y_pred - y_pred.max())
            y_pred_softmax = y_pred_exp / y_pred_exp.sum()
            error = y_pred_softmax - target
        else:
            error = torch.zeros_like(y_pred)

        # Eligibility trace
        E = self.hebbian.update(spikes, spikes)

        lr_ro  = self._get_lr_with_warmup(self.cfg.lr_readout)
        lr_rec = self._get_lr_with_warmup(self.cfg.lr_recurrent)

        with torch.no_grad():
            # Readout update (delta rule) with gradient clipping
            readout_update = lr_ro * torch.outer(error, spikes)
            norm = readout_update.norm()
            if norm > self.cfg.grad_clip_norm:
                readout_update = readout_update * (self.cfg.grad_clip_norm / norm)
            if hasattr(self.readout, 'W'):
                self.readout.W.add_(readout_update)
                if hasattr(self.readout, 'b'):
                    self.readout.b.add_(lr_ro * error)

            # Recurrent update (Hebbian)
            rec_update = lr_rec * E
            norm = rec_update.norm()
            if norm > self.cfg.grad_clip_norm:
                rec_update = rec_update * (self.cfg.grad_clip_norm / norm)
            if hasattr(self.rsnn, 'W_rec'):
                self.rsnn.W_rec.add_(rec_update)

        # Spike rate regularisation
        if self.cfg.lambda_rate > 0:
            actual_rate = spikes.mean()
            rate_error  = actual_rate - self.cfg.target_rate
            with torch.no_grad():
                self.readout.W.mul_(
                    1.0 - lr_ro * self.cfg.lambda_rate * rate_error.sign().item()
                )

        # Synaptic homeostasis
        if self.cfg.use_homeostasis and hasattr(self.rsnn, 'W_rec'):
            if self.step_count % self.cfg.homeo_every == 0 and self.step_count > 0:
                actual = self.rsnn.lif.get_firing_rates().mean().clamp(min=1e-6)
                scale  = float((self.cfg.target_rate / actual).clamp(
                    1.0 - self.cfg.homeo_bound,
                    1.0 + self.cfg.homeo_bound
                ))
                with torch.no_grad():
                    self.rsnn.W_rec.mul_(scale)

        loss = float(error.pow(2).mean().item())
        self._losses.append(loss)
        if len(self._losses) > 1000:
            self._losses.pop(0)

        return y_pred, error

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def get_metrics(self) -> dict:
        """Return a summary dict of training statistics."""
        losses = self._losses or [0.0]
        metrics = {
            "steps":     self.step_count,
            "loss_mean": sum(losses) / len(losses),
            "loss_last": losses[-1] if losses else 0.0,
            "lr_mult":   self._lr_multiplier,
        }
        if self._hdc_corrector is not None:
            metrics["ecc"] = self._hdc_corrector.get_stats()
            metrics["ecc_last"] = self._ecc_info
        return metrics

    # ------------------------------------------------------------------
    # Stream runner
    # ------------------------------------------------------------------

    def run_stream(
        self,
        stream: Iterator[Tuple[torch.Tensor, torch.Tensor]],
        callback: Optional[Callable] = None,
        n_steps: Optional[int] = None,
    ) -> dict:
        """Run the training loop over a data stream.

        Args:
            stream   : iterable yielding (x, y) pairs
            callback : optional fn(step, y_pred, error) called every log_every steps
            n_steps  : stop after this many steps (None = exhaust stream)

        Returns:
            metrics dict
        """
        log_every = getattr(self.cfg, 'log_every', 200)
        for i, (x, y) in enumerate(stream):
            if n_steps is not None and i >= n_steps:
                break
            y_pred, error = self.step(x, target=y)
            if callback is not None and self.step_count % log_every == 0:
                callback(self.step_count, y_pred, error)

        return self.get_metrics()
