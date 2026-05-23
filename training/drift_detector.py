"""
Drift Detection and Adaptive Mode Switching
===========================================
Online adaptation to concept drift for manufacturing-critical applications.

Implements:
- Drift detector gating slow updates
- True Online mode switching (paper Section 3.3)
- Health metrics logging
"""

import torch
from typing import Optional, Dict, Callable
from dataclasses import dataclass
from collections import deque
import time


@dataclass
class DriftConfig:
    """Configuration for drift detection and adaptation."""
    # Drift detection thresholds
    error_spike_threshold: float = 3.0      # Sigma multiplier for drift alert
    error_baseline_window: int = 50         # Timesteps for baseline error stats
    
    # Mode switching parameters (from paper)
    tau_fast_normal: float = 5.0
    tau_fast_online: float = 2.5            # halved for True Online
    tau_slow_normal: float = 50.0
    tau_slow_online: float = 40.0         # multiplied by 0.8 for True Online
    
    # Recovery parameters
    recovery_patience: int = 20             # Timesteps before returning to normal
    adaptation_rate: float = 0.1
    
    # Health logging
    log_interval: int = 100                 # Log health metrics every N steps


class DriftDetector:
    """
    Detects concept drift via error signal monitoring.
    
    Monitors RMS of error signal with short EMA. When error spikes
    above threshold (indicating sudden fault or process change),
    triggers True Online mode for rapid adaptation.
    """
    
    def __init__(
        self,
        config: Optional[DriftConfig] = None,
        device: Optional[str] = None
    ):
        self.config = config or DriftConfig()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Error history for baseline statistics
        self.error_history = deque(maxlen=self.config.error_baseline_window)
        
        # Running error EMA
        self.error_ema = 0.0
        self.error_ema_sq = 0.0  # For variance estimate
        
        # Drift state
        self.drift_detected = False
        self.drift_start_time = None
        self.steps_since_drift = 0
        
        # Statistics
        self.baseline_mean = 0.0
        self.baseline_std = 1.0
        
    def update(self, error: torch.Tensor) -> bool:
        """
        Update drift detector with current error.
        
        Args:
            error: Layer-local error signal (any shape)
            
        Returns:
            True if drift detected, False otherwise
        """
        # Compute error magnitude (RMS across dimensions)
        error_mag = error.pow(2).mean().sqrt().item()
        
        # Update history
        self.error_history.append(error_mag)
        
        # Update EMA
        alpha = self.config.adaptation_rate
        self.error_ema = (1 - alpha) * self.error_ema + alpha * error_mag
        self.error_ema_sq = (1 - alpha) * self.error_ema_sq + alpha * (error_mag ** 2)
        
        # Compute baseline statistics when we have enough history
        if len(self.error_history) >= self.config.error_baseline_window // 2:
            recent_errors = list(self.error_history)[-self.config.error_baseline_window//2:]
            self.baseline_mean = sum(recent_errors) / len(recent_errors)
            variance = sum((e - self.baseline_mean) ** 2 for e in recent_errors) / len(recent_errors)
            self.baseline_std = max(variance ** 0.5, 1e-6)
        
        # Drift detection: error exceeds threshold
        threshold = self.baseline_mean + self.config.error_spike_threshold * self.baseline_std
        
        if error_mag > threshold and not self.drift_detected:
            # Drift onset
            self.drift_detected = True
            self.drift_start_time = time.time()
            self.steps_since_drift = 0
            return True
        
        if self.drift_detected:
            self.steps_since_drift += 1
            
            # Check for recovery: error returned to normal range
            if error_mag < self.baseline_mean + 0.5 * self.baseline_std:
                if self.steps_since_drift >= self.config.recovery_patience:
                    # Recovery complete
                    self.drift_detected = False
                    self.steps_since_drift = 0
                    return False
        
        return self.drift_detected
    
    def get_drift_ratio(self) -> float:
        """
        Returns current error magnitude normalized by baseline.
        
        Ratio > 1.0 indicates elevated error (potential drift).
        """
        if self.baseline_std == 0:
            return 0.0
        return (self.error_ema - self.baseline_mean) / self.baseline_std
    
    def reset(self):
        """Reset drift detector state."""
        self.error_history.clear()
        self.error_ema = 0.0
        self.error_ema_sq = 0.0
        self.drift_detected = False
        self.drift_start_time = None
        self.steps_since_drift = 0
        self.baseline_mean = 0.0
        self.baseline_std = 1.0


class AdaptiveOnlineTrainer:
    """
    Online trainer with drift-gated mode switching.
    
    Switches between Batched Online and True Online modes based on
    drift detection status, per paper Section 3.3.
    """
    
    def __init__(
        self,
        rsnn,
        readout,
        hebbian,
        lr_readout: float = 1e-3,
        lr_recurrent: float = 5e-5,
        drift_config: Optional[DriftConfig] = None,
        device: Optional[str] = None
    ):
        self.rsnn = rsnn
        self.readout = readout
        self.hebbian = hebbian
        self.lr_readout = lr_readout
        self.lr_recurrent = lr_recurrent
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Drift detection
        self.drift_detector = DriftConfig(drift_config, device=self.device)
        
        # Mode tracking
        self.true_online_mode = False
        self.mode_switch_count = 0
        self.current_step = 0
        
        # Health metrics
        self.health_log = {
            'step': [],
            'mode': [],
            'error_rms': [],
            'drift_ratio': [],
            'momentum_norm': [],
        }
        
        # Store original time constants
        self._store_original_taus()
        
    def _store_original_taus(self):
        """Store original time constants from hebbian module."""
        h = self.hebbian._impl if hasattr(self.hebbian, '_impl') else self.hebbian
        self.original_tau_fast = getattr(h, 'tau_fast', 5.0)
        self.original_tau_slow = getattr(h, 'tau_slow', 50.0)
        
    def step(self, x: torch.Tensor, target: torch.Tensor) -> tuple:
        """
        Training step with drift-adaptive mode switching.
        
        Args:
            x: Input tensor
            target: Target tensor
            
        Returns:
            (y_pred, error, info_dict)
        """
        x = x.to(self.device)
        target = target.to(self.device)
        
        # Forward pass
        spikes = self.rsnn.forward(x)
        y_pred = self.readout.forward(spikes)
        
        # Compute error
        error = target - y_pred
        
        # Update drift detector
        drift_detected = self.drift_detector.update(error)
        
        # Mode switching logic
        if drift_detected and not self.true_online_mode:
            self._enter_true_online_mode()
        elif not drift_detected and self.true_online_mode:
            self._exit_true_online_mode()
        
        # Hebbian eligibility update
        h = self.hebbian._impl if hasattr(self.hebbian, '_impl') else self.hebbian
        E = h.update(spikes, spikes)
        
        # Update weights
        readout_update = self.lr_readout * torch.outer(error, spikes)
        self.readout.W += torch.clamp(readout_update, -0.1, 0.1)
        if hasattr(self.readout, 'b'):
            self.readout.b += torch.clamp(self.lr_readout * error, -0.1, 0.1)
        
        # Recurrent update (skip if drift detected and we want stability)
        if not drift_detected or self.true_online_mode:
            recurrent_update = self.lr_recurrent * E
            self.rsnn.W_rec += torch.clamp(recurrent_update, -0.1, 0.1)
        
        # Per-neuron weight projection (safety mechanism from paper)
        self._apply_weight_projection()
        
        # Log health metrics
        self._log_health(error, E)
        
        self.current_step += 1
        
        info = {
            'drift_detected': drift_detected,
            'true_online': self.true_online_mode,
            'drift_ratio': self.drift_detector.get_drift_ratio(),
        }
        
        return y_pred, error, info
    
    def _enter_true_online_mode(self):
        """Switch to True Online mode (paper Section 3.3)."""
        self.true_online_mode = True
        self.mode_switch_count += 1
        
        # Scale time constants per paper
        h = self.hebbian._impl if hasattr(self.hebbian, '_impl') else self.hebbian
        
        # τ_fast is halved, τ_slow is multiplied by 0.8
        h.tau_fast = self.drift_detector.config.tau_fast_online
        h.tau_slow = self.drift_detector.config.tau_slow_online
        
        # Recompute decay factors
        h.decay_fast = torch.exp(torch.tensor(-1.0 / h.tau_fast, device=self.device))
        h.decay_slow = torch.exp(torch.tensor(-1.0 / h.tau_slow, device=self.device))
        
        print(f"[Drift] Entered True Online mode at step {self.current_step}")
    
    def _exit_true_online_mode(self):
        """Return to Batched Online (normal) mode."""
        self.true_online_mode = False
        
        # Restore original time constants
        h = self.hebbian._impl if hasattr(self.hebbian, '_impl') else self.hebbian
        h.tau_fast = self.original_tau_fast
        h.tau_slow = self.original_tau_slow
        
        h.decay_fast = torch.exp(torch.tensor(-1.0 / h.tau_fast, device=self.device))
        h.decay_slow = torch.exp(torch.tensor(-1.0 / h.tau_slow, device=self.device))
        
        print(f"[Drift] Returned to Batched Online mode at step {self.current_step}")
    
    def _apply_weight_projection(self):
        """
        Per-neuron L2 weight projection from paper Section 3.4.
        
        Rescales incoming weight vectors exceeding threshold c_ℓ = 6.0
        using power-of-two division (bitshift in hardware).
        """
        threshold = 6.0
        
        # For each neuron, check incoming recurrent weights
        for i in range(self.rsnn.hidden_size):
            w_incoming = self.rsnn.W_rec[i, :]  # Weights from all neurons to neuron i
            l2_norm = w_incoming.norm(p=2).item()
            
            if l2_norm > threshold:
                # Power-of-two division (bitshift approximation)
                # Find smallest power of 2 divisor that brings norm below threshold
                scale = 1.0
                while l2_norm / scale > threshold:
                    scale *= 2.0
                
                # Apply rescaling
                self.rsnn.W_rec[i, :] = w_incoming / scale
    
    def _log_health(self, error: torch.Tensor, E: torch.Tensor):
        """Log health metrics."""
        config = self.drift_detector.config
        
        if self.current_step % config.log_interval == 0:
            self.health_log['step'].append(self.current_step)
            self.health_log['mode'].append('online' if self.true_online_mode else 'batch')
            self.health_log['error_rms'].append(error.pow(2).mean().sqrt().item())
            self.health_log['drift_ratio'].append(self.drift_detector.get_drift_ratio())
            
            # Momentum accumulator norm (paper metric for "decoder has re-stabilized")
            momentum_norm = E.norm(p='fro').item()
            self.health_log['momentum_norm'].append(momentum_norm)
    
    def get_health_report(self) -> Dict:
        """
        Get health metrics indicating decoder stability.
        
        A declining momentum norm after disruption indicates recovery.
        """
        return {
            'current_mode': 'True Online' if self.true_online_mode else 'Batched Online',
            'mode_switches': self.mode_switch_count,
            'current_drift_ratio': self.drift_detector.get_drift_ratio(),
            'baseline_error_mean': self.drift_detector.baseline_mean,
            'baseline_error_std': self.drift_detector.baseline_std,
            'health_history': self.health_log,
        }
    
    def inject_context_tag(self, tag: torch.Tensor):
        """
        Inject one-hot context vector for known manufacturing events.
        
        Tool wear or batch changeover can trigger explicit context
        to accelerate eligibility trace consolidation.
        
        Args:
            tag: One-hot context vector to inject into hidden layer
        """
        # This would modify the input to the first hidden layer
        # Implementation depends on model architecture
        pass
    
    def reset(self):
        """Reset trainer and drift detector."""
        self.drift_detector.reset()
        self.true_online_mode = False
        self.mode_switch_count = 0
        self.current_step = 0
        self.health_log = {
            'step': [],
            'mode': [],
            'error_rms': [],
            'drift_ratio': [],
            'momentum_norm': [],
        }
