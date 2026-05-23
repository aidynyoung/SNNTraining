"""
Kalman Filter and BPTT SNN baselines for fair comparison with Arthedain.
Both consume identical data streams.
"""
import torch
import numpy as np
from scipy.stats import pearsonr


# ── Kalman Filter baseline ────────────────────────────────────────────────────

class KinematicKalman:
    """
    Standard 4D kinematic Kalman filter for 2D cursor velocity decoding.
    State: [pos_x, pos_y, vel_x, vel_y]
    """
    def __init__(self, input_size: int, obs_noise: float = 0.1, proc_noise: float = 0.01):
        self.n_state = 4
        self.n_obs = input_size

        # State transition: constant-velocity model
        self.F = np.eye(4)
        self.F[0, 2] = 1.0
        self.F[1, 3] = 1.0

        # Observation matrix (fit by ridge regression on first N samples)
        self.H = np.random.randn(input_size, 4) * 0.01
        self._fitted = False

        self.Q = np.eye(4) * proc_noise     # process noise
        self.R = np.eye(input_size) * obs_noise  # observation noise
        self.P = np.eye(4)                   # state covariance
        self.x = np.zeros(4)                 # state estimate

    def fit(self, spikes: np.ndarray, targets: np.ndarray):
        """Ridge regression to initialize observation matrix H."""
        from numpy.linalg import lstsq
        # targets: (T, 2) velocity; spikes: (T, input_size)
        states = np.hstack([np.cumsum(targets, axis=0), targets])  # pos + vel
        self.H = lstsq(states, spikes, rcond=None)[0].T  # (input_size, 4)
        self._fitted = True

    def step(self, obs: np.ndarray) -> np.ndarray:
        # Predict
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q
        # Update
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)
        self.x = x_pred + K @ (obs - self.H @ x_pred)
        self.P = (np.eye(self.n_state) - K @ self.H) @ P_pred
        return self.x[2:4]  # return velocity estimate


# ── BPTT SNN baseline ─────────────────────────────────────────────────────────

class BPTTBaseline:
    """
    Same RSNN architecture, trained with BPTT (torch.autograd).
    Provides the 'cheating' upper bound for comparison.
    """
    def __init__(self, rsnn, readout, lr: float = 1e-3, device="cpu"):
        self.rsnn = rsnn
        self.readout = readout
        self.device = device
        
        # Collect parameters - handle both nn.Module and custom classes
        params = []
        if hasattr(rsnn, 'parameters'):
            params.extend(list(rsnn.parameters()))
        else:
            # Custom RSNN - manually collect weight tensors
            if hasattr(rsnn, 'W_in'):
                rsnn.W_in.requires_grad = True
                params.append(rsnn.W_in)
            if hasattr(rsnn, 'W_rec'):
                rsnn.W_rec.requires_grad = True
                params.append(rsnn.W_rec)
                
        if hasattr(readout, 'parameters'):
            params.extend(list(readout.parameters()))
        else:
            # Custom Readout - manually collect weight tensors
            if hasattr(readout, 'W'):
                readout.W.requires_grad = True
                params.append(readout.W)
            if hasattr(readout, 'b'):
                readout.b.requires_grad = True
                params.append(readout.b)
            
        self.optimizer = torch.optim.Adam(params, lr=lr)
        self.loss_fn = torch.nn.MSELoss()

        # For BPTT we need grad — disable no_grad context
        self._spike_buffer = []
        self._target_buffer = []
        self.truncation_length = 20  # TBPTT window

    def step(self, x: torch.Tensor, target: torch.Tensor):
        """
        Truncated BPTT: accumulate truncation_length steps then backprop.
        This is the standard 'fair' BPTT comparison for online settings.
        """
        # Forward WITH grad tracking
        spikes = self.rsnn.forward(x)
        y_pred = self.readout.forward(spikes)

        self._spike_buffer.append(y_pred)
        self._target_buffer.append(target)

        if len(self._spike_buffer) >= self.truncation_length:
            preds = torch.stack(self._spike_buffer)
            tgts = torch.stack(self._target_buffer)
            loss = self.loss_fn(preds, tgts)

            self.optimizer.zero_grad()
            loss.backward()
            # Collect parameters for gradient clipping
            clip_params = []
            if hasattr(self.rsnn, 'parameters'):
                clip_params.extend(list(self.rsnn.parameters()))
            else:
                if hasattr(self.rsnn, 'W_in'):
                    clip_params.append(self.rsnn.W_in)
                if hasattr(self.rsnn, 'W_rec'):
                    clip_params.append(self.rsnn.W_rec)
            if hasattr(self.readout, 'parameters'):
                clip_params.extend(list(self.readout.parameters()))
            else:
                if hasattr(self.readout, 'W'):
                    clip_params.append(self.readout.W)
                if hasattr(self.readout, 'b'):
                    clip_params.append(self.readout.b)
            
            torch.nn.utils.clip_grad_norm_(clip_params, 1.0)
            self.optimizer.step()

            self._spike_buffer.clear()
            self._target_buffer.clear()

        error = target - y_pred.detach()
        return y_pred.detach(), error
