import json
import os
import random
import time
import yaml
import torch
import numpy as np
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def load_config(config_path: str = "configs/default.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_device(device_config: str = "auto") -> torch.device:
    """Get appropriate device based on configuration."""
    if device_config == "auto":
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif device_config == "cuda":
        if not torch.cuda.is_available():
            print("Warning: CUDA requested but not available, falling back to CPU")
            return torch.device('cpu')
        return torch.device('cuda')
    return torch.device('cpu')


def _get_lif_tau(config: Dict[str, Any]) -> float:
    """Get LIF tau parameter with backward compatibility (tau / tau_m)."""
    lif_config = config.get('lif', {})
    return lif_config.get('tau', lif_config.get('tau_m', 20.0))


def seed_everything(seed: int) -> None:
    """Set all random seeds for reproducibility (Python/NumPy/PyTorch/CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def print_config_summary(config: Dict[str, Any]) -> None:
    """Print a summary of the loaded configuration."""
    print("=== Configuration Summary ===")
    print(f"Model: {config['model']['input_size']}->{config['model']['hidden_size']}->{config['model']['output_size']}")
    print(f"LIF tau: {_get_lif_tau(config)}")
    print(f"Hebbian tau_fast/slow: {config['hebbian']['tau_fast']}/{config['hebbian']['tau_slow']}")
    lr = config.get('training', {}).get('lr_readout',
           config.get('training', {}).get('learning_rate', 0.001))
    print(f"Training LR: {lr}")
    print("=" * 30)


# ─────────────────────────────────────────────────────────────────────────────
# Timer utilities
# ─────────────────────────────────────────────────────────────────────────────

class Timer:
    """Wall-clock timer with named laps.

    Usage::

        t = Timer()
        ...
        t.lap("encode")
        ...
        t.lap("train")
        print(t.report())
    """

    def __init__(self):
        self._start = time.perf_counter()
        self._laps: List[Tuple[str, float]] = []

    def lap(self, label: str = "") -> float:
        """Record a lap and return elapsed seconds since start."""
        elapsed = time.perf_counter() - self._start
        self._laps.append((label, elapsed))
        return elapsed

    def elapsed(self) -> float:
        return time.perf_counter() - self._start

    def report(self) -> str:
        lines = [f"  total: {self.elapsed():.3f}s"]
        for label, t in self._laps:
            lines.append(f"  {label}: {t:.3f}s")
        return "\n".join(lines)

    def reset(self):
        self._start = time.perf_counter()
        self._laps.clear()


@contextmanager
def timed(label: str = ""):
    """Context manager that prints elapsed time for a code block."""
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    tag = f"[{label}] " if label else ""
    print(f"{tag}{dt * 1000:.1f} ms")


# ─────────────────────────────────────────────────────────────────────────────
# Experiment tracking
# ─────────────────────────────────────────────────────────────────────────────

class ExperimentTracker:
    """
    Lightweight experiment tracker — no wandb/MLflow dependency.

    Records scalar metrics at each step, computes running statistics,
    and saves a JSON summary for reproducibility and reporting.

    Usage::

        tracker = ExperimentTracker("bci_run_01")
        for t, (x, y) in enumerate(stream):
            pred, r = trainer.step(x, y)
            tracker.log(t, pearson_r=r, error=float((pred - y).abs().mean()))
        tracker.save("results/bci_run_01.json")
        print(tracker.summary())
    """

    def __init__(self, name: str = "experiment"):
        self.name    = name
        self._data:  Dict[str, List[float]] = {}
        self._steps: List[int]              = []
        self._t0     = time.perf_counter()

    def log(self, step: int, **metrics):
        """Log one or more scalar metrics at a given step."""
        self._steps.append(step)
        for k, v in metrics.items():
            self._data.setdefault(k, []).append(float(v))

    def last(self, metric: str) -> Optional[float]:
        vals = self._data.get(metric, [])
        return vals[-1] if vals else None

    def best(self, metric: str, higher_is_better: bool = True) -> Optional[float]:
        vals = self._data.get(metric, [])
        if not vals:
            return None
        return max(vals) if higher_is_better else min(vals)

    def mean(self, metric: str, window: int = 0) -> Optional[float]:
        """Mean over last `window` values (0 = all)."""
        vals = self._data.get(metric, [])
        if not vals:
            return None
        subset = vals[-window:] if window > 0 else vals
        return sum(subset) / len(subset)

    def summary(self) -> Dict[str, Any]:
        elapsed = time.perf_counter() - self._t0
        out: Dict[str, Any] = {
            "name":      self.name,
            "n_steps":   len(self._steps),
            "elapsed_s": round(elapsed, 2),
        }
        for k, vals in self._data.items():
            out[k] = {
                "last": round(vals[-1], 6),
                "best": round(max(vals), 6),
                "mean": round(sum(vals) / len(vals), 6),
                "n":    len(vals),
            }
        return out

    def save(self, path: str):
        """Serialise full history + summary to JSON."""
        out = {"summary": self.summary(), "steps": self._steps, "metrics": self._data}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(out, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ExperimentTracker":
        with open(path) as f:
            data = json.load(f)
        t         = cls(data["summary"]["name"])
        t._steps  = data["steps"]
        t._data   = data["metrics"]
        return t

    def __repr__(self) -> str:
        return (f"ExperimentTracker({self.name!r}, "
                f"steps={len(self._steps)}, metrics={list(self._data)})")


# ─────────────────────────────────────────────────────────────────────────────
# Statistical helpers
# ─────────────────────────────────────────────────────────────────────────────

def moving_average(values: List[float], window: int) -> List[float]:
    """Simple moving average — no external dependencies."""
    if not values or window <= 0:
        return list(values)
    result = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        result.append(sum(values[lo : i + 1]) / (i - lo + 1))
    return result


def pearson_r_numpy(preds: List[float], targets: List[float]) -> float:
    """Compute Pearson R without PyTorch (for lightweight scripts)."""
    if len(preds) < 2:
        return 0.0
    p  = np.array(preds,   dtype=np.float64)
    t  = np.array(targets, dtype=np.float64)
    pc = p - p.mean()
    tc = t - t.mean()
    denom = np.linalg.norm(pc) * np.linalg.norm(tc)
    return float(np.dot(pc, tc) / denom) if denom > 1e-10 else 0.0


def exponential_moving_average(
    values: List[float],
    alpha:  float = 0.1,
) -> List[float]:
    """EMA over a list — matches the online EMA used inside Arthedain models."""
    if not values:
        return []
    ema = [values[0]]
    for v in values[1:]:
        ema.append((1 - alpha) * ema[-1] + alpha * v)
    return ema


def running_pearson_r(
    preds:   torch.Tensor,   # (T, K)
    targets: torch.Tensor,   # (T, K)
) -> torch.Tensor:
    """Per-output Pearson R — vectorised, no loop over K dimensions."""
    pc = preds   - preds.mean(0, keepdim=True)
    tc = targets - targets.mean(0, keepdim=True)
    num   = (pc * tc).sum(0)
    denom = (pc.norm(dim=0) * tc.norm(dim=0)).clamp(min=1e-8)
    return num / denom   # (K,)


# ─────────────────────────────────────────────────────────────────────────────
# Checkpointing
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    model,
    path:    str,
    step:    int  = 0,
    metrics: Optional[Dict[str, float]] = None,
):
    """
    Save a model checkpoint using PyTorch state_dict if available,
    otherwise pickle the object.

    Creates parent directories automatically.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "step":    step,
        "metrics": metrics or {},
    }
    if hasattr(model, "state_dict"):
        payload["state_dict"] = model.state_dict()
    else:
        payload["model"] = model
    torch.save(payload, path)


def load_checkpoint(model, path: str) -> Dict[str, Any]:
    """
    Load a checkpoint saved with save_checkpoint().

    Returns the payload dict (including step/metrics).
    Applies state_dict if available.
    """
    payload = torch.load(path, map_location="cpu")
    if "state_dict" in payload and hasattr(model, "load_state_dict"):
        model.load_state_dict(payload["state_dict"])
    return payload
