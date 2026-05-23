"""
oracle_defense.py
=================
HDC-based poison detection and outlier defense (Phase 5 of HDC integration plan).

Implements two complementary defenses:

1. Similarity oracle
   Encode each training sample as a hypervector, compare it to the class
   prototype built so far.  A sample is flagged as suspicious if its
   similarity to the claimed class is below `sim_thresh` AND it is
   simultaneously closer to a *different* class.

2. Statistical outlier filter
   Track the running mean and variance of per-class hypervectors.
   Flag a sample if its distance from the class mean exceeds `z_thresh`
   standard deviations (z-score in HD space ≈ cosine deviation).

Both defenses are fully online (O(1) memory) and hardware-compatible
because all operations reduce to Hamming / cosine distances over fixed-
length hypervectors — exactly the CIM Hamming pipeline in hdc/cim_hamming.py.

Usage
-----
    from training.oracle_defense import OracleDefense, DefenseConfig

    guard = OracleDefense(DefenseConfig(n_classes=8, hdc_dim=4096))

    # During training
    for x_hv, label in hv_stream:
        verdict = guard.check(x_hv, label)
        if verdict.clean:
            model.train_step(x_hv, label)
        guard.update(x_hv, label)     # always update guard state
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DefenseConfig:
    n_classes: int = 8
    hdc_dim: int = 4096
    hdc_mode: str = "bipolar"       # "binary" | "bipolar"

    # Similarity oracle thresholds
    sim_thresh: float = 0.15        # min cosine sim to claimed class
    cross_sim_margin: float = 0.05  # suspect if rival class is this much closer

    # Statistical outlier filter
    z_thresh: float = 3.0           # z-score threshold for outlier detection
    warmup_samples: int = 10        # per-class samples before z-filter activates

    # Logging
    log_suspects: bool = True
    device: Optional[str] = None


# ---------------------------------------------------------------------------
# Verdict dataclass
# ---------------------------------------------------------------------------

@dataclass
class Verdict:
    clean: bool
    claimed_label: int
    nearest_label: int
    sim_to_claimed: float
    sim_to_nearest: float
    z_score: float
    reason: str = ""

    def __repr__(self) -> str:
        tag = "CLEAN" if self.clean else "SUSPECT"
        return (f"[{tag}] label={self.claimed_label} "
                f"sim_claimed={self.sim_to_claimed:.3f} "
                f"sim_nearest={self.sim_to_nearest:.3f} "
                f"z={self.z_score:.2f} — {self.reason}")


# ---------------------------------------------------------------------------
# Oracle defense
# ---------------------------------------------------------------------------

class OracleDefense:
    """
    Online hypervector poison / outlier detector.

    State per class
    ---------------
    prototypes   : (n_classes, dim) — running bundle of clean samples
    counts       : (n_classes,)     — number of accepted samples per class
    mean_sim     : (n_classes,)     — running mean of within-class cosine sim
    var_sim      : (n_classes,)     — running variance (Welford's algorithm)
    """

    def __init__(self, config: Optional[DefenseConfig] = None) -> None:
        self.cfg = config or DefenseConfig()
        self.device = torch.device(self.cfg.device or (
            'cuda' if torch.cuda.is_available() else 'cpu'))

        dim = self.cfg.hdc_dim
        nc  = self.cfg.n_classes

        # Running prototypes (sum of accepted HVs per class, normalised at query)
        self.prototypes = torch.zeros(nc, dim, device=self.device)
        self.counts = torch.zeros(nc, device=self.device)

        # Welford running stats for cosine similarity
        self.mean_sim = torch.zeros(nc, device=self.device)
        self.var_sim  = torch.zeros(nc, device=self.device)

        self.suspects: List[Tuple[torch.Tensor, int, Verdict]] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cosine(self, a: torch.Tensor, b: torch.Tensor) -> float:
        an, bn = a.norm(), b.norm()
        if an < 1e-12 or bn < 1e-12:
            return 0.0
        return float((a @ b) / (an * bn))

    def _batch_cosine(self, q: torch.Tensor) -> torch.Tensor:
        """Cosine similarity of q to all class prototypes. Returns (n_classes,)."""
        norms = self.prototypes.norm(dim=1).clamp(min=1e-12)
        q_norm = q.norm().clamp(min=1e-12)
        return (self.prototypes @ q) / (norms * q_norm)

    def _welford_update(self, label: int, new_sim: float) -> None:
        """Update running mean and variance (Welford online algorithm)."""
        n = self.counts[label].item()
        delta = new_sim - self.mean_sim[label].item()
        self.mean_sim[label] += delta / max(1.0, n)
        delta2 = new_sim - self.mean_sim[label].item()
        self.var_sim[label] += delta * delta2

    def _z_score(self, label: int, sim: float) -> float:
        """Z-score of sim relative to class label's running distribution."""
        n = self.counts[label].item()
        if n < self.cfg.warmup_samples:
            return 0.0
        std = math.sqrt(self.var_sim[label].item() / max(1.0, n - 1))
        if std < 1e-8:
            return 0.0
        return abs(sim - self.mean_sim[label].item()) / std

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self, hv: torch.Tensor, claimed_label: int) -> Verdict:
        """
        Evaluate whether (hv, claimed_label) is a suspicious training sample.

        Parameters
        ----------
        hv : (dim,) — hypervector encoding of the sample
        claimed_label : int — the label asserted by the training data source

        Returns
        -------
        Verdict with clean=True if both oracle and z-filter pass.
        """
        hv = hv.to(self.device)
        sims = self._batch_cosine(hv)           # (n_classes,)

        sim_claimed = float(sims[claimed_label].item())

        # Find nearest class (excluding claimed)
        sims_excl = sims.clone()
        sims_excl[claimed_label] = -2.0
        nearest_label = int(sims_excl.argmax().item())
        sim_nearest   = float(sims_excl[nearest_label].item())

        z = self._z_score(claimed_label, sim_claimed)

        # Rule 1: similarity oracle
        oracle_fail = (
            sim_claimed < self.cfg.sim_thresh
            and sim_nearest > sim_claimed + self.cfg.cross_sim_margin
            and self.counts[claimed_label] >= self.cfg.warmup_samples
        )

        # Rule 2: statistical outlier
        outlier_fail = (z > self.cfg.z_thresh
                        and self.counts[claimed_label] >= self.cfg.warmup_samples)

        clean = not (oracle_fail or outlier_fail)
        reasons = []
        if oracle_fail:
            reasons.append(
                f"oracle: sim_claimed={sim_claimed:.3f} < {self.cfg.sim_thresh}, "
                f"rival={sim_nearest:.3f}"
            )
        if outlier_fail:
            reasons.append(f"outlier: z={z:.2f} > {self.cfg.z_thresh}")

        verdict = Verdict(
            clean=clean,
            claimed_label=claimed_label,
            nearest_label=nearest_label,
            sim_to_claimed=sim_claimed,
            sim_to_nearest=sim_nearest,
            z_score=z,
            reason="; ".join(reasons) if reasons else "ok",
        )

        if not clean and self.cfg.log_suspects:
            self.suspects.append((hv.cpu().clone(), claimed_label, verdict))

        return verdict

    def update(self, hv: torch.Tensor, label: int) -> None:
        """
        Add a (verified clean) sample to the guard's running state.
        Call this even for suspect samples if you want the guard to
        remain statistically calibrated — or skip it to prevent
        prototype contamination.
        """
        hv = hv.to(self.device)
        sim = self._cosine(hv, self.prototypes[label])
        self.prototypes[label] = self.prototypes[label] + hv
        self.counts[label] += 1
        self._welford_update(label, sim)

    def report(self) -> str:
        """Return a human-readable summary of all flagged suspects."""
        if not self.suspects:
            return "OracleDefense: no suspects detected."
        lines = [f"OracleDefense: {len(self.suspects)} suspect(s)"]
        for i, (_, lbl, v) in enumerate(self.suspects[:20]):
            lines.append(f"  [{i}] label={lbl}  {v}")
        if len(self.suspects) > 20:
            lines.append(f"  ... and {len(self.suspects) - 20} more")
        return "\n".join(lines)

    def reset_suspects(self) -> None:
        self.suspects.clear()


# ---------------------------------------------------------------------------
# Convenience: wrap a HDCEncoder for guard-protected training
# ---------------------------------------------------------------------------

class GuardedTrainer:
    """
    Thin wrapper that gates HDC training through OracleDefense.

    Usage
    -----
        trainer = GuardedTrainer(hdc_encoder, guard)
        for hv, label in stream:
            trainer.step(hv, label)
        trainer.finalize()
    """

    def __init__(self, hdc_model, guard: OracleDefense) -> None:
        self.model = hdc_model
        self.guard = guard
        self.n_accepted = 0
        self.n_rejected = 0

    def step(self, hv: torch.Tensor, label: int) -> Verdict:
        verdict = self.guard.check(hv, label)
        self.guard.update(hv, label)
        if verdict.clean:
            self.model.assoc_mem.add(hv, label)
            self.n_accepted += 1
        else:
            self.n_rejected += 1
        return verdict

    def finalize(self) -> None:
        self.model.assoc_mem.renormalize()

    def stats(self) -> str:
        total = self.n_accepted + self.n_rejected
        return (f"GuardedTrainer: {self.n_accepted}/{total} accepted, "
                f"{self.n_rejected} rejected")


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    dim = 256
    n_classes = 4
    guard = OracleDefense(DefenseConfig(
        n_classes=n_classes, hdc_dim=dim, warmup_samples=5
    ))

    # Warmup with clean samples
    for cls in range(n_classes):
        for _ in range(8):
            hv = torch.randn(dim)
            hv[cls * (dim // n_classes):(cls + 1) * (dim // n_classes)] += 3.0
            guard.update(hv / hv.norm(), cls)

    # Clean samples
    clean_pass = 0
    for cls in range(n_classes):
        hv = torch.randn(dim)
        hv[cls * (dim // n_classes):(cls + 1) * (dim // n_classes)] += 3.0
        v = guard.check(hv / hv.norm(), cls)
        if v.clean:
            clean_pass += 1

    # Poisoned samples (wrong class mapping)
    poison_caught = 0
    for cls in range(n_classes):
        hv = torch.randn(dim)
        wrong_cls = (cls + 1) % n_classes
        hv[wrong_cls * (dim // n_classes):(wrong_cls + 1) * (dim // n_classes)] += 3.0
        v = guard.check(hv / hv.norm(), cls)
        if not v.clean:
            poison_caught += 1

    print(f"Oracle defense smoke test:")
    print(f"  Clean pass:    {clean_pass}/{n_classes}")
    print(f"  Poison caught: {poison_caught}/{n_classes}")
    print(guard.report())
