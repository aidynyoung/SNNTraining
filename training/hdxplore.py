"""
hdxplore.py
===========
HDXplore: differential fuzz testing for SNN / HDC models.

Inspired by the HDXplore technique described in Section III-B of
"Brain-Inspired Hyperdimensional Computing for Ultra-Efficient Edge AI"
(Amrouch et al. 2022, NSF purl/10392362).

Finds adversarial spike inputs that:
  1. Flip the predicted class (evasion adversarials).
  2. Expose disagreements between two implementations (oracle vs. under-test).
  3. Maximise coverage of the HDC hypervector space (coverage-guided).

Three fuzzing strategies
------------------------
gradient_free
    Random mutations: flip bits in the input spike vector, accept if the
    prediction changes.  Works with any black-box classifier.

coverage_guided
    Track which neurons have spiked (neuron coverage map).  Prioritise
    mutations that activate previously-unseen neurons.

differential
    Run the same input through a reference model and a candidate model.
    Flag inputs where predictions disagree.

Usage
-----
    from training.hdxplore import HDXplore, FuzzConfig

    fuzzer = HDXplore(reference_model, candidate_model, FuzzConfig())
    report = fuzzer.run(seed_inputs)
    print(report.summary())
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple, Dict

import torch


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class FuzzConfig:
    strategy: str = "differential"     # "gradient_free" | "coverage_guided" | "differential"
    n_iterations: int = 1000
    mutation_rate: float = 0.05        # fraction of bits to flip per mutation
    max_mutations_per_seed: int = 20   # hill-climb depth per seed
    timeout_s: float = 60.0            # wall-clock budget
    seed: int = 42


# ---------------------------------------------------------------------------
# Fuzzing report
# ---------------------------------------------------------------------------

@dataclass
class FuzzReport:
    strategy: str
    n_tested: int = 0
    n_adversarials: int = 0
    n_disagreements: int = 0
    neuron_coverage: float = 0.0
    adversarial_inputs: List[torch.Tensor] = field(default_factory=list)
    disagreement_inputs: List[torch.Tensor] = field(default_factory=list)
    elapsed_s: float = 0.0

    def summary(self) -> str:
        return (
            f"HDXplore report [{self.strategy}]\n"
            f"  Tested:          {self.n_tested}\n"
            f"  Adversarials:    {self.n_adversarials}  "
            f"({100*self.n_adversarials/max(1,self.n_tested):.1f}%)\n"
            f"  Disagreements:   {self.n_disagreements}  "
            f"({100*self.n_disagreements/max(1,self.n_tested):.1f}%)\n"
            f"  Neuron coverage: {100*self.neuron_coverage:.1f}%\n"
            f"  Elapsed:         {self.elapsed_s:.2f}s"
        )


# ---------------------------------------------------------------------------
# Core fuzzer
# ---------------------------------------------------------------------------

ModelFn = Callable[[torch.Tensor], int]   # x → predicted_class


class HDXplore:
    """
    HDXplore differential fuzzer.

    Parameters
    ----------
    reference : callable
        Reference model f(x) → int label.  Treated as oracle.
    candidate : callable
        Candidate model g(x) → int label.  Under test.
    config : FuzzConfig
    n_neurons : int
        Number of hidden neurons (for coverage tracking).
    spike_hook : callable, optional
        Optional f(x) → spike_vector used for coverage tracking.
        If None, coverage tracking is disabled.
    """

    def __init__(
        self,
        reference: ModelFn,
        candidate: ModelFn,
        config: Optional[FuzzConfig] = None,
        n_neurons: int = 128,
        spike_hook: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        self.ref = reference
        self.cand = candidate
        self.cfg = config or FuzzConfig()
        self.n_neurons = n_neurons
        self.spike_hook = spike_hook
        self._rng = random.Random(self.cfg.seed)

        # Coverage bitmap: which (neuron, value) pairs have been observed
        self._neuron_seen: set = set()

    # ------------------------------------------------------------------
    # Mutation operators
    # ------------------------------------------------------------------

    def _mutate_bits(self, x: torch.Tensor) -> torch.Tensor:
        """Randomly flip bits in a binary spike input."""
        mask = torch.rand_like(x) < self.cfg.mutation_rate
        return (x + mask.float()) % 2   # XOR for binary

    def _mutate_rate(self, x: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to a continuous rate-coded input."""
        noise = torch.randn_like(x) * self.cfg.mutation_rate
        return (x + noise).clamp(0.0, 1.0)

    def _mutate(self, x: torch.Tensor) -> torch.Tensor:
        if x.max() <= 1.0 and x.min() >= 0.0 and x.float().eq(x.float().round()).all():
            return self._mutate_bits(x)
        return self._mutate_rate(x)

    # ------------------------------------------------------------------
    # Coverage tracking
    # ------------------------------------------------------------------

    def _update_coverage(self, x: torch.Tensor) -> int:
        """Return number of newly activated neurons for this input."""
        if self.spike_hook is None:
            return 0
        spikes = self.spike_hook(x)
        new = 0
        for i, s in enumerate(spikes):
            key = (i, int(s.item() > 0.5))
            if key not in self._neuron_seen:
                self._neuron_seen.add(key)
                new += 1
        return new

    def _coverage_ratio(self) -> float:
        """Fraction of (neuron × {0,1}) pairs seen."""
        total = 2 * self.n_neurons
        return len(self._neuron_seen) / total

    # ------------------------------------------------------------------
    # Gradient-free strategy (single-model evasion)
    # ------------------------------------------------------------------

    def _run_gradient_free(
        self, seeds: List[torch.Tensor], report: FuzzReport
    ) -> None:
        for seed in seeds:
            orig_label = self.ref(seed)
            for _ in range(self.cfg.max_mutations_per_seed):
                if time.time() - report.elapsed_s > self.cfg.timeout_s:
                    return
                mutant = self._mutate(seed)
                report.n_tested += 1
                self._update_coverage(mutant)
                new_label = self.ref(mutant)
                if new_label != orig_label:
                    report.n_adversarials += 1
                    report.adversarial_inputs.append(mutant.clone())
                    break

    # ------------------------------------------------------------------
    # Coverage-guided strategy
    # ------------------------------------------------------------------

    def _run_coverage_guided(
        self, seeds: List[torch.Tensor], report: FuzzReport
    ) -> None:
        queue = [s.clone() for s in seeds]
        idx = 0
        while report.n_tested < self.cfg.n_iterations:
            if time.time() - report.elapsed_s > self.cfg.timeout_s:
                break
            x = queue[idx % len(queue)]
            orig_label = self.ref(x)
            best_new = 0
            best_mut = None
            # Try several mutations, keep the one with most new coverage
            for _ in range(5):
                mutant = self._mutate(x)
                new = self._update_coverage(mutant)
                if new > best_new:
                    best_new = new
                    best_mut = mutant.clone()
            if best_mut is not None:
                queue.append(best_mut)
                report.n_tested += 1
                if self.ref(best_mut) != orig_label:
                    report.n_adversarials += 1
                    report.adversarial_inputs.append(best_mut.clone())
            idx += 1

    # ------------------------------------------------------------------
    # Differential strategy (reference vs. candidate)
    # ------------------------------------------------------------------

    def _run_differential(
        self, seeds: List[torch.Tensor], report: FuzzReport
    ) -> None:
        for seed in seeds:
            for _ in range(self.cfg.max_mutations_per_seed):
                if time.time() - report.elapsed_s > self.cfg.timeout_s:
                    return
                mutant = self._mutate(seed)
                report.n_tested += 1
                self._update_coverage(mutant)
                ref_label  = self.ref(mutant)
                cand_label = self.cand(mutant)
                if ref_label != cand_label:
                    report.n_disagreements += 1
                    report.disagreement_inputs.append(mutant.clone())
                    # Also flag as adversarial relative to candidate
                    orig = self.cand(seed)
                    if cand_label != orig:
                        report.n_adversarials += 1
                        report.adversarial_inputs.append(mutant.clone())

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, seed_inputs: List[torch.Tensor]) -> FuzzReport:
        """
        Run the configured fuzzing strategy on the given seed inputs.

        Parameters
        ----------
        seed_inputs : list of Tensors
            Starting points for mutation.  Should represent valid
            inputs (e.g. real spike observations from the validation set).

        Returns
        -------
        FuzzReport
        """
        report = FuzzReport(strategy=self.cfg.strategy)
        t0 = time.time()

        dispatch: Dict[str, object] = {
            "gradient_free": self._run_gradient_free,
            "coverage_guided": self._run_coverage_guided,
            "differential": self._run_differential,
        }
        fn = dispatch.get(self.cfg.strategy)
        if fn is None:
            raise ValueError(f"Unknown strategy: {self.cfg.strategy!r}")

        fn(seed_inputs, report)  # type: ignore[operator]

        report.elapsed_s = time.time() - t0
        report.neuron_coverage = self._coverage_ratio()
        return report


# ---------------------------------------------------------------------------
# Convenience: build reference/candidate wrappers from SNNHDCPipeline
# ---------------------------------------------------------------------------

def pipeline_to_fn(pipeline) -> ModelFn:
    """
    Wrap a SNNHDCPipeline as a callable int-returning function for HDXplore.
    Runs the pipeline for one full window and returns the predicted class.
    """
    def fn(x: torch.Tensor) -> int:
        pipeline.reset()
        window = pipeline.cfg.window
        label = 0
        for _ in range(window):
            l, _ = pipeline.predict(x)
            if l is not None:
                label = l
        return label
    return fn


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Simple classifiers: threshold on sum of first / second half of input
    def ref_model(x: torch.Tensor) -> int:
        return int(x[:10].sum() > x[10:].sum())

    def cand_model(x: torch.Tensor) -> int:
        # Slightly buggy: uses raw max instead of sum
        return int(x[:10].max() > x[10:].max())

    seeds = [torch.randint(0, 2, (20,)).float() for _ in range(10)]

    fuzzer = HDXplore(ref_model, cand_model, FuzzConfig(
        strategy="differential", n_iterations=200, timeout_s=5.0
    ))
    report = fuzzer.run(seeds)
    print(report.summary())
