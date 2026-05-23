"""HDXplore: Differential Fuzz Testing for HDC.
==========================================
Based on Section III-B of Amrouch et al. 2022:
"Robust Hyperdimensional Computing Against Hardware Errors and Attacks"

Key insight: Since HDC doesn't have gradients for adversarial
attack generation, coverage-guided fuzz testing is used instead.
HDXplore mutates inputs and compares predictions between original
and perturbed samples to find inconsistencies.

This provides automated testing for HDC model robustness without
requiring gradient information.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Callable
from dataclasses import dataclass, field


@dataclass
class FuzzConfig:
    """Configuration for fuzz testing."""
    n_mutations: int = 100  # Number of mutations per sample
    mutation_type: str = "pixel_flip"  # "pixel_flip", "gaussian", "salt_pepper"
    mutation_rate: float = 0.05  # Fraction of elements to mutate
    mutation_magnitude: float = 0.1  # Strength of continuous mutations
    max_queries: int = 1000  # Max total fuzz queries
    min_accuracy_drop: float = 0.1  # Flag if accuracy drops by this much
    seed: Optional[int] = None


class DifferentialFuzzer:
    """
    Black-box differential fuzz tester for HDC models.

    Compares original prediction with predictions on mutated
    inputs. If predictions differ, the mutation is flagged as
    a potential adversarial example.

    From the paper:
    "HDXplore takes the original input image without necessarily
    knowing its label. A set of mutation algorithms are then
    applied to create perturbed input. The perturbed input along
    with the original input are fed into the HDC model together
    for predictions. If the labels are inconsistent, the input
    is included in the adversarial set."
    """

    def __init__(
        self,
        model: nn.Module,
        encode_fn: Callable[[torch.Tensor], torch.Tensor],
        config: Optional[FuzzConfig] = None,
    ):
        self.model = model
        self.encode_fn = encode_fn  # Function to encode raw input → HV
        self.config = config or FuzzConfig()

        # Statistics
        self.total_queries = 0
        self.adversarial_found = 0
        self.mutation_history: List[dict] = []

    def mutate_pixel_flip(
        self,
        sample: torch.Tensor,
        rate: Optional[float] = None,
    ) -> torch.Tensor:
        """Flip random pixels (for image-like inputs).

        Args:
            sample: Input tensor of any shape
            rate: Fraction of elements to flip

        Returns:
            Mutated sample
        """
        rate = rate or self.config.mutation_rate
        mutated = sample.clone()
        mask = torch.rand_like(sample) < rate
        # Flip: for continuous values, add noise; for binary, flip
        if sample.dtype in (torch.float32, torch.float64):
            mutated[mask] = mutated[mask] + torch.randn(
                mask.sum().item()
            ) * self.config.mutation_magnitude
        else:
            mutated[mask] = 1 - mutated[mask]
        return mutated

    def mutate_gaussian(
        self,
        sample: torch.Tensor,
        rate: Optional[float] = None,
    ) -> torch.Tensor:
        """Add Gaussian noise to random elements.

        Args:
            sample: Input tensor
            rate: Fraction of elements to mutate

        Returns:
            Mutated sample
        """
        rate = rate or self.config.mutation_rate
        mutated = sample.clone()
        mask = torch.rand_like(sample) < rate
        noise = torch.randn(mask.sum().item()) * self.config.mutation_magnitude
        mutated[mask] = mutated[mask] + noise
        return mutated

    def mutate_salt_pepper(
        self,
        sample: torch.Tensor,
        rate: Optional[float] = None,
    ) -> torch.Tensor:
        """Salt-and-pepper noise: set elements to min or max.

        Args:
            sample: Input tensor
            rate: Fraction of elements to mutate

        Returns:
            Mutated sample
        """
        rate = rate or self.config.mutation_rate
        mutated = sample.clone()
        mask = torch.rand_like(sample) < rate
        salt = torch.rand(mask.sum().item()) > 0.5
        mn, mx = sample.min(), sample.max()
        values = torch.where(
            salt,
            torch.full_like(mutated[mask], mx),
            torch.full_like(mutated[mask], mn),
        )
        mutated[mask] = values
        return mutated

    def mutate(
        self,
        sample: torch.Tensor,
        mutation_type: Optional[str] = None,
    ) -> torch.Tensor:
        """Apply mutation to sample.

        Args:
            sample: Input tensor
            mutation_type: Type of mutation to apply

        Returns:
            Mutated sample
        """
        mt = mutation_type or self.config.mutation_type

        if mt == "pixel_flip":
            return self.mutate_pixel_flip(sample)
        elif mt == "gaussian":
            return self.mutate_gaussian(sample)
        elif mt == "salt_pepper":
            return self.mutate_salt_pepper(sample)
        else:
            return self.mutate_gaussian(sample)

    def test_sample(
        self,
        sample: torch.Tensor,
        n_mutations: Optional[int] = None,
    ) -> List[dict]:
        """Test a single sample with multiple mutations.

        Args:
            sample: Input tensor
            n_mutations: Number of mutations to try

        Returns:
            List of adversarial findings
        """
        n_mutations = n_mutations or self.config.n_mutations
        self.model.eval()

        # Original prediction
        with torch.no_grad():
            original_hv = self.encode_fn(sample)
            original_pred = self.model.predict(original_hv)

        findings = []

        for _ in range(n_mutations):
            if self.total_queries >= self.config.max_queries:
                break

            mutated = self.mutate(sample)
            with torch.no_grad():
                mutated_hv = self.encode_fn(mutated)
                mutated_pred = self.model.predict(mutated_hv)

            self.total_queries += 1

            if original_pred != mutated_pred:
                self.adversarial_found += 1
                finding = {
                    "original_pred": original_pred,
                    "mutated_pred": mutated_pred,
                    "sample": sample.clone(),
                    "mutated": mutated.clone(),
                    "mutation_rate": self.config.mutation_rate,
                }
                findings.append(finding)
                self.mutation_history.append(finding)

        return findings

    def test_batch(
        self,
        samples: torch.Tensor,
        n_mutations_per: Optional[int] = None,
    ) -> dict:
        """Test a batch of samples.

        Args:
            samples: (B, ...) input samples
            n_mutations_per: Mutations per sample

        Returns:
            Summary statistics
        """
        all_findings = []
        for i in range(samples.shape[0]):
            findings = self.test_sample(samples[i], n_mutations_per)
            all_findings.extend(findings)

        return {
            "total_queries": self.total_queries,
            "adversarial_found": self.adversarial_found,
            "adversarial_rate": self.adversarial_found / max(1, self.total_queries),
            "findings": all_findings,
        }

    def get_report(self) -> dict:
        """Get fuzzing report."""
        return {
            "total_queries": self.total_queries,
            "adversarial_found": self.adversarial_found,
            "adversarial_rate": (
                self.adversarial_found / max(1, self.total_queries)
            ),
            "config": {
                "mutation_type": self.config.mutation_type,
                "mutation_rate": self.config.mutation_rate,
                "max_queries": self.config.max_queries,
            },
        }


class HDXplore:
    """
    HDXplore: Automated HDC robustness testing tool.

    Wraps DifferentialFuzzer with additional analysis:
    - Identifies common mutation patterns that cause failures
    - Estimates robustness score (adversarial resistance)
    - Generates adversarial training data

    From the paper Section III-B:
    "We develop HDXplore, a highly automated testing tool
    specifically designed for HDC based on differential testing."
    """

    def __init__(
        self,
        model: nn.Module,
        encode_fn: Callable,
        config: Optional[FuzzConfig] = None,
    ):
        self.model = model
        self.encode_fn = encode_fn
        self.config = config or FuzzConfig()
        self.fuzzer = DifferentialFuzzer(model, encode_fn, config)

        # Collected adversarial examples for retraining
        self.adversarial_examples: List[Tuple[torch.Tensor, int]] = []

    def explore(
        self,
        samples: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        n_mutations_per: int = 100,
    ) -> dict:
        """Run full exploration on a set of samples.

        Args:
            samples: (B, ...) input samples
            labels: (B,) ground truth labels (optional)
            n_mutations_per: Mutations per sample

        Returns:
            Exploration report
        """
        report = self.fuzzer.test_batch(samples, n_mutations_per)

        # Collect adversarial examples
        for finding in report["findings"]:
            self.adversarial_examples.append(
                (finding["mutated"], finding["mutated_pred"])
            )

        # Compute robustness score
        robustness = 1.0 - report["adversarial_rate"]
        report["robustness_score"] = robustness

        # Label-based analysis if labels provided
        if labels is not None:
            report["label_available"] = True
            # Compute per-class vulnerability
            class_vuln = {}
            for finding in report["findings"]:
                c = finding["original_pred"]
                class_vuln[c] = class_vuln.get(c, 0) + 1
            report["class_vulnerability"] = class_vuln

        return report

    def generate_adversarial_dataset(self) -> torch.Tensor:
        """Return collected adversarial examples as a dataset.

        Returns:
            Tensor of adversarial examples
        """
        if not self.adversarial_examples:
            return torch.empty(0)
        return torch.stack([ex[0] for ex in self.adversarial_examples])

    def adversarial_retraining(
        self,
        clean_samples: torch.Tensor,
        clean_labels: torch.Tensor,
        n_epochs: int = 3,
    ):
        """Retrain model with adversarial examples mixed in.

        Args:
            clean_samples: Original training data
            clean_labels: Original labels
            n_epochs: Number of retraining epochs
        """
        adversarial = self.generate_adversarial_dataset()
        if adversarial.numel() == 0:
            return

        # Simple retraining: add adversarial examples to training
        all_samples = torch.cat([clean_samples, adversarial])
        adv_labels = torch.tensor(
            [ex[1] for ex in self.adversarial_examples]
        )
        all_labels = torch.cat([clean_labels, adv_labels])

        self.model.train()
        for _ in range(n_epochs):
            for i in range(all_samples.shape[0]):
                hv = self.encode_fn(all_samples[i])
                self.model.memory.add(hv, all_labels[i].item())
        self.model.memory.renormalize()


# ── Tests ────────────────────────────────────────────────────────────────────
def test_hdxplore():
    """Verify HDXplore fuzz testing."""
    print("=" * 60)
    print("Testing HDXplore: Differential Fuzz Testing for HDC")
    print("=" * 60)

    from models.hdc import HDCEncoder

    # Create a simple HDC model
    dim = 500
    n_classes = 5
    input_size = 20

    model = HDCEncoder(
        input_size=input_size,
        n_classes=n_classes,
        dim=dim,
        n_levels=13,
    )

    # Train with some data
    X = torch.randn(50, input_size)
    y = torch.randint(0, n_classes, (50,))
    model.train()
    for i in range(50):
        model.train_step(X[i], y[i].item())
    model.finalize()
    model.eval()

    # Fuzz test
    config = FuzzConfig(
        n_mutations=20,
        mutation_type="gaussian",
        mutation_rate=0.1,
        max_queries=200,
    )

    hdxplore = HDXplore(
        model=model,
        encode_fn=lambda x: model.encode(x),
        config=config,
    )

    test_samples = torch.randn(10, input_size)
    test_labels = torch.randint(0, n_classes, (10,))

    report = hdxplore.explore(test_samples, test_labels, n_mutations_per=20)

    print(f"\n  Total queries: {report['total_queries']}")
    print(f"  Adversarial found: {report['adversarial_found']}")
    print(f"  Adversarial rate: {report['adversarial_rate']:.3f}")
    print(f"  Robustness score: {report['robustness_score']:.3f}")

    if report.get("class_vulnerability"):
        print(f"  Per-class vulnerability: {report['class_vulnerability']}")

    # Test mutation types
    sample = torch.randn(input_size)
    fuzzer = DifferentialFuzzer(model, lambda x: model.encode(x), config)

    for mt in ["pixel_flip", "gaussian", "salt_pepper"]:
        mutated = fuzzer.mutate(sample, mt)
        diff = (mutated - sample).abs().sum().item()
        print(f"  {mt:>12}: total change = {diff:.3f}")

    print("\n✅ HDXplore test complete!")


if __name__ == "__main__":
    test_hdxplore()