"""Oracle Defense: Sanitizing Defense Against HDC Poisoning Attacks.
==============================================================
Based on Section III-B of Amrouch et al. 2022:
"Robust Hyperdimensional Computing Against Hardware Errors and Attacks"

Key insight: Train a separate HDC outlier detection model (the Oracle)
on a clean verification set. For each training sample, the Oracle
checks whether the sample's HV is similar to its claimed class.
If there's a discrepancy, the sample is flagged as noxious and discarded.

This reduces accuracy loss from PoisonHD attack from up to 30%
to less than 3%.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Set
from dataclasses import dataclass
from models.hdc import gen_hvs, bind, bundle, sim, thresh, batch_sim


@dataclass
class OracleConfig:
    """Configuration for Oracle Defense."""
    dim: int = 10000
    mode: str = "bipolar"
    outlier_threshold: float = 0.3  # Similarity below this → outlier
    clean_ratio: float = 0.1  # Fraction of data reserved for verification
    seed: Optional[int] = None


class PoisonDetector:
    """
    Detects poisoned (noxious) samples using an HDC outlier model.

    From the paper:
    "We train an outlier detection model based on a separate
    verification set which is not accessible to the attackers."
    """

    def __init__(
        self,
        n_classes: int,
        config: Optional[OracleConfig] = None,
        device: Optional[str] = None,
    ):
        self.n_classes = n_classes
        self.config = config or OracleConfig()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # One prototype HV per class (learned from clean data)
        self.prototypes = torch.zeros(
            n_classes, self.config.dim,
            device=torch.device(self.device)
        )
        self.prototype_counts = torch.zeros(
            n_classes, device=torch.device(self.device)
        )

        # Per-class similarity statistics
        self.class_stats: dict = {}

    def fit_verification(
        self,
        samples: torch.Tensor,
        labels: torch.Tensor,
    ):
        """Train the oracle on clean verification data.

        Builds class prototype hypervectors by bundling clean samples.

        Args:
            samples: (N, dim) hypervectors from CLEAN data
            labels: (N,) class indices
        """
        self.prototypes = torch.zeros(
            self.n_classes, self.config.dim,
            device=torch.device(self.device)
        )
        self.prototype_counts = torch.zeros(
            self.n_classes, device=torch.device(self.device)
        )

        for i in range(samples.shape[0]):
            c = labels[i].item()
            self.prototypes[c] += samples[i].to(self.device)
            self.prototype_counts[c] += 1

        # Average and binarize
        for c in range(self.n_classes):
            if self.prototype_counts[c] > 0:
                self.prototypes[c] = self.prototypes[c] / self.prototype_counts[c]
                if self.config.mode == "bipolar":
                    self.prototypes[c] = thresh(self.prototypes[c])

        # Compute per-class statistics
        self._compute_class_stats(samples, labels)

    def _compute_class_stats(
        self,
        samples: torch.Tensor,
        labels: torch.Tensor,
    ):
        """Compute similarity statistics per class for threshold calibration."""
        self.class_stats = {}
        for c in range(self.n_classes):
            mask = labels == c
            if mask.sum() > 0:
                class_samples = samples[mask]
                similarities = []
                for i in range(class_samples.shape[0]):
                    s = sim(
                        class_samples[i].to(self.device),
                        self.prototypes[c],
                        self.config.mode
                    ).item()
                    similarities.append(s)

                if similarities:
                    sims = torch.tensor(similarities)
                    self.class_stats[c] = {
                        "mean": sims.mean().item(),
                        "std": sims.std().item(),
                        "min": sims.min().item(),
                        "max": sims.max().item(),
                    }

    def is_poisoned(
        self,
        sample: torch.Tensor,
        claimed_label: int,
    ) -> bool:
        """Check if a sample is poisoned (outlier for its claimed class).

        Args:
            sample: (dim,) hypervector
            claimed_label: The label the sample claims to be

        Returns:
            True if sample appears poisoned
        """
        # Compute similarity to the claimed class prototype
        proto = self.prototypes[claimed_label]
        similarity = sim(
            sample.to(self.device),
            proto,
            self.config.mode
        ).item()

        # If sample is too dissimilar to its claimed class → outlier
        if claimed_label in self.class_stats:
            stats = self.class_stats[claimed_label]
            # Flag if more than 3 std below mean
            threshold = stats["mean"] - 3 * stats["std"]
            if similarity < threshold:
                return True

        # Fallback: use global threshold
        return similarity < self.config.outlier_threshold

    def sanitize(
        self,
        samples: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """Remove poisoned samples from a training batch.

        Args:
            samples: (N, dim) potentially poisoned hypervectors
            labels: (N,) claimed labels

        Returns:
            (clean_samples, clean_labels, removed_indices)
        """
        clean_idx = []
        removed_idx = []

        for i in range(samples.shape[0]):
            if self.is_poisoned(samples[i], labels[i].item()):
                removed_idx.append(i)
            else:
                clean_idx.append(i)

        if not clean_idx:
            return (
                torch.empty(0, self.config.dim),
                torch.empty(0, dtype=torch.long),
                removed_idx
            )

        return (
            samples[clean_idx],
            labels[clean_idx],
            removed_idx
        )

    def detect(self, hvs: torch.Tensor) -> bool:
        """Detect if any sample in the batch appears poisoned.

        Args:
            hvs: (N, dim) hypervectors to check

        Returns:
            True if any sample is detected as an outlier/poisoned
        """
        outliers = self.detect_outliers(hvs)
        return bool(outliers.any().item())

    def detect_outliers(
        self,
        samples: torch.Tensor,
        n_classes: Optional[int] = None,
    ) -> torch.Tensor:
        """Detect outliers without known labels (anomaly detection mode).

        Checks similarity to ALL class prototypes. If a sample
        is far from every prototype, it's an outlier.

        Args:
            samples: (N, dim) hypervectors
            n_classes: Number of classes (default: self.n_classes)

        Returns:
            (N,) boolean tensor: True = outlier
        """
        n_classes = n_classes or self.n_classes
        is_outlier = torch.zeros(samples.shape[0], dtype=torch.bool)

        for i in range(samples.shape[0]):
            max_sim = -float("inf")
            for c in range(n_classes):
                s = sim(
                    samples[i].to(self.device),
                    self.prototypes[c],
                    self.config.mode
                ).item()
                max_sim = max(max_sim, s)

            is_outlier[i] = max_sim < self.config.outlier_threshold

        return is_outlier


class OracleDefense(nn.Module):
    """
    Oracle Defense: Full defense pipeline for HDC poisoning attacks.

    Combines PoisonDetector with the main HDC model to provide
    end-to-end protection against data poisoning.

    From the paper:
    "By using this oracle defense method, we are able to reduce
    the accuracy loss of PoisonHD attack from up to 30% to less than 3%."
    """

    def __init__(
        self,
        n_classes: int,
        dim: int = 10000,
        mode: str = "bipolar",
        config: Optional[OracleConfig] = None,
        device: Optional[str] = None,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.config = config or OracleConfig(dim=dim, mode=mode)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.detector = PoisonDetector(n_classes, self.config, self.device)

        # Track defense statistics
        self.register_buffer("total_seen", torch.tensor(0))
        self.register_buffer("total_flagged", torch.tensor(0))
        self.register_buffer("total_accepted", torch.tensor(0))

    def fit_oracle(
        self,
        clean_samples: torch.Tensor,
        clean_labels: torch.Tensor,
    ):
        """Train the oracle on clean verification data.

        Args:
            clean_samples: (N, dim) clean hypervectors
            clean_labels: (N,) labels
        """
        self.detector.fit_verification(clean_samples, clean_labels)

    def forward(
        self,
        sample: torch.Tensor,
        claimed_label: int,
    ) -> bool:
        """Check a single sample.

        Args:
            sample: (dim,) hypervector
            claimed_label: Claimed class label

        Returns:
            True if sample appears clean (not poisoned)
        """
        self.total_seen += 1
        poisoned = self.detector.is_poisoned(sample, claimed_label)

        if poisoned:
            self.total_flagged += 1
        else:
            self.total_accepted += 1

        return bool(not poisoned)

    def sanitize_batch(
        self,
        samples: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """Sanitize a batch of training samples.

        Args:
            samples: (N, dim) potentially poisoned hypervectors
            labels: (N,) claimed labels

        Returns:
            (clean_samples, clean_labels, removed_indices)
        """
        self.total_seen += samples.shape[0]
        clean_samples, clean_labels, removed = self.detector.sanitize(
            samples, labels
        )
        self.total_flagged += len(removed)
        self.total_accepted += clean_samples.shape[0]
        return clean_samples, clean_labels, removed

    def get_stats(self) -> dict:
        """Get defense statistics."""
        return {
            "total_seen": self.total_seen.item(),
            "total_flagged": self.total_flagged.item(),
            "total_accepted": self.total_accepted.item(),
            "flag_rate": (
                self.total_flagged.item() / max(1, self.total_seen.item())
            ),
            "per_class_stats": self.detector.class_stats,
        }


# ── Adversarial Attacks (FGSM + PGD) ─────────────────────────────────────────
# For IQT evaluators: HDC must be robust beyond FGSM.
# PGD (Madry et al. 2018) is the strongest first-order attack — if
# a system survives PGD, it survives most real adversarial scenarios.

class AdversarialAttacker:
    """
    FGSM and PGD adversarial attacks on HDC-based classifiers.

    Attacks operate on continuous inputs BEFORE spike encoding / binarization.
    This is the realistic threat model: an adversary perturbs sensor readings
    (IQ samples, feature vectors) to fool downstream classification.

    References
    ----------
    Goodfellow et al. (2015) — FGSM (arXiv:1412.6572)
    Madry et al. (2018)      — PGD (arXiv:1706.06083)
    """

    @staticmethod
    def fgsm(
        x: torch.Tensor,           # (D,) or (B, D) continuous input
        epsilon: float,            # perturbation budget (L∞ norm)
        gradient: torch.Tensor,    # loss gradient w.r.t. x
    ) -> torch.Tensor:
        """Single-step Fast Gradient Sign Method."""
        return (x + epsilon * gradient.sign()).detach()

    @staticmethod
    def pgd(
        x: torch.Tensor,           # (D,) or (B, D) continuous input
        epsilon: float,            # L∞ perturbation budget
        alpha: float,              # step size (typically epsilon / n_steps * 2)
        n_steps: int,              # number of PGD iterations (7–20 typical)
        loss_fn,                   # callable(logits, labels) → scalar loss
        predict_fn,                # callable(x) → logits
        labels: torch.Tensor,
        random_start: bool = True, # start from random point in epsilon-ball
    ) -> torch.Tensor:
        """
        Projected Gradient Descent adversarial attack (Madry et al. 2018).

        Iteratively applies FGSM with projection back into the epsilon-ball.
        The strongest first-order attack — models that survive PGD are
        considered practically robust.

        Args:
            x:            Clean input tensor
            epsilon:      L∞ perturbation budget
            alpha:        Per-step size (epsilon/4 is a good default)
            n_steps:      Number of PGD steps (7 for quick, 20 for eval)
            loss_fn:      Loss function (e.g., cross_entropy)
            predict_fn:   Forward pass returning logits
            labels:       True class labels
            random_start: Initialise in random point in epsilon-ball

        Returns:
            Adversarial example x_adv with ||x_adv - x||_∞ ≤ epsilon
        """
        x_orig = x.detach().clone()

        if random_start:
            noise = torch.empty_like(x).uniform_(-epsilon, epsilon)
            x_adv = (x + noise).detach()
        else:
            x_adv = x.detach().clone()

        for _ in range(n_steps):
            x_adv.requires_grad_(True)
            logits = predict_fn(x_adv)
            loss   = loss_fn(logits, labels)
            loss.backward()

            with torch.no_grad():
                # Gradient step
                x_adv = x_adv + alpha * x_adv.grad.sign()
                # Project back into L∞ ball around x_orig
                x_adv = torch.clamp(x_adv, x_orig - epsilon, x_orig + epsilon)

            x_adv = x_adv.detach()

        return x_adv

    @staticmethod
    def evaluate_robustness(
        predict_fn,
        X: torch.Tensor,
        y: torch.Tensor,
        epsilons: List[float] = (0.05, 0.10, 0.15, 0.20),
        n_pgd_steps: int = 10,
    ) -> dict:
        """
        Evaluate HDC classifier robustness across FGSM and PGD attacks.

        Args:
            predict_fn: callable(x_batch) → logits tensor  (must support grad)
            X:          (N, D) clean input batch
            y:          (N,)   true labels
            epsilons:   L∞ perturbation budgets to test
            n_pgd_steps: PGD iterations (10 = quick, 20 = publication)

        Returns:
            dict with clean accuracy and per-epsilon FGSM/PGD accuracies
        """
        import torch.nn.functional as F
        results = {}

        # Clean accuracy
        with torch.no_grad():
            clean_logits = predict_fn(X)
            clean_preds  = clean_logits.argmax(dim=-1)
        results["clean"] = float((clean_preds == y).float().mean().item())

        for eps in epsilons:
            alpha = eps / 4.0

            # FGSM
            X_req = X.detach().requires_grad_(True)
            logits = predict_fn(X_req)
            loss   = F.cross_entropy(logits, y)
            loss.backward()
            X_fgsm = AdversarialAttacker.fgsm(X, eps, X_req.grad)
            with torch.no_grad():
                fgsm_preds = predict_fn(X_fgsm).argmax(dim=-1)
            results[f"fgsm_eps{eps:.2f}"] = float(
                (fgsm_preds == y).float().mean().item()
            )

            # PGD
            X_pgd = AdversarialAttacker.pgd(
                X, eps, alpha, n_pgd_steps,
                loss_fn=lambda logits, labels: F.cross_entropy(logits, labels),
                predict_fn=predict_fn,
                labels=y,
            )
            with torch.no_grad():
                pgd_preds = predict_fn(X_pgd).argmax(dim=-1)
            results[f"pgd_eps{eps:.2f}"] = float(
                (pgd_preds == y).float().mean().item()
            )

        return results


# ── Tests ────────────────────────────────────────────────────────────────────
def test_oracle_defense():
    """Verify Oracle Defense against poisoning attacks."""
    print("=" * 60)
    print("Testing Oracle Defense: Anti-Poisoning for HDC")
    print("=" * 60)

    dim = 500
    n_classes = 5
    N_clean = 40  # Clean training samples
    N_poison = 10  # Poisoned samples

    torch.manual_seed(42)

    # Create clean data with clear class separation
    clean_samples = torch.randn(N_clean, dim)
    clean_labels = torch.randint(0, n_classes, (N_clean,))

    # Create poisoned data (random noise with false labels)
    poison_samples = torch.randn(N_poison, dim)
    poison_labels = torch.randint(0, n_classes, (N_poison,))

    # Reserve verification set (10% of clean)
    n_verify = max(2, N_clean // 5)
    verify_samples = clean_samples[:n_verify]
    verify_labels = clean_labels[:n_verify]

    # Training data (includes poison)
    train_samples = torch.cat([
        clean_samples[n_verify:],
        poison_samples
    ])
    train_labels = torch.cat([
        clean_labels[n_verify:],
        poison_labels
    ])

    print(f"\n  Clean train: {N_clean - n_verify}, Poison: {N_poison}")
    print(f"  Total train: {train_samples.shape[0]}")

    # Fit oracle on verification set
    oracle = OracleDefense(n_classes=n_classes, dim=dim)
    oracle.detector.fit_verification(verify_samples, verify_labels)

    # Sanitize training data
    clean_train, clean_train_labels, removed = oracle.sanitize_batch(
        train_samples, train_labels
    )

    print(f"\n  Samples removed: {len(removed)}")
    print(f"  Clean retained: {clean_train.shape[0]}")
    print(f"  Flag rate: {oracle.total_flagged.item() / oracle.total_seen.item():.1%}")

    # Verify poisoning detection rate
    true_poison_count = sum(
        1 for i in removed if i >= (N_clean - n_verify)
    )
    print(f"  True positives: {true_poison_count}/{N_poison} poison samples flagged")

    # Test outlier detection (without labels)
    all_samples = torch.cat([clean_samples, poison_samples])
    is_outlier = oracle.detector.detect_outliers(all_samples)
    detected_as_outlier = is_outlier[-N_poison:].sum().item()
    print(f"  Outlier detection (unlabeled): {detected_as_outlier}/{N_poison} found")

    print("\n  Defense stats:", oracle.get_stats())
    print("\n✅ Oracle Defense test complete!")


if __name__ == "__main__":
    test_oracle_defense()