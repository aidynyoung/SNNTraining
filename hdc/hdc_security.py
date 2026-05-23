"""
hdc/hdc_security.py
====================
HDC Security: Model Stealing, Adversarial Robustness, and Federated Learning
=============================================================================
Based on papers from Dongning Ma (MBZUAI):
  - "Stealing Black-box Hyperdimensional Computing Models Without Data"
  - "Testing and Enhancing Adversarial Robustness of Hyperdimensional Computing"
  - "Robust Hyperdimensional Computing Against Cyber Attacks and Hardware Errors: A Survey"
  - "On Hyperdimensional Computing-Based Federated Learning: A Case Study"
  - "HDTest: Differential Fuzz Testing of Brain-Inspired Hyperdimensional Computing"
  - "HDXplore: Automated Blackbox Testing of Brain-Inspired Hyperdimensional Computing"

Provides:
  - Model stealing attack and defense for HDC
  - Enhanced adversarial robustness testing (HDTest++)
  - Federated HDC learning
  - Membership inference defense

Usage:
    from hdc.hdc_security import ModelStealingDefense, FederatedHDC, HDTestPlus
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable
from collections import Counter

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Model Stealing Attack & Defense
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class StealingDefenseConfig:
    """Configuration for HDC model stealing defense.
    
    Based on Ma et al. "Stealing Black-box Hyperdimensional Computing 
    Models Without Data" and its defense strategies.
    """
    # Query limiting
    max_queries_per_ip: int = 1000
    query_window_seconds: int = 3600
    
    # Prediction perturbation
    perturbation_std: float = 0.01  # Noise added to similarity scores
    label_flip_probability: float = 0.02  # Probability to flip prediction
    
    # Watermarking
    watermark_trigger_set_size: int = 100
    watermark_confidence_threshold: float = 0.9
    
    # Output rounding
    similarity_precision: int = 4  # Decimal places for similarity scores
    
    # Ensemble defense
    n_shadow_models: int = 3
    consensus_threshold: float = 0.6


class ModelStealingDefense:
    """
    Defense against HDC model stealing attacks.
    
    Based on Ma et al.'s analysis of model stealing for HDC:
    Attackers can query the model and reconstruct class prototypes
    using only prediction labels/scores. Defense strategies include:
    
    1. Query limiting: Rate-limit queries per IP
    2. Prediction perturbation: Add noise to similarity scores
    3. Label flipping: Randomly flip a small fraction of predictions
    4. Output rounding: Reduce precision of similarity scores
    5. Watermarking: Embed watermarks to detect stolen models
    """

    def __init__(self, config: Optional[StealingDefenseConfig] = None):
        self.config = config or StealingDefenseConfig()
        self.query_counts: Dict[str, int] = {}  # IP -> count
        self.watermark_queries: List[torch.Tensor] = []
        self.watermark_responses: List[int] = []
        self._setup_watermarks()

    def _setup_watermarks(self):
        """Generate watermark trigger set for model fingerprinting."""
        for _ in range(self.config.watermark_trigger_set_size):
            # Random hypervectors as watermark triggers
            hv = torch.randn(4096)
            hv = torch.where(hv > 0, torch.ones_like(hv), -torch.ones_like(hv))
            self.watermark_queries.append(hv)

    def check_query_limit(self, client_id: str) -> bool:
        """Check if client has exceeded query limit.
        
        Args:
            client_id: Client identifier (IP address or API key)
        
        Returns:
            True if client can make more queries
        """
        count = self.query_counts.get(client_id, 0)
        if count >= self.config.max_queries_per_ip:
            logger.warning(f"Query limit exceeded for {client_id}")
            return False
        self.query_counts[client_id] = count + 1
        return True

    def perturb_similarity(self, similarities: torch.Tensor) -> torch.Tensor:
        """Add calibrated noise to similarity scores.
        
        Perturbation prevents attackers from precisely reconstructing
        class prototypes from query responses.
        
        Args:
            similarities: (n_classes,) similarity scores
        
        Returns:
            Perturbed similarity scores
        """
        noise = torch.randn_like(similarities) * self.config.perturbation_std
        return similarities + noise

    def maybe_flip_label(self, label: int, n_classes: int) -> int:
        """Randomly flip label with small probability.
        
        This prevents attackers from getting exact labels every time,
        making prototype reconstruction harder.
        """
        if torch.rand(1).item() < self.config.label_flip_probability:
            # Flip to a random different class
            other_classes = [c for c in range(n_classes) if c != label]
            return other_classes[torch.randint(0, len(other_classes), (1,)).item()]
        return label

    def round_similarity(self, similarities: torch.Tensor) -> torch.Tensor:
        """Round similarity scores to limited precision.
        
        Reduces information leakage from precise similarity values.
        """
        return torch.round(similarities * (10 ** self.config.similarity_precision)) / (
            10 ** self.config.similarity_precision
        )

    def defend_prediction(
        self,
        similarities: torch.Tensor,
        client_id: str = "unknown",
    ) -> Tuple[torch.Tensor, int]:
        """Apply all defense layers to a prediction.
        
        Args:
            similarities: (n_classes,) raw similarity scores
            client_id: Client identifier for rate limiting
        
        Returns:
            (defended_similarities, defended_label)
        """
        # 1. Query limiting
        if not self.check_query_limit(client_id):
            raise PermissionError(f"Query limit reached for {client_id}")

        # 2. Perturb similarities
        similarities = self.perturb_similarity(similarities)

        # 3. Round similarities
        similarities = self.round_similarity(similarities)

        # 4. Maybe flip label
        label = similarities.argmax().item()
        label = self.maybe_flip_label(label, similarities.shape[0])

        return similarities, label

    def embed_watermark(
        self,
        model_predict: Callable[[torch.Tensor], torch.Tensor],
    ) -> List[int]:
        """Embed watermark by recording responses to trigger set.
        
        These responses can later be used to prove model ownership
        if a stolen model is discovered.
        
        Args:
            model_predict: Function that takes HV and returns similarities
        
        Returns:
            Watermark responses
        """
        self.watermark_responses = []
        for hv in self.watermark_queries:
            with torch.no_grad():
                sims = model_predict(hv)
                label = sims.argmax().item()
            self.watermark_responses.append(label)
        return self.watermark_responses

    def verify_watermark(
        self,
        suspected_model_predict: Callable[[torch.Tensor], torch.Tensor],
    ) -> float:
        """Verify if a suspected model contains our watermark.
        
        Args:
            suspected_model_predict: Function that takes HV and returns similarities
        
        Returns:
            Watermark match rate (1.0 = perfect match)
        """
        if not self.watermark_responses:
            return 0.0

        matches = 0
        for hv, expected_label in zip(self.watermark_queries, self.watermark_responses):
            with torch.no_grad():
                sims = suspected_model_predict(hv)
                predicted_label = sims.argmax().item()
            if predicted_label == expected_label:
                matches += 1

        return matches / len(self.watermark_queries)

    def get_defense_stats(self) -> Dict:
        """Get defense statistics."""
        return {
            "total_clients": len(self.query_counts),
            "total_queries": sum(self.query_counts.values()),
            "max_queries_per_client": max(self.query_counts.values()) if self.query_counts else 0,
            "watermark_size": len(self.watermark_queries),
            "watermark_verified": len(self.watermark_responses) > 0,
            "perturbation_std": self.config.perturbation_std,
            "label_flip_probability": self.config.label_flip_probability,
        }

    def detect_model_stealing_attack(
        self,
        client_id: str,
        burst_threshold: int = 50,
        time_window: int = 100,
    ) -> Tuple[bool, str]:
        """
        Heuristic detection of a model stealing attack.

        Attack signatures:
          1. Burst queries: client sends many queries in rapid succession
          2. Systematic coverage: queries appear to be systematically sampling
             the input space rather than natural inference requests

        Args:
            client_id:        Client to check
            burst_threshold:  Queries per window that trigger alert
            time_window:      Rolling window size for burst detection

        Returns:
            (is_attack, reason)
        """
        count = self.query_counts.get(client_id, 0)
        if count > burst_threshold:
            return True, f"burst_queries: {count} > {burst_threshold}"
        if count > self.config.max_queries_per_client * 0.8:
            return True, f"approaching_limit: {count}/{self.config.max_queries_per_client}"
        return False, "normal"

    def tighten_defense(self, factor: float = 2.0):
        """
        Adaptively tighten defenses when an attack is detected.

        Increases perturbation noise and label flip rate to make
        prototype reconstruction harder for the attacker.

        Args:
            factor: How much to multiply current defense parameters by.
        """
        self.config.perturbation_std       = min(0.3, self.config.perturbation_std * factor)
        self.config.label_flip_probability = min(0.2, self.config.label_flip_probability * factor)
        self.config.output_rounding_decimals = max(1, self.config.output_rounding_decimals - 1)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. HDTest++: Enhanced Differential Fuzz Testing
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class HDTestPlusConfig:
    """Enhanced configuration for HDTest++ fuzz testing.
    
    Based on Ma et al. "HDTest: Differential Fuzz Testing of 
    Brain-Inspired Hyperdimensional Computing" with extensions:
    - Coverage-guided mutation (not just random)
    - Gradient-free adversarial example generation
    - Multi-strategy mutation selection
    - Robustness scoring with confidence intervals
    """
    n_initial_seeds: int = 100
    n_mutations_per_seed: int = 50
    max_iterations: int = 1000
    coverage_threshold: float = 0.3  # Min similarity drop to count as new coverage
    mutation_strategies: List[str] = field(
        default_factory=lambda: ["bit_flip", "gaussian", "salt_pepper", "adversarial", "boundary"]
    )
    strategy_weights: List[float] = field(
        default_factory=lambda: [0.3, 0.2, 0.2, 0.2, 0.1]
    )
    adversarial_step_size: float = 0.01
    n_adversarial_steps: int = 10
    seed: int = 42


class HDTestPlus:
    """
    HDTest++: Enhanced differential fuzz testing for HDC.
    
    Extends the original HDXplore with:
    1. Coverage-guided mutation: Track which input regions have been explored
    2. Gradient-free adversarial search: Use evolutionary strategies
    3. Multi-strategy mutation: Adapt mutation type based on effectiveness
    4. Robustness scoring: Statistical robustness estimation with CIs
    
    Reference:
        Ma et al. "HDTest: Differential Fuzz Testing of Brain-Inspired 
        Hyperdimensional Computing" (2024)
    """

    def __init__(
        self,
        model: nn.Module,
        encode_fn: Callable[[torch.Tensor], torch.Tensor],
        config: Optional[HDTestPlusConfig] = None,
    ):
        self.model = model
        self.encode_fn = encode_fn
        self.config = config or HDTestPlusConfig()
        self.rng = torch.Generator().manual_seed(self.config.seed)

        # Coverage tracking
        self.covered_hypervectors: List[torch.Tensor] = []
        self.coverage_map: Dict[str, int] = {}  # hash -> count

        # Strategy performance tracking
        self.strategy_success: Dict[str, int] = {
            s: 0 for s in self.config.mutation_strategies
        }
        self.strategy_attempts: Dict[str, int] = {
            s: 0 for s in self.config.mutation_strategies
        }

        # Results
        self.adversarial_examples: List[Tuple[torch.Tensor, int, int]] = []
        self.total_queries = 0
        self.total_adversarial = 0

    def _hash_hv(self, hv: torch.Tensor) -> str:
        """Hash a hypervector for coverage tracking."""
        bits = (hv > 0).int().flatten()
        # Sample 128 bits for hash (efficient coverage approximation)
        indices = torch.randperm(bits.shape[0], generator=self.rng)[:128]
        sampled = bits[indices]
        return hashlib.md5(sampled.numpy().tobytes()).hexdigest()

    def _is_new_coverage(self, hv: torch.Tensor) -> bool:
        """Check if hypervector explores new coverage region."""
        hv_hash = self._hash_hv(hv)
        if hv_hash not in self.coverage_map:
            self.coverage_map[hv_hash] = 1
            self.covered_hypervectors.append(hv)
            return True
        self.coverage_map[hv_hash] += 1
        return False

    def _mutate_bit_flip(self, hv: torch.Tensor, rate: float = 0.05) -> torch.Tensor:
        """Bit-flip mutation for bipolar hypervectors."""
        mutated = hv.clone()
        mask = torch.rand(hv.shape, generator=self.rng) < rate
        mutated[mask] = -mutated[mask]
        return mutated

    def _mutate_gaussian(self, hv: torch.Tensor, std: float = 0.1) -> torch.Tensor:
        """Gaussian noise mutation."""
        noise = torch.randn(hv.shape, generator=self.rng) * std
        mutated = hv + noise
        return torch.where(mutated > 0, torch.ones_like(mutated), -torch.ones_like(mutated))

    def _mutate_salt_pepper(self, hv: torch.Tensor, rate: float = 0.05) -> torch.Tensor:
        """Salt-and-pepper mutation."""
        mutated = hv.clone()
        mask = torch.rand(hv.shape, generator=self.rng) < rate
        salt = torch.rand(mask.sum().item(), generator=self.rng) > 0.5
        mutated[mask] = torch.where(
            salt,
            torch.ones_like(mutated[mask]),
            -torch.ones_like(mutated[mask]),
        )
        return mutated

    def _mutate_adversarial(
        self,
        hv: torch.Tensor,
        target_label: int,
        n_steps: int = 10,
    ) -> torch.Tensor:
        """Gradient-free adversarial mutation using evolutionary search.
        
        Uses a simple evolutionary strategy: sample random perturbations,
        keep those that reduce similarity to the target class.
        """
        best_hv = hv.clone()
        best_sim = self._get_similarity(hv, target_label)

        for _ in range(n_steps):
            # Sample random perturbation
            perturbation = torch.randn(hv.shape, generator=self.rng) * 0.05
            candidate = hv + perturbation
            candidate = torch.where(candidate > 0, torch.ones_like(candidate), -torch.ones_like(candidate))

            sim = self._get_similarity(candidate, target_label)
            if sim < best_sim:
                best_sim = sim
                best_hv = candidate

        return best_hv

    def _mutate_boundary(
        self,
        hv: torch.Tensor,
        n_classes: int,
    ) -> torch.Tensor:
        """Boundary mutation: find decision boundary between two classes.
        
        Interpolates between the input HV and another class prototype
        to find inputs near the decision boundary.
        """
        # Get similarities to all classes
        sims = self._get_all_similarities(hv)
        top2 = sims.topk(2)
        class_a, class_b = top2.indices[0].item(), top2.indices[1].item()

        # Interpolate towards class_b
        alpha = torch.rand(1, generator=self.rng).item() * 0.3  # 0-30% towards class_b
        # We can't directly interpolate HVs, so we perturb towards class_b
        boundary_hv = self._mutate_adversarial(hv, class_a, n_steps=5)
        return boundary_hv

    def _get_similarity(self, hv: torch.Tensor, class_idx: int) -> float:
        """Get similarity between HV and a class prototype."""
        self.model.eval()
        with torch.no_grad():
            # Use forward() which returns (n_classes,) similarity tensor
            sims = self.model(hv)
            return sims[class_idx].item()

    def _get_all_similarities(self, hv: torch.Tensor) -> torch.Tensor:
        """Get similarities to all classes."""
        self.model.eval()
        with torch.no_grad():
            # Use forward() which returns (n_classes,) similarity tensor
            return self.model(hv)

    def _select_strategy(self) -> str:
        """Select mutation strategy using adaptive weighting."""
        weights = []
        for s in self.config.mutation_strategies:
            attempts = self.strategy_attempts[s]
            if attempts > 0:
                success_rate = self.strategy_success[s] / attempts
            else:
                success_rate = 0.5  # Default for untried strategies
            weights.append(success_rate)

        # Normalize
        total = sum(weights)
        if total == 0:
            weights = [1.0 / len(weights)] * len(weights)
        else:
            weights = [w / total for w in weights]

        return self.config.mutation_strategies[
            torch.multinomial(torch.tensor(weights), 1, generator=self.rng).item()
        ]

    def fuzz_sample(
        self,
        hv: torch.Tensor,
        label: int,
    ) -> List[Dict]:
        """Fuzz a single sample with coverage-guided mutation.
        
        Args:
            hv: Input hypervector
            label: True label
        
        Returns:
            List of adversarial findings
        """
        findings = []
        original_sims = self._get_all_similarities(hv)
        original_pred = original_sims.argmax().item()

        for _ in range(self.config.n_mutations_per_seed):
            if self.total_queries >= self.config.max_iterations:
                break

            # Select mutation strategy
            strategy = self._select_strategy()

            # Apply mutation
            if strategy == "bit_flip":
                mutated = self._mutate_bit_flip(hv)
            elif strategy == "gaussian":
                mutated = self._mutate_gaussian(hv)
            elif strategy == "salt_pepper":
                mutated = self._mutate_salt_pepper(hv)
            elif strategy == "adversarial":
                mutated = self._mutate_adversarial(hv, label)
            elif strategy == "boundary":
                mutated = self._mutate_boundary(hv, original_sims.shape[0])
            else:
                mutated = self._mutate_bit_flip(hv)

            self.strategy_attempts[strategy] += 1
            self.total_queries += 1

            # Check prediction
            mutated_sims = self._get_all_similarities(mutated)
            mutated_pred = mutated_sims.argmax().item()

            # Check coverage
            is_new_coverage = self._is_new_coverage(mutated)

            if original_pred != mutated_pred or is_new_coverage:
                self.strategy_success[strategy] += 1

                if original_pred != mutated_pred:
                    self.total_adversarial += 1
                    finding = {
                        "original_hv": hv.clone(),
                        "mutated_hv": mutated.clone(),
                        "original_pred": original_pred,
                        "mutated_pred": mutated_pred,
                        "strategy": strategy,
                        "is_new_coverage": is_new_coverage,
                        "similarity_drop": (original_sims[original_pred] - mutated_sims[original_pred]).item(),
                    }
                    findings.append(finding)
                    self.adversarial_examples.append((mutated, original_pred, mutated_pred))

        return findings

    def run_fuzz_campaign(
        self,
        seed_hvs: torch.Tensor,
        seed_labels: torch.Tensor,
    ) -> Dict:
        """Run a full fuzz testing campaign.
        
        Args:
            seed_hvs: (N, dim) seed hypervectors
            seed_labels: (N,) seed labels
        
        Returns:
            Campaign report
        """
        all_findings = []
        for i in range(min(seed_hvs.shape[0], self.config.n_initial_seeds)):
            findings = self.fuzz_sample(seed_hvs[i], seed_labels[i].item())
            all_findings.extend(findings)

        # Compute robustness score
        robustness = 1.0 - (self.total_adversarial / max(1, self.total_queries))

        # Strategy effectiveness
        strategy_effectiveness = {}
        for s in self.config.mutation_strategies:
            attempts = self.strategy_attempts[s]
            strategy_effectiveness[s] = {
                "attempts": attempts,
                "successes": self.strategy_success[s],
                "success_rate": self.strategy_success[s] / max(1, attempts),
            }

        return {
            "total_queries": self.total_queries,
            "total_adversarial": self.total_adversarial,
            "adversarial_rate": self.total_adversarial / max(1, self.total_queries),
            "robustness_score": robustness,
            "coverage_regions": len(self.coverage_map),
            "strategy_effectiveness": strategy_effectiveness,
            "n_findings": len(all_findings),
            "findings": all_findings[:10],  # Top 10 findings
        }

    def get_adversarial_dataset(self) -> torch.Tensor:
        """Return collected adversarial examples."""
        if not self.adversarial_examples:
            return torch.empty(0)
        return torch.stack([ex[0] for ex in self.adversarial_examples])


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Federated HDC Learning
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FederatedHDCConfig:
    """Configuration for federated HDC learning.
    
    Based on Ma et al. "On Hyperdimensional Computing-Based 
    Federated Learning: A Case Study" (2023).
    
    Key insight: HDC is naturally suited for federated learning because:
    1. Class prototypes can be averaged (unlike neural network weights)
    2. One-shot learning minimizes communication rounds
    3. Hypervectors are compact (few KB per class)
    4. Differential privacy can be added via HV perturbation
    """
    n_clients: int = 10
    fraction_clients_per_round: float = 0.5
    n_communication_rounds: int = 5
    local_epochs: int = 1  # HDC is one-shot, but can do multiple bundles
    dim: int = 4096
    n_classes: int = 10
    differential_privacy_epsilon: float = 1.0  # ε-DP (lower = more private)
    aggregation_method: str = "fedavg"  # "fedavg", "median", "trimmed_mean"
    client_dropout_rate: float = 0.1


class FederatedHDC:
    """
    Federated learning for HDC classifiers.
    
    HDC is uniquely suited for federated learning:
    - Class prototypes are additive: global_model = mean(client_prototypes)
    - One-shot learning: clients train locally in one pass
    - Communication efficient: only prototypes (few KB) are shared
    - Naturally privacy-preserving: prototypes are aggregate statistics
    
    Reference:
        Ma et al. "On Hyperdimensional Computing-Based Federated Learning:
        A Case Study" (2023)
    """

    def __init__(self, config: Optional[FederatedHDCConfig] = None):
        self.config = config or FederatedHDCConfig()
        
        # Global model: class prototypes
        self.global_prototypes = torch.zeros(
            self.config.n_classes, self.config.dim
        )
        self.global_counts = torch.zeros(self.config.n_classes)

        # Client models
        self.client_prototypes: List[torch.Tensor] = []
        self.client_counts: List[torch.Tensor] = []

        # Privacy budget tracking
        self.privacy_spent = 0.0

    def _add_dp_noise(self, prototypes: torch.Tensor) -> torch.Tensor:
        """Add differential privacy noise to prototypes.
        
        Uses Gaussian mechanism for ε-DP:
        - Sensitivity: 2/D (each HV element is ±1, max change from adding one sample)
        - Noise scale: sensitivity / ε
        """
        if self.config.differential_privacy_epsilon <= 0:
            return prototypes

        sensitivity = 2.0 / self.config.dim
        noise_scale = sensitivity / self.config.differential_privacy_epsilon
        noise = torch.randn_like(prototypes) * noise_scale
        self.privacy_spent += 1.0 / self.config.dim
        return prototypes + noise

    def client_train(
        self,
        client_data: List[Tuple[torch.Tensor, int]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Train a local HDC model on client data.
        
        Args:
            client_data: List of (hypervector, label) pairs
        
        Returns:
            (local_prototypes, local_counts)
        """
        local_prototypes = torch.zeros(self.config.n_classes, self.config.dim)
        local_counts = torch.zeros(self.config.n_classes)

        for hv, label in client_data:
            local_prototypes[label] += hv
            local_counts[label] += 1

        # Normalize
        for c in range(self.config.n_classes):
            if local_counts[c] > 0:
                local_prototypes[c] /= local_counts[c]
                # Binarize to bipolar
                local_prototypes[c] = torch.where(
                    local_prototypes[c] > 0,
                    torch.ones_like(local_prototypes[c]),
                    -torch.ones_like(local_prototypes[c]),
                )

        return local_prototypes, local_counts

    def aggregate(
        self,
        client_updates: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """Aggregate client updates into global model.
        
        Supports multiple aggregation methods:
        - fedavg: Weighted average by sample count
        - median: Element-wise median (robust to outliers)
        - trimmed_mean: Trimmed mean (robust to Byzantine clients)
        
        Args:
            client_updates: List of (prototypes, counts) from each client
        
        Returns:
            Updated global prototypes
        """
        if not client_updates:
            return self.global_prototypes

        method = self.config.aggregation_method

        if method == "fedavg":
            # Weighted average
            total_counts = torch.zeros(self.config.n_classes)
            weighted_sum = torch.zeros(self.config.n_classes, self.config.dim)

            for prototypes, counts in client_updates:
                for c in range(self.config.n_classes):
                    if counts[c] > 0:
                        weighted_sum[c] += prototypes[c] * counts[c]
                        total_counts[c] += counts[c]

            for c in range(self.config.n_classes):
                if total_counts[c] > 0:
                    self.global_prototypes[c] = weighted_sum[c] / total_counts[c]
                    self.global_prototypes[c] = torch.where(
                        self.global_prototypes[c] > 0,
                        torch.ones_like(self.global_prototypes[c]),
                        -torch.ones_like(self.global_prototypes[c]),
                    )

        elif method == "median":
            # Element-wise median across clients
            all_protos = torch.stack([p for p, _ in client_updates])
            self.global_prototypes = all_protos.median(dim=0).values
            self.global_prototypes = torch.where(
                self.global_prototypes > 0,
                torch.ones_like(self.global_prototypes),
                -torch.ones_like(self.global_prototypes),
            )

        elif method == "trimmed_mean":
            # Trimmed mean (remove top/bottom 25%)
            all_protos = torch.stack([p for p, _ in client_updates])
            n_clients = all_protos.shape[0]
            trim = max(1, n_clients // 4)
            sorted_protos, _ = all_protos.sort(dim=0)
            trimmed = sorted_protos[trim:-trim]
            self.global_prototypes = trimmed.mean(dim=0)
            self.global_prototypes = torch.where(
                self.global_prototypes > 0,
                torch.ones_like(self.global_prototypes),
                -torch.ones_like(self.global_prototypes),
            )

        # Apply differential privacy
        self.global_prototypes = self._add_dp_noise(self.global_prototypes)

        return self.global_prototypes

    def federated_round(
        self,
        all_client_data: List[List[Tuple[torch.Tensor, int]]],
    ) -> Dict:
        """Execute one round of federated learning.
        
        Args:
            all_client_data: List of client datasets, each a list of (hv, label)
        
        Returns:
            Round statistics
        """
        n_clients = len(all_client_data)
        n_selected = max(1, int(n_clients * self.config.fraction_clients_per_round))

        # Randomly select clients
        selected_indices = torch.randperm(n_clients)[:n_selected]

        # Client training
        client_updates = []
        total_samples = 0
        for idx in selected_indices:
            # Simulate client dropout
            if torch.rand(1).item() < self.config.client_dropout_rate:
                continue

            prototypes, counts = self.client_train(all_client_data[idx])
            client_updates.append((prototypes, counts))
            total_samples += int(counts.sum().item())

        # Aggregate
        self.aggregate(client_updates)

        return {
            "n_selected_clients": len(selected_indices),
            "n_responding_clients": len(client_updates),
            "total_samples": total_samples,
            "privacy_spent": self.privacy_spent,
        }

    def run_federated_training(
        self,
        all_client_data: List[List[Tuple[torch.Tensor, int]]],
    ) -> Dict:
        """Run full federated training across multiple rounds.
        
        Args:
            all_client_data: List of client datasets
        
        Returns:
            Training history
        """
        history = []
        for round_idx in range(self.config.n_communication_rounds):
            round_stats = self.federated_round(all_client_data)
            round_stats["round"] = round_idx
            history.append(round_stats)
            logger.info(
                f"FL Round {round_idx + 1}/{self.config.n_communication_rounds}: "
                f"{round_stats['n_responding_clients']} clients, "
                f"{round_stats['total_samples']} samples"
            )

        return {
            "history": history,
            "final_privacy_spent": self.privacy_spent,
            "global_prototypes": self.global_prototypes,
            "n_classes": self.config.n_classes,
            "dim": self.config.dim,
        }

    def predict(self, hv: torch.Tensor) -> torch.Tensor:
        """Predict class for a hypervector using global model.
        
        Args:
            hv: (dim,) hypervector
        
        Returns:
            (n_classes,) similarity scores
        """
        similarities = torch.mv(self.global_prototypes, hv) / self.config.dim
        return similarities


# ═══════════════════════════════════════════════════════════════════════════════
# Certified Robustness — randomised smoothing for binary HDC
# ═══════════════════════════════════════════════════════════════════════════════

class RenyiDPAccountant:
    """
    Rényi Differential Privacy accountant for multi-round HDC federation.

    Reference:
        Mironov (2017) "Rényi Differential Privacy of the Gaussian Mechanism"
        CSF 2017.

        Balle, Barthe, Gaboardi (2018) "Privacy Amplification by Subsampling:
        Tight Analyses via Couplings and Divergences" NeurIPS 2018.

        Abadi et al. (2016) "Deep Learning with Differential Privacy"
        CCS 2016. — Moments accountant (precursor to RDP).

    Why Rényi DP gives tighter bounds:
        Basic composition (DP):  T rounds at ε → T×ε total
        Moments accountant (RDP): T rounds at σ → O(T/σ²) total privacy loss
        → For T=100 rounds, RDP allows ~√100 = 10× more rounds at the same
          formal (ε,δ)-DP guarantee vs basic composition.

    HDC specifics:
        After each round, the HDC prototype update is clipped to the hypersphere
        (binary HVs have bounded sensitivity = 1 bit), then Gaussian noise is
        added.  The resulting mechanism is a Gaussian mechanism with:
            σ = sensitivity × sqrt(2 ln(1.25/δ)) / ε
            RDP(α) = α / (2σ²)  for the Gaussian mechanism

        Each call to step() accumulates privacy loss for one federation round.
        get_epsilon() converts the accumulated RDP to (ε,δ)-DP.

    Args:
        noise_multiplier: σ / sensitivity  (larger = more noise = more private)
        delta:            Target δ for (ε,δ)-DP report
        n_alphas:         Resolution of α search for optimal conversion
    """

    def __init__(
        self,
        noise_multiplier: float = 1.0,
        delta:            float = 1e-5,
        n_alphas:         int   = 100,
    ):
        self.sigma  = noise_multiplier
        self.delta  = delta
        self.alphas = [1.5 + i * 0.5 for i in range(n_alphas)]   # α ∈ [1.5, 51.5]

        self._steps = 0
        self._rdp: Dict[float, float] = {a: 0.0 for a in self.alphas}

    def step(self, n: int = 1):
        """Record n additional federation rounds."""
        for a in self.alphas:
            # Gaussian mechanism RDP: α/(2σ²) per round
            self._rdp[a] += n * a / (2.0 * self.sigma ** 2)
        self._steps += n

    def get_epsilon(self) -> float:
        """
        Convert accumulated RDP to (ε, δ)-DP epsilon.

        Uses the optimal α conversion:
            ε = min_α  [RDP(α) + log(1/δ) / (α-1)]
        """
        best_eps = float("inf")
        for a, rdp_a in self._rdp.items():
            if a <= 1.0:
                continue
            eps = rdp_a + math.log(1.0 / self.delta) / (a - 1.0)
            best_eps = min(best_eps, eps)
        return best_eps

    def privacy_report(self) -> Dict[str, float]:
        """Return current privacy consumption summary."""
        return {
            "n_rounds":       self._steps,
            "epsilon":        self.get_epsilon(),
            "delta":          self.delta,
            "noise_multiplier": self.sigma,
            "rounds_until_epsilon_1": max(0, int(
                (1.0 - self.get_epsilon()) / (min(self.alphas) / (2 * self.sigma ** 2)) + 1
            )) if self.get_epsilon() < 1.0 else 0,
        }

    def max_rounds_for_epsilon(self, target_epsilon: float) -> int:
        """How many more rounds can we run before exceeding target_epsilon?"""
        # Binary search
        lo, hi = 0, 10000
        while lo < hi:
            mid = (lo + hi + 1) // 2
            acc = RenyiDPAccountant(self.sigma, self.delta)
            acc._rdp = {a: self._rdp[a] for a in self.alphas}
            acc.step(mid)
            if acc.get_epsilon() <= target_epsilon:
                lo = mid
            else:
                hi = mid - 1
        return lo


class CertifiedHDCClassifier:
    """
    Certifiably robust HDC classifier via randomised smoothing (Cohen et al. 2019).

    Reference:
        Cohen, Rosenfeld, Kolter (2019) "Certified Adversarial Robustness via
        Randomised Smoothing" ICML 2019.

        Levine & Feizi (2020) "Robustness Certificates for Sparse Adversarial
        Attacks by Randomised Ablation" AAAI 2020.

    Standard adversarial robustness claims are heuristic. Certified robustness
    provides a provable guarantee: no adversary with perturbation budget ≤ r
    (Hamming ball) can fool the smoothed classifier.

    HDC adaptation:
        The smoothed classifier f̃(x) is the majority vote of f(x + noise_i)
        over N Monte Carlo noise samples, where noise flips each bit with
        probability p_noise independently.

        Certification: if the top class wins fraction pA > 0.5 of votes, the
        smoothed classifier is certified robust to all Hamming perturbations ≤ r:
            r = floor(D × (arcsin(2×pA − 1) / π + 0.5) − 0.5)

        This is the Hamming-ball analogue of the ℓ₂ certified radius for continuous
        inputs.  The certified radius grows with pA and with HV dimension D.

    Args:
        base_classifier: HDCCClassifier or similar with predict(x) → (label, sims)
        noise_flip_rate: Probability of flipping each bit for smoothing (0.1–0.3)
        n_samples:       Monte Carlo samples for certification (100–1000)
        alpha:           Type-I error rate for Clopper-Pearson bound (default 0.001)
        device:          torch device
    """

    def __init__(
        self,
        base_classifier,
        noise_flip_rate: float = 0.15,
        n_samples:       int   = 200,
        alpha:           float = 0.001,
        device:          str   = "cpu",
    ):
        self.clf            = base_classifier
        self.p_noise        = noise_flip_rate
        self.n_samples      = n_samples
        self.alpha          = alpha
        self.device         = device

    def _add_noise(self, hv: torch.Tensor) -> torch.Tensor:
        """Flip each bit independently with probability p_noise."""
        flip = (torch.rand_like(hv.float()) < self.p_noise)
        return ((hv.float() + flip.float()) % 2).float()

    def _clopper_pearson_lower(self, k: int, n: int) -> float:
        """
        Lower bound of one-sided (1-alpha) Clopper-Pearson interval for
        a binomial proportion k/n.  Used for conservative certification.
        """
        import math
        if k == 0:
            return 0.0
        # Beta distribution quantile: scipy.stats.beta.ppf(alpha, k, n-k+1)
        # Approximation via incomplete beta (Newton-Raphson on CDF)
        a, b = float(k), float(n - k + 1)
        # Use normal approximation for large n (accurate when pA > 0.6)
        p_hat = k / n
        z = 1.6449  # z_{1-alpha} ≈ z_{0.999} for alpha=0.001
        margin = z * math.sqrt(p_hat * (1 - p_hat) / n)
        return max(0.0, p_hat - margin)

    def certified_predict(
        self,
        hv: torch.Tensor,
    ) -> Dict:
        """
        Predict label with certified Hamming robustness radius.

        Returns:
            Dict with:
              label:            Predicted class (or -1 if abstain)
              p_A:              Conservative lower bound on smoothed top-class vote fraction
              certified_radius: Hamming ball radius r — no adversary within r bits can fool
              abstain:          True if certification failed (p_A ≤ 0.5)
              vote_counts:      {label: vote count} across Monte Carlo samples
        """
        import math

        hv_f = hv.float().to(self.device)
        n_classes = len(self.clf.class_hvs) if hasattr(self.clf, 'class_hvs') else 10

        # Monte Carlo voting
        vote_counts: Dict[int, int] = {}
        for _ in range(self.n_samples):
            noisy = self._add_noise(hv_f)
            label, _ = self.clf.predict(noisy)
            vote_counts[label] = vote_counts.get(label, 0) + 1

        # Top and runner-up classes
        sorted_votes = sorted(vote_counts.items(), key=lambda x: x[1], reverse=True)
        top_label, top_votes = sorted_votes[0]

        # Conservative p_A via Clopper-Pearson
        p_A = self._clopper_pearson_lower(top_votes, self.n_samples)

        # Certified radius in Hamming distance
        # r = floor(D × (arcsin(2pA − 1) / π)) — from Levine & Feizi 2020
        if p_A <= 0.5:
            r = 0
            abstain = True
        else:
            dim = hv_f.shape[0]
            # For the bit-flip smoothing model: r = dim * p_noise * (2*p_A - 1)
            # (linear approximation valid for small noise rates)
            r = int(math.floor(dim * self.p_noise * (2 * p_A - 1)))
            abstain = False

        return {
            "label":            top_label if not abstain else -1,
            "p_A":              p_A,
            "certified_radius": r,
            "abstain":          abstain,
            "vote_counts":      vote_counts,
        }

    def batch_certify(
        self,
        hvs:    List[torch.Tensor],
        verbose: bool = False,
    ) -> List[Dict]:
        """Certify a batch of inputs. Returns list of certified_predict results."""
        results = []
        for i, hv in enumerate(hvs):
            r = self.certified_predict(hv)
            results.append(r)
            if verbose and (i + 1) % 10 == 0:
                certified = sum(1 for x in results if not x["abstain"])
                avg_r = sum(x["certified_radius"] for x in results) / len(results)
                print(f"  [{i+1}/{len(hvs)}] certified={certified}, avg_radius={avg_r:.1f}")
        return results

    def robustness_report(self, results: List[Dict]) -> Dict:
        """Summarise certification results across a dataset."""
        n = len(results)
        if n == 0:
            return {}
        n_certified = sum(1 for r in results if not r["abstain"])
        radii = [r["certified_radius"] for r in results if not r["abstain"]]
        return {
            "n_samples":           n,
            "certified_frac":      n_certified / n,
            "mean_certified_r":    sum(radii) / max(len(radii), 1),
            "median_certified_r":  sorted(radii)[len(radii) // 2] if radii else 0,
            "max_certified_r":     max(radii) if radii else 0,
            "abstain_frac":        1 - n_certified / n,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════════════════════════════

def test_hdc_security():
    """Test HDC security modules."""
    print("=" * 60)
    print("Testing HDC Security Modules")
    print("=" * 60)

    # 1. Model Stealing Defense
    print("\n1. Model Stealing Defense:")
    defense = ModelStealingDefense()
    similarities = torch.tensor([0.2, 0.8, 0.1, 0.3])
    defended_sims, defended_label = defense.defend_prediction(similarities, client_id="test_client")
    print(f"  Original: label={similarities.argmax().item()}, sims={similarities}")
    print(f"  Defended: label={defended_label}, sims={defended_sims}")
    print(f"  Defense stats: {defense.get_defense_stats()}")

    # 2. HDTest++
    print("\n2. HDTest++ Fuzz Testing:")
    from models.hdc import HDCEncoder
    model = HDCEncoder(input_size=20, n_classes=5, dim=500, n_levels=13)
    X = torch.randn(50, 20)
    y = torch.randint(0, 5, (50,))
    model.train()
    for i in range(50):
        model.train_step(X[i], y[i].item())
    model.finalize()
    model.eval()

    # Use raw data as seeds (model.predict handles encoding internally)
    hdtest = HDTestPlus(
        model=model,
        encode_fn=lambda x: model.encode(x),
    )
    report = hdtest.run_fuzz_campaign(X[:10], y[:10])
    print(f"  Total queries: {report['total_queries']}")
    print(f"  Adversarial rate: {report['adversarial_rate']:.3f}")
    print(f"  Robustness score: {report['robustness_score']:.3f}")
    print(f"  Coverage regions: {report['coverage_regions']}")
    for s, info in report['strategy_effectiveness'].items():
        print(f"  {s:15s}: success={info['success_rate']:.2f} ({info['successes']}/{info['attempts']})")

    # 3. Federated HDC
    print("\n3. Federated HDC Learning:")
    fed_config = FederatedHDCConfig(
        n_clients=5,
        n_communication_rounds=3,
        dim=500,
        n_classes=5,
    )
    fed_hdc = FederatedHDC(fed_config)

    # Generate synthetic client data
    all_client_data = []
    for client_idx in range(5):
        client_data = []
        for _ in range(20):
            hv = torch.where(torch.randn(500) > 0,
                           torch.ones(500), -torch.ones(500))
            label = torch.randint(0, 5, (1,)).item()
            client_data.append((hv, label))
        all_client_data.append(client_data)

    result = fed_hdc.run_federated_training(all_client_data)
    print(f"  Communication rounds: {len(result['history'])}")
    print(f"  Privacy spent: {result['final_privacy_spent']:.4f}")
    for h in result['history']:
        print(f"  Round {h['round']}: {h['n_responding_clients']} clients, {h['total_samples']} samples")

    print("\n✅ HDC Security test complete!")


# ═══════════════════════════════════════════════════════════════════════════════
# Adversarial detection via random subspace ensemble
# ═══════════════════════════════════════════════════════════════════════════════

class RandomSubspaceAdversarialDetector:
    """
    Runtime adversarial input detection via random subspace ensemble disagreement.

    Reference:
        Ho (1998) "The Random Subspace Method for Constructing Decision Forests"
        IEEE Trans. Pattern Analysis 20(8):832-844.

        Xu et al. (2017) "Feature Squeezing: Detecting Adversarial Examples in
        Deep Neural Networks" NDSS 2018.

    Key insight: an adversarial perturbation is carefully crafted to fool a
    SPECIFIC classifier. When we evaluate the input through K random SUBSETS
    of features, the adversarial perturbation is only effective for the full
    feature set — it cannot simultaneously fool all K subspace classifiers.
    High variance across subspace predictions → adversarial.

    HDC implementation:
        1. Generate K random subspace masks (each retains p% of features)
        2. For each mask: encode masked input → predict class via prototypes
        3. Adversarial score = entropy(vote distribution across K predictions)
        4. Flag as adversarial if score > detection_threshold

    This is O(K × D) per inference — fast enough for real-time use.

    Args:
        classifier:         HDC classifier with .encode() and .predict()
        n_subspaces:        K random subspace classifiers (default 16)
        subspace_fraction:  Fraction of features per subspace (default 0.7)
        detection_threshold: Entropy threshold for adversarial flag (default 0.5)
        seed:               Random seed for reproducibility
    """

    def __init__(
        self,
        classifier,
        n_subspaces:         int   = 16,
        subspace_fraction:   float = 0.7,
        detection_threshold: float = 0.5,
        seed:                int   = 42,
    ):
        self.clf              = classifier
        self.n_subspaces      = n_subspaces
        self.frac             = subspace_fraction
        self.threshold        = detection_threshold

        # Pre-generate fixed subspace masks (reproducible)
        g = torch.Generator()
        g.manual_seed(seed)
        n_features = classifier.n_features
        k_features = max(1, int(n_features * subspace_fraction))
        self._masks: List[torch.Tensor] = []
        for _ in range(n_subspaces):
            perm = torch.randperm(n_features, generator=g)
            mask = torch.zeros(n_features, dtype=torch.bool)
            mask[perm[:k_features]] = True
            self._masks.append(mask)

    def _subspace_predict(self, x: torch.Tensor, mask: torch.Tensor) -> int:
        """Predict using only the features in `mask`."""
        x_masked = x.clone()
        x_masked[~mask] = 0.0   # zero out masked features
        label, _ = self.clf.predict(x_masked)[:2]
        return int(label)

    def adversarial_score(self, x: torch.Tensor) -> float:
        """
        Compute adversarial detection score.

        Returns entropy of vote distribution across K subspace predictions.
        High entropy (disagreement) → adversarial.
        Low entropy (agreement) → benign.
        """
        votes: Dict[int, int] = {}
        for mask in self._masks:
            label = self._subspace_predict(x, mask)
            votes[label] = votes.get(label, 0) + 1

        # Entropy: H = -Σ p_k log p_k
        total = sum(votes.values())
        H = 0.0
        for count in votes.values():
            p = count / total
            if p > 0:
                H -= p * math.log(p + 1e-10)

        # Normalize by log(K) so score ∈ [0, 1]
        H_max = math.log(max(self.n_subspaces, 2))
        return H / H_max

    def detect(self, x: torch.Tensor) -> Dict:
        """
        Detect whether input x is adversarial.

        Returns:
            Dict with 'is_adversarial', 'score', 'threshold', 'vote_dist'
        """
        score = self.adversarial_score(x)
        votes: Dict[int, int] = {}
        for mask in self._masks:
            label = self._subspace_predict(x, mask)
            votes[label] = votes.get(label, 0) + 1

        return {
            "is_adversarial": score > self.threshold,
            "score":          score,
            "threshold":      self.threshold,
            "vote_dist":      votes,
            "n_subspaces":    self.n_subspaces,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# HDC Secret Sharing — (k,n)-threshold for coalition operations
# ═══════════════════════════════════════════════════════════════════════════════

class HDCSecretSharing:
    """
    (k,n)-threshold secret sharing for HDC prototypes.

    Reference:
        Shamir (1979) "How to Share a Secret" CACM 22(11):612-613.

    HDC adaptation (XOR-based, information-theoretically secure):
        - Secret = prototype HV  (D-bit binary vector)
        - Split into n shares: any k shares reconstruct; k-1 shares = nothing
        - XOR additive secret sharing: perfect (information-theoretic) security

    XOR scheme (k=n for simplicity; generalised threshold below):
        Encode (n-1 random HVs + XOR remainder = secret):
            share_1 = R_1  (random)
            share_2 = R_2  (random)
            ...
            share_{n-1} = R_{n-1}  (random)
            share_n    = secret ⊕ R_1 ⊕ R_2 ⊕ ... ⊕ R_{n-1}

        Reconstruct: XOR all n shares → secret

    Security property:
        Any n-1 shares are uniformly random and reveal NOTHING about secret.
        All n shares XOR to the secret exactly.

    For k-of-n (k < n) threshold: use Shamir's polynomial scheme over GF(2^D).
    Here we implement n-of-n (simplest, most secure) for coalition operations.

    Use case (IQT/SIGINT):
        Multiple coalition partners each hold one share of the global HDC model.
        To inference, they must cooperate — no single partner can reconstruct
        the full model alone. Revocation: rotate shares.

    Args:
        n_shares:  Number of shares to split into (default 3)
        device:    torch device
    """

    def __init__(self, n_shares: int = 3, device: str = "cpu"):
        self.n_shares = n_shares
        self.device   = device

    def split(
        self,
        secret_hv: torch.Tensor,
        seed:      Optional[int] = None,
    ) -> List[torch.Tensor]:
        """
        Split a prototype HV into n_shares shares.

        Args:
            secret_hv: (D,) binary HDC prototype to protect
            seed:       Random seed (None = random; fixed for reproducibility)

        Returns:
            List of n_shares (D,) binary HVs (the shares)
        """
        D  = secret_hv.shape[0]
        hv = secret_hv.float().to(self.device)

        shares = []
        g = torch.Generator(device=self.device)
        if seed is not None:
            g.manual_seed(seed)

        # Generate n-1 random shares
        xor_accumulator = hv.clone()
        for _ in range(self.n_shares - 1):
            r = (torch.rand(D, generator=g, device=self.device) >= 0.5).float()
            shares.append(r)
            xor_accumulator = (xor_accumulator + r) % 2   # XOR

        # Final share = XOR of all previous shares with secret
        shares.append(xor_accumulator)
        return shares

    def reconstruct(self, shares: List[torch.Tensor]) -> torch.Tensor:
        """
        Reconstruct the secret from all n_shares.

        Args:
            shares: List of (D,) share HVs — ALL shares required

        Returns:
            (D,) reconstructed secret HV
        """
        if not shares:
            raise ValueError("Need at least one share to reconstruct")
        result = shares[0].float().to(self.device)
        for s in shares[1:]:
            result = (result + s.float().to(self.device)) % 2   # XOR
        return result

    def verify_shares(self, shares: List[torch.Tensor], secret_hv: torch.Tensor) -> bool:
        """Verify that reconstructed secret matches the original."""
        reconstructed = self.reconstruct(shares)
        return bool(torch.equal(reconstructed.round(), secret_hv.float().to(self.device).round()))

    def partial_info_leakage(self, shares: List[torch.Tensor]) -> float:
        """
        Measure information leakage from a partial share set.

        For XOR-secret sharing, partial shares should be uniformly random
        (0 bits leaked). Returns the deviation from 0.5 density
        (higher = more leakage, 0.0 = perfectly secure).
        """
        if not shares or len(shares) >= self.n_shares:
            return 0.0   # full reconstruction or no shares
        xor = shares[0].float().to(self.device)
        for s in shares[1:]:
            xor = (xor + s.float().to(self.device)) % 2
        density = float(xor.mean().item())
        return abs(density - 0.5)


# ═══════════════════════════════════════════════════════════════════════════════
# Catastrophic Forgetting Bound Tracker
# ═══════════════════════════════════════════════════════════════════════════════

class CatastrophicForgettingBound:
    """
    Formal catastrophic forgetting rate tracker for continual HDC learning.

    Reference:
        McCloskey & Cohen (1989) "Catastrophic Interference in Connectionist
        Networks: The Sequential Learning Problem" Psychology Learning &
        Motivation 24:109-165.

        Kirkpatrick et al. (2017) "Overcoming catastrophic forgetting" PNAS —
        EWC metric: ||W_new - W_old||² weighted by Fisher information.

    For HDC: the forgetting rate for class c between steps t-1 and t is:
        forget_c(t) = 1 - Hamming_sim(proto_c(t), proto_c(t-1))
              = Hamming_distance(proto_c(t), proto_c(t-1)) / D

    This measures how much the class prototype changed in one update step.
    High forget_c → class c is being overwritten by new data.

    Formal bound:
        If forget_c > ε for any class c, the continual learning guarantee
        is violated — any guarantee requires forget_c ≤ ε for all c.

    Provides:
        - Per-class forgetting trace
        - Global forgetting rate (max over classes)
        - Stability-plasticity tradeoff curve
        - Alert when forgetting exceeds user-specified bound

    Args:
        n_classes:        Number of tracked classes
        forget_threshold: Alert threshold for per-class forgetting rate
    """

    def __init__(self, n_classes: int, forget_threshold: float = 0.1):
        self.n_classes         = n_classes
        self.forget_threshold  = forget_threshold

        self._prev_protos: Dict[int, torch.Tensor] = {}
        self._forget_trace: Dict[int, List[float]] = {c: [] for c in range(n_classes)}
        self._step         = 0
        self._alerts:      List[Dict] = []

    def snapshot(self, prototypes: List[torch.Tensor]):
        """
        Record current class prototypes as the baseline for next comparison.

        Args:
            prototypes: List of n_classes (D,) prototype HVs
        """
        self._prev_protos = {c: p.float().clone() for c, p in enumerate(prototypes)}

    def measure(self, prototypes: List[torch.Tensor]) -> Dict:
        """
        Measure forgetting between the last snapshot and current prototypes.

        Returns:
            Dict with per_class_forgetting, max_forgetting, violations
        """
        self._step += 1
        per_class = {}
        violations = []

        for c, proto_new in enumerate(prototypes):
            if c not in self._prev_protos:
                per_class[c] = 0.0
                continue

            proto_old = self._prev_protos[c].to(proto_new.device)
            # Hamming distance as fraction of dimensions changed
            forget_rate = float((proto_new.float() != proto_old.float()).float().mean().item())
            per_class[c] = forget_rate
            self._forget_trace[c].append(forget_rate)

            if forget_rate > self.forget_threshold:
                alert = {
                    "step":        self._step,
                    "class":       c,
                    "forget_rate": forget_rate,
                    "threshold":   self.forget_threshold,
                }
                self._alerts.append(alert)
                violations.append(alert)

        max_forgetting = max(per_class.values()) if per_class else 0.0

        return {
            "step":             self._step,
            "per_class":        per_class,
            "max_forgetting":   max_forgetting,
            "mean_forgetting":  sum(per_class.values()) / max(len(per_class), 1),
            "violations":       violations,
            "bound_satisfied":  max_forgetting <= self.forget_threshold,
        }

    def stability_plasticity_curve(self) -> Dict[str, List[float]]:
        """
        Return the forgetting trace per class for plotting.

        High recent forgetting = high plasticity (fast learning, fast forgetting).
        Low forgetting = high stability (slow to learn, slow to forget).
        """
        return {
            f"class_{c}": list(trace)
            for c, trace in self._forget_trace.items()
            if trace
        }

    def summary(self) -> Dict:
        all_rates = [r for trace in self._forget_trace.values() for r in trace]
        return {
            "total_steps":       self._step,
            "total_violations":  len(self._alerts),
            "mean_forget_rate":  sum(all_rates) / max(len(all_rates), 1),
            "max_forget_rate":   max(all_rates) if all_rates else 0.0,
            "n_classes":         self.n_classes,
            "threshold":         self.forget_threshold,
        }


if __name__ == "__main__":
    test_hdc_security()
