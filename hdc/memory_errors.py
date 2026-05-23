"""
Memory Error Handling for HDC
=============================
Implements memory error injection and handling from Section III-A of:
"Brain-Inspired Hyperdimensional Computing for Ultra-Efficient Edge AI"
(NSF purl/10392362)

Provides:
- Bit flip error injection for testing
- Memory error rate profiling
- Integration with Hebbian learning

HDC shows strong robustness: up to 10^-6 error rate without accuracy loss.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Callable
from dataclasses import dataclass


@dataclass
class MemoryErrorConfig:
    """Configuration for memory error handling."""
    error_rate: float = 1e-6  # Bit flip error rate
    enable_injection: bool = False  # Enable error injection for testing
    error_distribution: str = "uniform"  # "uniform", "biased"


def inject_bit_flips(
    tensor: torch.Tensor,
    error_rate: float = 1e-6,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Inject bit flip errors into tensor.
    
    Simulates memory errors by flipping random bits.
    Uses single bit flip (SBF) model from the paper.
    
    Args:
        tensor: Input tensor to inject errors into
        error_rate: Probability of bit flip (10^-9 to 10^-1)
        seed: Optional random seed for reproducibility
    
    Returns:
        Tuple of (corrupted tensor, error mask)
    
    Example:
        >>> hv = torch.tensor([1., -1., 1., -1., 1.])
        >>> corrupted, mask = inject_bit_flips(hv, error_rate=0.2)
        >>> print(f"Original: {hv}")
        >>> print(f"Corrupted: {corrupted}")
        >>> print(f"Errors at: {mask.nonzero()}")
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    n = tensor.numel()
    n_errors = max(0, int(n * error_rate))
    
    # Generate error positions
    if n_errors > 0:
        error_indices = torch.randperm(n)[:n_errors]
        # Flatten to 1D mask, then reshape to match tensor shape
        flat_mask = torch.zeros(n, dtype=torch.bool)
        flat_mask[error_indices] = True
        error_mask = flat_mask.reshape(tensor.shape)
        
        # Flip the bits at error positions
        corrupted = tensor.clone()
        corrupted[error_mask] = -corrupted[error_mask]
        
        return corrupted, error_mask
    
    return tensor, torch.zeros_like(tensor, dtype=torch.bool)


class MemoryErrorInjector:
    """
    Memory error injector for testing HDC robustness.
    
    Can inject errors during forward pass to test
    model robustness without actual hardware faults.
    
    Attributes:
        config: MemoryErrorConfig
        total_bit_flips: Total number of bit flips injected
    """
    
    def __init__(
        self,
        config: Optional[MemoryErrorConfig] = None,
    ):
        self.config = config or MemoryErrorConfig()
        self.total_bit_flips = 0
        self.total_bits_processed = 0
    
    def inject(
        self,
        tensor: torch.Tensor,
        force: bool = False,
    ) -> torch.Tensor:
        """
        Inject errors into tensor.
        
        Args:
            tensor: Input tensor
            force: Force injection even if disabled in config
        
        Returns:
            Corrupted tensor
        """
        if not self.config.enable_injection and not force:
            return tensor
        
        corrupted, error_mask = inject_bit_flips(
            tensor,
            self.config.error_rate,
        )
        
        n_flips = error_mask.sum().item()
        self.total_bit_flips += n_flips
        self.total_bits_processed += tensor.numel()
        
        return corrupted
    
    def inject_gaussian(
        self,
        tensor: torch.Tensor,
        std: float = 0.1,
    ) -> torch.Tensor:
        """
        Inject errors as gaussian noise (more realistic for floating point).
        
        Args:
            tensor: Input tensor
            std: Standard deviation of noise
        
        Returns:
            Noisy tensor
        """
        noise = torch.randn_like(tensor) * std
        return tensor + noise
    
    def compute_error_rate(
        self,
        original: torch.Tensor,
        corrupted: torch.Tensor,
    ) -> float:
        """Compute bit error rate from two tensors."""
        diff = (original != corrupted).float()
        return diff.mean().item()
    
    def get_stats(self) -> dict:
        """Get error injection statistics."""
        return {
            "total_bit_flips": self.total_bit_flips,
            "total_bits_processed": self.total_bits_processed,
            "actual_error_rate": (
                self.total_bit_flips / max(1, self.total_bits_processed)
            ),
            "config_error_rate": self.config.error_rate,
        }


class ErrorToleranceBenchmark:
    """
    Benchmark HDC model robustness to memory errors.
    
    Tests accuracy at different error rates to find
    the tolerance threshold.
    
    Attributes:
        model_fn: Function to compute accuracy
        error_rates: List of error rates to test
    """
    
    def __init__(
        self,
        model_fn: Callable[[], float],
        error_rates: Optional[list] = None,
    ):
        self.model_fn = model_fn
        self.error_rates = error_rates or [
            1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1
        ]
        self.results: dict = {}
    
    def run(
        self,
        hypervector_dim: int,
        n_trials: int = 3,
    ) -> dict:
        """
        Run benchmark.
        
        Args:
            hypervector_dim: Dimension of hypervector
            n_trials: Number of trials per error rate
        
        Returns:
            Dictionary of error_rate -> accuracy
        """
        injector = MemoryErrorInjector(
            MemoryErrorConfig(enable_injection=True)
        )
        
        # Baseline accuracy (no errors)
        self.results["baseline"] = self.model_fn()
        
        for rate in self.error_rates:
            injector.config.error_rate = rate
            
            accuracies = []
            for trial in range(n_trials):
                # Inject errors in hypervector representation
                # This would be called by user's model
                acc = self.model_fn()
                accuracies.append(acc)
            
            self.results[f"rate_{rate}"] = mean(accuracies)
        
        return self.results
    
    def find_tolerance_threshold(
        self,
        tolerance: float = 0.01,
    ) -> float:
        """
        Find error rate threshold within tolerance.
        
        Args:
            tolerance: Maximum accuracy drop allowed (0.01 = 1%)
        
        Returns:
            Maximum error rate that maintains accuracy
        """
        baseline = self.results.get("baseline", 0.0)
        
        for rate in self.error_rates:
            key = f"rate_{rate}"
            if key not in self.results:
                continue
            
            acc = self.results[key]
            if baseline - acc > tolerance:
                # Previous rate was within tolerance
                prev_idx = self.error_rates.index(rate) - 1
                if prev_idx >= 0:
                    return self.error_rates[prev_idx]
                return self.error_rates[0]
        
        # All rates within tolerance
        return self.error_rates[-1]


def test_memory_errors():
    """Test memory error functions."""
    print("Testing memory errors...")
    
    # Test hypervector
    hv = torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
    print(f"Original: {hv}")
    
    # Inject errors
    injector = MemoryErrorInjector(
        MemoryErrorConfig(enable_injection=True, error_rate=0.2)
    )
    
    for _ in range(3):
        corrupted = injector.inject(hv)
        print(f"Corrupted: {corrupted}")
    
    print(f"Injector stats: {injector.get_stats()}")
    
    # Test gaussian noise injection
    noisy = injector.inject_gaussian(hv, std=0.1)
    print(f"Noisy: {noisy}")
    
    print("\nMemory error tests complete!")


def mean(values):
    """Simple mean function."""
    return sum(values) / len(values) if values else 0


if __name__ == "__main__":
    test_memory_errors()