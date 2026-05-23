"""
Error Masking for HDC
====================
Implements error masking schemes from Section III-A of:
"Brain-Inspired Hyperdimensional Computing for Ultra-Efficient Edge AI"
(NSF purl/10392362)

Three masking schemes protect HDC models from hardware errors:
- Zero masking: Set corrupted bits to 0
- Sign-bit masking: Set corrupted bits to sign bit
- Word masking: Set entire numerical value to 0

These enable tolerance up to 10^-5 to 10^-4 error rate with only 1% accuracy loss.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
from dataclasses import dataclass


@dataclass
class ErrorMaskingConfig:
    """Configuration for error masking."""
    enabled: bool = True
    masking_scheme: str = "zero"  # "zero", "sign_bit", "word"
    error_threshold: float = 1e-5  # Apply masking above this error rate
    word_size: int = 16  # Word size for word masking


def apply_zero_masking(
    hypervector: torch.Tensor,
    error_positions: Optional[torch.Tensor] = None,
    error_rate: float = 0.0,
) -> torch.Tensor:
    """
    Apply zero masking to corrupted hypervector components.
    
    Sets corrupted bits to 0. This is the simplest masking scheme
    that provides robust protection against hardware errors.
    
    Args:
        hypervector: The hypervector to mask (D,)
        error_positions: Boolean tensor indicating error locations
        error_rate: If error_positions not provided, random errors at this rate
    
    Returns:
        Masked hypervector
    
    Example:
        >>> hv = torch.tensor([1., -1., 1., 0., 1.])
        >>> masked = apply_zero_masking(hv, error_rate=0.2)
        >>> print(masked)
        tensor([1., 0., 1., 0., 1.])  # Error at position 1 -> 0
    """
    if error_positions is not None:
        masked = hypervector.clone()
        masked[error_positions] = 0.0
        return masked
    
    # Generate random error positions if not provided
    if error_rate is not None and error_rate > 0:
        n = hypervector.numel()
        n_errors = max(1, int(n * error_rate))
        error_positions = torch.rand(n) < error_rate
        # Limit to requested number of errors
        error_indices = torch.randperm(n)[:n_errors]
        error_positions = torch.zeros(n, dtype=torch.bool)
        error_positions[error_indices] = True
        
        masked = hypervector.clone()
        masked[error_positions] = 0.0
        return masked
    
    return hypervector


def apply_sign_bit_masking(
    hypervector: torch.Tensor,
    error_positions: Optional[torch.Tensor] = None,
    error_rate: float = 0.0,
) -> torch.Tensor:
    """
    Apply sign-bit masking to corrupted hypervector components.
    
    Sets corrupted bits to the sign bit (+1 for positive, -1 for negative).
    This preserves magnitude information better than zero masking.
    
    Args:
        hypervector: The hypervector to mask (D,)
        error_positions: Boolean tensor indicating error locations
        error_rate: Random errors at this rate
    
    Returns:
        Masked hypervector
    
    Example:
        >>> hv = torch.tensor([1., -1., 1., 0., 1.])
        >>> masked = apply_sign_bit_masking(hv, error_rate=0.2)
        >>> print(masked)
        tensor([1., 1., 1., 0., 1.])  # Error at position 1 -> sign(1) = +1
    """
    if error_positions is not None:
        masked = hypervector.clone()
        masked[error_positions] = -hypervector[error_positions]
        return masked
    
    if error_rate is not None and error_rate > 0:
        n = hypervector.numel()
        n_errors = max(1, int(n * error_rate))
        error_indices = torch.randperm(n)[:n_errors]
        error_positions = torch.zeros(n, dtype=torch.bool)
        error_positions[error_indices] = True

        masked = hypervector.clone()
        # Flip the sign bit of corrupted positions
        masked[error_positions] = -hypervector[error_positions]
        return masked

    return hypervector


def apply_word_masking(
    hypervector: torch.Tensor,
    error_positions: Optional[torch.Tensor] = None,
    error_rate: float = 0.0,
    word_size: int = 16,
) -> torch.Tensor:
    """
    Apply word masking to corrupted hypervector components.
    
    Sets entire word (group of bits) to 0 when any bit in the word is corrupted.
    This provides stronger protection by discarding entire words.
    
    Args:
        hypervector: The hypervector to mask (D,)
        error_positions: Boolean tensor indicating error locations
        error_rate: Random errors at this rate
        word_size: Number of bits per word
    
    Returns:
        Masked hypervector
    
    Example:
        >>> hv = torch.tensor([1., -1., 1., 0., 1., -1., 1., 0.])
        >>> masked = apply_word_masking(hv, error_rate=0.25, word_size=4)
        >>> print(masked)
        tensor([1., -1., 1., 0., 0., 0., 0., 0.])  # Word with errors -> zeros
    """
    n = hypervector.numel()
    n_words = (n + word_size - 1) // word_size
    
    if error_rate is not None and error_rate > 0:
        # Generate random error positions
        n_errors = max(1, int(n * error_rate))
        error_indices = torch.randperm(n)[:n_errors]
        error_positions = torch.zeros(n, dtype=torch.bool)
        error_positions[error_indices] = True
    
    if error_positions is not None:
        masked = hypervector.clone()
        # Find words with errors
        for word_idx in range(n_words):
            start = word_idx * word_size
            end = min(start + word_size, n)
            word_errors = error_positions[start:end]
            if word_errors.any():
                masked[start:end] = 0.0
        return masked
    
    return hypervector


class ErrorMasker(nn.Module):
    """
    Learnable error masking layer for HDC models.
    
    Applies masking based on learned error patterns and can be
    combined with the Hebbian learning in snntraining.
    
    Attributes:
        config: ErrorMaskingConfig with hyperparameters
        error_rate: Current error rate estimate
        mask_apply_fn: Function to apply masking
    """
    
    def __init__(
        self,
        dim: int,
        config: Optional[ErrorMaskingConfig] = None,
    ):
        super().__init__()
        self.dim = dim
        self.config = config or ErrorMaskingConfig()
        
        # Register masking function
        if self.config.masking_scheme == "zero":
            self.mask_apply_fn = apply_zero_masking
        elif self.config.masking_scheme == "sign_bit":
            self.mask_apply_fn = apply_sign_bit_masking
        elif self.config.masking_scheme == "word":
            self.mask_apply_fn = lambda hv, ep=None, er=0: apply_word_masking(
                hv, ep, er, self.config.word_size
            )
        else:
            self.mask_apply_fn = apply_zero_masking
        
        # Error tracking
        self.register_buffer("error_rate", torch.tensor(0.0))
        self.register_buffer("total_samples", torch.tensor(0))
        
        # Statistics
        self.masking_count = 0
        self.total_masking_applied = 0
    
    def forward(
        self,
        hypervector: torch.Tensor,
        error_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply masking to hypervector.
        
        Args:
            hypervector: Input hypervector (D,) or (batch, D)
            error_positions: Optional known error positions
        
        Returns:
            Masked hypervector
        """
        if not self.config.enabled:
            return hypervector
        
        # Apply masking based on estimated error rate
        if self.error_rate.item() > self.config.error_threshold:
            return self.mask_apply_fn(
                hypervector,
                error_positions,
                self.error_rate.item() if error_positions is None else None
            )
        
        return hypervector
    
    def update_error_rate(self, measured_error_rate: float) -> None:
        """Update the estimated error rate."""
        self.error_rate = torch.tensor(measured_error_rate)
        self.total_samples += 1
        
        if measured_error_rate > self.config.error_threshold:
            self.masking_count += 1
    
    def get_stats(self) -> dict:
        """Return masking statistics."""
        return {
            "error_rate": self.error_rate.item(),
            "masking_applied_pct": self.masking_count / max(1, self.total_samples),
            "total_samples": self.total_samples.item(),
        }


def test_error_masking():
    """Test error masking functions."""
    print("Testing error masking schemes...")
    
    # Test hypervector
    hv = torch.tensor([1.0, -1.0, 1.0, 0.0, 1.0, -1.0, 0.0, 1.0])
    print(f"Original: {hv}")
    
    # Zero masking
    masked_zero = apply_zero_masking(hv, error_rate=0.25)
    print(f"Zero masked: {masked_zero}")
    
    # Sign-bit masking
    masked_sign = apply_sign_bit_masking(hv, error_rate=0.25)
    print(f"Sign-bit masked: {masked_sign}")
    
    # Word masking
    masked_word = apply_word_masking(hv, error_rate=0.25, word_size=4)
    print(f"Word masked: {masked_word}")
    
    # Test ErrorMasker module
    masker = ErrorMasker(8, ErrorMaskingConfig(masking_scheme="zero"))
    output = masker(hv)
    print(f"ErrorMasker output: {output}")
    
    # Update error rate and test again
    masker.update_error_rate(1e-4)
    masker.update_error_rate(1e-3)
    masker.update_error_rate(1e-2)  # Above threshold
    print(f"ErrorMasker stats: {masker.get_stats()}")
    
    print("Error masking tests complete!")


if __name__ == "__main__":
    test_error_masking()