"""
test_error_masking.py
=====================
Tests for HDC error masking schemes (hdc/error_masking.py).

Implements the three masking schemes from:
  "Brain-Inspired HDC for Ultra-Efficient Edge AI" (NSF purl/10392362)

Validates:
  1. apply_zero_masking — corrupted positions → 0
  2. apply_sign_bit_masking — corrupted positions → sign-flipped
  3. apply_word_masking — entire word zeroed when any bit in word is corrupted
  4. ErrorMasker module — threshold gating, update_error_rate, get_stats
  5. All schemes preserve shape and don't modify original tensor
"""

from __future__ import annotations

import sys
import os
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from hdc.error_masking import (
    apply_zero_masking,
    apply_sign_bit_masking,
    apply_word_masking,
    ErrorMasker,
    ErrorMaskingConfig,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def bipolar_hv():
    """Small bipolar hypervector for deterministic tests."""
    return torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0])


@pytest.fixture
def random_hv():
    torch.manual_seed(42)
    return torch.randn(256)


@pytest.fixture
def error_mask_4():
    """Explicit error mask: positions 1 and 5 corrupted."""
    mask = torch.zeros(8, dtype=torch.bool)
    mask[1] = True
    mask[5] = True
    return mask


# ── apply_zero_masking ────────────────────────────────────────────────────────

class TestZeroMasking:
    def test_corrupted_positions_become_zero(self, bipolar_hv, error_mask_4):
        masked = apply_zero_masking(bipolar_hv, error_positions=error_mask_4)
        assert masked[1].item() == 0.0
        assert masked[5].item() == 0.0

    def test_uncorrupted_positions_unchanged(self, bipolar_hv, error_mask_4):
        masked = apply_zero_masking(bipolar_hv, error_positions=error_mask_4)
        for i in range(len(bipolar_hv)):
            if not error_mask_4[i]:
                assert masked[i].item() == bipolar_hv[i].item()

    def test_original_not_modified(self, bipolar_hv, error_mask_4):
        original = bipolar_hv.clone()
        apply_zero_masking(bipolar_hv, error_positions=error_mask_4)
        assert torch.equal(bipolar_hv, original)

    def test_shape_preserved(self, random_hv):
        masked = apply_zero_masking(random_hv, error_rate=0.1)
        assert masked.shape == random_hv.shape

    def test_zero_error_rate_noop(self, random_hv):
        masked = apply_zero_masking(random_hv, error_rate=0.0)
        assert torch.equal(masked, random_hv)

    def test_high_error_rate_many_zeros(self, random_hv):
        masked = apply_zero_masking(random_hv, error_rate=0.5)
        n_zeros = (masked == 0.0).sum().item()
        # With rate=0.5, expect ~50% zeros; allow generous bounds
        assert n_zeros >= len(random_hv) * 0.1

    def test_with_error_positions_ignores_rate(self, bipolar_hv, error_mask_4):
        """When error_positions is given, error_rate is irrelevant."""
        masked_explicit = apply_zero_masking(bipolar_hv, error_positions=error_mask_4)
        masked_rate = apply_zero_masking(bipolar_hv, error_positions=error_mask_4,
                                          error_rate=0.9)
        # Same because error_positions takes precedence
        assert torch.equal(masked_explicit, masked_rate)

    def test_no_args_noop(self, bipolar_hv):
        masked = apply_zero_masking(bipolar_hv)
        assert torch.equal(masked, bipolar_hv)


# ── apply_sign_bit_masking ────────────────────────────────────────────────────

class TestSignBitMasking:
    def test_corrupted_positions_sign_flipped(self, bipolar_hv, error_mask_4):
        masked = apply_sign_bit_masking(bipolar_hv, error_positions=error_mask_4)
        # Sign-bit masking negates the corrupted values
        assert masked[1].item() == -bipolar_hv[1].item()
        assert masked[5].item() == -bipolar_hv[5].item()

    def test_uncorrupted_positions_unchanged(self, bipolar_hv, error_mask_4):
        masked = apply_sign_bit_masking(bipolar_hv, error_positions=error_mask_4)
        for i in range(len(bipolar_hv)):
            if not error_mask_4[i]:
                assert masked[i].item() == bipolar_hv[i].item()

    def test_original_not_modified(self, bipolar_hv, error_mask_4):
        original = bipolar_hv.clone()
        apply_sign_bit_masking(bipolar_hv, error_positions=error_mask_4)
        assert torch.equal(bipolar_hv, original)

    def test_shape_preserved(self, random_hv):
        masked = apply_sign_bit_masking(random_hv, error_rate=0.1)
        assert masked.shape == random_hv.shape

    def test_zero_error_rate_noop(self, random_hv):
        masked = apply_sign_bit_masking(random_hv, error_rate=0.0)
        assert torch.equal(masked, random_hv)

    def test_no_args_noop(self, bipolar_hv):
        masked = apply_sign_bit_masking(bipolar_hv)
        assert torch.equal(masked, bipolar_hv)


# ── apply_word_masking ────────────────────────────────────────────────────────

class TestWordMasking:
    def test_word_with_error_zeroed(self):
        """If any bit in a word is corrupted, the whole word → 0."""
        hv = torch.ones(8)
        # Error at position 5 → word [4,5,6,7] should be zeroed
        mask = torch.zeros(8, dtype=torch.bool)
        mask[5] = True
        masked = apply_word_masking(hv, error_positions=mask, word_size=4)
        assert (masked[4:8] == 0.0).all()
        # Word [0,1,2,3] should be unchanged
        assert (masked[0:4] == 1.0).all()

    def test_clean_word_unchanged(self):
        hv = torch.ones(8)
        mask = torch.zeros(8, dtype=torch.bool)
        mask[0] = True  # Error only in first word
        masked = apply_word_masking(hv, error_positions=mask, word_size=4)
        # First word → 0
        assert (masked[0:4] == 0.0).all()
        # Second word → unchanged
        assert (masked[4:8] == 1.0).all()

    def test_shape_preserved(self, random_hv):
        masked = apply_word_masking(random_hv, error_rate=0.1, word_size=16)
        assert masked.shape == random_hv.shape

    def test_zero_error_rate_noop(self, random_hv):
        masked = apply_word_masking(random_hv, error_rate=0.0, word_size=16)
        assert torch.equal(masked, random_hv)

    def test_original_not_modified(self, bipolar_hv, error_mask_4):
        original = bipolar_hv.clone()
        apply_word_masking(bipolar_hv, error_positions=error_mask_4, word_size=4)
        assert torch.equal(bipolar_hv, original)

    def test_non_power_of_two_length(self):
        """Word masking handles HV lengths not divisible by word_size."""
        hv = torch.ones(10)
        mask = torch.zeros(10, dtype=torch.bool)
        mask[8] = True  # In partial last word [8,9]
        masked = apply_word_masking(hv, error_positions=mask, word_size=4)
        assert (masked[8:10] == 0.0).all()
        assert (masked[0:8] == 1.0).all()

    def test_word_size_1_is_element_masking(self):
        """Word size 1 reduces to element-level masking."""
        hv = torch.ones(8)
        mask = torch.zeros(8, dtype=torch.bool)
        mask[3] = True
        masked_word = apply_word_masking(hv, error_positions=mask, word_size=1)
        masked_zero = apply_zero_masking(hv, error_positions=mask)
        assert torch.equal(masked_word, masked_zero)


# ── ErrorMasker module ────────────────────────────────────────────────────────

class TestErrorMasker:
    def test_passthrough_below_threshold(self, random_hv):
        """Below threshold, masker is a no-op."""
        masker = ErrorMasker(256, ErrorMaskingConfig(enabled=True,
                                                      error_threshold=1e-4))
        masker.update_error_rate(1e-6)  # Below threshold
        output = masker(random_hv)
        assert torch.equal(output, random_hv)

    def test_masking_applied_above_threshold(self, random_hv):
        """Above threshold, masker modifies the HV."""
        masker = ErrorMasker(256, ErrorMaskingConfig(enabled=True,
                                                      masking_scheme="zero",
                                                      error_threshold=1e-5))
        masker.update_error_rate(1e-2)  # Above threshold
        output = masker(random_hv)
        # With zero masking at high error rate, some positions become 0
        assert not torch.equal(output, random_hv)

    def test_disabled_always_passthrough(self, random_hv):
        masker = ErrorMasker(256, ErrorMaskingConfig(enabled=False))
        masker.update_error_rate(1.0)  # Very high — should still be noop
        output = masker(random_hv)
        assert torch.equal(output, random_hv)

    def test_shape_preserved_all_schemes(self, random_hv):
        for scheme in ("zero", "sign_bit", "word"):
            masker = ErrorMasker(256, ErrorMaskingConfig(masking_scheme=scheme,
                                                          error_threshold=0.0))
            masker.update_error_rate(0.1)
            output = masker(random_hv)
            assert output.shape == random_hv.shape

    def test_update_error_rate_stored(self):
        masker = ErrorMasker(128)
        masker.update_error_rate(0.05)
        assert abs(masker.error_rate.item() - 0.05) < 1e-6

    def test_total_samples_increments(self):
        masker = ErrorMasker(128)
        for i in range(5):
            masker.update_error_rate(float(i) * 0.01)
        assert masker.total_samples.item() == 5

    def test_masking_count_tracks_above_threshold(self):
        masker = ErrorMasker(128, ErrorMaskingConfig(error_threshold=1e-3))
        masker.update_error_rate(1e-5)  # below
        masker.update_error_rate(1e-5)  # below
        masker.update_error_rate(1e-2)  # above
        masker.update_error_rate(1e-2)  # above
        assert masker.masking_count == 2

    def test_get_stats_keys(self):
        masker = ErrorMasker(128)
        masker.update_error_rate(0.01)
        stats = masker.get_stats()
        assert "error_rate" in stats
        assert "masking_applied_pct" in stats
        assert "total_samples" in stats

    def test_with_explicit_error_positions(self, bipolar_hv):
        masker = ErrorMasker(8, ErrorMaskingConfig(masking_scheme="zero",
                                                    error_threshold=0.0))
        masker.update_error_rate(1.0)  # Force masking active
        mask = torch.zeros(8, dtype=torch.bool)
        mask[2] = True
        mask[6] = True
        output = masker(bipolar_hv, error_positions=mask)
        assert output[2].item() == 0.0
        assert output[6].item() == 0.0

    def test_word_masker_uses_word_size_from_config(self):
        word_size = 8
        masker = ErrorMasker(32, ErrorMaskingConfig(masking_scheme="word",
                                                     word_size=word_size,
                                                     error_threshold=0.0))
        masker.update_error_rate(0.1)
        hv = torch.ones(32)
        mask = torch.zeros(32, dtype=torch.bool)
        mask[10] = True  # In word [8..15]
        output = masker(hv, error_positions=mask)
        # Entire word [8..15] → 0
        assert (output[8:16] == 0.0).all()
        # Other words unchanged
        assert (output[0:8] == 1.0).all()
        assert (output[16:] == 1.0).all()


# ── Scheme comparison ─────────────────────────────────────────────────────────

class TestSchemeComparison:
    """Zero vs sign_bit vs word masking differ in how they treat errors."""

    def test_zero_sets_to_zero(self, bipolar_hv, error_mask_4):
        masked = apply_zero_masking(bipolar_hv, error_positions=error_mask_4)
        assert (masked[error_mask_4] == 0.0).all()

    def test_sign_bit_flips_sign(self, bipolar_hv, error_mask_4):
        masked = apply_sign_bit_masking(bipolar_hv, error_positions=error_mask_4)
        expected = -bipolar_hv[error_mask_4]
        assert torch.allclose(masked[error_mask_4], expected)

    def test_word_zeros_entire_word(self, error_mask_4):
        hv = torch.ones(8) * 2.0
        masked = apply_word_masking(hv, error_positions=error_mask_4, word_size=4)
        # error at positions 1 (word 0) and 5 (word 1) → both words → 0
        assert (masked == 0.0).all()

    def test_schemes_differ_from_each_other(self, random_hv):
        mask = torch.zeros(256, dtype=torch.bool)
        mask[::10] = True  # Every 10th position

        m_zero = apply_zero_masking(random_hv, error_positions=mask)
        m_sign = apply_sign_bit_masking(random_hv, error_positions=mask)
        m_word = apply_word_masking(random_hv, error_positions=mask, word_size=8)

        assert not torch.equal(m_zero, m_sign)
        assert not torch.equal(m_zero, m_word)
