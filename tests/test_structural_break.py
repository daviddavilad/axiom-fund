"""Unit tests for the Quandt-Andrews structural-break test.

Covers two functions in src/axiom_fund/diagnostics/structural_break.py:
  - compute_hansen_pvalue
  - compute_quandt_andrews

Tests cover (a) statistical invariants on synthetic data with known
break structure, (b) Hansen p-value mechanics including reference
points, and (c) error paths.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from axiom_fund.diagnostics.structural_break import (
    compute_hansen_pvalue,
    compute_quandt_andrews,
)


# ============================================================================
# Test fixtures
# ============================================================================


def _make_no_break_series(
    n: int = 200, sigma: float = 1.0, seed: int = 42,
) -> NDArray[np.float64]:
    """A homogeneous-mean series (no structural break)."""
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, sigma, n).astype(np.float64)


def _make_break_series(
    n_pre: int = 100, n_post: int = 100,
    mean_pre: float = 0.0, mean_post: float = 1.0,
    sigma: float = 1.0, seed: int = 42,
) -> NDArray[np.float64]:
    """A series with a single break at index n_pre."""
    rng = np.random.default_rng(seed)
    pre = rng.normal(mean_pre, sigma, n_pre)
    post = rng.normal(mean_post, sigma, n_post)
    return np.concatenate([pre, post]).astype(np.float64)


# ============================================================================
# Hansen p-value approximation
# ============================================================================


class TestHansenPValue:
    def test_returns_probability(self) -> None:
        p = compute_hansen_pvalue(5.0, m=1, trimming=0.15)
        assert 0.0 <= p <= 1.0

    def test_low_sup_f_gives_high_pvalue(self) -> None:
        # A sup_F of ~1-2 is consistent with no break
        p = compute_hansen_pvalue(1.5, m=1, trimming=0.15)
        assert p > 0.5

    def test_high_sup_f_gives_low_pvalue(self) -> None:
        # A sup_F of 20+ should give a tiny p
        p = compute_hansen_pvalue(20.0, m=1, trimming=0.15)
        assert p < 0.01

    def test_andrews_reference_point_m1_pi015(self) -> None:
        # Andrews (1993) 5% critical value at m=1, π=0.15 is ~8.85
        # Hansen p-value at that sup_F should be close to 0.05
        p = compute_hansen_pvalue(8.85, m=1, trimming=0.15)
        # Allow tolerance: Hansen claims median error 0.0006, max 0.003
        assert 0.03 < p < 0.07

    def test_different_m_gives_different_p(self) -> None:
        # Same sup_F, different m → different p (m affects critical region)
        p_m1 = compute_hansen_pvalue(10.0, m=1, trimming=0.15)
        p_m3 = compute_hansen_pvalue(10.0, m=3, trimming=0.15)
        assert p_m1 != p_m3

    def test_transformed_zero_gives_p_one(self) -> None:
        # At sup_F where θ₀ + θ₁ * sup_F < 0, p should be 1.0
        # For m=1, π=0.15: θ₀=-0.99, θ₁=1.02 → transformed < 0 when sup_F < 0.97
        p = compute_hansen_pvalue(0.5, m=1, trimming=0.15)
        assert p == 1.0

    def test_negative_sup_f_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            compute_hansen_pvalue(-1.0, m=1)

    def test_invalid_m_raises(self) -> None:
        with pytest.raises(ValueError, match="m must be one of"):
            compute_hansen_pvalue(5.0, m=21)

    def test_invalid_trimming_raises(self) -> None:
        with pytest.raises(ValueError, match="trimming must be one of"):
            compute_hansen_pvalue(5.0, m=1, trimming=0.10)


# ============================================================================
# Quandt-Andrews sup-F test
# ============================================================================


class TestQuandtAndrews:
    def test_detects_known_break(self) -> None:
        # Real break at index 100 with mean shift 0 → 1
        y = _make_break_series(n_pre=100, n_post=100, mean_post=1.0, seed=42)
        X = np.ones((len(y), 1))
        sup_f, break_idx, p_val = compute_quandt_andrews(y, X)
        # Break detected with high confidence
        assert p_val < 0.01
        # Detected break location should be close to the true 100
        assert abs(break_idx - 100) <= 5

    def test_no_false_positive_on_no_break(self) -> None:
        y = _make_no_break_series(n=200, seed=42)
        X = np.ones((len(y), 1))
        _, _, p_val = compute_quandt_andrews(y, X)
        # Should not reject at conventional 5%
        assert p_val > 0.05

    def test_break_index_within_trimmed_window(self) -> None:
        y = _make_break_series(seed=42)
        X = np.ones((len(y), 1))
        _, break_idx, _ = compute_quandt_andrews(y, X, trimming=0.15)
        T = len(y)
        # Break must lie in middle 70%
        assert 0.15 * T <= break_idx <= 0.85 * T

    def test_larger_break_gives_smaller_pvalue(self) -> None:
        # Stronger break signal → lower p
        y_small = _make_break_series(mean_post=0.5, seed=42)
        y_large = _make_break_series(mean_post=2.0, seed=42)
        X = np.ones((len(y_small), 1))
        _, _, p_small = compute_quandt_andrews(y_small, X)
        _, _, p_large = compute_quandt_andrews(y_large, X)
        assert p_large < p_small

    def test_determinism(self) -> None:
        # Identical inputs → identical outputs
        y = _make_break_series(seed=42)
        X = np.ones((len(y), 1))
        result1 = compute_quandt_andrews(y, X)
        result2 = compute_quandt_andrews(y, X)
        assert result1 == result2

    def test_t_too_small_raises(self) -> None:
        y = np.array([1.0, 2.0, 3.0])  # T=3, too small
        X = np.ones((3, 1))
        with pytest.raises(ValueError, match="too small"):
            compute_quandt_andrews(y, X)

    def test_shape_mismatch_raises(self) -> None:
        y = np.zeros(100)
        X = np.ones((50, 1))
        with pytest.raises(ValueError, match="X.shape"):
            compute_quandt_andrews(y, X)

    def test_1d_x_raises(self) -> None:
        y = np.zeros(100)
        X = np.ones(100)  # 1D, should be 2D
        with pytest.raises(ValueError, match="X must be 2D"):
            compute_quandt_andrews(y, X)

    def test_invalid_trimming_raises(self) -> None:
        y = np.zeros(100)
        X = np.ones((100, 1))
        with pytest.raises(ValueError, match="trimming must be in"):
            compute_quandt_andrews(y, X, trimming=0.6)