"""Tests for the MVO optimizer.

Pure-function tests with synthetic data designed to verify specific
optimization behaviors: math correctness, constraint enforcement,
parameter sensitivity, error handling.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from axiom_fund.portfolio.optimizer import (
    OptimizationResult,
    optimize_portfolio,
)

# ============================================================================
# Helpers
# ============================================================================


def _make_alpha(values: list[float]) -> pd.Series:
    """Build a pandas Series of alphas indexed by permno = 1, 2, ..."""
    permnos = list(range(1, len(values) + 1))
    return pd.Series(values, index=permnos, name="alpha")


def _make_identity_cov(
    n: int, annual_var: float = 0.1008
) -> pd.DataFrame:
    """Build identity covariance scaled by annual_var."""
    permnos = list(range(1, n + 1))
    return pd.DataFrame(
        np.eye(n) * annual_var,
        index=permnos,
        columns=permnos,
    )


def _make_random_psd_cov(n: int, seed: int = 42) -> pd.DataFrame:
    """Build a random positive-semi-definite covariance matrix."""
    rng = np.random.default_rng(seed)
    a: npt.NDArray[np.float64] = rng.normal(0, 1, size=(n, n))
    cov = (a @ a.T) / n + np.eye(n) * 0.01  # ensure positive-definite
    permnos = list(range(1, n + 1))
    return pd.DataFrame(cov, index=permnos, columns=permnos)


# ============================================================================
# Input validation
# ============================================================================


class TestInputValidation:
    def test_alpha_must_be_series(self) -> None:
        alpha = np.array([1.0, 2.0])
        cov = _make_identity_cov(2)
        with pytest.raises(ValueError, match="Series"):
            optimize_portfolio(alpha, cov)  # type: ignore[arg-type]

    def test_covariance_must_be_dataframe(self) -> None:
        alpha = _make_alpha([1.0, 2.0])
        cov = np.eye(2)
        with pytest.raises(ValueError, match="DataFrame"):
            optimize_portfolio(alpha, cov)  # type: ignore[arg-type]

    def test_covariance_must_be_square(self) -> None:
        alpha = _make_alpha([1.0, 2.0])
        cov = pd.DataFrame(np.zeros((2, 3)))
        with pytest.raises(ValueError, match="square"):
            optimize_portfolio(alpha, cov)

    def test_alpha_and_cov_indices_must_match(self) -> None:
        alpha = pd.Series([1.0, 2.0], index=[1, 2])
        cov = pd.DataFrame(
            np.eye(2),
            index=[1, 3],  # different from alpha
            columns=[1, 3],
        )
        with pytest.raises(ValueError, match=r"alpha\.index"):
            optimize_portfolio(alpha, cov)

    def test_cov_index_must_match_columns(self) -> None:
        alpha = _make_alpha([1.0, 2.0])
        cov = pd.DataFrame(
            np.eye(2),
            index=[1, 2],
            columns=[3, 4],  # different from index
        )
        with pytest.raises(ValueError, match=r"covariance\.index must equal"):
            optimize_portfolio(alpha, cov)

    def test_alpha_with_nan_raises(self) -> None:
        alpha = pd.Series([1.0, np.nan], index=[1, 2])
        cov = _make_identity_cov(2)
        with pytest.raises(ValueError, match="alpha contains NaN"):
            optimize_portfolio(alpha, cov)

    def test_cov_with_nan_raises(self) -> None:
        alpha = _make_alpha([1.0, 2.0])
        cov = pd.DataFrame(
            [[1.0, np.nan], [np.nan, 1.0]],
            index=[1, 2], columns=[1, 2],
        )
        with pytest.raises(ValueError, match="covariance contains NaN"):
            optimize_portfolio(alpha, cov)

    def test_negative_risk_aversion_raises(self) -> None:
        alpha = _make_alpha([1.0, 2.0])
        cov = _make_identity_cov(2)
        with pytest.raises(ValueError, match="risk_aversion"):
            optimize_portfolio(alpha, cov, risk_aversion=-1.0)

    def test_invalid_position_cap_raises(self) -> None:
        alpha = _make_alpha([1.0, 2.0])
        cov = _make_identity_cov(2)
        with pytest.raises(ValueError, match="position_cap"):
            optimize_portfolio(alpha, cov, position_cap=0.0)
        with pytest.raises(ValueError, match="position_cap"):
            optimize_portfolio(alpha, cov, position_cap=1.5)

    def test_invalid_gross_cap_raises(self) -> None:
        alpha = _make_alpha([1.0, 2.0])
        cov = _make_identity_cov(2)
        with pytest.raises(ValueError, match="gross_cap"):
            optimize_portfolio(alpha, cov, gross_cap=0.0)
        with pytest.raises(ValueError, match="gross_cap"):
            optimize_portfolio(alpha, cov, gross_cap=-1.0)

    def test_too_few_assets_raises(self) -> None:
        alpha = pd.Series([1.0], index=[1])
        cov = pd.DataFrame([[0.1]], index=[1], columns=[1])
        with pytest.raises(ValueError, match="at least 2 assets"):
            optimize_portfolio(alpha, cov)


# ============================================================================
# Output structure
# ============================================================================


class TestOutputStructure:
    def test_returns_optimization_result(self) -> None:
        alpha = _make_alpha([1.0, -1.0])
        cov = _make_identity_cov(2)
        result = optimize_portfolio(alpha, cov)
        assert isinstance(result, OptimizationResult)

    def test_weights_indexed_by_permno(self) -> None:
        alpha = _make_alpha([1.0, -1.0, 0.0])
        cov = _make_identity_cov(3)
        result = optimize_portfolio(alpha, cov)
        assert list(result.weights.index) == [1, 2, 3]

    def test_solver_status_is_optimal(self) -> None:
        alpha = _make_alpha([1.0, -1.0, 0.0])
        cov = _make_identity_cov(3)
        result = optimize_portfolio(alpha, cov)
        assert result.solver_status in ("optimal", "optimal_inaccurate")

    def test_long_short_counts(self) -> None:
        # 5 positive alphas, 3 negative, 2 zero
        alpha = _make_alpha([2.0, 1.5, 1.0, 0.5, 0.1, 0.0, 0.0, -0.5, -1.0, -1.5])
        cov = _make_identity_cov(10)
        result = optimize_portfolio(alpha, cov, risk_aversion=0.001)
        # With low risk aversion, all non-zero alphas get loaded
        assert result.long_count == 5
        assert result.short_count == 3


# ============================================================================
# Sign correctness
# ============================================================================


class TestSignCorrectness:
    def test_positive_alpha_yields_positive_weight(self) -> None:
        alpha = _make_alpha([1.0, -1.0])
        cov = _make_identity_cov(2)
        result = optimize_portfolio(alpha, cov)
        assert result.weights[1] > 0
        assert result.weights[2] < 0

    def test_alpha_ranking_preserved_in_weights(self) -> None:
        """Higher alpha should yield higher (more positive) weight."""
        alpha = _make_alpha([-2.0, -1.0, 0.0, 1.0, 2.0])
        cov = _make_identity_cov(5)
        result = optimize_portfolio(alpha, cov, risk_aversion=1.0)
        # Weights should be monotonically increasing with alpha
        weights = result.weights.values
        for i in range(len(weights) - 1):
            assert weights[i] <= weights[i + 1] + 1e-8


# ============================================================================
# Constraint enforcement
# ============================================================================


class TestConstraintEnforcement:
    def test_position_cap_enforced(self) -> None:
        # Strong alpha + low risk aversion → should hit caps
        alpha = _make_alpha([10.0, -10.0])
        cov = _make_identity_cov(2)
        result = optimize_portfolio(alpha, cov, position_cap=0.015)
        assert result.weights.abs().max() <= 0.015 + 1e-8

    def test_position_cap_lower_bound(self) -> None:
        """Negative weights must respect -position_cap floor."""
        alpha = _make_alpha([10.0, -10.0])
        cov = _make_identity_cov(2)
        result = optimize_portfolio(alpha, cov, position_cap=0.015)
        assert result.weights.min() >= -0.015 - 1e-8

    def test_gross_leverage_cap_enforced(self) -> None:
        # 200 names with strong alpha → would exceed gross 1.5 if uncapped
        n = 200
        rng = np.random.default_rng(42)
        alphas = rng.normal(0, 1, size=n).tolist()
        alpha = _make_alpha(alphas)
        cov = _make_identity_cov(n)
        result = optimize_portfolio(alpha, cov, gross_cap=1.5)
        assert result.gross_leverage <= 1.5 + 1e-6

    def test_higher_risk_aversion_yields_smaller_portfolio(self) -> None:
        alpha = _make_alpha([2.0, -2.0])
        cov = _make_identity_cov(2)
        # Use very high λ where caps don't bind
        low_lam = optimize_portfolio(alpha, cov, risk_aversion=10000.0)
        very_high_lam = optimize_portfolio(alpha, cov, risk_aversion=100000.0)
        assert very_high_lam.gross_leverage < low_lam.gross_leverage


# ============================================================================
# Math correctness
# ============================================================================


class TestMathCorrectness:
    def test_zero_risk_aversion_maximizes_alpha(self) -> None:
        """With risk_aversion=0, optimizer should maximize α'w subject to caps."""
        alpha = _make_alpha([1.0, 2.0, 3.0])
        cov = _make_identity_cov(3)
        result = optimize_portfolio(alpha, cov, risk_aversion=0.0)
        # All positive alphas → all positive weights at the cap
        assert (result.weights > 0).all()
        assert result.weights.max() == pytest.approx(0.015, abs=1e-6)

    def test_zero_alpha_yields_zero_portfolio(self) -> None:
        """With α=0 everywhere, optimal is no portfolio (no edge)."""
        alpha = _make_alpha([0.0, 0.0, 0.0])
        cov = _make_identity_cov(3)
        result = optimize_portfolio(alpha, cov, risk_aversion=1.0)
        assert result.weights.abs().max() < 1e-6

    def test_expected_alpha_is_dot_product(self) -> None:
        alpha = _make_alpha([1.0, -1.0, 2.0])
        cov = _make_identity_cov(3)
        result = optimize_portfolio(alpha, cov)
        # Verify expected_alpha = α'w
        expected = float((alpha.to_numpy() * result.weights.to_numpy()).sum())
        assert result.expected_alpha == pytest.approx(expected, abs=1e-8)

    def test_expected_variance_is_quadratic_form(self) -> None:
        alpha = _make_alpha([1.0, -1.0, 2.0])
        cov = _make_random_psd_cov(3)
        result = optimize_portfolio(alpha, cov)
        # Verify expected_variance = w' Σ w
        w = result.weights.values
        expected = float(w @ cov.values @ w)
        assert result.expected_variance == pytest.approx(expected, abs=1e-8)


# ============================================================================
# Integration — works on realistic universe size
# ============================================================================


class TestRealisticScale:
    def test_optimizer_works_on_500_assets(self) -> None:
        """Solver should handle 500-asset portfolio without errors."""
        n = 500
        rng = np.random.default_rng(42)
        alphas = rng.normal(0, 1, size=n).tolist()
        alpha = _make_alpha(alphas)
        cov = _make_random_psd_cov(n)
        result = optimize_portfolio(alpha, cov)
        assert result.solver_status in ("optimal", "optimal_inaccurate")
        assert result.gross_leverage <= 1.5 + 1e-6
        assert result.weights.abs().max() <= 0.015 + 1e-6
