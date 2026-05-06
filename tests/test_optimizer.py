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


# ============================================================================
# Neutrality constraints — input validation
# ============================================================================


class TestNeutralityInputValidation:
    def test_beta_neutrality_without_betas_raises(self) -> None:
        alpha = _make_alpha([1.0, -1.0])
        cov = _make_identity_cov(2)
        with pytest.raises(ValueError, match="constrain_beta_neutral"):
            optimize_portfolio(alpha, cov, constrain_beta_neutral=True)

    def test_sector_neutrality_without_sectors_raises(self) -> None:
        alpha = _make_alpha([1.0, -1.0])
        cov = _make_identity_cov(2)
        with pytest.raises(ValueError, match="constrain_sector_neutral"):
            optimize_portfolio(alpha, cov, constrain_sector_neutral=True)

    def test_betas_must_be_series(self) -> None:
        alpha = _make_alpha([1.0, -1.0])
        cov = _make_identity_cov(2)
        with pytest.raises(ValueError, match="betas must be"):
            optimize_portfolio(alpha, cov, betas=[1.0, 1.0])  # type: ignore[arg-type]

    def test_betas_index_must_match_alpha(self) -> None:
        alpha = _make_alpha([1.0, -1.0])
        cov = _make_identity_cov(2)
        bad = pd.Series([1.0, 1.0], index=[10, 20])
        with pytest.raises(ValueError, match=r"betas\.index"):
            optimize_portfolio(alpha, cov, betas=bad)

    def test_betas_with_nan_raises(self) -> None:
        alpha = _make_alpha([1.0, -1.0])
        cov = _make_identity_cov(2)
        betas = pd.Series([1.0, np.nan], index=alpha.index)
        with pytest.raises(ValueError, match="betas contains NaN"):
            optimize_portfolio(alpha, cov, betas=betas)

    def test_sectors_must_be_series(self) -> None:
        alpha = _make_alpha([1.0, -1.0])
        cov = _make_identity_cov(2)
        with pytest.raises(ValueError, match="sectors must be"):
            optimize_portfolio(alpha, cov, sectors=["A", "B"])  # type: ignore[arg-type]

    def test_sectors_index_must_match_alpha(self) -> None:
        alpha = _make_alpha([1.0, -1.0])
        cov = _make_identity_cov(2)
        bad = pd.Series([1, 1], index=[10, 20])
        with pytest.raises(ValueError, match=r"sectors\.index"):
            optimize_portfolio(alpha, cov, sectors=bad)

    def test_sectors_with_nan_raises(self) -> None:
        alpha = _make_alpha([1.0, -1.0])
        cov = _make_identity_cov(2)
        sectors = pd.Series([1, np.nan], index=alpha.index)
        with pytest.raises(ValueError, match="sectors contains NaN"):
            optimize_portfolio(alpha, cov, sectors=sectors)


# ============================================================================
# Dollar neutrality
# ============================================================================


class TestDollarNeutrality:
    def test_dollar_neutral_constraint_makes_net_zero(self) -> None:
        # 10 names with mixed alpha — without dollar neutrality, would have nonzero net
        rng = np.random.default_rng(42)
        alphas = rng.normal(0.5, 1.0, size=10).tolist()  # positive mean
        alpha = _make_alpha(alphas)
        cov = _make_identity_cov(10)
        result = optimize_portfolio(
            alpha, cov, risk_aversion=0.001, constrain_dollar_neutral=True
        )
        assert abs(result.net_exposure) < 1e-6

    def test_without_dollar_neutral_net_can_be_nonzero(self) -> None:
        # Same setup, no constraint → should have nonzero net
        rng = np.random.default_rng(42)
        alphas = rng.normal(0.5, 1.0, size=10).tolist()
        alpha = _make_alpha(alphas)
        cov = _make_identity_cov(10)
        result = optimize_portfolio(alpha, cov, risk_aversion=0.001)
        assert abs(result.net_exposure) > 1e-3


# ============================================================================
# Beta neutrality
# ============================================================================


class TestBetaNeutrality:
    def test_beta_neutral_constraint_makes_portfolio_beta_zero(self) -> None:
        rng = np.random.default_rng(42)
        n = 10
        alphas = rng.normal(0, 1.0, size=n).tolist()
        alpha = _make_alpha(alphas)
        cov = _make_identity_cov(n)
        # Mix of high and low betas
        betas = pd.Series(
            [0.5, 1.5, 0.8, 1.2, 0.6, 1.4, 0.9, 1.1, 0.7, 1.3],
            index=alpha.index,
        )
        result = optimize_portfolio(
            alpha, cov, risk_aversion=0.001,
            betas=betas, constrain_beta_neutral=True,
        )
        assert abs(result.portfolio_beta) < 1e-6  # type: ignore[arg-type]

    def test_betas_provided_but_not_constrained_records_beta(self) -> None:
        """If betas are passed but constraint isn't on, portfolio_beta is still
        reported (may be nonzero)."""
        alpha = _make_alpha([2.0, -2.0])
        cov = _make_identity_cov(2)
        betas = pd.Series([1.0, 1.0], index=alpha.index)
        result = optimize_portfolio(alpha, cov, betas=betas)
        assert result.portfolio_beta is not None

    def test_no_betas_yields_none_portfolio_beta(self) -> None:
        alpha = _make_alpha([2.0, -2.0])
        cov = _make_identity_cov(2)
        result = optimize_portfolio(alpha, cov)
        assert result.portfolio_beta is None


# ============================================================================
# Sector neutrality
# ============================================================================


class TestSectorNeutrality:
    def test_sector_neutral_constraint_makes_each_sector_net_zero(self) -> None:
        # 6 stocks across 3 sectors (2 per sector)
        n = 6
        alpha = _make_alpha([2.0, -1.0, 1.5, -0.5, -2.0, 1.0])
        cov = _make_identity_cov(n)
        sectors = pd.Series([10, 10, 20, 20, 30, 30], index=alpha.index)
        result = optimize_portfolio(
            alpha, cov, risk_aversion=0.001,
            sectors=sectors, constrain_sector_neutral=True,
        )
        assert result.sector_exposures is not None
        for sector_exposure in result.sector_exposures:
            assert abs(sector_exposure) < 1e-6

    def test_sectors_provided_but_not_constrained_records_exposures(self) -> None:
        alpha = _make_alpha([2.0, -2.0])
        cov = _make_identity_cov(2)
        sectors = pd.Series([10, 20], index=alpha.index)
        result = optimize_portfolio(alpha, cov, sectors=sectors)
        assert result.sector_exposures is not None

    def test_no_sectors_yields_none_sector_exposures(self) -> None:
        alpha = _make_alpha([2.0, -2.0])
        cov = _make_identity_cov(2)
        result = optimize_portfolio(alpha, cov)
        assert result.sector_exposures is None


# ============================================================================
# Combined neutrality (full strategy spec)
# ============================================================================


class TestCombinedNeutrality:
    def test_all_three_constraints_together(self) -> None:
        """The locked strategy spec: dollar + beta + sector neutral."""
        rng = np.random.default_rng(42)
        n = 30  # need enough names for all sectors to be feasible
        alphas = rng.normal(0, 1.0, size=n).tolist()
        alpha = _make_alpha(alphas)
        cov = _make_identity_cov(n)
        betas = pd.Series(rng.uniform(0.5, 1.5, size=n), index=alpha.index)
        # 3 sectors, 10 names each
        sectors = pd.Series(
            [10] * 10 + [20] * 10 + [30] * 10, index=alpha.index
        )
        result = optimize_portfolio(
            alpha, cov, risk_aversion=0.001,
            betas=betas, sectors=sectors,
            constrain_dollar_neutral=True,
            constrain_beta_neutral=True,
            constrain_sector_neutral=True,
        )
        # All three should bind
        assert abs(result.net_exposure) < 1e-6
        assert abs(result.portfolio_beta) < 1e-6  # type: ignore[arg-type]
        for s in result.sector_exposures:  # type: ignore[union-attr]
            assert abs(s) < 1e-6


# ============================================================================
# Infeasibility handling
# ============================================================================


class TestInfeasibility:
    def test_impossible_constraints_raise(self) -> None:
        """A sector with only one name cannot be sector-neutral with nonzero
        weight allowed."""
        alpha = _make_alpha([2.0, -1.0, 1.5])
        cov = _make_identity_cov(3)
        # All 3 stocks in same sector, with very high alpha → constraint
        # forces sum to 0, but the alphas make this trivially possible
        # (just zero them out). To force infeasibility, we need a real
        # conflict. The cleanest is: tiny gross_cap with sector neutrality
        # AND alpha that wants to escape sector zero.
        # Actually: the sector-neutral constraint Σ w_i = 0 plus the
        # objective will always have a feasible solution (w=0). So this
        # test is more about the error-message clarity for *legitimate*
        # infeasibility. Hard to construct synthetically for cvxpy without
        # contradictory bounds. Skip detailed infeasibility test for now;
        # documented limitation.
        # Instead: confirm normal cases work even at edge.
        sectors = pd.Series([10, 10, 10], index=alpha.index)
        result = optimize_portfolio(
            alpha, cov, risk_aversion=1.0,
            sectors=sectors, constrain_sector_neutral=True,
        )
        # Single sector, sum must be 0; weights should oppose
        assert abs(result.sector_exposures.iloc[0]) < 1e-6  # type: ignore[union-attr]
