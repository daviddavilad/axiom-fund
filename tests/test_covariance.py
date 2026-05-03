"""Tests for the covariance estimation module.

Pure-function module — all tests are unit tests with synthetic data.
Tests cover input validation, schema, math correctness, and edge cases.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from axiom_fund.portfolio.covariance import (
    CovarianceEstimate,
    estimate_covariance,
)

# ============================================================================
# Helpers
# ============================================================================


def _make_iid_returns(
    n_obs: int, n_assets: int, daily_vol: float = 0.015, seed: int = 42
) -> pd.DataFrame:
    """Build N independent normal-return series with given daily vol."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0, daily_vol, size=(n_obs, n_assets))
    return pd.DataFrame(rets, columns=[f"P{i}" for i in range(n_assets)])


def _block_correlation(
    n: int, n_sectors: int, rho_within: float, rho_across: float
) -> npt.NDArray[np.float64]:
    """Block-diagonal correlation matrix for sector-structured tests."""
    sector_size = n // n_sectors
    corr = np.full((n, n), rho_across)
    for s in range(n_sectors):
        start = s * sector_size
        end = start + sector_size
        corr[start:end, start:end] = rho_within
    np.fill_diagonal(corr, 1.0)
    return corr


def _make_block_returns(
    n_obs: int, n_assets: int, n_sectors: int, daily_vol: float = 0.015,
    rho_within: float = 0.6, rho_across: float = 0.1, seed: int = 42,
) -> pd.DataFrame:
    """Build returns from a block-correlation covariance."""
    corr = _block_correlation(n_assets, n_sectors, rho_within, rho_across)
    cov = corr * (daily_vol**2)
    rng = np.random.default_rng(seed)
    rets = rng.multivariate_normal(np.zeros(n_assets), cov, size=n_obs)
    return pd.DataFrame(rets, columns=[f"P{i}" for i in range(n_assets)])


# ============================================================================
# Input validation
# ============================================================================


class TestInputValidation:
    def test_non_dataframe_raises(self) -> None:
        rets = np.zeros((10, 3))
        with pytest.raises(ValueError, match="DataFrame"):
            estimate_covariance(rets)  # type: ignore[arg-type]

    def test_too_few_assets_raises(self) -> None:
        rets = _make_iid_returns(100, 1)
        with pytest.raises(ValueError, match="at least 2 assets"):
            estimate_covariance(rets)

    def test_too_few_observations_raises(self) -> None:
        rets = _make_iid_returns(1, 5)
        with pytest.raises(ValueError, match="at least 2 valid observations"):
            estimate_covariance(rets)

    def test_all_nan_raises(self) -> None:
        rets = pd.DataFrame(np.full((10, 3), np.nan), columns=["A", "B", "C"])
        with pytest.raises(ValueError, match="at least 2 valid observations"):
            estimate_covariance(rets)


# ============================================================================
# Output structure
# ============================================================================


class TestOutputStructure:
    def test_returns_covariance_estimate(self) -> None:
        rets = _make_iid_returns(252, 10)
        result = estimate_covariance(rets)
        assert isinstance(result, CovarianceEstimate)

    def test_matrix_is_square_and_dataframe(self) -> None:
        rets = _make_iid_returns(252, 10)
        result = estimate_covariance(rets)
        assert isinstance(result.matrix, pd.DataFrame)
        assert result.matrix.shape == (10, 10)

    def test_matrix_index_matches_columns(self) -> None:
        rets = _make_iid_returns(252, 10)
        result = estimate_covariance(rets)
        # Both axes labeled with PERMNO
        assert list(result.matrix.index) == list(result.matrix.columns)
        assert list(result.matrix.index) == list(rets.columns)

    def test_n_obs_and_n_assets_correct(self) -> None:
        rets = _make_iid_returns(252, 10)
        result = estimate_covariance(rets)
        assert result.n_obs == 252
        assert result.n_assets == 10

    def test_shrinkage_in_unit_interval(self) -> None:
        rets = _make_iid_returns(252, 10)
        result = estimate_covariance(rets)
        assert 0.0 <= result.shrinkage <= 1.0


# ============================================================================
# Mathematical properties
# ============================================================================


class TestMatrixProperties:
    def test_matrix_is_symmetric(self) -> None:
        rets = _make_block_returns(252, 30, 3)
        result = estimate_covariance(rets)
        assert np.allclose(result.matrix.values, result.matrix.values.T)

    def test_matrix_is_positive_definite(self) -> None:
        """All eigenvalues > 0 — required for portfolio optimization."""
        rets = _make_block_returns(252, 30, 3)
        result = estimate_covariance(rets)
        eigvals = np.linalg.eigvalsh(result.matrix.values)
        assert eigvals.min() > 0

    def test_matrix_positive_definite_when_T_less_than_N(self) -> None:
        """Even with rank-deficient sample, output should be PD via shrinkage."""
        # T=20 < N=30 → sample is rank-deficient, but Ledoit-Wolf must rescue
        rets = _make_block_returns(20, 30, 3)
        result = estimate_covariance(rets)
        eigvals = np.linalg.eigvalsh(result.matrix.values)
        assert eigvals.min() > 0


# ============================================================================
# Shrinkage behavior
# ============================================================================


class TestShrinkageBehavior:
    def test_more_data_means_less_shrinkage(self) -> None:
        """Shrinkage should decrease as T grows (sample becomes more reliable)."""
        rets_few = _make_block_returns(60, 30, 3, seed=42)
        rets_many = _make_block_returns(2000, 30, 3, seed=42)
        s_few = estimate_covariance(rets_few).shrinkage
        s_many = estimate_covariance(rets_many).shrinkage
        assert s_few > s_many

    def test_low_T_high_N_yields_substantial_shrinkage(self) -> None:
        """In a noisy regime where sample is unreliable, shrinkage must be > 0."""
        rets = _make_block_returns(40, 30, 3, seed=42)
        result = estimate_covariance(rets)
        assert result.shrinkage > 0.10

    def test_target_matched_truth_yields_high_shrinkage(self) -> None:
        """When the target is approximately correct (uniform corr matches truth),
        optimal shrinkage approaches 1.0."""
        # Constant correlation 0.3 across all 20 assets — matches target
        n = 20
        cov_true = np.full((n, n), 0.3 * 0.0001)
        np.fill_diagonal(cov_true, 0.0001)
        rng = np.random.default_rng(42)
        rets = pd.DataFrame(
            rng.multivariate_normal(np.zeros(n), cov_true, size=100),
            columns=[f"P{i}" for i in range(n)],
        )
        result = estimate_covariance(rets)
        assert result.shrinkage > 0.5


# ============================================================================
# Annualization
# ============================================================================


class TestAnnualization:
    def test_annualize_true_multiplies_by_252(self) -> None:
        rets = _make_iid_returns(252, 5)
        ann = estimate_covariance(rets, annualize=True).matrix
        daily = estimate_covariance(rets, annualize=False).matrix
        ratio = ann.values / daily.values
        assert np.allclose(ratio, 252.0)

    def test_annualize_default_is_true(self) -> None:
        rets = _make_iid_returns(252, 5)
        explicit = estimate_covariance(rets, annualize=True).matrix
        default = estimate_covariance(rets).matrix
        assert np.allclose(explicit.values, default.values)


# ============================================================================
# NaN handling
# ============================================================================


class TestNaNHandling:
    def test_rows_with_any_nan_are_dropped(self) -> None:
        rets = _make_iid_returns(100, 5)
        # NaN out 10 rows
        rets.iloc[10:20, 0] = np.nan
        result = estimate_covariance(rets)
        assert result.n_obs == 90  # 100 - 10 NaN rows

    def test_partial_nan_columns_handled(self) -> None:
        """When different columns have different NaN locations, drop any-row-with-NaN."""
        rets = _make_iid_returns(100, 5)
        rets.iloc[5, 0] = np.nan
        rets.iloc[10, 1] = np.nan
        rets.iloc[15, 2] = np.nan
        result = estimate_covariance(rets)
        assert result.n_obs == 97  # 3 distinct rows dropped
