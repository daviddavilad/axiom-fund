"""Unit tests for residual diagnostics module.

Covers six pure functions in
src/axiom_fund/diagnostics/residual_diagnostics.py:
  - compute_durbin_watson
  - compute_qq_data
  - compute_residual_vs_fitted_data
  - compute_leverage
  - compute_cooks_distance
  - compute_breusch_pagan

Tests are organized by function (one class per function) and cover:
  1. Happy path on synthetic regression data
  2. Mathematical invariants (e.g., leverage sums to k)
  3. Edge cases (degenerate inputs, boundary values)
  4. Error paths (validation failures with matching error messages)
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from axiom_fund.diagnostics.residual_diagnostics import (
    compute_breusch_pagan,
    compute_cooks_distance,
    compute_durbin_watson,
    compute_leverage,
    compute_qq_data,
    compute_residual_vs_fitted_data,
)


# ============================================================================
# Test fixtures
# ============================================================================


def _make_regression_data(
    n_obs: int = 100, n_params: int = 4, seed: int = 42,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Generate (X, residuals, fitted) from a synthetic regression."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n_obs, n_params)).astype(np.float64)
    X[:, 0] = 1.0  # intercept
    # Beta of length n_params: deterministic per seed for reproducibility
    beta_true = rng.uniform(-1.0, 1.0, n_params)
    y = X @ beta_true + rng.normal(0, 1, n_obs)
    beta_hat = np.linalg.solve(X.T @ X, X.T @ y)
    fitted = X @ beta_hat
    residuals = y - fitted
    return X, residuals, fitted


# ============================================================================
# Durbin-Watson
# ============================================================================


class TestDurbinWatson:
    def test_random_residuals_near_two(self) -> None:
        rng = np.random.default_rng(42)
        resid = rng.normal(0, 1, 1000)
        dw = compute_durbin_watson(resid)
        # Sampling noise at N=1000 gives DW in [~1.80, ~2.20]
        assert 1.80 < dw < 2.20

    def test_perfect_positive_autocorrelation_near_zero(self) -> None:
        # e_t = e_{t-1} → diffs all zero → DW → 0
        resid = np.linspace(1.0, 2.0, 100)  # monotonic, highly autocorrelated
        dw = compute_durbin_watson(resid)
        assert dw < 0.1

    def test_perfect_alternating_near_four(self) -> None:
        # Alternating sign → maximum negative autocorrelation → DW → 4
        resid = np.array([1.0, -1.0] * 50)
        dw = compute_durbin_watson(resid)
        assert dw > 3.9

    def test_too_few_observations_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            compute_durbin_watson(np.array([1.0]))

    def test_2d_input_raises(self) -> None:
        with pytest.raises(ValueError, match="1D"):
            compute_durbin_watson(np.zeros((10, 2)))

    def test_zero_residuals_raises(self) -> None:
        with pytest.raises(ValueError, match="sum of squared"):
            compute_durbin_watson(np.zeros(10))


# ============================================================================
# Q-Q data
# ============================================================================


class TestQQData:
    def test_output_shape_and_columns(self) -> None:
        rng = np.random.default_rng(42)
        resid = rng.normal(0, 1, 100)
        df = compute_qq_data(resid)
        assert df.shape == (100, 2)
        assert list(df.columns) == ["theoretical_quantile", "sample_quantile"]

    def test_normal_residuals_align_with_theoretical(self) -> None:
        # If residuals are truly normal, the slope of sample on theoretical
        # quantiles should be near 1.0 and the intercept near 0.0.
        rng = np.random.default_rng(42)
        resid = rng.normal(0, 1, 500)
        df = compute_qq_data(resid)
        slope, intercept = np.polyfit(
            df["theoretical_quantile"], df["sample_quantile"], 1,
        )
        assert 0.9 < slope < 1.1
        assert abs(intercept) < 0.1

    def test_theoretical_quantiles_sorted(self) -> None:
        rng = np.random.default_rng(42)
        resid = rng.normal(0, 1, 50)
        df = compute_qq_data(resid)
        diffs = np.diff(df["theoretical_quantile"])
        assert (diffs >= 0).all()

    def test_empty_input_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            compute_qq_data(np.array([]))

    def test_2d_input_raises(self) -> None:
        with pytest.raises(ValueError, match="1D"):
            compute_qq_data(np.zeros((10, 2)))


# ============================================================================
# Residual vs fitted data
# ============================================================================


class TestResidualVsFittedData:
    def test_output_shape_and_columns(self) -> None:
        resid = np.array([0.1, -0.2, 0.3])
        fitted = np.array([1.0, 2.0, 3.0])
        df = compute_residual_vs_fitted_data(resid, fitted)
        assert df.shape == (3, 2)
        assert list(df.columns) == ["fitted", "residual"]

    def test_preserves_observation_order(self) -> None:
        resid = np.array([0.1, -0.2, 0.3])
        fitted = np.array([1.0, 2.0, 3.0])
        df = compute_residual_vs_fitted_data(resid, fitted)
        assert df["fitted"].iloc[0] == 1.0
        assert df["residual"].iloc[2] == 0.3

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="length"):
            compute_residual_vs_fitted_data(
                np.array([0.1, 0.2]), np.array([1.0, 2.0, 3.0]),
            )

    def test_empty_input_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            compute_residual_vs_fitted_data(np.array([]), np.array([]))


# ============================================================================
# Leverage
# ============================================================================


class TestLeverage:
    def test_sums_to_n_params(self) -> None:
        X, _, _ = _make_regression_data(n_obs=100, n_params=4)
        lev = compute_leverage(X)
        assert np.isclose(lev.sum(), 4.0, rtol=1e-9)

    def test_all_values_in_unit_interval(self) -> None:
        X, _, _ = _make_regression_data(n_obs=100, n_params=4)
        lev = compute_leverage(X)
        assert (lev >= 0).all()
        assert (lev <= 1).all()

    def test_mean_leverage_equals_k_over_n(self) -> None:
        X, _, _ = _make_regression_data(n_obs=200, n_params=5)
        lev = compute_leverage(X)
        assert np.isclose(lev.mean(), 5.0 / 200, rtol=1e-9)

    def test_singular_design_raises(self) -> None:
        # Two columns equal → X^T X singular
        X = np.array([[1.0, 2.0, 2.0], [1.0, 3.0, 3.0], [1.0, 4.0, 4.0]])
        with pytest.raises(ValueError, match="singular"):
            compute_leverage(X)

    def test_underdetermined_raises(self) -> None:
        # More params than obs
        X = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])  # 1 obs, 5 params
        with pytest.raises(ValueError, match="singular"):
            compute_leverage(X)

    def test_1d_input_raises(self) -> None:
        with pytest.raises(ValueError, match="2D"):
            compute_leverage(np.array([1.0, 2.0, 3.0]))


# ============================================================================
# Cook's distance
# ============================================================================


class TestCooksDistance:
    def test_single_zero_residual_yields_zero_cooks_at_that_index(self) -> None:
        # When observation i has e_i = 0, D_i = 0 by construction:
        # D_i ∝ e_i^2 = 0.
        X, _, _ = _make_regression_data(n_obs=50, n_params=3)
        rng = np.random.default_rng(123)
        residuals = rng.normal(0, 1, 50)
        residuals[7] = 0.0  # one observation has zero residual
        cooks = compute_cooks_distance(residuals, X, n_params=3)
        assert cooks[7] == 0.0
        # Other observations should have non-zero Cook's
        assert (cooks[np.arange(50) != 7] > 0).any()

    def test_shape_matches_residuals(self) -> None:
        X, resid, _ = _make_regression_data(n_obs=100, n_params=4)
        cooks = compute_cooks_distance(resid, X, n_params=4)
        assert cooks.shape == (100,)

    def test_residuals_x_length_mismatch_raises(self) -> None:
        X, _, _ = _make_regression_data(n_obs=50, n_params=3)
        bad_resid = np.zeros(10)
        with pytest.raises(ValueError, match="length"):
            compute_cooks_distance(bad_resid, X, n_params=3)

    def test_n_params_must_be_positive(self) -> None:
        X, resid, _ = _make_regression_data(n_obs=50, n_params=3)
        with pytest.raises(ValueError, match="positive"):
            compute_cooks_distance(resid, X, n_params=0)

    def test_zero_mse_raises(self) -> None:
        X, _, _ = _make_regression_data(n_obs=50, n_params=3)
        with pytest.raises(ValueError, match="MSE"):
            compute_cooks_distance(np.zeros(50), X, n_params=3)


# ============================================================================
# Breusch-Pagan
# ============================================================================


class TestBreuschPagan:
    def test_homoskedastic_high_pvalue(self) -> None:
        rng = np.random.default_rng(42)
        n_obs = 500
        X = rng.normal(0, 1, (n_obs, 4))
        X[:, 0] = 1.0
        resid = rng.normal(0, 1, n_obs)
        result = compute_breusch_pagan(resid, X)
        assert result["lm_pvalue"] > 0.05

    def test_heteroskedastic_low_pvalue(self) -> None:
        rng = np.random.default_rng(42)
        n_obs = 500
        X = rng.normal(0, 1, (n_obs, 4))
        X[:, 0] = 1.0
        # Residual variance proportional to X[:, 1]^2 → strong heteroskedasticity
        resid = rng.normal(0, 1, n_obs) * np.abs(X[:, 1])
        result = compute_breusch_pagan(resid, X)
        # Loose-but-meaningful threshold: test rejects at conventional 0.05
        assert result["lm_pvalue"] < 0.05

    def test_output_dict_keys(self) -> None:
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (50, 3))
        X[:, 0] = 1.0
        resid = rng.normal(0, 1, 50)
        result = compute_breusch_pagan(resid, X)
        assert set(result.keys()) == {
            "lm_statistic", "lm_pvalue", "f_statistic", "f_pvalue",
        }

    def test_no_intercept_raises(self) -> None:
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (50, 3))  # no constant column
        resid = rng.normal(0, 1, 50)
        with pytest.raises(ValueError, match="intercept"):
            compute_breusch_pagan(resid, X)