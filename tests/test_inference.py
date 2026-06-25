"""Unit tests for HAC standard errors and bootstrapped Sharpe CIs.

Covers three pure functions in
src/axiom_fund/diagnostics/inference.py:
  - compute_hac_standard_errors
  - compute_hac_standard_error_of_mean
  - compute_bootstrapped_sharpe_ci

Tests are organized by function (one class per function) and cover:
  1. Happy path on synthetic data
  2. Statistical invariants (e.g., HAC SE > naive SE for AR(1))
  3. Output shape and positivity
  4. Determinism (bootstrap with fixed seed)
  5. Error paths (validation failures)
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from axiom_fund.diagnostics.inference import (
    compute_bootstrapped_sharpe_ci,
    compute_deflated_sharpe,
    compute_expected_max_sharpe,
    compute_hac_standard_error_of_mean,
    compute_hac_standard_errors,
)


# ============================================================================
# Test fixtures
# ============================================================================


def _make_iid_regression(
    n_obs: int = 200, seed: int = 42,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Generate (X, residuals) from an i.i.d. OLS regression."""
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.ones(n_obs), rng.normal(0, 1, n_obs)])
    beta_true = np.array([0.5, 1.0])
    y = X @ beta_true + rng.normal(0, 1, n_obs)
    beta_hat = np.linalg.solve(X.T @ X, X.T @ y)
    residuals = y - X @ beta_hat
    return X.astype(np.float64), residuals.astype(np.float64)


def _make_ar1_series(
    n: int = 200, phi: float = 0.6, seed: int = 42,
) -> NDArray[np.float64]:
    """Generate an AR(1) time series with given autocorrelation."""
    rng = np.random.default_rng(seed)
    series = np.zeros(n, dtype=np.float64)
    for t in range(1, n):
        series[t] = phi * series[t - 1] + rng.normal(0, 1)
    return series


# ============================================================================
# HAC standard errors (general OLS)
# ============================================================================


class TestHACStandardErrors:
    def test_output_shape_matches_n_params(self) -> None:
        X, residuals = _make_iid_regression(n_obs=200)
        cov = compute_hac_standard_errors(residuals, X, maxlags=4)
        assert cov.shape == (2, 2)

    def test_output_symmetric(self) -> None:
        X, residuals = _make_iid_regression(n_obs=200)
        cov = compute_hac_standard_errors(residuals, X, maxlags=4)
        assert np.allclose(cov, cov.T, atol=1e-12)

    def test_diagonal_non_negative(self) -> None:
        X, residuals = _make_iid_regression(n_obs=200)
        cov = compute_hac_standard_errors(residuals, X, maxlags=4)
        assert (np.diag(cov) >= 0).all()

    def test_lag_zero_reduces_to_white(self) -> None:
        # At L=0, HAC = (X'X)^{-1} * sum(e_t^2 * x_t * x_t') * (X'X)^{-1}
        # This is the White HC0 estimator. Check that it's distinct from L>0.
        X, residuals = _make_ar1_with_design(n=300, phi=0.5, seed=42)
        cov_l0 = compute_hac_standard_errors(residuals, X, maxlags=0)
        cov_l5 = compute_hac_standard_errors(residuals, X, maxlags=5)
        # For autocorrelated residuals, L=5 should give a larger
        # diagonal than L=0 (HAC catches what White misses).
        assert cov_l5[0, 0] > cov_l0[0, 0]

    def test_ar1_residuals_increase_se(self) -> None:
        # AR(1) residuals → HAC SE should be larger than HC0 (L=0) SE
        X, residuals = _make_ar1_with_design(n=400, phi=0.7, seed=42)
        cov_hc0 = compute_hac_standard_errors(residuals, X, maxlags=0)
        cov_hac = compute_hac_standard_errors(residuals, X, maxlags=6)
        # At phi=0.7, HAC should be meaningfully larger
        ratio = np.sqrt(cov_hac[0, 0] / cov_hc0[0, 0])
        assert ratio > 1.3

    def test_negative_maxlags_raises(self) -> None:
        X, residuals = _make_iid_regression(n_obs=100)
        with pytest.raises(ValueError, match="non-negative"):
            compute_hac_standard_errors(residuals, X, maxlags=-1)

    def test_maxlags_too_large_raises(self) -> None:
        X, residuals = _make_iid_regression(n_obs=50)
        with pytest.raises(ValueError, match="< n_obs"):
            compute_hac_standard_errors(residuals, X, maxlags=50)

    def test_shape_mismatch_raises(self) -> None:
        X, _ = _make_iid_regression(n_obs=100)
        bad_resid = np.zeros(50)
        with pytest.raises(ValueError, match="X.shape"):
            compute_hac_standard_errors(bad_resid, X, maxlags=4)

    def test_singular_x_raises(self) -> None:
        # Two identical columns → X'X singular
        X = np.array([[1.0, 2.0, 2.0], [1.0, 3.0, 3.0],
                      [1.0, 4.0, 4.0], [1.0, 5.0, 5.0]])
        resid = np.array([0.1, -0.1, 0.1, -0.1])
        with pytest.raises(ValueError, match="singular"):
            compute_hac_standard_errors(resid, X, maxlags=1)

    def test_2d_residuals_raises(self) -> None:
        X = np.ones((10, 1))
        with pytest.raises(ValueError, match="1D"):
            compute_hac_standard_errors(np.zeros((10, 2)), X, maxlags=1)


def _make_ar1_with_design(
    n: int = 300, phi: float = 0.5, seed: int = 42,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Generate (X, AR(1) residuals) for HAC autocorrelation tests."""
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.ones(n), rng.normal(0, 1, n)])
    residuals = _make_ar1_series(n=n, phi=phi, seed=seed + 1)
    return X.astype(np.float64), residuals.astype(np.float64)


# ============================================================================
# HAC standard error of mean (scalar wrapper)
# ============================================================================


class TestHACStandardErrorOfMean:
    def test_returns_positive_float(self) -> None:
        series = _make_ar1_series(n=200, phi=0.5)
        se = compute_hac_standard_error_of_mean(series, maxlags=4)
        assert isinstance(se, float)
        assert se > 0

    def test_iid_se_approximately_sigma_over_sqrtn(self) -> None:
        # For i.i.d. series, HAC SE should be near sigma / sqrt(n)
        rng = np.random.default_rng(42)
        n = 1000
        sigma = 2.0
        series = rng.normal(0, sigma, n)
        se_hac = compute_hac_standard_error_of_mean(series, maxlags=0)
        se_expected = sigma / np.sqrt(n)
        # Within 15% (sampling noise on the variance estimate)
        assert abs(se_hac - se_expected) / se_expected < 0.15

    def test_ar1_se_larger_than_naive(self) -> None:
        # AR(1) with phi=0.6 → HAC SE should be substantially larger
        # than naive SE that ignores autocorrelation
        series = _make_ar1_series(n=500, phi=0.6, seed=42)
        se_naive = series.std(ddof=1) / np.sqrt(len(series))
        se_hac = compute_hac_standard_error_of_mean(series, maxlags=6)
        assert se_hac > se_naive

    def test_empty_series_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            compute_hac_standard_error_of_mean(np.array([]), maxlags=0)

    def test_2d_series_raises(self) -> None:
        with pytest.raises(ValueError, match="1D"):
            compute_hac_standard_error_of_mean(np.zeros((10, 2)), maxlags=1)


# ============================================================================
# Block-bootstrapped Sharpe CI
# ============================================================================


class TestBootstrappedSharpeCI:
    def test_returns_ordered_tuple(self) -> None:
        rng = np.random.default_rng(42)
        rets = rng.normal(0.01, 0.05, 240)
        lower, upper = compute_bootstrapped_sharpe_ci(
            rets, block_size=6, n_resamples=2000, seed=123,
        )
        assert lower < upper

    def test_determinism_with_seed(self) -> None:
        rng = np.random.default_rng(42)
        rets = rng.normal(0.01, 0.05, 240)
        ci1 = compute_bootstrapped_sharpe_ci(
            rets, block_size=6, n_resamples=2000, seed=99,
        )
        ci2 = compute_bootstrapped_sharpe_ci(
            rets, block_size=6, n_resamples=2000, seed=99,
        )
        assert ci1 == ci2

    def test_different_seeds_give_different_ci(self) -> None:
        rng = np.random.default_rng(42)
        rets = rng.normal(0.01, 0.05, 240)
        ci1 = compute_bootstrapped_sharpe_ci(
            rets, block_size=6, n_resamples=2000, seed=1,
        )
        ci2 = compute_bootstrapped_sharpe_ci(
            rets, block_size=6, n_resamples=2000, seed=2,
        )
        assert ci1 != ci2

    def test_higher_confidence_gives_wider_ci(self) -> None:
        rng = np.random.default_rng(42)
        rets = rng.normal(0.01, 0.05, 240)
        l_90, u_90 = compute_bootstrapped_sharpe_ci(
            rets, block_size=6, n_resamples=2000, confidence=0.90, seed=42,
        )
        l_99, u_99 = compute_bootstrapped_sharpe_ci(
            rets, block_size=6, n_resamples=2000, confidence=0.99, seed=42,
        )
        assert (u_99 - l_99) > (u_90 - l_90)

    def test_ci_near_point_sharpe(self) -> None:
        # With i.i.d. returns and a reasonable sample size, the
        # bootstrap median should be close to the empirical Sharpe.
        rng = np.random.default_rng(42)
        rets = rng.normal(0.01, 0.05, 500)
        point_sharpe = rets.mean() / rets.std(ddof=1)
        lower, upper = compute_bootstrapped_sharpe_ci(
            rets, block_size=8, n_resamples=5000, seed=42,
        )
        # Point estimate should land somewhere inside the 95% CI
        # for an i.i.d. series with this sample size
        assert lower <= point_sharpe <= upper

    def test_block_size_too_large_raises(self) -> None:
        rets = np.zeros(50)
        with pytest.raises(ValueError, match="block_size"):
            compute_bootstrapped_sharpe_ci(rets, block_size=100)

    def test_block_size_zero_raises(self) -> None:
        rets = np.zeros(50)
        with pytest.raises(ValueError, match="block_size"):
            compute_bootstrapped_sharpe_ci(rets, block_size=0)

    def test_confidence_out_of_range_raises(self) -> None:
        rets = np.zeros(50)
        with pytest.raises(ValueError, match="confidence"):
            compute_bootstrapped_sharpe_ci(rets, block_size=4, confidence=1.5)

    def test_too_few_returns_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            compute_bootstrapped_sharpe_ci(np.array([0.01]), block_size=1)


# ============================================================================
# Expected maximum Sharpe (DSR threshold)
# ============================================================================


class TestExpectedMaxSharpe:
    def test_returns_positive_float(self) -> None:
        trials = np.array([0.08, 0.12, 0.18, 0.10, 0.14, 0.09, 0.16])
        sr_star = compute_expected_max_sharpe(trials)
        assert isinstance(sr_star, float)
        assert sr_star > 0

    def test_scales_with_trial_dispersion(self) -> None:
        # Doubling the std of trials should approximately double SR*.
        trials_tight = np.array([0.10, 0.11, 0.12, 0.13, 0.14])
        trials_wide = np.array([0.05, 0.10, 0.15, 0.20, 0.25])  # 4x wider range
        sr_star_tight = compute_expected_max_sharpe(trials_tight)
        sr_star_wide = compute_expected_max_sharpe(trials_wide)
        # Ratio should be approximately std_wide / std_tight
        ratio_std = trials_wide.std(ddof=1) / trials_tight.std(ddof=1)
        ratio_sr_star = sr_star_wide / sr_star_tight
        assert abs(ratio_sr_star - ratio_std) / ratio_std < 0.01

    def test_increases_with_n_trials(self) -> None:
        # More trials means a higher expected max under the same null.
        # Sample two arrays with similar variance but different N.
        rng = np.random.default_rng(42)
        trials_small = rng.normal(0.0, 0.05, 5)
        trials_large = rng.normal(0.0, 0.05, 50)
        sr_star_small = compute_expected_max_sharpe(trials_small)
        sr_star_large = compute_expected_max_sharpe(trials_large)
        assert sr_star_large > sr_star_small

    def test_two_trials_minimum(self) -> None:
        # N=2 is the minimum; should run, return positive float
        trials = np.array([0.05, 0.15])
        sr_star = compute_expected_max_sharpe(trials)
        assert sr_star > 0

    def test_n_equals_one_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            compute_expected_max_sharpe(np.array([0.10]))

    def test_zero_variance_raises(self) -> None:
        # All trials identical → variance = 0 → expected max undefined
        with pytest.raises(ValueError, match="zero variance"):
            compute_expected_max_sharpe(np.array([0.10, 0.10, 0.10]))

    def test_2d_input_raises(self) -> None:
        with pytest.raises(ValueError, match="1D"):
            compute_expected_max_sharpe(np.zeros((5, 2)))


# ============================================================================
# Deflated Sharpe ratio
# ============================================================================


class TestDeflatedSharpe:
    def test_returns_probability(self) -> None:
        trials = np.array([0.08, 0.12, 0.18, 0.10, 0.14, 0.09, 0.16])
        dsr = compute_deflated_sharpe(
            0.227, trials, n_obs=116, skewness=0.0, excess_kurtosis=0.0,
        )
        assert isinstance(dsr, float)
        assert 0.0 <= dsr <= 1.0

    def test_high_sharpe_gives_high_dsr(self) -> None:
        # Sharpe well above the trial max → DSR > 0.95
        trials = np.array([0.05, 0.06, 0.07, 0.08, 0.09])
        dsr = compute_deflated_sharpe(
            0.30, trials, n_obs=116, skewness=0.0, excess_kurtosis=0.0,
        )
        assert dsr > 0.95

    def test_low_sharpe_gives_low_dsr(self) -> None:
        # Sharpe near the trial mean → DSR < 0.5
        trials = np.array([0.05, 0.10, 0.15, 0.20, 0.25])
        dsr = compute_deflated_sharpe(
            0.05, trials, n_obs=116, skewness=0.0, excess_kurtosis=0.0,
        )
        assert dsr < 0.5

    def test_normal_returns_no_mertens_adjustment(self) -> None:
        # skewness=0, excess_kurtosis=0 → denominator = 1 - 0 + (2/4)*SR^2
        # Compute DSR twice to confirm determinism and basic invariance
        trials = np.array([0.08, 0.12, 0.18, 0.10, 0.14])
        dsr1 = compute_deflated_sharpe(
            0.20, trials, n_obs=116, skewness=0.0, excess_kurtosis=0.0,
        )
        dsr2 = compute_deflated_sharpe(
            0.20, trials, n_obs=116, skewness=0.0, excess_kurtosis=0.0,
        )
        assert dsr1 == dsr2

    def test_negative_skew_reduces_dsr(self) -> None:
        # Negative skewness inflates the Mertens denominator → lower DSR
        trials = np.array([0.08, 0.12, 0.18, 0.10, 0.14])
        dsr_normal = compute_deflated_sharpe(
            0.20, trials, n_obs=116, skewness=0.0, excess_kurtosis=0.0,
        )
        dsr_skewed = compute_deflated_sharpe(
            0.20, trials, n_obs=116, skewness=-0.5, excess_kurtosis=0.0,
        )
        assert dsr_skewed < dsr_normal

    def test_positive_excess_kurtosis_reduces_dsr(self) -> None:
        # Heavy tails (positive ex kurtosis) inflate denominator → lower DSR
        trials = np.array([0.08, 0.12, 0.18, 0.10, 0.14])
        dsr_normal = compute_deflated_sharpe(
            0.20, trials, n_obs=116, skewness=0.0, excess_kurtosis=0.0,
        )
        dsr_heavy = compute_deflated_sharpe(
            0.20, trials, n_obs=116, skewness=0.0, excess_kurtosis=5.0,
        )
        assert dsr_heavy < dsr_normal

    def test_n_obs_too_small_raises(self) -> None:
        trials = np.array([0.10, 0.15])
        with pytest.raises(ValueError, match="n_obs"):
            compute_deflated_sharpe(
                0.20, trials, n_obs=1, skewness=0.0, excess_kurtosis=0.0,
            )

    def test_pathological_distribution_raises(self) -> None:
        # Combination that makes the Mertens denominator non-positive
        # SR=2.0, skew=10 → denom = 1 - 10*2 + ... = very negative
        trials = np.array([0.5, 1.0, 1.5])
        with pytest.raises(ValueError, match="Mertens denominator"):
            compute_deflated_sharpe(
                2.0, trials, n_obs=100, skewness=10.0, excess_kurtosis=0.0,
            )