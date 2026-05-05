"""Tests for the betas module.

Pure-function tests with synthetic data designed to verify specific
beta-estimation behaviors: math correctness, window selection, NaN handling,
edge cases.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from axiom_fund.portfolio.betas import _ols_beta, compute_betas

# ============================================================================
# Helpers
# ============================================================================


def _make_returns_panel(
    permnos: list[int],
    dates: pd.DatetimeIndex,
    betas: list[float],
    mktrf: npt.NDArray[np.float64],
    idio_vol: float = 0.005,
    seed: int = 9999,   # different from typical mktrf seed of 42
) -> pd.DataFrame:
    """Build a long-format returns panel where stock i has known beta β_i.

    r_i,t = β_i × mktrf_t + ε_i,t  with  ε ~ N(0, idio_vol²)
    """
    rng = np.random.default_rng(seed)
    rows = []
    for permno, beta in zip(permnos, betas, strict=True):
        eps = rng.normal(0, idio_vol, size=len(dates))
        r = beta * mktrf + eps
        for d, ret in zip(dates, r, strict=True):
            rows.append({"permno": permno, "date": d, "ret": ret})
    return pd.DataFrame(rows)


def _make_ff_panel(
    dates: pd.DatetimeIndex, mktrf: npt.NDArray[np.float64]
) -> pd.DataFrame:
    return pd.DataFrame({"date": dates, "mktrf": mktrf})


# ============================================================================
# Helper function _ols_beta tested directly
# ============================================================================


class TestOLSBetaHelper:
    def test_recovers_known_beta_no_noise(self) -> None:
        """With zero noise, beta is recovered exactly."""
        rng = np.random.default_rng(42)
        n = 100
        mktrf = rng.normal(0, 0.01, size=n)
        for true_beta in [0.5, 1.0, 1.5, 2.0]:
            r = true_beta * mktrf
            est = _ols_beta(r, mktrf)
            assert est == pytest.approx(true_beta, abs=1e-10)

    def test_recovers_known_beta_with_noise(self) -> None:
        rng = np.random.default_rng(42)
        n = 252
        mktrf = rng.normal(0, 0.015, size=n)
        eps = rng.normal(0, 0.005, size=n)
        true_beta = 1.2
        r = true_beta * mktrf + eps
        est = _ols_beta(r, mktrf)
        assert abs(est - true_beta) < 0.05

    def test_zero_market_variance_raises(self) -> None:
        n = 50
        mktrf = np.zeros(n)
        r = np.ones(n)
        with pytest.raises(ValueError, match="market variance"):
            _ols_beta(r, mktrf)

    def test_unequal_length_raises(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            _ols_beta(np.zeros(10), np.zeros(5))

    def test_too_few_observations_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            _ols_beta(np.array([0.0]), np.array([0.0]))


# ============================================================================
# Input validation
# ============================================================================


class TestInputValidation:
    def test_returns_missing_columns_raises(self) -> None:
        bad = pd.DataFrame({"permno": [1], "date": [pd.Timestamp("2020-01-01")]})
        ff = pd.DataFrame({"date": [pd.Timestamp("2020-01-01")], "mktrf": [0.0]})
        with pytest.raises(ValueError, match="returns_df missing"):
            compute_betas(bad, ff, "2020-01-01")

    def test_ff_missing_columns_raises(self) -> None:
        rets = pd.DataFrame(
            {"permno": [1], "date": [pd.Timestamp("2020-01-01")], "ret": [0.01]}
        )
        bad = pd.DataFrame({"date": [pd.Timestamp("2020-01-01")]})
        with pytest.raises(ValueError, match="ff_factors_df missing"):
            compute_betas(rets, bad, "2020-01-01")

    def test_invalid_window_raises(self) -> None:
        rets = pd.DataFrame(
            {"permno": [1], "date": [pd.Timestamp("2020-01-01")], "ret": [0.01]}
        )
        ff = pd.DataFrame({"date": [pd.Timestamp("2020-01-01")], "mktrf": [0.0]})
        with pytest.raises(ValueError, match="window"):
            compute_betas(rets, ff, "2020-01-01", window=0)

    def test_invalid_min_obs_raises(self) -> None:
        rets = pd.DataFrame(
            {"permno": [1], "date": [pd.Timestamp("2020-01-01")], "ret": [0.01]}
        )
        ff = pd.DataFrame({"date": [pd.Timestamp("2020-01-01")], "mktrf": [0.0]})
        with pytest.raises(ValueError, match="min_obs"):
            compute_betas(rets, ff, "2020-01-01", min_obs=1)

    def test_min_obs_exceeds_window_raises(self) -> None:
        rets = pd.DataFrame(
            {"permno": [1], "date": [pd.Timestamp("2020-01-01")], "ret": [0.01]}
        )
        ff = pd.DataFrame({"date": [pd.Timestamp("2020-01-01")], "mktrf": [0.0]})
        with pytest.raises(ValueError, match="must be <= window"):
            compute_betas(rets, ff, "2020-01-01", window=10, min_obs=20)


# ============================================================================
# Output structure
# ============================================================================


class TestOutputStructure:
    def test_returns_pandas_series(self) -> None:
        dates = pd.bdate_range("2020-01-01", periods=100)
        rng = np.random.default_rng(42)
        mktrf = rng.normal(0, 0.01, size=100)
        rets = _make_returns_panel([1, 2], dates, [1.0, 1.5], mktrf)
        ff = _make_ff_panel(dates, mktrf)
        result = compute_betas(rets, ff, dates[-1], window=100, min_obs=50)
        assert isinstance(result, pd.Series)
        assert result.name == "beta"

    def test_indexed_by_permno(self) -> None:
        dates = pd.bdate_range("2020-01-01", periods=100)
        rng = np.random.default_rng(42)
        mktrf = rng.normal(0, 0.01, size=100)
        rets = _make_returns_panel([1, 2, 3], dates, [0.8, 1.0, 1.2], mktrf)
        ff = _make_ff_panel(dates, mktrf)
        result = compute_betas(rets, ff, dates[-1], window=100, min_obs=50)
        assert sorted(result.index.tolist()) == [1, 2, 3]

    def test_dtype_is_float(self) -> None:
        dates = pd.bdate_range("2020-01-01", periods=100)
        rng = np.random.default_rng(42)
        mktrf = rng.normal(0, 0.01, size=100)
        rets = _make_returns_panel([1, 2], dates, [1.0, 1.5], mktrf)
        ff = _make_ff_panel(dates, mktrf)
        result = compute_betas(rets, ff, dates[-1], window=100, min_obs=50)
        assert result.dtype == float


# ============================================================================
# Math correctness
# ============================================================================


class TestMathCorrectness:
    def test_recovers_known_betas(self) -> None:
        """With idio noise vs. market vol of ~0.3, betas recover within ±0.05."""
        dates = pd.bdate_range("2019-01-01", periods=252)
        rng = np.random.default_rng(42)
        mktrf = rng.normal(0.0005, 0.015, size=252)
        permnos = [1, 2, 3, 4, 5]
        true_betas = [0.5, 0.8, 1.0, 1.2, 1.8]
        rets = _make_returns_panel(permnos, dates, true_betas, mktrf, idio_vol=0.005)
        ff = _make_ff_panel(dates, mktrf)
        result = compute_betas(rets, ff, dates[-1], window=252, min_obs=60)
        for permno, true_b in zip(permnos, true_betas, strict=True):
            assert abs(result.loc[permno] - true_b) < 0.05

    def test_zero_beta_recovered(self) -> None:
        """Pure idio noise (β=0) should yield β estimate near zero."""
        dates = pd.bdate_range("2019-01-01", periods=252)
        rng = np.random.default_rng(42)
        mktrf = rng.normal(0, 0.015, size=252)
        rets = _make_returns_panel([1], dates, [0.0], mktrf, idio_vol=0.01)
        ff = _make_ff_panel(dates, mktrf)
        result = compute_betas(rets, ff, dates[-1], window=252, min_obs=60)
        assert abs(result.loc[1]) < 0.1

    def test_negative_beta_recovered(self) -> None:
        """Negative beta (e.g., gold-like asset) should be recovered."""
        dates = pd.bdate_range("2019-01-01", periods=252)
        rng = np.random.default_rng(42)
        mktrf = rng.normal(0, 0.015, size=252)
        rets = _make_returns_panel([1], dates, [-0.7], mktrf, idio_vol=0.005)
        ff = _make_ff_panel(dates, mktrf)
        result = compute_betas(rets, ff, dates[-1], window=252, min_obs=60)
        assert abs(result.loc[1] - (-0.7)) < 0.05


# ============================================================================
# Window selection
# ============================================================================


class TestWindowSelection:
    def test_uses_window_days_only(self) -> None:
        """Earlier data outside window should be ignored."""
        # 500 days of data; window=100 should only use the last 100
        dates = pd.bdate_range("2018-01-01", periods=500)
        rng = np.random.default_rng(42)
        # First 400 days: market and stock unrelated (β=0)
        # Last 100 days: stock = 2.0 × market exactly
        mktrf = rng.normal(0, 0.015, size=500)
        ret_unrelated = rng.normal(0, 0.01, size=400)
        ret_related = 2.0 * mktrf[400:]
        ret_combined = np.concatenate([ret_unrelated, ret_related])

        rets = pd.DataFrame({
            "permno": [1] * 500,
            "date": dates,
            "ret": ret_combined,
        })
        ff = _make_ff_panel(dates, mktrf)

        result = compute_betas(rets, ff, dates[-1], window=100, min_obs=60)
        # Should be ~2.0, not the average of 0 and 2
        assert abs(result.loc[1] - 2.0) < 0.05

    def test_as_of_date_filters_future_data(self) -> None:
        """Data after as_of_date should be ignored."""
        dates = pd.bdate_range("2019-01-01", periods=400)
        rng = np.random.default_rng(42)
        mktrf = rng.normal(0, 0.015, size=400)
        # First 200 days β=1.0; next 200 days β=3.0 (would skew if included)
        ret_first = 1.0 * mktrf[:200] + rng.normal(0, 0.005, size=200)
        ret_second = 3.0 * mktrf[200:]
        ret_combined = np.concatenate([ret_first, ret_second])
        rets = pd.DataFrame({
            "permno": [1] * 400,
            "date": dates,
            "ret": ret_combined,
        })
        ff = _make_ff_panel(dates, mktrf)
        # as_of midway through; should only see first 200 days → β ≈ 1.0
        result = compute_betas(rets, ff, dates[199], window=200, min_obs=60)
        assert abs(result.loc[1] - 1.0) < 0.1


# ============================================================================
# NaN and degenerate cases
# ============================================================================


class TestEdgeCases:
    def test_insufficient_data_yields_nan(self) -> None:
        """A name with fewer than min_obs valid points gets NaN beta."""
        dates = pd.bdate_range("2020-01-01", periods=100)
        rng = np.random.default_rng(42)
        mktrf = rng.normal(0, 0.015, size=100)
        # Stock 1: full data; Stock 2: only first 30 days
        rets1 = _make_returns_panel([1], dates, [1.0], mktrf)
        rets2_partial = _make_returns_panel(
            [2], dates[:30], [1.5], mktrf[:30]
        )
        rets = pd.concat([rets1, rets2_partial], ignore_index=True)
        ff = _make_ff_panel(dates, mktrf)
        result = compute_betas(rets, ff, dates[-1], window=100, min_obs=60)
        assert not pd.isna(result.loc[1])
        assert pd.isna(result.loc[2])

    def test_nan_returns_dropped(self) -> None:
        """Rows with NaN return should be dropped per stock."""
        dates = pd.bdate_range("2020-01-01", periods=200)
        rng = np.random.default_rng(42)
        mktrf = rng.normal(0, 0.015, size=200)
        rets = _make_returns_panel([1], dates, [1.0], mktrf, idio_vol=0.005)
        # Inject NaN in 10 random rows
        nan_indices = rng.choice(200, size=10, replace=False)
        rets.loc[nan_indices, "ret"] = np.nan
        ff = _make_ff_panel(dates, mktrf)
        result = compute_betas(rets, ff, dates[-1], window=200, min_obs=60)
        # Should still recover β ≈ 1.0
        assert abs(result.loc[1] - 1.0) < 0.05

    def test_empty_returns_yields_empty_series(self) -> None:
        rets = pd.DataFrame(columns=["permno", "date", "ret"])
        ff = pd.DataFrame(
            {"date": pd.bdate_range("2020-01-01", periods=100),
             "mktrf": np.zeros(100)}
        )
        result = compute_betas(rets, ff, "2020-04-01", window=100, min_obs=60)
        assert len(result) == 0
