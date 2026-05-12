"""Tests for the performance metrics module.

Pure-function tests with hand-controllable synthetic data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from axiom_fund.backtest.metrics import (
    DrawdownEpisode,
    PerformanceMetrics,
    ReturnMoments,
    SharpeResult,
    compute_annualized_return,
    compute_annualized_vol,
    compute_drawdown_episodes,
    compute_hit_rate,
    compute_max_drawdown,
    compute_performance_metrics,
    compute_return_moments,
    compute_sharpe,
)

# ============================================================================
# Helpers
# ============================================================================


def _monthly_series(values: list[float], start: str = "2020-01-31") -> pd.Series:
    return pd.Series(
        values,
        index=pd.date_range(start, periods=len(values), freq="M"),
    )


def _equity_to_returns(equity: list[float]) -> pd.Series:
    """Convert an equity curve to a return series. Returns has length N-1."""
    arr = np.array(equity)
    rets = arr[1:] / arr[:-1] - 1
    return _monthly_series(rets.tolist())


# ============================================================================
# Annualized return
# ============================================================================


class TestAnnualizedReturn:
    def test_zero_returns_yield_zero(self) -> None:
        r = _monthly_series([0.0] * 12)
        assert compute_annualized_return(r) == pytest.approx(0.0, abs=1e-10)

    def test_constant_positive_returns(self) -> None:
        # 1% monthly for 12 months: should annualize to exactly (1.01)^12 - 1
        r = _monthly_series([0.01] * 12)
        ann = compute_annualized_return(r)
        expected = (1.01 ** 12) - 1
        assert ann == pytest.approx(expected, abs=1e-6)

    def test_empty_series_returns_nan(self) -> None:
        r = pd.Series([], dtype=float)
        assert np.isnan(compute_annualized_return(r))


# ============================================================================
# Annualized vol
# ============================================================================


class TestAnnualizedVol:
    def test_zero_vol(self) -> None:
        r = _monthly_series([0.01] * 12)
        assert compute_annualized_vol(r) == pytest.approx(0.0, abs=1e-10)

    def test_monthly_vol_scales_sqrt_12(self) -> None:
        rng = np.random.default_rng(42)
        r = _monthly_series(rng.normal(0, 0.01, size=120).tolist())
        ann = compute_annualized_vol(r)
        expected = r.std() * np.sqrt(12)
        assert ann == pytest.approx(expected, abs=1e-10)

    def test_too_short_returns_nan(self) -> None:
        r = _monthly_series([0.01])
        assert np.isnan(compute_annualized_vol(r))


# ============================================================================
# Sharpe with CI
# ============================================================================


class TestSharpe:
    def test_zero_excess_return_yields_zero_sharpe(self) -> None:
        r = _monthly_series([0.0] * 60)
        result = compute_sharpe(r)
        # With zero return AND zero vol, we return NaN
        # (can't compute Sharpe when ann_vol = 0)
        assert np.isnan(result.sharpe)

    def test_positive_constant_returns_undefined_sharpe(self) -> None:
        # Constant positive returns → vol is essentially zero → Sharpe undefined
        r = _monthly_series([0.01] * 60)
        result = compute_sharpe(r)
        assert np.isnan(result.sharpe)

    def test_known_sharpe_estimate(self) -> None:
        # Constructed series: 1% mean monthly, std 1% monthly → annualized:
        # ret ≈ 12.68%, vol ≈ 3.46%, Sharpe ≈ 3.7
        rng = np.random.default_rng(42)
        n = 600  # 50 years of monthly data → small SE
        # Use deterministic series with known properties
        r = _monthly_series((rng.normal(0.01, 0.01, size=n)).tolist())
        result = compute_sharpe(r)
        # Empirical Sharpe should be close to theoretical:
        # ratio of (1.01)^12 - 1 to 0.01 × √12 ≈ 0.1268 / 0.0346 ≈ 3.66
        assert 2.0 < result.sharpe < 6.0
        # CI should be tight given large N
        assert (result.ci_high - result.ci_low) < 1.0

    def test_ci_widens_with_smaller_n(self) -> None:
        rng = np.random.default_rng(42)
        small = _monthly_series(rng.normal(0.005, 0.01, size=24).tolist())
        large = _monthly_series(rng.normal(0.005, 0.01, size=1000).tolist())
        small_ci_width = compute_sharpe(small).ci_high - compute_sharpe(small).ci_low
        large_ci_width = compute_sharpe(large).ci_high - compute_sharpe(large).ci_low
        assert small_ci_width > large_ci_width

    def test_returns_sharpe_result_type(self) -> None:
        r = _monthly_series([0.01, -0.005, 0.02, -0.01] * 6)
        result = compute_sharpe(r)
        assert isinstance(result, SharpeResult)
        assert result.n_periods == 24


# ============================================================================
# Hit rate
# ============================================================================


class TestHitRate:
    def test_all_positive(self) -> None:
        r = _monthly_series([0.01, 0.02, 0.03])
        assert compute_hit_rate(r) == 1.0

    def test_all_negative(self) -> None:
        r = _monthly_series([-0.01, -0.02])
        assert compute_hit_rate(r) == 0.0

    def test_mixed(self) -> None:
        r = _monthly_series([0.01, -0.02, 0.03, 0.0, 0.04])
        # 3 of 5 strictly positive (zero doesn't count)
        assert compute_hit_rate(r) == 0.6


# ============================================================================
# Drawdown episodes
# ============================================================================


class TestDrawdownEpisodes:
    def test_no_drawdown(self) -> None:
        r = _monthly_series([0.01, 0.02, 0.03])
        episodes = compute_drawdown_episodes(r)
        assert len(episodes) == 0

    def test_single_recovered_episode(self) -> None:
        # Equity: 1 → 1.1 → 0.99 → 1.1
        r = _equity_to_returns([1.0, 1.10, 0.99, 1.10])
        episodes = compute_drawdown_episodes(r)
        assert len(episodes) == 1
        ep = episodes[0]
        assert ep.peak_date == r.index[0]
        assert ep.trough_date == r.index[1]
        assert ep.recovery_date == r.index[2]
        assert ep.max_depth == pytest.approx(-0.10, abs=1e-6)

    def test_two_distinct_episodes(self) -> None:
        # Equity: 1 → 1.10 → 0.99 → 1.155 → 1.045 → 1.155
        r = _equity_to_returns([1.0, 1.10, 0.99, 1.155, 1.045, 1.155])
        episodes = compute_drawdown_episodes(r)
        assert len(episodes) == 2
        # First episode -10%, recovered
        assert episodes[0].max_depth == pytest.approx(-0.10, abs=1e-6)
        assert episodes[0].recovery_date is not None
        # Second episode -9.52%, recovered
        assert episodes[1].max_depth == pytest.approx(-0.09524, abs=1e-4)
        assert episodes[1].recovery_date is not None

    def test_unrecovered_episode(self) -> None:
        # Equity: 1 → 1.10 → 0.99 (no recovery)
        r = _equity_to_returns([1.0, 1.10, 0.99])
        episodes = compute_drawdown_episodes(r)
        assert len(episodes) == 1
        assert episodes[0].recovery_date is None
        assert episodes[0].recovery_periods is None

    def test_max_drawdown_picks_deepest(self) -> None:
        # Two episodes: -10% and -5%; max_drawdown should be -10%
        r = _equity_to_returns([1.0, 1.10, 0.99, 1.155, 1.10, 1.155])
        max_dd = compute_max_drawdown(r)
        assert max_dd == pytest.approx(-0.10, abs=1e-6)

    def test_empty_series_returns_empty_tuple(self) -> None:
        episodes = compute_drawdown_episodes(pd.Series([], dtype=float))
        assert episodes == ()


# ============================================================================
# Return moments
# ============================================================================


class TestReturnMoments:
    def test_normal_distribution_moments(self) -> None:
        rng = np.random.default_rng(42)
        # 2000 months is enough to estimate moments accurately and fits
        # within pandas' Timestamp range (max year 2262)
        r = _monthly_series(rng.normal(0, 0.01, size=2000).tolist())
        m = compute_return_moments(r)
        assert m.mean == pytest.approx(0.0, abs=0.001)
        assert m.std == pytest.approx(0.01, abs=0.001)
        assert abs(m.skew) < 0.2  # normal has 0 skew
        assert abs(m.kurtosis) < 0.3  # normal has 0 excess kurtosis

    def test_constant_series_returns_zero_std(self) -> None:
        r = _monthly_series([0.01] * 100)
        m = compute_return_moments(r)
        assert m.std == pytest.approx(0.0, abs=1e-10)


# ============================================================================
# Performance metrics aggregator
# ============================================================================


class TestPerformanceMetrics:
    def test_aggregator_returns_correct_type(self) -> None:
        r = _monthly_series([0.01, -0.005, 0.02, -0.01] * 30)  # 120 months
        m = compute_performance_metrics(r)
        assert isinstance(m, PerformanceMetrics)
        assert isinstance(m.sharpe, SharpeResult)
        assert isinstance(m.moments, ReturnMoments)
        assert m.n_periods == 120

    def test_empty_input_raises(self) -> None:
        r = pd.Series([], dtype=float)
        with pytest.raises(ValueError, match="empty"):
            compute_performance_metrics(r)

    def test_max_dd_matches_drawdown_episodes(self) -> None:
        r = _equity_to_returns([1.0, 1.10, 0.99, 1.155, 1.045, 1.155])
        m = compute_performance_metrics(r)
        # Should match -10% (the deepest of the two episodes)
        assert m.max_drawdown == pytest.approx(-0.10, abs=1e-6)
        assert m.max_drawdown_episode is not None
        assert isinstance(m.max_drawdown_episode, DrawdownEpisode)
