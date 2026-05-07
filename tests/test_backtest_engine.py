"""Tests for the single-period backtest engine.

Pure-function tests with synthetic inputs designed to verify specific
behaviors: input validation, math correctness, edge cases.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from axiom_fund.backtest.engine import (
    BacktestPeriodInputs,
    BacktestPeriodResult,
    _compute_buy_and_hold_returns,
    run_backtest_period,
)

# ============================================================================
# Helpers
# ============================================================================


def _make_synthetic_inputs(
    n_assets: int = 30,
    rebalance: str = "2020-09-30",
    holding_period_days: int = 42,
    seed: int = 42,
) -> BacktestPeriodInputs:
    """Build a BacktestPeriodInputs with synthetic, well-conditioned data."""
    rng = np.random.default_rng(seed)
    permnos = list(range(1, n_assets + 1))

    alpha = pd.Series(
        rng.normal(0, 1, size=n_assets), index=permnos, name="composite_z"
    )

    # Diagonal covariance, ~25% annual vol
    cov_arr: npt.NDArray[np.float64] = np.eye(n_assets) * 0.06
    cov = pd.DataFrame(cov_arr, index=permnos, columns=permnos)

    betas = pd.Series(
        rng.uniform(0.5, 1.5, size=n_assets), index=permnos
    )
    sectors = pd.Series(
        ([10] * (n_assets // 3))
        + ([20] * (n_assets // 3))
        + ([30] * (n_assets - 2 * (n_assets // 3))),
        index=permnos,
    )

    rebalance_ts = pd.Timestamp(rebalance)
    holding_dates = pd.bdate_range(
        start=rebalance_ts + pd.Timedelta(days=1),
        periods=holding_period_days,
    )
    future_rets = pd.DataFrame(
        rng.normal(0.0005, 0.015, size=(holding_period_days, n_assets)),
        index=holding_dates,
        columns=permnos,
    )

    return BacktestPeriodInputs(
        rebalance_date=rebalance_ts,
        alpha=alpha,
        covariance=cov,
        betas=betas,
        sectors=sectors,
        holding_period_returns=future_rets,
    )


# ============================================================================
# Buy-and-hold helper tested directly
# ============================================================================


class TestBuyAndHoldHelper:
    def test_known_returns_known_weights(self) -> None:
        """Hand-controllable case: weights × cumulative returns."""
        n = 21
        # Stock 1 doubles (+100%), stock 2 flat, stock 3 halves (-50%)
        rets = pd.DataFrame({
            1: [(2.0 ** (1 / n)) - 1] * n,
            2: [0.0] * n,
            3: [(0.5 ** (1 / n)) - 1] * n,
        }, index=pd.bdate_range("2020-10-01", periods=n))
        weights = pd.Series([0.5, -0.5, 0.5], index=[1, 2, 3])
        result = _compute_buy_and_hold_returns(rets, weights, n)
        # 0.5 × 1.0 + (-0.5) × 0.0 + 0.5 × (-0.5) = 0.25
        assert result == pytest.approx(0.25, abs=1e-10)

    def test_zero_weights_yields_zero_return(self) -> None:
        n = 21
        rets = pd.DataFrame(
            np.random.default_rng(42).normal(0, 0.01, size=(n, 3)),
            columns=[1, 2, 3],
            index=pd.bdate_range("2020-10-01", periods=n),
        )
        weights = pd.Series([0.0, 0.0, 0.0], index=[1, 2, 3])
        assert _compute_buy_and_hold_returns(rets, weights, n) == 0.0

    def test_truncates_to_holding_days(self) -> None:
        """Only the first holding_days rows should be used."""
        n_full = 60
        rets = pd.DataFrame({
            1: [0.001] * n_full,
        }, index=pd.bdate_range("2020-10-01", periods=n_full))
        weights = pd.Series([1.0], index=[1])
        # 21 days of 0.1% daily → cumulative ~0.0212
        r21 = _compute_buy_and_hold_returns(rets, weights, 21)
        # 60 days → cumulative ~0.0617
        r60 = _compute_buy_and_hold_returns(rets, weights, 60)
        assert r21 < r60
        assert abs(r21 - ((1.001 ** 21) - 1)) < 1e-10

    def test_nan_treated_as_zero(self) -> None:
        n = 21
        rets = pd.DataFrame({
            1: [0.001] * n,
            2: [np.nan] * n,
        }, index=pd.bdate_range("2020-10-01", periods=n))
        weights = pd.Series([0.5, 0.5], index=[1, 2])
        result = _compute_buy_and_hold_returns(rets, weights, n)
        # Stock 2 with NaN returns treated as flat: contributes 0.5 × 0.0 = 0
        # Stock 1: 0.5 × ((1.001 ** 21) - 1)
        expected = 0.5 * ((1.001 ** 21) - 1)
        assert result == pytest.approx(expected, abs=1e-10)


# ============================================================================
# Input validation
# ============================================================================


class TestInputValidation:
    def test_inputs_must_be_dataclass(self) -> None:
        with pytest.raises(ValueError, match="BacktestPeriodInputs"):
            run_backtest_period({"alpha": "fake"})  # type: ignore[arg-type]

    def test_holding_days_must_be_positive(self) -> None:
        inputs = _make_synthetic_inputs()
        with pytest.raises(ValueError, match="holding_days"):
            run_backtest_period(inputs, holding_days=0)

    def test_covariance_misalignment_raises(self) -> None:
        inputs = _make_synthetic_inputs()
        # Build a covariance with different permnos
        bad_cov = inputs.covariance.copy()
        bad_cov.index = list(range(100, 100 + len(inputs.covariance)))
        bad_cov.columns = bad_cov.index
        bad_inputs = BacktestPeriodInputs(
            rebalance_date=inputs.rebalance_date,
            alpha=inputs.alpha,
            covariance=bad_cov,
            betas=inputs.betas,
            sectors=inputs.sectors,
            holding_period_returns=inputs.holding_period_returns,
        )
        with pytest.raises(ValueError, match=r"covariance\.index"):
            run_backtest_period(bad_inputs)

    def test_betas_misalignment_raises(self) -> None:
        inputs = _make_synthetic_inputs()
        bad_betas = pd.Series(
            inputs.betas.values,
            index=list(range(100, 100 + len(inputs.betas))),
        )
        bad_inputs = BacktestPeriodInputs(
            rebalance_date=inputs.rebalance_date,
            alpha=inputs.alpha,
            covariance=inputs.covariance,
            betas=bad_betas,
            sectors=inputs.sectors,
            holding_period_returns=inputs.holding_period_returns,
        )
        with pytest.raises(ValueError, match=r"betas\.index"):
            run_backtest_period(bad_inputs)

    def test_sectors_misalignment_raises(self) -> None:
        inputs = _make_synthetic_inputs()
        bad_sectors = pd.Series(
            inputs.sectors.values,
            index=list(range(100, 100 + len(inputs.sectors))),
        )
        bad_inputs = BacktestPeriodInputs(
            rebalance_date=inputs.rebalance_date,
            alpha=inputs.alpha,
            covariance=inputs.covariance,
            betas=inputs.betas,
            sectors=bad_sectors,
            holding_period_returns=inputs.holding_period_returns,
        )
        with pytest.raises(ValueError, match=r"sectors\.index"):
            run_backtest_period(bad_inputs)

    def test_holding_returns_missing_permnos_raises(self) -> None:
        inputs = _make_synthetic_inputs()
        # Drop one permno from the holding-period returns
        bad_hpr = inputs.holding_period_returns.iloc[:, :-1]
        bad_inputs = BacktestPeriodInputs(
            rebalance_date=inputs.rebalance_date,
            alpha=inputs.alpha,
            covariance=inputs.covariance,
            betas=inputs.betas,
            sectors=inputs.sectors,
            holding_period_returns=bad_hpr,
        )
        with pytest.raises(ValueError, match=r"holding_period_returns\.columns"):
            run_backtest_period(bad_inputs)


# ============================================================================
# Point-in-time discipline
# ============================================================================


class TestPointInTimeDiscipline:
    def test_holding_period_must_start_after_rebalance(self) -> None:
        """Look-ahead guard: holding-period returns must be strictly after."""
        inputs = _make_synthetic_inputs()
        # Construct holding returns whose first date == rebalance_date
        bad_dates = pd.DatetimeIndex(
            [
                inputs.rebalance_date,  # rebalance_date itself
                *pd.bdate_range(
                    start=inputs.rebalance_date + pd.Timedelta(days=1),
                    periods=41,
                ),
            ]
        )
        bad_hpr = pd.DataFrame(
            np.zeros((42, len(inputs.alpha))),
            index=bad_dates,
            columns=inputs.alpha.index,
        )
        bad_inputs = BacktestPeriodInputs(
            rebalance_date=inputs.rebalance_date,
            alpha=inputs.alpha,
            covariance=inputs.covariance,
            betas=inputs.betas,
            sectors=inputs.sectors,
            holding_period_returns=bad_hpr,
        )
        with pytest.raises(ValueError, match="strictly after"):
            run_backtest_period(bad_inputs)

    def test_holding_period_too_short_raises(self) -> None:
        inputs_full = _make_synthetic_inputs(holding_period_days=42)
        # Only 10 future days available, request 21
        short_hpr = inputs_full.holding_period_returns.iloc[:10]
        bad_inputs = BacktestPeriodInputs(
            rebalance_date=inputs_full.rebalance_date,
            alpha=inputs_full.alpha,
            covariance=inputs_full.covariance,
            betas=inputs_full.betas,
            sectors=inputs_full.sectors,
            holding_period_returns=short_hpr,
        )
        with pytest.raises(ValueError, match="holding_days=21"):
            run_backtest_period(bad_inputs, holding_days=21)


# ============================================================================
# End-to-end behavior
# ============================================================================


class TestEndToEnd:
    def test_returns_period_result(self) -> None:
        inputs = _make_synthetic_inputs()
        result = run_backtest_period(inputs)
        assert isinstance(result, BacktestPeriodResult)

    def test_optimizer_status_optimal(self) -> None:
        inputs = _make_synthetic_inputs()
        result = run_backtest_period(inputs)
        assert result.optimizer_status in ("optimal", "optimal_inaccurate")

    def test_neutrality_constraints_active_by_default(self) -> None:
        """Default (all neutrality on) should give net & beta ≈ 0."""
        inputs = _make_synthetic_inputs()
        result = run_backtest_period(inputs)
        assert abs(result.net_exposure) < 1e-6
        assert abs(result.portfolio_beta) < 1e-6  # type: ignore[arg-type]

    def test_realized_return_matches_buy_and_hold(self) -> None:
        """Engine output should equal _compute_buy_and_hold_returns(weights)."""
        inputs = _make_synthetic_inputs()
        result = run_backtest_period(inputs, holding_days=21)
        expected = _compute_buy_and_hold_returns(
            inputs.holding_period_returns, result.weights, 21
        )
        assert result.realized_return == pytest.approx(expected, abs=1e-10)

    def test_holding_period_end_is_last_used_day(self) -> None:
        inputs = _make_synthetic_inputs(holding_period_days=42)
        result = run_backtest_period(inputs, holding_days=21)
        # Should be the 21st trading day after rebalance
        expected = inputs.holding_period_returns.index[20]
        assert result.holding_period_end == expected

    def test_can_disable_neutrality_constraints(self) -> None:
        inputs = _make_synthetic_inputs()
        result_constrained = run_backtest_period(inputs)
        result_unconstrained = run_backtest_period(
            inputs,
            constrain_dollar_neutral=False,
            constrain_beta_neutral=False,
            constrain_sector_neutral=False,
        )
        # Unconstrained will generally have nonzero net/beta
        # (constrained must have ≈ 0)
        assert abs(result_constrained.net_exposure) < 1e-6
        # The unconstrained version doesn't need to satisfy neutrality;
        # just verify both ran successfully
        assert result_unconstrained.optimizer_status in (
            "optimal", "optimal_inaccurate"
        )


# ============================================================================
# Math correctness
# ============================================================================


class TestMathCorrectness:
    def test_zero_holding_returns_yields_zero_realized(self) -> None:
        """If all future returns are zero, realized_return must be 0."""
        inputs = _make_synthetic_inputs()
        zero_hpr = pd.DataFrame(
            np.zeros_like(inputs.holding_period_returns.values),
            index=inputs.holding_period_returns.index,
            columns=inputs.holding_period_returns.columns,
        )
        zero_inputs = BacktestPeriodInputs(
            rebalance_date=inputs.rebalance_date,
            alpha=inputs.alpha,
            covariance=inputs.covariance,
            betas=inputs.betas,
            sectors=inputs.sectors,
            holding_period_returns=zero_hpr,
        )
        result = run_backtest_period(zero_inputs)
        assert abs(result.realized_return) < 1e-10

    def test_uniform_positive_returns_yields_zero_for_dollar_neutral(self) -> None:
        """If every stock returns +1% uniformly and we're dollar-neutral,
        portfolio return should be ~0 (gross alpha cancels)."""
        inputs = _make_synthetic_inputs()
        n_days = len(inputs.holding_period_returns)
        n_assets = len(inputs.alpha)
        uniform_hpr = pd.DataFrame(
            np.full((n_days, n_assets), 0.001),  # +0.1% daily everywhere
            index=inputs.holding_period_returns.index,
            columns=inputs.holding_period_returns.columns,
        )
        uniform_inputs = BacktestPeriodInputs(
            rebalance_date=inputs.rebalance_date,
            alpha=inputs.alpha,
            covariance=inputs.covariance,
            betas=inputs.betas,
            sectors=inputs.sectors,
            holding_period_returns=uniform_hpr,
        )
        result = run_backtest_period(uniform_inputs)
        # All stocks have identical cumulative returns, so the dot
        # product with (dollar-neutral) weights is zero
        assert abs(result.realized_return) < 1e-10
