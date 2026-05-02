"""Tests for the Residual Momentum signal module.

Pure-function tests using synthetic data with known cross-sectional structure.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from axiom_fund.signals.residual_momentum import (
    RESMOM_RAW_COLUMNS,
    compute_residual_momentum,
)

# ============================================================================
# Helpers
# ============================================================================


def _make_synthetic_panel(
    n_stocks: int = 30,
    n_months: int = 24,
    seed: int = 42,
    industry_effects: dict[str, float] | None = None,
    size_coef: float = -0.005,
    idio_pattern: list[float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, float]]:
    """Build synthetic daily returns + fundamentals with known structure.

    Returns
    -------
    rets : daily returns DataFrame
    fund : fundamentals DataFrame
    idio_map : dict mapping permno -> assigned idiosyncratic monthly return
    """
    np.random.seed(seed)
    industries = ["Tech", "Finance", "Health"]
    if industry_effects is None:
        industry_effects = {"Tech": 0.020, "Finance": 0.0, "Health": 0.010}

    permnos = list(range(1, n_stocks + 1))
    stock_industries = [industries[i % 3] for i in range(n_stocks)]
    sizes = np.exp(np.linspace(np.log(1e9), np.log(1e12), n_stocks))

    idio_map: dict[int, float] = {}
    rows: list[dict[str, object]] = []
    fund_rows: list[dict[str, object]] = []
    for permno_idx, permno in enumerate(permnos):
        ggroup = stock_industries[permno_idx]
        size = sizes[permno_idx]
        log_size = np.log(size)
        if idio_pattern is None:
            idio = 0.015 if permno_idx % 2 == 0 else -0.015
        else:
            idio = idio_pattern[permno_idx]
        idio_map[permno] = idio

        for month in range(n_months):
            ym_start = pd.Timestamp("2020-01-01") + pd.DateOffset(months=month)
            month_end = ym_start + pd.offsets.MonthEnd(0)
            n_days = (month_end - ym_start).days + 1
            true_monthly = (
                industry_effects[ggroup]
                + size_coef * log_size
                + idio
                + np.random.normal(0, 0.005)
            )
            daily_return = (1 + true_monthly) ** (1 / n_days) - 1
            for day_offset in range(n_days):
                d = ym_start + pd.Timedelta(days=day_offset)
                rows.append(
                    {"permno": permno, "date": d, "ret": daily_return, "marketcap": size}
                )

        fund_rows.append(
            {
                "permno": permno,
                "rdq": pd.Timestamp("2019-12-31"),
                "datadate": pd.Timestamp("2019-09-30"),
                "ggroup": ggroup,
            }
        )

    return pd.DataFrame(rows), pd.DataFrame(fund_rows), idio_map


# ============================================================================
# Input validation
# ============================================================================


class TestInputValidation:
    def test_reversed_dates_raises(self) -> None:
        rets, fund, _ = _make_synthetic_panel()
        with pytest.raises(ValueError, match="start_date"):
            compute_residual_momentum(rets, fund, "2021-12-31", "2020-01-01")

    def test_missing_returns_columns_raises(self) -> None:
        _, fund, _ = _make_synthetic_panel()
        rets = pd.DataFrame({"foo": [1]})
        with pytest.raises(ValueError, match="returns_df missing"):
            compute_residual_momentum(rets, fund, "2020-01-01", "2021-12-31")

    def test_missing_fundamentals_columns_raises(self) -> None:
        rets, _, _ = _make_synthetic_panel()
        fund = pd.DataFrame({"foo": [1]})
        with pytest.raises(ValueError, match="fundamentals_df missing"):
            compute_residual_momentum(rets, fund, "2020-01-01", "2021-12-31")

    def test_formation_start_must_exceed_end(self) -> None:
        rets, fund, _ = _make_synthetic_panel()
        with pytest.raises(ValueError, match="formation_start"):
            compute_residual_momentum(
                rets, fund, "2020-01-01", "2021-12-31",
                formation_start=2, formation_end=12,
            )

    def test_min_months_exceeds_window_raises(self) -> None:
        rets, fund, _ = _make_synthetic_panel()
        with pytest.raises(ValueError, match="min_months"):
            compute_residual_momentum(
                rets, fund, "2020-01-01", "2021-12-31",
                formation_start=12, formation_end=2, min_months=20,
            )

    def test_min_months_too_small_raises(self) -> None:
        rets, fund, _ = _make_synthetic_panel()
        with pytest.raises(ValueError, match="min_months"):
            compute_residual_momentum(
                rets, fund, "2020-01-01", "2021-12-31", min_months=1
            )


# ============================================================================
# Output schema
# ============================================================================


class TestOutputSchema:
    def test_columns_match_canonical(self) -> None:
        rets, fund, _ = _make_synthetic_panel()
        result = compute_residual_momentum(rets, fund, "2020-01-01", "2021-12-31")
        assert tuple(result.columns) == RESMOM_RAW_COLUMNS

    def test_empty_returns_yields_empty_with_canonical_columns(self) -> None:
        _, fund, _ = _make_synthetic_panel()
        rets = pd.DataFrame(columns=["permno", "date", "ret", "marketcap"])
        result = compute_residual_momentum(rets, fund, "2020-01-01", "2021-12-31")
        assert len(result) == 0
        assert list(result.columns) == list(RESMOM_RAW_COLUMNS)

    def test_sorted_by_date_then_permno(self) -> None:
        rets, fund, _ = _make_synthetic_panel()
        result = compute_residual_momentum(rets, fund, "2020-01-01", "2021-12-31")
        for _date, group in result.groupby("date_filed"):
            assert group["permno"].is_monotonic_increasing


# ============================================================================
# Math correctness — residual recovery
# ============================================================================


class TestMathCorrectness:
    def test_idiosyncratic_signal_recovered(self) -> None:
        """Stocks with positive idio should have positive raw_signal,
        stocks with negative idio should have negative raw_signal."""
        rets, fund, idio_map = _make_synthetic_panel()
        result = compute_residual_momentum(rets, fund, "2020-01-01", "2021-12-31")

        result["true_idio"] = result["permno"].map(idio_map)
        positive_mean = result[result["true_idio"] > 0]["raw_signal"].mean()
        negative_mean = result[result["true_idio"] < 0]["raw_signal"].mean()
        assert positive_mean > negative_mean

    def test_idiosyncratic_magnitude_approximately_correct(self) -> None:
        """Spread should approximately equal the input idio spread × window."""
        rets, fund, idio_map = _make_synthetic_panel()
        result = compute_residual_momentum(rets, fund, "2020-01-01", "2021-12-31")

        result["true_idio"] = result["permno"].map(idio_map)
        positive_mean = result[result["true_idio"] > 0]["raw_signal"].mean()
        negative_mean = result[result["true_idio"] < 0]["raw_signal"].mean()
        spread = positive_mean - negative_mean
        # Theoretical: 11 months × 0.03 spread = 0.33
        # Allow 25% tolerance for noise
        assert 0.20 < spread < 0.45

    def test_industry_effect_stripped_out(self) -> None:
        """Stocks in the same industry but different idio direction should
        differ; industry alone should not predict raw_signal."""
        rets, fund, _ = _make_synthetic_panel()
        result = compute_residual_momentum(rets, fund, "2020-01-01", "2021-12-31")

        # Mean signal by industry — should NOT differ much (residualized away)
        industry_means = result.groupby("ggroup")["raw_signal"].mean()
        # Industries should have similar means (max - min < spread of idio)
        assert industry_means.max() - industry_means.min() < 0.10

    def test_size_effect_stripped_out(self) -> None:
        """raw_signal should not be strongly correlated with size."""
        rets, fund, _ = _make_synthetic_panel()
        result = compute_residual_momentum(rets, fund, "2020-01-01", "2021-12-31")
        # Correlation between signal and size should be small
        # (true idio dominates, size is residualized out)
        corr_matrix = result[["raw_signal", "size"]].corr()
        corr_value = float(corr_matrix.loc["raw_signal", "size"])  # type: ignore[arg-type]
        assert abs(corr_value) < 0.3


# ============================================================================
# Window and observation handling
# ============================================================================


class TestWindowHandling:
    def test_default_window_emits_only_after_formation_start_months(self) -> None:
        """First output should be at least formation_start months after the
        first date in the input data."""
        rets, fund, _ = _make_synthetic_panel()
        result = compute_residual_momentum(rets, fund, "2020-01-01", "2021-12-31")
        # First date should be at least 12 months in
        first_output = result["date_filed"].min()
        assert first_output >= pd.Timestamp("2020-12-01")

    def test_n_obs_within_expected_range(self) -> None:
        """n_obs should be between min_months and window_size = 11."""
        rets, fund, _ = _make_synthetic_panel()
        result = compute_residual_momentum(rets, fund, "2020-01-01", "2021-12-31")
        # window_size = formation_start - formation_end + 1 = 12 - 2 + 1 = 11
        assert result["n_obs"].min() >= 9   # min_months default
        assert result["n_obs"].max() <= 11  # window size


# ============================================================================
# Date and dtype handling
# ============================================================================


class TestDateHandling:
    def test_iso_string_dates_accepted(self) -> None:
        rets, fund, _ = _make_synthetic_panel()
        # Should not raise
        compute_residual_momentum(rets, fund, "2020-01-01", "2021-12-31")

    def test_date_object_dates_accepted(self) -> None:
        rets, fund, _ = _make_synthetic_panel()
        # Should not raise
        compute_residual_momentum(
            rets, fund, date(2020, 1, 1), date(2021, 12, 31)
        )

    def test_handles_mixed_datetime_precision(self) -> None:
        """Real WRDS returns may have second precision."""
        rets, fund, _ = _make_synthetic_panel()
        rets["date"] = rets["date"].astype("datetime64[s]")
        # Should not raise
        result = compute_residual_momentum(rets, fund, "2020-01-01", "2021-12-31")
        assert len(result) > 0


# ============================================================================
# Edge cases
# ============================================================================


class TestEdgeCases:
    def test_too_few_months_yields_empty(self) -> None:
        """If only 6 months of data, no signal can be computed."""
        rets, fund, _ = _make_synthetic_panel(n_months=6)
        result = compute_residual_momentum(rets, fund, "2020-01-01", "2021-12-31")
        assert len(result) == 0

    def test_too_few_stocks_skips_cross_section(self) -> None:
        """If only 3 stocks per month, regression can't run cleanly."""
        rets, fund, _ = _make_synthetic_panel(n_stocks=3)
        # Should not raise; output may be empty or sparse
        result = compute_residual_momentum(rets, fund, "2020-01-01", "2021-12-31")
        # The signal handles this gracefully, so just verify no crash
        assert isinstance(result, pd.DataFrame)
