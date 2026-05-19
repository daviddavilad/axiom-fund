"""Unit tests for the forward_returns module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from axiom_fund.backtest.forward_returns import (
    FORWARD_RETURN_COLUMNS,
    compute_forward_returns,
)

# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

def _make_returns(
    permnos: list[int],
    start_date: str = "2020-01-01",
    n_days: int = 60,
    daily_ret: float = 0.001,
) -> pd.DataFrame:
    """Build a synthetic returns panel with constant daily returns per permno."""
    dates = pd.date_range(start_date, periods=n_days, freq="B")
    rows = []
    for p in permnos:
        for d in dates:
            rows.append({"permno": p, "date": d, "ret": daily_ret})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Input validation
# ----------------------------------------------------------------------

class TestInputValidation:
    def test_missing_columns_raises(self) -> None:
        bad = pd.DataFrame({"foo": [1]})
        with pytest.raises(ValueError, match="missing columns"):
            compute_forward_returns(bad, [pd.Timestamp("2020-01-01")])

    def test_holding_days_zero_raises(self) -> None:
        rets = _make_returns([10001])
        with pytest.raises(ValueError, match="holding_days"):
            compute_forward_returns(rets, [pd.Timestamp("2020-01-15")], holding_days=0)

    def test_empty_rebalance_dates_returns_empty(self) -> None:
        rets = _make_returns([10001])
        result = compute_forward_returns(rets, [])
        assert list(result.columns) == list(FORWARD_RETURN_COLUMNS)
        assert len(result) == 0


# ----------------------------------------------------------------------
# Math correctness
# ----------------------------------------------------------------------

class TestMath:
    def test_constant_returns_compound_correctly(self) -> None:
        """For constant 0.1% daily over 21 days, compound is (1.001)^21 - 1."""
        rets = _make_returns([10001], n_days=60, daily_ret=0.001)
        result = compute_forward_returns(
            rets,
            [pd.Timestamp("2020-01-15")],
            holding_days=21,
        )
        assert len(result) == 1
        expected = (1.001) ** 21 - 1
        assert result["fwd_return"].iloc[0] == pytest.approx(expected, rel=1e-9)
        assert result["n_days"].iloc[0] == 21

    def test_partial_window_at_end_of_data(self) -> None:
        """When fewer than holding_days remain, return uses what's available."""
        rets = _make_returns([10001], n_days=10, daily_ret=0.01)
        # Rebalance early enough to have 8 days after
        rebal = pd.Timestamp("2020-01-02")
        result = compute_forward_returns(rets, [rebal], holding_days=21)
        # Only 8 trading days remain after 2020-01-02 in the 10-day fixture
        assert len(result) == 1
        assert result["n_days"].iloc[0] < 21

    def test_multiple_permnos(self) -> None:
        """All permnos should appear in output for a single rebalance date."""
        rets = _make_returns([10001, 10002, 10003], daily_ret=0.001)
        result = compute_forward_returns(
            rets,
            [pd.Timestamp("2020-01-15")],
            holding_days=21,
        )
        assert set(result["permno"].tolist()) == {10001, 10002, 10003}

    def test_multiple_rebalances(self) -> None:
        """Multiple rebalance dates produce stacked output."""
        rets = _make_returns([10001], n_days=120, daily_ret=0.001)
        result = compute_forward_returns(
            rets,
            [pd.Timestamp("2020-01-15"), pd.Timestamp("2020-03-15")],
            holding_days=21,
        )
        assert len(result) == 2
        # Both should have similar return (constant daily ret)
        expected = (1.001) ** 21 - 1
        for v in result["fwd_return"]:
            assert v == pytest.approx(expected, rel=1e-9)

    def test_no_data_after_rebalance_excludes_permno(self) -> None:
        """A permno with no returns after the rebalance date should not appear."""
        rets = _make_returns([10001], n_days=10, daily_ret=0.001)
        # Rebalance after the end of available data
        rebal = pd.Timestamp("2020-12-31")
        result = compute_forward_returns(rets, [rebal], holding_days=21)
        assert len(result) == 0


# ----------------------------------------------------------------------
# Output schema
# ----------------------------------------------------------------------

class TestOutputSchema:
    def test_columns_match_constant(self) -> None:
        rets = _make_returns([10001])
        result = compute_forward_returns(
            rets, [pd.Timestamp("2020-01-15")], holding_days=21
        )
        assert list(result.columns) == list(FORWARD_RETURN_COLUMNS)

    def test_dtypes_are_correct(self) -> None:
        rets = _make_returns([10001])
        result = compute_forward_returns(
            rets, [pd.Timestamp("2020-01-15")], holding_days=21
        )
        assert result["permno"].dtype == np.int64
        assert result["n_days"].dtype == np.int64

    def test_output_sorted(self) -> None:
        rets = _make_returns([10003, 10001, 10002], daily_ret=0.001)
        result = compute_forward_returns(
            rets,
            [pd.Timestamp("2020-03-15"), pd.Timestamp("2020-01-15")],
            holding_days=21,
        )
        # Should be sorted by (rebalance_date, permno)
        assert result["rebalance_date"].is_monotonic_increasing
