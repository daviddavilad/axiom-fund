"""Unit tests for the ic_analysis module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from axiom_fund.backtest.ic_analysis import (
    aggregate_ic,
    compute_period_correlations,
    compute_period_ic,
    summarize_correlations,
)

# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

def _make_signal_panel(
    permnos: list[int],
    signals: dict[str, list[float]],
) -> pd.DataFrame:
    """Build a signal panel with given values per signal."""
    data: dict[str, list[int | float]] = {"permno": list(permnos)}
    data.update(signals)
    return pd.DataFrame(data)


def _make_forward_returns(
    permnos: list[int],
    returns: list[float],
    rebal_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Build a forward returns panel for one date."""
    if rebal_date is None:
        rebal_date = pd.Timestamp("2020-01-15")
    return pd.DataFrame({
        "rebalance_date": rebal_date,
        "permno": permnos,
        "fwd_return": returns,
        "n_days": [21] * len(permnos),
    })


# ----------------------------------------------------------------------
# compute_period_ic — math correctness
# ----------------------------------------------------------------------

class TestComputePeriodIC:
    def test_perfect_correlation_returns_ic_1(self) -> None:
        """If signal == forward return (rank-preserving), Spearman IC = 1."""
        permnos = list(range(10001, 10031))  # 30 names
        values = [float(i - 10015) for i in permnos]  # -14 to +15
        panel = _make_signal_panel(permnos, {"z_test": values})
        fwd = _make_forward_returns(permnos, values)
        result = compute_period_ic(
            panel, fwd, ["z_test"], pd.Timestamp("2020-01-15"),
        )
        assert len(result) == 1
        assert result["ic_spearman"].iloc[0] == pytest.approx(1.0, abs=1e-9)
        assert result["ic_pearson"].iloc[0] == pytest.approx(1.0, abs=1e-9)
        assert result["n_names"].iloc[0] == 30

    def test_perfect_anticorrelation_returns_ic_neg1(self) -> None:
        permnos = list(range(10001, 10031))
        signal_vals = [float(i - 10015) for i in permnos]
        return_vals = [-v for v in signal_vals]
        panel = _make_signal_panel(permnos, {"z_test": signal_vals})
        fwd = _make_forward_returns(permnos, return_vals)
        result = compute_period_ic(
            panel, fwd, ["z_test"], pd.Timestamp("2020-01-15"),
        )
        assert result["ic_spearman"].iloc[0] == pytest.approx(-1.0, abs=1e-9)

    def test_random_signal_low_ic(self) -> None:
        """Independent signal/return → IC near 0."""
        rng = np.random.default_rng(seed=42)
        permnos = list(range(10001, 10501))  # 500 names
        signal_vals = rng.standard_normal(500).tolist()
        return_vals = rng.standard_normal(500).tolist()
        panel = _make_signal_panel(permnos, {"z_random": signal_vals})
        fwd = _make_forward_returns(permnos, return_vals)
        result = compute_period_ic(
            panel, fwd, ["z_random"], pd.Timestamp("2020-01-15"),
        )
        # 500 names, independent → IC should be small (|IC| < 0.1)
        assert abs(result["ic_spearman"].iloc[0]) < 0.1

    def test_multiple_signals(self) -> None:
        permnos = list(range(10001, 10031))
        good_signal = [float(i - 10015) for i in permnos]
        bad_signal = [float(i % 3) for i in range(30)]
        fwd_vals = good_signal
        panel = _make_signal_panel(
            permnos,
            {"z_good": good_signal, "z_bad": bad_signal},
        )
        fwd = _make_forward_returns(permnos, fwd_vals)
        result = compute_period_ic(
            panel, fwd, ["z_good", "z_bad"], pd.Timestamp("2020-01-15"),
        )
        assert len(result) == 2
        good_row = result[result["signal"] == "z_good"].iloc[0]
        assert good_row["ic_spearman"] == pytest.approx(1.0, abs=1e-9)
        bad_row = result[result["signal"] == "z_bad"].iloc[0]
        # Periodic 0,1,2 pattern has very low rank correlation with a linear signal
        assert abs(bad_row["ic_spearman"]) < 0.3

    def test_insufficient_names_returns_nan(self) -> None:
        permnos = list(range(10001, 10006))  # only 5 names
        panel = _make_signal_panel(permnos, {"z_test": [1.0, 2.0, 3.0, 4.0, 5.0]})
        fwd = _make_forward_returns(permnos, [1.0, 2.0, 3.0, 4.0, 5.0])
        result = compute_period_ic(
            panel, fwd, ["z_test"], pd.Timestamp("2020-01-15"),
        )
        assert pd.isna(result["ic_spearman"].iloc[0])
        assert result["n_names"].iloc[0] == 5

    def test_missing_columns_raise(self) -> None:
        panel = pd.DataFrame({"permno": [1, 2]})
        fwd = _make_forward_returns([1, 2], [0.1, 0.2])
        with pytest.raises(ValueError, match="missing signal columns"):
            compute_period_ic(panel, fwd, ["z_missing"], pd.Timestamp("2020-01-15"))


# ----------------------------------------------------------------------
# compute_period_correlations
# ----------------------------------------------------------------------

class TestComputePeriodCorrelations:
    def test_correlation_matrix_basic(self) -> None:
        permnos = list(range(10001, 10031))
        a_vals = [float(i - 10015) for i in permnos]
        b_vals = [v * 0.5 + 1.0 for v in a_vals]  # perfectly correlated
        panel = _make_signal_panel(permnos, {"z_a": a_vals, "z_b": b_vals})
        result = compute_period_correlations(
            panel, ["z_a", "z_b"], pd.Timestamp("2020-01-15"),
        )
        assert len(result) == 1  # one pair (a, b)
        assert result["correlation"].iloc[0] == pytest.approx(1.0, abs=1e-9)

    def test_anticorrelated_pair(self) -> None:
        permnos = list(range(10001, 10031))
        a_vals = [float(i - 10015) for i in permnos]
        b_vals = [-v for v in a_vals]
        panel = _make_signal_panel(permnos, {"z_a": a_vals, "z_b": b_vals})
        result = compute_period_correlations(
            panel, ["z_a", "z_b"], pd.Timestamp("2020-01-15"),
        )
        assert result["correlation"].iloc[0] == pytest.approx(-1.0, abs=1e-9)

    def test_three_signal_pairs(self) -> None:
        permnos = list(range(10001, 10031))
        panel = _make_signal_panel(
            permnos,
            {"z_a": [1.0] * 30, "z_b": [2.0] * 30, "z_c": [3.0] * 30},
        )
        result = compute_period_correlations(
            panel, ["z_a", "z_b", "z_c"], pd.Timestamp("2020-01-15"),
        )
        assert len(result) == 3  # 3 unordered pairs
        pairs = set(zip(result["signal_a"], result["signal_b"], strict=True))
        assert pairs == {("z_a", "z_b"), ("z_a", "z_c"), ("z_b", "z_c")}


# ----------------------------------------------------------------------
# aggregate_ic
# ----------------------------------------------------------------------

class TestAggregateIC:
    def _make_ic_long(self) -> pd.DataFrame:
        """Build a long-format IC table with known statistics."""
        # 10 periods, two signals. signal_a has mean IC 0.05, signal_b has 0.00
        np.random.seed(123)
        dates = pd.date_range("2020-01-01", periods=10, freq="M")
        rows = []
        for d in dates:
            rows.append({
                "rebalance_date": d, "signal": "a",
                "ic_pearson": 0.05, "ic_spearman": 0.05, "n_names": 100,
            })
            rows.append({
                "rebalance_date": d, "signal": "b",
                "ic_pearson": 0.0, "ic_spearman": 0.0, "n_names": 100,
            })
        return pd.DataFrame(rows)

    def test_all_aggregation_mean(self) -> None:
        ic = self._make_ic_long()
        result = aggregate_ic(ic, by="all")
        # Should have 2 rows (one per signal)
        assert len(result) == 2
        a_row = result[result["signal"] == "a"].iloc[0]
        assert a_row["mean_ic_spearman"] == pytest.approx(0.05, abs=1e-9)
        assert a_row["n_periods"] == 10
        # std is 0 (all values identical) → t_stat should be NaN
        assert pd.isna(a_row["t_stat"])

    def test_by_year(self) -> None:
        ic = self._make_ic_long()
        result = aggregate_ic(ic, by="year")
        assert "year" in result.columns
        # All periods are in 2020
        assert set(result["year"].tolist()) == {2020}

    def test_hit_rate_with_mixed_signs(self) -> None:
        # Build IC values: 7 positive, 3 negative for signal "a"
        dates = pd.date_range("2020-01-01", periods=10, freq="M")
        rows = []
        ic_vals = [0.05, -0.02, 0.08, -0.03, 0.04, 0.06, 0.07, -0.01, 0.05, 0.04]
        for d, v in zip(dates, ic_vals, strict=True):
            rows.append({
                "rebalance_date": d, "signal": "a",
                "ic_pearson": v, "ic_spearman": v, "n_names": 100,
            })
        ic = pd.DataFrame(rows)
        result = aggregate_ic(ic, by="all")
        a_row = result[result["signal"] == "a"].iloc[0]
        assert a_row["hit_rate"] == pytest.approx(0.7, abs=1e-9)

    def test_missing_columns_raise(self) -> None:
        bad = pd.DataFrame({"foo": [1]})
        with pytest.raises(ValueError, match="missing columns"):
            aggregate_ic(bad)


# ----------------------------------------------------------------------
# summarize_correlations
# ----------------------------------------------------------------------

class TestSummarizeCorrelations:
    def test_basic_aggregation(self) -> None:
        # Two periods, one pair (z_a, z_b), correlations 0.5 and 0.3
        rows = [
            {
                "rebalance_date": pd.Timestamp("2020-01-31"),
                "signal_a": "z_a", "signal_b": "z_b",
                "correlation": 0.5, "n_names": 100,
            },
            {
                "rebalance_date": pd.Timestamp("2020-02-29"),
                "signal_a": "z_a", "signal_b": "z_b",
                "correlation": 0.3, "n_names": 100,
            },
        ]
        corr_long = pd.DataFrame(rows)
        mean_corr, std_corr = summarize_correlations(corr_long)
        assert mean_corr.loc["z_a", "z_b"] == pytest.approx(0.4, abs=1e-9)
        # std of [0.5, 0.3] = 0.1 * sqrt(2)/sqrt(2-1) ≈ 0.1414...
        assert std_corr.loc["z_a", "z_b"] == pytest.approx(
            np.std([0.5, 0.3], ddof=1), abs=1e-9
        )
        # Symmetric
        assert mean_corr.loc["z_b", "z_a"] == mean_corr.loc["z_a", "z_b"]
        # Diagonal
        assert mean_corr.loc["z_a", "z_a"] == 1.0
        assert std_corr.loc["z_a", "z_a"] == 0.0

    def test_missing_columns_raise(self) -> None:
        bad = pd.DataFrame({"foo": [1]})
        with pytest.raises(ValueError, match="missing columns"):
            summarize_correlations(bad)
