"""Tests for the signal alignment layer.

Pure-function module, all tests use synthetic DataFrames.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from axiom_fund.signals.alignment import (
    ALIGNED_OUTPUT_COLUMNS,
    align_signal,
)

# ============================================================================
# Helpers
# ============================================================================


def _make_universe(
    dates: list[str], permnos_per_date: list[list[int]]
) -> pd.DataFrame:
    rows = []
    for d, permnos in zip(dates, permnos_per_date, strict=True):
        for p in permnos:
            rows.append({"date": d, "permno": p})
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _make_raw_signal(
    permnos: list[int], dates_filed: list[str], values: list[float]
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "permno": permnos,
            "date_filed": pd.to_datetime(dates_filed),
            "raw_signal": values,
        }
    )


# ============================================================================
# Input validation
# ============================================================================


class TestInputValidation:
    def test_missing_signal_columns_raises(self) -> None:
        signal = pd.DataFrame({"foo": [1]})
        uni = _make_universe(["2020-03-31"], [[1]])
        with pytest.raises(ValueError, match="raw_signal_df missing"):
            align_signal(signal, uni, ["2020-03-31"])

    def test_missing_universe_columns_raises(self) -> None:
        signal = _make_raw_signal([1], ["2020-04-30"], [0.2])
        uni = pd.DataFrame({"foo": [1]})
        with pytest.raises(ValueError, match="universe_df missing"):
            align_signal(signal, uni, ["2020-03-31"])

    def test_invalid_winsorize_raises(self) -> None:
        signal = _make_raw_signal([1], ["2020-04-30"], [0.2])
        uni = _make_universe(["2020-03-31"], [[1]])
        with pytest.raises(ValueError, match="winsorize_pct"):
            align_signal(signal, uni, ["2020-04-30"], winsorize_pct=0.6)

    def test_empty_rebalance_returns_empty(self) -> None:
        signal = _make_raw_signal([1], ["2020-04-30"], [0.2])
        uni = _make_universe(["2020-03-31"], [[1]])
        result = align_signal(signal, uni, [])
        assert len(result) == 0
        assert list(result.columns) == list(ALIGNED_OUTPUT_COLUMNS)

    def test_no_universe_for_rebalance_raises(self) -> None:
        signal = _make_raw_signal([1], ["2020-04-30"], [0.2])
        uni = _make_universe(["2020-06-30"], [[1]])
        # Rebalance is before the universe snapshot — no fallback available
        with pytest.raises(ValueError, match="No universe snapshot"):
            align_signal(signal, uni, ["2020-03-31"])


# ============================================================================
# Output schema
# ============================================================================


class TestOutputSchema:
    def test_columns_match_canonical(self) -> None:
        signal = _make_raw_signal([1, 2], ["2020-04-30"] * 2, [0.2, 0.1])
        uni = _make_universe(["2020-03-31"], [[1, 2]])
        result = align_signal(signal, uni, ["2020-06-30"])
        assert tuple(result.columns) == ALIGNED_OUTPUT_COLUMNS


# ============================================================================
# Forward-fill behavior
# ============================================================================


class TestForwardFill:
    def test_uses_most_recent_signal_per_permno(self) -> None:
        """A PERMNO with multiple signals before rebalance: use the latest."""
        signal = _make_raw_signal(
            [1, 1], ["2020-04-30", "2020-07-31"], [0.10, 0.20]
        )
        uni = _make_universe(["2020-09-30"], [[1]])
        result = align_signal(signal, uni, ["2020-09-30"])
        assert len(result) == 1
        assert result["raw_signal"].iloc[0] == 0.20

    def test_signal_after_rebalance_excluded(self) -> None:
        """A signal filed after the rebalance date is invisible (PIT)."""
        signal = _make_raw_signal(
            [1, 1], ["2020-04-30", "2020-12-31"], [0.10, 0.99]
        )
        uni = _make_universe(["2020-09-30"], [[1]])
        result = align_signal(signal, uni, ["2020-09-30"])
        assert result["raw_signal"].iloc[0] == 0.10

    def test_no_signal_yet_yields_nan(self) -> None:
        """A PERMNO in the universe but with no signal filed yet: NaN."""
        signal = _make_raw_signal([1], ["2020-12-31"], [0.5])
        uni = _make_universe(["2020-09-30"], [[1]])
        result = align_signal(signal, uni, ["2020-09-30"])
        assert pd.isna(result["raw_signal"].iloc[0])


# ============================================================================
# Universe filtering
# ============================================================================


class TestUniverseFiltering:
    def test_signal_outside_universe_dropped(self) -> None:
        """A PERMNO with a signal but not in universe: dropped from output."""
        signal = _make_raw_signal([1, 2], ["2020-04-30"] * 2, [0.2, 0.3])
        uni = _make_universe(["2020-03-31"], [[1]])  # only 1
        result = align_signal(signal, uni, ["2020-06-30"])
        assert set(result["permno"]) == {1}

    def test_universe_changes_across_rebalance_dates(self) -> None:
        """Different rebalance dates can have different universes."""
        signal = _make_raw_signal(
            [1, 2, 3, 4], ["2020-04-30"] * 4, [0.1, 0.2, 0.3, 0.4]
        )
        uni = _make_universe(
            ["2020-05-31", "2020-08-31"],
            [[1, 2], [3, 4]],
        )
        result = align_signal(
            signal, uni, ["2020-06-30", "2020-09-30"], winsorize_pct=0.0
        )
        june = result[result["date"] == pd.Timestamp("2020-06-30")]
        sept = result[result["date"] == pd.Timestamp("2020-09-30")]
        assert set(june["permno"]) == {1, 2}
        assert set(sept["permno"]) == {3, 4}


# ============================================================================
# Z-score behavior
# ============================================================================


class TestZScore:
    def test_zscore_mean_zero_std_one(self) -> None:
        """Cross-section z-scores have mean ~0, std ~1."""
        signal = _make_raw_signal(
            [1, 2, 3, 4], ["2020-04-30"] * 4, [0.1, 0.2, 0.3, 0.4]
        )
        uni = _make_universe(["2020-03-31"], [[1, 2, 3, 4]])
        result = align_signal(
            signal, uni, ["2020-06-30"], winsorize_pct=0.0
        )
        assert abs(result["z_score"].mean()) < 1e-10
        assert result["z_score"].std() == pytest.approx(1.0, abs=1e-10)

    def test_identical_values_yield_nan_zscore(self) -> None:
        """All-equal cross-section yields NaN z-score (zero std)."""
        signal = _make_raw_signal(
            [1, 2, 3], ["2020-04-30"] * 3, [0.2, 0.2, 0.2]
        )
        uni = _make_universe(["2020-03-31"], [[1, 2, 3]])
        result = align_signal(
            signal, uni, ["2020-06-30"], winsorize_pct=0.0
        )
        assert result["z_score"].isna().all()

    def test_nan_signal_propagates_through(self) -> None:
        """A PERMNO with NaN raw_signal stays NaN through winsorize+zscore."""
        signal = _make_raw_signal(
            [1, 2, 3], ["2020-04-30"] * 3, [np.nan, 0.2, 0.3]
        )
        uni = _make_universe(["2020-03-31"], [[1, 2, 3]])
        result = align_signal(
            signal, uni, ["2020-06-30"], winsorize_pct=0.0
        )
        nan_row = result[result["permno"] == 1]
        assert pd.isna(nan_row["winsorized"].iloc[0])
        assert pd.isna(nan_row["z_score"].iloc[0])


# ============================================================================
# Date handling
# ============================================================================


class TestDateHandling:
    def test_iso_string_rebalance_dates_accepted(self) -> None:
        signal = _make_raw_signal([1, 2], ["2020-04-30"] * 2, [0.2, 0.3])
        uni = _make_universe(["2020-03-31"], [[1, 2]])
        result = align_signal(signal, uni, ["2020-06-30"])
        assert len(result) == 2

    def test_date_object_rebalance_dates_accepted(self) -> None:
        signal = _make_raw_signal([1, 2], ["2020-04-30"] * 2, [0.2, 0.3])
        uni = _make_universe(["2020-03-31"], [[1, 2]])
        result = align_signal(signal, uni, [date(2020, 6, 30)])
        assert len(result) == 2

    def test_handles_mixed_datetime_precision(self) -> None:
        """Real WRDS data may have second precision; test data nanosecond."""
        signal = _make_raw_signal([1, 2], ["2020-04-30"] * 2, [0.2, 0.3])
        signal["date_filed"] = signal["date_filed"].astype("datetime64[s]")
        uni = _make_universe(["2020-03-31"], [[1, 2]])
        uni["date"] = uni["date"].astype("datetime64[s]")
        # Should not raise even with mixed precision
        result = align_signal(signal, uni, ["2020-06-30"])
        assert len(result) == 2
