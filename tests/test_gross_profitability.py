"""Tests for the Gross Profitability raw signal module.

The module computes only the raw signal — universe filtering,
winsorization, and z-scoring all live in alignment.py and are tested
in test_alignment.py.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from axiom_fund.signals.gross_profitability import (
    GP_RAW_COLUMNS,
    compute_gross_profitability,
)

# ============================================================================
# Helpers
# ============================================================================


def _make_fundamentals(
    permnos: list[int],
    rdqs: list[str],
    datadates: list[str],
    revtqs: list[float],
    cogsqs: list[float],
    atqs: list[float],
    gvkeys: list[str] | None = None,
) -> pd.DataFrame:
    """Build a long-format fundamentals panel from parallel lists."""
    if gvkeys is None:
        gvkeys = [f"G{p:04d}" for p in permnos]
    return pd.DataFrame(
        {
            "permno": permnos,
            "gvkey": gvkeys,
            "rdq": pd.to_datetime(rdqs),
            "datadate": pd.to_datetime(datadates),
            "revtq": revtqs,
            "cogsq": cogsqs,
            "atq": atqs,
        }
    )


# ============================================================================
# Input validation
# ============================================================================


class TestInputValidation:
    def test_reversed_dates_raises(self) -> None:
        f = _make_fundamentals(
            [1], ["2020-04-30"], ["2020-03-31"],
            [100.0], [60.0], [200.0],
        )
        with pytest.raises(ValueError, match="start_date"):
            compute_gross_profitability(
                fundamentals_df=f,
                start_date="2020-12-31",
                end_date="2020-01-01",
            )

    def test_missing_columns_raises(self) -> None:
        f = pd.DataFrame({"permno": [1], "rdq": [pd.Timestamp("2020-04-30")]})
        with pytest.raises(ValueError, match="fundamentals_df missing"):
            compute_gross_profitability(
                fundamentals_df=f,
                start_date="2020-01-01",
                end_date="2020-12-31",
            )

    def test_iso_date_string_accepted(self) -> None:
        f = _make_fundamentals(
            [1], ["2020-04-30"], ["2020-03-31"],
            [100.0], [60.0], [200.0],
        )
        # Should not raise
        compute_gross_profitability(
            fundamentals_df=f,
            start_date="2020-01-01",
            end_date="2020-12-31",
        )

    def test_date_object_accepted(self) -> None:
        f = _make_fundamentals(
            [1], ["2020-04-30"], ["2020-03-31"],
            [100.0], [60.0], [200.0],
        )
        # Should not raise
        compute_gross_profitability(
            fundamentals_df=f,
            start_date=date(2020, 1, 1),
            end_date=date(2020, 12, 31),
        )


# ============================================================================
# Output schema
# ============================================================================


class TestOutputSchema:
    def test_columns_match_canonical(self) -> None:
        f = _make_fundamentals(
            [1, 2], ["2020-04-30", "2020-04-30"],
            ["2020-03-31", "2020-03-31"],
            [100.0, 50.0], [60.0, 40.0], [200.0, 200.0],
        )
        result = compute_gross_profitability(
            f, "2020-01-01", "2020-12-31"
        )
        assert tuple(result.columns) == GP_RAW_COLUMNS

    def test_empty_input_returns_empty_with_canonical_columns(self) -> None:
        f = _make_fundamentals(
            [1], ["2020-04-30"], ["2020-03-31"],
            [100.0], [60.0], [200.0],
        )
        # No rdqs in window
        result = compute_gross_profitability(
            f, "2021-01-01", "2021-12-31"
        )
        assert len(result) == 0
        assert list(result.columns) == list(GP_RAW_COLUMNS)


# ============================================================================
# Raw signal computation
# ============================================================================


class TestRawSignal:
    def test_simple_calculation(self) -> None:
        """raw_signal = (revtq - cogsq) / atq."""
        f = _make_fundamentals(
            [1], ["2020-04-30"], ["2020-03-31"],
            [100.0], [60.0], [200.0],
        )
        result = compute_gross_profitability(f, "2020-01-01", "2020-12-31")
        expected = (100.0 - 60.0) / 200.0  # 0.20
        assert len(result) == 1
        assert result["raw_signal"].iloc[0] == pytest.approx(expected)

    def test_division_by_zero_yields_nan(self) -> None:
        f = _make_fundamentals(
            [1, 2], ["2020-04-30", "2020-04-30"],
            ["2020-03-31", "2020-03-31"],
            [100.0, 100.0], [60.0, 60.0], [200.0, 0.0],
        )
        result = compute_gross_profitability(f, "2020-01-01", "2020-12-31")
        permno2_signal = result.loc[result["permno"] == 2, "raw_signal"].iloc[0]
        assert pd.isna(permno2_signal)

    def test_nan_input_yields_nan_signal(self) -> None:
        f = _make_fundamentals(
            [1], ["2020-04-30"], ["2020-03-31"],
            [np.nan], [60.0], [200.0],
        )
        result = compute_gross_profitability(f, "2020-01-01", "2020-12-31")
        assert pd.isna(result["raw_signal"].iloc[0])

    def test_negative_gp_preserved(self) -> None:
        """Firms with cogsq > revtq have negative GP — keep as-is, no clipping."""
        f = _make_fundamentals(
            [1], ["2020-04-30"], ["2020-03-31"],
            [50.0], [80.0], [200.0],  # cogsq > revtq
        )
        result = compute_gross_profitability(f, "2020-01-01", "2020-12-31")
        expected = (50.0 - 80.0) / 200.0  # -0.15
        assert result["raw_signal"].iloc[0] == pytest.approx(expected)


# ============================================================================
# Date filtering
# ============================================================================


class TestDateFiltering:
    def test_rdq_outside_window_dropped(self) -> None:
        f = _make_fundamentals(
            [1, 2, 1, 2],
            ["2020-04-30", "2020-04-30", "2020-07-31", "2020-07-31"],
            ["2020-03-31", "2020-03-31", "2020-06-30", "2020-06-30"],
            [100.0, 50.0, 110.0, 55.0],
            [60.0, 45.0, 66.0, 50.0],
            [200.0] * 4,
        )
        result = compute_gross_profitability(f, "2020-04-01", "2020-05-31")
        assert (result["date_filed"] == pd.Timestamp("2020-04-30")).all()
        assert len(result) == 2

    def test_handles_mixed_datetime_precision(self) -> None:
        """Real WRDS data may have second precision while test data is
        nanosecond. The module should normalize."""
        f = _make_fundamentals(
            [1, 2], ["2020-04-30", "2020-04-30"],
            ["2020-03-31", "2020-03-31"],
            [100.0, 50.0], [60.0, 40.0], [200.0, 200.0],
        )
        f["rdq"] = f["rdq"].astype("datetime64[s]")
        f["datadate"] = f["datadate"].astype("datetime64[s]")
        # Should not raise even with second precision
        result = compute_gross_profitability(f, "2020-01-01", "2020-12-31")
        assert len(result) == 2


# ============================================================================
# Sort order
# ============================================================================


class TestSortOrder:
    def test_sorted_by_date_filed_then_permno(self) -> None:
        f = _make_fundamentals(
            [2, 1, 1, 2],
            ["2020-07-31", "2020-04-30", "2020-07-31", "2020-04-30"],
            ["2020-06-30", "2020-03-31", "2020-06-30", "2020-03-31"],
            [50.0, 100.0, 110.0, 55.0],
            [40.0, 60.0, 66.0, 45.0],
            [200.0] * 4,
        )
        result = compute_gross_profitability(f, "2020-01-01", "2020-12-31")
        # Expected order: (2020-04-30, 1), (2020-04-30, 2), (2020-07-31, 1), (2020-07-31, 2)
        expected_pairs = [
            (pd.Timestamp("2020-04-30"), 1),
            (pd.Timestamp("2020-04-30"), 2),
            (pd.Timestamp("2020-07-31"), 1),
            (pd.Timestamp("2020-07-31"), 2),
        ]
        actual_pairs = list(zip(result["date_filed"], result["permno"], strict=True))
        assert actual_pairs == expected_pairs
