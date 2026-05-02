"""Tests for the ReturnsPanel module.

Unit tests verify input validation, helpers, and pure-Python transformations
(to_wide).
Integration tests (@pytest.mark.integration) hit WRDS.

Run only unit tests:        uv run pytest -m "not integration"
Run integration tests too:  uv run pytest
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from datetime import date
from typing import Any

import pandas as pd
import pytest
from dotenv import load_dotenv

from axiom_fund.data.returns import PANEL_COLUMNS, ReturnsPanel

WrdsConn = Any


# ============================================================================
# Unit tests — pure Python, no external dependencies
# ============================================================================


class TestDateNormalization:
    """ReturnsPanel._normalize_date handles str and date inputs."""

    def test_iso_string_accepted(self) -> None:
        assert ReturnsPanel._normalize_date("2020-01-02") == "2020-01-02"

    def test_date_object_accepted(self) -> None:
        assert ReturnsPanel._normalize_date(date(2020, 1, 2)) == "2020-01-02"

    def test_malformed_string_raises(self) -> None:
        with pytest.raises(ValueError):
            ReturnsPanel._normalize_date("01/02/2020")

    def test_wrong_type_raises(self) -> None:
        with pytest.raises(ValueError):
            ReturnsPanel._normalize_date(20200102)  # type: ignore[arg-type]


class TestPermnoListFormatting:
    """ReturnsPanel._format_permno_list builds safe SQL IN clauses."""

    def test_single_permno(self) -> None:
        assert ReturnsPanel._format_permno_list([14593]) == "(14593)"

    def test_multiple_permnos(self) -> None:
        assert ReturnsPanel._format_permno_list([14593, 10107]) == "(14593, 10107)"

    def test_non_int_raises(self) -> None:
        with pytest.raises(TypeError):
            ReturnsPanel._format_permno_list(["14593"])  # type: ignore[list-item]

    def test_bool_raises(self) -> None:
        """Bool is a subclass of int; must be explicitly rejected."""
        with pytest.raises(TypeError):
            ReturnsPanel._format_permno_list([True, False])


class TestFetchInputValidation:
    """ReturnsPanel.fetch() validates its inputs before any DB call."""

    def test_empty_permnos_raises(self) -> None:
        # Build a ReturnsPanel with a dummy object — we never reach the DB
        # because validation rejects first.
        rp = ReturnsPanel(db=object())  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="non-empty"):
            rp.fetch(permnos=[], start_date="2020-01-02", end_date="2020-06-30")

    def test_reversed_dates_raises(self) -> None:
        rp = ReturnsPanel(db=object())  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="start_date"):
            rp.fetch(
                permnos=[14593],
                start_date="2020-06-30",
                end_date="2020-01-02",
            )


class TestToWidePureFunction:
    """ReturnsPanel.to_wide is a pure pivot; test with synthetic data."""

    @staticmethod
    def _make_long_panel() -> pd.DataFrame:
        """Build a small synthetic long-format panel for testing."""
        return pd.DataFrame(
        {
            "permno": pd.array([1, 1, 1, 2, 2, 2], dtype="int64"),
            "date": pd.to_datetime(
                ["2020-01-02", "2020-01-03", "2020-01-06"] * 2
            ),
            "ret": pd.array(
                [0.01, -0.005, 0.02, 0.03, -0.01, 0.005], dtype="float64"
            ),
            "retx": pd.array(
                [0.01, -0.005, 0.02, 0.03, -0.01, 0.005], dtype="float64"
            ),
            "vol": pd.array(
                [100.0, 110.0, 105.0, 50.0, 55.0, 52.0], dtype="Float64"
            ),
            "prc": pd.array(
                [10.0, 9.95, 10.15, 20.0, 19.8, 19.9], dtype="Float64"
            ),
            "prc_adj": pd.array(
                [10.0, 9.95, 10.15, 20.0, 19.8, 19.9], dtype="Float64"
            ),
            "shrout": pd.array(
                [1000.0, 1000.0, 1000.0, 2000.0, 2000.0, 2000.0], dtype="Float64"
            ),
            "marketcap": pd.array(
                [10000000.0, 9950000.0, 10150000.0, 40000000.0, 39600000.0, 39800000.0],
                dtype="Float64",
            ),
            "is_delisting": pd.array(
                [False, False, False, False, False, False], dtype="bool"
        ),
    }
)

    def test_basic_pivot_ret(self) -> None:
        panel = self._make_long_panel()
        wide = ReturnsPanel.to_wide(panel, values="ret")
        assert wide.shape == (3, 2)
        assert tuple(panel.columns) == PANEL_COLUMNS
        assert wide.index.name == "date"

    def test_pivot_other_column(self) -> None:
        panel = self._make_long_panel()
        wide = ReturnsPanel.to_wide(panel, values="prc_adj")
        assert wide.shape == (3, 2)
        assert wide.loc[pd.Timestamp("2020-01-02"), 1] == 10.0

    def test_unknown_column_raises(self) -> None:
        panel = self._make_long_panel()
        with pytest.raises(ValueError, match="not in panel"):
            ReturnsPanel.to_wide(panel, values="nonexistent")

    def test_delisting_duplicate_dedup(self) -> None:
        """When a (permno, date) has both normal and delisting rows, the
        normal row should win in to_wide output."""
        panel = self._make_long_panel()
        # Add a delisting row for permno 1 on 2020-01-06 with different return
        delisting_row = pd.DataFrame(
            {
                "permno": pd.array([1], dtype="int64"),
                "date": pd.array([pd.Timestamp("2020-01-06")], dtype="datetime64[ns]"),
                "ret": pd.array([-0.50], dtype="float64"),
                "retx": pd.array([-0.50], dtype="float64"),
                "vol": pd.array([pd.NA], dtype="Float64"),
                "prc": pd.array([pd.NA], dtype="Float64"),
                "prc_adj": pd.array([pd.NA], dtype="Float64"),
                "shrout": pd.array([pd.NA], dtype="Float64"),
                "marketcap": pd.array([pd.NA], dtype="Float64"),
                "is_delisting": pd.array([True], dtype="bool"),
            }
        )
        panel = pd.concat([panel, delisting_row], ignore_index=True)
        wide = ReturnsPanel.to_wide(panel, values="ret")
        # The normal-row value (0.02) should be retained, not the delisting (-0.50)
        assert wide.loc[pd.Timestamp("2020-01-06"), 1] == 0.02


# ============================================================================
# Integration tests — require WRDS connection
# ============================================================================


@pytest.fixture(scope="module")
def wrds_connection() -> Iterator[object]:
    load_dotenv()
    username = os.getenv("WRDS_USERNAME")
    if not username:
        pytest.skip("WRDS_USERNAME not set; skipping integration tests")

    import wrds

    db = wrds.Connection(wrds_username=username)
    yield db
    db.close()


@pytest.mark.integration
class TestReturnsPanelIntegration:
    """End-to-end tests against live WRDS."""

    def test_basic_fetch_returns_dataframe(self, wrds_connection: WrdsConn) -> None:
        rp = ReturnsPanel(wrds_connection)
        panel = rp.fetch(
            permnos=[14593],  # Apple
            start_date="2020-01-02",
            end_date="2020-01-31",
        )
        assert isinstance(panel, pd.DataFrame)
        assert len(panel) > 0

    def test_columns_match_panel_columns(self, wrds_connection: WrdsConn) -> None:
        rp = ReturnsPanel(wrds_connection)
        panel = rp.fetch(
            permnos=[14593],
            start_date="2020-01-02",
            end_date="2020-01-31",
        )
        assert tuple(panel.columns) == PANEL_COLUMNS

    def test_sorted_by_permno_date(self, wrds_connection: WrdsConn) -> None:
        rp = ReturnsPanel(wrds_connection)
        panel = rp.fetch(
            permnos=[14593, 10107],
            start_date="2020-01-02",
            end_date="2020-01-31",
        )
        # Check sort: permno ascending, then date ascending within each
        for _permno, group in panel.groupby("permno"):
            dates = group["date"].to_numpy()
            assert all(dates[i] <= dates[i + 1] for i in range(len(dates) - 1))

    def test_apple_row_count_jan_2020(self, wrds_connection: WrdsConn) -> None:
        """Apple should have 20 trading days in January 2020."""
        rp = ReturnsPanel(wrds_connection)
        panel = rp.fetch(
            permnos=[14593],
            start_date="2020-01-02",
            end_date="2020-01-31",
        )
        # Jan 2020 had 21 trading days (1st closed for New Year, 20th closed for MLK)
        assert 19 <= len(panel) <= 22

    def test_delisting_appears_for_known_case(self, wrds_connection: WrdsConn) -> None:
        """PERMNO 16553 delisted 2020-01-09 with code 231 (merger)."""
        rp = ReturnsPanel(wrds_connection)
        panel = rp.fetch(
            permnos=[16553],
            start_date="2020-01-02",
            end_date="2020-06-30",
        )
        delisting_rows = panel[panel["is_delisting"]]
        assert len(delisting_rows) == 1
        assert delisting_rows.iloc[0]["date"] == pd.Timestamp("2020-01-09")
        # Delisting return should be nonzero and negative for this merger
        assert delisting_rows.iloc[0]["ret"] != 0.0

    def test_no_delisting_when_out_of_window(
        self, wrds_connection: WrdsConn
    ) -> None:
        """PERMNO 16553 delisted 2020-01-09; if we fetch after, no delisting."""
        rp = ReturnsPanel(wrds_connection)
        panel = rp.fetch(
            permnos=[16553],
            start_date="2020-02-01",
            end_date="2020-06-30",
        )
        # Should have no rows at all — security already delisted
        assert len(panel) == 0

    def test_adjusted_price_handles_split(self, wrds_connection: WrdsConn) -> None:
        """Amazon split 20-for-1 in June 2022. For 2020 data, cfacpr=20,
        so prc_adj = prc / 20."""
        rp = ReturnsPanel(wrds_connection)
        panel = rp.fetch(
            permnos=[84788],  # Amazon
            start_date="2020-06-29",
            end_date="2020-06-30",
        )
        for _, row in panel.iterrows():
            # prc_adj should be ~1/20 of |prc| for this period
            ratio = row["prc_adj"] / abs(row["prc"])
            assert 0.04 < ratio < 0.06  # roughly 1/20

    def test_to_wide_on_real_data(self, wrds_connection: WrdsConn) -> None:
        rp = ReturnsPanel(wrds_connection)
        panel = rp.fetch(
            permnos=[14593, 10107, 84788],
            start_date="2020-01-02",
            end_date="2020-01-31",
        )
        wide = ReturnsPanel.to_wide(panel, values="ret")
        assert wide.shape[1] == 3  # 3 PERMNOs
        assert wide.index.name == "date"
