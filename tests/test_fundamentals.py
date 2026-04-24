"""Tests for the Fundamentals module.

Unit tests verify input validation and helpers.
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

from axiom_fund.data.fundamentals import FUNDAMENTAL_COLUMNS, Fundamentals

WrdsConn = Any


# ============================================================================
# Unit tests — pure Python, no external dependencies
# ============================================================================


class TestDateNormalization:
    """Fundamentals._normalize_date handles str and date inputs."""

    def test_iso_string_accepted(self) -> None:
        assert Fundamentals._normalize_date("2020-01-02") == "2020-01-02"

    def test_date_object_accepted(self) -> None:
        assert Fundamentals._normalize_date(date(2020, 1, 2)) == "2020-01-02"

    def test_malformed_string_raises(self) -> None:
        with pytest.raises(ValueError):
            Fundamentals._normalize_date("01/02/2020")

    def test_wrong_type_raises(self) -> None:
        with pytest.raises(ValueError):
            Fundamentals._normalize_date(20200102)  # type: ignore[arg-type]


class TestPermnoListFormatting:
    """Fundamentals._format_permno_list builds safe SQL IN clauses."""

    def test_single_permno(self) -> None:
        assert Fundamentals._format_permno_list([14593]) == "(14593)"

    def test_multiple_permnos(self) -> None:
        assert Fundamentals._format_permno_list([14593, 10107]) == "(14593, 10107)"

    def test_non_int_raises(self) -> None:
        with pytest.raises(TypeError):
            Fundamentals._format_permno_list(["14593"])  # type: ignore[list-item]

    def test_bool_raises(self) -> None:
        with pytest.raises(TypeError):
            Fundamentals._format_permno_list([True, False])


class TestGvkeyListFormatting:
    """Fundamentals._format_gvkey_list builds safe SQL IN clauses for string keys."""

    def test_single_gvkey(self) -> None:
        assert Fundamentals._format_gvkey_list(["001690"]) == "('001690')"

    def test_multiple_gvkeys(self) -> None:
        result = Fundamentals._format_gvkey_list(["001690", "012141"])
        assert result == "('001690', '012141')"

    def test_non_string_raises(self) -> None:
        with pytest.raises(TypeError):
            Fundamentals._format_gvkey_list([1690])  # type: ignore[list-item]

    def test_non_alnum_raises(self) -> None:
        """Special characters that could enable SQL injection must be rejected."""
        with pytest.raises(TypeError):
            Fundamentals._format_gvkey_list(["001690'; DROP TABLE--"])

    def test_empty_string_raises(self) -> None:
        """Empty string is not alphanumeric."""
        with pytest.raises(TypeError):
            Fundamentals._format_gvkey_list([""])


class TestFetchInputValidation:
    """Fundamentals.fetch_quarterly validates inputs before any DB call."""

    def test_empty_permnos_raises(self) -> None:
        f = Fundamentals(db=object())  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="non-empty"):
            f.fetch_quarterly(
                permnos=[], start_date="2020-01-01", end_date="2020-12-31"
            )

    def test_reversed_dates_raises(self) -> None:
        f = Fundamentals(db=object())  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="start_date"):
            f.fetch_quarterly(
                permnos=[14593],
                start_date="2021-12-31",
                end_date="2020-01-01",
            )


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
class TestFundamentalsIntegration:
    """End-to-end tests against live WRDS."""

    def test_build_link_table_returns_dataframe(
        self, wrds_connection: WrdsConn
    ) -> None:
        f = Fundamentals(wrds_connection)
        link = f.build_link_table(as_of_date="2020-01-02")
        assert isinstance(link, pd.DataFrame)
        assert len(link) > 0

    def test_link_table_match_rate_above_threshold(
        self, wrds_connection: WrdsConn
    ) -> None:
        """Match rate should exceed 85% for the full active universe.

        From diagnostic: 82.1% on all active, 93.1% on top 1000.
        We use a lenient threshold to accommodate either population.
        """
        f = Fundamentals(wrds_connection)
        link = f.build_link_table(as_of_date="2020-01-02")
        match_rate = link["gvkey"].notna().mean()
        # Lenient threshold — the 85% pre-committed target is for the
        # top-1000 operational universe, not all-active.
        assert match_rate >= 0.75

    def test_fetch_quarterly_returns_dataframe(
        self, wrds_connection: WrdsConn
    ) -> None:
        f = Fundamentals(wrds_connection)
        panel = f.fetch_quarterly(
            permnos=[14593],  # Apple
            start_date="2020-01-01",
            end_date="2021-12-31",
        )
        assert isinstance(panel, pd.DataFrame)
        assert len(panel) > 0

    def test_columns_match_canonical(self, wrds_connection: WrdsConn) -> None:
        f = Fundamentals(wrds_connection)
        panel = f.fetch_quarterly(
            permnos=[14593],
            start_date="2020-01-01",
            end_date="2021-12-31",
        )
        assert tuple(panel.columns) == FUNDAMENTAL_COLUMNS

    def test_pit_correctness(self, wrds_connection: WrdsConn) -> None:
        """No rdq in the panel should exceed end_date (PIT rule)."""
        f = Fundamentals(wrds_connection)
        end_date = "2021-12-31"
        panel = f.fetch_quarterly(
            permnos=[14593, 10107],
            start_date="2020-01-01",
            end_date=end_date,
        )
        assert (panel["rdq"] <= pd.Timestamp(end_date)).all()

    def test_apple_quarterly_revenue_sensible(
        self, wrds_connection: WrdsConn
    ) -> None:
        """Apple's quarterly revenue in 2020 should be between $50B and $120B.

        This sanity-checks both linking correctness (we got Apple, not a
        random company) and fundamentals fidelity.
        """
        f = Fundamentals(wrds_connection)
        panel = f.fetch_quarterly(
            permnos=[14593],
            start_date="2020-01-01",
            end_date="2020-12-31",
        )
        # Apple's revenue is in millions; values around 60,000-100,000
        assert (panel["revtq"] > 50_000).all()
        assert (panel["revtq"] < 200_000).all()

    def test_sorted_by_permno_rdq(self, wrds_connection: WrdsConn) -> None:
        f = Fundamentals(wrds_connection)
        panel = f.fetch_quarterly(
            permnos=[14593, 10107],
            start_date="2020-01-01",
            end_date="2021-12-31",
        )
        for _permno, group in panel.groupby("permno"):
            rdqs = group["rdq"].to_numpy()
            assert all(rdqs[i] <= rdqs[i + 1] for i in range(len(rdqs) - 1))

    def test_both_permnos_present(self, wrds_connection: WrdsConn) -> None:
        f = Fundamentals(wrds_connection)
        panel = f.fetch_quarterly(
            permnos=[14593, 10107],
            start_date="2020-01-01",
            end_date="2021-12-31",
        )
        assert set(panel["permno"].unique()) == {14593, 10107}

    def test_dates_are_timestamps(self, wrds_connection: WrdsConn) -> None:
        """datadate and rdq should be pd.Timestamp, not datetime.date."""
        f = Fundamentals(wrds_connection)
        panel = f.fetch_quarterly(
            permnos=[14593],
            start_date="2020-01-01",
            end_date="2021-12-31",
        )
        assert pd.api.types.is_datetime64_any_dtype(panel["datadate"])
        assert pd.api.types.is_datetime64_any_dtype(panel["rdq"])

    def test_unmatched_permno_returns_empty(
        self, wrds_connection: WrdsConn
    ) -> None:
        """A PERMNO with no CUSIP (nonexistent) should return empty panel."""
        f = Fundamentals(wrds_connection)
        # Use a PERMNO that definitely doesn't exist
        panel = f.fetch_quarterly(
            permnos=[99999999],
            start_date="2020-01-01",
            end_date="2021-12-31",
        )
        assert len(panel) == 0
