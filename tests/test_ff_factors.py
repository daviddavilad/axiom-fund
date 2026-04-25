"""Tests for the FF factors fetcher.

Unit tests use a stub connection. Integration tests (marked) hit real WRDS.
"""

from __future__ import annotations

import os
from datetime import date
from typing import Any

import pandas as pd
import pytest

from axiom_fund.data.ff_factors import (
    FF_FACTOR_COLUMNS,
    FFFactors,
    FFFactorsConfig,
)

# ============================================================================
# Stub connection for unit tests
# ============================================================================


class _StubConnection:
    """Records every SQL call and returns a canned DataFrame."""

    def __init__(self, return_df: pd.DataFrame | None = None) -> None:
        self._return_df = (
            return_df if return_df is not None
            else pd.DataFrame(columns=list(FF_FACTOR_COLUMNS))
        )
        self.calls: list[tuple[str, dict[str, Any] | None]] = []

    def raw_sql(self, sql: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
        self.calls.append((sql, params))
        return self._return_df


# ============================================================================
# Input validation
# ============================================================================


class TestInputValidation:
    def test_reversed_dates_raises(self) -> None:
        ff = FFFactors(_StubConnection())
        with pytest.raises(ValueError, match="start_date"):
            ff.fetch("2020-12-31", "2020-01-01")

    def test_iso_date_string_accepted(self) -> None:
        ff = FFFactors(_StubConnection())
        # Should not raise
        ff.fetch("2020-01-01", "2020-12-31")

    def test_date_object_accepted(self) -> None:
        ff = FFFactors(_StubConnection())
        # Should not raise
        ff.fetch(date(2020, 1, 1), date(2020, 12, 31))

    def test_malformed_string_raises(self) -> None:
        ff = FFFactors(_StubConnection())
        with pytest.raises(ValueError):
            ff.fetch("January 1, 2020", "2020-12-31")


# ============================================================================
# Output schema
# ============================================================================


class TestOutputSchema:
    def test_empty_result_has_canonical_columns(self) -> None:
        ff = FFFactors(_StubConnection())
        result = ff.fetch("2020-01-01", "2020-12-31")
        assert list(result.columns) == list(FF_FACTOR_COLUMNS)

    def test_columns_match_canonical_when_data(self) -> None:
        canned = pd.DataFrame(
            {
                "date": pd.to_datetime(["2020-01-02", "2020-01-03"]),
                "mktrf": [0.001, -0.002],
                "smb": [0.0005, -0.0003],
                "hml": [0.0001, 0.0002],
                "rf": [0.00005, 0.00005],
                "umd": [0.001, -0.001],
            }
        )
        ff = FFFactors(_StubConnection(return_df=canned))
        result = ff.fetch("2020-01-01", "2020-12-31")
        assert tuple(result.columns) == FF_FACTOR_COLUMNS
        assert len(result) == 2


# ============================================================================
# Config
# ============================================================================


class TestConfig:
    def test_default_config(self) -> None:
        cfg = FFFactorsConfig()
        assert cfg.library == "ff"
        assert cfg.table == "factors_daily"

    def test_config_is_frozen(self) -> None:
        from dataclasses import FrozenInstanceError
        cfg = FFFactorsConfig()
        with pytest.raises(FrozenInstanceError):
            cfg.library = "other" # type: ignore[misc]


# ============================================================================
# Integration tests (hit real WRDS)
# ============================================================================


@pytest.mark.integration
class TestIntegration:
    @pytest.fixture(scope="class")
    def db(self) -> Any:
        import wrds
        from dotenv import load_dotenv

        load_dotenv()
        username = os.getenv("WRDS_USERNAME")
        if not username:
            pytest.skip("WRDS_USERNAME not set")
        connection = wrds.Connection(wrds_username=username)
        yield connection
        connection.close()

    def test_fetches_real_data(self, db: Any) -> None:
        ff = FFFactors(db)
        result = ff.fetch("2020-01-01", "2020-01-31")
        assert len(result) > 0
        assert tuple(result.columns) == FF_FACTOR_COLUMNS
        # Sanity: factors are in reasonable daily-return range
        assert result["mktrf"].abs().max() < 0.30
        assert result["rf"].abs().max() < 0.01

    def test_date_range_respected(self, db: Any) -> None:
        ff = FFFactors(db)
        result = ff.fetch("2020-06-01", "2020-06-30")
        assert (result["date"] >= pd.Timestamp("2020-06-01")).all()
        assert (result["date"] <= pd.Timestamp("2020-06-30")).all()

    def test_sorted_ascending(self, db: Any) -> None:
        ff = FFFactors(db)
        result = ff.fetch("2020-01-01", "2020-03-31")
        assert result["date"].is_monotonic_increasing
