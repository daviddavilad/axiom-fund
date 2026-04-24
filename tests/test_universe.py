"""Tests for the Universe module.

Unit tests verify input validation and pure-Python helpers.
Integration tests (marked @pytest.mark.integration) hit WRDS and require
credentials in .env.

Run only unit tests:        uv run pytest -m "not integration"
Run integration tests too:  uv run pytest
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import FrozenInstanceError
from datetime import date
from typing import Any

import pandas as pd
import pytest
from dotenv import load_dotenv

from axiom_fund.data.universe import Universe, UniverseConfig

# Type alias for the WRDS connection parameter.
# wrds.Connection lacks type stubs; using `object` satisfies mypy without
# pretending to know the exact interface.
# We use Any here because the Protocol is internal to universe.py.
# Tests just pass the wrds connection through.
WrdsConn = Any

# ============================================================================
# Unit tests - no external dependencies
# ============================================================================


class TestUniverseConfig:
    """UniverseConfig defaults match docs/strategy_spec.md section 14."""

    def test_default_size_is_1000(self) -> None:
        assert UniverseConfig().size == 1000

    def test_default_price_floor_is_5(self) -> None:
        assert UniverseConfig().price_floor == 5.0

    def test_default_adv_floor_is_5m(self) -> None:
        assert UniverseConfig().adv_floor == 5_000_000.0

    def test_default_share_codes_are_common_stock(self) -> None:
        assert UniverseConfig().share_codes == (10, 11)

    def test_default_exchanges_are_major_us(self) -> None:
        assert UniverseConfig().exchange_codes == (1, 2, 3)

    def test_default_excludes_reits(self) -> None:
        assert 6798 in UniverseConfig().excluded_sic_codes

    def test_config_is_frozen(self) -> None:
        cfg = UniverseConfig()
        with pytest.raises(FrozenInstanceError):
            cfg.size = 500  # type: ignore[misc]

    def test_custom_config_accepts_overrides(self) -> None:
        cfg = UniverseConfig(size=500, price_floor=10.0)
        assert cfg.size == 500
        assert cfg.price_floor == 10.0
        assert cfg.adv_floor == 5_000_000.0


class TestDateNormalization:
    """Universe._normalize_date handles both str and date inputs."""

    def test_iso_string_accepted(self) -> None:
        assert Universe._normalize_date("2020-01-02") == "2020-01-02"

    def test_date_object_accepted(self) -> None:
        assert Universe._normalize_date(date(2020, 1, 2)) == "2020-01-02"

    def test_malformed_string_raises(self) -> None:
        with pytest.raises(ValueError):
            Universe._normalize_date("01/02/2020")

    def test_non_date_string_raises(self) -> None:
        with pytest.raises(ValueError):
            Universe._normalize_date("not a date")

    def test_wrong_type_raises(self) -> None:
        with pytest.raises(ValueError):
            Universe._normalize_date(20200102)  # type: ignore[arg-type]


class TestInClauseFormatting:
    """Universe._format_in_clause builds safe SQL IN clauses."""

    def test_single_value(self) -> None:
        assert Universe._format_in_clause((10,)) == "(10)"

    def test_multiple_values(self) -> None:
        assert Universe._format_in_clause((10, 11)) == "(10, 11)"

    def test_empty_tuple(self) -> None:
        assert Universe._format_in_clause(()) == "()"

    def test_non_int_raises(self) -> None:
        with pytest.raises(TypeError):
            Universe._format_in_clause(("10", "11"))  # type: ignore[arg-type]


# ============================================================================
# Integration tests - require WRDS connection
# ============================================================================


@pytest.fixture(scope="module")
def wrds_connection() -> Iterator[object]:
    """Open a WRDS connection for integration tests.

    Scoped at module level so we open the connection once per test file run,
    rather than once per test. Closed automatically after all tests complete.
    """
    load_dotenv()
    username = os.getenv("WRDS_USERNAME")
    if not username:
        pytest.skip("WRDS_USERNAME not set; skipping integration tests")

    import wrds

    db = wrds.Connection(wrds_username=username)
    yield db
    db.close()


@pytest.mark.integration
class TestUniverseAsOfIntegration:
    """End-to-end tests of Universe.as_of() against live WRDS."""

    def test_returns_dataframe(self, wrds_connection: WrdsConn) -> None:
        u = Universe(wrds_connection)
        result = u.as_of("2020-01-02")
        assert isinstance(result, pd.DataFrame)

    def test_default_returns_1000_rows(self, wrds_connection: WrdsConn) -> None:
        u = Universe(wrds_connection)
        result = u.as_of("2020-01-02")
        assert len(result) == 1000

    def test_custom_size_respected(self, wrds_connection: WrdsConn) -> None:
        u = Universe(wrds_connection, UniverseConfig(size=500))
        result = u.as_of("2020-01-02")
        assert len(result) == 500

    def test_has_required_columns(self, wrds_connection: WrdsConn) -> None:
        u = Universe(wrds_connection)
        result = u.as_of("2020-01-02")
        required = {
            "permno", "ticker", "comnam", "shrcd", "exchcd", "siccd",
            "prc", "shrout", "market_cap", "adv_20d",
        }
        assert required.issubset(set(result.columns))

    def test_sorted_by_market_cap_descending(self, wrds_connection: WrdsConn) -> None:
        u = Universe(wrds_connection)
        result = u.as_of("2020-01-02")
        mcaps = result["market_cap"].to_numpy()
        assert all(mcaps[i] >= mcaps[i + 1] for i in range(len(mcaps) - 1))

    def test_all_share_codes_are_common_stock(self, wrds_connection: WrdsConn) -> None:
        u = Universe(wrds_connection)
        result = u.as_of("2020-01-02")
        assert set(result["shrcd"].unique()).issubset({10, 11})

    def test_all_exchanges_are_major_us(self, wrds_connection: WrdsConn) -> None:
        u = Universe(wrds_connection)
        result = u.as_of("2020-01-02")
        assert set(result["exchcd"].unique()).issubset({1, 2, 3})

    def test_no_reits_included(self, wrds_connection: WrdsConn) -> None:
        u = Universe(wrds_connection)
        result = u.as_of("2020-01-02")
        assert (result["siccd"] != 6798).all()

    def test_all_prices_above_floor(self, wrds_connection: WrdsConn) -> None:
        u = Universe(wrds_connection)
        result = u.as_of("2020-01-02")
        assert (result["prc"].abs() > 5.0).all()

    def test_all_adv_above_floor(self, wrds_connection: WrdsConn) -> None:
        u = Universe(wrds_connection)
        result = u.as_of("2020-01-02")
        assert (result["adv_20d"] > 5_000_000).all()

    def test_apple_is_in_top_10_of_2020(self, wrds_connection: WrdsConn) -> None:
        """Sanity check: Apple (PERMNO 14593) was a top-10 name in early 2020."""
        u = Universe(wrds_connection)
        result = u.as_of("2020-01-02")
        assert 14593 in result.head(10)["permno"].to_numpy()

    def test_permnos_are_unique(self, wrds_connection: WrdsConn) -> None:
        u = Universe(wrds_connection)
        result = u.as_of("2020-01-02")
        assert result["permno"].nunique() == len(result)

    def test_historical_date_2015(self, wrds_connection: WrdsConn) -> None:
        """Historical date returns a different top-1 than a 2020 date."""
        u = Universe(wrds_connection)
        r2015 = u.as_of("2015-06-30")
        r2020 = u.as_of("2020-01-02")
        assert r2015.iloc[0]["market_cap"] != r2020.iloc[0]["market_cap"]
        xom_permno = 11850
        if xom_permno in r2015["permno"].to_numpy() and xom_permno in r2020["permno"].to_numpy():
            rank_2015 = r2015[r2015["permno"] == xom_permno].index[0]
            rank_2020 = r2020[r2020["permno"] == xom_permno].index[0]
            assert rank_2015 < rank_2020
