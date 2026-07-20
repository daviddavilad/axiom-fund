"""Rules-based universe construction from CRSP.

Given a rebalance date, returns the top-N U.S. common stocks by market
capitalization that pass the eligibility filters defined in docs/strategy_spec.md
section 3.

The universe is intentionally defined by a transparent rule rather than
committee-selected index membership. See docs/strategy_spec.md for rationale.

Usage:
    import wrds
    from axiom_fund.data.universe import Universe

    db = wrds.Connection(wrds_username="...")
    u = Universe(db)
    members = u.as_of("2020-01-02")
    # members is a pandas DataFrame with columns:
    #   permno, ticker, comnam, shrcd, exchcd, siccd,
    #   prc, shrout, market_cap, adv_20d
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Protocol

import pandas as pd
from sqlalchemy import text


class _WrdsConnection(Protocol):
    """Structural type for a WRDS connection object.

    We use a Protocol rather than importing wrds.Connection because the wrds
    library lacks type stubs. This Protocol captures only the attributes we
    actually use, making the dependency explicit and the class testable with
    mocks.
    """

    engine: Any  # SQLAlchemy engine; 'Any' because SQLAlchemy 1.4 lacks strict types here

@dataclass(frozen=True)
class UniverseConfig:
    """Configuration for rules-based universe construction.

    All parameters have defaults matching docs/strategy_spec.md section 14.
    These defaults are locked; changing them should be accompanied by
    a dated amendment in docs/strategy_spec.md.
    """

    size: int = 1000
    price_floor: float = 5.0
    adv_floor: float = 5_000_000.0
    adv_lookback_days: int = 40  # calendar days, gives ~20 trading days
    share_codes: tuple[int, ...] = (10, 11)
    exchange_codes: tuple[int, ...] = (1, 2, 3)
    excluded_sic_codes: tuple[int, ...] = (6798,)  # REITs


class Universe:
    """Construct the rules-based investable universe from CRSP.

    Takes a WRDS connection (or anything with a .engine attribute exposing
    a SQLAlchemy engine). Does not own the connection; caller is responsible
    for opening and closing it.

    The class is deliberately simple in this version: one method, one date.
    Bulk construction across many rebalance dates will be layered on top
    once the single-date path is verified.
    """

    def __init__(self, db: _WrdsConnection, config: UniverseConfig | None = None) -> None:
        """Initialize.

        Parameters
        ----------
        db : wrds.Connection
            An open WRDS connection. Must expose `.engine` (a SQLAlchemy engine).
        config : UniverseConfig, optional
            Universe construction parameters. Defaults to spec-locked values.
        """
        self._db: _WrdsConnection = db
        self._config = config or UniverseConfig()

    @property
    def config(self) -> UniverseConfig:
        """Return the configuration used by this Universe instance."""
        return self._config

    def as_of(self, rank_date: str | date) -> pd.DataFrame:
        """Return the universe as of a given date.

        Parameters
        ----------
        rank_date : str or date
            The date on which to construct the universe. Strings must be
            ISO format (YYYY-MM-DD).

        Returns
        -------
        pandas.DataFrame
            One row per universe member, sorted by market cap descending.
            Columns: permno, ticker, comnam, shrcd, exchcd, siccd,
                     prc, shrout, market_cap, adv_20d.
            Expected length: equal to config.size (1000 by default), though
            may be smaller on very early dates when fewer names are listed.

        Raises
        ------
        ValueError
            If rank_date is not a valid date string or date object.
        """
        rank_date_str = self._normalize_date(rank_date)
        sic_placeholder = self._format_in_clause(self._config.excluded_sic_codes)
        share_placeholder = self._format_in_clause(self._config.share_codes)
        exch_placeholder = self._format_in_clause(self._config.exchange_codes)

        sql = f"""
            WITH adv AS (
                SELECT
                    permno,
                    AVG(ABS(prc) * vol) AS adv_20d
                FROM crsp.dsf
                WHERE date <= :rank_date
                  AND date >= (CAST(:rank_date AS DATE) - INTERVAL '{self._config.adv_lookback_days} days')
                  AND prc IS NOT NULL
                  AND vol IS NOT NULL
                GROUP BY permno
            ),
            eligible AS (
                SELECT
                    sn.permno,
                    sn.shrcd,
                    sn.exchcd,
                    sn.siccd,
                    sn.ticker,
                    sn.comnam,
                    dsf.prc,
                    dsf.shrout,
                    ABS(dsf.prc) * dsf.shrout * 1000 AS market_cap,
                    adv.adv_20d
                FROM crsp.stocknames sn
                INNER JOIN crsp.dsf dsf
                    ON dsf.permno = sn.permno
                    AND dsf.date = :rank_date
                INNER JOIN adv
                    ON adv.permno = sn.permno
                WHERE sn.namedt <= :rank_date AND sn.nameenddt >= :rank_date
                  AND sn.shrcd IN {share_placeholder}
                  AND sn.exchcd IN {exch_placeholder}
                  AND sn.siccd NOT IN {sic_placeholder}
                  AND ABS(dsf.prc) > :price_floor
                  AND dsf.shrout IS NOT NULL
                  AND adv.adv_20d > :adv_floor
            )
            SELECT *
            FROM eligible
            ORDER BY market_cap DESC
            LIMIT :size;
        """

        params: dict[str, str | float | int] = {
            "rank_date": rank_date_str,
            "price_floor": self._config.price_floor,
            "adv_floor": self._config.adv_floor,
            "size": self._config.size,
        }

        # NOTE: pd.read_sql fails on SQLAlchemy 1.4 + pandas 2.3 with
        # `TypeError: Query must be a string unless using sqlalchemy`.
        # WRDS pins SQLAlchemy 1.4; pandas 2.3 requires 2.x for text() +
        # read_sql. Workaround: execute + fetchall + DataFrame(). Same
        # pattern applied in returns.py commit cc4b78c.
        with self._db.engine.connect() as conn:
            result = conn.execute(text(sql), params)
            df = pd.DataFrame(result.fetchall(), columns=list(result.keys()))
            # Coerce Decimal columns to float (fetchall returns Decimal for
            # numeric SQL types; pd.read_sql would have coerced these).
            for col in df.select_dtypes(include="object").columns:
                if col == "permno":
                    continue
                try:
                    df[col] = pd.to_numeric(df[col], errors="raise")
                except (ValueError, TypeError):
                    pass

        return df.reset_index(drop=True)

    @staticmethod
    def _normalize_date(d: str | date) -> str:
        """Normalize date input to ISO-format string."""
        if isinstance(d, date):
            return d.isoformat()
        if isinstance(d, str):
            # Basic validation: will raise if malformed
            parsed = date.fromisoformat(d)
            return parsed.isoformat()
        raise ValueError(f"rank_date must be str or date, got {type(d).__name__}")

    @staticmethod
    def _format_in_clause(values: tuple[int, ...]) -> str:
        """Format a tuple of ints into a SQL IN clause like '(10, 11)'.

        This exists because SQLAlchemy's text() with named parameters does
        not support binding a variable-length IN clause directly. Since our
        share/exchange/SIC codes come from configuration (not user input),
        it is safe to format them inline. Integer-only values are enforced
        by the type signature.
        """
        # Defensive: enforce int-only to avoid any injection risk
        if not all(isinstance(v, int) for v in values):
            raise TypeError("IN-clause values must be int")
        return f"({', '.join(str(v) for v in values)})"
