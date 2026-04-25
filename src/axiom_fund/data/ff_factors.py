"""Fama-French daily factors fetcher.

Pulls the canonical FF 3-factor data (plus momentum and risk-free rate)
from WRDS table ff.factors_daily. Used as input to signals that need
factor returns — primarily the Idiosyncratic Volatility signal.

Returns are in decimal form (0.0079 = 0.79%), matching CRSP's daily
returns. No unit conversion needed downstream.

Reference: Fama, E. and French, K. (1993). "Common risk factors in the
returns on stocks and bonds." Journal of Financial Economics, 33(1), 3-56.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Protocol

import pandas as pd

# Output columns in canonical order (date first, then factors, then rf, then umd).
FF_FACTOR_COLUMNS: tuple[str, ...] = (
    "date",
    "mktrf",
    "smb",
    "hml",
    "rf",
    "umd",
)


class _DBConnection(Protocol):
    """Protocol for a WRDS connection. Matches wrds.Connection structurally."""

    def raw_sql(self, sql: str, params: dict[str, object] | None = None) -> pd.DataFrame: ...


@dataclass(frozen=True)
class FFFactorsConfig:
    """Configuration for the FF factors fetcher.

    Defaults match the WRDS table ff.factors_daily.
    """

    library: str = "ff"
    table: str = "factors_daily"


class FFFactors:
    """Fetcher for Fama-French daily factors from WRDS.

    Pure data-layer class with dependency injection — accepts a database
    connection rather than creating one internally. Same pattern as
    ReturnsPanel and Fundamentals.
    """

    def __init__(
        self,
        db: _DBConnection,
        config: FFFactorsConfig | None = None,
    ) -> None:
        self._db = db
        self._config = config if config is not None else FFFactorsConfig()

    def fetch(
        self,
        start_date: str | date,
        end_date: str | date,
    ) -> pd.DataFrame:
        """Fetch FF factors for the given inclusive date window.

        Parameters
        ----------
        start_date, end_date : str or date
            Inclusive window. start_date must be <= end_date.

        Returns
        -------
        pandas.DataFrame
            Long-format with columns matching FF_FACTOR_COLUMNS, sorted
            by date ascending. Date column normalized to pd.Timestamp.

        Raises
        ------
        ValueError
            If start_date > end_date or dates are malformed.
        """
        start_str = _normalize_date(start_date)
        end_str = _normalize_date(end_date)
        if start_str > end_str:
            raise ValueError(
                f"start_date ({start_str}) must be <= end_date ({end_str})"
            )

        sql = f"""
            SELECT date, mktrf, smb, hml, rf, umd
            FROM {self._config.library}.{self._config.table}
            WHERE date >= CAST(%(start)s AS DATE)
              AND date <= CAST(%(end)s AS DATE)
            ORDER BY date
        """
        df = self._db.raw_sql(sql, params={"start": start_str, "end": end_str})

        if len(df) == 0:
            return pd.DataFrame(columns=list(FF_FACTOR_COLUMNS))

        df["date"] = pd.to_datetime(df["date"]).astype("datetime64[ns]")
        df = df[list(FF_FACTOR_COLUMNS)]
        return df.sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_date(d: str | date) -> str:
    """Normalize date input to ISO-format string."""
    if isinstance(d, date):
        return d.isoformat()
    if isinstance(d, str):
        parsed = date.fromisoformat(d)
        return parsed.isoformat()
    raise ValueError(f"date must be str or date, got {type(d).__name__}")
