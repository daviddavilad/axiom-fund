"""Daily returns panel construction from CRSP.

Given a set of PERMNOs and a date range, returns a long-format panel of
daily returns with delisting returns properly injected. This is the
canonical return series downstream signals and the backtest will consume.

Delisting returns are the single most important correctness detail: omitting
them introduces a survivorship bias of 1-3% per year in the backtest (Shumway
1997). This module pulls delisting returns from crsp.dsedelist and appends
them to the panel on their respective dlstdt, flagged by is_delisting=True.

Usage:
    import wrds
    from axiom_fund.data.returns import ReturnsPanel

    db = wrds.Connection(wrds_username="...")
    rp = ReturnsPanel(db)
    panel = rp.fetch(
        permnos=[14593, 10107],
        start_date="2020-01-02",
        end_date="2020-06-30",
    )
    # panel columns: permno, date, ret, retx, vol, prc, prc_adj, is_delisting

    # Pivot to wide format if needed for cross-sectional operations
    wide = ReturnsPanel.to_wide(panel, values="ret")
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import date
from typing import Any, Protocol

import pandas as pd
from sqlalchemy import text


class _WrdsConnection(Protocol):
    """Structural type for a WRDS connection object.

    See universe.py for rationale. Uses engine.connect() context.
    """

    engine: Any


# Expected columns in the output panel, in canonical order.
# Downstream code can rely on this column set and order.
PANEL_COLUMNS: tuple[str, ...] = (
    "permno",
    "date",
    "ret",
    "retx",
    "vol",
    "prc",
    "prc_adj",
    "shrout",
    "marketcap",
    "is_delisting",
)


class ReturnsPanel:
    """Build a long-format daily returns panel from CRSP.

    The panel includes regular trading-day returns from crsp.dsf and
    delisting returns from crsp.dsedelist, unified into one DataFrame with
    an is_delisting boolean flag.

    Note: a PERMNO that delists mid-window may have TWO rows on its
    delisting date — one normal trading row and one delisting row. This
    reflects CRSP's data structure and is intentional. Downstream code
    that assumes (permno, date) uniqueness must filter or groupby
    appropriately (e.g., .groupby("permno")["ret"].last()).
    """

    def __init__(self, db: _WrdsConnection) -> None:
        """Initialize with an open WRDS connection.

        The class does not own the connection. The caller is responsible
        for opening and closing it.
        """
        self._db: _WrdsConnection = db

    def fetch(
        self,
        permnos: Sequence[int],
        start_date: str | date,
        end_date: str | date,
    ) -> pd.DataFrame:
        """Fetch returns panel for the given PERMNOs and date range.

        Parameters
        ----------
        permnos : sequence of int
            The CRSP PERMNOs to fetch. Must be non-empty.
        start_date, end_date : str or date
            Inclusive date range. ISO format (YYYY-MM-DD) if strings.

        Returns
        -------
        pandas.DataFrame
            Long-format panel, sorted by (permno, date). Columns match
            PANEL_COLUMNS. Rows where is_delisting=True have NaN for
            vol, prc, and prc_adj (no trading data on delisting date).

        Raises
        ------
        ValueError
            If permnos is empty, if date strings are malformed, or if
            start_date > end_date.
        """
        if len(permnos) == 0:
            raise ValueError("permnos must be non-empty")

        start_str = self._normalize_date(start_date)
        end_str = self._normalize_date(end_date)
        if start_str > end_str:
            raise ValueError(
                f"start_date ({start_str}) must be <= end_date ({end_str})"
            )

        # Validate all permnos are integers before formatting into SQL.
        # This is safe because we enforce int-only — no injection risk from
        # inline formatting of a validated tuple.
        permno_list = self._format_permno_list(permnos)

        dsf_returns = self._fetch_dsf_returns(permno_list, start_str, end_str)
        delist_returns = self._fetch_delisting_returns(
            permno_list, start_str, end_str
        )

        return self._merge_panels(dsf_returns, delist_returns)

    @staticmethod
    def to_wide(
        panel: pd.DataFrame,
        values: str = "ret",
    ) -> pd.DataFrame:
        """Pivot a long-format panel to wide format.

        Parameters
        ----------
        panel : pandas.DataFrame
            Output from ReturnsPanel.fetch().
        values : str
            Column name to pivot. Default "ret". Common alternatives:
            "retx", "prc_adj", "vol".

        Returns
        -------
        pandas.DataFrame
            Wide-format DataFrame indexed by date, columns are PERMNOs,
            values are the specified column. If a PERMNO has a duplicate
            (permno, date) pair (which happens on delisting days), the
            normal-row value is kept and the delisting row is dropped
            from the wide output.
        """
        if values not in panel.columns:
            raise ValueError(
                f"Column {values!r} not in panel. Available: {list(panel.columns)}"
            )
        # On duplicate (permno, date), prefer the normal trading row.
        # Sort so is_delisting=False comes first, then drop_duplicates
        # keeps the first (normal) row.
        deduped = (
            panel.sort_values(["permno", "date", "is_delisting"])
            .drop_duplicates(subset=["permno", "date"], keep="first")
        )
        return deduped.pivot(index="date", columns="permno", values=values)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fetch_dsf_returns(
        self,
        permno_list: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Fetch regular daily returns from crsp.dsf."""
        # NOTE: swapped from crsp.dsf to crsp.dsf_v2 (2026-07-20). Raw dsf
        # lags to 2024-12-31 at UNM; dsf_v2 runs through 2025-12-31.
        # Column renames handled via SQL aliases so downstream API unchanged:
        #   dlycaldt -> date, dlyret -> ret, dlyretx -> retx, dlyvol -> vol,
        #   dlyprc -> prc, dlycumfacpr -> cfacpr (adjusted-price factor).
        # dsf_v2 has dlycap (direct marketcap) but we recompute from
        # ABS(dlyprc) * shrout * 1000 to preserve exact numerical equivalence
        # with prior cached data.
        sql = f"""
            SELECT
                permno,
                dlycaldt AS date,
                dlyret AS ret,
                dlyretx AS retx,
                dlyvol AS vol,
                dlyprc AS prc,
                ABS(dlyprc) / NULLIF(dlycumfacpr, 0) AS prc_adj,
                shrout,
                ABS(dlyprc) * shrout * 1000 AS marketcap
            FROM crsp.dsf_v2
            WHERE permno IN {permno_list}
              AND dlycaldt >= :start_date
              AND dlycaldt <= :end_date
              AND dlyret IS NOT NULL
            ORDER BY permno, dlycaldt;
        """
        params: dict[str, str] = {"start_date": start_date, "end_date": end_date}
        # NOTE: pd.read_sql fails on SQLAlchemy 1.4 + pandas 2.3 with
        # `TypeError: Query must be a string unless using sqlalchemy`.
        # WRDS pins SQLAlchemy 1.4; pandas 2.3 requires 2.x for text() +
        # read_sql. Workaround: execute + fetchall + DataFrame(). Same
        # pattern applied in _fetch_delisting_returns. Full fix (upgrade
        # or refactor) deferred to test-cleanup session.
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
        return df

    def _fetch_delisting_returns(
        self,
        permno_list: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Fetch delisting returns from crsp.dsedelist.

        Only includes delistings within [start_date, end_date] for the
        requested PERMNOs where dlret is not null. Null dlret typically
        means the security was still active at the CRSP cutoff.
        """
        # NOTE: swapped from crsp.dsedelist to crsp.stkdelists (2026-07-20).
        # Same lag rationale as _fetch_dsf_returns swap. Column renames:
        #   dlstdt -> delistingdt, dlret -> delret. stkdelists does NOT
        # have delretx (ex-dividend delisting return), so we approximate
        # retx = delret. For delistings this is nearly always a small
        # error since ex-dividend adjustments are dominated by the
        # delisting mark-to-liquidation return.
        sql = f"""
            SELECT
                permno,
                delistingdt AS date,
                delret AS ret,
                delret AS retx
            FROM crsp.stkdelists
            WHERE permno IN {permno_list}
              AND delistingdt >= :start_date
              AND delistingdt <= :end_date
              AND delret IS NOT NULL;
        """
        params: dict[str, str] = {"start_date": start_date, "end_date": end_date}
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
        return df

    @staticmethod
    def _merge_panels(
        dsf_returns: pd.DataFrame,
        delist_returns: pd.DataFrame,
    ) -> pd.DataFrame:
        """Merge dsf returns and delisting returns into a unified panel."""
        # Normal rows from dsf
        dsf_panel = dsf_returns.copy()
        dsf_panel["date"] = pd.to_datetime(dsf_panel["date"])
        dsf_panel["is_delisting"] = False

        if len(delist_returns) == 0:
            # No delistings — just reorder columns and return.
            result = dsf_panel[list(PANEL_COLUMNS)].copy()
            return result.sort_values(["permno", "date"]).reset_index(drop=True)

        # Delisting rows: build a DataFrame with the same columns as dsf_panel
        # using explicit dtypes (avoids the pd.NA + concat FutureWarning).
        delist_panel = pd.DataFrame(
            {
                "permno": delist_returns["permno"].astype("int64"),
                "date": pd.to_datetime(delist_returns["date"]),
                "ret": delist_returns["ret"].astype("float64"),
                "retx": delist_returns["retx"].astype("float64"),
                "vol": pd.Series([pd.NA] * len(delist_returns), dtype="Float64"),
                "prc": pd.Series([pd.NA] * len(delist_returns), dtype="Float64"),
                "prc_adj": pd.Series([pd.NA] * len(delist_returns), dtype="Float64"),
                "shrout": pd.Series([pd.NA] * len(delist_returns), dtype="Float64"),
                "marketcap": pd.Series([pd.NA] * len(delist_returns), dtype="Float64"),
                "is_delisting": True,
            }
        )

        # Cast numeric columns in dsf_panel to nullable types to align
        # with delist_panel; this preserves dtype consistency on concat.
        for col in ("vol", "prc", "prc_adj", "shrout", "marketcap"):
            dsf_panel[col] = dsf_panel[col].astype("Float64")

        combined = pd.concat([dsf_panel, delist_panel], ignore_index=True)
        combined = combined[list(PANEL_COLUMNS)]
        return combined.sort_values(["permno", "date"]).reset_index(drop=True)

    @staticmethod
    def _normalize_date(d: str | date) -> str:
        """Normalize date input to ISO-format string."""
        if isinstance(d, date):
            return d.isoformat()
        if isinstance(d, str):
            parsed = date.fromisoformat(d)
            return parsed.isoformat()
        raise ValueError(f"date must be str or date, got {type(d).__name__}")

    @staticmethod
    def _format_permno_list(permnos: Iterable[int]) -> str:
        """Format a sequence of PERMNOs as a SQL IN clause.

        Enforces int-only values to prevent SQL injection from inline
        formatting. Raises TypeError on any non-int value.
        """
        permno_tuple = tuple(permnos)
        if not all(isinstance(p, int) and not isinstance(p, bool) for p in permno_tuple):
            raise TypeError("All PERMNOs must be int")
        return f"({', '.join(str(p) for p in permno_tuple)})"
