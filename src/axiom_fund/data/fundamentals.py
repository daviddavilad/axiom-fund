"""Quarterly fundamentals from Compustat, linked to CRSP via 8-char CUSIP.

Given a set of PERMNOs and a date range, returns point-in-time (PIT) quarterly
fundamental data from comp.fundq, joined with industry metadata from
comp.company. Linking is via 8-digit CUSIP prefix (see limitations.md §5.5.1
for the diagnostic; 93.1% match rate on top-1000 universe sample).

Point-in-time correctness
-------------------------
The core PIT rule: fundamental data should only be usable in a backtest after
it was publicly available. We enforce this by filtering on `rdq` (earnings
report date), not `datadate` (fiscal period end date).

- `datadate` = when the quarter closed (e.g., 2020-03-31 for Q1 2020)
- `rdq`      = when the 10-Q was filed and earnings hit the wire
               (typically 30-60 days after datadate)

A row with `rdq = 2020-05-05` is considered "available" from 2020-05-05
onward, NOT from 2020-03-31. Using `datadate` would introduce forward-
looking bias: trading on data before it was public.

Cash flow fields
----------------
Compustat quarterly stores cash flow items as YEAR-TO-DATE values with the
`y` suffix (capxy, oancfy). To get just this quarter's value, downstream
code must compute: q_value = ytd_value - previous_quarter_ytd (resetting
at fiscal year boundaries). See the `fyearq` column for fiscal year.

We keep the raw YTD fields rather than pre-computing deltas because the
delta computation is signal-dependent (some signals want YTD, some want
quarterly).

Usage
-----
    import wrds
    from axiom_fund.data.fundamentals import Fundamentals

    db = wrds.Connection(wrds_username="...")
    f = Fundamentals(db)

    # Fetch fundamentals for a set of PERMNOs over a date range
    fund_panel = f.fetch_quarterly(
        permnos=[14593, 10107],
        start_date="2015-01-01",
        end_date="2022-12-31",
    )

    # Build just the linking table (useful for caching)
    link = f.build_link_table(as_of_date="2020-01-02")
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import date
from typing import Any, Protocol

import pandas as pd
from sqlalchemy import text


class _WrdsConnection(Protocol):
    """Structural type for a WRDS connection. Same pattern as Universe/ReturnsPanel."""

    engine: Any


# The canonical column set our fundamentals panel returns.
# Order is deliberate: identifiers, anchoring dates, income statement,
# balance sheet, cash flow YTD, industry metadata.
FUNDAMENTAL_COLUMNS: tuple[str, ...] = (
    # Identifiers & links
    "permno", "gvkey", "tic", "cusip",
    # Anchoring dates
    "datadate", "rdq", "fyearq",
    # Income statement (quarterly)
    "revtq", "cogsq", "xsgaq", "oibdpq", "niq",
    # Balance sheet (quarterly)
    "atq", "ceqq", "ltq", "dlttq", "cheq", "cshoq",
    # Cash flow (year-to-date — see module docstring)
    "capxy", "oancfy",
    # Industry metadata (from comp.company, joined on gvkey)
    "conm", "sic", "gsector", "ggroup", "gind",
)


class Fundamentals:
    """Fetch quarterly fundamentals from Compustat, linked to CRSP via CUSIP.

    Two-step query design:
      1. Resolve PERMNO -> gvkey via 8-char CUSIP match (comp.fundq.cusip vs
         crsp.stocknames.ncusip)
      2. Pull fundamentals by gvkey, filtered by rdq (report date) for PIT
         correctness, joined with industry metadata from comp.company

    The link table is cacheable (changes slowly). The quarterly panel grows
    with time but is small (~20 rows/year/gvkey).
    """

    def __init__(self, db: _WrdsConnection) -> None:
        """Initialize with an open WRDS connection."""
        self._db: _WrdsConnection = db

    def build_link_table(self, as_of_date: str | date) -> pd.DataFrame:
        """Build a PERMNO -> gvkey link table for securities active on as_of_date.

        This is the linking diagnostic in method form. Useful for caching
        the link table before pulling fundamentals for many dates.

        Returns a DataFrame with columns:
            permno, ticker, ncusip, cusip8, gvkey, tic_comp, conm_comp
        Unmatched PERMNOs have NaN gvkey.
        """
        as_of_str = self._normalize_date(as_of_date)

        # Step 1: universe active on as_of_date
        crsp_sql = """
            SELECT DISTINCT ON (permno)
                permno, ticker, ncusip, comnam AS comnam
            FROM crsp.stocknames
            WHERE namedt <= :as_of AND nameenddt >= :as_of
              AND shrcd IN (10, 11)
              AND exchcd IN (1, 2, 3)
            ORDER BY permno, namedt DESC;
        """
        with self._db.engine.connect() as conn:
            crsp_df = pd.read_sql(text(crsp_sql), conn, params={"as_of": as_of_str})

        # Step 2: all (gvkey, 9-char cusip) pairs from fundq
        comp_sql = """
            SELECT DISTINCT gvkey, cusip, tic AS tic_comp, conm AS conm_comp
            FROM comp.fundq
            WHERE cusip IS NOT NULL;
        """
        with self._db.engine.connect() as conn:
            comp_df = pd.read_sql(text(comp_sql), conn)

        # Match on 8-char CUSIP prefix
        comp_df["cusip8"] = comp_df["cusip"].str[:8].str.upper()
        crsp_df["cusip8"] = crsp_df["ncusip"].str.upper()

        linked = crsp_df.merge(
            comp_df[["gvkey", "cusip8", "tic_comp", "conm_comp"]],
            on="cusip8",
            how="left",
        )
        return linked

    def fetch_quarterly(
        self,
        permnos: Sequence[int],
        start_date: str | date,
        end_date: str | date,
    ) -> pd.DataFrame:
        """Fetch quarterly fundamentals panel for the given PERMNOs and window.

        Parameters
        ----------
        permnos : sequence of int
            CRSP PERMNOs.
        start_date, end_date : str or date
            Inclusive window, filtered on `rdq` (report date, for PIT
            correctness — see module docstring).

        Returns
        -------
        pandas.DataFrame
            Long-format panel, sorted by (permno, rdq). Columns match
            FUNDAMENTAL_COLUMNS. PERMNOs that cannot be linked to a
            Compustat gvkey are silently dropped (per §5.5.1 of
            limitations.md).

        Raises
        ------
        ValueError
            If permnos is empty, start_date > end_date, or dates malformed.
        """
        if len(permnos) == 0:
            raise ValueError("permnos must be non-empty")

        start_str = self._normalize_date(start_date)
        end_str = self._normalize_date(end_date)
        if start_str > end_str:
            raise ValueError(
                f"start_date ({start_str}) must be <= end_date ({end_str})"
            )

        permno_list = self._format_permno_list(permnos)

        # Step 1: resolve PERMNO -> gvkey via CUSIP
        link_sql = f"""
            WITH crsp_side AS (
                SELECT DISTINCT ON (permno)
                    permno, ncusip
                FROM crsp.stocknames
                WHERE permno IN {permno_list}
                  AND ncusip IS NOT NULL
                ORDER BY permno, namedt DESC
            )
            SELECT
                c.permno,
                UPPER(c.ncusip) AS cusip8_crsp
            FROM crsp_side c;
        """
        with self._db.engine.connect() as conn:
            crsp_links = pd.read_sql(text(link_sql), conn)

        # Pull Compustat-side cusip8->gvkey mapping
        comp_link_sql = """
            SELECT DISTINCT
                gvkey,
                UPPER(LEFT(cusip, 8)) AS cusip8_comp
            FROM comp.fundq
            WHERE cusip IS NOT NULL;
        """
        with self._db.engine.connect() as conn:
            comp_links = pd.read_sql(text(comp_link_sql), conn)

        # Join CRSP <-> Compustat on cusip8
        link_df = crsp_links.merge(
            comp_links,
            left_on="cusip8_crsp",
            right_on="cusip8_comp",
            how="inner",  # drop unmatched PERMNOs per design
        )

        if len(link_df) == 0:
            # No PERMNOs matched — return empty DataFrame with canonical columns
            return pd.DataFrame(columns=list(FUNDAMENTAL_COLUMNS))

        # Step 2: fetch fundamentals for the linked gvkeys
        gvkey_list = self._format_gvkey_list(link_df["gvkey"].unique().tolist())

        fund_sql = f"""
            SELECT
                f.gvkey, f.tic, f.cusip,
                f.datadate, f.rdq, f.fyearq,
                f.revtq, f.cogsq, f.xsgaq, f.oibdpq, f.niq,
                f.atq, f.ceqq, f.ltq, f.dlttq, f.cheq, f.cshoq,
                f.capxy, f.oancfy,
                c.conm, c.sic, c.gsector, c.ggroup, c.gind
            FROM comp.fundq f
            LEFT JOIN comp.company c USING (gvkey)
            WHERE f.gvkey IN {gvkey_list}
              AND f.rdq IS NOT NULL
              AND f.rdq >= :start_date
              AND f.rdq <= :end_date
              AND f.indfmt = 'INDL'
              AND f.datafmt = 'STD'
              AND f.popsrc = 'D'
              AND f.consol = 'C'
            ORDER BY f.gvkey, f.rdq;
        """
        params: dict[str, str] = {"start_date": start_str, "end_date": end_str}
        with self._db.engine.connect() as conn:
            fund_df = pd.read_sql(text(fund_sql), conn, params=params)

        if len(fund_df) == 0:
            return pd.DataFrame(columns=list(FUNDAMENTAL_COLUMNS))

        # Attach permno by joining back through the link table
        result = fund_df.merge(
            link_df[["permno", "gvkey"]],
            on="gvkey",
            how="inner",
        )

        # Normalize dates to pd.Timestamp (same lesson learned from ReturnsPanel)
        result["datadate"] = pd.to_datetime(result["datadate"])
        result["rdq"] = pd.to_datetime(result["rdq"])

        # Reorder columns to canonical order and sort
        result = result[list(FUNDAMENTAL_COLUMNS)]
        return result.sort_values(["permno", "rdq"]).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Helpers (shared pattern with Universe and ReturnsPanel)
    # ------------------------------------------------------------------

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
        """Format a sequence of PERMNOs as a SQL IN clause. Int-only."""
        permno_tuple = tuple(permnos)
        if not all(isinstance(p, int) and not isinstance(p, bool) for p in permno_tuple):
            raise TypeError("All PERMNOs must be int")
        return f"({', '.join(str(p) for p in permno_tuple)})"

    @staticmethod
    def _format_gvkey_list(gvkeys: Iterable[str]) -> str:
        """Format a sequence of gvkeys as a SQL IN clause.

        gvkeys are strings in Compustat (e.g., '001690' for Apple). Each
        must be a non-empty string; we validate this to prevent injection.
        """
        gvkey_tuple = tuple(gvkeys)
        if not all(isinstance(g, str) and g.isalnum() for g in gvkey_tuple):
            raise TypeError("All gvkeys must be alphanumeric strings")
        quoted = [f"'{g}'" for g in gvkey_tuple]
        return f"({', '.join(quoted)})"
