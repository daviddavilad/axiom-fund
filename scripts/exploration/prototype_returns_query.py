"""Prototype the returns + delisting query against CRSP.

Iterative script to develop and validate the SQL that returns daily returns
(with adjusted prices) merged with delisting returns for a set of PERMNOs
over a date range.

Not production code - will be replaced by src/axiom_fund/data/returns.py.
"""

from __future__ import annotations

import os
import sys

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import text

from axiom_fund import _warnings  # noqa: F401

# Test cases designed to exercise different scenarios:
#   14593 = Apple (no delisting in window, liquid)
#   82775 = Sprint (merged with T-Mobile April 2020 — but see earlier note,
#           this PERMNO might not show a delisting in our data)
#   13407 = Facebook/Meta (pre-name-change, renamed but not delisted)
#   16553 = A PERMNO that delisted in Jan 2020 per our earlier exploration
TEST_PERMNOS = [14593, 10107, 84788, 16553]
START_DATE = "2020-01-02"
END_DATE = "2020-06-30"


def query(db, sql: str, params: dict | None = None) -> pd.DataFrame:
    with db.engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params or {})


def main() -> int:
    load_dotenv()
    username = os.getenv("WRDS_USERNAME")
    if not username:
        print("ERROR: WRDS_USERNAME not set", file=sys.stderr)
        return 1

    import wrds
    db = wrds.Connection(wrds_username=username)

    permno_list = ", ".join(str(p) for p in TEST_PERMNOS)
    print(f"Fetching returns for PERMNOs: {TEST_PERMNOS}")
    print(f"Date range: {START_DATE} to {END_DATE}\n")

    # ------------------------------------------------------------------
    # Step 1: Daily returns from crsp.dsf
    # ------------------------------------------------------------------
    print("=" * 70)
    print("Step 1: Daily returns from crsp.dsf")
    print("=" * 70)
    dsf_sql = f"""
        SELECT
            permno,
            date,
            ret,
            retx,
            vol,
            prc,
            cfacpr,
            ABS(prc) / NULLIF(cfacpr, 0) AS prc_adj
        FROM crsp.dsf
        WHERE permno IN ({permno_list})
          AND date >= :start_date
          AND date <= :end_date
          AND ret IS NOT NULL
        ORDER BY permno, date;
    """
    dsf_returns = query(db, dsf_sql, {"start_date": START_DATE, "end_date": END_DATE})
    print(f"Rows: {len(dsf_returns):,}")
    print(f"PERMNOs present: {sorted(dsf_returns['permno'].unique().tolist())}")
    print("\nRows per PERMNO:")
    print(dsf_returns.groupby("permno").size().to_string())
    print("\nFirst 5 rows:")
    print(dsf_returns.head().to_string(index=False))
    print("\nLast 5 rows:")
    print(dsf_returns.tail().to_string(index=False))
    print()

    # ------------------------------------------------------------------
    # Step 2: Delisting returns from crsp.dsedelist
    # ------------------------------------------------------------------
    print("=" * 70)
    print("Step 2: Delisting returns from crsp.dsedelist")
    print("=" * 70)
    delist_sql = f"""
        SELECT
            permno,
            dlstdt AS date,
            dlret AS ret,
            dlretx AS retx,
            dlstcd
        FROM crsp.dsedelist
        WHERE permno IN ({permno_list})
          AND dlstdt >= :start_date
          AND dlstdt <= :end_date
          AND dlret IS NOT NULL;
    """
    delist_returns = query(db, delist_sql, {"start_date": START_DATE, "end_date": END_DATE})
    print(f"Rows: {len(delist_returns):,}")
    if len(delist_returns) > 0:
        print("\nDelisting returns:")
        print(delist_returns.to_string(index=False))
    else:
        print("No delistings in window for these PERMNOs")
    print()

    # ------------------------------------------------------------------
    # Step 3: Merge — normal returns + delisting returns into unified panel
    # ------------------------------------------------------------------
    print("=" * 70)
    print("Step 3: Merged panel with is_delisting flag")
    print("=" * 70)

    # Normalize column sets before concat
    dsf_panel = dsf_returns[["permno", "date", "ret", "retx", "vol", "prc", "prc_adj"]].copy()
    dsf_panel["is_delisting"] = False

    if len(delist_returns) > 0:
        delist_panel = delist_returns[["permno", "date", "ret", "retx"]].copy()
        # Delisting rows have no volume / price from dsf (security not trading)
        delist_panel["vol"] = pd.NA
        delist_panel["prc"] = pd.NA
        delist_panel["prc_adj"] = pd.NA
        delist_panel["is_delisting"] = True

        panel = pd.concat([dsf_panel, delist_panel], ignore_index=True)
    else:
        panel = dsf_panel

    panel = panel.sort_values(["permno", "date"]).reset_index(drop=True)

    print(f"Total rows: {len(panel):,}")
    print(f"  Normal rows: {(~panel['is_delisting']).sum()}")
    print(f"  Delisting rows: {panel['is_delisting'].sum()}")
    print()

    # Verify each PERMNO ends at its correct final date
    print("Last row for each PERMNO in the panel:")
    last_rows = panel.groupby("permno").tail(1)
    print(last_rows[["permno", "date", "ret", "is_delisting"]].to_string(index=False))
    print()

    # For PERMNO 16553, show its full trail including the delisting
    if 16553 in panel["permno"].unique():
        print("=" * 70)
        print("Full trail for PERMNO 16553 (delisted January 2020):")
        print("=" * 70)
        trail = panel[panel["permno"] == 16553]
        print(trail[["permno", "date", "ret", "prc_adj", "is_delisting"]].to_string(index=False))

    db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
