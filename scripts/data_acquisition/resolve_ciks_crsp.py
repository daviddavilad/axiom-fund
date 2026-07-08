"""Resolve CIKs for the Lazy Prices universe via CRSP-Compustat linking.

Rationale: SEC's company_tickers.json returns *current* ticker-to-CIK mappings.
For firms with 2019-2024 corporate reorganizations (e.g. BlackRock's 2024
holding-company restructure) or acquired/delisted firms, this either returns
the wrong entity or nothing at all. See docs/v2_item6_design.md for the
BlackRock case study.

This script resolves CIKs via the CRSP -> Compustat -> comp.company path:
  1. For each PERMNO in universe, get NCUSIP from CRSP as of the firm's
     first snapshot date (when it was active in the top-1000)
  2. Match NCUSIP (8-char) to comp.fundq to get gvkey
  3. Join gvkey to comp.company to get cik

Compustat generally preserves historical entity records for acquired/
delisted firms (they retain their original CIK on the gvkey), while
firms with corporate reorganizations may have moved to new CIKs. See
BLK case study in design doc: CRSP-Compustat gives us the same
'new BlackRock' CIK (2012383) as SEC, not the historical one (1364742).
This means C-full is a partial fix, not a complete one.

Input:  data/cache/lazy_prices_universe.parquet
Output: data/cache/lazy_prices_ciks_crsp.parquet
        Columns: permno, ticker, comnam, first_snapshot_date,
                 ncusip, cusip8, gvkey, conm_comp, cik_crsp
"""
# ruff: noqa: I001

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import text

from axiom_fund import _warnings  # noqa: F401


INPUT_PATH = Path("data/cache/lazy_prices_universe.parquet")
OUTPUT_PATH = Path("data/cache/lazy_prices_ciks_crsp.parquet")


def main() -> int:
    load_dotenv()
    username = os.getenv("WRDS_USERNAME")
    if not username:
        print("ERROR: WRDS_USERNAME not set in .env", file=sys.stderr)
        return 1

    if not INPUT_PATH.exists():
        print(f"ERROR: {INPUT_PATH} not found. Run build_universe.py first.",
              file=sys.stderr)
        return 1

    universe = pd.read_parquet(INPUT_PATH)
    print(f"Universe: {len(universe):,} unique firms")
    print(f"Snapshot dates in universe: "
          f"{sorted(universe.first_snapshot_date.unique())}")
    print()

    import wrds
    db = wrds.Connection(wrds_username=username)

    try:
        # Step 1: Get NCUSIP for each PERMNO as of that firm's first_snapshot_date
        # We query per snapshot date (there are only 6) to be efficient
        print("Step 1: fetching NCUSIP from CRSP per snapshot date...")
        crsp_rows: list[pd.DataFrame] = []
        for snapshot_date in sorted(universe.first_snapshot_date.unique()):
            sub_permnos = universe[
                universe.first_snapshot_date == snapshot_date
            ].permno.tolist()
            if not sub_permnos:
                continue
            permno_list = ",".join(str(p) for p in sub_permnos)
            sql = f"""
                SELECT DISTINCT ON (permno)
                    permno, ticker, ncusip, comnam
                FROM crsp.stocknames
                WHERE permno IN ({permno_list})
                  AND namedt <= :as_of AND nameenddt >= :as_of
                  AND shrcd IN (10, 11)
                ORDER BY permno, namedt DESC
            """
            with db.engine.connect() as conn:
                r = pd.read_sql(text(sql), conn, params={"as_of": snapshot_date})
            r["first_snapshot_date"] = snapshot_date
            crsp_rows.append(r)
            print(f"  {snapshot_date}: {len(sub_permnos)} permnos requested, "
                  f"{len(r)} matched with valid NCUSIP")
        crsp_df = pd.concat(crsp_rows, ignore_index=True)
        print(f"CRSP: {len(crsp_df):,} rows with NCUSIP")
        print()

        # Step 2: Get all (gvkey, cusip8, cik) tuples from Compustat
        # via comp.fundq (has CUSIP) joined to comp.company (has CIK)
        print("Step 2: fetching gvkey -> CIK via Compustat...")
        comp_sql = """
            SELECT DISTINCT
                f.gvkey,
                UPPER(SUBSTRING(f.cusip, 1, 8)) AS cusip8,
                f.tic AS tic_comp,
                c.conm AS conm_comp,
                c.cik AS cik_crsp
            FROM comp.fundq f
            LEFT JOIN comp.company c USING (gvkey)
            WHERE f.cusip IS NOT NULL
              AND c.cik IS NOT NULL
        """
        with db.engine.connect() as conn:
            comp_df = pd.read_sql(text(comp_sql), conn)
        print(f"Compustat: {len(comp_df):,} distinct (gvkey, cusip8) pairs "
              f"with CIK")
        print()

        # Step 3: Merge CRSP NCUSIP with Compustat CUSIP8
        print("Step 3: merging CRSP NCUSIP with Compustat CUSIP8...")
        crsp_df["cusip8"] = crsp_df["ncusip"].str.upper()
        linked = crsp_df.merge(
            comp_df[["gvkey", "cusip8", "conm_comp", "cik_crsp"]],
            on="cusip8",
            how="left",
        )

        n_resolved = linked["cik_crsp"].notna().sum()
        n_total = len(linked)
        print(f"Resolved: {n_resolved:,}/{n_total:,} "
              f"({100 * n_resolved / n_total:.1f}%)")
        print()

        # Distribution: how many gvkeys per PERMNO (in case of duplicates)
        counts = linked.groupby("permno").size()
        print(f"Distribution of Compustat matches per PERMNO:")
        for n in sorted(counts.unique()):
            print(f"  {n} match(es): {(counts == n).sum():>5} PERMNOs")
        print()

        # For PERMNOs with multiple matches, we keep them all (downstream
        # can decide via disambiguation, e.g. name match or filing count)
        # Save all matches
        output_cols = [
            "permno", "ticker", "comnam", "first_snapshot_date",
            "ncusip", "cusip8", "gvkey", "conm_comp", "cik_crsp",
        ]
        linked[output_cols].to_parquet(OUTPUT_PATH, index=False)
        print(f"Saved: {OUTPUT_PATH} ({len(linked)} rows)")

        # Metadata
        metadata_path = OUTPUT_PATH.with_suffix(".txt")
        metadata_path.write_text(
            f"Run: {datetime.now().isoformat()}\n"
            f"Input: {INPUT_PATH}\n"
            f"Universe size: {len(universe)}\n"
            f"CRSP-linked rows: {len(crsp_df)}\n"
            f"Compustat pool: {len(comp_df)}\n"
            f"Merged rows: {len(linked)}\n"
            f"Resolved with CIK: {n_resolved}\n"
            f"Resolution rate: {100 * n_resolved / n_total:.2f}%\n"
        )
        print(f"Metadata: {metadata_path}")

    finally:
        db.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())