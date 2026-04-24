"""Smoke test for the Fundamentals module.

Exercises all three code paths against live WRDS.
"""
# ruff: noqa: I001

from __future__ import annotations

import pandas as pd
import os
import sys

from axiom_fund import _warnings  # noqa: F401

from dotenv import load_dotenv

from axiom_fund.data.fundamentals import FUNDAMENTAL_COLUMNS, Fundamentals


def main() -> int:
    load_dotenv()
    username = os.getenv("WRDS_USERNAME")
    if not username:
        print("ERROR: WRDS_USERNAME not set", file=sys.stderr)
        return 1

    import wrds
    db = wrds.Connection(wrds_username=username)

    try:
        f = Fundamentals(db)

        # Case 1: link table
        print("=" * 70)
        print("Case 1: build_link_table for 2020-01-02")
        print("=" * 70)
        link = f.build_link_table(as_of_date="2020-01-02")
        print(f"Rows: {len(link)}")
        matched = link["gvkey"].notna().sum()
        print(f"Matched: {matched} / {len(link)} ({matched/len(link):.2%})")
        print("\nSample matched:")
        print(link[link["gvkey"].notna()].head(5)[["permno", "ticker", "ncusip", "gvkey", "tic_comp"]].to_string(index=False))
        print()

        # Case 2: fetch quarterly for a small set
        print("=" * 70)
        print("Case 2: fetch_quarterly for AAPL + MSFT, 2020-2021")
        print("=" * 70)
        fund = f.fetch_quarterly(
            permnos=[14593, 10107],  # Apple, Microsoft
            start_date="2020-01-01",
            end_date="2021-12-31",
        )
        print(f"Rows: {len(fund)}")
        print(f"Columns match canonical: {tuple(fund.columns) == FUNDAMENTAL_COLUMNS}")
        print(f"PERMNOs present: {sorted(fund['permno'].unique().tolist())}")
        print("\nFirst 3 AAPL rows:")
        aapl = fund[fund["permno"] == 14593].head(3)
        print(aapl[["permno", "datadate", "rdq", "revtq", "cogsq", "atq", "gsector"]].to_string(index=False))
        print()

        # Case 3: validate PIT correctness — no rdq after end_date
        print("=" * 70)
        print("Case 3: PIT check — no rdq after end_date (2021-12-31)")
        print("=" * 70)
        max_rdq = fund["rdq"].max()
        end_ts = pd.Timestamp("2021-12-31")
        print(f"Max rdq in panel: {max_rdq}")
        print(f"PIT correct (max <= end_date): {max_rdq <= end_ts}")
        print()

        # Case 4: empty permnos raises
        print("=" * 70)
        print("Case 4: empty permnos raises ValueError")
        print("=" * 70)
        try:
            f.fetch_quarterly(permnos=[], start_date="2020-01-01", end_date="2020-12-31")
            print("ERROR: did not raise!")
        except ValueError as e:
            print(f"Correctly raised: {e}")

    finally:
        db.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
