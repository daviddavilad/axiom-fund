"""CUSIP match rate diagnostic — tests the exit criterion.

Key questions:
  1. Does CRSP's ncusip (8-char) link cleanly to comp.fundq's cusip (9-char)
     via 8-char prefix matching?
  2. What's the match rate for our representative universe sample?
     If < 85%, we escalate to Sharadar per the pre-committed exit criterion.
  3. Are there one-to-many issues we need tiebreaker logic for?
"""
# ruff: noqa: I001

from __future__ import annotations

import os
import sys

from axiom_fund import _warnings  # noqa: F401

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import text


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

    # ------------------------------------------------------------------
    # 1. Verify fundq CUSIP format
    # ------------------------------------------------------------------
    print("=" * 70)
    print("1. Compustat fundq CUSIP format")
    print("=" * 70)
    sample = query(
        db,
        """
        SELECT DISTINCT gvkey, cusip, tic, conm
        FROM comp.fundq
        WHERE tic IN ('AAPL', 'MSFT', 'AMZN', 'GOOGL', 'JPM')
        LIMIT 10;
        """,
    )
    print(sample.to_string(index=False))
    if len(sample) > 0:
        print(f"\nCompustat cusip length: {sample['cusip'].str.len().iloc[0]}")
    print()

    # ------------------------------------------------------------------
    # 2. Build universe sample and attempt link
    # ------------------------------------------------------------------
    print("=" * 70)
    print("2. CUSIP match rate — THE EXIT CRITERION")
    print("=" * 70)

    universe_sample = query(
        db,
        """
        WITH latest_stock AS (
            SELECT DISTINCT ON (s.permno)
                s.permno, s.ticker, s.ncusip, s.comnam, s.shrcd, s.exchcd
            FROM crsp.stocknames s
            WHERE s.namedt <= '2020-01-02' AND s.nameenddt >= '2020-01-02'
              AND s.shrcd IN (10, 11)
              AND s.exchcd IN (1, 2, 3)
            ORDER BY s.permno, s.namedt DESC
        ),
        day_data AS (
            SELECT d.permno, d.prc, d.shrout, ABS(d.prc) * d.shrout AS market_cap
            FROM crsp.dsf d
            WHERE d.date = '2020-01-02' AND d.prc IS NOT NULL
        )
        SELECT s.permno, s.ticker, s.ncusip, s.comnam, d.market_cap
        FROM latest_stock s
        JOIN day_data d USING (permno)
        ORDER BY d.market_cap DESC
        LIMIT 1000;
        """,
    )
    print(f"Universe sample: {len(universe_sample)} names")

    # Get DISTINCT Compustat CUSIPs (fundq has one row per quarter, so distinct)
    comp_cusips = query(
        db,
        """
        SELECT DISTINCT gvkey, cusip, tic, conm
        FROM comp.fundq
        WHERE cusip IS NOT NULL;
        """,
    )
    print(f"Distinct Compustat (gvkey, cusip) pairs: {len(comp_cusips):,}")

    # 8-char CUSIP matching
    comp_cusips["cusip8"] = comp_cusips["cusip"].str[:8].str.upper()
    universe_sample["cusip8"] = universe_sample["ncusip"].str.upper()

    matched = universe_sample.merge(
        comp_cusips[["gvkey", "cusip8", "tic", "conm"]],
        on="cusip8",
        how="left",
        suffixes=("_crsp", "_comp"),
    )

    # Compute match rate — but first, account for one-to-many
    # "Matched" means at least one gvkey found
    matched_any = matched.groupby("permno")["gvkey"].apply(lambda x: x.notna().any())
    n_matched = matched_any.sum()
    n_total = len(universe_sample)
    match_rate = n_matched / n_total

    print("\nMatch results:")
    print(f"  PERMNOs with >=1 gvkey match: {n_matched} / {n_total}")
    print(f"  Match rate: {match_rate:.2%}")
    print("  Threshold: 85.00%")
    print(f"  Status: {'PASS' if match_rate >= 0.85 else 'FAIL - escalate to Sharadar'}")
    print()

    # ------------------------------------------------------------------
    # 3. One-to-many analysis
    # ------------------------------------------------------------------
    print("=" * 70)
    print("3. One-to-many: CRSP PERMNOs matching multiple gvkeys")
    print("=" * 70)
    match_counts = matched[matched["gvkey"].notna()].groupby("permno").size()
    multi = match_counts[match_counts > 1]
    print(f"PERMNOs matching multiple gvkeys: {len(multi)}")
    if len(multi) > 0:
        print("\nExamples of PERMNOs with multiple gvkey matches:")
        problematic = matched[matched["permno"].isin(multi.head(5).index)]
        print(problematic[["permno", "ticker", "ncusip", "gvkey", "tic", "conm"]].to_string(index=False))
    print()

    # ------------------------------------------------------------------
    # 4. Unmatched — what do they look like?
    # ------------------------------------------------------------------
    print("=" * 70)
    print("4. Unmatched PERMNOs — top 10 by market cap")
    print("=" * 70)
    unmatched_permnos = matched_any[~matched_any].index
    if len(unmatched_permnos) > 0:
        unmatched_df = universe_sample[universe_sample["permno"].isin(unmatched_permnos)]
        print(f"Total unmatched: {len(unmatched_df)}")
        print("\nTop 10:")
        print(unmatched_df.head(10)[["permno", "ticker", "ncusip", "comnam", "market_cap"]].to_string(index=False))
    else:
        print("All PERMNOs matched!")

    db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
