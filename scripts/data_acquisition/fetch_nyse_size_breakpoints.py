"""Fetch NYSE-median size breakpoints per rebalance date.

Standard academic methodology (CMN 2020, Fama-French, etc.):
  1. For each rebalance date, get market cap distribution of firms
     listed on NYSE (primaryexch = 'N') that are domestic common stock
     (sharetype = 'NS', securitytype = 'EQTY', securitysubtype = 'COM').
  2. Compute quintile breakpoints (20/40/60/80 percentiles) from that
     NYSE-only distribution.
  3. Apply those breakpoints to the full universe to assign size buckets.

Caches breakpoints to data/cache/nyse_size_breakpoints.parquet with
columns: date, p20, p40, p60, p80, n_nyse_firms, median.

Uses crsp.msf_v2 (monthly stock file, v2 tables current through
2025-12-31) with mthcap column directly (no need to compute
price * shrout). merge_asof pattern to look up on last trading
month at/before each rebalance date.
"""
from __future__ import annotations

from pathlib import Path
import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sqlalchemy import text
import wrds


OUTPUT = Path("data/cache/nyse_size_breakpoints.parquet")
QUANTILES = [0.20, 0.40, 0.60, 0.80]
REBALANCE_START = "2019-01-31"
REBALANCE_END = "2025-11-30"


def main() -> None:
    load_dotenv()
    db = wrds.Connection(wrds_username=os.environ["WRDS_USERNAME"])

    # Build rebalance calendar (calendar month-ends)
    rebalance_dates = pd.date_range(REBALANCE_START, REBALANCE_END, freq="ME")
    print(f"Rebalance calendar: {len(rebalance_dates)} monthly dates "
          f"({REBALANCE_START} to {REBALANCE_END})")

    print("Fetching NYSE common stock marketcap distributions from crsp.msf_v2...")
    print("  Filter: primaryexch='N' AND sharetype='NS' AND securitytype='EQTY' AND securitysubtype='COM'")

    # One big query, then compute percentiles per date in pandas
    sql = """
        SELECT mthcaldt, permno, mthcap
        FROM crsp.msf_v2
        WHERE primaryexch = 'N'
          AND sharetype = 'NS'
          AND securitytype = 'EQTY'
          AND securitysubtype = 'COM'
          AND mthcap IS NOT NULL
          AND mthcap > 0
          AND mthcaldt >= :start_date
          AND mthcaldt <= :end_date
        ORDER BY mthcaldt, permno;
    """
    try:
        with db.engine.connect() as conn:
            result = conn.execute(text(sql), {
                "start_date": REBALANCE_START,
                "end_date": REBALANCE_END,
            })
            df = pd.DataFrame(result.fetchall(), columns=list(result.keys()))
    finally:
        db.close()

    df["mthcaldt"] = pd.to_datetime(df["mthcaldt"])
    df["mthcap"] = pd.to_numeric(df["mthcap"], errors="coerce")
    print(f"  NYSE common stock rows: {len(df):,}, "
          f"unique months: {df.mthcaldt.nunique()}, "
          f"unique permnos: {df.permno.nunique()}")

    # Build a monthly index for merge_asof lookup: for each rebalance date,
    # find the NYSE distribution on the most recent CRSP monthly close at/before
    monthly_ends = sorted(df["mthcaldt"].unique())
    rebal_frame = pd.DataFrame({"rebalance_date": rebalance_dates}).sort_values("rebalance_date")
    monthly_frame = pd.DataFrame({"crsp_month": monthly_ends}).sort_values("crsp_month")

    lookup = pd.merge_asof(
        rebal_frame,
        monthly_frame,
        left_on="rebalance_date",
        right_on="crsp_month",
        direction="backward",
    )

    # Compute breakpoints per lookup row
    rows = []
    for _, row in lookup.iterrows():
        rebal = row["rebalance_date"]
        crsp_month = row["crsp_month"]
        if pd.isna(crsp_month):
            print(f"  WARN: no NYSE data at/before {rebal.date()}, skipping")
            continue
        mcaps = df.loc[df.mthcaldt == crsp_month, "mthcap"].dropna().values
        if len(mcaps) < 10:
            print(f"  WARN: only {len(mcaps)} NYSE firms on {crsp_month.date()}, skipping")
            continue
        p20, p40, p60, p80 = np.quantile(mcaps, QUANTILES)
        median = np.median(mcaps)
        rows.append({
            "date": rebal,
            "crsp_month_used": crsp_month,
            "n_nyse_firms": len(mcaps),
            "median": median,
            "p20": p20,
            "p40": p40,
            "p60": p60,
            "p80": p80,
        })

    breakpoints = pd.DataFrame(rows)
    print(f"\n  Breakpoints computed for {len(breakpoints)} rebalance dates")
    print()

    # Sample: show first, middle, and last month for sanity
    print("Sample breakpoints (mcap in millions USD):")
    for label, i in [("first", 0), ("middle", len(breakpoints) // 2), ("last", -1)]:
        r = breakpoints.iloc[i]
        print(f"  {label} ({r['date'].date()}): "
              f"NYSE n={r['n_nyse_firms']}, "
              f"median={r['median']:>10.0f}, "
              f"p20={r['p20']:>10.0f}, p40={r['p40']:>10.0f}, "
              f"p60={r['p60']:>10.0f}, p80={r['p80']:>10.0f}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    breakpoints.to_parquet(OUTPUT, index=False)
    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()