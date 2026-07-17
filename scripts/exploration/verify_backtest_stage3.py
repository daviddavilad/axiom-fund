"""Stage 3 backtest verification: 2021 Q5 breakdown.

Which firms sat in Q5 during 2021 and how did they perform?
Hypothesis: SPAC-merger firms with huge YoY text change 2020->2021
dominated Q5 and crashed during the mid-2021 SPAC bust.

Prints:
  1. Top 20 Q5-2021 firms by average forward return (worst first)
  2. Top 20 Q5-2021 firms by count-of-months-in-Q5 (persistence)
  3. Full sector breakdown if SIC available
  4. Per-month Q5 mean and drag firms per month
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from axiom_fund.backtest.forward_returns import compute_forward_returns


CACHE = Path("data/cache")
BACKTEST = CACHE / "lazy_prices_backtest"


def main() -> None:
    positions = pd.read_parquet(BACKTEST / "quintile_positions.parquet")
    signal = pd.read_parquet(CACHE / "lazy_prices_signal.parquet")
    returns = pd.read_parquet(CACHE / "lazy_prices_returns_daily.parquet")
    universe = pd.read_parquet(CACHE / "lazy_prices_universe.parquet")

    # Filter to Q5 in 2021
    q5_2021 = positions[
        (positions.quintile == 5.0)
        & (positions.date >= "2021-01-01")
        & (positions.date <= "2021-12-31")
    ].copy()
    print(f"Q5 firm-months in 2021: {len(q5_2021)}")
    print(f"Unique Q5 firms in 2021: {q5_2021.permno.nunique()}")
    print()

    # Compute forward returns for those dates
    rebalance_dates = sorted(q5_2021.date.unique())
    fwd = compute_forward_returns(
        returns_df=returns,
        rebalance_dates=list(pd.to_datetime(rebalance_dates)),
        holding_days=21,
    )
    fwd = fwd.rename(columns={"rebalance_date": "date"})

    # Merge Q5-2021 positions with forward returns
    q5_with_fwd = q5_2021.merge(fwd, on=["date", "permno"], how="inner")
    print(f"Q5-2021 firm-months with forward return data: {len(q5_with_fwd)}")

    # Enrich with ticker + comnam + sic from universe file, and signal
    ticker_map = signal[["permno", "ticker"]].drop_duplicates("permno")
    q5_with_fwd = q5_with_fwd.merge(ticker_map, on="permno", how="left")
    q5_with_fwd = q5_with_fwd.merge(
        universe[["permno", "comnam", "siccd"]].drop_duplicates("permno"),
        on="permno",
        how="left",
    )

    # -------------------------------------------------------------------
    # 1. Worst Q5-2021 firms by average forward return
    # -------------------------------------------------------------------
    print()
    print("=" * 80)
    print("Bottom 20 Q5-2021 firms by average 21-day forward return")
    print("=" * 80)
    by_firm = (
        q5_with_fwd.groupby(["permno", "ticker", "comnam"])
        .agg(
            n_months=("fwd_return", "size"),
            avg_fwd_ret=("fwd_return", "mean"),
            cum_contribution=("fwd_return", "sum"),
        )
        .sort_values("avg_fwd_ret")
        .head(20)
    )
    print(by_firm.to_string())

    # -------------------------------------------------------------------
    # 2. Q5-2021 firms with highest persistence (months in Q5)
    # -------------------------------------------------------------------
    print()
    print("=" * 80)
    print("Top 20 Q5-2021 firms by persistence (months in Q5) with their avg return")
    print("=" * 80)
    persistence = (
        q5_with_fwd.groupby(["permno", "ticker", "comnam"])
        .agg(
            n_months=("fwd_return", "size"),
            avg_fwd_ret=("fwd_return", "mean"),
        )
        .sort_values("n_months", ascending=False)
        .head(20)
    )
    print(persistence.to_string())

    # -------------------------------------------------------------------
    # 3. SIC 2-digit sector concentration in Q5 2021
    # -------------------------------------------------------------------
    print()
    print("=" * 80)
    print("Q5-2021 firm-months by SIC 2-digit (top 15 sectors)")
    print("=" * 80)
    q5_with_fwd["sic2"] = q5_with_fwd["siccd"].astype(str).str[:2]
    sic_summary = (
        q5_with_fwd.groupby("sic2")
        .agg(
            n_firm_months=("permno", "size"),
            n_unique_firms=("permno", "nunique"),
            avg_fwd_ret=("fwd_return", "mean"),
        )
        .sort_values("n_firm_months", ascending=False)
        .head(15)
    )
    print(sic_summary.to_string())

    # -------------------------------------------------------------------
    # 4. Per-month Q5 mean forward return + top 3 drag firms per month
    # -------------------------------------------------------------------
    print()
    print("=" * 80)
    print("2021 Q5 monthly performance + top-3 drag firms per month")
    print("=" * 80)
    for date, group in q5_with_fwd.groupby("date"):
        month_mean = group.fwd_return.mean()
        top_drags = group.nsmallest(3, "fwd_return")[
            ["ticker", "comnam", "fwd_return"]
        ]
        print(f"\n{date.date()}  Q5 mean fwd return: {month_mean:+.4f}  "
              f"N={len(group)}")
        for _, row in top_drags.iterrows():
            print(f"    {row.ticker:6s} {row.comnam[:40]:40s} "
                  f"{row.fwd_return:+.4f}")


if __name__ == "__main__":
    main()