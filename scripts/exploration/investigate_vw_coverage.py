"""Investigate why value-weighted N=50 vs equal-weighted N=71.

Loads both backtests, identifies which months have NaN ls_return in
value-weighted but valid ls_return in equal-weighted, and diagnoses
which leg (long or short) is causing the drop by examining n_long,
n_short, and marketcap coverage on those rebalance dates.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np


BACKTEST_EW = Path("data/cache/lazy_prices_backtest")
BACKTEST_VW = Path("data/cache/lazy_prices_backtest_vw")
RETURNS_CACHE = Path("data/cache/lazy_prices_returns_daily.parquet")


def main() -> None:
    ew = pd.read_parquet(BACKTEST_EW / "monthly_returns.parquet")
    vw = pd.read_parquet(BACKTEST_VW / "monthly_returns.parquet")
    positions_vw = pd.read_parquet(BACKTEST_VW / "quintile_positions.parquet")
    returns = pd.read_parquet(RETURNS_CACHE)

    ew["date"] = pd.to_datetime(ew["date"])
    vw["date"] = pd.to_datetime(vw["date"])
    positions_vw["date"] = pd.to_datetime(positions_vw["date"])
    returns["date"] = pd.to_datetime(returns["date"])

    print(f"Equal-weighted monthly rows: {len(ew)} (valid ls: {ew.ls_return.notna().sum()})")
    print(f"Value-weighted monthly rows: {len(vw)} (valid ls: {vw.ls_return.notna().sum()})")
    print()

    # Merge on date
    merged = ew[["date", "ls_return"]].rename(columns={"ls_return": "ls_ew"}).merge(
        vw[["date", "ls_return", "n_long", "n_short"]].rename(
            columns={"ls_return": "ls_vw"}
        ),
        on="date", how="outer",
    )

    # Identify: months where EW has valid ls but VW does not
    ew_only = merged[merged.ls_ew.notna() & merged.ls_vw.isna()]
    print(f"Months valid in EW but NaN in VW: {len(ew_only)}")
    print()

    if len(ew_only) == 0:
        print("No coverage gap — value-weighted covers all EW months.")
        return

    # For those dropped months, examine the quintile positions
    print("=" * 70)
    print("Detail on dropped months (why did VW go NaN?)")
    print("=" * 70)

    # Load marketcap data restricted to rebalance dates in dropped months
    dropped_dates = ew_only["date"].tolist()
    mcap_at_rebal = returns[returns["date"].isin(dropped_dates)][
        ["date", "permno", "marketcap"]
    ]

    rows = []
    for date in sorted(dropped_dates):
        pos = positions_vw[positions_vw.date == date]
        n_q1 = len(pos[pos.quintile == 1.0])
        n_q5 = len(pos[pos.quintile == 5.0])

        mcap_this = mcap_at_rebal[mcap_at_rebal.date == date]
        q1_permnos = pos[pos.quintile == 1.0]["permno"]
        q5_permnos = pos[pos.quintile == 5.0]["permno"]
        q1_with_mcap = mcap_this[mcap_this.permno.isin(q1_permnos)]["marketcap"].notna().sum()
        q5_with_mcap = mcap_this[mcap_this.permno.isin(q5_permnos)]["marketcap"].notna().sum()

        rows.append({
            "date": date,
            "n_q1_pos": n_q1,
            "n_q1_mcap": q1_with_mcap,
            "n_q5_pos": n_q5,
            "n_q5_mcap": q5_with_mcap,
        })

    detail = pd.DataFrame(rows)
    print(detail.to_string(index=False))
    print()

    # Summary
    print("=" * 70)
    print("Summary of coverage gap causes")
    print("=" * 70)
    q1_missing = (detail.n_q1_mcap == 0).sum()
    q5_missing = (detail.n_q5_mcap == 0).sum()
    print(f"  Months with 0 Q1 firms having marketcap: {q1_missing}")
    print(f"  Months with 0 Q5 firms having marketcap: {q5_missing}")
    print()

    # Coverage rate distribution
    detail["q1_mcap_rate"] = detail.n_q1_mcap / detail.n_q1_pos.clip(lower=1)
    detail["q5_mcap_rate"] = detail.n_q5_mcap / detail.n_q5_pos.clip(lower=1)
    print(f"  Q1 mcap coverage rate: median {detail.q1_mcap_rate.median():.1%}, "
          f"min {detail.q1_mcap_rate.min():.1%}")
    print(f"  Q5 mcap coverage rate: median {detail.q5_mcap_rate.median():.1%}, "
          f"min {detail.q5_mcap_rate.min():.1%}")


if __name__ == "__main__":
    main()