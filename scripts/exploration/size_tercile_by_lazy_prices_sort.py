"""3x5 sort: Size tercile (Small/Mid/Large) x Lazy Prices quintile.

Refines the 5x2 sort by splitting size into three buckets. Reveals
whether the CMN effect is:
  - Concentrated in a specific size range (e.g. Mid-caps)
  - Monotonic across size (stronger as size increases or decreases)
  - Reversed at both extremes with a peak in the middle

Prompted by finding on 2026-07-22 that 5x2 sort shows Big L/S = +4.24%
ann with Sharpe +0.675 (CMN direction) while value-weighted L/S is
-3.48% ann (opposite of CMN). This suggests value-weighting concentrates
too much on the very top of the "Big" bucket where the signal reverses.
Tercile split isolates whether "Large" (top 33%) shows this reversal.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np


BACKTEST_DIR = Path("data/cache/lazy_prices_backtest")
RETURNS_CACHE = Path("data/cache/lazy_prices_returns_daily.parquet")
SIGNAL_CACHE = Path("data/cache/lazy_prices_signal.parquet")
HOLDING_DAYS = 21


def main() -> None:
    print("Loading data...")
    positions = pd.read_parquet(BACKTEST_DIR / "quintile_positions.parquet")
    returns = pd.read_parquet(RETURNS_CACHE)
    signal = pd.read_parquet(SIGNAL_CACHE)

    positions["date"] = pd.to_datetime(positions["date"])
    returns["date"] = pd.to_datetime(returns["date"])
    signal["date_filed"] = pd.to_datetime(signal["date_filed"])

    rebal_dates = sorted(positions["date"].unique())
    print(f"  {len(rebal_dates)} rebalance dates")

    # Marketcap lookup at rebalance dates (merge_asof pattern)
    print("Building marketcap lookup at rebalance dates...")
    rebal_frame = pd.DataFrame({"date": pd.to_datetime(rebal_dates)}).sort_values("date")
    mcap = returns[["date", "permno", "marketcap"]].dropna(subset=["marketcap"])
    mcap_rows = []
    for permno, group in mcap.groupby("permno", sort=False):
        merged = pd.merge_asof(
            rebal_frame, group[["date", "marketcap"]].sort_values("date"),
            on="date", direction="backward",
        )
        merged["permno"] = permno
        mcap_rows.append(merged.dropna(subset=["marketcap"]))
    mcap_at_rebal = pd.concat(mcap_rows, ignore_index=True)
    print(f"  {len(mcap_at_rebal):,} rows")

    # Raw signal aligned to rebalance dates
    print("Aligning raw_signal to rebalance dates...")
    signal_min = signal[["permno", "date_filed", "raw_signal"]].sort_values(
        ["permno", "date_filed"]
    )
    aligned_rows = []
    for rebal_date in rebal_dates:
        eligible = signal_min[signal_min.date_filed <= rebal_date]
        latest = (
            eligible.sort_values(["permno", "date_filed"])
            .groupby("permno", as_index=False)
            .last()
        )
        latest["date"] = rebal_date
        aligned_rows.append(latest[["date", "permno", "raw_signal"]])
    aligned_signal = pd.concat(aligned_rows, ignore_index=True)

    joined = aligned_signal.merge(
        mcap_at_rebal, on=["date", "permno"], how="inner"
    )
    print(f"  {len(joined):,} rows after joining")

    # Assign size tercile per date (Small/Mid/Large by 33/67 pctiles)
    print("Assigning size terciles per date...")
    joined["size_bucket"] = ""
    for date_val, group in joined.groupby("date"):
        p33 = group["marketcap"].quantile(1/3)
        p67 = group["marketcap"].quantile(2/3)
        conditions = [
            group["marketcap"] < p33,
            (group["marketcap"] >= p33) & (group["marketcap"] < p67),
            group["marketcap"] >= p67,
        ]
        choices = ["Small", "Mid", "Large"]
        joined.loc[group.index, "size_bucket"] = np.select(
            conditions, choices, default=""
        )

    # Assign 5 LP quintiles within each (date, size_bucket)
    print("Assigning LP quintiles within each size tercile...")
    joined["lp_quintile"] = np.nan
    for (date_val, size), group in joined.groupby(["date", "size_bucket"]):
        if len(group) < 5:
            continue
        try:
            labels = pd.qcut(
                group["raw_signal"], q=5, labels=False, duplicates="drop"
            )
            joined.loc[group.index, "lp_quintile"] = labels.values + 1
        except ValueError:
            continue
    joined = joined.dropna(subset=["lp_quintile"])
    print(f"  {len(joined):,} rows with size+quintile assigned")

    # Forward returns
    print(f"Computing {HOLDING_DAYS}-day forward returns...")
    all_returns = returns[["permno", "date", "ret"]].sort_values(["permno", "date"])
    fwd_rows = []
    for rebal_date in rebal_dates:
        eligible_permnos = joined[joined.date == rebal_date]["permno"].unique()
        future = all_returns[
            (all_returns.date > rebal_date)
            & (all_returns.permno.isin(eligible_permnos))
        ].sort_values(["permno", "date"])
        future["rank"] = future.groupby("permno").cumcount()
        window = future[future["rank"] < HOLDING_DAYS]
        fwd = (
            window.groupby("permno")["ret"]
            .apply(lambda r: (1 + r).prod() - 1)
            .rename("fwd_return")
            .reset_index()
        )
        fwd["date"] = rebal_date
        fwd_rows.append(fwd)
    fwd_returns = pd.concat(fwd_rows, ignore_index=True)

    merged = joined.merge(fwd_returns, on=["date", "permno"], how="left")
    merged = merged.dropna(subset=["fwd_return"])
    print(f"  {len(merged):,} rows with forward returns")

    # Per-bucket monthly means
    per_bucket_monthly = (
        merged.groupby(["date", "size_bucket", "lp_quintile"])["fwd_return"]
        .agg(["mean", "count"])
        .reset_index()
    )

    # Summary across dates
    summary = (
        per_bucket_monthly.groupby(["size_bucket", "lp_quintile"])["mean"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary.columns = ["size_bucket", "lp_quintile", "mean_mo_ret", "std_mo_ret", "n_months"]
    summary["annualized_ret"] = (1 + summary["mean_mo_ret"]) ** 12 - 1
    summary["monthly_sharpe"] = summary["mean_mo_ret"] / summary["std_mo_ret"]
    summary["annualized_sharpe"] = summary["monthly_sharpe"] * np.sqrt(12)

    print()
    print("=" * 70)
    print("3x5 sort: Size tercile x Lazy Prices quintile")
    print("Equal-weighted within bucket; average of monthly bucket means")
    print("=" * 70)
    for size in ["Small", "Mid", "Large"]:
        print(f"\n{size}-cap:")
        s = summary[summary.size_bucket == size].sort_values("lp_quintile")
        print(s[["lp_quintile", "mean_mo_ret", "annualized_ret", "annualized_sharpe", "n_months"]].to_string(index=False))

    # 3x1 L/S summary
    print()
    print("=" * 70)
    print("L/S per size tercile (Q1 - Q5, per CMN direction)")
    print("=" * 70)
    for size in ["Small", "Mid", "Large"]:
        q1 = per_bucket_monthly[
            (per_bucket_monthly.size_bucket == size)
            & (per_bucket_monthly.lp_quintile == 1)
        ].set_index("date")["mean"]
        q5 = per_bucket_monthly[
            (per_bucket_monthly.size_bucket == size)
            & (per_bucket_monthly.lp_quintile == 5)
        ].set_index("date")["mean"]
        ls = (q1 - q5).dropna()
        ann_ret = (1 + ls.mean()) ** 12 - 1
        ann_sharpe = ls.mean() / ls.std() * np.sqrt(12)
        print(f"  {size}: L/S mean {ls.mean() * 100:+.3f}%/mo, "
              f"ann {ann_ret * 100:+.2f}%, Sharpe {ann_sharpe:+.3f} "
              f"(N={len(ls)})")

    # Save
    out = BACKTEST_DIR / "size_tercile_by_lazy_prices_sort.parquet"
    per_bucket_monthly.to_parquet(out, index=False)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()