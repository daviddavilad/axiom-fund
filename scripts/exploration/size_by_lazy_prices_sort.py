"""5x2 sort: Size (Small/Big) x Lazy Prices quintile.

For each rebalance date:
  1. Split universe into Small/Big by median marketcap
  2. Within each size group, sort into 5 Lazy Prices quintiles by raw_signal
  3. Compute equal-weighted 21-day forward return per bucket

Reveals whether the Lazy Prices signal:
  - Works in Small but not Big (small-cap phenomenon per CMN's original)
  - Reverses direction in Big (value-weighted L/S goes from CMN-direction
    to opposite as we move from small to large caps)

Prompted by finding on 2026-07-22 that value-weighted L/S = -3.48%/yr
while equal-weighted L/S = +2.13%/yr — a 5.6%/yr spread between weightings
suggests strong size-interaction with the signal.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np


BACKTEST_DIR = Path("data/cache/lazy_prices_backtest")
RETURNS_CACHE = Path("data/cache/lazy_prices_returns_daily.parquet")
SIGNAL_CACHE = Path("data/cache/lazy_prices_signal.parquet")
UNIVERSE_PATH = Path("data/cache/lazy_prices_universe.parquet")
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

    # Build per-rebalance-date marketcap lookup with merge_asof
    # (last trading day <= rebalance date)
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
    print(f"  {len(mcap_at_rebal):,} (date, permno, marketcap) rows")

    # For each rebalance date, need the raw_signal per permno.
    # positions has quintile but not raw_signal. Rebuild via the same
    # forward-carry logic used in the backtest: raw_signal is carried
    # forward from date_filed until the next filing per (permno).
    # Simplification: use signal.date_filed <= rebal_date, take latest per permno.
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
    print(f"  {len(aligned_signal):,} aligned signal rows")

    # Join marketcap + raw_signal
    joined = aligned_signal.merge(
        mcap_at_rebal, on=["date", "permno"], how="inner"
    )
    print(f"  {len(joined):,} rows after joining marketcap")

    # For each rebalance date: assign Small/Big by median marketcap
    print("Assigning Small/Big split by in-sample median marketcap per date...")
    joined["size_bucket"] = ""
    for date_val, group in joined.groupby("date"):
        med = group["marketcap"].median()
        joined.loc[group.index, "size_bucket"] = np.where(
            group["marketcap"] < med, "Small", "Big"
        )

    # Assign 5 Lazy Prices quintiles within each size bucket, per date
    print("Assigning Lazy Prices quintiles within each size bucket...")
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

    # Compute 21-day forward return per (rebal, permno)
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

    # Merge forward returns
    merged = joined.merge(fwd_returns, on=["date", "permno"], how="left")
    merged = merged.dropna(subset=["fwd_return"])
    print(f"  {len(merged):,} rows with forward returns")

    # Aggregate: per-date, per (size_bucket, lp_quintile) mean forward return
    per_bucket_monthly = (
        merged.groupby(["date", "size_bucket", "lp_quintile"])["fwd_return"]
        .agg(["mean", "count"])
        .reset_index()
    )

    # Overall: mean of monthly means per (size, quintile)
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
    print("5x2 sort: Size (Small/Big) x Lazy Prices quintile")
    print("Equal-weighted within bucket; average of monthly bucket means")
    print("=" * 70)
    for size in ["Small", "Big"]:
        print(f"\n{size}-cap:")
        s = summary[summary.size_bucket == size].sort_values("lp_quintile")
        print(s[["lp_quintile", "mean_mo_ret", "annualized_ret", "annualized_sharpe", "n_months"]].to_string(index=False))

    # 2x2 L/S summary: Small L/S vs Big L/S
    print()
    print("=" * 70)
    print("L/S per size bucket (Q1 - Q5, per CMN direction)")
    print("=" * 70)
    for size in ["Small", "Big"]:
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
    out = BACKTEST_DIR / "size_by_lazy_prices_sort.parquet"
    per_bucket_monthly.to_parquet(out, index=False)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()