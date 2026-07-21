"""Per-quintile monthly forward returns table.

For each rebalance date, computes the equal-weighted mean 21-day forward
return of each quintile (Q1 through Q5). Reveals whether the L/S signal
is a spread (monotonic across quintiles) or a boundary effect (Q1 vs Q5
only), and whether the direction is what we think it is.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np


BACKTEST_DIR = Path("data/cache/lazy_prices_backtest")
RETURNS_CACHE = Path("data/cache/lazy_prices_returns_daily.parquet")
HOLDING_DAYS = 21


def main() -> None:
    print("Loading quintile positions + returns...")
    positions = pd.read_parquet(BACKTEST_DIR / "quintile_positions.parquet")
    returns = pd.read_parquet(RETURNS_CACHE)

    positions["date"] = pd.to_datetime(positions["date"])
    returns["date"] = pd.to_datetime(returns["date"])

    print(f"  Positions: {len(positions):,} rows, {positions.date.nunique()} rebalance dates")
    print(f"  Returns:   {len(returns):,} rows, {returns.permno.nunique()} permnos")
    print()

    # For each rebalance date, compute cumulative 21-day forward return per permno
    print("Computing 21-day forward returns per (rebalance_date, permno)...")
    all_trading = returns[["permno", "date", "ret"]].copy()
    all_trading = all_trading.sort_values(["permno", "date"])

    fwd_rows = []
    for rebal_date in sorted(positions["date"].unique()):
        # Rebalance-eligible permnos
        rebal_permnos = positions[positions.date == rebal_date]["permno"].unique()

        # Compute forward return: product of (1+ret) for the next 21 trading days
        # per permno, starting the day AFTER rebalance
        future = all_trading[
            (all_trading.date > rebal_date)
            & (all_trading.permno.isin(rebal_permnos))
        ]
        future = future.sort_values(["permno", "date"])
        # Take first 21 trading days per permno
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
    print(f"  Forward returns: {len(fwd_returns):,} rows")

    # Join with quintile assignments
    print("Joining with quintile assignments...")
    merged = positions[["date", "permno", "quintile"]].merge(
        fwd_returns, on=["date", "permno"], how="left"
    )
    merged = merged.dropna(subset=["fwd_return"])
    print(f"  Merged: {len(merged):,} rows with returns")
    print()

    # Per-rebalance-date, per-quintile mean forward return
    monthly_quintile = (
        merged.groupby(["date", "quintile"])["fwd_return"]
        .agg(["mean", "count"])
        .reset_index()
    )

    # Aggregate: mean of monthly means per quintile (annualized approximately)
    print("=" * 70)
    print("Per-quintile mean 21-day forward return (across all rebalance months)")
    print("=" * 70)
    per_quintile = (
        monthly_quintile.groupby("quintile")["mean"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    per_quintile.columns = ["quintile", "mean_monthly_ret", "std_monthly_ret", "n_months"]
    per_quintile["annualized_ret"] = (1 + per_quintile["mean_monthly_ret"]) ** 12 - 1
    per_quintile["monthly_sharpe"] = (
        per_quintile["mean_monthly_ret"] / per_quintile["std_monthly_ret"]
    )
    per_quintile["annualized_sharpe"] = per_quintile["monthly_sharpe"] * np.sqrt(12)
    print(per_quintile.to_string(index=False))
    print()

    # Cross-quintile spread
    q1 = per_quintile[per_quintile.quintile == 1]["mean_monthly_ret"].iloc[0]
    q5 = per_quintile[per_quintile.quintile == 5]["mean_monthly_ret"].iloc[0]
    print(f"Q5 - Q1 spread (monthly):   {(q5 - q1) * 100:+.3f}%")
    print(f"Q5 - Q1 spread (annualized): {((1 + q5)**12 - (1 + q1)**12) * 100:+.3f}%")
    print()

    # Save for downstream
    OUT = BACKTEST_DIR / "per_quintile_monthly_returns.parquet"
    monthly_quintile.to_parquet(OUT, index=False)
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()