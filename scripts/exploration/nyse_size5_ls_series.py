"""Reconstruct NYSE Size5 L/S monthly series (both EW and VW).

Extracts Size5-only L/S returns per rebalance date under NYSE-breakpoint
methodology, matching the setup in size_quintile_nyse_breakpoints.py.
Saves both EW and VW monthly series for downstream FF6 spanning +
bootstrap analysis.

Prompted by 2026-07-25: NYSE Size5 EW L/S showed FF6-orthogonal alpha
+5.70%/yr HAC t=3.04. Now need parallel VW series to test whether the
result is EW-specific (mega-cap concentration dilutes VW) or robust
under both weightings.

Output: data/cache/lazy_prices_backtest/nyse_size5_ls_series.parquet
with columns: date, ls_ew, ls_vw, n_q1, n_q5, mcap_q1_sum, mcap_q5_sum.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np


BACKTEST_DIR = Path("data/cache/lazy_prices_backtest")
RETURNS_CACHE = Path("data/cache/lazy_prices_returns_daily.parquet")
SIGNAL_CACHE = Path("data/cache/lazy_prices_signal.parquet")
NYSE_BREAKPOINTS = Path("data/cache/nyse_size_breakpoints.parquet")
HOLDING_DAYS = 21
N_LP_QUINTILES = 5


def main() -> None:
    print("Loading data...")
    positions = pd.read_parquet(BACKTEST_DIR / "quintile_positions.parquet")
    returns = pd.read_parquet(RETURNS_CACHE)
    signal = pd.read_parquet(SIGNAL_CACHE)
    breakpoints = pd.read_parquet(NYSE_BREAKPOINTS)

    positions["date"] = pd.to_datetime(positions["date"])
    returns["date"] = pd.to_datetime(returns["date"])
    signal["date_filed"] = pd.to_datetime(signal["date_filed"])
    breakpoints["date"] = pd.to_datetime(breakpoints["date"])

    # NYSE breakpoints from thousands of dollars to raw dollars
    for col in ["p20", "p40", "p60", "p80", "median"]:
        breakpoints[col] = breakpoints[col] * 1000

    rebal_dates = sorted(positions["date"].unique())
    print(f"  {len(rebal_dates)} rebalance dates")

    # Marketcap lookup at rebalance dates
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

    # Raw signal aligned
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

    joined = aligned_signal.merge(mcap_at_rebal, on=["date", "permno"], how="inner")

    # NYSE breakpoint size bucket assignment
    print("Assigning size buckets via NYSE breakpoints...")
    joined = joined.merge(
        breakpoints[["date", "p20", "p40", "p60", "p80"]],
        on="date", how="left",
    )
    joined = joined.dropna(subset=["p20"])

    conditions = [
        joined["marketcap"] <= joined["p20"],
        joined["marketcap"] <= joined["p40"],
        joined["marketcap"] <= joined["p60"],
        joined["marketcap"] <= joined["p80"],
    ]
    choices = [1, 2, 3, 4]
    joined["size_bucket"] = np.select(conditions, choices, default=5)

    # Filter to Size5 only
    s5 = joined[joined.size_bucket == 5].copy()
    print(f"  Size5 (NYSE p80+) rows: {len(s5):,}")

    # LP quintiles within Size5 per date
    print("Assigning LP quintiles within Size5 per date...")
    s5["lp_quintile"] = np.nan
    for date_val, group in s5.groupby("date"):
        if len(group) < N_LP_QUINTILES:
            continue
        try:
            labels = pd.qcut(
                group["raw_signal"], q=N_LP_QUINTILES, labels=False, duplicates="drop"
            )
            s5.loc[group.index, "lp_quintile"] = labels.values + 1
        except ValueError:
            continue
    s5 = s5.dropna(subset=["lp_quintile"])
    s5["lp_quintile"] = s5["lp_quintile"].astype(int)

    # Forward returns
    print(f"Computing {HOLDING_DAYS}-day forward returns...")
    all_returns = returns[["permno", "date", "ret"]].sort_values(["permno", "date"])
    fwd_rows = []
    for rebal_date in rebal_dates:
        eligible_permnos = s5[s5.date == rebal_date]["permno"].unique()
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

    merged = s5.merge(fwd_returns, on=["date", "permno"], how="left")
    merged = merged.dropna(subset=["fwd_return"])
    print(f"  {len(merged):,} rows with forward returns")

    # Compute EW and VW L/S per date
    print("Computing EW and VW L/S per date...")
    rows = []
    for date_val, group in merged.groupby("date"):
        q1 = group[group.lp_quintile == 1]
        q5 = group[group.lp_quintile == 5]
        if len(q1) == 0 or len(q5) == 0:
            continue
        # EW
        q1_ew = q1["fwd_return"].mean()
        q5_ew = q5["fwd_return"].mean()
        ls_ew = q1_ew - q5_ew
        # VW
        q1_mcap_sum = q1["marketcap"].sum()
        q5_mcap_sum = q5["marketcap"].sum()
        if q1_mcap_sum <= 0 or q5_mcap_sum <= 0:
            continue
        q1_vw = (q1["marketcap"] / q1_mcap_sum * q1["fwd_return"]).sum()
        q5_vw = (q5["marketcap"] / q5_mcap_sum * q5["fwd_return"]).sum()
        ls_vw = q1_vw - q5_vw
        rows.append({
            "date": date_val,
            "ls_ew": ls_ew,
            "ls_vw": ls_vw,
            "n_q1": len(q1),
            "n_q5": len(q5),
            "mcap_q1_sum": q1_mcap_sum,
            "mcap_q5_sum": q5_mcap_sum,
        })

    series = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    print(f"  {len(series)} monthly L/S rows")

    # Sanity: point estimates
    print()
    print("=" * 70)
    print("NYSE Size5 L/S monthly series — point estimates")
    print("=" * 70)
    for label, col in [("EW", "ls_ew"), ("VW", "ls_vw")]:
        x = series[col]
        ann = (1 + x.mean()) ** 12 - 1
        sharpe = x.mean() / x.std() * np.sqrt(12)
        print(f"  {label}: mean {x.mean()*100:+.4f}%/mo, ann {ann*100:+.2f}%, "
              f"Sharpe {sharpe:+.3f}, N={len(x)}")
    print(f"  Avg n_q1 firms per date: {series.n_q1.mean():.0f}, "
          f"n_q5: {series.n_q5.mean():.0f}")

    out = BACKTEST_DIR / "nyse_size5_ls_series.parquet"
    series.to_parquet(out, index=False)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()