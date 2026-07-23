"""VW vs EW L/S per size quintile.

For each size bucket (Size1-Size5), compute L/S using both equal-
weighted and value-weighted schemes WITHIN THE SAME BUCKET. Reveals:
  1. Whether the global-VW L/S reversal (-3.48% ann) is driven by
     mega-mega-caps (should show Size5 VW L/S strongly negative)
     or by cross-bucket weighting effects (should show Size5 VW L/S
     ≈ Size5 EW L/S ≈ 0)
  2. Whether the CMN peak in Size4 (EW Sharpe +0.638) survives VW
     within that bucket

Prompted by 2026-07-23 finding that Size5 quintile EW L/S ≈ 0 but
global VW L/S = -3.48%, implying value-weighting distortion isn't
uniform across Size5 firms.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np


BACKTEST_DIR = Path("data/cache/lazy_prices_backtest")
RETURNS_CACHE = Path("data/cache/lazy_prices_returns_daily.parquet")
SIGNAL_CACHE = Path("data/cache/lazy_prices_signal.parquet")
HOLDING_DAYS = 21
N_SIZE_BUCKETS = 5
N_LP_QUINTILES = 5


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
    print(f"  {len(mcap_at_rebal):,} rows")

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

    joined = aligned_signal.merge(
        mcap_at_rebal, on=["date", "permno"], how="inner"
    )
    print(f"  {len(joined):,} rows after joining")

    # Size buckets per date
    print(f"Assigning {N_SIZE_BUCKETS} size buckets per date...")
    joined["size_bucket"] = np.nan
    for date_val, group in joined.groupby("date"):
        if len(group) < N_SIZE_BUCKETS:
            continue
        try:
            labels = pd.qcut(
                group["marketcap"], q=N_SIZE_BUCKETS, labels=False, duplicates="drop"
            )
            joined.loc[group.index, "size_bucket"] = labels.values + 1
        except ValueError:
            continue
    joined = joined.dropna(subset=["size_bucket"])
    joined["size_bucket"] = joined["size_bucket"].astype(int)

    # LP quintile within each (date, size_bucket)
    print("Assigning LP quintiles within each size bucket...")
    joined["lp_quintile"] = np.nan
    for (date_val, size), group in joined.groupby(["date", "size_bucket"]):
        if len(group) < N_LP_QUINTILES:
            continue
        try:
            labels = pd.qcut(
                group["raw_signal"], q=N_LP_QUINTILES, labels=False, duplicates="drop"
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

    # Compute EW and VW L/S per (date, size_bucket)
    print()
    print("Computing EW and VW L/S per (date, size_bucket)...")

    def _leg_returns(group: pd.DataFrame) -> pd.Series:
        """Return EW and VW leg returns for one (date, size_bucket)."""
        q1 = group[group["lp_quintile"] == 1.0]
        q5 = group[group["lp_quintile"] == 5.0]
        # EW
        q1_ew = q1["fwd_return"].mean() if len(q1) > 0 else np.nan
        q5_ew = q5["fwd_return"].mean() if len(q5) > 0 else np.nan
        # VW
        q1_vw = np.nan
        q5_vw = np.nan
        if len(q1) > 0 and q1["marketcap"].sum() > 0:
            w = q1["marketcap"] / q1["marketcap"].sum()
            q1_vw = float((w * q1["fwd_return"]).sum())
        if len(q5) > 0 and q5["marketcap"].sum() > 0:
            w = q5["marketcap"] / q5["marketcap"].sum()
            q5_vw = float((w * q5["fwd_return"]).sum())
        return pd.Series({
            "q1_ew": q1_ew, "q5_ew": q5_ew,
            "q1_vw": q1_vw, "q5_vw": q5_vw,
        })

    per_group = (
        merged.groupby(["date", "size_bucket"])
        .apply(_leg_returns, include_groups=False)
        .reset_index()
    )
    per_group["ls_ew"] = per_group["q1_ew"] - per_group["q5_ew"]
    per_group["ls_vw"] = per_group["q1_vw"] - per_group["q5_vw"]

    print()
    print("=" * 70)
    print("VW vs EW L/S per size quintile (Q1 - Q5, monthly, then annualized)")
    print("=" * 70)
    print(f"{'Bucket':<8} {'EW ann%':>10} {'EW Sharpe':>12} "
          f"{'VW ann%':>10} {'VW Sharpe':>12} {'VW-EW gap':>12}")
    print("-" * 70)
    for size in range(1, N_SIZE_BUCKETS + 1):
        sub = per_group[per_group.size_bucket == size].dropna(subset=["ls_ew", "ls_vw"])
        n = len(sub)
        ew_ann = (1 + sub.ls_ew.mean()) ** 12 - 1
        vw_ann = (1 + sub.ls_vw.mean()) ** 12 - 1
        ew_sh = sub.ls_ew.mean() / sub.ls_ew.std() * np.sqrt(12)
        vw_sh = sub.ls_vw.mean() / sub.ls_vw.std() * np.sqrt(12)
        gap_ann = vw_ann - ew_ann
        print(f"Size{size:<3} {ew_ann*100:>+9.2f}% {ew_sh:>+11.3f}  "
              f"{vw_ann*100:>+9.2f}% {vw_sh:>+11.3f}  {gap_ann*100:>+10.2f}%  (N={n})")

    # Save
    out = BACKTEST_DIR / "size_quintile_vw_ew_compare.parquet"
    per_group.to_parquet(out, index=False)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()