"""5x5 sort with NYSE-quintile size breakpoints (CMN standard).

Reproduces the 5x5 size_quintile x LP_quintile sort methodology from
size_quintile_by_lazy_prices_sort.py, but uses NYSE-based size
breakpoints (fetched by fetch_nyse_size_breakpoints.py) instead of
in-sample percentiles.

This is the standard academic robustness check: NYSE breakpoints
prevent NASDAQ's many small tech firms from inflating the in-sample
quintile cutoffs, which would systematically over-classify tech
firms as 'large'.

Compares directly to the in-sample results from 2026-07-23 (commit
9cb0fc0):
  Size1 L/S: -2.18% ann, Sharpe -0.117 (in-sample)
  Size2 L/S: -2.70% ann, Sharpe -0.257 (in-sample)
  Size3 L/S: -0.45% ann, Sharpe -0.050 (in-sample)
  Size4 L/S: +4.42% ann, Sharpe +0.638 PEAK (in-sample)
  Size5 L/S: +0.84% ann, Sharpe +0.140 (in-sample)

Also computes NYSE breakpoint-adjusted counterfactuals: top-N mega-cap
exclusion within Size5.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np


BACKTEST_DIR = Path("data/cache/lazy_prices_backtest")
RETURNS_CACHE = Path("data/cache/lazy_prices_returns_daily.parquet")
SIGNAL_CACHE = Path("data/cache/lazy_prices_signal.parquet")
NYSE_BREAKPOINTS = Path("data/cache/nyse_size_breakpoints.parquet")
CIK_MERGE = Path("data/cache/lazy_prices_ciks_merged_dedup.parquet")
HOLDING_DAYS = 21
N_LP_QUINTILES = 5


def assign_size_bucket_nyse(marketcap: float, p20: float, p40: float,
                             p60: float, p80: float) -> int:
    """Assign size bucket based on NYSE breakpoints.

    marketcap and breakpoints must be in the same units.
    """
    if marketcap <= p20:
        return 1
    elif marketcap <= p40:
        return 2
    elif marketcap <= p60:
        return 3
    elif marketcap <= p80:
        return 4
    else:
        return 5


def main() -> None:
    print("Loading data...")
    positions = pd.read_parquet(BACKTEST_DIR / "quintile_positions.parquet")
    returns = pd.read_parquet(RETURNS_CACHE)
    signal = pd.read_parquet(SIGNAL_CACHE)
    breakpoints = pd.read_parquet(NYSE_BREAKPOINTS)
    ciks = pd.read_parquet(CIK_MERGE)

    positions["date"] = pd.to_datetime(positions["date"])
    returns["date"] = pd.to_datetime(returns["date"])
    signal["date_filed"] = pd.to_datetime(signal["date_filed"])
    breakpoints["date"] = pd.to_datetime(breakpoints["date"])

    # Convert NYSE breakpoints from THOUSANDS OF DOLLARS to RAW DOLLARS
    # to match returns.marketcap units
    for col in ["p20", "p40", "p60", "p80", "median"]:
        breakpoints[col] = breakpoints[col] * 1000

    rebal_dates = sorted(positions["date"].unique())
    print(f"  {len(rebal_dates)} rebalance dates")
    print(f"  NYSE breakpoints: {len(breakpoints)} dates, "
          f"{breakpoints.date.min()} to {breakpoints.date.max()}")

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

    # Merge breakpoints by rebalance date
    print("Assigning size buckets via NYSE breakpoints...")
    joined = joined.merge(
        breakpoints[["date", "p20", "p40", "p60", "p80"]],
        on="date", how="left",
    )
    joined = joined.dropna(subset=["p20"])

    # Vectorized bucket assignment
    conditions = [
        joined["marketcap"] <= joined["p20"],
        joined["marketcap"] <= joined["p40"],
        joined["marketcap"] <= joined["p60"],
        joined["marketcap"] <= joined["p80"],
    ]
    choices = [1, 2, 3, 4]
    joined["size_bucket"] = np.select(conditions, choices, default=5)

    print(f"  Size bucket distribution across all (date, permno) rows:")
    dist = joined["size_bucket"].value_counts().sort_index()
    total = dist.sum()
    for size, count in dist.items():
        print(f"    Size{size}: {count:>6,} ({count/total*100:.1f}%)")

    # Assign LP quintiles within each (date, size_bucket)
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
    joined["lp_quintile"] = joined["lp_quintile"].astype(int)
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
    print("5x5 sort with NYSE-quintile size breakpoints")
    print("Equal-weighted within bucket; average of monthly bucket means")
    print("=" * 70)
    for size in range(1, 6):
        s = summary[summary.size_bucket == size].sort_values("lp_quintile")
        if len(s) == 0:
            print(f"\nSize{size}: no observations")
            continue
        print(f"\nSize{size}:")
        print(s[["lp_quintile", "mean_mo_ret", "annualized_ret", "annualized_sharpe", "n_months"]].to_string(index=False))

    # 5x1 L/S summary
    print()
    print("=" * 70)
    print("L/S per size bucket (Q1 - Q5, per CMN direction)")
    print("=" * 70)
    print(f"{'Bucket':<8} {'L/S ann':>10} {'Sharpe':>10} {'N months':>10}")
    print("-" * 42)
    for size in range(1, 6):
        q1 = per_bucket_monthly[
            (per_bucket_monthly.size_bucket == size)
            & (per_bucket_monthly.lp_quintile == 1)
        ].set_index("date")["mean"]
        q5 = per_bucket_monthly[
            (per_bucket_monthly.size_bucket == size)
            & (per_bucket_monthly.lp_quintile == 5)
        ].set_index("date")["mean"]
        ls = (q1 - q5).dropna()
        if len(ls) == 0:
            print(f"Size{size:<3} (no obs)")
            continue
        ann_ret = (1 + ls.mean()) ** 12 - 1
        ann_sharpe = ls.mean() / ls.std() * np.sqrt(12)
        print(f"Size{size:<3} {ann_ret*100:>+9.2f}% {ann_sharpe:>+9.3f} {len(ls):>10}")

    # Also: top-N Size5 mega-cap counterfactual under NYSE breakpoints
    print()
    print("=" * 70)
    print("Size5 VW L/S: full vs excluding top-N mega-caps")
    print("=" * 70)

    s5 = merged[merged.size_bucket == 5].copy()
    print(f"  Size5 (NYSE p80+) rows: {len(s5):,} across {s5.date.nunique()} dates")

    def _vw_ls(group: pd.DataFrame) -> float:
        q1 = group[group.lp_quintile == 1]
        q5 = group[group.lp_quintile == 5]
        q1_ret = np.nan
        q5_ret = np.nan
        if len(q1) > 0 and q1["marketcap"].sum() > 0:
            w = q1["marketcap"] / q1["marketcap"].sum()
            q1_ret = (w * q1["fwd_return"]).sum()
        if len(q5) > 0 and q5["marketcap"].sum() > 0:
            w = q5["marketcap"] / q5["marketcap"].sum()
            q5_ret = (w * q5["fwd_return"]).sum()
        if pd.isna(q1_ret) or pd.isna(q5_ret):
            return np.nan
        return q1_ret - q5_ret

    for top_n in [0, 5, 10]:
        if top_n == 0:
            sub = s5
            label = "Full Size5 (NYSE p80+)"
        else:
            # Mark top-N per date by mcap, then exclude
            s5_marked = s5.copy()
            s5_marked["top_n_flag"] = False
            for date_val, group in s5_marked.groupby("date"):
                top_permnos = group.nlargest(top_n, "marketcap")["permno"].values
                s5_marked.loc[
                    (s5_marked.date == date_val) & (s5_marked.permno.isin(top_permnos)),
                    "top_n_flag"
                ] = True
            sub = s5_marked[~s5_marked.top_n_flag]
            label = f"Size5 excluding top-{top_n}"

        ls = sub.groupby("date").apply(_vw_ls, include_groups=False).dropna()
        if len(ls) == 0:
            print(f"  {label}: no observations")
            continue
        ann = ls.mean() * 12 * 100
        sharpe = ls.mean() / ls.std() * np.sqrt(12)
        print(f"  {label}: ann {ann:+.2f}%, Sharpe {sharpe:+.3f}, N={len(ls)}")

    # Save
    out = BACKTEST_DIR / "size_quintile_nyse_breakpoints.parquet"
    per_bucket_monthly.to_parquet(out, index=False)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()