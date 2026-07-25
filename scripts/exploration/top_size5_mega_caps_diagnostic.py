"""Top-10 Size5 mega-caps diagnostic.

For each rebalance date, identifies the top-10 firms by marketcap within
the Size5 bucket (top 20% marketcap). Examines their raw_signal
distribution, LP quintile assignments, forward returns, and their
contribution to the VW L/S within Size5.

Also computes counterfactual: Size5 VW L/S EXCLUDING the top-10 firms
per date. If mega-cap hypothesis is correct, excluding top-10 should
either flip the sign or push toward zero.

Prompted by 2026-07-23 finding that Size5 EW L/S ≈ +0.140 Sharpe (near
zero) but Size5 VW L/S = -0.528 Sharpe (strong reversal). Diagnostic
identifies WHICH mega-caps drive the reversal.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np


BACKTEST_DIR = Path("data/cache/lazy_prices_backtest")
RETURNS_CACHE = Path("data/cache/lazy_prices_returns_daily.parquet")
SIGNAL_CACHE = Path("data/cache/lazy_prices_signal.parquet")
CIK_MERGE = Path("data/cache/lazy_prices_ciks_merged_dedup.parquet")
HOLDING_DAYS = 21
N_SIZE_BUCKETS = 5
N_LP_QUINTILES = 5
TOP_N = 10


def main() -> None:
    print("Loading data...")
    positions = pd.read_parquet(BACKTEST_DIR / "quintile_positions.parquet")
    returns = pd.read_parquet(RETURNS_CACHE)
    signal = pd.read_parquet(SIGNAL_CACHE)
    ciks = pd.read_parquet(CIK_MERGE)

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

    # LP quintiles within each (date, size_bucket)
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

    # Focus on Size5
    s5 = merged[merged.size_bucket == 5].copy()
    print(f"  Size5 rows across {s5.date.nunique()} dates: {len(s5):,}")

    # Identify top-N per date by marketcap
    print(f"\nIdentifying top-{TOP_N} mega-caps within Size5 per date...")
    s5["top_n_flag"] = False
    for date_val, group in s5.groupby("date"):
        top_permnos = group.nlargest(TOP_N, "marketcap")["permno"].values
        s5.loc[(s5.date == date_val) & (s5.permno.isin(top_permnos)), "top_n_flag"] = True

    ticker_map = ciks[["permno", "ticker"]].drop_duplicates(subset=["permno"])
    s5 = s5.merge(ticker_map, on="permno", how="left")

    # === TABLE 1: Firm-level stats ===
    print()
    print("=" * 100)
    print(f"TABLE 1: Top-{TOP_N} Size5 mega-caps — appearance count, avg signal, avg return")
    print("=" * 100)
    top_only = s5[s5.top_n_flag].copy()
    firm_stats = (
        top_only.groupby(["permno", "ticker"])
        .agg(
            n_months=("date", "count"),
            avg_marketcap_bn=("marketcap", lambda x: x.mean() / 1e9),
            avg_raw_signal=("raw_signal", "mean"),
            avg_lp_quintile=("lp_quintile", "mean"),
            avg_fwd_return_pct=("fwd_return", lambda x: x.mean() * 100),
        )
        .reset_index()
        .sort_values("n_months", ascending=False)
    )
    print(firm_stats[["ticker", "n_months", "avg_marketcap_bn",
                       "avg_raw_signal", "avg_lp_quintile", "avg_fwd_return_pct"]].to_string(index=False))

    # === TABLE 2: LP quintile distribution top-10 vs rest of Size5 ===
    print()
    print("=" * 70)
    print(f"TABLE 2: LP quintile distribution — top-{TOP_N} vs rest of Size5")
    print("=" * 70)
    top_dist = top_only["lp_quintile"].value_counts(normalize=True).sort_index() * 100
    rest_dist = s5[~s5.top_n_flag]["lp_quintile"].value_counts(normalize=True).sort_index() * 100
    dist_df = pd.DataFrame({
        "top_10_pct": top_dist.round(1),
        "rest_pct": rest_dist.round(1),
    }).fillna(0)
    print(dist_df.to_string())

    # === TABLE 3: Counterfactual Size5 VW L/S excluding top-N ===
    print()
    print("=" * 70)
    print(f"TABLE 3: Size5 VW L/S — full vs excluding top-{TOP_N}")
    print("=" * 70)

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

    full_ls = s5.groupby("date").apply(_vw_ls, include_groups=False).dropna()
    rest_ls = s5[~s5.top_n_flag].groupby("date").apply(_vw_ls, include_groups=False).dropna()

    for name, series in [
        ("Full Size5 VW L/S (baseline)", full_ls),
        (f"Size5 VW L/S excluding top-{TOP_N}", rest_ls),
    ]:
        n = len(series)
        ann_ret = series.mean() * 12 * 100
        ann_sharpe = series.mean() / series.std() * np.sqrt(12)
        print(f"  {name}: ann {ann_ret:+.2f}%, Sharpe {ann_sharpe:+.3f}, N={n}")

    # Save
    out = BACKTEST_DIR / "top_size5_mega_caps_diagnostic.parquet"
    top_only.to_parquet(out, index=False)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()