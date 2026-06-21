"""Apply v2 inference (HAC + bootstrap) to v1's existing statistics.

Per docs/v2_design.md Item 3, this script recomputes v1's headline
inference under HAC-corrected standard errors and block-bootstrapped
confidence intervals, then reports what changes vs the asymptotic
numbers v1 originally published.

Two sections:

1. **IC t-stats by signal.** Reads data/cache/ic_analysis_4sig/
   ic_long.parquet (116 monthly Spearman IC values per signal across
   GP, IVol, ResMom, PEAD). For each signal, reports:
     - mean IC
     - naive (i.i.d.) t-stat
     - HAC t-stat at L=3 and L=5
   The naive t-stat is what v1's onepager reports. The HAC version is
   the v2 correction. A drop in significance from naive to HAC means
   the autocorrelation in the IC time series was inflating v1's
   apparent significance.

2. **Sharpe bootstrap CIs.** Reads
   data/cache/backtest_full_top1000/net_returns.parquet (116 monthly
   portfolio returns, gross and net). For each, reports:
     - point Sharpe (annualized, mean/std * sqrt(12))
     - asymptotic 95% CI from SE = sqrt((1 + 0.5 SR^2) / n)
     - block-bootstrap 95% CI at block_size=6, n_resamples=10000
   The asymptotic CI assumes i.i.d. normal returns. The bootstrap CI
   does not assume normality and preserves short-range dependence.

Output:
  data/cache/inference_v2/ic_hac.parquet         — IC HAC t-stats
  data/cache/inference_v2/sharpe_bootstrap.parquet — Sharpe CIs
  data/cache/inference_v2/run_metadata.txt

Aggregate summary printed to console.

Design choices documented in code:
  - L=3 and L=5: covers Andrews/Newey rule of thumb at N=116
  - block_size=6: ~N^(1/3) for N=116 monthly observations
  - Spearman IC chosen over Pearson to match v1's onepager convention

This script is one-off; not part of the production backtest.
"""
# ruff: noqa: I001

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from axiom_fund.diagnostics.inference import (
    compute_bootstrapped_sharpe_ci,
    compute_hac_standard_error_of_mean,
)


IC_PATH = Path("data/cache/ic_analysis_4sig/ic_long.parquet")
RETURNS_PATH = Path("data/cache/backtest_full_top1000/net_returns.parquet")
OUTPUT_DIR = Path("data/cache/inference_v2")

# IC t-stat: lag choices spanning Andrews/Newey rule of thumb at N=116
HAC_LAGS = [3, 5]

# Bootstrap: block_size ~ N^(1/3), n_resamples large for tight CI
BLOCK_SIZE = 6
N_RESAMPLES = 10_000
SEED = 42
ANNUALIZATION = 12  # monthly returns → annual Sharpe


def _ic_hac_analysis(ic_df: pd.DataFrame) -> pd.DataFrame:
    """Per-signal: mean IC, naive t-stat, HAC t-stats at multiple lags."""
    rows = []
    for signal, group in ic_df.groupby("signal"):
        group = group.sort_values("rebalance_date").reset_index(drop=True)
        ic_series = group["ic_spearman"].to_numpy(dtype=np.float64)
        n = len(ic_series)
        mean_ic = float(ic_series.mean())
        naive_se = float(ic_series.std(ddof=1) / np.sqrt(n))
        naive_t = mean_ic / naive_se if naive_se > 0 else np.nan

        row = {
            "signal": signal,
            "n_periods": n,
            "mean_ic": mean_ic,
            "naive_se": naive_se,
            "naive_t": naive_t,
        }
        for L in HAC_LAGS:
            hac_se = compute_hac_standard_error_of_mean(ic_series, maxlags=L)
            hac_t = mean_ic / hac_se if hac_se > 0 else np.nan
            row[f"hac_se_L{L}"] = hac_se
            row[f"hac_t_L{L}"] = hac_t
        rows.append(row)

    return pd.DataFrame(rows).set_index("signal")


def _sharpe_bootstrap_analysis(returns_df: pd.DataFrame) -> pd.DataFrame:
    """Per series (gross, net): point Sharpe, asymptotic CI, bootstrap CI."""
    rows = []
    for col in ("gross_return", "net_return"):
        rets = returns_df[col].to_numpy(dtype=np.float64)
        n = len(rets)
        mean = float(rets.mean())
        std = float(rets.std(ddof=1))
        if std == 0:
            continue

        point_sharpe = mean / std
        annualized_sharpe = point_sharpe * np.sqrt(ANNUALIZATION)

        # Asymptotic 95% CI: SE_SR = sqrt((1 + 0.5 SR^2) / n)
        asy_se = np.sqrt((1 + 0.5 * point_sharpe ** 2) / n)
        asy_se_annual = asy_se * np.sqrt(ANNUALIZATION)
        asy_lower = annualized_sharpe - 1.96 * asy_se_annual
        asy_upper = annualized_sharpe + 1.96 * asy_se_annual

        # Block-bootstrap CI on monthly Sharpe, then annualize
        boot_lower_m, boot_upper_m = compute_bootstrapped_sharpe_ci(
            rets,
            block_size=BLOCK_SIZE,
            n_resamples=N_RESAMPLES,
            confidence=0.95,
            seed=SEED,
        )
        boot_lower = boot_lower_m * np.sqrt(ANNUALIZATION)
        boot_upper = boot_upper_m * np.sqrt(ANNUALIZATION)

        rows.append({
            "series": col,
            "n_periods": n,
            "mean_monthly": mean,
            "std_monthly": std,
            "sharpe_annual": annualized_sharpe,
            "asy_ci_lower": asy_lower,
            "asy_ci_upper": asy_upper,
            "boot_ci_lower": boot_lower,
            "boot_ci_upper": boot_upper,
        })
    return pd.DataFrame(rows).set_index("series")


def _print_ic_summary(ic_results: pd.DataFrame) -> None:
    print("=" * 80)
    print("v2 Item 3 — IC t-stats: naive vs HAC-corrected")
    print("=" * 80)
    print(f"\n(N = {int(ic_results['n_periods'].iloc[0])} monthly periods, Spearman IC)\n")
    print(f"{'Signal':<10} {'Mean IC':>10} {'Naive t':>10} "
          f"{'HAC t (L=3)':>14} {'HAC t (L=5)':>14}")
    print("-" * 60)
    for signal, row in ic_results.iterrows():
        print(f"{signal:<10} {row['mean_ic']:>10.4f} {row['naive_t']:>10.3f} "
              f"{row['hac_t_L3']:>14.3f} {row['hac_t_L5']:>14.3f}")

    # Highlight changes in significance
    print()
    for signal, row in ic_results.iterrows():
        naive = row["naive_t"]
        hac3 = row["hac_t_L3"]
        hac5 = row["hac_t_L5"]
        change_pct_3 = 100 * (abs(hac3) - abs(naive)) / abs(naive)
        change_pct_5 = 100 * (abs(hac5) - abs(naive)) / abs(naive)
        print(f"  {signal}: HAC vs naive t-stat change: "
              f"L=3 {change_pct_3:+.1f}%,  L=5 {change_pct_5:+.1f}%")

    # Significance against HLZ threshold
    print()
    print("Significance against Harvey-Liu-Zhu (2016) threshold t > 3.0:")
    for signal, row in ic_results.iterrows():
        marks = []
        for label, t in (("Naive", row["naive_t"]),
                         ("L=3", row["hac_t_L3"]),
                         ("L=5", row["hac_t_L5"])):
            marks.append(f"{label}: {'PASS' if abs(t) > 3.0 else 'fail'}")
        print(f"  {signal:<10}  " + "  |  ".join(marks))


def _print_sharpe_summary(sharpe_results: pd.DataFrame) -> None:
    print()
    print("=" * 80)
    print("v2 Item 3 — Sharpe CIs: asymptotic vs block-bootstrap")
    print("=" * 80)
    print(f"\n(N = {int(sharpe_results['n_periods'].iloc[0])} monthly returns; "
          f"block_size = {BLOCK_SIZE}, n_resamples = {N_RESAMPLES}, "
          f"annualized × √{ANNUALIZATION})\n")
    print(f"{'Series':<16} {'Sharpe':>8} {'Asymp 95% CI':>22} {'Bootstrap 95% CI':>26}")
    print("-" * 75)
    for series, row in sharpe_results.iterrows():
        asy = f"[{row['asy_ci_lower']:.3f}, {row['asy_ci_upper']:.3f}]"
        boot = f"[{row['boot_ci_lower']:.3f}, {row['boot_ci_upper']:.3f}]"
        print(f"{series:<16} {row['sharpe_annual']:>8.3f} {asy:>22} {boot:>26}")

    print()
    for series, row in sharpe_results.iterrows():
        asy_w = row["asy_ci_upper"] - row["asy_ci_lower"]
        boot_w = row["boot_ci_upper"] - row["boot_ci_lower"]
        diff_pct = 100 * (boot_w - asy_w) / asy_w
        print(f"  {series}: bootstrap CI width is {diff_pct:+.1f}% vs asymptotic")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading IC time series from {IC_PATH}...")
    ic_df = pd.read_parquet(IC_PATH)
    print(f"  {len(ic_df)} rows, {ic_df['signal'].nunique()} signals, "
          f"{ic_df['rebalance_date'].nunique()} periods")

    print(f"\nLoading portfolio returns from {RETURNS_PATH}...")
    returns_df = pd.read_parquet(RETURNS_PATH)
    print(f"  {len(returns_df)} monthly periods")

    print("\nRunning HAC analysis on IC t-stats...")
    ic_results = _ic_hac_analysis(ic_df)
    ic_results.to_parquet(OUTPUT_DIR / "ic_hac.parquet")

    print("Running bootstrap analysis on Sharpe CIs...")
    sharpe_results = _sharpe_bootstrap_analysis(returns_df)
    sharpe_results.to_parquet(OUTPUT_DIR / "sharpe_bootstrap.parquet")

    (OUTPUT_DIR / "run_metadata.txt").write_text(
        f"Run: {datetime.now().isoformat()}\n"
        f"HAC lags tested: {HAC_LAGS}\n"
        f"Bootstrap block_size: {BLOCK_SIZE}\n"
        f"Bootstrap n_resamples: {N_RESAMPLES}\n"
        f"Annualization factor: {ANNUALIZATION}\n"
        f"Seed: {SEED}\n"
        f"IC source: {IC_PATH}\n"
        f"Returns source: {RETURNS_PATH}\n"
    )

    _print_ic_summary(ic_results)
    _print_sharpe_summary(sharpe_results)
    print(f"\nSaved: {OUTPUT_DIR}/ic_hac.parquet")
    print(f"Saved: {OUTPUT_DIR}/sharpe_bootstrap.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())