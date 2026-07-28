"""Post-2020 AI-era subsample analysis on NYSE Size5 L/S.

Splits the 71-month NYSE Size5 L/S monthly series at two cutoffs to
test whether the mega-cap reversal + Size5 EW anomaly is specifically
an AI-era phenomenon or persists across regimes:

  Cutoff 1 (primary): 2022-11-30 (ChatGPT release, cultural landmark)
  Cutoff 2 (robustness): 2023-05-31 (Nvidia AI-earnings inflection)

Reports per-subsample:
  - Raw Sharpe (EW + VW) with asymptotic 95% CI
  - Mean monthly return, annualized return
  - Simple OLS FF6 alpha (no HAC — too little data for stable HAC on N~35)
  - N months per subsample

Substantive framing: N=34-37 per half. CIs will be wide. This is a
DESCRIPTIVE diagnostic to compare point-estimate direction and magnitude,
not a rigorous statistical test.

Prompted by 2026-07-27 session decision: paper story arc is complete
on axiom-fund's approximation (Size5 EW FF6 alpha +5.70%/yr t=3.04),
now testing whether the effect is regime-dependent (AI-era specific)
or persistent across the sample.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


SERIES_PATH = Path("data/cache/lazy_prices_backtest/nyse_size5_ls_series.parquet")
FF6_PATH = Path("data/cache/ff6_monthly.parquet")

CUTOFFS = [
    ("ChatGPT release", "2022-11-30"),
    ("Nvidia earnings inflection", "2023-05-31"),
]
ANNUAL_FACTOR = np.sqrt(12)


def _summarize_subsample(
    series: pd.DataFrame,
    ff6: pd.DataFrame,
    label: str,
    weighting: str,
) -> dict:
    """Compute Sharpe + simple FF6 alpha for a subsample."""
    col = f"ls_{weighting}"
    x = series[col].dropna()
    n = len(x)
    if n < 5:
        return {
            "label": label, "weighting": weighting, "n": n,
            "mean_monthly": np.nan, "annualized_ret": np.nan,
            "std_monthly": np.nan, "sharpe_ann": np.nan,
            "sharpe_ci_low": np.nan, "sharpe_ci_high": np.nan,
            "alpha_monthly": np.nan, "alpha_annual": np.nan,
            "alpha_t": np.nan,
        }
    mean_m = x.mean()
    std_m = x.std(ddof=1)
    sharpe_monthly = mean_m / std_m
    sharpe_ann = sharpe_monthly * ANNUAL_FACTOR
    # Lo (2002) asymptotic SE for periodic Sharpe
    se_sharpe_m = np.sqrt((1 + sharpe_monthly ** 2 / 2) / n)
    se_sharpe_ann = se_sharpe_m * ANNUAL_FACTOR
    ci_low = sharpe_ann - 1.96 * se_sharpe_ann
    ci_high = sharpe_ann + 1.96 * se_sharpe_ann

    # Simple OLS FF6 alpha
    dates = series["date"]
    ff6_local = ff6.set_index("date").loc[dates.values, ["mkt_rf", "smb", "hml", "rmw", "cma", "mom"]].reset_index(drop=True)
    y = x.values
    X = np.column_stack([np.ones(n), ff6_local.values])
    beta, resid_ss, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    resid = y - y_hat
    sigma2 = (resid ** 2).sum() / (n - X.shape[1])
    cov_beta = sigma2 * np.linalg.inv(X.T @ X)
    se_alpha_m = np.sqrt(cov_beta[0, 0])
    alpha_t = beta[0] / se_alpha_m if se_alpha_m > 0 else np.nan

    return {
        "label": label, "weighting": weighting, "n": n,
        "mean_monthly": mean_m,
        "annualized_ret": (1 + mean_m) ** 12 - 1,
        "std_monthly": std_m,
        "sharpe_ann": sharpe_ann,
        "sharpe_ci_low": ci_low, "sharpe_ci_high": ci_high,
        "alpha_monthly": beta[0],
        "alpha_annual": beta[0] * 12,
        "alpha_t": alpha_t,
    }


def _print_row(r: dict) -> None:
    print(f"  {r['label']:<20} {r['weighting'].upper():<3} "
          f"N={r['n']:<3} "
          f"raw ann={r['annualized_ret']*100:>+7.2f}% "
          f"Sharpe={r['sharpe_ann']:>+6.3f} "
          f"CI [{r['sharpe_ci_low']:>+.2f}, {r['sharpe_ci_high']:>+.2f}]  "
          f"alpha ann={r['alpha_annual']*100:>+7.2f}% t={r['alpha_t']:>+5.2f}")


def main() -> None:
    print("Loading data...")
    series = pd.read_parquet(SERIES_PATH)
    ff6 = pd.read_parquet(FF6_PATH)

    series["date"] = pd.to_datetime(series["date"])
    ff6["date"] = pd.to_datetime(ff6["date"])

    # Factors in decimal
    for c in ["mkt_rf", "smb", "hml", "rmw", "cma", "mom", "rf"]:
        ff6[c] = ff6[c] / 100.0

    print(f"  L/S series: {len(series)} months, "
          f"{series.date.min().date()} to {series.date.max().date()}")
    print()

    # Full sample baseline
    print("=" * 100)
    print("FULL SAMPLE (baseline)")
    print("=" * 100)
    for weighting in ["ew", "vw"]:
        r = _summarize_subsample(series, ff6, "Full 2020-2025", weighting)
        _print_row(r)

    # Each cutoff produces pre + post subsamples
    for label, cutoff_str in CUTOFFS:
        cutoff = pd.Timestamp(cutoff_str)
        pre = series[series.date <= cutoff].reset_index(drop=True)
        post = series[series.date > cutoff].reset_index(drop=True)
        print()
        print("=" * 100)
        print(f"CUTOFF: {label} ({cutoff.date()})")
        print(f"  Pre-cutoff:  {len(pre)} months ({pre.date.min().date() if len(pre) else 'N/A'} "
              f"to {pre.date.max().date() if len(pre) else 'N/A'})")
        print(f"  Post-cutoff: {len(post)} months ({post.date.min().date() if len(post) else 'N/A'} "
              f"to {post.date.max().date() if len(post) else 'N/A'})")
        print("=" * 100)
        for weighting in ["ew", "vw"]:
            r_pre = _summarize_subsample(pre, ff6, "PRE", weighting)
            _print_row(r_pre)
        for weighting in ["ew", "vw"]:
            r_post = _summarize_subsample(post, ff6, "POST", weighting)
            _print_row(r_post)

    print()
    print("=" * 100)
    print("Notes:")
    print("  - N=34-37 per half. CIs are wide. Focus on point-estimate direction/magnitude.")
    print("  - Alpha t-stats without HAC (too little data for stable HAC). Interpret loosely.")
    print("  - Simple OLS residual variance for alpha SE; not robust to heteroskedasticity.")
    print("=" * 100)


if __name__ == "__main__":
    main()