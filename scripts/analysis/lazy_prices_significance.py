"""Statistical significance of the Lazy Prices L/S Sharpe.

Runs bootstrap CI and HAC t-stat on the monthly L/S return series
from commit cc4b78c's backtest. Reports:
  1. Point Sharpe (recomputed for sanity)
  2. HAC t-stat with Newey-West lags = 4 (primary) + 6, 8 (sensitivity)
  3. Block-bootstrap 95% CI (annualized), block_size = 4 (N^(1/3))
     with sensitivity across [3, 4, 6, 8]

Interpretation:
  - Bootstrap CI excludes zero -> significant regardless of distribution
  - HAC |t| > 1.96 -> significant under asymptotic normality
  - Both together -> strong evidence against the null of zero mean
"""

from __future__ import annotations

from math import erf, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

from axiom_fund.diagnostics.inference import (
    compute_bootstrapped_sharpe_ci,
    compute_hac_standard_error_of_mean,
)


BACKTEST = Path("data/cache/lazy_prices_backtest")
SEED = 42
ANNUAL_FACTOR = np.sqrt(12)


def two_sided_normal_pvalue(t: float) -> float:
    """P-value for |t| under standard normal, two-sided."""
    return 1.0 - erf(abs(t) / sqrt(2.0))


def main() -> None:
    monthly = pd.read_parquet(BACKTEST / "monthly_returns.parquet")
    ls = monthly["ls_return"].dropna()
    ls_arr = np.asarray(ls.values, dtype=np.float64)
    n = len(ls_arr)

    mean_monthly = float(ls_arr.mean())
    std_monthly = float(ls_arr.std(ddof=1))
    sharpe_monthly = mean_monthly / std_monthly
    sharpe_annual = sharpe_monthly * ANNUAL_FACTOR
    mean_annual = mean_monthly * 12
    vol_annual = std_monthly * ANNUAL_FACTOR

    print("=" * 70)
    print("L/S monthly return series")
    print("=" * 70)
    print(f"  N months:            {n}")
    print(f"  Date range:          {monthly.date.min().date()} "
          f"to {monthly.date.max().date()}")
    print(f"  Mean/month:          {mean_monthly:+.4%}")
    print(f"  Std/month:           {std_monthly:.4%}")
    print(f"  Annualized return:   {mean_annual:+.4%}")
    print(f"  Annualized vol:      {vol_annual:.4%}")
    print(f"  Sharpe (annualized): {sharpe_annual:+.4f}")

    print()
    print("=" * 70)
    print("HAC t-stat (Newey-West autocorrelation-robust SE)")
    print("=" * 70)
    print(f"  H0: mean L/S return = 0")
    print(f"  H1: mean L/S return != 0 (two-sided)")
    print()
    for lags in [4, 6, 8]:
        se_monthly = compute_hac_standard_error_of_mean(ls_arr, maxlags=lags)
        t = mean_monthly / se_monthly
        pval = two_sided_normal_pvalue(t)
        se_annual = se_monthly * 12  # SE of annualized mean
        marker = "  ***" if pval < 0.01 else ("  **" if pval < 0.05 else "")
        print(f"  lags={lags}: SE(month)={se_monthly:.5f}  "
              f"t-stat={t:+.3f}  p-value={pval:.4f}{marker}")

    print()
    print("=" * 70)
    print("Block-bootstrap 95% CI for Sharpe (annualized)")
    print("=" * 70)
    print(f"  N resamples: 10,000  seed: {SEED}")
    print()
    for block_size in [3, 4, 6, 8]:
        lo_m, hi_m = compute_bootstrapped_sharpe_ci(
            ls_arr,
            block_size=block_size,
            n_resamples=10_000,
            confidence=0.95,
            seed=SEED,
        )
        lo_a = lo_m * ANNUAL_FACTOR
        hi_a = hi_m * ANNUAL_FACTOR
        excludes_zero = (lo_a * hi_a) > 0
        marker = "  (excludes 0)" if excludes_zero else "  (includes 0)"
        print(f"  block_size={block_size}: "
              f"[{lo_a:+.3f}, {hi_a:+.3f}]{marker}")

    print()
    print("For comparison: asymptotic 95% CI (from metrics.py, "
          "assumes iid returns): [-0.75, -0.21]")


if __name__ == "__main__":
    main()