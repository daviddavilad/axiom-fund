"""Apply v2 Item 4 deflated Sharpe ratio to v1's holdout variants.

Per docs/v2_design.md Item 4, this script recomputes v1's holdout
Sharpe inference under Bailey & López de Prado (2014) deflated
Sharpe ratio (DSR), which corrects for selection bias when reporting
the best of N strategy variants.

The three v1 holdout variants reported in docs/onepager.md:
  - 3-sig (GP+IVol+ResMom):     gross holdout Sharpe = 1.18
  - 4-sig (GP+IVol+ResMom+PEAD): gross holdout Sharpe = 1.44
  - no-ResMom (GP+IVol+PEAD):   gross holdout Sharpe = 1.77

These are the same hypothesis under different signal weightings, so
DSR is methodologically appropriate. The hard interpretive question
is N (the trial count). This script reports DSR at three N choices:

  N = 3: the published variants alone — most defensible narrow
         claim
  N = 7: rough estimate of total signal combinations considered
         during v1 development (subsets of {GP, IVol, ResMom, PEAD,
         equal vs IC weighting})
  N = 20: an upper-bound estimate including implicit choices
         (formation windows, lookback periods, optimizer constraints)

For N > 3, the function builds synthetic trial arrays preserving the
observed variance of the 3 actuals; this is the BLP-recommended way
to apply DSR when reported variants underestimate the true search.

The denominator uses the Mertens (2002) non-normality correction
with sample skewness and excess kurtosis from each variant's
holdout return series.

Output:
  data/cache/dsr_v2/dsr_results.parquet — DSR per variant × N
  data/cache/dsr_v2/run_metadata.txt

Source data: data/cache/backtest_full_top1000{_4sig,_no_resmom}/
results.csv for the three variants' realized_return time series.

This script is one-off; not part of the production backtest.
"""
# ruff: noqa: I001

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from axiom_fund.diagnostics.inference import (
    compute_deflated_sharpe,
    compute_expected_max_sharpe,
)


VARIANT_PATHS = {
    "3-sig": Path("data/cache/backtest_full_top1000/results.csv"),
    "4-sig": Path("data/cache/backtest_full_top1000_4sig/results.csv"),
    "no-ResMom": Path("data/cache/backtest_full_top1000_no_resmom/results.csv"),
}

HOLDOUT_START = "2023-01-01"  # Holdout begins 2023-01-31
OUTPUT_DIR = Path("data/cache/dsr_v2")

# Trial-count sensitivity analysis
N_TRIAL_CHOICES = [3, 7, 20]


def _load_holdout_returns(path: Path) -> np.ndarray:
    """Load monthly gross returns for the holdout window from a results.csv."""
    df = pd.read_csv(path, parse_dates=["rebalance_date"])
    holdout = df[df["rebalance_date"] >= HOLDOUT_START]
    return holdout["realized_return"].to_numpy(dtype=np.float64)


def _make_synthetic_trials(
    observed: np.ndarray, target_n: int,
) -> np.ndarray:
    """Build a length-target_n array with the same variance as observed.

    Constructs an array via standardized normal quantiles (deterministic)
    rescaled to match observed mean and standard deviation. This preserves
    what we observed (variance) while letting us see N-sensitivity.

    Honest about what this is: the variance estimate comes from the
    actual K observations; the count is a hypothetical N >= K. BLP
    discuss this exact construction when reported variants under-
    represent the true search space.
    """
    if target_n == len(observed):
        return observed.copy()
    target_std = observed.std(ddof=1)
    target_mean = observed.mean()
    # Equally-spaced standard normal quantiles, then standardize to exact
    # zero mean and unit variance, then scale to target.
    z = norm.ppf(np.linspace(0.05, 0.95, target_n))
    z = (z - z.mean()) / z.std(ddof=1)
    return target_mean + target_std * z


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load each variant's holdout monthly returns
    print("Loading holdout return series for v1 variants...")
    variant_returns: dict[str, np.ndarray] = {}
    variant_stats: dict[str, dict] = {}
    for name, path in VARIANT_PATHS.items():
        rets = _load_holdout_returns(path)
        variant_returns[name] = rets
        n = len(rets)
        mean = float(rets.mean())
        std = float(rets.std(ddof=1))
        sharpe_monthly = mean / std
        sharpe_annual = sharpe_monthly * np.sqrt(12)
        skew = float(pd.Series(rets).skew())
        ex_kurt = float(pd.Series(rets).kurtosis())  # pandas kurtosis is excess
        variant_stats[name] = {
            "n_obs": n,
            "mean_monthly": mean,
            "std_monthly": std,
            "sharpe_monthly": sharpe_monthly,
            "sharpe_annual": sharpe_annual,
            "skewness": skew,
            "excess_kurtosis": ex_kurt,
        }
        print(f"  {name}: n={n}, monthly Sharpe={sharpe_monthly:.4f}, "
              f"annual={sharpe_annual:.3f}, skew={skew:.3f}, ex_kurt={ex_kurt:.3f}")

    # Build the array of observed monthly Sharpes (used as trials)
    observed_monthly_sharpes = np.array([
        variant_stats[name]["sharpe_monthly"] for name in VARIANT_PATHS
    ])
    print(f"\nObserved trial Sharpes (monthly): {observed_monthly_sharpes}")
    print(f"  std across trials: {observed_monthly_sharpes.std(ddof=1):.4f}")

    # Compute DSR for each variant × each N choice
    print("\nComputing DSR per variant × N...")
    rows = []
    for variant_name, stats in variant_stats.items():
        for n_trials in N_TRIAL_CHOICES:
            synthetic = _make_synthetic_trials(observed_monthly_sharpes, n_trials)
            sr_star = compute_expected_max_sharpe(synthetic)
            dsr = compute_deflated_sharpe(
                sharpe_observed=stats["sharpe_monthly"],
                sharpe_trials=synthetic,
                n_obs=stats["n_obs"],
                skewness=stats["skewness"],
                excess_kurtosis=stats["excess_kurtosis"],
            )
            rows.append({
                "variant": variant_name,
                "n_trials": n_trials,
                "sharpe_monthly": stats["sharpe_monthly"],
                "sharpe_annual": stats["sharpe_annual"],
                "sr_star_monthly": sr_star,
                "sr_star_annual": sr_star * np.sqrt(12),
                "dsr": dsr,
                "skewness": stats["skewness"],
                "excess_kurtosis": stats["excess_kurtosis"],
                "n_obs": stats["n_obs"],
            })

    results = pd.DataFrame(rows)
    results.to_parquet(OUTPUT_DIR / "dsr_results.parquet", index=False)

    # Print summary table
    print()
    print("=" * 80)
    print("v2 Item 4 — Deflated Sharpe Ratio applied to v1 holdout variants")
    print("=" * 80)
    print(f"\n(Holdout window: 22 monthly periods, 2023-01 through 2024-11)\n")
    print(f"{'Variant':<12} {'Annual SR':>10} {'N':>4} "
          f"{'SR* (annual)':>13} {'DSR':>8}  Pass(0.95)")
    print("-" * 65)
    for _, row in results.iterrows():
        pass_95 = "yes" if row["dsr"] > 0.95 else "no"
        print(
            f"{row['variant']:<12} {row['sharpe_annual']:>10.3f} "
            f"{row['n_trials']:>4} {row['sr_star_annual']:>13.3f} "
            f"{row['dsr']:>8.4f}  {pass_95}"
        )

    (OUTPUT_DIR / "run_metadata.txt").write_text(
        f"Run: {datetime.now().isoformat()}\n"
        f"Holdout start: {HOLDOUT_START}\n"
        f"Variants: {list(VARIANT_PATHS)}\n"
        f"N choices: {N_TRIAL_CHOICES}\n"
        f"DSR threshold reported: 0.95\n"
        f"Sharpe convention: monthly mean / monthly std, annualized × √12\n"
        f"Moments: pandas Series.skew() and .kurtosis() (excess kurtosis)\n"
    )

    print(f"\nSaved: {OUTPUT_DIR}/dsr_results.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())