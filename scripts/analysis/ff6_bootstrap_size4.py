"""Block-bootstrap CI for Size4 L/S raw Sharpe and FF6 alpha.

Complements ff6_spanning_size4.py by producing CIs that don't depend on
HAC lag choice. Uses moving block bootstrap (Kunsch 1989) matching the
existing compute_bootstrapped_sharpe_ci implementation:
  1. Fixed block_size
  2. Sample block start indices uniformly with replacement
  3. Concatenate blocks up to length N of original
  4. Refit statistic on bootstrap sample
  5. Percentile CI + 2-sided empirical p-value from replicates

Both statistics computed jointly per replicate on the same block sample:
  - Raw Sharpe (mean / std) of the L/S returns (annualized in output)
  - FF6 alpha from OLS y ~ [const, mkt_rf, smb, hml, rmw, cma, mom]

Block size sensitivity: [3, 4, 6, 8]; standard is N^(1/3) ≈ 4 for N=70.

Prompted by 2026-07-24 FF6 spanning result: Size4 VW alpha significant
at HAC lag ≥6 (p=0.03-0.04), marginal at lag 4 (p=0.06). Block bootstrap
is a lag-independent robustness check.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


SORT_PATH = Path("data/cache/lazy_prices_backtest/size_quintile_by_lazy_prices_sort.parquet")
VW_EW_PATH = Path("data/cache/lazy_prices_backtest/size_quintile_vw_ew_compare.parquet")
FF6_PATH = Path("data/cache/ff6_monthly.parquet")
SIZE_BUCKET = 4
BLOCK_SIZES = [3, 4, 6, 8]
N_RESAMPLES = 10_000
CONFIDENCE = 0.95
SEED = 42
ANNUAL_FACTOR = np.sqrt(12)


def load_ls_series(weighting: str) -> pd.Series:
    if weighting == "ew":
        sort = pd.read_parquet(SORT_PATH)
        sort["date"] = pd.to_datetime(sort["date"])
        s = sort[sort.size_bucket == SIZE_BUCKET].copy()
        q1 = s[s.lp_quintile == 1].set_index("date")["mean"].rename("q1")
        q5 = s[s.lp_quintile == 5].set_index("date")["mean"].rename("q5")
        return (q1 - q5).rename("ls_return").dropna()
    elif weighting == "vw":
        vw = pd.read_parquet(VW_EW_PATH)
        vw["date"] = pd.to_datetime(vw["date"])
        s = vw[vw.size_bucket == SIZE_BUCKET].copy().sort_values("date")
        return s.set_index("date")["ls_vw"].dropna().rename("ls_return")
    else:
        raise ValueError(f"weighting must be 'ew' or 'vw', got {weighting!r}")


def _sample_block_indices(n: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    """Moving block bootstrap: uniformly sample block starts, concatenate,
    truncate to length n."""
    if block_size < 1 or block_size > n:
        raise ValueError(f"block_size {block_size} out of range for n={n}")
    n_blocks = -(-n // block_size)  # ceil
    starts = rng.integers(0, n - block_size + 1, size=n_blocks)
    blocks = [np.arange(s, s + block_size) for s in starts]
    return np.concatenate(blocks)[:n]


def _bootstrap_stats(
    y: np.ndarray,
    X: np.ndarray,
    block_size: int,
    n_resamples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Joint bootstrap of raw Sharpe and FF6 alpha.

    For each replicate:
      1. Sample block indices via moving block bootstrap
      2. Apply to y AND X (preserves row alignment / joint distribution)
      3. Compute Sharpe from y_boot
      4. Refit OLS to get alpha = intercept
    """
    n = len(y)
    if X.shape[0] != n:
        raise ValueError(f"y len {n} != X rows {X.shape[0]}")
    rng = np.random.default_rng(seed)
    sharpes = np.empty(n_resamples)
    alphas = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = _sample_block_indices(n, block_size, rng)
        y_b = y[idx]
        X_b = X[idx]
        sharpes[i] = y_b.mean() / y_b.std(ddof=1)
        beta, *_ = np.linalg.lstsq(X_b, y_b, rcond=None)
        alphas[i] = beta[0]
    return sharpes, alphas


def _two_sided_empirical_p(x: np.ndarray) -> float:
    """Bootstrap 2-sided empirical p-value for H0: statistic = 0."""
    return 2 * min((x <= 0).mean(), (x >= 0).mean())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--weighting", choices=["ew", "vw"], default="ew")
    p.add_argument("--n-resamples", type=int, default=N_RESAMPLES)
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args()

    print(f"Loading data (weighting={args.weighting.upper()})...")
    ls = load_ls_series(args.weighting)
    ff6 = pd.read_parquet(FF6_PATH)
    ff6["date"] = pd.to_datetime(ff6["date"])
    for c in ["mkt_rf", "smb", "hml", "rmw", "cma", "mom", "rf"]:
        ff6[c] = ff6[c] / 100.0

    joined = pd.DataFrame({"ls": ls}).join(
        ff6.set_index("date")[["mkt_rf", "smb", "hml", "rmw", "cma", "mom"]],
        how="inner",
    )
    print(f"  N months: {len(joined)}")

    y = joined["ls"].values
    X = np.column_stack([
        np.ones(len(joined)),
        joined[["mkt_rf", "smb", "hml", "rmw", "cma", "mom"]].values,
    ])

    # Point estimates
    beta_full, *_ = np.linalg.lstsq(X, y, rcond=None)
    alpha_point = beta_full[0]
    sharpe_point = float(y.mean() / y.std(ddof=1))

    print()
    print("=" * 90)
    print(f"Block bootstrap CI ({args.weighting.upper()}), N_resamples = {args.n_resamples:,}")
    print("=" * 90)
    print(f"Point estimates:")
    print(f"  Annualized Sharpe: {sharpe_point * ANNUAL_FACTOR:+.4f}")
    print(f"  Annualized alpha:  {alpha_point * 12 * 100:+.4f}%")
    print()

    print(f"{'Block':<8} {'Sharpe 95% CI (ann)':>22} {'Sharpe p':>10}"
          f"     {'Alpha 95% CI (ann %)':>26} {'Alpha p':>10}")
    print("-" * 90)

    for bs in BLOCK_SIZES:
        sharpes, alphas = _bootstrap_stats(y, X, bs, args.n_resamples, args.seed + bs)

        sharpe_lo_ann = np.percentile(sharpes, 2.5) * ANNUAL_FACTOR
        sharpe_hi_ann = np.percentile(sharpes, 97.5) * ANNUAL_FACTOR
        alpha_lo_ann = np.percentile(alphas, 2.5) * 12 * 100
        alpha_hi_ann = np.percentile(alphas, 97.5) * 12 * 100

        p_sharpe = _two_sided_empirical_p(sharpes)
        p_alpha = _two_sided_empirical_p(alphas)

        print(f"bs={bs:<5} [{sharpe_lo_ann:>+7.3f}, {sharpe_hi_ann:>+7.3f}]"
              f"    {p_sharpe:>7.4f}"
              f"     [{alpha_lo_ann:>+7.3f}%, {alpha_hi_ann:>+7.3f}%]"
              f"    {p_alpha:>7.4f}")

    print()
    print("Interpretation:")
    print("  - Sharpe CI excludes zero -> raw effect significant")
    print("  - Alpha CI excludes zero  -> FF6-adjusted alpha significant")
    print("  - Both consistent across block sizes -> lag-robust conclusion")

    out = Path(f"data/cache/lazy_prices_backtest/ff6_bootstrap_size4_{args.weighting}.parquet")
    result = pd.DataFrame({
        "weighting": [args.weighting],
        "n_months": [len(joined)],
        "sharpe_point_ann": [sharpe_point * ANNUAL_FACTOR],
        "alpha_point_ann": [alpha_point * 12],
    })
    result.to_parquet(out, index=False)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()