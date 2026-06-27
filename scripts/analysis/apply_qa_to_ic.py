"""Apply Quandt-Andrews structural-break test to v1's IC time series.

Per docs/v2_design.md Item 5, this script applies the QA sup-F test
to each of the four signals' monthly IC series, reporting whether
there is evidence of a mean shift at an unknown date.

The Item 3 analysis (commit 473234a) reported HAC-corrected naive
t-stats for the four signals computed assuming a constant mean IC.
QA asks a different question: is the mean IC actually constant, or
does it shift at some date within the sample?

Source data: `data/cache/ic_analysis_4sig/ic_long.parquet` (116
monthly Spearman IC values per signal across GP, IVol, ResMom,
PEAD; 2015-01 to 2024-11).

For each signal:
  - Regression specification: ic_monthly = intercept + epsilon
    (so X is a column of ones, m = 1)
  - Trimming π₀ = 0.15 (standard, candidate breaks searched over
    the middle 70% of the sample)
  - sup_F statistic, break date, and Hansen p-value reported

Output:
  data/cache/qa_v2/qa_results.parquet — per-signal sup_F, break
    date, p-value, pre-break and post-break sample means
  data/cache/qa_v2/run_metadata.txt

Aggregate summary printed to console.

Multiple-testing note: four signals are tested. Family-wise error
rate at α = 0.05 is approximately 1 - (1 - 0.05)^4 ≈ 0.19. The raw
p-values are reported; readers should apply Bonferroni or similar
correction at their preference. The findings document discusses
this explicitly.

This script is one-off; not part of the production backtest.
"""
# ruff: noqa: I001

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from axiom_fund.diagnostics.structural_break import compute_quandt_andrews


IC_PATH = Path("data/cache/ic_analysis_4sig/ic_long.parquet")
OUTPUT_DIR = Path("data/cache/qa_v2")
TRIMMING = 0.15


def _qa_for_signal(group: pd.DataFrame) -> dict:
    """Run QA on one signal's IC time series; return result row."""
    group = group.sort_values("rebalance_date").reset_index(drop=True)
    y = group["ic_spearman"].to_numpy(dtype=np.float64)
    n = len(y)
    X = np.ones((n, 1), dtype=np.float64)

    sup_f, break_idx, pvalue = compute_quandt_andrews(y, X, trimming=TRIMMING)

    # Pre- and post-break sample statistics for interpretive context
    pre = y[:break_idx]
    post = y[break_idx:]
    return {
        "signal": group["signal"].iloc[0],
        "n_periods": n,
        "sup_f": sup_f,
        "break_index": break_idx,
        "break_date": group["rebalance_date"].iloc[break_idx],
        "hansen_pvalue": pvalue,
        "mean_pre": float(pre.mean()) if len(pre) > 0 else np.nan,
        "mean_post": float(post.mean()) if len(post) > 0 else np.nan,
        "n_pre": len(pre),
        "n_post": len(post),
    }


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading IC time series from {IC_PATH}...")
    ic_df = pd.read_parquet(IC_PATH)
    print(f"  {len(ic_df)} rows, {ic_df['signal'].nunique()} signals, "
          f"{ic_df['rebalance_date'].nunique()} periods")

    print(f"\nRunning Quandt-Andrews per signal "
          f"(trimming = {TRIMMING}, m = 1)...")
    rows = [_qa_for_signal(group) for _, group in ic_df.groupby("signal")]
    results = pd.DataFrame(rows).sort_values("hansen_pvalue").reset_index(drop=True)
    results.to_parquet(OUTPUT_DIR / "qa_results.parquet", index=False)

    # Console summary
    print()
    print("=" * 80)
    print("v2 Item 5 — Quandt-Andrews structural-break test on IC series")
    print("=" * 80)
    print(f"\n(N = {int(results['n_periods'].iloc[0])} monthly periods per signal, "
          f"trimming = {TRIMMING})\n")
    print(f"{'Signal':<10} {'sup_F':>8} {'Break date':>13} "
          f"{'Mean pre':>10} {'Mean post':>10} {'p-value':>10}  Reject?")
    print("-" * 75)
    for _, row in results.iterrows():
        reject_raw = "yes" if row["hansen_pvalue"] < 0.05 else "no"
        reject_bonf = "yes" if row["hansen_pvalue"] < 0.05 / 4 else "no"
        print(
            f"{row['signal']:<10} {row['sup_f']:>8.3f} "
            f"{row['break_date'].strftime('%Y-%m-%d'):>13} "
            f"{row['mean_pre']:>10.4f} {row['mean_post']:>10.4f} "
            f"{row['hansen_pvalue']:>10.4f}  "
            f"raw:{reject_raw} bonf:{reject_bonf}"
        )

    print()
    print("Notes:")
    print(f"  - Family-wise error rate at alpha=0.05 across 4 signals: ~19%.")
    print(f"  - Bonferroni-adjusted threshold: 0.05/4 = 0.0125.")
    print(f"  - Break dates are 0-indexed positions in the sorted IC series.")

    (OUTPUT_DIR / "run_metadata.txt").write_text(
        f"Run: {datetime.now().isoformat()}\n"
        f"Source: {IC_PATH}\n"
        f"N periods per signal: {results['n_periods'].iloc[0]}\n"
        f"Trimming: {TRIMMING}\n"
        f"Restriction count m: 1 (intercept-only regression)\n"
        f"Signals tested: {sorted(results['signal'].tolist())}\n"
    )

    print(f"\nSaved: {OUTPUT_DIR}/qa_results.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())