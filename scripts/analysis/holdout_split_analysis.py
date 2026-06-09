"""Holdout split analysis — Analysis B of the holdout test design.

Slices the existing 116-period backtest into in-sample (2015–2022,
94 periods) and holdout (2023–2024, 22 periods) windows for all
three variants (3-signal, 4-signal, no-ResMom). Reports performance
metrics on each window separately.

This is the 'acknowledged-contamination' analysis per
docs/holdout_test_design.md: we acknowledge that the 2023-2024 data
has been seen during model development, but report the split as a
diagnostic anyway.

Output:
  data/cache/holdout_split_analysis/metrics_long.parquet
  data/cache/holdout_split_analysis/metrics_long.csv
  data/cache/holdout_split_analysis/run_metadata.txt

Wide-format summary printed to console.

Companion analysis (Analysis A — strict OOS) is in a separate
driver. Combined writeup in docs/holdout_test_results.md after
both are complete.
"""
# ruff: noqa: I001

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from axiom_fund import _warnings  # noqa: F401

from axiom_fund.backtest.metrics import compute_performance_metrics


HOLDOUT_CUTOFF = pd.Timestamp("2023-01-01")

OUTPUT_DIR = Path("data/cache/holdout_split_analysis")

# Variant config: which parquets to load
VARIANTS = [
    {
        "label": "3-sig (GP+IVol+ResMom)",
        "summary_path": "data/cache/backtest_full_top1000/backtest_summary.parquet",
        "net_path": "data/cache/backtest_full_top1000/net_returns.parquet",
    },
    {
        "label": "4-sig (GP+IVol+ResMom+PEAD)",
        "summary_path": "data/cache/backtest_full_top1000_4sig/backtest_summary.parquet",
        "net_path": None,  # no cost overlay on 4-sig
    },
    {
        "label": "no-ResMom (GP+IVol+PEAD)",
        "summary_path": "data/cache/backtest_full_top1000_no_resmom/backtest_summary.parquet",
        "net_path": None,  # no cost overlay on no-ResMom
    },
]


def _load_with_datetime_index(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df.sort_index()


def _metrics_row(
    returns: pd.Series, label: str, window: str, return_type: str,
) -> dict[str, object]:
    """Compute metrics for one (variant, window, return_type) and return a row."""
    metrics = compute_performance_metrics(returns, periods_per_year=12)
    return {
        "variant": label,
        "window": window,
        "return_type": return_type,
        "n_periods": metrics.n_periods,
        "start_date": metrics.start_date,
        "end_date": metrics.end_date,
        "years_covered": round(metrics.years_covered, 2),
        "cumulative_return": metrics.cumulative_return,
        "annualized_return": metrics.annualized_return,
        "annualized_vol": metrics.annualized_vol,
        "sharpe": metrics.sharpe.sharpe,
        "sharpe_ci_low": metrics.sharpe.ci_low,
        "sharpe_ci_high": metrics.sharpe.ci_high,
        "sharpe_se": metrics.sharpe.standard_error,
        "max_drawdown": metrics.max_drawdown,
        "hit_rate": metrics.hit_rate,
        "best_period": metrics.best_period,
        "worst_period": metrics.worst_period,
    }


def _print_wide_summary(df: pd.DataFrame) -> None:
    """Print a human-readable wide-format summary."""
    print()
    print("=" * 110)
    print("Holdout split summary (gross returns)")
    print("=" * 110)
    print(f"  In-sample: 2015-01 → 2022-12 (~94 periods)")
    print(f"  Holdout:   2023-01 → 2024-11 (~22 periods)")
    print()
    print(
        f'{"Variant":<32} {"Window":<10} {"N":>4} {"Cum":>9} {"Ann":>8} '
        f'{"Vol":>7} {"Sharpe":>7} {"Hit":>6} {"MaxDD":>8}'
    )
    print("-" * 110)
    gross = df[df["return_type"] == "gross"].sort_values(
        ["variant", "window"], key=lambda s: s.map({"in_sample": 0, "holdout": 1})
    )
    for _, row in gross.iterrows():
        print(
            f'{row["variant"]:<32} '
            f'{row["window"]:<10} '
            f'{row["n_periods"]:>4d} '
            f'{row["cumulative_return"] * 100:>+8.2f}% '
            f'{row["annualized_return"] * 100:>+7.2f}% '
            f'{row["annualized_vol"] * 100:>6.2f}% '
            f'{row["sharpe"]:>+7.3f} '
            f'{row["hit_rate"] * 100:>5.1f}% '
            f'{row["max_drawdown"] * 100:>+7.2f}%'
        )

    net = df[df["return_type"] == "net"]
    if len(net) > 0:
        print()
        print("Net returns (3-sig only — only variant with cost overlay)")
        print("-" * 110)
        net_sorted = net.sort_values(
            "window", key=lambda s: s.map({"in_sample": 0, "holdout": 1})
        )
        for _, row in net_sorted.iterrows():
            print(
                f'{row["variant"]:<32} '
                f'{row["window"]:<10} '
                f'{row["n_periods"]:>4d} '
                f'{row["cumulative_return"] * 100:>+8.2f}% '
                f'{row["annualized_return"] * 100:>+7.2f}% '
                f'{row["annualized_vol"] * 100:>6.2f}% '
                f'{row["sharpe"]:>+7.3f} '
                f'{row["hit_rate"] * 100:>5.1f}% '
                f'{row["max_drawdown"] * 100:>+7.2f}%'
            )


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []

    for variant in VARIANTS:
        if not os.path.exists(variant["summary_path"]):
            print(f"WARN: missing {variant['summary_path']}, skipping", file=sys.stderr)
            continue

        # Gross returns
        df = _load_with_datetime_index(variant["summary_path"])
        is_returns = df.loc[df.index < HOLDOUT_CUTOFF, "realized_return"]
        oos_returns = df.loc[df.index >= HOLDOUT_CUTOFF, "realized_return"]

        rows.append(_metrics_row(is_returns, variant["label"], "in_sample", "gross"))
        rows.append(_metrics_row(oos_returns, variant["label"], "holdout", "gross"))

        # Net returns (3-sig only)
        if variant["net_path"] and os.path.exists(variant["net_path"]):
            net_df = _load_with_datetime_index(variant["net_path"])
            net_is = net_df.loc[net_df.index < HOLDOUT_CUTOFF, "net_return"]
            net_oos = net_df.loc[net_df.index >= HOLDOUT_CUTOFF, "net_return"]
            rows.append(_metrics_row(net_is, variant["label"], "in_sample", "net"))
            rows.append(_metrics_row(net_oos, variant["label"], "holdout", "net"))

    df_out = pd.DataFrame(rows)

    # Save long format
    df_out.to_parquet(OUTPUT_DIR / "metrics_long.parquet", index=False)
    df_out.to_csv(OUTPUT_DIR / "metrics_long.csv", index=False)

    # Save run_metadata
    meta = OUTPUT_DIR / "run_metadata.txt"
    meta.write_text(
        f"Run: {datetime.now().isoformat()}\n"
        f"Holdout cutoff: {HOLDOUT_CUTOFF.date()}\n"
        f"Variants: {len(VARIANTS)}\n"
        f"Rows in metrics table: {len(df_out)}\n"
    )

    # Print wide summary
    _print_wide_summary(df_out)

    print()
    print(f"Saved metrics to: {OUTPUT_DIR}/metrics_long.parquet (+ .csv)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())