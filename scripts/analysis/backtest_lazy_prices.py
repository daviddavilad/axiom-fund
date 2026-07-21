"""Backtest runner: Lazy Prices quintile-sort L/S portfolio.

Pipeline (all methodology pre-committed at docs/v2_item6_design.md
"Backtest scope (2026-07-12)"):
  1. Load aligned signal + universe + daily returns
  2. Build monthly rebalance calendar (2019-01-31 - 2024-12-31)
  3. Align raw signal to rebalance dates via signals.alignment.align_signal
     (annual carry rule via max_age_days=None)
  4. Assign quintiles per rebalance date via
     backtest.quintile_sort.assign_quintiles
  5. Compute 21-trading-day forward returns via
     backtest.forward_returns.compute_forward_returns
  6. Compute L/S monthly return series (long top quintile, short bottom
     quintile, equal-weighted within quintile) via
     backtest.quintile_sort.compute_long_short_returns
  7. Report Sharpe + hit rate + drawdowns via
     backtest.metrics.compute_performance_metrics

Universe (ad-hoc today, not pre-committed): union-set from
lazy_prices_universe.parquet, with each firm "in universe" from its
first_snapshot_date onward. Coarse but respects timing of firm's first
appearance. Production universe (per-date top-1000) deferred.

Outputs:
  data/cache/lazy_prices_backtest/monthly_returns.parquet
  data/cache/lazy_prices_backtest/summary.txt
  data/cache/lazy_prices_backtest/quintile_positions.parquet
"""
# ruff: noqa: I001

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from axiom_fund import _warnings  # noqa: F401
from axiom_fund.backtest.forward_returns import compute_forward_returns
from axiom_fund.backtest.metrics import compute_performance_metrics
from axiom_fund.backtest.quintile_sort import (
    assign_quintiles,
    compute_long_short_returns,
)
from axiom_fund.data.returns import ReturnsPanel
from axiom_fund.signals.alignment import align_signal


SIGNAL_PATH = Path("data/cache/lazy_prices_signal.parquet")
UNIVERSE_PATH = Path("data/cache/lazy_prices_universe.parquet")
RETURNS_CACHE = Path("data/cache/lazy_prices_returns_daily.parquet")

# Output paths built in main() based on weighting mode

# Rebalance calendar: month-end dates 2019-01 through 2024-12
REBALANCE_START = "2019-01-31"
# Extended 2026-07-20: dsf_v2 has returns through 2025-12-31, so last valid
# rebalance with 21-day forward window is 2025-11-30.
REBALANCE_END = "2025-11-30"

# Forward-return holding window matches monthly rebalance
HOLDING_DAYS = 21


def _fetch_or_load_returns(permnos: list[int], start: str, end: str) -> pd.DataFrame:
    """Load cached daily returns if available; else fetch from WRDS and cache."""
    if RETURNS_CACHE.exists():
        print(f"  Loading cached returns from {RETURNS_CACHE}")
        return pd.read_parquet(RETURNS_CACHE)

    print(f"  Cache miss. Fetching from WRDS ({len(permnos)} permnos)...")
    import wrds
    load_dotenv()
    db = wrds.Connection(wrds_username=os.environ["WRDS_USERNAME"])
    try:
        rp = ReturnsPanel(db)
        returns = rp.fetch(permnos=permnos, start_date=start, end_date=end)
    finally:
        db.close()

    RETURNS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    returns.to_parquet(RETURNS_CACHE, index=False)
    print(f"  Cached to {RETURNS_CACHE} ({len(returns):,} rows)")
    return returns


def _build_universe_panel(
    universe_union: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Build per-date universe using 'first_snapshot_date onward' rule.

    Each firm is 'in universe' at every rebalance date >= its
    first_snapshot_date. Coarse but respects timing.
    """
    universe_union = universe_union.copy()
    universe_union["first_snapshot_date"] = pd.to_datetime(
        universe_union["first_snapshot_date"]
    )
    rows = []
    for date in rebalance_dates:
        in_universe = universe_union[universe_union["first_snapshot_date"] <= date]
        rows.append(pd.DataFrame({
            "date": date,
            "permno": in_universe["permno"].values,
        }))
    return pd.concat(rows, ignore_index=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Lazy Prices L/S backtest"
    )
    parser.add_argument(
        "--value-weighted",
        action="store_true",
        help="Value-weight legs by market cap at rebalance date "
        "(default: equal-weight)",
    )
    args = parser.parse_args()

    # Output paths + label depend on weighting mode
    if args.value_weighted:
        output_dir = Path("data/cache/lazy_prices_backtest_vw")
        weighting_label = "value-weighted"
    else:
        output_dir = Path("data/cache/lazy_prices_backtest")
        weighting_label = "equal-weighted"
    monthly_returns_path = output_dir / "monthly_returns.parquet"
    positions_path = output_dir / "quintile_positions.parquet"
    summary_path = output_dir / "summary.txt"

    for path in (SIGNAL_PATH, UNIVERSE_PATH):
        if not path.exists():
            print(f"ERROR: {path} not found.", file=sys.stderr)
            return 1

    print(f"Loading signal: {SIGNAL_PATH}")
    signal = pd.read_parquet(SIGNAL_PATH)
    print(f"  {len(signal):,} rows, {signal.permno.nunique()} permnos")

    print(f"Loading universe union: {UNIVERSE_PATH}")
    universe_union = pd.read_parquet(UNIVERSE_PATH)
    print(f"  {len(universe_union):,} firms")

    # Build rebalance calendar (monthly)
    rebalance_dates = pd.date_range(REBALANCE_START, REBALANCE_END, freq="ME")
    print(f"Rebalance calendar: {len(rebalance_dates)} monthly dates "
          f"({REBALANCE_START} to {REBALANCE_END})")

    # Fetch daily returns for all union permnos (covers full sample)
    print(f"Fetching daily returns:")
    returns = _fetch_or_load_returns(
        permnos=universe_union["permno"].tolist(),
        start="2019-01-01",
        end="2025-12-31",  # dsf_v2 max date
    )
    print(f"  {len(returns):,} return rows")

    # Build per-date universe panel (ad-hoc coarse rule)
    print("Building universe panel (first_snapshot_date onward rule)...")
    universe_panel = _build_universe_panel(universe_union, rebalance_dates)
    print(f"  {len(universe_panel):,} (date, permno) rows")

    # Align raw signal to rebalance dates
    print("Aligning signal (max_age_days=None -> annual carry)...")
    signal_minimal = signal[["permno", "date_filed", "raw_signal"]]
    aligned = align_signal(
        raw_signal_df=signal_minimal,
        universe_df=universe_panel,
        rebalance_dates=list(rebalance_dates),
        max_age_days=None,
    )
    print(f"  {len(aligned):,} aligned rows")

    # Quintile assignment
    print("Assigning quintiles...")
    quintiles = assign_quintiles(aligned, n_quintiles=5)
    print(f"  {len(quintiles):,} quintile rows")

    # Forward returns per (rebalance_date, permno)
    print(f"Computing {HOLDING_DAYS}-day forward returns...")
    fwd = compute_forward_returns(
        returns_df=returns,
        rebalance_dates=list(rebalance_dates),
        holding_days=HOLDING_DAYS,
    )
    print(f"  {len(fwd):,} forward-return rows")

    # L/S portfolio
    print(f"Computing L/S portfolio returns ({weighting_label})...")
    if args.value_weighted:
        # Weights = market cap at each rebalance date
        weights_df = (
            returns.loc[
                returns["date"].isin(rebalance_dates),
                ["date", "permno", "marketcap"],
            ]
            .rename(columns={"marketcap": "weight"})
            .copy()
        )
        print(f"  Weight rows: {len(weights_df):,}")
        ls_returns = compute_long_short_returns(
            quintiles, fwd, n_quintiles=5, weights_df=weights_df,
        )
    else:
        ls_returns = compute_long_short_returns(
            quintiles, fwd, n_quintiles=5
        )
    print(f"  {len(ls_returns):,} monthly return rows")

    # Save monthly return artifact
    output_dir.mkdir(parents=True, exist_ok=True)
    ls_returns.to_parquet(monthly_returns_path, index=False)
    print(f"Saved: {monthly_returns_path}")

    # Also save quintile positions for downstream analysis
    quintiles.to_parquet(positions_path, index=False)
    print(f"Saved: {positions_path}")

    # Performance metrics
    print()
    print("=" * 70)
    print(f"Performance summary ({weighting_label})")
    print("=" * 70)

    ls_series = ls_returns.set_index("date")["ls_return"].dropna()
    long_series = ls_returns.set_index("date")["long_return"].dropna()
    short_series = ls_returns.set_index("date")["short_return"].dropna()

    ls_metrics = compute_performance_metrics(ls_series)
    long_metrics = compute_performance_metrics(long_series)
    short_metrics = compute_performance_metrics(short_series)

    def _print_metrics(label: str, m, series: pd.Series) -> None:
        print(f"\n{label} (N={len(series)} months)")
        print(f"  Annualized return: {m.annualized_return:.2%}")
        print(f"  Annualized vol:    {m.annualized_vol:.2%}")
        print(f"  Sharpe:            {m.sharpe.sharpe:.3f}  "
              f"(95% CI: [{m.sharpe.ci_low:.3f}, {m.sharpe.ci_high:.3f}])")
        print(f"  Hit rate:          {m.hit_rate:.1%}")
        print(f"  Max drawdown:      {m.max_drawdown:.2%}")

    _print_metrics("Long-only (top quintile)", long_metrics, long_series)
    _print_metrics("Short-only (bottom quintile)", short_metrics, short_series)
    _print_metrics("Long-Short", ls_metrics, ls_series)

    # Metadata summary
    with open(summary_path, "w") as f:
        f.write(f"Run: {datetime.now().isoformat()}\n")
        f.write(f"Weighting: {weighting_label}\n")
        f.write(f"Rebalance dates: {len(rebalance_dates)}\n")
        f.write(f"L/S N months: {len(ls_series)}\n")
        f.write(f"L/S Sharpe: {ls_metrics.sharpe.sharpe:.4f}\n")
        f.write(f"L/S Sharpe 95% CI: "
                f"[{ls_metrics.sharpe.ci_low:.4f}, {ls_metrics.sharpe.ci_high:.4f}]\n")
        f.write(f"L/S annualized return: {ls_metrics.annualized_return:.4f}\n")
        f.write(f"L/S annualized vol: {ls_metrics.annualized_vol:.4f}\n")
        f.write(f"L/S hit rate: {ls_metrics.hit_rate:.4f}\n")
        f.write(f"L/S max drawdown: {ls_metrics.max_drawdown:.4f}\n")

    print(f"\nSummary written to: {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())