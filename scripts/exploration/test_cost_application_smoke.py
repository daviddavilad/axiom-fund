"""Smoke test for cost application on a single backtest period.

Verifies the data fetch, per-name stats, and cost computation work
end-to-end on one real period (2020-09-30) from the existing top-1000
backtest.
"""
# ruff: noqa: I001

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

from axiom_fund import _warnings  # noqa: F401

from dotenv import load_dotenv

from axiom_fund.backtest.cost_application import apply_costs_to_period


def main() -> int:
    load_dotenv()
    username = os.getenv("WRDS_USERNAME")
    if not username:
        print("ERROR: WRDS_USERNAME not set", file=sys.stderr)
        return 1

    backtest_dir = Path("data/cache/backtest_full_top1000")
    if not backtest_dir.exists():
        print(f"ERROR: {backtest_dir} not found", file=sys.stderr)
        return 1

    summary = pd.read_parquet(backtest_dir / "backtest_summary.parquet")
    if not isinstance(summary.index, pd.DatetimeIndex):
        summary.index = pd.to_datetime(summary.index)
    summary = summary.sort_index()

    # Pick a representative period: Sept 2020
    rebalance_date = pd.Timestamp("2020-09-30")
    if rebalance_date not in summary.index:
        print(f"ERROR: {rebalance_date} not in backtest summary")
        return 1

    weights_file = backtest_dir / f"weights_{rebalance_date.strftime('%Y-%m-%d')}.parquet"
    if not weights_file.exists():
        print(f"ERROR: {weights_file} not found")
        return 1

    weights_new_df = pd.read_parquet(weights_file)
    print(f"Weights file shape: {weights_new_df.shape}")
    print(f"Weights file columns: {list(weights_new_df.columns)}")
    print(f"Weights file head:")
    print(weights_new_df.head())
    print()

    # Treat this as the first rebalance (empty weights_old)
    weights_old = pd.Series(dtype="float64")
    if "weight" in weights_new_df.columns:
        weights_new = weights_new_df["weight"]
    else:
        weights_new = weights_new_df.iloc[:, 0]
    weights_new.index = weights_new.index.astype(int)

    print(f"Active positions: {(weights_new != 0).sum()}")
    print(f"Gross leverage: {weights_new.abs().sum():.4f}")
    print(f"Net exposure: {weights_new.sum():.4e}")
    print()

    gross_return = float(summary.loc[rebalance_date, "realized_return"])
    print(f"Gross return for {rebalance_date.date()}: {gross_return * 100:+.4f}%")
    print()

    import wrds
    db = wrds.Connection(wrds_username=username)
    try:
        print("Applying costs (this fetches market data and computes per-name stats)...")
        period = apply_costs_to_period(
            db=db,
            rebalance_date=rebalance_date,
            weights_old=weights_old,
            weights_new=weights_new,
            gross_return=gross_return,
            nav=1.0,
        )

        print()
        print("=" * 70)
        print(f"Cost breakdown for {period.rebalance_date.date()}:")
        print("=" * 70)
        print(f"  Gross return:    {period.gross_return * 100:+8.4f}%")
        print(f"  N trades:        {period.n_trades}")
        print()
        print(f"  Commission:      {period.commission_bps:8.4f} bps")
        print(f"  Spread:          {period.spread_bps:8.4f} bps")
        print(f"  Impact:          {period.impact_bps:8.4f} bps")
        print(f"  Short borrow:    {period.short_borrow_bps:8.4f} bps")
        print(f"  Total cost:      {period.cost_bps:8.4f} bps")
        print()
        print(f"  Net return:      {period.net_return * 100:+8.4f}%")
        print(f"  Cost drag:       {(period.gross_return - period.net_return) * 100:8.4f}%")

    finally:
        db.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())