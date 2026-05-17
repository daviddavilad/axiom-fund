"""Apply transaction costs to the full top-1000 backtest.

Reads the existing gross-of-cost backtest output and produces a
net-of-cost time series alongside it. Saves results to:
    data/cache/backtest_full_top1000/net_returns.parquet

Estimated runtime: ~30-60 minutes for 116 periods.
"""
# ruff: noqa: I001

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

from axiom_fund import _warnings  # noqa: F401

from dotenv import load_dotenv

from axiom_fund.backtest.cost_application import apply_costs_to_backtest


BACKTEST_DIR = Path("data/cache/backtest_full_top1000")


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    load_dotenv()
    username = os.getenv("WRDS_USERNAME")
    if not username:
        print("ERROR: WRDS_USERNAME not set", file=sys.stderr)
        return 1

    if not BACKTEST_DIR.exists():
        print(f"ERROR: {BACKTEST_DIR} not found", file=sys.stderr)
        return 1

    print("=" * 70)
    print("Applying transaction costs to full top-1000 backtest")
    print("=" * 70)
    print(f"Backtest directory: {BACKTEST_DIR}")
    print()

    start_time = time.time()

    import wrds
    db = wrds.Connection(wrds_username=username)
    try:
        result = apply_costs_to_backtest(
            db=db,
            backtest_dir=BACKTEST_DIR,
            nav=1.0,
        )
    finally:
        db.close()

    elapsed = time.time() - start_time
    print()
    print(f"Cost application complete in {elapsed/60:.1f} minutes")
    print(f"Periods processed: {len(result)}")
    print()

    # Save results
    result.to_parquet(BACKTEST_DIR / "net_returns.parquet")
    result.to_csv(BACKTEST_DIR / "net_returns.csv")
    print(f"Saved to {BACKTEST_DIR / 'net_returns.parquet'}")
    print()

    # Summary stats
    import pandas as pd
    import numpy as np

    gross = result["gross_return"]
    net = result["net_return"]
    cost = result["cost_bps"]

    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"  Gross cumulative:   {((1 + gross).prod() - 1) * 100:+8.2f}%")
    print(f"  Net cumulative:     {((1 + net).prod() - 1) * 100:+8.2f}%")
    print()
    print(f"  Gross monthly mean: {gross.mean() * 100:+8.4f}%")
    print(f"  Net monthly mean:   {net.mean() * 100:+8.4f}%")
    print(f"  Avg cost (bps/mo):  {cost.mean():8.2f}")
    print(f"  Total cost drag:    {(cost.mean() / 100) * 12:+8.2f}% per year")
    print()
    print(f"  Gross vol (ann):    {gross.std() * np.sqrt(12) * 100:8.2f}%")
    print(f"  Net vol (ann):      {net.std() * np.sqrt(12) * 100:8.2f}%")
    print()
    print(f"  Gross Sharpe:       {gross.mean() / gross.std() * np.sqrt(12):8.3f}")
    print(f"  Net Sharpe:         {net.mean() / net.std() * np.sqrt(12):8.3f}")
    print()

    # Cost component breakdown
    print("=" * 70)
    print("Cost component averages (bps/month):")
    print("=" * 70)
    print(f"  Commission:    {result['commission_bps'].mean():8.2f}")
    print(f"  Spread:        {result['spread_bps'].mean():8.2f}")
    print(f"  Impact:        {result['impact_bps'].mean():8.2f}")
    print(f"  Short borrow:  {result['short_borrow_bps'].mean():8.2f}")
    print(f"  Total:         {result['cost_bps'].mean():8.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())