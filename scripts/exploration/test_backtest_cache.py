"""Smoke test for the backtest data cache.

Pulls a small backtest window's worth of data and inspects what came back.
"""
# ruff: noqa: I001

from __future__ import annotations

import os
import sys

from axiom_fund import _warnings  # noqa: F401

import pandas as pd
from dotenv import load_dotenv

from axiom_fund.backtest.engine import _build_cache


def main() -> int:
    load_dotenv()
    username = os.getenv("WRDS_USERNAME")
    if not username:
        print("ERROR: WRDS_USERNAME not set", file=sys.stderr)
        return 1

    import wrds
    db = wrds.Connection(wrds_username=username)

    try:
        # 3 rebalance dates: end of Q3 2020, Q4 2020, Q1 2021
        rebalance_dates = [
            pd.Timestamp("2020-09-30"),
            pd.Timestamp("2020-10-30"),
            pd.Timestamp("2020-11-30"),
        ]

        print(f"Building backtest cache for {len(rebalance_dates)} rebalance dates...")
        print("(Pulling universe + returns + fundamentals + FF in single calls each.)")
        print()

        cache = _build_cache(
            db,
            rebalance_dates=rebalance_dates,
            universe_size=100,
        )

        # Diagnostics
        print("=" * 70)
        print("Cache contents")
        print("=" * 70)
        print(f"Rebalance dates: {len(cache.rebalance_dates)}")
        print(f"Total unique permnos in cache: {len(cache.permnos_all)}")
        print(f"Universe panel rows: {len(cache.universe_panel):,} "
              f"(should be ~{100 * len(rebalance_dates)} = "
              f"{100 * len(rebalance_dates)} if no dups)")
        print()
        print(f"Returns: {len(cache.returns_full):,} rows")
        print(f"  Date range: "
              f"{cache.returns_full['date'].min()} to "
              f"{cache.returns_full['date'].max()}")
        print(f"  Permnos: {cache.returns_full['permno'].nunique()}")
        print()
        print(f"Fundamentals: {len(cache.fundamentals_full):,} rows")
        print(f"  rdq range: "
              f"{cache.fundamentals_full['rdq'].min()} to "
              f"{cache.fundamentals_full['rdq'].max()}")
        print()
        print(f"FF factors: {len(cache.ff_full):,} rows")
        print(f"  Date range: "
              f"{cache.ff_full['date'].min()} to "
              f"{cache.ff_full['date'].max()}")

        # Sanity check: per-rebalance universe size
        print()
        print("--- Universe per rebalance date ---")
        for rdate in rebalance_dates:
            count = (cache.universe_panel["date"] == rdate).sum()
            print(f"  {rdate.strftime('%Y-%m-%d')}: {count} permnos")

    finally:
        db.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
