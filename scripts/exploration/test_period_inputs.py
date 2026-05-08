"""Smoke test for _fetch_period_inputs.

Builds the backtest cache, then extracts inputs for a single rebalance date.
Verifies the inputs are well-formed and a backtest period can be run.
"""
# ruff: noqa: I001

from __future__ import annotations

import os
import sys

from axiom_fund import _warnings  # noqa: F401

import pandas as pd
from dotenv import load_dotenv

from axiom_fund.backtest.engine import (
    _build_cache,
    _fetch_period_inputs,
    run_backtest_period,
)


def main() -> int:
    load_dotenv()
    username = os.getenv("WRDS_USERNAME")
    if not username:
        print("ERROR: WRDS_USERNAME not set", file=sys.stderr)
        return 1

    import wrds
    db = wrds.Connection(wrds_username=username)

    try:
        # Single-period cache (proof of concept)
        rebalance_date = pd.Timestamp("2020-09-30")
        rebalance_dates = [rebalance_date]

        print("Building cache (1 rebalance date)...")
        cache = _build_cache(
            db,
            rebalance_dates=rebalance_dates,
            universe_size=100,
        )
        print(f"  Cache built: {len(cache.permnos_all)} permnos")
        print()

        print(f"Building inputs for {rebalance_date.strftime('%Y-%m-%d')}...")
        inputs = _fetch_period_inputs(cache, rebalance_date)

        if inputs is None:
            print("  ERROR: _fetch_period_inputs returned None")
            return 1

        print(f"  Alpha:           {len(inputs.alpha)} names")
        print(f"  Covariance:      {inputs.covariance.shape}")
        print(f"  Betas:           {len(inputs.betas)} (mean={inputs.betas.mean():.3f})")
        print(f"  Sectors:         {len(inputs.sectors)} ({inputs.sectors.nunique()} unique)")
        print(f"  Holding returns: {inputs.holding_period_returns.shape}")
        print()

        # Run backtest period
        print("Running single-period backtest from cache-derived inputs...")
        result = run_backtest_period(inputs)
        print()
        print("=" * 70)
        print("Result")
        print("=" * 70)
        print(f"Solver status:        {result.optimizer_status}")
        print(f"N names:              {result.n_names}")
        print(f"Long / Short:         {result.long_count} / {result.short_count}")
        print(f"Net exposure:         {result.net_exposure:+.6e}")
        print(f"Portfolio beta:       {result.portfolio_beta:+.6e}")
        print(f"Realized return:      {result.realized_return:+.4%}")

    finally:
        db.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())