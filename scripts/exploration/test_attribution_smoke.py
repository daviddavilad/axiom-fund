"""Smoke test for single-signal attribution on one rebalance period.

Builds the cache for a single date, computes the composite panel and
optimizer inputs, then runs the optimizer for each of the 4 signals
individually. Prints realized returns per signal and the sum, plus
the 4-signal composite realized return for comparison.

Goal: verify the math runs end-to-end before launching the full
116-period × 4-signal run tomorrow.
"""
# ruff: noqa: I001

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

from axiom_fund import _warnings  # noqa: F401

from dotenv import load_dotenv

from axiom_fund.backtest.attribution import (
    SIGNAL_SIGNS,
    _build_optimizer_inputs,
    compute_single_signal_period,
    rebuild_composite_for_date,
)
from axiom_fund.backtest.engine import _build_cache, _fetch_period_inputs, run_backtest_period
from axiom_fund.portfolio.covariance import estimate_covariance


def main() -> int:
    load_dotenv()
    username = os.getenv("WRDS_USERNAME")
    if not username:
        print("ERROR: WRDS_USERNAME not set", file=sys.stderr)
        return 1

    rebalance_date = pd.Timestamp("2020-09-30")
    print(f"Smoke test: single-signal attribution for {rebalance_date.date()}")
    print()

    import wrds
    db = wrds.Connection(wrds_username=username)
    try:
        # Build cache for just this one date with full top-1000 universe
        cache = _build_cache(
            db,
            rebalance_dates=[rebalance_date],
            universe_size=1000,
        )
        print(f"Cache: {len(cache.permnos_all)} permnos, "
              f"{len(cache.returns_full)} returns, "
              f"{len(cache.fundamentals_full)} fundamentals")
        print()

        # 1. Rebuild composite panel (with all 4 z-cols visible)
        print("Rebuilding composite panel...")
        composite = rebuild_composite_for_date(cache, rebalance_date)
        if composite is None or composite.empty:
            print("ERROR: composite is empty")
            return 1
        print(f"  N names with composite: {len(composite)}")
        print(f"  Per-signal non-null counts:")
        for sig in SIGNAL_SIGNS:
            print(f"    {sig}: {composite[sig].notna().sum()}")
        print()

        # 2. Build optimizer inputs (cov, betas, sectors, holding rets)
        print("Building optimizer inputs (cov/betas/sectors/holding rets)...")
        inputs = _build_optimizer_inputs(cache, rebalance_date)
        if inputs is None:
            print("ERROR: optimizer inputs unavailable")
            return 1
        cov_wide, betas_full, sector_map, holding_wide = inputs
        cov_estimate = estimate_covariance(cov_wide)
        print(f"  Cov matrix: {cov_estimate.matrix.shape}")
        print(f"  Betas: {len(betas_full.dropna())} non-null")
        print(f"  Sectors: {len(sector_map.dropna())} non-null")
        print(f"  Holding rets: {holding_wide.shape}")
        print()

        # 3. Run 4-signal composite baseline for comparison
        print("Running 4-signal composite (baseline)...")
        baseline_inputs = _fetch_period_inputs(cache, rebalance_date)
        if baseline_inputs is None:
            print("WARNING: baseline composite inputs unavailable")
            baseline_return: float | None = None
        else:
            baseline_result = run_backtest_period(baseline_inputs)
            baseline_return = float(baseline_result.realized_return)
            print(f"  4-signal realized return: {baseline_return * 100:+.4f}%")
        print()

        # 4. Run each single-signal portfolio
        print("Running single-signal portfolios...")
        single_results = {}
        for signal_name in SIGNAL_SIGNS:
            result = compute_single_signal_period(
                cache=cache,
                rebalance_date=rebalance_date,
                signal_name=signal_name,
                composite_panel=composite,
                cov_estimate=cov_estimate,
                betas_for_engine=betas_full,
                sectors_for_engine=sector_map,
                hpr_for_engine=holding_wide,
            )
            if result is None:
                print(f"  {signal_name}: SKIPPED")
                continue
            single_results[signal_name] = result
            print(
                f"  {signal_name}: {result.realized_return * 100:+8.4f}%  "
                f"({result.n_names} names, "
                f"L/S {result.long_count}/{result.short_count}, "
                f"gross {result.gross_leverage:.2f}, "
                f"{result.optimizer_status})"
            )

        print()
        print("=" * 70)
        print("Summary")
        print("=" * 70)
        sum_signals = sum(r.realized_return for r in single_results.values())
        print(f"  Sum of single-signal returns:  {sum_signals * 100:+.4f}%")
        if baseline_return is not None:
            print(f"  4-signal composite return:     {baseline_return * 100:+.4f}%")
            print(f"  Difference:                    "
                  f"{(sum_signals - baseline_return) * 100:+.4f}%")
            print()
            print("Note: sum ≠ composite is EXPECTED (optimizer nonlinear).")
            print("The interpretation is 'what each signal alone would have done'.")

    finally:
        db.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())