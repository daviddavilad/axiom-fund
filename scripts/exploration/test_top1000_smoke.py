"""Smoke test for top-1000 universe backtest over 12 months.

Same as test_runner_12mo.py but with universe_size=1000. Catches scale
issues (cvxpy at 1000 assets, WRDS data pull size, optimizer infeasibility
at the small-cap tail) before committing to a full historical run.
"""
# ruff: noqa: I001

from __future__ import annotations

import logging
import os
import sys
import time

from axiom_fund import _warnings  # noqa: F401

from dotenv import load_dotenv

from axiom_fund.backtest.engine import run_historical_backtest


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

    import wrds
    db = wrds.Connection(wrds_username=username)
    start_time = time.time()

    try:
        df = run_historical_backtest(
            db,
            start_date="2020-01-01",
            end_date="2020-12-31",
            universe_size=1000,
            position_cap=0.005,  # 0.5% cap for top-1000 spec
            cache_dir="data/cache/backtest_top1000_smoke_v2",
        )

        elapsed = time.time() - start_time
        print()
        print("=" * 70)
        print(f"Smoke test complete in {elapsed/60:.1f} minutes")
        print("=" * 70)
        print(df[["realized_return", "n_names", "long_count", "short_count",
                  "gross_leverage", "net_exposure", "portfolio_beta",
                  "optimizer_status"]])
        print()

        n_periods = len(df)
        if n_periods == 0:
            print("ERROR: zero periods completed.")
            return 1

        # Quick stats
        cum_return = (1 + df["realized_return"]).prod() - 1
        avg_n_names = df["n_names"].mean()
        max_n_names = df["n_names"].max()
        min_n_names = df["n_names"].min()
        avg_gross = df["gross_leverage"].mean()

        print(f"Periods completed:   {n_periods}/12")
        print(f"N names:             min={min_n_names}, mean={avg_n_names:.1f}, max={max_n_names}")
        print(f"Avg gross leverage:  {avg_gross:.3f}")
        print(f"Cumulative return:   {cum_return*100:+.2f}%")
        print(f"Avg monthly return:  {df['realized_return'].mean()*100:+.4f}%")
        print(f"Std monthly return:  {df['realized_return'].std()*100:+.4f}%")

        # Compare to top-100 smoke baseline (run_runner_12mo from session ago)
        # That had cum -1.02%, hit rate 58.3%
        print()
        print("vs top-100 baseline (1.5% cap) for same 2020 window:")
        print(f"  Top-1000 0.5% cap:  {cum_return*100:+7.2f}% cum, {df['realized_return'].std()*100:.4f}% monthly std")
        print("  Top-1000 1.5% cap:  -11.13% cum, 4.1829% monthly std (previous run)")
        print("  Top-100  1.5% cap:   -1.02% cum, 1.3829% monthly std (Phase 4 baseline)")

        # Active positions diagnostic
        df['active'] = df['long_count'] + df['short_count']
        df['gross_per_active'] = df['gross_leverage'] / df['active']
        print()
        print("Active positions diagnostic:")
        print(f"  Active names per period: min={df['active'].min()}, "
              f"mean={df['active'].mean():.1f}, max={df['active'].max()}")
        print(f"  gross_per_active range: [{df['gross_per_active'].min():.5f}, "
              f"{df['gross_per_active'].max():.5f}]")
        print("  (0.005 = per-name cap binding)")

        # Net exposure / portfolio beta — should all be ~0
        max_abs_net = df["net_exposure"].abs().max()
        max_abs_beta = df["portfolio_beta"].abs().max()
        print()
        print("Neutrality (should all be ~1e-10 or smaller):")
        print(f"  Max |net exposure|: {max_abs_net:.2e}")
        print(f"  Max |portfolio beta|: {max_abs_beta:.2e}")

    finally:
        db.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
