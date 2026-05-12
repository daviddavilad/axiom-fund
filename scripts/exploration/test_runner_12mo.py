"""Smoke test for the historical backtest runner over a 12-month window."""
# ruff: noqa: I001

from __future__ import annotations

import logging
import os
import sys

from axiom_fund import _warnings  # noqa: F401

from dotenv import load_dotenv

from axiom_fund.backtest.engine import run_historical_backtest


def main() -> int:
    # Set up logging so we can watch progress
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

    try:
        df = run_historical_backtest(
            db,
            start_date="2020-01-01",
            end_date="2020-12-31",
            universe_size=100,
            cache_dir="data/cache/backtest_smoke",
        )

        print()
        print("=" * 70)
        print("Backtest summary")
        print("=" * 70)
        print(df[["realized_return", "n_names", "long_count", "short_count",
                  "gross_leverage", "net_exposure", "portfolio_beta",
                  "optimizer_status"]])
        print()

        # Quick stats
        print(f"Periods: {len(df)}")
        print(f"Cumulative return: {((1 + df['realized_return']).prod() - 1) * 100:.2f}%")
        print(f"Avg monthly return: {df['realized_return'].mean() * 100:+.4f}%")
        print(f"Std monthly return: {df['realized_return'].std() * 100:+.4f}%")
        print(f"Hit rate: {(df['realized_return'] > 0).mean() * 100:.1f}%")
        if df['realized_return'].std() > 0:
            sharpe_monthly = df['realized_return'].mean() / df['realized_return'].std()
            sharpe_annualized = sharpe_monthly * (12 ** 0.5)
            print(f"Sharpe (annualized, gross of costs): {sharpe_annualized:.2f}")

    finally:
        db.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
