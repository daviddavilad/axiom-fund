"""End-to-end test of ResMom signal against real WRDS data.

Pulls a small slice of returns + fundamentals, runs the signal, prints
diagnostics. Verifies the pipeline composes correctly and produces
reasonable Residual Momentum values for known names.
"""
# ruff: noqa: I001

from __future__ import annotations

import os
import sys

from axiom_fund import _warnings  # noqa: F401

import pandas as pd
from dotenv import load_dotenv

from axiom_fund.data.fundamentals import Fundamentals
from axiom_fund.data.returns import ReturnsPanel
from axiom_fund.signals.residual_momentum import compute_residual_momentum


def main() -> int:
    load_dotenv()
    username = os.getenv("WRDS_USERNAME")
    if not username:
        print("ERROR: WRDS_USERNAME not set", file=sys.stderr)
        return 1

    import wrds
    db = wrds.Connection(wrds_username=username)

    try:
        # 8 well-known stocks across multiple sectors
        permnos = [14593, 10107, 47896, 11308, 86580, 76076, 27983, 22592]
        ticker_map = {
            14593: "AAPL",
            10107: "MSFT",
            47896: "JPM",
            11308: "KO",
            86580: "NVDA",
            76076: "GOOGL",
            27983: "XOM",
            22592: "WMT",
        }

        # Need at least 14 months leading buffer for 12-month formation window
        print("Pulling daily returns (3 years for 12+ month formation buffer)...")
        rp = ReturnsPanel(db)
        rets = rp.fetch(
            permnos=permnos,
            start_date="2018-01-01",
            end_date="2020-12-31",
        )
        print(f"Returns panel: {len(rets):,} rows")
        print()

        print("Pulling fundamentals (for industry classification)...")
        fund_obj = Fundamentals(db)
        fund = fund_obj.fetch_quarterly(
            permnos=permnos,
            start_date="2017-01-01",
            end_date="2020-12-31",
        )
        print(f"Fundamentals panel: {len(fund):,} rows")
        print()

        print("Computing Residual Momentum signal...")
        signal = compute_residual_momentum(
            returns_df=rets,
            fundamentals_df=fund,
            start_date="2019-06-01",  # ~17 months after returns start
            end_date="2020-12-31",
        )
        print(f"Signal panel: {len(signal):,} rows")
        print()

        if len(signal) == 0:
            print("WARNING: empty signal output")
            return 1

        print("=" * 80)
        print("Mean signal metrics per ticker")
        print("=" * 80)
        print(f"{'Ticker':<8} {'Industry':<6} {'Mean size':>10} {'Mean signal':>12}")
        print("-" * 80)
        for permno in permnos:
            ticker = ticker_map[permno]
            sub = signal[signal["permno"] == permno]
            if len(sub) == 0:
                print(f"{ticker:<8} (no data)")
                continue
            ggroup = sub["ggroup"].iloc[0] if len(sub) > 0 else "?"
            print(
                f"{ticker:<8} {str(ggroup):<6} "
                f"{sub['size'].mean():>10.2f} "
                f"{sub['raw_signal'].mean():>12.4f}"
            )
        print()

        # Sample row
        print("=" * 80)
        print("Sample row from middle of panel:")
        print("=" * 80)
        mid_idx = len(signal) // 2
        print(signal.iloc[mid_idx])
        print()

        # Distribution stats
        print("=" * 80)
        print("Signal distribution across all (permno, date) observations")
        print("=" * 80)
        print(signal["raw_signal"].describe())

    finally:
        db.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())