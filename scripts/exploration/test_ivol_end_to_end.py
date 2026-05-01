"""End-to-end test of IVol signal against real WRDS data.

Pulls a small slice of returns + FF factors, runs the signal, prints
diagnostics. Verifies the pipeline composes correctly and produces
reasonable IVol values for known names.
"""
# ruff: noqa: I001

from __future__ import annotations

import os
import sys

from axiom_fund import _warnings  # noqa: F401

import pandas as pd
from dotenv import load_dotenv

from axiom_fund.data.ff_factors import FFFactors
from axiom_fund.data.returns import ReturnsPanel
from axiom_fund.signals.idiosyncratic_volatility import (
    compute_idiosyncratic_volatility,
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
        # Five well-known stocks: AAPL, MSFT, JPM, KO, NVDA
        # PERMNOs from CRSP
        permnos = [14593, 10107, 47896, 11308, 86580]
        ticker_map = {
            14593: "AAPL",
            10107: "MSFT",
            47896: "JPM",
            11308: "KO",
            86580: "NVDA",
        }

        print("Pulling 2020 returns for 5 well-known stocks...")
        rp = ReturnsPanel(db)
        # Need extra leading buffer for 60-day rolling window
        rets = rp.fetch(
            permnos=permnos,
            start_date="2019-09-01",  # ~3 months buffer
            end_date="2020-12-31",
        )
        print(f"Returns panel: {len(rets):,} rows")
        print()

        print("Pulling FF factors for same date range...")
        ff = FFFactors(db)
        factors = ff.fetch("2019-09-01", "2020-12-31")
        print(f"FF factors panel: {len(factors):,} rows")
        print()

        print("Computing Idiosyncratic Volatility signal...")
        ivol = compute_idiosyncratic_volatility(
            returns_df=rets,
            ff_factors_df=factors,
            start_date="2020-01-02",
            end_date="2020-12-31",
        )
        print(f"IVol signal panel: {len(ivol):,} rows")
        print()

        # Diagnostics per ticker
        print("=" * 80)
        print("Mean IVol metrics per ticker (2020)")
        print("=" * 80)
        print(f"{'Ticker':<8} {'Beta_mkt':>10} {'Beta_smb':>10} {'Beta_hml':>10} "
              f"{'ResStd':>10} {'AnnIVol':>10}")
        print("-" * 80)
        for permno in permnos:
            ticker = ticker_map[permno]
            sub = ivol[ivol["permno"] == permno]
            if len(sub) == 0:
                print(f"{ticker:<8} (no data)")
                continue
            print(
                f"{ticker:<8} "
                f"{sub['beta_mkt'].mean():>10.3f} "
                f"{sub['beta_smb'].mean():>10.3f} "
                f"{sub['beta_hml'].mean():>10.3f} "
                f"{sub['residual_std'].mean():>10.4f} "
                f"{sub['raw_signal'].mean():>10.3f}"
            )
        print()

        # Sanity expectations:
        # - Beta_mkt should be ~1 for large caps (KO ~0.6, AAPL/MSFT ~1.1, NVDA ~1.5+)
        # - Annualized IVol typically 0.20-0.50 for large caps
        # - NVDA should have higher IVol than KO

        print("=" * 80)
        print("Sample row from middle of panel:")
        print("=" * 80)
        if len(ivol) > 0:
            mid_idx = len(ivol) // 2
            sample = ivol.iloc[mid_idx]
            print(sample)

    finally:
        db.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())