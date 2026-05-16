"""Smoke test for the PEAD signal against real Compustat data.

Pulls 8 years of fundamentals for three well-known names (AAPL, MSFT,
TSLA) and verifies the SUE computation produces sensible values:
- Non-trivial range
- Mean near zero (expected for own-name standardization)
- Q4 2024 AAPL should produce a negative SUE given the EPS decline
- Winsorization shouldn't bind on small sample

Run before formal tests to catch math errors early.
"""
# ruff: noqa: I001

from __future__ import annotations

import os
import sys

from axiom_fund import _warnings  # noqa: F401

from dotenv import load_dotenv

from axiom_fund.data.fundamentals import Fundamentals
from axiom_fund.signals.pead import compute_pead_signal


def main() -> int:
    load_dotenv()
    username = os.getenv("WRDS_USERNAME")
    if not username:
        print("ERROR: WRDS_USERNAME not set", file=sys.stderr)
        return 1

    import wrds
    db = wrds.Connection(wrds_username=username)

    try:
        fund = Fundamentals(db)
        # Need 8+ quarters of history before 2020 for trailing std → start 2017
        df_fund = fund.fetch_quarterly(
            permnos=[14593, 10107, 93436],
            start_date="2017-01-01",
            end_date="2024-12-31",
        )
        print(f"Fundamentals: {len(df_fund)} rows")
        print()

        signal = compute_pead_signal(
            fundamentals=df_fund,
            start_date="2020-01-01",
            end_date="2024-12-31",
        )
        print(f"PEAD signal: {len(signal)} rows")
        print()

        print("=" * 70)
        print("AAPL PEAD signal:")
        print("=" * 70)
        aapl = signal[signal["permno"] == 14593].sort_values("date_filed")
        print(
            aapl[
                [
                    "date_filed", "fyearq", "fqtr", "epspxq",
                    "surprise", "sue_raw", "sue",
                ]
            ].to_string(index=False)
        )
        print()

        print("=" * 70)
        print("Distribution checks:")
        print("=" * 70)
        print(
            f"  Mean SUE:   {signal['sue'].mean():+.3f}"
            f"  (expected near 0 for own-name standardization)"
        )
        print(f"  Std SUE:    {signal['sue'].std():.3f}")
        print(f"  Min SUE:    {signal['sue'].min():+.3f}")
        print(f"  Max SUE:    {signal['sue'].max():+.3f}")
        print(
            f"  Winsorized: {(signal['sue'].abs() == 3.0).mean() * 100:.1f}%"
        )
        print()

        print("=" * 70)
        print("Largest positive SUE per name (biggest beats):")
        print("=" * 70)
        for permno in [14593, 10107, 93436]:
            sub = signal[signal["permno"] == permno].sort_values(
                "sue", ascending=False
            ).head(1)
            print(
                sub[
                    ["permno", "date_filed", "fyearq", "fqtr", "surprise", "sue"]
                ].to_string(index=False)
            )
        print()

        print("=" * 70)
        print("Largest negative SUE per name (biggest misses):")
        print("=" * 70)
        for permno in [14593, 10107, 93436]:
            sub = signal[signal["permno"] == permno].sort_values("sue").head(1)
            print(
                sub[
                    ["permno", "date_filed", "fyearq", "fqtr", "surprise", "sue"]
                ].to_string(index=False)
            )

    finally:
        db.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())