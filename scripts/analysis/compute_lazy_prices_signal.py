"""Runner: compute the Lazy Prices signal on the canonical corpus.

Loads data/cache/lazy_prices_sections.parquet, calls the library
function from axiom_fund.signals.lazy_prices, saves output panel to
data/cache/lazy_prices_signal.parquet.

Prints summary distribution stats to compare against the July 6 pilot
expected numbers (median 0.96-0.98, IQR 0.04-0.05 per section).

Library methodology and pre-commitments documented in
src/axiom_fund/signals/lazy_prices.py.
"""
# ruff: noqa: I001

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from axiom_fund.signals.lazy_prices import (
    compute_lazy_prices_signal,
    LAZY_PRICES_SECTIONS,
)


INPUT_PATH = Path("data/cache/lazy_prices_sections.parquet")
OUTPUT_PATH = Path("data/cache/lazy_prices_signal.parquet")
METADATA_PATH = Path("data/cache/lazy_prices_signal.txt")


def main() -> int:
    if not INPUT_PATH.exists():
        print(f"ERROR: {INPUT_PATH} not found.", file=sys.stderr)
        return 1

    print(f"Loading {INPUT_PATH}")
    sections_df = pd.read_parquet(INPUT_PATH)
    print(f"  {len(sections_df):,} section rows across "
          f"{sections_df.ticker.nunique()} tickers")

    print("Computing signal...")
    signal_df = compute_lazy_prices_signal(sections_df)
    print(f"  {len(signal_df):,} signal rows")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    signal_df.to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved: {OUTPUT_PATH}")

    # Distribution per section (compare to pilot expectations)
    print()
    print("=" * 70)
    print("Signal distribution per section (compare to pilot: median 0.96-0.98, IQR 0.04-0.05)")
    print("=" * 70)
    print(f"{'Section':<10} {'N pairs':>8} {'min':>7} {'p25':>7} "
          f"{'median':>8} {'p75':>7} {'max':>7} {'IQR':>7}")
    print("-" * 66)
    for section in LAZY_PRICES_SECTIONS:
        sub = signal_df[signal_df.section_id == section].similarity
        if len(sub) == 0:
            print(f"{section:<10} no pairs")
            continue
        p25, p50, p75 = np.percentile(sub, [25, 50, 75])
        print(f"{section:<10} {len(sub):>8}  {sub.min():.3f}  {p25:.3f}   "
              f"{p50:.3f}   {p75:.3f}  {sub.max():.3f}   {p75-p25:.3f}")

    # Cross-section correlations (compare to pilot: 0.52-0.71)
    print()
    print("Cross-section correlations at (ticker, filing_year) level:")
    pivoted = signal_df.pivot_table(
        index=["ticker", "filing_year"],
        columns="section_id",
        values="similarity",
    )
    corr = pivoted.corr()
    print(corr.to_string())

    # Metadata
    METADATA_PATH.write_text(
        f"Run: {datetime.now().isoformat()}\n"
        f"Input: {INPUT_PATH}\n"
        f"Input rows: {len(sections_df)}\n"
        f"Signal rows: {len(signal_df)}\n"
        f"Firms with signal: {signal_df.ticker.nunique()}\n"
    )
    print(f"\nMetadata: {METADATA_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())