"""Runner: compute the Lazy Prices signal on the canonical corpus.

Loads data/cache/lazy_prices_sections.parquet, enriches with permno
from data/cache/lazy_prices_ciks.parquet, calls the library function
from axiom_fund.signals.lazy_prices, saves output panel to
data/cache/lazy_prices_signal.parquet.

Output schema conforms to docs/signal_design.md §2.1: one row per
(permno, date_filed) with per-section component similarities plus
raw_signal.

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
    LAZY_PRICES_RAW_COLUMNS,
)


SECTIONS_PATH = Path("data/cache/lazy_prices_sections.parquet")
# Use the merged + deduped CIK file, not the SEC-only lazy_prices_ciks.parquet
# (which is 1,117 firms only). The deduped file is the canonical input to the
# extraction pipeline that produced lazy_prices_sections.parquet.
CIKS_PATH = Path("data/cache/lazy_prices_ciks_merged_dedup.parquet")
OUTPUT_PATH = Path("data/cache/lazy_prices_signal.parquet")
METADATA_PATH = Path("data/cache/lazy_prices_signal.txt")


def main() -> int:
    for path in (SECTIONS_PATH, CIKS_PATH):
        if not path.exists():
            print(f"ERROR: {path} not found.", file=sys.stderr)
            return 1

    print(f"Loading sections: {SECTIONS_PATH}")
    sections_df = pd.read_parquet(SECTIONS_PATH)
    print(f"  {len(sections_df):,} section rows across "
          f"{sections_df.ticker.nunique()} tickers")

    print(f"Loading CIK->permno map: {CIKS_PATH}")
    ciks_df = pd.read_parquet(CIKS_PATH)
    print(f"  {len(ciks_df):,} CIK rows")

    # Enrich sections with permno via ticker join.
    # Note: some tickers map to >1 permno legitimately — e.g. "S" maps to
    # both PERMNO 14040 (Sprint historical) and 21415 (SentinelOne).
    # These are different firms that shared the ticker at different times.
    # The join creates one row per (section, permno), which is correct:
    # downstream code should key on permno, not ticker.
    ticker_to_permno = ciks_df[["ticker", "permno"]].drop_duplicates()

    # Diagnostic: report tickers with multiple permno mappings.
    per_ticker = ticker_to_permno.groupby("ticker").size()
    multi_permno_tickers = per_ticker[per_ticker > 1]
    if len(multi_permno_tickers) > 0:
        print(f"  Note: {len(multi_permno_tickers)} tickers map to >1 permno "
              f"(different firms sharing ticker over time):")
        for ticker in sorted(multi_permno_tickers.index):
            permnos = sorted(
                ticker_to_permno[ticker_to_permno.ticker == ticker].permno.tolist()
            )
            print(f"    {ticker}: permnos {permnos}")

    n_before = len(sections_df)
    sections_df = sections_df.merge(ticker_to_permno, on="ticker", how="inner")
    n_after = len(sections_df)
    delta = n_after - n_before
    if delta > 0:
        print(f"  Enriched: {n_after:,} rows (+{delta} from multi-permno joins)")
    elif delta < 0:
        print(f"  Enriched: {n_after:,} rows ({-delta} dropped for unmapped ticker)")
    else:
        print(f"  Enriched: {n_after:,} rows (all tickers mapped 1-to-1)")

    print("Computing signal...")
    signal_df = compute_lazy_prices_signal(sections_df)
    print(f"  {len(signal_df):,} signal rows")

    assert tuple(signal_df.columns) == LAZY_PRICES_RAW_COLUMNS, (
        "Signal output schema does not match LAZY_PRICES_RAW_COLUMNS"
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    signal_df.to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved: {OUTPUT_PATH}")

    # ------------------------------------------------------------------
    # Distribution reporting
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("Per-section YoY cosine similarity distribution")
    print("=" * 70)
    print(f"{'Section':<12} {'N':>6} {'min':>7} {'p25':>7} "
          f"{'median':>8} {'p75':>7} {'max':>7} {'IQR':>7}")
    print("-" * 66)
    for label, col in [("Item 1", "sim_item1"),
                       ("Item 1A", "sim_item1a"),
                       ("Item 7", "sim_item7")]:
        sub = signal_df[col].dropna()
        if sub.empty:
            print(f"{label:<12} no data")
            continue
        p25, p50, p75 = np.percentile(sub, [25, 50, 75])
        print(f"{label:<12} {len(sub):>6}  {sub.min():.3f}  {p25:.3f}   "
              f"{p50:.3f}   {p75:.3f}  {sub.max():.3f}   {p75 - p25:.3f}")

    print()
    print("Raw signal distribution (= 1 - mean(sim across 3 sections)):")
    rs = signal_df["raw_signal"].dropna()
    p25, p50, p75 = np.percentile(rs, [25, 50, 75])
    print(f"{'raw_signal':<12} {len(rs):>6}  {rs.min():.3f}  {p25:.3f}   "
          f"{p50:.3f}   {p75:.3f}  {rs.max():.3f}   {p75 - p25:.3f}")

    print()
    print("Cross-section correlations among sim columns:")
    corr = signal_df[["sim_item1", "sim_item1a", "sim_item7"]].corr()
    print(corr.to_string())

    # Metadata
    METADATA_PATH.write_text(
        f"Run: {datetime.now().isoformat()}\n"
        f"Sections input: {SECTIONS_PATH}\n"
        f"CIKs input: {CIKS_PATH}\n"
        f"Signal rows: {len(signal_df)}\n"
        f"Firms with signal: {signal_df.permno.nunique()}\n"
        f"Schema: {list(LAZY_PRICES_RAW_COLUMNS)}\n"
    )
    print(f"\nMetadata: {METADATA_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())