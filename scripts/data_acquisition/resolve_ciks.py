"""Resolve CIKs for the Lazy Prices universe tickers.

Reads data/cache/lazy_prices_universe.parquet, resolves each ticker to a
CIK via SEC EDGAR's company_tickers.json, and writes:

  data/cache/lazy_prices_ciks.parquet — one row per successfully-resolved
    (ticker, permno, cik)

Also reports the count of failed resolutions with sampling for inspection.
Failed tickers are logged but do not stop the run (foreign issuers, delisted
firms, ticker changes, and ADRs are all expected causes).
"""
# ruff: noqa: I001

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from axiom_fund.data.edgar import EdgarClient


INPUT_PATH = Path("data/cache/lazy_prices_universe.parquet")
OUTPUT_PATH = Path("data/cache/lazy_prices_ciks.parquet")
FAILED_PATH = Path("data/cache/lazy_prices_ciks_failed.parquet")


def main() -> int:
    if not INPUT_PATH.exists():
        print(f"ERROR: {INPUT_PATH} not found. Run build_universe.py first.",
              file=sys.stderr)
        return 1

    universe = pd.read_parquet(INPUT_PATH)
    print(f"Universe: {len(universe)} unique firms")
    print()

    # EdgarClient loads .env internally
    client = EdgarClient(output_dir=Path("data/raw/edgar"))

    # Force CIK map fetch upfront (single HTTP call)
    print("Fetching SEC company_tickers.json...")
    _ = client.get_cik("AAPL")  # forces map load
    print(f"CIK map loaded: {len(client._cik_map):,} tickers indexed")
    print()

    resolved: list[dict] = []
    failed: list[dict] = []

    print("Resolving tickers...")
    for i, row in universe.iterrows():
        ticker = row["ticker"]
        permno = row["permno"]
        try:
            cik = client.get_cik(ticker)
            resolved.append({
                "ticker": ticker,
                "permno": permno,
                "comnam": row["comnam"],
                "cik": cik,
            })
        except KeyError:
            failed.append({
                "ticker": ticker,
                "permno": permno,
                "comnam": row["comnam"],
                "reason": "not_in_sec_map",
            })
        if (i + 1) % 200 == 0:
            print(f"  {i + 1:>4}/{len(universe)}: "
                  f"{len(resolved)} resolved, {len(failed)} failed")

    print(f"\nFinal: {len(resolved):,} resolved, {len(failed):,} failed "
          f"({100 * len(failed) / len(universe):.1f}% failure)")

    # Save results
    resolved_df = pd.DataFrame(resolved)
    resolved_df.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved resolved: {OUTPUT_PATH} ({len(resolved_df)} rows)")

    if failed:
        failed_df = pd.DataFrame(failed)
        failed_df.to_parquet(FAILED_PATH, index=False)
        print(f"Saved failed:  {FAILED_PATH} ({len(failed_df)} rows)")
        print(f"\nSample of failures (first 10):")
        for _, r in failed_df.head(10).iterrows():
            print(f"  {r['ticker']:<6} {r['comnam']}")

    # Metadata
    metadata_path = OUTPUT_PATH.with_suffix(".txt")
    metadata_path.write_text(
        f"Run: {datetime.now().isoformat()}\n"
        f"Universe source: {INPUT_PATH}\n"
        f"Universe size: {len(universe)}\n"
        f"Resolved: {len(resolved)}\n"
        f"Failed: {len(failed)}\n"
        f"Failure rate: {100 * len(failed) / len(universe):.2f}%\n"
    )
    print(f"Metadata: {metadata_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())