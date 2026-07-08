"""Download 10-K filings for the Lazy Prices sample universe (2019-2024).

Reads data/cache/lazy_prices_ciks.parquet, downloads all 10-Ks filed in
2019-2024 for each resolved firm.

Storage: data/raw/edgar/{TICKER}/10-K/{FILING_DATE}/{PRIMARY_DOC}

Idempotent: reruns skip already-downloaded filings. Progress is checkpointed
to data/cache/lazy_prices_download_progress.parquet so a mid-run interruption
can resume without redoing the filing-history queries.
"""
# ruff: noqa: I001

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from axiom_fund.data.edgar import EdgarClient


DEFAULT_INPUT_PATH = Path("data/cache/lazy_prices_ciks.parquet")
DEFAULT_OUTPUT_DIR = Path("data/raw/edgar")
DEFAULT_PROGRESS_PATH = Path("data/cache/lazy_prices_download_progress.parquet")
DEFAULT_METADATA_PATH = Path("data/cache/lazy_prices_download.txt")

SAMPLE_START = "2019-01-01"
SAMPLE_END = "2025-01-01"


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH,
                        help=f"Input CIK parquet (default: {DEFAULT_INPUT_PATH})")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help=f"Download destination (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--progress", type=Path, default=DEFAULT_PROGRESS_PATH,
                        help=f"Progress checkpoint (default: {DEFAULT_PROGRESS_PATH})")
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_PATH,
                        help=f"Metadata output (default: {DEFAULT_METADATA_PATH})")
    args = parser.parse_args()

    INPUT_PATH = args.input
    OUTPUT_DIR = args.output_dir
    PROGRESS_PATH = args.progress
    METADATA_PATH = args.metadata

    if not INPUT_PATH.exists():
        print(f"ERROR: {INPUT_PATH} not found. Run resolve_ciks.py first.",
              file=sys.stderr)
        return 1

    ciks_df = pd.read_parquet(INPUT_PATH)
    print(f"Firms to process: {len(ciks_df)}")
    print(f"Sample window: {SAMPLE_START} to {SAMPLE_END}")
    print()

    # Load progress if resuming
    if PROGRESS_PATH.exists():
        progress = pd.read_parquet(PROGRESS_PATH)
        done_tickers = set(progress[progress["status"] == "complete"]["ticker"])
        print(f"Resuming: {len(done_tickers)} tickers already complete")
    else:
        progress = pd.DataFrame(columns=["ticker", "cik", "status", "n_filings", "error"])
        done_tickers = set()

    client = EdgarClient(output_dir=OUTPUT_DIR)

    progress_rows = progress.to_dict("records")
    total_filings = 0
    total_bytes = 0

    for i, row in ciks_df.iterrows():
        ticker = row["ticker"]
        cik = row["cik"]

        if ticker in done_tickers:
            continue

        try:
            filings = client.get_10k_history(cik, earliest_filing_date=SAMPLE_START)
            windowed = [
                f for f in filings
                if SAMPLE_START <= f.filing_date < SAMPLE_END
            ]
            n_this_firm = 0
            for f in windowed:
                path = client.download_filing(cik, f, ticker=ticker)
                total_bytes += path.stat().st_size
                n_this_firm += 1
                total_filings += 1
            progress_rows.append({
                "ticker": ticker,
                "cik": cik,
                "status": "complete",
                "n_filings": n_this_firm,
                "error": "",
            })
        except Exception as e:
            progress_rows.append({
                "ticker": ticker,
                "cik": cik,
                "status": "error",
                "n_filings": 0,
                "error": f"{type(e).__name__}: {str(e)[:200]}",
            })

        # Checkpoint every 50 firms
        if (i + 1) % 50 == 0:
            pd.DataFrame(progress_rows).to_parquet(PROGRESS_PATH, index=False)
            n_complete = sum(1 for r in progress_rows if r["status"] == "complete")
            n_error = sum(1 for r in progress_rows if r["status"] == "error")
            print(f"  {i + 1:>4}/{len(ciks_df)}: "
                  f"{n_complete} complete, {n_error} error, "
                  f"{total_filings} filings, "
                  f"{total_bytes / 1024**2:,.0f} MB")

    # Final checkpoint
    progress_final = pd.DataFrame(progress_rows)
    progress_final.to_parquet(PROGRESS_PATH, index=False)

    # Summary
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    n_complete = (progress_final["status"] == "complete").sum()
    n_error = (progress_final["status"] == "error").sum()
    print(f"Firms complete: {n_complete}")
    print(f"Firms errored:  {n_error}")
    print(f"Total filings:  {total_filings:,}")
    print(f"Total size:     {total_bytes:,} bytes ({total_bytes / 1024**2:,.0f} MB)")
    if n_error > 0:
        print("\nFirst 10 error samples:")
        errors = progress_final[progress_final["status"] == "error"].head(10)
        for _, r in errors.iterrows():
            print(f"  {r['ticker']:<6} {r['error'][:100]}")

    METADATA_PATH.write_text(
        f"Run: {datetime.now().isoformat()}\n"
        f"Firms input: {len(ciks_df)}\n"
        f"Firms complete: {n_complete}\n"
        f"Firms errored: {n_error}\n"
        f"Total filings: {total_filings}\n"
        f"Total bytes: {total_bytes}\n"
        f"Sample window: [{SAMPLE_START}, {SAMPLE_END})\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())