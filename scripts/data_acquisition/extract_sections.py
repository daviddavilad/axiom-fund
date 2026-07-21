"""Batch extract Items 1, 1A, 7 from the full Lazy Prices corpus.

Reads: data/cache/lazy_prices_ciks.parquet (1,117 firms with resolved CIKs)
Reads: data/raw/edgar/{TICKER}/10-K/{DATE}/ (6,286 downloaded filings)

Writes: data/cache/lazy_prices_sections.parquet
  Long format: one row per (ticker, filing_date, section_id) with columns:
    - ticker, cik, filing_date, accession_no, section_id
    - text, length, is_cross_reference

Writes: data/cache/lazy_prices_sections_failed.parquet
  Failed extractions with error type and message.

Idempotent: reruns skip already-processed firms based on progress file.

edgartools is put in strict local storage mode pointed at
data/raw/edgar. This means:
  - Filings we downloaded on 2026-07-05 are read from local disk
  - Filings not on disk raise an error (allow_network_fallback=False)
  - Runtime is expected to be ~10-15 min based on per-filing extraction
    benchmarks (see session notes 2026-07-06)
"""
# ruff: noqa: I001

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

import edgar
edgar.set_identity("Axiom Fund Research daviddavilacorraliza@gmail.com")
# Note: no longer using edgar.Company — see Path C design 2026-07-09.
# CIK-based routing via edgar.get_entity() replaces ticker-based routing.

from axiom_fund.data.section_extractor import (
    EdgartoolsSectionExtractor,
    MINIMUM_SUPPORTED_YEAR,
)


DEFAULT_INPUT_PATH = Path("data/cache/lazy_prices_ciks.parquet")
DEFAULT_OUTPUT_PATH = Path("data/cache/lazy_prices_sections.parquet")
DEFAULT_FAILED_PATH = Path("data/cache/lazy_prices_sections_failed.parquet")
DEFAULT_PROGRESS_PATH = Path("data/cache/lazy_prices_extraction_progress.parquet")
DEFAULT_METADATA_PATH = Path("data/cache/lazy_prices_sections.txt")

LOCAL_STORAGE_PATH = "data/raw/edgar"

SAMPLE_START_YEAR = 2019
SAMPLE_END_YEAR = 2026


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH,
                        help=f"Input CIK parquet (default: {DEFAULT_INPUT_PATH})")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH,
                        help=f"Sections output parquet (default: {DEFAULT_OUTPUT_PATH})")
    parser.add_argument("--failed", type=Path, default=DEFAULT_FAILED_PATH,
                        help=f"Failed extractions parquet (default: {DEFAULT_FAILED_PATH})")
    parser.add_argument("--progress", type=Path, default=DEFAULT_PROGRESS_PATH,
                        help=f"Progress checkpoint (default: {DEFAULT_PROGRESS_PATH})")
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_PATH,
                        help=f"Metadata output (default: {DEFAULT_METADATA_PATH})")
    args = parser.parse_args()

    INPUT_PATH = args.input
    OUTPUT_PATH = args.output
    FAILED_PATH = args.failed
    PROGRESS_PATH = args.progress
    METADATA_PATH = args.metadata

    if not INPUT_PATH.exists():
        print(f"ERROR: {INPUT_PATH} not found. Run resolve_ciks.py first.",
              file=sys.stderr)
        return 1

    # Enable strict local storage
    raw_path = os.path.abspath(LOCAL_STORAGE_PATH)
    if not os.path.isdir(raw_path):
        print(f"ERROR: {raw_path} does not exist. Run download_10k_corpus.py first.",
              file=sys.stderr)
        return 1
    edgar.use_local_storage(raw_path, allow_network_fallback=False)
    print(f"Local storage: {raw_path}")
    print(f"is_using_local_storage: {edgar.is_using_local_storage()}")
    print()

    ciks_df = pd.read_parquet(INPUT_PATH)
    print(f"Firms to process: {len(ciks_df)}")
    print(f"Sample window: {SAMPLE_START_YEAR}-{SAMPLE_END_YEAR}")
    print()

    # Load progress if resuming
    if PROGRESS_PATH.exists():
        progress = pd.read_parquet(PROGRESS_PATH)
        done_tickers = set(progress[progress["status"] == "complete"]["ticker"])
        print(f"Resuming: {len(done_tickers)} tickers already complete")
    else:
        progress = pd.DataFrame(columns=[
            "ticker", "cik", "status", "n_filings", "n_sections", "error"
        ])
        done_tickers = set()

    # Load existing extractions (append rather than overwrite)
    if OUTPUT_PATH.exists():
        existing_sections = pd.read_parquet(OUTPUT_PATH)
    else:
        existing_sections = pd.DataFrame(columns=[
            "ticker", "cik", "filing_date", "accession_no",
            "section_id", "text", "length", "is_cross_reference",
        ])
    if FAILED_PATH.exists():
        existing_failed = pd.read_parquet(FAILED_PATH)
    else:
        existing_failed = pd.DataFrame(columns=[
            "ticker", "cik", "filing_date", "accession_no", "error_type", "error_msg"
        ])

    extractor = EdgartoolsSectionExtractor()

    progress_rows = progress.to_dict("records")
    section_rows: list[dict] = existing_sections.to_dict("records")
    failed_rows: list[dict] = existing_failed.to_dict("records")

    total_filings_extracted = 0
    total_sections_extracted = 0

    for i, row in ciks_df.iterrows():
        ticker = row["ticker"]
        cik = row["cik"]

        if ticker in done_tickers:
            continue

        try:
            # Path C: CIK-based routing (not ticker-based). Handles firms
            # whose ticker moved to a different entity mid-window (Sprint
            # -> SentinelOne, etc.) by resolving directly to the historical
            # entity via CIK.
            cik_padded = str(cik).zfill(10)
            entity = edgar.get_entity(cik_padded)
            filings = list(entity.get_filings(form="10-K"))
            # Strict 10-K filter: Entity API (and Company API) return 10-K/A
            # amendments when form='10-K' is requested. For CMN signal
            # analysis we want one primary filing per fiscal year, not
            # amendments. See investigation 2026-07-09.
            filings = [f for f in filings if f.form == "10-K"]
            windowed = [
                f for f in filings
                if SAMPLE_START_YEAR <= f.filing_date.year <= SAMPLE_END_YEAR
            ]

            n_filings_this_ticker = 0
            n_sections_this_ticker = 0

            for f in windowed:
                try:
                    results = extractor.extract(f)
                except ValueError:
                    # window guard rejected — should not happen given our filter
                    failed_rows.append({
                        "ticker": ticker,
                        "cik": cik,
                        "filing_date": str(f.filing_date),
                        "accession_no": f.accession_no,
                        "error_type": "WindowGuardError",
                        "error_msg": f"Filing year {f.filing_date.year} below minimum {MINIMUM_SUPPORTED_YEAR}",
                    })
                    continue
                except Exception as e:
                    failed_rows.append({
                        "ticker": ticker,
                        "cik": cik,
                        "filing_date": str(f.filing_date),
                        "accession_no": f.accession_no,
                        "error_type": type(e).__name__,
                        "error_msg": str(e)[:500],
                    })
                    continue

                n_filings_this_ticker += 1
                for section_id, result in results.items():
                    section_rows.append({
                        "ticker": ticker,
                        "cik": cik,
                        "filing_date": str(f.filing_date),
                        "accession_no": f.accession_no,
                        "section_id": section_id,
                        "text": result.text,
                        "length": result.length,
                        "is_cross_reference": result.is_cross_reference,
                    })
                    n_sections_this_ticker += 1

            total_filings_extracted += n_filings_this_ticker
            total_sections_extracted += n_sections_this_ticker

            progress_rows.append({
                "ticker": ticker,
                "cik": cik,
                "status": "complete",
                "n_filings": n_filings_this_ticker,
                "n_sections": n_sections_this_ticker,
                "error": "",
            })

        except Exception as e:
            progress_rows.append({
                "ticker": ticker,
                "cik": cik,
                "status": "error",
                "n_filings": 0,
                "n_sections": 0,
                "error": f"{type(e).__name__}: {str(e)[:200]}",
            })

        # Checkpoint every 50 firms
        if (i + 1) % 50 == 0:
            pd.DataFrame(progress_rows).to_parquet(PROGRESS_PATH, index=False)
            pd.DataFrame(section_rows).to_parquet(OUTPUT_PATH, index=False)
            if failed_rows:
                pd.DataFrame(failed_rows).to_parquet(FAILED_PATH, index=False)
            n_complete = sum(1 for r in progress_rows if r["status"] == "complete")
            n_error = sum(1 for r in progress_rows if r["status"] == "error")
            n_failed_filings = len(failed_rows)
            msg = (f"  {i + 1:>4}/{len(ciks_df)}: "
                   f"{n_complete} tickers complete, {n_error} tickers errored, "
                   f"{total_filings_extracted} filings extracted, "
                   f"{n_failed_filings} filings failed, "
                   f"{total_sections_extracted} sections")
            print(msg, flush=True)

    # Final save
    pd.DataFrame(progress_rows).to_parquet(PROGRESS_PATH, index=False)
    section_df = pd.DataFrame(section_rows)
    section_df.to_parquet(OUTPUT_PATH, index=False)
    if failed_rows:
        pd.DataFrame(failed_rows).to_parquet(FAILED_PATH, index=False)

    # Summary
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    n_complete = sum(1 for r in progress_rows if r["status"] == "complete")
    n_ticker_error = sum(1 for r in progress_rows if r["status"] == "error")
    print(f"Tickers complete: {n_complete}")
    print(f"Tickers errored:  {n_ticker_error}")
    print(f"Filings extracted: {total_filings_extracted:,}")
    print(f"Filings failed:    {len(failed_rows):,}")
    print(f"Section rows:      {len(section_rows):,}")

    if section_df.empty:
        print("\nNo sections extracted — skipping distribution.")
        return 0

    print(f"\nSection breakdown:")
    for section in ["Item 1", "Item 1A", "Item 7"]:
        sec_df = section_df[section_df["section_id"] == section]
        n_total = len(sec_df)
        n_cross_ref = sec_df["is_cross_reference"].sum()
        n_usable = n_total - n_cross_ref
        print(f"  {section:<10}: {n_total:>5} extracted, "
              f"{n_cross_ref:>4} cross-ref, {n_usable:>5} usable")

    # Metadata
    METADATA_PATH.write_text(
        f"Run: {datetime.now().isoformat()}\n"
        f"Input firms: {len(ciks_df)}\n"
        f"Tickers complete: {n_complete}\n"
        f"Tickers errored: {n_ticker_error}\n"
        f"Filings extracted: {total_filings_extracted}\n"
        f"Filings failed: {len(failed_rows)}\n"
        f"Section rows: {len(section_rows)}\n"
        f"Sample window: {SAMPLE_START_YEAR}-{SAMPLE_END_YEAR}\n"
    )
    print(f"\nMetadata: {METADATA_PATH}")

    return 0


if __name__ == "__main__":
    sys.exit(main())