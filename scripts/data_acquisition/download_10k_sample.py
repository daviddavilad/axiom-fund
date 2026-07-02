"""Pilot 10-K download from SEC EDGAR for 10 diverse tickers.

Per docs/v2_item6_design.md, this script validates the EDGAR client
end-to-end on a small diverse set of tickers before scaling to the
full v1 universe. The ticker set spans sectors and firm ages,
maximizing the chance of surfacing 10-K format edge cases early.

Tickers: AAPL, MSFT, JPM, XOM, JNJ, WMT, T, NEE, KO, GE

Output: data/raw/edgar/{TICKER}/10-K/{FILING_DATE}/{PRIMARY_DOC}

Sample window: 10-K filings with filing_date in [2015-01-01, 2025-01-01).
That gives ~10 filings per firm and matches the v1 backtest window.

This script is idempotent: filings already downloaded are skipped.
"""
# ruff: noqa: I001

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from axiom_fund.data.edgar import EdgarClient


PILOT_TICKERS = ["AAPL", "MSFT", "JPM", "XOM", "JNJ", "WMT", "T", "NEE", "KO", "GE"]
SAMPLE_START = "2015-01-01"
SAMPLE_END = "2025-01-01"
OUTPUT_DIR = Path("data/raw/edgar")


def main() -> int:
    print(f"Item 6 pilot download: {len(PILOT_TICKERS)} tickers")
    print(f"Sample window: {SAMPLE_START} to {SAMPLE_END}")
    print(f"Output: {OUTPUT_DIR}\n")

    client = EdgarClient(output_dir=OUTPUT_DIR)

    total_filings = 0
    total_bytes = 0
    failures: list[tuple[str, str]] = []

    for ticker in PILOT_TICKERS:
        print(f"[{ticker}]")
        try:
            cik = client.get_cik(ticker)
            filings = client.get_10k_history(cik, earliest_filing_date=SAMPLE_START)
            # Filter to sample window
            windowed = [
                f for f in filings
                if SAMPLE_START <= f.filing_date < SAMPLE_END
            ]
            print(f"  CIK={cik}, {len(filings)} total 10-Ks, "
                  f"{len(windowed)} in window")

            for f in windowed:
                path = client.download_filing(cik, f, ticker=ticker)
                size = path.stat().st_size
                total_bytes += size
                total_filings += 1
                print(f"  {f.filing_date}: {path.name} ({size:,} bytes)")

        except Exception as e:
            failures.append((ticker, f"{type(e).__name__}: {e}"))
            print(f"  FAILED: {type(e).__name__}: {e}")

    # Summary
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Tickers processed: {len(PILOT_TICKERS)}")
    print(f"Filings downloaded: {total_filings}")
    print(f"Total bytes: {total_bytes:,} ({total_bytes / 1024**2:.1f} MB)")
    print(f"Failures: {len(failures)}")
    for ticker, err in failures:
        print(f"  {ticker}: {err}")

    # Metadata file for reproducibility
    metadata_path = OUTPUT_DIR / "pilot_run_metadata.txt"
    metadata_path.write_text(
        f"Run: {datetime.now().isoformat()}\n"
        f"Tickers: {PILOT_TICKERS}\n"
        f"Sample window: [{SAMPLE_START}, {SAMPLE_END})\n"
        f"Filings downloaded: {total_filings}\n"
        f"Total bytes: {total_bytes}\n"
        f"Failures: {len(failures)}\n"
    )
    print(f"\nMetadata: {metadata_path}")

    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())