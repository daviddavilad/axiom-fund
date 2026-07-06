"""Prototype the Lazy Prices signal on the 10-ticker pilot corpus.

Purpose: decide whether to invest ~2-3 hours in full-corpus section
extraction (6,286 filings) based on whether year-over-year cosine
similarity looks like a plausible signal on a small pilot.

Pilot: 10 diverse tickers × 6 years (2019-2024) = up to 50 consecutive-year
comparison pairs across each section. This is too small for statistical
inference but sufficient for a directional "is there any signal here?"
check.

Pipeline:
  1. Load each ticker's 10-Ks via edgartools (re-fetches from SEC)
  2. Extract Items 1, 1A, 7 via EdgartoolsSectionExtractor
  3. Drop cross-referenced sections (is_cross_reference=True)
  4. Fit TF-IDF vectorizer per section on all firms' text
  5. For each (ticker, section), compute cosine similarity between
     consecutive-year filings
  6. Report per-section distribution + per-ticker breakdown
  7. Evaluate against pre-committed directional criteria

Pre-committed success criteria (locked before running):
  - Dispersion: IQR of cosine similarity materially above zero (~0.02+)
  - Plausibility: median in [0.5, 0.99]
  - Section independence: cross-section correlations not all 0.99+

Decision from this prototype:
  - PASS all three: proceed to full-corpus extraction next session
  - FAIL any: reconsider methodology before spending 2-3 hours on
    full-corpus infrastructure work
"""
# ruff: noqa: I001

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import edgar
edgar.set_identity("Axiom Fund Research daviddavilacorraliza@gmail.com")
from edgar import Company

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from axiom_fund.data.section_extractor import EdgartoolsSectionExtractor


PILOT_TICKERS = ["AAPL", "MSFT", "JPM", "XOM", "JNJ", "WMT", "T", "NEE", "KO", "GE"]
SAMPLE_YEARS = list(range(2019, 2025))  # 2019-2024 inclusive
SECTIONS = ["Item 1", "Item 1A", "Item 7"]


def extract_pilot_corpus() -> dict[tuple[str, int], dict[str, str]]:
    """Extract Items 1, 1A, 7 for each ticker/year, dropping cross-references.

    Returns
    -------
    dict[(ticker, year), dict[section_id, text]]
        Only sections that extracted cleanly and were NOT cross-references.
    """
    ext = EdgartoolsSectionExtractor()
    corpus: dict[tuple[str, int], dict[str, str]] = {}

    for ticker in PILOT_TICKERS:
        print(f"  {ticker}...", end=" ", flush=True)
        try:
            filings = list(Company(ticker).get_filings(form="10-K"))
        except Exception as e:
            print(f"FAILED to fetch filings: {type(e).__name__}")
            continue

        for filing in filings:
            year = filing.filing_date.year
            if year not in SAMPLE_YEARS:
                continue
            try:
                results = ext.extract(filing)
            except Exception:
                continue

            # Keep only successful, non-cross-reference sections
            usable = {
                sid: r.text
                for sid, r in results.items()
                if not r.is_cross_reference
            }
            if usable:
                corpus[(ticker, year)] = usable

        n_years = sum(1 for k in corpus if k[0] == ticker)
        print(f"{n_years} years")

    return corpus


def compute_yoy_similarities(
    corpus: dict[tuple[str, int], dict[str, str]],
) -> dict[str, list[dict[str, Any]]]:
    """Compute year-over-year cosine similarity per (ticker, section).

    Returns
    -------
    dict[section_id, list[{ticker, year_pair, similarity}]]
    """
    results: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for section in SECTIONS:
        # Collect all firm-year documents that have this section
        docs: list[tuple[str, int, str]] = []
        for (ticker, year), sections_dict in corpus.items():
            if section in sections_dict:
                docs.append((ticker, year, sections_dict[section]))

        if not docs:
            print(f"  No usable docs for {section}, skipping")
            continue

        # Fit TF-IDF on all documents for this section
        texts = [d[2] for d in docs]
        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            min_df=2,     # drop tokens appearing in only 1 doc
            token_pattern=r"(?u)\b[a-zA-Z]{2,}\b",  # 2+ letter words, no digits
        )
        tfidf = vectorizer.fit_transform(texts)
        print(f"  {section}: {len(docs)} docs, {tfidf.shape[1]:,} vocab tokens")

        # Index by (ticker, year) for lookup
        doc_index: dict[tuple[str, int], int] = {
            (t, y): i for i, (t, y, _) in enumerate(docs)
        }

        # For each ticker, compute consecutive-year pair similarities
        for ticker in PILOT_TICKERS:
            ticker_years = sorted(y for (t, y) in doc_index if t == ticker)
            for i in range(len(ticker_years) - 1):
                y1, y2 = ticker_years[i], ticker_years[i + 1]
                if y2 - y1 != 1:
                    continue  # only consecutive-year pairs
                idx1 = doc_index[(ticker, y1)]
                idx2 = doc_index[(ticker, y2)]
                sim = float(cosine_similarity(tfidf[idx1], tfidf[idx2])[0, 0])
                results[section].append({
                    "ticker": ticker,
                    "year_pair": f"{y1}-{y2}",
                    "similarity": sim,
                })

    return dict(results)


def report(sims: dict[str, list[dict[str, Any]]]) -> None:
    """Print distribution + per-ticker + evaluate pre-commitments."""
    print()
    print("=" * 70)
    print("Distribution of year-over-year cosine similarity per section")
    print("=" * 70)
    print(f"{'Section':<10} {'N pairs':>8} {'min':>6} {'p25':>6} "
          f"{'median':>7} {'p75':>6} {'max':>6} {'IQR':>6}")
    print("-" * 60)
    stats_by_section: dict[str, dict[str, float]] = {}
    for section in SECTIONS:
        vals = [r["similarity"] for r in sims.get(section, [])]
        if not vals:
            print(f"{section:<10} no data")
            continue
        arr = np.array(vals)
        p25, p50, p75 = np.percentile(arr, [25, 50, 75])
        iqr = p75 - p25
        stats_by_section[section] = {
            "n": len(vals), "min": arr.min(), "p25": p25, "median": p50,
            "p75": p75, "max": arr.max(), "iqr": iqr,
        }
        print(f"{section:<10} {len(vals):>8} {arr.min():>6.3f} {p25:>6.3f} "
              f"{p50:>7.3f} {p75:>6.3f} {arr.max():>6.3f} {iqr:>6.3f}")

    # Per-ticker mean similarity per section
    print()
    print("=" * 70)
    print("Per-ticker mean similarity (across year pairs)")
    print("=" * 70)
    print(f"{'Ticker':<8}", end="")
    for section in SECTIONS:
        print(f"{section:>10}", end="")
    print()
    for ticker in PILOT_TICKERS:
        print(f"{ticker:<8}", end="")
        for section in SECTIONS:
            vals = [r["similarity"] for r in sims.get(section, [])
                    if r["ticker"] == ticker]
            if vals:
                print(f"{np.mean(vals):>10.3f}", end="")
            else:
                print(f"{'—':>10}", end="")
        print()

    # Cross-section correlation at the firm-year level
    print()
    print("=" * 70)
    print("Cross-section correlation at the (ticker, year_pair) level")
    print("=" * 70)
    all_rows: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for section, rows in sims.items():
        for r in rows:
            all_rows[(r["ticker"], r["year_pair"])][section] = r["similarity"]
    pairs_df = pd.DataFrame([
        {"ticker": t, "year_pair": p, **secs}
        for (t, p), secs in all_rows.items()
    ])
    if len(pairs_df) > 0:
        corr = pairs_df[SECTIONS].corr()
        print(corr.to_string(float_format="%.3f"))

    # Pre-committed evaluation
    print()
    print("=" * 70)
    print("Pre-committed directional criteria")
    print("=" * 70)

    # 1. Dispersion
    print("\n1. Usable dispersion (IQR materially above zero, ~0.02+):")
    for section, stats in stats_by_section.items():
        verdict = "PASS" if stats["iqr"] >= 0.02 else "FAIL"
        print(f"   {section}: IQR = {stats['iqr']:.3f}  [{verdict}]")

    # 2. Plausibility
    print("\n2. Directionally plausible (median in [0.5, 0.99]):")
    for section, stats in stats_by_section.items():
        verdict = "PASS" if 0.5 <= stats["median"] <= 0.99 else "FAIL"
        print(f"   {section}: median = {stats['median']:.3f}  [{verdict}]")

    # 3. Section independence
    print("\n3. Section independence (not all correlations at 0.99+):")
    if len(pairs_df) > 0 and len(SECTIONS) > 1:
        off_diag = corr.where(~np.eye(len(SECTIONS), dtype=bool)).stack()
        max_off_diag = off_diag.max()
        verdict = "PASS" if max_off_diag < 0.99 else "FAIL"
        print(f"   Max off-diagonal correlation = {max_off_diag:.3f}  [{verdict}]")
    else:
        print("   Insufficient data")


def main() -> int:
    print("Loading and extracting pilot corpus...")
    corpus = extract_pilot_corpus()
    print(f"\nCorpus: {len(corpus)} (ticker, year) entries with usable sections")

    print("\nComputing year-over-year cosine similarities...")
    sims = compute_yoy_similarities(corpus)

    report(sims)
    return 0


if __name__ == "__main__":
    sys.exit(main())