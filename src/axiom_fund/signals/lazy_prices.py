"""Lazy Prices raw signal (Cohen-Malloy-Nguyen 2020).

Signal
------
For each firm and each of {Item 1, Item 1A, Item 7} in the 10-K, compute
year-over-year TF-IDF cosine similarity between consecutive-year filings.

Higher similarity → less textual change → CMN predicts *underperformance*
subsequently (the "lazy" signal). The signal value reported here is the
raw similarity; downstream (e.g. alignment layer) can transform to
1 - similarity (change score) or z-score cross-sectionally as needed.

CMN's paper uses this per-section similarity as the signal input, computing
portfolio sorts on 1 - similarity. See docs/v2_item6_design.md.

Methodology
-----------
Pre-committed parameters (locked at pilot on 2026-07-06, see
scripts/analysis/prototype_lazy_prices_signal.py):

  - Per-section TF-IDF fit: each of Item 1, 1A, 7 has its own vocabulary
    (they use different language and shouldn't share tokens).
  - TfidfVectorizer(lowercase=True, stop_words='english', min_df=2,
                    token_pattern=r'(?u)\\b[a-zA-Z]{2,}\\b')
    (letters only, 2+ characters, sklearn's default English stopwords,
    words in ≥2 documents to drop hapaxes)
  - Consecutive-year pairs only: year_curr - year_prev == 1 in filing_date.
    Firms with fiscal shifts (multiple 10-Ks per calendar year) resolved
    by keeping the latest 10-K per (ticker, year). 3 firms affected in
    2019-2024 sample (SMCI 2019, SYM 2022, TLRY 2021).
  - Cross-references dropped before fitting: sections flagged as
    `is_cross_reference=True` in the extractor (typically Item 7 in
    pre-iXBRL filings that reference exhibits).

Output panel columns (in order):
  ticker
  filing_year   — calendar year of the current 10-K (year N)
  prior_filing_year — calendar year of the prior 10-K (year N-1)
  section_id    — one of "Item 1", "Item 1A", "Item 7"
  similarity    — cosine similarity of TF-IDF vectors, ∈ [0, 1]

Deviation from other signals' convention
----------------------------------------
Unlike GP, PEAD, ResMom, IVol which take (start_date, end_date)
parameters, this signal does not. The input sections corpus is already
pre-filtered to the sample window (2019-2024) by the corpus construction
pipeline; further windowing at compute time would produce inconsistent
year-over-year pairs. Sample window is a corpus-level decision, not a
compute-time choice.

Keying is also different: (ticker, filing_year, section_id) rather than
(permno, date_filed). Downstream alignment must handle this.

Usage
-----
    from axiom_fund.signals.lazy_prices import compute_lazy_prices_signal

    sections_df = pd.read_parquet("data/cache/lazy_prices_sections.parquet")
    raw = compute_lazy_prices_signal(sections_df)
    # raw: (ticker, filing_year, prior_filing_year, section_id, similarity)
"""
# ruff: noqa: I001

from __future__ import annotations

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


LAZY_PRICES_SECTIONS = ["Item 1", "Item 1A", "Item 7"]

# Pre-committed TF-IDF parameters (from July 6 pilot).
# See scripts/analysis/prototype_lazy_prices_signal.py.
_TFIDF_PARAMS = dict(
    lowercase=True,
    stop_words="english",
    min_df=2,
    token_pattern=r"(?u)\b[a-zA-Z]{2,}\b",
)


LAZY_PRICES_COLUMNS = [
    "ticker", "filing_year", "prior_filing_year", "section_id", "similarity",
]


def compute_lazy_prices_signal(sections_df: pd.DataFrame) -> pd.DataFrame:
    """Compute the raw Lazy Prices year-over-year similarity signal.

    Parameters
    ----------
    sections_df : pd.DataFrame
        Extracted 10-K sections corpus. Must contain columns:
        'ticker', 'filing_date', 'accession_no', 'section_id', 'text',
        'is_cross_reference'.
        Expected input: data/cache/lazy_prices_sections.parquet.

    Returns
    -------
    pd.DataFrame
        Signal panel with columns matching LAZY_PRICES_COLUMNS. One row
        per (ticker, filing_year, section_id) with a valid consecutive
        year-over-year pair. Sorted by (section_id, ticker, filing_year).

    Raises
    ------
    ValueError
        If required columns are missing from sections_df.
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    required_cols = {"ticker", "filing_date", "accession_no", "section_id",
                     "text", "is_cross_reference"}
    missing = required_cols - set(sections_df.columns)
    if missing:
        raise ValueError(
            f"sections_df missing required columns: {sorted(missing)}"
        )

    df = sections_df.copy()

    # Handle empty input early: pandas can lose columns on boolean-indexing
    # an empty DataFrame with dtype-object columns.
    if df.empty:
        return pd.DataFrame(columns=LAZY_PRICES_COLUMNS)

    # Drop cross-references (methodology from pilot)
    df = df[~df.is_cross_reference]

    # Extract filing year for pairing
    df["filing_year"] = pd.to_datetime(df["filing_date"]).dt.year

    # Fiscal-shift handling: for firms with >1 filing per calendar year,
    # keep only the LATEST accession per (ticker, year). See design doc
    # limitation 11 and this session's decision (2026-07-11).
    latest_per_year = (
        df.sort_values("filing_date")
        .groupby(["ticker", "filing_year"], as_index=False)
        .accession_no.last()
    )
    df = df.merge(
        latest_per_year, on=["ticker", "filing_year", "accession_no"]
    )

    # ------------------------------------------------------------------
    # Per-section similarity computation
    # ------------------------------------------------------------------
    all_results: list[dict] = []

    for section in LAZY_PRICES_SECTIONS:
        section_df = df[df.section_id == section].reset_index(drop=True)
        if len(section_df) < 2:
            continue

        # Fit TF-IDF vectorizer with pre-committed parameters
        vectorizer = TfidfVectorizer(**_TFIDF_PARAMS)
        tfidf = vectorizer.fit_transform(section_df["text"].tolist())

        # For each ticker with >1 filing, compute consecutive-year pairs
        section_df["row_idx"] = section_df.index
        for ticker in section_df.ticker.unique():
            ticker_df = section_df[section_df.ticker == ticker].sort_values(
                "filing_year"
            )
            if len(ticker_df) < 2:
                continue
            years = ticker_df.filing_year.values
            row_idxs = ticker_df.row_idx.values
            for i in range(len(years) - 1):
                y1, y2 = int(years[i]), int(years[i + 1])
                if y2 - y1 != 1:
                    continue
                sim = float(
                    cosine_similarity(
                        tfidf[row_idxs[i]], tfidf[row_idxs[i + 1]]
                    )[0, 0]
                )
                all_results.append({
                    "ticker": ticker,
                    "filing_year": y2,
                    "prior_filing_year": y1,
                    "section_id": section,
                    "similarity": sim,
                })

    if not all_results:
        return pd.DataFrame(columns=LAZY_PRICES_COLUMNS)

    signal_df = pd.DataFrame(all_results)[LAZY_PRICES_COLUMNS]
    signal_df = signal_df.sort_values(
        ["section_id", "ticker", "filing_year"]
    ).reset_index(drop=True)
    return signal_df