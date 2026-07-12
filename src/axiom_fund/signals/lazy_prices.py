"""Lazy Prices raw signal (Cohen-Malloy-Nguyen 2020).

Signal
------
For each firm, at each 10-K filing date, compute the year-over-year
TF-IDF cosine similarity between the current 10-K and the prior
10-K, separately for Items 1, 1A, and 7. The three per-section
similarities are averaged, and the raw signal is:

    raw_signal = 1 - mean(sim_item1, sim_item1a, sim_item7)

Higher raw_signal → more textual change → CMN predicts subsequent
*outperformance*. (CMN's headline finding is that "lazy" filings
with LOW change underperform; the sign convention here maps
directly onto that.)

Per docs/signal_design.md §2.1: signals output at natural cadence.
Lazy Prices natural cadence is per-10-K-filing. The portfolio layer
(alignment) will forward-fill until the next filing per §2.1's
frequency-alignment principle. Winsorization and cross-sectional
z-scoring are alignment-layer concerns per the same section,
revised 2026-04-26.

Methodology
-----------
Pre-committed parameters (locked at July 6 pilot, see
scripts/analysis/prototype_lazy_prices_signal.py):

  - Per-section TF-IDF fit: each of Item 1, 1A, 7 has its own
    vocabulary (they use different language and shouldn't share
    tokens).
  - TfidfVectorizer(lowercase=True, stop_words='english', min_df=2,
                    token_pattern=r'(?u)\\b[a-zA-Z]{2,}\\b')
  - Consecutive-year pairs only: filing_year(current) -
    filing_year(prior) == 1. Firms with fiscal shifts (multiple
    10-Ks per calendar year) resolved by keeping the latest 10-K
    per (permno, calendar year). 3 firms affected in 2019-2024
    sample (SMCI 2019, SYM 2022, TLRY 2021).
  - Cross-references dropped before fitting: sections flagged
    `is_cross_reference=True` by the extractor (typically Item 7
    in pre-iXBRL filings that reference exhibits).
  - Aggregation across sections: equal-weighted mean. This is a
    documented ad-hoc decision (see docs/v2_item6_design.md
    "Backtest scope 2026-07-12"). CMN 2020 also uses equal-weighted
    aggregation across sections.

Output panel columns (in order, matches LAZY_PRICES_RAW_COLUMNS):
  permno
  ticker              — audit trail (per GP's gvkey pattern)
  date_filed          — filing_date of the current 10-K (PIT anchor)
  prior_date_filed    — filing_date of the prior 10-K used for YoY
  sim_item1           — component: Item 1 YoY cosine similarity
  sim_item1a          — component: Item 1A YoY cosine similarity
  sim_item7           — component: Item 7 YoY cosine similarity
  raw_signal          — 1 - mean(sim_item1, sim_item1a, sim_item7)

Components are retained per GP's pattern of preserving the constituent
inputs for auditability.

NaN handling
------------
If a section is missing for a filing (cross-reference dropped, or
section not extracted), that section's similarity is omitted from
the mean. If ALL three sections are missing for either the current
or prior filing, no signal row is emitted for that (permno, date_filed).

Deviation from GP/PEAD/ResMom/IVol
-----------------------------------
No start_date/end_date parameters: the input sections corpus is
already pre-filtered to the sample window (2019-2024) by the corpus
construction pipeline; further windowing here would produce
inconsistent year-over-year pairs.

Usage
-----
    from axiom_fund.signals.lazy_prices import compute_lazy_prices_signal

    # sections_df must include a 'permno' column; the runner enriches
    # from data/cache/lazy_prices_ciks.parquet before calling.
    raw = compute_lazy_prices_signal(sections_df)
"""
# ruff: noqa: I001

from __future__ import annotations

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


LAZY_PRICES_SECTIONS = ["Item 1", "Item 1A", "Item 7"]

# Column name suffix used to store per-section similarities.
_SECTION_TO_COL = {
    "Item 1": "sim_item1",
    "Item 1A": "sim_item1a",
    "Item 7": "sim_item7",
}


# Pre-committed TF-IDF parameters (July 6 pilot).
_TFIDF_PARAMS = dict(
    lowercase=True,
    stop_words="english",
    min_df=2,
    token_pattern=r"(?u)\b[a-zA-Z]{2,}\b",
)


LAZY_PRICES_RAW_COLUMNS: tuple[str, ...] = (
    "permno",
    "ticker",
    "date_filed",
    "prior_date_filed",
    "sim_item1",
    "sim_item1a",
    "sim_item7",
    "raw_signal",
)


REQUIRED_INPUT_COLUMNS = {
    "permno", "ticker", "filing_date", "accession_no", "section_id",
    "text", "is_cross_reference",
}


def compute_lazy_prices_signal(sections_df: pd.DataFrame) -> pd.DataFrame:
    """Compute the raw Lazy Prices YoY similarity signal per firm-filing.

    Parameters
    ----------
    sections_df : pd.DataFrame
        Extracted 10-K sections corpus, with permno enrichment.
        Required columns: 'permno', 'ticker', 'filing_date',
        'accession_no', 'section_id', 'text', 'is_cross_reference'.
        Expected input: data/cache/lazy_prices_sections.parquet
        joined with data/cache/lazy_prices_ciks.parquet for permno.

    Returns
    -------
    pd.DataFrame
        Long-format raw signal panel sorted by (date_filed, permno)
        with columns matching LAZY_PRICES_RAW_COLUMNS.
        One row per (permno, date_filed) with at least one non-null
        per-section similarity in both current and prior filing.

    Raises
    ------
    ValueError
        If required columns are missing from sections_df.
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    missing = REQUIRED_INPUT_COLUMNS - set(sections_df.columns)
    if missing:
        raise ValueError(
            f"sections_df missing required columns: {sorted(missing)}"
        )

    df = sections_df.copy()

    # Handle empty input early: pandas can lose columns on boolean-indexing
    # an empty DataFrame with dtype-object columns.
    if df.empty:
        return pd.DataFrame(columns=list(LAZY_PRICES_RAW_COLUMNS))

    # Drop cross-references (methodology from pilot)
    df = df[~df.is_cross_reference]

    # Normalize filing_date to date type; add filing_year for pairing.
    df["filing_date"] = pd.to_datetime(df["filing_date"])
    df["filing_year"] = df["filing_date"].dt.year

    # Fiscal-shift handling: for firms with >1 filing per calendar year,
    # keep only the LATEST accession per (permno, year). See design doc
    # limitation 11 and Backtest scope (2026-07-12).
    latest_per_year = (
        df.sort_values("filing_date")
        .groupby(["permno", "filing_year"], as_index=False)
        .accession_no.last()
    )
    df = df.merge(
        latest_per_year, on=["permno", "filing_year", "accession_no"]
    )

    # ------------------------------------------------------------------
    # Per-section similarity computation
    # ------------------------------------------------------------------
    # Build a dict keyed on (permno, filing_year, section) -> similarity
    # of current filing vs. prior year. Then aggregate to
    # (permno, date_filed) rows.

    per_pair_rows: list[dict] = []

    for section in LAZY_PRICES_SECTIONS:
        section_df = df[df.section_id == section].reset_index(drop=True)
        if len(section_df) < 2:
            continue

        # Fit TF-IDF vectorizer with pre-committed parameters
        vectorizer = TfidfVectorizer(**_TFIDF_PARAMS)
        tfidf = vectorizer.fit_transform(section_df["text"].tolist())

        # For each permno with >1 filing in this section, compute
        # consecutive-year pair similarities.
        section_df["row_idx"] = section_df.index
        for permno in section_df.permno.unique():
            firm_df = section_df[section_df.permno == permno].sort_values(
                "filing_year"
            )
            if len(firm_df) < 2:
                continue
            years = firm_df.filing_year.values
            dates = firm_df.filing_date.values
            row_idxs = firm_df.row_idx.values
            ticker = firm_df.ticker.iloc[0]
            for i in range(len(years) - 1):
                y1, y2 = int(years[i]), int(years[i + 1])
                if y2 - y1 != 1:
                    continue
                sim = float(
                    cosine_similarity(
                        tfidf[row_idxs[i]], tfidf[row_idxs[i + 1]]
                    )[0, 0]
                )
                per_pair_rows.append({
                    "permno": permno,
                    "ticker": ticker,
                    "date_filed": pd.Timestamp(dates[i + 1]),
                    "prior_date_filed": pd.Timestamp(dates[i]),
                    "section_col": _SECTION_TO_COL[section],
                    "similarity": sim,
                })

    if not per_pair_rows:
        return pd.DataFrame(columns=list(LAZY_PRICES_RAW_COLUMNS))

    # ------------------------------------------------------------------
    # Aggregate per-section rows to (permno, date_filed) signal rows
    # ------------------------------------------------------------------
    long_df = pd.DataFrame(per_pair_rows)
    wide = long_df.pivot_table(
        index=["permno", "ticker", "date_filed", "prior_date_filed"],
        columns="section_col",
        values="similarity",
        aggfunc="first",
    ).reset_index()

    # Ensure all three section columns exist even if no data for one.
    for col in ("sim_item1", "sim_item1a", "sim_item7"):
        if col not in wide.columns:
            wide[col] = pd.NA

    # raw_signal = 1 - row-wise mean of the 3 section similarities.
    # If all three are NaN, raw_signal is NaN (and row will be dropped).
    sim_cols = ["sim_item1", "sim_item1a", "sim_item7"]
    wide["raw_signal"] = 1.0 - wide[sim_cols].mean(axis=1, skipna=True)

    # Drop rows where raw_signal couldn't be computed (all 3 sections NaN).
    wide = wide[wide["raw_signal"].notna()]

    # Sort and column-order per LAZY_PRICES_RAW_COLUMNS.
    result = (
        wide[list(LAZY_PRICES_RAW_COLUMNS)]
        .sort_values(["date_filed", "permno"])
        .reset_index(drop=True)
    )
    result.columns.name = None
    return result
