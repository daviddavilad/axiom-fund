"""Tests for the Lazy Prices signal module.

Covers input validation, cross-reference filtering, fiscal-shift
handling (take latest), consecutive-year pairing, and TF-IDF math.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from axiom_fund.signals.lazy_prices import (
    LAZY_PRICES_COLUMNS,
    LAZY_PRICES_SECTIONS,
    compute_lazy_prices_signal,
)


# ============================================================================
# Helpers
# ============================================================================


def _make_section_row(
    ticker: str,
    filing_date: str,
    accession_no: str,
    section_id: str,
    text: str,
    is_cross_reference: bool = False,
    length: int | None = None,
) -> dict:
    """Build one section-row dict matching the corpus schema."""
    return {
        "ticker": ticker,
        "filing_date": pd.to_datetime(filing_date).date(),
        "accession_no": accession_no,
        "section_id": section_id,
        "text": text,
        "is_cross_reference": is_cross_reference,
        "length": length if length is not None else len(text),
    }


def _make_sections_df(rows: list[dict]) -> pd.DataFrame:
    """Build a sections DataFrame from a list of row dicts."""
    return pd.DataFrame(rows)


# Reusable long text bodies for TF-IDF tests. Real English words so
# stop-word removal + min_df=2 don't wipe them out.
_TEXT_A = (
    "The company designs and manufactures consumer electronics products. "
    "Revenue growth was driven by strong demand for our smartphone lineup. "
    "We continue to invest in research and development activities. "
    "Our supply chain remains diversified across multiple regions."
) * 5

_TEXT_B = (
    "The company designs and manufactures consumer electronics products. "
    "Revenue growth was driven by strong demand for our smartphone lineup. "
    "We continue to invest in research and development activities. "
    "Our supply chain remains diversified across multiple regions."
) * 5

_TEXT_C = (
    "Restaurant operations expanded through franchise agreements globally. "
    "Same-store sales increased in most geographic segments this year. "
    "Menu innovations drove customer traffic across our brand portfolio. "
    "Delivery partnerships now account for a growing revenue mix."
) * 5


# ============================================================================
# Input validation
# ============================================================================


class TestInputValidation:
    def test_missing_columns_raises(self) -> None:
        df = pd.DataFrame({"ticker": ["A"], "text": ["x"]})
        with pytest.raises(ValueError, match="missing required columns"):
            compute_lazy_prices_signal(df)

    def test_empty_input_returns_empty_frame(self) -> None:
        df = _make_sections_df([])
        # Empty DataFrames need the schema columns for the validation to pass
        df = pd.DataFrame(
            columns=[
                "ticker", "filing_date", "accession_no", "section_id",
                "text", "is_cross_reference",
            ]
        )
        result = compute_lazy_prices_signal(df)
        assert list(result.columns) == LAZY_PRICES_COLUMNS
        assert len(result) == 0


# ============================================================================
# Cross-reference filtering
# ============================================================================


class TestCrossReferenceFilter:
    def test_cross_references_dropped(self) -> None:
        """Cross-refs should be filtered before pairing."""
        rows = [
            _make_section_row("AAA", "2019-03-01", "acc-2019", "Item 1",
                              _TEXT_A, is_cross_reference=False),
            _make_section_row("AAA", "2020-03-01", "acc-2020", "Item 1",
                              _TEXT_A, is_cross_reference=True),
            _make_section_row("BBB", "2019-03-01", "acc-2019b", "Item 1",
                              _TEXT_B, is_cross_reference=False),
            _make_section_row("BBB", "2020-03-01", "acc-2020b", "Item 1",
                              _TEXT_B, is_cross_reference=False),
        ]
        result = compute_lazy_prices_signal(_make_sections_df(rows))
        # AAA has a cross-ref for 2020, so no valid pair
        aaa_pairs = result[result.ticker == "AAA"]
        bbb_pairs = result[result.ticker == "BBB"]
        assert len(aaa_pairs) == 0
        assert len(bbb_pairs) == 1


# ============================================================================
# Year pairing
# ============================================================================


class TestConsecutiveYearOnly:
    def test_gap_year_skipped(self) -> None:
        """Filings in 2019, 2020, 2022 → only 2019→2020 pair (2021 missing)."""
        rows = [
            _make_section_row("AAA", "2019-03-01", "acc-2019", "Item 1", _TEXT_A),
            _make_section_row("AAA", "2020-03-01", "acc-2020", "Item 1", _TEXT_A),
            _make_section_row("AAA", "2022-03-01", "acc-2022", "Item 1", _TEXT_A),
            _make_section_row("BBB", "2019-03-01", "acc-b19", "Item 1", _TEXT_B),
            _make_section_row("BBB", "2020-03-01", "acc-b20", "Item 1", _TEXT_B),
        ]
        result = compute_lazy_prices_signal(_make_sections_df(rows))
        aaa = result[result.ticker == "AAA"]
        # Only the 2019→2020 pair should exist
        assert len(aaa) == 1
        assert aaa.iloc[0].filing_year == 2020
        assert aaa.iloc[0].prior_filing_year == 2019


class TestFiscalShift:
    def test_latest_filing_per_year_kept(self) -> None:
        """Two 10-Ks in same calendar year → keep only the latest."""
        rows = [
            # AAA has TWO filings in 2019 (fiscal shift)
            _make_section_row("AAA", "2019-01-15", "acc-early",
                              "Item 1", _TEXT_A),
            _make_section_row("AAA", "2019-11-15", "acc-late",
                              "Item 1", _TEXT_C),
            _make_section_row("AAA", "2020-11-15", "acc-2020",
                              "Item 1", _TEXT_C),
            _make_section_row("BBB", "2019-03-01", "acc-b19",
                              "Item 1", _TEXT_B),
            _make_section_row("BBB", "2020-03-01", "acc-b20",
                              "Item 1", _TEXT_B),
        ]
        result = compute_lazy_prices_signal(_make_sections_df(rows))
        aaa = result[result.ticker == "AAA"]
        # AAA 2019→2020 pair should use the LATE 2019 filing (_TEXT_C)
        # AAA 2019-late is _TEXT_C, 2020 is _TEXT_C → similarity should be 1.0
        assert len(aaa) == 1
        assert aaa.iloc[0].similarity > 0.99


# ============================================================================
# TF-IDF math
# ============================================================================


class TestSimilarityMath:
    def test_identical_texts_give_similarity_one(self) -> None:
        rows = [
            _make_section_row("AAA", "2019-03-01", "acc-a19", "Item 1", _TEXT_A),
            _make_section_row("AAA", "2020-03-01", "acc-a20", "Item 1", _TEXT_A),
            # BBB needed to build a vocab (min_df=2 requires ≥2 docs per token)
            _make_section_row("BBB", "2019-03-01", "acc-b19", "Item 1", _TEXT_A),
            _make_section_row("BBB", "2020-03-01", "acc-b20", "Item 1", _TEXT_A),
        ]
        result = compute_lazy_prices_signal(_make_sections_df(rows))
        assert len(result) == 2
        assert result.similarity.iloc[0] == pytest.approx(1.0, abs=1e-9)
        assert result.similarity.iloc[1] == pytest.approx(1.0, abs=1e-9)

    def test_disjoint_vocab_gives_low_similarity(self) -> None:
        """Firms with entirely different vocab should have similarity ~0."""
        rows = [
            # AAA uses _TEXT_A vocab both years, keeps stable
            _make_section_row("AAA", "2019-03-01", "acc-a19", "Item 1", _TEXT_A),
            _make_section_row("AAA", "2020-03-01", "acc-a20", "Item 1", _TEXT_A),
            # BBB shifts from _TEXT_A to _TEXT_C (disjoint)
            _make_section_row("BBB", "2019-03-01", "acc-b19", "Item 1", _TEXT_A),
            _make_section_row("BBB", "2020-03-01", "acc-b20", "Item 1", _TEXT_C),
        ]
        result = compute_lazy_prices_signal(_make_sections_df(rows))
        bbb = result[result.ticker == "BBB"].iloc[0]
        # Different vocabularies but share stopwords + common words; expect
        # meaningfully low similarity, not necessarily zero
        assert bbb.similarity < 0.5


# ============================================================================
# Output schema
# ============================================================================


class TestOutputSchema:
    def test_columns_match_canonical(self) -> None:
        rows = [
            _make_section_row("AAA", "2019-03-01", "acc-a19", "Item 1", _TEXT_A),
            _make_section_row("AAA", "2020-03-01", "acc-a20", "Item 1", _TEXT_A),
            _make_section_row("BBB", "2019-03-01", "acc-b19", "Item 1", _TEXT_B),
            _make_section_row("BBB", "2020-03-01", "acc-b20", "Item 1", _TEXT_B),
        ]
        result = compute_lazy_prices_signal(_make_sections_df(rows))
        assert list(result.columns) == LAZY_PRICES_COLUMNS

    def test_sorted_by_section_ticker_year(self) -> None:
        rows = [
            _make_section_row("AAA", "2019-03-01", "acc-a19", "Item 1", _TEXT_A),
            _make_section_row("AAA", "2020-03-01", "acc-a20", "Item 1", _TEXT_A),
            _make_section_row("BBB", "2019-03-01", "acc-b19", "Item 1", _TEXT_B),
            _make_section_row("BBB", "2020-03-01", "acc-b20", "Item 1", _TEXT_B),
            _make_section_row("AAA", "2019-03-01", "acc-a19", "Item 1A", _TEXT_A),
            _make_section_row("AAA", "2020-03-01", "acc-a20", "Item 1A", _TEXT_A),
            _make_section_row("BBB", "2019-03-01", "acc-b19", "Item 1A", _TEXT_B),
            _make_section_row("BBB", "2020-03-01", "acc-b20", "Item 1A", _TEXT_B),
        ]
        result = compute_lazy_prices_signal(_make_sections_df(rows))
        # First rows should be Item 1 (before Item 1A alphabetically)
        assert result.iloc[0].section_id == "Item 1"
        # Within Item 1, AAA before BBB
        item1 = result[result.section_id == "Item 1"]
        assert item1.iloc[0].ticker == "AAA"