"""Tests for the Lazy Prices signal module.

Covers input validation, cross-reference filtering, fiscal-shift
handling (take latest), consecutive-year pairing, TF-IDF math,
and schema/sort output guarantees.
"""

from __future__ import annotations

import pandas as pd
import pytest

from axiom_fund.signals.lazy_prices import (
    LAZY_PRICES_RAW_COLUMNS,
    LAZY_PRICES_SECTIONS,
    compute_lazy_prices_signal,
)


def _make_section_row(
    permno, ticker, filing_date, accession_no, section_id, text,
    is_cross_reference=False,
):
    return {
        "permno": permno,
        "ticker": ticker,
        "filing_date": pd.to_datetime(filing_date).date(),
        "accession_no": accession_no,
        "section_id": section_id,
        "text": text,
        "is_cross_reference": is_cross_reference,
    }


def _make_sections_df(rows):
    return pd.DataFrame(rows)


def _all_three(permno, ticker, filing_date, accession_no, text_by_section):
    return [
        _make_section_row(permno, ticker, filing_date, accession_no, s, text_by_section[s])
        for s in LAZY_PRICES_SECTIONS
    ]


_TEXT_A = ("The company designs and manufactures consumer electronics products. "
           "Revenue growth was driven by strong demand for our smartphone lineup. "
           "We continue to invest in research and development activities. "
           "Our supply chain remains diversified across multiple regions.") * 5

_TEXT_B = _TEXT_A

_TEXT_C = ("Restaurant operations expanded through franchise agreements globally. "
           "Same-store sales increased in most geographic segments this year. "
           "Menu innovations drove customer traffic across our brand portfolio. "
           "Delivery partnerships now account for a growing revenue mix.") * 5


class TestInputValidation:
    def test_missing_columns_raises(self):
        df = pd.DataFrame({"ticker": ["A"], "text": ["x"]})
        with pytest.raises(ValueError, match="missing required columns"):
            compute_lazy_prices_signal(df)

    def test_missing_permno_specifically_raises(self):
        df = pd.DataFrame({
            "ticker": ["A"], "filing_date": ["2020-01-01"],
            "accession_no": ["a"], "section_id": ["Item 1"],
            "text": ["hello"], "is_cross_reference": [False],
        })
        with pytest.raises(ValueError, match="permno"):
            compute_lazy_prices_signal(df)

    def test_empty_input_returns_empty_frame(self):
        df = pd.DataFrame(columns=[
            "permno", "ticker", "filing_date", "accession_no",
            "section_id", "text", "is_cross_reference",
        ])
        result = compute_lazy_prices_signal(df)
        assert list(result.columns) == list(LAZY_PRICES_RAW_COLUMNS)
        assert len(result) == 0


class TestCrossReferenceFilter:
    def test_all_year_n_cross_referenced_drops_row(self):
        rows = []
        rows.extend(_all_three(100, "AAA", "2019-03-01", "acc-a19",
                               {s: _TEXT_A for s in LAZY_PRICES_SECTIONS}))
        for s in LAZY_PRICES_SECTIONS:
            rows.append(_make_section_row(
                100, "AAA", "2020-03-01", "acc-a20", s, _TEXT_A,
                is_cross_reference=True,
            ))
        rows.extend(_all_three(200, "BBB", "2019-03-01", "acc-b19",
                               {s: _TEXT_B for s in LAZY_PRICES_SECTIONS}))
        rows.extend(_all_three(200, "BBB", "2020-03-01", "acc-b20",
                               {s: _TEXT_B for s in LAZY_PRICES_SECTIONS}))
        result = compute_lazy_prices_signal(_make_sections_df(rows))
        assert 100 not in result.permno.values
        assert 200 in result.permno.values


class TestConsecutiveYearOnly:
    def test_gap_year_skipped(self):
        rows = []
        for date, acc in [("2019-03-01", "a19"), ("2020-03-01", "a20"),
                          ("2022-03-01", "a22")]:
            rows.extend(_all_three(100, "AAA", date, acc,
                                   {s: _TEXT_A for s in LAZY_PRICES_SECTIONS}))
        rows.extend(_all_three(200, "BBB", "2019-03-01", "b19",
                               {s: _TEXT_B for s in LAZY_PRICES_SECTIONS}))
        rows.extend(_all_three(200, "BBB", "2020-03-01", "b20",
                               {s: _TEXT_B for s in LAZY_PRICES_SECTIONS}))
        result = compute_lazy_prices_signal(_make_sections_df(rows))
        aaa = result[result.permno == 100]
        assert len(aaa) == 1
        assert aaa.iloc[0].date_filed.year == 2020


class TestFiscalShift:
    def test_latest_filing_per_year_kept(self):
        rows = []
        rows.extend(_all_three(100, "AAA", "2019-01-15", "early",
                               {s: _TEXT_A for s in LAZY_PRICES_SECTIONS}))
        rows.extend(_all_three(100, "AAA", "2019-11-15", "late",
                               {s: _TEXT_C for s in LAZY_PRICES_SECTIONS}))
        rows.extend(_all_three(100, "AAA", "2020-11-15", "y2020",
                               {s: _TEXT_C for s in LAZY_PRICES_SECTIONS}))
        rows.extend(_all_three(200, "BBB", "2019-03-01", "b19",
                               {s: _TEXT_B for s in LAZY_PRICES_SECTIONS}))
        rows.extend(_all_three(200, "BBB", "2020-03-01", "b20",
                               {s: _TEXT_B for s in LAZY_PRICES_SECTIONS}))
        result = compute_lazy_prices_signal(_make_sections_df(rows))
        aaa = result[result.permno == 100]
        assert len(aaa) == 1
        assert aaa.iloc[0].raw_signal < 0.05


class TestSignalMath:
    def test_identical_texts_give_raw_signal_near_zero(self):
        rows = []
        for permno, ticker in [(100, "AAA"), (200, "BBB")]:
            for date, acc in [("2019-03-01", ticker + "19"),
                              ("2020-03-01", ticker + "20")]:
                rows.extend(_all_three(permno, ticker, date, acc,
                                       {s: _TEXT_A for s in LAZY_PRICES_SECTIONS}))
        result = compute_lazy_prices_signal(_make_sections_df(rows))
        assert len(result) == 2
        assert (result.raw_signal < 1e-6).all()

    def test_disjoint_vocab_gives_higher_raw_signal(self):
        rows = []
        for date, acc in [("2019-03-01", "a19"), ("2020-03-01", "a20")]:
            rows.extend(_all_three(100, "AAA", date, acc,
                                   {s: _TEXT_A for s in LAZY_PRICES_SECTIONS}))
        rows.extend(_all_three(200, "BBB", "2019-03-01", "b19",
                               {s: _TEXT_A for s in LAZY_PRICES_SECTIONS}))
        rows.extend(_all_three(200, "BBB", "2020-03-01", "b20",
                               {s: _TEXT_C for s in LAZY_PRICES_SECTIONS}))
        result = compute_lazy_prices_signal(_make_sections_df(rows))
        aaa = result[result.permno == 100].iloc[0]
        bbb = result[result.permno == 200].iloc[0]
        assert aaa.raw_signal < 1e-6
        assert bbb.raw_signal > aaa.raw_signal
        assert bbb.raw_signal > 0.3


class TestMissingSection:
    def test_row_produced_with_partial_sections(self):
        rows = [
            _make_section_row(100, "AAA", "2019-03-01", "a19", "Item 1", _TEXT_A),
            _make_section_row(100, "AAA", "2020-03-01", "a20", "Item 1", _TEXT_A),
            _make_section_row(200, "BBB", "2019-03-01", "b19", "Item 1", _TEXT_A),
            _make_section_row(200, "BBB", "2020-03-01", "b20", "Item 1", _TEXT_A),
        ]
        result = compute_lazy_prices_signal(_make_sections_df(rows))
        assert len(result) == 2
        for _, row in result.iterrows():
            assert row.sim_item1 == pytest.approx(1.0, abs=1e-6)
            assert pd.isna(row.sim_item1a)
            assert pd.isna(row.sim_item7)
            assert row.raw_signal == pytest.approx(0.0, abs=1e-6)


class TestOutputSchema:
    def test_columns_match_locked_schema(self):
        rows = []
        for permno, ticker in [(100, "AAA"), (200, "BBB")]:
            for date, acc in [("2019-03-01", ticker + "19"),
                              ("2020-03-01", ticker + "20")]:
                rows.extend(_all_three(permno, ticker, date, acc,
                                       {s: _TEXT_A for s in LAZY_PRICES_SECTIONS}))
        result = compute_lazy_prices_signal(_make_sections_df(rows))
        assert tuple(result.columns) == LAZY_PRICES_RAW_COLUMNS

    def test_sorted_by_date_filed_then_permno(self):
        rows = []
        for permno, ticker in [(200, "BBB"), (100, "AAA")]:
            for date, acc in [("2019-03-01", ticker + "19"),
                              ("2020-03-01", ticker + "20")]:
                rows.extend(_all_three(permno, ticker, date, acc,
                                       {s: _TEXT_A for s in LAZY_PRICES_SECTIONS}))
        result = compute_lazy_prices_signal(_make_sections_df(rows))
        assert result.iloc[0].permno == 100
        assert result.iloc[1].permno == 200
