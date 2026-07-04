"""Unit tests for SectionExtractor.

Tests EdgartoolsSectionExtractor by mocking edgartools' Filing and TenK
objects at the interface boundary. Real edgartools integration is
validated separately via the pilot verification script (not part of
the unit test suite because it requires network access).
"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import pytest

from axiom_fund.data.section_extractor import (
    DEFAULT_CROSS_REFERENCE_THRESHOLD,
    EXTRACTED_SECTIONS,
    MINIMUM_SUPPORTED_YEAR,
    EdgartoolsSectionExtractor,
    SectionResult,
)


# ============================================================================
# Fixtures
# ============================================================================


def _make_mock_filing(
    filing_year: int,
    business: str | None = None,
    risk_factors: str | None = None,
    management_discussion: str | None = None,
    obj_raises: type[Exception] | None = None,
) -> MagicMock:
    """Build a mock edgar.Filing whose obj() returns a mock TenK."""
    filing = MagicMock()
    filing.filing_date = date(filing_year, 6, 15)

    if obj_raises is not None:
        filing.obj.side_effect = obj_raises("mock obj() failure")
        return filing

    tenk = MagicMock()
    # Set section attributes explicitly (None means "don't set")
    if business is not None:
        tenk.business = business
    else:
        tenk.business = None
    if risk_factors is not None:
        tenk.risk_factors = risk_factors
    else:
        tenk.risk_factors = None
    if management_discussion is not None:
        tenk.management_discussion = management_discussion
    else:
        tenk.management_discussion = None

    filing.obj.return_value = tenk
    return filing


# ============================================================================
# Constants
# ============================================================================


class TestConstants:
    def test_extracted_sections_matches_scope(self) -> None:
        # Item 7A must NOT be in the extracted set (excluded per A' design)
        assert "Item 7A" not in EXTRACTED_SECTIONS
        assert set(EXTRACTED_SECTIONS) == {"Item 1", "Item 1A", "Item 7"}

    def test_minimum_year_is_iXBRL_transition(self) -> None:
        assert MINIMUM_SUPPORTED_YEAR == 2019

    def test_default_threshold_is_positive(self) -> None:
        assert DEFAULT_CROSS_REFERENCE_THRESHOLD > 0


# ============================================================================
# EdgartoolsSectionExtractor initialization
# ============================================================================


class TestInitialization:
    def test_default_threshold(self) -> None:
        ext = EdgartoolsSectionExtractor()
        assert ext._threshold == DEFAULT_CROSS_REFERENCE_THRESHOLD

    def test_custom_threshold(self) -> None:
        ext = EdgartoolsSectionExtractor(cross_reference_threshold=500)
        assert ext._threshold == 500

    def test_negative_threshold_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            EdgartoolsSectionExtractor(cross_reference_threshold=-1)


# ============================================================================
# Window guard
# ============================================================================


class TestWindowGuard:
    def test_pre_2019_raises(self) -> None:
        ext = EdgartoolsSectionExtractor()
        filing = _make_mock_filing(filing_year=2018)
        with pytest.raises(ValueError, match="MINIMUM_SUPPORTED_YEAR"):
            ext.extract(filing)

    def test_2019_accepted(self) -> None:
        ext = EdgartoolsSectionExtractor()
        filing = _make_mock_filing(
            filing_year=2019,
            business="x" * 5000,
            risk_factors="x" * 5000,
            management_discussion="x" * 5000,
        )
        # Should not raise
        results = ext.extract(filing)
        assert len(results) == 3


# ============================================================================
# Extraction happy path
# ============================================================================


class TestExtractionHappyPath:
    def test_all_three_sections_extracted(self) -> None:
        ext = EdgartoolsSectionExtractor()
        filing = _make_mock_filing(
            filing_year=2023,
            business="x" * 15000,
            risk_factors="x" * 70000,
            management_discussion="x" * 20000,
        )
        results = ext.extract(filing)
        assert set(results.keys()) == {"Item 1", "Item 1A", "Item 7"}
        assert results["Item 1"].length == 15000
        assert results["Item 1A"].length == 70000
        assert results["Item 7"].length == 20000
        # All above threshold: none should be flagged
        for r in results.values():
            assert not r.is_cross_reference

    def test_section_result_fields_populated(self) -> None:
        ext = EdgartoolsSectionExtractor()
        filing = _make_mock_filing(
            filing_year=2023,
            business="business text",
            risk_factors="risk text",
            management_discussion="mda text",
        )
        results = ext.extract(filing)
        r = results["Item 1"]
        assert r.section_id == "Item 1"
        assert r.text == "business text"
        assert r.length == len("business text")


# ============================================================================
# Cross-reference detection
# ============================================================================


class TestCrossReferenceDetection:
    def test_short_section_flagged(self) -> None:
        ext = EdgartoolsSectionExtractor()
        filing = _make_mock_filing(
            filing_year=2023,
            business="x" * 15000,
            risk_factors="x" * 70000,
            management_discussion="See pages 64-169",  # short = cross-ref
        )
        results = ext.extract(filing)
        assert results["Item 7"].is_cross_reference is True
        assert results["Item 1"].is_cross_reference is False
        assert results["Item 1A"].is_cross_reference is False

    def test_threshold_boundary(self) -> None:
        # At exactly the threshold: NOT flagged (uses < comparison)
        ext = EdgartoolsSectionExtractor(cross_reference_threshold=100)
        filing = _make_mock_filing(
            filing_year=2023,
            business="x" * 100,  # exactly at threshold
            risk_factors="x" * 99,  # one below
            management_discussion="x" * 101,  # one above
        )
        results = ext.extract(filing)
        assert results["Item 1"].is_cross_reference is False  # not < 100
        assert results["Item 1A"].is_cross_reference is True  # < 100
        assert results["Item 7"].is_cross_reference is False  # > 100

    def test_custom_threshold_applied(self) -> None:
        # A 3000-char section is a cross-ref under a 5000 threshold but
        # not under a 1000 threshold
        filing = _make_mock_filing(
            filing_year=2023,
            business="x" * 3000,
            risk_factors="x" * 10000,
            management_discussion="x" * 10000,
        )
        strict = EdgartoolsSectionExtractor(cross_reference_threshold=5000)
        lenient = EdgartoolsSectionExtractor(cross_reference_threshold=1000)
        assert strict.extract(filing)["Item 1"].is_cross_reference is True
        assert lenient.extract(filing)["Item 1"].is_cross_reference is False


# ============================================================================
# Extraction failure handling
# ============================================================================


class TestExtractionFailure:
    def test_non_string_section_omitted(self) -> None:
        # If edgartools returns None (extraction failed), that section
        # should be omitted from the result dict, not returned with empty text
        ext = EdgartoolsSectionExtractor()
        filing = _make_mock_filing(
            filing_year=2023,
            business="x" * 5000,
            risk_factors=None,  # will be set to None
            management_discussion="x" * 5000,
        )
        results = ext.extract(filing)
        assert "Item 1" in results
        assert "Item 1A" not in results  # omitted due to None
        assert "Item 7" in results

    def test_attribute_error_on_section_omitted(self) -> None:
        # If accessing an attribute raises (e.g., edgartools' internal
        # OptionError), that section is omitted
        ext = EdgartoolsSectionExtractor()

        tenk = MagicMock()
        tenk.business = "x" * 5000
        # Raise on risk_factors access
        type(tenk).risk_factors = property(
            lambda self: (_ for _ in ()).throw(RuntimeError("edgartools failure"))
        )
        tenk.management_discussion = "x" * 5000

        filing = MagicMock()
        filing.filing_date = date(2023, 6, 15)
        filing.obj.return_value = tenk

        results = ext.extract(filing)
        assert "Item 1" in results
        assert "Item 1A" not in results  # raise → omitted
        assert "Item 7" in results