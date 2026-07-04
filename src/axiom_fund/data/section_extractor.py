"""Section extraction from 10-K filings.

Implements the SectionExtractor protocol used by Item 6a (Lazy Prices
signal, per docs/v2_item6_design.md). The protocol lets us swap
extraction backends without touching downstream signal code.

Current backend: EdgartoolsSectionExtractor using the edgartools
library. Design decisions documented in docs/v2_item6_design.md:

- Sample window restricted to 2019+ filings (post-iXBRL mandate).
  Pre-iXBRL 10-Ks systematically cross-reference sections to
  exhibits or page ranges, which corrupts naive text similarity.

- Item 7A (Quantitative and Qualitative Market Risk Disclosures)
  is intentionally NOT extracted. Empirical analysis of the pilot
  sample showed Item 7A is systematically cross-referenced or
  boilerplate (< 500 chars) for large firms including JPM, NEE,
  JNJ, XOM — half our pilot universe. Including it in the signal
  would inject cross-firm structural noise.

- Cross-reference detection uses a minimum-length threshold
  (default 2000 chars). Sections below this are flagged as likely
  cross-references. The 2000-char threshold was chosen after
  empirical analysis of the pilot sample (see length_distribution
  investigation, session 2026-07-05).

- Extraction returns SectionResult objects with is_cross_reference
  metadata rather than filtering. Downstream code decides whether
  to include cross-referenced sections in the signal.

References
----------
- Cohen, Malloy, Nguyen (2020), "Lazy Prices," JF 75(3):1371-1415
- edgartools library: github.com/dgunning/edgartools
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

# Section identifiers extracted per CMN methodology (Item 7A excluded)
EXTRACTED_SECTIONS = ("Item 1", "Item 1A", "Item 7")

# Filings before this year are not supported; iXBRL mandate + cross-reference
# problems make pre-2019 text extraction structurally unreliable.
MINIMUM_SUPPORTED_YEAR = 2019

# Minimum section length in characters. Sections below this are flagged as
# likely cross-references (empirically justified against the pilot sample).
DEFAULT_CROSS_REFERENCE_THRESHOLD = 2000


@dataclass(frozen=True)
class SectionResult:
    """Extraction result for one section of one filing."""
    section_id: str          # e.g., "Item 1"
    text: str                # extracted text (may be short cross-reference)
    length: int              # character count
    is_cross_reference: bool # True if length < threshold


class SectionExtractor(Protocol):
    """Protocol for 10-K section extraction backends.

    Implementations should raise ValueError on unsupported filings
    (e.g., pre-2019) rather than returning empty results.
    """
    def extract(self, filing: object) -> dict[str, SectionResult]:
        """Extract sections from a 10-K filing.

        Parameters
        ----------
        filing : object
            Backend-specific filing handle. For EdgartoolsSectionExtractor,
            this is an edgar.Filing object.

        Returns
        -------
        dict[str, SectionResult]
            Keys are section identifiers ("Item 1", "Item 1A", "Item 7").
            Missing sections indicate extraction failure (not cross-reference).
        """
        ...


class EdgartoolsSectionExtractor:
    """Section extractor using the edgartools library.

    Verified on 60 pilot filings (10 tickers × 6 years, 2019-2024):
    Items 1 and 1A extract cleanly with zero failures. Item 7 extracts
    cleanly for 8/10 pilot tickers; XOM and T systematically cross-
    reference Item 7 with ~270-char pointers to consolidated MD&A
    elsewhere in the filing.

    The extractor faithfully returns whatever edgartools produces,
    with is_cross_reference flagged based on length threshold.
    """

    def __init__(
        self,
        cross_reference_threshold: int = DEFAULT_CROSS_REFERENCE_THRESHOLD,
    ) -> None:
        """Initialize the extractor.

        Parameters
        ----------
        cross_reference_threshold : int
            Sections with fewer characters than this are flagged as
            likely cross-references. Default 2000 (see module docstring).
        """
        if cross_reference_threshold < 0:
            raise ValueError(
                f"cross_reference_threshold must be non-negative, "
                f"got {cross_reference_threshold}"
            )
        self._threshold = cross_reference_threshold

    def extract(self, filing: object) -> dict[str, SectionResult]:
        """Extract Items 1, 1A, 7 from a 10-K filing.

        Parameters
        ----------
        filing : object
            An edgar.Filing object (from edgartools).

        Returns
        -------
        dict[str, SectionResult]
            Keys "Item 1", "Item 1A", "Item 7". Sections that fail to
            extract are omitted from the dict (not returned with empty
            text).

        Raises
        ------
        ValueError
            If the filing's year is before MINIMUM_SUPPORTED_YEAR (2019).
        """
        filing_year = filing.filing_date.year  # type: ignore[attr-defined]
        if filing_year < MINIMUM_SUPPORTED_YEAR:
            raise ValueError(
                f"filing year {filing_year} is before "
                f"MINIMUM_SUPPORTED_YEAR {MINIMUM_SUPPORTED_YEAR}. "
                f"Pre-iXBRL filings systematically cross-reference "
                f"sections and cannot be reliably compared via text "
                f"similarity. Restrict the sample window or use a "
                f"different extractor."
            )

        tenk = filing.obj()  # type: ignore[attr-defined]

        # Named-property mapping from edgartools TenK
        section_attrs = {
            "Item 1": "business",
            "Item 1A": "risk_factors",
            "Item 7": "management_discussion",
        }

        results: dict[str, SectionResult] = {}
        for section_id, attr in section_attrs.items():
            try:
                text = getattr(tenk, attr)
            except Exception:
                # edgartools raises various exceptions on failure paths
                # (OptionError, TypeError, KeyError observed in the wild).
                # Any failure means we omit this section from results.
                continue

            if not isinstance(text, str):
                continue

            length = len(text)
            results[section_id] = SectionResult(
                section_id=section_id,
                text=text,
                length=length,
                is_cross_reference=(length < self._threshold),
            )

        return results