"""SEC EDGAR data acquisition client.

Direct HTTP client (stdlib urllib) for downloading 10-K filings from
SEC EDGAR. Handles ticker -> CIK mapping, filing history lookup,
and filing document download with SEC-compliant rate limiting and
retry logic.

Used by Item 6 (Lazy Prices signal) per docs/v2_item6_design.md.

SEC requirements enforced here:
  - User-agent header identifying the requester (email required)
  - Rate limit <= 10 requests/second (we target ~8 with jitter)
  - Retry with backoff on transient failures (429, 5xx)

User-agent email is loaded from environment variable
SEC_USER_AGENT_EMAIL via .env file. Never hard-coded, never in git.

Example
-------
    client = EdgarClient(output_dir=Path("data/raw/edgar"))
    cik = client.get_cik("AAPL")
    filings = client.get_10k_history(cik)
    for filing in filings:
        client.download_filing(cik, filing, ticker="AAPL")

References
----------
- SEC EDGAR API documentation: https://www.sec.gov/edgar/sec-api-documentation
- SEC fair access policy: https://www.sec.gov/os/webmaster-faq#code-support
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


# Load .env from repo root explicitly (find_dotenv fails in some contexts)
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
load_dotenv(_REPO_ROOT / ".env")


# SEC endpoints
_SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_SEC_SUBMISSIONS_URL_TEMPLATE = "https://data.sec.gov/submissions/CIK{cik}.json"
_SEC_ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data"

# Rate limiting: target 8 req/sec (well under SEC's 10/sec cap)
_TARGET_REQUESTS_PER_SECOND = 8.0
_RATE_WINDOW_SECONDS = 1.0

# Retry policy
_MAX_RETRIES = 3
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


@dataclass(frozen=True)
class Filing:
    """A single 10-K filing record from EDGAR."""
    accession_number: str    # e.g., "0000320193-24-000123"
    filing_date: str         # YYYY-MM-DD
    report_date: str         # YYYY-MM-DD (fiscal year end)
    primary_document: str    # e.g., "aapl-20240928.htm"
    form: str                # "10-K"


class EdgarClient:
    """SEC EDGAR client with rate limiting and retry logic.

    Loads user-agent email from SEC_USER_AGENT_EMAIL environment
    variable (must be set in .env). Rate limits at ~8 req/sec.
    Retries transient failures with exponential backoff.

    Not thread-safe; instantiate one client per thread if parallelizing.
    """

    def __init__(
        self,
        output_dir: Path,
        user_agent_name: str = "Axiom Fund Research",
    ) -> None:
        """Initialize the client.

        Parameters
        ----------
        output_dir : Path
            Root directory for downloaded filings. Structure:
            {output_dir}/{ticker}/10-K/{filing_date}/{primary_doc}
        user_agent_name : str
            Name portion of the User-Agent header. Combined with
            email from environment to form the full header.

        Raises
        ------
        RuntimeError
            If SEC_USER_AGENT_EMAIL environment variable is not set.
        """
        email = os.environ.get("SEC_USER_AGENT_EMAIL")
        if not email:
            raise RuntimeError(
                "SEC_USER_AGENT_EMAIL environment variable not set. "
                "Add it to your .env file (see .env.example)."
            )
        self._user_agent = f"{user_agent_name} {email}"
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

        # Rate limiter: track recent request timestamps
        self._request_times: deque[float] = deque()

        # CIK cache: lazily populated on first get_cik call
        self._cik_map: dict[str, str] | None = None

    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------

    def get_cik(self, ticker: str) -> str:
        """Resolve ticker to zero-padded 10-digit CIK.

        Downloads the SEC ticker-CIK map on first call, caches it
        in-memory for subsequent calls.

        Parameters
        ----------
        ticker : str
            Ticker symbol, case-insensitive.

        Returns
        -------
        str
            Zero-padded 10-digit CIK (e.g., "0000320193" for AAPL).

        Raises
        ------
        KeyError
            If ticker is not found in the SEC map.
        """
        if self._cik_map is None:
            self._cik_map = self._load_cik_map()
        ticker_upper = ticker.upper()
        if ticker_upper not in self._cik_map:
            raise KeyError(
                f"Ticker {ticker!r} not found in SEC company_tickers.json. "
                f"Note: only U.S.-listed common stocks are indexed; ADRs "
                f"and foreign issuers may need CIK lookup by other means."
            )
        return self._cik_map[ticker_upper]

    def get_10k_history(
        self,
        cik: str,
        earliest_filing_date: str | None = None,
    ) -> list[Filing]:
        """Return all 10-K filings for a CIK, most recent first.

        Uses the SEC's submissions API. Returns filings labeled
        exactly "10-K" (excludes 10-K/A amendments, 10-KT, 20-F).

        SEC's API structures submissions in two parts: a "recent"
        block (last ~1000 filings inline) and paginated "files"
        (older filings in separate JSON files). For high-filing-
        volume issuers (banks, structured-product issuers), the
        recent block may cover only weeks of history, so older
        10-Ks live in the paginated files. This method fetches both.

        Parameters
        ----------
        cik : str
            Zero-padded 10-digit CIK.
        earliest_filing_date : str | None
            Optional YYYY-MM-DD; skip paginated files whose entire
            date range is older. Reduces HTTP calls for high-volume
            issuers. If None, fetches all history.

        Returns
        -------
        list[Filing]
            10-K filings ordered by filing date descending, dedup'd
            by accession number across both data sources.
        """
        # Zero-pad CIK to 10 digits — SEC's submissions API rejects unpadded
        # CIKs with 404. Accepts both int-like strings ("785786") and
        # already-padded strings ("0000785786").
        cik_padded = str(cik).lstrip("0").zfill(10)
        url = _SEC_SUBMISSIONS_URL_TEMPLATE.format(cik=cik_padded)
        raw = self._get_json(url)

        # Parse the recent block
        recent = raw.get("filings", {}).get("recent", {})
        filings = self._extract_10ks(recent)

        # Fetch and parse paginated older files
        files_block = raw.get("filings", {}).get("files", [])
        for file_info in files_block:
            # Optional date-range pruning
            if earliest_filing_date is not None:
                filing_to = file_info.get("filingTo", "")
                if filing_to and filing_to < earliest_filing_date:
                    continue
            older = self._get_older_filings(file_info["name"])
            filings.extend(self._extract_10ks(older))

        # Dedup by accession (recent + files can overlap in edge cases)
        seen: set[str] = set()
        unique: list[Filing] = []
        for f in filings:
            if f.accession_number not in seen:
                seen.add(f.accession_number)
                unique.append(f)

        # Sort by filing date descending
        unique.sort(key=lambda f: f.filing_date, reverse=True)
        return unique

    def _get_older_filings(self, file_name: str) -> dict[str, list[Any]]:
        """Fetch a paginated submissions file and return its parsed block."""
        url = f"https://data.sec.gov/submissions/{file_name}"
        return self._get_json(url)  # type: ignore[no-any-return]

    @staticmethod
    def _extract_10ks(block: dict[str, list[Any]]) -> list[Filing]:
        """Extract 10-K rows from a submissions block (recent or older)."""
        forms = block.get("form", [])
        accession_numbers = block.get("accessionNumber", [])
        filing_dates = block.get("filingDate", [])
        report_dates = block.get("reportDate", [])
        primary_docs = block.get("primaryDocument", [])

        filings: list[Filing] = []
        for i, form in enumerate(forms):
            if form == "10-K":
                filings.append(Filing(
                    accession_number=accession_numbers[i],
                    filing_date=filing_dates[i],
                    report_date=report_dates[i],
                    primary_document=primary_docs[i],
                    form=form,
                ))
        return filings

    def download_filing(
        self,
        cik: str,
        filing: Filing,
        ticker: str,
    ) -> Path:
        """Download a filing's primary document to disk.

        Storage layout: {output_dir}/{ticker}/10-K/{filing_date}/{primary_document}

        Parameters
        ----------
        cik : str
            Zero-padded 10-digit CIK.
        filing : Filing
            Filing record from get_10k_history.
        ticker : str
            Ticker (for directory naming).

        Returns
        -------
        Path
            Local path to the downloaded filing.
        """
        # SEC accession numbers in URLs strip dashes: 0000320193-24-000123 -> 000032019324000123
        accession_stripped = filing.accession_number.replace("-", "")
        cik_int = str(int(cik))  # remove leading zeros for URL
        url = (
            f"{_SEC_ARCHIVES_BASE}/{cik_int}/{accession_stripped}/"
            f"{filing.primary_document}"
        )

        dest_dir = self._output_dir / ticker.upper() / "10-K" / filing.filing_date
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / filing.primary_document

        if dest.exists():
            # Already downloaded; skip
            return dest

        content = self._get_bytes(url)
        dest.write_bytes(content)
        return dest

    # ------------------------------------------------------------
    # Private HTTP methods
    # ------------------------------------------------------------

    def _load_cik_map(self) -> dict[str, str]:
        """Fetch and index SEC's ticker-CIK JSON."""
        raw = self._get_json(_SEC_TICKERS_URL)
        # Format: {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "..."}}
        cik_map: dict[str, str] = {}
        for entry in raw.values():
            ticker = entry["ticker"].upper()
            cik = f"{entry['cik_str']:010d}"
            cik_map[ticker] = cik
        return cik_map

    def _get_json(self, url: str) -> Any:
        """GET a URL and parse JSON response."""
        content = self._get_bytes(url)
        return json.loads(content.decode("utf-8"))

    def _get_bytes(self, url: str) -> bytes:
        """GET a URL with rate limiting, retries, and proper headers."""
        for attempt in range(_MAX_RETRIES + 1):
            self._enforce_rate_limit()
            try:
                request = urllib.request.Request(
                    url,
                    headers={
                        "User-Agent": self._user_agent,
                        "Accept-Encoding": "gzip, deflate",
                        "Host": self._extract_host(url),
                    },
                )
                with urllib.request.urlopen(request, timeout=30) as response:
                    return self._read_response_body(response)
            except urllib.error.HTTPError as e:
                if e.code in _RETRYABLE_STATUS_CODES and attempt < _MAX_RETRIES:
                    delay = 2.0 ** attempt  # 1s, 2s, 4s
                    time.sleep(delay)
                    continue
                raise
            except urllib.error.URLError:
                if attempt < _MAX_RETRIES:
                    delay = 2.0 ** attempt
                    time.sleep(delay)
                    continue
                raise
        raise RuntimeError(f"Failed to fetch {url} after {_MAX_RETRIES} retries")

    def _enforce_rate_limit(self) -> None:
        """Block until we can issue another request within the target rate."""
        now = time.monotonic()
        # Purge timestamps older than the window
        while self._request_times and now - self._request_times[0] > _RATE_WINDOW_SECONDS:
            self._request_times.popleft()
        # If at capacity, sleep until oldest timestamp exits window
        if len(self._request_times) >= _TARGET_REQUESTS_PER_SECOND:
            sleep_time = _RATE_WINDOW_SECONDS - (now - self._request_times[0])
            if sleep_time > 0:
                time.sleep(sleep_time)
        self._request_times.append(time.monotonic())

    @staticmethod
    def _extract_host(url: str) -> str:
        """Extract host from URL for Host header."""
        # SEC requires the Host header explicitly on some endpoints
        prefix_stripped = url.split("://", 1)[1]
        return prefix_stripped.split("/", 1)[0]

    @staticmethod
    def _read_response_body(response: Any) -> bytes:
        """Read response body, handling gzip encoding if present."""
        content = response.read()
        if response.headers.get("Content-Encoding") == "gzip":
            import gzip
            content = gzip.decompress(content)
        return content