"""Unit tests for the SEC EDGAR client.

Covers src/axiom_fund/data/edgar.py:
  - EdgarClient initialization and env var handling
  - CIK map indexing and lookup
  - 10-K filtering from submissions blocks
  - Recent + files pagination with dedup
  - Rate limiter window enforcement
  - Download URL construction

Mocks are applied at the private-method boundary (_get_json,
_get_bytes) rather than at urllib. The transport layer worked in
yesterday's real-SEC pilot; these tests target the logic above it.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from axiom_fund.data.edgar import EdgarClient, Filing


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> EdgarClient:
    """EdgarClient with a valid email env var and a tmp output dir."""
    monkeypatch.setenv("SEC_USER_AGENT_EMAIL", "test@example.com")
    return EdgarClient(output_dir=tmp_path)


def _submissions_block(rows: list[dict[str, str]]) -> dict[str, list[str]]:
    """Build a submissions block (recent or files sub-JSON) from row dicts.

    Each row has keys: form, accessionNumber, filingDate, reportDate,
    primaryDocument. Returns the columnar layout SEC uses.
    """
    block: dict[str, list[str]] = {
        "form": [],
        "accessionNumber": [],
        "filingDate": [],
        "reportDate": [],
        "primaryDocument": [],
    }
    for row in rows:
        for key in block:
            block[key].append(row[key])
    return block


# ============================================================================
# Client initialization
# ============================================================================


class TestClientInitialization:
    def test_requires_email_env_var(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("SEC_USER_AGENT_EMAIL", raising=False)
        with pytest.raises(RuntimeError, match="SEC_USER_AGENT_EMAIL"):
            EdgarClient(output_dir=tmp_path)


# ============================================================================
# CIK map
# ============================================================================


class TestCikMap:
    def test_load_pads_and_uppercases(self, client: EdgarClient) -> None:
        # SEC returns: {"0": {"cik_str": 320193, "ticker": "aapl", "title": "..."}}
        fake_json = {
            "0": {"cik_str": 320193, "ticker": "aapl", "title": "Apple Inc."},
            "1": {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft"},
        }
        with patch.object(EdgarClient, "_get_json", return_value=fake_json):
            cik_aapl = client.get_cik("AAPL")
            cik_msft = client.get_cik("msft")  # lowercase input
        assert cik_aapl == "0000320193"  # zero-padded to 10
        assert cik_msft == "0000789019"

    def test_unknown_ticker_raises(self, client: EdgarClient) -> None:
        fake_json = {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple"}}
        with patch.object(EdgarClient, "_get_json", return_value=fake_json):
            with pytest.raises(KeyError, match="NOTREAL"):
                client.get_cik("NOTREAL")


# ============================================================================
# _extract_10ks
# ============================================================================


class TestExtractTenKs:
    def test_filters_to_exact_form_match(self, client: EdgarClient) -> None:
        # Should include 10-K but drop 10-K/A, 10-KT, 10-K405
        block = _submissions_block([
            {"form": "10-K", "accessionNumber": "0000320193-24-001",
             "filingDate": "2024-11-01", "reportDate": "2024-09-28",
             "primaryDocument": "aapl-20240928.htm"},
            {"form": "10-K/A", "accessionNumber": "0000320193-24-002",
             "filingDate": "2024-12-01", "reportDate": "2024-09-28",
             "primaryDocument": "aapl-amend.htm"},
            {"form": "10-KT", "accessionNumber": "0000320193-24-003",
             "filingDate": "2024-12-15", "reportDate": "2024-09-28",
             "primaryDocument": "aapl-transition.htm"},
            {"form": "10-K405", "accessionNumber": "0000320193-98-001",
             "filingDate": "1998-12-01", "reportDate": "1998-09-30",
             "primaryDocument": "aapl-old.htm"},
        ])
        results = EdgarClient._extract_10ks(block)
        assert len(results) == 1
        assert results[0].form == "10-K"
        assert results[0].accession_number == "0000320193-24-001"

    def test_handles_empty_block(self, client: EdgarClient) -> None:
        block: dict[str, list[Any]] = {
            "form": [],
            "accessionNumber": [],
            "filingDate": [],
            "reportDate": [],
            "primaryDocument": [],
        }
        results = EdgarClient._extract_10ks(block)
        assert results == []

    def test_preserves_all_fields(self, client: EdgarClient) -> None:
        block = _submissions_block([
            {"form": "10-K", "accessionNumber": "0000019617-24-000225",
             "filingDate": "2024-02-16", "reportDate": "2023-12-31",
             "primaryDocument": "jpm-20231231.htm"},
        ])
        results = EdgarClient._extract_10ks(block)
        assert len(results) == 1
        f = results[0]
        assert f.accession_number == "0000019617-24-000225"
        assert f.filing_date == "2024-02-16"
        assert f.report_date == "2023-12-31"
        assert f.primary_document == "jpm-20231231.htm"
        assert f.form == "10-K"


# ============================================================================
# get_10k_history: pagination + dedup
# ============================================================================


class TestGet10KHistory:
    def test_deduplicates_across_recent_and_files(
        self, client: EdgarClient,
    ) -> None:
        # Recent block has one 10-K; files block has two, one duplicating
        # the recent one. Result should have 2 unique filings.
        recent = _submissions_block([
            {"form": "10-K", "accessionNumber": "0000019617-24-000225",
             "filingDate": "2024-02-16", "reportDate": "2023-12-31",
             "primaryDocument": "jpm-20231231.htm"},
        ])
        older = _submissions_block([
            {"form": "10-K", "accessionNumber": "0000019617-24-000225",
             "filingDate": "2024-02-16", "reportDate": "2023-12-31",
             "primaryDocument": "jpm-20231231.htm"},  # duplicate
            {"form": "10-K", "accessionNumber": "0000019617-23-000231",
             "filingDate": "2023-02-21", "reportDate": "2022-12-31",
             "primaryDocument": "jpm-20221231.htm"},
        ])

        # First call = submissions API root (with recent block + files list);
        # subsequent = older paginated file
        submissions_json = {
            "filings": {
                "recent": recent,
                "files": [{
                    "name": "CIK0000019617-submissions-001.json",
                    "filingCount": 2, "filingFrom": "2023-01-01",
                    "filingTo": "2024-06-30",
                }],
            }
        }
        call_returns = [submissions_json, older]
        with patch.object(
            EdgarClient, "_get_json", side_effect=call_returns,
        ):
            results = client.get_10k_history("0000019617")

        assert len(results) == 2
        accessions = {f.accession_number for f in results}
        assert accessions == {
            "0000019617-24-000225", "0000019617-23-000231",
        }
        # Sorted descending by filing date
        assert results[0].filing_date == "2024-02-16"
        assert results[1].filing_date == "2023-02-21"

    def test_earliest_filing_date_prunes_files(
        self, client: EdgarClient,
    ) -> None:
        # Two files blocks: one wholly older than cutoff, one overlapping.
        # Only the overlapping one should be fetched.
        recent = _submissions_block([])  # no 10-Ks in recent

        submissions_json = {
            "filings": {
                "recent": recent,
                "files": [
                    {
                        "name": "old-block.json",
                        "filingFrom": "2000-01-01",
                        "filingTo": "2010-12-31",  # entirely before cutoff
                    },
                    {
                        "name": "overlap-block.json",
                        "filingFrom": "2013-01-01",
                        "filingTo": "2016-12-31",  # partial overlap
                    },
                ],
            }
        }
        overlap_block = _submissions_block([
            {"form": "10-K", "accessionNumber": "test-2015",
             "filingDate": "2015-06-01", "reportDate": "2014-12-31",
             "primaryDocument": "test.htm"},
        ])
        # Should be called exactly twice: submissions root + overlap file
        call_returns = [submissions_json, overlap_block]
        with patch.object(
            EdgarClient, "_get_json", side_effect=call_returns,
        ) as mock_get:
            results = client.get_10k_history(
                "0000000001", earliest_filing_date="2015-01-01",
            )

        assert mock_get.call_count == 2  # NOT 3 — old block was pruned
        assert len(results) == 1
        assert results[0].accession_number == "test-2015"


# ============================================================================
# Rate limiter
# ============================================================================


class TestRateLimiter:
    def test_enforces_window(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Mock monotonic and sleep. At capacity, sleep must be called."""
        monkeypatch.setenv("SEC_USER_AGENT_EMAIL", "test@example.com")
        client = EdgarClient(output_dir=tmp_path)

        # Simulate 8 recent requests (at the target rate cap) all within
        # the last 0.1 seconds
        current_time = [100.0]

        def mock_monotonic() -> float:
            return current_time[0]

        sleep_calls: list[float] = []

        def mock_sleep(seconds: float) -> None:
            sleep_calls.append(seconds)
            current_time[0] += seconds  # advance the clock

        # Pre-populate the deque with 8 recent timestamps at t=99.9
        for _ in range(8):
            client._request_times.append(99.9)

        with patch("axiom_fund.data.edgar.time.monotonic", mock_monotonic), \
             patch("axiom_fund.data.edgar.time.sleep", mock_sleep):
            # This call should sleep, because deque is at capacity
            # (8 requests in the last 0.1 sec = well above 8/sec rate)
            client._enforce_rate_limit()

        # Sleep must have been called at least once
        assert len(sleep_calls) >= 1
        assert sleep_calls[0] > 0  # positive sleep duration

    def test_no_sleep_when_below_capacity(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("SEC_USER_AGENT_EMAIL", "test@example.com")
        client = EdgarClient(output_dir=tmp_path)

        # Empty deque = no prior requests = should not sleep
        sleep_calls: list[float] = []

        def mock_sleep(seconds: float) -> None:
            sleep_calls.append(seconds)

        with patch("axiom_fund.data.edgar.time.sleep", mock_sleep):
            client._enforce_rate_limit()

        assert sleep_calls == []  # no sleep needed


# ============================================================================
# URL construction
# ============================================================================


class TestUrlConstruction:
    def test_download_url_strips_accession_dashes(
        self, client: EdgarClient, tmp_path: Path,
    ) -> None:
        """Verify download URL builds correctly.

        SEC's archives URL uses:
          - CIK without leading zeros (integer form)
          - Accession number with dashes stripped
        """
        filing = Filing(
            accession_number="0000320193-24-000123",  # dashed
            filing_date="2024-11-01",
            report_date="2024-09-28",
            primary_document="aapl-20240928.htm",
            form="10-K",
        )

        captured_urls: list[str] = []

        def fake_get_bytes(url: str) -> bytes:
            captured_urls.append(url)
            return b"fake content"

        with patch.object(
            EdgarClient, "_get_bytes", side_effect=fake_get_bytes,
        ):
            client.download_filing(
                cik="0000320193", filing=filing, ticker="AAPL",
            )

        assert len(captured_urls) == 1
        url = captured_urls[0]
        # CIK integer form (no leading zeros)
        assert "/data/320193/" in url
        # Accession dashes stripped
        assert "/000032019324000123/" in url
        # Primary document appended
        assert url.endswith("/aapl-20240928.htm")