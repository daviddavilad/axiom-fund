# v2 Item 6 — Design: Lazy Prices signal

Status: **scoping (Item 6a in progress)**
Owner: David Davila
Reference: Cohen, Malloy, Nguyen (2020), "Lazy Prices," *Journal of Finance* 75(3):1371-1415

## Motivation

Cohen-Malloy-Nguyen (CMN) 2020 report that firms whose annual 10-K filings change **less** year-over-year subsequently underperform. Their interpretation: managers copy from prior filings when the fundamental picture is deteriorating or uninformative, rather than rewriting to reflect genuine changes. The signal is orthogonal to standard fundamentals (accruals, momentum, quality) — it's about disclosure behavior, not financial performance directly.

Item 6 asks: does the CMN finding replicate on the top-1000 universe from 2015-2024, and can it be modernized with contemporary text-representation techniques?

This is the "text channel" referenced in v2 Phase 2 planning. It complements the existing four signals (GP, IVol, ResMom, PEAD), all of which use structured price/fundamentals data.

## The signal

Per CMN:

- **Data:** annual 10-K filings from SEC EDGAR
- **Comparison:** for each firm-year, compute cosine similarity between the current 10-K and the prior year's 10-K
- **Sections:** Items 1 (Business), 1A (Risk Factors), 7 (MD&A), 7A (Quantitative and Qualitative Market Risk Disclosures)
- **Signal definition:** `signal = 1 - cosine_similarity`. Higher signal = more textual change = we predict *outperformance* per CMN's inverse relationship
- **Portfolio:** long-top-quintile vs short-bottom-quintile on the signal, holding period ~12 months (until next 10-K filing)
- **Timing:** signal observable at 10-K filing date (typically 60-90 days after fiscal year end); portfolio formed at first month-end following filing

## Scope: Item 6a, 6b, 6c

Item 6 is split into three sub-items with explicit dependencies. Each has a defined "success" and "failure" outcome, pre-committed *before* results are known.

### Item 6a — TF-IDF replication (this iteration)

**Scope:** Faithful replication of CMN methodology on v1's top-1000 universe, using TF-IDF cosine similarity as the text representation.

**Deliverables:**
1. EDGAR data acquisition pipeline: `src/axiom_fund/data/edgar.py` (direct HTTP client with rate limiting)
2. 10-K corpus for v1 universe intersect firms with ≥2 consecutive filings, 2015-2024
3. Section extraction: Items 1, 1A, 7, 7A extracted per filing
4. TF-IDF vectorization + year-over-year cosine similarity per firm-year
5. Portfolio-sort backtest: quintile spread returns, holding period stats, Sharpe, hit rate, max drawdown
6. Findings document: does CMN replicate on our universe?

**Success:** Long-top-minus-short-bottom quintile spread produces positive Sharpe with the sign consistent with CMN's finding. Statistical significance tested via the same HAC + bootstrap framework as Item 3.

**Failure:** Signal has zero or negative Sharpe on our universe. Documented as a clean negative replication finding.

### Item 6b — Sentence-transformer extension

**Scope:** Rebuild the signal using contemporary transformer embeddings (candidate: `all-mpnet-base-v2` or FinBERT). Same portfolio-sort framework as 6a. Compare Sharpe/hit-rate to 6a's TF-IDF version.

**Purpose:** Distinguish whether the CMN effect (if it replicates) is driven by *surface* text similarity (which TF-IDF captures) or *semantic* similarity (which transformers capture). The comparison finding is intellectually valuable regardless of direction.

**Prerequisite:** Item 6a substantively complete (both success and failure outcomes are actionable for 6b).

### Item 6c — Composite integration (conditional)

**Scope:** If Item 6a produces a working standalone signal, evaluate incremental contribution to the existing 4-signal composite. Address the cadence-mismatch problem (annual signal in monthly-rebalanced portfolio) with explicit weighting choice.

**Prerequisite:** Item 6a shows positive Sharpe (i.e., a working signal exists to integrate).

## Universe and data

- **Firms:** intersection of (a) v1's top-1000 by market cap and (b) firms with ≥2 consecutive 10-K filings 2015-2024
- **Sample coverage estimate:** ~700-900 firms with ~7-9 year-over-year comparisons each, roughly 5000-7500 firm-year observations
- **Look-ahead prevention:** signal observable only at filing date + a small buffer (5 trading days) to reflect realistic public disclosure lag
- **Rebalance:** monthly, using most recent available signal per firm; signals become stale between filings

## EDGAR acquisition plan

- **Client:** direct HTTP via `httpx`, no third-party EDGAR wrapper (`sec-edgar-downloader` considered and rejected in favor of learning value + full control)
- **Rate limit:** ≤10 req/sec per SEC guidelines; implementation targets 8 req/sec with jitter
- **User-agent:** SEC-compliant identifier
- **Two-step flow:** (1) `https://data.sec.gov/submissions/CIK{cik}.json` for a firm's filing history, (2) filing index page for document URLs
- **Ticker→CIK mapping:** SEC's `company_tickers.json` (public, ~500KB)
- **Storage:** local filesystem, `data/raw/edgar/{ticker}/10-K/{filing_date}/`
- **Pilot:** 10 diverse tickers (AAPL, MSFT, JPM, XOM, JNJ, WMT, T, NEE, KO, GE) to validate parser against structurally different 10-K styles before scaling

## Section extraction plan (Item 6a, later in project)

Not implemented in this session. Design note: 10-K format has evolved over the last decade. Items 1, 1A, 7, 7A must be extracted via regex-based section header matching, with fallback for missing or renamed sections. Strict mode (drop filings where any of the 4 sections missing) vs lenient mode (use available sections) to be decided empirically after pilot corpus is downloaded.

## Pre-commitments on evaluation

### What "success" means (before running the backtest)

For Item 6a to be considered a successful replication:
- Long-top-quintile minus short-bottom-quintile portfolio has Sharpe > 0.3 on the full sample, with holding-period-consistent bootstrap CI excluding zero
- Direction of sign matches CMN (high change → outperformance, low change → underperformance)
- Effect survives HAC-corrected t-statistic on Fama-MacBeth cross-sectional regression IC

### What "failure" means (before running the backtest)

Item 6a fails to replicate if any of:
- Sharpe below 0.1 or negative
- Sign flipped relative to CMN (would need honest re-interpretation)
- Statistical significance lost under any standard inference correction

A clean failure is documented honestly, matching Items 4 and 5 (which both produced "less than the loose narrative suggested" findings). Failure to replicate a 5-year-old paper on out-of-sample data is a genuine research contribution, not a project death sentence.

## Known risks and open questions

1. **CMN period was 1995-2014.** Their finding is out-of-sample tested on 2015-2024. Real risk it doesn't hold: text disclosure conventions have shifted (rise of algorithmic 10-K reviews, boilerplate risk factor disclosures, generative AI drafting tools in latter years).

2. **Cadence mismatch with existing signals.** 10-Ks are filed annually; Lazy Prices signal updates once per year per firm. Composite integration (Item 6c) needs to address this explicitly.

3. **Section extraction fragility.** Filing structure has changed over the sample window. Extraction failures may bias the sample toward better-formatted (larger, more sophisticated) firms.

4. **Look-ahead risk.** If we use fiscal-year-end as the signal date instead of filing date, we introduce look-ahead of ~60-90 days. Filing date required.

5. **Universe survivorship.** v1's top-1000 as of what date? Need consistent construction across the sample. If we use current top-1000 back-projected, we introduce survivor bias.

6. **Small-cap tickers may not have consistent 10-K filings.** ADRs, foreign issuers filing 20-Fs instead of 10-Ks, recently-IPO'd firms with insufficient history — all need explicit handling.

7. **Mid-window filing format transition (2019-2020 iXBRL mandate) — INVESTIGATED.** SEC mandated inline XBRL for large accelerated filers in 2019-2020. Pilot data shows a ~5-10x jump in primary-document file size across this transition. Investigation (2026-07-05) revealed the root cause: pre-iXBRL 10-Ks systematically use cross-references. For example, JPM 2015 Item 7 is 395 characters saying "See pages 64-169," and JNJ 2015 Item 1A is 546 characters saying "See Exhibit 99." The substantive content lives elsewhere in the filing. Any text-similarity signal computed across the format transition would be dominated by structural artifacts. **Resolution: sample window restricted to 2019-2024 (see Sample Window Decision below).** Surfaced by pilot download (commit 241ee5f).

8. **Primary-document ambiguity — SUPERSEDED.** Originally documented as a separate risk (JNJ 2015 297 KB thin document), but investigation (2026-07-05) confirmed this is the same underlying phenomenon as risk 7: pre-iXBRL cross-referencing. The JNJ 2015 primary document is a thin index because the substantive Items 1A / 7 content is in Exhibit 99 or referenced pages. Same resolution as risk 7. Surfaced by pilot download (commit 241ee5f).

## Time budget estimate

Realistic scope of full Item 6 (across all sub-items): 30-50 hours across ~10-15 sessions. CFA Level II (Aug 28) has priority; Item 6 work fits around CFA prep. Item 6a alone: ~15-25 hours.

## Sample window decision (2026-07-05)

**Decision: restrict sample to 2019-2024.**

Rationale: investigation of edgartools section extraction across 5 tickers × 5 years revealed that pre-iXBRL filings (roughly 2015-2018 depending on firm) systematically use cross-references for Items 7 and 1A. The substantive content lives in referenced exhibits or page ranges within the filing, not in the item itself. Any signal computed from cross-referenced sections would be dominated by structural artifacts (filing format change), not by disclosure change (which is the CMN methodology's target).

CMN's original paper covered 1995-2014. Our restricted window (2019-2024) is an out-of-sample extension to a different regulatory era, not a corrupted replication attempt.

Trade-off: sample size reduced from ~5000-7500 firm-year observations (10 years) to ~3500-4500 (6 years). Statistical power reduced but methodology cleanly defensible.

## Section scope decision (2026-07-05)

**Decision: extract Items 1, 1A, 7. Drop Item 7A.**

Rationale: empirical analysis of 60 filings (10 pilot tickers × 6 years, 2019-2024) showed Item 7A is systematically cross-referenced or boilerplate for the majority of large firms:
- JPM: 250 chars every year (2019-2024)
- NEE: 157 chars every year (2019-2024)
- JNJ: ~485 chars every year (2019-2024)
- XOM: 407-1248 chars depending on year
- T 2019: 260 chars

Median Item 7A length across the sample: 2,120 chars. Compared to median Item 1A of 68,601 chars, Item 7A is systematically thinner. Roughly 40% of the sample has Item 7A below 1500 chars, and this fraction is dominated by specific firms (JPM, NEE, JNJ, XOM) whose Item 7A is systematically boilerplate or cross-referenced across all sample years — indicating structural firm-level filing patterns, not random data quality.

Including Item 7A in the signal would inject cross-firm structural noise: comparing JPM's 250-char boilerplate year-over-year gives essentially zero change, which is not a research finding.

Trade-off: CMN's original methodology used all four sections. Dropping Item 7A is a deviation. The empirical justification is strong (see distribution table above) and is transparent in the writeup.

## Cross-reference detection (2026-07-05)

**Decision: flag sections shorter than 2000 characters as likely cross-references.**

Rationale: empirical distribution from the pilot showed a clear gap:
- Clear cross-references: 157-546 chars (JPM, NEE, JNJ boilerplate)
- Gray zone: 1180-1248 chars (XOM Item 7A in some years — possibly thin but real)
- Real content: 5000+ chars in nearly every case

The 2000-char threshold captures all clear cross-references while preserving false-negative risk (dropping some real thin content) over false-positive risk (including boilerplate in the signal). This is a conservative choice; sensitivity analysis with alternate thresholds could be reported in the final signal writeup.

Implementation: `EdgartoolsSectionExtractor` returns `SectionResult` objects with `is_cross_reference: bool` metadata based on this threshold. Downstream code (signal computation, not this extractor) decides whether to include cross-referenced sections in the analysis.

## Extraction backend decision (2026-07-05)

**Decision: adopt `edgartools` for section extraction.**

Rationale: after investigating the ecosystem, `edgartools` (github.com/dgunning/edgartools) is the most complete Python library for SEC filing parsing. It handles iXBRL natively, provides named-property access to 10-K items, and is used in production at multiple hedge funds. MIT-licensed, actively maintained.

Verified on 60 pilot filings (10 tickers × 6 years, 2019-2024):
- Item 1 extracts cleanly: 60/60, minimum length 7,405 chars
- Item 1A extracts cleanly: 60/60, minimum length 8,216 chars
- Item 7 extracts cleanly for 8/10 firms: 59/60, with 6 cross-references (XOM 2020-2024, T 2019) correctly flagged by our threshold
- 1 outright failure (WMT 2015 pre-window, correctly rejected by the window guard)

Trade-off: adopting `edgartools` added 33 transitive dependencies to the repo (see `uv add edgartools` output). This is a deliberate deviation from the "minimize deps" ethos held earlier in Item 6. Reasoning: `edgartools` provides section extraction that would take substantial engineering effort to reproduce at comparable quality using stdlib + BS4. One tutorial from a commercial competitor (sec-api.io) claims regex-based approaches achieve only ~30% coverage, though this figure comes from a self-interested source and is not authoritative. Empirical verification of `edgartools` on our pilot sample showed clean extraction of Items 1 and 1A across 100% of 2019-2024 filings and Item 7 across 80% (with the remaining 20% correctly flagged as cross-references). Reproducing that quality independently is not a good use of session time relative to actual signal methodology work. The dependency is bounded to the section-extraction path only; all other Item 6 code (EDGAR client, TF-IDF pipeline, backtest) remains dependency-lean.

**Bus factor risk noted:** `edgartools` is maintained primarily by one person. If it stops, we would need to migrate to a fallback. The `SectionExtractor` protocol in `src/axiom_fund/data/section_extractor.py` provides the abstraction: swapping backends means implementing one new class, not rewriting the pipeline.

## Test coverage note

Unit tests in `tests/test_section_extractor.py` mock `edgartools` at the boundary (Filing → TenK). They protect against regressions in our own extractor's logic (window guard, cross-reference detection, section-name mapping, error handling) but do not catch upstream `edgartools` regressions. Real-integration verification (network + real SEC data) is via the ad-hoc `/tmp/verify_extractor.py` script pattern, not automated tests. For a production system this gap would need closing (via VCR-style recorded fixtures or pinned `edgartools` version + integration test suite); for this research repo the gap is documented.

## Corpus expansion (2026-07-05)

**Universe: 1,448 unique firms** as the union of top-1000 U.S. common stocks by market cap at 6 year-end snapshots (2018-12-31 through 2023-12-29). Snapshot approach was rejected in favor of union to avoid single-date selection bias; distribution shows 672 firms present in all 6 snapshots (stable core), 776 firms present in ≤4 snapshots (churn). Constructed via `scripts/data_acquisition/build_universe.py` using the existing `Universe` module.

**CIK resolution: 1,117 firms resolved** (77.1% of universe). 331 firms failed resolution — nearly all are firms that were acquired or delisted during the sample window (SIVB, CELG, CERN, TECD, and similar). SEC's `company_tickers.json` only contains currently-active tickers. This creates a real survivorship bias (see Known Limitations below).

**Corpus downloaded: 6,286 10-K filings, 27.7 GB** across the 1,117 resolved firms. Average 5.6 filings per firm across the 6-year sample window. Zero download failures. Scripts: `resolve_ciks.py`, `download_10k_corpus.py`. Storage under `data/raw/edgar/{ticker}/10-K/{filing_date}/`.

## Next steps

**Next session:** run `EdgartoolsSectionExtractor` on the full 6,286-filing corpus. Store extracted sections per (ticker, filing_date, section) to parquet. Report failure rate + cross-reference rate. Estimated 60-90 min extraction time based on per-filing parsing costs.

**Following sessions:** build TF-IDF vectorization pipeline. Compute year-over-year cosine similarity per firm. Portfolio-sort backtest (long-top-quintile minus short-bottom-quintile). Findings document.

**Later:** Item 6b (transformer extension). Item 6c (composite integration, conditional on 6a success).

## Known limitations

Consolidated list of documented issues that a reviewer would appropriately flag, retained inline so future readers see the methodological trade-offs alongside the methodology.

1. **Survivorship bias in CIK resolution.** 22.9% of the top-1000 union universe (331 firms) failed CIK resolution via SEC's `company_tickers.json`, which only contains currently-active tickers. Missing firms are almost all acquisitions or delistings during 2019-2024 (SIVB, CELG, CERN, TECD, and similar). This biases the sample toward survivors: the effect is directional against distressed and acquired firms, precisely the tail where CMN-style disclosure divergence might be strongest. A fix via CRSP-Compustat linking (PERMNO → gvkey → CIK through `comp.security`) was scoped at 30-50 minutes and deferred to a future robustness session; not blocking corpus expansion.

2. **`Universe.as_of()` silently fails on non-trading days.** The universe module returns 0 rows without warning when passed a non-trading date (e.g., `2022-12-31` fell on a Saturday). Discovered when the initial `build_universe.py` returned 0-row snapshots for 2022 and 2023 year-ends. Worked around by using actual last trading days (2022-12-30, 2023-12-29). The underlying bug remains in `Universe.as_of()`; a proper fix (either raise on non-trading dates, or auto-align to nearest prior trading day) is deferred but noted for future infrastructure work.

3. **`market_cap` in universe parquet is stale.** Because we take the union across 6 snapshots and dedup to keep the first appearance, the `market_cap` and other size columns reflect the value at the firm's first snapshot only. A firm first appearing in 2018 has its 2018 market cap saved, not its 2023 value. Downstream code that uses size for weighting must not use this column directly; either regenerate with the specific date needed, or accept the caveat.

4. **Test coverage gap for edgartools regressions.** Unit tests for `EdgartoolsSectionExtractor` mock the edgartools boundary and protect against regressions in our own logic. They do not catch upstream `edgartools` regressions. For a production system this would need closing (via VCR-style recorded fixtures or pinned version + integration test suite); for this research repo the gap is documented.

5. **Snapshot bias in market_cap universe construction.** Even with the union approach, "top-1000 by market cap at each year-end" is one specific methodology choice. Alternative constructions (rolling top-1000 rebalanced annually, top-1000 by average market cap across the window, top-1000 by trailing 12-month median) would produce different samples. Our choice is defensible but not the only defensible one.

6. **Item 7A dropped from signal scope (empirical, not accidental).** Investigation showed Item 7A is systematically thin (boilerplate or cross-referenced) for large firms including JPM, NEE, JNJ, XOM — 40% of the pilot sample. Signal scope is Items 1, 1A, 7 only. This is a documented deviation from CMN's original methodology, empirically justified against the pilot distribution.

7. **Pre-iXBRL filings excluded (2015-2018).** Investigation revealed pre-iXBRL 10-Ks systematically use cross-references. Sample window restricted to 2019-2024 (post-iXBRL mandate). Sample size reduced from a hypothetical ~5000-7500 firm-year observations to ~3500-4500. Documented deviation from CMN's original 1995-2014 window; our study is an out-of-sample extension, not a corrupted replication attempt.