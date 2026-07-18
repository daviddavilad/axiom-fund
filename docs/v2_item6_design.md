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

- **Firms:** union of top-1000 by market cap across 6 year-end snapshots (2018-12-31 through 2023-12-29). 1,448 unique firms in the universe.
- **CIK resolution:** 96.1% (1,392 of 1,448) via merged SEC + CRSP-Compustat methodology (see Corpus construction below). 56 firms unresolvable via either source.
- **Extraction:** CIK-based via edgartools' `get_entity()` API; strict `form=='10-K'` filter excludes 10-K/A amendments.
- **Actual coverage:** 1,369 firms with extracted sections, 7,359 filings, 21,932 section rows across 2019-2024. Average 5.4 filings per firm; ~3.0 section rows per filing (Items 1, 1A, 7).
- **Look-ahead prevention:** signal observable only at filing date + a small buffer (5 trading days) to reflect realistic public disclosure lag.
- **Rebalance:** monthly, using most recent available signal per firm; signals become stale between filings.

## EDGAR acquisition plan

- **Client:** direct HTTP via `httpx`, no third-party EDGAR wrapper (`sec-edgar-downloader` considered and rejected in favor of learning value + full control)
- **Rate limit:** ≤10 req/sec per SEC guidelines; implementation targets 8 req/sec with jitter
- **User-agent:** SEC-compliant identifier
- **Two-step flow:** (1) `https://data.sec.gov/submissions/CIK{cik}.json` for a firm's filing history, (2) filing index page for document URLs
- **Ticker→CIK mapping:** SEC's `company_tickers.json` (public, ~500KB)
- **Storage:** local filesystem, `data/raw/edgar/{ticker}/10-K/{filing_date}/`
- **Pilot:** 10 diverse tickers (AAPL, MSFT, JPM, XOM, JNJ, WMT, T, NEE, KO, GE) to validate parser against structurally different 10-K styles before scaling

## Section extraction (implemented)

Implemented via `scripts/data_acquisition/extract_sections.py` using edgartools' TenK object accessors (`business`, `risk_factors`, `management_discussion`). Items 1 (Business), 1A (Risk Factors), 7 (MD&A) extracted per filing. Item 7A dropped based on empirical evidence that large firms systematically use cross-references or boilerplate for that section (see Section scope decision below).

**Path C (CIK-based) rationale:** initial ticker-based extraction (via `Company(ticker)`) failed for firms whose ticker moved between SEC entities mid-window. Example: Sprint's ticker `S` moved to SentinelOne in 2021, so `Company('S').get_filings()` returns only SentinelOne's filings and misses historical Sprint. Rebuilt as CIK-based (via `edgar.get_entity(cik)`) to resolve each firm to its historical entity via the CIK we resolved in the merged CIK file.

**Amendment filter:** both `Company.get_filings(form='10-K')` and `Entity.get_filings(form='10-K')` return 10-K/A amendments alongside primary 10-Ks. Amendments are near-duplicates of the original filing (contaminate year-over-year signal). Added strict `f.form == '10-K'` filter to exclude them. Fixed ~65 firm-years of amendment contamination present in earlier extraction runs.

**Cross-reference detection:** sections under 2,000 characters flagged via `is_cross_reference` column. Threshold set empirically from pilot analysis; pre-iXBRL filings systematically use references like "See pages 64-169" or "See Exhibit 99" for the substantive content.

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

## Signal prototype (2026-07-06)

Pilot-scale prototype on the 10-ticker × 6-year corpus (60 filings) to decide whether to invest full-corpus extraction time before validating the methodology at any scale. Script: `scripts/analysis/prototype_lazy_prices_signal.py`.

Pre-committed directional criteria (locked before running):
1. Usable dispersion: IQR of year-over-year cosine similarity materially above zero (≥ 0.02)
2. Directionally plausible: median in [0.5, 0.99]
3. Section independence: cross-section correlations not all 0.99+

Results:

| Section | N pairs | Min | Median | IQR |
|---|---|---|---|---|
| Item 1 | 50 | 0.061 | 0.967 | 0.040 |
| Item 1A | 50 | 0.117 | 0.980 | 0.048 |
| Item 7 | 43 | 0.712 | 0.964 | 0.039 |

Cross-section correlations (at the ticker-year_pair level): Item 1 vs Item 1A = 0.71; Item 1 vs Item 7 = 0.52; Item 1A vs Item 7 = 0.57.

**All three criteria passed.** Decision: proceed to full-corpus section extraction next session.

Caveats named honestly:
- The IQR threshold of 0.02 was generous; a stricter pre-commit would have required IQR ≥ 0.10, which we would have failed. The dispersion we found is real but modest.
- Signal strength depends on the left tail (firms with the largest year-over-year textual change), not on the median. Tail firms exist in the pilot (GE 2023-2024, T 2021-2022) and correspond to real corporate events (GE's Aerospace/Vernova/HealthCare split, AT&T's WarnerMedia spin-off), not extraction artifacts. Whether tail firms predict returns is the actual research question, not answered by the prototype.
- XOM's Item 7 is systematically cross-referenced and correctly excluded by the extractor; expect similar section-specific dropout across the full universe.

## Corpus construction — final (2026-07-11)

Universe: 1,448 firms (union of top-1000 across 6 year-end snapshots, 2018-2023).

**CIK resolution** uses a merged SEC + CRSP-Compustat approach:
- SEC's `company_tickers.json` gives current ticker→CIK mapping (77.1% coverage).
- CRSP-Compustat path (PERMNO → CUSIP → gvkey → CIK via `comp.company`) recovers acquired/delisted firms preserved in Compustat with historical CIKs (+275 firms).
- 17 disagreements between the two sources resolved via empirical validation: query SEC EDGAR for 10-K counts under each candidate CIK, pick whichever has filings in window (tiebreak: prefer CRSP historical entity).
- Merged resolution: 1,392 firms (96.1%). 56 firms unresolvable by either source.

**Extraction** uses `edgar.get_entity(cik).get_filings(form='10-K')` with strict `f.form == '10-K'` filter to exclude 10-K/A amendments. CIK-based routing (not ticker-based) required because some tickers moved between entities during the window (Sprint → SentinelOne after the T-Mobile merger).

**Dependencies:** pandas ≥ 2.2 required for edgartools' section parsers (the `future.no_silent_downcasting` option, added in pandas 2.2, is used in some code paths).

**Final corpus:** 1,369 firms, 7,359 filings, 21,932 section rows. Full iteration history in git log commits `d92e434`, `a091b08`, `f69ad47`, `e1ce819`.

## Backtest scope (2026-07-12)

Scoping session before running the portfolio-sort backtest surfaced two things worth documenting: (1) existing v2 backtest infrastructure is substantial and should be reused, not rebuilt; (2) the current `src/axiom_fund/signals/lazy_prices.py` module's output schema `(ticker, filing_year, prior_filing_year, section_id, similarity)` does not match the pre-committed signal-panel schema from `docs/signal_design.md` §2.1: `(date, permno, raw_signal, winsorized, z_score)`. Backtest execution deferred until the signal module is refactored to conform.

**Signal module refactor required (next session)**

Signal-schema mismatch has three fixable pieces:
- Output `date` (filing_date or filing_date + 5-trading-day disclosure buffer) instead of `filing_year`
- Output `permno` (via ticker lookup from `lazy_prices_ciks.parquet`) instead of `ticker`
- Aggregate the three per-section similarities to a single `raw_signal = 1 - mean(similarity)` at the library level. This is a documented ad-hoc decision (equal weights across sections); CMN 2020 uses the same aggregation.

The forward-fill carry rule ("until next 10-K filing," pre-committed at line 23 of this doc) is a portfolio-layer concern per `docs/signal_design.md` §2.1 and should be handled by extending `src/axiom_fund/signals/alignment.py`, not by the signal module itself.

**Existing infrastructure to consume**

- `src/axiom_fund/data/returns.py::ReturnsPanel.fetch()` — CRSP daily returns with delisting adjustment (Shumway 1997)
- `src/axiom_fund/backtest/forward_returns.py::compute_forward_returns()` — holding-period forward returns per (rebalance_date, permno)
- `src/axiom_fund/backtest/metrics.py::compute_performance_metrics()` — Sharpe with 95% CI, hit rate, drawdowns
- `src/axiom_fund/backtest/engine.py::monthly_rebalance_dates()` — monthly rebalance calendar utility
- `src/axiom_fund/signals/alignment.py` — signal-frequency alignment (extend for annual carry rather than build parallel logic)

**Backtest methodology decisions**

The following are pre-committed in this doc:
- Portfolio: long-top-quintile, short-bottom-quintile, equal-weighted within quintile
- Signal date: filing_date + 5 trading days (line 63, "look-ahead prevention")
- Holding period: until next 10-K filing (~12 months, line 23)

The following are ad-hoc decisions surfacing during scoping today, not pre-committed:
- Section aggregation: equal-weighted mean of the three per-section similarities per firm-year
- Universe filter at portfolio-formation time: any firm with a signal (no additional filter). CMN 2020 uses their full sample without further filter, so this is defensible but should be documented as a choice rather than treated as neutral.
- Portfolio weighting inside each quintile: equal-weight for the base backtest. CMN 2020 reports both equal- and value-weighted; we defer value-weighted to a robustness pass.

**Time estimate for the full pipeline (next 2 sessions)**

- Signal module refactor + updated tests: ~60-90 min
- Alignment-layer extension for annual carry rule: ~30-45 min
- Backtest script (thin runner over existing infrastructure): ~45-60 min
- Sharpe + hit rate + drawdown report + sanity checks: ~30-45 min
- Bootstrap CI + HAC t-stat via existing Item 3 framework: ~45-60 min

Total: ~3.5-5 hours across two sessions. Full success/failure evaluation against pre-commitments at line 94-104 lands at the end.

## Backtest verification and negative L/S finding (2026-07-16)

First-run L/S backtest (commit cc4b78c) produced Sharpe -0.48 [95% CI: -0.75, -0.21], with long-only +0.48 and short-only +0.71 Sharpe. Both legs positive; short leg stronger. Sign is flipped relative to CMN 2020's long-top-minus-short-bottom prediction.

Verification session traced the number through the pipeline:

- **Sign convention verified.** raw_signal = 1 - mean(similarity) (high = big text change). Alignment z-scores preserve the ordering. assign_quintiles maps highest z-score to Q5, which the runner treats as the long leg. GE 2020-02-24 (raw_signal 0.952, sample max) correctly sits in Q5 from Feb 2020 forward. AAPL (raw_signal ~0.04 every year, boring stable filings) sits in Q3 → Q2 as expected.
- **Return data verified.** GE Feb 24 - Mar 30 2020 traced against CRSP: cumulative -35.5%, matching the COVID crash.
- **Per-quintile ranking is monotonic.** Q1 Sharpe +0.77 > Q2 +0.72 > Q3 +0.68 > Q4 +0.67 > Q5 +0.58. A monotonic ranking across five quintiles from 59 months is too clean for noise; the signal is working, just in the direction opposite to CMN's prediction.
- **2021 alone drives the aggregate.** Per-year L/S Sharpe: 2020 +1.16 (CMN direction), 2021 -2.47, 2022 -0.41, 2023 -0.30, 2024 -1.04. 2021 accounts for the entire negative aggregate.
- **Q5 2021 composition explains the drag.** Persistent Q5 firms in 2021 (12 months in the quintile) include SPCE -11%/month avg, VVNT -8%/month, ALLK -15%/month, DCPH single-month -74% (Phase 3 failure), TGTX -51% Oct 2021, GME meme squeeze, MSTR -31%. SIC 99 (unclassifiable — SPACs, holdcos) is the largest single sector in Q5 2021 at 302 firm-months with -1.1% average return.

**Substantive interpretation.** The signal identifies firms with large year-over-year disclosure change as designed. In our 2019-2024 sample, that cohort is disproportionately (a) post-SPAC firms transitioning from shell-company language to operating-company narrative, many of which subsequently operationally failed; (b) clinical-stage biotechs whose Risk Factors sections ballooned as Phase 3 trials approached, some of which hit binary failures; (c) speculative names caught in 2021's meme cycle. CMN's 1995-2014 sample did not have this composition — their high-change firms were more likely industrial restructurings and spin-offs of stable businesses.

The negative L/S sign is a real finding about sample composition, not signal noise or infrastructure bugs. Whether the CMN Long-Short prediction generalizes to periods dominated by SPAC and biotech binary events is the empirical question this study answers with "no, and here is why."

Verification scripts committed under scripts/exploration/verify_backtest_stage{1,2,3}.py.

Statistical significance testing (bootstrap CI, HAC t-stat via the Item 3 framework), value-weighted robustness, and formal write-up deferred to next sessions.

## Significance testing and metrics.py bug fix (2026-07-17)

Follow-up statistical significance testing on the L/S monthly return series (N=59) via `scripts/analysis/lazy_prices_significance.py` produced two substantive findings.

**1. Neither L/S nor either leg is statistically distinguishable from zero.**

HAC-corrected t-stat for the mean L/S return via Newey-West standard error, tested at lags 4, 6, 8: t-stats -1.08, -1.11, -1.10; p-values 0.28, 0.27, 0.27. Stable across lag choices; autocorrelation is not adding materially to variance.

Stationary block-bootstrap 95% CI for the L/S Sharpe (10,000 resamples, seed 42), across block sizes 3, 4, 6, 8: [-1.45, +0.41], [-1.53, +0.41], [-1.45, +0.31], [-1.41, +0.28]. All four include zero. Stable across block sizes.

**2. src/axiom_fund/backtest/metrics.py::compute_sharpe had a bug.**

The Lo (2002) asymptotic SE formula is `Var(SR_p) ≈ (1 + 0.5·SR_p²) / T` for the PERIODIC Sharpe SR_p. The code applied this formula directly to the ANNUALIZED Sharpe SR_a. Since `SR_a = √q · SR_p` where q = periods_per_year, the correct annualized SE is `SE(SR_a) = √((q + 0.5·SR_a²) / T)`. Applying the periodic formula to the annualized value under-stated SE by a factor of ~√q. Fixed by replacing `1` with `periods_per_year` in the SE expression. All 25 metrics tests still pass under the fix.

**Corrected first-run backtest CIs (compare to Backtest verification 2026-07-16 above):**

- Long-only:  Sharpe +0.48 [-0.41, +1.37]  (was [+0.21, +0.75])
- Short-only: Sharpe +0.71 [-0.18, +1.61]  (was [+0.43, +1.00])
- Long-Short: Sharpe -0.48 [-1.37, +0.41]  (was [-0.75, -0.21])

All three legs now include zero. The corrected asymptotic CI agrees with HAC + bootstrap.

**Substantive conclusion.** The compositional finding from 2026-07-16 stands: Q5 in 2019-2024 is dominated by post-SPAC operational failures and clinical-stage biotech binary events, and the point estimate for L/S is negative because those firms crashed. But with N=59 months, we cannot claim statistical significance for the L/S effect (or for either leg individually). The honest research statement is: "The point estimate suggests the CMN sign reverses in a sample dominated by SPAC and biotech binary events. The sample is too small to reject H0: L/S mean = 0 with confidence. Longer horizons or replication study needed to distinguish real out-of-sample sign flip from sampling noise at this cross-section composition."

**Spillover to prior work.** Prior asymptotic Sharpe CIs quoted in v1 documents (docs/onepager.md, docs/holdout_test_results.md) were affected by the same bug. `docs/v2_diagnostics_findings.md` already acknowledged that Item 1 asymptotic CIs understated uncertainty and directed readers to Item 3's bootstrap Sharpe CIs as authoritative; that acknowledgment remains correct. The metrics.py fix here removes the specific formula error going forward; v1 asymptotic CIs remain reported as they were computed at the time.

## Value-weighted robustness (2026-07-18)

`compute_long_short_returns` extended with optional `weights_df` parameter; runner extended with `--value-weighted` flag. Weights = market cap at each rebalance date; firms with missing weight excluded from that leg's weighted average.

Same signal panel, same universe, same rebalance calendar. Value-weighted results in `data/cache/lazy_prices_backtest_vw/`:

- Long-only:  Sharpe **+1.12** [+0.03, +2.20]  (equal-weight +0.48)
- Short-only: Sharpe **+1.11** [+0.02, +2.19]  (equal-weight +0.71)
- Long-Short: Sharpe **+0.31** [-0.75, +1.38]  (equal-weight -0.48)

**Sign of the L/S point estimate flips under value-weighting** (-0.48 → +0.31, a shift of +0.79). Neither result is statistically distinguishable from zero individually, but the change in point estimate is exactly what the compositional hypothesis from 2026-07-16 predicted:

- Under equal-weighting, the small-cap Q5 cohort (post-SPAC operational failures SPCE/VVNT/CHPT/VLDR; clinical-stage biotech binary events ALLK/DCPH/TGTX/BEAM/NVAX) dominates the leg average and drags L/S negative.
- Under value-weighting, the large-cap Q5 cohort (persistent transformations VRT/HXL/WSC/CNC/AVNT, which showed positive returns even in 2021) dominates via proportional market-cap weight.
- Q1 (low text change) is naturally dominated by mega-cap stable firms under both weighting schemes — minimal compositional change.
- Net: the direction of the L/S effect depends on which sub-cohort of Q5 receives the most weight.

**Substantive statement.** CMN's original Long-Short sign holds under value-weighting on 2019-2024 data. Under equal-weighting, small-cap post-SPAC and biotech binary-event names dominate the Q5 tail and reverse the sign. Both point estimates fall within their statistical uncertainty bands, so we cannot reject H0: mean L/S = 0 for either. But the sign-dependence on weighting scheme is a real finding about signal implementation, not signal validity — the CMN effect appears in the value-weighted sub-portfolio dominated by legitimate large-cap corporate transformations.

**Caveats.**
1. N dropped from 59 to 41 months. ~28% of quintile positions lack market cap data at the rebalance date, so months where either leg had all-NaN weights get NaN ls_return. Investigation deferred.
2. Neither estimate is significant; we cannot claim CMN-direction "confirmed" — only that the sign question is compositional.
3. Value-weighted single-leg Sharpes (+1.11, +1.12) are very high. Consistent with mega-cap dominance in 2019-2024 rather than pure signal alpha.

## Next steps

**Next session:** refactor `src/axiom_fund/signals/lazy_prices.py` output schema to conform to `docs/signal_design.md` §2.1 signal-panel contract (date, permno, raw_signal, winsorized, z_score). Update the 9 tests. Recompute the signal file. See Backtest scope (2026-07-12).

**Following sessions:** extend `signals/alignment.py` with annual carry rule; write the backtest runner over existing infrastructure (`forward_returns.py`, `metrics.py`, `data/returns.py`); report Sharpe, hit rate, drawdowns, then bootstrap CI and HAC t-stat per success-criteria pre-commitment at line 94-104.

**Later:** Item 6b (transformer extension). Item 6c (composite integration, conditional on 6a success).

## Known limitations

Consolidated list of documented issues that a reviewer would appropriately flag, retained inline so future readers see the methodological trade-offs alongside the methodology.

1. **Survivorship bias in CIK resolution — SUBSTANTIALLY RESOLVED.** Initial SEC-based resolution failed 22.9% of the universe (331 firms) because `company_tickers.json` only maps current tickers. CRSP-Compustat linking (PERMNO → CUSIP → gvkey → CIK via `comp.company`) was implemented and merged with SEC resolution. Recovery: +275 acquired/delisted firms (TECD, CELG, CERN, MXIM, DNKN, SPLK, WP, FB, etc.) plus 8 ticker-recycling wins (AMTD → historical TD Ameritrade CIK, APC → historical Anadarko, etc.). Final universe resolution: 96.1% (see Corpus construction — final). Residual sample loss documented as items 8-11 below.

2. **`Universe.as_of()` silently fails on non-trading days.** The universe module returns 0 rows without warning when passed a non-trading date (e.g., `2022-12-31` fell on a Saturday). Discovered when the initial `build_universe.py` returned 0-row snapshots for 2022 and 2023 year-ends. Worked around by using actual last trading days (2022-12-30, 2023-12-29). The underlying bug remains in `Universe.as_of()`; a proper fix (either raise on non-trading dates, or auto-align to nearest prior trading day) is deferred but noted for future infrastructure work.

3. **`market_cap` in universe parquet is stale.** Because we take the union across 6 snapshots and dedup to keep the first appearance, the `market_cap` and other size columns reflect the value at the firm's first snapshot only. A firm first appearing in 2018 has its 2018 market cap saved, not its 2023 value. Downstream code that uses size for weighting must not use this column directly; either regenerate with the specific date needed, or accept the caveat.

4. **Test coverage gap for edgartools regressions.** Unit tests for `EdgartoolsSectionExtractor` mock the edgartools boundary and protect against regressions in our own logic. They do not catch upstream `edgartools` regressions. For a production system this would need closing (via VCR-style recorded fixtures or pinned version + integration test suite); for this research repo the gap is documented.

5. **Snapshot bias in market_cap universe construction.** Even with the union approach, "top-1000 by market cap at each year-end" is one specific methodology choice. Alternative constructions (rolling top-1000 rebalanced annually, top-1000 by average market cap across the window, top-1000 by trailing 12-month median) would produce different samples. Our choice is defensible but not the only defensible one.

6. **Item 7A dropped from signal scope (empirical, not accidental).** Investigation showed Item 7A is systematically thin (boilerplate or cross-referenced) for large firms including JPM, NEE, JNJ, XOM — 40% of the pilot sample. Signal scope is Items 1, 1A, 7 only. This is a documented deviation from CMN's original methodology, empirically justified against the pilot distribution.

7. **Pre-iXBRL filings excluded (2015-2018).** Investigation revealed pre-iXBRL 10-Ks systematically use cross-references. Sample window restricted to 2019-2024 (post-iXBRL mandate). Sample size reduced from a hypothetical ~5000-7500 firm-year observations to ~3500-4500. Documented deviation from CMN's original 1995-2014 window; our study is an out-of-sample extension, not a corrupted replication attempt.

8. **Unresolvable firms (residual).** 56 firms (3.9% of universe) unresolvable by either SEC or CRSP-Compustat. These are firms whose corporate reorganizations moved both SEC and Compustat records to new CIKs with no filings in the 2019-2024 window (BLK, PNFP, OZK, LB, and others). Legitimate sample loss after CRSP-Compustat merge; not further recoverable without per-firm SEC EDGAR full-text search.

9. **Zero-filing firms.** 15 firms with resolved CIK but no 10-Ks in the 2019-2024 window under that CIK. Includes firms whose acquisition or reorganization pre-dated the window (VVC, NFX, ATHN, IDTI, VSM, ESL, TSRO, LOXO, RYZB, FRC, SBNY) and firms whose current-entity CIK has yet to file (BLK, PNFP, OZK, LB). Documented sample loss.

10. **Partial-section extractions.** 140 filings (1.9%) have only 1 or 2 of the 3 sections extracted rather than all 3. Cause: edgartools' HTMLParser has a fallback path (warning: `'NoneType' object has no attribute 'download'`) that occasionally produces incomplete section objects. Real edgartools quirk affecting specific filing formats; minor sample-quality impact given <2% incidence.

11. **Corporate reorganization mid-window (8 firms).** For 8 firms with valid 10-Ks under both SEC-current and CRSP-historical CIKs (Sprint/SentinelOne, MTCH, COHR, CZR-tier), the merge picks CRSP CIK (pre-reorganization entity). Comparing pre-reorganization to post-reorganization text within one firm-year would corrupt the CMN signal. Downstream signal computation should either flag these firms or filter YoY pairs that straddle a reorganization event. The merged CIK file's `cik_source` column marks these as `both_valid_prefer_crsp`.
