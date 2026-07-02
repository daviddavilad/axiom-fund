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

7. **Mid-window filing format transition (2019-2020 iXBRL mandate).** SEC mandated inline XBRL for large accelerated filers in 2019-2020. Pilot data shows a ~5-10x jump in primary-document file size across this transition (e.g., WMT: ~920 KB in 2015-2017 → ~3 MB in 2018+; JNJ: 297 KB in 2015 → 3.4 MB in 2016). The text content is present in both formats, but structural differences mean naive cross-format comparison would be dominated by format changes, not disclosure changes. Section extraction (deferred) must be format-agnostic; year-over-year comparisons that cross the transition need explicit verification. Surfaced by pilot download (commit TBD).

8. **Primary-document ambiguity.** SEC's "primaryDocument" field on some filings points to a thin index or cover page rather than the full 10-K body. Pilot example: JNJ 2015 primary document is 297 KB while surrounding-year JNJ 10-Ks are 3-5 MB. This is not a download failure — the file is exactly what SEC labeled as primary — but the content is a filing index, not a filing body. Section extraction must detect this case and either follow references to the actual filing body or exclude the filing. Estimated affected rate: <5% of filings based on pilot, but needs full-corpus verification. Surfaced by pilot download (commit TBD).

## Time budget estimate

Realistic scope of full Item 6 (across all sub-items): 30-50 hours across ~10-15 sessions. CFA Level II (Aug 16) has priority; Item 6 work fits around CFA prep. Item 6a alone: ~15-25 hours.

## Next steps

**This session (complete):** shipped EDGAR client (`src/axiom_fund/data/edgar.py`) and pilot download (`scripts/data_acquisition/download_10k_sample.py`). 100 filings across 10 diverse tickers, zero failures. Two data-quality risks (format transition, primary-document ambiguity) surfaced and documented as risks 7 and 8 above.

**Next session:** investigate the two data-quality risks. Specifically: (a) inspect a 2015 pre-iXBRL filing and a 2020 post-iXBRL filing side-by-side; verify section extraction can handle both; (b) inspect the JNJ 2015 filing to characterize what "thin primary document" actually contains and how to detect it programmatically.

**Following 2-3 sessions:** scale corpus to full v1 universe (~700-900 firms), implement section extraction (Items 1, 1A, 7, 7A) with format-agnostic parsing, build TF-IDF vectorization pipeline.

**Later sessions:** Item 6a portfolio-sort backtest and findings document. Then Item 6b (transformer extension), then conditional 6c (composite integration).