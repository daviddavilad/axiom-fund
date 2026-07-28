# Axiom Fund — Project History

A quantitative research engine for market-neutral long/short U.S. equity, built by David Davila. This document is the chronological technical history of the project — architecture, milestones, findings, corrections. See `docs/v2_item6_design.md` and `docs/signal_design.md` for design-doc pre-commitments referenced throughout.

## Project Identity

- **Repository:** github.com/daviddavilad/axiom-fund
- **Architecture:** Pure-function design with rigorous pre-commitment docs. WRDS + CRSP + Compustat data layer, signal library, backtest engine, diagnostics, portfolio composite.
- **Universe:** CRSP top-1000
- **Test discipline:** ~465 non-integration tests as of July 2026; TDD for new modules

## v1 — Four-Signal Engine (through late 2024)

**Signals:** Gross Profitability, Idiosyncratic Volatility, Residual Momentum, PEAD.

**Key results:**
- Gross Sharpe ratio ~0.78
- Single-signal attribution: GP alone Sharpe 0.80; IVol Sharpe -0.29; PEAD Sharpe 0.70; ResMom near-zero IC but positive realized PnL
- Critical covariance estimation bug fixed (commit 1bbe545)
- No-ResMom variant: Sharpe 0.816
- v1 holdout testing: 3-signal Sharpe ~1.18

**Deliverables:**
- Private one-page reference doc and 24-pair defense Q&A
- Signals refactor (commit 5516785) added parameter-optional signals + 11 new tests (308 → 319)

## v2 — Scoping and Phase 1 (early 2026)

Scoped at approximately 89 hours across 4 phases, targeting mid-to-late September 2026. Pre-commitment design doc committed. Scope shaped by BlackRock SAE rejection feedback.

**Phase 1 additions:**
- HAC/Newey-West standard errors
- Bootstrapped confidence intervals
- HLZ multiple-testing correction
- Residual diagnostics
- Deflated Sharpe (Bailey & López de Prado)

**Phase 1 findings:**
- Liquidity audit: closed as non-actionable
- Residual diagnostics: 86% of months reject homoskedasticity; identified 34σ GameStop outlier
- HAC SE and bootstrapped Sharpe CIs: 2-10% corrections
- **Deflated Sharpe conclusion:** No-ResMom variant barely clears DSR at N=3 but fails at larger N; 3-signal and 4-signal variants fail at all N. Overturns v1 qualitative conclusion.

## v2 — Phase 2 (mid-2026)

**Additions:** Quandt-Andrews structural break tests, Lazy Prices NLP signal.

**Phase 2 findings:**
- Structural break: no IC signal shows statistically significant structural break
- Repository at 73+ commits, 406+ passing tests

## v2 Item 6a — Lazy Prices NLP Signal (2026-07-11 forward)

Replication of Cohen-Malloy-Nguyen (2020) "Lazy Prices" (Journal of Finance).

### Infrastructure (2026-07-11 through -18)

**Sample scope:** 2019-2024 window, Items 1/1A/7, edgartools 5.40.1.

**Infrastructure built:**
- `EdgarClient` (stdlib urllib, 8 req/sec)
- Full SEC submissions API pagination
- `EdgartoolsSectionExtractor` with 27.7 GB local corpus
- CRSP-Compustat CIK resolution pipeline (96.1% of 1,448-firm universe resolved)
- Path C (CIK-based extraction via `edgar.get_entity`), strict form=='10-K' filter to exclude 10-K/A amendments
- Signal library: `src/axiom_fund/signals/lazy_prices.py` with 9 tests (commit 87c161a, 2026-07-11)
- Portfolio-sort machinery: `src/axiom_fund/backtest/quintile_sort.py` with 10 tests (commit cc4b78c, 2026-07-13)
- Value-weighted extension: `weights_df` parameter added to `compute_long_short_returns`, 17 tests (2026-07-18)

**Signal output schema:** `(permno, ticker, date_filed, prior_date_filed, sim_item1, sim_item1a, sim_item7, raw_signal)`. Matches `docs/signal_design.md` §2.1 signal-panel contract. Alignment layer (`signals/alignment.py`) handles winsorization, z-scoring, forward-carry via `merge_asof` with `max_age_days=None` for annual carry.

**Corpus:** Canonical 1,369 firms, 7,359 filings, 21,932 section rows (commit b08927b, 2026-07-11).

**Full-corpus TF-IDF results:**
- Item 1 median 0.972 IQR 0.045
- Item 1A median 0.974 IQR 0.039
- Item 7 median 0.943 IQR 0.065
- Cross-section correlations 0.42-0.49

### Wrong-Convention Backtests (2026-07-13 through -18)

Initial L/S backtest reported Long-only Sharpe +0.48, Short-only +0.71, L/S -0.48 (equal-weighted, N=72 rebalances, 21-day holding). At the time framed as "L/S direction opposite to CMN 2020, possibly regime-driven."

Sign-convention issue not caught until 2026-07-20 (see below). Interpretations from this period included:
- "Compositional hypothesis" — Q5 dominated by SPAC operational meltdowns and biotech binary failures 2020-2021
- "Sample too small" — HAC t=-1.08 p=0.28 across lags; block bootstrap CI [-1.5, +0.4] across block sizes
- **`metrics.py` bug fix (2026-07-17):** Lo (2002) SE formula was applied to annualized Sharpe instead of periodic, understating SE by √12. Fixed; all 25 metrics tests still pass.

Value-weighted variant (N=41 due to marketcap coverage gap): L/S Sharpe +0.31 [-0.75, +1.38]. Initial "sign flip" interpretation later invalidated when the underlying sign convention was corrected and the N=50 vs N=71 discrepancy diagnosed.

### WRDS Text Database Investigation (2026-07-19)

UNM's WRDS subscription reviewed for pre-2019 text data. Result: `ciqsamp_transcripts` sample-only (~20 rows); `compsamp_computext` 12 years / 588 firms / 122K documents / 2.2M sentences, but topic-tagged extraction not raw section text. **Conclusion: WRDS does NOT unlock Lazy Prices extension to pre-2019 data.** Post-CFA project queued: topic-stability signal on `compsamp_computext` as separate research.

### Forward Extension Infrastructure (2026-07-19, commit 860089c)

Universe expanded 1,448 → 1,519 firms. Corpus grew 841 → 9,908 filings across 1,451 firms. CIK resolution 1,462/1,519 (96.2%).

**Bugs found and fixed along the way:**
1. `universe.py` fetchall (pandas 2.3 + SQLAlchemy 1.4 incompatibility, same pattern as `returns.py`)
2. `resolve_ciks_crsp.py` fetchall (two sites)
3. `merge_ciks.py._normalize_cik` was comparing SEC padded strings to CRSP float representations → 0 CIK agreements → fixed with proper coercion
4. `edgar.py.get_10k_history` missing `zfill(10)` on CIK URL template → 404s for every unpadded CIK

**2025-12-31 universe snapshot returned 0 rows** — UNM WRDS CRSP monthly lag ends 2024-12-31. Sample effectively capped by CRSP catch-up.

### Sign Convention Correction (2026-07-20, commit 3d50ed0)

**Discovery:** Per-quintile diagnostic (prompted by David asking to reconcile equal-weighted N=71 backtest against per-quintile numbers) revealed:
- Q1=1.27%, Q2=1.33%, Q3=1.25%, Q4=1.14%, Q5=1.08% mean monthly returns
- Monotonic Q1 > Q5 in Sharpe space (0.75 → 0.57)
- Q5-Q1 spread -2.6% annualized

**This IS CMN's original direction** — low-change firms outperform. The runner had been longing Q5 (top of `raw_signal` = highest text change) and shorting Q1 across ALL prior sessions. Convention was BACKWARDS relative to CMN.

**All prior "L/S sign flip vs CMN" claims across sessions 2026-07-13 through -18 were arithmetically correct on the wrong-convention L/S but conceptually inverted.** The signal has ranking power in the CMN sense; the L/S Sharpe magnitude is ~0.39 at N=71, roughly 1/3 of CMN's published result, still not stat-sig at N=71.

**Substantive reframe:** Prior compositional narrative (SPAC / biotech Q5 driving negative L/S) invalidated. Correct framing: directionally consistent with CMN, economically ~1/3 magnitude, underpowered N.

**CRSP v2 tables migration:** Returns cache refactored to `crsp.dsf_v2` + `crsp.stkdelists` via SQL aliases in `returns.py`. Raw `crsp.dsf` lags to 2024-12-31 at UNM; v2 runs through 2025-12-31.

### Sign Convention Fix + Regression Tests (2026-07-21, commit 7d33227)

**Fix:** Explicit `long_quintile: Literal["top", "bottom"] = "top"` parameter added to `compute_long_short_returns` in `src/axiom_fund/backtest/quintile_sort.py`. Default preserves backwards compatibility for GP/IVol/PEAD/ResMom. Lazy Prices runner sets `LONG_QUINTILE = "bottom"` as module constant.

**Regression tests:** New `TestLongQuintileConvention` class with 5 direction-locking tests in `test_quintile_sort.py`. Full suite 465 passing (was 460, +5 new).

**Corrected results (Equal-weighted, N=71):**
- Long-only Q1 Sharpe 0.693 [-0.120, +1.507]
- Short-only Q5 Sharpe 0.480 [-0.330, +1.289]
- L/S (Q1-Q5) Sharpe +0.334 [-0.474, +1.142]

**Value-weighted variant (N=50 at this point):** L/S Sharpe -0.044 [-1.004, +0.917]. Read as vol-drag on near-zero mean, but N=50 vs N=71 discrepancy flagged for investigation.

### Value-Weighted N=50 Bug Fix + Size-Tercile Discovery (2026-07-22, commit b77c019)

**Root cause of VW N=50:** `weights_df` was filtering returns by exact `rebalance_date` match. `pd.date_range(freq="ME")` produces CALENDAR month-ends, but returns data has TRADING days. 21 rebalance dates fell on weekends (2020-02-29 Saturday, 2020-05-31 Sunday, ...) or US holidays (2021-05-31 Memorial Day) with no matching trading day → empty `weights_df` → NaN L/S. **The bug systematically dropped 21 months, and those 21 months were biased in favor of the wrong-sign-VW narrative.**

**Fix:** Per-permno `merge_asof` with `direction="backward"` to look up marketcap on last trading day ≤ rebalance date. VW N: 50 → 71 matching EW.

**Corrected value-weighted results (N=71):**
- Long-only Q1 Sharpe 0.819 [+0.002, +1.636]
- Short-only Q5 Sharpe 0.943 [+0.122, +1.763]
- L/S Sharpe -0.313 [-1.120, +0.495]

**On like-for-like N=71: EW L/S +2.13%/yr vs VW L/S -3.48%/yr.** Weightings genuinely oppose each other.

**Size-tercile diagnostic (`scripts/exploration/size_tercile_by_lazy_prices_sort.py`):**
- Small L/S: -4.94% ann, Sharpe -0.370 (opposite of CMN)
- Mid L/S: -1.86% ann, Sharpe -0.256 (weakly opposite)
- **Large L/S: +7.30% ann, Sharpe +1.197 (CMN direction, monotonic)**

Rough 95% CI on Large L/S Sharpe: [+0.14, +2.25]. First plausibly stat-sig finding of the project.

### Size Quintile Refinement (2026-07-23, commit 9cb0fc0)

**5x5 sort** (size quintile × LP quintile, equal-weighted within bucket):
- Size1 L/S: -2.18% ann, Sharpe -0.117 (weak reverse)
- Size2 L/S: -2.70% ann, Sharpe -0.257 (weak reverse)
- Size3 L/S: -0.45% ann, Sharpe -0.050 (zero)
- **Size4 L/S: +4.42% ann, Sharpe +0.638 (PEAK, CMN direction)**
- Size5 L/S: +0.84% ann, Sharpe +0.140 (near zero, NOT reversed)

**Prior tercile "Large +1.197" reframed:** Large tercile (top 33%) spans Size4-upper-half + Size5, combining peak signal with near-zero. Size4 alone is the cleaner research target.

**VW-vs-EW within each size bucket:**
- Size1-Size3: gaps under ±0.5% between EW and VW (matched)
- **Size4: EW +0.638 vs VW +0.770 (matched, VW slightly stronger)**
- **Size5: EW +0.140 vs VW -0.528 (gap -6.66%/yr, BIG DIVERGENCE)**

**Substantive finding: CMN Lazy Prices signal is BIMODAL BY SIZE.**
1. Positive CMN direction in Size4 upper-mid-caps, Sharpe +0.638-0.770 robust to EW/VW
2. Reversed in mega-mega-caps at top of Size5 (VW-only, Sharpe -0.528). Median Size5 firm is near zero.
3. Noise in Size1-Size3

Global pooled backtests miss the structure — pooled EW dilutes Size4 with noise; pooled VW gets dominated by Size5 mega-cap reversal. Must condition on size to see the actual signal.

### FF6 Spanning Regression (2026-07-24, commit 366a365)

**Infrastructure:** `scripts/data_acquisition/fetch_ff6_monthly.py` downloads FF5 + Momentum monthly factors from Ken French's website (Dartmouth). Cache at `data/cache/ff6_monthly.parquet`. Separate from `src/axiom_fund/data/ff_factors.py` (which fetches FF3 daily from WRDS for IVol/ResMom signal computation).

`scripts/analysis/ff6_spanning_size4.py`: OLS with HAC (Newey-West lags 4/6/8 sensitivity), CLI `--weighting {ew, vw}`.

**Model:** Size4 L/S ~ Mkt-RF + SMB + HML + RMW + CMA + MOM.

**Results (Size4, N=70):**

| Metric | EW | VW |
|---|---|---|
| Monthly alpha | +0.261% | +0.331% |
| Annualized alpha | +3.13% | +3.97% |
| Raw L/S ann | +4.33% | +5.23% |
| HAC t (lag=4) | 1.48 (p=0.14) | 1.86 (p=0.063) |
| HAC t (lag=6) | 1.64 (p=0.10) | **2.06 (p=0.039)** |
| HAC t (lag=8) | 1.73 (p=0.083) | **2.19 (p=0.029)** |
| Adjusted R² | 0.008 | 0.014 |

**Factor loadings (VW, HAC lag 6):**
- HML +0.176 t=2.02 (significant, value tilt)
- CMA -0.154 t=-1.66 (marginal, conservative-investment tilt)
- SMB -0.130 t=-1.40 (large-cap tilt within Size4)
- Mkt-RF, RMW, MOM: noise

**Substantive finding:** Size4 CMN Lazy Prices signal is factor-orthogonal (adj R² near zero). NOT a repackaging of known factors. VW is stronger than EW under FF6 — opposite of Size5 where VW REVERSES the effect. Alpha crosses 5% significance at HAC lag ≥6 for VW; marginal at lag 4.

**Unusual HAC lag sensitivity direction:** SEs shrink with longer lags (0.178 → 0.161 → 0.151), meaning residuals have negative autocorrelation at short lags being captured. Not a red flag but worth diagnosing eventually.

### Block Bootstrap Corroboration (2026-07-24, commit 8690d97)

**Infrastructure:** `scripts/analysis/ff6_bootstrap_size4.py`. 10K resamples per block size [3, 4, 6, 8]. Joint bootstrap of raw Sharpe and FF6 alpha. Moving block bootstrap matching `compute_bootstrapped_sharpe_ci` in `inference.py`.

**Size4 EW bootstrap p-values (N=70):**
- Sharpe: bs=3 0.079, bs=4 0.060, bs=6 0.012, **bs=8 0.004**
- Alpha: bs=3 0.206, bs=4 0.159, bs=6 0.077, bs=8 0.052

**Size4 VW bootstrap p-values (N=70):**
- Sharpe: **bs=3 0.044, bs=4 0.032, bs=6 0.004, bs=8 0.001**
- Alpha: bs=3 0.124, bs=4 0.087, **bs=6 0.039, bs=8 0.017**

**Two-methodology convergence with HAC:**
1. Point estimates identical (Sharpe +0.770, alpha +3.97% ann)
2. Same unusual sensitivity direction (longer block/lag → more sig)
3. Same threshold-crossing behavior (marginal at standard, SIG at longer)

**Strongest defensible finding to date:** Size4 VW L/S raw Sharpe +0.770 is stat-sig at 5% under standard block bootstrap bs=4 (p=0.032) AND under HAC at any lag ≥4. FF6-adjusted alpha +3.97% ann is marginal at standard conventions but stat-sig at longer bandwidths. Raw Sharpe claim is defensible without hedging; alpha claim needs robustness disclaimer.

### CMN 2020 Replication Code Discovered (2026-07-24 evening)

**Source:** Wiley Online Library, supplementary material for jofi.12885. Files inspected: `step1_prepExtract.pl`, `step1b_prepExtract.do`, `run_step2.sh`, `step2_extractSimilarity.pl`, `step2b_extractSimilarity.do`, `step3_runPortfolio.do`. Perl + Stata stack. Reference paths in Perl code confirm authors' actual code (Quoc Nguyen's Dropbox).

**Critical methodological differences from axiom-fund's implementation:**

| Aspect | axiom-fund | CMN 2020 code |
|---|---|---|
| Signal metric | TF-IDF cosine | **Jaccard** (step3 line 34: `gen sim_use = simjaccard`) |
| Section granularity | Item 1 + 1A + 7 average | **Whole document** (step2 Perl reads full 10-K) |
| Binning | Current cross-section quintiles | **Prior year percentiles** |
| Carry rule | Until next filing | **Lag 1-5 months, fill-forward** |
| Factor model | FF5 + MOM | **FF3 + MOM + Pastor-Stambaugh Liquidity** (mktrf smb hml umd ps_vwf) |
| Sample | 2019-2025 | 1995-2017 |

**Implication:** axiom-fund's +3.97% ann VW alpha and CMN's +22%/yr are **NOT directly comparable** — different signals, different controls, different samples. Prior "our result is ~1/7 of CMN's" framing was rough at best. Faithful replication (deferred to a separate future repo) is now much more concrete since the code IS the spec.

### Mega-Cap Reversal Mechanism (2026-07-24, commit 96)

The Size5 VW reversal is caused by ~10 firms.

**Counterfactual (`scripts/exploration/top_size5_mega_caps_diagnostic.py`):**
- Full Size5 VW L/S: -5.98% ann, Sharpe -0.528
- Size5 VW L/S excluding top-10 firms per rebalance date: **+0.91% ann, Sharpe +0.119**

Excluding just 10 firms flips the sign from strongly negative back to slightly positive (CMN direction).

**Firms identified (Magnificent Seven + Big Healthcare/Finance rotators):**

| Ticker | Months in top-10 | Avg market cap | Avg LP quintile | Avg fwd return |
|---|---|---|---|---|
| AMZN | 70/70 | $1.64T | 1.86 (Q1-Q2, low-change) | +1.75%/mo |
| GOOG | 70/70 | $795B | 3.11 (Q3, moderate) | +2.80%/mo |
| GOOGL | 70/70 | $793B | 3.11 (Q3, moderate) | +2.79%/mo |
| MSFT | 65/70 | $2.5T | 2.78 (Q3, moderate) | +1.93%/mo |
| FB / META | 62/70 | $877B | 3.94 (Q4, high-change) | +2.54%/mo |
| AAPL | 62/70 | $2.8T | 3.37 (Q3-Q4) | +1.95%/mo |
| TSLA | 61/70 | $827B | 3.89 (Q4, high-change) | +2.13%/mo |
| **NVDA** | 52/70 | $1.8T | **4.81 (Q5, highest change)** | **+4.89%/mo** |

Plus rotating Big Healthcare/Finance appearances: JNJ, JPM, UNH, AVGO, LLY.

**Mechanism story:** at mega-mega-cap scale, heavy 10-K change signals **business evolution** (new AI product lines, cloud infrastructure buildout, new therapeutic pipelines) not risk disclosure. NVDA is the poster child: nearly always in Q5 (high change), with +4.89%/mo forward returns (~59% annualized). These firms would be SHORTED by pure CMN and crush that trade. AMZN is the counterexample — low-change 10-K (Q1-Q2), moderate returns; Amazon does not drive the reversal.

**Refined substantive picture: the Lazy Prices anomaly is a two-regime phenomenon by size.**

1. **Regime 1 — sub-mega-cap:** CMN direction works. Effect peaks in Size4 upper mid-cap (Sharpe +0.638-0.770, factor-orthogonal alpha). Heavy 10-K change = risk disclosure.
2. **Regime 2 — mega-mega-cap (top ~10 firms):** CMN direction REVERSES. Just 10 firms drive it (mostly Magnificent Seven). Heavy 10-K change = growth disclosure.

The signal has different semantic meaning at different size levels. This is much stronger than "bimodal by size" — it's a mechanism story that explains WHY the direction changes.

### NYSE-Breakpoint Robustness — Peak Moves to Size5 (2026-07-25)

Standard academic methodology (CMN 2020, Fama-French, etc.) uses NYSE-based size breakpoints rather than in-sample percentiles, because in-sample buckets are systematically biased by universe composition. axiom-fund's universe (CRSP top-1000) skews larger, so in-sample "Size4" (60-80th percentile) tilts toward mid-caps in real dollar terms.

**Infrastructure:**
- `scripts/data_acquisition/fetch_nyse_size_breakpoints.py`: fetches NYSE common stock marketcap distribution from `crsp.msf_v2` per month (filter: `primaryexch='N' AND sharetype='NS' AND securitytype='EQTY' AND securitysubtype='COM'` — excludes ETFs, ADRs, closed-end funds). Computes p20/p40/p60/p80 breakpoints per rebalance date. Cached to `data/cache/nyse_size_breakpoints.parquet`.
- `scripts/exploration/size_quintile_nyse_breakpoints.py`: 5×5 sort using NYSE breakpoints, direct comparison to in-sample results.
- `scripts/analysis/ff6_spanning_size5_nyse.py`: FF6 regression on NYSE Size5 L/S.
- `scripts/analysis/ff6_bootstrap_size5_nyse.py`: bootstrap CIs on NYSE Size5 L/S.

**NYSE-breakpoint bucket distribution** (much different from equal-weighted 20% per bucket under in-sample):
- Size1 (< NYSE p20): 1.9% of universe
- Size2 (p20-p40): 5.8%
- Size3 (p40-p60): 27.5%
- Size4 (p60-p80): 34.5%
- Size5 (> NYSE p80): 30.4%

**Peak moves from Size4 to Size5:**

| Bucket | L/S ann | Sharpe |
|---|---|---|
| Size1 | -28.29% | -0.421 (noisy, only ~24 firms/month) |
| Size2 | -4.44% | -0.122 |
| Size3 | -0.36% | -0.034 |
| Size4 (was in-sample peak +0.638) | +0.50% | **+0.068 (near zero)** |
| **Size5 (was in-sample near-zero)** | **+6.26%** | **+1.177** |

The prior "Size4 is the peak" narrative was an in-sample percentile artifact. Under CMN's own methodology, the CMN peak is in Size5 (large caps above NYSE p80 ~$12-19B).

### FF6 Spanning + Bootstrap on NYSE Size5 EW L/S — Publishable Alpha (2026-07-25)

**FF6 spanning results (NYSE Size5 EW, N=71):**

| Metric | Value |
|---|---|
| Raw L/S ann | +6.08% |
| FF6-adjusted alpha (annual) | **+5.70%** |
| HAC t (lag=4) | **+3.04, p=0.0024** |
| HAC t (lag=6) | +3.18, p=0.0015 |
| HAC t (lag=8) | +3.32, p=0.0009 |
| Adj R² | -0.030 (factors explain essentially nothing) |

**Bootstrap corroboration (10K resamples, all block sizes 3/4/6/8):**
- Sharpe p-values: 0.0024–0.0038 (uniformly SIG at 1%)
- Alpha p-values: 0.0076–0.0092 (uniformly SIG at 1%)
- All CIs exclude zero
- No sensitivity to bandwidth choice

**All factor loadings are individually not significant.** Real: SMB, HML, RMW, CMA, MOM, MKT-RF — none reject the null. NYSE Size5 L/S is genuinely orthogonal to known factors.

**Comparison to prior Size4 in-sample finding:**

| Metric | Size4 in-sample EW | NYSE Size5 EW |
|---|---|---|
| Raw L/S ann | +4.33% | **+6.08%** |
| Alpha ann | +3.13% | **+5.70%** |
| HAC t (lag=4) | 1.48 (p=0.14) | **3.04 (p=0.002)** |
| Bootstrap alpha p (bs=4) | 0.087 | **0.0076** |
| Adj R² | 0.008 | -0.030 |

NYSE Size5 is unambiguously stronger than Size4 in-sample on every dimension.

**Mega-cap reversal survives under NYSE breakpoints (VW, top-10 exclusion counterfactual):**
- Full Size5 VW L/S: -2.31% ann, Sharpe -0.212 (weaker than in-sample -0.528 but same direction)
- Excluding top-5: +1.48% ann, Sharpe +0.201 (flips)
- Excluding top-10: **+3.55% ann, Sharpe +0.537** (strongly CMN)

Same ~10 firms drive the VW reversal under either bucket definition.

**Refined substantive picture, robust to methodology:**
- The Lazy Prices anomaly is concentrated in large-cap firms (above NYSE 80th percentile in 2019-2025 modern sample)
- Raw L/S +6.08%/yr, FF6-orthogonal alpha +5.70%/yr, HAC t=3.04 (bandwidth-robust to at least lag 8 and block size 8), bootstrap p<0.01 uniformly
- Comparison to CMN 2020's original result (+22%/yr on 1995-2014): axiom-fund's +5.70%/yr is roughly 1/4 the magnitude, consistent with a "modern regime, effect weaker but still statistically significant" reading, and on a methodological approximation (TF-IDF cosine per section vs CMN's Jaccard on whole documents)
- Within large-caps, the very top ~10 mega-mega-cap firms (Magnificent Seven + Big Healthcare rotators) partially offset the effect under value-weighting; equal-weighted result is robust because it doesn't concentrate on the top few

### FF6 Spanning + Bootstrap on NYSE Size5 VW L/S — Confirms Mechanism (2026-07-26)

Companion analysis to prior EW result. Same infrastructure, `--weighting {ew, vw}` CLI added to both `ff6_spanning_size5_nyse.py` and `ff6_bootstrap_size5_nyse.py`. Both scripts now load from `data/cache/lazy_prices_backtest/nyse_size5_ls_series.parquet` (produced by new `scripts/exploration/nyse_size5_ls_series.py`).

**NYSE Size5 VW L/S FF6 spanning (N=71):**

| Metric | Value |
|---|---|
| Raw L/S ann | -2.31% |
| FF6-adjusted alpha (annual) | -2.57% |
| HAC t (lag=4) | -0.46, p=0.64 |
| HAC t (lag=6) | -0.43, p=0.66 |
| HAC t (lag=8) | -0.43, p=0.67 |
| Adj R² | -0.085 |

**Bootstrap corroboration (all block sizes):**
- Alpha p-values: 0.56 / 0.61 / 0.72 / 0.72 — nowhere near significance
- CI widths ~22 percentage points, uniformly straddling zero

**Substantive:** VW is essentially noise around a slightly-negative point estimate. Standard errors are ~3× larger than EW (0.46% vs 0.16%) — mega-cap concentration adds volatility without corresponding directional signal. Consistent with prior top-10 exclusion counterfactual (Size5 VW L/S: -0.212 Sharpe → +0.537 excluding top-10).

**Complete EW vs VW comparison:**

| Metric | EW | VW |
|---|---|---|
| Raw L/S ann | +6.08% | -2.31% |
| FF6 alpha ann | **+5.70%** | -2.57% |
| HAC t (lag 4) | **+3.04, p=0.002** | -0.46, p=0.64 |
| Bootstrap alpha p (bs=4) | **0.008** | 0.61 |
| Alpha SE | 0.156% | 0.461% (3× wider) |
| Factor loadings | All noise | All noise |

**Full paper-quality story arc is now complete:**

1. **Main result:** NYSE Size5 EW, FF6-orthogonal alpha +5.70%/yr, HAC t=3.04, bootstrap p<0.01 uniform, factor-orthogonal (adj R² negative)
2. **Robustness:** all HAC bandwidth choices (lags 4/6/8), all bootstrap block sizes (bs=3/4/6/8), NYSE-breakpoint methodology (CMN standard)
3. **Mechanism:** VW L/S is null due to mega-mega-cap concentration; top-10 exclusion counterfactual restores the effect
4. **Specific culprits:** Magnificent Seven + Big Healthcare/Finance rotators identified individually with per-firm stats

This is the complete result section a first-draft paper would need.

### AI-Era Subsample Analysis — Mega-Cap Reversal is Post-ChatGPT Specific (2026-07-27)

Descriptive diagnostic splitting the 71-month NYSE Size5 L/S series at two cutoffs to test whether the mega-cap reversal is regime-dependent:
- **Cutoff 1 (primary):** ChatGPT release, 2022-11-30
- **Cutoff 2 (robustness):** Nvidia earnings inflection, 2023-05-31

Implementation: `scripts/exploration/ai_era_subsample_analysis.py`. Simple OLS FF6 alpha per subsample (no HAC due to N=30-41). Asymptotic Lo (2002) Sharpe CIs.

**ChatGPT cutoff results:**

| Metric | PRE (N=35) | POST (N=36) |
|---|---|---|
| EW raw ann | +9.86% | +2.86% |
| **EW FF6 alpha ann** | **+9.20% (t=2.55)** | +2.65% (t=0.76) |
| VW raw ann | **+8.79%** | **-12.05%** |
| **VW FF6 alpha ann** | **+9.03% (t=1.71)** | **-17.15% (t=-1.85)** |

**Nvidia cutoff results (robustness):**

| Metric | PRE (N=41) | POST (N=30) |
|---|---|---|
| EW FF6 alpha ann | +7.80% (t=2.40) | +5.58% (t=1.23) |
| VW FF6 alpha ann | +1.90% (t=0.32) | -7.20% (t=-0.64) |

**The single cleanest sub-finding of the project:**

**The mega-cap reversal is POST-CHATGPT SPECIFIC.** Pre-ChatGPT, both EW and VW L/S have strongly positive alpha (+9%/yr under both weightings). Post-ChatGPT, EW alpha weakens to +2.65% and VW alpha reverses to -17.15%. The full-sample alpha (+5.70% t=3.04) is a weighted average of a strong pre-ChatGPT regime and a mixed post-ChatGPT regime.

ChatGPT cutoff separates the regimes much more cleanly than the Nvidia cutoff. Substantive result on its own — the regime break coincides with the ChatGPT cultural landmark, not with the Nvidia earnings market inflection.

**Refined mechanism story with empirical support:**

Before ChatGPT: mega-cap 10-K changes had the same semantic meaning as smaller firms — risk disclosure. The CMN direction worked uniformly across the size distribution above NYSE p80.

After ChatGPT: mega-cap 10-K changes started disclosing new AI/product initiatives (Microsoft Copilot, Google Bard/Gemini, Amazon Bedrock, Meta AI, Tesla FSD, Nvidia H100/Blackwell). At mega-mega-cap scale, heavy 10-K change signals GROWTH not RISK. The signal's semantic meaning diverged by size, producing:
- Weakened EW alpha (smaller large-caps still work; mega-caps drag on average)
- Strongly reversed VW alpha (mega-caps concentrate weight, growth-signal dominates)

**Empirical consistency with CMN 2020:** CMN's 1995-2014 sample is entirely pre-ChatGPT. Their +22%/yr result was measured under the regime where mega-caps drove the effect, not against it — consistent with our pre-ChatGPT half showing +9.20% EW / +9.03% VW.

**Caveats:**
- N=30-41 per subsample. CIs are wide; alpha t-stats between 0.7-2.6.
- Simple OLS residual variance for alpha SE (no HAC — too little data). Interpret loosely.
- ChatGPT cutoff was chosen after seeing the data; formal regime-change test would require pre-specification (Chow test would be marginal at these sample sizes).
- Descriptive analysis, not confirmatory. Sample extension backward (queue item i) becomes critical for rigorous regime-change evidence.

**Strategic implication:** the sample-extension queue item (i) just became the highest-value next step for the paper. Pre-2019 data would extend the "pre-ChatGPT" sample by 20+ years, dramatically improving power on the "did the reversal exist before ChatGPT?" question.

## Current State (2026-07-24)

- Repository at commit 94 (`8690d97`) with 465 tests passing (non-integration)
- Test-suite integration cleanup queued (fundamentals.py + ff_factors.py fetchall bugs)
- **Strongest research finding:** Size4 VW L/S Sharpe +0.770 (stat-sig), FF6-adjusted alpha +3.97% ann (marginal-to-significant depending on bandwidth choice)
- Substantive picture (robust to NYSE-breakpoint methodology, but time-varying): Lazy Prices anomaly concentrates in **large-cap firms** (above NYSE p80). Full-sample: FF6-orthogonal alpha +5.70%/yr, HAC t=3.04, bootstrap p<0.01 uniformly. **Regime dependence:** the effect was strong and uniform (both EW and VW positive) pre-ChatGPT (2020-01 to 2022-11); post-ChatGPT the mega-cap reversal emerged and EW effect weakened. Full-sample alpha is a weighted average of these regimes.

## Research Directions

### Potential Paper: Size Heterogeneity in the Lazy Prices Anomaly

Working titles under consideration:
- "Heterogeneity in the Lazy Prices Anomaly Across the Firm Size Distribution"
- "Does the Lazy Prices Anomaly Survive Among Mega-Capitalization Firms?"

**Literature check (2026-07-23):** No prior work found decomposing Lazy Prices by size quintile or documenting mega-cap VW reversal within Size5. Sadlo 2020/2021 Goethe master thesis is closest (S&P 1500 replication finds asymmetry: changers underperform, non-changers don't outperform) but doesn't do size decomposition. CMN 2020 itself claims size uniformity in robustness section (methodological detail from their code appendix now needed to verify their check was coarse enough to miss quintile-level bimodality).

**Publication-quality novelty would need:** longer sample (extending backward to 1995-2018), FF5 or FF3+MOM+PSL spanning, robustness to NYSE-median size cutoff (CMN's standard). N=71 sample is a serious limitation.

**Realistic paths:** SSRN preprint (achievable), practitioner blog note, or journal submission (year+ of work).

### Faithful Replication as Separate Project

Strategic decision (2026-07-24): axiom-fund Lazy Prices work stays exploratory here; faithful CMN 2020 replication is deferred to a future separate GitHub repo. Weeks of work. Trigger: continued convergence of evidence that Size4 (or similar bucket) has genuine factor-orthogonal alpha.

Given the CMN code has been located, the replication project is now concrete: reproduce Perl + Stata methodology (Jaccard on whole documents, prior-year percentile binning, lag 1-5 fill-forward, FF3+MOM+PSL, 1995-2017) either as a straight port to Python or by running the original code with modern data.

## Conventions and Discipline

- Pre-commitment design docs before code (`docs/*_design.md`)
- Pure-function library modules with signal-panel contract per `docs/signal_design.md` §2.1
- Signal modules in `src/axiom_fund/signals/`, runners in `scripts/data_acquisition/`, analysis in `scripts/analysis/`, exploration in `scripts/exploration/`
- Test-first discipline (TDD) for new modules; direction-locking regression tests where sign convention matters
- Multi-line commit messages via `/tmp/commit_msg.txt`, kept concise
- Cache outputs gitignored (`*.parquet`, `*.log`, `*.bak_*`); metadata `.txt` files committed
- WRDS auth via `load_dotenv()`, username in `.env`
- `python -u` for unbuffered output on long runs, `tee` for durable logs
- Inspect scripts before running for `SAMPLE_END` constants and stale progress files
