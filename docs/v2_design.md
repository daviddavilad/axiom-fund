# Axiom Fund v2 — Design Document

**Status**: Pre-commitment document. Records the scope, implementation order, acceptance criteria, and reporting commitments for the v2 release *before* any v2 code is written. Once committed, this document is intended to constrain scope drift during implementation.

**Authored**: 2026-06-12
**Author**: David Dávila
**Target release**: mid-to-late September 2026

---

## Motivation

The v1 release of Axiom Fund (June 2026) was a complete market-neutral L/S equity research engine with four signals, an honest cost model, single-signal attribution, and a pre-committed out-of-sample holdout test. v1's primary methodological limitations were identified through external review during May-June 2026 by senior quantitative-research practitioners.

Two themes dominated the feedback.

**Statistical inference at small N.** v1 reports IC t-statistics, Sharpe ratios, and other estimates with asymptotic standard errors. With 116 monthly observations on signals and 22-period holdout windows, asymptotic approximations are not adequate. Specific gaps: no Newey-West HAC standard errors on autocorrelated time series, no bootstrapped CIs, no residual diagnostics on regressions, no formal structural break testing, no multiple-testing correction (Harvey-Liu-Zhu 2016 threshold is t > 3.0, not 2.0). v1's IC t-stats of 2.34 (GP), 2.33 (IVol), and 2.67 (PEAD) are all *below* the HLZ threshold; under proper multiple-testing correction, no signal in v1 is confirmed at conventional significance.

**Data quality and signal construction.** v1's universe screens are loose (price > $5, 20-day ADV > $5M). A specific gap raised in review: stale-price names (long stretches of zero returns) inflate IC artificially via spurious autocorrelation. v1 has no liquidity audit beyond the initial filter.

v2's scope responds to these gaps. v2 does not change the strategy itself — same four signals, same neutrality constraints, same cost model — but applies rigorous statistical inference and tighter data quality throughout, plus adds two new variants (walk-forward IC weighting, regime overlay) and one new signal (Lazy Prices NLP) that respond directly to the methodological feedback.

## Scope: what's in v2

Nine work items, grouped into four phases.

### Phase 1: Foundation (~15 hours, was ~20 hours)

**Item 1 — Tighter liquidity screens.** ~~Add minimum dollar volume threshold, maximum share of zero-volume days, and a stale-price detector.~~ **Audited and not actionable** (audit committed in `scripts/analysis/liquidity_audit.py`). The v1 universe construction (top-1000 by market cap + 20-day ADV ≥ $5M) already eliminates stale-price contamination: maximum zero-volume share across 112,177 audited name-months is 1.67%; 99th percentile of consecutive zero-return runs is 1 day; candidate v2 screens would exclude 0-2 of 112,177 name-months. The original concern applies to small-cap or micro-cap universes, not to the top-1000 here. **Status**: closed; audit script preserved as methodology. Phase 1 reduces from 3 items to 2 items.

**Item 2 — Residual diagnostics framework.** New module `src/axiom_fund/diagnostics/residual_diagnostics.py` exposing six pure functions: Q-Q plot data, residual-vs-fitted plot data, Durbin-Watson statistic, Breusch-Pagan heteroskedasticity test, Cook's distance, leverage. All take raw arrays (residuals, fitted values, design matrix) and return DataFrames or floats; plotting is a downstream concern. No coupling to statsmodels or any specific regression library.

Applied to v1's two regression sites:
- **Residual momentum** (`src/axiom_fund/signals/residual_momentum.py`): cross-sectional regression per month, ~116 regressions. Applicable diagnostics: Q-Q, residual-vs-fitted, Breusch-Pagan, Cook's distance, leverage. Durbin-Watson is *not* applied here because it tests time-series autocorrelation, which is undefined for cross-sectional residuals.
- **Idiosyncratic volatility** (`src/axiom_fund/signals/idiosyncratic_volatility.py`): FF3 trailing-60-day regression per (permno, date), tens of thousands of regressions across the backtest. Durbin-Watson applied across all regressions, reported as an aggregate distribution.

Beta estimation (`_ols_beta` in `src/axiom_fund/portfolio/betas.py`) uses the closed-form Cov(r,m)/Var(m) formula, not a full regression. No residuals exist, so diagnostics do not apply.

**Item 3 — HAC standard errors and bootstrapped CIs.** New module `src/axiom_fund/diagnostics/inference.py` exposing `compute_hac_standard_errors()` (statsmodels wrapper with maxlags parameter) and `compute_bootstrapped_sharpe_ci()` (block bootstrap, configurable block size and number of resamples). Used wherever v1 currently reports asymptotic standard errors.

### Phase 2: Rigorous reporting (~9 hours)

**Item 4 — Deflated Sharpe on variant comparison.** Apply Bailey & López de Prado (2014) to the three v1 variants (3-sig, 4-sig, no-ResMom). Report deflated Sharpe alongside raw Sharpe. Expected magnitude of correction: small (3 variants is low multiplicity). Implementation: new function `compute_deflated_sharpe()` in inference module; new script `scripts/analysis/deflated_sharpe_analysis.py`.

**Item 5 — Quandt-Andrews structural break test.** Apply to the IC time series for each signal. Replaces v1's informal bull/bear regime classification (which split years by sign of cross-signal IC) with a formal break-point detection. Implementation: `statsmodels.api.diagnostic` or custom F-statistic sweep. New script `scripts/analysis/structural_break_analysis.py`.

### Phase 3: Lazy Prices NLP signal (~40 hours)

**Item 6 — Lazy Prices.** Year-over-year cosine distance between consecutive 10-K filings (Risk Factors + MD&A sections), per Cohen-Malloy-Nguyen (2020). Data source: SEC EDGAR (free). Embedding: `sentence-transformers/all-MiniLM-L6-v2` running locally (avoids API costs). Signal interpretation: high distance = "changed filing" = behavioral signal. Implementation in `src/axiom_fund/signals/lazy_prices.py` matching the existing signal-module pattern (pure functions, alignment via `alignment.py`, full backtest integration).

Sub-tasks:
- EDGAR scraping infrastructure with rate limiting and caching
- 10-K text parsing (Risk Factors + MD&A section extraction)
- Local embedding pipeline with batch processing
- Signal alignment to monthly rebalance dates with PIT-discipline
- Integration into the `signals` parameter in `run_historical_backtest`
- Full backtest of 5-signal variant
- Same diagnostic battery as the existing four signals

### Phase 4: New research variants (~25 hours)

**Item 7 — Walk-forward IC weighting variant.** For each rebalance date t, compute trailing-36-month IC of each signal, weight signals proportional to those ICs, build composite. Walk forward. Implementation: new function in `src/axiom_fund/portfolio/composite.py` for IC-weighted aggregation; new backtest driver `scripts/run_full_backtest_ic_weighted.py`.

This is a *research exercise on overfit*, not necessarily a deployment recommendation. The expected result, per DeMiguel-Garlappi-Uppal (2009), is that IC-weighted underperforms equal-weight out-of-sample because the estimation error in IC weights exceeds the gain from being optimal. Reporting honestly whether that's the case in this dataset is the point.

**Item 8 — Simple regime overlay.** Binary regime indicator (default: 12-month trailing S&P 500 return sign, or rolling-60-day S&P z-score). When indicator signals "stress," reduce gross exposure from 1.5× to 1.0×. No other changes (signals stay equal-weighted, neutrality constraints intact). Walk-forward by construction. Implementation: new optional parameter on `run_historical_backtest`; new backtest driver.

Addresses the most concrete v1 gap: the holdout window was regime-friendly, so the v1 OOS test did not stress the strategy. The regime overlay, if it works, would have reduced exposure during 2020 and avoided the -14.5% drawdown.

### Phase 5: Release artifacts (~included in above phases)

- Updated README with v2 results section, v1-vs-v2 comparison table
- New design-time docs: `docs/v2_methodology_audit.md`, updated `docs/limitations.md`
- All Phase 1-4 work shipped together as the v2 release commit

## Scope: what's explicitly NOT in v2

To prevent scope drift, the following are NOT in v2 and will not be added during v2 implementation:

- **Form 4 insider buying signal.** Reassigned to v3.
- **Scale-aware backtest** ($1B NAV impact modeling). Reassigned to v3.
- **Extended backtest window** (pre-2015 GFC inclusion). Reassigned to v3.
- **Higher-frequency signal updating** (weekly or daily PEAD). Out of scope for any planned release.
- **Hidden Markov Model regime detection.** v2 uses a simple observable indicator; HMM is overkill given the data we have and risks overfitting. v3 candidate.
- **Multi-strategy combination** (combining Axiom signals with other strategies). Out of scope.
- **Live trading infrastructure** (broker integration, execution algorithms). Out of scope.
- **IBES analyst-based PEAD.** WRDS subscription does not include IBES. Out of scope for this project.
- **Alternative cost models** beyond v1's commission + Corwin-Schultz spread + sqrt impact + borrow.
- **Cryptocurrency or fixed-income extensions.** Out of scope.

If during implementation any of the above feels tempting, the doc says: not in v2. Defer or drop.

## Implementation order

Strictly sequential, no parallel work. Each phase's outputs feed the next.

1. **Phase 1: Foundation** (~15 hours, was ~20). ~~Liquidity screens →~~ diagnostics framework → HAC/bootstrap utilities. Item 1 closed by audit (see scope section). Order within phase: diagnostics first (applies to existing regressions); HAC/bootstrap second (composable building block applied throughout subsequent phases).

2. **Phase 2: Rigorous reporting** (~9 hours). Deflated Sharpe → Quandt-Andrews. Applied to existing v1 results plus v2's new outputs.

3. **Phase 3: Lazy Prices** (~40 hours). Self-contained signal addition; can begin in parallel to Phase 4 in principle, but for cognitive simplicity executed sequentially.

4. **Phase 4: New research variants** (~25 hours). Walk-forward IC weighting first (~10h), regime overlay second (~15h).

5. **Release**: README update, v1-vs-v2 comparison, final test sweep, push as v2 release commit with annotated tag.

## Acceptance criteria

Each item has explicit acceptance criteria. "Done" means all criteria met, not "mostly done."

**Item 1 (liquidity screens):** Closed. Audit (commit `d69d670`, `scripts/analysis/liquidity_audit.py`) found no contamination in v1's universe; tighter screens would exclude 0-2 of 112,177 name-months. The audit script itself is the methodological contribution.

**Item 2 (residual diagnostics):**
- New module `src/axiom_fund/diagnostics/residual_diagnostics.py`
- 6 pure functions exposed (Q-Q data, residual-vs-fitted data, DW, Breusch-Pagan, Cook's, leverage)
- Function contract: takes raw arrays (residuals, fitted, design matrix) and returns DataFrames or floats; no coupling to statsmodels
- Unit tests added (≥10 tests covering edge cases)
- Applied to residual momentum estimation in v1 (cross-sectional, 5 of 6 diagnostics excluding DW)
- Applied to idiosyncratic volatility in v1 (FF3 trailing-60-day per regression, aggregate DW distribution across all regressions)
- Findings reported in `docs/v2_diagnostics_findings.md` (new file): heteroskedasticity, outliers, autocorrelation if any

**Item 3 (HAC + bootstrap):**
- New module `src/axiom_fund/diagnostics/inference.py`
- `compute_hac_standard_errors()` with configurable maxlags (default: based on Newey-West 1994 rule)
- `compute_bootstrapped_sharpe_ci()` with block bootstrap, configurable n_resamples (default: 10000) and block_length (default: 6 months)
- Unit tests (≥8 tests)
- Applied to v1's IC t-stats: report HAC-corrected t-stats vs raw, document any changes in significance

**Item 4 (deflated Sharpe):**
- Function in inference module
- Applied to (3-sig, 4-sig, no-ResMom) variant comparison
- Report deflated vs raw Sharpe; explain magnitude of correction

**Item 5 (Quandt-Andrews):**
- New script applied to IC time series for each signal
- Identifies break points (if any); reports F-statistic and significance
- Cross-references with v1's informal bull/bear classification (do the formal breaks match?)

**Item 6 (Lazy Prices):**
- EDGAR scraping module with caching and rate limit handling
- 10-K parsing with Risk Factors + MD&A extraction
- Local embedding pipeline (sentence-transformers all-MiniLM-L6-v2)
- Signal aligned to monthly rebalance, PIT-discipline preserved
- Integration into `signals` parameter on `run_historical_backtest`
- Full 5-signal backtest result documented in v2 README
- Single-signal attribution for Lazy Prices alone
- IC analysis for Lazy Prices alongside existing four signals
- Lazy Prices' incremental Sharpe contribution reported honestly (could be zero or negative)

**Item 7 (walk-forward IC):**
- New backtest driver
- Backtest result for IC-weighted variant alongside equal-weighted
- Statistical comparison: is the difference between IC-weighted and equal-weighted statistically significant given bootstrap CIs? Likely answer per DeMiguel-Garlappi-Uppal: no.

**Item 8 (regime overlay):**
- New optional parameter on `run_historical_backtest`
- Walk-forward regime indicator (no look-ahead)
- Backtest result for regime-overlay variant
- Honest reporting: does the overlay improve net Sharpe? Does it reduce max drawdown? What's the cost in normal regimes?

## v1 results: which to rerun, which to leave

When v2 changes things (universe screens, statistical methodology), some v1 numbers will move. Pre-committed treatment:

**Rerun with v2 methodology** (v1 numbers preserved in `docs/v1_results_archive.md`):
- All Sharpe estimates → HAC + bootstrapped CIs
- All IC t-stats → HAC standard errors
- All variant comparisons → with deflated Sharpe annotation
- New backtest variants from Phase 4 (walk-forward IC-weighted, regime overlay)

**Leave as v1 artifacts** (kept in `data/cache/backtest_full_top1000_*/` unchanged):
- Underlying parquet data for v1 backtests (regeneration would lose history; instead create new `_v2/` directories)
- v1 README sections clearly labeled as "v1 (June 2026)"

**Treat as deprecated** (kept on disk, not referenced in v2 docs):
- The informal bull/bear regime classification used in v1's IC analysis (superseded by Quandt-Andrews in v2)

## Pre-committed reporting honesty

After running v2 analyses, I commit to the following reporting principles regardless of outcome:

1. **HAC vs raw t-stats**: if HAC reduces a t-stat below conventional significance (e.g., GP's 2.34 drops below 1.96), report this prominently. Do not hide it.

2. **Multiple testing under HLZ**: report the t>3 Harvey-Liu-Zhu threshold and which signals (if any) clear it. If none clear, state it clearly.

3. **Deflated Sharpe magnitude**: report the deflation amount. If it's small (which is likely with 3 variants), don't oversell its importance.

4. **Liquidity audit transparency**: the audit script and its findings (no contamination in v1 universe) are committed to the repo. v2 does NOT add tighter screens because they would be no-ops; report this honestly rather than adding cosmetic infrastructure.

5. **Walk-forward IC weighting result**: if equal-weighted beats IC-weighted out-of-sample, report it. Do not bury.

6. **Regime overlay result**: if the overlay does not improve performance, report it. Do not abandon and pretend it was never attempted.

7. **Lazy Prices incremental contribution**: report Lazy Prices' marginal Sharpe contribution honestly. If it's small or zero, that's the finding.

8. **No retroactive narrative changes**: existing v1 findings (e.g., "ResMom is a noise diluent") are not reframed in v2 to match new results. If v2 numbers contradict v1 framing, both are reported.

## Timeline and milestones

| Week | Phase | Deliverable |
|---|---|---|
| W1 (Jun 12-18) | Phase 1 — Liquidity audit | audit committed, Item 1 closed as not actionable |
| W2 (Jun 19-25) | Phase 1 — Diagnostics framework | residual_diagnostics.py module + applied to v1 regressions |
| W3 (Jun 26 – Jul 2) | Phase 1 — Inference utilities | inference.py module + HAC applied throughout |
| W4 (Jul 3-9) | Phase 2 | Deflated Sharpe + Quandt-Andrews |
| W5-9 (Jul 10 – Aug 13) | Phase 3 | Lazy Prices end-to-end |
| W10-11 (Aug 14-27) | Phase 4 | Walk-forward IC + regime overlay |
| W12 (Aug 28 – Sep 3) | Buffer / debugging | Catch up on slippage |
| W13 (Sep 4-10) | Release prep | README rewrite, v2 results doc, v1 archive |
| W14 (Sep 11-17) | Release | Final test sweep, push v2 tag |

At 5-7 hours per week, target completion is mid-to-late September 2026. CFA Level II exam on August 16, 2026 is the primary constraint; Phase 3 (Lazy Prices) is scheduled around it.

Weekly check: am I on schedule? If yes, continue. If no, what slipped and why? Adjust the next week's plan accordingly.

## What this document does NOT do

- Does not lock specific numerical results. Cannot pre-commit "HAC t-stats will be X%". Only commits to *reporting them honestly*.
- Does not lock the timeline against unforeseen blockers (WRDS downtime, EDGAR rate limit issues for Lazy Prices, compute failures).
- Does not lock the v3 scope (Form 4, scale-aware, GFC extension are listed but not specified at v2's level of detail).
- Does not extend to architecture decisions about future Axiom releases beyond v3.