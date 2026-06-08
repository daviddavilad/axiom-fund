# Holdout Test Design — Axiom Fund

**Status**: Pre-commitment document. Written and committed *before* running any holdout analysis on the strategy. Once committed, this document is read-only — its purpose is to record what was decided before seeing results, so the analysis cannot be reverse-engineered to fit the outcome.

**Authored**: 2026-06-08
**Author**: David Davila

---

## Motivation

The Axiom Fund strategy was developed iteratively from 2026-01 onward, with backtest results computed on the 2015-01 to 2024-11 period (116 successful periods). The original strategy spec (`docs/strategy_spec.md`) declared:

- Train: 2005–2014
- OOS: 2015–2022
- Holdout: 2023–2025

In practice, the IC analysis, PEAD addition, no-ResMom variant, and exhibits were all evaluated on the full 2015–2024 window. **The 2023-2024 portion of the data has been seen during model development.** A strict claim of "out-of-sample" performance on that window is not defensible.

This document defines two separate analyses to extract what meaningful information remains, despite the contamination.

## Honest acknowledgment

I cannot un-see the 2023-2024 results. Any decision I made about the strategy during 2026-01 to 2026-06 implicitly was informed by them. The framing of "regime-dependent diluent" for ResMom, the inclusion of PEAD, the no-ResMom variant — none of these were chosen on a strict 2015-2022 window.

A truly clean OOS test would require new data the strategy has never seen. CRSP delivers daily data through 2024-12-31; new data is not available yet.

The two analyses below are the best available given that constraint. Neither is pristine.

## Analysis A — Strict OOS on frozen 3-signal pre-2023 strategy

### Specification

- **Strategy frozen as of 2022-12-31**: 3 signals (GP + IVol + ResMom). PEAD, no-ResMom variant, and any subsequent enhancements explicitly excluded by this freeze.
- **Backtest window**: 2023-01-01 to 2024-11-30 (the last period in the data). Expected ~22 successful rebalance dates.
- **All other parameters from `strategy_spec.md`**: top-1000 universe, monthly rebalance, 0.5% position cap, 1.5× gross, dollar/beta/sector neutrality, Ledoit-Wolf covariance.
- **Cost overlay**: same conservative model as `apply_costs_to_full_backtest.py`. Commission 5 bps, Corwin-Schultz spread, sqrt impact, 50 bps annualized borrow.

### Output

- `data/cache/holdout_strict_3sig/backtest_summary.parquet` (gross returns per period)
- `data/cache/holdout_strict_3sig/net_returns.parquet` (with cost overlay)
- `docs/holdout_test_results.md` — analysis report with side-by-side comparison

## Analysis B — Acknowledged-contamination split

### Specification

For each of the three variants (3-signal, 4-signal, no-ResMom), split the existing 116-period backtest into:
- **In-sample**: 2015-01 to 2022-12 (~96 periods)
- **Holdout (seen during development)**: 2023-01 to 2024-11 (~20 periods)

Report all primary metrics on **each window separately**. Note explicitly that the holdout window has been seen.

### Output

- Tables added to `docs/holdout_test_results.md`
- New chart `docs/exhibits/09_in_sample_vs_holdout.png` showing cumulative returns split at 2022-12-31, marking the boundary visually

## Pre-committed metrics

These are the metrics that determine "success" — committed in writing before any analysis is run.

| Metric | Definition | Decision threshold |
|---|---|---|
| **Primary: Net Sharpe (conservative cost)** | Annualized net return / annualized net vol | ≥ 0.10 = strategy survives realistic execution; below → in-sample overfit signal |
| **Primary: Gross Sharpe** | Annualized gross return / annualized gross vol | ≥ 0.40 in holdout = signal is real but degraded; ≥ 0.60 = no meaningful degradation |
| **Secondary: Max drawdown** | Worst peak-to-trough cum return | If max DD in holdout > 1.5× max DD in-sample, regime detection / risk overlay is required for deployment |
| **Secondary: Hit rate** | Fraction of periods with positive return | Drop > 5pp from in-sample to holdout = signal decay |
| **Secondary: Gross-vs-net spread** | Ann gross − ann net | Should be similar to in-sample (~4.5%); much wider → execution model is mis-calibrated |

## Pre-committed failure protocol

If the holdout test produces results inconsistent with the in-sample claims in the README, I commit to the following responses:

| Outcome | Response |
|---|---|
| **Net Sharpe in holdout < 0.0** | Strategy fails OOS. Document in README. Do not add signals to "recover" the gap. State that the in-sample 0.18-0.48 net Sharpe was overstated. |
| **Gross Sharpe in holdout < 0.40** | Document signal decay. Add the holdout result to the README findings section. Investigate which signal(s) drove the decay using single-signal attribution on the holdout window. |
| **Max DD in holdout >> in-sample DD** | Document the tail-risk amplification. Add explicit "max drawdown was worse out-of-sample" note. Do not retroactively reframe ResMom's role. |
| **All metrics survive** | Report honestly. Update README to note that performance was confirmed on the partially-contaminated 2023-2024 window. Acknowledge contamination explicitly. |
| **Strict OOS shows degraded performance but acknowledged-contamination split survives** | This is the most likely outcome. Document both. Note that the strategy as evaluated since 2026-01 has had access to 2023-2024 data, and that the strict OOS (which excludes PEAD and the no-ResMom variant) is the more conservative test. |

## What this analysis cannot tell us

- Whether the strategy works in regimes outside 2015-2024 (no 1990s, no 2008 crisis, no 1970s inflation)
- Whether the strategy works at scale ($1B NAV vs NAV=1)
- Whether IBES-based PEAD (which we don't have access to) would change results
- Whether published-factor decay has accelerated past 2024 (no data)

These limitations were known and documented in `docs/limitations.md` before this test was designed.

## What I commit to NOT doing

After seeing the holdout results, I commit to **not**:
1. Adding new signals specifically to close any gap between in-sample and holdout
2. Reframing existing findings retroactively (e.g., "ResMom was actually a regime hedge all along") to make the holdout look better
3. Cherry-picking the metric that makes the strategy look best — all primary metrics will be reported together
4. Adjusting cost model parameters to make net Sharpe look better in holdout

## Implementation plan

1. Implement Analysis B (acknowledged-contamination split) first — uses existing data, faster (~2 hours)
2. Implement Analysis A (strict OOS rerun) second — requires building a frozen 3-signal driver for 2023-2024 (~3 hours)
3. Write `docs/holdout_test_results.md` reporting both analyses
4. Update README with the holdout result, framed honestly

Will be executed in 1-2 future sessions.