# Axiom Fund — Strategy Specification v1.0

**Fund:** Axiom Fund
**Manager:** Arcadia
**Document:** Strategy specification, v1.0
**Author:** David Davila
**Status:** Locked — do not modify without dated amendment below

---

## 1. Mandate

Axiom Fund is a systematic, market-neutral U.S. equity long/short strategy that seeks to
generate risk-adjusted returns uncorrelated with broad equity markets. The fund combines
three academically-grounded cross-sectional signals through a constrained portfolio
optimizer with explicit risk neutralization.

**This document describes a research prototype, not a live fund.** All performance
figures derived from this project are historical simulation results subject to the
limitations enumerated in `limitations.md`.

### Target profile

| Metric                          | Target                       |
|---------------------------------|------------------------------|
| Annualized excess return        | 4–6% over cash               |
| Annualized volatility           | 6–8%                         |
| Net Sharpe ratio (after costs)  | 0.7–1.0                      |
| Beta to S&P 500                 | −0.10 to +0.10               |
| Max drawdown tolerance          | 12% (kill-switch at 12%)     |
| Annualized turnover             | 300–500%                     |
| Correlation to HFRI EMN         | < 0.6                        |

A Sharpe target above 1.0 for a multi-factor U.S. equity market-neutral strategy is not
realistic for a student-built prototype. Claims otherwise indicate either methodological
errors or unrealistic cost assumptions.

## 2. Edge hypothesis

Axiom does not claim a novel economic edge. The strategy captures residual return from
well-documented cross-sectional anomalies after neutralizing exposure to standard risk
factors. Claimed sources of incremental value are:

1. **Implementation discipline** — realistic cost modeling, filing-date lag, locked
   out-of-sample evaluation
2. **Construction quality** — residual momentum rather than raw momentum; gross
   profitability rather than ROE; idiosyncratic rather than total volatility
3. **Factor neutralization** — explicit neutralization to Fama-French 5-factor plus
   momentum, so reported returns are not repackaged factor beta

This is a deliberately modest edge claim. See `limitations.md` section 3 for discussion
of why stronger claims would not be credible.

## 3. Investable universe

**Primary universe:** Top 1,000 U.S. common stocks by CRSP market capitalization,
reconstituted monthly on the last trading day of each calendar month.

**Membership construction (rules-based):** On each reconstitution date, rank all
eligible securities by CRSP market capitalization (shares outstanding × closing
price) as of the rank date. Select the top 1,000. This universe replaces a
committee-selected index (e.g., S&P 500) with a transparent, fully reproducible
rule.

**Eligibility filters (applied at each reconstitution):**
- Common stock only (CRSP share codes 10, 11)
- Price > $5 at rank date
- Trailing 20-day ADV > $5M
- Listed on NYSE, NASDAQ, or AMEX (CRSP exchange codes 1, 2, 3)
- Excluded: REITs (SIC 6798), ADRs, limited partnerships, closed-end funds

**Rationale for rules-based construction:** This universe is defined by a
transparent rule rather than index-provider committee decisions. It is fully
reproducible from CRSP stock files alone, does not depend on third-party
membership data, and aligns with the construction approach used by systematic
institutional investors (e.g., factor-based funds at AQR, Dimensional). It
avoids the dependence on S&P Dow Jones Indices' committee decisions that would
introduce a discretionary element inconsistent with a purely systematic strategy.

**Comparison to S&P 500 + S&P MidCap 400:** The resulting universe overlaps
substantially with S&P 500 + MidCap 400 (~80% name overlap by market cap) but
differs in the tails. The rules-based universe includes some names the S&P
committee has not added and excludes some the committee retains for continuity.
For our purposes, this difference is immaterial — the strategy is cross-sectional
and relies on relative ranking within a large-cap universe, not on benchmarking
to the S&P 500.

**Methodological note:** An earlier draft of this specification proposed S&P 500 +
S&P MidCap 400 as the universe. We revised to a rules-based construction after
determining that our WRDS subscription does not include access to `crsp_a_indexes`
(the CRSP S&P index constituent files). Rather than introduce dependency on
external index-membership data of uncertain quality, we adopted the rules-based
approach, which is methodologically preferable for a systematic strategy. See
`limitations.md` section 3 for further detail.

## 4. Rebalance frequency

Monthly, on the last trading day of each calendar month. Signals computed using data
available as of T-1 close; trades assumed executed at T+1 VWAP. This avoids all look-
ahead including the common "execute at close on signal date" error.

## 5. Alpha signals

Three signals, equal-weighted in the composite score at the outset. Composite weighting
is a locked parameter; it will not be tuned based on in-sample performance.

### Signal 1 — Gross Profitability (Quality)
- **Construction:** (Revenue − COGS) / Total Assets, from Compustat
- **Source paper:** Novy-Marx (2013), *JFE*
- **Vintage handling:** Fundamentals available T + 45 calendar days after fiscal
  quarter end (conservative filing-date lag)
- **Cross-sectional treatment:** Winsorize at 1%/99%, z-score within universe,
  neutralize to GICS sector
- **Expected decay note:** Documented 2005–2015; performance post-2015 weaker. See
  `limitations.md` section 3.1.

### Signal 2 — Idiosyncratic Volatility (Low-Vol)
- **Construction:** Residual standard deviation from a 60-trading-day rolling regression
  of daily excess returns on the Fama-French 3-factor model. Lower IVOL = higher score
  (inverted).
- **Source paper:** Ang, Hodrick, Xing, Zhang (2006), *JF*
- **Cross-sectional treatment:** Winsorize at 1%/99%, z-score within universe,
  neutralize to GICS sector
- **Regression window:** 60 trading days, minimum 40 valid observations
- **Expected decay note:** Low-vol has underperformed 2016–2024 relative to historical
  Sharpe. Explicitly tested in robustness. See `limitations.md` section 3.2.

### Signal 3 — Residual Momentum 12-1 (Industry + Size Neutralized)
- **Construction:** Compute 12-1 momentum (cumulative return from month t−12 to t−1,
  skipping the most recent month). At each rebalance, regress cross-sectionally against
  (a) log market cap and (b) GICS industry group dummies. Use residuals as the signal.
- **Source paper:** Blitz, Huij, Martens (2011), *JEF* — "Residual Momentum"
- **Rationale over raw 12-1 momentum:** Residual momentum has higher risk-adjusted
  returns, substantially lower drawdowns during momentum crashes (2009, 2016, 2020),
  and is not repackaged size or industry beta.
- **Cross-sectional treatment:** Winsorize residuals at 1%/99%, z-score within universe
- **Differentiation from CQA 2025 project:** The CQA team used raw 12-1 momentum in a
  competition context. Residual momentum with factor-neutral construction is a
  methodologically distinct signal and portfolio construction.

### Composite
$$\text{score}_{i,t} = \frac{1}{3}\left(z^{\text{prof}}_{i,t} + z^{\text{ivol}}_{i,t} + z^{\text{resmom}}_{i,t}\right)$$

Equal weighting is a locked, pre-committed choice. Optimizing composite weights on
three signals over limited data is a known overfitting failure mode.

## 6. Portfolio construction

Daily: no. Monthly: yes — portfolio rebalanced at each monthly rebalance date via
constrained mean-variance optimization in `cvxpy`.

$$\max_w \; \alpha^T w - \frac{\lambda}{2} w^T \Sigma w - \tau \cdot \|w - w_{\text{prev}}\|_1$$

### Constraints

| Constraint            | Specification                                            |
|-----------------------|----------------------------------------------------------|
| Dollar neutrality     | $\mathbf{1}^T w = 0$                                     |
| Beta neutrality       | $\beta^T w = 0$ (60-day rolling beta to S&P 500)         |
| Sector neutrality     | $\|B_s^T w\| \le 0.02$ for each GICS Level 1 sector      |
| Factor neutrality     | $\|F_k^T w\| \le 0.05$ for FF5 + Momentum factors        |
| Position size         | $\|w_i\| \le 0.015$ (1.5% per name)                      |
| Gross leverage        | $\sum \|w_i\| \le 1.5$                                   |
| Liquidity             | $\|w_i\| \cdot \text{AUM} \le 0.05 \cdot \text{ADV}_i$   |

### Covariance estimation
Ledoit-Wolf shrinkage toward constant-correlation target. Estimation window:
252 trading days. Shrinkage intensity estimated per-period (not locked).

### Transaction cost penalty
$\tau = 0.002$ (20bps L1 penalty on trading). Calibrated to produce realistic turnover;
not tuned on backtest performance.

### Solver
`cvxpy` with ECOS or OSQP backend. Problem is a QP with linear constraints; should
solve in seconds per rebalance.

### Baseline benchmark
At every reporting checkpoint, report the constrained optimizer's results side-by-side
with a simple equal-weight top-decile-long / bottom-decile-short portfolio on the same
signal. If the optimizer does not beat equal-weight on net Sharpe after costs, the
optimizer is adding noise and the report should acknowledge this.

## 7. Transaction cost model

Every trade incurs:
- **Commission:** 5 bps per side (realistic for institutional execution)
- **Half-spread:** Corwin-Schultz estimate from daily high/low, floor 2 bps
- **Market impact:** $\kappa \sigma_i \sqrt{|\Delta w_i| \cdot \text{AUM} / \text{ADV}_i}$ with $\kappa = 0.1$
- **Short borrow:** 50 bps/year flat on short book, accrued daily

### Cost sensitivity (mandatory in final report)
Strategy re-run at 0.5×, 1×, 2×, 3× base cost assumptions. Results reported at all four.

## 8. Risk management

### Ex-ante (computed at each rebalance, before trading)
- Factor exposures: beta, FF5, momentum, quality, low-vol
- Sector exposures: gross and net by GICS Level 1
- Concentration: top-10 names, HHI
- Ex-ante volatility forecast from $\sqrt{w^T \Sigma w}$; target 6–8% annualized
- Ex-ante tracking error to zero (dollar-neutral target)

### Ex-post (computed monthly during backtest)
- Realized vs. ex-ante volatility ratio
- Rolling 60-day Sharpe, drawdown, beta
- Factor attribution (Brinson-style: factor vs. specific return)
- Regime attribution (bull, bear, high-vol, low-vol)

### Pre-committed kill-switches
- If trailing 60-day drawdown > 8%: reduce gross by 50% next rebalance
- If trailing 60-day drawdown > 12%: halt new risk, liquidate existing positions over 5 days
- If realized 60-day vol > 1.5× target: reduce gross by 33% next rebalance

These are tested in the backtest. They are pre-committed, not tuned.

## 9. Data

### Primary sources (WRDS)
| Dataset        | Purpose                                              |
|----------------|------------------------------------------------------|
| CRSP Daily     | Prices, returns, volume, shares outstanding, delisting returns |
| Compustat Quarterly (point-in-time snapshot) | Fundamentals for profitability signal |
| CCM Link Table | PERMNO ↔ GVKEY linking                               |
| CRSP Indices   | S&P 500 / MidCap 400 point-in-time membership        |

### Secondary sources (free)
- **FRED:** Risk-free rate (1-month T-Bill), macro controls
- **Ken French Data Library:** FF5 + momentum factor returns, industry definitions

### Parallel pipeline
- **SEC EDGAR XBRL:** Independent fundamentals extraction for robustness check against
  Compustat. Scope: v1.5 side project; not required for core results. See
  `limitations.md` section 5 for kill criteria.

### Storage
Parquet files partitioned by date, queried via DuckDB. No CSVs in production code paths.

### CRSP–Compustat linking (CUSIP-based)

The CRSP/Compustat Merged (CCM) link table, which is the academic standard for
joining CRSP PERMNO to Compustat GVKEY, is not accessible in our WRDS subscription
tier. In its place, we link CRSP to Compustat via 8-digit CUSIP (comparing CRSP's
`ncusip` or `cusip` to Compustat's `cusip`, matching on the first 8 characters to
avoid share-class issues). On each reconstitution date, each CRSP PERMNO is
matched to the Compustat record whose CUSIP matches and whose fundamentals are
the most recent available as of (rank date − 45 calendar days).

**Expected match rate:** Based on academic literature using similar approaches,
we expect to match 85–92% of CRSP names to Compustat fundamentals. Unmatched
names are dropped from the fundamentals-dependent signals (Gross Profitability)
but retained in price-only signals (Idiosyncratic Volatility, Residual Momentum).
The actual match rate will be reported in the white paper.

**Pre-committed exit criterion:** If the realized match rate is below 85%, the
project will pause for an explicit re-evaluation. Options at that point include
purchasing Sharadar (Nasdaq Data Link SEP + SF1, ~$100/month) for clean
point-in-time fundamentals with institutional linking, or narrowing the
universe to large-caps where CUSIP matching is most reliable. This exit
criterion is pre-committed before the matching code is written.

## 10. Backtest methodology

### Windowing (locked at project start)
| Period            | Dates            | Use                                        |
|-------------------|------------------|--------------------------------------------|
| Training          | 2005-01 to 2014-12 | Methodology development, parameter choices |
| Out-of-sample     | 2015-01 to 2022-12 | Single evaluation allowed (for paper)     |
| Strict holdout    | 2023-01 to 2025-12 | Untouched until final report run          |

The strict holdout is untouchable. Accessing it before the final locked run
invalidates it permanently. This discipline is the primary defense against p-hacking.

### Execution assumptions
- Signal computed using data through T-1 close
- Trade executed at T+1 VWAP (approximated as average of T+1 open and close)
- No fills for names with zero volume on T+1
- No fills for names that are halted, delisted, or subject to corporate actions on T+1

### Accounting
- Daily P&L: $w_{t-1}^T r_t - \text{costs}_t$
- Long book financed at SOFR + 50 bps
- Short rebate at SOFR − 50 bps
- Idle cash at SOFR

## 11. Evaluation metrics

All reported with Newey-West standard errors and, where applicable, 95% confidence
intervals. Sharpe ratios reported with CI and deflated Sharpe ratio (Bailey–López de
Prado 2014).

| Category         | Metrics                                                     |
|------------------|-------------------------------------------------------------|
| Return           | CAGR, annualized excess return, alpha vs. FF5+Mom           |
| Risk             | Vol, max DD, DD duration, downside deviation, VaR 95/99     |
| Risk-adjusted    | Sharpe + 95% CI, Sortino, Calmar, deflated Sharpe           |
| Portfolio        | Gross, net, turnover (annualized), holding period, hit rate |
| Attribution      | Factor vs. specific return; per-signal contribution         |
| Costs            | Gross → commission → spread → impact → borrow → net          |

## 12. Deliverables

1. GitHub repository: `axiom-fund` — typed Python, tests, CI, reproducible from README
2. Strategy white paper (PDF, 25–30 pages) — methodology, results, limitations
3. Tear sheet PDF — full performance summary, generated from code
4. README — project overview, how to reproduce, key results summary
5. `limitations.md` — living document, updated as limitations are discovered
6. Loom walkthrough (5–10 min) — recorded walkthrough for recruiting use
7. EDGAR parser (optional, v1.5) — standalone package for XBRL fundamentals extraction

## 13. Timeline (20 weeks, gated)

| Weeks  | Phase                         | Ship-gate                                    |
|--------|-------------------------------|----------------------------------------------|
| 1–3    | WRDS setup, data pipeline, universe | Clean daily return + monthly fundamentals panels 2005–2025 |
| 4–6    | Signal construction           | Three z-scored signal panels with sensible cross-sections |
| 7–10   | Baseline backtest + cost model | Equity curves for each signal + composite, decile sort, baseline version |
| 11–14  | Optimizer + factor neutralization | Optimizer beats equal-weight baseline net of costs |
| 15–17  | Robustness, regime tests, attribution | All results reconcile; anomalies investigated |
| 18–19  | White paper                   | Draft complete, limitations fully documented |
| 20     | Polish, tear sheet, Loom, final commit | Shippable artifact                       |

### Scope exit criteria
- If EDGAR parser is not substantially complete by end of week 10, it is dropped from v1
  and deferred to a post-submission side project.
- If any week's gate is not met by +1 week, the project pauses for re-scoping. No
  silent scope expansion.

## 14. Locked parameter registry

The following parameters are **locked** as of this document's commit and will not be
changed based on in-sample performance:

- Universe: Top 1,000 U.S. common stocks by CRSP market cap, reconstituted monthly
- Universe filters: common stock (share codes 10/11), price > $5, 20-day ADV > $5M, exchanges NYSE/NASDAQ/AMEX, exclude REITs/ADRs/LPs/CEFs
- CRSP-Compustat linking: 8-digit CUSIP merge; exit criterion <85% match rate
- Three signals: Gross Profitability, Idiosyncratic Volatility, Residual Momentum 12-1
- Composite weighting: equal (1/3 each)
- Winsorization: 1% / 99%
- IVOL regression window: 60 trading days
- Fundamentals lag: 45 calendar days
- Covariance: Ledoit-Wolf shrinkage, 252-day window
- Gross cap: 1.5x
- Position cap: 1.5%
- Rebalance: monthly, last trading day
- Train / OOS / holdout windows as specified in §10
- Cost model coefficients as specified in §7

Parameters not in this registry (e.g., turnover penalty coefficient, exact sector
neutrality tolerance) are free and may be adjusted. Any locked parameter change
requires a dated amendment below.

## 15. Amendments

### Amendment 1 — 2026-04-20
- Universe changed from S&P 500 + S&P MidCap 400 to top 1,000 by CRSP market cap
- Added CUSIP-based CRSP-Compustat linking methodology
- Added match-rate exit criterion (<85% triggers Sharadar re-evaluation)
- Root cause: WRDS subscription tier does not include crsp_a_indexes or crsp_a_ccm
