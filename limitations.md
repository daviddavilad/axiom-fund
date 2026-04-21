# Axiom Fund — Known Limitations

**Author:** David Davila
**Status:** Pre-committed before any code is written. This document is expected to grow.
**Philosophy:** Every limitation identified is worth more than one extra backtest run.

This document enumerates known and anticipated limitations of the Axiom Fund prototype.
It is committed before any code is written in order to pre-commit the project's honest
constraints — not to generate them after-the-fact to explain disappointing results.

---

## 1. This is a backtest, not evidence of alpha

No backtest, regardless of rigor, is evidence that a strategy will generate future
returns. Even a flawless historical simulation over 20+ years can reflect regime-
specific patterns that do not persist. The correct interpretation of any positive result
from this project is: "the methodology and infrastructure are sound, and under the
specific historical data and assumptions used, the strategy would have produced these
results." No stronger claim is supported.

## 2. Universe construction and subscription gap

Universe is restricted to the top 1,000 U.S. common stocks by CRSP market
capitalization (see `strategy_spec.md` §3). This reflects two deliberate
choices:

1. **Rules-based over committee-selected.** Using a transparent market-cap
   rule rather than S&P 500/400 committee decisions is methodologically
   preferable for a systematic strategy and ensures full reproducibility.

2. **Forced by subscription tier.** UNM's WRDS subscription does not include
   `crsp_a_indexes` (S&P index constituents) or `crsp_a_ccm` (CRSP/Compustat
   Merged link table). The rules-based construction is the methodologically
   clean response to this constraint.

Residual limitations:
- Small-caps (below top 1,000) excluded; short-side borrow cannot be
  realistically modeled without institutional prime broker data
- Micro-caps excluded entirely
- International equities excluded entirely
- ADRs, REITs, LPs, CEFs excluded per filter rules
- The resulting universe is not directly benchmarkable to S&P 500 or other
  index returns without careful adjustment

Results cannot be generalized to small-cap, international, or non-rules-based
strategies.

## 3. Known signal decay risks

### 3.1 Gross profitability
Novy-Marx (2013) documented the profitability premium using data through ~2010. McLean
& Pontiff (2016) find ~30% post-publication decay in published anomalies on average.
The profitability premium specifically has weakened but not disappeared post-2013.
The backtest will likely show stronger performance 2005–2014 than 2015–2025.

### 3.2 Idiosyncratic volatility
The low-vol anomaly has underperformed 2016–2024 as low-vol factor ETFs (USMV, SPLV)
attracted substantial flows and the trade crowded. Performance in the 2023–2025 strict
holdout period is a particular concern.

### 3.3 Residual momentum
Momentum strategies experienced severe crashes in 2009, 2016, and 2020 ("momentum
crashes"). Residual momentum construction mitigates but does not eliminate this risk.
The 2015–2022 OOS period contains at least two regimes (2016, 2020) where momentum
strategies underperformed sharply.

## 4. Transaction cost model limitations

- **Market impact coefficient κ = 0.1** is borrowed from Almgren et al. (2005). It is
  not calibrated to this strategy's actual trading and is almost certainly wrong for
  some names. Sensitivity analysis at 0.5×, 1×, 2×, 3× is mandatory.
- **Spread estimation** uses Corwin-Schultz from daily high/low. This is reasonable for
  liquid names but systematically biased for stocks with intra-day volatility patterns
  different from the model's assumptions.
- **Commissions at 5 bps/side** are institutional-realistic but not achievable by retail.
- **No accounting for adverse selection** in the cost model. Some trading strategies
  (especially momentum) may have negative alpha decay that looks like higher costs.

## 5. Short-side modeling limitations

- **Borrow cost assumed flat at 50 bps/year.** Real borrow rates range from 25 bps
  (easy-to-borrow) to 2000+ bps (hard-to-borrow) and vary through time. Crowded shorts
  periodically see dramatic rate spikes (2021 meme stocks, 2008 financials).
- **No modeling of recall risk.** Real short positions can be recalled by the lender,
  forcing involuntary covering. This is the mechanism behind short squeezes.
- **No modeling of short sale restrictions.** The 2008 short ban on financial stocks and
  various regional bans are not modeled.
- **Sensitivity analysis** at 100 bps and 200 bps borrow will be reported but does not
  capture the full tail risk of short positions.

## 5.5 CRSP–Compustat linking via CUSIP

The CRSP/Compustat Merged (CCM) link table is the academic standard for joining
CRSP PERMNO to Compustat GVKEY. It is not accessible in our WRDS subscription
tier. We replace it with an 8-digit CUSIP-based merge.

Known risks of CUSIP-based linking:
- **Share class ambiguity.** A company with multiple share classes (e.g.,
  Google's GOOG/GOOGL, Berkshire's BRK.A/BRK.B) shares the first 6 digits of
  CUSIP but differs in the last 3. Using 8-digit matching handles most cases
  but may cause occasional misattribution.
- **Historical CUSIP changes.** CUSIPs occasionally change (corporate actions,
  reorganizations). CRSP's `ncusip` field captures some changes but not all.
- **Coverage gaps.** Some CRSP names have no CUSIP in CRSP; some Compustat
  names have no CUSIP in Compustat. These are dropped from fundamentals-
  dependent signals.

Expected match rate: 85–92% based on academic literature. Actual match rate
will be reported in the white paper.

**Pre-committed exit criterion:** If the realized match rate is below 85%, the
project will pause for explicit re-evaluation, with likely options being:
(a) purchase Nasdaq Data Link Sharadar SEP + SF1 (~$100/month) for three
months to obtain clean linked fundamentals, or (b) narrow the universe to
large-caps where CUSIP matching is most reliable and accept the reduced
breadth. This exit criterion is pre-committed before the matching code is
written.

Unmatched names are dropped from the Gross Profitability signal but retained
in the Idiosyncratic Volatility and Residual Momentum signals. The composite
score for unmatched names uses only the two price-based signals, which
introduces a small structural bias relative to matched names. This bias will
be quantified in the robustness section of the white paper.

## 6. EDGAR XBRL parser limitations (v1.5 parallel work)

The EDGAR parser is being built in parallel to the core project as a robustness artifact.
Known limitations:
- XBRL coverage begins 2009 (2011 for smaller filers); pre-2009 requires HTML parsing
- Taxonomy changes over time create classification ambiguities
- Filers use custom extensions inconsistently
- Restatements and amendments require amendment-date tracking

**Kill criterion:** If the EDGAR parser is not substantially complete and producing
values within 2% of Compustat on a random sample of 50 firm-quarters by end of week 10,
it is dropped from v1 and deferred to a post-submission project.

## 7. Backtesting methodology limitations

### 7.1 Multiple testing
Even with a locked parameter registry, many small decisions (winsorization thresholds,
regression windows, neutralization choices) will be made during the project. Each is a
silent hypothesis test. Reported Sharpe should be interpreted with this in mind;
deflated Sharpe (Bailey–López de Prado 2014) will be reported as a partial correction.

### 7.2 Regime overfit in training
The 2005–2014 training window contains the GFC (2008) and the immediate post-crisis
low-rate era. Parameters chosen to work well in this window may not transfer to the
zero-rate, pandemic-disrupted, high-rate 2015–2025 period.

### 7.3 Capacity not modeled
The backtest assumes zero market impact from the fund's own trading. At any meaningful
AUM, this is false. A capacity estimate will be reported as an order-of-magnitude guess,
not a calibrated number.

### 7.4 Survivorship bias (mitigated, not eliminated)
CRSP delisting returns eliminate the primary form of survivorship bias. Residual forms
(mid-period bankruptcies handled imperfectly, corporate action edge cases) likely
remain at a small level.

## 8. Model-specific limitations

### 8.1 Covariance estimation
Ledoit-Wolf shrinkage with 252-day window will produce risk forecasts that are 20–40%
off from realized during regime shifts. Realized vs. ex-ante tracking error will be
reported.

### 8.2 Beta estimation
60-day rolling beta is noisy and lags regime changes. Names with short listing history
or illiquid returns will have unreliable betas and may be excluded at the rebalance.

### 8.3 Factor model for neutralization
FF5 + momentum is the standard but incomplete. Missing exposures (e.g., to short-term
reversal, earnings quality, investment) may show up as residual alpha that is actually
factor beta. The attribution will probe this.

## 9. Live trading limitations not addressed

This prototype does not address:
- Order routing and execution (smart order routing, dark pools, VWAP algorithms)
- Real-time data latency
- Counterparty and prime broker selection
- Margin and financing details
- Tax optimization
- Fund administration, compliance, regulatory reporting

A real fund requires all of these. A prototype demonstrating research capability does
not. This is a feature, not an omission — scope discipline is part of what is being
demonstrated.

## 10. Dependence on author's judgment

The author (David Davila) is a 20-year-old undergraduate with ~2 years of serious quant
finance study. Decisions made during this project reflect that level of experience.
Specific areas where senior quant researcher judgment would likely differ:
- Covariance model choice (a PM with 10 years of experience would likely use a
  commercial factor risk model like Axioma or Barra)
- Signal selection (a senior researcher would likely have proprietary signals beyond
  the academic canon)
- Cost model calibration (a senior researcher would have access to actual execution
  data to calibrate κ)
- Regime detection (a senior researcher might include explicit regime-conditional
  weighting not included here)

These are not flaws to hide. They are the honest boundary of a student project.

## 11. Amendments

*(update as limitations are discovered during the build)*
