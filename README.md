# Axiom Fund

**Systematic U.S. equity market-neutral long/short strategy.** A research portfolio engine and backtest framework written from scratch in Python.

## Current results

Top-1000 U.S. equity universe, 2015-01 → 2024-11, monthly rebalance, 116 successful periods.

|                     | 3-signal     | 4-signal (+PEAD) | No-ResMom (GP+IVol+PEAD) |
|---------------------|--------------|------------------|--------------------------|
| Gross Sharpe        | 0.79         | 0.78             | **0.82**                 |
| Net Sharpe (conservative) | 0.18    | similar          | -                        |
| Net Sharpe (50% execution improvement) | 0.48 | similar | -                  |
| Cumulative gross    | +68.3%       | +65.9%           | +64.8%                   |
| Cumulative net      | +10.7%       | similar          | -                        |
| Max drawdown (gross)| -14.5%       | -15.1%           | **-20.5%**               |
| Hit rate (gross)    | 57.8%        | 58.6%            | **64.7%**                |
| Annualized gross vol| 7.18%        | 7.05%            | **6.61%**                |

**The gross Sharpe of ~0.78 is a research artifact.** The net Sharpe range of 0.18–0.48 reflects realistic transaction costs (commission + Corwin-Schultz half-spread + sqrt impact + borrow). The lower bound assumes retail-grade execution (full half-spread per trade); the upper bound reflects a 50% execution improvement consistent with institutional VWAP/POV/midpoint algorithms.

![Cumulative returns 2015-2024](./docs/exhibits/01_cumulative_returns.png)

![Drawdown over time](./docs/exhibits/02_drawdown.png)

The 3-signal and 4-signal Sharpes are statistically identical. Signal-correlation and IC analysis (`scripts/analysis/ic_analysis.py`) explain why: the original three signals are nearly orthogonal and already exploit their available diversification, while PEAD overlaps modestly with GP and ResMom (correlations ~0.2).

The **no-ResMom variant** drops Residual Momentum entirely (which the IC analysis below shows has zero predictive power in this window). The variant achieves a higher Sharpe (0.82 vs 0.79) and hit rate (65% vs 58%) on lower volatility, but its max drawdown widens by 6 percentage points — the year-by-year return spread grows from 30pp to 48pp, with the 2020 stress amplified (-17.6% vs -8.5%) and the 2021 recovery sharper (+30.1% vs +16.8%). ResMom acts as a **noise diluent**: by absorbing 25% of the composite weight at near-zero IC, it dampens *both* signal and noise from the other three signals. See [Findings](#findings) below.

![Three-variant cumulative returns](./docs/exhibits/08_variant_comparison.png)

This repository contains a research prototype. **It is not a live fund**, and no figures produced by this code constitute evidence of alpha. See [`docs/limitations.md`](./docs/limitations.md) for the full pre-committed enumeration of known methodological limitations.

## Strategy

- **Universe**: top 1,000 U.S. common stocks by CRSP market cap, monthly reconstitution. Filters: share codes 10/11, price > $5, 20-day ADV > $5M, NYSE/NASDAQ/AMEX. Overlaps ~80% with S&P 500 + MidCap 400 by name, but is fully reproducible from CRSP rather than committee-selected.
- **Signals (equal-weight z-scores)**: Gross Profitability (Novy-Marx 2013), Idiosyncratic Volatility (Ang-Hodrick-Xing-Zhang 2006), Residual Momentum 12-1 (Blitz-Huij-Martens 2011), and PEAD time-series SUE (Bernard-Thomas 1989).
- **Portfolio**: dollar/beta/sector-neutral convex MVO (cvxpy), Ledoit-Wolf shrunk covariance, 1.5× gross leverage cap, 0.5% per-name position cap.
- **Costs**: 5 bps commission, Corwin-Schultz rolling spread (Corwin-Schultz 2012), square-root market impact (Almgren et al. 2005) at κ=0.1, 50 bps annualized borrow.
- **Windows**: train 2005–2014, OOS 2015–2022, holdout 2023–2025. CRSP daily data ends 2024-12-31, so periods after that are skipped.

Full specification in [`docs/strategy_spec.md`](./docs/strategy_spec.md). Signal-layer design and PIT considerations in [`docs/signal_design.md`](./docs/signal_design.md).

## Methodology

Particular care was taken with point-in-time discipline. The strategy uses Compustat `rdq` (earnings announcement date) rather than fiscal period end for fundamentals-based signals; merges CRSP delisting returns per Shumway (1997); reconstitutes the universe monthly to avoid survivorship bias; and uses Compustat's `ajexq` to split-adjust EPS before computing PEAD seasonal differences. The full pre-implementation methodology audit, including five bias categories examined and the resolution for each, is in [`docs/methodology_audit.md`](./docs/methodology_audit.md).

## Findings

A 116-period IC analysis across the four signals (Spearman rank correlation vs 21-day forward returns):

| Signal    | Mean IC | Std IC | t-stat | p-value | Hit rate |
|-----------|---------|--------|--------|---------|----------|
| z_gp      | 0.0235  | 0.108  | 2.34   | 0.021   | 62.1%    |
| z_ivol    | 0.0334  | 0.154  | 2.33   | 0.021   | 54.3%    |
| z_pead    | 0.0209  | 0.084  | 2.67   | 0.0086  | 63.8%    |
| z_resmom  | 0.0029  | 0.119  | 0.26   | 0.797   | 52.6%    |

![IC per signal with 95% CI](./docs/exhibits/04_ic_per_signal.png)

Three of the four signals are statistically positive at the 5% level. **PEAD has the smallest mean IC but the strongest t-stat** — its low magnitude is offset by remarkable consistency (lowest std IC, highest hit rate). Residual momentum has effectively zero predictive power over this 9.8-year window (t=0.26, p=0.80), consistent with the post-publication factor decay documented in McLean-Pontiff (2016) and Asness-Frazzini (2013).

Splitting periods into bull regimes (92 periods) versus bear regimes (24 periods) sharpens the picture further: three of the four signals exhibit positive IC in bull markets and zero-or-negative IC in bear markets. IVol exhibits the classic low-vol anomaly cyclicality (t = +3.83 in bull, -2.47 in bear). PEAD shows a milder version of the same pattern. **The gross Sharpe of 0.78 is primarily delivered by bull-regime exposure**; a more sophisticated production system would require regime detection.

![IC by regime: bull vs bear](./docs/exhibits/05_ic_bull_vs_bear.png)

Average pairwise correlation across periods:

|          | z_gp   | z_ivol | z_pead | z_resmom |
|----------|--------|--------|--------|----------|
| z_gp     | 1.000  | -0.033 | 0.215  | 0.049    |
| z_ivol   | -0.033 | 1.000  | 0.086  | -0.053   |
| z_pead   | 0.215  | 0.086  | 1.000  | 0.197    |
| z_resmom | 0.049  | -0.053 | 0.197  | 1.000    |

The original three signals are nearly perfectly orthogonal (|corr| < 0.06 across all pairs of GP, IVol, ResMom). PEAD has consistent ~0.2 positive correlation with GP and ResMom, reflecting shared exposure to a "quality earnings" theme.

The IC finding above — that ResMom has zero predictive power in this window — was validated empirically by re-running the backtest without ResMom (see [Current results](#current-results) above and `scripts/run_full_backtest_no_resmom.py`). Dropping ResMom raises Sharpe from 0.79 to 0.82 and hit rate from 58% to 65%, but worsens max drawdown from -14.5% to -20.5%. A zero-IC signal can still reduce tail risk when it absorbs composite weight that would otherwise concentrate on noisier signals — a methodological reminder that IC and risk are distinct measurements, not different views of the same quantity.

### Out-of-sample holdout test

The 2023-2024 portion of the backtest had been seen during development (IC analysis, PEAD addition, no-ResMom variant). To extract meaningful OOS evidence anyway, two analyses were executed per a pre-commitment doc ([`docs/holdout_test_design.md`](./docs/holdout_test_design.md)) that locked the metrics and failure protocol *before* any holdout numbers were inspected:

- **Analysis A**: strict OOS rerun of the 3-signal strategy (frozen pre-2023, PEAD excluded) on 2023-01 to 2024-11. 22 successful periods. Gross Sharpe **1.17**, 95% CI [0.64, 1.73]; net Sharpe **0.53**; max drawdown **-4.65%**; hit rate **59.1%**.
- **Analysis B**: contamination-acknowledged split of the existing 116-period backtest at 2022-12-31. 3-signal holdout Sharpe **1.18**, 4-signal **1.44**, no-ResMom **1.77**. The latter two cannot be claimed OOS due to development contamination.

All pre-committed thresholds met. Analysis A and B's 3-signal numbers cross-check to the rounding floor of cumulative compounding, confirming the backtest is deterministic and reproducible. Full writeup in [`docs/holdout_test_results.md`](./docs/holdout_test_results.md).

![In-sample vs holdout split](./docs/exhibits/09_in_sample_vs_holdout.png)

The honest one-sentence finding: *the strategy survives the holdout in pre-committed terms, but the window itself appears to have been regime-friendly* — max drawdowns under 5% across all variants, hit rates uniformly elevated, no 2020-style stress event. A robust OOS claim requires data containing a stress regime, which this window did not.

## Lazy Prices research (v2 Item 6a)

Replication and extension of Cohen-Malloy-Nguyen (2020), *"Lazy Prices"*, on a modern 2020-2025 sample. The signal measures year-over-year text change between consecutive 10-K filings; the original claim is that low-change firms (those whose 10-Ks read similarly to the prior year's) outperform high-change firms. axiom-fund uses TF-IDF cosine similarity across Items 1, 1A, and 7 as an approximation of the paper's exact methodology, which uses Jaccard similarity on whole documents (see [`docs/axiom-fund-history.md`](./docs/axiom-fund-history.md) for the full methodology comparison).

### The finding

The Lazy Prices anomaly concentrates in **large-cap firms** above the NYSE 80th percentile. Under CMN-standard NYSE-breakpoint methodology, the equal-weighted L/S portfolio in NYSE Size5 delivers annualized return **+6.08%**, FF6-adjusted alpha **+5.70%** with HAC t-statistic **3.04** (p = 0.002), and block-bootstrap p < 0.01 uniformly across block sizes 3-8. Adjusted R² is negative — the signal is factor-orthogonal.

![NYSE Size5 cumulative L/S, EW vs VW](./docs/exhibits/lazy_prices_01_size5_cumulative.png)

The equal-weighted L/S accrues steadily to +42% cumulative over the 71-month sample. The value-weighted L/S tracks EW through late 2022 and then declines to -15% cumulative — driven by a mega-cap concentration effect addressed below.

### Size decomposition

Under NYSE-breakpoint size buckets (rather than in-sample percentiles), the effect concentrates entirely in Size5 (firms above the NYSE 80th percentile, ≈ $12-19B market cap in modern dollars). Size2 through Size4 are effectively noise.

![Lazy Prices L/S by NYSE size bucket](./docs/exhibits/lazy_prices_02_size_heterogeneity.png)

Size1's -0.42 Sharpe is driven by extreme return magnitudes in a very small cohort (~24 firms per month), including a single-month outlier at +278% annualized. This is a low-N artifact, not a reliable pattern.

An earlier iteration of this analysis used in-sample percentiles for the size buckets and reported the peak at Size4. That framing was an artifact of universe composition — axiom-fund's CRSP top-1000 universe is size-biased toward larger firms, which compresses in-sample percentile cutoffs downward. NYSE breakpoints (CMN's standard) correct this.

### Mega-cap reversal mechanism

Within Size5, the value-weighted L/S portfolio suffers because ~10 mega-mega-cap firms drive returns opposite to the CMN direction. Excluding the top-10 firms per rebalance date restores the CMN direction:

![Mega-cap reversal: full vs excluding top-10](./docs/exhibits/lazy_prices_03_top10_mechanism.png)

The identified firms are the Magnificent Seven (AAPL, MSFT, AMZN, NVDA, GOOG/GOOGL, META, TSLA) plus rotating Big Healthcare / Big Finance names (JNJ, JPM, UNH, AVGO, LLY). Mechanism interpretation: at mega-mega-cap scale, heavy 10-K change signals business evolution (new AI product lines, cloud infrastructure expansion, new therapeutics), not risk disclosure. NVDA is the poster child — average LP quintile 4.81 (nearly always in the high-change bucket) with average forward return +4.89% per month.

### AI-era regime

Splitting the 71-month sample at ChatGPT's November 2022 release reveals that the mega-cap reversal is post-ChatGPT specific:

![Pre vs post ChatGPT FF6 alpha](./docs/exhibits/lazy_prices_04_ai_era_regime.png)

Pre-ChatGPT (2020-01 to 2022-11, N=35), both weighting schemes showed strongly positive alpha (+9.2% EW, +9.0% VW). Post-ChatGPT (2022-12 to 2025-11, N=36), EW weakened to +2.6% and VW reversed to -17.2%. The full-sample +5.70% alpha is a weighted average across these two regimes.

This finding is descriptive rather than confirmatory (N=35-36 per half; simple OLS residual variance for alpha SE; no HAC). But it is directionally clean, and the ChatGPT cutoff separates regimes more sharply than a Nvidia earnings-inflection cutoff (2023-05) does. Mechanism reading: mega-cap 10-K changes prior to ChatGPT signaled risk in the same direction as smaller firms; after ChatGPT, they signaled AI and product growth, inverting the CMN direction at that scale. Extending the sample backward to CMN's original 1995-2014 window would sharpen this claim substantially; that infrastructure work is queued as a separate research project.

### Robustness and caveats

The full-sample +5.70% ann alpha is robust across:

- HAC standard-error bandwidths (Newey-West lags 4, 6, 8 — all p < 0.005)
- Block-bootstrap block sizes (3, 4, 6, 8 — all p < 0.01)
- NYSE-breakpoint methodology (CMN standard) vs in-sample percentiles

Key caveats:

- **N = 71 months.** CMN's original result was measured on ~20 years of data. Extending the sample backward is the highest-value next research step.
- **Methodological approximation.** axiom-fund uses TF-IDF cosine per section; CMN's actual replication code uses Jaccard similarity on whole documents, with prior-year percentile binning and FF3+MOM+Pastor-Stambaugh liquidity factor controls (not FF5+MOM). A faithful replication project is scoped in a separate repository.
- **Regime-dependence.** The full-sample alpha averages a strong pre-ChatGPT regime with a substantially weakened post-ChatGPT regime.

Full technical audit trail — sign-convention correction, size-heterogeneity discovery, FF6 spanning and bootstrap detail, mega-cap firm identification, CMN methodology comparison — in [`docs/axiom-fund-history.md`](./docs/axiom-fund-history.md).

## Architecture

Four layers, each a separately-testable Python module:

- **Data** (`src/axiom_fund/data/`): CRSP universe construction, returns panel with delisting handling, Compustat fundamentals via CUSIP linkage, Ken French factors, parquet caching.
- **Signals** (`src/axiom_fund/signals/`): GP, IVol, ResMom, PEAD as pure functions consuming a fundamentals or returns panel. Shared alignment layer (`alignment.py`) handles forward-fill to rebalance calendar, winsorization, and cross-sectional z-scoring, with optional max-age filter for event-driven signals like PEAD.
- **Portfolio** (`src/axiom_fund/portfolio/`): Composite alpha aggregation, Ledoit-Wolf covariance, beta estimation, and cvxpy convex optimizer with dollar/beta/sector neutrality constraints.
- **Backtest** (`src/axiom_fund/backtest/`): point-in-time data cache, single-period engine, historical runner with checkpointing, transaction cost model, IC analysis framework, performance metrics.

Pure-function pattern throughout — no I/O in computation modules, dependency injection of the WRDS connection. 308 unit tests, 34 integration tests against real CRSP data.

## Tech stack

- Python 3.12, [uv](https://docs.astral.sh/uv/) for package management
- pandas 2.1, numpy 1.26, pyarrow
- cvxpy for convex optimization
- statsmodels for residualizing momentum
- scipy for IC t-statistics
- WRDS Python client for CRSP / Compustat / FF factor access
- pytest, ruff, mypy for development

Dependencies pinned in `pyproject.toml` and locked in `uv.lock`.

## Repository structure

```bash
axiom-fund/
├── docs/
│   ├── strategy_spec.md         locked strategy specification
│   ├── methodology_audit.md     PIT discipline & bias audit
│   ├── signal_design.md         signal-layer design rationale
│   └── limitations.md           pre-committed limitations
├── src/axiom_fund/
│   ├── data/                    universe, returns, fundamentals, FF factors
│   ├── signals/                 GP, IVol, ResMom, PEAD, alignment
│   ├── portfolio/               composite, covariance, betas, optimizer
│   └── backtest/                engine, costs, metrics, IC analysis
├── tests/                       308 unit tests
├── scripts/
│   ├── analysis/                IC analysis driver
│   ├── exploration/             one-off smoke tests
│   ├── run_full_backtest.py     full historical run driver
│   └── apply_costs_to_full_backtest.py    cost overlay driver
├── pyproject.toml
└── LICENSE
```

## Reproduction

Prerequisites: Python 3.12 (or let uv manage it), uv, a WRDS account with CRSP + Compustat subscriptions, and `~/.pgpass` configured for WRDS.
```bash
git clone git@github.com:daviddavilad/axiom-fund.git
cd axiom-fund
uv sync
```

Create a `.env` file with `WRDS_USERNAME=your_username`. Verify connectivity:
```bash
uv run python scripts/test_wrds_connection.py
```

Run the test suite (no WRDS needed for unit tests):
```bash
uv run pytest -m "not integration"
```

Run the full historical backtest (~5 hours on a laptop, mostly WRDS data fetch):
```bash
uv run python scripts/run_full_backtest.py
```

Apply costs to the gross backtest (~10 min):
```bash
uv run python scripts/apply_costs_to_full_backtest.py
```

Run the IC analysis (~4 hours given the per-period composite rebuild):
```bash
uv run python scripts/analysis/ic_analysis.py
```

## Limitations

A few important ones, in addition to those in [`docs/limitations.md`](./docs/limitations.md):

- **PEAD uses time-series SUE rather than analyst-based SUE**, because the WRDS subscription available to me does not include IBES. Livnat-Mendenhall (2006) suggests analyst-based SUE delivers roughly 30% higher IC; if this strategy were deployed under a subscription with IBES access, the PEAD signal would likely be measurably stronger.
- **Transaction cost model is conservative** — full half-spread per trade, Corwin-Schultz (which is known to overshoot for illiquid names by 5–15%). Realistic institutional execution would capture some of this back, producing net Sharpe in the 0.30–0.50 range.

![Cost sensitivity: gross vs net under different execution assumptions](./docs/exhibits/07_cost_sensitivity.png)

- **9.8 years is a short backtest window for factor research.** The IC t-statistics above are computed on 116 monthly observations; the literature typically uses 50+ years of data. Findings here are suggestive of in-sample patterns, not definitive about long-run signal quality.
- **No regime detection.** The signals reverse in bear regimes (especially IVol). A deployable production system would gate exposure by a market-regime indicator, or weight signals by trailing-window IC.

## Status and roadmap

Phases 5 through 7 (v1) complete. v1 release: June 2026. Phase 7 delivered:

- **Single-signal return attribution** (`scripts/analysis/run_attribution.py`) running each signal in isolation under the same neutrality + position-cap constraints as the composite. Headline: GP alone matches the 4-signal composite at Sharpe 0.80; IVol single-signal is *negative* at Sharpe -0.29 (driven by bear-regime reversal); PEAD delivers Sharpe 0.70. Reusable framework in `src/axiom_fund/backtest/attribution.py`.
- **No-ResMom variant backtest** confirming the IC finding empirically. Sharpe 0.82 vs 0.79, hit rate 65% vs 58%, but max drawdown widens to -20.5%. Documented above; the variant infrastructure (`signals` parameter on `run_historical_backtest`) generalizes to future composites.
- **Holdout test** with pre-committed thresholds and failure protocol ([`docs/holdout_test_design.md`](./docs/holdout_test_design.md)). Two analyses, full writeup in [`docs/holdout_test_results.md`](./docs/holdout_test_results.md). All thresholds met; honest framing of the regime-friendly-window caveat.

### v2 (target: mid-September 2026)

Methodological upgrades to v1's statistical inference, liquidity treatment, and out-of-sample testing. Progress tracked in [`docs/v2_design.md`](./docs/v2_design.md).

**Phase 1 (foundation) — complete:**

- **Item 1 — Liquidity audit (closed).** Audit (`scripts/analysis/liquidity_audit.py`) found v1's existing top-1000 + $5M ADV construction already eliminates stale-price contamination. Candidate tighter screens would exclude 0-2 of 112,177 name-months. Cosmetic no-op infrastructure not added; finding reported honestly.
- **Item 2 — Residual diagnostics framework (closed).** Six pure functions in [`src/axiom_fund/diagnostics/residual_diagnostics.py`](./src/axiom_fund/diagnostics/residual_diagnostics.py); 30 unit tests; applied to both ResMom (cross-sectional, 121 months) and IVol (FF3 trailing-60-day, 203,833 regressions). Findings in [`docs/v2_diagnostics_findings.md`](./docs/v2_diagnostics_findings.md): ResMom is heteroskedastic in 86% of months at p<0.01 with extreme residual kurtosis (max 931, with GameStop in January 2021 producing the largest standardized residual at 34σ); IVol residuals are clean (DW distribution centered at 2.034, no systematic autocorrelation).
- **Item 3 — HAC standard errors and bootstrapped CIs (closed).** Newey-West HAC and stationary block-bootstrap implementations in [`src/axiom_fund/diagnostics/inference.py`](./src/axiom_fund/diagnostics/inference.py); 24 unit tests; applied to v1's existing IC t-stats and Sharpe CIs ([`scripts/analysis/apply_inference_v2.py`](./scripts/analysis/apply_inference_v2.py)). Findings: HAC corrections on IC t-stats are 2-10% (largest for IVol at 10%, methodologically expected from its rolling-window construction); bootstrap Sharpe CIs are 11-13% wider than asymptotic. None of v1's qualitative conclusions overturned; the most fragile result is IVol's apparent significance (HAC L=5 t = 2.09, barely clears t > 2.0).

**Phase 2 (rigorous reporting + research extensions) — in progress:**

- **Item 4 — Deflated Sharpe Ratio (closed).** Bailey & López de Prado (2014) DSR implementation in [`src/axiom_fund/diagnostics/inference.py`](./src/axiom_fund/diagnostics/inference.py); 15 unit tests; applied to v1's three holdout variants at N = 3, 7, 20 ([`scripts/analysis/apply_dsr_to_v1.py`](./scripts/analysis/apply_dsr_to_v1.py)). **First v2 finding that materially changes a v1 qualitative conclusion**: 3-sig and 4-sig composites containing ResMom fail DSR at all N; no-ResMom (the headline Sharpe 1.77 variant) clears DSR = 0.965 at N = 3 but fails at any broader trial count. The 1.77 nominal Sharpe is significant only under the narrowest possible variant interpretation. Full analysis in [`docs/v2_diagnostics_findings.md`](./docs/v2_diagnostics_findings.md).
- **Item 5 — Quandt-Andrews structural-break test (closed).** Hansen (1997) sup-F implementation with full Table 2 lookup in [`src/axiom_fund/diagnostics/structural_break.py`](./src/axiom_fund/diagnostics/structural_break.py); 18 unit tests including independent cross-check against Andrews (1993) critical values; applied to v1's four monthly IC series ([`scripts/analysis/apply_qa_to_ic.py`](./scripts/analysis/apply_qa_to_ic.py)). **No signal shows statistically significant evidence of a structural break in mean IC.** Strongest candidate is IVol (sup_F = 7.17, break 2021-01-29, p = 0.097), which does not clear conventional 5%. v1's regime references should be treated as descriptive observations about subperiods, not established empirical findings. Full analysis in [`docs/v2_diagnostics_findings.md`](./docs/v2_diagnostics_findings.md).
- **Item 6a — Lazy Prices NLP signal (closed).** Cohen-Malloy-Nguyen (2020) replication on a 2020-2025 sample, using TF-IDF cosine across Items 1/1A/7 as an approximation of the paper's Jaccard-on-whole-document methodology. NYSE Size5 EW L/S produces FF6-orthogonal alpha +5.70%/yr (HAC t = 3.04, bootstrap p < 0.01). Mega-cap reversal mechanism identified: excluding the top-10 mega-caps under value-weighting flips the sign. AI-era regime finding: reversal is post-ChatGPT specific. Full findings in the [Lazy Prices research](#lazy-prices-research-v2-item-6a) section above; technical audit trail in [`docs/axiom-fund-history.md`](./docs/axiom-fund-history.md).
- **Item 7 — Walk-forward IC weighting variant** as a research exercise on overfitting risk in signal-weight selection.
- **Item 8 — Simple regime overlay** with binary regime indicator gating gross exposure, directly addressing the holdout-was-easy caveat.

### v3 (target: 2027)

- **Form 4 insider-buying signal** — opportunistic vs routine classification per Cohen-Malloy-Pomorski (2012); free via SEC EDGAR
- **Scale-aware backtest** with realistic market-impact modeling at $1B NAV
- **Extended backtest window** through 2008-2014 to include the GFC stress regime

## References

- Almgren, R., Thum, C., Hauptmann, E., & Li, H. (2005). Direct estimation of equity market impact. *Risk*.
- Ang, A., Hodrick, R. J., Xing, Y., & Zhang, X. (2006). The cross-section of volatility and expected returns. *Journal of Finance*.
- Bernard, V. L., & Thomas, J. K. (1989). Post-earnings-announcement drift: delayed price response or risk premium? *Journal of Accounting Research*.
- Blitz, D., Huij, J., & Martens, M. (2011). Residual momentum. *Journal of Empirical Finance*.
- Cohen, L., Malloy, C., & Nguyen, Q. (2020). Lazy prices. *Journal of Finance*.
- Corwin, S. A., & Schultz, P. (2012). A simple way to estimate bid-ask spreads from daily high and low prices. *Journal of Finance*.
- Fama, E. F., & French, K. R. (1993). Common risk factors in the returns on stocks and bonds. *Journal of Financial Economics*.
- Foster, G., Olsen, C., & Shevlin, T. (1984). Earnings releases, anomalies, and the behavior of security returns. *The Accounting Review*.
- Ledoit, O., & Wolf, M. (2003). Improved estimation of the covariance matrix of stock returns with an application to portfolio selection. *Journal of Empirical Finance*.
- Livnat, J., & Mendenhall, R. R. (2006). Comparing the post-earnings announcement drift for surprises calculated from analyst and time series forecasts. *Journal of Accounting Research*.
- McLean, R. D., & Pontiff, J. (2016). Does academic research destroy stock return predictability? *Journal of Finance*.
- Novy-Marx, R. (2013). The other side of value: the gross profitability premium. *Journal of Financial Economics*.
- Shumway, T. (1997). The delisting bias in CRSP data. *Journal of Finance*.

## License

MIT. See [`LICENSE`](./LICENSE).

## Acknowledgements

This project was built with assistance from Claude (Anthropic) for code review, debugging, methodology discussion, and writing. All algorithmic choices, the locked strategy specification, and the methodology audit are mine; AI suggestions were hand-verified, and all code committed to this repository was reviewed before commit.

I am grateful to Dr. Subramanian Iyer (UNM Anderson School of Management) for ongoing mentorship on portfolio construction and academic research methodology.

## Author

**David Davila** — University of New Mexico (BBA Finance + BS Applied Math, class of 2027). CFA Level II candidate (August 2026). Targeting MFE programs and quantitative research roles.

This project is built solo as a research portfolio artifact. It is intended to demonstrate methodological discipline and quantitative research infrastructure, not to claim alpha.