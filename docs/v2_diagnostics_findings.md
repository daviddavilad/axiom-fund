# v2 Diagnostics Findings

Empirical findings from applying the v2 residual diagnostics framework (commit `f052edb`) to the two regression sites in v1: residual momentum (cross-sectional, monthly) and idiosyncratic volatility (FF3, trailing-60-day, per (permno, terminal_date)). Per the v2 design doc (`docs/v2_design.md`), the diagnostics return findings honestly regardless of outcome.

## Scope

Two diagnostic runs, both committed:

- **ResMom**: 5 of 6 diagnostics (Durbin-Watson excluded — cross-sectional residuals are not a time series). 121 months in 2015-01 to 2024-12. Mean 1,442 observations per month. Script: `scripts/analysis/apply_diagnostics_resmom.py`.
- **IVol**: Durbin-Watson only (5 other diagnostics applicable but not in scope for this run; FF3 regression is the natural site for autocorrelation testing). 203,833 regressions across 1,900 permnos × 120 month-end terminal dates, subsampled from production's per-trading-day frequency. Subsampling at month-ends is unbiased because each regression is self-contained on a 60-day window. Script: `scripts/analysis/apply_dw_to_ivol.py`.

## ResMom findings

The cross-sectional regression `monthly_ret ~ industry_dummies + log(marketcap) + intercept` produces residuals that drive the residual momentum signal. The diagnostics characterize those residuals.

### Heteroskedasticity is overwhelming

Breusch-Pagan LM rejects homoskedasticity in **108 of 121 months at p<0.05 (89.3%)** and **104 of 121 at p<0.01 (86.0%)**. Median p-value across months: approximately zero.

This is a well-known property of cross-sectional return regressions. Stocks differ in idiosyncratic risk; the regressors (industry dummies, log size) explain mean returns but not variance. The implication for v1: OLS standard errors on the regression coefficients understate uncertainty. The residual signal itself is unbiased — the residuals are correctly computed — but within-month inference based on those OLS standard errors is misleading.

### Residual non-normality is severe

| Statistic | Mean | Std | Min | Max |
|---|---|---|---|---|
| Skewness | +2.23 | 3.42 | -1.42 | +28.04 |
| Excess kurtosis | +35.93 | 95.38 | +1.33 | +931.41 |

Monthly cross-sectional return distributions have heavy right tails and significant positive skew. This is again standard for equity returns — episodic winners (post-earnings surprises, takeover targets, single-name short squeezes) produce extreme right-tail observations. The implication: confidence intervals from a t-distribution will be too narrow on either tail. Inference assuming normality is not appropriate.

### High-influence observations exist

After filtering 8 degenerate observations (leverage = 1.0, solo-industry in their month), the distribution of maximum Cook's distance per month:

| Percentile | Cook's |
|---|---|
| 50th | 0.063 |
| 75th | 0.103 |
| 90th | 0.188 |
| 95th | 0.277 |
| 99th | 0.644 |
| max | 0.723 |

All these are well above the rule-of-thumb 4/n threshold (approximately 0.003 for 1,500 observations). A handful of high-influence names per month are pulling the regression toward themselves. The residual momentum signal amplifies for these observations because their residual is partly "the regression got pulled away from where the rest of the data wanted it."

The single largest standardized residual across all months: **permno 89301 (GameStop), January 2021, at 34.2 sigma.** This is the WallStreetBets short squeeze. The detection of this observation as the most extreme residual in the dataset validates that the diagnostic framework is identifying real outliers — not script artifacts.

### Solo-industry artifact

8 observations across 5 months had leverage exactly equal to 1.0 — corresponding to stocks that were the only member of their industry group in that month. The regression fits these observations exactly by construction; residual is zero up to floating-point precision. These observations do not bias the strategy (a zero residual contributes nothing to the momentum signal), but they slightly reduce the effective rank of the regression and are noted for transparency.

### Mean R-squared: 0.131

A 13% R-squared is unsurprising and arguably the entire point. Industry and size explain 13% of cross-sectional monthly return variance; the remaining 87% is what residual momentum captures.

## IVol findings

The FF3 regression `excess_ret ~ MKT + SMB + HML + intercept` is fit on a trailing 60-day window for each (permno, terminal_date) pair. Durbin-Watson tests the time-series autocorrelation of the residuals.

### Clean null result

Distribution of DW across 203,833 regressions:

| Statistic | Value |
|---|---|
| Mean | 2.034 |
| Median | 2.037 |
| Std | 0.302 |
| 5th percentile | 1.537 |
| 95th percentile | 2.524 |

Classification using critical values at N=60, k=3 regressors (5% one-sided):

| Range | Count | Share |
|---|---|---|
| DW < 1.44 (strong positive autocorrelation) | 5,445 | 2.7% |
| 1.44 ≤ DW < 1.73 (indeterminate, positive side) | 25,979 | 12.7% |
| 1.73 ≤ DW ≤ 2.27 (no evidence of autocorrelation) | 128,650 | 63.1% |
| 2.27 < DW ≤ 2.56 (indeterminate, negative side) | 35,757 | 17.5% |
| DW > 2.56 (strong negative autocorrelation) | 8,002 | 3.9% |

Under H₀ of no autocorrelation, ~10% would fall in the strong-rejection zones (5% each side) by chance. We see 6.6% combined — less than chance. The FF3 residuals are consistent with no systematic autocorrelation.

### Mild bid-ask bounce visible

The mean (2.034) and median (2.037) are slightly above 2.0 — a small shift toward negative autocorrelation. This is consistent with bid-ask bounce in daily returns: when a stock alternates trading at the bid and ask, daily returns have mild negative serial correlation. The effect is small (~0.034 above 2.0) but visible at this sample size. It does not affect IVol's signal — the signal is residual *standard deviation*, not residual autocorrelation — but is worth noting as a microstructure observation.

## Combined interpretation

The two regressions exhibit asymmetric properties, both methodologically sensible:

| Regression | Heteroskedasticity | Non-normality | Autocorrelation |
|---|---|---|---|
| ResMom (cross-sectional, monthly) | Yes (86% at p<0.01) | Yes (kurtosis up to 931) | N/A |
| IVol (time-series, 60-day) | not tested in this run | not tested in this run | No (consistent with null) |

ResMom's residuals fail the assumptions of standard OLS inference. IVol's residuals satisfy the no-autocorrelation assumption.

Neither result is novel as an academic finding (both are well-documented in cross-sectional asset pricing literature). What is novel for this project is the empirical confirmation that v1's residuals exhibit these properties on the specific dataset and universe used. The diagnostics are not just theoretical; they describe what is actually present in v1's output.

The diagnostics distinguish two questions that the v2 design did not separate cleanly:

1. **Are the residuals correctly computed?** Yes. The regression mechanics — fit OLS, take residuals — do not require homoskedasticity or normality. The residual signal v1 uses is an unbiased estimate of what it claims to estimate (the component of monthly return orthogonal to industry and size).

2. **Do the residuals support the interpretation that "high ResMom = persistent momentum"?** Partially. In the central mass of the distribution, yes. In the tails, the heavy-right-skew and extreme kurtosis findings imply that high ResMom values are increasingly likely to be one-off events (GameStop in January 2021 being the extreme illustration) rather than persistent momentum signals. v1's signal pipeline does not distinguish these cases; the composite z-score treats a 34-sigma residual and a 2-sigma residual as ordered points on the same scale.

What is biased, separately from both points above, is the **inference about the signal's quality** — IC t-statistics, Sharpe confidence intervals, and any claim that requires homoskedastic or normal residuals to be statistically valid. That is what v2 Item 3 addresses.

## Item 3: HAC + bootstrap applied

The Item 2 findings (heteroskedastic ResMom residuals, non-normal residual distributions across both regressions) motivated the v2 Item 3 inference tools shipped in commit `e310e3d`: HAC standard errors and stationary block-bootstrapped Sharpe confidence intervals. This section applies those tools to v1's reported statistics and reports what changes.

Methodology choices, documented in code for reproducibility:
- HAC lag truncation L = 3 and L = 5 (spans Andrews/Newey rule of thumb at N = 116 monthly periods)
- Block bootstrap: block_size = 6 (approximately N^(1/3) for monthly), n_resamples = 10,000, percentile method
- Spearman IC used (matches v1's onepager convention)
- Annualization factor √12 for Sharpe (monthly returns)

Source data: `data/cache/ic_analysis_4sig/ic_long.parquet` (116 monthly Spearman ICs per signal) and `data/cache/backtest_full_top1000/net_returns.parquet` (116 monthly portfolio returns, gross and net). Script: `scripts/analysis/apply_inference_v2.py`.

### IC t-stats under HAC correction

| Signal | Mean IC | Naive t | HAC t (L=3) | HAC t (L=5) |
|---|---|---|---|---|
| GP | 0.0235 | 2.340 | 2.193 | 2.285 |
| IVol | 0.0334 | 2.334 | 2.133 | 2.092 |
| PEAD | 0.0209 | 2.672 | 2.699 | 2.584 |
| ResMom | 0.0029 | 0.258 | 0.278 | 0.289 |

The HAC correction reduces the naive t-stat by 2-10% for GP, IVol, and PEAD, and slightly inflates it for ResMom (where the underlying IC time series shows mild negative serial dependence — the correction technically goes the other way). The magnitude of the correction is largest for IVol, which is methodologically expected: IVol's signal is a rolling 60-day residual standard deviation, so consecutive month-end values share 59 of 60 daily inputs by construction. This induces persistence in the IVol cross-section and therefore in the per-month IC time series. PEAD's IC is essentially unchanged — its 21-day post-earnings window introduces less month-to-month persistence than the 60-day IVol window.

The conclusion under HAC: **all four signals still fail the Harvey-Liu-Zhu (2016) t > 3.0 multiple-testing threshold.** This is the same conclusion v1 reached under naive inference, now confirmed under HAC. The HAC correction does not rescue any of v1's signals from the HLZ bar, but neither does it overturn any of them. The result is a corroboration of v1's existing claim that none of the four individual signals clears the strict multiple-testing threshold, which is why v1's defense rests on the composite.

ResMom's HAC t-stat remains essentially zero. v1's existing no-ResMom variant analysis already demonstrated that ResMom contributes negligibly to composite Sharpe; the HAC IC analysis now confirms this from the signal-construction side as well.

### Sharpe ratio confidence intervals under bootstrap

Annualized Sharpe (× √12) and 95% confidence intervals:

| Series | Sharpe | Asymptotic CI | Bootstrap CI | Width change |
|---|---|---|---|---|
| Gross | 0.787 | [0.148, 1.425] | [0.143, 1.586] | +13.0% |
| Net | 0.181 | [-0.450, 0.811] | [-0.450, 0.957] | +11.5% |

The block-bootstrap CIs are 11-13% wider than the asymptotic CIs. The lower bounds are essentially identical between methods; the bootstrap widens the upper bound. This is the heavy right-tail of returns showing up — bootstrap resamples that happen to include disproportionately many of the best months push the upper percentile out further than a Gaussian asymptotic would predict. The asymptotic CI assumes returns are i.i.d. normal; v1's monthly portfolio returns are neither.

The **net Sharpe CI includes zero under both inference methods**. v1's reported net Sharpe of 0.18 is not statistically distinguishable from zero at conventional significance. This was already implicit in v1's onepager (which reported the wide CI), but the bootstrap formalization makes it explicit. Transaction costs (mean 31 bps round-trip across 116 rebalances) eat most of the gross signal; the remaining net return is consistent with sampling noise at this N.

The gross Sharpe lower bound at 0.14 is comfortably above zero under both inference methods. The strategy's gross signal is statistically significant; what is not significant is the residual after costs.

### Summary

The v2 Item 3 inference confirms what v1's naive inference already suggested:
1. No individual signal clears Harvey-Liu-Zhu's t > 3.0 multiple-testing threshold.
2. The 4-signal composite's gross Sharpe is statistically distinguishable from zero but its net Sharpe is not.
3. ResMom's marginal contribution remains negligible.

None of v1's qualitative conclusions are overturned by the rigorous inference, but the picture is more nuanced than "nothing changed":

- **IVol's apparent significance erodes meaningfully.** Naive t = 2.33 falls to HAC t = 2.09 at L = 5, a 10% drop. This is the largest correction among the four signals and is methodologically expected from IVol's rolling-window construction (consecutive month-end IVol values share 59 of 60 daily inputs by construction, inducing autocorrelation in the cross-section and therefore in the IC time series). The HAC-corrected t still narrowly clears the conventional t > 2.0 threshold but is closer to that boundary than the naive number suggests. Under any further correction — multiple testing, deflated Sharpe on the Sharpe side of the analysis, or a more aggressive L = 7 or 10 — IVol's apparent significance is the most fragile of the four signals.
- **PEAD's t-stat is robust.** Naive 2.67 vs HAC L=3 2.70, L=5 2.58. The smallest IC autocorrelation; the most assumption-robust signal.
- **Sharpe CIs widen by 12-13%, but the qualitative conclusions hold.** Gross Sharpe remains statistically positive; net Sharpe remains statistically indistinguishable from zero.

The honest reading: v1's standard asymptotic inference was approximately correct on this dataset because the IC series autocorrelation happened to be modest. This is not a virtue of v1's methodology — it is a property of the underlying data. If the IC autocorrelation had been larger (which is possible at different sampling frequencies or with different signal-construction choices), v1's reported t-stats would have been more inflated relative to the truth. The Item 3 framework now insures against that risk for future iterations.

The findings that *would* have qualitatively flipped a conclusion, and that did not occur: an IC t-stat that was significant under naive inference but failed under HAC, or a Sharpe CI that excluded zero under asymptotic inference but included zero under bootstrap. Neither happened on this dataset.

## Item 4: deflated Sharpe ratio applied

The v2 design doc Item 4 calls for Bailey & López de Prado (2014) deflated Sharpe ratio (DSR) applied to v1's variant comparison. DSR corrects two distinct biases when reporting the best Sharpe across multiple trials: (1) the Mertens (2002) correction for non-normality of returns, and (2) the selection bias from picking the best of N variants. The function implementations are in commit `9505755`; tests in `d87d0aa`. This section applies them to v1's holdout variants.

Methodology choices, documented in code:
- DSR computed on monthly Sharpes (formula is scale-sensitive through the Mertens denominator; mixing monthly and annualized would be incorrect)
- Trial Sharpes: the three v1 holdout variants (3-sig, 4-sig, no-ResMom), giving N = 3 directly
- N-sensitivity: also reported at N = 7 (rough estimate of total signal combinations considered during v1 development) and N = 20 (upper-bound including implicit choices like formation windows). For N > 3, synthetic trial arrays are constructed preserving the observed variance of the 3 actuals (the BLP-recommended approach when reported variants under-represent the true search space).
- Mertens denominator uses sample skewness and excess kurtosis of each variant's 22-period holdout return series.

Source data: `data/cache/backtest_full_top1000{_4sig,_no_resmom}/results.csv`. Script: `scripts/analysis/apply_dsr_to_v1.py`.

### Per-variant statistics from the holdout

| Variant | n | Monthly Sharpe | Annual Sharpe | Skewness | Excess kurtosis |
|---|---|---|---|---|---|
| 3-sig | 22 | 0.339 | 1.17 | -0.46 | -0.03 |
| 4-sig | 22 | 0.408 | 1.41 | -0.94 | +1.45 |
| no-ResMom | 22 | 0.499 | 1.73 | -0.01 | +0.97 |

Note: the annualized Sharpe values are ~3% lower than the onepager's reported 1.18, 1.44, 1.77. The discrepancy is a methodology choice — the onepager uses geometric annualized return ÷ annualized vol; we use the standard arithmetic Sharpe (monthly mean ÷ monthly std × √12). DSR is invariant to this choice provided all inputs are on the same scale; this analysis uses monthly throughout.

### DSR results

| Variant | Annual SR | N | SR* (annual) | DSR | Clears 0.95? |
|---|---|---|---|---|---|
| 3-sig | 1.17 | 3 | 0.24 | 0.870 | no |
| 3-sig | 1.17 | 7 | 0.39 | 0.828 | no |
| 3-sig | 1.17 | 20 | 0.53 | 0.781 | no |
| 4-sig | 1.41 | 3 | 0.24 | 0.896 | no |
| 4-sig | 1.41 | 7 | 0.39 | 0.864 | no |
| 4-sig | 1.41 | 20 | 0.53 | 0.828 | no |
| no-ResMom | 1.73 | 3 | 0.24 | 0.965 | barely (yes) |
| no-ResMom | 1.73 | 7 | 0.39 | 0.948 | no |
| no-ResMom | 1.73 | 20 | 0.53 | 0.927 | no |

### Interpretation

This is the first v2 finding that materially changes one of v1's qualitative conclusions. Items 2 and 3 corroborated v1's existing claims; Item 4 does not.

**Three substantive observations:**

1. **The two composites containing ResMom fail DSR at all N.** Neither 3-sig nor 4-sig clears the conventional DSR > 0.95 threshold even at the narrowest N = 3. Their reported holdout Sharpes are statistically compatible with selection-bias chance given the variant search. This is consistent with Item 3's IC analysis showing ResMom contributes negligibly (HAC t ≈ 0.3); DSR now confirms from the Sharpe side that 3-sig and 4-sig are not robust holdout claims.

2. **The no-ResMom variant barely clears DSR at N = 3, fails at any larger N.** At the narrowest possible trial interpretation (only the three published variants count), no-ResMom's DSR is 0.965 — just above the 0.95 threshold. As soon as we acknowledge any wider search (N = 7 reflecting signal-weighting alternatives considered during development; N = 20 including implicit decisions), DSR drops below 0.95. Whether no-ResMom survives multiple-testing correction depends on how many trials we count.

3. **The DSR conclusion is sensitive to N in a way that asymptotic CIs were not.** v1's onepager reports no-ResMom's 95% CI as [1.10, 2.44] — comfortably positive. Item 3's bootstrap analysis (commit `473234a`) widened these but did not change qualitative conclusions. Item 4 is the first inference framework where the answer flips: significant at N = 3, not significant at N > 3.

**The honest reading of v1's holdout claim:** the strategy survives standard inference and HAC + bootstrap (Items 2 and 3), but not deflated Sharpe under any reasonable expansion of the trial count. The no-ResMom Sharpe of 1.77 is the headline number; under DSR with N = 7, it is consistent with being the lucky best of seven equally-zero strategies. Whether to believe N = 3 (narrow, defensible by published-variants-only argument) or N > 3 (broader, defensible by counting development trials honestly) is a methodological choice, not a statistical one. The pre-commitment to honest reporting requires acknowledging both.

**What does not change:** the no-ResMom composite still has the best out-of-sample Sharpe and the highest hit rate. DSR does not say the signal is fake; it says the headline number is not robust to selection bias correction at a reasonable trial count. The strategy may still work; the reported Sharpe is just less informative than it appeared.

**Implication for the v2 deliverable**: the no-ResMom variant's headline Sharpe of 1.77 should not be claimed as "robust to multiple-testing correction" in any forward-facing communication. The honest framing is "1.77 nominal Sharpe; DSR at N=3 = 0.96 just clears 0.95; DSR at any broader trial count fails."

## Item 5: Quandt-Andrews structural-break test applied

The v2 design doc Item 5 calls for the Quandt-Andrews structural-break test on IC time series, "replacing the existing ad-hoc bull/bear regime classification with formal break-point detection." The module + tests are in commit `0c50a86`. This section applies them to v1's four monthly IC series.

Methodology choices, documented in code:
- For each signal, the test is `ic_monthly = intercept + epsilon` (regression on a constant), testing for a structural break in the mean. m = 1.
- Trimming π₀ = 0.15 (standard; candidate break dates are searched over the middle 70% of the 116-period sample).
- sup_F statistic + Hansen (1997) chi-squared p-value via Table 2 lookup.

Source data: `data/cache/ic_analysis_4sig/ic_long.parquet` (116 monthly Spearman ICs per signal, Jan 2015 - Nov 2024). Script: `scripts/analysis/apply_qa_to_ic.py`.

### Results

| Signal | sup_F | Break date | Mean pre | Mean post | Hansen p |
|---|---|---|---|---|---|
| IVol | 7.17 | 2021-01-29 | +0.004 | +0.080 | 0.097 |
| GP | 2.65 | 2020-07-31 | +0.038 | +0.005 | 0.634 |
| PEAD | 2.11 | 2020-12-31 | +0.012 | +0.035 | 0.762 |
| ResMom | 1.98 | 2023-01-31 | -0.005 | +0.035 | 0.793 |

For context: Andrews (1993) critical values at m = 1, π₀ = 0.15 are sup_F ≈ 8.85 (5%), 11.79 (1%). None of the four signals reach the 5% threshold; the most extreme (IVol) falls short at p = 0.097.

### Interpretation

**Headline: no signal shows statistically significant evidence of a structural break in mean IC.** The strongest candidate is IVol at p = 0.097, which does not clear the conventional 5% threshold and is far from the Bonferroni-adjusted threshold (0.05 / 4 = 0.0125) that family-wise multiple-testing across the four signals would require.

This is a clean negative result for v1's regime-dependent narrative. v1's onepager has language about bull vs bear regime performance, implying that signals behave differently in different macro environments. Quandt-Andrews finds no statistical support for that implied story at the individual signal level. Whether v1's *portfolio* performance shows regime dependence (rather than its individual signals' IC) is a different question this test does not address.

**Three observations worth noting, without overclaiming:**

1. **IVol is the only signal in the "interesting" zone.** Pre-break IVol IC of +0.004 (essentially zero) shifting to +0.080 (substantial positive IC) is a large relative shift; the break date 2021-01-29 aligns with the COVID / meme-stock regime change. But p = 0.097 does not constitute statistical evidence at any conventional threshold, and Item 3 already showed IVol has the most autocorrelation in its IC series — both findings consistently point to IVol's IC series being the least stable of the four, but neither finding rises to "we have detected a regime change in this signal."

2. **ResMom's apparent shift coincides with the holdout window.** ResMom's largest local F-statistic occurs at 2023-01-31, exactly the holdout window start. Pre-break mean (the 2015-2022 in-sample period) is essentially zero; post-break mean is positive. At p = 0.793 this is *clearly not* a statistically significant finding — but the timing aligns with v1's observation that the holdout window was regime-friendly. Speculative; not a claim.

3. **N = 116 is small for break detection.** Power for detecting moderate breaks is limited. The "no significant break" conclusion for the three quiet signals could mean either (a) means are genuinely stable, or (b) breaks exist but our power isn't sufficient. Item 5 does not distinguish these cases. The honest conclusion is the conjunction: "with this sample size and this test, no evidence of a break."

### Implication for the v1 narrative

The v1 onepager's regime discussion was presented as descriptive characterization, not as a tested claim. Item 5 does not invalidate it (Item 5 tests structural breaks in mean IC, not regime-conditional performance generally), but Item 5 also provides no statistical backing for it. **The honest framing going forward: regime references in v1 should be treated as descriptive observations about subperiods, not as established empirical findings.** Establishing actual regime dependence would require either a much larger sample or a different test (e.g., Markov-switching, regime-classification + sub-sample inference with adequate power).

This is the second v2 finding where rigorous methodology produces a "less than the loose narrative suggested" conclusion. Item 4 found that v1's headline 1.77 Sharpe is sensitive to multiple-testing N. Item 5 finds that v1's regime story has no formal structural-break support. Both findings constrain how v1's results should be communicated, not whether v1's strategy works.

## Implications for v2

These findings motivate the work items already in the v2 design doc:

- **Item 3 (HAC standard errors + bootstrapped CIs) — closed.** Results documented in the section above. HAC corrections on IC t-stats are 2-10% in magnitude (largest for IVol); bootstrap CIs are 11-13% wider than asymptotic CIs. No qualitative conclusion from v1 was overturned; the most fragile result under further corrections is IVol's apparent significance. Note: HAC and bootstrap *do not fix the underlying residual distribution* identified in Item 2 — they only change how uncertainty is reported about statistics computed from those residuals. Addressing the residual distribution itself (winsorization, robust regression, or restricting the signal's effective range) is out of v2 scope and remains a v3 candidate.

- **Item 5 (Quandt-Andrews structural break test)**: replaces the informal bull/bear regime classification with formal break-point detection on IC time series. Independent of the diagnostics findings above but consistent with the broader rigor agenda.

## Reproducibility
```bash
uv run python scripts/analysis/apply_diagnostics_resmom.py

uv run python scripts/analysis/apply_dw_to_ivol.py
```

Runtime: ResMom approximately 3 minutes, IVol approximately 7 minutes. Output data in `data/cache/diagnostics_resmom/` and `data/cache/diagnostics_ivol_dw/`. Source data fetched from WRDS at runtime; not cached.