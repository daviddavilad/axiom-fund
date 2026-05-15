# Methodology Audit

Verification pass against known methodological pitfalls in
quantitative equity backtesting. Each bias category is checked
against the existing code with line references.

## Survivorship bias

**Status:** Handled by construction.

The `Universe` class rebuilds the eligible universe at each rebalance
date from CRSP, using `crsp.dsf` filtered by share code (10, 11),
exchange (NYSE, NASDAQ, AMEX), price > $5, and 20-day ADV > $5M as
of the rebalance date. Stocks that delisted after a rebalance are
correctly present in earlier universes and absent in later ones. No
look-ahead via "today's S&P 500" or similar shortcut.

`src/axiom_fund/data/universe.py`

## Delisting returns (Shumway 1997)

**Status:** Handled.

The `ReturnsPanel` class merges normal daily returns from `crsp.dsf`
with delisting returns from `crsp.dsedelist`, flagging delisting
rows with `is_delisting=True`. When a PERMNO delists, its final
return captures both the last trading day return and the post-trading
delisting return (`dlret`), so failed companies' negative returns
are not silently missed.

`src/axiom_fund/data/returns.py` lines 207-264

In the 2015-2024 backtest window, CRSP records ~5,500 delistings
across the full database, of which ~2,700 are bankruptcy-adjacent
(`dlstcd` codes 450, 574, 584, 552, 560, 570, 580-585). The vast
majority of these are micro-caps below our top-1000 universe, but
the methodology correctly handles any that did appear in our
universe.

## Point-in-time fundamentals (reporting lag)

**Status:** Handled.

All Compustat fundamental data is anchored to `rdq` (report date of
quarterly results), not `datadate` (fiscal period end). A Q1 (Mar 31
datadate) earnings filing with rdq=2024-05-08 only becomes available
to the strategy after May 8, 2024 — preventing look-ahead from the
filing-vs-period-end gap.

Verified in:
- `src/axiom_fund/signals/gross_profitability.py` line 121:
  `fund = fund[(fund["rdq"] >= start_ts) & (fund["rdq"] <= end_ts)]`
- `src/axiom_fund/signals/residual_momentum.py` line 166:
  "As-of merge: for each monthly row, get the most-recent ggroup
  with rdq <= month_end_date"

Restatements: Compustat values are as-reported when available; this
matches institutional research practice for undergraduate scope. A
production system would use Compustat point-in-time fundamentals
(WRDS "PIT" subscription) to handle restatements rigorously.

## Universe point-in-time

**Status:** Handled by construction.

Market cap rank, share code, exchange, price threshold, and ADV
threshold all use point-in-time CRSP data on the rebalance date.
Each universe is built from scratch — no "use today's universe and
backtest" shortcut.

`src/axiom_fund/data/universe.py`

## Transaction costs

**Status:** NOT YET BUILT (next pass).

Locked spec includes commission (5 bps), Corwin-Schultz effective
spread proxy (from daily H/L), square-root market impact (κ=0.1
per share fraction of ADV), and 50 bps annualized short borrow.
Per-name application using stock-level ADV and spread proxies.

Current Sharpe (0.77 over 9.8 years on top-1000) is gross of all
of the above. Expected net Sharpe estimate: 0.4-0.55.

## Constraint binding (sanity check)

**Status:** Verified at machine epsilon.

All dollar, beta, and sector neutrality constraints bind to
~1e-13 or smaller across all 116 successful periods in the full
historical backtest. cvxpy CLARABEL solver returns `optimal` status
on every period.

## Beta construction

**Status:** Historical regression, no shrinkage.

Per-asset betas computed via OLS regression of daily excess returns
on market excess return over rolling 252-day window with minimum 60
observations. No Vasicek (1973) shrinkage toward 1 or industry
average — historical betas used directly. Standard for academic
backtests; would add shrinkage in a production system.

`src/axiom_fund/portfolio/betas.py`

## What's NOT in this audit

Items deferred or not addressed:
- Dual-class shares (GOOG/GOOGL etc.) — documented in
  `docs/limitations.md` §12, accepted for current scope
- Sector classification at GICS level 2 vs level 1 — using level 1;
  sensitivity check deferred
- 1.5% position cap at top-100 vs 0.5% at top-1000 — documented in
  Phase 5 commit, acknowledged as "two parameters changed at once,
  cannot cleanly attribute improvement"
- Holding period turnover analysis — deferred until cost model is
  built, so we can compute turnover-cost product per signal