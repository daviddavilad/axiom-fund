# Axiom Fund — Signal Layer Design

**Author:** David Davila
**Status:** Pre-committed before signal module code is written.
**Purpose:** Lock implementation decisions for Phase 2 (the three signal
modules) so that when we write code, we build to a spec rather than
deriving the spec and the code at the same time.

This document covers **implementation** decisions. Signal **formulas**
live in `strategy_spec.md` §5 ("Alpha signals") and the parameter
registry in §14. Known issues live in `limitations.md`.

---

## 1. Scope

Phase 2 produces three signal modules, one per locked signal:

- `src/axiom_fund/signals/gross_profitability.py` — Novy-Marx (2013),
  (revtq − cogsq) / atq, quarterly-updating
- `src/axiom_fund/signals/idiosyncratic_volatility.py` — Ang, Hodrick,
  Xing, Zhang (2006), daily-updating from rolling residuals
- `src/axiom_fund/signals/residual_momentum.py` — Blitz, Huij, Martens
  (2011), monthly-updating from industry+size-neutralized 12-1 returns

Each module outputs a long-format panel with columns:
`date, permno, raw_signal, winsorized, z_score`.

---

## 2. Locked design decisions

### 2.1 Natural-frequency output (decision A)

Each signal module outputs at its own natural cadence:

- Gross Profitability updates only when new `rdq` is reported
  (quarterly per firm)
- Idiosyncratic Volatility updates every trading day (rolling window
  of daily residuals)
- Residual Momentum updates monthly (formation window is a rolling
  12-month calculation)

Frequency alignment to the monthly rebalance date is a **portfolio
layer** concern, not a signal-module concern. The portfolio layer
will forward-fill quarterly signals until the next `rdq`, and
point-sample daily signals at month-end.

**Rationale:** keeps signal-module code focused on the signal itself
rather than on calendar management. Also makes it trivial to later
experiment with alternative rebalancing cadences (weekly, quarterly)
without rewriting signal code.

### 2.2 Missing values emitted as NaN (decision B)

When a signal cannot be computed for a name on a date (e.g., a PERMNO
without a linked Compustat record cannot have a GP score), the module
emits a row with NaN in the signal columns.

**Rationale:** preserves information about which names actually had
valid signal values on each date. Enables diagnostic questions like
"what fraction of the universe had valid GP signals on date X?" —
exactly the kind of data-quality reporting that matters for defending
the backtest.

Downstream composite logic becomes: for each (date, permno), average
whichever z-scores are non-null, with at least 2 of 3 signals required
for inclusion. See `strategy_spec.md` §6 for the composite rule.

### 2.3 Cross-sectional z-scoring happens in the alignment layer (revised 2026-04-26)

**Original decision (2026-04-26):** signal modules z-score within the
current Universe.

**Amendment (2026-04-26 same day, after integration testing):** the
z-scoring step moves out of signal modules and into a shared alignment
layer (`src/axiom_fund/signals/alignment.py`). Reason: signal modules
emit values keyed by their natural cadence (e.g., GP keyed by rdq),
which produces tiny cross-sections (median 7 names) when z-scoring
per-rdq. The cross-section we actually want is the full investable
universe at each rebalance date.

The alignment layer's responsibility:

1. Takes a raw signal panel (long-format with at minimum
   `permno, date_filed, raw_signal`) and a list of rebalance dates
2. For each rebalance date and each PERMNO in the current Universe,
   finds the most-recent `date_filed` signal value (forward-fill)
3. Winsorizes within Universe at each rebalance date
4. Z-scores within Universe at each rebalance date

This separation means signal modules become pure functions of
"compute the raw signal value per natural reporting unit," and the
alignment layer handles all calendar-and-cross-section logic.

### 2.4 Pure-function interface (decision: raw DataFrames)

Signal modules are pure functions that consume raw DataFrames and
produce a signal panel. They do not depend on the Universe,
ReturnsPanel, or Fundamentals classes directly.

```python
def compute_gross_profitability(
    fundamentals_df: pd.DataFrame,    # from Fundamentals.fetch_quarterly
    start_date: str | date,
    end_date: str | date,
) -> pd.DataFrame:
    """Compute raw GP values per (permno, rdq).

    Returns long-format DataFrame with columns:
        permno, gvkey, date_filed, date_period, revtq, cogsq, atq, raw_signal
    """

def align_signal(
    raw_signal_df: pd.DataFrame,      # from any compute_<signal>() function
    universe_df: pd.DataFrame,        # from Universe.as_of across dates
    rebalance_dates: list[date],      # rebalance calendar
    winsorize_pct: float = 0.01,
) -> pd.DataFrame:
    """Align a raw signal to rebalance dates, winsorize, and z-score
    within the current Universe.

    Returns long-format: date, permno, raw_signal, winsorized, z_score.
    """
```

Analogous signatures for `compute_idiosyncratic_volatility` and
`compute_residual_momentum`.

**Rationale:**
- **Testability:** pure functions accept synthetic DataFrames and
  return deterministic output — no database connection required
- **Caching:** pre-loaded Parquet files work as inputs identically to
  live WRDS data
- **Flexibility:** same signal computation works against alternative
  data sources (e.g., Sharadar) with no code changes
- **Composition:** signal modules can be combined in pipelines or run
  in parallel without coupling

---

## 3. Signal pipeline

The full pipeline now has two stages:

### Stage 1 — Raw signal computation (in each signal module)

For each row in the input fundamentals/returns panel:

1. **Compute** the raw signal value (signal-specific formula)
2. **Output** the raw signal keyed by its natural unit (e.g., per
   `(permno, rdq)` for GP, per `(permno, trading_date)` for IVol)

No universe filter, no winsorize, no z-score. The output is just the
raw signal panel with whatever metadata is useful for diagnostics
(e.g., the input components used to compute it).

### Stage 2 — Alignment to rebalance calendar (in alignment layer)

For each rebalance date:

1. **For each PERMNO in the current Universe**, find the most-recent
   raw signal value with `date_filed <= rebalance_date` (forward-fill)
2. **Winsorize** within the Universe at the 1st/99th cross-sectional
   percentile (configurable via `winsorize_pct`)
3. **Z-score** the winsorized values: subtract cross-sectional mean,
   divide by cross-sectional std

Each step in stage 2 operates **cross-sectionally per rebalance date**
across the Universe. No time-series operations.

The aligned output panel has columns:
`date, permno, raw_signal, winsorized, z_score`

where `date` is the rebalance date and `raw_signal` is the (possibly
forward-filled) value of the raw signal as of that date.

---

## 4. Signal-specific notes

### 4.1 Gross Profitability

- Formula: `(revtq - cogsq) / atq`
- Input: `fundamentals_df` with columns `permno, rdq, revtq, cogsq, atq`
- PIT rule: a GP value with `rdq = 2020-05-05` is visible on dates
  `>= 2020-05-05`, not before
- Forward-fill between `rdq` dates (portfolio layer does this, not
  the signal module — signal module only emits rows on `rdq` dates)

### 4.2 Idiosyncratic Volatility

- Formula: standard deviation of residuals from a daily regression of
  excess returns on the Fama-French 3-factor model, over a rolling
  60-day window (see `strategy_spec.md` §5 for exact specification)
- Input: `returns_df` (daily returns panel) + `ff_factors_df`
  (Fama-French factors from WRDS or Ken French's data library)
- Output: one row per (permno, date) for every trading day where the
  rolling window has ≥ 40 non-missing observations

### 4.3 Residual Momentum

- Formula: 12-1 month returns computed on the residuals of a rolling
  cross-sectional regression of monthly stock returns on industry
  (GICS sector) and size (log market cap) factors
- Input: `returns_df`, `universe_df`, `fundamentals_df` (for industry
  classification via `gsector`)
- Most complex of the three signals: requires (a) computing monthly
  returns, (b) running a 36-month rolling cross-sectional regression,
  (c) extracting residuals, (d) computing 12-1 cumulative residual
  returns, (e) z-scoring
- Output: one row per (permno, month-end date)

---

## 5. Testing requirements

Each signal module must have:
- **Unit tests** with synthetic DataFrames verifying:
  - Raw formula correctness (for known synthetic inputs, output
    matches hand-calculated expectation)
  - Winsorization at correct percentiles
  - Z-score has mean ≈ 0 and std ≈ 1 cross-sectionally per date
  - NaN handling (missing inputs → NaN output, preserved through
    winsorize and z-score steps)
  - Column set matches canonical `date, permno, raw_signal,
    winsorized, z_score`
- **Integration tests** against small real-data slices verifying:
  - Signal values are in sensible ranges (e.g., GP between -0.5 and 2
    for most firms)
  - No forward-looking bias (no rows dated before input availability)
  - Cross-section size matches universe size (modulo missing values)

---

## 6. Cross-references

- **Formulas and parameters:** `strategy_spec.md` §5 ("Alpha signals")
  and §14 ("Locked parameter registry")
- **Known issues:** `limitations.md` §3 (signal decay risks), §5.5
  (CRSP-Compustat linking)
- **Composite construction:** `strategy_spec.md` §6 ("Portfolio
  construction")

---

## 7. Amendments

*(update as design decisions are revisited during build)*