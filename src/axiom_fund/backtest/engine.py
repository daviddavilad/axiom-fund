"""Single-period backtest engine.

For a given rebalance date, this engine:
  1. Takes pre-built strategy inputs (alpha, covariance, betas, sectors,
     and the realized returns for the holding period)
  2. Calls the optimizer to produce portfolio weights
  3. Computes the realized buy-and-hold portfolio return over the
     holding period

Architectural pattern (matches the rest of the project)
-------------------------------------------------------
- BacktestPeriodInputs: an immutable container for all pre-built inputs
- run_backtest_period: pure function, no I/O, fully testable without WRDS
- _fetch_period_inputs: thin wrapper that pulls everything from WRDS for
  one rebalance date and returns a BacktestPeriodInputs

Point-in-time correctness
-------------------------
The pure function accepts inputs that have already been filtered to
data available at the rebalance date. The fetcher enforces this by:
  - using as_of_date filtering when calling Universe, signals, betas
  - pulling holding-period returns from a DIFFERENT (forward-looking)
    window than the strategy-input window
  - never letting holding-period returns leak into strategy-input
    computation

Buy-and-hold realized return
----------------------------
For each name with weight w_i at rebalance, the cumulative return over
the holding period is computed as:
    r_i = ∏ over days d in holding period of (1 + r_{i,d}) - 1

The portfolio realized return is then:
    r_portfolio = Σ_i w_i × r_i

Note: this is buy-and-hold (weights fixed at rebalance, prices drift).
NOT the same as Σ over days of w_t' × r_d (that would be daily
rebalancing). Buy-and-hold compounds the within-name performance
through the holding period.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from axiom_fund.portfolio.optimizer import (
    OptimizationResult,
    optimize_portfolio,
)

_DEFAULT_HOLDING_DAYS: int = 21    # ~1 month of trading days
_DEFAULT_RISK_AVERSION: float = 1.0


@dataclass(frozen=True)
class BacktestPeriodInputs:
    """All pre-built inputs needed to run a single rebalance period.

    Attributes
    ----------
    rebalance_date : pd.Timestamp
        The date at which the portfolio is built.
    alpha : pd.Series
        Composite alpha (z-scored) per permno, computed using only data
        available on or before rebalance_date.
    covariance : pd.DataFrame
        Annualized covariance matrix from Ledoit-Wolf, indexed by the
        same permnos as alpha (in same order).
    betas : pd.Series
        Per-asset market beta, indexed by the same permnos as alpha.
    sectors : pd.Series
        Per-asset sector code (GICS gsector), indexed by the same
        permnos as alpha.
    holding_period_returns : pd.DataFrame
        Wide-format daily returns for the holding period AFTER
        rebalance_date. Index = trading dates strictly after
        rebalance_date. Columns = permnos (same set as alpha.index).
    """

    rebalance_date: pd.Timestamp
    alpha: pd.Series
    covariance: pd.DataFrame
    betas: pd.Series
    sectors: pd.Series
    holding_period_returns: pd.DataFrame


@dataclass(frozen=True)
class BacktestPeriodResult:
    """Output of a single backtest period.

    Attributes
    ----------
    rebalance_date : pd.Timestamp
        Start of the holding period (when weights were set).
    holding_period_end : pd.Timestamp
        End of the holding period (last day used in realized return).
    weights : pd.Series
        Portfolio weights at rebalance, indexed by permno.
    realized_return : float
        Gross-of-cost realized portfolio return over the holding period.
    n_names : int
        Number of names in the optimization universe.
    long_count : int
    short_count : int
    gross_leverage : float
    net_exposure : float
    portfolio_beta : float | None
    optimizer_status : str
    """

    rebalance_date: pd.Timestamp
    holding_period_end: pd.Timestamp
    weights: pd.Series
    realized_return: float
    n_names: int
    long_count: int
    short_count: int
    gross_leverage: float
    net_exposure: float
    portfolio_beta: float | None
    optimizer_status: str


def run_backtest_period(
    inputs: BacktestPeriodInputs,
    risk_aversion: float = _DEFAULT_RISK_AVERSION,
    position_cap: float = 0.015,
    constrain_dollar_neutral: bool = True,
    constrain_beta_neutral: bool = True,
    constrain_sector_neutral: bool = True,
    holding_days: int = _DEFAULT_HOLDING_DAYS,
) -> BacktestPeriodResult:
    """Run one rebalance period of the strategy and return realized P&L.

    Parameters
    ----------
    inputs : BacktestPeriodInputs
        Pre-built strategy inputs (alpha, cov, betas, sectors, future
        returns) for the rebalance date.
    risk_aversion : float
        λ parameter passed to the optimizer.
    constrain_dollar_neutral, constrain_beta_neutral, constrain_sector_neutral
        Toggles for each neutrality constraint. Default: all True
        (the locked spec).
    holding_days : int, default 21
        Number of trading days in the holding period. The first
        `holding_days` rows of `holding_period_returns` are used.

    Returns
    -------
    BacktestPeriodResult

    Raises
    ------
    ValueError
        If inputs are misaligned, or the holding period has fewer
        than holding_days available days.
    RuntimeError
        If the optimizer fails to find an optimal solution.
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if not isinstance(inputs, BacktestPeriodInputs):
        raise ValueError(
            f"inputs must be BacktestPeriodInputs, got {type(inputs).__name__}"
        )
    if holding_days < 1:
        raise ValueError(f"holding_days must be >= 1, got {holding_days}")

    # Universe alignment: alpha, covariance, betas, sectors, and the
    # columns of holding_period_returns must all share the same permno set
    alpha_permnos = list(inputs.alpha.index)
    if list(inputs.covariance.index) != alpha_permnos:
        raise ValueError("covariance.index must equal alpha.index")
    if list(inputs.betas.index) != alpha_permnos:
        raise ValueError("betas.index must equal alpha.index")
    if list(inputs.sectors.index) != alpha_permnos:
        raise ValueError("sectors.index must equal alpha.index")
    if set(inputs.holding_period_returns.columns) != set(alpha_permnos):
        raise ValueError(
            "holding_period_returns.columns must equal the alpha permno set"
        )

    # Holding period must have enough days, and must all come strictly
    # after the rebalance date (point-in-time discipline)
    hpr = inputs.holding_period_returns.copy()
    hpr.index = pd.to_datetime(hpr.index)
    if hpr.index.min() <= inputs.rebalance_date:
        raise ValueError(
            f"holding_period_returns must have all dates strictly after "
            f"rebalance_date={inputs.rebalance_date}, but earliest date is "
            f"{hpr.index.min()}"
        )
    if len(hpr) < holding_days:
        raise ValueError(
            f"holding_period_returns has {len(hpr)} rows, "
            f"but holding_days={holding_days} requested"
        )

    # ------------------------------------------------------------------
    # Run optimizer
    # ------------------------------------------------------------------
    opt_result: OptimizationResult = optimize_portfolio(
        alpha=inputs.alpha,
        covariance=inputs.covariance,
        risk_aversion=risk_aversion,
        position_cap=position_cap,
        betas=inputs.betas,
        sectors=inputs.sectors,
        constrain_dollar_neutral=constrain_dollar_neutral,
        constrain_beta_neutral=constrain_beta_neutral,
        constrain_sector_neutral=constrain_sector_neutral,
    )

    # ------------------------------------------------------------------
    # Realized buy-and-hold return over holding period
    # ------------------------------------------------------------------
    holding_window = hpr.iloc[:holding_days]
    holding_period_end = pd.Timestamp(holding_window.index[-1])

    # Reorder columns to match alpha (and weights) ordering
    holding_window = holding_window[alpha_permnos]

    # Per-name cumulative return: ∏(1 + r_d) - 1, treating NaN as 0
    # (a name with NaN returns mid-period is treated as flat for those
    # days; this is conservative but standard for monthly rebalancing)
    one_plus_r = (holding_window.fillna(0.0) + 1.0).cumprod(axis=0)
    cumulative_returns = one_plus_r.iloc[-1] - 1.0  # Series indexed by permno

    weights_arr = opt_result.weights.to_numpy()
    cumret_arr = cumulative_returns.to_numpy()
    realized_return = float(weights_arr @ cumret_arr)

    return BacktestPeriodResult(
        rebalance_date=inputs.rebalance_date,
        holding_period_end=holding_period_end,
        weights=opt_result.weights,
        realized_return=realized_return,
        n_names=len(opt_result.weights),
        long_count=opt_result.long_count,
        short_count=opt_result.short_count,
        gross_leverage=opt_result.gross_leverage,
        net_exposure=opt_result.net_exposure,
        portfolio_beta=opt_result.portfolio_beta,
        optimizer_status=opt_result.solver_status,
    )


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _compute_buy_and_hold_returns(
    holding_period_returns: pd.DataFrame,
    weights: pd.Series,
    holding_days: int,
) -> float:
    """Pure-function helper for testing the realized-return math directly.

    Computes Σ_i w_i × (∏(1 + r_{i,d}) - 1) over the first holding_days
    rows of holding_period_returns.

    Public-ish via the underscore — exposed so tests can verify the math
    independently of the optimizer.
    """
    window = holding_period_returns.iloc[:holding_days]
    aligned = window[list(weights.index)]
    cumret = (aligned.fillna(0.0) + 1.0).cumprod(axis=0).iloc[-1] - 1.0
    return float(weights.to_numpy() @ cumret.to_numpy())


# ============================================================================
# Backtest data cache
# ============================================================================
#
# A multi-period backtest needs to pull each data source once, not once per
# rebalance date. The _BacktestDataCache holds raw data for the full backtest
# window and exposes slicing methods that subset to whatever a particular
# rebalance date needs.
#
# Architecture:
#   - cache.returns_full: long-format returns for ALL permnos in
#     [start - lookback_buffer, end + holding_buffer]
#   - cache.fundamentals_full: similar for fundamentals
#   - cache.ff_full: FF factors over the same window
#   - cache.universe_panel: maps each rebalance_date → set of permnos
#
# This converts O(n_periods) SQL queries into O(1) per data source.


# Database fetcher protocol (avoid hard dep on wrds.Connection in type hints)
from typing import Protocol  # noqa: E402

from axiom_fund.data.ff_factors import FFFactors  # noqa: E402
from axiom_fund.data.fundamentals import Fundamentals  # noqa: E402
from axiom_fund.data.returns import ReturnsPanel  # noqa: E402
from axiom_fund.data.universe import Universe  # noqa: E402


class _DBLike(Protocol):
    """Minimal protocol for objects that can be passed where wrds.Connection is expected."""
    def raw_sql(self, sql: str, params: object = ...) -> pd.DataFrame: ...


@dataclass(frozen=True)
class _BacktestDataCache:
    """Pre-fetched data spanning the entire backtest window.

    All time-indexed dataframes are sliced per-period downstream by
    `_fetch_period_inputs`. Universe panel is rebalance-date keyed.
    """

    rebalance_dates: list[pd.Timestamp]
    universe_panel: pd.DataFrame    # cols: date, permno
    returns_full: pd.DataFrame      # long-format: permno, date, ret
    fundamentals_full: pd.DataFrame # long: permno, rdq, datadate, gsector, ...
    ff_full: pd.DataFrame           # date, mktrf, smb, hml, rf, umd
    permnos_all: list[int]          # union of permnos across all rebalance dates


def _build_cache(
    db: _DBLike,
    rebalance_dates: list[pd.Timestamp],
    universe_size: int = 100,
    lookback_buffer_days: int = 504,  # 2 years for IVol + ResMom + cov windows
    holding_buffer_days: int = 45,    # ~31 trading days post-rebalance
) -> _BacktestDataCache:
    """Pre-fetch all data needed for a backtest covering the given dates.

    This is the heavy SQL-query phase of the backtest. After this returns,
    `_fetch_period_inputs` operates entirely from the in-memory cache.

    Parameters
    ----------
    db
        WRDS database connection.
    rebalance_dates
        Rebalance dates (typically last business day of each month).
    universe_size
        Top-N stocks by market cap to include each rebalance date.
    lookback_buffer_days
        Calendar days of return history to pull before the earliest
        rebalance date. Must cover IVol (60d) + ResMom (252d for residual
        regression) + covariance (252d), so 504 is the safe default.
    holding_buffer_days
        Calendar days of returns to pull AFTER the latest rebalance date,
        for computing holding-period realized P&L.
    """
    if len(rebalance_dates) == 0:
        raise ValueError("rebalance_dates must be non-empty")

    earliest = min(rebalance_dates)
    latest = max(rebalance_dates)

    # Window boundaries
    data_start = earliest - pd.Timedelta(days=lookback_buffer_days)
    data_end = latest + pd.Timedelta(days=holding_buffer_days)
    fundamentals_start = earliest - pd.Timedelta(days=365 * 3)  # 3 years for GP

    # ------------------------------------------------------------------
    # Universe per rebalance date — must call as_of() per date because
    # universe composition changes month to month
    # ------------------------------------------------------------------
    universe_obj = Universe(db)  # type: ignore[arg-type]
    universe_rows: list[pd.DataFrame] = []
    permnos_all: set[int] = set()
    for rdate in rebalance_dates:
        snapshot = universe_obj.as_of(rdate.strftime("%Y-%m-%d")).head(universe_size)
        snapshot = snapshot[["permno"]].copy()
        snapshot["date"] = rdate
        universe_rows.append(snapshot)
        permnos_all.update(int(p) for p in snapshot["permno"].tolist())

    universe_panel = pd.concat(universe_rows, ignore_index=True)
    permnos_list = sorted(permnos_all)

    # ------------------------------------------------------------------
    # Returns: ONE call covering all permnos × full date range
    # ------------------------------------------------------------------
    returns_full = ReturnsPanel(db).fetch(  # type: ignore[arg-type]
        permnos=permnos_list,
        start_date=data_start.strftime("%Y-%m-%d"),
        end_date=data_end.strftime("%Y-%m-%d"),
    )

    # ------------------------------------------------------------------
    # Fundamentals: ONE call
    # ------------------------------------------------------------------
    fundamentals_full = Fundamentals(db).fetch_quarterly(  # type: ignore[arg-type]
        permnos=permnos_list,
        start_date=fundamentals_start.strftime("%Y-%m-%d"),
        end_date=latest.strftime("%Y-%m-%d"),
    )

    # ------------------------------------------------------------------
    # FF factors: ONE call
    # ------------------------------------------------------------------
    ff_full = FFFactors(db).fetch(
        start_date=data_start.strftime("%Y-%m-%d"),
        end_date=latest.strftime("%Y-%m-%d"),
    )

    return _BacktestDataCache(
        rebalance_dates=list(rebalance_dates),
        universe_panel=universe_panel,
        returns_full=returns_full,
        fundamentals_full=fundamentals_full,
        ff_full=ff_full,
        permnos_all=permnos_list,
    )


# ============================================================================
# Per-period input fetcher
# ============================================================================


def _fetch_period_inputs(
    cache: _BacktestDataCache,
    rebalance_date: pd.Timestamp,
    holding_days: int = _DEFAULT_HOLDING_DAYS,
    cov_window_days: int = 252,
    beta_window_days: int = 252,
    beta_min_obs: int = 60,
    signals: list[str] | None = None,
) -> BacktestPeriodInputs | None:
    """Build a BacktestPeriodInputs for one rebalance date by slicing the cache.

    Returns None if any required data is unavailable (e.g., insufficient
    history at the start of the backtest, missing fundamentals for the
    universe, etc.) — caller is expected to skip the period and log.

    Parameters
    ----------
    cache
        Pre-fetched data covering the full backtest window.
    rebalance_date
        The date to build inputs for. Must be in cache.rebalance_dates.
    holding_days, cov_window_days, beta_window_days, beta_min_obs
        Window parameters; defaults match locked spec.
    """
    # Imports needed for this function — all already-shipped pure functions
    from axiom_fund.portfolio.betas import compute_betas
    from axiom_fund.portfolio.composite import compute_composite_alpha
    from axiom_fund.portfolio.covariance import estimate_covariance
    from axiom_fund.signals.alignment import align_signal
    from axiom_fund.signals.gross_profitability import compute_gross_profitability
    from axiom_fund.signals.idiosyncratic_volatility import (
        compute_idiosyncratic_volatility,
    )
    from axiom_fund.signals.pead import compute_pead_signal
    from axiom_fund.signals.residual_momentum import compute_residual_momentum

    # Universe for this rebalance date
    permnos_today = cache.universe_panel.loc[
        cache.universe_panel["date"] == rebalance_date, "permno"
    ].astype(int).tolist()
    if len(permnos_today) == 0:
        return None

    # Slice raw data to today's universe
    rets_universe = cache.returns_full[
        cache.returns_full["permno"].isin(permnos_today)
    ].copy()
    fund_universe = cache.fundamentals_full[
        cache.fundamentals_full["permno"].isin(permnos_today)
    ].copy()

    # Strategy window: returns up to and including rebalance_date
    rets_strategy = rets_universe[
        rets_universe["date"] <= rebalance_date
    ].copy()
    rets_holding = rets_universe[
        rets_universe["date"] > rebalance_date
    ].copy()

    if len(rets_strategy) == 0 or len(rets_holding) == 0:
        return None

    # Fundamentals up to rebalance_date
    fund_strategy = fund_universe[
        fund_universe["rdq"] <= rebalance_date
    ].copy()

    # FF factors up to rebalance_date
    ff_strategy = cache.ff_full[
        cache.ff_full["date"] <= rebalance_date
    ].copy()

    # Universe panel for alignment (one date)
    universe_panel_today = pd.DataFrame({
        "permno": permnos_today,
        "date": rebalance_date,
    })

    rebalance_str = rebalance_date.strftime("%Y-%m-%d")

    # ------------------------------------------------------------------
    # Resolve which signals to compute
    # ------------------------------------------------------------------
    # signals=None means all 4 signals (backwards-compatible default).
    # A list like ["gp", "ivol", "pead"] runs the strategy without ResMom.
    if signals is None:
        active_signals = {"gp", "ivol", "resmom", "pead"}
    else:
        active_signals = set(signals)

    # ------------------------------------------------------------------
    # Compute signals
    # ------------------------------------------------------------------
    # Each signal needs an explicit start_date that respects its lookback
    sig_lookback_start = (rebalance_date - pd.Timedelta(days=400)).strftime("%Y-%m-%d")

    raw_gp = None
    raw_ivol = None
    raw_resmom = None
    raw_pead = None

    if "gp" in active_signals:
        raw_gp = compute_gross_profitability(
            fundamentals_df=fund_strategy,
            start_date=sig_lookback_start,
            end_date=rebalance_str,
        )
    if "ivol" in active_signals:
        raw_ivol = compute_idiosyncratic_volatility(
            returns_df=rets_strategy,
            ff_factors_df=ff_strategy,
            start_date=sig_lookback_start,
            end_date=rebalance_str,
        )
    if "resmom" in active_signals:
        raw_resmom = compute_residual_momentum(
            returns_df=rets_strategy,
            fundamentals_df=fund_strategy,
            start_date=sig_lookback_start,
            end_date=rebalance_str,
        )
    if "pead" in active_signals:
        # PEAD needs ~2.5 years of history (8 quarter lookback + buffer for the
        # trailing std calculation per name). Use a wider lookback.
        pead_lookback_start = (
            rebalance_date - pd.Timedelta(days=900)
        ).strftime("%Y-%m-%d")
        raw_pead_signal = compute_pead_signal(
            fundamentals=fund_strategy,
            start_date=pead_lookback_start,
            end_date=rebalance_str,
        )
        # Adapt PEAD output to the alignment interface
        raw_pead = raw_pead_signal[["permno", "date_filed", "sue"]].rename(
            columns={"sue": "raw_signal"}
        )

    aligned_gp = (
        align_signal(raw_gp, universe_panel_today, [rebalance_str])
        if raw_gp is not None else None
    )
    aligned_ivol = (
        align_signal(raw_ivol, universe_panel_today, [rebalance_str])
        if raw_ivol is not None else None
    )
    aligned_resmom = (
        align_signal(raw_resmom, universe_panel_today, [rebalance_str])
        if raw_resmom is not None else None
    )
    # PEAD: forward-fill with 90-day max age (Hirshleifer et al. 2021 finds
    # drift persists up to ~60-90 days)
    aligned_pead = (
        align_signal(raw_pead, universe_panel_today, [rebalance_str], max_age_days=90)
        if raw_pead is not None else None
    )
    composite = compute_composite_alpha(
        aligned_gp, aligned_ivol, aligned_resmom, aligned_pead
    )
    if len(composite) == 0:
        return None

    # ------------------------------------------------------------------
    # Covariance: 252-day window from strategy returns
    # ------------------------------------------------------------------
    cov_wide = (
        rets_strategy.pivot_table(
            index="date", columns="permno", values="ret", aggfunc="last"
        )
        .iloc[-cov_window_days:]
    )
    if len(cov_wide) < cov_window_days:
        return None

    # ------------------------------------------------------------------
    # Betas: 252-day window
    # ------------------------------------------------------------------
    betas_full = compute_betas(
        returns_df=rets_strategy,
        ff_factors_df=ff_strategy,
        as_of_date=rebalance_str,
        window=beta_window_days,
        min_obs=beta_min_obs,
    )

    # ------------------------------------------------------------------
    # Sectors: most-recent gsector per permno before rebalance_date
    # ------------------------------------------------------------------
    sector_map = (
        fund_strategy.sort_values("rdq")
        .groupby("permno")["gsector"]
        .last()
    )

    # ------------------------------------------------------------------
    # Common universe across all sources
    # ------------------------------------------------------------------
    holding_wide = rets_holding.pivot_table(
        index="date", columns="permno", values="ret", aggfunc="last"
    )
    holding_wide.index = pd.to_datetime(holding_wide.index)

    if len(holding_wide) < holding_days:
        return None

    common = sorted(
        set(composite["permno"])
        & set(cov_wide.columns)
        & set(betas_full.dropna().index)
        & set(sector_map.dropna().index)
        & set(holding_wide.columns)
    )
    if len(common) < 5:
        # Too few names to form a sensible portfolio
        return None

    # Filter and align all inputs to common universe
    composite_aligned = (
        composite[composite["permno"].isin(common)]
        .sort_values("permno")
    )
    alpha = pd.Series(
        composite_aligned["composite_z"].to_numpy(),
        index=composite_aligned["permno"].to_numpy(),
        name="composite_z",
    ).dropna()
    alpha = alpha.loc[sorted(alpha.index)]

    cov_filtered = cov_wide[alpha.index.tolist()]
    cov_estimate = estimate_covariance(cov_filtered)
    cov_for_engine = cov_estimate.matrix.loc[alpha.index, alpha.index]
    betas_for_engine = betas_full.loc[alpha.index]
    sectors_for_engine = sector_map.loc[alpha.index].astype(int)
    hpr_for_engine = holding_wide[alpha.index.tolist()].copy()

    return BacktestPeriodInputs(
        rebalance_date=rebalance_date,
        alpha=alpha,
        covariance=cov_for_engine,
        betas=betas_for_engine,
        sectors=sectors_for_engine,
        holding_period_returns=hpr_for_engine,
    )


# ============================================================================
# Historical backtest runner
# ============================================================================
#
# The runner orchestrates a multi-period backtest:
#   1. Build the cache once (heavy SQL phase)
#   2. For each rebalance date, build period inputs and run the engine
#   3. Persist per-period results to disk (so a crash doesn't lose work)
#   4. Return a DataFrame summarizing the entire backtest


import logging  # noqa: E402
from pathlib import Path  # noqa: E402

_logger = logging.getLogger(__name__)


def monthly_rebalance_dates(
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
) -> list[pd.Timestamp]:
    """Build a list of last-business-day-of-month rebalance dates.

    Returns dates in [start_date, end_date] (inclusive of any month-end
    that falls in the range).
    """
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    # Month-end business days — use BMonthEnd offset (version-agnostic
    # vs. the "BME" string alias which is pandas 2.2+ only)
    return list(pd.date_range(start=start, end=end, freq=pd.offsets.BMonthEnd()))


def run_historical_backtest(
    db: _DBLike,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    universe_size: int = 100,
    risk_aversion: float = _DEFAULT_RISK_AVERSION,
    position_cap: float = 0.015,
    constrain_dollar_neutral: bool = True,
    constrain_beta_neutral: bool = True,
    constrain_sector_neutral: bool = True,
    holding_days: int = _DEFAULT_HOLDING_DAYS,
    cache_dir: Path | str | None = None,
    signals: list[str] | None = None,
) -> pd.DataFrame:
    """Run a multi-period backtest from start_date to end_date.

    Parameters
    ----------
    db : WRDS database connection
    start_date, end_date : Bounds on the rebalance date range. Last
        business day of each month within these bounds is used.
    universe_size : Top-N universe each period (default 100).
    risk_aversion, position_cap, constrain_*, holding_days : passed to run_backtest_period.
    cache_dir : Optional directory to checkpoint per-period results.
        If provided, each period's result is saved as Parquet immediately
        after the period runs.
    signals : Optional list of signal names to include in the composite.
        Valid: any subset of {"gp", "ivol", "resmom", "pead"} that
        contains both gp and ivol and has at least 2 distinct entries.
        Default None means all four signals (backwards-compatible).
        Example: signals=["gp", "ivol", "pead"] runs the no-ResMom variant.

    Returns
    -------
    pd.DataFrame
        Indexed by rebalance_date. Columns: realized_return, n_names,
        long_count, short_count, gross_leverage, net_exposure,
        portfolio_beta, optimizer_status. Skipped periods are NOT in the
        result; check logs for details.
    """
    # ------------------------------------------------------------------
    # Validate signals
    # ------------------------------------------------------------------
    _ALLOWED_SIGNALS = {"gp", "ivol", "resmom", "pead"}
    _REQUIRED_SIGNALS = {"gp", "ivol"}  # composite needs at minimum these two
    if signals is not None:
        invalid = set(signals) - _ALLOWED_SIGNALS
        if invalid:
            raise ValueError(
                f"Unknown signal names: {sorted(invalid)}. "
                f"Allowed: {sorted(_ALLOWED_SIGNALS)}"
            )
        if len(set(signals)) < 2:
            raise ValueError(
                f"signals list must contain at least 2 distinct signals, "
                f"got {sorted(set(signals))}"
            )
        missing_required = _REQUIRED_SIGNALS - set(signals)
        if missing_required:
            raise ValueError(
                f"signals list must include gp and ivol (composite_alpha "
                f"requires these as anchors); missing: {sorted(missing_required)}"
            )

    rebalance_dates = monthly_rebalance_dates(start_date, end_date)
    if len(rebalance_dates) == 0:
        raise ValueError(
            f"No month-end dates found between {start_date} and {end_date}"
        )

    _logger.info(
        "Backtest: %d rebalance dates from %s to %s, universe_size=%d",
        len(rebalance_dates),
        rebalance_dates[0].strftime("%Y-%m-%d"),
        rebalance_dates[-1].strftime("%Y-%m-%d"),
        universe_size,
    )

    # Set up checkpoint dir
    cache_dir_path: Path | None = None
    if cache_dir is not None:
        cache_dir_path = Path(cache_dir)
        cache_dir_path.mkdir(parents=True, exist_ok=True)

    # Build the data cache once
    _logger.info("Pre-fetching data for entire backtest window...")
    cache = _build_cache(
        db,
        rebalance_dates=rebalance_dates,
        universe_size=universe_size,
    )
    _logger.info(
        "Cache built: %d permnos, %d return rows, %d fundamentals rows",
        len(cache.permnos_all),
        len(cache.returns_full),
        len(cache.fundamentals_full),
    )

    # Iterate periods
    results: list[dict[str, object]] = []
    n_succeeded = 0
    n_skipped = 0

    for rdate in rebalance_dates:
        try:
            inputs = _fetch_period_inputs(
                cache, rdate,
                holding_days=holding_days,
                signals=signals,
            )
            if inputs is None:
                _logger.warning(
                    "Skipping %s: insufficient data for inputs",
                    rdate.strftime("%Y-%m-%d"),
                )
                n_skipped += 1
                continue

            result = run_backtest_period(
                inputs=inputs,
                risk_aversion=risk_aversion,
                position_cap=position_cap,
                constrain_dollar_neutral=constrain_dollar_neutral,
                constrain_beta_neutral=constrain_beta_neutral,
                constrain_sector_neutral=constrain_sector_neutral,
                holding_days=holding_days,
            )

            row = {
                "rebalance_date": result.rebalance_date,
                "holding_period_end": result.holding_period_end,
                "realized_return": result.realized_return,
                "n_names": result.n_names,
                "long_count": result.long_count,
                "short_count": result.short_count,
                "gross_leverage": result.gross_leverage,
                "net_exposure": result.net_exposure,
                "portfolio_beta": result.portfolio_beta,
                "optimizer_status": result.optimizer_status,
            }
            results.append(row)

            # Checkpoint: save period weights to disk
            if cache_dir_path is not None:
                weights_file = (
                    cache_dir_path
                    / f"weights_{rdate.strftime('%Y-%m-%d')}.parquet"
                )
                result.weights.to_frame().to_parquet(weights_file)

            n_succeeded += 1
            _logger.info(
                "%s: realized %+.4f%%, %d names, gross %.2f",
                rdate.strftime("%Y-%m-%d"),
                result.realized_return * 100,
                result.n_names,
                result.gross_leverage,
            )

        except Exception as e:
            _logger.warning(
                "Skipping %s: error %s",
                rdate.strftime("%Y-%m-%d"),
                str(e)[:200],
            )
            n_skipped += 1
            continue

    _logger.info(
        "Backtest complete: %d succeeded, %d skipped",
        n_succeeded,
        n_skipped,
    )

    if len(results) == 0:
        raise RuntimeError(
            "All backtest periods skipped — check WRDS access, "
            "data window, and universe_size."
        )

    df = pd.DataFrame(results).set_index("rebalance_date").sort_index()

    # Save aggregated results if cache_dir provided
    if cache_dir_path is not None:
        df.to_parquet(cache_dir_path / "backtest_summary.parquet")

    return df
