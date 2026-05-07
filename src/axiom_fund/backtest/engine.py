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
