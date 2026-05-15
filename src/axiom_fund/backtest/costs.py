"""Transaction cost model for backtest evaluation.

Computes per-trade transaction costs for a portfolio rebalance event,
using a four-component model that combines commission, bid-ask spread,
market impact, and short borrow. All functions are pure — they take
trade and market data and return cost in dollars or bps without I/O.

The cost model
--------------
For a rebalance event at time t with weight changes Δw_i:

    cost_i = trade_dollars_i × (commission_bps + spread_bps_i/2 +
                                 impact_bps_i) / 10000
           + short_borrow_drag_i

where trade_dollars_i = |Δw_i| × portfolio_NAV.

The /2 on spread reflects that we cross the half-spread on each trade.
The impact_bps is per-trade, not annualized.

The short borrow drag is computed separately as a continuous fee on
the short position size (not on trades):

    short_borrow_drag_i = max(0, -w_i) × portfolio_NAV ×
                          borrow_rate_annual / periods_per_year

Locked parameters per the strategy spec:
  - commission_bps:  5  (per side)
  - kappa:           0.1  (square-root impact coefficient)
  - borrow_rate:     0.005 (50 bps/year on shorts)
  - periods_per_year: 12  (monthly rebalance)

Spread estimation
-----------------
We use Corwin-Schultz (2012) which estimates effective spread from
daily high-low ranges. This avoids needing actual bid-ask quote data.

The estimator:
    β = E[(ln(H_t/L_t))² + (ln(H_{t+1}/L_{t+1}))²]
    γ = (ln(max(H_t, H_{t+1}) / min(L_t, L_{t+1})))²
    α = (√(2β) - √β) / (3 - 2√2) - √(γ / (3 - 2√2))
    spread = 2 × (exp(α) - 1) / (1 + exp(α))

Negative single-day estimates are clipped to zero (per the paper).
We average over a rolling window (default 20 days) for stability.

Market impact
-------------
Square-root impact model (Almgren et al. 2005):

    impact_bps = kappa × sigma_daily × sqrt(trade_dollars / ADV_dollars) × 10000

where:
  - kappa is a fitted constant (~0.1 in academic literature)
  - sigma_daily is daily return vol (rolling 60-day std)
  - ADV_dollars is average daily dollar volume (rolling 20-day mean)
  - trade_dollars is absolute change in dollar position

References
----------
Corwin, S. A., & Schultz, P. (2012). "A Simple Way to Estimate Bid-Ask
    Spreads from Daily High and Low Prices." Journal of Finance, 67(2).
Almgren, R., Thum, C., Hauptmann, E., & Li, H. (2005). "Direct Estimation
    of Equity Market Impact." Risk, 18(7).
Shumway, T. (1997). "The Delisting Bias in CRSP Data." Journal of Finance,
    52(1). (Not directly used here, but referenced in returns.py.)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Locked parameter defaults per strategy spec
_DEFAULT_COMMISSION_BPS: float = 5.0
_DEFAULT_KAPPA: float = 0.1
_DEFAULT_BORROW_RATE_ANNUAL: float = 0.005  # 50 bps/year
_DEFAULT_PERIODS_PER_YEAR: int = 12

# Corwin-Schultz constants (from the paper)
# 3 - 2√2 ≈ 0.17157
_CS_DENOM: float = 3.0 - 2.0 * np.sqrt(2.0)


# ----------------------------------------------------------------------
# Result dataclasses
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class PerNameCostBreakdown:
    """Cost components for a single name's trade.

    All values in dollars. Sum of (commission_dollars + spread_dollars +
    impact_dollars) gives the per-trade cost; short_borrow_dollars is
    additional drag on holding (not trading) the short position.
    """

    permno: int
    trade_dollars: float
    commission_dollars: float
    spread_dollars: float
    impact_dollars: float
    short_borrow_dollars: float
    total_dollars: float


@dataclass(frozen=True)
class PeriodCostResult:
    """Aggregate cost for a single rebalance period.

    Attributes
    ----------
    period_date : pd.Timestamp
        The rebalance date this cost applies to.
    nav : float
        Portfolio NAV used as the base for percent calculations.
    commission_bps : float
        Total commission cost as bps of NAV.
    spread_bps : float
        Total spread cost as bps of NAV.
    impact_bps : float
        Total market impact cost as bps of NAV.
    short_borrow_bps : float
        Total short borrow cost as bps of NAV (for this period only,
        i.e., monthly drag, not annualized).
    total_bps : float
        Sum of all components.
    total_return_drag : float
        Same as total_bps but as a decimal return (e.g., 0.0030 = 30bps).
    n_trades : int
        Number of names with non-zero trade size.
    """

    period_date: pd.Timestamp
    nav: float
    commission_bps: float
    spread_bps: float
    impact_bps: float
    short_borrow_bps: float
    total_bps: float
    total_return_drag: float
    n_trades: int


# ----------------------------------------------------------------------
# Corwin-Schultz spread estimation
# ----------------------------------------------------------------------


def estimate_cs_spread_daily(
    high: pd.Series,
    low: pd.Series,
) -> pd.Series:
    """Compute the daily Corwin-Schultz spread estimate.

    Parameters
    ----------
    high : pd.Series
        Daily high prices for one stock, indexed by date (sorted).
    low : pd.Series
        Daily low prices for one stock, indexed by date (sorted).

    Returns
    -------
    pd.Series
        Estimated effective spread for each day t (using day t and day t+1).
        The last day has NaN since it needs day t+1 data.
        Negative estimates are clipped to zero per the paper.
    """
    if len(high) != len(low):
        raise ValueError("high and low must have same length")
    if len(high) < 2:
        return pd.Series(dtype=float, index=high.index)

    # Log of intra-day H/L for each day. Wrap in pd.Series explicitly
    # because np.log on Series → Series in practice but mypy infers
    # ndarray via numpy stubs and loses the .shift() method.
    hl_log: pd.Series = pd.Series(np.log(high / low), index=high.index)
    hl_log_sq: pd.Series = hl_log ** 2

    # beta_t = (ln(H_t/L_t))² + (ln(H_{t+1}/L_{t+1}))²
    beta = hl_log_sq + hl_log_sq.shift(-1)

    # 2-day H and L
    h_2day = pd.concat([high, high.shift(-1)], axis=1).max(axis=1)
    l_2day = pd.concat([low, low.shift(-1)], axis=1).min(axis=1)
    gamma = np.log(h_2day / l_2day) ** 2

    # alpha
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / _CS_DENOM - np.sqrt(
        gamma / _CS_DENOM
    )

    # spread = 2 × (exp(α) - 1) / (1 + exp(α))
    spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))

    # Clip negative estimates to zero (per the paper)
    spread = spread.clip(lower=0)

    return pd.Series(spread, index=high.index)


def estimate_cs_spread_rolling(
    high: pd.Series,
    low: pd.Series,
    window: int = 20,
) -> pd.Series:
    """Rolling-window average of daily Corwin-Schultz spread estimates.

    Smooths noise in the daily estimates. The result for date t is the
    average over the trailing `window` days.

    Parameters
    ----------
    high, low : pd.Series
        Daily high/low for one stock, sorted by date.
    window : int
        Rolling window size in trading days.

    Returns
    -------
    pd.Series
        Smoothed spread estimate for each date. NaN for the first
        (window - 1) days.
    """
    daily = estimate_cs_spread_daily(high, low)
    return daily.rolling(window=window, min_periods=max(window // 2, 5)).mean()


# ----------------------------------------------------------------------
# Market impact
# ----------------------------------------------------------------------


def compute_market_impact_bps(
    trade_dollars: float,
    adv_dollars: float,
    sigma_daily: float,
    kappa: float = _DEFAULT_KAPPA,
) -> float:
    """Square-root market impact estimate in bps.

    impact_bps = kappa × sigma_daily × sqrt(trade_dollars / adv_dollars) × 10000

    Parameters
    ----------
    trade_dollars : float
        Absolute dollar value of the trade.
    adv_dollars : float
        Average daily dollar volume for the stock.
    sigma_daily : float
        Daily return volatility (e.g., 0.02 for 2%).
    kappa : float
        Impact coefficient (default 0.1 per Almgren).

    Returns
    -------
    float
        Estimated round-trip impact in basis points.
    """
    if trade_dollars <= 0 or adv_dollars <= 0:
        return 0.0
    participation = trade_dollars / adv_dollars
    impact_bps = kappa * sigma_daily * np.sqrt(participation) * 10000.0
    return float(impact_bps)


# ----------------------------------------------------------------------
# Short borrow drag
# ----------------------------------------------------------------------


def compute_short_borrow_dollars(
    position_dollars: float,
    borrow_rate_annual: float = _DEFAULT_BORROW_RATE_ANNUAL,
    periods_per_year: int = _DEFAULT_PERIODS_PER_YEAR,
) -> float:
    """Periodic short borrow cost on a single position.

    Returns 0 for long positions. For shorts, returns the borrow cost
    over one rebalance period (e.g., one month for monthly rebalancing).

    Parameters
    ----------
    position_dollars : float
        Signed dollar position (negative for shorts).
    borrow_rate_annual : float
        Annual borrow rate (e.g., 0.005 for 50bps/year).
    periods_per_year : int
        Number of periods per year (12 for monthly).

    Returns
    -------
    float
        Borrow cost in dollars for this period.
    """
    if position_dollars >= 0:
        return 0.0
    short_size = -position_dollars  # positive number
    period_rate = borrow_rate_annual / periods_per_year
    return float(short_size * period_rate)


# ----------------------------------------------------------------------
# Per-name trade cost (commission + spread + impact only; not borrow)
# ----------------------------------------------------------------------


def compute_per_name_trade_cost(
    permno: int,
    trade_dollars: float,
    spread_bps_estimate: float,
    adv_dollars: float,
    sigma_daily: float,
    new_position_dollars: float,
    commission_bps: float = _DEFAULT_COMMISSION_BPS,
    kappa: float = _DEFAULT_KAPPA,
    borrow_rate_annual: float = _DEFAULT_BORROW_RATE_ANNUAL,
    periods_per_year: int = _DEFAULT_PERIODS_PER_YEAR,
) -> PerNameCostBreakdown:
    """Compute the full cost breakdown for a single name.

    Combines commission, spread, market impact, and (for short positions)
    short borrow drag for the current period.

    Parameters
    ----------
    permno : int
        CRSP PERMNO identifier (for tracking only).
    trade_dollars : float
        Absolute dollar value of this trade (|Δw_i| × NAV).
    spread_bps_estimate : float
        Estimated half-spread × 2 = effective spread in bps for this name.
    adv_dollars : float
        Average daily dollar volume for impact calculation.
    sigma_daily : float
        Daily return volatility for impact calculation.
    new_position_dollars : float
        Signed dollar position after rebalance (negative for shorts).
    commission_bps : float
        Flat commission per side.
    kappa : float
        Impact coefficient.
    borrow_rate_annual : float
        Annual borrow rate.
    periods_per_year : int
        For per-period borrow calculation.

    Returns
    -------
    PerNameCostBreakdown
        All cost components in dollars, plus the total.
    """
    if trade_dollars < 0:
        raise ValueError("trade_dollars must be non-negative (absolute value)")

    # Commission: flat bps on the trade
    commission_dollars = trade_dollars * commission_bps / 10000.0

    # Spread: half-spread crossed per trade direction; full spread for
    # entry and exit. Here we model entry only, so cost is half_spread.
    spread_dollars = trade_dollars * (spread_bps_estimate / 2.0) / 10000.0

    # Market impact
    impact_bps = compute_market_impact_bps(
        trade_dollars=trade_dollars,
        adv_dollars=adv_dollars,
        sigma_daily=sigma_daily,
        kappa=kappa,
    )
    impact_dollars = trade_dollars * impact_bps / 10000.0

    # Short borrow on the new position (not the trade)
    short_borrow_dollars = compute_short_borrow_dollars(
        position_dollars=new_position_dollars,
        borrow_rate_annual=borrow_rate_annual,
        periods_per_year=periods_per_year,
    )

    total = (
        commission_dollars
        + spread_dollars
        + impact_dollars
        + short_borrow_dollars
    )

    return PerNameCostBreakdown(
        permno=permno,
        trade_dollars=trade_dollars,
        commission_dollars=commission_dollars,
        spread_dollars=spread_dollars,
        impact_dollars=impact_dollars,
        short_borrow_dollars=short_borrow_dollars,
        total_dollars=total,
    )


# ----------------------------------------------------------------------
# Period-level aggregator
# ----------------------------------------------------------------------


def compute_period_cost(
    period_date: pd.Timestamp,
    weights_old: pd.Series,
    weights_new: pd.Series,
    spread_bps: pd.Series,
    adv_dollars: pd.Series,
    sigma_daily: pd.Series,
    nav: float = 1.0,
    commission_bps: float = _DEFAULT_COMMISSION_BPS,
    kappa: float = _DEFAULT_KAPPA,
    borrow_rate_annual: float = _DEFAULT_BORROW_RATE_ANNUAL,
    periods_per_year: int = _DEFAULT_PERIODS_PER_YEAR,
) -> PeriodCostResult:
    """Aggregate cost for a single rebalance period.

    For each name in the union of old + new weights:
      - Compute trade_dollars = |w_new - w_old| × NAV
      - Apply commission, spread, impact per-name
      - Add short borrow drag for names with w_new < 0

    Sums to total period cost. Returns both dollar and bps figures.

    Parameters
    ----------
    period_date : pd.Timestamp
    weights_old : pd.Series
        Portfolio weights before rebalance, indexed by permno.
    weights_new : pd.Series
        Portfolio weights after rebalance, indexed by permno.
    spread_bps : pd.Series
        Effective spread in bps per permno (from Corwin-Schultz rolling).
    adv_dollars : pd.Series
        Average daily dollar volume per permno.
    sigma_daily : pd.Series
        Daily return volatility per permno.
    nav : float
        Portfolio NAV; defaults to 1.0 (i.e., weights expressed as
        fractions of NAV). The result is in same dollar units as NAV.
    commission_bps, kappa, borrow_rate_annual, periods_per_year : float / int
        Cost model parameters; see compute_per_name_trade_cost.

    Returns
    -------
    PeriodCostResult
        Aggregate cost breakdown for this period.
    """
    # Align on union of permnos
    all_permnos = weights_old.index.union(weights_new.index)
    w_old = weights_old.reindex(all_permnos, fill_value=0.0)
    w_new = weights_new.reindex(all_permnos, fill_value=0.0)
    trade_size = (w_new - w_old).abs() * nav

    # Initialize totals
    commission_total = 0.0
    spread_total = 0.0
    impact_total = 0.0
    borrow_total = 0.0
    n_trades = 0

    for permno in all_permnos:
        trade_d = float(trade_size.loc[permno])
        if trade_d <= 0 and w_new.loc[permno] >= 0:
            # No trade and not short, skip
            continue

        breakdown = compute_per_name_trade_cost(
            permno=int(permno),
            trade_dollars=trade_d,
            spread_bps_estimate=float(spread_bps.get(permno, 0.0)),
            adv_dollars=float(adv_dollars.get(permno, np.inf)),
            sigma_daily=float(sigma_daily.get(permno, 0.0)),
            new_position_dollars=float(w_new.loc[permno]) * nav,
            commission_bps=commission_bps,
            kappa=kappa,
            borrow_rate_annual=borrow_rate_annual,
            periods_per_year=periods_per_year,
        )
        commission_total += breakdown.commission_dollars
        spread_total += breakdown.spread_dollars
        impact_total += breakdown.impact_dollars
        borrow_total += breakdown.short_borrow_dollars
        if trade_d > 0:
            n_trades += 1

    total = commission_total + spread_total + impact_total + borrow_total
    total_bps = (total / nav) * 10000.0
    total_drag = total / nav

    return PeriodCostResult(
        period_date=period_date,
        nav=nav,
        commission_bps=(commission_total / nav) * 10000.0,
        spread_bps=(spread_total / nav) * 10000.0,
        impact_bps=(impact_total / nav) * 10000.0,
        short_borrow_bps=(borrow_total / nav) * 10000.0,
        total_bps=total_bps,
        total_return_drag=total_drag,
        n_trades=n_trades,
    )
