"""Apply transaction costs to an existing backtest summary.

Takes a completed gross-of-cost backtest summary (with per-period weights
checkpointed to disk) and produces a net-of-cost return time series by:

  1. For each rebalance date, fetch H/L/volume data for the trailing 90
     days from CRSP
  2. Compute per-name rolling Corwin-Schultz spread, dollar ADV, and
     return volatility
  3. Apply the cost model (commission + spread + impact + borrow) to
     each period's trades
  4. Subtract from the gross return to produce net return

The cost model itself is in `costs.py`. This module is the application
layer that pulls data, applies the model, and produces a comparable
time series.

Design
------
Post-process pattern (not integrated into the backtest engine). The
existing gross backtest is preserved as a baseline; this module adds a
layer on top. This allows easy sensitivity analysis (different κ,
commission, borrow rate) without rerunning the backtest itself.

Returns
-------
For each backtest period:
  - gross_return: original realized return (from backtest_summary)
  - cost_bps: total cost as bps of NAV
  - net_return: gross_return - cost_bps/10000
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
from sqlalchemy import text

from axiom_fund.backtest.costs import (
    _DEFAULT_BORROW_RATE_ANNUAL,
    _DEFAULT_COMMISSION_BPS,
    _DEFAULT_KAPPA,
    _DEFAULT_PERIODS_PER_YEAR,
    compute_period_cost,
    estimate_cs_spread_rolling,
)

_logger = logging.getLogger(__name__)


# Default rolling windows
_DEFAULT_SPREAD_WINDOW: int = 20
_DEFAULT_ADV_WINDOW: int = 20
_DEFAULT_SIGMA_WINDOW: int = 60
# Lookback for the SQL pull (covers the longest rolling window + buffer)
_DEFAULT_DATA_LOOKBACK_DAYS: int = 90


class _DBLike(Protocol):
    """Minimal protocol for objects with a wrds.Connection-like interface."""

    @property
    def engine(self) -> object: ...


# ----------------------------------------------------------------------
# Result dataclass
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class NetReturnPeriod:
    """One period's net return after cost application."""

    rebalance_date: pd.Timestamp
    gross_return: float
    cost_bps: float
    commission_bps: float
    spread_bps: float
    impact_bps: float
    short_borrow_bps: float
    net_return: float
    n_trades: int


# ----------------------------------------------------------------------
# Data fetching
# ----------------------------------------------------------------------


def _fetch_market_data(
    db: _DBLike,
    permnos: list[int],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """Pull H/L/vol/prc/ret from CRSP for a date range.

    Returns a DataFrame with columns: permno, date, prc, askhi, bidlo,
    vol, ret. All prices are abs(prc) to handle the CRSP convention
    where negative prc indicates a bid/ask midpoint estimate.
    """
    if len(permnos) == 0:
        return pd.DataFrame(
            columns=["permno", "date", "prc", "askhi", "bidlo", "vol", "ret"]
        )

    permno_list = "(" + ",".join(str(int(p)) for p in permnos) + ")"
    sql = f"""
        SELECT
            permno::integer AS permno,
            date,
            ABS(prc)::float8 AS prc,
            askhi::float8 AS askhi,
            bidlo::float8 AS bidlo,
            vol::float8 AS vol,
            ret::float8 AS ret
        FROM crsp.dsf
        WHERE permno IN {permno_list}
          AND date >= :start_date
          AND date <= :end_date
          AND prc IS NOT NULL
          AND askhi IS NOT NULL
          AND bidlo IS NOT NULL
          AND vol IS NOT NULL
        ORDER BY permno, date
    """
    params = {
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
    }
    with db.engine.connect() as conn:  # type: ignore[attr-defined]
        df = pd.read_sql(text(sql), conn, params=params)

    df["date"] = pd.to_datetime(df["date"])
    return df


# ----------------------------------------------------------------------
# Per-name rolling statistics
# ----------------------------------------------------------------------


def _compute_per_name_stats_for_date(
    market_df: pd.DataFrame,
    permnos: list[int],
    as_of_date: pd.Timestamp,
    spread_window: int = _DEFAULT_SPREAD_WINDOW,
    adv_window: int = _DEFAULT_ADV_WINDOW,
    sigma_window: int = _DEFAULT_SIGMA_WINDOW,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Compute per-name rolling stats as of `as_of_date`.

    Returns
    -------
    spread_bps : pd.Series
        Corwin-Schultz spread × 10000, rolling mean over spread_window
        days. Indexed by permno.
    adv_dollars : pd.Series
        Average daily dollar volume, rolling mean over adv_window days.
        vol × prc, where vol is in shares.
    sigma_daily : pd.Series
        Daily return std over sigma_window days. Indexed by permno.

    Names with insufficient history return NaN; downstream cost
    aggregator will treat NaN as missing data (effectively zero cost
    contribution from that name's missing channel, which is a small
    conservative bias).
    """
    spreads: dict[int, float] = {}
    advs: dict[int, float] = {}
    sigmas: dict[int, float] = {}

    for permno in permnos:
        sub = market_df[market_df["permno"] == permno].sort_values("date")
        sub = sub[sub["date"] <= as_of_date]

        if len(sub) < max(spread_window, adv_window) // 2:
            spreads[permno] = float("nan")
            advs[permno] = float("nan")
            sigmas[permno] = float("nan")
            continue

        high = sub["askhi"]
        low = sub["bidlo"]
        spread_series = estimate_cs_spread_rolling(
            high=high, low=low, window=spread_window
        )
        spreads[permno] = (
            float(spread_series.iloc[-1]) * 10000.0
            if len(spread_series) > 0 and pd.notna(spread_series.iloc[-1])
            else float("nan")
        )

        # Dollar volume = vol (shares) × prc
        dollar_vol = (sub["vol"] * sub["prc"]).tail(adv_window)
        advs[permno] = float(dollar_vol.mean()) if len(dollar_vol) > 0 else float("nan")

        # Daily return vol
        rets = sub["ret"].tail(sigma_window).dropna()
        sigmas[permno] = float(rets.std()) if len(rets) > 1 else float("nan")

    return (
        pd.Series(spreads, dtype="float64"),
        pd.Series(advs, dtype="float64"),
        pd.Series(sigmas, dtype="float64"),
    )


# ----------------------------------------------------------------------
# Period cost application
# ----------------------------------------------------------------------


def apply_costs_to_period(
    db: _DBLike,
    rebalance_date: pd.Timestamp,
    weights_old: pd.Series,
    weights_new: pd.Series,
    gross_return: float,
    nav: float = 1.0,
    commission_bps: float = _DEFAULT_COMMISSION_BPS,
    kappa: float = _DEFAULT_KAPPA,
    borrow_rate_annual: float = _DEFAULT_BORROW_RATE_ANNUAL,
    periods_per_year: int = _DEFAULT_PERIODS_PER_YEAR,
    data_lookback_days: int = _DEFAULT_DATA_LOOKBACK_DAYS,
) -> NetReturnPeriod:
    """Apply transaction costs for a single backtest period.

    Pulls market data for the relevant permnos, computes rolling stats,
    invokes compute_period_cost, and produces the net-of-cost return.
    """
    # Union of all permnos involved in this trade
    all_permnos: list[int] = sorted(
        {int(p) for p in weights_old.index.union(weights_new.index)}
    )
    if len(all_permnos) == 0:
        return NetReturnPeriod(
            rebalance_date=rebalance_date,
            gross_return=gross_return,
            cost_bps=0.0,
            commission_bps=0.0,
            spread_bps=0.0,
            impact_bps=0.0,
            short_borrow_bps=0.0,
            net_return=gross_return,
            n_trades=0,
        )

    data_start = rebalance_date - pd.Timedelta(days=data_lookback_days)
    market_df = _fetch_market_data(
        db, all_permnos, data_start, rebalance_date
    )

    spread_bps, adv_dollars, sigma_daily = _compute_per_name_stats_for_date(
        market_df=market_df,
        permnos=all_permnos,
        as_of_date=rebalance_date,
    )

    # Defensive fill: replace NaN with sentinel values
    # Missing spread → assume 10 bps (conservative typical large-cap value)
    # Missing ADV → assume infinity (no impact penalty)
    # Missing sigma → assume 0 (no impact penalty)
    spread_bps = spread_bps.fillna(10.0)
    adv_dollars = adv_dollars.fillna(np.inf)
    sigma_daily = sigma_daily.fillna(0.0)

    period_cost = compute_period_cost(
        period_date=rebalance_date,
        weights_old=weights_old,
        weights_new=weights_new,
        spread_bps=spread_bps,
        adv_dollars=adv_dollars,
        sigma_daily=sigma_daily,
        nav=nav,
        commission_bps=commission_bps,
        kappa=kappa,
        borrow_rate_annual=borrow_rate_annual,
        periods_per_year=periods_per_year,
    )

    net_return = gross_return - period_cost.total_return_drag

    return NetReturnPeriod(
        rebalance_date=rebalance_date,
        gross_return=gross_return,
        cost_bps=period_cost.total_bps,
        commission_bps=period_cost.commission_bps,
        spread_bps=period_cost.spread_bps,
        impact_bps=period_cost.impact_bps,
        short_borrow_bps=period_cost.short_borrow_bps,
        net_return=net_return,
        n_trades=period_cost.n_trades,
    )


# ----------------------------------------------------------------------
# Full backtest cost application
# ----------------------------------------------------------------------


def apply_costs_to_backtest(
    db: _DBLike,
    backtest_dir: Path | str,
    nav: float = 1.0,
    commission_bps: float = _DEFAULT_COMMISSION_BPS,
    kappa: float = _DEFAULT_KAPPA,
    borrow_rate_annual: float = _DEFAULT_BORROW_RATE_ANNUAL,
) -> pd.DataFrame:
    """Apply costs across all periods of an existing backtest.

    Reads:
      - {backtest_dir}/backtest_summary.parquet  (gross results)
      - {backtest_dir}/weights_<YYYY-MM-DD>.parquet  (per-period weights)

    Returns a DataFrame indexed by rebalance_date with columns:
        gross_return, cost_bps, commission_bps, spread_bps, impact_bps,
        short_borrow_bps, net_return, n_trades.

    Notes
    -----
    For the first period, weights_old is empty (initial portfolio
    construction). All trades are charged commission + spread + impact.
    """
    backtest_path = Path(backtest_dir)
    summary = pd.read_parquet(backtest_path / "backtest_summary.parquet")

    if not isinstance(summary.index, pd.DatetimeIndex):
        summary.index = pd.to_datetime(summary.index)

    summary = summary.sort_index()
    dates = list(summary.index)

    results: list[dict[str, object]] = []
    weights_old: pd.Series = pd.Series(dtype="float64")

    for idx, rebalance_date in enumerate(dates):
        weights_file = (
            backtest_path
            / f"weights_{rebalance_date.strftime('%Y-%m-%d')}.parquet"
        )
        if not weights_file.exists():
            _logger.warning(
                "Missing weights file for %s, skipping cost calc",
                rebalance_date.strftime("%Y-%m-%d"),
            )
            continue

        weights_new_df = pd.read_parquet(weights_file)
        # weights are stored as a DataFrame with permno index
        # and a single weight column
        if "weight" in weights_new_df.columns:
            weights_new = weights_new_df["weight"]
        else:
            # Pick the only column
            weights_new = weights_new_df.iloc[:, 0]
        weights_new.index = weights_new.index.astype(int)

        period = apply_costs_to_period(
            db=db,
            rebalance_date=pd.Timestamp(rebalance_date),
            weights_old=weights_old,
            weights_new=weights_new,
            gross_return=float(summary.loc[rebalance_date, "realized_return"]),
            nav=nav,
            commission_bps=commission_bps,
            kappa=kappa,
            borrow_rate_annual=borrow_rate_annual,
        )

        results.append({
            "rebalance_date": period.rebalance_date,
            "gross_return": period.gross_return,
            "cost_bps": period.cost_bps,
            "commission_bps": period.commission_bps,
            "spread_bps": period.spread_bps,
            "impact_bps": period.impact_bps,
            "short_borrow_bps": period.short_borrow_bps,
            "net_return": period.net_return,
            "n_trades": period.n_trades,
        })

        weights_old = weights_new

        if (idx + 1) % 10 == 0:
            _logger.info(
                "Cost application progress: %d/%d periods",
                idx + 1,
                len(dates),
            )

    df = pd.DataFrame(results).set_index("rebalance_date").sort_index()
    return df
