"""Per-signal return attribution for the 4-signal composite strategy.

For each rebalance period, runs the optimizer 4 separate times — once
with only z_gp as alpha (others zeroed), once with only z_ivol, etc.
Each single-signal portfolio satisfies the same neutrality and
position-cap constraints as the composite portfolio. Realized return
per signal is computed by applying the single-signal weights to the
holding-period returns that actually occurred.

This is Approach 1 attribution (per the design discussion): the most
rigorous of the available methods. The sum of single-signal returns
will NOT exactly equal the composite return, because:
  (a) the optimizer is nonlinear in alpha (constraints, position caps,
      covariance penalty interact)
  (b) some names will be in the universe for composite but not for a
      single signal (e.g., a name missing GP but with IVol/ResMom/PEAD
      participates in composite but not in the GP-only portfolio)

The interpretation: each single-signal return tells us "what would
the strategy have realized that period if it only had this one signal
to work with, under the same risk constraints."

Runtime: ~30 sec per (period, signal) call → 116 periods × 4 signals
= ~4 hours full run.

References
----------
Grinold, R. C., & Kahn, R. N. (1999). Active Portfolio Management.
    (Chapter 15: performance attribution methods.)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

from axiom_fund.portfolio.betas import compute_betas
from axiom_fund.portfolio.composite import compute_composite_alpha
from axiom_fund.portfolio.covariance import estimate_covariance
from axiom_fund.portfolio.optimizer import optimize_portfolio
from axiom_fund.signals.alignment import align_signal
from axiom_fund.signals.gross_profitability import compute_gross_profitability
from axiom_fund.signals.idiosyncratic_volatility import (
    compute_idiosyncratic_volatility,
)
from axiom_fund.signals.pead import compute_pead_signal
from axiom_fund.signals.residual_momentum import compute_residual_momentum


_logger = logging.getLogger(__name__)


# Signal sign conventions, mirror those in compute_composite_alpha
SIGNAL_SIGNS: Final[dict[str, float]] = {
    "z_gp": 1.0,
    "z_ivol": -1.0,
    "z_resmom": 1.0,
    "z_pead": 1.0,
}

DEFAULT_HOLDING_DAYS: Final[int] = 21
DEFAULT_COV_WINDOW_DAYS: Final[int] = 252
DEFAULT_BETA_WINDOW_DAYS: Final[int] = 252
DEFAULT_BETA_MIN_OBS: Final[int] = 60


# ----------------------------------------------------------------------
# Output dataclasses
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class SingleSignalPeriodResult:
    """Result of a single-signal portfolio run for one rebalance period."""

    rebalance_date: pd.Timestamp
    signal: str
    realized_return: float
    n_names: int
    long_count: int
    short_count: int
    gross_leverage: float
    optimizer_status: str


# ----------------------------------------------------------------------
# Composite + per-signal alignment for one date
# ----------------------------------------------------------------------


def rebuild_composite_for_date(
    cache,
    rebalance_date: pd.Timestamp,
) -> pd.DataFrame | None:
    """Rebuild composite panel with all 4 z-score columns for one date.

    Replicates the signal-and-alignment pipeline from
    `_fetch_period_inputs` in `engine.py` but stops at the composite
    panel (skipping covariance/betas/sectors). Returns the panel with
    'z_gp', 'z_ivol', 'z_resmom', 'z_pead' columns so callers can
    access individual signal values for attribution analysis.

    This is the same helper used by ic_analysis.py — promoted to a
    production module since it's reusable infrastructure.
    """
    permnos_today = cache.universe_panel.loc[
        cache.universe_panel["date"] == rebalance_date, "permno"
    ].astype(int).tolist()
    if not permnos_today:
        return None

    rets_universe = cache.returns_full[
        cache.returns_full["permno"].isin(permnos_today)
    ].copy()
    fund_universe = cache.fundamentals_full[
        cache.fundamentals_full["permno"].isin(permnos_today)
    ].copy()
    rets_strategy = rets_universe[rets_universe["date"] <= rebalance_date].copy()
    if rets_strategy.empty:
        return None
    fund_strategy = fund_universe[fund_universe["rdq"] <= rebalance_date].copy()
    ff_strategy = cache.ff_full[cache.ff_full["date"] <= rebalance_date].copy()

    universe_panel_today = pd.DataFrame({
        "permno": permnos_today,
        "date": rebalance_date,
    })

    sig_lookback_start = (
        rebalance_date - pd.Timedelta(days=400)
    ).strftime("%Y-%m-%d")
    pead_lookback_start = (
        rebalance_date - pd.Timedelta(days=900)
    ).strftime("%Y-%m-%d")
    rebalance_str = rebalance_date.strftime("%Y-%m-%d")

    raw_gp = compute_gross_profitability(
        fundamentals_df=fund_strategy,
        start_date=sig_lookback_start,
        end_date=rebalance_str,
    )
    raw_ivol = compute_idiosyncratic_volatility(
        returns_df=rets_strategy,
        ff_factors_df=ff_strategy,
        start_date=sig_lookback_start,
        end_date=rebalance_str,
    )
    raw_resmom = compute_residual_momentum(
        returns_df=rets_strategy,
        fundamentals_df=fund_strategy,
        start_date=sig_lookback_start,
        end_date=rebalance_str,
    )
    raw_pead_signal = compute_pead_signal(
        fundamentals=fund_strategy,
        start_date=pead_lookback_start,
        end_date=rebalance_str,
    )
    raw_pead = raw_pead_signal[["permno", "date_filed", "sue"]].rename(
        columns={"sue": "raw_signal"}
    )

    aligned_gp = align_signal(raw_gp, universe_panel_today, [rebalance_str])
    aligned_ivol = align_signal(raw_ivol, universe_panel_today, [rebalance_str])
    aligned_resmom = align_signal(raw_resmom, universe_panel_today, [rebalance_str])
    aligned_pead = align_signal(
        raw_pead, universe_panel_today, [rebalance_str], max_age_days=90,
    )
    composite = compute_composite_alpha(
        aligned_gp, aligned_ivol, aligned_resmom, aligned_pead,
    )
    return composite if not composite.empty else None


# ----------------------------------------------------------------------
# Per-period inputs for the optimizer (cov, betas, sectors, holding rets)
# ----------------------------------------------------------------------


def _build_optimizer_inputs(
    cache,
    rebalance_date: pd.Timestamp,
    holding_days: int = DEFAULT_HOLDING_DAYS,
    cov_window_days: int = DEFAULT_COV_WINDOW_DAYS,
    beta_window_days: int = DEFAULT_BETA_WINDOW_DAYS,
    beta_min_obs: int = DEFAULT_BETA_MIN_OBS,
) -> tuple[
    pd.DataFrame,        # covariance (wide DataFrame indexed by permno)
    pd.Series,           # betas (indexed by permno)
    pd.Series,           # sectors (indexed by permno)
    pd.DataFrame,        # holding-period returns wide-format
] | None:
    """Build the non-alpha inputs needed by the optimizer for one date.

    Returns covariance, betas, sectors, and holding returns. Returns
    None if any input is insufficient for the period.
    """
    permnos_today = cache.universe_panel.loc[
        cache.universe_panel["date"] == rebalance_date, "permno"
    ].astype(int).tolist()
    if not permnos_today:
        return None

    rets_universe = cache.returns_full[
        cache.returns_full["permno"].isin(permnos_today)
    ].copy()
    fund_universe = cache.fundamentals_full[
        cache.fundamentals_full["permno"].isin(permnos_today)
    ].copy()
    rets_strategy = rets_universe[rets_universe["date"] <= rebalance_date].copy()
    rets_holding = rets_universe[rets_universe["date"] > rebalance_date].copy()
    if len(rets_strategy) == 0 or len(rets_holding) == 0:
        return None

    fund_strategy = fund_universe[fund_universe["rdq"] <= rebalance_date].copy()
    ff_strategy = cache.ff_full[cache.ff_full["date"] <= rebalance_date].copy()

    rebalance_str = rebalance_date.strftime("%Y-%m-%d")

    # Covariance
    cov_wide = (
        rets_strategy.pivot_table(
            index="date", columns="permno", values="ret", aggfunc="last"
        )
        .iloc[-cov_window_days:]
    )
    if len(cov_wide) < cov_window_days:
        return None

    # Betas
    betas_full = compute_betas(
        returns_df=rets_strategy,
        ff_factors_df=ff_strategy,
        as_of_date=rebalance_str,
        window=beta_window_days,
        min_obs=beta_min_obs,
    )

    # Sectors
    sector_map = (
        fund_strategy.sort_values("rdq")
        .groupby("permno")["gsector"]
        .last()
    )

    # Holding-period returns
    holding_wide = rets_holding.pivot_table(
        index="date", columns="permno", values="ret", aggfunc="last"
    )
    holding_wide.index = pd.to_datetime(holding_wide.index)
    if len(holding_wide) < holding_days:
        return None

    return cov_wide, betas_full, sector_map, holding_wide


# ----------------------------------------------------------------------
# Single-signal portfolio for one rebalance period
# ----------------------------------------------------------------------


def compute_single_signal_period(
    cache,
    rebalance_date: pd.Timestamp,
    signal_name: str,
    composite_panel: pd.DataFrame,
    cov_wide: pd.DataFrame,
    betas_for_engine: pd.Series,
    sectors_for_engine: pd.Series,
    hpr_for_engine: pd.DataFrame,
    holding_days: int = DEFAULT_HOLDING_DAYS,
    risk_aversion: float = 1.0,
    position_cap: float = 0.005,
    gross_cap: float = 1.5,
) -> SingleSignalPeriodResult | None:
    """Run the optimizer for one (period, signal) combination.

    The 'alpha' for this signal is just the column from the composite
    panel, with the sign convention applied (z_ivol is flipped).
    Common universe, covariance, betas, sectors, and holding returns
    are taken from the prepared inputs (to avoid recomputing them
    for each signal).
    """
    if signal_name not in SIGNAL_SIGNS:
        raise ValueError(f"Unknown signal: {signal_name}")
    sign = SIGNAL_SIGNS[signal_name]

    # Extract single-signal z-score, apply sign, drop NaN
    sub = composite_panel[["permno", signal_name]].dropna(subset=[signal_name])
    if len(sub) < 10:
        return None

    alpha = pd.Series(
        sub[signal_name].to_numpy() * sign,
        index=sub["permno"].astype(int).to_numpy(),
        name=signal_name,
    )

    # Intersect with covariance / betas / sectors / holding returns universes
    common = sorted(
        set(alpha.index)
        & set(cov_wide.columns)
        & set(betas_for_engine.dropna().index)
        & set(sectors_for_engine.dropna().index)
        & set(hpr_for_engine.columns)
    )
    if len(common) < 5:
        return None

    alpha = alpha.loc[common]
    # Estimate covariance on this signal's universe, mirroring engine pattern.
    # Calling estimate_covariance on the full unfiltered cov_wide can fail
    # with sparse data; the filtered subset is the engine's approach.
    cov_filtered_returns = cov_wide.loc[:, common]
    cov_estimate_signal = estimate_covariance(cov_filtered_returns)
    cov_filtered = cov_estimate_signal.matrix.loc[common, common]
    betas_aligned = betas_for_engine.loc[common]
    sectors_aligned = sectors_for_engine.loc[common].astype(int)
    hpr_aligned = hpr_for_engine[common].copy()

    # Optimize
    try:
        opt_result = optimize_portfolio(
            alpha=alpha,
            covariance=cov_filtered,
            risk_aversion=risk_aversion,
            position_cap=position_cap,
            gross_cap=gross_cap,
            betas=betas_aligned,
            sectors=sectors_aligned,
            constrain_dollar_neutral=True,
            constrain_beta_neutral=True,
            constrain_sector_neutral=True,
        )
    except Exception as e:
        _logger.warning(
            "%s for %s: optimizer raised: %s",
            rebalance_date.strftime("%Y-%m-%d"), signal_name, str(e)[:100],
        )
        return None

    if opt_result.solver_status != "optimal":
        _logger.warning(
            "%s for %s: optimizer status %s",
            rebalance_date.strftime("%Y-%m-%d"),
            signal_name,
            opt_result.solver_status,
        )

    # Compute realized return: apply single-signal weights to holding returns
    weights = opt_result.weights
    # Compound holding-period returns per name
    hpr_first_n_days = hpr_aligned.iloc[:holding_days].dropna(axis=1, how="all")
    log_rets = np.log1p(hpr_first_n_days.astype("float64"))
    cum_log_rets = log_rets.sum()  # sum across days per name
    name_holding_returns = np.expm1(cum_log_rets)

    # Align weights and returns on common names
    common_names = sorted(set(weights.index) & set(name_holding_returns.index))
    if not common_names:
        return None
    w_aligned = weights.loc[common_names]
    r_aligned = name_holding_returns.loc[common_names]
    realized_return = float((w_aligned * r_aligned).sum())

    return SingleSignalPeriodResult(
        rebalance_date=rebalance_date,
        signal=signal_name,
        realized_return=realized_return,
        n_names=len(common_names),
        long_count=int((weights > 1e-9).sum()),
        short_count=int((weights < -1e-9).sum()),
        gross_leverage=float(weights.abs().sum()),
        optimizer_status=opt_result.solver_status,
    )