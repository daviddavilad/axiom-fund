"""End-to-end test of the single-period backtest engine on real WRDS data.

Pulls real data for rebalance date 2020-09-30, runs the full pipeline,
and computes the realized 21-day buy-and-hold portfolio return.

This is the first time the strategy produces a real realized return.
"""
# ruff: noqa: I001

from __future__ import annotations

import os
import sys

from axiom_fund import _warnings  # noqa: F401

import pandas as pd
from dotenv import load_dotenv

from axiom_fund.backtest.engine import BacktestPeriodInputs, run_backtest_period
from axiom_fund.data.ff_factors import FFFactors
from axiom_fund.data.fundamentals import Fundamentals
from axiom_fund.data.returns import ReturnsPanel
from axiom_fund.data.universe import Universe
from axiom_fund.portfolio.betas import compute_betas
from axiom_fund.portfolio.composite import compute_composite_alpha
from axiom_fund.portfolio.covariance import estimate_covariance
from axiom_fund.signals.alignment import align_signal
from axiom_fund.signals.gross_profitability import compute_gross_profitability
from axiom_fund.signals.idiosyncratic_volatility import (
    compute_idiosyncratic_volatility,
)
from axiom_fund.signals.residual_momentum import compute_residual_momentum


def main() -> int:
    load_dotenv()
    username = os.getenv("WRDS_USERNAME")
    if not username:
        print("ERROR: WRDS_USERNAME not set", file=sys.stderr)
        return 1

    import wrds
    db = wrds.Connection(wrds_username=username)

    try:
        rebalance_date = "2020-09-30"
        rebalance_ts = pd.Timestamp(rebalance_date)
        holding_days = 21

        # Holding period: ~30 calendar days after rebalance to ensure we
        # have at least 21 trading days available
        holding_window_end = (rebalance_ts + pd.Timedelta(days=45)).strftime("%Y-%m-%d")

        print(f"Building universe as of {rebalance_date}...")
        u = Universe(db).as_of(rebalance_date)
        permnos = u.head(100)["permno"].tolist()
        print(f"  Got {len(permnos)} permnos")

        universe_panel = pd.DataFrame({
            "permno": permnos,
            "date": rebalance_ts,
        })

        print("Pulling returns (covariance + signal window)...")
        rets_strategy = ReturnsPanel(db).fetch(
            permnos=permnos, start_date="2018-09-01", end_date=rebalance_date,
        )
        print(f"  Strategy window returns: {len(rets_strategy):,} rows")

        print("Pulling future returns (holding period)...")
        rets_holding = ReturnsPanel(db).fetch(
            permnos=permnos,
            start_date=(rebalance_ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            end_date=holding_window_end,
        )
        print(f"  Holding window returns: {len(rets_holding):,} rows")

        print("Pulling fundamentals + FF factors...")
        fund = Fundamentals(db).fetch_quarterly(
            permnos=permnos, start_date="2017-01-01", end_date=rebalance_date,
        )
        ff = FFFactors(db).fetch("2018-09-01", rebalance_date)
        print(f"  Fundamentals: {len(fund):,} rows  |  FF: {len(ff):,} rows")
        print()

        # Compute signals
        print("Computing signals...")
        raw_gp = compute_gross_profitability(
            fundamentals_df=fund, start_date="2018-09-01", end_date=rebalance_date,
        )
        raw_ivol = compute_idiosyncratic_volatility(
            returns_df=rets_strategy, ff_factors_df=ff,
            start_date="2019-01-02", end_date=rebalance_date,
        )
        raw_resmom = compute_residual_momentum(
            returns_df=rets_strategy, fundamentals_df=fund,
            start_date="2019-06-01", end_date=rebalance_date,
        )

        aligned_gp = align_signal(raw_gp, universe_panel, [rebalance_date])
        aligned_ivol = align_signal(raw_ivol, universe_panel, [rebalance_date])
        aligned_resmom = align_signal(raw_resmom, universe_panel, [rebalance_date])
        composite = compute_composite_alpha(aligned_gp, aligned_ivol, aligned_resmom)
        print(f"  Composite alpha: {len(composite)} rows")

        # Covariance
        cov_wide = rets_strategy.pivot_table(
            index="date", columns="permno", values="ret", aggfunc="last"
        ).iloc[-252:]

        # Betas (uses the same strategy-window returns)
        print("Computing betas (252-day window)...")
        betas_full = compute_betas(
            returns_df=rets_strategy, ff_factors_df=ff,
            as_of_date=rebalance_date, window=252, min_obs=60,
        )

        # Sector classification
        fund_pre = fund[fund["rdq"] <= rebalance_ts].copy()
        sector_map = (
            fund_pre.sort_values("rdq")
            .groupby("permno")["gsector"]
            .last()
        )

        # Find common universe
        common = sorted(
            set(composite["permno"])
            & set(cov_wide.columns)
            & set(betas_full.dropna().index)
            & set(sector_map.dropna().index)
        )

        # Restrict holding-period returns to common universe
        hpr_wide = rets_holding.pivot_table(
            index="date", columns="permno", values="ret", aggfunc="last"
        )
        common = sorted(set(common) & set(hpr_wide.columns))
        print(f"\nCommon universe (alpha + cov + betas + sectors + future rets): "
              f"{len(common)} names")

        # Build aligned inputs
        composite_aligned = (
            composite[composite["permno"].isin(common)].sort_values("permno")
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
        hpr_for_engine = hpr_wide[alpha.index.tolist()].copy()
        hpr_for_engine.index = pd.to_datetime(hpr_for_engine.index)

        # Build inputs
        inputs = BacktestPeriodInputs(
            rebalance_date=rebalance_ts,
            alpha=alpha,
            covariance=cov_for_engine,
            betas=betas_for_engine,
            sectors=sectors_for_engine,
            holding_period_returns=hpr_for_engine,
        )

        # Run period
        print(f"\nRunning {holding_days}-day backtest period from "
              f"{rebalance_date}...\n")
        result = run_backtest_period(inputs, holding_days=holding_days)

        # Display
        print("=" * 70)
        print("Backtest period result")
        print("=" * 70)
        print(f"Rebalance date:       {result.rebalance_date.strftime('%Y-%m-%d')}")
        print(f"Holding period end:   {result.holding_period_end.strftime('%Y-%m-%d')}")
        print(f"N names:              {result.n_names}")
        print(f"Long / Short:         {result.long_count} / {result.short_count}")
        print(f"Gross leverage:       {result.gross_leverage:.4f}")
        print(f"Max long weight:      {result.weights.max():.4f}")
        print(f"Min short weight:     {result.weights.min():.4f}")
        print()
        print("--- Neutrality bindings ---")
        print(f"Net exposure:         {result.net_exposure:+.6e}")
        print(f"Portfolio beta:       {result.portfolio_beta:+.6e}")
        print()
        print("--- REALIZED P&L ---")
        print(f"Realized return:      {result.realized_return:+.4%}")
        print(f"Annualized (×12):     {result.realized_return * 12:+.2%}")
        print(f"Optimizer status:     {result.optimizer_status}")

        # Decompose return into long and short legs for diagnostic
        long_weights = result.weights[result.weights > 1e-10]
        short_weights = result.weights[result.weights < -1e-10]
        long_permnos = long_weights.index.tolist()
        short_permnos = short_weights.index.tolist()

        # Per-name cumulative return over holding period
        holding_window = inputs.holding_period_returns.iloc[:holding_days]
        cumret = (holding_window.fillna(0) + 1).cumprod().iloc[-1] - 1

        long_pnl = float(long_weights.to_numpy() @ cumret.loc[long_permnos].to_numpy())
        short_pnl = float(short_weights.to_numpy() @ cumret.loc[short_permnos].to_numpy())
        print()
        print("--- Long/Short decomposition ---")
        print(f"Long leg P&L:         {long_pnl:+.4%}  ({len(long_weights)} names)")
        print(f"Short leg P&L:        {short_pnl:+.4%}  ({len(short_weights)} names)")
        print(f"Total (long + short): {long_pnl + short_pnl:+.4%}")
        print()

        # Top contributors and detractors
        contributions = result.weights * cumret
        contributions = contributions.sort_values()
        print("Top 5 detractors (most negative contribution):")
        for permno, contrib in contributions.head(5).items():
            print(f"  permno {permno}: weight={result.weights[permno]:+.4f}, "
                  f"return={cumret[permno]:+.4%}, contrib={contrib:+.4%}")
        print()
        print("Top 5 contributors (most positive contribution):")
        for permno, contrib in contributions.tail(5).items():
            print(f"  permno {permno}: weight={result.weights[permno]:+.4f}, "
                  f"return={cumret[permno]:+.4%}, contrib={contrib:+.4%}")

    finally:
        db.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
