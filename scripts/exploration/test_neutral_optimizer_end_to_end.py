"""End-to-end test of the MVO optimizer with full neutrality constraints.

Pulls real WRDS data and runs the complete pipeline:
  1. Universe + Returns + Fundamentals + FF factors
  2. Three raw signals + alignment + composite alpha
  3. Ledoit-Wolf covariance
  4. Market betas (252-day OLS regression)
  5. Sector classification (gsector from fundamentals)
  6. MVO optimizer with dollar + beta + sector neutrality

Verifies the locked strategy spec works end-to-end on real data and that
all three neutrality constraints bind to ≈ 0 simultaneously.
"""
# ruff: noqa: I001

from __future__ import annotations

import os
import sys

from axiom_fund import _warnings  # noqa: F401

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from axiom_fund.data.ff_factors import FFFactors
from axiom_fund.data.fundamentals import Fundamentals
from axiom_fund.data.returns import ReturnsPanel
from axiom_fund.data.universe import Universe
from axiom_fund.portfolio.betas import compute_betas
from axiom_fund.portfolio.composite import compute_composite_alpha
from axiom_fund.portfolio.covariance import estimate_covariance
from axiom_fund.portfolio.optimizer import optimize_portfolio
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

        print(f"Building universe as of {rebalance_date}...")
        u = Universe(db).as_of(rebalance_date)
        permnos = u.head(100)["permno"].tolist()
        print(f"  Got {len(permnos)} permnos")

        universe_panel = pd.DataFrame({
            "permno": permnos,
            "date": pd.Timestamp(rebalance_date),
        })

        print("Pulling returns...")
        rets = ReturnsPanel(db).fetch(
            permnos=permnos, start_date="2018-09-01", end_date=rebalance_date,
        )
        print(f"  {len(rets):,} rows")

        print("Pulling fundamentals...")
        fund = Fundamentals(db).fetch_quarterly(
            permnos=permnos, start_date="2017-01-01", end_date=rebalance_date,
        )
        print(f"  {len(fund):,} rows")

        print("Pulling FF factors...")
        ff = FFFactors(db).fetch("2018-09-01", rebalance_date)
        print(f"  {len(ff):,} rows")
        print()

        # Compute signals
        print("Computing signals...")
        raw_gp = compute_gross_profitability(
            fundamentals_df=fund, start_date="2018-09-01", end_date=rebalance_date,
        )
        raw_ivol = compute_idiosyncratic_volatility(
            returns_df=rets, ff_factors_df=ff,
            start_date="2019-01-02", end_date=rebalance_date,
        )
        raw_resmom = compute_residual_momentum(
            returns_df=rets, fundamentals_df=fund,
            start_date="2019-06-01", end_date=rebalance_date,
        )

        aligned_gp = align_signal(raw_gp, universe_panel, [rebalance_date])
        aligned_ivol = align_signal(raw_ivol, universe_panel, [rebalance_date])
        aligned_resmom = align_signal(raw_resmom, universe_panel, [rebalance_date])

        composite = compute_composite_alpha(aligned_gp, aligned_ivol, aligned_resmom)
        print(f"  Composite alpha: {len(composite)} rows")

        # Covariance
        print("Computing covariance (252-day window)...")
        cov_window_start = (
            pd.Timestamp(rebalance_date) - pd.DateOffset(days=400)
        ).strftime("%Y-%m-%d")
        cov_rets = ReturnsPanel(db).fetch(
            permnos=permnos, start_date=cov_window_start, end_date=rebalance_date,
        )
        cov_wide = cov_rets.pivot_table(
            index="date", columns="permno", values="ret", aggfunc="last"
        ).iloc[-252:]

        # Compute betas (uses same returns + ff data)
        print("Computing market betas (252-day window)...")
        betas_full = compute_betas(
            returns_df=cov_rets,
            ff_factors_df=ff,
            as_of_date=rebalance_date,
            window=252,
            min_obs=60,
        )
        print(f"  Betas computed: {len(betas_full)} names")
        print(f"  Beta distribution: median={betas_full.median():.3f}, "
              f"min={betas_full.min():.3f}, max={betas_full.max():.3f}")

        # Build sector mapping per permno from fundamentals
        # Use most recent gsector per permno before rebalance_date
        fund_pre = fund[fund["rdq"] <= pd.Timestamp(rebalance_date)].copy()
        sector_map = (
            fund_pre.sort_values("rdq")
            .groupby("permno")["gsector"]
            .last()
        )
        print(f"  Sector classifications: {len(sector_map)} names, "
              f"{sector_map.nunique()} unique sectors")
        print()

        # Find common universe across all data
        composite_permnos = set(composite["permno"])
        cov_permnos = set(cov_wide.columns)
        beta_permnos = set(betas_full.dropna().index)
        sector_permnos = set(sector_map.dropna().index)
        common = sorted(
            composite_permnos & cov_permnos & beta_permnos & sector_permnos
        )
        print(f"Common universe across alpha + covariance + betas + sectors: "
              f"{len(common)} names")
        print()

        # Filter all inputs to common universe
        composite_aligned = composite[composite["permno"].isin(common)].sort_values("permno")
        cov_filtered = cov_wide[common]
        cov_estimate = estimate_covariance(cov_filtered)

        alpha = pd.Series(
            composite_aligned["composite_z"].to_numpy(),
            index=composite_aligned["permno"].to_numpy(),
            name="composite_z",
        ).dropna()

        # Final alignment: same permnos in same order across all 4 inputs
        alpha = alpha.loc[sorted(alpha.index)]
        cov_for_opt = cov_estimate.matrix.loc[alpha.index, alpha.index]
        betas_for_opt = betas_full.loc[alpha.index]
        sectors_for_opt = sector_map.loc[alpha.index].astype(int)

        print(f"Final optimizer inputs: {len(alpha)} names")
        print(f"  Sectors: {sectors_for_opt.nunique()} unique "
              f"({sectors_for_opt.value_counts().to_dict()})")
        print()

        # Optimize with all three neutrality constraints
        print("Running MVO optimizer with FULL NEUTRALITY (dollar + beta + sector)...")
        result = optimize_portfolio(
            alpha=alpha,
            covariance=cov_for_opt,
            risk_aversion=1.0,
            position_cap=0.015,
            gross_cap=1.5,
            betas=betas_for_opt,
            sectors=sectors_for_opt,
            constrain_dollar_neutral=True,
            constrain_beta_neutral=True,
            constrain_sector_neutral=True,
        )
        print()

        print("=" * 70)
        print("Optimization result (locked spec: full neutrality)")
        print("=" * 70)
        print(f"Solver status:        {result.solver_status}")
        print(f"Number of names:      {len(result.weights)}")
        print(f"Long count:           {result.long_count}")
        print(f"Short count:          {result.short_count}")
        print(f"Gross leverage:       {result.gross_leverage:.4f}")
        print(f"Max long weight:      {result.weights.max():.4f}")
        print(f"Min short weight:     {result.weights.min():.4f}")
        print(f"Expected alpha:       {result.expected_alpha:+.4f}")
        print(f"Expected variance:    {result.expected_variance:.6f}")
        print(f"Expected vol (ann):   {np.sqrt(result.expected_variance):.4f}")
        print()
        print("--- Neutrality bindings (should all be ≈ 0) ---")
        print(f"Net exposure (dollar): {result.net_exposure:+.6e}")
        print(f"Portfolio beta:        {result.portfolio_beta:+.6e}")
        print()
        print("Per-sector exposure:")
        for sector, exposure in result.sector_exposures.items():
            print(f"  Sector {sector}: {exposure:+.6e}")

    finally:
        db.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())