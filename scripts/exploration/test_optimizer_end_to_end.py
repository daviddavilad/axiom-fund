"""End-to-end test of the MVO optimizer against real WRDS data.

Pulls a top-100 universe, runs the full Phase 2 + Phase 3 pipeline:
  1. Universe + Returns + Fundamentals + FF factors
  2. Three raw signals → aligned signals → composite alpha
  3. Covariance estimate (Ledoit-Wolf)
  4. MVO optimizer

Verifies the entire chain composes and produces sensible portfolio weights
on real data.
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
        print(f"Got {len(permnos)} permnos")
        print()

        # Universe panel for the alignment layer
        universe_panel = pd.DataFrame({
            "permno": permnos,
            "date": pd.Timestamp(rebalance_date),
        })

        print("Pulling returns (with leading buffer)...")
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
        print("Computing raw signals...")
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
        print(f"  GP: {len(raw_gp):,} | IVol: {len(raw_ivol):,} | ResMom: {len(raw_resmom):,}")
        print()

        # Align all to rebalance date
        print(f"Aligning signals to {rebalance_date}...")
        aligned_gp = align_signal(raw_gp, universe_panel, [rebalance_date])
        aligned_ivol = align_signal(raw_ivol, universe_panel, [rebalance_date])
        aligned_resmom = align_signal(raw_resmom, universe_panel, [rebalance_date])
        print(f"  GP: {len(aligned_gp)} | IVol: {len(aligned_ivol)} | ResMom: {len(aligned_resmom)}")
        print()

        # Composite alpha
        print("Computing composite alpha...")
        composite = compute_composite_alpha(aligned_gp, aligned_ivol, aligned_resmom)
        print(f"  {len(composite)} rows with composite_z")
        print()

        # Covariance — pull 252-day window of returns, pivot to wide
        print("Computing covariance (252-day window)...")
        cov_window_start = (
            pd.Timestamp(rebalance_date) - pd.DateOffset(days=400)
        ).strftime("%Y-%m-%d")
        cov_rets = ReturnsPanel(db).fetch(
            permnos=permnos, start_date=cov_window_start, end_date=rebalance_date,
        )
        cov_wide = cov_rets.pivot_table(
            index="date", columns="permno", values="ret", aggfunc="last"
        )
        # Use only the most recent 252 trading days
        cov_wide = cov_wide.iloc[-252:]
        print(f"  Wide returns: {cov_wide.shape}")

        # Need to subset the universe to permnos that have BOTH composite alpha AND covariance
        common_permnos = sorted(set(composite["permno"]) & set(cov_wide.columns))
        print(f"  Common permnos in alpha and covariance: {len(common_permnos)}")

        # Filter both to common universe
        composite_aligned = composite[composite["permno"].isin(common_permnos)].sort_values("permno")
        cov_filtered = cov_wide[common_permnos]

        cov_estimate = estimate_covariance(cov_filtered)
        print(f"  Covariance shrinkage: {cov_estimate.shrinkage:.4f}")
        print()

        # Build alpha Series (must be aligned to covariance permnos)
        alpha = pd.Series(
            composite_aligned["composite_z"].values,
            index=composite_aligned["permno"].values,
            name="composite_z",
        )
        # Drop any NaNs in alpha (shouldn't be any, but defensive)
        alpha = alpha.dropna()
        # Reindex covariance to alpha's order
        cov_for_opt = cov_estimate.matrix.loc[alpha.index, alpha.index]

        # Optimize
        print("Running MVO optimizer...")
        result = optimize_portfolio(
            alpha=alpha,
            covariance=cov_for_opt,
            risk_aversion=1.0,
            position_cap=0.015,
            gross_cap=1.5,
        )
        print()

        # Results
        print("=" * 70)
        print("Optimization result")
        print("=" * 70)
        print(f"Solver status:        {result.solver_status}")
        print(f"Number of names:      {len(result.weights)}")
        print(f"Long count:           {result.long_count}")
        print(f"Short count:          {result.short_count}")
        print(f"Gross leverage:       {result.gross_leverage:.4f}")
        print(f"Net exposure:         {result.weights.sum():+.4f}")
        print(f"Max long weight:      {result.weights.max():.4f}")
        print(f"Min short weight:     {result.weights.min():.4f}")
        print(f"Expected alpha:       {result.expected_alpha:+.4f}")
        print(f"Expected variance:    {result.expected_variance:.6f}")
        print(f"Expected vol (ann):   {np.sqrt(result.expected_variance):.4f}")
        print()

        # Top longs and shorts
        weights_sorted = result.weights.sort_values()
        print("Top 5 shorts:")
        for permno, w in weights_sorted.head(5).items():
            a = alpha.loc[permno]
            print(f"  permno {permno}: weight={w:+.5f}, composite_z={a:+.3f}")
        print()
        print("Top 5 longs:")
        for permno, w in weights_sorted.tail(5).items():
            a = alpha.loc[permno]
            print(f"  permno {permno}: weight={w:+.5f}, composite_z={a:+.3f}")

    finally:
        db.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
