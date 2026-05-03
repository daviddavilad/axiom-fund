"""End-to-end test of composite alpha against real WRDS data.

Pulls a top-100 universe over a multi-month window, runs all three signal
modules, aligns each via the alignment layer to monthly rebalance dates,
then runs the composite. Verifies the full Phase 2 + composite pipeline
composes correctly and produces sensible cross-sectional alpha values.
"""
# ruff: noqa: I001

from __future__ import annotations

import os
import sys

from axiom_fund import _warnings  # noqa: F401

import pandas as pd
from dotenv import load_dotenv

from axiom_fund.data.ff_factors import FFFactors
from axiom_fund.data.fundamentals import Fundamentals
from axiom_fund.data.returns import ReturnsPanel
from axiom_fund.data.universe import Universe
from axiom_fund.signals.alignment import align_signal
from axiom_fund.signals.gross_profitability import compute_gross_profitability
from axiom_fund.signals.idiosyncratic_volatility import (
    compute_idiosyncratic_volatility,
)
from axiom_fund.signals.residual_momentum import compute_residual_momentum
from axiom_fund.portfolio.composite import compute_composite_alpha


def main() -> int:
    load_dotenv()
    username = os.getenv("WRDS_USERNAME")
    if not username:
        print("ERROR: WRDS_USERNAME not set", file=sys.stderr)
        return 1

    import wrds
    db = wrds.Connection(wrds_username=username)

    try:
        # Build universe panel for 4 quarter-end dates in 2020
        rebalance_dates = [
            "2020-01-31", "2020-04-30", "2020-07-31", "2020-10-30",
        ]
        print("Building universe panel for 4 monthly rebalance dates...")
        u = Universe(db)
        universe_rows = []
        all_permnos = set()
        for d in rebalance_dates:
            snap = u.as_of(d).head(100)[["permno"]].copy()
            snap["date"] = pd.Timestamp(d)
            universe_rows.append(snap)
            all_permnos.update(snap["permno"].tolist())
        universe_panel = pd.concat(universe_rows, ignore_index=True)
        permnos = sorted(all_permnos)
        print(f"Universe panel: {len(universe_panel):,} rows, {len(permnos)} unique permnos")
        print()

        # Pull all data with leading buffers for the signals' lookback windows
        print("Pulling daily returns (with 14-month buffer)...")
        rp = ReturnsPanel(db)
        rets = rp.fetch(
            permnos=permnos, start_date="2018-10-01", end_date="2020-10-30"
        )
        print(f"  {len(rets):,} rows")

        print("Pulling fundamentals...")
        f = Fundamentals(db)
        fund = f.fetch_quarterly(
            permnos=permnos, start_date="2017-01-01", end_date="2020-10-30"
        )
        print(f"  {len(fund):,} rows")

        print("Pulling FF factors...")
        ff = FFFactors(db).fetch("2018-10-01", "2020-10-30")
        print(f"  {len(ff):,} rows")
        print()

        # Compute raw signals
        print("Computing raw signals...")
        raw_gp = compute_gross_profitability(
            fundamentals_df=fund,
            start_date="2018-10-01",
            end_date="2020-10-30",
        )
        print(f"  GP raw: {len(raw_gp):,} rows")

        raw_ivol = compute_idiosyncratic_volatility(
            returns_df=rets, ff_factors_df=ff,
            start_date="2019-01-02", end_date="2020-10-30",
        )
        print(f"  IVol raw: {len(raw_ivol):,} rows")

        raw_resmom = compute_residual_momentum(
            returns_df=rets, fundamentals_df=fund,
            start_date="2019-06-01", end_date="2020-10-30",
        )
        print(f"  ResMom raw: {len(raw_resmom):,} rows")
        print()

        # Align each signal to rebalance dates
        print("Aligning all 3 signals to rebalance dates...")
        aligned_gp = align_signal(
            raw_signal_df=raw_gp.rename(columns={}),
            universe_df=universe_panel,
            rebalance_dates=rebalance_dates,
        )
        # ResMom signal output uses date_filed already
        aligned_ivol = align_signal(
            raw_signal_df=raw_ivol,
            universe_df=universe_panel,
            rebalance_dates=rebalance_dates,
        )
        aligned_resmom = align_signal(
            raw_signal_df=raw_resmom,
            universe_df=universe_panel,
            rebalance_dates=rebalance_dates,
        )
        print(f"  Aligned GP:     {len(aligned_gp):,} rows")
        print(f"  Aligned IVol:   {len(aligned_ivol):,} rows")
        print(f"  Aligned ResMom: {len(aligned_resmom):,} rows")
        print()

        # Compute composite alpha
        print("Computing composite alpha...")
        composite = compute_composite_alpha(aligned_gp, aligned_ivol, aligned_resmom)
        print(f"  Composite: {len(composite):,} rows")
        print()

        # Diagnostics
        print("=" * 70)
        print("Composite distribution by date")
        print("=" * 70)
        for d, group in composite.groupby("date"):
            print(
                f"\n{d.strftime('%Y-%m-%d')}: {len(group)} stocks"
                f"  | composite_z mean: {group['composite_z'].mean():.4f}"
                f"  | std: {group['composite_z'].std():.4f}"
            )
            print(f"  n_signals breakdown:")
            print(group["n_signals"].value_counts().sort_index().to_string())
            print(f"  composite_z range: [{group['composite_z'].min():.3f}, "
                  f"{group['composite_z'].max():.3f}]")

        print()
        print("=" * 70)
        print("Sample top-5 and bottom-5 by composite_z (latest rebalance date)")
        print("=" * 70)
        latest_date = composite["date"].max()
        latest = composite[composite["date"] == latest_date].sort_values("composite_z")
        print(f"\nLatest date: {latest_date.strftime('%Y-%m-%d')}")
        print("\nBottom 5 (likely shorts):")
        print(latest.head(5)[["permno", "z_gp", "z_ivol", "z_resmom", "composite_z"]].to_string(index=False))
        print("\nTop 5 (likely longs):")
        print(latest.tail(5)[["permno", "z_gp", "z_ivol", "z_resmom", "composite_z"]].to_string(index=False))

    finally:
        db.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())