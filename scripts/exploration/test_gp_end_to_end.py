"""End-to-end test of the Gross Profitability signal against real WRDS data.

Pulls a small slice of universe + fundamentals, runs the signal, and prints
results for visual inspection.
"""
# ruff: noqa: I001

from __future__ import annotations

import os
import sys

from axiom_fund import _warnings  # noqa: F401

import pandas as pd
from dotenv import load_dotenv

from axiom_fund.data.fundamentals import Fundamentals
from axiom_fund.data.universe import Universe
from axiom_fund.signals.gross_profitability import compute_gross_profitability


def main() -> int:
    load_dotenv()
    username = os.getenv("WRDS_USERNAME")
    if not username:
        print("ERROR: WRDS_USERNAME not set", file=sys.stderr)
        return 1

    import wrds
    db = wrds.Connection(wrds_username=username)

    try:
        # Build a small universe panel: 4 quarter-end dates in 2020
        print("Building universe snapshots for 2020 quarter-ends...")
        u = Universe(db)
        universe_dates = ["2020-01-02", "2020-04-01", "2020-07-01", "2020-10-01"]
        universe_rows = []
        for d in universe_dates:
            snap = u.as_of(d)[["permno"]].copy()
            snap["date"] = pd.Timestamp(d)
            universe_rows.append(snap)
        universe_panel = pd.concat(universe_rows, ignore_index=True)
        print(f"Universe panel: {len(universe_panel):,} rows over {len(universe_dates)} dates")

        # Get the unique permnos for fundamentals fetch
        all_permnos = sorted(universe_panel["permno"].unique().tolist())
        print(f"Distinct PERMNOs across universe panel: {len(all_permnos):,}")
        print()

        # Pull fundamentals for those permnos in 2020
        print("Pulling 2020 fundamentals...")
        f = Fundamentals(db)
        fund_panel = f.fetch_quarterly(
            permnos=all_permnos,
            start_date="2020-01-01",
            end_date="2020-12-31",
        )
        print(f"Fundamentals panel: {len(fund_panel):,} rows")
        print()

        # Compute the signal
        print("Computing Gross Profitability signal...")
        # Stage 1: raw signal
        raw_gp = compute_gross_profitability(
            fundamentals_df=fund_panel,
            start_date="2020-01-01",
            end_date="2020-12-31",
        )
        print(f"Raw signal panel: {len(raw_gp):,} rows")

        # Stage 2: align to monthly rebalance dates, winsorize, z-score
        from axiom_fund.signals.alignment import align_signal

        # Use the universe snapshot dates as rebalance dates for this test
        rebalance_dates = sorted(universe_panel["date"].unique())
        signal = align_signal(
            raw_signal_df=raw_gp,
            universe_df=universe_panel,
            rebalance_dates=rebalance_dates,
            winsorize_pct=0.01,
        )
        print(f"Signal panel: {len(signal):,} rows")
        print()

        # Diagnostic statistics
        print("=" * 70)
        print("Signal diagnostics")
        print("=" * 70)
        print(f"Unique date values: {signal['date'].nunique()}")
        print(f"Unique permnos: {signal['permno'].nunique()}")
        print()

        print("Raw signal distribution (full panel):")
        print(signal["raw_signal"].describe())
        print()

        print("Z-score distribution (should be mean ~0, std ~1 per date):")
        print(signal["z_score"].describe())
        print()

        # Show a sample cross-section
        print("Sample cross-section: first 10 rows of earliest rdq")
        first_rdq = signal["date"].min()
        sample = signal[signal["date"] == first_rdq].head(10)
        print(sample[["date", "permno", "raw_signal", "winsorized", "z_score"]].to_string(index=False))
        print()

        # Top-10 and bottom-10 by z-score across panel
        print("Top 10 z_scores in panel:")
        top10 = signal.nlargest(10, "z_score")
        print(top10[["date", "permno", "raw_signal", "z_score"]].to_string(index=False))
        print()

        print("Bottom 10 z_scores in panel:")
        bot10 = signal.nsmallest(10, "z_score")
        print(bot10[["date", "permno", "raw_signal", "z_score"]].to_string(index=False))

    finally:
        db.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
