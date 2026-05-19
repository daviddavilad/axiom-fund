"""Run full-sample IC analysis on the 4-signal composite.

For each successful rebalance date in the 4-signal backtest:
  1. Rebuild the composite z-score panel via _fetch_period_inputs
  2. Compute per-signal IC (Spearman) vs 21-day forward returns
  3. Compute pairwise signal correlations

Persists per-period results to parquet for later re-analysis, then
prints aggregate summary tables (overall, by year, by regime).

Runtime: ~30 min (~5 min cache build + ~12 sec/period × 116 periods).
"""
# ruff: noqa: I001

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from axiom_fund import _warnings  # noqa: F401

from dotenv import load_dotenv

from axiom_fund.backtest.engine import _build_cache
from axiom_fund.backtest.forward_returns import compute_forward_returns
from axiom_fund.backtest.ic_analysis import (
    aggregate_ic,
    compute_period_correlations,
    compute_period_ic,
    summarize_correlations,
)
from axiom_fund.portfolio.composite import compute_composite_alpha
from axiom_fund.signals.alignment import align_signal
from axiom_fund.signals.gross_profitability import compute_gross_profitability
from axiom_fund.signals.idiosyncratic_volatility import (
    compute_idiosyncratic_volatility,
)
from axiom_fund.signals.pead import compute_pead_signal
from axiom_fund.signals.residual_momentum import compute_residual_momentum


BACKTEST_DIR = Path("data/cache/backtest_full_top1000_4sig")
ANALYSIS_DIR = Path("data/cache/ic_analysis_4sig")
SIGNAL_COLUMNS = ["z_gp", "z_ivol", "z_resmom", "z_pead"]
HOLDING_DAYS = 21


def _rebuild_composite_for_date(
    cache, rebalance_date: pd.Timestamp,
) -> pd.DataFrame | None:
    """Rebuild composite z-score panel for one date."""
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


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    load_dotenv()
    username = os.getenv("WRDS_USERNAME")
    if not username:
        print("ERROR: WRDS_USERNAME not set", file=sys.stderr)
        return 1

    if not BACKTEST_DIR.exists():
        print(f"ERROR: backtest dir not found: {BACKTEST_DIR}", file=sys.stderr)
        return 1

    summary = pd.read_parquet(BACKTEST_DIR / "backtest_summary.parquet")
    if not isinstance(summary.index, pd.DatetimeIndex):
        summary.index = pd.to_datetime(summary.index)
    summary = summary.sort_index()
    rebalance_dates: list[pd.Timestamp] = [
        pd.Timestamp(d) for d in summary.index
    ]

    print("=" * 70)
    print(f"IC analysis for 4-signal composite over {len(rebalance_dates)} periods")
    print(f"Signals:       {SIGNAL_COLUMNS}")
    print(f"Holding days:  {HOLDING_DAYS}")
    print("=" * 70)

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    start = time.time()

    import wrds
    db = wrds.Connection(wrds_username=username)
    try:
        logging.info("Building cache (this is the slow step)...")
        cache = _build_cache(
            db,
            rebalance_dates=rebalance_dates,
            universe_size=1000,
        )
        logging.info(
            "Cache: %d permnos, %d returns, %d fundamentals",
            len(cache.permnos_all),
            len(cache.returns_full),
            len(cache.fundamentals_full),
        )

        # Compute forward returns once for all periods
        logging.info("Computing forward returns for all periods...")
        fwd = compute_forward_returns(
            cache.returns_full,
            rebalance_dates,
            holding_days=HOLDING_DAYS,
        )
        logging.info("Forward returns: %d rows", len(fwd))

        ic_rows: list[pd.DataFrame] = []
        corr_rows: list[pd.DataFrame] = []

        for idx, rebal in enumerate(rebalance_dates):
            panel = _rebuild_composite_for_date(cache, rebal)
            if panel is None or panel.empty:
                continue

            # IC for this period
            fwd_for_date = fwd[fwd["rebalance_date"] == rebal]
            ic = compute_period_ic(panel, fwd_for_date, SIGNAL_COLUMNS, rebal)
            ic_rows.append(ic)

            # Correlations for this period
            corr = compute_period_correlations(panel, SIGNAL_COLUMNS, rebal)
            corr_rows.append(corr)

            if (idx + 1) % 10 == 0:
                logging.info(
                    "Progress: %d/%d periods",
                    idx + 1,
                    len(rebalance_dates),
                )

    finally:
        db.close()

    if not ic_rows:
        print("No periods produced IC data. Aborting.", file=sys.stderr)
        return 1

    ic_long = pd.concat(ic_rows, ignore_index=True)
    corr_long = pd.concat(corr_rows, ignore_index=True)

    # Persist
    ic_long.to_parquet(ANALYSIS_DIR / "ic_long.parquet")
    corr_long.to_parquet(ANALYSIS_DIR / "corr_long.parquet")
    print(f"Saved ic_long.parquet ({len(ic_long)} rows) and "
          f"corr_long.parquet ({len(corr_long)} rows) to {ANALYSIS_DIR}/")

    # ------------------------------------------------------------------
    # Summary tables
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("OVERALL IC SUMMARY (Spearman rank correlation)")
    print("=" * 70)
    overall = aggregate_ic(ic_long, by="all")
    print(overall.round(4).to_string(index=False))
    print()

    print("=" * 70)
    print("BY-YEAR IC SUMMARY")
    print("=" * 70)
    by_year = aggregate_ic(ic_long, by="year")
    # Reshape for readability: rows = year, columns = signal × stat
    pivot = by_year.pivot(
        index="year", columns="signal", values="mean_ic_spearman"
    ).round(4)
    pivot["n_periods_per_year"] = by_year.groupby("year")["n_periods"].first()
    print(pivot.to_string())
    print()

    # Regime: bull vs bear by year (positive vs negative full-sample year return)
    print("=" * 70)
    print("BY-REGIME IC (bull = year mean IC > 0 across all signals)")
    print("=" * 70)
    ic_long_with_regime = ic_long.copy()
    ic_long_with_regime["year"] = pd.to_datetime(
        ic_long_with_regime["rebalance_date"]
    ).dt.year
    # Mark each period bull/bear by whether the avg IC across signals is +/-
    yearly_avg_ic = ic_long.copy()
    yearly_avg_ic["year"] = pd.to_datetime(
        yearly_avg_ic["rebalance_date"]
    ).dt.year
    yearly_signs = yearly_avg_ic.groupby("year")["ic_spearman"].mean()
    bull_years = set(yearly_signs[yearly_signs > 0].index)
    ic_long_with_regime["regime"] = ic_long_with_regime["year"].apply(
        lambda y: "bull" if y in bull_years else "bear"
    )
    by_regime = aggregate_ic(ic_long_with_regime, by="regime", regime_col="regime")
    print(by_regime.round(4).to_string(index=False))
    print()

    print("=" * 70)
    print("AVERAGE CORRELATION MATRIX (across periods)")
    print("=" * 70)
    mean_corr, std_corr = summarize_correlations(corr_long)
    print("\nMean correlations:")
    print(mean_corr.round(3).to_string())
    print("\nStd of correlations across periods (regime stability):")
    print(std_corr.round(3).to_string())

    elapsed = time.time() - start
    print()
    print(f"Total runtime: {elapsed/60:.1f} minutes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())