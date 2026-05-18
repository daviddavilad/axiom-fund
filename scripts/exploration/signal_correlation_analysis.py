"""Signal correlation and IC analysis for the 4-signal composite.

Re-runs _fetch_period_inputs for a sample of rebalance dates to recover
per-period z-scores for all 4 signals (z_gp, z_ivol, z_resmom, z_pead).
Computes:

  1. Cross-sectional Pearson correlation between every pair of signals
     within each period; averages across periods.
  2. Information Coefficient (IC) per signal: Spearman rank correlation
     between signal at time t and realized return over (t, t+1m).
     Averaged across periods.

Output: a correlation matrix and IC table printed to stdout.
"""
# ruff: noqa: I001

from __future__ import annotations

import os
import sys
from datetime import date

import numpy as np
import pandas as pd

from axiom_fund import _warnings  # noqa: F401

from dotenv import load_dotenv

from axiom_fund.backtest.engine import _build_cache, _fetch_period_inputs
from axiom_fund.portfolio.composite import compute_composite_alpha
from axiom_fund.signals.alignment import align_signal
from axiom_fund.signals.gross_profitability import compute_gross_profitability
from axiom_fund.signals.idiosyncratic_volatility import (
    compute_idiosyncratic_volatility,
)
from axiom_fund.signals.pead import compute_pead_signal
from axiom_fund.signals.residual_momentum import compute_residual_momentum


# Sample dates: roughly one per year, mid-year (June end)
SAMPLE_DATES = [
    "2015-06-30", "2016-06-30", "2017-06-30", "2018-06-29",
    "2019-06-28", "2020-06-30", "2021-06-30", "2022-06-30",
    "2023-06-30", "2024-06-28",
]


def _rebuild_composite_for_date(
    cache, rebalance_date: pd.Timestamp,
) -> pd.DataFrame | None:
    """Rebuild composite z-score panel for one date, returning all 4 z-cols."""
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

    sig_lookback_start = (rebalance_date - pd.Timedelta(days=400)).strftime("%Y-%m-%d")
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
    pead_lookback_start = (rebalance_date - pd.Timedelta(days=900)).strftime("%Y-%m-%d")
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


def _forward_return(
    rets_full: pd.DataFrame, rebalance_date: pd.Timestamp, holding_days: int = 21,
) -> pd.Series:
    """Compound returns from rebalance to rebalance+holding_days per permno."""
    period_mask = (rets_full["date"] > rebalance_date) & (
        rets_full["date"] <= rebalance_date + pd.Timedelta(days=int(holding_days * 1.6))
    )
    sub = rets_full.loc[period_mask].copy()
    sub = sub.groupby("permno").head(holding_days)
    sub["log_ret"] = np.log1p(sub["ret"].astype("float64"))
    cum = sub.groupby("permno")["log_ret"].sum()
    return np.expm1(cum)


def main() -> int:
    load_dotenv()
    username = os.getenv("WRDS_USERNAME")
    if not username:
        print("ERROR: WRDS_USERNAME not set", file=sys.stderr)
        return 1

    import wrds
    db = wrds.Connection(wrds_username=username)
    try:
        rebal_dates = [pd.Timestamp(d) for d in SAMPLE_DATES]
        # Pull a window large enough for all sample dates
        full_dates = rebal_dates + [pd.Timestamp("2024-07-31")]  # extra for holding
        print(f"Building cache for {len(rebal_dates)} sample dates...")
        cache = _build_cache(db, rebalance_dates=full_dates, universe_size=1000)
        print(f"Cache: {len(cache.permnos_all)} permnos, "
              f"{len(cache.returns_full)} returns, "
              f"{len(cache.fundamentals_full)} fundamentals")
        print()

        all_panels: list[pd.DataFrame] = []
        all_pair_corrs: list[pd.DataFrame] = []
        all_ic_rows: list[dict[str, float]] = []

        for rb in rebal_dates:
            panel = _rebuild_composite_for_date(cache, rb)
            if panel is None or panel.empty:
                print(f"  {rb.date()}: skipped (no data)")
                continue

            zcols = ["z_gp", "z_ivol", "z_resmom", "z_pead"]
            present = [c for c in zcols if c in panel.columns]
            sub = panel[["permno"] + present].dropna(subset=present, how="all")
            corr = sub[present].corr(method="pearson")
            all_pair_corrs.append(corr)

            # IC computation
            fwd = _forward_return(cache.returns_full, rb, holding_days=21)
            merged = sub.set_index("permno").join(
                fwd.rename("fwd_ret"), how="inner",
            ).dropna(subset=["fwd_ret"])
            if len(merged) >= 50:
                ic_row: dict[str, float] = {"date": rb.strftime("%Y-%m-%d")}
                for col in present:
                    ic_row[col] = float(
                        merged[col].corr(merged["fwd_ret"], method="spearman")
                    )
                all_ic_rows.append(ic_row)
                print(f"  {rb.date()}: {len(merged)} names, IC computed")
            else:
                print(f"  {rb.date()}: {len(merged)} names (too few for IC)")

        print()
        print("=" * 70)
        print("AVERAGE PAIRWISE CORRELATION ACROSS PERIODS")
        print("=" * 70)
        avg_corr = pd.concat(all_pair_corrs).groupby(level=0).mean()
        # Reorder for clarity
        order = [c for c in ["z_gp", "z_ivol", "z_resmom", "z_pead"]
                 if c in avg_corr.columns]
        print(avg_corr.loc[order, order].round(3).to_string())
        print()
        print("=" * 70)
        print("INFORMATION COEFFICIENT (Spearman rank corr w/ 21-day forward return)")
        print("=" * 70)
        ic_df = pd.DataFrame(all_ic_rows)
        if not ic_df.empty:
            print("Period-by-period IC:")
            print(ic_df.set_index("date").round(3).to_string())
            print()
            print("Mean IC across periods:")
            mean_ic = ic_df[[c for c in ic_df.columns if c != "date"]].mean()
            std_ic = ic_df[[c for c in ic_df.columns if c != "date"]].std()
            n_periods = len(ic_df)
            t_stat = mean_ic / (std_ic / np.sqrt(n_periods))
            summary = pd.DataFrame({
                "mean_ic": mean_ic.round(4),
                "std_ic": std_ic.round(4),
                "t_stat": t_stat.round(2),
                "n_periods": n_periods,
            })
            print(summary.to_string())

    finally:
        db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())