"""Apply Durbin-Watson diagnostic to v1's idiosyncratic volatility regressions.

Per the v2 design doc (Item 2, scope amended in commit 2259dca), this
script computes the aggregate distribution of Durbin-Watson statistics
across the FF3 trailing-60-day regressions used in v1's IVol signal.

DW is a time-series autocorrelation test, well-defined for the FF3
regression because each fit is a 60-day time series of (excess return,
factors). DW near 2.0 indicates no autocorrelation; below 2.0 indicates
positive autocorrelation; above 2.0 indicates negative.

Scope: one regression per (permno, month-end) for permnos in the v1
universe, 2015-01 through 2024-11. Approximately 228K regressions.
This is a subsample of production's per-trading-day regressions, but
month-end is a uniform sample of windows; the DW distribution is
unbiased.

Mirrors the FF3 regression setup in
src/axiom_fund/signals/idiosyncratic_volatility.py _run_ff3_regression().

Output:
  data/cache/diagnostics_ivol_dw/dw_long.parquet
    one row per (permno, month_end) with the DW statistic and N obs
  data/cache/diagnostics_ivol_dw/run_metadata.txt

Aggregate distribution printed to console.
"""
# ruff: noqa: I001

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from dotenv import load_dotenv

from axiom_fund import _warnings  # noqa: F401

import wrds

from axiom_fund.data.ff_factors import FFFactors
from axiom_fund.data.returns import ReturnsPanel
from axiom_fund.diagnostics.residual_diagnostics import compute_durbin_watson


WEIGHTS_DIR = Path("data/cache/backtest_full_top1000")
OUTPUT_DIR = Path("data/cache/diagnostics_ivol_dw")

START_DATE = "2014-10-01"  # buffer for first 60-day window in Jan 2015
END_DATE = "2024-12-31"

# 60 trading-day window per the v1 spec
WINDOW = 60


def _load_universe_permnos() -> set[int]:
    """Union of permnos across all 116 v1 rebalance dates."""
    permnos: set[int] = set()
    for p in sorted(WEIGHTS_DIR.glob("weights_*.parquet")):
        df = pd.read_parquet(p)
        permnos.update(df.index.tolist())
    return permnos


def _build_excess_returns_panel(
    returns_df: pd.DataFrame, ff_df: pd.DataFrame,
) -> pd.DataFrame:
    """Join daily returns with FF factors, compute excess returns.

    Output: long-format (permno, date, excess_ret, mktrf, smb, hml).
    """
    returns_df = returns_df[["permno", "date", "ret"]].copy()
    returns_df["date"] = pd.to_datetime(returns_df["date"])

    ff_df = ff_df[["date", "mktrf", "smb", "hml", "rf"]].copy()
    ff_df["date"] = pd.to_datetime(ff_df["date"])

    panel = returns_df.merge(ff_df, on="date", how="inner")
    panel["excess_ret"] = panel["ret"] - panel["rf"]
    return panel[
        ["permno", "date", "excess_ret", "mktrf", "smb", "hml"]
    ].sort_values(["permno", "date"]).reset_index(drop=True)


def _run_ff3_dw(window_df: pd.DataFrame) -> float | None:
    """Fit FF3 on a window and compute DW from residuals.

    Mirrors idiosyncratic_volatility._run_ff3_regression() except
    we keep the residuals and compute DW instead of residual_std.

    Returns None if regression cannot run.
    """
    clean = window_df.dropna(subset=["excess_ret", "mktrf", "smb", "hml"])
    if len(clean) < 4:
        return None

    y = clean["excess_ret"].to_numpy()
    X = clean[["mktrf", "smb", "hml"]].to_numpy()
    X = sm.add_constant(X)

    try:
        results = sm.OLS(y, X).fit()
    except (np.linalg.LinAlgError, ValueError):
        return None

    residuals = np.asarray(results.resid, dtype=np.float64)
    if len(residuals) < 2 or np.all(residuals == 0):
        return None
    try:
        dw = compute_durbin_watson(residuals)
    except ValueError:
        return None
    return dw


def _compute_dw_per_permno(
    panel: pd.DataFrame, month_ends: list[pd.Timestamp],
) -> list[dict]:
    """For one permno's daily panel, fit FF3 + DW at each month-end window."""
    out = []
    permno = panel["permno"].iloc[0]
    panel = panel.set_index("date").sort_index()
    for me in month_ends:
        # Take trailing 60 trading days ending at month-end (inclusive)
        window = panel[panel.index <= me].tail(WINDOW)
        if len(window) < WINDOW:
            continue
        dw = _run_ff3_dw(window.reset_index())
        if dw is None:
            continue
        out.append({
            "permno": permno,
            "month_end": me,
            "n_obs": len(window),
            "dw": dw,
        })
    return out


def _print_aggregate_summary(df: pd.DataFrame) -> None:
    print()
    print("=" * 80)
    print("v2 Item 2 — IVol FF3 regressions — Durbin-Watson distribution")
    print("=" * 80)
    print(f"\nRegressions analyzed: {len(df):,}")
    print(f"Unique permnos:       {df['permno'].nunique():,}")
    print(f"Unique month-ends:    {df['month_end'].nunique()}")
    print()

    print("DW distribution (asymptotic H_0: DW = 2.0):")
    desc = df["dw"].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    for k in ("mean", "min", "1%", "5%", "25%", "50%", "75%", "95%", "99%", "max"):
        print(f"  {k:5s}  {desc[k]:.4f}")
    print(f"  std    {desc['std']:.4f}")

    # Conventional bounds for FF3 at N=60: roughly dL=1.44, dU=1.73 at 5% one-sided
    # (positive autocorrelation). Two-sided test: indeterminate zone [1.44, 1.73]
    # rejects H_0 at p<0.05 if DW < 1.44 or DW > (4-1.44)=2.56.
    n_strong_positive = (df["dw"] < 1.44).sum()
    n_strong_negative = (df["dw"] > 2.56).sum()
    n_indeterminate_low = ((df["dw"] >= 1.44) & (df["dw"] < 1.73)).sum()
    n_indeterminate_high = ((df["dw"] > 4 - 1.73) & (df["dw"] <= 2.56)).sum()
    n_no_evidence = (
        (df["dw"] >= 1.73) & (df["dw"] <= 4 - 1.73)
    ).sum()

    total = len(df)
    print(f"\nClassification (DW critical values for N=60, k=3 regressors, 5% one-sided):")
    print(f"  DW < 1.44 (strong positive autocorrelation):"
          f" {n_strong_positive:>7,} ({100*n_strong_positive/total:5.1f}%)")
    print(f"  1.44 <= DW < 1.73 (indeterminate, positive side):"
          f" {n_indeterminate_low:>7,} ({100*n_indeterminate_low/total:5.1f}%)")
    print(f"  1.73 <= DW <= 2.27 (no evidence of autocorrelation):"
          f" {n_no_evidence:>7,} ({100*n_no_evidence/total:5.1f}%)")
    print(f"  2.27 < DW <= 2.56 (indeterminate, negative side):"
          f" {n_indeterminate_high:>7,} ({100*n_indeterminate_high/total:5.1f}%)")
    print(f"  DW > 2.56 (strong negative autocorrelation):"
          f" {n_strong_negative:>7,} ({100*n_strong_negative/total:5.1f}%)")


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading v1 universe permnos from weights cache...")
    permnos = sorted(_load_universe_permnos())
    print(f"  {len(permnos):,} unique permnos in v1 backtest")

    load_dotenv()
    username = os.getenv("WRDS_USERNAME")
    if not username:
        print("ERROR: WRDS_USERNAME not in .env", file=sys.stderr)
        return 1

    print(f"\nFetching returns + FF factors from {START_DATE} to {END_DATE}...")
    db = wrds.Connection(wrds_username=username)
    try:
        returns_panel = ReturnsPanel(db)
        returns_df = returns_panel.fetch(permnos, START_DATE, END_DATE)
        print(f"  Returns: {len(returns_df):,} rows")

        ff = FFFactors(db)
        ff_df = ff.fetch(START_DATE, END_DATE)
        print(f"  FF factors: {len(ff_df):,} rows")
    finally:
        db.close()

    print("\nBuilding excess-returns panel...")
    panel = _build_excess_returns_panel(returns_df, ff_df)
    print(f"  {len(panel):,} (permno, date) rows")

    # Build month-end calendar from the FF factor dates (real trading days)
    ff_df_indexed = ff_df.set_index("date").sort_index()
    month_ends_all = ff_df_indexed.groupby(ff_df_indexed.index.to_period("M")).tail(1).index
    # Restrict to 2015-01 onward (need 60-day buffer before)
    month_ends = [
        pd.Timestamp(d) for d in month_ends_all
        if pd.Timestamp(d) >= pd.Timestamp("2015-01-01")
    ]
    print(f"  {len(month_ends)} month-end terminal dates")

    print(f"\nRunning DW diagnostic across {len(permnos):,} permnos × "
          f"{len(month_ends)} month-ends...")
    print("  (this will take a few minutes)")

    all_results = []
    grouped = panel.groupby("permno")
    for i, (permno, group) in enumerate(grouped, 1):
        results = _compute_dw_per_permno(group, month_ends)
        all_results.extend(results)
        if i % 200 == 0 or i == len(grouped):
            print(f"  {i}/{len(grouped):,} permnos processed; "
                  f"{len(all_results):,} regressions so far")

    df_out = pd.DataFrame(all_results)
    df_out.to_parquet(OUTPUT_DIR / "dw_long.parquet", index=False)

    (OUTPUT_DIR / "run_metadata.txt").write_text(
        f"Run: {datetime.now().isoformat()}\n"
        f"Window: {START_DATE} → {END_DATE}\n"
        f"Permnos: {len(permnos):,}\n"
        f"Month-ends: {len(month_ends)}\n"
        f"Regressions: {len(df_out):,}\n"
        f"Trailing window per regression: {WINDOW} trading days\n"
    )

    _print_aggregate_summary(df_out)
    print(f"\nSaved: {OUTPUT_DIR}/dw_long.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())