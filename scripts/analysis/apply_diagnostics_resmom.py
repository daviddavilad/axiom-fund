"""Apply v2 residual diagnostics to v1's residual momentum regressions.

Per the v2 design doc (Item 2), this script applies 5 of 6 diagnostics
to the cross-sectional regression embedded in residual momentum:

  monthly_ret_i = α + γ_industry · industry_dummies_i + β · log_size_i + ε_i

Run once per month across the 116-period backtest window (2015-01 →
2024-11). Durbin-Watson is *not* applied — the regression is cross-
sectional, so DW (a time-series autocorrelation test) is undefined.

The diagnostic data pipeline mirrors src/axiom_fund/signals/
residual_momentum.py (lines 145-220) for consistency. The regression
setup mirrors the body of _cross_sectional_residual() (lines 258-289).

Output:
  data/cache/diagnostics_resmom/per_month_diagnostics.parquet
    one row per month with summary stats: BP p-value, max Cook's,
    max leverage, n_obs, n_params, R²
  data/cache/diagnostics_resmom/per_observation_diagnostics.parquet
    one row per (month, permno) with leverage, Cook's, standardized
    residual. Useful for ranking which names drove the worst months.
  data/cache/diagnostics_resmom/run_metadata.txt

Aggregate summary printed to console.

This script is one-off; not part of the production backtest.
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

from axiom_fund.data.fundamentals import Fundamentals
from axiom_fund.data.returns import ReturnsPanel
from axiom_fund.diagnostics.residual_diagnostics import (
    compute_breusch_pagan,
    compute_cooks_distance,
    compute_leverage,
    compute_qq_data,
    compute_residual_vs_fitted_data,
)


WEIGHTS_DIR = Path("data/cache/backtest_full_top1000")
OUTPUT_DIR = Path("data/cache/diagnostics_resmom")

START_DATE = "2014-12-01"  # buffer for first signal date
END_DATE = "2024-12-31"


def _load_universe_permnos() -> set[int]:
    """Union of permnos across all 116 v1 rebalance dates."""
    permnos: set[int] = set()
    for p in sorted(WEIGHTS_DIR.glob("weights_*.parquet")):
        df = pd.read_parquet(p)
        permnos.update(df.index.tolist())
    return permnos


def _build_monthly_panel(
    returns_df: pd.DataFrame, fundamentals_df: pd.DataFrame,
) -> pd.DataFrame:
    """Mirror residual_momentum.py's monthly panel construction.

    Output: long-format DataFrame with columns
    (permno, month_end, monthly_ret, log_size, ggroup).
    """
    df = returns_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["month_end"] = df["date"] + pd.offsets.MonthEnd(0)

    # Monthly returns from compounded daily returns
    monthly = (
        df.groupby(["permno", "month_end"])
        .agg(
            monthly_ret=("ret", lambda s: (1 + s).prod() - 1),
            month_end_marketcap=("marketcap", "last"),
        )
        .reset_index()
    )

    # As-of merge with fundamentals to attach ggroup
    # merge_asof requires the on= columns to be globally sorted (not by group).
    # Mirrors the pattern in src/axiom_fund/signals/residual_momentum.py L164-170.
    fund = fundamentals_df[["permno", "rdq", "ggroup"]].dropna(subset=["ggroup"]).copy()
    fund["rdq"] = pd.to_datetime(fund["rdq"])
    fund = fund.sort_values("rdq").reset_index(drop=True)
    monthly = monthly.sort_values("month_end").reset_index(drop=True)

    monthly = pd.merge_asof(
        monthly,
        fund,
        left_on="month_end",
        right_on="rdq",
        by="permno",
        direction="backward",
    )
    monthly = monthly.dropna(subset=["ggroup"])
    monthly["log_size"] = np.log(
        monthly["month_end_marketcap"].clip(lower=1.0)
    ).astype("float64")
    monthly = monthly.dropna(subset=["monthly_ret", "log_size"])
    return monthly[["permno", "month_end", "monthly_ret", "log_size", "ggroup"]]


def _cross_sectional_regression_full(
    group: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int] | None:
    """Re-implementation of _cross_sectional_residual returning full state.

    Returns (X, residuals, fitted, n_params) or None if regression cannot run.
    Mirrors the production logic exactly.
    """
    if len(group) < 5:
        return None

    n_groups = group["ggroup"].nunique()
    if n_groups < 2:
        X = group[["log_size"]].to_numpy()
    else:
        dummies = pd.get_dummies(group["ggroup"], drop_first=True, dtype=float)
        X = np.column_stack([
            dummies.to_numpy(),
            group["log_size"].to_numpy().reshape(-1, 1),
        ])

    X = sm.add_constant(X)
    y = group["monthly_ret"].to_numpy()

    try:
        results = sm.OLS(y, X).fit()
    except (np.linalg.LinAlgError, ValueError):
        return None

    return (
        X.astype(np.float64),
        np.asarray(results.resid, dtype=np.float64),
        np.asarray(results.fittedvalues, dtype=np.float64),
        int(X.shape[1]),
    )


def _diagnose_one_month(
    month: pd.Timestamp, group: pd.DataFrame,
) -> tuple[dict | None, pd.DataFrame | None]:
    """Run 5 diagnostics on one month's regression.

    Returns (summary_row, per_observation_df) or (None, None) if
    diagnostics cannot run (regression failed, etc.).
    """
    reg = _cross_sectional_regression_full(group)
    if reg is None:
        return None, None
    X, residuals, fitted, n_params = reg

    # Compute diagnostics
    try:
        bp = compute_breusch_pagan(residuals, X)
    except ValueError:
        bp = {"lm_pvalue": np.nan, "f_pvalue": np.nan,
              "lm_statistic": np.nan, "f_statistic": np.nan}

    try:
        leverage = compute_leverage(X)
    except ValueError:
        return None, None  # Singular X; whole month invalid

    try:
        cooks = compute_cooks_distance(residuals, X, n_params=n_params)
    except ValueError:
        cooks = np.full_like(residuals, np.nan)

    # Q-Q and residual-vs-fitted are diagnostic data, not statistics; aggregate
    # only by summary statistics here (full data saved to disk per observation).

    # Standardized residuals for outlier ranking
    mse = float(np.mean(residuals ** 2)) if len(residuals) > 0 else 0.0
    sigma = np.sqrt(mse) if mse > 0 else 1.0
    standardized_resid = residuals / sigma

    # Summary row
    n_obs = len(residuals)
    # Detect numerically degenerate observations (leverage ~ 1.0).
    # These correspond to solo-category observations (e.g., single
    # stock in its industry group); the regression fits them exactly
    # by construction, residual = 0 by floating-point precision, and
    # Cook's distance becomes numerically unstable due to division by
    # (1 - h_ii)^2.
    is_degenerate = leverage > 0.99
    n_degenerate = int(is_degenerate.sum())
    r_squared = 1.0 - np.var(residuals) / np.var(group["monthly_ret"].to_numpy())
    summary = {
        "month_end": month,
        "n_obs": n_obs,
        "n_params": n_params,
        "n_degenerate_leverage": n_degenerate,
        "r_squared": float(r_squared),
        "bp_lm_pvalue": bp["lm_pvalue"],
        "bp_f_pvalue": bp["f_pvalue"],
        "max_leverage": float(leverage.max()),
        "mean_leverage": float(leverage.mean()),
        "n_high_leverage": int((leverage > 2 * n_params / n_obs).sum()),
        "max_cooks_full": float(np.nanmax(cooks)),
        "max_cooks_clean": float(np.nanmax(np.where(is_degenerate, np.nan, cooks))),
        "n_high_cooks": int(((cooks > 4 / n_obs) & ~is_degenerate).sum()),
        "residual_skew": float(pd.Series(residuals).skew()),
        "residual_kurtosis": float(pd.Series(residuals).kurtosis()),
    }

    # Per-observation rows
    per_obs = pd.DataFrame({
        "month_end": month,
        "permno": group["permno"].to_numpy(),
        "residual": residuals,
        "standardized_residual": standardized_resid,
        "fitted": fitted,
        "leverage": leverage,
        "cooks_distance": cooks,
    })

    return summary, per_obs


def _print_aggregate_summary(monthly_df: pd.DataFrame) -> None:
    print()
    print("=" * 80)
    print("v2 Item 2 — ResMom diagnostics — aggregate across months")
    print("=" * 80)
    print(f"\nMonths analyzed: {len(monthly_df)}")
    print(f"Mean obs per month: {monthly_df['n_obs'].mean():.1f}")
    print(f"Mean n_params:     {monthly_df['n_params'].mean():.1f}")
    print(f"Mean R²:           {monthly_df['r_squared'].mean():.4f}")

    print("\nBreusch-Pagan LM p-value distribution across months:")
    desc = monthly_df["bp_lm_pvalue"].describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99])
    for k in ("mean", "50%", "75%", "90%", "95%", "99%", "max"):
        print(f"  {k:5s} {desc[k]:.4f}")
    n_reject_05 = (monthly_df["bp_lm_pvalue"] < 0.05).sum()
    n_reject_01 = (monthly_df["bp_lm_pvalue"] < 0.01).sum()
    print(f"  Months rejecting at p<0.05: {n_reject_05}/{len(monthly_df)} "
          f"({100*n_reject_05/len(monthly_df):.1f}%)")
    print(f"  Months rejecting at p<0.01: {n_reject_01}/{len(monthly_df)} "
          f"({100*n_reject_01/len(monthly_df):.1f}%)")

    n_degen_total = monthly_df["n_degenerate_leverage"].sum()
    n_months_with_degen = (monthly_df["n_degenerate_leverage"] > 0).sum()
    print(f"\nDegenerate observations (leverage > 0.99, solo-category):")
    print(f"  Total degenerate observations: {n_degen_total}")
    print(f"  Months with at least one degenerate: {n_months_with_degen}/{len(monthly_df)}")
    print(f"  These are removed from Cook's distance reporting below.")

    print("\nMax Cook's distance per month — non-degenerate observations only:")
    desc = monthly_df["max_cooks_clean"].describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99])
    for k in ("mean", "50%", "75%", "90%", "95%", "99%", "max"):
        print(f"  {k:5s} {desc[k]:.4f}")

    print("\nMax leverage per month:")
    desc = monthly_df["max_leverage"].describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99])
    for k in ("mean", "50%", "75%", "90%", "95%", "99%", "max"):
        print(f"  {k:5s} {desc[k]:.4f}")

    print("\nResidual skewness distribution:")
    desc = monthly_df["residual_skew"].describe()
    print(f"  mean {desc['mean']:.3f}, std {desc['std']:.3f}, "
          f"min {desc['min']:.3f}, max {desc['max']:.3f}")

    print("\nResidual excess kurtosis distribution:")
    desc = monthly_df["residual_kurtosis"].describe()
    print(f"  mean {desc['mean']:.3f}, std {desc['std']:.3f}, "
          f"min {desc['min']:.3f}, max {desc['max']:.3f}")


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

    print(f"\nFetching returns + fundamentals from {START_DATE} to {END_DATE}...")
    db = wrds.Connection(wrds_username=username)
    try:
        returns_panel = ReturnsPanel(db)
        returns_df = returns_panel.fetch(permnos, START_DATE, END_DATE)
        print(f"  Returns: {len(returns_df):,} rows")

        fundamentals = Fundamentals(db)
        fundamentals_df = fundamentals.fetch_quarterly(permnos, START_DATE, END_DATE)
        print(f"  Fundamentals: {len(fundamentals_df):,} rows")
    finally:
        db.close()

    print("\nBuilding monthly panel...")
    monthly = _build_monthly_panel(returns_df, fundamentals_df)
    n_months = monthly["month_end"].nunique()
    print(f"  {len(monthly):,} (permno, month) rows, {n_months} distinct months")

    print(f"\nRunning diagnostics on {n_months} months...")
    summary_rows = []
    per_obs_frames = []
    for i, (month, group) in enumerate(monthly.groupby("month_end"), 1):
        summary, per_obs = _diagnose_one_month(month, group)
        if summary is not None:
            summary_rows.append(summary)
            per_obs_frames.append(per_obs)
        if i % 20 == 0 or i == n_months:
            print(f"  {i}/{n_months}: {month.date()}")

    if not summary_rows:
        print("ERROR: no months produced valid diagnostics", file=sys.stderr)
        return 1

    monthly_df = pd.DataFrame(summary_rows)
    per_obs_df = pd.concat(per_obs_frames, ignore_index=True)

    monthly_df.to_parquet(OUTPUT_DIR / "per_month_diagnostics.parquet", index=False)
    per_obs_df.to_parquet(OUTPUT_DIR / "per_observation_diagnostics.parquet", index=False)

    (OUTPUT_DIR / "run_metadata.txt").write_text(
        f"Run: {datetime.now().isoformat()}\n"
        f"Window: {START_DATE} → {END_DATE}\n"
        f"Months analyzed: {len(monthly_df)}\n"
        f"Total observations: {len(per_obs_df):,}\n"
    )

    _print_aggregate_summary(monthly_df)
    print(f"\nSaved: {OUTPUT_DIR}/per_month_diagnostics.parquet")
    print(f"Saved: {OUTPUT_DIR}/per_observation_diagnostics.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())