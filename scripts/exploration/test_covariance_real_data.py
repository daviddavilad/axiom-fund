"""End-to-end test of covariance estimation against real WRDS data.

Pulls a 252-day window of daily returns for the top-100 universe,
runs the Ledoit-Wolf estimator, prints diagnostics. Verifies the
matrix has the right shape, values, and properties for portfolio
optimization.
"""
# ruff: noqa: I001

from __future__ import annotations

import os
import sys

from axiom_fund import _warnings  # noqa: F401

import numpy as np
from dotenv import load_dotenv

from axiom_fund.data.returns import ReturnsPanel
from axiom_fund.data.universe import Universe
from axiom_fund.portfolio.covariance import estimate_covariance


def main() -> int:
    load_dotenv()
    username = os.getenv("WRDS_USERNAME")
    if not username:
        print("ERROR: WRDS_USERNAME not set", file=sys.stderr)
        return 1

    import wrds
    db = wrds.Connection(wrds_username=username)

    try:
        # Top-100 universe as of 2020-12-31
        print("Pulling top-100 universe as of 2020-12-31...")
        u = Universe(db).as_of("2020-12-31")
        permnos = u.head(100)["permno"].tolist()
        print(f"Got {len(permnos)} PERMNOs")
        print()

        # 252 trading days of daily returns ending 2020-12-31
        print("Pulling 252 trading days of returns ending 2020-12-31...")
        rp = ReturnsPanel(db)
        panel = rp.fetch(
            permnos=permnos,
            start_date="2019-12-01",
            end_date="2020-12-31",
        )
        print(f"Long-format panel: {len(panel):,} rows")

        # Pivot to wide format: dates × permnos
        wide = panel.pivot_table(
            index="date", columns="permno", values="ret", aggfunc="last"
        )
        print(f"Wide format: {wide.shape[0]} dates × {wide.shape[1]} stocks")
        print(f"NaN density: {wide.isna().sum().sum() / wide.size:.2%}")
        print()

        # Estimate covariance
        print("Computing Ledoit-Wolf covariance...")
        result = estimate_covariance(wide)
        print()

        print("=" * 70)
        print("Covariance estimate diagnostics")
        print("=" * 70)
        print(f"Shrinkage:     {result.shrinkage:.4f}")
        print(f"n_obs:         {result.n_obs}")
        print(f"n_assets:      {result.n_assets}")
        print()

        # Diagonal entries: annualized variances → annualized vols
        diag = np.diag(result.matrix.values)
        ann_vol = np.sqrt(diag)
        print("Annualized volatility distribution (diagonal):")
        print(f"  Min:    {ann_vol.min():.3f} ({ann_vol.min()*100:.1f}%)")
        print(f"  Median: {np.median(ann_vol):.3f} ({np.median(ann_vol)*100:.1f}%)")
        print(f"  Mean:   {ann_vol.mean():.3f} ({ann_vol.mean()*100:.1f}%)")
        print(f"  Max:    {ann_vol.max():.3f} ({ann_vol.max()*100:.1f}%)")
        print()

        # Off-diagonal: correlations
        std_outer = np.outer(ann_vol, ann_vol)
        corr = result.matrix.values / std_outer
        n = corr.shape[0]
        mask_off = ~np.eye(n, dtype=bool)
        off_diag_corr = corr[mask_off]
        print("Pairwise correlation distribution (off-diagonal):")
        print(f"  Min:    {off_diag_corr.min():.3f}")
        print(f"  Median: {np.median(off_diag_corr):.3f}")
        print(f"  Mean:   {off_diag_corr.mean():.3f}")
        print(f"  Max:    {off_diag_corr.max():.3f}")
        print()

        # Eigenvalue check
        eigvals = np.linalg.eigvalsh(result.matrix.values)
        print("Eigenvalue distribution:")
        print(f"  Min: {eigvals.min():.6f} (must be > 0 for portfolio optimization)")
        print(f"  Max: {eigvals.max():.4f}")
        print(f"  Condition number: {eigvals.max() / eigvals.min():.1f}")

    finally:
        db.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())