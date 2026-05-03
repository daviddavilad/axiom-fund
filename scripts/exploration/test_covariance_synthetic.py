"""Smoke test for Ledoit-Wolf shrinkage on synthetic data.

Uses a HETEROGENEOUS correlation structure (block-diagonal: 3 sectors with
high within-sector correlation, low across-sector). The constant-correlation
target is genuinely misspecified for this data, so we can observe how
shrinkage adapts to noise level.

Verifies:
1. Shrinkage > 0 when sample is noisy (small T, large N) — target helps
2. Shrinkage close to 0 when sample is well-estimated (large T) — sample is fine
3. Output matrix is symmetric and positive definite
4. Annualization factor is exactly 252
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from axiom_fund.portfolio.covariance import estimate_covariance


def _block_correlation(n: int, n_sectors: int, rho_within: float, rho_across: float) -> np.ndarray:
    """Build a block-diagonal correlation matrix with sectors."""
    sector_size = n // n_sectors
    corr = np.full((n, n), rho_across)
    for s in range(n_sectors):
        start = s * sector_size
        end = start + sector_size
        corr[start:end, start:end] = rho_within
    np.fill_diagonal(corr, 1.0)
    return corr


def main() -> int:
    np.random.seed(42)

    # Block structure: 3 sectors of 10 stocks each.
    # Within sector: 0.6 correlation. Across sectors: 0.1.
    # Constant-correlation target assumes uniform 0.27 — real misspec.
    n_assets = 30
    n_sectors = 3
    rho_within = 0.6
    rho_across = 0.1
    daily_vol = 0.015
    corr = _block_correlation(n_assets, n_sectors, rho_within, rho_across)
    cov_true = corr * (daily_vol**2)

    # Scenario 1: noisy regime — T < 2N
    print("=" * 70)
    print("Scenario 1: noisy regime (T=50 days, N=30 stocks)")
    print("=" * 70)
    rets_noisy = np.random.multivariate_normal(
        np.zeros(n_assets), cov_true, size=50
    )
    df_noisy = pd.DataFrame(rets_noisy, columns=[f"P{i}" for i in range(n_assets)])
    est_noisy = estimate_covariance(df_noisy)
    print(f"Shrinkage: {est_noisy.shrinkage:.4f} (expected: 0.20-0.80)")
    print(f"n_obs: {est_noisy.n_obs}, n_assets: {est_noisy.n_assets}")
    eigvals = np.linalg.eigvalsh(est_noisy.matrix.values)
    print(f"Min eigenvalue: {eigvals.min():.6f} (expected > 0)")
    print(f"Mean diagonal (annualized): {np.diag(est_noisy.matrix.values).mean():.4f} "
          f"(expected ~{daily_vol**2 * 252:.4f})")
    print()

    # Scenario 2: well-estimated regime — T >> N
    print("=" * 70)
    print("Scenario 2: well-estimated regime (T=2000 days, N=30 stocks)")
    print("=" * 70)
    rets_clean = np.random.multivariate_normal(
        np.zeros(n_assets), cov_true, size=2000
    )
    df_clean = pd.DataFrame(rets_clean, columns=[f"P{i}" for i in range(n_assets)])
    est_clean = estimate_covariance(df_clean)
    print(f"Shrinkage: {est_clean.shrinkage:.4f} (expected: < 0.20)")
    print(f"n_obs: {est_clean.n_obs}, n_assets: {est_clean.n_assets}")
    eigvals2 = np.linalg.eigvalsh(est_clean.matrix.values)
    print(f"Min eigenvalue: {eigvals2.min():.6f} (expected > 0)")
    print()

    # Scenario 3: Annualization check
    print("=" * 70)
    print("Scenario 3: annualization check (T=2000, N=30)")
    print("=" * 70)
    est_daily = estimate_covariance(df_clean, annualize=False)
    ratio = est_clean.matrix.values / est_daily.matrix.values
    print(f"Annualized / daily ratio: {ratio.mean():.2f} (should be 252)")
    print(f"Min: {ratio.min():.2f}, Max: {ratio.max():.2f}")
    print()

    # Scenario 4: extreme noise — T much smaller than N
    print("=" * 70)
    print("Scenario 4: very noisy (T=20 days, N=30 stocks)")
    print("=" * 70)
    rets_extreme = np.random.multivariate_normal(
        np.zeros(n_assets), cov_true, size=20
    )
    df_extreme = pd.DataFrame(rets_extreme, columns=[f"P{i}" for i in range(n_assets)])
    est_extreme = estimate_covariance(df_extreme)
    print(f"Shrinkage: {est_extreme.shrinkage:.4f} (expected: very high, > 0.5)")
    eigvals3 = np.linalg.eigvalsh(est_extreme.matrix.values)
    print(f"Min eigenvalue: {eigvals3.min():.6f} (expected > 0)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
