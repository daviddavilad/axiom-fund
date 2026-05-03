"""Covariance estimation via Ledoit-Wolf shrinkage with constant correlation
target (Ledoit & Wolf 2004).

Why shrinkage matters
---------------------
The naive sample covariance matrix is a poor input to portfolio optimization
for two reasons:

1. **Singularity.** With T < N (e.g., 252 trading days, 1000 stocks), the
   sample covariance is rank-deficient. Optimizers see zero variance in
   some directions and load up infinitely on those.

2. **Estimation error.** Even with adequate sample size, the smallest
   eigenvalues of the sample covariance are noisy. Mean-variance
   optimizers exploit precisely those low-variance directions, which are
   mostly noise. Realized portfolio variance ends up much higher than
   the optimizer expected.

Ledoit-Wolf shrinkage solves both: it shrinks the sample covariance
toward a structured target by an analytically-optimal amount α:

    Σ_shrunk = α × T + (1 - α) × S

where T is the target matrix and S is the sample covariance. The optimal
α minimizes the expected Frobenius distance between Σ_shrunk and the
true covariance — closed-form, no cross-validation needed.

Constant correlation target
---------------------------
This module uses the constant-correlation target from Ledoit & Wolf
(2004). The target T has:
- Diagonal entries equal to the sample variances (preserves volatility
  structure)
- Off-diagonal entries equal to the average sample correlation, scaled
  by the geometric mean of variances:
      T_ij = r_bar × sqrt(s_i × s_j)
  where r_bar is the average pairwise correlation and s_i is variance i.

This target preserves real volatility differences between assets while
imposing a common correlation. It works well for equity universes where
real correlation structure is roughly homogeneous.

Annualization
-------------
The estimator operates on daily returns. The output covariance matrix
is the *annualized* covariance: Σ_daily × 252. This means downstream
optimizers interpret weights, expected returns, and risk targets in
annualized units throughout.

Reference
---------
Ledoit, O. & Wolf, M. (2004). "Honey, I shrunk the sample covariance
matrix." Journal of Portfolio Management, 30(4), 110-119.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

# Number of trading days per year, used to annualize the covariance matrix.
_TRADING_DAYS_PER_YEAR: int = 252


@dataclass(frozen=True)
class CovarianceEstimate:
    """Output of the covariance estimator.

    Attributes
    ----------
    matrix : pd.DataFrame
        The estimated annualized covariance matrix, indexed by permno on
        both axes. Square, symmetric, positive-definite.
    shrinkage : float
        The shrinkage intensity α used. 0 = pure sample covariance;
        1 = pure target. Typical values: 0.05–0.30 for equity universes.
    n_obs : int
        Number of trailing daily observations used.
    n_assets : int
        Number of assets in the matrix (rank N).
    """

    matrix: pd.DataFrame
    shrinkage: float
    n_obs: int
    n_assets: int


def estimate_covariance(
    returns_wide: pd.DataFrame,
    annualize: bool = True,
) -> CovarianceEstimate:
    """Estimate the covariance matrix via Ledoit-Wolf with constant correlation.

    Parameters
    ----------
    returns_wide : pd.DataFrame
        Wide-format daily returns: index is dates, columns are PERMNOs,
        values are decimal returns (0.01 = 1%). NaN values are dropped
        per-row before estimation (so rows with any NaN are excluded
        entirely from the sample).
    annualize : bool, default True
        If True, multiply the covariance by 252 (trading days/year)
        to express the matrix in annualized units.

    Returns
    -------
    CovarianceEstimate
        Dataclass with fields: matrix, shrinkage, n_obs, n_assets.

    Raises
    ------
    ValueError
        If returns_wide is empty, has fewer than 2 assets, or has
        fewer than 2 valid observations after NaN dropping.
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if not isinstance(returns_wide, pd.DataFrame):
        raise ValueError(
            f"returns_wide must be a pandas DataFrame, got {type(returns_wide).__name__}"
        )
    if returns_wide.shape[1] < 2:
        raise ValueError(
            f"need at least 2 assets, got {returns_wide.shape[1]}"
        )

    # Drop rows with any NaN. A more sophisticated implementation could
    # use pairwise-complete observations, but row-wise dropping is simpler
    # and standard for portfolio optimization.
    clean = returns_wide.dropna(how="any")
    n_obs, n_assets = clean.shape

    if n_obs < 2:
        raise ValueError(
            f"need at least 2 valid observations after NaN dropping, got {n_obs}"
        )

    # ------------------------------------------------------------------
    # Sample covariance and correlation
    # ------------------------------------------------------------------
    returns_array = clean.to_numpy()
    sample_cov = _sample_covariance(returns_array)

    # ------------------------------------------------------------------
    # Constant correlation target
    # ------------------------------------------------------------------
    target = _constant_correlation_target(sample_cov)

    # ------------------------------------------------------------------
    # Shrinkage intensity (closed-form, Ledoit & Wolf 2004 eq. 2.6 - 2.10)
    # ------------------------------------------------------------------
    shrinkage = _shrinkage_intensity(returns_array, sample_cov, target)

    # ------------------------------------------------------------------
    # Final shrunk covariance
    # ------------------------------------------------------------------
    shrunk = shrinkage * target + (1.0 - shrinkage) * sample_cov

    if annualize:
        shrunk = shrunk * _TRADING_DAYS_PER_YEAR

    permnos = list(clean.columns)
    matrix_df = pd.DataFrame(shrunk, index=permnos, columns=permnos)

    return CovarianceEstimate(
        matrix=matrix_df,
        shrinkage=shrinkage,
        n_obs=n_obs,
        n_assets=n_assets,
    )


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _sample_covariance(
    returns: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Compute the maximum-likelihood sample covariance matrix.

    Note: uses ddof=0 (divide by T, not T-1) — this matches the convention
    in Ledoit & Wolf (2004). The shrinkage formula assumes ML covariance.
    """
    centered = returns - returns.mean(axis=0)
    n_obs = returns.shape[0]
    cov: npt.NDArray[np.float64] = (centered.T @ centered) / n_obs
    return cov


def _constant_correlation_target(
    sample_cov: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Build the constant-correlation target matrix.

    Diagonal: sample variances (preserved).
    Off-diagonal: r_bar × sqrt(s_i × s_j)
    where r_bar is the average pairwise correlation.
    """
    variances = np.diag(sample_cov)
    std = np.sqrt(variances)

    # Build sample correlation matrix
    outer_std = np.outer(std, std)
    # Avoid divide-by-zero for assets with zero variance (degenerate)
    with np.errstate(invalid="ignore", divide="ignore"):
        sample_corr = sample_cov / outer_std

    # Average correlation: mean of off-diagonal entries
    n = sample_cov.shape[0]
    mask_off_diag = ~np.eye(n, dtype=bool)
    r_bar = float(np.nanmean(sample_corr[mask_off_diag]))

    # Build target: diagonal = variances, off-diagonal = r_bar * sqrt(s_i * s_j)
    target: npt.NDArray[np.float64] = r_bar * outer_std
    np.fill_diagonal(target, variances)

    return target


def _shrinkage_intensity(
    returns: npt.NDArray[np.float64],
    sample_cov: npt.NDArray[np.float64],
    target: npt.NDArray[np.float64],
) -> float:
    """Compute the optimal shrinkage intensity α via Ledoit-Wolf 2004.

    α = π / (π + γ) where:
      π = sum of asymptotic variances of sample covariance entries
      γ = squared Frobenius norm of (sample - target)

    The result is clipped to [0, 1].

    See Ledoit & Wolf (2004) eq. 2.6, 2.10 for derivations.
    """
    n_obs = returns.shape[0]

    # ------------------------------------------------------------------
    # π: estimate of asymptotic variance of sample covariance
    # π_ij = Var(sqrt(T) × s_ij), summed over i,j
    # In sample form: π̂_ij = (1/T) sum_t (x_it × x_jt - s_ij)^2
    # ------------------------------------------------------------------
    centered = returns - returns.mean(axis=0)
    # squared products: (T, N, N) is too memory-heavy for N=1000; vectorize differently
    # E[x_t x_t'] - s = x_t x_t' - s, want sum over t of element-wise squared
    # equivalent: (1/T) sum_t (x_t x_t')^2  -  s^2 (element-wise)
    # which equals: (1/T) (centered**2).T @ (centered**2)  -  s_ij^2
    pi_matrix = (centered**2).T @ (centered**2) / n_obs - sample_cov**2
    pi_hat = float(pi_matrix.sum())

    # ------------------------------------------------------------------
    # γ: squared Frobenius norm of (sample - target)
    # ------------------------------------------------------------------
    diff = sample_cov - target
    gamma_hat = float((diff**2).sum())

    # ------------------------------------------------------------------
    # κ̂ = π̂ / γ̂  →  α = κ̂ / T, clipped to [0, 1]
    # ------------------------------------------------------------------
    if gamma_hat <= 0:
        # Sample equals target — no shrinkage needed
        return 0.0

    kappa = pi_hat / gamma_hat
    alpha = max(0.0, min(1.0, kappa / n_obs))
    return alpha
