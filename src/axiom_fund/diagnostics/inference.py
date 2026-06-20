"""HAC standard errors and bootstrapped confidence intervals.

This module implements v2 Item 3 (per docs/v2_design.md): inference
tools that do not require homoskedastic or normally-distributed
residuals. Motivated by the v2 Item 2 findings (see
docs/v2_diagnostics_findings.md) that v1's residuals exhibit
heteroskedasticity and severe non-normality.

Functions:
  - compute_hac_standard_errors: Newey-West HAC covariance matrix
    for OLS regression coefficients
  - compute_hac_standard_error_of_mean: scalar SE of a sample mean
    accounting for autocorrelation (thin wrapper around the above)
  - compute_bootstrapped_sharpe_ci: block-bootstrap confidence
    interval for the Sharpe ratio

All functions take raw numpy arrays and return scalars or arrays.
Pure numpy implementation; no statsmodels or other regression-library
dependency. The HAC estimator is implemented directly from the
Newey-West 1987 definition (Bartlett kernel, lag truncation L).

Design notes:
  - maxlags is a required parameter on HAC functions; defaults hide
    a judgment call. Common choices: L ~ N^(1/4) for low-frequency,
    L ~ 4(N/100)^(2/9) (Newey-West 1994) for higher-frequency data.
  - block_size is required for the bootstrap; defaults hide a
    judgment call. Common choice: block_size ~ N^(1/3) for monthly.
  - The bootstrap is a stationary block bootstrap (Politis-Romano
    1994) preserving short-range dependence in the return series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def compute_hac_standard_errors(
    residuals: NDArray[np.float64],
    X: NDArray[np.float64],
    maxlags: int,
) -> NDArray[np.float64]:
    """Newey-West HAC covariance matrix for OLS regression coefficients.

    Computes the heteroskedasticity-and-autocorrelation-consistent
    (HAC) covariance matrix using Bartlett weights:

      Cov_hac(beta_hat) = (X'X)^{-1} * S * (X'X)^{-1}
      S = sum_{j=-L}^{L} w(j,L) * sum_t e_t * e_{t-|j|} * x_t * x_{t-|j|}'

    where w(j,L) = 1 - |j|/(L+1) (Bartlett) and L = maxlags.

    Parameters
    ----------
    residuals : NDArray[np.float64]
        1D array of OLS residuals, length n_obs. Must be ordered in
        time; HAC is a time-series correction.
    X : NDArray[np.float64]
        Design matrix of shape (n_obs, n_params), same ordering as
        residuals. Should include an intercept column if the
        regression had one.
    maxlags : int
        Bartlett truncation L. Must be >= 0. L=0 reduces to the
        White (HC0) heteroskedasticity-only estimator.

    Returns
    -------
    NDArray[np.float64]
        Covariance matrix of shape (n_params, n_params). Standard
        errors are sqrt(diag(.)).

    Raises
    ------
    ValueError
        If inputs are mis-shaped, empty, maxlags < 0, or X^T X is
        singular.
    """
    if residuals.ndim != 1:
        raise ValueError(f"residuals must be 1D, got shape {residuals.shape}")
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")
    n_obs = len(residuals)
    if n_obs != X.shape[0]:
        raise ValueError(
            f"residuals length {n_obs} != X.shape[0] {X.shape[0]}"
        )
    if n_obs == 0:
        raise ValueError("residuals is empty")
    if maxlags < 0:
        raise ValueError(f"maxlags must be non-negative, got {maxlags}")
    if maxlags >= n_obs:
        raise ValueError(
            f"maxlags ({maxlags}) must be < n_obs ({n_obs})"
        )

    # Compute (X'X)^{-1}
    xtx = X.T @ X
    try:
        xtx_inv = np.linalg.inv(xtx)
    except np.linalg.LinAlgError as e:
        raise ValueError(f"X^T X is singular: {e}") from e

    # Compute S using Bartlett weights
    # Lag 0 term: sum_t e_t^2 * x_t * x_t^T
    s = np.zeros_like(xtx)
    for t in range(n_obs):
        s += residuals[t] ** 2 * np.outer(X[t], X[t])

    # Higher-lag terms: w(j) * sum_t (e_t * e_{t-j} * (x_t * x_{t-j}^T + x_{t-j} * x_t^T))
    for j in range(1, maxlags + 1):
        weight = 1.0 - j / (maxlags + 1.0)
        gamma_j = np.zeros_like(xtx)
        for t in range(j, n_obs):
            gamma_j += residuals[t] * residuals[t - j] * np.outer(X[t], X[t - j])
        s += weight * (gamma_j + gamma_j.T)

    cov = xtx_inv @ s @ xtx_inv
    return cov.astype(np.float64)


def compute_hac_standard_error_of_mean(
    series: NDArray[np.float64],
    maxlags: int,
) -> float:
    """HAC-corrected standard error of the sample mean of a time series.

    For inference on the mean of an autocorrelated, possibly
    heteroskedastic series (e.g., a time series of monthly IC values
    or monthly portfolio returns). Wraps compute_hac_standard_errors
    with X = column of ones; the resulting cov(intercept) is the
    HAC-corrected variance of the sample mean.

    Parameters
    ----------
    series : NDArray[np.float64]
        1D time series of length n.
    maxlags : int
        Bartlett truncation L. See compute_hac_standard_errors.

    Returns
    -------
    float
        HAC-corrected standard error of the mean (positive).

    Raises
    ------
    ValueError
        Same conditions as compute_hac_standard_errors.
    """
    if series.ndim != 1:
        raise ValueError(f"series must be 1D, got shape {series.shape}")
    n = len(series)
    if n == 0:
        raise ValueError("series is empty")

    # Construct the "intercept-only" regression
    X = np.ones((n, 1), dtype=np.float64)
    mean = float(series.mean())
    residuals = series - mean
    cov = compute_hac_standard_errors(residuals, X, maxlags)
    return float(np.sqrt(cov[0, 0]))


def compute_bootstrapped_sharpe_ci(
    returns: NDArray[np.float64],
    block_size: int,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int | None = None,
) -> tuple[float, float]:
    """Block-bootstrap confidence interval for the Sharpe ratio.

    Uses the stationary (overlapping) block bootstrap: for each of
    n_resamples bootstrap iterations, sample blocks of length
    block_size with replacement from the returns series, concatenate
    until the bootstrap sample matches the length of the original,
    compute the Sharpe ratio, and record it. Confidence interval
    from the percentile method.

    The block bootstrap preserves short-range serial dependence in
    the returns (clustering of volatility, autocorrelation). For
    block_size = 1 it reduces to the standard i.i.d. bootstrap,
    which is biased for serially-correlated data.

    Sharpe ratio computed as mean / std (no annualization here; the
    caller annualizes the returns first if desired).

    Parameters
    ----------
    returns : NDArray[np.float64]
        1D array of returns. Must be in time order.
    block_size : int
        Size of contiguous blocks resampled from the original series.
        Required (no default); the choice depends on the series'
        autocorrelation. Rule of thumb: N^(1/3).
    n_resamples : int, default 10_000
        Number of bootstrap samples to draw.
    confidence : float, default 0.95
        Confidence level for the interval. Must be in (0, 1).
    seed : int | None, default None
        Random seed for reproducibility. None uses the global RNG.

    Returns
    -------
    tuple[float, float]
        (lower, upper) confidence interval for the Sharpe ratio.

    Raises
    ------
    ValueError
        If inputs are mis-shaped, block_size invalid, or confidence
        outside (0, 1).
    """
    if returns.ndim != 1:
        raise ValueError(f"returns must be 1D, got shape {returns.shape}")
    n = len(returns)
    if n < 2:
        raise ValueError(f"need at least 2 returns, got {n}")
    if block_size < 1 or block_size > n:
        raise ValueError(
            f"block_size must be in [1, {n}], got {block_size}"
        )
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be >= 1, got {n_resamples}")
    if not (0 < confidence < 1):
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    rng = np.random.default_rng(seed)
    sharpes = np.empty(n_resamples, dtype=np.float64)

    # Number of blocks per bootstrap sample to reach length n
    n_blocks = (n + block_size - 1) // block_size

    for i in range(n_resamples):
        # Sample n_blocks starting indices uniformly from [0, n-block_size+1)
        starts = rng.integers(0, n - block_size + 1, size=n_blocks)
        # Construct the bootstrap sample by concatenating blocks
        sample = np.concatenate([
            returns[s : s + block_size] for s in starts
        ])[:n]
        std = sample.std(ddof=1)
        if std == 0:
            sharpes[i] = np.nan
        else:
            sharpes[i] = sample.mean() / std

    # Filter any NaNs (degenerate samples) before computing percentiles
    valid = sharpes[~np.isnan(sharpes)]
    if len(valid) == 0:
        raise ValueError("all bootstrap samples had zero std; cannot compute CI")
    alpha = (1.0 - confidence) / 2.0
    lower = float(np.quantile(valid, alpha))
    upper = float(np.quantile(valid, 1.0 - alpha))
    return lower, upper
