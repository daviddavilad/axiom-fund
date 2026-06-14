"""Pure functions for regression residual diagnostics.

This module exposes six diagnostic functions for regression residuals:
  - compute_durbin_watson:        time-series autocorrelation
  - compute_qq_data:              residual normality data
  - compute_residual_vs_fitted_data: residual structure data
  - compute_leverage:             diagonal of hat matrix
  - compute_cooks_distance:       observation influence
  - compute_breusch_pagan:        heteroskedasticity test

All functions take raw numpy arrays (residuals, fitted values, design
matrix) and return DataFrames, floats, or dicts. No coupling to
statsmodels or any specific regression library. Plotting is a
downstream concern; this module returns data only.

Design rationale: by accepting raw arrays rather than fitted model
objects, these diagnostics apply uniformly across the codebase
regardless of how the underlying regression was estimated. The v1
codebase has two regression sites — residual momentum
(cross-sectional, monthly) and idiosyncratic volatility (FF3
trailing-60-day per permno) — both of which fit this contract.

See docs/v2_design.md Item 2 for scope and acceptance criteria.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats


def compute_durbin_watson(residuals: NDArray[np.float64]) -> float:
    """Compute the Durbin-Watson statistic for time-series autocorrelation.

    DW = sum_{t=2}^{T} (e_t - e_{t-1})^2 / sum_{t=1}^{T} e_t^2

    A DW of 2.0 indicates no autocorrelation. Values below 2.0
    indicate positive autocorrelation; values above 2.0 indicate
    negative autocorrelation. Range is [0, 4].

    This statistic is meaningful only for time-series residuals where
    observation order matters. It is undefined for cross-sectional
    residuals and should not be applied to v1's residual momentum
    regression (cross-sectional). It is appropriate for v1's
    idiosyncratic volatility regressions (time-series).

    Parameters
    ----------
    residuals : NDArray[np.float64]
        1D array of residuals in time order. Length must be >= 2.

    Returns
    -------
    float
        The Durbin-Watson statistic.

    Raises
    ------
    ValueError
        If residuals has fewer than 2 observations.
    """
    if residuals.ndim != 1:
        raise ValueError(f"residuals must be 1D, got shape {residuals.shape}")
    if len(residuals) < 2:
        raise ValueError(f"need at least 2 residuals, got {len(residuals)}")

    diffs = np.diff(residuals)
    numerator = float(np.sum(diffs ** 2))
    denominator = float(np.sum(residuals ** 2))
    if denominator == 0.0:
        raise ValueError("sum of squared residuals is zero")
    return numerator / denominator


def compute_qq_data(residuals: NDArray[np.float64]) -> pd.DataFrame:
    """Compute Q-Q plot data: theoretical vs sample quantiles.

    Returns a DataFrame with the data points that, when plotted as a
    scatter plot, produce a Q-Q plot against the standard normal
    distribution. Points falling on the diagonal indicate normally-
    distributed residuals.

    Uses scipy.stats.probplot which sorts residuals and computes the
    corresponding theoretical quantiles via the Filliben formula.

    Parameters
    ----------
    residuals : NDArray[np.float64]
        1D array of residuals.

    Returns
    -------
    pd.DataFrame
        Columns: 'theoretical_quantile', 'sample_quantile'. Length
        equals len(residuals). Rows are ordered by theoretical
        quantile (ascending).

    Raises
    ------
    ValueError
        If residuals is empty or not 1D.
    """
    if residuals.ndim != 1:
        raise ValueError(f"residuals must be 1D, got shape {residuals.shape}")
    if len(residuals) == 0:
        raise ValueError("residuals is empty")

    # probplot returns ((osm, osr), (slope, intercept, r))
    # We only need the first tuple: ordered statistic medians + ordered residuals
    (theoretical, sample), _ = stats.probplot(residuals, dist="norm", fit=True)
    return pd.DataFrame({
        "theoretical_quantile": theoretical,
        "sample_quantile": sample,
    })


def compute_residual_vs_fitted_data(
    residuals: NDArray[np.float64],
    fitted: NDArray[np.float64],
) -> pd.DataFrame:
    """Pair residuals with fitted values for residual-vs-fitted plots.

    A residual-vs-fitted scatter plot is the canonical diagnostic
    for non-linearity (curved pattern), heteroskedasticity (funnel
    shape), or outliers (isolated extreme points). This function
    returns the data; plotting is downstream.

    Parameters
    ----------
    residuals : NDArray[np.float64]
        1D array of residuals.
    fitted : NDArray[np.float64]
        1D array of fitted values, same length as residuals.

    Returns
    -------
    pd.DataFrame
        Columns: 'fitted', 'residual'. Length equals len(residuals).
        Rows in original observation order.

    Raises
    ------
    ValueError
        If inputs are not 1D, not the same length, or empty.
    """
    if residuals.ndim != 1:
        raise ValueError(f"residuals must be 1D, got shape {residuals.shape}")
    if fitted.ndim != 1:
        raise ValueError(f"fitted must be 1D, got shape {fitted.shape}")
    if len(residuals) != len(fitted):
        raise ValueError(
            f"residuals length {len(residuals)} != fitted length {len(fitted)}"
        )
    if len(residuals) == 0:
        raise ValueError("residuals is empty")

    return pd.DataFrame({
        "fitted": fitted,
        "residual": residuals,
    })


def compute_leverage(X: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute leverage values (diagonal of the hat matrix).

    Leverage h_ii measures how much an observation's design-matrix
    position influences its own fitted value. Range: [0, 1]. A high
    leverage value indicates an observation is at an extreme of the
    predictor space; combined with a large residual, it produces a
    high Cook's distance.

    Rule of thumb: average leverage is k/n (k = number of
    parameters, n = number of observations); values above 2k/n are
    typically flagged as high.

    Computed as the diagonal of H = X (X^T X)^{-1} X^T, using
    np.einsum to avoid materializing the full n×n hat matrix.

    Parameters
    ----------
    X : NDArray[np.float64]
        Design matrix of shape (n_obs, n_params). Should already
        include an intercept column if the regression has one.

    Returns
    -------
    NDArray[np.float64]
        1D array of length n_obs containing leverage values.

    Raises
    ------
    ValueError
        If X is not 2D, has zero observations, or has a singular
        X^T X (degenerate design).
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")
    n_obs, n_params = X.shape
    if n_obs == 0:
        raise ValueError("X has zero observations")
    if n_obs < n_params:
        raise ValueError(
            f"X has {n_obs} observations but {n_params} parameters; "
            f"X^T X is singular"
        )

    xtx = X.T @ X
    try:
        inv_xtx = np.linalg.inv(xtx)
    except np.linalg.LinAlgError as e:
        raise ValueError(f"X^T X is singular: {e}") from e

    # Diagonal of X @ inv_xtx @ X^T without materializing the full matrix.
    # For each row i: h_ii = x_i^T @ inv_xtx @ x_i
    leverage = np.einsum("ij,jk,ik->i", X, inv_xtx, X)
    return leverage.astype(np.float64)


def compute_cooks_distance(
    residuals: NDArray[np.float64],
    X: NDArray[np.float64],
    n_params: int,
) -> NDArray[np.float64]:
    """Compute Cook's distance for each observation.

    Cook's distance measures the influence of each observation on
    the full set of fitted values. It combines residual magnitude
    with leverage:

      D_i = (e_i^2 / (k * MSE)) * (h_ii / (1 - h_ii)^2)

    where e_i is the residual, h_ii is the leverage, k is the
    number of parameters, and MSE is the mean squared error.

    Rule of thumb: observations with D_i > 4/n or D_i > 1 are
    flagged for review.

    Parameters
    ----------
    residuals : NDArray[np.float64]
        1D array of residuals, length n_obs.
    X : NDArray[np.float64]
        Design matrix of shape (n_obs, n_params).
    n_params : int
        Number of parameters in the regression (including intercept
        if any). Passed explicitly rather than inferred from
        X.shape[1] because the caller may know the effective rank
        is lower (e.g., if dummies were dropped to avoid collinearity).

    Returns
    -------
    NDArray[np.float64]
        1D array of length n_obs containing Cook's distance values.

    Raises
    ------
    ValueError
        If inputs are mis-shaped, empty, n_params <= 0, or if any
        leverage value equals exactly 1 (would divide by zero).
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
    if n_params <= 0:
        raise ValueError(f"n_params must be positive, got {n_params}")

    leverage = compute_leverage(X)
    mse = float(np.mean(residuals ** 2))
    if mse == 0.0:
        raise ValueError("MSE is zero; cannot compute Cook's distance")

    # Guard against h_ii = 1 (would divide by zero)
    one_minus_h = 1.0 - leverage
    if np.any(one_minus_h == 0):
        raise ValueError(
            "at least one leverage value equals 1; "
            "Cook's distance is undefined"
        )

    cooks = (residuals ** 2 / (n_params * mse)) * (leverage / one_minus_h ** 2)
    return cooks.astype(np.float64)


def compute_breusch_pagan(
    residuals: NDArray[np.float64],
    X: NDArray[np.float64],
) -> dict[str, float]:
    """Breusch-Pagan test for heteroskedasticity.

    Tests whether the variance of regression residuals depends on
    the regressors. Algorithm:
      1. Compute squared residuals e_i^2
      2. Regress e_i^2 on X (auxiliary regression)
      3. LM statistic: n * R^2_aux, distributed chi^2 with (k-1) df
         under H_0 of homoskedasticity
      4. Also report the F-form

    A low p-value rejects homoskedasticity. In v1's residual
    momentum regression, heteroskedasticity would suggest that
    larger-cap (or industry-dummy-heavy) names have systematically
    different residual variance — relevant because v1's IC t-stats
    assume homoskedastic errors.

    Wraps statsmodels.stats.diagnostic.het_breuschpagan.

    Parameters
    ----------
    residuals : NDArray[np.float64]
        1D array of residuals.
    X : NDArray[np.float64]
        Design matrix of shape (n_obs, n_params). Must include an
        intercept (constant column); the test requires it.

    Returns
    -------
    dict[str, float]
        Keys: 'lm_statistic', 'lm_pvalue', 'f_statistic',
        'f_pvalue'. lm_pvalue is the standard reported value;
        F-form is included for completeness.

    Raises
    ------
    ValueError
        If inputs are mis-shaped, empty, or X has no constant
        column (intercept).
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

    # Check that X has a constant column (intercept). Without one, the
    # auxiliary regression has no baseline and the test is malformed.
    has_intercept = any(
        np.all(X[:, j] == X[0, j]) for j in range(X.shape[1])
    )
    if not has_intercept:
        raise ValueError(
            "X must contain a constant column (intercept); "
            "Breusch-Pagan requires it"
        )

    # Import locally to keep the module's top-level statsmodels
    # dependency narrow. statsmodels is a heavy import.
    from statsmodels.stats.diagnostic import het_breuschpagan

    lm_stat, lm_pvalue, f_stat, f_pvalue = het_breuschpagan(residuals, X)
    return {
        "lm_statistic": float(lm_stat),
        "lm_pvalue": float(lm_pvalue),
        "f_statistic": float(f_stat),
        "f_pvalue": float(f_pvalue),
    }
