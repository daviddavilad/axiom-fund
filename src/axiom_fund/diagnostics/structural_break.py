"""Structural-break testing: Quandt-Andrews sup-F with Hansen p-values.

This module implements v2 Item 5 (per docs/v2_design.md). The
Quandt-Andrews test detects a single unknown break point in
regression coefficients by computing Chow F-statistics at every
candidate break in the middle (1 - 2π₀) fraction of the sample
and taking the supremum.

The supremum statistic does not have a standard asymptotic
distribution. Critical values come from Andrews (1993, Econometrica
61:821-856); p-value approximation from Hansen (1997, JBES
15(1):60-67) using polynomial-transformed chi-squared.

Functions:
  - compute_quandt_andrews: sup-F statistic, break location, and
    Hansen p-value for a regression structural-break test
  - compute_hansen_pvalue: Hansen (1997) chi-squared approximation
    for SupF asymptotic distribution. Exposed in case the caller
    wants to compute p-values for sup-F values computed elsewhere.

Pure numpy + scipy.stats.chi2. Coefficients embedded from Hansen
(1997) Table 2 (SupF distribution); supports m ∈ {1..20, 25, 30,
35, 40} and π₀ ∈ {0.01, 0.05, 0.15, 0.25, 0.35}. Outside this grid,
the function raises rather than extrapolating.

For our v1 IC time series use case (testing for break in mean of
a single signal's IC), m = 1 and π₀ = 0.15. Other (m, π₀) values
are supported because future Phase 2 work may need them.

Implementation references:
  - Quandt (1960), Andrews (1993): theoretical foundation
  - Hansen (1997) eq. (8) and Table 2: p-value approximation
  - Hansen (1997) reports approximation error: median 0.0006, max
    0.0030 for p in [0, 0.5] — well within reporting precision
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import chi2


# Hansen (1997) Table 2: SupF distribution coefficients.
# Indexed by (m, π₀). Values are (θ₀, θ₁, η) where the p-value is
# p = 1 - chi2.cdf(θ₀ + θ₁ * sup_F, df=η).
# For SupF, the polynomial degree is 1 (linear in sup_F).
_HANSEN_SUPF_TABLE: dict[tuple[int, float], tuple[float, float, float]] = {
    # (m, π₀): (θ₀, θ₁, η)
    (1, 0.01): (-1.79, 1.17, 4.5),   (1, 0.05): (-1.39, 1.07, 3.6),
    (1, 0.15): (-0.99, 1.02, 3.0),   (1, 0.25): (-0.73, 0.98, 2.5),
    (1, 0.35): (-0.50, 0.96, 2.1),
    (2, 0.01): (-3.06, 1.18, 6.1),   (2, 0.05): (-2.38, 1.11, 5.4),
    (2, 0.15): (-1.65, 1.06, 4.7),   (2, 0.25): (-1.16, 1.02, 4.1),
    (2, 0.35): (-0.78, 0.97, 3.5),
    (3, 0.01): (-4.09, 1.21, 7.8),   (3, 0.05): (-3.31, 1.10, 6.5),
    (3, 0.15): (-2.05, 1.13, 6.8),   (3, 0.25): (-1.61, 1.03, 5.5),
    (3, 0.35): (-1.06, 1.01, 4.9),
    (4, 0.01): (-5.33, 1.21, 8.9),   (4, 0.05): (-4.08, 1.14, 8.2),
    (4, 0.15): (-2.52, 1.11, 8.0),   (4, 0.25): (-1.91, 1.04, 7.0),
    (4, 0.35): (-1.45, 0.97, 5.7),
    (5, 0.01): (-6.39, 1.18, 9.4),   (5, 0.05): (-4.84, 1.15, 9.3),
    (5, 0.15): (-3.46, 1.07, 8.3),   (5, 0.25): (-2.63, 1.02, 7.5),
    (5, 0.35): (-1.82, 1.00, 7.0),
    (6, 0.01): (-7.08, 1.26, 11.8),  (6, 0.05): (-5.37, 1.19, 11.2),
    (6, 0.15): (-4.05, 1.08, 9.5),   (6, 0.25): (-2.94, 1.05, 9.0),
    (6, 0.35): (-1.79, 1.03, 8.6),
    (7, 0.01): (-8.49, 1.17, 11.1),  (7, 0.05): (-6.21, 1.21, 12.6),
    (7, 0.15): (-4.42, 1.10, 11.0),  (7, 0.25): (-3.23, 1.05, 10.1),
    (7, 0.35): (-2.21, 1.01, 9.3),
    (8, 0.01): (-9.20, 1.17, 12.2),  (8, 0.05): (-7.24, 1.13, 11.9),
    (8, 0.15): (-5.36, 1.08, 11.3),  (8, 0.25): (-3.65, 1.06, 11.4),
    (8, 0.35): (-1.69, 1.10, 12.2),
    (9, 0.01): (-10.22, 1.14, 12.3), (9, 0.05): (-8.07, 1.11, 12.4),
    (9, 0.15): (-5.43, 1.10, 13.1),  (9, 0.25): (-4.38, 1.01, 11.3),
    (9, 0.35): (-2.83, 1.00, 11.1),
    (10, 0.01): (-11.01, 1.14, 13.3),  (10, 0.05): (-8.84, 1.11, 13.2),
    (10, 0.15): (-6.47, 1.06, 12.8),   (10, 0.25): (-4.97, 1.01, 12.0),
    (10, 0.35): (-2.92, 1.05, 13.0),
    (11, 0.01): (-11.90, 1.11, 13.4),  (11, 0.05): (-9.56, 1.06, 13.1),
    (11, 0.15): (-6.79, 1.04, 13.5),   (11, 0.25): (-4.62, 1.05, 14.4),
    (11, 0.35): (-3.26, 1.01, 13.4),
    (12, 0.01): (-12.88, 1.06, 12.8),  (12, 0.05): (-10.35, 1.09, 14.5),
    (12, 0.15): (-7.80, 1.02, 13.6),   (12, 0.25): (-5.32, 1.05, 15.1),
    (12, 0.35): (-3.91, 1.00, 13.8),
    (13, 0.01): (-13.88, 1.09, 14.1),  (13, 0.05): (-11.07, 1.07, 14.8),
    (13, 0.15): (-7.93, 1.07, 15.9),   (13, 0.25): (-5.80, 1.04, 15.8),
    (13, 0.35): (-4.14, 1.00, 14.9),
    (14, 0.01): (-14.61, 1.15, 16.6),  (14, 0.05): (-11.52, 1.11, 16.8),
    (14, 0.15): (-8.54, 1.05, 16.1),   (14, 0.25): (-5.90, 1.05, 17.2),
    (14, 0.35): (-4.06, 1.02, 16.5),
    (15, 0.01): (-15.49, 1.04, 14.1),  (15, 0.05): (-12.44, 1.08, 16.6),
    (15, 0.15): (-9.05, 1.05, 17.2),   (15, 0.25): (-6.59, 1.04, 17.6),
    (15, 0.35): (-3.10, 1.08, 20.0),
    (16, 0.01): (-16.34, 1.15, 17.8),  (16, 0.05): (-12.27, 1.20, 21.4),
    (16, 0.15): (-9.13, 1.09, 19.3),   (16, 0.25): (-7.00, 1.04, 18.5),
    (16, 0.35): (-4.79, 0.99, 17.6),
    (17, 0.01): (-17.20, 1.15, 18.8),  (17, 0.05): (-13.73, 1.15, 20.1),
    (17, 0.15): (-10.45, 1.05, 18.3),  (17, 0.25): (-7.23, 1.05, 19.8),
    (17, 0.35): (-5.01, 1.02, 19.1),
    (18, 0.01): (-18.10, 1.17, 20.0),  (18, 0.05): (-14.15, 1.14, 21.2),
    (18, 0.15): (-10.63, 1.05, 19.5),  (18, 0.25): (-7.76, 1.04, 20.2),
    (18, 0.35): (-5.11, 1.02, 20.2),
    (19, 0.01): (-18.19, 1.04, 17.2),  (19, 0.05): (-14.94, 0.97, 16.3),
    (19, 0.15): (-12.14, 0.90, 14.9),  (19, 0.25): (-9.84, 0.89, 15.3),
    (19, 0.35): (-7.09, 0.91, 16.8),
    (20, 0.01): (-18.99, 1.02, 17.0),  (20, 0.05): (-16.09, 0.99, 17.0),
    (20, 0.15): (-12.14, 0.97, 18.3),  (20, 0.25): (-8.87, 1.00, 20.5),
    (20, 0.35): (-5.94, 1.00, 21.3),
    (25, 0.01): (-23.42, 1.06, 21.0),  (25, 0.05): (-19.06, 1.06, 23.4),
    (25, 0.15): (-14.16, 1.05, 25.0),  (25, 0.25): (-10.65, 1.03, 25.7),
    (25, 0.35): (-6.57, 1.02, 27.1),
    (30, 0.01): (-27.30, 1.03, 22.5),  (30, 0.05): (-22.91, 1.04, 25.1),
    (30, 0.15): (-17.06, 1.03, 27.8),  (30, 0.25): (-11.51, 1.07, 32.8),
    (30, 0.35): (-6.79, 1.05, 34.1),
    (35, 0.01): (-30.01, 0.92, 20.0),  (35, 0.05): (-25.88, 0.97, 25.2),
    (35, 0.15): (-20.09, 0.98, 28.4),  (35, 0.25): (-15.78, 0.97, 30.2),
    (35, 0.35): (-10.44, 0.98, 33.1),
    (40, 0.01): (-34.24, 0.97, 24.8),  (40, 0.05): (-29.24, 0.98, 28.0),
    (40, 0.15): (-21.65, 1.05, 36.7),  (40, 0.25): (-14.18, 1.07, 42.7),
    (40, 0.35): (-11.95, 0.94, 35.0),
}

_VALID_M = sorted({m for (m, _) in _HANSEN_SUPF_TABLE})
_VALID_PI0 = sorted({pi for (_, pi) in _HANSEN_SUPF_TABLE})


def compute_hansen_pvalue(
    sup_f: float, m: int, trimming: float = 0.15,
) -> float:
    """Hansen (1997) chi-squared p-value approximation for SupF.

    Looks up the polynomial coefficients (θ₀, θ₁, η) from Hansen
    (1997) Table 2 indexed by (m, π₀ = trimming), then returns

      p = 1 - chi2_cdf(θ₀ + θ₁ * sup_f, df=η)

    Hansen reports median approximation error 0.0006 and maximum
    0.0030 for p-values in [0, 0.5]; suitable for any reporting
    precision below three decimal places.

    Parameters
    ----------
    sup_f : float
        Observed sup-F statistic (the supremum of the Chow F-stat
        across candidate break dates). Must be non-negative.
    m : int
        Number of restrictions tested (e.g., m = 1 for a break in
        a single regression coefficient or the mean of a series).
        Must be one of: 1..20, 25, 30, 35, 40.
    trimming : float, default 0.15
        Trimming parameter π₀: candidate break points are searched
        over [π₀, 1 - π₀] of the sample. Must be one of: 0.01,
        0.05, 0.15, 0.25, 0.35.

    Returns
    -------
    float
        p-value in [0, 1]. Small values reject the null of no
        structural break.

    Raises
    ------
    ValueError
        If sup_f is negative, m is outside the tabulated set, or
        trimming is outside the tabulated set.
    """
    if sup_f < 0:
        raise ValueError(f"sup_f must be non-negative, got {sup_f}")
    if m not in _VALID_M:
        raise ValueError(
            f"m must be one of {_VALID_M}, got {m}. "
            f"Extending requires adding rows from Hansen (1997) Table 2."
        )
    # Match trimming to tabulated values with floating-point tolerance
    pi0 = None
    for valid_pi in _VALID_PI0:
        if abs(trimming - valid_pi) < 1e-9:
            pi0 = valid_pi
            break
    if pi0 is None:
        raise ValueError(
            f"trimming must be one of {_VALID_PI0}, got {trimming}. "
            f"Extending requires interpolating Hansen Table 2 or "
            f"using a different reference."
        )

    theta0, theta1, eta = _HANSEN_SUPF_TABLE[(m, pi0)]
    transformed = theta0 + theta1 * sup_f
    if transformed <= 0:
        # chi-squared CDF is 0 here, so p = 1 (no rejection)
        return 1.0
    return float(1.0 - chi2.cdf(transformed, df=eta))


def compute_quandt_andrews(
    y: NDArray[np.float64],
    X: NDArray[np.float64],
    trimming: float = 0.15,
) -> tuple[float, int, float]:
    """Quandt-Andrews sup-F test for structural break at unknown date.

    Tests whether the regression y = X β + ε has a single break in
    β at some unknown date t* in the middle (1 - 2π₀) fraction of
    the sample. The Chow F-statistic is computed at every candidate
    break date in the trimmed window, and the supremum is reported
    along with its location and Hansen (1997) p-value.

    Algorithm:
      1. Fit the full-sample regression; record SSR_R.
      2. For each candidate break t ∈ [⌈π₀ T⌉, ⌊(1-π₀) T⌋]:
         - Fit two sub-sample regressions (pre and post t)
         - SSR_U(t) = SSR_pre + SSR_post
         - F(t) = ((SSR_R - SSR_U(t)) / k) / (SSR_U(t) / (T - 2k))
      3. sup_F = max F(t); break_index = argmax F(t)
      4. p-value via Hansen (1997) Table 2 lookup with m = k

    Parameters
    ----------
    y : NDArray[np.float64]
        1D array of T outcome observations. Must be in time order.
    X : NDArray[np.float64]
        Design matrix of shape (T, k). Include an intercept column
        if testing for a break in the mean. For testing a break in
        the mean of a series, use X = ones((T, 1)) and k = 1.
    trimming : float, default 0.15
        Trimming parameter π₀: candidate break dates are searched
        over [⌈π₀ T⌉, ⌊(1 - π₀) T⌋]. Standard choice is 0.15.

    Returns
    -------
    tuple[float, int, float]
        (sup_F_statistic, break_index, hansen_pvalue) where
        break_index is the 0-indexed position in y at which the
        supremum was attained.

    Raises
    ------
    ValueError
        If inputs are mis-shaped, T is too small for the trimming,
        the design has zero observations in either sub-sample, or
        the resulting m is outside Hansen's tabulated set.
    """
    if y.ndim != 1:
        raise ValueError(f"y must be 1D, got shape {y.shape}")
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")
    T = len(y)
    if T != X.shape[0]:
        raise ValueError(f"y length {T} != X.shape[0] {X.shape[0]}")
    k = X.shape[1]
    if T < 2 * k + 2:
        raise ValueError(
            f"T = {T} too small for k = {k}; need T >= 2k + 2 for sub-sample fits"
        )
    if not (0 < trimming < 0.5):
        raise ValueError(f"trimming must be in (0, 0.5), got {trimming}")

    # Candidate break-date range
    t_lo = int(np.ceil(trimming * T))
    t_hi = int(np.floor((1.0 - trimming) * T))
    # Sub-sample regressions need at least k+1 obs each side
    t_lo = max(t_lo, k + 1)
    t_hi = min(t_hi, T - k - 1)
    if t_hi <= t_lo:
        raise ValueError(
            f"trimming = {trimming} leaves no candidate breaks for T = {T}, k = {k}"
        )

    # Full-sample regression: SSR_R
    beta_full, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid_full = y - X @ beta_full
    ssr_r = float(resid_full @ resid_full)

    # Iterate over candidate breaks, recording F(t)
    best_f = -np.inf
    best_t = t_lo
    for t in range(t_lo, t_hi + 1):
        # Pre-sample: rows [0, t)
        # Post-sample: rows [t, T)
        X_pre, y_pre = X[:t], y[:t]
        X_post, y_post = X[t:], y[t:]
        # Sub-sample regressions
        beta_pre, _, _, _ = np.linalg.lstsq(X_pre, y_pre, rcond=None)
        beta_post, _, _, _ = np.linalg.lstsq(X_post, y_post, rcond=None)
        resid_pre = y_pre - X_pre @ beta_pre
        resid_post = y_post - X_post @ beta_post
        ssr_u = float(resid_pre @ resid_pre + resid_post @ resid_post)
        if ssr_u <= 0:
            continue  # perfect fit somewhere; skip
        f_stat = ((ssr_r - ssr_u) / k) / (ssr_u / (T - 2 * k))
        if f_stat > best_f:
            best_f = f_stat
            best_t = t

    if best_f < 0:
        # SSR_U > SSR_R for all candidates — numerical artifact
        best_f = 0.0

    pvalue = compute_hansen_pvalue(best_f, m=k, trimming=trimming)
    return float(best_f), best_t, pvalue