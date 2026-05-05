"""Per-asset market beta estimation via rolling OLS regression.

For each PERMNO with sufficient history, estimates the market beta β_i:

    r_i,t  =  α_i  +  β_i × mktrf_t  +  ε_i,t

over a trailing window (default 252 trading days, ending at as_of_date).

This is the same regression structure as IVol (idiosyncratic_volatility),
but here we only need β_i (the market exposure), not the residuals. Used
by the portfolio optimizer to enforce beta-neutrality:  β' w = 0.

Lookback choice
---------------
252 days (~1 year) is the standard practitioner window for beta estimation.
Beta is more stable over time than IVol, so we want enough data to get a
robust estimate without smoothing across regime changes. Per spec.

NaN handling
------------
- Returns or mktrf with NaN are dropped before regression
- Names with fewer than min_obs valid observations get NaN beta in output
  (caller should drop these names from the optimizer universe)

Output
------
A pandas Series indexed by PERMNO with float values. NaN for names without
sufficient data.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd

_DEFAULT_WINDOW: int = 252
_DEFAULT_MIN_OBS: int = 60  # need at least 3 months of data


def compute_betas(
    returns_df: pd.DataFrame,
    ff_factors_df: pd.DataFrame,
    as_of_date: str | pd.Timestamp,
    window: int = _DEFAULT_WINDOW,
    min_obs: int = _DEFAULT_MIN_OBS,
) -> pd.Series:
    """Compute per-asset market beta as of as_of_date.

    Parameters
    ----------
    returns_df : pd.DataFrame
        Long-format daily returns panel with at least 'permno', 'date',
        'ret' columns. Same schema as ReturnsPanel.fetch() output.
    ff_factors_df : pd.DataFrame
        Daily Fama-French factors with at least 'date' and 'mktrf'
        columns. Same schema as FFFactors.fetch() output.
    as_of_date : str or pd.Timestamp
        End of estimation window. Beta is computed using returns from
        (as_of_date - window) to as_of_date inclusive.
    window : int, default 252
        Number of trailing trading days to use.
    min_obs : int, default 60
        Minimum number of valid (return, mktrf) pairs required to
        estimate a beta. Names with fewer obs get NaN.

    Returns
    -------
    pd.Series
        Index: permno (int). Values: beta (float, NaN if insufficient
        data). Series is named 'beta'.

    Raises
    ------
    ValueError
        If required columns are missing or window/min_obs are invalid.
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    required_returns = {"permno", "date", "ret"}
    if not required_returns.issubset(returns_df.columns):
        missing = required_returns - set(returns_df.columns)
        raise ValueError(f"returns_df missing columns: {sorted(missing)}")

    required_ff = {"date", "mktrf"}
    if not required_ff.issubset(ff_factors_df.columns):
        missing = required_ff - set(ff_factors_df.columns)
        raise ValueError(f"ff_factors_df missing columns: {sorted(missing)}")

    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    if min_obs < 2:
        raise ValueError(f"min_obs must be >= 2, got {min_obs}")
    if min_obs > window:
        raise ValueError(f"min_obs ({min_obs}) must be <= window ({window})")

    # ------------------------------------------------------------------
    # Window selection
    # ------------------------------------------------------------------
    as_of = pd.to_datetime(as_of_date).normalize()
    rets = returns_df.copy()
    rets["date"] = pd.to_datetime(rets["date"]).dt.normalize()

    ff = ff_factors_df.copy()
    ff["date"] = pd.to_datetime(ff["date"]).dt.normalize()

    # Filter both to dates ≤ as_of_date, keep only most recent `window`
    # trading dates available
    available_dates = sorted(set(ff["date"]) & set(rets["date"]))
    available_dates = [d for d in available_dates if d <= as_of]
    if len(available_dates) < min_obs:
        # Not enough data overall to estimate any betas
        return pd.Series([], name="beta", dtype=float)

    window_dates = available_dates[-window:]
    window_start = window_dates[0]

    rets_window = rets[
        (rets["date"] >= window_start) & (rets["date"] <= as_of)
    ][["permno", "date", "ret"]]
    ff_window = ff[
        (ff["date"] >= window_start) & (ff["date"] <= as_of)
    ][["date", "mktrf"]]

    # Merge returns with mktrf on date
    merged = rets_window.merge(ff_window, on="date", how="inner")
    merged = merged.dropna(subset=["ret", "mktrf"])

    if len(merged) == 0:
        return pd.Series([], name="beta", dtype=float)

    # ------------------------------------------------------------------
    # OLS per permno via vectorized formula:
    #     β_i = Cov(r_i, mktrf) / Var(mktrf)
    # We compute this per group rather than running a full OLS — it's
    # equivalent for a single regressor, and much faster.
    # ------------------------------------------------------------------
    betas: dict[int, float] = {}

    for permno, group in merged.groupby("permno"):
        n = len(group)
        if n < min_obs:
            betas[int(permno)] = float("nan")
            continue

        r = group["ret"].to_numpy()
        m = group["mktrf"].to_numpy()

        m_mean = m.mean()
        m_var = ((m - m_mean) ** 2).sum() / (n - 1)

        if m_var < 1e-20:
            # Market is essentially constant in this window — degenerate
            betas[int(permno)] = float("nan")
            continue

        r_mean = r.mean()
        cov = ((r - r_mean) * (m - m_mean)).sum() / (n - 1)

        betas[int(permno)] = float(cov / m_var)

    return pd.Series(betas, name="beta", dtype=float).sort_index()


# ----------------------------------------------------------------------
# Helpers: enable testing of the math without going through the full
# permno groupby pipeline.
# ----------------------------------------------------------------------


def _ols_beta(
    stock_returns: npt.NDArray[np.float64],
    market_returns: npt.NDArray[np.float64],
) -> float:
    """Single-stock OLS beta from two aligned 1D arrays.

    Beta = Cov(r, m) / Var(m). Used directly in tests.
    """
    n = len(stock_returns)
    if n != len(market_returns):
        raise ValueError("arrays must be same length")
    if n < 2:
        raise ValueError("need at least 2 observations")

    m_mean = market_returns.mean()
    m_var = ((market_returns - m_mean) ** 2).sum() / (n - 1)
    if m_var < 1e-20:
        raise ValueError("market variance is essentially zero")

    r_mean = stock_returns.mean()
    cov = ((stock_returns - r_mean) * (market_returns - m_mean)).sum() / (n - 1)
    return float(cov / m_var)
