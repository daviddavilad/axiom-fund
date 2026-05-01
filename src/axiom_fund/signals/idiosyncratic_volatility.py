"""Idiosyncratic Volatility signal (Ang, Hodrick, Xing, Zhang 2006).

For each (permno, trading_date), runs an OLS regression of trailing
60 days of stock excess returns on the Fama-French 3 factors:

    (r_stock - r_f) = α + β_mkt × MKTRF + β_smb × SMB + β_hml × HML + ε

The signal is the standard deviation of regression residuals (idiosyncratic
component), annualized by × √252.

Conceptually: stocks with high IVol bounce around for firm-specific reasons
not explained by market, size, or value exposure. AHXZ 2006 documented that
high-IVol names tend to underperform — the foundation for shorting them in
a market-neutral L/S strategy.

Output panel columns (in order):
  permno
  date_filed       trading date this regression terminates on (PIT anchor)
  n_obs            number of valid observations in the 60-day window
  beta_mkt         market factor loading (diagnostic)
  beta_smb         size factor loading (diagnostic)
  beta_hml         value factor loading (diagnostic)
  residual_std     std of residuals in the window (daily, decimal)
  raw_signal       annualized: residual_std × √252

Implementation notes
--------------------
- Returns and FF factors are inner-joined on date. Days where either is
  missing are excluded entirely.
- A regression is computed only when the trailing 60-day window has at
  least 40 valid observations (configurable). Below that, the row is
  dropped from output rather than emitted as NaN, because the diagnostic
  columns (betas, residual_std) wouldn't be meaningful.
- Output is keyed by trading date — natural cadence is daily. Alignment
  to rebalance dates happens in alignment.py (see docs/signal_design.md).

Reference
---------
Ang, A., Hodrick, R., Xing, Y., Zhang, X. (2006). "The cross-section of
volatility and expected returns." Journal of Finance, 61(1), 259-299.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import statsmodels.api as sm

# Output column order is part of the module's public API.
IVOL_RAW_COLUMNS: tuple[str, ...] = (
    "permno",
    "date_filed",
    "n_obs",
    "beta_mkt",
    "beta_smb",
    "beta_hml",
    "residual_std",
    "raw_signal",
)

# Locked parameters per docs/strategy_spec.md §5 and signal_design.md §4.2
_WINDOW_SIZE: int = 60
_MIN_OBS: int = 40
_TRADING_DAYS_PER_YEAR: int = 252


def compute_idiosyncratic_volatility(
    returns_df: pd.DataFrame,
    ff_factors_df: pd.DataFrame,
    start_date: str | date,
    end_date: str | date,
    window_size: int = _WINDOW_SIZE,
    min_obs: int = _MIN_OBS,
) -> pd.DataFrame:
    """Compute the raw IVol signal per (permno, trading_date).

    Parameters
    ----------
    returns_df : pandas.DataFrame
        Long-format daily returns panel from ReturnsPanel. Must contain:
        'permno', 'date', 'ret'. Returns in decimal form (0.0079 = 0.79%).
    ff_factors_df : pandas.DataFrame
        Long-format daily FF factors from FFFactors. Must contain:
        'date', 'mktrf', 'smb', 'hml', 'rf'. All in decimal form.
    start_date, end_date : str or date
        Inclusive window on output `date_filed`. Note: returns_df should
        be pre-fetched with extra leading buffer (window_size trading days
        before start_date) so the rolling regression has data to begin with.
    window_size : int, default 60
        Trailing window size in trading days for each regression.
    min_obs : int, default 40
        Minimum valid observations required within the window to emit a row.

    Returns
    -------
    pandas.DataFrame
        Long-format raw signal panel sorted by (date_filed, permno) with
        columns matching IVOL_RAW_COLUMNS.

    Raises
    ------
    ValueError
        If start_date > end_date, required columns missing, or invalid params.
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    start_str = _normalize_date(start_date)
    end_str = _normalize_date(end_date)
    if start_str > end_str:
        raise ValueError(
            f"start_date ({start_str}) must be <= end_date ({end_str})"
        )

    required_returns = {"permno", "date", "ret"}
    if not required_returns.issubset(returns_df.columns):
        missing = required_returns - set(returns_df.columns)
        raise ValueError(f"returns_df missing columns: {sorted(missing)}")

    required_ff = {"date", "mktrf", "smb", "hml", "rf"}
    if not required_ff.issubset(ff_factors_df.columns):
        missing = required_ff - set(ff_factors_df.columns)
        raise ValueError(f"ff_factors_df missing columns: {sorted(missing)}")

    if window_size < 2:
        raise ValueError(f"window_size must be >= 2, got {window_size}")
    if min_obs < 4:
        # Need at least n>k+1=4+1=5 for OLS with 3 factors + intercept,
        # but we relax to 4 since residuals from saturated fit are fine.
        raise ValueError(f"min_obs must be >= 4 (regression has 4 params), got {min_obs}")
    if min_obs > window_size:
        raise ValueError(
            f"min_obs ({min_obs}) cannot exceed window_size ({window_size})"
        )

    # ------------------------------------------------------------------
    # Normalize and join
    # ------------------------------------------------------------------
    rets = returns_df.copy()
    rets["date"] = pd.to_datetime(rets["date"]).astype("datetime64[ns]")

    ff = ff_factors_df.copy()
    ff["date"] = pd.to_datetime(ff["date"]).astype("datetime64[ns]")

    # Inner join: drop any date where either side is missing
    merged = rets.merge(ff[["date", "mktrf", "smb", "hml", "rf"]], on="date", how="inner")

    # Excess return
    merged["excess_ret"] = merged["ret"] - merged["rf"]

    # Drop rows where excess_ret is NaN (e.g., missing return)
    merged = merged.dropna(subset=["excess_ret"])

    if len(merged) == 0:
        return pd.DataFrame(columns=list(IVOL_RAW_COLUMNS))

    # ------------------------------------------------------------------
    # Rolling regression per PERMNO
    # ------------------------------------------------------------------
    output_rows = []
    start_ts = pd.Timestamp(start_str)
    end_ts = pd.Timestamp(end_str)

    for permno, group in merged.groupby("permno", sort=False):
        group = group.sort_values("date").reset_index(drop=True)

        # For each terminal date in the group, run regression on trailing window
        for i in range(window_size - 1, len(group)):
            terminal_date = group["date"].iloc[i]

            # Skip rows outside the requested output window
            if terminal_date < start_ts or terminal_date > end_ts:
                continue

            window_slice = group.iloc[i - window_size + 1 : i + 1]
            n_valid = window_slice["excess_ret"].notna().sum()

            if n_valid < min_obs:
                continue

            result = _run_ff3_regression(window_slice)
            if result is None:
                continue

            output_rows.append(
                {
                    "permno": permno,
                    "date_filed": terminal_date,
                    "n_obs": n_valid,
                    "beta_mkt": result["beta_mkt"],
                    "beta_smb": result["beta_smb"],
                    "beta_hml": result["beta_hml"],
                    "residual_std": result["residual_std"],
                    "raw_signal": result["residual_std"] * np.sqrt(_TRADING_DAYS_PER_YEAR),
                }
            )

    if not output_rows:
        return pd.DataFrame(columns=list(IVOL_RAW_COLUMNS))

    result_df = pd.DataFrame(output_rows)
    result_df = result_df[list(IVOL_RAW_COLUMNS)]
    return result_df.sort_values(["date_filed", "permno"]).reset_index(drop=True)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _run_ff3_regression(window_df: pd.DataFrame) -> dict[str, float] | None:
    """Run a single FF3 regression on a window of (excess_ret, factors).

    Returns dict with beta_mkt, beta_smb, beta_hml, residual_std.
    Returns None if regression fails (e.g., singular matrix).
    """
    # Drop NaN rows for clean regression
    clean = window_df.dropna(subset=["excess_ret", "mktrf", "smb", "hml"])
    if len(clean) < 4:  # need at least k+1 = 4 obs for 3-factor + intercept
        return None

    y = clean["excess_ret"].to_numpy()
    X = clean[["mktrf", "smb", "hml"]].to_numpy()  # noqa: N806 (statistical convention)
    X = sm.add_constant(X)  # noqa: N806 (statistical convention)

    try:
        model = sm.OLS(y, X)
        results = model.fit()
    except (np.linalg.LinAlgError, ValueError):
        return None

    # results.params: [intercept, beta_mkt, beta_smb, beta_hml]
    # results.resid:  residuals over the window
    return {
        "beta_mkt": float(results.params[1]),
        "beta_smb": float(results.params[2]),
        "beta_hml": float(results.params[3]),
        "residual_std": float(np.std(results.resid, ddof=1)),
    }


def _normalize_date(d: str | date) -> str:
    """Normalize date input to ISO-format string."""
    if isinstance(d, date):
        return d.isoformat()
    if isinstance(d, str):
        parsed = date.fromisoformat(d)
        return parsed.isoformat()
    raise ValueError(f"date must be str or date, got {type(d).__name__}")
