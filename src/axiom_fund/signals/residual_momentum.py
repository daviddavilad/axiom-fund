"""Residual Momentum signal (Blitz, Huij, Martens 2011).

For each (permno, month_end), the signal is the sum of cross-sectional
residuals over months t-12 through t-2 (skipping t-1 to avoid contamination
from short-term reversal), where each monthly residual is computed by
regressing that month's stock returns on industry dummies and log market
cap, then taking the residual.

Conceptually: standard 12-1 momentum measures cumulative returns. But raw
returns reflect both stock-specific performance AND industry/size effects.
Residual momentum strips out the industry and size components, leaving the
"pure" idiosyncratic momentum that has the strongest predictive power
(Blitz et al. 2011).

Pipeline (per month):
  1. Aggregate daily returns to monthly: (1 + r_d)_compounded - 1
  2. Cross-sectional regression: r_monthly ~ industry_dummies + log(mkt_cap)
  3. Extract residual per stock
  4. For each stock at month t, sum residuals from t-12 through t-2

Output panel columns (in order):
  permno
  date_filed       month-end date (PIT anchor)
  n_obs            number of months in the t-12 to t-2 window with valid residual
  size             log market cap at month_end (diagnostic)
  ggroup           GICS industry group (diagnostic)
  raw_signal       sum of residuals from t-12 to t-2

Reference
---------
Blitz, D., Huij, J., Martens, M. (2011). "Residual momentum."
Journal of Empirical Finance, 18(3), 506-521.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import numpy.typing as npt
import pandas as pd
import statsmodels.api as sm

# Output column order is part of the module's public API.
RESMOM_RAW_COLUMNS: tuple[str, ...] = (
    "permno",
    "date_filed",
    "n_obs",
    "size",
    "ggroup",
    "raw_signal",
)

# Locked parameters per docs/strategy_spec.md §5
_FORMATION_START: int = 12  # months back to start summing
_FORMATION_END: int = 2     # months back to stop summing (skip t-1)
_MIN_MONTHS: int = 9        # minimum valid months out of 11 in window


def compute_residual_momentum(
    returns_df: pd.DataFrame,
    fundamentals_df: pd.DataFrame,
    start_date: str | date,
    end_date: str | date,
    formation_start: int = _FORMATION_START,
    formation_end: int = _FORMATION_END,
    min_months: int = _MIN_MONTHS,
) -> pd.DataFrame:
    """Compute the raw Residual Momentum signal per (permno, month_end).

    Parameters
    ----------
    returns_df : pandas.DataFrame
        Long-format daily returns from ReturnsPanel. Must contain:
        'permno', 'date', 'ret', 'marketcap'.
    fundamentals_df : pandas.DataFrame
        Long-format quarterly fundamentals from Fundamentals.fetch_quarterly.
        Must contain: 'permno', 'rdq', 'datadate', 'ggroup'.
        Industry classification is forward-filled from each rdq through to
        the next rdq for the same permno.
    start_date, end_date : str or date
        Inclusive window on output `date_filed`. Note: returns_df should
        be pre-fetched with at least 14 months of leading buffer so the
        first signal date has its full t-12 window available.
    formation_start : int, default 12
        Months back to start summing (exclusive of current month).
    formation_end : int, default 2
        Months back to stop summing (skip most recent t-1 month for reversal).
    min_months : int, default 9
        Minimum valid monthly residuals required in the window to emit a row.

    Returns
    -------
    pandas.DataFrame
        Long-format raw signal panel sorted by (date_filed, permno) with
        columns matching RESMOM_RAW_COLUMNS.

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

    required_returns = {"permno", "date", "ret", "marketcap"}
    if not required_returns.issubset(returns_df.columns):
        missing = required_returns - set(returns_df.columns)
        raise ValueError(f"returns_df missing columns: {sorted(missing)}")

    required_fund = {"permno", "rdq", "datadate", "ggroup"}
    if not required_fund.issubset(fundamentals_df.columns):
        missing = required_fund - set(fundamentals_df.columns)
        raise ValueError(f"fundamentals_df missing columns: {sorted(missing)}")

    if formation_start <= formation_end:
        raise ValueError(
            f"formation_start ({formation_start}) must be > "
            f"formation_end ({formation_end})"
        )
    window_size = formation_start - formation_end + 1
    if min_months > window_size:
        raise ValueError(
            f"min_months ({min_months}) cannot exceed window size ({window_size})"
        )
    if min_months < 2:
        raise ValueError(f"min_months must be >= 2, got {min_months}")

    # ------------------------------------------------------------------
    # Step 1: Aggregate daily returns to monthly
    # ------------------------------------------------------------------
    rets = returns_df.copy()
    rets["date"] = pd.to_datetime(rets["date"]).astype("datetime64[ns]")
    rets = rets.dropna(subset=["ret"])

    # Compound daily returns to monthly. Group by (permno, year-month).
    rets["year_month"] = rets["date"].dt.to_period("M")
    monthly = (
        rets.groupby(["permno", "year_month"])
        .agg(
            monthly_ret=("ret", lambda s: (1 + s).prod() - 1),
            month_end_marketcap=("marketcap", "last"),  # last day's marketcap
            month_end_date=("date", "last"),
        )
        .reset_index()
    )

    if len(monthly) == 0:
        return pd.DataFrame(columns=list(RESMOM_RAW_COLUMNS))

    # ------------------------------------------------------------------
    # Step 2: Attach industry classification (forward-fill from rdq)
    # ------------------------------------------------------------------
    fund = fundamentals_df.copy()
    fund["rdq"] = pd.to_datetime(fund["rdq"]).astype("datetime64[ns]")
    fund = fund[["permno", "rdq", "ggroup"]].dropna(subset=["ggroup"])
    fund = fund.sort_values(["permno", "rdq"]).reset_index(drop=True)

    # As-of merge: for each monthly row, get the most-recent ggroup with rdq <= month_end_date
    monthly = monthly.sort_values("month_end_date").reset_index(drop=True)
    monthly = pd.merge_asof(
        monthly,
        fund.sort_values("rdq").rename(columns={"rdq": "rdq_used"}),
        left_on="month_end_date",
        right_on="rdq_used",
        by="permno",
        direction="backward",
    )

    # Drop rows without industry classification (no fundamentals filed yet)
    monthly = monthly.dropna(subset=["ggroup"])

    if len(monthly) == 0:
        return pd.DataFrame(columns=list(RESMOM_RAW_COLUMNS))

    # ------------------------------------------------------------------
    # Step 3: Cross-sectional regression per month
    # ------------------------------------------------------------------
    # For each month, run: monthly_ret ~ industry_dummies + log(marketcap)
    # Extract residuals.
    # Cast through numpy float64 explicitly — pandas' Float64Dtype (nullable
    # float) returns object-dtype arrays from .to_numpy() if any NA is present,
    # which breaks downstream statsmodels OLS calls.
    log_size_raw = np.log(monthly["month_end_marketcap"].clip(lower=1.0))
    monthly["log_size"] = log_size_raw.astype("float64")
    monthly = monthly.dropna(subset=["monthly_ret", "log_size"])

    monthly["residual"] = np.nan
    for _ym, group in monthly.groupby("year_month"):
        if len(group) < 5:  # need enough names for stable regression
            continue

        residuals = _cross_sectional_residual(group)
        if residuals is not None:
            monthly.loc[group.index, "residual"] = residuals

    monthly = monthly.dropna(subset=["residual"])

    if len(monthly) == 0:
        return pd.DataFrame(columns=list(RESMOM_RAW_COLUMNS))

    # ------------------------------------------------------------------
    # Step 4: For each (permno, month_t), sum residuals from t-12 to t-2
    # ------------------------------------------------------------------
    output_rows = []
    start_ts = pd.Timestamp(start_str)
    end_ts = pd.Timestamp(end_str)

    for permno, group in monthly.groupby("permno", sort=False):
        group = group.sort_values("year_month").reset_index(drop=True)

        for i in range(formation_start, len(group)):
            terminal_row = group.iloc[i]
            terminal_date = pd.Timestamp(terminal_row["month_end_date"])

            # Skip terminal dates outside requested output window
            if terminal_date < start_ts or terminal_date > end_ts:
                continue

            # Window: months at indices [i - formation_start, i - formation_end]
            window = group.iloc[i - formation_start : i - formation_end + 1]
            valid_residuals = window["residual"].dropna()

            if len(valid_residuals) < min_months:
                continue

            output_rows.append(
                {
                    "permno": permno,
                    "date_filed": terminal_date,
                    "n_obs": len(valid_residuals),
                    "size": float(terminal_row["log_size"]),
                    "ggroup": terminal_row["ggroup"],
                    "raw_signal": float(valid_residuals.sum()),
                }
            )

    if not output_rows:
        return pd.DataFrame(columns=list(RESMOM_RAW_COLUMNS))

    result_df = pd.DataFrame(output_rows)
    result_df = result_df[list(RESMOM_RAW_COLUMNS)]
    return result_df.sort_values(["date_filed", "permno"]).reset_index(drop=True)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _cross_sectional_residual(group: pd.DataFrame) -> npt.NDArray[np.float64] | None:
    """Run cross-sectional regression and return residuals.

    Regression: monthly_ret ~ industry_dummies + log(marketcap) + intercept.
    Returns None if regression fails or has too few groups.
    """
    if len(group) < 5:
        return None

    # Need at least 2 unique industries for industry dummies to make sense
    n_groups = group["ggroup"].nunique()
    if n_groups < 2:
        # No industry variation — regress only on size
        X = group[["log_size"]].to_numpy()  # noqa: N806 (statistical convention)
    else:
        # Build industry dummies (drop_first to avoid multicollinearity)
        dummies = pd.get_dummies(group["ggroup"], drop_first=True, dtype=float)
        X = np.column_stack([  # noqa: N806 (statistical convention)
            dummies.to_numpy(),
            group["log_size"].to_numpy().reshape(-1, 1),
        ])

    X = sm.add_constant(X)  # noqa: N806 (statistical convention)
    y = group["monthly_ret"].to_numpy()

    try:
        results = sm.OLS(y, X).fit()
    except (np.linalg.LinAlgError, ValueError):
        return None

    return np.asarray(results.resid)


def _normalize_date(d: str | date) -> str:
    """Normalize date input to ISO-format string."""
    if isinstance(d, date):
        return d.isoformat()
    if isinstance(d, str):
        parsed = date.fromisoformat(d)
        return parsed.isoformat()
    raise ValueError(f"date must be str or date, got {type(d).__name__}")
