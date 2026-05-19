"""Forward return computation for IC analysis and signal evaluation.

Given a daily returns panel and a list of rebalance dates, compute the
holding-period total return per (rebalance_date, permno). Used by IC
analysis to correlate signal values at rebalance time with subsequent
realized returns.

Design
------
Pure functions, no I/O. Caller provides the returns DataFrame
(typically from cache.returns_full) and parameters; the module returns
a long-format DataFrame keyed by (rebalance_date, permno).

For holding-period returns, we compound log returns over the holding
window and convert back, which handles compounding correctly and is
numerically stable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FORWARD_RETURN_COLUMNS: tuple[str, ...] = (
    "rebalance_date",
    "permno",
    "fwd_return",
    "n_days",
)


def compute_forward_returns(
    returns_df: pd.DataFrame,
    rebalance_dates: list[pd.Timestamp] | pd.DatetimeIndex,
    holding_days: int = 21,
) -> pd.DataFrame:
    """Compute holding-period forward returns per (rebalance_date, permno).

    Parameters
    ----------
    returns_df : pd.DataFrame
        Long-format daily returns. Must contain 'permno', 'date', 'ret'.
    rebalance_dates : list of Timestamp or DatetimeIndex
        Dates at which forward returns are computed. For each date,
        the forward window is (date, date + holding_days trading days].
    holding_days : int, default 21
        Number of trading days in the holding window. 21 ≈ 1 month.

    Returns
    -------
    pd.DataFrame
        Long-format panel with columns matching FORWARD_RETURN_COLUMNS:
        rebalance_date, permno, fwd_return (compounded), n_days (actual
        number of trading days included, useful for diagnostics).
        Sorted by (rebalance_date, permno).

    Notes
    -----
    Forward returns use trading days, not calendar days. If holding_days
    is 21 but a name has only 18 trading days of data before the next
    rebalance (e.g., end of period, delisting), the return is computed
    over the available 18 days and n_days reflects this.

    Names with no return data in the forward window are excluded from
    output (rather than emitting NaN). This is the convention used by
    other modules in the project.
    """
    required = {"permno", "date", "ret"}
    if not required.issubset(returns_df.columns):
        missing = required - set(returns_df.columns)
        raise ValueError(f"returns_df missing columns: {sorted(missing)}")

    if holding_days < 1:
        raise ValueError(f"holding_days must be >= 1, got {holding_days}")

    if len(rebalance_dates) == 0:
        return pd.DataFrame(columns=list(FORWARD_RETURN_COLUMNS))

    rets = returns_df[["permno", "date", "ret"]].copy()
    rets["date"] = pd.to_datetime(rets["date"]).astype("datetime64[ns]")
    rets["ret"] = pd.to_numeric(rets["ret"], errors="coerce")
    rets = rets.dropna(subset=["ret"]).sort_values(["permno", "date"])
    rets["log_ret"] = np.log1p(rets["ret"].astype("float64"))

    rebal_ts = pd.DatetimeIndex(pd.to_datetime(pd.Series(list(rebalance_dates))))

    rows: list[pd.DataFrame] = []
    for rebal in rebal_ts:
        sub = rets[rets["date"] > rebal].copy()
        if sub.empty:
            continue
        # Take the first `holding_days` rows per permno (trading days after rebal)
        sub = sub.groupby("permno", as_index=False).head(holding_days)
        agg = sub.groupby("permno").agg(
            log_ret_sum=("log_ret", "sum"),
            n_days=("log_ret", "size"),
        )
        agg = agg.reset_index()
        agg["fwd_return"] = np.expm1(agg["log_ret_sum"])
        agg["rebalance_date"] = rebal
        rows.append(agg[["rebalance_date", "permno", "fwd_return", "n_days"]])

    if not rows:
        return pd.DataFrame(columns=list(FORWARD_RETURN_COLUMNS))

    result = pd.concat(rows, ignore_index=True)
    result["permno"] = result["permno"].astype("int64")
    result["n_days"] = result["n_days"].astype("int64")
    return result.sort_values(
        ["rebalance_date", "permno"]
    ).reset_index(drop=True)
