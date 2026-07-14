"""Portfolio-sort backtest primitives.

Generic quintile-sort machinery for signal-based long-short portfolios.
Consumes the output of align_signal (aligned z-score panel) and
forward_returns (holding-period returns per firm), produces monthly
long-short portfolio return series suitable for compute_performance_metrics.

Design
------
Two pure functions:
  1. assign_quintiles: per rebalance date, sort firms by z_score into
     n_quintiles buckets. Uses pd.qcut with duplicates='drop', matching
     Cohen-Malloy-Nguyen (2020) methodology.
  2. compute_long_short_returns: for each rebalance date, average forward
     returns within top quintile (long) and bottom quintile (short),
     return the difference.

No I/O. No universe fetching. Callers provide inputs; functions transform.

See docs/v2_item6_design.md "Backtest scope (2026-07-12)" for the
pre-commitments used by the Lazy Prices runner.
"""
# ruff: noqa: I001

from __future__ import annotations

import numpy as np
import pandas as pd


LS_RETURN_COLUMNS: tuple[str, ...] = (
    "date",
    "long_return",
    "short_return",
    "ls_return",
    "n_long",
    "n_short",
)


def assign_quintiles(
    aligned_signal_df: pd.DataFrame,
    n_quintiles: int = 5,
) -> pd.DataFrame:
    """Assign each (date, permno) row to a quintile based on z_score.

    Uses pd.qcut with duplicates='drop', so quintile sizes may be
    unequal if the z_score cross-section has ties at quintile boundaries.
    This matches the standard finance practice used by CMN 2020.

    Parameters
    ----------
    aligned_signal_df : pd.DataFrame
        Output of align_signal. Required columns: 'date', 'permno',
        'z_score'. Extra columns are preserved but ignored.
    n_quintiles : int, default 5
        Number of buckets. Rank ascending: quintile=1 is the lowest
        z_score (short leg), quintile=n_quintiles is the highest (long leg).

    Returns
    -------
    pd.DataFrame
        Long-format panel with columns ('date', 'permno', 'quintile').
        Rows with NaN z_score get quintile=NaN and are effectively
        excluded from downstream portfolio construction.
        Sorted by (date, permno).

    Raises
    ------
    ValueError
        If required columns are missing, n_quintiles < 2, or if
        aligned_signal_df is empty (returned as empty with correct columns).
    """
    required = {"date", "permno", "z_score"}
    missing = required - set(aligned_signal_df.columns)
    if missing:
        raise ValueError(
            f"aligned_signal_df missing columns: {sorted(missing)}"
        )
    if n_quintiles < 2:
        raise ValueError(
            f"n_quintiles must be >= 2, got {n_quintiles}"
        )

    if aligned_signal_df.empty:
        return pd.DataFrame(columns=["date", "permno", "quintile"])

    # Reset index to guarantee uniqueness — callers may pass in
    # concatenated frames with duplicate indices.
    df = aligned_signal_df[["date", "permno", "z_score"]].copy().reset_index(drop=True)
    df["quintile"] = np.nan

    for date_val, group in df.groupby("date"):
        valid_mask = group["z_score"].notna()
        if valid_mask.sum() < n_quintiles:
            continue
        try:
            labels = pd.qcut(
                group.loc[valid_mask, "z_score"],
                q=n_quintiles,
                labels=False,
                duplicates="drop",
            )
            # labels 0..n-1; shift to 1..n
            df.loc[labels.index, "quintile"] = labels.values + 1
        except ValueError:
            # All z_scores identical after dedup
            continue

    return (
        df[["date", "permno", "quintile"]]
        .sort_values(["date", "permno"])
        .reset_index(drop=True)
    )


def compute_long_short_returns(
    quintiles_df: pd.DataFrame,
    forward_returns_df: pd.DataFrame,
    n_quintiles: int = 5,
) -> pd.DataFrame:
    """Compute equal-weighted L/S portfolio return per rebalance date.

    Long leg = top quintile (highest z_score). Short leg = bottom quintile.
    Both legs equal-weighted. ls_return = long_return - short_return.

    Firms in a quintile but with no forward return in forward_returns_df
    are dropped from that quintile's average.

    Parameters
    ----------
    quintiles_df : pd.DataFrame
        Output of assign_quintiles. Columns: 'date', 'permno', 'quintile'.
    forward_returns_df : pd.DataFrame
        Output of compute_forward_returns. Columns: 'rebalance_date',
        'permno', 'fwd_return'.
    n_quintiles : int, default 5
        Must match the number used in assign_quintiles. Determines
        which quintile is "top" for the long leg.

    Returns
    -------
    pd.DataFrame
        One row per rebalance date with columns matching LS_RETURN_COLUMNS:
        (date, long_return, short_return, ls_return, n_long, n_short).
        Sorted by date. Dates with an empty long or short quintile get
        NaN for that leg's return and 0 for its count.

    Raises
    ------
    ValueError
        If required columns are missing or n_quintiles < 2.
    """
    q_required = {"date", "permno", "quintile"}
    fr_required = {"rebalance_date", "permno", "fwd_return"}
    q_missing = q_required - set(quintiles_df.columns)
    if q_missing:
        raise ValueError(f"quintiles_df missing columns: {sorted(q_missing)}")
    fr_missing = fr_required - set(forward_returns_df.columns)
    if fr_missing:
        raise ValueError(
            f"forward_returns_df missing columns: {sorted(fr_missing)}"
        )
    if n_quintiles < 2:
        raise ValueError(f"n_quintiles must be >= 2, got {n_quintiles}")

    if quintiles_df.empty or forward_returns_df.empty:
        return pd.DataFrame(columns=list(LS_RETURN_COLUMNS))

    # Merge quintile assignments with forward returns
    merged = quintiles_df.merge(
        forward_returns_df.rename(columns={"rebalance_date": "date"}),
        on=["date", "permno"],
        how="left",
    )

    # Drop rows without a forward return (firm was in universe but had no
    # return data — e.g., delisted, missing CRSP row).
    merged = merged[merged["fwd_return"].notna()]

    top_q = float(n_quintiles)
    bot_q = 1.0

    def _per_date(group: pd.DataFrame) -> pd.Series:
        longs = group[group["quintile"] == top_q]["fwd_return"]
        shorts = group[group["quintile"] == bot_q]["fwd_return"]
        long_return = longs.mean() if len(longs) > 0 else np.nan
        short_return = shorts.mean() if len(shorts) > 0 else np.nan
        ls_return = long_return - short_return
        return pd.Series({
            "long_return": long_return,
            "short_return": short_return,
            "ls_return": ls_return,
            "n_long": len(longs),
            "n_short": len(shorts),
        })

    result = (
        merged.groupby("date", as_index=False)
        .apply(_per_date, include_groups=False)
        .reset_index(drop=True)
    )

    # Ensure integer counts
    result["n_long"] = result["n_long"].astype(int)
    result["n_short"] = result["n_short"].astype(int)

    return result[list(LS_RETURN_COLUMNS)].sort_values("date").reset_index(drop=True)