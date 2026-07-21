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

from typing import Literal

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
    weights_df: pd.DataFrame | None = None,
    long_quintile: Literal["top", "bottom"] = "top",
) -> pd.DataFrame:
    """Compute L/S portfolio return per rebalance date.

    Long leg = quintile designated by long_quintile ("top" = highest z_score
    quintile, "bottom" = lowest). Short leg = the opposite quintile.
    ls_return = long_return - short_return.

    Weighting scheme:
      - weights_df=None (default): equal-weighted within each leg.
      - weights_df provided: value-weighted using the 'weight' column
        (typically market cap at the rebalance date). Firms with NaN
        weight are excluded from that leg's weighted average. Weights
        within each (date, quintile) are normalized to sum to 1.

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
        Must match the number used in assign_quintiles.
    weights_df : pd.DataFrame | None, default None
        Optional weights panel. When provided, must have columns
        'date', 'permno', 'weight'. Weights must be non-negative.
    long_quintile : Literal["top", "bottom"], default "top"
        Which quintile to LONG. "top" longs quintile n_quintiles (highest
        z_score) and shorts quintile 1. "bottom" longs quintile 1 and
        shorts quintile n_quintiles. Explicit for signals where the
        theoretically-longable end is the low z_score end (e.g. Lazy
        Prices: low text change firms outperform per CMN 2020).

    Returns
    -------
    pd.DataFrame
        One row per rebalance date with LS_RETURN_COLUMNS.

    Raises
    ------
    ValueError
        If required columns missing, n_quintiles < 2, or weights_df
        contains negative weights.
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
    if long_quintile not in ("top", "bottom"):
        raise ValueError(
            f"long_quintile must be 'top' or 'bottom', got {long_quintile!r}"
        )

    if weights_df is not None:
        w_required = {"date", "permno", "weight"}
        w_missing = w_required - set(weights_df.columns)
        if w_missing:
            raise ValueError(
                f"weights_df missing columns: {sorted(w_missing)}"
            )
        neg = weights_df["weight"].dropna() < 0
        if neg.any():
            raise ValueError(
                f"weights_df has {int(neg.sum())} negative-weight rows"
            )

    if quintiles_df.empty or forward_returns_df.empty:
        return pd.DataFrame(columns=list(LS_RETURN_COLUMNS))

    # Merge quintile assignments with forward returns
    merged = quintiles_df.merge(
        forward_returns_df.rename(columns={"rebalance_date": "date"}),
        on=["date", "permno"],
        how="left",
    )

    # Drop rows without a forward return
    merged = merged[merged["fwd_return"].notna()]

    # Merge weights if provided
    if weights_df is not None:
        merged = merged.merge(
            weights_df[["date", "permno", "weight"]],
            on=["date", "permno"],
            how="left",
        )

    # Swap based on convention: for "top", long the top quintile; for
    # "bottom", long the bottom quintile (short becomes the opposite).
    if long_quintile == "top":
        long_q = float(n_quintiles)
        short_q = 1.0
    else:  # "bottom"
        long_q = 1.0
        short_q = float(n_quintiles)
    use_weights = weights_df is not None

    def _leg_return(leg: pd.DataFrame) -> tuple[float, int]:
        """Return (leg_return, n_firms) for one quintile within a date."""
        if len(leg) == 0:
            return np.nan, 0
        if not use_weights:
            return float(leg["fwd_return"].mean()), len(leg)
        # Value-weighted path: drop firms with NaN weight
        weighted = leg.dropna(subset=["weight"])
        if len(weighted) == 0:
            return np.nan, 0
        total_w = float(weighted["weight"].sum())
        if total_w == 0.0:
            return np.nan, len(weighted)
        w = weighted["weight"] / total_w
        return float((w * weighted["fwd_return"]).sum()), len(weighted)

    def _per_date(group: pd.DataFrame) -> pd.Series:
        longs = group[group["quintile"] == long_q]
        shorts = group[group["quintile"] == short_q]
        long_return, n_long = _leg_return(longs)
        short_return, n_short = _leg_return(shorts)
        ls_return = long_return - short_return
        return pd.Series({
            "long_return": long_return,
            "short_return": short_return,
            "ls_return": ls_return,
            "n_long": n_long,
            "n_short": n_short,
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