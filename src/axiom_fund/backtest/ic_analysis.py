"""Information Coefficient (IC) and signal correlation analysis.

Pure-function module for diagnosing signal quality. Given per-period
signal z-score panels and forward returns, computes:

  1. Cross-sectional information coefficient (rank correlation of signal
     with subsequent return) per signal per period
  2. Cross-sectional correlation between signals per period
  3. Aggregate statistics (mean, std, t-stat) over arbitrary groupings
     (full sample, by year, by regime)

The framework is designed to be reusable as the signal set grows.
Adding a new signal (Form 4, news sentiment, etc.) requires no changes
to this module — just pass its z-score column in the signal_panel.

References
----------
Grinold, R. C., & Kahn, R. N. (1999). Active Portfolio Management.
    (Foundational treatment of IC; defines IC as rank correlation
    between forecast and realization, with t-stat = mean_IC × sqrt(N).)
Ding, Y. (2010). "Bias of the Information Coefficient as a Performance
    Measure." Journal of Investment Management.
"""

from __future__ import annotations

from typing import Final, Literal

import numpy as np
import pandas as pd
from scipy import stats

# Output column conventions
IC_LONG_COLUMNS: Final[tuple[str, ...]] = (
    "rebalance_date",
    "signal",
    "ic_pearson",
    "ic_spearman",
    "n_names",
)

CORR_LONG_COLUMNS: Final[tuple[str, ...]] = (
    "rebalance_date",
    "signal_a",
    "signal_b",
    "correlation",
    "n_names",
)

IC_SUMMARY_COLUMNS: Final[tuple[str, ...]] = (
    "signal",
    "n_periods",
    "mean_ic_spearman",
    "std_ic_spearman",
    "t_stat",
    "p_value",
    "ic_ir",
    "hit_rate",
)


def compute_period_ic(
    signal_panel: pd.DataFrame,
    forward_returns: pd.DataFrame,
    signal_columns: list[str],
    rebalance_date: pd.Timestamp,
) -> pd.DataFrame:
    """Compute IC per signal for one rebalance date.

    Parameters
    ----------
    signal_panel : pd.DataFrame
        Per-name signal z-scores at the rebalance date. Must contain
        'permno' and each of `signal_columns`.
    forward_returns : pd.DataFrame
        Output of `compute_forward_returns()` filtered to this rebalance
        date, or a single-date subset. Must contain 'permno', 'fwd_return'.
    signal_columns : list of str
        Column names in signal_panel to compute IC for (e.g.,
        ['z_gp', 'z_ivol', 'z_resmom', 'z_pead']).
    rebalance_date : pd.Timestamp
        The date this analysis is for. Included in output for joining.

    Returns
    -------
    pd.DataFrame
        Long-format with columns matching IC_LONG_COLUMNS. One row per
        signal in `signal_columns`.

    Notes
    -----
    Both Pearson and Spearman IC are computed. Spearman (rank correlation)
    is the standard for signal evaluation because z-scores are often
    non-normal and rank-based measures are robust to outliers.
    """
    if "permno" not in signal_panel.columns:
        raise ValueError("signal_panel missing 'permno' column")
    missing_sigs = [c for c in signal_columns if c not in signal_panel.columns]
    if missing_sigs:
        raise ValueError(f"signal_panel missing signal columns: {missing_sigs}")
    if not {"permno", "fwd_return"}.issubset(forward_returns.columns):
        raise ValueError("forward_returns missing 'permno' or 'fwd_return'")

    merged = signal_panel[["permno", *signal_columns]].merge(
        forward_returns[["permno", "fwd_return"]], on="permno", how="inner"
    )
    merged = merged.dropna(subset=["fwd_return"])

    rows: list[dict[str, object]] = []
    for sig in signal_columns:
        sub = merged.dropna(subset=[sig])
        n = len(sub)
        if n < 10:
            rows.append({
                "rebalance_date": rebalance_date,
                "signal": sig,
                "ic_pearson": np.nan,
                "ic_spearman": np.nan,
                "n_names": n,
            })
            continue
        # Pearson IC
        ic_p = float(sub[sig].corr(sub["fwd_return"], method="pearson"))
        # Spearman IC (rank-based, the standard)
        ic_s = float(sub[sig].corr(sub["fwd_return"], method="spearman"))
        rows.append({
            "rebalance_date": rebalance_date,
            "signal": sig,
            "ic_pearson": ic_p,
            "ic_spearman": ic_s,
            "n_names": n,
        })

    return pd.DataFrame(rows, columns=list(IC_LONG_COLUMNS))


def compute_period_correlations(
    signal_panel: pd.DataFrame,
    signal_columns: list[str],
    rebalance_date: pd.Timestamp,
) -> pd.DataFrame:
    """Compute pairwise correlation between signals for one rebalance date.

    Returns long-format panel with one row per unique unordered pair
    (signal_a < signal_b lexicographically). Diagonal is omitted.

    Parameters
    ----------
    signal_panel : pd.DataFrame
        Per-name signal z-scores at the rebalance date. Must contain
        each column in `signal_columns`.
    signal_columns : list of str
        Signal columns to compute pairwise correlations for.
    rebalance_date : pd.Timestamp
        The date being analyzed.

    Returns
    -------
    pd.DataFrame
        Long-format with columns matching CORR_LONG_COLUMNS.
    """
    missing = [c for c in signal_columns if c not in signal_panel.columns]
    if missing:
        raise ValueError(f"signal_panel missing signal columns: {missing}")

    pairs: list[dict[str, object]] = []
    for i, a in enumerate(signal_columns):
        for b in signal_columns[i + 1:]:
            sub = signal_panel[[a, b]].dropna()
            n = len(sub)
            corr = np.nan if n < 10 else float(sub[a].corr(sub[b], method="pearson"))
            pairs.append({
                "rebalance_date": rebalance_date,
                "signal_a": a,
                "signal_b": b,
                "correlation": corr,
                "n_names": n,
            })

    return pd.DataFrame(pairs, columns=list(CORR_LONG_COLUMNS))


def aggregate_ic(
    ic_long: pd.DataFrame,
    by: Literal["all", "year", "regime"] | None = None,
    regime_col: str | None = None,
) -> pd.DataFrame:
    """Aggregate per-period IC into summary statistics.

    Parameters
    ----------
    ic_long : pd.DataFrame
        Long-format IC output from compute_period_ic (or concatenation
        thereof across periods). Must contain 'rebalance_date', 'signal',
        'ic_spearman'.
    by : 'all', 'year', 'regime', or None
        Grouping for aggregation. 'all' produces one row per signal
        (full-window stats). 'year' produces one row per (signal, year).
        'regime' uses the regime_col argument. None defaults to 'all'.
    regime_col : str, optional
        Column in ic_long indicating regime label. Required when by='regime'.

    Returns
    -------
    pd.DataFrame
        Summary table with columns matching IC_SUMMARY_COLUMNS.
        For by='year' / 'regime', adds the grouping column.

    Statistics
    ----------
    mean_ic_spearman : average IC across the grouping
    std_ic_spearman  : std of IC across the grouping
    t_stat           : mean / (std / sqrt(n)); tests H0: mean_IC = 0
    p_value          : two-sided p-value from t-distribution
    ic_ir            : Information Ratio of the IC series itself
                       (mean / std), unannualized
    hit_rate         : fraction of periods with IC > 0
    """
    if by is None:
        by = "all"
    required = {"rebalance_date", "signal", "ic_spearman"}
    if not required.issubset(ic_long.columns):
        missing = required - set(ic_long.columns)
        raise ValueError(f"ic_long missing columns: {sorted(missing)}")

    df = ic_long.dropna(subset=["ic_spearman"]).copy()

    group_cols: list[str] = ["signal"]
    if by == "year":
        df["year"] = pd.to_datetime(df["rebalance_date"]).dt.year
        group_cols = ["year", "signal"]
    elif by == "regime":
        if regime_col is None or regime_col not in df.columns:
            raise ValueError(
                "regime_col must be provided and present in ic_long when by='regime'"
            )
        group_cols = [regime_col, "signal"]

    # Aggregate using groupby().agg() (cleaner pandas types than .apply()
    # with a custom function that returns a Series). We build a long list
    # of named aggregations, then post-process for t-stat / p-value / IR.
    agg_basic = df.groupby(group_cols, as_index=False).agg(
        n_periods=("ic_spearman", "size"),
        mean_ic_spearman=("ic_spearman", "mean"),
        std_ic_spearman=("ic_spearman", "std"),
        hit_rate=("ic_spearman", lambda s: float((s > 0).mean())),
    )

    # Post-process for t-stat, p-value, ic_ir (need both mean and std)
    n = agg_basic["n_periods"].astype("float64")
    mean = agg_basic["mean_ic_spearman"].astype("float64")
    std = agg_basic["std_ic_spearman"].astype("float64")
    # t = mean / (std / sqrt(n)); guarded against zero/NaN std
    se = std / np.sqrt(n)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stat = np.where((n >= 2) & (std > 0) & std.notna(), mean / se, np.nan)
        ic_ir = np.where(std.notna() & (std > 0), mean / std, np.nan)
    # Two-sided p-value from t-distribution with n-1 dof
    p_value = np.where(
        np.isnan(t_stat),
        np.nan,
        2 * (1 - stats.t.cdf(np.abs(t_stat), df=np.maximum(n - 1, 1))),
    )
    agg_basic["t_stat"] = t_stat
    agg_basic["p_value"] = p_value
    agg_basic["ic_ir"] = ic_ir

    cols_order = [*group_cols, "n_periods", "mean_ic_spearman",
                  "std_ic_spearman", "t_stat", "p_value", "ic_ir", "hit_rate"]
    return agg_basic[cols_order].sort_values(group_cols).reset_index(drop=True)


def summarize_correlations(
    corr_long: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate per-period pairwise correlations.

    Parameters
    ----------
    corr_long : pd.DataFrame
        Output of compute_period_correlations concatenated across periods.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (mean_correlations, std_correlations) — each is a square
        symmetric DataFrame of signal × signal with mean/std of the
        pairwise correlation across periods on the off-diagonal.
        Diagonal is 1.0 (for mean) or 0.0 (for std).
    """
    required = {"signal_a", "signal_b", "correlation"}
    if not required.issubset(corr_long.columns):
        missing = required - set(corr_long.columns)
        raise ValueError(f"corr_long missing columns: {sorted(missing)}")

    df = corr_long.dropna(subset=["correlation"]).copy()
    # Get all unique signals
    signals = sorted(set(df["signal_a"]) | set(df["signal_b"]))

    mean_corr = pd.DataFrame(
        np.eye(len(signals)), index=signals, columns=signals
    )
    std_corr = pd.DataFrame(
        np.zeros((len(signals), len(signals))), index=signals, columns=signals
    )

    grouped = df.groupby(["signal_a", "signal_b"])
    for (a, b), g in grouped:
        m = float(g["correlation"].mean())
        s = float(g["correlation"].std()) if len(g) > 1 else 0.0
        mean_corr.loc[a, b] = m
        mean_corr.loc[b, a] = m
        std_corr.loc[a, b] = s
        std_corr.loc[b, a] = s

    return mean_corr, std_corr
