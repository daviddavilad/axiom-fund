"""Alignment layer: forward-fills raw signals to a rebalance calendar,
winsorizes, and z-scores within the current Universe.

This module is shared across all three signals. Each signal module emits
raw values keyed by its natural cadence (e.g., GP keyed by rdq, IVol
keyed by trading day). The alignment layer takes that raw output plus a
universe panel and a list of rebalance dates, and produces the final
cross-sectionally z-scored signal panel.

See docs/signal_design.md §2.3 (revised) and §3 for the design rationale.

Pipeline (per rebalance date):
  1. Determine the universe at this date (from universe_df)
  2. For each PERMNO in the universe, find the most-recent raw_signal
     value with date_filed <= rebalance_date (forward-fill)
  3. Winsorize the cross-section at 1st/99th percentile
  4. Z-score the cross-section: (x - mean) / std

Output schema:
  date          — rebalance date
  permno        — security identifier
  raw_signal    — forward-filled raw value (NaN if no prior signal)
  winsorized    — cross-section clipped at percentile bounds
  z_score       — cross-section z-score
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

ALIGNED_OUTPUT_COLUMNS: tuple[str, ...] = (
    "date",
    "permno",
    "raw_signal",
    "winsorized",
    "z_score",
)


def align_signal(
    raw_signal_df: pd.DataFrame,
    universe_df: pd.DataFrame,
    rebalance_dates: list[date] | list[str] | pd.DatetimeIndex,
    winsorize_pct: float = 0.01,
    max_age_days: int | None = None,
) -> pd.DataFrame:
    """Align a raw signal to rebalance dates, winsorize, and z-score.

    Parameters
    ----------
    raw_signal_df : pandas.DataFrame
        Long-format raw signal panel. Must contain at minimum:
            permno, date_filed, raw_signal
        Multiple rows per (permno, date_filed) are not supported.
    universe_df : pandas.DataFrame
        Long-format universe panel. Must contain at minimum:
            date, permno
        A PERMNO is in the universe on date d if (d, permno) is in
        the panel. Universe must include rows for at least the
        rebalance dates.
    rebalance_dates : list of date-likes
        The dates on which to compute the cross-sectional z-score.
        Each date must have a corresponding universe snapshot.
    winsorize_pct : float, default 0.01
        Cross-sectional winsorization fraction. 0.01 means clip at the
        1st and 99th percentile. Set to 0.0 to disable.
    max_age_days : int or None, default None
        Maximum age (in calendar days) for a raw signal to be eligible
        for forward-fill into a rebalance date. None (default) means no
        age limit — the most-recent prior signal is always used, no
        matter how old. Used for event-driven signals like PEAD where
        information decays beyond a known horizon (~60-90 days).

    Returns
    -------
    pandas.DataFrame
        Long-format aligned panel with columns matching
        ALIGNED_OUTPUT_COLUMNS, sorted by (date, permno).

    Raises
    ------
    ValueError
        On missing required columns, malformed dates, or invalid
        winsorize_pct.
    """
    # ------------------------------------------------------------------
    # 1. Input validation
    # ------------------------------------------------------------------
    required_signal = {"permno", "date_filed", "raw_signal"}
    if not required_signal.issubset(raw_signal_df.columns):
        missing = required_signal - set(raw_signal_df.columns)
        raise ValueError(f"raw_signal_df missing columns: {sorted(missing)}")

    required_universe = {"date", "permno"}
    if not required_universe.issubset(universe_df.columns):
        missing = required_universe - set(universe_df.columns)
        raise ValueError(f"universe_df missing columns: {sorted(missing)}")

    if not 0.0 <= winsorize_pct < 0.5:
        raise ValueError(
            f"winsorize_pct must be in [0, 0.5), got {winsorize_pct}"
        )

    if len(rebalance_dates) == 0:
        return pd.DataFrame(columns=list(ALIGNED_OUTPUT_COLUMNS))

    # ------------------------------------------------------------------
    # 2. Normalize dtypes for joining (lesson from the GP integration test)
    # ------------------------------------------------------------------
    # Convert to DatetimeIndex first, which accepts any of the supported
    # input types and produces a uniform output. Then normalize precision.
    rebal_ts = pd.DatetimeIndex(pd.to_datetime(pd.Series(list(rebalance_dates)))).astype("datetime64[ns]")
    rebal_ts = rebal_ts.sort_values()

    raw = raw_signal_df.copy()
    raw["date_filed"] = pd.to_datetime(raw["date_filed"]).astype("datetime64[ns]")

    uni = universe_df.copy()
    uni["date"] = pd.to_datetime(uni["date"]).astype("datetime64[ns]")

    # ------------------------------------------------------------------
    # 3. For each rebalance date, find the universe snapshot that applies
    #    (most-recent universe date <= rebalance date)
    # ------------------------------------------------------------------
    universe_dates_sorted = uni["date"].drop_duplicates().sort_values()
    rebal_to_uni = pd.merge_asof(
        pd.DataFrame({"rebal_date": rebal_ts}).sort_values("rebal_date"),
        universe_dates_sorted.to_frame("uni_date"),
        left_on="rebal_date",
        right_on="uni_date",
        direction="backward",
    )

    if rebal_to_uni["uni_date"].isna().any():
        bad = rebal_to_uni[rebal_to_uni["uni_date"].isna()]["rebal_date"].tolist()
        raise ValueError(
            f"No universe snapshot available for rebalance dates: {bad}. "
            f"Earliest universe date: {universe_dates_sorted.min()}"
        )

    # ------------------------------------------------------------------
    # 4. For each rebalance date, build the cross-section
    # ------------------------------------------------------------------
    output_rows = []

    for _, row in rebal_to_uni.iterrows():
        rebal_date = row["rebal_date"]
        uni_date = row["uni_date"]

        # Universe at this rebalance date
        universe_permnos = uni.loc[uni["date"] == uni_date, "permno"].unique()

        # For each PERMNO in the universe, find the most-recent raw_signal
        # with date_filed <= rebal_date
        candidate_signals = raw[
            (raw["permno"].isin(universe_permnos))
            & (raw["date_filed"] <= rebal_date)
        ]
        if max_age_days is not None:
            cutoff = rebal_date - pd.Timedelta(days=max_age_days)
            candidate_signals = candidate_signals[
                candidate_signals["date_filed"] >= cutoff
            ]

        if len(candidate_signals) == 0:
            # No signal data yet — emit NaN for every PERMNO
            cross_section = pd.DataFrame(
                {
                    "permno": universe_permnos,
                    "raw_signal": np.nan,
                }
            )
        else:
            # Take the most-recent signal per PERMNO
            most_recent = (
                candidate_signals.sort_values("date_filed")
                .groupby("permno", as_index=False)
                .tail(1)[["permno", "raw_signal"]]
            )
            # Ensure every universe PERMNO has a row (NaN if no signal yet)
            cross_section = pd.DataFrame({"permno": universe_permnos}).merge(
                most_recent, on="permno", how="left"
            )

        # Winsorize within the cross-section
        if winsorize_pct > 0.0 and cross_section["raw_signal"].notna().any():
            lo = cross_section["raw_signal"].quantile(winsorize_pct)
            hi = cross_section["raw_signal"].quantile(1.0 - winsorize_pct)
            cross_section["winsorized"] = cross_section["raw_signal"].clip(
                lower=lo, upper=hi
            )
        else:
            cross_section["winsorized"] = cross_section["raw_signal"]

        # Z-score within the cross-section (using winsorized values)
        wmean = cross_section["winsorized"].mean()
        wstd = cross_section["winsorized"].std()
        if pd.isna(wstd) or wstd < 1e-12:
            cross_section["z_score"] = np.nan
        else:
            cross_section["z_score"] = (
                cross_section["winsorized"] - wmean
            ) / wstd

        cross_section["date"] = rebal_date
        output_rows.append(cross_section[list(ALIGNED_OUTPUT_COLUMNS)])

    if not output_rows:
        return pd.DataFrame(columns=list(ALIGNED_OUTPUT_COLUMNS))

    result = pd.concat(output_rows, ignore_index=True)
    return result.sort_values(["date", "permno"]).reset_index(drop=True)
