"""Composite alpha construction: combine multiple signal z-scores into one.

For each (date, permno), the composite alpha is computed as:

    1. Apply each signal's sign convention (flip IVol; GP and ResMom unchanged)
    2. Average the available signed z-scores (require at least min_signals)
    3. Re-z-score the composite cross-sectionally so output has unit std

This module consumes the output of `signals.alignment.align_signal()` for
each of the three locked alpha signals (Gross Profitability, Idiosyncratic
Volatility, Residual Momentum) and emits a single composite signal panel
that the optimizer will use as expected returns.

Sign conventions (per docs/strategy_spec.md §5):
  - GP: high → good (Novy-Marx 2013)
  - IVol: high → bad (AHXZ 2006); the composite uses -z_ivol so that
    "high composite alpha" consistently means "want to go long"
  - ResMom: high → good (Blitz-Huij-Martens 2011)

Missing signal handling (per docs/signal_design.md §2.2):
  - A name may have z-scores for some but not all three signals (e.g.,
    insufficient earnings history for GP, insufficient return history
    for IVol or ResMom).
  - We compute the composite from whichever signals are available, as
    long as at least min_signals (default 2) are non-NaN. Names with
    fewer than min_signals are dropped from output for that date.

Re-z-scoring (per Decision 3):
  - The simple mean of independent z-scores has std < 1 (~1/√k for k
    orthogonal signals). The composite_raw column preserves this.
  - composite_z is the cross-sectional z-score of composite_raw within
    each rebalance date, so it has mean=0, std=1 on a per-date basis.
  - The optimizer should consume composite_z, not composite_raw.

Output panel columns (in order):
  date              rebalance date
  permno            security identifier
  z_gp              GP z-score (input, for diagnostics)
  z_ivol            IVol z-score, SIGN-FLIPPED (i.e., already negated)
  z_resmom          ResMom z-score (input, for diagnostics)
  n_signals         number of non-NaN signals used (1, 2, or 3)
  composite_raw     simple mean of available signed z-scores
  composite_z       composite_raw re-z-scored cross-sectionally per date
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Output column order is part of the module's public API.
COMPOSITE_OUTPUT_COLUMNS: tuple[str, ...] = (
    "date",
    "permno",
    "z_gp",
    "z_ivol",
    "z_resmom",
    "z_pead",
    "n_signals",
    "composite_raw",
    "composite_z",
)

# Sign conventions per docs/strategy_spec.md §5
_SIGN_GP: float = 1.0
_SIGN_IVOL: float = -1.0
_SIGN_RESMOM: float = 1.0
_SIGN_PEAD: float = 1.0  # high SUE → long (positive surprise predicts positive drift)

# Minimum non-NaN signals required to emit a composite
_DEFAULT_MIN_SIGNALS: int = 2


def compute_composite_alpha(
    aligned_gp: pd.DataFrame,
    aligned_ivol: pd.DataFrame,
    aligned_resmom: pd.DataFrame | None = None,
    aligned_pead: pd.DataFrame | None = None,
    min_signals: int = _DEFAULT_MIN_SIGNALS,
) -> pd.DataFrame:
    """Combine three or four aligned signal panels into a composite alpha.

    Parameters
    ----------
    aligned_gp, aligned_ivol : pd.DataFrame
        Output of `signals.alignment.align_signal()` for each signal.
        Each must contain at least 'date', 'permno', 'z_score' columns.
    aligned_resmom : pd.DataFrame or None, default None
        Optional residual momentum signal. When None, the composite
        is computed without ResMom — useful for variants that drop
        signals empirically shown to have zero predictive power.
    aligned_pead : pd.DataFrame or None, default None
        Optional fourth signal (PEAD). When None, the composite uses
        only the three core signals (backward-compatible). When present,
        the composite is equal-weighted across all four signals on a
        per-row basis, with missing signals tolerated up to min_signals.
    min_signals : int, default 2
        Minimum number of non-NaN signals required to emit a row.
        Must be in {1, 2, 3, 4}. When aligned_pead is None, the upper
        bound is effectively 3.

    Returns
    -------
    pd.DataFrame
        Long-format composite panel with columns matching
        COMPOSITE_OUTPUT_COLUMNS, sorted by (date, permno). The z_pead
        column is always present; values are NaN when aligned_pead is
        None or when no PEAD data is available for that name.

    Raises
    ------
    ValueError
        If any input is missing required columns, or min_signals is
        outside {1, 2, 3, 4}.
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if min_signals not in {1, 2, 3, 4}:
        raise ValueError(f"min_signals must be 1, 2, 3, or 4, got {min_signals}")

    required = {"date", "permno", "z_score"}
    signal_panels: list[tuple[str, pd.DataFrame]] = [
        ("aligned_gp", aligned_gp),
        ("aligned_ivol", aligned_ivol),
    ]
    if aligned_resmom is not None:
        signal_panels.append(("aligned_resmom", aligned_resmom))
    if aligned_pead is not None:
        signal_panels.append(("aligned_pead", aligned_pead))
    for name, df in signal_panels:
        if not required.issubset(df.columns):
            missing = required - set(df.columns)
            raise ValueError(f"{name} missing columns: {sorted(missing)}")

    # ------------------------------------------------------------------
    # Extract just (date, permno, z_score) from each, rename z_score per signal
    # ------------------------------------------------------------------
    gp = aligned_gp[["date", "permno", "z_score"]].rename(columns={"z_score": "z_gp"})
    ivol = aligned_ivol[["date", "permno", "z_score"]].rename(columns={"z_score": "z_ivol"})

    # Apply sign conventions and normalize dates for the always-present signals
    gp["z_gp"] = gp["z_gp"] * _SIGN_GP
    ivol["z_ivol"] = ivol["z_ivol"] * _SIGN_IVOL
    for df in (gp, ivol):
        df["date"] = pd.to_datetime(df["date"]).astype("datetime64[ns]")

    resmom: pd.DataFrame | None = None
    if aligned_resmom is not None:
        resmom = aligned_resmom[["date", "permno", "z_score"]].rename(
            columns={"z_score": "z_resmom"}
        )
        resmom["z_resmom"] = resmom["z_resmom"] * _SIGN_RESMOM
        resmom["date"] = pd.to_datetime(resmom["date"]).astype("datetime64[ns]")

    pead: pd.DataFrame | None = None
    if aligned_pead is not None:
        pead = aligned_pead[["date", "permno", "z_score"]].rename(
            columns={"z_score": "z_pead"}
        )
        pead["z_pead"] = pead["z_pead"] * _SIGN_PEAD
        pead["date"] = pd.to_datetime(pead["date"]).astype("datetime64[ns]")

    # ------------------------------------------------------------------
    # Outer-merge all on (date, permno)
    # ------------------------------------------------------------------
    merged = gp.merge(ivol, on=["date", "permno"], how="outer")
    if resmom is not None:
        merged = merged.merge(resmom, on=["date", "permno"], how="outer")
    else:
        # Ensure z_resmom column exists (will be all NaN) so output schema
        # stays consistent with COMPOSITE_OUTPUT_COLUMNS.
        merged["z_resmom"] = np.nan
    if pead is not None:
        merged = merged.merge(pead, on=["date", "permno"], how="outer")
    else:
        # Ensure z_pead column exists (will be all NaN) so output schema
        # stays consistent with COMPOSITE_OUTPUT_COLUMNS.
        merged["z_pead"] = np.nan

    if len(merged) == 0:
        return pd.DataFrame(columns=list(COMPOSITE_OUTPUT_COLUMNS))

    # ------------------------------------------------------------------
    # Count available signals per row
    # ------------------------------------------------------------------
    # Use all four z-columns when PEAD is provided; otherwise just three.
    # When PEAD is absent, z_pead is all NaN so it contributes nothing
    # to counts or means even if naively included — but we be explicit.
    z_cols = ["z_gp", "z_ivol"]
    if aligned_resmom is not None:
        z_cols.append("z_resmom")
    if aligned_pead is not None:
        z_cols.append("z_pead")
    merged["n_signals"] = merged[z_cols].notna().sum(axis=1).astype("int64")

    # Drop rows below threshold
    merged = merged[merged["n_signals"] >= min_signals].copy()

    if len(merged) == 0:
        return pd.DataFrame(columns=list(COMPOSITE_OUTPUT_COLUMNS))

    # ------------------------------------------------------------------
    # Composite raw: mean of available z-scores per row
    # ------------------------------------------------------------------
    merged["composite_raw"] = merged[z_cols].mean(axis=1, skipna=True)

    # ------------------------------------------------------------------
    # Cross-sectional z-score of composite_raw per date
    # ------------------------------------------------------------------
    merged["composite_z"] = merged.groupby("date")["composite_raw"].transform(
        _zscore_within_group
    )

    # ------------------------------------------------------------------
    # Final shape
    # ------------------------------------------------------------------
    result = merged[list(COMPOSITE_OUTPUT_COLUMNS)]
    return result.sort_values(["date", "permno"]).reset_index(drop=True)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _zscore_within_group(s: pd.Series) -> pd.Series:
    """Z-score a Series. Returns NaN for all values if std is ~0."""
    mean = s.mean()
    std = s.std()
    if pd.isna(std) or std < 1e-12:
        return pd.Series(np.nan, index=s.index)
    return (s - mean) / std
