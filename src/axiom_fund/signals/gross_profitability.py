"""Gross Profitability raw signal computation (Novy-Marx 2013).

Formula: gp = (revtq - cogsq) / atq

This module computes the raw signal value per (permno, rdq) row and
nothing else. Universe filtering, winsorization, and cross-sectional
z-scoring all happen in the alignment layer
(see src/axiom_fund/signals/alignment.py).

See docs/signal_design.md §2 (revised) for the design rationale behind
the separation of raw signal computation from alignment.

Output panel columns (in order):
  permno
  gvkey
  date_filed   — the rdq, our PIT-natural anchor
  date_period  — the datadate (fiscal period end), retained for reference
  revtq        — input component
  cogsq        — input component
  atq          — input component (denominator)
  raw_signal   — (revtq - cogsq) / atq, NaN if any input is NaN or atq=0

Usage
-----
    from axiom_fund.signals.gross_profitability import compute_gross_profitability
    from axiom_fund.signals.alignment import align_signal

    # Stage 1: raw signal
    raw = compute_gross_profitability(
        fundamentals_df=fund,
        start_date="2020-01-01",
        end_date="2020-12-31",
    )

    # Stage 2: align to rebalance calendar, winsorize, z-score
    aligned = align_signal(
        raw_signal_df=raw,
        universe_df=universe,
        rebalance_dates=["2020-01-31", "2020-02-29", ...],
    )

Reference
---------
Novy-Marx, R. (2013). "The other side of value: The gross profitability
premium." Journal of Financial Economics, 108(1), 1-28.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

# Output column order is part of the module's public API.
GP_RAW_COLUMNS: tuple[str, ...] = (
    "permno",
    "gvkey",
    "date_filed",
    "date_period",
    "revtq",
    "cogsq",
    "atq",
    "raw_signal",
)


def compute_gross_profitability(
    fundamentals_df: pd.DataFrame,
    start_date: str | date,
    end_date: str | date,
) -> pd.DataFrame:
    """Compute the raw Gross Profitability signal per (permno, rdq).

    Parameters
    ----------
    fundamentals_df : pandas.DataFrame
        Long-format quarterly fundamentals. Must contain columns
        'permno', 'gvkey', 'rdq', 'datadate', 'revtq', 'cogsq', 'atq'.
        (See FUNDAMENTAL_COLUMNS in fundamentals.py for the canonical
        full schema; only the listed subset is required here.)
    start_date, end_date : str or date
        Inclusive window on `rdq` (date_filed). Rows with rdq outside
        this range are dropped.

    Returns
    -------
    pandas.DataFrame
        Long-format raw signal panel sorted by (date_filed, permno) with
        columns matching GP_RAW_COLUMNS. NaN for raw_signal if any input
        is NaN or atq=0.

    Raises
    ------
    ValueError
        If start_date > end_date, or required columns are missing.
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

    required = {"permno", "gvkey", "rdq", "datadate", "revtq", "cogsq", "atq"}
    if not required.issubset(fundamentals_df.columns):
        missing = required - set(fundamentals_df.columns)
        raise ValueError(f"fundamentals_df missing columns: {sorted(missing)}")

    # ------------------------------------------------------------------
    # Filter to date window
    # ------------------------------------------------------------------
    start_ts = pd.Timestamp(start_str)
    end_ts = pd.Timestamp(end_str)
    fund = fundamentals_df.copy()
    fund["rdq"] = pd.to_datetime(fund["rdq"]).astype("datetime64[ns]")
    fund["datadate"] = pd.to_datetime(fund["datadate"]).astype("datetime64[ns]")
    fund = fund[(fund["rdq"] >= start_ts) & (fund["rdq"] <= end_ts)]

    if len(fund) == 0:
        return pd.DataFrame(columns=list(GP_RAW_COLUMNS))

    # ------------------------------------------------------------------
    # Compute raw GP. Division by zero or NaN inputs propagate as NaN.
    # ------------------------------------------------------------------
    fund["raw_signal"] = (fund["revtq"] - fund["cogsq"]) / fund["atq"]
    fund["raw_signal"] = fund["raw_signal"].replace([np.inf, -np.inf], np.nan)

    # ------------------------------------------------------------------
    # Rename columns and reorder
    # ------------------------------------------------------------------
    result = fund.rename(columns={"rdq": "date_filed", "datadate": "date_period"})
    result = result[list(GP_RAW_COLUMNS)]
    return result.sort_values(["date_filed", "permno"]).reset_index(drop=True)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _normalize_date(d: str | date) -> str:
    """Normalize date input to ISO-format string."""
    if isinstance(d, date):
        return d.isoformat()
    if isinstance(d, str):
        parsed = date.fromisoformat(d)
        return parsed.isoformat()
    raise ValueError(f"date must be str or date, got {type(d).__name__}")
