"""Post-Earnings Announcement Drift (PEAD) signal.

Computes the Standardized Unexpected Earnings (SUE) signal per
Bernard & Thomas (1989), using a seasonal random walk forecast and
own-name standardization over a trailing 8-quarter window.

The signal
----------
For each name n at fiscal quarter q with sufficient history:

    surprise_q = epspxq_q - epspxq_{q-4}
    SUE_q = surprise_q / std(surprise_{q-7..q})

where the standardization uses the trailing 8 quarters of seasonal
differences for the same name. The seasonal random walk forecast
(prior-year same-quarter EPS) is the original Bernard-Thomas spec.

Modern PEAD work (Livnat-Mendenhall 2006) uses analyst-based forecasts
(IBES consensus) instead. The time-series version is weaker by ~30%
but doesn't require IBES data, which our WRDS subscription doesn't
include.

The signal is winsorized at ±3 to prevent any single extreme earnings
event from dominating the cross-section.

Point-in-time discipline
------------------------
The SUE for a given quarter is only "known" after the earnings
announcement (rdq). For a rebalance date R, we use the most-recent
announced SUE where rdq ≤ R. This is the same as-of-merge pattern
used by gross_profitability.py and residual_momentum.py.

References
----------
Bernard, V. L., & Thomas, J. K. (1989). "Post-Earnings-Announcement
    Drift: Delayed Price Response or Risk Premium?"
    Journal of Accounting Research, 27, 1-36.
Foster, G., Olsen, C., & Shevlin, T. (1984). "Earnings Releases,
    Anomalies, and the Behavior of Security Returns."
    The Accounting Review, 59(4), 574-603.
Livnat, J., & Mendenhall, R. R. (2006). "Comparing the Post-Earnings
    Announcement Drift for Surprises Calculated from Analyst and
    Time Series Forecasts." Journal of Accounting Research, 44(1).
Hirshleifer, D., Hou, K., Lim, S. S., & Teoh, S. H. (2021).
    "Driven to Distraction: Extraneous Events and Underreaction to
    Earnings News." (For modern drift-window analysis.)
"""

from __future__ import annotations

from datetime import date
from typing import Final

import numpy as np
import pandas as pd


# Required input columns from Fundamentals
_REQUIRED_FUNDAMENTAL_COLUMNS: Final[set[str]] = {
    "permno",
    "rdq",
    "datadate",
    "fyearq",
    "fqtr",
    "epspxq",
    "ajexq",
}

# Default parameters
_DEFAULT_LOOKBACK_QUARTERS: int = 8
_DEFAULT_WINSORIZE_LIMIT: float = 3.0


def compute_pead_signal(
    fundamentals: pd.DataFrame,
    start_date: str | date,
    end_date: str | date,
    lookback_quarters: int = _DEFAULT_LOOKBACK_QUARTERS,
    winsorize_limit: float = _DEFAULT_WINSORIZE_LIMIT,
) -> pd.DataFrame:
    """Compute the raw PEAD signal per (permno, rdq) row.

    Parameters
    ----------
    fundamentals : pd.DataFrame
        Output from Fundamentals.fetch_quarterly. Must contain at
        minimum: 'permno', 'rdq', 'datadate', 'fyearq', 'fqtr', 'epspxq'.
    start_date, end_date : str or date
        Inclusive window on `rdq`. Rows with rdq outside this window
        are excluded from the output (but their data may still be
        used for the trailing 8-quarter standardization of in-window
        rows).
    lookback_quarters : int
        Trailing quarters for standardization. Default 8 per
        Foster-Olsen-Shevlin (1984).
    winsorize_limit : float
        Absolute SUE value beyond which signal is clipped. Default 3.

    Returns
    -------
    pd.DataFrame
        Columns: 'permno', 'date_filed', 'date_period', 'fyearq', 'fqtr',
        'epspxq', 'surprise', 'sue_raw', 'sue'. One row per
        (permno, rdq) combination with a valid (non-null, in-window)
        SUE estimate.

        - date_filed = rdq (announcement date, our PIT anchor)
        - date_period = datadate (fiscal quarter end, retained for ref)
        - surprise = epspxq_q - epspxq_{q-4}
        - sue_raw = surprise / std(surprise over lookback_quarters)
        - sue = sue_raw winsorized at ±winsorize_limit

    Notes
    -----
    Pure function — no I/O. Caller (typically the signal alignment
    layer) is responsible for forwarding-filling to the rebalance
    calendar and z-scoring cross-sectionally per rebalance date.

    Quarters with no t-4 match (e.g., first 4 quarters of a stock's
    history), or with fewer than `lookback_quarters` trailing
    surprises, are excluded.
    """
    missing = _REQUIRED_FUNDAMENTAL_COLUMNS - set(fundamentals.columns)
    if missing:
        raise ValueError(
            f"fundamentals missing required columns: {sorted(missing)}"
        )

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    fund = fundamentals.copy()
    fund["rdq"] = pd.to_datetime(fund["rdq"]).astype("datetime64[ns]")
    fund["datadate"] = pd.to_datetime(fund["datadate"]).astype("datetime64[ns]")
    # Drop rows with null required fields
    fund = fund.dropna(subset=["rdq", "fyearq", "fqtr", "epspxq", "ajexq"])
    # Drop rows where ajexq is non-positive (Compustat data quality guard)
    fund = fund[fund["ajexq"] > 0]

    # Split-adjust EPS using Compustat's cumulative adjustment factor.
    # Compustat's `epspxq` is as-reported, NOT retroactively split-adjusted.
    # A stock that splits 4:1 will show pre-split epspxq at 4x scale and
    # post-split at 1x. The `ajexq` column is the cumulative adjustment
    # factor — dividing epspxq by ajexq normalizes all quarters to a
    # common (current) share count. This is critical for PEAD because
    # seasonal-difference surprises (epspxq_q - epspxq_{q-4}) would be
    # spurious for any name with a split in the trailing 4 quarters.
    fund["epspxq_adj"] = fund["epspxq"] / fund["ajexq"]

    # Sort by permno and fiscal time. We use (fyearq, fqtr) as the
    # canonical quarter ordering rather than rdq, because rdq can
    # have small irregularities (companies sometimes restate or
    # re-file with slightly different rdq dates).
    fund = fund.sort_values(["permno", "fyearq", "fqtr"]).reset_index(drop=True)

    # Compute the surprise per permno using a groupby + shift by 4
    # quarters. shift(4) gives same-quarter-prior-year. Use the
    # split-adjusted EPS.
    fund["epspxq_t4_adj"] = fund.groupby("permno")["epspxq_adj"].shift(4)
    fund["surprise"] = fund["epspxq_adj"] - fund["epspxq_t4_adj"]

    # Compute trailing std of surprise per permno using a rolling
    # window of `lookback_quarters` rows. min_periods is the full
    # window to ensure we have enough history.
    fund["surprise_std"] = (
        fund.groupby("permno")["surprise"]
        .transform(
            lambda s: s.rolling(
                window=lookback_quarters, min_periods=lookback_quarters
            ).std()
        )
    )

    # SUE is the standardized surprise. Skip rows where std is zero
    # (constant surprise across history — implies no information)
    # or null (insufficient history).
    fund = fund.dropna(subset=["surprise", "surprise_std"])
    fund = fund[fund["surprise_std"] > 0]
    fund["sue_raw"] = fund["surprise"] / fund["surprise_std"]

    # Winsorize
    fund["sue"] = fund["sue_raw"].clip(
        lower=-winsorize_limit, upper=winsorize_limit
    )

    # Restrict to rdq window
    fund = fund[(fund["rdq"] >= start_ts) & (fund["rdq"] <= end_ts)]
    if len(fund) == 0:
        return pd.DataFrame(columns=[
            "permno", "date_filed", "date_period", "fyearq", "fqtr",
            "epspxq", "surprise", "sue_raw", "sue",
        ])

    result = fund.rename(
        columns={"rdq": "date_filed", "datadate": "date_period"}
    )
    return result[[
        "permno",
        "date_filed",
        "date_period",
        "fyearq",
        "fqtr",
        "epspxq",
        "ajexq",
        "epspxq_adj",
        "surprise",
        "sue_raw",
        "sue",
    ]].reset_index(drop=True)