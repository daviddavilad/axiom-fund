"""Stage 1 backtest verification: single-firm end-to-end trace.

Pick a firm known to be in the extreme right tail of raw_signal
(GE 2020-02-24, raw_signal ~0.95) and follow it through:
  1. Signal file (raw_signal at filing_date)
  2. Aligned panel (z_score at next month-end after filing + buffer)
  3. Quintile assignment (should be Q5 = long leg)
  4. Forward return (21-day; Feb 2020 -> Mar 2020 = COVID crash)
  5. Contribution to the L/S monthly return

Also picks a matched "boring" mega-cap (AAPL, MSFT, or JNJ) for
contrast — should have low raw_signal and land in Q1-Q3.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


CACHE = Path("data/cache")
BACKTEST = CACHE / "lazy_prices_backtest"


def main() -> None:
    signal = pd.read_parquet(CACHE / "lazy_prices_signal.parquet")
    returns = pd.read_parquet(CACHE / "lazy_prices_returns_daily.parquet")
    positions = pd.read_parquet(BACKTEST / "quintile_positions.parquet")
    monthly = pd.read_parquet(BACKTEST / "monthly_returns.parquet")

    print("=" * 70)
    print("GE trace")
    print("=" * 70)

    ge_signal = signal[signal.ticker == "GE"].sort_values("date_filed")
    print("\n1. Signal history:")
    print(ge_signal[["ticker", "date_filed", "raw_signal"]].to_string(index=False))

    ge_permno = int(ge_signal.permno.iloc[0])
    print(f"\nGE permno: {ge_permno}")

    # First rebalance after 2020-02-24
    print("\n2. Quintile assignments at each month-end (2020 through 2021):")
    ge_positions = positions[
        (positions.permno == ge_permno)
        & (positions.date >= "2020-01-31")
        & (positions.date <= "2021-06-30")
    ].sort_values("date")
    print(ge_positions.to_string(index=False))

    print("\n3. GE daily returns Feb 24 - Mar 30 2020 (COVID window):")
    ge_returns = returns[
        (returns.permno == ge_permno)
        & (returns.date >= "2020-02-24")
        & (returns.date <= "2020-03-30")
    ].sort_values("date")
    print(ge_returns[["date", "ret", "prc"]].to_string(index=False))

    # Cumulative return over that window
    if len(ge_returns) > 0:
        cum = (1 + ge_returns.ret).prod() - 1
        print(f"\nCumulative return Feb 24 - Mar 30: {cum:.2%}")

    # Now a boring mega-cap for contrast
    print()
    print("=" * 70)
    print("AAPL trace (boring mega-cap contrast)")
    print("=" * 70)

    aapl_signal = signal[signal.ticker == "AAPL"].sort_values("date_filed")
    print("\n1. Signal history:")
    print(aapl_signal[["ticker", "date_filed", "raw_signal"]].to_string(index=False))

    aapl_permno = int(aapl_signal.permno.iloc[0])
    print(f"\nAAPL permno: {aapl_permno}")

    print("\n2. Quintile assignments at each month-end (2020 through 2021):")
    aapl_positions = positions[
        (positions.permno == aapl_permno)
        & (positions.date >= "2020-01-31")
        & (positions.date <= "2021-06-30")
    ].sort_values("date")
    print(aapl_positions.to_string(index=False))

    # L/S monthly return for the rebalance right after GE's Feb 2020 filing
    print()
    print("=" * 70)
    print("Feb-Mar 2020 L/S month")
    print("=" * 70)
    feb_mar = monthly[
        (monthly.date >= "2020-02-01") & (monthly.date <= "2020-04-30")
    ]
    print(feb_mar.to_string(index=False))


if __name__ == "__main__":
    main()