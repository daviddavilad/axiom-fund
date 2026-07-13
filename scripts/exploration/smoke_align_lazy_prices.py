"""One-off smoke test: verify align_signal handles our annual Lazy Prices
signal end-to-end using the union-set as a coarse universe.

Not a production check. Real check for the backtest will build a
per-date universe panel via CRSP.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from axiom_fund.signals.alignment import align_signal


SIGNAL_PATH = Path("data/cache/lazy_prices_signal.parquet")
UNIVERSE_PATH = Path("data/cache/lazy_prices_universe.parquet")


def main() -> int:
    signal = pd.read_parquet(SIGNAL_PATH)
    universe_union = pd.read_parquet(UNIVERSE_PATH)

    print(f"Signal rows: {len(signal):,}")
    print(f"Signal permnos: {signal.permno.nunique():,}")
    print(f"Signal date range: {signal.date_filed.min()} to {signal.date_filed.max()}")
    print(f"Union universe firms: {len(universe_union):,}")

    # Build a coarse per-date universe panel: every union firm is 'in'
    # for every month-end 2019-01 through 2024-12.
    rebalance_dates = pd.date_range("2019-01-31", "2024-12-31", freq="ME")
    union_permnos = universe_union.permno.unique()

    universe_panel = pd.DataFrame(
        [(d, p) for d in rebalance_dates for p in union_permnos],
        columns=["date", "permno"],
    )
    print(f"Universe panel: {len(universe_panel):,} rows "
          f"({len(rebalance_dates)} dates x {len(union_permnos)} firms)")
    print()

    # Signal has (permno, date_filed, raw_signal, ...) — align_signal
    # requires exactly permno, date_filed, raw_signal at minimum.
    signal_minimal = signal[["permno", "date_filed", "raw_signal"]]

    print("Calling align_signal (max_age_days=None, annual carry)...")
    aligned = align_signal(
        raw_signal_df=signal_minimal,
        universe_df=universe_panel,
        rebalance_dates=list(rebalance_dates),
        max_age_days=None,
    )
    print(f"Aligned rows: {len(aligned):,}")
    print(f"Aligned columns: {list(aligned.columns)}")
    print()

    # Per-rebalance stats: how many firms have non-null raw_signal?
    print("Coverage per rebalance date (first & last 3):")
    coverage = (
        aligned.groupby("date")
        .agg(
            n_total=("permno", "size"),
            n_with_signal=("raw_signal", "count"),
        )
        .assign(pct_covered=lambda d: 100 * d.n_with_signal / d.n_total)
    )
    print(coverage.head(3).to_string())
    print("...")
    print(coverage.tail(3).to_string())
    print()

    # Sanity: forward-fill should mean coverage grows over time as more
    # firms file their first 10-K in-window
    print(f"Coverage summary: min={coverage.pct_covered.min():.1f}%, "
          f"max={coverage.pct_covered.max():.1f}%, "
          f"mean={coverage.pct_covered.mean():.1f}%")
    print()

    # Sample from the last rebalance to eyeball values
    last = aligned[aligned.date == aligned.date.max()]
    valid = last[last.raw_signal.notna()]
    print(f"At {aligned.date.max().date()}: {len(valid)} firms with a signal")
    print(f"  raw_signal quartiles: "
          f"{valid.raw_signal.quantile([0.25, 0.5, 0.75]).to_list()}")
    print(f"  z_score range: [{valid.z_score.min():.2f}, {valid.z_score.max():.2f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())