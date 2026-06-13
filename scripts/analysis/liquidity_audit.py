"""v2 Phase 1 liquidity audit.

Measures stale-price contamination in the v1 universe (the actual
permnos selected by the 116-period top-1000 backtest). Drives the
threshold-selection conversation for v2's tighter liquidity screens.

Diagnostics:
  1. For each (permno, rebalance_date), compute fraction of trailing
     60 trading days with zero volume.
  2. For each (permno, rebalance_date), compute longest run of
     consecutive zero-return days in trailing 60.
  3. Headline: what % of name-months would v2 candidate screens exclude?

Output:
  data/cache/liquidity_audit/contamination_long.parquet
  data/cache/liquidity_audit/contamination_long.csv
  Console summary in wide format.

This script is one-off; not part of the production backtest.
"""
# ruff: noqa: I001

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from axiom_fund import _warnings  # noqa: F401

import wrds

from axiom_fund.data.returns import ReturnsPanel


WEIGHTS_DIR = Path("data/cache/backtest_full_top1000")
OUTPUT_DIR = Path("data/cache/liquidity_audit")

TRAILING_DAYS = 60  # trading days

# Candidate v2 thresholds — tested against the audit data
CANDIDATE_THRESHOLDS = {
    "zero_volume_share_10pct":   {"zero_vol_share_min": 0.10},
    "max_consec_zero_5":          {"consec_zero_min": 5},
    "max_consec_zero_10":         {"consec_zero_min": 10},
    "union_10pct_and_5consec":    {"zero_vol_share_min": 0.10,
                                    "consec_zero_min": 5},
}


def _load_universe_permnos() -> tuple[set[int], list[pd.Timestamp]]:
    """Union of permnos across all 116 rebalance dates + the date list."""
    paths = sorted(WEIGHTS_DIR.glob("weights_*.parquet"))
    permnos: set[int] = set()
    dates: list[pd.Timestamp] = []
    for p in paths:
        df = pd.read_parquet(p)
        permnos.update(df.index.tolist())
        # extract date from filename: weights_2015-01-30.parquet
        date_str = p.stem.removeprefix("weights_")
        dates.append(pd.Timestamp(date_str))
    return permnos, sorted(dates)


def _load_v1_universe_per_date() -> dict[pd.Timestamp, set[int]]:
    """Permnos active in v1's universe at each rebalance date."""
    paths = sorted(WEIGHTS_DIR.glob("weights_*.parquet"))
    out: dict[pd.Timestamp, set[int]] = {}
    for p in paths:
        df = pd.read_parquet(p)
        date_str = p.stem.removeprefix("weights_")
        out[pd.Timestamp(date_str)] = set(df.index.tolist())
    return out


def _longest_zero_run(returns: pd.Series) -> int:
    """Length of the longest consecutive zero-return run in the series."""
    if len(returns) == 0:
        return 0
    is_zero = (returns == 0.0) | returns.isna()
    longest = current = 0
    for z in is_zero:
        if z:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _audit_one_date(
    panel: pd.DataFrame,
    permnos_at_date: set[int],
    date: pd.Timestamp,
    trailing_days: int = TRAILING_DAYS,
) -> pd.DataFrame:
    """For one rebalance date, compute diagnostics for each permno in the universe."""
    cutoff_start = date - pd.Timedelta(days=trailing_days * 2)  # generous calendar buffer
    window = panel[
        (panel["date"] >= cutoff_start) & (panel["date"] <= date)
    ]
    rows = []
    for permno in permnos_at_date:
        sub = window[window["permno"] == permno].sort_values("date").tail(trailing_days)
        if len(sub) == 0:
            rows.append({
                "rebalance_date": date,
                "permno": permno,
                "n_days_available": 0,
                "zero_vol_share": pd.NA,
                "max_consec_zero": pd.NA,
            })
            continue
        n = len(sub)
        zero_vol = ((sub["vol"] == 0) | sub["vol"].isna()).sum()
        zero_vol_share = zero_vol / n
        max_consec = _longest_zero_run(sub["ret"])
        rows.append({
            "rebalance_date": date,
            "permno": permno,
            "n_days_available": n,
            "zero_vol_share": zero_vol_share,
            "max_consec_zero": max_consec,
        })
    return pd.DataFrame(rows)


def _print_distribution(df: pd.DataFrame) -> None:
    """Console summary."""
    print()
    print("=" * 80)
    print("v2 Phase 1 Liquidity Audit — Contamination of v1 Universe")
    print("=" * 80)
    valid = df.dropna(subset=["zero_vol_share"])
    print(f"\nName-months audited: {len(df):,}")
    print(f"Name-months with valid data: {len(valid):,}")
    print(f"Name-months with NO data (gap, delisting, etc): {len(df) - len(valid):,}")

    print("\nDistribution of zero_vol_share (trailing 60 trading days):")
    desc = valid["zero_vol_share"].describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99])
    print(f'  mean        {desc["mean"]:.4f}')
    print(f'  p50         {desc["50%"]:.4f}')
    print(f'  p75         {desc["75%"]:.4f}')
    print(f'  p90         {desc["90%"]:.4f}')
    print(f'  p95         {desc["95%"]:.4f}')
    print(f'  p99         {desc["99%"]:.4f}')
    print(f'  max         {desc["max"]:.4f}')

    print("\nDistribution of max_consec_zero (trailing 60 trading days):")
    desc = valid["max_consec_zero"].describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99])
    print(f'  mean        {desc["mean"]:.2f}')
    print(f'  p50         {desc["50%"]:.0f}')
    print(f'  p75         {desc["75%"]:.0f}')
    print(f'  p90         {desc["90%"]:.0f}')
    print(f'  p95         {desc["95%"]:.0f}')
    print(f'  p99         {desc["99%"]:.0f}')
    print(f'  max         {desc["max"]:.0f}')

    print("\nCandidate v2 threshold contamination:")
    for name, params in CANDIDATE_THRESHOLDS.items():
        mask = pd.Series(False, index=valid.index)
        if "zero_vol_share_min" in params:
            mask |= valid["zero_vol_share"] >= params["zero_vol_share_min"]
        if "consec_zero_min" in params:
            mask |= valid["max_consec_zero"] >= params["consec_zero_min"]
        excluded = mask.sum()
        share = excluded / len(valid) * 100
        print(f"  {name:<32} excludes {excluded:,} / {len(valid):,} ({share:.2f}%)")


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load universe by date
    print("Loading v1 universe from weights cache...")
    universe_by_date = _load_v1_universe_per_date()
    all_permnos: set[int] = set()
    for s in universe_by_date.values():
        all_permnos.update(s)
    dates = sorted(universe_by_date.keys())
    print(f"  {len(dates)} rebalance dates, {len(all_permnos):,} unique permnos in union")

    # Bulk WRDS fetch
    earliest = min(dates) - pd.Timedelta(days=120)  # buffer for trailing-60 window
    latest = max(dates)
    print(f"\nFetching daily returns + volume from {earliest.date()} to {latest.date()}...")
    print("  (this may take 1-2 minutes)")
    load_dotenv()
    username = os.getenv("WRDS_USERNAME")
    if not username:
        print("ERROR: WRDS_USERNAME not found in .env", file=sys.stderr)
        return 1
    db = wrds.Connection(wrds_username=username)
    try:
        returns_panel = ReturnsPanel(db)
        panel = returns_panel.fetch(
            permnos=sorted(all_permnos),
            start_date=earliest.date().isoformat(),
            end_date=latest.date().isoformat(),
        )
    finally:
        db.close()
    print(f"  fetched {len(panel):,} rows")
    panel["date"] = pd.to_datetime(panel["date"])

    # Audit per rebalance date
    print(f"\nAuditing {len(dates)} rebalance dates...")
    all_results = []
    for i, date in enumerate(dates, 1):
        if i % 20 == 0 or i == len(dates):
            print(f"  {i}/{len(dates)}: {date.date()}")
        permnos_at_date = universe_by_date[date]
        result = _audit_one_date(panel, permnos_at_date, date)
        all_results.append(result)

    df_out = pd.concat(all_results, ignore_index=True)

    # Save
    df_out.to_parquet(OUTPUT_DIR / "contamination_long.parquet", index=False)
    df_out.to_csv(OUTPUT_DIR / "contamination_long.csv", index=False)

    # Metadata
    meta = OUTPUT_DIR / "run_metadata.txt"
    meta.write_text(
        f"Run: {datetime.now().isoformat()}\n"
        f"Trailing days: {TRAILING_DAYS}\n"
        f"Rebalance dates: {len(dates)}\n"
        f"Total name-months audited: {len(df_out):,}\n"
    )

    _print_distribution(df_out)

    print(f"\nSaved to: {OUTPUT_DIR}/contamination_long.parquet (+ .csv)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())