"""Build the Lazy Prices sample universe: union of top-1000 across 2019-2026.

Per docs/v2_item6_design.md, Item 6a originally operated on a 2019-2024
window. Extended 2026-07-19 to include 2025 + 2026 YTD filings for improved
statistical power (see "Sample forward extension (2026-07-19)" section).
This script constructs the universe as the union of top-1000 U.S. common
stocks by market cap at each year-end 2018-12-31 through 2025-12-31 (8
snapshots).

Rationale for union rather than snapshot: firms enter/exit the top-1000 over
the sample window. Using a single-date snapshot would exclude firms that
grew or shrunk into the top-1000 during the window, biasing the sample.

Output:
  data/cache/lazy_prices_universe.parquet — one row per (permno, ticker,
  first_date_in_top1000) with metadata columns

The output is used by download_10k_corpus.py (next phase) to resolve CIKs
and fetch filings.
"""
# ruff: noqa: I001

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from axiom_fund import _warnings  # noqa: F401
from axiom_fund.data.universe import Universe, UniverseConfig


# Year-end snapshots covering 2019-2024 filing window.
# 2018-12-31: universe entering 2019 (first year of sample)
# 2023-12-31: universe entering 2024 (last year of sample)
# Note: use actual last trading days of each year. Universe.as_of() does an
# exact date match against CRSP monthly file and returns 0 rows silently on
# non-trading days. 2022-12-31 (Sat) and 2023-12-31 (Sun) fell on weekends.
SNAPSHOT_DATES = [
    "2018-12-31",  # Monday, trading day
    "2019-12-31",  # Tuesday, trading day
    "2020-12-31",  # Thursday, trading day
    "2021-12-31",  # Friday, trading day
    "2022-12-30",  # Friday (Dec 31 was Saturday)
    "2023-12-29",  # Friday (Dec 31 was Sunday)
    "2024-12-31",  # Tuesday, trading day (added 2026-07-19)
    "2025-12-31",  # Wednesday, trading day (added 2026-07-19)
]

OUTPUT_PATH = Path("data/cache/lazy_prices_universe.parquet")


def main() -> int:
    load_dotenv()
    username = os.getenv("WRDS_USERNAME")
    if not username:
        print("ERROR: WRDS_USERNAME not set in .env", file=sys.stderr)
        return 1

    print(f"Building universe: union of top-1000 across {len(SNAPSHOT_DATES)} snapshots")
    print(f"Snapshots: {SNAPSHOT_DATES}")
    print()

    import wrds
    db = wrds.Connection(wrds_username=username)

    all_members: list[pd.DataFrame] = []
    try:
        u = Universe(db)
        print(f"Config: size={u.config.size}, share_codes={u.config.share_codes}, "
              f"price_floor={u.config.price_floor}, adv_floor={u.config.adv_floor:,.0f}")
        print()

        for snapshot_date in SNAPSHOT_DATES:
            print(f"Querying universe as of {snapshot_date}...")
            members = u.as_of(snapshot_date)
            members = members.copy()
            members["snapshot_date"] = snapshot_date
            all_members.append(members)
            print(f"  Returned {len(members)} rows")

    finally:
        db.close()

    # Concatenate all snapshots
    combined = pd.concat(all_members, ignore_index=True)
    print(f"\nTotal rows before dedup: {len(combined):,}")

    # Union: one row per permno, keep the earliest snapshot_date it appeared in
    combined = combined.sort_values(["permno", "snapshot_date"])
    unique = combined.drop_duplicates(subset=["permno"], keep="first").reset_index(drop=True)
    unique = unique.rename(columns={"snapshot_date": "first_snapshot_date"})
    print(f"Unique firms in union: {len(unique):,}")

    # Sanity: how many snapshots did each firm appear in?
    snapshot_counts = combined.groupby("permno").size()
    print(f"\nDistribution of snapshot appearances per firm:")
    for n_snaps in range(1, 7):
        n_firms = (snapshot_counts == n_snaps).sum()
        print(f"  {n_snaps} snapshot{'s' if n_snaps > 1 else ' '}: {n_firms:>5} firms")

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    unique.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved: {OUTPUT_PATH}")
    print(f"Rows: {len(unique)}, Columns: {list(unique.columns)}")

    # Write metadata
    metadata_path = OUTPUT_PATH.with_suffix(".txt")
    metadata_path.write_text(
        f"Run: {datetime.now().isoformat()}\n"
        f"Snapshot dates: {SNAPSHOT_DATES}\n"
        f"Universe config: size={u.config.size}, "
        f"share_codes={u.config.share_codes}, "
        f"price_floor={u.config.price_floor}, "
        f"adv_floor={u.config.adv_floor}\n"
        f"Total unique firms: {len(unique)}\n"
    )
    print(f"Metadata: {metadata_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())