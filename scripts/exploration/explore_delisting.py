"""Explore CRSP delisting data to understand its structure.

We need to know:
  - Which tables hold delisting information (dsedelist, mse62delist, etc.)
  - What columns they have (especially dlret, dlstcd, dlstdt)
  - What values dlstcd takes and what they mean (delisting reason codes)
  - Whether delisting returns (dlret) are also embedded in crsp.dsf
    (they often are, on the last trading day of the delisted security)
  - How many delistings happen in a typical year
"""

from __future__ import annotations

import os
import sys

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import text

from axiom_fund import _warnings  # noqa: F401


def query(db, sql: str, params: dict | None = None) -> pd.DataFrame:
    with db.engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params or {})


def main() -> int:
    load_dotenv()
    username = os.getenv("WRDS_USERNAME")
    if not username:
        print("ERROR: WRDS_USERNAME not set", file=sys.stderr)
        return 1

    import wrds
    db = wrds.Connection(wrds_username=username)

    # ------------------------------------------------------------------
    # 1. Find all delisting-related tables
    # ------------------------------------------------------------------
    print("=" * 70)
    print("Delisting-related tables")
    print("=" * 70)
    tables = query(
        db,
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'crsp'
          AND (table_name ILIKE '%delist%' OR table_name ILIKE '%dlst%')
        ORDER BY table_name;
        """,
    )
    print(tables.to_string(index=False))
    print()

    # ------------------------------------------------------------------
    # 2. Schema of dsedelist (daily delisting events)
    # ------------------------------------------------------------------
    print("=" * 70)
    print("Schema: crsp.dsedelist")
    print("=" * 70)
    try:
        schema = query(
            db,
            """
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = 'crsp' AND table_name = 'dsedelist'
            ORDER BY ordinal_position;
            """,
        )
        print(schema.to_string(index=False))
    except Exception as e:
        print(f"Error: {e}")
    print()

    # ------------------------------------------------------------------
    # 3. Sample of dsedelist rows
    # ------------------------------------------------------------------
    print("=" * 70)
    print("Sample: crsp.dsedelist (first 10 rows in 2020)")
    print("=" * 70)
    try:
        sample = query(
            db,
            """
            SELECT *
            FROM crsp.dsedelist
            WHERE dlstdt >= '2020-01-01' AND dlstdt <= '2020-12-31'
            ORDER BY dlstdt
            LIMIT 10;
            """,
        )
        print(sample.to_string(index=False))
    except Exception as e:
        print(f"Error: {e}")
    print()

    # ------------------------------------------------------------------
    # 4. Distribution of dlstcd (delisting reason codes) in 2020
    # ------------------------------------------------------------------
    print("=" * 70)
    print("Distribution of delisting codes (dlstcd) in 2020")
    print("=" * 70)
    try:
        dist = query(
            db,
            """
            SELECT dlstcd, COUNT(*) AS n_delists
            FROM crsp.dsedelist
            WHERE dlstdt >= '2020-01-01' AND dlstdt <= '2020-12-31'
            GROUP BY dlstcd
            ORDER BY n_delists DESC;
            """,
        )
        print(dist.to_string(index=False))
    except Exception as e:
        print(f"Error: {e}")
    print()

    # ------------------------------------------------------------------
    # 5. Does crsp.dsf contain a delisting return column?
    # ------------------------------------------------------------------
    print("=" * 70)
    print("Does crsp.dsf have a delisting return column?")
    print("=" * 70)
    dsf_cols = query(
        db,
        """
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = 'crsp' AND table_name = 'dsf'
          AND (column_name ILIKE '%dl%' OR column_name ILIKE '%delist%')
        ORDER BY column_name;
        """,
    )
    if len(dsf_cols) == 0:
        print("No delisting-related columns found in crsp.dsf")
        print("Delisting returns are only in crsp.dsedelist — must be joined")
    else:
        print(dsf_cols.to_string(index=False))
    print()

    # ------------------------------------------------------------------
    # 6. Check a known delisting: Sprint (PERMNO 82775), delisted 2020-04-01
    #    when T-Mobile acquired it
    # ------------------------------------------------------------------
    print("=" * 70)
    print("Known delisting: Sprint (PERMNO 82775), T-Mobile merger April 2020")
    print("=" * 70)
    try:
        sprint_delist = query(
            db,
            """
            SELECT *
            FROM crsp.dsedelist
            WHERE permno = 82775;
            """,
        )
        print(sprint_delist.to_string(index=False))
    except Exception as e:
        print(f"Error: {e}")
    print()

    db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
