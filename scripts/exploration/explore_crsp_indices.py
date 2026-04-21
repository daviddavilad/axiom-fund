"""One-shot exploration of CRSP index membership tables.

Goal: understand schema and content of the tables we'll use to construct
the S&P 500 + S&P MidCap 400 point-in-time universe.

Not production code. Delete after Phase 1 universe module is built.
"""

from __future__ import annotations

import os
import sys

from axiom_fund import _warnings  # noqa: F401

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import text


def main() -> int:
    load_dotenv()
    username = os.getenv("WRDS_USERNAME")
    if not username:
        print("ERROR: WRDS_USERNAME not set", file=sys.stderr)
        return 1

    import wrds
    db = wrds.Connection(wrds_username=username)

    # ------------------------------------------------------------------
    # 1. List CRSP index-related tables
    # ------------------------------------------------------------------
    print("=" * 70)
    print("CRSP tables related to index membership")
    print("=" * 70)
    query = text(
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'crsp'
          AND (
              table_name ILIKE '%sp500%'
              OR table_name ILIKE '%sp400%'
              OR table_name ILIKE '%msp%'
              OR table_name ILIKE '%dsp%'
              OR table_name ILIKE '%_list%'
          )
        ORDER BY table_name;
        """
    )
    with db.engine.connect() as conn:
        tables = pd.read_sql(query, conn)
    print(tables.to_string(index=False))
    print()

    # ------------------------------------------------------------------
    # 2. Inspect the S&P 500 constituents table
    # ------------------------------------------------------------------
    print("=" * 70)
    print("Schema: crsp.dsp500list_v2")
    print("=" * 70)
    query = text(
        """
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = 'crsp' AND table_name = 'dsp500list_v2'
        ORDER BY ordinal_position;
        """
    )
    with db.engine.connect() as conn:
        schema_500 = pd.read_sql(query, conn)
    print(schema_500.to_string(index=False))
    print()

    print("=" * 70)
    print("Sample rows: crsp.dsp500list_v2 (first 10)")
    print("=" * 70)
    with db.engine.connect() as conn:
        sample_500 = pd.read_sql(
            text("SELECT * FROM crsp.dsp500list_v2 LIMIT 10"), conn
        )
    print(sample_500.to_string(index=False))
    print()

    # ------------------------------------------------------------------
    # 3. Row count and date range for S&P 500 membership
    # ------------------------------------------------------------------
    print("=" * 70)
    print("S&P 500 membership: row count and date range")
    print("=" * 70)
    with db.engine.connect() as conn:
        stats_500 = pd.read_sql(
            text(
                """
                SELECT COUNT(*) AS n_rows,
                       MIN(start) AS earliest_start,
                       MAX(start) AS latest_start,
                       MAX(ending) AS latest_ending,
                       COUNT(*) FILTER (WHERE ending IS NULL) AS currently_in_index
                FROM crsp.dsp500list_v2
                """
            ),
            conn,
        )
    print(stats_500.to_string(index=False))
    print()

    # ------------------------------------------------------------------
    # 4. Count of S&P 500 members active on a specific date
    # ------------------------------------------------------------------
    print("=" * 70)
    print("S&P 500 members active on 2020-01-02")
    print("=" * 70)
    with db.engine.connect() as conn:
        members = pd.read_sql(
            text(
                """
                SELECT COUNT(*) AS n_members
                FROM crsp.dsp500list_v2
                WHERE start <= '2020-01-02'
                  AND (ending IS NULL OR ending >= '2020-01-02')
                """
            ),
            conn,
        )
    print(members.to_string(index=False))
    print()

    db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
