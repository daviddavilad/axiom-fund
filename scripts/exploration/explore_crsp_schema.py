"""Explore CRSP stock file schema to identify columns needed for universe construction.

One-shot exploration script. We need to know:
  - Which columns are in crsp.dsf (prices, returns, volume)
  - Which columns are in crsp.stocknames or crsp.dse (share code, exchange, SIC)
  - What the data looks like for a few sample PERMNOs
  - What values show up in share code, exchange code, SIC code columns
"""

from __future__ import annotations

import os
import sys

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import text

from axiom_fund import _warnings  # noqa: F401


def query(db, sql: str) -> pd.DataFrame:
    with db.engine.connect() as conn:
        return pd.read_sql(text(sql), conn)


def main() -> int:
    load_dotenv()
    username = os.getenv("WRDS_USERNAME")
    if not username:
        print("ERROR: WRDS_USERNAME not set", file=sys.stderr)
        return 1

    import wrds
    db = wrds.Connection(wrds_username=username)

    # ------------------------------------------------------------------
    # 1. All CRSP tables we can access
    # ------------------------------------------------------------------
    print("=" * 70)
    print("Accessible CRSP tables")
    print("=" * 70)
    tables = query(
        db,
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'crsp'
          AND table_type = 'BASE TABLE'
        ORDER BY table_name;
        """,
    )
    print(tables.to_string(index=False))
    print(f"\n({len(tables)} tables)\n")

    # ------------------------------------------------------------------
    # 2. Schema of crsp.dsf (daily stock file)
    # ------------------------------------------------------------------
    print("=" * 70)
    print("Schema: crsp.dsf")
    print("=" * 70)
    dsf_schema = query(
        db,
        """
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = 'crsp' AND table_name = 'dsf'
        ORDER BY ordinal_position;
        """,
    )
    print(dsf_schema.to_string(index=False))
    print()

    # ------------------------------------------------------------------
    # 3. Find the "names" table (likely stocknames, stocknames_v2, or dse)
    # ------------------------------------------------------------------
    print("=" * 70)
    print("Tables with 'name' or 'header' in name")
    print("=" * 70)
    names_tables = query(
        db,
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'crsp'
          AND (table_name ILIKE '%name%' OR table_name ILIKE '%header%'
               OR table_name ILIKE '%dse%' OR table_name ILIKE '%stock%')
        ORDER BY table_name;
        """,
    )
    print(names_tables.to_string(index=False))
    print()

    # ------------------------------------------------------------------
    # 4. Schema of crsp.stocknames (if accessible)
    # ------------------------------------------------------------------
    print("=" * 70)
    print("Schema: crsp.stocknames (names/header info)")
    print("=" * 70)
    try:
        stocknames_schema = query(
            db,
            """
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = 'crsp' AND table_name = 'stocknames'
            ORDER BY ordinal_position;
            """,
        )
        print(stocknames_schema.to_string(index=False))
    except Exception as e:
        print(f"stocknames not accessible: {e}")
    print()

    # ------------------------------------------------------------------
    # 5. Sample rows from crsp.dsf for a known PERMNO (AAPL = 14593)
    # ------------------------------------------------------------------
    print("=" * 70)
    print("Sample: crsp.dsf for PERMNO 14593 (Apple) around 2020-01-02")
    print("=" * 70)
    sample = query(
        db,
        """
        SELECT *
        FROM crsp.dsf
        WHERE permno = 14593
          AND date BETWEEN '2019-12-30' AND '2020-01-06'
        ORDER BY date;
        """,
    )
    print(sample.to_string(index=False))
    print()

    # ------------------------------------------------------------------
    # 6. Distribution of share codes on a sample date
    # ------------------------------------------------------------------
    print("=" * 70)
    print("Distribution of CRSP fields of interest (2020-01-02)")
    print("=" * 70)

    # If dsf has shrcd/exchcd/siccd directly, we can query them.
    # If not, we'll need a join — we'll see from the dsf schema above.
    try:
        dist = query(
            db,
            """
            SELECT
                COUNT(*) AS n_rows,
                COUNT(DISTINCT permno) AS n_permnos,
                COUNT(*) FILTER (WHERE prc > 0) AS n_positive_prc,
                COUNT(*) FILTER (WHERE prc < 0) AS n_negative_prc,
                COUNT(*) FILTER (WHERE prc IS NULL) AS n_null_prc,
                COUNT(*) FILTER (WHERE ret IS NULL) AS n_null_ret
            FROM crsp.dsf
            WHERE date = '2020-01-02';
            """,
        )
        print(dist.to_string(index=False))
    except Exception as e:
        print(f"Query failed: {e}")

    db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
