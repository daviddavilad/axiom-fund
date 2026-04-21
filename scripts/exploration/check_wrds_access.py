"""Check which CRSP/Compustat sub-schemas are accessible with this account."""

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

    # List all schemas the user can access
    print("=" * 70)
    print("All schemas visible to this account")
    print("=" * 70)
    with db.engine.connect() as conn:
        schemas = pd.read_sql(
            text(
                """
                SELECT schema_name
                FROM information_schema.schemata
                WHERE schema_name LIKE 'crsp%'
                   OR schema_name LIKE 'comp%'
                   OR schema_name LIKE 'wrds%'
                ORDER BY schema_name
                """
            ),
            conn,
        )
    print(schemas.to_string(index=False))
    print()

    # Test each candidate schema/table we'll likely need
    print("=" * 70)
    print("Access probe: can we SELECT from key tables?")
    print("=" * 70)

    probes = [
        ("crsp.dsf", "CRSP daily stock file"),
        ("crsp.msf", "CRSP monthly stock file"),
        ("crsp.dsp500list_v2", "S&P 500 membership (view)"),
        ("crsp_a_indexes.dsp500list_v2", "S&P 500 membership (underlying)"),
        ("crsp_a_stock.dsf", "CRSP daily (annual)"),
        ("crsp_m_stock.msf", "CRSP monthly (annual)"),
        ("comp.fundq", "Compustat quarterly fundamentals"),
        ("comp.company", "Compustat company master"),
        ("comp_na_daily_all.fundq", "Compustat quarterly (full product)"),
        ("crsp.ccmxpf_linktable", "CRSP/Compustat Merged link table"),
        ("crsp_a_ccm.ccmxpf_linktable", "CCM link table (annual)"),
    ]

    for table, desc in probes:
        try:
            with db.engine.connect() as conn:
                pd.read_sql(text(f"SELECT 1 FROM {table} LIMIT 1"), conn)
            status = "OK  "
        except Exception as e:
            err_str = str(e).split("\n")[0][:80]
            status = f"FAIL: {err_str}"
        print(f"  {status:<50} {table:<40} {desc}")

    db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
