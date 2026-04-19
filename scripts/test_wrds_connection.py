"""Diagnostic script to verify WRDS connectivity.

Runs a minimal query against CRSP to confirm:
  1. Credentials in .env are loaded correctly
  2. WRDS library can authenticate
  3. .pgpass is configured properly
  4. A trivial SQL query returns expected data

Queries go through the SQLAlchemy engine directly rather than through
wrds.Connection.raw_sql(), which has a compatibility bug with pandas>=2.2.
Using the engine directly is also the standard pattern we will use in
production pipeline code.

This is a diagnostic script, not part of the production pipeline.
"""

from __future__ import annotations

import os
import sys

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import text


def main() -> int:
    """Run the connection test. Returns 0 on success, 1 on failure."""
    load_dotenv()

    username = os.getenv("WRDS_USERNAME")
    if not username:
        print("ERROR: WRDS_USERNAME not found in environment.", file=sys.stderr)
        print("Is .env present in the project root with WRDS_USERNAME set?", file=sys.stderr)
        return 1

    print(f"Connecting to WRDS as user: {username}")

    import wrds

    try:
        db = wrds.Connection(wrds_username=username)
    except Exception as e:
        print(f"ERROR: Failed to connect to WRDS: {e}", file=sys.stderr)
        return 1

    print("Connection established.\n")

    print("Running test query against CRSP...")
    query = text(
        """
        SELECT COUNT(*) AS n_rows
        FROM crsp.dsf
        WHERE date = '2020-01-02'
        """
    )

    try:
        # Use the SQLAlchemy engine directly to avoid wrds.raw_sql bug
        with db.engine.connect() as conn:
            result = pd.read_sql(query, conn)
    except Exception as e:
        print(f"ERROR: Query failed: {e}", file=sys.stderr)
        db.close()
        return 1

    n_rows = int(result.iloc[0]["n_rows"])
    print(f"CRSP daily stock file contains {n_rows:,} rows for 2020-01-02.")

    db.close()
    print("\nConnection closed successfully.")
    print("WRDS setup verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
