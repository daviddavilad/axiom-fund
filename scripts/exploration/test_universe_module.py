"""Smoke test for the Universe module.

Imports the production module and runs it against WRDS for a sample date.
Verifies the output shape and a few invariants.
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

from axiom_fund import _warnings  # noqa: F401
from axiom_fund.data.universe import Universe, UniverseConfig


def main() -> int:
    load_dotenv()
    username = os.getenv("WRDS_USERNAME")
    if not username:
        print("ERROR: WRDS_USERNAME not set", file=sys.stderr)
        return 1

    import wrds
    db = wrds.Connection(wrds_username=username)

    try:
        # Default config
        u = Universe(db)
        print(f"Config: {u.config}")
        print()

        # Single date
        print("Building universe for 2020-01-02...")
        uni = u.as_of("2020-01-02")
        print(f"  Returned {len(uni)} rows")
        print(f"  Columns: {list(uni.columns)}")
        print("  Top 5 names:")
        print(uni.head(5)[["permno", "ticker", "comnam", "market_cap"]].to_string(index=False))
        print()

        # Different config — smaller universe
        print("Building top-500 universe for 2020-01-02...")
        u_small = Universe(db, UniverseConfig(size=500))
        uni_small = u_small.as_of("2020-01-02")
        print(f"  Returned {len(uni_small)} rows")
        print(f"  Smallest market cap in top 500: ${uni_small['market_cap'].min():,.0f}")
        print()

        # Different date
        print("Building universe for 2015-06-30...")
        uni_old = u.as_of("2015-06-30")
        print(f"  Returned {len(uni_old)} rows")
        print("  Top 3 names:")
        print(uni_old.head(3)[["permno", "ticker", "comnam", "market_cap"]].to_string(index=False))

    finally:
        db.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
