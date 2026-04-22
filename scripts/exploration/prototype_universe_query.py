"""Prototype the universe construction query against CRSP.

Iterative script to develop and validate the SQL query that returns
the top-N rules-based universe on a given date.

Not production code - will be replaced by src/axiom_fund/data/universe.py.
"""

from __future__ import annotations

import os
import sys

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import text

from axiom_fund import _warnings  # noqa: F401

TARGET_DATE = "2020-01-02"
UNIVERSE_SIZE = 1000
PRICE_FLOOR = 5.0
ADV_FLOOR = 5_000_000.0  # $5M
ADV_WINDOW_DAYS = 20


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
    # Step A: Eligible PERMNOs on the target date based on stocknames.
    # A PERMNO is eligible if, for the target_date:
    #   - namedt <= target_date <= nameenddt
    #   - shrcd IN (10, 11)         -- common stock
    #   - exchcd IN (1, 2, 3)       -- NYSE, AMEX, NASDAQ
    #   - siccd != 6798             -- not REITs
    # ------------------------------------------------------------------
    print("=" * 70)
    print(f"Step A: Eligible PERMNOs on {TARGET_DATE}")
    print("=" * 70)
    eligible = query(
        db,
        """
        SELECT
            permno,
            shrcd,
            exchcd,
            siccd,
            ticker,
            comnam
        FROM crsp.stocknames
        WHERE namedt <= :d AND nameenddt >= :d
          AND shrcd IN (10, 11)
          AND exchcd IN (1, 2, 3)
          AND siccd != 6798
        """,
        {"d": TARGET_DATE},
    )
    print(f"Eligible PERMNOs (rules-based): {len(eligible):,}")
    print(f"Share code distribution:\n{eligible['shrcd'].value_counts().to_string()}")
    print(f"Exchange distribution:\n{eligible['exchcd'].value_counts().to_string()}")
    print(f"\nSample:\n{eligible.head(5).to_string(index=False)}")
    print()

    # ------------------------------------------------------------------
    # Step B: Join eligible PERMNOs with DSF price and shrout on target date.
    # Add price floor filter.
    # ------------------------------------------------------------------
    print("=" * 70)
    print(f"Step B: Add price and shrout, filter price > ${PRICE_FLOOR}")
    print("=" * 70)
    priced = query(
        db,
        """
        SELECT
            sn.permno,
            sn.shrcd,
            sn.exchcd,
            sn.siccd,
            sn.ticker,
            sn.comnam,
            dsf.prc,
            dsf.shrout,
            ABS(dsf.prc) * dsf.shrout * 1000 AS market_cap
        FROM crsp.stocknames sn
        INNER JOIN crsp.dsf dsf
            ON dsf.permno = sn.permno
            AND dsf.date = :d
        WHERE sn.namedt <= :d AND sn.nameenddt >= :d
          AND sn.shrcd IN (10, 11)
          AND sn.exchcd IN (1, 2, 3)
          AND sn.siccd != 6798
          AND ABS(dsf.prc) > :price_floor
          AND dsf.shrout IS NOT NULL
        """,
        {"d": TARGET_DATE, "price_floor": PRICE_FLOOR},
    )
    print(f"After price filter: {len(priced):,} PERMNOs")
    print(f"Market cap range: ${priced['market_cap'].min():,.0f} to ${priced['market_cap'].max():,.0f}")
    print(f"Market cap median: ${priced['market_cap'].median():,.0f}")
    print()

    # ------------------------------------------------------------------
    # Step C: Compute 20-day ADV as of target date, filter, then rank and select top N.
    # ADV = trailing 20-day average of (abs(prc) * vol).
    # ------------------------------------------------------------------
    print("=" * 70)
    print(f"Step C: Add 20-day ADV, filter > ${ADV_FLOOR:,.0f}, rank top {UNIVERSE_SIZE}")
    print("=" * 70)

    universe = query(
        db,
        """
        WITH adv AS (
            SELECT
                permno,
                AVG(ABS(prc) * vol) AS adv_20d
            FROM crsp.dsf
            WHERE date <= :d
            AND date >= (CAST(:d AS DATE) - INTERVAL '40 days')
              AND prc IS NOT NULL
              AND vol IS NOT NULL
            GROUP BY permno
        ),
        eligible AS (
            SELECT
                sn.permno,
                sn.shrcd,
                sn.exchcd,
                sn.siccd,
                sn.ticker,
                sn.comnam,
                dsf.prc,
                dsf.shrout,
                ABS(dsf.prc) * dsf.shrout * 1000 AS market_cap,
                adv.adv_20d
            FROM crsp.stocknames sn
            INNER JOIN crsp.dsf dsf
                ON dsf.permno = sn.permno
                AND dsf.date = :d
            INNER JOIN adv
                ON adv.permno = sn.permno
            WHERE sn.namedt <= :d AND sn.nameenddt >= :d
              AND sn.shrcd IN (10, 11)
              AND sn.exchcd IN (1, 2, 3)
              AND sn.siccd != 6798
              AND ABS(dsf.prc) > :price_floor
              AND dsf.shrout IS NOT NULL
              AND adv.adv_20d > :adv_floor
        )
        SELECT *
        FROM eligible
        ORDER BY market_cap DESC
        LIMIT :n;
        """,
        {
            "d": TARGET_DATE,
            "price_floor": PRICE_FLOOR,
            "adv_floor": ADV_FLOOR,
            "n": UNIVERSE_SIZE,
        },
    )

    print(f"Universe size: {len(universe):,}")
    if len(universe) > 0:
        print("\nTop 10 by market cap:")
        print(universe.head(10)[["permno", "ticker", "comnam", "market_cap", "adv_20d"]].to_string(index=False))
        print("\nBottom 5 of universe:")
        print(universe.tail(5)[["permno", "ticker", "comnam", "market_cap", "adv_20d"]].to_string(index=False))
        print("\nMarket cap distribution (percentiles):")
        print(universe["market_cap"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).to_string())
        print(f"\nExchange distribution:\n{universe['exchcd'].value_counts().to_string()}")

    db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
