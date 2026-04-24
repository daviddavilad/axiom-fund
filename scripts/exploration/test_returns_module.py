"""Smoke test for the ReturnsPanel module.

Not formal tests — just runs through the main code paths against live WRDS.
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

from axiom_fund import _warnings  # noqa: F401
from axiom_fund.data.returns import PANEL_COLUMNS, ReturnsPanel


def main() -> int:
    load_dotenv()
    username = os.getenv("WRDS_USERNAME")
    if not username:
        print("ERROR: WRDS_USERNAME not set", file=sys.stderr)
        return 1

    import wrds
    db = wrds.Connection(wrds_username=username)

    try:
        rp = ReturnsPanel(db)

        # Case 1: liquid large-caps, no delistings
        print("=" * 70)
        print("Case 1: AAPL + MSFT, H1 2020 (no delistings expected)")
        print("=" * 70)
        panel1 = rp.fetch(
            permnos=[14593, 10107],
            start_date="2020-01-02",
            end_date="2020-06-30",
        )
        print(f"Rows: {len(panel1)}")
        print(f"Columns: {list(panel1.columns)}")
        print(f"Columns match PANEL_COLUMNS: {list(panel1.columns) == list(PANEL_COLUMNS)}")
        print(f"Delisting rows: {panel1['is_delisting'].sum()}")
        print()

        # Case 2: includes a name that delisted in January 2020
        print("=" * 70)
        print("Case 2: AAPL + PERMNO 16553 (delisted 2020-01-09)")
        print("=" * 70)
        panel2 = rp.fetch(
            permnos=[14593, 16553],
            start_date="2020-01-02",
            end_date="2020-06-30",
        )
        print(f"Rows: {len(panel2)}")
        print(f"Delisting rows: {panel2['is_delisting'].sum()}")
        print()
        print("Full trail for PERMNO 16553:")
        trail = panel2[panel2["permno"] == 16553]
        print(trail[["permno", "date", "ret", "is_delisting"]].to_string(index=False))
        print()

        # Case 3: to_wide
        print("=" * 70)
        print("Case 3: pivot panel1 to wide format")
        print("=" * 70)
        wide = ReturnsPanel.to_wide(panel1, values="ret")
        print(f"Wide shape: {wide.shape}")
        print(f"Index name: {wide.index.name}")
        print(f"Columns: {list(wide.columns)}")
        print("First 5 rows:")
        print(wide.head().to_string())
        print()

        # Case 4: to_wide on panel2 (has delisting duplicate)
        print("=" * 70)
        print("Case 4: pivot panel2 to wide — verify no duplicate date errors")
        print("=" * 70)
        wide2 = ReturnsPanel.to_wide(panel2, values="ret")
        print(f"Wide shape: {wide2.shape}")
        print(f"PERMNO 16553 column non-null count: {wide2[16553].notna().sum()}")
        print()

        # Case 5: empty permnos should raise
        print("=" * 70)
        print("Case 5: empty permnos raises ValueError")
        print("=" * 70)
        try:
            rp.fetch(permnos=[], start_date="2020-01-02", end_date="2020-06-30")
            print("ERROR: did not raise!")
        except ValueError as e:
            print(f"Correctly raised ValueError: {e}")

    finally:
        db.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
