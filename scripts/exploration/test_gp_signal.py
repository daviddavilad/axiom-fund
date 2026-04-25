"""Smoke test for the Gross Profitability signal with synthetic data."""

from __future__ import annotations

import pandas as pd

from axiom_fund.signals.gross_profitability import (
    GP_OUTPUT_COLUMNS,
    compute_gross_profitability,
)


def main() -> int:
    # Synthetic universe: 4 permnos, two snapshot dates
    universe = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2020-03-31"] * 4 + ["2020-06-30"] * 4
            ),
            "permno": [1, 2, 3, 4] * 2,
        }
    )

    # Synthetic fundamentals: 4 permnos, two report dates each
    # Permno 1: high GP firm
    # Permno 2: low GP firm
    # Permno 3: middle
    # Permno 4: outlier (will be winsorized)
    fundamentals = pd.DataFrame(
        {
            "permno": [1, 2, 3, 4] * 2,
            "gvkey": ["A", "B", "C", "D"] * 2,
            "rdq": pd.to_datetime(
                ["2020-04-30"] * 4 + ["2020-07-31"] * 4
            ),
            "datadate": pd.to_datetime(
                ["2020-03-31"] * 4 + ["2020-06-30"] * 4
            ),
            "revtq": [100.0, 50.0, 75.0, 200.0,
                      110.0, 55.0, 80.0, 220.0],
            "cogsq": [60.0, 45.0, 50.0, 50.0,
                      66.0, 50.0, 55.0, 55.0],
            "atq": [200.0, 200.0, 200.0, 200.0,
                    200.0, 200.0, 200.0, 200.0],
        }
    )
    # Compute expected raw GPs by hand:
    # Q1: [40/200, 5/200, 25/200, 150/200] = [0.20, 0.025, 0.125, 0.75]
    # Q2: [44/200, 5/200, 25/200, 165/200] = [0.22, 0.025, 0.125, 0.825]

    print("=" * 70)
    print("Synthetic input")
    print("=" * 70)
    print(f"Universe: {len(universe)} rows over 2 dates, 4 permnos each")
    print(f"Fundamentals: {len(fundamentals)} rows")
    print()

    # Run the signal
    signal = compute_gross_profitability(
        universe_df=universe,
        fundamentals_df=fundamentals,
        start_date="2020-04-01",
        end_date="2020-08-31",
        winsorize_pct=0.0,  # disable winsorization for clean check
    )

    print("=" * 70)
    print("Signal output (no winsorization)")
    print("=" * 70)
    print(f"Rows: {len(signal)}")
    print(f"Columns: {list(signal.columns)}")
    print(f"Match canonical: {tuple(signal.columns) == GP_OUTPUT_COLUMNS}")
    print()
    print(signal.to_string(index=False))
    print()

    # Verify Q1 cross-section z-scores
    q1 = signal[signal["date_filed"] == pd.Timestamp("2020-04-30")]
    print("Q1 raw signals:", q1["raw_signal"].tolist())
    print("Q1 mean:", q1["raw_signal"].mean())
    print("Q1 std:", q1["raw_signal"].std())
    print("Q1 z_scores sum to ~0:", abs(q1["z_score"].sum()) < 1e-10)

    # With winsorization, permno 4's outlier 0.75 should clip down
    print()
    print("=" * 70)
    print("With winsorization at 0.10 (10th/90th percentile)")
    print("=" * 70)
    signal_w = compute_gross_profitability(
        universe_df=universe,
        fundamentals_df=fundamentals,
        start_date="2020-04-01",
        end_date="2020-08-31",
        winsorize_pct=0.10,
    )
    q1_w = signal_w[signal_w["date_filed"] == pd.Timestamp("2020-04-30")]
    print(q1_w[["permno", "raw_signal", "winsorized", "z_score"]].to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
