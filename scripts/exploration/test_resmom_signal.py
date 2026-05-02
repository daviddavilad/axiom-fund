"""Smoke test: verify ResMom signal on synthetic data with known structure.

Build 30 stocks across 3 industries with known cross-sectional structure,
run signal, verify residuals correctly strip out industry and size effects.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from axiom_fund.signals.residual_momentum import (
    RESMOM_RAW_COLUMNS,
    compute_residual_momentum,
)


def main() -> int:
    np.random.seed(42)

    # 30 stocks across 3 industries, 24 months of daily returns
    n_stocks = 30
    n_months = 24
    industries = ["Tech", "Finance", "Health"]

    # Each stock has a fixed industry and size
    permnos = list(range(1, n_stocks + 1))
    stock_industries = [industries[i % 3] for i in range(n_stocks)]
    # Sizes: range from $1B to $1T
    sizes = np.exp(np.linspace(np.log(1e9), np.log(1e12), n_stocks))

    # Industry effects (Tech outperformed by 2%, Health by 1%, Finance flat)
    industry_effect = {"Tech": 0.020, "Finance": 0.0, "Health": 0.010}
    # Size effect: large caps -0.5% per ln(size)
    size_coef = -0.005

    # Build daily returns: each month, stock_return = industry + size + idiosyncratic
    rows = []
    fund_rows = []
    for permno_idx, permno in enumerate(permnos):
        ggroup = stock_industries[permno_idx]
        size = sizes[permno_idx]
        log_size = np.log(size)

        # Idiosyncratic momentum: half stocks have +1.5%/month, half -1.5%/month
        idio_perm = 0.015 if permno_idx % 2 == 0 else -0.015

        for month in range(n_months):
            year_month_start = pd.Timestamp("2020-01-01") + pd.DateOffset(months=month)
            month_end = year_month_start + pd.offsets.MonthEnd(0)
            n_days = (month_end - year_month_start).days + 1

            # Monthly return = industry + size_effect*log_size + idio + noise
            true_monthly = (
                industry_effect[ggroup]
                + size_coef * log_size
                + idio_perm
                + np.random.normal(0, 0.005)
            )
            # Spread over daily returns evenly (compounding handled by signal)
            daily_return = (1 + true_monthly) ** (1 / n_days) - 1

            for day_offset in range(n_days):
                d = year_month_start + pd.Timedelta(days=day_offset)
                rows.append(
                    {
                        "permno": permno,
                        "date": d,
                        "ret": daily_return,
                        "marketcap": size,
                    }
                )

        # Fundamentals: one annual filing per stock
        fund_rows.append(
            {
                "permno": permno,
                "rdq": pd.Timestamp("2019-12-31"),
                "datadate": pd.Timestamp("2019-09-30"),
                "ggroup": ggroup,
            }
        )

    rets = pd.DataFrame(rows)
    fund = pd.DataFrame(fund_rows)

    print("=" * 70)
    print("Synthetic input")
    print("=" * 70)
    print(f"{n_stocks} stocks × {n_months} months")
    print("3 industries: Tech (+2%), Health (+1%), Finance (0%)")
    print("Size effect: -0.5% per ln(size)")
    print("Idiosyncratic momentum: alternating +1.5% / -1.5% per month")
    print()

    signal = compute_residual_momentum(
        returns_df=rets,
        fundamentals_df=fund,
        start_date="2020-01-01",
        end_date="2021-12-31",
    )

    print("=" * 70)
    print("Signal output")
    print("=" * 70)
    print(f"Rows: {len(signal)}")
    print(f"Columns match canonical: {tuple(signal.columns) == RESMOM_RAW_COLUMNS}")
    print()

    if len(signal) == 0:
        print("WARNING: empty output — check date range and buffer")
        return 1

    print("Sample row:")
    print(signal.iloc[len(signal) // 2])
    print()

    # Sanity: even (winners) should have higher raw_signal than odd (losers)
    even_perm = signal[signal["permno"] % 2 == 0]["raw_signal"].mean()
    odd_perm = signal[signal["permno"] % 2 == 1]["raw_signal"].mean()
    print("=" * 70)
    print("Cross-permno momentum check")
    print("=" * 70)
    print(f"Even-permno (winners) mean raw_signal: {even_perm:.4f}")
    print(f"Odd-permno (losers)   mean raw_signal: {odd_perm:.4f}")
    print(f"Difference: {even_perm - odd_perm:.4f}")
    print("Expected: ~0.165 (= 11 months × 0.015 spread × 2)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
