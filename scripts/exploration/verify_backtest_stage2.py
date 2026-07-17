"""Stage 2 backtest verification: period + quintile breakdown.

  1. L/S Sharpe per year — which year drives the negative aggregate?
  2. Per-quintile Sharpe (Q1..Q5) — is the ranking monotonic?
     Real CMN signal should show Q1 < Q2 < Q3 < Q4 < Q5 in returns.
  3. Coverage per rebalance date — how many firms are in each quintile
     over time? If early sample is dominated by SPACs, that's period bias.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


CACHE = Path("data/cache")
BACKTEST = CACHE / "lazy_prices_backtest"


def sharpe(x: pd.Series) -> float:
    """Annualized Sharpe from monthly returns."""
    x = x.dropna()
    if len(x) < 3 or x.std() < 1e-12:
        return float("nan")
    return float(x.mean() / x.std() * np.sqrt(12))


def main() -> None:
    monthly = pd.read_parquet(BACKTEST / "monthly_returns.parquet")
    positions = pd.read_parquet(BACKTEST / "quintile_positions.parquet")
    returns = pd.read_parquet(CACHE / "lazy_prices_returns_daily.parquet")

    # -------------------------------------------------------------------
    # 1. L/S per year
    # -------------------------------------------------------------------
    monthly = monthly.copy()
    monthly["year"] = monthly.date.dt.year
    print("=" * 70)
    print("L/S Sharpe per year (annualized)")
    print("=" * 70)
    for year, group in monthly.groupby("year"):
        n = group.ls_return.notna().sum()
        mean_ret = group.ls_return.mean()
        vol = group.ls_return.std()
        sh = sharpe(group.ls_return)
        print(f"  {year}: N={n:2d}  mean={mean_ret:+.4f}  vol={vol:.4f}  "
              f"Sharpe={sh:+.3f}")

    total_sharpe = sharpe(monthly.ls_return)
    print(f"\n  Full sample Sharpe: {total_sharpe:+.3f}")

    # -------------------------------------------------------------------
    # 2. Per-quintile monthly returns (Q1, Q2, Q3, Q4, Q5)
    # -------------------------------------------------------------------
    # Need forward returns joined with quintile positions
    from axiom_fund.backtest.forward_returns import compute_forward_returns

    rebalance_dates = sorted(monthly.date.unique())
    fwd = compute_forward_returns(
        returns_df=returns,
        rebalance_dates=list(pd.to_datetime(rebalance_dates)),
        holding_days=21,
    )
    fwd = fwd.rename(columns={"rebalance_date": "date"})

    merged = positions.merge(fwd, on=["date", "permno"], how="inner")

    print()
    print("=" * 70)
    print("Per-quintile monthly return Sharpe (equal-weighted within quintile)")
    print("=" * 70)
    quintile_returns = merged.groupby(["date", "quintile"]).fwd_return.mean().reset_index()
    for q in [1.0, 2.0, 3.0, 4.0, 5.0]:
        sub = quintile_returns[quintile_returns.quintile == q].fwd_return
        n = sub.notna().sum()
        mean = sub.mean()
        vol = sub.std()
        sh = sharpe(sub)
        print(f"  Q{int(q)}: N={n:2d}  mean={mean:+.4f}  vol={vol:.4f}  Sharpe={sh:+.3f}")

    print()
    print("Real CMN pattern: Q5 highest > Q4 > Q3 > Q2 > Q1 lowest")
    print("If Q1 > Q5: sign is flipped (lazy firms outperform)")
    print("If non-monotonic: no clear signal, just noise")

    # -------------------------------------------------------------------
    # 3. Coverage per year — how many firms in each quintile over time?
    # -------------------------------------------------------------------
    print()
    print("=" * 70)
    print("Cross-section size per year (avg firms per quintile per month)")
    print("=" * 70)
    positions_year = positions.copy()
    positions_year["year"] = positions_year.date.dt.year
    per_year = (
        positions_year[positions_year.quintile.notna()]
        .groupby(["year", "quintile"])
        .size()
        .div(positions_year.groupby("year").date.nunique())
        .round(0)
        .astype(int)
        .unstack()
    )
    print(per_year.to_string())


if __name__ == "__main__":
    main()