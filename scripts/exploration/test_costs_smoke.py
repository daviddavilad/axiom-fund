"""Smoke test for the transaction cost module.

Verifies the four cost components (commission, spread, impact, borrow)
behave correctly on hand-controllable synthetic data. Runs before
formal tests are written to catch obvious math errors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from axiom_fund.backtest.costs import (
    compute_market_impact_bps,
    compute_per_name_trade_cost,
    compute_period_cost,
    compute_short_borrow_dollars,
    estimate_cs_spread_daily,
    estimate_cs_spread_rolling,
)


def test_cs_spread() -> None:
    """Corwin-Schultz on constant H/L should approach zero spread."""
    print("=" * 70)
    print("Test 1: Corwin-Schultz spread on synthetic")
    print("=" * 70)
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    high = pd.Series([101.0] * 30, index=dates)
    low = pd.Series([99.0] * 30, index=dates)
    daily = estimate_cs_spread_daily(high, low)
    rolling = estimate_cs_spread_rolling(high, low, window=20)
    print(f"  CS daily mean (constant H/L = 101/99): {daily.mean() * 10000:.2f} bps")
    print(f"  CS rolling[-1] (20d):                  {rolling.iloc[-1] * 10000:.2f} bps")
    print("  Expected ≈ 0 bps (no overnight movement)")
    print()


def test_cs_spread_with_overnight() -> None:
    """Stocks with overnight jumps should produce nonzero spread."""
    print("=" * 70)
    print("Test 1b: Corwin-Schultz with overnight jumps")
    print("=" * 70)
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    # Daily 2% range, but overnight jumps cause 2-day H/L to be wider
    np.random.seed(42)
    base = 100 + np.cumsum(np.random.normal(0, 0.5, 30))
    high = pd.Series(base * 1.01, index=dates)
    low = pd.Series(base * 0.99, index=dates)
    daily = estimate_cs_spread_daily(high, low)
    rolling = estimate_cs_spread_rolling(high, low, window=20)
    print(f"  CS daily mean (with jumps):  {daily.mean() * 10000:.2f} bps")
    print(f"  CS rolling[-1] (20d):        {rolling.iloc[-1] * 10000:.2f} bps")
    print("  Expected > 0 bps (overnight movement implies spread)")
    print()


def test_market_impact() -> None:
    """Square-root impact: 1% of ADV at 2% vol → expected ~20 bps."""
    print("=" * 70)
    print("Test 2: Market impact (square-root model)")
    print("=" * 70)
    impact = compute_market_impact_bps(
        trade_dollars=1_000_000,
        adv_dollars=100_000_000,
        sigma_daily=0.02,
        kappa=0.1,
    )
    expected = 0.1 * 0.02 * np.sqrt(0.01) * 10000
    print(f"  1% of ADV, 2% daily vol, kappa=0.1: {impact:.2f} bps")
    print(f"  Expected (closed form):              {expected:.2f} bps")
    print(f"  Match: {abs(impact - expected) < 0.01}")

    # Scaling check: 4x trade size → 2x impact (sqrt)
    impact_4x = compute_market_impact_bps(
        trade_dollars=4_000_000,
        adv_dollars=100_000_000,
        sigma_daily=0.02,
        kappa=0.1,
    )
    print(f"  4x trade size (4% of ADV):           {impact_4x:.2f} bps")
    print(f"  Ratio impact_4x / impact_1x:         {impact_4x / impact:.2f} (expected 2.0)")
    print()


def test_short_borrow() -> None:
    """Short borrow: $1M short at 50bps/yr monthly → $416.67."""
    print("=" * 70)
    print("Test 3: Short borrow drag")
    print("=" * 70)
    borrow_short = compute_short_borrow_dollars(-1_000_000, 0.005, 12)
    expected = 1_000_000 * 0.005 / 12
    print(f"  $1M short, 50bps/yr, monthly: ${borrow_short:.2f}")
    print(f"  Expected:                     ${expected:.2f}")

    borrow_long = compute_short_borrow_dollars(1_000_000, 0.005, 12)
    print(f"  $1M LONG (should be zero):    ${borrow_long:.2f}")
    print()


def test_per_name_cost() -> None:
    """Single-name cost combining all components."""
    print("=" * 70)
    print("Test 4: Per-name cost breakdown")
    print("=" * 70)
    result = compute_per_name_trade_cost(
        permno=12345,
        trade_dollars=500_000,
        spread_bps_estimate=20.0,    # 20 bps effective spread
        adv_dollars=50_000_000,       # 1% of ADV
        sigma_daily=0.025,            # 2.5% daily vol
        new_position_dollars=-500_000,  # short
    )
    print(f"  Trade size:   ${result.trade_dollars:,.0f}")
    print(f"  Commission:   ${result.commission_dollars:.2f}")
    print(f"  Spread:       ${result.spread_dollars:.2f}")
    print(f"  Impact:       ${result.impact_dollars:.2f}")
    print(f"  Short borrow: ${result.short_borrow_dollars:.2f}")
    print(f"  TOTAL:        ${result.total_dollars:.2f}")
    print()
    print(f"  As fraction of trade size: "
          f"{result.total_dollars / result.trade_dollars * 10000:.2f} bps")
    print()


def test_period_cost() -> None:
    """Full period rebalance cost across 4 names."""
    print("=" * 70)
    print("Test 5: Period cost (4-name rebalance)")
    print("=" * 70)
    # Reposition portfolio: keep 1001, flip 1002 short→long, flip 1003 long→short,
    # keep 1004 short
    weights_old = pd.Series({1001: 0.005, 1002: -0.005, 1003: 0.005, 1004: -0.005})
    weights_new = pd.Series({1001: 0.005, 1002: 0.005, 1003: -0.005, 1004: -0.005})
    spread_bps = pd.Series({1001: 10, 1002: 15, 1003: 20, 1004: 25})
    adv_dollars = pd.Series({1001: 100e6, 1002: 50e6, 1003: 30e6, 1004: 20e6})
    sigma_daily = pd.Series({1001: 0.015, 1002: 0.02, 1003: 0.025, 1004: 0.03})

    result = compute_period_cost(
        period_date=pd.Timestamp("2024-01-31"),
        weights_old=weights_old,
        weights_new=weights_new,
        spread_bps=spread_bps,
        adv_dollars=adv_dollars,
        sigma_daily=sigma_daily,
        nav=10_000_000,
    )
    print(f"  Period:          {result.period_date.date()}")
    print(f"  NAV:             ${result.nav:,.0f}")
    print(f"  N trades:        {result.n_trades}")
    print(f"  Commission:      {result.commission_bps:.2f} bps")
    print(f"  Spread:          {result.spread_bps:.2f} bps")
    print(f"  Impact:          {result.impact_bps:.2f} bps")
    print(f"  Short borrow:    {result.short_borrow_bps:.2f} bps")
    print(f"  Total:           {result.total_bps:.2f} bps")
    print(f"  Return drag:     {result.total_return_drag * 100:.4f}%")
    print()
    print("  Note: 1001 has no trade, but should still incur 0 cost.")
    print("        1004 has no trade but IS short → borrow cost only.")
    print()


def main() -> int:
    test_cs_spread()
    test_cs_spread_with_overnight()
    test_market_impact()
    test_short_borrow()
    test_per_name_cost()
    test_period_cost()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
