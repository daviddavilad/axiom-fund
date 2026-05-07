"""Smoke test for the single-period backtest engine on synthetic data.

Verifies:
1. Engine runs end-to-end with synthetic inputs
2. Realized return matches the expected buy-and-hold computation
3. Hand-controllable case: known weights × known returns → known P&L
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from axiom_fund.backtest.engine import (
    BacktestPeriodInputs,
    _compute_buy_and_hold_returns,
    run_backtest_period,
)


def main() -> int:
    rng = np.random.default_rng(42)
    n_assets = 30
    permnos = list(range(1, n_assets + 1))
    rebalance = pd.Timestamp("2020-09-30")
    holding_days = 21
    holding_dates = pd.bdate_range(
        start=rebalance + pd.Timedelta(days=1), periods=holding_days * 2
    )

    # Test 1: hand-controllable buy-and-hold math
    print("=" * 70)
    print("Test 1: hand-controllable BH math")
    print("=" * 70)
    # 3 stocks, simple known returns: stock A doubles, stock B stays flat,
    # stock C drops 50%
    weights = pd.Series([0.5, -0.5, 0.5], index=[1, 2, 3])
    # Use actual returns: 21 days each
    n = 21
    rets = pd.DataFrame({
        1: [(2.0 ** (1/n)) - 1] * n,    # geometric: 21 days compound to +100%
        2: [0.0] * n,                    # flat
        3: [(0.5 ** (1/n)) - 1] * n,    # geometric: 21 days compound to -50%
    }, index=holding_dates[:n])
    bh = _compute_buy_and_hold_returns(rets, weights, n)
    expected = 0.5 * 1.0 + (-0.5) * 0.0 + 0.5 * (-0.5)
    print(f"  Expected: 0.5×(+100%) + (-0.5)×0% + 0.5×(-50%) = {expected:+.4f}")
    print(f"  Computed: {bh:+.4f}")
    print(f"  Match: {abs(bh - expected) < 1e-10}")
    print()

    # Test 2: full engine end-to-end with synthetic data
    print("=" * 70)
    print("Test 2: full engine with synthetic inputs")
    print("=" * 70)
    alpha = pd.Series(
        rng.normal(0, 1, size=n_assets), index=permnos, name="composite_z"
    )
    cov = pd.DataFrame(
        np.eye(n_assets) * 0.06,  # ~25% ann vol diagonal
        index=permnos, columns=permnos,
    )
    betas = pd.Series(
        rng.uniform(0.5, 1.5, size=n_assets), index=permnos
    )
    sectors = pd.Series(
        ([10] * 10 + [20] * 10 + [30] * 10), index=permnos
    )
    # Synthetic future returns: random daily returns, vol ~ 1.5%
    future_rets = pd.DataFrame(
        rng.normal(0.0005, 0.015, size=(holding_days * 2, n_assets)),
        index=holding_dates,
        columns=permnos,
    )

    inputs = BacktestPeriodInputs(
        rebalance_date=rebalance,
        alpha=alpha,
        covariance=cov,
        betas=betas,
        sectors=sectors,
        holding_period_returns=future_rets,
    )

    result = run_backtest_period(inputs)

    print(f"Rebalance date:       {result.rebalance_date.strftime('%Y-%m-%d')}")
    print(f"Holding period end:   {result.holding_period_end.strftime('%Y-%m-%d')}")
    print(f"N names:              {result.n_names}")
    print(f"Long / Short:         {result.long_count} / {result.short_count}")
    print(f"Gross leverage:       {result.gross_leverage:.4f}")
    print(f"Net exposure:         {result.net_exposure:+.6e}")
    print(f"Portfolio beta:       {result.portfolio_beta:+.6e}")
    print(f"Optimizer status:     {result.optimizer_status}")
    print(f"Realized return:      {result.realized_return:+.4f}")
    print()

    # Sanity check: verify realized return matches manual computation
    expected_bh = _compute_buy_and_hold_returns(
        future_rets, result.weights, holding_days,
    )
    print(f"Manual BH check: {expected_bh:+.6f}")
    print(f"Engine output:   {result.realized_return:+.6f}")
    print(f"Match: {abs(expected_bh - result.realized_return) < 1e-10}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
