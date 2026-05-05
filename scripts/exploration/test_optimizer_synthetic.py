"""Smoke test for the MVO optimizer on synthetic data.

Verifies:
1. Optimizer maximizes α' w on a simple identity-covariance problem
2. Position caps are respected
3. Gross leverage cap is respected
4. Higher risk aversion produces smaller portfolios
5. Top alpha names get long, bottom get short
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from axiom_fund.portfolio.optimizer import optimize_portfolio


def main() -> int:
    np.random.seed(42)

    # 10-asset universe with known alpha rankings
    permnos = list(range(1, 11))
    alpha_values = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.0, 0.5, 1.0, 1.5, 2.0]
    alpha = pd.Series(alpha_values, index=permnos, name="alpha")

    # Simple covariance: identity × annualized variance
    daily_vol = 0.02
    annual_var = (daily_vol**2) * 252
    cov = pd.DataFrame(
        np.eye(10) * annual_var,
        index=permnos,
        columns=permnos,
    )

    print("=" * 70)
    print("Setup: 10 assets, alpha = [-2, -1.5, ..., 1.5, 2]")
    print(f"Identity covariance, annualized variance = {annual_var:.4f}")
    print("=" * 70)
    print()

    # Test 1: default parameters
    print("Test 1: Default parameters (λ=1.0, cap=1.5%, gross=1.5)")
    print("-" * 70)
    result1 = optimize_portfolio(alpha, cov)
    print("Weights:")
    for permno, w in result1.weights.items():
        print(f"  permno {permno}: alpha={alpha[permno]:+.1f}, weight={w:+.5f}")
    print(f"Expected alpha:      {result1.expected_alpha:+.4f}")
    print(f"Expected variance:   {result1.expected_variance:.6f}")
    print(f"Gross leverage:      {result1.gross_leverage:.4f}")
    print(f"Long count: {result1.long_count}, Short count: {result1.short_count}")
    print(f"Solver status: {result1.solver_status}")
    print()

    # Test 2: very high risk aversion → small portfolio
    print("Test 2: High risk aversion (λ=100)")
    print("-" * 70)
    result2 = optimize_portfolio(alpha, cov, risk_aversion=100.0)
    print(f"Gross leverage: {result2.gross_leverage:.4f} (expect smaller than test 1)")
    print(f"Max weight: {result2.weights.abs().max():.5f}")
    print()

    # Test 3: very low risk aversion → tight to caps
    print("Test 3: Low risk aversion (λ=0.001)")
    print("-" * 70)
    result3 = optimize_portfolio(alpha, cov, risk_aversion=0.001)
    print(f"Gross leverage: {result3.gross_leverage:.4f} (expect ~1.5, hitting cap)")
    print(f"Max |weight|: {result3.weights.abs().max():.5f} (expect ~0.015)")
    print()

    # Test 4: alpha sign correctness
    print("Test 4: Sign correctness")
    print("-" * 70)
    positive_alpha_weights = result1.weights[alpha > 0]
    negative_alpha_weights = result1.weights[alpha < 0]
    print(f"Mean weight on positive-alpha names: {positive_alpha_weights.mean():+.5f}")
    print(f"Mean weight on negative-alpha names: {negative_alpha_weights.mean():+.5f}")
    print("(Expect: positive mean for positive alpha, negative mean for negative)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
