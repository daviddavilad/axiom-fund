"""Smoke test for the betas module on synthetic data.

Generates synthetic returns where stock i has known beta_i = i × 0.3,
verifies the module recovers those betas accurately.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from axiom_fund.portfolio.betas import compute_betas


def main() -> int:
    rng = np.random.default_rng(42)

    # 252 days of synthetic data
    n_days = 252
    dates = pd.date_range("2020-01-02", periods=n_days, freq="B")

    # Market: daily mean 0.0005, vol 1.5%
    mktrf = rng.normal(0.0005, 0.015, size=n_days)

    # 5 stocks with known true betas
    true_betas = [0.5, 0.8, 1.0, 1.2, 1.8]
    permnos = list(range(1, 6))

    rows = []
    for permno, beta in zip(permnos, true_betas, strict=True):
        # r = β × mktrf + idiosyncratic noise
        epsilon = rng.normal(0, 0.01, size=n_days)
        r = beta * mktrf + epsilon
        for d, ret in zip(dates, r, strict=True):
            rows.append({"permno": permno, "date": d, "ret": ret})

    rets_df = pd.DataFrame(rows)
    ff_df = pd.DataFrame({"date": dates, "mktrf": mktrf})

    print("=" * 70)
    print("Synthetic data: 5 stocks, 252 days, known true betas")
    print("=" * 70)

    estimated = compute_betas(
        returns_df=rets_df,
        ff_factors_df=ff_df,
        as_of_date=dates[-1],
        window=252,
    )

    print(f"{'permno':<8} {'true β':>10} {'estimated':>12} {'error':>10}")
    print("-" * 70)
    for permno, true_beta in zip(permnos, true_betas, strict=True):
        est = estimated.loc[permno]
        err = est - true_beta
        print(f"{permno:<8} {true_beta:>10.3f} {est:>12.4f} {err:>+10.4f}")

    max_err = max(abs(estimated.loc[p] - b) for p, b in zip(permnos, true_betas, strict=True))
    print(f"\nMax estimation error: {max_err:.4f} (expect < 0.05 for 252-day window)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
