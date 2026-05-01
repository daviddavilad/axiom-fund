"""Smoke test for IVol signal with orthogonal synthetic factors.

Naive randn factors have spurious cross-correlations of 0.1-0.2 in any
finite sample, which can mislead OLS coefficient recovery. This test
constructs factors via Gram-Schmidt orthogonalization so the regression
should recover the true betas precisely.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from axiom_fund.signals.idiosyncratic_volatility import (
    IVOL_RAW_COLUMNS,
    compute_idiosyncratic_volatility,
)


def main() -> int:
    np.random.seed(42)
    n_days = 200
    dates = pd.date_range("2020-01-02", periods=n_days, freq="B")

    # Build orthogonal factors via QR decomposition
    raw = np.random.normal(0, 1, size=(n_days, 3))
    q, _ = np.linalg.qr(raw)  # q is orthonormal columns

    # Scale to realistic factor std levels
    factors = q * np.array([0.012, 0.005, 0.006]) * np.sqrt(n_days)

    ff = pd.DataFrame(
        {
            "date": dates,
            "mktrf": factors[:, 0] + 0.0005,
            "smb": factors[:, 1] + 0.0001,
            "hml": factors[:, 2] + 0.0001,
            "rf": [0.00005] * n_days,
        }
    )

    print("=" * 70)
    print("Factor orthogonality check")
    print("=" * 70)
    corr = ff[["mktrf", "smb", "hml"]].corr()
    print(corr.round(4))
    print()

    # Synthetic stock returns
    true_alpha = 0.0001
    true_betas = (1.2, 0.3, -0.1)

    np.random.seed(100)
    eps_high = np.random.normal(0, 0.030, n_days)
    eps_low = np.random.normal(0, 0.005, n_days)

    excess_high = (
        true_alpha
        + true_betas[0] * ff["mktrf"]
        + true_betas[1] * ff["smb"]
        + true_betas[2] * ff["hml"]
        + eps_high
    )
    excess_low = (
        true_alpha
        + true_betas[0] * ff["mktrf"]
        + true_betas[1] * ff["smb"]
        + true_betas[2] * ff["hml"]
        + eps_low
    )

    rets = pd.DataFrame(
        {
            "permno": [1] * n_days + [2] * n_days,
            "date": list(dates) + list(dates),
            "ret": list(excess_high + ff["rf"]) + list(excess_low + ff["rf"]),
        }
    )

    print("=" * 70)
    print("Setup")
    print("=" * 70)
    print(f"{n_days} business days, 2 PERMNOs (high vs low IVol)")
    print(f"True alpha:        {true_alpha}")
    print(f"True (β_mkt, β_smb, β_hml):  {true_betas}")
    print("True residual std: 0.030 (high) / 0.005 (low)")
    print()

    signal = compute_idiosyncratic_volatility(
        returns_df=rets,
        ff_factors_df=ff,
        start_date="2020-01-02",
        end_date="2020-12-31",
    )

    print(f"Output rows: {len(signal)}")
    print(f"Columns match canonical: {tuple(signal.columns) == IVOL_RAW_COLUMNS}")
    print()

    print("=" * 70)
    print("Beta recovery (averaged across all rolling windows)")
    print("=" * 70)
    for permno in [1, 2]:
        rows = signal[signal["permno"] == permno]
        print(f"\n  PERMNO {permno} ({len(rows)} windows):")
        print(f"    beta_mkt mean: {rows['beta_mkt'].mean():.4f}  (true: 1.20)")
        print(f"    beta_smb mean: {rows['beta_smb'].mean():.4f}  (true: 0.30)")
        print(f"    beta_hml mean: {rows['beta_hml'].mean():.4f}  (true: -0.10)")
        print(f"    residual_std mean: {rows['residual_std'].mean():.4f}")
        print(f"    raw_signal mean:   {rows['raw_signal'].mean():.4f}")

    p1 = signal[signal["permno"] == 1]["raw_signal"].mean()
    p2 = signal[signal["permno"] == 2]["raw_signal"].mean()
    print()
    print("=" * 70)
    print("Cross-PERMNO ratio")
    print("=" * 70)
    print(f"PERMNO 1 mean: {p1:.4f}")
    print(f"PERMNO 2 mean: {p2:.4f}")
    print(f"Ratio: {p1 / p2:.2f}x  (expected: 6.0x = 0.030/0.005)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
