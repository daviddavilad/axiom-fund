"""Smoke test for composite alpha module on synthetic aligned signal panels.

Verifies:
1. Sign convention is applied (IVol flip)
2. Missing signal handling (require at least 2 of 3)
3. Composite z-score has mean ~0, std ~1 cross-sectionally per date
4. Output columns match canonical schema
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from axiom_fund.portfolio.composite import (
    COMPOSITE_OUTPUT_COLUMNS,
    compute_composite_alpha,
)


def _make_aligned(
    permnos: list[int], z_scores: list[float], date: str = "2020-06-30"
) -> pd.DataFrame:
    """Build a synthetic aligned signal panel."""
    return pd.DataFrame(
        {
            "date": pd.to_datetime([date] * len(permnos)),
            "permno": permnos,
            "raw_signal": z_scores,  # simplified: pre-z-scored
            "winsorized": z_scores,
            "z_score": z_scores,
        }
    )


def main() -> int:
    # 5 stocks with controlled z-scores per signal
    permnos = [1, 2, 3, 4, 5]

    # Constructed so true composite z-score is recognizable
    # All three signals identical → composite equals each input
    z_gp = [-2.0, -1.0, 0.0, 1.0, 2.0]
    z_ivol = [2.0, 1.0, 0.0, -1.0, -2.0]   # OPPOSITE sign — high IVol = bad
    z_resmom = [-2.0, -1.0, 0.0, 1.0, 2.0]

    print("=" * 70)
    print("Test 1: All three signals fully populated, IVol pre-flipped")
    print("=" * 70)
    print("Each stock: z_gp = z_resmom = (after flip) z_ivol = same sign/value")
    print(f"  z_gp:     {z_gp}")
    print(f"  z_ivol:   {z_ivol}")
    print(f"  z_resmom: {z_resmom}")
    print()

    gp = _make_aligned(permnos, z_gp)
    ivol = _make_aligned(permnos, z_ivol)
    resmom = _make_aligned(permnos, z_resmom)

    result = compute_composite_alpha(gp, ivol, resmom)

    print(f"Output rows: {len(result)}")
    print(f"Columns: {list(result.columns)}")
    print(f"Match canonical: {tuple(result.columns) == COMPOSITE_OUTPUT_COLUMNS}")
    print()
    print(result.to_string(index=False))
    print()
    print(f"Composite_z mean: {result['composite_z'].mean():.6f}")
    print(f"Composite_z std:  {result['composite_z'].std():.6f}")
    print()

    # Test 2: missing GP for one stock
    print("=" * 70)
    print("Test 2: Stock 3 has missing GP — should still emit composite from 2 signals")
    print("=" * 70)
    z_gp_missing = [-2.0, -1.0, np.nan, 1.0, 2.0]
    gp2 = _make_aligned(permnos, z_gp_missing)
    result2 = compute_composite_alpha(gp2, ivol, resmom)
    print(result2.to_string(index=False))
    print()
    print(f"Stock 3 n_signals: {result2.loc[result2['permno'] == 3, 'n_signals'].iloc[0]}")
    print()

    # Test 3: stock with only one signal — should be dropped
    print("=" * 70)
    print("Test 3: Stock 3 has only ResMom (GP and IVol both NaN) — should be dropped")
    print("=" * 70)
    z_gp3 = [-2.0, -1.0, np.nan, 1.0, 2.0]
    z_ivol3 = [2.0, 1.0, np.nan, -1.0, -2.0]
    gp3 = _make_aligned(permnos, z_gp3)
    ivol3 = _make_aligned(permnos, z_ivol3)
    result3 = compute_composite_alpha(gp3, ivol3, resmom)
    print(f"Output rows: {len(result3)} (expected 4, stock 3 dropped)")
    print(f"Permnos in output: {sorted(result3['permno'].tolist())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
