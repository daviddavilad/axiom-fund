"""Tests for the quintile_sort backtest primitives."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from axiom_fund.backtest.quintile_sort import (
    LS_RETURN_COLUMNS,
    assign_quintiles,
    compute_long_short_returns,
)


# ============================================================================
# Helpers
# ============================================================================


def _aligned_signal(
    date: str,
    permnos: list[int],
    z_scores: list[float],
) -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.to_datetime([date] * len(permnos)),
        "permno": permnos,
        "z_score": z_scores,
    })


def _forward_returns(
    date: str,
    permnos: list[int],
    returns: list[float],
) -> pd.DataFrame:
    return pd.DataFrame({
        "rebalance_date": pd.to_datetime([date] * len(permnos)),
        "permno": permnos,
        "fwd_return": returns,
    })


# ============================================================================
# assign_quintiles
# ============================================================================


class TestAssignQuintiles:
    def test_basic_five_by_five(self):
        # 25 firms with z_scores 1..25 -> 5 firms per quintile
        permnos = list(range(100, 125))
        z_scores = list(range(1, 26))
        aligned = _aligned_signal("2020-06-30", permnos, [float(z) for z in z_scores])
        result = assign_quintiles(aligned, n_quintiles=5)
        assert len(result) == 25
        # Bottom 5 z_scores (permnos 100-104) should be quintile 1
        bottom = result[result.permno.between(100, 104)]
        assert (bottom.quintile == 1).all()
        # Top 5 z_scores (permnos 120-124) should be quintile 5
        top = result[result.permno.between(120, 124)]
        assert (top.quintile == 5).all()

    def test_nan_z_scores_get_nan_quintile(self):
        permnos = list(range(100, 110))
        z_scores = [1.0, 2.0, 3.0, 4.0, 5.0, np.nan, np.nan, 6.0, 7.0, 8.0]
        aligned = _aligned_signal("2020-06-30", permnos, z_scores)
        result = assign_quintiles(aligned, n_quintiles=5)
        nan_rows = result[result.permno.isin([105, 106])]
        assert nan_rows.quintile.isna().all()
        non_nan = result[~result.permno.isin([105, 106])]
        assert non_nan.quintile.notna().all()

    def test_multi_date_independent(self):
        # Same permnos on two dates with different z_score orderings
        d1 = _aligned_signal(
            "2020-06-30",
            [100, 101, 102, 103, 104],
            [1.0, 2.0, 3.0, 4.0, 5.0],
        )
        d2 = _aligned_signal(
            "2020-07-31",
            [100, 101, 102, 103, 104],
            [5.0, 4.0, 3.0, 2.0, 1.0],  # reversed
        )
        aligned = pd.concat([d1, d2])
        result = assign_quintiles(aligned, n_quintiles=5)
        # On 2020-06-30, permno 100 (z=1) should be quintile 1
        # On 2020-07-31, permno 100 (z=5) should be quintile 5
        p100_d1 = result[(result.date == "2020-06-30") & (result.permno == 100)]
        p100_d2 = result[(result.date == "2020-07-31") & (result.permno == 100)]
        assert p100_d1.iloc[0].quintile == 1
        assert p100_d2.iloc[0].quintile == 5

    def test_missing_columns_raises(self):
        df = pd.DataFrame({"date": ["2020-06-30"], "permno": [100]})
        with pytest.raises(ValueError, match="missing columns"):
            assign_quintiles(df)

    def test_empty_input_returns_empty(self):
        empty = pd.DataFrame(columns=["date", "permno", "z_score"])
        result = assign_quintiles(empty)
        assert list(result.columns) == ["date", "permno", "quintile"]
        assert len(result) == 0


# ============================================================================
# compute_long_short_returns
# ============================================================================


class TestComputeLongShort:
    def test_ls_synthetic(self):
        # 10 firms on one date; top 2 quintile 5 return 10%, bottom 2 quintile 1 return 2%
        permnos = list(range(100, 110))
        z_scores = [float(z) for z in range(1, 11)]
        aligned = _aligned_signal("2020-06-30", permnos, z_scores)
        quintiles = assign_quintiles(aligned, n_quintiles=5)
        # Assign returns: permnos 100-101 (short) at 2%, 108-109 (long) at 10%, rest at 5%
        returns_by_permno = {p: 0.05 for p in permnos}
        returns_by_permno[100] = 0.02
        returns_by_permno[101] = 0.02
        returns_by_permno[108] = 0.10
        returns_by_permno[109] = 0.10
        fwd = _forward_returns(
            "2020-06-30",
            permnos,
            [returns_by_permno[p] for p in permnos],
        )
        result = compute_long_short_returns(quintiles, fwd, n_quintiles=5)
        assert len(result) == 1
        row = result.iloc[0]
        assert row.long_return == pytest.approx(0.10)
        assert row.short_return == pytest.approx(0.02)
        assert row.ls_return == pytest.approx(0.08)
        assert row.n_long == 2
        assert row.n_short == 2

    def test_missing_forward_return_excluded(self):
        # Firm in quintile 5 but no forward return -> not counted in long_return
        permnos = list(range(100, 110))
        z_scores = [float(z) for z in range(1, 11)]
        aligned = _aligned_signal("2020-06-30", permnos, z_scores)
        quintiles = assign_quintiles(aligned, n_quintiles=5)
        # Drop forward return for permno 109 (quintile 5)
        fwd = _forward_returns(
            "2020-06-30",
            [p for p in permnos if p != 109],
            [0.05] * 9,
        )
        result = compute_long_short_returns(quintiles, fwd, n_quintiles=5)
        assert result.iloc[0].n_long == 1  # only permno 108 had return

    def test_empty_quintile_returns_nan(self):
        # All permnos assigned to middle quintiles; long/short empty
        # Contrive by feeding tiny cross-section where qcut can't form 5 buckets
        permnos = [100, 101, 102]
        z_scores = [1.0, 1.0, 1.0]  # all identical -> qcut fails -> all NaN
        aligned = _aligned_signal("2020-06-30", permnos, z_scores)
        quintiles = assign_quintiles(aligned, n_quintiles=5)
        fwd = _forward_returns("2020-06-30", permnos, [0.05, 0.05, 0.05])
        result = compute_long_short_returns(quintiles, fwd, n_quintiles=5)
        # All NaN quintiles -> no long/short leg
        assert result.iloc[0].n_long == 0
        assert result.iloc[0].n_short == 0
        assert pd.isna(result.iloc[0].long_return)

    def test_missing_columns_raises(self):
        bad_q = pd.DataFrame({"date": ["2020-06-30"], "permno": [100]})
        fwd = _forward_returns("2020-06-30", [100], [0.05])
        with pytest.raises(ValueError, match="quintiles_df missing"):
            compute_long_short_returns(bad_q, fwd)

    def test_columns_match_locked_schema(self):
        permnos = list(range(100, 110))
        aligned = _aligned_signal("2020-06-30", permnos, [float(z) for z in range(1, 11)])
        quintiles = assign_quintiles(aligned, n_quintiles=5)
        fwd = _forward_returns("2020-06-30", permnos, [0.05] * 10)
        result = compute_long_short_returns(quintiles, fwd)
        assert tuple(result.columns) == LS_RETURN_COLUMNS