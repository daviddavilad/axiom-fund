"""Tests for the composite alpha module.

Pure-function module — all tests are unit tests with synthetic data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from axiom_fund.portfolio.composite import (
    COMPOSITE_OUTPUT_COLUMNS,
    compute_composite_alpha,
)

# ============================================================================
# Helpers
# ============================================================================


def _make_aligned(
    permnos: list[int], z_scores: list[float], date: str = "2020-06-30"
) -> pd.DataFrame:
    """Build a synthetic aligned signal panel."""
    return pd.DataFrame(
        {
            "date": pd.to_datetime([date] * len(permnos)),
            "permno": permnos,
            "raw_signal": z_scores,
            "winsorized": z_scores,
            "z_score": z_scores,
        }
    )


# ============================================================================
# Input validation
# ============================================================================


class TestInputValidation:
    def test_min_signals_invalid_raises(self) -> None:
        gp = _make_aligned([1, 2], [0.5, -0.5])
        ivol = _make_aligned([1, 2], [0.5, -0.5])
        resmom = _make_aligned([1, 2], [0.5, -0.5])
        with pytest.raises(ValueError, match="min_signals"):
            compute_composite_alpha(gp, ivol, resmom, min_signals=0)
        with pytest.raises(ValueError, match="min_signals"):
            compute_composite_alpha(gp, ivol, resmom, min_signals=4)

    def test_missing_columns_in_gp_raises(self) -> None:
        bad = pd.DataFrame({"foo": [1]})
        ivol = _make_aligned([1], [0.5])
        resmom = _make_aligned([1], [0.5])
        with pytest.raises(ValueError, match="aligned_gp"):
            compute_composite_alpha(bad, ivol, resmom)

    def test_missing_columns_in_ivol_raises(self) -> None:
        gp = _make_aligned([1], [0.5])
        bad = pd.DataFrame({"foo": [1]})
        resmom = _make_aligned([1], [0.5])
        with pytest.raises(ValueError, match="aligned_ivol"):
            compute_composite_alpha(gp, bad, resmom)

    def test_missing_columns_in_resmom_raises(self) -> None:
        gp = _make_aligned([1], [0.5])
        ivol = _make_aligned([1], [0.5])
        bad = pd.DataFrame({"foo": [1]})
        with pytest.raises(ValueError, match="aligned_resmom"):
            compute_composite_alpha(gp, ivol, bad)


# ============================================================================
# Output schema
# ============================================================================


class TestOutputSchema:
    def test_columns_match_canonical(self) -> None:
        gp = _make_aligned([1, 2, 3], [-1.0, 0.0, 1.0])
        ivol = _make_aligned([1, 2, 3], [1.0, 0.0, -1.0])
        resmom = _make_aligned([1, 2, 3], [-1.0, 0.0, 1.0])
        result = compute_composite_alpha(gp, ivol, resmom)
        assert tuple(result.columns) == COMPOSITE_OUTPUT_COLUMNS

    def test_empty_input_yields_empty_with_canonical_columns(self) -> None:
        empty = pd.DataFrame(columns=["date", "permno", "z_score"])
        result = compute_composite_alpha(empty, empty, empty)
        assert len(result) == 0
        assert list(result.columns) == list(COMPOSITE_OUTPUT_COLUMNS)

    def test_sorted_by_date_then_permno(self) -> None:
        gp = _make_aligned([3, 1, 2], [-1.0, 1.0, 0.0])
        ivol = _make_aligned([3, 1, 2], [1.0, -1.0, 0.0])
        resmom = _make_aligned([3, 1, 2], [-1.0, 1.0, 0.0])
        result = compute_composite_alpha(gp, ivol, resmom)
        assert result["permno"].is_monotonic_increasing


# ============================================================================
# Sign convention
# ============================================================================


class TestSignConvention:
    def test_ivol_sign_is_flipped(self) -> None:
        """IVol z-score in the output should equal -1 × input z-score."""
        gp = _make_aligned([1], [0.0])
        ivol = _make_aligned([1], [2.5])  # input z = 2.5
        resmom = _make_aligned([1], [0.0])
        # need at least 1 signal; relax min_signals to 1 to inspect raw values
        # ...actually we still need the row to survive, and with all 3 signals
        # populated, n_signals=3 and the row will be in output regardless
        result = compute_composite_alpha(gp, ivol, resmom, min_signals=1)
        assert result["z_ivol"].iloc[0] == pytest.approx(-2.5)

    def test_gp_sign_is_preserved(self) -> None:
        gp = _make_aligned([1], [1.5])
        ivol = _make_aligned([1], [0.0])
        resmom = _make_aligned([1], [0.0])
        result = compute_composite_alpha(gp, ivol, resmom, min_signals=1)
        assert result["z_gp"].iloc[0] == pytest.approx(1.5)

    def test_resmom_sign_is_preserved(self) -> None:
        gp = _make_aligned([1], [0.0])
        ivol = _make_aligned([1], [0.0])
        resmom = _make_aligned([1], [1.5])
        result = compute_composite_alpha(gp, ivol, resmom, min_signals=1)
        assert result["z_resmom"].iloc[0] == pytest.approx(1.5)


# ============================================================================
# Composite math
# ============================================================================


class TestCompositeMath:
    def test_composite_raw_is_simple_mean(self) -> None:
        """When all three signals present, composite_raw = mean of signed z's."""
        gp = _make_aligned([1], [3.0])
        ivol = _make_aligned([1], [-2.0])  # flipped → +2.0
        resmom = _make_aligned([1], [1.0])
        result = compute_composite_alpha(gp, ivol, resmom)
        # Expected: (3.0 + 2.0 + 1.0) / 3 = 2.0
        assert result["composite_raw"].iloc[0] == pytest.approx(2.0)

    def test_composite_z_has_mean_zero_per_date(self) -> None:
        """Cross-sectional mean of composite_z must be 0 within each date."""
        permnos = list(range(1, 11))
        z_values = [float(i - 5.5) for i in permnos]  # -4.5 to 4.5
        gp = _make_aligned(permnos, z_values)
        ivol = _make_aligned(permnos, [-v for v in z_values])  # opposite
        resmom = _make_aligned(permnos, z_values)
        result = compute_composite_alpha(gp, ivol, resmom)
        for _date, group in result.groupby("date"):
            assert abs(group["composite_z"].mean()) < 1e-10

    def test_composite_z_has_unit_std_per_date(self) -> None:
        """Cross-sectional std of composite_z must be 1 within each date."""
        permnos = list(range(1, 11))
        z_values = [float(i - 5.5) for i in permnos]
        gp = _make_aligned(permnos, z_values)
        ivol = _make_aligned(permnos, [-v for v in z_values])
        resmom = _make_aligned(permnos, z_values)
        result = compute_composite_alpha(gp, ivol, resmom)
        for _date, group in result.groupby("date"):
            assert group["composite_z"].std() == pytest.approx(1.0, abs=1e-10)

    def test_identical_composite_raw_yields_nan_zscore(self) -> None:
        """If all composite_raw values are equal cross-sectionally, std is 0
        → composite_z is NaN."""
        permnos = [1, 2, 3]
        # All zero → all composites zero → std 0
        gp = _make_aligned(permnos, [0.0, 0.0, 0.0])
        ivol = _make_aligned(permnos, [0.0, 0.0, 0.0])
        resmom = _make_aligned(permnos, [0.0, 0.0, 0.0])
        result = compute_composite_alpha(gp, ivol, resmom)
        assert result["composite_z"].isna().all()


# ============================================================================
# Missing signal handling
# ============================================================================


class TestMissingSignalHandling:
    def test_default_min_signals_is_two(self) -> None:
        """Stock with only 1 signal should be dropped by default."""
        # Stock 2 has only ResMom (GP and IVol NaN)
        gp = _make_aligned([1, 2], [1.0, np.nan])
        ivol = _make_aligned([1, 2], [-1.0, np.nan])
        resmom = _make_aligned([1, 2], [1.0, 1.0])
        result = compute_composite_alpha(gp, ivol, resmom)
        # Stock 2 should be dropped
        assert 2 not in result["permno"].tolist()

    def test_two_signals_sufficient_by_default(self) -> None:
        """Stock with 2 signals should be retained."""
        gp = _make_aligned([1, 2], [1.0, np.nan])  # stock 2 missing GP
        ivol = _make_aligned([1, 2], [-1.0, -1.0])
        resmom = _make_aligned([1, 2], [1.0, 1.0])
        result = compute_composite_alpha(gp, ivol, resmom)
        assert 2 in result["permno"].tolist()
        # n_signals for stock 2 should be 2
        assert result.loc[result["permno"] == 2, "n_signals"].iloc[0] == 2

    def test_min_signals_three_drops_partial(self) -> None:
        """With min_signals=3, partial-signal rows are dropped."""
        gp = _make_aligned([1, 2], [1.0, np.nan])
        ivol = _make_aligned([1, 2], [-1.0, -1.0])
        resmom = _make_aligned([1, 2], [1.0, 1.0])
        result = compute_composite_alpha(gp, ivol, resmom, min_signals=3)
        assert 2 not in result["permno"].tolist()

    def test_min_signals_one_keeps_all(self) -> None:
        """With min_signals=1, even single-signal rows are kept."""
        gp = _make_aligned([1, 2, 3], [1.0, np.nan, np.nan])
        ivol = _make_aligned([1, 2, 3], [-1.0, -1.0, np.nan])
        resmom = _make_aligned([1, 2, 3], [1.0, 1.0, 1.0])
        result = compute_composite_alpha(gp, ivol, resmom, min_signals=1)
        # All three stocks should be present
        assert set(result["permno"].tolist()) == {1, 2, 3}

    def test_n_signals_count_correct(self) -> None:
        gp = _make_aligned([1, 2, 3], [1.0, 1.0, np.nan])
        ivol = _make_aligned([1, 2, 3], [-1.0, np.nan, -1.0])
        resmom = _make_aligned([1, 2, 3], [1.0, 1.0, 1.0])
        result = compute_composite_alpha(gp, ivol, resmom, min_signals=1)
        # Stock 1: all 3 signals
        # Stock 2: GP + ResMom = 2
        # Stock 3: IVol + ResMom = 2
        result = result.sort_values("permno").reset_index(drop=True)
        assert result["n_signals"].tolist() == [3, 2, 2]


# ============================================================================
# Multi-date
# ============================================================================


class TestMultiDate:
    def test_zscoring_independent_per_date(self) -> None:
        """Composite z-scoring should be cross-sectional WITHIN each date."""
        # Date 1: composite_raw values [-1, 0, 1]
        # Date 2: composite_raw values [10, 20, 30] (very different scale)
        # Both should z-score to mean=0, std=1 independently
        d1 = pd.to_datetime("2020-06-30")
        d2 = pd.to_datetime("2020-09-30")

        gp = pd.DataFrame({
            "date": [d1, d1, d1, d2, d2, d2],
            "permno": [1, 2, 3, 1, 2, 3],
            "z_score": [-1.0, 0.0, 1.0, 10.0, 20.0, 30.0],
            "raw_signal": [-1.0, 0.0, 1.0, 10.0, 20.0, 30.0],
            "winsorized": [-1.0, 0.0, 1.0, 10.0, 20.0, 30.0],
        })
        ivol = pd.DataFrame({
            "date": [d1, d1, d1, d2, d2, d2],
            "permno": [1, 2, 3, 1, 2, 3],
            "z_score": [1.0, 0.0, -1.0, -10.0, -20.0, -30.0],
            "raw_signal": [1.0, 0.0, -1.0, -10.0, -20.0, -30.0],
            "winsorized": [1.0, 0.0, -1.0, -10.0, -20.0, -30.0],
        })
        resmom = pd.DataFrame({
            "date": [d1, d1, d1, d2, d2, d2],
            "permno": [1, 2, 3, 1, 2, 3],
            "z_score": [-1.0, 0.0, 1.0, 10.0, 20.0, 30.0],
            "raw_signal": [-1.0, 0.0, 1.0, 10.0, 20.0, 30.0],
            "winsorized": [-1.0, 0.0, 1.0, 10.0, 20.0, 30.0],
        })

        result = compute_composite_alpha(gp, ivol, resmom)
        for _date, group in result.groupby("date"):
            assert abs(group["composite_z"].mean()) < 1e-10
            assert group["composite_z"].std() == pytest.approx(1.0, abs=1e-10)
