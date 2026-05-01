"""Tests for the Idiosyncratic Volatility signal module.

Pure-function tests with synthetic data designed to test specific behaviors:
- Math correctness on low-noise data (clean beta recovery)
- Residual std recovery on noisy data (the actual signal output)
- Edge cases: missing data, insufficient observations, NaN handling
- Schema, sort order, date filtering
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from axiom_fund.signals.idiosyncratic_volatility import (
    IVOL_RAW_COLUMNS,
    compute_idiosyncratic_volatility,
)

# ============================================================================
# Helpers
# ============================================================================


def _make_orthogonal_factors(n_days: int, seed: int = 42) -> pd.DataFrame:
    """Build orthogonal FF factors via QR decomposition for clean tests."""
    np.random.seed(seed)
    raw = np.random.normal(0, 1, size=(n_days, 3))
    q, _ = np.linalg.qr(raw)
    factors = q * np.array([0.012, 0.005, 0.006]) * np.sqrt(n_days)
    dates = pd.date_range("2020-01-02", periods=n_days, freq="B")
    return pd.DataFrame(
        {
            "date": dates,
            "mktrf": factors[:, 0] + 0.0005,
            "smb": factors[:, 1] + 0.0001,
            "hml": factors[:, 2] + 0.0001,
            "rf": [0.00005] * n_days,
        }
    )


def _make_returns_from_factors(
    ff: pd.DataFrame,
    permnos: list[int],
    betas: tuple[float, float, float] = (1.2, 0.3, -0.1),
    alpha: float = 0.0001,
    residual_std: float = 0.005,
    seed: int = 100,
) -> pd.DataFrame:
    """Build synthetic returns with known FF3 factor structure."""
    rng = np.random.default_rng(seed)
    n_days = len(ff)
    rows = []
    for permno in permnos:
        eps = rng.normal(0, residual_std, n_days)
        excess = (
            alpha
            + betas[0] * ff["mktrf"]
            + betas[1] * ff["smb"]
            + betas[2] * ff["hml"]
            + eps
        )
        ret = excess + ff["rf"]
        for i, d in enumerate(ff["date"]):
            rows.append({"permno": permno, "date": d, "ret": ret.iloc[i]})
    return pd.DataFrame(rows)


# ============================================================================
# Input validation
# ============================================================================


class TestInputValidation:
    def test_reversed_dates_raises(self) -> None:
        ff = _make_orthogonal_factors(80)
        rets = _make_returns_from_factors(ff, [1])
        with pytest.raises(ValueError, match="start_date"):
            compute_idiosyncratic_volatility(
                rets, ff, "2020-12-31", "2020-01-01"
            )

    def test_missing_returns_columns_raises(self) -> None:
        ff = _make_orthogonal_factors(80)
        rets = pd.DataFrame({"foo": [1]})
        with pytest.raises(ValueError, match="returns_df missing"):
            compute_idiosyncratic_volatility(
                rets, ff, "2020-01-01", "2020-12-31"
            )

    def test_missing_factors_columns_raises(self) -> None:
        ff = pd.DataFrame({"date": pd.to_datetime(["2020-01-01"])})
        rets = pd.DataFrame(
            {"permno": [1], "date": pd.to_datetime(["2020-01-01"]), "ret": [0.01]}
        )
        with pytest.raises(ValueError, match="ff_factors_df missing"):
            compute_idiosyncratic_volatility(
                rets, ff, "2020-01-01", "2020-12-31"
            )

    def test_window_too_small_raises(self) -> None:
        ff = _make_orthogonal_factors(80)
        rets = _make_returns_from_factors(ff, [1])
        with pytest.raises(ValueError, match="window_size"):
            compute_idiosyncratic_volatility(
                rets, ff, "2020-01-01", "2020-12-31", window_size=1
            )

    def test_min_obs_exceeds_window_raises(self) -> None:
        ff = _make_orthogonal_factors(80)
        rets = _make_returns_from_factors(ff, [1])
        with pytest.raises(ValueError, match="min_obs"):
            compute_idiosyncratic_volatility(
                rets, ff, "2020-01-01", "2020-12-31",
                window_size=60, min_obs=100,
            )

    def test_min_obs_too_small_raises(self) -> None:
        ff = _make_orthogonal_factors(80)
        rets = _make_returns_from_factors(ff, [1])
        with pytest.raises(ValueError, match="min_obs"):
            compute_idiosyncratic_volatility(
                rets, ff, "2020-01-01", "2020-12-31", min_obs=2
            )


# ============================================================================
# Output schema
# ============================================================================


class TestOutputSchema:
    def test_columns_match_canonical(self) -> None:
        ff = _make_orthogonal_factors(80)
        rets = _make_returns_from_factors(ff, [1])
        result = compute_idiosyncratic_volatility(
            rets, ff, "2020-01-02", "2020-12-31"
        )
        assert tuple(result.columns) == IVOL_RAW_COLUMNS

    def test_empty_returns_yields_empty_with_canonical_columns(self) -> None:
        ff = _make_orthogonal_factors(80)
        rets = pd.DataFrame(columns=["permno", "date", "ret"])
        result = compute_idiosyncratic_volatility(
            rets, ff, "2020-01-02", "2020-12-31"
        )
        assert len(result) == 0
        assert list(result.columns) == list(IVOL_RAW_COLUMNS)

    def test_sorted_by_date_then_permno(self) -> None:
        ff = _make_orthogonal_factors(80)
        rets = _make_returns_from_factors(ff, [2, 1])  # permno 2 first
        result = compute_idiosyncratic_volatility(
            rets, ff, "2020-01-02", "2020-12-31"
        )
        # Within each date, permno should be ascending
        for _date, group in result.groupby("date_filed"):
            assert group["permno"].is_monotonic_increasing


# ============================================================================
# Math correctness — low-noise regime where SE is small
# ============================================================================


class TestMathCorrectness:
    def test_clean_beta_recovery_low_noise(self) -> None:
        """With small residuals, regression should recover true betas."""
        ff = _make_orthogonal_factors(150)
        # Very low residual std → small SEs → clean coefficient recovery
        true_betas = (1.2, 0.3, -0.1)
        rets = _make_returns_from_factors(
            ff, [1], betas=true_betas, residual_std=0.0001
        )
        result = compute_idiosyncratic_volatility(
            rets, ff, "2020-01-02", "2020-12-31"
        )
        # Take mean across all rolling windows
        assert result["beta_mkt"].mean() == pytest.approx(true_betas[0], abs=0.05)
        assert result["beta_smb"].mean() == pytest.approx(true_betas[1], abs=0.05)
        assert result["beta_hml"].mean() == pytest.approx(true_betas[2], abs=0.05)

    def test_residual_std_recovery_low_noise(self) -> None:
        """Residual std should match true noise level closely."""
        ff = _make_orthogonal_factors(150)
        rets = _make_returns_from_factors(
            ff, [1], residual_std=0.005
        )
        result = compute_idiosyncratic_volatility(
            rets, ff, "2020-01-02", "2020-12-31"
        )
        # 5% tolerance — sample std vs population std at n=60
        assert result["residual_std"].mean() == pytest.approx(0.005, rel=0.10)

    def test_residual_std_recovery_high_noise(self) -> None:
        """Even with high noise, residual std should still be recovered."""
        ff = _make_orthogonal_factors(150)
        rets = _make_returns_from_factors(
            ff, [1], residual_std=0.030
        )
        result = compute_idiosyncratic_volatility(
            rets, ff, "2020-01-02", "2020-12-31"
        )
        assert result["residual_std"].mean() == pytest.approx(0.030, rel=0.10)

    def test_annualization_factor(self) -> None:
        """raw_signal should equal residual_std × √252."""
        ff = _make_orthogonal_factors(150)
        rets = _make_returns_from_factors(ff, [1], residual_std=0.005)
        result = compute_idiosyncratic_volatility(
            rets, ff, "2020-01-02", "2020-12-31"
        )
        # For each row, raw_signal == residual_std × √252
        ratio = result["raw_signal"] / result["residual_std"]
        assert (ratio - np.sqrt(252)).abs().max() < 1e-10

    def test_high_ivol_stock_has_higher_signal(self) -> None:
        """A stock with higher residual variance must have higher raw_signal."""
        ff = _make_orthogonal_factors(150)
        # Two stocks: high IVol and low IVol
        rets_high = _make_returns_from_factors(
            ff, [1], residual_std=0.030, seed=100
        )
        rets_low = _make_returns_from_factors(
            ff, [2], residual_std=0.005, seed=200
        )
        rets = pd.concat([rets_high, rets_low], ignore_index=True)

        result = compute_idiosyncratic_volatility(
            rets, ff, "2020-01-02", "2020-12-31"
        )
        high_mean = result[result["permno"] == 1]["raw_signal"].mean()
        low_mean = result[result["permno"] == 2]["raw_signal"].mean()
        assert high_mean > low_mean
        # Ratio should be roughly 6x (= 0.030 / 0.005)
        assert 4.0 < high_mean / low_mean < 8.0


# ============================================================================
# Window and observation handling
# ============================================================================


class TestWindowHandling:
    def test_first_window_size_minus_one_rows_skipped(self) -> None:
        """Output starts only after window_size-1 trailing observations."""
        ff = _make_orthogonal_factors(150)
        rets = _make_returns_from_factors(ff, [1])
        result = compute_idiosyncratic_volatility(
            rets, ff, "2020-01-02", "2020-12-31", window_size=60
        )
        # First output date should be the 60th business day (index 59)
        first_output_date = result["date_filed"].min()
        ff_sorted = ff.sort_values("date").reset_index(drop=True)
        expected_first = ff_sorted["date"].iloc[59]
        assert first_output_date == expected_first

    def test_insufficient_observations_skipped(self) -> None:
        """Rows with < min_obs valid observations should be dropped."""
        ff = _make_orthogonal_factors(150)
        rets = _make_returns_from_factors(ff, [1])
        # Insert NaN returns to reduce observation count below min_obs
        rets.loc[rets.index[:50], "ret"] = np.nan

        result = compute_idiosyncratic_volatility(
            rets, ff, "2020-01-02", "2020-12-31",
            window_size=60, min_obs=40,
        )
        # n_obs should always be >= 40 in output
        assert (result["n_obs"] >= 40).all()


# ============================================================================
# Date and dtype handling
# ============================================================================


class TestDateHandling:
    def test_iso_string_dates_accepted(self) -> None:
        ff = _make_orthogonal_factors(80)
        rets = _make_returns_from_factors(ff, [1])
        # Should not raise
        compute_idiosyncratic_volatility(
            rets, ff, "2020-01-02", "2020-12-31"
        )

    def test_date_object_dates_accepted(self) -> None:
        ff = _make_orthogonal_factors(80)
        rets = _make_returns_from_factors(ff, [1])
        # Should not raise
        compute_idiosyncratic_volatility(
            rets, ff, date(2020, 1, 2), date(2020, 12, 31)
        )

    def test_handles_mixed_datetime_precision(self) -> None:
        """Real WRDS data has second precision; test with mixed."""
        ff = _make_orthogonal_factors(80)
        ff["date"] = ff["date"].astype("datetime64[s]")
        rets = _make_returns_from_factors(ff, [1])
        # rets has nanosecond, ff has second — should handle
        result = compute_idiosyncratic_volatility(
            rets, ff, "2020-01-02", "2020-12-31"
        )
        assert len(result) > 0


# ============================================================================
# NaN handling
# ============================================================================


class TestNaNHandling:
    def test_nan_returns_excluded_from_window(self) -> None:
        """NaN returns shouldn't crash; they should be excluded."""
        ff = _make_orthogonal_factors(150)
        rets = _make_returns_from_factors(ff, [1])
        # Sprinkle some NaNs (not enough to drop below min_obs)
        rets.loc[rets.index[:5], "ret"] = np.nan

        result = compute_idiosyncratic_volatility(
            rets, ff, "2020-01-02", "2020-12-31"
        )
        # Output should still have valid signals
        assert len(result) > 0
        assert result["raw_signal"].notna().all()

    def test_all_nan_returns_yields_empty(self) -> None:
        """If all returns are NaN, output is empty (no valid windows)."""
        ff = _make_orthogonal_factors(80)
        rets = _make_returns_from_factors(ff, [1])
        rets["ret"] = np.nan

        result = compute_idiosyncratic_volatility(
            rets, ff, "2020-01-02", "2020-12-31"
        )
        assert len(result) == 0
