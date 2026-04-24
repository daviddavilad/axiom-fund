"""Tests for the storage module (Parquet save/load helpers)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from axiom_fund.data.storage import (
    load_parquet,
    load_partitioned,
    save_parquet,
    save_partitioned,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """A small, typed DataFrame for storage round-trip tests."""
    return pd.DataFrame(
        {
            "permno": [14593, 14593, 10107, 10107],
            "date": pd.to_datetime(
                ["2020-01-02", "2020-01-03", "2020-01-02", "2020-01-03"]
            ),
            "ret": [0.01, -0.005, 0.008, 0.003],
            "year": [2020, 2020, 2020, 2020],
        }
    )


class TestRoundTrip:
    """save then load should return equal data."""

    def test_basic_round_trip(self, tmp_path: Path, sample_df: pd.DataFrame) -> None:
        file_path = tmp_path / "test.parquet"
        save_parquet(sample_df, file_path)
        loaded = load_parquet(file_path)
        pd.testing.assert_frame_equal(loaded, sample_df)

    def test_round_trip_creates_parent_dirs(
        self, tmp_path: Path, sample_df: pd.DataFrame
    ) -> None:
        file_path = tmp_path / "nested" / "subdir" / "test.parquet"
        save_parquet(sample_df, file_path)
        assert file_path.exists()

    def test_column_selection_on_load(
        self, tmp_path: Path, sample_df: pd.DataFrame
    ) -> None:
        file_path = tmp_path / "test.parquet"
        save_parquet(sample_df, file_path)
        loaded = load_parquet(file_path, columns=["permno", "ret"])
        assert list(loaded.columns) == ["permno", "ret"]

    def test_load_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_parquet(tmp_path / "nonexistent.parquet")


class TestPartitioned:
    """Partitioned save/load."""

    def test_save_partitioned_creates_subdirs(
        self, tmp_path: Path, sample_df: pd.DataFrame
    ) -> None:
        # Extend sample with another year so we get two partitions
        df = pd.concat(
            [
                sample_df,
                pd.DataFrame(
                    {
                        "permno": [14593],
                        "date": pd.to_datetime(["2021-01-04"]),
                        "ret": [0.02],
                        "year": [2021],
                    }
                ),
            ],
            ignore_index=True,
        )
        root = tmp_path / "partitioned_returns"
        save_partitioned(df, root, partition_col="year")
        assert (root / "year=2020" / "data.parquet").exists()
        assert (root / "year=2021" / "data.parquet").exists()

    def test_partitioned_round_trip(
        self, tmp_path: Path, sample_df: pd.DataFrame
    ) -> None:
        root = tmp_path / "partitioned_returns"
        save_partitioned(sample_df, root, partition_col="year")
        loaded = load_partitioned(root)
        # Sort both to compare by content, ignoring row order
        loaded_sorted = loaded.sort_values(["permno", "date"]).reset_index(drop=True)
        expected_sorted = sample_df.sort_values(["permno", "date"]).reset_index(
            drop=True
        )
        # Compare non-partition columns; the partition column (year) comes
        # back as categorical from pyarrow, which is expected behavior.
        non_partition_cols = [c for c in sample_df.columns if c != "year"]
        pd.testing.assert_frame_equal(
            loaded_sorted[non_partition_cols],
            expected_sorted[non_partition_cols],
        )
        # Separately verify the partition values survive round-trip
        assert sorted(loaded["year"].unique().tolist()) == sorted(
            expected_sorted["year"].unique().tolist()
        )

    def test_missing_partition_col_raises(
        self, tmp_path: Path, sample_df: pd.DataFrame
    ) -> None:
        with pytest.raises(ValueError, match="not in DataFrame"):
            save_partitioned(sample_df, tmp_path, partition_col="nonexistent")

    def test_load_nonexistent_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_partitioned(tmp_path / "nonexistent")
