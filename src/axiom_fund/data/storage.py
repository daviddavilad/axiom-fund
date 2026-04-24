"""Parquet storage helpers for the axiom_fund data layer.

Thin wrappers around pandas/pyarrow Parquet I/O with consistent conventions:
  - All tabular data is stored as Parquet (columnar, fast, typed)
  - Partitioning by year for large panels, to enable incremental reads
  - Consistent snake_case naming
  - Explicit compression (snappy: fast, reasonable ratio, industry default)

Usage:
    from axiom_fund.data.storage import save_parquet, load_parquet

    save_parquet(panel_df, "data/processed/returns/aapl_msft_2020.parquet")
    panel_df = load_parquet("data/processed/returns/aapl_msft_2020.parquet")

    # Partitioned storage for large panels:
    save_partitioned(panel_df, "data/processed/returns", partition_col="year")
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

CompressionType = Literal["snappy", "gzip", "brotli", "lz4", "zstd"]

DEFAULT_COMPRESSION: CompressionType = "snappy"


def save_parquet(
    df: pd.DataFrame,
    path: str | Path,
    *,
    compression: CompressionType = DEFAULT_COMPRESSION,
) -> None:
    """Save a DataFrame to a Parquet file.

    Parameters
    ----------
    df : pandas.DataFrame
        Data to save. Must have a valid schema (no object columns that
        Parquet cannot serialize).
    path : str or Path
        Destination path. Parent directories are created if they don't exist.
    compression : str
        Parquet compression codec. Default "snappy" (fast, widely supported).
        Alternatives: "gzip", "zstd", "brotli", "none".
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, compression=compression, index=False)


def load_parquet(
    path: str | Path,
    *,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Load a DataFrame from a Parquet file.

    Parameters
    ----------
    path : str or Path
        Source path.
    columns : list of str, optional
        If given, only load these columns. Parquet's columnar layout makes
        this genuinely cheap — it reads only the selected columns from disk.

    Returns
    -------
    pandas.DataFrame
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")
    return pd.read_parquet(path, columns=columns)


def save_partitioned(
    df: pd.DataFrame,
    root_path: str | Path,
    partition_col: str,
    *,
    compression: CompressionType = DEFAULT_COMPRESSION,
) -> None:
    """Save a DataFrame as a partitioned Parquet dataset.

    Writes one file per unique value of partition_col, under root_path/.
    This layout lets readers load only the partitions they need (e.g., a
    single year out of a 20-year panel).

    Parameters
    ----------
    df : pandas.DataFrame
        Data to save. Must contain partition_col.
    root_path : str or Path
        Directory where partitions are written. Created if it doesn't exist.
    partition_col : str
        Column to partition on. Values become subdirectory names like
        `{partition_col}={value}/data.parquet`.
    compression : str
        See save_parquet.

    Raises
    ------
    ValueError
        If partition_col is not in df.
    """
    if partition_col not in df.columns:
        raise ValueError(
            f"partition_col {partition_col!r} not in DataFrame columns: "
            f"{list(df.columns)}"
        )

    root_path = Path(root_path)
    root_path.mkdir(parents=True, exist_ok=True)

    for partition_value, group in df.groupby(partition_col, sort=True):
        partition_dir = root_path / f"{partition_col}={partition_value}"
        partition_dir.mkdir(parents=True, exist_ok=True)
        file_path = partition_dir / "data.parquet"
        # Drop the partition column before writing — pyarrow reconstructs it
        # from the directory name on read. Storing it in both places causes
        # a type-mismatch error when reading the whole dataset back.
        group_to_write = group.drop(columns=[partition_col])
        group_to_write.to_parquet(file_path, compression=compression, index=False)

def load_partitioned(
    root_path: str | Path,
    *,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Load all partitions of a partitioned Parquet dataset into one DataFrame.

    Parameters
    ----------
    root_path : str or Path
        Directory containing the partitions (as written by save_partitioned).
    columns : list of str, optional
        If given, only load these columns.

    Returns
    -------
    pandas.DataFrame
        All partitions concatenated into a single DataFrame, reset_index'd.

    Raises
    ------
    FileNotFoundError
        If root_path does not exist.
    """
    root_path = Path(root_path)
    if not root_path.exists():
        raise FileNotFoundError(f"Partitioned dataset not found: {root_path}")

    # pandas.read_parquet can read a directory of partitions directly via
    # pyarrow's dataset API, which handles the Hive-style partitioning we
    # write in save_partitioned.
    return pd.read_parquet(root_path, columns=columns).reset_index(drop=True)
