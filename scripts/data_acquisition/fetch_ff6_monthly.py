"""Download and cache Fama-French 5 factors + Momentum from Ken French's website.

Sources:
  FF5: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip
  MOM: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_CSV.zip

Output: data/cache/ff6_monthly.parquet with columns
  date (month-end), mkt_rf, smb, hml, rmw, cma, mom, rf

All returns are in percentage points (as Ken French publishes). Divide by
100 for use in regressions if needed.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path
import urllib.request
import pandas as pd


OUTPUT = Path("data/cache/ff6_monthly.parquet")
FF5_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip"
MOM_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_CSV.zip"


def _download_zip(url: str) -> bytes:
    """Download a ZIP from Ken French's website and return the inner CSV bytes."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        zip_bytes = r.read()
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        csv_name = z.namelist()[0]
        return z.read(csv_name)


def _parse_ff_csv(csv_bytes: bytes) -> pd.DataFrame:
    """Parse Ken French's CSV format.

    Files have a preamble, then a monthly section, then an annual section.
    The monthly section is what we want. The annual section starts with a
    header like " Annual Factors: ..." or a blank line after the last monthly row.
    Monthly rows have YYYYMM as the index.
    """
    text = csv_bytes.decode("latin-1", errors="ignore")
    lines = text.splitlines()

    # Find the first line that looks like a data row (starts with 6-digit YYYYMM)
    data_start = None
    for i, line in enumerate(lines):
        parts = [p.strip() for p in line.split(",")]
        if parts and parts[0].isdigit() and len(parts[0]) == 6:
            data_start = i
            break
    if data_start is None:
        raise ValueError("Could not find monthly data start in CSV")

    # Find where monthly data ends (first non-digit YYYYMM index)
    data_end = len(lines)
    for i in range(data_start, len(lines)):
        parts = [p.strip() for p in lines[i].split(",")]
        if not parts[0].isdigit() or len(parts[0]) != 6:
            data_end = i
            break

    # Header is the line just before data_start
    header_line = lines[data_start - 1]
    header = [h.strip() for h in header_line.split(",")]
    # First column has no header (it's the date)
    if not header[0]:
        header[0] = "date_ym"

    # Build DataFrame from data rows
    rows = []
    for line in lines[data_start:data_end]:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < len(header):
            continue
        rows.append(parts[:len(header)])
    df = pd.DataFrame(rows, columns=header)
    df["date_ym"] = df["date_ym"].astype(int)
    for c in header[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def main() -> None:
    print(f"Downloading FF5 from Ken French's website...")
    ff5_bytes = _download_zip(FF5_URL)
    ff5 = _parse_ff_csv(ff5_bytes)
    print(f"  FF5 monthly rows: {len(ff5)}, range: {ff5.date_ym.min()} to {ff5.date_ym.max()}")

    print(f"Downloading Momentum from Ken French's website...")
    mom_bytes = _download_zip(MOM_URL)
    mom = _parse_ff_csv(mom_bytes)
    print(f"  MOM monthly rows: {len(mom)}, range: {mom.date_ym.min()} to {mom.date_ym.max()}")

    # Merge on date_ym
    merged = ff5.merge(mom, on="date_ym", how="left")
    print(f"  Merged rows: {len(merged)}")

    # Rename columns to snake_case
    rename_map = {}
    for c in merged.columns:
        if c == "date_ym":
            continue
        rename_map[c] = c.lower().replace("-", "_").strip()
    merged = merged.rename(columns=rename_map)
    print(f"  Columns after rename: {list(merged.columns)}")

    # Convert date_ym to month-end date
    merged["date"] = pd.to_datetime(
        merged["date_ym"].astype(str), format="%Y%m"
    ) + pd.offsets.MonthEnd(0)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(OUTPUT, index=False)
    print(f"\nSaved: {OUTPUT}")
    print(f"Last 5 rows:")
    print(merged.tail(5).to_string())


if __name__ == "__main__":
    main()