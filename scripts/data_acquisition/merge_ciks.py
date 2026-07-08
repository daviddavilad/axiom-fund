"""Merge SEC-based and CRSP-Compustat-based CIK resolutions.

Rules:
1. Both sources agree → use shared CIK (source='agree')
2. Both have CIK but disagree → query SEC EDGAR for both, pick whichever
   has 10-K filings in our 2019-2024 window. Tiebreaker: prefer CRSP.
   (source='crsp_won', 'sec_won', or 'both_valid_prefer_crsp')
3. SEC-only → use SEC CIK (source='sec_only')
4. CRSP-only → use CRSP CIK (source='crsp_only')
5. Neither → firm stays unresolved (source='none')

Output: data/cache/lazy_prices_ciks_merged.parquet
Columns: permno, ticker, comnam, cik, cik_source
"""
# ruff: noqa: I001

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from axiom_fund.data.edgar import EdgarClient


SEC_PATH = Path("data/cache/lazy_prices_ciks.parquet")
CRSP_PATH = Path("data/cache/lazy_prices_ciks_crsp.parquet")
UNIVERSE_PATH = Path("data/cache/lazy_prices_universe.parquet")
OUTPUT_PATH = Path("data/cache/lazy_prices_ciks_merged.parquet")

SAMPLE_START = "2019-01-01"
SAMPLE_END = "2025-01-01"


def _normalize_cik(cik) -> str | None:
    """Convert CIK to int-string (no leading zeros) for comparison."""
    if pd.isna(cik):
        return None
    s = str(cik).lstrip("0")
    return s if s else None


def _count_10ks_in_window(client: EdgarClient, cik: str) -> int:
    """Count 10-K filings in the 2019-2024 window for a given CIK."""
    try:
        filings = client.get_10k_history(cik, earliest_filing_date=SAMPLE_START)
        return sum(1 for f in filings
                   if SAMPLE_START <= f.filing_date < SAMPLE_END)
    except Exception:
        return 0


def main() -> int:
    for p in [SEC_PATH, CRSP_PATH, UNIVERSE_PATH]:
        if not p.exists():
            print(f"ERROR: {p} not found.", file=sys.stderr)
            return 1

    sec = pd.read_parquet(SEC_PATH)
    crsp = pd.read_parquet(CRSP_PATH)
    universe = pd.read_parquet(UNIVERSE_PATH)

    print(f"SEC-based: {len(sec):,} firms")
    print(f"CRSP-based: {len(crsp):,} firms")
    print(f"Universe: {len(universe):,} firms")
    print()

    # Normalize CIKs to int-strings for comparison
    sec = sec.copy()
    sec["cik_sec_norm"] = sec["cik"].apply(_normalize_cik)
    crsp = crsp.copy()
    crsp["cik_crsp_norm"] = crsp["cik_crsp"].apply(_normalize_cik)

    # Full outer merge on permno
    merged = universe[["permno", "ticker", "comnam"]].merge(
        sec[["permno", "cik_sec_norm"]].rename(columns={"cik_sec_norm": "cik_sec"}),
        on="permno", how="left"
    ).merge(
        crsp[["permno", "cik_crsp_norm"]].rename(columns={"cik_crsp_norm": "cik_crsp"}),
        on="permno", how="left"
    )
    print(f"Merged rows: {len(merged):,}")
    print()

    # Categorize each row
    def categorize(row):
        s, c = row["cik_sec"], row["cik_crsp"]
        if pd.isna(s) and pd.isna(c):
            return "none"
        if pd.isna(s) and not pd.isna(c):
            return "crsp_only"
        if not pd.isna(s) and pd.isna(c):
            return "sec_only"
        if s == c:
            return "agree"
        return "disagree"

    merged["category"] = merged.apply(categorize, axis=1)

    print("Category counts:")
    print(merged["category"].value_counts())
    print()

    # Rule 1 and 3, 4: assign CIKs for non-disagree cases
    merged["cik"] = None
    merged["cik_source"] = "none"

    agree_mask = merged["category"] == "agree"
    merged.loc[agree_mask, "cik"] = merged.loc[agree_mask, "cik_sec"]
    merged.loc[agree_mask, "cik_source"] = "agree"

    sec_only_mask = merged["category"] == "sec_only"
    merged.loc[sec_only_mask, "cik"] = merged.loc[sec_only_mask, "cik_sec"]
    merged.loc[sec_only_mask, "cik_source"] = "sec_only"

    crsp_only_mask = merged["category"] == "crsp_only"
    merged.loc[crsp_only_mask, "cik"] = merged.loc[crsp_only_mask, "cik_crsp"]
    merged.loc[crsp_only_mask, "cik_source"] = "crsp_only"

    # Rule 2: for disagree, query SEC EDGAR for both CIKs, pick winner
    disagree = merged[merged["category"] == "disagree"].copy()
    print(f"Rule 2 (empirical validation): resolving {len(disagree)} disagreements...")
    print()

    client = EdgarClient(output_dir=Path("data/raw/edgar"))
    # Force CIK map load once
    _ = client.get_cik("AAPL")

    disagreement_log = []
    for i, row in disagree.iterrows():
        ticker = row["ticker"]
        cik_sec = row["cik_sec"]
        cik_crsp = row["cik_crsp"]

        # Pad CIKs to 10 digits for EDGAR client
        cik_sec_padded = str(cik_sec).zfill(10)
        cik_crsp_padded = str(cik_crsp).zfill(10)

        n_sec = _count_10ks_in_window(client, cik_sec_padded)
        n_crsp = _count_10ks_in_window(client, cik_crsp_padded)

        if n_sec == 0 and n_crsp == 0:
            source = "both_invalid"
            chosen = None
        elif n_sec > 0 and n_crsp == 0:
            source = "sec_won"
            chosen = cik_sec
        elif n_sec == 0 and n_crsp > 0:
            source = "crsp_won"
            chosen = cik_crsp
        else:
            # Both valid; tiebreaker: prefer CRSP
            source = "both_valid_prefer_crsp"
            chosen = cik_crsp

        merged.loc[i, "cik"] = chosen
        merged.loc[i, "cik_source"] = source

        disagreement_log.append({
            "ticker": ticker,
            "comnam": row["comnam"],
            "cik_sec": cik_sec,
            "n_10ks_sec": n_sec,
            "cik_crsp": cik_crsp,
            "n_10ks_crsp": n_crsp,
            "chosen_cik": chosen,
            "source": source,
        })

        if (len(disagreement_log)) % 20 == 0:
            print(f"  {len(disagreement_log)}/{len(disagree)}...", flush=True)

    # Summary of disagreement resolution
    log_df = pd.DataFrame(disagreement_log)
    print()
    print("Disagreement resolution outcomes:")
    print(log_df["source"].value_counts())
    print()
    print("Sample disagreements (first 15, chronological by input order):")
    print(log_df[["ticker", "comnam", "n_10ks_sec", "n_10ks_crsp", "source"]].head(15).to_string(index=False))
    print()

    # Final summary
    print("=" * 70)
    print("Final merged CIK sources:")
    print(merged["cik_source"].value_counts())
    print()
    print(f"Total resolved: {merged['cik'].notna().sum():,}/{len(merged):,}")

    # Save
    output = merged[["permno", "ticker", "comnam", "cik", "cik_source"]].copy()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved: {OUTPUT_PATH} ({len(output)} rows)")

    # Save disagreement log for provenance
    log_path = OUTPUT_PATH.with_name("lazy_prices_ciks_merged_disagreements.parquet")
    log_df.to_parquet(log_path, index=False)
    print(f"Saved: {log_path} ({len(log_df)} rows)")

    # Metadata
    metadata_path = OUTPUT_PATH.with_suffix(".txt")
    metadata_path.write_text(
        f"Run: {datetime.now().isoformat()}\n"
        f"Universe size: {len(universe)}\n"
        f"SEC-based resolutions: {len(sec)}\n"
        f"CRSP-based resolutions: {len(crsp)}\n"
        f"Merged total: {len(output)}\n"
        f"Resolved: {output['cik'].notna().sum()}\n"
        f"By source:\n"
        + merged["cik_source"].value_counts().to_string() + "\n"
    )
    print(f"Metadata: {metadata_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())