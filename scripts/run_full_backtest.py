"""Full historical backtest runner.

Runs the locked Axiom Fund strategy from 2015-01-01 to 2025-12-31
on a top-100 universe, with all neutrality constraints active.

Outputs to data/cache/backtest_full/:
  - backtest_summary.parquet:   per-period DataFrame (saved by runner)
  - weights_<date>.parquet:     per-period weights checkpoint
  - results.csv:                 summary as CSV (this script)
  - run_metadata.txt:            timestamp, parameters, environment

Estimated runtime: 35-50 minutes for 132 monthly periods.
"""
# ruff: noqa: I001

from __future__ import annotations

import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from axiom_fund import _warnings  # noqa: F401

from dotenv import load_dotenv

from axiom_fund.backtest.engine import run_historical_backtest


# Locked parameters for this run
START_DATE = "2015-01-01"
END_DATE = "2025-12-31"
UNIVERSE_SIZE = 1000
POSITION_CAP = 0.005  # 0.5% per name for top-1000 spec
RISK_AVERSION = 1.0
HOLDING_DAYS = 21
CACHE_DIR = Path("data/cache/backtest_full_top1000_4sig")


def main() -> int:
    # Configure logging — INFO so we see one line per period
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    load_dotenv()
    username = os.getenv("WRDS_USERNAME")
    if not username:
        print("ERROR: WRDS_USERNAME not set", file=sys.stderr)
        return 1

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Save run metadata for later reference
    metadata = {
        "start_date": START_DATE,
        "end_date": END_DATE,
        "universe_size": UNIVERSE_SIZE,
        "position_cap": POSITION_CAP,
        "risk_aversion": RISK_AVERSION,
        "holding_days": HOLDING_DAYS,
        "constrain_dollar_neutral": True,
        "constrain_beta_neutral": True,
        "constrain_sector_neutral": True,
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    metadata_path = CACHE_DIR / "run_metadata.txt"
    with metadata_path.open("w") as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")

    print("=" * 70)
    print("Axiom Fund — Full Historical Backtest")
    print("=" * 70)
    print(f"Date range:           {START_DATE} → {END_DATE}")
    print(f"Universe size:        Top-{UNIVERSE_SIZE} by market cap")
    print(f"Position cap:         {POSITION_CAP*100:.2f}% per name")
    print(f"Holding period:       {HOLDING_DAYS} trading days (~1 month)")
    print(f"Risk aversion (λ):    {RISK_AVERSION}")
    print("Neutrality:           Dollar + Beta + Sector (all on)")
    print(f"Output directory:     {CACHE_DIR}")
    print("=" * 70)
    print()

    import wrds
    db = wrds.Connection(wrds_username=username)
    start_time = time.time()

    try:
        df = run_historical_backtest(
            db,
            start_date=START_DATE,
            end_date=END_DATE,
            universe_size=UNIVERSE_SIZE,
            position_cap=POSITION_CAP,
            risk_aversion=RISK_AVERSION,
            holding_days=HOLDING_DAYS,
            cache_dir=CACHE_DIR,
        )

        elapsed = time.time() - start_time
        print()
        print("=" * 70)
        print(f"Backtest complete in {elapsed/60:.1f} minutes")
        print("=" * 70)
        print(f"Periods: {len(df)}")
        if len(df) == 0:
            print("WARNING: No periods completed.")
            return 1

        # CSV mirror for easy inspection
        df.to_csv(CACHE_DIR / "results.csv")

        # Quick stats
        cum_return = (1 + df["realized_return"]).prod() - 1
        years = (df.index[-1] - df.index[0]).days / 365.25
        annualized_return = ((1 + cum_return) ** (1 / years)) - 1
        std_monthly = df["realized_return"].std()
        std_annualized = std_monthly * (12 ** 0.5)
        sharpe = (
            (df["realized_return"].mean() * 12) / std_annualized
            if std_annualized > 0 else float("nan")
        )
        hit_rate = (df["realized_return"] > 0).mean()

        # Drawdown
        cum_curve = (1 + df["realized_return"]).cumprod()
        running_max = cum_curve.cummax()
        drawdown = (cum_curve / running_max) - 1
        max_dd = drawdown.min()

        print()
        print(f"Cumulative return (gross):    {cum_return * 100:+.2f}%")
        print(f"Annualized return:            {annualized_return * 100:+.2f}%")
        print(f"Annualized vol:               {std_annualized * 100:.2f}%")
        print(f"Sharpe ratio (annualized):    {sharpe:.2f}")
        print(f"Hit rate:                     {hit_rate * 100:.1f}%")
        print(f"Max drawdown:                 {max_dd * 100:+.2f}%")
        print()

        # Year-by-year breakdown
        print("Year-by-year returns:")
        annual = (
            (1 + df["realized_return"]).groupby(df.index.year).prod() - 1
        )
        for year, ret in annual.items():
            n_months = (df.index.year == year).sum()
            print(f"  {year}: {ret * 100:+7.2f}%  ({n_months} months)")

        # Append metadata with end timestamp and stats
        with metadata_path.open("a") as f:
            f.write(f"finished_at: {datetime.now().isoformat(timespec='seconds')}\n")
            f.write(f"elapsed_minutes: {elapsed/60:.1f}\n")
            f.write(f"n_periods: {len(df)}\n")
            f.write(f"cumulative_return: {cum_return:.6f}\n")
            f.write(f"annualized_return: {annualized_return:.6f}\n")
            f.write(f"annualized_vol: {std_annualized:.6f}\n")
            f.write(f"sharpe: {sharpe:.4f}\n")
            f.write(f"hit_rate: {hit_rate:.4f}\n")
            f.write(f"max_drawdown: {max_dd:.6f}\n")

    finally:
        db.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
