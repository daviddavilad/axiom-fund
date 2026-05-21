"""Run single-signal attribution over the full backtest window.

For each of the 116 successful rebalance dates in the 4-signal
backtest, runs the optimizer 4 separate times — once per signal —
under the same neutrality and position-cap constraints as the
composite portfolio. Saves per-(period, signal) realized returns to
parquet for analysis.

Output: data/cache/attribution_4sig/single_signal_returns.parquet

Runtime estimate: ~4 hours (116 periods × 4 signals × ~30 sec/optimizer
call, plus cache build ~9 min). Can be left to run unattended.
"""
# ruff: noqa: I001

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd

from axiom_fund import _warnings  # noqa: F401

from dotenv import load_dotenv

from axiom_fund.backtest.attribution import (
    SIGNAL_SIGNS,
    _build_optimizer_inputs,
    compute_single_signal_period,
    rebuild_composite_for_date,
)
from axiom_fund.backtest.engine import _build_cache


BACKTEST_DIR = Path("data/cache/backtest_full_top1000_4sig")
OUTPUT_DIR = Path("data/cache/attribution_4sig")


def main() -> int:
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

    if not BACKTEST_DIR.exists():
        print(f"ERROR: backtest dir not found: {BACKTEST_DIR}", file=sys.stderr)
        return 1

    # Load rebalance dates from the existing 4-signal backtest
    summary = pd.read_parquet(BACKTEST_DIR / "backtest_summary.parquet")
    if not isinstance(summary.index, pd.DatetimeIndex):
        summary.index = pd.to_datetime(summary.index)
    summary = summary.sort_index()
    rebalance_dates: list[pd.Timestamp] = [
        pd.Timestamp(d) for d in summary.index
    ]

    print("=" * 70)
    print(f"Single-signal attribution over {len(rebalance_dates)} periods")
    print(f"Signals: {list(SIGNAL_SIGNS.keys())}")
    print(f"Output:  {OUTPUT_DIR}")
    print("=" * 70)
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    start = time.time()

    import wrds
    db = wrds.Connection(wrds_username=username)
    try:
        logging.info("Building cache for entire backtest window...")
        cache = _build_cache(
            db,
            rebalance_dates=rebalance_dates,
            universe_size=1000,
        )
        logging.info(
            "Cache: %d permnos, %d returns, %d fundamentals",
            len(cache.permnos_all),
            len(cache.returns_full),
            len(cache.fundamentals_full),
        )

        rows: list[dict[str, object]] = []
        for idx, rebal in enumerate(rebalance_dates):
            # Rebuild composite + optimizer inputs ONCE per period
            composite = rebuild_composite_for_date(cache, rebal)
            if composite is None or composite.empty:
                logging.warning(
                    "%s: composite empty, skipping all signals",
                    rebal.strftime("%Y-%m-%d"),
                )
                continue

            opt_inputs = _build_optimizer_inputs(cache, rebal)
            if opt_inputs is None:
                logging.warning(
                    "%s: optimizer inputs unavailable, skipping",
                    rebal.strftime("%Y-%m-%d"),
                )
                continue
            cov_wide, betas_full, sector_map, holding_wide = opt_inputs

            # Run each signal
            for signal_name in SIGNAL_SIGNS:
                try:
                    result = compute_single_signal_period(
                        cache=cache,
                        rebalance_date=rebal,
                        signal_name=signal_name,
                        composite_panel=composite,
                        cov_wide=cov_wide,
                        betas_for_engine=betas_full,
                        sectors_for_engine=sector_map,
                        hpr_for_engine=holding_wide,
                    )
                except ValueError as e:
                    logging.warning(
                        "%s for %s: %s",
                        rebal.strftime("%Y-%m-%d"), signal_name, str(e)[:120],
                    )
                    result = None
                if result is None:
                    rows.append({
                        "rebalance_date": rebal,
                        "signal": signal_name,
                        "realized_return": None,
                        "n_names": None,
                        "long_count": None,
                        "short_count": None,
                        "gross_leverage": None,
                        "optimizer_status": "skipped",
                    })
                    continue
                rows.append({
                    "rebalance_date": result.rebalance_date,
                    "signal": result.signal,
                    "realized_return": result.realized_return,
                    "n_names": result.n_names,
                    "long_count": result.long_count,
                    "short_count": result.short_count,
                    "gross_leverage": result.gross_leverage,
                    "optimizer_status": result.optimizer_status,
                })

            if (idx + 1) % 5 == 0:
                elapsed = time.time() - start
                periods_done = idx + 1
                periods_total = len(rebalance_dates)
                rate = elapsed / periods_done
                eta_min = (periods_total - periods_done) * rate / 60.0
                logging.info(
                    "Progress: %d/%d periods (~%.0f min remaining)",
                    periods_done, periods_total, eta_min,
                )

    finally:
        db.close()

    if not rows:
        print("No attribution results. Aborting.", file=sys.stderr)
        return 1

    df = pd.DataFrame(rows)
    df.to_parquet(OUTPUT_DIR / "single_signal_returns.parquet")
    df.to_csv(OUTPUT_DIR / "single_signal_returns.csv", index=False)
    elapsed = time.time() - start
    print()
    print(f"Total runtime: {elapsed/60:.1f} minutes")
    print(f"Saved: {OUTPUT_DIR}/single_signal_returns.parquet")
    print(f"Rows:  {len(df)}")
    print()

    # Quick summary
    valid = df[df["realized_return"].notna()]
    if len(valid) > 0:
        print("=" * 70)
        print("Per-signal summary (cumulative log-compound)")
        print("=" * 70)
        for sig in SIGNAL_SIGNS:
            sub = valid[valid["signal"] == sig].sort_values("rebalance_date")
            cum = float((1 + sub["realized_return"]).prod() - 1)
            n = len(sub)
            mean = float(sub["realized_return"].mean())
            std = float(sub["realized_return"].std())
            sharpe = mean / std * (12 ** 0.5) if std > 0 else 0.0
            print(
                f"  {sig:10s}: cum {cum*100:+8.2f}%  "
                f"mean {mean*100:+7.4f}%  Sharpe {sharpe:+.3f}  ({n} periods)"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())