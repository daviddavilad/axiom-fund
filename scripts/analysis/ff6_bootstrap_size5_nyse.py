"""Block-bootstrap CI for NYSE Size5 L/S raw Sharpe and FF6 alpha.

Companion to ff6_bootstrap_size4.py. CLI --weighting {ew, vw} loads
the appropriate series from nyse_size5_ls_series.parquet.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


SERIES_PATH = Path("data/cache/lazy_prices_backtest/nyse_size5_ls_series.parquet")
FF6_PATH = Path("data/cache/ff6_monthly.parquet")
BLOCK_SIZES = [3, 4, 6, 8]
N_RESAMPLES = 10_000
SEED = 42
ANNUAL_FACTOR = np.sqrt(12)


def _sample_block_indices(n: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    n_blocks = -(-n // block_size)
    starts = rng.integers(0, n - block_size + 1, size=n_blocks)
    blocks = [np.arange(s, s + block_size) for s in starts]
    return np.concatenate(blocks)[:n]


def _bootstrap_stats(y, X, block_size, n_resamples, seed):
    n = len(y)
    rng = np.random.default_rng(seed)
    sharpes = np.empty(n_resamples)
    alphas = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = _sample_block_indices(n, block_size, rng)
        y_b, X_b = y[idx], X[idx]
        sharpes[i] = y_b.mean() / y_b.std(ddof=1)
        beta, *_ = np.linalg.lstsq(X_b, y_b, rcond=None)
        alphas[i] = beta[0]
    return sharpes, alphas


def _two_sided_p(x):
    return 2 * min((x <= 0).mean(), (x >= 0).mean())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--weighting", choices=["ew", "vw"], default="ew")
    args = p.parse_args()

    print(f"Loading NYSE Size5 L/S ({args.weighting.upper()})...")
    series = pd.read_parquet(SERIES_PATH)
    ff6 = pd.read_parquet(FF6_PATH)

    series["date"] = pd.to_datetime(series["date"])
    ff6["date"] = pd.to_datetime(ff6["date"])

    col = f"ls_{args.weighting}"
    ls = series.set_index("date")[col].dropna().rename("ls_return")

    for c in ["mkt_rf", "smb", "hml", "rmw", "cma", "mom", "rf"]:
        ff6[c] = ff6[c] / 100.0

    joined = pd.DataFrame({"ls": ls}).join(
        ff6.set_index("date")[["mkt_rf", "smb", "hml", "rmw", "cma", "mom"]],
        how="inner",
    )
    print(f"  N months: {len(joined)}")

    y = joined["ls"].values
    X = np.column_stack([
        np.ones(len(joined)),
        joined[["mkt_rf", "smb", "hml", "rmw", "cma", "mom"]].values,
    ])

    beta_full, *_ = np.linalg.lstsq(X, y, rcond=None)
    alpha_point = beta_full[0]
    sharpe_point = float(y.mean() / y.std(ddof=1))

    print()
    print("=" * 90)
    print(f"Block bootstrap CI (NYSE Size5 {args.weighting.upper()}), N_resamples = {N_RESAMPLES:,}")
    print("=" * 90)
    print(f"Point estimates:")
    print(f"  Annualized Sharpe: {sharpe_point * ANNUAL_FACTOR:+.4f}")
    print(f"  Annualized alpha:  {alpha_point * 12 * 100:+.4f}%")
    print()

    print(f"{'Block':<8} {'Sharpe 95% CI (ann)':>22} {'Sharpe p':>10}"
          f"     {'Alpha 95% CI (ann %)':>26} {'Alpha p':>10}")
    print("-" * 90)

    for bs in BLOCK_SIZES:
        sharpes, alphas = _bootstrap_stats(y, X, bs, N_RESAMPLES, SEED + bs)
        s_lo = np.percentile(sharpes, 2.5) * ANNUAL_FACTOR
        s_hi = np.percentile(sharpes, 97.5) * ANNUAL_FACTOR
        a_lo = np.percentile(alphas, 2.5) * 12 * 100
        a_hi = np.percentile(alphas, 97.5) * 12 * 100
        p_sharpe = _two_sided_p(sharpes)
        p_alpha = _two_sided_p(alphas)

        print(f"bs={bs:<5} [{s_lo:>+7.3f}, {s_hi:>+7.3f}]"
              f"    {p_sharpe:>7.4f}"
              f"     [{a_lo:>+7.3f}%, {a_hi:>+7.3f}%]"
              f"    {p_alpha:>7.4f}")

    out = Path(f"data/cache/lazy_prices_backtest/ff6_bootstrap_size5_nyse_{args.weighting}.parquet")
    result = pd.DataFrame({
        "weighting": [args.weighting],
        "n_months": [len(joined)],
        "sharpe_point_ann": [sharpe_point * ANNUAL_FACTOR],
        "alpha_point_ann": [alpha_point * 12],
    })
    result.to_parquet(out, index=False)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()