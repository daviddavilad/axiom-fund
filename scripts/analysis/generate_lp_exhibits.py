"""Generate Lazy Prices exhibits for the axiom-fund README.

Reads existing parquet outputs from Item 6a Lazy Prices analyses and
produces 4 PNG charts to docs/exhibits/. Runtime ~10 seconds.

Charts
------
  lazy_prices_01_size5_cumulative   — Flagship: NYSE Size5 EW vs VW L/S,
                                       cumulative log return, ChatGPT marker
  lazy_prices_02_size_heterogeneity — L/S annualized by NYSE size bucket
  lazy_prices_03_top10_mechanism    — Size5 VW: full vs excluding top-10
  lazy_prices_04_ai_era_regime      — Pre/post ChatGPT alpha bar chart
"""
# ruff: noqa: I001

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from axiom_fund import _warnings  # noqa: F401

from axiom_fund.backtest.exhibits import (
    COLOR_DRAWDOWN,
    COLOR_GROSS_3SIG,
    COLOR_GROSS_4SIG,
    COLOR_NET,
    COLOR_NEUTRAL,
    save_figure,
    setup_style,
)


BACKTEST_DIR = Path("data/cache/lazy_prices_backtest")
LS_SERIES_PATH = BACKTEST_DIR / "nyse_size5_ls_series.parquet"
SIZE_QUINTILE_PATH = BACKTEST_DIR / "size_quintile_nyse_breakpoints.parquet"
MEGA_CAPS_PATH = BACKTEST_DIR / "top_size5_mega_caps_diagnostic.parquet"
FF6_PATH = Path("data/cache/ff6_monthly.parquet")
EXHIBITS_DIR = Path("docs/exhibits")

CHATGPT_DATE = pd.Timestamp("2022-11-30")


# ----------------------------------------------------------------------
# Chart 1: Flagship cumulative L/S
# ----------------------------------------------------------------------

def chart_01_size5_cumulative() -> Path:
    """NYSE Size5 EW vs VW cumulative L/S over time.

    Shows the flagship EW result (steady positive drift) and the null-to-
    negative VW result (mega-cap concentration dominates), with the
    ChatGPT release marked as regime break.
    """
    series = pd.read_parquet(LS_SERIES_PATH)
    series["date"] = pd.to_datetime(series["date"])
    series = series.sort_values("date").reset_index(drop=True)

    cum_ew = (1.0 + series["ls_ew"]).cumprod()
    cum_vw = (1.0 + series["ls_vw"]).cumprod()

    fig, ax = plt.subplots()
    ax.plot(series["date"], (cum_ew - 1.0) * 100.0,
            label="Equal-weighted", color=COLOR_GROSS_3SIG, linewidth=2)
    ax.plot(series["date"], (cum_vw - 1.0) * 100.0,
            label="Value-weighted", color=COLOR_GROSS_4SIG, linewidth=2)

    ax.axvline(CHATGPT_DATE, color=COLOR_NEUTRAL, linestyle="--",
               linewidth=1, alpha=0.7)
    ax.text(CHATGPT_DATE, ax.get_ylim()[1] * 0.95, "  ChatGPT release",
            fontsize=9, color=COLOR_NEUTRAL, verticalalignment="top")

    ax.axhline(0, color="black", linewidth=0.5, alpha=0.3)

    ax.set_title("NYSE Size5 Lazy Prices L/S: Cumulative Return, 2020-2025")
    ax.set_xlabel("")
    ax.set_ylabel("Cumulative return (%)")
    ax.legend(loc="upper left")

    return save_figure(fig, EXHIBITS_DIR, "lazy_prices_01_size5_cumulative")


# ----------------------------------------------------------------------
# Chart 2: Size heterogeneity across NYSE size buckets
# ----------------------------------------------------------------------

def chart_02_size_heterogeneity() -> Path:
    """L/S Sharpe by NYSE size bucket (Size1-Size5).

    Shows the effect concentrates in Size5 (large-caps above NYSE p80),
    with weak-to-noisy signal in Size1-Size4. Sharpe on y-axis (not
    annualized return) so Size1's noise-driven extreme magnitude doesn't
    visually dominate. Size5 highlighted; others muted.
    """
    sort = pd.read_parquet(SIZE_QUINTILE_PATH)
    sort["date"] = pd.to_datetime(sort["date"])

    rows = []
    for size in range(1, 6):
        s = sort[sort.size_bucket == size]
        q1 = s[s.lp_quintile == 1].set_index("date")["mean"]
        q5 = s[s.lp_quintile == 5].set_index("date")["mean"]
        ls = (q1 - q5).dropna()
        if len(ls) == 0:
            rows.append({"bucket": size, "ann_ret": np.nan, "sharpe": np.nan})
            continue
        rows.append({
            "bucket": size,
            "ann_ret": ((1 + ls.mean()) ** 12 - 1) * 100,
            "sharpe": ls.mean() / ls.std() * np.sqrt(12),
        })
    df = pd.DataFrame(rows)

    colors = [COLOR_NEUTRAL if b < 5 else COLOR_GROSS_3SIG for b in df["bucket"]]

    fig, ax = plt.subplots()
    bars = ax.bar([f"Size{b}" for b in df["bucket"]], df["sharpe"],
                  color=colors, edgecolor="white", linewidth=1)

    for bar, val, ann in zip(bars, df["sharpe"], df["ann_ret"]):
        y = bar.get_height()
        va = "bottom" if y >= 0 else "top"
        offset = 0.03 if y >= 0 else -0.03
        ax.text(bar.get_x() + bar.get_width() / 2, y + offset,
                f"Sh={val:+.2f}\n({ann:+.1f}% ann)",
                ha="center", va=va, fontsize=9)

    ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)
    ax.set_title("Lazy Prices L/S by NYSE Size Bucket, 2020-2025")
    ax.set_ylabel("Annualized Sharpe ratio")
    ax.margins(y=0.20)

    return save_figure(fig, EXHIBITS_DIR, "lazy_prices_02_size_heterogeneity")


# ----------------------------------------------------------------------
# Chart 3: Top-10 mega-cap counterfactual
# ----------------------------------------------------------------------

def chart_03_top10_mechanism() -> Path:
    """NYSE Size5 VW L/S: full vs excluding top-10 mega-caps.

    Shows the mega-cap reversal mechanism under CMN-standard NYSE
    breakpoint methodology — excluding 10 firms per rebalance date
    flips the sign from -2.31% to +3.55% annualized.
    """
    labels = ["Full Size5 VW", "Excluding top-10\nmega-caps"]
    values = [-2.31, +3.55]
    colors = [COLOR_DRAWDOWN, COLOR_NET]

    fig, ax = plt.subplots()
    bars = ax.bar(labels, values, color=colors, edgecolor="white",
                  linewidth=1, width=0.5)

    for bar, val in zip(bars, values):
        y = bar.get_height()
        va = "bottom" if y >= 0 else "top"
        offset = 0.25 if y >= 0 else -0.25
        ax.text(bar.get_x() + bar.get_width() / 2, y + offset,
                f"{val:+.2f}%", ha="center", va=va, fontsize=11,
                fontweight="bold")

    ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)
    ax.set_title("Mega-Cap Reversal Mechanism: ~10 Firms Drive Size5 VW")
    ax.set_ylabel("Annualized L/S return (%)")
    ax.margins(y=0.30)

    return save_figure(fig, EXHIBITS_DIR, "lazy_prices_03_top10_mechanism")


# ----------------------------------------------------------------------
# Chart 4: AI-era regime split
# ----------------------------------------------------------------------

def chart_04_ai_era_regime() -> Path:
    """Pre/post ChatGPT alpha bar chart for EW and VW Size5 L/S.

    Shows the regime break — pre-ChatGPT both weightings had strongly
    positive alpha; post-ChatGPT EW weakened and VW reversed.
    """
    series = pd.read_parquet(LS_SERIES_PATH)
    ff6 = pd.read_parquet(FF6_PATH)

    series["date"] = pd.to_datetime(series["date"])
    ff6["date"] = pd.to_datetime(ff6["date"])
    for c in ["mkt_rf", "smb", "hml", "rmw", "cma", "mom", "rf"]:
        ff6[c] = ff6[c] / 100.0

    def _alpha_ann(sub: pd.DataFrame, weighting: str) -> float:
        col = f"ls_{weighting}"
        x = sub[col].dropna()
        if len(x) < 10:
            return np.nan
        ff6_local = ff6.set_index("date").loc[sub["date"].values,
            ["mkt_rf", "smb", "hml", "rmw", "cma", "mom"]].values
        y = x.values
        X = np.column_stack([np.ones(len(x)), ff6_local])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        return beta[0] * 12 * 100

    pre = series[series.date <= CHATGPT_DATE]
    post = series[series.date > CHATGPT_DATE]

    pre_ew = _alpha_ann(pre, "ew")
    pre_vw = _alpha_ann(pre, "vw")
    post_ew = _alpha_ann(post, "ew")
    post_vw = _alpha_ann(post, "vw")

    labels = ["Pre-ChatGPT\n(2020-01 to 2022-11)",
              "Post-ChatGPT\n(2022-12 to 2025-11)"]
    ew_vals = [pre_ew, post_ew]
    vw_vals = [pre_vw, post_vw]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots()
    bars1 = ax.bar(x - width / 2, ew_vals, width,
                   label="Equal-weighted", color=COLOR_GROSS_3SIG,
                   edgecolor="white", linewidth=1)
    bars2 = ax.bar(x + width / 2, vw_vals, width,
                   label="Value-weighted", color=COLOR_GROSS_4SIG,
                   edgecolor="white", linewidth=1)

    for bars in (bars1, bars2):
        for bar in bars:
            y = bar.get_height()
            va = "bottom" if y >= 0 else "top"
            offset = 0.4 if y >= 0 else -0.4
            ax.text(bar.get_x() + bar.get_width() / 2, y + offset,
                    f"{y:+.1f}%", ha="center", va=va, fontsize=9)

    ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("FF6-Adjusted Alpha: Pre vs Post ChatGPT")
    ax.set_ylabel("Annualized FF6 alpha (%)")
    ax.legend(loc="upper right")
    ax.margins(y=0.25)

    return save_figure(fig, EXHIBITS_DIR, "lazy_prices_04_ai_era_regime")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> None:
    setup_style()
    print("Generating Lazy Prices exhibits...")
    for chart_fn in (
        chart_01_size5_cumulative,
        chart_02_size_heterogeneity,
        chart_03_top10_mechanism,
        chart_04_ai_era_regime,
    ):
        path = chart_fn()
        print(f"  {path}")
    print("Done.")


if __name__ == "__main__":
    main()