"""Generate README exhibits for the Axiom Fund project.

Reads existing parquet outputs from backtest runs and IC analysis,
produces 7 PNG charts to docs/exhibits/.

Runtime: ~30 seconds total (charts are CPU-bound matplotlib calls;
no WRDS or heavy computation).

Charts:
  01_cumulative_returns       — 3-sig gross, 4-sig gross, 3-sig net
  02_drawdown                 — gross 3-signal drawdown over time
  03_rolling_sharpe           — 12-month rolling annualized Sharpe
  04_ic_per_signal            — bar chart, full sample, with t-stat
  05_ic_bull_vs_bear          — side-by-side IC, bull vs bear regimes
  06_yearly_returns_with_cost — net returns by year + cost decomposition
  07_cost_sensitivity         — net cumulative at 0%, 25%, 50% improvement
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
    COLOR_NEUTRAL,
    COLOR_NET,
    SIGNAL_COLORS,
    save_figure,
    setup_style,
)


BACKTEST_3SIG = Path("data/cache/backtest_full_top1000")
BACKTEST_4SIG = Path("data/cache/backtest_full_top1000_4sig")
BACKTEST_NO_RESMOM = Path("data/cache/backtest_full_top1000_no_resmom")
IC_DIR = Path("data/cache/ic_analysis_4sig")
EXHIBITS_DIR = Path("docs/exhibits")

# Signal display labels (more readable than column names)
SIGNAL_LABELS = {
    "z_gp": "Gross Profitability",
    "z_ivol": "Idiosyncratic Volatility",
    "z_resmom": "Residual Momentum",
    "z_pead": "PEAD (Earnings Drift)",
}


# ----------------------------------------------------------------------
# Data loaders
# ----------------------------------------------------------------------

def _load_3sig_summary() -> pd.DataFrame:
    df = pd.read_parquet(BACKTEST_3SIG / "backtest_summary.parquet")
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df.sort_index()


def _load_4sig_summary() -> pd.DataFrame:
    df = pd.read_parquet(BACKTEST_4SIG / "backtest_summary.parquet")
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df.sort_index()


def _load_no_resmom_summary() -> pd.DataFrame:
    df = pd.read_parquet(BACKTEST_NO_RESMOM / "backtest_summary.parquet")
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df.sort_index()


def _load_net_returns() -> pd.DataFrame:
    df = pd.read_parquet(BACKTEST_3SIG / "net_returns.parquet")
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df.sort_index()


def _load_ic_long() -> pd.DataFrame:
    df = pd.read_parquet(IC_DIR / "ic_long.parquet")
    df["rebalance_date"] = pd.to_datetime(df["rebalance_date"])
    return df


# ----------------------------------------------------------------------
# Chart 1: cumulative returns
# ----------------------------------------------------------------------

def chart_01_cumulative_returns() -> Path:
    s3 = _load_3sig_summary()
    s4 = _load_4sig_summary()
    net = _load_net_returns()

    cum_gross_3 = (1.0 + s3["realized_return"]).cumprod()
    cum_gross_4 = (1.0 + s4["realized_return"]).cumprod()
    cum_net = (1.0 + net["net_return"]).cumprod()

    fig, ax = plt.subplots()
    ax.plot(cum_gross_3.index, (cum_gross_3 - 1.0) * 100.0,
            label="Gross (3-signal)",
            color=COLOR_GROSS_3SIG, linewidth=1.8)
    ax.plot(cum_gross_4.index, (cum_gross_4 - 1.0) * 100.0,
            label="Gross (4-signal, with PEAD)",
            color=COLOR_GROSS_4SIG, linewidth=1.8, linestyle="--")
    ax.plot(cum_net.index, (cum_net - 1.0) * 100.0,
            label="Net of costs (3-signal, conservative)",
            color=COLOR_NET, linewidth=1.8)
    ax.axhline(0, color="black", linewidth=0.6, alpha=0.4)
    ax.set_title("Cumulative Returns, 2015 – 2024")
    ax.set_xlabel("Rebalance date")
    ax.set_ylabel("Cumulative return (%)")
    ax.legend(loc="upper left")
    return save_figure(fig, EXHIBITS_DIR, "01_cumulative_returns")


# ----------------------------------------------------------------------
# Chart 2: drawdown
# ----------------------------------------------------------------------

def chart_02_drawdown() -> Path:
    s3 = _load_3sig_summary()
    cum = (1.0 + s3["realized_return"]).cumprod()
    running_peak = cum.cummax()
    drawdown_pct = (cum / running_peak - 1.0) * 100.0

    fig, ax = plt.subplots()
    ax.fill_between(drawdown_pct.index, drawdown_pct.values, 0,
                     color=COLOR_DRAWDOWN, alpha=0.35, linewidth=0)
    ax.plot(drawdown_pct.index, drawdown_pct.values,
             color=COLOR_DRAWDOWN, linewidth=1.4)
    ax.axhline(0, color="black", linewidth=0.6, alpha=0.4)
    max_dd = float(drawdown_pct.min())
    max_dd_date = drawdown_pct.idxmin()
    ax.axhline(max_dd, color=COLOR_DRAWDOWN, linewidth=0.8,
               linestyle=":", alpha=0.6)
    ax.text(drawdown_pct.index[5], max_dd + 0.6,
             f"Max drawdown: {max_dd:.1f}% on {max_dd_date.date()}",
             fontsize=10, color=COLOR_DRAWDOWN, va="bottom")
    ax.set_title("Drawdown (gross, 3-signal)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")
    return save_figure(fig, EXHIBITS_DIR, "02_drawdown")


# ----------------------------------------------------------------------
# Chart 3: rolling 12-month Sharpe
# ----------------------------------------------------------------------

def chart_03_rolling_sharpe() -> Path:
    s3 = _load_3sig_summary()
    rets = s3["realized_return"]
    # 12-period rolling Sharpe, annualized (×√12)
    rolling_mean = rets.rolling(window=12, min_periods=12).mean()
    rolling_std = rets.rolling(window=12, min_periods=12).std()
    rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(12.0)
    rolling_sharpe = rolling_sharpe.dropna()

    full_sample_sharpe = float(rets.mean() / rets.std() * np.sqrt(12.0))

    fig, ax = plt.subplots()
    ax.plot(rolling_sharpe.index, rolling_sharpe.values,
             color=COLOR_GROSS_3SIG, linewidth=1.6,
             label="12-month rolling Sharpe")
    ax.axhline(0, color="black", linewidth=0.6, alpha=0.4)
    ax.axhline(full_sample_sharpe, color=COLOR_NEUTRAL,
               linewidth=1.2, linestyle="--",
               label=f"Full-sample Sharpe = {full_sample_sharpe:.2f}")
    ax.set_title("Rolling 12-Month Sharpe (gross, 3-signal)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Annualized Sharpe")
    ax.legend(loc="lower left")
    return save_figure(fig, EXHIBITS_DIR, "03_rolling_sharpe")


# ----------------------------------------------------------------------
# Chart 4: IC per signal, full sample
# ----------------------------------------------------------------------

def chart_04_ic_per_signal() -> Path:
    ic = _load_ic_long().dropna(subset=["ic_spearman"])
    # Order: GP, IVol, ResMom, PEAD (matches README table order)
    signal_order = ["z_gp", "z_ivol", "z_resmom", "z_pead"]
    stats = []
    for sig in signal_order:
        sub = ic[ic["signal"] == sig]
        n = len(sub)
        mean = float(sub["ic_spearman"].mean())
        std = float(sub["ic_spearman"].std())
        se = std / np.sqrt(n) if n > 0 else float("nan")
        stats.append({"signal": sig, "mean": mean, "se": se, "n": n})
    stats_df = pd.DataFrame(stats)

    fig, ax = plt.subplots()
    x_pos = np.arange(len(stats_df))
    colors = [SIGNAL_COLORS[s] for s in stats_df["signal"]]
    bars = ax.bar(
        x_pos, stats_df["mean"] * 100.0,
        yerr=stats_df["se"] * 100.0 * 1.96,  # 95% CI
        color=colors, alpha=0.85,
        capsize=6, error_kw={"linewidth": 1.5, "ecolor": "#333333"},
    )
    ax.axhline(0, color="black", linewidth=0.6, alpha=0.4)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([SIGNAL_LABELS[s] for s in stats_df["signal"]],
                        rotation=0, ha="center", fontsize=10)
    ax.set_title("Information Coefficient per Signal (116 periods, with 95% CI)")
    ax.set_ylabel("Mean Spearman IC × 100")
    # Annotate t-stats above each bar
    for i, row in stats_df.iterrows():
        t = row["mean"] / row["se"] if row["se"] > 0 else 0
        y_pos = row["mean"] * 100.0 + (row["se"] * 100.0 * 1.96) + 0.3
        if row["mean"] < 0:
            y_pos = row["mean"] * 100.0 - (row["se"] * 100.0 * 1.96) - 0.6
        ax.text(i, y_pos, f"t = {t:.2f}", ha="center", fontsize=10,
                fontweight="bold")
    return save_figure(fig, EXHIBITS_DIR, "04_ic_per_signal")


# ----------------------------------------------------------------------
# Chart 5: IC bull vs bear regimes
# ----------------------------------------------------------------------

def chart_05_ic_bull_vs_bear() -> Path:
    ic = _load_ic_long().dropna(subset=["ic_spearman"]).copy()
    ic["year"] = pd.to_datetime(ic["rebalance_date"]).dt.year
    # Bull = year where cross-signal mean IC is positive (matches earlier analysis)
    yearly_avg = ic.groupby("year")["ic_spearman"].mean()
    bull_years = set(yearly_avg[yearly_avg > 0].index)
    ic["regime"] = ic["year"].apply(
        lambda y: "Bull" if y in bull_years else "Bear"
    )

    signal_order = ["z_gp", "z_ivol", "z_resmom", "z_pead"]
    bull_means = []
    bear_means = []
    for sig in signal_order:
        bull = ic[(ic["signal"] == sig) & (ic["regime"] == "Bull")]
        bear = ic[(ic["signal"] == sig) & (ic["regime"] == "Bear")]
        bull_means.append(float(bull["ic_spearman"].mean()))
        bear_means.append(float(bear["ic_spearman"].mean()))

    fig, ax = plt.subplots()
    x_pos = np.arange(len(signal_order))
    bar_width = 0.38
    ax.bar(x_pos - bar_width / 2, [v * 100.0 for v in bull_means],
           bar_width, label="Bull regime (92 periods)",
           color="#2ca02c", alpha=0.85)
    ax.bar(x_pos + bar_width / 2, [v * 100.0 for v in bear_means],
           bar_width, label="Bear regime (24 periods)",
           color="#d62728", alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.6, alpha=0.4)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([SIGNAL_LABELS[s] for s in signal_order],
                       rotation=0, ha="center", fontsize=10)
    ax.set_title("Information Coefficient by Regime (signal flips in bear)")
    ax.set_ylabel("Mean Spearman IC × 100")
    ax.legend(loc="upper right")
    return save_figure(fig, EXHIBITS_DIR, "05_ic_bull_vs_bear")


# ----------------------------------------------------------------------
# Chart 6: yearly returns with cost decomposition
# ----------------------------------------------------------------------

def chart_06_yearly_returns_with_cost() -> Path:
    net = _load_net_returns().copy()
    net["year"] = net.index.year

    # Aggregate per year
    yearly = net.groupby("year").agg(
        gross_compound=("gross_return", lambda s: (1 + s).prod() - 1),
        net_compound=("net_return", lambda s: (1 + s).prod() - 1),
        cost_total_bps=("cost_bps", "sum"),
    )
    # Convert to annualized cost as a "return drag"
    yearly["cost_drag_pct"] = yearly["cost_total_bps"] / 100.0  # bps -> pct

    fig, ax = plt.subplots()
    x_pos = np.arange(len(yearly))
    bar_width = 0.38
    ax.bar(x_pos - bar_width / 2, yearly["gross_compound"] * 100.0,
            bar_width, label="Gross return", color=COLOR_GROSS_3SIG, alpha=0.85)
    ax.bar(x_pos + bar_width / 2, yearly["net_compound"] * 100.0,
            bar_width, label="Net return (after costs)",
            color=COLOR_NET, alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.6, alpha=0.4)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(y) for y in yearly.index], rotation=0)
    ax.set_title("Year-by-Year Returns: Gross vs Net (3-signal)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Annual return (%)")
    ax.legend(loc="upper right")
    return save_figure(fig, EXHIBITS_DIR, "06_yearly_returns")


# ----------------------------------------------------------------------
# Chart 7: cost sensitivity (net cumulative at 0%, 25%, 50% execution improvement)
# ----------------------------------------------------------------------

def chart_07_cost_sensitivity() -> Path:
    net = _load_net_returns().copy()

    # Net at conservative (0% improvement): use existing net_return directly
    # Net at 25% improvement: subtract 75% of cost_bps from gross
    # Net at 50% improvement: subtract 50% of cost_bps from gross
    gross = net["gross_return"]
    cost = net["cost_bps"] / 10000.0  # bps -> decimal

    net_0 = gross - cost
    net_25 = gross - cost * 0.75
    net_50 = gross - cost * 0.50

    cum_gross = (1 + gross).cumprod()
    cum_net_0 = (1 + net_0).cumprod()
    cum_net_25 = (1 + net_25).cumprod()
    cum_net_50 = (1 + net_50).cumprod()

    final_gross = (cum_gross.iloc[-1] - 1) * 100.0
    final_0 = (cum_net_0.iloc[-1] - 1) * 100.0
    final_25 = (cum_net_25.iloc[-1] - 1) * 100.0
    final_50 = (cum_net_50.iloc[-1] - 1) * 100.0

    fig, ax = plt.subplots()
    ax.plot(cum_gross.index, (cum_gross - 1) * 100.0,
            label=f"Gross (+{final_gross:.1f}%)",
            color=COLOR_GROSS_3SIG, linewidth=1.6, linestyle="--")
    ax.plot(cum_net_50.index, (cum_net_50 - 1) * 100.0,
            label=f"50% execution improvement (+{final_50:.1f}%)",
            color="#2ca02c", linewidth=1.8)
    ax.plot(cum_net_25.index, (cum_net_25 - 1) * 100.0,
            label=f"25% execution improvement (+{final_25:.1f}%)",
            color="#ff7f0e", linewidth=1.8)
    ax.plot(cum_net_0.index, (cum_net_0 - 1) * 100.0,
            label=f"Conservative (+{final_0:.1f}%)",
            color="#d62728", linewidth=1.8)
    ax.axhline(0, color="black", linewidth=0.6, alpha=0.4)
    ax.set_title("Net Cumulative Returns under Different Execution Assumptions")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative return (%)")
    ax.legend(loc="upper left")
    return save_figure(fig, EXHIBITS_DIR, "07_cost_sensitivity")


# ----------------------------------------------------------------------
# Chart 8: three-variant comparison (3-sig, 4-sig, no-ResMom)
# ----------------------------------------------------------------------

def chart_08_variant_comparison() -> Path:
    s3 = _load_3sig_summary()
    s4 = _load_4sig_summary()
    s_nrm = _load_no_resmom_summary()

    cum_3 = (1.0 + s3["realized_return"]).cumprod()
    cum_4 = (1.0 + s4["realized_return"]).cumprod()
    cum_nrm = (1.0 + s_nrm["realized_return"]).cumprod()

    final_3 = (cum_3.iloc[-1] - 1.0) * 100.0
    final_4 = (cum_4.iloc[-1] - 1.0) * 100.0
    final_nrm = (cum_nrm.iloc[-1] - 1.0) * 100.0

    fig, ax = plt.subplots()
    ax.plot(cum_3.index, (cum_3 - 1.0) * 100.0,
            label=f"3-signal: GP+IVol+ResMom  (+{final_3:.1f}%, Sharpe 0.79)",
            color=COLOR_GROSS_3SIG, linewidth=1.8)
    ax.plot(cum_4.index, (cum_4 - 1.0) * 100.0,
            label=f"4-signal: + PEAD  (+{final_4:.1f}%, Sharpe 0.78)",
            color=COLOR_GROSS_4SIG, linewidth=1.8, linestyle="--")
    ax.plot(cum_nrm.index, (cum_nrm - 1.0) * 100.0,
            label=f"No-ResMom: GP+IVol+PEAD  (+{final_nrm:.1f}%, Sharpe 0.82)",
            color=COLOR_DRAWDOWN, linewidth=1.8)
    ax.axhline(0, color="black", linewidth=0.6, alpha=0.4)
    ax.set_title("Cumulative Returns — Three Composite Variants")
    ax.set_xlabel("Rebalance date")
    ax.set_ylabel("Cumulative return (%)")
    ax.legend(loc="upper left")
    return save_figure(fig, EXHIBITS_DIR, "08_variant_comparison")


# ----------------------------------------------------------------------
# Chart 9: in-sample vs holdout split (boundary at 2022-12-31)
# ----------------------------------------------------------------------

def chart_09_in_sample_vs_holdout() -> Path:
    """Cumulative returns with a vertical boundary at the holdout cutoff.

    The boundary marks where in-sample (2015-2022, 94 periods) ends and
    the partially-contaminated holdout window (2023-2024, 22 periods)
    begins. The contamination is acknowledged in
    docs/holdout_test_design.md.
    """
    import pandas as pd  # local import to keep top-level light

    s3 = _load_3sig_summary()
    s4 = _load_4sig_summary()
    s_nrm = _load_no_resmom_summary()

    cum_3 = (1.0 + s3["realized_return"]).cumprod()
    cum_4 = (1.0 + s4["realized_return"]).cumprod()
    cum_nrm = (1.0 + s_nrm["realized_return"]).cumprod()

    fig, ax = plt.subplots()
    ax.plot(cum_3.index, (cum_3 - 1.0) * 100.0,
            label="3-signal: GP+IVol+ResMom",
            color=COLOR_GROSS_3SIG, linewidth=1.8)
    ax.plot(cum_4.index, (cum_4 - 1.0) * 100.0,
            label="4-signal: + PEAD",
            color=COLOR_GROSS_4SIG, linewidth=1.8, linestyle="--")
    ax.plot(cum_nrm.index, (cum_nrm - 1.0) * 100.0,
            label="No-ResMom: GP+IVol+PEAD",
            color=COLOR_DRAWDOWN, linewidth=1.8)
    ax.axhline(0, color="black", linewidth=0.6, alpha=0.4)

    # Boundary marker at 2022-12-31
    boundary = pd.Timestamp("2022-12-31")
    ax.axvline(boundary, color="#666666", linewidth=1.5, linestyle=":",
                alpha=0.85, label="Holdout boundary (2022-12-31)")

    # Region shading for visual emphasis
    y_min, y_max = ax.get_ylim()
    ax.axvspan(
        boundary, cum_3.index[-1],
        alpha=0.08, color="#666666", zorder=0,
    )

    # Region annotations — placed near bottom to avoid the legend
    y_min, _ = ax.get_ylim()
    y_label = y_min + (y_max - y_min) * 0.03
    ax.text(pd.Timestamp("2018-06-30"), y_label,
            "← in-sample (94 periods)",
            fontsize=10, color="#666666", ha="right", style="italic")
    ax.text(pd.Timestamp("2023-09-30"), y_label,
            "holdout (22 periods) →",
            fontsize=10, color="#666666", ha="center", style="italic")

    ax.set_title("Cumulative Returns: In-Sample vs Holdout Split")
    ax.set_xlabel("Rebalance date")
    ax.set_ylabel("Cumulative return (%)")
    ax.legend(loc="upper left", framealpha=0.9)
    return save_figure(fig, EXHIBITS_DIR, "09_in_sample_vs_holdout")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> int:
    setup_style()
    print(f"Output directory: {EXHIBITS_DIR}")
    print()

    charts = [
        ("01_cumulative_returns", chart_01_cumulative_returns),
        ("02_drawdown", chart_02_drawdown),
        ("03_rolling_sharpe", chart_03_rolling_sharpe),
        ("04_ic_per_signal", chart_04_ic_per_signal),
        ("05_ic_bull_vs_bear", chart_05_ic_bull_vs_bear),
        ("06_yearly_returns", chart_06_yearly_returns_with_cost),
        ("07_cost_sensitivity", chart_07_cost_sensitivity),
        ("08_variant_comparison", chart_08_variant_comparison),
        ("09_in_sample_vs_holdout", chart_09_in_sample_vs_holdout),
    ]
    for name, fn in charts:
        path = fn()
        print(f"  ✓ {path}")

    print()
    print(f"Generated {len(charts)} exhibits")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())