"""Apply the performance metrics suite to the historical backtest results."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from axiom_fund.backtest.metrics import compute_performance_metrics


def main() -> int:
    summary_path = Path("data/cache/backtest_full/backtest_summary.parquet")
    if not summary_path.exists():
        print(f"ERROR: {summary_path} not found. Run scripts/run_full_backtest.py first.")
        return 1

    df = pd.read_parquet(summary_path)
    returns = df["realized_return"]
    metrics = compute_performance_metrics(returns)

    print("=" * 72)
    print("Axiom Fund — Performance Analysis (gross of costs)")
    print("=" * 72)
    print()
    print(f"Date range:           {metrics.start_date.date()} → {metrics.end_date.date()}")
    print(f"N periods:            {metrics.n_periods}")
    print(f"Years covered:        {metrics.years_covered:.2f}")
    print()

    print("Returns")
    print("-" * 72)
    print(f"  Cumulative return:    {metrics.cumulative_return*100:+8.2f}%")
    print(f"  Annualized return:    {metrics.annualized_return*100:+8.2f}%")
    print(f"  Annualized vol:       {metrics.annualized_vol*100:8.2f}%")
    print()

    print("Risk-Adjusted")
    print("-" * 72)
    sr = metrics.sharpe
    print(f"  Sharpe ratio:         {sr.sharpe:8.3f}")
    print(f"  Standard error:       {sr.standard_error:8.3f}")
    print(f"  95% CI:               [{sr.ci_low:+.3f}, {sr.ci_high:+.3f}]")
    if sr.ci_low <= 0 <= sr.ci_high:
        print("  → Sharpe is NOT statistically distinguishable from 0 at 95% level")
    else:
        print("  → Sharpe IS statistically different from 0 at 95% level")
    print()

    print("Distribution")
    print("-" * 72)
    m = metrics.moments
    print(f"  Mean (monthly):       {m.mean*100:+8.4f}%")
    print(f"  Std dev (monthly):    {m.std*100:8.4f}%")
    print(f"  Skewness:             {m.skew:+8.3f}")
    print(f"  Excess kurtosis:      {m.kurtosis:+8.3f}")
    print(f"  Best month:           {metrics.best_period*100:+8.2f}%")
    print(f"  Worst month:          {metrics.worst_period*100:+8.2f}%")
    print(f"  Hit rate:             {metrics.hit_rate*100:8.1f}%")
    print(f"  Avg win:              {metrics.avg_win*100:+8.4f}%")
    print(f"  Avg loss:             {metrics.avg_loss*100:+8.4f}%")
    print(f"  Win/loss ratio:       {metrics.win_loss_ratio:8.3f}")
    print()

    print("Drawdown Analysis")
    print("-" * 72)
    print(f"  Max drawdown:         {metrics.max_drawdown*100:+8.2f}%")
    print()

    eps = metrics.all_drawdown_episodes
    print(f"  Distinct drawdown episodes: {len(eps)}")
    print()

    # Show top 5 deepest episodes
    sorted_eps = sorted(eps, key=lambda ep: ep.max_depth)
    print(f"  {'Peak':<12}  {'Trough':<12}  {'Recovery':<12}  {'Depth':>8}  {'Dur':>4}  {'Rec':>4}")
    print(f"  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*8}  {'-'*4}  {'-'*4}")
    for ep in sorted_eps[:10]:
        recov = ep.recovery_date.date().isoformat() if ep.recovery_date else "N/A"
        rec_p = str(ep.recovery_periods) if ep.recovery_periods is not None else "N/A"
        print(
            f"  {ep.peak_date.date()}  {ep.trough_date.date()}  {recov:<12}  "
            f"{ep.max_depth*100:+7.2f}%  {ep.duration_periods:>4}  {rec_p:>4}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
