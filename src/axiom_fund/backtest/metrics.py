"""Performance metrics for backtest evaluation.

All functions take a pd.Series of period returns (typically monthly,
indexed by rebalance date) and return scalars or small dataclasses.
The aggregator `compute_performance_metrics` packages everything into
a single PerformanceMetrics dataclass.

Conventions
-----------
- Returns are simple period returns (not log returns) in decimal form
  (0.01 = 1%).
- Annualization assumes ~12 periods per year for monthly data; this
  is parameterized via `periods_per_year`.
- "Return" annualizes geometrically: (1 + cum)^(1/years) - 1
- "Vol" annualizes arithmetically: std × √periods_per_year
- Sharpe uses geometric return in numerator, arithmetic vol in
  denominator. Risk-free rate is 0 by default (gross-of-cost analysis).

Sharpe confidence interval
--------------------------
The 95% CI uses the asymptotic standard error for monthly Sharpe:

    SE(SR) ≈ √((1 + 0.5 × SR²) / N)

This is from Lo (2002) "The Statistics of Sharpe Ratios". Valid for
N ≥ 30 or so, which we satisfy.

References
----------
Lo, A. W. (2002). "The Statistics of Sharpe Ratios."
    Financial Analysts Journal, 58(4), 36-52.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Standard annualization for monthly data
_DEFAULT_PERIODS_PER_YEAR: int = 12


# ----------------------------------------------------------------------
# Result dataclasses
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class SharpeResult:
    """Annualized Sharpe ratio with confidence interval.

    Attributes
    ----------
    sharpe : float
        Annualized Sharpe ratio (geometric return / arithmetic vol).
    standard_error : float
        Asymptotic standard error of the Sharpe estimate.
    ci_low : float
    ci_high : float
        95% asymptotic confidence interval bounds (sharpe ± 1.96 × SE).
    n_periods : int
        Number of return observations used.
    """

    sharpe: float
    standard_error: float
    ci_low: float
    ci_high: float
    n_periods: int


@dataclass(frozen=True)
class DrawdownEpisode:
    """One distinct drawdown episode (peak → trough → recovery).

    Attributes
    ----------
    peak_date : pd.Timestamp
        Date of the equity peak before the drawdown.
    trough_date : pd.Timestamp
        Date of the lowest equity point during the drawdown.
    recovery_date : pd.Timestamp | None
        Date when equity first equaled or exceeded the peak again.
        None if the drawdown never recovered within the sample.
    max_depth : float
        Peak-to-trough drawdown depth (negative number, e.g., -0.0886).
    duration_periods : int
        Number of periods from peak to trough.
    recovery_periods : int | None
        Number of periods from trough to recovery (None if no recovery).
    """

    peak_date: pd.Timestamp
    trough_date: pd.Timestamp
    recovery_date: pd.Timestamp | None
    max_depth: float
    duration_periods: int
    recovery_periods: int | None


@dataclass(frozen=True)
class ReturnMoments:
    """First four moments of the return distribution.

    Attributes
    ----------
    mean : float
        Sample mean.
    std : float
        Sample standard deviation (ddof=1).
    skew : float
        Sample skewness.
    kurtosis : float
        Excess kurtosis (Fisher's definition; 0 for normal).
    """

    mean: float
    std: float
    skew: float
    kurtosis: float


@dataclass(frozen=True)
class PerformanceMetrics:
    """Full performance metric suite for a return series."""

    n_periods: int
    periods_per_year: int
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    years_covered: float

    # Returns
    cumulative_return: float
    annualized_return: float
    annualized_vol: float

    # Risk-adjusted
    sharpe: SharpeResult

    # Distribution
    moments: ReturnMoments
    hit_rate: float
    avg_win: float
    avg_loss: float
    win_loss_ratio: float
    best_period: float
    worst_period: float

    # Drawdown
    max_drawdown: float
    max_drawdown_episode: DrawdownEpisode | None
    all_drawdown_episodes: tuple[DrawdownEpisode, ...]


# ----------------------------------------------------------------------
# Core metric functions
# ----------------------------------------------------------------------


def compute_annualized_return(
    returns: pd.Series,
    periods_per_year: int = _DEFAULT_PERIODS_PER_YEAR,
) -> float:
    """Geometric annualized return.

    Computed as (1 + cumulative_return)^(periods_per_year / n_periods) - 1.

    This uses the number of return observations rather than calendar-day
    span: n monthly returns represents n months of investment exposure
    regardless of whether the index timestamps span n or n-1 months.
    """
    if len(returns) == 0:
        return float("nan")
    prod_val: float = float((1 + returns).prod())  # type: ignore[arg-type]
    cum = prod_val - 1.0
    n = len(returns)
    return float((1 + cum) ** (periods_per_year / n) - 1.0)


def compute_annualized_vol(
    returns: pd.Series,
    periods_per_year: int = _DEFAULT_PERIODS_PER_YEAR,
) -> float:
    """Arithmetic annualized volatility: std × √periods_per_year."""
    if len(returns) < 2:
        return float("nan")
    return float(returns.std() * np.sqrt(periods_per_year))


def compute_sharpe(
    returns: pd.Series,
    periods_per_year: int = _DEFAULT_PERIODS_PER_YEAR,
    risk_free_rate: float = 0.0,
) -> SharpeResult:
    """Annualized Sharpe ratio with 95% confidence interval.

    Uses geometric return in the numerator, arithmetic vol in the
    denominator. Risk-free rate is in annualized form (e.g., 0.04 for 4%).

    Confidence interval via Lo (2002) asymptotic standard error:
        SE(SR) ≈ √((1 + 0.5 × SR²) / N)
    """
    if len(returns) < 2:
        return SharpeResult(
            sharpe=float("nan"),
            standard_error=float("nan"),
            ci_low=float("nan"),
            ci_high=float("nan"),
            n_periods=len(returns),
        )

    ann_return = compute_annualized_return(returns, periods_per_year)
    ann_vol = compute_annualized_vol(returns, periods_per_year)

    if ann_vol < 1e-10:
        return SharpeResult(
            sharpe=float("nan"),
            standard_error=float("nan"),
            ci_low=float("nan"),
            ci_high=float("nan"),
            n_periods=len(returns),
        )

    sr = (ann_return - risk_free_rate) / ann_vol
    n = len(returns)
    se = float(np.sqrt((1 + 0.5 * sr**2) / n))
    return SharpeResult(
        sharpe=float(sr),
        standard_error=se,
        ci_low=float(sr - 1.96 * se),
        ci_high=float(sr + 1.96 * se),
        n_periods=n,
    )


def compute_hit_rate(returns: pd.Series) -> float:
    """Fraction of periods with strictly positive return."""
    if len(returns) == 0:
        return float("nan")
    return float((returns > 0).mean())


def compute_return_moments(returns: pd.Series) -> ReturnMoments:
    """Compute first four moments of the return distribution."""
    if len(returns) < 2:
        return ReturnMoments(
            mean=float("nan"),
            std=float("nan"),
            skew=float("nan"),
            kurtosis=float("nan"),
        )
    return ReturnMoments(
        mean=float(returns.mean()),
        std=float(returns.std()),
        skew=float(returns.skew()),  # type: ignore[arg-type]  
        kurtosis=float(returns.kurtosis()),  # type: ignore[arg-type]
    )


def compute_drawdown_episodes(
    returns: pd.Series,
) -> tuple[DrawdownEpisode, ...]:
    """Identify all distinct drawdown episodes in the return series.

    A drawdown episode starts when equity falls below its running max
    and ends when equity recovers to (or exceeds) the prior peak. If
    equity never recovers by the end of the sample, the episode has
    recovery_date=None.

    Returns
    -------
    tuple[DrawdownEpisode, ...]
        All episodes, sorted by start date. Empty tuple if no drawdowns.
    """
    if len(returns) == 0:
        return ()

    cum = (1 + returns).cumprod()
    running_max = cum.cummax()
    in_drawdown = cum < running_max

    if not in_drawdown.any():
        return ()

    episodes: list[DrawdownEpisode] = []
    i = 0
    n = len(cum)
    while i < n:
        if not in_drawdown.iloc[i]:
            i += 1
            continue

        # Find the peak (last date before this drawdown started)
        peak_idx = i - 1
        peak_value = running_max.iloc[i]

        # Find the trough and end of this episode
        j = i
        while j < n and cum.iloc[j] < peak_value:
            j += 1

        # Episode runs from i to j-1 (exclusive of j)
        episode_slice = cum.iloc[i - 1 : j]  # include the peak point
        if len(episode_slice) < 2:
            i = j
            continue

        trough_local_idx = episode_slice.idxmin()
        trough_value = float(episode_slice.min())
        depth = trough_value / peak_value - 1

        peak_date = cum.index[peak_idx]
        trough_date = pd.Timestamp(trough_local_idx)

        if j >= n:
            recovery_date: pd.Timestamp | None = None
            recovery_periods: int | None = None
        else:
            recovery_date = cum.index[j]
            trough_pos_int = int(cum.index.get_indexer(pd.Index([trough_date]))[0])
            recovery_periods = int(j - trough_pos_int)

        peak_pos = peak_idx
        trough_pos_int = int(cum.index.get_indexer(pd.Index([trough_date]))[0])
        duration_periods = int(trough_pos_int - peak_pos)

        episodes.append(DrawdownEpisode(
            peak_date=peak_date,
            trough_date=trough_date,
            recovery_date=recovery_date,
            max_depth=float(depth),
            duration_periods=duration_periods,
            recovery_periods=recovery_periods,
        ))

        i = j

    return tuple(episodes)


def compute_max_drawdown(returns: pd.Series) -> float:
    """Single deepest drawdown depth as a negative float."""
    episodes = compute_drawdown_episodes(returns)
    if len(episodes) == 0:
        return 0.0
    return min(ep.max_depth for ep in episodes)


# ----------------------------------------------------------------------
# Aggregator
# ----------------------------------------------------------------------


def compute_performance_metrics(
    returns: pd.Series,
    periods_per_year: int = _DEFAULT_PERIODS_PER_YEAR,
    risk_free_rate: float = 0.0,
) -> PerformanceMetrics:
    """Compute the full performance metric suite.

    Convenience aggregator that calls all the small functions and
    packages their results.
    """
    if len(returns) == 0:
        raise ValueError("returns series is empty")

    moments = compute_return_moments(returns)
    sharpe = compute_sharpe(returns, periods_per_year, risk_free_rate)
    episodes = compute_drawdown_episodes(returns)

    # Win/loss stats
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
    avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0
    win_loss_ratio = (
        float(avg_win / -avg_loss) if avg_loss < 0 else float("inf")
    )

    # Cumulative return
    cum_prod: float = float((1 + returns).prod())  # type: ignore[arg-type]
    cum_return = cum_prod - 1.0

    # Max drawdown episode (the deepest)
    if len(episodes) > 0:
        max_dd_episode_resolved: DrawdownEpisode = min(
            episodes, key=lambda ep: ep.max_depth
        )
        max_dd_episode: DrawdownEpisode | None = max_dd_episode_resolved
        max_dd = max_dd_episode_resolved.max_depth
    else:
        max_dd_episode = None
        max_dd = 0.0

    # Date span — use n_periods / periods_per_year for consistency with
    # compute_annualized_return rather than date arithmetic
    start = pd.Timestamp(returns.index[0])
    end = pd.Timestamp(returns.index[-1])
    years = float(len(returns)) / periods_per_year

    return PerformanceMetrics(
        n_periods=len(returns),
        periods_per_year=periods_per_year,
        start_date=start,
        end_date=end,
        years_covered=years,
        cumulative_return=cum_return,
        annualized_return=compute_annualized_return(returns, periods_per_year),
        annualized_vol=compute_annualized_vol(returns, periods_per_year),
        sharpe=sharpe,
        moments=moments,
        hit_rate=compute_hit_rate(returns),
        avg_win=avg_win,
        avg_loss=avg_loss,
        win_loss_ratio=win_loss_ratio,
        best_period=float(returns.max()),
        worst_period=float(returns.min()),
        max_drawdown=max_dd,
        max_drawdown_episode=max_dd_episode,
        all_drawdown_episodes=episodes,
    )
