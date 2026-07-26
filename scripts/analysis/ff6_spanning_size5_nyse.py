"""FF6 spanning regression on NYSE Size5 L/S returns.

Companion to ff6_spanning_size4.py. Tests whether the NYSE-breakpoint
Size5 CMN peak (raw Sharpe +1.177 EW at N=71) has incremental alpha
after controlling for FF5 + Momentum.

Robustness of the in-sample Size4 finding: under CMN-standard NYSE
breakpoints, the CMN peak moves to Size5 with Sharpe +1.177 (up from
Size4 in-sample +0.638). This script tests whether the stronger raw
signal is factor-orthogonal or absorbed by known exposures.

Model:
  NYSE_Size5_LS ~ alpha + Mkt-RF + SMB + HML + RMW + CMA + MOM

Standard errors: HAC (Newey-West) with lags 4/6/8 for sensitivity.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm


SORT_PATH = Path("data/cache/lazy_prices_backtest/size_quintile_nyse_breakpoints.parquet")
FF6_PATH = Path("data/cache/ff6_monthly.parquet")
SIZE_BUCKET = 5
HAC_LAGS = [4, 6, 8]


def main() -> None:
    print(f"Loading NYSE Size{SIZE_BUCKET} L/S series (EW)...")
    sort = pd.read_parquet(SORT_PATH)
    ff6 = pd.read_parquet(FF6_PATH)

    sort["date"] = pd.to_datetime(sort["date"])
    ff6["date"] = pd.to_datetime(ff6["date"])

    s = sort[sort.size_bucket == SIZE_BUCKET].copy()
    q1 = s[s.lp_quintile == 1].set_index("date")["mean"].rename("q1")
    q5 = s[s.lp_quintile == 5].set_index("date")["mean"].rename("q5")
    ls = (q1 - q5).rename("ls_return").dropna()
    print(f"  Series: {len(ls)} months, {ls.index.min()} to {ls.index.max()}")

    ff6_dec = ff6.copy()
    for c in ["mkt_rf", "smb", "hml", "rmw", "cma", "mom", "rf"]:
        ff6_dec[c] = ff6_dec[c] / 100.0

    joined = pd.DataFrame({"ls": ls}).join(
        ff6_dec.set_index("date")[["mkt_rf", "smb", "hml", "rmw", "cma", "mom"]],
        how="inner",
    )
    print(f"  Joined with FF6: {len(joined)} months")

    if len(joined) < 30:
        raise RuntimeError(f"Too few months: {len(joined)}")

    y = joined["ls"].values
    factors = ["mkt_rf", "smb", "hml", "rmw", "cma", "mom"]
    X = sm.add_constant(joined[factors].values)

    print()
    print("=" * 70)
    print(f"FF6 spanning: NYSE Size{SIZE_BUCKET} L/S (EW) ~ Mkt-RF + SMB + HML + RMW + CMA + MOM")
    print("=" * 70)

    for lag in HAC_LAGS:
        results = sm.OLS(y, X, hasconst=True).fit(cov_type="HAC", cov_kwds={"maxlags": lag})
        params, tvals, pvals, stderrs = results.params, results.tvalues, results.pvalues, results.bse
        r2, r2_adj = results.rsquared, results.rsquared_adj
        param_names = ["alpha"] + factors

        print(f"\nHAC lag={lag}:")
        print(f"  {'name':<10} {'coef':>12} {'se':>12} {'t':>8} {'p':>8}")
        print("  " + "-" * 55)
        for i, name in enumerate(param_names):
            if name == "alpha":
                print(f"  {name:<10} {params[i]*100:>+11.4f}% {stderrs[i]*100:>11.4f}% "
                      f"{tvals[i]:>+8.3f} {pvals[i]:>8.4f}")
            else:
                print(f"  {name:<10} {params[i]:>+12.4f} {stderrs[i]:>12.4f} "
                      f"{tvals[i]:>+8.3f} {pvals[i]:>8.4f}")
        print(f"  R^2 = {r2:.4f}, adj R^2 = {r2_adj:.4f}, N = {len(joined)}")

    print()
    print("=" * 70)
    print(f"Annualized alpha summary (NYSE Size{SIZE_BUCKET} EW, HAC lag=4)")
    print("=" * 70)
    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
    alpha_monthly = model.params[0]
    alpha_annual = alpha_monthly * 12
    alpha_t = alpha_monthly / model.bse[0]
    print(f"  Monthly alpha:    {alpha_monthly*100:+.4f}% (t = {alpha_t:+.3f})")
    print(f"  Annualized alpha: {alpha_annual*100:+.2f}%")
    print(f"  Raw L/S mean (unspanned): {joined['ls'].mean()*12*100:+.2f}% annualized")

    out = Path(f"data/cache/lazy_prices_backtest/ff6_spanning_size5_nyse.parquet")
    joined.reset_index().rename(columns={"index": "date"}).to_parquet(out, index=False)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()