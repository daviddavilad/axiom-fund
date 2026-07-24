"""FF6 spanning regression on Size4 Lazy Prices L/S returns.

Tests whether the CMN Lazy Prices signal within the Size4 (upper-mid-cap)
bucket has incremental alpha after controlling for known factor exposures
(Mkt-RF, SMB, HML, RMW, CMA, MOM).

Model:
  Size4_LS_return ~ alpha + beta_mktrf * Mkt-RF + beta_smb * SMB
                    + beta_hml * HML + beta_rmw * RMW + beta_cma * CMA
                    + beta_mom * MOM

Standard errors: HAC (Newey-West) with lags 4/6/8 for sensitivity.
Reports alpha, alpha t-stat, factor loadings, R^2, adjusted R^2.

Weighting scheme selected by --weighting {ew, vw} CLI flag. EW reads
from size_quintile_by_lazy_prices_sort.parquet (reconstructs Q1-Q5).
VW reads from size_quintile_vw_ew_compare.parquet (uses precomputed
ls_vw column).

Prompted by 2026-07-23 finding of Size4 being the CMN peak: Sharpe
+0.638 (EW) / +0.770 (VW). FF6 spanning is the definitive test of
whether that peak is genuine alpha or a packaging of known factors.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm


SORT_PATH = Path("data/cache/lazy_prices_backtest/size_quintile_by_lazy_prices_sort.parquet")
VW_EW_PATH = Path("data/cache/lazy_prices_backtest/size_quintile_vw_ew_compare.parquet")
FF6_PATH = Path("data/cache/ff6_monthly.parquet")
SIZE_BUCKET = 4
HAC_LAGS = [4, 6, 8]


def load_ls_series(weighting: str) -> pd.Series:
    """Load Size4 L/S monthly series for the requested weighting."""
    if weighting == "ew":
        sort = pd.read_parquet(SORT_PATH)
        sort["date"] = pd.to_datetime(sort["date"])
        s = sort[sort.size_bucket == SIZE_BUCKET].copy()
        q1 = s[s.lp_quintile == 1].set_index("date")["mean"].rename("q1")
        q5 = s[s.lp_quintile == 5].set_index("date")["mean"].rename("q5")
        return (q1 - q5).rename("ls_return").dropna()
    elif weighting == "vw":
        vw = pd.read_parquet(VW_EW_PATH)
        vw["date"] = pd.to_datetime(vw["date"])
        s = vw[vw.size_bucket == SIZE_BUCKET].copy().sort_values("date")
        return s.set_index("date")["ls_vw"].dropna().rename("ls_return")
    else:
        raise ValueError(f"weighting must be 'ew' or 'vw', got {weighting!r}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--weighting", choices=["ew", "vw"], default="ew",
                   help="Weighting scheme for the Size4 L/S series (default: ew)")
    args = p.parse_args()

    print(f"Loading data (weighting={args.weighting.upper()})...")
    ls = load_ls_series(args.weighting)
    ff6 = pd.read_parquet(FF6_PATH)
    ff6["date"] = pd.to_datetime(ff6["date"])

    print(f"  Size4 L/S series ({args.weighting.upper()}): {len(ls)} months, "
          f"{ls.index.min()} to {ls.index.max()}")

    # FF6 factors in %-points → convert to decimal to match ls scale
    ff6_dec = ff6.copy()
    factor_cols = ["mkt_rf", "smb", "hml", "rmw", "cma", "mom", "rf"]
    for c in factor_cols:
        ff6_dec[c] = ff6_dec[c] / 100.0

    joined = pd.DataFrame({"ls": ls}).join(
        ff6_dec.set_index("date")[["mkt_rf", "smb", "hml", "rmw", "cma", "mom"]],
        how="inner",
    )
    print(f"  Joined with FF6: {len(joined)} months (dropped {len(ls) - len(joined)} unmatched)")

    if len(joined) < 30:
        raise RuntimeError(f"Too few months after join: {len(joined)}")

    y = joined["ls"].values
    factors = ["mkt_rf", "smb", "hml", "rmw", "cma", "mom"]
    X = sm.add_constant(joined[factors].values)

    print()
    print("=" * 70)
    print(f"FF6 spanning regression ({args.weighting.upper()}): "
          f"Size{SIZE_BUCKET} L/S ~ Mkt-RF + SMB + HML + RMW + CMA + MOM")
    print("=" * 70)

    for lag in HAC_LAGS:
        model = sm.OLS(y, X, hasconst=True)
        results = model.fit(cov_type="HAC", cov_kwds={"maxlags": lag})
        params = results.params
        tvals = results.tvalues
        pvals = results.pvalues
        stderrs = results.bse
        r2 = results.rsquared
        r2_adj = results.rsquared_adj
        param_names = ["alpha"] + factors

        print(f"\nHAC lag={lag}:")
        print(f"  {'name':<10} {'coef':>12} {'se':>12} {'t':>8} {'p':>8}")
        print("  " + "-" * 55)
        for i, name in enumerate(param_names):
            if name == "alpha":
                coef_disp = params[i] * 100
                se_disp = stderrs[i] * 100
                print(f"  {name:<10} {coef_disp:>+11.4f}% {se_disp:>11.4f}% "
                      f"{tvals[i]:>+8.3f} {pvals[i]:>8.4f}")
            else:
                print(f"  {name:<10} {params[i]:>+12.4f} {stderrs[i]:>12.4f} "
                      f"{tvals[i]:>+8.3f} {pvals[i]:>8.4f}")
        print(f"  R^2 = {r2:.4f}, adj R^2 = {r2_adj:.4f}, N = {len(joined)}")

    # Annualized alpha summary
    print()
    print("=" * 70)
    print(f"Annualized alpha summary ({args.weighting.upper()}, HAC lag=4)")
    print("=" * 70)
    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
    alpha_monthly = model.params[0]
    alpha_se_monthly = model.bse[0]
    alpha_annual = alpha_monthly * 12
    alpha_t = alpha_monthly / alpha_se_monthly
    print(f"  Monthly alpha:    {alpha_monthly * 100:+.4f}% (t = {alpha_t:+.3f})")
    print(f"  Annualized alpha: {alpha_annual * 100:+.2f}%")
    raw_mean_annual = joined["ls"].mean() * 12
    print(f"  Raw L/S mean (unspanned): {raw_mean_annual * 100:+.2f}% annualized")

    out = Path(f"data/cache/lazy_prices_backtest/ff6_spanning_size4_{args.weighting}.parquet")
    joined.reset_index().rename(columns={"index": "date"}).to_parquet(out, index=False)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()