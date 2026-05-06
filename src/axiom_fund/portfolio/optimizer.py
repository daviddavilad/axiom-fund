"""Mean-variance portfolio optimizer (cvxpy-based).

This module solves the long/short portfolio construction problem:

    maximize:    α' w  -  λ × w' Σ w
    subject to:  -position_cap ≤ w_i ≤ position_cap   (per-name cap)
                 ||w||_1 ≤ gross_cap                   (gross leverage cap)
                 [optional]  Σ w = 0                   (dollar neutral)
                 [optional]  β' w = 0                  (beta neutral)
                 [optional]  Σ_{i∈s} w_i = 0  ∀ s     (sector neutral)

where:
    α  = composite_z column from compute_composite_alpha (expected returns
         in z-score units; high = want long, low = want short)
    Σ  = annualized covariance matrix from estimate_covariance
    λ  = risk aversion parameter (higher → more risk-averse)
    w  = portfolio weights vector (positive = long, negative = short)

Neutrality constraints
----------------------
The neutrality constraints are equality constraints, optional, and toggled
independently. Each requires its own input data:

  - dollar neutral: requires nothing additional (constrains weight sum = 0)
  - beta neutral:   requires `betas` (Series of per-asset market betas)
  - sector neutral: requires `sectors` (Series of per-asset sector codes)

When a neutrality constraint is active, the corresponding diagnostic in
the result reflects the constraint binding (e.g., portfolio_beta ≈ 0 when
constrain_beta_neutral=True).

The optimizer uses CLARABEL as the primary solver (cvxpy's modern default
interior-point method, well-suited to QPs with equality + inequality
constraints).

Universe alignment
------------------
The alpha and covariance must be aligned: alpha.index must equal
covariance.index (and covariance.columns). If betas or sectors are
provided, they must also be indexed by the same permnos. Silent
inner-joining would make it easy to accidentally drop names without
noticing — we prefer loud failures during development.

Reference
---------
Markowitz, H. (1952). "Portfolio Selection." Journal of Finance, 7(1), 77-91.
"""

from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import pandas as pd

# Locked defaults per docs/strategy_spec.md §6
_DEFAULT_POSITION_CAP: float = 0.015   # 1.5% per name
_DEFAULT_GROSS_CAP: float = 1.5        # 1.5× gross leverage
_DEFAULT_RISK_AVERSION: float = 1.0    # caller can tune


@dataclass(frozen=True)
class OptimizationResult:
    """Output of the mean-variance optimizer.

    Attributes
    ----------
    weights : pd.Series
        Portfolio weights indexed by permno. Sum may be nonzero
        if dollar-neutrality is not active.
    expected_alpha : float
        α' w  — the alpha contribution to the objective.
    expected_variance : float
        w' Σ w — annualized portfolio variance.
    gross_leverage : float
        Σ |w_i| — the L1 norm of the weights.
    long_count : int
        Number of strictly positive weights.
    short_count : int
        Number of strictly negative weights.
    net_exposure : float
        Σ w_i — the signed sum of weights. Should be ≈ 0 when
        constrain_dollar_neutral=True.
    portfolio_beta : float | None
        β' w — portfolio beta against the market factor. Computed only
        if betas were provided to the optimizer; None otherwise.
        Should be ≈ 0 when constrain_beta_neutral=True.
    sector_exposures : pd.Series | None
        Series of per-sector net exposure (sum of weights within each
        sector). Computed only if sectors were provided. Each value
        should be ≈ 0 when constrain_sector_neutral=True.
    solver_status : str
        cvxpy solver status: 'optimal', 'optimal_inaccurate', etc.
    """

    weights: pd.Series
    expected_alpha: float
    expected_variance: float
    gross_leverage: float
    long_count: int
    short_count: int
    net_exposure: float
    portfolio_beta: float | None
    sector_exposures: pd.Series | None
    solver_status: str


def optimize_portfolio(
    alpha: pd.Series,
    covariance: pd.DataFrame,
    risk_aversion: float = _DEFAULT_RISK_AVERSION,
    position_cap: float = _DEFAULT_POSITION_CAP,
    gross_cap: float = _DEFAULT_GROSS_CAP,
    betas: pd.Series | None = None,
    sectors: pd.Series | None = None,
    constrain_dollar_neutral: bool = False,
    constrain_beta_neutral: bool = False,
    constrain_sector_neutral: bool = False,
) -> OptimizationResult:
    """Solve the MVO problem with optional neutrality constraints.

    Parameters
    ----------
    alpha : pd.Series
        Expected return per asset, indexed by permno. Typically the
        composite_z column from compute_composite_alpha.
    covariance : pd.DataFrame
        Annualized covariance matrix, square, indexed by permno on both axes.
        Index and columns must equal alpha.index.
    risk_aversion : float, default 1.0
        The λ parameter. Higher = more risk-averse.
    position_cap : float, default 0.015
        Per-name absolute weight cap. 0.015 = 1.5%.
    gross_cap : float, default 1.5
        Gross leverage cap. 1.5 = 150% gross exposure.
    betas : pd.Series | None, default None
        Per-asset market beta, indexed by permno. Required if
        constrain_beta_neutral=True. If provided, portfolio_beta is
        included in the result regardless of whether the constraint
        is active.
    sectors : pd.Series | None, default None
        Per-asset sector code (e.g., GICS gsector), indexed by permno.
        Required if constrain_sector_neutral=True. If provided,
        sector_exposures is included in the result.
    constrain_dollar_neutral : bool, default False
        If True, add Σ w = 0 to the constraints.
    constrain_beta_neutral : bool, default False
        If True, add β' w = 0 to the constraints. Requires `betas`.
    constrain_sector_neutral : bool, default False
        If True, for each unique sector s, add Σ_{i∈s} w_i = 0.
        Requires `sectors`.

    Returns
    -------
    OptimizationResult

    Raises
    ------
    ValueError
        If inputs are misaligned, NaN, or constraints are requested
        without their required data.
    RuntimeError
        If the solver fails to find an optimal solution (often due
        to mutually-infeasible constraints).
    """
    # ------------------------------------------------------------------
    # Input validation — alpha and covariance
    # ------------------------------------------------------------------
    if not isinstance(alpha, pd.Series):
        raise ValueError(f"alpha must be a pandas Series, got {type(alpha).__name__}")
    if not isinstance(covariance, pd.DataFrame):
        raise ValueError(
            f"covariance must be a pandas DataFrame, got {type(covariance).__name__}"
        )
    if covariance.shape[0] != covariance.shape[1]:
        raise ValueError(
            f"covariance must be square, got shape {covariance.shape}"
        )
    if not alpha.index.equals(covariance.index):
        raise ValueError(
            "alpha.index must equal covariance.index (use same permno ordering)"
        )
    if not covariance.index.equals(covariance.columns):
        raise ValueError(
            "covariance.index must equal covariance.columns"
        )
    if alpha.isna().any():
        raise ValueError("alpha contains NaN values; pre-clean before optimizing")
    if covariance.isna().any().any():
        raise ValueError("covariance contains NaN values; pre-clean before optimizing")

    if risk_aversion < 0:
        raise ValueError(f"risk_aversion must be >= 0, got {risk_aversion}")
    if not 0 < position_cap <= 1.0:
        raise ValueError(f"position_cap must be in (0, 1], got {position_cap}")
    if gross_cap <= 0:
        raise ValueError(f"gross_cap must be > 0, got {gross_cap}")

    n = len(alpha)
    if n < 2:
        raise ValueError(f"need at least 2 assets, got {n}")

    # ------------------------------------------------------------------
    # Input validation — betas and sectors
    # ------------------------------------------------------------------
    if constrain_beta_neutral and betas is None:
        raise ValueError(
            "constrain_beta_neutral=True requires `betas` to be provided"
        )
    if constrain_sector_neutral and sectors is None:
        raise ValueError(
            "constrain_sector_neutral=True requires `sectors` to be provided"
        )

    if betas is not None:
        if not isinstance(betas, pd.Series):
            raise ValueError(
                f"betas must be a pandas Series, got {type(betas).__name__}"
            )
        if not betas.index.equals(alpha.index):
            raise ValueError("betas.index must equal alpha.index")
        if betas.isna().any():
            raise ValueError("betas contains NaN values; pre-clean before optimizing")

    if sectors is not None:
        if not isinstance(sectors, pd.Series):
            raise ValueError(
                f"sectors must be a pandas Series, got {type(sectors).__name__}"
            )
        if not sectors.index.equals(alpha.index):
            raise ValueError("sectors.index must equal alpha.index")
        if sectors.isna().any():
            raise ValueError(
                "sectors contains NaN values; pre-clean before optimizing"
            )

    # ------------------------------------------------------------------
    # Build cvxpy problem
    # ------------------------------------------------------------------
    alpha_vec = alpha.to_numpy()
    sigma_mat = covariance.to_numpy()

    # Decision variable: portfolio weights
    w = cp.Variable(n)

    # Objective: maximize α'w - λ × w'Σw
    quadratic_form = cp.quad_form(w, cp.psd_wrap(sigma_mat))  # type: ignore[attr-defined]
    objective = cp.Minimize(-alpha_vec @ w + risk_aversion * quadratic_form)

    # Constraints
    constraints: list[cp.Constraint] = [
        w >= -position_cap,           # per-name lower bound
        w <= position_cap,            # per-name upper bound
        cp.norm1(w) <= gross_cap,  # type: ignore[attr-defined]  # gross leverage
    ]

    # Optional: dollar neutrality
    if constrain_dollar_neutral:
        constraints.append(cp.sum(w) == 0)  # type: ignore[attr-defined]

    # Optional: beta neutrality
    if constrain_beta_neutral:
        # betas validated above; not None here
        beta_vec = betas.to_numpy()  # type: ignore[union-attr]
        constraints.append(beta_vec @ w == 0)

    # Optional: sector neutrality — one constraint per sector
    sector_codes = None
    if constrain_sector_neutral:
        sector_codes = sectors.to_numpy()  # type: ignore[union-attr]
        unique_sectors = np.unique(sector_codes)
        for s in unique_sectors:
            mask = (sector_codes == s).astype(float)
            constraints.append(mask @ w == 0)

    problem = cp.Problem(objective, constraints)

    # ------------------------------------------------------------------
    # Solve with CLARABEL
    # ------------------------------------------------------------------
    try:
        problem.solve(solver=cp.CLARABEL)  # type: ignore[no-untyped-call]
    except cp.error.SolverError as e:
        raise RuntimeError(f"CLARABEL solver failed: {e}") from e

    if problem.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(
            f"Optimizer did not find optimal solution. "
            f"Status: {problem.status}. Constraints may be infeasible."
        )

    if w.value is None:
        raise RuntimeError("Optimizer returned no solution despite optimal status")

    # ------------------------------------------------------------------
    # Build the result
    # ------------------------------------------------------------------
    weights = pd.Series(np.asarray(w.value), index=alpha.index, name="weight")

    expected_alpha = float(alpha_vec @ weights.to_numpy())
    expected_variance = float(weights.to_numpy() @ sigma_mat @ weights.to_numpy())
    gross_leverage = float(weights.abs().sum())
    net_exposure = float(weights.sum())

    tol = 1e-8
    long_count = int((weights > tol).sum())
    short_count = int((weights < -tol).sum())

    portfolio_beta: float | None = None
    if betas is not None:
        portfolio_beta = float(betas.to_numpy() @ weights.to_numpy())

    sector_exposures: pd.Series | None = None
    if sectors is not None:
        sector_exposures = weights.groupby(sectors).sum()
        sector_exposures.name = "sector_exposure"

    return OptimizationResult(
        weights=weights,
        expected_alpha=expected_alpha,
        expected_variance=expected_variance,
        gross_leverage=gross_leverage,
        long_count=long_count,
        short_count=short_count,
        net_exposure=net_exposure,
        portfolio_beta=portfolio_beta,
        sector_exposures=sector_exposures,
        solver_status=problem.status,
    )
