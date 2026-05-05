"""Mean-variance portfolio optimizer (cvxpy-based).

This module solves the long/short portfolio construction problem:

    maximize:    α' w  -  λ × w' Σ w
    subject to:  -position_cap ≤ w_i ≤ position_cap   (per-name cap)
                 ||w||_1 ≤ gross_cap                   (gross leverage cap)

where:
    α  = composite_z column from compute_composite_alpha (expected returns
         in z-score units; high = want long, low = want short)
    Σ  = annualized covariance matrix from estimate_covariance
    λ  = risk aversion parameter (higher → more risk-averse)
    w  = portfolio weights vector (positive = long, negative = short)

Today's version is the *unconstrained* mean-variance optimizer — no dollar
neutrality, beta neutrality, or sector neutrality constraints. Those will
be added in a follow-up commit. The output here is therefore not strictly
market-neutral, but it does respect position size and gross leverage limits.

The optimizer uses CLARABEL as the primary solver (cvxpy's modern default
interior-point method, well-suited to the QP structure here).

Universe alignment
------------------
The alpha and covariance must be aligned: alpha.index must equal
covariance.index (and covariance.columns). If they aren't, the function
raises ValueError. Silent inner-joining would make it easy to accidentally
drop names without noticing — we prefer loud failures during development.

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
        (no dollar-neutrality constraint in this version).
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
    solver_status : str
        cvxpy solver status: 'optimal', 'optimal_inaccurate', etc.
    """

    weights: pd.Series
    expected_alpha: float
    expected_variance: float
    gross_leverage: float
    long_count: int
    short_count: int
    solver_status: str


def optimize_portfolio(
    alpha: pd.Series,
    covariance: pd.DataFrame,
    risk_aversion: float = _DEFAULT_RISK_AVERSION,
    position_cap: float = _DEFAULT_POSITION_CAP,
    gross_cap: float = _DEFAULT_GROSS_CAP,
) -> OptimizationResult:
    """Solve the unconstrained MVO problem.

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
        Gross leverage cap. 1.5 = 150% gross exposure (e.g., 75% long
        plus 75% short = 150% gross).

    Returns
    -------
    OptimizationResult

    Raises
    ------
    ValueError
        If alpha and covariance indices don't match, or parameters are
        out of valid range.
    RuntimeError
        If the solver fails to find an optimal solution.
    """
    # ------------------------------------------------------------------
    # Input validation
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
    # Build cvxpy problem
    # ------------------------------------------------------------------
    alpha_vec = alpha.to_numpy()
    sigma_mat = covariance.to_numpy()

    # Decision variable: portfolio weights
    w = cp.Variable(n)

    # Objective: maximize α'w - λ × w'Σw
    # cvxpy convention: minimize the negative
    quadratic_form = cp.quad_form(w, cp.psd_wrap(sigma_mat))  # type: ignore[attr-defined]
    objective = cp.Minimize(-alpha_vec @ w + risk_aversion * quadratic_form)

    # Constraints
    constraints = [
        w >= -position_cap,           # per-name lower bound
        w <= position_cap,            # per-name upper bound
        cp.norm1(w) <= gross_cap,  # type: ignore[attr-defined]  # gross leverage
    ]

    problem = cp.Problem(objective, constraints)

    # ------------------------------------------------------------------
    # Solve with CLARABEL (cvxpy default for QPs)
    # ------------------------------------------------------------------
    try:
        problem.solve(solver=cp.CLARABEL)  # type: ignore[no-untyped-call]
    except cp.error.SolverError as e:
        raise RuntimeError(f"CLARABEL solver failed: {e}") from e

    if problem.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(
            f"Optimizer did not find optimal solution. "
            f"Status: {problem.status}"
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
    # Use a small tolerance to count "real" longs/shorts (not numerical zeros)
    tol = 1e-8
    long_count = int((weights > tol).sum())
    short_count = int((weights < -tol).sum())

    return OptimizationResult(
        weights=weights,
        expected_alpha=expected_alpha,
        expected_variance=expected_variance,
        gross_leverage=gross_leverage,
        long_count=long_count,
        short_count=short_count,
        solver_status=problem.status,
    )
