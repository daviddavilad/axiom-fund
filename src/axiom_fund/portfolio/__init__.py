"""Portfolio construction modules for the Axiom Fund strategy.

This subpackage implements Phase 3 of the build:
  - covariance.py:  Ledoit-Wolf shrinkage estimator (constant correlation)
  - composite.py:   combine multiple signal z-scores into one alpha (later)
  - optimizer.py:   cvxpy mean-variance optimizer with neutrality
                    constraints (later)

Each module is a pure-function design — input DataFrames or numpy arrays,
output the same. Same architectural pattern as signals/.
"""

from __future__ import annotations
