"""Statistical diagnostics for the Axiom Fund v2 release.

This subpackage implements Phase 1 of the v2 methodological upgrades
documented in docs/v2_design.md. Modules:
  - residual_diagnostics.py: pure functions for regression residual
    analysis (Q-Q data, residual-vs-fitted data, Durbin-Watson,
    Breusch-Pagan, Cook's distance, leverage). Takes raw numpy
    arrays; no coupling to statsmodels or any specific regression
    library.
  - inference.py (planned, Item 3): HAC standard errors and
    bootstrapped confidence intervals.

All functions are pure (no I/O, no global state). Plotting is a
downstream concern; these modules return DataFrames or floats only.
"""

from __future__ import annotations
