"""Signal computation modules for the Axiom Fund strategy.

Each signal module exports a single pure function with the signature:

    compute_<signal_name>(
        universe_df: pd.DataFrame,
        ...other inputs...,
        start_date: str | date,
        end_date: str | date,
        winsorize_pct: float = 0.01,
    ) -> pd.DataFrame

Signal modules are pure functions, not classes — they consume DataFrames
and produce DataFrames with no side effects and no database access. See
docs/signal_design.md for the full design rationale.
"""

from __future__ import annotations
