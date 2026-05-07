"""Backtest engine for the Axiom Fund strategy.

This subpackage implements Phase 4 of the build:
  - engine.py: single-period backtest engine (this commit)
  - runner.py: date loop, full historical backtest (later)
  - metrics.py: Sharpe, drawdown, turnover, exhibits (later)

Same architectural pattern as portfolio/: pure functions where possible,
explicit point-in-time correctness, comprehensive synthetic tests.
"""

from __future__ import annotations
