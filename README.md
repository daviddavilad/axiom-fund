# Axiom Fund

**Systematic U.S. equity market-neutral long/short fund prototype.**
A research and portfolio construction engine managed by Arcadia.

---

## Status

**Phase 1 — Foundation (Weeks 1–3).** Setting up data pipeline, universe
construction, and returns panel. No alpha generation code yet.

This repository contains a research prototype. It is **not a live fund** and
no performance figures produced by this code constitute evidence of alpha.
See [`limitations.md`](./limitations.md) for a pre-committed enumeration of
known methodological limitations.

## Strategy summary

- **Universe:** S&P 500 + S&P MidCap 400 (~900 names), point-in-time membership via CRSP
- **Signals:** Gross Profitability, Idiosyncratic Volatility, Residual Momentum (12-1)
- **Rebalance:** Monthly, last trading day
- **Construction:** Dollar- / beta- / sector- / factor-neutral via constrained MVO (cvxpy)
- **Backtest window:** Train 2005–2014, OOS 2015–2022, strict holdout 2023–2025
- **Data:** CRSP + Compustat via WRDS; FRED; Ken French Data Library

Full specification in [`strategy_spec.md`](./strategy_spec.md).

## Project layout

    axiom-fund/
    ├── strategy_spec.md          # Locked strategy specification
    ├── limitations.md            # Pre-committed limitations
    ├── pyproject.toml            # Project metadata and dependencies
    ├── uv.lock                   # Locked dependency resolution
    ├── src/axiom_fund/           # Main package (signals, portfolio, risk, backtest)
    ├── tests/                    # Unit and integration tests
    ├── scripts/                  # Operational scripts (connection tests, data pulls)
    └── notebooks/                # Research notebooks (created in later phases)

## Environment

- **Python:** 3.12
- **Package manager:** [uv](https://docs.astral.sh/uv/)
- **Core:** pandas 2.1.x, numpy 1.26.x, pyarrow, wrds, python-dotenv
- **Dev:** pytest, ruff, mypy, ipykernel, pandas-stubs

Package versions are pinned in `pyproject.toml` and locked in `uv.lock`.

## Reproduction

### Prerequisites

- Python 3.12 (or let uv manage it)
- uv installed via the official installer
- WRDS account with CRSP and Compustat subscriptions
- `~/.pgpass` configured for WRDS (see WRDS documentation)

### Setup

Clone, then run `uv sync` to install the pinned environment.

Create a `.env` file in the project root with:

    WRDS_USERNAME=your_username
    WRDS_PASSWORD=your_password

### Verify WRDS connectivity

    uv run python scripts/test_wrds_connection.py

Expected output ends with "WRDS setup verified."

### Run checks

    uv run ruff check .
    uv run mypy src scripts
    uv run pytest

## License

MIT. See `LICENSE`.

## Author

**David Davila** — University of New Mexico (Finance BBA, Applied Math BS, Class of 2027).
Targeting MFE programs and quantitative trading roles.

This project is built solo as a research portfolio artifact. It is intended to
demonstrate methodological discipline and quantitative research infrastructure,
not to claim alpha.
