"""Shared style helpers for backtest exhibit charts.

Centralizes chart styling — colors, fonts, sizes, save behavior — so
all exhibits in docs/exhibits/ share a consistent look. Each chart
function in scripts/analysis/generate_exhibits.py calls setup_style()
before plotting and save_figure() after, inheriting the configuration
defined here.

Design
------
- matplotlib only (no seaborn dependency for charts; seaborn-style
  is applied via mpl.style)
- 10 x 5.5 inch figure (good ratio for README inline + standalone view)
- 150 DPI (sharp on Retina, sensible file size)
- White background (renders correctly on GitHub light + dark modes)
- Brand color palette matches axiom-fund signal conventions
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import matplotlib as mpl
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# Color palette
# ----------------------------------------------------------------------

# Strategy variants
COLOR_GROSS_3SIG: Final[str] = "#1f77b4"  # blue
COLOR_GROSS_4SIG: Final[str] = "#ff7f0e"  # orange
COLOR_NET: Final[str] = "#2ca02c"  # green
COLOR_DRAWDOWN: Final[str] = "#d62728"  # red
COLOR_NEUTRAL: Final[str] = "#7f7f7f"  # gray

# Signals (consistent across IC charts)
SIGNAL_COLORS: Final[dict[str, str]] = {
    "z_gp": "#1f77b4",      # blue
    "z_ivol": "#ff7f0e",    # orange
    "z_resmom": "#2ca02c",  # green
    "z_pead": "#d62728",    # red
}

# Cost component colors (stacked-bar version)
COST_COMPONENT_COLORS: Final[dict[str, str]] = {
    "commission_bps": "#7f7f7f",      # gray
    "spread_bps": "#1f77b4",          # blue (dominant)
    "impact_bps": "#ff7f0e",          # orange
    "short_borrow_bps": "#2ca02c",    # green
}


# ----------------------------------------------------------------------
# Defaults
# ----------------------------------------------------------------------

DEFAULT_FIGSIZE: Final[tuple[float, float]] = (10.0, 5.5)
DEFAULT_DPI: Final[int] = 150
TITLE_FONTSIZE: Final[int] = 14
AXIS_LABEL_FONTSIZE: Final[int] = 11
TICK_LABEL_FONTSIZE: Final[int] = 10
LEGEND_FONTSIZE: Final[int] = 10


def setup_style() -> None:
    """Apply project-wide matplotlib style. Call once before plotting."""
    plt.style.use("seaborn-v0_8-whitegrid")
    mpl.rcParams.update({
        "figure.figsize": DEFAULT_FIGSIZE,
        "figure.dpi": 100,  # display dpi (lower for fast preview)
        "savefig.dpi": DEFAULT_DPI,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "savefig.edgecolor": "none",
        "axes.titlesize": TITLE_FONTSIZE,
        "axes.labelsize": AXIS_LABEL_FONTSIZE,
        "xtick.labelsize": TICK_LABEL_FONTSIZE,
        "ytick.labelsize": TICK_LABEL_FONTSIZE,
        "legend.fontsize": LEGEND_FONTSIZE,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "#cccccc",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.family": ["DejaVu Sans", "Helvetica", "Arial", "sans-serif"],
    })


def save_figure(fig: plt.Figure, output_dir: Path | str, filename: str) -> Path:
    """Save figure to PNG, returning the path. Creates output_dir if needed.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to save.
    output_dir : Path or str
        Directory for the output PNG. Created if missing.
    filename : str
        Filename without extension. .png is appended.

    Returns
    -------
    Path
        Full path to the saved PNG file.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{filename}.png"
    fig.savefig(path)
    plt.close(fig)
    return path