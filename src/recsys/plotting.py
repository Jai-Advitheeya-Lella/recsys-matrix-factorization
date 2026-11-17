"""Matplotlib styling shared by all figures."""

from __future__ import annotations

import matplotlib as mpl

# categorical palette (fixed order) and neutrals
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948",
)
CATEGORICAL = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT2 = "#52514e"
MUTED = "#9a9893"
GRID = "#e6e5e1"

# model family -> colour (identity follows the family across every figure)
FAMILY_COLORS = {"baseline": MUTED, "neighbourhood": AQUA, "mf": BLUE, "neural": ORANGE, "reference": "#c9c7c1"}


def family(name: str) -> str:
    if "(ref)" in name or "surprise" in name:
        return "reference"
    if name in ("Popularity", "GlobalMean", "BiasBaseline"):
        return "baseline"
    if name == "ItemKNN":
        return "neighbourhood"
    if name in ("GMF", "MLP", "NeuMF"):
        return "neural"
    return "mf"


def apply_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": TEXT2,
        "axes.titlecolor": TEXT,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "xtick.color": TEXT2,
        "ytick.color": TEXT2,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "font.family": "DejaVu Sans",
        "lines.linewidth": 2.0,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })
