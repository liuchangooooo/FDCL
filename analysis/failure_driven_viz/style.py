"""Shared visual style helpers for failure-driven analysis figures."""

from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt

FIGURE_BG = "#ffffff"
TITLE_COLOR = "#0f172a"
SUBTITLE_COLOR = "#64748b"
TEXT_COLOR = "#0f172a"
MUTED_TEXT = "#475569"

PANEL_BG = "#f8fbff"
PANEL_EDGE = "#4f6f98"
PANEL_EDGE_LIGHT = "#9bb7d7"
WORKSPACE_BG = "#eef5fb"
REFERENCE_LINE = "#9db4cf"

START_BLOCK_FILL = "#23a36d"
START_BLOCK_EDGE = "#0f7a4d"
GOAL_BLOCK_FILL = "#8cc8ff"
GOAL_BLOCK_EDGE = "#1f9cf0"
GOAL_TEXT = "#1f9cf0"

OBSTACLE_FILL = "#ff5a5f"
OBSTACLE_EDGE = "#b91c1c"
OBSTACLE_TEXT = "#ffffff"

STATIC_ACCENT = "#64748b"
EVOLVE_ACCENT = "#2563eb"
NEUTRAL_ACCENT = "#334155"

SUCCESS_LINE = "#2f7cff"
SUCCESS_FILL = "#cfe1ff"
COLLISION_LINE = "#ff5f63"
TIMEOUT_LINE = "#355b88"
FALL_LINE = "#f1a84f"
PLATEAU_LINE = "#b9922a"
GRID_COLOR = "#c7d8ea"
CARD_BG = "#f8fbff"
CARD_EDGE = "#b9cee6"


def configure_matplotlib() -> None:
    """Install a lightweight paper-like matplotlib theme."""
    plt.rcParams.update(
        {
            "figure.facecolor": FIGURE_BG,
            "axes.facecolor": PANEL_BG,
            "axes.edgecolor": PANEL_EDGE,
            "axes.labelcolor": TEXT_COLOR,
            "xtick.color": MUTED_TEXT,
            "ytick.color": MUTED_TEXT,
            "text.color": TEXT_COLOR,
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titleweight": "bold",
        }
    )


def style_chart_axis(
    axis: plt.Axes,
    *,
    facecolor: str = CARD_BG,
    edgecolor: str = PANEL_EDGE,
    grid_axis: str = "both",
) -> None:
    """Style a standard chart axis with the paper/video palette."""
    axis.set_facecolor(facecolor)
    axis.grid(axis=grid_axis, alpha=0.35, linestyle="--", linewidth=0.8, color=GRID_COLOR)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(edgecolor)
    axis.spines["bottom"].set_color(edgecolor)
    axis.spines["left"].set_linewidth(1.1)
    axis.spines["bottom"].set_linewidth(1.1)
    axis.tick_params(axis="both", labelsize=10.0, width=1.0, length=4.0, color=edgecolor)


def style_scene_axis(axis: plt.Axes) -> None:
    """Turn an axis into a paper-style scene card."""
    axis.set_facecolor(PANEL_BG)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color(PANEL_EDGE)
        spine.set_linewidth(2.0)


def add_badge(
    axis: plt.Axes,
    x: float,
    y: float,
    text: str,
    *,
    facecolor: str,
    edgecolor: str | None = None,
    textcolor: str = "#ffffff",
    ha: str = "left",
    va: str = "top",
    fontsize: float = 8.3,
    alpha: float = 1.0,
    fontweight: str = "bold",
) -> None:
    axis.text(
        x,
        y,
        text,
        transform=axis.transAxes,
        ha=ha,
        va=va,
        fontsize=fontsize,
        color=textcolor,
        fontweight=fontweight,
        bbox={
            "boxstyle": "round,pad=0.28,rounding_size=0.2",
            "facecolor": facecolor,
            "edgecolor": edgecolor or facecolor,
            "linewidth": 0.9,
            "alpha": alpha,
        },
        zorder=20,
    )


def add_summary_card(
    axis: plt.Axes,
    title: str,
    lines: Iterable[str],
    *,
    accent: str,
    badge: str | None = None,
) -> None:
    """Render a compact left-side summary card."""
    axis.set_axis_off()
    axis.set_facecolor(FIGURE_BG)

    axis.text(
        0.04,
        0.78,
        title,
        ha="left",
        va="top",
        fontsize=12.6,
        fontweight="bold",
        color=TITLE_COLOR,
    )
    if badge:
        axis.text(
            0.04,
            0.68,
            badge,
            ha="left",
            va="top",
            fontsize=9.2,
            fontweight="bold",
            color=accent,
        )

    axis.text(
        0.04,
        0.58,
        "\n".join(lines),
        ha="left",
        va="top",
        fontsize=10.2,
        color=MUTED_TEXT,
        linespacing=1.45,
        bbox={
            "boxstyle": "round,pad=0.55,rounding_size=0.22",
            "facecolor": "#f8fafc",
            "edgecolor": PANEL_EDGE_LIGHT,
            "linewidth": 1.2,
            "alpha": 0.98,
        },
    )

    axis.axvline(0.02, ymin=0.10, ymax=0.88, color=accent, linewidth=3.6, alpha=0.95)


def run_accent_color(label: str) -> str:
    lowered = label.lower()
    if "failure-driven" in lowered or "evolve" in lowered:
        return EVOLVE_ACCENT
    if "static" in lowered:
        return STATIC_ACCENT
    return NEUTRAL_ACCENT
