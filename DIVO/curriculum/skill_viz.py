"""Stage 1 diagnostic visualizations (Task 15 / Requirement 11.5).

  - ``plot_skill_fan``: K skills' T-block trajectories in one scene (fan plot),
    with the T-block shape drawn at the final pose of each skill.
  - ``plot_obstacle_rollout_overlay``: multiple skills' paths overlaid on the
    same obstacle layout.

Trajectories come from ``rollout_fixed_skill`` (states are [x, y, cos, sin] in
normalized T-block coordinates); ``tblock_corners`` mirrors
``motion_decoder/base.get_tblock_feature_points``. Rendering is headless (Agg),
so these run without a display and without the MuJoCo renderer.
"""

from __future__ import annotations

from typing import Any, List, Mapping, Optional, Sequence

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# T-block corner offsets (mirror motion_decoder/base.get_tblock_feature_points).
_TBLOCK_REL = np.array([[0.2, 0.06], [-0.2, 0.06], [0.06, -0.34], [-0.06, -0.34]], dtype=np.float64)


def tblock_corners(state: Sequence[float]) -> np.ndarray:
    """Four T-block feature points for a state [x, y, cos, sin]. Returns [4, 2]."""
    arr = np.asarray(state, dtype=np.float64).reshape(-1)
    cx, cy = arr[0], arr[1]
    cos = arr[2] if arr.size > 2 else 1.0
    sin = arr[3] if arr.size > 3 else 0.0
    R = np.array([[cos, -sin], [sin, cos]], dtype=np.float64)
    return (R @ _TBLOCK_REL.T).T + np.array([cx, cy])


def _draw_obstacles(ax, obstacles, obstacle_half=0.01):
    for o in obstacles or []:
        try:
            x, y = float(o["x"]), float(o["y"])
        except (KeyError, TypeError, ValueError):
            continue
        ax.add_patch(
            plt.Rectangle(
                (x - obstacle_half, y - obstacle_half),
                2 * obstacle_half, 2 * obstacle_half,
                color="black", alpha=0.6,
            )
        )


def _draw_tblock(ax, state, color, alpha=0.9):
    corners = tblock_corners(state)
    # order corners into the T outline: top edge then stem
    order = [0, 1, 3, 2]
    poly = corners[order]
    ax.add_patch(plt.Polygon(poly, closed=True, fill=False, edgecolor=color, alpha=alpha, lw=1.5))


def plot_skill_fan(
    states_per_skill: Sequence[Sequence[Sequence[float]]],
    obstacles: Optional[Sequence[Mapping[str, float]]],
    out_path: str,
    title: str = "Skill fan",
    goal_xy: Sequence[float] = (0.0, 0.0),
    obstacle_half: float = 0.01,
) -> str:
    """Fan plot: each skill's T-block center path + final-pose T-block shape.

    ``states_per_skill[k]`` is a list of states [x, y, cos, sin] for skill k.
    Saves a PNG to ``out_path`` and returns the path.
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    cmap = plt.get_cmap("tab10")
    for k, states in enumerate(states_per_skill):
        arr = np.asarray(states, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] == 0:
            continue
        color = cmap(k % 10)
        ax.plot(arr[:, 0], arr[:, 1], "-", color=color, alpha=0.8, label=f"skill {k+1}")
        ax.scatter(arr[0, 0], arr[0, 1], color=color, marker="o", s=20)
        _draw_tblock(ax, arr[-1], color)

    _draw_obstacles(ax, obstacles, obstacle_half)
    ax.scatter([goal_xy[0]], [goal_xy[1]], color="red", marker="*", s=120, label="goal")

    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlim(-0.3, 0.3)
    ax.set_ylim(-0.3, 0.3)
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_obstacle_rollout_overlay(
    skill_records: Sequence[Mapping[str, Any]],
    obstacles: Optional[Sequence[Mapping[str, float]]],
    out_path: str,
    title: str = "Obstacle rollout overlay",
    goal_xy: Sequence[float] = (0.0, 0.0),
    obstacle_half: float = 0.01,
) -> str:
    """Overlay multiple skills' rollouts on one obstacle layout.

    ``skill_records[k]`` is a dict with ``states`` and optional ``success``.
    Successful rollouts are drawn solid, failures dashed.
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    cmap = plt.get_cmap("tab10")
    for k, rec in enumerate(skill_records):
        arr = np.asarray(rec.get("states", []), dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] == 0:
            continue
        color = cmap(k % 10)
        style = "-" if rec.get("success") else "--"
        ax.plot(arr[:, 0], arr[:, 1], style, color=color, alpha=0.8,
                label=f"skill {rec.get('skill_id', k+1)}{' ✓' if rec.get('success') else ''}")
        ax.scatter(arr[0, 0], arr[0, 1], color=color, marker="o", s=18)

    _draw_obstacles(ax, obstacles, obstacle_half)
    ax.scatter([goal_xy[0]], [goal_xy[1]], color="red", marker="*", s=120, label="goal")

    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlim(-0.3, 0.3)
    ax.set_ylim(-0.3, 0.3)
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
