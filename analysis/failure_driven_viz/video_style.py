"""Shared Push-T environment render helpers for analysis figures."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List, Tuple

import matplotlib.image as mpimg
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

VIDEO_CANVAS_W = 320.0
VIDEO_CANVAS_H = 240.0

VIDEO_BG_DARK = "#2c5784"
VIDEO_CURRENT_FILL = "#5756f2"
VIDEO_CURRENT_EDGE = "#3737c8"
VIDEO_TARGET_FILL = "#a8ff9a"
VIDEO_TARGET_EDGE = "#96ef8a"
VIDEO_OBS_FILL = "#ff5f63"
VIDEO_OBS_EDGE = "#d94347"

VIDEO_WORLD_TO_PIXEL_X_SCALE = 422.5757575757576
VIDEO_WORLD_TO_PIXEL_X_BIAS = 160.8030303030303
VIDEO_WORLD_TO_PIXEL_Y_SCALE = -418.3333333333333
VIDEO_WORLD_TO_PIXEL_Y_BIAS = 117.83333333333333

ASSET_DIR = Path(__file__).resolve().parent / "assets"
VIDEO_TEMPLATE_PATH = ASSET_DIR / "pusht_render_template.png"


def setup_video_axis(axis: plt.Axes) -> None:
    axis.set_facecolor(VIDEO_BG_DARK)
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_xlim(0.0, VIDEO_CANVAS_W)
    axis.set_ylim(VIDEO_CANVAS_H, 0.0)
    axis.set_aspect("equal")


def draw_video_background(axis: plt.Axes) -> None:
    template = load_video_background_template()
    if template is not None:
        axis.imshow(template, extent=(0.0, VIDEO_CANVAS_W, VIDEO_CANVAS_H, 0.0), interpolation="nearest", zorder=0)


def world_to_video_xy(x: float, y: float) -> Tuple[float, float]:
    px = VIDEO_WORLD_TO_PIXEL_X_SCALE * x + VIDEO_WORLD_TO_PIXEL_X_BIAS
    py = VIDEO_WORLD_TO_PIXEL_Y_SCALE * y + VIDEO_WORLD_TO_PIXEL_Y_BIAS
    return px, py


def world_bounds_to_video_extent(x_min: float, x_max: float, y_min: float, y_max: float) -> Tuple[float, float, float, float]:
    left, _ = world_to_video_xy(x_min, 0.0)
    right, _ = world_to_video_xy(x_max, 0.0)
    _, bottom = world_to_video_xy(0.0, y_min)
    _, top = world_to_video_xy(0.0, y_max)
    return left, right, bottom, top


def world_edges_to_video_edges(x_edges: np.ndarray, y_edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pixel_x_edges = np.asarray([world_to_video_xy(float(x), 0.0)[0] for x in x_edges], dtype=float)
    pixel_y_edges = np.asarray([world_to_video_xy(0.0, float(y))[1] for y in y_edges], dtype=float)
    return pixel_x_edges, pixel_y_edges


def draw_video_tblock(
    axis: plt.Axes,
    center_world: Tuple[float, float],
    theta_deg: float,
    *,
    fill: str = VIDEO_CURRENT_FILL,
    edge: str = VIDEO_CURRENT_EDGE,
    zorder: int = 4,
    alpha: float = 0.98,
    linewidth: float = 1.0,
) -> None:
    cx, cy = center_world
    theta_rad = np.deg2rad(theta_deg)
    rotation = np.array(
        [
            [np.cos(theta_rad), -np.sin(theta_rad)],
            [np.sin(theta_rad), np.cos(theta_rad)],
        ],
        dtype=float,
    )
    for center_local, width, height in _tblock_components():
        corners = np.array(
            [
                [-width / 2.0, -height / 2.0],
                [-width / 2.0, height / 2.0],
                [width / 2.0, height / 2.0],
                [width / 2.0, -height / 2.0],
            ],
            dtype=float,
        )
        corners += np.array(center_local, dtype=float)
        corners = corners @ rotation.T
        corners[:, 0] += cx
        corners[:, 1] += cy
        pixel_corners = np.asarray([world_to_video_xy(float(x), float(y)) for x, y in corners], dtype=float)
        axis.add_patch(
            patches.Polygon(
                pixel_corners,
                closed=True,
                facecolor=fill,
                edgecolor=edge,
                linewidth=linewidth,
                alpha=alpha,
                zorder=zorder,
                joinstyle="round",
            )
        )


def draw_video_obstacle(
    axis: plt.Axes,
    center_world: Tuple[float, float],
    *,
    half_extent: float = 0.01,
    fill: str = VIDEO_OBS_FILL,
    edge: str = VIDEO_OBS_EDGE,
    zorder: int = 6,
    alpha: float = 1.0,
    linewidth: float = 0.8,
) -> None:
    cx, cy = center_world
    corners = np.array(
        [
            [cx - half_extent, cy - half_extent],
            [cx - half_extent, cy + half_extent],
            [cx + half_extent, cy + half_extent],
            [cx + half_extent, cy - half_extent],
        ],
        dtype=float,
    )
    pixel_corners = np.asarray([world_to_video_xy(float(x), float(y)) for x, y in corners], dtype=float)
    axis.add_patch(
        patches.Polygon(
            pixel_corners,
            closed=True,
            facecolor=fill,
            edgecolor=edge,
            linewidth=linewidth,
            alpha=alpha,
            zorder=zorder,
            joinstyle="round",
        )
    )


@lru_cache(maxsize=1)
def load_video_background_template():
    if not VIDEO_TEMPLATE_PATH.exists():
        return None
    return mpimg.imread(VIDEO_TEMPLATE_PATH)


def _tblock_components() -> List[Tuple[Tuple[float, float], float, float]]:
    return [
        ((0.0, 0.0), 0.10, 0.03),
        ((0.0, -0.05), 0.03, 0.07),
    ]
