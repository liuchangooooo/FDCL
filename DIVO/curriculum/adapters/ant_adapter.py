from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np

from DIVO.ant.obstacles import ObstacleSpec
from DIVO.curriculum.obstacle_geometry import (
    ObstacleZ,
    decode_z_to_xy,
    encode_xy_to_z,
)


def ant_effective_radius(spec: ObstacleSpec) -> float:
    """Return a circular effective radius for Ant obstacle attribution.

    Box obstacles are projected to their circumscribed-circle radius. This is a
    lossy projection: angle and aspect ratio stay available in the original
    ObstacleSpec stored alongside the encoded z-space record.
    """

    shape = spec.shape.lower()
    if shape == "circle":
        if spec.radius is None:
            raise ValueError("circle obstacle requires radius")
        return float(spec.radius)
    if shape == "box":
        if spec.half_size is None:
            raise ValueError("box obstacle requires half_size")
        hx, hy = spec.half_size
        return float(np.sqrt(float(hx) ** 2 + float(hy) ** 2))
    raise ValueError(f"Unsupported Ant obstacle shape: {spec.shape}")


def ant_encode_obstacle(
    spec: ObstacleSpec,
    start_xy: Sequence[float],
    goal_xy: Sequence[float],
    corridor_width: float,
) -> ObstacleZ:
    return encode_xy_to_z(
        obs_x=float(spec.center[0]),
        obs_y=float(spec.center[1]),
        effective_radius=ant_effective_radius(spec),
        start_xy=start_xy,
        goal_xy=goal_xy,
        corridor_width=corridor_width,
    )


def ant_decode_obstacle(
    z: ObstacleZ,
    start_xy: Sequence[float],
    goal_xy: Sequence[float],
    shape: str = "circle",
    label: str = "decoded",
) -> ObstacleSpec:
    x, y = decode_z_to_xy(z.alpha, z.beta, start_xy=start_xy, goal_xy=goal_xy)
    shape_norm = shape.lower()
    if shape_norm == "circle":
        return ObstacleSpec(
            shape="circle",
            center=(float(x), float(y)),
            radius=float(z.effective_radius),
            label=label,
        )
    if shape_norm == "box":
        half = float(z.effective_radius) / np.sqrt(2.0)
        return ObstacleSpec(
            shape="box",
            center=(float(x), float(y)),
            half_size=(half, half),
            label=label,
        )
    raise ValueError(f"Unsupported Ant obstacle decode shape: {shape}")


def ant_encode_layout(
    specs: Iterable[ObstacleSpec],
    start_xy: Sequence[float],
    goal_xy: Sequence[float],
    corridor_width: float,
) -> List[ObstacleZ]:
    return [
        ant_encode_obstacle(
            spec,
            start_xy=start_xy,
            goal_xy=goal_xy,
            corridor_width=corridor_width,
        )
        for spec in specs
    ]


def ant_decode_layout(
    zs: Iterable[ObstacleZ],
    start_xy: Sequence[float],
    goal_xy: Sequence[float],
    shape: str = "circle",
    label: str = "decoded",
) -> List[ObstacleSpec]:
    return [
        ant_decode_obstacle(
            z,
            start_xy=start_xy,
            goal_xy=goal_xy,
            shape=shape,
            label=label,
        )
        for z in zs
    ]
