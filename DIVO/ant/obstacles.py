from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class ObstacleSpec:
    """2D virtual obstacle used by AntNavObstacle."""

    shape: str
    center: Tuple[float, float]
    radius: Optional[float] = None
    half_size: Optional[Tuple[float, float]] = None
    angle: float = 0.0
    active_after: float = 0.0
    label: str = "obstacle"

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def coerce_obstacle(spec: ObstacleSpec | Dict[str, object]) -> ObstacleSpec:
    if isinstance(spec, ObstacleSpec):
        return spec
    shape = str(spec.get("shape", "circle")).lower()
    center_raw = spec.get("center", (0.0, 0.0))
    center = (float(center_raw[0]), float(center_raw[1]))  # type: ignore[index]
    radius = spec.get("radius")
    half_size = spec.get("half_size")
    return ObstacleSpec(
        shape=shape,
        center=center,
        radius=None if radius is None else float(radius),
        half_size=None
        if half_size is None
        else (float(half_size[0]), float(half_size[1])),  # type: ignore[index]
        angle=float(spec.get("angle", 0.0)),
        active_after=float(spec.get("active_after", 0.0)),
        label=str(spec.get("label", "obstacle")),
    )


def coerce_obstacles(specs: Iterable[ObstacleSpec | Dict[str, object]]) -> List[ObstacleSpec]:
    return [coerce_obstacle(spec) for spec in specs]


def is_active(obstacle: ObstacleSpec, progress_ratio: float) -> bool:
    return progress_ratio >= obstacle.active_after


def point_signed_margin(point_xy: Sequence[float], obstacle: ObstacleSpec) -> float:
    """Return signed collision margin; positive means outside, negative inside."""

    point = np.asarray(point_xy, dtype=np.float64)
    center = np.asarray(obstacle.center, dtype=np.float64)

    if obstacle.shape == "circle":
        if obstacle.radius is None:
            raise ValueError("circle obstacle requires radius")
        return float(np.linalg.norm(point - center) - obstacle.radius)

    if obstacle.shape == "box":
        if obstacle.half_size is None:
            raise ValueError("box obstacle requires half_size")
        rel = point - center
        if obstacle.angle:
            c, s = np.cos(-obstacle.angle), np.sin(-obstacle.angle)
            rot = np.array([[c, -s], [s, c]], dtype=np.float64)
            rel = rot @ rel
        half = np.asarray(obstacle.half_size, dtype=np.float64)
        q = np.abs(rel) - half
        outside = np.linalg.norm(np.maximum(q, 0.0))
        inside = min(max(float(q[0]), float(q[1])), 0.0)
        return float(outside + inside)

    raise ValueError(f"Unsupported obstacle shape: {obstacle.shape}")


def collides(
    point_xy: Sequence[float],
    obstacles: Sequence[ObstacleSpec],
    progress_ratio: float,
    body_radius: float = 0.35,
) -> Tuple[bool, Optional[ObstacleSpec], float]:
    """Check torso collision against active virtual obstacles."""

    best_margin = float("inf")
    best_obstacle: Optional[ObstacleSpec] = None
    for obstacle in obstacles:
        if not is_active(obstacle, progress_ratio):
            continue
        margin = point_signed_margin(point_xy, obstacle) - body_radius
        if margin < best_margin:
            best_margin = margin
            best_obstacle = obstacle
        if margin <= 0.0:
            return True, obstacle, float(margin)
    return False, best_obstacle, float(best_margin)


def obstacle_feature_vector(
    point_xy: Sequence[float],
    obstacles: Sequence[ObstacleSpec],
    progress_ratio: float,
    max_obstacles: int,
    scale: float = 1.0,
) -> np.ndarray:
    """Fixed-size relative obstacle feature vector.

    Each slot is [dx, dy, sx, sy, active], where sx/sy are radius-like extents.
    Unused slots are zero-padded.
    """

    point = np.asarray(point_xy, dtype=np.float64)
    scale = max(float(scale), 1e-8)
    features = np.zeros((max_obstacles, 5), dtype=np.float32)
    for idx, obstacle in enumerate(obstacles[:max_obstacles]):
        center = np.asarray(obstacle.center, dtype=np.float64)
        rel = (center - point) / scale
        if obstacle.shape == "circle":
            sx = sy = float(obstacle.radius or 0.0) / scale
        elif obstacle.shape == "box":
            half = obstacle.half_size or (0.0, 0.0)
            sx, sy = float(half[0]) / scale, float(half[1]) / scale
        else:
            sx = sy = 0.0
        features[idx] = np.array(
            [rel[0], rel[1], sx, sy, 1.0 if is_active(obstacle, progress_ratio) else 0.0],
            dtype=np.float32,
        )
    return features.reshape(-1)


def progress_ratio(start: Sequence[float], goal: Sequence[float], point_xy: Sequence[float]) -> float:
    start_arr = np.asarray(start, dtype=np.float64)
    goal_arr = np.asarray(goal, dtype=np.float64)
    point = np.asarray(point_xy, dtype=np.float64)
    direction = goal_arr - start_arr
    denom = float(np.dot(direction, direction))
    if denom <= 1e-12:
        return 0.0
    return float(np.clip(np.dot(point - start_arr, direction) / denom, 0.0, 1.0))
