from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


LOGGER = logging.getLogger(__name__)
EPS = 1e-6


@dataclass(frozen=True)
class ObstacleZ:
    """Task-relative obstacle coordinates.

    alpha and beta describe placement relative to the start-goal segment.
    effective_radius is fixed benchmark metadata, not a generator-controlled
    size variable.
    """

    alpha: float
    beta: float
    effective_radius: float
    d_start: float
    d_goal: float
    blockage: float
    abs_x: float
    abs_y: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "alpha": float(self.alpha),
            "beta": float(self.beta),
            "effective_radius": float(self.effective_radius),
            "d_start": float(self.d_start),
            "d_goal": float(self.d_goal),
            "blockage": float(self.blockage),
            "abs_x": float(self.abs_x),
            "abs_y": float(self.abs_y),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "ObstacleZ":
        return cls(
            alpha=float(data["alpha"]),
            beta=float(data["beta"]),
            effective_radius=float(data["effective_radius"]),
            d_start=float(data["d_start"]),
            d_goal=float(data["d_goal"]),
            blockage=float(data["blockage"]),
            abs_x=float(data["abs_x"]),
            abs_y=float(data["abs_y"]),
        )


def _as_xy(value: Sequence[float], name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size < 2:
        raise ValueError(f"{name} must contain at least two coordinates")
    return arr[:2]


def compute_corridor_frame(
    start_xy: Sequence[float],
    goal_xy: Sequence[float],
    eps: float = EPS,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return the start-goal unit vector, left-normal vector, and path length."""

    start = _as_xy(start_xy, "start_xy")
    goal = _as_xy(goal_xy, "goal_xy")
    vec = goal - start
    raw_length = float(np.linalg.norm(vec))
    if raw_length < eps:
        LOGGER.warning(
            "Degenerate obstacle frame: start and goal are nearly identical; "
            "using fallback x-axis frame."
        )
        u = np.array([1.0, 0.0], dtype=np.float64)
        length = float(eps)
    else:
        u = vec / raw_length
        length = raw_length
    perp = np.array([-u[1], u[0]], dtype=np.float64)
    return u, perp, length


def encode_xy_to_z(
    obs_x: float,
    obs_y: float,
    effective_radius: float,
    start_xy: Sequence[float],
    goal_xy: Sequence[float],
    corridor_width: float,
) -> ObstacleZ:
    """Encode an absolute obstacle center into task-relative coordinates."""

    if corridor_width <= 0:
        raise ValueError("corridor_width must be positive")
    if effective_radius < 0:
        raise ValueError("effective_radius must be non-negative")

    start = _as_xy(start_xy, "start_xy")
    goal = _as_xy(goal_xy, "goal_xy")
    obstacle = np.array([float(obs_x), float(obs_y)], dtype=np.float64)
    u, perp, length = compute_corridor_frame(start, goal)

    delta = obstacle - start
    alpha = float(np.dot(delta, u) / length)
    beta = float(np.dot(delta, perp) / length)
    d_start = float(np.linalg.norm(delta) / length)
    d_goal = float(np.linalg.norm(obstacle - goal) / length)

    alpha_clamped = float(np.clip(alpha, 0.0, 1.0))
    closest = start + alpha_clamped * (goal - start)
    segment_distance = float(np.linalg.norm(obstacle - closest))
    clearance = max(segment_distance - float(effective_radius), 0.0)
    blockage = float(np.clip(1.0 - clearance / float(corridor_width), 0.0, 1.0))

    return ObstacleZ(
        alpha=alpha,
        beta=beta,
        effective_radius=float(effective_radius),
        d_start=d_start,
        d_goal=d_goal,
        blockage=blockage,
        abs_x=float(obstacle[0]),
        abs_y=float(obstacle[1]),
    )


def decode_z_to_xy(
    alpha: float,
    beta: float,
    start_xy: Sequence[float],
    goal_xy: Sequence[float],
) -> Tuple[float, float]:
    """Decode task-relative placement coordinates into an absolute center."""

    start = _as_xy(start_xy, "start_xy")
    u, perp, length = compute_corridor_frame(start, goal_xy)
    xy = start + float(alpha) * length * u + float(beta) * length * perp
    return float(xy[0]), float(xy[1])


def encode_layout(
    obstacles_xy_radius: Iterable[Sequence[float]],
    start_xy: Sequence[float],
    goal_xy: Sequence[float],
    corridor_width: float,
) -> List[ObstacleZ]:
    """Encode a layout of (x, y, effective_radius) records."""

    encoded = []
    for obstacle in obstacles_xy_radius:
        values = list(obstacle)
        if len(values) < 3:
            raise ValueError("Each obstacle must contain x, y, and effective_radius")
        encoded.append(
            encode_xy_to_z(
                values[0],
                values[1],
                values[2],
                start_xy=start_xy,
                goal_xy=goal_xy,
                corridor_width=corridor_width,
            )
        )
    return encoded


def format_z_for_prompt(z: ObstacleZ, precision: int = 2) -> str:
    fmt = f"{{:.{precision}f}}"
    return (
        f"alpha={fmt.format(z.alpha)}, "
        f"beta={fmt.format(z.beta)}, "
        f"blockage={fmt.format(z.blockage)}, "
        f"d_start={fmt.format(z.d_start)}, "
        f"d_goal={fmt.format(z.d_goal)}"
    )


def format_layout_for_prompt(zs: Sequence[ObstacleZ], precision: int = 2) -> str:
    parts = [
        f"obstacle_{idx}: {format_z_for_prompt(z, precision=precision)}"
        for idx, z in enumerate(zs)
    ]
    return "[" + "; ".join(parts) + "]"
