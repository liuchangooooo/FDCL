from __future__ import annotations

from typing import List, Sequence

import numpy as np

from DIVO.ant.obstacles import ObstacleSpec


def _rng(seed=None) -> np.random.Generator:
    return np.random.default_rng(seed)


def _as_xy(value: Sequence[float]) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)[:2]


def _corridor_frame(start: Sequence[float], goal: Sequence[float]):
    start_arr = _as_xy(start)
    goal_arr = _as_xy(goal)
    direction = goal_arr - start_arr
    distance = float(np.linalg.norm(direction))
    if distance < 1e-8:
        direction = np.array([1.0, 0.0], dtype=np.float64)
        distance = 1.0
    unit = direction / distance
    perp = np.array([-unit[1], unit[0]], dtype=np.float64)
    return start_arr, goal_arr, unit, perp, distance


def _between_center(
    rng: np.random.Generator,
    start: Sequence[float],
    goal: Sequence[float],
    alpha_min: float = 0.0,
    alpha_max: float = 1.0,
    lateral_ratio: float | None = 0.15,
    lateral_width: float | None = None,
):
    """Sample a DIVO-style obstacle between start and goal.

    Original DIVO Push-T places obstacles on the object-to-target segment with
    a small perpendicular perturbation. This is the Ant analogue: sample along
    the start-goal segment, then offset laterally by a task-scale-normalized
    amount.
    """

    _, _, _, _, distance = _corridor_frame(start, goal)
    alpha = float(rng.uniform(alpha_min, alpha_max))
    if lateral_width is None:
        if lateral_ratio is None:
            lateral_width = 0.0
        else:
            lateral_width = float(lateral_ratio) * distance
    lateral = float(rng.uniform(-float(lateral_width), float(lateral_width)))
    return _between_center_from_params(start, goal, alpha=alpha, lateral=lateral)


def _between_center_from_params(
    start: Sequence[float],
    goal: Sequence[float],
    alpha: float,
    lateral: float,
):
    start_arr, goal_arr, _, perp, _ = _corridor_frame(start, goal)
    return goal_arr + alpha * (start_arr - goal_arr) + lateral * perp


def _has_endpoint_clearance(
    center: np.ndarray,
    start: Sequence[float],
    goal: Sequence[float],
    radius: float,
    body_radius: float,
    start_clearance: float,
    goal_clearance: float,
) -> bool:
    start_min_distance = float(radius) + max(float(body_radius), 0.0) + max(float(start_clearance), 0.0)
    goal_min_distance = float(radius) + max(float(body_radius), 0.0) + max(float(goal_clearance), 0.0)
    return (
        float(np.linalg.norm(center - _as_xy(start))) > start_min_distance
        and float(np.linalg.norm(center - _as_xy(goal))) > goal_min_distance
    )


def generate_train_obstacle(
    seed=None,
    start=(-4.0, 0.0),
    goal=(4.0, 0.0),
    radius: float = 0.45,
    alpha_min: float = 0.0,
    alpha_max: float = 1.0,
    lateral_ratio: float = 0.15,
    lateral_width: float | None = None,
    body_radius: float = 0.35,
    start_clearance: float = 0.20,
    goal_clearance: float = 0.25,
    max_attempts: int = 100,
) -> List[ObstacleSpec]:
    """Generate the training obstacle using the DIVO between placement flow.

    1. Start and goal are already fixed/sampled by the environment.
    2. Sample an obstacle on the start-goal segment plus perpendicular noise.
    3. Reject if it geometrically conflicts with the initial Ant body.
    4. Reject if it geometrically conflicts with the goal-side reachable area.
    """

    rng = _rng(seed)
    center = None
    for _ in range(max(int(max_attempts), 1)):
        candidate = _between_center(
            rng,
            start,
            goal,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            lateral_ratio=lateral_ratio,
            lateral_width=lateral_width,
        )
        if _has_endpoint_clearance(
            candidate,
            start,
            goal,
            radius,
            body_radius,
            start_clearance,
            goal_clearance,
        ):
            center = candidate
            break
    if center is None:
        center = _between_center_from_params(start, goal, alpha=0.5, lateral=0.0)
        if not _has_endpoint_clearance(
            center,
            start,
            goal,
            radius,
            body_radius,
            start_clearance,
            goal_clearance,
        ):
            raise ValueError(
                "Cannot place a DIVO-between Ant obstacle without colliding "
                "with the start or goal endpoint constraints. "
                f"start={start}, goal={goal}, radius={radius}, "
                f"body_radius={body_radius}, start_clearance={start_clearance}, "
                f"goal_clearance={goal_clearance}"
            )
    return [
        ObstacleSpec(
            shape="circle",
            center=(float(center[0]), float(center[1])),
            radius=float(radius),
            label="train_divo_between",
        )
    ]


def generate_big_obstacle(
    seed=None,
    start=(-4.0, 0.0),
    goal=(4.0, 0.0),
    radius: float = 0.75,
) -> List[ObstacleSpec]:
    rng = _rng(seed)
    center = _between_center(rng, start, goal, alpha_min=0.35, alpha_max=0.65, lateral_ratio=0.1125)
    return [
        ObstacleSpec(
            shape="circle",
            center=(float(center[0]), float(center[1])),
            radius=float(radius),
            label="B_big",
        )
    ]


def generate_multiple_obstacles(
    seed=None,
    start=(-4.0, 0.0),
    goal=(4.0, 0.0),
    radius: float = 0.45,
) -> List[ObstacleSpec]:
    rng = _rng(seed)
    start_arr, goal_arr, _, perp, distance = _corridor_frame(start, goal)
    centers = []
    for alpha in (0.38, 0.62):
        beta = float(rng.uniform(-0.125 * distance, 0.125 * distance))
        centers.append(goal_arr + alpha * (start_arr - goal_arr) + beta * perp)
    return [
        ObstacleSpec(
            shape="circle",
            center=(float(center[0]), float(center[1])),
            radius=float(radius),
            label=f"M_multiple_{idx}",
        )
        for idx, center in enumerate(centers)
    ]


def generate_u_shape_obstacle(
    seed=None,
    start=(-4.0, 0.0),
    goal=(4.0, 0.0),
) -> List[ObstacleSpec]:
    rng = _rng(seed)
    start_arr, goal_arr, unit, perp, _ = _corridor_frame(start, goal)
    alpha = float(rng.uniform(0.42, 0.58))
    base = start_arr + alpha * (goal_arr - start_arr)
    opening_sign = 1.0 if rng.random() < 0.5 else -1.0
    lateral = 0.85
    forward = 0.45
    thickness = 0.18
    depth = 1.0

    back_center = base + opening_sign * lateral * perp
    side_a = base - forward * unit + opening_sign * 0.35 * lateral * perp
    side_b = base + forward * unit + opening_sign * 0.35 * lateral * perp
    angle = float(np.arctan2(unit[1], unit[0]))

    return [
        ObstacleSpec(
            shape="box",
            center=(float(back_center[0]), float(back_center[1])),
            half_size=(forward + thickness, thickness),
            angle=angle,
            label="U_back",
        ),
        ObstacleSpec(
            shape="box",
            center=(float(side_a[0]), float(side_a[1])),
            half_size=(thickness, depth / 2.0),
            angle=angle,
            label="U_side_0",
        ),
        ObstacleSpec(
            shape="box",
            center=(float(side_b[0]), float(side_b[1])),
            half_size=(thickness, depth / 2.0),
            angle=angle,
            label="U_side_1",
        ),
    ]


def generate_dynamic_obstacle(
    seed=None,
    start=(-4.0, 0.0),
    goal=(4.0, 0.0),
    radius: float = 0.65,
    active_after: float = 0.45,
) -> List[ObstacleSpec]:
    rng = _rng(seed)
    center = _between_center(rng, start, goal, alpha_min=0.55, alpha_max=0.70, lateral_ratio=0.04375)
    return [
        ObstacleSpec(
            shape="circle",
            center=(float(center[0]), float(center[1])),
            radius=float(radius),
            active_after=float(active_after),
            label="D_dynamic_block",
        )
    ]


def generate_obstacles(
    seed=None,
    start=(-4.0, 0.0),
    goal=(4.0, 0.0),
    mode: str = "train",
    train_radius: float = 0.45,
    train_alpha_min: float = 0.0,
    train_alpha_max: float = 1.0,
    train_lateral_ratio: float = 0.15,
    train_lateral_width: float | None = None,
    train_body_radius: float = 0.35,
    train_start_clearance: float = 0.20,
    train_goal_clearance: float = 0.25,
    train_corridor_width: float | None = None,
) -> List[ObstacleSpec]:
    mode_norm = str(mode).lower()
    if mode_norm in ("train", "seen"):
        return generate_train_obstacle(
            seed=seed,
            start=start,
            goal=goal,
            radius=train_radius,
            alpha_min=train_alpha_min,
            alpha_max=train_alpha_max,
            lateral_ratio=train_lateral_ratio,
            lateral_width=train_lateral_width,
            body_radius=train_body_radius,
            start_clearance=train_start_clearance,
            goal_clearance=train_goal_clearance,
        )
    if mode_norm in ("b", "big"):
        return generate_big_obstacle(seed=seed, start=start, goal=goal)
    if mode_norm in ("m", "multiple"):
        return generate_multiple_obstacles(seed=seed, start=start, goal=goal)
    if mode_norm in ("u", "u_shape", "u-shape"):
        return generate_u_shape_obstacle(seed=seed, start=start, goal=goal)
    if mode_norm in ("d", "dynamic"):
        return generate_dynamic_obstacle(seed=seed, start=start, goal=goal)
    if mode_norm in ("none", "empty", "no_obstacle"):
        return []
    raise ValueError(f"Unknown Ant obstacle generation mode: {mode}")
