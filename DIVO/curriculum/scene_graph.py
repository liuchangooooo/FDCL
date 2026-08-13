from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


EPS = 1e-9


def build_task_axis_scene_graph(
    task: str,
    start_pose: Sequence[float],
    goal_pose: Sequence[float],
    obstacle_config: Sequence[Mapping[str, Any]],
    obstacle_z: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """Build a JSON-serializable scene graph around the start-goal task axis."""

    start = _pose3(start_pose)
    goal = _pose3(goal_pose)
    obstacles = [obs for obs in obstacle_config if isinstance(obs, Mapping)]
    z_values = [z for z in (obstacle_z or []) if isinstance(z, Mapping)]
    axis_length = _distance(start[:2], goal[:2])
    axis_angle = _angle_between(start[:2], goal[:2])

    nodes: List[Dict[str, Any]] = [
        {
            "id": "start",
            "type": "start",
            "pose": start,
        },
        {
            "id": "goal",
            "type": "goal",
            "pose": goal,
        },
        {
            "id": "task_axis",
            "type": "start_goal_axis",
            "features": {
                "length": axis_length,
                "angle": axis_angle,
            },
        },
    ]

    obstacle_points: List[Tuple[float, float]] = []
    for idx, obstacle in enumerate(obstacles):
        x = _safe_float(obstacle.get("x", 0.0))
        y = _safe_float(obstacle.get("y", 0.0))
        z = dict(z_values[idx]) if idx < len(z_values) else {}
        obstacle_points.append((x, y))
        nodes.append(
            {
                "id": f"obs_{idx}",
                "type": "obstacle",
                "index": idx,
                "xy": [x, y],
                "purpose": str(obstacle.get("purpose", "")),
                "z": _jsonable_mapping(z),
            }
        )

    edges: List[Dict[str, Any]] = [
        {
            "src": "start",
            "dst": "goal",
            "type": "start_goal",
            "features": {
                "distance": axis_length,
                "angle": axis_angle,
            },
        }
    ]

    for idx, point in enumerate(obstacle_points):
        obs_id = f"obs_{idx}"
        z = z_values[idx] if idx < len(z_values) else {}
        edges.append(
            {
                "src": "start",
                "dst": obs_id,
                "type": "start_obstacle",
                "features": _point_relation_features(start[:2], point),
            }
        )
        edges.append(
            {
                "src": "goal",
                "dst": obs_id,
                "type": "goal_obstacle",
                "features": _point_relation_features(goal[:2], point),
            }
        )
        edges.append(
            {
                "src": obs_id,
                "dst": "task_axis",
                "type": "obstacle_task_axis",
                "features": _axis_relation_features(z),
            }
        )

    for i in range(len(obstacle_points)):
        for j in range(i + 1, len(obstacle_points)):
            edges.append(
                {
                    "src": f"obs_{i}",
                    "dst": f"obs_{j}",
                    "type": "obstacle_pair",
                    "features": _pair_relation_features(
                        obstacle_points[i],
                        obstacle_points[j],
                        z_values[i] if i < len(z_values) else {},
                        z_values[j] if j < len(z_values) else {},
                    ),
                }
            )

    return {
        "version": "task_axis_scene_graph_v1",
        "task": str(task),
        "nodes": nodes,
        "edges": edges,
        "global_features": _global_features(obstacle_points, z_values, axis_length, axis_angle),
    }


def _global_features(
    obstacle_points: Sequence[Tuple[float, float]],
    z_values: Sequence[Mapping[str, Any]],
    axis_length: float,
    axis_angle: float,
) -> Dict[str, Any]:
    blockages = [_safe_float(z.get("blockage", 0.0)) for z in z_values]
    pair_distances = [
        _distance(obstacle_points[i], obstacle_points[j])
        for i in range(len(obstacle_points))
        for j in range(i + 1, len(obstacle_points))
    ]
    corridor_pressure = 0.0
    for z in z_values:
        alpha = _safe_float(z.get("alpha", 0.0))
        blockage = _safe_float(z.get("blockage", 0.0))
        if 0.0 <= alpha <= 1.0:
            alpha_weight = 1.0
        elif -0.25 <= alpha < 0.0 or 1.0 < alpha <= 1.25:
            alpha_weight = 0.5
        else:
            alpha_weight = 0.15
        corridor_pressure += alpha_weight * blockage

    return {
        "axis_length": axis_length,
        "axis_angle": axis_angle,
        "num_obstacles": len(obstacle_points),
        "max_blockage": max(blockages) if blockages else 0.0,
        "mean_blockage": float(np.mean(blockages)) if blockages else 0.0,
        "min_pair_distance": min(pair_distances) if pair_distances else None,
        "mean_pair_distance": float(np.mean(pair_distances)) if pair_distances else None,
        "combined_corridor_pressure": float(corridor_pressure),
    }


def _axis_relation_features(z: Mapping[str, Any]) -> Dict[str, Any]:
    alpha = _safe_float(z.get("alpha", 0.0))
    beta = _safe_float(z.get("beta", 0.0))
    return {
        "alpha": alpha,
        "beta": beta,
        "abs_beta": abs(beta),
        "blockage": _safe_float(z.get("blockage", 0.0)),
        "d_start": _safe_float(z.get("d_start", 0.0)),
        "d_goal": _safe_float(z.get("d_goal", 0.0)),
        "effective_radius": _safe_float(z.get("effective_radius", 0.0)),
        "side": _side_label(beta),
    }


def _pair_relation_features(
    first_xy: Sequence[float],
    second_xy: Sequence[float],
    first_z: Mapping[str, Any],
    second_z: Mapping[str, Any],
) -> Dict[str, Any]:
    first_beta = _safe_float(first_z.get("beta", 0.0))
    second_beta = _safe_float(second_z.get("beta", 0.0))
    first_side = _side_sign(first_beta)
    second_side = _side_sign(second_beta)
    first_blockage = _safe_float(first_z.get("blockage", 0.0))
    second_blockage = _safe_float(second_z.get("blockage", 0.0))
    return {
        "distance": _distance(first_xy, second_xy),
        "delta_alpha": abs(_safe_float(first_z.get("alpha", 0.0)) - _safe_float(second_z.get("alpha", 0.0))),
        "delta_beta": abs(first_beta - second_beta),
        "same_side": first_side != 0 and first_side == second_side,
        "opposite_side": first_side != 0 and second_side != 0 and first_side != second_side,
        "min_blockage": min(first_blockage, second_blockage),
        "max_blockage": max(first_blockage, second_blockage),
        "mean_blockage": 0.5 * (first_blockage + second_blockage),
    }


def _point_relation_features(src_xy: Sequence[float], dst_xy: Sequence[float]) -> Dict[str, float]:
    return {
        "distance": _distance(src_xy, dst_xy),
        "angle": _angle_between(src_xy, dst_xy),
        "dx": _safe_float(dst_xy[0]) - _safe_float(src_xy[0]),
        "dy": _safe_float(dst_xy[1]) - _safe_float(src_xy[1]),
    }


def _pose3(value: Sequence[float]) -> List[float]:
    if value is None:
        values: List[Any] = []
    else:
        values = list(np.asarray(value, dtype=np.float64).reshape(-1))
    while len(values) < 3:
        values.append(0.0)
    return [_safe_float(values[0]), _safe_float(values[1]), _wrap_angle(_safe_float(values[2]))]


def _distance(first_xy: Sequence[float], second_xy: Sequence[float]) -> float:
    dx = _safe_float(second_xy[0]) - _safe_float(first_xy[0])
    dy = _safe_float(second_xy[1]) - _safe_float(first_xy[1])
    return float(math.hypot(dx, dy))


def _angle_between(first_xy: Sequence[float], second_xy: Sequence[float]) -> float:
    dx = _safe_float(second_xy[0]) - _safe_float(first_xy[0])
    dy = _safe_float(second_xy[1]) - _safe_float(first_xy[1])
    return _wrap_angle(math.atan2(dy, dx))


def _side_label(beta: float) -> str:
    sign = _side_sign(beta)
    if sign > 0:
        return "positive"
    if sign < 0:
        return "negative"
    return "center"


def _side_sign(beta: float) -> int:
    beta = _safe_float(beta)
    if abs(beta) < 0.1:
        return 0
    return 1 if beta > 0.0 else -1


def _wrap_angle(angle: float) -> float:
    return float((float(angle) + math.pi) % (2.0 * math.pi) - math.pi)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return float(default)
    if math.isnan(val) or math.isinf(val):
        return float(default)
    return val


def _jsonable_mapping(data: Mapping[str, Any]) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, Mapping):
            output[str(key)] = _jsonable_mapping(value)
        elif isinstance(value, (list, tuple)):
            output[str(key)] = [_jsonable_scalar(item) for item in value]
        else:
            output[str(key)] = _jsonable_scalar(value)
    return output


def _jsonable_scalar(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return _safe_float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float):
        return _safe_float(value)
    return value
