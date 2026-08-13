from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from DIVO.curriculum.obstacle_geometry import (
    ObstacleZ,
    decode_z_to_xy,
    encode_xy_to_z,
)
from DIVO.curriculum.scene_graph import build_task_axis_scene_graph


PUSHT_TARGET_XY = (0.0, 0.0)
PUSHT_OBSTACLE_HALF_SIZE = 0.01
PUSHT_CORRIDOR_WIDTH = 0.05


def pusht_effective_radius(obstacle_half_size: float = PUSHT_OBSTACLE_HALF_SIZE) -> float:
    """Use the circumscribed-circle radius for square Push-T obstacles."""

    return float(np.sqrt(2.0) * float(obstacle_half_size))


def pusht_encode_obstacle(
    obs_dict: Dict[str, float],
    tblock_pose: Sequence[float],
    target_xy: Tuple[float, float] = PUSHT_TARGET_XY,
    obstacle_half_size: float = PUSHT_OBSTACLE_HALF_SIZE,
    corridor_width: float = PUSHT_CORRIDOR_WIDTH,
) -> ObstacleZ:
    start_xy = np.asarray(tblock_pose, dtype=np.float64).reshape(-1)[:2]
    return encode_xy_to_z(
        obs_x=float(obs_dict["x"]),
        obs_y=float(obs_dict["y"]),
        effective_radius=pusht_effective_radius(obstacle_half_size),
        start_xy=start_xy,
        goal_xy=target_xy,
        corridor_width=corridor_width,
    )


def pusht_decode_obstacle(
    z: ObstacleZ,
    tblock_pose: Sequence[float],
    target_xy: Tuple[float, float] = PUSHT_TARGET_XY,
    purpose: str = "",
) -> Dict[str, float]:
    start_xy = np.asarray(tblock_pose, dtype=np.float64).reshape(-1)[:2]
    x, y = decode_z_to_xy(z.alpha, z.beta, start_xy=start_xy, goal_xy=target_xy)
    return {"x": float(x), "y": float(y), "purpose": str(purpose)}


def pusht_encode_layout(
    obs_list: Iterable[Dict[str, float]],
    tblock_pose: Sequence[float],
    target_xy: Tuple[float, float] = PUSHT_TARGET_XY,
    obstacle_half_size: float = PUSHT_OBSTACLE_HALF_SIZE,
    corridor_width: float = PUSHT_CORRIDOR_WIDTH,
) -> List[ObstacleZ]:
    return [
        pusht_encode_obstacle(
            obs,
            tblock_pose=tblock_pose,
            target_xy=target_xy,
            obstacle_half_size=obstacle_half_size,
            corridor_width=corridor_width,
        )
        for obs in obs_list
    ]


def pusht_build_scene_graph(
    obs_list: Iterable[Dict[str, float]],
    tblock_pose: Sequence[float],
    target_pose: Optional[Sequence[float]] = None,
    obstacle_z: Optional[Sequence[Mapping[str, float]]] = None,
    obstacle_half_size: float = PUSHT_OBSTACLE_HALF_SIZE,
    corridor_width: float = PUSHT_CORRIDOR_WIDTH,
) -> Dict[str, object]:
    """Build a Push-T scene graph without changing the generator or simulator."""

    obstacles = list(obs_list or [])
    if target_pose is None:
        target_pose = [PUSHT_TARGET_XY[0], PUSHT_TARGET_XY[1], -np.pi / 4]
    goal_xy = tuple(np.asarray(target_pose, dtype=np.float64).reshape(-1)[:2])

    z_dicts: List[Mapping[str, float]]
    provided_z = list(obstacle_z or [])
    if len(provided_z) == len(obstacles):
        z_dicts = provided_z
    else:
        z_dicts = [
            z.to_dict()
            for z in pusht_encode_layout(
                obstacles,
                tblock_pose=tblock_pose,
                target_xy=goal_xy,
                obstacle_half_size=obstacle_half_size,
                corridor_width=corridor_width,
            )
        ]

    return build_task_axis_scene_graph(
        task="PushT",
        start_pose=tblock_pose,
        goal_pose=target_pose,
        obstacle_config=obstacles,
        obstacle_z=z_dicts,
    )


def pusht_decode_layout(
    zs: Iterable[ObstacleZ],
    tblock_pose: Sequence[float],
    target_xy: Tuple[float, float] = PUSHT_TARGET_XY,
    purpose: str = "",
) -> List[Dict[str, float]]:
    return [
        pusht_decode_obstacle(
            z,
            tblock_pose=tblock_pose,
            target_xy=target_xy,
            purpose=purpose,
        )
        for z in zs
    ]
