from __future__ import annotations

from typing import Any, Dict

from DIVO.ant.ant_nav_obstacle_env import AntNavObstacleEnv


def get_ant_env(**class_args: Dict[str, Any]) -> AntNavObstacleEnv:
    target = class_args.pop("_target_", "ant_nav_obstacle")
    if target not in ("ant_nav_obstacle", "ant_nav_obstacle_llm"):
        raise NotImplementedError(f"Ant env type {target} not implemented.")
    return AntNavObstacleEnv(**class_args)
