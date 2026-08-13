from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


ANTMAZE_FAILURE_KEYS = [
    "success",
    "timeout",
    "stuck",
    "low_progress",
    "fall",
]


def _as_xy(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
    except Exception:
        return None
    if arr.shape[0] < 2:
        return None
    return arr[:2]


def _extract_xy(obs: Any) -> Optional[np.ndarray]:
    if isinstance(obs, dict):
        if "achieved_goal" in obs:
            return _as_xy(obs.get("achieved_goal"))
        if "observation" in obs:
            return _extract_xy(obs.get("observation"))
    return _as_xy(obs)


def _extract_goal(obs: Any, fallback_goal: Any = None) -> Optional[np.ndarray]:
    if isinstance(obs, dict) and "desired_goal" in obs:
        return _as_xy(obs.get("desired_goal"))
    return _as_xy(fallback_goal)


def quantize_maze_region(
    xy: Optional[np.ndarray],
    goal_xy: Optional[np.ndarray] = None,
    maze_shape: Optional[Sequence[int]] = None,
) -> str:
    """
    Produce a compact spatial label for prompt/debugging.

    If the maze shape is available, the label is map-relative. Otherwise it falls
    back to a goal-relative direction.
    """
    if xy is None:
        return "none"

    if maze_shape is not None and len(maze_shape) >= 2:
        h, w = int(maze_shape[0]), int(maze_shape[1])
        # AntMaze coordinates are scaled cell centers. The exact transform lives
        # inside MazeEnv, so this is intentionally coarse.
        cx = np.clip((xy[0] / max(w, 1)) + 0.5, 0.0, 0.999)
        cy = np.clip((xy[1] / max(h, 1)) + 0.5, 0.0, 0.999)
        col = "left" if cx < 1 / 3 else "right" if cx > 2 / 3 else "center"
        row = "bottom" if cy < 1 / 3 else "top" if cy > 2 / 3 else "middle"
        if col == "center" and row == "middle":
            return "center"
        return f"{row}_{col}"

    if goal_xy is None:
        return "unknown"

    delta = xy - goal_xy
    if np.linalg.norm(delta) < 1e-6:
        return "near_goal"
    x, y = float(delta[0]), float(delta[1])
    horiz = "east" if x > 0 else "west" if x < 0 else ""
    vert = "north" if y > 0 else "south" if y < 0 else ""
    if abs(x) > 2 * abs(y):
        return horiz
    if abs(y) > 2 * abs(x):
        return vert
    return f"{vert}_{horiz}".strip("_")


@dataclass
class AntMazeEpisodeTrace:
    observations: List[Any] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    infos: List[Dict[str, Any]] = field(default_factory=list)
    done: bool = False
    truncated: bool = False
    success: Optional[bool] = None
    initial_goal: Any = None
    maze_shape: Optional[Sequence[int]] = None


class AntMazeFailureAnalyzer:
    """Summarize AntMaze rollouts into failure evidence for generator revision."""

    def __init__(
        self,
        success_distance_threshold: float = 0.45,
        stuck_window: int = 30,
        stuck_displacement_threshold: float = 0.20,
        low_progress_ratio: float = 0.25,
        fall_z_threshold: float = 0.25,
    ):
        self.success_distance_threshold = float(success_distance_threshold)
        self.stuck_window = int(stuck_window)
        self.stuck_displacement_threshold = float(stuck_displacement_threshold)
        self.low_progress_ratio = float(low_progress_ratio)
        self.fall_z_threshold = float(fall_z_threshold)

    def classify_episode(self, trace: AntMazeEpisodeTrace) -> Dict[str, Any]:
        observations = list(trace.observations)
        infos = list(trace.infos)
        positions = [_extract_xy(obs) for obs in observations]
        positions = [p for p in positions if p is not None]

        goal = None
        if observations:
            goal = _extract_goal(observations[0], fallback_goal=trace.initial_goal)
        if goal is None and trace.initial_goal is not None:
            goal = _as_xy(trace.initial_goal)

        final_xy = positions[-1] if positions else None
        start_xy = positions[0] if positions else None
        start_dist = float(np.linalg.norm(start_xy - goal)) if start_xy is not None and goal is not None else None
        final_dist = float(np.linalg.norm(final_xy - goal)) if final_xy is not None and goal is not None else None
        progress = None
        progress_ratio = None
        if start_dist is not None and final_dist is not None:
            progress = start_dist - final_dist
            progress_ratio = progress / max(start_dist, 1e-6)

        success = trace.success
        if success is None and infos:
            maybe = infos[-1].get("success")
            success = bool(maybe) if maybe is not None else None
        if success is None and final_dist is not None:
            success = final_dist <= self.success_distance_threshold
        success = bool(success)

        min_torso_z = None
        for obs in observations:
            raw = obs.get("observation") if isinstance(obs, dict) else obs
            try:
                arr = np.asarray(raw, dtype=np.float64).reshape(-1)
            except Exception:
                continue
            if arr.shape[0] >= 3:
                z = float(arr[2])
                min_torso_z = z if min_torso_z is None else min(min_torso_z, z)

        fell = bool(min_torso_z is not None and min_torso_z < self.fall_z_threshold)

        stuck = False
        if len(positions) >= max(2, self.stuck_window):
            recent = positions[-self.stuck_window:]
            displacement = float(np.linalg.norm(recent[-1] - recent[0]))
            stuck = displacement < self.stuck_displacement_threshold

        low_progress = False
        if progress_ratio is not None:
            low_progress = progress_ratio < self.low_progress_ratio

        if success:
            failure_type = "success"
        elif fell:
            failure_type = "fall"
        elif stuck:
            failure_type = "stuck"
        elif low_progress:
            failure_type = "low_progress"
        else:
            failure_type = "timeout"

        return {
            "failure_type": failure_type,
            "success": success,
            "steps": len(observations),
            "episode_reward": float(np.sum(trace.rewards)) if trace.rewards else 0.0,
            "start_xy": start_xy.tolist() if start_xy is not None else None,
            "final_xy": final_xy.tolist() if final_xy is not None else None,
            "goal_xy": goal.tolist() if goal is not None else None,
            "start_distance": start_dist,
            "final_distance": final_dist,
            "progress": progress,
            "progress_ratio": progress_ratio,
            "min_torso_z": min_torso_z,
            "stuck": stuck,
            "low_progress": low_progress,
            "truncated": bool(trace.truncated),
            "failure_region": quantize_maze_region(final_xy, goal, trace.maze_shape),
        }

    def summarize_batch(self, traces: Sequence[AntMazeEpisodeTrace]) -> Dict[str, Any]:
        episode_records = [self.classify_episode(trace) for trace in traces]
        counts = {key: 0 for key in ANTMAZE_FAILURE_KEYS}
        region_counts: Dict[str, int] = {}

        final_distances = []
        progress_ratios = []
        episode_rewards = []
        for record in episode_records:
            key = record["failure_type"]
            counts[key] = counts.get(key, 0) + 1
            region = record.get("failure_region", "none")
            region_counts[region] = region_counts.get(region, 0) + 1
            if record.get("final_distance") is not None:
                final_distances.append(float(record["final_distance"]))
            if record.get("progress_ratio") is not None:
                progress_ratios.append(float(record["progress_ratio"]))
            episode_rewards.append(float(record.get("episode_reward", 0.0)))

        total = max(len(episode_records), 1)
        distribution = {key: value / total for key, value in counts.items()}
        failure_only = {key: value for key, value in counts.items() if key != "success"}
        dominant_failure_type = max(failure_only, key=failure_only.get) if sum(failure_only.values()) > 0 else "success"
        dominant_region = max(region_counts, key=region_counts.get) if region_counts else "none"
        dominant_region_conf = region_counts.get(dominant_region, 0) / total

        success_rate = distribution.get("success", 0.0)
        reliability = "weak"
        if total >= 8 and max(distribution.values()) >= 0.5:
            reliability = "strong"
        elif total >= 4 and max(distribution.values()) >= 0.35:
            reliability = "medium"

        return {
            "batch_stats": counts,
            "failure_counts": counts,
            "failure_distribution": distribution,
            "dominant_failure_type": dominant_failure_type,
            "success_rate": success_rate,
            "mean_episode_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
            "mean_final_distance": float(np.mean(final_distances)) if final_distances else None,
            "mean_progress_ratio": float(np.mean(progress_ratios)) if progress_ratios else None,
            "diagnosis": {
                "failure_region": {
                    "label": dominant_region,
                    "confidence": dominant_region_conf,
                },
                "behavior_bias": {
                    "label": dominant_failure_type,
                    "confidence": distribution.get(dominant_failure_type, 0.0),
                },
                "reliability": reliability,
                "sample_count": total,
            },
            "episode_records": episode_records,
        }


def collect_antmaze_rollout_evidence(
    model,
    env,
    n_episodes: int = 5,
    deterministic: bool = True,
    analyzer: Optional[AntMazeFailureAnalyzer] = None,
) -> Dict[str, Any]:
    """
    Roll out a SB3-style policy and return AntMaze failure evidence.

    This helper is intentionally lightweight and intended for probe/eval loops.
    It supports standard Gymnasium envs and SB3 VecEnv-like envs.
    """
    analyzer = analyzer or AntMazeFailureAnalyzer()
    traces: List[AntMazeEpisodeTrace] = []

    is_vec_env = hasattr(env, "num_envs")
    num_envs = int(getattr(env, "num_envs", 1)) if is_vec_env else 1
    active_traces = [AntMazeEpisodeTrace() for _ in range(num_envs)]
    completed = 0

    obs = env.reset()
    if isinstance(obs, tuple) and len(obs) == 2:
        obs, reset_info = obs
    else:
        reset_info = [{} for _ in range(num_envs)] if is_vec_env else {}

    def _obs_at(observation, idx: int):
        if isinstance(observation, dict):
            return {key: value[idx] for key, value in observation.items()}
        return observation[idx]

    while completed < n_episodes:
        action, _ = model.predict(obs, deterministic=deterministic)
        step_result = env.step(action)

        if len(step_result) == 5:
            next_obs, rewards, terminated, truncated, infos = step_result
            dones = np.logical_or(terminated, truncated)
        else:
            next_obs, rewards, dones, infos = step_result
            truncated = dones

        if not is_vec_env:
            rewards = [rewards]
            dones = [dones]
            infos = [infos]
            truncated = [truncated]

        for i in range(num_envs):
            trace = active_traces[i]
            current_obs = _obs_at(next_obs, i) if is_vec_env else next_obs
            done_info = dict(infos[i])
            if bool(dones[i]) and "terminal_observation" in done_info:
                current_obs = done_info["terminal_observation"]
            trace.observations.append(current_obs)
            trace.rewards.append(float(rewards[i]))
            trace.infos.append(done_info)

            if bool(dones[i]):
                trace.done = True
                trace.truncated = bool(truncated[i]) if np.ndim(truncated) > 0 else bool(truncated)
                maybe_success = infos[i].get("success")
                if maybe_success is not None:
                    trace.success = bool(maybe_success)
                traces.append(trace)
                completed += 1
                active_traces[i] = AntMazeEpisodeTrace()
                if completed >= n_episodes:
                    break

        obs = next_obs

    return analyzer.summarize_batch(traces[:n_episodes])
