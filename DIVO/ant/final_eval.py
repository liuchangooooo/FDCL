from __future__ import annotations

import argparse
import copy
import json
import pathlib
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import torch
import wandb
from omegaconf import OmegaConf

from DIVO.ant import AntNavObstacleEnv
from DIVO.policy import get_policy
from DIVO.utils.util import save_anim


DEFAULT_BENCHMARKS = ["Seen", "B", "M", "U", "D"]

if not OmegaConf.has_resolver("now"):
    OmegaConf.register_new_resolver("now", lambda pattern: datetime.now().strftime(pattern))


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _find_run_config(checkpoint: pathlib.Path) -> Optional[pathlib.Path]:
    for parent in [checkpoint.parent, *checkpoint.parents]:
        candidate = parent / ".hydra" / "config.yaml"
        if candidate.exists():
            return candidate
    return None


def _load_config(checkpoint: pathlib.Path, config_path: Optional[str]):
    if config_path:
        path = pathlib.Path(config_path).expanduser().resolve()
    else:
        found = _find_run_config(checkpoint)
        if found is None:
            raise FileNotFoundError(
                "Could not infer run config from checkpoint path. Pass --config explicitly."
            )
        path = found
    cfg = OmegaConf.load(path)
    OmegaConf.resolve(cfg)
    return cfg, path


def _make_env_from_cfg(cfg, mode: str, render_mode: Optional[str], max_steps: Optional[int]):
    env_cfg = copy.deepcopy(OmegaConf.to_container(cfg.env, resolve=True))
    env_cfg.pop("_target_", None)
    env_cfg["obstacle_mode"] = "seen" if mode == "Seen" else mode
    env_cfg["obstacle"] = mode.lower() not in ("none", "no_obstacle")
    if render_mode is not None:
        env_cfg["render_mode"] = render_mode
    if max_steps is not None:
        env_cfg["max_episode_steps"] = int(max_steps)
    return AntNavObstacleEnv(**env_cfg)


def _load_policy(cfg, checkpoint: pathlib.Path, device: torch.device):
    env = _make_env_from_cfg(cfg, mode="Seen", render_mode=None, max_steps=None)
    policy = get_policy(env, **cfg.policy).to(device)
    state = torch.load(str(checkpoint), map_location=device)
    policy.load_state_dict(state)
    policy.eval()
    env.close()
    return policy


def _rollout(policy, env, device: torch.device, seed: int, max_steps: int, save_video: bool):
    obs, info = env.reset(seed=seed)
    frames = []
    if save_video:
        frame = env.render()
        if frame is not None:
            frames.append(frame)

    total_reward = 0.0
    path = []
    terminated = False
    truncated = False
    final_info = info

    for step_idx in range(max_steps):
        obs_th = torch.tensor(obs, dtype=torch.float32, device=device)
        with torch.no_grad():
            action = policy.predict_action(obs_th).detach().cpu().numpy()[0]
        obs, reward, terminated, truncated, final_info = env.step(action)
        total_reward += float(reward)
        path.append(np.asarray(final_info.get("xy_position", [0.0, 0.0]), dtype=np.float32))

        if save_video:
            frame = env.render()
            if frame is not None:
                frames.append(frame)

        if terminated or truncated:
            break

    path_arr = np.asarray(path, dtype=np.float32)
    traveled = 0.0
    if len(path_arr) > 1:
        traveled = float(np.linalg.norm(path_arr[1:] - path_arr[:-1], axis=1).sum())
    start_xy = np.asarray(final_info.get("start_xy", [0.0, 0.0]), dtype=np.float32)
    goal_xy = np.asarray(final_info.get("goal_xy", [0.0, 0.0]), dtype=np.float32)
    straight = float(np.linalg.norm(goal_xy - start_xy))
    path_efficiency = float(straight / traveled) if traveled > 1e-6 else 0.0

    return {
        "success": bool(final_info.get("success", False)),
        "collision": bool(final_info.get("collided", final_info.get("collision", False))),
        "unhealthy": bool(final_info.get("unhealthy", False)),
        "timeout": bool(truncated and not final_info.get("success", False)),
        "episode_reward": float(total_reward),
        "episode_length": int(len(path_arr)),
        "final_distance": float(final_info.get("goal_distance", np.nan)),
        "final_progress_ratio": float(final_info.get("progress_ratio", np.nan)),
        "path_efficiency": path_efficiency,
        "start_xy": start_xy,
        "goal_xy": goal_xy,
        "obstacles": final_info.get("obstacles", []),
        "frames": frames,
    }


def evaluate_family(
    policy,
    cfg,
    family: str,
    output_dir: pathlib.Path,
    device: torch.device,
    seed: int,
    num_episodes: int,
    max_steps: int,
    num_videos: int,
    video_fps: int,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    media_dir = output_dir / "media"
    media_dir.mkdir(parents=True, exist_ok=True)
    env = _make_env_from_cfg(
        cfg,
        mode=family,
        render_mode="rgb_array" if num_videos > 0 else None,
        max_steps=max_steps,
    )

    episodes = []
    video_paths = []
    for ep_idx in range(num_episodes):
        result = _rollout(
            policy,
            env,
            device=device,
            seed=seed + ep_idx,
            max_steps=max_steps,
            save_video=ep_idx < num_videos,
        )
        frames = result.pop("frames")
        if frames:
            video_path_no_ext = media_dir / f"{family}_episode_{ep_idx:03d}"
            save_anim(frames, str(video_path_no_ext), fps=video_fps, framerate=video_fps)
            video_paths.append(str(video_path_no_ext.with_suffix(".mp4")))
        episodes.append(result)

    env.close()
    success_values = [float(ep["success"]) for ep in episodes]
    collision_values = [float(ep["collision"]) for ep in episodes]
    unhealthy_values = [float(ep["unhealthy"]) for ep in episodes]
    timeout_values = [float(ep["timeout"]) for ep in episodes]
    reward_values = [float(ep["episode_reward"]) for ep in episodes]
    length_values = [float(ep["episode_length"]) for ep in episodes]
    distance_values = [float(ep["final_distance"]) for ep in episodes]
    progress_values = [float(ep["final_progress_ratio"]) for ep in episodes]
    efficiency_values = [float(ep["path_efficiency"]) for ep in episodes]

    summary = {
        "success_rate": float(np.mean(success_values)) if episodes else 0.0,
        "collision_rate": float(np.mean(collision_values)) if episodes else 0.0,
        "unhealthy_rate": float(np.mean(unhealthy_values)) if episodes else 0.0,
        "timeout_rate": float(np.mean(timeout_values)) if episodes else 0.0,
        "mean_reward": float(np.mean(reward_values)) if episodes else 0.0,
        "std_reward": float(np.std(reward_values)) if episodes else 0.0,
        "mean_ep_length": float(np.mean(length_values)) if episodes else 0.0,
        "mean_final_distance": float(np.mean(distance_values)) if episodes else 0.0,
        "mean_progress_ratio": float(np.mean(progress_values)) if episodes else 0.0,
        "mean_path_efficiency": float(np.mean(efficiency_values)) if episodes else 0.0,
        "num_episodes": int(num_episodes),
        "max_steps": int(max_steps),
        "video_paths": video_paths,
        "example_obstacles": episodes[0]["obstacles"] if episodes else [],
    }
    payload = {"summary": summary, "episodes": episodes}
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, default=_json_default),
        encoding="utf-8",
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description="Final Seen/B/M/U/D evaluation for AntNavObstacle TD3 policies.")
    parser.add_argument("--checkpoint", required=True, help="Path to TD3 policy state_dict checkpoint.")
    parser.add_argument("--config", default=None, help="Optional run config. Inferred from checkpoint path if omitted.")
    parser.add_argument("--output-dir", required=True, help="Output directory for benchmark summaries.")
    parser.add_argument("--benchmarks", default="Seen,B,M,U,D", help="Comma-separated benchmark families.")
    parser.add_argument("--num-episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=700)
    parser.add_argument("--num-videos", type=int, default=3)
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wandb", action="store_true", help="Log benchmark metrics to wandb.")
    parser.add_argument("--wandb-project", default="DivPolicy-Ant")
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-group", default="ant_final_eval")
    args = parser.parse_args()

    checkpoint = pathlib.Path(args.checkpoint).expanduser().resolve()
    output_dir = pathlib.Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg, config_path = _load_config(checkpoint, args.config)
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    policy = _load_policy(cfg, checkpoint, device)
    benchmarks = [item.strip() for item in args.benchmarks.split(",") if item.strip()]

    summaries: Dict[str, Dict[str, float]] = {}
    for family in benchmarks:
        print(f"\n==> Evaluating Ant benchmark {family}")
        summary = evaluate_family(
            policy=policy,
            cfg=cfg,
            family=family,
            output_dir=output_dir / family,
            device=device,
            seed=args.seed,
            num_episodes=args.num_episodes,
            max_steps=args.max_steps,
            num_videos=args.num_videos,
            video_fps=args.video_fps,
        )
        summaries[family] = summary
        print(
            f"[{family}] success_rate={summary['success_rate']:.4f}, "
            f"collision_rate={summary['collision_rate']:.4f}, "
            f"mean_reward={summary['mean_reward']:.4f}, "
            f"mean_final_distance={summary['mean_final_distance']:.4f}"
        )

    heldout = [name for name in benchmarks if name != "Seen"]
    aggregate_names = heldout or benchmarks
    aggregate = {
        "avg_success_rate": float(np.mean([summaries[name]["success_rate"] for name in aggregate_names]))
        if aggregate_names
        else 0.0,
        "avg_collision_rate": float(np.mean([summaries[name]["collision_rate"] for name in aggregate_names]))
        if aggregate_names
        else 0.0,
        "avg_mean_reward": float(np.mean([summaries[name]["mean_reward"] for name in aggregate_names]))
        if aggregate_names
        else 0.0,
        "avg_final_distance": float(np.mean([summaries[name]["mean_final_distance"] for name in aggregate_names]))
        if aggregate_names
        else 0.0,
        "families": aggregate_names,
    }
    payload = {
        "mode": "ant_policy_only",
        "policy_checkpoint": str(checkpoint),
        "config_path": str(config_path),
        "benchmarks": summaries,
        "aggregate": aggregate,
    }
    summary_path = output_dir / "benchmark_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    if args.wandb:
        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name or output_dir.name,
            group=args.wandb_group,
            config={
                "checkpoint": str(checkpoint),
                "config_path": str(config_path),
                "benchmarks": benchmarks,
                "num_episodes": args.num_episodes,
                "max_steps": args.max_steps,
                "seed": args.seed,
            },
        )
        log_data = {}
        for family, summary in summaries.items():
            prefix = f"benchmark/{family}"
            for key in (
                "success_rate",
                "collision_rate",
                "unhealthy_rate",
                "timeout_rate",
                "mean_reward",
                "mean_final_distance",
                "mean_progress_ratio",
                "mean_path_efficiency",
            ):
                log_data[f"{prefix}/{key}"] = summary[key]
        for key, value in aggregate.items():
            if isinstance(value, (int, float)):
                log_data[f"benchmark/AVG/{key}"] = value
        run.log(log_data)
        run.finish()
    print("\n===== Ant Final Benchmark Summary =====")
    for family in benchmarks:
        summary = summaries[family]
        print(
            f"[{family}] success_rate={summary['success_rate']:.4f}, "
            f"collision_rate={summary['collision_rate']:.4f}, "
            f"mean_reward={summary['mean_reward']:.4f}"
        )
    print(
        f"[AVG] success_rate={aggregate['avg_success_rate']:.4f}, "
        f"collision_rate={aggregate['avg_collision_rate']:.4f}, "
        f"mean_reward={aggregate['avg_mean_reward']:.4f}"
    )
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
