from __future__ import annotations

import argparse
import json

import numpy as np

from DIVO.ant import AntNavObstacleEnv


def main():
    parser = argparse.ArgumentParser(description="Sanity-check AntNavObstacleEnv.")
    parser.add_argument("--base-env-id", default="Ant-v4")
    parser.add_argument("--mode", default="train", choices=["none", "train", "seen", "B", "M", "U", "D"])
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    env = AntNavObstacleEnv(
        base_env_id=args.base_env_id,
        obstacle=args.mode != "none",
        obstacle_mode=args.mode,
        render_mode="rgb_array" if args.render else None,
    )
    obs, info = env.reset(seed=args.seed)
    print("base_env_id:", env.base_env_id)
    print("action_dim:", env.action_dim)
    print("obs_dim:", env.obs_dim)
    print("obs_shape:", obs.shape)
    print("state_dim:", env.state_dim)
    print("start_xy:", info["start_xy"])
    print("goal_xy:", info["goal_xy"])
    print("obstacles:", json.dumps(info["obstacles"], indent=2))

    total_reward = 0.0
    final_info = info
    for step in range(args.steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, final_info = env.step(action)
        total_reward += reward
        if args.render:
            frame = env.render()
            print("render_frame:", None if frame is None else frame.shape)
        print(
            f"step={step:03d} reward={reward:.3f} dist={final_info['goal_distance']:.3f} "
            f"progress={final_info['progress_ratio']:.3f} collision={final_info.get('collision', False)} "
            f"success={final_info.get('success', False)} term={terminated} trunc={truncated}"
        )
        if terminated or truncated:
            break

    print("total_reward:", float(total_reward))
    print("final_xy:", np.asarray(final_info["xy_position"]).tolist())
    env.close()


if __name__ == "__main__":
    main()
