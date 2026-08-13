import argparse
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(ROOT)
ENVIRONMENTS_ROOT = os.path.join(ROOT, "environments")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if ENVIRONMENTS_ROOT not in sys.path:
    sys.path.insert(0, ENVIRONMENTS_ROOT)

import gymnasium as gym

import Curriculum  # noqa: F401  # registers Curriculum/* environments
from DIVO.env.antmaze import format_maze_map
from utils.antmaze_generator_utils import make_antmaze_env_kwargs_from_generator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator", required=True, help="Path to a generate_maze_map Python file.")
    parser.add_argument("--env-id", default="Curriculum/AntMaze_UMaze")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-map", default=None)
    args = parser.parse_args()

    env_kwargs = make_antmaze_env_kwargs_from_generator(
        args.generator,
        seed=args.seed,
        save_map_path=args.save_map,
    )
    print("[AntMaze] Generated maze_map:")
    print(format_maze_map(env_kwargs["maze_map"]))

    env = gym.make(args.env_id, **env_kwargs)
    obs, info = env.reset(seed=args.seed)
    print("[AntMaze] gym.make/reset succeeded")
    print(f"[AntMaze] observation keys: {list(obs.keys())}")
    print(f"[AntMaze] success at reset: {info.get('success')}")
    env.close()


if __name__ == "__main__":
    main()
