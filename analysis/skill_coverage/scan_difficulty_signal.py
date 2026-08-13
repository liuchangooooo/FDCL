"""
Phase-0 signal scan: does the probe produce a non-empty boundary/mastery signal,
and at which obstacle SIZE does it appear?

For a fixed starting-point policy (e.g. llm_static_s0, trained obstacle_num=2),
sweep obstacle_size and, per scene, inject K skills z~N(0,I), measure
feasible/realized, and label each scene:
  infeasible : feasible == 0                  (no skill solves -> avoid/too hard)
  mastery    : realized >= high               (library already solves it -> harden)
  boundary   : feasible==1 and realized<high  (library split -> learnable frontier)

Reports the occupancy fractions per size = the "operable difficulty interval".

Run:
  /home/hnu-w/anaconda3/envs/divo/bin/python analysis/skill_coverage/scan_difficulty_signal.py \
      --ckpt data/outputs/2026.04.29/llm_static_s0/checkpoints/best_eval.pt \
      --sizes 0.01 0.05 0.08 --n_scenes 40 --k_skills 12
"""

import argparse
import sys
import pathlib

import numpy as np
import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from analysis.skill_coverage.measure_coverage_diversity import (
    build_env, load_policy, sample_obstacles, rollout_fixed_z,
)


def scan_one_size(env, policy, latent_dim, n_scenes, k_skills, max_steps, high, rng, device):
    z_bank = torch.randn(k_skills, latent_dim, dtype=torch.float32, device=device)
    rows = []
    scene_idx = 0
    attempts = 0
    while scene_idx < n_scenes and attempts < n_scenes * 25:
        attempts += 1
        start = env.sample_valid_tblock_pose()
        cfg = sample_obstacles(env, start, env.task.obstacle_num, rng)
        if cfg is None:
            continue
        succ = []
        for k in range(k_skills):
            ok, _ = rollout_fixed_z(env, policy, start, cfg, z_bank[k:k+1], device, max_steps)
            succ.append(int(ok))
        deployed_ok, _ = rollout_fixed_z(env, policy, start, cfg, None, device, max_steps)
        feasible = int(max(succ))
        realized = float(np.mean(succ))
        if feasible == 0:
            label = "infeasible"
        elif realized >= high:
            label = "mastery"
        else:
            label = "boundary"
        rows.append({"feasible": feasible, "realized": realized,
                     "deployed": int(deployed_ok), "label": label})
        scene_idx += 1
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--sizes", type=float, nargs="+", default=[0.01, 0.05, 0.08])
    ap.add_argument("--obstacle_num", type=int, default=2)
    ap.add_argument("--n_scenes", type=int, default=40)
    ap.add_argument("--k_skills", type=int, default=12)
    ap.add_argument("--max_steps", type=int, default=10)
    ap.add_argument("--high", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[setup] ckpt={args.ckpt}")
    print(f"[setup] obstacle_num={args.obstacle_num} sizes={args.sizes} "
          f"n_scenes={args.n_scenes} k={args.k_skills} high={args.high}\n")

    print(f"{'size':>6} | {'feasible':>8} {'realized':>8} {'deployed':>8} | "
          f"{'infeas%':>8} {'bound%':>8} {'mastery%':>8}")
    print("-" * 70)

    results = []
    for size in args.sizes:
        rng = np.random.default_rng(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        env = build_env(args.obstacle_num, size)
        policy, latent_dim = load_policy(env, args.ckpt, device)
        rows = scan_one_size(env, policy, latent_dim, args.n_scenes, args.k_skills,
                             args.max_steps, args.high, rng, device)
        n = len(rows)
        if n == 0:
            print(f"{size:>6} |  (no valid scenes)")
            continue
        feas = np.mean([r["feasible"] for r in rows])
        real = np.mean([r["realized"] for r in rows])
        dep = np.mean([r["deployed"] for r in rows])
        labels = [r["label"] for r in rows]
        f_inf = labels.count("infeasible") / n
        f_bnd = labels.count("boundary") / n
        f_mas = labels.count("mastery") / n
        print(f"{size:>6.2f} | {feas:>8.3f} {real:>8.3f} {dep:>8.3f} | "
              f"{f_inf:>8.2f} {f_bnd:>8.2f} {f_mas:>8.2f}")
        results.append((size, feas, real, dep, f_inf, f_bnd, f_mas, n))

    print("\nInterpretation: boundary% is the operable curriculum signal; "
          "all-mastery (low size) or all-infeasible (high size) = empty signal.")


if __name__ == "__main__":
    main()
