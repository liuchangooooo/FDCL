"""
Skill-coverage gap diagnostic for DIVO Push-T (stage-1 latent policy only).

Question being tested (the "B-door" novelty hypothesis):
  On Push-T, are there scenes that are solvable by SOME skill in the latent
  library (feasible) but NOT reliably solved by an average / deployed skill
  (realized)?  A large feasible-realized gap => skill-coverage is a real,
  measurable axis to generate environments around.  A ~0 gap => that door is
  also closed and we stop chasing it.

Method (no stage-2 sampler, no stage-3):
  - Load a trained LatentDetPolicy (encoder->z, decoder([state, z])).
  - Build a batch of FIXED scenes (obstacle layout + start pose fixed).
  - For each scene, inject K skills z_k ~ N(0, I)^latent_dim, roll out once each,
    record success_k.  Also run the deterministic deployed policy (encoder z).
  - feasible(c)  = max_k success_k          (does any library skill solve it)
  - realized(c)  = mean_k success_k         (random-skill average success)
  - deployed(c)  = success of encoder(obs) z (what stage-1 actually deploys)
  - gap(c)       = feasible(c) - realized(c)

Run:
  /home/hnu-w/anaconda3/envs/divo/bin/python analysis/skill_coverage/measure_coverage_gap.py \
      --ckpt data/outputs/2026.04.29/llm_evolve_s0/checkpoints/best_eval.pt \
      --n_scenes 40 --k_skills 12 --obstacle_size 0.05
"""

import argparse
import json
import os
import sys
import pathlib

import numpy as np
import torch
from omegaconf import OmegaConf

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from DIVO.env import get_env_class
from DIVO.policy import get_policy


def build_env(obstacle_num, obstacle_size):
    return get_env_class(
        _target_="pusht_mujoco_llm",
        obstacle=True,
        obstacle_num=obstacle_num,
        obstacle_size=obstacle_size,
        obstacle_shape="box",
        obstacle_dist="between",
        action_scale=4,
        NUM_SUBSTEPS=25,
        action_dim=[6],
        obs_dim=[4 + 2 * obstacle_num],
        action_reg=True,
        reg_coeff=1.0,
        dynamics_randomization=False,  # isolate z as the only source of variation
    )


def load_policy(env, ckpt_path, device):
    run_dir = ckpt_path.rsplit("/", 2)[0]
    p_cfg = OmegaConf.load(f"{run_dir}/.hydra/config.yaml")
    policy = get_policy(env, **p_cfg.policy).to(device)
    policy.load_state_dict(torch.load(ckpt_path, map_location=device))
    policy.eval()
    latent_dim = int(p_cfg.policy.encoder_net.out_chan)
    return policy, latent_dim


def sample_obstacles(env, start, num_obstacles, rng, max_tries=400):
    """Sample obstacles uniformly over the board, keep the first valid config.

    Unbiased over the valid region (matches the 'unseen random' eval regime).
    The goal/start exclusion zones are large, so we use rejection sampling.
    """
    for _ in range(max_tries):
        cfg = [
            {"x": float(rng.uniform(-0.18, 0.18)), "y": float(rng.uniform(-0.18, 0.18))}
            for _ in range(num_obstacles)
        ]
        if env.is_obstacle_config_valid(cfg, start):
            return cfg
    return None  # give up; caller resamples scene


@torch.no_grad()
def rollout_fixed_z(env, policy, start, obstacle_cfg, z, device, max_steps=10):
    """Run one episode on a fixed scene with a fixed injected skill z.
    If z is None, use the deployed deterministic encoder policy."""
    env.set_obstacle_config(obstacle_cfg)
    obs = env.reset(tblock_pos=np.asarray(start, dtype=np.float64), force_tblock_pos=True)
    done = False
    steps = 0
    info = {"success": False, "termination": "timeout"}
    while not done:
        obs_th = torch.as_tensor(obs, dtype=torch.float32, device=device)
        if obs_th.ndim == 1:
            obs_th = obs_th.unsqueeze(0)
        state = env.obs2state(obs_th)
        if z is None:
            z_use = policy.encoder(obs_th)
        else:
            z_use = z
        action = policy.decoder(torch.cat([state, z_use], dim=-1))
        action_np = action.detach().cpu().numpy()
        obs, reward, done, info = env.step(action_np[0])
        steps += 1
        if steps >= max_steps:
            done = True
    return bool(info["success"]), info.get("termination", "timeout")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n_scenes", type=int, default=40)
    ap.add_argument("--k_skills", type=int, default=12)
    ap.add_argument("--obstacle_num", type=int, default=2)
    ap.add_argument("--obstacle_size", type=float, default=0.05)
    ap.add_argument("--max_steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)   # env start-pose sampling uses global np.random
    torch.manual_seed(args.seed)

    env = build_env(args.obstacle_num, args.obstacle_size)
    policy, latent_dim = load_policy(env, args.ckpt, device)
    print(f"[setup] ckpt={args.ckpt}")
    print(f"[setup] latent_dim={latent_dim} obstacle_num={args.obstacle_num} "
          f"obstacle_size={args.obstacle_size} device={device}")
    print(f"[setup] n_scenes={args.n_scenes} k_skills={args.k_skills} max_steps={args.max_steps}")

    rows = []
    scene_idx = 0
    attempts = 0
    while scene_idx < args.n_scenes and attempts < args.n_scenes * 20:
        attempts += 1
        start = env.sample_valid_tblock_pose()
        cfg = sample_obstacles(env, start, args.obstacle_num, rng)
        if cfg is None:
            continue

        successes = []
        for _ in range(args.k_skills):
            z = torch.randn(1, latent_dim, dtype=torch.float32, device=device)
            ok, _term = rollout_fixed_z(env, policy, start, cfg, z, device, args.max_steps)
            successes.append(int(ok))
        deployed_ok, _ = rollout_fixed_z(env, policy, start, cfg, None, device, args.max_steps)

        # record the ACTUAL placed obstacle positions (env may nudge them)
        actual_obs = env.get_obstacle_positions()

        feasible = int(max(successes)) if successes else 0
        realized = float(np.mean(successes)) if successes else 0.0
        rows.append({
            "scene": scene_idx,
            "start": [float(v) for v in start],
            "obstacles_requested": cfg,
            "obstacles_actual": [{"x": float(o["x"]), "y": float(o["y"])} for o in actual_obs],
            "successes": successes,
            "feasible": feasible,
            "realized": realized,
            "deployed": int(deployed_ok),
            "gap": feasible - realized,
        })
        scene_idx += 1
        if scene_idx % 5 == 0:
            print(f"[progress] scenes={scene_idx}/{args.n_scenes} "
                  f"(last: feasible={feasible} realized={realized:.2f} deployed={int(deployed_ok)})")

    n = len(rows)
    if n == 0:
        print("No valid scenes generated.")
        return

    feasible_arr = np.array([r["feasible"] for r in rows], dtype=float)
    realized_arr = np.array([r["realized"] for r in rows], dtype=float)
    deployed_arr = np.array([r["deployed"] for r in rows], dtype=float)
    gap_arr = feasible_arr - realized_arr

    feas_mask = feasible_arr > 0.5
    # coverage-opportunity scenes: some skill solves it, but deployed policy fails
    opp_mask = feas_mask & (deployed_arr < 0.5)

    summary = {
        "ckpt": args.ckpt,
        "n_scenes": n,
        "k_skills": args.k_skills,
        "obstacle_num": args.obstacle_num,
        "obstacle_size": args.obstacle_size,
        "mean_feasible": float(feasible_arr.mean()),
        "mean_realized": float(realized_arr.mean()),
        "mean_deployed": float(deployed_arr.mean()),
        "mean_gap": float(gap_arr.mean()),
        "realized_given_feasible": float(realized_arr[feas_mask].mean()) if feas_mask.any() else 0.0,
        "deployed_given_feasible": float(deployed_arr[feas_mask].mean()) if feas_mask.any() else 0.0,
        "frac_coverage_opportunity": float(opp_mask.mean()),
    }

    print("\n===== Skill-Coverage Gap Summary =====")
    print(f"scenes evaluated            : {n}")
    print(f"mean feasible (any skill)   : {summary['mean_feasible']:.3f}   "
          f"<- fraction of scenes solvable by >=1 of {args.k_skills} skills")
    print(f"mean realized (avg skill)   : {summary['mean_realized']:.3f}   "
          f"<- avg success of a random skill draw")
    print(f"mean deployed (encoder z)   : {summary['mean_deployed']:.3f}   "
          f"<- what stage-1 actually deploys")
    print(f"mean coverage gap           : {summary['mean_gap']:.3f}   "
          f"(feasible - realized)")
    print(f"realized | feasible         : {summary['realized_given_feasible']:.3f}")
    print(f"deployed | feasible         : {summary['deployed_given_feasible']:.3f}")
    print(f"coverage-opportunity scenes : {summary['frac_coverage_opportunity']:.3f}   "
          f"<- solvable by some skill BUT deployed policy fails")

    out_dir = pathlib.Path(args.out) if args.out else (
        REPO_ROOT / "analysis" / "skill_coverage" / "results")
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = pathlib.Path(args.ckpt).parts[-3] if len(pathlib.Path(args.ckpt).parts) >= 3 else "run"
    json_path = out_dir / f"coverage_gap_{tag}.json"
    json_path.write_text(json.dumps({"summary": summary, "scenes": rows}, indent=2), encoding="utf-8")
    print(f"\nsaved: {json_path}")


if __name__ == "__main__":
    main()
