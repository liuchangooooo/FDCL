"""
Skill-coverage + trajectory-DIVERSITY diagnostic for DIVO Push-T (stage-1 only).

Extends measure_coverage_gap.py with the missing piece: are the skills that
solve a scene behaviourally DISTINCT (different routes), or just z-noise on the
same route?  Without this, "skill library" is not justified.

Per scene c, inject K skills z_k ~ N(0,I), roll out each on the FIXED scene,
record success_k AND the full state trajectory (T-block route+rotation,
state = [x, y, cos th, sin th]).  Then, among the SUCCESSFUL skills:

  within_div(c) = mean_{i<j} || e_i - e_j ||      (e = padded flattened traj)
  feasible/realized/deployed/gap as before.

For scale, across_div = mean pairwise distance between one representative
successful trajectory of DIFFERENT scenes.  ratio = within_div / across_div:
  ratio ~ 0  => all solving skills take the same route (NO real skill diversity)
  ratio high => z induces genuinely different solving routes (skills are real)

Because dynamics_randomization=False and start+obstacles are fixed, a rollout is
deterministic given z, so any across-z trajectory spread is purely the skill.

Run (use the divo env python):
  /home/hnu-w/anaconda3/envs/divo/bin/python analysis/skill_coverage/measure_coverage_diversity.py \
      --ckpt data/outputs/2026.04.29/llm_evolve_s0/checkpoints/best_eval.pt \
      --n_scenes 50 --k_skills 16 --obstacle_size 0.08
"""

import argparse
import json
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
    for _ in range(max_tries):
        cfg = [
            {"x": float(rng.uniform(-0.18, 0.18)), "y": float(rng.uniform(-0.18, 0.18))}
            for _ in range(num_obstacles)
        ]
        if env.is_obstacle_config_valid(cfg, start):
            return cfg
    return None


@torch.no_grad()
def rollout_fixed_z(env, policy, start, obstacle_cfg, z, device, max_steps=10):
    """Run one episode; return (success, state_trajectory[list of 4-vec])."""
    env.set_obstacle_config(obstacle_cfg)
    obs = env.reset(tblock_pos=np.asarray(start, dtype=np.float64), force_tblock_pos=True)
    done = False
    steps = 0
    info = {"success": False}
    traj = []
    while not done:
        obs_th = torch.as_tensor(obs, dtype=torch.float32, device=device)
        if obs_th.ndim == 1:
            obs_th = obs_th.unsqueeze(0)
        state = env.obs2state(obs_th)
        traj.append(state.detach().cpu().numpy().reshape(-1)[:4].astype(np.float64))
        z_use = policy.encoder(obs_th) if z is None else z
        action = policy.decoder(torch.cat([state, z_use], dim=-1))
        obs, reward, done, info = env.step(action.detach().cpu().numpy()[0])
        steps += 1
        if steps >= max_steps:
            done = True
    return bool(info["success"]), traj


def embed_traj(traj, max_steps):
    """Pad/truncate a state trajectory to fixed length, flatten to a vector."""
    arr = np.asarray(traj, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return np.zeros(max_steps * 4)
    if arr.shape[0] < max_steps:
        pad = np.repeat(arr[-1:], max_steps - arr.shape[0], axis=0)
        arr = np.concatenate([arr, pad], axis=0)
    else:
        arr = arr[:max_steps]
    return arr.reshape(-1)


def mean_pairwise_dist(embs):
    if len(embs) < 2:
        return None
    E = np.stack(embs, axis=0)
    d = 0.0
    n = 0
    for i in range(len(E)):
        for j in range(i + 1, len(E)):
            d += float(np.linalg.norm(E[i] - E[j]))
            n += 1
    return d / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n_scenes", type=int, default=50)
    ap.add_argument("--k_skills", type=int, default=16)
    ap.add_argument("--obstacle_num", type=int, default=2)
    ap.add_argument("--obstacle_size", type=float, default=0.08)
    ap.add_argument("--max_steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = build_env(args.obstacle_num, args.obstacle_size)
    policy, latent_dim = load_policy(env, args.ckpt, device)
    print(f"[setup] ckpt={args.ckpt}")
    print(f"[setup] latent_dim={latent_dim} obstacle_num={args.obstacle_num} "
          f"obstacle_size={args.obstacle_size} n_scenes={args.n_scenes} k={args.k_skills}")

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
        succ_embs = []
        for _ in range(args.k_skills):
            z = torch.randn(1, latent_dim, dtype=torch.float32, device=device)
            ok, traj = rollout_fixed_z(env, policy, start, cfg, z, device, args.max_steps)
            successes.append(int(ok))
            if ok:
                succ_embs.append(embed_traj(traj, args.max_steps))
        deployed_ok, _ = rollout_fixed_z(env, policy, start, cfg, None, device, args.max_steps)

        feasible = int(max(successes)) if successes else 0
        realized = float(np.mean(successes)) if successes else 0.0
        within_div = mean_pairwise_dist(succ_embs)
        rows.append({
            "scene": scene_idx,
            "feasible": feasible,
            "realized": realized,
            "deployed": int(deployed_ok),
            "gap": feasible - realized,
            "n_success": int(sum(successes)),
            "within_div": within_div,
            "rep_emb": succ_embs[0].tolist() if succ_embs else None,
        })
        scene_idx += 1
        if scene_idx % 5 == 0:
            wd = f"{within_div:.3f}" if within_div is not None else "NA"
            print(f"[progress] {scene_idx}/{args.n_scenes} "
                  f"feasible={feasible} realized={realized:.2f} deployed={int(deployed_ok)} within_div={wd}")

    n = len(rows)
    if n == 0:
        print("No valid scenes.")
        return

    feasible_arr = np.array([r["feasible"] for r in rows], dtype=float)
    realized_arr = np.array([r["realized"] for r in rows], dtype=float)
    deployed_arr = np.array([r["deployed"] for r in rows], dtype=float)
    gap_arr = feasible_arr - realized_arr

    within_vals = [r["within_div"] for r in rows if r["within_div"] is not None]
    # across-scene diversity: pairwise distance between representative trajs of different scenes
    reps = [np.asarray(r["rep_emb"]) for r in rows if r["rep_emb"] is not None]
    across_div = mean_pairwise_dist(reps)
    within_mean = float(np.mean(within_vals)) if within_vals else 0.0
    ratio = (within_mean / across_div) if (across_div and across_div > 1e-9) else 0.0

    feas_mask = feasible_arr > 0.5
    opp_mask = feas_mask & (deployed_arr < 0.5)

    summary = {
        "ckpt": args.ckpt,
        "n_scenes": n,
        "k_skills": args.k_skills,
        "obstacle_size": args.obstacle_size,
        "mean_feasible": float(feasible_arr.mean()),
        "mean_realized": float(realized_arr.mean()),
        "mean_deployed": float(deployed_arr.mean()),
        "mean_gap": float(gap_arr.mean()),
        "frac_coverage_opportunity": float(opp_mask.mean()),
        "within_div_mean": within_mean,
        "across_div": across_div if across_div is not None else 0.0,
        "within_over_across_ratio": ratio,
        "n_scenes_div_measurable": len(within_vals),
    }

    print("\n===== Skill-Coverage + Diversity Summary =====")
    print(f"scenes evaluated            : {n}")
    print(f"mean feasible (any skill)   : {summary['mean_feasible']:.3f}")
    print(f"mean realized (avg skill)   : {summary['mean_realized']:.3f}")
    print(f"mean deployed (encoder z)   : {summary['mean_deployed']:.3f}")
    print(f"mean coverage gap           : {summary['mean_gap']:.3f}")
    print(f"coverage-opportunity scenes : {summary['frac_coverage_opportunity']:.3f}")
    print(f"within-scene route diversity: {summary['within_div_mean']:.3f}  "
          f"(mean over {len(within_vals)} scenes with >=2 solving skills)")
    print(f"across-scene route diversity: {summary['across_div']:.3f}")
    print(f"within/across ratio         : {summary['within_over_across_ratio']:.3f}  "
          f"(->0 same route; high => skills take distinct routes)")

    out_dir = pathlib.Path(args.out) if args.out else (
        REPO_ROOT / "analysis" / "skill_coverage" / "results_diversity")
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = pathlib.Path(args.ckpt).parts[-3] if len(pathlib.Path(args.ckpt).parts) >= 3 else "run"
    json_path = out_dir / f"coverage_div_{tag}.json"
    json_path.write_text(json.dumps({"summary": summary, "scenes": rows}, indent=2), encoding="utf-8")
    print(f"\nsaved: {json_path}")


if __name__ == "__main__":
    main()
