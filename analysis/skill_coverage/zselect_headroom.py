"""Discriminative experiment: does selecting z from the library beat encoder(obs)?

For a trained DIVO stage-1 policy (+ its TD3 critic), on fixed scenes we compare
three deployment-time z-selection strategies:

  deployed     : z = encoder(obs)              (current bare deployment)
  bestK_critic : each step, sample K z~N(0,I), pick the action with the highest
                 min-over-critics Q(obs, a)    (training-free best-of-K via critic)
  feasible      : oracle upper bound = max over K fixed-z rollouts succeeds
  realized     : mean over K fixed-z rollouts  (library average, for reference)

If bestK_critic ~ deployed and feasible ~ deployed -> z-selection has no headroom,
encoder is already near the library ceiling (route A). If feasible >> deployed and
bestK_critic recovers some of it -> z-selection helps (route B worth pursuing).

Runs on the policy's NATIVE env (pusht_mujoco_llm, obstacle_num from ckpt cfg,
matched dims) so the critic input is clean (no eval-time obstacle-feature padding).

  python analysis/skill_coverage/zselect_headroom.py \
      --ckpt <full .ckpt with critic> --sizes 0.05 0.08 --n_scenes 40 --k_skills 16
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

from DIVO.policy import get_policy
from DIVO.critic import get_critic
from analysis.skill_coverage.measure_coverage_diversity import build_env, sample_obstacles


def load_policy_and_critic(env, ckpt_path, device):
    import dill
    run_dir = ckpt_path.rsplit("/", 2)[0]
    cfg = OmegaConf.load(f"{run_dir}/.hydra/config.yaml")
    policy = get_policy(env, **cfg.policy).to(device)
    critic = get_critic(**cfg.critic).to(device)
    payload = torch.load(open(ckpt_path, "rb"), pickle_module=dill, map_location=device)
    sd = payload["state_dicts"]
    policy.load_state_dict(sd["model"])
    critic.load_state_dict(sd["critic"])
    policy.eval()
    critic.eval()
    latent_dim = int(cfg.policy.encoder_net.out_chan)
    return policy, critic, latent_dim


@torch.no_grad()
def _obs_state(env, obs, device):
    obs_th = torch.as_tensor(obs, dtype=torch.float32, device=device)
    if obs_th.ndim == 1:
        obs_th = obs_th.unsqueeze(0)
    state = env.obs2state(obs_th)
    return obs_th, state


@torch.no_grad()
def rollout(env, policy, critic, start, cfg, mode, z_bank, device, max_steps=10):
    """mode in {'deployed','bestK','fixed'}. For 'fixed', z_bank is a single z."""
    env.set_obstacle_config(cfg)
    obs = env.reset(tblock_pos=np.asarray(start, dtype=np.float64), force_tblock_pos=True)
    done, steps, info = False, 0, {"success": False}
    while not done:
        obs_th, state = _obs_state(env, obs, device)
        if mode == "deployed":
            z = policy.encoder(obs_th)
            action = policy.decoder(torch.cat([state, z], dim=-1))
        elif mode == "fixed":
            action = policy.decoder(torch.cat([state, z_bank], dim=-1))
        else:  # bestK via critic
            K = z_bank.shape[0]
            state_rep = state.repeat(K, 1)
            obs_rep = obs_th.repeat(K, 1)
            acts = policy.decoder(torch.cat([state_rep, z_bank], dim=-1))
            qs = critic(obs_rep, acts)  # list of [K,1]
            qmin = torch.min(torch.stack([q.squeeze(-1) for q in qs], dim=0), dim=0).values
            best = int(torch.argmax(qmin).item())
            action = acts[best:best + 1]
        obs, _, done, info = env.step(action.detach().cpu().numpy()[0])
        steps += 1
        if steps >= max_steps:
            done = True
    return bool(info["success"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--sizes", type=float, nargs="+", default=[0.05, 0.08])
    ap.add_argument("--n_scenes", type=int, default=40)
    ap.add_argument("--k_skills", type=int, default=16)
    ap.add_argument("--obstacle_num", type=int, default=2)
    ap.add_argument("--max_steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[setup] ckpt={args.ckpt}")
    print(f"{'size':>6} | {'deployed':>9} {'bestK':>9} {'feasible':>9} {'realized':>9} | {'bestK-dep':>9} {'feas-dep':>9}")
    print("-" * 78)

    all_rows = []
    for size in args.sizes:
        rng = np.random.default_rng(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        env = build_env(args.obstacle_num, size)
        policy, critic, latent_dim = load_policy_and_critic(env, args.ckpt, device)

        dep, best, feas, real = [], [], [], []
        n, attempts = 0, 0
        while n < args.n_scenes and attempts < args.n_scenes * 25:
            attempts += 1
            start = env.sample_valid_tblock_pose()
            cfg = sample_obstacles(env, start, args.obstacle_num, rng)
            if cfg is None:
                continue
            z_bank = torch.randn(args.k_skills, latent_dim, dtype=torch.float32, device=device)
            dep.append(rollout(env, policy, critic, start, cfg, "deployed", None, device, args.max_steps))
            best.append(rollout(env, policy, critic, start, cfg, "bestK", z_bank, device, args.max_steps))
            succ = [rollout(env, policy, critic, start, cfg, "fixed", z_bank[k:k+1], device, args.max_steps)
                    for k in range(args.k_skills)]
            feas.append(1.0 if max(succ) else 0.0)
            real.append(float(np.mean(succ)))
            n += 1

        d, b, f, r = np.mean(dep), np.mean(best), np.mean(feas), np.mean(real)
        print(f"{size:>6.2f} | {d:>9.3f} {b:>9.3f} {f:>9.3f} {r:>9.3f} | {b-d:>+9.3f} {f-d:>+9.3f}")
        all_rows.append({"size": size, "n": n, "deployed": d, "bestK_critic": b,
                         "feasible": f, "realized": r, "bestK_minus_deployed": b - d,
                         "feasible_minus_deployed": f - d})

    out = pathlib.Path(args.out) if args.out else (
        REPO_ROOT / "analysis" / "skill_coverage" / "results_zselect" /
        f"zselect_{pathlib.Path(args.ckpt).parts[-3]}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"ckpt": args.ckpt, "k_skills": args.k_skills,
                               "obstacle_num": args.obstacle_num, "rows": all_rows}, indent=2))
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
