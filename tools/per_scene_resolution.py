"""Per-environment resolution: single-policy feedback vs. the skill library.

Claim under test (Q2, stated at the per-environment level): a single deployed
policy labels each environment with a binary outcome, so all environments it
solves look alike and provide no basis for telling them apart. Probing the same
environment with the K skills of the library instead yields a graded response
r = (#skills that solve it) / K, which can separate environments that the single
policy scores identically.

Protocol: one probe pass over M layouts drawn from a fixed generator, with the
policy frozen. Every layout is rolled out with the deployment skill w_0 (the
single-policy reading) and with each probe skill w_1..w_K (the library reading),
all under the same start and obstacle configuration, without exploration noise.
Environments are then grouped by the w_0 outcome and the spread of r within each
group is reported. Because this is a statement about the distribution of r within
a single-policy label class, it does not depend on ranking generators and is
therefore unaffected by the probe noise that limits generator-level comparisons.

Usage:
  python3 tools/per_scene_resolution.py --run data/outputs/<date>/<run> \
      [--ckpt <file>] [--generator <file>] [-M 60] [--seed 0]
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch
from omegaconf import OmegaConf

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

try:
    OmegaConf.register_new_resolver("now", lambda p: "now")
except Exception:
    pass

from DIVO.env import get_env_class
from DIVO.policy import get_policy
from DIVO.curriculum.candidate_verifier import SandboxPushTExecutor
from DIVO.curriculum.skill_signal import (
    SceneSamplingStats,
    build_z_bank,
    rollout_scene_with_skills,
    rollout_scene_with_z_bank,
    sample_generator_scene,
)


def pick_checkpoint(run: pathlib.Path) -> pathlib.Path:
    ckpt_dir = run / "checkpoints"
    latest = ckpt_dir / "model_latest.pt"
    if latest.exists():
        return latest
    cands = sorted(ckpt_dir.glob("model_epoch=*.pt"))
    if not cands:
        raise FileNotFoundError(f"no checkpoint under {ckpt_dir}")
    return max(cands, key=lambda p: p.stat().st_mtime)


def probe(env, policy, executor, K, M, device, max_steps, seed, max_tries_mult=25,
          probe_source="w_probe", latent_dim=3):
    """Probe M layouts. ``probe_source`` selects the library being measured.

    ``w_probe``  : the trained skill codebook w_1..w_K (what the method defines).
    ``random_z`` : a bank of random latents, i.e. untrained behaviours. Kept as a
                   contrast because a graded response obtained this way does not
                   evidence anything about the *learned* library.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    stats = SceneSamplingStats()
    z_bank = None
    if probe_source == "random_z":
        z_bank = build_z_bank(K=K, latent_dim=latent_dim, seed=seed, device=device)
    scenes = []
    tries = 0
    while len(scenes) < M and tries < M * max_tries_mult:
        tries += 1
        sc = sample_generator_scene(env, executor, 2, timeout_sec=5, stats=stats)
        if sc is None:
            continue
        if probe_source == "random_z":
            res = rollout_scene_with_z_bank(
                env=env,
                policy=policy,
                start=sc["start"],
                obstacles=sc["obstacles"],
                z_bank=z_bank,
                device=device,
                max_steps=max_steps,
            )
        else:
            res = rollout_scene_with_skills(
                env=env,
                policy=policy,
                start=sc["start"],
                obstacles=sc["obstacles"],
                K=K,
                device=device,
                max_steps=max_steps,
            )
        y = [int(bool(r["success"])) for r in res["routes"]]
        scenes.append(
            {
                "y": y,
                "r": float(np.mean(y)),
                "w0": int(bool(res["deployed_success"])),
            }
        )
        if len(scenes) % 10 == 0:
            print(f"  probed {len(scenes)}/{M}", flush=True)
    return scenes, stats


def report(scenes, K):
    r = np.array([s["r"] for s in scenes])
    w0 = np.array([s["w0"] for s in scenes])
    n = len(scenes)

    print()
    print("=" * 70)
    print("Per-environment resolution: single policy vs skill library")
    print("=" * 70)
    print(f"  probe scenes M = {n},  library size K = {K}")
    print(f"  deployment (w_0) success rate = {w0.mean():.3f}")
    print(f"  mean library response r       = {r.mean():.3f}")
    # Table-2 style solvability columns, measured with the trained library.
    feasible = np.array([1.0 if max(s["y"]) > 0 else 0.0 for s in scenes])
    print()
    print(f"  feasible (>=1 library skill solves it) = {feasible.mean():.3f}")
    print(f"  realized (mean fraction of skills)     = {r.mean():.3f}")
    print(f"  deployed (w_0)                         = {w0.mean():.3f}")
    print(f"  gap = feasible - deployed              = "
          f"{feasible.mean() - w0.mean():+.3f}")
    print()

    levels = sorted(set(np.round(r, 6)))
    print(f"  distinct per-environment readings")
    print(f"    single policy (w_0 outcome) : 2   {{0, 1}}")
    print(f"    skill library (r)          : {len(levels)}   "
          f"{[round(float(x), 3) for x in levels]}")
    print()

    for label, mask in (("w_0 SUCCEEDS", w0 == 1), ("w_0 FAILS", w0 == 0)):
        m = int(mask.sum())
        print(f"  --- environments where {label}  (n = {m}, "
              f"{100.0 * m / max(n, 1):.0f}% of probe) ---")
        if m == 0:
            print("      none")
            print()
            continue
        rr = r[mask]
        print("      library response r within this single-policy label class:")
        for k in range(K + 1):
            v = k / K
            c = int((np.abs(rr - v) < 1e-9).sum())
            bar = "#" * int(round(40.0 * c / m))
            print(f"        r={v:4.2f} : {c:4d} ({100.0 * c / m:5.1f}%) {bar}")
        distinct = len(set(np.round(rr, 6)))
        differentiated = float(np.mean((rr > 0) & (rr < 1)))
        print(f"      distinct r values in this class : {distinct}")
        print(f"      of these environments, the library disagrees internally "
              f"(0 < r < 1): {100.0 * differentiated:.1f}%")
        print(f"      r range in this class           : "
              f"[{rr.min():.2f}, {rr.max():.2f}]   sd = {rr.std():.3f}")
        print()

    # The headline number: among environments the single policy calls solved,
    # how many are NOT uniformly solved by the library.
    solved = w0 == 1
    if solved.sum():
        rs = r[solved]
        frac = float(np.mean(rs < 1.0 - 1e-9))
        print("  headline")
        print(f"    among the {int(solved.sum())} environments the deployed policy solves,")
        print(f"    {100.0 * frac:.1f}% are NOT solved by every library skill, i.e. the")
        print(f"    library separates environments that single-policy feedback")
        print(f"    labels identically.")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="training run directory")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--generator", default=None)
    ap.add_argument("-M", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--probe", choices=["w_probe", "random_z"], default="w_probe")
    ap.add_argument("--probe-K", type=int, default=None,
                    help="override library size (random_z only)")
    ap.add_argument("--obstacle-size", type=float, default=None)
    args = ap.parse_args()

    run = pathlib.Path(args.run)
    cfg_path = run / ".hydra" / "config.yaml"
    ckpt = pathlib.Path(args.ckpt) if args.ckpt else pick_checkpoint(run)
    gen = pathlib.Path(args.generator) if args.generator else run / "current_generator.py"

    print(f"run       : {run}")
    print(f"config    : {cfg_path}")
    print(f"checkpoint: {ckpt.name}")
    print(f"generator : {gen}")

    cfg = OmegaConf.load(str(cfg_path))
    K = int(cfg.skill.K)
    if not bool(cfg.skill.skill_enabled):
        raise RuntimeError("this run is not skill-enabled; w_probe needs trained skills")
    max_steps = int(cfg.max_steps)
    obstacle_size = float(cfg.env.obstacle_size) if "obstacle_size" in cfg.env else 0.01

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_cfg = OmegaConf.to_container(cfg.env, resolve=True)
    if args.obstacle_size is not None:
        # The simulated obstacle geometry lives in the env, not in the sandbox
        # executor; both must be set for an obstacle-size sweep to be meaningful.
        env_cfg["obstacle_size"] = float(args.obstacle_size)
    env = get_env_class(**env_cfg)
    policy = get_policy(env, **OmegaConf.to_container(cfg.policy, resolve=True)).to(device)
    policy.load_state_dict(torch.load(str(ckpt), map_location=device), strict=False)
    policy.eval()
    if not getattr(policy, "skill_enabled", False):
        raise RuntimeError("loaded policy is not skill_enabled")

    if args.obstacle_size is not None:
        obstacle_size = float(args.obstacle_size)
    if args.probe == "random_z" and args.probe_K:
        K = int(args.probe_K)

    ex = SandboxPushTExecutor(obstacle_size=obstacle_size)
    ex.load(gen.read_text(encoding="utf-8"))

    print(f"probe={args.probe}, K={K}, M={args.M}, max_steps={max_steps}, "
          f"obstacle_size={obstacle_size} ...", flush=True)
    scenes, stats = probe(
        env, policy, ex, K, args.M, device, max_steps, args.seed,
        probe_source=args.probe, latent_dim=int(cfg.latent_dim),
    )
    print(f"sampling stats: {stats.to_dict()}")
    report(scenes, K)

    if args.out:
        pathlib.Path(args.out).write_text(
            json.dumps({"K": K, "M": len(scenes), "scenes": scenes}, indent=2)
        )
        print(f"written: {args.out}")


if __name__ == "__main__":
    main()
