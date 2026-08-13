"""
Visualize the DIVO stage-1 skill library on Push-T.

Two panels that make "what the skill library looks like" legible:
  (A) Route overlay: on a few FIXED layouts, inject K skills z~N(0,I) and draw
      each T-block route, colored by z (3-D z -> RGB). Successful routes solid,
      failed routes faint dashed. Obstacles / start / target drawn to scale.
  (B) Trajectory embedding (t-SNE, PCA fallback) of all collected routes,
      colored by z (RGB), marker by layout. Shows z structures behavior.

This re-collects trajectories (the gap-diagnostic JSON only kept one rep traj),
because we need every per-z route + its z value.

Run:
  /home/hnu-w/anaconda3/envs/divo/bin/python analysis/skill_coverage/plot_skill_library.py \
      --ckpt data/outputs/2026.04.29/llm_evolve_s0/checkpoints/best_eval.pt \
      --k_skills 24 --n_layouts 3 --obstacle_size 0.08
"""

import argparse
import pathlib
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from analysis.skill_coverage.measure_coverage_diversity import (
    build_env, load_policy, sample_obstacles, embed_traj,
)

DESK = 0.25  # half-size; state xy = world / DESK


@torch.no_grad()
def rollout_route(env, policy, start, cfg, z, device, max_steps=10):
    """Return (success, route Nx2 in normalized state frame)."""
    env.set_obstacle_config(cfg)
    obs = env.reset(tblock_pos=np.asarray(start, dtype=np.float64), force_tblock_pos=True)
    done, steps = False, 0
    info = {"success": False}
    route = []
    while not done:
        obs_th = torch.as_tensor(obs, dtype=torch.float32, device=device)
        if obs_th.ndim == 1:
            obs_th = obs_th.unsqueeze(0)
        state = env.obs2state(obs_th)
        s = state.detach().cpu().numpy().reshape(-1)[:4]
        route.append(s[:2].astype(np.float64))
        z_use = policy.encoder(obs_th) if z is None else z
        action = policy.decoder(torch.cat([state, z_use], dim=-1))
        obs, _, done, info = env.step(action.detach().cpu().numpy()[0])
        steps += 1
        if steps >= max_steps:
            done = True
    return bool(info["success"]), np.asarray(route)


def z_to_rgb(zs):
    """Map array of 3-D z to RGB in [0,1] via per-dim min-max."""
    Z = np.asarray(zs, dtype=float)
    lo, hi = Z.min(0), Z.max(0)
    rng = np.where(hi - lo < 1e-6, 1.0, hi - lo)
    return np.clip((Z - lo) / rng, 0, 1)


def draw_layout(ax, start, obstacles_world, obstacle_size, routes, successes, zrgb, max_steps):
    half = obstacle_size / DESK  # obstacle half-extent in normalized frame
    # obstacles
    for o in obstacles_world:
        ox, oy = o["x"] / DESK, o["y"] / DESK
        if abs(ox) > 1.5 or abs(oy) > 1.5:  # nudged off-desk -> skip
            continue
        ax.add_patch(Rectangle((ox - half, oy - half), 2 * half, 2 * half,
                               facecolor="0.35", edgecolor="k", lw=0.8, zorder=2))
    # target (origin) and start
    ax.scatter([0], [0], marker="*", s=320, c="gold", edgecolor="k",
               lw=0.8, zorder=5, label="target")
    sx, sy = start[0] / DESK, start[1] / DESK
    ax.scatter([sx], [sy], marker="o", s=70, c="white", edgecolor="k",
               lw=1.2, zorder=5, label="start")
    # routes
    for r, ok, c in zip(routes, successes, zrgb):
        if r.shape[0] < 2:
            continue
        if ok:
            ax.plot(r[:, 0], r[:, 1], "-", color=c, lw=1.8, alpha=0.9, zorder=4)
            ax.scatter(r[-1, 0], r[-1, 1], color=c, s=16, zorder=4)
        else:
            ax.plot(r[:, 0], r[:, 1], "--", color="0.7", lw=0.8, alpha=0.5, zorder=3)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--k_skills", type=int, default=24)
    ap.add_argument("--n_layouts", type=int, default=3)
    ap.add_argument("--min_success", type=int, default=6)
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
    print(f"[setup] ckpt={args.ckpt} latent_dim={latent_dim} "
          f"k={args.k_skills} n_layouts={args.n_layouts} size={args.obstacle_size}")

    # fixed skill bank shared across layouts (so colors mean the same z everywhere)
    z_bank = torch.randn(args.k_skills, latent_dim, dtype=torch.float32, device=device)
    z_np = z_bank.cpu().numpy()

    layouts = []
    attempts = 0
    while len(layouts) < args.n_layouts and attempts < args.n_layouts * 40:
        attempts += 1
        start = env.sample_valid_tblock_pose()
        cfg = sample_obstacles(env, start, args.obstacle_num, rng)
        if cfg is None:
            continue
        routes, succ = [], []
        for k in range(args.k_skills):
            ok, r = rollout_route(env, policy, start, cfg, z_bank[k:k + 1], device, args.max_steps)
            routes.append(r); succ.append(int(ok))
        if sum(succ) < args.min_success:
            continue
        obstacles_world = env.get_obstacle_positions()
        layouts.append({"start": start, "obstacles": obstacles_world,
                        "routes": routes, "succ": succ})
        print(f"[layout {len(layouts)}] n_success={sum(succ)}/{args.k_skills}")

    if not layouts:
        print("No suitable layout found."); return

    zrgb = z_to_rgb(z_np)

    # ---- figure ----
    ncol = len(layouts) + 1
    fig, axes = plt.subplots(1, ncol, figsize=(4.2 * ncol, 4.4))
    if ncol == 1:
        axes = [axes]
    for i, lay in enumerate(layouts):
        draw_layout(axes[i], lay["start"], lay["obstacles"], args.obstacle_size,
                    lay["routes"], lay["succ"], zrgb, args.max_steps)
        axes[i].set_title(f"Layout {i+1}  ({sum(lay['succ'])}/{args.k_skills} skills solve)",
                          fontsize=11)
    axes[0].legend(loc="upper left", fontsize=8, framealpha=0.9)

    # ---- embedding panel ----
    embs, cols, marks = [], [], []
    markers = ["o", "s", "^", "D", "v", "P"]
    for i, lay in enumerate(layouts):
        for k in range(args.k_skills):
            r = lay["routes"][k]
            if r.shape[0] < 2:
                continue
            embs.append(embed_traj([np.concatenate([p, [0, 0]]) for p in r], args.max_steps))
            cols.append(zrgb[k]); marks.append(i)
    embs = np.stack(embs)
    cols = np.asarray(cols)
    marks = np.asarray(marks)

    try:
        from sklearn.manifold import TSNE
        perp = max(5, min(30, len(embs) // 4))
        xy = TSNE(n_components=2, perplexity=perp, init="pca",
                  random_state=0).fit_transform(embs)
        emb_name = f"t-SNE (perplexity={perp})"
    except Exception as e:
        print(f"[warn] t-SNE unavailable ({e}); using PCA")
        Xc = embs - embs.mean(0)
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        xy = Xc @ Vt[:2].T
        emb_name = "PCA"

    axE = axes[-1]
    for i in range(len(layouts)):
        m = marks == i
        axE.scatter(xy[m, 0], xy[m, 1], c=cols[m], marker=markers[i % len(markers)],
                    s=55, edgecolor="k", lw=0.4, label=f"Layout {i+1}")
    axE.set_title(f"Route embedding ({emb_name})\ncolor = skill z (RGB)", fontsize=11)
    axE.set_xticks([]); axE.set_yticks([])
    axE.legend(loc="best", fontsize=8)

    fig.suptitle("DIVO stage-1 skill library: one latent z -> one solving route",
                 fontsize=13, y=1.02)
    fig.tight_layout()

    tag = pathlib.Path(args.ckpt).parts[-3] if len(pathlib.Path(args.ckpt).parts) >= 3 else "run"
    out = args.out or str(REPO_ROOT / "analysis" / "skill_coverage" / f"skill_library_{tag}.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
