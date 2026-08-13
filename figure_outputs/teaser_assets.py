"""Render REAL teaser assets from a trained DIVO checkpoint (headless).

Outputs go to figure_outputs/teaser_assets/ as transparent, axis-free vector
(+ png) overlays meant to be COMPOSITED BY HAND in Illustrator/Inkscape:

  asset_A_deployed.(svg|png)     - obstacles + goal + single deployed rollout w_0
  asset_B_skillfan.(svg|png)     - obstacles + goal + K probe-skill rollouts
                                   (success solid / failure dashed)
  asset_C_learnability.(svg|png) - clean lv = p(1-p) curve, tau-band, real point
  asset_mujoco_scene.png         - real MuJoCo photographic render (if GL works)
  scene_meta.json                - the chosen layout + per-skill success + realized

The K skill trajectories come from `rollout_scene_with_z_bank` (K different
latents injected into the deterministic decoder) = the paper's real probe, and
run WITHOUT the MuJoCo renderer (physics stepping is headless-safe). The layout
is auto-selected near the learnable boundary so the teaser is both real and
on-message (some skills solve, some fail).
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

import numpy as np
import torch

os.environ.setdefault("MUJOCO_GL", "egl")

REPO = pathlib.Path("/home/hnu-w/DIVO")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from omegaconf import OmegaConf

try:
    OmegaConf.register_new_resolver("now", lambda pat: "now")
except Exception:
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle

from DIVO.env import get_env_class
from DIVO.policy import get_policy
from DIVO.curriculum.skill_signal import (
    SceneSamplingStats,
    build_z_bank,
    rollout_scene_with_z_bank,
    sample_generator_scene,
)
from DIVO.curriculum.candidate_verifier import SandboxPushTExecutor

# ---- config -----------------------------------------------------------------
CKPT_DIR = REPO / "data/outputs/2026.06.22_g0_skill_signal_evolve_s0_clean"
CKPT = CKPT_DIR / "checkpoints/best430k.pt"
CFG = CKPT_DIR / ".hydra/config.yaml"
GEN_CANDIDATES = [
    REPO / "data/outputs/2026.04.29/g0_seed2/current_generator.py",
    REPO / "data/outputs/2026.04.29/g0_seed2/initial_generator.py",
    REPO / "data/outputs/2026.04.27/20.24.50_td3_pusht_llm_curriculum/initial_generator.py",
]
OUT = REPO / "figure_outputs/teaser_assets"

K = 6              # probe skills in the fan
NUM_OBST = 2
MAX_STEPS = 10
N_LAYOUT_TRIES = 40
SEED = 7

# T-block geometry (mirror skill_viz / motion_decoder feature points)
_TBLOCK_REL = np.array([[0.2, 0.06], [-0.2, 0.06], [0.06, -0.34], [-0.06, -0.34]], float)
_ORDER = [0, 1, 3, 2]
GOAL = (0.0, 0.0)
LIM = 0.30
SKILL_CMAP = plt.get_cmap("tab10")
# DIVO Fig.1 visual grammar: red initial T, green goal T, orange obstacles,
# black end-effector dot, dashed trajectories.
C_DEPLOY = "#c0392b"
C_INIT = "#d62728"   # initial T-block (red)
C_GOAL = "#2ca02c"   # goal T-block (green)
C_OK = "#2ca02c"
C_BAND = "#f4b400"
C_OBST = "#ff7f0e"   # obstacles (orange, DIVO style)
C_EE = "#111111"     # end-effector dot


def tblock_poly(state, scale=1.0):
    a = np.asarray(state, float).reshape(-1)
    cx, cy = a[0], a[1]
    cos = a[2] if a.size > 2 else 1.0
    sin = a[3] if a.size > 3 else 0.0
    n = np.hypot(cos, sin) or 1.0
    cos, sin = cos / n, sin / n
    R = np.array([[cos, -sin], [sin, cos]])
    return ((R @ (_TBLOCK_REL * scale).T).T + [cx, cy])[_ORDER]


def new_axes():
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_xlim(-LIM, LIM)
    ax.set_ylim(-LIM, LIM)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    return fig, ax


def draw_layout(ax, obstacles, start, ee=True):
    # goal T-block (filled green) -- DIVO grammar, no star
    ax.add_patch(Polygon(tblock_poly((GOAL[0], GOAL[1], 1.0, 0.0)), closed=True,
                         facecolor=C_GOAL, edgecolor=C_GOAL, lw=1.0, alpha=0.45, zorder=3))
    # obstacles as orange circles (DIVO grammar)
    from matplotlib.patches import Circle
    for o in obstacles:
        ax.add_patch(Circle((o["x"], o["y"]), 0.022, facecolor=C_OBST,
                            edgecolor="#c0620a", lw=0.8, zorder=4))
    # initial T-block (filled red)
    ax.add_patch(Polygon(tblock_poly(start), closed=True, facecolor=C_INIT,
                         edgecolor=C_INIT, lw=1.0, alpha=0.85, zorder=6))
    if ee:
        ax.scatter([start[0]], [start[1]], color=C_EE, s=42, zorder=7,
                   edgecolors="white", linewidths=0.6)


def save(fig, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.svg", transparent=True, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{stem}.png", dpi=300, transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def load_generator():
    for path in GEN_CANDIDATES:
        if not path.exists():
            continue
        ex = SandboxPushTExecutor(obstacle_size=0.01)
        if ex.load(path.read_text(encoding="utf-8")):
            print(f"[gen] loaded {path}")
            return ex
    print("[gen] no generator file loaded; will hand-place obstacles")
    return None


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = OmegaConf.load(str(CFG))
    env = get_env_class(**OmegaConf.to_container(cfg.env, resolve=True))
    policy = get_policy(env, **OmegaConf.to_container(cfg.policy, resolve=True)).to(device)
    policy.load_state_dict(torch.load(str(CKPT), map_location=device), strict=False)
    policy.eval()
    latent_dim = int(cfg.latent_dim)
    z_bank = build_z_bank(K=K, latent_dim=latent_dim, seed=SEED, device=device)

    gen = load_generator()
    stats = SceneSamplingStats()

    # Search layouts; keep the one with realized closest to the boundary (0.5)
    # that is feasible (>=1 skill solves) but not trivial (not all solve).
    best = None
    for _ in range(N_LAYOUT_TRIES):
        if gen is not None:
            scene = sample_generator_scene(env, gen, NUM_OBST, timeout_sec=5, stats=stats)
            if scene is None:
                continue
            start, obstacles = scene["start"], scene["obstacles"]
        else:
            start = list(map(float, np.asarray(env.sample_valid_tblock_pose())))
            obstacles = [{"x": -0.02, "y": -0.03, "purpose": ""},
                         {"x": 0.08, "y": 0.06, "purpose": ""}]
        res = rollout_scene_with_z_bank(env, policy, start, obstacles, z_bank, device, MAX_STEPS)
        succ = [int(r["success"]) for r in res["routes"]]
        realized = float(np.mean(succ))
        feasible = max(succ)
        if feasible == 0 or realized >= 1.0:
            continue
        score = -abs(realized - 0.5)  # closer to 0.5 is better
        cand = dict(start=start, obstacles=obstacles, res=res, realized=realized,
                    succ=succ, score=score)
        if best is None or cand["score"] > best["score"]:
            best = cand
        if abs(realized - 0.5) < 1e-6:
            break

    if best is None:
        raise RuntimeError("no feasible boundary layout found; loosen search or hand-place")

    start, obstacles, res = best["start"], best["obstacles"], best["res"]
    realized, succ = best["realized"], best["succ"]
    print(f"[scene] realized={realized:.3f}  per-skill success={succ}  start={np.round(start,3)}")

    # ---- asset A: deployed rollout (single skill w_0 = encoder z) ----
    fig, ax = new_axes()
    draw_layout(ax, obstacles, start)
    dtr = np.asarray(res["deployed_route"]["states"], float)
    if dtr.ndim == 2 and len(dtr):
        ax.plot(dtr[:, 0], dtr[:, 1], "-", color=C_DEPLOY, lw=3.2, zorder=7,
                solid_capstyle="round")
        ax.add_patch(Polygon(tblock_poly(dtr[-1]), closed=True, fill=False,
                             edgecolor=C_DEPLOY, lw=1.6, alpha=0.6, zorder=7))
    save(fig, "asset_A_deployed")

    # ---- asset B: K-skill fan (success solid / failure dashed) ----
    fig, ax = new_axes()
    draw_layout(ax, obstacles, start)
    for k, r in enumerate(res["routes"]):
        arr = np.asarray(r["states"], float)
        if arr.ndim != 2 or not len(arr):
            continue
        color = SKILL_CMAP(k % 10)
        ax.plot(arr[:, 0], arr[:, 1], "-" if r["success"] else "--", color=color,
                lw=2.6, alpha=0.95, zorder=6, solid_capstyle="round")
        ax.add_patch(Polygon(tblock_poly(arr[-1]), closed=True, fill=False,
                             edgecolor=color, lw=1.2, alpha=0.7, zorder=6))
    save(fig, "asset_B_skillfan")

    # ---- asset C: learnability curve ----
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    fig.patch.set_alpha(0.0)
    p = np.linspace(0, 1, 300)
    lv = p * (1 - p)
    ax.plot(p, lv, color="#222", lw=3.0)
    tau = 0.5 / K if K else 0.125
    band = (p > tau) & (p < 1 - tau)
    ax.fill_between(p, 0, lv, where=band, color=C_BAND, alpha=0.35)
    ax.scatter([realized], [realized * (1 - realized)], color=C_DEPLOY, s=150,
               zorder=6, edgecolors="white", linewidths=1.0)
    for xv in (tau, 1 - tau):
        ax.axvline(xv, color=C_BAND, ls=":", lw=1.4)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.27)
    ax.set_xlabel("realized  $p$", fontsize=13)
    ax.set_ylabel("$lv = p(1-p)$", fontsize=13)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=11)
    save(fig, "asset_C_learnability")

    # ---- asset: real MuJoCo photographic render (best-effort) ----
    mj_ok = False
    try:
        env.set_obstacle_config(obstacles)
        if hasattr(env, "_record_frame"):
            env._record_frame = True
            env.frames = []
        obs = env.reset(tblock_pos=np.asarray(start, float), force_tblock_pos=True)
        frames = list(getattr(env, "frames", []) or [])
        if not frames and hasattr(env, "render"):
            frames = [env.render()]
        if frames:
            import matplotlib.image as mpimg
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.axis("off")
            ax.imshow(np.asarray(frames[0]))
            OUT.mkdir(parents=True, exist_ok=True)
            fig.savefig(OUT / "asset_mujoco_scene.png", dpi=300, bbox_inches="tight", pad_inches=0)
            plt.close(fig)
            mj_ok = True
    except Exception as e:  # noqa: BLE001
        print(f"[mujoco] render skipped: {type(e).__name__}: {e}")

    meta = dict(checkpoint=str(CKPT), K=K, realized=realized, per_skill_success=succ,
                start=list(map(float, start)), obstacles=obstacles,
                deployed_success=bool(res["deployed_route"]["success"]),
                tau=tau, mujoco_render=mj_ok)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "scene_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[done] assets in {OUT}  (mujoco_render={mj_ok})")


if __name__ == "__main__":
    main()
