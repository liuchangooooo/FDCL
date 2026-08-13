"""Figure 1 (problem): single-policy feedback cannot see the skill-library frontier.

2x2 grid of REAL Push-T scenes from a trained checkpoint:
  columns = two layouts that the deployed policy BOTH solve (single-policy: same).
  row 1 (single-policy view): deployed w_0 rollout + big check on both -> "same".
  row 2 (skill-library view): K skills injected -> left mostly solve (mastered,
         realized~1.0), right only some solve (boundary, realized~0.5) -> "different".

Outputs (figure_outputs/fig1_problem/):
  panel_*.svg/png  : transparent, axis-free panels for hand-assembly
  fig1_problem.png/pdf : a labeled draft composite (grid of real panels)
  fig1_meta.json   : the two layouts + per-skill success + realized

DIVO visual grammar: red initial T, green goal T, orange obstacles, black
end-effector dot, dashed=fail / solid=solve trajectories.
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
from matplotlib import font_manager
from matplotlib.patches import Circle, Polygon

# Chinese font (Noto Sans CJK SC on this system)
for _fp in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"):
    if pathlib.Path(_fp).exists():
        try:
            font_manager.fontManager.addfont(_fp)
        except Exception:
            pass
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from DIVO.env import get_env_class
from DIVO.policy import get_policy
from DIVO.curriculum.skill_signal import (
    SceneSamplingStats, build_z_bank, rollout_scene_with_z_bank, sample_generator_scene,
)
from DIVO.curriculum.candidate_verifier import SandboxPushTExecutor

CKPT_DIR = REPO / "data/outputs/2026.06.22_g0_skill_signal_evolve_s0_clean"
CKPT = CKPT_DIR / "checkpoints/best430k.pt"
CFG = CKPT_DIR / ".hydra/config.yaml"
GEN_CANDIDATES = [
    REPO / "data/outputs/2026.04.29/g0_seed2/current_generator.py",
    REPO / "data/outputs/2026.04.29/g0_seed2/initial_generator.py",
]
OUT = REPO / "figure_outputs/fig1_problem"

K = 6
NUM_OBST = 2
MAX_STEPS = 10
N_TRIES = 160
SEED = 3

_TBLOCK_REL = np.array([[0.2, 0.06], [-0.2, 0.06], [0.06, -0.34], [-0.06, -0.34]], float)
_ORDER = [0, 1, 3, 2]
GOAL = (0.0, 0.0)
LIM = 0.30
CMAP = plt.get_cmap("tab10")
C_INIT, C_GOAL, C_OBST, C_EE = "#d62728", "#2ca02c", "#ff7f0e", "#111111"
C_DEPLOY, C_OK = "#c0392b", "#2ca02c"


def tpoly(state):
    a = np.asarray(state, float).reshape(-1)
    cx, cy = a[0], a[1]
    cos = a[2] if a.size > 2 else 1.0
    sin = a[3] if a.size > 3 else 0.0
    n = np.hypot(cos, sin) or 1.0
    R = np.array([[cos / n, -sin / n], [sin / n, cos / n]])
    return ((R @ _TBLOCK_REL.T).T + [cx, cy])[_ORDER]


def new_ax():
    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM)
    ax.set_aspect("equal"); ax.axis("off")
    fig.patch.set_alpha(0.0); ax.patch.set_alpha(0.0)
    return fig, ax


def draw_layout(ax, obstacles, start):
    ax.add_patch(Polygon(tpoly((0, 0, 1, 0)), closed=True, facecolor=C_GOAL,
                         edgecolor=C_GOAL, lw=1.0, alpha=0.45, zorder=3))
    for o in obstacles:
        ax.add_patch(Circle((o["x"], o["y"]), 0.022, facecolor=C_OBST,
                            edgecolor="#c0620a", lw=0.8, zorder=4))
    ax.add_patch(Polygon(tpoly(start), closed=True, facecolor=C_INIT,
                         edgecolor=C_INIT, lw=1.0, alpha=0.85, zorder=6))
    ax.scatter([start[0]], [start[1]], color=C_EE, s=40, zorder=7,
               edgecolors="white", linewidths=0.6)


def draw_single(ax, scene):
    draw_layout(ax, scene["obstacles"], scene["start"])
    tr = np.asarray(scene["res"]["deployed_route"]["states"], float)
    if tr.ndim == 2 and len(tr):
        ax.plot(tr[:, 0], tr[:, 1], "-", color=C_DEPLOY, lw=3.0, zorder=7,
                solid_capstyle="round")


def draw_library(ax, scene):
    draw_layout(ax, scene["obstacles"], scene["start"])
    for k, r in enumerate(scene["res"]["routes"]):
        arr = np.asarray(r["states"], float)
        if arr.ndim != 2 or not len(arr):
            continue
        ax.plot(arr[:, 0], arr[:, 1], "-" if r["success"] else "--",
                color=CMAP(k % 10), lw=2.3, alpha=0.95, zorder=6,
                solid_capstyle="round")


def save_panel(draw_fn, scene, stem):
    fig, ax = new_ax()
    draw_fn(ax, scene)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.svg", transparent=True, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{stem}.png", dpi=300, transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def load_gen():
    for p in GEN_CANDIDATES:
        if p.exists():
            ex = SandboxPushTExecutor(obstacle_size=0.01)
            if ex.load(p.read_text(encoding="utf-8")):
                print(f"[gen] {p}")
                return ex
    return None


def main():
    np.random.seed(SEED); torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = OmegaConf.load(str(CFG))
    env = get_env_class(**OmegaConf.to_container(cfg.env, resolve=True))
    policy = get_policy(env, **OmegaConf.to_container(cfg.policy, resolve=True)).to(device)
    policy.load_state_dict(torch.load(str(CKPT), map_location=device), strict=False)
    policy.eval()
    z_bank = build_z_bank(K=K, latent_dim=int(cfg.latent_dim), seed=SEED, device=device)
    gen = load_gen()
    stats = SceneSamplingStats()

    mastered = None   # deployed success, realized high
    boundary = None   # deployed success, realized ~0.5
    for _ in range(N_TRIES):
        if gen is not None:
            sc = sample_generator_scene(env, gen, NUM_OBST, timeout_sec=5, stats=stats)
            if sc is None:
                continue
            start, obstacles = sc["start"], sc["obstacles"]
        else:
            start = list(map(float, np.asarray(env.sample_valid_tblock_pose())))
            obstacles = [{"x": -0.02, "y": -0.03, "purpose": ""}, {"x": 0.08, "y": 0.06, "purpose": ""}]
        res = rollout_scene_with_z_bank(env, policy, start, obstacles, z_bank, device, MAX_STEPS)
        succ = [int(r["success"]) for r in res["routes"]]
        realized = float(np.mean(succ))
        dep = bool(res["deployed_route"]["success"])
        if not dep:
            continue  # both columns must be deployed-success
        cand = dict(start=start, obstacles=obstacles, res=res, realized=realized, succ=succ)
        if realized >= 0.83:
            if mastered is None or realized > mastered["realized"]:
                mastered = cand
        if 0.34 <= realized <= 0.6:
            if boundary is None or abs(realized - 0.5) < abs(boundary["realized"] - 0.5):
                boundary = cand
        if mastered is not None and boundary is not None and mastered["realized"] >= 0.99:
            break

    if mastered is None or boundary is None:
        raise RuntimeError(f"search failed: mastered={mastered is not None} boundary={boundary is not None}")

    print(f"[mastered] realized={mastered['realized']:.3f} succ={mastered['succ']}")
    print(f"[boundary] realized={boundary['realized']:.3f} succ={boundary['succ']}")

    # column 1 = mastered layout, column 2 = boundary layout
    save_panel(draw_single, mastered, "panel_col1_single")
    save_panel(draw_single, boundary, "panel_col2_single")
    save_panel(draw_library, mastered, "panel_col1_library")
    save_panel(draw_library, boundary, "panel_col2_library")

    # ---- draft composite (grid of real panels + labels) ----
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 7.6))
    draw_single(axes[0, 0], mastered); draw_single(axes[0, 1], boundary)
    draw_library(axes[1, 0], mastered); draw_library(axes[1, 1], boundary)
    for ax in axes.ravel():
        ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor("#cccccc")
    # big checks on single-policy row
    for ax in (axes[0, 0], axes[0, 1]):
        ax.text(0.20, 0.22, "\u2713", color=C_OK, fontsize=30, fontweight="bold",
                ha="center", va="center")
    # realized tags on library row
    axes[1, 0].set_title(f"能解比例 $p={mastered['realized']:.2f}$  \u2192 已掌握",
                         fontsize=12, color="#333")
    axes[1, 1].set_title(f"能解比例 $p={boundary['realized']:.2f}$  \u2192 可学边界",
                         fontsize=12, color=C_DEPLOY)
    # column headers
    axes[0, 0].annotate("布局 1", xy=(0.5, 1.14), xycoords="axes fraction",
                        ha="center", fontsize=13, fontweight="bold")
    axes[0, 1].annotate("布局 2", xy=(0.5, 1.14), xycoords="axes fraction",
                        ha="center", fontsize=13, fontweight="bold")
    # row labels
    axes[0, 0].annotate("单策略反馈", xy=(-0.15, 0.5), xycoords="axes fraction",
                        ha="center", va="center", fontsize=13, fontweight="bold", rotation=90)
    axes[1, 0].annotate("技能库\n(K 个技能)", xy=(-0.15, 0.5), xycoords="axes fraction",
                        ha="center", va="center", fontsize=13, fontweight="bold", rotation=90)
    fig.suptitle("单策略反馈认为两个布局一样（都 \u2713 成功）；技能库却看出它们截然不同",
                 fontsize=13, y=0.985)
    fig.text(0.5, 0.015,
             "部署成功率相同，训练价值相反：布局 1 已被大多数技能掌握，布局 2 仍处于可学边界。",
             ha="center", fontsize=11, color="#333")
    fig.subplots_adjust(left=0.10, right=0.98, top=0.90, bottom=0.06, wspace=0.06, hspace=0.16)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig1_problem.png", dpi=200, facecolor="white")
    fig.savefig(OUT / "fig1_problem.pdf", facecolor="white")
    plt.close(fig)

    meta = dict(checkpoint=str(CKPT), K=K,
                mastered=dict(realized=mastered["realized"], succ=mastered["succ"],
                              start=list(map(float, mastered["start"])), obstacles=mastered["obstacles"]),
                boundary=dict(realized=boundary["realized"], succ=boundary["succ"],
                              start=list(map(float, boundary["start"])), obstacles=boundary["obstacles"]))
    (OUT / "fig1_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[done] {OUT}")


if __name__ == "__main__":
    main()
