"""Environment-distribution change: a few rounds side by side (clean heatmaps).

Each panel = top-down obstacle-placement density of the accepted generator at
that round (real samples). Reading left to right shows the training distribution
shifting inward toward the object-goal corridor. Minimal annotation.

No fabrication: generators are real accepted programs; obstacle points really
sampled; per-round realized read from acgs_evolve_records.jsonl.
"""
from __future__ import annotations
import os, json, pathlib, sys
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, torch
from omegaconf import OmegaConf
try:
    OmegaConf.register_new_resolver("now", lambda p: "now")
except Exception:
    pass

REPO = pathlib.Path("/home/hnu-w/DIVO")
sys.path.insert(0, str(REPO))
from DIVO.env import get_env_class
from DIVO.curriculum.skill_signal import SceneSamplingStats, sample_generator_scene
from DIVO.curriculum.candidate_verifier import SandboxPushTExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
try:
    from scipy.ndimage import gaussian_filter
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

RUN = REPO / "data/outputs/2026.06.22_g0_skill_signal_evolve_s0_clean"
CFG = RUN / ".hydra/config.yaml"
OUT = REPO / "figure_outputs/dist_evolution"
ROUNDS = [0, 9, 19]
N_SCENES = 320
BINS = 46
LIM = 0.21
SEED = 5

C_GOAL = "#1a9850"
_TBLOCK_REL = 0.5 * np.array([[0.2, 0.06], [-0.2, 0.06], [0.06, -0.34], [-0.06, -0.34]], float)
_ORDER = [0, 1, 3, 2]


def goal_poly():
    return _TBLOCK_REL[_ORDER]


def load_realized():
    m = {}
    with open(RUN / "acgs_evolve_records.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if not r.get("accepted"):
                continue
            reason = r.get("reason", "")
            k = int(reason.split("evolve_index=")[1].split("/")[0]) if "evolve_index=" in reason else None
            p = r.get("skill_signal_profile") or {}
            if k is not None:
                m[k] = p.get("mean_realized")
    return m


def sample_points(env, idx, n, seed):
    ex = SandboxPushTExecutor(obstacle_size=0.05)
    ex.load((RUN / f"generators/generator_{idx:03d}.py").read_text(encoding="utf-8"))
    st = SceneSamplingStats()
    rng = np.random.RandomState(seed)
    pts, tries = [], 0
    while len(pts) < n and tries < n * 20:
        tries += 1
        np.random.seed(rng.randint(1 << 30))
        sc = sample_generator_scene(env, ex, 2, timeout_sec=5, stats=st)
        if sc:
            for o in sc["obstacles"]:
                pts.append((o["x"], o["y"]))
    return np.array(pts)


def density(pts):
    H, _, _ = np.histogram2d(pts[:, 0], pts[:, 1], bins=BINS,
                             range=[[-LIM, LIM], [-LIM, LIM]])
    H = H.T
    if HAVE_SCIPY:
        H = gaussian_filter(H, sigma=1.3)
    return H / (H.max() + 1e-9)


def main():
    torch.manual_seed(SEED)
    cfg = OmegaConf.load(str(CFG))
    env = get_env_class(**OmegaConf.to_container(cfg.env, resolve=True))
    realized = load_realized()
    OUT.mkdir(parents=True, exist_ok=True)

    dens = []
    for r in ROUNDS:
        p = sample_points(env, r, N_SCENES, SEED + r)
        dens.append(density(p))
        print(f"round {r}: {len(p)} points")

    n = len(ROUNDS)
    fig, axes = plt.subplots(1, n, figsize=(n * 2.7 + 0.5, 3.0))
    axes = np.atleast_1d(axes)
    im = None
    xx = np.linspace(-LIM, LIM, BINS)
    for i, r in enumerate(ROUNDS):
        ax = axes[i]
        im = ax.imshow(dens[i], origin="lower", extent=[-LIM, LIM, -LIM, LIM],
                       cmap="magma_r", vmin=0, vmax=1, aspect="equal")
        ax.contour(xx, xx, dens[i], levels=[0.25, 0.5], colors="0.6",
                   linewidths=0.5, alpha=0.6)
        # small goal-T marker at center
        ax.add_patch(Polygon(goal_poly(), closed=True, facecolor=C_GOAL,
                             edgecolor="white", lw=0.8, alpha=0.9, zorder=5))
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor("0.7")
        rl = realized.get(r)
        tag = "initial $g_0$" if r == 0 else f"realized={rl:.2f}"
        ax.set_title(f"Round {r}", fontsize=11)
        ax.set_xlabel(tag, fontsize=9, color="0.3")

    for i in range(n - 1):
        x = (axes[i].get_position().x1 + axes[i + 1].get_position().x0) / 2
        fig.text(x, 0.5, "\u2192", ha="center", va="center", fontsize=22, color="0.4")

    cbar = fig.colorbar(im, ax=axes.tolist(), fraction=0.022, pad=0.012)
    cbar.set_label("obstacle density", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_ticks([0, 0.5, 1.0])

    fig.legend(handles=[plt.Line2D([], [], marker='v', ls='', mfc=C_GOAL,
               mec='white', ms=9, label='goal-T')],
               loc='upper right', fontsize=8, frameon=False,
               bbox_to_anchor=(0.99, 1.06))
    fig.suptitle("Training-obstacle distribution shifts inward toward the "
                 "object\u2013goal region over evolution rounds", fontsize=11, y=1.04)
    fig.patch.set_facecolor("white")
    fig.savefig(OUT / "dist_evolution_strip.png", dpi=180, facecolor="white",
                bbox_inches="tight")
    fig.savefig(OUT / "dist_evolution_strip.pdf", facecolor="white",
                bbox_inches="tight")
    plt.close(fig)
    print(f"[done] saved to {OUT}  scipy={HAVE_SCIPY}")


if __name__ == "__main__":
    main()
