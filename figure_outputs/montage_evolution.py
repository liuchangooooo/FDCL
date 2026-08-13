"""V3: Distribution-evolution montage (real accepted generators).

Rows = evolution rounds (real accepted generator programs from the clean run).
Cols = layouts sampled from that round's generator, drawn in the DIVO visual
grammar (red initial T, green goal T, orange obstacles) from the TRUE sampled
start pose + obstacle positions. Each row is annotated with the round's REAL
skill-signal stats from acgs_evolve_records.jsonl (single-policy deployed
success vs skill-library realized), so the montage shows the training
distribution refocusing toward the learnable band while single-policy success
saturates.

No fabrication: generators are the actual accepted programs; poses/obstacles are
really sampled from them; stats are logged.
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
from matplotlib.patches import Circle, Polygon

RUN = REPO / "data/outputs/2026.06.22_g0_skill_signal_evolve_s0_clean"
CFG = RUN / ".hydra/config.yaml"
OUT = REPO / "figure_outputs/montage_evolution"
ROUNDS = [0, 9, 19]          # a few rounds spanning the run
N_PER = 4                    # example environments per row
SEED = 7
LIM = 0.24

C_INIT, C_GOAL, C_OBST, C_EE = "#d62728", "#2ca02c", "#ff7f0e", "#111111"
_TBLOCK_REL = np.array([[0.2, 0.06], [-0.2, 0.06], [0.06, -0.34], [-0.06, -0.34]], float)
_ORDER = [0, 1, 3, 2]


def tpoly(state):
    a = np.asarray(state, float).reshape(-1)
    cx, cy = a[0], a[1]
    th = a[2] if a.size > 2 else 0.0
    cos, sin = np.cos(th), np.sin(th)
    R = np.array([[cos, -sin], [sin, cos]])
    return ((R @ _TBLOCK_REL.T).T + [cx, cy])[_ORDER]


def draw_scene(ax, obstacles, start):
    ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor("0.8"); sp.set_linewidth(0.6)
    ax.add_patch(Polygon(tpoly((0, 0, 0)), closed=True, facecolor=C_GOAL,
                         edgecolor=C_GOAL, lw=1.0, alpha=0.40, zorder=2))
    ax.add_patch(Polygon(tpoly(start), closed=True, facecolor=C_INIT,
                         edgecolor=C_INIT, lw=1.0, alpha=0.85, zorder=5))
    ax.scatter([start[0]], [start[1]], color=C_EE, s=16, zorder=6,
               edgecolors="white", linewidths=0.5)
    for o in obstacles:
        ax.add_patch(Circle((o["x"], o["y"]), 0.025, facecolor=C_OBST,
                            edgecolor="#c0620a", lw=0.8, zorder=4))


def load_stats():
    m = {}
    with open(RUN / "acgs_evolve_records.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if not r.get("accepted"):
                continue
            reason = r.get("reason", "")
            k = None
            if "evolve_index=" in reason:
                try:
                    k = int(reason.split("evolve_index=")[1].split("/")[0])
                except Exception:
                    k = None
            p = r.get("skill_signal_profile") or {}
            if k is not None:
                m[k] = dict(deployed=p.get("mean_deployed"),
                            realized=p.get("mean_realized"),
                            lv=p.get("mean_lv"),
                            infeas=p.get("frac_infeasible"))
    return m


def sample_row(env, gen_path, n, seed):
    ex = SandboxPushTExecutor(obstacle_size=0.05)
    ex.load(gen_path.read_text(encoding="utf-8"))
    stats = SceneSamplingStats()
    rng = np.random.RandomState(seed)
    scenes = []
    tries = 0
    while len(scenes) < n and tries < n * 25:
        tries += 1
        np.random.seed(rng.randint(1 << 30))
        sc = sample_generator_scene(env, ex, 2, timeout_sec=5, stats=stats)
        if sc is not None:
            scenes.append(sc)
    return scenes


def main():
    torch.manual_seed(SEED)
    cfg = OmegaConf.load(str(CFG))
    env = get_env_class(**OmegaConf.to_container(cfg.env, resolve=True))
    stat_map = load_stats()
    OUT.mkdir(parents=True, exist_ok=True)

    rows = []
    for r in ROUNDS:
        gen = RUN / f"generators/generator_{r:03d}.py"
        sc = sample_row(env, gen, N_PER, SEED + r)
        rows.append(sc)
        print(f"round {r}: {len(sc)} scenes")

    nrows, ncols = len(ROUNDS), N_PER
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 1.6 + 1.5, nrows * 1.6))
    axes = np.atleast_2d(axes)
    for i, r in enumerate(ROUNDS):
        for j in range(ncols):
            ax = axes[i, j]
            if j < len(rows[i]):
                draw_scene(ax, rows[i][j]["obstacles"], rows[i][j]["start"])
            else:
                ax.set_xticks([]); ax.set_yticks([])
        st = stat_map.get(r)
        if r == 0:
            lab = "Round 0\n(initial)"
        elif st and st.get("realized") is not None:
            lab = f"Round {r}\nrealized={st['realized']:.2f}"
        else:
            lab = f"Round {r}"
        axes[i, 0].set_ylabel(lab, rotation=0, ha="right", va="center",
                              fontsize=9.5, labelpad=40)

    fig.suptitle("Example training environments sampled from the accepted generator "
                 "at different evolution rounds\n(red: initial T-object,  green: goal region,  "
                 "orange: obstacles)",
                 fontsize=10)
    # legend
    handles = [plt.Line2D([], [], marker='s', ls='', mfc=C_INIT, mec=C_INIT, ms=8, label='initial T'),
               plt.Line2D([], [], marker='s', ls='', mfc=C_GOAL, mec=C_GOAL, ms=8, alpha=0.5, label='goal T'),
               plt.Line2D([], [], marker='o', ls='', mfc=C_OBST, mec="#c0620a", ms=8, label='obstacle')]
    fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.patch.set_facecolor("white")
    fig.tight_layout(rect=(0.04, 0.03, 1.0, 0.94))
    fig.savefig(OUT / "montage_evolution.png", dpi=150, facecolor="white")
    fig.savefig(OUT / "montage_evolution.pdf", facecolor="white")
    plt.close(fig)
    print(f"[done] saved montage to {OUT}")

    traj = {r: stat_map.get(r) for r in ROUNDS}
    (OUT / "montage_stats.json").write_text(json.dumps(traj, indent=2))


if __name__ == "__main__":
    main()
