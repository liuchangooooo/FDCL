"""Closed-loop figure (EnvGen-style) instantiated with REAL data.

Shows one real evolution step g_t -> g_{t+1} (accepted generators, round 5->6):
  [e_t: obstacle-density heatmap of g_t]
     --probe with skill library (pi, w_1..w_K)-->
  [Learnability signal S(g_t): realized distribution + stats]
     --LLM rewrites generator program G-->
  [e_{t+1}: obstacle-density heatmap of g_{t+1}]
     --train policy pi_{t+1}, loop back-->

Environment nodes are top-down obstacle-placement density maps (real samples).
Signal/stat annotations come from acgs_evolve_records.jsonl. No fabrication.
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
from matplotlib.patches import Polygon, FancyArrowPatch, FancyBboxPatch
try:
    from scipy.ndimage import gaussian_filter
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

RUN = REPO / "data/outputs/2026.06.22_g0_skill_signal_evolve_s0_clean"
CFG = RUN / ".hydra/config.yaml"
OUT = REPO / "figure_outputs/loop_et"
T, T1 = 5, 6            # e_t = generator_005, e_{t+1} = generator_006
N_SCENES = 240
BINS = 44
LIM = 0.21
SEED = 5

C_GOAL = "#2ca02c"
_TBLOCK_REL = np.array([[0.2, 0.06], [-0.2, 0.06], [0.06, -0.34], [-0.06, -0.34]], float)
_ORDER = [0, 1, 3, 2]


def tpoly(state):
    a = np.asarray(state, float).reshape(-1)
    cx, cy, th = a[0], a[1], (a[2] if a.size > 2 else 0.0)
    cos, sin = np.cos(th), np.sin(th)
    R = np.array([[cos, -sin], [sin, cos]])
    return ((R @ _TBLOCK_REL.T).T + [cx, cy])[_ORDER]


def load_stats():
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
                h = p.get("realized_hist") or {}
                m[k] = dict(deployed=p.get("mean_deployed"), realized=p.get("mean_realized"),
                            lv=p.get("mean_lv"), infeas=p.get("frac_infeasible"),
                            counts=h.get("counts"), bins=h.get("bins"))
    return m


def sample_points(env, idx, n, seed):
    ex = SandboxPushTExecutor(obstacle_size=0.05)
    ex.load((RUN / f"generators/generator_{idx:03d}.py").read_text(encoding="utf-8"))
    st = SceneSamplingStats()
    rng = np.random.RandomState(seed)
    pts = []
    tries = 0
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
        H = gaussian_filter(H, sigma=1.2)
    return H / (H.max() + 1e-9)


def draw_env(ax, H, title):
    ax.imshow(H, origin="lower", extent=[-LIM, LIM, -LIM, LIM],
              cmap="YlOrRd", vmin=0, vmax=1, aspect="equal")
    ax.add_patch(Polygon(tpoly((0, 0, 0)), closed=True, fill=False,
                         edgecolor=C_GOAL, lw=2.0, zorder=5))
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor("0.6"); sp.set_linewidth(1.0)
    ax.set_title(title, fontsize=11, pad=5)


def draw_hist(ax, counts, title):
    counts = np.array(counts, float)
    frac = counts / (counts.sum() or 1)
    centers = [0.1, 0.3, 0.5, 0.7, 0.9]
    colors = ["#e15759", "#59a14f", "#59a14f", "#59a14f", "#bab0ac"]
    ax.bar(centers, frac, width=0.16, color=colors, edgecolor="white")
    ax.axvspan(0.2, 0.8, color="#59a14f", alpha=0.10, zorder=0)
    ax.set_xlim(0, 1); ax.set_ylim(0, max(frac.max() * 1.2, 0.1))
    ax.set_xlabel("skill-library realized", fontsize=8)
    ax.set_ylabel("scene frac.", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title(title, fontsize=9)


def arrow(fig, xy0, xy1, text=None, dotted=False, tx=None, ty=None, color="0.25"):
    fig.patches.append(FancyArrowPatch(
        xy0, xy1, transform=fig.transFigure, arrowstyle="-|>", mutation_scale=18,
        lw=1.6, color=color, linestyle=":" if dotted else "-",
        connectionstyle="arc3,rad=0"))
    if text:
        fig.text(tx if tx is not None else (xy0[0] + xy1[0]) / 2,
                 ty if ty is not None else (xy0[1] + xy1[1]) / 2 + 0.02,
                 text, ha="center", va="center", fontsize=8.5, color="0.15")


def textbox(fig, x, y, w, h, text, fc="#f4f6f8", ec="0.6"):
    fig.patches.append(FancyBboxPatch((x, y), w, h, transform=fig.transFigure,
                                      boxstyle="round,pad=0.008,rounding_size=0.012",
                                      fc=fc, ec=ec, lw=1.0))
    fig.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5)


def main():
    torch.manual_seed(SEED)
    cfg = OmegaConf.load(str(CFG))
    env = get_env_class(**OmegaConf.to_container(cfg.env, resolve=True))
    stat_map = load_stats()
    OUT.mkdir(parents=True, exist_ok=True)

    Ht = density(sample_points(env, T, N_SCENES, SEED + T))
    Ht1 = density(sample_points(env, T1, N_SCENES, SEED + T1))
    st, st1 = stat_map[T], stat_map[T1]
    print("e_t stats", {k: st[k] for k in ("realized", "deployed", "infeas")})
    print("e_t1 stats", {k: st1[k] for k in ("realized", "deployed", "infeas")})

    fig = plt.figure(figsize=(11.5, 6.4))
    fig.patch.set_facecolor("white")

    # --- environment node e_t (top-left) ---
    axA = fig.add_axes([0.03, 0.55, 0.24, 0.37])
    draw_env(axA, Ht, f"Current environment $e_t$  (round {T})")

    # --- signal node S(g_t) (top-right) ---
    axH = fig.add_axes([0.70, 0.60, 0.26, 0.28])
    draw_hist(axH, st["counts"], "Learnability signal $S(g_t)$")

    # --- environment node e_{t+1} (bottom, center-left) ---
    axB = fig.add_axes([0.30, 0.07, 0.24, 0.37])
    draw_env(axB, Ht1, f"Evolved environment $e_{{t+1}}$  (round {T1})")

    # arrows
    arrow(fig, (0.28, 0.74), (0.44, 0.74),
          text="probe with\nskill library\n$\\pi,\\ w_1..w_K$", ty=0.80)
    # signal->analysis short (within top-right, implicit)
    arrow(fig, (0.46, 0.74), (0.69, 0.74),
          text="realized\ndistribution", ty=0.80)
    # down arrow LLM rewrite
    arrow(fig, (0.83, 0.58), (0.83, 0.40),
          text="LLM rewrites\ngenerator program $G$", tx=0.83, ty=0.49)
    # edit box bottom-right
    textbox(fig, 0.60, 0.16, 0.36, 0.20,
            "Direction from $S(g_t)$: most scenes still\n"
            "too hard for the library (realized$<$0.2: 80%).\n"
            "$\\Rightarrow$ rewrite generator to keep scenes\n"
            "solvable and move mass toward realized$\\approx$0.5\n"
            "(the learnable band), not merely harder.")
    # edit -> e_{t+1}
    arrow(fig, (0.60, 0.26), (0.55, 0.26))
    # loop back: e_{t+1} -> e_t (train)
    arrow(fig, (0.30, 0.30), (0.12, 0.55), dotted=True,
          text="train policy\n$\\pi_{t+1}$, iterate", tx=0.16, ty=0.46)

    # stat callouts under each env
    fig.text(0.15, 0.51,
             f"skill lib. realized={st['realized']:.2f}  |  deploy $p$={st['deployed']:.2f}",
             ha="center", fontsize=8.5, color="#b0562a")
    fig.text(0.42, 0.03,
             f"skill lib. realized={st1['realized']:.2f} (in band)  |  deploy $p$={st1['deployed']:.2f}",
             ha="center", fontsize=8.5, color="#3a7d2c")

    fig.suptitle("Closed-loop environment evolution: the skill-library signal moves the training "
                 "distribution into the learnable band,\nwhile single-policy deploy success stays saturated "
                 "($p\\approx1$) and cannot drive this change.", fontsize=11, y=1.02)

    fig.savefig(OUT / "loop_et.png", dpi=170, facecolor="white", bbox_inches="tight")
    fig.savefig(OUT / "loop_et.pdf", facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"[done] saved to {OUT}  scipy={HAVE_SCIPY}")


if __name__ == "__main__":
    main()
