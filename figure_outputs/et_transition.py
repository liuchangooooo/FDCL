"""e_t -> e_{t+1} environment-evolution figure with REAL top-down renders.

Same task (fixed T start pose + fixed goal). For each curriculum round we call
that round's accepted generator to place obstacles, then render the actual
MuJoCo Push-T scene top-down (free camera). Reading left to right shows the
obstacle generator evolving the training environment.

Real: generators are accepted programs; renders are the actual simulator.
"""
from __future__ import annotations
import os, pathlib, sys
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
from DIVO.curriculum.candidate_verifier import SandboxPushTExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN = REPO / "data/outputs/2026.06.22_g0_skill_signal_evolve_s0_clean"
CFG = RUN / ".hydra/config.yaml"
OUT = REPO / "figure_outputs/et_transition"
ROUNDS = [0, 9, 19]
SEED = 4


def render(env):
    # default dm_control framing (240x320, top-down free camera) -- the good view
    return np.asarray(env.physics.render())


def gen_obstacles(ex, start, n=2, seed=0):
    rng = np.random.RandomState(seed)
    for _ in range(60):
        np.random.seed(rng.randint(1 << 30))
        obs = ex.generate(np.asarray(start, float), n)
        if obs and len(obs) >= n:
            return obs[:n]
    return obs if obs else []


def valid_start(env):
    # a fixed, clearly visible start pose away from the goal at origin
    return np.array([0.12, 0.11, 5.5], float)


def main():
    torch.manual_seed(SEED)
    cfg = OmegaConf.load(str(CFG))
    envcfg = OmegaConf.to_container(cfg.env, resolve=True)
    envcfg["obstacle_size"] = 0.05   # obstacles clearly visible in the figure
    env = get_env_class(**envcfg)
    OUT.mkdir(parents=True, exist_ok=True)
    start = valid_start(env)

    frames = []
    for r in ROUNDS:
        ex = SandboxPushTExecutor(obstacle_size=0.05)
        ex.load((RUN / f"generators/generator_{r:03d}.py").read_text(encoding="utf-8"))
        obs = gen_obstacles(ex, start, 2, seed=SEED + r)
        env.set_obstacle_config(obs)
        env.reset(tblock_pos=start, force_tblock_pos=True)
        img = render(env)
        frames.append(img)
        print(f"round {r}: obst={[(round(o['x'],2),round(o['y'],2)) for o in obs]} "
              f"img={img.shape} mean={img.mean():.0f}")

    n = len(ROUNDS)
    fig, axes = plt.subplots(1, n, figsize=(n * 3.0 + 0.4, 3.2))
    axes = np.atleast_1d(axes)
    labels = ([f"Original environment $e_t$\n(round {ROUNDS[0]})"] +
              [f"round {r}" for r in ROUNDS[1:-1]] +
              [f"Perturbed environment $e_{{t+1}}$\n(round {ROUNDS[-1]})"])
    for i, r in enumerate(ROUNDS):
        ax = axes[i]
        ax.imshow(frames[i])
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor("0.7")
        ax.set_title(labels[i], fontsize=10)

    for i in range(n - 1):
        x = (axes[i].get_position().x1 + axes[i + 1].get_position().x0) / 2
        fig.text(x, 0.5, "\u2192", ha="center", va="center", fontsize=24, color="0.35")

    fig.suptitle("The obstacle generator evolves the training environment across "
                 "curriculum rounds (same start & goal; real top-down renders)",
                 fontsize=10.5, y=1.02)
    fig.patch.set_facecolor("white")
    fig.savefig(OUT / "et_transition.png", dpi=180, facecolor="white", bbox_inches="tight")
    fig.savefig(OUT / "et_transition.pdf", facecolor="white", bbox_inches="tight")
    # also save the raw panels for hand-assembly / a 2-panel version
    for i, r in enumerate(ROUNDS):
        plt.imsave(OUT / f"panel_round{r:02d}.png", frames[i])
    plt.close(fig)
    print(f"[done] saved to {OUT}")


if __name__ == "__main__":
    main()
