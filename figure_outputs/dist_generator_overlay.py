"""Generator-distribution figure on the REAL environment view.

We cannot place many obstacle bodies in one MuJoCo scene (constraint-arena
overflow), so we:
  1) render one clean environment (real top-down sim view) as the background,
  2) calibrate a world->pixel map by moving a real obstacle to known positions,
  3) overlay the TRUE obstacle-placement distribution of each generator
     (many samples, obstacle size 0.01) as a scatter on that background.

Two panels: g_t distribution -> g_{t+1} distribution. Real data throughout.
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
from DIVO.curriculum.skill_signal import SceneSamplingStats, sample_generator_scene
from DIVO.curriculum.candidate_verifier import SandboxPushTExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import LinearSegmentedColormap
try:
    from scipy.ndimage import gaussian_filter
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

RUN = REPO / "data/outputs/2026.06.22_g0_skill_signal_evolve_s0_clean"
CFG = RUN / ".hydra/config.yaml"
OUT = REPO / "figure_outputs/dist_generator"
ROUNDS = [0, 9, 19]
N_SAMPLES = 200          # obstacle positions per generator (both obstacles used)
OBST_SIZE = 0.01         # training obstacle size (faithful)
ZTOP = 1.111 + OBST_SIZE
ZHIDE = -10.0
SEED = 3


def hide_all(env):
    for i in range(env.task.obstacle_num):
        env.task._obstacle[i].set_pose(env.physics, position=(0, 0, ZHIDE), quaternion=(1, 0, 0, 0))
    env.physics.forward()


def place_one(env, x, y):
    env.task._obstacle[0].set_pose(env.physics, position=(x, y, ZTOP), quaternion=(1, 0, 0, 0))
    for i in range(1, env.task.obstacle_num):
        env.task._obstacle[i].set_pose(env.physics, position=(0, 0, ZHIDE), quaternion=(1, 0, 0, 0))
    env.physics.forward()


def render(env):
    return np.asarray(env.physics.render())


def calibrate(env, bg, log):
    """Fit u = a*x+b*y+c ; v = d*x+e*y+f by moving a real obstacle."""
    world = [(0.15, 0.0), (-0.15, 0.0), (0.0, 0.15), (0.0, -0.15), (0.10, 0.10), (-0.10, -0.10)]
    U, V, W = [], [], []
    for (x, y) in world:
        place_one(env, x, y)
        img = render(env).astype(float)
        diff = np.abs(img - bg).sum(axis=2)
        mask = diff > 40
        if mask.sum() < 5:
            continue
        ys, xs = np.nonzero(mask)
        U.append(xs.mean()); V.append(ys.mean()); W.append((x, y))
    W = np.array(W); A = np.column_stack([W[:, 0], W[:, 1], np.ones(len(W))])
    cu, *_ = np.linalg.lstsq(A, np.array(U), rcond=None)
    cv, *_ = np.linalg.lstsq(A, np.array(V), rcond=None)
    # residuals
    ru = A @ cu - np.array(U); rv = A @ cv - np.array(V)
    print(f"[calib] n={len(W)} resid_u={np.abs(ru).max():.2f}px resid_v={np.abs(rv).max():.2f}px", file=log, flush=True)
    scale = np.hypot(cu[0], cv[0])  # px per world unit (x-axis)
    return cu, cv, scale


def world_to_px(cu, cv, x, y):
    return cu[0] * x + cu[1] * y + cu[2], cv[0] * x + cv[1] * y + cv[2]


def sample_obstacles(env, idx, n, seed):
    ex = SandboxPushTExecutor(obstacle_size=OBST_SIZE)
    ex.load((RUN / f"generators/generator_{idx:03d}.py").read_text(encoding="utf-8"))
    st = SceneSamplingStats(); rng = np.random.RandomState(seed)
    pts, tries = [], 0
    while len(pts) < n and tries < n * 12:
        tries += 1
        np.random.seed(rng.randint(1 << 30))
        sc = sample_generator_scene(env, ex, 2, timeout_sec=5, stats=st)
        if sc:
            for o in sc["obstacles"]:
                pts.append((float(o["x"]), float(o["y"])))
    return np.array(pts[:n])


def main():
    torch.manual_seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    log = open(OUT / "build.log", "w")
    cfg = OmegaConf.load(str(CFG))
    ec = OmegaConf.to_container(cfg.env, resolve=True)
    ec["obstacle_num"] = 2; ec["obstacle_size"] = OBST_SIZE
    env = get_env_class(**ec)
    # get a valid initial physics state
    env.set_obstacle_config([{"x": 0.15, "y": 0.15}, {"x": -0.15, "y": -0.15}])
    env.reset()

    hide_all(env)
    bg = render(env).astype(float)
    print(f"[bg] shape={bg.shape} mean={int(bg.mean())}", file=log, flush=True)

    cu, cv, scale = calibrate(env, bg, log)
    bg_img = bg.astype(np.uint8)

    # sample each generator's obstacle distribution
    clouds = {r: sample_obstacles(env, r, N_SAMPLES, SEED + r) for r in ROUNDS}
    for r in ROUNDS:
        print(f"[gen {r}] {len(clouds[r])} obstacle positions", file=log, flush=True)

    # obstacle radius in pixels (true 0.01 size, slightly upscaled for visibility)
    r_px = max(3.0, scale * OBST_SIZE * 1.4)
    H, W = bg_img.shape[:2]

    # translucent orange colormap for the density shading layer
    dens_cmap = LinearSegmentedColormap.from_list(
        "orange_dens", [(1, 1, 1, 0.0), (1.0, 0.62, 0.28, 0.55), (0.85, 0.33, 0.0, 0.85)])

    def px_cloud(r):
        us, vs = [], []
        for (x, y) in clouds[r]:
            u, v = world_to_px(cu, cv, x, y)
            us.append(u); vs.append(v)
        return np.array(us), np.array(vs)

    def density_img(us, vs):
        Hd, _, _ = np.histogram2d(vs, us, bins=[H // 6, W // 6],
                                  range=[[0, H], [0, W]])
        if HAVE_SCIPY:
            Hd = gaussian_filter(Hd, sigma=1.4)
        return Hd / (Hd.max() + 1e-9)

    n = len(ROUNDS)
    fig, axes = plt.subplots(1, n, figsize=(n * 3.1 + 0.4, 3.3))
    axes = np.atleast_1d(axes)
    labels = ([f"Original distribution $g_t$\n(round {ROUNDS[0]})"] +
              [f"round {r}" for r in ROUNDS[1:-1]] +
              [f"Evolved distribution $g_{{t+1}}$\n(round {ROUNDS[-1]})"])
    for i, r in enumerate(ROUNDS):
        ax = axes[i]
        ax.imshow(bg_img)
        us, vs = px_cloud(r)
        # light density shading under the points
        ax.imshow(density_img(us, vs), extent=[0, W, H, 0], origin="upper",
                  cmap=dens_cmap, vmin=0, vmax=1, interpolation="bilinear", zorder=3)
        # individual obstacle points (true size)
        for u, v in zip(us, vs):
            ax.add_patch(Circle((u, v), r_px, facecolor="#ff7f0e", edgecolor="none",
                                alpha=0.42, zorder=5))
        ax.set_xlim(0, W); ax.set_ylim(H, 0)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(labels[i], fontsize=10)

    for i in range(n - 1):
        x = (axes[i].get_position().x1 + axes[i + 1].get_position().x0) / 2
        fig.text(x, 0.5, "\u2192", ha="center", va="center", fontsize=22, color="0.35")

    fig.suptitle("Obstacle-placement distribution of the generator on the real "
                 "environment view, across evolution rounds\n(obstacle size 0.01; "
                 "~%d samples per panel; shading = placement density)" % N_SAMPLES,
                 fontsize=10.5, y=1.05)
    fig.patch.set_facecolor("white")
    fig.savefig(OUT / "dist_generator.png", dpi=180, facecolor="white", bbox_inches="tight")
    fig.savefig(OUT / "dist_generator.pdf", facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("[done]", file=log, flush=True)
    log.close()


if __name__ == "__main__":
    main()
