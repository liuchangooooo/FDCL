"""Panel (1) real Push-T layout thumbnails, captured from the MuJoCo sim.

Samples N valid layouts from the current generator, renders each with
`physics.render()` (real sim frame), and saves clean PNGs + a contact sheet.
Pick 4-6 for the figure's "Current Generator & Distribution" panel.
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

CFG = REPO / "data/outputs/2026.06.22_g0_skill_signal_evolve_s0_clean/.hydra/config.yaml"
GEN = REPO / "data/outputs/2026.04.29/g0_seed2/current_generator.py"
OUT = REPO / "figure_outputs/panel1_layouts"
N = 8
RES = 512
SEED = 11


def render(env):
    # single 'top_camera' (top-down); framebuffer capped at 480x640.
    for kw in (dict(height=480, width=480, camera_id=0),
               dict(height=480, width=480), dict()):
        try:
            return np.asarray(env.physics.render(**kw))
        except Exception:
            continue
    return None


def main():
    np.random.seed(SEED); torch.manual_seed(SEED)
    cfg = OmegaConf.load(str(CFG))
    env = get_env_class(**OmegaConf.to_container(cfg.env, resolve=True))
    ex = SandboxPushTExecutor(obstacle_size=0.01)
    ex.load(GEN.read_text(encoding="utf-8"))
    stats = SceneSamplingStats()
    OUT.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frames = []
    tries = 0
    while len(frames) < N and tries < N * 20:
        tries += 1
        sc = sample_generator_scene(env, ex, 2, timeout_sec=5, stats=stats)
        if sc is None:
            continue
        env.set_obstacle_config(sc["obstacles"])
        env.reset(tblock_pos=np.asarray(sc["start"], float), force_tblock_pos=True)
        img = render(env)
        if img is None:
            print("render failed"); return
        idx = len(frames)
        plt.imsave(OUT / f"layout_{idx:02d}.png", img)
        frames.append(img)
        print(f"[{idx}] start={np.round(sc['start'],3)} obst={[(round(o['x'],2),round(o['y'],2)) for o in sc['obstacles']]}")

    cols = 4
    rows = (len(frames) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))
    for i, ax in enumerate(np.atleast_1d(axes).ravel()):
        ax.axis("off")
        if i < len(frames):
            ax.imshow(frames[i]); ax.set_title(f"layout {i}", fontsize=9)
    fig.patch.set_facecolor("white"); fig.tight_layout()
    fig.savefig(OUT / "_contact_sheet.png", dpi=140, facecolor="white")
    plt.close(fig)
    print(f"[done] {len(frames)} layouts in {OUT}")


if __name__ == "__main__":
    main()
