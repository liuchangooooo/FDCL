"""Render top-down frames of the OOD evaluation columns so we can SEE the scenes.

For each obstacle family, build the eval env, reset several times, grab a rendered
frame, and save a montage PNG per column.
"""
import sys, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from DIVO.env import get_env_class

COLUMNS = {
    "DIVO-M_n3_s0.03": dict(obstacle_num=3, obstacle_size=0.03),  # DIVO original M
    "MP_n4_s0.03": dict(obstacle_num=4, obstacle_size=0.03),
    "MPP_n4_s0.04": dict(obstacle_num=4, obstacle_size=0.04),
    "MPPP_n4_s0.05": dict(obstacle_num=4, obstacle_size=0.05),
    "BP_n2_s0.06": dict(obstacle_num=2, obstacle_size=0.06),
}
N_SAMPLES = 4


def build(obstacle_num, obstacle_size):
    return get_env_class(
        _target_="pusht_mujoco",
        obstacle=True,
        obstacle_num=obstacle_num,
        obstacle_size=obstacle_size,
        obstacle_shape="box",
        obstacle_dist="random",
        action_scale=4,
        NUM_SUBSTEPS=5,
        action_dim=[6],
        obs_dim=[6],
        action_reg=True,
        reg_coeff=1.0,
        generate_dataset=False,
        motion_pred=False,
        eval=True,
        dynamics_randomization=False,
    )


def main():
    out = REPO_ROOT / "analysis" / "skill_coverage" / "ood_scene_previews"
    out.mkdir(parents=True, exist_ok=True)
    ncols = len(COLUMNS)
    fig, axes = plt.subplots(N_SAMPLES, ncols, figsize=(3.2 * ncols, 3.2 * N_SAMPLES))
    for c, (name, params) in enumerate(COLUMNS.items()):
        env = build(**params)
        env._record_frame = True
        for r in range(N_SAMPLES):
            np.random.seed(100 + r)
            env.reset()
            try:
                frame = env.physics.render()
            except Exception as e:
                frame = np.zeros((240, 240, 3), dtype=np.uint8)
                print(f"[warn] {name} render failed: {e}")
            ax = axes[r, c]
            ax.imshow(frame)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(name, fontsize=11)
    fig.suptitle("OOD evaluation columns (top-down) | training = n2/size0.01/between", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p = out / "deepen_columns_montage.png"
    fig.savefig(p, dpi=130)
    print(f"[saved] {p}")


if __name__ == "__main__":
    main()
