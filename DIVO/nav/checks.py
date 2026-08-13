"""导航环境验证脚本(合并 inspect / smoke / benchmark render / geometry check)。

用法(在 safenav 环境下):
    MUJOCO_GL=egl python -m nav.checks all       # 跑全部
    MUJOCO_GL=egl python -m nav.checks obs        # 打印 obs 结构
    MUJOCO_GL=egl python -m nav.checks render     # 渲染正式 B/M/U/D + 训练/压力样例
    MUJOCO_GL=egl python -m nav.checks difficulty # 布局几何量化检查
"""
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

from nav import nav_env as NE
from nav import benchmarks as Bmk
from nav.nav_adapter import NavEnvAdapter
from nav.protocol import BENCHMARK_VERSION
from safety_gymnasium.utils.common_utils import ResamplingError

OUTDIR = os.path.join(os.path.dirname(__file__), "templates", BENCHMARK_VERSION)


# ----------------------------- obs 结构 -----------------------------
def check_obs():
    """打印 pillar 版 obs 的有序分量与 (state, z) 切分。"""
    layout = {
        "start": NE.START,
        "goal": NE.GOAL,
        "pillars": [(-0.25, 0.55), (0.25, -0.55)],
    }
    env = NE.make_env(layout)
    task = env.task
    task.observation_flatten = False
    task.build_observation_space()
    obs, _ = env.reset(seed=0)

    print("=" * 68)
    print("obs 分量(flatten=False):")
    offset = 0
    for k, v in obs.items():
        n = np.asarray(v).size
        print(f"  {k:16s} [{offset:2d}:{offset + n:2d}]  dim={n}")
        offset += n
    print(f"  TOTAL dim = {offset}")
    print(f"  action_space = {env.action_space}")
    print(f"  => STATE_SLICE={NE.STATE_SLICE}  Z_INPUT_SLICE={NE.Z_INPUT_SLICE}")
    print("=" * 68)
    env.close()


# ------------------------------ 渲染 ------------------------------
def render_all():
    import imageio.v2 as imageio

    os.makedirs(OUTDIR, exist_ok=True)
    print(f"渲染到 {OUTDIR}")

    names = {"B": "Big", "M": "Multiple", "U": "Unstructured", "D": "Dynamic"}
    static_ad = NavEnvAdapter(render_mode="rgb_array")
    dynamic_ad = NavEnvAdapter(render_mode="rgb_array", dynamic=True)
    for family in ("B", "M", "U", "D"):
        rendered = False
        for scene_seed in range(100, 120):
            scene = Bmk.sample_benchmark_scene(family, seed=scene_seed)
            ad = dynamic_ad if scene.get("dynamic", False) else static_ad
            ad.set_layout(
                scene["pillars"],
                start=scene["start"],
                goal=scene["goal"],
                pillar_size=scene["size"],
                pillar_keepout=scene["keepout"],
                gremlin_travel=scene.get("travel"),
            )
            try:
                ad.reset(seed=scene["reset_seed"], start=scene["start"])
                if scene.get("dynamic", False):
                    for _ in range(20):
                        ad.step(np.zeros(2))
                imageio.imwrite(os.path.join(OUTDIR, f"benchmark_{family}.png"), ad.render())
                print(
                    f"  [{family}:{names[family]:12s}] obstacles={len(scene['pillars']):2d}  "
                    f"scene_seed={scene_seed}  OK"
                )
                rendered = True
                break
            except ResamplingError:
                continue
        if not rendered:
            print(f"  [{family}:{names[family]:12s}] no valid reset, skipped")
    static_ad.close()
    dynamic_ad.close()

    # 手工结构只是额外压力测试，不构成正式 B/M/U/D。
    stress_ad = NavEnvAdapter(render_mode="rgb_array")
    for name, fn in NE.STRUCTURED_STRESS_TESTS.items():
        pillars = NE.dedupe(fn())
        stress_ad.set_layout(pillars, start=NE.START, goal=NE.GOAL)
        try:
            stress_ad.reset(seed=0)
            imageio.imwrite(os.path.join(OUTDIR, f"stress_{name}.png"), stress_ad.render())
            print(f"  [stress/{name:8s}] pillars={len(pillars):2d}  OK")
        except ResamplingError:
            print(f"  [stress/{name:8s}] invalid layout, skipped")
    stress_ad.close()

    rng = np.random.default_rng(0)
    for i in range(2):
        pillars = NE.dedupe(NE.sample_training_layout(rng))
        env = NE.make_env(
            {"start": NE.START, "goal": NE.GOAL, "pillars": pillars}, render_mode="rgb_array"
        )
        try:
            env.reset(seed=0)
            imageio.imwrite(os.path.join(OUTDIR, f"train_sample_{i}.png"), env.render())
            print(f"  [train_{i}     ] pillars={len(pillars):2d}  OK")
        finally:
            env.close()


# --------------------------- 难度量化检查 ---------------------------
def _min_gap_on_line(pillars, n=400):
    if not pillars:
        return np.inf
    P = np.array(pillars)
    ts = np.linspace(0, 1, n)
    pts = np.array(NE.START)[None, :] + ts[:, None] * (np.array(NE.GOAL) - np.array(NE.START))[None, :]
    d = np.linalg.norm(pts[:, None, :] - P[None, :, :], axis=2)
    return d.min(axis=1).min()


def _first_block_x(pillars, clearance, n=400):
    if not pillars:
        return None
    P = np.array(pillars)
    ts = np.linspace(0, 1, n)
    pts = np.array(NE.START)[None, :] + ts[:, None] * (np.array(NE.GOAL) - np.array(NE.START))[None, :]
    for pt in pts:
        if np.linalg.norm(P - pt[None, :], axis=1).min() < clearance:
            return round(float(pt[0]), 3)
    return None


def _min_pairwise(pillars):
    if len(pillars) < 2:
        return np.inf
    P = np.array(pillars)
    d = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
    d[np.eye(len(P), dtype=bool)] = np.inf
    return d.min()


def _report(name, pillars, size=NE.PILLAR_SIZE, keepout=NE.PILLAR_KEEPOUT):
    pillars = NE.dedupe(pillars)
    clearance = float(size) + NE.AGENT_RADIUS
    gap = _min_gap_on_line(pillars)
    fbx = _first_block_x(pillars, clearance)
    mp = _min_pairwise(pillars)
    blocked = "BLOCKED" if gap < clearance else "open"
    keepout_ok = mp >= 2 * keepout - 1e-6
    physical_ok = mp >= 2 * size - 1e-6
    print(
        f"  [{name:12s}] n={len(pillars):2d}  line_gap={gap:.3f}({blocked})  "
        f"first_block_x={fbx}  min_pair={mp:.3f} "
        f"keepout_ok={keepout_ok} physical_ok={physical_ok}"
    )


def check_difficulty():
    print("=" * 90)
    print(
        f"TRAIN_SIZE={NE.PILLAR_SIZE} AGENT_R={NE.AGENT_RADIUS}  "
        "(line_gap<size+agent_radius => 直线被挡)"
    )
    print("-" * 90)
    print("正式 held-out B/M/U/D:")
    formal_names = {"B": "Big", "M": "Multiple", "U": "Unstructured", "D": "Dynamic"}
    for family in ("B", "M", "U", "D"):
        scene = Bmk.sample_benchmark_scene(family, seed=100)
        _report(
            f"{family}:{formal_names[family]}",
            scene["pillars"],
            size=scene["size"],
            keepout=scene["keepout"],
        )
    print("-" * 90)
    print("结构压力测试(不计入 B/M/U/D):")
    for nm, fn in NE.STRUCTURED_STRESS_TESTS.items():
        _report(f"stress/{nm}", fn())
    print("-" * 90)
    print("固定 2-pillar 训练分布:")
    rng = np.random.default_rng(0)
    for i in range(4):
        _report(f"train_{i}", NE.sample_training_layout(rng))
    print("=" * 90)


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("all", "obs"):
        check_obs()
    if what in ("all", "difficulty"):
        check_difficulty()
    if what in ("all", "render"):
        render_all()


if __name__ == "__main__":
    main()
