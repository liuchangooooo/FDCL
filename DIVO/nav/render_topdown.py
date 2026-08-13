"""俯视示意图(对标 DIVO 训练环境图):一行五格 = 训练环境 + B/M/U/D。

风格:干净俯视,虚线红框=障碍分布区,粉/红点=采样 pillar,X=start,绿星=goal。
用法(safenav):
  MUJOCO_GL=egl python -m nav.render_topdown
"""
import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch

from nav import nav_env as NE
from nav import benchmarks as B
from nav.protocol import BENCHMARK_VERSION

GOAL = NE.GOAL
PILL = "#d94a4a"       # pillar 颜色
DIST = "#e06666"       # 分布区红


def _axis(ax, title):
    ax.set_xlim(NE.EXTENTS[0], NE.EXTENTS[2])
    ax.set_ylim(NE.EXTENTS[1], NE.EXTENTS[3])
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#3b5b92"); s.set_linewidth(2)
    ax.set_facecolor("#f7f9fc")
    ax.set_title(title, fontsize=13, pad=6)


def _goal(ax):
    ax.plot(*GOAL, marker="*", color="#2e9e4f", ms=22, mec="#1c6b34", mew=1.2, zorder=5)


def _start_x(ax, xy):
    ax.plot(*xy, marker="X", color="#555", ms=14, mew=2, zorder=5)


def _pillars(ax, pillars, size=NE.PILLAR_SIZE, color=PILL, alpha=0.9):
    for x, y in pillars:
        ax.add_patch(Circle((x, y), size, color=color, alpha=alpha, zorder=3))


def panel_training(ax):
    _axis(ax, "Training Environment")
    # 障碍分布区(训练 DOF 中央带)虚线圆角框
    x0, x1 = NE.TRAIN_X_RANGE; y0, y1 = NE.TRAIN_Y_RANGE
    box = FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                         boxstyle="round,pad=0.08,rounding_size=0.25",
                         fill=True, fc=DIST, ec=DIST, alpha=0.13, ls="--", lw=2, zorder=1)
    ax.add_patch(box)
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                 boxstyle="round,pad=0.08,rounding_size=0.25",
                 fill=False, ec=DIST, ls=(0, (5, 4)), lw=2, zorder=2))
    # 训练协议:恰好 2 个标准静态 pillar。
    rng = np.random.default_rng(2)
    pts = NE.sample_training_layout(rng)
    _pillars(ax, pts[1:], color="#f1a6a6", alpha=0.9)
    ax.add_patch(Circle(pts[0], NE.PILLAR_SIZE, color=PILL, zorder=4))
    # start / goal
    _start_x(ax, NE.START); _goal(ax)
    # 标注
    ax.annotate("sampled pillar", xy=pts[0], xytext=(-0.82, 0.78), fontsize=9, color=PILL,
                arrowprops=dict(arrowstyle="->", color=PILL, lw=1.5))
    ax.text(0.0, y1 + 0.28, "obstacle distribution", color=DIST, fontsize=9, ha="center")
    ax.text(NE.START[0], NE.START[1] - 0.28, "start", fontsize=9, ha="center", color="#555")
    ax.text(GOAL[0], GOAL[1] - 0.30, "goal", fontsize=9, ha="center", color="#2e9e4f")


def panel_bench(ax, fam, title, seed=7):
    _axis(ax, title)
    sc = B.sample_benchmark_scene(fam, seed)
    _pillars(ax, sc["pillars"], size=sc.get("size", NE.PILLAR_SIZE))
    if sc.get("dynamic", False):
        travel = sc["travel"]
        for x, y in sc["pillars"]:
            ax.add_patch(Circle((x, y), travel, fill=False, ec="#3b5b92", ls="--", lw=1.2))
            ax.add_patch(FancyArrowPatch(
                (x + travel, y), (x, y + travel), connectionstyle="arc3,rad=0.35",
                arrowstyle="->", mutation_scale=8, color="#3b5b92", lw=1.1,
            ))
    # 与正式 evaluator 相同的固定起点
    _start_x(ax, sc["start"]); _goal(ax)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out", default=f"nav/templates/{BENCHMARK_VERSION}/topdown_overview.png"
    )
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    fig, axes = plt.subplots(1, 5, figsize=(20, 4.2))
    panel_training(axes[0])
    panel_bench(axes[1], "B", "Big (B)")
    panel_bench(axes[2], "M", "Multiple (M)")
    panel_bench(axes[3], "U", "Unstructured (U)")
    panel_bench(axes[4], "D", "Dynamic (D)")
    fig.suptitle("Navigation (SafetyPointGoal) — Training Environment & B/M/U/D benchmarks",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
