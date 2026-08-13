"""Render an annotated map of Nav train/validation/BMUD and Push-T geometry.

Run from DIVO/DIVO:
  python -m nav.render_protocol_map --out nav/runs/nav_protocol_map.png
"""
import argparse
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/divo_mpl_cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches, transforms
import numpy as np

from nav import benchmarks as B
from nav import nav_env as NE
from nav.curriculum.generator_source import GOAL_CLEAR, START_CLEAR
from nav.eval_dist import sample_validation_scene


START_COLOR = "#2563eb"
GOAL_COLOR = "#16a34a"
OBSTACLE_COLOR = "#ea580c"
SUPPORT_COLOR = "#f59e0b"


def _rect(ax, x0, y0, width, height, **kwargs):
    patch = patches.Rectangle((x0, y0), width, height, **kwargs)
    ax.add_patch(patch)
    return patch


def _circle(ax, center, radius, **kwargs):
    patch = patches.Circle(center, radius, **kwargs)
    ax.add_patch(patch)
    return patch


def _nav_base(ax, title, obstacle_region, show_start_support):
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_aspect("equal")
    ax.set_xlim(-1.08, 1.08)
    ax.set_ylim(-1.08, 1.08)
    ax.set_xticks(np.arange(-1.0, 1.01, 0.25))
    ax.set_yticks(np.arange(-1.0, 1.01, 0.25))
    ax.grid(True, color="#d1d5db", linewidth=0.5, alpha=0.65)
    ax.set_xlabel("world x")
    ax.set_ylabel("world y")

    _rect(
        ax, -NE.OOB_BOUND, -NE.OOB_BOUND, 2 * NE.OOB_BOUND, 2 * NE.OOB_BOUND,
        fill=False, edgecolor="#7c3aed", linewidth=1.5, linestyle=":"
    )
    _rect(
        ax, NE.EXTENTS[0], NE.EXTENTS[1],
        NE.EXTENTS[2] - NE.EXTENTS[0], NE.EXTENTS[3] - NE.EXTENTS[1],
        fill=False, edgecolor="black", linewidth=2.0,
    )
    _rect(
        ax, -obstacle_region, -obstacle_region, 2 * obstacle_region, 2 * obstacle_region,
        facecolor=SUPPORT_COLOR, edgecolor=OBSTACLE_COLOR, alpha=0.13,
        linewidth=1.5, linestyle="--",
    )
    if show_start_support:
        _rect(
            ax, NE.START_X_RANGE[0], NE.START_Y_RANGE[0],
            NE.START_X_RANGE[1] - NE.START_X_RANGE[0],
            NE.START_Y_RANGE[1] - NE.START_Y_RANGE[0],
            facecolor=START_COLOR, edgecolor=START_COLOR, alpha=0.12,
            linewidth=1.5,
        )
    _circle(ax, NE.GOAL, NE.GOAL_SIZE, facecolor=GOAL_COLOR, alpha=0.35,
            edgecolor=GOAL_COLOR, linewidth=2.0)
    ax.scatter(*NE.GOAL, marker="*", s=120, color=GOAL_COLOR, zorder=8)
    ax.annotate(
        f"fixed goal {NE.GOAL}\nsuccess radius={NE.GOAL_SIZE:.2f}",
        NE.GOAL, xytext=(8, 12), textcoords="offset points", fontsize=8,
    )
    ax.annotate(
        "placement extents 1.8 x 1.8",
        (-0.88, -0.88), xytext=(4, 4), textcoords="offset points", fontsize=8,
    )
    ax.annotate(
        "OOB termination at |x| or |y| > 1.0",
        (-0.98, 1.0), xytext=(3, -12), textcoords="offset points",
        fontsize=8, color="#7c3aed",
    )


def _draw_nav_scene(ax, start, pillars, show_clearance=True, route=True):
    start = np.asarray(start, dtype=float)
    if route:
        ax.plot([start[0], NE.GOAL[0]], [start[1], NE.GOAL[1]],
                color="#64748b", linestyle="--", linewidth=1.4, zorder=2)
    if show_clearance:
        _circle(ax, start, START_CLEAR, fill=False, edgecolor=START_COLOR,
                linestyle=":", linewidth=1.3)
        _circle(ax, NE.GOAL, GOAL_CLEAR, fill=False, edgecolor=GOAL_COLOR,
                linestyle=":", linewidth=1.3)
    _circle(ax, start, NE.AGENT_RADIUS, facecolor=START_COLOR, alpha=0.75,
            edgecolor="#1e3a8a", linewidth=1.5, zorder=6)
    ax.scatter(*start, color="#1e3a8a", s=18, zorder=8)
    for i, pillar in enumerate(pillars, 1):
        _circle(ax, pillar, NE.PILLAR_SIZE, facecolor=OBSTACLE_COLOR, alpha=0.65,
                edgecolor="#9a3412", linewidth=1.5, zorder=5)
        ax.text(pillar[0], pillar[1], f"P{i}", ha="center", va="center",
                fontsize=8, fontweight="bold", color="white", zorder=8)
    return start


def _training_panel(ax):
    _nav_base(ax, "A. Nav training distribution", NE.OBSTACLE_REGION, True)
    start = (-0.65, 0.0)
    pillars = [(-0.10, 0.30), (0.05, -0.30)]
    _draw_nav_scene(ax, start, pillars)
    ax.annotate("example sampled start", start, xytext=(-5, -23),
                textcoords="offset points", ha="center", fontsize=8, color=START_COLOR)
    ax.text(
        -1.04, 1.045,
        "start ~ U([-0.8,-0.1] x [-0.45,0.45])\n"
        "then G_t(start, goal) generates exactly 2 static Pillars\n"
        "Pillar center support: [-0.5,0.5]^2",
        ha="left", va="top", fontsize=8.2,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.88),
    )
    ax.text(
        -1.04, -1.045,
        f"r_agent={NE.AGENT_RADIUS:.2f}, r_pillar={NE.PILLAR_SIZE:.2f}; "
        f"clear(start,P)>={START_CLEAR:.2f}, clear(goal,P)>={GOAL_CLEAR:.2f}, "
        f"P-P>={NE.PILLAR_MIN_SEPARATION:.2f}",
        ha="left", va="bottom", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.9),
    )


def _validation_panel(ax):
    _nav_base(ax, "B. Nav model-selection validation", NE.VAL_OBSTACLE_REGION, True)
    scene = sample_validation_scene(778)
    start = _draw_nav_scene(ax, scene["start"], scene["pillars"])
    segment = np.asarray(NE.GOAL) - start
    distances = []
    for pillar in scene["pillars"]:
        p = np.asarray(pillar)
        t = np.clip(np.dot(p - start, segment) / np.dot(segment, segment), 0.0, 1.0)
        distances.append(np.linalg.norm(p - (start + t * segment)))
    selected = int(np.argmin(distances))
    ax.annotate(
        "between obstacle\n(blocks straight route)", scene["pillars"][selected],
        xytext=(25, 28), textcoords="offset points", fontsize=8,
        arrowprops=dict(arrowstyle="->", color="#9a3412"),
    )
    ax.text(
        -1.04, 1.045,
        f"2 Pillars sampled first in [-{NE.VAL_OBSTACLE_REGION:.2f},"
        f"{NE.VAL_OBSTACLE_REGION:.2f}]^2 (same as training)\n"
        f"start behind one Pillar; radial offset "
        f"[{NE.BETWEEN_OFFSET_RANGE[0]:.2f},{NE.BETWEEN_OFFSET_RANGE[1]:.2f}]\n"
        "out-of-range / clearance failure => reject whole scene (no clip)",
        ha="left", va="top", fontsize=8.2,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.88),
    )
    ax.text(
        -1.04, -1.045,
        "20 deterministic scenes; w0 mean episode return selects best.pt; not BMUD",
        ha="left", va="bottom", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.9),
    )


def _bmud_panel(ax):
    _nav_base(ax, "C. Nav final BMUD evaluation", NE.OBSTACLE_REGION, False)
    scene = B.sample_benchmark_scene("M", 2025)
    _draw_nav_scene(ax, scene["start"], scene["pillars"])
    ax.annotate("fixed eval start (-0.65, 0)", scene["start"],
                xytext=(-8, -23), textcoords="offset points", ha="center",
                fontsize=8, color=START_COLOR)
    ax.text(
        -1.04, 1.045,
        "fixed start and fixed goal; obstacle positions remain random\n"
        "same center support [-0.5,0.5]^2; no path-conditioned placement",
        ha="left", va="top", fontsize=8.2,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.88),
    )
    ax.text(
        -1.04, -1.045,
        "B: 1 big r=.30 | M: 3 standard r=.15 (shown) | "
        "U: 7-element compound | D: 3 moving r=.10, travel=.08",
        ha="left", va="bottom", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.9),
    )


def _t_shape(ax, center, angle, color, alpha, label):
    trans = transforms.Affine2D().rotate_around(center[0], center[1], angle) + ax.transData
    for xy, width, height in [
        ((center[0] - 0.05, center[1] - 0.015), 0.10, 0.03),
        ((center[0] - 0.015, center[1] - 0.085), 0.03, 0.07),
    ]:
        patch = patches.Rectangle(xy, width, height, facecolor=color,
                                  edgecolor=color, alpha=alpha, transform=trans)
        ax.add_patch(patch)
    ax.text(center[0], center[1], label, fontsize=7, ha="center", va="center",
            color="white", fontweight="bold", zorder=8)


def _pusht_panel(ax):
    ax.set_title("D. Push-T reference geometry", fontsize=13, fontweight="bold")
    ax.set_aspect("equal")
    ax.set_xlim(-0.29, 0.29)
    ax.set_ylim(-0.29, 0.29)
    ax.set_xticks(np.arange(-0.25, 0.251, 0.05))
    ax.set_yticks(np.arange(-0.25, 0.251, 0.05))
    ax.grid(True, color="#d1d5db", linewidth=0.5, alpha=0.65)
    ax.set_xlabel("table x")
    ax.set_ylabel("table y")
    _rect(ax, -0.25, -0.25, 0.5, 0.5, fill=False, edgecolor="black", linewidth=2.0)
    _rect(ax, -0.20, -0.20, 0.40, 0.40, facecolor=SUPPORT_COLOR,
          edgecolor=OBSTACLE_COLOR, alpha=0.12, linestyle="--", linewidth=1.5)
    _rect(ax, -0.18, -0.18, 0.36, 0.36, facecolor=START_COLOR,
          edgecolor=START_COLOR, alpha=0.10, linewidth=1.5)
    _rect(ax, -0.10, -0.10, 0.20, 0.20, facecolor="#6b7280",
          edgecolor="#374151", alpha=0.16, hatch="///", linewidth=1.0)

    _t_shape(ax, (0.0, 0.0), -np.pi / 4, GOAL_COLOR, 0.7, "target")
    _t_shape(ax, (-0.15, 0.15), 3 * np.pi / 4, START_COLOR, 0.75, "start")
    for x, y in [(-0.02, 0.14), (0.14, -0.05)]:
        _rect(ax, x - 0.01, y - 0.01, 0.02, 0.02,
              facecolor=OBSTACLE_COLOR, edgecolor="#9a3412", linewidth=1.2)
        _rect(ax, x - 0.05, y - 0.05, 0.10, 0.10,
              fill=False, edgecolor=OBSTACLE_COLOR, linestyle=":", linewidth=1.0)
    ax.text(
        -0.278, 0.278,
        "desk: [-0.25,0.25]^2; target T fixed at (0,0), angle=-pi/4\n"
        "T-start base support: [-0.18,0.18]^2; center square rejected\n"
        "2 obstacle centers: [-0.2,0.2]^2; physical side=.02",
        ha="left", va="top", fontsize=8.2,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
    )
    ax.text(
        -0.278, -0.278,
        "dotted obstacle box: SAT check uses side .02 + threshold .08 = .10\n"
        "non-trivial start also requires initial reward <= -3",
        ha="left", va="bottom", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.9),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="nav/runs/nav_protocol_map.png")
    args = parser.parse_args()

    fig, axes = plt.subplots(2, 2, figsize=(17, 16), constrained_layout=True)
    _training_panel(axes[0, 0])
    _validation_panel(axes[0, 1])
    _bmud_panel(axes[1, 0])
    _pusht_panel(axes[1, 1])
    fig.suptitle(
        "DIVO task geometry: supports, physical footprints, and rejection clearances",
        fontsize=17, fontweight="bold",
    )
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
