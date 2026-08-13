"""Build Figure 1 from the recorded Push-T rollout panels.

The figure intentionally makes only the claim supported by the recorded data:
the same deployment outcome can correspond to different fractions of successful
probe-skill executions.  It does not assign either scene to a curriculum band or
claim that one has greater future learning utility.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
PANEL_DIR = HERE.parents[1] / "figure_outputs" / "fig1_problem"
META_PATH = PANEL_DIR / "fig1_meta.json"


def main() -> None:
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    k = int(meta["K"])
    left_successes = sum(meta["mastered"]["succ"])
    right_successes = sum(meta["boundary"]["succ"])

    panel_names = (
        ("panel_col1_single.png", "panel_col2_single.png"),
        ("panel_col1_library.png", "panel_col2_library.png"),
    )

    fig, axes = plt.subplots(2, 2, figsize=(7.4, 7.2))
    for row in range(2):
        for col in range(2):
            axes[row, col].imshow(mpimg.imread(PANEL_DIR / panel_names[row][col]))
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            for spine in axes[row, col].spines.values():
                spine.set_edgecolor("#cccccc")

    for ax in axes[0]:
        ax.text(
            0.82,
            0.82,
            "\N{CHECK MARK}",
            transform=ax.transAxes,
            color="#2ca02c",
            fontsize=30,
            fontweight="bold",
            ha="center",
            va="center",
        )

    axes[0, 0].set_title("Layout 1", fontsize=13, fontweight="bold", pad=9)
    axes[0, 1].set_title("Layout 2", fontsize=13, fontweight="bold", pad=9)
    axes[1, 0].set_title(
        rf"${left_successes}/{k}$ probe skills succeed", fontsize=12, pad=8
    )
    axes[1, 1].set_title(
        rf"${right_successes}/{k}$ probe skills succeed", fontsize=12, pad=8
    )

    fig.text(
        0.025,
        0.70,
        "Deployment skill",
        rotation=90,
        fontsize=12,
        fontweight="bold",
        ha="center",
        va="center",
    )
    fig.text(
        0.025,
        0.29,
        rf"Probe library ($K={k}$)",
        rotation=90,
        fontsize=12,
        fontweight="bold",
        ha="center",
        va="center",
    )
    fig.suptitle(
        "Same deployment outcome, different skill-conditioned responses",
        fontsize=13,
        y=0.985,
    )
    fig.text(
        0.52,
        0.018,
        "Task-level success is identical; library-relative responses differ.",
        ha="center",
        fontsize=11,
        color="#333333",
    )
    fig.subplots_adjust(
        left=0.08, right=0.99, top=0.90, bottom=0.07, wspace=0.05, hspace=0.17
    )

    fig.savefig(HERE / "fig_problem.pdf", facecolor="white", bbox_inches="tight")
    fig.savefig(
        HERE / "fig_problem.png",
        dpi=220,
        facecolor="white",
        bbox_inches="tight",
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
