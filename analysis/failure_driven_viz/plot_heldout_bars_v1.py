"""Plot held-out generalization success-rate bars from the CAA V1 Table 1.

Single deterministic values per (method, benchmark), so no error bars are
drawn. Chinese method names rendered via Noto Sans CJK SC. Only the four
held-out families B/M/U/D are shown (no overall-average column), no title.

Run (use the divo conda env, which has a working matplotlib):
    ~/anaconda3/envs/divo/bin/python \
        analysis/failure_driven_viz/plot_heldout_bars_v1.py \
        --output figure_outputs/fig_heldout_bars.png
"""

from __future__ import annotations

if __name__ == "__main__":
    import os
    import pathlib
    import sys

    ROOT_DIR = str(pathlib.Path(__file__).resolve().parents[2])
    if ROOT_DIR not in sys.path:
        sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

# --- Chinese font setup (Noto Sans CJK SC) ---------------------------------
_CJK_FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
try:
    fm.fontManager.addfont(_CJK_FONT_PATH)
    _CJK_NAME = fm.FontProperties(fname=_CJK_FONT_PATH).get_name()
except Exception:
    _CJK_NAME = "Noto Sans CJK SC"
plt.rcParams["font.sans-serif"] = [_CJK_NAME, "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 12

# --- palette (user-specified RGB, in method order) -------------------------
C_MANUAL = (233 / 255, 211 / 255, 137 / 255)  # 手工设定 (yellow)
C_STATIC = (167 / 255, 211 / 255, 152 / 255)  # 静态生成器 (green)
C_OURS = (116 / 255, 163 / 255, 212 / 255)    # 本文方法 (blue)

TITLE_COLOR = "#1f2937"
MUTED_TEXT = "#5b6470"
CARD_BG = "#ffffff"

BENCHMARK_ORDER = ["B", "M", "U", "D"]
# x-axis display labels (full name + abbreviation)
BENCHMARK_LABELS = {
    "B": "Big (B)",
    "M": "Multiple (M)",
    "U": "U-shape (U)",
    "D": "Dynamics (D)",
}

# CAA long-abstract V1, Table 1 (held-out success rate).
TABLE1 = {
    "manual_between": {"B": 0.417, "M": 0.300, "U": 0.533, "D": 0.100},
    "llm_static":     {"B": 0.530, "M": 0.650, "U": 0.810, "D": 0.180},
    "ours":           {"B": 0.600, "M": 0.750, "U": 0.900, "D": 0.410},
}

METHOD_SPECS = [
    ("manual_between", "手工设定", C_MANUAL),
    ("llm_static", "静态生成器", C_STATIC),
    ("ours", "本文方法", C_OURS),
]


def plot_grouped_bars(output_path: Path) -> Path:
    x = np.arange(len(BENCHMARK_ORDER), dtype=float)
    width = 0.26

    fig, ax = plt.subplots(figsize=(8.4, 5.0), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor(CARD_BG)

    # clean axis: keep left + bottom spines, light y grid only
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#cbd5e1")
    ax.tick_params(colors=MUTED_TEXT, length=0)
    ax.grid(axis="y", color="#e5e9f0", linewidth=1.0)
    ax.set_axisbelow(True)

    for idx, (method_key, label, color) in enumerate(METHOD_SPECS):
        means = [TABLE1[method_key][bench] for bench in BENCHMARK_ORDER]
        offset = (idx - 1) * width
        bars = ax.bar(
            x + offset,
            means,
            width=width,
            label=label,
            color=color,
            edgecolor="white",
            linewidth=1.2,
            zorder=3,
        )
        for bar, mean in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                mean + 0.015,
                f"{mean:.2f}",
                ha="center",
                va="bottom",
                fontsize=9.5,
                fontweight="bold",
                color=TITLE_COLOR,
            )

    ax.set_xticks(x, [BENCHMARK_LABELS[b] for b in BENCHMARK_ORDER])
    ax.set_ylabel("成功率", color=TITLE_COLOR, fontsize=12.5)
    ax.set_ylim(0.0, 1.0)

    ax.legend(
        loc="upper right",
        ncol=1,
        frameon=False,
        fontsize=12.5,
        labelspacing=0.6,
        handletextpad=0.6,
        borderaxespad=0.8,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot CAA V1 Table 1 held-out success-rate bars.")
    parser.add_argument("--output", default="figure_outputs/fig_heldout_bars.png", help="Output PNG path.")
    args = parser.parse_args()
    saved = plot_grouped_bars(Path(args.output).expanduser().resolve())
    print(f"Saved figure to: {saved}")


if __name__ == "__main__":
    main()
