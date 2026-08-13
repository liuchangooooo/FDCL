"""Plot train-time and final benchmark summaries for multiple policy runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis.failure_driven_viz.style import (
    CARD_BG,
    CARD_EDGE,
    MUTED_TEXT,
    PANEL_EDGE_LIGHT,
    SUCCESS_LINE,
    TITLE_COLOR,
    add_badge,
    configure_matplotlib,
    style_chart_axis,
)

configure_matplotlib()

DEFAULT_BENCHMARK_ORDER = ["B", "M", "U", "D"]
PALETTE = [
    "#2f7cff",
    "#ff5f63",
    "#2ca58d",
    "#f1a84f",
    "#6f63ff",
]


def load_method(
    label: str,
    train_summary_path: Path,
    benchmark_summary_path: Path,
    include_seen: bool = False,
) -> Dict:
    with train_summary_path.open("r", encoding="utf-8") as handle:
        train_summary = json.load(handle)
    with benchmark_summary_path.open("r", encoding="utf-8") as handle:
        benchmark_summary = json.load(handle)

    benchmarks = benchmark_summary.get("benchmarks", {})
    benchmark_order = [name for name in DEFAULT_BENCHMARK_ORDER if name in benchmarks]
    for name in benchmarks:
        if name not in benchmark_order:
            benchmark_order.append(name)

    success_series: List[float] = []
    reward_series: List[float] = []

    if include_seen:
        success_series.append(float(train_summary.get("eval/success_rate[seen]", np.nan)))
        reward_series.append(float(train_summary.get("validate_reward", np.nan)))

    for benchmark_name in benchmark_order:
        benchmark_info = benchmarks[benchmark_name]
        success_series.append(float(benchmark_info.get("success_rate", np.nan)))
        reward_series.append(float(benchmark_info.get("mean_reward", np.nan)))

    aggregate = benchmark_summary.get("aggregate", {})
    success_series.append(float(aggregate.get("avg_success_rate", np.nan)))
    reward_series.append(float(aggregate.get("avg_mean_reward", np.nan)))

    return {
        "label": label,
        "train_summary_path": train_summary_path,
        "benchmark_summary_path": benchmark_summary_path,
        "benchmark_order": benchmark_order,
        "success_series": success_series,
        "reward_series": reward_series,
        "include_seen": include_seen,
    }


def plot_policy_eval_compare(
    methods: List[Dict],
    output_path: Path,
    title: str,
    subtitle: str | None = None,
) -> Path:
    if not methods:
        raise ValueError("At least one method is required.")

    benchmark_order = methods[0]["benchmark_order"]
    include_seen = bool(methods[0].get("include_seen", False))
    x_labels = (["Seen"] if include_seen else []) + benchmark_order + ["Avg"]
    x = np.arange(len(x_labels), dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.6), constrained_layout=False)
    ax_success, ax_reward = axes
    fig.patch.set_facecolor("white")

    for axis in axes:
        style_chart_axis(axis, facecolor=CARD_BG, grid_axis="y")

    for index, method in enumerate(methods):
        color = PALETTE[index % len(PALETTE)]
        ax_success.plot(
            x,
            method["success_series"],
            label=method["label"],
            color=color,
            linewidth=2.8,
            marker="o",
            markersize=6.8,
            markerfacecolor="white",
            markeredgewidth=2.0,
        )
        ax_reward.plot(
            x,
            method["reward_series"],
            label=method["label"],
            color=color,
            linewidth=2.8,
            marker="o",
            markersize=6.8,
            markerfacecolor="white",
            markeredgewidth=2.0,
        )
        _annotate_last_point(ax_success, x[-1], method["success_series"][-1], color)
        _annotate_last_point(ax_reward, x[-1], method["reward_series"][-1], color)

    ax_success.set_title("Success Rate", fontsize=14, fontweight="bold", pad=10, color=TITLE_COLOR)
    ax_success.set_ylabel("Rate")
    ax_success.set_xticks(x, x_labels)
    ax_success.set_ylim(0.0, 1.05)
    add_badge(ax_success, 0.02, 0.97, "policy transfer", facecolor="#284a76", fontsize=8.2)

    ax_reward.set_title("Reward", fontsize=14, fontweight="bold", pad=10, color=TITLE_COLOR)
    ax_reward.set_ylabel("Mean Reward")
    ax_reward.set_xticks(x, x_labels)
    reward_values = np.asarray([value for method in methods for value in method["reward_series"]], dtype=float)
    reward_min = float(np.nanmin(reward_values))
    reward_max = float(np.nanmax(reward_values))
    reward_pad = max(0.35, 0.08 * (reward_max - reward_min if reward_max > reward_min else 1.0))
    ax_reward.set_ylim(reward_min - reward_pad, reward_max + reward_pad)
    add_badge(ax_reward, 0.02, 0.97, "benchmark rollouts", facecolor="#284a76", fontsize=8.2)

    for axis in axes:
        for idx in range(len(x_labels)):
            if idx % 2 == 0:
                axis.axvspan(idx - 0.5, idx + 0.5, color="#edf5ff", alpha=0.45, zorder=0)

    fig.suptitle(title, x=0.5, y=0.97, ha="center", fontsize=17, fontweight="bold", color=TITLE_COLOR)
    if subtitle:
        fig.text(
            0.5,
            0.92,
            subtitle,
            ha="center",
            va="bottom",
            fontsize=10,
            color=MUTED_TEXT,
        )

    handles, labels = ax_success.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=max(1, len(methods)),
        frameon=False,
        fontsize=11.5,
        columnspacing=1.4,
        handletextpad=0.6,
    )

    top = 0.83 if subtitle else 0.84
    fig.subplots_adjust(left=0.07, right=0.985, top=top, bottom=0.19, wspace=0.17)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _annotate_last_point(ax: plt.Axes, x_value: float, y_value: float, color: str) -> None:
    ax.text(
        x_value + 0.06,
        y_value,
        f"{y_value:.3f}",
        fontsize=9.2,
        color=color,
        va="center",
        ha="left",
        fontweight="semibold",
    )


def _style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor(CARD_BG)
    ax.grid(axis="y", alpha=0.20, linestyle="--", linewidth=0.9)
    ax.grid(axis="x", alpha=0.08, linestyle="--", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(CARD_EDGE)
    ax.spines["bottom"].set_color(CARD_EDGE)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.tick_params(axis="both", labelsize=10.5, width=1.0, length=4.5, color=CARD_EDGE)
    ax.tick_params(axis="x", labelrotation=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot policy comparison figure from train and benchmark summaries.")
    parser.add_argument(
        "--method",
        nargs=3,
        action="append",
        metavar=("LABEL", "TRAIN_SUMMARY", "BENCHMARK_SUMMARY"),
        required=True,
        help="Method label, path to wandb-summary.json, and path to benchmark_summary.json.",
    )
    parser.add_argument("--title", default="Push-T Policy Comparison", help="Figure title.")
    parser.add_argument("--subtitle", default=None, help="Optional subtitle text.")
    parser.add_argument("--include-seen", action="store_true", help="Include train-time seen validation as the first x-axis point.")
    parser.add_argument("--output", required=True, help="Output PNG path.")
    args = parser.parse_args()

    methods = [
        load_method(
            label=label,
            train_summary_path=Path(train_summary).expanduser().resolve(),
            benchmark_summary_path=Path(benchmark_summary).expanduser().resolve(),
            include_seen=args.include_seen,
        )
        for label, train_summary, benchmark_summary in args.method
    ]

    output_path = Path(args.output).expanduser().resolve()
    saved_path = plot_policy_eval_compare(
        methods=methods,
        output_path=output_path,
        title=args.title,
        subtitle=args.subtitle,
    )
    print(f"Saved policy comparison plot to: {saved_path}")


if __name__ == "__main__":
    main()
