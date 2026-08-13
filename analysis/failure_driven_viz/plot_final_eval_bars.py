"""Plot final policy-only benchmark success rates with mean/std bars.

This script aggregates multiple seed-level ``benchmark_summary.json`` files
under a root directory such as:

    data/outputs/2026.05.02/final_eval_bestckpt/

Expected run directory names:

    manual_between_s0
    manual_between_s1
    manual_between_s2
    llm_static_s0
    llm_static_s1
    llm_static_s2
    llm_evolve_s0
    llm_evolve_s1
    llm_evolve_s2
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
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis.failure_driven_viz.style import (
    CARD_BG,
    MUTED_TEXT,
    TITLE_COLOR,
    configure_matplotlib,
    style_chart_axis,
)

configure_matplotlib()

BENCHMARK_ORDER = ["B", "M", "U", "D", "AVG"]
METHOD_SPECS = [
    ("manual_between", "Manual-Between", "#6b7280"),
    ("llm_static", "LLM-Static", "#3b82f6"),
    ("llm_evolve", "LLM-Evolve", "#ff5a5f"),
]


def _load_summary(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def collect_results(root: Path) -> Dict[str, Dict[str, List[float]]]:
    results: Dict[str, Dict[str, List[float]]] = {}
    for method_key, _, _ in METHOD_SPECS:
        method_runs = sorted(root.glob(f"{method_key}_s*/benchmark_summary.json"))
        if not method_runs:
            raise FileNotFoundError(f"No benchmark summaries found for {method_key} under {root}")

        bench_to_values = {bench: [] for bench in BENCHMARK_ORDER}
        for run_path in method_runs:
            summary = _load_summary(run_path)
            for bench in BENCHMARK_ORDER[:-1]:
                bench_to_values[bench].append(float(summary["benchmarks"][bench]["success_rate"]))
            bench_to_values["AVG"].append(float(summary["aggregate"]["avg_success_rate"]))
        results[method_key] = bench_to_values
    return results


def summarize_results(results: Dict[str, Dict[str, List[float]]]) -> Dict[str, Dict[str, Tuple[float, float]]]:
    summary: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for method_key, bench_to_values in results.items():
        summary[method_key] = {}
        for bench, values in bench_to_values.items():
            arr = np.asarray(values, dtype=float)
            summary[method_key][bench] = (float(arr.mean()), float(arr.std(ddof=0)))
    return summary


def write_csv(summary: Dict[str, Dict[str, Tuple[float, float]]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "benchmark", "mean_success_rate", "std_success_rate"])
        for method_key, _, _ in METHOD_SPECS:
            for bench in BENCHMARK_ORDER:
                mean, std = summary[method_key][bench]
                writer.writerow([method_key, bench, f"{mean:.6f}", f"{std:.6f}"])


def write_markdown(summary: Dict[str, Dict[str, Tuple[float, float]]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["Method"] + BENCHMARK_ORDER
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for method_key, label, _ in METHOD_SPECS:
        cells = [label]
        for bench in BENCHMARK_ORDER:
            mean, std = summary[method_key][bench]
            cells.append(f"{mean:.3f} +/- {std:.3f}")
        lines.append("| " + " | ".join(cells) + " |")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_grouped_bars(
    summary: Dict[str, Dict[str, Tuple[float, float]]],
    output_path: Path,
    title: str,
    subtitle: str | None = None,
) -> Path:
    x = np.arange(len(BENCHMARK_ORDER), dtype=float)
    width = 0.21

    fig, ax = plt.subplots(figsize=(10.6, 5.4), constrained_layout=False)
    fig.patch.set_facecolor("white")
    style_chart_axis(ax, facecolor=CARD_BG, grid_axis="y")
    ax.set_axisbelow(True)

    # Light region emphasis for the hardest benchmark and the aggregate column.
    ax.axvspan(2.5, 3.5, color="#fff4ea", alpha=0.45, zorder=0)
    ax.axvspan(3.5, 4.5, color="#f3f6fb", alpha=0.85, zorder=0)

    for idx, (method_key, label, color) in enumerate(METHOD_SPECS):
        means = [summary[method_key][bench][0] for bench in BENCHMARK_ORDER]
        stds = [summary[method_key][bench][1] for bench in BENCHMARK_ORDER]
        offset = (idx - 1) * width
        bars = ax.bar(
            x + offset,
            means,
            width=width,
            label=label,
            color=color,
            edgecolor="white",
            linewidth=1.1,
            yerr=stds,
            capsize=4,
            error_kw={"elinewidth": 1.2, "ecolor": "#334155"},
            zorder=3,
        )
        for bar, mean in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                mean + 0.018,
                f"{mean:.2f}",
                ha="center",
                va="bottom",
                fontsize=8.8,
                fontweight="semibold",
                color=TITLE_COLOR,
            )

    ax.set_xticks(x, BENCHMARK_ORDER)
    ax.set_ylabel("Success rate")
    ax.set_ylim(0.0, 1.05)

    ax.axvline(3.5, color="#cbd5e1", linewidth=1.2, linestyle="--", zorder=1)
    ax.text(1.5, 1.02, "Held-out obstacle families", ha="center", va="bottom", fontsize=9.2, color=MUTED_TEXT)
    ax.text(4.0, 1.02, "Overall average", ha="center", va="bottom", fontsize=9.2, color=MUTED_TEXT)

    fig.suptitle(title, x=0.5, y=0.972, ha="center", fontsize=16.5, fontweight="bold", color=TITLE_COLOR)
    if subtitle:
        fig.text(0.5, 0.892, subtitle, ha="center", va="bottom", fontsize=10.0, color=MUTED_TEXT)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.835),
        ncol=3,
        frameon=False,
        fontsize=11.3,
        columnspacing=1.8,
        handletextpad=0.6,
    )
    fig.text(
        0.08,
        0.055,
        "Bars show mean success rate across seeds; error bars show standard deviation.",
        ha="left",
        va="center",
        fontsize=8.8,
        color=MUTED_TEXT,
    )
    fig.subplots_adjust(left=0.08, right=0.985, top=0.66 if subtitle else 0.73, bottom=0.14)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot grouped final evaluation success-rate bars.")
    parser.add_argument("--root", required=True, help="Root directory containing *_s*/benchmark_summary.json files.")
    parser.add_argument("--output", required=True, help="Output PNG path.")
    parser.add_argument("--csv", default=None, help="Optional CSV summary output path.")
    parser.add_argument("--markdown", default=None, help="Optional Markdown table output path.")
    parser.add_argument("--title", default="Push-T Final Generalization Benchmarks", help="Figure title.")
    parser.add_argument("--subtitle", default="Policy-only evaluation with best seen-validation checkpoints; 3 seeds per method.", help="Optional subtitle.")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    summary = summarize_results(collect_results(root))

    output_path = Path(args.output).expanduser().resolve()
    saved_path = plot_grouped_bars(summary, output_path, args.title, args.subtitle)
    print(f"Saved figure to: {saved_path}")

    if args.csv:
        csv_path = Path(args.csv).expanduser().resolve()
        write_csv(summary, csv_path)
        print(f"Saved CSV summary to: {csv_path}")

    if args.markdown:
        md_path = Path(args.markdown).expanduser().resolve()
        write_markdown(summary, md_path)
        print(f"Saved Markdown table to: {md_path}")


if __name__ == "__main__":
    main()
