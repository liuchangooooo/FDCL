"""Draw Push-T manuscript figures directly from data tables.

This is the "clean" Nature-style figure path: it reads benchmark JSON and
parsed CSV tables, then draws each figure from scratch with matplotlib. It does
not call the existing analysis plotting scripts, so visual layout is decoupled
from the earlier dashboard-style figures.
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
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch
from matplotlib.transforms import Affine2D
import numpy as np

from DIVO.utils.failure_driven_parser import parse_run_directory


BENCHMARKS = ["B", "M", "U", "D"]
METHODS = [
    ("manual_between", "Manual", "#7A7F8C"),
    ("llm_static", "LLM-static", "#6C8CCF"),
    ("llm_evolve", "LLM-evolve", "#C85B70"),
]
PHASES = ["Early", "Middle", "Late"]
EXPORT_FORMATS = ("svg", "pdf", "tiff", "png")

COLORS = {
    "text": "#222222",
    "muted": "#6B6F7A",
    "axis": "#3C4250",
    "grid": "#E7EAF2",
    "paper": "#FFFFFF",
    "panel": "#F8F8FB",
    "blue": "#484878",
    "blue_mid": "#7884B4",
    "blue_soft": "#DDE3F4",
    "rose": "#C85B70",
    "rose_soft": "#F0D7DE",
    "green": "#2E9E44",
    "red": "#E53935",
    "orange": "#C77C2D",
}


def apply_nature_style() -> None:
    """Apply Nature-style defaults with editable SVG text."""

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "font.size": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "axes.edgecolor": COLORS["axis"],
            "axes.labelcolor": COLORS["text"],
            "xtick.color": COLORS["axis"],
            "ytick.color": COLORS["axis"],
            "legend.frameon": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def save_all(fig: plt.Figure, stem: Path) -> List[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    for suffix in EXPORT_FORMATS:
        path = stem.with_suffix(f".{suffix}")
        if suffix == "tiff":
            fig.savefig(path, dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
        elif suffix == "png":
            fig.savefig(path, dpi=300, bbox_inches="tight")
        else:
            fig.savefig(path, bbox_inches="tight")
        paths.append(path)
    return paths


def add_panel_label(ax: plt.Axes, label: str, x: float = -0.08, y: float = 1.06) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.5,
        fontweight="bold",
        color=COLORS["text"],
    )


def _coerce(value: str | None) -> Any:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    if text in ("True", "False"):
        return text == "True"
    try:
        if text.isdigit() or (text.startswith("-") and text[1:].isdigit()):
            return int(text)
        return float(text)
    except ValueError:
        return text


def load_csv_table(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [{key: _coerce(value) for key, value in row.items()} for row in reader]


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_parsed_tables(run_dir: Path) -> Path:
    parsed_dir = run_dir / "parsed"
    required = [
        "run_meta.json",
        "layout_snapshots.csv",
        "obstacle_points.csv",
        "batch_stats.csv",
        "evolve_rounds.csv",
    ]
    if parsed_dir.exists() and all((parsed_dir / name).exists() for name in required):
        return parsed_dir
    parsed = parse_run_directory(str(run_dir))
    return parsed.export(parsed_dir)


def collect_benchmark_summary(root: Path) -> Dict[str, Dict[str, Tuple[float, float, List[float]]]]:
    summary: Dict[str, Dict[str, Tuple[float, float, List[float]]]] = {}
    for method_key, _, _ in METHODS:
        values_by_bench: Dict[str, List[float]] = {bench: [] for bench in BENCHMARKS + ["AVG"]}
        for path in sorted(root.glob(f"{method_key}_s*/benchmark_summary.json")):
            payload = load_json(path)
            for bench in BENCHMARKS:
                values_by_bench[bench].append(float(payload["benchmarks"][bench]["success_rate"]))
            values_by_bench["AVG"].append(float(payload["aggregate"]["avg_success_rate"]))
        if not values_by_bench["AVG"]:
            raise FileNotFoundError(f"No benchmark summaries for {method_key} under {root}")
        summary[method_key] = {}
        for bench, values in values_by_bench.items():
            arr = np.asarray(values, dtype=float)
            summary[method_key][bench] = (float(arr.mean()), float(arr.std(ddof=0)), values)
    return summary


def draw_results_figure(final_eval_root: Path) -> plt.Figure:
    summary = collect_benchmark_summary(final_eval_root)
    fig = plt.figure(figsize=(7.2, 3.05), constrained_layout=False)
    gs = fig.add_gridspec(1, 2, width_ratios=[4.2, 1.05], wspace=0.20)
    ax = fig.add_subplot(gs[0, 0])
    ax_avg = fig.add_subplot(gs[0, 1])

    _style_numeric_axis(ax)
    _style_numeric_axis(ax_avg)
    add_panel_label(ax, "a", x=-0.075, y=1.08)
    ax.set_title("Unseen obstacle families", loc="left", fontsize=8.6, fontweight="bold", pad=7)
    ax_avg.set_title("Average", loc="left", fontsize=8.6, fontweight="bold", pad=7)

    x = np.arange(len(BENCHMARKS), dtype=float)
    width = 0.22
    for idx, (method_key, label, color) in enumerate(METHODS):
        offset = (idx - 1) * width
        means = [summary[method_key][bench][0] for bench in BENCHMARKS]
        stds = [summary[method_key][bench][1] for bench in BENCHMARKS]
        bars = ax.bar(
            x + offset,
            means,
            width=width,
            yerr=stds,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            capsize=2.5,
            error_kw={"elinewidth": 0.8, "ecolor": COLORS["axis"]},
            label=label,
            zorder=3,
        )
        for bar, mean in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                min(1.03, mean + 0.035),
                f"{mean:.2f}",
                ha="center",
                va="bottom",
                fontsize=6.4,
                color=COLORS["text"],
            )

        avg_mean, avg_std, _ = summary[method_key]["AVG"]
        ax_avg.bar(
            [idx],
            [avg_mean],
            yerr=[avg_std],
            color=color,
            edgecolor="white",
            linewidth=0.6,
            capsize=2.5,
            error_kw={"elinewidth": 0.8, "ecolor": COLORS["axis"]},
            zorder=3,
        )
        ax_avg.text(idx, min(1.03, avg_mean + 0.035), f"{avg_mean:.2f}", ha="center", va="bottom", fontsize=6.4)

    ax.set_xticks(x, BENCHMARKS)
    ax.set_ylabel("Success rate")
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax_avg.set_ylim(0, 1.05)
    ax_avg.set_yticks(np.linspace(0, 1, 6))
    ax_avg.set_xticks(np.arange(len(METHODS)), [label for _, label, _ in METHODS], rotation=38, ha="right")
    ax_avg.set_ylabel("")
    ax_avg.tick_params(axis="y", labelleft=False)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.56, 0.98), ncol=3, handlelength=1.6, columnspacing=1.2)
    fig.text(
        0.06,
        0.98,
        "Push-T zero-shot generalization",
        ha="left",
        va="top",
        fontsize=10.0,
        fontweight="bold",
        color=COLORS["text"],
    )
    fig.text(
        0.06,
        0.075,
        "Bars show mean across 3 seeds; whiskers show s.d.",
        ha="left",
        va="center",
        fontsize=6.8,
        color=COLORS["muted"],
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.80, bottom=0.24)
    return fig


def _style_numeric_axis(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.55, zorder=0)
    ax.spines["left"].set_color(COLORS["axis"])
    ax.spines["bottom"].set_color(COLORS["axis"])
    ax.tick_params(axis="both", length=2.6, width=0.8)


def load_parsed_run(run_dir: Path) -> Dict[str, Any]:
    parsed_dir = ensure_parsed_tables(run_dir)
    return {
        "parsed_dir": parsed_dir,
        "run_meta": load_json(parsed_dir / "run_meta.json"),
        "snapshots": load_csv_table(parsed_dir / "layout_snapshots.csv"),
        "obstacles": load_csv_table(parsed_dir / "obstacle_points.csv"),
        "batches": load_csv_table(parsed_dir / "batch_stats.csv"),
        "evolves": load_csv_table(parsed_dir / "evolve_rounds.csv"),
    }


def compute_phase_heatmaps(
    artifacts: Dict[str, Any],
    *,
    bins: int,
    limit: float,
    num_phases: int = 3,
) -> Tuple[List[np.ndarray], List[int], np.ndarray, np.ndarray]:
    snapshots = artifacts["snapshots"]
    obstacles = artifacts["obstacles"]
    snapshot_by_id = {int(row["snapshot_id"]): row for row in snapshots}
    episodes = [int(row["episode"]) for row in snapshots]
    min_ep, max_ep = min(episodes), max(episodes)
    span = max(1, max_ep - min_ep)
    xedges = np.linspace(-limit, limit, bins + 1)
    yedges = np.linspace(-limit, limit, bins + 1)
    points_by_phase: List[List[Tuple[float, float]]] = [[] for _ in range(num_phases)]

    for obstacle in obstacles:
        snapshot = snapshot_by_id.get(int(obstacle["snapshot_id"]))
        if snapshot is None:
            continue
        progress = (int(snapshot["episode"]) - min_ep) / span
        phase = min(num_phases - 1, int(progress * num_phases))
        points_by_phase[phase].append((float(obstacle["obs_x"]), float(obstacle["obs_y"])))

    heatmaps: List[np.ndarray] = []
    counts: List[int] = []
    for points in points_by_phase:
        counts.append(len(points))
        if points:
            arr = np.asarray(points)
            heatmap, _, _ = np.histogram2d(arr[:, 0], arr[:, 1], bins=[xedges, yedges])
            heatmap = heatmap / max(1.0, heatmap.sum())
        else:
            heatmap = np.zeros((bins, bins), dtype=float)
        heatmaps.append(heatmap.T)
    return heatmaps, counts, xedges, yedges


def draw_distribution_figure(static_run: Path, evolve_run: Path) -> plt.Figure:
    static = load_parsed_run(static_run)
    evolve = load_parsed_run(evolve_run)
    limit = 0.25
    bins = 38
    static_h, static_n, xedges, yedges = compute_phase_heatmaps(static, bins=bins, limit=limit)
    evolve_h, evolve_n, _, _ = compute_phase_heatmaps(evolve, bins=bins, limit=limit)
    vmax = max([h.max() for h in static_h + evolve_h] + [1e-8])
    cmap = LinearSegmentedColormap.from_list("nature_obstacle_density", ["#FFFFFF", "#F4CDD4", "#C85B70"])

    fig = plt.figure(figsize=(7.2, 4.0), constrained_layout=False)
    gs = fig.add_gridspec(2, 4, width_ratios=[0.58, 1, 1, 1], wspace=0.15, hspace=0.16)
    label_axes = [fig.add_subplot(gs[row, 0]) for row in range(2)]
    axes = [[fig.add_subplot(gs[row, col + 1]) for col in range(3)] for row in range(2)]

    for ax, title, subtitle, color in [
        (label_axes[0], "Static generator", "one fixed program", COLORS["blue_mid"]),
        (label_axes[1], "Failure-conditioned\nrevision", "38 deployed revisions", COLORS["rose"]),
    ]:
        ax.set_axis_off()
        ax.text(0.00, 0.63, title, ha="left", va="center", fontsize=8.6, fontweight="bold", color=COLORS["text"], linespacing=1.1)
        ax.text(0.00, 0.43, subtitle, ha="left", va="center", fontsize=6.8, color=COLORS["muted"])
        ax.plot([0.0, 0.0], [0.18, 0.86], color=color, linewidth=2.2, solid_capstyle="round")

    images = []
    for row, (heatmaps, counts) in enumerate([(static_h, static_n), (evolve_h, evolve_n)]):
        for col, (heatmap, count) in enumerate(zip(heatmaps, counts)):
            ax = axes[row][col]
            image = ax.imshow(
                heatmap,
                origin="lower",
                extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                cmap=cmap,
                vmin=0,
                vmax=vmax,
                interpolation="nearest",
            )
            images.append(image)
            _style_distribution_axis(ax, limit)
            _draw_target_t(ax, 0.0, 0.0, scale=0.085, alpha=0.35)
            ax.text(0.02, 0.94, f"n={count}", transform=ax.transAxes, ha="left", va="top", fontsize=6.3, color=COLORS["muted"])
            if row == 0:
                ax.set_title(PHASES[col], fontsize=8.0, fontweight="bold", pad=5)
            if col == 0:
                ax.set_ylabel("y")
            else:
                ax.set_yticklabels([])
            if row == 1:
                ax.set_xlabel("x")
            else:
                ax.set_xticklabels([])

    add_panel_label(axes[0][0], "a", x=-0.22, y=1.20)
    fig.text(0.055, 0.965, "Generator revision changes the training obstacle distribution", ha="left", va="top", fontsize=9.8, fontweight="bold")
    cax = fig.add_axes([0.92, 0.20, 0.012, 0.58])
    cb = fig.colorbar(images[0], cax=cax)
    cb.set_label("Normalized occupancy", fontsize=6.8)
    cb.ax.tick_params(labelsize=6.2, length=2)
    fig.subplots_adjust(left=0.06, right=0.90, top=0.86, bottom=0.13)
    return fig


def _style_distribution_axis(ax: plt.Axes, limit: float) -> None:
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal")
    ax.set_xticks([-0.2, 0.0, 0.2])
    ax.set_yticks([-0.2, 0.0, 0.2])
    ax.tick_params(length=2.0, width=0.6, labelsize=6.2)
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)
        spine.set_color("#C7CBD8")


def _draw_target_t(ax: plt.Axes, x: float, y: float, *, scale: float, alpha: float = 0.4) -> None:
    trans = Affine2D().rotate_deg_around(x, y, -45) + ax.transData
    fill = "#B9E9AE"
    edge = "#7BCB76"
    rects = [
        patches.Rectangle((x - 0.010, y - 0.040), 0.020, 0.080, transform=trans, facecolor=fill, edgecolor=edge, linewidth=0.6, alpha=alpha),
        patches.Rectangle((x - 0.045, y + 0.020), 0.090, 0.020, transform=trans, facecolor=fill, edgecolor=edge, linewidth=0.6, alpha=alpha),
    ]
    for rect in rects:
        ax.add_patch(rect)


def draw_failure_chain_figure(evolve_run: Path) -> plt.Figure:
    artifacts = load_parsed_run(evolve_run)
    evolves = artifacts["evolves"]
    batches = artifacts["batches"]
    obstacles = artifacts["obstacles"]
    selected_indices = _select_evolve_indices(evolves, max_count=3)
    selected = [evolves[idx] for idx in selected_indices]

    fig = plt.figure(figsize=(7.2, 5.2), constrained_layout=False)
    gs = fig.add_gridspec(len(selected), 5, width_ratios=[1.35, 1.0, 1.0, 1.05, 0.12], hspace=0.36, wspace=0.22)
    limit = 0.25
    bins = 30
    cmap = LinearSegmentedColormap.from_list("revision_density", ["#FFFFFF", "#F3CBD2", "#C85B70"])

    max_density = 1e-8
    cached_heatmaps: Dict[Tuple[int, int], np.ndarray] = {}
    for row in selected:
        before_stage = int(row["evolve_count_before"])
        after_stage = int(row["evolve_id"])
        for stage in [before_stage, after_stage]:
            h = _stage_heatmap(obstacles, stage, bins=bins, limit=limit)
            cached_heatmaps[(int(row["evolve_round_index"]), stage)] = h
            max_density = max(max_density, float(h.max()))

    for ridx, row in enumerate(selected):
        ax_diag = fig.add_subplot(gs[ridx, 0])
        ax_before = fig.add_subplot(gs[ridx, 1])
        ax_after = fig.add_subplot(gs[ridx, 2])
        ax_val = fig.add_subplot(gs[ridx, 3])
        _draw_diagnosis_card_clean(ax_diag, row)

        before_stage = int(row["evolve_count_before"])
        after_stage = int(row["evolve_id"])
        for ax, stage, title, color in [
            (ax_before, before_stage, f"before G{before_stage}", COLORS["blue_mid"]),
            (ax_after, after_stage, f"after G{after_stage}", COLORS["rose"]),
        ]:
            heatmap = cached_heatmaps[(int(row["evolve_round_index"]), stage)]
            ax.imshow(heatmap, origin="lower", extent=[-limit, limit, -limit, limit], cmap=cmap, vmin=0, vmax=max_density)
            _style_distribution_axis(ax, limit)
            _draw_target_t(ax, 0.0, 0.0, scale=0.080, alpha=0.32)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.text(0.02, 0.94, title, transform=ax.transAxes, ha="left", va="top", fontsize=6.6, fontweight="bold", color=color)
        _draw_validation_delta(ax_val, row, batches)

    add_panel_label(fig.axes[0], "a", x=-0.12, y=1.18)
    fig.text(0.055, 0.965, "Failure evidence is linked to generator revisions", ha="left", va="top", fontsize=9.8, fontweight="bold")
    fig.text(0.055, 0.925, "Each row shows one deployed revision: diagnosed failure, obstacle distribution before/after, and subsequent batch change.", ha="left", va="top", fontsize=6.9, color=COLORS["muted"])
    fig.subplots_adjust(left=0.06, right=0.985, top=0.86, bottom=0.08)
    return fig


def _select_evolve_indices(evolves: Sequence[Dict[str, Any]], max_count: int) -> List[int]:
    if len(evolves) <= max_count:
        return list(range(len(evolves)))
    raw = [0, len(evolves) // 2, len(evolves) - 1]
    return sorted(set(raw))[:max_count]


def _stage_heatmap(obstacles: List[Dict[str, Any]], stage_id: int, *, bins: int, limit: float) -> np.ndarray:
    pts = [(float(row["obs_x"]), float(row["obs_y"])) for row in obstacles if row.get("stage_id") == stage_id]
    if not pts:
        return np.zeros((bins, bins), dtype=float)
    arr = np.asarray(pts, dtype=float)
    edges = np.linspace(-limit, limit, bins + 1)
    heatmap, _, _ = np.histogram2d(arr[:, 0], arr[:, 1], bins=[edges, edges])
    return (heatmap / max(1.0, heatmap.sum())).T


def _draw_diagnosis_card_clean(ax: plt.Axes, row: Dict[str, Any]) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    failure = str(row.get("dominant_failure_type") or "unknown")
    color = {"timeout": COLORS["blue"], "collision": COLORS["red"], "fall": COLORS["orange"]}.get(failure, COLORS["axis"])
    ax.add_patch(
        patches.FancyBboxPatch(
            (0.02, 0.08),
            0.94,
            0.82,
            boxstyle="round,pad=0.018,rounding_size=0.04",
            facecolor="#FFFFFF",
            edgecolor="#D8DCE8",
            linewidth=0.9,
        )
    )
    ax.add_patch(patches.Rectangle((0.02, 0.08), 0.020, 0.82, facecolor=color, edgecolor="none"))
    title = f"E{int(row['evolve_id'])}: {failure}"
    lines = [
        f"episode {int(row['episode_total'])//1000}k",
        f"region: {row.get('failure_region') or 'n/a'}",
        f"bias: {row.get('behavior_bias') or 'n/a'}",
        f"samples: {row.get('sample_count') or 'n/a'}",
    ]
    ax.text(0.09, 0.76, title, ha="left", va="center", fontsize=7.6, fontweight="bold", color=COLORS["text"])
    ax.text(0.09, 0.43, "\n".join(lines), ha="left", va="center", fontsize=6.7, color=COLORS["muted"], linespacing=1.35)


def _draw_validation_delta(ax: plt.Axes, row: Dict[str, Any], batches: List[Dict[str, Any]]) -> None:
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(False)
    linked = int(row.get("linked_batch_idx") or 0)
    pre = next((b for b in batches if int(b["batch_idx"]) == linked), None)
    post_rows = [b for b in batches if linked < int(b["batch_idx"]) <= linked + 2]
    if pre is None or not post_rows:
        ax.set_axis_off()
        return

    metrics = [
        ("success", "success_rate", COLORS["green"], True),
        ("collision", "collision_rate", COLORS["red"], False),
        ("timeout", "timeout_rate", COLORS["blue"], False),
    ]
    y = np.arange(len(metrics))[::-1]
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.6, len(metrics) - 0.4)
    ax.set_yticks(y, [label for label, *_ in metrics])
    ax.set_xticks([0, 0.5, 1.0], ["0", "0.5", "1"])
    ax.tick_params(length=2, width=0.6, labelsize=6.2)
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.5)
    ax.set_title("post-revision", loc="left", fontsize=7.2, fontweight="bold", pad=3)
    for yi, (label, key, color, higher_is_better) in zip(y, metrics):
        pre_val = float(pre.get(key) or 0.0)
        post_val = float(np.mean([float(b.get(key) or 0.0) for b in post_rows]))
        ax.plot([pre_val, post_val], [yi, yi], color="#B8BECC", lw=1.2, zorder=1)
        ax.scatter([pre_val], [yi], s=22, facecolor="white", edgecolor=COLORS["axis"], linewidth=0.8, zorder=2)
        ax.scatter([post_val], [yi], s=24, facecolor=color, edgecolor="white", linewidth=0.5, zorder=3)
        delta = post_val - pre_val
        good = delta > 0 if higher_is_better else delta < 0
        delta_color = COLORS["green"] if good else COLORS["red"]
        ax.text(1.02, yi, f"{delta:+.2f}", ha="left", va="center", fontsize=6.3, color=delta_color, fontweight="bold")


def draw_method_schematic() -> plt.Figure:
    fig = plt.figure(figsize=(7.2, 3.8), constrained_layout=False)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0.04, 0.31, 0.92, 0.55])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    boxes = [
        (0.02, "Policy\nrollouts", "sampled layouts\nwith current policy", COLORS["blue_soft"], COLORS["blue_mid"]),
        (0.27, "Failure\ndiagnosis", "success, collision,\ntimeout, region", "#F0F0F7", COLORS["blue"]),
        (0.52, "Generator\nrevision", "constrained LLM\nprogram edit", COLORS["rose_soft"], COLORS["rose"]),
        (0.77, "New training\ndistribution", "feasible and\nfailure-relevant", "#F8E8EC", COLORS["rose"]),
    ]
    for idx, (x, title, body, face, edge) in enumerate(boxes):
        ax.add_patch(
            patches.FancyBboxPatch(
                (x, 0.20),
                0.19,
                0.58,
                boxstyle="round,pad=0.018,rounding_size=0.035",
                facecolor=face,
                edgecolor=edge,
                linewidth=1.0,
            )
        )
        ax.text(x + 0.025, 0.63, title, ha="left", va="center", fontsize=8.0, fontweight="bold", color=COLORS["text"], linespacing=1.05)
        ax.text(x + 0.025, 0.39, body, ha="left", va="center", fontsize=6.9, color=COLORS["muted"], linespacing=1.20)
        _draw_icon(ax, idx, x + 0.098, 0.27)
        if idx < len(boxes) - 1:
            ax.add_patch(
                FancyArrowPatch(
                    (x + 0.205, 0.49),
                    (boxes[idx + 1][0] - 0.015, 0.49),
                    arrowstyle="-|>",
                    mutation_scale=9,
                    linewidth=0.9,
                    color=COLORS["axis"],
                )
            )

    ax.add_patch(
        FancyArrowPatch(
            (0.84, 0.15),
            (0.115, 0.15),
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=1.0,
            color=COLORS["rose"],
            connectionstyle="arc3,rad=-0.22",
        )
    )
    ax.text(0.50, 0.035, "closed-loop training-environment design", ha="center", va="center", fontsize=7.2, color=COLORS["rose"], fontweight="bold")

    ax_obj = fig.add_axes([0.08, 0.08, 0.32, 0.16])
    ax_eval = fig.add_axes([0.48, 0.08, 0.44, 0.16])
    _draw_revision_objective(ax_obj)
    _draw_bmud_protocol(ax_eval)

    fig.text(0.045, 0.955, "a", ha="left", va="top", fontsize=10, fontweight="bold")
    fig.text(0.085, 0.955, "Failure-conditioned training-environment generator revision", ha="left", va="top", fontsize=10.2, fontweight="bold")
    fig.text(0.085, 0.905, "The policy is fixed as the learner; failure feedback revises the obstacle generator that supplies future training environments.", ha="left", va="top", fontsize=6.9, color=COLORS["muted"])
    return fig


def _draw_icon(ax: plt.Axes, idx: int, x: float, y: float) -> None:
    if idx == 0:
        ax.plot([x - 0.045, x, x + 0.045], [y, y + 0.045, y + 0.020], color=COLORS["blue"], lw=1.0)
        ax.scatter([x - 0.045], [y], s=14, color=COLORS["green"])
        ax.scatter([x + 0.045], [y + 0.020], s=28, marker="*", color=COLORS["rose"])
    elif idx == 1:
        heights = [0.04, 0.075, 0.03]
        for j, h in enumerate(heights):
            ax.add_patch(patches.Rectangle((x - 0.045 + 0.035 * j, y - 0.02), 0.024, h, facecolor=[COLORS["blue_mid"], COLORS["rose_soft"], "#D8D8D8"][j], edgecolor="white", lw=0.4))
    elif idx == 2:
        ax.text(x - 0.052, y + 0.015, "$g_t$", fontsize=7.5, color=COLORS["blue"], fontweight="bold")
        ax.text(x - 0.005, y + 0.015, "->", fontsize=7, color=COLORS["muted"])
        ax.text(x + 0.028, y + 0.015, "$g_{t+1}$", fontsize=7.5, color=COLORS["rose"], fontweight="bold")
        for k in range(3):
            ax.plot([x - 0.045, x + 0.055], [y - 0.012 - 0.017 * k, y - 0.012 - 0.017 * k], color="#B8B8B8", lw=0.6)
    else:
        ax.scatter([x - 0.040, x + 0.010, x + 0.050], [y + 0.030, y - 0.005, y + 0.045], s=[50, 65, 55], color=[COLORS["rose_soft"], COLORS["blue_soft"], "#EDA0A9"], edgecolor="none")
        ax.add_patch(patches.Rectangle((x - 0.010, y - 0.050), 0.030, 0.030, facecolor=COLORS["rose"], edgecolor="none"))


def _draw_revision_objective(ax: plt.Axes) -> None:
    ax.set_axis_off()
    ax.text(0.0, 0.93, "b", ha="left", va="top", fontsize=9.5, fontweight="bold")
    ax.text(0.12, 0.83, "Revision target", ha="left", va="center", fontsize=7.6, fontweight="bold")
    terms = [("failure-relevant", COLORS["blue"]), ("feasible", COLORS["green"]), ("non-trivial", COLORS["rose"])]
    for i, (term, color) in enumerate(terms):
        ax.text(0.16, 0.56 - 0.20 * i, "+", color=color, fontsize=9, fontweight="bold", ha="center", va="center")
        ax.text(0.25, 0.56 - 0.20 * i, term, fontsize=6.8, color=COLORS["text"], ha="left", va="center")


def _draw_bmud_protocol(ax: plt.Axes) -> None:
    ax.set_axis_off()
    ax.text(0.0, 0.93, "c", ha="left", va="top", fontsize=9.5, fontweight="bold")
    ax.text(0.10, 0.83, "Unseen obstacle families", ha="left", va="center", fontsize=7.6, fontweight="bold")
    for i, label in enumerate(["B", "M", "U", "D"]):
        x = 0.17 + 0.18 * i
        ax.add_patch(patches.Circle((x, 0.42), 0.058, facecolor="white", edgecolor=COLORS["blue_mid"], linewidth=0.9))
        ax.text(x, 0.42, label, ha="center", va="center", fontsize=7.6, color=COLORS["text"])
    ax.text(0.10, 0.13, "big, multiple, non-convex and dynamic obstacles", ha="left", va="center", fontsize=6.4, color=COLORS["muted"])


def write_source_summary(final_eval_root: Path, output_dir: Path) -> None:
    summary = collect_benchmark_summary(final_eval_root)
    rows = [["method", "benchmark", "mean", "std", "seed_values"]]
    for method_key, _, _ in METHODS:
        for bench in BENCHMARKS + ["AVG"]:
            mean, std, values = summary[method_key][bench]
            rows.append([method_key, bench, f"{mean:.6f}", f"{std:.6f}", ";".join(f"{v:.6f}" for v in values)])
    with (output_dir / "source_push_t_success_clean.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)


def write_manifest(output_dir: Path, manifest: Dict[str, List[str]]) -> None:
    contract = {
        "backend": "Python/matplotlib",
        "style": "nature-figure clean redraw; no legacy plotting functions",
        "figures": manifest,
        "notes": [
            "Figures are redrawn from JSON/CSV data tables.",
            "SVG text remains editable via svg.fonttype=none.",
            "Failure-chain revisions are deployed revisions, not learning-value accepted revisions.",
        ],
    }
    (output_dir / "manifest_clean.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export clean Nature-style Push-T manuscript figures.")
    parser.add_argument("--output-dir", default="data/outputs/paper_figures/nature_push_t_clean")
    parser.add_argument("--final-eval-root", default="data/outputs/2026.05.02/final_eval_bestckpt")
    parser.add_argument("--static-run", default="data/outputs/2026.04.29/llm_static_s0")
    parser.add_argument("--evolve-run", default="data/outputs/2026.04.29/llm_evolve_s0")
    args = parser.parse_args()

    apply_nature_style()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    final_eval_root = Path(args.final_eval_root).expanduser().resolve()
    static_run = Path(args.static_run).expanduser().resolve()
    evolve_run = Path(args.evolve_run).expanduser().resolve()

    manifest: Dict[str, List[str]] = {}
    figures = [
        ("fig1_push_t_results_bars", draw_results_figure(final_eval_root)),
        ("fig2_generator_distribution_heatmap", draw_distribution_figure(static_run, evolve_run)),
        ("fig3_failure_revision_chain", draw_failure_chain_figure(evolve_run)),
        ("fig4_method_schematic", draw_method_schematic()),
    ]
    for name, fig in figures:
        paths = save_all(fig, output_dir / name)
        manifest[name] = [str(path) for path in paths]
        plt.close(fig)

    write_source_summary(final_eval_root, output_dir)
    write_manifest(output_dir, manifest)
    print(f"Saved clean Nature-style figures to: {output_dir}")
    for name, paths in manifest.items():
        print(f"{name}: {paths[0]}")


if __name__ == "__main__":
    main()
