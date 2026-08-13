"""Plot side-by-side obstacle heatmap comparisons for two runs.

This comparison script aligns runs by training progress phases rather than
generator stage ids, so it works for both:

- failure-driven runs with many evolve stages
- static baselines with a single generator
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, PowerNorm
import numpy as np

from analysis.failure_driven_viz.parse_logs import (
    ensure_parsed_dir,
    load_parsed_artifacts,
)
from analysis.failure_driven_viz.plot_heatmap import (
    COORDINATE_FRAMES,
    HEATMAP_CMAP,
    build_transformed_obstacle_rows,
    coordinate_frame_description,
    coordinate_frame_anchor_style,
    coordinate_frame_axis_labels,
    coordinate_frame_title,
)
from analysis.failure_driven_viz.style import (
    CARD_BG,
    CARD_EDGE,
    MUTED_TEXT,
    PANEL_EDGE_LIGHT,
    TITLE_COLOR,
    add_badge,
    add_summary_card,
    configure_matplotlib,
    run_accent_color,
    style_scene_axis,
)
from analysis.failure_driven_viz.video_style import (
    VIDEO_TARGET_EDGE,
    VIDEO_TARGET_FILL,
    draw_video_background,
    draw_video_tblock,
    setup_video_axis,
    world_edges_to_video_edges,
)

configure_matplotlib()

PHASE_LABELS_3 = ["Early", "Middle", "Late"]
PHASE_LABELS_4 = ["Early", "Mid-Early", "Mid-Late", "Late"]


def plot_compare_heatmap(
    parsed_dir_a: Path,
    parsed_dir_b: Path,
    output_path: Path,
    label_a: Optional[str] = None,
    label_b: Optional[str] = None,
    num_phases: int = 3,
    bins: int = 25,
    xy_limit: float = 0.25,
    normalize: bool = True,
    coordinate_frame: str = "absolute",
) -> Path:
    artifacts_a = load_parsed_artifacts(str(parsed_dir_a))
    artifacts_b = load_parsed_artifacts(str(parsed_dir_b))

    comparison_a = _prepare_progress_heatmaps(
        artifacts=artifacts_a,
        label=label_a or _default_label(artifacts_a["run_meta"]),
        num_phases=num_phases,
        bins=bins,
        xy_limit=xy_limit,
        normalize=normalize,
        coordinate_frame=coordinate_frame,
    )
    comparison_b = _prepare_progress_heatmaps(
        artifacts=artifacts_b,
        label=label_b or _default_label(artifacts_b["run_meta"]),
        num_phases=num_phases,
        bins=bins,
        xy_limit=xy_limit,
        normalize=normalize,
        coordinate_frame=coordinate_frame,
    )

    vmax = max(comparison_a["vmax"], comparison_b["vmax"], 1e-8)
    norm = PowerNorm(gamma=0.55, vmin=0.0, vmax=vmax)

    fig = plt.figure(figsize=(4.55 * num_phases + 3.0, 7.4), constrained_layout=False)
    fig.patch.set_facecolor("#ffffff")
    grid = fig.add_gridspec(
        2,
        num_phases + 1,
        width_ratios=[1.08] + [1.0] * num_phases,
        wspace=0.16,
        hspace=0.18,
    )

    mesh = None
    heat_axes = []
    axis_xlabel, axis_ylabel = coordinate_frame_axis_labels(coordinate_frame)
    anchor_color, anchor_marker, anchor_size = coordinate_frame_anchor_style(coordinate_frame)
    for row_index, comparison in enumerate([comparison_a, comparison_b]):
        summary_axis = fig.add_subplot(grid[row_index, 0])
        _draw_summary_axis(summary_axis, comparison)
        for phase_index in range(num_phases):
            axis = fig.add_subplot(grid[row_index, phase_index + 1])
            heat_axes.append(axis)
            panel = comparison["panels"][phase_index]
            heatmap = panel["heatmap"]
            if coordinate_frame == "absolute":
                mesh = _draw_absolute_progress_panel(
                    axis=axis,
                    panel=panel,
                    xedges=comparison["xedges"],
                    yedges=comparison["yedges"],
                    norm=norm,
                )
            else:
                style_scene_axis(axis)
                axis.set_facecolor(CARD_BG)
                mesh = axis.pcolormesh(
                    comparison["xedges"],
                    comparison["yedges"],
                    heatmap.T,
                    cmap=HEATMAP_CMAP,
                    norm=norm,
                    shading="auto",
                )
                axis.scatter([0.0], [0.0], color=anchor_color, marker=anchor_marker, s=anchor_size - 2, linewidths=2.2, zorder=4)
                axis.axhline(0.0, color="white", linewidth=0.6, alpha=0.18)
                axis.axvline(0.0, color="white", linewidth=0.6, alpha=0.18)
                axis.set_xlim(-xy_limit, xy_limit)
                axis.set_ylim(-xy_limit, xy_limit)
                axis.set_aspect("equal")
                axis.grid(alpha=0.10, linestyle=":")
                axis.spines["top"].set_visible(False)
                axis.spines["right"].set_visible(False)
                axis.set_xlabel(axis_xlabel)
                axis.set_ylabel(axis_ylabel)
                add_badge(axis, 0.03, 0.97, panel["phase_label"], facecolor="#284a76", fontsize=8.8)
                add_badge(
                    axis,
                    0.97,
                    0.97,
                    f"obs {panel['obstacle_count']}",
                    facecolor="#ffffff",
                    edgecolor=CARD_EDGE,
                    textcolor=MUTED_TEXT,
                    ha="right",
                    fontsize=8.0,
                )
                axis.text(
                    0.03,
                    0.90,
                    panel["progress_label"],
                    transform=axis.transAxes,
                    ha="left",
                    va="top",
                    fontsize=8.2,
                    color=MUTED_TEXT,
                )

            if row_index == 0:
                axis.text(
                    0.5,
                    1.055,
                    panel["phase_label"],
                    transform=axis.transAxes,
                    ha="center",
                    va="bottom",
                    fontsize=12.5,
                    fontweight="bold",
                    color=TITLE_COLOR,
                )

    fig.suptitle(
        coordinate_frame_title(coordinate_frame).replace("Across Stages", "by Training Progress"),
        x=0.08,
        y=0.972,
        ha="left",
        fontsize=16,
        fontweight="bold",
        color=TITLE_COLOR,
    )
    if coordinate_frame != "absolute":
        fig.text(
            0.08,
            0.942,
            _comparison_subtitle(comparison_a, comparison_b, normalize, coordinate_frame),
            ha="left",
            va="bottom",
            fontsize=9,
            color=MUTED_TEXT,
        )
    fig.subplots_adjust(left=0.06, right=0.92, bottom=0.04, top=0.87, wspace=0.16, hspace=0.26)

    if mesh is not None:
        colorbar = fig.colorbar(mesh, ax=heat_axes, shrink=0.84, pad=0.015)
        colorbar.set_label("Normalized occupancy" if normalize else "Obstacle count")
        colorbar.outline.set_edgecolor(PANEL_EDGE_LIGHT)
        colorbar.outline.set_linewidth(1.0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two runs using progress-aligned obstacle heatmaps.")
    parser.add_argument("--run-a", default=None, help="First experiment output directory.")
    parser.add_argument("--run-b", default=None, help="Second experiment output directory.")
    parser.add_argument("--parsed-a", default=None, help="Existing parsed directory for run A.")
    parser.add_argument("--parsed-b", default=None, help="Existing parsed directory for run B.")
    parser.add_argument("--export-dir-a", default=None, help="Optional parsed export directory when using --run-a.")
    parser.add_argument("--export-dir-b", default=None, help="Optional parsed export directory when using --run-b.")
    parser.add_argument("--force-reparse", action="store_true", help="Regenerate parsed tables before plotting.")
    parser.add_argument("--label-a", default=None, help="Display label for run A.")
    parser.add_argument("--label-b", default=None, help="Display label for run B.")
    parser.add_argument("--num-phases", type=int, default=3, help="Number of progress phases to compare.")
    parser.add_argument("--bins", type=int, default=25, help="2D histogram bin count per axis.")
    parser.add_argument("--xy-limit", type=float, default=0.25, help="Plot range in both x/y directions.")
    parser.add_argument(
        "--coordinate-frame",
        choices=COORDINATE_FRAMES,
        default="absolute",
        help="How to express obstacle positions: absolute world coordinates, relative to start A, or aligned to A heading.",
    )
    parser.add_argument("--raw-counts", action="store_true", help="Disable per-panel normalization.")
    parser.add_argument("--output", required=True, help="Output PNG path.")
    args = parser.parse_args()

    parsed_dir_a = ensure_parsed_dir(
        run_dir=args.run_a,
        parsed_dir=args.parsed_a,
        export_dir=args.export_dir_a,
        force_reparse=args.force_reparse,
    )
    parsed_dir_b = ensure_parsed_dir(
        run_dir=args.run_b,
        parsed_dir=args.parsed_b,
        export_dir=args.export_dir_b,
        force_reparse=args.force_reparse,
    )

    output_path = Path(args.output).expanduser().resolve()
    saved_path = plot_compare_heatmap(
        parsed_dir_a=parsed_dir_a,
        parsed_dir_b=parsed_dir_b,
        output_path=output_path,
        label_a=args.label_a,
        label_b=args.label_b,
        num_phases=args.num_phases,
        bins=args.bins,
        xy_limit=args.xy_limit,
        normalize=not args.raw_counts,
        coordinate_frame=args.coordinate_frame,
    )
    print(f"Saved two-run heatmap comparison to: {saved_path}")


def _prepare_progress_heatmaps(
    artifacts: Dict[str, Any],
    label: str,
    num_phases: int,
    bins: int,
    xy_limit: float,
    normalize: bool,
    coordinate_frame: str,
) -> Dict[str, Any]:
    snapshot_rows = artifacts["layout_snapshots"]
    run_meta = artifacts["run_meta"]
    transformed_rows, skipped_points = build_transformed_obstacle_rows(
        obstacle_rows=artifacts["obstacle_points"],
        snapshot_rows=snapshot_rows,
        coordinate_frame=coordinate_frame,
    )

    if not snapshot_rows or not transformed_rows:
        raise ValueError(f"Parsed directory for {label} is missing snapshots or obstacle points.")

    snapshot_rows_sorted = sorted(snapshot_rows, key=lambda row: int(row["episode"]))
    max_episode = max(int(row["episode"]) for row in snapshot_rows_sorted)
    min_episode = min(int(row["episode"]) for row in snapshot_rows_sorted)
    episode_span = max(1, max_episode - min_episode)

    xedges = np.linspace(-xy_limit, xy_limit, bins + 1)
    yedges = np.linspace(-xy_limit, xy_limit, bins + 1)

    snapshot_to_phase: Dict[int, int] = {}
    panel_stats: List[Dict[str, Any]] = []
    phase_labels = _phase_labels(num_phases)
    vmax = 0.0

    for phase_index in range(num_phases):
        lower = phase_index / float(num_phases)
        upper = (phase_index + 1) / float(num_phases)
        selected_snapshot_ids = []
        selected_snapshot_episodes = []

        for row in snapshot_rows_sorted:
            episode = int(row["episode"])
            progress = (episode - min_episode) / float(episode_span)
            in_phase = (progress >= lower) and (progress < upper or (phase_index == num_phases - 1 and progress <= upper))
            if in_phase:
                snapshot_id = int(row["snapshot_id"])
                snapshot_to_phase[snapshot_id] = phase_index
                selected_snapshot_ids.append(snapshot_id)
                selected_snapshot_episodes.append(episode)

        phase_obstacles = [
            row
            for row in transformed_rows
            if int(row["snapshot_id"]) in snapshot_to_phase and snapshot_to_phase[int(row["snapshot_id"])] == phase_index
        ]
        xs = np.array([float(row["plot_x"]) for row in phase_obstacles], dtype=float)
        ys = np.array([float(row["plot_y"]) for row in phase_obstacles], dtype=float)
        heatmap, _, _ = np.histogram2d(xs, ys, bins=[xedges, yedges])
        if normalize and len(phase_obstacles) > 0:
            heatmap = heatmap / float(len(phase_obstacles))
        vmax = max(vmax, float(np.max(heatmap)) if heatmap.size else 0.0)

        phase_start = min(selected_snapshot_episodes) if selected_snapshot_episodes else None
        phase_end = max(selected_snapshot_episodes) if selected_snapshot_episodes else None

        panel_stats.append(
            {
                "phase_index": phase_index,
                "phase_label": phase_labels[phase_index],
                "progress_label": _progress_label(lower, upper, phase_start, phase_end),
                "snapshot_count": len(selected_snapshot_ids),
                "obstacle_count": len(phase_obstacles),
                "heatmap": heatmap,
            }
        )

    return {
        "label": label,
        "run_meta": run_meta,
        "xedges": xedges,
        "yedges": yedges,
        "panels": panel_stats,
        "vmax": vmax,
        "skipped_points": skipped_points,
    }


def _draw_summary_axis(axis: plt.Axes, comparison: Dict[str, Any]) -> None:
    run_meta = comparison["run_meta"]
    label = comparison["label"]
    run_name = Path(str(run_meta.get("output_dir", run_meta.get("run_id", label)))).name
    add_summary_card(
        axis,
        title=_run_badge(label),
        lines=[
            f"Run: {run_name}",
            f"Seen success rate: {_format_metric(run_meta.get('final_seen_success_rate'))}",
            f"Validate reward: {_format_metric(run_meta.get('final_validate_reward'))}",
            f"Evolve rounds: {_format_metric(run_meta.get('final_evolve_count'), integer=True)}",
            f"Skipped points: {int(comparison.get('skipped_points', 0))}",
        ],
        accent=run_accent_color(label),
        badge=run_name,
    )


def _draw_absolute_progress_panel(
    axis: plt.Axes,
    panel: Dict[str, Any],
    xedges: np.ndarray,
    yedges: np.ndarray,
    norm: Normalize,
):
    setup_video_axis(axis)
    draw_video_background(axis)
    draw_video_tblock(
        axis,
        center_world=(0.0, 0.0),
        theta_deg=-45.0,
        fill=VIDEO_TARGET_FILL,
        edge=VIDEO_TARGET_EDGE,
        zorder=2,
        alpha=0.72,
        linewidth=0.9,
    )
    masked = np.ma.masked_less_equal(panel["heatmap"].T, 0.0)
    pixel_x_edges, pixel_y_edges = world_edges_to_video_edges(xedges, yedges)
    image = axis.pcolormesh(
        pixel_x_edges,
        pixel_y_edges,
        masked,
        cmap=HEATMAP_CMAP,
        norm=norm,
        zorder=5,
        shading="auto",
    )
    add_badge(axis, 0.02, 0.97, f"snaps {panel['snapshot_count']}", facecolor="#284a76", fontsize=8.3)
    add_badge(
        axis,
        0.98,
        0.97,
        f"obs {panel['obstacle_count']}",
        facecolor="#f8fbff",
        edgecolor="#9fc1e8",
        textcolor="#42556f",
        ha="right",
        fontsize=8.0,
    )
    axis.text(
        0.03,
        0.11,
        panel["progress_label"],
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=8.4,
        color="#111827",
        clip_on=True,
    )
    return image


def _comparison_subtitle(
    comparison_a: Dict[str, Any],
    comparison_b: Dict[str, Any],
    normalize: bool,
    coordinate_frame: str,
) -> str:
    metric = "normalized occupancy" if normalize else "raw obstacle count"
    skipped_total = int(comparison_a.get("skipped_points", 0)) + int(comparison_b.get("skipped_points", 0))
    skipped_text = f" | skipped={skipped_total}" if skipped_total > 0 else ""
    return (
        f"{comparison_a['label']} vs {comparison_b['label']} | aligned by training progress | "
        f"frame={coordinate_frame} ({coordinate_frame_description(coordinate_frame)}) | metric={metric}{skipped_text}"
    )


def _default_label(run_meta: Dict[str, Any]) -> str:
    output_name = Path(str(run_meta.get("output_dir", run_meta.get("run_id", "run")))).name
    if run_meta.get("enable_acgs_loop"):
        return f"{output_name} (failure-driven)"
    return f"{output_name} (static)"


def _run_badge(label: str) -> str:
    lowered = label.lower()
    if "failure-driven" in lowered or "evolve" in lowered:
        return "Failure-driven curriculum"
    if "static" in lowered:
        return "Static baseline"
    return "Experiment run"


def _phase_labels(num_phases: int) -> List[str]:
    if num_phases == 3:
        return PHASE_LABELS_3
    if num_phases == 4:
        return PHASE_LABELS_4
    return [f"Phase {index + 1}" for index in range(num_phases)]


def _progress_label(lower: float, upper: float, phase_start: Optional[int], phase_end: Optional[int]) -> str:
    progress = f"{int(round(lower * 100))}-{int(round(upper * 100))}%"
    if phase_start is None or phase_end is None:
        return progress
    return f"{progress} | {_fmt_k(phase_start)}-{_fmt_k(phase_end)}"


def _fmt_k(value: object) -> str:
    if value is None:
        return "n/a"
    numeric = int(float(value))
    if abs(numeric) >= 1000:
        return f"{numeric // 1000}k"
    return str(numeric)


def _format_metric(value: object, integer: bool = False) -> str:
    if value is None:
        return "n/a"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if integer:
        return str(int(round(numeric)))
    return f"{numeric:.2f}"


if __name__ == "__main__":
    main()
