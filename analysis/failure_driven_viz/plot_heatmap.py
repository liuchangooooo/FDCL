"""Plot obstacle-position heatmaps for selected generator stages."""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, PowerNorm, to_rgba
import numpy as np

from analysis.failure_driven_viz.parse_logs import (
    default_figure_dir,
    ensure_parsed_dir,
    load_parsed_artifacts,
    sample_stage_ids,
)
from analysis.failure_driven_viz.style import (
    CARD_BG,
    CARD_EDGE,
    MUTED_TEXT,
    PANEL_EDGE_LIGHT,
    TITLE_COLOR,
    add_badge,
    configure_matplotlib,
    style_scene_axis,
)
from analysis.failure_driven_viz.video_style import (
    VIDEO_OBS_FILL,
    VIDEO_TARGET_EDGE,
    VIDEO_TARGET_FILL,
    draw_video_background,
    draw_video_tblock,
    setup_video_axis,
    world_edges_to_video_edges,
)

COORDINATE_FRAMES = ("absolute", "relative", "aligned")
HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "video_heat",
    [
        (0.00, (0.0, 0.0, 0.0, 0.00)),
        (0.08, (1.00, 0.86, 0.87, 0.35)),
        (0.30, (1.00, 0.63, 0.65, 0.58)),
        (0.62, (1.00, 0.47, 0.50, 0.82)),
        (1.00, to_rgba(VIDEO_OBS_FILL, alpha=0.98)),
    ],
)

configure_matplotlib()


def plot_heatmap(
    parsed_dir: Path,
    output_path: Path,
    stage_ids: List[int] | None = None,
    bins: int = 25,
    xy_limit: float = 0.25,
    max_stages: int = 6,
    selection_mode: str = "paper",
    normalize: bool = True,
    coordinate_frame: str = "absolute",
) -> Path:
    artifacts = load_parsed_artifacts(str(parsed_dir))
    snapshot_rows = artifacts["layout_snapshots"]
    batch_rows = artifacts["batch_stats"]
    evolve_rows = artifacts["evolve_rounds"]
    run_meta = artifacts["run_meta"]

    transformed_rows, skipped_points = build_transformed_obstacle_rows(
        obstacle_rows=artifacts["obstacle_points"],
        snapshot_rows=snapshot_rows,
        coordinate_frame=coordinate_frame,
    )
    if not transformed_rows:
        raise ValueError(f"No obstacle points found in parsed tables for frame={coordinate_frame}.")

    grouped: Dict[int, List[dict]] = defaultdict(list)
    for row in transformed_rows:
        if row["stage_id"] is None:
            continue
        grouped[int(row["stage_id"])].append(row)

    stage_meta = _build_stage_metadata(snapshot_rows, batch_rows, evolve_rows)
    chosen_stage_ids = _select_stage_ids(
        available_stage_ids=grouped.keys(),
        stage_meta=stage_meta,
        explicit_stage_ids=stage_ids,
        selection_mode=selection_mode,
        max_stages=max_stages,
    )
    if not chosen_stage_ids:
        raise ValueError("No stage ids are available for heatmap plotting.")

    ncols = min(3, len(chosen_stage_ids))
    nrows = math.ceil(len(chosen_stage_ids) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.55 * ncols, 3.70 * nrows + 0.55),
        squeeze=False,
        constrained_layout=False,
    )
    fig.patch.set_facecolor("#ffffff")

    xedges = np.linspace(-xy_limit, xy_limit, bins + 1)
    yedges = np.linspace(-xy_limit, xy_limit, bins + 1)

    stage_heatmaps: Dict[int, np.ndarray] = {}
    stage_counts: Dict[int, int] = {}
    density_max = 0.0
    raw_count_max = 0.0

    for stage_id in chosen_stage_ids:
        rows = grouped.get(stage_id, [])
        xs = np.array([row["plot_x"] for row in rows], dtype=float)
        ys = np.array([row["plot_y"] for row in rows], dtype=float)
        heatmap, _, _ = np.histogram2d(xs, ys, bins=[xedges, yedges])
        stage_counts[stage_id] = len(rows)
        if normalize and len(rows) > 0:
            heatmap = heatmap / float(len(rows))
            density_max = max(density_max, float(np.max(heatmap)))
        else:
            raw_count_max = max(raw_count_max, float(np.max(heatmap)))
        stage_heatmaps[stage_id] = heatmap

    vmax = density_max if normalize else raw_count_max
    if vmax <= 0:
        vmax = 1.0
    norm = PowerNorm(gamma=0.55, vmin=0.0, vmax=vmax)

    for axis in axes.flat:
        axis.set_visible(False)

    mesh = None
    axis_xlabel, axis_ylabel = coordinate_frame_axis_labels(coordinate_frame)
    anchor_color, anchor_marker, anchor_size = coordinate_frame_anchor_style(coordinate_frame)
    for axis, stage_id in zip(axes.flat, chosen_stage_ids):
        rows = grouped.get(stage_id, [])
        axis.set_visible(True)
        heatmap = stage_heatmaps[stage_id]
        _, subtitle = _format_stage_title(stage_id, stage_meta.get(stage_id), stage_counts.get(stage_id, len(rows)))
        if coordinate_frame == "absolute":
            mesh = _draw_absolute_heatmap_panel(
                axis=axis,
                stage_id=stage_id,
                heatmap=heatmap,
                xedges=xedges,
                yedges=yedges,
                norm=norm,
                subtitle=subtitle,
                episode_text=_stage_episode_text(stage_meta.get(stage_id)),
                failure_region_text=_stage_failure_region_text(stage_meta.get(stage_id)),
            )
        else:
            style_scene_axis(axis)
            axis.set_facecolor(CARD_BG)
            mesh = axis.pcolormesh(xedges, yedges, heatmap.T, cmap=HEATMAP_CMAP, norm=norm, shading="auto")
            axis.scatter([0.0], [0.0], color=anchor_color, marker=anchor_marker, s=anchor_size, linewidths=2.2, zorder=4)
            axis.axhline(0.0, color="white", linewidth=0.6, alpha=0.18)
            axis.axvline(0.0, color="white", linewidth=0.6, alpha=0.18)
            axis.set_xlim(-xy_limit, xy_limit)
            axis.set_ylim(-xy_limit, xy_limit)
            axis.set_aspect("equal")
            axis.grid(alpha=0.10, linestyle=":")
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            add_badge(axis, 0.03, 0.97, f"G{stage_id}", facecolor="#284a76", fontsize=8.8)
            add_badge(
                axis,
                0.97,
                0.97,
                _count_badge_text(stage_counts.get(stage_id, len(rows))),
                facecolor="#ffffff",
                edgecolor=CARD_EDGE,
                textcolor=MUTED_TEXT,
                ha="right",
                fontsize=8.1,
            )
            axis.text(
                0.03,
                0.90,
                subtitle,
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=8.4,
                color=MUTED_TEXT,
            )
            axis.set_xlabel(axis_xlabel)
            axis.set_ylabel(axis_ylabel)

    fig.suptitle(
        coordinate_frame_title(coordinate_frame),
        x=0.08,
        y=0.982,
        ha="left",
        fontsize=16,
        fontweight="bold",
        color=TITLE_COLOR,
    )
    if coordinate_frame != "absolute":
        fig.text(
            0.08,
            0.952,
            _heatmap_subtitle(run_meta, chosen_stage_ids, normalize, selection_mode, coordinate_frame, skipped_points),
            ha="left",
            va="bottom",
            fontsize=9,
            color=MUTED_TEXT,
        )
    fig.subplots_adjust(left=0.04, right=0.90, bottom=0.04, top=0.89, wspace=0.08, hspace=0.10)
    if mesh is not None:
        colorbar = fig.colorbar(mesh, ax=axes.ravel().tolist(), shrink=0.84, pad=0.015)
        colorbar.set_label("Normalized occupancy" if normalize else "Obstacle count")
        colorbar.outline.set_edgecolor(PANEL_EDGE_LIGHT)
        colorbar.outline.set_linewidth(1.0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot obstacle heatmaps for selected stages.")
    parser.add_argument("--run-dir", default=None, help="Experiment output directory.")
    parser.add_argument("--parsed-dir", default=None, help="Existing parsed directory.")
    parser.add_argument("--export-dir", default=None, help="Optional parsed export directory when using --run-dir.")
    parser.add_argument("--force-reparse", action="store_true", help="Regenerate parsed tables before plotting.")
    parser.add_argument("--output", default=None, help="Output PNG path.")
    parser.add_argument("--stage-ids", nargs="*", type=int, default=None, help="Optional explicit stage ids.")
    parser.add_argument("--max-stages", type=int, default=6, help="Maximum number of stages when auto-selecting.")
    parser.add_argument("--bins", type=int, default=25, help="2D histogram bin count per axis.")
    parser.add_argument("--xy-limit", type=float, default=0.25, help="Plot range in both x/y directions.")
    parser.add_argument(
        "--selection-mode",
        choices=["paper", "even"],
        default="paper",
        help="Automatic stage selection strategy when --stage-ids is not provided.",
    )
    parser.add_argument(
        "--coordinate-frame",
        choices=COORDINATE_FRAMES,
        default="absolute",
        help="How to express obstacle positions: absolute world coordinates, relative to start A, or aligned to A heading.",
    )
    parser.add_argument(
        "--raw-counts",
        action="store_true",
        help="Disable per-stage normalization and plot raw obstacle counts instead.",
    )
    args = parser.parse_args()

    parsed_dir = ensure_parsed_dir(
        run_dir=args.run_dir,
        parsed_dir=args.parsed_dir,
        export_dir=args.export_dir,
        force_reparse=args.force_reparse,
    )
    output_path = Path(args.output).expanduser().resolve() if args.output else default_figure_dir(parsed_dir) / "heatmap.png"
    saved_path = plot_heatmap(
        parsed_dir,
        output_path,
        stage_ids=args.stage_ids,
        bins=args.bins,
        xy_limit=args.xy_limit,
        max_stages=args.max_stages,
        selection_mode=args.selection_mode,
        normalize=not args.raw_counts,
        coordinate_frame=args.coordinate_frame,
    )
    print(f"Saved heatmap plot to: {saved_path}")


def _draw_absolute_heatmap_panel(
    axis: plt.Axes,
    stage_id: int,
    heatmap: np.ndarray,
    xedges: np.ndarray,
    yedges: np.ndarray,
    norm: Normalize,
    subtitle: str,
    episode_text: str,
    failure_region_text: str,
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
    masked = np.ma.masked_less_equal(heatmap.T, 0.0)
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
    add_badge(axis, 0.02, 0.97, f"G{stage_id}", facecolor="#284a76", fontsize=8.8)
    add_badge(
        axis,
        0.98,
        0.97,
        episode_text,
        facecolor="#f8fbff",
        edgecolor="#9fc1e8",
        textcolor="#42556f",
        ha="right",
        fontsize=8.1,
    )
    axis.text(
        0.98,
        0.885,
        failure_region_text,
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=7.9,
        color="#243447",
        bbox={
            "boxstyle": "round,pad=0.18,rounding_size=0.12",
            "facecolor": "#ffffff",
            "edgecolor": "#d6e3f2",
            "linewidth": 0.8,
            "alpha": 0.96,
        },
        clip_on=True,
        zorder=10,
    )
    axis.text(
        0.03,
        0.11,
        subtitle,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        color="#111827",
        clip_on=True,
    )
    return image


def _build_stage_metadata(
    snapshot_rows: List[dict],
    batch_rows: List[dict],
    evolve_rows: List[dict],
) -> Dict[int, dict]:
    metadata: Dict[int, dict] = {}
    snapshots_by_stage: Dict[int, List[dict]] = defaultdict(list)

    for row in snapshot_rows:
        if row["stage_id"] is None:
            continue
        snapshots_by_stage[int(row["stage_id"])].append(row)

    evolve_by_stage: Dict[int, dict] = {}
    for row in evolve_rows:
        evolve_id = row.get("evolve_id")
        if evolve_id is None:
            continue
        evolve_by_stage[int(evolve_id)] = row

    batch_by_stage: Dict[int, dict] = {}
    for row in batch_rows:
        after = row.get("generator_id_after_batch")
        if after is None:
            continue
        after_int = int(after)
        if after_int not in batch_by_stage and row.get("new_generator_loaded"):
            batch_by_stage[after_int] = row

    for stage_id, rows in snapshots_by_stage.items():
        episodes = sorted(int(row["episode"]) for row in rows)
        entry = {
            "stage_id": stage_id,
            "episode_start": episodes[0],
            "episode_end": episodes[-1],
            "num_snapshots": len(rows),
        }
        if stage_id == 0:
            entry["trigger_reason"] = "initial"
            entry["dominant_failure_type"] = "initial_layout"
        if stage_id in evolve_by_stage:
            evolve_row = evolve_by_stage[stage_id]
            entry["trigger_reason"] = evolve_row.get("trigger_reason")
            entry["dominant_failure_type"] = evolve_row.get("dominant_failure_type")
            entry["episode_total"] = evolve_row.get("episode_total")
            entry["failure_region"] = evolve_row.get("failure_region")
            entry["behavior_bias"] = evolve_row.get("behavior_bias")
        elif stage_id in batch_by_stage:
            batch_row = batch_by_stage[stage_id]
            entry["trigger_reason"] = batch_row.get("evolve_trigger_reason")
            entry["dominant_failure_type"] = batch_row.get("dominant_failure_type")
            entry["episode_total"] = batch_row.get("batch_end_episode")
            entry["failure_region"] = batch_row.get("failure_region")
            entry["behavior_bias"] = batch_row.get("behavior_bias")
        metadata[stage_id] = entry

    return metadata


def _select_stage_ids(
    available_stage_ids,
    stage_meta: Dict[int, dict],
    explicit_stage_ids: List[int] | None,
    selection_mode: str,
    max_stages: int,
) -> List[int]:
    available = sorted(int(stage_id) for stage_id in available_stage_ids)
    if explicit_stage_ids:
        return [stage_id for stage_id in explicit_stage_ids if stage_id in available]

    if selection_mode == "even":
        return sorted(sample_stage_ids(available, max_count=max_stages))

    return sorted(_paper_stage_selection(available, stage_meta, max_stages=max_stages))


def _paper_stage_selection(available: List[int], stage_meta: Dict[int, dict], max_stages: int) -> List[int]:
    chosen: List[int] = []

    def add(stage_id: int | None) -> None:
        if stage_id is None:
            return
        if stage_id in available and stage_id not in chosen:
            chosen.append(stage_id)

    add(0 if 0 in available else (available[0] if available else None))

    too_hard = [stage_id for stage_id in available if _reason_family(stage_meta.get(stage_id, {}).get("trigger_reason")) == "too_hard"]
    plateau = [stage_id for stage_id in available if _reason_family(stage_meta.get(stage_id, {}).get("trigger_reason")) == "plateau"]
    too_easy = [stage_id for stage_id in available if _reason_family(stage_meta.get(stage_id, {}).get("trigger_reason")) == "too_easy"]
    timeout_targeted = [
        stage_id
        for stage_id in available
        if "timeout" in str(stage_meta.get(stage_id, {}).get("dominant_failure_type", "")).lower()
    ]

    add(too_hard[0] if too_hard else None)
    add(too_hard[-1] if too_hard else None)
    add(plateau[0] if plateau else None)
    add(too_easy[0] if too_easy else None)
    add(timeout_targeted[0] if timeout_targeted else None)
    add(available[-1] if available else None)

    if len(chosen) < max_stages:
        fill_candidates = sample_stage_ids(available, max_count=max_stages * 2)
        for stage_id in fill_candidates:
            add(stage_id)
            if len(chosen) >= max_stages:
                break

    if len(chosen) > max_stages:
        # Keep first and last, then evenly thin the middle set.
        core = chosen[1:-1]
        keep_middle = max(0, max_stages - 2)
        reduced = sample_stage_ids(core, max_count=keep_middle) if keep_middle else []
        chosen = [chosen[0], *reduced, chosen[-1]]

    return chosen[:max_stages]


def _format_stage_title(stage_id: int, meta: dict | None, obstacle_count: int) -> Tuple[str, str]:
    if meta is None:
        return f"Stage {stage_id}", f"n={obstacle_count}"

    if stage_id == 0:
        title = "Initial Generator"
    else:
        reason = _reason_family(meta.get("trigger_reason"))
        dominant = _failure_family(meta.get("dominant_failure_type"))
        title = f"G{stage_id}  {reason}"
        if dominant != "unknown":
            title += f" / {dominant}"

    episode_start = meta.get("episode_start")
    episode_end = meta.get("episode_end")
    subtitle = (
        f"episodes {_fmt_k(episode_start)}-{_fmt_k(episode_end)}"
        if episode_start is not None and episode_end is not None
        else "episodes n/a"
    )
    subtitle += f" | obs={obstacle_count}"
    return title, subtitle


def _stage_episode_text(meta: dict | None) -> str:
    if not meta:
        return "ep n/a"
    episode_start = meta.get("episode_start")
    episode_end = meta.get("episode_end")
    if episode_start is None or episode_end is None:
        return "ep n/a"
    if int(episode_start) == int(episode_end):
        return f"ep {_fmt_k(episode_start)}"
    return f"{_fmt_k(episode_start)}-{_fmt_k(episode_end)}"


def _stage_failure_region_text(meta: dict | None) -> str:
    if not meta:
        return "region: n/a"
    region = str(meta.get("failure_region") or "").strip()
    if region:
        return f"region: {region}"
    if int(meta.get("stage_id", -1)) == 0:
        return "region: initial"
    return "region: none"


def _count_badge_text(obstacle_count: int) -> str:
    return f"obs {int(obstacle_count)}"


def _heatmap_subtitle(
    run_meta: dict,
    chosen_stage_ids: List[int],
    normalize: bool,
    selection_mode: str,
    coordinate_frame: str,
    skipped_points: int,
) -> str:
    run_name = Path(str(run_meta.get("output_dir", run_meta.get("run_id", "run")))).name
    metric = "normalized occupancy" if normalize else "raw count"
    chosen = ", ".join(str(stage_id) for stage_id in chosen_stage_ids)
    skipped_text = f" | skipped={skipped_points}" if skipped_points > 0 else ""
    return (
        f"{run_name} | selection={selection_mode} | frame={coordinate_frame} | "
        f"metric={metric} | stages=[{chosen}]{skipped_text}"
    )


def build_transformed_obstacle_rows(
    obstacle_rows: List[dict],
    snapshot_rows: List[dict],
    coordinate_frame: str,
) -> Tuple[List[dict], int]:
    if coordinate_frame not in COORDINATE_FRAMES:
        raise ValueError(f"Unsupported coordinate frame: {coordinate_frame}")

    snapshot_lookup = {int(row["snapshot_id"]): row for row in snapshot_rows if row.get("snapshot_id") is not None}
    transformed_rows: List[dict] = []
    skipped_points = 0

    for row in obstacle_rows:
        projected = _project_point(row, snapshot_lookup, coordinate_frame)
        if projected is None:
            skipped_points += 1
            continue
        plot_x, plot_y = projected
        transformed_rows.append({**row, "plot_x": plot_x, "plot_y": plot_y})

    return transformed_rows, skipped_points


def coordinate_frame_title(coordinate_frame: str) -> str:
    if coordinate_frame == "absolute":
        return "Obstacle Density Evolution Across Stages"
    if coordinate_frame == "relative":
        return "Obstacle Density Relative to Start A Across Stages"
    return "Obstacle Density in Start-Aligned Frame Across Stages"


def coordinate_frame_description(coordinate_frame: str) -> str:
    if coordinate_frame == "absolute":
        return "world coordinates around the fixed goal"
    if coordinate_frame == "relative":
        return "goal-independent offsets from sampled start A"
    return "start-centered offsets rotated into A heading"


def coordinate_frame_axis_labels(coordinate_frame: str) -> Tuple[str, str]:
    if coordinate_frame == "absolute":
        return "x", "y"
    if coordinate_frame == "relative":
        return "dx from start A", "dy from start A"
    return "forward from start A", "lateral from start A"


def coordinate_frame_anchor_style(coordinate_frame: str) -> Tuple[str, str, int]:
    if coordinate_frame == "absolute":
        return "deepskyblue", "x", 70
    return "#1b9e77", "x", 70


def _project_point(
    obstacle_row: dict,
    snapshot_lookup: Dict[int, dict],
    coordinate_frame: str,
) -> Tuple[float, float] | None:
    obs_x = float(obstacle_row["obs_x"])
    obs_y = float(obstacle_row["obs_y"])
    if coordinate_frame == "absolute":
        return obs_x, obs_y

    snapshot_id = int(obstacle_row["snapshot_id"])
    snapshot = snapshot_lookup.get(snapshot_id)
    if snapshot is None:
        return None

    tblock_x = snapshot.get("tblock_x")
    tblock_y = snapshot.get("tblock_y")
    tblock_theta_deg = snapshot.get("tblock_theta_deg")
    if tblock_x is None or tblock_y is None or tblock_theta_deg is None:
        return None

    dx = obs_x - float(tblock_x)
    dy = obs_y - float(tblock_y)
    if coordinate_frame == "relative":
        return dx, dy

    theta_rad = np.deg2rad(float(tblock_theta_deg))
    cos_theta = float(np.cos(theta_rad))
    sin_theta = float(np.sin(theta_rad))
    forward = cos_theta * dx + sin_theta * dy
    lateral = -sin_theta * dx + cos_theta * dy
    return forward, lateral


def _reason_family(trigger_reason: object) -> str:
    if not trigger_reason:
        return "unknown"
    return str(trigger_reason).split("(", 1)[0].strip() or "unknown"


def _failure_family(dominant_failure_type: object) -> str:
    if not dominant_failure_type:
        return "unknown"
    text = str(dominant_failure_type).lower()
    if "timeout" in text:
        return "timeout"
    if "fall" in text:
        return "fall"
    if "collision" in text:
        return "collision"
    if "initial" in text:
        return "initial"
    return "unknown"


def _fmt_k(value: object) -> str:
    if value is None:
        return "n/a"
    numeric = int(float(value))
    if abs(numeric) >= 1000:
        return f"{numeric // 1000}k"
    return str(numeric)


if __name__ == "__main__":
    main()
