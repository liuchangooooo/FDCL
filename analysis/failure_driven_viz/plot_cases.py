"""Plot representative scene cases for selected curriculum stages."""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from analysis.failure_driven_viz.parse_logs import (
    default_figure_dir,
    ensure_parsed_dir,
    load_parsed_artifacts,
    sample_stage_ids,
)
from analysis.failure_driven_viz.plot_heatmap import (
    COORDINATE_FRAMES,
    coordinate_frame_axis_labels,
    coordinate_frame_description,
)
from analysis.failure_driven_viz.style import (
    GOAL_BLOCK_EDGE,
    GOAL_BLOCK_FILL,
    GOAL_TEXT,
    MUTED_TEXT,
    OBSTACLE_EDGE,
    OBSTACLE_FILL,
    OBSTACLE_TEXT,
    PANEL_EDGE_LIGHT,
    REFERENCE_LINE,
    START_BLOCK_EDGE,
    START_BLOCK_FILL,
    SUBTITLE_COLOR,
    TITLE_COLOR,
    WORKSPACE_BG,
    add_badge,
    configure_matplotlib,
    style_scene_axis,
)
from analysis.failure_driven_viz.video_style import (
    VIDEO_CURRENT_EDGE,
    VIDEO_CURRENT_FILL,
    VIDEO_OBS_EDGE,
    VIDEO_OBS_FILL,
    VIDEO_TARGET_EDGE,
    VIDEO_TARGET_FILL,
    draw_video_background,
    draw_video_obstacle,
    draw_video_tblock,
    setup_video_axis,
)

configure_matplotlib()

TARGET_THETA_DEG = -45.0


def plot_cases(
    parsed_dir: Path,
    output_path: Path,
    stage_ids: List[int] | None = None,
    xy_limit: float = 0.3,
    max_cases: int = 6,
    selection_mode: str = "paper",
    coordinate_frame: str = "absolute",
) -> Path:
    artifacts = load_parsed_artifacts(str(parsed_dir))
    snapshot_rows = artifacts["layout_snapshots"]
    obstacle_rows = artifacts["obstacle_points"]
    batch_rows = artifacts["batch_stats"]
    evolve_rows = artifacts["evolve_rounds"]
    run_meta = artifacts["run_meta"]

    if not snapshot_rows or not obstacle_rows:
        raise ValueError("Representative case plotting requires snapshots and obstacle points.")

    stage_meta = _build_stage_metadata(snapshot_rows, batch_rows, evolve_rows)

    snapshots_by_stage: Dict[int, List[dict]] = defaultdict(list)
    for row in snapshot_rows:
        if row["stage_id"] is None:
            continue
        snapshots_by_stage[int(row["stage_id"])].append(row)

    chosen_stage_ids = _select_stage_ids(
        available_stage_ids=snapshots_by_stage.keys(),
        stage_meta=stage_meta,
        explicit_stage_ids=stage_ids,
        selection_mode=selection_mode,
        max_stages=max_cases,
    )
    if not chosen_stage_ids:
        raise ValueError("No stage ids are available for case plotting.")

    obstacles_by_snapshot: Dict[int, List[dict]] = defaultdict(list)
    for row in obstacle_rows:
        obstacles_by_snapshot[int(row["snapshot_id"])].append(row)

    chosen_snapshots: List[dict] = []
    for stage_id in chosen_stage_ids:
        stage_snapshots = sorted(snapshots_by_stage.get(stage_id, []), key=lambda row: row["episode"])
        if not stage_snapshots:
            continue
        chosen_snapshots.append(_select_representative_snapshot(stage_snapshots))

    if not chosen_snapshots:
        raise ValueError("No representative snapshots were selected.")

    ncols = min(3, len(chosen_snapshots))
    nrows = math.ceil(len(chosen_snapshots) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.55 * ncols, 3.65 * nrows + 0.55),
        squeeze=False,
        constrained_layout=False,
    )
    fig.patch.set_facecolor("#ffffff")

    for axis in axes.flat:
        axis.set_visible(False)

    for axis, snapshot in zip(axes.flat, chosen_snapshots):
        axis.set_visible(True)
        _draw_case_panel(
            axis=axis,
            snapshot=snapshot,
            obstacles=obstacles_by_snapshot.get(int(snapshot["snapshot_id"]), []),
            stage_meta=stage_meta.get(int(snapshot["stage_id"])),
            xy_limit=xy_limit,
            coordinate_frame=coordinate_frame,
        )

    fig.suptitle(
        _cases_title(coordinate_frame),
        x=0.08,
        y=0.982,
        ha="left",
        fontsize=16,
        fontweight="bold",
        color=TITLE_COLOR,
    )
    fig.subplots_adjust(left=0.04, right=0.995, bottom=0.04, top=0.89, wspace=0.10, hspace=0.08)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot representative scene cases.")
    parser.add_argument("--run-dir", default=None, help="Experiment output directory.")
    parser.add_argument("--parsed-dir", default=None, help="Existing parsed directory.")
    parser.add_argument("--export-dir", default=None, help="Optional parsed export directory when using --run-dir.")
    parser.add_argument("--force-reparse", action="store_true", help="Regenerate parsed tables before plotting.")
    parser.add_argument("--output", default=None, help="Output PNG path.")
    parser.add_argument("--stage-ids", nargs="*", type=int, default=None, help="Optional explicit stage ids.")
    parser.add_argument("--max-cases", type=int, default=6, help="Maximum number of auto-selected cases.")
    parser.add_argument("--xy-limit", type=float, default=0.3, help="Plot range in both x/y directions.")
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
        help="How to express scene geometry: absolute world coordinates, relative to start A, or aligned to A heading.",
    )
    args = parser.parse_args()

    parsed_dir = ensure_parsed_dir(
        run_dir=args.run_dir,
        parsed_dir=args.parsed_dir,
        export_dir=args.export_dir,
        force_reparse=args.force_reparse,
    )
    output_path = Path(args.output).expanduser().resolve() if args.output else default_figure_dir(parsed_dir) / "cases.png"
    saved_path = plot_cases(
        parsed_dir,
        output_path,
        stage_ids=args.stage_ids,
        xy_limit=args.xy_limit,
        max_cases=args.max_cases,
        selection_mode=args.selection_mode,
        coordinate_frame=args.coordinate_frame,
    )
    print(f"Saved representative cases plot to: {saved_path}")


def _draw_case_panel(
    axis: plt.Axes,
    snapshot: dict,
    obstacles: List[dict],
    stage_meta: dict | None,
    xy_limit: float,
    coordinate_frame: str,
    title_override: str | None = None,
    subtitle_override: str | None = None,
    show_axes: bool = False,
) -> None:
    if coordinate_frame == "absolute":
        _draw_video_case_panel(
            axis=axis,
            snapshot=snapshot,
            obstacles=obstacles,
            stage_meta=stage_meta,
            title_override=title_override,
            subtitle_override=subtitle_override,
        )
        return

    stage_id = int(snapshot["stage_id"])
    episode = int(snapshot["episode"])
    panel = _build_case_panel_geometry(snapshot, obstacles, coordinate_frame)
    style_scene_axis(axis)

    workspace = patches.Rectangle(
        (-xy_limit, -xy_limit),
        2 * xy_limit,
        2 * xy_limit,
        linewidth=1.25,
        edgecolor=PANEL_EDGE_LIGHT,
        facecolor=WORKSPACE_BG,
        zorder=0,
    )
    axis.add_patch(workspace)

    start_x, start_y = panel["start_point"]
    goal_x, goal_y = panel["goal_point"]
    axis.plot(
        [start_x, goal_x],
        [start_y, goal_y],
        linestyle=(0, (3.0, 2.0)),
        linewidth=1.15,
        color=REFERENCE_LINE,
        alpha=0.95,
        zorder=1,
    )

    _draw_tblock(
        axis,
        center=(goal_x, goal_y),
        theta_deg=float(panel["goal_heading_deg"]),
        facecolor=GOAL_BLOCK_FILL,
        edgecolor=GOAL_BLOCK_EDGE,
        alpha=0.34,
        zorder=2,
    )
    _draw_tblock(
        axis,
        center=(start_x, start_y),
        theta_deg=float(panel["start_heading_deg"]),
        facecolor=START_BLOCK_FILL,
        edgecolor=START_BLOCK_EDGE,
        alpha=0.95,
        zorder=4,
    )

    axis.text(
        start_x - 0.008,
        start_y - 0.022,
        "A",
        fontsize=9.2,
        fontweight="bold",
        color=START_BLOCK_EDGE,
        ha="center",
        va="top",
        zorder=6,
    )
    axis.text(
        goal_x + 0.016,
        goal_y + 0.014,
        "goal",
        fontsize=9.0,
        fontweight="bold",
        color=GOAL_TEXT,
        ha="left",
        va="center",
        zorder=6,
    )

    obstacle_summary_lines = []
    for obstacle in panel["obstacles"]:
        x = float(obstacle["plot_x"])
        y = float(obstacle["plot_y"])
        idx = int(obstacle["obstacle_idx"])
        purpose = str(obstacle["purpose"])

        square = patches.Rectangle(
            (x - 0.01, y - 0.01),
            0.02,
            0.02,
            linewidth=1.1,
            edgecolor=OBSTACLE_EDGE,
            facecolor=OBSTACLE_FILL,
            zorder=7,
        )
        axis.add_patch(square)
        axis.text(
            x,
            y,
            str(idx),
            fontsize=8.1,
            fontweight="bold",
            color=OBSTACLE_TEXT,
            ha="center",
            va="center",
            zorder=8,
        )
        obstacle_summary_lines.append(_obstacle_summary_line(idx, purpose, x, y))

    title, subtitle = _format_case_title(stage_id, stage_meta, episode)
    add_badge(axis, 0.03, 0.97, title.split("  ", 1)[0], facecolor="#1e3a5f")
    add_badge(axis, 0.97, 0.97, f"ep {_fmt_k(episode)}", facecolor="#ffffff", edgecolor=PANEL_EDGE_LIGHT, textcolor=MUTED_TEXT, ha="right")

    header_text = subtitle
    if subtitle_override is not None:
        header_text = subtitle_override
    axis.text(
        0.03,
        0.90,
        header_text,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        color=MUTED_TEXT,
        zorder=11,
    )

    if title_override is not None:
        axis.text(
            0.5,
            1.06,
            title_override,
            transform=axis.transAxes,
            ha="center",
            va="bottom",
            fontsize=12.2,
            fontweight="bold",
            color=TITLE_COLOR,
        )

    if obstacle_summary_lines:
        axis.text(
            0.03,
            0.04,
            _compact_obstacle_caption(obstacle_summary_lines),
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.0,
            color=MUTED_TEXT,
            bbox={
                "boxstyle": "round,pad=0.25,rounding_size=0.18",
                "facecolor": "#ffffff",
                "edgecolor": PANEL_EDGE_LIGHT,
                "linewidth": 0.9,
                "alpha": 0.96,
            },
            zorder=12,
        )

    axis.set_xlim(-xy_limit, xy_limit)
    axis.set_ylim(-xy_limit, xy_limit)
    axis.set_aspect("equal")
    if show_axes:
        axis.set_xlabel(coordinate_frame_axis_labels(coordinate_frame)[0])
        axis.set_ylabel(coordinate_frame_axis_labels(coordinate_frame)[1])
        axis.grid(alpha=0.10, linestyle=":")
    else:
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_xlabel("")
        axis.set_ylabel("")


def _draw_video_case_panel(
    axis: plt.Axes,
    snapshot: dict,
    obstacles: List[dict],
    stage_meta: dict | None,
    title_override: str | None,
    subtitle_override: str | None,
) -> None:
    stage_id = int(snapshot["stage_id"])
    episode = int(snapshot["episode"])
    _, subtitle = _format_case_title(stage_id, stage_meta, episode)
    subtitle = _video_short_subtitle(stage_id, stage_meta) if subtitle_override is None else subtitle_override

    setup_video_axis(axis)
    draw_video_background(axis)
    draw_video_tblock(
        axis,
        center_world=(0.0, 0.0),
        theta_deg=TARGET_THETA_DEG,
        fill=VIDEO_TARGET_FILL,
        edge=VIDEO_TARGET_EDGE,
        zorder=3,
        alpha=0.98,
    )
    draw_video_tblock(
        axis,
        center_world=(float(snapshot["tblock_x"]), float(snapshot["tblock_y"])),
        theta_deg=float(snapshot["tblock_theta_deg"]),
        fill=VIDEO_CURRENT_FILL,
        edge=VIDEO_CURRENT_EDGE,
        zorder=4,
        alpha=0.98,
    )
    for obstacle in sorted(obstacles, key=lambda row: int(row["obstacle_idx"])):
        draw_video_obstacle(
            axis,
            center_world=(float(obstacle["obs_x"]), float(obstacle["obs_y"])),
        )

    add_badge(axis, 0.02, 0.97, f"G{stage_id}", facecolor="#284a76", fontsize=8.8)
    add_badge(
        axis,
        0.98,
        0.97,
        f"ep {_fmt_k(episode)}",
        facecolor="#f8fbff",
        edgecolor="#9fc1e8",
        textcolor="#42556f",
        ha="right",
        fontsize=8.3,
    )
    axis.text(
        0.03,
        0.075,
        subtitle if title_override is None else title_override,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=8.6,
        color="#eef5ff",
        clip_on=True,
    )
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
            entry["failure_region"] = evolve_row.get("failure_region")
            entry["behavior_bias"] = evolve_row.get("behavior_bias")
        elif stage_id in batch_by_stage:
            batch_row = batch_by_stage[stage_id]
            entry["trigger_reason"] = batch_row.get("evolve_trigger_reason")
            entry["dominant_failure_type"] = batch_row.get("dominant_failure_type")
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
    add(timeout_targeted[-1] if timeout_targeted else None)
    add(available[-1] if available else None)

    if len(chosen) < max_stages:
        fill_candidates = sample_stage_ids(available, max_count=max_stages * 2)
        for stage_id in fill_candidates:
            add(stage_id)
            if len(chosen) >= max_stages:
                break

    if len(chosen) > max_stages:
        core = chosen[1:-1]
        keep_middle = max(0, max_stages - 2)
        reduced = sample_stage_ids(core, max_count=keep_middle) if keep_middle else []
        chosen = [chosen[0], *reduced, chosen[-1]]

    return chosen[:max_stages]


def _format_case_title(stage_id: int, stage_meta: dict | None, episode: int) -> Tuple[str, str]:
    if stage_meta is None:
        return f"G{stage_id}", "representative snapshot"

    if stage_id == 0:
        title = "G0"
    else:
        reason = _reason_family(stage_meta.get("trigger_reason"))
        dominant = _failure_family(stage_meta.get("dominant_failure_type"))
        title = f"G{stage_id}"
        subtitle_parts = [reason.replace("_", " ")]
        if dominant != "unknown":
            subtitle_parts.append(dominant)
    if stage_id == 0:
        subtitle_parts = ["initial generator"]

    region = stage_meta.get("failure_region")
    bias = stage_meta.get("behavior_bias")
    if region:
        subtitle_parts.append(region)
    if bias and bias != region:
        subtitle_parts.append(f"bias {bias}")
    return title, " | ".join(subtitle_parts)


def _video_short_subtitle(stage_id: int, stage_meta: dict | None) -> str:
    if stage_id == 0 or not stage_meta:
        return "initial generator"

    parts: List[str] = []
    reason = _reason_family(stage_meta.get("trigger_reason"))
    dominant = _failure_family(stage_meta.get("dominant_failure_type"))
    if reason != "unknown":
        parts.append(reason.replace("_", " "))
    if dominant != "unknown":
        parts.append(dominant)
    if not parts:
        return f"generator {stage_id}"
    return " | ".join(parts)


def _cases_subtitle(run_meta: dict, chosen_stage_ids: List[int], selection_mode: str, coordinate_frame: str) -> str:
    run_name = Path(str(run_meta.get("output_dir", run_meta.get("run_id", "run")))).name
    chosen = ", ".join(str(stage_id) for stage_id in chosen_stage_ids)
    return (
        f"{run_name} | stages [{chosen}] | selection={selection_mode} | "
        f"frame={coordinate_frame} ({coordinate_frame_description(coordinate_frame)})"
    )


def _cases_title(coordinate_frame: str) -> str:
    if coordinate_frame == "absolute":
        return "Representative Obstacle Scenes in the Environment View"
    if coordinate_frame == "relative":
        return "Representative Obstacle Scenes Relative to Start A"
    return "Representative Obstacle Scenes in the Start-Aligned Frame"


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


def _obstacle_summary_line(idx: int, purpose: str, x: float, y: float) -> str:
    if purpose:
        return f"O{idx} {purpose}"
    return f"O{idx}"


def _compact_obstacle_caption(lines: List[str]) -> str:
    return "   |   ".join(lines)


def _select_representative_snapshot(stage_snapshots: List[dict]) -> dict:
    return stage_snapshots[len(stage_snapshots) // 2]


def _build_case_panel_geometry(snapshot: dict, obstacles: List[dict], coordinate_frame: str) -> Dict[str, object]:
    world_heading_deg = float(snapshot["tblock_theta_deg"])
    start_point = _start_point(snapshot, coordinate_frame)
    goal_point = _transform_world_point(0.0, 0.0, snapshot, coordinate_frame)
    goal_heading_deg = _transform_heading_deg(TARGET_THETA_DEG, world_heading_deg, coordinate_frame)

    projected_obstacles = []
    for obstacle in sorted(obstacles, key=lambda row: int(row["obstacle_idx"])):
        plot_x, plot_y = _transform_world_point(float(obstacle["obs_x"]), float(obstacle["obs_y"]), snapshot, coordinate_frame)
        projected_obstacles.append(
            {
                "obstacle_idx": int(obstacle["obstacle_idx"]),
                "plot_x": plot_x,
                "plot_y": plot_y,
                "purpose": str(obstacle.get("purpose") or "").strip(),
            }
        )

    return {
        "start_point": start_point,
        "goal_point": goal_point,
        "start_heading_deg": 0.0 if coordinate_frame == "aligned" else world_heading_deg,
        "goal_heading_deg": goal_heading_deg,
        "world_heading_deg": world_heading_deg,
        "obstacles": projected_obstacles,
    }


def _start_point(snapshot: dict, coordinate_frame: str) -> Tuple[float, float]:
    if coordinate_frame == "absolute":
        return float(snapshot["tblock_x"]), float(snapshot["tblock_y"])
    return 0.0, 0.0


def _transform_world_point(x: float, y: float, snapshot: dict, coordinate_frame: str) -> Tuple[float, float]:
    if coordinate_frame == "absolute":
        return x, y

    tblock_x = float(snapshot["tblock_x"])
    tblock_y = float(snapshot["tblock_y"])
    dx = x - tblock_x
    dy = y - tblock_y
    if coordinate_frame == "relative":
        return dx, dy

    theta_rad = np.deg2rad(float(snapshot["tblock_theta_deg"]))
    cos_theta = float(np.cos(theta_rad))
    sin_theta = float(np.sin(theta_rad))
    forward = cos_theta * dx + sin_theta * dy
    lateral = -sin_theta * dx + cos_theta * dy
    return forward, lateral


def _transform_heading_deg(angle_deg: float, start_heading_deg: float, coordinate_frame: str) -> float:
    if coordinate_frame == "aligned":
        return _normalize_angle_deg(angle_deg - start_heading_deg)
    return _normalize_angle_deg(angle_deg)


def _normalize_angle_deg(angle_deg: float) -> float:
    normalized = (float(angle_deg) + 180.0) % 360.0 - 180.0
    if normalized == -180.0:
        return 180.0
    return normalized


def _draw_tblock(
    axis: plt.Axes,
    center: Tuple[float, float],
    theta_deg: float,
    facecolor: str,
    edgecolor: str,
    alpha: float,
    zorder: int,
) -> None:
    cx, cy = center
    for center_local, width, height in _tblock_components():
        corners = np.array(
            [
                [-width / 2.0, -height / 2.0],
                [-width / 2.0, height / 2.0],
                [width / 2.0, height / 2.0],
                [width / 2.0, -height / 2.0],
            ],
            dtype=float,
        )
        corners += np.array(center_local, dtype=float)
        rotated = _rotate_points(corners, theta_deg)
        rotated[:, 0] += cx
        rotated[:, 1] += cy
        axis.add_patch(
            patches.Polygon(
                rotated,
                closed=True,
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=1.2,
                alpha=alpha,
                zorder=zorder,
                joinstyle="round",
            )
        )


def _tblock_components() -> List[Tuple[Tuple[float, float], float, float]]:
    return [
        ((0.0, 0.0), 0.10, 0.03),
        ((0.0, -0.05), 0.03, 0.07),
    ]


def _rotate_points(points: np.ndarray, theta_deg: float) -> np.ndarray:
    theta_rad = np.deg2rad(theta_deg)
    rotation = np.array(
        [
            [np.cos(theta_rad), -np.sin(theta_rad)],
            [np.sin(theta_rad), np.cos(theta_rad)],
        ],
        dtype=float,
    )
    return points @ rotation.T


def _fmt_k(value: object) -> str:
    if value is None:
        return "n/a"
    numeric = int(float(value))
    if abs(numeric) >= 1000:
        return f"{numeric // 1000}k"
    return str(numeric)


if __name__ == "__main__":
    main()
