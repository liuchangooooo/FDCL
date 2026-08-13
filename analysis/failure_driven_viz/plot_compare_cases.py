"""Plot side-by-side representative scene case comparisons for two runs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis.failure_driven_viz.parse_logs import ensure_parsed_dir, load_parsed_artifacts
from analysis.failure_driven_viz.plot_cases import (
    _build_stage_metadata,
    _draw_case_panel,
    _fmt_k,
    _select_representative_snapshot,
    _video_short_subtitle,
)
from analysis.failure_driven_viz.plot_heatmap import COORDINATE_FRAMES, coordinate_frame_description
from analysis.failure_driven_viz.style import (
    SUBTITLE_COLOR,
    TITLE_COLOR,
    add_summary_card,
    configure_matplotlib,
    run_accent_color,
)

configure_matplotlib()

PHASE_LABELS_3 = ["Early", "Middle", "Late"]
PHASE_LABELS_4 = ["Early", "Mid-Early", "Mid-Late", "Late"]


def plot_compare_cases(
    parsed_dir_a: Path,
    parsed_dir_b: Path,
    output_path: Path,
    label_a: Optional[str] = None,
    label_b: Optional[str] = None,
    num_phases: int = 3,
    xy_limit: float = 0.3,
    coordinate_frame: str = "absolute",
) -> Path:
    artifacts_a = load_parsed_artifacts(str(parsed_dir_a))
    artifacts_b = load_parsed_artifacts(str(parsed_dir_b))

    comparison_a = _prepare_progress_cases(
        artifacts=artifacts_a,
        label=label_a or _default_label(artifacts_a["run_meta"]),
        num_phases=num_phases,
    )
    comparison_b = _prepare_progress_cases(
        artifacts=artifacts_b,
        label=label_b or _default_label(artifacts_b["run_meta"]),
        num_phases=num_phases,
    )

    fig = plt.figure(figsize=(4.75 * num_phases + 3.0, 7.2), constrained_layout=False)
    fig.patch.set_facecolor("#ffffff")
    grid = fig.add_gridspec(
        2,
        num_phases + 1,
        width_ratios=[1.08] + [1.0] * num_phases,
        wspace=0.22,
        hspace=0.24,
    )

    for row_index, comparison in enumerate([comparison_a, comparison_b]):
        summary_axis = fig.add_subplot(grid[row_index, 0])
        _draw_summary_axis(summary_axis, comparison, coordinate_frame)

        for phase_index in range(num_phases):
            axis = fig.add_subplot(grid[row_index, phase_index + 1])
            panel = comparison["panels"][phase_index]

            if panel["snapshot"] is None:
                axis.set_axis_off()
                axis.text(
                    0.5,
                    0.5,
                    "No snapshot",
                    transform=axis.transAxes,
                    ha="center",
                    va="center",
                    fontsize=12,
                    color="#6b7280",
                )
                continue

            _draw_case_panel(
                axis=axis,
                snapshot=panel["snapshot"],
                obstacles=panel["obstacles"],
                stage_meta=panel["stage_meta"],
                xy_limit=xy_limit,
                coordinate_frame=coordinate_frame,
                subtitle_override=panel["detail_label"],
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
        _compare_cases_title(coordinate_frame),
        x=0.08,
        y=0.972,
        ha="left",
        fontsize=16,
        fontweight="bold",
        color=TITLE_COLOR,
    )
    fig.subplots_adjust(left=0.07, right=0.995, bottom=0.04, top=0.86, wspace=0.24, hspace=0.34)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two runs using progress-aligned representative scene panels.")
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
    parser.add_argument("--xy-limit", type=float, default=0.3, help="Plot range in both x/y directions.")
    parser.add_argument(
        "--coordinate-frame",
        choices=COORDINATE_FRAMES,
        default="absolute",
        help="How to express scene geometry: absolute world coordinates, relative to start A, or aligned to A heading.",
    )
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
    saved_path = plot_compare_cases(
        parsed_dir_a=parsed_dir_a,
        parsed_dir_b=parsed_dir_b,
        output_path=output_path,
        label_a=args.label_a,
        label_b=args.label_b,
        num_phases=args.num_phases,
        xy_limit=args.xy_limit,
        coordinate_frame=args.coordinate_frame,
    )
    print(f"Saved two-run case comparison to: {saved_path}")


def _prepare_progress_cases(
    artifacts: Dict[str, Any],
    label: str,
    num_phases: int,
) -> Dict[str, Any]:
    snapshot_rows = artifacts["layout_snapshots"]
    obstacle_rows = artifacts["obstacle_points"]
    batch_rows = artifacts["batch_stats"]
    evolve_rows = artifacts["evolve_rounds"]
    run_meta = artifacts["run_meta"]

    if not snapshot_rows or not obstacle_rows:
        raise ValueError(f"Parsed directory for {label} is missing snapshots or obstacle points.")

    stage_meta = _build_stage_metadata(snapshot_rows, batch_rows, evolve_rows)
    snapshots_sorted = sorted(snapshot_rows, key=lambda row: int(row["episode"]))
    obstacles_by_snapshot: Dict[int, List[dict]] = {}
    for row in obstacle_rows:
        obstacles_by_snapshot.setdefault(int(row["snapshot_id"]), []).append(row)

    min_episode = min(int(row["episode"]) for row in snapshots_sorted)
    max_episode = max(int(row["episode"]) for row in snapshots_sorted)
    episode_span = max(1, max_episode - min_episode)
    phase_labels = _phase_labels(num_phases)

    panels: List[Dict[str, Any]] = []
    for phase_index in range(num_phases):
        lower = phase_index / float(num_phases)
        upper = (phase_index + 1) / float(num_phases)
        phase_snapshots = []

        for row in snapshots_sorted:
            episode = int(row["episode"])
            progress = (episode - min_episode) / float(episode_span)
            in_phase = (progress >= lower) and (progress < upper or (phase_index == num_phases - 1 and progress <= upper))
            if in_phase:
                phase_snapshots.append(row)

        representative = _select_representative_snapshot(phase_snapshots) if phase_snapshots else None
        panels.append(
            {
                "phase_index": phase_index,
                "phase_label": phase_labels[phase_index],
                "progress_label": _progress_label(
                    lower=lower,
                    upper=upper,
                    phase_start=min(int(row["episode"]) for row in phase_snapshots) if phase_snapshots else None,
                    phase_end=max(int(row["episode"]) for row in phase_snapshots) if phase_snapshots else None,
                ),
                "snapshot": representative,
                "obstacles": obstacles_by_snapshot.get(int(representative["snapshot_id"]), []) if representative is not None else [],
                "stage_meta": stage_meta.get(int(representative["stage_id"])) if representative is not None else None,
                "stage_id": int(representative["stage_id"]) if representative is not None else None,
                "episode": int(representative["episode"]) if representative is not None else None,
                "detail_label": _detail_label(representative, stage_meta.get(int(representative["stage_id"])) if representative is not None else None),
            }
        )

    return {
        "label": label,
        "run_meta": run_meta,
        "panels": panels,
    }


def _draw_summary_axis(axis: plt.Axes, comparison: Dict[str, Any], coordinate_frame: str) -> None:
    run_meta = comparison["run_meta"]
    accent = run_accent_color(comparison["label"])
    title = _run_badge(comparison["label"])
    run_name = Path(str(run_meta.get("output_dir", run_meta.get("run_id", comparison["label"])))).name
    lines = [
        f"Run: {run_name}",
        f"Seen success rate: {_format_metric(run_meta.get('final_seen_success_rate'))}",
        f"Validate reward: {_format_metric(run_meta.get('final_validate_reward'))}",
        f"Evolve rounds: {_format_metric(run_meta.get('final_evolve_count'), integer=True)}",
        f"View: {coordinate_frame}",
    ]
    add_summary_card(
        axis,
        title=title,
        lines=lines,
        accent=accent,
        badge=run_name,
    )


def _comparison_subtitle(comparison_a: Dict[str, Any], comparison_b: Dict[str, Any], coordinate_frame: str) -> str:
    return (
        f"Progress-aligned representative snapshots | frame={coordinate_frame} "
        f"({coordinate_frame_description(coordinate_frame)}) | {comparison_a['label']} vs {comparison_b['label']}"
    )


def _compare_cases_title(coordinate_frame: str) -> str:
    if coordinate_frame == "absolute":
        return "Static vs Failure-Driven Obstacle Scenes in the Environment View"
    if coordinate_frame == "relative":
        return "Static vs Failure-Driven Scenes Relative to Start A"
    return "Static vs Failure-Driven Scenes in the Start-Aligned Frame"


def _default_label(run_meta: Dict[str, Any]) -> str:
    output_name = Path(str(run_meta.get("output_dir", run_meta.get("run_id", "run")))).name
    if run_meta.get("enable_acgs_loop"):
        return f"{output_name} (failure-driven)"
    return f"{output_name} (static)"


def _run_badge(label: str) -> str:
    lowered = label.lower()
    if "failure-driven" in lowered:
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


def _detail_label(snapshot: Optional[dict], stage_meta: Optional[dict]) -> str:
    if snapshot is None:
        return "no snapshot"
    stage_id = int(snapshot.get("stage_id", 0))
    return _video_short_subtitle(stage_id, stage_meta)


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
