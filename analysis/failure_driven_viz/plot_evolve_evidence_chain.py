"""Plot the core diagnosis -> revision -> validation evidence chain.

This figure is designed to answer the most important visualization question
for the failure-driven obstacle curriculum:

1. What weakness did the system diagnose?
2. How did the generator family change after revision?
3. Did the new layout family improve the targeted failure pattern?
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.colors import PowerNorm
from matplotlib.ticker import PercentFormatter
import numpy as np

from analysis.failure_driven_viz.parse_logs import (
    default_figure_dir,
    ensure_parsed_dir,
    load_parsed_artifacts,
    sample_stage_ids,
)
from analysis.failure_driven_viz.plot_cases import (
    _build_stage_metadata,
    _failure_family,
    _fmt_k,
    _reason_family,
)
from analysis.failure_driven_viz.plot_heatmap import HEATMAP_CMAP
from analysis.failure_driven_viz.style import (
    CARD_BG,
    CARD_EDGE,
    COLLISION_LINE,
    EVOLVE_ACCENT,
    FALL_LINE,
    GRID_COLOR,
    MUTED_TEXT,
    PANEL_EDGE_LIGHT,
    STATIC_ACCENT,
    SUCCESS_LINE,
    TEXT_COLOR,
    TIMEOUT_LINE,
    TITLE_COLOR,
    add_badge,
    configure_matplotlib,
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

RATE_COLORS = {
    "success_rate": SUCCESS_LINE,
    "collision_rate": COLLISION_LINE,
    "timeout_rate": TIMEOUT_LINE,
    "fall_rate": FALL_LINE,
}

RATE_LABELS = {
    "success_rate": "success",
    "collision_rate": "collision",
    "timeout_rate": "timeout",
    "fall_rate": "fall",
}


def plot_evolve_evidence_chain(
    parsed_dir: Path,
    output_path: Path,
    evolve_round_indices: Optional[List[int]] = None,
    max_rounds: int = 4,
    bins: int = 25,
    xy_limit: float = 0.25,
) -> Path:
    artifacts = load_parsed_artifacts(str(parsed_dir))
    evolve_rows = artifacts["evolve_rounds"]
    batch_rows = artifacts["batch_stats"]
    snapshot_rows = artifacts["layout_snapshots"]
    obstacle_rows = artifacts["obstacle_points"]
    run_meta = artifacts["run_meta"]

    if not evolve_rows:
        raise ValueError("Evidence-chain plotting requires evolve rounds.")
    if not snapshot_rows or not obstacle_rows:
        raise ValueError("Evidence-chain plotting requires snapshots and obstacle points.")

    stage_meta = _build_stage_metadata(snapshot_rows, batch_rows, evolve_rows)
    entries = _prepare_evidence_entries(
        evolve_rows=evolve_rows,
        batch_rows=batch_rows,
        snapshot_rows=snapshot_rows,
        obstacle_rows=obstacle_rows,
        stage_meta=stage_meta,
        bins=bins,
        xy_limit=xy_limit,
    )
    if not entries:
        raise ValueError("No evolve rounds have enough evidence to build the chain figure.")

    chosen_entries = _select_entries(entries, evolve_round_indices=evolve_round_indices, max_rounds=max_rounds)
    if not chosen_entries:
        raise ValueError("No evidence rows were selected.")

    nrows = len(chosen_entries)
    fig = plt.figure(figsize=(15.6, 3.05 * nrows + 1.25), constrained_layout=False)
    fig.patch.set_facecolor("#ffffff")
    grid = fig.add_gridspec(
        nrows,
        4,
        width_ratios=[1.55, 1.0, 1.0, 1.55],
        wspace=0.18,
        hspace=0.34,
    )

    for row_index, entry in enumerate(chosen_entries):
        diagnosis_axis = fig.add_subplot(grid[row_index, 0])
        before_axis = fig.add_subplot(grid[row_index, 1])
        after_axis = fig.add_subplot(grid[row_index, 2])
        validation_axis = fig.add_subplot(grid[row_index, 3])

        _draw_diagnosis_card(diagnosis_axis, entry)
        _draw_stage_revision_panel(
            before_axis,
            entry["before_stage"],
            entry["before_heatmap"],
            entry["row_norm"],
            role_label="Before",
            accent=STATIC_ACCENT,
            episode_label=entry["before_episode_label"],
        )
        _draw_stage_revision_panel(
            after_axis,
            entry["after_stage"],
            entry["after_heatmap"],
            entry["row_norm"],
            role_label="After",
            accent=EVOLVE_ACCENT,
            episode_label=entry["after_episode_label"],
        )
        _draw_validation_panel(validation_axis, entry)

    fig.suptitle(
        "Failure-Driven Evidence Chain: Diagnosis -> Layout Revision -> Validation",
        x=0.06,
        y=0.985,
        ha="left",
        fontsize=17,
        fontweight="bold",
        color=TITLE_COLOR,
    )
    fig.text(
        0.06,
        0.952,
        _figure_subtitle(run_meta, chosen_entries),
        ha="left",
        va="top",
        fontsize=9.4,
        color=MUTED_TEXT,
    )

    headers = [
        ("Diagnosis", 0.150),
        ("Revision: Previous Layout Family", 0.434),
        ("Revision: New Layout Family", 0.603),
        ("Validation", 0.865),
    ]
    for text, xpos in headers:
        fig.text(
            xpos,
            0.915,
            text,
            ha="center",
            va="bottom",
            fontsize=11.0,
            fontweight="bold",
            color=TITLE_COLOR,
        )

    fig.subplots_adjust(left=0.035, right=0.995, bottom=0.04, top=0.88, wspace=0.18, hspace=0.32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot diagnosis -> revision -> validation evidence chains.")
    parser.add_argument("--run-dir", default=None, help="Experiment output directory.")
    parser.add_argument("--parsed-dir", default=None, help="Existing parsed directory.")
    parser.add_argument("--export-dir", default=None, help="Optional parsed export directory when using --run-dir.")
    parser.add_argument("--force-reparse", action="store_true", help="Regenerate parsed tables before plotting.")
    parser.add_argument("--output", default=None, help="Output PNG path.")
    parser.add_argument("--max-rounds", type=int, default=4, help="Maximum number of evolve rounds to show.")
    parser.add_argument("--bins", type=int, default=25, help="2D histogram bin count per axis.")
    parser.add_argument("--xy-limit", type=float, default=0.25, help="World range used for heatmap projection.")
    parser.add_argument(
        "--evolve-rounds",
        nargs="*",
        type=int,
        default=None,
        help="Optional explicit evolve round indices to show.",
    )
    args = parser.parse_args()

    parsed_dir = ensure_parsed_dir(
        run_dir=args.run_dir,
        parsed_dir=args.parsed_dir,
        export_dir=args.export_dir,
        force_reparse=args.force_reparse,
    )
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else default_figure_dir(parsed_dir) / "evidence_chain.png"
    )
    saved_path = plot_evolve_evidence_chain(
        parsed_dir=parsed_dir,
        output_path=output_path,
        evolve_round_indices=args.evolve_rounds,
        max_rounds=args.max_rounds,
        bins=args.bins,
        xy_limit=args.xy_limit,
    )
    print(f"Saved evidence-chain plot to: {saved_path}")


def _prepare_evidence_entries(
    *,
    evolve_rows: List[dict],
    batch_rows: List[dict],
    snapshot_rows: List[dict],
    obstacle_rows: List[dict],
    stage_meta: Dict[int, dict],
    bins: int,
    xy_limit: float,
) -> List[dict]:
    batch_by_idx = {int(row["batch_idx"]): row for row in batch_rows if row.get("batch_idx") is not None}
    stage_points: Dict[int, List[dict]] = {}
    for row in obstacle_rows:
        if row.get("stage_id") is None:
            continue
        stage_points.setdefault(int(row["stage_id"]), []).append(row)

    entries: List[dict] = []
    for evolve_row in evolve_rows:
        if not evolve_row.get("evolve_id"):
            continue
        linked_batch_idx = evolve_row.get("linked_batch_idx")
        if linked_batch_idx is None:
            continue
        linked_batch = batch_by_idx.get(int(linked_batch_idx))
        if linked_batch is None:
            continue

        before_stage = int(linked_batch["generator_id_before_batch"])
        after_stage = int(evolve_row["evolve_id"])
        before_points = stage_points.get(before_stage, [])
        after_points = stage_points.get(after_stage, [])
        if not before_points or not after_points:
            continue

        post_batches = _collect_post_batches(batch_rows, after_stage)
        if not post_batches:
            continue

        pre_metrics = _extract_pre_metrics(evolve_row, linked_batch)
        post_metrics = _aggregate_batch_metrics(post_batches)
        before_heatmap = _build_stage_heatmap(before_points, bins=bins, xy_limit=xy_limit)
        after_heatmap = _build_stage_heatmap(after_points, bins=bins, xy_limit=xy_limit)
        vmax = max(float(np.max(before_heatmap)), float(np.max(after_heatmap)), 1e-6)

        entry = {
            "evolve_row": evolve_row,
            "linked_batch": linked_batch,
            "before_stage": before_stage,
            "after_stage": after_stage,
            "before_meta": stage_meta.get(before_stage, {}),
            "after_meta": stage_meta.get(after_stage, {}),
            "before_points_count": len(before_points),
            "after_points_count": len(after_points),
            "before_heatmap": before_heatmap,
            "after_heatmap": after_heatmap,
            "row_norm": PowerNorm(gamma=0.55, vmin=0.0, vmax=vmax),
            "pre_metrics": pre_metrics,
            "post_metrics": post_metrics,
            "post_batches": post_batches,
            "before_episode_label": _stage_episode_badge(stage_meta.get(before_stage, {})),
            "after_episode_label": _stage_episode_badge(stage_meta.get(after_stage, {})),
        }
        entry.update(_build_validation_summary(entry))
        entries.append(entry)

    return entries


def _select_entries(entries: List[dict], evolve_round_indices: Optional[List[int]], max_rounds: int) -> List[dict]:
    entries_sorted = sorted(entries, key=lambda row: int(row["evolve_row"]["evolve_round_index"]))
    if evolve_round_indices:
        allowed = set(int(index) for index in evolve_round_indices)
        return [row for row in entries_sorted if int(row["evolve_row"]["evolve_round_index"]) in allowed]

    if len(entries_sorted) <= max_rounds:
        return entries_sorted

    chosen: List[dict] = []

    def add(candidate: Optional[dict]) -> None:
        if candidate is None:
            return
        if candidate not in chosen:
            chosen.append(candidate)

    add(entries_sorted[0])
    add(_first_matching_entry(entries_sorted, lambda row: _target_rate_key(row["evolve_row"]) == "timeout_rate"))
    add(entries_sorted[len(entries_sorted) // 2])
    add(entries_sorted[-1])

    if len(chosen) < max_rounds:
        sampled = sample_stage_ids(
            [int(row["evolve_row"]["evolve_round_index"]) for row in entries_sorted],
            max_count=max_rounds * 2,
        )
        lookup = {int(row["evolve_row"]["evolve_round_index"]): row for row in entries_sorted}
        for evolve_round_index in sampled:
            add(lookup.get(int(evolve_round_index)))
            if len(chosen) >= max_rounds:
                break

    chosen = sorted(chosen, key=lambda row: int(row["evolve_row"]["evolve_round_index"]))
    if len(chosen) > max_rounds:
        keep_rounds = set(
            sample_stage_ids(
                [int(row["evolve_row"]["evolve_round_index"]) for row in chosen],
                max_count=max_rounds,
            )
        )
        chosen = [row for row in chosen if int(row["evolve_row"]["evolve_round_index"]) in keep_rounds]

    return chosen[:max_rounds]


def _first_matching_entry(entries: List[dict], predicate) -> Optional[dict]:
    for entry in entries:
        if predicate(entry):
            return entry
    return None


def _collect_post_batches(batch_rows: List[dict], after_stage: int) -> List[dict]:
    stage_batches = [
        row
        for row in batch_rows
        if row.get("generator_id_before_batch") is not None and int(row["generator_id_before_batch"]) == int(after_stage)
    ]
    return sorted(stage_batches, key=lambda row: int(row["batch_start_episode"]))


def _extract_pre_metrics(evolve_row: dict, linked_batch: dict) -> Dict[str, float]:
    counts = {
        "success_count": evolve_row.get("coarse_success_count"),
        "collision_count": evolve_row.get("coarse_collision_count"),
        "timeout_count": evolve_row.get("coarse_timeout_count"),
        "fall_count": evolve_row.get("coarse_fall_count"),
    }
    if all(value is not None for value in counts.values()):
        total = sum(int(value) for value in counts.values())
        return {
            "success_rate": int(counts["success_count"]) / total,
            "collision_rate": int(counts["collision_count"]) / total,
            "timeout_rate": int(counts["timeout_count"]) / total,
            "fall_rate": int(counts["fall_count"]) / total,
            "total_episodes": float(total),
        }

    return {
        "success_rate": float(linked_batch.get("success_rate") or 0.0),
        "collision_rate": float(linked_batch.get("collision_rate") or 0.0),
        "timeout_rate": float(linked_batch.get("timeout_rate") or 0.0),
        "fall_rate": float(linked_batch.get("fall_rate") or 0.0),
        "total_episodes": float(linked_batch.get("batch_episodes") or 0.0),
    }


def _aggregate_batch_metrics(batch_rows: List[dict]) -> Dict[str, float]:
    totals = {
        "success_count": 0.0,
        "collision_count": 0.0,
        "timeout_count": 0.0,
        "fall_count": 0.0,
    }
    for row in batch_rows:
        totals["success_count"] += float(row.get("success_count") or 0.0)
        totals["collision_count"] += float(row.get("collision_count") or 0.0)
        totals["timeout_count"] += float(row.get("timeout_count") or 0.0)
        totals["fall_count"] += float(row.get("fall_count") or 0.0)

    total_episodes = sum(totals.values())
    if total_episodes <= 0:
        return {
            "success_rate": 0.0,
            "collision_rate": 0.0,
            "timeout_rate": 0.0,
            "fall_rate": 0.0,
            "total_episodes": 0.0,
        }

    return {
        "success_rate": totals["success_count"] / total_episodes,
        "collision_rate": totals["collision_count"] / total_episodes,
        "timeout_rate": totals["timeout_count"] / total_episodes,
        "fall_rate": totals["fall_count"] / total_episodes,
        "total_episodes": total_episodes,
    }


def _build_stage_heatmap(stage_rows: List[dict], *, bins: int, xy_limit: float) -> np.ndarray:
    xs = np.array([float(row["obs_x"]) for row in stage_rows], dtype=float)
    ys = np.array([float(row["obs_y"]) for row in stage_rows], dtype=float)
    xedges = np.linspace(-xy_limit, xy_limit, bins + 1)
    yedges = np.linspace(-xy_limit, xy_limit, bins + 1)
    heatmap, _, _ = np.histogram2d(xs, ys, bins=[xedges, yedges])
    if len(stage_rows) > 0:
        heatmap = heatmap / float(len(stage_rows))
    return heatmap


def _build_validation_summary(entry: dict) -> Dict[str, object]:
    evolve_row = entry["evolve_row"]
    pre_metrics = entry["pre_metrics"]
    post_metrics = entry["post_metrics"]
    post_batches = entry["post_batches"]

    target_key = _target_rate_key(evolve_row)
    target_pre = float(pre_metrics.get(target_key, 0.0))
    target_post = float(post_metrics.get(target_key, 0.0))
    success_delta = float(post_metrics["success_rate"]) - float(pre_metrics["success_rate"])
    target_reduction = target_pre - target_post

    verdict, verdict_color = _effectiveness_verdict(success_delta, target_reduction)
    post_start = min(int(row["batch_start_episode"]) for row in post_batches)
    post_end = max(int(row["batch_end_episode"]) for row in post_batches)

    return {
        "target_key": target_key,
        "target_label": RATE_LABELS[target_key],
        "target_reduction": target_reduction,
        "success_delta": success_delta,
        "verdict": verdict,
        "verdict_color": verdict_color,
        "validation_window_label": f"{_fmt_k(post_start)}-{_fmt_k(post_end)}",
        "validation_batches_label": f"{len(post_batches)} batches",
    }


def _target_rate_key(evolve_row: dict) -> str:
    failure_family = _failure_family(evolve_row.get("dominant_failure_type"))
    if failure_family == "timeout":
        return "timeout_rate"
    if failure_family == "fall":
        return "fall_rate"
    return "collision_rate"


def _effectiveness_verdict(success_delta: float, target_reduction: float) -> tuple[str, str]:
    if target_reduction >= 0.05 and success_delta >= 0.0:
        return "effective", "#0f9d58"
    if target_reduction >= 0.02 or success_delta >= 0.03:
        return "mixed", "#c48a00"
    return "weak", "#c2410c"


def _draw_diagnosis_card(axis: plt.Axes, entry: dict) -> None:
    evolve_row = entry["evolve_row"]
    pre_metrics = entry["pre_metrics"]
    linked_batch = entry["linked_batch"]

    axis.set_axis_off()
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_facecolor("#ffffff")

    axis.add_patch(
        patches.FancyBboxPatch(
            (0.02, 0.04),
            0.96,
            0.92,
            boxstyle="round,pad=0.02,rounding_size=0.04",
            linewidth=1.2,
            edgecolor=PANEL_EDGE_LIGHT,
            facecolor="#f8fbff",
            zorder=0,
        )
    )
    axis.plot([0.04, 0.04], [0.12, 0.90], color=EVOLVE_ACCENT, linewidth=3.0, solid_capstyle="round", zorder=1)

    evolve_id = int(evolve_row["evolve_id"])
    add_badge(axis, 0.08, 0.93, f"E{evolve_id}", facecolor="#284a76", fontsize=8.6)
    add_badge(
        axis,
        0.94,
        0.93,
        f"G{entry['before_stage']} -> G{entry['after_stage']}",
        facecolor="#ffffff",
        edgecolor=PANEL_EDGE_LIGHT,
        textcolor=MUTED_TEXT,
        ha="right",
        fontsize=8.0,
    )

    dominant = _pretty_failure_name(evolve_row.get("dominant_failure_type"))
    axis.text(
        0.08,
        0.83,
        dominant,
        ha="left",
        va="top",
        fontsize=12.0,
        fontweight="bold",
        color=TITLE_COLOR,
    )
    axis.text(
        0.08,
        0.75,
        _pretty_trigger_name(evolve_row.get("trigger_reason")),
        ha="left",
        va="top",
        fontsize=9.2,
        color=MUTED_TEXT,
    )

    lines = [
        f"Episode: {_fmt_k(evolve_row.get('episode_total'))}",
        f"Reliability: {evolve_row.get('diagnosis_reliability') or 'n/a'}",
        f"Samples: {_safe_int(evolve_row.get('sample_count'))}",
        f"Failure focus: {_display_text(evolve_row.get('failure_region'))}",
        f"Behavior bias: {_display_text(evolve_row.get('behavior_bias'))}",
        (
            "Pre batch: "
            f"{_fmt_k(linked_batch.get('batch_start_episode'))}-{_fmt_k(linked_batch.get('batch_end_episode'))}"
        ),
        (
            "Pre rates: "
            f"SR {_pct(pre_metrics['success_rate'])} | "
            f"C {_pct(pre_metrics['collision_rate'])} | "
            f"T {_pct(pre_metrics['timeout_rate'])}"
        ),
    ]

    y = 0.66
    for line in lines:
        axis.text(
            0.08,
            y,
            line,
            ha="left",
            va="top",
            fontsize=9.0,
            color=TEXT_COLOR,
        )
        y -= 0.085


def _draw_stage_revision_panel(
    axis: plt.Axes,
    stage_id: int,
    heatmap: np.ndarray,
    norm: PowerNorm,
    *,
    role_label: str,
    accent: str,
    episode_label: str,
) -> None:
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
    xedges = np.linspace(-0.25, 0.25, heatmap.shape[0] + 1)
    yedges = np.linspace(-0.25, 0.25, heatmap.shape[1] + 1)
    pixel_x_edges, pixel_y_edges = world_edges_to_video_edges(xedges, yedges)
    axis.pcolormesh(
        pixel_x_edges,
        pixel_y_edges,
        masked,
        cmap=HEATMAP_CMAP,
        norm=norm,
        zorder=5,
        shading="auto",
    )

    add_badge(axis, 0.02, 0.97, f"{role_label} G{stage_id}", facecolor=accent, fontsize=8.2)
    add_badge(
        axis,
        0.98,
        0.97,
        episode_label,
        facecolor="#ffffff",
        edgecolor="#d5e2f0",
        textcolor="#243447",
        ha="right",
        fontsize=7.8,
    )
    axis.text(
        0.03,
        0.11,
        "redder = more frequent obstacle placement",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=7.8,
        color="#111827",
        clip_on=True,
    )


def _draw_validation_panel(axis: plt.Axes, entry: dict) -> None:
    pre_metrics = entry["pre_metrics"]
    post_metrics = entry["post_metrics"]
    target_key = entry["target_key"]

    axis.set_facecolor(CARD_BG)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(CARD_EDGE)
    axis.spines["bottom"].set_color(CARD_EDGE)
    axis.spines["left"].set_linewidth(1.1)
    axis.spines["bottom"].set_linewidth(1.1)
    axis.grid(axis="x", alpha=0.35, linestyle="--", linewidth=0.8, color=GRID_COLOR)
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(-0.8, 3.4)

    metric_keys = ["success_rate", "collision_rate", "timeout_rate"]
    if target_key == "fall_rate":
        metric_keys[-1] = "fall_rate"

    y_positions = list(reversed(range(len(metric_keys))))
    for y_pos, metric_key in zip(y_positions, metric_keys):
        pre_value = float(pre_metrics.get(metric_key, 0.0))
        post_value = float(post_metrics.get(metric_key, 0.0))
        color = RATE_COLORS[metric_key]
        label = RATE_LABELS[metric_key]
        is_target = metric_key == target_key

        axis.plot([pre_value, post_value], [y_pos, y_pos], color="#b1bfd0", linewidth=2.2, zorder=1)
        axis.scatter(
            [pre_value],
            [y_pos],
            s=52,
            facecolor="#ffffff",
            edgecolor="#4b5563",
            linewidth=1.3,
            zorder=3,
        )
        axis.scatter(
            [post_value],
            [y_pos],
            s=62 if is_target else 54,
            facecolor=color,
            edgecolor="#ffffff",
            linewidth=1.1,
            zorder=4,
        )
        axis.text(
            1.02,
            y_pos,
            _delta_label(post_value - pre_value),
            transform=axis.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=8.7,
            color=color if is_target else MUTED_TEXT,
            fontweight="bold" if is_target else "normal",
        )

    axis.set_yticks(y_positions)
    axis.set_yticklabels(
        [
            RATE_LABELS[key] + (" *" if key == target_key else "")
            for key in metric_keys
        ],
        fontsize=9.2,
        color=TEXT_COLOR,
    )
    axis.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    axis.tick_params(axis="x", labelsize=8.7)
    axis.tick_params(axis="y", length=0)

    axis.text(
        0.02,
        0.97,
        (
            f"post window: {entry['validation_window_label']} | {entry['validation_batches_label']}\n"
            f"targeted {entry['target_label']} reduction: {_signed_pts(entry['target_reduction'])} pts"
        ),
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=8.4,
        color=MUTED_TEXT,
    )
    add_badge(
        axis,
        0.98,
        0.97,
        entry["verdict"],
        facecolor=entry["verdict_color"],
        ha="right",
        fontsize=8.0,
    )
    axis.text(
        0.02,
        0.06,
        "white = pre-diagnosis, colored = post-revision",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.8,
        color=MUTED_TEXT,
    )


def _pretty_trigger_name(trigger_reason: object) -> str:
    family = _reason_family(trigger_reason)
    if family == "first_fixed":
        return "trigger: first scheduled evolve"
    if family == "too_easy":
        return "trigger: current curriculum became too easy"
    if family == "too_hard":
        return "trigger: current curriculum became too hard"
    if family == "plateau":
        return "trigger: learning plateau"
    return f"trigger: {family}"


def _pretty_failure_name(dominant_failure_type: object) -> str:
    text = str(dominant_failure_type or "unknown").replace("_", " ").strip()
    if not text:
        return "unknown failure"
    return text


def _figure_subtitle(run_meta: dict, chosen_entries: List[dict]) -> str:
    run_name = Path(str(run_meta.get("output_dir", run_meta.get("run_id", "run")))).name
    shown = ", ".join(f"E{int(entry['evolve_row']['evolve_id'])}" for entry in chosen_entries)
    return (
        f"{run_name} | selected evolves: {shown} | "
        "revision panels show stage-level obstacle density on the real environment render"
    )


def _stage_episode_badge(meta: dict) -> str:
    if not meta:
        return "ep n/a"
    start = meta.get("episode_start")
    end = meta.get("episode_end")
    if start is None or end is None:
        return "ep n/a"
    if int(start) == int(end):
        return f"ep {_fmt_k(start)}"
    return f"{_fmt_k(start)}-{_fmt_k(end)}"


def _pct(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _signed_pts(value: float) -> str:
    return f"{100.0 * float(value):+.1f}"


def _delta_label(delta: float) -> str:
    sign = "+" if delta >= 0 else ""
    return f"{sign}{100.0 * float(delta):.1f} pts"


def _display_text(value: object) -> str:
    text = str(value or "").strip()
    return text if text else "n/a"


def _safe_int(value: object) -> str:
    if value is None or value == "":
        return "n/a"
    return str(int(float(value)))


if __name__ == "__main__":
    main()
