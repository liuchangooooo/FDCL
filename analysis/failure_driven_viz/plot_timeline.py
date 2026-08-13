"""Plot a failure-driven training timeline from parsed batch statistics."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

from analysis.failure_driven_viz.parse_logs import default_figure_dir, ensure_parsed_dir, load_parsed_artifacts
from analysis.failure_driven_viz.style import (
    CARD_BG,
    CARD_EDGE,
    COLLISION_LINE,
    FALL_LINE,
    GRID_COLOR,
    MUTED_TEXT,
    PANEL_EDGE_LIGHT,
    PLATEAU_LINE,
    SUCCESS_FILL,
    SUCCESS_LINE,
    TIMEOUT_LINE,
    TITLE_COLOR,
    add_badge,
    configure_matplotlib,
    style_chart_axis,
)

configure_matplotlib()

REASON_COLOR_MAP = {
    "first_fixed": "#6b7280",
    "too_hard": COLLISION_LINE,
    "plateau": PLATEAU_LINE,
    "too_easy": SUCCESS_LINE,
    "unknown": "#444444",
}

FAILURE_MARKER_MAP = {
    "collision": "s",
    "fall": "D",
    "timeout": "o",
    "unknown": "^",
}


def plot_timeline(
    parsed_dir: Path,
    output_path: Path,
    annotate_triggers: bool = True,
    smoothing_window: int = 3,
    show_event_strip: bool = True,
) -> Path:
    artifacts = load_parsed_artifacts(str(parsed_dir))
    batch_rows = artifacts["batch_stats"]
    run_meta = artifacts["run_meta"]

    if not batch_rows:
        raise ValueError("No batch statistics found. Timeline plot requires ACGS-enabled runs.")

    x = np.asarray([float(row["batch_end_episode"]) for row in batch_rows], dtype=float)
    success = np.asarray([float(row["success_rate"]) for row in batch_rows], dtype=float)
    collision = np.asarray([float(row["collision_rate"]) for row in batch_rows], dtype=float)
    timeout = np.asarray([float(row["timeout_rate"]) for row in batch_rows], dtype=float)
    fall = np.asarray([float(row["fall_rate"]) for row in batch_rows], dtype=float)

    success_smooth = _moving_average(success, smoothing_window)
    collision_smooth = _moving_average(collision, smoothing_window)
    timeout_smooth = _moving_average(timeout, smoothing_window)
    fall_smooth = _moving_average(fall, smoothing_window)

    trigger_rows = [row for row in batch_rows if row["evolve_decision"] == "triggered"]

    if show_event_strip and trigger_rows:
        fig, (ax, ax_events) = plt.subplots(
            2,
            1,
            figsize=(13.4, 7.8),
            sharex=True,
            constrained_layout=False,
            gridspec_kw={"height_ratios": [5.0, 1.35]},
        )
    else:
        fig, ax = plt.subplots(figsize=(13.4, 6.8), constrained_layout=False)
        ax_events = None
    fig.patch.set_facecolor("#ffffff")

    low_threshold = run_meta.get("configured_success_rate_low")
    high_threshold = run_meta.get("configured_success_rate_high")
    style_chart_axis(ax, facecolor=CARD_BG, grid_axis="both")
    _draw_threshold_bands(ax, low_threshold, high_threshold)

    ax.fill_between(x, 0.0, success_smooth, color=SUCCESS_FILL, alpha=0.28, zorder=1)
    ax.plot(x, success, linewidth=1.0, alpha=0.20, color=SUCCESS_LINE)
    ax.plot(x, success_smooth, label="success", linewidth=3.0, color=SUCCESS_LINE, solid_capstyle="round", zorder=3)

    ax.plot(x, collision_smooth, label="collision", linewidth=2.0, color=COLLISION_LINE, alpha=0.95, zorder=3)
    ax.plot(x, timeout_smooth, label="timeout", linewidth=2.0, color=TIMEOUT_LINE, alpha=0.95, zorder=3)
    ax.plot(x, fall_smooth, label="fall", linewidth=2.0, color=FALL_LINE, alpha=0.95, zorder=3)

    for row in trigger_rows:
        episode = row["batch_end_episode"]
        success_rate = float(row["success_rate"])
        reason_family = _reason_family(row.get("evolve_trigger_reason"))
        failure_family = _failure_family(row.get("dominant_failure_type"))
        color = REASON_COLOR_MAP.get(reason_family, REASON_COLOR_MAP["unknown"])
        marker = FAILURE_MARKER_MAP.get(failure_family, FAILURE_MARKER_MAP["unknown"])

        ax.axvline(episode, linestyle="-", color=color, linewidth=0.9, alpha=0.10, zorder=0)
        ax.scatter(
            episode,
            success_rate,
            s=52,
            marker=marker,
            facecolor=color,
            edgecolor="white",
            linewidth=0.9,
            zorder=5,
        )

    ax.set_title("Failure-Driven Training Timeline", loc="left", pad=16, fontweight="bold", color=TITLE_COLOR)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Rate")
    ax.set_ylim(0.0, 1.02)
    ax.set_xlim(float(x.min()), float(x.max()))
    ax.xaxis.set_major_formatter(FuncFormatter(_format_episode_tick))

    line_legend = ax.legend(loc="upper left", ncol=2, frameon=False)
    ax.add_artist(line_legend)

    summary_text = "\n".join(
        [
            f"final seen SR = {_format_metric(run_meta.get('final_seen_success_rate'))}",
            f"final validate reward = {_format_metric(run_meta.get('final_validate_reward'))}",
            f"evolves = {_format_metric(run_meta.get('final_evolve_count'), integer=True)}",
        ]
    )
    ax.text(
        0.995,
        0.98,
        summary_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color=TITLE_COLOR,
        bbox={
            "boxstyle": "round,pad=0.38,rounding_size=0.2",
            "facecolor": "#ffffff",
            "edgecolor": PANEL_EDGE_LIGHT,
            "alpha": 0.97,
        },
    )
    add_badge(ax, 0.01, 0.98, "batch metrics", facecolor="#284a76", fontsize=8.4)

    if ax_events is not None:
        style_chart_axis(ax_events, facecolor="#f3f8fe", grid_axis="x")
        _draw_event_strip(ax_events, trigger_rows, annotate_triggers=annotate_triggers)
        ax_events.xaxis.set_major_formatter(FuncFormatter(_format_episode_tick))
        event_handles = _build_event_legend_handles()
        ax_events.legend(
            handles=event_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.26),
            ncol=4,
            frameon=False,
            fontsize=9,
            columnspacing=1.2,
            handletextpad=0.5,
        )

    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.10 if ax_events is not None else 0.11, top=0.92, hspace=0.22)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the failure-driven training timeline.")
    parser.add_argument("--run-dir", default=None, help="Experiment output directory.")
    parser.add_argument("--parsed-dir", default=None, help="Existing parsed directory.")
    parser.add_argument("--export-dir", default=None, help="Optional parsed export directory when using --run-dir.")
    parser.add_argument("--force-reparse", action="store_true", help="Regenerate parsed tables before plotting.")
    parser.add_argument("--output", default=None, help="Output PNG path.")
    parser.add_argument("--no-annotate", action="store_true", help="Disable trigger annotations.")
    parser.add_argument("--smooth-window", type=int, default=3, help="Moving-average window for plotted rates.")
    parser.add_argument("--no-event-strip", action="store_true", help="Disable the evolve event strip.")
    args = parser.parse_args()

    parsed_dir = ensure_parsed_dir(
        run_dir=args.run_dir,
        parsed_dir=args.parsed_dir,
        export_dir=args.export_dir,
        force_reparse=args.force_reparse,
    )
    output_path = Path(args.output).expanduser().resolve() if args.output else default_figure_dir(parsed_dir) / "timeline.png"
    saved_path = plot_timeline(
        parsed_dir,
        output_path,
        annotate_triggers=not args.no_annotate,
        smoothing_window=args.smooth_window,
        show_event_strip=not args.no_event_strip,
    )
    print(f"Saved timeline plot to: {saved_path}")


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) <= 2:
        return values.copy()

    window = min(window, len(values))
    if window % 2 == 0:
        window += 1 if window < len(values) else 0

    pad = window // 2
    padded = np.pad(values, pad_width=pad, mode="edge")
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def _draw_threshold_bands(ax: plt.Axes, low_threshold: float | None, high_threshold: float | None) -> None:
    if low_threshold is not None:
        ax.axhspan(0.0, low_threshold, facecolor="#ffe2e3", alpha=0.75, zorder=-2)
        ax.axhline(low_threshold, color=COLLISION_LINE, linestyle="--", linewidth=1.2, alpha=0.88)
        ax.text(
            0.995,
            min(0.985, low_threshold + 0.02),
            "too hard zone",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="bottom",
            fontsize=9,
            color=COLLISION_LINE,
        )

    if low_threshold is not None and high_threshold is not None and high_threshold > low_threshold:
        ax.axhspan(low_threshold, high_threshold, facecolor="#eef6ff", alpha=0.78, zorder=-2)
        ax.text(
            0.995,
            (low_threshold + high_threshold) / 2.0,
            "working zone",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=9,
            color="#345b86",
        )

    if high_threshold is not None:
        ax.axhspan(high_threshold, 1.0, facecolor="#ddecff", alpha=0.72, zorder=-2)
        ax.axhline(high_threshold, color=SUCCESS_LINE, linestyle="--", linewidth=1.2, alpha=0.88)
        ax.text(
            0.995,
            min(0.985, high_threshold + 0.02),
            "too easy zone",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="bottom",
            fontsize=9,
            color=SUCCESS_LINE,
        )


def _draw_event_strip(ax: plt.Axes, trigger_rows: list[dict], annotate_triggers: bool) -> None:
    ax.axhline(0.0, color="#7f98b8", linewidth=1.2, alpha=0.9)

    previous_annotation_key: tuple[str, str] | None = None
    for index, row in enumerate(trigger_rows):
        episode = float(row["batch_end_episode"])
        reason_family = _reason_family(row.get("evolve_trigger_reason"))
        failure_family = _failure_family(row.get("dominant_failure_type"))
        color = REASON_COLOR_MAP.get(reason_family, REASON_COLOR_MAP["unknown"])
        marker = FAILURE_MARKER_MAP.get(failure_family, FAILURE_MARKER_MAP["unknown"])
        label_key = (reason_family, failure_family)

        ax.scatter(
            episode,
            0.0,
            s=62,
            marker=marker,
            facecolor=color,
            edgecolor="#17314f",
            linewidth=0.6,
            zorder=3,
        )

        if annotate_triggers and (index == 0 or label_key != previous_annotation_key):
            y_offset = 0.42 if index % 2 == 0 else -0.46
            va = "bottom" if y_offset > 0 else "top"
            ax.text(
                episode,
                y_offset,
                f"{reason_family}\n{failure_family}",
                ha="center",
                va=va,
                fontsize=8,
                color=MUTED_TEXT,
            )
            previous_annotation_key = label_key

    ax.set_ylim(-0.85, 0.85)
    ax.set_yticks([])
    ax.set_ylabel("evolve", fontsize=10)
    ax.grid(axis="x", alpha=0.18, linestyle="--", color=GRID_COLOR)
    ax.spines["left"].set_visible(False)


def _build_event_legend_handles() -> list[Line2D]:
    handles: list[Line2D] = []
    for reason, color in REASON_COLOR_MAP.items():
        if reason == "unknown":
            continue
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=color,
                markeredgecolor="none",
                markersize=8,
                label=f"reason: {reason}",
            )
        )

    for failure, marker in FAILURE_MARKER_MAP.items():
        if failure == "unknown":
            continue
        handles.append(
            Line2D(
                [0],
                [0],
                marker=marker,
                color="black",
                markerfacecolor="white",
                markeredgecolor="black",
                linewidth=0,
                markersize=8,
                label=f"failure: {failure}",
            )
        )

    return handles


def _reason_family(trigger_reason: object) -> str:
    if not trigger_reason:
        return "unknown"
    reason_text = str(trigger_reason)
    return reason_text.split("(", 1)[0].strip() or "unknown"


def _failure_family(dominant_failure_type: object) -> str:
    if not dominant_failure_type:
        return "unknown"
    failure_text = str(dominant_failure_type).lower()
    if "timeout" in failure_text:
        return "timeout"
    if "fall" in failure_text:
        return "fall"
    if "collision" in failure_text:
        return "collision"
    return "unknown"


def _format_episode_tick(value: float, _: float) -> str:
    if value >= 1000:
        return f"{int(round(value / 1000.0))}k"
    return str(int(value))


def _timeline_subtitle(run_meta: dict) -> str:
    run_name = Path(str(run_meta.get("output_dir", run_meta.get("run_id", "run")))).name
    batch_size = run_meta.get("batch_size_episodes")
    threshold_low = run_meta.get("configured_success_rate_low")
    threshold_high = run_meta.get("configured_success_rate_high")
    return (
        f"{run_name} | batch={_format_metric(batch_size, integer=True)} eps | "
        f"success thresholds=({_format_metric(threshold_low)}, {_format_metric(threshold_high)})"
    )


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
