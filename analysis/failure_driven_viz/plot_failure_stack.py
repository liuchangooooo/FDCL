"""Plot stacked failure composition over training batches."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis.failure_driven_viz.parse_logs import default_figure_dir, ensure_parsed_dir, load_parsed_artifacts
from analysis.failure_driven_viz.style import (
    COLLISION_LINE,
    FALL_LINE,
    MUTED_TEXT,
    PANEL_EDGE_LIGHT,
    SUCCESS_FILL,
    SUCCESS_LINE,
    TIMEOUT_LINE,
    TITLE_COLOR,
    add_badge,
    configure_matplotlib,
    style_chart_axis,
)

configure_matplotlib()


def plot_failure_stack(parsed_dir: Path, output_path: Path) -> Path:
    artifacts = load_parsed_artifacts(str(parsed_dir))
    batch_rows = artifacts["batch_stats"]
    run_meta = artifacts["run_meta"]

    if not batch_rows:
        raise ValueError("No batch statistics found. Failure stack plot requires ACGS-enabled runs.")

    x = [row["batch_end_episode"] for row in batch_rows]
    success = [row["success_rate"] for row in batch_rows]
    collision = [row["collision_rate"] for row in batch_rows]
    timeout = [row["timeout_rate"] for row in batch_rows]
    fall = [row["fall_rate"] for row in batch_rows]

    fig, ax = plt.subplots(figsize=(13.0, 6.4), constrained_layout=False)
    fig.patch.set_facecolor("#ffffff")
    style_chart_axis(ax, facecolor="#f8fbff", grid_axis="y")
    ax.stackplot(
        x,
        success,
        collision,
        timeout,
        fall,
        labels=["success", "collision", "timeout", "fall"],
        colors=[SUCCESS_LINE, COLLISION_LINE, TIMEOUT_LINE, FALL_LINE],
        alpha=0.84,
    )

    for row in batch_rows:
        if row["evolve_decision"] == "triggered":
            ax.axvline(row["batch_end_episode"], linestyle="--", color="#284a76", linewidth=1.0, alpha=0.20)

    ax.set_title("Failure Composition Over Time", loc="left", pad=16, color=TITLE_COLOR)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Rate")
    ax.set_ylim(0.0, 1.0)
    add_badge(ax, 0.01, 0.98, "stacked batch outcomes", facecolor="#284a76", fontsize=8.4)
    ax.legend(loc="upper center", bbox_to_anchor=(0.72, 1.00), frameon=False, ncol=2)

    summary_text = "\n".join(
        [
            f"final seen SR = {_format_metric(run_meta.get('final_seen_success_rate'))}",
            f"validate reward = {_format_metric(run_meta.get('final_validate_reward'))}",
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

    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.12, top=0.90)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot stacked batch failure composition.")
    parser.add_argument("--run-dir", default=None, help="Experiment output directory.")
    parser.add_argument("--parsed-dir", default=None, help="Existing parsed directory.")
    parser.add_argument("--export-dir", default=None, help="Optional parsed export directory when using --run-dir.")
    parser.add_argument("--force-reparse", action="store_true", help="Regenerate parsed tables before plotting.")
    parser.add_argument("--output", default=None, help="Output PNG path.")
    args = parser.parse_args()

    parsed_dir = ensure_parsed_dir(
        run_dir=args.run_dir,
        parsed_dir=args.parsed_dir,
        export_dir=args.export_dir,
        force_reparse=args.force_reparse,
    )
    output_path = Path(args.output).expanduser().resolve() if args.output else default_figure_dir(parsed_dir) / "failure_stack.png"
    saved_path = plot_failure_stack(parsed_dir, output_path)
    print(f"Saved failure stack plot to: {saved_path}")

def _stack_subtitle(run_meta: dict) -> str:
    run_name = Path(str(run_meta.get("output_dir", run_meta.get("run_id", "run")))).name
    return f"{run_name} | stacked success and failure rates across curriculum batches"


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
