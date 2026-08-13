"""Export paper-ready Push-T figures with a Nature-style contract.

The script keeps the existing analysis pipeline as the source of truth and only
adds a publication export layer:

1. Push-T final B/M/U/D success-rate bars.
2. Static-vs-evolved obstacle distribution heatmaps.
3. Failure diagnosis -> generator revision -> validation evidence chain.
4. Method schematic for failure-conditioned generator revision.

Outputs are written as SVG/PDF/TIFF/PNG so the SVG/PDF can be edited for the
manuscript while the PNG/TIFF are useful for quick previews and submission.
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
import json
from pathlib import Path
from typing import Callable, Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.patches import FancyArrowPatch

from analysis.failure_driven_viz.parse_logs import ensure_parsed_dir
from analysis.failure_driven_viz.plot_compare_heatmap import plot_compare_heatmap
from analysis.failure_driven_viz.plot_evolve_evidence_chain import plot_evolve_evidence_chain
from analysis.failure_driven_viz.plot_final_eval_bars import (
    collect_results,
    plot_grouped_bars,
    summarize_results,
    write_csv,
    write_markdown,
)


EXPORT_FORMATS = ("svg", "pdf", "tiff", "png")

PALETTE_NMI = {
    "baseline_dark": "#484878",
    "baseline_mid": "#7884B4",
    "baseline_soft": "#B4C0E4",
    "ours_soft": "#E4CCD8",
    "ours_main": "#C95B70",
    "bg_lilac": "#F2F2FA",
    "bg_aqua": "#EEF7F7",
    "bg_peach": "#F7F0EA",
    "neutral_light": "#D8D8D8",
    "neutral_mid": "#8A8A8A",
    "neutral_dark": "#404040",
    "text": "#202124",
    "muted": "#686D76",
    "delta_up": "#2E9E44",
    "delta_down": "#E53935",
}


def apply_nature_style() -> None:
    """Apply compact journal-style matplotlib defaults.

    The first three rcParams are mandatory for editable SVG text.
    """

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "font.size": 8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "axes.labelsize": 8,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.2,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def save_figure_all_formats(fig: plt.Figure, stem: Path, *, dpi: int = 600) -> List[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    saved_paths: List[Path] = []
    for suffix in EXPORT_FORMATS:
        path = stem.with_suffix(f".{suffix}")
        if suffix == "tiff":
            fig.savefig(path, dpi=dpi, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
        elif suffix == "png":
            fig.savefig(path, dpi=300, bbox_inches="tight")
        else:
            fig.savefig(path, bbox_inches="tight")
        saved_paths.append(path)
    return saved_paths


def export_existing_plot(
    stem: Path,
    plotter: Callable[[Path], Path],
    *,
    formats: Iterable[str] = EXPORT_FORMATS,
) -> List[Path]:
    saved_paths: List[Path] = []
    stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in formats:
        path = stem.with_suffix(f".{suffix}")
        plotter(path)
        saved_paths.append(path)
    return saved_paths


def export_results_bars(final_eval_root: Path, output_dir: Path) -> Dict[str, List[str]]:
    summary = summarize_results(collect_results(final_eval_root))
    write_csv(summary, output_dir / "source_push_t_final_success.csv")
    write_markdown(summary, output_dir / "source_push_t_final_success.md")

    def _plot(path: Path) -> Path:
        apply_nature_style()
        return plot_grouped_bars(
            summary=summary,
            output_path=path,
            title="Push-T Generalization to Unseen Obstacles",
            subtitle="Policy-only evaluation with best seen-validation checkpoints; bars show mean +/- s.d. across 3 seeds.",
        )

    paths = export_existing_plot(output_dir / "fig1_push_t_results_bars", _plot)
    return {"figure": [str(path) for path in paths]}


def export_distribution_heatmap(
    static_run_dir: Path,
    evolve_run_dir: Path,
    output_dir: Path,
    *,
    force_reparse: bool = False,
) -> Dict[str, List[str]]:
    static_parsed = ensure_parsed_dir(run_dir=str(static_run_dir), force_reparse=force_reparse)
    evolve_parsed = ensure_parsed_dir(run_dir=str(evolve_run_dir), force_reparse=force_reparse)

    def _plot(path: Path) -> Path:
        apply_nature_style()
        return plot_compare_heatmap(
            parsed_dir_a=static_parsed,
            parsed_dir_b=evolve_parsed,
            output_path=path,
            label_a="LLM-Static",
            label_b="LLM-Evolve",
            num_phases=3,
            bins=28,
            xy_limit=0.25,
            normalize=True,
            coordinate_frame="absolute",
        )

    paths = export_existing_plot(output_dir / "fig2_generator_distribution_heatmap", _plot)
    return {
        "figure": [str(path) for path in paths],
        "parsed_static": [str(static_parsed)],
        "parsed_evolve": [str(evolve_parsed)],
    }


def export_failure_chain(
    evolve_run_dir: Path,
    output_dir: Path,
    *,
    force_reparse: bool = False,
) -> Dict[str, List[str]]:
    evolve_parsed = ensure_parsed_dir(run_dir=str(evolve_run_dir), force_reparse=force_reparse)

    def _plot(path: Path) -> Path:
        apply_nature_style()
        return plot_evolve_evidence_chain(
            parsed_dir=evolve_parsed,
            output_path=path,
            evolve_round_indices=None,
            max_rounds=3,
            bins=28,
            xy_limit=0.25,
        )

    paths = export_existing_plot(output_dir / "fig3_failure_revision_chain", _plot)
    return {
        "figure": [str(path) for path in paths],
        "parsed_evolve": [str(evolve_parsed)],
    }


def export_method_schematic(output_dir: Path) -> Dict[str, List[str]]:
    apply_nature_style()
    fig = plot_method_schematic()
    paths = save_figure_all_formats(fig, output_dir / "fig4_method_schematic")
    plt.close(fig)
    return {"figure": [str(path) for path in paths]}


def plot_method_schematic() -> plt.Figure:
    """Draw the failure-conditioned generator revision pipeline."""

    fig = plt.figure(figsize=(7.2, 4.35), constrained_layout=False)
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(2, 5, height_ratios=[1.65, 1.0], hspace=0.22, wspace=0.12)

    ax = fig.add_subplot(gs[0, :])
    ax.set_axis_off()
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)

    stage_specs = [
        {
            "x": 0.02,
            "w": 0.18,
            "title": "Policy rollouts",
            "body": "Current policy\ninteracts with sampled\ntraining layouts",
            "face": PALETTE_NMI["bg_aqua"],
            "edge": PALETTE_NMI["baseline_mid"],
        },
        {
            "x": 0.25,
            "w": 0.18,
            "title": "Failure diagnosis",
            "body": "success, collision,\ntimeout, failure\nregion, bias",
            "face": PALETTE_NMI["bg_lilac"],
            "edge": PALETTE_NMI["baseline_dark"],
        },
        {
            "x": 0.48,
            "w": 0.18,
            "title": "Generator revision",
            "body": "LLM revises the\nprogrammatic obstacle\ngenerator",
            "face": PALETTE_NMI["bg_peach"],
            "edge": PALETTE_NMI["ours_main"],
        },
        {
            "x": 0.71,
            "w": 0.25,
            "title": "Revised training distribution",
            "body": "Validity-checked generator\nsamples feasible but\nfailure-relevant obstacles",
            "face": "#FFF6F7",
            "edge": PALETTE_NMI["ours_main"],
        },
    ]

    for idx, spec in enumerate(stage_specs):
        _draw_round_box(
            ax,
            spec["x"],
            0.27,
            spec["w"],
            0.56,
            facecolor=spec["face"],
            edgecolor=spec["edge"],
        )
        ax.text(
            spec["x"] + 0.02,
            0.74,
            spec["title"],
            fontsize=8.4,
            fontweight="bold",
            color=PALETTE_NMI["text"],
            ha="left",
            va="center",
        )
        ax.text(
            spec["x"] + 0.02,
            0.54,
            spec["body"],
            fontsize=7.4,
            color=PALETTE_NMI["muted"],
            ha="left",
            va="center",
            linespacing=1.25,
        )
        if idx < len(stage_specs) - 1:
            _arrow(
                ax,
                spec["x"] + spec["w"] + 0.012,
                0.55,
                stage_specs[idx + 1]["x"] - 0.012,
                0.55,
                color=PALETTE_NMI["neutral_dark"],
            )

    _draw_rollout_icon(ax, 0.055, 0.35, scale=0.9)
    _draw_failure_icon(ax, 0.287, 0.34, scale=0.9)
    _draw_code_icon(ax, 0.520, 0.33, scale=0.88)
    _draw_distribution_icon(ax, 0.775, 0.35, scale=0.88)

    _arrow(ax, 0.84, 0.25, 0.115, 0.24, color=PALETTE_NMI["ours_main"], curved=True)
    ax.text(
        0.50,
        0.15,
        "closed-loop training-environment design",
        fontsize=8.0,
        fontweight="bold",
        color=PALETTE_NMI["ours_main"],
        ha="center",
        va="center",
    )

    fig.text(
        0.03,
        0.965,
        "a",
        fontsize=11,
        fontweight="bold",
        ha="left",
        va="top",
        color=PALETTE_NMI["text"],
    )
    fig.text(
        0.08,
        0.965,
        "Failure-conditioned obstacle generator revision",
        fontsize=11.5,
        fontweight="bold",
        ha="left",
        va="top",
        color=PALETTE_NMI["text"],
    )
    fig.text(
        0.08,
        0.925,
        "Policy failures identify missing training challenges; a constrained revision operator updates the generator rather than the policy.",
        fontsize=7.8,
        ha="left",
        va="top",
        color=PALETTE_NMI["muted"],
    )

    ax_b = fig.add_subplot(gs[1, 0:2])
    ax_c = fig.add_subplot(gs[1, 2:5])
    _draw_objective_panel(ax_b)
    _draw_protocol_panel(ax_c)
    fig.subplots_adjust(left=0.045, right=0.985, top=0.88, bottom=0.08)
    return fig


def _draw_objective_panel(ax: plt.Axes) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _panel_label(ax, "b")
    _draw_round_box(ax, 0.03, 0.08, 0.92, 0.80, facecolor="#FAFAFC", edgecolor="#D7D7E6")
    ax.text(
        0.08,
        0.78,
        "Revision target",
        fontsize=8.2,
        fontweight="bold",
        ha="left",
        va="center",
        color=PALETTE_NMI["text"],
    )
    terms = [
        ("Failure-relevant", "+", PALETTE_NMI["baseline_dark"]),
        ("Feasible", "+", PALETTE_NMI["delta_up"]),
        ("Non-trivial", "+", PALETTE_NMI["ours_main"]),
    ]
    y = 0.58
    for name, sign, color in terms:
        ax.text(0.11, y, sign, fontsize=10, fontweight="bold", color=color, ha="center", va="center")
        ax.text(0.18, y, name, fontsize=7.5, color=PALETTE_NMI["text"], ha="left", va="center")
        y -= 0.18
    ax.text(
        0.08,
        0.13,
        "avoid trivial layouts and impossible scenes",
        fontsize=6.9,
        color=PALETTE_NMI["muted"],
        ha="left",
        va="center",
    )


def _draw_protocol_panel(ax: plt.Axes) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _panel_label(ax, "c")
    _draw_round_box(ax, 0.02, 0.08, 0.95, 0.80, facecolor="#FAFAFC", edgecolor="#D7D7E6")
    ax.text(
        0.06,
        0.78,
        "Train-test obstacle uncertainty",
        fontsize=8.2,
        fontweight="bold",
        ha="left",
        va="center",
        color=PALETTE_NMI["text"],
    )
    xs = [0.13, 0.35, 0.57, 0.79]
    labels = ["B\nbig", "M\nmultiple", "U\nnon-convex", "D\ndynamic"]
    for x, label in zip(xs, labels):
        ax.add_patch(
            patches.Circle(
                (x, 0.45),
                0.075,
                facecolor="#FFFFFF",
                edgecolor=PALETTE_NMI["baseline_mid"],
                linewidth=0.9,
            )
        )
        ax.text(x, 0.45, label, ha="center", va="center", fontsize=7.0, color=PALETTE_NMI["text"], linespacing=1.0)
    ax.text(
        0.06,
        0.17,
        "Evaluate zero-shot generalization on obstacle families unseen during training.",
        fontsize=6.9,
        color=PALETTE_NMI["muted"],
        ha="left",
        va="center",
    )


def _draw_round_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    facecolor: str,
    edgecolor: str,
) -> None:
    ax.add_patch(
        patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.025",
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=1.0,
        )
    )


def _arrow(
    ax: plt.Axes,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    *,
    color: str,
    curved: bool = False,
) -> None:
    connectionstyle = "arc3,rad=-0.38" if curved else "arc3,rad=0.0"
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.0,
            color=color,
            connectionstyle=connectionstyle,
            shrinkA=0,
            shrinkB=0,
        )
    )


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.02,
        0.98,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        ha="left",
        va="top",
        color=PALETTE_NMI["text"],
    )


def _draw_rollout_icon(ax: plt.Axes, x: float, y: float, *, scale: float) -> None:
    ax.plot([x, x + 0.05 * scale, x + 0.11 * scale], [y, y + 0.06 * scale, y + 0.02 * scale], color="#505A7A", lw=1.2)
    ax.scatter([x], [y], s=16, color="#3B8A5B", zorder=5)
    ax.scatter([x + 0.11 * scale], [y + 0.02 * scale], s=22, marker="*", color=PALETTE_NMI["ours_main"], zorder=5)
    ax.add_patch(patches.Rectangle((x + 0.055 * scale, y + 0.02 * scale), 0.035 * scale, 0.035 * scale, angle=18, facecolor="#E28B8E", edgecolor="none"))


def _draw_failure_icon(ax: plt.Axes, x: float, y: float, *, scale: float) -> None:
    for i, h in enumerate([0.06, 0.11, 0.04]):
        ax.add_patch(
            patches.Rectangle(
                (x + i * 0.037 * scale, y),
                0.022 * scale,
                h * scale,
                facecolor=[PALETTE_NMI["baseline_soft"], PALETTE_NMI["ours_soft"], PALETTE_NMI["neutral_light"]][i],
                edgecolor="white",
                linewidth=0.4,
            )
        )
    ax.plot([x - 0.005 * scale, x + 0.12 * scale], [y - 0.006 * scale, y - 0.006 * scale], color=PALETTE_NMI["neutral_mid"], lw=0.7)
    ax.text(x + 0.06 * scale, y + 0.13 * scale, "F", fontsize=7, fontweight="bold", ha="center", color=PALETTE_NMI["baseline_dark"])


def _draw_code_icon(ax: plt.Axes, x: float, y: float, *, scale: float) -> None:
    ax.text(x, y + 0.08 * scale, "g_t", fontsize=8.4, fontweight="bold", color=PALETTE_NMI["baseline_dark"], ha="left")
    ax.text(x + 0.045 * scale, y + 0.08 * scale, "->", fontsize=8.0, color=PALETTE_NMI["neutral_mid"], ha="left")
    ax.text(x + 0.083 * scale, y + 0.08 * scale, "g_{t+1}", fontsize=8.4, fontweight="bold", color=PALETTE_NMI["ours_main"], ha="left")
    for dy in [0.045, 0.018, -0.009]:
        ax.plot([x + 0.01 * scale, x + 0.125 * scale], [y + dy * scale, y + dy * scale], color=PALETTE_NMI["neutral_mid"], lw=0.7, alpha=0.7)


def _draw_distribution_icon(ax: plt.Axes, x: float, y: float, *, scale: float) -> None:
    ax.add_patch(patches.Circle((x + 0.035 * scale, y + 0.07 * scale), 0.018 * scale, color=PALETTE_NMI["ours_soft"]))
    ax.add_patch(patches.Circle((x + 0.085 * scale, y + 0.05 * scale), 0.016 * scale, color=PALETTE_NMI["baseline_soft"]))
    ax.add_patch(patches.Circle((x + 0.12 * scale, y + 0.09 * scale), 0.020 * scale, color="#F0A4AA"))
    ax.add_patch(patches.Rectangle((x + 0.052 * scale, y + 0.02 * scale), 0.032 * scale, 0.032 * scale, facecolor="#D36B72", edgecolor="none"))


def write_figure_contract(output_dir: Path) -> Path:
    contract = {
        "backend": "Python/matplotlib",
        "target_output": ["svg", "pdf", "tiff", "png"],
        "figures": {
            "fig1_push_t_results_bars": {
                "conclusion": "Failure-conditioned generator revision improves zero-shot Push-T success across unseen B/M/U/D obstacle families.",
                "archetype": "quantitative grid",
                "source_data": "final_eval_bestckpt benchmark_summary.json across three seeds",
            },
            "fig2_generator_distribution_heatmap": {
                "conclusion": "LLM-Evolve changes the training obstacle distribution over progress, unlike a static generator.",
                "archetype": "asymmetric mixed-modality figure",
                "source_data": "parsed Push-T rollout obstacle snapshots",
            },
            "fig3_failure_revision_chain": {
                "conclusion": "Policy failure evidence is linked to subsequent generator revisions and post-revision validation changes.",
                "archetype": "schematic-led composite",
                "source_data": "parsed ACGS batch statistics and evolve prompts",
            },
            "fig4_method_schematic": {
                "conclusion": "The method closes the loop between policy learning and programmatic training-environment design.",
                "archetype": "schematic-led composite",
                "source_data": "method specification",
            },
        },
        "reviewer_risk": [
            "Heatmaps are representative seed-level evidence, while success bars aggregate three seeds.",
            "Failure-chain panels should be described as deployed revisions unless learning-value acceptance is added later.",
            "Static/evolve heatmaps use normalized occupancy and should not be interpreted as absolute episode counts.",
        ],
    }
    output_path = output_dir / "figure_contract.json"
    output_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Nature-style Push-T manuscript figures.")
    parser.add_argument("--output-dir", default="data/outputs/paper_figures/nature_push_t", help="Figure export directory.")
    parser.add_argument("--final-eval-root", default="data/outputs/2026.05.02/final_eval_bestckpt", help="Root with final benchmark_summary.json files.")
    parser.add_argument("--static-run", default="data/outputs/2026.04.29/llm_static_s0", help="Representative LLM-static training run.")
    parser.add_argument("--evolve-run", default="data/outputs/2026.04.29/llm_evolve_s0", help="Representative LLM-evolve training run.")
    parser.add_argument("--force-reparse", action="store_true", help="Regenerate parsed run tables before plotting.")
    args = parser.parse_args()

    apply_nature_style()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Dict[str, List[str]]] = {}
    manifest["results_bars"] = export_results_bars(Path(args.final_eval_root).expanduser().resolve(), output_dir)
    manifest["distribution_heatmap"] = export_distribution_heatmap(
        Path(args.static_run).expanduser().resolve(),
        Path(args.evolve_run).expanduser().resolve(),
        output_dir,
        force_reparse=args.force_reparse,
    )
    manifest["failure_chain"] = export_failure_chain(
        Path(args.evolve_run).expanduser().resolve(),
        output_dir,
        force_reparse=args.force_reparse,
    )
    manifest["method_schematic"] = export_method_schematic(output_dir)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    contract_path = write_figure_contract(output_dir)

    print(f"Saved Nature-style figure exports to: {output_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Figure contract: {contract_path}")
    for group, payload in manifest.items():
        for path in payload.get("figure", []):
            print(f"{group}: {path}")


if __name__ == "__main__":
    main()
