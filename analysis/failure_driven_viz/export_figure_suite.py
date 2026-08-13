"""Export a named bundle of failure-driven visualization figures.

This script is meant to be the "one-click" entrypoint for generating the
figures we have discussed for papers / reports:

- single-run mechanism figures: timeline, failure stack, heatmap, cases
- two-run comparison figures: compare_heatmap, compare_cases
- optional final policy comparison when benchmark summaries are provided

The exported images are grouped into one directory whose name is derived from
the experiment run names, so the generated assets are easy to trace back to the
source experiments.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

from analysis.failure_driven_viz.parse_logs import (
    ensure_parsed_dir,
    load_parsed_artifacts,
)
from analysis.failure_driven_viz.plot_cases import plot_cases
from analysis.failure_driven_viz.plot_compare_cases import plot_compare_cases
from analysis.failure_driven_viz.plot_compare_heatmap import plot_compare_heatmap
from analysis.failure_driven_viz.plot_failure_stack import plot_failure_stack
from analysis.failure_driven_viz.plot_heatmap import plot_heatmap
from analysis.failure_driven_viz.plot_evolve_evidence_chain import plot_evolve_evidence_chain
from analysis.failure_driven_viz.plot_policy_eval_compare import (
    load_method,
    plot_policy_eval_compare,
)
from analysis.failure_driven_viz.plot_timeline import plot_timeline


def export_figure_suite(
    run_a: str,
    *,
    run_b: str | None = None,
    label_a: str | None = None,
    label_b: str | None = None,
    benchmark_a: str | None = None,
    benchmark_b: str | None = None,
    output_root: str | None = None,
    force_reparse: bool = False,
    include_seen_in_policy_compare: bool = True,
) -> Dict[str, object]:
    run_a_path = Path(run_a).expanduser().resolve()
    run_b_path = Path(run_b).expanduser().resolve() if run_b else None

    parsed_a = ensure_parsed_dir(run_dir=str(run_a_path), force_reparse=force_reparse)
    parsed_b = ensure_parsed_dir(run_dir=str(run_b_path), force_reparse=force_reparse) if run_b_path else None

    title_a = run_title(run_a_path)
    title_b = run_title(run_b_path) if run_b_path else None
    bundle_name = build_bundle_name(run_a_path, run_b_path)
    bundle_dir = resolve_bundle_dir(run_a_path, run_b_path, output_root=output_root) / bundle_name
    bundle_dir.mkdir(parents=True, exist_ok=True)

    outputs: Dict[str, str] = {}
    notes: List[str] = []

    outputs.update(
        export_single_run_figures(
            run_path=run_a_path,
            parsed_dir=parsed_a,
            output_dir=bundle_dir,
            title=title_a,
            notes=notes,
        )
    )

    if run_b_path is not None and parsed_b is not None and title_b is not None:
        outputs.update(
            export_single_run_figures(
                run_path=run_b_path,
                parsed_dir=parsed_b,
                output_dir=bundle_dir,
                title=title_b,
                notes=notes,
            )
        )
        outputs.update(
            export_comparison_figures(
                parsed_a=parsed_a,
                parsed_b=parsed_b,
                output_dir=bundle_dir,
                bundle_name=bundle_name,
                label_a=label_a or default_label_for_run(run_a_path),
                label_b=label_b or default_label_for_run(run_b_path),
            )
        )

    if run_b_path is not None:
        policy_outputs = export_policy_compare_if_available(
            run_a_path=run_a_path,
            run_b_path=run_b_path,
            output_dir=bundle_dir,
            bundle_name=bundle_name,
            label_a=label_a or default_label_for_run(run_a_path),
            label_b=label_b or default_label_for_run(run_b_path),
            benchmark_a=benchmark_a,
            benchmark_b=benchmark_b,
            include_seen=include_seen_in_policy_compare,
            notes=notes,
        )
        outputs.update(policy_outputs)

    manifest = {
        "bundle_name": bundle_name,
        "bundle_dir": str(bundle_dir),
        "run_a": str(run_a_path),
        "run_b": str(run_b_path) if run_b_path else None,
        "parsed_a": str(parsed_a),
        "parsed_b": str(parsed_b) if parsed_b else None,
        "outputs": outputs,
        "notes": notes,
    }
    manifest_path = bundle_dir / f"{bundle_name}_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    outputs["manifest"] = str(manifest_path)
    return manifest


def export_single_run_figures(
    *,
    run_path: Path,
    parsed_dir: Path,
    output_dir: Path,
    title: str,
    notes: List[str],
) -> Dict[str, str]:
    outputs: Dict[str, str] = {}
    artifacts = load_parsed_artifacts(str(parsed_dir))
    batch_rows = artifacts["batch_stats"]
    snapshot_rows = artifacts["layout_snapshots"]
    obstacle_rows = artifacts["obstacle_points"]

    if batch_rows:
        timeline_path = output_dir / f"{title}_timeline.png"
        plot_timeline(parsed_dir, timeline_path)
        outputs["timeline"] = str(timeline_path)

        failure_stack_path = output_dir / f"{title}_failure_stack.png"
        plot_failure_stack(parsed_dir, failure_stack_path)
        outputs["failure_stack"] = str(failure_stack_path)
    else:
        notes.append(f"{title}: skipped timeline / failure_stack because batch_stats is empty.")

    if snapshot_rows and obstacle_rows:
        heatmap_path = output_dir / f"{title}_heatmap.png"
        plot_heatmap(parsed_dir, heatmap_path, coordinate_frame="absolute")
        outputs["heatmap"] = str(heatmap_path)

        cases_path = output_dir / f"{title}_cases.png"
        plot_cases(parsed_dir, cases_path, coordinate_frame="absolute")
        outputs["cases"] = str(cases_path)
    else:
        notes.append(f"{title}: skipped heatmap / cases because snapshots or obstacle points are missing.")

    if evolve_rows := artifacts["evolve_rounds"]:
        evidence_chain_path = output_dir / f"{title}_evidence_chain.png"
        plot_evolve_evidence_chain(parsed_dir, evidence_chain_path)
        outputs["evidence_chain"] = str(evidence_chain_path)
    else:
        notes.append(f"{title}: skipped evidence_chain because evolve_rounds is empty.")

    return prefix_output_keys(outputs, title)


def export_comparison_figures(
    *,
    parsed_a: Path,
    parsed_b: Path,
    output_dir: Path,
    bundle_name: str,
    label_a: str,
    label_b: str,
) -> Dict[str, str]:
    outputs: Dict[str, str] = {}

    compare_heatmap_path = output_dir / f"{bundle_name}_compare_heatmap.png"
    plot_compare_heatmap(
        parsed_dir_a=parsed_a,
        parsed_dir_b=parsed_b,
        output_path=compare_heatmap_path,
        label_a=label_a,
        label_b=label_b,
        coordinate_frame="absolute",
    )
    outputs["compare_heatmap"] = str(compare_heatmap_path)

    compare_cases_path = output_dir / f"{bundle_name}_compare_cases.png"
    plot_compare_cases(
        parsed_dir_a=parsed_a,
        parsed_dir_b=parsed_b,
        output_path=compare_cases_path,
        label_a=label_a,
        label_b=label_b,
        coordinate_frame="absolute",
    )
    outputs["compare_cases"] = str(compare_cases_path)
    return outputs


def export_policy_compare_if_available(
    *,
    run_a_path: Path,
    run_b_path: Path,
    output_dir: Path,
    bundle_name: str,
    label_a: str,
    label_b: str,
    benchmark_a: str | None,
    benchmark_b: str | None,
    include_seen: bool,
    notes: List[str],
) -> Dict[str, str]:
    benchmark_a_path = Path(benchmark_a).expanduser().resolve() if benchmark_a else None
    benchmark_b_path = Path(benchmark_b).expanduser().resolve() if benchmark_b else None
    train_summary_a = find_latest_wandb_summary(run_a_path)
    train_summary_b = find_latest_wandb_summary(run_b_path)

    if train_summary_a is None or train_summary_b is None:
        notes.append("Skipped policy_eval_compare because one or both wandb-summary.json files could not be found.")
        return {}
    if benchmark_a_path is None or benchmark_b_path is None:
        notes.append("Skipped policy_eval_compare because benchmark summary paths were not provided.")
        return {}
    if not benchmark_a_path.exists() or not benchmark_b_path.exists():
        notes.append("Skipped policy_eval_compare because one or both benchmark summary paths do not exist.")
        return {}

    methods = [
        load_method(
            label=label_a,
            train_summary_path=train_summary_a,
            benchmark_summary_path=benchmark_a_path,
            include_seen=include_seen,
        ),
        load_method(
            label=label_b,
            train_summary_path=train_summary_b,
            benchmark_summary_path=benchmark_b_path,
            include_seen=include_seen,
        ),
    ]
    output_path = output_dir / f"{bundle_name}_policy_eval_compare.png"
    plot_policy_eval_compare(
        methods=methods,
        output_path=output_path,
        title=f"Policy Comparison: {bundle_name}",
        subtitle=f"{label_a} vs {label_b}",
    )
    return {"policy_eval_compare": str(output_path)}


def prefix_output_keys(outputs: Dict[str, str], prefix: str) -> Dict[str, str]:
    return {f"{prefix}:{key}": value for key, value in outputs.items()}


def resolve_bundle_dir(run_a_path: Path, run_b_path: Path | None, *, output_root: str | None) -> Path:
    if output_root is not None:
        return Path(output_root).expanduser().resolve()

    if run_b_path is None:
        return run_a_path.parent / "viz_exports"

    common_parent = Path(run_a_path).parent
    if run_b_path.parent == common_parent:
        return common_parent / "viz_exports"
    return common_path(run_a_path, run_b_path) / "viz_exports"


def build_bundle_name(run_a_path: Path, run_b_path: Path | None) -> str:
    title_a = run_title(run_a_path)
    if run_b_path is None:
        return title_a
    return f"{title_a}__vs__{run_title(run_b_path)}"


def run_title(run_path: Path | None) -> str:
    if run_path is None:
        return "unknown_run"
    parent_name = run_path.parent.name
    run_name = run_path.name
    if parent_name and parent_name.startswith("20"):
        return sanitize_name(f"{parent_name}_{run_name}")
    return sanitize_name(run_name)


def sanitize_name(text: str) -> str:
    cleaned = text.replace(" ", "_").replace("/", "_")
    return "".join(char if char.isalnum() or char in "._-+" else "_" for char in cleaned)


def default_label_for_run(run_path: Path) -> str:
    lowered = run_path.name.lower()
    if "evolve" in lowered or "llm" in lowered:
        return "failure-driven"
    if "static" in lowered or "baseline" in lowered:
        return "static baseline"
    return run_path.name


def find_latest_wandb_summary(run_path: Path) -> Path | None:
    candidates = list(run_path.glob("wandb/run-*/files/wandb-summary.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def common_path(path_a: Path, path_b: Path) -> Path:
    shared_parts = []
    for part_a, part_b in zip(path_a.parts, path_b.parts):
        if part_a != part_b:
            break
        shared_parts.append(part_a)
    if not shared_parts:
        return Path("/")
    return Path(*shared_parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a named bundle of failure-driven visualization figures.")
    parser.add_argument("--run-a", required=True, help="Primary experiment output directory.")
    parser.add_argument("--run-b", default=None, help="Optional comparison experiment output directory.")
    parser.add_argument("--label-a", default=None, help="Display label for run A in comparison figures.")
    parser.add_argument("--label-b", default=None, help="Display label for run B in comparison figures.")
    parser.add_argument("--benchmark-a", default=None, help="Optional benchmark_summary.json for run A.")
    parser.add_argument("--benchmark-b", default=None, help="Optional benchmark_summary.json for run B.")
    parser.add_argument("--output-root", default=None, help="Optional root directory that will contain the named figure bundle.")
    parser.add_argument("--force-reparse", action="store_true", help="Regenerate parsed tables before plotting.")
    parser.add_argument(
        "--no-include-seen",
        action="store_true",
        help="Do not include seen validation as the first point in policy comparison plots.",
    )
    args = parser.parse_args()

    manifest = export_figure_suite(
        run_a=args.run_a,
        run_b=args.run_b,
        label_a=args.label_a,
        label_b=args.label_b,
        benchmark_a=args.benchmark_a,
        benchmark_b=args.benchmark_b,
        output_root=args.output_root,
        force_reparse=args.force_reparse,
        include_seen_in_policy_compare=not args.no_include_seen,
    )

    print(f"Saved figure bundle to: {manifest['bundle_dir']}")
    print(f"Bundle name: {manifest['bundle_name']}")
    for key, value in sorted(dict(manifest["outputs"]).items()):
        print(f"- {key}: {value}")
    if manifest["notes"]:
        print("Notes:")
        for note in manifest["notes"]:
            print(f"  * {note}")


if __name__ == "__main__":
    main()
