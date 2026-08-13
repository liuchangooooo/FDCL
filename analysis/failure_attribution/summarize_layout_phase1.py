from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize multiple layout-level attribution Phase-1 reports."
    )
    parser.add_argument("--report-root", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--lift-threshold", type=float, default=2.0)
    parser.add_argument("--failure-rate-threshold", type=float, default=0.5)
    parser.add_argument(
        "--exclude-label",
        action="append",
        default=[],
        help="Report directory name to exclude. Can be passed multiple times.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    excluded = set(args.exclude_label or [])
    reports = [
        path
        for path in sorted(args.report_root.glob("*/layout_attribution_map.json"))
        if path.parent.name not in excluded
    ]
    rows = [
        summarize_report(
            report,
            lift_threshold=args.lift_threshold,
            failure_rate_threshold=args.failure_rate_threshold,
        )
        for report in reports
    ]
    rows = [row for row in rows if row]
    write_csv(rows, args.output_csv)
    write_markdown(
        rows,
        args.output_md,
        lift_threshold=args.lift_threshold,
        failure_rate_threshold=args.failure_rate_threshold,
    )
    print(f"Summarized {len(rows)} reports")
    print(f"Saved markdown: {args.output_md}")
    print(f"Saved csv: {args.output_csv}")


def summarize_report(
    report_path: Path,
    lift_threshold: float,
    failure_rate_threshold: float,
) -> Dict[str, Any]:
    data = json.loads(report_path.read_text(encoding="utf-8"))
    label = report_path.parent.name
    counts = data.get("global_counts") or {}
    comparison = data.get("comparison") or {}
    single = comparison.get("single_obstacle_best") or {}
    family_best = comparison.get("layout_family_best") or {}
    families = data.get("families") or {}
    best_layout = select_best_layout(family_best)
    pressure_coverage = informative_coverage(
        families.get("layout_pressure") or [],
        total_episodes=int(counts.get("num_episodes", 0)),
        lift_threshold=lift_threshold,
        failure_rate_threshold=failure_rate_threshold,
    )
    basic_coverage = informative_coverage(
        families.get("layout_basic") or [],
        total_episodes=int(counts.get("num_episodes", 0)),
        lift_threshold=lift_threshold,
        failure_rate_threshold=failure_rate_threshold,
    )
    return {
        "label": label,
        "rollouts": data.get("metadata", {}).get("rollouts", ""),
        "episodes": int(counts.get("num_episodes", 0)),
        "success_count": int(counts.get("success_count", 0)),
        "failure_count": int(counts.get("failure_count", 0)),
        "failure_rate": float(counts.get("failure_rate", 0.0)),
        "single_cell_id": single.get("cell_id", ""),
        "single_support": int(single.get("support", 0) or 0),
        "single_failure_rate": float(single.get("failure_rate", 0.0) or 0.0),
        "single_lift": float(single.get("failure_lift", 0.0) or 0.0),
        "best_layout_family": best_layout.get("family", ""),
        "best_layout_pattern": best_layout.get("pattern_id", ""),
        "best_layout_support": int(best_layout.get("support", 0) or 0),
        "best_layout_failure_rate": float(best_layout.get("failure_rate", 0.0) or 0.0),
        "best_layout_lift": float(best_layout.get("failure_lift", 0.0) or 0.0),
        "best_layout_dominant_failure": best_layout.get("dominant_failure_key", ""),
        "pressure_informative_coverage": pressure_coverage["episode_fraction"],
        "pressure_informative_failure_fraction": pressure_coverage["failure_fraction"],
        "pressure_informative_pattern_count": pressure_coverage["pattern_count"],
        "basic_informative_coverage": basic_coverage["episode_fraction"],
        "basic_informative_failure_fraction": basic_coverage["failure_fraction"],
        "basic_informative_pattern_count": basic_coverage["pattern_count"],
    }


def select_best_layout(family_best: Mapping[str, Any]) -> Dict[str, Any]:
    candidates: List[Dict[str, Any]] = []
    for family, payload in family_best.items():
        best = dict((payload or {}).get("best_failure_lift") or {})
        if best:
            best["family"] = family
            candidates.append(best)
    if not candidates:
        return {}
    candidates.sort(
        key=lambda row: (
            float(row.get("failure_rate", 0.0)),
            float(row.get("failure_lift", 0.0)),
            int(row.get("support", 0)),
        ),
        reverse=True,
    )
    return candidates[0]


def informative_coverage(
    rows: Sequence[Mapping[str, Any]],
    total_episodes: int,
    lift_threshold: float,
    failure_rate_threshold: float,
) -> Dict[str, Any]:
    selected = [
        row
        for row in rows
        if row.get("supported")
        and float(row.get("failure_lift", 0.0)) >= lift_threshold
        and float(row.get("failure_rate", 0.0)) >= failure_rate_threshold
    ]
    episode_count = sum(int(row.get("total_count", 0)) for row in selected)
    failure_count = sum(int(row.get("failure_count", 0)) for row in selected)
    return {
        "pattern_count": len(selected),
        "episode_fraction": episode_count / total_episodes if total_episodes else 0.0,
        "failure_fraction": failure_count / episode_count if episode_count else 0.0,
    }


def write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "label",
        "episodes",
        "failure_rate",
        "single_support",
        "single_failure_rate",
        "single_lift",
        "best_layout_family",
        "best_layout_support",
        "best_layout_failure_rate",
        "best_layout_lift",
        "best_layout_dominant_failure",
        "pressure_informative_coverage",
        "pressure_informative_failure_fraction",
        "pressure_informative_pattern_count",
        "basic_informative_coverage",
        "basic_informative_failure_fraction",
        "basic_informative_pattern_count",
        "single_cell_id",
        "best_layout_pattern",
        "rollouts",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_markdown(
    rows: Sequence[Mapping[str, Any]],
    path: Path,
    lift_threshold: float,
    failure_rate_threshold: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase 1 Layout Attribution Cross-Run Summary",
        "",
        "This report compares single-obstacle attribution against layout-level attribution. "
        "Raw lift should be read together with global failure rate, because an all-failure "
        "pattern has lift = 1 / global_failure_rate.",
        "",
        f"Informative coverage threshold: lift >= {lift_threshold:.2f}, "
        f"pattern failure_rate >= {failure_rate_threshold:.2f}.",
        "",
        "## Numeric Summary",
        "",
        "| run | episodes | global fail | single best | layout best | pressure coverage | basic coverage |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {label} | {episodes} | {failure_rate:.3f} | "
            "{single_failure_rate:.3f}/{single_lift:.2f} ({single_support}) | "
            "{best_layout_failure_rate:.3f}/{best_layout_lift:.2f} ({best_layout_support}) | "
            "{pressure_informative_coverage:.3f} | {basic_informative_coverage:.3f} |".format(
                **row
            )
        )

    lines.extend(
        [
            "",
            "The `single best` and `layout best` columns are formatted as "
            "`pattern_failure_rate / lift (support)`.",
            "",
            "## Top Layout Patterns",
            "",
        ]
    )
    for row in rows:
        lines.extend(
            [
                f"### {row['label']}",
                "",
                f"- rollouts: `{row['rollouts']}`",
                f"- single-obstacle best: `{row['single_cell_id']}`",
                f"- layout best family: `{row['best_layout_family']}`",
                f"- layout best dominant failure: `{row['best_layout_dominant_failure']}`",
                f"- layout best pattern: `{row['best_layout_pattern']}`",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
