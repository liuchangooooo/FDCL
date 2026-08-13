from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from DIVO.curriculum.attribution import read_jsonl  # noqa: E402
from DIVO.curriculum.layout_features import extract_layout_z  # noqa: E402
from DIVO.curriculum.pattern_discovery import (  # noqa: E402
    PatternDiscoveryConfig,
    discover_failure_graph_patterns,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Discover failure-associated scene-graph layout patterns from rollout JSONL. "
            "This is an offline diagnostic and does not modify training."
        )
    )
    parser.add_argument("--rollouts", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--num-bins", type=int, default=4)
    parser.add_argument("--max-conditions", type=int, default=3)
    parser.add_argument("--min-support", type=int, default=100)
    parser.add_argument("--min-failure-count", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--beam-width", type=int, default=250)
    parser.add_argument("--max-overlap", type=float, default=0.90)
    parser.add_argument("--min-lift", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.rollouts)
    pattern_records, skipped = build_pattern_records(records)
    config = PatternDiscoveryConfig(
        num_bins=args.num_bins,
        max_conditions=args.max_conditions,
        min_support=args.min_support,
        min_failure_count=args.min_failure_count,
        top_k=args.top_k,
        beam_width=args.beam_width,
        max_overlap=args.max_overlap,
        min_lift=args.min_lift,
    )
    result = discover_failure_graph_patterns(
        pattern_records,
        config=config,
        metadata={
            "rollouts": str(args.rollouts),
            "raw_record_count": len(records),
            "scene_graph_record_count": len(pattern_records),
            "skipped_without_scene_graph": skipped,
        },
    )
    write_outputs(result, args.output_dir)
    print(f"Loaded {len(records)} rollout records from {args.rollouts}")
    print(f"Used {len(pattern_records)} records with scene_graph; skipped {skipped}")
    print(f"Saved graph pattern attribution outputs to {args.output_dir}")


def build_pattern_records(records: Sequence[Mapping[str, Any]]) -> tuple[List[Dict[str, Any]], int]:
    output: List[Dict[str, Any]] = []
    skipped = 0
    for record in records:
        scene_graph = record.get("scene_graph")
        if not isinstance(scene_graph, Mapping):
            skipped += 1
            continue
        try:
            layout_z = extract_layout_z(scene_graph)
        except Exception:
            skipped += 1
            continue
        output.append(
            {
                "episode_id": record.get("episode_id"),
                "layout_z": layout_z,
                "termination": record.get("termination", "unknown"),
                "failure_key": record.get("failure_key", record.get("termination", "unknown")),
                "reward": record.get("reward"),
                "steps": record.get("steps"),
            }
        )
    return output, skipped


def write_outputs(result: Mapping[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "failure_graph_patterns.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    write_csv(result, output_dir / "failure_graph_patterns.csv")
    (output_dir / "failure_graph_pattern_summary.txt").write_text(
        build_summary_text(result),
        encoding="utf-8",
    )


def write_csv(result: Mapping[str, Any], path: Path) -> None:
    fields = [
        "rank",
        "support",
        "failure_count",
        "failure_rate",
        "failure_lift",
        "failure_rate_lcb",
        "failure_lift_lcb",
        "dominant_failure_key",
        "dominant_failure_count",
        "dominant_failure_lift",
        "score",
        "conditions",
        "example_episode_ids",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rank, pattern in enumerate(result.get("top_patterns", []), start=1):
            row = dict(pattern)
            row["rank"] = rank
            row["conditions"] = " AND ".join(pattern.get("conditions", []))
            row["example_episode_ids"] = json.dumps(pattern.get("example_episode_ids", []))
            writer.writerow({field: row.get(field, "") for field in fields})


def build_summary_text(result: Mapping[str, Any]) -> str:
    metadata = result.get("metadata", {})
    stats = result.get("global_stats", {})
    lines = [
        "Failure-Conditioned Graph Pattern Attribution",
        "================================================",
        f"rollouts: {metadata.get('rollouts')}",
        f"raw_record_count: {metadata.get('raw_record_count')}",
        f"scene_graph_record_count: {metadata.get('scene_graph_record_count')}",
        f"skipped_without_scene_graph: {metadata.get('skipped_without_scene_graph')}",
        f"num_bins: {metadata.get('num_bins')}",
        f"max_conditions: {metadata.get('max_conditions')}",
        f"min_support: {metadata.get('min_support')}",
        f"min_failure_count: {metadata.get('min_failure_count')}",
        "",
        "Global stats:",
        f"- num_episodes: {stats.get('num_episodes')}",
        f"- failure_rate: {float(stats.get('failure_rate', 0.0)):.3f}",
        f"- failure_count: {stats.get('failure_count')}",
        f"- success_count: {stats.get('success_count')}",
        "",
        f"atomic_predicate_count: {result.get('atomic_predicate_count')}",
        f"candidate_count: {result.get('candidate_count')}",
        "",
        "Top patterns:",
    ]
    for idx, pattern in enumerate(result.get("top_patterns", []), start=1):
        lines.extend(
            [
                "",
                f"{idx}. {' AND '.join(pattern.get('conditions', []))}",
                f"   support={pattern.get('support')}, "
                f"failure_rate={float(pattern.get('failure_rate', 0.0)):.3f}, "
                f"lift={float(pattern.get('failure_lift', 0.0)):.3f}, "
                f"lcb_lift={float(pattern.get('failure_lift_lcb', 0.0)):.3f}",
                f"   dominant_failure={pattern.get('dominant_failure_key')} "
                f"({pattern.get('dominant_failure_count')}), "
                f"dominant_lift={float(pattern.get('dominant_failure_lift', 0.0)):.3f}",
            ]
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
