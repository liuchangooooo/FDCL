from __future__ import annotations

import argparse
import contextlib
import csv
import io
import importlib.util
import json
import math
import re
import sys
import types
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from DIVO.curriculum.adapters.pusht_adapter import pusht_encode_layout
from DIVO.curriculum.attribution import AttributionConfig, cell_key


def analytic_obs_collision_check(Tblock_angle, obs_center, obs_size, threshold=0.01):
    horizontal_length = 0.10
    horizontal_thickness = 0.03
    horizontal_center = (0, 0)

    vertical_length = 0.07
    vertical_thickness = 0.03
    vertical_center = (0, -0.05)

    def rotate_point_around_origin(point, center, angle):
        px, py = point
        ox, oy = center
        qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
        qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
        return qx, qy

    def edge_vectors(points):
        return [points[(i + 1) % len(points)] - points[i] for i in range(len(points))]

    def normalize(v):
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm

    def project_polygon(axis, polygon):
        dots = [np.dot(vertex, axis) for vertex in polygon]
        return min(dots), max(dots)

    def overlap(min_a, max_a, min_b, max_b):
        return max_a >= min_b and max_b >= min_a

    def separating_axis_theorem(polygon1, polygon2):
        edges = edge_vectors(polygon1) + edge_vectors(polygon2)
        axes = [normalize(np.array([-edge[1], edge[0]])) for edge in edges]
        for axis in axes:
            min_a, max_a = project_polygon(axis, polygon1)
            min_b, max_b = project_polygon(axis, polygon2)
            if not overlap(min_a, max_a, min_b, max_b):
                return False
        return True

    half_side = (obs_size + threshold) / 2
    vertices_arbitrary_square = np.array(
        [
            (obs_center[0] - half_side, obs_center[1] - half_side),
            (obs_center[0] + half_side, obs_center[1] - half_side),
            (obs_center[0] + half_side, obs_center[1] + half_side),
            (obs_center[0] - half_side, obs_center[1] + half_side),
        ]
    )

    for rect_center, width, height in [
        (horizontal_center, horizontal_length, horizontal_thickness),
        (vertical_center, vertical_thickness, vertical_length),
    ]:
        vertices = [
            (rect_center[0] - width / 2, rect_center[1] - height / 2),
            (rect_center[0] + width / 2, rect_center[1] - height / 2),
            (rect_center[0] + width / 2, rect_center[1] + height / 2),
            (rect_center[0] - width / 2, rect_center[1] + height / 2),
        ]
        rotated_vertices = np.array(
            [rotate_point_around_origin(v, (0, 0), Tblock_angle) for v in vertices]
        )
        if separating_axis_theorem(rotated_vertices, vertices_arbitrary_square):
            return True

    return False


def _load_strategy_executor_class():
    util_stub = types.ModuleType("DIVO.utils.util")
    util_stub.analytic_obs_collision_check = analytic_obs_collision_check
    sys.modules.setdefault("DIVO.utils.util", util_stub)

    module_path = REPO_ROOT / "DIVO" / "env" / "pusht" / "llm_topology_generator.py"
    spec = importlib.util.spec_from_file_location("_audit_llm_topology_generator", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load StrategyExecutor from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.StrategyExecutor


StrategyExecutor = _load_strategy_executor_class()


ROUND_RE = re.compile(r"evolve_(\d+)_ep_(\d+)")


def main() -> None:
    args = parse_args()
    if args.run_root:
        audit_run(args)
    else:
        if not args.generator or not args.attribution_map:
            raise SystemExit("Either --run-root or both --generator and --attribution-map are required.")
        attr = load_json(Path(args.attribution_map))
        result = audit_generator(
            generator_path=Path(args.generator),
            attribution_map=attr,
            num_pose_samples=args.num_samples,
            num_obstacles=args.num_obstacles,
            seed=args.seed,
            target_top_k=args.target_top_k,
            min_target_lift=args.min_target_lift,
            low_top_k=args.low_top_k,
            quiet=not args.verbose_generator,
        )
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(result, output_dir / "generator_coverage_audit.json")
        print(f"Wrote {output_dir / 'generator_coverage_audit.json'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Offline audit for whether evolved Push-T generators cover attribution-derived cells."
        )
    )
    parser.add_argument("--run-root", type=Path, help="Training output root with generators/ and failure_attribution/.")
    parser.add_argument("--generator", type=Path, help="Single generator .py file.")
    parser.add_argument("--attribution-map", type=Path, help="Single attribution_map.json.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for JSON/CSV audit outputs.")
    parser.add_argument("--num-samples", type=int, default=5000, help="Number of tblock poses to sample per generator.")
    parser.add_argument("--num-obstacles", type=int, default=2, help="Obstacle count passed to generate_obstacles.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-top-k", type=int, default=5)
    parser.add_argument("--min-target-lift", type=float, default=1.0)
    parser.add_argument("--low-top-k", type=int, default=5)
    parser.add_argument("--verbose-generator", action="store_true", help="Do not suppress generator prints/errors.")
    return parser.parse_args()


def audit_run(args: argparse.Namespace) -> None:
    run_root = Path(args.run_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generator_dir = run_root / "generators"
    attribution_dir = run_root / "failure_attribution"
    generators = {
        parse_generator_index(path): path
        for path in sorted(generator_dir.glob("generator_*.py"))
    }
    maps = sorted(attribution_dir.glob("evolve_*_ep_*/attribution_map.json"))
    if not generators:
        raise SystemExit(f"No generator_*.py files found under {generator_dir}")
    if not maps:
        raise SystemExit(f"No attribution_map.json files found under {attribution_dir}")

    rounds: List[Dict[str, Any]] = []
    csv_rows: List[Dict[str, Any]] = []
    for map_path in maps:
        round_idx, episode_idx = parse_round_from_path(map_path)
        before_path = generators.get(round_idx - 1)
        after_path = generators.get(round_idx)
        if before_path is None:
            print(f"Skipping evolve {round_idx}: missing generator_{round_idx - 1:03d}.py")
            continue

        attr = load_json(map_path)
        before = audit_generator(
            generator_path=before_path,
            attribution_map=attr,
            num_pose_samples=args.num_samples,
            num_obstacles=args.num_obstacles,
            seed=args.seed,
            target_top_k=args.target_top_k,
            min_target_lift=args.min_target_lift,
            low_top_k=args.low_top_k,
            quiet=not args.verbose_generator,
        )
        after = None
        if after_path is not None:
            after = audit_generator(
                generator_path=after_path,
                attribution_map=attr,
                num_pose_samples=args.num_samples,
                num_obstacles=args.num_obstacles,
                seed=args.seed,
                target_top_k=args.target_top_k,
                min_target_lift=args.min_target_lift,
                low_top_k=args.low_top_k,
                quiet=not args.verbose_generator,
            )

        delta = build_delta(before, after)
        record = {
            "round_idx": round_idx,
            "episode_idx": episode_idx,
            "attribution_map": str(map_path),
            "before_generator": str(before_path),
            "after_generator": str(after_path) if after_path else None,
            "evidence": before["evidence"],
            "before": before["coverage"],
            "after": after["coverage"] if after else None,
            "delta_after_minus_before": delta,
        }
        rounds.append(record)
        csv_rows.append(flatten_round_for_csv(record))
        print(
            f"round {round_idx:03d}: "
            f"target {fmt_pct(before['coverage']['target_coverage'])}"
            + (
                f" -> {fmt_pct(after['coverage']['target_coverage'])} "
                f"(delta {fmt_signed_pct(delta['target_coverage_delta'])})"
                if after
                else " -> missing after generator"
            ),
            flush=True,
        )
        write_run_outputs(
            run_root=run_root,
            args=args,
            rounds=rounds,
            csv_rows=csv_rows,
            output_dir=output_dir,
        )

    write_run_outputs(
        run_root=run_root,
        args=args,
        rounds=rounds,
        csv_rows=csv_rows,
        output_dir=output_dir,
    )
    print(f"Wrote {output_dir / 'round_coverage_audit.json'}")
    print(f"Wrote {output_dir / 'round_coverage_audit.csv'}")
    print(f"Wrote {output_dir / 'round_coverage_summary.txt'}")


def write_run_outputs(
    run_root: Path,
    args: argparse.Namespace,
    rounds: Sequence[Mapping[str, Any]],
    csv_rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> None:
    report = {
        "run_root": str(run_root),
        "settings": {
            "num_samples": args.num_samples,
            "num_obstacles": args.num_obstacles,
            "seed": args.seed,
            "target_top_k": args.target_top_k,
            "min_target_lift": args.min_target_lift,
            "low_top_k": args.low_top_k,
        },
        "aggregate": aggregate_rounds(rounds),
        "rounds": list(rounds),
    }
    write_json(report, output_dir / "round_coverage_audit.json")
    write_csv(csv_rows, output_dir / "round_coverage_audit.csv")
    write_text_summary(report, output_dir / "round_coverage_summary.txt")


def audit_generator(
    generator_path: Path,
    attribution_map: Mapping[str, Any],
    num_pose_samples: int,
    num_obstacles: int,
    seed: int,
    target_top_k: int,
    min_target_lift: float,
    low_top_k: int,
    quiet: bool = True,
) -> Dict[str, Any]:
    cfg = AttributionConfig.from_dict(
        attribution_map.get("metadata", {}).get("config", {})
    )
    target_cells, used_fallback_targets = select_target_cells(
        attribution_map,
        min_support=cfg.min_support,
        min_lift=min_target_lift,
        top_k=target_top_k,
    )
    low_cells = select_low_lift_overcovered_cells(
        attribution_map,
        min_support=cfg.min_support,
        top_k=low_top_k,
    )
    target_ids = [str(row["cell_id"]) for row in target_cells]
    low_ids = [str(row["cell_id"]) for row in low_cells]

    executor = load_executor(generator_path, quiet=quiet)
    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    cell_counts: Counter[str] = Counter()
    alpha_counts: Counter[str] = Counter()
    beta_abs_counts: Counter[str] = Counter()
    blockage_counts: Counter[str] = Counter()
    invalid_reason_counts: Counter[str] = Counter()
    empty_generations = 0
    invalid_generations = 0
    wrong_count_generations = 0
    accepted_generations = 0
    obstacle_samples = 0

    for _ in range(num_pose_samples):
        pose = sample_tblock_pose(rng)
        obstacles = call_generator(executor, pose, num_obstacles, quiet=quiet)
        if not obstacles:
            empty_generations += 1
            continue
        if len(obstacles) != num_obstacles:
            wrong_count_generations += 1

        is_valid, reason = executor.validate_obstacles(obstacles, pose)
        if not is_valid:
            invalid_generations += 1
            invalid_reason_counts[str(reason)] += 1
            continue

        accepted_generations += 1
        for z in pusht_encode_layout(obstacles, pose):
            key = cell_key(z.to_dict(), cfg)
            alpha_bin, beta_abs_bin, blockage_bin = key
            cell_id = format_cell_id(key)
            cell_counts[cell_id] += 1
            alpha_counts[str(alpha_bin)] += 1
            beta_abs_counts[str(beta_abs_bin)] += 1
            blockage_counts[str(blockage_bin)] += 1
            obstacle_samples += 1

    coverage = build_coverage_summary(
        generator_path=generator_path,
        cell_counts=cell_counts,
        alpha_counts=alpha_counts,
        beta_abs_counts=beta_abs_counts,
        blockage_counts=blockage_counts,
        target_ids=target_ids,
        low_ids=low_ids,
        num_pose_samples=num_pose_samples,
        accepted_generations=accepted_generations,
        empty_generations=empty_generations,
        invalid_generations=invalid_generations,
        wrong_count_generations=wrong_count_generations,
        invalid_reason_counts=invalid_reason_counts,
        obstacle_samples=obstacle_samples,
    )
    evidence = build_evidence_summary(
        attribution_map=attribution_map,
        target_cells=target_cells,
        low_cells=low_cells,
        used_fallback_targets=used_fallback_targets,
    )
    return {
        "generator_path": str(generator_path),
        "evidence": evidence,
        "coverage": coverage,
    }


def build_coverage_summary(
    generator_path: Path,
    cell_counts: Counter[str],
    alpha_counts: Counter[str],
    beta_abs_counts: Counter[str],
    blockage_counts: Counter[str],
    target_ids: Sequence[str],
    low_ids: Sequence[str],
    num_pose_samples: int,
    accepted_generations: int,
    empty_generations: int,
    invalid_generations: int,
    wrong_count_generations: int,
    invalid_reason_counts: Counter[str],
    obstacle_samples: int,
) -> Dict[str, Any]:
    def share(count: int) -> float:
        return float(count / obstacle_samples) if obstacle_samples else 0.0

    far_side_low = sum(
        count
        for cell_id, count in cell_counts.items()
        if "beta_abs=far_side" in cell_id and "blockage=low" in cell_id
    )
    medium_or_high = sum(
        count
        for cell_id, count in cell_counts.items()
        if "blockage=medium" in cell_id or "blockage=high" in cell_id
    )
    high_blockage = sum(
        count for cell_id, count in cell_counts.items() if "blockage=high" in cell_id
    )
    target_count = sum(cell_counts.get(cell_id, 0) for cell_id in target_ids)
    low_count = sum(cell_counts.get(cell_id, 0) for cell_id in low_ids)
    top_sampled = [
        {"cell_id": cell_id, "count": count, "coverage": share(count)}
        for cell_id, count in cell_counts.most_common(10)
    ]

    return {
        "generator_file": str(generator_path),
        "generator_bytes": generator_path.stat().st_size,
        "num_pose_samples": int(num_pose_samples),
        "accepted_generations": int(accepted_generations),
        "acceptance_rate": float(accepted_generations / num_pose_samples) if num_pose_samples else 0.0,
        "empty_generations": int(empty_generations),
        "invalid_generations": int(invalid_generations),
        "wrong_count_generations": int(wrong_count_generations),
        "invalid_reason_counts": dict(invalid_reason_counts.most_common(10)),
        "num_obstacle_samples": int(obstacle_samples),
        "occupied_cells": len(cell_counts),
        "coverage_entropy": entropy(cell_counts.values()),
        "coverage_entropy_normalized": normalized_entropy(cell_counts.values()),
        "target_coverage": share(target_count),
        "low_lift_overcovered_coverage": share(low_count),
        "far_side_low_coverage": share(far_side_low),
        "medium_or_high_blockage_coverage": share(medium_or_high),
        "high_blockage_coverage": share(high_blockage),
        "by_alpha_bin": proportions(alpha_counts),
        "by_beta_abs_bin": proportions(beta_abs_counts),
        "by_blockage_bin": proportions(blockage_counts),
        "top_sampled_cells": top_sampled,
    }


def build_evidence_summary(
    attribution_map: Mapping[str, Any],
    target_cells: Sequence[Mapping[str, Any]],
    low_cells: Sequence[Mapping[str, Any]],
    used_fallback_targets: bool,
) -> Dict[str, Any]:
    metadata = attribution_map.get("metadata", {})
    target_ids = [str(row["cell_id"]) for row in target_cells]
    low_ids = [str(row["cell_id"]) for row in low_cells]
    current_target_coverage = sum(float(row.get("p_cell", 0.0)) for row in target_cells)
    current_low_coverage = sum(float(row.get("p_cell", 0.0)) for row in low_cells)
    return {
        "evolve_index": metadata.get("evolve_index"),
        "episode_idx": metadata.get("total_episode_count"),
        "batch_episode_count": metadata.get("batch_episode_count"),
        "success_rate": metadata.get("success_rate"),
        "trigger_reason": metadata.get("trigger_reason"),
        "used_fallback_targets": used_fallback_targets,
        "target_cell_ids": target_ids,
        "target_cells": [compact_cell(row) for row in target_cells],
        "current_target_coverage": current_target_coverage,
        "low_lift_overcovered_cell_ids": low_ids,
        "low_lift_overcovered_cells": [compact_cell(row) for row in low_cells],
        "current_low_lift_overcovered_coverage": current_low_coverage,
    }


def select_target_cells(
    attribution_map: Mapping[str, Any],
    min_support: int,
    min_lift: float,
    top_k: int,
) -> Tuple[List[Mapping[str, Any]], bool]:
    cells = list(attribution_map.get("cells", []))
    candidates = [
        row
        for row in cells
        if int(row.get("total_count", 0)) >= min_support
        and int(row.get("failure_count", 0)) > 0
        and float(row.get("failure_lift", 0.0)) > min_lift
    ]
    candidates.sort(
        key=lambda row: (
            float(row.get("failure_lift", 0.0)),
            int(row.get("failure_count", 0)),
            int(row.get("total_count", 0)),
        ),
        reverse=True,
    )
    if candidates:
        return candidates[:top_k], False

    fallback = [
        row
        for row in attribution_map.get("top_cells", cells)
        if int(row.get("failure_count", 0)) > 0
    ]
    fallback.sort(
        key=lambda row: (
            float(row.get("failure_lift", 0.0)),
            int(row.get("failure_count", 0)),
            int(row.get("total_count", 0)),
        ),
        reverse=True,
    )
    return fallback[:top_k], True


def select_low_lift_overcovered_cells(
    attribution_map: Mapping[str, Any],
    min_support: int,
    top_k: int,
) -> List[Mapping[str, Any]]:
    cells = list(attribution_map.get("cells", []))
    candidates = [
        row
        for row in cells
        if int(row.get("total_count", 0)) >= min_support
        and float(row.get("failure_lift", 0.0)) < 1.0
    ]
    candidates.sort(key=lambda row: float(row.get("p_cell", 0.0)), reverse=True)
    return candidates[:top_k]


def compact_cell(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "cell_id": row.get("cell_id"),
        "total_count": row.get("total_count"),
        "failure_count": row.get("failure_count"),
        "failure_rate": row.get("failure_rate"),
        "failure_lift": row.get("failure_lift"),
        "p_cell": row.get("p_cell"),
        "dominant_failure_key": row.get("dominant_failure_key"),
    }


def build_delta(before: Mapping[str, Any], after: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if after is None:
        return {}
    before_cov = before["coverage"]
    after_cov = after["coverage"]
    fields = [
        "target_coverage",
        "low_lift_overcovered_coverage",
        "far_side_low_coverage",
        "medium_or_high_blockage_coverage",
        "high_blockage_coverage",
        "coverage_entropy_normalized",
    ]
    return {
        f"{field}_delta": float(after_cov.get(field, 0.0) - before_cov.get(field, 0.0))
        for field in fields
    }


def aggregate_rounds(rounds: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    deltas = [row.get("delta_after_minus_before", {}) for row in rounds]
    target_deltas = [
        float(delta["target_coverage_delta"])
        for delta in deltas
        if "target_coverage_delta" in delta
    ]
    low_deltas = [
        float(delta["low_lift_overcovered_coverage_delta"])
        for delta in deltas
        if "low_lift_overcovered_coverage_delta" in delta
    ]
    size_deltas = []
    for row in rounds:
        before = row.get("before") or {}
        after = row.get("after") or {}
        if before and after:
            size_deltas.append(float(after.get("generator_bytes", 0) - before.get("generator_bytes", 0)))

    return {
        "num_rounds": len(rounds),
        "rounds_with_target_coverage_increase": sum(delta > 0.0 for delta in target_deltas),
        "rounds_with_target_coverage_decrease": sum(delta < 0.0 for delta in target_deltas),
        "mean_target_coverage_delta": mean(target_deltas),
        "mean_low_lift_overcovered_coverage_delta": mean(low_deltas),
        "mean_generator_byte_delta": mean(size_deltas),
        "max_generator_bytes_after": max(
            [float((row.get("after") or {}).get("generator_bytes", 0.0)) for row in rounds] or [0.0]
        ),
    }


def flatten_round_for_csv(record: Mapping[str, Any]) -> Dict[str, Any]:
    evidence = record["evidence"]
    before = record["before"]
    after = record.get("after") or {}
    delta = record.get("delta_after_minus_before") or {}
    return {
        "round_idx": record["round_idx"],
        "episode_idx": record["episode_idx"],
        "success_rate": evidence.get("success_rate"),
        "trigger_reason": evidence.get("trigger_reason"),
        "target_cell_count": len(evidence.get("target_cell_ids", [])),
        "used_fallback_targets": evidence.get("used_fallback_targets"),
        "current_target_coverage": evidence.get("current_target_coverage"),
        "before_target_coverage": before.get("target_coverage"),
        "after_target_coverage": after.get("target_coverage"),
        "target_coverage_delta": delta.get("target_coverage_delta"),
        "current_low_lift_overcovered_coverage": evidence.get("current_low_lift_overcovered_coverage"),
        "before_low_lift_overcovered_coverage": before.get("low_lift_overcovered_coverage"),
        "after_low_lift_overcovered_coverage": after.get("low_lift_overcovered_coverage"),
        "low_lift_overcovered_coverage_delta": delta.get("low_lift_overcovered_coverage_delta"),
        "before_far_side_low_coverage": before.get("far_side_low_coverage"),
        "after_far_side_low_coverage": after.get("far_side_low_coverage"),
        "far_side_low_coverage_delta": delta.get("far_side_low_coverage_delta"),
        "before_medium_or_high_blockage_coverage": before.get("medium_or_high_blockage_coverage"),
        "after_medium_or_high_blockage_coverage": after.get("medium_or_high_blockage_coverage"),
        "medium_or_high_blockage_coverage_delta": delta.get("medium_or_high_blockage_coverage_delta"),
        "before_high_blockage_coverage": before.get("high_blockage_coverage"),
        "after_high_blockage_coverage": after.get("high_blockage_coverage"),
        "high_blockage_coverage_delta": delta.get("high_blockage_coverage_delta"),
        "before_generator_bytes": before.get("generator_bytes"),
        "after_generator_bytes": after.get("generator_bytes"),
        "generator_byte_delta": (
            after.get("generator_bytes", 0) - before.get("generator_bytes", 0)
            if after
            else None
        ),
        "target_cell_ids": " ; ".join(evidence.get("target_cell_ids", [])),
    }


def write_text_summary(report: Mapping[str, Any], path: Path) -> None:
    lines = [
        "Generator Coverage Audit",
        "========================",
        f"run_root: {report['run_root']}",
        f"num_rounds: {report['aggregate']['num_rounds']}",
        f"mean_target_coverage_delta: {fmt_signed_pct(report['aggregate']['mean_target_coverage_delta'])}",
        f"rounds_target_increase: {report['aggregate']['rounds_with_target_coverage_increase']}",
        f"rounds_target_decrease: {report['aggregate']['rounds_with_target_coverage_decrease']}",
        f"mean_generator_byte_delta: {report['aggregate']['mean_generator_byte_delta']:.1f}",
        "",
        "Per-round summary:",
    ]
    for row in report["rounds"]:
        before = row["before"]
        after = row.get("after") or {}
        delta = row.get("delta_after_minus_before") or {}
        evidence = row["evidence"]
        lines.append(
            f"- round {row['round_idx']:03d} ep={row['episode_idx']}: "
            f"sr={evidence.get('success_rate')}; "
            f"target {fmt_pct(before.get('target_coverage', 0.0))}"
            f" -> {fmt_pct(after.get('target_coverage', 0.0))}; "
            f"delta={fmt_signed_pct(delta.get('target_coverage_delta', 0.0))}; "
            f"far_side_low {fmt_pct(before.get('far_side_low_coverage', 0.0))}"
            f" -> {fmt_pct(after.get('far_side_low_coverage', 0.0))}; "
            f"bytes {before.get('generator_bytes')} -> {after.get('generator_bytes')}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_executor(generator_path: Path, quiet: bool) -> StrategyExecutor:
    executor = StrategyExecutor()
    code = generator_path.read_text(encoding="utf-8")
    if quiet:
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            ok = executor.load_topology_generator(code)
    else:
        ok = executor.load_topology_generator(code)
    if not ok:
        raise RuntimeError(f"Failed to load generator: {generator_path}")
    return executor


def call_generator(
    executor: StrategyExecutor,
    pose: np.ndarray,
    num_obstacles: int,
    quiet: bool,
) -> List[Dict[str, Any]]:
    if quiet:
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            return executor.generate(pose, num_obstacles)
    return executor.generate(pose, num_obstacles)


def sample_tblock_pose(rng: np.random.Generator) -> np.ndarray:
    x = float(rng.uniform(-0.18, 0.18))
    y = float(rng.uniform(-0.18, 0.18))
    while abs(x) < 0.1 and abs(y) < 0.1:
        x = float(rng.uniform(-0.18, 0.18))
        y = float(rng.uniform(-0.18, 0.18))
    theta = float(rng.uniform(0.0, 2.0 * np.pi))
    return np.array([x, y, theta], dtype=np.float64)


def parse_generator_index(path: Path) -> int:
    match = re.search(r"generator_(\d+)\.py$", path.name)
    if not match:
        raise ValueError(f"Cannot parse generator index from {path}")
    return int(match.group(1))


def parse_round_from_path(path: Path) -> Tuple[int, int]:
    for part in path.parts:
        match = ROUND_RE.match(part)
        if match:
            return int(match.group(1)), int(match.group(2))
    raise ValueError(f"Cannot parse evolve round from {path}")


def format_cell_id(key: Sequence[str]) -> str:
    alpha_bin, beta_abs_bin, blockage_bin = key
    return f"alpha={alpha_bin}|beta_abs={beta_abs_bin}|blockage={blockage_bin}"


def entropy(counts: Iterable[int]) -> float:
    values = [int(count) for count in counts if int(count) > 0]
    total = sum(values)
    if total <= 0:
        return 0.0
    return float(-sum((count / total) * math.log(count / total) for count in values))


def normalized_entropy(counts: Iterable[int]) -> float:
    values = [int(count) for count in counts if int(count) > 0]
    if len(values) <= 1:
        return 0.0
    return float(entropy(values) / math.log(len(values)))


def proportions(counter: Counter[str]) -> Dict[str, float]:
    total = sum(counter.values())
    if total <= 0:
        return {}
    return {key: float(value / total) for key, value in sorted(counter.items())}


def mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def fmt_pct(value: Any) -> str:
    return f"{float(value) * 100:.2f}%"


def fmt_signed_pct(value: Any) -> str:
    value = float(value)
    return f"{value * 100:+.2f}%"


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(data: Mapping[str, Any], path: Path) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True), encoding="utf-8")


def write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
