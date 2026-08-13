from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from DIVO.curriculum.attribution import (  # noqa: E402
    AttributionConfig,
    compute_attribution,
    read_jsonl,
)
from DIVO.curriculum.layout_features import extract_layout_z  # noqa: E402


EPS = 1e-9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build episode-level layout attribution from Push-T obstacle rollout JSONL. "
            "This is an offline Phase-1 diagnostic: it does not change training."
        )
    )
    parser.add_argument("--rollouts", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--min-support", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument(
        "--obstacle-min-support",
        type=int,
        default=50,
        help="Support threshold used for the single-obstacle attribution comparison.",
    )
    parser.add_argument(
        "--min-key-count",
        type=int,
        default=20,
        help=(
            "Minimum in-pattern count required before reporting a failure-key lift. "
            "This avoids ranking tiny count=1/2 behavior labels as strong signals."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.rollouts)
    result = compute_layout_attribution(
        records,
        min_support=args.min_support,
        top_k=args.top_k,
        obstacle_min_support=args.obstacle_min_support,
        min_key_count=args.min_key_count,
        metadata={"rollouts": str(args.rollouts)},
    )
    write_outputs(result, args.output_dir)
    print(f"Loaded {len(records)} episodes from {args.rollouts}")
    print(f"Saved layout attribution outputs to {args.output_dir}")


def compute_layout_attribution(
    records: Iterable[Mapping[str, Any]],
    min_support: int = 50,
    top_k: int = 12,
    obstacle_min_support: int = 50,
    min_key_count: int = 20,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    records_list = [record for record in records if isinstance(record, Mapping)]
    global_counts = build_global_counts(records_list)
    rows_by_family: Dict[str, List[Dict[str, Any]]] = {}
    top_by_family: Dict[str, Dict[str, Any]] = {}

    for family in pattern_families():
        rows = compute_family_rows(records_list, family, min_support=min_support)
        rows_by_family[family["name"]] = rows
        top_by_family[family["name"]] = {
            "top_failure_lift": rows[:top_k],
            "top_failure_key_lifts": top_failure_key_rows(
                rows, top_k=top_k, min_key_count=min_key_count
            ),
        }

    obstacle_result = compute_attribution(
        records_list,
        AttributionConfig(min_support=max(1, int(obstacle_min_support)), top_k=top_k),
        metadata={"comparison": "single_obstacle_cell"},
    )
    obstacle_top = [cell.to_dict() for cell in obstacle_result.top_cells[:top_k]]
    comparison = build_comparison(rows_by_family, obstacle_top, min_key_count=min_key_count)

    return {
        "metadata": {
            "method": "layout_level_attribution_phase1",
            "min_support": int(min_support),
            "top_k": int(top_k),
            "min_key_count": int(min_key_count),
            "pattern_families": [family["name"] for family in pattern_families()],
            **dict(metadata or {}),
        },
        "global_counts": global_counts,
        "comparison": comparison,
        "families": rows_by_family,
        "top_by_family": top_by_family,
        "single_obstacle_baseline": {
            "min_support": int(obstacle_min_support),
            "top_cells": obstacle_top,
        },
    }


def build_global_counts(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    term_counts: Counter = Counter()
    failure_key_counts: Counter = Counter()
    for record in records:
        termination = str(record.get("termination", "unknown"))
        failure_key = str(record.get("failure_key", termination))
        term_counts[termination] += 1
        if termination != "success":
            failure_key_counts[failure_key] += 1
    total = len(records)
    success = int(term_counts.get("success", 0))
    failure = total - success
    return {
        "num_episodes": total,
        "termination_counts": dict(term_counts),
        "failure_key_counts": dict(failure_key_counts),
        "success_count": success,
        "failure_count": failure,
        "failure_rate": failure / total if total else 0.0,
    }


def pattern_families() -> List[Dict[str, Any]]:
    return [
        {
            "name": "layout_basic",
            "keys": (
                "start_region",
                "path_len_bin",
                "num_obstacles_bin",
                "max_blockage_bin",
                "medhigh_count_bin",
                "pair_side_mode",
                "min_pair_dist_bin",
            ),
        },
        {
            "name": "layout_path_conditioned",
            "keys": (
                "start_region",
                "path_angle_bin",
                "path_len_bin",
                "near_count_bin",
                "medhigh_count_bin",
                "pair_side_mode",
            ),
        },
        {
            "name": "layout_pressure",
            "keys": (
                "path_len_bin",
                "max_blockage_bin",
                "medhigh_count_bin",
                "near_count_bin",
                "center_count_bin",
                "min_pair_dist_bin",
                "pressure_bin",
            ),
        },
        {
            "name": "start_behavior_context",
            "keys": (
                "start_region",
                "start_theta_bin",
                "path_angle_bin",
                "near_count_bin",
                "far_count_bin",
                "medhigh_count_bin",
            ),
        },
    ]


def compute_family_rows(
    records: Sequence[Mapping[str, Any]],
    family: Mapping[str, Any],
    min_support: int,
) -> List[Dict[str, Any]]:
    stats: Dict[str, Dict[str, Any]] = defaultdict(empty_pattern_stats)
    global_failure_keys: Counter = Counter()

    for record in records:
        layout = encode_layout(record)
        pattern_id = format_pattern(layout, family["keys"])
        termination = str(record.get("termination", "unknown"))
        failure_key = str(record.get("failure_key", termination))
        failed = termination != "success"
        row = stats[pattern_id]
        row["total_count"] += 1
        row["termination_counts"][termination] += 1
        row["reward_sum"] += safe_float(record.get("reward", 0.0))
        row["steps_sum"] += safe_float(record.get("steps", 0.0))
        row["examples"].append(example_record(record, layout))
        if failed:
            row["failure_count"] += 1
            row["failure_key_counts"][failure_key] += 1
            global_failure_keys[failure_key] += 1
        else:
            row["success_count"] += 1

    total_patterns = sum(row["total_count"] for row in stats.values())
    total_failures = sum(row["failure_count"] for row in stats.values())
    rows: List[Dict[str, Any]] = []
    for pattern_id, row in stats.items():
        total_count = int(row["total_count"])
        failure_count = int(row["failure_count"])
        success_count = int(row["success_count"])
        p_pattern = total_count / total_patterns if total_patterns else 0.0
        p_pattern_given_failure = failure_count / total_failures if total_failures else 0.0
        failure_lift = p_pattern_given_failure / p_pattern if p_pattern > 0 else 0.0
        dominant_key, dominant_key_count = most_common(row["failure_key_counts"])
        failure_key_lifts: Dict[str, float] = {}
        for key, global_count in global_failure_keys.items():
            key_count = int(row["failure_key_counts"].get(key, 0))
            p_pattern_given_key = key_count / global_count if global_count else 0.0
            failure_key_lifts[key] = (
                p_pattern_given_key / p_pattern if p_pattern > 0 else 0.0
            )

        rows.append(
            {
                "family": family["name"],
                "pattern_id": pattern_id,
                "total_count": total_count,
                "success_count": success_count,
                "failure_count": failure_count,
                "failure_rate": failure_count / total_count if total_count else 0.0,
                "p_pattern": p_pattern,
                "p_pattern_given_failure": p_pattern_given_failure,
                "failure_lift": failure_lift,
                "dominant_failure_key": dominant_key,
                "dominant_failure_count": dominant_key_count,
                "mean_reward": row["reward_sum"] / total_count if total_count else 0.0,
                "mean_steps": row["steps_sum"] / total_count if total_count else 0.0,
                "termination_counts": dict(row["termination_counts"]),
                "failure_key_counts": dict(row["failure_key_counts"]),
                "failure_key_lifts": failure_key_lifts,
                "examples": row["examples"][:3],
                "supported": total_count >= min_support,
            }
        )

    rows.sort(
        key=lambda row: (
            row["supported"],
            row["failure_lift"],
            row["failure_count"],
            row["total_count"],
        ),
        reverse=True,
    )
    return rows


def empty_pattern_stats() -> Dict[str, Any]:
    return {
        "total_count": 0,
        "success_count": 0,
        "failure_count": 0,
        "termination_counts": Counter(),
        "failure_key_counts": Counter(),
        "reward_sum": 0.0,
        "steps_sum": 0.0,
        "examples": [],
    }


def encode_layout(record: Mapping[str, Any]) -> Dict[str, Any]:
    scene_graph = record.get("scene_graph")
    if isinstance(scene_graph, Mapping):
        try:
            layout_z = extract_layout_z(scene_graph)
            return encode_layout_from_layout_z(layout_z)
        except Exception:
            pass

    start_pose = list_or_empty(record.get("start_pose"))
    goal_pose = list_or_empty(record.get("goal_pose"))
    obstacles_z = [z for z in list_or_empty(record.get("obstacle_z")) if isinstance(z, Mapping)]
    obstacles = [obs for obs in list_or_empty(record.get("obstacle_config")) if isinstance(obs, Mapping)]

    start_x = safe_float(start_pose[0] if len(start_pose) > 0 else 0.0)
    start_y = safe_float(start_pose[1] if len(start_pose) > 1 else 0.0)
    start_theta = wrap_angle(safe_float(start_pose[2] if len(start_pose) > 2 else 0.0))
    goal_x = safe_float(goal_pose[0] if len(goal_pose) > 0 else 0.0)
    goal_y = safe_float(goal_pose[1] if len(goal_pose) > 1 else 0.0)
    dx = goal_x - start_x
    dy = goal_y - start_y
    path_len = float(math.hypot(dx, dy))
    path_angle = wrap_angle(math.atan2(dy, dx) if path_len > EPS else 0.0)

    blockages = [safe_float(z.get("blockage", 0.0)) for z in obstacles_z]
    beta_values = [safe_float(z.get("beta", 0.0)) for z in obstacles_z]
    abs_beta = [abs(v) for v in beta_values]
    alpha_values = [safe_float(z.get("alpha", 0.0)) for z in obstacles_z]
    n_obs = len(obstacles_z)
    max_blockage = max(blockages) if blockages else 0.0
    mean_blockage = float(np.mean(blockages)) if blockages else 0.0
    near_count = sum(1 for v in abs_beta if 0.1 <= v < 0.25)
    center_count = sum(1 for v in abs_beta if v < 0.1)
    far_count = sum(1 for v in abs_beta if v >= 0.25)
    medhigh_count = sum(1 for v in blockages if v >= 0.33)
    pressure = combined_corridor_pressure(alpha_values, blockages)
    pair_distances = compute_pair_distances(obstacles_z, obstacles)
    min_pair_distance = min(pair_distances) if pair_distances else None
    mean_pair_distance = float(np.mean(pair_distances)) if pair_distances else None
    pair_side_mode = compute_pair_side_mode(beta_values)

    return {
        "start_x": start_x,
        "start_y": start_y,
        "start_theta": start_theta,
        "path_len": path_len,
        "path_angle": path_angle,
        "num_obstacles": n_obs,
        "mean_blockage": mean_blockage,
        "max_blockage": max_blockage,
        "near_count": near_count,
        "center_count": center_count,
        "far_count": far_count,
        "medhigh_count": medhigh_count,
        "min_pair_distance": min_pair_distance,
        "mean_pair_distance": mean_pair_distance,
        "pair_side_mode": pair_side_mode,
        "combined_corridor_pressure": pressure,
        "start_region": xy_region(start_x, start_y),
        "start_theta_bin": angle_bin(start_theta),
        "path_angle_bin": angle_bin(path_angle),
        "path_len_bin": path_len_bin(path_len),
        "num_obstacles_bin": count_bin(n_obs, cap=3),
        "mean_blockage_bin": blockage_bin(mean_blockage),
        "max_blockage_bin": blockage_bin(max_blockage),
        "near_count_bin": count_bin(near_count, cap=3),
        "center_count_bin": count_bin(center_count, cap=3),
        "far_count_bin": count_bin(far_count, cap=3),
        "medhigh_count_bin": count_bin(medhigh_count, cap=3),
        "min_pair_dist_bin": pair_distance_bin(min_pair_distance),
        "pressure_bin": pressure_bin(pressure),
    }


def encode_layout_from_layout_z(layout_z: Mapping[str, Any]) -> Dict[str, Any]:
    axis = layout_z.get("axis") if isinstance(layout_z.get("axis"), Mapping) else {}
    start = layout_z.get("start") if isinstance(layout_z.get("start"), Mapping) else {}
    obstacle_edges = [
        edge
        for edge in list_or_empty(layout_z.get("obstacle_axis_edges"))
        if isinstance(edge, Mapping)
    ]
    pair_edges = [
        edge
        for edge in list_or_empty(layout_z.get("obstacle_pair_edges"))
        if isinstance(edge, Mapping)
    ]

    start_x = safe_float(start.get("x", 0.0))
    start_y = safe_float(start.get("y", 0.0))
    start_theta = wrap_angle(safe_float(start.get("theta", 0.0)))
    path_len = safe_float(axis.get("length", 0.0))
    path_angle = wrap_angle(safe_float(axis.get("angle", 0.0)))
    alpha_values = [safe_float(edge.get("alpha", 0.0)) for edge in obstacle_edges]
    beta_values = [safe_float(edge.get("beta", 0.0)) for edge in obstacle_edges]
    abs_beta = [abs(value) for value in beta_values]
    blockages = [safe_float(edge.get("blockage", 0.0)) for edge in obstacle_edges]
    n_obs = len(obstacle_edges)
    max_blockage = max(blockages) if blockages else 0.0
    mean_blockage = float(np.mean(blockages)) if blockages else 0.0
    near_count = sum(1 for value in abs_beta if 0.1 <= value < 0.25)
    center_count = sum(1 for value in abs_beta if value < 0.1)
    far_count = sum(1 for value in abs_beta if value >= 0.25)
    medhigh_count = sum(1 for value in blockages if value >= 0.33)
    pressure = combined_corridor_pressure(alpha_values, blockages)
    pair_distances = [safe_float(edge.get("distance")) for edge in pair_edges]
    min_pair_distance = min(pair_distances) if pair_distances else None
    mean_pair_distance = float(np.mean(pair_distances)) if pair_distances else None
    pair_side_mode = compute_pair_side_mode(beta_values)

    return {
        "start_x": start_x,
        "start_y": start_y,
        "start_theta": start_theta,
        "path_len": path_len,
        "path_angle": path_angle,
        "num_obstacles": n_obs,
        "mean_blockage": mean_blockage,
        "max_blockage": max_blockage,
        "near_count": near_count,
        "center_count": center_count,
        "far_count": far_count,
        "medhigh_count": medhigh_count,
        "min_pair_distance": min_pair_distance,
        "mean_pair_distance": mean_pair_distance,
        "pair_side_mode": pair_side_mode,
        "combined_corridor_pressure": pressure,
        "start_region": xy_region(start_x, start_y),
        "start_theta_bin": angle_bin(start_theta),
        "path_angle_bin": angle_bin(path_angle),
        "path_len_bin": path_len_bin(path_len),
        "num_obstacles_bin": count_bin(n_obs, cap=3),
        "mean_blockage_bin": blockage_bin(mean_blockage),
        "max_blockage_bin": blockage_bin(max_blockage),
        "near_count_bin": count_bin(near_count, cap=3),
        "center_count_bin": count_bin(center_count, cap=3),
        "far_count_bin": count_bin(far_count, cap=3),
        "medhigh_count_bin": count_bin(medhigh_count, cap=3),
        "min_pair_dist_bin": pair_distance_bin(min_pair_distance),
        "pressure_bin": pressure_bin(pressure),
    }


def combined_corridor_pressure(alphas: Sequence[float], blockages: Sequence[float]) -> float:
    score = 0.0
    for alpha, blockage in zip(alphas, blockages):
        alpha = float(alpha)
        blockage = float(blockage)
        if 0.0 <= alpha <= 1.0:
            alpha_weight = 1.0
        elif -0.25 <= alpha < 0.0 or 1.0 < alpha <= 1.25:
            alpha_weight = 0.5
        else:
            alpha_weight = 0.15
        score += alpha_weight * blockage
    return float(score)


def compute_pair_distances(
    obstacles_z: Sequence[Mapping[str, Any]],
    obstacles: Sequence[Mapping[str, Any]],
) -> List[float]:
    points: List[Tuple[float, float]] = []
    for z in obstacles_z:
        if "abs_x" in z and "abs_y" in z:
            points.append((safe_float(z.get("abs_x")), safe_float(z.get("abs_y"))))
    if len(points) < len(obstacles):
        points = []
        for obs in obstacles:
            if "x" in obs and "y" in obs:
                points.append((safe_float(obs.get("x")), safe_float(obs.get("y"))))

    distances: List[float] = []
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            dx = points[i][0] - points[j][0]
            dy = points[i][1] - points[j][1]
            distances.append(float(math.hypot(dx, dy)))
    return distances


def compute_pair_side_mode(beta_values: Sequence[float]) -> str:
    signs = []
    for beta in beta_values:
        if abs(beta) < 0.1:
            signs.append(0)
        elif beta > 0:
            signs.append(1)
        else:
            signs.append(-1)
    if len(signs) < 2:
        return "single"
    nonzero = [s for s in signs if s != 0]
    if len(nonzero) < 2:
        return "center_involved"
    has_same = False
    has_opp = False
    for i in range(len(nonzero)):
        for j in range(i + 1, len(nonzero)):
            if nonzero[i] == nonzero[j]:
                has_same = True
            else:
                has_opp = True
    if has_same and has_opp:
        return "mixed_side"
    if has_opp:
        return "opposite_side"
    return "same_side"


def format_pattern(layout: Mapping[str, Any], keys: Sequence[str]) -> str:
    return "|".join(f"{key}={layout.get(key, 'unknown')}" for key in keys)


def top_failure_key_rows(
    rows: Sequence[Mapping[str, Any]],
    top_k: int,
    min_key_count: int = 1,
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for row in rows:
        if not row.get("supported"):
            continue
        lifts = row.get("failure_key_lifts") or {}
        counts = row.get("failure_key_counts") or {}
        for key, lift in lifts.items():
            count = int(counts.get(key, 0))
            if count < min_key_count:
                continue
            candidates.append(
                {
                    "family": row["family"],
                    "pattern_id": row["pattern_id"],
                    "failure_key": key,
                    "failure_key_lift": float(lift),
                    "failure_key_count": count,
                    "total_count": row["total_count"],
                    "failure_rate": row["failure_rate"],
                    "mean_reward": row["mean_reward"],
                    "mean_steps": row["mean_steps"],
                    "examples": row.get("examples", [])[:3],
                }
            )
    candidates.sort(
        key=lambda row: (
            row["failure_key_lift"],
            row["failure_key_count"],
            row["total_count"],
        ),
        reverse=True,
    )
    return candidates[:top_k]


def build_comparison(
    rows_by_family: Mapping[str, Sequence[Mapping[str, Any]]],
    obstacle_top: Sequence[Mapping[str, Any]],
    min_key_count: int = 1,
) -> Dict[str, Any]:
    obstacle_supported = [row for row in obstacle_top if row.get("total_count", 0) > 0]
    obstacle_best = obstacle_supported[0] if obstacle_supported else None
    family_best = {}
    for family, rows in rows_by_family.items():
        supported = [row for row in rows if row.get("supported")]
        best = supported[0] if supported else None
        key_best = top_failure_key_rows(rows, top_k=1, min_key_count=min_key_count)
        family_best[family] = {
            "best_failure_lift": simplify_row(best),
            "best_failure_key_lift": simplify_key_row(key_best[0] if key_best else None),
        }
    return {
        "single_obstacle_best": simplify_obstacle_row(obstacle_best),
        "layout_family_best": family_best,
    }


def write_outputs(result: Mapping[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(result, output_dir / "layout_attribution_map.json")
    write_csv(result, output_dir / "layout_patterns.csv")
    summary = build_summary_text(result)
    (output_dir / "layout_attribution_summary.txt").write_text(summary, encoding="utf-8")


def write_csv(result: Mapping[str, Any], path: Path) -> None:
    rows: List[Dict[str, Any]] = []
    for family_rows in (result.get("families") or {}).values():
        for row in family_rows:
            rows.append(flatten_row(row))
    fields = [
        "family",
        "pattern_id",
        "supported",
        "total_count",
        "success_count",
        "failure_count",
        "failure_rate",
        "failure_lift",
        "dominant_failure_key",
        "dominant_failure_count",
        "mean_reward",
        "mean_steps",
        "termination_counts",
        "failure_key_counts",
        "failure_key_lifts",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def build_summary_text(result: Mapping[str, Any]) -> str:
    counts = result["global_counts"]
    lines = [
        "Layout-Level Failure Attribution Summary",
        "========================================",
        f"rollouts: {result['metadata'].get('rollouts')}",
        f"episodes: {counts['num_episodes']}",
        f"failure_rate: {counts['failure_rate']:.3f}",
        f"min_support: {result['metadata']['min_support']}",
        f"min_key_count: {result['metadata'].get('min_key_count', 1)}",
        "",
        "Termination counts:",
    ]
    for key, value in sorted(counts["termination_counts"].items()):
        lines.append(f"- {key}: {value}")

    lines.extend(["", "Single-obstacle baseline best:"])
    single = result["comparison"].get("single_obstacle_best")
    if single:
        lines.append(
            f"- {single['cell_id']} | support={single['support']}, "
            f"failure_rate={single['failure_rate']:.3f}, lift={single['failure_lift']:.3f}"
        )
    else:
        lines.append("- none")

    lines.extend(["", "Best layout-level patterns by family:"])
    family_best = result["comparison"].get("layout_family_best", {})
    for family, payload in family_best.items():
        lines.append(f"\n[{family}]")
        best = payload.get("best_failure_lift")
        if best:
            lines.append(
                f"- failure_lift: {best['pattern_id']} | support={best['support']}, "
                f"failure_rate={best['failure_rate']:.3f}, lift={best['failure_lift']:.3f}, "
                f"dominant={best['dominant_failure_key']}"
            )
        else:
            lines.append("- failure_lift: none")
        key_best = payload.get("best_failure_key_lift")
        if key_best:
            lines.append(
                f"- failure_key_lift: key={key_best['failure_key']}, "
                f"lift={key_best['failure_key_lift']:.3f}, count={key_best['failure_key_count']}, "
                f"pattern={key_best['pattern_id']}"
            )
        else:
            lines.append("- failure_key_lift: none")

    lines.extend(["", "Top failure-lift rows:"])
    top_by_family = result.get("top_by_family") or {}
    for family, payload in top_by_family.items():
        rows = payload.get("top_failure_lift") or []
        lines.append(f"\n[{family}]")
        for idx, row in enumerate(rows[:5], start=1):
            lines.append(
                f"{idx}. {row['pattern_id']} | support={row['total_count']}, "
                f"failure_rate={row['failure_rate']:.3f}, lift={row['failure_lift']:.3f}, "
                f"dominant={row['dominant_failure_key']} ({row['dominant_failure_count']})"
            )
    return "\n".join(lines) + "\n"


def example_record(record: Mapping[str, Any], layout: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "episode_id": record.get("episode_id"),
        "termination": record.get("termination"),
        "failure_key": record.get("failure_key"),
        "reward": record.get("reward"),
        "steps": record.get("steps"),
        "start_pose": record.get("start_pose"),
        "obstacle_config": record.get("obstacle_config"),
        "layout_excerpt": {
            "start_region": layout.get("start_region"),
            "path_len_bin": layout.get("path_len_bin"),
            "path_angle_bin": layout.get("path_angle_bin"),
            "max_blockage_bin": layout.get("max_blockage_bin"),
            "near_count_bin": layout.get("near_count_bin"),
            "medhigh_count_bin": layout.get("medhigh_count_bin"),
            "pair_side_mode": layout.get("pair_side_mode"),
        },
    }


def flatten_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    flat = dict(row)
    for key in ("termination_counts", "failure_key_counts", "failure_key_lifts"):
        flat[key] = json.dumps(flat.get(key, {}), ensure_ascii=True, sort_keys=True)
    flat.pop("examples", None)
    return flat


def simplify_row(row: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    if not row:
        return None
    return {
        "family": row["family"],
        "pattern_id": row["pattern_id"],
        "support": row["total_count"],
        "failure_rate": row["failure_rate"],
        "failure_lift": row["failure_lift"],
        "dominant_failure_key": row["dominant_failure_key"],
        "dominant_failure_count": row["dominant_failure_count"],
        "mean_reward": row["mean_reward"],
        "mean_steps": row["mean_steps"],
    }


def simplify_key_row(row: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    if not row:
        return None
    return {
        "family": row["family"],
        "pattern_id": row["pattern_id"],
        "failure_key": row["failure_key"],
        "failure_key_lift": row["failure_key_lift"],
        "failure_key_count": row["failure_key_count"],
        "support": row["total_count"],
        "failure_rate": row["failure_rate"],
    }


def simplify_obstacle_row(row: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    if not row:
        return None
    return {
        "cell_id": row["cell_id"],
        "support": row["total_count"],
        "failure_rate": row["failure_rate"],
        "failure_lift": row["failure_lift"],
        "dominant_failure_key": row["dominant_failure_key"],
        "dominant_failure_count": row["dominant_failure_count"],
    }


def write_json(data: Mapping[str, Any], path: Path) -> None:
    path.write_text(json.dumps(to_jsonable(data), indent=2, ensure_ascii=True), encoding="utf-8")


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def xy_region(x: float, y: float) -> str:
    return f"x_{axis_bin(x)}__y_{axis_bin(y)}"


def axis_bin(value: float) -> str:
    if value < -0.07:
        return "neg"
    if value > 0.07:
        return "pos"
    return "mid"


def angle_bin(angle: float, sectors: int = 8) -> str:
    wrapped = wrap_angle(angle)
    idx = int(math.floor((wrapped + math.pi) / (2.0 * math.pi) * sectors))
    idx = max(0, min(sectors - 1, idx))
    return f"sector_{idx}"


def path_len_bin(length: float) -> str:
    if length < 0.08:
        return "short"
    if length < 0.14:
        return "medium"
    return "long"


def blockage_bin(value: float) -> str:
    if value < 0.33:
        return "low"
    if value < 0.67:
        return "medium"
    return "high"


def pressure_bin(value: float) -> str:
    if value < 0.33:
        return "low"
    if value < 0.85:
        return "medium"
    return "high"


def count_bin(value: int, cap: int = 3) -> str:
    value = int(value)
    if value >= cap:
        return f"{cap}_plus"
    return str(value)


def pair_distance_bin(value: Optional[float]) -> str:
    if value is None:
        return "none"
    if value < 0.05:
        return "very_close"
    if value < 0.09:
        return "close"
    if value < 0.14:
        return "medium"
    return "far"


def wrap_angle(angle: float) -> float:
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return float(default)
    if math.isnan(val) or math.isinf(val):
        return float(default)
    return val


def list_or_empty(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def most_common(counter: Mapping[str, int]) -> Tuple[str, int]:
    if not counter:
        return "none", 0
    key, value = Counter(counter).most_common(1)[0]
    return str(key), int(value)


if __name__ == "__main__":
    main()
