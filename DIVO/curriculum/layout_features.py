from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np


def extract_layout_z(scene_graph: Mapping[str, Any]) -> Dict[str, Any]:
    """Extract a raw graph-derived layout representation from a scene graph.

    The returned representation intentionally contains only numeric graph
    features and permutation-invariant summaries. It does not create manual
    bins or named motifs; those should be discovered by downstream attribution.
    """

    nodes = [node for node in scene_graph.get("nodes", []) if isinstance(node, Mapping)]
    edges = [edge for edge in scene_graph.get("edges", []) if isinstance(edge, Mapping)]
    node_by_id = {str(node.get("id", "")): node for node in nodes}

    start = _extract_pose_node(node_by_id.get("start"))
    task_axis = _extract_task_axis(node_by_id.get("task_axis"), scene_graph)
    obstacle_axis_edges = _extract_obstacle_axis_edges(edges)
    obstacle_pair_edges = _extract_obstacle_pair_edges(edges)

    return {
        "version": "layout_z_raw_v1",
        "source": {
            "type": "scene_graph",
            "version": str(scene_graph.get("version", "")),
            "task": str(scene_graph.get("task", "")),
        },
        "axis": task_axis,
        "start": start,
        "obstacle_axis_edges": obstacle_axis_edges,
        "obstacle_pair_edges": obstacle_pair_edges,
        "summary": _build_summary(task_axis, obstacle_axis_edges, obstacle_pair_edges),
    }


def _extract_pose_node(node: Optional[Mapping[str, Any]]) -> Dict[str, float]:
    pose = list(node.get("pose", []) if isinstance(node, Mapping) else [])
    while len(pose) < 3:
        pose.append(0.0)
    return {
        "x": _safe_float(pose[0]),
        "y": _safe_float(pose[1]),
        "theta": _wrap_angle(_safe_float(pose[2])),
    }


def _extract_task_axis(
    node: Optional[Mapping[str, Any]],
    scene_graph: Mapping[str, Any],
) -> Dict[str, float]:
    features = node.get("features", {}) if isinstance(node, Mapping) else {}
    global_features = scene_graph.get("global_features", {})
    if not isinstance(features, Mapping):
        features = {}
    if not isinstance(global_features, Mapping):
        global_features = {}
    return {
        "length": _safe_float(features.get("length", global_features.get("axis_length", 0.0))),
        "angle": _wrap_angle(_safe_float(features.get("angle", global_features.get("axis_angle", 0.0)))),
    }


def _extract_obstacle_axis_edges(edges: Sequence[Mapping[str, Any]]) -> List[Dict[str, float]]:
    axis_edges: List[Dict[str, float]] = []
    for edge in edges:
        if edge.get("type") != "obstacle_task_axis":
            continue
        features = edge.get("features", {})
        if not isinstance(features, Mapping):
            features = {}
        beta = _safe_float(features.get("beta", 0.0))
        axis_edges.append(
            {
                "obstacle_id": str(edge.get("src", "")),
                "alpha": _safe_float(features.get("alpha", 0.0)),
                "beta": beta,
                "abs_beta": abs(beta),
                "blockage": _safe_float(features.get("blockage", 0.0)),
                "d_start": _safe_float(features.get("d_start", 0.0)),
                "d_goal": _safe_float(features.get("d_goal", 0.0)),
                "effective_radius": _safe_float(features.get("effective_radius", 0.0)),
            }
        )
    axis_edges.sort(key=lambda item: item["obstacle_id"])
    return axis_edges


def _extract_obstacle_pair_edges(edges: Sequence[Mapping[str, Any]]) -> List[Dict[str, float]]:
    pair_edges: List[Dict[str, float]] = []
    for edge in edges:
        if edge.get("type") != "obstacle_pair":
            continue
        features = edge.get("features", {})
        if not isinstance(features, Mapping):
            features = {}
        src = str(edge.get("src", ""))
        dst = str(edge.get("dst", ""))
        pair_edges.append(
            {
                "src": src,
                "dst": dst,
                "distance": _safe_float(features.get("distance", 0.0)),
                "delta_alpha": _safe_float(features.get("delta_alpha", 0.0)),
                "delta_beta": _safe_float(features.get("delta_beta", 0.0)),
                "min_blockage": _safe_float(features.get("min_blockage", 0.0)),
                "max_blockage": _safe_float(features.get("max_blockage", 0.0)),
                "mean_blockage": _safe_float(features.get("mean_blockage", 0.0)),
            }
        )
    pair_edges.sort(key=lambda item: (item["src"], item["dst"]))
    return pair_edges


def _build_summary(
    axis: Mapping[str, float],
    obstacle_axis_edges: Sequence[Mapping[str, float]],
    obstacle_pair_edges: Sequence[Mapping[str, float]],
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "axis_length": _safe_float(axis.get("length", 0.0)),
        "axis_angle": _wrap_angle(_safe_float(axis.get("angle", 0.0))),
        "num_obstacles": len(obstacle_axis_edges),
        "num_pairs": len(obstacle_pair_edges),
    }

    _add_stats(summary, "alpha", [edge.get("alpha", 0.0) for edge in obstacle_axis_edges])
    _add_stats(summary, "beta", [edge.get("beta", 0.0) for edge in obstacle_axis_edges])
    _add_stats(summary, "abs_beta", [edge.get("abs_beta", 0.0) for edge in obstacle_axis_edges])
    _add_stats(summary, "blockage", [edge.get("blockage", 0.0) for edge in obstacle_axis_edges], include_sum=True)
    _add_stats(summary, "d_start", [edge.get("d_start", 0.0) for edge in obstacle_axis_edges])
    _add_stats(summary, "d_goal", [edge.get("d_goal", 0.0) for edge in obstacle_axis_edges])
    _add_stats(summary, "effective_radius", [edge.get("effective_radius", 0.0) for edge in obstacle_axis_edges])

    _add_stats(summary, "pair_distance", [edge.get("distance", 0.0) for edge in obstacle_pair_edges])
    _add_stats(summary, "pair_delta_alpha", [edge.get("delta_alpha", 0.0) for edge in obstacle_pair_edges])
    _add_stats(summary, "pair_delta_beta", [edge.get("delta_beta", 0.0) for edge in obstacle_pair_edges])
    _add_stats(summary, "pair_min_blockage", [edge.get("min_blockage", 0.0) for edge in obstacle_pair_edges])
    _add_stats(summary, "pair_max_blockage", [edge.get("max_blockage", 0.0) for edge in obstacle_pair_edges])
    _add_stats(summary, "pair_mean_blockage", [edge.get("mean_blockage", 0.0) for edge in obstacle_pair_edges])

    return summary


def _add_stats(
    output: Dict[str, Any],
    prefix: str,
    values: Sequence[Any],
    include_sum: bool = False,
) -> None:
    clean = [_safe_float(value) for value in values]
    if not clean:
        output[f"{prefix}_min"] = None
        output[f"{prefix}_mean"] = None
        output[f"{prefix}_max"] = None
        output[f"{prefix}_std"] = None
        if include_sum:
            output[f"{prefix}_sum"] = 0.0
        return

    arr = np.asarray(clean, dtype=np.float64)
    output[f"{prefix}_min"] = float(np.min(arr))
    output[f"{prefix}_mean"] = float(np.mean(arr))
    output[f"{prefix}_max"] = float(np.max(arr))
    output[f"{prefix}_std"] = float(np.std(arr))
    if include_sum:
        output[f"{prefix}_sum"] = float(np.sum(arr))


def _wrap_angle(angle: float) -> float:
    return float((float(angle) + math.pi) % (2.0 * math.pi) - math.pi)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return float(default)
    if math.isnan(val) or math.isinf(val):
        return float(default)
    return val
