from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np


EPS = 1e-12


@dataclass(frozen=True)
class PatternDiscoveryConfig:
    num_bins: int = 4
    max_conditions: int = 3
    min_support: int = 100
    min_failure_count: int = 20
    top_k: int = 20
    beam_width: int = 250
    max_overlap: float = 0.90
    min_lift: float = 1.0


def discover_failure_graph_patterns(
    records: Iterable[Mapping[str, Any]],
    config: Optional[PatternDiscoveryConfig] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Discover failure-associated layout graph patterns from raw layout_z records."""

    cfg = config or PatternDiscoveryConfig()
    rows = [record for record in records if isinstance(record, Mapping)]
    feature_rows = [flatten_layout_z_features(row.get("layout_z", {})) for row in rows]
    labels = [str(row.get("termination", "unknown")) for row in rows]
    failure_keys = [str(row.get("failure_key", labels[idx])) for idx, row in enumerate(rows)]
    total_count = len(rows)
    failure_indices = {idx for idx, label in enumerate(labels) if label != "success"}
    failure_count = len(failure_indices)
    global_failure_rate = failure_count / total_count if total_count else 0.0
    failure_key_counts = _failure_key_counts(failure_keys, labels)

    predicates = build_quantile_predicates(feature_rows, cfg)
    atomic_candidates = [
        _make_candidate((predicate,), predicate["matches"], rows, labels, failure_keys, failure_key_counts)
        for predicate in predicates
        if len(predicate["matches"]) >= cfg.min_support
    ]
    all_candidates = [
        candidate
        for candidate in atomic_candidates
        if _candidate_is_valid(candidate, cfg, global_failure_rate)
    ]

    frontier = sorted(
        atomic_candidates,
        key=lambda item: (item["score"], item["failure_lift"], item["support"]),
        reverse=True,
    )[: cfg.beam_width]

    for depth in range(2, max(1, cfg.max_conditions) + 1):
        next_candidates = []
        seen_keys: Set[Tuple[str, ...]] = set()
        for candidate in frontier:
            used_ids = set(candidate["_predicate_ids"])
            last_id = candidate["_predicate_ids"][-1]
            used_features = set(candidate["_features"])
            for predicate in predicates:
                predicate_id = predicate["id"]
                if predicate_id <= last_id or predicate_id in used_ids:
                    continue
                if predicate["feature"] in used_features:
                    continue
                predicate_ids = tuple(sorted((*candidate["_predicate_ids"], predicate_id)))
                if predicate_ids in seen_keys:
                    continue
                seen_keys.add(predicate_ids)
                matches = candidate["_matches"].intersection(predicate["matches"])
                if len(matches) < cfg.min_support:
                    continue
                predicate_map = {pred["id"]: pred for pred in predicates}
                candidate_predicates = tuple(predicate_map[pred_id] for pred_id in predicate_ids)
                next_candidates.append(
                    _make_candidate(
                        candidate_predicates,
                        matches,
                        rows,
                        labels,
                        failure_keys,
                        failure_key_counts,
                    )
                )
        valid_next = [
            candidate
            for candidate in next_candidates
            if _candidate_is_valid(candidate, cfg, global_failure_rate)
        ]
        all_candidates.extend(valid_next)
        frontier = sorted(
            next_candidates,
            key=lambda item: (item["score"], item["failure_lift"], item["support"]),
            reverse=True,
        )[: cfg.beam_width]
        if not frontier:
            break

    selected = select_nonredundant_patterns(all_candidates, cfg)
    public_patterns = [strip_internal_fields(pattern) for pattern in selected[: cfg.top_k]]

    feature_count = len(set(itertools.chain.from_iterable(row.keys() for row in feature_rows)))

    return {
        "metadata": {
            "method": "failure_conditioned_graph_pattern_mining",
            "discretization": "quantile",
            "num_bins": cfg.num_bins,
            "max_conditions": cfg.max_conditions,
            "min_support": cfg.min_support,
            "min_failure_count": cfg.min_failure_count,
            "top_k": cfg.top_k,
            **dict(metadata or {}),
        },
        "global_stats": {
            "num_episodes": total_count,
            "success_count": int(labels.count("success")),
            "failure_count": failure_count,
            "failure_rate": global_failure_rate,
            "failure_key_counts": failure_key_counts,
        },
        "feature_count": feature_count,
        "atomic_predicate_count": len(predicates),
        "candidate_count": len(all_candidates),
        "top_patterns": public_patterns,
    }


def flatten_layout_z_features(layout_z: Mapping[str, Any]) -> Dict[str, float]:
    """Flatten numeric layout_z fields into a feature table row."""

    features: Dict[str, float] = {}
    if not isinstance(layout_z, Mapping):
        return features

    axis = layout_z.get("axis", {})
    if isinstance(axis, Mapping):
        _add_numeric(features, "axis.length", axis.get("length"))
        _add_numeric(features, "axis.angle", axis.get("angle"))

    start = layout_z.get("start", {})
    if isinstance(start, Mapping):
        _add_numeric(features, "start.x", start.get("x"))
        _add_numeric(features, "start.y", start.get("y"))
        _add_numeric(features, "start.theta", start.get("theta"))

    summary = layout_z.get("summary", {})
    if isinstance(summary, Mapping):
        for key, value in summary.items():
            _add_numeric(features, f"summary.{key}", value)

    return features


def build_quantile_predicates(
    feature_rows: Sequence[Mapping[str, float]],
    config: PatternDiscoveryConfig,
) -> List[Dict[str, Any]]:
    predicates: List[Dict[str, Any]] = []
    if not feature_rows:
        return predicates

    features = sorted(set(itertools.chain.from_iterable(row.keys() for row in feature_rows)))
    for feature in features:
        values_by_index = {
            idx: float(row[feature])
            for idx, row in enumerate(feature_rows)
            if feature in row and _is_finite(row[feature])
        }
        if len(values_by_index) < config.min_support:
            continue
        values = np.asarray(list(values_by_index.values()), dtype=np.float64)
        unique_values = np.unique(values)
        if unique_values.size <= 1:
            continue
        if unique_values.size <= max(2, config.num_bins):
            predicates.extend(_equality_predicates(feature, values_by_index, unique_values, config))
        else:
            predicates.extend(_quantile_interval_predicates(feature, values_by_index, values, config))

    predicates.sort(key=lambda predicate: predicate["id"])
    return predicates


def _equality_predicates(
    feature: str,
    values_by_index: Mapping[int, float],
    unique_values: Sequence[float],
    config: PatternDiscoveryConfig,
) -> List[Dict[str, Any]]:
    predicates: List[Dict[str, Any]] = []
    for value in unique_values:
        matches = {idx for idx, val in values_by_index.items() if val == value}
        if len(matches) < config.min_support:
            continue
        label = f"{feature} == {_format_number(value)}"
        predicates.append(
            {
                "id": f"{feature}::eq::{_format_number(value)}",
                "feature": feature,
                "operator": "==",
                "value": float(value),
                "label": label,
                "matches": matches,
            }
        )
    return predicates


def _quantile_interval_predicates(
    feature: str,
    values_by_index: Mapping[int, float],
    values: np.ndarray,
    config: PatternDiscoveryConfig,
) -> List[Dict[str, Any]]:
    quantiles = np.linspace(0.0, 1.0, int(config.num_bins) + 1)
    edges = np.quantile(values, quantiles)
    edges = np.unique(edges)
    if edges.size <= 1:
        return []

    predicates: List[Dict[str, Any]] = []
    for bin_idx in range(edges.size - 1):
        lower = float(edges[bin_idx])
        upper = float(edges[bin_idx + 1])
        if bin_idx == 0:
            matches = {idx for idx, value in values_by_index.items() if value <= upper}
            label = f"{feature} <= {_format_number(upper)}"
        elif bin_idx == edges.size - 2:
            matches = {idx for idx, value in values_by_index.items() if value > lower}
            label = f"{feature} > {_format_number(lower)}"
        else:
            matches = {
                idx
                for idx, value in values_by_index.items()
                if value > lower and value <= upper
            }
            label = f"{_format_number(lower)} < {feature} <= {_format_number(upper)}"
        if len(matches) < config.min_support:
            continue
        predicates.append(
            {
                "id": f"{feature}::q{bin_idx}::{_format_number(lower)}::{_format_number(upper)}",
                "feature": feature,
                "operator": "interval",
                "lower": lower,
                "upper": upper,
                "bin_index": bin_idx,
                "label": label,
                "matches": matches,
            }
        )
    return predicates


def _make_candidate(
    predicates: Sequence[Mapping[str, Any]],
    matches: Set[int],
    rows: Sequence[Mapping[str, Any]],
    labels: Sequence[str],
    failure_keys: Sequence[str],
    failure_key_counts: Mapping[str, int],
) -> Dict[str, Any]:
    support = len(matches)
    failed = [idx for idx in matches if labels[idx] != "success"]
    failure_count = len(failed)
    failure_rate = failure_count / support if support else 0.0
    total_count = len(labels)
    global_failure_rate = sum(1 for label in labels if label != "success") / total_count if total_count else 0.0
    failure_lift = failure_rate / global_failure_rate if global_failure_rate > 0 else 0.0
    lcb = wilson_lower_bound(failure_count, support)
    lift_lcb = lcb / global_failure_rate if global_failure_rate > 0 else 0.0
    dominant_key, dominant_count = _dominant_failure_key(failed, failure_keys)
    pattern_probability = support / total_count if total_count else 0.0
    key_global_count = failure_key_counts.get(dominant_key, 0)
    dominant_key_lift = (
        (dominant_count / key_global_count) / pattern_probability
        if dominant_key != "none" and key_global_count > 0 and pattern_probability > 0
        else 0.0
    )
    example_ids = [rows[idx].get("episode_id") for idx in sorted(matches)[:5]]
    score = lift_lcb * math.log1p(support)

    return {
        "conditions": [predicate["label"] for predicate in predicates],
        "predicates": [
            {key: value for key, value in predicate.items() if key != "matches"}
            for predicate in predicates
        ],
        "support": support,
        "failure_count": failure_count,
        "failure_rate": failure_rate,
        "failure_lift": failure_lift,
        "failure_rate_lcb": lcb,
        "failure_lift_lcb": lift_lcb,
        "dominant_failure_key": dominant_key,
        "dominant_failure_count": dominant_count,
        "dominant_failure_lift": dominant_key_lift,
        "score": score,
        "example_episode_ids": example_ids,
        "_matches": set(matches),
        "_predicate_ids": tuple(predicate["id"] for predicate in predicates),
        "_features": tuple(predicate["feature"] for predicate in predicates),
    }


def _candidate_is_valid(
    candidate: Mapping[str, Any],
    config: PatternDiscoveryConfig,
    global_failure_rate: float,
) -> bool:
    if candidate["support"] < config.min_support:
        return False
    if candidate["failure_count"] < config.min_failure_count:
        return False
    if global_failure_rate <= 0:
        return False
    if candidate["failure_lift"] < config.min_lift:
        return False
    return True


def select_nonredundant_patterns(
    candidates: Sequence[Mapping[str, Any]],
    config: PatternDiscoveryConfig,
) -> List[Dict[str, Any]]:
    ordered = sorted(
        candidates,
        key=lambda item: (
            item["score"],
            item["failure_lift"],
            item["failure_rate"],
            item["support"],
        ),
        reverse=True,
    )
    selected: List[Dict[str, Any]] = []
    for candidate in ordered:
        candidate_matches = candidate["_matches"]
        redundant = False
        for existing in selected:
            overlap = jaccard(candidate_matches, existing["_matches"])
            if overlap >= config.max_overlap:
                redundant = True
                break
        if not redundant:
            selected.append(dict(candidate))
        if len(selected) >= config.top_k:
            break
    return selected


def strip_internal_fields(pattern: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in pattern.items()
        if not str(key).startswith("_")
    }


def wilson_lower_bound(successes: int, total: int, z: float = 1.96) -> float:
    if total <= 0:
        return 0.0
    phat = successes / total
    denom = 1.0 + z * z / total
    centre = phat + z * z / (2.0 * total)
    margin = z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * total)) / total)
    return max(0.0, (centre - margin) / denom)


def jaccard(first: Set[int], second: Set[int]) -> float:
    if not first and not second:
        return 1.0
    union = len(first.union(second))
    if union == 0:
        return 0.0
    return len(first.intersection(second)) / union


def _failure_key_counts(failure_keys: Sequence[str], labels: Sequence[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for key, label in zip(failure_keys, labels):
        if label == "success":
            continue
        counts[key] = counts.get(key, 0) + 1
    return counts


def _dominant_failure_key(indices: Sequence[int], failure_keys: Sequence[str]) -> Tuple[str, int]:
    counts: Dict[str, int] = {}
    for idx in indices:
        key = failure_keys[idx]
        counts[key] = counts.get(key, 0) + 1
    if not counts:
        return "none", 0
    key, count = max(counts.items(), key=lambda item: (item[1], item[0]))
    return str(key), int(count)


def _add_numeric(features: Dict[str, float], key: str, value: Any) -> None:
    if value is None:
        return
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return
    if _is_finite(numeric):
        features[key] = numeric


def _is_finite(value: Any) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(numeric)


def _format_number(value: float) -> str:
    return f"{float(value):.6g}"
