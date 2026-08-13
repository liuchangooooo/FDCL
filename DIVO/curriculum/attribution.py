from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


TERMINATION_KEYS = ("success", "collision", "timeout", "fall")


@dataclass(frozen=True)
class BinSpec:
    label: str
    low: float
    high: float

    @classmethod
    def from_any(cls, value: Any) -> "BinSpec":
        if isinstance(value, BinSpec):
            return value
        if isinstance(value, Mapping):
            low = value.get("low", -math.inf)
            high = value.get("high", math.inf)
            return cls(str(value["label"]), _parse_bound(low, -math.inf), _parse_bound(high, math.inf))
        label, low, high = value
        return cls(str(label), _parse_bound(low, -math.inf), _parse_bound(high, math.inf))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "low": None if self.low == -math.inf else self.low,
            "high": None if self.high == math.inf else self.high,
        }


DEFAULT_ALPHA_BINS = (
    BinSpec("before_start", -math.inf, 0.0),
    BinSpec("start_to_025", 0.0, 0.25),
    BinSpec("025_to_050", 0.25, 0.5),
    BinSpec("050_to_075", 0.5, 0.75),
    BinSpec("075_to_goal", 0.75, 1.0),
    BinSpec("after_goal", 1.0, math.inf),
)
DEFAULT_BETA_ABS_BINS = (
    BinSpec("centerline", 0.0, 0.1),
    BinSpec("near_side", 0.1, 0.25),
    BinSpec("far_side", 0.25, math.inf),
)
DEFAULT_BLOCKAGE_BINS = (
    BinSpec("low", 0.0, 0.33),
    BinSpec("medium", 0.33, 0.67),
    BinSpec("high", 0.67, math.inf),
)


@dataclass(frozen=True)
class AttributionConfig:
    alpha_bins: Tuple[BinSpec, ...] = DEFAULT_ALPHA_BINS
    beta_abs_bins: Tuple[BinSpec, ...] = DEFAULT_BETA_ABS_BINS
    blockage_bins: Tuple[BinSpec, ...] = DEFAULT_BLOCKAGE_BINS
    min_support: int = 10
    top_k: int = 10
    low_k: int = 5
    beta_center_eps: float = 1e-6
    include_failure_key_lifts: bool = True

    def __post_init__(self) -> None:
        if self.min_support < 1:
            raise ValueError("min_support must be >= 1")
        if self.top_k < 1:
            raise ValueError("top_k must be >= 1")
        if self.low_k < 0:
            raise ValueError("low_k must be >= 0")
        if self.beta_center_eps < 0:
            raise ValueError("beta_center_eps must be >= 0")
        _validate_bins(self.alpha_bins, "alpha_bins")
        _validate_bins(self.beta_abs_bins, "beta_abs_bins")
        _validate_bins(self.blockage_bins, "blockage_bins")

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, Any]]) -> "AttributionConfig":
        if not data:
            return cls()
        kwargs: Dict[str, Any] = {}
        for key in (
            "min_support",
            "top_k",
            "low_k",
            "beta_center_eps",
            "include_failure_key_lifts",
        ):
            if key in data:
                kwargs[key] = data[key]
        for key in ("alpha_bins", "beta_abs_bins", "blockage_bins"):
            if key in data and data[key] is not None:
                kwargs[key] = tuple(BinSpec.from_any(item) for item in data[key])
        return cls(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alpha_bins": [bin_spec.to_dict() for bin_spec in self.alpha_bins],
            "beta_abs_bins": [bin_spec.to_dict() for bin_spec in self.beta_abs_bins],
            "blockage_bins": [bin_spec.to_dict() for bin_spec in self.blockage_bins],
            "min_support": self.min_support,
            "top_k": self.top_k,
            "low_k": self.low_k,
            "beta_center_eps": self.beta_center_eps,
            "include_failure_key_lifts": self.include_failure_key_lifts,
        }


@dataclass
class CellAttribution:
    alpha_bin: str
    beta_abs_bin: str
    blockage_bin: str
    cell_id: str
    total_count: int
    success_count: int
    failure_count: int
    failure_rate: float
    p_cell: float
    p_cell_given_failure: float
    failure_lift: float
    dominant_failure_key: str
    dominant_failure_count: int
    collision_rate: float
    timeout_rate: float
    fall_rate: float
    mean_alpha: float
    mean_beta: float
    mean_blockage: float
    mean_d_start: float
    mean_d_goal: float
    beta_left_count: int
    beta_right_count: int
    beta_center_count: int
    termination_counts: Dict[str, int] = field(default_factory=dict)
    failure_key_counts: Dict[str, int] = field(default_factory=dict)
    failure_key_lifts: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        row = asdict(self)
        failure_key_lifts = row.pop("failure_key_lifts", {})
        for key, value in failure_key_lifts.items():
            row[f"lift__{key}"] = value
            row[f"count__{key}"] = self.failure_key_counts.get(key, 0)
        row["failure_key_lifts"] = failure_key_lifts
        return row


@dataclass
class CoverageInfo:
    num_obstacle_samples: int
    occupied_cells: int
    supported_cells: int
    min_support: int
    most_sampled_cells: List[CellAttribution]
    alpha_stats: Dict[str, float]
    beta_abs_stats: Dict[str, float]
    blockage_stats: Dict[str, float]
    bin_proportions: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_obstacle_samples": self.num_obstacle_samples,
            "occupied_cells": self.occupied_cells,
            "supported_cells": self.supported_cells,
            "min_support": self.min_support,
            "most_sampled_cells": [cell.to_dict() for cell in self.most_sampled_cells],
            "alpha_stats": dict(self.alpha_stats),
            "beta_abs_stats": dict(self.beta_abs_stats),
            "blockage_stats": dict(self.blockage_stats),
            "bin_proportions": dict(self.bin_proportions),
        }


@dataclass
class AttributionResult:
    metadata: Dict[str, Any]
    global_counts: Dict[str, Any]
    cells: List[CellAttribution]
    top_cells: List[CellAttribution]
    low_cells: List[CellAttribution]
    coverage: CoverageInfo

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metadata": dict(self.metadata),
            "global_counts": dict(self.global_counts),
            "cells": [cell.to_dict() for cell in self.cells],
            "top_cells": [cell.to_dict() for cell in self.top_cells],
            "low_cells": [cell.to_dict() for cell in self.low_cells],
            "coverage": self.coverage.to_dict(),
        }


def compute_attribution(
    records: Iterable[Mapping[str, Any]],
    config: Optional[AttributionConfig] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> AttributionResult:
    cfg = config or AttributionConfig()
    records_list = list(records)
    cells: Dict[Tuple[str, str, str], Dict[str, Any]] = defaultdict(_empty_cell_stats)
    global_termination_counts: Counter = Counter()
    global_failure_key_counts: Counter = Counter()
    alpha_values: List[float] = []
    beta_abs_values: List[float] = []
    blockage_values: List[float] = []
    total_samples = 0
    total_failure_samples = 0
    total_episodes = 0

    for record in records_list:
        total_episodes += 1
        termination = str(record.get("termination", "unknown"))
        failure_key = str(record.get("failure_key", termination))
        failed = is_failure(record)
        obstacles_z = record.get("obstacle_z") or []
        if not isinstance(obstacles_z, list):
            continue

        for z in obstacles_z:
            if not isinstance(z, Mapping):
                continue
            alpha = _safe_float(z.get("alpha", 0.0))
            beta = _safe_float(z.get("beta", 0.0))
            blockage = _safe_float(z.get("blockage", 0.0))
            key = cell_key(z, cfg)
            stats = cells[key]
            stats["total_count"] += 1
            stats["termination_counts"][termination] += 1
            stats["beta_side_counts"][beta_side(beta, cfg.beta_center_eps)] += 1
            stats["alpha_sum"] += alpha
            stats["beta_sum"] += beta
            stats["blockage_sum"] += blockage
            stats["d_start_sum"] += _safe_float(z.get("d_start", 0.0))
            stats["d_goal_sum"] += _safe_float(z.get("d_goal", 0.0))

            total_samples += 1
            global_termination_counts[termination] += 1
            alpha_values.append(alpha)
            beta_abs_values.append(abs(beta))
            blockage_values.append(blockage)
            if failed:
                stats["failure_count"] += 1
                stats["failure_key_counts"][failure_key] += 1
                global_failure_key_counts[failure_key] += 1
                total_failure_samples += 1
            else:
                stats["success_count"] += 1

    cell_rows: List[CellAttribution] = []
    for (alpha_bin, beta_abs_bin, blockage_bin), stats in cells.items():
        total_count = int(stats["total_count"])
        failure_count = int(stats["failure_count"])
        success_count = int(stats["success_count"])
        failure_rate = failure_count / total_count if total_count else 0.0
        p_cell = total_count / total_samples if total_samples else 0.0
        p_cell_given_failure = (
            failure_count / total_failure_samples if total_failure_samples else 0.0
        )
        failure_lift = p_cell_given_failure / p_cell if p_cell > 0 else 0.0
        dominant_failure_key, dominant_failure_count = most_common(stats["failure_key_counts"])
        failure_key_lifts: Dict[str, float] = {}
        if cfg.include_failure_key_lifts:
            for failure_key, global_count in global_failure_key_counts.items():
                key_count = int(stats["failure_key_counts"].get(failure_key, 0))
                p_cell_given_key = key_count / global_count if global_count else 0.0
                failure_key_lifts[str(failure_key)] = (
                    p_cell_given_key / p_cell if p_cell > 0 else 0.0
                )

        cell_rows.append(
            CellAttribution(
                alpha_bin=alpha_bin,
                beta_abs_bin=beta_abs_bin,
                blockage_bin=blockage_bin,
                cell_id=f"alpha={alpha_bin}|beta_abs={beta_abs_bin}|blockage={blockage_bin}",
                total_count=total_count,
                success_count=success_count,
                failure_count=failure_count,
                failure_rate=failure_rate,
                p_cell=p_cell,
                p_cell_given_failure=p_cell_given_failure,
                failure_lift=failure_lift,
                dominant_failure_key=dominant_failure_key,
                dominant_failure_count=dominant_failure_count,
                collision_rate=rate(stats["termination_counts"], "collision", total_count),
                timeout_rate=rate(stats["termination_counts"], "timeout", total_count),
                fall_rate=rate(stats["termination_counts"], "fall", total_count),
                mean_alpha=mean(stats, "alpha_sum"),
                mean_beta=mean(stats, "beta_sum"),
                mean_blockage=mean(stats, "blockage_sum"),
                mean_d_start=mean(stats, "d_start_sum"),
                mean_d_goal=mean(stats, "d_goal_sum"),
                beta_left_count=int(stats["beta_side_counts"].get("left", 0)),
                beta_right_count=int(stats["beta_side_counts"].get("right", 0)),
                beta_center_count=int(stats["beta_side_counts"].get("center", 0)),
                termination_counts=dict(stats["termination_counts"]),
                failure_key_counts=dict(stats["failure_key_counts"]),
                failure_key_lifts=failure_key_lifts,
            )
        )

    cell_rows.sort(
        key=lambda row: (
            row.total_count >= cfg.min_support,
            row.failure_lift,
            row.failure_count,
            row.total_count,
        ),
        reverse=True,
    )

    supported_rows = [row for row in cell_rows if row.total_count >= cfg.min_support]
    top_cells = supported_rows[: cfg.top_k]
    low_candidates = [row for row in supported_rows if row.failure_lift < 1.0]
    low_cells = sorted(
        low_candidates or supported_rows,
        key=lambda row: (row.failure_lift, row.failure_rate, -row.total_count),
    )[: cfg.low_k]
    most_sampled = sorted(cell_rows, key=lambda row: row.total_count, reverse=True)[: cfg.top_k]
    bin_proportions = {
        row.cell_id: (row.total_count / total_samples if total_samples else 0.0)
        for row in most_sampled[:3]
    }
    coverage = CoverageInfo(
        num_obstacle_samples=total_samples,
        occupied_cells=len([row for row in cell_rows if row.total_count > 0]),
        supported_cells=len(supported_rows),
        min_support=cfg.min_support,
        most_sampled_cells=most_sampled,
        alpha_stats=distribution_stats(alpha_values),
        beta_abs_stats=distribution_stats(beta_abs_values),
        blockage_stats=distribution_stats(blockage_values),
        bin_proportions=bin_proportions,
    )
    global_counts = {
        "num_episodes": total_episodes,
        "num_obstacle_samples": total_samples,
        "num_failure_samples": total_failure_samples,
        "termination_counts": dict(global_termination_counts),
        "failure_key_counts": dict(global_failure_key_counts),
        "failure_rate": total_failure_samples / total_samples if total_samples else 0.0,
        "min_support": int(cfg.min_support),
        "has_any_failure": total_failure_samples > 0,
        "is_empty": total_samples == 0,
    }
    result_metadata = {
        "alpha_bins": [bin_spec.to_dict() for bin_spec in cfg.alpha_bins],
        "beta_abs_bins": [bin_spec.to_dict() for bin_spec in cfg.beta_abs_bins],
        "blockage_bins": [bin_spec.to_dict() for bin_spec in cfg.blockage_bins],
        "cell_definition": "alpha_bin x abs(beta)_bin x blockage_bin",
        "config": cfg.to_dict(),
    }
    if metadata:
        result_metadata.update(dict(metadata))
    return AttributionResult(
        metadata=result_metadata,
        global_counts=global_counts,
        cells=cell_rows,
        top_cells=top_cells,
        low_cells=low_cells,
        coverage=coverage,
    )


def build_attribution(
    records: Iterable[Mapping[str, Any]],
    min_support: int = 10,
) -> Dict[str, Any]:
    config = AttributionConfig(min_support=min_support)
    return compute_attribution(records, config).to_dict()


def summarize_attribution_result(
    result: AttributionResult,
    success_rate_change: Optional[float] = None,
) -> Dict[str, Any]:
    top_cell = result.top_cells[0].to_dict() if result.top_cells else None
    metadata = dict(result.metadata or {})
    summary = {
        "evolve_index": metadata.get("evolve_index"),
        "episode_idx": metadata.get("total_episode_count"),
        "batch_episode_count": metadata.get("batch_episode_count"),
        "success_rate_at_evolve": metadata.get("success_rate"),
        "trigger_reason": metadata.get("trigger_reason"),
        "failure_rate": result.global_counts.get("failure_rate", 0.0),
        "num_high_cells": len([cell for cell in result.cells if cell.failure_lift > 1.0]),
        "num_obstacle_samples": result.global_counts.get("num_obstacle_samples", 0),
        "dominant_cell": top_cell,
    }
    summary = {key: value for key, value in summary.items() if value is not None}
    if success_rate_change is not None:
        summary["success_rate_change"] = success_rate_change
    return summary


def append_attribution_history(
    history: List[Dict[str, Any]],
    result: AttributionResult,
    max_len: int = 3,
    success_rate_change: Optional[float] = None,
) -> List[Dict[str, Any]]:
    history.append(summarize_attribution_result(result, success_rate_change))
    if max_len > 0 and len(history) > max_len:
        del history[:-max_len]
    return history


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return records


def write_attribution_outputs(result: AttributionResult | Mapping[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    result_dict = result.to_dict() if isinstance(result, AttributionResult) else dict(result)
    (output_dir / "attribution_map.json").write_text(
        json.dumps(result_dict, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    write_cells_csv(result_dict.get("cells", []), output_dir / "obstacle_z_bins.csv")
    (output_dir / "attribution_summary.txt").write_text(
        build_summary_text(result_dict),
        encoding="utf-8",
    )


def write_cells_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    base_fields = [
        "cell_id",
        "alpha_bin",
        "beta_abs_bin",
        "blockage_bin",
        "total_count",
        "success_count",
        "failure_count",
        "failure_rate",
        "p_cell",
        "p_cell_given_failure",
        "failure_lift",
        "dominant_failure_key",
        "dominant_failure_count",
        "collision_rate",
        "timeout_rate",
        "fall_rate",
        "mean_alpha",
        "mean_beta",
        "mean_blockage",
        "mean_d_start",
        "mean_d_goal",
        "beta_left_count",
        "beta_right_count",
        "beta_center_count",
    ]
    dynamic_fields = sorted(
        {
            key
            for row in rows
            for key in row.keys()
            if key.startswith("lift__") or key.startswith("count__")
        }
    )
    fields = base_fields + dynamic_fields
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def build_summary_text(result: Mapping[str, Any], top_k: int = 8) -> str:
    counts = result["global_counts"]
    lines = [
        "Obstacle Failure Attribution Summary",
        "====================================",
        f"episodes: {counts['num_episodes']}",
        f"obstacle_samples: {counts['num_obstacle_samples']}",
        f"failure_samples: {counts['num_failure_samples']}",
        f"overall_failure_rate: {counts['failure_rate']:.3f}",
        f"min_support: {counts['min_support']}",
        "",
        "Termination counts:",
    ]
    for key in TERMINATION_KEYS:
        lines.append(f"- {key}: {counts['termination_counts'].get(key, 0)}")
    extra_terms = sorted(set(counts["termination_counts"]) - set(TERMINATION_KEYS))
    for key in extra_terms:
        lines.append(f"- {key}: {counts['termination_counts'].get(key, 0)}")

    lines.extend(["", "Top high-lift cells:"])
    top_cells = result.get("top_cells", [])[:top_k]
    if not top_cells:
        lines.append("- none with the requested support")
    for idx, row in enumerate(top_cells, start=1):
        lines.append(
            f"{idx}. {row['cell_id']} | "
            f"support={row['total_count']}, failures={row['failure_count']}, "
            f"failure_rate={row['failure_rate']:.3f}, lift={row['failure_lift']:.3f}, "
            f"dominant={row['dominant_failure_key']} ({row['dominant_failure_count']})"
        )
        lines.append(
            f"   mean alpha={row['mean_alpha']:.3f}, "
            f"beta={row['mean_beta']:.3f}, blockage={row['mean_blockage']:.3f}; "
            f"collision={row['collision_rate']:.3f}, "
            f"timeout={row['timeout_rate']:.3f}, fall={row['fall_rate']:.3f}"
        )
    return "\n".join(lines) + "\n"


def bin_value(value: float, bins: Sequence[BinSpec]) -> str:
    value = float(value)
    for idx, bin_spec in enumerate(bins):
        is_last = idx == len(bins) - 1
        if bin_spec.low <= value < bin_spec.high or (is_last and value == bin_spec.high):
            return bin_spec.label
    return bins[-1].label


def beta_side(beta: float, eps: float = 1e-6) -> str:
    beta = float(beta)
    if beta > eps:
        return "left"
    if beta < -eps:
        return "right"
    return "center"


def is_failure(record: Mapping[str, Any]) -> bool:
    return str(record.get("termination", "unknown")) != "success"


def cell_key(z: Mapping[str, Any], config: Optional[AttributionConfig] = None) -> Tuple[str, str, str]:
    cfg = config or AttributionConfig()
    alpha = _safe_float(z.get("alpha", 0.0))
    beta = _safe_float(z.get("beta", 0.0))
    blockage = _safe_float(z.get("blockage", 0.0))
    return (
        bin_value(alpha, cfg.alpha_bins),
        bin_value(abs(beta), cfg.beta_abs_bins),
        bin_value(blockage, cfg.blockage_bins),
    )


def most_common(counter: Counter) -> Tuple[str, int]:
    if not counter:
        return "none", 0
    key, count = counter.most_common(1)[0]
    return str(key), int(count)


def rate(counter: Counter, key: str, total: int) -> float:
    return float(counter.get(key, 0) / total) if total else 0.0


def mean(stats: Mapping[str, Any], sum_key: str) -> float:
    total = int(stats["total_count"])
    return float(stats[sum_key] / total) if total else 0.0


def distribution_stats(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    n = len(values)
    mean_value = sum(values) / n
    var = sum((value - mean_value) ** 2 for value in values) / n
    return {
        "mean": float(mean_value),
        "std": float(math.sqrt(var)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _empty_cell_stats() -> Dict[str, Any]:
    return {
        "total_count": 0,
        "success_count": 0,
        "failure_count": 0,
        "termination_counts": Counter(),
        "failure_key_counts": Counter(),
        "beta_side_counts": Counter(),
        "alpha_sum": 0.0,
        "beta_sum": 0.0,
        "blockage_sum": 0.0,
        "d_start_sum": 0.0,
        "d_goal_sum": 0.0,
    }


def _parse_bound(value: Any, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, str):
        norm = value.strip().lower()
        if norm in ("inf", "+inf", "infinity", "+infinity"):
            return math.inf
        if norm in ("-inf", "-infinity"):
            return -math.inf
    return float(value)


def _validate_bins(bins: Sequence[BinSpec], name: str) -> None:
    if not bins:
        raise ValueError(f"{name} must not be empty")
    for bin_spec in bins:
        if bin_spec.low >= bin_spec.high:
            raise ValueError(f"{name} has invalid bin: {bin_spec}")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
