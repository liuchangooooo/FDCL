"""Utilities for parsing failure-driven curriculum training logs.

This module builds a unified parsing layer for the Push-T failure-driven
obstacle-generation experiments. It extracts structured tables from:

- ``train.log``
- ``evolve_prompt.log``
- ``wandb-summary.json``

It can also export the parsed result into ``csv/json/jsonl`` files for later
visualization scripts.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]

LOG_PREFIX = r"(?:\[[^\]]+\]\[[^\]]+\]\[(?:INFO|WARNING|ERROR)\] - )?"
NUM_PATTERN = r"[-+]?\d+(?:\.\d+)?"


LLM_EPISODE_RE = re.compile(
    r"^"
    + LOG_PREFIX
    + r"\[LLM\] Episode (?P<episode>\d+): Generated (?P<num_obstacles>\d+) obstacles$"
)
T_BLOCK_RE = re.compile(
    r"^\s*T-block:\s*\((?P<x>"
    + NUM_PATTERN
    + r"),\s*(?P<y>"
    + NUM_PATTERN
    + r"),\s*(?P<theta>"
    + NUM_PATTERN
    + r")\s*(?:°|deg)\s*\)$"
)
OBSTACLE_RE = re.compile(
    r"^\s*Obs(?P<idx>\d+):\s*\((?P<x>"
    + NUM_PATTERN
    + r"),\s*(?P<y>"
    + NUM_PATTERN
    + r")\)(?:\s*-\s*(?P<purpose>.+))?$"
)
BATCH_COMPLETE_RE = re.compile(
    r"^\[ACGS\] Batch complete \((?P<batch_episodes>\d+) episodes, total (?P<total_episodes>\d+)\)$"
)
STATS_RE = re.compile(
    r"^"
    + LOG_PREFIX
    + r"\[ACGS\] Stats: \{'success': (?P<success>\d+), 'collision': (?P<collision>\d+), "
    r"'timeout': (?P<timeout>\d+), 'fall': (?P<fall>\d+)\}$"
)
SUCCESS_RATE_RE = re.compile(
    r"^"
    + LOG_PREFIX
    + r"\[ACGS\] Success rate: (?P<success_rate>\d+\.\d+), Collision rate: (?P<collision_rate>\d+\.\d+)$"
)
REPLAY_COUNT_RE = re.compile(
    r"^"
    + LOG_PREFIX
    + r"\[ACGS\] Failure replays: (?P<failure>\d+), Success replays: (?P<success>\d+)$"
)
DIAG_RE = re.compile(
    r"^"
    + LOG_PREFIX
    + r"\[ACGS\]\[Diag\] dominant=(?P<dominant>[^,]+), samples=(?P<samples>\d+), "
    r"motion_peak_confidence=(?P<motion>[0-9.]+), start_peak_confidence=(?P<start>[0-9.]+), "
    r"chosen_source=(?P<source>.+)$"
)
TRIGGER_RE = re.compile(
    r"^" + LOG_PREFIX + r"\[ACGS\] >>> Triggering evolve: (?P<reason>.+)$"
)
SKIP_COOLDOWN_RE = re.compile(
    r"^"
    + LOG_PREFIX
    + r"\[ACGS\] Skipping: cooldown \((?P<current>\d+)\/(?P<required>\d+) episodes since last evolve\)$"
)
NO_EVOLVE_RE = re.compile(
    r"^" + LOG_PREFIX + r"\[ACGS\] No evolve needed: (?P<message>.+)$"
)
PROMPT_LENGTH_RE = re.compile(
    r"^" + LOG_PREFIX + r"\[ACGS\] Prompt length: (?P<length>\d+) chars$"
)
EVOLVE_SUCCESS_RE = re.compile(
    r"^"
    + LOG_PREFIX
    + r"\[ACGS\] Evolve #(?P<evolve_id>\d+) succeeded, new generator loaded \(cooldown (?P<cooldown>\d+) eps\)$"
)

EVOLVE_BLOCK_HEADER_RE = re.compile(r"^\[EVOLVE PROMPT\] time=(?P<time>.+)$")
EVOLVE_EPISODE_HEADER_RE = re.compile(
    r"^episode_total=(?P<episode_total>\d+), batch_episodes=(?P<batch_episodes>\d+), evolve_count=(?P<evolve_count>\d+)$"
)
EVOLVE_REASON_RE = re.compile(r"^reason=(?P<reason>.+)$")
EVOLVE_DIAG_HEADER_RE = re.compile(
    r"^dominant_failure_type=(?P<dominant>[^,]+), diagnosis_reliability=(?P<reliability>.+)$"
)
SECTION_HEADER_RE = re.compile(r"^(?P<section>[A-Z])\.\s+.+$")
TOTAL_EPISODES_RE = re.compile(r"- total_episodes:\s*(?P<value>\d+)")
COUNT_LINE_RE = re.compile(r"- (?P<key>success|collision|timeout|fall):\s*(?P<count>\d+)")
FAILURE_REGION_RE = re.compile(r"- failure_region:\s*(?P<region>.+?)\s+\(confidence=(?P<confidence>[0-9.]+)\)")
BEHAVIOR_BIAS_RE = re.compile(r"- behavior_bias:\s*(?P<bias>.+?)\s+\(confidence=(?P<confidence>[0-9.]+),")
SAMPLE_COUNT_RE = re.compile(r"- sample_count:\s*(?P<count>\d+)")
RELIABILITY_IN_SECTION_RE = re.compile(r"- reliability:\s*(?P<value>.+)$")

CONFIG_PATTERNS = {
    "configured_obstacle_num": re.compile(r"^obstacle_num:\s*(?P<value>\d+)\s*$"),
    "enable_acgs_loop": re.compile(r"^\s*enable_acgs_loop:\s*(?P<value>true|false)\s*$"),
    "configured_evaluation_interval": re.compile(r"^\s*evaluation_interval:\s*(?P<value>\d+)\s*$"),
    "configured_success_rate_high": re.compile(r"^\s*success_rate_high:\s*(?P<value>" + NUM_PATTERN + r")\s*$"),
    "configured_success_rate_low": re.compile(r"^\s*success_rate_low:\s*(?P<value>" + NUM_PATTERN + r")\s*$"),
    "configured_plateau_window": re.compile(r"^\s*plateau_window:\s*(?P<value>\d+)\s*$"),
    "configured_plateau_threshold": re.compile(r"^\s*plateau_threshold:\s*(?P<value>" + NUM_PATTERN + r")\s*$"),
}

VIDEO_RE = re.compile(r"^sim_video_(?P<slot>\d+)_(?P<step>\d+)_.*\.mp4$")


@dataclass
class ParseWarning:
    run_id: str
    category: str
    message: str
    line_no: Optional[int] = None
    episode: Optional[int] = None
    batch_idx: Optional[int] = None
    snapshot_id: Optional[int] = None


@dataclass
class SnapshotRecord:
    run_id: str
    snapshot_id: int
    episode: int
    num_obstacles: int
    generator_id: int
    source_log_line: int
    tblock_x: Optional[float] = None
    tblock_y: Optional[float] = None
    tblock_theta_deg: Optional[float] = None
    batch_idx: Optional[int] = None
    stage_id: Optional[int] = None
    stage_label: Optional[str] = None
    parse_complete: bool = True
    collected_obstacle_count: int = 0


@dataclass
class ObstacleRecord:
    run_id: str
    snapshot_id: int
    episode: int
    generator_id: int
    obstacle_idx: int
    obs_x: float
    obs_y: float
    purpose: str = ""
    purpose_family: Optional[str] = None
    batch_idx: Optional[int] = None
    stage_id: Optional[int] = None
    stage_label: Optional[str] = None


@dataclass
class BatchRecord:
    run_id: str
    batch_idx: int
    batch_start_episode: int
    batch_end_episode: int
    batch_episodes: int
    generator_id_before_batch: int
    generator_id_after_batch: Optional[int] = None
    success_count: Optional[int] = None
    collision_count: Optional[int] = None
    timeout_count: Optional[int] = None
    fall_count: Optional[int] = None
    success_rate: Optional[float] = None
    collision_rate: Optional[float] = None
    timeout_rate: Optional[float] = None
    fall_rate: Optional[float] = None
    success_rate_logged: Optional[float] = None
    collision_rate_logged: Optional[float] = None
    failure_replays: Optional[int] = None
    success_replays: Optional[int] = None
    dominant_failure_type: Optional[str] = None
    diag_samples: Optional[int] = None
    motion_peak_confidence: Optional[float] = None
    start_peak_confidence: Optional[float] = None
    chosen_source: Optional[str] = None
    evolve_decision: str = "unknown"
    evolve_trigger_reason: Optional[str] = None
    evolve_decision_message: Optional[str] = None
    cooldown_progress_current: Optional[int] = None
    cooldown_progress_required: Optional[int] = None
    prompt_length_chars: Optional[int] = None
    new_generator_loaded: bool = False
    evolve_id_loaded: Optional[int] = None
    cooldown_episodes: Optional[int] = None
    linked_evolve_round_index: Optional[int] = None
    diagnosis_reliability: Optional[str] = None


@dataclass
class EvolveRoundRecord:
    run_id: str
    evolve_round_index: int
    time: str
    episode_total: int
    batch_episodes: int
    evolve_count_before: int
    trigger_reason: str
    dominant_failure_type: str
    diagnosis_reliability: str
    evolve_id: Optional[int] = None
    prompt_length_chars: Optional[int] = None
    generator_loaded_successfully: Optional[bool] = None
    linked_batch_idx: Optional[int] = None
    coarse_total_episodes: Optional[int] = None
    coarse_success_count: Optional[int] = None
    coarse_collision_count: Optional[int] = None
    coarse_timeout_count: Optional[int] = None
    coarse_fall_count: Optional[int] = None
    diagnosis_reliability_inner: Optional[str] = None
    failure_region: Optional[str] = None
    failure_region_confidence: Optional[float] = None
    behavior_bias: Optional[str] = None
    behavior_bias_confidence: Optional[float] = None
    sample_count: Optional[int] = None
    current_generator_section_count: int = 0
    section_counts_json: str = ""
    section_B_raw: str = ""
    section_C_raw: str = ""
    section_D_raw: str = ""
    section_E_raw: str = ""
    section_F_raw: str = ""
    section_G_raw: str = ""
    section_H_raw: str = ""
    section_I_raw: str = ""
    prompt_text_raw: str = ""


@dataclass
class ParsedRun:
    run_meta: Dict[str, Any]
    layout_snapshots: List[SnapshotRecord]
    obstacle_points: List[ObstacleRecord]
    batch_stats: List[BatchRecord]
    evolve_rounds: List[EvolveRoundRecord]
    media_index: List[Dict[str, Any]]
    parse_warnings: List[ParseWarning] = field(default_factory=list)

    def export(self, export_dir: Optional[Path] = None) -> Path:
        export_path = Path(export_dir) if export_dir is not None else Path(self.run_meta["output_dir"]) / "parsed"
        export_path.mkdir(parents=True, exist_ok=True)

        _write_json(export_path / "run_meta.json", self.run_meta)
        _write_csv(export_path / "layout_snapshots.csv", SNAPSHOT_FIELDNAMES, [_snapshot_to_row(r) for r in self.layout_snapshots])
        _write_csv(export_path / "obstacle_points.csv", OBSTACLE_FIELDNAMES, [_obstacle_to_row(r) for r in self.obstacle_points])
        _write_csv(export_path / "batch_stats.csv", BATCH_FIELDNAMES, [_batch_to_row(r) for r in self.batch_stats])
        _write_csv(export_path / "evolve_rounds.csv", EVOLVE_FIELDNAMES, [_evolve_to_row(r) for r in self.evolve_rounds])
        _write_jsonl(export_path / "evolve_rounds_raw.jsonl", [asdict(r) for r in self.evolve_rounds])
        _write_csv(export_path / "media_index.csv", MEDIA_FIELDNAMES, self.media_index)
        _write_csv(export_path / "parse_warnings.csv", WARNING_FIELDNAMES, [asdict(r) for r in self.parse_warnings])

        return export_path


SNAPSHOT_FIELDNAMES = [
    "run_id",
    "snapshot_id",
    "episode",
    "num_obstacles",
    "generator_id",
    "batch_idx",
    "stage_id",
    "stage_label",
    "tblock_x",
    "tblock_y",
    "tblock_theta_deg",
    "source_log_line",
    "parse_complete",
    "collected_obstacle_count",
]

OBSTACLE_FIELDNAMES = [
    "run_id",
    "snapshot_id",
    "episode",
    "generator_id",
    "obstacle_idx",
    "obs_x",
    "obs_y",
    "purpose",
    "purpose_family",
    "batch_idx",
    "stage_id",
    "stage_label",
]

BATCH_FIELDNAMES = [
    "run_id",
    "batch_idx",
    "batch_start_episode",
    "batch_end_episode",
    "batch_episodes",
    "generator_id_before_batch",
    "generator_id_after_batch",
    "success_count",
    "collision_count",
    "timeout_count",
    "fall_count",
    "success_rate",
    "collision_rate",
    "timeout_rate",
    "fall_rate",
    "success_rate_logged",
    "collision_rate_logged",
    "failure_replays",
    "success_replays",
    "dominant_failure_type",
    "diag_samples",
    "motion_peak_confidence",
    "start_peak_confidence",
    "chosen_source",
    "evolve_decision",
    "evolve_trigger_reason",
    "evolve_decision_message",
    "cooldown_progress_current",
    "cooldown_progress_required",
    "prompt_length_chars",
    "new_generator_loaded",
    "evolve_id_loaded",
    "cooldown_episodes",
    "linked_evolve_round_index",
    "diagnosis_reliability",
]

EVOLVE_FIELDNAMES = [
    "run_id",
    "evolve_round_index",
    "evolve_id",
    "linked_batch_idx",
    "time",
    "episode_total",
    "batch_episodes",
    "evolve_count_before",
    "trigger_reason",
    "dominant_failure_type",
    "diagnosis_reliability",
    "prompt_length_chars",
    "generator_loaded_successfully",
    "coarse_total_episodes",
    "coarse_success_count",
    "coarse_collision_count",
    "coarse_timeout_count",
    "coarse_fall_count",
    "diagnosis_reliability_inner",
    "failure_region",
    "failure_region_confidence",
    "behavior_bias",
    "behavior_bias_confidence",
    "sample_count",
    "current_generator_section_count",
    "section_counts_json",
]

MEDIA_FIELDNAMES = [
    "run_id",
    "update_step",
    "video_slot",
    "video_path",
    "file_name",
]

WARNING_FIELDNAMES = [
    "run_id",
    "category",
    "message",
    "line_no",
    "episode",
    "batch_idx",
    "snapshot_id",
]


def parse_run_directory(run_dir: str, export_dir: Optional[str] = None) -> ParsedRun:
    run_path = Path(run_dir).expanduser().resolve()
    if not run_path.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_path}")

    run_id = _infer_run_id(run_path)
    train_log_path = run_path / "train.log"
    if not train_log_path.exists():
        raise FileNotFoundError(f"Missing train.log: {train_log_path}")

    evolve_prompt_log_path = run_path / "evolve_prompt.log"
    hydra_config_path = run_path / ".hydra" / "config.yaml"
    wandb_summary_path = _find_wandb_summary_path(run_path)

    train_tables = _parse_train_log(train_log_path, run_id)
    evolve_rounds = _parse_evolve_prompt_log(evolve_prompt_log_path, run_id) if evolve_prompt_log_path.exists() else []
    summary_meta = _parse_wandb_summary(wandb_summary_path) if wandb_summary_path is not None else {}
    config_meta = _parse_hydra_config(hydra_config_path) if hydra_config_path.exists() else {}
    media_index = _index_eval_videos(run_path, run_id)

    parsed_run = _merge_and_validate(
        run_id=run_id,
        run_path=run_path,
        hydra_config_path=hydra_config_path if hydra_config_path.exists() else None,
        train_log_path=train_log_path,
        evolve_prompt_log_path=evolve_prompt_log_path if evolve_prompt_log_path.exists() else None,
        wandb_summary_path=wandb_summary_path,
        config_meta=config_meta,
        summary_meta=summary_meta,
        train_tables=train_tables,
        evolve_rounds=evolve_rounds,
        media_index=media_index,
    )

    if export_dir is not None:
        parsed_run.export(Path(export_dir))

    return parsed_run


def _parse_train_log(train_log_path: Path, run_id: str) -> Dict[str, Any]:
    snapshots: List[SnapshotRecord] = []
    obstacles: List[ObstacleRecord] = []
    batches: List[BatchRecord] = []
    warnings: List[ParseWarning] = []

    current_snapshot: Optional[SnapshotRecord] = None
    current_batch: Optional[BatchRecord] = None
    current_generator_id = 0
    snapshot_id = -1
    batch_idx = -1

    def finalize_snapshot() -> None:
        nonlocal current_snapshot
        if current_snapshot is None:
            return

        if current_snapshot.collected_obstacle_count != current_snapshot.num_obstacles:
            current_snapshot.parse_complete = False
            warnings.append(
                ParseWarning(
                    run_id=run_id,
                    category="incomplete_snapshot",
                    message=(
                        f"Snapshot episode {current_snapshot.episode} collected "
                        f"{current_snapshot.collected_obstacle_count}/{current_snapshot.num_obstacles} obstacles"
                    ),
                    line_no=current_snapshot.source_log_line,
                    episode=current_snapshot.episode,
                    snapshot_id=current_snapshot.snapshot_id,
                )
            )

        snapshots.append(current_snapshot)
        current_snapshot = None

    def finalize_batch() -> None:
        nonlocal current_batch
        if current_batch is None:
            return

        if current_batch.generator_id_after_batch is None:
            current_batch.generator_id_after_batch = current_batch.generator_id_before_batch

        if current_batch.success_count is not None:
            total = (
                current_batch.success_count
                + current_batch.collision_count
                + current_batch.timeout_count
                + current_batch.fall_count
            )
            if total > 0:
                current_batch.success_rate = current_batch.success_count / total
                current_batch.collision_rate = current_batch.collision_count / total
                current_batch.timeout_rate = current_batch.timeout_count / total
                current_batch.fall_rate = current_batch.fall_count / total

        batches.append(current_batch)
        current_batch = None

    with train_log_path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\n")

            match = LLM_EPISODE_RE.match(line)
            if match:
                finalize_snapshot()
                snapshot_id += 1
                current_snapshot = SnapshotRecord(
                    run_id=run_id,
                    snapshot_id=snapshot_id,
                    episode=int(match.group("episode")),
                    num_obstacles=int(match.group("num_obstacles")),
                    generator_id=current_generator_id,
                    source_log_line=line_no,
                )
                continue

            match = T_BLOCK_RE.match(line)
            if match:
                if current_snapshot is None:
                    warnings.append(
                        ParseWarning(
                            run_id=run_id,
                            category="orphan_tblock",
                            message="Encountered T-block line without an active snapshot",
                            line_no=line_no,
                        )
                    )
                    continue
                current_snapshot.tblock_x = float(match.group("x"))
                current_snapshot.tblock_y = float(match.group("y"))
                current_snapshot.tblock_theta_deg = float(match.group("theta"))
                continue

            match = OBSTACLE_RE.match(line)
            if match:
                if current_snapshot is None:
                    warnings.append(
                        ParseWarning(
                            run_id=run_id,
                            category="orphan_obstacle",
                            message="Encountered obstacle line without an active snapshot",
                            line_no=line_no,
                        )
                    )
                    continue

                obstacle = ObstacleRecord(
                    run_id=run_id,
                    snapshot_id=current_snapshot.snapshot_id,
                    episode=current_snapshot.episode,
                    generator_id=current_snapshot.generator_id,
                    obstacle_idx=int(match.group("idx")),
                    obs_x=float(match.group("x")),
                    obs_y=float(match.group("y")),
                    purpose=(match.group("purpose") or "").strip(),
                )
                obstacles.append(obstacle)
                current_snapshot.collected_obstacle_count += 1
                if current_snapshot.collected_obstacle_count >= current_snapshot.num_obstacles:
                    finalize_snapshot()
                continue

            match = BATCH_COMPLETE_RE.match(line)
            if match:
                finalize_batch()
                batch_idx += 1
                batch_episodes = int(match.group("batch_episodes"))
                batch_end_episode = int(match.group("total_episodes"))
                current_batch = BatchRecord(
                    run_id=run_id,
                    batch_idx=batch_idx,
                    batch_start_episode=batch_end_episode - batch_episodes,
                    batch_end_episode=batch_end_episode,
                    batch_episodes=batch_episodes,
                    generator_id_before_batch=current_generator_id,
                )
                continue

            match = STATS_RE.match(line)
            if match:
                if current_batch is None:
                    warnings.append(
                        ParseWarning(
                            run_id=run_id,
                            category="orphan_batch_stats",
                            message="Encountered batch stats without an active batch record",
                            line_no=line_no,
                        )
                    )
                    continue
                current_batch.success_count = int(match.group("success"))
                current_batch.collision_count = int(match.group("collision"))
                current_batch.timeout_count = int(match.group("timeout"))
                current_batch.fall_count = int(match.group("fall"))
                continue

            match = SUCCESS_RATE_RE.match(line)
            if match:
                if current_batch is not None:
                    current_batch.success_rate_logged = float(match.group("success_rate"))
                    current_batch.collision_rate_logged = float(match.group("collision_rate"))
                continue

            match = REPLAY_COUNT_RE.match(line)
            if match:
                if current_batch is not None:
                    current_batch.failure_replays = int(match.group("failure"))
                    current_batch.success_replays = int(match.group("success"))
                continue

            match = DIAG_RE.match(line)
            if match:
                if current_batch is not None:
                    current_batch.dominant_failure_type = match.group("dominant").strip()
                    current_batch.diag_samples = int(match.group("samples"))
                    current_batch.motion_peak_confidence = float(match.group("motion"))
                    current_batch.start_peak_confidence = float(match.group("start"))
                    current_batch.chosen_source = match.group("source").strip()
                continue

            match = TRIGGER_RE.match(line)
            if match:
                if current_batch is not None:
                    current_batch.evolve_decision = "triggered"
                    current_batch.evolve_trigger_reason = match.group("reason").strip()
                continue

            match = SKIP_COOLDOWN_RE.match(line)
            if match:
                if current_batch is not None:
                    current_batch.evolve_decision = "skipped_cooldown"
                    current_batch.cooldown_progress_current = int(match.group("current"))
                    current_batch.cooldown_progress_required = int(match.group("required"))
                continue

            match = NO_EVOLVE_RE.match(line)
            if match:
                if current_batch is not None:
                    current_batch.evolve_decision = "skipped_normal"
                    current_batch.evolve_decision_message = match.group("message").strip()
                continue

            match = PROMPT_LENGTH_RE.match(line)
            if match:
                if current_batch is not None:
                    current_batch.prompt_length_chars = int(match.group("length"))
                continue

            match = EVOLVE_SUCCESS_RE.match(line)
            if match:
                if current_batch is None:
                    warnings.append(
                        ParseWarning(
                            run_id=run_id,
                            category="orphan_evolve_success",
                            message="Encountered evolve success without an active batch record",
                            line_no=line_no,
                        )
                    )
                    continue
                evolve_id = int(match.group("evolve_id"))
                current_batch.new_generator_loaded = True
                current_batch.evolve_id_loaded = evolve_id
                current_batch.generator_id_after_batch = evolve_id
                current_batch.cooldown_episodes = int(match.group("cooldown"))
                current_generator_id = evolve_id
                continue

    finalize_snapshot()
    finalize_batch()

    return {
        "layout_snapshots": snapshots,
        "obstacle_points": obstacles,
        "batch_stats": batches,
        "warnings": warnings,
    }


def _parse_evolve_prompt_log(evolve_prompt_log_path: Path, run_id: str) -> List[EvolveRoundRecord]:
    with evolve_prompt_log_path.open("r", encoding="utf-8") as handle:
        lines = [line.rstrip("\n") for line in handle]

    blocks = _split_evolve_blocks(lines)
    rows: List[EvolveRoundRecord] = []

    for block_index, block_lines in enumerate(blocks):
        if not block_lines:
            continue

        row = _parse_single_evolve_block(run_id, block_index, block_lines)
        if row is not None:
            rows.append(row)

    return rows


def _split_evolve_blocks(lines: List[str]) -> List[List[str]]:
    blocks: List[List[str]] = []
    current: List[str] = []

    for line in lines:
        if EVOLVE_BLOCK_HEADER_RE.match(line):
            if current:
                blocks.append(current)
            current = [line]
        elif current:
            current.append(line)

    if current:
        blocks.append(current)

    return blocks


def _parse_single_evolve_block(run_id: str, block_index: int, block_lines: List[str]) -> Optional[EvolveRoundRecord]:
    if len(block_lines) < 4:
        return None

    header_time = EVOLVE_BLOCK_HEADER_RE.match(block_lines[0])
    header_episode = EVOLVE_EPISODE_HEADER_RE.match(block_lines[1])
    header_reason = EVOLVE_REASON_RE.match(block_lines[2])
    header_diag = EVOLVE_DIAG_HEADER_RE.match(block_lines[3])

    if not (header_time and header_episode and header_reason and header_diag):
        return None

    sections = _split_sections(block_lines[4:])
    section_counts = {name: len(chunks) for name, chunks in sections.items()}

    row = EvolveRoundRecord(
        run_id=run_id,
        evolve_round_index=block_index,
        time=header_time.group("time").strip(),
        episode_total=int(header_episode.group("episode_total")),
        batch_episodes=int(header_episode.group("batch_episodes")),
        evolve_count_before=int(header_episode.group("evolve_count")),
        trigger_reason=header_reason.group("reason").strip(),
        dominant_failure_type=header_diag.group("dominant").strip(),
        diagnosis_reliability=header_diag.group("reliability").strip(),
        current_generator_section_count=section_counts.get("I", 0),
        section_counts_json=json.dumps(section_counts, ensure_ascii=False, sort_keys=True),
        section_B_raw="\n\n".join(sections.get("B", [])),
        section_C_raw="\n\n".join(sections.get("C", [])),
        section_D_raw="\n\n".join(sections.get("D", [])),
        section_E_raw="\n\n".join(sections.get("E", [])),
        section_F_raw="\n\n".join(sections.get("F", [])),
        section_G_raw="\n\n".join(sections.get("G", [])),
        section_H_raw="\n\n".join(sections.get("H", [])),
        section_I_raw="\n\n".join(sections.get("I", [])),
        prompt_text_raw="\n".join(block_lines),
    )

    _fill_evolve_structured_fields(row)
    return row


def _split_sections(block_lines: List[str]) -> Dict[str, List[str]]:
    sections: Dict[str, List[str]] = defaultdict(list)
    current_name: Optional[str] = None
    current_lines: List[str] = []

    for line in block_lines:
        section_match = SECTION_HEADER_RE.match(line)
        if section_match:
            if current_name is not None:
                sections[current_name].append("\n".join(current_lines).strip())
            current_name = section_match.group("section")
            current_lines = [line]
        elif current_name is not None:
            current_lines.append(line)

    if current_name is not None:
        sections[current_name].append("\n".join(current_lines).strip())

    return sections


def _fill_evolve_structured_fields(row: EvolveRoundRecord) -> None:
    if row.section_B_raw:
        total_match = TOTAL_EPISODES_RE.search(row.section_B_raw)
        if total_match:
            row.coarse_total_episodes = int(total_match.group("value"))

        counts: Dict[str, int] = {}
        for match in COUNT_LINE_RE.finditer(row.section_B_raw):
            counts[match.group("key")] = int(match.group("count"))
        row.coarse_success_count = counts.get("success")
        row.coarse_collision_count = counts.get("collision")
        row.coarse_timeout_count = counts.get("timeout")
        row.coarse_fall_count = counts.get("fall")

    if row.section_D_raw:
        reliability_match = RELIABILITY_IN_SECTION_RE.search(row.section_D_raw)
        if reliability_match:
            row.diagnosis_reliability_inner = reliability_match.group("value").strip()

        failure_region_match = FAILURE_REGION_RE.search(row.section_D_raw)
        if failure_region_match:
            row.failure_region = failure_region_match.group("region").strip()
            row.failure_region_confidence = float(failure_region_match.group("confidence"))

        behavior_bias_match = BEHAVIOR_BIAS_RE.search(row.section_D_raw)
        if behavior_bias_match:
            row.behavior_bias = behavior_bias_match.group("bias").strip()
            row.behavior_bias_confidence = float(behavior_bias_match.group("confidence"))

        sample_count_match = SAMPLE_COUNT_RE.search(row.section_D_raw)
        if sample_count_match:
            row.sample_count = int(sample_count_match.group("count"))


def _parse_hydra_config(config_path: Path) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    with config_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.rstrip("\n")
            for key, pattern in CONFIG_PATTERNS.items():
                match = pattern.match(stripped)
                if not match:
                    continue
                raw_value = match.group("value")
                if raw_value in {"true", "false"}:
                    result[key] = raw_value == "true"
                elif "." in raw_value or "e" in raw_value.lower():
                    result[key] = float(raw_value)
                else:
                    result[key] = int(raw_value)
    return result


def _parse_wandb_summary(summary_path: Path) -> Dict[str, Any]:
    with summary_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    return {
        "final_validate_reward": data.get("validate_reward"),
        "final_seen_success_rate": data.get("eval/success_rate[seen]"),
        "final_evolve_count": data.get("acgs/evolve_count"),
        "final_batch_success_rate": data.get("acgs/batch_success_rate"),
        "final_batch_collision_rate": data.get("acgs/batch_collision_rate"),
        "final_num_timesteps": data.get("num_timesteps"),
        "final_num_episode": data.get("num_episode"),
        "final_episode_mean_reward": data.get("episode_mean_reward"),
    }


def _find_wandb_summary_path(run_path: Path) -> Optional[Path]:
    candidates = sorted(
        run_path.glob("wandb/run-*/files/wandb-summary.json"),
        key=lambda path: path.stat().st_mtime,
    )
    return candidates[-1].resolve() if candidates else None


def _index_eval_videos(run_path: Path, run_id: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    video_paths = sorted(run_path.glob("wandb/run-*/files/media/videos/eval/*.mp4"))
    for video_path in video_paths:
        match = VIDEO_RE.match(video_path.name)
        if not match:
            continue
        rows.append(
            {
                "run_id": run_id,
                "update_step": int(match.group("step")),
                "video_slot": int(match.group("slot")),
                "video_path": str(video_path.resolve()),
                "file_name": video_path.name,
            }
        )
    return rows


def _merge_and_validate(
    run_id: str,
    run_path: Path,
    hydra_config_path: Optional[Path],
    train_log_path: Path,
    evolve_prompt_log_path: Optional[Path],
    wandb_summary_path: Optional[Path],
    config_meta: Dict[str, Any],
    summary_meta: Dict[str, Any],
    train_tables: Dict[str, Any],
    evolve_rounds: List[EvolveRoundRecord],
    media_index: List[Dict[str, Any]],
) -> ParsedRun:
    snapshots: List[SnapshotRecord] = train_tables["layout_snapshots"]
    obstacles: List[ObstacleRecord] = train_tables["obstacle_points"]
    batches: List[BatchRecord] = train_tables["batch_stats"]
    warnings: List[ParseWarning] = list(train_tables["warnings"])

    evolve_by_episode = {row.episode_total: row for row in evolve_rounds}
    for batch in batches:
        if batch.evolve_decision != "triggered":
            continue
        evolve_row = evolve_by_episode.get(batch.batch_end_episode)
        if evolve_row is None:
            warnings.append(
                ParseWarning(
                    run_id=run_id,
                    category="missing_evolve_round",
                    message=f"Triggered batch {batch.batch_idx} has no matching evolve prompt round",
                    batch_idx=batch.batch_idx,
                )
            )
            continue

        batch.linked_evolve_round_index = evolve_row.evolve_round_index
        batch.diagnosis_reliability = evolve_row.diagnosis_reliability
        evolve_row.linked_batch_idx = batch.batch_idx
        evolve_row.prompt_length_chars = batch.prompt_length_chars
        evolve_row.generator_loaded_successfully = batch.new_generator_loaded
        if batch.evolve_id_loaded is not None:
            evolve_row.evolve_id = batch.evolve_id_loaded

    for snapshot in snapshots:
        matched_batch_idx: Optional[int] = None
        for batch in batches:
            if batch.batch_start_episode <= snapshot.episode < batch.batch_end_episode:
                matched_batch_idx = batch.batch_idx
                break
        snapshot.batch_idx = matched_batch_idx
        snapshot.stage_id = snapshot.generator_id
        snapshot.stage_label = f"generator_{snapshot.generator_id}"

    snapshot_by_id = {snapshot.snapshot_id: snapshot for snapshot in snapshots}
    for obstacle in obstacles:
        snapshot = snapshot_by_id.get(obstacle.snapshot_id)
        if snapshot is None:
            warnings.append(
                ParseWarning(
                    run_id=run_id,
                    category="orphan_obstacle_record",
                    message=f"Obstacle record references missing snapshot {obstacle.snapshot_id}",
                    snapshot_id=obstacle.snapshot_id,
                    episode=obstacle.episode,
                )
            )
            continue
        obstacle.batch_idx = snapshot.batch_idx
        obstacle.stage_id = snapshot.stage_id
        obstacle.stage_label = snapshot.stage_label

    _validate_snapshots(run_id, snapshots, obstacles, warnings)
    _validate_batches(run_id, batches, warnings)
    _validate_evolve_alignment(run_id, batches, evolve_rounds, warnings)
    _validate_generator_monotonicity(run_id, snapshots, warnings)

    run_meta: Dict[str, Any] = {
        "run_id": run_id,
        "output_dir": str(run_path),
        "train_log_path": str(train_log_path),
        "evolve_prompt_log_path": str(evolve_prompt_log_path) if evolve_prompt_log_path is not None else None,
        "wandb_summary_path": str(wandb_summary_path) if wandb_summary_path is not None else None,
        "hydra_config_path": str(hydra_config_path) if hydra_config_path is not None else None,
        "enable_acgs_loop": config_meta.get("enable_acgs_loop", bool(batches or evolve_rounds)),
        "detected_acgs_activity": bool(batches or evolve_rounds),
        "configured_obstacle_num": config_meta.get("configured_obstacle_num"),
        "configured_evaluation_interval": config_meta.get("configured_evaluation_interval"),
        "configured_success_rate_high": config_meta.get("configured_success_rate_high"),
        "configured_success_rate_low": config_meta.get("configured_success_rate_low"),
        "configured_plateau_window": config_meta.get("configured_plateau_window"),
        "configured_plateau_threshold": config_meta.get("configured_plateau_threshold"),
        "batch_size_episodes": config_meta.get("configured_evaluation_interval") or (batches[0].batch_episodes if batches else None),
        "num_obstacles": snapshots[0].num_obstacles if snapshots else config_meta.get("configured_obstacle_num"),
        "num_snapshots": len(snapshots),
        "num_obstacle_points": len(obstacles),
        "num_batches": len(batches),
        "num_evolve_rounds": len(evolve_rounds),
        "num_media_files": len(media_index),
        "num_parse_warnings": len(warnings),
        "max_generator_id": max((snapshot.generator_id for snapshot in snapshots), default=0),
        "final_generator_id": max((snapshot.generator_id for snapshot in snapshots), default=0),
    }
    run_meta.update(summary_meta)

    return ParsedRun(
        run_meta=run_meta,
        layout_snapshots=snapshots,
        obstacle_points=obstacles,
        batch_stats=batches,
        evolve_rounds=evolve_rounds,
        media_index=media_index,
        parse_warnings=warnings,
    )


def _validate_snapshots(
    run_id: str,
    snapshots: List[SnapshotRecord],
    obstacles: List[ObstacleRecord],
    warnings: List[ParseWarning],
) -> None:
    obstacle_count_by_snapshot: Dict[int, int] = defaultdict(int)
    for obstacle in obstacles:
        obstacle_count_by_snapshot[obstacle.snapshot_id] += 1

    for snapshot in snapshots:
        count = obstacle_count_by_snapshot.get(snapshot.snapshot_id, 0)
        if count != snapshot.num_obstacles:
            warnings.append(
                ParseWarning(
                    run_id=run_id,
                    category="snapshot_obstacle_count_mismatch",
                    message=(
                        f"Snapshot {snapshot.snapshot_id} expected {snapshot.num_obstacles} "
                        f"obstacles, parsed {count}"
                    ),
                    episode=snapshot.episode,
                    snapshot_id=snapshot.snapshot_id,
                )
            )
        if snapshot.tblock_x is None or snapshot.tblock_y is None or snapshot.tblock_theta_deg is None:
            warnings.append(
                ParseWarning(
                    run_id=run_id,
                    category="missing_tblock_pose",
                    message=f"Snapshot {snapshot.snapshot_id} is missing T-block pose",
                    episode=snapshot.episode,
                    snapshot_id=snapshot.snapshot_id,
                )
            )


def _validate_batches(run_id: str, batches: List[BatchRecord], warnings: List[ParseWarning]) -> None:
    for batch in batches:
        if None in (batch.success_count, batch.collision_count, batch.timeout_count, batch.fall_count):
            warnings.append(
                ParseWarning(
                    run_id=run_id,
                    category="incomplete_batch_stats",
                    message=f"Batch {batch.batch_idx} is missing one or more coarse stats fields",
                    batch_idx=batch.batch_idx,
                )
            )
            continue

        total = batch.success_count + batch.collision_count + batch.timeout_count + batch.fall_count
        if total != batch.batch_episodes:
            warnings.append(
                ParseWarning(
                    run_id=run_id,
                    category="batch_total_mismatch",
                    message=(
                        f"Batch {batch.batch_idx} stats sum to {total}, "
                        f"expected {batch.batch_episodes}"
                    ),
                    batch_idx=batch.batch_idx,
                )
            )


def _validate_evolve_alignment(
    run_id: str,
    batches: List[BatchRecord],
    evolve_rounds: List[EvolveRoundRecord],
    warnings: List[ParseWarning],
) -> None:
    evolve_episode_set = {row.episode_total for row in evolve_rounds}
    for batch in batches:
        if batch.evolve_decision == "triggered" and batch.batch_end_episode not in evolve_episode_set:
            warnings.append(
                ParseWarning(
                    run_id=run_id,
                    category="triggered_batch_missing_evolve_round",
                    message=(
                        f"Batch {batch.batch_idx} triggered evolve at episode {batch.batch_end_episode} "
                        "but no evolve prompt block was found"
                    ),
                    batch_idx=batch.batch_idx,
                )
            )


def _validate_generator_monotonicity(
    run_id: str,
    snapshots: List[SnapshotRecord],
    warnings: List[ParseWarning],
) -> None:
    previous_generator = -1
    for snapshot in sorted(snapshots, key=lambda record: record.episode):
        if snapshot.generator_id < previous_generator:
            warnings.append(
                ParseWarning(
                    run_id=run_id,
                    category="generator_id_decreased",
                    message=(
                        f"Generator id decreased from {previous_generator} "
                        f"to {snapshot.generator_id} at episode {snapshot.episode}"
                    ),
                    episode=snapshot.episode,
                    snapshot_id=snapshot.snapshot_id,
                )
            )
        previous_generator = snapshot.generator_id


def _infer_run_id(run_path: Path) -> str:
    try:
        return str(run_path.relative_to(REPO_ROOT))
    except ValueError:
        return f"{run_path.parent.name}/{run_path.name}"


def _snapshot_to_row(record: SnapshotRecord) -> Dict[str, Any]:
    return {key: getattr(record, key) for key in SNAPSHOT_FIELDNAMES}


def _obstacle_to_row(record: ObstacleRecord) -> Dict[str, Any]:
    return {key: getattr(record, key) for key in OBSTACLE_FIELDNAMES}


def _batch_to_row(record: BatchRecord) -> Dict[str, Any]:
    return {key: getattr(record, key) for key in BATCH_FIELDNAMES}


def _evolve_to_row(record: EvolveRoundRecord) -> Dict[str, Any]:
    return {key: getattr(record, key) for key in EVOLVE_FIELDNAMES}


def _write_csv(path: Path, fieldnames: List[str], rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse failure-driven Push-T experiment logs.")
    parser.add_argument("--run-dir", required=True, help="Path to one experiment output directory.")
    parser.add_argument(
        "--export-dir",
        default=None,
        help="Optional export directory. Defaults to <run-dir>/parsed.",
    )
    args = parser.parse_args()

    parsed_run = parse_run_directory(args.run_dir)
    export_dir = parsed_run.export(Path(args.export_dir) if args.export_dir is not None else None)

    print(f"Parsed run: {parsed_run.run_meta['run_id']}")
    print(f"Exported tables to: {export_dir}")
    print(
        "Counts: "
        f"snapshots={len(parsed_run.layout_snapshots)}, "
        f"obstacles={len(parsed_run.obstacle_points)}, "
        f"batches={len(parsed_run.batch_stats)}, "
        f"evolve_rounds={len(parsed_run.evolve_rounds)}, "
        f"warnings={len(parsed_run.parse_warnings)}"
    )


if __name__ == "__main__":
    main()
