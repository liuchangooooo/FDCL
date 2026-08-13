#!/usr/bin/env python3
"""Render evolve prompts for QA without running training or calling an LLM."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from DIVO.curriculum.attribution import append_attribution_history, compute_attribution, read_jsonl
from DIVO.gpt.prompt_builder import PromptBuilder


def _load_json(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None:
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_text(path: Optional[Path]) -> str:
    if path is None:
        return _default_generator_code()
    return path.read_text(encoding="utf-8")


def _default_generator_code() -> str:
    return """def generate_obstacles(tblock_pose: np.ndarray, num_obstacles: int) -> list:
    tx, ty, ttheta = tblock_pose
    obstacles = []
    for _ in range(num_obstacles):
        obstacles.append({"x": 0.12, "y": 0.12, "purpose": "preview"})
    return obstacles
"""


def _counts_from_result(result: Optional[Dict[str, Any]]) -> Dict[str, int]:
    counts = {"success": 0, "collision": 0, "timeout": 0, "fall": 0}
    if not result:
        return {"success": 12, "collision": 6, "timeout": 4, "fall": 1}

    term_counts = result.get("global_counts", {}).get("termination_counts", {})
    for key in counts:
        counts[key] = int(term_counts.get(key, 0))
    return counts


def _difficulty_reason(
    batch_stats: Dict[str, int],
    low: float = 0.20,
    high: float = 0.80,
) -> str:
    total = max(sum(batch_stats.values()), 1)
    success_rate = batch_stats.get("success", 0) / total
    schedule = "preview_fixed_schedule(evolve_index=1/1)"
    if success_rate > high:
        signal = f"too_easy(sr={success_rate:.3f}>{high:.3f})"
    elif success_rate < low:
        signal = f"too_hard(sr={success_rate:.3f}<{low:.3f})"
    else:
        signal = f"balanced(sr={success_rate:.3f}, range=[{low:.3f},{high:.3f}])"
    return f"{schedule}|difficulty={signal}"


def _history_from_result(result: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not result:
        return []
    records = _records_from_result_dict(result)
    if not records:
        return []
    recomputed = compute_attribution(records)
    history: List[Dict[str, Any]] = []
    append_attribution_history(history, recomputed, max_len=3)
    return history


def _records_from_result_dict(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build a tiny synthetic record set for history previews only.

    Prompt history uses the same summary shape as online training. If the user
    passes an attribution_map.json instead of raw JSONL, reconstructing exact
    records is impossible, so this helper creates a lightweight approximation
    from top cells. It is intentionally used only for preview history text.
    """
    records: List[Dict[str, Any]] = []
    for idx, cell in enumerate(result.get("top_cells", [])[:3], start=1):
        mean_alpha = float(cell.get("mean_alpha", 0.5))
        mean_beta = float(cell.get("mean_beta", 0.2))
        mean_blockage = float(cell.get("mean_blockage", 0.5))
        failures = max(1, int(cell.get("failure_count", 1)))
        for _ in range(failures):
            records.append(
                {
                    "episode_id": idx,
                    "termination": "collision",
                    "failure_key": "collision",
                    "obstacle_z": [
                        {
                            "alpha": mean_alpha,
                            "beta": mean_beta,
                            "blockage": mean_blockage,
                            "d_start": 0.5,
                            "d_goal": 0.5,
                        }
                    ],
                }
            )
    return records


def _result_from_rollouts(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None:
        return None
    result = compute_attribution(read_jsonl(path))
    return result.to_dict()


def _mock_cfa() -> Dict[str, Any]:
    return {
        "factor_recovery_rates": {
            "PATH": 0.32,
            "START": 0.08,
            "GOAL": 0.14,
            "EXIST": 0.41,
        },
        "cause_counts": {
            "path_blockage": 5,
            "start_interference": 1,
            "goal_interference": 2,
            "not_obstacle_specific": 4,
        },
        "summary": "Synthetic preview evidence; not produced by simulator.",
    }


def _write_prompt(
    builder: PromptBuilder,
    output_dir: Path,
    mode: str,
    batch_stats: Dict[str, int],
    generator_code: str,
    attribution_result: Optional[Dict[str, Any]],
    cfa_result: Optional[Dict[str, Any]],
) -> None:
    coverage = attribution_result.get("coverage") if attribution_result else None
    history = _history_from_result(attribution_result) if mode in ("attribution", "cfa") else []
    reason = _difficulty_reason(batch_stats)
    prompt = builder.build_evolve_user(
        batch_stats=batch_stats,
        reason=reason,
        current_generator_code=generator_code,
        feedback_mode=mode,
        attribution_result=attribution_result,
        coverage_summary=coverage,
        attribution_history=history,
        cfa_result=cfa_result if mode == "cfa" else None,
    )
    (output_dir / f"{mode}.txt").write_text(prompt, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview coarse/attribution/cfa evolve prompts without LLM calls."
    )
    parser.add_argument("--task", default="PushT", help="Prompt task directory name.")
    parser.add_argument(
        "--prompt-dir",
        type=Path,
        default=ROOT_DIR / "DIVO" / "gpt" / "prompt",
        help="Root directory containing task prompt templates.",
    )
    parser.add_argument(
        "--attribution-json",
        type=Path,
        default=None,
        help="Optional attribution_map.json produced by online/offline attribution.",
    )
    parser.add_argument(
        "--rollouts-jsonl",
        type=Path,
        default=None,
        help="Optional obstacle_rollouts.jsonl; ignored if --attribution-json is set.",
    )
    parser.add_argument(
        "--generator-code",
        type=Path,
        default=None,
        help="Optional current_generator.py or initial_generator.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "tools" / "prompt_previews",
        help="Directory for coarse.txt, attribution.txt, and cfa.txt.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    attribution_result = _load_json(args.attribution_json)
    if attribution_result is None:
        attribution_result = _result_from_rollouts(args.rollouts_jsonl)

    batch_stats = _counts_from_result(attribution_result)
    generator_code = _load_text(args.generator_code)
    builder = PromptBuilder(task_name=args.task, prompt_dir=str(args.prompt_dir))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for mode in ("coarse", "attribution", "cfa"):
        _write_prompt(
            builder=builder,
            output_dir=args.output_dir,
            mode=mode,
            batch_stats=batch_stats,
            generator_code=generator_code,
            attribution_result=attribution_result,
            cfa_result=_mock_cfa(),
        )

    print(f"Saved prompt previews to {args.output_dir}")
    for mode in ("coarse", "attribution", "cfa"):
        print(f"- {args.output_dir / (mode + '.txt')}")


if __name__ == "__main__":
    main()
