from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from DIVO.curriculum.learnable_frontier import evaluate_learnable_frontier_shift


def summarize_signal_result(
    signal_result: Optional[Mapping[str, Any]],
    low_threshold: float = 0.2,
    high_threshold: float = 0.8,
) -> Dict[str, Any]:
    """Summarize a skill-signal probe for Phase 4 monitoring only."""

    if not signal_result:
        return _empty_profile()

    profile = dict(signal_result.get("difficulty_profile", {}) or {})
    per_scene = list(signal_result.get("per_scene", []) or [])
    realized = np.asarray(
        [float(row.get("realized", 0.0)) for row in per_scene],
        dtype=np.float64,
    )
    feasible = np.asarray(
        [float(row.get("feasible", 0.0)) for row in per_scene],
        dtype=np.float64,
    )
    lv = np.asarray(
        [float(row.get("lv", 0.0)) for row in per_scene],
        dtype=np.float64,
    )

    if realized.size:
        frac_mastery = float(np.mean(realized >= float(high_threshold)))
        frac_boundary = float(np.mean(
            (feasible > 0.0)
            & (realized > float(low_threshold))
            & (realized < float(high_threshold))
        ))
        frac_low = float(np.mean(realized <= float(low_threshold)))
    else:
        frac_mastery = 0.0
        frac_boundary = 0.0
        frac_low = 0.0

    mean_lv = float(profile.get("mean_lv", np.mean(lv) if lv.size else 0.0))
    return {
        "n_scenes": int(profile.get("n_scenes", len(per_scene))),
        "K": int(profile.get("K", 0)),
        "mean_realized": float(profile.get("mean_realized", realized.mean() if realized.size else 0.0)),
        "mean_lv": mean_lv,
        "frac_infeasible": float(profile.get(
            "frac_infeasible",
            float(np.mean(feasible <= 0.0)) if feasible.size else 0.0,
        )),
        "mean_feasible": float(profile.get("mean_feasible", feasible.mean() if feasible.size else 0.0)),
        "mean_deployed": float(profile.get("mean_deployed", 0.0)),
        "frac_mastery": frac_mastery,
        "frac_boundary": frac_boundary,
        "frac_low_realized": frac_low,
        "realized_hist": profile.get("realized_hist", {}),
        "sampling": profile.get("sampling", {}),
    }


def classify_controller_state(
    profile: Mapping[str, Any],
    easy_realized: float = 0.85,
    hard_realized: float = 0.20,
    infeasible_high: float = 0.30,
    mastery_high: float = 0.70,
    boundary_high: float = 0.25,
) -> str:
    """Classify the current controller state for Phase 4 verdicts."""

    mean_realized = float(profile.get("mean_realized", 0.0))
    frac_infeasible = float(profile.get("frac_infeasible", 0.0))
    frac_mastery = float(profile.get("frac_mastery", 0.0))
    frac_boundary = float(profile.get("frac_boundary", 0.0))

    if frac_infeasible >= float(infeasible_high) or mean_realized <= float(hard_realized):
        return "too_hard"
    if mean_realized >= float(easy_realized) or frac_mastery >= float(mastery_high):
        return "too_easy"
    if frac_boundary >= float(boundary_high):
        return "frontier"
    return "mixed"


def judge_difficulty_shift(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    state_before: Optional[str] = None,
    infeasible_tolerance: float = 0.05,
    min_lv_delta: float = 0.0,
    min_realized_delta: float = 0.0,
    infeasible_cap: Optional[float] = None,
) -> Dict[str, Any]:
    """Judge whether old->new generator moved toward the learnable band."""

    state = state_before or classify_controller_state(before)
    deltas = profile_deltas(before, after)
    frontier_shift = evaluate_learnable_frontier_shift(
        before,
        after,
        min_lv_delta=float(min_lv_delta),
        infeasible_cap=infeasible_cap,
    )
    direction_ok = bool(frontier_shift["accepted"])
    verdict = _frontier_verdict(state, frontier_shift)

    return {
        "state_before": state,
        "direction_ok": bool(direction_ok),
        "verdict": verdict,
        "frontier_shift": frontier_shift,
        **deltas,
    }


def profile_deltas(before: Mapping[str, Any], after: Mapping[str, Any]) -> Dict[str, float]:
    keys = [
        "mean_realized",
        "mean_lv",
        "frac_infeasible",
        "mean_feasible",
        "mean_deployed",
        "frac_mastery",
        "frac_boundary",
    ]
    return {
        f"{key}_delta": float(after.get(key, 0.0)) - float(before.get(key, 0.0))
        for key in keys
    }


def compute_unique_bin_coverage(
    signal_result: Optional[Mapping[str, Any]],
    grid: int = 8,
    xy_limit: float = 0.2,
) -> Dict[str, Any]:
    """Compute a simple 2D obstacle-position bin coverage for collapse checks."""

    grid = max(1, int(grid))
    xy_limit = float(xy_limit)
    if not signal_result:
        return _empty_coverage(grid)

    occupied = set()
    obstacle_count = 0
    scene_count = 0
    for scene in signal_result.get("per_scene", []) or []:
        scene_count += 1
        for obs in scene.get("obstacles", []) or []:
            try:
                x = float(obs.get("x", 0.0))
                y = float(obs.get("y", 0.0))
            except Exception:
                continue
            ix = _to_bin(x, grid=grid, xy_limit=xy_limit)
            iy = _to_bin(y, grid=grid, xy_limit=xy_limit)
            occupied.add((ix, iy))
            obstacle_count += 1

    total_bins = grid * grid
    unique_bins = len(occupied)
    return {
        "grid": int(grid),
        "xy_limit": float(xy_limit),
        "n_scenes": int(scene_count),
        "n_obstacles": int(obstacle_count),
        "unique_bins": int(unique_bins),
        "total_bins": int(total_bins),
        "coverage_ratio": float(unique_bins / total_bins) if total_bins else 0.0,
        "occupied_bins": [[int(i), int(j)] for i, j in sorted(occupied)],
    }


def build_phase4_record(
    evolve_index: int,
    total_episode_count: int,
    context_signal: Optional[Mapping[str, Any]],
    verifier_audits: Sequence[Mapping[str, Any]],
    paired_before: Optional[Mapping[str, Any]],
    paired_after: Optional[Mapping[str, Any]],
    fresh_before: Optional[Mapping[str, Any]] = None,
    fresh_after: Optional[Mapping[str, Any]] = None,
    paired_config: Optional[Mapping[str, Any]] = None,
    fresh_config: Optional[Mapping[str, Any]] = None,
    coverage_enabled: bool = True,
    coverage_grid: int = 8,
    low_threshold: float = 0.2,
    high_threshold: float = 0.8,
    infeasible_cap: Optional[float] = None,
) -> Dict[str, Any]:
    context_profile = summarize_signal_result(
        context_signal,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
    )
    paired_before_profile = summarize_signal_result(
        paired_before,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
    )
    paired_after_profile = summarize_signal_result(
        paired_after,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
    )
    state_before = classify_controller_state(paired_before_profile)
    paired_shift = judge_difficulty_shift(
        paired_before_profile,
        paired_after_profile,
        state_before=state_before,
        infeasible_cap=infeasible_cap,
    )

    record: Dict[str, Any] = {
        "evolve_index": int(evolve_index),
        "total_episode_count": int(total_episode_count),
        "context_profile": context_profile,
        "verifier": summarize_verifier_audits(verifier_audits),
        "paired_shift": {
            "config": dict(paired_config or {}),
            "before": paired_before_profile,
            "after": paired_after_profile,
            "judgement": paired_shift,
        },
    }
    if coverage_enabled:
        record["paired_shift"]["bin_coverage"] = compare_bin_coverage(
            paired_before,
            paired_after,
            grid=coverage_grid,
        )

    if fresh_before is not None and fresh_after is not None:
        fresh_before_profile = summarize_signal_result(
            fresh_before,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
        )
        fresh_after_profile = summarize_signal_result(
            fresh_after,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
        )
        fresh_shift = judge_difficulty_shift(
            fresh_before_profile,
            fresh_after_profile,
            state_before=classify_controller_state(fresh_before_profile),
            infeasible_cap=infeasible_cap,
        )
        record["fresh_validation"] = {
            "config": dict(fresh_config or {}),
            "before": fresh_before_profile,
            "after": fresh_after_profile,
            "judgement": fresh_shift,
        }
        if coverage_enabled:
            record["fresh_validation"]["bin_coverage"] = compare_bin_coverage(
                fresh_before,
                fresh_after,
                grid=coverage_grid,
            )

    return record


def _frontier_verdict(state: str, frontier_shift: Mapping[str, Any]) -> str:
    if bool(frontier_shift.get("accepted", False)):
        return f"{state} -> moved toward learnable frontier"
    reason = str(frontier_shift.get("reason", "candidate_not_accepted"))
    if reason == "candidate_infeasible_cap_exceeded":
        return f"{state} -> infeasible cap exceeded"
    if reason == "candidate_moved_away_from_frontier":
        return f"{state} -> learning value improved but difficulty center moved away"
    if reason == "candidate_did_not_improve_learning_value":
        return f"{state} -> did not improve learning value"
    if reason == "candidate_did_not_improve_learning_value_and_moved_away_from_frontier":
        return f"{state} -> did not improve learning value and moved away"
    return f"{state} -> {reason}"


def summarize_verifier_audits(
    audits: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    rows = [dict(row) for row in audits or []]
    accepted = [row for row in rows if bool(row.get("accepted", False))]
    selected = accepted[-1] if accepted else None
    return {
        "attempt_count": int(len(rows)),
        "accepted": bool(selected is not None),
        "selected_attempt": int(selected.get("attempt")) if selected else None,
        "selected_reason": selected.get("reason") if selected else None,
        "selected_score_delta": float(selected.get("score_delta", 0.0)) if selected else None,
        "attempts": rows,
    }


def compare_bin_coverage(
    before_signal: Optional[Mapping[str, Any]],
    after_signal: Optional[Mapping[str, Any]],
    grid: int = 8,
) -> Dict[str, Any]:
    before = compute_unique_bin_coverage(before_signal, grid=grid)
    after = compute_unique_bin_coverage(after_signal, grid=grid)
    return {
        "before": before,
        "after": after,
        "unique_bins_delta": int(after["unique_bins"]) - int(before["unique_bins"]),
        "coverage_ratio_delta": float(after["coverage_ratio"]) - float(before["coverage_ratio"]),
        "collapsed": bool(
            after["unique_bins"] < max(1, int(0.5 * max(before["unique_bins"], 1)))
        ),
    }


def _to_bin(value: float, grid: int, xy_limit: float) -> int:
    scaled = (float(value) + xy_limit) / (2.0 * xy_limit)
    idx = int(np.floor(scaled * grid))
    return int(np.clip(idx, 0, grid - 1))


def _empty_profile() -> Dict[str, Any]:
    return {
        "n_scenes": 0,
        "K": 0,
        "mean_realized": 0.0,
        "mean_lv": 0.0,
        "frac_infeasible": 0.0,
        "mean_feasible": 0.0,
        "mean_deployed": 0.0,
        "frac_mastery": 0.0,
        "frac_boundary": 0.0,
        "frac_low_realized": 0.0,
        "realized_hist": {},
        "sampling": {},
    }


def _empty_coverage(grid: int) -> Dict[str, Any]:
    return {
        "grid": int(grid),
        "xy_limit": 0.2,
        "n_scenes": 0,
        "n_obstacles": 0,
        "unique_bins": 0,
        "total_bins": int(grid) * int(grid),
        "coverage_ratio": 0.0,
        "occupied_bins": [],
    }
