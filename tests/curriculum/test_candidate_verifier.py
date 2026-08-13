import numpy as np
import pytest

from DIVO.curriculum.attribution import AttributionConfig
from DIVO.curriculum.candidate_verifier import (
    CandidateVerifierConfig,
    SkillGeneratorScore,
    audit_pusht_generator_code,
    build_rank_evidence_values,
    decide_skill_acceptance,
    verify_pusht_candidate,
)
from DIVO.curriculum.learnable_frontier import evaluate_learnable_frontier_shift
from DIVO.curriculum.phase4_monitor import judge_difficulty_shift


def _cell(cell_id, lift, total=20, failures=10, p_cell=0.1, failure_rate=0.5):
    alpha, beta_abs, blockage = [
        part.split("=", 1)[1] for part in cell_id.split("|")
    ]
    return {
        "cell_id": cell_id,
        "alpha_bin": alpha,
        "beta_abs_bin": beta_abs,
        "blockage_bin": blockage,
        "total_count": total,
        "failure_count": failures,
        "failure_lift": lift,
        "p_cell": p_cell,
        "failure_rate": failure_rate,
    }


def _attr(cells, min_support=2):
    cfg = AttributionConfig(min_support=min_support)
    return {
        "metadata": {"config": cfg.to_dict()},
        "cells": cells,
    }


EVIDENCE_CODE = """
def generate_obstacles(tblock_pose, num_obstacles):
    tx, ty, theta = tblock_pose
    start = np.array([tx, ty], dtype=float)
    choices = [(0.50, 0.0), (0.45, 0.06), (0.55, -0.06), (0.40, 0.0)]
    obstacles = []
    for alpha, beta in choices:
        x, y = decode_obstacle(alpha, beta, start)
        if -0.2 <= x <= 0.2 and -0.2 <= y <= 0.2 and is_safe(x, y, tx, ty, theta):
            obstacles.append({"x": float(x), "y": float(y), "purpose": "evidence"})
            break
    return obstacles[:num_obstacles]
"""


LOW_VALUE_CODE = """
def generate_obstacles(tblock_pose, num_obstacles):
    tx, ty, theta = tblock_pose
    start = np.array([tx, ty], dtype=float)
    choices = [(0.50, 0.35), (0.45, -0.35), (0.60, 0.32), (0.40, -0.32)]
    obstacles = []
    for alpha, beta in choices:
        x, y = decode_obstacle(alpha, beta, start)
        if -0.2 <= x <= 0.2 and -0.2 <= y <= 0.2 and is_safe(x, y, tx, ty, theta):
            obstacles.append({"x": float(x), "y": float(y), "purpose": "low_value"})
            break
    return obstacles[:num_obstacles]
"""


EMPTY_CODE = """
def generate_obstacles(tblock_pose, num_obstacles):
    return []
"""


def test_build_rank_evidence_values_filters_and_normalizes():
    attr = _attr(
        [
            _cell("alpha=050_to_075|beta_abs=centerline|blockage=high", 4.0, total=20),
            _cell("alpha=025_to_050|beta_abs=centerline|blockage=high", 2.0, total=20),
            _cell("alpha=075_to_goal|beta_abs=far_side|blockage=low", 0.9, total=20),
            _cell("alpha=start_to_025|beta_abs=near_side|blockage=medium", 9.0, total=1),
            _cell("alpha=after_goal|beta_abs=far_side|blockage=low", 3.0, total=20, failures=0),
        ],
        min_support=2,
    )

    values = build_rank_evidence_values(attr)

    assert [row.cell_id for row in values] == [
        "alpha=050_to_075|beta_abs=centerline|blockage=high",
        "alpha=025_to_050|beta_abs=centerline|blockage=high",
    ]
    assert values[0].value == pytest.approx(1.0)
    assert values[1].value == pytest.approx(0.5)


def test_audit_score_prefers_evidence_cells_on_same_poses():
    attr = _attr(
        [
            _cell("alpha=050_to_075|beta_abs=centerline|blockage=high", 4.0),
            _cell("alpha=025_to_050|beta_abs=centerline|blockage=high", 2.0),
        ]
    )
    cfg = AttributionConfig.from_dict(attr["metadata"]["config"])
    values = {cell.cell_id: cell.value for cell in build_rank_evidence_values(attr, cfg)}
    poses = [
        np.array([0.18, 0.18, 0.0], dtype=np.float64),
        np.array([-0.18, 0.16, 1.0], dtype=np.float64),
        np.array([0.16, -0.18, 2.0], dtype=np.float64),
    ]

    low = audit_pusht_generator_code(LOW_VALUE_CODE, poses, 1, cfg, values)
    high = audit_pusht_generator_code(EVIDENCE_CODE, poses, 1, cfg, values)

    assert high.num_obstacle_samples > 0
    assert high.score > low.score


def test_verify_candidate_accepts_increase_for_too_easy():
    attr = _attr(
        [
            _cell("alpha=050_to_075|beta_abs=centerline|blockage=high", 4.0),
            _cell("alpha=025_to_050|beta_abs=centerline|blockage=high", 2.0),
        ]
    )
    result = verify_pusht_candidate(
        current_generator_code=LOW_VALUE_CODE,
        candidate_generator_code=EVIDENCE_CODE,
        attribution_result=attr,
        config=CandidateVerifierConfig(
            num_pose_samples=20,
            obstacle_num=1,
            seed=1,
            success_rate=0.9,
            success_high=0.8,
            success_low=0.2,
            min_valid_generation_rate=0.0,
        ),
    )

    assert result.accepted is True
    assert result.direction == "increase"
    assert result.candidate.score > result.current.score


def test_verify_candidate_rejects_lower_score_for_too_easy():
    attr = _attr(
        [
            _cell("alpha=050_to_075|beta_abs=centerline|blockage=high", 4.0),
            _cell("alpha=025_to_050|beta_abs=centerline|blockage=high", 2.0),
        ]
    )
    result = verify_pusht_candidate(
        current_generator_code=EVIDENCE_CODE,
        candidate_generator_code=LOW_VALUE_CODE,
        attribution_result=attr,
        config=CandidateVerifierConfig(
            num_pose_samples=20,
            obstacle_num=1,
            seed=1,
            success_rate=0.9,
            success_high=0.8,
            success_low=0.2,
            min_valid_generation_rate=0.0,
        ),
    )

    assert result.accepted is False
    assert result.reason == "candidate_did_not_increase_evidence_score"
    assert "current_evidence_score" in result.feedback_text


def test_verify_candidate_rejects_low_valid_generation_rate():
    attr = _attr(
        [
            _cell("alpha=050_to_075|beta_abs=centerline|blockage=high", 4.0),
            _cell("alpha=025_to_050|beta_abs=centerline|blockage=high", 2.0),
        ]
    )
    result = verify_pusht_candidate(
        current_generator_code=LOW_VALUE_CODE,
        candidate_generator_code=EMPTY_CODE,
        attribution_result=attr,
        config=CandidateVerifierConfig(
            num_pose_samples=20,
            obstacle_num=1,
            seed=1,
            success_rate=0.9,
            success_high=0.8,
            success_low=0.2,
            min_valid_generation_rate=0.95,
        ),
    )

    assert result.accepted is False
    assert result.reason == "candidate_low_valid_generation_rate"
    assert result.candidate.valid_generation_rate == 0.0


def _skill_score(score, mean_realized, frac_infeasible=0.0, valid_generation_rate=1.0):
    return SkillGeneratorScore(
        score=float(score),
        frac_infeasible=float(frac_infeasible),
        mean_realized=float(mean_realized),
        mean_feasible=1.0,
        mean_deployed=float(mean_realized),
        valid_generation_rate=float(valid_generation_rate),
    )


def test_learnable_frontier_shift_accepts_harden_and_ease_without_state_thresholds():
    harden = evaluate_learnable_frontier_shift(
        {"mean_lv": 0.02, "mean_realized": 0.95, "frac_infeasible": 0.0},
        {"mean_lv": 0.08, "mean_realized": 0.75, "frac_infeasible": 0.0},
        infeasible_cap=0.30,
    )
    ease = evaluate_learnable_frontier_shift(
        {"mean_lv": 0.02, "mean_realized": 0.05, "frac_infeasible": 0.20},
        {"mean_lv": 0.08, "mean_realized": 0.25, "frac_infeasible": 0.10},
        infeasible_cap=0.30,
    )

    assert harden["accepted"] is True
    assert harden["frontier_distance_delta"] < 0.0
    assert ease["accepted"] is True
    assert ease["frontier_distance_delta"] < 0.0


def test_skill_acceptance_rejects_lv_gain_that_moves_away_from_frontier():
    current = _skill_score(score=0.09, mean_realized=0.84)
    candidate = _skill_score(score=0.10, mean_realized=0.88)

    accepted, reason, frontier_shift = decide_skill_acceptance(
        current=current,
        candidate=candidate,
        min_valid_generation_rate=0.95,
        infeasible_cap=0.30,
        min_score_delta=0.0,
    )

    assert accepted is False
    assert reason == "candidate_moved_away_from_frontier"
    assert frontier_shift["lv_improved"] is True
    assert frontier_shift["frontier_distance_delta"] > 0.0


def test_skill_acceptance_still_enforces_absolute_infeasible_cap():
    current = _skill_score(score=0.02, mean_realized=0.95, frac_infeasible=0.0)
    candidate = _skill_score(score=0.08, mean_realized=0.75, frac_infeasible=0.40)

    accepted, reason, frontier_shift = decide_skill_acceptance(
        current=current,
        candidate=candidate,
        min_valid_generation_rate=0.95,
        infeasible_cap=0.30,
        min_score_delta=0.0,
    )

    assert accepted is False
    assert reason == "candidate_infeasible_cap_exceeded"
    assert frontier_shift["infeasible_cap_ok"] is False


def test_phase4_judgement_uses_same_learnable_frontier_semantics():
    judgement = judge_difficulty_shift(
        before={"mean_lv": 0.09, "mean_realized": 0.84, "frac_infeasible": 0.0},
        after={"mean_lv": 0.10, "mean_realized": 0.88, "frac_infeasible": 0.0},
        infeasible_cap=0.30,
    )

    assert judgement["direction_ok"] is False
    assert judgement["frontier_shift"]["reason"] == "candidate_moved_away_from_frontier"
    assert "moved away" in judgement["verdict"]
