from DIVO.curriculum.attribution import AttributionConfig, compute_attribution
from DIVO.gpt.prompt_builder import PromptBuilder


def _builder():
    return PromptBuilder("PushT", "DIVO/gpt/prompt")


def _record(termination, alpha=0.3, beta=0.1, blockage=0.5):
    return {
        "termination": termination,
        "failure_key": termination,
        "obstacle_z": [
            {
                "alpha": alpha,
                "beta": beta,
                "blockage": blockage,
                "d_start": 0.5,
                "d_goal": 0.5,
            }
        ],
    }


def test_attribution_prompt_hides_top_cells_when_no_failures():
    result = compute_attribution(
        [_record("success") for _ in range(5)],
        AttributionConfig(min_support=1),
    ).to_dict()

    prompt = _builder().build_evolve_user(
        batch_stats={"success": 5, "collision": 0, "timeout": 0, "fall": 0},
        reason="fixed_schedule(evolve_index=1/8)|difficulty=too_easy(sr=1.000>0.800)",
        current_generator_code="def generate_obstacles(tblock_pose, num_obstacles):\n    return []",
        feedback_mode="attribution",
        attribution_result=result,
    )

    assert "no failed obstacle samples" in prompt
    assert "top failure-associated cells" not in prompt


def test_attribution_prompt_has_difficulty_signal_and_no_type1_labels():
    result = compute_attribution(
        [
            _record("collision", alpha=0.3, beta=0.05, blockage=0.8),
            _record("success", alpha=0.8, beta=0.3, blockage=0.1),
        ],
        AttributionConfig(min_support=1, top_k=1, low_k=0),
    ).to_dict()

    prompt = _builder().build_evolve_user(
        batch_stats={"success": 1, "collision": 1, "timeout": 0, "fall": 0},
        reason="fixed_schedule(evolve_index=1/8)|difficulty=balanced(sr=0.500, range=[0.200,0.800])",
        current_generator_code="def generate_obstacles(tblock_pose, num_obstacles):\n    return []",
        feedback_mode="attribution",
        attribution_result=result,
    )

    assert "difficulty_signal: balanced" in prompt
    assert "top failure-associated cells" in prompt
    assert "collision_rod_mid" not in prompt
    assert "collision_tblock_early" not in prompt


def test_evolve_prompt_includes_eurekaverse_style_difficulty_rules():
    prompt = _builder().build_evolve_user(
        batch_stats={"success": 9, "collision": 1, "timeout": 0, "fall": 0},
        reason="fixed_schedule(evolve_index=1/8)|difficulty=too_easy(sr=0.900>0.800)",
        current_generator_code="def generate_obstacles(tblock_pose, num_obstacles):\n    return []",
        feedback_mode="coarse",
    )

    assert "Please follow the guidelines below:" in prompt
    assert "If success_rate is over 80%" in prompt
    assert "If success_rate is below 20%" in prompt
    assert "similar difficulty but different obstacle layouts" in prompt
    assert "number and size of obstacles are fixed" in prompt
    assert "difficulty_signal: too_easy" in prompt


def test_coarse_prompt_does_not_claim_obstacle_level_evidence():
    prompt = _builder().build_evolve_user(
        batch_stats={"success": 6, "collision": 2, "timeout": 2, "fall": 0},
        reason="fixed_schedule(evolve_index=1/8)|difficulty=balanced(sr=0.600, range=[0.200,0.800])",
        current_generator_code="def generate_obstacles(tblock_pose, num_obstacles):\n    return []",
        feedback_mode="coarse",
    )

    assert "success_rate: 0.600" in prompt
    assert "termination_distribution" in prompt
    assert "obstacle-level evidence" not in prompt
    assert "Obstacle-level attribution evidence" not in prompt
    assert "Generator coverage evidence" not in prompt


def test_attribution_prompt_wraps_mode_specific_evidence():
    result = compute_attribution(
        [
            _record("collision", alpha=0.3, beta=0.05, blockage=0.8),
            _record("success", alpha=0.8, beta=0.3, blockage=0.1),
        ],
        AttributionConfig(min_support=1, top_k=1, low_k=0),
    ).to_dict()

    prompt = _builder().build_evolve_user(
        batch_stats={"success": 1, "collision": 1, "timeout": 0, "fall": 0},
        reason="fixed_schedule(evolve_index=1/8)|difficulty=balanced(sr=0.500, range=[0.200,0.800])",
        current_generator_code="def generate_obstacles(tblock_pose, num_obstacles):\n    return []",
        feedback_mode="attribution",
        attribution_result=result,
    )

    assert "We also computed obstacle-level evidence for the current generator" in prompt
    assert "Obstacle-level attribution evidence:" in prompt
    assert "Generator coverage evidence:" in prompt
    assert "top failure-associated cells" in prompt


def test_graph_pattern_prompt_uses_scene_graph_evidence_without_cell_attribution():
    graph_pattern_result = {
        "metadata": {
            "raw_record_count": 120,
            "scene_graph_record_count": 120,
            "skipped_without_scene_graph": 0,
        },
        "global_stats": {
            "num_episodes": 120,
            "failure_count": 48,
            "success_count": 72,
            "failure_rate": 0.4,
        },
        "atomic_predicate_count": 16,
        "candidate_count": 3,
        "top_patterns": [
            {
                "conditions": [
                    "summary.blockage_mean > 0.25",
                    "summary.pair_distance_min <= 0.06",
                ],
                "support": 40,
                "failure_count": 24,
                "failure_rate": 0.6,
                "failure_lift": 1.5,
                "failure_lift_lcb": 1.1,
                "dominant_failure_key": "collision_tblock_mid",
                "dominant_failure_count": 12,
            }
        ],
    }

    prompt = _builder().build_evolve_user(
        batch_stats={"success": 72, "collision": 24, "timeout": 24, "fall": 0},
        reason="fixed_schedule(evolve_index=1/8)|difficulty=too_easy(sr=0.900>0.800)",
        current_generator_code="def generate_obstacles(tblock_pose, num_obstacles):\n    return []",
        feedback_mode="graph_pattern",
        graph_pattern_result=graph_pattern_result,
    )

    assert "Failure-conditioned scene-graph pattern evidence" in prompt
    assert "summary.blockage_mean > 0.25" in prompt
    assert "increase the feasible sampling probability" in prompt
    assert "We also computed obstacle-level evidence" not in prompt
    assert "Obstacle-level attribution evidence" not in prompt
