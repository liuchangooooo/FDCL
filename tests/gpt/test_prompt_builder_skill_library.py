from DIVO.gpt.prompt_builder import PromptBuilder


def _scene(scene_id, realized):
    return {
        "scene_id": scene_id,
        "realized": realized,
        "feasible": int(realized > 0.0),
        "deployed": int(realized >= 0.5),
        "start": [0.1, 0.1, 0.0],
        "obstacles": [{"x": 0.05, "y": 0.02, "purpose": "test"}],
    }


def _skill_signal(direction):
    return {
        "boundary_signal": {
            "K": 4,
            "n_scenes": 3,
            "tau": 0.125,
            "target_delta": 0.15,
            "boundary_count": 1,
            "boundary_rate": 1.0 / 3.0,
            "r_hard": 1.0 / 3.0,
            "r_easy": 1.0 / 3.0,
            "target_rate": 1.0 / 3.0,
            "mean_b": 0.083,
            "valid_rate": 1.0,
            "duplicate_rate": 0.0,
            "w0_success_rate": 0.667,
            "evolution_direction": direction,
        },
        "design_context": {
            "focus": [_scene(1, 0.5)],
            "harden": [_scene(2, 1.0)],
            "avoid": [_scene(3, 0.0)],
        },
    }


def _prompt(direction):
    return PromptBuilder(
        task_name="PushT", prompt_dir="DIVO/gpt/prompt"
    ).build_evolve_user(
        batch_stats={"success": 8, "collision": 2, "timeout": 0, "fall": 0},
        current_generator_code=(
            "def generate_obstacles(tblock_pose, num_obstacles):\n"
            "    return []"
        ),
        feedback_mode="skill_library",
        skill_signal_result=_skill_signal(direction),
    )


def test_skill_library_prompt_includes_boundary_evidence_and_direction():
    builder = PromptBuilder(task_name="PushT", prompt_dir="DIVO/gpt/prompt")
    skill_signal_result = {
        "boundary_signal": {
            "K": 4,
            "n_scenes": 40,
            "tau": 0.125,
            "target_delta": 0.15,
            "boundary_count": 3,
            "boundary_rate": 0.075,
            "r_hard": 0.025,
            "r_easy": 0.900,
            "target_rate": 0.0,
            "mean_b": 0.014,
            "valid_rate": 0.870,
            "duplicate_rate": 0.0,
            "w0_success_rate": 0.925,
            "evolution_direction": "HARDEN",
        },
        "design_context": {"focus": [], "avoid": []},
    }

    prompt = builder.build_evolve_user(
        batch_stats={"success": 4805, "collision": 187, "timeout": 8, "fall": 0},
        reason="fixed_schedule|difficulty=too_easy(sr=0.961>0.800)",
        current_generator_code=(
            "def generate_obstacles(tblock_pose, num_obstacles):\n"
            "    return []"
        ),
        feedback_mode="skill_library",
        skill_signal_result=skill_signal_result,
    )

    assert "Skill-library boundary probe evidence (W_probe):" in prompt
    assert "Difficulty direction: HARDEN" in prompt
    assert "boundary_count: 3" in prompt
    assert "r_hard (p<=tau, library cannot solve): 0.025" in prompt
    assert "r_easy (p>=1-tau, library solves all): 0.900" in prompt
    assert "mean_b (mean p(1-p)): 0.014" in prompt
    assert "w0_success_rate (deployment diagnostic, excluded from p): 0.925" in prompt
    assert "difficulty_signal: too_easy" not in prompt
    assert "create a harder generator" not in prompt


def test_harden_prompt_renders_easy_source_boundary_target_and_hard_guard():
    prompt = _prompt("HARDEN")

    assert "SOURCE_EASY" in prompt
    assert "TARGET_BOUNDARY" in prompt
    assert "GUARD_HARD" in prompt
    assert prompt.index("realized=1.000") < prompt.index("realized=0.500")
    assert prompt.index("realized=0.500") < prompt.index("realized=0.000")


def test_relax_prompt_renders_hard_source_boundary_target_and_easy_guard():
    prompt = _prompt("RELAX")

    assert "SOURCE_HARD" in prompt
    assert "TARGET_BOUNDARY" in prompt
    assert "GUARD_EASY" in prompt
    assert prompt.index("realized=0.000") < prompt.index("realized=0.500")
    assert prompt.index("realized=0.500") < prompt.index("realized=1.000")
