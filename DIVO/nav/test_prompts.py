"""Navigate prompts mirror the Push-T prompt structure with task-only substitutions."""
import os

from gpt.prompt_builder import PromptBuilder
from nav import nav_env as NE
from nav.curriculum import prompt_builder as PB


PUSHT_PROMPT_DIR = os.path.join(os.path.dirname(PB.PROMPT_DIR), "PushT")


def _load_pusht(name):
    with open(os.path.join(PUSHT_PROMPT_DIR, name), "r", encoding="utf-8") as f:
        return f.read()


def _headings(text):
    structural = {
        "Problem Statement:", "Function Contract:", "Inputs:", "Return Format:",
        "Environment Constraints:", "Design Goal:", "Allowed Tools:", "Final Answer:",
        "Inputs and return format:", "Environment constraints:", "Revision principles:",
        "Initial generator requirements:", "Please follow the guidelines below:",
    }
    return [line.strip() for line in text.splitlines() if line.strip() in structural]


def _bullet_count(text):
    return sum(line.startswith("- ") for line in text.splitlines())


def test_prompt_contract():
    # Nav is a thin adapter over the exact shared runtime renderer.
    assert isinstance(PB.SHARED_BUILDER, PromptBuilder)
    assert PB.SHARED_BUILDER.task_name == "Navigate"

    initial_system = PB.load_initial_system()
    initial_user = PB.load_initial_user()
    evolve_system = PB.load_evolve_system()
    evolve_user = PB.load_evolve_user()

    # 四个模板与 Push-T 保持相同章节顺序和约束密度。
    for name, nav_text in (
        ("initial_system.txt", initial_system),
        ("initial_user.txt", initial_user),
        ("evolve_system.txt", evolve_system),
        ("evolve_user.txt", evolve_user),
    ):
        pusht_text = _load_pusht(name)
        assert _headings(nav_text) == _headings(pusht_text), name
        assert _bullet_count(nav_text) == _bullet_count(pusht_text), name

    required = (
        "def generate_pillars",
        "`(+0.65, 0.0)`",
        "`x,y in [-0.5, 0.5]`",
        "is_safe(x, y, sx, sy, gx, gy)",
    )
    for text in (initial_system, evolve_system):
        missing = [token for token in required if token not in text]
        assert not missing, f"Navigate task substitutions missing:{missing}"

    # 不向 Nav generator 暴露 Push-T prompt 中没有的 benchmark/物理先验。
    forbidden = (
        "B/M/U/D", "Gremlin", "barrier", "dead-end", "size=", "keepout=",
        ">= 0.31", ">= 0.35", ">= 0.25",
    )
    all_nav = "\n".join((initial_system, initial_user, evolve_system, evolve_user))
    leaked = [token for token in forbidden if token in all_nav]
    assert not leaked, f"Navigate prompt contains extra benchmark/physics priors:{leaked}"

    system, user = PB.build_initial((-0.8, 0.2), NE.GOAL)
    assert system == initial_system
    assert "x=-0.800, y=0.200" in user
    assert "x=0.650, y=0.000" in user

    signal = {
        "K": 4,
        "N": 4,
        "tau": 0.125,
        "target_delta": 0.15,
        "boundary_count": 2,
        "boundary_rate": 0.5,
        "mean_b": 0.2,
        "r_hard": 0.1,
        "r_easy": 0.2,
        "target_rate": 0.3,
        "valid_rate": 1.0,
        "duplicate_rate": 0.0,
        "w0_success_rate": 0.5,
        "include_behavior": True,
        "design_context": {
            "focus": [{
                "scene_id": 0, "start": [-0.7, 0.1],
                "obstacles": [
                    {"x": 0.0, "y": 0.2, "purpose": ""},
                    {"x": 0.2, "y": -0.2, "purpose": ""},
                ],
                "feasible": 1, "realized": 0.5, "deployed": 1,
                "behavior_summary": {"routes": [{
                    "skill_index": 1, "success": True,
                    "waypoints": [{"alpha": 0.0, "beta": 0.0}],
                }]},
            }],
            "harden": [],
            "avoid": [],
        },
    }
    current_code = (
        "def generate_pillars(agent_start, goal, num_pillars):\n"
        "    return [{'x': 0.0, 'y': 0.4, 'purpose': 'fallback'}]"
    )
    system, user = PB.build_evolve(
        current_code,
        signal,
        direction="HARDEN",
        batch_stats={"success": 3, "collision": 1, "timeout": 0, "fall": 0},
    )
    assert system == evolve_system
    assert current_code in user
    required_runtime = (
        "Skill-library boundary probe evidence (W_probe):",
        "Difficulty direction: HARDEN",
        "Boundary snapshot (K=4, n=4):",
        "SOURCE_EASY",
        "TARGET_BOUNDARY",
        "GUARD_HARD",
        "task_frame_waypoints(alpha,beta)",
        "Return exactly one complete fenced Python code block defining `generate_pillars`.",
    )
    missing = [token for token in required_runtime if token not in user]
    assert not missing, f"shared runtime sections missing:{missing}"
    assert "pillars:" in user
    assert "out_of_bounds" in user
    assert "T-block" not in user
    assert "obstacle" not in user.lower()
    for placeholder in (
        "{policy_statistics}",
        "{feedback_evidence}",
        "{current_generator_context}",
        "{evolution_request}",
        "{task_objective_block}",
        "{performance_block}",
        "{revision_request_block}",
    ):
        assert placeholder not in user


def main():
    test_prompt_contract()
    print("Navigate/Push-T prompt structural parity OK")
    print("ALL PASS")


if __name__ == "__main__":
    main()
