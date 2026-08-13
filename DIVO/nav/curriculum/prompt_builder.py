"""Navigate adapter for the shared Push-T/Navigation prompt builder.

Prompt rendering lives in :mod:`gpt.prompt_builder`.  This module only keeps
the Nav curriculum's direction rule and translates its flat probe statistics
into the shared builder's task-neutral evidence schema.
"""
import os

from gpt.prompt_builder import PromptBuilder
from nav.skill_signal import TARGET_DELTA, TAU


PROMPT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "gpt", "prompt"))
SHARED_BUILDER = PromptBuilder(task_name="Navigate", prompt_dir=PROMPT_ROOT)
PROMPT_DIR = SHARED_BUILDER.task_prompt_dir


def load_initial_system():
    return SHARED_BUILDER.load_initial_system()


def load_initial_user():
    return SHARED_BUILDER.load_initial_user()


def load_evolve_system():
    return SHARED_BUILDER.load_evolve_system()


def load_evolve_user():
    return SHARED_BUILDER.load_evolve_user()


def decide_direction(signal, r_easy_max=0.8, r_hard_max=0.8, v_min=0.8):
    """Use the same validity-first, symmetric boundary direction as Push-T."""
    if signal.get("valid_rate", 1.0) < v_min:
        return "FIX_VALIDITY"
    if signal.get("r_easy", 0.0) >= r_easy_max:
        return "HARDEN"
    if signal.get("r_hard", 0.0) >= r_hard_max:
        return "RELAX"
    return "PRESERVE_AND_DIVERSIFY"


def build_initial(agent_start, goal):
    """Return the shared builder's Navigate Stage-0 prompt pair."""
    return (
        SHARED_BUILDER.load_initial_system(),
        SHARED_BUILDER.build_initial_user_nav(agent_start, goal),
    )


def _to_shared_skill_signal(signal, direction):
    """Translate Nav's flat W_probe result without inventing missing evidence."""
    return {
        "boundary_signal": {
            "K": int(signal.get("K", 0)),
            "n_scenes": int(signal.get("N", len(signal.get("per_scene_p", [])))),
            "tau": float(signal.get("tau", TAU)),
            "target_delta": float(signal.get("target_delta", TARGET_DELTA)),
            "boundary_count": int(signal.get("boundary_count", 0)),
            "boundary_rate": float(signal.get("boundary_rate", 0.0)),
            "r_hard": float(signal.get("r_hard", 0.0)),
            "r_easy": float(signal.get("r_easy", 0.0)),
            "target_rate": float(signal.get("target_rate", 0.0)),
            "mean_b": float(signal.get("mean_b", 0.0)),
            "valid_rate": float(signal.get("valid_rate", 0.0)),
            "duplicate_rate": float(signal.get("duplicate_rate", 0.0)),
            "w0_success_rate": float(signal.get("w0_success_rate", 0.0)),
            "evolution_direction": direction,
        },
        "design_context": signal.get("design_context") or {
            "focus": [], "harden": [], "avoid": [],
        },
    }


def build_evolve(current_code, signal, direction=None, batch_stats=None):
    """Return a shared-builder Navigate evolve prompt pair.

    ``batch_stats`` is the real training-batch termination distribution.  The
    W_probe result remains a separate curriculum signal, as in Push-T.
    """
    if direction is None:
        direction = decide_direction(signal)
    counts = {
        key: int((batch_stats or {}).get(key, 0))
        for key in ("success", "collision", "timeout", "fall")
    }
    user = SHARED_BUILDER.build_evolve_user(
        batch_stats=counts,
        current_generator_code=current_code,
        feedback_mode="skill_library",
        skill_signal_result=_to_shared_skill_signal(signal, direction),
        include_behavior=bool(signal.get("include_behavior", True)),
    )
    return SHARED_BUILDER.load_evolve_system(), user
