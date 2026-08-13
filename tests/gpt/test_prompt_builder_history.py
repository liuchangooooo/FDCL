from DIVO.gpt.prompt_builder import PromptBuilder


def test_attribution_history_uses_real_evolve_index_and_episode():
    builder = PromptBuilder(task_name="PushT", prompt_dir="DIVO/gpt/prompt")
    text = builder._format_attribution_history_block(
        [
            {
                "evolve_index": 1,
                "episode_idx": 1000,
                "success_rate_at_evolve": 0.034,
                "trigger_reason": "fixed_schedule(evolve_index=1/4)|difficulty=too_hard",
            }
        ],
        current_success_rate=0.216,
    )

    assert "after Round 1 (episode=1000):" in text
    assert "success_rate 0.034 -> current 0.216 (+0.182)" in text
    assert "Round -1" not in text
    assert "dominant_cell" not in text
    assert "difficulty=too_hard" not in text
