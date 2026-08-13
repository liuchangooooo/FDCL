"""Regression checks for the Nav Hydra configuration/runtime contract."""
import os

from omegaconf import OmegaConf

from nav.train_stage2 import _cfg_to_args


CONFIG = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "config", "nav", "nav_curriculum.yaml",
))


def _load():
    return OmegaConf.load(CONFIG)


def test_runtime_mapping():
    cfg = _load()
    args = _cfg_to_args(cfg)
    assert args.provider == "mock"
    assert (args.provider_temperature, args.provider_max_tokens) == (0.7, 1500)
    assert args.provider_timeout_sec == 5
    assert bool(args.provider_api_key) is True
    assert args.init_mode == "llm"
    assert args.save_artifacts is True
    assert args.verifier_enabled is True
    assert (args.tau_saturation, args.target_delta) == (0.125, 0.15)
    assert args.diversify_on_hold is True
    assert (args.diversify_bc_tolerance, args.diversify_easy_eps) == (2, 0.05)


def test_contract_mismatch_is_rejected():
    cfg = _load()
    cfg.env.goal[0] = 0.60
    try:
        _cfg_to_args(cfg)
    except ValueError as exc:
        assert "env.goal" in str(exc)
    else:
        raise AssertionError("YAML/runtime environment drift must fail fast")


def test_reward_contract_mismatch_is_rejected():
    cfg = _load()
    cfg.env.collision_penalty = 10.0
    try:
        _cfg_to_args(cfg)
    except ValueError as exc:
        assert "env.collision_penalty" in str(exc)
    else:
        raise AssertionError("YAML/runtime failure penalty drift must fail fast")


def test_unvalidated_final_step_is_rejected():
    cfg = _load()
    cfg.training.total_steps = int(cfg.training.eval_every) + 1
    try:
        _cfg_to_args(cfg)
    except ValueError as exc:
        assert "must be divisible" in str(exc)
    else:
        raise AssertionError("final training step must coincide with validation")


def test_unsupported_noop_is_rejected():
    cfg = _load()
    cfg.skill.beta_div = 0.01
    try:
        _cfg_to_args(cfg)
    except ValueError as exc:
        assert "beta_div" in str(exc)
    else:
        raise AssertionError("unsupported diversity reward must not be silently ignored")


def main():
    test_runtime_mapping()
    test_contract_mismatch_is_rejected()
    test_reward_contract_mismatch_is_rejected()
    test_unvalidated_final_step_is_rejected()
    test_unsupported_noop_is_rejected()
    print("Nav YAML/runtime configuration contract OK")
    print("ALL PASS")


if __name__ == "__main__":
    main()
