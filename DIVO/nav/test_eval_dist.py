"""验证 best 选模使用整回合 reward,成功率只作诊断。"""
import numpy as np

from nav import nav_env as NE
from nav.curriculum.generator_source import (
    GOAL_CLEAR,
    MIN_PAIR,
    START_CLEAR,
    validate_pillars,
)
from nav.eval_dist import _between_start, evaluate_validation, sample_validation_scene


class _Body:
    def __init__(self):
        self.xpos = np.zeros(3, dtype=float)


class _Data:
    def __init__(self):
        self._body = _Body()

    def body(self, _name):
        return self._body


class _Adapter:
    goal = (0.0, 0.0)

    def __init__(self):
        self._episodes = [
            [(0.2, False, False), (1.3, True, True)],
            [(0.1, False, False), (0.2, False, True)],
        ]
        self._episode = -1
        self._step = 0
        self._env = type("Env", (), {
            "task": type("Task", (), {"data": _Data()})(),
        })()

    def set_layout(self, *_args, **_kwargs):
        return None

    def reset(self, **_kwargs):
        self._episode += 1
        self._step = 0
        return np.zeros(44, dtype=np.float32)

    def step(self, _action):
        reward, success, terminated = self._episodes[self._episode][self._step]
        self._step += 1
        return np.zeros(44, dtype=np.float32), reward, terminated, False, {"success": success}

    @staticmethod
    def success(info):
        return bool(info["success"])


def test_episode_return_is_model_selection_score():
    result = evaluate_validation(
        _Adapter(), lambda _obs: np.zeros(2), n_env=2, max_steps=4,
    )
    # episode returns:1.5 and 0.3 -> mean=0.9;success rate=1/2。
    assert np.isclose(result["test_mean_score"], 0.9)
    assert np.isclose(result["mean_episode_return"], 0.9)
    assert np.isclose(result["success_rate"], 0.5)
    assert np.isclose(result["collision_rate"], 0.0)
    assert np.isclose(result["oob_rate"], 0.0)
    assert np.isclose(result["timeout_rate"], 0.5)
    assert np.isclose(
        result["success_rate"]
        + result["collision_rate"]
        + result["oob_rate"]
        + result["timeout_rate"],
        1.0,
    )


class _OutOfRangeRng:
    @staticmethod
    def integers(_n):
        return 0

    @staticmethod
    def uniform(low, high):
        return high


def test_between_start_is_not_clipped():
    start = _between_start(
        _OutOfRangeRng(), pillars=[(0.0, 0.4)], goal=NE.GOAL,
    )
    assert start[1] > NE.START_Y_RANGE[1]


def test_validation_geometry_matches_training_contract():
    assert np.isclose(START_CLEAR, 0.45)
    assert np.isclose(GOAL_CLEAR, 0.39)
    assert np.isclose(MIN_PAIR, 0.34)
    assert np.allclose(NE.BETWEEN_OFFSET_RANGE, (0.45, 0.65))
    assert np.allclose(NE.BETWEEN_LATERAL_RANGE, (-0.10, 0.10))
    assert np.isclose(NE.VAL_OBSTACLE_REGION, NE.OBSTACLE_REGION)
    assert np.isclose(NE.VAL_OBSTACLE_REGION, 0.5)


def test_sampled_scenes_pass_shared_generator_checks():
    for seed in range(50):
        first = sample_validation_scene(seed)
        second = sample_validation_scene(seed)
        assert first == second, f"seed={seed} is not deterministic"
        sx, sy = first["start"]
        assert NE.START_X_RANGE[0] <= sx <= NE.START_X_RANGE[1]
        assert NE.START_Y_RANGE[0] <= sy <= NE.START_Y_RANGE[1]
        assert sx not in NE.START_X_RANGE and sy not in NE.START_Y_RANGE
        distances = np.linalg.norm(
            np.asarray(first["pillars"], dtype=float)
            - np.asarray(first["start"], dtype=float),
            axis=1,
        )
        assert np.all(distances >= START_CLEAR - 1e-9)
        ok, reason = validate_pillars(
            first["pillars"], first["start"], first["goal"], num=2,
        )
        assert ok, f"seed={seed}: {reason}"


def main():
    test_episode_return_is_model_selection_score()
    test_between_start_is_not_clipped()
    test_validation_geometry_matches_training_contract()
    test_sampled_scenes_pass_shared_generator_checks()
    print("ALL PASS")


if __name__ == "__main__":
    main()
