"""Nav B/M/U/D v3 协议回归（safenav 下运行 ``python -m nav.test_benchmarks``）。"""
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

from nav import benchmarks as B
from nav import nav_env as NE
from nav.nav_adapter import NavEnvAdapter
from safety_gymnasium.utils.common_utils import ResamplingError


def _min_pairwise(points):
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        return np.inf
    dist = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    dist[np.eye(len(points), dtype=bool)] = np.inf
    return float(dist.min())


def test_sampling_contract():
    expected_count = {"B": 1, "M": 3, "U": 7, "D": 3}
    layouts = {family: set() for family in expected_count}
    for seed in range(128):
        training = NE.sample_training_layout(np.random.default_rng(seed))
        assert len(training) == NE.TRAIN_NUM_PILLARS == 2
        assert _min_pairwise(training) >= NE.PILLAR_MIN_SEPARATION - 1e-9
        for family, count in expected_count.items():
            scene = B.sample_benchmark_scene(family, seed)
            assert scene == B.sample_benchmark_scene(family, seed), "同 seed 场景必须确定"
            assert len(scene["pillars"]) == count
            assert isinstance(scene["reset_seed"], int)
            assert scene["start"] == NE.START
            layouts[family].add(tuple(scene["pillars"]))
            travel = scene["travel"] if scene["dynamic"] else 0.0
            start_clear = B._start_clearance(scene["size"], scene["keepout"], travel)
            goal_clear = B._goal_clearance(scene["size"], scene["keepout"])
            for x, y in scene["pillars"]:
                assert np.hypot(x - NE.START[0], y - NE.START[1]) >= start_clear - 1e-9
                assert np.hypot(x - NE.GOAL[0], y - NE.GOAL[1]) >= goal_clear - 1e-9

        multiple = B.sample_benchmark_scene("M", seed)
        assert multiple["size"] == NE.PILLAR_SIZE
        assert multiple["keepout"] == NE.PILLAR_KEEPOUT
        assert _min_pairwise(multiple["pillars"]) >= NE.PILLAR_MIN_SEPARATION - 1e-9
        dynamic = B.sample_benchmark_scene("D", seed)
        assert dynamic["dynamic"] is True
        assert dynamic["travel"] == NE.GREMLIN_TRAVEL

    assert all(len(family_layouts) > 1 for family_layouts in layouts.values())
    print("  sampling contract OK (fixed start, random collision-free obstacles)")


class _AlwaysInvalidAdapter:
    def __init__(self, dynamic):
        self.dynamic = bool(dynamic)

    def set_layout(self, *args, **kwargs):
        return None

    def reset(self, *args, **kwargs):
        raise ResamplingError("intentional invalid scene")


def test_fail_fast_and_incomplete():
    zero = lambda obs: np.zeros(2)
    try:
        B.evaluate_benchmark(_AlwaysInvalidAdapter(True), zero, "B", n_env=1)
    except ValueError:
        pass
    else:
        raise AssertionError("B must reject a dynamic adapter")

    try:
        B.evaluate_benchmark(_AlwaysInvalidAdapter(False), zero, "D", n_env=1)
    except ValueError:
        pass
    else:
        raise AssertionError("D must reject a static adapter")

    try:
        B.evaluate_bmud(
            _AlwaysInvalidAdapter(False), zero, n_env=1, max_steps=1,
            dyn_adapter=_AlwaysInvalidAdapter(True),
        )
    except RuntimeError as exc:
        assert "incomplete" in str(exc)
    else:
        raise AssertionError("incomplete formal evaluation must not return a score")
    print("  adapter-mode and incomplete-evaluation guards OK")


def test_environment_smoke():
    static = NavEnvAdapter()
    dynamic = NavEnvAdapter(dynamic=True)
    try:
        result = B.evaluate_bmud(
            static, lambda obs: np.zeros(2), n_env=1, max_steps=2,
            dyn_adapter=dynamic, base_seed=31415,
        )
    finally:
        static.close()
        dynamic.close()

    expected = {"B", "M", "U", "D", "D_static", "D_dynamic", "dynamic_drop", "AVG"}
    assert set(result) == expected
    assert all(type(value) is float and np.isfinite(value) for value in result.values())
    assert result["D"] == result["D_dynamic"]
    assert np.isclose(result["dynamic_drop"], result["D_static"] - result["D_dynamic"])
    assert np.isclose(result["AVG"], np.mean([result[k] for k in ("B", "M", "U", "D")]))
    print(f"  environment smoke OK: {result}")


def main():
    test_sampling_contract()
    test_fail_fast_and_incomplete()
    test_environment_smoke()
    print("ALL PASS")


if __name__ == "__main__":
    main()
