import math

import numpy as np
import pytest

from DIVO.ant.obstacles import ObstacleSpec
from DIVO.curriculum.adapters.ant_adapter import (
    ant_decode_obstacle,
    ant_encode_obstacle,
)
from DIVO.curriculum.adapters.pusht_adapter import (
    pusht_decode_obstacle,
    pusht_encode_obstacle,
)
from DIVO.curriculum.obstacle_geometry import (
    decode_z_to_xy,
    encode_xy_to_z,
    format_z_for_prompt,
)


def assert_xy_close(a, b, tol=1e-9):
    assert np.linalg.norm(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)) < tol


def test_roundtrip_encode_decode():
    start = np.array([0.2, -0.1])
    goal = np.array([-0.3, 0.4])
    obstacle = np.array([0.05, 0.08])
    z = encode_xy_to_z(obstacle[0], obstacle[1], 0.02, start, goal, 0.05)
    decoded = decode_z_to_xy(z.alpha, z.beta, start, goal)
    assert_xy_close(decoded, obstacle)


def test_alpha_at_start():
    z = encode_xy_to_z(0.0, 0.0, 0.02, [0.0, 0.0], [1.0, 0.0], 0.05)
    assert z.alpha == pytest.approx(0.0)
    assert z.beta == pytest.approx(0.0)


def test_alpha_at_goal():
    z = encode_xy_to_z(1.0, 0.0, 0.02, [0.0, 0.0], [1.0, 0.0], 0.05)
    assert z.alpha == pytest.approx(1.0)
    assert z.beta == pytest.approx(0.0)


def test_alpha_at_midpoint():
    z = encode_xy_to_z(0.5, 0.0, 0.02, [0.0, 0.0], [1.0, 0.0], 0.05)
    assert z.alpha == pytest.approx(0.5)
    assert z.beta == pytest.approx(0.0)


def test_beta_sign_left():
    z = encode_xy_to_z(0.5, 0.2, 0.02, [0.0, 0.0], [1.0, 0.0], 0.05)
    assert z.beta > 0.0


def test_beta_sign_right():
    z = encode_xy_to_z(0.5, -0.2, 0.02, [0.0, 0.0], [1.0, 0.0], 0.05)
    assert z.beta < 0.0


def test_blockage_on_path():
    z = encode_xy_to_z(0.5, 0.0, 0.02, [0.0, 0.0], [1.0, 0.0], 0.05)
    assert z.blockage == pytest.approx(1.0)


def test_blockage_zero_at_radius_plus_corridor_width():
    radius = 0.02
    corridor_width = 0.05
    z = encode_xy_to_z(
        0.5,
        radius + corridor_width,
        radius,
        [0.0, 0.0],
        [1.0, 0.0],
        corridor_width,
    )
    assert z.blockage == pytest.approx(0.0)


def test_blockage_uses_segment_not_infinite_line():
    z = encode_xy_to_z(-1.0, 0.0, 0.02, [0.0, 0.0], [1.0, 0.0], 0.05)
    assert z.alpha < 0.0
    assert z.beta == pytest.approx(0.0)
    assert z.blockage == pytest.approx(0.0)


def test_alpha_outside_unit_interval_not_clipped():
    z = encode_xy_to_z(1.5, 0.0, 0.02, [0.0, 0.0], [1.0, 0.0], 0.05)
    assert z.alpha == pytest.approx(1.5)


def test_pusht_adapter_roundtrip():
    tblock_pose = np.array([0.15, 0.15, math.pi / 4])
    obstacle = {"x": 0.05, "y": 0.02, "purpose": "test"}
    z = pusht_encode_obstacle(obstacle, tblock_pose)
    decoded = pusht_decode_obstacle(z, tblock_pose, purpose="decoded")
    assert decoded["purpose"] == "decoded"
    assert_xy_close((decoded["x"], decoded["y"]), (obstacle["x"], obstacle["y"]))


def test_ant_adapter_circle_roundtrip():
    spec = ObstacleSpec(shape="circle", center=(0.2, 0.1), radius=0.3)
    z = ant_encode_obstacle(spec, start_xy=[-1.0, 0.0], goal_xy=[1.0, 0.0], corridor_width=0.6)
    decoded = ant_decode_obstacle(z, start_xy=[-1.0, 0.0], goal_xy=[1.0, 0.0])
    assert decoded.shape == "circle"
    assert decoded.radius == pytest.approx(0.3)
    assert_xy_close(decoded.center, spec.center)


def test_ant_adapter_box_does_not_crash():
    spec = ObstacleSpec(shape="box", center=(0.2, 0.1), half_size=(0.2, 0.1), angle=0.5)
    z = ant_encode_obstacle(spec, start_xy=[-1.0, 0.0], goal_xy=[1.0, 0.0], corridor_width=0.6)
    decoded = ant_decode_obstacle(
        z,
        start_xy=[-1.0, 0.0],
        goal_xy=[1.0, 0.0],
        shape="box",
    )
    assert decoded.shape == "box"
    assert decoded.half_size is not None
    assert_xy_close(decoded.center, spec.center)


def test_degenerate_zero_distance_does_not_crash():
    z = encode_xy_to_z(0.1, 0.2, 0.02, [0.0, 0.0], [0.0, 0.0], 0.05)
    decoded = decode_z_to_xy(z.alpha, z.beta, [0.0, 0.0], [0.0, 0.0])
    assert len(decoded) == 2


def test_format_z_for_prompt_ascii():
    z = encode_xy_to_z(0.5, 0.0, 0.02, [0.0, 0.0], [1.0, 0.0], 0.05)
    text = format_z_for_prompt(z)
    assert "alpha=" in text
    assert "beta=" in text
    assert "blockage=" in text
    assert "alpha" in text.encode("ascii").decode("ascii")
