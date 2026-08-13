import json
import math

from DIVO.curriculum.adapters.pusht_adapter import pusht_build_scene_graph
from DIVO.curriculum.layout_features import extract_layout_z


def _obstacles():
    return [
        {"x": 0.10, "y": 0.02, "purpose": "first"},
        {"x": 0.04, "y": -0.05, "purpose": "second"},
        {"x": -0.03, "y": 0.08, "purpose": "third"},
    ]


def test_extract_layout_z_returns_raw_graph_features_only():
    graph = pusht_build_scene_graph(
        _obstacles()[:2],
        tblock_pose=[0.15, -0.10, 0.2],
        target_pose=[0.0, 0.0, -0.785],
    )
    layout_z = extract_layout_z(graph)

    assert layout_z["version"] == "layout_z_raw_v1"
    assert layout_z["source"]["type"] == "scene_graph"
    assert "bins" not in layout_z
    assert "motifs" not in layout_z
    assert len(layout_z["obstacle_axis_edges"]) == 2
    assert len(layout_z["obstacle_pair_edges"]) == 1
    assert layout_z["summary"]["num_obstacles"] == 2
    assert layout_z["summary"]["num_pairs"] == 1
    json.dumps(layout_z)


def test_extract_layout_z_supports_variable_obstacle_counts():
    for count in (1, 2, 3):
        graph = pusht_build_scene_graph(
            _obstacles()[:count],
            tblock_pose=[0.15, -0.10, 0.2],
            target_pose=[0.0, 0.0, -0.785],
        )
        layout_z = extract_layout_z(graph)

        assert len(layout_z["obstacle_axis_edges"]) == count
        assert len(layout_z["obstacle_pair_edges"]) == count * (count - 1) // 2
        assert layout_z["summary"]["num_obstacles"] == count


def test_extract_layout_z_summary_is_permutation_invariant():
    graph_a = pusht_build_scene_graph(
        _obstacles(),
        tblock_pose=[0.15, -0.10, 0.2],
        target_pose=[0.0, 0.0, -0.785],
    )
    graph_b = pusht_build_scene_graph(
        list(reversed(_obstacles())),
        tblock_pose=[0.15, -0.10, 0.2],
        target_pose=[0.0, 0.0, -0.785],
    )

    summary_a = extract_layout_z(graph_a)["summary"]
    summary_b = extract_layout_z(graph_b)["summary"]

    for key, value_a in summary_a.items():
        value_b = summary_b[key]
        if value_a is None:
            assert value_b is None
        else:
            assert math.isclose(float(value_a), float(value_b), rel_tol=1e-12, abs_tol=1e-12)
