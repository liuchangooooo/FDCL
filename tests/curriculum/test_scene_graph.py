import json

from DIVO.curriculum.adapters.pusht_adapter import pusht_build_scene_graph


def _obstacles(count):
    base = [
        {"x": 0.10, "y": 0.02, "purpose": "first"},
        {"x": 0.04, "y": -0.05, "purpose": "second"},
        {"x": -0.03, "y": 0.08, "purpose": "third"},
    ]
    return base[:count]


def test_pusht_scene_graph_supports_variable_obstacle_counts():
    for count in (1, 2, 3):
        graph = pusht_build_scene_graph(
            _obstacles(count),
            tblock_pose=[0.15, -0.10, 0.2],
            target_pose=[0.0, 0.0, -0.785],
        )

        assert graph["version"] == "task_axis_scene_graph_v1"
        assert graph["task"] == "PushT"
        assert len(graph["nodes"]) == 3 + count
        assert graph["global_features"]["num_obstacles"] == count

        pair_edges = [edge for edge in graph["edges"] if edge["type"] == "obstacle_pair"]
        assert len(pair_edges) == count * (count - 1) // 2

        axis_edges = [edge for edge in graph["edges"] if edge["type"] == "obstacle_task_axis"]
        assert len(axis_edges) == count
        for edge in axis_edges:
            features = edge["features"]
            assert "alpha" in features
            assert "beta" in features
            assert "blockage" in features

        json.dumps(graph)


def test_pusht_scene_graph_uses_task_axis_node_name():
    graph = pusht_build_scene_graph(
        _obstacles(2),
        tblock_pose=[0.15, -0.10, 0.2],
        target_pose=[0.0, 0.0, -0.785],
    )

    node_ids = {node["id"] for node in graph["nodes"]}
    assert "task_axis" in node_ids
    assert "path" not in node_ids
