import json

from DIVO.curriculum.pattern_discovery import (
    PatternDiscoveryConfig,
    discover_failure_graph_patterns,
)


def _layout_z(axis_length, blockage_max, episode_id):
    return {
        "version": "layout_z_raw_v1",
        "axis": {"length": axis_length, "angle": 0.0},
        "start": {"x": 0.0, "y": 0.0, "theta": 0.0},
        "obstacle_axis_edges": [],
        "obstacle_pair_edges": [],
        "summary": {
            "axis_length": axis_length,
            "axis_angle": 0.0,
            "num_obstacles": 2,
            "num_pairs": 1,
            "blockage_max": blockage_max,
            "pair_distance_min": 1.0 - blockage_max,
        },
        "episode_id": episode_id,
    }


def test_discover_failure_graph_patterns_finds_data_driven_conjunction():
    records = []
    for idx in range(400):
        axis_length = idx / 399.0
        blockage_max = ((idx * 37) % 400) / 399.0
        failed = axis_length > 0.75 and blockage_max > 0.75
        records.append(
            {
                "episode_id": idx,
                "layout_z": _layout_z(axis_length, blockage_max, idx),
                "termination": "collision" if failed else "success",
                "failure_key": "collision_tblock_early" if failed else "success",
            }
        )

    result = discover_failure_graph_patterns(
        records,
        PatternDiscoveryConfig(
            num_bins=4,
            max_conditions=3,
            min_support=10,
            min_failure_count=5,
            top_k=10,
            beam_width=100,
            min_lift=1.0,
        ),
    )

    assert result["global_stats"]["failure_count"] > 0
    assert result["top_patterns"]
    assert result["top_patterns"][0]["failure_lift"] > 1.0
    assert any(
        (
            "summary.axis_length" in " ".join(pattern["conditions"])
            or "axis.length" in " ".join(pattern["conditions"])
        )
        and "summary.blockage_max" in " ".join(pattern["conditions"])
        for pattern in result["top_patterns"]
    )
    json.dumps(result)


def test_discover_failure_graph_patterns_handles_empty_records():
    result = discover_failure_graph_patterns(
        [],
        PatternDiscoveryConfig(min_support=1, min_failure_count=1),
    )
    assert result["global_stats"]["num_episodes"] == 0
    assert result["top_patterns"] == []
