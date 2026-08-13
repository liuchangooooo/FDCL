import pytest

from DIVO.curriculum.attribution import (
    AttributionConfig,
    bin_value,
    compute_attribution,
)
from analysis.failure_attribution.build_obstacle_attribution import (
    build_attribution as legacy_build_attribution,
)


def _z(alpha, beta, blockage, d_start=0.4, d_goal=0.6):
    return {
        "alpha": alpha,
        "beta": beta,
        "blockage": blockage,
        "d_start": d_start,
        "d_goal": d_goal,
    }


def _record(termination, zs, failure_key=None):
    return {
        "termination": termination,
        "failure_key": failure_key or termination,
        "obstacle_z": zs,
    }


def _cell_map(result):
    return {cell.cell_id: cell for cell in result.cells}


def test_attribution_config_defaults_and_from_dict():
    config = AttributionConfig.from_dict({})
    assert config.min_support == 10
    assert config.top_k == 10
    assert config.low_k == 5

    config = AttributionConfig.from_dict({"min_support": 2, "top_k": 3})
    assert config.min_support == 2
    assert config.top_k == 3


def test_bin_boundaries_preserve_existing_semantics():
    config = AttributionConfig()
    assert bin_value(-0.1, config.alpha_bins) == "before_start"
    assert bin_value(0.0, config.alpha_bins) == "start_to_025"
    assert bin_value(0.5, config.alpha_bins) == "050_to_075"
    assert bin_value(1.0, config.alpha_bins) == "after_goal"
    assert bin_value(1.2, config.alpha_bins) == "after_goal"


def test_beta_abs_binning_and_side_counts():
    records = [
        _record("success", [_z(0.4, 0.2, 0.5)]),
        _record("collision", [_z(0.4, -0.2, 0.5)], "collision_rod_mid"),
    ]
    result = compute_attribution(records, AttributionConfig(min_support=1))
    cell = result.cells[0]
    assert cell.beta_abs_bin == "near_side"
    assert cell.beta_left_count == 1
    assert cell.beta_right_count == 1
    assert cell.beta_center_count == 0


def test_failure_lift_formula_and_top_low_cells():
    records = [
        _record("collision", [_z(0.3, 0.05, 0.8)], "collision_rod_mid"),
        _record("collision", [_z(0.3, 0.05, 0.8)], "collision_rod_mid"),
        _record("success", [_z(0.3, 0.05, 0.8)]),
        _record("success", [_z(0.8, 0.3, 0.1)]),
    ]
    result = compute_attribution(records, AttributionConfig(min_support=1, top_k=2, low_k=1))
    cells = _cell_map(result)
    high = cells["alpha=025_to_050|beta_abs=centerline|blockage=high"]
    low = cells["alpha=075_to_goal|beta_abs=far_side|blockage=low"]

    assert high.failure_count == 2
    assert high.total_count == 3
    assert high.p_cell == pytest.approx(3 / 4)
    assert high.p_cell_given_failure == pytest.approx(1.0)
    assert high.failure_lift == pytest.approx((1.0) / (3 / 4))
    assert result.top_cells[0].cell_id == high.cell_id
    assert result.low_cells[0].cell_id == low.cell_id


def test_coverage_stats_and_most_sampled_cells():
    records = [
        _record("collision", [_z(0.0, 0.0, 1.0)], "collision_tblock_early"),
        _record("success", [_z(0.5, 0.2, 0.5)]),
        _record("success", [_z(1.0, -0.4, 0.0)]),
    ]
    result = compute_attribution(records, AttributionConfig(min_support=1, top_k=3))
    coverage = result.coverage
    assert coverage.num_obstacle_samples == 3
    assert coverage.occupied_cells == 3
    assert coverage.supported_cells == 3
    assert coverage.alpha_stats["mean"] == pytest.approx(0.5)
    assert coverage.beta_abs_stats["mean"] == pytest.approx(0.2)
    assert coverage.blockage_stats["mean"] == pytest.approx(0.5)
    assert len(coverage.bin_proportions) == 3


def test_empty_records_return_empty_result():
    result = compute_attribution([], AttributionConfig(min_support=1))
    assert result.global_counts["num_episodes"] == 0
    assert result.global_counts["num_obstacle_samples"] == 0
    assert result.global_counts["is_empty"] is True
    assert result.cells == []
    assert result.top_cells == []
    assert result.coverage.alpha_stats["mean"] == 0.0


def test_all_success_keeps_legacy_top_cells_with_zero_lift():
    records = [
        _record("success", [_z(0.3, 0.05, 0.8)]),
        _record("success", [_z(0.3, 0.05, 0.8)]),
    ]
    result = compute_attribution(records, AttributionConfig(min_support=1))
    assert result.global_counts["has_any_failure"] is False
    assert result.top_cells
    assert result.top_cells[0].failure_lift == 0.0


def test_multi_obstacle_episode_counts_each_obstacle_sample():
    records = [
        _record(
            "collision",
            [_z(0.3, 0.05, 0.8), _z(0.8, -0.3, 0.1)],
            "collision_rod_mid",
        )
    ]
    result = compute_attribution(records, AttributionConfig(min_support=1))
    assert result.global_counts["num_episodes"] == 1
    assert result.global_counts["num_obstacle_samples"] == 2
    assert result.global_counts["num_failure_samples"] == 2
    assert len(result.cells) == 2


def test_to_dict_matches_offline_wrapper_on_core_fields():
    records = [
        _record("collision", [_z(0.3, 0.05, 0.8)], "collision_rod_mid"),
        _record("success", [_z(0.3, 0.05, 0.8)]),
        _record("timeout", [_z(0.8, 0.3, 0.1)], "timeout"),
    ]
    new = compute_attribution(records, AttributionConfig(min_support=1)).to_dict()
    old = legacy_build_attribution(records, min_support=1)
    new_cells = {row["cell_id"]: row for row in new["cells"]}
    old_cells = {row["cell_id"]: row for row in old["cells"]}

    assert set(new_cells) == set(old_cells)
    for cell_id in new_cells:
        new_cell = new_cells[cell_id]
        old_cell = old_cells[cell_id]
        for key in (
            "alpha_bin",
            "beta_abs_bin",
            "blockage_bin",
            "total_count",
            "success_count",
            "failure_count",
            "dominant_failure_key",
        ):
            assert new_cell[key] == old_cell[key]
        assert new_cell["failure_rate"] == pytest.approx(old_cell["failure_rate"])
        assert new_cell["p_cell"] == pytest.approx(old_cell["p_cell"])
        assert new_cell["p_cell_given_failure"] == pytest.approx(old_cell["p_cell_given_failure"])
        assert new_cell["failure_lift"] == pytest.approx(old_cell["failure_lift"])

    assert [row["cell_id"] for row in new["top_cells"]] == [
        row["cell_id"] for row in old["top_cells"]
    ]
