# Phase 1 Layout Attribution Cross-Run Summary

This report compares single-obstacle attribution against layout-level attribution. Raw lift should be read together with global failure rate, because an all-failure pattern has lift = 1 / global_failure_rate.

Informative coverage threshold: lift >= 2.00, pattern failure_rate >= 0.50.

## Numeric Summary

| run | episodes | global fail | single best | layout best | pressure coverage | basic coverage |
|---|---:|---:|---:|---:|---:|---:|
| evolve_attribution_shared_seed42_old | 320934 | 0.118 | 0.971/8.20 (273) | 1.000/8.44 (215) | 0.018 | 0.015 |
| evolve_attribution_verifier_shared_seed42 | 339265 | 0.168 | 0.830/4.92 (698) | 1.000/5.94 (2180) | 0.063 | 0.082 |
| evolve_coarse_shared_seed42 | 338174 | 0.137 | 0.850/6.23 (780) | 1.000/7.32 (192) | 0.036 | 0.040 |
| static_llm_shared_seed42 | 322072 | 0.084 | 0.752/9.00 (254) | 0.792/9.47 (221) | 0.001 | 0.000 |

The `single best` and `layout best` columns are formatted as `pattern_failure_rate / lift (support)`.

## Top Layout Patterns

### evolve_attribution_shared_seed42_old

- rollouts: `/home/hnu-w/DIVO/data/outputs/2026.05.30/14.34.48_td3_pusht_llm_curriculum/obstacle_rollouts.jsonl`
- single-obstacle best: `alpha=025_to_050|beta_abs=centerline|blockage=high`
- layout best family: `layout_pressure`
- layout best dominant failure: `collision_tblock_early`
- layout best pattern: `path_len_bin=long|max_blockage_bin=medium|medhigh_count_bin=2|near_count_bin=2|center_count_bin=0|min_pair_dist_bin=close|pressure_bin=high`

### evolve_attribution_verifier_shared_seed42

- rollouts: `/home/hnu-w/DIVO/data/outputs/2026.06.01/21.12.29_td3_pusht_llm_curriculum/obstacle_rollouts.jsonl`
- single-obstacle best: `alpha=050_to_075|beta_abs=centerline|blockage=high`
- layout best family: `layout_basic`
- layout best dominant failure: `collision_tblock_early`
- layout best pattern: `start_region=x_pos__y_pos|path_len_bin=long|num_obstacles_bin=2|max_blockage_bin=medium|medhigh_count_bin=2|pair_side_mode=opposite_side|min_pair_dist_bin=close`

### evolve_coarse_shared_seed42

- rollouts: `/home/hnu-w/DIVO/data/outputs/2026.05.30/14.34.30_td3_pusht_llm_curriculum/obstacle_rollouts.jsonl`
- single-obstacle best: `alpha=050_to_075|beta_abs=centerline|blockage=high`
- layout best family: `layout_basic`
- layout best dominant failure: `collision_tblock_early`
- layout best pattern: `start_region=x_pos__y_neg|path_len_bin=long|num_obstacles_bin=2|max_blockage_bin=medium|medhigh_count_bin=2|pair_side_mode=opposite_side|min_pair_dist_bin=close`

### static_llm_shared_seed42

- rollouts: `/home/hnu-w/DIVO/data/outputs/2026.05.29/22.26.51_td3_pusht_llm_curriculum/obstacle_rollouts.jsonl`
- single-obstacle best: `alpha=025_to_050|beta_abs=near_side|blockage=medium`
- layout best family: `layout_pressure`
- layout best dominant failure: `collision_tblock_early`
- layout best pattern: `path_len_bin=long|max_blockage_bin=medium|medhigh_count_bin=1|near_count_bin=1|center_count_bin=0|min_pair_dist_bin=far|pressure_bin=medium`
