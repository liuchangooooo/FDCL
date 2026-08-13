# Attribution Distribution and Final Generator Summary

This report summarizes all eight evolution rounds for the old attribution-only run and the new attribution+verifier run. Old-run candidate audits are re-computed offline with 100 pose samples, so their score deltas are for inspection rather than final statistical estimates. Near-zero old-run deltas, such as round 7, should be treated as borderline rather than stable accept/reject evidence.

Runs:
- old: `/home/hnu-w/DIVO/data/outputs/2026.05.30/14.34.48_td3_pusht_llm_curriculum`
- new: `/home/hnu-w/DIVO/data/outputs/2026.06.01/21.12.29_td3_pusht_llm_curriculum`

Key columns:
- `rollout ...`: actual distribution in the training batch that produced the attribution map.
- `final ...`: distribution sampled from the finally accepted candidate generator under the same attribution map.
- `score_delta`: candidate evidence score minus current score. Positive is desired for `increase`/too_easy; negative is desired for `decrease`/too_hard; non-negative is desired for `preserve`/balanced.

## Compact Table

| run | round | ep | sr | direction | score_delta | accepted / would accept | reason | rollout far+low | final far+low | rollout med/high | final med/high | length |
|---|---:|---:|---:|---|---:|---|---|---:|---:|---:|---:|---:|
| old_no_verifier | 1 | 10000 | 0.079 | decrease | -0.0842 | actual yes; verifier yes | candidate_decreased_evidence_score | 95.3% | 79.5% | 0.1% | 0.0% | 6347->9860 (+3513) |
| old_no_verifier | 2 | 20000 | 0.752 | preserve | +0.1310 | actual yes; verifier yes | candidate_preserved_evidence_score | 75.5% | 85.0% | 0.0% | 0.0% | 9860->13767 (+3907) |
| old_no_verifier | 3 | 30000 | 0.868 | increase | +0.0264 | actual yes; verifier yes | candidate_increased_evidence_score | 83.5% | 91.0% | 0.0% | 0.0% | 13767->13512 (-255) |
| old_no_verifier | 4 | 40000 | 0.877 | increase | -0.0400 | actual yes; verifier no | candidate_did_not_increase_evidence_score | 92.7% | 93.0% | 0.1% | 1.5% | 13512->16051 (+2539) |
| old_no_verifier | 5 | 50000 | 0.845 | increase | +0.0195 | actual yes; verifier yes | candidate_increased_evidence_score | 92.9% | 91.5% | 2.5% | 5.0% | 16051->16003 (-48) |
| old_no_verifier | 6 | 60000 | 0.724 | preserve | -0.0358 | actual yes; verifier no | candidate_reduced_evidence_score | 87.9% | 94.5% | 7.4% | 2.5% | 16003->17231 (+1228) |
| old_no_verifier | 7 | 70000 | 0.869 | increase | -0.0044 | actual yes; verifier no | candidate_did_not_increase_evidence_score | 94.7% | 90.0% | 2.2% | 3.5% | 17231->16717 (-514) |
| old_no_verifier | 8 | 80000 | 0.785 | preserve | -0.0250 | actual yes; verifier no | candidate_reduced_evidence_score | 87.9% | 93.5% | 5.6% | 1.0% | 16717->17769 (+1052) |
| new_with_verifier | 1 | 10000 | 0.079 | decrease | -0.1003 | yes | candidate_decreased_evidence_score | 95.3% | 74.7% | 0.1% | 0.0% | 6347->8299 (+1952) |
| new_with_verifier | 2 | 20000 | 0.769 | preserve | +0.1668 | yes | candidate_preserved_evidence_score | 75.7% | 95.8% | 0.0% | 0.0% | 8299->11434 (+3135) |
| new_with_verifier | 3 | 30000 | 0.886 | increase | +0.2000 | yes | candidate_increased_evidence_score | 95.5% | 99.0% | 0.0% | 0.0% | 11434->13337 (+1903) |
| new_with_verifier | 4 | 40000 | 0.845 | increase | +0.1610 | yes | candidate_increased_evidence_score | 99.4% | 99.8% | 0.0% | 0.0% | 13337->13185 (-152) |
| new_with_verifier | 5 | 50000 | 0.864 | increase | +0.0206 | yes (2 attempts) | candidate_increased_evidence_score | 99.6% | 97.4% | 0.0% | 0.1% | 13185->17082 (+3897) |
| new_with_verifier | 6 | 60000 | 0.872 | increase | +0.0265 | yes | candidate_increased_evidence_score | 97.6% | 90.2% | 0.2% | 5.0% | 17082->17654 (+572) |
| new_with_verifier | 7 | 70000 | 0.806 | increase | +0.0368 | yes | candidate_increased_evidence_score | 91.7% | 86.4% | 3.8% | 8.0% | 17654->17821 (+167) |
| new_with_verifier | 8 | 80000 | 0.783 | preserve | -0.0158 | no (3 attempts) | candidate_reduced_evidence_score | 84.7% | 90.1% | 10.1% | 7.0% | 17821->17821 (+0) |
| new_with_verifier | 8 | 90000 | 0.794 | preserve | +0.0109 | yes (2 attempts) | candidate_preserved_evidence_score | 85.0% | 84.9% | 9.7% | 11.4% | 17821->15959 (-1862) |

## New Run Attempt-Level Verifier Results

| round | ep | attempt | accepted | direction | score_delta | reason | top candidate cells |
|---:|---:|---:|---|---|---:|---|---|
| 1 | 10000 | 1 | True | decrease | -0.1003 | candidate_decreased_evidence_score | alpha=after_goal|beta_abs=far_side|blockage=low (cov=43.5%, n=435, ev=0.00); alpha=after_goal|beta_abs=near_side|blockage=low (cov=15.2%, n=152, ev=0.00); alpha=after_goal|beta_abs=centerline|blockage=low (cov=10.1%, n=101, ev=0.00) |
| 2 | 20000 | 1 | True | preserve | +0.1668 | candidate_preserved_evidence_score | alpha=after_goal|beta_abs=far_side|blockage=low (cov=33.4%, n=334, ev=0.00); alpha=050_to_075|beta_abs=far_side|blockage=low (cov=19.5%, n=195, ev=0.60); alpha=025_to_050|beta_abs=far_side|blockage=low (cov=17.0%, n=170, ev=0.40) |
| 3 | 30000 | 1 | True | increase | +0.2000 | candidate_increased_evidence_score | alpha=025_to_050|beta_abs=far_side|blockage=low (cov=39.1%, n=391, ev=0.67); alpha=050_to_075|beta_abs=far_side|blockage=low (cov=33.7%, n=337, ev=0.33); alpha=after_goal|beta_abs=far_side|blockage=low (cov=9.8%, n=98, ev=0.00) |
| 4 | 40000 | 1 | True | increase | +0.1610 | candidate_increased_evidence_score | alpha=050_to_075|beta_abs=far_side|blockage=low (cov=45.4%, n=454, ev=1.00); alpha=025_to_050|beta_abs=far_side|blockage=low (cov=44.1%, n=441, ev=0.50); alpha=075_to_goal|beta_abs=far_side|blockage=low (cov=3.8%, n=38, ev=0.00) |
| 5 | 50000 | 1 | False | increase | -0.0240 | candidate_did_not_increase_evidence_score | alpha=050_to_075|beta_abs=far_side|blockage=low (cov=37.3%, n=373, ev=0.40); alpha=025_to_050|beta_abs=far_side|blockage=low (cov=35.9%, n=359, ev=0.20); alpha=after_goal|beta_abs=far_side|blockage=low (cov=10.7%, n=107, ev=0.00) |
| 5 | 50000 | 2 | True | increase | +0.0206 | candidate_increased_evidence_score | alpha=050_to_075|beta_abs=far_side|blockage=low (cov=48.8%, n=488, ev=0.40); alpha=025_to_050|beta_abs=far_side|blockage=low (cov=39.6%, n=396, ev=0.20); alpha=075_to_goal|beta_abs=far_side|blockage=low (cov=5.9%, n=59, ev=0.00) |
| 6 | 60000 | 1 | True | increase | +0.0265 | candidate_increased_evidence_score | alpha=025_to_050|beta_abs=far_side|blockage=low (cov=35.9%, n=358, ev=0.20); alpha=050_to_075|beta_abs=far_side|blockage=low (cov=32.2%, n=321, ev=0.00); alpha=after_goal|beta_abs=far_side|blockage=low (cov=13.8%, n=138, ev=0.00) |
| 7 | 70000 | 1 | True | increase | +0.0368 | candidate_increased_evidence_score | alpha=after_goal|beta_abs=far_side|blockage=low (cov=32.6%, n=322, ev=0.00); alpha=050_to_075|beta_abs=far_side|blockage=low (cov=21.7%, n=214, ev=0.00); alpha=025_to_050|beta_abs=far_side|blockage=low (cov=15.9%, n=157, ev=0.00) |
| 8 | 80000 | 1 | False | preserve | -0.0672 | candidate_reduced_evidence_score | alpha=025_to_050|beta_abs=far_side|blockage=low (cov=26.9%, n=260, ev=0.00); alpha=050_to_075|beta_abs=far_side|blockage=low (cov=26.5%, n=257, ev=0.00); alpha=start_to_025|beta_abs=far_side|blockage=low (cov=20.8%, n=201, ev=0.00) |
| 8 | 80000 | 2 | False | preserve | -0.0002 | candidate_reduced_evidence_score | alpha=after_goal|beta_abs=far_side|blockage=low (cov=33.3%, n=326, ev=0.00); alpha=025_to_050|beta_abs=far_side|blockage=low (cov=18.2%, n=178, ev=0.00); alpha=050_to_075|beta_abs=far_side|blockage=low (cov=14.4%, n=141, ev=0.00) |
| 8 | 80000 | 3 | False | preserve | -0.0158 | candidate_reduced_evidence_score | alpha=025_to_050|beta_abs=far_side|blockage=low (cov=29.0%, n=282, ev=0.00); alpha=050_to_075|beta_abs=far_side|blockage=low (cov=28.8%, n=280, ev=0.00); alpha=before_start|beta_abs=far_side|blockage=low (cov=11.6%, n=113, ev=0.00) |
| 8 | 90000 | 1 | False | preserve | -0.0765 | candidate_reduced_evidence_score | alpha=050_to_075|beta_abs=far_side|blockage=low (cov=27.5%, n=269, ev=0.00); alpha=025_to_050|beta_abs=far_side|blockage=low (cov=27.5%, n=269, ev=0.00); alpha=before_start|beta_abs=far_side|blockage=low (cov=13.9%, n=136, ev=0.00) |
| 8 | 90000 | 2 | True | preserve | +0.0109 | candidate_preserved_evidence_score | alpha=050_to_075|beta_abs=far_side|blockage=low (cov=24.1%, n=238, ev=0.00); alpha=025_to_050|beta_abs=far_side|blockage=low (cov=23.4%, n=231, ev=0.00); alpha=after_goal|beta_abs=far_side|blockage=low (cov=22.7%, n=224, ev=0.00) |

## Old Attribution-Only Run, Re-Audited

### Round 1 @ episode 10000

- batch: success_rate=0.079; success=786, collision=4297, timeout=4563, fall=354
- trigger: `fixed_schedule(evolve_index=1/8, eps=10000, target=10000)|difficulty=too_hard(sr=0.079<0.200)`
- audit: direction=decrease, accepted=True, reason=candidate_decreased_evidence_score, score=0.1667->0.0825 (-0.0842), attempts=1
- valid generations: current=100/100, candidate=100/100
- generator length: 6347 -> 9860 (+3513)
- rollout distribution: low=99.9%, far+low=95.3%, med/high=0.1%, near+med/high=0.1%, center+med/high=0.0%
- final generator sampled distribution: low=100.0%, far+low=79.5%, med/high=0.0%, near+med/high=0.0%, center+med/high=0.0%

Top failure-associated attribution cells:
- alpha=before_start|beta_abs=centerline|blockage=low (lift=1.085, n=13, fail=13, cov=0.07%, fr=100.0%)
- alpha=025_to_050|beta_abs=near_side|blockage=low (lift=1.085, n=11, fail=11, cov=0.06%, fr=100.0%)
- alpha=050_to_075|beta_abs=near_side|blockage=low (lift=1.085, n=11, fail=11, cov=0.06%, fr=100.0%)
- alpha=before_start|beta_abs=far_side|blockage=low (lift=1.013, n=674, fail=629, cov=3.37%, fr=93.3%)
- alpha=025_to_050|beta_abs=far_side|blockage=low (lift=1.005, n=5665, fail=5244, cov=28.32%, fr=92.6%)

Training rollout most-sampled cells before evolve:
- alpha=050_to_075|beta_abs=far_side|blockage=low (cov=30.8%, n=6151)
- alpha=025_to_050|beta_abs=far_side|blockage=low (cov=28.3%, n=5665)
- alpha=after_goal|beta_abs=far_side|blockage=low (cov=17.4%, n=3476)
- alpha=075_to_goal|beta_abs=far_side|blockage=low (cov=9.2%, n=1836)
- alpha=start_to_025|beta_abs=far_side|blockage=low (cov=6.3%, n=1265)

Final candidate generator most-sampled cells:
- alpha=after_goal|beta_abs=far_side|blockage=low (cov=41.0%, n=82, ev=0.00)
- alpha=050_to_075|beta_abs=far_side|blockage=low (cov=15.5%, n=31, ev=0.17)
- alpha=after_goal|beta_abs=near_side|blockage=low (cov=14.5%, n=29, ev=0.00)
- alpha=025_to_050|beta_abs=far_side|blockage=low (cov=9.5%, n=19, ev=0.33)
- alpha=after_goal|beta_abs=centerline|blockage=low (cov=6.0%, n=12, ev=0.00)

### Round 2 @ episode 20000

- batch: success_rate=0.752; success=7516, collision=346, timeout=2118, fall=20
- trigger: `fixed_schedule(evolve_index=2/8, eps=20000, target=20000)|difficulty=balanced(sr=0.752, range=[0.200,0.800])`
- audit: direction=preserve, accepted=True, reason=candidate_preserved_evidence_score, score=0.1580->0.2890 (+0.1310), attempts=1
- valid generations: current=100/100, candidate=100/100
- generator length: 9860 -> 13767 (+3907)
- rollout distribution: low=100.0%, far+low=75.5%, med/high=0.0%, near+med/high=0.0%, center+med/high=0.0%
- final generator sampled distribution: low=100.0%, far+low=85.0%, med/high=0.0%, near+med/high=0.0%, center+med/high=0.0%

Top failure-associated attribution cells:
- alpha=start_to_025|beta_abs=far_side|blockage=low (lift=1.208, n=733, fail=220, cov=3.67%, fr=30.0%)
- alpha=before_start|beta_abs=far_side|blockage=low (lift=1.168, n=931, fail=270, cov=4.66%, fr=29.0%)
- alpha=025_to_050|beta_abs=far_side|blockage=low (lift=1.144, n=1882, fail=535, cov=9.41%, fr=28.4%)
- alpha=075_to_goal|beta_abs=far_side|blockage=low (lift=1.107, n=1389, fail=382, cov=6.94%, fr=27.5%)
- alpha=050_to_075|beta_abs=far_side|blockage=low (lift=1.058, n=2302, fail=605, cov=11.51%, fr=26.3%)

Training rollout most-sampled cells before evolve:
- alpha=after_goal|beta_abs=far_side|blockage=low (cov=39.4%, n=7872)
- alpha=after_goal|beta_abs=near_side|blockage=low (cov=14.8%, n=2959)
- alpha=050_to_075|beta_abs=far_side|blockage=low (cov=11.5%, n=2302)
- alpha=after_goal|beta_abs=centerline|blockage=low (cov=9.6%, n=1914)
- alpha=025_to_050|beta_abs=far_side|blockage=low (cov=9.4%, n=1882)

Final candidate generator most-sampled cells:
- alpha=after_goal|beta_abs=far_side|blockage=low (cov=28.0%, n=56, ev=0.00)
- alpha=025_to_050|beta_abs=far_side|blockage=low (cov=19.5%, n=39, ev=0.60)
- alpha=050_to_075|beta_abs=far_side|blockage=low (cov=18.0%, n=36, ev=0.20)
- alpha=after_goal|beta_abs=near_side|blockage=low (cov=10.0%, n=20, ev=0.00)
- alpha=075_to_goal|beta_abs=far_side|blockage=low (cov=8.5%, n=17, ev=0.40)

### Round 3 @ episode 30000

- batch: success_rate=0.868; success=8683, collision=514, timeout=801, fall=2
- trigger: `fixed_schedule(evolve_index=3/8, eps=30000, target=30000)|difficulty=too_easy(sr=0.868>0.800)`
- audit: direction=increase, accepted=True, reason=candidate_increased_evidence_score, score=0.2443->0.2707 (+0.0264), attempts=1
- valid generations: current=100/100, candidate=100/100
- generator length: 13767 -> 13512 (-255)
- rollout distribution: low=100.0%, far+low=83.5%, med/high=0.0%, near+med/high=0.0%, center+med/high=0.0%
- final generator sampled distribution: low=100.0%, far+low=91.0%, med/high=0.0%, near+med/high=0.0%, center+med/high=0.0%

Top failure-associated attribution cells:
- alpha=before_start|beta_abs=centerline|blockage=low (lift=3.996, n=19, fail=10, cov=0.10%, fr=52.6%)
- alpha=before_start|beta_abs=near_side|blockage=low (lift=2.441, n=28, fail=9, cov=0.14%, fr=32.1%)
- alpha=before_start|beta_abs=far_side|blockage=low (lift=1.300, n=917, fail=157, cov=4.58%, fr=17.1%)
- alpha=025_to_050|beta_abs=far_side|blockage=low (lift=1.253, n=4122, fail=680, cov=20.61%, fr=16.5%)
- alpha=start_to_025|beta_abs=far_side|blockage=low (lift=1.176, n=1046, fail=162, cov=5.23%, fr=15.5%)

Training rollout most-sampled cells before evolve:
- alpha=after_goal|beta_abs=far_side|blockage=low (cov=30.6%, n=6127)
- alpha=025_to_050|beta_abs=far_side|blockage=low (cov=20.6%, n=4122)
- alpha=050_to_075|beta_abs=far_side|blockage=low (cov=16.1%, n=3227)
- alpha=after_goal|beta_abs=near_side|blockage=low (cov=8.7%, n=1734)
- alpha=after_goal|beta_abs=centerline|blockage=low (cov=7.6%, n=1519)

Final candidate generator most-sampled cells:
- alpha=050_to_075|beta_abs=far_side|blockage=low (cov=24.5%, n=49, ev=0.29)
- alpha=025_to_050|beta_abs=far_side|blockage=low (cov=23.5%, n=47, ev=0.57)
- alpha=after_goal|beta_abs=far_side|blockage=low (cov=20.5%, n=41, ev=0.00)
- alpha=075_to_goal|beta_abs=far_side|blockage=low (cov=14.0%, n=28, ev=0.14)
- alpha=after_goal|beta_abs=near_side|blockage=low (cov=6.0%, n=12, ev=0.00)

### Round 4 @ episode 40000

- batch: success_rate=0.877; success=8774, collision=868, timeout=357, fall=1
- trigger: `fixed_schedule(evolve_index=4/8, eps=40000, target=40000)|difficulty=too_easy(sr=0.877>0.800)`
- audit: direction=increase, accepted=False, reason=candidate_did_not_increase_evidence_score, score=0.1736->0.1336 (-0.0400), attempts=1
- valid generations: current=100/100, candidate=100/100
- generator length: 13512 -> 16051 (+2539)
- rollout distribution: low=99.9%, far+low=92.7%, med/high=0.1%, near+med/high=0.1%, center+med/high=0.0%
- final generator sampled distribution: low=98.5%, far+low=93.0%, med/high=1.5%, near+med/high=1.0%, center+med/high=0.0%

Top failure-associated attribution cells:
- alpha=025_to_050|beta_abs=near_side|blockage=medium (lift=6.010, n=19, fail=14, cov=0.10%, fr=73.7%)
- alpha=025_to_050|beta_abs=near_side|blockage=low (lift=3.670, n=20, fail=9, cov=0.10%, fr=45.0%)
- alpha=before_start|beta_abs=centerline|blockage=low (lift=3.444, n=90, fail=38, cov=0.45%, fr=42.2%)
- alpha=before_start|beta_abs=near_side|blockage=low (lift=3.131, n=99, fail=38, cov=0.50%, fr=38.4%)
- alpha=025_to_050|beta_abs=far_side|blockage=low (lift=1.220, n=5181, fail=775, cov=25.91%, fr=15.0%)

Training rollout most-sampled cells before evolve:
- alpha=025_to_050|beta_abs=far_side|blockage=low (cov=25.9%, n=5181)
- alpha=after_goal|beta_abs=far_side|blockage=low (cov=23.5%, n=4695)
- alpha=050_to_075|beta_abs=far_side|blockage=low (cov=22.1%, n=4418)
- alpha=075_to_goal|beta_abs=far_side|blockage=low (cov=12.9%, n=2575)
- alpha=start_to_025|beta_abs=far_side|blockage=low (cov=5.6%, n=1114)

Final candidate generator most-sampled cells:
- alpha=050_to_075|beta_abs=far_side|blockage=low (cov=33.0%, n=66, ev=0.14)
- alpha=after_goal|beta_abs=far_side|blockage=low (cov=23.0%, n=46, ev=0.00)
- alpha=075_to_goal|beta_abs=far_side|blockage=low (cov=17.0%, n=34, ev=0.00)
- alpha=025_to_050|beta_abs=far_side|blockage=low (cov=15.0%, n=30, ev=0.43)
- alpha=start_to_025|beta_abs=far_side|blockage=low (cov=3.0%, n=6, ev=0.00)

### Round 5 @ episode 50000

- batch: success_rate=0.845; success=8446, collision=1349, timeout=203, fall=2
- trigger: `fixed_schedule(evolve_index=5/8, eps=50000, target=50000)|difficulty=too_easy(sr=0.845>0.800)`
- audit: direction=increase, accepted=True, reason=candidate_increased_evidence_score, score=0.0268->0.0464 (+0.0195), attempts=1
- valid generations: current=100/100, candidate=100/100
- generator length: 16051 -> 16003 (-48)
- rollout distribution: low=97.5%, far+low=92.9%, med/high=2.5%, near+med/high=2.2%, center+med/high=0.1%
- final generator sampled distribution: low=95.0%, far+low=91.5%, med/high=5.0%, near+med/high=4.5%, center+med/high=0.5%

Top failure-associated attribution cells:
- alpha=050_to_075|beta_abs=near_side|blockage=high (lift=6.155, n=23, fail=22, cov=0.11%, fr=95.7%)
- alpha=025_to_050|beta_abs=centerline|blockage=high (lift=6.113, n=20, fail=19, cov=0.10%, fr=95.0%)
- alpha=050_to_075|beta_abs=near_side|blockage=medium (lift=6.064, n=52, fail=49, cov=0.26%, fr=94.2%)
- alpha=025_to_050|beta_abs=near_side|blockage=high (lift=5.560, n=103, fail=89, cov=0.52%, fr=86.4%)
- alpha=050_to_075|beta_abs=near_side|blockage=low (lift=5.445, n=13, fail=11, cov=0.07%, fr=84.6%)

Training rollout most-sampled cells before evolve:
- alpha=050_to_075|beta_abs=far_side|blockage=low (cov=32.3%, n=6453)
- alpha=after_goal|beta_abs=far_side|blockage=low (cov=24.7%, n=4943)
- alpha=075_to_goal|beta_abs=far_side|blockage=low (cov=19.8%, n=3957)
- alpha=025_to_050|beta_abs=far_side|blockage=low (cov=11.1%, n=2218)
- alpha=before_start|beta_abs=far_side|blockage=low (cov=3.0%, n=610)

Final candidate generator most-sampled cells:
- alpha=after_goal|beta_abs=far_side|blockage=low (cov=30.5%, n=61, ev=0.00)
- alpha=050_to_075|beta_abs=far_side|blockage=low (cov=28.0%, n=56, ev=0.00)
- alpha=025_to_050|beta_abs=far_side|blockage=low (cov=18.5%, n=37, ev=0.09)
- alpha=075_to_goal|beta_abs=far_side|blockage=low (cov=9.5%, n=19, ev=0.00)
- alpha=start_to_025|beta_abs=far_side|blockage=low (cov=3.5%, n=7, ev=0.00)

### Round 6 @ episode 60000

- batch: success_rate=0.724; success=7241, collision=2621, timeout=137, fall=1
- trigger: `fixed_schedule(evolve_index=6/8, eps=60000, target=60000)|difficulty=balanced(sr=0.724, range=[0.200,0.800])`
- audit: direction=preserve, accepted=False, reason=candidate_reduced_evidence_score, score=0.1183->0.0825 (-0.0358), attempts=1
- valid generations: current=100/100, candidate=100/100
- generator length: 16003 -> 17231 (+1228)
- rollout distribution: low=92.6%, far+low=87.9%, med/high=7.4%, near+med/high=5.6%, center+med/high=1.4%
- final generator sampled distribution: low=97.5%, far+low=94.5%, med/high=2.5%, near+med/high=2.0%, center+med/high=0.5%

Top failure-associated attribution cells:
- alpha=025_to_050|beta_abs=centerline|blockage=high (lift=3.562, n=173, fail=170, cov=0.86%, fr=98.3%)
- alpha=050_to_075|beta_abs=centerline|blockage=high (lift=3.554, n=103, fail=101, cov=0.52%, fr=98.1%)
- alpha=050_to_075|beta_abs=near_side|blockage=low (lift=3.485, n=78, fail=75, cov=0.39%, fr=96.2%)
- alpha=050_to_075|beta_abs=near_side|blockage=high (lift=3.413, n=137, fail=129, cov=0.69%, fr=94.2%)
- alpha=050_to_075|beta_abs=near_side|blockage=medium (lift=3.310, n=277, fail=253, cov=1.39%, fr=91.3%)

Training rollout most-sampled cells before evolve:
- alpha=after_goal|beta_abs=far_side|blockage=low (cov=29.7%, n=5936)
- alpha=050_to_075|beta_abs=far_side|blockage=low (cov=27.0%, n=5405)
- alpha=025_to_050|beta_abs=far_side|blockage=low (cov=18.8%, n=3766)
- alpha=075_to_goal|beta_abs=far_side|blockage=low (cov=7.8%, n=1552)
- alpha=after_goal|beta_abs=near_side|blockage=low (cov=2.9%, n=583)

Final candidate generator most-sampled cells:
- alpha=after_goal|beta_abs=far_side|blockage=low (cov=34.0%, n=68, ev=0.00)
- alpha=050_to_075|beta_abs=far_side|blockage=low (cov=31.5%, n=63, ev=0.17)
- alpha=025_to_050|beta_abs=far_side|blockage=low (cov=19.0%, n=38, ev=0.08)
- alpha=075_to_goal|beta_abs=far_side|blockage=low (cov=7.0%, n=14, ev=0.00)
- alpha=start_to_025|beta_abs=far_side|blockage=low (cov=2.5%, n=5, ev=0.00)

### Round 7 @ episode 70000

- batch: success_rate=0.869; success=8686, collision=1142, timeout=169, fall=3
- trigger: `fixed_schedule(evolve_index=7/8, eps=70000, target=70000)|difficulty=too_easy(sr=0.869>0.800)`
- audit: direction=increase, accepted=False, reason=candidate_did_not_increase_evidence_score, score=0.0911->0.0867 (-0.0044), attempts=1
- valid generations: current=100/100, candidate=100/100
- generator length: 17231 -> 16717 (-514)
- rollout distribution: low=97.8%, far+low=94.7%, med/high=2.2%, near+med/high=2.1%, center+med/high=0.0%
- final generator sampled distribution: low=96.5%, far+low=90.0%, med/high=3.5%, near+med/high=2.5%, center+med/high=0.5%

Top failure-associated attribution cells:
- alpha=025_to_050|beta_abs=near_side|blockage=high (lift=7.365, n=31, fail=30, cov=0.15%, fr=96.8%)
- alpha=050_to_075|beta_abs=near_side|blockage=high (lift=7.264, n=22, fail=21, cov=0.11%, fr=95.5%)
- alpha=050_to_075|beta_abs=near_side|blockage=medium (lift=6.666, n=129, fail=113, cov=0.65%, fr=87.6%)
- alpha=050_to_075|beta_abs=near_side|blockage=low (lift=6.628, n=62, fail=54, cov=0.31%, fr=87.1%)
- alpha=025_to_050|beta_abs=near_side|blockage=low (lift=5.784, n=75, fail=57, cov=0.38%, fr=76.0%)

Training rollout most-sampled cells before evolve:
- alpha=after_goal|beta_abs=far_side|blockage=low (cov=33.8%, n=6769)
- alpha=050_to_075|beta_abs=far_side|blockage=low (cov=27.7%, n=5547)
- alpha=025_to_050|beta_abs=far_side|blockage=low (cov=19.8%, n=3951)
- alpha=075_to_goal|beta_abs=far_side|blockage=low (cov=9.5%, n=1907)
- alpha=start_to_025|beta_abs=far_side|blockage=low (cov=2.4%, n=473)

Final candidate generator most-sampled cells:
- alpha=after_goal|beta_abs=far_side|blockage=low (cov=26.5%, n=53, ev=0.00)
- alpha=025_to_050|beta_abs=far_side|blockage=low (cov=22.5%, n=45, ev=0.22)
- alpha=050_to_075|beta_abs=far_side|blockage=low (cov=20.5%, n=41, ev=0.11)
- alpha=075_to_goal|beta_abs=far_side|blockage=low (cov=10.5%, n=21, ev=0.00)
- alpha=start_to_025|beta_abs=far_side|blockage=low (cov=6.5%, n=13, ev=0.00)

### Round 8 @ episode 80000

- batch: success_rate=0.785; success=7854, collision=1984, timeout=162, fall=0
- trigger: `fixed_schedule(evolve_index=8/8, eps=80000, target=80000)|difficulty=balanced(sr=0.785, range=[0.200,0.800])`
- audit: direction=preserve, accepted=False, reason=candidate_reduced_evidence_score, score=0.1046->0.0796 (-0.0250), attempts=1
- valid generations: current=100/100, candidate=100/100
- generator length: 16717 -> 17769 (+1052)
- rollout distribution: low=94.4%, far+low=87.9%, med/high=5.6%, near+med/high=5.0%, center+med/high=0.4%
- final generator sampled distribution: low=99.0%, far+low=93.5%, med/high=1.0%, near+med/high=1.0%, center+med/high=0.0%

Top failure-associated attribution cells:
- alpha=050_to_075|beta_abs=centerline|blockage=high (lift=4.660, n=26, fail=26, cov=0.13%, fr=100.0%)
- alpha=025_to_050|beta_abs=centerline|blockage=high (lift=4.499, n=58, fail=56, cov=0.29%, fr=96.6%)
- alpha=050_to_075|beta_abs=near_side|blockage=high (lift=4.379, n=83, fail=78, cov=0.41%, fr=94.0%)
- alpha=050_to_075|beta_abs=near_side|blockage=low (lift=4.062, n=78, fail=68, cov=0.39%, fr=87.2%)
- alpha=025_to_050|beta_abs=near_side|blockage=low (lift=4.060, n=202, fail=176, cov=1.01%, fr=87.1%)

Training rollout most-sampled cells before evolve:
- alpha=after_goal|beta_abs=far_side|blockage=low (cov=27.2%, n=5430)
- alpha=025_to_050|beta_abs=far_side|blockage=low (cov=22.8%, n=4554)
- alpha=050_to_075|beta_abs=far_side|blockage=low (cov=20.1%, n=4014)
- alpha=075_to_goal|beta_abs=far_side|blockage=low (cov=8.3%, n=1669)
- alpha=start_to_025|beta_abs=far_side|blockage=low (cov=5.0%, n=1007)

Final candidate generator most-sampled cells:
- alpha=025_to_050|beta_abs=far_side|blockage=low (cov=35.5%, n=71, ev=0.17)
- alpha=after_goal|beta_abs=far_side|blockage=low (cov=24.5%, n=49, ev=0.00)
- alpha=050_to_075|beta_abs=far_side|blockage=low (cov=18.0%, n=36, ev=0.08)
- alpha=075_to_goal|beta_abs=far_side|blockage=low (cov=8.5%, n=17, ev=0.00)
- alpha=before_start|beta_abs=far_side|blockage=low (cov=3.5%, n=7, ev=0.00)


## New Attribution + Verifier Run

### Round 1 @ episode 10000

- batch: success_rate=0.079; success=786, collision=4297, timeout=4563, fall=354
- trigger: `fixed_schedule(evolve_index=1/8, eps=10000, target=10000)|difficulty=too_hard(sr=0.079<0.200)`
- audit: direction=decrease, accepted=True, reason=candidate_decreased_evidence_score, score=0.1585->0.0582 (-0.1003), attempts=1
- valid generations: current=500/500, candidate=500/500
- generator length: 6347 -> 8299 (+1952)
- rollout distribution: low=99.9%, far+low=95.3%, med/high=0.1%, near+med/high=0.1%, center+med/high=0.0%
- final generator sampled distribution: low=100.0%, far+low=74.7%, med/high=0.0%, near+med/high=0.0%, center+med/high=0.0%

Top failure-associated attribution cells:
- alpha=before_start|beta_abs=centerline|blockage=low (lift=1.085, n=13, fail=13, cov=0.07%, fr=100.0%)
- alpha=025_to_050|beta_abs=near_side|blockage=low (lift=1.085, n=11, fail=11, cov=0.06%, fr=100.0%)
- alpha=050_to_075|beta_abs=near_side|blockage=low (lift=1.085, n=11, fail=11, cov=0.06%, fr=100.0%)
- alpha=before_start|beta_abs=far_side|blockage=low (lift=1.013, n=674, fail=629, cov=3.37%, fr=93.3%)
- alpha=025_to_050|beta_abs=far_side|blockage=low (lift=1.005, n=5665, fail=5244, cov=28.32%, fr=92.6%)

Training rollout most-sampled cells before evolve:
- alpha=050_to_075|beta_abs=far_side|blockage=low (cov=30.8%, n=6151)
- alpha=025_to_050|beta_abs=far_side|blockage=low (cov=28.3%, n=5665)
- alpha=after_goal|beta_abs=far_side|blockage=low (cov=17.4%, n=3476)
- alpha=075_to_goal|beta_abs=far_side|blockage=low (cov=9.2%, n=1836)
- alpha=start_to_025|beta_abs=far_side|blockage=low (cov=6.3%, n=1265)

Final candidate generator most-sampled cells:
- alpha=after_goal|beta_abs=far_side|blockage=low (cov=43.5%, n=435, ev=0.00)
- alpha=after_goal|beta_abs=near_side|blockage=low (cov=15.2%, n=152, ev=0.00)
- alpha=after_goal|beta_abs=centerline|blockage=low (cov=10.1%, n=101, ev=0.00)
- alpha=050_to_075|beta_abs=far_side|blockage=low (cov=8.6%, n=86, ev=0.17)
- alpha=025_to_050|beta_abs=far_side|blockage=low (cov=7.9%, n=79, ev=0.33)

### Round 2 @ episode 20000

- batch: success_rate=0.769; success=7690, collision=396, timeout=1904, fall=10
- trigger: `fixed_schedule(evolve_index=2/8, eps=20000, target=20000)|difficulty=balanced(sr=0.769, range=[0.200,0.800])`
- audit: direction=preserve, accepted=True, reason=candidate_preserved_evidence_score, score=0.1672->0.3340 (+0.1668), attempts=1
- valid generations: current=500/500, candidate=500/500
- generator length: 8299 -> 11434 (+3135)
- rollout distribution: low=100.0%, far+low=75.7%, med/high=0.0%, near+med/high=0.0%, center+med/high=0.0%
- final generator sampled distribution: low=100.0%, far+low=95.8%, med/high=0.0%, near+med/high=0.0%, center+med/high=0.0%

Top failure-associated attribution cells:
- alpha=before_start|beta_abs=far_side|blockage=low (lift=1.264, n=877, fail=256, cov=4.38%, fr=29.2%)
- alpha=start_to_025|beta_abs=far_side|blockage=low (lift=1.207, n=692, fail=193, cov=3.46%, fr=27.9%)
- alpha=050_to_075|beta_abs=far_side|blockage=low (lift=1.131, n=1757, fail=459, cov=8.79%, fr=26.1%)
- alpha=025_to_050|beta_abs=far_side|blockage=low (lift=1.097, n=1523, fail=386, cov=7.61%, fr=25.3%)
- alpha=075_to_goal|beta_abs=far_side|blockage=low (lift=1.037, n=1332, fail=319, cov=6.66%, fr=23.9%)

Training rollout most-sampled cells before evolve:
- alpha=after_goal|beta_abs=far_side|blockage=low (cov=44.8%, n=8960)
- alpha=after_goal|beta_abs=near_side|blockage=low (cov=15.2%, n=3041)
- alpha=after_goal|beta_abs=centerline|blockage=low (cov=9.0%, n=1806)
- alpha=050_to_075|beta_abs=far_side|blockage=low (cov=8.8%, n=1757)
- alpha=025_to_050|beta_abs=far_side|blockage=low (cov=7.6%, n=1523)

Final candidate generator most-sampled cells:
- alpha=after_goal|beta_abs=far_side|blockage=low (cov=33.4%, n=334, ev=0.00)
- alpha=050_to_075|beta_abs=far_side|blockage=low (cov=19.5%, n=195, ev=0.60)
- alpha=025_to_050|beta_abs=far_side|blockage=low (cov=17.0%, n=170, ev=0.40)
- alpha=075_to_goal|beta_abs=far_side|blockage=low (cov=11.9%, n=119, ev=0.20)
- alpha=start_to_025|beta_abs=far_side|blockage=low (cov=7.4%, n=74, ev=0.80)

### Round 3 @ episode 30000

- batch: success_rate=0.886; success=8864, collision=505, timeout=630, fall=1
- trigger: `fixed_schedule(evolve_index=3/8, eps=30000, target=30000)|difficulty=too_easy(sr=0.886>0.800)`
- audit: direction=increase, accepted=True, reason=candidate_increased_evidence_score, score=0.1740->0.3740 (+0.2000), attempts=1
- valid generations: current=500/500, candidate=500/500
- generator length: 11434 -> 13337 (+1903)
- rollout distribution: low=100.0%, far+low=95.5%, med/high=0.0%, near+med/high=0.0%, center+med/high=0.0%
- final generator sampled distribution: low=100.0%, far+low=99.0%, med/high=0.0%, near+med/high=0.0%, center+med/high=0.0%

Top failure-associated attribution cells:
- alpha=before_start|beta_abs=near_side|blockage=low (lift=3.201, n=11, fail=4, cov=0.06%, fr=36.4%)
- alpha=025_to_050|beta_abs=far_side|blockage=low (lift=1.189, n=3271, fail=442, cov=16.36%, fr=13.5%)
- alpha=050_to_075|beta_abs=far_side|blockage=low (lift=1.152, n=3692, fail=483, cov=18.46%, fr=13.1%)
- alpha=075_to_goal|beta_abs=far_side|blockage=low (lift=0.995, n=2478, fail=280, cov=12.39%, fr=11.3%)
- alpha=before_start|beta_abs=far_side|blockage=low (lift=0.951, n=1175, fail=127, cov=5.88%, fr=10.8%)

Training rollout most-sampled cells before evolve:
- alpha=after_goal|beta_abs=far_side|blockage=low (cov=34.8%, n=6970)
- alpha=050_to_075|beta_abs=far_side|blockage=low (cov=18.5%, n=3692)
- alpha=025_to_050|beta_abs=far_side|blockage=low (cov=16.4%, n=3271)
- alpha=075_to_goal|beta_abs=far_side|blockage=low (cov=12.4%, n=2478)
- alpha=start_to_025|beta_abs=far_side|blockage=low (cov=7.6%, n=1523)

Final candidate generator most-sampled cells:
- alpha=025_to_050|beta_abs=far_side|blockage=low (cov=39.1%, n=391, ev=0.67)
- alpha=050_to_075|beta_abs=far_side|blockage=low (cov=33.7%, n=337, ev=0.33)
- alpha=after_goal|beta_abs=far_side|blockage=low (cov=9.8%, n=98, ev=0.00)
- alpha=075_to_goal|beta_abs=far_side|blockage=low (cov=6.9%, n=69, ev=0.00)
- alpha=start_to_025|beta_abs=far_side|blockage=low (cov=5.2%, n=52, ev=0.00)

### Round 4 @ episode 40000

- batch: success_rate=0.845; success=8454, collision=1173, timeout=370, fall=3
- trigger: `fixed_schedule(evolve_index=4/8, eps=40000, target=40000)|difficulty=too_easy(sr=0.845>0.800)`
- audit: direction=increase, accepted=True, reason=candidate_increased_evidence_score, score=0.5135->0.6745 (+0.1610), attempts=1
- valid generations: current=500/500, candidate=500/500
- generator length: 13337 -> 13185 (-152)
- rollout distribution: low=100.0%, far+low=99.4%, med/high=0.0%, near+med/high=0.0%, center+med/high=0.0%
- final generator sampled distribution: low=100.0%, far+low=99.8%, med/high=0.0%, near+med/high=0.0%, center+med/high=0.0%

Top failure-associated attribution cells:
- alpha=050_to_075|beta_abs=far_side|blockage=low (lift=1.134, n=6746, fail=1183, cov=33.73%, fr=17.5%)
- alpha=025_to_050|beta_abs=far_side|blockage=low (lift=1.098, n=7351, fail=1248, cov=36.75%, fr=17.0%)
- alpha=075_to_goal|beta_abs=far_side|blockage=low (lift=0.798, n=1565, fail=193, cov=7.83%, fr=12.3%)
- alpha=after_goal|beta_abs=centerline|blockage=low (lift=0.770, n=42, fail=5, cov=0.21%, fr=11.9%)
- alpha=start_to_025|beta_abs=far_side|blockage=low (lift=0.713, n=1080, fail=119, cov=5.40%, fr=11.0%)

Training rollout most-sampled cells before evolve:
- alpha=025_to_050|beta_abs=far_side|blockage=low (cov=36.8%, n=7351)
- alpha=050_to_075|beta_abs=far_side|blockage=low (cov=33.7%, n=6746)
- alpha=after_goal|beta_abs=far_side|blockage=low (cov=10.5%, n=2104)
- alpha=075_to_goal|beta_abs=far_side|blockage=low (cov=7.8%, n=1565)
- alpha=start_to_025|beta_abs=far_side|blockage=low (cov=5.4%, n=1080)

Final candidate generator most-sampled cells:
- alpha=050_to_075|beta_abs=far_side|blockage=low (cov=45.4%, n=454, ev=1.00)
- alpha=025_to_050|beta_abs=far_side|blockage=low (cov=44.1%, n=441, ev=0.50)
- alpha=075_to_goal|beta_abs=far_side|blockage=low (cov=3.8%, n=38, ev=0.00)
- alpha=after_goal|beta_abs=far_side|blockage=low (cov=2.3%, n=23, ev=0.00)
- alpha=before_start|beta_abs=far_side|blockage=low (cov=2.2%, n=22, ev=0.00)

### Round 5 @ episode 50000

- batch: success_rate=0.864; success=8637, collision=1118, timeout=241, fall=4
- trigger: `fixed_schedule(evolve_index=5/8, eps=50000, target=50000)|difficulty=too_easy(sr=0.864>0.800)`
- audit: direction=increase, accepted=True, reason=candidate_increased_evidence_score, score=0.2716->0.2922 (+0.0206), attempts=2
- valid generations: current=500/500, candidate=500/500
- generator length: 13185 -> 17082 (+3897)
- rollout distribution: low=100.0%, far+low=99.6%, med/high=0.0%, near+med/high=0.0%, center+med/high=0.0%
- final generator sampled distribution: low=99.9%, far+low=97.4%, med/high=0.1%, near+med/high=0.0%, center+med/high=0.0%

Top failure-associated attribution cells:
- alpha=025_to_050|beta_abs=near_side|blockage=low (lift=6.208, n=13, fail=11, cov=0.07%, fr=84.6%)
- alpha=050_to_075|beta_abs=near_side|blockage=low (lift=6.208, n=13, fail=11, cov=0.07%, fr=84.6%)
- alpha=after_goal|beta_abs=centerline|blockage=low (lift=1.223, n=18, fail=3, cov=0.09%, fr=16.7%)
- alpha=050_to_075|beta_abs=far_side|blockage=low (lift=1.050, n=8762, fail=1254, cov=43.81%, fr=14.3%)
- alpha=025_to_050|beta_abs=far_side|blockage=low (lift=1.045, n=8997, fail=1282, cov=44.98%, fr=14.2%)

Training rollout most-sampled cells before evolve:
- alpha=025_to_050|beta_abs=far_side|blockage=low (cov=45.0%, n=8997)
- alpha=050_to_075|beta_abs=far_side|blockage=low (cov=43.8%, n=8762)
- alpha=075_to_goal|beta_abs=far_side|blockage=low (cov=4.7%, n=940)
- alpha=after_goal|beta_abs=far_side|blockage=low (cov=2.5%, n=492)
- alpha=before_start|beta_abs=far_side|blockage=low (cov=1.9%, n=383)

Final candidate generator most-sampled cells:
- alpha=050_to_075|beta_abs=far_side|blockage=low (cov=48.8%, n=488, ev=0.40)
- alpha=025_to_050|beta_abs=far_side|blockage=low (cov=39.6%, n=396, ev=0.20)
- alpha=075_to_goal|beta_abs=far_side|blockage=low (cov=5.9%, n=59, ev=0.00)
- alpha=after_goal|beta_abs=far_side|blockage=low (cov=1.7%, n=17, ev=0.00)
- alpha=start_to_025|beta_abs=far_side|blockage=low (cov=1.4%, n=14, ev=0.00)

### Round 6 @ episode 60000

- batch: success_rate=0.872; success=8722, collision=1101, timeout=175, fall=2
- trigger: `fixed_schedule(evolve_index=6/8, eps=60000, target=60000)|difficulty=too_easy(sr=0.872>0.800)`
- audit: direction=increase, accepted=True, reason=candidate_increased_evidence_score, score=0.1014->0.1279 (+0.0265), attempts=1
- valid generations: current=500/500, candidate=499/500
- generator length: 17082 -> 17654 (+572)
- rollout distribution: low=99.8%, far+low=97.6%, med/high=0.2%, near+med/high=0.2%, center+med/high=0.0%
- final generator sampled distribution: low=95.0%, far+low=90.2%, med/high=5.0%, near+med/high=4.6%, center+med/high=0.0%

Top failure-associated attribution cells:
- alpha=025_to_050|beta_abs=near_side|blockage=medium (lift=5.766, n=19, fail=14, cov=0.10%, fr=73.7%)
- alpha=025_to_050|beta_abs=near_side|blockage=low (lift=5.172, n=118, fail=78, cov=0.59%, fr=66.1%)
- alpha=050_to_075|beta_abs=near_side|blockage=low (lift=4.923, n=89, fail=56, cov=0.45%, fr=62.9%)
- alpha=050_to_075|beta_abs=near_side|blockage=medium (lift=4.564, n=12, fail=7, cov=0.06%, fr=58.3%)
- alpha=025_to_050|beta_abs=far_side|blockage=low (lift=1.004, n=8085, fail=1037, cov=40.42%, fr=12.8%)

Training rollout most-sampled cells before evolve:
- alpha=050_to_075|beta_abs=far_side|blockage=low (cov=48.6%, n=9728)
- alpha=025_to_050|beta_abs=far_side|blockage=low (cov=40.4%, n=8085)
- alpha=075_to_goal|beta_abs=far_side|blockage=low (cov=5.4%, n=1078)
- alpha=after_goal|beta_abs=far_side|blockage=low (cov=2.0%, n=406)
- alpha=start_to_025|beta_abs=far_side|blockage=low (cov=1.1%, n=210)

Final candidate generator most-sampled cells:
- alpha=025_to_050|beta_abs=far_side|blockage=low (cov=35.9%, n=358, ev=0.20)
- alpha=050_to_075|beta_abs=far_side|blockage=low (cov=32.2%, n=321, ev=0.00)
- alpha=after_goal|beta_abs=far_side|blockage=low (cov=13.8%, n=138, ev=0.00)
- alpha=075_to_goal|beta_abs=far_side|blockage=low (cov=5.0%, n=50, ev=0.00)
- alpha=025_to_050|beta_abs=near_side|blockage=medium (cov=2.7%, n=27, ev=1.00)

### Round 7 @ episode 70000

- batch: success_rate=0.806; success=8065, collision=1823, timeout=110, fall=2
- trigger: `fixed_schedule(evolve_index=7/8, eps=70000, target=70000)|difficulty=too_easy(sr=0.806>0.800)`
- audit: direction=increase, accepted=True, reason=candidate_increased_evidence_score, score=0.0322->0.0690 (+0.0368), attempts=1
- valid generations: current=495/500, candidate=494/500
- generator length: 17654 -> 17821 (+167)
- rollout distribution: low=96.2%, far+low=91.7%, med/high=3.8%, near+med/high=3.3%, center+med/high=0.0%
- final generator sampled distribution: low=92.0%, far+low=86.4%, med/high=8.0%, near+med/high=7.0%, center+med/high=0.0%

Top failure-associated attribution cells:
- alpha=050_to_075|beta_abs=near_side|blockage=medium (lift=4.594, n=270, fail=240, cov=1.35%, fr=88.9%)
- alpha=050_to_075|beta_abs=near_side|blockage=low (lift=4.204, n=252, fail=205, cov=1.26%, fr=81.3%)
- alpha=025_to_050|beta_abs=near_side|blockage=medium (lift=4.187, n=390, fail=316, cov=1.95%, fr=81.0%)
- alpha=050_to_075|beta_abs=far_side|blockage=medium (lift=3.661, n=24, fail=17, cov=0.12%, fr=70.8%)
- alpha=025_to_050|beta_abs=near_side|blockage=low (lift=3.642, n=254, fail=179, cov=1.27%, fr=70.5%)

Training rollout most-sampled cells before evolve:
- alpha=025_to_050|beta_abs=far_side|blockage=low (cov=35.9%, n=7179)
- alpha=050_to_075|beta_abs=far_side|blockage=low (cov=33.7%, n=6748)
- alpha=after_goal|beta_abs=far_side|blockage=low (cov=13.2%, n=2632)
- alpha=075_to_goal|beta_abs=far_side|blockage=low (cov=3.9%, n=778)
- alpha=start_to_025|beta_abs=far_side|blockage=low (cov=2.8%, n=555)

Final candidate generator most-sampled cells:
- alpha=after_goal|beta_abs=far_side|blockage=low (cov=32.6%, n=322, ev=0.00)
- alpha=050_to_075|beta_abs=far_side|blockage=low (cov=21.7%, n=214, ev=0.00)
- alpha=025_to_050|beta_abs=far_side|blockage=low (cov=15.9%, n=157, ev=0.00)
- alpha=075_to_goal|beta_abs=far_side|blockage=low (cov=8.0%, n=79, ev=0.00)
- alpha=before_start|beta_abs=far_side|blockage=low (cov=4.8%, n=47, ev=0.00)

### Round 8 @ episode 80000

- batch: success_rate=0.783; success=7826, collision=2043, timeout=130, fall=1
- trigger: `fixed_schedule(evolve_index=8/8, eps=80000, target=80000)|difficulty=balanced(sr=0.783, range=[0.200,0.800])`
- audit: direction=preserve, accepted=False, reason=candidate_reduced_evidence_score, score=0.0663->0.0505 (-0.0158), attempts=3
- valid generations: current=491/500, candidate=486/500
- generator length: 17821 -> 17821 (+0)
- rollout distribution: low=89.9%, far+low=84.7%, med/high=10.1%, near+med/high=9.2%, center+med/high=0.0%
- final generator sampled distribution: low=93.0%, far+low=90.1%, med/high=7.0%, near+med/high=6.2%, center+med/high=0.4%

Top failure-associated attribution cells:
- alpha=025_to_050|beta_abs=near_side|blockage=high (lift=4.169, n=32, fail=29, cov=0.16%, fr=90.6%)
- alpha=050_to_075|beta_abs=near_side|blockage=medium (lift=4.112, n=839, fail=750, cov=4.20%, fr=89.4%)
- alpha=050_to_075|beta_abs=near_side|blockage=high (lift=3.965, n=29, fail=25, cov=0.14%, fr=86.2%)
- alpha=025_to_050|beta_abs=near_side|blockage=low (lift=3.947, n=162, fail=139, cov=0.81%, fr=85.8%)
- alpha=050_to_075|beta_abs=near_side|blockage=low (lift=3.852, n=166, fail=139, cov=0.83%, fr=83.7%)

Training rollout most-sampled cells before evolve:
- alpha=after_goal|beta_abs=far_side|blockage=low (cov=32.6%, n=6519)
- alpha=050_to_075|beta_abs=far_side|blockage=low (cov=21.2%, n=4242)
- alpha=025_to_050|beta_abs=far_side|blockage=low (cov=16.6%, n=3327)
- alpha=075_to_goal|beta_abs=far_side|blockage=low (cov=7.0%, n=1400)
- alpha=025_to_050|beta_abs=near_side|blockage=medium (cov=4.7%, n=949)

Final candidate generator most-sampled cells:
- alpha=025_to_050|beta_abs=far_side|blockage=low (cov=29.0%, n=282, ev=0.00)
- alpha=050_to_075|beta_abs=far_side|blockage=low (cov=28.8%, n=280, ev=0.00)
- alpha=before_start|beta_abs=far_side|blockage=low (cov=11.6%, n=113, ev=0.00)
- alpha=after_goal|beta_abs=far_side|blockage=low (cov=10.8%, n=105, ev=0.00)
- alpha=075_to_goal|beta_abs=far_side|blockage=low (cov=6.5%, n=63, ev=0.00)

### Round 8 @ episode 90000

- batch: success_rate=0.794; success=7941, collision=1885, timeout=171, fall=3
- trigger: `fixed_schedule(evolve_index=8/8, eps=90000, target=80000)|difficulty=balanced(sr=0.794, range=[0.200,0.800])`
- audit: direction=preserve, accepted=True, reason=candidate_preserved_evidence_score, score=0.0760->0.0869 (+0.0109), attempts=2
- valid generations: current=489/500, candidate=493/500
- generator length: 17821 -> 15959 (-1862)
- rollout distribution: low=90.2%, far+low=85.0%, med/high=9.7%, near+med/high=8.8%, center+med/high=0.0%
- final generator sampled distribution: low=88.6%, far+low=84.9%, med/high=11.4%, near+med/high=10.5%, center+med/high=0.1%

Top failure-associated attribution cells:
- alpha=050_to_075|beta_abs=near_side|blockage=high (lift=4.428, n=34, fail=31, cov=0.17%, fr=91.2%)
- alpha=025_to_050|beta_abs=near_side|blockage=high (lift=4.234, n=39, fail=34, cov=0.19%, fr=87.2%)
- alpha=050_to_075|beta_abs=near_side|blockage=medium (lift=4.207, n=777, fail=673, cov=3.89%, fr=86.6%)
- alpha=025_to_050|beta_abs=near_side|blockage=medium (lift=3.938, n=915, fail=742, cov=4.58%, fr=81.1%)
- alpha=050_to_075|beta_abs=near_side|blockage=low (lift=3.885, n=150, fail=120, cov=0.75%, fr=80.0%)

Training rollout most-sampled cells before evolve:
- alpha=after_goal|beta_abs=far_side|blockage=low (cov=32.5%, n=6506)
- alpha=050_to_075|beta_abs=far_side|blockage=low (cov=21.1%, n=4222)
- alpha=025_to_050|beta_abs=far_side|blockage=low (cov=16.9%, n=3384)
- alpha=075_to_goal|beta_abs=far_side|blockage=low (cov=7.1%, n=1430)
- alpha=025_to_050|beta_abs=near_side|blockage=medium (cov=4.6%, n=915)

Final candidate generator most-sampled cells:
- alpha=050_to_075|beta_abs=far_side|blockage=low (cov=24.1%, n=238, ev=0.00)
- alpha=025_to_050|beta_abs=far_side|blockage=low (cov=23.4%, n=231, ev=0.00)
- alpha=after_goal|beta_abs=far_side|blockage=low (cov=22.7%, n=224, ev=0.00)
- alpha=before_start|beta_abs=far_side|blockage=low (cov=6.4%, n=63, ev=0.00)
- alpha=025_to_050|beta_abs=near_side|blockage=medium (cov=5.4%, n=53, ev=0.70)


## Short Interpretation

- Old run: all candidates were accepted by the training loop, but the 100-sample re-audit shows several updates would not satisfy the current evidence-direction criterion, especially rounds 4, 6, and 8. Round 7 is near zero and should be treated as borderline.
- New run: all finally accepted generators move in the requested evidence-score direction. Round 5 required one retry; round 8 failed at episode 80000 and succeeded at episode 90000.
- The verifier fixes directionality more than intensity. Many accepted generators still retain substantial low-blockage / far-side-low coverage, so this is not yet a complete solution to weak curriculum strength.
- Generator length still grows in both runs; the verifier does not address code compactness.
