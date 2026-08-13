# Attribution + Generator Round Summary

This report compares the previous attribution-only run and the new attribution+verifier run. `score_delta` is candidate evidence score minus current evidence score under the round attribution map.

## Compact Table

| run | round | ep | sr | direction | delta | accepted | reason | len_delta |
|---|---:|---:|---:|---|---:|---|---|---:|
| old_attribution | 1 | 10000 | 0.079 | None | n/a | None | None | 3513 |
| old_attribution | 2 | 20000 | 0.752 | None | n/a | None | None | 3907 |
| old_attribution | 3 | 30000 | 0.868 | None | n/a | None | None | -255 |
| old_attribution | 4 | 40000 | 0.877 | None | n/a | None | None | 2539 |
| old_attribution | 5 | 50000 | 0.845 | None | n/a | None | None | -48 |
| old_attribution | 6 | 60000 | 0.724 | None | n/a | None | None | 1228 |
| old_attribution | 7 | 70000 | 0.869 | None | n/a | None | None | -514 |
| old_attribution | 8 | 80000 | 0.785 | None | n/a | None | None | 1052 |
| new_verifier | 1 | 10000 | 0.079 | decrease | -0.100 | True | candidate_decreased_evidence_score | 1952 |
| new_verifier | 2 | 20000 | 0.769 | preserve | 0.167 | True | candidate_preserved_evidence_score | 3135 |
| new_verifier | 3 | 30000 | 0.886 | increase | 0.200 | True | candidate_increased_evidence_score | 1903 |
| new_verifier | 4 | 40000 | 0.845 | increase | 0.161 | True | candidate_increased_evidence_score | -152 |
| new_verifier | 5 | 50000 | 0.864 | increase | 0.021 | True | candidate_increased_evidence_score | 3897 |
| new_verifier | 6 | 60000 | 0.872 | increase | 0.026 | True | candidate_increased_evidence_score | 572 |
| new_verifier | 7 | 70000 | 0.806 | increase | 0.037 | True | candidate_increased_evidence_score | 167 |
| new_verifier | 8 | 80000 | 0.783 | preserve | -0.016 | False | candidate_reduced_evidence_score | -1862 |
| new_verifier | 8 | 90000 | 0.794 | preserve | 0.011 | True | candidate_preserved_evidence_score | -1862 |

## old_attribution

### Round 1 @ episode 10000

- batch success_rate: 0.079; stats: success=786, collision=4297, timeout=4563, fall=354
- trigger: `fixed_schedule(evolve_index=1/8, eps=10000, target=10000)|difficulty=too_hard(sr=0.079<0.200)`
- generator length: 6347 -> 9860 (+3513)
- verifier/final audit: accepted=None, direction=None, score=n/a -> n/a, delta=n/a, reason=None, attempts=0

Top attribution cells:
- alpha=before_start|beta_abs=centerline|blockage=low (lift=1.085, n=13, fail=13, p=0.1%, fr=100.0%)
- alpha=025_to_050|beta_abs=near_side|blockage=low (lift=1.085, n=11, fail=11, p=0.1%, fr=100.0%)
- alpha=050_to_075|beta_abs=near_side|blockage=low (lift=1.085, n=11, fail=11, p=0.1%, fr=100.0%)
- alpha=before_start|beta_abs=far_side|blockage=low (lift=1.013, n=674, fail=629, p=3.4%, fr=93.3%)
- alpha=025_to_050|beta_abs=far_side|blockage=low (lift=1.005, n=5665, fail=5244, p=28.3%, fr=92.6%)

Current generator most sampled cells in rollout batch:
- alpha=050_to_075|beta_abs=far_side|blockage=low (cov=30.8%, n=6151)
- alpha=025_to_050|beta_abs=far_side|blockage=low (cov=28.3%, n=5665)
- alpha=after_goal|beta_abs=far_side|blockage=low (cov=17.4%, n=3476)
- alpha=075_to_goal|beta_abs=far_side|blockage=low (cov=9.2%, n=1836)
- alpha=start_to_025|beta_abs=far_side|blockage=low (cov=6.3%, n=1265)

Accepted/final candidate most sampled cells in verifier audit:
- no candidate audit saved

### Round 2 @ episode 20000

- batch success_rate: 0.752; stats: success=7516, collision=346, timeout=2118, fall=20
- trigger: `fixed_schedule(evolve_index=2/8, eps=20000, target=20000)|difficulty=balanced(sr=0.752, range=[0.200,0.800])`
- generator length: 9860 -> 13767 (+3907)
- verifier/final audit: accepted=None, direction=None, score=n/a -> n/a, delta=n/a, reason=None, attempts=0

Top attribution cells:
- alpha=start_to_025|beta_abs=far_side|blockage=low (lift=1.208, n=733, fail=220, p=3.7%, fr=30.0%)
- alpha=before_start|beta_abs=far_side|blockage=low (lift=1.168, n=931, fail=270, p=4.7%, fr=29.0%)
- alpha=025_to_050|beta_abs=far_side|blockage=low (lift=1.144, n=1882, fail=535, p=9.4%, fr=28.4%)
- alpha=075_to_goal|beta_abs=far_side|blockage=low (lift=1.107, n=1389, fail=382, p=6.9%, fr=27.5%)
- alpha=050_to_075|beta_abs=far_side|blockage=low (lift=1.058, n=2302, fail=605, p=11.5%, fr=26.3%)

Current generator most sampled cells in rollout batch:
- alpha=after_goal|beta_abs=far_side|blockage=low (cov=39.4%, n=7872)
- alpha=after_goal|beta_abs=near_side|blockage=low (cov=14.8%, n=2959)
- alpha=050_to_075|beta_abs=far_side|blockage=low (cov=11.5%, n=2302)
- alpha=after_goal|beta_abs=centerline|blockage=low (cov=9.6%, n=1914)
- alpha=025_to_050|beta_abs=far_side|blockage=low (cov=9.4%, n=1882)

Accepted/final candidate most sampled cells in verifier audit:
- no candidate audit saved

### Round 3 @ episode 30000

- batch success_rate: 0.868; stats: success=8683, collision=514, timeout=801, fall=2
- trigger: `fixed_schedule(evolve_index=3/8, eps=30000, target=30000)|difficulty=too_easy(sr=0.868>0.800)`
- generator length: 13767 -> 13512 (-255)
- verifier/final audit: accepted=None, direction=None, score=n/a -> n/a, delta=n/a, reason=None, attempts=0

Top attribution cells:
- alpha=before_start|beta_abs=centerline|blockage=low (lift=3.996, n=19, fail=10, p=0.1%, fr=52.6%)
- alpha=before_start|beta_abs=near_side|blockage=low (lift=2.441, n=28, fail=9, p=0.1%, fr=32.1%)
- alpha=before_start|beta_abs=far_side|blockage=low (lift=1.300, n=917, fail=157, p=4.6%, fr=17.1%)
- alpha=025_to_050|beta_abs=far_side|blockage=low (lift=1.253, n=4122, fail=680, p=20.6%, fr=16.5%)
- alpha=start_to_025|beta_abs=far_side|blockage=low (lift=1.176, n=1046, fail=162, p=5.2%, fr=15.5%)

Current generator most sampled cells in rollout batch:
- alpha=after_goal|beta_abs=far_side|blockage=low (cov=30.6%, n=6127)
- alpha=025_to_050|beta_abs=far_side|blockage=low (cov=20.6%, n=4122)
- alpha=050_to_075|beta_abs=far_side|blockage=low (cov=16.1%, n=3227)
- alpha=after_goal|beta_abs=near_side|blockage=low (cov=8.7%, n=1734)
- alpha=after_goal|beta_abs=centerline|blockage=low (cov=7.6%, n=1519)

Accepted/final candidate most sampled cells in verifier audit:
- no candidate audit saved

### Round 4 @ episode 40000

- batch success_rate: 0.877; stats: success=8774, collision=868, timeout=357, fall=1
- trigger: `fixed_schedule(evolve_index=4/8, eps=40000, target=40000)|difficulty=too_easy(sr=0.877>0.800)`
- generator length: 13512 -> 16051 (+2539)
- verifier/final audit: accepted=None, direction=None, score=n/a -> n/a, delta=n/a, reason=None, attempts=0

Top attribution cells:
- alpha=025_to_050|beta_abs=near_side|blockage=medium (lift=6.010, n=19, fail=14, p=0.1%, fr=73.7%)
- alpha=025_to_050|beta_abs=near_side|blockage=low (lift=3.670, n=20, fail=9, p=0.1%, fr=45.0%)
- alpha=before_start|beta_abs=centerline|blockage=low (lift=3.444, n=90, fail=38, p=0.4%, fr=42.2%)
- alpha=before_start|beta_abs=near_side|blockage=low (lift=3.131, n=99, fail=38, p=0.5%, fr=38.4%)
- alpha=025_to_050|beta_abs=far_side|blockage=low (lift=1.220, n=5181, fail=775, p=25.9%, fr=15.0%)

Current generator most sampled cells in rollout batch:
- alpha=025_to_050|beta_abs=far_side|blockage=low (cov=25.9%, n=5181)
- alpha=after_goal|beta_abs=far_side|blockage=low (cov=23.5%, n=4695)
- alpha=050_to_075|beta_abs=far_side|blockage=low (cov=22.1%, n=4418)
- alpha=075_to_goal|beta_abs=far_side|blockage=low (cov=12.9%, n=2575)
- alpha=start_to_025|beta_abs=far_side|blockage=low (cov=5.6%, n=1114)

Accepted/final candidate most sampled cells in verifier audit:
- no candidate audit saved

### Round 5 @ episode 50000

- batch success_rate: 0.845; stats: success=8446, collision=1349, timeout=203, fall=2
- trigger: `fixed_schedule(evolve_index=5/8, eps=50000, target=50000)|difficulty=too_easy(sr=0.845>0.800)`
- generator length: 16051 -> 16003 (-48)
- verifier/final audit: accepted=None, direction=None, score=n/a -> n/a, delta=n/a, reason=None, attempts=0

Top attribution cells:
- alpha=050_to_075|beta_abs=near_side|blockage=high (lift=6.155, n=23, fail=22, p=0.1%, fr=95.7%)
- alpha=025_to_050|beta_abs=centerline|blockage=high (lift=6.113, n=20, fail=19, p=0.1%, fr=95.0%)
- alpha=050_to_075|beta_abs=near_side|blockage=medium (lift=6.064, n=52, fail=49, p=0.3%, fr=94.2%)
- alpha=025_to_050|beta_abs=near_side|blockage=high (lift=5.560, n=103, fail=89, p=0.5%, fr=86.4%)
- alpha=050_to_075|beta_abs=near_side|blockage=low (lift=5.445, n=13, fail=11, p=0.1%, fr=84.6%)

Current generator most sampled cells in rollout batch:
- alpha=050_to_075|beta_abs=far_side|blockage=low (cov=32.3%, n=6453)
- alpha=after_goal|beta_abs=far_side|blockage=low (cov=24.7%, n=4943)
- alpha=075_to_goal|beta_abs=far_side|blockage=low (cov=19.8%, n=3957)
- alpha=025_to_050|beta_abs=far_side|blockage=low (cov=11.1%, n=2218)
- alpha=before_start|beta_abs=far_side|blockage=low (cov=3.0%, n=610)

Accepted/final candidate most sampled cells in verifier audit:
- no candidate audit saved

### Round 6 @ episode 60000

- batch success_rate: 0.724; stats: success=7241, collision=2621, timeout=137, fall=1
- trigger: `fixed_schedule(evolve_index=6/8, eps=60000, target=60000)|difficulty=balanced(sr=0.724, range=[0.200,0.800])`
- generator length: 16003 -> 17231 (+1228)
- verifier/final audit: accepted=None, direction=None, score=n/a -> n/a, delta=n/a, reason=None, attempts=0

Top attribution cells:
- alpha=025_to_050|beta_abs=centerline|blockage=high (lift=3.562, n=173, fail=170, p=0.9%, fr=98.3%)
- alpha=050_to_075|beta_abs=centerline|blockage=high (lift=3.554, n=103, fail=101, p=0.5%, fr=98.1%)
- alpha=050_to_075|beta_abs=near_side|blockage=low (lift=3.485, n=78, fail=75, p=0.4%, fr=96.2%)
- alpha=050_to_075|beta_abs=near_side|blockage=high (lift=3.413, n=137, fail=129, p=0.7%, fr=94.2%)
- alpha=050_to_075|beta_abs=near_side|blockage=medium (lift=3.310, n=277, fail=253, p=1.4%, fr=91.3%)

Current generator most sampled cells in rollout batch:
- alpha=after_goal|beta_abs=far_side|blockage=low (cov=29.7%, n=5936)
- alpha=050_to_075|beta_abs=far_side|blockage=low (cov=27.0%, n=5405)
- alpha=025_to_050|beta_abs=far_side|blockage=low (cov=18.8%, n=3766)
- alpha=075_to_goal|beta_abs=far_side|blockage=low (cov=7.8%, n=1552)
- alpha=after_goal|beta_abs=near_side|blockage=low (cov=2.9%, n=583)

Accepted/final candidate most sampled cells in verifier audit:
- no candidate audit saved

### Round 7 @ episode 70000

- batch success_rate: 0.869; stats: success=8686, collision=1142, timeout=169, fall=3
- trigger: `fixed_schedule(evolve_index=7/8, eps=70000, target=70000)|difficulty=too_easy(sr=0.869>0.800)`
- generator length: 17231 -> 16717 (-514)
- verifier/final audit: accepted=None, direction=None, score=n/a -> n/a, delta=n/a, reason=None, attempts=0

Top attribution cells:
- alpha=025_to_050|beta_abs=near_side|blockage=high (lift=7.365, n=31, fail=30, p=0.2%, fr=96.8%)
- alpha=050_to_075|beta_abs=near_side|blockage=high (lift=7.264, n=22, fail=21, p=0.1%, fr=95.5%)
- alpha=050_to_075|beta_abs=near_side|blockage=medium (lift=6.666, n=129, fail=113, p=0.6%, fr=87.6%)
- alpha=050_to_075|beta_abs=near_side|blockage=low (lift=6.628, n=62, fail=54, p=0.3%, fr=87.1%)
- alpha=025_to_050|beta_abs=near_side|blockage=low (lift=5.784, n=75, fail=57, p=0.4%, fr=76.0%)

Current generator most sampled cells in rollout batch:
- alpha=after_goal|beta_abs=far_side|blockage=low (cov=33.8%, n=6769)
- alpha=050_to_075|beta_abs=far_side|blockage=low (cov=27.7%, n=5547)
- alpha=025_to_050|beta_abs=far_side|blockage=low (cov=19.8%, n=3951)
- alpha=075_to_goal|beta_abs=far_side|blockage=low (cov=9.5%, n=1907)
- alpha=start_to_025|beta_abs=far_side|blockage=low (cov=2.4%, n=473)

Accepted/final candidate most sampled cells in verifier audit:
- no candidate audit saved

### Round 8 @ episode 80000

- batch success_rate: 0.785; stats: success=7854, collision=1984, timeout=162, fall=0
- trigger: `fixed_schedule(evolve_index=8/8, eps=80000, target=80000)|difficulty=balanced(sr=0.785, range=[0.200,0.800])`
- generator length: 16717 -> 17769 (+1052)
- verifier/final audit: accepted=None, direction=None, score=n/a -> n/a, delta=n/a, reason=None, attempts=0

Top attribution cells:
- alpha=050_to_075|beta_abs=centerline|blockage=high (lift=4.660, n=26, fail=26, p=0.1%, fr=100.0%)
- alpha=025_to_050|beta_abs=centerline|blockage=high (lift=4.499, n=58, fail=56, p=0.3%, fr=96.6%)
- alpha=050_to_075|beta_abs=near_side|blockage=high (lift=4.379, n=83, fail=78, p=0.4%, fr=94.0%)
- alpha=050_to_075|beta_abs=near_side|blockage=low (lift=4.062, n=78, fail=68, p=0.4%, fr=87.2%)
- alpha=025_to_050|beta_abs=near_side|blockage=low (lift=4.060, n=202, fail=176, p=1.0%, fr=87.1%)

Current generator most sampled cells in rollout batch:
- alpha=after_goal|beta_abs=far_side|blockage=low (cov=27.2%, n=5430)
- alpha=025_to_050|beta_abs=far_side|blockage=low (cov=22.8%, n=4554)
- alpha=050_to_075|beta_abs=far_side|blockage=low (cov=20.1%, n=4014)
- alpha=075_to_goal|beta_abs=far_side|blockage=low (cov=8.3%, n=1669)
- alpha=start_to_025|beta_abs=far_side|blockage=low (cov=5.0%, n=1007)

Accepted/final candidate most sampled cells in verifier audit:
- no candidate audit saved

## new_verifier

### Round 1 @ episode 10000

- batch success_rate: 0.079; stats: success=786, collision=4297, timeout=4563, fall=354
- trigger: `fixed_schedule(evolve_index=1/8, eps=10000, target=10000)|difficulty=too_hard(sr=0.079<0.200)`
- generator length: 6347 -> 8299 (+1952)
- verifier/final audit: accepted=True, direction=decrease, score=0.159 -> 0.058, delta=-0.100, reason=candidate_decreased_evidence_score, attempts=1

Top attribution cells:
- alpha=before_start|beta_abs=centerline|blockage=low (lift=1.085, n=13, fail=13, p=0.1%, fr=100.0%)
- alpha=025_to_050|beta_abs=near_side|blockage=low (lift=1.085, n=11, fail=11, p=0.1%, fr=100.0%)
- alpha=050_to_075|beta_abs=near_side|blockage=low (lift=1.085, n=11, fail=11, p=0.1%, fr=100.0%)
- alpha=before_start|beta_abs=far_side|blockage=low (lift=1.013, n=674, fail=629, p=3.4%, fr=93.3%)
- alpha=025_to_050|beta_abs=far_side|blockage=low (lift=1.005, n=5665, fail=5244, p=28.3%, fr=92.6%)

Current generator most sampled cells in rollout batch:
- alpha=050_to_075|beta_abs=far_side|blockage=low (cov=30.8%, n=6151)
- alpha=025_to_050|beta_abs=far_side|blockage=low (cov=28.3%, n=5665)
- alpha=after_goal|beta_abs=far_side|blockage=low (cov=17.4%, n=3476)
- alpha=075_to_goal|beta_abs=far_side|blockage=low (cov=9.2%, n=1836)
- alpha=start_to_025|beta_abs=far_side|blockage=low (cov=6.3%, n=1265)

Accepted/final candidate most sampled cells in verifier audit:
- alpha=after_goal|beta_abs=far_side|blockage=low (cov=43.5%, n=435, ev=0.000)
- alpha=after_goal|beta_abs=near_side|blockage=low (cov=15.2%, n=152, ev=0.000)
- alpha=after_goal|beta_abs=centerline|blockage=low (cov=10.1%, n=101, ev=0.000)
- alpha=050_to_075|beta_abs=far_side|blockage=low (cov=8.6%, n=86, ev=0.167)
- alpha=025_to_050|beta_abs=far_side|blockage=low (cov=7.9%, n=79, ev=0.333)

### Round 2 @ episode 20000

- batch success_rate: 0.769; stats: success=7690, collision=396, timeout=1904, fall=10
- trigger: `fixed_schedule(evolve_index=2/8, eps=20000, target=20000)|difficulty=balanced(sr=0.769, range=[0.200,0.800])`
- generator length: 8299 -> 11434 (+3135)
- verifier/final audit: accepted=True, direction=preserve, score=0.167 -> 0.334, delta=0.167, reason=candidate_preserved_evidence_score, attempts=1

Top attribution cells:
- alpha=before_start|beta_abs=far_side|blockage=low (lift=1.264, n=877, fail=256, p=4.4%, fr=29.2%)
- alpha=start_to_025|beta_abs=far_side|blockage=low (lift=1.207, n=692, fail=193, p=3.5%, fr=27.9%)
- alpha=050_to_075|beta_abs=far_side|blockage=low (lift=1.131, n=1757, fail=459, p=8.8%, fr=26.1%)
- alpha=025_to_050|beta_abs=far_side|blockage=low (lift=1.097, n=1523, fail=386, p=7.6%, fr=25.3%)
- alpha=075_to_goal|beta_abs=far_side|blockage=low (lift=1.037, n=1332, fail=319, p=6.7%, fr=23.9%)

Current generator most sampled cells in rollout batch:
- alpha=after_goal|beta_abs=far_side|blockage=low (cov=44.8%, n=8960)
- alpha=after_goal|beta_abs=near_side|blockage=low (cov=15.2%, n=3041)
- alpha=after_goal|beta_abs=centerline|blockage=low (cov=9.0%, n=1806)
- alpha=050_to_075|beta_abs=far_side|blockage=low (cov=8.8%, n=1757)
- alpha=025_to_050|beta_abs=far_side|blockage=low (cov=7.6%, n=1523)

Accepted/final candidate most sampled cells in verifier audit:
- alpha=after_goal|beta_abs=far_side|blockage=low (cov=33.4%, n=334, ev=0.000)
- alpha=050_to_075|beta_abs=far_side|blockage=low (cov=19.5%, n=195, ev=0.600)
- alpha=025_to_050|beta_abs=far_side|blockage=low (cov=17.0%, n=170, ev=0.400)
- alpha=075_to_goal|beta_abs=far_side|blockage=low (cov=11.9%, n=119, ev=0.200)
- alpha=start_to_025|beta_abs=far_side|blockage=low (cov=7.4%, n=74, ev=0.800)

### Round 3 @ episode 30000

- batch success_rate: 0.886; stats: success=8864, collision=505, timeout=630, fall=1
- trigger: `fixed_schedule(evolve_index=3/8, eps=30000, target=30000)|difficulty=too_easy(sr=0.886>0.800)`
- generator length: 11434 -> 13337 (+1903)
- verifier/final audit: accepted=True, direction=increase, score=0.174 -> 0.374, delta=0.200, reason=candidate_increased_evidence_score, attempts=1

Top attribution cells:
- alpha=before_start|beta_abs=near_side|blockage=low (lift=3.201, n=11, fail=4, p=0.1%, fr=36.4%)
- alpha=025_to_050|beta_abs=far_side|blockage=low (lift=1.189, n=3271, fail=442, p=16.4%, fr=13.5%)
- alpha=050_to_075|beta_abs=far_side|blockage=low (lift=1.152, n=3692, fail=483, p=18.5%, fr=13.1%)
- alpha=075_to_goal|beta_abs=far_side|blockage=low (lift=0.995, n=2478, fail=280, p=12.4%, fr=11.3%)
- alpha=before_start|beta_abs=far_side|blockage=low (lift=0.951, n=1175, fail=127, p=5.9%, fr=10.8%)

Current generator most sampled cells in rollout batch:
- alpha=after_goal|beta_abs=far_side|blockage=low (cov=34.8%, n=6970)
- alpha=050_to_075|beta_abs=far_side|blockage=low (cov=18.5%, n=3692)
- alpha=025_to_050|beta_abs=far_side|blockage=low (cov=16.4%, n=3271)
- alpha=075_to_goal|beta_abs=far_side|blockage=low (cov=12.4%, n=2478)
- alpha=start_to_025|beta_abs=far_side|blockage=low (cov=7.6%, n=1523)

Accepted/final candidate most sampled cells in verifier audit:
- alpha=025_to_050|beta_abs=far_side|blockage=low (cov=39.1%, n=391, ev=0.667)
- alpha=050_to_075|beta_abs=far_side|blockage=low (cov=33.7%, n=337, ev=0.333)
- alpha=after_goal|beta_abs=far_side|blockage=low (cov=9.8%, n=98, ev=0.000)
- alpha=075_to_goal|beta_abs=far_side|blockage=low (cov=6.9%, n=69, ev=0.000)
- alpha=start_to_025|beta_abs=far_side|blockage=low (cov=5.2%, n=52, ev=0.000)

### Round 4 @ episode 40000

- batch success_rate: 0.845; stats: success=8454, collision=1173, timeout=370, fall=3
- trigger: `fixed_schedule(evolve_index=4/8, eps=40000, target=40000)|difficulty=too_easy(sr=0.845>0.800)`
- generator length: 13337 -> 13185 (-152)
- verifier/final audit: accepted=True, direction=increase, score=0.513 -> 0.674, delta=0.161, reason=candidate_increased_evidence_score, attempts=1

Top attribution cells:
- alpha=050_to_075|beta_abs=far_side|blockage=low (lift=1.134, n=6746, fail=1183, p=33.7%, fr=17.5%)
- alpha=025_to_050|beta_abs=far_side|blockage=low (lift=1.098, n=7351, fail=1248, p=36.8%, fr=17.0%)
- alpha=075_to_goal|beta_abs=far_side|blockage=low (lift=0.798, n=1565, fail=193, p=7.8%, fr=12.3%)
- alpha=after_goal|beta_abs=centerline|blockage=low (lift=0.770, n=42, fail=5, p=0.2%, fr=11.9%)
- alpha=start_to_025|beta_abs=far_side|blockage=low (lift=0.713, n=1080, fail=119, p=5.4%, fr=11.0%)

Current generator most sampled cells in rollout batch:
- alpha=025_to_050|beta_abs=far_side|blockage=low (cov=36.8%, n=7351)
- alpha=050_to_075|beta_abs=far_side|blockage=low (cov=33.7%, n=6746)
- alpha=after_goal|beta_abs=far_side|blockage=low (cov=10.5%, n=2104)
- alpha=075_to_goal|beta_abs=far_side|blockage=low (cov=7.8%, n=1565)
- alpha=start_to_025|beta_abs=far_side|blockage=low (cov=5.4%, n=1080)

Accepted/final candidate most sampled cells in verifier audit:
- alpha=050_to_075|beta_abs=far_side|blockage=low (cov=45.4%, n=454, ev=1.000)
- alpha=025_to_050|beta_abs=far_side|blockage=low (cov=44.1%, n=441, ev=0.500)
- alpha=075_to_goal|beta_abs=far_side|blockage=low (cov=3.8%, n=38, ev=0.000)
- alpha=after_goal|beta_abs=far_side|blockage=low (cov=2.3%, n=23, ev=0.000)
- alpha=before_start|beta_abs=far_side|blockage=low (cov=2.2%, n=22, ev=0.000)

### Round 5 @ episode 50000

- batch success_rate: 0.864; stats: success=8637, collision=1118, timeout=241, fall=4
- trigger: `fixed_schedule(evolve_index=5/8, eps=50000, target=50000)|difficulty=too_easy(sr=0.864>0.800)`
- generator length: 13185 -> 17082 (+3897)
- verifier/final audit: accepted=True, direction=increase, score=0.272 -> 0.292, delta=0.021, reason=candidate_increased_evidence_score, attempts=2

Top attribution cells:
- alpha=025_to_050|beta_abs=near_side|blockage=low (lift=6.208, n=13, fail=11, p=0.1%, fr=84.6%)
- alpha=050_to_075|beta_abs=near_side|blockage=low (lift=6.208, n=13, fail=11, p=0.1%, fr=84.6%)
- alpha=after_goal|beta_abs=centerline|blockage=low (lift=1.223, n=18, fail=3, p=0.1%, fr=16.7%)
- alpha=050_to_075|beta_abs=far_side|blockage=low (lift=1.050, n=8762, fail=1254, p=43.8%, fr=14.3%)
- alpha=025_to_050|beta_abs=far_side|blockage=low (lift=1.045, n=8997, fail=1282, p=45.0%, fr=14.2%)

Current generator most sampled cells in rollout batch:
- alpha=025_to_050|beta_abs=far_side|blockage=low (cov=45.0%, n=8997)
- alpha=050_to_075|beta_abs=far_side|blockage=low (cov=43.8%, n=8762)
- alpha=075_to_goal|beta_abs=far_side|blockage=low (cov=4.7%, n=940)
- alpha=after_goal|beta_abs=far_side|blockage=low (cov=2.5%, n=492)
- alpha=before_start|beta_abs=far_side|blockage=low (cov=1.9%, n=383)

Accepted/final candidate most sampled cells in verifier audit:
- alpha=050_to_075|beta_abs=far_side|blockage=low (cov=48.8%, n=488, ev=0.400)
- alpha=025_to_050|beta_abs=far_side|blockage=low (cov=39.6%, n=396, ev=0.200)
- alpha=075_to_goal|beta_abs=far_side|blockage=low (cov=5.9%, n=59, ev=0.000)
- alpha=after_goal|beta_abs=far_side|blockage=low (cov=1.7%, n=17, ev=0.000)
- alpha=start_to_025|beta_abs=far_side|blockage=low (cov=1.4%, n=14, ev=0.000)

Attempts:
- candidate_001_audit.json: accepted=False, delta=-0.024, reason=candidate_did_not_increase_evidence_score
- candidate_002_audit.json: accepted=True, delta=0.021, reason=candidate_increased_evidence_score

### Round 6 @ episode 60000

- batch success_rate: 0.872; stats: success=8722, collision=1101, timeout=175, fall=2
- trigger: `fixed_schedule(evolve_index=6/8, eps=60000, target=60000)|difficulty=too_easy(sr=0.872>0.800)`
- generator length: 17082 -> 17654 (+572)
- verifier/final audit: accepted=True, direction=increase, score=0.101 -> 0.128, delta=0.026, reason=candidate_increased_evidence_score, attempts=1

Top attribution cells:
- alpha=025_to_050|beta_abs=near_side|blockage=medium (lift=5.766, n=19, fail=14, p=0.1%, fr=73.7%)
- alpha=025_to_050|beta_abs=near_side|blockage=low (lift=5.172, n=118, fail=78, p=0.6%, fr=66.1%)
- alpha=050_to_075|beta_abs=near_side|blockage=low (lift=4.923, n=89, fail=56, p=0.4%, fr=62.9%)
- alpha=050_to_075|beta_abs=near_side|blockage=medium (lift=4.564, n=12, fail=7, p=0.1%, fr=58.3%)
- alpha=025_to_050|beta_abs=far_side|blockage=low (lift=1.004, n=8085, fail=1037, p=40.4%, fr=12.8%)

Current generator most sampled cells in rollout batch:
- alpha=050_to_075|beta_abs=far_side|blockage=low (cov=48.6%, n=9728)
- alpha=025_to_050|beta_abs=far_side|blockage=low (cov=40.4%, n=8085)
- alpha=075_to_goal|beta_abs=far_side|blockage=low (cov=5.4%, n=1078)
- alpha=after_goal|beta_abs=far_side|blockage=low (cov=2.0%, n=406)
- alpha=start_to_025|beta_abs=far_side|blockage=low (cov=1.1%, n=210)

Accepted/final candidate most sampled cells in verifier audit:
- alpha=025_to_050|beta_abs=far_side|blockage=low (cov=35.9%, n=358, ev=0.200)
- alpha=050_to_075|beta_abs=far_side|blockage=low (cov=32.2%, n=321, ev=0.000)
- alpha=after_goal|beta_abs=far_side|blockage=low (cov=13.8%, n=138, ev=0.000)
- alpha=075_to_goal|beta_abs=far_side|blockage=low (cov=5.0%, n=50, ev=0.000)
- alpha=025_to_050|beta_abs=near_side|blockage=medium (cov=2.7%, n=27, ev=1.000)

### Round 7 @ episode 70000

- batch success_rate: 0.806; stats: success=8065, collision=1823, timeout=110, fall=2
- trigger: `fixed_schedule(evolve_index=7/8, eps=70000, target=70000)|difficulty=too_easy(sr=0.806>0.800)`
- generator length: 17654 -> 17821 (+167)
- verifier/final audit: accepted=True, direction=increase, score=0.032 -> 0.069, delta=0.037, reason=candidate_increased_evidence_score, attempts=1

Top attribution cells:
- alpha=050_to_075|beta_abs=near_side|blockage=medium (lift=4.594, n=270, fail=240, p=1.4%, fr=88.9%)
- alpha=050_to_075|beta_abs=near_side|blockage=low (lift=4.204, n=252, fail=205, p=1.3%, fr=81.3%)
- alpha=025_to_050|beta_abs=near_side|blockage=medium (lift=4.187, n=390, fail=316, p=1.9%, fr=81.0%)
- alpha=050_to_075|beta_abs=far_side|blockage=medium (lift=3.661, n=24, fail=17, p=0.1%, fr=70.8%)
- alpha=025_to_050|beta_abs=near_side|blockage=low (lift=3.642, n=254, fail=179, p=1.3%, fr=70.5%)

Current generator most sampled cells in rollout batch:
- alpha=025_to_050|beta_abs=far_side|blockage=low (cov=35.9%, n=7179)
- alpha=050_to_075|beta_abs=far_side|blockage=low (cov=33.7%, n=6748)
- alpha=after_goal|beta_abs=far_side|blockage=low (cov=13.2%, n=2632)
- alpha=075_to_goal|beta_abs=far_side|blockage=low (cov=3.9%, n=778)
- alpha=start_to_025|beta_abs=far_side|blockage=low (cov=2.8%, n=555)

Accepted/final candidate most sampled cells in verifier audit:
- alpha=after_goal|beta_abs=far_side|blockage=low (cov=32.6%, n=322, ev=0.000)
- alpha=050_to_075|beta_abs=far_side|blockage=low (cov=21.7%, n=214, ev=0.000)
- alpha=025_to_050|beta_abs=far_side|blockage=low (cov=15.9%, n=157, ev=0.000)
- alpha=075_to_goal|beta_abs=far_side|blockage=low (cov=8.0%, n=79, ev=0.000)
- alpha=before_start|beta_abs=far_side|blockage=low (cov=4.8%, n=47, ev=0.000)

### Round 8 @ episode 80000

- batch success_rate: 0.783; stats: success=7826, collision=2043, timeout=130, fall=1
- trigger: `fixed_schedule(evolve_index=8/8, eps=80000, target=80000)|difficulty=balanced(sr=0.783, range=[0.200,0.800])`
- generator length: 17821 -> 15959 (-1862)
- verifier/final audit: accepted=False, direction=preserve, score=0.066 -> 0.051, delta=-0.016, reason=candidate_reduced_evidence_score, attempts=3

Top attribution cells:
- alpha=025_to_050|beta_abs=near_side|blockage=high (lift=4.169, n=32, fail=29, p=0.2%, fr=90.6%)
- alpha=050_to_075|beta_abs=near_side|blockage=medium (lift=4.112, n=839, fail=750, p=4.2%, fr=89.4%)
- alpha=050_to_075|beta_abs=near_side|blockage=high (lift=3.965, n=29, fail=25, p=0.1%, fr=86.2%)
- alpha=025_to_050|beta_abs=near_side|blockage=low (lift=3.947, n=162, fail=139, p=0.8%, fr=85.8%)
- alpha=050_to_075|beta_abs=near_side|blockage=low (lift=3.852, n=166, fail=139, p=0.8%, fr=83.7%)

Current generator most sampled cells in rollout batch:
- alpha=after_goal|beta_abs=far_side|blockage=low (cov=32.6%, n=6519)
- alpha=050_to_075|beta_abs=far_side|blockage=low (cov=21.2%, n=4242)
- alpha=025_to_050|beta_abs=far_side|blockage=low (cov=16.6%, n=3327)
- alpha=075_to_goal|beta_abs=far_side|blockage=low (cov=7.0%, n=1400)
- alpha=025_to_050|beta_abs=near_side|blockage=medium (cov=4.7%, n=949)

Accepted/final candidate most sampled cells in verifier audit:
- alpha=025_to_050|beta_abs=far_side|blockage=low (cov=29.0%, n=282, ev=0.000)
- alpha=050_to_075|beta_abs=far_side|blockage=low (cov=28.8%, n=280, ev=0.000)
- alpha=before_start|beta_abs=far_side|blockage=low (cov=11.6%, n=113, ev=0.000)
- alpha=after_goal|beta_abs=far_side|blockage=low (cov=10.8%, n=105, ev=0.000)
- alpha=075_to_goal|beta_abs=far_side|blockage=low (cov=6.5%, n=63, ev=0.000)

Rejected attempts:
- candidate_001_audit.json: delta=-0.067, reason=candidate_reduced_evidence_score, score=0.071->0.003
- candidate_002_audit.json: delta=-0.000, reason=candidate_reduced_evidence_score, score=0.065->0.064
- candidate_003_audit.json: delta=-0.016, reason=candidate_reduced_evidence_score, score=0.066->0.051

### Round 8 @ episode 90000

- batch success_rate: 0.794; stats: success=7941, collision=1885, timeout=171, fall=3
- trigger: `fixed_schedule(evolve_index=8/8, eps=90000, target=80000)|difficulty=balanced(sr=0.794, range=[0.200,0.800])`
- generator length: 17821 -> 15959 (-1862)
- verifier/final audit: accepted=True, direction=preserve, score=0.076 -> 0.087, delta=0.011, reason=candidate_preserved_evidence_score, attempts=2

Top attribution cells:
- alpha=050_to_075|beta_abs=near_side|blockage=high (lift=4.428, n=34, fail=31, p=0.2%, fr=91.2%)
- alpha=025_to_050|beta_abs=near_side|blockage=high (lift=4.234, n=39, fail=34, p=0.2%, fr=87.2%)
- alpha=050_to_075|beta_abs=near_side|blockage=medium (lift=4.207, n=777, fail=673, p=3.9%, fr=86.6%)
- alpha=025_to_050|beta_abs=near_side|blockage=medium (lift=3.938, n=915, fail=742, p=4.6%, fr=81.1%)
- alpha=050_to_075|beta_abs=near_side|blockage=low (lift=3.885, n=150, fail=120, p=0.8%, fr=80.0%)

Current generator most sampled cells in rollout batch:
- alpha=after_goal|beta_abs=far_side|blockage=low (cov=32.5%, n=6506)
- alpha=050_to_075|beta_abs=far_side|blockage=low (cov=21.1%, n=4222)
- alpha=025_to_050|beta_abs=far_side|blockage=low (cov=16.9%, n=3384)
- alpha=075_to_goal|beta_abs=far_side|blockage=low (cov=7.1%, n=1430)
- alpha=025_to_050|beta_abs=near_side|blockage=medium (cov=4.6%, n=915)

Accepted/final candidate most sampled cells in verifier audit:
- alpha=050_to_075|beta_abs=far_side|blockage=low (cov=24.1%, n=238, ev=0.000)
- alpha=025_to_050|beta_abs=far_side|blockage=low (cov=23.4%, n=231, ev=0.000)
- alpha=after_goal|beta_abs=far_side|blockage=low (cov=22.7%, n=224, ev=0.000)
- alpha=before_start|beta_abs=far_side|blockage=low (cov=6.4%, n=63, ev=0.000)
- alpha=025_to_050|beta_abs=near_side|blockage=medium (cov=5.4%, n=53, ev=0.700)

Attempts:
- candidate_001_audit.json: accepted=False, delta=-0.077, reason=candidate_reduced_evidence_score
- candidate_002_audit.json: accepted=True, delta=0.011, reason=candidate_preserved_evidence_score
