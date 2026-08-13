"""Unit tests for the band-based acceptance judgement (Phase ① rewrite).

These pin down the new ``evaluate_learnable_frontier_shift`` logic with
hand-constructed current/candidate profiles -- no RL, no GPU, milliseconds.

Five scenarios:
  1. in-band lateral variation (same difficulty, more diverse) -> ACCEPT
  2. over-hardening (boundary mass lost to the hard end, still under cap) -> REJECT
  3. drift to trivial (more layouts solved by every skill) -> REJECT
  4. infeasible (too many unsolvable scenes, over cap) -> REJECT
  5. coverage collapse (in band, but layouts collapse to one region) -> REJECT
"""

from DIVO.curriculum.learnable_frontier import evaluate_learnable_frontier_shift


def _profile(frac_infeasible, frac_trivial, frac_boundary, coverage_per_scene,
             mean_realized=0.5, mean_lv=0.25):
    return {
        "frac_infeasible": frac_infeasible,
        "frac_trivial": frac_trivial,
        "frac_boundary": frac_boundary,
        "boundary_coverage": {"coverage_per_scene": coverage_per_scene},
        "mean_realized": mean_realized,
        "mean_lv": mean_lv,
    }


CAP = 0.30


def test_in_band_lateral_variation_accepted():
    # Same difficulty distribution, strictly more diverse layouts in the band.
    current = _profile(frac_infeasible=0.0, frac_trivial=0.10, frac_boundary=0.90,
                       coverage_per_scene=1.0)
    candidate = _profile(frac_infeasible=0.0, frac_trivial=0.10, frac_boundary=0.90,
                         coverage_per_scene=1.6)
    out = evaluate_learnable_frontier_shift(current, candidate, infeasible_cap=CAP)
    assert out["accepted"] is True
    assert out["reason"] == "candidate_kept_learnable_band"


def test_over_hardening_loses_boundary_mass_rejected():
    # Mass moved out of the band toward the hard end, but infeasible stays under cap.
    current = _profile(frac_infeasible=0.05, frac_trivial=0.10, frac_boundary=0.85,
                       coverage_per_scene=1.2)
    candidate = _profile(frac_infeasible=0.20, frac_trivial=0.10, frac_boundary=0.70,
                         coverage_per_scene=1.2)
    out = evaluate_learnable_frontier_shift(current, candidate, infeasible_cap=CAP)
    assert out["accepted"] is False
    assert out["reason"] == "candidate_lost_boundary_mass"


def test_drift_to_trivial_rejected():
    # More layouts become trivially solved by every skill.
    current = _profile(frac_infeasible=0.0, frac_trivial=0.10, frac_boundary=0.90,
                       coverage_per_scene=1.2)
    candidate = _profile(frac_infeasible=0.0, frac_trivial=0.40, frac_boundary=0.60,
                         coverage_per_scene=1.2)
    out = evaluate_learnable_frontier_shift(current, candidate, infeasible_cap=CAP)
    assert out["accepted"] is False
    assert out["reason"] == "candidate_drifted_to_trivial"


def test_infeasible_cap_exceeded_rejected():
    # Too many unsolvable scenes (over the cap).
    current = _profile(frac_infeasible=0.05, frac_trivial=0.10, frac_boundary=0.85,
                       coverage_per_scene=1.2)
    candidate = _profile(frac_infeasible=0.50, frac_trivial=0.10, frac_boundary=0.40,
                         coverage_per_scene=1.2)
    out = evaluate_learnable_frontier_shift(current, candidate, infeasible_cap=CAP)
    assert out["accepted"] is False
    assert out["reason"] == "candidate_infeasible_cap_exceeded"


def test_coverage_collapse_rejected():
    # In band and feasible, but layout diversity collapses to one region.
    current = _profile(frac_infeasible=0.0, frac_trivial=0.10, frac_boundary=0.90,
                       coverage_per_scene=1.5)
    candidate = _profile(frac_infeasible=0.0, frac_trivial=0.10, frac_boundary=0.90,
                         coverage_per_scene=0.4)
    out = evaluate_learnable_frontier_shift(current, candidate, infeasible_cap=CAP)
    assert out["accepted"] is False
    assert out["reason"] == "candidate_collapsed_coverage"
