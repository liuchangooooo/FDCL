from DIVO.curriculum.learnable_frontier import select_generator_boundary
from DIVO.curriculum.skill_signal import build_design_context


def _scene(scene_id, realized, invalid=False):
    return {
        "scene_id": scene_id,
        "start": [0.1, 0.1, 0.0],
        "obstacles": [{"x": 0.05, "y": 0.02, "purpose": "test"}],
        "feasible": int(realized > 0.0),
        "realized": realized,
        "deployed": int(realized >= 0.5),
        "lv": realized * (1.0 - realized),
        "routes": [],
        "invalid": invalid,
    }


def _signal(boundary_count, r_hard, r_easy, n_scenes=30):
    return {
        "boundary_count": boundary_count,
        "mean_b": 0.1,
        "r_hard": r_hard,
        "r_easy": r_easy,
        "valid_rate": 1.0,
        "duplicate_rate": 0.0,
        "n_scenes": n_scenes,
    }


def test_design_context_strictly_separates_hard_boundary_and_easy():
    context = build_design_context(
        per_scene=[
            _scene(0, 0.0),
            _scene(1, 0.25),
            _scene(2, 0.5),
            _scene(3, 0.75),
            _scene(4, 1.0),
            _scene(5, 0.0, invalid=True),
        ],
        n=4,
        env=None,
        route_waypoints=0,
        include_behavior=False,
        tau_saturation=0.125,
        strict_boundary_bins=True,
    )

    assert [row["realized"] for row in context["focus"]] == [0.5, 0.25, 0.75]
    assert [row["realized"] for row in context["harden"]] == [1.0]
    assert [row["realized"] for row in context["avoid"]] == [0.0]


def test_harden_escape_rejects_candidate_that_is_easier():
    current = _signal(boundary_count=1, r_hard=0.0, r_easy=0.90)
    easier = _signal(boundary_count=4, r_hard=0.0, r_easy=0.93)

    decision = select_generator_boundary(current, [easier])

    assert decision["action"] == "hold"
    assert decision["reason"] == "saturated_but_no_directional_candidate"


def test_harden_escape_accepts_easy_to_boundary_without_overhardening():
    current = _signal(boundary_count=1, r_hard=0.0, r_easy=0.90)
    candidate = _signal(boundary_count=4, r_hard=0.03, r_easy=0.80)

    decision = select_generator_boundary(current, [candidate])

    assert decision["action"] == "replace"
    assert decision["reason"] == "saturation_escape_HARDEN"


def test_harden_escape_rejects_candidate_that_overshoots_to_hard():
    current = _signal(boundary_count=1, r_hard=0.0, r_easy=0.90)
    candidate = _signal(boundary_count=4, r_hard=0.10, r_easy=0.70)

    decision = select_generator_boundary(current, [candidate])

    assert decision["action"] == "hold"
    assert decision["reason"] == "saturated_but_no_directional_candidate"


def test_relax_escape_rejects_candidate_that_is_harder():
    current = _signal(boundary_count=1, r_hard=0.90, r_easy=0.0)
    harder = _signal(boundary_count=4, r_hard=0.93, r_easy=0.0)

    decision = select_generator_boundary(current, [harder])

    assert decision["action"] == "hold"
    assert decision["reason"] == "saturated_but_no_directional_candidate"


def test_relax_escape_accepts_hard_to_boundary_without_overrelaxing():
    current = _signal(boundary_count=1, r_hard=0.90, r_easy=0.0)
    candidate = _signal(boundary_count=4, r_hard=0.80, r_easy=0.03)

    decision = select_generator_boundary(current, [candidate])

    assert decision["action"] == "replace"
    assert decision["reason"] == "saturation_escape_RELAX"


def test_diversify_on_hold_rotates_to_fresh_equal_difficulty_candidate():
    current = _signal(boundary_count=29, r_hard=0.0, r_easy=0.03)
    candidate = _signal(boundary_count=28, r_hard=0.03, r_easy=0.03)

    decision = select_generator_boundary(
        current,
        [candidate],
        candidate_fresh_flags=[True],
        diversify_on_hold=True,
        diversify_bc_tolerance=2,
        diversify_easy_eps=0.05,
    )

    assert decision["action"] == "replace"
    assert decision["reason"] == "diversify_preserve_difficulty"


def test_diversify_on_hold_rejects_identical_generator_code():
    current = _signal(boundary_count=29, r_hard=0.0, r_easy=0.03)
    candidate = _signal(boundary_count=28, r_hard=0.03, r_easy=0.03)

    decision = select_generator_boundary(
        current,
        [candidate],
        candidate_fresh_flags=[False],
        diversify_on_hold=True,
    )

    assert decision["action"] == "hold"
    assert decision["reason"] == "no_fresh_candidate"


def test_diversify_on_hold_keeps_legacy_hold_when_disabled():
    current = _signal(boundary_count=29, r_hard=0.0, r_easy=0.03)
    candidate = _signal(boundary_count=28, r_hard=0.03, r_easy=0.03)

    decision = select_generator_boundary(
        current,
        [candidate],
        candidate_fresh_flags=[True],
        diversify_on_hold=False,
    )

    assert decision["action"] == "hold"
    assert decision["reason"] == "no_net_boundary_improvement"


def test_diversify_on_hold_rejects_candidate_that_drifts_easier():
    current = _signal(boundary_count=20, r_hard=0.10, r_easy=0.10)
    candidate = _signal(boundary_count=19, r_hard=0.03, r_easy=0.20)

    decision = select_generator_boundary(
        current,
        [candidate],
        candidate_fresh_flags=[True],
        diversify_on_hold=True,
        diversify_easy_eps=0.05,
    )

    assert decision["action"] == "hold"
    assert decision["reason"] == "no_net_boundary_improvement"
