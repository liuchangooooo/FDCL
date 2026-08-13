from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence


# Kept for backward compatibility with older imports; no longer used to pull the
# difficulty center toward a single point. The target is the learnable *band*
# (some injected skills solve a scene, some fail), not realized == 0.5.
FRONTIER_REALIZED = 0.5

DEFAULT_INFEASIBLE_CAP = 0.30
DEFAULT_EPS = 1e-6


def evaluate_learnable_frontier_shift(
    current: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    infeasible_cap: Optional[float] = DEFAULT_INFEASIBLE_CAP,
    eps: float = DEFAULT_EPS,
    # Deprecated / ignored. Retained so existing callers that still pass these
    # keep working until they are migrated. The acceptance rule no longer
    # requires learning-value to strictly increase, nor pulls realized to 0.5.
    min_lv_delta: float = 0.0,
    frontier_realized: float = FRONTIER_REALIZED,
) -> Dict[str, Any]:
    """Judge whether a candidate generator keeps the training distribution inside
    the skill-library learnable band.

    The rule is a set of *guards* (do not make the distribution worse), not a
    "strictly better" objective. A candidate is accepted when, relative to the
    current generator on the same paired probe, it:

    1. stays under the infeasible cap (not too many unsolvable scenes),
    2. does not drift toward trivial (frac_trivial not worse),
    3. keeps at least as much mass in the learnable band (frac_boundary not worse),
    4. does not collapse layout diversity within the band (coverage not worse).

    Forward progress (rising difficulty) comes from the policy improving between
    rounds and the generator re-centering on the moving boundary, both of which
    these "not worse" guards allow -- including difficulty-neutral, diversity-
    improving lateral variations that the old strict learning-value rule rejected.

    The generator-validity gate is intentionally handled by the caller (verifier)
    before this function is reached.
    """

    cur_inf = _as_float(current.get("frac_infeasible", 0.0))
    cand_inf = _as_float(candidate.get("frac_infeasible", 0.0))
    cur_triv = _as_float(current.get("frac_trivial", 0.0))
    cand_triv = _as_float(candidate.get("frac_trivial", 0.0))
    cur_bound = _as_float(current.get("frac_boundary", 0.0))
    cand_bound = _as_float(candidate.get("frac_boundary", 0.0))
    cur_cov = _coverage_per_scene(current)
    cand_cov = _coverage_per_scene(candidate)

    # Retained only for logging/reporting; not part of the decision.
    cur_lv = _as_float(current.get("mean_lv", current.get("score", 0.0)))
    cand_lv = _as_float(candidate.get("mean_lv", candidate.get("score", 0.0)))
    cur_realized = _as_float(current.get("mean_realized", 0.0))
    cand_realized = _as_float(candidate.get("mean_realized", 0.0))

    eps = float(eps)
    infeasible_cap_ok = (
        True if infeasible_cap is None else bool(cand_inf <= float(infeasible_cap) + eps)
    )
    trivial_not_worse = bool(cand_triv <= cur_triv + eps)
    boundary_not_worse = bool(cand_bound >= cur_bound - eps)
    coverage_not_worse = bool(cand_cov >= cur_cov - eps)

    accepted = bool(
        infeasible_cap_ok
        and trivial_not_worse
        and boundary_not_worse
        and coverage_not_worse
    )
    reason = _band_reason(
        accepted=accepted,
        infeasible_cap_ok=infeasible_cap_ok,
        trivial_not_worse=trivial_not_worse,
        boundary_not_worse=boundary_not_worse,
        coverage_not_worse=coverage_not_worse,
    )

    return {
        "accepted": accepted,
        "reason": reason,
        "infeasible_cap": None if infeasible_cap is None else float(infeasible_cap),
        # band-mass gates
        "current_frac_boundary": float(cur_bound),
        "candidate_frac_boundary": float(cand_bound),
        "frac_boundary_delta": float(cand_bound - cur_bound),
        "current_frac_trivial": float(cur_triv),
        "candidate_frac_trivial": float(cand_triv),
        "frac_trivial_delta": float(cand_triv - cur_triv),
        "current_frac_infeasible": float(cur_inf),
        "candidate_frac_infeasible": float(cand_inf),
        "frac_infeasible_delta": float(cand_inf - cur_inf),
        # coverage / anti-collapse gate
        "current_coverage_per_scene": float(cur_cov),
        "candidate_coverage_per_scene": float(cand_cov),
        "coverage_delta": float(cand_cov - cur_cov),
        # decision flags
        "infeasible_cap_ok": infeasible_cap_ok,
        "trivial_not_worse": trivial_not_worse,
        "boundary_not_worse": boundary_not_worse,
        "coverage_not_worse": coverage_not_worse,
        # reporting only (not used for the decision)
        "current_mean_lv": float(cur_lv),
        "candidate_mean_lv": float(cand_lv),
        "mean_lv_delta": float(cand_lv - cur_lv),
        "current_mean_realized": float(cur_realized),
        "candidate_mean_realized": float(cand_realized),
        "mean_realized_delta": float(cand_realized - cur_realized),
    }


def _band_reason(
    *,
    accepted: bool,
    infeasible_cap_ok: bool,
    trivial_not_worse: bool,
    boundary_not_worse: bool,
    coverage_not_worse: bool,
) -> str:
    if accepted:
        return "candidate_kept_learnable_band"
    # Priority: hard feasibility issue first, then difficulty drift, then collapse.
    if not infeasible_cap_ok:
        return "candidate_infeasible_cap_exceeded"
    if not trivial_not_worse:
        return "candidate_drifted_to_trivial"
    if not boundary_not_worse:
        return "candidate_lost_boundary_mass"
    return "candidate_collapsed_coverage"


def _coverage_per_scene(profile: Mapping[str, Any]) -> float:
    """Extract per-boundary-scene coverage from a profile.

    Accepts either a nested ``boundary_coverage`` dict (as produced by
    ``skill_signal.compute_difficulty_profile``) or a flat ``coverage_per_scene``
    field. Missing coverage is treated as 0.0.
    """

    cov = profile.get("boundary_coverage")
    if isinstance(cov, Mapping):
        return _as_float(cov.get("coverage_per_scene", 0.0))
    return _as_float(profile.get("coverage_per_scene", 0.0))


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


# ===========================================================================
# Stage 2 V0: single-scalar boundary selection + symmetric saturation escape
# (Task 17 / Requirement 8). Replaces the multi-axis "not-worse" conjunction
# above, which ratcheted to a deadlock at maturity (evolve_index=17 stall). The
# new rule has a single scalar objective (boundary_count) that always has an
# argmax, an ABSOLUTE hard gate (not "better than G_t"), and a saturation escape
# so a library-solved generator cannot block harder generators.
# ===========================================================================

DEFAULT_V_MIN = 0.8
DEFAULT_DUP_MAX = 0.25
DEFAULT_MIN_BOUNDARY_COUNT_DELTA = 4
DEFAULT_R_EASY_MAX = 0.8
DEFAULT_R_HARD_MAX = 0.8


def boundary_hard_gate(
    signal: Mapping[str, Any],
    v_min: float = DEFAULT_V_MIN,
    d_max: float = DEFAULT_DUP_MAX,
    code_pass: bool = True,
) -> Dict[str, Any]:
    """Absolute hard gate on an LLM candidate (Requirement 8.2).

    Only filters candidates for *usability*; it is NOT a "better than G_t"
    comparison and ``boundary_rate`` is NOT a gate. The current G_t is exempt
    (always an incumbent) -- callers pass candidates here, not G_t.
    """
    valid_rate = _as_float(signal.get("valid_rate", 0.0))
    dup = _as_float(signal.get("duplicate_rate", 1.0))
    checks = {
        "code_pass": bool(code_pass),
        "valid_rate_ok": bool(valid_rate >= float(v_min)),
        "duplicate_rate_ok": bool(dup <= float(d_max)),
    }
    return {"passed": all(checks.values()), "checks": checks}


def is_saturated(
    signal: Mapping[str, Any],
    r_easy_max: float = DEFAULT_R_EASY_MAX,
    r_hard_max: float = DEFAULT_R_HARD_MAX,
) -> Dict[str, Any]:
    """Symmetric saturation test (Requirement 8.5).

    r_easy high  -> library solved the generator (too easy)   -> HARDEN escape.
    r_hard high  -> library cannot solve the generator (hard)  -> RELAX escape.
    """
    r_easy = _as_float(signal.get("r_easy", 0.0))
    r_hard = _as_float(signal.get("r_hard", 0.0))
    if r_easy >= float(r_easy_max):
        return {"saturated": True, "direction": "HARDEN", "trigger": "r_easy"}
    if r_hard >= float(r_hard_max):
        return {"saturated": True, "direction": "RELAX", "trigger": "r_hard"}
    return {"saturated": False, "direction": None, "trigger": None}


def _boundary_key(signal: Mapping[str, Any]) -> tuple:
    """argmax key: boundary_count primary, mean_b tiebreak (Requirement 8.3)."""
    return (int(signal.get("boundary_count", 0)), _as_float(signal.get("mean_b", 0.0)))


def select_generator_boundary(
    current_signal: Mapping[str, Any],
    candidate_signals: Sequence[Mapping[str, Any]],
    *,
    v_min: float = DEFAULT_V_MIN,
    d_max: float = DEFAULT_DUP_MAX,
    min_boundary_count_delta: int = DEFAULT_MIN_BOUNDARY_COUNT_DELTA,
    r_easy_max: float = DEFAULT_R_EASY_MAX,
    r_hard_max: float = DEFAULT_R_HARD_MAX,
    code_pass_flags: Optional[Sequence[bool]] = None,
    candidate_fresh_flags: Optional[Sequence[bool]] = None,
    diversify_on_hold: bool = False,
    diversify_bc_tolerance: int = 2,
    diversify_prefer_lower_easy: bool = False,
    diversify_r_easy_soft: float = 0.5,
    diversify_easy_eps: float = 0.05,
) -> Dict[str, Any]:
    """Single-scalar generator selection over {G_t} ∪ {passing candidates}.

    Rules (Requirement 8.3/8.4/8.5):
      * candidates must pass the absolute hard gate; G_t never does (incumbent);
      * objective J = boundary_count (mean_b tiebreak); argmax always exists;
      * NOT saturated: replace only if a candidate reaches
        ``boundary_count >= Gt_boundary_count + min_boundary_count_delta``;
        otherwise optionally rotate to a fresh, equal-difficulty candidate when
        ``diversify_on_hold`` is enabled;
      * saturated (r_easy or r_hard maxed): G_t no longer blocks replacement,
        but candidates must move in the requested HARDEN/RELAX direction without
        crossing into the opposite saturated tail. Pick the best directional
        candidate by boundary_count; if none qualifies, hold.
    """
    sat = is_saturated(current_signal, r_easy_max=r_easy_max, r_hard_max=r_hard_max)
    cur_bc = int(current_signal.get("boundary_count", 0))

    gate_reports = []
    passing = []
    for i, cand in enumerate(candidate_signals):
        cp = True if code_pass_flags is None else bool(code_pass_flags[i])
        fresh = True if candidate_fresh_flags is None else bool(candidate_fresh_flags[i])
        gate = boundary_hard_gate(cand, v_min=v_min, d_max=d_max, code_pass=cp)
        gate_reports.append({"index": int(i), **gate, "fresh_code_ok": fresh})
        if gate["passed"] and fresh:
            passing.append((int(i), cand))

    decision: Dict[str, Any] = {
        "saturated": bool(sat["saturated"]),
        "saturation_direction": sat["direction"],
        "saturation_trigger": sat["trigger"],
        "current_boundary_count": cur_bc,
        "current_mean_b": _as_float(current_signal.get("mean_b", 0.0)),
        "n_candidates": int(len(candidate_signals)),
        "n_passing": int(len(passing)),
        "gate_reports": gate_reports,
        "min_boundary_count_delta": int(min_boundary_count_delta),
    }

    if not passing:
        hard_gate_passed = any(report["passed"] for report in gate_reports)
        no_fresh = hard_gate_passed and not any(
            report["passed"] and report["fresh_code_ok"] for report in gate_reports
        )
        decision.update({
            "action": "hold",
            "reason": (
                "saturated_but_no_fresh_candidate"
                if sat["saturated"] and no_fresh
                else "no_fresh_candidate"
                if no_fresh
                else "saturated_but_no_candidate_passed_hard_gate"
                if sat["saturated"]
                else "no_candidate_passed_hard_gate"
            ),
            "chosen_index": None,
            "chosen_boundary_count": cur_bc,
        })
        return decision

    if sat["saturated"]:
        cur_hard = _as_float(current_signal.get("r_hard", 0.0))
        cur_easy = _as_float(current_signal.get("r_easy", 0.0))
        cur_n = max(1, int(current_signal.get("n_scenes", 0)))
        directional = []
        directional_reports = []
        for i, cand in passing:
            cand_hard = _as_float(cand.get("r_hard", 0.0))
            cand_easy = _as_float(cand.get("r_easy", 0.0))
            cand_n = max(1, int(cand.get("n_scenes", 0)))
            cur_hard_count = int(round(cur_hard * cur_n))
            cur_easy_count = int(round(cur_easy * cur_n))
            cand_hard_count = int(round(cand_hard * cand_n))
            cand_easy_count = int(round(cand_easy * cand_n))
            if sat["direction"] == "HARDEN":
                direction_ok = cand_easy < cur_easy
                opposite_tail_ok = cand_hard_count <= cur_hard_count + 1
            else:
                direction_ok = cand_hard < cur_hard
                opposite_tail_ok = cand_easy_count <= cur_easy_count + 1
            passed = bool(direction_ok and opposite_tail_ok)
            directional_reports.append({
                "index": int(i),
                "passed": passed,
                "direction_ok": bool(direction_ok),
                "opposite_tail_ok": bool(opposite_tail_ok),
                "opposite_tail_count_tolerance": 1,
            })
            if passed:
                directional.append((i, cand))

        decision["n_directional_passing"] = int(len(directional))
        decision["directional_reports"] = directional_reports
        if not directional:
            decision.update({
                "action": "hold",
                "reason": "saturated_but_no_directional_candidate",
                "chosen_index": None,
                "chosen_boundary_count": cur_bc,
            })
            return decision

        best_i, best_cand = max(directional, key=lambda t: _boundary_key(t[1]))
        best_bc = int(best_cand.get("boundary_count", 0))
        decision.update({
            "action": "replace",
            "reason": f"saturation_escape_{sat['direction']}",
            "chosen_index": best_i,
            "chosen_boundary_count": best_bc,
        })
        return decision

    best_i, best_cand = max(passing, key=lambda t: _boundary_key(t[1]))
    best_bc = int(best_cand.get("boundary_count", 0))

    # Not saturated: G_t is incumbent; require a net-improvement margin.
    if best_bc >= cur_bc + int(min_boundary_count_delta):
        decision.update({
            "action": "replace",
            "reason": "boundary_count_net_improve",
            "chosen_index": best_i,
            "chosen_boundary_count": best_bc,
        })
        return decision

    # Not saturated AND no net boundary improvement -> would normally HOLD.
    # PRESERVE_AND_DIVERSIFY swap (points 2/3, DEFAULT OFF via diversify_on_hold):
    # instead of freezing G_t, rotate to a code-fresh candidate at
    # *equal* difficulty so the training distribution keeps diversifying without
    # drifting easy. Eligible candidate must (a) pass the hard gate (already in
    # `passing`), (b) keep difficulty: boundary_count >= cur_bc - tolerance,
    # (c) NOT be easier: r_easy <= cur r_easy + eps (guards against easy-drift).
    if diversify_on_hold:
        cur_easy = _as_float(current_signal.get("r_easy", 0.0))
        eligible = [
            (i, c) for (i, c) in passing
            if int(c.get("boundary_count", 0)) >= cur_bc - int(diversify_bc_tolerance)
            and _as_float(c.get("r_easy", 0.0)) <= cur_easy + float(diversify_easy_eps)
        ]
        if eligible:
            if diversify_prefer_lower_easy and cur_easy > float(diversify_r_easy_soft):
                # Point 3: r_easy elevated -> gently re-harden by preferring the
                # LOWEST-r_easy candidate among equal-difficulty ones (tiebreak:
                # higher boundary_count). boundary_count never drops below
                # cur_bc - tolerance, so this cannot over-harden into too-hard.
                div_i, div_cand = min(
                    eligible,
                    key=lambda t: (
                        _as_float(t[1].get("r_easy", 0.0)),
                        -int(t[1].get("boundary_count", 0)),
                    ),
                )
                div_reason = "diversify_reduce_easy"
            else:
                # Point 2: keep difficulty, rotate structure -> nearest-to-frontier
                # (highest boundary_count, mean_b tiebreak) among equal-difficulty.
                div_i, div_cand = max(eligible, key=lambda t: _boundary_key(t[1]))
                div_reason = "diversify_preserve_difficulty"
            decision.update({
                "action": "replace",
                "reason": div_reason,
                "chosen_index": div_i,
                "chosen_boundary_count": int(div_cand.get("boundary_count", 0)),
                "diversify": True,
            })
            return decision

    decision.update({
        "action": "hold",
        "reason": "no_net_boundary_improvement",
        "chosen_index": None,
        "chosen_boundary_count": cur_bc,
    })
    return decision
