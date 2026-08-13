"""Stage 1 go/no-go evaluation (Task 13 / Requirements 10.3, 10.5, 10.6, 11.x).

Runs the fixed difficulty ladder (Task 12) through the skill library and reports
the Route-B go/no-go signals:

  - per-scene p_i = fraction of PROBE skills (w_1..w_K) that solve the scene, and
    the library ``realized`` (== mean p_i);
  - non-degenerate p_i stratification (easy ~1, hard ~0, mid 0<p<1);
  - deployment (w_0) success on the ladder;
  - per-skill non-trivial coverage (Progress >= tau_p AND success-rate gate);
  - hard gates: B/M/U >= baseline - delta, w_0 not dropped, K_eff >= 0.5K,
    non-trivial skill fraction >= r_s.

B/M/U/D themselves come from the standard evaluation harness (evaluation.py);
best checkpoints are selected by ``test_mean_score`` (never by B/M/U/D). The
aggregation functions here are env-free and unit-tested; ``run_stage1_probe``
needs a live Push-T env (smoke run).
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from DIVO.curriculum.stage1_ladder import LADDER_CATEGORIES


# Categories whose p_i is expected high / mid / low (for the non-degenerate check).
# Supports both the sampled tiers (easy/mid/hard) and the legacy hand-placed
# ladder categories.
_EASY_CATS = ("easy",)
_MID_CATS = ("mid", "two-route", "left-preferred", "right-preferred")
_HARD_CATS = ("hard", "narrow", "near-goal")


def _layout_is_valid(env: Any, start: Sequence[float], obstacles: Sequence[Mapping[str, Any]]) -> bool:
    """Physical validity of a layout = no initial contact after reset (Bug 3 fix).

    We deliberately do NOT use ``is_obstacle_config_valid`` here: that is an
    LLM-authoring constraint (keep obstacles clear of start/target T-blocks) which
    is stricter than physical feasibility and wrongly rejects env-generated
    harder scenes (obstacles near the path). The honest validity test is that the
    scene has no initial overlap (get_ncon == 0).
    """
    import numpy as _np
    start_arr = _np.asarray(start, dtype=_np.float64)
    try:
        if hasattr(env, "set_obstacle_config"):
            env.set_obstacle_config(list(obstacles))
        if hasattr(env, "reset"):
            env.reset(tblock_pos=start_arr, force_tblock_pos=True)
        if hasattr(env, "get_ncon") and int(env.get_ncon()) != 0:
            return False
    except Exception:
        return False
    return True


def per_scene_p(skill_successes: Sequence[int]) -> float:
    """p_i = fraction of probe skills that solve scene i."""
    arr = np.asarray(list(skill_successes), dtype=np.float64)
    return float(arr.mean()) if arr.size else 0.0


def aggregate_by_category(results: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Mean p / deploy success per category."""
    out: Dict[str, Dict[str, float]] = {}
    categories = []
    for r in results:
        c = r.get("category")
        if c not in categories:
            categories.append(c)
    for category in categories:
        rows = [r for r in results if r.get("category") == category]
        if not rows:
            continue
        out[category] = {
            "mean_p": float(np.mean([r["p"] for r in rows])),
            "mean_deploy": float(np.mean([r["deploy_success"] for r in rows])),
            "n": int(len(rows)),
        }
    return out


def check_p_distribution(
    category_means: Mapping[str, Mapping[str, float]],
    easy_hi: float = 0.8,
    hard_lo: float = 0.2,
) -> Dict[str, Any]:
    """Non-degenerate p_i stratification check (Requirement 10.6).

    Passes when there exist easy scenes with high p, hard scenes with low p, and
    (implied) intermediate scenes in between -- i.e. the library is neither
    all-solving nor all-failing on the ladder.
    """
    easy_p = [category_means[c]["mean_p"] for c in _EASY_CATS if c in category_means]
    mid_p = [category_means[c]["mean_p"] for c in _MID_CATS if c in category_means]
    hard_p = [category_means[c]["mean_p"] for c in _HARD_CATS if c in category_means]

    has_easy = any(p >= easy_hi for p in easy_p)
    has_hard = any(p <= hard_lo for p in hard_p)
    has_mid = any(hard_lo < p < easy_hi for p in (mid_p + easy_p + hard_p))
    return {
        "non_degenerate": bool(has_easy and has_hard and has_mid),
        "has_easy_high": bool(has_easy),
        "has_hard_low": bool(has_hard),
        "has_mid": bool(has_mid),
        "easy_p": easy_p,
        "mid_p": mid_p,
        "hard_p": hard_p,
    }


def per_skill_stats(results: Sequence[Mapping[str, Any]], K: int) -> Dict[str, Any]:
    """Per-skill success rate and progress across the ladder.

    ``results[i]["skill_success"]`` and ``["skill_progress"]`` are length-K lists
    for probe skills w_1..w_K.
    """
    succ = np.zeros(int(K), dtype=np.float64)
    prog = np.zeros(int(K), dtype=np.float64)
    n = 0
    for r in results:
        s = np.asarray(r.get("skill_success", []), dtype=np.float64)
        p = np.asarray(r.get("skill_progress", []), dtype=np.float64)
        if s.size == int(K):
            succ += s
            prog += p
            n += 1
    if n == 0:
        return {"success_rate": [0.0] * int(K), "progress": [0.0] * int(K)}
    return {"success_rate": (succ / n).tolist(), "progress": (prog / n).tolist()}


def nontrivial_skill_fraction(
    per_skill: Mapping[str, Any],
    tau_s: float = 0.3,
    tau_p: float = 0.0,
) -> float:
    """Fraction of probe skills that are non-trivial (Requirement 10.3.a/b).

    A skill counts as non-trivial when its ladder success-rate >= tau_s AND its
    mean progress >= tau_p.
    """
    sr = np.asarray(per_skill.get("success_rate", []), dtype=np.float64)
    pr = np.asarray(per_skill.get("progress", []), dtype=np.float64)
    if sr.size == 0:
        return 0.0
    ok = (sr >= float(tau_s)) & (pr >= float(tau_p))
    return float(np.mean(ok))


def stage1_gonogo(
    results: Sequence[Mapping[str, Any]],
    K: int,
    k_eff: float,
    bmu: Optional[Mapping[str, float]] = None,
    bmu_baseline: Optional[Mapping[str, float]] = None,
    deploy_val: Optional[float] = None,
    deploy_baseline: Optional[float] = None,
    *,
    delta: float = 0.05,
    r_s: float = 0.5,
    tau_s: float = 0.3,
    tau_p: float = 0.0,
) -> Dict[str, Any]:
    """Combine the Stage 1 hard gates (Requirement 10.3, 10.3.a/b, 10.4).

    Hard gates:
      - K_eff >= 0.5 * K
      - non-trivial skill fraction >= r_s
      - (if provided) B/M/U >= baseline - delta
      - (if provided) deployment (w_0) not significantly dropped
    D (dead-end) is an observation only at Stage 1 (Requirement 10.4).
    """
    cat_means = aggregate_by_category(results)
    p_dist = check_p_distribution(cat_means)
    per_skill = per_skill_stats(results, K)
    nt_frac = nontrivial_skill_fraction(per_skill, tau_s=tau_s, tau_p=tau_p)

    keff_ok = bool(k_eff >= 0.5 * int(K))
    nt_ok = bool(nt_frac >= float(r_s))

    bmu_ok = True
    if bmu is not None and bmu_baseline is not None:
        bmu_ok = all(
            float(bmu.get(k, 0.0)) >= float(bmu_baseline.get(k, 0.0)) - float(delta)
            for k in ("B", "M", "U")
        )
    deploy_ok = True
    if deploy_val is not None and deploy_baseline is not None:
        deploy_ok = bool(deploy_val >= deploy_baseline - float(delta))

    passed = bool(keff_ok and nt_ok and bmu_ok and deploy_ok)
    return {
        "passed": passed,
        "gates": {
            "k_eff_ok": keff_ok,
            "nontrivial_ok": nt_ok,
            "bmu_ok": bmu_ok,
            "deploy_ok": deploy_ok,
        },
        "k_eff": float(k_eff),
        "nontrivial_fraction": float(nt_frac),
        "p_distribution": p_dist,
        "category_means": cat_means,
        "per_skill": per_skill,
    }


def run_stage1_probe(
    env: Any,
    policy: Any,
    ladder: Mapping[str, List[Mapping[str, Any]]],
    K: int,
    device: Any,
    max_steps: int = 10,
) -> List[Dict[str, Any]]:
    """Roll out deploy (w_0) and probe skills (w_1..w_K) over the ladder.

    Returns one record per layout with p, realized, deploy success, and per-skill
    success/progress. Requires a live Push-T env (smoke run).
    """
    from DIVO.curriculum.skill_signal import rollout_fixed_skill
    from DIVO.curriculum.stage1_ladder import flatten_ladder

    layouts = flatten_ladder(ladder) if isinstance(ladder, dict) else list(ladder)
    records: List[Dict[str, Any]] = []
    n_invalid = 0
    for layout in layouts:
        start = layout["start"]
        obstacles = layout.get("obstacles", [])
        # Bug fix: filter physically invalid ladder layouts (obstacle overlaps
        # start/goal or initial contact) instead of scoring them as p=0, which
        # would corrupt the difficulty stratification and per-skill stats.
        if not _layout_is_valid(env, start, obstacles):
            n_invalid += 1
            continue
        deploy = rollout_fixed_skill(env, policy, start, obstacles, 0, device, max_steps)
        skill_success = []
        skill_progress = []
        skill_states = []
        for k in range(1, int(K) + 1):
            r = rollout_fixed_skill(env, policy, start, obstacles, k, device, max_steps)
            skill_success.append(1 if r["success"] else 0)
            skill_progress.append(float(r["quality"].get("progress", 0.0)))
            skill_states.append(r["states"])
        records.append({
            "category": layout.get("category", "?"),
            "start": start,
            "obstacles": obstacles,
            "p": per_scene_p(skill_success),
            "realized": per_scene_p(skill_success),
            "deploy_success": 1 if deploy["success"] else 0,
            "skill_success": skill_success,
            "skill_progress": skill_progress,
            "skill_states": skill_states,
        })
    if n_invalid:
        print(f"[stage1_probe] skipped {n_invalid}/{len(layouts)} physically-invalid ladder layouts")
    return records
