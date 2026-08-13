"""Skill-library learnability signal for Navigation.

The probe protocol mirrors Push-T: all skills share the same scene/start,
``p`` excludes deployment skill ``w_0``, and representative hard/boundary/easy
scenes use the shared PromptBuilder evidence schema.
"""
import numpy as np

from nav import nav_env as NE
from safety_gymnasium.utils.common_utils import ResamplingError

TAU = 0.125
TARGET_DELTA = 0.15


def default_scene_sampler(rng):
    """Stage-1 proxy generator: random training start and two-pillar layout."""
    start = NE.sample_valid_start(rng)
    pillars = NE.dedupe(NE.sample_training_layout(rng))
    if not pillars:
        pillars = [(0.0, 0.0)]
    return start, pillars


def _rollout(adapter, act_fn, pillars, start, skill_id, max_steps, seed=0,
             collect_route=False):
    """Roll out one skill and optionally retain its already-observed XY route."""
    adapter.set_layout(pillars, start=start)
    try:
        obs = adapter.reset(seed=seed, start=start)
    except ResamplingError:
        return None

    points = []
    if collect_route:
        points.append(np.asarray(adapter.agent_xy(), dtype=float).tolist())
    last_info = {}
    for _ in range(max_steps):
        obs, _, term, trunc, info = adapter.step(act_fn(obs, skill_id))
        last_info = info
        if collect_route:
            points.append(np.asarray(adapter.agent_xy(), dtype=float).tolist())
        if adapter.success(info):
            return {"success": True, "outcome": "success", "points": points}
        if term or trunc:
            break

    if last_info.get("collision", False):
        outcome = "collision"
    elif last_info.get("oob", False):
        outcome = "fall"  # shared batch key; rendered as out_of_bounds for Navigate
    else:
        outcome = "timeout"
    return {"success": False, "outcome": outcome, "points": points}


def _solve(adapter, act_fn, pillars, start, skill_id, max_steps, seed=0):
    """Compatibility wrapper returning bool/None for one deterministic rollout."""
    result = _rollout(
        adapter, act_fn, pillars, start, skill_id, max_steps, seed,
        collect_route=False,
    )
    return None if result is None else bool(result["success"])


def library_p_on_scene(adapter, act_fn, pillars, start, K, max_steps=500,
                       probe_skills=None, collect_routes=False):
    """Return ``(Y,p,b)``; with ``collect_routes``, append route records."""
    skills = list(range(1, K + 1)) if probe_skills is None else list(probe_skills)
    outcomes = []
    for skill_id in skills:
        result = _rollout(
            adapter, act_fn, pillars, start, skill_id, max_steps,
            collect_route=collect_routes,
        )
        if result is None:
            return None
        outcomes.append((skill_id, result))
    Y = np.asarray([1.0 if result["success"] else 0.0 for _, result in outcomes])
    p = float(Y.mean())
    base = (Y, p, p * (1 - p))
    if not collect_routes:
        return base
    routes = [
        {
            "skill_index": int(skill_id),
            "success": bool(result["success"]),
            "points": result["points"],
        }
        for skill_id, result in outcomes
    ]
    return base + (routes,)


def single_p_on_scene(adapter, act_fn, pillars, start, max_steps=500, skill_id=0):
    """Deployment skill success on one scene."""
    result = _rollout(adapter, act_fn, pillars, start, skill_id, max_steps)
    return None if result is None else (1.0 if result["success"] else 0.0)


def _pillar_records(pillars):
    records = []
    for pillar in pillars:
        if isinstance(pillar, dict):
            records.append({
                "x": float(pillar["x"]),
                "y": float(pillar["y"]),
                "purpose": str(pillar.get("purpose", "")),
            })
        else:
            records.append({
                "x": float(pillar[0]), "y": float(pillar[1]), "purpose": "",
            })
    return records


def _route_waypoints(points, start, goal, max_waypoints):
    if not points or max_waypoints <= 0:
        return []
    indices = sorted({
        int(round(value))
        for value in np.linspace(0, len(points) - 1, min(len(points), max_waypoints))
    })
    start = np.asarray(start, dtype=float)
    axis = np.asarray(goal, dtype=float) - start
    length = max(float(np.linalg.norm(axis)), 1e-8)
    unit = axis / length
    perp = np.asarray([-unit[1], unit[0]])
    waypoints = []
    for idx in indices:
        point = np.asarray(points[idx], dtype=float)
        delta = point - start
        waypoints.append({
            "step": int(idx),
            "x": float(point[0]),
            "y": float(point[1]),
            "alpha": float(np.dot(delta, unit) / length),
            "beta": float(np.dot(delta, perp) / length),
        })
    return waypoints


def _summarize_scene(scene, include_behavior, route_waypoints):
    summary = {
        key: scene[key]
        for key in ("scene_id", "start", "obstacles", "feasible", "realized", "deployed")
    }
    if include_behavior:
        success = [route for route in scene["routes"] if route["success"]][:2]
        failure = [route for route in scene["routes"] if not route["success"]][:1]
        selected = success + failure
        summary["behavior_summary"] = {
            "requested": {"success": 2, "failure": 1},
            "actual": {"success": len(success), "failure": len(failure)},
            "routes": [
                {
                    "skill_index": route["skill_index"],
                    "success": route["success"],
                    "waypoints": _route_waypoints(
                        route["points"], scene["start"], NE.GOAL, route_waypoints,
                    ),
                }
                for route in selected
            ],
        }
    return summary


def _build_design_context(scenes, n, include_behavior, route_waypoints, tau):
    """Use the same strict tau-bin SOURCE/TARGET/GUARD selection as Push-T."""
    n = max(int(n), 0)
    boundary = sorted(
        (row for row in scenes if tau < row["realized"] < 1 - tau),
        key=lambda row: abs(row["realized"] - 0.5),
    )[:n]
    easy = sorted(
        (row for row in scenes if row["realized"] >= 1 - tau),
        key=lambda row: row["realized"], reverse=True,
    )[:n]
    hard = sorted(
        (row for row in scenes if row["realized"] <= tau),
        key=lambda row: row["realized"],
    )[:n]
    summarize = lambda rows: [
        _summarize_scene(row, include_behavior, route_waypoints) for row in rows
    ]
    return {"focus": summarize(boundary), "harden": summarize(easy), "avoid": summarize(hard)}


def extract_skill_signals(adapter, act_fn, K, M=64, seed=123, max_steps=500,
                          scene_sampler=None, invalid_scene_policy="resample",
                          probe_skills=None, context_n=4, include_behavior=True,
                          route_waypoints=5, tau=TAU,
                          target_delta=TARGET_DELTA):
    """Probe ``M`` scenes and return aggregate plus representative evidence."""
    scene_sampler = scene_sampler or default_scene_sampler
    rng = np.random.default_rng(seed)
    ps, bs, w0s, scenes = [], [], [], []
    valid, dup_hits, attempts = 0, 0, 0
    seen = set()
    max_attempts = M * 30
    collect_routes = bool(include_behavior and context_n > 0)

    while len(ps) < M and attempts < max_attempts:
        attempts += 1
        try:
            start, pillars = scene_sampler(rng)
        except ResamplingError:
            if invalid_scene_policy == "zero":
                ps.append(0.0); bs.append(0.0); w0s.append(0.0)
            continue
        key = (round(start[0], 2), round(start[1], 2),
               tuple(sorted((round(x, 2), round(y, 2)) for x, y in pillars)))
        if key in seen:
            dup_hits += 1
        result = library_p_on_scene(
            adapter, act_fn, pillars, start, K, max_steps, probe_skills,
            collect_routes=collect_routes,
        )
        if result is None:
            if invalid_scene_policy == "zero":
                ps.append(0.0); bs.append(0.0); w0s.append(0.0)
                seen.add(key)
            continue
        Y, p, b = result[:3]
        routes = result[3] if collect_routes else []
        valid += 1
        seen.add(key)
        ps.append(p); bs.append(b)
        w0_result = _rollout(
            adapter, act_fn, pillars, start, 0, max_steps,
            collect_route=False,
        )
        w0 = 0.0 if w0_result is None else float(w0_result["success"])
        w0s.append(w0)
        scenes.append({
            "scene_id": len(scenes),
            "start": [float(start[0]), float(start[1])],
            "obstacles": _pillar_records(pillars),
            "feasible": int(bool(np.max(Y))) if len(Y) else 0,
            "realized": p,
            "deployed": int(w0),
            "routes": routes,
        })

    ps = np.asarray(ps); bs = np.asarray(bs); w0s = np.asarray(w0s)
    N = max(len(ps), 1)
    tau = float(tau)
    target_delta = float(target_delta)
    boundary_count = int(((ps > tau) & (ps < 1 - tau)).sum())
    effective_K = len(probe_skills) if probe_skills is not None else int(K)
    return {
        "K": effective_K,
        "N": len(ps),
        "tau": tau,
        "target_delta": target_delta,
        "probe_source": "w_probe" if probe_skills is None else "single",
        "per_scene_p": ps.tolist(),
        "boundary_count": boundary_count,
        "boundary_rate": boundary_count / N,
        "mean_b": float(bs.mean()) if len(bs) else 0.0,
        "r_hard": float((ps <= tau).mean()) if len(ps) else 0.0,
        "r_easy": float((ps >= 1 - tau).mean()) if len(ps) else 0.0,
        "target_rate": float((np.abs(ps - 0.5) <= target_delta).mean()) if len(ps) else 0.0,
        "valid_rate": valid / max(attempts, 1),
        "duplicate_rate": dup_hits / max(attempts, 1),
        "w0_success_rate": float(w0s.mean()) if len(w0s) else 0.0,
        "include_behavior": bool(include_behavior),
        "design_context": _build_design_context(
            scenes, context_n, include_behavior, route_waypoints, tau,
        ),
    }
