"""Stage 1 fixed difficulty ladder (Task 12 / Requirements 10.1, 10.2).

Deterministic, constructable Push-T layouts used to validate whether the skill
library forms a non-degenerate p_i distribution (easy ~1, hard ~0, mid 0<p<1)
WITHOUT the LLM curriculum. Each category yields N fixed layouts with explicit
obstacle coordinates (exactly ``obstacle_num`` obstacles) and a fixed start.

Coordinates are Push-T world coordinates: goal at (0, 0), obstacles within about
[-0.2, 0.2]. A "blocking" obstacle is placed at a fraction ``t`` along the
start->goal segment with a perpendicular offset ``o``; the remaining filler
obstacles sit in far corners off the corridor. Physical validity (contacts /
``is_obstacle_config_valid``) is filtered downstream by the probe/eval runner;
this module only defines the deterministic geometry.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


PUSHT_GOAL_XY: Tuple[float, float] = (0.0, 0.0)
DEFAULT_START_XY: Tuple[float, float] = (0.15, 0.15)
DEFAULT_START_THETA: float = -np.pi / 4
FAR_CORNERS: Tuple[Tuple[float, float], ...] = (
    (-0.17, 0.17), (0.17, -0.17), (-0.17, -0.17), (0.17, 0.17),
)

# category -> (corridor fraction t, perpendicular offset o) for the blocking
# obstacle. None means "no blocking obstacle" (easy). "narrow" is a special case
# handled explicitly (two straddling obstacles).
LADDER_CATEGORIES = ("easy", "two-route", "left-preferred", "right-preferred", "narrow", "near-goal")
_BLOCK_SPEC: Dict[str, Any] = {
    "easy": None,
    "two-route": (0.5, 0.0),
    "left-preferred": (0.5, 0.05),   # blocks the right side -> left route easier
    "right-preferred": (0.5, -0.05),  # blocks the left side -> right route easier
    "narrow": "narrow",
    "near-goal": (0.85, 0.0),
}


def _corridor_frame(start_xy, goal_xy):
    sx, sy = float(start_xy[0]), float(start_xy[1])
    gx, gy = float(goal_xy[0]), float(goal_xy[1])
    dx, dy = gx - sx, gy - sy
    norm = float(np.hypot(dx, dy)) + 1e-9
    px, py = -dy / norm, dx / norm  # unit perpendicular
    return (sx, sy), (dx, dy), (px, py)


def _point_on_corridor(start_xy, goal_xy, t, o):
    (sx, sy), (dx, dy), (px, py) = _corridor_frame(start_xy, goal_xy)
    x = sx + t * dx + o * px
    y = sy + t * dy + o * py
    return float(np.clip(x, -0.19, 0.19)), float(np.clip(y, -0.19, 0.19))


def _obstacle(x, y, purpose=""):
    return {"x": float(x), "y": float(y), "purpose": str(purpose)}


def _pad_with_fillers(obstacles: List[Dict[str, float]], obstacle_num: int) -> List[Dict[str, float]]:
    """Pad/trim to exactly obstacle_num using far-corner filler obstacles."""
    out = list(obstacles)
    ci = 0
    while len(out) < int(obstacle_num):
        fx, fy = FAR_CORNERS[ci % len(FAR_CORNERS)]
        out.append(_obstacle(fx, fy, "filler"))
        ci += 1
    return out[: int(obstacle_num)]


def _build_category_layout(category, start_xy, start_theta, obstacle_num, jitter):
    """Build one layout for a category given a (seeded) jitter vector."""
    jt, jo, jsx, jsy = jitter
    start = [float(start_xy[0] + jsx), float(start_xy[1] + jsy), float(start_theta)]
    goal = PUSHT_GOAL_XY
    spec = _BLOCK_SPEC[category]

    obstacles: List[Dict[str, float]] = []
    if spec is None:  # easy: no corridor blocker
        pass
    elif spec == "narrow":  # two straddling obstacles forming a narrow gap
        x1, y1 = _point_on_corridor(start, goal, 0.5 + jt, 0.06 + jo)
        x2, y2 = _point_on_corridor(start, goal, 0.5 + jt, -0.06 - jo)
        obstacles.append(_obstacle(x1, y1, "narrow"))
        obstacles.append(_obstacle(x2, y2, "narrow"))
    else:
        t, o = spec
        bx, by = _point_on_corridor(start, goal, t + jt, o + jo)
        obstacles.append(_obstacle(bx, by, "block"))

    obstacles = _pad_with_fillers(obstacles, obstacle_num)
    return {"start": start, "obstacles": obstacles, "category": category}


def build_stage1_ladder(
    obstacle_num: int = 2,
    n_per_category: int = 5,
    seed: int = 0,
    start_xy: Sequence[float] = DEFAULT_START_XY,
    start_theta: float = DEFAULT_START_THETA,
) -> Dict[str, List[Dict[str, Any]]]:
    """Build the fixed Stage 1 ladder: category -> list of N layouts.

    Deterministic given ``seed`` (jitter is drawn from a seeded RNG), so the
    ladder is reproducible across runs (Requirement 10.2).
    """
    rng = np.random.default_rng(int(seed))
    ladder: Dict[str, List[Dict[str, Any]]] = {}
    for category in LADDER_CATEGORIES:
        layouts = []
        for _ in range(int(n_per_category)):
            jitter = (
                float(rng.uniform(-0.03, 0.03)),   # t jitter
                float(rng.uniform(-0.01, 0.01)),   # o jitter
                float(rng.uniform(-0.02, 0.02)),   # start x jitter
                float(rng.uniform(-0.02, 0.02)),   # start y jitter
            )
            layouts.append(
                _build_category_layout(category, start_xy, start_theta, obstacle_num, jitter)
            )
        ladder[category] = layouts
    return ladder


def flatten_ladder(ladder: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Flatten the ladder dict into a single list of layouts (order preserved).

    Iterates the dict's own keys so it works for both the legacy hand-placed
    categories and the sampled easy/mid/hard tiers.
    """
    out: List[Dict[str, Any]] = []
    for category in ladder.keys():
        out.extend(ladder.get(category, []))
    return out


def _env_layout_valid(env, start, obstacles) -> bool:
    """Env-side validity: obstacle-config valid + no initial contact."""
    start_arr = np.asarray(start, dtype=np.float64)
    try:
        if hasattr(env, "is_obstacle_config_valid"):
            if not env.is_obstacle_config_valid(list(obstacles), start_arr):
                return False
        if hasattr(env, "set_obstacle_config"):
            env.set_obstacle_config(list(obstacles))
        if hasattr(env, "reset"):
            env.reset(tblock_pos=start_arr, force_tblock_pos=True)
        if hasattr(env, "get_ncon") and int(env.get_ncon()) != 0:
            return False
    except Exception:
        return False
    return True


def _point_to_segment_dist(p, a, b) -> float:
    p = np.asarray(p, dtype=np.float64); a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    ab = b - a
    denom = float(ab @ ab) + 1e-12
    t = float(np.clip(((p - a) @ ab) / denom, 0.0, 1.0))
    proj = a + t * ab
    return float(np.linalg.norm(p - proj))


def layout_difficulty_proxy(start, obstacles, goal_xy=PUSHT_GOAL_XY) -> float:
    """A-priori difficulty proxy: min distance from any obstacle to the
    start->goal segment. Smaller = obstacle pinches the path = harder.
    """
    a = np.asarray(start, dtype=np.float64)[:2]
    b = np.asarray(goal_xy, dtype=np.float64)[:2]
    dists = [_point_to_segment_dist((float(o["x"]), float(o["y"])), a, b) for o in obstacles]
    return float(min(dists)) if dists else 1e9


def build_sampled_stage1_ladder(
    env,
    obstacle_num: int = 2,
    n_per_tier: int = 6,
    seed: int = 0,
    n_sample: int = 400,
) -> Dict[str, List[Dict[str, Any]]]:
    """Env-valid, difficulty-stratified ladder (recommended over the hand-placed
    corridor ladder, which is geometrically infeasible on Push-T).

    Samples env-valid random-obstacle layouts, ranks them by
    ``layout_difficulty_proxy`` (min obstacle-to-path distance), and buckets into
    ``easy`` (far obstacles), ``mid``, and ``hard`` (obstacles pinching the path).
    Every layout is env-valid by construction.
    """
    rng = np.random.RandomState(int(seed))
    np.random.seed(int(seed))
    try:
        env.seed(int(seed))
    except Exception:
        pass
    desk = float(getattr(getattr(env, "task", object()), "_desk_size", 1.0))
    if hasattr(env, "clear_obstacle_config"):
        env.clear_obstacle_config()

    samples = []
    for _ in range(int(n_sample)):
        try:
            obs = env.reset()
            positions = env.get_obstacle_positions()
            obstacles = [{"x": float(p["x"]), "y": float(p["y"])} for p in positions]
            if len(obstacles) != int(obstacle_num):
                continue
            arr = np.asarray(obs, dtype=np.float64).reshape(-1)
            start = [float(arr[0] * desk), float(arr[1] * desk), float(np.arctan2(arr[3], arr[2]))]
            samples.append({"start": start, "obstacles": obstacles,
                            "difficulty": layout_difficulty_proxy(start, obstacles)})
        except Exception:
            continue
    if not samples:
        return {"easy": [], "mid": [], "hard": []}

    samples.sort(key=lambda s: s["difficulty"], reverse=True)  # far->near (easy->hard)
    n = len(samples)
    third = max(1, n // 3)
    tiers = {
        "easy": samples[:third],
        "mid": samples[third: 2 * third],
        "hard": samples[2 * third:],
    }
    out: Dict[str, List[Dict[str, Any]]] = {}
    for tier, rows in tiers.items():
        picked = rows[: int(n_per_tier)]
        for r in picked:
            r["category"] = tier
        out[tier] = picked
    return out


def build_valid_stage1_ladder(
    env,
    obstacle_num: int = 2,
    n_per_category: int = 5,
    seed: int = 0,
    start_xy: Sequence[float] = DEFAULT_START_XY,
    start_theta: float = DEFAULT_START_THETA,
    max_tries_per_layout: int = 200,
) -> Dict[str, List[Dict[str, Any]]]:
    """Build a ladder whose layouts are validated against the env.

    For each category, jitter the geometric template and keep only layouts that
    pass ``is_obstacle_config_valid`` + no initial contact. This fixes the
    constructor producing mostly physically-invalid layouts. Categories that
    cannot reach ``n_per_category`` valid layouts return however many were found
    (reported by the caller).
    """
    rng = np.random.default_rng(int(seed))
    ladder: Dict[str, List[Dict[str, Any]]] = {}
    for category in LADDER_CATEGORIES:
        layouts: List[Dict[str, Any]] = []
        tries = 0
        while len(layouts) < int(n_per_category) and tries < int(max_tries_per_layout) * int(n_per_category):
            tries += 1
            # larger jitter to explore valid placements around the template
            jitter = (
                float(rng.uniform(-0.06, 0.06)),
                float(rng.uniform(-0.03, 0.03)),
                float(rng.uniform(-0.03, 0.03)),
                float(rng.uniform(-0.03, 0.03)),
            )
            lay = _build_category_layout(category, start_xy, start_theta, obstacle_num, jitter)
            if _env_layout_valid(env, lay["start"], lay["obstacles"]):
                layouts.append(lay)
        ladder[category] = layouts
    return ladder
