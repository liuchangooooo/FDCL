"""参数化 B/M/U/D 零样本泛化基准(对齐 DIVO `config/evaluation/eval.yaml` 口径)。

DIVO 的 B/M/U/D 不是手工拓扑,而是按 (num/size/shape/dist) 定义的参数化障碍类别,
起点/goal 固定、障碍物位置与朝向随机采样、20 回合平均。随机候选只做
起终点净空与障碍间距的物理合法性拒绝,不按起点位置或起点—目标路径构造布局。
导航映射:
  B: 1 个大 pillar
  M: 3 个 pillar
  U: 1 个由 7 个对称元素组成的 U 形复合障碍
  D: 3 个错相连续运动的 Gremlin;同时评测几何完全一致的冻结 D_static

只作最终零样本测试,SHALL NOT 参与选 best。结构化压力测试
仅在此登记为可选 eval 拓展入口,别处不得使用。
"""
import numpy as np

from nav import nav_env as NE
from nav.nav_adapter import NavEnvAdapter
from safety_gymnasium.utils.common_utils import ResamplingError

# 参数化类别定义(协议 v2 固定值;keepout 与物理尺寸共同约束合法放置)
BENCHMARKS = {
    "B": {"num": 1, "size": 0.30, "keepout": 0.33},          # Big:1 个大障碍
    "M": {"num": 3, "size": NE.PILLAR_SIZE,
          "keepout": NE.PILLAR_KEEPOUT},                         # Multiple:只把标准障碍增至3个
    # U:由 7 个对称元素组成的 U 形复合块,随机位置+朝向
    "U": {"kind": "ushape", "size": 0.11, "keepout": 0.09, "arm": 0.24},  # 紧凑复合块,keepout<间距/2
    # Dynamics:3 个错相连续周期运动的 Gremlin 顶替 pillar(obs 仍 44)
    "D": {"num": 3, "size": NE.GREMLIN_SIZE, "keepout": NE.GREMLIN_KEEPOUT,
          "travel": NE.GREMLIN_TRAVEL, "dynamic": True},
}


def _ushape_block(center, theta, arm=0.24):
    """紧凑 U 形 pillar 块(开口朝 -x 局部方向,⊃ 形),按 theta 旋转、平移到 center。

    局部:上横条(y=+arm)、下横条(y=-arm)、右竖条(x=+arm);开口在 -x 侧。
    """
    local = [
        (-arm, arm), (0.0, arm), (arm, arm),
        (arm, 0.0),
        (arm, -arm), (0.0, -arm), (-arm, -arm),
    ]
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    cen = np.asarray(center, float)
    return [tuple(np.round(cen + R @ np.array(p), 3)) for p in local]

# pillar 随机放置区(= 规范障碍区 ±OBSTACLE_REGION,与训练/验证一致,只变 num/size)
_PX = (-NE.OBSTACLE_REGION, NE.OBSTACLE_REGION)
_PY = (-NE.OBSTACLE_REGION, NE.OBSTACLE_REGION)


def _goal_clearance(size, keepout):
    """同时满足实体不重叠与 Safety-Gym placement keepout 的最小 goal 净空。"""
    return max(NE.GOAL_SIZE + float(size), NE.GOAL_KEEPOUT + float(keepout))


def _start_clearance(size, keepout, travel=0.0):
    """固定起点的最小净空;动态障碍的实体边界包含扫掠半径。"""
    return max(
        NE.AGENT_RADIUS + float(size) + float(travel),
        NE.AGENT_KEEPOUT + float(keepout),
    )


def _sample_pillars(rng, num, size, keepout, start, goal, travel=0.0):
    s = np.asarray(start, dtype=float)
    g = np.asarray(goal, dtype=float)
    min_sep = max(2 * keepout + 0.03, 2 * size + 0.01, NE.PILLAR_MIN_SEPARATION)
    start_clear = _start_clearance(size, keepout, travel)
    goal_clear = _goal_clearance(size, keepout)
    pts = []
    tries = 0
    while len(pts) < num and tries < 400:
        tries += 1
        x = round(float(rng.uniform(*_PX)), 3)
        y = round(float(rng.uniform(*_PY)), 3)
        if np.hypot(x - s[0], y - s[1]) < start_clear:
            continue
        if np.hypot(x - g[0], y - g[1]) < goal_clear:
            continue
        if all(np.hypot(x - px, y - py) >= min_sep for px, py in pts):
            pts.append((x, y))
    return pts


def _ushape_is_valid(pillars, size, keepout, start, goal):
    """旋转后的 7 个元素都必须在规范障碍区内,且逐点避开固定起终点。"""
    s = np.asarray(start, dtype=float)
    g = np.asarray(goal, dtype=float)
    return len(pillars) == 7 and all(
        _PX[0] <= x <= _PX[1]
        and _PY[0] <= y <= _PY[1]
        and np.hypot(x - s[0], y - s[1]) >= _start_clearance(size, keepout)
        and np.hypot(x - g[0], y - g[1]) >= _goal_clearance(size, keepout)
        for x, y in pillars
    )


def sample_benchmark_scene(family, seed):
    """采一个 family 的场景:固定起终点,随机采样障碍参数。"""
    spec = BENCHMARKS[family]
    geom_seq, reset_seq = np.random.SeedSequence(int(seed)).spawn(2)
    rng = np.random.default_rng(geom_seq)
    if spec.get("kind") == "ushape":
        # U 形复合块:随机中心+朝向;验证旋转后的全部 7 点,不只验证中心。
        pillars = []
        for _ in range(400):
            cx = float(rng.uniform(-0.2, 0.2))
            cy = float(rng.uniform(-0.2, 0.2))
            theta = float(rng.uniform(0, 2 * np.pi))
            candidate = NE.dedupe(_ushape_block((cx, cy), theta, arm=spec["arm"]))
            if _ushape_is_valid(
                candidate, spec["size"], spec["keepout"], NE.START, NE.GOAL
            ):
                pillars = candidate
                break
        if len(pillars) != 7:
            raise ResamplingError("U 场景采样失败:cannot place all 7 elements")
    else:
        pillars = NE.dedupe(
            _sample_pillars(
                rng,
                spec["num"],
                spec["size"],
                spec["keepout"],
                NE.START,
                NE.GOAL,
                travel=float(spec.get("travel", 0.0)),
            )
        )
        if len(pillars) != spec["num"]:
            raise ResamplingError(
                f"{family} 场景采样失败:expected {spec['num']} obstacles, got {len(pillars)}"
            )
    return {
        "pillars": pillars,
        "size": spec["size"],
        "keepout": spec["keepout"],
        "dynamic": bool(spec.get("dynamic", False)),
        "travel": float(spec.get("travel", 0.0)),
        "goal": NE.GOAL,
        "start": NE.START,
        "reset_seed": int(reset_seq.generate_state(1, dtype=np.uint32)[0]),
    }


def evaluate_benchmark(adapter, act_fn, family, n_env=20, max_steps=500, base_seed=2024,
                       gremlin_travel=None):
    """在 family 上跑 w_0 部署(固定起点,随机障碍),返回 success_rate。"""
    expected_dynamic = bool(BENCHMARKS[family].get("dynamic", False))
    if bool(adapter.dynamic) != expected_dynamic:
        mode = "dynamic Gremlin" if expected_dynamic else "static Pillar"
        raise ValueError(f"family {family} requires a {mode} adapter")
    succ = []
    ep, attempts = 0, 0
    while ep < n_env and attempts < n_env * 10:
        attempts += 1
        try:
            scene = sample_benchmark_scene(family, base_seed + attempts)
            start = scene["start"]
            travel = scene["travel"] if gremlin_travel is None else float(gremlin_travel)
            adapter.set_layout(
                scene["pillars"], start=start, goal=scene["goal"],
                pillar_size=scene["size"], pillar_keepout=scene["keepout"],
                gremlin_travel=travel,
            )
            obs = adapter.reset(seed=scene["reset_seed"], start=start)
        except ResamplingError:
            continue
        ep += 1
        reached = False
        for _ in range(max_steps):
            obs, r, term, trunc, info = adapter.step(act_fn(obs))
            if adapter.success(info):
                reached = True
                break
            if term or trunc:
                break
        succ.append(1.0 if reached else 0.0)
    return {
        "family": family,
        "success_rate": float(np.mean(succ)) if succ else 0.0,
        "n_eval": len(succ),
        "n_requested": int(n_env),
        "complete": len(succ) == int(n_env),
    }


def evaluate_bmud(adapter, act_fn, n_env=20, max_steps=500, dyn_adapter=None,
                  base_seed=2024):
    """跑全部 B/M/U/D,返回各族 success_rate 与 AVG。

    B/M/U 用传入的静态 adapter。D_static 与 D_dynamic 使用同一个
    Gremlin adapter、同一组场景/起点/种子,只把 travel 从 0 改为 GREMLIN_TRAVEL。
    D 保留为 D_dynamic 的兼容别名,AVG 只计 B/M/U/D 四项。
    """
    if adapter.dynamic:
        raise ValueError("B/M/U require a static Pillar adapter")
    res = {}
    n_eval = {}
    for family in ("B", "M", "U"):
        stat = evaluate_benchmark(
            adapter, act_fn, family, n_env, max_steps, base_seed=base_seed
        )
        res[family] = stat["success_rate"]
        n_eval[family] = stat["n_eval"]

    dad = dyn_adapter or NavEnvAdapter(dynamic=True)
    if not dad.dynamic:
        if dyn_adapter is None:
            dad.close()
        raise ValueError("dyn_adapter must be constructed with dynamic=True")
    try:
        frozen = evaluate_benchmark(
            dad, act_fn, "D", n_env, max_steps, base_seed=base_seed,
            gremlin_travel=0.0,
        )
        moving = evaluate_benchmark(
            dad, act_fn, "D", n_env, max_steps, base_seed=base_seed,
            gremlin_travel=NE.GREMLIN_TRAVEL,
        )
    finally:
        if dyn_adapter is None:
            dad.close()

    res["D_static"] = frozen["success_rate"]
    res["D_dynamic"] = moving["success_rate"]
    res["D"] = res["D_dynamic"]
    res["dynamic_drop"] = res["D_static"] - res["D_dynamic"]
    n_eval.update({
        "D_static": frozen["n_eval"],
        "D_dynamic": moving["n_eval"],
    })
    incomplete = {key: value for key, value in n_eval.items() if value != int(n_env)}
    if incomplete:
        raise RuntimeError(
            f"B/M/U/D evaluation incomplete:requested={int(n_env)}, n_eval={n_eval}"
        )
    res["AVG"] = float(np.mean([res[k] for k in ("B", "M", "U", "D")]))
    return res


# --- 结构化拓展套件(仅可选 eval 拓展入口,别处不得使用) ---
def evaluate_structured_extension(adapter, act_fn, max_steps=500):
    """手工 barrier/maze/u/dead-end——仅作 B/M/U/D 之外的可选 eval。

    这些结构相对 NE.START -> NE.GOAL 连线设计,故用固定/受限起点保陷阱语义,
    每类单实例(可后续扩多实例);不参与选 best、不用于训练/探针/motivation。
    """
    if adapter.dynamic:
        raise ValueError("structured stress tests require a static Pillar adapter")
    out = {}
    for name, fn in NE.STRUCTURED_STRESS_TESTS.items():
        adapter.set_layout(NE.dedupe(fn()), start=NE.START, goal=NE.GOAL)
        try:
            obs = adapter.reset(seed=0, start=NE.START)
        except ResamplingError:
            out[name] = None
            continue
        reached = False
        for _ in range(max_steps):
            obs, r, term, trunc, info = adapter.step(act_fn(obs))
            if adapter.success(info):
                reached = True
                break
            if term or trunc:
                break
        out[name] = 1.0 if reached else 0.0
    return out
