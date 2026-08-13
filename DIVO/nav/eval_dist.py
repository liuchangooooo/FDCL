"""验证分布(DIVO `between` 范式的导航版)——用于选 best,不参与 B/M/U/D。

范式:先放 num=2 的中等难度 pillar,再把 agent 起点采在"某 pillar 与 goal 之间"
(goal 固定)。分布固定、独立于课程 G_t、与 B/M/U/D 不重叠(B/M/U/D 是各自的参数化
类别 + 固定起点)。选 best 只用 w_0 部署在本分布上的指标(test_mean_score, mode=max)。

对齐 Push-T:n_env_validate=20、独立固定验证分布、TopK(mode=max)。Nav 的逐步
reward 是到目标的距离进展(+成功奖励),故 test_mean_score 使用平均 episode return,
而不是只看末步 reward 或二值成功率。
"""
import numpy as np

from nav import nav_env as NE
from nav.nav_adapter import NavEnvAdapter
from nav.curriculum.generator_source import validate_pillars
from safety_gymnasium.utils.common_utils import ResamplingError

VAL_NUM_PILLARS = 2
# pillar 放置区(与训练的障碍中心支持集一致)
_PX = (-NE.VAL_OBSTACLE_REGION, NE.VAL_OBSTACLE_REGION)
_PY = (-NE.VAL_OBSTACLE_REGION, NE.VAL_OBSTACLE_REGION)
_MIN_SEP = NE.PILLAR_MIN_SEPARATION


def _sample_pillars(rng):
    pts = []
    tries = 0
    while len(pts) < VAL_NUM_PILLARS and tries < 200:
        tries += 1
        x = float(rng.uniform(*_PX))
        y = float(rng.uniform(*_PY))
        if all(np.hypot(x - px, y - py) >= _MIN_SEP for px, py in pts):
            pts.append((round(x, 3), round(y, 3)))
    return pts


def sample_validation_scene(seed, max_attempts=200):
    """拒绝采样一个合法验证场景 {pillars, start, goal}(between 起点)。"""
    rng = np.random.default_rng(int(seed))
    for _ in range(int(max_attempts)):
        pillars = NE.dedupe(_sample_pillars(rng))
        if len(pillars) != VAL_NUM_PILLARS:
            continue
        start = _between_start(rng, pillars, NE.GOAL)
        if not _start_in_region(start) or not NE.start_goal_ok(start, NE.GOAL):
            continue
        ok, _ = validate_pillars(
            pillars, start, NE.GOAL, num=VAL_NUM_PILLARS,
        )
        if ok:
            return {"pillars": pillars, "start": start, "goal": NE.GOAL}
    raise RuntimeError(
        f"failed to sample a valid between-validation scene after {max_attempts} attempts"
    )


def _start_in_region(start):
    x, y = start
    return (
        NE.START_X_RANGE[0] <= x <= NE.START_X_RANGE[1]
        and NE.START_Y_RANGE[0] <= y <= NE.START_Y_RANGE[1]
    )


def _between_start(rng, pillars, goal):
    """与 NavEnvAdapter.sample_between_start 同算法(此处不依赖 env 实例)。"""
    P = np.asarray(pillars, dtype=float)
    g = np.asarray(goal, dtype=float)
    p = P[int(rng.integers(len(P)))]
    d = p - g
    n = float(np.linalg.norm(d))
    if n < 1e-6:
        return NE.sample_valid_start(rng, goal)
    u = d / n
    perp = np.array([-u[1], u[0]])
    # 保留 between 结构，同时按未缩小的 agent/pillar 实体尺寸留足机动空间。
    offset = float(rng.uniform(*NE.BETWEEN_OFFSET_RANGE))
    lateral = float(rng.uniform(*NE.BETWEEN_LATERAL_RANGE))
    start = p + u * offset + perp * lateral
    return float(start[0]), float(start[1])


def evaluate_validation(adapter, act_fn, n_env=20, max_steps=500, base_seed=777):
    """在验证分布上跑 w_0 部署(确定性),返回选 best 指标。

    act_fn(obs) -> action(2,);应为 w_0 部署的确定性动作。
    test_mean_score = mean episode return(mode=max);另报成功/碰撞/越界/超时率和最终距离。
    """
    outcomes = {"success": 0, "collision": 0, "oob": 0, "timeout": 0}
    succ, episode_returns, final_d = [], [], []
    ep = 0
    attempts = 0
    while ep < n_env and attempts < n_env * 10:
        attempts += 1
        scene = sample_validation_scene(base_seed + attempts)
        adapter.set_layout(scene["pillars"], start=scene["start"], goal=scene["goal"])
        try:
            obs = adapter.reset(seed=0, start=scene["start"])
        except ResamplingError:
            continue
        ep += 1
        reached = False
        episode_return = 0.0
        info = {}
        for _ in range(max_steps):
            obs, r, term, trunc, info = adapter.step(act_fn(obs))
            episode_return += float(r)
            if adapter.success(info):
                reached = True
                break
            if term or trunc:
                break
        if reached:
            outcome = "success"
        elif info.get("collision", False):
            outcome = "collision"
        elif info.get("oob", False):
            outcome = "oob"
        else:
            outcome = "timeout"
        outcomes[outcome] += 1
        succ.append(1.0 if reached else 0.0)
        episode_returns.append(episode_return)
        gxy = np.asarray(adapter.goal)
        axy = adapter._env.task.data.body("agent").xpos[:2]
        final_d.append(float(np.linalg.norm(gxy - axy)))
    sr = float(np.mean(succ)) if succ else 0.0
    mean_return = float(np.mean(episode_returns)) if episode_returns else 0.0
    n_evaluated = len(succ)
    return {
        "test_mean_score": mean_return,  # 选 best 用(mode=max)
        "mean_episode_return": mean_return,
        "success_rate": sr,
        "collision_rate": outcomes["collision"] / n_evaluated if n_evaluated else 0.0,
        "oob_rate": outcomes["oob"] / n_evaluated if n_evaluated else 0.0,
        "timeout_rate": outcomes["timeout"] / n_evaluated if n_evaluated else 0.0,
        "mean_final_dist": float(np.mean(final_d)) if final_d else float("nan"),
        "n_eval": n_evaluated,
    }
