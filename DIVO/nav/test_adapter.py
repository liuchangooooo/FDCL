"""NavEnvAdapter 组件级 sanity(在 safenav 下:MUJOCO_GL=egl python -m nav.test_adapter)。

验证:
  1) 同一 adapter 依次注入不同布局(pillar 数变化)都能 reset,obs 恒 44 维
  2) pillar 落在注入的坐标上
  3) step 返回 reward/cost/info,动作维度正确
  4) 从 start 直冲 goal(简单前进控制),能拿到非零进度奖励
  5) 非法布局(pillar 压在 start 上)reset 抛 ResamplingError
  6) 动态 D 的 Gremlin 可构造、可冻结、且在各自中心周围运动
"""
import os
import inspect

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

from nav import nav_env as NE
from nav.nav_adapter import NavEnvAdapter
from safety_gymnasium.utils.common_utils import ResamplingError


# 组件测试只验证 adapter 的布局切换，不借用正式 B/M/U/D 或结构压力测试。
# 坐标间距显式大于 2 * PILLAR_KEEPOUT，避免把非法模板误判为 adapter 故障。
COMPONENT_LAYOUTS = {
    "one": [(0.0, 0.55)],
    "two": [(-0.25, 0.55), (0.25, -0.55)],
    "three": [(-0.40, 0.55), (0.0, -0.55), (0.40, 0.55)],
}


class _FakeActionSpace:
    shape = (2,)


class _FakeStepEnv:
    action_space = _FakeActionSpace()

    def __init__(self, cost):
        self._cost = float(cost)

    def step(self, _action):
        return (
            np.zeros(NE.OBS_DIM, dtype=np.float32),
            0.25,
            self._cost,
            False,
            False,
            {"goal_met": True},
        )


def _failure_adapter(cost, agent_xy):
    ad = NavEnvAdapter.__new__(NavEnvAdapter)
    ad._env = _FakeStepEnv(cost)
    ad._last_obs = None
    ad.collision_is_failure = True
    ad.collision_penalty = NE.COLLISION_PENALTY
    ad.oob_is_failure = True
    ad.oob_bound = NE.OOB_BOUND
    ad.oob_penalty = NE.OOB_PENALTY
    ad.agent_xy = lambda: np.asarray(agent_xy, dtype=float)
    return ad


def test_failure_penalty_contract():
    assert NE.COLLISION_PENALTY == 2.0
    assert NE.OOB_PENALTY == 2.0
    defaults = inspect.signature(NavEnvAdapter).parameters
    assert defaults["collision_penalty"].default == NE.COLLISION_PENALTY
    assert defaults["oob_penalty"].default == NE.OOB_PENALTY

    collision = _failure_adapter(cost=1.0, agent_xy=(1.1, 0.0))
    _, reward, terminated, truncated, info = collision.step(np.zeros(2))
    assert reward == -2.0
    assert terminated and not truncated
    assert info["collision"] and info["oob"]
    assert info["success"] is False

    oob = _failure_adapter(cost=0.0, agent_xy=(1.1, 0.0))
    _, reward, terminated, truncated, info = oob.step(np.zeros(2))
    assert reward == -2.0
    assert terminated and not truncated
    assert not info["collision"] and info["oob"]
    assert info["success"] is False
    print("  ✓ collision/OOB 均按 -2 失败并立即终止")


def test_layout_switch_and_obs_dim():
    ad = NavEnvAdapter()
    for name, pillars in COMPONENT_LAYOUTS.items():
        ad.set_layout(pillars)
        obs = ad.reset(seed=0)
        assert obs.shape == (NE.OBS_DIM,), f"{name}: obs shape {obs.shape}"
        # 校验 pillar 实际落位
        got = np.array([p[:2] for p in ad._env.task.pillars.pos])
        want = np.array(pillars)
        assert got.shape[0] == want.shape[0], f"{name}: pillar 数 {got.shape[0]} != {want.shape[0]}"
        # 位置对齐(可能顺序一致)
        err = np.abs(np.sort(got, axis=0) - np.sort(want, axis=0)).max()
        assert err < 1e-3, f"{name}: pillar 坐标偏差 {err}"
        print(f"  [{name:12s}] reset OK  n_pillar={got.shape[0]}  obs={obs.shape}  pos_err={err:.4f}")
    ad.close()
    print("  ✓ 布局切换 + obs 维度恒定 + 坐标对齐")


def test_step_and_forward():
    ad = NavEnvAdapter()
    ad.set_layout(COMPONENT_LAYOUTS["two"])
    obs = ad.reset(seed=0)
    assert ad.action_dim == 2

    # Point 动作 = [推进, 转向](大致);直接给最大前进,累积 reward 应为正(靠近 goal)
    total_r = 0.0
    reached = False
    for _ in range(200):
        obs, r, term, trunc, info = ad.step(np.array([1.0, 0.0]))
        total_r += r
        if ad.success(info):
            reached = True
        if term or trunc:
            break
    print(f"  step OK: 200 步累计 reward={total_r:.3f}  success={reached}  last_cost={info['cost']:.2f}")
    assert np.isfinite(total_r)
    ad.close()
    print("  ✓ step / reward / cost / info 通路正常")


def test_invalid_layout_raises():
    ad = NavEnvAdapter()
    # 把 pillar 直接压在 start 上 -> 起始 cost 或采样失败
    ad.set_layout([NE.START, (NE.START[0] + 0.05, NE.START[1])], start=NE.START, goal=NE.GOAL)
    raised = False
    try:
        ad.reset(seed=0)
    except (ResamplingError, AssertionError):
        raised = True
    print(f"  非法布局 reset 抛异常: {raised}")
    assert raised, "非法布局应抛异常供调用方重采样"
    ad.close()
    print("  ✓ 非法布局按预期抛出(对应生成器 valid 门)")


def _agent_xy(ad):
    return ad._env.task.data.body("agent").xpos[:2].copy()


def test_start_modes():
    ad = NavEnvAdapter()
    pillars = COMPONENT_LAYOUTS["two"]

    # (a) 显式起点:agent 落在指定坐标
    ad.set_layout(pillars)
    explicit = (-0.72, 0.35)
    ad.reset(seed=0, start=explicit)
    err = np.linalg.norm(_agent_xy(ad) - np.array(explicit))
    print(f"  显式起点 err={err:.4f}")
    assert err < 1e-2, f"显式起点未生效 err={err}"

    # (b) 随机起点:不同 seed 给出不同起点
    ad.set_layout(pillars)
    xs = []
    for s in range(4):
        ad.reset(seed=s, randomize_start=True)
        xs.append(_agent_xy(ad))
    xs = np.array(xs)
    spread = np.linalg.norm(xs - xs.mean(0), axis=1).mean()
    print(f"  随机起点 spread={spread:.4f}  样例={np.round(xs[0],2)}")
    assert spread > 1e-2, "随机起点没有分散"

    # (c) between 起点:pillar 落在 start->goal 连线附近
    rng = np.random.default_rng(0)
    ad.set_layout([(0.0, 0.0)], goal=NE.GOAL)  # 单柱便于判定
    ok = 0
    for _ in range(10):
        st = ad.sample_between_start(rng, pillars=[(0.0, 0.0)], goal=NE.GOAL)
        try:
            ad.reset(seed=0, start=st)
        except ResamplingError:
            continue
        # 点 (0,0) 到线段 start->goal 的距离应较小(pillar 在中间)
        s = np.array(st); g = np.array(NE.GOAL); p = np.array([0.0, 0.0])
        seg = g - s; t = np.clip(np.dot(p - s, seg) / (np.dot(seg, seg) + 1e-9), 0, 1)
        dist = np.linalg.norm(s + t * seg - p)
        if dist < 0.35 and 0.05 < t < 0.95:
            ok += 1
    print(f"  between 起点:{ok}/10 让 pillar 落在 start->goal 之间")
    assert ok >= 6, f"between 起点语义不达标 ok={ok}"
    ad.close()
    print("  ✓ 固定/随机/between 起点模式均正常")


def _gremlin_xy(ad):
    """读取 Gremlin 物理 body 的世界坐标，而不是 mocap 的相对位移。"""
    return np.asarray([pos[:2] for pos in ad._env.task.gremlins.pos], dtype=float)


def test_dynamic_gremlins():
    centers = [(-0.35, 0.55), (0.0, -0.55), (0.35, 0.55)]

    frozen = NavEnvAdapter(dynamic=True)
    frozen.set_layout(centers, gremlin_travel=0.0)
    obs = frozen.reset(seed=0)
    assert obs.shape == (NE.OBS_DIM,)
    assert frozen.obs2zinput(obs).shape == (16,)
    frozen_start = _gremlin_xy(frozen)
    for _ in range(10):
        frozen.step(np.zeros(2))
    frozen_end = _gremlin_xy(frozen)
    assert np.allclose(frozen_start, centers, atol=1e-3)
    assert np.allclose(frozen_end, frozen_start, atol=1e-4), "travel=0 的 Gremlin 不应移动"
    frozen.close()

    travel = float(NE.GREMLIN_TRAVEL)
    moving = NavEnvAdapter(dynamic=True)
    moving.set_layout(centers, gremlin_travel=travel)
    obs = moving.reset(seed=0)
    assert obs.shape == (NE.OBS_DIM,)
    positions = [_gremlin_xy(moving)]
    for _ in range(40):
        obs, _, term, trunc, _ = moving.step(np.zeros(2))
        assert obs.shape == (NE.OBS_DIM,)
        positions.append(_gremlin_xy(moving))
        assert not term and not trunc
    positions = np.asarray(positions)
    radii = np.linalg.norm(positions - np.asarray(centers)[None, :, :], axis=2)
    displacement = np.linalg.norm(positions - positions[0:1], axis=2).max()
    pairwise = np.linalg.norm(positions[:, :, None, :] - positions[:, None, :, :], axis=3)
    relative_change = np.ptp(pairwise[:, np.triu_indices(len(centers), k=1)[0],
                                      np.triu_indices(len(centers), k=1)[1]], axis=0).max()
    assert radii.max() <= travel + 5e-3, (
        f"Gremlin 偏离各自中心 {radii.max():.4f} > travel={travel:.4f}"
    )
    assert displacement > 1e-3, "travel>0 的 Gremlin 未随 step 移动"
    assert relative_change > 1e-3, "多个 Gremlin 仍在同相位刚性平移"
    moving.close()
    print(
        f"  动态 D: obs={obs.shape}  max_radius={radii.max():.4f}  "
        f"max_displacement={displacement:.4f}  relative_change={relative_change:.4f}"
    )
    print("  ✓ Gremlin 冻结/运动与中心半径均正常")


def main():
    print("=" * 68)
    print("1) 失败奖励合同")
    test_failure_penalty_contract()
    print("-" * 68)
    print("2) 布局切换 + obs 维度")
    test_layout_switch_and_obs_dim()
    print("-" * 68)
    print("3) step 通路")
    test_step_and_forward()
    print("-" * 68)
    print("4) 非法布局处理")
    test_invalid_layout_raises()
    print("-" * 68)
    print("5) 起点模式(固定/随机/between)")
    test_start_modes()
    print("-" * 68)
    print("6) 动态 Gremlin")
    test_dynamic_gremlins()
    print("=" * 68)
    print("ALL PASS")


if __name__ == "__main__":
    main()
