"""NavEnvAdapter —— 导航环境的运行时适配层。

对应 Push-T 的 `set_obstacle_config`:建一次底层 env,之后每换一个布局
(pillar 坐标/数量、start/goal)就地重建,而不新建进程级 env。

统一对外接口(供 TD3 训练 / 库探针 / 课程 / 验证复用):
    adapter.set_layout(pillars, start=None, goal=None, gremlin_travel=None)
                                                               # 注入布局(下次 reset 生效)
    obs = adapter.reset(seed=None, start=None, randomize_start=False)   # -> flat obs (44,)
    obs, reward, terminated, truncated, info = adapter.step(action)
    adapter.success(info) -> bool
    adapter.obs2state(obs) / adapter.obs2zinput(obs)
    adapter.sample_between_start(rng, pillars=None, goal=None)  # between 起点(验证分布用)

起点模式:
- 固定起点(默认):reset() 用 self._start(信号探针给定 fixed_start,或默认坐标)。
- 显式起点:reset(start=(x,y)) 把 agent 放在指定坐标(between 起点即走这条:先算好坐标再传入)。
- 随机起点:reset(randomize_start=True) 让 Safety-Gym 在避开 pillar/goal keepout 下全场采样
  （通用组件/展示用途；正式 B/M/U/D 使用 NE.START 固定起点）。

约定:
- pillar 数量可在两次 reset 间变化(内部走 world.rebuild),但恒 >= 1(obs 恒 44 维)。
- 布局非法(keepout/净空/reset 失败)抛 ResamplingError,调用方重采样(= 生成器 valid 门)。
"""
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

from nav import nav_env as NE
from safety_gymnasium.utils.common_utils import ResamplingError


class NavEnvAdapter:
    def __init__(self, start=NE.START, goal=NE.GOAL, render_mode=None, seed=0,
                 camera_name="fixedfar", collision_is_failure=True,
                 collision_penalty=NE.COLLISION_PENALTY,
                 oob_is_failure=True, oob_bound=NE.OOB_BOUND,
                 oob_penalty=NE.OOB_PENALTY,
                 dynamic=False, gremlin_travel=None):
        # dynamic=True:障碍用会动的 Gremlins 顶替 pillar(D 动态;obs 仍 44)。
        self.dynamic = bool(dynamic)
        # 对齐 Push-T:碰到障碍即失败(done + 惩罚),训练/测试同一规则。
        self.collision_is_failure = bool(collision_is_failure)
        self.collision_penalty = float(collision_penalty)
        # 出界即失败(方案a):agent 越过 oob_bound 即 done + 惩罚。
        self.oob_is_failure = bool(oob_is_failure)
        self.oob_bound = float(oob_bound)
        self.oob_penalty = float(oob_penalty)
        self._start = tuple(start)
        self._goal = tuple(goal)
        self._pillars = [(0.0, 0.9)]  # 当前底层 task 里的 pillar(占位 1 根 => obs 恒 44)
        self._pillar_size = NE.GREMLIN_SIZE if self.dynamic else NE.PILLAR_SIZE
        self._pillar_keepout = NE.GREMLIN_KEEPOUT if self.dynamic else NE.PILLAR_KEEPOUT
        self._pillar_height = NE.PILLAR_HEIGHT
        self._default_gremlin_travel = (NE.GREMLIN_TRAVEL if gremlin_travel is None
                                        else float(gremlin_travel))
        self._gremlin_travel = self._default_gremlin_travel
        init_layout = {"start": self._start, "goal": self._goal, "pillars": list(self._pillars)}
        self._env = NE.make_env(init_layout, render_mode=render_mode, camera_name=camera_name,
                                dynamic=self.dynamic)
        self._last_obs = None
        self._default_seed = seed

    # ------------------------- 布局注入 -------------------------
    def set_layout(self, pillars, start=None, goal=None, pillar_size=None, pillar_keepout=None,
                   pillar_height=None, gremlin_travel=None):
        """设置下次 reset 的布局。pillars: [(x,y), ...](>=1)。

        pillar_size / pillar_keepout:可选,供 B/M/U/D 基准调物理尺寸。
        静态默认为 PILLAR_*,动态默认为 GREMLIN_*;gremlin_travel=0 可构造
        与 D 几何完全匹配的冻结对照。
        """
        pillars = NE.dedupe(list(pillars))
        assert len(pillars) >= 1, "至少需要 1 根 pillar(保证 obs 维度恒定)"
        self._pillars = pillars
        # None => 回落 nav_env 默认(不保留上一次的值,避免跨布局泄漏)
        default_size = NE.GREMLIN_SIZE if self.dynamic else NE.PILLAR_SIZE
        default_keepout = NE.GREMLIN_KEEPOUT if self.dynamic else NE.PILLAR_KEEPOUT
        self._pillar_size = default_size if pillar_size is None else float(pillar_size)
        self._pillar_keepout = default_keepout if pillar_keepout is None else float(pillar_keepout)
        self._pillar_height = NE.PILLAR_HEIGHT if pillar_height is None else float(pillar_height)
        self._gremlin_travel = (self._default_gremlin_travel if gremlin_travel is None
                                else float(gremlin_travel))
        if start is not None:
            self._start = tuple(start)
        if goal is not None:
            self._goal = tuple(goal)

    def _prepare(self, start, randomize_start):
        """把当前布局 + 起点模式写进底层 task,并强制重建 placements。"""
        task = self._env.task
        if self.dynamic:
            # 动态:障碍是会动的 gremlin(顶替 pillar);用同一批坐标作 gremlin 中心
            g = task.gremlins
            g.num = len(self._pillars)
            g.locations = list(self._pillars)
            g.size = float(self._pillar_size)
            g.keepout = float(self._pillar_keepout)
            g.travel = float(self._gremlin_travel)
        else:
            task.pillars.num = len(self._pillars)
            task.pillars.locations = list(self._pillars)
            task.pillars.size = float(self._pillar_size)   # 每次显式写入,防跨布局泄漏
            task.pillars.keepout = float(self._pillar_keepout)
            task.pillars.height = float(self._pillar_height)
        task.goal.locations = [self._goal]
        if randomize_start:
            task.agent.locations = []                 # 让 safety-gym 均匀采起点
        else:
            fixed = tuple(start) if start is not None else self._start
            task.agent.locations = [fixed]
        # 关键:置空 placements 触发 _build_placements_dict 重读新 locations/num/起点模式
        task.placements_conf.placements = None

    # --------------------------- 交互 ---------------------------
    def reset(self, seed=None, start=None, randomize_start=False):
        self._prepare(start, randomize_start)
        s = self._default_seed if seed is None else seed
        obs, _ = self._env.reset(seed=s)  # 非法布局抛 ResamplingError,交由调用方处理
        self._last_obs = np.asarray(obs, dtype=np.float32)
        return self._last_obs

    def step(self, action):
        action = np.asarray(action, dtype=np.float64).reshape(self.action_dim)
        obs, reward, cost, terminated, truncated, info = self._env.step(action)
        obs = np.asarray(obs, dtype=np.float32)
        self._last_obs = obs
        info = dict(info)
        info["cost"] = float(cost)
        collided = float(cost) > 0.0          # NavPillarTask 只有 pillar 有 cost => cost>0 即碰撞
        info["collision"] = bool(collided)
        reward = float(reward)
        goal_met = bool(info.get("goal_met", False))
        axy = self.agent_xy()
        oob = bool(max(abs(float(axy[0])), abs(float(axy[1]))) > self.oob_bound)
        info["oob"] = oob
        if self.collision_is_failure and collided:
            # 碰撞即失败:终止 + 惩罚 + 不算 success(碰撞优先于到达,同 Push-T)
            terminated = True
            reward = -self.collision_penalty
            info["success"] = False
        elif self.oob_is_failure and oob:
            # 出界即失败(方案a)
            terminated = True
            reward = -self.oob_penalty
            info["success"] = False
        else:
            info["success"] = goal_met
        return obs, reward, bool(terminated), bool(truncated), info

    def agent_xy(self):
        return self._env.task.data.body("agent").xpos[:2].copy()

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()

    # ------------------------- 起点采样器 -------------------------
    def sample_between_start(self, rng, pillars=None, goal=None):
        """DIVO `between` 起点:把 agent 放在"某 pillar 与 goal 之间"的另一侧,
        使该 pillar 落在 start->goal 连线附近(先摆障碍、再拒绝采样起点)。

        越界或未通过共享生成器净空检查的候选会被拒绝,不做 clip。
        """
        P = np.asarray(pillars if pillars is not None else self._pillars, dtype=float)
        g = np.asarray(goal if goal is not None else self._goal, dtype=float)
        from nav.curriculum.generator_source import validate_pillars

        for _ in range(50):
            # 随机挑一根 pillar 作"被绕过"的障碍
            p = P[int(rng.integers(len(P)))]
            d = p - g
            n = float(np.linalg.norm(d))
            if n < 1e-6:
                continue
            u = d / n                              # 从 goal 指向 pillar 的方向
            perp = np.array([-u[1], u[0]])
            offset = float(rng.uniform(*NE.BETWEEN_OFFSET_RANGE))
            lateral = float(rng.uniform(*NE.BETWEEN_LATERAL_RANGE))
            start = p + u * offset + perp * lateral
            sx, sy = float(start[0]), float(start[1])
            if not (
                NE.START_X_RANGE[0] <= sx <= NE.START_X_RANGE[1]
                and NE.START_Y_RANGE[0] <= sy <= NE.START_Y_RANGE[1]
            ):
                continue
            candidate = (sx, sy)
            if not NE.start_goal_ok(candidate, g):
                continue
            ok, _ = validate_pillars(
                [tuple(xy) for xy in P], candidate, g, num=len(P),
            )
            if ok:
                return candidate
        raise ResamplingError

    # ------------------------- 便捷/属性 -------------------------
    @staticmethod
    def obs2state(obs):
        return NE.obs2state(obs)

    @staticmethod
    def obs2zinput(obs):
        return NE.obs2zinput(obs)

    @staticmethod
    def success(info):
        return bool(info.get("success", info.get("goal_met", False)))

    @property
    def obs_dim(self):
        return NE.OBS_DIM

    @property
    def action_dim(self):
        return int(np.prod(self._env.action_space.shape))

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def start(self):
        return self._start

    @property
    def goal(self):
        return self._goal
