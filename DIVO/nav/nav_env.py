"""导航任务(SafetyPointGoal + 自定义 pillar 布局)—— 单一事实来源。

本文件统一管理导航环境的全部核心定义,供训练/课程/验证复用:
  - 坐标系与 pillar 物理参数(固定 DOF)
  - obs -> (state, z) 切分索引
  - NavPillarTask:GoalLevel0 + 显式 start/goal + 显式坐标 pillar
  - make_env:运行时注入 layout 造 env(对应 Push-T 的 set_obstacle_config)
  - 可选结构化压力测试 + 训练分布采样器(受限 DOF,防泄漏)

环境依赖:conda env `safenav`(safety-gymnasium 1.0.0 / gymnasium 0.28.1 /
mujoco 2.3.3),与 Push-T 的 `divo` 环境隔离。运行需 MUJOCO_GL=egl。

============================ 关键设计常量 ============================
坐标系(小世界:整体等比缩小 ~×0.5,让固定尺寸 pillar 相对世界更大 => 自然挡路,贴 Push-T):
    start ≈ (-0.65, 0.0)   goal = (+0.65, 0.0)   直线路径 y=0
    extents = [-0.9, -0.9, 0.9, 0.9];OOB ±1.0;goal 成功半径 0.2
pillar 物理(固定,不缩,不进课程 DOF):
    size = 0.15, keepout = 0.13
    Point 智能体半径 = 0.10  => 可通行间隙需 >= 0.20;柱间距 0.30 => 完全挡死
    placement keepout 门仅要求 0.26,但训练按实体直径另加余量,中心距至少 0.34

obs 结构(44 维,pillar 版):
    accelerometer[0:3] velocimeter[3:6] gyro[6:9] magnetometer[9:12]
    goal_lidar[12:28]  pillars_lidar[28:44]
  => state = obs[0:28](本体+目标方向,对应 Push-T 的 obs2state)
  => obstacle slice = obs[28:44](障碍 lidar；obs2zinput 兼容接口)
  注:当前 SkillActor 的 z_encoder 吃完整 obs[0:44]，不是只吃该 16 维切片。

训练生成器 DOF 边界(课程只能在此范围内进化,B/M/U/D 结构不在其中):
    num_pillars = 2;每根 x,y in [-0.5, 0.5](= ±OBSTACLE_REGION)
    size/keepout 固定,起点采样分布固定(不由课程进化),goal 固定
  => 训练分布 = 中央带内两根稀疏散柱;参数化 B/M/U/D 另见 benchmarks.py
"""
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
from nav.protocol import TRAIN_NUM_PILLARS
from safety_gymnasium import tasks
from safety_gymnasium.builder import Builder
from safety_gymnasium.assets.geoms import Pillars
from safety_gymnasium.assets.mocaps import Gremlins, MOCAPS_REGISTER
from safety_gymnasium.tasks.safe_navigation.goal.goal_level0 import GoalLevel0

# ----------------------------- 常量 -----------------------------
# 小世界:坐标几何整体 ~×0.5 缩小;pillar/agent/keepout 为物理尺寸,不缩。
START = (-0.65, 0.0)
GOAL = (0.65, 0.0)
EXTENTS = [-0.9, -0.9, 0.9, 0.9]

PILLAR_SIZE = 0.15
PILLAR_KEEPOUT = 0.13
PILLAR_HEIGHT = 0.15    # 圆柱半高(总高=2*height=0.30);矮柱更美观,不影响 lidar/碰撞/训练
PILLAR_MIN_SEPARATION = max(2 * PILLAR_SIZE, 2 * PILLAR_KEEPOUT) + 0.04
AGENT_RADIUS = 0.10
GOAL_SIZE = 0.2         # goal 成功半径(小世界里缩小,否则太好达)
# goal/agent 的放置 keepout(默认 0.305/0.4 对 ±0.9 小世界太大,和 pillar 冲突 -> reset 失败);缩小
GOAL_KEEPOUT = 0.18
AGENT_KEEPOUT = 0.18

# D(动态)用错相连续运动的 Gremlins 顶替 pillar(obs 仍 44,gremlin_lidar 占原 pillar 通道)
GREMLIN_SIZE = 0.10
GREMLIN_TRAVEL = 0.08     # 每个 gremlin 绕自己中心的小圆半径
GREMLIN_KEEPOUT = 0.20    # 放置 keepout(需容纳 travel+size≈0.18 的移动路径)

# 出界失败(方案a):agent |x| 或 |y| 超过 OOB_BOUND 即判失败(对齐 Push-T "关在工作区内")
OOB_BOUND = 1.0                  # 贴 EXTENTS(±0.9)边 + 余量;越界即失败,竖直绕行受限
# Nav 完整成功回报约为 +2；失败惩罚按同一相对尺度设置，避免 -10 主导距离进展奖励。
COLLISION_PENALTY = 2.0
OOB_PENALTY = 2.0
# 起点平凡性拒绝(对齐 Push-T 拒绝"太靠目标=太易"的起点)
MIN_START_GOAL_DIST = 0.5

# 规范起点区(偏 goal 对侧的左半场,所有阶段共用;小世界缩小版)
# goal 在 +0.65,起点 x<=-0.1 => 障碍(±0.5)始终在 start->goal 之间
START_X_RANGE = (-0.8, -0.1)
START_Y_RANGE = (-0.45, 0.45)
CLEARANCE = PILLAR_SIZE + AGENT_RADIUS  # 圆心距 < 0.25 视为挡住 Point
# 训练/validation 共用的起点机动净空；0.45 中心距对应 0.20 实体表面间隙。
START_PILLAR_CLEARANCE = 0.45
# Push-T `between` 结构的 Nav 尺度：障碍先采样，起点位于选中障碍背离 goal 一侧。
BETWEEN_OFFSET_RANGE = (0.45, 0.65)
BETWEEN_LATERAL_RANGE = (-0.10, 0.10)

# obs -> (state, z) 切分
OBS_DIM = 44
STATE_SLICE = slice(0, 28)      # 本体传感器 + goal_lidar
Z_INPUT_SLICE = slice(28, 44)   # pillars/gremlins lidar（兼容名；encoder 实际吃完整 obs）

# 规范障碍区(所有阶段共用,只变 num/size;对齐 Push-T"障碍范围一致、只变 num/size")
OBSTACLE_REGION = 0.5           # 主障碍区 x,y ∈ [-0.5, 0.5](start/goal 在 ±0.65,留净空)
VAL_OBSTACLE_REGION = OBSTACLE_REGION  # 验证与训练共用同一障碍中心支持集

# 训练 DOF(障碍区统一到 ±OBSTACLE_REGION,只保留 num 作 DOF)
# 兼容旧调用方对该常量的引用;训练协议现固定为 2 根。
TRAIN_NUM_RANGE = (TRAIN_NUM_PILLARS, TRAIN_NUM_PILLARS + 1)
TRAIN_X_RANGE = (-OBSTACLE_REGION, OBSTACLE_REGION)
TRAIN_Y_RANGE = (-OBSTACLE_REGION, OBSTACLE_REGION)

# make_env 在构造前注入
_LAYOUT = {"start": START, "goal": GOAL, "pillars": []}


def dedupe(points, tol=1e-6):
    """去重合点(墙角会重复),返回四舍五入后的坐标列表。"""
    out = []
    for p in points:
        if not any(abs(p[0] - q[0]) < tol and abs(p[1] - q[1]) < tol for q in out):
            out.append((round(float(p[0]), 4), round(float(p[1]), 4)))
    return out


# --------------------------- 任务定义 ---------------------------
class NavPillarTask(GoalLevel0):
    """GoalLevel0 + 固定 start/goal + 由 _LAYOUT 注入的显式坐标 pillar 布局。"""

    def __init__(self, config) -> None:
        super().__init__(config=config)
        self.placements_conf.extents = list(EXTENTS)

        pillars = dedupe(_LAYOUT["pillars"])
        if pillars:
            self._add_geoms(
                Pillars(
                    num=len(pillars),
                    locations=pillars,
                    size=PILLAR_SIZE,
                    height=PILLAR_HEIGHT,
                    keepout=PILLAR_KEEPOUT,
                    is_constrained=True,
                )
            )
        self.agent.locations = [tuple(_LAYOUT["start"])]
        self.goal.locations = [tuple(_LAYOUT["goal"])]
        self.goal.size = GOAL_SIZE          # 成功半径(小世界缩小)
        self.goal.keepout = GOAL_KEEPOUT    # 放置 keepout 缩小(默认 0.305 对小世界太大)
        self.agent.keepout = AGENT_KEEPOUT  # 默认 0.4 对小世界太大 -> 和 pillar 冲突
        # 到达 goal 即终止(否则默认会重采样新 goal,破坏固定 goal 设定)
        self.mechanism_conf.continue_goal = False


class NavGremlins(Gremlins):
    """原生 Gremlin 物理与传感器，使用确定性的错相连续圆周运动。

    上游实现让所有实例同相位平移；这里为第 i 个实例加入固定相位差，使多个
    动态障碍的相对几何也随时间变化，同时保持每个障碍都在 travel 半径内。
    """

    def move(self):
        phase = float(self.engine.data.time)
        count = max(int(self.num), 1)
        for i in range(self.num):
            theta = phase + 2 * np.pi * i / count
            target = np.array([np.sin(theta), np.cos(theta)]) * self.travel
            self.set_mocap_pos(f"gremlin{i}mocap", np.r_[target, [self.size]])


# Safety-Gymnasium 1.0.0 对 mocap 使用精确 type 白名单；显式注册本地运动类。
if NavGremlins not in MOCAPS_REGISTER:
    MOCAPS_REGISTER.append(NavGremlins)


class NavGremlinTask(GoalLevel0):
    """D(动态)任务:与 NavPillarTask 同布局,但障碍换成会动的 Gremlins(顶替 pillar)。

    obs 仍 44 维(gremlin_lidar 占原 pillar 通道 [28:44]);dist_cost=0 => 接触才失败
    (与 pillar 一致);障碍采用任务特定的错相连续周期运动。
    """

    def __init__(self, config) -> None:
        super().__init__(config=config)
        self.placements_conf.extents = list(EXTENTS)

        grem = dedupe(_LAYOUT["pillars"])       # 复用 pillars 键作为 gremlin 中心
        if grem:
            self._add_mocaps(
                NavGremlins(
                    num=len(grem),
                    locations=grem,
                    size=GREMLIN_SIZE,
                    travel=GREMLIN_TRAVEL,
                    keepout=GREMLIN_KEEPOUT,
                    contact_cost=1.0,
                    dist_cost=0.0,           # 接触才算失败(对齐 pillar)
                    is_constrained=True,
                )
            )
        self.agent.locations = [tuple(_LAYOUT["start"])]
        self.goal.locations = [tuple(_LAYOUT["goal"])]
        self.goal.size = GOAL_SIZE
        self.goal.keepout = GOAL_KEEPOUT
        self.agent.keepout = AGENT_KEEPOUT
        self.mechanism_conf.continue_goal = False


# 安装版 1.0.0 的 task 类名由 task_id 解析:'SafetyPointGoalN-v0' -> 'GoalLevelN'
setattr(tasks, "GoalLevel3", NavPillarTask)     # 静态 pillar(训练/验证/B/M/U)
setattr(tasks, "GoalLevel4", NavGremlinTask)    # 动态 gremlin(D)


def make_env(layout, render_mode=None, width=640, height=640, camera_name="fixedfar",
             dynamic=False):
    """用给定 layout 造一个全新 env。dynamic=True -> gremlin(D 动态)任务,否则 pillar 任务。

    layout: {"start": (x,y), "goal": (x,y), "pillars": [(x,y), ...]}(dynamic 时 pillars 作 gremlin 中心)
    """
    global _LAYOUT
    _LAYOUT = layout
    return Builder(
        task_id="SafetyPointGoal4-v0" if dynamic else "SafetyPointGoal3-v0",
        config={"agent_name": "Point"},
        render_mode=render_mode,
        width=width,
        height=height,
        camera_name=camera_name,
    )


# ------------------ 结构压力测试 + 训练采样 ------------------
def _vwall(x, ys):
    return [(x, y) for y in ys]


def _hwall(y, xs):
    return [(x, y) for x in xs]


def template_barrier():
    """单道竖墙,缺口在上方 -> 一次绕行。中心间距 0.30,满足几何与 keepout 约束。"""
    return _vwall(0.0, [-0.75, -0.45, -0.15, 0.15, 0.45])


def template_maze():
    """两道错位墙:wall1 下开口,wall2 上开口 -> S 形。"""
    w1 = _vwall(-0.3, [-0.15, 0.15, 0.45, 0.75])
    w2 = _vwall(0.3, [-0.75, -0.45, -0.15, 0.15])
    return w1 + w2


def template_ushape():
    """开口朝 goal(+x)的杯形,back 墙在 start 侧 -> 绕杯外侧。"""
    back = _vwall(-0.3, [-0.6, -0.3, 0.0, 0.3, 0.6])
    top = _hwall(0.6, [-0.3, 0.0, 0.3])
    bot = _hwall(-0.6, [-0.3, 0.0, 0.3])
    return back + top + bot


def template_deadend():
    """开口朝 start(-x)的口袋,back 墙在 goal 侧 -> 需先回退再绕行。"""
    back = _vwall(0.3, [-0.45, -0.15, 0.15, 0.45])
    top = _hwall(0.45, [-0.3, 0.0, 0.3])
    bot = _hwall(-0.45, [-0.3, 0.0, 0.3])
    return back + top + bot


STRUCTURED_STRESS_TESTS = {
    "barrier": template_barrier,
    "maze": template_maze,
    "ushape": template_ushape,
    "deadend": template_deadend,
}
# 兼容旧的调用代码;键名已不再借用 B/M/U/D 正式基准名称。
TEMPLATES = STRUCTURED_STRESS_TESTS


def sample_training_layout(rng):
    """训练 DOF 内的一个样例:中央带内固定 2 根稀疏散柱。"""
    num = TRAIN_NUM_PILLARS
    pts, tries = [], 0
    min_sep = PILLAR_MIN_SEPARATION
    while len(pts) < num and tries < 200:
        tries += 1
        x = round(float(rng.uniform(*TRAIN_X_RANGE)), 3)
        y = round(float(rng.uniform(*TRAIN_Y_RANGE)), 3)
        if all(np.hypot(x - px, y - py) >= min_sep for px, py in pts):
            pts.append((x, y))
    if len(pts) != num:
        raise RuntimeError(f"训练布局采样失败:expected {num} pillars, got {len(pts)}")
    return pts


# ------------------------- obs 便捷切分 -------------------------
def start_goal_ok(start, goal, min_dist=MIN_START_GOAL_DIST):
    """非平凡起点判据:起点离 goal 不能太近(太近=直冲即达,平凡)。"""
    return float(np.hypot(start[0] - goal[0], start[1] - goal[1])) >= float(min_dist)


def sample_valid_start(rng, goal=GOAL, min_dist=MIN_START_GOAL_DIST, tries=50):
    """全阶段统一的起点采样器(对齐 Push-T 全程用一个 sample_valid_tblock_pose)。

    在规范起点区 x∈START_X_RANGE, y∈START_Y_RANGE 内均匀采样,拒绝离 goal
    太近的平凡起点(此走廊区内 dist(goal)>=0.75 恒成立,该拒绝仅作冗余兜底)。
    """
    for _ in range(int(tries)):
        x = float(rng.uniform(*START_X_RANGE))
        y = float(rng.uniform(*START_Y_RANGE))
        if start_goal_ok((x, y), goal, min_dist):
            return (round(x, 3), round(y, 3))
    return (-0.65, 0.0)


def obstacle_on_path(pillars, start, goal, clearance=None):
    """是否有 pillar 落在 start->goal 连线附近(障碍是否真的挡路)。"""
    if not pillars:
        return False
    clr = (PILLAR_SIZE + AGENT_RADIUS) if clearance is None else clearance
    s = np.asarray(start, float); g = np.asarray(goal, float); seg = g - s
    denom = float(seg @ seg) + 1e-9
    for x, y in pillars:
        p = np.array([x, y]); t = np.clip((p - s) @ seg / denom, 0, 1)
        if np.linalg.norm(s + t * seg - p) < clr:
            return True
    return False


def obs2state(obs):
    """本体 + 目标方向(decoder 的 state 通道)。"""
    return np.asarray(obs)[..., STATE_SLICE]


def obs2zinput(obs):
    """返回障碍 lidar 诊断切片；保留旧接口名，当前 encoder 实际输入完整 obs。"""
    return np.asarray(obs)[..., Z_INPUT_SLICE]
