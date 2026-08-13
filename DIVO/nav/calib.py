"""校准:Point 能否到达 goal、需多少步。用朝向 goal 的启发式控制。"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
from nav import nav_env as NE
from nav.nav_adapter import NavEnvAdapter

ad = NavEnvAdapter()
# 一个把柱子放在路径之外、且位于当前小世界内的简单场景(不挡直线)
ad.set_layout([(0.0, 0.55)])
obs = ad.reset(seed=0)

# 观察 Point 的 body 朝向与到 goal 的信息
task = ad._env.task
def agent_xy():
    return task.data.body("agent").xpos[:2].copy()

print("start agent xy:", np.round(agent_xy(), 3), "goal:", ad.goal)

# 启发式:用 goal 相对方位控制转向 + 前进
best_d = 1e9
reached_step = None
for t in range(800):
    axy = agent_xy()
    gxy = np.array(ad.goal)
    vec = gxy - axy
    # agent 朝向:从 body xmat 取前向
    mat = task.data.body("agent").xmat.reshape(3, 3)
    heading = np.arctan2(mat[1, 0], mat[0, 0])
    target = np.arctan2(vec[1], vec[0])
    ang = (target - heading + np.pi) % (2 * np.pi) - np.pi
    a = np.array([1.0, np.clip(2.0 * ang, -1, 1)])  # [forward, turn]
    obs, r, term, trunc, info = ad.step(a)
    d = np.linalg.norm(np.array(ad.goal) - agent_xy())
    best_d = min(best_d, d)
    if ad.success(info) and reached_step is None:
        reached_step = t + 1
        break
    if term or trunc:
        break

print(f"reached_step={reached_step}  min_dist_to_goal={best_d:.3f}  goal_size={task.goal.size}")
ad.close()
