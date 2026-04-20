"""
LLM Topology Generator - Phase 0 拓扑生成器

核心功能：
1. 使用 LLM 生成拓扑生成器代码（Python 函数）
2. StrategyExecutor 安全执行生成的代码
3. 基于 Eurekaverse, OMNI, Voyager 文献的 Few-shot prompt 设计

与 V3 的区别：
- V3: 生成具体坐标 [{'x': 0.07, 'y': 0.07}, ...]
- V4: 生成 Python 函数，可根据不同 tblock_pose 自适应生成障碍物
"""
import numpy as np
import os
import logging
import importlib
import signal
from typing import List, Dict, Optional
from DIVO.utils.util import analytic_obs_collision_check

# 延迟导入 openai：避免开发环境未安装时触发静态导入报错
openai = None
_OPENAI_IMPORT_ERROR = None
try:
    openai = importlib.import_module("openai")
except Exception as exc:
    _OPENAI_IMPORT_ERROR = exc

# 屏蔽 httpx 的 HTTP 请求日志
logging.getLogger("httpx").setLevel(logging.WARNING)


# ============================================================================
# Phase 0: 拓扑生成器 Prompt
# ============================================================================

def build_phase0_topology_generator_prompt(
    tblock_pose: np.ndarray, 
    num_obstacles: int = 2
) -> str:
    """
    Phase 0: 冷启动阶段的拓扑生成器 prompt
    
    设计原则（基于 Eurekaverse, OMNI, Voyager 文献）：
    1. 提供 2-3 个 few-shot 示例（启发而非模板）
    2. 明确约束和可用 API
    3. 给出设计原则（不限制具体实现）
    4. 结构化输出格式
    5. 不预定义模式列表
    
    Args:
        tblock_pose: T-block 起点位姿 [x, y, θ]
        num_obstacles: 障碍物数量
    
    Returns:
        完整的 prompt 字符串
    """
    tx, ty, ttheta = tblock_pose
    
    prompt = f"""你是一个强化学习环境设计专家。你的任务是为Push-T机器人操作任务编写一个障碍物生成函数。

## 任务背景

**Push-T 任务**：控制一个圆形推杆，将T形方块从随机起点推到固定终点，并对齐角度。

**物理参数**：
- 工作空间：0.5m × 0.5m，坐标范围 x, y ∈ [-0.2, 0.2]
- T-block（被推物体）：宽 0.10m × 高 0.12m，不对称 T 形
- 障碍物：0.02m × 0.02m 方块，固定在桌面上
- 推杆：直径 0.02m 圆形

**当前 episode 配置**：
- 起点：x = {tx:.3f}m, y = {ty:.3f}m, θ = {np.degrees(ttheta):.0f}°
- 终点：x = 0.0m, y = 0.0m, θ = -45°（固定不变）

## 训练阶段：Phase 0（冷启动）

这是策略网络训练的**初始阶段**，机器人刚开始学习。

**训练目标**：
- 让机器人学会基本的推动操作（直线推、简单避障）
- 在有轻微空间干扰的环境中保持稳定
- 积累初始经验，为后续更难的阶段做准备

**难度要求**：
- 简单（easy）
- **不要完全阻挡**从起点到终点的直线路径
- 障碍物应该提供"存在感"和轻微压迫感，但不过度干扰
- 让机器人能够在 80% 以上的尝试中成功完成任务

---

## 你的任务

编写一个 Python 函数 `generate_obstacles(tblock_pose, num_obstacles)`，该函数能够：
1. 根据 T-block 起点位姿生成障碍物配置
2. 每次调用生成不同的配置（多样性）
3. 确保生成的配置满足物理约束

---

## 函数接口规范（必须严格遵守）

**函数签名**：
```python
def generate_obstacles(tblock_pose: np.ndarray, num_obstacles: int) -> list:
    \"\"\"
    生成障碍物配置
    
    Args:
        tblock_pose: numpy 数组 [x, y, θ]，T-block 起点位姿
        num_obstacles: 整数，需要生成的障碍物数量
    
    Returns:
        障碍物列表，每个元素是字典：
        [{{'x': float, 'y': float, 'purpose': str}}, ...]
    \"\"\"
    # 你的代码
    return obstacles
```

**输入**：
- `tblock_pose`: numpy 数组，形状 (3,)，包含 [x, y, θ]
- `num_obstacles`: 整数，本次需要生成 {num_obstacles} 个障碍物

**输出**：
- Python 列表，长度为 `num_obstacles`
- 每个元素是字典，必须包含三个字段：
  - `'x'`: float，障碍物中心 x 坐标（米）
  - `'y'`: float，障碍物中心 y 坐标（米）
  - `'purpose'`: str，简短描述该障碍物的作用（如"路径侧方干扰"）

---

## 物理约束（必须严格遵守）

### 1. 碰撞检测（核心约束）
- **所有障碍物**必须通过 SAT（分离轴定理）碰撞检测
- 系统会检查障碍物是否与起点或终点的 T-block 碰撞
- T-block 尺寸：宽 0.10m × 高 0.12m，不对称 T 形
- SAT 会考虑 T-block 的真实形状和旋转角度
- 如果碰撞，该配置会被拒绝

### 2. 障碍物间距
- 任意两个障碍物中心之间的距离必须 > 0.03m
- 原因：障碍物尺寸为 0.02m × 0.02m，避免重叠

### 3. 坐标范围
- x, y 必须在 [-0.2, 0.2] 范围内
- 建议：使用 [-0.2, 0.2]，避免超出可用工作区

---

## 可用工具和库

### 可以导入的库
```python
import numpy as np
```

### 系统提供的辅助函数
```python
is_safe(obs_x: float, obs_y: float, 
        tblock_x: float, tblock_y: float, 
        tblock_theta: float) -> bool
```
**功能**：检查障碍物是否与 T-block 安全（使用精确的 SAT 碰撞检测）

**参数**：
- `obs_x, obs_y`: 障碍物中心坐标
- `tblock_x, tblock_y, tblock_theta`: T-block 位姿

**返回**：
- `True`: 安全，可以使用该位置
- `False`: 不安全，会与起点或终点 T-block 碰撞

**注意**：该函数会同时检查与起点和终点两个 T-block 的碰撞

---

## 设计原则（供参考，不是强制要求）

在设计障碍物布局时，你可以考虑（但不限于）：

1. **空间关系推理**：
   - 起点和终点的相对位置（距离、方向）
   - 从起点到终点的路径方向
   - 路径的垂直方向（侧方）
   - T-block 的朝向（θ 角度）

2. **Phase 0 的布局哲学**：
   - Phase 0 是冷启动阶段，目标是让机器人学会基础推动
   - **推荐策略**：
     * 将障碍物放在路径的侧方（垂直于起点→终点方向）
     * 将障碍物放在路径的中段或远端（不要太靠近起点正前方）
     * 将障碍物放在工作台的边缘或角落
   - **避免策略**：
     * 不要将障碍物放在起点正前方（会立即阻挡推动）
     * 不要将障碍物密集地放在起点→终点的连线上
   - **关键**：SAT 会自动检查碰撞，你不需要手动计算"距离起点多远"

3. **多样性**：
   - 使用 `np.random.uniform()` 添加随机扰动
   - 避免每次生成完全相同的配置
   - 该函数会被调用多次（不同的 `tblock_pose`）

4. **鲁棒性**：
   - 使用拒绝采样确保生成的障碍物满足约束
   - 建议最多尝试 50 次，避免无限循环
   - 如果 50 次都失败，返回已生成的障碍物（可能少于 `num_obstacles`）

---

## 示例代码（仅供启发，不要照抄）

以下提供三个示例，展示不同的设计思路。这些示例**仅供参考**，你可以创造完全不同的布局策略。

### 示例 1：路径侧方布局策略

**设计思路**：在起点→终点路径的两侧放置障碍物，提供轻微的"通道感"，但不阻挡路径。

```python
def generate_obstacles(tblock_pose: np.ndarray, num_obstacles: int) -> list:
    import numpy as np
    
    tx, ty, ttheta = tblock_pose
    obstacles = []
    
    # 计算路径方向（起点→终点）
    path_vec = np.array([-tx, -ty])  # 指向终点 (0, 0)
    path_length = np.linalg.norm(path_vec)
    
    if path_length > 0.01:
        path_unit = path_vec / path_length
        # 垂直于路径的方向
        perp_unit = np.array([-path_unit[1], path_unit[0]])
    else:
        # 起点太接近终点，使用默认方向
        perp_unit = np.array([0, 1])
    
    for i in range(num_obstacles):
        for attempt in range(50):
            # 在路径的中段位置（40-70%）
            t = 0.4 + i * 0.3 + np.random.uniform(-0.1, 0.1)
            t = np.clip(t, 0.3, 0.8)
            
            # 路径上的点
            point_on_path = np.array([tx, ty]) * (1 - t)
            
            # 向路径垂直方向偏移（交替放在两侧）
            side = 1 if i % 2 == 0 else -1
            offset_dist = 0.10 + np.random.uniform(-0.02, 0.02)
            
            candidate = point_on_path + side * offset_dist * perp_unit
            cand_x, cand_y = candidate
            
            # 限制范围
            cand_x = np.clip(cand_x, -0.2, 0.2)
            cand_y = np.clip(cand_y, -0.2, 0.2)
            
            # 检查安全性（SAT 碰撞检测）
            if is_safe(cand_x, cand_y, tx, ty, ttheta):
                obstacles.append({{
                    'x': float(cand_x),
                    'y': float(cand_y),
                    'purpose': f'路径{{"左" if side > 0 else "右"}}侧压迫'
                }})
                break
    
    return obstacles
```

### 示例 2：象限分散布局策略

**设计思路**：根据起点所在的象限，在对侧或边缘区域分散放置障碍物，避开起点→终点的直线路径。

```python
def generate_obstacles(tblock_pose: np.ndarray, num_obstacles: int) -> list:
    import numpy as np
    
    tx, ty, ttheta = tblock_pose
    obstacles = []
    
    # 计算起点的角度（相对于终点）
    start_angle = np.arctan2(ty, tx)
    
    for i in range(num_obstacles):
        for attempt in range(50):
            # 在起点角度的侧方分散放置（避开起点→终点方向）
            # 偏移 ±60° 到 ±120°
            angle_offset = (60 + i * 60) * np.pi / 180
            if i % 2 == 0:
                angle_offset = -angle_offset
            
            obstacle_angle = start_angle + angle_offset
            
            # 距离原点（终点）0.12-0.18m
            radius = 0.12 + np.random.uniform(0, 0.06)
            
            cand_x = radius * np.cos(obstacle_angle)
            cand_y = radius * np.sin(obstacle_angle)
            
            # 限制范围
            cand_x = np.clip(cand_x, -0.2, 0.2)
            cand_y = np.clip(cand_y, -0.2, 0.2)
            
            # 检查安全性（SAT 碰撞检测）
            if is_safe(cand_x, cand_y, tx, ty, ttheta):
                obstacles.append({{
                    'x': float(cand_x),
                    'y': float(cand_y),
                    'purpose': '象限边缘干扰'
                }})
                break
    
    return obstacles
```

### 示例 3：自适应密度布局策略

**设计思路**：根据起点到终点的距离，自适应调整障碍物的分布策略。路径长时分散，路径短时集中在侧方。

```python
def generate_obstacles(tblock_pose: np.ndarray, num_obstacles: int) -> list:
    import numpy as np
    
    tx, ty, ttheta = tblock_pose
    obstacles = []
    
    # 计算起点到终点的距离
    path_length = np.sqrt(tx**2 + ty**2)
    
    # 根据路径长度选择策略
    if path_length > 0.25:
        # 路径长：在路径中段放置
        positions = [0.4, 0.6]
        offset_range = 0.08
    else:
        # 路径短：在路径侧方放置，偏移更大
        positions = [0.5, 0.5]
        offset_range = 0.12
    
    # 路径方向
    if path_length > 0.01:
        path_angle = np.arctan2(-ty, -tx)
        perp_angle = path_angle + np.pi / 2
    else:
        perp_angle = 0
    
    for i in range(num_obstacles):
        for attempt in range(50):
            # 在路径上的位置
            t = positions[i % len(positions)] + np.random.uniform(-0.1, 0.1)
            point_x = tx * (1 - t)
            point_y = ty * (1 - t)
            
            # 向垂直方向偏移
            side = 1 if i % 2 == 0 else -1
            offset = offset_range + np.random.uniform(-0.02, 0.02)
            
            cand_x = point_x + side * offset * np.cos(perp_angle)
            cand_y = point_y + side * offset * np.sin(perp_angle)
            
            # 限制范围
            cand_x = np.clip(cand_x, -0.2, 0.2)
            cand_y = np.clip(cand_y, -0.2, 0.2)
            
            # 检查安全性（SAT 碰撞检测）
            if is_safe(cand_x, cand_y, tx, ty, ttheta):
                obstacles.append({{
                    'x': float(cand_x),
                    'y': float(cand_y),
                    'purpose': '自适应路径干扰'
                }})
                break
    
    return obstacles
```

---

## 重要说明

**以上三个示例仅供启发**，展示了不同的空间推理思路：
1. 示例 1：基于路径垂直方向的几何推理（侧方布局）
2. 示例 2：基于角度偏移的空间推理（象限分散）
3. 示例 3：基于路径长度的自适应推理（动态调整策略）

**关键观察**：
- 所有示例都只使用 `is_safe()` 检查碰撞，不手动计算"距离起点多远"
- 所有示例都强调"空间关系"（侧方、角度、路径方向）而非"绝对距离"
- 所有示例都避免将障碍物放在"起点正前方"

**你可以**：
- 创造完全不同的布局策略
- 组合多种思路
- 根据 `tblock_pose` 的不同特征（位置、朝向、距离）自适应调整
- 使用示例中没有的几何关系或空间推理

**你不应该**：
- 直接复制粘贴示例代码
- 只是修改示例中的几个数字
- 认为只有这三种策略是有效的
- 将障碍物密集地放在起点→终点的连线上

---

## 输出要求

**直接输出 Python 代码，不要添加任何解释文字或代码块标记（如 ```python）。**

代码必须：
1. 函数名必须是 `generate_obstacles`
2. 函数签名必须与规范完全一致
3. 可以使用 `import numpy as np`
4. 可以使用 `is_safe()` 辅助函数
5. 返回格式必须是 `[{{'x': float, 'y': float, 'purpose': str}}, ...]`

---

现在，请生成你的障碍物生成函数。记住：
- 遵守所有物理约束
- 不要完全阻挡直线路径（Phase 0 冷启动阶段）
- 确保多样性（使用随机扰动）
- 自由创造你认为合理的布局策略
"""
    
    return prompt

# ...existing code...

def build_phase0_topology_generator_prompt_compact(
    tblock_pose: np.ndarray,
    num_obstacles: int = 2
) -> str:
    tx, ty, ttheta = tblock_pose
    return f"""你是强化学习环境设计专家。为 Push-T 任务编写障碍物生成函数 generate_obstacles。

当前起点：x={tx:.3f}m, y={ty:.3f}m, θ={np.degrees(ttheta):.0f}°
终点固定：(0,0,-45°)
工作空间：x,y ∈ [-0.2,0.2]

硬约束（必须满足）：
1) 返回 list，元素为 dict：{{'x': float, 'y': float, 'purpose': str}}，长度尽量为 {num_obstacles}。
2) 每个 (x,y) 在 [-0.2,0.2]，且 is_safe(x,y, tx,ty,ttheta) 为 True。
3) 任意两障碍物中心距离 > 0.03m。
4) Phase0：不要完全阻挡起点→终点直线通路；轻微压迫即可，多数情况下可成功。

函数接口（必须严格遵守，否则会被判定失败）：
def generate_obstacles(tblock_pose: np.ndarray, num_obstacles: int) -> list:
    # return [{{'x': float, 'y': float, 'purpose': str}}, ...]

可用：
- np（可直接用；也允许 import numpy as np）
- is_safe(obs_x, obs_y, tblock_x, tblock_y, tblock_theta) -> bool
- 严禁使用 hash()（沙箱中未保证可用）

生成要求：
- 必须有随机性（np.random.uniform 等）
- 拒绝采样：每个障碍物最多尝试 50 次；失败可跳过，最终可少于 num_obstacles
- 避免把障碍物堆在起点正前方或直线路径核心区域

输出要求：
- 只输出 Python 代码
- 不要解释文字，不要 markdown 代码块
- 只定义 generate_obstacles（可包含少量内部辅助函数）
"""


def build_phase0_prompt_stage_a(
    tblock_pose: np.ndarray,
    num_obstacles: int = 2
) -> str:
    """
    
    Args:
        tblock_pose: T-block 起点位姿 [x, y, θ]
        num_obstacles: 障碍物数量
    
    Returns:
        完整的 prompt 字符串
    """
    tx, ty, ttheta = tblock_pose
    
    return f"""你是强化学习环境设计专家。请为 Push-T 写一个障碍物生成函数 generate_obstacles（Phase 0 冷启动：简单、稳定、轻微压迫）。

场景：
- 起点 tblock_pose: x={tx:.3f}, y={ty:.3f}, θ={np.degrees(ttheta):.0f}°
- 终点固定: (0, 0, -45°)

物理参数：
- 桌面物理边界: x,y ∈ [-0.25, 0.25] 米
- T-block 采样范围: x,y ∈ [-0.18, 0.18] 米
- T-block 尺寸: 横条 0.1m × 0.03m，竖条 0.03m × 0.07m，整体包络约 0.1m × 0.1m
- 障碍物尺寸: 0.02m × 0.02m × 0.02m（立方体，半边长 0.01m）
- 障碍物采样范围: x,y ∈ [-0.2, 0.2] 米

你必须输出"仅 Python 代码"（无解释、无 markdown 代码块），且只定义 generate_obstacles。

================= 硬约束（违反即失败） =================

A. 函数签名必须完全一致：
   def generate_obstacles(tblock_pose: np.ndarray, num_obstacles: int) -> list:

B. 返回值必须是 list，元素为 dict，字段必须齐全：
   [{{'x': float, 'y': float, 'purpose': str}}, ...]
   - 若 num_obstacles > 0：最终至少返回 1 个障碍物（绝不能返回空列表）

C. 每个障碍物必须满足：
    - x,y 在 [-0.2,0.2]
   - is_safe(x,y, tx,ty,ttheta) == True

D. 障碍物间距：
   - 任意两障碍物中心距离 > 0.03

E. Phase0 难度：
   - 不要把障碍物放在"起点→终点直线通路的核心区域"造成完全阻挡
   - 轻微压迫/存在感即可（优先放侧方、边缘、角落）

================= 运行环境限制（非常重要） =================

你生成的代码会在沙箱里执行：
- 只允许使用 np（可直接用；也允许 import numpy as np）
- 只允许使用 is_safe(...)
- 不要 import 任何除 numpy 之外的模块
- 不要使用 print / sorted / set / any / all / hash 等（未必在内置函数白名单里）

================= 强引导：实现骨架（建议按此写） =================

1) 读取 tx,ty,ttheta；定义 goal=(0,0)

2) 计算起点→终点向量 v = [-tx, -ty]，以及单位方向 dir、垂直方向 perp

3) 定义一个"软约束"避免阻挡直线路径（不要太苛刻以免采样失败）：
   - 计算候选点到线段 start->goal 的距离 d_seg
   - 若 d_seg < corridor（例如 0.035~0.05 之间取一个），则拒绝该候选点

4) 候选点生成用"混合分布"提高成功率与多样性（每次调用要随机）：
   - 方案1：路径侧方 corridor（在路径 35%~80% 的位置，沿 perp 偏移 0.08~0.13）
    - 方案2：四角/边缘（靠近 ±0.2 附近，加少量 jitter）
   
   每个障碍物做拒绝采样：最多 50 次；满足硬约束 + 与已选障碍物距离 >0.03 就接受

5) 若采样完 obstacles 仍为空（num_obstacles>0），做兜底：
   - 在边缘/角落区域继续随机采样，直到得到 1 个合法点（同样检查 is_safe 和范围）

purpose 字段要求：
- 简短即可，例如 "side_pressure" / "edge_presence" / "corner_hint"

最后再次强调：只输出 Python 代码，只定义 generate_obstacles。
"""


# ...existing code...

# ============================================================================
# StrategyExecutor: 安全执行 LLM 生成的拓扑生成器代码
# ============================================================================

class StrategyExecutor:
    """
    安全执行 LLM 生成的拓扑生成器代码
    
    功能：
    1. 在沙箱环境中执行 LLM 生成的 Python 代码
    2. 提供 is_safe() 辅助函数（使用精确的 SAT 碰撞检测）
    3. 验证生成的障碍物配置
    """
    
    def __init__(self, obstacle_size=0.01, target_pose=[0, 0, -np.pi/4]):
        """
        初始化执行器
        
        Args:
            obstacle_size: 障碍物半边长（默认 0.01m，即 0.02m × 0.02m）
            target_pose: 目标 T-block 位姿 [x, y, θ]
        """
        self.obstacle_size = obstacle_size
        self.target_pose = target_pose
        
        # 定义 is_safe 辅助函数（使用真实的 SAT 碰撞检测）
        def is_safe(obs_x: float, obs_y: float, 
                   tblock_x: float, tblock_y: float, tblock_theta: float) -> bool:
            """
            检查障碍物是否与 T-block 安全（使用精确的 SAT 碰撞检测）
            
            返回：
            - True: 安全，可以使用
            - False: 不安全，会碰撞
            
            注意：threshold = 0.04 * 2 = 0.08m（与原版 DIVO 一致）
            """
            obstacle_pos = np.array([obs_x, obs_y])
            start_pos = np.array([tblock_x, tblock_y])
            target_pos = np.array(self.target_pose[:2])
            
            # 检查与起点 T-block 的碰撞（使用 SAT）
            if analytic_obs_collision_check(
                Tblock_angle=tblock_theta,
                obs_center=obstacle_pos - start_pos,
                obs_size=self.obstacle_size * 2,
                threshold=0.04 * 2  # 0.08m，与原版 DIVO 一致
            ):
                return False
            
            # 检查与终点 T-block 的碰撞（使用 SAT）
            if analytic_obs_collision_check(
                Tblock_angle=self.target_pose[-1],
                obs_center=obstacle_pos - target_pos,
                obs_size=self.obstacle_size * 2,
                threshold=0.04 * 2  # 0.08m，与原版 DIVO 一致
            ):
                return False
            
            return True
        # 受控 import：仅允许 numpy（避免 LLM 代码 import 其它模块）
        def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "numpy":
                return np
            raise ImportError(f"Import blocked in sandbox: {name}")
        
        # 沙箱环境
        self.sandbox_globals = {
            'np': np,
            'numpy': np,
            'is_safe': is_safe,  # 注入真实的 SAT 碰撞检测
            '__builtins__': {
                '__import__': _safe_import,   # <-- 新增这一行
                'range': range,
                'len': len,
                'abs': abs,
                'hash': hash,
                'min': min,
                'max': max,
                'float': float,
                'int': int,
                'list': list,
                'dict': dict,
                'enumerate': enumerate,
                'zip': zip,
                'sum': sum,
                'round': round,
            }
        }
        
        self.generate_obstacles = None
        self._output_format_error_count = 0

    
    def load_topology_generator(self, code: str) -> bool:
        """
        加载拓扑生成器代码
        
        Args:
            code: LLM 生成的 Python 代码
            
        Returns:
            True: 加载成功
            False: 加载失败
        """
        try:
            # 在沙箱中执行代码
            exec(code, self.sandbox_globals)
            
            # 检查函数是否存在
            if 'generate_obstacles' not in self.sandbox_globals:
                raise ValueError("代码中没有 generate_obstacles 函数")
            
            import inspect

            fn = self.sandbox_globals['generate_obstacles']
            if not callable(fn):
                raise ValueError("generate_obstacles 不是可调用对象")

            # 校验函数签名：必须能以 (tblock_pose, num_obstacles) 方式调用
            sig = inspect.signature(fn)
            params = list(sig.parameters.values())
            has_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params)
            has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)

            if not (has_varargs or has_varkw):
                # 常规函数：至少需要 2 个位置/关键字参数
                if len(params) < 2:
                    raise ValueError(
                        f"generate_obstacles 签名不正确：期望至少 2 个参数 (tblock_pose, num_obstacles)，实际为 {sig}"
                    )

            self.generate_obstacles = fn
            self._output_format_error_count = 0
            print("✓ 拓扑生成器加载成功")
            return True
            
        except Exception as e:
            print(f"❌ 代码加载失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def generate(self, tblock_pose: np.ndarray, num_obstacles: int, timeout_sec: int = 5) -> list:
        """
        使用拓扑生成器生成障碍物（带超时保护）
        
        Args:
            tblock_pose: T-block 位姿 [x, y, θ]
            num_obstacles: 障碍物数量
            timeout_sec: 超时秒数，防止 LLM 生成的代码有无限循环
            
        Returns:
            障碍物列表 [{'x': ..., 'y': ..., 'purpose': ...}, ...]
        """
        if self.generate_obstacles is None:
            print("❌ 拓扑生成器未加载")
            return []
        
        def _timeout_handler(signum, frame):
            raise TimeoutError("generate_obstacles 调用超时")
        
        try:
            # 设置超时保护
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(timeout_sec)
            
            try:
                obstacles = self.generate_obstacles(tblock_pose, num_obstacles)
            finally:
                # 无论成功失败，都取消 alarm 并恢复旧 handler
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

            if obstacles is None:
                self._output_format_error_count += 1
                if self._output_format_error_count <= 3 or self._output_format_error_count % 100 == 0:
                    print(
                        "⚠ 输出格式警告：generate_obstacles 返回 None，"
                        "已按空列表处理（请确保函数始终 return list）"
                    )
                return []
            
            # 验证输出格式
            if not isinstance(obstacles, list):
                self._output_format_error_count += 1
                if self._output_format_error_count <= 3 or self._output_format_error_count % 100 == 0:
                    print(f"❌ 输出格式错误：期望 list，得到 {type(obstacles)}")
                return []
            
            # 验证每个障碍物的格式
            valid_obstacles = []
            for i, obs in enumerate(obstacles):
                # 兼容非 dict 格式：list/tuple/ndarray [x, y] → dict
                if hasattr(obs, '__len__') and not isinstance(obs, dict):
                    try:
                        obs = list(obs)  # ndarray → list
                        if len(obs) >= 2:
                            obs = {'x': obs[0], 'y': obs[1], 'purpose': obs[2] if len(obs) > 2 else ''}
                        else:
                            print(f"⚠ 障碍物 {i} 序列长度不足: {obs}")
                            continue
                    except Exception:
                        print(f"⚠ 障碍物 {i} 无法转换: {type(obs)}")
                        continue

                if not isinstance(obs, dict):
                    print(f"⚠ 障碍物 {i} 格式错误：期望 dict，得到 {type(obs)}")
                    continue

                if 'x' not in obs or 'y' not in obs:
                    print(f"⚠ 障碍物 {i} 缺少 x 或 y 字段")
                    continue

                # 转换为 float
                try:
                    obs['x'] = float(obs['x'])
                    obs['y'] = float(obs['y'])
                    if 'purpose' not in obs:
                        obs['purpose'] = ''
                    valid_obstacles.append(obs)
                except (ValueError, TypeError) as e:
                    print(f"⚠ 障碍物 {i} 坐标转换失败: {e}")
                    continue

            return valid_obstacles
            
        except TimeoutError:
            print(f"❌ generate_obstacles 超时 ({timeout_sec}s)，LLM 生成的代码可能有无限循环")
            return []
        except Exception as e:
            print(f"❌ 生成失败: {e}")
            import traceback
            traceback.print_exc()
            return []

    
    def validate_obstacles(self, obstacles: list, tblock_pose: np.ndarray) -> tuple:
        """
        验证生成的障碍物配置
        
        Args:
            obstacles: 障碍物列表
            tblock_pose: T-block 位姿
            
        Returns:
            (is_valid, reason): 是否有效，原因
        """
        if len(obstacles) == 0:
            return False, "没有生成任何障碍物"
        
        # 检查每个障碍物
        for i, obs in enumerate(obstacles):
            x, y = obs['x'], obs['y']
            
            # 检查坐标范围
            if not (-0.2 <= x <= 0.2 and -0.2 <= y <= 0.2):
                return False, f"障碍物 {i} 超出范围: ({x:.3f}, {y:.3f})"
            
            # 检查与 T-block 的碰撞（使用 SAT）
            is_safe_func = self.sandbox_globals['is_safe']
            if not is_safe_func(x, y, tblock_pose[0], tblock_pose[1], tblock_pose[2]):
                return False, f"障碍物 {i} 与 T-block 碰撞: ({x:.3f}, {y:.3f})"
        
        # 检查障碍物之间的距离
        for i in range(len(obstacles)):
            for j in range(i + 1, len(obstacles)):
                dist = np.sqrt(
                    (obstacles[i]['x'] - obstacles[j]['x'])**2 +
                    (obstacles[i]['y'] - obstacles[j]['y'])**2
                )
                if dist < 0.03:
                    return False, f"障碍物 {i} 和 {j} 距离过近: {dist:.3f}m < 0.03m"
        
        return True, "验证通过"

    def sanity_check(self, num_tests: int = 5, num_obstacles: int = 2) -> bool:
        """
        快速检查当前生成器是否能正常工作。

        用几个随机 tblock_pose 调用 generate，检查是否能在超时内返回有效结果。

        Args:
            num_tests: 测试次数
            num_obstacles: 每次生成的障碍物数量

        Returns:
            True: 全部通过
            False: 有任何一次失败（超时/返回空/返回 None）
        """
        for i in range(num_tests):
            # 随机生成一个 tblock_pose
            x = np.random.uniform(-0.18, 0.18)
            y = np.random.uniform(-0.18, 0.18)
            while abs(x) < 0.1 and abs(y) < 0.1:
                x = np.random.uniform(-0.18, 0.18)
                y = np.random.uniform(-0.18, 0.18)
            theta = np.random.uniform(0, 2 * np.pi)
            tblock_pose = np.array([x, y, theta])

            result = self.generate(tblock_pose, num_obstacles)
            if not result or len(result) == 0:
                print(f"[SanityCheck] 失败: 第 {i+1}/{num_tests} 次测试返回空结果")
                return False
        print(f"[SanityCheck] 通过: {num_tests} 次测试全部成功")
        return True



# ============================================================================
# LLMTopologyGenerator: LLM 拓扑生成器
# ============================================================================

class LLMTopologyGenerator:
    """
    LLM 拓扑生成器
    
    功能：
    1. 调用 LLM 生成拓扑生成器代码（Python 函数）
    2. 提取和验证生成的代码
    
    与 V3 的区别：
    - V3: 生成具体坐标
    - V4: 生成 Python 函数（拓扑生成器）
    """
    
    def __init__(self,
                 api_type: str = "deepseek",
                 api_key: str = None,
                 model: str = None,
                 base_url: str = None,
                 temperature: float = 0.7,
                 verbose: bool = True):
        """
        初始化 LLM 拓扑生成器
        
        Args:
            api_type: API 类型 ("deepseek" 或 "openai")
            api_key: API 密钥
            model: 模型名称
            base_url: API 基础 URL
            temperature: 温度参数（控制随机性）
            verbose: 是否打印详细信息
        """
        self.api_type = api_type
        self.temperature = temperature
        self.verbose = verbose
        self.model = model or ("deepseek-chat" if api_type == "deepseek" else "gpt-4")

        if openai is None:
            raise ImportError(
                "Missing dependency 'openai'. Install it with `pip install openai`."
            ) from _OPENAI_IMPORT_ERROR
        if not hasattr(openai, "OpenAI"):
            raise ImportError(
                "The installed 'openai' package is too old. Please install openai>=1.0.0."
            )
        
        # 初始化客户端
        if api_type == "deepseek":
            self.client = openai.OpenAI(
                api_key=api_key or os.getenv("DEEPSEEK_API_KEY"),
                base_url=base_url or "https://api.deepseek.com"
            )
        elif api_type == "openai":
            self.client = openai.OpenAI(
                api_key=api_key or os.getenv("OPENAI_API_KEY"),
                base_url=base_url
            )
        else:
            raise ValueError(f"不支持的 api_type: {api_type}")
        
        if self.verbose:
            print(f"✓ LLMTopologyGenerator 初始化成功 (model: {self.model})")

    
    def generate_topology_generator(self, 
                                    tblock_pose: np.ndarray, 
                                    num_obstacles: int = 2) -> Optional[str]:
        """
        生成拓扑生成器代码
        
        Args:
            tblock_pose: T-block 起点位姿 [x, y, θ]
            num_obstacles: 障碍物数量
        
        Returns:
            LLM 生成的 Python 代码字符串，如果失败则返回 None
        """
        # 构建 prompt
        prompt = build_phase0_topology_generator_prompt(tblock_pose, num_obstacles)
        
        if self.verbose:
            print(f"\n=== Phase 0: 生成拓扑生成器 ===")
            print(f"T-block 位置: ({tblock_pose[0]:.3f}, {tblock_pose[1]:.3f}, {np.degrees(tblock_pose[2]):.0f}°)")
            print(f"障碍物数量: {num_obstacles}")
        
        # 调用 LLM
        response = self._call_llm(prompt)
        
        if response is None:
            if self.verbose:
                print("❌ LLM 调用失败")
            return None
        
        # 提取代码
        code = self._extract_code(response)
        
        if code is None:
            if self.verbose:
                print("❌ 代码提取失败")
            return None
        
        if self.verbose:
            print("✓ 拓扑生成器代码生成成功")
            print(f"\n生成的代码（前 500 字符）:\n{code[:500]}...")
        
        return code
    
    def evolve(self, acgs_prompt: str) -> Optional[str]:
        """
        ACGS 闭环: 根据批次原始数据生成新的拓扑生成器代码

        Args:
            acgs_prompt: 由 td3_curriculum_workspace._build_acgs_prompt() 构建的完整 prompt，
                         包含四样数据：批次统计 + 失败回放 + 当前策略 + Q值

        Returns:
            LLM 生成的新 Python 代码字符串，如果失败则返回 None
        """
        if self.verbose:
            print(f"\n=== ACGS Evolve: 请求 LLM 分析并生成新环境 ===")

        # 添加 system prompt 引导 LLM 的双重角色
        system_prompt = (
            "你是一个Push-T强化学习环境生成代码专家。\n"
            "在Push-T任务中，机器人rod推动T-block从随机起点到目标位姿(0, 0, -45°)，桌面上存在障碍物。\n"
            "桌面范围为[-0.25, 0.25]，障碍物坐标必须严格落在[-0.2, 0.2]。\n\n"
            "你的任务是根据user_prompt中提供的当前训练信息和当前生成函数，输出新的 "
            "generate_obstacles(tblock_pose, num_obstacles)Python 函数。\n\n"
            "硬性要求：\n" 
            "1. 只输出Python代码，不要输出任何解释文字。\n"
            "2. 必须输出完整可运行的generate_obstacles(tblock_pose, num_obstacles) 函数，不要只输出局部片段。\n"
            "3. 每个障碍物必须返回dict:{'x': float, 'y': float, 'purpose': str}。\n"
            "4. 所有障碍物坐标都必须满足 x, y ∈ [-0.2, 0.2]。\n"
            "5. 必须遵守user prompt中给出的当前约束、目标和统计信号。\n"
            "6. 生成的障碍物布局应保持可解，并使任务难度与当前策略能力相匹配。\n"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": acgs_prompt}
                ],
                temperature=self.temperature,
                max_tokens=2000
            )
            raw_response = response.choices[0].message.content
            code = self._extract_code(raw_response)

            if code and self.verbose:
                print(f"✓ Evolve 生成代码 ({len(code)} chars)")
            elif self.verbose:
                print(f"❌ Evolve 代码提取失败")

            return code

        except Exception as e:
            print(f"❌ Evolve LLM 调用失败: {e}")
            return None

    def _call_llm(self, prompt: str) -> Optional[str]:
        """
        调用 LLM
        
        Args:
            prompt: 完整的 prompt
        
        Returns:
            LLM 的响应，如果失败则返回 None
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个强化学习环境设计专家。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=2000
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"❌ LLM 调用失败: {e}")
            return None

    
    def _extract_code(self, response: str) -> Optional[str]:
        """
        从 LLM 响应中提取 Python 代码
        
        Args:
            response: LLM 的原始响应
        
        Returns:
            提取的 Python 代码，如果提取失败则返回 None
        """
        if response is None:
            return None
        
        # 方法 1：查找 ```python 代码块
        if '```python' in response:
            start = response.find('```python') + 9
            end = response.find('```', start)
            if end != -1:
                return response[start:end].strip()
        
        # 方法 2：查找 ``` 代码块
        if '```' in response:
            start = response.find('```') + 3
            end = response.find('```', start)
            if end != -1:
                code = response[start:end].strip()
                # 跳过可能的语言标识符
                if code.startswith('python'):
                    code = code[6:].strip()
                return code
        
        # 方法 3：查找 def generate_obstacles
        if 'def generate_obstacles' in response:
            start = response.find('def generate_obstacles')
            # 找到函数结束（简单处理：取到响应结束）
            return response[start:].strip()
        
        # 方法 4：假设整个响应就是代码
        if 'import numpy' in response or 'import np' in response:
            return response.strip()
        
        return None
