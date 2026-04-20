"""
"""
import numpy as np
import json
import os
import logging
import openai
from typing import List, Dict, Optional, Tuple
from DIVO.utils.util import analytic_obs_collision_check

# 屏蔽 httpx 的 HTTP 请求日志
logging.getLogger("httpx").setLevel(logging.WARNING)


SYSTEM_PROMPT = """你是一名专门设计机器人训练环境的强化学习工程师。
你的目标是为 Push-T 任务设计多样化的障碍物布局，用于训练机器人的操作策略。

每次生成都是一次独特的实验。探索整个可能性空间，而不是寻找"最优解"。

输出要求：直接输出 JSON 对象，不要添加额外文字或代码块标记。"""



TASK_SPECIFICATION = """
## 问题定义：

Push-T 是一个机器人操作任务:控制一个圆柱形的“推杆(Rod)”,通过推动的方式,将一个T字形的方块从桌面的随机位置,精确地推送到指定的中心目标位置,并对齐角度。


这个任务测试机器人的：
- 路径规划能力（绕过障碍物）
- 精确操作能力（在狭窄空间中操作）
- 旋转控制能力（调整 T-block 朝向）

## 环境规格

### 工作空间
- 桌面大小: 0.5m × 0.5m
- 坐标系原点: 桌面中心 (0, 0)
- 坐标范围: x ∈ [-0.25, 0.25], y ∈ [-0.25, 0.25]（理论最大范围）
- 障碍物有效放置范围: x, y ∈ [-0.2, 0.2]（留出边缘缓冲）

### T-block（被推物体）
- 形状: T 形，由两个矩形组成
  - 横杠: 0.10m × 0.03m（中心在 T 形中心）
  - 竖杠: 0.03m × 0.07m（中心向下偏移 0.05m）
- 整体包络: 宽 0.10m × 高 0.12m（从横杠顶部到竖杠底部）
- 位姿表示: [x, y, θ]，θ 为绕 z 轴旋转角度（弧度）
- 质量: 0.2 kg（两个部分各 0.1 kg）
- 颜色: 蓝色

### 起点位置（每次随机）
- 采样范围: x, y ∈ [-0.18, 0.18]
- 排除中心区域: |x| < 0.1 且 |y| < 0.1（避免起点太靠近终点）
- 朝向: θ ∈ [0, 2π]（随机）
- 距离约束: 起点到终点的距离必须足够远（reward <= -3.0）
- 实际起点分布: 主要在桌面的四个象限，不会出现在中心区域

### 末端执行器（推杆）
- 形状: 圆柱形
- 直径: 0.02m
- 长度: 0.1m
- 可以从任意方向推动 T-block

### 障碍物
- 形状: 正方形方块
- 尺寸: 0.02m × 0.02m（边长）
- 质量: 100 kg（固定在桌面上，不可移动）
- 颜色: 红色（用于可视化）
- 放置范围: x, y ∈ [-0.2, 0.2]

### 目标（终点）
- 位置: (0, 0)，即桌面中心（固定不变）
- 朝向: -45°（-π/4 弧度，固定不变）
- 可视化: 绿色半透明 T 形轮廓


## 设计原则

1. **可学习性**: 障碍物应该创造有意义的挑战，但不能使任务不可能完成
2. **多样性**: 不同的障碍物配置应该测试不同的技能
3. **渐进性**: 难度应该可以通过参数控制
4. **物理合理性**: 必须留出足够的空间让 T-block 通过
"""


IN_CONTEXT_EXAMPLE = """
## 格式参考与生成警告（重要）

以下【仅为 JSON 格式的结构参考】。里面的坐标和推理是随机示例，不代表最优解或推荐策略。

**输入**：
- T-block 初始位置: x=0.150m, y=0.150m, θ=135°
- 目标位置: x=0.000m, y=0.000m, θ=-45°
- 障碍物数量: 2
- 难度: hard

**输出格式参考**：
{
  "reasoning": "基于几何关系设计障碍物布局。这只是格式示例，实际设计应该完全不同。",
  "obstacles": [
    {"x": 0.073, "y": -0.041, "purpose": "示例位置 1"},
    {"x": -0.127, "y": 0.089, "purpose": "示例位置 2"}
  ]
}

**输出格式要求**：
- 直接输出 JSON 对象，以 { 开始，以 } 结束
- 绝对不要在 JSON 前后添加任何解释文字或代码块标记（严禁使用 ```json）
- JSON 可以有换行和缩进（如上所示）

### ⚠️ 生成前必读：认知重置

**刚才的示例仅用于展示 JSON 格式，请立即忘掉示例中的具体坐标和布局！**

在实际生成中，你必须：

❌ 不要模仿示例：
- 不要使用示例中的坐标或类似坐标
- 不要模仿示例的布局模式（如"一个在右上，一个在左上"）
- 不要认为存在固定的"正确套路"

✅ 自由探索全空间：
- 探索 [-0.20, 0.20] 的每一个角落
- 尝试任何距离：从极近（0.04m）到较远（0.18m）
- 尝试任何拓扑：混沌散落、规整阵列、密集聚集、稀疏分布

✅ 挑战物理极限：
- 利用 T-block 不对称性（扁平面约 0.03m，长尾巴约 0.086m）
- 起点朝向随机，可以"赌"极近距离（0.04m~0.08m）进行盲狙压迫
- 终点朝向固定 -45°，可以在特定凹角方向精确打造入库卡槽

✅ 确保每次都不同：
- 每次生成都应该是独特的、不可预测的
- 从极致混沌到极致秩序，任何拓扑都有价值
- 只要满足物理碰撞检测（不穿模），就是有效的训练数据

记住：你的目标是最大化空间拓扑多样性，而不是找到"最优解"！
"""


def build_generation_prompt(
    tblock_pose: List[float],
    num_obstacles: int,
    difficulty: str
) -> str:
    """
    构建初始生成 prompt（无场景类型版本）
    
    基于几何分析和难度级别，让 LLM 自主设计障碍物布局
    """
    
    # 计算几何信息
    start_x, start_y, start_theta = tblock_pose
    target_x, target_y, target_theta = 0, 0, -np.pi/4
    
    distance = np.sqrt(start_x**2 + start_y**2)
    angle_to_target = np.arctan2(-start_y, -start_x)
    # 路径垂直方向（用于空间参考，不暴露中点坐标以避免锚定效应）
    perp_angle = angle_to_target + np.pi / 2
    perp_dx, perp_dy = np.cos(perp_angle), np.sin(perp_angle)
    rotation_needed = target_theta - start_theta
    # 归一化到 [-π, π]
    rotation_needed = (rotation_needed + np.pi) % (2 * np.pi) - np.pi
    
    # 判断 T-block 所在象限
    if start_x >= 0 and start_y >= 0:
        quadrant = "第一象限（右上）"
    elif start_x < 0 and start_y >= 0:
        quadrant = "第二象限（左上）"
    elif start_x < 0 and start_y < 0:
        quadrant = "第三象限（左下）"
    else:
        quadrant = "第四象限（右下）"
    
    # 分析 T-block 朝向
    degrees_theta = np.degrees(start_theta)
    if -45 <= degrees_theta <= 45 or 135 <= abs(degrees_theta) <= 180:
        orientation_desc = "横杠大致水平"
    else:
        orientation_desc = "横杠大致垂直"
    
    # 难度说明（意图驱动 + 物理红线）
    difficulty_descriptions = {
        "easy": """
**难度: 简单 (easy) - 基础空间感知**

【考核技能：反应式避障与鲁棒性】
- 基础的**反应式避障 (Reactive Avoidance)**能力。
- 在开阔空间中对轻微几何干扰的**航向修正与保持 (Heading Maintenance)**。
- 验证机器人的策略没有过度拟合"绝对空旷无物"的理想环境，能够在有视觉/空间压迫感时保持稳定。

【拓扑与布局特征】
- 障碍物应该像"路肩"或"警示牌"一样，靠近主要移动意图的边缘，制造空间压迫感。
- 绝不能彻底切断直线直达的可能性，不需要机器人进行大角度绕行。

【物理红线（严格遵守）】
- 必须保留一条宽度 > 0.18m 的直达宽敞通道。
- 🚫 避坑指南：千万不要为了"简单"就把障碍物扔到完全无关的角落（如背对目标的象限）。
  如果障碍物距离 T-block 的潜在运动轨迹超过 0.15m，将被评估器判定为"无效生成"而扣分！

【量化参考（仅供校验）】
- 预期绕行比: 1.0~1.2""",
        
        "medium": """
**难度: 中等 (medium) - 全局规划与拓扑推理**

【考核技能：克服局部最优与大尺度规划】
- **克服局部最优 (Overcoming Local Optima)**：迫使机器人放弃贪婪的"直线最短"策略，学会为了最终目标而暂时偏离直接航向。
- **拓扑路径选择 (Topological Routing)**：识别并规划 C 型、S 型或多段折线等大尺度非凸绕行轨迹。
- **方向切换与子目标衔接 (Direction Switching)**：在绕行路径的转折点平滑切换推动方向，而不是在弯道处卡住或反复碰撞障碍物。

【拓扑与布局特征】
- 必须从拓扑上切断起点到终点的直接连线。
- 鼓励使用不对称的散落布局，将工作空间分割成至少两条可选但曲折的路径。
- 不要只是生硬地堵在正中间，尝试利用障碍物引导机器人偏离航线再回归。

【物理红线（严格遵守）】
- 绕行通道的最小宽度应控制在 0.15m - 0.18m 之间。
- 保证通道足够宽敞，使得 T-block 在平移通过时，不需要进行高难度的极限角度微调。

【量化参考（仅供校验）】
- 预期绕行比: 1.3~1.6""",
        
        "hard": """
**难度: 困难 (hard) - 精确姿态控制与路径强制绕行**

【考核技能：少量障碍物下的高效对抗设计】
- **精确间隙通过 (Precision Gap Traversal)**：用 2-3 个障碍物构造一个关键间隙，T-block 必须以特定角度才能通过。
- **不可逆推操作意识 (Irreversible Push Awareness)**：推操作无法"向后拉"，T-block 一旦被推入错误位置就难以修正。迫使机器人提前规划进入角度，而不是到了瓶颈才临时调整。
- **绕行+姿态耦合 (Detour-Pose Coupling)**：障碍物既阻挡直线路径迫使绕行，又通过间隙约束迫使 T-block 提前旋转到正确角度。

【拓扑生成法则（3 个 2cm 障碍物的有效战术）】
- 🚫 **核心禁忌：绝对不要把障碍物放在终点 (0,0) 或起点附近 0.12m 以内！** 战场在路径中段。
- 🎯 **核心目标**：用最少的障碍物制造最大的路径约束。3 个 2cm 障碍物虽小，但摆放得当可以实现：
  1. **阻断+偏转**：1-2 个障碍物挡住起点到终点的直线路径，迫使 T-block 必须绕行。
  2. **精确间隙**：2 个障碍物形成一个非对称间隙（一侧宽、一侧窄），T-block 必须以特定角度侧身通过。
  3. **路径分段**：障碍物将路径分成"绕行段"和"精确通过段"，机器人必须在不同阶段切换策略。
- 💡 **自由发挥**：以上是 3 个 2cm 障碍物能实现的基本功能。请根据当前几何关系自行推理最佳坐标组合，不要套用固定模板。

【物理红线（致命约束，必须严格遵守）】
- 瓶颈间隙（两障碍物之间）的最窄宽度: **0.14m - 0.16m**。
- ⚠️ 物理常识：T-block 宽 0.1m，旋转时对角线包络约 0.14m。小于 0.14m 的间隙是死局！
- ⚠️ **绝对禁区**：所有障碍物距起点和终点 T-block 中心必须 > **0.12m**。不要试图把障碍物塞进 T-block 的凹口或贴着侧面——你的坐标精度做不到，物理引擎会直接判碰撞！

【量化参考（仅供校验）】
- 预期绕行比: 1.5~2.0"""
    }
    
    # 几何分析提示
    geometric_analysis = f"""
### 几何分析

**空间信息：**
- T-block 起点: ({start_x:.3f}, {start_y:.3f})，位于{quadrant}
- 目标: (0, 0)
- 起点到目标距离: {distance:.3f}m
- 路径方向角: {np.degrees(angle_to_target):.1f}°
- 路径垂直方向: {np.degrees(perp_angle):.1f}°（沿此方向放置障碍物可形成侧面约束）
- 有效放置区域: x ∈ [-0.20, 0.20], y ∈ [-0.20, 0.20]

**朝向信息：**
- T-block 当前朝向: {np.degrees(start_theta):.1f}° ({orientation_desc})
- 目标朝向: -45°
- 需要旋转角度: {np.degrees(rotation_needed):.1f}°

**⚠️ 特别警告：终点附近是危险区！**
- 终点 (0, 0) 附近 0.12m 范围内是高风险区域
- 不要尝试在终点附近放置障碍物！
- 常见错误位置（禁止使用）：(-0.05, 0.05), (0.05, 0.05), (-0.08, 0.08), (0.08, -0.08) 等
- 这些位置距离终点太近，会被系统拒绝

**安全距离指南（重要！）**：
- 系统碰撞检测阈值：0.04m（如果障碍物距离 T-block 中心 < 0.04m 会被拒绝）
- 推荐设计范围：0.08m-0.15m
  * 0.08m-0.10m：较近，更具挑战性，但可能被拒绝
  * 0.10m-0.12m：适中，推荐使用
  * 0.12m-0.15m：较远，绝对安全
  * > 0.15m：太远，失去阻挡作用

**设计考虑：**
1. 障碍物应该放在路径中段（起点和终点之间的区域），避免靠近起点或终点
2. 考虑旋转需求：需要旋转 {np.degrees(rotation_needed):.1f}° 才能对齐目标
3. 评估空间约束：T-block 宽度 0.1m，需要留出足够的通过空间
4. **⚠️ 障碍物间距检查**：多个障碍物中心之间的距离必须 > 0.03m
5. 鼓励自由探索整个工作空间，但要避开起点和终点附近的危险区
"""
    
    prompt = f"""{TASK_SPECIFICATION}

{IN_CONTEXT_EXAMPLE}

---

## 当前任务

### 输入参数
- **T-block 初始位置**: x={start_x:.3f}m, y={start_y:.3f}m, θ={np.degrees(start_theta):.1f}°
- **目标位置**: x=0, y=0, θ=-45°
- **障碍物数量**: {num_obstacles}
- **难度**: {difficulty}

### ⚠️ 多样性要求（重要）
每次生成的障碍物位置必须与之前不同。请自由探索整个工作空间的不同区域（路径中段、路径两侧、不同象限等），不要重复之前的布局。

{geometric_analysis}

## 设计目标（质量评估维度参考）

生成的障碍物配置将根据以下四个维度进行评估。请在设计中考虑这些维度，但**精确的数值验证将由系统自动完成**：

### 1. 可解性 (Solvability) - 设计原则
**设计目标**：确保配置是可解的
- 避免在起点周围放置过多障碍物，确保 T-block 有移动空间
- 确保障碍物布局不会完全阻挡从起点到终点的路径
- 避免障碍物与起点/终点太近（建议保持合理距离）

**注意**：系统会自动检查起点是否被困、是否存在可行路径，不满足的配置会被拒绝。

### 2. 难度 (Difficulty) - 根据难度级别设计
**设计目标**：根据难度级别控制挑战程度

**路径复杂度**：
- 简单：障碍物远离主要路径，不需要明显绕行
- 中等：障碍物阻挡直线路径，需要一定绕行
- 困难：障碍物形成复杂约束，需要大幅绕行

**空间约束**：
- 简单：留出宽敞的通道（明显大于 T-block 宽度 0.1m）
- 中等：形成需要精确控制的通道（略大于 T-block 宽度）
- 困难：形成狭窄通道（接近 T-block 宽度）

**旋转难度**：
- 考虑 T-block 需要旋转的角度和路径上的旋转空间
- 简单：旋转空间充足
- 中等：旋转空间适中
- 困难：旋转空间受限

**障碍物密度**：
- 根据难度级别调整障碍物数量（简单少一些，困难多一些）

**注意**：系统会自动计算路径复杂度、空间约束、旋转难度等数值，用于质量评分。

### 3. 有效性 (Effectiveness) - 设计原则
**设计目标**：确保障碍物形成有意义的约束
- 障碍物应该靠近起点到终点的路径，形成实际影响
- **⚠️ 障碍物间距要求（重要）**：任意两个障碍物中心之间的距离必须 > 0.03m（30mm）。障碍物尺寸为 0.02m × 0.02m，如果距离 < 0.03m 会导致重叠或过于接近，这在物理上不合理且会被系统判定为无效配置。
- 不能过于简单（应该阻挡直线路径，迫使规划）
- 根据整体难度，形成相应强度的约束

**注意**：系统会自动检查障碍物是否太远、是否重叠（距离 < 0.03m）、是否过于简单，不符合要求的配置会被扣分。

### 4. 多样性 (Diversity) - 重要要求
- **⚠️ 每次生成必须不同**：即使是相同的输入参数，每次生成的障碍物位置应该不同
- 系统会自动评估与历史配置的差异
- **鼓励尝试不同的布局策略**：
  - 障碍物可以分布在路径的不同位置（起点附近、中点附近、终点附近）
  - 可以尝试不同的聚集模式（分散、聚集、线性排列等）
  - 可以尝试不同的空间分布（对称、非对称、随机分布等）
- **不要重复之前的布局**：每次生成时，尝试新的、不同的障碍物位置组合

---

{difficulty_descriptions.get(difficulty, difficulty_descriptions["medium"])}

## 约束条件（必须严格遵守）

1. **坐标范围**: x, y 必须在 [-0.2, 0.2] 之间
2. **⚠️ 避开起点和终点（T 形碰撞区，非圆形！）**: 系统使用 SAT (分离轴定理) 精确碰撞检测，T-block 的碰撞排斥区是 **T 形**，不是简单的圆形！
   - **stem（竖杠）方向 0.12m 内危险**，crossbar 端点方向 0.07m，缺口方向 0.05m
   - **简单安全规则：距离 > 0.12m → 任何方向都安全**
   - 详细的 stem 方向和危险区域见下方「碰撞排斥区」章节
3. **⚠️ 障碍物间距（关键约束）**: 任意两个障碍物中心之间的距离必须严格 > 0.03m（30mm）。
4. **可行性**: 必须存在至少一条宽度 ≥ 0.11m 的可行路径

## 输出格式

⚠️ 重要：直接输出纯 JSON，不要添加代码块标记（```json）、注释或额外文字。

请严格按以下 JSON 格式输出，以左花括号开始，以右花括号结束：

```json
{{
  "reasoning": "详细说明设计思路，包括如何满足可解性、难度、有效性要求，基于什么几何分析",
  "analysis": {{
    "solvability_check": "如何确保起点不被困、存在可行路径",
    "difficulty_breakdown": {{
      "path_complexity": "预期的路径复杂度（绕行比例，如 1.2 表示需要绕行20%）",
      "space_constraint": "预期的空间约束（最小间隙，单位：米）",
      "rotation_difficulty": "预期的旋转难度（考虑旋转角度和旋转空间）",
      "density": "障碍物数量"
    }},
    "effectiveness_check": "如何确保障碍物靠近路径、不重叠（所有障碍物对之间距离 > 0.03m）、形成有效约束",
    "estimated_quality": "预估的质量分数（0-1，考虑可解性、难度、有效性）"
  }},
  "obstacles": [
    {{"x": 0.0, "y": 0.0, "purpose": "这个障碍物的作用"}}
  ]
}}
```

注意：
- 不要在 JSON 中使用注释（// 或 #）
- 不要在最后一个元素后添加逗号
- 确保所有字符串用双引号
- 输出后不要添加任何解释文字
"""
    return prompt


def build_feedback_evolution_prompt(
    tblock_pose: List[float],
    previous_config: List[Dict],
    quality_score: float,
    detailed_scores: Dict,
    feedback: str,
    num_obstacles: int,
    difficulty: str = "medium"
) -> str:
    """
    构建基于质量评估反馈的进化 prompt
    
    Args:
        tblock_pose: T-block 初始位姿
        previous_config: 之前的障碍物配置
        quality_score: 质量评分 (0-1)
        detailed_scores: 详细评分字典
        feedback: 评估反馈文本
        num_obstacles: 障碍物数量
        difficulty: 难度级别
    """
    start_x, start_y, start_theta = tblock_pose
    
    # 格式化之前的配置
    prev_config_str = json.dumps(previous_config, indent=2, ensure_ascii=False)
    
    # 提取问题
    issues = detailed_scores.get('issues', [])
    issues_text = "\n".join([f"- {issue}" for issue in issues]) if issues else "无重大问题"
    
    # 提取详细分数
    solvability = detailed_scores.get('solvability', 0)
    difficulty_score = detailed_scores.get('difficulty', 0)
    diversity = detailed_scores.get('diversity', 0)
    effectiveness = detailed_scores.get('effectiveness', 0)
    
    # 检查之前的配置是否阻挡了路径
    from DIVO.env.pusht.obstacle_quality_evaluator import ObstacleQualityEvaluator
    temp_evaluator = ObstacleQualityEvaluator()
    target_pose = [0, 0, -np.pi/4]
    previous_path_blocked = temp_evaluator._is_path_blocked(
        np.array(tblock_pose[:2]),
        np.array(target_pose[:2]),
        previous_config
    )
    
    # 分析当前配置的优缺点
    strengths = []  # 需要保持的好特性
    weaknesses = []  # 需要改进的问题
    
    # 分析优点（需要保持的特性）
    if previous_path_blocked:
        strengths.append(f"✓ **路径阻挡**：成功阻挡了直线路径（有效性核心，必须保持）")
    if solvability > 0.5:
        strengths.append(f"✓ **可解性**：配置是可解的（必须保持）")
    if difficulty_score >= 0.5:
        strengths.append(f"✓ **难度适中**：难度分数 {difficulty_score:.3f}（保持或微调）")
    if effectiveness >= 0.7:
        strengths.append(f"✓ **有效性良好**：有效性分数 {effectiveness:.3f}（保持）")
    
    # 分析缺点（需要改进的问题）
    if not previous_path_blocked:
        weaknesses.append(f"✗ **缺少路径阻挡**：没有阻挡直线路径，导致过于简单（必须修复）")
    if issues:
        for issue in issues:
            if "距离过近" in issue:
                weaknesses.append(f"✗ **障碍物距离过近**：{issue}（必须修复，但不要移出路径）")
            elif "距离路径太远" in issue:
                weaknesses.append(f"✗ **障碍物距离路径太远**：{issue}（需要移动到路径附近）")
            elif "过于简单" in issue:
                weaknesses.append(f"✗ **过于简单**：{issue}（需要增加挑战性）")
    if diversity < 0.3:
        weaknesses.append(f"✗ **多样性不足**：多样性分数 {diversity:.3f}（需要增加，但不能牺牲路径阻挡）")
    if effectiveness < 0.7:
        weaknesses.append(f"✗ **有效性不足**：有效性分数 {effectiveness:.3f}（需要改进）")
    
    # 构建改进指导
    improvement_suggestions = []
    
    if strengths:
        improvement_suggestions.append("**✅ 必须保持的好特性（这些特性让配置质量高）：**")
        improvement_suggestions.extend(strengths)
        improvement_suggestions.append("")
        improvement_suggestions.append("⚠️ **关键原则**：在修复问题的过程中，绝对不能破坏上述好特性！")
        improvement_suggestions.append("")
    
    if weaknesses:
        improvement_suggestions.append("**❌ 需要改进的问题：**")
        improvement_suggestions.extend(weaknesses)
        improvement_suggestions.append("")
        improvement_suggestions.append("⚠️ **修复策略**：")
        if "距离过近" in str(weaknesses):
            improvement_suggestions.append("- 如果障碍物距离过近：沿着直线路径方向分散，不要移出路径")
            improvement_suggestions.append("  例如：如果两个障碍物在路径上距离太近，可以一个稍微前移，一个稍微后移，但都保持在路径上")
        if "缺少路径阻挡" in str(weaknesses):
            improvement_suggestions.append("- 如果没有路径阻挡：将障碍物移动到直线路径上或非常靠近路径")
        if "多样性不足" in str(weaknesses):
            improvement_suggestions.append("- 如果多样性不足：改变障碍物在路径上的具体位置，但保持路径阻挡效果")
    
    if not strengths and not weaknesses:
        improvement_suggestions.append("当前配置质量中等，尝试在保持现有特性的基础上，微调障碍物位置以增加多样性。")
    
    improvement_text = "\n".join(improvement_suggestions)
    
    prompt = f"""{TASK_SPECIFICATION}

---

## 反馈进化任务

我们评估了之前生成的障碍物配置，以下是质量评估结果：

### 质量评估结果
- **总体质量分数**: {quality_score:.3f} / 1.0
- **可解性**: {solvability:.3f}
- **难度**: {difficulty_score:.3f}
- **多样性**: {diversity:.3f}
- **有效性**: {effectiveness:.3f}

### 发现的问题
{issues_text}

### 详细反馈
{feedback}

### 之前的障碍物配置
```json
{prev_config_str}
```

### 当前 T-block 位置
- x={start_x:.3f}m, y={start_y:.3f}m, θ={np.degrees(start_theta):.1f}°
- 目标位置: x=0m, y=0m
- **直线路径**: 从 ({start_x:.3f}, {start_y:.3f}) 到 (0, 0) 的直线路径
- **路径中点**: ({start_x/2.0:.3f}, {start_y/2.0:.3f})

### 路径阻挡状态
- 之前的配置{'**成功阻挡**' if previous_path_blocked else '**没有阻挡**'}了直线路径
- 进化后的配置必须{'**继续阻挡**' if previous_path_blocked else '**添加阻挡**'}直线路径

## 改进要求

**核心目标**: 生成一个**质量分数更高**的障碍物配置。当前质量分数为 {quality_score:.3f}，进化后的配置质量分数必须**至少达到 {max(quality_score + 0.1, 0.7):.3f}** 或更高。

{improvement_text}

**进化原则（按优先级排序）**:

1. **⚠️ 最高优先级：保持好特性，修复坏特性**
   - **保持**：如果之前的配置有好的特性（如路径阻挡、可解性），进化后必须保持
   - **修复**：如果之前有问题（如距离过近），必须修复，但不能破坏好特性
   - **提升**：在保持和修复的基础上，尝试提升质量分数

2. **⚠️ 质量分数目标**
   - 当前质量分数: {quality_score:.3f}
   - 进化后的质量分数必须 ≥ {max(quality_score + 0.05, 0.65):.3f}
   - 如果当前质量已经很高（≥0.8），至少保持不下降

3. **⚠️ 路径阻挡（如果之前有，必须保持）**
   - 如果之前的配置阻挡了路径：进化后必须继续阻挡，障碍物保持在路径上或附近
   - 如果之前的配置没有阻挡路径：进化后必须添加阻挡效果
   - **修复距离过近问题的正确方法**：沿着路径方向分散障碍物，而不是移出路径
     - 例如：两个障碍物在路径上距离0.02m，应该一个稍微前移（沿路径方向），一个稍微后移，都保持在路径上

4. **保持或改进难度级别**: {difficulty}
   - 如果当前难度适中，保持；如果过低，适当提升

5. **增加多样性**（可选，不能牺牲其他特性）
   - 改变障碍物在路径上的具体位置
   - 尝试不同的聚集模式
   - 但不能破坏路径阻挡效果

## 约束条件（必须严格遵守）

1. **坐标范围**: x, y 必须在 [-0.2, 0.2] 之间
2. **⚠️ 避开起点和终点（T 形碰撞区，非圆形！）**: T-block 碰撞区是 T 形。stem 方向 0.12m 内危险，详见下方「碰撞排斥区」章节。
   - **简单安全规则：距离 > 0.12m → 任何方向都安全**
3. **⚠️ 障碍物间距**: 任意两个障碍物中心之间的距离必须严格 > 0.03m。
4. **可行性**: 必须存在至少一条可行路径

## 输出格式

⚠️ 重要：直接输出纯 JSON，不要添加代码块标记、注释或额外文字。

请严格按以下 JSON 格式输出：

```json
{{
  "reasoning": "分析之前配置的问题，说明如何基于反馈进行改进，并确保质量分数从 {quality_score:.3f} 提升到至少 {max(quality_score + 0.1, 0.7):.3f}",
  "improvements": "具体改进了哪些方面（修复的问题、增加的多样性等），以及如何确保质量分数提升",
  "expected_quality_improvement": "预期质量分数提升的原因（例如：修复了距离过近问题、保持了路径阻挡、增加了多样性等）",
  "obstacles": [
    {{"x": 0.0, "y": 0.0, "purpose": "这个障碍物的作用"}}
  ]
}}
```

注意：不要使用注释、不要有尾随逗号、输出后不要添加解释。
"""
    return prompt


def build_evolution_prompt(
    tblock_pose: List[float],
    previous_config: List[Dict],
    policy_stats: Dict,
    num_obstacles: int
) -> str:
    """
    构建进化 prompt（参考 Eurekaverse 的 Evolution Prompt）
    
    Args:
        tblock_pose: T-block 初始位姿
        previous_config: 之前的障碍物配置
        policy_stats: 策略训练统计信息
        num_obstacles: 障碍物数量
    """
    
    start_x, start_y, start_theta = tblock_pose
    
    # 格式化之前的配置
    prev_config_str = json.dumps(previous_config, indent=2, ensure_ascii=False)
    
    # 提取统计信息
    success_rate = policy_stats.get("success_rate", 0.5)
    avg_steps = policy_stats.get("avg_steps", 100)
    collision_rate = policy_stats.get("collision_rate", 0.1)
    avg_reward = policy_stats.get("avg_reward", 0)
    
    # 根据成功率决定进化方向
    if success_rate > 0.8:
        evolution_direction = "increase"
        direction_instruction = """
**进化方向: 增加难度**
当前策略表现优秀（成功率 > 80%），请创建更具挑战性的配置：
- 缩小绕行空间
- 增加障碍物数量或调整位置使路径更复杂
- 考虑需要多步操作的布局
- 但仍需确保任务可完成"""
    elif success_rate < 0.2:
        evolution_direction = "decrease"
        direction_instruction = """
**进化方向: 降低难度**
当前策略表现较差（成功率 < 20%），请创建更简单的配置：
- 增大绕行空间
- 减少障碍物对路径的阻挡程度
- 确保有明显的可行路径
- 检查之前的配置是否过于困难"""
    else:
        evolution_direction = "vary"
        direction_instruction = """
**进化方向: 保持难度，增加多样性**
当前策略表现中等，请创建难度相近但布局不同的配置：
- 保持类似的绕行空间
- 改变障碍物的具体位置
- 测试不同的技能组合
- 避免与之前配置过于相似"""
    
    prompt = f"""{TASK_SPECIFICATION}

---

## 进化任务

我们使用之前的障碍物配置训练了一个 Push-T 策略，以下是训练统计：

### 策略性能统计
- **成功率**: {success_rate*100:.1f}%
- **平均步数**: {avg_steps:.1f}
- **碰撞率**: {collision_rate*100:.1f}%
- **平均奖励**: {avg_reward:.2f}

### 之前的障碍物配置
```json
{prev_config_str}
```

### 当前 T-block 位置
- x={start_x:.3f}m, y={start_y:.3f}m, θ={np.degrees(start_theta):.1f}°

{direction_instruction}

## 约束条件（必须严格遵守）

1. **坐标范围**: x, y 必须在 [-0.2, 0.2] 之间
2. **⚠️ 避开起点和终点（T 形碰撞区，非圆形！）**: T-block 碰撞区是 T 形。stem 方向 0.12m 内危险，详见下方「碰撞排斥区」章节。
   - **简单安全规则：距离 > 0.12m → 任何方向都安全**
3. **⚠️ 障碍物间距**: 任意两个障碍物中心之间的距离必须严格 > 0.03m。
4. **可行性**: 必须存在至少一条可行路径

## 输出格式

⚠️ 重要：直接输出纯 JSON，不要添加代码块标记、注释或额外文字。

请严格按以下 JSON 格式输出：

```json
{{
  "reasoning": "分析之前配置的问题，说明改进思路",
  "changes": "相比之前配置的主要变化",
  "expected_improvement": "预期这个配置会如何改善训练",
  "obstacles": [
    {{"x": 0.0, "y": 0.0, "purpose": "这个障碍物的作用"}}
  ]
}}
```

注意：不要使用注释、不要有尾随逗号、输出后不要添加解释。
"""
    return prompt


# ============================================================================
# LLM Obstacle Generator V2
# ============================================================================

class LLMObstacleGeneratorV3:
    """
    改进版 LLM 障碍物生成器
    
    主要改进：
    1. 更详细的 prompt（参考 Eurekaverse）
    2. 支持进化模式（根据策略反馈调整）
    3. 更好的几何分析
    4. 更结构化的输出
    """
    
    def __init__(self,
                 api_type: str = "deepseek",
                 api_key: str = None,
                 model: str = None,
                 base_url: str = None,
                 temperature: float = 0.9,
                 max_retry_attempts: int = 3,
                 verbose: bool = True):
        """初始化"""
        self.api_type = api_type
        self.temperature = temperature  # 提高temperature以增加多样性
        self.verbose = verbose
        self.model = model or ("deepseek-chat" if api_type == "deepseek" else "gpt-4")
        self.max_retry_attempts = max(1, int(max_retry_attempts))
        
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
        
        # 任务参数
        self.target_pose = [0, 0, -np.pi/4]
        self.obstacle_size = 0.01  # 半边长
        
        # 历史记录（用于进化）
        self.generation_history = []
        # 最近一次生成的调试信息（用于可视化失败case：包含被过滤的障碍物与原因）
        self.last_debug: Dict = {}
        
        if self.verbose:
            print(f"✓ LLMObstacleGeneratorV3 初始化成功 (model: {self.model})")
    
    def generate(self,
                 tblock_pose: List[float],
                 num_obstacles: int = 1,
                 difficulty: str = "medium",
                 generation_id: int = None) -> List[Dict]:
        """
        生成障碍物配置（初始生成）
        
        Args:
            tblock_pose: T-block 初始位姿 [x, y, θ]
            num_obstacles: 障碍物数量
            difficulty: 难度级别 ("easy", "medium", "hard")
            generation_id: 生成序号（用于多样性提示，可选）
        """
        prompt = build_generation_prompt(
            tblock_pose=tblock_pose,
            num_obstacles=num_obstacles,
            difficulty=difficulty
        )
        config = self._generate_with_targeted_retries(
            base_prompt=prompt,
            tblock_pose=tblock_pose,
            num_obstacles=num_obstacles,
            generation_id=generation_id,
            difficulty=difficulty
        )
        
        # 记录历史
        self.generation_history.append({
            "type": "initial",
            "tblock_pose": tblock_pose,
            "config": config,
            "difficulty": difficulty
        })
        
        return config
    
    def evolve(self,
               tblock_pose: List[float],
               previous_config: List[Dict],
               policy_stats: Dict,
               num_obstacles: int = None) -> List[Dict]:
        """
        进化障碍物配置（根据策略反馈）
        
        Args:
            tblock_pose: T-block 初始位姿
            previous_config: 之前的障碍物配置
            policy_stats: 策略训练统计，包含：
                - success_rate: 成功率 (0-1)
                - avg_steps: 平均步数
                - collision_rate: 碰撞率
                - avg_reward: 平均奖励
            num_obstacles: 障碍物数量（默认与之前相同）
        """
        if num_obstacles is None:
            num_obstacles = len(previous_config)
        
        prompt = build_evolution_prompt(
            tblock_pose=tblock_pose,
            previous_config=previous_config,
            policy_stats=policy_stats,
            num_obstacles=num_obstacles
        )
        config = self._generate_with_targeted_retries(
            base_prompt=prompt,
            tblock_pose=tblock_pose,
            num_obstacles=num_obstacles
        )
        
        # 记录历史
        self.generation_history.append({
            "type": "evolution",
            "tblock_pose": tblock_pose,
            "previous_config": previous_config,
            "policy_stats": policy_stats,
            "new_config": config
        })
        
        return config
    
    def evolve_with_feedback(self,
                             tblock_pose: List[float],
                             previous_config: List[Dict],
                             quality_score: float,
                             detailed_scores: Dict,
                             feedback: str,
                             num_obstacles: int = None,
                             difficulty: str = "medium") -> List[Dict]:
        """
        基于质量评估反馈进化障碍物配置
        
        Args:
            tblock_pose: T-block 初始位姿
            previous_config: 之前的障碍物配置
            quality_score: 质量评分 (0-1)
            detailed_scores: 详细评分字典
            feedback: 评估反馈文本
            num_obstacles: 障碍物数量（默认与之前相同）
            difficulty: 难度级别
        """
        if num_obstacles is None:
            num_obstacles = len(previous_config)
        
        prompt = build_feedback_evolution_prompt(
            tblock_pose=tblock_pose,
            previous_config=previous_config,
            quality_score=quality_score,
            detailed_scores=detailed_scores,
            feedback=feedback,
            num_obstacles=num_obstacles,
            difficulty=difficulty
        )
        config = self._generate_with_targeted_retries(
            base_prompt=prompt,
            tblock_pose=tblock_pose,
            num_obstacles=num_obstacles,
            difficulty=difficulty
        )
        
        # 记录历史
        self.generation_history.append({
            "type": "feedback_evolution",
            "tblock_pose": tblock_pose,
            "previous_config": previous_config,
            "quality_score": quality_score,
            "detailed_scores": detailed_scores,
            "feedback": feedback,
            "new_config": config
        })
        
        return config

    def _generate_with_targeted_retries(self,
                                        base_prompt: str,
                                        tblock_pose: List[float],
                                        num_obstacles: int,
                                        generation_id: int = None,
                                        difficulty: str = "medium") -> List[Dict]:
        """生成并在失败时基于拒绝原因定向重试，而不是盲重试。"""
        # 注入 T 形排斥区几何描述，让 LLM 自主推理安全坐标
        exclusion_desc = self._describe_exclusion_zone(tblock_pose)
        base_prompt = base_prompt + exclusion_desc

        # 跨重试累积已通过的障碍物，避免 JSON 解析失败时丢失
        best_accepted: List[Dict] = []
        current_prompt = base_prompt

        for attempt in range(1, self.max_retry_attempts + 1):
            call_generation_id = generation_id if attempt == 1 else None
            response = self._call_llm(current_prompt, generation_id=call_generation_id)
            config = self._parse_response(response, tblock_pose, num_obstacles)

            # 合并：取本轮与历史中更多的那个
            if len(config) > len(best_accepted):
                best_accepted = config

            if len(best_accepted) >= num_obstacles:
                return best_accepted[:num_obstacles]

            if attempt < self.max_retry_attempts:
                if self.verbose:
                    print(
                        f"  ⚠️ 尝试 {attempt}: 有效障碍物数量不足 (期望 {num_obstacles}, 实际 {len(best_accepted)})，"
                        "将基于拒绝原因定向重试..."
                    )
                # 确保 last_debug 中保留最佳 accepted 记录（即使本轮解析失败）
                if len(config) < len(best_accepted):
                    self.last_debug["accepted_obstacles"] = best_accepted
                current_prompt = self._build_targeted_retry_prompt(
                    base_prompt=base_prompt,
                    tblock_pose=tblock_pose,
                    num_obstacles=num_obstacles,
                    attempt=attempt + 1
                )

        return best_accepted

    def _build_targeted_retry_prompt(self,
                                     base_prompt: str,
                                     tblock_pose: List[float],
                                     num_obstacles: int,
                                     attempt: int) -> str:
        """构建精简的重试 prompt，不再拼接完整 base_prompt 以避免过长导致 JSON 解析失败。"""
        accepted = self.last_debug.get("accepted_obstacles", [])
        rejected = self.last_debug.get("rejected_obstacles", [])
        debug_error = self.last_debug.get("error")

        accepted_text = (
            "\n".join([
                f"- ({obs['x']:.3f}, {obs['y']:.3f})：已通过"
                for obs in accepted
            ])
            if accepted else "- 无"
        )
        rejected_text = (
            "\n".join([
                f"- ({obs['x']:.3f}, {obs['y']:.3f})：距T-block中心太近，被碰撞检测拒绝"
                for obs in rejected
            ])
            if rejected else "- 无"
        )

        start_x, start_y = tblock_pose[0], tblock_pose[1]
        need_count = max(0, num_obstacles - len(accepted))

        retry_prompt = f"""你正在为 Push-T 任务生成障碍物（0.02m×0.02m 方块）。

起点 T-block: ({start_x:.3f}, {start_y:.3f})
终点 T-block: (0.000, 0.000)
坐标范围: [-0.2, 0.2]

上次生成部分失败，请修复。

## 已通过的障碍物（原样保留）
{accepted_text}

## 被拒绝的障碍物（禁止复用这些坐标）
{rejected_text}

## 任务
输出 {num_obstacles} 个障碍物（保留已通过的 {len(accepted)} 个，新增 {need_count} 个）。

## 安全规则
- 所有障碍物距起点和终点中心必须 > **0.12m**
- 障碍物之间距离 > 0.03m

直接输出 JSON，不要代码块标记：
{{"obstacles": [{{"x": 0.0, "y": 0.0, "purpose": "..."}}]}}"""

        return retry_prompt
    
    def _call_llm(self, prompt: str, generation_id: int = None) -> str:
        """调用 LLM"""
        try:
            # 将本轮已生成的历史布局注入 prompt，让 LLM 自行规避重复
            if generation_id is not None and len(self.generation_history) > 0:
                # 只取最近同批次的 initial 类型历史
                recent = [
                    h for h in self.generation_history
                    if h.get("type") == "initial" and h.get("config")
                ]
                if recent:
                    lines = [
                        "\n\n---",
                        f"## ⚠️ 已有布局（第 {generation_id} 次生成，以下是之前已产生的配置，本次必须不同）\n",
                    ]
                    for i, h in enumerate(recent[-5:], 1):  # 最多展示最近5条
                        coords = ", ".join(
                            f"({obs['x']:.3f}, {obs['y']:.3f})" for obs in h["config"]
                        )
                        lines.append(f"- 第 {i} 次: [{coords}]")
                    lines.append(
                        "\n请生成与上述布局**不同**的障碍物配置。"
                    )
                    prompt = prompt + "\n".join(lines)
            
            # 后续生成适当提高温度，增加探索性
            temp = self.temperature
            if generation_id is not None and generation_id > 1:
                temp = min(self.temperature + 0.1 * (generation_id - 1), 1.5)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=temp,
                max_tokens=2000,
                response_format={"type": "json_object"}  # 强制 JSON 输出
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"LLM 调用失败: {e}")
            return None
    
    def _parse_response(self, response: str, tblock_pose: List[float], num_obstacles: int) -> List[Dict]:
        """解析 LLM 响应（改进版：更鲁棒的 JSON 提取）"""
        if response is None:
            print("⚠ LLM 响应为空，返回空配置")
            self.last_debug = {
                "ok": False,
                "error": "empty_response",
                "raw_response": None,
                "expected_num_obstacles": num_obstacles,
                "all_obstacles": [],
                "accepted_obstacles": [],
                "rejected_obstacles": []
            }
            return []
        
        try:
            # 改进的 JSON 提取：处理多种格式
            data = self._extract_json_robust(response)
            if data is None:
                raise ValueError("无法提取有效 JSON")
            
            # 打印分析信息
            if self.verbose:
                if "reasoning" in data:
                    print(f"\n📝 LLM 设计思路: {data['reasoning']}")
                if "analysis" in data:
                    print(f"📊 分析: {data['analysis']}")
            
            # 提取障碍物（同时保留被过滤的障碍物，便于失败case可视化）
            obstacles = []  # accepted
            all_obstacles = []  # accepted + rejected
            for obs in data.get("obstacles", []):
                x = float(obs["x"])
                y = float(obs["y"])
                
                # 验证范围
                x = np.clip(x, -0.2, 0.2)
                y = np.clip(y, -0.2, 0.2)
                
                purpose = obs.get('purpose', '')
                # 检查碰撞（返回详细原因字符串，空串=安全）
                collision_reason = self._check_collision(np.array([x, y]), tblock_pose)
                if not collision_reason:
                    accepted = {
                        'x': x,
                        'y': y,
                        'purpose': purpose
                    }
                    obstacles.append(accepted)
                    all_obstacles.append({**accepted, "status": "accepted"})
                    if self.verbose:
                        print(f"✓ 障碍物 ({x:.3f}, {y:.3f}): {purpose}")
                else:
                    rejected = {
                        'x': x,
                        'y': y,
                        'purpose': purpose,
                        "status": "rejected",
                        "reject_reason": collision_reason
                    }
                    all_obstacles.append(rejected)
                    if self.verbose:
                        print(f"✗ 障碍物 ({x:.3f}, {y:.3f}) {collision_reason}")
            
            # 不再自动补充备用障碍物，直接返回当前解析到的结果
            if len(obstacles) < num_obstacles:
                if self.verbose:
                    print(f"⚠ 有效障碍物数量不足：期望 {num_obstacles} 个，实际 {len(obstacles)} 个")

            accepted_trimmed = obstacles[:num_obstacles]
            rejected_only = [o for o in all_obstacles if o.get("status") == "rejected"]
            self.last_debug = {
                "ok": True,
                "error": None,
                "raw_response": response,
                "expected_num_obstacles": num_obstacles,
                "all_obstacles": all_obstacles,
                "accepted_obstacles": accepted_trimmed,
                "rejected_obstacles": rejected_only
            }
            return accepted_trimmed
            
        except Exception as e:
            # 解析失败时保留上一轮的 accepted/rejected，避免丢失已通过的障碍物
            print(f"❌ JSON 解析失败: {e}")
            if self.verbose:
                print(f"原始响应（前 500 字符）:\n{response[:500]}")
            prev_accepted = self.last_debug.get("accepted_obstacles", [])
            prev_rejected = self.last_debug.get("rejected_obstacles", [])
            self.last_debug = {
                "ok": False,
                "error": f"parse_error: {e}",
                "raw_response": response,
                "expected_num_obstacles": num_obstacles,
                "all_obstacles": [],
                "accepted_obstacles": prev_accepted,
                "rejected_obstacles": prev_rejected
            }
            return []
    
    def _check_collision(self, obstacle_pos: np.ndarray, tblock_pose: List[float]) -> str:
        """检查障碍物是否与 T-block 碰撞或过近。

        Returns:
            空字符串 "" 表示安全（无碰撞）；
            非空字符串为碰撞原因的人类可读描述，会直接反馈给 LLM。
        """
        start_pos = np.array(tblock_pose[:2])
        target_pos = np.array(self.target_pose[:2])

        # 检查与初始 T-block 的碰撞
        if analytic_obs_collision_check(
            Tblock_angle=tblock_pose[-1],
            obs_center=obstacle_pos - start_pos,
            obs_size=self.obstacle_size * 2,
            threshold=0.02
        ):
            dist = np.linalg.norm(obstacle_pos - start_pos)
            return (f"与起点T-block碰撞（距起点中心{dist:.3f}m，"
                    f"起点在({start_pos[0]:.3f},{start_pos[1]:.3f})，"
                    f"竖杆方向约{np.degrees(tblock_pose[-1])-90:.0f}°）")

        # 检查与目标 T-block 的碰撞
        if analytic_obs_collision_check(
            Tblock_angle=self.target_pose[-1],
            obs_center=obstacle_pos - target_pos,
            obs_size=self.obstacle_size * 2,
            threshold=0.02
        ):
            dist = np.linalg.norm(obstacle_pos - target_pos)
            return (f"与终点T-block碰撞（距终点中心{dist:.3f}m，"
                    f"终点在({target_pos[0]:.3f},{target_pos[1]:.3f})，"
                    f"竖杆方向约{np.degrees(self.target_pose[-1])-90:.0f}°）")

        return ""

    def _describe_exclusion_zone(self, tblock_pose: List[float]) -> str:
        """生成 T-block 排斥区的精确几何描述，让 LLM 自主推理安全坐标。"""
        start_x, start_y, start_theta = tblock_pose
        target_x, target_y, target_theta = self.target_pose

        def _stem_tip(cx, cy, angle):
            """计算 T-block 竖杆末端在世界坐标中的位置"""
            local_tip = np.array([0, -0.085])
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            return np.array([
                cx + cos_a * local_tip[0] - sin_a * local_tip[1],
                cy + sin_a * local_tip[0] + cos_a * local_tip[1]
            ])

        start_stem = _stem_tip(start_x, start_y, start_theta)
        target_stem = _stem_tip(target_x, target_y, target_theta)

        start_stem_dir = np.degrees(np.arctan2(
            start_stem[1] - start_y, start_stem[0] - start_x))
        target_stem_dir = np.degrees(np.arctan2(
            target_stem[1] - target_y, target_stem[0] - target_x))

        lines = [
            "",
            "### ⚠️ T-block 碰撞排斥区（关键几何知识）",
            "",
            "T-block 形状如下（局部坐标，0° 时）：",
            "```",
            "     ┌──────────┐         ← 横杆 0.10m×0.03m",
            "     │          │",
            "     └──┬────┬──┘",
            "        │    │             ← 竖杆 0.03m×0.07m",
            "        │    │",
            "        └────┘",
            "```",
            "",
            "排斥区是 **T 形**（不是圆），但你不需要计算具体方向。",
            "",
            "**唯一安全规则：障碍物距 T-block 中心 > 0.12m → 任何方向都绝对安全。**",
            "⚠️ 不要试图利用某些方向'安全距离更短'来靠近 T-block，你的角度计算精度不够，物理引擎会直接判碰撞！",
            "",
            "",
            f"**起点 T-block** ({start_x:.3f}, {start_y:.3f}), θ={np.degrees(start_theta):.0f}°：",
            f"  - ⛔ 障碍物必须距起点中心 > 0.12m",
            "",
            f"**终点 T-block** ({target_x:.3f}, {target_y:.3f}), θ={np.degrees(target_theta):.0f}°：",
            f"  - ⛔ 障碍物必须距终点中心 > 0.12m",
            "",
            "**设计建议：**",
            "- 距离 > 0.12m 绝对安全，不需要考虑方向",
            "- 系统会自动验证坐标。若碰撞会返回具体原因并允许重试",
            "",
        ]
        return "\n".join(lines)
    
    def _fallback_generation(self, tblock_pose: List[float], num_obstacles: int) -> List[Dict]:
        """备用生成策略"""
        obstacles = []
        for i in range(num_obstacles):
            t = 0.4 + i * 0.15
            t = np.clip(t, 0.3, 0.7)
            
            x = tblock_pose[0] * (1 - t)
            y = tblock_pose[1] * (1 - t)
            
            # 添加随机扰动
            x += np.random.uniform(-0.03, 0.03)
            y += np.random.uniform(-0.03, 0.03)
            
            x = np.clip(x, -0.2, 0.2)
            y = np.clip(y, -0.2, 0.2)
            
            obstacles.append({'x': x, 'y': y, 'purpose': 'fallback'})
        
        return obstacles
    
    def _extract_json_robust(self, response: str) -> Optional[Dict]:
        """鲁棒的 JSON 提取（处理多种格式问题）"""
        
        # 方法 1：标准提取（查找第一个 { 到最后一个 }）
        try:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                raw_json = response[json_start:json_end]
                return json.loads(raw_json)
        except json.JSONDecodeError:
            pass
        
        # 方法 2：查找 ```json 代码块
        try:
            if '```json' in response:
                start = response.find('```json') + 7
                end = response.find('```', start)
                if end != -1:
                    raw_json = response[start:end].strip()
                    return json.loads(raw_json)
            elif '```' in response:
                start = response.find('```') + 3
                end = response.find('```', start)
                if end != -1:
                    raw_json = response[start:end].strip()
                    # 跳过可能的语言标识符
                    if raw_json.startswith('json'):
                        raw_json = raw_json[4:].strip()
                    return json.loads(raw_json)
        except json.JSONDecodeError:
            pass
        
        # 方法 3：逐个尝试所有 {...} 块
        try:
            brace_count = 0
            start_idx = -1
            
            for i, char in enumerate(response):
                if char == '{':
                    if brace_count == 0:
                        start_idx = i
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0 and start_idx != -1:
                        # 找到一个完整的 JSON 对象
                        candidate = response[start_idx:i+1]
                        try:
                            data = json.loads(candidate)
                            # 验证是否包含 obstacles 字段
                            if 'obstacles' in data:
                                return data
                        except json.JSONDecodeError:
                            continue
                        start_idx = -1
        except Exception:
            pass
        
        # 方法 4：尝试修复常见的 JSON 错误
        try:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                raw_json = response[json_start:json_end]
                
                # 修复：移除注释
                lines = []
                for line in raw_json.split('\n'):
                    # 移除 // 注释
                    if '//' in line:
                        line = line[:line.find('//')]
                    lines.append(line)
                raw_json = '\n'.join(lines)
                
                # 修复：移除尾随逗号
                raw_json = raw_json.replace(',]', ']').replace(',}', '}')
                
                return json.loads(raw_json)
        except json.JSONDecodeError:
            pass
        
        return None
    
    def get_history(self) -> List[Dict]:
        """获取生成历史"""
        return self.generation_history
    
    def clear_history(self):
        """清空历史"""
        self.generation_history = []
