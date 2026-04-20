"""
LLM 障碍物生成器 V2
参考 Eurekaverse 的 prompt 设计，改进 Push-T 任务的障碍物生成

主要改进：
1. 更详细的任务规格说明（参考 Eurekaverse 的 Environment Specifications）
2. 添加 in-context example（参考 Eurekaverse 的示例驱动生成）
3. 支持进化模式（根据策略反馈调整难度）
4. 更结构化的约束说明
5. 支持生成障碍物配置函数（而不仅是坐标）
"""
import numpy as np
import json
import os
from typing import List, Dict, Optional, Tuple
from DIVO.utils.util import analytic_obs_collision_check

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


# ============================================================================
# Prompt Templates (参考 Eurekaverse 设计)
# ============================================================================

SYSTEM_PROMPT = """你是一个机器人操作任务的环境设计专家，专门为 Push-T 任务设计障碍物配置。
你的目标是创建有意义的、可学习的障碍物布局，帮助训练更鲁棒的机器人策略。

请仔细阅读任务规格，遵守所有约束条件，并输出精确的 JSON 格式结果。"""


TASK_SPECIFICATION = """
## 任务背景：Push-T

Push-T 是一个机器人操作任务：一个圆柱形末端执行器需要将 T 形方块从初始位置推到目标位置。
这个任务测试机器人的：
- 路径规划能力（绕过障碍物）
- 精确操作能力（在狭窄空间中操作）
- 旋转控制能力（调整 T-block 朝向）

## 环境规格

### 工作空间
- 桌面大小: 0.5m × 0.5m
- 坐标系原点: 桌面中心
- 坐标范围: x ∈ [-0.25, 0.25], y ∈ [-0.25, 0.25]
- 障碍物有效放置范围: x, y ∈ [-0.2, 0.2]（留出边缘缓冲）

### T-block（被推物体）
- 形状: T 形，由两个矩形组成
  - 横杠: 0.1m × 0.03m
  - 竖杠: 0.03m × 0.07m
- 整体包围盒: 约 0.1m × 0.1m
- 位姿表示: [x, y, θ]，θ 为绕 z 轴旋转角度（弧度）

### 末端执行器（推杆）
- 形状: 圆柱形
- 直径: 0.02m
- 可以从任意方向推动 T-block

### 障碍物
- 形状: 正方形方块
- 尺寸: 0.02m × 0.02m（边长）
- 固定在桌面上，不可移动
- 颜色: 红色（用于可视化）

### 目标
- 位置: (0, 0)，即桌面中心
- 朝向: -45°（-π/4 弧度）
- 可视化: 绿色半透明 T 形轮廓

## 设计原则

1. **可学习性**: 障碍物应该创造有意义的挑战，但不能使任务不可能完成
2. **多样性**: 不同的障碍物配置应该测试不同的技能
3. **渐进性**: 难度应该可以通过参数控制
4. **物理合理性**: 必须留出足够的空间让 T-block 通过
"""


IN_CONTEXT_EXAMPLE = """
## 示例

以下是一个障碍物配置示例，展示了正确的格式和设计思路：

### 输入
- T-block 初始位置: x=0.15m, y=0.15m, θ=135°
- 目标位置: x=0, y=0, θ=-45°
- 障碍物数量: 2
- 场景类型: path_blocking
- 难度: medium

### 输出
```json
{
  "reasoning": "T-block 在第一象限，需要向原点移动。直线路径经过 (0.075, 0.075) 附近。放置一个障碍物在路径中点阻挡直线移动，另一个在稍偏位置形成需要绕行的布局。这样机器人必须学会规划非直线路径。",
  "analysis": {
    "start_to_target_distance": 0.212,
    "direct_path_midpoint": [0.075, 0.075],
    "tblock_current_orientation": "横杠朝向左上",
    "required_rotation": "需要顺时针旋转约180°"
  },
  "obstacles": [
    {"x": 0.07, "y": 0.07, "purpose": "阻挡直线路径中点"},
    {"x": 0.04, "y": 0.10, "purpose": "迫使从上方绕行"}
  ]
}
```

### 设计要点
1. 分析了起点到终点的几何关系
2. 障碍物放置有明确的目的
3. 留出了足够的绕行空间（约 0.12m 宽度 > T-block 宽度 0.1m）
4. 两个障碍物形成协同约束
"""


def build_generation_prompt(
    tblock_pose: List[float],
    scenario_type: str,
    num_obstacles: int,
    difficulty: str
) -> str:
    """
    构建初始生成 prompt（参考 Eurekaverse 的 Initial Generation）
    """
    
    # 计算一些有用的几何信息
    start_x, start_y, start_theta = tblock_pose
    target_x, target_y, target_theta = 0, 0, -np.pi/4
    
    distance = np.sqrt(start_x**2 + start_y**2)
    midpoint_x, midpoint_y = start_x / 2, start_y / 2
    angle_to_target = np.arctan2(-start_y, -start_x)
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
    
    # 场景类型说明
    scenario_descriptions = {
        "path_blocking": """
**场景: 路径阻挡 (path_blocking)**
目标: 在 T-block 到目标的直线路径上放置障碍物，迫使机器人学会绕行。
策略建议:
- 计算起点到终点的直线路径
- 在路径中点或关键位置放置障碍物
- 确保存在可行的绕行路线（至少 0.12m 宽度）
技能测试: 路径规划、避障能力""",
        
        "corridor": """
**场景: 狭窄通道 (corridor)**
目标: 用障碍物形成一个狭窄通道，T-block 必须精确通过。
策略建议:
- 在路径两侧对称放置障碍物
- 通道宽度: 0.12-0.15m（略大于 T-block 宽度 0.1m）
- 通道应该在路径的关键位置
技能测试: 精确控制、直线推动能力""",
        
        "rotation": """
**场景: 旋转挑战 (rotation)**
目标: 障碍物位置要求 T-block 必须先旋转才能通过。
策略建议:
- 分析 T-block 当前朝向
- 放置障碍物使当前朝向无法直接通过
- 考虑 T 形的不对称性
技能测试: 旋转控制、姿态调整能力""",
        
        "surrounding": """
**场景: 包围目标 (surrounding)**
目标: 在目标位置周围放置障碍物，限制进入角度。
策略建议:
- 在目标 (0, 0) 周围放置障碍物
- 留出一个或两个进入通道
- 通道方向应该与 T-block 初始位置不直接对齐
技能测试: 多步规划、进入角度选择""",
        
        "auto": """
**场景: 自动设计 (auto)**
根据 T-block 当前位置和朝向，自动选择最有挑战性的配置。
请分析几何关系，选择最能测试策略能力的障碍物布局。"""
    }
    
    # 难度说明
    difficulty_descriptions = {
        "easy": """
**难度: 简单 (easy)**
- 障碍物远离直线路径，只做轻微干扰
- 绕行空间充足（> 0.15m）
- 主要测试基本避障意识""",
        
        "medium": """
**难度: 中等 (medium)**
- 障碍物阻挡主要路径，但有明确的绕行方案
- 绕行空间适中（0.12-0.15m）
- 测试路径规划和基本精确控制""",
        
        "hard": """
**难度: 困难 (hard)**
- 障碍物形成复杂约束
- 绕行空间较窄（0.11-0.12m，接近 T-block 宽度）
- 可能需要多步操作或精确旋转
- 测试高级规划和精确操作能力"""
    }
    
    prompt = f"""{TASK_SPECIFICATION}

{IN_CONTEXT_EXAMPLE}

---

## 当前任务

### 输入参数
- **T-block 初始位置**: x={start_x:.3f}m, y={start_y:.3f}m, θ={np.degrees(start_theta):.1f}°
- **目标位置**: x=0, y=0, θ=-45°
- **障碍物数量**: {num_obstacles}
- **场景类型**: {scenario_type}
- **难度**: {difficulty}

### 几何分析（供参考）
- T-block 所在位置: {quadrant}
- 到目标距离: {distance:.3f}m
- 直线路径中点: ({midpoint_x:.3f}, {midpoint_y:.3f})
- 到目标方向角: {np.degrees(angle_to_target):.1f}°
- 需要旋转角度: {np.degrees(rotation_needed):.1f}°

{scenario_descriptions.get(scenario_type, scenario_descriptions["auto"])}

{difficulty_descriptions.get(difficulty, difficulty_descriptions["medium"])}

## 约束条件（必须严格遵守）

1. **坐标范围**: x, y 必须在 [-0.2, 0.2] 之间
2. **避开起点**: 障碍物中心与 T-block 初始位置 ({start_x:.2f}, {start_y:.2f}) 的距离必须 > 0.08m
3. **避开终点**: 障碍物中心与目标位置 (0, 0) 的距离必须 > 0.08m
4. **障碍物间距**: 多个障碍物之间的距离必须 > 0.03m
5. **可行性**: 必须存在至少一条宽度 ≥ 0.11m 的可行路径

## 输出格式

请严格按以下 JSON 格式输出，不要有其他内容：

```json
{{
  "reasoning": "详细说明设计思路，包括为什么选择这些位置",
  "analysis": {{
    "path_strategy": "描述预期的绕行策略",
    "skill_tested": "这个配置主要测试什么技能",
    "estimated_difficulty": "实际难度评估"
  }},
  "obstacles": [
    {{"x": 0.0, "y": 0.0, "purpose": "这个障碍物的作用"}}
  ]
}}
```
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
2. **避开起点**: 障碍物中心与 T-block 初始位置的距离必须 > 0.08m
3. **避开终点**: 障碍物中心与目标位置 (0, 0) 的距离必须 > 0.08m
4. **障碍物间距**: 多个障碍物之间的距离必须 > 0.03m
5. **可行性**: 必须存在至少一条可行路径

## 输出格式

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
"""
    return prompt


# ============================================================================
# LLM Obstacle Generator V2
# ============================================================================

class LLMObstacleGeneratorV2:
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
                 temperature: float = 0.7):
        """初始化"""
        self.api_type = api_type
        self.temperature = temperature
        self.model = model or ("deepseek-chat" if api_type == "deepseek" else "gpt-4")
        
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
        
        print(f"✓ LLMObstacleGeneratorV2 初始化成功 (model: {self.model})")
    
    def generate(self,
                 tblock_pose: List[float],
                 scenario_type: str = "auto",
                 num_obstacles: int = 1,
                 difficulty: str = "medium") -> List[Dict]:
        """
        生成障碍物配置（初始生成）
        """
        prompt = build_generation_prompt(
            tblock_pose=tblock_pose,
            scenario_type=scenario_type,
            num_obstacles=num_obstacles,
            difficulty=difficulty
        )
        
        response = self._call_llm(prompt)
        config = self._parse_response(response, tblock_pose, num_obstacles)
        
        # 记录历史
        self.generation_history.append({
            "type": "initial",
            "tblock_pose": tblock_pose,
            "config": config,
            "scenario_type": scenario_type,
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
        
        response = self._call_llm(prompt)
        config = self._parse_response(response, tblock_pose, num_obstacles)
        
        # 记录历史
        self.generation_history.append({
            "type": "evolution",
            "tblock_pose": tblock_pose,
            "previous_config": previous_config,
            "policy_stats": policy_stats,
            "new_config": config
        })
        
        return config
    
    def _call_llm(self, prompt: str) -> str:
        """调用 LLM"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"LLM 调用失败: {e}")
            return None
    
    def _parse_response(self, response: str, tblock_pose: List[float], num_obstacles: int) -> List[Dict]:
        """解析 LLM 响应"""
        if response is None:
            return self._fallback_generation(tblock_pose, num_obstacles)
        
        try:
            # 提取 JSON
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start == -1 or json_end == 0:
                raise ValueError("未找到 JSON")
            
            data = json.loads(response[json_start:json_end])
            
            # 打印分析信息
            if "reasoning" in data:
                print(f"\n📝 LLM 设计思路: {data['reasoning']}")
            if "analysis" in data:
                print(f"📊 分析: {data['analysis']}")
            
            # 提取障碍物
            obstacles = []
            for obs in data.get("obstacles", []):
                x = float(obs["x"])
                y = float(obs["y"])
                
                # 验证范围
                x = np.clip(x, -0.2, 0.2)
                y = np.clip(y, -0.2, 0.2)
                
                # 检查碰撞
                if not self._check_collision(np.array([x, y]), tblock_pose):
                    obstacles.append({
                        'x': x, 
                        'y': y,
                        'purpose': obs.get('purpose', '')
                    })
                    print(f"✓ 障碍物 ({x:.3f}, {y:.3f}): {obs.get('purpose', '')}")
                else:
                    print(f"✗ 障碍物 ({x:.3f}, {y:.3f}) 碰撞，跳过")
            
            # 补充不足的障碍物
            while len(obstacles) < num_obstacles:
                fallback = self._fallback_generation(tblock_pose, 1)[0]
                obstacles.append(fallback)
            
            return obstacles[:num_obstacles]
            
        except Exception as e:
            print(f"解析失败: {e}")
            return self._fallback_generation(tblock_pose, num_obstacles)
    
    def _check_collision(self, obstacle_pos: np.ndarray, tblock_pose: List[float]) -> bool:
        """检查碰撞"""
        # 检查与初始 T-block 的碰撞
        if analytic_obs_collision_check(
            Tblock_angle=tblock_pose[-1],
            obs_center=obstacle_pos - np.array(tblock_pose[:2]),
            obs_size=self.obstacle_size * 2,
            threshold=0.01
        ):
            return True
        
        # 检查与目标 T-block 的碰撞
        if analytic_obs_collision_check(
            Tblock_angle=self.target_pose[-1],
            obs_center=obstacle_pos - np.array(self.target_pose[:2]),
            obs_size=self.obstacle_size * 2,
            threshold=0.01
        ):
            return True
        
        return False
    
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
    
    def get_history(self) -> List[Dict]:
        """获取生成历史"""
        return self.generation_history
    
    def clear_history(self):
        """清空历史"""
        self.generation_history = []
