# 障碍物位置策略设计

## 核心思想

**难度 ≠ 障碍物数量，难度 = 障碍物的战略位置**

同样是 2 个障碍物：
- 放在角落 → 简单（不影响路径）
- 放在路径中点 → 中等（需要绕行）
- 一个挡路径，一个挡绕行 → 困难（需要精确规划）

---

## 1. Easy 模式：外围干扰策略

### 策略描述
```
障碍物放在外围区域，远离 T-block 到目标的关键路径
目标：让策略学会"意识到障碍物的存在"，但不造成实质性阻挡
```

### 位置约束
```python
{
    "strategy_type": "off_path",
    "min_distance_to_path": 0.08,      # 距离直线路径 ≥ 8cm
    "min_distance_to_start": 0.10,     # 距离起点 ≥ 10cm
    "min_distance_to_target": 0.10,    # 距离终点 ≥ 10cm
    "min_passage_width": 0.15,         # 通道宽度 ≥ 15cm
    "placement_zone": "peripheral",    # 外围区域
    "block_direct_path": False         # 不阻挡直线路径
}
```

### 可视化示例

```
桌面 (0.5m × 0.5m)
┌─────────────────────────────────────┐
│  ●                            ●     │  ← 障碍物在角落
│                                     │
│                                     │
│         🔵 T-block                  │
│          ╲                          │
│           ╲  直线路径畅通            │
│            ╲                        │
│             ╲                       │
│              ╲                      │
│               🟢 Target             │
│                                     │
│  ●                            ●     │  ← 障碍物在角落
└─────────────────────────────────────┘

特点：
- 障碍物不在路径上
- 直线推动即可完成
- 主要测试基本的避障意识
```

### LLM Prompt 示例
```
难度: 简单 (easy)

位置策略：
- 将障碍物放置在桌面外围区域
- 确保障碍物距离 T-block 到目标的直线路径至少 8cm
- 不要阻挡任何可能的推动路径
- 目标是让机器人"看到"障碍物，但不需要绕行

示例：如果 T-block 在 (0.15, 0.15)，目标在 (0, 0)
- 可以放在 (-0.15, 0.15) 或 (0.15, -0.15) 等角落位置
- 不要放在 (0.075, 0.075) 这样的路径中点
```

---

## 2. Medium 模式：路径阻挡策略

### 策略描述
```
障碍物阻挡直线路径，但留有明显的绕行空间
目标：迫使策略学会路径规划和绕行
```

### 位置约束
```python
{
    "strategy_type": "partial_blocking",
    "path_blocking_probability": 0.6,  # 60% 概率阻挡直线路径
    "min_distance_to_start": 0.08,
    "min_distance_to_target": 0.08,
    "min_passage_width": 0.12,         # 通道宽度 12cm
    "placement_zone": "mid_path",      # 路径中段
    "block_direct_path": True,
    "alternative_path_width": 0.13     # 绕行路径宽度 13cm
}
```

### 可视化示例

#### 场景 A：单点阻挡
```
┌─────────────────────────────────────┐
│                                     │
│                                     │
│         🔵 T-block                  │
│          ╲                          │
│           ╲                         │
│            ●  ← 障碍物挡住直线路径   │
│           ╱ ╲                       │
│      绕行 ╱   ╲ 绕行                │
│         ╱     ╲                     │
│               🟢 Target             │
│                                     │
└─────────────────────────────────────┘

特点：
- 直线路径被阻挡
- 左右两侧都有 12-13cm 的绕行空间
- 需要规划绕行路径
```

#### 场景 B：狭窄通道
```
┌─────────────────────────────────────┐
│                                     │
│         🔵 T-block                  │
│          │                          │
│          │                          │
│          │  ●        ●              │
│          │   ╲      ╱               │
│          │    ╲    ╱  ← 12cm 通道   │
│          │     ╲  ╱                 │
│          ▼      ╲╱                  │
│               🟢 Target             │
│                                     │
└─────────────────────────────────────┘

特点：
- 两个障碍物形成通道
- 通道宽度 12cm（略大于 T-block 10cm）
- 需要精确控制通过
```

### LLM Prompt 示例
```
难度: 中等 (medium)

位置策略：
- 在 T-block 到目标的直线路径中点附近放置障碍物
- 阻挡直线推动，迫使机器人绕行
- 但要确保留有至少 12cm 宽的绕行通道
- 可以考虑：
  1. 单个障碍物挡在路径中点
  2. 两个障碍物形成狭窄通道
  3. 一个挡路径，一个在侧面引导绕行方向

约束：
- 障碍物距离起点和终点至少 8cm
- 绕行路径宽度 ≥ 12cm
```

---

## 3. Hard 模式：复杂约束策略

### 策略描述
```
障碍物形成复杂的空间约束，需要：
- 精确的路径规划（多个障碍物协同约束）
- 旋转控制（某些朝向无法通过）
- 多步推理（先推到中间位置调整，再推到目标）
```

### 位置约束
```python
{
    "strategy_type": "complex_constraint",
    "path_blocking_probability": 0.9,
    "min_distance_to_start": 0.06,     # 可以更靠近
    "min_distance_to_target": 0.06,
    "min_passage_width": 0.11,         # 窄通道（接近 T-block）
    "placement_zone": "strategic",     # 战略位置
    "block_direct_path": True,
    "block_alternative_paths": True,   # 也阻挡部分绕行路径
    "require_rotation": 0.4,           # 40% 需要旋转
    "multi_step_required": 0.3         # 30% 需要多步规划
}
```

### 可视化示例

#### 场景 A：多重阻挡
```
┌─────────────────────────────────────┐
│                                     │
│         🔵 T-block                  │
│          ╲                          │
│           ●  ← 挡住直线路径          │
│          ╱ ╲                        │
│         ╱   ●  ← 挡住左侧绕行        │
│        ╱      ╲                     │
│       ╱        ╲  ← 只能从右侧绕    │
│      ╱          ╲                   │
│               🟢 Target             │
│                                     │
└─────────────────────────────────────┘

特点：
- 多个障碍物协同约束
- 只留一条窄通道（11cm）
- 需要精确规划和控制
```

#### 场景 B：包围目标
```
┌─────────────────────────────────────┐
│                                     │
│         🔵 T-block                  │
│          │                          │
│          │                          │
│          │      ●                   │
│          │    ╱   ╲                 │
│          │   ●  🟢  ●  ← 目标被包围  │
│          │    ╲   ╱                 │
│          ▼      ●                   │
│           ╲                         │
│            ╲  ← 只能从缝隙进入       │
└─────────────────────────────────────┘

特点：
- 障碍物围绕目标
- 只留一个进入角度
- 需要精确对准缝隙
```

#### 场景 C：旋转挑战
```
┌─────────────────────────────────────┐
│                                     │
│    🔵 T-block (横杠朝右)            │
│    ├─┤                              │
│     │                               │
│     │                               │
│     │   ●     ●                     │
│     │    ╲   ╱  ← 11cm 窄通道       │
│     │     ╲ ╱   (横杠无法通过)      │
│     ▼      ╲                        │
│          🟢 Target                  │
│                                     │
└─────────────────────────────────────┘

特点：
- 通道宽度 11cm
- T-block 横杠宽度 10cm
- 当前朝向无法通过，必须先旋转
```

### LLM Prompt 示例
```
难度: 困难 (hard)

位置策略：
- 创建复杂的空间约束，测试高级能力
- 可以考虑以下策略：

1. 多重阻挡：
   - 一个障碍物挡直线路径
   - 另一个障碍物挡主要绕行路径
   - 只留一条窄通道（11cm）

2. 包围目标：
   - 在目标周围放置 3-4 个障碍物
   - 只留一个进入缝隙
   - 需要精确对准

3. 旋转挑战：
   - 分析 T-block 当前朝向
   - 放置障碍物使当前朝向无法通过
   - 迫使先旋转再推动

4. 多步规划：
   - 障碍物位置使得无法直接推到目标
   - 需要先推到中间位置调整
   - 再从另一个角度推到目标

约束：
- 通道宽度可以窄至 11cm（但必须可通过）
- 障碍物可以靠近起点/终点（最近 6cm）
- 必须确保任务可完成（存在解）
```

---

## 4. 位置策略的量化指标

### 如何衡量一个配置的难度？

```python
def compute_difficulty_score(obstacle_config, tblock_pose, target_pose):
    """
    计算障碍物配置的难度分数
    """
    score = 0
    
    # 1. 路径阻挡程度 (0-30分)
    direct_path_blocked = check_path_blocked(obstacle_config, tblock_pose, target_pose)
    if direct_path_blocked:
        score += 30
    
    # 2. 最小通道宽度 (0-25分)
    min_passage = compute_min_passage_width(obstacle_config)
    if min_passage < 0.11:
        score += 25
    elif min_passage < 0.13:
        score += 15
    elif min_passage < 0.15:
        score += 5
    
    # 3. 绕行路径复杂度 (0-20分)
    detour_complexity = compute_detour_complexity(obstacle_config)
    score += detour_complexity * 20
    
    # 4. 是否需要旋转 (0-15分)
    rotation_required = check_rotation_required(obstacle_config, tblock_pose)
    if rotation_required:
        score += 15
    
    # 5. 障碍物协同约束 (0-10分)
    coordination_score = compute_coordination_score(obstacle_config)
    score += coordination_score * 10
    
    return score  # 总分 0-100

# 难度分级
# 0-30:  Easy
# 31-60: Medium
# 61-100: Hard
```

---

## 5. 与 LLM 的集成

### 如何让 LLM 理解位置策略？

在 prompt 中明确说明：

```python
def build_prompt_with_strategy(tblock_pose, difficulty_config):
    strategy = difficulty_config.placement_strategy
    
    prompt = f"""
## 当前难度: {difficulty_config.level}

## 位置策略要求:
{strategy['description']}

## 具体约束:
- 策略类型: {strategy['strategy_type']}
- 是否阻挡直线路径: {'是' if strategy['block_direct_path'] else '否'}
- 最小通道宽度: {strategy['min_passage_width']}m
- 距离起点最小距离: {strategy['min_distance_to_start']}m
- 距离终点最小距离: {strategy['min_distance_to_target']}m

## 设计目标:
根据上述策略，为当前 T-block 位置 {tblock_pose} 设计障碍物配置。
确保配置符合策略要求，并能有效测试相应的技能。
"""
    return prompt
```

---

## 6. 总结

| 难度 | 核心策略 | 关键指标 | 测试能力 |
|------|---------|---------|---------|
| Easy | 外围干扰 | 不阻挡路径，通道 >15cm | 基本避障意识 |
| Medium | 路径阻挡 | 阻挡直线，通道 12-13cm | 路径规划、绕行 |
| Hard | 复杂约束 | 多重阻挡，通道 11cm | 精确控制、旋转、多步规划 |

**关键洞察**：
- 1 个障碍物在路径中点 > 3 个障碍物在角落
- 位置的战略性 > 数量的堆砌
- LLM 的语义理解能力可以生成更有意义的配置
