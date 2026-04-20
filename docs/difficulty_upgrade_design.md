# Push-T 环境难度升级设计

## 1. 设计原则

### 1.1 多维度量化
难度不是单一指标，而是多个维度的组合：

```
难度 = f(障碍物数量, 通道宽度, 场景类型, 初始位置)
```

### 1.2 渐进式升级
避免难度跳跃过大，使用"耐心机制"：

```
连续 N 次满足条件 → 才升级
```

### 1.3 双向调整
支持升级和降级，避免过难导致学习停滞。

---

## 2. 难度级别定义

### Easy（简单）
```yaml
障碍物数量: 1个
通道宽度: >0.15m (宽松)
场景类型: 
  - path_blocking: 70%  # 主要测试基本避障
  - corridor: 20%
  - rotation: 10%
  
升级条件:
  - 成功率 > 80%
  - 碰撞率 < 10%
  - 平均步数 < 150
  - 至少训练 50 episodes

目标: 让策略学会基本的避障和路径规划
```

### Medium（中等）
```yaml
障碍物数量: 2个
通道宽度: 0.12-0.15m (适中)
场景类型:
  - path_blocking: 40%
  - corridor: 30%
  - rotation: 20%
  - surrounding: 10%

升级条件:
  - 成功率 > 75%
  - 碰撞率 < 15%
  - 平均步数 < 180
  - 至少训练 100 episodes

降级条件:
  - 成功率 < 20%
  - 或 碰撞率 > 50%

目标: 测试路径规划 + 精确控制的组合能力
```

### Hard（困难）
```yaml
障碍物数量: 3个
通道宽度: 0.11-0.12m (接近 T-block 宽度 0.1m)
场景类型:
  - path_blocking: 20%
  - corridor: 20%
  - rotation: 30%
  - surrounding: 20%
  - auto: 10%  # LLM 自主决策

降级条件:
  - 成功率 < 25%
  - 或 碰撞率 > 45%

目标: 测试高级规划、精确操作、多步推理
```

---

## 3. 升级决策流程

```
┌─────────────────────────────────────────────────────────────┐
│                    难度升级决策流程                          │
└─────────────────────────────────────────────────────────────┘

每 N 个 Episode (例如 50)
         │
         ▼
┌─────────────────────┐
│  计算统计指标        │
│  - success_rate     │
│  - collision_rate   │
│  - avg_steps        │
└──────────┬──────────┘
           │
           ▼
    ┌──────────────┐
    │ 检查升级条件  │
    └──────┬───────┘
           │
     ┌─────┴─────┐
     │           │
     ▼           ▼
  满足条件    不满足
     │           │
     ▼           ▼
 upgrade_    重置计数器
 counter++       │
     │           │
     ▼           │
 counter ≥ 2?   │
     │           │
  是 │  否       │
     │   │       │
     ▼   └───────┴──► 保持当前难度
  升级难度
     │
     ▼
  重置计数器
  清空历史
```

---

## 4. 关键设计细节

### 4.1 为什么需要"耐心机制"？

```python
# 不好的设计：立即升级
if success_rate > 0.8:
    upgrade()  # 可能是偶然的好运气

# 好的设计：连续满足才升级
if success_rate > 0.8:
    counter += 1
    if counter >= 2:  # 连续 2 次
        upgrade()
```

**原因**：
- 避免因为运气好的几个 episode 就升级
- 确保策略真正掌握了当前难度
- 减少难度震荡

### 4.2 为什么需要降级？

```
场景 1: 升级过快
easy (成功率 85%) → medium (成功率 15%) ← 降级回 easy

场景 2: 学习停滞
hard (成功率 20%, 100 episodes 无进展) ← 降级到 medium
```

**原因**：
- 避免策略陷入"太难学不会"的困境
- 给策略更多时间巩固基础技能

### 4.3 多指标综合评估

```python
# 不好：只看成功率
if success_rate > 0.8:
    upgrade()

# 好：综合评估
if (success_rate > 0.8 and 
    collision_rate < 0.1 and 
    avg_steps < 150):
    upgrade()
```

**原因**：
- 成功率高但碰撞多 → 策略不够安全
- 成功率高但步数多 → 策略效率低
- 综合评估确保策略质量

---

## 5. 与 LLM 的集成

### 5.1 难度参数如何影响 LLM？

```python
# Easy 模式的 prompt
"""
难度: 简单 (easy)
- 障碍物远离直线路径，只做轻微干扰
- 绕行空间充足（> 0.15m）
- 主要测试基本避障意识
"""

# Hard 模式的 prompt
"""
难度: 困难 (hard)
- 障碍物形成复杂约束
- 绕行空间较窄（0.11-0.12m，接近 T-block 宽度）
- 可能需要多步操作或精确旋转
- 测试高级规划和精确操作能力
"""
```

### 5.2 场景类型的动态选择

```python
# Easy: 主要练习基本避障
scenario_weights = {
    "path_blocking": 0.7,  # 70% 概率
    "corridor": 0.2,
    "rotation": 0.1
}

# Hard: 多样化场景
scenario_weights = {
    "path_blocking": 0.2,
    "corridor": 0.2,
    "rotation": 0.3,
    "surrounding": 0.2,
    "auto": 0.1  # LLM 自主决策
}
```

---

## 6. 完整训练流程示例

```python
from DIVO.env.pusht.difficulty_design import DifficultyManager
from DIVO.env.pusht.llm_obstacle_generator_v3 import LLMObstacleGeneratorV3

# 初始化
difficulty_mgr = DifficultyManager(initial_level="easy")
llm_generator = LLMObstacleGeneratorV3()

for episode in range(1000):
    # 1. 获取当前难度配置
    config = difficulty_mgr.get_current_config()
    scenario = difficulty_mgr.get_scenario_type()
    
    # 2. Reset 环境
    obs = env.reset()
    tblock_pose = env.get_tblock_pose()
    
    # 3. LLM 生成障碍物（使用当前难度）
    obstacle_config = llm_generator.generate(
        tblock_pose=tblock_pose,
        scenario_type=scenario,
        num_obstacles=config.num_obstacles,
        difficulty=config.level
    )
    
    # 4. 应用障碍物
    env.set_obstacle_config(obstacle_config)
    
    # 5. 执行 episode
    success, steps, collision = run_episode(env, policy)
    
    # 6. 记录结果
    difficulty_mgr.record_episode(success, steps, collision)
    
    # 7. 定期评估并更新难度
    if (episode + 1) % 50 == 0:
        new_level, eval_result = difficulty_mgr.evaluate_and_update()
        print(f"Episode {episode+1}: {eval_result['action']}")
```

---

## 7. 预期训练曲线

```
成功率
  │
1.0│                                    ┌─────────── hard
  │                          ┌─────────┤
0.8│              ┌──────────┤ medium  │
  │    ┌─────────┤          │         │
0.6│────┤  easy   │          │         │
  │    │         │          │         │
0.4│    │         │          │         │
  │    │         │          │         │
0.2│    │         │          │         │
  │    │         │          │         │
0.0└────┴─────────┴──────────┴─────────┴──────────► Episode
    0   50       150        300       500

难度变化点:
- Episode 50-100: easy → medium (成功率稳定在 80%+)
- Episode 150-200: medium → hard (成功率稳定在 75%+)
- Episode 500+: 在 hard 难度下持续训练
```

---

## 8. 可调参数

```python
DifficultyManager(
    initial_level="easy",           # 起始难度
    evaluation_window=50,           # 统计窗口大小
    upgrade_patience=2,             # 连续满足几次才升级
    allow_downgrade=True            # 是否允许降级
)

# 自定义阈值
DIFFICULTY_CONFIGS["medium"].upgrade_thresholds = {
    "success_rate": 0.75,  # 可以调整
    "collision_rate": 0.15,
    "avg_steps": 180,
    "min_episodes": 100
}
```

---

## 9. 总结

这个设计的核心优势：

1. **多维度量化**：不只看成功率，综合评估策略质量
2. **渐进式升级**：避免难度跳跃，确保稳定学习
3. **双向调整**：支持降级，避免学习停滞
4. **与 LLM 深度集成**：难度参数直接影响 LLM 生成策略
5. **可解释性强**：每次升降级都有明确的数据支撑

这样的设计既保证了训练的稳定性，又充分利用了 LLM 的语义理解能力。
