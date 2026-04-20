# LLM 障碍物生成系统 - 完整设计文档

## 概述

本系统实现了基于 LLM 的障碍物环境生成，结合质量评价和课程学习，用于 Push-T 任务的强化学习训练。

**核心理念**: 让 LLM 在约束条件下自由创造障碍物配置，通过质量评价系统筛选，而不是预定义场景类型。

## 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    LLM 障碍物生成系统                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │         1. LLM 障碍物生成器              │
        │   (llm_obstacle_generator_v3.py)        │
        │                                         │
        │  - 初始生成 (generate)                   │
        │  - 进化生成 (evolve)                     │
        │  - Eurekaverse 风格 prompt              │
        └─────────────────┬───────────────────────┘
                          │
                          ▼
        ┌─────────────────────────────────────────┐
        │         2. 质量评价器                     │
        │   (obstacle_quality_evaluator.py)       │
        │                                         │
        │  评价维度:                               │
        │  ├─ 可解性 (0/1)                        │
        │  ├─ 难度 (0-1)                          │
        │  ├─ 多样性 (0-1)                        │
        │  └─ 有效性 (0-1)                        │
        └─────────────────┬───────────────────────┘
                          │
                          ▼
        ┌─────────────────────────────────────────┐
        │         3. 训练环境                       │
        │   (pusht_mj_rod_llm.py)                 │
        │                                         │
        │  - set_obstacle_config()                │
        │  - 执行 Push-T 任务                      │
        │  - 收集性能指标                          │
        └─────────────────┬───────────────────────┘
                          │
                          ▼
        ┌─────────────────────────────────────────┐
        │         4. 课程学习管理器                 │
        │   (curriculum_manager.py)               │
        │                                         │
        │  - 记录 episode 结果                     │
        │  - 统计成功率/碰撞率/步数                 │
        │  - 难度升级/降级决策                      │
        └─────────────────┬───────────────────────┘
                          │
                          ▼
        ┌─────────────────────────────────────────┐
        │         5. 训练管理器                     │
        │   (train_with_llm_quality.py)           │
        │                                         │
        │  - 整合所有组件                          │
        │  - 完整训练循环                          │
        │  - 反馈机制                              │
        └─────────────────────────────────────────┘
```

## 核心组件详解

### 1. LLM 障碍物生成器 (llm_obstacle_generator_v3.py)

**功能**: 使用 LLM 生成障碍物配置

**两种模式**:

#### 1.1 初始生成 (generate)
```python
config = generator.generate(
    tblock_pose=[0.15, 0.15, np.pi/4],
    scenario_type="auto",
    num_obstacles=2,
    difficulty="medium"
)
```

**输入**:
- `tblock_pose`: T-block 初始位姿 [x, y, θ]
- `scenario_type`: 场景类型提示（"auto", "path_blocking", "corridor", "rotation", "surrounding"）
- `num_obstacles`: 障碍物数量
- `difficulty`: 难度等级（"easy", "medium", "hard"）

**输出**:
```python
[
    {"x": 0.07, "y": 0.07, "purpose": "阻挡直线路径"},
    {"x": 0.04, "y": 0.10, "purpose": "迫使从上方绕行"}
]
```

#### 1.2 进化生成 (evolve)
```python
config = generator.evolve(
    tblock_pose=[0.15, 0.15, np.pi/4],
    previous_config=[...],
    policy_stats={
        "success_rate": 0.85,
        "avg_steps": 45,
        "collision_rate": 0.05,
        "avg_reward": -1.2
    }
)
```

**特点**:
- 根据策略表现调整难度
- 成功率高 → 增加难度
- 成功率低 → 降低难度
- 保持多样性

### 2. 质量评价器 (obstacle_quality_evaluator.py)

**功能**: 评价生成的障碍物配置质量

#### 2.1 可解性评价 (Solvability) - 必须满足

**检查项**:
1. ✓ 初始位置是否与障碍物碰撞
2. ✓ 是否被完全包围（检查8个方向）
3. ✓ 障碍物数量是否合理（≤6个）
4. ✓ 基本连通性（是否存在路径）

**不检查**:
- ✗ 目标位置是否被阻挡（LLM 已确保）
- ✗ 目标位置旋转空间（旋转在推动过程中完成）

**返回**: (True/False, 原因)

#### 2.2 难度评价 (Difficulty) - 0到1分数

**维度**:
1. **路径复杂度**: 绕行距离 vs 直线距离
2. **空间约束**: 最小间隙大小
3. **旋转难度**: 需要旋转角度 × 旋转空间限制
4. **障碍物密度**: 障碍物数量 / 工作空间面积

**计算**: 四个维度的平均值

#### 2.3 多样性评价 (Diversity) - 0到1分数

**特征提取**:
- 障碍物数量
- 空间分布（质心、方差）
- 对称性
- 聚集度

**计算**: 1 - max_similarity（与历史配置的最大相似度）

#### 2.4 有效性评价 (Effectiveness) - 0到1分数

**检查项**:
1. 障碍物是否距离路径太远（>15cm）
2. 障碍物是否重叠（<3cm）
3. 是否形成有意义的约束

**计算**: 从1.0开始，每个问题扣分

#### 2.5 综合评分

```python
overall_score = (
    0.30 * solvability +      # 可解性（0或1）
    0.25 * difficulty +       # 难度
    0.25 * diversity +        # 多样性
    0.20 * effectiveness      # 有效性
)
```

**阈值**: 通常设置为 0.5

### 3. 训练环境 (pusht_mj_rod_llm.py)

**新增接口**:

```python
# 设置障碍物配置
env.set_obstacle_config([
    {'x': 0.1, 'y': 0.1},
    {'x': -0.1, 'y': -0.1}
])

# 重置环境（应用配置）
obs = env.reset()

# 获取当前障碍物位置
positions = env.get_obstacle_positions()

# 清除配置（恢复随机模式）
env.clear_obstacle_config()
```

**正确的初始化流程**:
1. 设置目标 T 位置（固定在原点）
2. 采样初始 T-block 位置
3. 根据 T-block 位置生成/应用障碍物配置

### 4. 课程学习管理器 (curriculum_manager.py)

**功能**: 管理难度升级/降级

**难度等级**:
- `easy`: 外围干扰
- `medium`: 路径阻挡
- `hard`: 复杂约束

**升级/降级规则**:
```python
if success_rate > 0.8:
    upgrade()  # 升级难度
elif success_rate < 0.2:
    downgrade()  # 降级难度
else:
    maintain()  # 保持难度
```

**统计窗口**: 最近 N 个 episode（默认20）

### 5. 训练管理器 (train_with_llm_quality.py)

**完整训练流程**:

```python
manager = LLMTrainingManager(
    env=env,
    llm_generator=generator,
    quality_threshold=0.5,
    max_regenerate_attempts=3
)

for episode in range(num_episodes):
    # 1. 运行 episode
    info = manager.run_episode(policy=policy)
    
    # 2. 检查并更新难度
    manager.check_and_update_difficulty(check_interval=20)
```

**内部流程**:
```
1. 重置环境，获取 T-block 位置
2. LLM 生成障碍物配置
3. 质量评价
4. 如果不达标，重新生成（最多3次）
5. 应用配置到环境
6. 执行训练
7. 收集反馈（成功率、步数、碰撞率）
8. 记录到课程学习管理器
9. 每N个episode检查并更新难度
10. 每M个episode使用进化模式生成
```

## 使用指南

### 快速开始

```python
import os
from DIVO.env.pusht.mujoco.pusht_mj_rod_llm import PushT_mj_rod_LLM
from DIVO.env.pusht.llm_obstacle_generator_v3 import LLMObstacleGeneratorV3
from DIVO.env.pusht.train_with_llm_quality import LLMTrainingManager

# 1. 初始化环境
env = PushT_mj_rod_LLM(
    obstacle=True,
    obstacle_num=2,
    obstacle_size=0.01
)

# 2. 初始化 LLM 生成器
generator = LLMObstacleGeneratorV3(
    api_type="deepseek",
    api_key=os.getenv("DEEPSEEK_API_KEY")
)

# 3. 初始化训练管理器
manager = LLMTrainingManager(
    env=env,
    llm_generator=generator,
    quality_threshold=0.5
)

# 4. 运行训练
for episode in range(100):
    info = manager.run_episode(policy=your_policy)
    manager.check_and_update_difficulty(check_interval=20)
```

### 测试质量评价器

```bash
cd ~/DIVO
python -m DIVO.env.pusht.test_quality_evaluator
```

### 完整训练示例

```bash
cd ~/DIVO
python -m DIVO.env.pusht.train_with_llm_quality
```

## 设计原则

### 1. 人类 vs LLM 的职责划分

**人类负责**:
- 定义约束条件（坐标范围、碰撞检查）
- 设计评价体系（可解性、难度、多样性、有效性）
- 设置难度框架（easy/medium/hard）
- 定义升级/降级逻辑

**LLM 负责**:
- 在约束内生成具体配置
- 分析几何关系
- 创造性地设计障碍物布局
- 根据反馈调整生成策略

### 2. 为什么不预定义场景类型？

**问题**: 预定义场景类型（如"路径阻挡"、"狭窄通道"）会限制 LLM 的创造力

**解决方案**: 
- 只提供场景类型作为"提示"（hint），不作为硬性规则
- 核心是评价生成结果的质量，而不是检查是否符合预定义类型
- LLM 可以创造出人类没想到的有效配置

### 3. 质量评价 vs 场景分类

**质量评价**（我们的方法）:
- ✓ 评价配置的客观属性（可解性、难度、多样性）
- ✓ 不限制 LLM 的创造空间
- ✓ 可以发现新的有效场景类型

**场景分类**（避免的方法）:
- ✗ 需要预定义所有场景类型
- ✗ 限制了 LLM 的创造力
- ✗ 难以覆盖所有可能的有效配置

### 4. 课程学习的作用

**目标**: 根据策略表现动态调整难度

**不是**: 简单地增加障碍物数量

**而是**: 
- 改变障碍物的位置策略
- 从外围干扰 → 路径阻挡 → 复杂约束
- 测试不同的技能（绕行、精确控制、旋转、多步规划）

### 5. 多样性的重要性

**为什么重要**: 训练目标是强泛化能力，需要见过各种场景

**如何实现**:
- 提取配置特征（数量、分布、对称性、聚集度）
- 计算与历史配置的相似度
- 多样性分数 = 1 - 最大相似度
- 低多样性配置会被拒绝或降低质量分数

## 关键设计决策

### 决策1: 可解性检查的范围

**不检查目标位置旋转空间**:
- 原因: T-block 的旋转是在推动过程中逐渐完成的
- 不需要在目标位置有足够空间一次性旋转到位

**不检查目标位置是否被阻挡**:
- 原因: LLM 的 prompt 已经明确要求避开目标位置
- 重复检查是冗余的

### 决策2: 质量阈值设置

**默认阈值: 0.5**

**原因**:
- 可解性必须为1（权重0.3），贡献0.3分
- 其他三个维度平均0.5分，贡献0.35分
- 总分约0.65，有一定容错空间

**可调整**:
- 训练初期可以降低阈值（0.4），接受更多配置
- 训练后期可以提高阈值（0.6），要求更高质量

### 决策3: 进化 vs 初始生成

**进化模式触发条件**:
- 每10个episode触发一次
- 需要有之前的配置和性能统计

**优势**:
- 根据实际训练效果调整
- 保持与策略能力的匹配
- 避免难度跳跃过大

### 决策4: 重新生成策略

**最多尝试3次**:
- 第1次: 初始生成
- 第2次: 如果质量不达标，重新生成
- 第3次: 最后一次机会
- 如果仍不达标，使用最后一次的结果

**原因**:
- 避免无限循环
- LLM 调用有成本
- 即使质量略低，也能提供训练价值

## 文件清单

### 核心文件

1. **llm_obstacle_generator_v3.py** (最新版)
   - LLM 障碍物生成器
   - 支持初始生成和进化
   - Eurekaverse 风格 prompt

2. **obstacle_quality_evaluator.py** (最新版)
   - 质量评价系统
   - 四维评价（可解性、难度、多样性、有效性）
   - 已移除冗余检查

3. **pusht_mj_rod_llm.py**
   - LLM 支持的 Push-T 环境
   - set_obstacle_config() 接口

4. **curriculum_manager.py**
   - 课程学习管理器
   - 难度升级/降级逻辑

5. **train_with_llm_quality.py** (最新版)
   - 完整训练管理器
   - 整合所有组件

### 测试文件

6. **test_quality_evaluator.py** (最新版)
   - 测试质量评价器
   - 多种测试场景

### 文档文件

7. **docs/llm_obstacle_generation_system.md** (本文件)
   - 完整系统文档

8. **docs/diversity_for_generalization.md**
   - 多样性设计理念

9. **docs/skill_difficulty_curriculum.md**
   - 技能树 × 难度树设计

## 下一步工作

### 1. 测试质量评价器 ✓
```bash
python -m DIVO.env.pusht.test_quality_evaluator
```

### 2. 测试 LLM 生成器
- 验证 LLM 能否生成合理配置
- 检查 prompt 是否清晰
- 测试不同难度等级

### 3. 集成测试
- 运行完整训练循环
- 验证质量评价 → 重新生成流程
- 检查课程学习是否正常工作

### 4. 实际训练
- 使用真实策略（而非随机策略）
- 收集训练数据
- 分析生成的配置质量

### 5. 优化调整
- 根据实际效果调整质量阈值
- 优化 LLM prompt
- 调整课程学习参数

## 常见问题

### Q1: 为什么不直接用随机生成？

**A**: 随机生成难以保证：
- 配置的有意义性（可能太简单或太难）
- 多样性（可能重复相似配置）
- 与策略能力的匹配（无法根据反馈调整）

### Q2: LLM 生成会不会太慢？

**A**: 
- 每个 episode 只调用一次 LLM
- 可以批量生成多个配置缓存
- 进化模式可以减少调用次数

### Q3: 质量评价会不会太严格？

**A**:
- 阈值可调（默认0.5）
- 最多重试3次，避免卡住
- 即使不达标也会使用最后的配置

### Q4: 如何确保训练的多样性？

**A**:
- 多样性是评价的一个维度（权重0.25）
- 记录历史配置，计算相似度
- 低多样性配置会被拒绝

### Q5: 课程学习会不会导致过拟合？

**A**:
- 难度升级对应不同场景类型，不是简单参数变化
- 多样性评价确保不重复相似配置
- 目标是泛化能力，不是特定场景的性能

## 总结

本系统实现了一个完整的 LLM 驱动的障碍物生成框架，核心特点：

1. **灵活性**: LLM 在约束内自由创造，不受预定义场景限制
2. **质量保证**: 四维评价体系确保配置质量
3. **自适应**: 课程学习根据策略表现动态调整难度
4. **多样性**: 避免重复配置，促进泛化
5. **可扩展**: 易于添加新的评价维度或调整策略

**设计哲学**: 人类定义框架和评价标准，LLM 负责创造性生成，质量评价系统确保输出质量。
