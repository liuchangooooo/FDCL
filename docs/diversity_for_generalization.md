# 基于泛化性的障碍物场景设计

## 核心目标

**不是设计"难"的场景，而是设计"多样化"的场景**

目标：训练出在未知环境中具有强泛化性的策略

---

## 1. 泛化性需要什么？

### 1.1 技能覆盖的多样性

策略需要学会：
```
1. 基本推动（直线、曲线）
2. 避障（单个、多个、密集）
3. 绕行（左绕、右绕、大弯、小弯）
4. 旋转（原地旋转、边推边转）
5. 精确控制（窄通道、精确对齐）
6. 路径规划（短路径、长路径、多步）
7. 姿态调整（不同角度的推动）
8. 恢复能力（碰撞后调整）
```

### 1.2 场景覆盖的多样性

需要见过：
```
- 不同数量的障碍物（0, 1, 2, 3, 4, 5+）
- 不同位置的障碍物（路径上、路径旁、目标周围、起点周围）
- 不同密度的障碍物（稀疏、适中、密集）
- 不同配置的障碍物（对称、不对称、随机、结构化）
- 不同初始位置（近、远、不同象限）
- 不同初始朝向（0°, 45°, 90°, 135°, 180°, ...）
```

### 1.3 约束覆盖的多样性

需要应对：
```
- 无约束（空旷环境）
- 轻微约束（外围障碍物）
- 路径约束（需要绕行）
- 空间约束（狭窄通道）
- 姿态约束（需要旋转）
- 顺序约束（需要规划）
- 组合约束（多种约束同时存在）
```

---

## 2. 多样性 vs 难度

### 2.1 错误的思路

```
Level 1: 简单场景（1个障碍物）
Level 2: 中等场景（2个障碍物）
Level 3: 困难场景（3个障碍物）
...
Level 7: 极难场景（6个障碍物 + 复杂约束）

问题：
- 只是线性增加难度
- 没有增加多样性
- 策略只学会"应对更多障碍物"
- 但没有学会"应对不同类型的约束"
```

### 2.2 正确的思路

```
每个难度级别内部，都要有多样化的场景

Level 1: 基础多样性
- 场景 A: 1个障碍物在左侧
- 场景 B: 1个障碍物在右侧
- 场景 C: 1个障碍物在路径中点
- 场景 D: 1个障碍物在目标附近
- 场景 E: 2个障碍物在两侧（对称）
→ 学会基本的避障和绕行

Level 2: 路径多样性
- 场景 A: 需要左绕行
- 场景 B: 需要右绕行
- 场景 C: 需要大弯绕行
- 场景 D: 需要之字形绕行
- 场景 E: 有多条可选路径
→ 学会不同的路径规划策略

Level 3: 姿态多样性
- 场景 A: 需要顺时针旋转
- 场景 B: 需要逆时针旋转
- 场景 C: 需要旋转 + 推动
- 场景 D: 需要多次旋转
- 场景 E: 不同初始朝向
→ 学会姿态控制

...
```

---

## 3. 多样性的维度

### 3.1 空间维度

```yaml
障碍物位置的多样性:
  - 象限分布: [第一象限, 第二象限, 第三象限, 第四象限]
  - 距离分布: [近距离 <0.1m, 中距离 0.1-0.15m, 远距离 >0.15m]
  - 相对位置: [起点附近, 路径中点, 目标附近, 路径两侧]
  
障碍物配置的多样性:
  - 对称性: [完全对称, 部分对称, 完全不对称]
  - 密度: [稀疏, 适中, 密集]
  - 结构: [线性排列, 圆形排列, 随机分布, 聚集分布]
```

### 3.2 任务维度

```yaml
初始状态的多样性:
  - T-block 位置: 均匀采样整个工作空间
  - T-block 朝向: [0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°]
  - 到目标距离: [近 <0.15m, 中 0.15-0.20m, 远 >0.20m]
  - 需要旋转角度: [小 <45°, 中 45°-90°, 大 >90°]

路径特征的多样性:
  - 路径长度: [短, 中, 长]
  - 路径曲率: [直线, 小弯, 大弯, 多弯]
  - 路径选择: [唯一路径, 多条路径]
```

### 3.3 约束维度

```yaml
约束类型的多样性:
  - 无约束: 空旷环境，测试基本推动
  - 单一约束: 
    - 路径阻挡（需要绕行）
    - 空间限制（需要精确控制）
    - 姿态限制（需要旋转）
  - 双重约束:
    - 路径阻挡 + 空间限制
    - 路径阻挡 + 姿态限制
    - 空间限制 + 姿态限制
  - 多重约束:
    - 路径 + 空间 + 姿态
    - 顺序 + 精确 + 旋转

约束强度的多样性:
  - 弱约束: 容易绕过
  - 中等约束: 需要规划
  - 强约束: 需要精确执行
```

---

## 4. 基于多样性的难度设计

### Level 1: 基础技能 + 空间多样性

```yaml
目标: 学会基本推动和简单避障

场景多样性:
  - 障碍物数量: 0-2个
  - 障碍物位置: 
    - 左侧、右侧、前方、后方
    - 近、中、远
  - 初始朝向: 随机
  - 路径类型: 直线、小弯

技能覆盖:
  - 直线推动
  - 简单绕行（左/右）
  - 基本避障意识

多样性指标:
  - 位置熵: 高（障碍物位置随机）
  - 配置熵: 中（1-2个障碍物的组合）
  - 约束熵: 低（主要是位置约束）
```

### Level 2: 路径规划 + 配置多样性

```yaml
目标: 学会不同的路径规划策略

场景多样性:
  - 障碍物数量: 1-3个
  - 障碍物配置:
    - 线性排列（需要绕大弯）
    - 分散分布（多条路径可选）
    - 聚集分布（需要避开密集区）
  - 路径类型: 直线、大弯、之字形

技能覆盖:
  - 路径规划（选择最优路径）
  - 大弯绕行
  - 多步推动

多样性指标:
  - 位置熵: 高
  - 配置熵: 高（多种配置方式）
  - 约束熵: 中（路径约束为主）
```

### Level 3: 精确控制 + 姿态多样性

```yaml
目标: 学会精确控制和姿态调整

场景多样性:
  - 障碍物数量: 2-4个
  - 障碍物配置:
    - 形成通道（宽度 12-15cm）
    - 需要旋转的配置
    - 需要精确对齐的配置
  - 初始朝向: 8个方向均匀采样
  - 路径类型: 窄通道、需要旋转

技能覆盖:
  - 精确路径跟踪
  - 旋转控制
  - 姿态调整

多样性指标:
  - 位置熵: 高
  - 配置熵: 高
  - 约束熵: 高（空间 + 姿态约束）
```

### Level 4-7: 组合约束 + 全面多样性

```yaml
目标: 学会应对复杂的组合约束

场景多样性:
  - 障碍物数量: 2-6个
  - 障碍物配置: 所有类型的组合
  - 约束类型: 随机组合
  - 初始状态: 全空间采样

技能覆盖:
  - 所有技能的组合
  - 自适应策略选择
  - 错误恢复

多样性指标:
  - 位置熵: 最高
  - 配置熵: 最高
  - 约束熵: 最高
```

---

## 5. 多样性的量化指标

### 5.1 位置熵（Position Entropy）

```python
def compute_position_entropy(obstacle_configs):
    """
    计算障碍物位置的多样性
    """
    # 将工作空间划分为网格
    grid = create_grid(workspace, cell_size=0.05)
    
    # 统计每个网格的障碍物频率
    frequencies = count_obstacles_per_cell(obstacle_configs, grid)
    
    # 计算熵
    entropy = -sum(p * log(p) for p in frequencies if p > 0)
    
    return entropy

# 高熵 = 障碍物位置分布均匀
# 低熵 = 障碍物总是在相同位置
```

### 5.2 配置熵（Configuration Entropy）

```python
def compute_configuration_entropy(obstacle_configs):
    """
    计算障碍物配置的多样性
    """
    # 提取配置特征
    features = []
    for config in obstacle_configs:
        feature = {
            "num_obstacles": len(config),
            "symmetry": compute_symmetry(config),
            "density": compute_density(config),
            "structure": classify_structure(config)
        }
        features.append(feature)
    
    # 计算特征空间的熵
    entropy = compute_feature_entropy(features)
    
    return entropy

# 高熵 = 配置方式多样
# 低熵 = 配置方式单一
```

### 5.3 技能覆盖率（Skill Coverage）

```python
def compute_skill_coverage(episodes):
    """
    计算训练过程中技能的覆盖率
    """
    required_skills = [
        "straight_push",
        "left_detour",
        "right_detour",
        "rotation",
        "precise_control",
        "multi_step_planning"
    ]
    
    # 分析每个 episode 使用了哪些技能
    skills_used = set()
    for episode in episodes:
        skills = analyze_trajectory(episode)
        skills_used.update(skills)
    
    coverage = len(skills_used) / len(required_skills)
    
    return coverage

# 高覆盖率 = 见过各种技能场景
# 低覆盖率 = 只见过少数几种场景
```

---

## 6. LLM 生成的多样性策略

### 6.1 Prompt 设计

```python
# 错误的 Prompt（导致单一场景）
prompt = """
生成一个中等难度的障碍物配置。
"""

# 正确的 Prompt（鼓励多样性）
prompt = f"""
生成一个障碍物配置，要求：

1. 场景类型: {random.choice(["路径阻挡", "狭窄通道", "包围目标", "需要旋转"])}
2. 障碍物数量: {random.randint(2, 4)}
3. 空间特征: {random.choice(["对称分布", "不对称分布", "聚集分布", "分散分布"])}
4. 约束类型: {random.choice(["路径约束", "空间约束", "姿态约束", "组合约束"])}

当前 T-block 位置: {tblock_pose}
当前 T-block 朝向: {tblock_angle}°

请设计一个与之前配置不同的新场景，确保多样性。
"""
```

### 6.2 多样性采样策略

```python
class DiversityAwareSampler:
    """
    多样性感知的场景采样器
    """
    
    def __init__(self):
        self.history = []  # 历史配置
        self.feature_counts = {}  # 特征计数
    
    def sample_scenario_params(self):
        """
        采样场景参数，确保多样性
        """
        # 1. 统计历史特征分布
        feature_dist = self.compute_feature_distribution()
        
        # 2. 选择出现频率低的特征
        rare_features = self.find_rare_features(feature_dist)
        
        # 3. 优先采样稀有特征
        scenario_type = self.sample_from_rare(
            rare_features["scenario_type"]
        )
        spatial_config = self.sample_from_rare(
            rare_features["spatial_config"]
        )
        constraint_type = self.sample_from_rare(
            rare_features["constraint_type"]
        )
        
        return {
            "scenario_type": scenario_type,
            "spatial_config": spatial_config,
            "constraint_type": constraint_type
        }
    
    def record_scenario(self, config):
        """记录生成的场景"""
        self.history.append(config)
        self.update_feature_counts(config)
```

---

## 7. 训练策略

### 7.1 多样性驱动的课程学习

```python
class DiversityCurriculum:
    """
    基于多样性的课程学习
    """
    
    def __init__(self):
        self.current_level = 1
        self.diversity_targets = {
            1: {"position_entropy": 2.0, "skill_coverage": 0.3},
            2: {"position_entropy": 2.5, "skill_coverage": 0.5},
            3: {"position_entropy": 3.0, "skill_coverage": 0.7},
            4: {"position_entropy": 3.5, "skill_coverage": 0.9}
        }
    
    def should_upgrade(self, stats):
        """
        升级条件：不仅看成功率，还看多样性
        """
        target = self.diversity_targets[self.current_level]
        
        # 条件 1: 成功率达标
        success_ok = stats["success_rate"] > 0.75
        
        # 条件 2: 多样性达标
        diversity_ok = (
            stats["position_entropy"] > target["position_entropy"] and
            stats["skill_coverage"] > target["skill_coverage"]
        )
        
        return success_ok and diversity_ok
```

### 7.2 多样性监控

```python
def monitor_diversity(training_history):
    """
    监控训练过程中的多样性
    """
    print("多样性报告:")
    print(f"位置熵: {compute_position_entropy(training_history):.2f}")
    print(f"配置熵: {compute_configuration_entropy(training_history):.2f}")
    print(f"技能覆盖率: {compute_skill_coverage(training_history):.2%}")
    
    # 可视化
    plot_position_heatmap(training_history)
    plot_skill_coverage(training_history)
    plot_configuration_distribution(training_history)
```

---

## 8. 总结

### 核心原则

```
泛化性 = 多样性 × 技能覆盖

不是：设计极难的场景
而是：设计多样化的场景，覆盖各种可能的情况
```

### 设计要点

1. **空间多样性**：障碍物在不同位置、不同配置
2. **任务多样性**：不同初始状态、不同路径特征
3. **约束多样性**：不同类型、不同强度的约束
4. **技能多样性**：覆盖所有需要的技能

### 评估指标

1. **位置熵**：障碍物位置分布的均匀性
2. **配置熵**：障碍物配置方式的多样性
3. **技能覆盖率**：训练中见过的技能类型比例
4. **成功率**：仍然需要保证学习效果

### 预期效果

```
训练后的策略能够：
- 应对各种障碍物配置
- 在未见过的环境中泛化
- 自适应选择合适的技能
- 鲁棒地完成任务
```

这才是真正有意义的难度设计！
