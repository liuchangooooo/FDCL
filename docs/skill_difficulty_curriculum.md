# 技能树 × 难度树：双维度课程学习系统

## 核心思想

```
泛化性 = 技能多样性 × 难度渐进性

技能树：确保覆盖所有技能（横向扩展）
难度树：确保每个技能逐步掌握（纵向深入）
课程学习：根据反馈动态调整
```

---

## 1. 双维度设计

### 1.1 技能维度（横向）

```
技能类型（Skill Types）:
├─ S1: 直线推动 (Straight Push)
├─ S2: 绕行 (Detour)
│   ├─ S2.1: 左绕行
│   └─ S2.2: 右绕行
├─ S3: 旋转 (Rotation)
│   ├─ S3.1: 原地旋转
│   └─ S3.2: 边推边转
├─ S4: 精确控制 (Precise Control)
├─ S5: 路径规划 (Path Planning)
│   ├─ S5.1: 单步规划
│   └─ S5.2: 多步规划
└─ S6: 组合技能 (Combined Skills)
```

### 1.2 难度维度（纵向）

```
每个技能都有 3-5 个难度级别：

以 S2 (绕行) 为例：
├─ D1: 简单绕行（障碍物远离路径，绕行空间充足）
├─ D2: 中等绕行（障碍物在路径上，需要规划绕行）
├─ D3: 困难绕行（多个障碍物，绕行空间有限）
└─ D4: 极限绕行（密集障碍物，需要精确绕行）

以 S3 (旋转) 为例：
├─ D1: 简单旋转（空旷空间，旋转角度小 <45°）
├─ D2: 中等旋转（适中空间，旋转角度中 45-90°）
├─ D3: 困难旋转（狭窄空间，旋转角度大 >90°）
└─ D4: 极限旋转（极窄空间，需要多次调整）
```

---

## 2. 技能-难度矩阵

### 2.1 矩阵结构

```
        │ D1(简单) │ D2(中等) │ D3(困难) │ D4(极限) │
────────┼──────────┼──────────┼──────────┼──────────┤
S1:直线 │   ✓      │    ✓     │    ✓     │    ✓     │
S2:绕行 │   ✓      │    ✓     │    ✓     │    ✓     │
S3:旋转 │   ✓      │    ✓     │    ✓     │    ✓     │
S4:精确 │   ✓      │    ✓     │    ✓     │    ✓     │
S5:规划 │   ✓      │    ✓     │    ✓     │    ✓     │
S6:组合 │   -      │    ✓     │    ✓     │    ✓     │

每个格子 = 一类场景
总共约 20-25 类场景
```

### 2.2 训练路径

```
阶段 1: 横向扩展（技能覆盖）
┌─────────────────────────────────────┐
│ S1-D1 → S2-D1 → S3-D1 → S4-D1 → S5-D1 │
└─────────────────────────────────────┘
目标：在简单难度下，学会所有基础技能

阶段 2: 纵向深入（难度提升）
┌─────────────────────────────────────┐
│ S1-D2, S2-D2, S3-D2, S4-D2, S5-D2   │
└─────────────────────────────────────┘
目标：在中等难度下，巩固所有技能

阶段 3: 交叉训练（技能+难度）
┌─────────────────────────────────────┐
│ 随机采样 (Si-Dj)，确保覆盖所有格子  │
└─────────────────────────────────────┘
目标：全面掌握，提升泛化性

阶段 4: 组合挑战（高难度+多技能）
┌─────────────────────────────────────┐
│ S6-D2, S6-D3, S6-D4                 │
└─────────────────────────────────────┘
目标：应对复杂场景
```

---

## 3. 具体场景设计

### 3.1 S1: 直线推动

#### S1-D1: 简单直线
```yaml
场景描述: 无障碍物或障碍物在外围
障碍物数量: 0-1
空间约束: 无
技能要求: 基本推动
成功标准: 成功率 > 90%
```

#### S1-D2: 中等直线
```yaml
场景描述: 路径旁有障碍物，但不阻挡
障碍物数量: 1-2
空间约束: 障碍物距离路径 5-8cm
技能要求: 直线推动 + 避障意识
成功标准: 成功率 > 85%
```

#### S1-D3: 困难直线
```yaml
场景描述: 路径两侧有障碍物，形成通道
障碍物数量: 2-3
空间约束: 通道宽度 12-15cm
技能要求: 精确直线推动
成功标准: 成功率 > 75%
```

#### S1-D4: 极限直线
```yaml
场景描述: 极窄通道
障碍物数量: 2-4
空间约束: 通道宽度 10.5-11cm
技能要求: 极精确推动
成功标准: 成功率 > 60%
```

---

### 3.2 S2: 绕行

#### S2-D1: 简单绕行
```yaml
场景描述: 1个障碍物在路径中点，绕行空间充足
障碍物数量: 1
空间约束: 绕行空间 > 15cm
技能要求: 基本路径规划
成功标准: 成功率 > 85%

示例配置:
  T-block: (0.15, 0.15)
  障碍物: (0.075, 0.075)
  绕行方案: 左侧或右侧都有充足空间
```

#### S2-D2: 中等绕行
```yaml
场景描述: 1-2个障碍物，绕行空间适中
障碍物数量: 1-2
空间约束: 绕行空间 12-15cm
技能要求: 路径规划 + 精确控制
成功标准: 成功率 > 75%

示例配置:
  障碍物 1: 阻挡直线路径
  障碍物 2: 限制一侧绕行空间
  绕行方案: 需要选择合适的绕行方向
```

#### S2-D3: 困难绕行
```yaml
场景描述: 多个障碍物，需要之字形绕行
障碍物数量: 2-3
空间约束: 绕行空间 11-12cm
技能要求: 复杂路径规划 + 连续精确控制
成功标准: 成功率 > 65%

示例配置:
  障碍物形成之字形通道
  需要多次转向
```

#### S2-D4: 极限绕行
```yaml
场景描述: 密集障碍物，极窄绕行空间
障碍物数量: 3-4
空间约束: 绕行空间 10.5-11cm
技能要求: 高级规划 + 极精确控制
成功标准: 成功率 > 55%
```

---

### 3.3 S3: 旋转

#### S3-D1: 简单旋转
```yaml
场景描述: 空旷空间，小角度旋转
旋转角度: < 45°
空间约束: 充足空间（半径 > 15cm）
技能要求: 基本旋转控制
成功标准: 成功率 > 85%
```

#### S3-D2: 中等旋转
```yaml
场景描述: 适中空间，中等角度旋转
旋转角度: 45-90°
空间约束: 适中空间（半径 12-15cm）
技能要求: 旋转控制 + 空间感知
成功标准: 成功率 > 75%

示例场景:
  需要旋转才能通过窄门
  门宽 9cm，横杠 10cm，竖杠 7cm
```

#### S3-D3: 困难旋转
```yaml
场景描述: 狭窄空间，大角度旋转
旋转角度: 90-180°
空间约束: 狭窄空间（半径 11-12cm）
技能要求: 精确旋转 + 避障
成功标准: 成功率 > 65%

示例场景:
  在狭窄通道中旋转
  需要多次微调
```

#### S3-D4: 极限旋转
```yaml
场景描述: 极窄空间，多次旋转
旋转角度: > 180° 或需要多次调整
空间约束: 极窄空间（半径 10.5-11cm）
技能要求: 极精确旋转 + 多步规划
成功标准: 成功率 > 55%
```

---

### 3.4 S6: 组合技能

#### S6-D2: 中等组合
```yaml
场景描述: 绕行 + 旋转
障碍物数量: 2-3
技能要求: 
  - 先绕行到合适位置
  - 再旋转到目标朝向
成功标准: 成功率 > 70%
```

#### S6-D3: 困难组合
```yaml
场景描述: 绕行 + 旋转 + 精确控制
障碍物数量: 3-4
技能要求:
  - 绕行通过障碍区
  - 在狭窄空间旋转
  - 精确推到目标
成功标准: 成功率 > 60%
```

#### S6-D4: 极限组合
```yaml
场景描述: 多步规划 + 所有技能
障碍物数量: 4-5
技能要求:
  - 全局路径规划
  - 多次旋转调整
  - 精确通过多个关卡
成功标准: 成功率 > 50%
```

---

## 4. 课程学习策略

### 4.1 自适应采样

```python
class SkillDifficultyCurriculum:
    """
    技能-难度双维度课程学习
    """
    
    def __init__(self):
        # 技能-难度矩阵
        self.skill_types = ["S1", "S2", "S3", "S4", "S5", "S6"]
        self.difficulty_levels = ["D1", "D2", "D3", "D4"]
        
        # 每个格子的掌握程度
        self.mastery = {
            (skill, diff): 0.0 
            for skill in self.skill_types 
            for diff in self.difficulty_levels
        }
        
        # 当前训练阶段
        self.stage = "horizontal"  # horizontal, vertical, mixed, advanced
        
        # 历史统计
        self.history = {
            (skill, diff): {
                "attempts": 0,
                "successes": 0,
                "avg_steps": [],
                "collisions": 0
            }
            for skill in self.skill_types 
            for diff in self.difficulty_levels
        }
    
    def sample_scenario(self):
        """
        根据当前阶段和掌握程度，采样场景
        """
        if self.stage == "horizontal":
            # 阶段 1: 横向扩展（简单难度，覆盖所有技能）
            return self._sample_horizontal()
        
        elif self.stage == "vertical":
            # 阶段 2: 纵向深入（提升每个技能的难度）
            return self._sample_vertical()
        
        elif self.stage == "mixed":
            # 阶段 3: 混合训练（随机采样，确保覆盖）
            return self._sample_mixed()
        
        else:  # advanced
            # 阶段 4: 高级训练（组合技能，高难度）
            return self._sample_advanced()
    
    def _sample_horizontal(self):
        """
        横向采样：在 D1 难度下，遍历所有技能
        """
        # 找出掌握程度最低的技能
        skill_mastery = {
            skill: self.mastery[(skill, "D1")]
            for skill in self.skill_types[:-1]  # 排除 S6（组合技能）
        }
        
        # 选择掌握程度最低的技能
        target_skill = min(skill_mastery, key=skill_mastery.get)
        
        return (target_skill, "D1")
    
    def _sample_vertical(self):
        """
        纵向采样：对每个技能，逐步提升难度
        """
        # 找出可以提升难度的技能
        candidates = []
        for skill in self.skill_types[:-1]:
            for i, diff in enumerate(self.difficulty_levels[:-1]):
                current_mastery = self.mastery[(skill, diff)]
                next_diff = self.difficulty_levels[i + 1]
                next_mastery = self.mastery[(skill, next_diff)]
                
                # 如果当前难度已掌握，但下一难度未掌握
                if current_mastery > 0.75 and next_mastery < 0.5:
                    candidates.append((skill, next_diff))
        
        if candidates:
            return random.choice(candidates)
        else:
            # 如果没有候选，随机选择
            return self._sample_mixed()
    
    def _sample_mixed(self):
        """
        混合采样：确保覆盖所有格子
        """
        # 计算每个格子的采样权重（掌握程度低的权重高）
        weights = {}
        for skill in self.skill_types[:-1]:
            for diff in self.difficulty_levels:
                mastery = self.mastery[(skill, diff)]
                attempts = self.history[(skill, diff)]["attempts"]
                
                # 权重 = (1 - 掌握程度) * (1 + 稀有度)
                rarity = 1.0 / (attempts + 1)
                weights[(skill, diff)] = (1 - mastery) * (1 + rarity)
        
        # 归一化权重
        total = sum(weights.values())
        probs = {k: v / total for k, v in weights.items()}
        
        # 采样
        scenarios = list(probs.keys())
        probabilities = list(probs.values())
        return random.choices(scenarios, weights=probabilities)[0]
    
    def _sample_advanced(self):
        """
        高级采样：组合技能 + 高难度
        """
        return ("S6", random.choice(["D2", "D3", "D4"]))
    
    def update(self, scenario, success, steps, collision):
        """
        更新统计信息和掌握程度
        """
        skill, diff = scenario
        
        # 更新历史
        self.history[(skill, diff)]["attempts"] += 1
        if success:
            self.history[(skill, diff)]["successes"] += 1
        self.history[(skill, diff)]["avg_steps"].append(steps)
        if collision:
            self.history[(skill, diff)]["collisions"] += 1
        
        # 计算掌握程度
        attempts = self.history[(skill, diff)]["attempts"]
        successes = self.history[(skill, diff)]["successes"]
        
        if attempts >= 20:  # 至少 20 次尝试
            success_rate = successes / attempts
            avg_steps = np.mean(self.history[(skill, diff)]["avg_steps"][-20:])
            collision_rate = self.history[(skill, diff)]["collisions"] / attempts
            
            # 综合评分
            mastery = (
                success_rate * 0.6 +
                (1 - collision_rate) * 0.2 +
                (1 - min(avg_steps / 200, 1.0)) * 0.2
            )
            
            self.mastery[(skill, diff)] = mastery
    
    def should_advance_stage(self):
        """
        检查是否应该进入下一阶段
        """
        if self.stage == "horizontal":
            # 所有技能在 D1 难度下都掌握了
            d1_mastery = [
                self.mastery[(skill, "D1")]
                for skill in self.skill_types[:-1]
            ]
            if all(m > 0.75 for m in d1_mastery):
                self.stage = "vertical"
                print("📈 进入阶段 2: 纵向深入")
                return True
        
        elif self.stage == "vertical":
            # 所有技能在 D2 难度下都掌握了
            d2_mastery = [
                self.mastery[(skill, "D2")]
                for skill in self.skill_types[:-1]
            ]
            if all(m > 0.70 for m in d2_mastery):
                self.stage = "mixed"
                print("📈 进入阶段 3: 混合训练")
                return True
        
        elif self.stage == "mixed":
            # 大部分格子都掌握了
            all_mastery = [
                self.mastery[(skill, diff)]
                for skill in self.skill_types[:-1]
                for diff in self.difficulty_levels
            ]
            if np.mean(all_mastery) > 0.65:
                self.stage = "advanced"
                print("📈 进入阶段 4: 高级训练")
                return True
        
        return False
    
    def get_coverage_report(self):
        """
        生成覆盖率报告
        """
        print("\n" + "="*60)
        print("技能-难度覆盖率报告")
        print("="*60)
        
        # 打印矩阵
        print(f"{'':8}", end="")
        for diff in self.difficulty_levels:
            print(f"{diff:10}", end="")
        print()
        
        for skill in self.skill_types:
            print(f"{skill:8}", end="")
            for diff in self.difficulty_levels:
                mastery = self.mastery[(skill, diff)]
                attempts = self.history[(skill, diff)]["attempts"]
                print(f"{mastery:.2f}({attempts:3d})", end=" ")
            print()
        
        print("="*60)
        
        # 统计
        total_cells = len(self.skill_types) * len(self.difficulty_levels)
        mastered_cells = sum(
            1 for m in self.mastery.values() if m > 0.7
        )
        print(f"掌握格子数: {mastered_cells}/{total_cells}")
        print(f"平均掌握程度: {np.mean(list(self.mastery.values())):.2%}")
        print(f"当前阶段: {self.stage}")
        print("="*60 + "\n")
```

---

## 5. 反馈机制

### 5.1 实时反馈

```python
def provide_feedback(scenario, result):
    """
    根据执行结果提供反馈
    """
    skill, diff = scenario
    
    if result["success"]:
        if result["steps"] < 100:
            feedback = "优秀！高效完成"
        elif result["steps"] < 150:
            feedback = "良好！成功完成"
        else:
            feedback = "完成，但效率可提升"
    else:
        if result["collision"]:
            feedback = "失败：发生碰撞，需要提升精确控制"
        elif result["timeout"]:
            feedback = "失败：超时，需要提升规划效率"
        else:
            feedback = "失败：未达到目标"
    
    print(f"[{skill}-{diff}] {feedback}")
    
    # 根据反馈调整采样权重
    if not result["success"]:
        # 失败的场景，增加采样权重
        increase_sampling_weight(scenario)
```

### 5.2 周期性评估

```python
def periodic_evaluation(curriculum, interval=100):
    """
    每 N 个 episode 进行一次全面评估
    """
    if episode % interval == 0:
        # 1. 生成覆盖率报告
        curriculum.get_coverage_report()
        
        # 2. 识别薄弱环节
        weak_spots = curriculum.find_weak_spots()
        print(f"薄弱环节: {weak_spots}")
        
        # 3. 检查是否应该进入下一阶段
        if curriculum.should_advance_stage():
            print("阶段提升！")
        
        # 4. 可视化
        plot_mastery_heatmap(curriculum.mastery)
        plot_skill_progress(curriculum.history)
```

---

## 6. 总结

### 核心设计

```
双维度课程学习 = 技能树（横向） × 难度树（纵向）

技能树：确保覆盖所有技能类型
难度树：确保每个技能逐步掌握
课程学习：根据反馈自适应调整
```

### 训练流程

```
阶段 1 (Horizontal): 
  在简单难度下，学会所有基础技能
  S1-D1, S2-D1, S3-D1, S4-D1, S5-D1

阶段 2 (Vertical):
  对每个技能，逐步提升难度
  S1-D2, S2-D2, S3-D2, ...

阶段 3 (Mixed):
  混合训练，确保全面覆盖
  随机采样 (Si-Dj)

阶段 4 (Advanced):
  组合技能 + 高难度
  S6-D2, S6-D3, S6-D4
```

### 预期效果

```
- 技能覆盖率: 100%（所有技能都见过）
- 难度覆盖率: 100%（每个技能的所有难度都见过）
- 泛化能力: 强（见过各种组合）
- 鲁棒性: 高（在各种难度下都能应对）
```

这样既保证了多样性（技能树），又保证了渐进性（难度树），还有反馈机制（课程学习）！
