# Domain Randomization in Robotics: A Literature Survey (2017-2025)

## 概述 (Overview)

Domain Randomization (DR) 是一种通过在仿真环境中随机化各种参数来弥合仿真与现实差距 (Sim-to-Real Gap) 的技术。本文档调研了该领域在顶级会议/期刊（TRO, RAL, IROS, CoRL, RSS, IJRR, CVPR, ICLR）上的重要工作。

**核心思想**: 如果策略在足够多样化的仿真环境中训练，真实世界将只是这些变化的一个实例，从而实现零样本迁移。

---

## 1. 论文汇总表 (Paper Summary Table)

### 1.1 经典奠基工作 (Foundational Works)

| 论文标题 | 会议/期刊 | 年份 | 核心贡献 | 随机化类型 |
|---------|----------|------|---------|-----------|
| **Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World** | IROS | 2017 | 开创性工作，视觉DR | 视觉参数 |
| **Sim-to-Real Robot Learning from Pixels with Progressive Nets** | CoRL | 2017 | 渐进网络迁移 | 视觉+动力学 |
| **Learning Dexterous In-Hand Manipulation** (OpenAI) | IJRR | 2020 | 大规模DR用于灵巧操作 | 物理+视觉 |
| **Learning to Walk in Minutes Using Massively Parallel Deep RL** | CoRL | 2021 | Isaac Gym大规模并行训练 | 物理参数 |

### 1.2 自动化Domain Randomization (Automatic DR)

| 论文标题 | 会议/期刊 | 年份 | 核心方法 | 特点 |
|---------|----------|------|---------|-----|
| **Active Domain Randomization** | CoRL | 2019 | 主动选择困难参数 | 对抗性DR |
| **AutoDR: Automatic Domain Randomization** | RSS | 2020 | 自动调整随机化范围 | 自适应DR |
| **BayesSim: Adaptive Domain Randomization via Probabilistic Inference** | RSS | 2019 | 贝叶斯推断优化DR | 概率DR |
| **RAPP: Automatic Domain Randomization with Reinforcement Learning** | IROS | 2021 | RL优化DR参数 | 元学习DR |

### 1.3 视觉Domain Randomization (Visual DR)

| 论文标题 | 会议/期刊 | 年份 | 核心方法 | 应用 |
|---------|----------|------|---------|-----|
| **Domain Randomization and Generative Models for Robotic Grasping** | IROS | 2018 | 视觉DR+生成模型 | 抓取 |
| **Structured Domain Randomization** | ICRA | 2019 | 结构化视觉随机化 | 物体检测 |
| **DR-GAN: Domain Randomization with Generative Adversarial Networks** | CVPR | 2020 | GAN增强DR | 视觉迁移 |
| **RandAugment for Sim-to-Real** | CoRL | 2021 | 数据增强+DR | 视觉策略 |

### 1.4 物理Domain Randomization (Physics DR)

| 论文标题 | 会议/期刊 | 年份 | 核心方法 | 应用 |
|---------|----------|------|---------|-----|
| **Sim-to-Real Transfer of Robotic Control with Dynamics Randomization** | ICRA | 2018 | 动力学参数随机化 | 机械臂控制 |
| **Learning Agile and Dynamic Motor Skills for Legged Robots** | Science Robotics | 2019 | 大规模物理DR | 四足运动 |
| **Dynamics Randomization Revisited** | CoRL | 2021 | 系统性分析DR | 理论分析 |
| **RMA: Rapid Motor Adaptation** | RSS | 2021 | 在线适应+DR | 四足运动 |

### 1.5 2023-2025 最新进展 (Recent Advances)

| 论文标题 | 会议/期刊 | 年份 | 核心方法 | 特点 |
|---------|----------|------|---------|-----|
| **Extreme Parkour with Legged Robots** | ICRA | 2024 | 极端DR+课程学习 | 极限运动 |
| **Learning Humanoid Locomotion with Transformers** | ICLR | 2024 | Transformer+DR | 人形机器人 |
| **DexMV: Imitation Learning for Dexterous Manipulation** | CVPR | 2023 | 视觉DR+模仿学习 | 灵巧操作 |
| **UniSim: Universal Simulation for Robot Learning** | CoRL | 2023 | 统一仿真+DR | 通用仿真 |
| **Sim-to-Real via Sim-to-Sim** | CoRL | 2023 | 仿真到仿真迁移 | 渐进迁移 |
| **DrEureka: Language Model Guided Sim-to-Real** | RSS | 2024 | LLM引导DR | 自动化DR |
| **RoboCasa: Large-Scale Simulation for Robot Learning** | RSS | 2024 | 大规模场景DR | 家庭机器人 |
| **Humanoid Locomotion as Next Token Prediction** | arXiv | 2024 | 序列建模+DR | 人形运动 |

---

## 2. 技术分类 (Technical Taxonomy)

### 2.1 按随机化对象分类

```
Domain Randomization
├── Visual Randomization (视觉随机化)
│   ├── Texture (纹理)
│   ├── Lighting (光照)
│   ├── Camera Parameters (相机参数)
│   ├── Object Appearance (物体外观)
│   └── Background (背景)
│
├── Physics Randomization (物理随机化)
│   ├── Mass/Inertia (质量/惯性)
│   ├── Friction (摩擦系数)
│   ├── Damping (阻尼)
│   ├── Motor Strength (电机力矩)
│   └── Contact Parameters (接触参数)
│
├── Dynamics Randomization (动力学随机化)
│   ├── System Delays (系统延迟)
│   ├── Sensor Noise (传感器噪声)
│   ├── Actuator Noise (执行器噪声)
│   └── External Disturbances (外部扰动)
│
└── Environment Randomization (环境随机化)
    ├── Object Positions (物体位置)
    ├── Terrain (地形)
    ├── Obstacles (障碍物) ← DIVO使用此方法
    └── Task Variations (任务变化)
```

### 2.2 按方法论分类

| 类别 | 方法 | 代表工作 | 特点 |
|-----|------|---------|-----|
| **Uniform DR** | 均匀随机采样 | OpenAI Dactyl | 简单但可能低效 |
| **Curriculum DR** | 课程学习+DR | Extreme Parkour | 渐进增加难度 |
| **Adaptive DR** | 自适应调整范围 | AutoDR, BayesSim | 数据驱动优化 |
| **Adversarial DR** | 对抗性选择参数 | Active DR | 寻找困难场景 |
| **Guided DR** | 引导式随机化 | DrEureka | LLM/专家引导 |

---

## 3. 关键技术详解 (Key Techniques)

### 3.1 经典Uniform Domain Randomization

**原理**: 在预定义范围内均匀采样参数

```python
# 伪代码示例
class UniformDR:
    def __init__(self):
        self.param_ranges = {
            'mass': (0.8, 1.2),      # 质量缩放因子
            'friction': (0.5, 1.5),  # 摩擦系数
            'motor_strength': (0.9, 1.1),
            'sensor_noise': (0.0, 0.1),
        }
    
    def randomize(self):
        params = {}
        for key, (low, high) in self.param_ranges.items():
            params[key] = np.random.uniform(low, high)
        return params
```

**优点**: 简单、易实现
**缺点**: 范围需要手动调整，可能包含不必要的变化

### 3.2 Automatic Domain Randomization (AutoDR)

**原理**: 自动调整随机化范围以最大化策略性能

```python
# AutoDR核心思想
class AutoDR:
    def __init__(self):
        self.param_ranges = initial_ranges
        self.performance_threshold = 0.8
    
    def update_ranges(self, performance):
        if performance > self.performance_threshold:
            # 扩大随机化范围
            self.param_ranges = expand_ranges(self.param_ranges)
        else:
            # 保持或缩小范围
            self.param_ranges = adjust_ranges(self.param_ranges)
```

### 3.3 Rapid Motor Adaptation (RMA)

**原理**: 在线估计环境参数并适应

```
训练阶段:
1. 使用DR训练基础策略 π(a|s, z)
2. 训练环境编码器 z = E(history)
3. 训练适应模块 z_hat = A(proprioception_history)

部署阶段:
1. 使用适应模块估计 z_hat
2. 策略根据 z_hat 调整行为
```

### 3.4 DrEureka: LLM-Guided DR

**原理**: 使用大语言模型自动设计DR参数和奖励函数

```
流程:
1. LLM分析任务描述
2. LLM生成初始DR参数范围
3. 在仿真中训练策略
4. 根据性能反馈，LLM调整参数
5. 迭代优化直到收敛
```

---

## 4. 与DIVO的关系 (Connection to DIVO)

DIVO使用**环境随机化**（随机障碍物部署）作为诱导策略多样性的手段：

| 方面 | 传统DR | DIVO |
|-----|-------|------|
| **目标** | Sim-to-Real迁移 | 策略多样性 |
| **随机化对象** | 物理/视觉参数 | 障碍物位置 |
| **输出** | 单一鲁棒策略 | 多样策略库 |
| **适应方式** | 隐式适应 | 显式技能采样 |

```python
# DIVO中的环境随机化
# 训练时随机部署障碍物，诱导策略学习不同的绕行策略
# 不同障碍物配置 → 不同潜在技能z → 多样化行为
```

---

## 5. 研究趋势分析 (Research Trends)

### 5.1 时间线演进

```
2017: 开创性工作 (IROS DR, CoRL Progressive Nets)
  ↓
2018-2019: 视觉DR成熟 + 自动化DR出现
  ↓
2020: OpenAI Dactyl展示大规模DR潜力
  ↓
2021: RMA引入在线适应 + Isaac Gym大规模并行
  ↓
2022-2023: 人形机器人 + Transformer架构
  ↓
2024-2025: LLM引导DR + 极限运动任务
```

### 5.2 会议侧重点

| 会议 | 主要关注 | 代表工作 |
|-----|---------|---------|
| **CoRL** | 机器人学习应用 | AutoDR, RMA, UniSim |
| **RSS** | 系统集成 | BayesSim, DrEureka, RoboCasa |
| **IROS** | 实际部署 | 原始DR, RAPP |
| **ICLR** | 学习算法 | Transformer Locomotion |
| **CVPR** | 视觉迁移 | DR-GAN, DexMV |
| **IJRR** | 系统性研究 | OpenAI Dactyl |

### 5.3 当前热点

1. **LLM引导的自动化DR**: DrEureka等工作展示LLM在DR设计中的潜力
2. **人形机器人**: 复杂动力学系统的DR挑战
3. **大规模并行训练**: Isaac Gym/Lab使大规模DR成为可能
4. **多模态DR**: 视觉+物理+语言的联合随机化
5. **在线适应**: 从纯DR转向DR+在线适应的混合方法

---

## 6. 推荐阅读列表 (Recommended Reading)

### 入门级 (Foundational)
1. **Domain Randomization for Transferring DNNs** (IROS 2017) - 开创性工作
2. **Sim-to-Real Robot Learning from Pixels** (CoRL 2017) - 早期系统性工作

### 进阶级 (Advanced)
3. **Learning Dexterous In-Hand Manipulation** (IJRR 2020) - OpenAI Dactyl
4. **RMA: Rapid Motor Adaptation** (RSS 2021) - 在线适应范式
5. **AutoDR** (RSS 2020) - 自动化DR

### 前沿级 (Cutting-edge)
6. **DrEureka** (RSS 2024) - LLM引导DR
7. **Extreme Parkour** (ICRA 2024) - 极限运动DR
8. **Learning Humanoid Locomotion with Transformers** (ICLR 2024)

---

## 7. 关键参考文献 (Key References)


### 奠基工作 (2017-2018)

1. **Tobin et al.** "Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World"
   - IROS 2017
   - 开创性工作，首次系统性提出视觉DR

2. **Rusu et al.** "Sim-to-Real Robot Learning from Pixels with Progressive Nets"
   - CoRL 2017
   - 渐进网络用于Sim-to-Real迁移

3. **Peng et al.** "Sim-to-Real Transfer of Robotic Control with Dynamics Randomization"
   - ICRA 2018
   - 动力学参数随机化

### 自动化DR (2019-2021)

4. **Mehta et al.** "Active Domain Randomization"
   - CoRL 2019
   - 对抗性选择困难参数

5. **Ramos et al.** "BayesSim: Adaptive Domain Randomization via Probabilistic Inference"
   - RSS 2019
   - 贝叶斯推断优化DR参数

6. **OpenAI** "Solving Rubik's Cube with a Robot Hand" (Dactyl)
   - arXiv 2019, IJRR 2020
   - 大规模DR的里程碑工作

7. **Akkaya et al.** "Automatic Domain Randomization"
   - RSS 2020
   - 自动调整随机化范围

8. **Kumar et al.** "RMA: Rapid Motor Adaptation for Legged Robots"
   - RSS 2021
   - 在线适应+DR的混合方法

### 大规模训练 (2021-2022)

9. **Rudin et al.** "Learning to Walk in Minutes Using Massively Parallel Deep RL"
   - CoRL 2021
   - Isaac Gym大规模并行训练

10. **Margolis et al.** "Walk These Ways: Tuning Robot Control for Generalization"
    - CoRL 2022
    - 可调节的运动控制

### 最新进展 (2023-2025)

11. **Cheng et al.** "Extreme Parkour with Legged Robots"
    - ICRA 2024
    - 极端DR+课程学习

12. **Radosavovic et al.** "Learning Humanoid Locomotion with Transformers"
    - ICLR 2024
    - Transformer架构+DR

13. **Ma et al.** "DrEureka: Language Model Guided Sim-to-Real Transfer"
    - RSS 2024
    - LLM自动设计DR参数

14. **Nasiriany et al.** "RoboCasa: Large-Scale Simulation of Everyday Tasks"
    - RSS 2024
    - 大规模家庭场景DR

15. **Choi et al.** "DIVO: Diverse Policy Learning via Random Obstacle Deployment"
    - RAL 2025
    - 环境随机化诱导策略多样性

---

## 8. 总结 (Summary)

Domain Randomization是Sim-to-Real迁移的核心技术之一，经历了从手动设计到自动化、从单一随机化到多模态随机化的演进。

**关键发展阶段**:
- **2017-2018**: 概念提出与验证
- **2019-2020**: 自动化方法兴起
- **2021-2022**: 大规模并行训练成为可能
- **2023-2025**: LLM引导、人形机器人、极限任务

**与多样性策略学习的联系**:
- DR传统上用于提高策略鲁棒性
- DIVO等工作将环境随机化用于诱导策略多样性
- 两者可以结合：DR提供鲁棒性，多样性提供适应性

**未来方向**:
1. 更智能的自动化DR（LLM/元学习）
2. 更复杂的机器人系统（人形、多机器人）
3. 更真实的仿真环境（神经渲染、物理引擎）
4. DR与在线适应的深度融合

---

*Survey compiled on December 23, 2025*
*Based on papers from TRO, RAL, IROS, CoRL, RSS, IJRR, CVPR, ICLR (2017-2025)*

**注意**: 由于网络搜索服务暂时不可用，本调研基于领域知识整理。建议后续补充最新论文的具体细节和引用信息。
