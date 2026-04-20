# 多样性策略学习与安全强化学习：近五年发展综述 (2020-2025)

## 概述

本文档从**多样性**和**安全性**两个维度，系统梳理近五年（2020-2025）机器人策略学习领域的方法演进。

---

## 一、多样性策略学习发展脉络

### 1.1 技术演进时间线

```
2018-2019          2020-2021           2022-2023           2024-2025
    │                  │                   │                   │
    ▼                  ▼                   ▼                   ▼
┌─────────┐      ┌──────────┐       ┌──────────┐       ┌──────────┐
│ DIAYN   │      │ QD-RL    │       │ Diffusion│       │ G-QDIL   │
│ DADS    │ ──▶  │ MAP-Elites│ ──▶  │ Policy   │ ──▶  │ DIVO     │
│         │      │ +Deep RL │       │ Flow     │       │ FLOWER   │
└─────────┘      └──────────┘       │ Matching │       └──────────┘
 互信息技能发现    QD+深度学习融合     生成模型主导        QD+生成模型融合
```

---

### 1.2 方法分类与代表工作

#### A. 互信息技能发现 (2018-2021)

| 方法 | 年份 | 会议 | 核心思想 | 论文链接 |
|-----|-----|-----|---------|---------|
| **DIAYN** | 2019 | ICLR | max I(s;z) - I(a;z\|s)，最大化状态-技能互信息 | [arXiv](https://arxiv.org/abs/1802.06070) |
| **DADS** | 2020 | ICLR | 同时学习技能和动力学模型，支持连续技能空间 | [arXiv](https://arxiv.org/abs/1907.01657) |
| **ComSD** | 2023 | NeurIPS | 对比学习估计MI，平衡质量与多样性 | [arXiv](https://arxiv.org/abs/2309.17203) |
| **LSD** | 2022 | ICLR | Lipschitz约束技能发现，支持零样本目标跟随 | [arXiv](https://arxiv.org/abs/2202.00914) |

**特点**：无监督、信息论驱动、技能可解释性强
**局限**：技能数量有限、难以扩展到复杂任务

---

#### B. Quality-Diversity进化方法 (2015-2024)

| 方法 | 年份 | 会议 | 核心思想 | 论文链接 |
|-----|-----|-----|---------|---------|
| **MAP-Elites** | 2015 | - | 行为空间网格化，每格保留最优解 | [arXiv](https://arxiv.org/abs/1504.04909) |
| **QD-RL** | 2022 | ICLR | QD + Off-policy RL，样本高效 | [OpenReview](https://openreview.net/forum?id=8FRw857AYba) |
| **PGA-MAP-Elites** | 2021 | GECCO | 策略梯度 + MAP-Elites | [arXiv](https://arxiv.org/abs/2006.08505) |
| **Real-World QD** | 2025 | RSS | 物理机器人无重置QD学习 | [PMLR](https://proceedings.mlr.press/v305/grillotti25a.html) |

**特点**：同时优化质量+多样性、维护策略库
**局限**：计算开销大、行为描述符设计困难

---

#### C. 生成模型方法 (2023-2025)

| 方法 | 年份 | 会议 | 核心思想 | 论文链接 |
|-----|-----|-----|---------|---------|
| **Diffusion Policy** | 2023 | RSS | DDPM生成动作序列，建模多模态分布 | [Project](https://diffusion-policy.cs.columbia.edu/) |
| **ManiFlow** | 2024 | CoRL | Flow Matching用于操作任务 | [OpenReview](https://openreview.net/forum?id=vtEn8NJWlz) |
| **DIVO** | 2025 | RAL | Flow Matching采样潜在技能 | [IEEE](https://ieeexplore.ieee.org/document/10847909) |
| **FLOWER** | 2025 | ICLR | Flow + 进化算法 | [ICLR](https://iclr.cc/virtual/2025/) |
| **G-QDIL** | 2025 | ICLR | QD + 生成模型模仿学习 | [ICLR](https://www.iclr.cc/virtual/2025/32382) |

**特点**：天然建模多模态、高维动作空间友好
**趋势**：2024-2025主流方向

---

#### D. 混合专家方法 (2022-2024)

| 方法 | 年份 | 会议 | 核心思想 | 论文链接 |
|-----|-----|-----|---------|---------|
| **Di-SkilL** | 2024 | ICML | MoE + 课程学习，每个专家学不同技能 | [ICML](https://icml.cc/virtual/2024/poster/34802) |
| **QMP** | 2025 | ICLR | Q-switch策略混合，多任务行为共享 | [CLVRAI](https://clvrai.com/publications) |

**特点**：结构化多样性、专家可解释
**局限**：专家数量预设、路由机制设计

---

#### E. 环境诱导方法 (2020-2025)

| 方法 | 年份 | 会议 | 核心思想 | 论文链接 |
|-----|-----|-----|---------|---------|
| **Domain Randomization** | 2017+ | - | 物理参数随机化 | [arXiv](https://arxiv.org/abs/1703.06907) |
| **QED** | 2020 | - | Quality-Environment-Diversity | [ResearchGate](https://www.researchgate.net/publication/346744761) |
| **DIVO** | 2025 | RAL | 随机障碍物部署诱导策略多样性 | [IEEE](https://ieeexplore.ieee.org/document/10847909) |

**特点**：简单有效、无需额外目标函数
**优势**：多样性自然涌现、泛化能力强

---

### 1.3 多样性方法演进总结

```
┌─────────────────────────────────────────────────────────────────┐
│                    多样性诱导方式演进                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  2018-2020: 目标函数驱动                                        │
│  ├── 互信息最大化 (DIAYN, DADS)                                 │
│  └── 进化算法多样性压力 (MAP-Elites)                            │
│                                                                 │
│  2021-2023: 结构+目标混合                                       │
│  ├── QD + 深度RL融合 (QD-RL, PGA-MAP-Elites)                   │
│  └── 混合专家结构 (Di-SkilL)                                    │
│                                                                 │
│  2024-2025: 生成模型主导                                        │
│  ├── 扩散/流模型建模多模态 (Diffusion Policy, ManiFlow)         │
│  ├── 环境诱导+生成采样 (DIVO)                                   │
│  └── QD + 生成模型 (G-QDIL, FLOWER)                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 二、安全强化学习发展脉络

### 2.1 技术演进时间线

```
2017-2019          2020-2021           2022-2023           2024-2025
    │                  │                   │                   │
    ▼                  ▼                   ▼                   ▼
┌─────────┐      ┌──────────┐       ┌──────────┐       ┌──────────┐
│ CPO     │      │ Recovery │       │ CBF+RL   │       │ Safe RL  │
│ Lagrangian│──▶ │ RL       │ ──▶  │ Neural   │ ──▶  │ +Prediction│
│ Methods │      │ Backup   │       │ CBF      │       │ DIVO     │
└─────────┘      └──────────┘       └──────────┘       └──────────┘
 约束优化方法      恢复策略方法        学习安全函数        预测+过滤方法
```

---

### 2.2 方法分类与代表工作

#### A. 约束优化方法 (2017-2023)

| 方法 | 年份 | 会议 | 核心思想 | 论文链接 |
|-----|-----|-----|---------|---------|
| **CPO** | 2017 | ICML | 约束策略优化，近似满足约束 | [arXiv](https://arxiv.org/abs/1705.10528) |
| **PCPO** | 2020 | ICLR | 投影约束策略优化 | [arXiv](https://arxiv.org/abs/2010.03152) |
| **CVPO** | 2022 | AAAI | 变分约束策略优化 | [arXiv](https://arxiv.org/abs/2201.11927) |
| **SCPO** | 2023 | NeurIPS | 状态级约束策略优化 | [arXiv](https://arxiv.org/abs/2306.12594) |
| **CCPO** | 2024 | NeurIPS | 约束条件化策略优化 | [OpenReview](https://openreview.net/forum?id=yeOmel9X5K) |

**特点**：训练时保证约束、理论保证
**局限**：累积约束、难以处理瞬时安全

---

#### B. 恢复策略方法 (2020-2024)

| 方法 | 年份 | 会议 | 核心思想 | 论文链接 |
|-----|-----|-----|---------|---------|
| **Recovery RL** | 2021 | RAL | 学习恢复区域，危险时切换恢复策略 | [arXiv](https://arxiv.org/abs/2010.15920) |
| **Safe RL (Kiemel)** | 2024 | RAL | Backup Policy + N步rollout风险评估 | [arXiv](https://arxiv.org/abs/2411.05784) |
| **TU-Recovery** | 2023 | - | 三阶段架构：安全评估→恢复→任务 | [arXiv](https://arxiv.org/abs/2309.11907) |
| **RbSL** | 2024 | - | 离线恢复策略学习 | [arXiv](https://arxiv.org/abs/2403.01734) |

**特点**：部署时保证安全、任务策略与安全策略分离
**局限**：恢复策略可能中断任务

---

#### C. 控制屏障函数方法 (2019-2024)

| 方法 | 年份 | 会议 | 核心思想 | 论文链接 |
|-----|-----|-----|---------|---------|
| **CBF-QP** | 2019+ | - | CBF约束二次规划修正动作 | [IEEE](https://ieeexplore.ieee.org/document/8796030) |
| **Neural CBF** | 2021 | CoRL | 神经网络学习CBF | [arXiv](https://arxiv.org/abs/2109.06697) |
| **NeRF-CBF** | 2023 | ICRA | NeRF + CBF视觉安全控制 | [arXiv](https://arxiv.org/abs/2209.12266) |
| **Lyapunov-Barrier** | 2021 | - | 联合稳定性+安全性 | [arXiv](https://arxiv.org/abs/2109.06697) |

**特点**：理论保证强、可证明安全
**局限**：需要系统模型、CBF设计困难

---

#### D. 预测+过滤方法 (2022-2025)

| 方法 | 年份 | 会议 | 核心思想 | 论文链接 |
|-----|-----|-----|---------|---------|
| **DIVO** | 2025 | RAL | TCN预测N步轨迹 + 约束过滤 | [IEEE](https://ieeexplore.ieee.org/document/10847909) |
| **Safe RL (Kiemel)** | 2024 | RAL | Backup rollout + Risk Network | [arXiv](https://arxiv.org/abs/2411.05784) |
| **Predictive Safety** | 2019+ | - | 预测未来状态判断安全 | [arXiv](https://arxiv.org/abs/1910.00399) |

**特点**：无需理论证明、工程实现简单
**优势**：适用于复杂动态环境

---

### 2.3 安全方法演进总结

```
┌─────────────────────────────────────────────────────────────────┐
│                    安全保障方式演进                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  训练时安全 (Training-time Safety)                              │
│  ├── 约束优化: CPO, PCPO, CVPO (累积约束)                       │
│  ├── 安全奖励塑形: 惩罚不安全行为                               │
│  └── 安全探索: 限制探索范围                                     │
│                                                                 │
│  部署时安全 (Deployment-time Safety)                            │
│  ├── 动作修正: CBF-QP最小修正                                   │
│  ├── 策略切换: Recovery RL, Backup Policy                       │
│  ├── 预测过滤: DIVO (采样多个→预测→过滤)                        │
│  └── 风险评估: Risk Network阈值判断                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 三、多样性 + 安全性 结合方式

### 3.1 现有结合方式对比

| 结合方式 | 代表方法 | 多样性来源 | 安全机制 | 特点 |
|---------|---------|-----------|---------|-----|
| **采样+过滤** | DIVO | Flow Matching | TCN预测+约束检查 | 简单有效 |
| **策略切换** | Safe RL | 单一Task Policy | Backup Policy | 可能中断任务 |
| **约束训练** | CPO+多样性 | 约束下多样性 | 训练时约束 | 理论保证 |
| **无显式安全** | Diffusion Policy | 扩散模型 | 依赖数据 | 无安全保证 |

---

### 3.2 DIVO的独特贡献

DIVO是目前**唯一**同时解决多样性和安全性的完整框架：

```
┌─────────────────────────────────────────────────────────────────┐
│  DIVO: 多样性 + 安全性 统一框架                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  多样性模块:                                                    │
│  ├── 训练: 随机障碍物部署 → 自然诱导多样策略                    │
│  └── 部署: Flow Matching采样M个技能z                            │
│                                                                 │
│  安全性模块:                                                    │
│  ├── 预测: TCN预测每个技能对应的N步轨迹                         │
│  └── 过滤: 约束检查 c_new(s_traj) ≤ 0，保留安全动作             │
│                                                                 │
│  关键: 从多样策略库中"选择"安全的，而非"替换"为安全策略          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 四、未来研究方向

### 4.1 多样性方向
1. **条件生成**：直接生成满足约束的多样动作
2. **层次化技能**：多层次技能组合
3. **在线适应**：部署时动态扩展技能库

### 4.2 安全性方向
1. **学习安全函数**：从数据学习CBF/安全集
2. **不确定性感知**：考虑感知和模型不确定性
3. **多智能体安全**：协作场景下的安全保证

### 4.3 结合方向
1. **Safety-aware生成**：生成模型直接建模安全约束
2. **安全引导采样**：Classifier-free guidance引导安全方向
3. **层次化安全架构**：高层多样性选择 + 底层安全保障

---

## 五、参考文献

### 多样性方法

1. Eysenbach et al., "Diversity is All You Need (DIAYN)", ICLR 2019
2. Sharma et al., "Dynamics-Aware Unsupervised Discovery of Skills (DADS)", ICLR 2020
3. Mouret & Clune, "Illuminating search spaces by mapping elites (MAP-Elites)", 2015
4. Nilsson & Cully, "MAP-Elites enables Powerful Stepping Stones", Frontiers 2021
5. Cideron et al., "QD-RL: Efficient Mixing of Quality and Diversity in RL", ICLR 2022
6. Chi et al., "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion", RSS 2023
7. Choi et al., "DIVO: Diverse Policy Learning via Random Obstacle Deployment", RAL 2025
8. "G-QDIL: Generative Quality Diversity Imitation Learning", ICLR 2025
9. Chalumeau et al., "Di-SkilL: Acquiring Diverse Skills using Curriculum RL with MoE", ICML 2024
10. "Iteratively Learn Diverse Strategies with State Distance Information", NeurIPS 2023

### 安全方法

11. Achiam et al., "Constrained Policy Optimization (CPO)", ICML 2017
12. Thananjeyan et al., "Recovery RL: Safe RL with Learned Recovery Zones", RAL 2021
13. Kiemel et al., "Safe RL of Robot Trajectories in the Presence of Moving Obstacles", RAL 2024
14. Chow et al., "Safe Policy Learning for Continuous Control", CoRL 2021
15. Ames et al., "Control Barrier Functions: Theory and Applications", 2019
16. Cheng et al., "End-to-End Safe RL through Barrier Functions", AAAI 2019
17. Brunke et al., "Safe Learning in Robotics: From Learning-Based Control to Safe RL", Annual Review 2022
18. "State-wise Constrained Policy Optimization (SCPO)", NeurIPS 2023
19. "CCPO: Constraint-Conditioned Policy Optimization", NeurIPS 2024
20. "Learning to Recover for Safe Reinforcement Learning", 2023

---

*Survey compiled on December 23, 2025*
