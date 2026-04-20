# 具身智能安全控制：博士课题研究方向调研

> 本文档系统调研"具身智能安全控制"领域的研究现状、发展趋势和博士课题方向，提供最新论文支撑。

---

## 一、领域概述与定义

### 1.1 什么是具身智能安全

**具身智能安全 (Embodied AI Safety)** 研究如何让物理世界中的AI系统（机器人、自动驾驶、无人机等）在执行任务时保持安全。与纯软件AI不同，具身AI的错误可能导致物理伤害、财产损失甚至生命危险。

### 1.2 安全的多维度定义

| 维度 | 定义 | 示例场景 |
|-----|------|---------|
| **物理安全** | 避免碰撞、损坏、伤害 | 机械臂避障、人机协作安全距离 |
| **任务安全** | 避免执行危险操作 | 不执行"把刀递给婴儿"的指令 |
| **系统安全** | 抵抗攻击、故障容错 | 对抗样本攻击、传感器故障 |
| **伦理安全** | 符合人类价值观 | 不执行有害指令、隐私保护 |

### 1.3 核心挑战

1. **不确定性**: 真实环境的感知噪声、模型误差、人类行为不可预测
2. **泛化性**: 训练环境与部署环境的差异 (Sim-to-Real Gap)
3. **实时性**: 安全决策需要在毫秒级完成
4. **可解释性**: 安全决策需要可理解、可验证

---

## 二、研究方向分类与发展趋势

### 2.1 技术路线全景图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      具身智能安全控制技术路线                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐             │
│  │  控制理论方法    │  │  学习方法       │  │  大模型方法     │             │
│  ├─────────────────┤  ├─────────────────┤  ├─────────────────┤             │
│  │ • CBF/CLF       │  │ • Safe RL       │  │ • VLA安全对齐   │             │
│  │ • MPC安全约束   │  │ • World Model   │  │ • LLM任务规划   │             │
│  │ • 可达性分析    │  │ • 模仿学习+安全 │  │ • 对抗鲁棒性    │             │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘             │
│           │                    │                    │                       │
│           └────────────────────┼────────────────────┘                       │
│                                ▼                                            │
│                    ┌─────────────────────┐                                  │
│                    │   融合方法 (趋势)    │                                  │
│                    │ • 学习CBF           │                                  │
│                    │ • World Model+安全  │                                  │
│                    │ • LLM+形式化验证    │                                  │
│                    └─────────────────────┘                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 发展趋势时间线

```
2017-2019              2020-2022              2023-2024              2025+
    │                      │                      │                    │
    ▼                      ▼                      ▼                    ▼
┌─────────┐          ┌──────────┐          ┌──────────┐          ┌──────────┐
│ CPO     │          │ Recovery │          │ SafeDreamer│         │ VLA安全  │
│ 约束优化 │   ──▶   │ RL       │   ──▶   │ World Model│  ──▶   │ 对齐     │
│         │          │ Neural CBF│          │ 潜在安全  │          │ LLM规划  │
└─────────┘          └──────────┘          └──────────┘          └──────────┘
 约束RL起步           恢复策略+学习CBF       世界模型+安全          大模型安全
```

---

## 三、主要研究方向详解

### 方向1: 安全强化学习 (Safe RL)

#### 3.1.1 约束优化方法

**核心思想**: 将安全约束建模为CMDP (Constrained MDP)，在优化奖励的同时满足约束。

| 方法 | 年份/会议 | 核心思想 | 论文链接 |
|-----|---------|---------|---------|
| **CPO** | ICML 2017 | 约束策略优化，信赖域内满足约束 | [arXiv](https://arxiv.org/abs/1705.10528) |
| **PCPO** | ICLR 2020 | 投影约束策略优化 | [arXiv](https://arxiv.org/abs/2010.03152) |
| **SCPO** | NeurIPS 2023 | 状态级约束策略优化 | [arXiv](https://arxiv.org/abs/2306.12594) |
| **CCPO** | NeurIPS 2024 | 约束条件化策略优化 | [OpenReview](https://openreview.net/forum?id=yeOmel9X5K) |

**优势**: 理论保证、训练时安全
**局限**: 累积约束、难以处理瞬时安全

#### 3.1.2 恢复策略方法

**核心思想**: 学习一个恢复策略，在危险时切换到安全行为。

| 方法 | 年份/会议 | 核心思想 | 论文链接 |
|-----|---------|---------|---------|
| **Recovery RL** | RAL 2021 | 学习恢复区域，危险时切换恢复策略 | [arXiv](https://arxiv.org/abs/2010.15920) |
| **Safe RL (Kiemel)** | RAL 2024 | Backup Policy + N步rollout风险评估 | [arXiv](https://arxiv.org/abs/2411.05784) |
| **RbSL** | 2024 | 离线恢复策略学习 | [arXiv](https://arxiv.org/abs/2403.01734) |

**优势**: 部署时安全、任务与安全分离
**局限**: 恢复可能中断任务

#### 3.1.3 综述论文

| 综述 | 年份 | 内容 | 链接 |
|-----|------|------|------|
| **Safe RL Survey (IEEE TPAMI)** | 2024 | 方法、理论、应用全面综述 | [IEEE](https://ieeexplore.ieee.org/document/10675394/) |
| **Safe RL in Robotics** | 2025 | 机器人领域Safe RL方法 | [PDF](https://www.itm-conferences.org/articles/itmconf/pdf/2025/09/itmconf_cseit2025_01014.pdf) |
| **Constraint Formulations Survey** | IJCAI 2024 | 约束形式化方法综述 | [PDF](https://ijcai.org/proceedings/2024/0913.pdf) |
| **Lyapunov & Barrier Functions** | 2025 | 基于Lyapunov和CBF的Safe RL | [arXiv](https://arxiv.org/abs/2508.09128) |

---

### 方向2: 控制屏障函数 (Control Barrier Functions)

#### 3.2.1 核心概念

**CBF定义**: 一个函数 h(x) 使得 h(x) ≥ 0 定义安全集，通过约束 ḣ(x) + αh(x) ≥ 0 保证系统不离开安全集。

```
┌─────────────────────────────────────────────────────────────────┐
│  CBF-QP 安全过滤器:                                              │
│                                                                 │
│  u* = argmin ||u - u_nominal||²                                 │
│       s.t.   ḣ(x,u) + αh(x) ≥ 0   (CBF约束)                    │
│              u ∈ U                 (输入约束)                    │
│                                                                 │
│  作用: 最小修正名义控制器，保证安全                              │
└─────────────────────────────────────────────────────────────────┘
```

#### 3.2.2 代表工作

| 方法 | 年份/会议 | 核心思想 | 论文链接 |
|-----|---------|---------|---------|
| **Neural CBF** | CoRL 2021 | 神经网络学习CBF | [arXiv](https://arxiv.org/abs/2109.06697) |
| **CBF as Safety Instructor** | 2025 | CBF指导RL学习安全行为 | [arXiv](https://arxiv.org/abs/2505.18858) |
| **T-CBF** | 2024 | 可通行性CBF，超越碰撞避免 | [PDF](https://cs.gmu.edu/~xxiao2/papers/t_cbf.pdf) |
| **Fault Tolerant Neural CBF** | 2025 | 传感器故障下的鲁棒CBF | [IEEE](https://ieeexplore.ieee.org/document/10610491/) |
| **HOCBF for HRI** | 2024 | 高阶CBF用于人机交互 | [IEEE](https://ieeexplore.ieee.org/document/10319090/) |
| **Explicit CBF Safety Filters** | 2025 | 显式CBF安全过滤器实现 | [arXiv](https://arxiv.org/abs/2512.10118) |
| **Uncertainty-aware CBF** | 2025 | 不确定性感知的CBF参数自适应 | [arXiv](https://arxiv.org/abs/2409.14616) |

**发展趋势**:
- 从手工设计CBF → 神经网络学习CBF
- 从单一碰撞避免 → 多种安全约束 (可通行性、任务安全)
- 从确定性 → 不确定性感知

---

### 方向3: 世界模型与安全预测

#### 3.3.1 核心思想

**World Model**: 学习环境动力学模型，在"想象空间"中预测未来，评估安全性。

```
┌─────────────────────────────────────────────────────────────────┐
│  World Model 安全预测流程:                                       │
│                                                                 │
│  观测 → 编码器 → 潜在状态 → 动力学模型 → 未来状态预测            │
│                                    ↓                            │
│                              安全评估器 → 安全/不安全            │
│                                    ↓                            │
│                              动作选择/修正                       │
└─────────────────────────────────────────────────────────────────┘
```

#### 3.3.2 代表工作

| 方法 | 年份/会议 | 核心思想 | 论文链接 |
|-----|---------|---------|---------|
| **SafeDreamer** | ICLR 2024 | Lagrangian + Dreamer，近零违规 | [arXiv](https://arxiv.org/abs/2307.07176) |
| **PIGDreamer** | ICML 2025 | 特权信息引导，解决POMDP安全 | [OpenReview](https://openreview.net/forum?id=mtk8tTKWs0) |
| **ActSafe** | ICLR 2025 | 乐观探索 + 悲观安全 | [arXiv](https://arxiv.org/abs/2410.09486) |
| **UNISafe** | CoRL 2025 | 不确定性感知潜在安全过滤器 | [arXiv](https://arxiv.org/abs/2505.00779) |
| **Latent-Space Reachability** | RSS 2025 | 潜在空间可达性分析 | [PDF](https://www.roboticsproceedings.org/rss21/p113.pdf) |
| **Zero-shot Safety Prediction** | 2024 | 基础世界模型零样本安全预测 | [arXiv](https://arxiv.org/abs/2404.00462) |
| **DreamerV3** | Nature 2025 | 通用世界模型 | [Nature](https://www.nature.com/articles/s41586-025-08744-2) |

**发展趋势**:
- 从Model-free → Model-based (样本效率)
- 从状态输入 → 视觉输入 (实用性)
- 从单一安全 → 不确定性感知 (鲁棒性)

#### 3.3.3 多样性策略学习与安全的结合 (Diverse Policy Learning + Safety)

**核心思想**: 多样性策略库提供**安全选项**——当某个动作不安全时，可以从多样策略中选择安全的替代方案。

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  多样性 + 安全 结合框架:                                                     │
│                                                                             │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌───────────┐ │
│  │ 多样策略库  │ ──▶ │ 候选动作采样 │ ──▶ │ 安全预测/评估│ ──▶ │ 安全动作  │ │
│  │ (Flow/Diff) │     │ (M个技能)   │     │ (WM/TCN)    │     │ 执行      │ │
│  └─────────────┘     └─────────────┘     └─────────────┘     └───────────┘ │
│                                                                             │
│  关键: 多样性解决"有什么选择"，安全性解决"选哪个安全"                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

**为什么多样性对安全重要**:
1. **提供安全备选**: 单一策略遇到障碍可能无解，多样策略库提供绕行方案
2. **增强泛化性**: 多样策略覆盖更多行为模式，面对新场景更可能有安全选项
3. **零样本避障**: 无需针对特定障碍训练，从多样策略中选择安全的即可

**代表工作**:

| 方法 | 年份/会议 | 核心思想 | 论文链接 |
|-----|---------|---------|---------|
| **DIVO** | RAL 2025 | 环境随机化诱导多样性 + TCN预测安全过滤 | [IEEE](https://ieeexplore.ieee.org/document/10847909) |
| **Diffusion Policy** | RSS 2023 | 扩散模型建模多模态动作分布 | [Project](https://diffusion-policy.cs.columbia.edu/) |
| **FlowPolicy** | AAAI 2025 | Consistency Flow Matching快速采样 | [arXiv](https://arxiv.org/abs/2412.04987) |
| **VFP** | 2025 | 变分Flow Matching处理多模态 | [arXiv](https://arxiv.org/abs/2508.01622) |
| **G-QDIL** | ICLR 2025 | QD + 生成模型模仿学习 | [ICLR](https://www.iclr.cc/virtual/2025/32382) |
| **Di-SkilL** | ICML 2024 | MoE + 课程学习多样技能 | [ICML](https://icml.cc/virtual/2024/poster/34802) |

**多样性方法分类**:

| 方法类型 | 代表工作 | 多样性来源 | 与安全结合方式 |
|---------|---------|-----------|---------------|
| **生成模型** | Diffusion Policy, FlowPolicy | 扩散/流模型采样 | 采样多个→安全过滤 |
| **环境诱导** | DIVO | 随机障碍物部署 | 自然涌现多样性 |
| **Quality-Diversity** | G-QDIL, MAP-Elites | 进化算法 | 维护多样策略库 |
| **混合专家** | Di-SkilL | MoE结构 | 专家专门化 |

**研究空白与机会**:

| 空白 | 描述 | 潜在创新 |
|-----|------|---------|
| **多样性+World Model** | 现有WM方法只学单一策略 | 在WM潜在空间学习多样技能 |
| **安全感知采样** | 现有方法先采样后过滤 | 直接采样安全的多样动作 |
| **OOD多样性安全** | 多样策略在OOD场景的安全性未知 | 不确定性感知的多样性选择 |

**与World Model结合的技术路线**:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Diverse World Model (研究空白):                                             │
│                                                                             │
│  训练: 随机环境 → World Model → 潜在空间多样技能发现                         │
│                                                                             │
│  部署: 观测 → WM编码 → 多样技能采样 → WM想象预测 → 安全评估 → 执行           │
│                                                                             │
│  创新点:                                                                     │
│  1. 在WM潜在空间而非观测空间学习多样性                                       │
│  2. 用WM想象替代TCN预测进行安全评估                                          │
│  3. Safety-conditioned采样直接生成安全的多样动作                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### 方向4: 大模型驱动的机器人安全

#### 3.4.1 VLA/VLM安全对齐

**核心问题**: Vision-Language-Action模型可能执行危险动作，需要安全对齐。

| 方法 | 年份 | 核心思想 | 论文链接 |
|-----|------|---------|---------|
| **SafeVLA** | 2025 | 约束学习对齐VLA安全 | [arXiv](https://arxiv.org/abs/2503.03480) |
| **AttackVLA** | 2025 | VLA对抗攻击benchmark | [arXiv](https://arxiv.org/abs/2511.12149) |
| **VLA-Fool** | 2025 | 多模态对抗攻击VLA | [arXiv](https://arxiv.org/abs/2511.16203) |
| **BadVLA** | 2025 | VLA后门攻击 | [arXiv](https://arxiv.org/abs/2505.16640) |
| **FreezeVLA** | 2025 | 动作冻结攻击VLA | [arXiv](https://arxiv.org/abs/2509.19870) |
| **Phantom Menace** | 2025 | 物理传感器攻击VLA | [arXiv](https://arxiv.org/abs/2511.10008) |
| **Adversarial Vulnerabilities** | 2024 | VLA对抗脆弱性探索 | [arXiv](https://arxiv.org/abs/2411.13587) |

#### 3.4.2 LLM安全任务规划

**核心问题**: LLM生成的任务计划可能违反安全约束。

| 方法 | 年份/会议 | 核心思想 | 论文链接 |
|-----|---------|---------|---------|
| **SELP** | ICRA 2025 | 安全高效LLM规划 | [PDF](https://cs.purdue.edu/homes/lintan/publications/selp-icra25.pdf) |
| **SafePlan** | 2025 | 形式逻辑 + CoT增强安全 | [arXiv](https://arxiv.org/abs/2503.06892) |
| **Safety Aware Planning** | 2025 | LLM安全感知任务规划 | [arXiv](https://arxiv.org/abs/2503.15707) |
| **Graphormer-Guided** | 2025 | 图神经网络引导LLM安全 | [arXiv](https://arxiv.org/abs/2503.06866) |
| **LLM + Knowledge Graph** | 2024 | 知识图谱增强LLM安全 | [arXiv](https://arxiv.org/abs/2405.17846) |
| **Cross-Layer Supervision** | 2024 | 跨层序列监督机制 | [IEEE](https://ieeexplore.ieee.org/document/10801576/) |

**发展趋势**:
- 从信任LLM → 验证LLM输出
- 从单一模态 → 多模态安全
- 从被动防御 → 主动安全对齐

---

### 方向5: 人机协作安全

#### 3.5.1 核心挑战

- 人类行为不可预测
- 需要实时响应
- 安全与效率的平衡

#### 3.5.2 代表工作

| 方法 | 年份/会议 | 核心思想 | 论文链接 |
|-----|---------|---------|---------|
| **Proactive HRI CBF** | 2025 | 主动层次化CBF人机交互 | [arXiv](https://arxiv.org/abs/2505.16055) |
| **Human Digital Twin** | 2025 | 人类数字孪生 + 混合现实 | [PDF](https://sites.gc.sjtu.edu.cn/youyibi/wp-content/uploads/sites/3/2025/03/RCIM2025-HRI.pdf) |
| **Tool-aware Collision** | 2025 | 工具感知碰撞避免 | [arXiv](https://arxiv.org/abs/2508.20457) |
| **Dynamic Risk Assessment** | 2025 | 动态风险评估HRC | [arXiv](https://arxiv.org/abs/2503.08316) |
| **Safety-oriented HRC** | 2025 | 人类反馈引导安全行为 | [SciOpen](https://www.sciopen.com/article/10.26599/JIC.2025.9180092) |
| **Proactive Motion Planning** | 2025 | 主动人类运动预测 | [PDF](https://dyalab.mines.edu/2025/icra-workshop/19.pdf) |

---

### 方向6: Sim-to-Real安全迁移

#### 3.6.1 核心问题

仿真中学习的安全策略如何迁移到真实世界？

#### 3.6.2 代表工作

| 方法 | 年份 | 核心思想 | 论文链接 |
|-----|------|---------|---------|
| **SPiDR** | 2025 | 零样本Sim-to-Real安全 | [arXiv](https://arxiv.org/abs/2509.18648) |
| **Real-is-Sim** | 2025 | 动态数字孪生弥合差距 | [arXiv](https://arxiv.org/abs/2504.03597) |
| **FalconGym** | 2025 | 零样本视觉四旋翼安全 | [arXiv](https://arxiv.org/abs/2503.02198) |
| **Real-to-Sim-to-Real** | 2024 | 真实→仿真→真实鲁棒操作 | [arXiv](https://arxiv.org/abs/2403.03949) |

---

## 四、综述论文汇总

| 综述主题 | 年份 | 会议/期刊 | 链接 |
|---------|------|---------|------|
| **Safe RL Methods, Theories, Applications** | 2024 | IEEE TPAMI | [IEEE](https://ieeexplore.ieee.org/document/10675394/) |
| **Safe RL in Robotics** | 2025 | ITM Conf | [PDF](https://www.itm-conferences.org/articles/itmconf/pdf/2025/09/itmconf_cseit2025_01014.pdf) |
| **Safe RL & CMDP Survey** | 2025 | arXiv | [arXiv](https://arxiv.org/abs/2505.17342) |
| **Safe Learning for Contact-Rich Tasks** | 2025 | arXiv | [arXiv](https://arxiv.org/abs/2512.11908) |
| **Safety of Embodied Navigation** | IJCAI 2025 | IJCAI | [PDF](https://www.ijcai.org/proceedings/2025/1189.pdf) |
| **Embodied AI Vulnerabilities & Attacks** | 2025 | arXiv | [arXiv](https://arxiv.org/abs/2502.13175) |
| **World Models for Embodied AI** | 2025 | arXiv | [arXiv](https://arxiv.org/abs/2510.16732) |
| **Robotics with Foundation Models** | 2024 | arXiv | [ADS](https://ui.adsabs.harvard.edu/abs/2024arXiv240202385X/abstract) |
| **Deep RL for Robotics** | 2025 | Annual Reviews | [Link](https://www.annualreviews.org/content/journals/10.1146/annurev-control-030323-022510) |


---

## 五、博士课题方向建议

### 5.1 方向对比分析

| 方向 | 成熟度 | 创新空间 | 工程难度 | 发表难度 | 应用前景 |
|-----|-------|---------|---------|---------|---------|
| **Safe RL (约束优化)** | 高 | 中 | 中 | 中 | 高 |
| **Neural CBF** | 中 | 高 | 高 | 中 | 高 |
| **World Model + Safety** | 中 | 高 | 高 | 低 | 高 |
| **VLA/LLM安全** | 低 | 很高 | 中 | 低 | 很高 |
| **人机协作安全** | 中 | 中 | 高 | 中 | 很高 |
| **多样性+安全** | 中 | 高 | 中 | 中 | 高 |
| **Sim-to-Real安全** | 中 | 高 | 很高 | 中 | 高 |

### 5.2 推荐研究方向

#### 方向A: World Model + 安全预测 (推荐度: ⭐⭐⭐⭐⭐)

**理由**:
- 2024-2025最热门方向 (SafeDreamer, PIGDreamer, UNISafe)
- 样本效率高，适合真实机器人
- 支持视觉输入，实用性强
- 有清晰的技术路线和开源代码

**研究问题**:
1. 如何在世界模型中建模安全约束？
2. 如何处理世界模型预测误差导致的安全问题？
3. 如何在潜在空间进行可达性分析？

**关键论文**:
- SafeDreamer (ICLR 2024)
- PIGDreamer (ICML 2025)
- UNISafe (CoRL 2025)
- Latent-Space Reachability (RSS 2025)

---

#### 方向B: VLA/LLM安全对齐 (推荐度: ⭐⭐⭐⭐)

**理由**:
- 最前沿方向，与大模型结合
- 研究空白大，创新机会多
- 工业界高度关注
- 发表机会好 (新兴领域)

**研究问题**:
1. 如何对齐VLA模型的安全偏好？
2. 如何防御VLA的对抗攻击？
3. 如何验证LLM生成的任务计划安全？

**关键论文**:
- SafeVLA (2025)
- AttackVLA (2025)
- SELP (ICRA 2025)
- SafePlan (2025)

---

#### 方向C: 学习控制屏障函数 (推荐度: ⭐⭐⭐⭐)

**理由**:
- 理论基础扎实
- 可证明安全保证
- 与学习方法结合是趋势
- 工业应用成熟

**研究问题**:
1. 如何从数据学习有效的CBF？
2. 如何处理输入约束下的CBF？
3. 如何在不确定性下保证CBF有效？

**关键论文**:
- Neural CBF (CoRL 2021)
- CBF as Safety Instructor (2025)
- Uncertainty-aware CBF (2025)

---

#### 方向D: 人机协作安全 (推荐度: ⭐⭐⭐)

**理由**:
- 应用价值高，工业需求大
- 与人因工程结合
- 真实场景验证机会多

**研究问题**:
1. 如何预测人类意图和运动？
2. 如何在保证安全的同时不过度保守？
3. 如何处理接触安全？

**关键论文**:
- Proactive HRI CBF (2025)
- Human Digital Twin (2025)
- Dynamic Risk Assessment (2025)

---

#### 方向E: 多样性策略学习 + 安全控制 (推荐度: ⭐⭐⭐⭐)

**理由**:
- 多样性为安全提供选项库，是安全的"来源"
- 与World Model结合是研究空白
- 有清晰的技术路线 (DIVO → World Model扩展)
- 实验驱动，效果说话

**研究问题**:
1. 如何在World Model潜在空间学习多样技能？
2. 如何直接采样安全的多样动作 (Safety-conditioned sampling)？
3. 如何评估多样策略在OOD场景的安全性？

**关键论文**:
- DIVO (RAL 2025) - 多样性+安全的完整框架
- Diffusion Policy (RSS 2023) - 生成模型多样性
- FlowPolicy (AAAI 2025) - 快速Flow采样
- SafeDreamer (ICLR 2024) - World Model安全

**与其他方向的结合**:
- **+ World Model**: 在想象空间评估多样策略安全性
- **+ CBF**: 用CBF过滤多样策略中的不安全动作
- **+ 人机协作**: 多样策略提供人类友好的替代方案

---

### 5.3 具体课题建议

#### 课题1: 不确定性感知的世界模型安全控制

```
核心思想:
┌─────────────────────────────────────────────────────────────────┐
│  观测 → World Model → 潜在状态 + 认知不确定性                    │
│                           ↓                                     │
│  想象未来轨迹 → 安全评估 + 不确定性惩罚 → 安全动作选择           │
└─────────────────────────────────────────────────────────────────┘

创新点:
1. 用世界模型的认知不确定性作为安全信号
2. 在OOD场景自动降低置信度
3. 结合可达性分析进行安全验证
```

**参考**: UNISafe (CoRL 2025), Latent-Space Reachability (RSS 2025)

---

#### 课题2: VLA模型的安全对齐与鲁棒性

```
核心思想:
┌─────────────────────────────────────────────────────────────────┐
│  VLA模型 → 安全约束微调 (RLHF/DPO) → 安全对齐的VLA              │
│                           ↓                                     │
│  对抗训练 → 鲁棒性增强 → 部署时安全监控                          │
└─────────────────────────────────────────────────────────────────┘

创新点:
1. 设计VLA安全对齐的数据集和方法
2. 研究VLA的对抗鲁棒性
3. 构建VLA安全评估benchmark
```

**参考**: SafeVLA (2025), AttackVLA (2025)

---

#### 课题3: LLM任务规划的形式化安全验证

```
核心思想:
┌─────────────────────────────────────────────────────────────────┐
│  自然语言指令 → LLM规划 → 形式化验证 → 安全任务计划              │
│                           ↓                                     │
│  时序逻辑约束 → 模型检验 → 安全保证                              │
└─────────────────────────────────────────────────────────────────┘

创新点:
1. 将LLM输出转换为可验证的形式化表示
2. 设计机器人任务的安全时序逻辑
3. 实时验证与修正机制
```

**参考**: SafePlan (2025), SELP (ICRA 2025)

---

#### 课题4: 多样性策略学习 + World Model安全预测

```
核心思想:
┌─────────────────────────────────────────────────────────────────┐
│  训练: 随机环境 → World Model → 潜在空间多样技能发现             │
│                           ↓                                     │
│  部署: 观测 → WM编码 → Flow Matching采样M个技能                  │
│                           ↓                                     │
│        WM想象预测 → 安全评估 → 选择最优安全技能 → 执行           │
└─────────────────────────────────────────────────────────────────┘

创新点:
1. 在World Model潜在空间学习多样技能 (vs DIVO在观测空间)
2. Safety-conditioned Flow Matching直接采样安全动作
3. 不确定性感知的多样性选择
```

**参考**: DIVO (RAL 2025), SafeDreamer (ICLR 2024), FlowPolicy (AAAI 2025)

**与DIVO的对比**:

| 方面 | DIVO | 本课题 |
|-----|------|-------|
| 多样性空间 | 观测空间 | WM潜在空间 |
| 安全预测 | TCN轨迹预测 | World Model想象 |
| 输入模态 | 状态向量 | 视觉/多模态 |
| OOD处理 | 无 | 不确定性感知 |

---

## 六、关键资源

### 6.1 重要课程

| 课程 | 机构 | 讲师 | 链接 |
|-----|------|------|------|
| **Embodied AI Safety** | CMU | Andrea Bajcsy | [Link](https://abajcsy.github.io/embodied-ai-safety/) |
| **Safe Learning in Robotics** | Berkeley | - | - |
| **Robot Learning** | Stanford | Chelsea Finn | - |

### 6.2 重要实验室

| 实验室 | 机构 | 方向 | 代表工作 |
|-------|------|------|---------|
| **IntentLab** | CMU | 安全+人机交互 | UNISafe, Latent Reachability |
| **PKU-Alignment** | 北京大学 | Safe RL, World Model | SafeDreamer, PIGDreamer |
| **REALM** | MIT | 学习CBF | Neural CBF, PNCBF |
| **LAS** | ETH Zurich | Safe Exploration | ActSafe |
| **RAIL** | Berkeley | Robot Learning | - |

### 6.3 开源代码库

| 项目 | 用途 | 链接 |
|-----|------|------|
| **Safety-Gymnasium** | Safe RL Benchmark | [GitHub](https://github.com/PKU-Alignment/safety-gymnasium) |
| **SafeDreamer** | World Model + Safe RL | [GitHub](https://github.com/PKU-Alignment/SafeDreamer) |
| **OmniSafe** | Safe RL算法库 | [GitHub](https://github.com/PKU-Alignment/omnisafe) |
| **safe-control-gym** | 安全控制Benchmark | [GitHub](https://github.com/utiasDSL/safe-control-gym) |

### 6.4 重要会议

| 会议 | 领域 | 相关Track |
|-----|------|----------|
| **CoRL** | Robot Learning | Safe Learning |
| **ICRA** | Robotics | Safety & HRI |
| **RSS** | Robotics | Safety |
| **ICLR** | Machine Learning | Safe RL |
| **NeurIPS** | Machine Learning | Safe RL |

---

## 七、总结

### 7.1 领域发展趋势

1. **从约束优化 → 学习安全**: 从手工设计约束到从数据学习安全
2. **从Model-free → Model-based**: 世界模型提供样本效率和安全预测
3. **从单一安全 → 多维安全**: 物理安全、任务安全、对抗安全
4. **从小模型 → 大模型**: VLA/LLM带来新的安全挑战和机会
5. **从仿真 → 真实**: Sim-to-Real安全迁移成为关键

### 7.2 研究建议

1. **选择热门但有空间的方向**: World Model + Safety, VLA Safety
2. **关注实际应用**: 人机协作、自动驾驶、服务机器人
3. **重视实验验证**: 真实机器人实验是加分项
4. **跟踪最新进展**: 关注CoRL, ICRA, RSS, ICLR

### 7.3 核心论文清单 (必读)

1. SafeDreamer (ICLR 2024) - World Model + Safe RL
2. UNISafe (CoRL 2025) - 不确定性感知安全
3. SafeVLA (2025) - VLA安全对齐
4. SELP (ICRA 2025) - LLM安全规划
5. Neural CBF (CoRL 2021) - 学习CBF
6. Safe RL Survey (IEEE TPAMI 2024) - 综述

---

*调研完成于 2025年12月24日*
