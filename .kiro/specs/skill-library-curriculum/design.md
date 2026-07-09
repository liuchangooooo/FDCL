# 技术设计文档:技能库可学性课程(路B / Skill-Library Curriculum)

## Overview

### 定位
在 **DIVO**(隐策略 π(a|s,z) 零样本泛化,RA-L 2025)基础上,构建一个**以"技能库可学性(skill-library learnability)"为课程信号的 LLM 障碍课程**。核心创新点:借鉴 SFL 的 p(1−p) 学习价值思想,把 SFL 的**单策略成功率 p** 换成**技能库的 realized(库内技能的成功比例)**,让 LLM 生成的障碍课程始终对准"技能库当前学得会又不完全会"的前沿。只做 **Push-T**(Ant 无源码)。

本项目并行推进两条路线,本文档只描述**路B**:
- **路C**(不在本文档范围):库不训,只当"出题镜头",仅改 verifier 治锁,z/训练不动。思想与代码改点已另行确定。
- **路B**(本文档):**真训一族多样且都近最优的技能库**,新增独立技能码 w,使 realized 随训练真实上升、课程闭环真闭。路B = 路C(治锁 verifier)+ "真训库"。

### 与 DIVO 现状的关键差异(先讲清,避免张冠李戴)
以下均为已核实的代码事实,是本设计所有决策的地基:

1. **DIVO stage-1 策略是确定性的**。`LatentDetPolicy`(`DIVO/policy/ldpi.py`)的 `predict_action`:`z = encoder(obs)`,`state = obs2state(obs)`,`action = decoder(cat([state, z]))`。一个 obs 对应一个确定 z(MLP,非随机采样)。
2. **障碍信息只经 z 进 decoder**。`obs2state(obs) = obs[:, :4]`(只含 T-block 位姿),障碍布局只能通过 `z = encoder(obs)` 这一条通道进入 decoder。**因此 z 是"情形编码"通道,不能被 overload 成技能码**——这是路B 必须另立技能码 w 的根本原因。
3. **DIVO stage-1 没有"每情形多技能"**。DIVO 的"多技能/动作多样性"来自 **stage-2/3 的 latent skill sampler + motion predictor**(`sampler/fm.py` 等)。本项目**部署定死为 encoder 单技能、单阶段,不接 stage-2/3**。
4. **表 II 的 "action diversity"**(`evaluation.py::calculate_action_diversity`)= 500 obs × 每个采 100 技能 → 解码动作 → `np.var(actions, axis=0).sum()` → 再对 obs 求平均。**这是动作方差,依赖 stage-2/3 sampler**;stage-1 确定性策略在此指标上多样性=0。本项目用它作**对标口径**,但通过 w-codebook 而非 stage-2/3 sampler 产生多技能。
5. **现有 verifier 是四门不变差合取(病灶)**。`decide_skill_acceptance` → `evaluate_learnable_frontier_shift` = `infeasible_cap_ok AND trivial_not_worse AND boundary_not_worse AND coverage_not_worse` 合取;`mean_lv` 只进日志、不参与决策。成熟期各门顶格 → 棘轮锁 → 停摆(6/22 run `evolve_index=17` 卡死已实证)。路B 课程阶段要**治锁**。

### 设计边界与非目标
- **部署严格单技能、单阶段**:部署时 `z = encoder(obs)` + 固定技能码 `w_0`,与标准 DIVO 可比。不引入 stage-2/3。
- **技能库可学性是唯一核心创新点**,不为讲故事堆无测量支撑的组件。
- **评测严格对齐 DIVO 的 B/M/U/D**(barrier/maze/u-shape/dead-end),不为找赢临时调评测列(禁 p-hacking)。**验证/选 best 用固定验证分布(DIVO `between`,`pusht_mujoco_llm` + obstacle_dist=between + num=2 + size=0.01)上的 `test_mean_score`(w_0 部署)**,不用会随课程漂移的 seen G_t、不用 B/M/U/D(防泄漏)。实现复用现有 `llm_eval_env`(不 `set_obstacle_config`,普通 reset 走 between 原版分支),不新写环境/评测代码。
- **多样性目标是生死关**:必须保证"解法多样"而非"动作多样"或"多样失败"。

### 已知关键风险(正视,不掩盖)
1. **zselect 红旗**:`zselect_headroom` 显示库 oracle 只比 encoder 单 z 高 2.5–10%,best-of-K 更差。"训一族技能库 → 单 z 部署受益"这条因果**是赌注**。Stage 1 go/no-go 就是证伪或证实它。
2. **w 塌缩**:名义 K 个技能码可能只学出 2–3 个有效模式(K_eff ≪ K)。
3. **稀疏奖励下早期多样信号弱**。
4. **干扰部署**:多样性压力可能拖累 w_0 部署性能(B/M/U/D 掉)。
5. **DIVO 表 II 对标**:action diversity 的 0.62 是**参考上界/对标值,不是 Stage 1 硬门**(其口径来自 stage-2/3 sampler,机制与 K-codebook 相近但不同)。
6. **算力**:Push-T 是 CPU 物理仿真,采集多技能 rollout 成本高。

## Architecture

### 整体数据流
```
                    ┌─────────────────────────────────────────┐
                    │  LLM 障碍生成器 (generator)               │
                    │  输入(重写 prompt): 当前 generator code + │
                    │    p 分布统计 + 进化方向 + boundary 例子 + │
                    │    拒绝原因                                │
                    │  输出: generate_obstacles(tblock_pose,    │
                    │        num_obstacles)                      │
                    └───────────────┬─────────────────────────┘
                                    │ 障碍布局 obstacles
                                    ▼
   ┌──────────────────────────────────────────────────────────────┐
   │  Push-T MuJoCo 环境 (obs = [T-block位姿(4), 障碍(2*num)])       │
   └───────────────┬──────────────────────────────────────────────┘
                   │ obs
                   ▼
   ┌──────────────────────────────────────────────────────────────┐
   │  技能条件策略 (Skill-Conditioned Policy)                        │
   │    state = obs2state(obs)        # T-block 位姿, obs[:, :4]     │
   │    z     = encoder(obs)          # 情形/障碍码 (唯一障碍通道)    │
   │    w     ∈ codebook {w_0, w_1..w_K}   # 技能/打法码 (新增)      │
   │    a     = decoder(cat([state, z, w]))                          │
   └───────────────┬──────────────────────────────────────────────┘
                   │ a
                   ▼
   ┌──────────────────────────────────────────────────────────────┐
   │  技能条件 critic  Q(obs, w, a)  (w-conditioned, 新增 w 输入)     │
   └──────────────────────────────────────────────────────────────┘

   训练回合采集:部署技能 w_0 与 probe 技能 w_1..w_K 都进 buffer,打 w 标签
   课程编排:周期 probe → 库可学性信号 → LLM 出题 → 治锁 verifier 验收
```

### 三条改动主线
1. **策略/网络**:decoder 增加 w 输入 + 引入技能码 codebook(`DIVO/policy/ldpi.py` + 对应 nets)。
2. **训练**:executed-w-only actor loss(`L_actor = L_exec + L_z`,无独立"覆盖项")+ w-条件 critic + skill-labeled replay buffer + reward routing(w_0→reward_task,probe→reward_total);采集技能回合入 buffer 打 w 标签(`td3_curriculum_workspace.py` + `DIVO/critic/mcritic.py`)。Form B 多样性奖励是 Stage 1 可选/已实现机制,Stage 2 V0 关闭(β_div=0);`L_cov-all` 只是默认关闭的 ablation,不是主方法组件。
3. **课程/验收**:单标量治锁 verifier + 库 boundary 信号 `b=p(1−p)`(`candidate_verifier.py` + `learnable_frontier.py`)。

### 技能的形式化定义
在路B 里,技能**不是**裸 latent,而是"由技能码 w 诱导、在当前网络参数下的行为策略":
```
π_t^k(a | s, obs) = D_θt( s, E_ηt(obs), w_k )
技能库 S_t = { D_θt( s, E_ηt(obs), w_k ) }_{k=0..K}
```
- `w_0`:**部署技能**(独立,不进 probe bank)。
- `W_probe = {w_1, ..., w_K}`:**探针技能库**(度量可学性/多样性)。

z 仍只承载"障碍/情形"信息(唯一障碍通道),w 承载"用哪种打法解"。两者正交:同一障碍情形(z 固定)下,不同 w 给出不同解法。

## Components and Interfaces

### 组件 1:技能条件策略(`DIVO/policy/ldpi.py`)
Decoder 输入维度由 `dim(state) + dim(z)` 改为 `dim(state) + dim(z) + d_w`:
```
a = decoder( cat([state, z, w], dim=-1) )
```
接口:
- `codebook`:常量张量,形状 `[K+1, d_w]`(索引 0 为 w_0 部署,1..K 为 probe)。**第一版不参与梯度**。
- `predict_action(obs)`:默认拼 w_0 部署,保持与现有调用兼容。
- `predict_action_with_skill(obs, w)` / `decode_with_skill(state, z, w)`:供训练与 probe 使用。
- decoder MLP 首层 `in_chan += d_w`(`nets` 配置)。

### 组件 2:技能条件 critic(`DIVO/critic/mcritic.py`)
- `SingleFCCritic.forward(obs, actions)` → `forward(obs, w, actions)`,内部 `q_net(cat([obs, w, actions], dim=1))`;`MultiCritic` 透传 w。
- **理由**:不同技能在同一 obs 下价值不同,critic 必须以 w 为条件才能给每个技能估准价值。
- **注意**:`compute_next_q_value` / `compute_critic_loss` 里所有 `critic(obs, action)` 调用都要带该转移的 w 标签。

### 组件 3:actor 损失(`td3_curriculum_workspace.py::compute_policy_loss`)
**Stage 1 主损失以 executed-w-only 为准(消除 L_cov 与 executed-w 之间的张力):**
```
L_actor = L_exec + L_z
L_exec = - E_{(obs, w_exec)}[ Q(obs, w_exec, D(s, E(obs), w_exec)) ],  w_exec ∈ {w_0, w_1..w_K}
```
- `L_exec`:actor 损失只对该转移**实际执行过的技能** w_exec 更新(按 buffer 的 w 标签 mask)。w_0 与每个 w_k 都各自在自己的 rollout 上被训到,**Stage 1 不再写独立的 all-w L_cov 主项**。
- **命名为 `L_exec` 而非 `L_task`**:单 critic 下 probe 技能的 `Q(obs, w_k, ·)` 已含多样性回报(reward_total,见组件 4),因此 `L_exec` 优化的是 total return,不是纯任务价值。**多样性不作为独立可微 actor 项**(那是 Form A,已降级监控),只经 reward_total 进 Q。
- `L_z`:保留 DIVO 现有 gaussian z 正则(把 z 的 batch 均值/方差拉向 N(0,1)),挂在部署路径。
- **为什么不写 all-w L_cov 主项**:若对同一 obs 用所有 w_k 求 `Q(obs, w_k, D(obs, w_k))`,会引入"该 obs 下从未执行过的 (obs, w_k)"组合;即使 critic 带 w,它在该 obs 上也没见过这些技能动作,off-policy 外推易过估把 actor 带偏——这正是最想避免的问题。写成 all-w 会让该问题回来。
- **all-w coverage 降为 ablation/warm-up 可选项**:`L_cov-all = -(1/K) Σ_k Q(obs, w_k, D(obs, w_k))`,仅在 critic 稳定后以 `λ_cov-all` 从 0 warm-up 引入,不作 Stage 1 主线。

### 组件 4:多样性目标(生死关,经 reward_div/β_div 进 critic)
**默认 Form B(轨迹级、软 near-success 门控、同环境配对、margin 有界)= 真多样性目标:**

- **轨迹嵌入 φ**:**任务坐标系**(SE2 归一化 / `task_frame_waypoints`,跨场景可比)下的 state 轨迹嵌入(复用 `skill_signal._embed_route` 定长展平 + 任务坐标归一化)。

- **同环境配对(diversity_probe_batch)**:多样性必须在**同一环境 seed** 内比较技能,否则会混入环境差异而非技能差异。对同一批环境 seeds `E = {e_1..e_N}`,枚举技能 w_1..w_K 得 τ_{i,k};距离**只在同一 e_i 内**计算 `d(φ(τ_{i,k}), φ(τ_{i,l}))`。Stage 1 用同环境 paired rollout;后续可用 archive 加速。

- **软 near-success 门控(替代硬 1[success])**:早期成功轨迹稀少,硬门会让 r_div≈0 启动不了。用轨迹质量 soft gate:
  ```
  q(τ) = α·success + β·progress − γ·collision − η·timeout
  g(τ) = σ( (q(τ) − τ_q) / T )
  r_div(i,k) = g_{i,k} · [ Σ_{l≠k} g_{i,l} · min(d(φ_{i,k}, φ_{i,l}), m) ] / [ Σ_{l≠k} g_{i,l} + ε ]
  ```
  成功轨迹权重大,near-success 也能在早期提供分化信号;保留"只奖励有效技能"的原则但不用硬门。

- **margin 有界**:距离超过 margin `m` 后不再加分(替代"方差+饱和惩罚",避免技能往动作边界冲,少调一个权重)。

- **走 RL,但严格区分两种 reward(防污染 critic 任务语义)**:r_div 加进**技能回合**回报走 RL,同时分别记录:
  ```
  reward_task                          # 纯任务奖励
  reward_total = reward_task + β_div · reward_div
  ```
  - **w_0 回合不加 r_div**(w_0 不在多样集),故 `Q(obs, w_0, ·)` 自动是纯任务价值,**部署不被多样性污染**——这是独立 w_0 + executed-w-only 的自洽收益。
  - probe 技能回合用 reward_total 训练多样性。
  - **success floor 与所有 go/no-go 一律基于 reward_task / success,绝不用 reward_total**。
  - 若单 critic 下仍担心 probe 技能 Q 偏好"多样但任务略差",可选升级双 critic `Q_task` / `Q_total`(Q_task 供 w_0 与任务监控,Q_total 供 probe 多样训练);Stage 1 先用单 critic + 双 reward 记录,双 critic 留后备。

**Form A(动作方差排斥,decoder-only 可微)= 降级为"廉价监控 + 塌缩探测器"**,不当真目标:
- 计算 `ActionVar`(对标表 II)与 `SatRate`(饱和率)仅作日志与塌缩报警(期望 ≈0 报警)。
- 不当真目标的理由:三个假多样风险——(1) 动作方向不同但轨迹殊途同归(ActionVar↑ 但 TrajDiv≈0);(2) 多样权重过大把技能推到动作边界形成"多样失败";(3) 在所有 replay state 上强拉开所有 w 会扰动未充分训练的技能。补丁堵不住风险 1(局部动作指标无法保证全局解法不同)。

### 组件 5:约束式平衡与调度(DOMiNO 式)
目标 `max Diversity s.t. Success ≥ 阈值`。**调度的是 `β_div`**(Form B 经 `reward_total = reward_task + β_div·reward_div` 进 critic,不存在独立可微多样项,故无 λ_div;λ 名字仅留给未来 Form A 可微项 `λ_A`)。按 eval 周期慢调,带 bound:
```
if avg_success < success_floor:            # 多样性伤任务
    beta_div *= 0.5
elif K_eff < eff_floor and avg_success >= success_floor:   # 任务能做但技能塌缩
    beta_div *= 1.2
else:
    keep beta_div
beta_div = clip(beta_div, beta_div_min, beta_div_max)
```
**分阶段启动**:task warm-up(仅训 `L_exec + L_z`,`β_div=0`)→ 小 β_div 打开(1e-3~1e-2)→ 动态调。`avg_success`/floor 一律取 `reward_task`/真实 success。

### 组件 6:回合采集(`td3_curriculum_workspace.py`)
- 按比例采集技能回合:w_0 与 probe 技能 w_1..w_K 都采。
- **w_0 采样强度不低于任一 probe 技能**(防部署被 probe 库稀释):配置 `w0_rollout_ratio` 与 `probe_rollout_ratio`,第一版直接约束 w_0 的回合数 ≥ 每个 w_k 的回合数。
- 每条转移在 buffer 打 w 标签,供 executed-w-only mask 与 w-条件 critic 使用。
- **诚实保留**:即便如此,w_0 仍可能只通过共享 decoder 间接受益;go/no-go 必须专门盯 w_0 的部署指标,不能只看 probe 库多样性(见 Testing Strategy)。

## Stage 0: LLM Initial Generator Bootstrap

主线起点。技能库训练前先用 LLM 生成并验证初始障碍生成器 G_0,使"用 LLM 替代人工障碍环境部署"从第一步就闭环(复用现有 Stage A/phase0 代码,零改动)。

- **输入**:Push-T task description、workspace bounds、obstacle count/size、validity constraints、`generate_obstacles(tblock_pose, num_obstacles)` 契约。
- **实现复用**:`use_llm_obstacles=true` + `_init_topology_generator` → `generator_source.get_initial_code(..., acgs_api)` + `build_phase0_prompt_stage_a`;需可用 LLM API(deepseek/openai key)。
- **输出**:候选生成器 G_0(遵循现有契约)。
- **选择判据**:`code_pass`(SandboxPushTExecutor)+ physical validity(`valid_rate` / `duplicate_rate` / contact-`get_ncon`)+ **轻量难度散布启发式**;只保证 G_0 不"全无效/全极易/全极难"。
- **明确禁止**:Stage 0 此时**没有 W_probe**,**SHALL NOT 用 p_i / b_i / boundary 选 G_0**(这些信号要到 Stage 1 训出探针库后才存在)。
- **产物流向**:Stage 1 训练分布 = G_0 动态采样(`_make_llm_scene`);Stage 2 从 `current G_t = G_0` 起周期进化。

## Stage 2 架构(Route B V0:库可学性驱动的 LLM 障碍生成器进化)

Stage 2 用 LLM 根据**当前技能库的能力边界**周期性**重写障碍生成器 G_t**,替代 DIVO stage-1 人工设计的障碍训练分布。核心与旧版的本质区别:

> **更新的是生成器 G_t → G_{t+1},不是障碍池;不做训练池、不做 cache。** 训练分布是"当前生成器诱导的分布" `D_t ~ G_t`,每次 reset 从 G_t 动态采样。

**全程 LLM 驱动的完整闭环**:初始生成器 G_0 由 LLM 生成(见上文 §Stage 0),Stage 2 从 `current G_t = G_0` 起进化:G_0 由 LLM 起题 → 库探针评估 → LLM 按边界重写 G_t → verifier 选择 → 继续训练。
- **主线定义**:完整自动 Route B 主流程 = **Stage 0 LLM 生成 G_0 → Stage 1 在 G_0 上预训 w_0+W_probe → Stage 2 从 G_0 起 evolve**。`use_llm_obstacles=false` 的随机障碍训练**仅作隔离技能库能力的 debug/ablation,不属于主线**。
- **依赖**:全程需可用的 LLM API(deepseek/openai key)。
- **对比公平性(诚实提醒)**:早期 Stage 1 的强结果是在随机障碍上得到的(ablation);四格正式对比里"无课程"臂(a/c)固定同一初始分布、"有课程"臂(b/d)都从**同一个 LLM 生成的 G_0** 起步再 evolve(仅信号源不同),避免把"换了训练分布"误算成课程收益。

**V0 明确不做**(全部延后):多样性奖励(β_div=0、不跑 diversity_batch)、加权覆盖 S、训练池/accepted pool/cache、双头架构、best-of-K 主部署、coverage-aware score。V0 只保留 **boundary 信号 `b=p(1−p)`**。

### 组件 7:库可学性 boundary 信号(`skill_signal.py`,复用 `extract_skill_signals`)
**只改探针源,不新造接口**:`extract_skill_signals` 的注入源从"随机 z_bank"改为**训练好的技能码 W_probe = {w_1..w_K}**(用 `rollout_fixed_skill`);生成器仍用现有契约 `generate_obstacles(tblock_pose, num_obstacles)`,环境用 `env.set_obstacle_config`。

对当前 G_t 采样 M 个**临时评估**布局 `E_t = {e_1..e_M} ~ G_t`(用完即丢,不 cache):
```
Y_{i,k} = 1[π_{w_k} solves e_i]            # 只用 K 个探针技能,不含 w_0(M5=a)
p_i     = (1/K) Σ_k Y_{i,k}                # 库中多少比例技能能解 e_i
b_i     = p_i (1 − p_i)                     # SFL 式学习价值(不做加权覆盖)
```
- **boundary 判定用远离饱和的区间,不写精确 0<p<1**:`τ < p_i < 1−τ`;K=4、单次确定 rollout 时 `τ=0.125` → `p∈{0.25,0.5,0.75}` 属 boundary,`p∈{0,1}` 属 near-hard/near-easy 饱和。
- 统计:`r_hard=Pr(p≤τ)`、`r_easy=Pr(p≥1−τ)`、`boundary_rate=Pr(τ<p<1−τ)`、`target_rate=Pr(|p−0.5|≤δ)`(δ≈0.15)、`mean_b`、`valid_rate`、`duplicate_rate`、`w0_success_rate`(仅部署诊断,不进 p_i)。
- **p_i 只用探针库(M5=a)的代价**:w_0 比探针略强,探针前沿的场景对 w_0 可能偏易;**必须监控 w_0 是否随课程持续变强**,停滞则再议是否把 w_0 纳入 p 或调探针强度。

### 组件 8:进化方向(Eurekaverse 式三段反馈,进 prompt)
根据当前 G_t 的 p-分布给 LLM 方向(前置:`valid_rate < v_min` 时优先修合法性,不是第四种方向):
- `r_hard` 高(太多 p≈0)→ **RELAX**:把场景从 p≈0 移向 0.25/0.5(进 band,不是单纯变简单)。
- `r_easy` 高(太多 p≈1)→ **HARDEN**:从 p≈1 移向 0.75/0.5。
- `boundary_rate` 足够 → **PRESERVE_BOUNDARY_AND_DIVERSIFY**:保持难度,改变结构/布局但仍在 band。

### 组件 9:治锁 verifier(单标量 + 饱和逃逸,`learnable_frontier.py` + `candidate_verifier.py`)
LLM 每 cycle 输出 R 个候选生成器 `G̃^(1..R)`(R=3)。验收**根除旧的"多轴非降合取死锁"**:

1. **绝对硬门(只筛 LLM 候选,不与 G_t 比、不 AND 成非降链)**:`code_pass` + `valid_rate ≥ v_min` + `duplicate_rate ≤ d_max`。**当前 G_t 永远作为 incumbent 参选,不被候选硬门淘汰**(它正在用,不应因某轮临时评估 valid_rate 波动被踢);只有饱和逃逸时 G_t 才不再作为可保留 incumbent。**`boundary_rate` 不是硬门**。
2. **单一标量目标 J(防死锁关键——单标量永远有 argmax);优先用离散计数版避免浮点歧义**:
   ```
   primary : boundary_count = #{ τ < p_i < 1−τ }         # boundary_rate = boundary_count / N
   tiebreak: 仅当 boundary_count 相等时,再比 mean_b       # 不写 J = boundary_rate + α·mean_b(不引权重)
   ```
   用 boundary_count/rate 而非 mean_b 作主:mean_b 最大化会塌到"全 p=0.5"尖峰;boundary 计数奖励"多少落在带内",更宽、更抗塌;结构多样由 `duplicate_rate` 兜。纯 `p(1−p)` 口径,不引入覆盖、不做复合分数。
3. **含当前 G_t 的 argmax + 计数净改进**:在 `{G_t} ∪ {通过硬门的候选}` 上取 argmax(先 boundary_count、并列再 mean_b);**替换条件优先用计数版** `cand_boundary_count ≥ Gt_boundary_count + min_boundary_count_delta`(等价 `J(best)>J(G_t)+ε`);否则 **hold**(generator-level reject-and-hold)。
4. **饱和逃逸(easy/hard 对称,根除停摆)**:`saturated = (r_easy ≥ r_easy_max) OR (r_hard ≥ r_hard_max)`。饱和时**当前 G_t 不再阻塞替换**(不用自身 J 挡住),在通过硬门的候选里按 boundary_count 选最优——`r_easy` 高对应 HARDEN escape、`r_hard` 高对应 RELAX escape。饱和逃逸**不是接受坏 generator**(候选仍须过绝对硬门),而是不让饱和的 G_t 卡住进化。方向反馈引导 LLM 往哪走,boundary 计数给结果打分。

**为何结构上根除旧死锁**:旧版 `infeasible_cap_ok AND trivial_not_worse AND boundary_not_worse AND coverage_not_worse` 是"候选须在多轴同时不劣于 G_t"的合取,成熟期顶格→冻死→停摆;新版是①单标量无合取、②硬门是绝对下限非"比 G_t 好"、③库学透触发饱和逃逸强制加难,三处同时断开旧链条。唯一 hold 情形是"G_t 确实最好且未饱和"(正确行为)。

**PRESERVE_AND_DIVERSIFY 换代(点2/点3,默认关 `diversify_on_hold=false`)**:成熟期库变强后,LLM 常产不出更难的合法场景 → 连续 hold → 生成策略被冻住,`PRESERVE_AND_DIVERSIFY` 里的 "DIVERSIFY" 落不了地(每回合仍按 tblock pose 采样,但**生成策略本身不变**)。开启后,在"未饱和 + 无净改进(本应 hold)"时,不冻结 G_t,而是换一个**等难度**候选(过硬门、`boundary_count ≥ cur − diversify_bc_tolerance`、`r_easy ≤ cur.r_easy + diversify_easy_eps` 即**不更易**,护栏防 easy-drift)以轮换生成结构;点3(`diversify_prefer_lower_easy` 且 `cur.r_easy > diversify_r_easy_soft`)进一步在等难度候选里**偏好 r_easy 最低**者,做温和连续 re-harden(避免只在 `r_easy_max=0.8` 硬阈值才 HARDEN 的"死区"静默变易)。优先级低于净改进 replace 与饱和逃逸;无等难度候选则仍 hold。

### 组件 10:候选验证三层 + 训练期使用(复用现有件)
- **Code verifier** = 现有 `SandboxPushTExecutor`(能执行/有指定函数/签名/可复现/超时/禁危险 import/不改 env/reward/success/policy)。
- **Physical verifier**:候选采样 N 个临时布局,查障碍数/大小/在界内/不重叠/不与初始 T-block 接触/`env.reset` 有效/`get_ncon==0`/不过度重复 → `valid_rate`。V0 `v_min` 建议 **≥0.8**(不用 0.60,过低浪费训练重采样)。
- **Boundary verifier** = 组件 7 的 W_probe 探针,算候选的 `boundary_count/boundary_rate/mean_b/target_rate`。
- **两个独立职责的探针(方向探针 vs 验收探针,不强制同源)**:一个 evolve 周期用两个探针:
  - **方向探针(上下文/prompt)**:随机起点 + `invalid_scene_policy='resample'`,只在 VALID 场景上度量库能力,产出 boundary 统计与进化方向进 prompt;`probe_source='w_probe'`。
  - **验收探针(verifier)**:`sample_fixed_starts` 固定起点,G_t 与所有候选在**同一批固定起点上 paired 打分**(`invalid_scene_policy='zero'`);verifier 自行重探 G_t(`current_signal_result=None`),不复用方向探针结果。
  - **为何不强制"双探针同源"(经验教训)**:早期曾把方向探针也对齐成 `fixed_starts + 'zero'` 以求"方向↔验收单次共享",但在硬 G_0 上 `invalid_scene_policy='zero'` 把生成器无效场景(那些起点摆不出合法障碍)误记为"库解不开"(realized=0),使 r_hard 虚高、boundary_count=0 → 冷启动连续 ~17 轮盲目 RELAX → 库学歪,实测部署 D=0.00(AVG 0.49 vs 回退前 0.675)。方向探针改回随机+resample 后方向诚实;方向与验收的轻微采样差异是良性的——验收由 verifier 在自身一致的固定批上做 argmax 拍板,方向只是给 LLM 的软提示。同源的本质需求是"候选与 G_t 考同一批**有效**题"(由验收探针 paired 保证),不需要连无效场景处理都一致。
- **训练期**:接受后**直接从 G_{t+1} 动态采样**(复用 `_make_llm_scene`:cheap physical check + 重采样),**不在训练期跑 W_probe verifier**;W_probe 只在 curriculum cycle 评估生成器。
- **触发**:Eurekaverse 式**固定步数间隔**(`evolve_interval_steps` 30k–50k),而非成功率阈值临时触发;HARDEN 方向兜住饱和。

### 组件 11:第一版建议参数
```
evolve_interval_steps: 30000        candidate_generators_per_cycle (R): 3
eval_layouts_M: 64 (先减半跑通)       eval_layouts_per_candidate: 64
R (candidates/cycle): 3             evolve_interval_steps: 30000
tau_saturation: 0.125               target_delta: 0.15
# 绝对硬门(只筛候选,不含 boundary_rate!):
valid_rate_min (v_min): 0.8         duplicate_rate_max: 0.25
# 净改进(优先用计数版,避免浮点歧义):
min_boundary_count_delta: 4         J_epsilon: 0.05
# 饱和逃逸(easy/hard 对称):
r_easy_max: 0.8                     r_hard_max: 0.8
β_div: 0(V0 关多样性奖励)
```
> 注意:**`boundary_rate` 不是硬门**,只用于 argmax 目标 J 与净改进;硬门只有 `code_pass + valid_rate + duplicate_rate`。

`decide_skill_acceptance` 改为调用新判据;`SkillVerifierConfig` 加 `criterion=boundary_count_argmax_escape / min_boundary_count_delta / J_epsilon / r_easy_max / r_hard_max / tau_saturation / target_delta`(实现用 boundary_count;boundary_rate 仅日志/等价比例)。

## Data Models

### Codebook
- 常量张量 `codebook: FloatTensor[K+1, d_w]`;第一版 one-hot,`d_w = K+1`,索引 0 = w_0(部署),1..K = W_probe。不参与梯度。
- 升级路径:K→8→16(每次重测 K_eff);learnable `w_k = Emb(k)` 留后续。**one-hot 首选理由**:d_w 低时随机连续向量 pairwise 距离不均,相近码易被映射成相似行为,给塌缩排查添噪声;one-hot 天然等距。

### Replay Buffer 扩展
- 现有 `observations / actions / dones` 增加:`skill_id`(0=w_0,1..K)、来源标签 `source ∈ {normal, diversity_paired}`,以及三列奖励 `reward_task / reward_div / reward_total`(不覆盖原单一 reward 字段)。
- `reward_for_critic` 按技能路由:`w_0` 转移用 `reward_task`,`W_probe` 转移用 `reward_total`。
- batch 采样支持按来源分层(`paired_sample_ratio ≤ 0.3`),且 `P(w_0)≥0.5` 按进入 batch 的最终分布控制。
- 采样时同步返回 `skill_id`(可映射出 w)、来源标签与三列奖励,供 mask、critic 路由与调度使用。

### SkillGeneratorScore(扩展)
- 现有:`score(=mean_lv) / frac_infeasible / mean_realized / mean_feasible / mean_deployed / valid_generation_rate / frac_trivial / frac_boundary / coverage_per_scene / difficulty_profile`。
- 新增(Stage 2 V0):`boundary_count / boundary_rate / mean_b / r_hard / r_easy / target_rate`(§组件 7 的 skill-library frontier signal);**不实现 `weighted_coverage_S`**。

### 配置参数(第一版建议默认值)
| 参数 | 含义 | 默认 |
|---|---|---|
| `K` | probe 技能数 | 4 |
| `d_w` | 技能码维度(one-hot) | K+1 |
| `codebook_type` | 技能码类型 | one-hot(固定) |
| `w0_rollout_ratio` | w_0 回合占比(≥ 单 probe) | 待标定 |
| `probe_rollout_ratio` | probe 技能回合总占比 | 待标定 |
| `lambda_cov_all` | all-w coverage ablation 权重 | 0(默认关) |
| `beta_div` | reward_total 多样奖励系数(动态调度) | 起始 1e-3~1e-2 |
| `beta_div_min/max` | β_div bound | [1e-4, 0.1] 待调 |
| `paired_sample_ratio` | batch 中 diversity_paired 上限 | ≤0.3 |
| `rdiv_distribution` | episode r_div→transition 分配 | uniform(默认) |
| `div_margin` (m) | Form B margin 上界 | 待标定 |
| `soft_gate α/β/γ/η` | 轨迹质量 q(τ) 系数 | 待标定 |
| `soft_gate τ_q / T` | soft gate 阈值/温度 | 待标定 |
| `success_floor` | 多样调度成功率下限 | 待标定 |
| `eff_floor` | K_eff 下限 | ~K 的 60~75% |
| `criterion` | verifier 判据(实现用 boundary_count) | `boundary_count_argmax_escape` |
| `min_boundary_count_delta` | 计数净改进阈值 | 4 |
| `J_epsilon` (ε) | 净改进裕度(等价浮点版) | 0.05 |
| `r_easy_max / r_hard_max` | 饱和逃逸阈(对称) | 0.8 / 0.8 |
| `tau_saturation / target_delta` | boundary 带 / 近 0.5 判定 | 0.125 / 0.15 |
| `task_warmup_steps` | 多样项开启前 warm-up | 待标定 |
| `use_dual_critic` | 双 critic(Q_task/Q_total)后备 | False |

> "待标定"参数在 Stage 1 冒烟中确定,不预先拍死。

### 代码改动点映射
| 文件 | 改动 |
|---|---|
| `DIVO/policy/ldpi.py` | 加 `codebook`、`decode_with_skill`、`predict_action_with_skill`;`predict_action` 默认拼 w_0 |
| `DIVO/policy/nets`(decoder) | decoder MLP 首层 `in_chan += d_w` |
| `DIVO/critic/mcritic.py` | `SingleFCCritic.forward(obs, w, a)`、`MultiCritic` 透传 w |
| `DIVO/workspace/rl_workspace/td3_curriculum_workspace.py` | `compute_policy_loss` 改为 `L_exec + L_z`(executed-w mask,无独立可微多样项);critic 用 `reward_for_critic`(w_0→reward_task,probe→reward_total)带 w;buffer 打 skill_id + 三列奖励 + 来源标签;采集技能回合;β_div 调度;K_eff/ActionVar/SatRate 监控 |
| `DIVO/curriculum/skill_signal.py` | probe 注入源改为 W_probe;加 `p_i/b_i/boundary_count` 统计(不做加权覆盖) |
| `DIVO/curriculum/learnable_frontier.py` | 四门合取 → 绝对硬门 + 单标量 boundary_count argmax(含 G_t)+ 对称饱和逃逸 |
| `DIVO/curriculum/candidate_verifier.py` | `decide_skill_acceptance` 调新判据;`SkillVerifierConfig` 加 boundary/饱和字段 |

## Correctness Properties

### Property 1: z 通道正交性
障碍信息只经 z 进 decoder;w 不承载障碍信息。部署时 `z = encoder(obs)` 保持不变。
**Validates: Requirements 1.4**

### Property 2: 部署可比性
部署路径 `a = decoder(cat([state, encoder(obs), w_0]))` 与标准 DIVO 单 z 部署严格可比(单技能、单阶段)。
**Validates: Requirements 9.1, 9.2**

### Property 3: executed-w 一致性
任务损失只在实际执行过对应 w 的转移上更新,critic 不对未执行的 (obs, w) 组合外推。
**Validates: Requirements 3.2**

### Property 4: probe 源一致性(消除 random-z 失配)
probe 注入源 = 训练技能库 W_probe(与训练同一组 w),从架构上消除 random-z / probe-训练失配。
注:"realized 随训练上升"**不是**架构自动保证的性质(依赖技能不塌、per-skill 有能力、分布合适、多样不伤任务),因此列为 Testing Strategy 的核心假设验证,而非 correctness property。
**Validates: Requirements 7.1**

### Property 5: 课程不锁
治锁 verifier 保证成熟期不出现四门顶格棘轮锁;饱和时主动出更难题。
**Validates: Requirements 8.3, 8.4, 8.5**

### Property 6: 多样不塌任务
约束式平衡保证 `avg_success`(基于 reward_task)低于 floor 时降 β_div,多样性提升不以成功率塌陷为代价。
**Validates: Requirements 5.2, 11.6**

### Property 7: 无评测泄漏 + 验证分布固定
best checkpoint 用**固定 `between` 验证分布**上的 `test_mean_score`(w_0 部署)选,不用会随课程漂移的 seen G_t、不用 B/M/U/D(B/M/U/D 只作最终零样本测试)。
**Validates: Requirements 11.1**

## Error Handling

- **w 塌缩(K_eff ≪ K)**:周期性 K_eff 监控;塌缩时按调度上调 β_div;若长期 K_eff 不达标 → Stage 1 go/no-go 判负,回到风险分析,不硬堆组件。
- **多样失败(动作饱和)**:`SatRate` 报警;Form B margin 有界从设计上规避;若 Form A 监控报警说明多样性靠动作边界实现,判为坏信号。
- **部署被拖累**:每 eval 周期对比 w_0 的 B/M/U/D 与标准 DIVO,并监控 `Q(obs, w_0, ·)` 是否异常;若 B/M/U 掉 → 降 β_div、启用 dual critic 或回退。
- **课程卡死**:饱和早停触发出更难题;硬可行门防不可行刷屏。
- **critic 外推**:executed-w-only mask 是主防线;`all-w` 只在 critic 稳定后 warm-up 引入。
- **LLM 生成非法**:沿用现有 `valid_generation_rate` 门与 sandbox 校验。

## Testing Strategy

### 评测口径(严格对齐 DIVO,禁 p-hacking)
- **验证集(选 best)= 固定 `between`**:训练分布是移动的 LLM 课程 G_t,不能拿它选 best(`test_mean_score` 会随课程变难而漂移,把 best 冻在课程早期)。故 validate/选 best 用一个**固定、独立于课程、且不碰 B/M/U/D** 的 `between` 分布(`pusht_mujoco_llm`,num=2,size=0.01,obstacle_dist=between),每 `validate_steps` 在其上跑 w_0 部署 20 eps、`reward='last'` 均值 = `test_mean_score`;`TopKCheckpointManager(mode=max)` 选最高者。与原版 DIVO"固定分布选 best"、Eurekaverse"固定 benchmark 评测"、CurricuLLM"固定 main 目标选 best"同一原则。实现复用 `llm_eval_env`(不再 `set_obstacle_config`)。
- **B/M/U/D**:barrier/maze/u-shape/dead-end 四类泛化成功率(**只作最终零样本测试,SHALL NOT 参与选 best**);在选出的 best ckpt 上跑一次 `evaluation.py`(policy_only + benchmarks)。
- **action diversity**:对标 `evaluation.py::calculate_action_diversity`(500 obs × 采技能 → 动作方差和 → 平均),用 W_probe 的 K 技能替代 stage-2/3 sampler。**DIVO 表 II 的 0.62 是参考上界/对标值,不是 Stage 1 硬门**(其口径来自 stage-2/3 sampler + 每 obs 采 100 技能,机制与 K=4/8/16 codebook 相近但不同);仅作辅助诊断。
- **realized**:闭环验证——随训练真实上升。
- **K_eff**:固定协议 `K_eff = exp(H(P_cluster))`(占用簇数仅日志、不作 go/no-go);K_eff → K 表示未塌缩。
- **within/across 轨迹多样**:复用 `analysis/skill_coverage/measure_coverage_diversity.py`。
- **成功率不掉**:防"多样失败"。
- **D vs evolve 的 0.32**:evolve(归因验收)当前 D 最高;路B 需在 D 上有竞争力且 B/M/U 不掉。

### 可视化(go/no-go 产出)
- **K 技能 T-block 扇形图**:同障碍情形下 K 技能 T-block 轨迹叠图(`motion_decoder/base.py::get_tblock_feature_points` 画角点 + `rollout_fixed_skill` 记轨迹 + `physics.render()` 渲染)。
- **障碍 rollout 叠图**:多技能同布局路径叠加。

### 可直接复用的 DIVO 现成件
- `base_evaluator.measure_diversity`(轨迹唯一格数)。
- `analysis/skill_coverage/measure_coverage_diversity.py`(within/across 轨迹距离)。
- `sampler/fm.py::MMDCalculator(L2_traj)`(轨迹分布距离)。
- `motion_decoder/base.py::get_tblock_feature_points`、`rollout_fixed_skill`(路B 新 API:w 固定、z 每步实时;由旧 `rollout_fixed_z` 记录思路改造)、`physics.render()`。

### 分阶段执行(go/no-go 门)
**Stage 0:LLM 初始生成器 G_0 bootstrap**
- LLM 生成候选 G_0(复用 Stage A/phase0);选择只用 `code_pass + physical validity(valid_rate/duplicate_rate/contact-get_ncon) + 轻量难度散布启发式`。
- **此时无 W_probe,SHALL NOT 用 p_i/b_i/boundary 选 G_0**;Stage 0 只保证 G_0 不"全无效/全极易/全极难"。

**Stage 1:在 G_0 上预训技能库(证/证伪核心因果)**
- 内容:加 w + one-hot K=4 + 独立 w_0 + executed-w 任务损失 + Form B 多样(软门 + 同环境配对) + w-条件 critic + 双 reward 记录。
- **训练分布 `e ~ G_0`(动态采样,`_make_llm_scene`),不用 E_stage1 作主训练分布**(E_stage1 仅 debug/诊断)。
- **核心假设验证**:realized(库 W_probe)是否随训练上升(假设,非保证)。
- **go/no-go 硬门**(Stage 1 尚未接 evolve,不硬要求 D 超 baseline):
  - `B/M/U ≥ baseline − δ`,且 w_0 overall validation 不明显下降;
  - `K_eff ≥ 0.5K`(不塌);
  - per-skill 非平凡:`Progress(w_k)≥τ_p` 且 `库内 Success≥τ_s 比例 ≥ r_s`。
- **p 分布非退化验证**:在**从 G_0 采样的 validation layouts**(可选 held-out LLM 布局)上算 p_i,验证存在高 p / 低 p / 中间 `0<p<1`(不再以手工 E_stage1 分层为准)。
- **观察项(非硬门)**:D 是否提升、action diversity(参考对标)。D 的真正提升放到 Stage 2/3。
- 硬门任一长期不达标 → 回风险分析,不硬撑堆组件。

**Stage 2:接 LLM 库可学性课程(Route B V0)+ 治锁 verifier**
- Stage 1 通过后接入:探针源换 W_probe、`b=p(1−p)` boundary 信号、Eurekaverse 三段方向、单标量 J=boundary_rate 验收 + 饱和逃逸。
- 验证:generator 随库能力边界演化(RELAX/HARDEN/PRESERVE 生效)、**不卡死**(对照旧版 evolve_index=17 停摆)、`boundary_rate` 维持。
- **w_0 是部署诊断与最终评测,不是 generator 验收门**(V0 的 p_i 只用探针、不含 w_0,承认前沿可能错位):只监控 `w0_success_rate` 是否随课程改善;若 probe-boundary 对 w_0 长期偏易或 w_0 停滞,记为 M5 风险、进 V1 决策,**不作为 Stage 2 选择条件**。
- 主对比(核心):**(d) 库信号课程 vs (b) 单策略课程(evolve) vs (c) 库无课程 vs (a) 标准 DIVO**,同协议 B/M/U/D + D,≥3 seeds;红线:(d) 需 > (c) 且 > (b),否则退回或重述。

**Stage 3:完整对比 + 消融**
- 主对比(对齐社区口径):`random / static / eureka(success 引导) / 本方法(库信号引导)`;并含路B vs 路C vs evolve vs 固定基线。
- 内部机制三臂(消融):臂1 库引导(本方法,boundary 信号 `b=p(1−p)` + 单标量治锁)、臂2 单策略引导(复现 evolve 的归因证据分,现成不锁)、臂3 随机地板(仅 valid 门,永不锁)。
- 消融:训库/不训、多样 on/off、单技能/挑技能、K(4/8/16)、β_div、rollout ratio。

## 设计中的诚实保留(不掩盖)
1. **zselect 红旗未消除**:训库→单 z 受益是赌注,Stage 1 go/no-go 检验它;若不成立,路B 主张需重述(可能退化为"多技能作课程出题器"而非"帮部署")。
2. **独立 w_0 的代价**:库帮部署只剩"共享 decoder 变宽"一条间接路径。
3. **多样性早期弱信号**:task warm-up + 动态 β_div + 软 near-success 门是缓解手段,不是保证;go/no-go 不能只看轨迹距离,须同时看 K_eff / per-skill success / progress / p_i 分层 / reward_task success 不塌。
4. **算力**:CPU rollout 贵,扩 K 前必须先在 K=4 见效。
