# Requirements Document

**Feature: Nav Skill-Library Curriculum(第二任务 / 导航版技能库可学性课程)**

## Introduction

本需求文档描述**第二任务(自主导航)**的用户可见需求:把第一任务(Push-T)已建立的"生成器进化 + 隐技能库可学性标尺"框架迁移到 **Safety-Gymnasium 的 `SafetyPointGoal`**(Point 智能体 + Goal 任务),证明该框架**跨任务范式通用**。核心科学主张与 Push-T 一致:单确定性策略每场景 `p∈{0,1}`,SFL 式 `p(1−p)` 信号退化;一族多样技能库把 `p_i=(1/K)Σ_k 1[π_{w_k} solves e_i]` 重新连续化,让课程信号复活。设计细节见同目录 `design.md`,本文档只规定"系统必须做到什么"及其可验证判据。

范围与边界(与 design.md 一致):
- 只做 **SafetyPointGoal + Point 智能体**,用 **Pillars**(可碰撞刚体)作难度主元;不换 Car/Ant。
- 部署严格**单技能、单阶段**:`a = decoder([state, encoder(obs), w_0])`,与单策略 TD3 baseline 可比。
- **障碍信息只经 z 进 decoder**(`z=encoder(pillars_lidar)`);`w` 只承载打法/路径,不承载障碍信息。
- **目标(goal)固定、起点(agent start)采样**(对齐 Push-T:target 固定、tblock 起点采样)。
- 全程 **LLM 驱动闭环**(Stage 0 起):Stage 0(LLM 生成 `generate_pillars` 的 G_0)→ Stage 1(在 G_0 上预训 w_0+W_probe,验证信号复活)→ Stage 2(用 W_probe 的 `p(1−p)` boundary 信号周期进化 G_t + 单标量治锁 verifier)→ Stage 3(完整对比 + 消融)。**不设非 LLM 预备阶段**(核心因果已在 Push-T 验证)。
- 独立运行环境 `safenav`(safety-gymnasium 1.0.0 / gymnasium 0.28.1 / mujoco 2.3.3 / torch 2.3.0+cu118),与 Push-T 的 `divo` 隔离。

关键风险已在 design.md 记录(H1/H2 库帮不帮部署的赌注、w 塌缩、信号复活是假设、LLM 基建移植成本、算力),本文档的 go/no-go 判据即为对这些风险的可验证约束。

## Glossary

- **NavEnvAdapter**:导航运行时适配层(`nav/nav_adapter.py`);`set_layout(pillars,start,goal)` → `reset` → `step`,`obs2state`/`obs2zinput`/`success`;柱数可变、obs 恒 44;非法布局抛 `ResamplingError`。
- **obs 切分**:44 维 = `[本体+goal_lidar(0:28) | pillars_lidar(28:44)]`;`state=obs[0:28]`、`z 通道输入=obs[28:44]`。
- **z / encoder**:`z=encoder(obs[28:44])`,确定性 MLP 编码(非采样),情形/障碍编码,decoder 唯一障碍通道。
- **w / 技能码**:新增打法/路径码;第一版 one-hot,`d_w=K+1`。
- **Codebook**:常量张量 `[K+1, d_w]`,索引 0 = `w_0`(部署),1..K = `W_probe`(探针);第一版不参与梯度。
- **w_0 / W_probe**:部署技能(独立)/ 探针技能库 `{w_1..w_K}`。
- **Skill-Conditioned Policy / Critic**:`a=decoder([state,z,w])` / `Q(obs,w,a)`。
- **executed-w-only**:actor 损失只对该转移实际执行过的 w 更新(按 buffer w 标签 mask)。
- **realized / p_i / b_i**:场景 e_i 内 `p_i=(1/K)Σ_k 1[w_k 到达 goal]`(只用探针,不含 w_0)、`b_i=p_i(1−p_i)`。
- **boundary_count/rate**:`#{τ<p_i<1−τ}`(K=4 → τ=0.125)。
- **reward_task / reward_total**:原生 dense 任务奖励 / `reward_task + β_div·reward_div`。
- **Form B 多样奖励**:任务坐标系轨迹嵌入、软 near-success 门控、同环境配对、margin 有界的 `r_div`。
- **generate_pillars**:LLM 生成器契约 `generate_pillars(agent_start, goal, num) -> [(x,y)]`;主训练分布 = G_t 动态采样。
- **验证分布(between)**:DIVO `between` 范式的导航版——先放固定中等难度 pillar(num≈2),再把 agent 起点采在"某 pillar 与 goal 之间",goal 固定;仅用 w_0 部署选 best。
- **B/M/U/D**:参数化 pillar 类别(num/size/shape/dist)+ 均匀随机起点 + 20 回合平均的零样本泛化基准(照搬 DIVO `config/evaluation/eval.yaml` 口径)。
- **结构化拓展套件(TEMPLATES)**:手工 barrier/maze/u/dead-end(`nav/nav_env.py::TEMPLATES`);**仅作可选 eval 拓展基准**,别处一律不用。
- **治锁 verifier(V0)**:绝对硬门(code/valid/duplicate)+ 单标量 `boundary_count`(mean_b tiebreak)在 {G_t+候选} argmax + 对称饱和逃逸。

## Requirements

### Requirement 1: 技能条件确定性策略与技能码 codebook

**User Story:** 作为研究者,我要给导航的确定性策略新增技能码 w 通道,让同一障碍情形下不同 w 产生不同路径,同时保持 z 只承载障碍信息,以对齐 Push-T 的隐技能库结构。

#### Acceptance Criteria

1. THE Skill-Conditioned Policy SHALL 以 `a = decoder(cat([state, z, w]))` 解码动作,其中 `state = obs[0:28]`、`z = encoder(obs[28:44])`、`w` 取自 Codebook;动作维度为 2、tanh 有界。
2. THE `z` SHALL 为对 `obs[28:44]`(pillars_lidar)的**确定性** MLP 编码(非随机采样),保持"单确定性策略"的 motivation。
3. THE `w` 通道 SHALL NOT 承载任何障碍/情形信息(障碍信息只经 `z` 进入 decoder)。
4. THE Codebook SHALL 为形状 `[K+1, d_w]` 的常量张量,索引 0 对应 `w_0`、索引 1..K 对应 `W_probe`;第一版为 one-hot 且不参与梯度。
5. THE 策略 SHALL 提供 `predict_action(obs)` 默认拼接 `w_0` 部署,并提供 `predict_action_with_skill(obs, w)` / `decode_with_skill(state, z, w)` 供训练与 probe 使用。
6. WHERE `K` 由配置提供,THE 系统 SHALL 支持 `K ∈ {4, 8, 16}`,第一版默认 `K=4`。

### Requirement 2: 技能条件 critic

**User Story:** 作为研究者,我要 critic 以 w 为条件,使每个技能价值被单独估准。

#### Acceptance Criteria

1. THE Skill-Conditioned Critic SHALL 计算 `Q(obs, w, a)`,内部 `q(cat([obs, w, a]))`;twin critic 均以 w 为条件。
2. WHEN 计算目标 Q 值,THE 系统 SHALL 使用与该转移一致的 `w` 标签。
3. WHEN 计算 critic 损失,THE 系统 SHALL 使用 buffer 中每条转移记录的 `w` 标签,而非默认单一 w。

### Requirement 3: Actor 损失(executed-w-only)

**User Story:** 作为研究者,我要 actor 损失只更新实际执行过的技能,避免 critic 对未执行 (obs,w) 外推。

#### Acceptance Criteria

1. THE Stage 1 actor 损失 SHALL 为 `L_actor = L_exec` (可选加 `L_z`)。
2. THE `L_exec` SHALL 为 `-E_{(obs, w_exec)}[ Q(obs, w_exec, D(s, E(obs), w_exec)) ]`,`w_exec` 只取该转移实际执行过的技能(按 buffer w 标签 mask)。
3. THE Stage 1 actor 损失 SHALL NOT 包含独立的 all-w coverage 主项;critic SHALL NOT 对该 obs 上从未执行过的 (obs, w) 组合被 actor 优化。
4. WHERE 保留 `L_z`,THE `L_z` SHALL 为对 `z` 的 gaussian 正则(batch 均值/方差拉向 N(0,1)),挂在部署路径。

### Requirement 4: 多样性目标(Form B;V0 默认关)

**User Story:** 作为研究者,我要多样性度量"路径多样"而非"动作多样"或"多样失败",且早期成功稀少时仍有分化信号。

#### Acceptance Criteria

1. THE 多样性 SHALL 基于**任务坐标系**(start→goal 平移+旋转+按 |goal−start| 归一化)下的 agent (x,y) 轨迹嵌入 `φ`,使不同场景可比。
2. THE 多样性距离 SHALL 只在**同一环境布局**内的技能对之间计算(同批 seeds 枚举 w_1..w_K),不得跨布局比较。
3. THE 多样性奖励 SHALL 使用软 near-success 门控 `q(τ)=α·success+β·progress−γ·cost_pillars−η·timeout`、`g=σ((q−τ_q)/T)`,并按 margin 有界的 `r_div` 计算;SHALL NOT 使用硬 `1[success]` 作为唯一门。
4. THE 系统 SHALL 分别记录 `reward_task`(原生 dense)与 `reward_total = reward_task + β_div·reward_div`;`w_0` 回合 SHALL 有 `reward_total = reward_task`(不加 r_div)。
5. THE 所有成功率、go/no-go、B/M/U/D 判定 SHALL 只基于 `reward_task` 与真实 success(到达 goal),SHALL NOT 使用 `reward_total`。
6. THE Form A(动作方差 ActionVar / 饱和率 SatRate)SHALL 仅作日志监控与塌缩探测,SHALL NOT 作为多样性训练目标。
7. THE V0 默认配置 SHALL 令 `β_div = 0`(关多样性),多样性为 Stage 1 可选机制。

### Requirement 5: β_div 约束式调度(V0 默认不启用)

**User Story:** 作为研究者,我要多样性在"成功率不塌"约束下最大化,权重按评测周期自适应。

#### Acceptance Criteria

1. THE 系统 SHALL 提供 task warm-up(`task_warmup_steps` 内 `β_div=0`,仅训 `L_exec`(+`L_z`)),warm-up 后再以小 `β_div` 开启。
2. WHEN `avg_success < success_floor`,THE 系统 SHALL 执行 `β_div *= 0.5`。
3. WHEN `avg_success ≥ success_floor` 且 `K_eff < eff_floor`,THE 系统 SHALL 执行 `β_div *= 1.2`。
4. THE `β_div` SHALL 被裁剪到 `[β_div_min, β_div_max]`,且按 eval 周期(慢调)调整。
5. THE `avg_success`/`success_floor` 判定 SHALL 基于 `reward_task`/真实 success。

### Requirement 6: 技能回合采集与 Replay Buffer 标签

**User Story:** 作为研究者,我要采集覆盖所有技能的回合并打技能标签,且部署技能采样强度不被 probe 库稀释。

#### Acceptance Criteria

1. THE 系统 SHALL 采集 `w_0` 与每个 `w_k ∈ W_probe` 的技能回合;每条转移在 buffer 记录其 `skill_id`。
2. THE buffer SHALL 增列 `skill_id / reward_task / reward_div / reward_total / source`,采样时同步返回。
3. THE `reward_for_critic` SHALL 按技能路由:`w_0` 转移用 `reward_task`,`W_probe` 转移用 `reward_total`。
4. THE `w_0` 的普通训练回合采样占比 SHALL 默认 `P(w_0) ≥ 0.5`,其余 0.5 在 `w_1..w_K` 间均匀分配;第一版 SHALL NOT 让 `P(w_0) < 0.5`。

### Requirement 7: 库可学性课程信号(信号复活)

**User Story:** 作为研究者,我要用训练技能库的 realized(而非随机 z)作为课程信号,并**在生成器采样场景上**验证"单策略退化、库复活"。

#### Acceptance Criteria

1. THE probe 注入源 SHALL 为训练技能库 `W_probe`(与训练同一组 w),而非随机 z_bank。
2. THE 每场景信号 SHALL 为 `p_i=(1/K)Σ_k 1[π_{w_k} 到达 goal(e_i)]`(**只用探针,不含 w_0**)与 `b_i=p_i(1−p_i)`;boundary 判定 `τ<p_i<1−τ`(K=4 → τ=0.125)。
3. THE 探针 SHALL 在**同一 scene e_i 内让 K 技能共享同一 start**(paired),使 `p_i` 的分数性只来自技能多样性;SHALL NOT 通过对单策略在同一 scene 跨多起点平均得到分数型 p。
4. THE 信号复活验证 SHALL 在**生成器采样场景(G_t / 训练 DOF / 参数化分布)**上进行:单确定性策略 `p_i∈{0,1}`(boundary_count≈0),技能库 `p_i` 出现中间值(boundary_count>0)。SHALL NOT 用结构化拓展套件(TEMPLATES)度量此性质。
5. THE 系统 SHALL 记录 `r_hard/r_easy/boundary_rate/target_rate/mean_b/valid_rate/duplicate_rate/w0_success_rate`(w0 仅部署诊断,不进 p_i)。
6. THE 每个 evolve 周期 SHALL 使用两个独立职责的探针(照搬 Push-T),SHALL NOT 强制"方向与验收单次共享同一批起点":
   - 6.a 方向探针(prompt):随机起点 + `invalid_scene_policy='resample'`,只在 VALID 场景度量库能力;`probe_source='w_probe'`。
   - 6.b 验收探针(verifier):固定起点(paired),G_t 与所有候选在**同一批固定起点上 paired 打分**(`invalid_scene_policy='zero'`);verifier 自行重探 G_t,SHALL NOT 复用方向探针结果。

### Requirement 8: 单标量治锁 verifier(V0,Stage 2)

**User Story:** 作为研究者,我要课程验收用单标量目标(而非多轴非降合取)选择,杜绝棘轮死锁,并在库学透时对称加/减难。

#### Acceptance Criteria

1. THE 验收判据 SHALL NOT 使用任何"候选须在多指标上同时不劣于 G_t"的合取。
2. THE 绝对硬门 SHALL 仅为 `code_pass` + `valid_rate ≥ v_min` + `duplicate_rate ≤ d_max`(只筛 LLM 候选);`boundary_rate` SHALL NOT 作为硬门;当前 G_t SHALL 永远作为 incumbent 参选(除饱和逃逸外)。
3. THE 单一标量目标 SHALL 为 `boundary_count`(并列以 `mean_b` 打破);在 `{G_t}∪{通过硬门候选}` 上取 argmax。
4. THE 替换条件 SHALL 为 `cand_boundary_count ≥ Gt_boundary_count + min_boundary_count_delta`;否则 hold。
5. WHEN `saturated = (r_easy ≥ r_easy_max) OR (r_hard ≥ r_hard_max)`,THE 系统 SHALL 令 G_t 不再阻塞替换,在通过硬门候选中按 `boundary_count` 选最优(候选仍须过硬门)。
6. THE 系统 SHALL 复用 Push-T 的验收判据逻辑(`learnable_frontier`/`candidate_verifier` 的 env-agnostic 部分,抽为共享或轻量重实现),探针执行换成 NavEnvAdapter。

### Requirement 9: 单技能部署

**User Story:** 作为研究者,我要部署严格单技能单阶段,与单策略 baseline 可比。

#### Acceptance Criteria

1. THE 部署 SHALL 执行 `a = decoder([state, encoder(obs), w_0])`(固定 `w_0`,单技能、单阶段)。
2. THE 部署 SHALL NOT 使用任何多技能推理或 best-of-K 作为主结果;best-of-K 仅作 ablation。

### Requirement 10: Stage 0 LLM 初始生成器 + Stage 1 go/no-go

**User Story:** 作为研究者,我要整条流程全程 LLM 驱动:先用 LLM 生成 `generate_pillars` 的 G_0,再在 G_0 上预训技能库,并在生成器采样场景上做 go/no-go。

#### Acceptance Criteria

1. THE 系统 SHALL 在技能库训练前用 LLM 生成初始生成器 G_0(契约 `generate_pillars(agent_start, goal, num) -> [(x,y)]`),SHALL NOT 设非 LLM 预备阶段作为主线。
2. THE G_0 SHALL 通过 code/physical 检查(code_pass + keepout/在界内/离 start&goal 净空/`reset` 有效不抛 ResamplingError/不平凡围死 goal + 轻量难度散布);**此时无 W_probe,SHALL NOT 用 p/b/boundary 选 G_0**。
3. THE Stage 1 训练分布 SHALL 从 G_0 动态采样(先采起点、再 `generate_pillars(start,goal,num)`);`nav_env.sample_training_layout` 仅作代码冒烟/单测,不作主训练分布。
4. THE Stage 1 go/no-go 硬门 SHALL 包括:
   - 4.a **信号复活**:生成器采样场景上单策略 `p_i∈{0,1}`(boundary_count≈0)、库 `p_i` 出现中间值(boundary_count>0)。
   - 4.b `K_eff ≥ 0.5K`(不塌);per-skill 非平凡:`Progress(w_k) ≥ τ_p` 且 `(1/K)Σ_k 1[Success(w_k)≥τ_s] ≥ r_s`(`r_s∈[0.5,0.7]`)。
   - 4.c w_0 部署在验证分布 SHALL NOT 明显低于单策略 baseline;B/M/U SHALL NOT 明显下降。
5. THE Stage 1 SHALL NOT 把 "D ≥ baseline" 作为硬门(D 观察项)。
6. WHEN 任一硬门长期不达标,THE 流程 SHALL 回到风险分析,SHALL NOT 继续堆叠组件。

### Requirement 11: 评测与度量(对齐 DIVO,禁 p-hacking)

**User Story:** 作为研究者,我要评测口径与 DIVO/Push-T 逐项一致,best checkpoint 选择不泄漏。

#### Acceptance Criteria

1. THE best checkpoint SHALL 用**固定验证分布(DIVO `between` 范式)上的 w_0 部署指标**选,SHALL NOT 用会随课程漂移的 G_t,SHALL NOT 用 B/M/U/D 或结构化拓展套件选。
   - 1.a THE 验证分布 SHALL 为"先放 num≈2 pillar、再把 agent 起点采在某 pillar 与 goal 之间"(goal 固定),独立于 G_t 且与 B/M/U/D 不重叠;`n_env_validate=20`、`validate_steps=5000`、每回合取末步指标作 `test_mean_score`、`TopKCheckpointManager(mode=max, k=10)`。
   - 1.b THE 四格对比(a/b/c/d)与所有 seed SHALL 共用同一验证分布选 best。
2. THE B/M/U/D SHALL 为**参数化 pillar 类别 + 均匀随机起点 + 20 回合平均**(goal 固定),照搬 DIVO `config/evaluation/eval.yaml` 口径,只作最终零样本测试、不参与选 best;nav 映射(size 待标定):B=1 大 pillar、M=3 pillar、U=1 不规则障碍、D=3 动态障碍(gremlins,可选真动态)。
3. THE 结构化拓展套件(TEMPLATES)SHALL 仅作 B/M/U/D 之外的**一个可选零样本 eval 拓展基准**报告,SHALL NOT 用于训练/信号探针/motivation/选 best/多样性度量。
4. THE 系统 SHALL 报告 K_eff(固定协议 `K_eff=exp(H(P_cluster))`,task-frame 轨迹嵌入聚类,仅监控不塌)、within/across 轨迹多样、realized 随训练变化。
5. THE 系统 SHALL 产出可视化:同布局 K 技能 agent 路径叠图、B/M/U/D 或拓展套件 rollout 叠图。
6. WHEN 多样性提升,THE 成功率(基于 reward_task)SHALL NOT 塌陷(防"多样失败")。

### Requirement 12: 对比实验设计(Stage 2/3)

**User Story:** 作为研究者,我要主对比对齐社区口径并保留内部机制消融;大规模对照集中在 Stage 2/3。

#### Acceptance Criteria

1. THE Stage 2 主对比 SHALL 为:(d) 库信号课程 / (b) 单策略课程 / (c) 库无课程 / (a) 单策略 baseline,同协议 B/M/U/D、≥3 seeds、共用同一验证分布选 best;红线:(d) 需 > (c) 且 > (b),否则退回或重述。
2. THE 四格对比的"有课程"臂(b/d)SHALL 都从**同一个 LLM 生成的 G_0** 起步再 evolve(仅信号源不同),"无课程"臂(a/c)固定同一初始分布,避免把"换了训练分布"误算成课程收益。
3. THE Stage 3 主对比 SHALL 对齐口径 `random / static / eureka(success 引导) / 本方法(库信号引导)`。
4. THE Stage 3 消融 SHALL 覆盖:训库/不训、多样 on/off、单技能/挑技能、K∈{4,8,16}、β_div、rollout ratio;并把最难的 B/M/U/D 类作为重点观察。

### Requirement 13: NavEnvAdapter 与运行时约定

**User Story:** 作为使用现有原型的研究者,我要环境适配层支持课程/探针/验证所需的布局与起点模式,且行为可复现。

#### Acceptance Criteria

1. THE NavEnvAdapter SHALL 提供 `set_layout(pillars, start, goal)` → `reset(seed)` → `step(action)`,`obs2state`/`obs2zinput`/`success`,且柱数在两次 reset 间可变、obs 维度恒 44。
2. WHEN 布局非法(keepout/净空/`reset` 失败),THE NavEnvAdapter SHALL 抛 `ResamplingError`,供调用方重采样并计入 `valid_rate`。
3. THE NavEnvAdapter SHALL 支持两种起点模式:
   - 3.a between 起点采样(验证分布):先固定 pillar,再按 between 规则采 agent 起点(某 pillar 落在 start→goal 之间),goal 固定。
   - 3.b 探针/训练起点:每个 scene 采一个 start(方向探针随机+resample;验收探针 fixed_starts paired+zero),同 scene 内 K 技能共享该 start。
4. THE 系统 SHALL 由配置种子派生 RNG,给定相同种子与配置产生一致结果。
5. THE 系统 SHALL 记录每轮的 `reward_task/reward_total`、K_eff、ActionVar、SatRate、boundary_count/boundary_rate/mean_b,严格区分任务与多样性增广回报。
