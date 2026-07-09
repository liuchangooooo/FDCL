# Requirements Document

**Feature: Skill-Library Curriculum(路B / 技能库可学性课程)**

## Introduction

本需求文档描述**路B**的用户可见需求:在 DIVO(隐策略 π(a|s,z) 确定性 stage-1)基础上,新增**技能码 w** 训练出一族多样且都近最优的技能库,并以**技能库可学性(realized)**为课程信号驱动 LLM 障碍课程。设计细节见同目录 `design.md`,本文档只规定"系统必须做到什么"及其可验证判据。

范围与边界(与 design.md 一致):
- 只做 **Push-T**;部署严格**单技能、单阶段**(`z = encoder(obs)` + 固定技能码 `w_0`),不接 DIVO stage-2/3。
- 障碍信息只经 `z` 进 decoder;`w` 只承载"打法/技能",不承载障碍信息。
- 分阶段推进(全程 LLM 驱动闭环):**Stage 0**(LLM 生成初始障碍生成器 G_0,复用现有 Stage A/phase0)→ **Stage 1**(在 G_0 动态采样的分布上预训 w_0 + W_probe,证/证伪核心因果)→ **Stage 2**(用 W_probe 的 `p(1−p)` boundary 信号周期性进化 G_t + 单标量治锁 verifier)→ **Stage 3**(完整对比 + 消融)。
- **固定难度阶梯 E_stage1 仅作 debug/诊断,不作主训练分布或方法组件**;主训练分布来自 LLM 生成的 G_0/G_t。
- 技能库可学性是唯一核心创新点;不为讲故事堆无测量支撑的组件。

关键风险已在 design.md 记录(zselect 红旗、w 塌缩、早期弱多样信号、干扰部署、算力),本文档的 go/no-go 判据即为对这些风险的可验证约束。

## Glossary

- **LatentDetPolicy**:现有确定性隐策略 `DIVO/policy/ldpi.py`;`predict_action(obs)` 执行 `a = decoder(cat([obs2state(obs), encoder(obs)]))`。
- **z / encoder**:`z = encoder(obs)`,情形/障碍编码,decoder 的唯一障碍通道。
- **w / 技能码**:新增打法码;第一版 one-hot,`d_w = K+1`。
- **Codebook**:常量张量 `[K+1, d_w]`,索引 0 = `w_0`(部署技能),1..K = `W_probe`(探针技能库);第一版不参与梯度。
- **w_0**:部署技能,独立,不放入 `W_probe`;部署时固定使用。
- **W_probe**:探针技能库 `{w_1..w_K}`,用于度量可学性与多样性。
- **Skill-Conditioned Policy**:改造后策略,`a = decoder(cat([state, z, w]))`。
- **Skill-Conditioned Critic**:改造后 critic,`Q(obs, w, a)`(w-条件)。
- **executed-w-only**:任务损失只对该转移实际执行过的技能 w 更新(按 buffer 的 w 标签 mask)。
- **realized**:一个场景中 `W_probe` 内技能成功比例;`lv = realized·(1−realized)`。
- **K_eff**:有效技能数(周期多技能 probe 按轨迹聚类估得)。
- **Form B 多样奖励**:轨迹级、软 near-success 门控、同环境配对、margin 有界的多样性回报 `r_div`。
- **reward_task / reward_total**:纯任务奖励 / `reward_task + β_div·reward_div`。
- **库 boundary 信号(Stage 2 V0)**:`p_i=(1/K)Σ_k 1[π_{w_k} solves e_i]`(只用探针)、`b_i=p_i(1−p_i)`、`boundary_count/rate = #{τ<p_i<1−τ}`。不做加权覆盖 S。
- **治锁 verifier(V0)**:绝对硬门(code/valid/duplicate)+ 单标量 `J=boundary_count`(mean_b tiebreak)在 {G_t+候选} 上 argmax + 对称饱和逃逸(r_easy/r_hard)。
- **G_0 / G_t**:LLM 生成的障碍生成器(Stage 0 起题、Stage 2 周期重写),契约 `generate_obstacles(tblock_pose, num_obstacles)`;主训练分布 = G_t 动态采样。
- **难度阶梯 E_stage1**(降级):`{easy, two-route, ...}` 手工阶梯,**仅 debug/诊断用**,非主训练分布。
- **G_0 validation layouts**:从 G_0 采样的临时验证布局(用完即丢),用于 Stage 1 go/no-go 的 p/b/boundary 评估。
- **B/M/U/D**:barrier / maze / u-shape / dead-end 四类泛化场景成功率。
- **SkillVerifierConfig**:`DIVO/curriculum/candidate_verifier.py` 现有验收配置。

## Requirements

### Requirement 1: 技能条件策略与技能码 codebook

**User Story:** 作为研究者,我要给确定性隐策略新增一个技能码 w 通道,让同一障碍情形下能由不同 w 产生不同解法,同时保持 z 只承载障碍信息,以便建立"技能=(w 诱导的行为)"。

#### Acceptance Criteria

1. THE Skill-Conditioned Policy SHALL 以 `a = decoder(cat([state, z, w]))` 解码动作,其中 `state = obs2state(obs)`、`z = encoder(obs)`、`w` 取自 Codebook。
2. THE decoder 输入维度 SHALL 由 `dim(state)+dim(z)` 增加到 `dim(state)+dim(z)+d_w`。
3. THE Codebook SHALL 为形状 `[K+1, d_w]` 的常量张量,索引 0 对应 `w_0`、索引 1..K 对应 `W_probe`;第一版为 one-hot 且不参与梯度。
4. THE `w` 通道 SHALL NOT 承载任何障碍/情形信息(障碍信息只经 `z` 进入 decoder)。
5. THE LatentDetPolicy SHALL 提供 `predict_action(obs)` 默认拼接 `w_0` 部署,保持与现有调用签名兼容,并提供 `predict_action_with_skill(obs, w)` / `decode_with_skill(state, z, w)` 供训练与 probe 使用。
6. WHERE `K` 由配置提供,THE 系统 SHALL 支持 `K ∈ {4, 8, 16}`,第一版默认 `K=4`。

### Requirement 2: 技能条件 critic

**User Story:** 作为研究者,我要 critic 以 w 为条件,使每个技能的价值被单独估准,避免跨技能价值混淆。

#### Acceptance Criteria

1. THE Skill-Conditioned Critic SHALL 计算 `Q(obs, w, a)`,内部 `q_net(cat([obs, w, a], dim=1))`;MultiCritic SHALL 对所有子 critic 透传 `w`。
2. WHEN 计算目标 Q 值(`compute_next_q_value`),THE 系统 SHALL 使用与该转移一致的 `w` 标签。
3. WHEN 计算 critic 损失(`compute_critic_loss`),THE 系统 SHALL 使用 buffer 中每条转移记录的 `w` 标签,而非默认单一 w。

### Requirement 3: Actor 损失(executed-w-only)

**User Story:** 作为研究者,我要 actor 损失严格只更新实际执行过的技能,避免 critic 对未执行 (obs, w) 组合外推把 actor 带偏;并且命名不得暗示"纯任务价值",因为 probe 技能的 Q 已含多样性回报。

#### Acceptance Criteria

1. THE Stage 1 actor 损失 SHALL 为 `L_actor = L_exec + L_z`。
2. THE `L_exec` SHALL 为 `-E_{(obs, w_exec)}[ Q(obs, w_exec, D(s, E(obs), w_exec)) ]`,其中 `w_exec` 只取该转移实际执行过的技能(`w_exec ∈ {w_0, w_1..w_K}`,按 buffer 的 w 标签 mask)。
3. THE 损失 SHALL 命名为 `L_exec`(而非 `L_task`),因为在单 critic 下 probe 技能的 `Q(obs, w_k, ·)` 已包含多样性回报(reward_total),`L_exec` 优化的是 total return 而非纯任务价值。
4. THE 多样性 SHALL NOT 作为独立可微 actor 项进入 `L_actor`;多样性只经 reward_total 进入 Q(见 Requirement 4)。Form A(动作方差)只作监控。
5. THE Stage 1 actor 损失 SHALL NOT 包含独立的 all-w L_cov 主项。
6. THE `L_z` SHALL 保留 DIVO 现有 gaussian z 正则(把 z 的 batch 均值/方差拉向 N(0,1))并挂在部署路径。
7. WHERE `lambda_cov_all > 0`,THE 系统 SHALL 支持可选 all-w coverage 项 `L_cov-all = -(1/K)Σ_k Q(obs, w_k, D(obs, w_k))`,且其默认值为 0,仅作为 critic 稳定后从 0 warm-up 的 ablation,不作为 Stage 1 主线。

### Requirement 4: 多样性目标(Form B:软门、同环境配对、margin 有界)

**User Story:** 作为研究者,我要多样性度量的是"解法多样"而非"动作多样"或"多样失败",且早期成功稀少时仍有分化信号。

#### Acceptance Criteria

1. THE 多样性 SHALL 基于**任务坐标系**(SE2 归一化 / task_frame_waypoints)下的轨迹嵌入 `φ`,使不同场景可比。
2. THE 多样性距离 SHALL 只在**同一环境 seed** `e_i` 内的技能对之间计算(`diversity_probe_batch`:同批 seeds 枚举 w_1..w_K),不得跨环境比较。
3. THE 多样性奖励 SHALL 使用软 near-success 门控:`q(τ)=α·success+β·progress−γ·collision−η·timeout`、`g(τ)=σ((q(τ)−τ_q)/T)`,并按 `r_div(i,k)=g_{i,k}·[Σ_{l≠k} g_{i,l}·min(d(φ_{i,k},φ_{i,l}), m)]/[Σ_{l≠k} g_{i,l}+ε]` 计算;SHALL NOT 使用硬 `1[success]` 作为唯一门。
4. THE 多样性距离 SHALL 使用 margin `m` 有界(距离超过 m 后不再加分)。
5. THE 系统 SHALL 分别记录 `reward_task` 与 `reward_total = reward_task + β_div·reward_div`。
6. THE 单 critic 在默认配置下 SHALL 学习 `reward_total`,其中对 `w_0` 有 `reward_total = reward_task`(w_0 回合不加 r_div,使 `Q(obs, w_0, ·)` 为纯任务价值),对 probe 技能有 `reward_total = reward_task + β_div·reward_div`。
7. THE 所有成功率、go/no-go、B/M/U/D 判定 SHALL 只基于 `reward_task` 与真实 success,SHALL NOT 使用 `reward_total`。
8. WHEN `w_0` 部署指标被拖累,THE 系统 SHALL 支持启用双 critic 或降低 `β_div` 作为缓解手段;`use_dual_critic` 默认 False。
9. THE 系统 SHALL 提供专用的 paired diversity rollout 例程:在**同一批环境 seeds** 上评估所有 probe 技能 `w_1..w_K`,计算轨迹级 `r_div`,并将对应 transition 以 `reward_total` 与来源标签写入 replay buffer(可与普通 rollout 共用 buffer,但需可区分来源)。
10. THE 普通训练采集 SHALL 每回合执行一个 `w`,存 `(obs, w, a, reward_task, next_obs, done)` 供 TD3 主训练;diversity paired 采集按周期进行,不必每 step。
11. THE Form A(动作方差 `ActionVar` 与饱和率 `SatRate`)SHALL 仅用作日志监控与塌缩探测,SHALL NOT 作为多样性训练目标。

### Requirement 5: 约束式平衡与 β_div 调度

**User Story:** 作为研究者,我要多样性在"成功率不塌"的约束下最大化,权重按评测周期自适应而非人工死调。多样性经 `reward_total` 进 critic,故被调度的是 `β_div`(不存在独立可微多样项,故无 λ_div)。

#### Acceptance Criteria

1. THE 系统 SHALL 提供 task warm-up 阶段(`task_warmup_steps` 内 `β_div=0`,仅训 `L_exec` + `L_z`),warm-up 结束后再以小 `β_div` 开启多样(经 reward_total)。
2. WHEN `avg_success < success_floor`,THE 系统 SHALL 执行 `beta_div *= 0.5`。
3. WHEN `avg_success ≥ success_floor` 且 `K_eff < eff_floor`,THE 系统 SHALL 执行 `beta_div *= 1.2`。
4. THE `beta_div` SHALL 被裁剪到 `[beta_div_min, beta_div_max]`。
5. THE `β_div` 调整 SHALL 按 eval 周期(慢调)进行,而非每个 train step。
6. THE `avg_success` 与 `success_floor` 判定 SHALL 基于 `reward_task / success`,SHALL NOT 使用 `reward_total`。
7. THE 名称 `λ` SHALL 仅保留给未来可选的 Form A 可微项(`λ_A`);Stage 1 SHALL NOT 使用 `λ_div`。

### Requirement 6: 技能回合采集与 Replay Buffer 标签

**User Story:** 作为研究者,我要采集覆盖所有技能的回合并给每条转移打上技能标签,且部署技能采样强度不被 probe 库稀释。

#### Acceptance Criteria

1. THE 系统 SHALL 采集 `w_0` 与每个 `w_k ∈ W_probe` 的技能回合。
2. THE 每条转移 SHALL 在 buffer 中记录其执行技能标签(`skill_id` 或 `w` 向量)。
3. THE buffer 采样 SHALL 与 obs/action/reward/done 同步返回对应 `w` 标签。
4. THE `w_0` 的普通训练回合采样占比 SHALL 默认 `P(w_0) ≥ 0.5`,其余 0.5 在 `w_1..w_K` 间均匀分配(`P(w_k) = 0.5/K`),以防单一部署技能被 probe 库稀释(K+1 均分时 w_0 仅占 1/(K+1))。
5. WHERE 后续发现 probe 技能过弱,THE 系统 SHALL 允许调高 probe 采样占比;但第一版 SHALL NOT 让 `P(w_0) < 0.5`。

### Requirement 7: 库可学性课程信号

**User Story:** 作为研究者,我要用训练技能库的 realized(而非随机 z)作为可学性信号,以每场景 `b=p(1−p)` 的 boundary 判据引导课程(Stage 2 V0,不做加权覆盖)。

#### Acceptance Criteria

1. THE probe 注入源 SHALL 为训练技能库 `W_probe`(与训练用同一组 w),而非随机 z_bank。
2. THE 每场景信号 SHALL 为 `p_i=(1/K)Σ_k 1[π_{w_k} solves e_i]`(**只用探针,不含 w_0**)与 `b_i=p_i(1−p_i)`;boundary 判定 `τ<p_i<1−τ`(K=4 时 τ=0.125),`boundary_count/rate = #{τ<p_i<1−τ}`。
3. THE 系统 SHALL NOT 实现 `weighted_coverage_S` / `frontier_bias_eta` / 覆盖 descriptor(V0 明确不做;若做属 V1 optional)。
4. THE 系统 SHALL 复用现有 `skill_signal.extract_skill_signals` 框架(仅换注入源为 W_probe);生成器保持现有契约 `generate_obstacles(tblock_pose, num_obstacles)`。
5. THE 系统 SHALL 记录 `r_hard/r_easy/boundary_rate/target_rate/mean_b/valid_rate/duplicate_rate/w0_success_rate`(w0 仅部署诊断,不进 p_i)。
6. THE 每个 evolve 周期 SHALL 使用两个**独立职责**的探针,SHALL NOT 强制"方向与验收单次共享同一批固定起点":
   - 6.a 方向探针(上下文/prompt):SHALL 用**随机起点 + `invalid_scene_policy='resample'`**(只在 VALID 场景上度量库能力 realized/boundary),以在 M 个有效题下给出稳定的进化方向;`probe_source='w_probe'`。
   - 6.b 验收探针(verifier):SHALL 用 `sample_fixed_starts` 固定起点,G_t 与所有候选 SHALL 在**同一批固定起点上 paired 打分**(`invalid_scene_policy='zero'`);verifier SHALL 自行重探 G_t(`current_signal_result=None`),SHALL NOT 复用方向探针结果。
   - 6.c 理由(反面约束):SHALL NOT 强制方向探针也用 `fixed_starts + 'zero'`——该"双探针同源"在硬 G_0 上把生成器无效场景误记为库失败(realized=0),触发冷启动 RELAX 塌陷(实测 D=0.00);方向与验收的轻微采样差异是良性的(verifier 在自身一致的固定批上做 argmax 拍板)。

### Requirement 8: 单标量治锁 verifier(V0)

**User Story:** 作为研究者,我要课程验收用单标量目标(而非多轴非降合取)在候选与当前 G_t 间选择,从结构上杜绝棘轮死锁,并在库学透当前生成器时对称地(过易/过难)强制加/减难。

#### Acceptance Criteria

1. THE 验收判据 SHALL 替换 `evaluate_learnable_frontier_shift` 的四门不变差合取,SHALL NOT 使用任何"候选须在多指标上同时不劣于 G_t"的合取。
2. THE 绝对硬门 SHALL 仅为 `code_pass` + `valid_rate ≥ v_min` + `duplicate_rate ≤ d_max`(只筛 LLM 候选);`boundary_rate` SHALL NOT 作为硬门。当前 G_t SHALL 永远作为 incumbent 参选、不被候选硬门淘汰(除饱和逃逸外)。
3. THE 单一标量目标 SHALL 为 `boundary_count`(并列时以 `mean_b` 打破,SHALL NOT 使用 `boundary_rate + α·mean_b` 复合分);在 `{G_t}∪{通过硬门候选}` 上取 argmax。
4. THE 替换条件 SHALL 为 `cand_boundary_count ≥ Gt_boundary_count + min_boundary_count_delta`(等价浮点版 `J(best)>J(G_t)+ε`);否则 hold(generator-level reject-and-hold)。
5. WHEN `saturated = (r_easy ≥ r_easy_max) OR (r_hard ≥ r_hard_max)`,THE 系统 SHALL 令当前 G_t 不再阻塞替换,在通过硬门的候选中按 `boundary_count` 选最优(候选仍须过绝对硬门);SHALL NOT 因此接受未过硬门的候选。
6. THE SkillVerifierConfig SHALL 新增 `criterion=boundary_count_argmax_escape`、`min_boundary_count_delta`、`J_epsilon`、`r_easy_max`、`r_hard_max`、`tau_saturation`、`target_delta`;`decide_skill_acceptance` SHALL 调用新判据(实现用 boundary_count)。
7. WHERE `diversify_on_hold=true`(默认 false = 旧 hold 行为),WHEN 未饱和且无 boundary 净改进(本应 hold),THE 系统 SHALL 执行 PRESERVE_AND_DIVERSIFY 换代而非冻结 G_t:在**等难度**候选(过硬门、`boundary_count ≥ cur − diversify_bc_tolerance`、`r_easy ≤ cur.r_easy + diversify_easy_eps` 即不更易)中挑一个替换,使训练分布持续多样化而不往易漂;若无等难度候选则仍 hold。
   - 7.a WHERE `diversify_prefer_lower_easy=true` 且 `cur.r_easy > diversify_r_easy_soft`,THE diversify 换代 SHALL 在等难度候选里优先选 **r_easy 最低**者(温和连续 re-harden,tiebreak 取 boundary_count 高者),SHALL NOT 让 boundary_count 低于 `cur − diversify_bc_tolerance`(防过度加难)。
   - 7.b THE diversify 换代 SHALL NOT 覆盖 8.3 的净改进 replace 与 8.5 的饱和逃逸(二者优先级更高)。

### Requirement 9: 单技能部署

**User Story:** 作为研究者,我要部署严格单技能单阶段,与标准 DIVO 可比。

#### Acceptance Criteria

1. THE 部署 SHALL 执行 `a = decoder(cat([state, encoder(obs), w_0]))`(固定 `w_0`,单技能、单阶段)。
2. THE 部署 SHALL NOT 使用 DIVO stage-2/3 的 latent skill sampler 或 motion predictor。
3. WHERE 提供 best-of-K 挑技能对照,THE 系统 SHALL 仅将其作为 ablation,SHALL NOT 作为主结果报告。

### Requirement 10: Stage 0 LLM 初始生成器 + Stage 1 go/no-go(在 G_0 上)

**User Story:** 作为研究者,我要整条 Route B 全程 LLM 驱动:先用 LLM 生成初始障碍生成器 G_0,再在 G_0 动态采样的分布上预训 w_0 + W_probe,并用 G_0 验证布局做 go/no-go——不再用手工固定阶梯当主训练分布。

#### Acceptance Criteria

1. THE 系统 SHALL 在技能库训练前用 LLM 生成初始障碍生成器 G_0(`use_llm_obstacles=true`,复用现有 `_init_topology_generator`/`get_initial_code`/`build_phase0_prompt_stage_a`)。
2. THE G_0 SHALL 遵循现有契约 `generate_obstacles(tblock_pose, num_obstacles)`。
3. THE G_0 SHALL 在 Stage 1 训练前通过 code/physical/validity 检查(**Stage 0 此时无 W_probe,SHALL NOT 用 p_i/b_i/boundary 选 G_0**;只用 code_pass/valid_rate/duplicate_rate/contact + 轻量难度散布启发式)。
4. THE Stage 1 训练分布 SHALL 从 G_0 动态采样(`_make_llm_scene`),SHALL NOT 用手工 `E_stage1` 作主训练分布;`E_stage1` 仅作 debug/诊断。
5. THE Stage 1 go/no-go 硬门 SHALL 包括:`B/M/U ≥ baseline − δ` 且 w_0 overall validation 不明显下降;`K_eff ≥ 0.5K`;技能非平凡比例达标(见 5.a/5.b)。
   - 5.a THE 单技能非平凡判据 SHALL 为 `Progress(w_k) ≥ τ_p`。
   - 5.b THE 库级非平凡判据 SHALL 为 `(1/K)·Σ_k 1[Success(w_k) ≥ τ_s] ≥ r_s`(`r_s ∈ [0.5,0.7]`);SHALL NOT 要求每个 w_k 都高成功率。
6. THE Stage 1 go/no-go 的 `p_i` 分布非退化验证 SHALL 在**从 G_0(及可选 held-out LLM 布局)采样的 validation layouts** 上做(不再在手工阶梯上做):存在高 p、低 p、中间 `0<p<1` 场景。
7. THE Stage 1 SHALL NOT 把 "D ≥ 标准 DIVO" 作为硬门(D 观察项)。
8. WHEN 任一硬门长期不达标,THE 流程 SHALL 回到风险分析,SHALL NOT 继续堆叠组件。

### Requirement 11: 评测与度量(对齐 DIVO,禁 p-hacking)

**User Story:** 作为研究者,我要评测严格对齐 DIVO 口径,不为找赢临时调评测列,且 best checkpoint 选择不泄漏。

#### Acceptance Criteria

1. THE best checkpoint SHALL 用**固定验证分布(DIVO `between`)上的 `test_mean_score`**(w_0 部署)选,SHALL NOT 用会随课程漂移的 seen G_t,SHALL NOT 用 B/M/U/D 选。
   - 1.a THE 验证分布 SHALL 为固定的 `between`(`pusht_mujoco_llm`,`obstacle_num=2`、`obstacle_size=0.01`、`obstacle_dist=between`),独立于 evolve 的 G_t、且与 B/M/U/D 不重叠;实现 SHALL 复用现有 `llm_eval_env`(validate 时普通 `reset()`、不 `set_obstacle_config`,走 between 原版分支),SHALL NOT 新写环境/评测代码。
   - 1.b THE 四格对比(a/b/c/d)与所有 seed SHALL 共用同一验证分布选 best。
2. THE 系统 SHALL 报告 B/M/U/D 四类泛化成功率(只作最终零样本测试,不参与选 best)。
3. THE 系统 SHALL 报告 K_eff、within/across 轨迹多样(复用 `analysis/skill_coverage/measure_coverage_diversity.py`)、以及 realized 随训练变化。
3.a THE K_eff SHALL 按**固定协议**计算并作为可复现硬门:固定 validation probe 环境集 → 每个环境枚举 K 技能 → task-frame 轨迹嵌入 `φ_{i,k}` → 固定 embedding/距离度量/聚类方法/阈值下聚类 → `K_eff = exp(H(P_cluster))`(簇分布熵的指数)。占用簇数 MAY 单独记录日志,但 SHALL NOT 作为 go/no-go 指标(它会高估严重不均匀的行为簇分布)。embedding、metric、clustering、threshold、probe 环境集 SHALL 全部固定,不得随实现变动。
4. THE action diversity(对标 `evaluation.py::calculate_action_diversity`)SHALL 作为参考对标/辅助诊断;`0.62` SHALL NOT 作为 Stage 1 硬门。
5. THE 系统 SHALL 产出可视化:K 技能 T-block 扇形叠图(复用 `get_tblock_feature_points` + `rollout_fixed_skill` + `physics.render()`)与障碍 rollout 叠图。
6. WHEN 多样性提升,THE 成功率(基于 reward_task)SHALL NOT 塌陷(防"多样失败")。

### Requirement 12: 对比实验设计(Stage 3 为主)

**User Story:** 作为研究者,我要主对比对齐社区口径并保留内部机制消融;但大规模对照只在 Stage 3 展开,避免前期代码为支持所有对照而膨胀。

#### Acceptance Criteria

1. THE 完整对比(下列 2–5)SHALL 标注为 **Stage 3 only**,SHALL NOT 影响 Stage 1/2 的开发优先级与代码范围。
2. THE Stage 1 对比范围 SHALL 限定为最小集:DIVO baseline、路B(no-div)、路B + Form B,以及可选 `w_0` vs best-of-K 诊断。
3. THE Stage 3 主对比 SHALL 对齐口径 `random / static / eureka(success 引导) / 本方法(库信号引导)`,并含路B vs 路C vs evolve vs 固定基线。
4. THE Stage 3 内部机制三臂消融 SHALL 包括:库引导(本方法,boundary 信号 `b=p(1−p)` + 单标量治锁)、单策略引导(复现 evolve 的归因证据分)、随机地板(仅 valid 门、永不锁)。
5. THE Stage 3 消融 SHALL 覆盖:训库/不训、多样 on/off、单技能/挑技能、K∈{4,8,16}、β_div、rollout ratio;并与 evolve 的 dead-end `D=0.32` 参照对比,且要求 B/M/U 不掉。

### Requirement 13: 可复现性、日志与向后兼容

**User Story:** 作为使用现有实验框架的研究者,我要路B 改动可复现、可诊断,且在关闭技能库时不破坏现有 DIVO 训练。

#### Acceptance Criteria

1. THE Codebook、技能采集、probe SHALL 由配置种子派生所有 RNG,给定相同种子与配置产生一致结果。
2. THE 系统 SHALL 记录每轮的 `reward_task` 与 `reward_total`、K_eff、ActionVar、SatRate、boundary_count/boundary_rate/mean_b,并在关键指标上严格区分任务与多样性增广回报。
3. THE 系统 SHALL 区分两种退化模式,不得混为一谈:
   - **模式 A(严格 DIVO)**:`a = D(s, E(obs))`,decoder 输入**不含** w,加载旧 DIVO 配置与网络维度完全一致,用于与原始 DIVO 逐字节可比。
   - **模式 B(skill-disabled)**:`a = D(s, E(obs), w_0)`,decoder 输入含固定 `w_0`(维度含 d_w),是路B 的单技能版本,**不等于**严格 DIVO。
4. THE 现有公共接口(`predict_action`、critic forward 的既有调用点、`decide_skill_acceptance` 签名的向后兼容层)SHALL 通过新增可选参数扩展,默认保持现有行为。
5. WHEN 加载不含技能库状态的旧 checkpoint,THE 系统 SHALL 按模式 A 正常恢复训练;WHEN 加载路B checkpoint,THE 系统 SHALL 恢复技能库状态(含 codebook 与 w 标签 buffer)。
