# Implementation Plan: Skill-Library Curriculum(路B / 技能库可学性课程)

## Overview

范围约定(依据 Requirement 10 / 12.2):**首批主线 = Stage 0 LLM 初始生成器 G_0 bootstrap + Stage 1 技能库预训(on G_0)**。因为主训练分布来自 LLM 生成的 G_0,Stage 0 是 Stage 1 的必要前置(正确顺序:`G_0 bootstrap → Stage 1 training on G_0`),不能放在 Stage 2 之后或当附属。
- **Stage 0**:用 LLM 生成并验证 G_0(复用现有 Stage A/phase0 代码,零改动;只用 code/physical/duplicate/contact + 轻量难度散布启发式选择,**此时无 W_probe,不用 p_i/b_i/boundary 选 G_0**)。
- **Stage 1**:在 G_0 动态采样分布上训练 w_0 + W_probe——加 w + one-hot K=4 + 独立 w_0 + executed-w 损失 + Form B 多样(软门+同环境配对,经 reward_total 走 RL)+ w-条件 critic + 在 G_0 validation layouts 上做 go/no-go。
- **Stage 2/3**(库可学性课程 + 治锁 verifier / 完整对比消融)仍**默认不在首批开发范围**,待 Stage 1 go/no-go 通过后再启动。
- **固定阶梯 E_stage1** 已降级为 debug/诊断,非主训练分布、非主 go/no-go。

工程约定(本轮修订钉死):
- **只有一个多样性权重 `β_div`**(Form B 经 `reward_total = reward_task + β_div·reward_div` 进 critic);Stage 1 调度的是 `β_div`,不是 λ_div。Form A 只监控,不做训练目标(若将来启用其可微项才叫 `λ_A`)。
- **reward_for_critic 按技能路由**:`w_0` 用 `reward_task`,`W_probe` 用 `reward_total`;评测/go/no-go 一律用 `reward_task`/真实 success。
- **技能探针固定的是技能码 w**:`w_k` 在 episode 内固定,`z_t = E(obs_t)` 每步实时计算(函数名 `rollout_fixed_skill`,不叫 rollout_fixed_z)。
- **接口向后兼容在 Task 1/2 就设计**:`predict_action(obs, skill_id=None)`、`critic.forward(obs, actions, w=None)`;`skill_enabled=False` 时退化严格 DIVO。

每个任务只做编码/测试,不含长跑训练本身(训练由用户手动发起;但提供工程 smoke config)。

## Tasks

### Stage 1:技能条件策略最小闭环

- [x] 1. 技能码 codebook 与 w-条件策略(`DIVO/policy/ldpi.py` + decoder nets)
  - [x] 1.1 新增常量 `codebook`(形状 `[K+1, d_w]`,索引 0=w_0,1..K=W_probe;第一版 one-hot、`requires_grad=False`),由配置 `K`、`codebook_type` 构造
    - _Requirements: 1.3, 1.6_
  - [x] 1.2 decoder MLP 首层 `in_chan += d_w`;新增 `decode_with_skill(state, z, w)` 与 `predict_action_with_skill(obs, w)`
    - _Requirements: 1.1, 1.2_
  - [x] 1.3 **接口兼容**:`predict_action(obs, skill_id=None)`,默认 `skill_id=0`(拼 w_0,模式 B);`skill_enabled=False` 时走严格 DIVO(模式 A,decoder 输入不含 w、维度与旧 DIVO 一致)
    - _Requirements: 1.5, 9.1, 13.3_
  - [x] 1.4 单元测试:w 通道不影响 z 计算;模式 A 输出维度与旧 DIVO 一致;`predict_action(obs)` 等价于拼 w_0
    - _Requirements: 1.4, 13.3_

- [x] 2. w-条件 critic(`DIVO/critic/mcritic.py`)
  - [x] 2.1 **接口兼容**:`SingleFCCritic.forward(obs, actions, w=None)`;`w is None`→`q_net(cat([obs, actions]))`(严格 DIVO),`w` 给定→`q_net(cat([obs, w, actions]))`;`MultiCritic` 透传
    - _Requirements: 2.1, 13.3_
  - [x] 2.2 q_network `in_chan` 在 skill_enabled 下 `+= d_w`;配置对齐
    - _Requirements: 2.1_
  - [x] 2.3 单元测试:同 obs/action 下不同 w 给出不同 Q;`w=None` 时与旧 critic 逐值一致;批量形状正确
    - _Requirements: 2.1, 13.3_

- [x] 3. Replay buffer 技能标签与三列奖励(`DIVO/RL/component.py`)
  - [x] 3.1 每条转移存 `skill_id`(0=w_0,1..K)与来源标签(`normal` / `diversity_paired`)
    - _Requirements: 6.2, 4.9_
  - [x] 3.2 奖励存三列:`reward_task`、`reward_div`、`reward_total`(不覆盖原单一 reward 字段)
    - _Requirements: 4.5_
  - [x] 3.3 采样返回 `skill_id`、来源标签与三列奖励(供 critic 路由与调度)
    - _Requirements: 6.3, 2.2, 2.3_
  - [x] 3.4 单元测试:存取往返一致;来源标签与三列奖励可区分
    - _Requirements: 6.2, 4.5, 4.9_

- [x] 4. reward routing 与 source-stratified 采样(`DIVO/RL/component.py` + workspace)
  - [x] 4.1 定义 `reward_for_critic`:`w_0` 转移用 `reward_task`,`W_probe` 转移用 `reward_total`
    - _Requirements: 4.6, 4.7_
  - [x] 4.2 batch 采样支持按来源分层 / 上限 `paired_sample_ratio`(默认 ≤ 0.3),防 paired 数据主导训练分布
    - _Requirements: 4.9_
  - [x] 4.3 `P(w_0) ≥ 0.5` 按**进入训练 batch 的最终样本分布**控制(含 normal + paired),而非仅普通 rollout 采样分布
    - _Requirements: 6.4, 6.5_
  - [x] 4.4 日志:每 batch 的 `w_0/probe/paired` 有效占比
    - _Requirements: 6.4_
  - [x] 4.5 单元测试:reward_for_critic 路由按 skill_id 正确;分层采样比例符合配置
    - _Requirements: 4.6, 4.9_

- [x] 5. `rollout_fixed_skill` 统一 API(`DIVO/curriculum/skill_signal.py` 或新 `skill_rollout.py`)
  - [x] 5.1 实现 `rollout_fixed_skill(env, policy, skill_id, layout, layout_seed)`:episode 内 `w=w_{skill_id}` 固定,`z_t = E(obs_t)` 每步实时计算;返回轨迹 states/actions、`reward_task` 序列、success/progress/collision/timeout、transition 列表
    - _Requirements: 4.2_
  - [x] 5.2 复用现有 `rollout_fixed_z` 的记录思路但语义改为固定技能;保留 `rollout_fixed_z`(供 random-z 基线/兼容),不混用命名
    - _Requirements: 4.2_
  - [x] 5.3 单元测试:同一 episode 内 w 恒定、z 随 obs 变化;返回字段完整
    - _Requirements: 4.2_

- [x] 6. 技能回合采集与占比(`td3_curriculum_workspace.py`)
  - [x] 6.1 普通训练采集:每回合按分布抽一个 w 执行,存 `(obs, w, a, reward_task, next_obs, done)`,来源标签 `normal`
    - _Requirements: 6.1, 4.10_
  - [x] 6.2 采样分布默认 `P(w_0) ≥ 0.5`、`P(w_k)=0.5/K`,由 `w0_rollout_ratio`/`probe_rollout_ratio` 配置,断言禁止 `P(w_0) < 0.5`
    - _Requirements: 6.4, 6.5_
  - [x] 6.3 单元测试:普通 rollout 经验分布符合配置占比
    - _Requirements: 6.4_

- [x] 7. executed-w-only actor 损失与 critic 路由(`compute_policy_loss` / `compute_critic_loss` / `compute_next_q_value`)
  - [x] 7.1 `L_actor = L_exec + L_z`,`L_exec = -E_{(obs,w_exec)}[Q(obs, w_exec, D(s,E(obs),w_exec))]`,按 batch `skill_id` mask(每条转移只用其执行技能)
    - _Requirements: 3.1, 3.2_
  - [x] 7.2 保留 gaussian z 正则 `L_z` 挂 w_0 路径;命名 `L_exec`(非纯任务价值,附注释);不新增独立可微 L_div
    - _Requirements: 3.3, 3.4, 3.6_
  - [x] 7.3 critic 损失/目标 Q 用 `reward_for_critic`(w_0→reward_task,probe→reward_total)并带 w 标签
    - _Requirements: 2.2, 2.3, 4.6_
  - [x] 7.4 可选 `L_cov-all`(默认 `lambda_cov_all=0`,warm-up 才启用)
    - _Requirements: 3.5, 3.7_
  - [x] 7.5 单元测试(具体化):给定 batch 全为 `w2` 且 `lambda_cov_all=0`,`compute_policy_loss` 只用 `w2` 调 decoder 计 `L_exec`,**不**用 w1/w3... 计任务损失(用 hook/mock 统计 skill_id);打开 `lambda_cov_all>0` 才有 all-w 梯度
    - _Requirements: 3.2, 3.3, 3.5_

- [x] 8. 任务坐标轨迹嵌入(`DIVO/curriculum/skill_diversity.py`)
  - [x] 8.1 任务坐标系(SE2 归一化 / task_frame_waypoints)下轨迹嵌入 `φ`,复用 `_embed_route` 定长展平 + 归一化
    - _Requirements: 4.1_
  - [x] 8.2 单元测试:同轨迹在不同起点/朝向下嵌入近似不变
    - _Requirements: 4.1_

- [x] 9. paired diversity rollout 与 Form B 奖励(`skill_diversity.py` + workspace)
  - [x] 9.1 diversity paired rollout:同批环境 seeds `E={e_1..e_N}` 上用 `rollout_fixed_skill` 枚举 `w_1..w_K` 得 `τ_{i,k}`
    - _Requirements: 4.2, 4.9_
  - [x] 9.2 软 near-success 门 `q(τ)=α·success+β·progress−γ·collision−η·timeout`、`g(τ)=σ((q−τ_q)/T)`,同环境内 margin 有界 `r_div(i,k)`
    - _Requirements: 4.3, 4.4_
  - [x] 9.3 **episode 级 r_div → transition 分配**:默认按 episode 长度均摊 `reward_div_t = r_div / episode_len`(配置可切换 terminal-only / 折扣),对 r_div 做 margin/归一化
    - _Requirements: 4.3, 4.5_
  - [x] 9.4 组装三列奖励:w_0 回合 `reward_div=0`;probe 回合 `reward_total=reward_task+β_div·reward_div`;paired transition 以 `diversity_paired` 标签写回 buffer
    - _Requirements: 4.5, 4.6, 4.7, 4.9_
  - [x] 9.5 (可选后备)`use_dual_critic` 分支骨架,默认 False
    - _Requirements: 4.8_
  - [x] 9.6 单元测试:距离只在同一 e_i 内;全失败批 r_div=0 不报错;软门在 near-success 给正信号;均摊后回合 reward_div 之和≈r_div
    - _Requirements: 4.2, 4.3_

- [x] 10. 动态 `β_div` 调度与 warm-up(`td3_curriculum_workspace.py`)
  - [x] 10.1 task warm-up:`task_warmup_steps` 内仅训 `L_exec + L_z`(`β_div=0`);之后以小 `β_div` 开启多样
    - _Requirements: 5.1_
  - [x] 10.2 按 eval 周期调 `β_div`:`success<floor→×0.5`;`success≥floor 且 K_eff<eff_floor→×1.2`;裁剪 `[β_div_min, β_div_max]`(统一权重,不再用 λ_div)
    - _Requirements: 5.2, 5.3, 5.4, 5.5_
  - [x] 10.3 调度所用 `avg_success`/floor 一律取 `reward_task`/真实 success
    - _Requirements: 5.6, 4.7_
  - [x] 10.4 单元测试:三分支切换与裁剪边界正确
    - _Requirements: 5.2, 5.3, 5.4_

- [x] 11. K_eff 固定协议与多样性监控(`skill_diversity.py`)
  - [x] 11.1 **K_eff 唯一定义**:`K_eff = exp(H(P_cluster))`(轨迹 φ 聚类后簇分布熵的指数);占用簇数只作日志,**不作 go/no-go 指标**。固定 validation seeds/φ/距离度量/聚类算法/阈值
    - _Requirements: 11.3, 11.3.a_
  - [x] 11.2 Form A 监控:`ActionVar`(对标 `calculate_action_diversity`,用 K 技能替代 sampler)与 `SatRate`,仅日志
    - _Requirements: 4.11, 11.4_
  - [x] 11.3 单元测试:同输入 K_eff 可复现;不均匀簇分布 K_eff 明显低于占用簇数;ActionVar 与参考口径一致
    - _Requirements: 11.3.a_

- [x] 12. **Stage 0:LLM 初始生成器 G_0 bootstrap**(主线起点,**纯复用现有代码,零改动**)+ 阶梯降级为诊断
  - [x] 12.1 配置 `use_llm_obstacles=true`(config 已设 + openai/base_url/model,key 从环境变量读取);训练启动时现场生成 G_0
    - _Requirements: 10.1_
  - [x] 12.2 现有 `_init_llm_generator`→`_init_topology_generator`→`get_initial_code`→`ACGS_API.init_generator`(用 `build_phase0_prompt_stage_a`)生成 G_0,遵循契约 `generate_obstacles(tblock_pose, num_obstacles)`——**零改动**
    - _Requirements: 10.1, 10.2_
  - [x] 12.3 G_0 由现有 sandbox code-pass 检查把关(`load_generator_code`);训练期 `_make_llm_scene` 逐场景做 physical/contact 重采样(`is_obstacle_config_valid`/`get_ncon`)。**不做额外 Stage 0 选择门、不用 p_i/b_i/boundary 选 G_0**(V0 决定:先跑最小闭环,公平性/固定 G_0 留 Stage 3)
    - _Requirements: 10.3_
  - [x] 12.4 Stage 1/2 训练分布从 G_0 动态采样(`_make_llm_scene`,现有代码);**不用 E_stage1 作主训练分布**
    - _Requirements: 10.4_
  - [x] 12.5 (诊断,已实现,降级)固定阶梯 `stage1_ladder.py` + `build_sampled_stage1_ladder`(从当前分布采样验证布局)——**仅 debug/诊断,非主训练分布**
    - _Requirements: 10.4_

- [x] 13. Stage 1 部署评测与 go/no-go 汇总(`analysis/` 或 evaluator 脚本)
  - [x] 13.1 部署评测路径:`a = decoder(cat([state, encoder(obs), w_0]))` 单技能;可选 best-of-K 仅诊断,不作主结果
    - _Requirements: 9.1, 9.2, 9.3_
  - [x] 13.2 用 `rollout_fixed_skill` 在**从 G_0 采样的 validation layouts**(`build_sampled_stage1_ladder` 指向当前 G_0 分布;可选 held-out LLM 布局)上算 per-scene `p_i` 与库 realized,验证 `p` 分布非退化(高 p / 低 p / 中间 0<p<1);**不再以手工 E_stage1 为准**
    - _Requirements: 10.6_
  - [x] 13.3 汇报 B/M/U/D(w_0,只作最终测试);best checkpoint 用**固定 between 验证分布**上的 `test_mean_score`(w_0 部署)选,不用 seen G_t、不用 B/M/U/D;汇报 K_eff、within/across 轨迹多样、realized 曲线
    - _Requirements: 11.1, 11.2, 11.3_
  - [x] 13.4 硬门判定:`B/M/U ≥ baseline−δ` 且 w_0 不明显掉;`K_eff ≥ 0.5K`;`Progress(w_k)≥τ_p` 且 `库内 Success≥τ_s 比例 ≥ r_s(0.5~0.7)`
    - _Requirements: 10.3, 10.3.a, 10.3.b_
  - [x] 13.5 观察项(非门):D、action diversity(0.62 仅参考);多样提升不塌 reward_task success
    - _Requirements: 10.4, 11.4, 11.6_

- [x] 14. 配置接线、向后兼容与 smoke config(Hydra config)
  - [x] 14.1 新增配置:`K / d_w / codebook_type / skill_enabled / w0_rollout_ratio / probe_rollout_ratio / paired_sample_ratio / lambda_cov_all / beta_div(min/max) / div_margin / rdiv_distribution / soft_gate(α,β,γ,η,τ_q,T) / success_floor / eff_floor / task_warmup_steps / use_dual_critic / τ_p / τ_s / r_s`
    - _Requirements: 1.6, 3.7, 4.8, 5.4, 6.4, 10.3.b_
  - [x] 14.2 `skill_enabled=False`(模式 A,严格 DIVO)确保训练/部署与旧 DIVO 逐字节可比
    - _Requirements: 13.3, 13.4_
  - [x] 14.3 checkpoint 兼容:旧(无技能库)按模式 A 恢复;路B checkpoint 恢复 codebook 与含 skill_id 的 buffer
    - _Requirements: 13.5_
  - [x] 14.4 **Stage 1 smoke-run config**:K=2 或 4;训练分布用 Stage 0 生成的 G_0 或一个 mock G_0 generator,从 G_0 动态采样(`e ~ G_0`)少量 episodes/steps,跑通 normal + paired rollout + 一次 update,断言 buffer 写入 skill_id、critic 收到 w、paired 计出 r_div、必需指标已记录(E_stage1 小子集仅作 debug-only sanity,不作 smoke 默认起点)
    - _Requirements: 13.3, 13.4, 13.5_
  - [x] 14.5 集成测试:模式 A 与旧 DIVO 一致;skill_enabled 端到端 smoke 跑通
    - _Requirements: 13.3, 13.4, 13.5_

- [x] 15. Stage 1 可视化(诊断产出)
  - [x] 15.1 K 技能 T-block 扇形叠图(`get_tblock_feature_points` + `rollout_fixed_skill` + `physics.render()`)
    - _Requirements: 11.5_
  - [x] 15.2 障碍 rollout 叠图(同布局多技能路径叠加)
    - _Requirements: 11.5_

### Stage 2(Route B V0:库可学性驱动 generator 进化)——Stage 1 已通过,开始开发

> 定稿约定:更新 generator `G_t→G_{t+1}`(不做池/cache);V0 只用 boundary 信号 `b=p(1−p)`(不做加权覆盖、不做多样性奖励 β_div=0);p_i 只用探针库(不含 w_0);**尽量复用现有接口/模块**(`generate_obstacles`/`set_obstacle_config`/`SandboxPushTExecutor`/`extract_skill_signals`/`ACGS_API.evolve`/`_make_llm_scene`),思路对即可、在现有基础上改。

- [x] 16. 库 boundary 信号(改 `skill_signal.py`,复用 `extract_skill_signals`)
  - [x] 16.1 `extract_skill_signals` 加 `probe_source` 开关:`random_z`(默认,旧 evolve 路径不破坏)| `w_probe`(Stage 2 V0,opt-in);`w_probe` 用**训练好的 W_probe** 经 `rollout_fixed_skill`(新增 `rollout_scene_with_skills`,w_1..w_K 探针 + w_0 诊断),K 取自 `policy.K`,要求 `skill_enabled`;生成器保持契约 `generate_obstacles(pose,num)`。Task 18 编排显式传 `probe_source='w_probe'`
    - _Requirements: 7.1, 7.5_
  - [x] 16.2 每场景 `p_i=(1/K)Σ_k 1[π_{w_k} solves e_i]`(**只用探针,不含 w_0**;w_0 单独 rollout 作诊断)、`b_i=p_i(1−p_i)`;boundary 判定 `τ<p_i<1−τ`(K=4→τ=0.125),`compute_boundary_signal` 实现
    - _Requirements: 7.2_
  - [x] 16.3 `compute_boundary_signal` 统计 `boundary_count/boundary_rate/r_hard/r_easy/target_rate/mean_b/valid_rate/duplicate_rate/w0_success_rate`(w0 仅诊断);**未实现 weighted_coverage_S / frontier_bias / 覆盖 descriptor**(纯 p(1−p))
    - _Requirements: 7.2_
  - [x] 16.4 组件级 sanity 已验证(p/b/boundary_count/rate/r_hard/r_easy/target_rate/mean_b/duplicate 数值正确;`probe_source='w_probe'` 走 W_probe 且要求 skill_enabled)——按用户约定不写单独单测
    - _Requirements: 7.1, 7.2_

- [x] 17. 进化方向 + 单标量治锁 verifier(`learnable_frontier.py` + `candidate_verifier.py`)
  - [x] 17.1 三段方向:`_skill_library_direction`(workspace)= `valid_rate<v_min→FIX_VALIDITY`(前置)/`r_hard≥r_hard_max→RELAX`/`r_easy≥r_easy_max→HARDEN`/否则 `PRESERVE_AND_DIVERSIFY`
    - _Requirements: 8.1, 8.4_
  - [x] 17.2 **替换四门合取为单标量(计数版)**:`select_generator_boundary`(learnable_frontier)primary=`boundary_count`(=#{τ<p_i<1−τ}),`mean_b` 仅并列 tiebreak,无复合分;绝对硬门 `boundary_hard_gate`=`code_pass + valid_rate≥v_min + duplicate_rate≤d_max`(只筛候选);boundary_rate 仅日志。旧四门 `evaluate_learnable_frontier_shift` 保留供 legacy
    - _Requirements: 8.1, 8.2, 8.6_
  - [x] 17.3 含当前 G_t 的 argmax:在 `{G_t}∪{通过硬门候选}` 上按 `boundary_count`(mean_b 并列)argmax;仅当 `cand_boundary_count ≥ Gt_boundary_count + min_boundary_count_delta` 才 replace,否则 hold
    - _Requirements: 8.3_
  - [x] 17.4 **饱和逃逸(easy/hard 对称)**:`is_saturated` = `(r_easy≥r_easy_max) OR (r_hard≥r_hard_max)`;饱和时 G_t 不阻塞,在过硬门候选里按 boundary_count 选最优(HARDEN/RELAX escape);无候选过门则 hold
    - _Requirements: 8.4, 8.5_
  - [x] 17.5 `SkillVerifierConfig` 加 `criterion / probe_source / tau_saturation / target_delta / valid_rate_min / duplicate_rate_max / min_boundary_count_delta / J_epsilon / r_easy_max / r_hard_max / candidates_per_cycle / eval_layouts_M`(默认 legacy-safe:`criterion=learnable_frontier_shift`、`probe_source=random_z`;skill_library 分支显式设 boundary 值);`decide_boundary_selection` 调新判据
    - _Requirements: 8.6_
  - [x] 17.6 组件级 sanity 已验证 8 分支:①多轴顶格不死锁 ②饱和强制换(HARDEN/RELAX)③无候选过门 hold ④未饱和无净改进 hold ⑤tiebreak mean_b ⑥硬门失败 hold ⑦真实 SkillVerifierConfig 走通 ⑧饱和但无候选过门 hold(按用户约定不写单独单测)
    - _Requirements: 8.3, 8.4_

- [x] 18. Route B V0 闭环编排(`td3_curriculum_workspace.py` + `ACGS_API`)
  > 起点 = G_0(Task 12 现场 bootstrap);Stage 2 从 `current G_t` 起周期进化。复用现有 evolve 编排(context 探针 / prompt / 重试 / artifacts / phase4),仅换探针源与判据。
  - [x] 18.1 注册 `feedback_mode='skill_library'`(加入合法集合 + 三处 skill-signal 成员判断);评估用 W_probe 探针(`_compute_skill_signal_for_evolve` 传 `probe_source='w_probe'`);触发沿用现有 `trigger_mode`(固定 episode 间隔)
    - _Requirements: 7.1_
  - [x] 18.2 **best-of-R argmax(偏差1→方案1A 已做)**:`ACGS_API.generate_candidates(R)` 同一 evolve prompt 采样 R=3 个候选(各自 sanity,不改持久 generator,结束恢复);`_evolve_skill_library_best_of_r` 对 current+R 候选用 W_probe 配对探针打分,`decide_boundary_selection` 在 `{G_t}∪R候选` 上一次性 argmax(boundary_count 主、mean_b tiebreak)+ 饱和逃逸,replace 才 `load_generator_code(chosen)`,否则 hold。R 由 `skill_library.candidates_per_cycle` 配置(默认 3)。**方向进 prompt(偏差2B 已做)**:`_format_boundary_signal_block` 渲染 W_probe 口径 + τ-band 统计 + 唯一方向(信号→prompt→verifier 同一套定义)
    - _Requirements: 8.1_
  - [x] 18.3 候选三层验证(`SkillCandidateVerifierSession._verify_boundary`):code=`SandboxPushTExecutor` load + physical=boundary_signal 的 `valid_rate/duplicate_rate` + boundary=W_probe 探针;判据 `decide_boundary_selection`(增量硬门 + 净改进/饱和逃逸),不过则 reject→hold
    - _Requirements: 8.2, 8.3_
  - [x] 18.4 接受后训练直接从 G_{t+1} 动态采样(现有 `_make_llm_scene`,零改动);β_div=0、不跑 diversity_batch(config skill 块默认 `diversity_every=0`、`beta_div` 不启用)
    - _Requirements: 7.1_
  - [x] 18.5 监控 `w0_success_rate`(boundary_signal 内,`_verify_boundary`/context 探针日志)与 K_eff(Stage 1 协议);V0 **不自动开 β_div**(记为 M5 风险,属 V1/ablation)
    - _Requirements: 10.5, 11.3_
  - [x] 18.6 **两个独立职责的探针(偏差2 回退)**:方向探针(上下文/prompt)用随机起点 + `invalid_scene_policy='resample'` + `probe_source='w_probe'`,只在 VALID 场景上度量库能力、定进化方向;验收探针(verifier)用 `sample_fixed_starts` 固定起点,G_t 与候选 paired 打分(`invalid_scene_policy='zero'`),`_evolve_skill_library_best_of_r` 传 `current_signal_result=None`(verifier 自探 G_t,不复用)。**不强制"双探针同源"**:早期把方向探针也对齐成 `fixed_starts+'zero'`,在硬 G_0 上把生成器无效场景误记为库失败(realized=0)→ 冷启动 RELAX 塌陷、部署 D=0.00(AVG 0.49 vs 0.675);回退后方向诚实,方向↔验收轻微采样差异良性(verifier 在固定批 argmax 拍板)。legacy random_z 路径不变
    - _Requirements: 7.6_

### Stage 3(完整对比 + 消融)——待 Stage 2 跑通后

- [ ] 19. 完整对比与消融框架
  - [ ] 19.1 **四格主对比**(同协议 B/M/U/D + D,≥3 seeds):(a) 标准 DIVO baseline / (b) 单策略 LLM-evolve 课程(**从同一个 LLM-generated G_0 起步**)/ (c) 技能库无 evolution(**在同一个 G_0 上训练,不 rewrite generator**)/ **(d) 技能库 + W_probe boundary evolution(本方法,从同一个 G_0 起步并 evolve)**;b/c/d **必须使用同一个 LLM-generated G_0 起点**(仅信号源不同),避免把"换了训练分布"误算成课程收益;红线:(d)>(c)(证 generator evolution 有效)且 (d)>(b)(证库信号优于单策略信号)
    - _Requirements: 12.3_
  - [ ] 19.2 内部消融:库引导 vs 单策略引导 vs 随机;evolve 触发/饱和逃逸消融
    - _Requirements: 12.4_
  - [ ] 19.3 参数/组件消融:β_div on/off(多样性奖励是否有额外价值)、K、evolve_interval、τ;对照 evolve D=0.32
    - _Requirements: 12.5_

## Task Dependency Graph

```json
{
  "waves": [
    { "wave": 1, "tasks": ["1"], "rationale": "策略/codebook 地基" },
    { "wave": 2, "tasks": ["2", "3", "5", "8"], "rationale": "critic/buffer/rollout_fixed_skill/嵌入 均只依赖 1" },
    { "wave": 3, "tasks": ["4", "12"], "rationale": "4 依赖 2/3(reward 路由+采样);12 = Stage 0 G_0 bootstrap(复用现有,先于 Stage 1 训练采样与 go/no-go)" },
    { "wave": 4, "tasks": ["6", "14"], "rationale": "6 依赖 3/4(采集占比按最终 batch);14 配置/兼容/ smoke" },
    { "wave": 5, "tasks": ["7", "9"], "rationale": "7 依赖 2/3/4/6;9 依赖 4/5/6/8" },
    { "wave": 6, "tasks": ["10", "11", "15"], "rationale": "10 依赖 7/9;11 依赖 8/9;15 依赖 5/12" },
    { "wave": 7, "tasks": ["13"], "rationale": "go/no-go 汇总,依赖 7/9/10/11/12" },
    { "wave": 8, "tasks": ["16"], "rationale": "Stage2 起点:探针源换 W_probe + boundary 信号;门槛 Stage1 通过" },
    { "wave": 9, "tasks": ["17"], "rationale": "单标量治锁 verifier + 饱和逃逸,依赖 16" },
    { "wave": 10, "tasks": ["18"], "rationale": "Route B V0 闭环编排,依赖 16/17" },
    { "wave": 11, "tasks": ["19"], "rationale": "Stage3 完整对比/消融,依赖 Stage2(18)跑通" }
  ]
}
```

```mermaid
graph TD
  T1[1. codebook + w-策略] --> T2[2. w-critic]
  T1 --> T3[3. buffer 标签+三列奖励]
  T1 --> T5[5. rollout_fixed_skill]
  T1 --> T8[8. 任务坐标嵌入]
  T2 --> T4[4. reward 路由+分层采样]
  T3 --> T4
  T12[12. Stage0: LLM G_0 bootstrap（复用）]
  T3 --> T6[6. 技能采集/占比]
  T4 --> T6
  T12 --> T6
  T2 --> T7[7. executed-w 损失+critic路由]
  T3 --> T7
  T4 --> T7
  T6 --> T7
  T4 --> T9[9. paired diversity + Form B]
  T5 --> T9
  T6 --> T9
  T8 --> T9
  T7 --> T10[10. 动态 β_div 调度]
  T9 --> T10
  T8 --> T11[11. K_eff=exp(H) + 监控]
  T9 --> T11
  T5 --> T15[15. 可视化]
  T12 --> T15
  T7 --> T13[13. go/no-go 汇总]
  T9 --> T13
  T10 --> T13
  T11 --> T13
  T12 --> T13
  T13 -->|Stage1 go/no-go 通过| T16[16. 库 boundary 信号 p1-p]
  T16 --> T17[17. 单标量治锁 verifier + 饱和逃逸]
  T16 --> T18[18. Route B V0 闭环编排]
  T17 --> T18
  T18 -->|Stage2 跑通| T19[19. 完整对比/消融]
```

## Notes

- **权重统一**:全程只有一个多样性权重 `β_div`(经 reward_total)。Task 10 调度的就是 β_div;Form A 仅监控。
- **reward 路由**:buffer 存 `reward_task/reward_div/reward_total` 三列;`reward_for_critic` = w_0→reward_task、probe→reward_total;go/no-go 与最终报告只用 reward_task/真实 success。
- **命名**:技能探针用 `rollout_fixed_skill`(w 固定、z 每步实时),与旧 `rollout_fixed_z` 区分。
- **接口兼容**:`predict_action(obs, skill_id=None)`、`critic.forward(obs, actions, w=None)` 在 Task 1/2 就实现,`skill_enabled=False` 退化严格 DIVO。
- **训练不在任务内**:`- [ ]` 项只做实现与单元/集成测试;Task 14.4 提供工程 smoke config 证明端到端跑通,长跑与最终指标由用户手动发起。
- **不改评测列**:go/no-go 与报告严格用既有 B/M/U/D(只作最终测试);best 用**固定 between 验证分布**的 `test_mean_score`(w_0 部署)选——不用移动的 seen G_t、不用 B/M/U/D(防泄漏)。实现复用 `llm_eval_env`(validate 时普通 reset、不 set_obstacle_config → 走 between),四格/所有 seed 共用同一验证分布。
- **诚实门**:任一 Stage 1 硬门长期不达标,回 design.md §风险分析,不继续堆叠组件。
- **可追踪性**:Stage 2/3 引用的 7.6/7.7/8.6/12.3–12.5 已在 requirements.md 存在;编号随需求更新保持同步。
