# 技术设计文档:导航版技能库可学性课程(第二任务 / Nav Skill-Library Curriculum)

## Overview

### 定位与目标
把第一任务(Push-T)已建立的**"生成器进化 + 隐技能库可学性标尺"**框架迁移到**自主导航**,证明该框架**跨任务范式通用**(物体操作 → 导航),而非仅适用于推物单一场景。核心科学主张保持不变:

> 单确定性策略在每个场景上 p∈{0,1},SFL 式 `p(1−p)` 学习价值信号退化、且无法区分"可行但难"与"不可行";一族多样技能库把 `p_i=(1/K)Σ_k 1[π_{w_k} solves e_i]` 重新连续化,让课程信号复活、两类可分。

基座换成 **Safety-Gymnasium 的 `SafetyPointGoal`**(Point 智能体 + Goal 任务),用 **Pillars**(可碰撞刚体)作难度主元。**B/M/U/D 零样本基准照搬 DIVO 的参数化障碍类别定义**(num/size/shape/dist + 均匀随机起点,见 Testing Strategy),不是手工拓扑;我们额外手工设计的 barrier/maze/u/dead-end 墙体结构作为**B/M/U/D 之外的结构化拓展套件**(诊断/motivation 示意 + 未来额外实验)。方法本体(w-codebook、executed-w loss、w-条件 critic、库 boundary 信号、单标量治锁 verifier)与 Push-T 一致,只换 env adapter 与轨迹嵌入坐标系。

### 已验证的事实(本设计的实锚,均已在 `nav/` 原型跑通)
1. **障碍位置可显式指定**:`Pillars.locations` = "Fixed locations to override placements",生成器可直接输出坐标(`nav/nav_adapter.py` 已验证 B/M/U/D 柱数 6/10/11/8 热切换、坐标零误差)。
2. **obs 结构与 (state,z) 切分**(44 维):`accelerometer[0:3] velocimeter[3:6] gyro[6:9] magnetometer[9:12] goal_lidar[12:28] pillars_lidar[28:44]`。
   - `state = obs[0:28]`(本体 + 目标方向,对应 Push-T 的 `obs2state`)
   - `z 通道 = obs[28:44]`(pillars_lidar,**唯一障碍入口**,喂 encoder)
3. **动作** = `Box(-1,1,(2,))`;**success** = 原生 `goal_achieved`(dist ≤ goal.size=0.3);**reward** = 原生 dense(距离进步 + 到达 bonus);附带 `cost_pillars` 碰撞第二轴。任务可解(启发式 151 步到达)。
4. **motivation 已在导航复现**(`nav/probe_degeneracy.py`,单确定性 TD3,144 场景):
   - p_i∈{0,1},`boundary_count=0`,`mean_b=0.0000`;
   - "可行但难"(dead-end,p̄=0.083)与"不可行"(围死 goal,p̄=0.000)都塌向 0、不可分;
   - dead-end 有 ~8% 被单策略解开 → 证明该类可行(非死场景),单确定性贪心策略系统性失败(需回退,它表达不了)。

### 与 Push-T 的差异(避免张冠李戴)
- **环境隔离**:导航跑在独立 conda env `safenav`(safety-gymnasium 1.0.0 / gymnasium 0.28.1 / mujoco 2.3.3 / torch 2.3.0+cu118),与 Push-T 的 `divo` 环境版本冲突,不共享运行时。
- **轨迹嵌入坐标系**:Push-T 用 T-block 位姿轨迹;导航用 agent (x,y) 路径,在 start→goal 任务坐标系归一化。
- **生成器契约**:`generate_pillars(agent_start, goal, num) -> [(x,y),...]`(替代 Push-T 的 `generate_obstacles`)。
- **无边界墙**:导航 arena 无物理边界,中央墙体可从端点绕过 → "不可行"场景须**围死 goal**(一圈柱)而非"整道墙"。
- **确定性策略实现**:导航不复用 DIVO 的 `ldpi.py`,而在 `nav/` 内新实现确定性隐技能策略(TD3 骨架已在 `nav/td3.py`)。

### 设计边界与非目标
- **只做 SafetyPointGoal + Point**;不换 Car/Ant(留 future work)。
- **部署严格单技能、单阶段**:`a = decoder([state, encoder(obs), w_0])`,与单策略 TD3 baseline 可比。
- **起点/目标处理对齐 Push-T:goal 固定、agent 起点采样**(Push-T target 固定、tblock 起点采样)。训练/探针"先采起点、再 `generate_pillars(start,goal)`";验证 between"先摆 pillar、再采起点"。同一探针 scene 内 K 技能共享起点。
  - 注:`nav/` 原型(motivation)为冒烟简化用了**固定** start=(-1.3,0),主线改为采样起点(goal 固定);motivation 不受影响(同 scene 单起点仍二值)。
- **生成器 DOF 刻意受限**(防零样本泄漏,同 Push-T):训练柱 `num∈[1,4]`、中央带 `x∈[-0.5,0.5]/y∈[-0.8,0.8]`、size/keepout 固定。
- **B/M/U/D 基准 = 参数化 pillar 类别 + 均匀随机起点(完全贴 DIVO,已核实 `config/evaluation/eval.yaml`)**:DIVO 的 B/M/U/D 不是手工拓扑,而是按 (num/size/shape/dist) 定义的参数化障碍类别,起点均匀随机(`dist=random/random_step` → 起点 `pass` 均匀采,仅排除靠中心/近目标/碰撞),20 回合平均。导航照搬这套(见 Testing Strategy 的 nav 映射)。
- **结构化拓展套件(仅一个可选 eval 拓展,别处一律不用)**:我们手工设计的 barrier/maze/u-shape/dead-end(`nav/nav_env.py::TEMPLATES`)**只能作为 B/M/U/D 之外的一个可选零样本 eval 拓展基准**。SHALL NOT 用于:训练分布、课程信号探针 p_i、motivation 演示、验证选 best、多样性度量或任何其他环节。信号探针与 motivation **必须在生成器采样场景(G_t / 训练 DOF / 参数化分布)上做**,不得用手工结构。
- **多样性目标**:必须"解法多样"(不同绕行/回退路径),非"动作多样"或"多样失败"。

### 已知关键风险(正视)
1. **库帮不帮部署(H1/H2 赌注)**:训一族技能库 → 单 w_0 部署是否受益,是 Push-T 未决的赌注,迁移到导航仍未决,靠 (d)-vs-(b) ablation 定。
2. **w 塌缩**(K_eff ≪ K)。
3. **dead-end 是否被库真正复活**:Stage 1 核心 go/no-go —— 若库对 dead-end 仍全 p=0,则主张证伪。
4. **算力**:CPU 物理仿真,K 技能 rollout 成本随 K 线性增长(单策略 250k 步约 25 分钟 / 3090Ti,K=4 探针评估另计)。
5. **LLM 生成器基建移植(前置)**:Push-T 的 acgs_api/prompt_builder/sandbox 是 Push-T 专用,导航需移植到 `generate_pillars` 契约;因主线 Stage 0 起即 LLM 驱动(用户确认不做非 LLM 预备阶段),此项是 Stage 0/1 的前置工作,不再后置到 Stage 2。

## Architecture

### 整体数据流(与 Push-T 同构)
```
   LLM pillar 生成器 G_t  --(generate_pillars)-->  pillar 坐标
                                   │
                                   ▼
   NavEnvAdapter (set_layout -> reset)   obs(44) = [本体+goal_lidar(28) | pillars_lidar(16)]
                                   │
                                   ▼
   技能条件策略:  state=obs[0:28],  z=encoder(obs[28:44]),  w∈codebook{w_0..w_K}
                  a = decoder(cat([state, z, w])) ∈ R^2 (tanh, 确定性)
                                   │
                                   ▼
   技能条件 critic  Q(obs, w, a)
   训练采集 w_0 + probe w_1..w_K 入 buffer 打 w 标签
   课程:周期 W_probe 探针 -> p_i/b_i/boundary -> LLM 出题 -> 单标量治锁 verifier
   部署:固定 w_0,单技能单阶段
```

### 三条改动主线(相对已跑通的最简 TD3)
1. **策略/网络**(`nav/td3.py` → 扩展):actor 加 z-encoder(pillars_lidar→z)+ w 输入 + one-hot codebook;deterministic 保持。
2. **训练**(`nav/train_skill.py` 新):executed-w-only actor loss + w-条件 critic + skill-labeled buffer + reward routing(w_0→reward_task,probe→reward_total);Form B 多样(V0 关,β_div=0)。
3. **课程/验收**(Stage 2):库 boundary 信号 `b=p(1−p)` + 单标量治锁 verifier + LLM pillar 生成器进化。

### 技能的形式化定义(同 Push-T)
```
π_t^k(a|s,obs) = D_θt( s, E_ηt(obs_pillar), w_k )
S_t = { D_θt(s, E_ηt(obs_pillar), w_k) }_{k=0..K}
w_0 = 部署技能(独立,不进 probe);W_probe = {w_1..w_K} = 探针库
```
z 只承载障碍情形(pillars_lidar 的编码,唯一障碍通道),w 承载"用哪种打法/路径解"。

## Components and Interfaces

### 组件 0:NavEnvAdapter(已完成,`nav/nav_adapter.py`)
- `set_layout(pillars, start, goal)` → `reset(seed) -> obs(44)` → `step(a) -> obs,r,term,trunc,info`。
- `obs2state`/`obs2zinput`/`success`;柱数可变、obs 恒 44;非法布局抛 `ResamplingError`(= 生成器 valid 门)。
- **需小幅扩展(对齐 Push-T:目标固定、起点采样)**:
  - between 起点采样模式(验证分布):先固定 pillar,再按 between 规则采 agent 起点(某 pillar 落在 start→goal 之间),goal 固定。
  - 探针起点模式:每个 scene 采一个 start(方向探针随机 + resample;验收探针 fixed_starts paired + zero),同 scene 内 K 技能共享该 start。
  - 其余接口不改。

### 组件 1:技能条件确定性策略(`nav/policy.py`,由 `td3.py` 的 Actor 扩展)
```
state = obs[:, 0:28]
z     = z_encoder(obs[:, 28:44])          # MLP: 16 -> d_z(建议 8),确定性(非采样)
a     = decoder(cat([state, z, w]))       # MLP -> 2, tanh
```
- **结构与 Push-T 严格一致(用户确认)**:保留 `z=encoder(障碍通道)` 的确定性编码瓶颈 + `w` 拼接,不把 pillars_lidar 直接喂 decoder;这样 z/w 正交、部署路径与 Push-T 同构、可复用同一套 executed-w/critic 设计。
- `codebook`:常量张量 `[K+1, d_w]`,one-hot,`d_w=K+1`,索引 0=w_0、1..K=W_probe,不参与梯度。
- `predict_action(obs)`:默认拼 w_0(部署兼容);`predict_action_with_skill(obs, w)` / `decode_with_skill(state, z, w)` 供训练/probe。
- **z 是确定性 MLP 编码**(非随机采样),保持"单确定性策略"motivation;可选保留 DIVO 式 gaussian 正则 `L_z`(把 z 的 batch 均值/方差拉向 N(0,1))。
- K 默认 4,支持 {4,8,16}。

### 组件 2:技能条件 critic(`nav/td3.py::Critic` 扩展)
- `Q(obs, w, a)`:twin critic 内部 `q(cat([obs, w, a]))`;target/loss 处所有 critic 调用带该转移 w 标签。

### 组件 3:actor 损失(executed-w-only,`nav/train_skill.py`)
```
L_actor = L_exec (+ L_z)
L_exec  = -E_{(obs, w_exec)} Q(obs, w_exec, D(s, E(obs), w_exec))   # 按 buffer w 标签 mask
```
- 只对该转移实际执行过的 w 更新;不写 all-w L_cov 主项(避免 off-policy 外推)。命名 `L_exec`(probe 的 Q 含 reward_total)。

### 组件 4:多样性目标(Form B,`nav/diversity.py`,V0 默认关 β_div=0)
- **轨迹嵌入 φ**:agent (x,y) 路径,在 **start→goal 任务坐标系**(平移+旋转+按 |goal−start| 归一化)下定长重采样展平。
- **同环境配对**:同一批布局 seeds 内枚举 w_1..w_K,距离只在同 seed 内算。
- **软 near-success 门控**:`q(τ)=α·success+β·progress−γ·collision(cost_pillars)−η·timeout`,`g=σ((q−τ_q)/T)`;`r_div` 与 Push-T 同式,margin 有界。
- reward 双记:`reward_task`(原生 dense)、`reward_total=reward_task+β_div·reward_div`;w_0 回合不加 r_div(部署不被污染)。
- Form A(动作方差)仅监控/塌缩探测,不当目标。

### 组件 5:β_div 约束式调度(同 Push-T,V0 不启用)
`avg_success<floor → β_div*=0.5`;`success≥floor 且 K_eff<eff_floor → β_div*=1.2`;clip 到 bound;按 eval 周期慢调;floor 一律基于 reward_task/真实 success。

### 组件 6:回合采集与 Replay Buffer(`nav/train_skill.py` + `nav/td3.py::ReplayBuffer` 扩展)
- 采 w_0 与每个 w_k 的回合;每转移打 `skill_id`;`P(w_0)≥0.5`,其余 0.5 在 w_1..w_K 均分。
- buffer 增列 `skill_id / reward_task / reward_div / reward_total / source`;critic 按 w_0→reward_task、probe→reward_total 路由。

### 组件 7:库可学性 boundary 信号(`nav/skill_signal.py`,新;逻辑同 Push-T)
对当前分布采 M 个临时布局 `e_i`,用 W_probe 探针:
```
Y_{i,k} = 1[π_{w_k} 到达 goal(e_i)]        # 只用 K 探针,不含 w_0
p_i = (1/K)Σ_k Y_{i,k};  b_i = p_i(1−p_i)
boundary: τ<p_i<1−τ(K=4 → τ=0.125);boundary_count/rate、r_hard/r_easy/target_rate/mean_b
```
- 探针 = 确定性 rollout(单次即定);**同一 scene e_i 内 K 技能共享同一 start(paired)打分**(单技能二值、p_i 分数性只来自库多样性,保护 motivation,见 Testing Strategy 职责分离);`w0_success_rate` 仅部署诊断,不进 p_i。
- 方向探针(prompt)与验收探针(verifier)分工同 Push-T:方向探针随机起点 + resample(只在 VALID 场景度量库能力),验收探针固定起点 paired(G_t 与候选同题打分);二者均不改起点随机化到"分数型单策略 p"的程度。
- **Stage 1 核心验证**:dead-end 类 p_i 回到 (0,1)、围死 goal 仍为 0 → boundary_count>0、两类可分(motivation 的正面反证)。

### 组件 8:LLM pillar 生成器进化(Stage 2,`nav/curriculum/`)
- 生成器契约 `generate_pillars(agent_start, goal, num) -> [(x,y)]`;进化方向 RELAX/HARDEN/PRESERVE 按 p 分布(Eurekaverse 式)。
- **单标量治锁 verifier**:绝对硬门(code_pass + valid_rate≥v_min + duplicate_rate≤d_max)+ 单标量 `boundary_count` 在 {G_t}∪候选 argmax + 计数净改进 + 对称饱和逃逸(r_easy/r_hard)+ 方向单调护栏。
- **代码复用策略(设计决策)**:Push-T 的 `learnable_frontier.py`/`candidate_verifier.py` 判据逻辑是 env-agnostic 纯 numpy,**抽为共享模块或在 `nav/curriculum/` 轻量重实现**;`skill_signal` 的探针执行换成 NavEnvAdapter。LLM 侧(prompt/acgs/sandbox)需移植到 nav 契约与安全校验。
- 物理 verifier:候选采样 N 个临时布局,查柱数/size/在界内/互斥 keepout/不与 start/goal 净空冲突/`reset` 有效(不抛 ResamplingError)/不平凡围死 goal → valid_rate。

### 组件 9:部署(单技能单阶段)
`a = decoder([state, encoder(obs), w_0])`;不引入任何多技能推理;best-of-K 仅作 ablation。

## Data Models

### Codebook
常量 `[K+1, d_w]` one-hot,`d_w=K+1`;不参与梯度;升级 K→8→16 每次重测 K_eff。

### Replay Buffer 扩展
现有 `obs/act/rew/next_obs/done` 增 `skill_id`、`source∈{normal,diversity_paired}`、三列奖励 `reward_task/reward_div/reward_total`;`reward_for_critic` 按 skill 路由;采样返回 skill_id + 三列奖励。

### 场景/信号记录(NavGeneratorScore)
`boundary_count / boundary_rate / mean_b / r_hard / r_easy / target_rate / valid_rate / duplicate_rate / w0_success_rate`。

### 配置参数(第一版建议默认)
| 参数 | 含义 | 默认 |
|---|---|---|
| `K` | probe 技能数 | 4 |
| `d_w` | 技能码维度(one-hot) | K+1 |
| `d_z` | z 编码维度 | 8 |
| `P(w_0)` | 部署技能采样占比 | ≥0.5 |
| `beta_div` | 多样奖励系数 | 0(V0 关) |
| `tau_saturation / target_delta` | boundary 带 / 近 0.5 | 0.125 / 0.15 |
| `v_min / d_max` | 生成器硬门 | 0.8 / 0.25 |
| `min_boundary_count_delta` | 计数净改进 | 4 |
| `r_easy_max / r_hard_max` | 饱和逃逸(对称) | 0.8 / 0.8 |
| `max_ep_steps` | 回合上限 | 500 |
| `use_dual_critic` | 双 critic 后备 | False |

### 代码改动点映射(nav/)
| 文件 | 状态 | 内容 |
|---|---|---|
| `nav/nav_env.py` | ✅ | 常量 / obs 切分 / NavPillarTask / 模板 / DOF |
| `nav/nav_adapter.py` | ✅ | 运行时布局注入 |
| `nav/td3.py` | ✅→扩展 | 加 z-encoder + w 输入 + w-条件 critic |
| `nav/policy.py` | 新 | 技能条件确定性策略封装(codebook / predict_action_with_skill) |
| `nav/train_skill.py` | 新 | executed-w loss + reward routing + skill-labeled buffer + 采集 |
| `nav/diversity.py` | 新(V0 可空跑) | Form B 轨迹嵌入 + r_div |
| `nav/skill_signal.py` | 新 | W_probe 探针 + p_i/b_i/boundary 统计 |
| `nav/curriculum/` | 新(Stage 2) | 治锁 verifier(可复用 Push-T 逻辑) + LLM pillar 生成器 |
| `nav/eval.py` | 新 | B/M/U/D 评测 + 固定验证分布选 best |

## Correctness Properties

### Property 1: z 通道正交性
障碍信息只经 `z=encoder(pillars_lidar)` 进 decoder;w 不承载障碍信息。
**Validates: Requirements 1.4**

### Property 2: 部署可比性
部署 `a=decoder([state, encoder(obs), w_0])` 单技能单阶段,与单策略 TD3 baseline 严格可比。
**Validates: Requirements 9.1**

### Property 3: executed-w 一致性
actor 损失只在实际执行过 w 的转移上更新,critic 不对未执行 (obs,w) 外推。
**Validates: Requirements 3.2**

### Property 4: probe 源一致性
probe 注入源 = 训练库 W_probe(与训练同一组 w),消除 random-z 失配。
**Validates: Requirements 7.1**

### Property 5: 信号复活(核心,与 motivation 对偶)
**在生成器采样场景(G_t / 训练 DOF / 参数化分布)上**:单确定性策略的 `p_i∈{0,1}`(boundary_count≈0);技能库把 `p_i` 复活到含中间值(boundary_count>0),从而 `p(1−p)` 重新给出可用的课程梯度。SHALL NOT 用手工 TEMPLATES 度量此性质。此为 Testing Strategy 的核心假设验证(依赖库不塌、技能有能力),非架构自动保证。
**Validates: Requirements 7.2**

### Property 6: 课程不锁
单标量 boundary_count argmax + 对称饱和逃逸保证成熟期不棘轮死锁。
**Validates: Requirements 8.5**

### Property 7: 无评测泄漏
best ckpt 用固定验证分布上 w_0 success 选,不用漂移的 G_t、不用 B/M/U/D。
**Validates: Requirements 11.1**

## Error Handling
- **非法布局**:NavEnvAdapter reset 抛 `ResamplingError`,采集/探针/verifier 层重采样并计入 valid_rate。
- **w 塌缩**:周期 K_eff 监控;长期不达标 → Stage 1 go/no-go 判负,回风险分析。
- **多样失败**:SatRate 报警 + Form B margin 有界规避。
- **部署被拖累**:每 eval 对比 w_0 的 B/M/U/D 与单策略 baseline;掉则降 β_div / dual critic / 回退。
- **课程卡死**:饱和逃逸出更难题;硬可行门防不可行刷屏。

## Testing Strategy

### 评测口径(与 Push-T 实现严格一致,只换任务;禁 p-hacking)
Push-T 已核实事实:**目标(target)固定、起点(start)采样**;验证与课程探针用**两套不同的起点顺序**。导航照搬这套,只把 T-block→agent、generate_obstacles→generate_pillars。

- **验证分布(选 best)= DIVO `between`:先放障碍,再采起点**(照搬 Push-T `llm_eval_env`):
  - 先放置 `num≈2` 的 pillar(固定中等难度分布),**再把 agent 起点采在"某 pillar 与 goal 之间"**(nav 版 between:起点采样使一根 pillar 落在 start→goal 连线附近),goal 固定。
  - 验证走普通 `reset()`(不注入课程 G_t 布局),分布**固定、独立于 G_t、与 B/M/U/D 不重叠**。
  - 与 Push-T 一致的口径:`n_env_validate=20`、`validate_steps=5000`、每回合取末步指标(Push-T 的 `reward=last`;nav 用**到达 success / 末步负距离**作 `test_mean_score`),`TopKCheckpointManager(monitor_key=test_mean_score, mode=max, k=10)`。**只用 w_0 部署**,SHALL NOT 用 G_t、SHALL NOT 用 B/M/U/D 选。
  - 实现:`NavEnvAdapter` 增 between 起点采样模式(先固定 pillar,再按 between 规则采 agent 起点;goal 固定)。
- **课程场景 / 探针 = "先起点,再生成器"**(照搬 Push-T `sample_generator_scene`):
  - `start` 先采(方向探针随机起点 + `resample`;验收探针 `fixed_starts` paired + `zero`),再 `generate_pillars(start, goal, num)` 条件于起点生成 pillar。
  - **同一 scene 内 K 技能共享同一 start**(paired),p_i = (1/K)Σ_k 1[w_k 解 (start, pillars)]。
- **起点处理的职责分离(关键正确性约束,保护 motivation)**:
  - motivation 的本质不是"固定起点",而是"**同一 scene 内 K 技能共享同一起点**":单策略在该起点二值,p_i 的分数性**只来自技能多样性(库)**。
  - 验证/最终评测用"采样起点 + 平均"(每回合独立起点,降噪、贴 DIVO);探针每个 scene 一个共享起点。二者都不会让单策略在**单个 scene**上得到分数型 p(那需要对同一 scene 的一个策略跨多起点平均,本设计任何环节都不这么算 p_i)。
- **B/M/U/D(参数化 pillar 类别 + 均匀随机起点,贴 DIVO)**:只作最终零样本测试,不参与选 best;goal 固定,pillar 按类别随机放置,agent 起点均匀采样(排除靠近 goal/碰撞),每类 20 回合平均。nav 映射(对应 Push-T `eval.yaml` 的 B=num1/0.05、M=num3/0.03、U=num1/0.08 unstructured、D=num3/0.03 random_step):
  | 族 | Push-T | nav pillar 提案(size 待标定) |
  |---|---|---|
  | B | 1 大障碍 | 1 大 pillar(size≈0.30) |
  | M | 3 障碍 | 3 pillar(size≈0.15) |
  | U | 1 不规则(unstructured) | 1 大不规则障碍(大 pillar 或 pillar 簇) |
  | D | 3 + random_step(动态) | 3 动态障碍(safety-gym **gremlins** 移动障碍,真动态版;或 3 pillar 退化等价) |
  - 说明:Push-T 现码 `random_step` 实际等价 `random`;nav 的 D 建议用 gremlins 做**真动态**以恢复 D 的区分度(可选,标注)。
- **结构化拓展套件(仅一个可选 eval 拓展)**:`nav/nav_env.py::TEMPLATES` 的手工 barrier/maze/u/dead-end,仅作 B/M/U/D 之外的可选零样本 eval 基准报告;SHALL NOT 用于训练/信号探针/motivation/选 best。
- **realized**:随训练上升(闭环验证)。
- **K_eff**:固定协议 `K_eff=exp(H(P_cluster))`,task-frame 轨迹嵌入聚类;仅监控不塌。
- **多样性**:within/across 轨迹距离(task frame);成功率不塌(防多样失败)。
- **可视化**:同布局 K 技能 agent 路径叠图(扇形/绕行分化)、B/M/U/D rollout 叠图。

### 分阶段执行(go/no-go)
**Stage 0(LLM 生成 G_0,导航版;主线,不做非 LLM 预备阶段)** — 移植 LLM 生成器基建到导航契约:LLM 生成 `generate_pillars(agent_start, goal, num)` 初始生成器 G_0,选择只用 code_pass + physical validity(keepout/净空/reset 有效/不平凡围死 goal)+ 轻量难度散布;此时无 W_probe,不用 p/b/boundary 选。
- **决策(用户确认)**:核心因果("库把信号复活")已在 Push-T 验证,导航**不再单设固定/随机生成器的预备阶段**,直接走 LLM 主线。`sample_training_layout` 仅作代码冒烟/单测,不作主训练分布。
- 需移植:LLM 调用(acgs/api)、nav prompt builder、nav sandbox executor(校验 `generate_pillars` 契约与安全),复用 Push-T 的 prompt/verifier 逻辑骨架。

**Stage 1(技能库 + 信号复活,当前 todo #3 的核心)**
- 内容:加 w + one-hot K=4 + 独立 w_0 + executed-w loss + w-条件 critic + (V0 关多样)。
- **训练分布 = G_0 动态采样**(LLM 生成,主线);`sample_training_layout` 仅冒烟用。
- **核心 go/no-go(与 motivation 对偶)**:
  1. **信号复活**:在**生成器采样场景(G_0 / 训练 DOF)**上,单策略 `p_i∈{0,1}`(boundary_count≈0),库 `p_i` 出现中间值、`boundary_count>0`。SHALL NOT 用手工 TEMPLATES 度量。
  2. `K_eff ≥ 0.5K`(不塌);per-skill 非平凡:`Progress(w_k)≥τ_p` 且库内 `Success≥τ_s` 比例 ≥ r_s。
  3. w_0 部署在验证分布不明显低于单策略 baseline;B/M/U 不明显掉。
- **观察项(非硬门)**:dead-end(D)成功率是否提升(库多样是否帮到回退类)。
- 硬门长期不达标 → 回风险分析,不硬堆组件。

**Stage 2(LLM 库可学性课程 + 治锁 verifier)**
- 接入:W_probe 探针源、`b=p(1−p)`、三段方向、单标量 boundary_count argmax + 饱和逃逸 + 方向护栏、LLM pillar 生成器进化。
- 验证:生成器随库能力边界演化、不卡死、boundary_rate 维持;w_0 仅部署诊断不作验收门。
- 主对比:(d) 库信号课程 vs (b) 单策略课程 vs (c) 库无课程 vs (a) 单策略 baseline,同协议 B/M/U/D,≥3 seeds;红线 (d)>(c) 且 (d)>(b)。

**Stage 3(完整对比 + 消融)**
- 对齐社区口径:random / static / eureka(success 引导)/ 本方法(库信号引导)。
- 消融:训库/不训、多样 on/off、单技能/挑技能、K∈{4,8,16}、β_div、rollout ratio;最难的 B/M/U/D 类作为重点观察(哪类单策略系统性失败、库最能帮)。

### 已跑通的验证件(可直接用作 Stage 1 基线/图)
- `nav/train_min.py` + `nav/runs/motiv0/`:单策略 baseline(训练分布 success=1.0)。
- `nav/probe_degeneracy.py`:单策略退化证据(boundary_count=0)。**注:现原型用了手工 dead-end/goal-ring 场景,按新约束需改为在生成器采样场景(G_0/训练 DOF)上度量**;Stage 1 扩为 K 技能版对比"库把 p 复活"时一并改。
- `nav/checks.py`:结构化拓展套件(手工 barrier/maze/u/dead-end)难度量化 + 渲染(诊断用,非主 B/M/U/D)。

## 设计中的诚实保留(不掩盖)
1. **H1/H2(库帮不帮部署)迁移仍未决**:靠 (d)-vs-(b) ablation 定;若库只帮"出题"不帮"部署",主张需重述为"多技能作课程标尺"。
2. **信号复活是假设非保证**:Stage 1 go/no-go 检验;若库对 dead-end 仍全 p=0(技能不够多样/回退能力弱),核心主张证伪。
3. **LLM 基建移植成本(前置)**:主线 Stage 0 起即 LLM 驱动,`generate_pillars` 契约的 LLM 调用/prompt/sandbox 需在 Stage 1 之前移植就绪(用户确认不做非 LLM 预备阶段,因 Push-T 已验证核心因果)。
4. **算力**:K 技能探针 rollout 成本随 K 线性增长,扩 K 前先在 K=4 见效。
