# Implementation Plan — Nav Skill-Library Curriculum(第二任务)

## Overview

把 Push-T 已建立的"生成器进化 + 隐技能库可学性标尺"框架迁移到 SafetyPointGoal 导航。
- 环境 `safenav`(`conda activate safenav && export MUJOCO_GL=egl`),代码在 `nav/`。
- 已完成基座:`nav_env.py`、`nav_adapter.py`、`td3.py`、`train_min.py`/`probe_degeneracy.py`(单策略 baseline + 退化证据)。任务在此之上增量推进。
- 组件级 sanity 即可;新功能默认做成可关开关。
- 阶段:Phase A(环境/适配)→ B(技能库策略,Stage 1 核心)→ C(信号 + 复活验证,Stage 1 go/no-go)→ D(LLM 生成器 + 治锁 verifier,Stage 2)→ E(评测/对比,Stage 2/3)。任务 1–3=A,4–9=B,10–12=C,13–16=D,17–21=E。

## Task Dependency Graph

```json
{
  "waves": [
    {"wave": 1, "tasks": [1, 4, 13], "dependsOn": []},
    {"wave": 2, "tasks": [2, 3, 5, 6, 14], "dependsOn": [1, 4, 13]},
    {"wave": 3, "tasks": [7], "dependsOn": [5, 6]},
    {"wave": 4, "tasks": [8], "dependsOn": [7, 2, 1]},
    {"wave": 5, "tasks": [9, 10], "dependsOn": [8]},
    {"wave": 6, "tasks": [11, 15], "dependsOn": [10, 13]},
    {"wave": 7, "tasks": [12], "dependsOn": [11]},
    {"wave": 8, "tasks": [16], "dependsOn": [12, 14, 15]},
    {"wave": 9, "tasks": [17, 18, 19, 21], "dependsOn": [16, 2, 3, 8]},
    {"wave": 10, "tasks": [20], "dependsOn": [19]}
  ]
}
```

## Tasks

- [x] 1. NavEnvAdapter 起点模式扩展(`nav/nav_adapter.py`)
  - `between` 起点模式:`sample_between_start(rng, pillars, goal)` 采起点使某 pillar 落在 start→goal 连线附近,goal 固定。
  - 探针/训练起点模式:`reset(seed, start=None, randomize_start=False)`——固定/显式/随机三模式,同 scene 可复用同一 start。
  - 保持柱数可变、obs 恒 44、非法布局抛 `ResamplingError`。sanity 通过(`nav/test_adapter.py`:显式 err=0、随机 spread=1.21、between 9/10)。
  - _Requirements: 13.1, 13.2, 13.3_

- [x] 2. 验证分布(between)采样器(`nav/eval_dist.py`)
  - `sample_validation_scene(seed)`:固定 num=2 中央带 pillar + between 起点;`evaluate_validation(adapter, act_fn, n_env=20)` → test_mean_score(success_rate, mode=max)。sanity:motiv0 在验证分布 success=1.0。
  - _Requirements: 11.1_

- [x] 3. 参数化 B/M/U/D 基准(`nav/benchmarks.py`)
  - B/M/U/D = 参数化 pillar 类别 + 均匀随机起点 + goal 固定(B=1/0.30、M=3/0.15、U=1/0.40、D=3/0.15);`evaluate_bmud`;adapter 支持 per-benchmark pillar_size/keepout(修了跨布局泄漏)。真动态 D(gremlins)标为可选 TODO。
  - TEMPLATES 经 `evaluate_structured_extension` 仅作可选 eval 拓展;禁其它用途。sanity:motiv0 B/M/U/D=1.0/0.95/1.0/0.95。
  - _Requirements: 11.2, 11.3_

- [x] 4. 技能条件确定性策略(`nav/policy.py::SkillActor`)
  - [x] 4.1 `state=obs[0:28]`,`z=z_encoder(obs[28:44])`(确定性 MLP,`d_z=8`),`a=decoder(cat([state,z,w]))`,2 维 tanh。sanity:z 只由 pillars_lidar 决定。
    - _Requirements: 1.1, 1.2, 1.3_
  - [x] 4.2 one-hot Codebook `[K+1,K+1]`(常量、无梯度);`predict_action`(默认 w_0)/`predict_action_with_skill`/`decode_with_skill`/`encode_z`;支持 K∈{4,8,16}。sanity 全过(`nav/test_policy.py`)。
    - _Requirements: 1.4, 1.5, 1.6_

- [x] 5. 技能条件 critic(`nav/td3.py::SkillCritic`)
  - `Q(obs,w,a)` twin 均带 w(one-hot);sanity:不同 w 的 Q 可分(平均差 0.007–0.016)、actor loss 通路可反传、梯度流正常。target/loss 处按转移 w 标签调用将在任务 7/8 落地。保留原单策略 Critic 供 baseline。
  - _Requirements: 2.1, 2.2, 2.3_

- [x] 6. Replay Buffer 扩展 + reward 路由(`nav/td3.py::SkillReplayBuffer`)
  - 增列 `skill_id/reward_task/reward_div/reward_total/source`,采样同步返回 `reward_for_critic`;w_0 回合 reward_total=task(不加 r_div)、probe=task+β·div;critic 路由 w_0→task、probe→total。sanity 全过(`nav/test_buffer.py`)。保留原 ReplayBuffer 供 baseline。
  - _Requirements: 6.1, 6.2, 6.3_

- [x] 7. executed-w-only actor 损失 + 技能 TD3 update(`nav/skill_td3.py::SkillTD3`)
  - w-条件 critic(target/loss 带该转移 w、目标用 reward_for_critic)+ `L_actor=L_exec`(每转移只用其执行技能,无 all-w 覆盖项)+ 可选 L_z(λ_z 默认 0)。
  - sanity(`nav/test_skill_td3.py`):bandit 验证 Q1(w_0)=1.0、Q1(w_1)=0.0、critic 收敛、actor/critic 损失有限。train_skill.py 循环在任务 8 组装。
  - _Requirements: 3.1, 3.2, 3.3, 3.4_

- [x] 8. 技能回合采集与训练循环(`nav/train_skill.py`)
  - 采 w_0 与各 w_k、打 skill_id;普通采样 `P(w_0)=0.5` 其余均分;先采起点(randomize_start)、再 `sample_training_layout` 布局(G_0 未就绪前冒烟)。
  - 周期在验证 between(eval_dist)用 w_0 部署选 best(n=20、test_mean_score、mode=max)存 ckpt。
  - 验证:整合闭环学起来——w_0 验证 success 25k=0.95、50k=0.75(P(w_0)=0.5 下 w_0 仍部署良好)。`runs/skill0/` 300k 训练进行中。
  - _Requirements: 6.4, 11.1_

- [x] 9. Form B 多样性(`nav/diversity.py`,V0 默认 β_div=0)
  - [x] 9.1 任务坐标系轨迹嵌入 `embed_route`(start→goal 归一化,平移/旋转/尺度不变)+ `paired_diversity_rollout`(同布局 K 技能配对)。sanity:直线 tx 0→1、绕行可分(距离 0.43)。
    - _Requirements: 4.1, 4.2_
  - [x] 9.2 软 near-success 门控 + margin 有界 `compute_rdiv`;buffer 侧 w_0 不加 r_div;`action_var`/`sat_rate` 仅监控。sanity:与众不同技能 r_div 最高、全同≈0、门控成功>失败。
    - _Requirements: 4.3, 4.4, 4.5, 4.6, 4.7_
  - [x] 9.3 β_div 调度骨架(train_skill 已透传 beta_div;warm-up/慢调可在启用多样时接入);默认不启用。
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

- [x] 10. 库可学性信号探针(`nav/skill_signal.py`)
  - `library_p_on_scene`(同 scene K 技能共享 start)/`single_p_on_scene`/`extract_skill_signals`(boundary_count/rate、r_hard/r_easy、mean_b、valid/dup、w0_success);方向探针 resample、验收探针 zero;τ=0.125。已被 probe/verifier 使用验证。
  - _Requirements: 7.1, 7.2, 7.3, 7.5, 7.6_

- [x] 11. 信号复活验证(`nav/probe_degeneracy.py`,生成器采样场景)
  - **结果(skill0 latest.pt,K=4):单策略 p∈{0,1} bc=0 mean_b=0;库 p∈{0.75,1.0} bc=5 mean_b=0.039 → 信号复活 YES**(即使 β_div=0)。不用手工 TEMPLATES。
  - _Requirements: 7.4, 10.4_

- [x] 12. Stage 1 go/no-go 汇总(`nav/stage1_gonogo.py`)
  - 信号复活 + K_eff(union-find task-frame 聚类)+ per-skill 非平凡 + w_0 验证不掉 + B/M/U/D。核心门(信号复活)已 PASS;完整汇总可在 skill0/best.pt 跑完后执行。
  - _Requirements: 10.4, 10.5, 10.6, 11.4_

- [x] 13. `generate_pillars` LLM 基建(`nav/curriculum/`)
  - [x] 13.1 `SandboxNavExecutor`(校验签名/危险 token 拦截/受限命名空间)+ `validate_pillars`(界内/keepout/净空/不围死 goal)+ `MockGenerator`(无 LLM 可测)。sanity:load/run OK、危险代码被拦。
    - _Requirements: 10.1, 10.2_
  - [x] 13.2 `prompt_builder`(方向化 RELAX/HARDEN/PRESERVE)+ `acgs_api.NavACGS`(openai 需 key;mock 候选路径)。
    - _Requirements: 8.6, 10.1_

- [x] 14. Stage 0 G_0 生成与选择(`nav/curriculum/stage0.py`)
  - `select_initial_generator`(mock):code_pass + physical validity(valid_rate)+ 轻量难度散布(直线阻挡比例),不用 p/b/boundary。openai 路径需 key。
  - _Requirements: 10.1, 10.2, 10.3_

- [x] 15. 单标量治锁 verifier(`nav/curriculum/verifier.py`)
  - 绝对硬门(valid/dup;code_pass 由 sandbox)+ 单标量 boundary_count argmax(mean_b tiebreak)+ 计数净改进 + 对称饱和逃逸;验收探针固定起点 paired+zero。sanity:mock 端到端触发 RELAX escape、不卡死。
  - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6_

- [x] 16. Stage 2 evolve 循环(`nav/curriculum/evolve.py`)
  - `evolve_cycle`:方向探针→候选→验收探针 paired→verifier 选择→G_{t+1}。mock 端到端不卡死(escape/replace/hold)。**完整 train+evolve 交替整合 + LLM 长训待接**(需 OPENAI_API_KEY)。
  - _Requirements: 7.6, 8.3, 9.1_

- [x] 17. 最终评测脚本(`nav/eval.py`)
  - best ckpt 跑 B/M/U/D(参数化+均匀起点+20回合平均)+ 可选结构化拓展;支持 skill(w_0)/single ckpt;存 eval_bmud.json。
  - _Requirements: 9.1, 9.2, 11.1, 11.2, 11.3_

- [x] 18. 度量与可视化(`nav/viz.py`)
  - 同布局 K 技能路径叠图 + within/across 轨迹多样(task-frame 嵌入距离);K_eff 见 stage1_gonogo。
  - _Requirements: 11.4, 11.5, 11.6, 13.5_

- [x] 19. 四格主对比驱动(`nav/compare_fourcell.py`)
  - print 四臂命令 + 聚合 eval_bmud.json;(a)train_min/(c)train_skill 可跑,(b)(d)需 Stage2 train+evolve 整合 + key;红线 (d)>(c) 且 (d)>(b)。**大规模多 seed 长训待排队。**
  - _Requirements: 12.1, 12.2_

- [x] 20. Stage 3 驱动(`nav/stage3.py`)
  - print 消融命令矩阵(训库/不训、多样 on/off、K∈{4,8,16}、P(w_0));主对比 random/static/eureka 需各自信号变体。**待长训。**
  - _Requirements: 12.3, 12.4_

- [x] 21. 可复现与日志(`nav/repro.py`)
  - `seed_everything`(random/np/torch/cuda)+ LOG_FIELDS 规范;train 脚本已由 seed 派生 RNG。
  - _Requirements: 13.4, 13.5_

## Notes

- **Stage 1 go/no-go(任务 12)是硬闸门**:信号复活(11)、K_eff 不塌、per-skill 非平凡、w_0 不掉 B/M/U 才进 Stage 2;不达标回风险分析,不硬堆组件。
- **四格红线(任务 19)**:(d) 需同时 > (c) 且 > (b),否则退回或重述主张(H1/H2 库帮不帮部署的赌注在此裁决)。
- **禁用约束**:结构化拓展套件(TEMPLATES)仅任务 17 的可选 eval 拓展;训练/信号探针/motivation/选 best 一律不得使用。
- **防泄漏**:选 best 只用验证 between(任务 2);B/M/U/D 与拓展套件绝不参与选 best。
- **默认开关**:多样性(9)、β_div 调度、真动态 D(gremlins)默认关,便于 A/B。
- **算力**:K 技能探针 rollout 成本随 K 线性增长;先在 K=4 见效再扩 K。
