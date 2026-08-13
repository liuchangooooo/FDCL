# nav — 导航任务(第二任务:跨范式泛化)

目标:把"生成器进化 + 隐技能库难度标尺"从物体操作(Push-T)迁移到自主导航,
证明框架任务无关。基于 **Safety-Gymnasium** 的 `SafetyPointGoal`(Point 智能体
+ Goal 任务),用 **Pillars**(可碰撞刚体)作难度主元造 B/M/U/D。

## 环境
独立 conda env(与 Push-T 的 `divo` 隔离,因依赖版本冲突):
```
conda activate safenav      # safety-gymnasium 1.0.0 / gymnasium 0.28.1 / mujoco 2.3.3
export MUJOCO_GL=egl
```

## 文件
- `nav_env.py` — **单一事实来源**:坐标系、Pillar/Gremlin 参数、
  obs→(state,z) 切分、`NavPillarTask`、`make_env`、固定两障碍训练采样器，
  以及可选的 `STRUCTURED_STRESS_TESTS`。
- `nav_adapter.py` — `NavEnvAdapter`:运行时布局注入(对应 Push-T `set_obstacle_config`)。
  建一次 env,`set_layout` 换布局(柱数可变)→ `reset`/`step`,统一对外接口；
  `dynamic=True` 时使用固定错相、连续运动的 Gremlin。
- `benchmarks.py` — 正式参数化 B/M/U/D held-out 协议与配对的
  `D_static`/`D_dynamic` 评测。
- `checks.py` — 合并的验证脚本(obs 结构 / 渲染 / 难度量化)。
- `test_adapter.py` — adapter 组件级 sanity(布局切换 / step / 非法布局)。
- `templates/nav_bmud_v2_train2_dynamic_pair/` — 当前正式协议渲染；根目录旧图已归档为 legacy。

## 位置与运行
代码位于 `/home/hnu-w/DIVO/DIVO/nav`,从 `/home/hnu-w/DIVO/DIVO` 目录下以 `python -m nav.*` 运行。
```
cd /home/hnu-w/DIVO/DIVO
python -m nav.checks all         # obs 结构 + 难度检查 + 渲染
python -m nav.checks obs
python -m nav.checks render
python -m nav.checks difficulty
python -m nav.render_bmud_gallery --per 1  # 正式 B/M/U/D 图 + 真实动态 D 视频
python -m nav.test_adapter       # adapter sanity(布局切换/step/非法布局/起点模式)
python -m nav.test_benchmarks    # B/M/U/D 采样、动态配对与返回合同
python -m nav.test_protocol      # manifest 与旧结果隔离合同
python -m nav.test_prompts       # Navigate prompt 与两柱/小场地协议合同
```

## 已敲死的接口(供训练/课程复用)
| 项 | 值 |
|---|---|
| start / goal | (-0.65,0) / (+0.65,0) 默认值；起点可在统一起点区采样 |
| extents | [-0.9,-0.9,0.9,0.9] |
| pillar size / keepout | 0.15 / 0.13(固定,不进课程 DOF) |
| 训练柱中心最小间距 | 0.31（实体直径 0.30 + 数值余量） |
| gremlin size / keepout / travel | 0.10 / 0.20 / 0.08(仅正式 D) |
| Point 半径 | 0.10;可通行间隙需 ≥0.20 |
| obs (44) | state=obs[0:28],障碍切片=obs[28:44]；z-encoder 输入完整 44 维 obs |
| 动作 | Box(2,) |
| success | 原生 `goal_achieved`(dist ≤ goal.size) |
| reward | 原生 dense(距离进步 + 到达 bonus);障碍接触附带 cost |
| 训练 DOF | 恰好 2 个标准静态 Pillar,x,y∈[-0.5,0.5] |
| 生成器契约 | `generate_pillars(start, goal, num=2) -> [(x,y), (x,y)]`;数量必须等于 2 |

## 训练、选模与正式 held-out 协议
训练和课程生成始终使用 **2 个标准静态 Pillar**。`between` 是手工基线及
独立验证分布的布局原则，不是 B/M/U/D 的一部分。正式零样本测试只由
`benchmarks.py` 定义：

| 族 | 正式名称 | Nav 实例化 | 相对两障碍训练的变化 |
|---|---|---|---|
| B | Big | 1 个大 Pillar(size=0.30) | 数量与尺寸 |
| M | Multiple | 3 个标准静态 Pillar | 数量 |
| U | Unstructured | 1 个由 7 个对称元素构成、整体随机旋转/平移的 U 形复合障碍 | 形状与拓扑 |
| D | Dynamic | 3 个围绕各自中心错相连续运动的 Gremlin | 数量与时变动力学 |

Nav D 与 Push-T `random_step` 共享“测试时障碍随时间变化”的抽象因素，
但 Nav 采用任务特定的确定性错相连续轨道，各障碍的相对几何会随时间变化。评测同时保存几何、起点和种子完全配对的
`D_static`(travel=0)、`D_dynamic`(travel=0.08) 和
`dynamic_drop=D_static-D_dynamic`。主指标 `D=D_dynamic`，`AVG` 只计 B/M/U/D 四项。

`STRUCTURED_STRESS_TESTS` 中的 barrier/maze/ushape/deadend 只是额外结构压力测试；
不参与训练、选 best 或主 B/M/U/D 结果。

新训练会在 checkpoint 同目录写 `training_manifest.json`，记录
`nav_train_v2_two_static_pillars`、2 个静态 Pillar 和
`nav_bmud_v2_train2_dynamic_pair`。正式 `nav.eval` 默认拒绝无 manifest 的旧 checkpoint；
正式评测参数固定为 `n_env=20,max_steps=500,base_seed=2024`，
`--allow-unverified-checkpoint` 仅用于诊断，写入 `eval_bmud_diagnostic.json` 且不会被
`nav.compare_fourcell --aggregate` 聚合。无 manifest 的非空旧目录也不能由新训练补签，
必须使用新的 `navv2_*` tag 重训。

## 技能库模块(Stage 1/2,spec: nav-skill-library-curriculum)
- `policy.py` SkillActor(z-encoder+w codebook)、`td3.py` SkillCritic/SkillReplayBuffer、
  `skill_td3.py` SkillTD3(w-critic + executed-w loss)、`train_skill.py` 训练循环。
- `diversity.py` Form B 多样(默认关)、`skill_signal.py` 库 boundary 信号、
  `probe_degeneracy.py` 信号复活、`stage1_gonogo.py` Stage1 闸门。
- `curriculum/` Stage2:generator_source(sandbox+mock)、prompt_builder、acgs_api、
  verifier(单标量治锁)、evolve(cycle)、stage0(G_0 选择)。
- `benchmarks.py` 参数化 B/M/U/D 及 D 冻结对照、`eval.py` 最终评测、`viz.py` 路径叠图、
  `compare_fourcell.py`/`stage3.py` 对比驱动、`repro.py` 复现。
- 各 `test_*.py` 组件 sanity 全过。

### 重要语义(对齐 Push-T):碰撞即失败
`NavEnvAdapter.step`:pillar 接触(cost>0)⇒ **terminated=True + reward=-collision_penalty(默认2) + success=False**,
训练与测试同规则(照搬 Push-T `collide_obstacle -> reward=-10, done=True`)。默认开(`collision_is_failure=True`)。
所有 rollout 循环靠 term/success 判定,自动生效。

### 正式 checkpoint 语义
- `best.pt`:固定 validation 分布上历史平均 episode return 最高的模型；正式 B/M/U/D 只使用它。
- `latest.pt`:最近一次已经完成 validation 的模型，仅作诊断，不作为正式结果。
- `total_steps` 必须整除 `eval_every`；训练严格停在预算步，不跑完残余 episode，
  也不会在结束时用未经验证的权重覆盖 `latest.pt`。

> 注：所有没有 v2 manifest 的 `skill0`/`skill1_collfail` 权重均为 legacy，不得用于正式结果；
> 下方旧数字只作历史诊断，需在新的 `navv2_*` run 上重跑。

### 历史诊断结果（skill0 K=4,β_div=0；非 v2 正式结果）
```
单策略 w_0 :  p∈{0,1}      boundary_count=0  mean_b=0.000   (信号退化)
技能库 K=4 :  p∈{0.75,1.0}  boundary_count=5  mean_b=0.039   (信号复活 YES)
```
Stage 1 核心 go/no-go(信号复活)PASS——库把单确定性策略退化的 p(1−p) 信号复活。

### 待长训/需 LLM key
- 使用 `OPENAI_API_KEY` 跑完整 Stage 2 LLM 长训与生成器演进。
- 四格对比(S0/b'/c/d)≥3 seeds、Stage 3 消融:多 seed 长训排队。

## 历史待办
1. ~~`NavEnvAdapter`:运行时布局注入~~ ✅ 完成(`nav_adapter.py`,`test_adapter.py`:
   合法的 1/2/3 柱组件布局切换、obs 恒 44、坐标对齐、非法布局拒绝，
   以及 Gremlin 冻结/连续运动回归)。
2. ~~最简确定性 TD3 验 motivation~~ ✅ 完成。
   - legacy `motiv0` 曾跑到训练分布 success=1.0；v2 重跑命令：
     `python -m nav.train_min --total_steps 250000 --tag navv2_motiv0`。
   - 历史探针曾使用 legacy `nav/runs/motiv0/best.pt`；该权重不属于 v2 正式协议。
   - 结果(144 场景):p_i∈{0,1},boundary_count=0,mean_b=0.0000;
     "可行但难"(dead-end,p̄=0.083)与"不可行"(围死 goal,p̄=0.000)都塌向 0,不可分。
   - 结论:单确定性策略下 SFL 式 p(1−p) 信号退化 —— 与 Push-T motivation 一致,跨范式复现。
3. 移植隐技能库(w-codebook + executed-w + w-critic),验证库把 p_i 复活到 (0,1)、两类可分。

## 接口备忘(NavEnvAdapter)
```python
from nav.nav_adapter import NavEnvAdapter
ad = NavEnvAdapter()                                # 建一次
ad.set_layout(pillars, start, goal)                 # 换布局(pillars>=1;下次 reset 生效)
obs = ad.reset(seed=0)                              # 固定起点(默认)
obs = ad.reset(seed=0, start=(x, y))                # 显式起点(between 走这条)
obs = ad.reset(seed=0, randomize_start=True)        # 通用 Safety-Gym 全场随机起点
st  = ad.sample_between_start(rng, pillars, goal)   # between 起点坐标(验证分布)
obs, r, term, trunc, info = ad.step(action)         # action: (2,)
ad.success(info); ad.obs2state(obs); ad.obs2zinput(obs)

dad = NavEnvAdapter(dynamic=True)                   # 正式 D / D_static
dad.set_layout(centers, gremlin_travel=0.08)       # D_dynamic
dad.set_layout(centers, gremlin_travel=0.0)        # 完全配对的 D_static
```
- 正式 B/M/U/D 固定 `NE.START`/`NE.GOAL`，障碍物位置与朝向按 scene seed 随机采样；只做起终点净空与障碍间距的物理合法性拒绝，不按起点或路径构造布局。
- 非法布局(净空/keepout 不满足)`reset` 抛 `ResamplingError`,调用方重采样(= 生成器 valid 门)。
- 起点模式:固定 / 显式 / 随机 三种;正式 BMUD 对齐 Push-T 实际评测路径，固定 start/goal。
