# 图2 SLC 框架图 — 完整设计规格 (Two-tier)

面向 ICRA/IROS/TRO 的扁平矢量架构图。手工在 Illustrator/Inkscape 落地；代码只
产出可嵌入的真实小图。结构 = **CoEvolve 两层骨架** + **EurekaVerse 候选流** +
**DIVO Push-T 视觉语法**。记号严格对齐 `paper/method_routeB.tex`。

---

## 0. 全局

- 画布：双栏宽 **180 mm × 高 ~92 mm**；纯白背景，无纹理、无投影、无 3D。
- 两层：**顶层总览带**(~22 mm 高) + **底层三放大面板**(~60 mm) + 顶部 `SLC Cycle` 弧。
- 模块框：圆角 4 px，描边 1 pt，极浅填充；箭头 1.2 pt；`SLC Cycle` 弧 1.6 pt。
- 字体：Inter/Helvetica。总览节点标题 10 pt 半粗；面板标题 9 pt 半粗；子步/公式 8 pt；箭头标签 7 pt。全部**英文**。
- 语法约定（沿用 DIVO + 图1）：红 init T、绿 goal T、橙障碍、黑 EE 点、**实线=solve / 淡出置灰=fail 或 rejected**。

## 1. 淡色配色（低饱和）

| 用途 | 填充 | 描边/强调 |
|---|---|---|
| Stage 1 / Panel A (Estimation) | `#EAF1FB` | `#5B84C4` |
| Stage 2 / Panel B (Rewriting) | `#F1ECF9` | `#8A6DBF` |
| Stage 3 / Panel C (Validation) | `#E9F5EF` | `#4CA07A` |
| Learnable band（贯穿主题）+ SLC Cycle 弧 + 选中候选 | `#FBEFCB` | `#D9A63C` |
| init T | `#E0897B` / 描边 `#C56B5C` | goal T `#93C1A4` / `#6FA485` |
| obstacle | `#F0B678` / `#D79450` | EE dot `#4A4A4A` |
| 数据流箭头/中性 | `#6B7280` | 文本 `#222` |
| 候选条：current | `#AEC4E8`(浅蓝) | 候选 candidates `#C9B8E6`(浅紫) |

三阶段用蓝/紫/绿浅色分型，靠深浅区分，不用高饱和。

## 2. 顶层：Overview strip + SLC Cycle 弧

三个带图标节点，左→右，"图标+2词"箭头串联；顶部一根琥珀弧从 Stage 3 绕回 Stage 1。

```
        ╭──────────────────────  SLC Cycle  ──────────────────────╮
        │                                                          │
   [g0]→│ ①🤖 Stage 1        →  ②✎ Stage 2        →  ③✔ Stage 3  │
        │   Learnability         LLM-Guided            Validation- │
        │   Estimation           Rewriting             Gated Update│
        ╰──────────────────────────────────────────────────────────╯
   dashed↓ (zoom-in)      dashed↓                dashed↓
   [Panel A]              [Panel B]              [Panel C]
```

- 节点图标：Stage1 = 极简网络/robot；Stage2 = 文档改写 ✎（**非品牌 logo**）；Stage3 = laptop+✔。
- 节点间箭头标签：`S(gₜ)` (profile) → `R candidates` → `accept gₜ₊₁`。
- 左侧入口：小块 `Stage 0: LLM init generator g₀` → 箭头进 Stage 1。
- `SLC Cycle` 弧标签：`gₜ₊₁ → Dₜ₊₁ (co-evolution)`。
- 三条竖直虚线把每个 Stage 节点连到下方对应放大面板（CoEvolve 的 zoom-in 约定）。

## 3. Panel A — Learnability Estimation（最宽 ~40%，novelty 所在，底色 `#EAF1FB`）

顶部 novelty 对比条（琥珀高亮，全图最抢眼）：
- 左：塌平灰曲线 + `single policy: p ∈ {0,1} ⇒ lv ≡ 0`
- 右：琥珀抛物线 + `K skills ⇒ continuous lv`
- 压条标题：`Learnability from a skill library, not a single policy`

面板内横排三个子步（左上角圆形序号徽标 1/2/3）：

**① Current distribution** —— 一根横向"分布带"，左→右 realized 0→1，三段色：
`#E0897B` infeasible · `#FBEFCB` boundary · `#93C1A4` mastery；每段挂 1 张极简 Push-T 缩略图。
标签 `Training distribution Dₜ ~ gₜ`；`</>` 图标标 `gₜ`，注 `#obstacles = 2`。

**② Multi-skill probing** —— 嵌真实 `asset_B_skillfan.svg`（同布局 K 技能，实线 solve / **淡出 fail**）；
右侧 K×1 的 ✓/✗ 竖列，顶标 `w₁ … w_K`。标签 `Inject probe skills W_probe = {w₁,…,w_K}`，小字 `(excludes deploy w₀)`。
**策略 callout**（浅灰卡）：迷你网络图标，两输入进 decoder——`z = E(o)` 标 `obstacle channel`、`w` 标 `skill code`，输出 `a = D(s,z,w)`；底部强调 `Deploy: a = D(s,E(o),w₀) — single-skill, single-stage`。

**③ Learnability signal** —— 嵌真实 `asset_C_learnability.svg`（`lv=p(1−p)` 曲线 + 琥珀 τ-band + 真实点）；
右侧三行公式（只作标签）：`feasible = maxₖ ρ` / `realized = (1/K)Σₖ ρ` / `lv = realized(1−realized)`；下 `boundary: τ < realized < 1−τ`。

## 4. A→B 桥：Evolution direction（菱形决策 + 三出线）

`r_hard ↑ → RELAX` / `r_easy ↑ → HARDEN` / `else → PRESERVE & DIVERSIFY`；前置细支 `valid_rate low → FIX_VALIDITY`。每线配极简图标（↓/↑/↻/盾）。

## 5. Panel B — LLM-Guided Rewriting（~30%，底色 `#F1ECF9`）

自上而下：
- **Design context `S(gₜ)`**：横排三卡 + 图标 `focus 🎯` / `harden ↗` / `avoid ⊘`；上方小字 `realized distribution stats`。
- **LLM Rewriter**：浅紫圆角块 + 文档改写图标；标 `LLM rewrites generator program`，小字 `not a single layout`。
- **候选流（EurekaVerse 式箭头条堆）**：一摞箭头条——1 条浅蓝 = `gₜ`(current)，R 条浅紫 = `{g'⁽¹⁾…g'⁽ᴿ⁾}`(candidates)。
- 右下角目标标签（方程 5 简写）：`max_g E[lv]  s.t.  E[1(feasible=0)] ≤ τ_inf`。

## 6. Panel C — Validation-Gated Update（~30%，底色 `#E9F5EF`）

- **Paired probing**：两列同起点迷你场景并排（`gₜ` vs `candidate`），顶标 `paired probe · same W_probe · same starts`。一条虚线连回 Panel A② 探针框，标 `shared probe`（信号/方向/验收同源）。
- **验收三步竖直条**（浅绿小块 + 向下箭头）：
  1. `Hard gate: code · valid_rate ≥ v_min · duplicate ≤ d_max`
  2. `argmax boundary_count over {gₜ}∪candidates  (+δ)`
  3. `Saturation escape: r_hard / r_easy → RELAX / HARDEN`
- **输出**：选中的候选条变琥珀 = `gₜ₊₁`（呼应 EurekaVerse red=best）。
- 右下角可选 **mini-algorithm inset**（5 行伪代码缩影）：标题 `Selection (single-scalar, deadlock-free)`。
- 差异化标签：`single scalar · no deadlock`。

## 7. 闭环 + 可选分布演化条

- 顶层 `SLC Cycle` 琥珀弧回流（见 §2），不从中间穿返回箭头。
- **可选亮点**：弧下方放 2–3 联 realized 分布小直方图，从 `t=0`(偏右/mastery) → `t=T`(集中到琥珀 band)，标 `distribution shifts onto the learnable band`。真实数据可从 `phase4_monitor` 记录跑。

## 8. 数据流箭头标签（精确英文）

顶层：`g₀ → Stage1`｜`S(gₜ)`｜`R candidates`｜`accept gₜ₊₁`｜弧 `gₜ₊₁ → Dₜ₊₁`
Panel A：`Dₜ ~ gₜ` → `M sampled scenes` → `ρ ∈ {0,1}` → `learnability profile S(gₜ)`
桥：`evolution direction`
Panel B：`design context` → `LLM rewrite` → `R candidate programs`
Panel C：`paired probe` → `accept gₜ₊₁ / hold gₜ`

## 9. 图标清单（统一 1.2 pt 线性、无填充或极浅填充）

网络/robot(policy)、`</>`(generator)、✓/✗(outcomes)、抛物线(lv)、分布带/直方图(distribution)、🎯/↗/⊘(focus/harden/avoid)、菱形(direction)、盾(validity)、laptop+✔(verifier)、箭头条(candidates)、↻(cycle)。全部同线宽同圆角。

## 10. 可嵌入的真实素材（代码产出）

已就绪：`teaser_assets/asset_B_skillfan.svg`(A②)、`teaser_assets/asset_C_learnability.svg`(A③)。
可再渲：A① realized 分布带(真实 M 场景)、§7 分布演化条(跨 evolve 轮)、A① 三类 Push-T 缩略图、候选箭头条 SVG 模板、方向开关图标骨架。

## 11. 一致性铁律

- 记号只用 tex：`z / w / w₀ / W_probe / feasible / realized / lv / boundary_count / gₜ / R / δ / τ / η / r_hard / r_easy`。
  禁止 `z₁..z_K`、`L=4R(1−R)`、候选 `e`、旧 "four safeguards"。
- 场景色：红 init T、绿 goal T、橙障碍、黑 EE；实线 solve / 淡出 fail-or-rejected。
- 验收器 = **单标量 boundary_count argmax + 对称饱和逃逸**（非四门合取）。
- 全英文标签，只标身份，不写长句/推导。
- 与图1 同一视觉语言与色板。

## 12. 图注（可直接用）

> **Fig. 2.** SLC co-evolution framework. **Stage 1** trains a single-stage deploy
> policy on `Dₜ~gₜ` and probes each layout with a skill library `W_probe`, turning
> the fraction of solving skills `realized` into a continuous learnability signal
> `lv=realized(1−realized)` (a single policy gives `p∈{0,1}`, `lv≡0`). **Stage 2**
> conditions an LLM on this signal and a single evolution direction to rewrite the
> obstacle-generator program into `R` candidates. **Stage 3** paired-probes
> candidates against `gₜ` with the same `W_probe` and selects `gₜ₊₁` by a
> single-scalar, deadlock-free rule (hard gate + `boundary_count` argmax +
> symmetric saturation escape), closing the loop.
