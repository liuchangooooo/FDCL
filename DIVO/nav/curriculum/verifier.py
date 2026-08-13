"""单标量治锁 verifier(V0)——复用 Push-T 判据逻辑,探针走 NavEnvAdapter。

验收探针:G_t 与所有候选在**同一批固定起点上 paired 打分**(invalid_scene_policy='zero')。
判据:
  绝对硬门:valid_rate>=v_min AND duplicate_rate<=d_max(code_pass 由 sandbox 保证)。
  单标量目标:boundary_count(并列比 mean_b)。
  含 G_t argmax + 计数净改进:cand_bc >= Gt_bc + min_boundary_count_delta 才 replace,否则 hold。
  等难刷新:开关启用时,would-be hold 可换入通过硬门、代码不同且 bc 基本不退化的候选。
  对称饱和逃逸:r_easy>=r_easy_max 或 r_hard>=r_hard_max 时 G_t 不再阻塞,在过门候选里 argmax。
"""
from dataclasses import dataclass

import numpy as np

try:
    from DIVO.curriculum.learnable_frontier import select_generator_boundary
except ModuleNotFoundError:
    # 正式训练从 DIVO/DIVO 启动，此时共享包名为 curriculum。
    from curriculum.learnable_frontier import select_generator_boundary
from nav import nav_env as NE
from nav.skill_signal import library_p_on_scene, TAU
from nav.curriculum.generator_source import validate_pillars
from safety_gymnasium.utils.common_utils import ResamplingError


@dataclass
class SkillVerifierConfig:
    criterion: str = "boundary_count_argmax_escape"
    v_min: float = 0.8
    d_max: float = 0.25
    min_boundary_count_delta: int = 4    # 对齐 Push-T(未饱和净改进阈值)
    r_easy_max: float = 0.8
    r_hard_max: float = 0.8
    diversify_on_hold: bool = False
    diversify_bc_tolerance: int = 2
    diversify_easy_eps: float = 0.05
    tau_saturation: float = TAU
    target_delta: float = 0.15
    num_pillars: int = 2
    M: int = 40                              # 验收/方向探针场景数(对齐 Push-T)
    max_steps: int = 500
    # Edit-1(对齐 Push-T exp_llm_curriculum:已开启):把无效场景(固定起点摆不出合法布局)
    # 从 boundary 统计(bc/r_hard/r_easy/mean_b)剔除,只计入 valid_rate,避免"生成器无效"
    # 被当成"库解不开"(修 r_hard 虚高/误 RELAX/mean_b 稀释)。
    exclude_invalid_from_boundary: bool = True


def sample_fixed_starts(M, seed):
    rng = np.random.default_rng(seed)
    return [NE.sample_valid_start(rng) for _ in range(M)]


def evaluate_generator(adapter, act_fn, generator, K, fixed_starts, goal, cfg, probe_skills=None):
    """验收探针:对固定起点用 generator 产 pillar,探针 paired 打分('zero' 非法记0)。

    probe_skills=None -> W_probe(arm d);[0] -> 单策略 w_0(arm b)。
    """
    exclude_invalid = bool(getattr(cfg, "exclude_invalid_from_boundary", True))
    ps = []                # 用于 boundary 统计的 p(exclude 时仅含有效场景)
    total = 0              # 固定起点总数(算 valid_rate 分母)
    valid = 0
    seen = set()
    dup = 0
    for start in fixed_starts:
        total += 1
        invalid = False
        try:
            pillars = generator.generate(start, goal, cfg.num_pillars)
        except Exception:
            invalid = True
            pillars = None
        if not invalid:
            ok, _ = validate_pillars(pillars, start, goal, num=cfg.num_pillars)
            invalid = not ok
        res = None
        if not invalid:
            key = tuple(sorted((round(x, 2), round(y, 2)) for x, y in pillars))
            if key in seen:
                dup += 1
            seen.add(key)
            res = library_p_on_scene(adapter, act_fn, pillars, start, K, cfg.max_steps, probe_skills)
            invalid = res is None            # reset 非法
        if invalid:
            if not exclude_invalid:
                ps.append(0.0)               # legacy:invalid 记 realized=0(near-hard)
            continue                         # exclude:从 boundary 统计剔除,只影响 valid_rate
        valid += 1
        ps.append(res[1])
    ps = np.array(ps)
    N = max(len(ps), 1)
    bc = int(((ps > cfg.tau_saturation) & (ps < 1 - cfg.tau_saturation)).sum())
    return {
        "boundary_count": bc,
        "boundary_rate": bc / N,
        "mean_b": float((ps * (1 - ps)).mean()) if len(ps) else 0.0,
        "r_hard": float((ps <= cfg.tau_saturation).mean()) if len(ps) else 0.0,
        "r_easy": float((ps >= 1 - cfg.tau_saturation).mean()) if len(ps) else 0.0,
        "target_rate": float((np.abs(ps - 0.5) <= cfg.target_delta).mean()) if len(ps) else 0.0,
        "valid_rate": valid / max(total, 1),
        "duplicate_rate": dup / max(valid, 1),
        "n_scenes": len(ps),
        "per_scene_p": ps.tolist(),
    }


def decide_boundary_selection(cur_sig, cand_sigs, cfg, candidate_fresh_flags=None):
    """单标量 boundary_count argmax + 对称饱和逃逸(照 Push-T select_generator_boundary)。

    返回 dict(action, choice_index, reason)。action ∈ {'replace','hold','escape'}。
      * 候选须过绝对硬门(valid_rate/duplicate_rate);G_t 作 incumbent 不过门。
      * 目标 J = boundary_count(mean_b tiebreak)。
      * 未饱和:仅当某候选 bc >= Gt_bc + min_boundary_count_delta 才 replace,否则 hold。
      * 开启 diversify_on_hold 后,would-be hold 可换入代码不同且基本等难的候选。
      * 饱和(r_easy 或 r_hard 触顶):G_t 不再阻挡,但候选须朝 HARDEN/RELAX 方向移动
        且不越到对侧饱和尾(count 容差 1),再在合格候选里按 bc argmax(escape)。
    """
    shared = select_generator_boundary(
        cur_sig,
        cand_sigs,
        v_min=cfg.v_min,
        d_max=cfg.d_max,
        min_boundary_count_delta=cfg.min_boundary_count_delta,
        r_easy_max=cfg.r_easy_max,
        r_hard_max=cfg.r_hard_max,
        candidate_fresh_flags=candidate_fresh_flags,
        diversify_on_hold=cfg.diversify_on_hold,
        diversify_bc_tolerance=cfg.diversify_bc_tolerance,
        diversify_easy_eps=cfg.diversify_easy_eps,
    )
    reason_map = {
        "boundary_count_net_improve": "net_boundary_improvement",
        "no_candidate_passed_hard_gate": "no_valid_candidate",
        "saturated_but_no_candidate_passed_hard_gate": "saturated_no_valid_candidate",
        "no_net_boundary_improvement": "no_net_improvement",
    }
    action = shared["action"]
    reason = str(shared["reason"])
    if action == "replace" and reason.startswith("saturation_escape_"):
        action = "escape"
    return {
        **shared,
        "action": action,
        "choice_index": shared.get("chosen_index"),
        "reason": reason_map.get(reason, reason),
    }
