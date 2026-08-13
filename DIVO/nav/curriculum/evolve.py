"""Stage 2 evolve 循环:方向探针 -> prompt/候选 -> 验收探针 paired -> 治锁 verifier 选择。

一个 evolve cycle:
  1) 方向探针(skill_signal.extract_skill_signals, resample)在当前 G_t 上产 signal + 方向。
  2) 产 R 候选生成器(mock:acgs.mock_candidates;openai:LLM 重写代码)。
  3) 验收探针(固定起点 paired,'zero'):评 G_t(cur_sig)+ 各候选(cand_sigs)。
  4) decide_boundary_selection -> replace/escape/hold,返回 G_{t+1}。

完整 Stage 2(训练与 evolve 交替)需接入 train_skill 的训练循环(长训 + LLM);
本模块提供 evolve cycle 与独立驱动(验证 verifier 选择/不卡死),训练交替留整合。
"""
import numpy as np

from nav import nav_env as NE
from nav.skill_signal import extract_skill_signals
from nav.curriculum.generator_source import validate_pillars
from nav.curriculum.prompt_builder import decide_direction, build_evolve
from nav.curriculum.verifier import (SkillVerifierConfig, sample_fixed_starts,
                                     evaluate_generator, decide_boundary_selection)
from safety_gymnasium.utils.common_utils import ResamplingError


def _generator_scene_sampler(generator, goal, num):
    """把 generator 包成 skill_signal 的 scene_sampler:先采起点、再 generator(start)。"""
    def sampler(rng):
        start = NE.sample_valid_start(rng, goal)
        try:
            pillars = generator.generate(start, goal, num)
        except Exception as exc:
            raise ResamplingError("generator failed during direction probe") from exc
        ok, reason = validate_pillars(pillars, start, goal, num=num)
        if not ok:
            raise ResamplingError(f"invalid generated layout:{reason}")
        return start, pillars
    return sampler


def evolve_cycle(adapter, act_fn, K, current_generator, acgs, cfg=None,
                 goal=NE.GOAL, R=3, cycle_seed=0, provider="mock", probe_skills=None,
                 context_M=None, context_n=4, include_behavior=True,
                 route_waypoints=5, batch_stats=None, verifier_enabled=True):
    """执行一个 evolve cycle,返回 dict(new_generator, decision, direction, cur_sig, cand_sigs)。

    probe_skills=None -> 库信号(arm d);[0] -> 单策略信号(arm b)。
    context_M:方向探针场景数(对齐 Push-T skill_signal.probe_M=40);None -> cfg.M。
    验收探针用 cfg.M(对齐 Push-T candidate_verifier.probe_M=30)。
    """
    cfg = cfg or SkillVerifierConfig()
    ctx_M = int(context_M) if context_M else cfg.M

    # 1) 方向探针(resample):在当前 G_t 上度量能力,定方向
    dir_signal = extract_skill_signals(
        adapter, act_fn, K, M=ctx_M, seed=1000 + cycle_seed,
        max_steps=cfg.max_steps,
        scene_sampler=_generator_scene_sampler(current_generator, goal, cfg.num_pillars),
        invalid_scene_policy="resample", probe_skills=probe_skills,
        context_n=context_n, include_behavior=include_behavior,
        route_waypoints=route_waypoints, tau=cfg.tau_saturation,
        target_delta=cfg.target_delta)
    direction = decide_direction(dir_signal, cfg.r_easy_max, cfg.r_hard_max, cfg.v_min)

    # 2) 候选
    if provider == "mock":
        candidates = acgs.mock_candidates(direction, R=R, seed=cycle_seed)
    else:
        # 真实 LLM:用 Navigate txt 模板组 prompt,采 R 个候选生成器(过 sandbox)
        current_code = getattr(current_generator, "code", None)
        system, user = build_evolve(
            current_code, dir_signal, direction=direction, batch_stats=batch_stats,
        )
        candidates = []
        candidate_errors = []
        for _ in range(R):
            ex, info = acgs.generate_code(system, user)
            if ex is not None:
                candidates.append(ex)
            else:
                candidate_errors.append(str(info))
        if not candidates:
            # 全部候选非法:hold(不替换)
            detail = ";".join(candidate_errors)
            reason = "no_valid_llm_candidate"
            if detail:
                reason += f"[{detail[:240]}]"
            return {"new_generator": current_generator,
                    "decision": {"action": "hold", "choice_index": None,
                                 "reason": reason},
                    "direction": direction, "dir_signal": dir_signal,
                    "cur_sig": None, "cand_sigs": []}

    if not verifier_enabled:
        selected = candidates[0]
        if provider == "mock" and hasattr(selected, "difficulty"):
            acgs.set_mock_difficulty(selected.difficulty)
        return {
            "new_generator": selected,
            "decision": {"action": "replace", "choice_index": 0,
                         "reason": "verifier_disabled"},
            "direction": direction, "dir_signal": dir_signal,
            "cur_sig": None, "cand_sigs": [],
        }

    # 3) 验收探针(固定起点 paired,'zero'):G_t + 候选
    fixed_starts = sample_fixed_starts(cfg.M, seed=5000 + cycle_seed)
    cur_sig = evaluate_generator(adapter, act_fn, current_generator, K, fixed_starts, goal, cfg,
                                 probe_skills=probe_skills)
    cand_sigs = [evaluate_generator(adapter, act_fn, g, K, fixed_starts, goal, cfg,
                                    probe_skills=probe_skills)
                 for g in candidates]

    # 4) 决策
    current_code = getattr(current_generator, "code", None)
    candidate_fresh_flags = [
        (
            current_code is None
            or getattr(candidate, "code", None) is None
            or candidate.code.strip() != current_code.strip()
        )
        for candidate in candidates
    ]
    decision = decide_boundary_selection(
        cur_sig,
        cand_sigs,
        cfg,
        candidate_fresh_flags=candidate_fresh_flags,
    )
    if decision["action"] in ("replace", "escape"):
        new_gen = candidates[decision["choice_index"]]
        # 同步 mock 难度状态(便于连续 evolve)
        if provider == "mock" and hasattr(new_gen, "difficulty"):
            acgs.set_mock_difficulty(new_gen.difficulty)
    else:
        new_gen = current_generator

    return {"new_generator": new_gen, "decision": decision, "direction": direction,
            "dir_signal": dir_signal, "cur_sig": cur_sig, "cand_sigs": cand_sigs}
