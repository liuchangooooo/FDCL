"""Stage 2 总装线:训练技能库 ↔ 周期 evolve G_t 交替(完整闭环)。

闭环(对齐 Push-T td3_curriculum_workspace 主循环):
  Stage0 LLM 生成 G_0
  -> 循环:从 G_t 采场景(先采起点、再 generate_pillars)训练技能库 N 步
          -> 每 evolve_interval_steps:方向探针 -> LLM 重写候选 -> 治锁 verifier 选 -> 更新 G_t
          -> 每 eval_every:验证 between 选 best
  -> 输出 best ckpt(供 Stage3 B/M/U/D)

provider:
  mock  (默认,无需 key):MockGenerator 参数化替身,验证闭环逻辑。
  openai(需 OPENAI_API_KEY):真 LLM 生成/重写 generate_pillars。

配置:hydra 一站式 yaml `config/nav/nav_curriculum.yaml`(对照 Push-T
      config/pusht/train_td3_curriculum_mujoco_obstacle.yaml)。覆盖用 hydra key=value 语法。

用法(safenav,cwd=/home/hnu-w/DIVO/DIVO):
  MUJOCO_GL=egl python -m nav.train_stage2                                  # 默认 = arm d 本方法
  MUJOCO_GL=egl python -m nav.train_stage2 seed=1 tag=navv4_d_libcur_s1
  MUJOCO_GL=egl python -m nav.train_stage2 training.train_library=false curriculum.evolve.probe_feedback=single tag=navv4_S0_s0
  OPENAI_API_KEY=sk-... MUJOCO_GL=egl python -m nav.train_stage2 provider=openai tag=navv4_stage2_llm
"""
import os
import time
from types import SimpleNamespace

os.environ.setdefault("MUJOCO_GL", "egl")

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from nav import nav_env as NE
from nav.nav_adapter import NavEnvAdapter
from nav.protocol import write_training_manifest
from nav.skill_td3 import SkillTD3
from nav.td3 import SkillReplayBuffer
from nav.eval_dist import evaluate_validation
from nav.curriculum.generator_source import validate_pillars, MockGenerator, SandboxNavExecutor
from nav.curriculum.stage0 import select_initial_generator
from nav.curriculum.acgs_api import NavACGS
from nav.curriculum import prompt_builder as PB
from nav.curriculum.verifier import SkillVerifierConfig
from nav.curriculum.evolve import evolve_cycle
from safety_gymnasium.utils.common_utils import ResamplingError

OUTROOT = os.path.join(os.path.dirname(__file__), "runs")
STAGE0_RETRIES = 5      # Stage0 LLM G_0 生成的最大重试次数(失败才回退 mock)


def _validate_env_contract(cfg):
    """Fail fast when the readable YAML task contract drifts from nav_env.py."""
    checks = {
        "obs_dim": (int(cfg.obs_dim), NE.OBS_DIM),
        "action_dim": (int(cfg.action_dim), 2),
        "env.obs_dim": (int(cfg.env.obs_dim), NE.OBS_DIM),
        "env.action_dim": (int(cfg.env.action_dim), 2),
        "env.start_sample.x_range": (list(cfg.env.start_sample.x_range), list(NE.START_X_RANGE)),
        "env.start_sample.y_range": (list(cfg.env.start_sample.y_range), list(NE.START_Y_RANGE)),
        "env.goal": (list(cfg.env.goal), list(NE.GOAL)),
        "env.goal_size": (float(cfg.env.goal_size), NE.GOAL_SIZE),
        "env.extents": (list(cfg.env.extents), list(NE.EXTENTS)),
        "env.pillar.size": (float(cfg.env.pillar.size), NE.PILLAR_SIZE),
        "env.pillar.keepout": (float(cfg.env.pillar.keepout), NE.PILLAR_KEEPOUT),
        "env.pillar.height": (float(cfg.env.pillar.height), NE.PILLAR_HEIGHT),
        "env.agent_radius": (float(cfg.env.agent_radius), NE.AGENT_RADIUS),
        "env.collision_is_failure": (
            bool(cfg.env.collision_is_failure), True
        ),
        "env.collision_penalty": (
            float(cfg.env.collision_penalty), NE.COLLISION_PENALTY
        ),
        "env.oob_bound": (float(cfg.env.oob_bound), NE.OOB_BOUND),
        "env.oob_is_failure": (bool(cfg.env.oob_is_failure), True),
        "env.oob_penalty": (float(cfg.env.oob_penalty), NE.OOB_PENALTY),
        "env.obstacle_region": (float(cfg.env.obstacle_region), NE.OBSTACLE_REGION),
        "env.val_obstacle_region": (float(cfg.env.val_obstacle_region), NE.VAL_OBSTACLE_REGION),
        "env.start_pillar_clearance": (
            float(cfg.env.start_pillar_clearance), NE.START_PILLAR_CLEARANCE
        ),
        "env.between_offset_range": (
            list(cfg.env.between_offset_range), list(NE.BETWEEN_OFFSET_RANGE)
        ),
        "env.between_lateral_range": (
            list(cfg.env.between_lateral_range), list(NE.BETWEEN_LATERAL_RANGE)
        ),
        "env.min_start_goal_dist": (float(cfg.env.min_start_goal_dist), NE.MIN_START_GOAL_DIST),
    }
    mismatches = []
    for name, (configured, runtime) in checks.items():
        if not np.allclose(np.asarray(configured), np.asarray(runtime)):
            mismatches.append(f"{name}: yaml={configured}, runtime={runtime}")
    if mismatches:
        raise ValueError(
            "Nav YAML task contract does not match nav_env.py: " + "; ".join(mismatches)
        )


def _require_two_pillar_training(num):
    """正式 Nav 训练协议固定为两根静态 Pillar，拒绝把 M 的三柱泄漏进训练。"""
    num = int(num)
    if num != NE.TRAIN_NUM_PILLARS:
        raise ValueError(
            f"Nav training protocol requires exactly {NE.TRAIN_NUM_PILLARS} pillars; got {num}"
        )
    return num


def sample_scene_from_generator(rng, generator, goal, num, reject_trivial=True, tries=40):
    """先采起点、再 generator.generate(start,goal,num)(对齐 Push-T 'start then generator')。

    返回 (start, pillars) 或 None。起点非平凡 + 布局过物理校验。
    """
    for _ in range(tries):
        start = NE.sample_valid_start(rng, goal)
        if reject_trivial and not NE.start_goal_ok(start, goal):
            continue
        try:
            pillars = NE.dedupe(generator.generate(start, goal, num))
        except Exception:
            continue
        if not pillars:
            continue
        ok, _ = validate_pillars(pillars, start, goal, num=num)
        if not ok:
            continue
        return start, pillars
    return None


def make_initial_generator(provider, acgs, goal, num, seed, log, init_mode="llm",
                           init_path=None, timeout_sec=5):
    """Stage 0:生成/选择 G_0。

    init_path 给定 -> 从文件载入固定 G_0 代码(四格公平对比:各臂共享同一 G_0);
    否则 mock -> MockGenerator(同 seed 确定,可跨臂共享);openai -> LLM 生成(失败回退 mock)。
    """
    if init_mode == "file":
        from nav.curriculum.generator_source import SandboxNavExecutor
        if not init_path:
            raise ValueError("curriculum.generator.init_mode=file requires init_path")
        with open(init_path, "r", encoding="utf-8") as f:
            code = f.read()
        ex = SandboxNavExecutor(timeout_sec=timeout_sec)
        ok, reason = ex.load_code(code)
        if ok:
            log(f"[Stage0] 载入固定 G_0: {init_path}")
            return ex
        log(f"[Stage0] 固定 G_0 载入失败({reason}),回退生成")
    if provider == "openai":
        rng = np.random.default_rng(seed)
        # 重试若干次:单次 LLM 生成可能产出过约束/偶发非法的 generator,不应一次失败就回退 mock
        for attempt in range(STAGE0_RETRIES):
            start = NE.sample_valid_start(rng, goal)
            system, user = PB.build_initial(np.array(start), np.array(goal))
            ex, info = acgs.generate_code(system, user)
            if ex is not None:
                sc = sample_scene_from_generator(rng, ex, goal, num)   # 轻量校验:能产出合法布局
                if sc is not None:
                    log(f"[Stage0] LLM G_0 OK (attempt {attempt + 1})")
                    return ex
            reason = info if ex is None else "no_valid_layout"
            log(f"[Stage0] LLM G_0 attempt {attempt + 1}/{STAGE0_RETRIES} 失败({reason})")
        log(f"[Stage0] LLM G_0 连续 {STAGE0_RETRIES} 次失败,回退 mock")
    gen, info = select_initial_generator(provider="mock", goal=goal, seed=seed, num=num)
    log(f"[Stage0] mock G_0 info={info}")
    return gen


def _cfg_to_args(cfg):
    """把 hydra OmegaConf 摊平成 run() 用的扁平 namespace(配置单一来源 = yaml)。

    结构参照 Push-T exp_llm_curriculum:curriculum.evolve.{skill_library,skill_signal,
    candidate_verifier}。max_evolve_count=0 或 evolve.enabled=false 均关进化(arm a/c)。
    """
    _validate_env_contract(cfg)
    ev = cfg.curriculum.evolve
    sl = ev.skill_library
    cv = ev.candidate_verifier
    provider = str(cfg.provider).lower()
    if provider not in ("mock", "openai"):
        raise ValueError(f"provider must be mock or openai; got {provider}")
    if provider == "openai" and str(cfg.curriculum.api_type).lower() != "openai":
        raise ValueError("provider=openai requires curriculum.api_type=openai")
    if str(cfg.curriculum.task_name) != "Navigate":
        raise ValueError("curriculum.task_name must be Navigate")
    if str(ev.feedback_mode) != "skill_library":
        raise ValueError("Nav Stage-2 currently supports feedback_mode=skill_library only")
    if str(ev.trigger_mode) != "fixed":
        raise ValueError("Nav Stage-2 currently supports trigger_mode=fixed only")
    init_mode = str(cfg.curriculum.generator.init_mode).lower()
    if init_mode not in ("llm", "file"):
        raise ValueError(f"generator.init_mode must be llm or file; got {init_mode}")
    if int(cfg.skill.diversity_every) != 0 or float(cfg.skill.beta_div) != 0.0:
        raise ValueError("Nav Route-B V0 requires diversity_every=0 and beta_div=0.0")
    total_steps = int(cfg.training.total_steps)
    eval_every = int(cfg.training.eval_every)
    if total_steps <= 0 or eval_every <= 0:
        raise ValueError("training.total_steps and training.eval_every must be positive")
    if total_steps % eval_every != 0:
        raise ValueError(
            "training.total_steps must be divisible by training.eval_every so the "
            "exact final-step model is validated before checkpoint selection"
        )
    max_evolve = int(ev.max_evolve_count) if bool(ev.enabled) else 0
    num_pillars = _require_two_pillar_training(cfg.num_pillars)
    return SimpleNamespace(
        seed=int(cfg.seed), tag=str(cfg.tag), provider=provider, device=str(cfg.device),
        # skill
        K=int(cfg.skill.K), d_z=int(cfg.skill.d_z), p_w0=float(cfg.skill.w0_rollout_ratio),
        beta_div=float(cfg.skill.beta_div), lambda_z=float(cfg.skill.lambda_z),
        # rl / 优化
        gamma=float(cfg.rl.gamma), tau=float(cfg.rl.tau),
        actor_lr=float(cfg.rl.actor_lr), critic_lr=float(cfg.rl.critic_lr),
        policy_noise=float(cfg.rl.policy_noise), noise_clip=float(cfg.rl.noise_clip),
        policy_delay=int(cfg.rl.policy_delay), hidden=[int(h) for h in cfg.rl.hidden],
        batch=int(cfg.rl.batch), expl_noise=float(cfg.rl.expl_noise),
        start_steps=int(cfg.rl.start_steps), replay_size=int(cfg.rl.replay_size),
        # skill codebook 表示
        codebook_type=str(cfg.skill.codebook_type),
        d_w=(None if cfg.skill.d_w is None else int(cfg.skill.d_w)),
        # training
        total_steps=total_steps, max_ep_steps=int(cfg.training.max_ep_steps),
        eval_every=eval_every, eval_n=int(cfg.training.eval_n),
        reject_trivial=1 if bool(cfg.training.reject_trivial) else 0,
        # backbone 训练:整库 or 仅 w_0(与课程信号解耦)
        train_library=bool(cfg.training.train_library),
        # 课程反馈信号:library(W_probe boundary)or single(仅 w_0 退化二值)
        evolve_feedback=str(ev.probe_feedback),
        # curriculum / LLM
        num_pillars=num_pillars, provider_model=str(cfg.curriculum.model),
        provider_temperature=float(cfg.curriculum.temperature),
        provider_max_tokens=int(cfg.curriculum.max_tokens),
        provider_timeout_sec=int(cfg.curriculum.timeout_sec),
        provider_api_key=(None if cfg.curriculum.get("api_key") is None
                          else str(cfg.curriculum.api_key)),
        base_url=(None if cfg.curriculum.base_url is None else str(cfg.curriculum.base_url)),
        init_mode=init_mode,
        init_generator_path=(None if cfg.curriculum.generator.init_path is None
                             else str(cfg.curriculum.generator.init_path)),
        save_artifacts=bool(cfg.curriculum.generator.save_artifacts),
        # evolve schedule
        evolve_after=int(ev.evolve_after), evolve_interval_steps=int(ev.evolve_interval_steps),
        max_evolve_count=max_evolve, R=int(sl.candidates_per_cycle),
        context_probe_M=int(ev.skill_signal.probe_M),
        context_n=int(ev.skill_signal.context_n),
        include_behavior=bool(ev.skill_signal.include_behavior),
        route_waypoints=int(ev.skill_signal.route_waypoints),
        verifier_enabled=bool(cv.enabled) and bool(cv.use_verifier),
        probe_M=int(cv.probe_M),
        # verifier 阈值
        valid_rate_min=float(sl.valid_rate_min), duplicate_rate_max=float(sl.duplicate_rate_max),
        min_boundary_count_delta=int(sl.min_boundary_count_delta),
        r_easy_max=float(sl.r_easy_max), r_hard_max=float(sl.r_hard_max),
        diversify_on_hold=bool(sl.diversify_on_hold),
        diversify_bc_tolerance=int(sl.diversify_bc_tolerance),
        diversify_easy_eps=float(sl.diversify_easy_eps),
        tau_saturation=float(sl.tau_saturation), target_delta=float(sl.target_delta),
        exclude_invalid_from_boundary=bool(sl.exclude_invalid_from_boundary),
    )


def run(args):
    _require_two_pillar_training(args.num_pillars)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    outdir = os.path.join(OUTROOT, args.tag)
    os.makedirs(outdir, exist_ok=True)
    write_training_manifest(
        outdir, trainer="nav.train_stage2", seed=args.seed, tag=args.tag
    )
    logf = open(os.path.join(outdir, "train.log"), "a")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n"); logf.flush()

    train_ad = NavEnvAdapter(seed=args.seed)
    eval_ad = NavEnvAdapter(seed=args.seed)
    probe_ad = NavEnvAdapter(seed=args.seed)
    agent = SkillTD3(NE.OBS_DIM, 2, K=args.K, device=args.device, lambda_z=args.lambda_z,
                     gamma=args.gamma, tau=args.tau, actor_lr=args.actor_lr, critic_lr=args.critic_lr,
                     policy_noise=args.policy_noise, noise_clip=args.noise_clip,
                     policy_delay=args.policy_delay, d_z=args.d_z, hidden=tuple(args.hidden),
                     codebook_type=args.codebook_type, d_w=args.d_w)
    buf = SkillReplayBuffer(NE.OBS_DIM, 2, size=args.replay_size, device=agent.device)
    acgs = NavACGS(
        provider=args.provider, model=args.provider_model,
        temperature=args.provider_temperature, max_tokens=args.provider_max_tokens,
        timeout_sec=args.provider_timeout_sec, base_url=args.base_url,
        api_key=args.provider_api_key,
    )
    cfg = SkillVerifierConfig(M=args.probe_M, max_steps=args.max_ep_steps,
                              num_pillars=args.num_pillars,
                              min_boundary_count_delta=args.min_boundary_count_delta,
                              v_min=args.valid_rate_min, d_max=args.duplicate_rate_max,
                              r_easy_max=args.r_easy_max, r_hard_max=args.r_hard_max,
                              diversify_on_hold=args.diversify_on_hold,
                              diversify_bc_tolerance=args.diversify_bc_tolerance,
                              diversify_easy_eps=args.diversify_easy_eps,
                              tau_saturation=args.tau_saturation,
                              target_delta=args.target_delta,
                              exclude_invalid_from_boundary=args.exclude_invalid_from_boundary)

    goal = NE.GOAL
    Gt = make_initial_generator(
        args.provider, acgs, goal, args.num_pillars, args.seed, log,
        init_mode=args.init_mode, init_path=args.init_generator_path,
        timeout_sec=args.provider_timeout_sec,
    )
    reject_trivial = bool(args.reject_trivial)
    # 存盘 G_0 代码(若为 LLM/文件生成器),供其它臂用 --init_generator_path 共享同一 G_0
    g0_code = getattr(Gt, "code", None)
    if args.save_artifacts and g0_code:
        with open(os.path.join(outdir, "g0_code.py"), "w", encoding="utf-8") as f:
            f.write(g0_code)

    # ---- 两个解耦开关(拆开原 single_policy)----
    # backbone:train_library=True 训整库(P(w_0)=args.p_w0);False 只训 w_0(p_w0=1.0)
    train_library = bool(args.train_library)
    p_w0 = args.p_w0 if train_library else 1.0
    # 课程反馈信号:library -> 用 W_probe(w_1..K)算 boundary;single -> 只用 w_0(退化二值)
    probe_skills = None if args.evolve_feedback == "library" else [0]
    # 防呆:只训 w_0 时无法用库信号(probe 技能没被训练)
    if (not train_library) and args.evolve_feedback == "library":
        raise ValueError("evolve_feedback='library' 需 train_library=True(否则 probe 技能未训练)")

    log(f"start tag={args.tag} provider={args.provider} K={args.K} "
        f"num_pillars={args.num_pillars} obstacle_motion=static "
        f"codebook={args.codebook_type}(d_w={agent.d_w}) train_library={train_library} "
        f"evolve_feedback={args.evolve_feedback} p_w0={p_w0} "
        f"evolve_every={args.evolve_interval_steps} total_steps={args.total_steps}")

    act_lib = lambda o, k: agent.act(o, skill_id=k, noise=0.0)
    best = -1e9
    step = 0
    ep = 0
    evolve_count = 0
    next_evolve = args.evolve_after
    t0 = time.time()
    batch_stats = {"success": 0, "collision": 0, "timeout": 0, "fall": 0}

    def new_episode():
        for _ in range(30):
            sc = sample_scene_from_generator(rng, Gt, goal, args.num_pillars, reject_trivial)
            if sc is None:
                continue
            start, pillars = sc
            train_ad.set_layout(pillars, start=start)
            try:
                return train_ad.reset(seed=int(rng.integers(0, 2**31 - 1)), start=start)
            except ResamplingError:
                continue
        return None

    while step < args.total_steps:
        obs = new_episode()
        if obs is None:
            continue
        skill = 0 if rng.random() < p_w0 else int(rng.integers(1, args.K + 1))
        ep += 1

        for ep_step in range(args.max_ep_steps):
            if step < args.start_steps:
                a = np.random.uniform(-1, 1, size=2)
            else:
                a = agent.act(obs, skill_id=skill, noise=args.expl_noise)
            next_obs, r_task, term, trunc, info = train_ad.step(a)
            success = train_ad.success(info)
            done = term or success
            buf.add(obs, a, next_obs, float(done), skill_id=skill,
                    reward_task=r_task, reward_div=0.0, beta_div=args.beta_div)
            obs = next_obs
            step += 1

            reached_step_limit = ep_step + 1 >= args.max_ep_steps
            episode_finished = bool(term or trunc or success or reached_step_limit)
            if episode_finished:
                if success:
                    batch_stats["success"] += 1
                elif info.get("collision", False):
                    batch_stats["collision"] += 1
                elif info.get("oob", False):
                    # Shared schema uses Push-T's ``fall`` slot internally; the
                    # Navigate renderer exposes this as ``out_of_bounds``.
                    batch_stats["fall"] += 1
                else:
                    batch_stats["timeout"] += 1

            if step >= args.start_steps and buf.n >= args.batch:
                agent.train_step(buf.sample(args.batch))

            # ---- 周期 evolve:更新 G_t ----
            if step >= next_evolve and evolve_count < args.max_evolve_count:
                res = evolve_cycle(probe_ad, act_lib, args.K, Gt, acgs, cfg=cfg,
                                   goal=goal, R=args.R, cycle_seed=evolve_count,
                                   provider=args.provider, probe_skills=probe_skills,
                                   context_M=args.context_probe_M,
                                   context_n=args.context_n,
                                   include_behavior=args.include_behavior,
                                   route_waypoints=args.route_waypoints,
                                   batch_stats=dict(batch_stats),
                                   verifier_enabled=args.verifier_enabled)
                Gt = res["new_generator"]
                cur_bc = res["cur_sig"]["boundary_count"] if res.get("cur_sig") else -1
                log(f"[evolve #{evolve_count}] step={step} dir={res['direction']} "
                    f"cur_bc={cur_bc} -> {res['decision']['action']}({res['decision']['reason']})")
                evolve_count += 1
                next_evolve += args.evolve_interval_steps
                evolved_code = getattr(Gt, "code", None)
                if args.save_artifacts and evolved_code:
                    artifact = os.path.join(outdir, f"g{evolve_count}_code.py")
                    with open(artifact, "w", encoding="utf-8") as f:
                        f.write(evolved_code)
                batch_stats = {"success": 0, "collision": 0, "timeout": 0, "fall": 0}

            # ---- 周期验证选 best ----
            if step % args.eval_every == 0:
                val = evaluate_validation(eval_ad, lambda o: agent.act(o, skill_id=0, noise=0.0),
                                          n_env=args.eval_n, max_steps=args.max_ep_steps)
                sps = step / (time.time() - t0)
                log(f"step={step} ep={ep} val_return={val['test_mean_score']:.3f} "
                    f"(succ={val['success_rate']:.3f} "
                    f"collision={val['collision_rate']:.3f} "
                    f"oob={val['oob_rate']:.3f} timeout={val['timeout_rate']:.3f} "
                    f"finald={val['mean_final_dist']:.3f}) "
                    f"evolves={evolve_count} sps={sps:.0f}")
                agent.save(os.path.join(outdir, "latest.pt"))
                if val["test_mean_score"] > best:
                    best = val["test_mean_score"]
                    agent.save(os.path.join(outdir, "best.pt"))

            # total_steps 是严格交互预算；不能为跑完当前 episode 继续更新并覆盖
            # 已验证 checkpoint。正式结果始终使用 best.pt。
            if step >= args.total_steps or term or trunc or success:
                break

    log(f"DONE steps={step} best_val={best:.3f} evolves={evolve_count} selected=best.pt")
    logf.close()


@hydra.main(version_base=None, config_path="../../config/nav", config_name="nav_curriculum")
def main(cfg):
    print(OmegaConf.to_yaml(cfg), flush=True)
    run(_cfg_to_args(cfg))


if __name__ == "__main__":
    main()
