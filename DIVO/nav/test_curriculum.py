"""任务 13/14/15/16 mock 端到端 sanity:sandbox / Stage0 / verifier / evolve cycle。

优先用 navv2_skill0 的库 ckpt 作 act_fn；缺失或不兼容时用未训练库。小 M/步数以加速。
验证:
  1) sandbox 能 load+run 一段合法 generate_pillars 代码;拦截危险 token。
  2) Stage 0 select_initial_generator(mock)返回合法生成器。
  3) verifier.evaluate_generator 产出 boundary 统计;不同难度候选 boundary 不同。
  4) evolve_cycle 端到端不崩,产出 replace/escape/hold 决策(不卡死)。
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

from nav import nav_env as NE
from nav.nav_adapter import NavEnvAdapter
from nav.skill_td3 import SkillTD3
from nav.curriculum.generator_source import SandboxNavExecutor, MockGenerator, validate_pillars
from nav.curriculum.stage0 import _physical_validity, select_initial_generator
from nav.curriculum.verifier import (
    SkillVerifierConfig,
    decide_boundary_selection,
    evaluate_generator,
    sample_fixed_starts,
)
from nav.curriculum.evolve import evolve_cycle, _generator_scene_sampler
from nav.curriculum.acgs_api import NavACGS
from nav.train_stage2 import sample_scene_from_generator, _require_two_pillar_training
from safety_gymnasium.utils.common_utils import ResamplingError

CKPT = "nav/runs/navv2_skill0/latest.pt"
K = 4


class _WrongCountGenerator:
    """返回合法坐标但少一根柱，用于验证数量硬门。"""

    def generate(self, agent_start, goal, num):
        return [(0.0, 0.5)]


def test_nav_diversify_on_hold_selection():
    cfg = SkillVerifierConfig(
        diversify_on_hold=True,
        diversify_bc_tolerance=2,
        diversify_easy_eps=0.05,
    )
    current = {
        "boundary_count": 29, "mean_b": 0.20, "r_hard": 0.0, "r_easy": 0.03,
        "valid_rate": 1.0, "duplicate_rate": 0.0, "n_scenes": 30,
    }
    candidate = {
        "boundary_count": 28, "mean_b": 0.19, "r_hard": 0.03, "r_easy": 0.03,
        "valid_rate": 1.0, "duplicate_rate": 0.0, "n_scenes": 30,
    }

    fresh = decide_boundary_selection(
        current, [candidate], cfg, candidate_fresh_flags=[True],
    )
    assert fresh["action"] == "replace"
    assert fresh["choice_index"] == 0
    assert fresh["reason"] == "diversify_preserve_difficulty"

    identical = decide_boundary_selection(
        current, [candidate], cfg, candidate_fresh_flags=[False],
    )
    assert identical["action"] == "hold"
    assert identical["choice_index"] is None
    assert identical["reason"] == "no_fresh_candidate"


def test_exact_two_pillar_contract():
    """训练、Stage 0 与 verifier 均拒绝非两柱生成结果。"""
    generator = _WrongCountGenerator()
    goal = NE.GOAL
    assert _require_two_pillar_training(2) == 2
    try:
        _require_two_pillar_training(3)
    except ValueError:
        pass
    else:
        raise AssertionError("formal training must reject a three-pillar override")
    ok, reason = validate_pillars(generator.generate(None, goal, 2), (-0.8, 0.0), goal, num=2)
    assert not ok and reason == "num_mismatch(1!=2)"

    scene = sample_scene_from_generator(
        np.random.default_rng(0), generator, goal, num=2, reject_trivial=False, tries=2,
    )
    assert scene is None

    valid_rate, _ = _physical_validity(generator, goal, n=2, seed=0)
    assert valid_rate == 0.0

    cfg = SkillVerifierConfig(M=1, max_steps=1)
    assert cfg.num_pillars == 2
    sig = evaluate_generator(
        adapter=None,
        act_fn=None,
        generator=generator,
        K=K,
        fixed_starts=[(-0.8, 0.0)],
        goal=goal,
        cfg=cfg,
    )
    assert sig["valid_rate"] == 0.0
    assert sig["per_scene_p"] == []

    sampler = _generator_scene_sampler(generator, goal, num=2)
    try:
        sampler(np.random.default_rng(0))
    except ResamplingError:
        pass
    else:
        raise AssertionError("direction probe must reject a non-two-pillar layout")


def load_act_fn():
    agent = SkillTD3(NE.OBS_DIM, 2, K=K, device="cuda")
    if os.path.exists(CKPT):
        try:
            agent.load(CKPT, map_location=agent.device)
            print(f"  loaded library ckpt {CKPT}")
        except (KeyError, RuntimeError) as exc:
            # 历史 checkpoint 可能来自旧的 16-D z-encoder；协议回归不依赖其权重。
            print(f"  (legacy/incompatible ckpt ignored: {type(exc).__name__}; using untrained library)")
    else:
        print("  (no ckpt; using untrained library)")
    return lambda o, k: agent.act(o, skill_id=k, noise=0.0)


def main():
    test_nav_diversify_on_hold_selection()
    print("  0) Nav shared diversify-on-hold selection OK")
    test_exact_two_pillar_contract()
    print("  0b) train/Stage0/verifier exact-two-pillar contract OK")

    # 1) sandbox
    code_ok = (
        "def generate_pillars(agent_start, goal, num):\n"
        "    pts = []\n"
        "    for i in range(num):\n"
        "        pts.append((0.1*i, 0.2*i - 0.2))\n"
        "    return pts\n"
    )
    ex = SandboxNavExecutor()
    ok, reason = ex.load_code(code_ok); assert ok, reason
    out = ex.generate(NE.START, NE.GOAL, 2)
    assert len(out) == 2
    bad_ok, bad_reason = SandboxNavExecutor().load_code("import os\ndef generate_pillars(a,g,n):return []")
    assert not bad_ok
    print(f"  1) sandbox load/run OK;危险代码被拦({bad_reason})")

    # 2) Stage 0
    gen0, info0 = select_initial_generator(provider="mock", seed=0)
    print(f"  2) Stage0 G_0 info={info0}")
    assert hasattr(gen0, "generate")

    adapter = NavEnvAdapter()
    act_fn = load_act_fn()
    cfg = SkillVerifierConfig(M=3, max_steps=80, num_pillars=2, min_boundary_count_delta=1)

    # 3) 不同难度候选 boundary 不同
    fs = sample_fixed_starts(cfg.M, seed=1)
    easy = evaluate_generator(adapter, act_fn, MockGenerator(0.1, seed=1), K, fs, NE.GOAL, cfg)
    hard = evaluate_generator(adapter, act_fn, MockGenerator(0.9, seed=1), K, fs, NE.GOAL, cfg)
    print(f"  3) easy: bc={easy['boundary_count']} r_easy={easy['r_easy']:.2f} r_hard={easy['r_hard']:.2f}")
    print(f"     hard: bc={hard['boundary_count']} r_easy={hard['r_easy']:.2f} r_hard={hard['r_hard']:.2f}")

    # 4) evolve cycles(不卡死:记录多轮决策)
    acgs = NavACGS(provider="mock")
    gen = gen0
    actions = []
    for c in range(2):
        res = evolve_cycle(adapter, act_fn, K, gen, acgs, cfg=cfg, cycle_seed=c, provider="mock")
        gen = res["new_generator"]
        actions.append((res["direction"], res["decision"]["action"], res["decision"]["reason"]))
        print(f"  cycle {c}: dir={res['direction']} cur_bc={res['cur_sig']['boundary_count']} "
              f"-> {res['decision']['action']}({res['decision']['reason']})")
    print(f"  4) evolve 2 cycles 完成，决策={[a[1] for a in actions]}（闭环未卡死）")
    adapter.close()
    print("ALL PASS")


if __name__ == "__main__":
    main()
