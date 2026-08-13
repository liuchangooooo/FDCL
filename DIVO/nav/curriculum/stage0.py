"""Stage 0:LLM 生成初始生成器 G_0 + 选择(此时无 W_probe,不用 p/b/boundary)。

选择判据:code_pass(sandbox)+ physical validity(采 N 布局的 valid_rate)+ 轻量难度散布
(不全无效/不全极易极难)。provider='mock' 时直接用 MockGenerator。
"""
import numpy as np

from nav import nav_env as NE
from nav.curriculum.generator_source import validate_pillars, MockGenerator


def _physical_validity(generator, goal, n=32, seed=0, num=2):
    """采 n 个 (start, generator(start)) 查 valid_rate 与难度散布(直线阻挡比例)。"""
    rng = np.random.default_rng(seed)
    valid = 0
    blocked = 0
    for _ in range(n):
        start = NE.sample_valid_start(rng, goal)
        try:
            pillars = generator.generate(start, goal, num)
        except Exception:
            continue
        ok, _ = validate_pillars(pillars, start, goal, num=num)
        if not ok:
            continue
        valid += 1
        # 直线阻挡:pillar 是否靠近 start->goal 连线(难度散布启发式)
        s = np.array(start); g = np.array(goal); seg = g - s
        for x, y in pillars:
            p = np.array([x, y]); t = np.clip((p - s) @ seg / (seg @ seg + 1e-9), 0, 1)
            if np.linalg.norm(s + t * seg - p) < NE.PILLAR_SIZE + NE.AGENT_RADIUS:
                blocked += 1
                break
    vr = valid / n
    br = blocked / max(valid, 1)
    return vr, br


def select_initial_generator(acgs=None, goal=NE.GOAL, provider="mock",
                             v_min=0.8, seed=0, num=2, candidates=3):
    """返回 (generator, info)。mock:挑 valid_rate 达标且难度散布适中的 MockGenerator。"""
    if provider == "mock":
        best = None
        best_info = None
        for c in range(candidates):
            gen = MockGenerator(difficulty=0.5, seed=seed * 100 + c, base_num=2)
            vr, br = _physical_validity(gen, goal, seed=seed + c, num=num)
            info = {"valid_rate": vr, "blocked_rate": br, "difficulty": gen.difficulty}
            # 轻量难度散布:既非全无效、也非全无阻挡/全阻挡
            if vr >= v_min and 0.1 <= br <= 0.95:
                if best is None or vr > best_info["valid_rate"]:
                    best, best_info = gen, info
        if best is None:
            best, best_info = MockGenerator(difficulty=0.5, seed=seed), {"valid_rate": None, "note": "fallback"}
        return best, best_info
    raise RuntimeError("provider='openai' 需 acgs 与 key;此处仅实现 mock 主路径供测试")
