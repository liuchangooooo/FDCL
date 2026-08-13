"""信号复活验证(Stage 1 核心 go/no-go)——在生成器采样场景上对比单策略 vs 技能库。

论点:
  单确定性策略 => 每场景 p∈{0,1} => boundary_count≈0、mean_b≈0(信号退化)。
  技能库 => p_i=(1/K)Σ_k 1[w_k 到达] 出现中间值 => boundary_count>0、mean_b>0(信号复活)。

**只在生成器采样场景(训练 DOF / G_0)上度量,不用手工 TEMPLATES。**
同一 scene 内 K 技能共享同一 start(paired),p_i 的分数性只来自库多样性。

用法:
  MUJOCO_GL=egl python -m nav.probe_degeneracy --ckpt nav/runs/navv2_skill0/best.pt --K 4
"""
import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

from nav import nav_env as NE
from nav.nav_adapter import NavEnvAdapter
from nav.skill_td3 import SkillTD3
from nav.skill_signal import (default_scene_sampler, library_p_on_scene,
                              single_p_on_scene, TAU)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--M", type=int, default=64)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--max_steps", type=int, default=500)
    args = ap.parse_args()

    agent = SkillTD3(NE.OBS_DIM, 2, K=args.K, device="cuda")
    agent.load(args.ckpt, map_location=agent.device)
    act_fn = lambda o, k: agent.act(o, skill_id=k, noise=0.0)
    ad = NavEnvAdapter()

    rng = np.random.default_rng(args.seed)
    single_ps, lib_ps = [], []
    got = 0
    attempts = 0
    while got < args.M and attempts < args.M * 30:
        attempts += 1
        start, pillars = default_scene_sampler(rng)
        res = library_p_on_scene(ad, act_fn, pillars, start, args.K, args.max_steps)
        if res is None:
            continue
        _, p_lib, _ = res
        p_single = single_p_on_scene(ad, act_fn, pillars, start, args.max_steps, skill_id=0)
        if p_single is None:
            continue
        lib_ps.append(p_lib)
        single_ps.append(p_single)
        got += 1

    sp = np.array(single_ps); lp = np.array(lib_ps)

    def stats(ps):
        b = ps * (1 - ps)
        bc = int(((ps > TAU) & (ps < 1 - TAU)).sum())
        return bc, float(b.mean()), sorted(set(np.round(ps, 3).tolist()))

    s_bc, s_mb, s_vals = stats(sp)
    l_bc, l_mb, l_vals = stats(lp)

    print("=" * 84)
    print(f"信号复活验证(生成器采样场景 N={got}, K={args.K}, τ={TAU})")
    print("-" * 84)
    print(f"[单策略 w_0]  p∈{s_vals}  boundary_count={s_bc}  mean_b={s_mb:.4f}")
    print(f"[技能库 K={args.K}] p∈{l_vals}  boundary_count={l_bc}  mean_b={l_mb:.4f}")
    print("-" * 84)
    revived = (l_bc > s_bc) and (l_mb > s_mb + 1e-6)
    print(f"信号复活:{'YES' if revived else 'NO'} "
          f"(库 boundary_count {l_bc} > 单策略 {s_bc} 且 mean_b {l_mb:.4f} > {s_mb:.4f})")
    print("=" * 84)
    ad.close()
    return revived


if __name__ == "__main__":
    main()
