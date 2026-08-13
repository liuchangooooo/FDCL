"""Stage 1 go/no-go 汇总(Req 10.4/10.5/10.6, 11.4)。

硬门:
  (a) 信号复活:生成器采样场景上 库 boundary_count > 单策略、且 mean_b 更高。
  (b) K_eff ≥ 0.5K(技能不塌)。
  (c) per-skill 非平凡:Progress(w_k)≥τ_p 且 (1/K)Σ 1[Success(w_k)≥τ_s] ≥ r_s。
  (d) w_0 验证部署 ≥ 单策略 baseline − δ;B/M/U 不明显掉(D 观察项)。

K_eff:固定 probe 环境集,每环境枚举 K 技能 -> task-frame 轨迹嵌入 -> 阈值聚类(union-find)
      -> K_eff = mean_env exp(H(P_cluster))。占用簇仅日志。

用法:
  MUJOCO_GL=egl python -m nav.stage1_gonogo --ckpt nav/runs/navv2_skill0/best.pt --K 4 \
      --baseline_w0 0.9
"""
import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

from nav import nav_env as NE
from nav.nav_adapter import NavEnvAdapter
from nav.skill_td3 import SkillTD3
from nav.skill_signal import default_scene_sampler, library_p_on_scene, single_p_on_scene, TAU
from nav.diversity import embed_route
from nav.eval_dist import evaluate_validation
from nav import benchmarks as B
from safety_gymnasium.utils.common_utils import ResamplingError


def _rollout_path(adapter, act_fn, pillars, start, skill_id, max_steps=500):
    adapter.set_layout(pillars, start=start)
    try:
        obs = adapter.reset(seed=0, start=start)
    except ResamplingError:
        return None, None
    path = [adapter._env.task.data.body("agent").xpos[:2].copy()]
    succ = False
    g = np.array(adapter.goal); d0 = np.linalg.norm(g - path[0])
    for _ in range(max_steps):
        obs, r, term, trunc, info = adapter.step(act_fn(obs, skill_id))
        path.append(adapter._env.task.data.body("agent").xpos[:2].copy())
        if adapter.success(info):
            succ = True; break
        if term or trunc:
            break
    dT = np.linalg.norm(g - path[-1])
    progress = float(np.clip((d0 - dT) / (d0 + 1e-9), 0, 1))
    return {"path": path, "success": succ, "progress": progress, "start": start, "goal": adapter.goal}, None


def _union_find_clusters(E, theta):
    n = len(E)
    parent = list(range(n))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(E[i] - E[j]) < theta:
                parent[find(i)] = find(j)
    roots = {}
    for i in range(n):
        roots.setdefault(find(i), 0)
        roots[find(i)] += 1
    return np.array(list(roots.values()), float)


def compute_keff(adapter, act_fn, K, n_env=8, theta=0.15, seed=7, max_steps=500):
    rng = np.random.default_rng(seed)
    keffs = []
    got = 0; attempts = 0
    while got < n_env and attempts < n_env * 10:
        attempts += 1
        start, pillars = default_scene_sampler(rng)
        E = []
        ok = True
        for k in range(1, K + 1):
            r, _ = _rollout_path(adapter, act_fn, pillars, start, k, max_steps)
            if r is None:
                ok = False; break
            E.append(embed_route(r["path"], r["start"], r["goal"]))
        if not ok:
            continue
        sizes = _union_find_clusters(np.array(E), theta)
        p = sizes / sizes.sum()
        H = -(p * np.log(p + 1e-12)).sum()
        keffs.append(float(np.exp(H)))
        got += 1
    return float(np.mean(keffs)) if keffs else 0.0, got


def per_skill_nontrivial(adapter, act_fn, K, n_scene=12, seed=9, max_steps=500,
                         tau_p=0.0, tau_s=0.3, r_s=0.5):
    """对齐 Push-T nontrivial_skill_fraction:单技能非平凡 = (success_rate≥τ_s AND
    progress≥τ_p);库非平凡 = 非平凡技能占比 ≥ r_s。(Push-T τ_s=0.3, τ_p=0.0, r_s=0.5)"""
    rng = np.random.default_rng(seed)
    succ = {k: [] for k in range(1, K + 1)}
    prog = {k: [] for k in range(1, K + 1)}
    got = 0; attempts = 0
    while got < n_scene and attempts < n_scene * 10:
        attempts += 1
        start, pillars = default_scene_sampler(rng)
        rr = {}
        ok = True
        for k in range(1, K + 1):
            r, _ = _rollout_path(adapter, act_fn, pillars, start, k, max_steps)
            if r is None:
                ok = False; break
            rr[k] = r
        if not ok:
            continue
        for k in range(1, K + 1):
            succ[k].append(1.0 if rr[k]["success"] else 0.0)
            prog[k].append(rr[k]["progress"])
        got += 1
    per_skill = {k: {"success": float(np.mean(succ[k])) if succ[k] else 0.0,
                     "progress": float(np.mean(prog[k])) if prog[k] else 0.0}
                 for k in range(1, K + 1)}
    # 单技能非平凡 = (sr≥τ_s AND pr≥τ_p);再取占比 ≥ r_s(与 Push-T 一致)
    nt_flags = [1.0 if (per_skill[k]["success"] >= tau_s and per_skill[k]["progress"] >= tau_p)
                else 0.0 for k in range(1, K + 1)]
    nt_frac = float(np.mean(nt_flags)) if nt_flags else 0.0
    return per_skill, {"nontrivial_fraction": nt_frac,
                       "lib_nontrivial": bool(nt_frac >= r_s), "n_scene": got}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--M", type=int, default=48)
    ap.add_argument("--baseline_w0", type=float, default=0.9, help="单策略 w_0 验证 baseline")
    ap.add_argument("--delta", type=float, default=0.05, help="对齐 Push-T Stage1 delta")
    ap.add_argument("--keff_theta", type=float, default=0.15)
    ap.add_argument("--max_steps", type=int, default=500)
    args = ap.parse_args()

    agent = SkillTD3(NE.OBS_DIM, 2, K=args.K, device="cuda")
    agent.load(args.ckpt, map_location=agent.device)
    act_fn = lambda o, k: agent.act(o, skill_id=k, noise=0.0)
    ad = NavEnvAdapter()

    # (a) 信号复活
    rng = np.random.default_rng(123)
    sp, lp = [], []
    got = 0; att = 0
    while got < args.M and att < args.M * 30:
        att += 1
        start, pillars = default_scene_sampler(rng)
        res = library_p_on_scene(ad, act_fn, pillars, start, args.K, args.max_steps)
        if res is None:
            continue
        s0 = single_p_on_scene(ad, act_fn, pillars, start, args.max_steps, 0)
        if s0 is None:
            continue
        lp.append(res[1]); sp.append(s0); got += 1
    sp = np.array(sp); lp = np.array(lp)
    s_bc = int(((sp > TAU) & (sp < 1 - TAU)).sum()); s_mb = float((sp * (1 - sp)).mean())
    l_bc = int(((lp > TAU) & (lp < 1 - TAU)).sum()); l_mb = float((lp * (1 - lp)).mean())
    revived = (l_bc > s_bc) and (l_mb > s_mb + 1e-6)

    # (b) K_eff
    keff, keff_n = compute_keff(ad, act_fn, args.K, theta=args.keff_theta, max_steps=args.max_steps)

    # (c) per-skill 非平凡
    per_skill, nt = per_skill_nontrivial(ad, act_fn, args.K, max_steps=args.max_steps)

    # (d) w_0 验证 + B/M/U
    val = evaluate_validation(ad, lambda o: agent.act(o, skill_id=0, noise=0.0),
                              n_env=20, max_steps=args.max_steps)
    bmud = B.evaluate_bmud(ad, lambda o: agent.act(o, skill_id=0, noise=0.0),
                           n_env=20, max_steps=args.max_steps)

    # baseline_w0 历来是成功率阈值；best 选模改用 return 后这里仍应比较同口径 success_rate。
    w0_ok = val["success_rate"] >= args.baseline_w0 - args.delta
    keff_ok = keff >= 0.5 * args.K

    print("=" * 88)
    print(f"Stage 1 go/no-go  ckpt={args.ckpt}  K={args.K}")
    print("-" * 88)
    print(f"(a) 信号复活: 库 bc={l_bc} mean_b={l_mb:.4f} vs 单策略 bc={s_bc} mean_b={s_mb:.4f} "
          f"-> {'PASS' if revived else 'FAIL'}  (N={got})")
    print(f"(b) K_eff={keff:.2f} (>= 0.5K={0.5*args.K}) -> {'PASS' if keff_ok else 'FAIL'}  (n_env={keff_n})")
    print(f"(c) per-skill: {[ (k, round(v['success'],2), round(v['progress'],2)) for k,v in per_skill.items() ]}")
    print(f"    lib_nontrivial(非平凡占比={nt['nontrivial_fraction']:.2f}≥r_s=0.5)={nt['lib_nontrivial']}")
    print(f"(d) w_0 success={val['success_rate']:.3f} return={val['test_mean_score']:.3f} "
          f"(success baseline {args.baseline_w0}-{args.delta}) "
          f"-> {'PASS' if w0_ok else 'FAIL'};  B/M/U/D={ {k:round(v,2) for k,v in bmud.items()} }")
    print("-" * 88)
    go = revived and keff_ok and nt["lib_nontrivial"] and w0_ok
    print(f"GO/NO-GO: {'GO' if go else 'NO-GO'}(信号复活+K_eff+库非平凡+w_0 不掉;D 为观察项)")
    print("=" * 88)
    ad.close()


if __name__ == "__main__":
    main()
