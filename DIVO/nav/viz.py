"""度量与可视化(Req 11.5):同布局 K 技能 agent 路径叠图 + within/across 轨迹多样度量。

- skill_paths_overlay:同一 layout 下 w_0..w_K 的 agent 路径叠加渲染成图(看解法分化)。
- traj_diversity:within(同布局跨技能路径距离)/across(跨布局同技能)轨迹多样度量。
用法:
  MUJOCO_GL=egl python -m nav.viz --ckpt nav/runs/navv2_d_libcur_s0/best.pt --K 4
"""
import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

from nav import nav_env as NE
from nav.nav_adapter import NavEnvAdapter
from nav.diversity import embed_route
from safety_gymnasium.utils.common_utils import ResamplingError


def rollout_path(adapter, act_fn, pillars, start, skill_id, max_steps=500):
    adapter.set_layout(pillars, start=start)
    try:
        obs = adapter.reset(seed=0, start=start)
    except ResamplingError:
        return None
    path = [adapter._env.task.data.body("agent").xpos[:2].copy()]
    for _ in range(max_steps):
        obs, r, term, trunc, info = adapter.step(act_fn(obs, skill_id))
        path.append(adapter._env.task.data.body("agent").xpos[:2].copy())
        if adapter.success(info) or term or trunc:
            break
    return np.array(path)


def skill_paths_overlay(adapter, act_fn, pillars, start, K, out_path, max_steps=500):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axp = plt.subplots(figsize=(5, 5))
    for x, y in pillars:
        axp.add_patch(plt.Circle((x, y), NE.PILLAR_SIZE, color="0.6"))
    axp.plot(*start, "gs", ms=10, label="start")
    axp.plot(*adapter.goal, "r*", ms=16, label="goal")
    colors = plt.cm.viridis(np.linspace(0, 1, K + 1))
    for k in range(K + 1):
        p = rollout_path(adapter, act_fn, pillars, start, k, max_steps)
        if p is not None:
            axp.plot(p[:, 0], p[:, 1], color=colors[k], lw=1.5,
                     label=f"w_{k}" + ("(deploy)" if k == 0 else ""))
    axp.set_xlim(NE.EXTENTS[0], NE.EXTENTS[2])
    axp.set_ylim(NE.EXTENTS[1], NE.EXTENTS[3])
    axp.set_aspect("equal")
    axp.legend(fontsize=7, loc="upper left"); axp.set_title("K-skill paths (same layout)")
    fig.savefig(out_path, dpi=120, bbox_inches="tight"); plt.close(fig)
    return out_path


def traj_diversity(adapter, act_fn, K, n_layouts=6, seed=3, max_steps=500):
    """within:同布局跨技能路径嵌入平均两两距离;across:同技能跨布局平均两两距离。"""
    rng = np.random.default_rng(seed)
    emb = {}   # (layout_idx, skill) -> embed
    goal = adapter.goal
    li = 0
    tries = 0
    while li < n_layouts and tries < n_layouts * 6:
        tries += 1
        start = NE.sample_valid_start(rng)
        pillars = NE.dedupe(NE.sample_training_layout(rng))
        ok = True
        tmp = {}
        for k in range(K + 1):
            p = rollout_path(adapter, act_fn, pillars, start, k, max_steps)
            if p is None:
                ok = False; break
            tmp[k] = embed_route(p, start, goal)
        if not ok:
            continue
        for k in range(K + 1):
            emb[(li, k)] = tmp[k]
        li += 1

    def pair_mean(vecs):
        d = [np.linalg.norm(vecs[i] - vecs[j])
             for i in range(len(vecs)) for j in range(i + 1, len(vecs))]
        return float(np.mean(d)) if d else 0.0

    within = np.mean([pair_mean([emb[(l, k)] for k in range(K + 1)]) for l in range(li)]) if li else 0.0
    across = np.mean([pair_mean([emb[(l, k)] for l in range(li)]) for k in range(K + 1)]) if li else 0.0
    return {"within_skill_div": float(within), "across_layout_div": float(across), "n_layouts": li}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max_steps", type=int, default=500)
    args = ap.parse_args()

    from nav.skill_td3 import SkillTD3
    agent = SkillTD3(NE.OBS_DIM, 2, K=args.K, device="cuda")
    agent.load(args.ckpt, map_location=agent.device)
    act_fn = lambda o, k: agent.act(o, skill_id=k, noise=0.0)
    ad = NavEnvAdapter()

    rng = np.random.default_rng(0)
    start = NE.START
    pillars = NE.dedupe(NE.sample_training_layout(rng))
    out = args.out or os.path.join(os.path.dirname(args.ckpt), "skill_paths.png")
    skill_paths_overlay(ad, act_fn, pillars, start, args.K, out, args.max_steps)
    print(f"saved overlay -> {out}")
    div = traj_diversity(ad, act_fn, args.K, max_steps=args.max_steps)
    print(f"trajectory diversity: {div}")
    ad.close()


if __name__ == "__main__":
    main()
