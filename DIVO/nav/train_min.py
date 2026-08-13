"""最简确定性 TD3 训练(单策略)—— motivation 验证第一步。

在训练 DOF(中央带稀疏散柱)上训一个确定性导航策略,周期性评测:
  - held-out 训练分布评测集(固定 N 个布局)的 success rate
  - B/M/U/D 四类零样本 success rate
并存 checkpoint,供 probe_degeneracy.py 验证"单确定性策略 p∈{0,1} 退化"。

用法(safenav 环境):
  MUJOCO_GL=egl python -m nav.train_min --total_steps 300000 --tag navv2_motiv0
"""
import argparse
import os
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

from nav import nav_env as NE
from nav import benchmarks as Bmk
from nav.nav_adapter import NavEnvAdapter
from nav.protocol import write_training_manifest
from nav.td3 import TD3, ReplayBuffer
from safety_gymnasium.utils.common_utils import ResamplingError

OUTROOT = os.path.join(os.path.dirname(__file__), "runs")


def make_eval_set(n, seed):
    """固定的 held-out 训练分布评测布局(用独立 seed,和训练采样不重叠语义)。"""
    rng = np.random.default_rng(seed)
    layouts = []
    while len(layouts) < n:
        pillars = NE.dedupe(NE.sample_training_layout(rng))
        if pillars:
            layouts.append(pillars)
    return layouts


def rollout(adapter, agent, pillars, max_steps, noise=0.0, seed=0):
    """在给定布局上跑一条确定性(或带噪)轨迹,返回 (success, ep_return, steps)。"""
    adapter.set_layout(pillars)
    try:
        obs = adapter.reset(seed=seed)
    except ResamplingError:
        return None  # 非法布局
    ep_r, success = 0.0, False
    for t in range(max_steps):
        a = agent.act(obs, noise=noise)
        obs, r, term, trunc, info = adapter.step(a)
        ep_r += r
        if adapter.success(info):
            success = True
            break
        if term or trunc:
            break
    return success, ep_r, t + 1


def evaluate(adapter, agent, eval_layouts, max_steps):
    succ = []
    for pillars in eval_layouts:
        out = rollout(adapter, agent, pillars, max_steps, noise=0.0, seed=0)
        if out is None:
            continue
        succ.append(out[0])
    return float(np.mean(succ)) if succ else 0.0, len(succ)


def evaluate_bmud(static_adapter, dynamic_adapter, agent, max_steps):
    """用正式参数化 B/M/U/D 各跑 1 回合；完整多场景结果由 nav.eval 产生。"""
    act_fn = lambda obs: agent.act(obs, noise=0.0)
    return Bmk.evaluate_bmud(
        static_adapter,
        act_fn,
        n_env=1,
        max_steps=max_steps,
        dyn_adapter=dynamic_adapter,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total_steps", type=int, default=300000)
    ap.add_argument("--start_steps", type=int, default=10000)
    ap.add_argument("--max_ep_steps", type=int, default=500)
    ap.add_argument("--eval_every", type=int, default=20000)
    ap.add_argument("--eval_n", type=int, default=64)
    ap.add_argument("--expl_noise", type=float, default=0.1)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", type=str, default="navv2_motiv0")
    args = ap.parse_args()

    np.random.seed(args.seed)
    import torch
    torch.manual_seed(args.seed)

    outdir = os.path.join(OUTROOT, args.tag)
    os.makedirs(outdir, exist_ok=True)
    write_training_manifest(
        outdir, trainer="nav.train_min", seed=args.seed, tag=args.tag
    )
    logf = open(os.path.join(outdir, "train.log"), "a")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    train_ad = NavEnvAdapter(seed=args.seed)
    eval_ad = NavEnvAdapter(seed=args.seed)
    dynamic_eval_ad = NavEnvAdapter(seed=args.seed, dynamic=True)
    agent = TD3(NE.OBS_DIM, 2)
    buf = ReplayBuffer(NE.OBS_DIM, 2, size=1_000_000, device=agent.device)

    eval_layouts = make_eval_set(args.eval_n, seed=10_000 + args.seed)
    log(f"start tag={args.tag} device={agent.device} eval_n={len(eval_layouts)} "
        f"total_steps={args.total_steps}")

    rng = np.random.default_rng(args.seed)
    best_eval = -1.0
    step = 0
    ep = 0
    t0 = time.time()

    while step < args.total_steps:
        # 采一个训练布局(合法为止)
        obs = None
        for _ in range(20):
            pillars = NE.dedupe(NE.sample_training_layout(rng))
            if not pillars:
                continue
            train_ad.set_layout(pillars)
            try:
                obs = train_ad.reset(seed=int(rng.integers(0, 2**31 - 1)))
                break
            except ResamplingError:
                continue
        if obs is None:
            continue

        ep += 1
        ep_r = 0.0
        for t in range(args.max_ep_steps):
            if step < args.start_steps:
                a = np.random.uniform(-1, 1, size=2)
            else:
                a = agent.act(obs, noise=args.expl_noise)
            next_obs, r, term, trunc, info = train_ad.step(a)
            success = train_ad.success(info)
            done = term or success
            buf.add(obs, a, r, next_obs, float(done))
            obs = next_obs
            ep_r += r
            step += 1

            if step >= args.start_steps and buf.n >= args.batch:
                agent.train_step(buf, args.batch)

            if step % args.eval_every == 0:
                ev, nev = evaluate(eval_ad, agent, eval_layouts, args.max_ep_steps)
                bmud = evaluate_bmud(eval_ad, dynamic_eval_ad, agent, args.max_ep_steps)
                sps = step / (time.time() - t0)
                log(f"step={step} ep={ep} eval_succ={ev:.3f}(n={nev}) "
                    f"BMUD={ {k: round(float(bmud[k]), 3) for k in ('B', 'M', 'U', 'D')} } "
                    f"sps={sps:.0f}")
                agent.save(os.path.join(outdir, "latest.pt"))
                if ev >= best_eval:
                    best_eval = ev
                    agent.save(os.path.join(outdir, "best.pt"))

            if term or trunc or success:
                break

    agent.save(os.path.join(outdir, "latest.pt"))
    log(f"DONE steps={step} best_eval={best_eval:.3f}")
    train_ad.close()
    eval_ad.close()
    dynamic_eval_ad.close()
    logf.close()


if __name__ == "__main__":
    main()
