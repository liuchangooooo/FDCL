"""技能库 TD3 训练(Stage 1 主体)。

- 采 w_0 与各 w_k∈W_probe 的回合,打 skill_id;普通采样 P(w_0)>=0.5,其余均分。
- 每回合"先采起点、再生成布局":G_0 未就绪前用 nav_env.sample_training_layout 冒烟(仅冒烟)。
- 周期在验证 between 分布(eval_dist)上用 w_0 的平均 episode return 选 best(mode=max),存 ckpt。
- Form B 多样 V0 默认关(beta_div=0)。

用法(safenav,cwd=/home/hnu-w/DIVO/DIVO):
  MUJOCO_GL=egl python -m nav.train_skill --total_steps 300000 --tag navv2_skill0
"""
import argparse
import os
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch

from nav import nav_env as NE
from nav.nav_adapter import NavEnvAdapter
from nav.protocol import write_training_manifest
from nav.skill_td3 import SkillTD3
from nav.td3 import SkillReplayBuffer
from nav.eval_dist import evaluate_validation
from safety_gymnasium.utils.common_utils import ResamplingError

OUTROOT = os.path.join(os.path.dirname(__file__), "runs")


def sample_skill_for_episode(rng, K, p_w0=0.5):
    """普通采样:P(w_0)=p_w0,其余在 w_1..w_K 均分。"""
    if rng.random() < p_w0:
        return 0
    return int(rng.integers(1, K + 1))


def reset_new_scene(adapter, rng, reject_trivial=True):
    """先采起点(randomize)、再生成训练布局;合法且非平凡为止。返回 obs 或 None。

    非平凡:起点离 goal 不能太近(对齐 Push-T 拒平凡起点)。
    """
    for _ in range(30):
        start = NE.sample_valid_start(rng)
        if reject_trivial and not NE.start_goal_ok(start, adapter.goal):
            continue
        pillars = NE.dedupe(NE.sample_training_layout(rng))
        if not pillars:
            continue
        adapter.set_layout(pillars, start=start)
        try:
            obs = adapter.reset(seed=int(rng.integers(0, 2**31 - 1)), start=start)
        except ResamplingError:
            continue
        return obs
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total_steps", type=int, default=300000)
    ap.add_argument("--start_steps", type=int, default=10000)
    ap.add_argument("--max_ep_steps", type=int, default=500)
    ap.add_argument("--eval_every", type=int, default=25000)
    ap.add_argument("--eval_n", type=int, default=20)
    ap.add_argument("--expl_noise", type=float, default=0.1)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--p_w0", type=float, default=0.5)
    ap.add_argument("--beta_div", type=float, default=0.0)   # V0 关多样
    ap.add_argument("--lambda_z", type=float, default=1.0)   # z 高斯正则,对齐 Push-T reg_coeff=1.0
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reject_trivial", type=int, default=1, help="拒绝起点太靠 goal 的平凡场景")
    ap.add_argument("--tag", type=str, default="navv2_skill0")
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    outdir = os.path.join(OUTROOT, args.tag)
    os.makedirs(outdir, exist_ok=True)
    write_training_manifest(
        outdir, trainer="nav.train_skill", seed=args.seed, tag=args.tag
    )
    logf = open(os.path.join(outdir, "train.log"), "a")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n"); logf.flush()

    train_ad = NavEnvAdapter(seed=args.seed)
    eval_ad = NavEnvAdapter(seed=args.seed)
    agent = SkillTD3(NE.OBS_DIM, 2, K=args.K, device="cuda", lambda_z=args.lambda_z)
    buf = SkillReplayBuffer(NE.OBS_DIM, 2, size=1_000_000, device=agent.device)

    log(f"start tag={args.tag} device={agent.device} K={args.K} p_w0={args.p_w0} "
        f"beta_div={args.beta_div} total_steps={args.total_steps}")

    best = -1e9
    step = 0
    ep = 0
    t0 = time.time()

    while step < args.total_steps:
        obs = reset_new_scene(train_ad, rng, reject_trivial=bool(args.reject_trivial))
        if obs is None:
            continue
        skill = sample_skill_for_episode(rng, args.K, args.p_w0)  # 本回合执行的技能
        ep += 1

        for _ in range(args.max_ep_steps):
            if step < args.start_steps:
                a = np.random.uniform(-1, 1, size=2)
            else:
                a = agent.act(obs, skill_id=skill, noise=args.expl_noise)
            next_obs, r_task, term, trunc, info = train_ad.step(a)
            success = train_ad.success(info)
            done = term or success
            # V0:reward_div=0(多样关);buffer 内部按 skill 路由 reward_total/critic
            buf.add(obs, a, next_obs, float(done), skill_id=skill,
                    reward_task=r_task, reward_div=0.0, beta_div=args.beta_div)
            obs = next_obs
            step += 1

            if step >= args.start_steps and buf.n >= args.batch:
                agent.train_step(buf.sample(args.batch))

            if step % args.eval_every == 0:
                # 选 best:验证 between,w_0 部署
                val = evaluate_validation(
                    eval_ad, lambda o: agent.act(o, skill_id=0, noise=0.0),
                    n_env=args.eval_n, max_steps=args.max_ep_steps)
                sps = step / (time.time() - t0)
                log(f"step={step} ep={ep} val_test_mean_score={val['test_mean_score']:.3f} "
                    f"(succ={val['success_rate']:.3f} finald={val['mean_final_dist']:.3f}) sps={sps:.0f}")
                agent.save(os.path.join(outdir, "latest.pt"))
                if val["test_mean_score"] > best:
                    best = val["test_mean_score"]
                    agent.save(os.path.join(outdir, "best.pt"))

            if term or trunc or success:
                break

    agent.save(os.path.join(outdir, "latest.pt"))
    log(f"DONE steps={step} best_val={best:.3f}")
    logf.close()


if __name__ == "__main__":
    main()
