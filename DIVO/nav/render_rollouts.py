"""用 fixedfar 相机(远景俯视,和 rollout_train/validation 同风格)录 w_0 在各类场景的 rollout。
输出:所给 checkpoint 同目录的 showcase/roll_<scene>.mp4（train/validation/B/M/U/D）

用法:MUJOCO_GL=egl python -m nav.render_rollouts --ckpt nav/runs/navv2_d_libcur_s0/best.pt --K 4
"""
import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import imageio.v2 as imageio

from nav import nav_env as NE
from nav.nav_adapter import NavEnvAdapter
from nav import benchmarks as Bmk
from nav.eval_dist import sample_validation_scene
from safety_gymnasium.utils.common_utils import ResamplingError


def record(ad, act_fn, name, pillars, start, outdir, size=None, keepout=None,
           travel=None, randomize_start=False, max_steps=250, fps=30, seed=0):
    ad.set_layout(pillars, start=start, pillar_size=size, pillar_keepout=keepout,
                  gremlin_travel=travel)
    try:
        obs = ad.reset(seed=seed, start=start, randomize_start=randomize_start)
    except ResamplingError:
        return False
    frames = [ad.render()]
    reached = False
    for _ in range(max_steps):
        obs, r, term, trunc, info = ad.step(act_fn(obs))
        frames.append(ad.render())
        if ad.success(info):
            reached = True; break
        if term or trunc:
            break
    path = f"{outdir}/roll_{name}.mp4"
    imageio.mimsave(path, frames, fps=fps)
    print(f"  {name}: {len(frames)} 帧, reached={reached} -> {path}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--max_steps", type=int, default=250)
    args = ap.parse_args()
    outdir = os.path.join(os.path.dirname(args.ckpt), "showcase")
    os.makedirs(outdir, exist_ok=True)

    from nav.skill_td3 import SkillTD3
    agent = SkillTD3(NE.OBS_DIM, 2, K=args.K, device="cuda")
    agent.load(args.ckpt, map_location=agent.device)
    w0 = lambda o: agent.act(o, skill_id=0, noise=0.0)

    ad = NavEnvAdapter(render_mode="rgb_array", camera_name="fixedfar")
    dynamic_ad = NavEnvAdapter(
        render_mode="rgb_array", camera_name="fixedfar", dynamic=True
    )
    rng = np.random.default_rng(0)

    # 训练分布
    tr = NE.dedupe(NE.sample_training_layout(rng)) or [(0.0, 0.0)]
    record(ad, w0, "train", tr, NE.START, outdir, max_steps=args.max_steps)
    # 验证 between
    vs = sample_validation_scene(3)
    record(ad, w0, "validation", vs["pillars"], vs["start"], outdir, max_steps=args.max_steps)
    # B/M/U/D
    names = {"B": "Big", "M": "Multiple", "U": "Unstructured", "D": "Dynamic"}
    for fam in ("B", "M", "U", "D"):
        print(f"  {fam} ({names[fam]}):")
        recorded = False
        for scene_seed in range(7, 27):
            sc = Bmk.sample_benchmark_scene(fam, scene_seed)
            scene_ad = dynamic_ad if sc.get("dynamic", False) else ad
            recorded = record(
                scene_ad, w0, fam, sc["pillars"], sc["start"], outdir,
                size=sc["size"], keepout=sc["keepout"], travel=sc.get("travel"),
                max_steps=args.max_steps, seed=sc["reset_seed"],
            )
            if recorded:
                break
        if not recorded:
            print(f"  {fam}: 20 次场景采样均无法 reset，跳过")

    ad.close()
    dynamic_ad.close()
    print(f"\n输出目录: {outdir}")


if __name__ == "__main__":
    main()
