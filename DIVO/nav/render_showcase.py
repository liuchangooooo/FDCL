"""渲染训练环境展示:场景画廊(PNG)+ 训练好的 w_0 导航视频(MP4)+ K 技能路径叠图。

用法(safenav,cwd=/home/hnu-w/DIVO/DIVO):
  MUJOCO_GL=egl python -m nav.render_showcase --ckpt nav/runs/navv2_d_libcur_s0/best.pt --K 4
输出到所给 checkpoint 同目录的 showcase/。
"""
import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import imageio.v2 as imageio

from nav import nav_env as NE
from nav.nav_adapter import NavEnvAdapter
from nav import benchmarks as B
from nav.eval_dist import sample_validation_scene
from safety_gymnasium.utils.common_utils import ResamplingError


def save_scene_png(ad, path, pillars, start=None, goal=None, size=None, keepout=None,
                   travel=None, randomize_start=False, warmup_steps=0, seed=0):
    ad.set_layout(pillars, start=start, goal=goal, pillar_size=size,
                  pillar_keepout=keepout, gremlin_travel=travel)
    ad.reset(seed=seed, start=start, randomize_start=randomize_start)
    for _ in range(warmup_steps):
        ad.step(np.zeros(2))
    imageio.imwrite(path, ad.render())


def record_rollout(ad, act_fn, path, pillars, start, max_steps=200, fps=30):
    ad.set_layout(pillars, start=start)
    try:
        obs = ad.reset(seed=0, start=start)
    except ResamplingError:
        print(f"  rollout skip(invalid layout): {path}")
        return
    frames = [ad.render()]
    for _ in range(max_steps):
        obs, r, term, trunc, info = ad.step(act_fn(obs))
        frames.append(ad.render())
        if ad.success(info) or term or trunc:
            break
    imageio.mimsave(path, frames, fps=fps)
    print(f"  video({len(frames)} frames) -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--max_steps", type=int, default=250)
    args = ap.parse_args()

    outdir = os.path.join(os.path.dirname(args.ckpt), "showcase")
    os.makedirs(outdir, exist_ok=True)

    ad = NavEnvAdapter(render_mode="rgb_array")
    dynamic_ad = NavEnvAdapter(render_mode="rgb_array", dynamic=True)
    rng = np.random.default_rng(0)

    # ---- 场景画廊 ----
    print("场景画廊:")
    # 训练分布代表性场景(统一规范起点)
    tr = NE.dedupe(NE.sample_training_layout(rng)) or [(0.0, 0.0)]
    save_scene_png(ad, f"{outdir}/scene_train.png", tr, start=NE.START)
    print(f"  训练分布 -> scene_train.png (pillars={len(tr)})")
    # 验证 between
    vs = sample_validation_scene(3)
    save_scene_png(ad, f"{outdir}/scene_validation.png", vs["pillars"], start=vs["start"], goal=vs["goal"])
    print(f"  验证 between -> scene_validation.png")
    # B/M/U/D 各一张(参数化)
    names = {"B": "Big", "M": "Multiple", "U": "Unstructured", "D": "Dynamic"}
    for fam in ("B", "M", "U", "D"):
        saved = False
        for scene_seed in range(7, 27):
            sc = B.sample_benchmark_scene(fam, scene_seed)
            scene_ad = dynamic_ad if sc.get("dynamic", False) else ad
            try:
                save_scene_png(
                    scene_ad,
                    f"{outdir}/bench_{fam}.png",
                    sc["pillars"],
                    start=sc["start"],
                    goal=sc["goal"],
                    size=sc["size"],
                    keepout=sc["keepout"],
                    travel=sc.get("travel"),
                    warmup_steps=20 if sc.get("dynamic", False) else 0,
                    seed=sc["reset_seed"],
                )
            except ResamplingError:
                continue
            print(
                f"  {fam} ({names[fam]}) -> bench_{fam}.png "
                f"(num={len(sc['pillars'])}, size={sc['size']}, scene_seed={scene_seed})"
            )
            saved = True
            break
        if not saved:
            print(f"  {fam} ({names[fam]}): 20 次场景采样均无法 reset，跳过")

    # ---- 训练好的 w_0 导航视频 ----
    if os.path.exists(args.ckpt):
        from nav.skill_td3 import SkillTD3
        agent = SkillTD3(NE.OBS_DIM, 2, K=args.K, device="cuda")
        agent.load(args.ckpt, map_location=agent.device)
        w0 = lambda o: agent.act(o, skill_id=0, noise=0.0)
        print("w_0 导航视频:")
        # 验证 between 场景上录一段
        record_rollout(ad, w0, f"{outdir}/rollout_validation.mp4", vs["pillars"], vs["start"], args.max_steps)
        # 训练分布场景(固定起点便于观看)
        record_rollout(ad, w0, f"{outdir}/rollout_train.mp4", tr, NE.START, args.max_steps)

        # ---- K 技能路径叠图 ----
        from nav.viz import skill_paths_overlay
        act_k = lambda o, k: agent.act(o, skill_id=k, noise=0.0)
        skill_paths_overlay(ad, act_k, tr, NE.START, args.K,
                            f"{outdir}/kskill_paths.png", args.max_steps)
        print(f"  K 技能路径叠图 -> kskill_paths.png")
    else:
        print(f"(未找到 ckpt {args.ckpt},跳过视频/叠图)")

    ad.close()
    dynamic_ad.close()
    print(f"\n全部输出在: {outdir}")


if __name__ == "__main__":
    main()
