"""渲染 between 验证分布的若干场景为一张画廊 PNG(供肉眼查看验证环境)。

between 验证(eval_dist.sample_validation_scene):num=2 pillar(±0.50 区)+ 起点采在
"某 pillar 与 goal 之间"(径向偏移 0.45～0.65,goal 固定 (0.65,0)),
用于 w_0 部署选 best,不参与 B/M/U/D。

用法(safenav,cwd=/home/hnu-w/DIVO/DIVO):
  MUJOCO_GL=egl python -m nav.render_between --n 6 --out nav/runs/between_val.png
"""
import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import imageio.v2 as imageio

from nav.nav_adapter import NavEnvAdapter
from nav.eval_dist import sample_validation_scene
from safety_gymnasium.utils.common_utils import ResamplingError


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--base_seed", type=int, default=777)
    ap.add_argument("--out", type=str, default="nav/runs/between_val.png")
    args = ap.parse_args()

    ad = NavEnvAdapter(render_mode="rgb_array", camera_name="fixedfar")
    frames = []
    seed = args.base_seed
    while len(frames) < args.n and seed < args.base_seed + args.n * 30:
        seed += 1
        sc = sample_validation_scene(seed)
        ad.set_layout(sc["pillars"], start=sc["start"], goal=sc["goal"])
        try:
            ad.reset(seed=0, start=sc["start"])
        except ResamplingError:
            continue
        img = ad.render()
        frames.append(img)
        print(f"  scene {len(frames)}: start={tuple(round(v,2) for v in sc['start'])} "
              f"pillars={[(round(x,2),round(y,2)) for x,y in sc['pillars']]}")
    ad.close()

    # 拼成一行(或两行)网格
    n = len(frames)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    h, w, c = frames[0].shape
    canvas = np.full((rows * h, cols * w, c), 255, np.uint8)
    for i, f in enumerate(frames):
        r, cc = divmod(i, cols)
        canvas[r*h:(r+1)*h, cc*w:(cc+1)*w] = f
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    imageio.imwrite(args.out, canvas)
    print(f"saved {args.out}  ({n} between-validation scenes, fixedfar 俯视)")


if __name__ == "__main__":
    main()
