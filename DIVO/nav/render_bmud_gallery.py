"""正式 B/M/U/D 场景可视化:PNG 画廊 + 真实动态 D 视频。

不依赖策略 checkpoint；所有场景均来自 benchmarks.sample_benchmark_scene。
D 视频让 Point 保持零动作，只展示正式 Gremlin 动态本身。

用法:MUJOCO_GL=egl python -m nav.render_bmud_gallery --per 1
"""
import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import imageio.v2 as imageio

from nav.nav_adapter import NavEnvAdapter
from nav import benchmarks as B
from nav.protocol import BENCHMARK_VERSION
from safety_gymnasium.utils.common_utils import ResamplingError


def render_scene(static_ad, dynamic_ad, fam, seed):
    sc = B.sample_benchmark_scene(fam, seed)
    ad = dynamic_ad if sc.get("dynamic", False) else static_ad
    ad.set_layout(sc["pillars"], start=sc["start"], goal=sc["goal"],
                  pillar_size=sc["size"], pillar_keepout=sc["keepout"],
                  gremlin_travel=sc.get("travel"))
    try:
        ad.reset(seed=sc["reset_seed"], start=sc["start"])
        # D 的 PNG 取真实动态环境经过若干 step 后的帧，而非静态 pillar 代图。
        if sc.get("dynamic", False):
            for _ in range(20):
                ad.step(np.zeros(2))
        return ad.render()
    except ResamplingError:
        return None


def record_dynamic_d(dynamic_ad, seed, path, steps=320, fps=30, frame_stride=2):
    """录制一个正式 D scene；每帧前进真实仿真，不依赖策略权重。"""
    sc = B.sample_benchmark_scene("D", seed)
    dynamic_ad.set_layout(
        sc["pillars"],
        start=sc["start"],
        goal=sc["goal"],
        pillar_size=sc["size"],
        pillar_keepout=sc["keepout"],
        gremlin_travel=sc["travel"],
    )
    dynamic_ad.reset(seed=sc["reset_seed"], start=sc["start"])
    frame_count = 1
    with imageio.get_writer(
        path, fps=fps, codec="libx264", quality=8, pixelformat="yuv420p"
    ) as writer:
        writer.append_data(np.asarray(dynamic_ad.render()))
        for step in range(1, int(steps) + 1):
            dynamic_ad.step(np.zeros(2))
            if step % int(frame_stride) == 0 or step == int(steps):
                writer.append_data(np.asarray(dynamic_ad.render()))
                frame_count += 1
    print(
        f"D(Dynamic): {frame_count} frames/{int(steps)} env steps, fps={int(fps)}, "
        f"scene_seed={seed}, travel={sc['travel']} -> {path}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per", type=int, default=4, help="每类实例数")
    ap.add_argument(
        "--base-seed", type=int, default=2025,
        help="展示场景起始种子；2025 是正式 base_seed=2024 的首个 attempt",
    )
    ap.add_argument("--video-steps", type=int, default=320, help="D 视频仿真步数；0 表示不生成")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--frame-stride", type=int, default=2, help="D 视频每隔多少仿真步取一帧")
    ap.add_argument(
        "--out", default=f"nav/templates/{BENCHMARK_VERSION}/bmud_gallery.png"
    )
    ap.add_argument("--video-out", default=None)
    args = ap.parse_args()
    if args.per < 1:
        ap.error("--per must be >= 1")
    if args.video_steps < 0:
        ap.error("--video-steps must be >= 0")
    if args.fps < 1:
        ap.error("--fps must be >= 1")
    if args.frame_stride < 1:
        ap.error("--frame-stride must be >= 1")
    outdir = os.path.dirname(args.out) or "."
    os.makedirs(outdir, exist_ok=True)

    fams = ["B", "M", "U", "D"]
    names = {"B": "Big", "M": "Multiple", "U": "Unstructured", "D": "Dynamic"}
    static_ad = NavEnvAdapter(render_mode="rgb_array", camera_name="fixedfar")
    dynamic_ad = NavEnvAdapter(render_mode="rgb_array", camera_name="fixedfar", dynamic=True)

    grid = {}
    scene_seeds = {}
    for fam in fams:
        grid[fam] = []
        scene_seeds[fam] = []
        got, seed = 0, int(args.base_seed)
        while got < args.per and seed < int(args.base_seed) + args.per * 20:
            scene_seed = seed
            img = render_scene(static_ad, dynamic_ad, fam, scene_seed)
            seed += 1
            if img is not None:
                grid[fam].append(np.asarray(img))
                scene_seeds[fam].append(scene_seed)
                got += 1
        if not grid[fam]:
            static_ad.close()
            dynamic_ad.close()
            raise RuntimeError(f"no valid {fam} scene in sampled seed range")
        print(
            f"{fam}({names[fam]}): {len(grid[fam])} instances, "
            f"scene_seeds={scene_seeds[fam]}"
        )

    # 单场景原图便于逐类查看；与画廊第一列严格一致。
    for fam in fams:
        imageio.imwrite(os.path.join(outdir, f"benchmark_{fam}.png"), grid[fam][0])

    if args.video_steps:
        video_out = args.video_out or os.path.join(outdir, "benchmark_D_dynamic.mp4")
        video_dir = os.path.dirname(video_out) or "."
        os.makedirs(video_dir, exist_ok=True)
        record_dynamic_d(
            dynamic_ad,
            scene_seeds["D"][0],
            video_out,
            steps=args.video_steps,
            fps=args.fps,
            frame_stride=args.frame_stride,
        )
    static_ad.close()
    dynamic_ad.close()

    # 单实例用紧凑 2x2；多实例保持每行一类。
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if args.per == 1:
        fig, axes = plt.subplots(2, 2, figsize=(8, 8))
        for ax, fam in zip(axes.flat, fams):
            ax.axis("off")
            ax.imshow(grid[fam][0])
            ax.set_title(
                f"{fam} ({names[fam]}) · seed {scene_seeds[fam][0]}",
                loc="left",
                fontsize=12,
            )
    else:
        fig, axes = plt.subplots(
            len(fams), args.per, figsize=(args.per * 3, len(fams) * 3)
        )
        for r, fam in enumerate(fams):
            for c in range(args.per):
                ax = axes[r][c]
                ax.axis("off")
                if c < len(grid[fam]):
                    ax.imshow(grid[fam][c])
                    ax.set_title(
                        f"seed {scene_seeds[fam][c]}", loc="left", fontsize=9
                    )
                if c == 0:
                    ax.set_ylabel(f"{fam} ({names[fam]})", fontsize=12)
    fig.suptitle(
        f"Navigation B/M/U/D benchmark instances (from seed {int(args.base_seed)})",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"saved gallery -> {args.out}")
    print(f"saved first scene of each family -> {outdir}/benchmark_{{B,M,U,D}}.png")


if __name__ == "__main__":
    main()
