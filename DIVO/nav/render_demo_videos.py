"""同一场景 + 同一 w_0 策略,生成两版演示视频供选择:
  A. 示意风格(matplotlib 矢量俯视,带轨迹尾巴)
  B. 真实 MuJoCo 引擎俯视渲染

用法(safenav):
  MUJOCO_GL=egl python -m nav.render_demo_videos --ckpt nav/runs/navv2_d_libcur_s0/best.pt --K 4
输出:所给 checkpoint 同目录的 showcase/demoA_schematic.mp4 / demoB_mujoco.mp4
"""
import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from nav import nav_env as NE
from nav.nav_adapter import NavEnvAdapter
from nav import benchmarks as Bmk
from safety_gymnasium.utils.common_utils import ResamplingError

def run_and_capture(ad, act_fn, pillars, start, max_steps, camera_capture=True, seed=0):
    """跑一条 w_0 轨迹,返回 (agent_xy_path, mujoco_frames)。"""
    ad.set_layout(pillars, start=start)
    obs = ad.reset(seed=seed, start=start)
    path = [ad._env.task.data.body("agent").xpos[:2].copy()]
    frames = [ad.render()] if camera_capture else []
    for _ in range(max_steps):
        obs, r, term, trunc, info = ad.step(act_fn(obs))
        path.append(ad._env.task.data.body("agent").xpos[:2].copy())
        if camera_capture:
            frames.append(ad.render())
        if ad.success(info) or term or trunc:
            break
    return np.array(path), frames


def schematic_frames(path, pillars, start, goal, pillar_size):
    """把 agent 轨迹画成示意风格逐帧(带尾巴),返回 rgb 帧列表。"""
    frames = []
    for i in range(1, len(path) + 1):
        fig, ax = plt.subplots(figsize=(4.5, 4.5), dpi=110)
        ax.set_xlim(NE.EXTENTS[0], NE.EXTENTS[2])
        ax.set_ylim(NE.EXTENTS[1], NE.EXTENTS[3])
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#3b5b92"); s.set_linewidth(2)
        ax.set_facecolor("#f7f9fc")
        for x, y in pillars:
            ax.add_patch(Circle((x, y), pillar_size, color="#d94a4a", alpha=0.85, zorder=2))
        ax.plot(*start, marker="X", color="#555", ms=13, mew=2, zorder=3)
        ax.plot(*goal, marker="*", color="#2e9e4f", ms=20, mec="#1c6b34", zorder=3)
        seg = path[:i]
        ax.plot(seg[:, 0], seg[:, 1], "-", color="#2b6cb0", lw=2, alpha=0.8, zorder=4)
        ax.plot(seg[-1, 0], seg[-1, 1], "o", color="#2b6cb0", ms=10, zorder=5)
        ax.set_title("Navigation demo (w_0)  schematic", fontsize=11)
        fig.tight_layout()
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))[..., :3]
        frames.append(buf.copy())
        plt.close(fig)
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--max_steps", type=int, default=250)
    ap.add_argument("--camera", default="fixednear")
    args = ap.parse_args()

    outdir = os.path.join(os.path.dirname(args.ckpt), "showcase")
    os.makedirs(outdir, exist_ok=True)

    from nav.skill_td3 import SkillTD3
    agent = SkillTD3(NE.OBS_DIM, 2, K=args.K, device="cuda")
    agent.load(args.ckpt, map_location=agent.device)
    w0 = lambda o: agent.act(o, skill_id=0, noise=0.0)

    # 选一个视觉清晰的场景:M(3 柱)基准,固定起点
    sc = Bmk.sample_benchmark_scene("M", 7)
    pillars, size = sc["pillars"], sc["size"]
    start = sc["start"]

    # 版本 B:真实 MuJoCo(top-down 相机)
    ad = NavEnvAdapter(render_mode="rgb_array", camera_name=args.camera)
    ad.set_layout(pillars, start=start, pillar_size=size, pillar_keepout=sc["keepout"])
    path, frames = run_and_capture(
        ad, w0, pillars, start, args.max_steps, camera_capture=True,
        seed=sc["reset_seed"],
    )
    imageio.mimsave(f"{outdir}/demoB_mujoco.mp4", frames, fps=30)
    print(f"版本 B(MuJoCo,{args.camera},{len(frames)} 帧)-> demoB_mujoco.mp4")
    ad.close()

    # 版本 A:示意风格(用同一条轨迹)
    sframes = schematic_frames(path, pillars, start, NE.GOAL, size)
    imageio.mimsave(f"{outdir}/demoA_schematic.mp4", sframes, fps=30)
    print(f"版本 A(示意,{len(sframes)} 帧)-> demoA_schematic.mp4")
    print(f"\n输出目录: {outdir}")


if __name__ == "__main__":
    main()
