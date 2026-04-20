import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import importlib.util

import numpy as np

# Headless render (server / SSH)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Allow running from any cwd
REPO_ROOT = Path(__file__).resolve().parents[3]  # /home/hnu-w/DIVO
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_llm_topology_generator_module():
    """Load llm_topology_generator.py without importing DIVO.env (avoids dm_control dependency)."""
    module_path = REPO_ROOT / "DIVO" / "env" / "pusht" / "llm_topology_generator.py"
    spec = importlib.util.spec_from_file_location("llm_topology_generator", str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_topo = _load_llm_topology_generator_module()
LLMTopologyGenerator = _topo.LLMTopologyGenerator
StrategyExecutor = _topo.StrategyExecutor
build_phase0_topology_generator_prompt_compact = _topo.build_phase0_topology_generator_prompt_compact


def sample_tblock_poses(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-0.18, 0.18, size=(n, 2))
    theta = rng.uniform(-np.pi, np.pi, size=(n, 1))
    return np.concatenate([xy, theta], axis=1)


def draw_tblock(
    ax,
    x: float,
    y: float,
    angle: float,
    color: str = "tab:blue",
    alpha: float = 0.25,
    lw: float = 1.2,
    label: str = "",
):
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

    bar1 = np.array([[-0.05, 0.00], [0.05, 0.00], [0.05, 0.03], [-0.05, 0.03]])
    bar2 = np.array([[-0.015, -0.07], [0.015, -0.07], [0.015, 0.00], [-0.015, 0.00]])

    bar1 = bar1 @ rot.T + np.array([x, y])
    bar2 = bar2 @ rot.T + np.array([x, y])

    ax.add_patch(
        patches.Polygon(
            bar1,
            closed=True,
            facecolor=color,
            edgecolor="black",
            alpha=alpha,
            linewidth=lw,
        )
    )
    ax.add_patch(
        patches.Polygon(
            bar2,
            closed=True,
            facecolor=color,
            edgecolor="black",
            alpha=alpha,
            linewidth=lw,
        )
    )

    if label:
        ax.text(x, y, label, fontsize=9, ha="center", va="center")


def draw_obstacle(
    ax,
    x: float,
    y: float,
    size: float = 0.02,
    color: str = "tab:red",
    alpha: float = 0.65,
    text: str = "",
):
    half = size / 2.0
    rect = patches.Rectangle(
        (x - half, y - half),
        size,
        size,
        facecolor=color,
        edgecolor="black",
        alpha=alpha,
    )
    ax.add_patch(rect)
    if text:
        ax.text(x, y + half + 0.006, text, fontsize=7, ha="center", va="bottom")


def render_scene(
    tblock_pose: np.ndarray,
    obstacles: List[Dict[str, Any]],
    ok: bool,
    reason: str,
    out_path: str,
    obstacle_size: float,
):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.2, 0.2)
    ax.set_ylim(-0.2, 0.2)
    ax.grid(True, alpha=0.25)

    ax.add_patch(
        patches.Rectangle(
            (-0.2, -0.2),
            0.4,
            0.4,
            fill=False,
            edgecolor="black",
            linewidth=1.2,
        )
    )

    x, y, theta = float(tblock_pose[0]), float(tblock_pose[1]), float(tblock_pose[2])

    draw_tblock(ax, x, y, theta, color="tab:blue", alpha=0.25, label="start")
    draw_tblock(ax, 0.0, 0.0, -np.pi / 4, color="tab:green", alpha=0.18, label="target")

    ax.plot([x, 0.0], [y, 0.0], linestyle="--", color="gray", linewidth=1.0, alpha=0.7)

    for i, obs in enumerate(obstacles):
        purpose = str(obs.get("purpose", ""))
        draw_obstacle(ax, float(obs["x"]), float(obs["y"]), size=obstacle_size, text=f"{i}:{purpose[:10]}")

    ax.set_title(f"ok={ok}")

    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-type", choices=["deepseek", "openai"], default="deepseek")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--temperature", type=float, default=0.3)

    ap.add_argument("-n", "--num-scenes", type=int, default=50)
    ap.add_argument("-k", "--num-obstacles", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--outdir", default="compact_scenes")
    ap.add_argument(
        "--per-scene-llm",
        action="store_true",
        help="每个场景都调用 LLM 重新生成函数（成本高）",
    )
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    meta_path = os.path.join(args.outdir, "meta.jsonl")
    summary_path = os.path.join(args.outdir, "summary.json")

    poses = sample_tblock_poses(args.num_scenes, args.seed)

    llm = LLMTopologyGenerator(
        api_type=args.api_type,
        api_key=args.api_key,
        model=args.model,
        base_url=args.base_url,
        temperature=args.temperature,
        verbose=False,
    )
    executor = StrategyExecutor()

    loaded_once = False
    cached_code = None

    stats: Dict[str, Any] = {
        "num_scenes": int(args.num_scenes),
        "num_obstacles_requested": int(args.num_obstacles),
        "seed": int(args.seed),
        "api_type": args.api_type,
        "model": llm.model,
        "temperature": float(args.temperature),
        "per_scene_llm": bool(args.per_scene_llm),
        "stages": {"ok": 0, "llm": 0, "extract": 0, "load": 0, "validate_fail": 0},
        "load_ok_rate": 0.0,
        "valid_ok_rate": 0.0,
        "avg_generated_n": 0.0,
    }

    generated_ns: List[int] = []
    load_ok_count = 0
    valid_ok_count = 0

    with open(meta_path, "w", encoding="utf-8") as fmeta:
        for i in range(args.num_scenes):
            pose = poses[i]

            if args.per_scene_llm or (not loaded_once):
                prompt = build_phase0_topology_generator_prompt_compact(pose, args.num_obstacles)
                resp = llm._call_llm(prompt)
                if resp is None:
                    stats["stages"]["llm"] += 1
                    fmeta.write(
                        json.dumps(
                            {
                                "i": i,
                                "tblock_pose": pose.tolist(),
                                "stage": "llm",
                                "ok": False,
                                "reason": "llm response None",
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    continue

                code = llm._extract_code(resp)
                if code is None:
                    stats["stages"]["extract"] += 1
                    fmeta.write(
                        json.dumps(
                            {
                                "i": i,
                                "tblock_pose": pose.tolist(),
                                "stage": "extract",
                                "ok": False,
                                "reason": "extract None",
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    continue

                if not executor.load_topology_generator(code):
                    stats["stages"]["load"] += 1
                    fmeta.write(
                        json.dumps(
                            {
                                "i": i,
                                "tblock_pose": pose.tolist(),
                                "stage": "load",
                                "ok": False,
                                "reason": "exec failed",
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    continue

                cached_code = code
                loaded_once = True

            load_ok_count += 1

            obstacles = executor.generate(pose, args.num_obstacles) or []
            generated_n = len(obstacles)
            generated_ns.append(generated_n)

            ok, reason = executor.validate_obstacles(obstacles, pose)
            ok = bool(ok)

            if ok:
                valid_ok_count += 1
                stats["stages"]["ok"] += 1
            else:
                stats["stages"]["validate_fail"] += 1

            out_png = os.path.join(args.outdir, f"scene_{i:03d}.png")
            render_scene(
                tblock_pose=pose,
                obstacles=obstacles,
                ok=ok,
                reason=str(reason),
                out_path=out_png,
                obstacle_size=executor.obstacle_size * 2,
            )

            record = {
                "i": i,
                "tblock_pose": pose.tolist(),
                "ok": ok,
                "reason": str(reason),
                "generated_n": generated_n,
                "obstacles": obstacles,
                "png": out_png,
                "used_cached_code": (not args.per_scene_llm),
                "code_len": (len(cached_code) if cached_code is not None else 0),
            }
            fmeta.write(json.dumps(record, ensure_ascii=False) + "\n")

    stats["load_ok_rate"] = (load_ok_count / args.num_scenes) if args.num_scenes else 0.0
    stats["valid_ok_rate"] = (valid_ok_count / args.num_scenes) if args.num_scenes else 0.0
    stats["avg_generated_n"] = float(np.mean(generated_ns)) if generated_ns else 0.0

    with open(summary_path, "w", encoding="utf-8") as fsum:
        json.dump(stats, fsum, ensure_ascii=False, indent=2)

    print(f"saved scenes to: {args.outdir}")
    print(f"meta: {meta_path}")
    print(f"summary: {summary_path}")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
