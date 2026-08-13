"""四格主对比(Stage 2)驱动 + 结果聚合(Req 12.1/12.2)。

对齐 Push-T:四臂来自**同一入口 train_stage2**,只切训练库、反馈信号与进化开关,
采样机制/代码路径完全相同:
  (S0) 只训 w_0 + 单技能反馈进化
  (b') 训练技能库 + 单技能反馈进化
  (c)  训练技能库 + 固定 G_0(不进化)
  (d)  训练技能库 + 技能库 boundary 反馈进化(本方法)
红线:(d) > (c) 且 (d) > (b')。

公平性:四臂共用同一 G_0 + 共用验证 between 选 best,≥3 seeds。
  - mock:同 seed 的 G_0 确定,四臂自动一致。
  - openai:先跑 (d) 生成并存盘 runs/<d_tag>/g0_code.py,(S0)(b')(c) 用
    --init_generator_path 载入同一 G_0(见 print_commands 输出)。
train_min.py / train_skill.py 降级为 legacy(不再用于四格,避免采样混淆)。
本脚本:print 各臂命令 + 聚合已存在的 eval_bmud.json;不自动起长训。
"""
import argparse
import glob
import json
import math
import os
import re
import statistics

from nav.protocol import (
    BENCHMARK_VERSION,
    FORMAL_EVALUATION_CONFIG,
    TRAINING_PROTOCOL_VERSION,
    ProtocolError,
    load_training_manifest,
    training_manifest_sha256,
)

RUNS = os.path.join(os.path.dirname(__file__), "runs")
METHOD_PATTERNS = (
    ("S0", re.compile(r"^navv4_S0_s\d+$")),
    ("b'", re.compile(r"^navv4_bprime_s\d+$")),
    ("c", re.compile(r"^navv4_c_lib_s\d+$")),
    ("d (ours)", re.compile(r"^navv4_d_libcur_s\d+$")),
)


def print_commands(seeds, provider="mock", steps=300000):
    print(f"# 主对比命令(hydra 覆盖 key=value;provider={provider};每臂 >=3 seeds;"
          f"两个解耦开关 training.train_library × curriculum.evolve.probe_feedback):")
    print(f"# 归因:d−b'=信号增益;b'−S0=多技能辅助增益;d−c=课程(进化 vs 固定)增益")
    for s in seeds:
        print(f"\n## seed {s}")
        tags = {
            "d": f"navv4_d_libcur_s{s}",
            "bprime": f"navv4_bprime_s{s}",
            "S0": f"navv4_S0_s{s}",
            "c": f"navv4_c_lib_s{s}",
        }
        # openai:各臂共享同一 G_0(先 (d) 生成 g0,余臂载入);mock:同 seed 自动一致
        g0 = os.path.join(RUNS, tags["d"], "g0_code.py")   # 绝对路径,避免 cwd/hydra 歧义
        shared = (
            f" curriculum.generator.init_mode=file curriculum.generator.init_path={g0}"
            if provider == "openai" else ""
        )
        base = f"python -m nav.train_stage2 provider={provider} training.total_steps={steps} seed={s}"
        print(f"# (d) 本方法:整库 + 库信号进化(openai 时先跑,存盘 {g0} 供余臂共享)")
        print(
            f"{base} training.train_library=true "
            f"curriculum.evolve.probe_feedback=library tag={tags['d']}"
        )
        print(f"python -m nav.eval --ckpt nav/runs/{tags['d']}/best.pt --kind skill --K 4")
        print(f"# (b') 信号对照:整库(同 d) + 单 w_0 信号进化  [d−b'=信号增益]")
        print(
            f"{base} training.train_library=true "
            f"curriculum.evolve.probe_feedback=single{shared} tag={tags['bprime']}"
        )
        print(f"python -m nav.eval --ckpt nav/runs/{tags['bprime']}/best.pt --kind skill --K 4")
        print(f"# (S0) 基线:只训 w_0 + 单信号进化  [b'−S0=多技能辅助增益]")
        print(
            f"{base} training.train_library=false "
            f"curriculum.evolve.probe_feedback=single{shared} tag={tags['S0']}"
        )
        print(f"python -m nav.eval --ckpt nav/runs/{tags['S0']}/best.pt --kind skill --K 4")
        print(f"# (c) 库无课程:整库 + 不进化(固定 G_0)  [d−c=课程增益]")
        print(
            f"{base} training.train_library=true "
            f"curriculum.evolve.enabled=false{shared} tag={tags['c']}"
        )
        print(f"python -m nav.eval --ckpt nav/runs/{tags['c']}/best.pt --kind skill --K 4")


def aggregate():
    rows = {}
    skipped = {}
    for f in glob.glob(os.path.join(RUNS, "*", "eval_bmud.json")):
        try:
            with open(f, "r", encoding="utf-8") as result_file:
                d = json.load(result_file)
        except (OSError, json.JSONDecodeError):
            skipped["unreadable result"] = skipped.get("unreadable result", 0) + 1
            continue
        tag = os.path.basename(os.path.dirname(f))
        reason = None
        if d.get("benchmark_version") != BENCHMARK_VERSION:
            reason = "benchmark version missing/mismatch"
        elif d.get("evaluation_config") != FORMAL_EVALUATION_CONFIG:
            reason = "evaluation config missing/non-standard"
        elif d.get("training_protocol_version") != TRAINING_PROTOCOL_VERSION:
            reason = "training protocol version missing/mismatch"
        elif d.get("training_protocol_verified") is not True:
            reason = "training protocol not verified"
        elif d.get("diagnostic_override") is not False:
            reason = "diagnostic/ineligible result"
        elif d.get("formal_aggregate_eligible") is not True:
            reason = "diagnostic/ineligible result"
        else:
            run_dir = os.path.dirname(os.path.abspath(f))
            ckpt = d.get("ckpt")
            if (
                not isinstance(ckpt, str)
                or not os.path.isfile(ckpt)
                or os.path.realpath(os.path.dirname(ckpt)) != os.path.realpath(run_dir)
                or os.path.basename(ckpt) != "best.pt"
            ):
                reason = "checkpoint is not the run-directory best.pt"
            else:
                try:
                    manifest = load_training_manifest(run_dir)
                except ProtocolError:
                    reason = "training manifest missing/invalid"
                else:
                    if manifest["run"]["tag"] != tag:
                        reason = "manifest tag does not match result directory"
                    elif d.get("training_manifest_sha256") != training_manifest_sha256(manifest):
                        reason = "training manifest digest mismatch"
        bmud = d.get("BMUD")
        required = ("B", "M", "U", "D", "D_static", "D_dynamic", "dynamic_drop", "AVG")
        if reason is None and (
            not isinstance(bmud, dict)
            or any(
                type(bmud.get(key)) not in (int, float) or not math.isfinite(float(bmud[key]))
                for key in required
            )
        ):
            reason = "BMUD payload missing/non-scalar/non-finite"
        if reason is None and (
            abs(float(bmud["D"]) - float(bmud["D_dynamic"])) > 1e-12
            or abs(float(bmud["dynamic_drop"]) - (
                float(bmud["D_static"]) - float(bmud["D_dynamic"])
            )) > 1e-12
            or abs(float(bmud["AVG"]) - sum(float(bmud[key]) for key in ("B", "M", "U", "D")) / 4) > 1e-12
        ):
            reason = "BMUD derived fields are inconsistent"
        if reason is not None:
            skipped[reason] = skipped.get(reason, 0) + 1
            continue
        rows[tag] = d.get("BMUD", {})
    if skipped:
        summary = ", ".join(
            f"{reason}: {count}" for reason, count in sorted(skipped.items())
        )
        print(f"忽略非正式 Nav v4 结果({summary})")
    if not rows:
        print(f"(暂无 {BENCHMARK_VERSION} 的 eval_bmud.json 结果)")
        return
    print(f"benchmark_version={BENCHMARK_VERSION}")
    print(f"{'tag':28s}  B     M     U     D     D_static  dynamic_drop  AVG")
    for tag, b in sorted(rows.items()):
        print(f"{tag:28s}  " + "  ".join(
            f"{b.get(k, float('nan')):.2f}"
            for k in ("B", "M", "U", "D", "D_static", "dynamic_drop", "AVG")
        ))

    grouped = {}
    for tag, result in rows.items():
        method = next((name for name, pattern in METHOD_PATTERNS if pattern.match(tag)), None)
        if method is not None:
            grouped.setdefault(method, []).append(result)
    if grouped:
        keys = ("B", "M", "U", "D", "D_static", "dynamic_drop", "AVG")
        print("\n跨 seed 汇总（sample mean±std）")
        print(f"{'method':12s}  n  " + "  ".join(f"{key:>13s}" for key in keys))
        for method, _ in METHOD_PATTERNS:
            results = grouped.get(method, [])
            if not results:
                continue
            cells = []
            for key in keys:
                values = [float(result[key]) for result in results]
                std = statistics.stdev(values) if len(values) > 1 else 0.0
                cells.append(f"{statistics.mean(values):.3f}±{std:.3f}")
            print(f"{method:12s}  {len(results):d}  " + "  ".join(f"{cell:>13s}" for cell in cells))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--provider", choices=["mock", "openai"], default="mock")
    ap.add_argument("--steps", type=int, default=300000)
    ap.add_argument("--aggregate", action="store_true")
    args = ap.parse_args()
    if args.aggregate:
        aggregate()
    else:
        print_commands(args.seeds, provider=args.provider, steps=args.steps)


if __name__ == "__main__":
    main()
