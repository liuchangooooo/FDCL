"""最终评测脚本:在 best ckpt 上跑 B/M/U/D(固定起终点+随机障碍+20回合平均)+ 可选结构化拓展。

w_0 单技能单阶段部署;B/M/U/D 与拓展套件不参与选 best(仅最终零样本报告)。
支持技能库 ckpt(SkillTD3,用 w_0)与单策略 ckpt(TD3)。

用法:
  MUJOCO_GL=egl python -m nav.eval --ckpt nav/runs/navv2_skill0/best.pt --kind skill --K 4
  MUJOCO_GL=egl python -m nav.eval --ckpt nav/runs/navv2_motiv0/best.pt --kind single
"""
import argparse
import json
import os

os.environ.setdefault("MUJOCO_GL", "egl")

from nav import nav_env as NE
from nav.nav_adapter import NavEnvAdapter
from nav import benchmarks as B
from nav.protocol import (
    BENCHMARK_VERSION,
    FORMAL_EVALUATION_CONFIG,
    ProtocolError,
    training_manifest_sha256,
    validate_checkpoint_training_protocol,
)


def load_act_fn(kind, ckpt, K):
    if kind == "skill":
        from nav.skill_td3 import SkillTD3
        agent = SkillTD3(NE.OBS_DIM, 2, K=K, device="cuda")
        agent.load(ckpt, map_location=agent.device)
        return lambda o: agent.act(o, skill_id=0, noise=0.0)   # w_0 部署
    else:
        from nav.td3 import TD3
        agent = TD3(NE.OBS_DIM, 2, device="cuda")
        agent.load(ckpt, map_location=agent.device)
        return lambda o: agent.act(o, noise=0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--kind", choices=["skill", "single"], default="skill")
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--n_env", type=int, default=FORMAL_EVALUATION_CONFIG["n_env"])
    ap.add_argument(
        "--max_steps", type=int, default=FORMAL_EVALUATION_CONFIG["max_steps"]
    )
    ap.add_argument(
        "--base-seed",
        type=int,
        default=FORMAL_EVALUATION_CONFIG["base_seed"],
        help="B/M/U/D 随机障碍场景的基准种子；非 2024 时仅作诊断",
    )
    ap.add_argument("--extension", action="store_true", help="也跑结构化拓展套件")
    ap.add_argument(
        "--allow-unverified-checkpoint",
        action="store_true",
        help="仅作旧 checkpoint 诊断；结果会标为不可正式聚合",
    )
    args = ap.parse_args()

    args.ckpt = os.path.abspath(args.ckpt)
    manifest = None
    protocol_error = None
    try:
        manifest = validate_checkpoint_training_protocol(args.ckpt)
    except ProtocolError as exc:
        protocol_error = str(exc)
        if not args.allow_unverified_checkpoint:
            ap.error(
                f"{exc}. Formal Nav v4 evaluation requires best.pt with a same-directory "
                "training manifest; use --allow-unverified-checkpoint only for diagnostics"
            )
        print(f"WARNING: unverified checkpoint diagnostic: {exc}")

    evaluation_config = {
        "n_env": int(args.n_env),
        "max_steps": int(args.max_steps),
        "base_seed": int(args.base_seed),
    }
    standard_evaluation = evaluation_config == FORMAL_EVALUATION_CONFIG
    if not standard_evaluation:
        print(
            "WARNING: non-standard evaluation config; result is diagnostic-only: "
            f"{evaluation_config} (formal={FORMAL_EVALUATION_CONFIG})"
        )

    diagnostic_mode = bool(args.allow_unverified_checkpoint) or not standard_evaluation
    protocol_verified = manifest is not None
    formal_eligible = protocol_verified and not diagnostic_mode
    act_fn = load_act_fn(args.kind, args.ckpt, args.K)
    ad = NavEnvAdapter()

    bmud = B.evaluate_bmud(
        ad,
        act_fn,
        n_env=args.n_env,
        max_steps=args.max_steps,
        base_seed=args.base_seed,
    )
    result = {
        "benchmark_version": BENCHMARK_VERSION,
        "ckpt": args.ckpt,
        "kind": args.kind,
        "training_protocol_version": (
            manifest["training_protocol"]["version"] if manifest is not None else None
        ),
        "training_protocol_verified": protocol_verified,
        "training_manifest_sha256": (
            training_manifest_sha256(manifest) if manifest is not None else None
        ),
        "diagnostic_override": diagnostic_mode,
        "formal_aggregate_eligible": formal_eligible,
        "evaluation_config": evaluation_config,
        "BMUD": bmud,
    }
    if protocol_error is not None:
        result["training_protocol_error"] = protocol_error
    print("=" * 70)
    result_mode = "formal" if formal_eligible else "diagnostic-only"
    print(
        f"eval {args.ckpt} ({args.kind}, w_0 部署; {BENCHMARK_VERSION}; {result_mode})"
    )
    print(f"  B/M/U/D = { {k: round(v,3) for k,v in bmud.items()} }")
    if args.extension:
        ext = B.evaluate_structured_extension(ad, act_fn, max_steps=args.max_steps)
        result["structured_extension"] = ext
        print(f"  结构化拓展(非主基准)= {ext}")
    print("=" * 70)

    filename = "eval_bmud_diagnostic.json" if diagnostic_mode else "eval_bmud.json"
    out = os.path.join(os.path.dirname(args.ckpt), filename)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"saved -> {out}")
    ad.close()


if __name__ == "__main__":
    main()
