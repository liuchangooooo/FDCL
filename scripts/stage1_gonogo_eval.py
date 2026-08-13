"""Stage 1 go/no-go evaluation CLI (Task 13 entry point).

Loads a trained skill-library checkpoint and probes it on the fixed difficulty
ladder to report the Route-B Stage 1 signals:
  - per-scene p_i and library realized, per-category stratification (easy~1,
    hard~0, mid 0<p<1);
  - K_eff = exp(H(cluster distribution));
  - per-skill success/progress and non-trivial fraction;
  - the Stage 1 hard-gate summary;
  - K-skill T-block fan plots for a few scenes.

Usage:
  python scripts/stage1_gonogo_eval.py --ckpt <run>/checkpoints/model_latest.pt
"""

import argparse
import json
import pathlib
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from DIVO.env import get_env_class
from DIVO.policy import get_policy
from DIVO.curriculum.stage1_ladder import build_sampled_stage1_ladder
from DIVO.curriculum.stage1_eval import run_stage1_probe, stage1_gonogo
from DIVO.curriculum.skill_diversity import task_frame_embed, compute_k_eff
from DIVO.curriculum.skill_viz import plot_skill_fan


def _load_cfg(ckpt_path):
    run_dir = str(pathlib.Path(ckpt_path).resolve().parents[1])
    cfg = OmegaConf.load(f"{run_dir}/.hydra/config.yaml")
    # Do NOT OmegaConf.resolve() the whole config: evaluator.output_dir uses the
    # hydra-only ${now:...} resolver. Needed fields (env/policy/skill) resolve
    # lazily on access.
    return cfg, run_dir


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="model state_dict .pt (in <run>/checkpoints/)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n_per_category", type=int, default=5)
    ap.add_argument("--keff_threshold", type=float, default=None)
    ap.add_argument("--n_fan_plots", type=int, default=6)
    ap.add_argument("--out", default=None, help="output dir (default: <run>/stage1_gonogo)")
    args = ap.parse_args(argv)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg, run_dir = _load_cfg(args.ckpt)
    out_dir = pathlib.Path(args.out or f"{run_dir}/stage1_gonogo")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not bool(cfg.policy.get("skill_enabled", False)):
        print("[WARN] checkpoint was trained with skill_enabled=false (mode A); "
              "probe skills will all equal the deployment policy.")
    K = int(cfg.skill.K) if "skill" in cfg else int(cfg.policy.get("K", 0))
    max_steps = int(cfg.max_steps)
    threshold = args.keff_threshold
    if threshold is None:
        threshold = float(cfg.skill.get("keff_cluster_threshold", 0.5)) if "skill" in cfg else 0.5

    env = get_env_class(**cfg.env)
    policy = get_policy(env, **cfg.policy).to(device)
    policy.load_state_dict(torch.load(args.ckpt, map_location=device))
    policy.eval()
    print(f"[loaded] {args.ckpt}  (skill_enabled={policy.skill_enabled}, K={K}, device={device})")

    # Env-valid, difficulty-stratified ladder (the hand-placed corridor ladder is
    # geometrically infeasible on Push-T; sampled tiers are valid by construction).
    ladder = build_sampled_stage1_ladder(
        env,
        obstacle_num=int(cfg.env.obstacle_num),
        n_per_tier=int(args.n_per_category),
        seed=int(getattr(cfg.training, "seed", 0)),
    )
    records = run_stage1_probe(env, policy, ladder, K=K, device=device, max_steps=max_steps)

    # K_eff (fixed protocol) from probe skill trajectories
    emb_per_env = [[task_frame_embed(st, max_steps) for st in r["skill_states"]] for r in records]
    keff = compute_k_eff(emb_per_env, threshold=threshold)

    report = stage1_gonogo(records, K=K, k_eff=keff["k_eff"])

    # ---- print summary ----
    print("\n===== Stage 1 go/no-go =====")
    print(f"K_eff = {keff['k_eff']:.3f}  (occupied_clusters={keff['occupied_clusters']:.2f}, threshold={threshold})")
    print(f"library realized (mean p_i) = {np.mean([r['realized'] for r in records]):.3f}")
    print(f"deploy (w_0) success = {np.mean([r['deploy_success'] for r in records]):.3f}")
    print("\nper-tier mean p_i (stratification):")
    for cat in ("easy", "mid", "hard"):
        cm = report["category_means"].get(cat)
        if cm:
            print(f"  {cat:16s} p={cm['mean_p']:.3f}  deploy={cm['mean_deploy']:.3f}  n={cm['n']}")
    print(f"\np_i non-degenerate: {report['p_distribution']['non_degenerate']} "
          f"(easy_high={report['p_distribution']['has_easy_high']}, "
          f"hard_low={report['p_distribution']['has_hard_low']}, mid={report['p_distribution']['has_mid']})")
    print(f"per-skill success rate: {[round(x,3) for x in report['per_skill']['success_rate']]}")
    print(f"non-trivial skill fraction: {report['nontrivial_fraction']:.3f}")
    print(f"\nHARD GATES: {report['gates']}")
    print(f"GO/NO-GO PASSED (K_eff & non-trivial): {report['passed']}")
    print("(B/M/U/D come from evaluation.py; best ckpt by test_mean_score. D is observation-only at Stage 1.)")

    # ---- fan plots for a few scenes ----
    n_plots = min(int(args.n_fan_plots), len(records))
    for i in range(n_plots):
        r = records[i]
        plot_skill_fan(
            r["skill_states"], r["obstacles"],
            str(out_dir / f"fan_{i:02d}_{r['category']}.png"),
            title=f"{r['category']}  p={r['p']:.2f} deploy={r['deploy_success']}",
        )
    print(f"\n[saved] {n_plots} fan plots + report -> {out_dir}")

    (out_dir / "gonogo_report.json").write_text(
        json.dumps({"k_eff": keff, "report": report,
                    "mean_realized": float(np.mean([r['realized'] for r in records])),
                    "mean_deploy": float(np.mean([r['deploy_success'] for r in records]))},
                   indent=2, default=float),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
