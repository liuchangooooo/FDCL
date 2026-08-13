"""Panel (3) real learnability-signal assets from a real M-scene probe.

Runs the K-skill probe over M generator layouts (rollout_scene_with_z_bank),
then produces:
  hist_realized.(svg|png) : realized distribution, 3-class colored (infeasible /
                            boundary / mastery) -- the "Distribution summary".
  fha_focus/harden/avoid.png : real top-down renders of a representative
                            focus (realized~0.5), harden (high), avoid (realized=0)
                            layout for the S(g_t) mini-exhibit.
  signalA_meta.json : frac_* / boundary_count / mean_lv / per-scene realized.
"""
from __future__ import annotations
import json, os, pathlib, sys
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, torch
from omegaconf import OmegaConf
try:
    OmegaConf.register_new_resolver("now", lambda p: "now")
except Exception:
    pass

REPO = pathlib.Path("/home/hnu-w/DIVO")
sys.path.insert(0, str(REPO))
from DIVO.env import get_env_class
from DIVO.policy import get_policy
from DIVO.curriculum.skill_signal import (
    SceneSamplingStats, build_z_bank, rollout_scene_with_z_bank, sample_generator_scene,
)
from DIVO.curriculum.candidate_verifier import SandboxPushTExecutor

CKPT_DIR = REPO / "data/outputs/2026.06.22_g0_skill_signal_evolve_s0_clean"
CKPT = CKPT_DIR / "checkpoints/best430k.pt"
CFG = CKPT_DIR / ".hydra/config.yaml"
GEN = REPO / "data/outputs/2026.04.29/g0_seed2/current_generator.py"
OUT = REPO / "figure_outputs/panelA_signal"
K, M, MAX_STEPS, SEED = 6, 40, 10, 5

C_INF, C_BND, C_MAS = "#E0897B", "#F4C95D", "#93C1A4"  # pale infeasible/boundary/mastery


def render(env):
    for kw in (dict(height=480, width=480, camera_id=0), dict()):
        try:
            return np.asarray(env.physics.render(**kw))
        except Exception:
            continue
    return None


def main():
    np.random.seed(SEED); torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = OmegaConf.load(str(CFG))
    env = get_env_class(**OmegaConf.to_container(cfg.env, resolve=True))
    policy = get_policy(env, **OmegaConf.to_container(cfg.policy, resolve=True)).to(device)
    policy.load_state_dict(torch.load(str(CKPT), map_location=device), strict=False)
    policy.eval()
    z_bank = build_z_bank(K=K, latent_dim=int(cfg.latent_dim), seed=SEED, device=device)
    ex = SandboxPushTExecutor(obstacle_size=0.01)
    ex.load(GEN.read_text(encoding="utf-8"))
    stats = SceneSamplingStats()
    OUT.mkdir(parents=True, exist_ok=True)

    scenes = []
    tries = 0
    while len(scenes) < M and tries < M * 25:
        tries += 1
        sc = sample_generator_scene(env, ex, 2, timeout_sec=5, stats=stats)
        if sc is None:
            continue
        res = rollout_scene_with_z_bank(env, policy, sc["start"], sc["obstacles"], z_bank, device, MAX_STEPS)
        realized = float(np.mean([int(r["success"]) for r in res["routes"]]))
        scenes.append(dict(start=sc["start"], obstacles=sc["obstacles"], realized=realized))
    print(f"[probe] collected {len(scenes)} scenes")

    rz = np.array([s["realized"] for s in scenes])
    frac_inf = float(np.mean(rz == 0.0))
    frac_mas = float(np.mean(rz >= 1.0 - 1e-9))
    frac_bnd = float(1.0 - frac_inf - frac_mas)
    bcount = int(np.sum((rz > 0.0) & (rz < 1.0)))
    mean_lv = float(np.mean(rz * (1.0 - rz)))
    print(f"[summary] frac_inf={frac_inf:.2f} frac_bnd={frac_bnd:.2f} frac_mas={frac_mas:.2f} "
          f"boundary_count={bcount} mean_lv={mean_lv:.3f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ---- realized histogram, colored by class (bins at k/K) ----
    edges = (np.arange(K + 2) - 0.5) / K  # centers at 0,1/K,...,1
    counts, _ = np.histogram(rz, bins=edges)
    centers = np.arange(K + 1) / K
    colors = [C_INF] + [C_BND] * (K - 1) + [C_MAS]
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    fig.patch.set_alpha(0.0)
    ax.bar(centers, counts, width=0.8 / K, color=colors, edgecolor="#666", linewidth=0.6)
    ax.set_xlabel("realized  $p$", fontsize=12)
    ax.set_ylabel("# scenes", fontsize=12)
    ax.set_xlim(-0.5 / K, 1 + 0.5 / K)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=10)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=C_INF, label="infeasible"),
                       Patch(facecolor=C_BND, label="boundary"),
                       Patch(facecolor=C_MAS, label="mastery")],
              fontsize=8, frameon=False, loc="upper center")
    fig.tight_layout()
    fig.savefig(OUT / "hist_realized.svg", transparent=True, bbox_inches="tight")
    fig.savefig(OUT / "hist_realized.png", dpi=300, transparent=True, bbox_inches="tight")
    plt.close(fig)

    # ---- focus / harden / avoid representative real renders ----
    def pick(key):
        if key == "focus":
            return min(scenes, key=lambda s: abs(s["realized"] - 0.5))
        if key == "harden":
            cand = [s for s in scenes if 0 < s["realized"] < 1.0] or scenes
            return max(cand, key=lambda s: s["realized"])
        avoid = [s for s in scenes if s["realized"] == 0.0]
        return avoid[0] if avoid else min(scenes, key=lambda s: s["realized"])

    picks = {}
    for key in ("focus", "harden", "avoid"):
        s = pick(key)
        env.set_obstacle_config(s["obstacles"])
        env.reset(tblock_pos=np.asarray(s["start"], float), force_tblock_pos=True)
        img = render(env)
        if img is not None:
            plt.imsave(OUT / f"fha_{key}.png", img)
        picks[key] = s["realized"]
        print(f"[{key}] realized={s['realized']:.3f}")

    (OUT / "signalA_meta.json").write_text(json.dumps(dict(
        K=K, M=len(scenes), frac_infeasible=frac_inf, frac_boundary=frac_bnd,
        frac_mastery=frac_mas, boundary_count=bcount, mean_lv=mean_lv,
        picks_realized=picks, realized=[float(x) for x in rz.tolist()],
    ), indent=2))
    print(f"[done] {OUT}")


if __name__ == "__main__":
    main()
