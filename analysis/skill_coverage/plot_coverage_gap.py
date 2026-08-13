"""Plot the skill-coverage gap diagnostic results."""
import json
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent


def load(p):
    return json.loads(pathlib.Path(p).read_text())["summary"]


easy = load(ROOT / "results" / "coverage_gap_llm_evolve_s0.json")          # 0.05 evolve
hard = load(ROOT / "results_hard" / "coverage_gap_llm_evolve_s0.json")     # 0.08 evolve
ev = load(ROOT / "results_paired" / "coverage_gap_llm_evolve_s0.json")     # 0.08 evolve (paired)
st = load(ROOT / "results_paired" / "coverage_gap_llm_static_s0.json")     # 0.08 static (paired)

metrics = ["mean_feasible", "mean_deployed", "mean_realized"]
labels = ["feasible\n(best-of-K)", "deployed\n(encoder z)", "realized\n(random z)"]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

# Panel A: difficulty effect (evolve)
ax = axes[0]
x = np.arange(len(metrics))
w = 0.38
ax.bar(x - w / 2, [easy[m] for m in metrics], w, label="obstacle 0.05", color="#4C9F70")
ax.bar(x + w / 2, [hard[m] for m in metrics], w, label="obstacle 0.08", color="#C25B56")
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylim(0, 1.05); ax.set_ylabel("success rate")
ax.set_title("(a) Coverage gap vs difficulty (LLM-Evolve)")
for i, m in enumerate(metrics):
    ax.text(i - w / 2, easy[m] + 0.02, f"{easy[m]:.2f}", ha="center", fontsize=8)
    ax.text(i + w / 2, hard[m] + 0.02, f"{hard[m]:.2f}", ha="center", fontsize=8)
ax.legend(fontsize=8)

# Panel B: evolve vs static at 0.08 (paired scenes)
ax = axes[1]
ax.bar(x - w / 2, [ev[m] for m in metrics], w, label="LLM-Evolve", color="#3B6EA5")
ax.bar(x + w / 2, [st[m] for m in metrics], w, label="LLM-Static", color="#E1A730")
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylim(0, 1.05); ax.set_ylabel("success rate")
ax.set_title("(b) Evolve vs Static (obstacle 0.08, paired)")
for i, m in enumerate(metrics):
    ax.text(i - w / 2, ev[m] + 0.02, f"{ev[m]:.2f}", ha="center", fontsize=8)
    ax.text(i + w / 2, st[m] + 0.02, f"{st[m]:.2f}", ha="center", fontsize=8)
ax.legend(fontsize=8)

fig.tight_layout()
out = ROOT / "coverage_gap_summary.png"
fig.savefig(out, dpi=140)
print(f"saved: {out}")

print("\n--- numbers ---")
for name, s in [("evolve@0.05", easy), ("evolve@0.08", hard),
                ("evolve@0.08(paired)", ev), ("static@0.08(paired)", st)]:
    print(f"{name:22s} feasible={s['mean_feasible']:.3f} "
          f"deployed={s['mean_deployed']:.3f} realized={s['mean_realized']:.3f} "
          f"feas-real={s['mean_gap']:.3f} cov-opp={s['frac_coverage_opportunity']:.3f}")
