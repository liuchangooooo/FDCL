"""Plot the deployed-vs-library decoupling over evolve rounds for a curriculum run.

Reads acgs_evolve_records.jsonl from a run directory and produces:
  - a CSV + markdown table of per-evolve metrics
  - a PNG curve: training sr / probe deployed vs library realized (+ lv, infeasible)
"""
import argparse
import json
import re
import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SR_RE = re.compile(r"sr=([0-9.]+)")


def load_records(run_dir: pathlib.Path):
    path = run_dir / "acgs_evolve_records.jsonl"
    rows = []
    for line in path.open():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        prof = r.get("skill_signal_profile") or {}
        reason = r.get("reason") or ""
        m = SR_RE.search(reason)
        train_sr = float(m.group(1)) if m else float("nan")
        rows.append({
            "ep": int(r.get("total_episode_count", 0)),
            "accepted": bool(r.get("accepted")),
            "train_sr": train_sr,
            "deployed": prof.get("mean_deployed", float("nan")),
            "realized": prof.get("mean_realized", float("nan")),
            "lv": prof.get("mean_lv", float("nan")),
            "infeasible": prof.get("frac_infeasible", float("nan")),
            "boundary": prof.get("frac_boundary", float("nan")),
        })
    return rows


def write_table(rows, out_csv: pathlib.Path):
    cols = ["round", "ep", "accepted", "train_sr", "deployed", "realized", "lv", "infeasible", "boundary"]
    lines = [",".join(cols)]
    md = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for i, r in enumerate(rows, 1):
        vals = [i, r["ep"], int(r["accepted"]),
                f"{r['train_sr']:.3f}", f"{r['deployed']:.3f}", f"{r['realized']:.3f}",
                f"{r['lv']:.3f}", f"{r['infeasible']:.3f}", f"{r['boundary']:.3f}"]
        lines.append(",".join(str(v) for v in vals))
        md.append("| " + " | ".join(str(v) for v in vals) + " |")
    out_csv.write_text("\n".join(lines) + "\n")
    return "\n".join(md)


def plot(rows, out_png: pathlib.Path, title: str):
    x = np.arange(1, len(rows) + 1)
    train_sr = [r["train_sr"] for r in rows]
    deployed = [r["deployed"] for r in rows]
    realized = [r["realized"] for r in rows]
    lv = [r["lv"] for r in rows]
    infe = [r["infeasible"] for r in rows]
    acc = [r["accepted"] for r in rows]

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.plot(x, train_sr, "-o", color="#1b9e77", lw=2.2, ms=6, label="training rollout success (deployed task)")
    ax.plot(x, deployed, "--s", color="#66c2a5", lw=1.8, ms=5, label="probe mean_deployed (best deployed z)")
    ax.plot(x, realized, "-^", color="#d95f02", lw=2.2, ms=6, label="probe mean_realized (skill library avg)")
    ax.plot(x, lv, "-D", color="#7570b3", lw=1.6, ms=5, label="probe mean_lv = realized*(1-realized)")
    ax.plot(x, infe, ":v", color="#999999", lw=1.6, ms=5, label="probe frac_infeasible")

    # shade the decoupling gap (deployed task sr vs library realized)
    ax.fill_between(x, realized, train_sr, color="#d95f02", alpha=0.08)

    # mark accepted evolves
    for xi, a in zip(x, acc):
        if a:
            ax.axvline(xi, color="#cccccc", lw=0.6, zorder=0)

    ax.set_xlabel("evolve trigger round")
    ax.set_ylabel("success / metric value")
    ax.set_ylim(-0.02, 1.05)
    ax.set_xticks(x)
    ax.grid(True, alpha=0.25)
    ax.set_title(title)
    ax.legend(loc="center right", framealpha=0.92, fontsize=9)

    # annotate the decoupling region
    ax.annotate("deployed mastered (~1.0)\nbut library realized stuck ~0.3\n=> curriculum no longer\npushes the deployed policy",
                xy=(len(rows) * 0.72, 0.62), fontsize=9, color="#7a3b00",
                ha="left", va="center",
                bbox=dict(boxstyle="round,pad=0.4", fc="#fff3e6", ec="#d95f02", alpha=0.9))

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"[saved] {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    run_dir = pathlib.Path(args.run_dir)
    out_dir = pathlib.Path(args.out_dir) if args.out_dir else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_records(run_dir)
    md = write_table(rows, out_dir / "decoupling_table.csv")
    plot(rows, out_dir / "decoupling_curve.png", f"Deployed vs skill-library decoupling — {run_dir.name}")
    print(md)


if __name__ == "__main__":
    main()
