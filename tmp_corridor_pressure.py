import json, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN = "data/outputs/2026.06.03/22.17.22_td3_pusht_llm_curriculum"
JSONL = f"{RUN}/obstacle_rollouts.jsonl"
WINDOW = 5000  # episodes per evolve round


def alpha_weight(a):
    if 0.0 <= a <= 1.0:
        return 1.0
    if (-0.25 <= a < 0.0) or (1.0 < a <= 1.25):
        return 0.5
    return 0.15


def corridor_pressure(obstacle_z):
    p = 0.0
    for z in obstacle_z:
        if not isinstance(z, dict):
            continue
        a = float(z.get("alpha", 0.0))
        b = float(z.get("blockage", 0.0))
        p += alpha_weight(a) * b
    return p


eids, press, fail = [], [], []
with open(JSONL) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        eid = r.get("episode_id")
        if eid is None:
            continue
        oz = r.get("obstacle_z") or []
        eids.append(int(eid))
        press.append(corridor_pressure(oz))
        fail.append(0 if str(r.get("termination")) == "success" else 1)

eids = np.array(eids); press = np.array(press); fail = np.array(fail)
print(f"episodes={len(eids)}  pressure: mean={press.mean():.4f} max={press.max():.4f} "
      f"p90={np.quantile(press,0.9):.4f} p99={np.quantile(press,0.99):.4f} "
      f"frac>0.1={np.mean(press>0.1):.3f} frac>0.3={np.mean(press>0.3):.3f}")
print(f"overall failure_rate={fail.mean():.3f}")

# ---- Validation: does failure rate rise with corridor_pressure? ----
print("\n[Validation] failure rate vs corridor_pressure (quantile bins):")
qs = np.quantile(press, np.linspace(0, 1, 7))
qs = np.unique(qs)
print(f"{'pressure range':>22} {'n':>7} {'meanP':>7} {'failRate':>9}")
for i in range(len(qs) - 1):
    lo, hi = qs[i], qs[i + 1]
    m = (press >= lo) & (press <= hi if i == len(qs) - 2 else press < hi)
    if m.sum() == 0:
        continue
    print(f"[{lo:7.3f},{hi:7.3f}] {m.sum():>7d} {press[m].mean():>7.3f} {fail[m].mean():>9.3f}")

# fixed-bin view too (robust to skew)
print("\n[Validation] fixed bins:")
edges = [0.0, 1e-9, 0.05, 0.1, 0.2, 0.4, 10.0]
labels = ["==0", "0-0.05", "0.05-0.1", "0.1-0.2", "0.2-0.4", ">0.4"]
print(f"{'bin':>10} {'n':>7} {'failRate':>9}")
for lab, lo, hi in zip(labels, edges[:-1], edges[1:]):
    m = (press >= lo) & (press < hi)
    if m.sum() == 0:
        print(f"{lab:>10} {0:>7d} {'--':>9}")
        continue
    print(f"{lab:>10} {m.sum():>7d} {fail[m].mean():>9.3f}")

# ---- Per-round (evolve) modulation curve ----
rnd = eids // WINDOW
rounds = sorted(set(rnd.tolist()))
r_mid, r_meanP, r_fail, r_n = [], [], [], []
for w in rounds:
    m = rnd == w
    if m.sum() == 0:
        continue
    r_mid.append(w + 1)
    r_meanP.append(press[m].mean())
    r_fail.append(fail[m].mean())
    r_n.append(int(m.sum()))

print("\n[Modulation] per evolve round (graph_pattern, open-loop):")
print(f"{'round':>5} {'ep_end':>7} {'meanPressure':>13} {'failRate':>9} {'n':>6}")
for i in range(len(r_mid)):
    print(f"{r_mid[i]:>5} {r_mid[i]*WINDOW:>7} {r_meanP[i]:>13.4f} {r_fail[i]:>9.3f} {r_n[i]:>6}")

# ---- Figures ----
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
# left: failure rate vs pressure (fixed bins)
xs, ys = [], []
for lab, lo, hi in zip(labels, edges[:-1], edges[1:]):
    m = (press >= lo) & (press < hi)
    if m.sum() >= 20:
        xs.append(lab); ys.append(fail[m].mean())
ax[0].bar(xs, ys, color="#c0392b")
ax[0].set_title("failure rate vs corridor_pressure")
ax[0].set_ylabel("failure rate"); ax[0].set_xlabel("corridor_pressure bin")
ax[0].tick_params(axis='x', rotation=30)
# right: per-round modulation
ax2 = ax[1]
ax2.plot(r_mid, r_meanP, "-o", color="#2980b9", label="mean corridor_pressure")
ax2.set_xlabel("evolve round"); ax2.set_ylabel("mean corridor_pressure", color="#2980b9")
ax3 = ax2.twinx()
ax3.plot(r_mid, [1-x for x in r_fail], "-s", color="#27ae60", label="success rate")
ax3.set_ylabel("success rate", color="#27ae60")
ax2.set_title("graph_pattern: difficulty vs round (open-loop)")
fig.tight_layout()
out = f"{RUN}/tmp_corridor_pressure_diag.png"
fig.savefig(out, dpi=130)
print(f"\nsaved figure -> {out}")
