"""Correlate library-coverage metrics (probed on existing best checkpoints, common
seed/setting) against deployment generalization B/M/U/D (100ep). No retraining."""
import numpy as np

# probed at obstacle_num=2, obstacle_size=0.08, seed=0, n_scenes=40, k=16
# (from analysis/skill_coverage/results_diversity/coverage_div_*.json)
rows = {
    #            realized feasible deployed div_ratio   B     M     U     D
    "band_guard": (0.252, 0.925, 0.875, 0.569, 0.81, 0.69, 0.86, 0.08),
    "static":     (0.284, 0.925, 0.800, 0.582, 0.76, 0.70, 0.77, 0.06),
    "evolve":     (0.323, 0.950, 0.850, 0.619, 0.83, 0.66, 0.81, 0.32),
    "between":    (0.420, 0.925, 0.825, 0.563, 0.84, 0.70, 0.78, 0.13),
}
names = list(rows)
arr = np.array([rows[n] for n in names], float)
realized, feasible, deployed, divratio, B, M, U, D = [arr[:, i] for i in range(8)]
AVG = (B + M + U + D) / 4

def spearman(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    d2 = np.sum((rx - ry) ** 2); n = len(x)
    return 1 - 6 * d2 / (n * (n * n - 1))

def pearson(x, y):
    return float(np.corrcoef(x, y)[0, 1])

xs = {"realized": realized, "feasible": feasible, "deployed": deployed, "div_ratio": divratio}
ys = {"B": B, "M": M, "U": U, "D": D, "AVG": AVG}

print("methods (sorted by realized):")
order = np.argsort(realized)
print(f"{'method':<12}{'realized':>9}{'feasible':>9}{'deployed':>9}{'div_ratio':>10}{'D':>7}{'AVG':>7}")
for i in order:
    print(f"{names[i]:<12}{realized[i]:>9.3f}{feasible[i]:>9.3f}{deployed[i]:>9.3f}{divratio[i]:>10.3f}{D[i]:>7.2f}{AVG[i]:>7.3f}")

print("\nPearson / Spearman  (coverage-metric  vs  generalization):")
label = "x_vs_y"
print(f"{label:<12}" + "".join(f"{y:>16}" for y in ys))
for xn, x in xs.items():
    cells = []
    for yn, y in ys.items():
        cells.append(f"{pearson(x,y):>+6.2f}/{spearman(x,y):>+5.2f}")
    print(f"{xn:<12}" + "".join(f"{c:>16}" for c in cells))
