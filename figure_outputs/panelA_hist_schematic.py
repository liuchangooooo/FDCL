"""Schematic 3-class realized histogram for the Fig.2 mechanism panel (3).

This is an ILLUSTRATIVE distribution (method overview), showing all three classes
present -- infeasible / boundary / mastery -- as the intended trained W_probe
would induce. The measured real distribution (panelA_signal/hist_realized) is
boundary-heavy with empty mastery on the available random-z checkpoint and is
better used in the results/decoupling figure, not here.
"""
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

OUT = pathlib.Path("/home/hnu-w/DIVO/figure_outputs/panelA_signal")
C_INF, C_BND, C_MAS = "#E0897B", "#F4C95D", "#93C1A4"
K = 6
# illustrative counts across realized = 0, 1/K .. 1 (unimodal, band-heavy, all 3 present)
counts = np.array([4, 6, 9, 11, 9, 6, 5])
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
ax.legend(handles=[Patch(facecolor=C_INF, label="infeasible"),
                   Patch(facecolor=C_BND, label="boundary"),
                   Patch(facecolor=C_MAS, label="mastery")],
          fontsize=8, frameon=False, loc="upper right")
fig.tight_layout()
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "hist_realized_schematic.svg", transparent=True, bbox_inches="tight")
fig.savefig(OUT / "hist_realized_schematic.png", dpi=300, transparent=True, bbox_inches="tight")
plt.close(fig)
print("wrote", OUT / "hist_realized_schematic.svg")
