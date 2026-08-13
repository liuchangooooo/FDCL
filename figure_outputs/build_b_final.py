import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import FancyBboxPatch

RAW = pathlib.Path("/home/hnu-w/DIVO/figure_outputs/b27/raw")
OUT = pathlib.Path("/home/hnu-w/DIVO/figure_outputs/fig_b_panel.png")
frames = ["f000", "f174", "f232", "f464"]
col = "#4F9D5B"  # B = green

FW, FH, GAP, PAD = 4.0, 3.0, 0.10, 0.28
NCOL = len(frames)
total_w = NCOL * FW + (NCOL - 1) * GAP + 2 * PAD
total_h = FH + 2 * PAD
fig, ax = plt.subplots(figsize=(10.5, 10.5 * total_h / total_w))
ax.set_xlim(0, total_w); ax.set_ylim(0, total_h); ax.axis("off"); ax.set_aspect("equal")
ax.add_patch(FancyBboxPatch((0, 0), total_w, FH + 2 * PAD,
             boxstyle="round,pad=0,rounding_size=0.38", linewidth=3.0,
             edgecolor=col, facecolor="white", zorder=2))
for c, fn in enumerate(frames):
    fx = PAD + c * (FW + GAP)
    ax.imshow(mpimg.imread(RAW / f"{fn}.png"),
              extent=[fx, fx + FW, PAD, PAD + FH], aspect="auto", zorder=3)
fig.savefig(OUT, dpi=240, bbox_inches="tight", facecolor="white")
print("saved", OUT)
