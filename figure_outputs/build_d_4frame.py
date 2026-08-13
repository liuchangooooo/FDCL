"""D as 4-frame strip; on the frames where obstacles re-randomize, mark each
obstacle's PREVIOUS position with a dashed red box and draw a bold red arrow to
its new position (nearest matching). No text label."""
import pathlib, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image

DD = pathlib.Path("/home/hnu-w/DIVO/figure_outputs/dblob_u82")
OUT = pathlib.Path("/home/hnu-w/DIVO/figure_outputs/fig_d_panel.png")

# obstacle configs (per-blob centroids, pixel coords) from detection
A = [(153, 202), (168, 55), (242, 178)]   # f95-190
B = [(168, 180), (242, 103), (244, 176)]  # f230-284
C = [(123, 80), (200, 99), (203, 146)]    # f400-474

def nearest_pairs(old, new):
    new = list(new); pairs = []
    for o in old:
        j = min(range(len(new)), key=lambda k: (o[0]-new[k][0])**2 + (o[1]-new[k][1])**2)
        pairs.append((o, new[j])); new.pop(j)
    return pairs

# frame: (image, old_config_or_None, new_config_or_None)
frames = [
    ("f95.png", None, None),
    ("f284.png", A, B),
    ("f442.png", None, None),
    ("f474.png", None, None),
]
col = "#E07B39"
RED = "#E81E1E"
HALF = 17  # obstacle half-size in px for dashed box
MIN_DISP = 40  # only annotate obstacles that moved at least this many px

FW, FH, GAP, PAD = 4.0, 3.0, 0.10, 0.28
NCOL = 4
total_w = NCOL * FW + (NCOL - 1) * GAP + 2 * PAD
total_h = FH + 2 * PAD
fig, ax = plt.subplots(figsize=(11.5, 11.5 * total_h / total_w))
ax.set_xlim(0, total_w); ax.set_ylim(0, total_h); ax.axis("off"); ax.set_aspect("equal")
ax.add_patch(FancyBboxPatch((0, 0), total_w, total_h,
             boxstyle="round,pad=0,rounding_size=0.38", linewidth=3.0,
             edgecolor=col, facecolor="white", zorder=2))

for c, (imgf, oldc, newc) in enumerate(frames):
    im = np.asarray(Image.open(DD / imgf))
    H, W = im.shape[:2]
    fx = PAD + c * (FW + GAP)
    ax.imshow(im, extent=[fx, fx + FW, PAD, PAD + FH], aspect="auto", zorder=3)

    def tox(px, py):
        return fx + (px / W) * FW, PAD + (1 - py / H) * FH

    if oldc is not None:
        for (o, n) in nearest_pairs(oldc, newc):
            disp = ((o[0] - n[0])**2 + (o[1] - n[1])**2) ** 0.5
            # dashed box at old position for EVERY obstacle
            bx, by = tox(o[0] - HALF, o[1] + HALF)  # lower-left in axes
            w = (2 * HALF / W) * FW; h = (2 * HALF / H) * FH
            ax.add_patch(Rectangle((bx, by), w, h, fill=False, edgecolor=RED,
                         linewidth=2.0, linestyle=(0, (3, 2)), zorder=6))
            # arrow only when the obstacle actually moved a visible amount
            if disp >= 12:
                a0 = tox(*o); a1 = tox(*n)
                ax.add_patch(FancyArrowPatch(a0, a1, connectionstyle="arc3,rad=0.12",
                             arrowstyle="-|>", mutation_scale=20, linewidth=3.2,
                             color=RED, shrinkA=2, shrinkB=2, zorder=7))

fig.savefig(OUT, dpi=240, bbox_inches="tight", facecolor="white")
print("saved", OUT)
