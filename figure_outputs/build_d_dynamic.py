"""Reference-style dynamic-obstacle representation: ghost (faded) past obstacle
positions + solid final + red curved arrows showing motion, in one frame."""
import pathlib, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

_CJK = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
fm.fontManager.addfont(_CJK)
plt.rcParams["font.sans-serif"] = [fm.FontProperties(fname=_CJK).get_name(), "DejaVu Sans"]

DD = pathlib.Path("/home/hnu-w/DIVO/figure_outputs/ddet")

def red_mask(a):
    R, G, B = a[..., 0].astype(int), a[..., 1].astype(int), a[..., 2].astype(int)
    return (R > 120) & (R - G > 50) & (R - B > 50)

# frames: ghosts (old positions) + base (final, solid)
ghosts = [("f190.png", 0.40), ("f253.png", 0.60)]
base_name = "f474.png"
# detected centroids (x,y)
path_pts = [(188, 145), (219, 152), (176, 109)]

base = np.asarray(Image.open(DD / base_name)).astype(float)
comp = base.copy()
for fn, alpha in ghosts:
    g = np.asarray(Image.open(DD / fn)).astype(float)
    m = red_mask(g)
    # slight dilation
    from scipy.ndimage import binary_dilation
    try:
        m = binary_dilation(m, iterations=2)
    except Exception:
        pass
    comp[m] = (1 - alpha) * comp[m] + alpha * g[m]
comp = np.clip(comp, 0, 255).astype(np.uint8)
H, W = comp.shape[:2]

FW, FH, PAD = 4.0, 3.0, 0.28
total_w = FW + 2 * PAD
total_h = FH + 2 * PAD
fig, ax = plt.subplots(figsize=(5.2, 5.2 * total_h / total_w))
ax.set_xlim(0, total_w); ax.set_ylim(0, total_h); ax.axis("off"); ax.set_aspect("equal")
col = "#E07B39"
ax.add_patch(FancyBboxPatch((0, 0), total_w, total_h,
             boxstyle="round,pad=0,rounding_size=0.38", linewidth=3.0,
             edgecolor=col, facecolor="white", zorder=2))
# image extent
x0, y0 = PAD, PAD
ax.imshow(comp, extent=[x0, x0 + FW, y0, y0 + FH], aspect="auto", zorder=3)

def to_axes(px, py):
    # image pixel (px,py) -> axes coords (y flipped)
    ax_x = x0 + (px / W) * FW
    ax_y = y0 + (1 - py / H) * FH
    return ax_x, ax_y

# red curved arrows along the obstacle motion path
for (p0, p1) in zip(path_pts[:-1], path_pts[1:]):
    a0 = to_axes(*p0); a1 = to_axes(*p1)
    ax.add_patch(FancyArrowPatch(a0, a1, connectionstyle="arc3,rad=0.35",
                 arrowstyle="-|>", mutation_scale=18, linewidth=2.6,
                 color="#E03020", zorder=6))
# label
lx, ly = to_axes(*path_pts[1])
ax.annotate("动态障碍", xy=(lx, ly), xytext=(x0 + FW * 0.5, y0 + FH + 0.45),
            ha="center", va="bottom", fontsize=12, fontweight="bold", color=col,
            arrowprops=dict(arrowstyle="-|>", color=col, lw=2.0),
            annotation_clip=False, zorder=7)

fig.savefig("/home/hnu-w/DIVO/figure_outputs/fig_d_dynamic_u82.png",
            dpi=240, bbox_inches="tight", facecolor="white")
print("saved fig_d_dynamic_u82.png")
