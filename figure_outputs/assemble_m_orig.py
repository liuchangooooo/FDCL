"""Assemble the M-scene panel from ORIGINAL eval-video frames (proper rendering)."""
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import FancyBboxPatch

_CJK = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
fm.fontManager.addfont(_CJK)
plt.rcParams["font.sans-serif"] = [fm.FontProperties(fname=_CJK).get_name(), "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

RAW = pathlib.Path("/home/hnu-w/DIVO/figure_outputs/m_orig/raw")
OUT = pathlib.Path("/home/hnu-w/DIVO/figure_outputs/fig_m_panel.png")
NCOL = 6
col = "#3D7FC4"

FW, FH, GAP, PAD = 4.0, 3.0, 0.10, 0.28
row_w = NCOL * FW + (NCOL - 1) * GAP
total_w = row_w + 2 * PAD
total_h = FH + 2 * PAD + 1.0

fig, ax = plt.subplots(figsize=(13.0, 13.0 * total_h / total_w))
ax.set_xlim(0, total_w); ax.set_ylim(0, total_h); ax.axis("off"); ax.set_aspect("equal")

y0 = 0.25
ax.add_patch(FancyBboxPatch((0, y0), total_w, FH + 2 * PAD,
             boxstyle="round,pad=0,rounding_size=0.38",
             linewidth=3.0, edgecolor=col, facecolor="white", zorder=2))
for c in range(NCOL):
    fx = PAD + c * (FW + GAP)
    img = mpimg.imread(RAW / f"m_{c}.png")
    ax.imshow(img, extent=[fx, fx + FW, y0 + PAD, y0 + PAD + FH], aspect="auto", zorder=3)

ax.add_patch(FancyBboxPatch((0.25, y0 + FH + 2 * PAD - 0.42), 4.7, 0.95,
             boxstyle="round,pad=0,rounding_size=0.28", linewidth=0,
             facecolor=col, zorder=5))
ax.text(0.25 + 2.35, y0 + FH + 2 * PAD + 0.05, "Multiple (M)  多障碍",
        ha="center", va="center", color="white", fontsize=13, fontweight="bold", zorder=6)
ax.annotate("", xy=(total_w - PAD, y0 - 0.05), xytext=(PAD, y0 - 0.05),
            arrowprops=dict(arrowstyle="-|>", color="#8A93A0", lw=2.0), annotation_clip=False)
ax.text(total_w / 2, y0 - 0.5, "初始状态  →  到达目标", ha="center", va="top",
        fontsize=11, color="#6B7785", fontweight="bold")

fig.savefig(OUT, dpi=220, bbox_inches="tight", facecolor="white")
print("saved", OUT)
