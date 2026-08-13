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

RAW = pathlib.Path("/home/hnu-w/DIVO/figure_outputs/m41/raw")
ENDD = pathlib.Path("/home/hnu-w/DIVO/figure_outputs/m41/end")

# ---- (1) main 4-frame M panel ----
frames = [0, 134, 178, 199]
col = "#3D7FC4"
FW, FH, GAP, PAD = 4.0, 3.0, 0.10, 0.28
NCOL = len(frames)
row_w = NCOL * FW + (NCOL - 1) * GAP
total_w = row_w + 2 * PAD
total_h = FH + 2 * PAD + 1.0
fig, ax = plt.subplots(figsize=(10.5, 10.5 * total_h / total_w))
ax.set_xlim(0, total_w); ax.set_ylim(0, total_h); ax.axis("off"); ax.set_aspect("equal")
y0 = 0.25
ax.add_patch(FancyBboxPatch((0, y0), total_w, FH + 2 * PAD,
             boxstyle="round,pad=0,rounding_size=0.38", linewidth=3.0,
             edgecolor=col, facecolor="white", zorder=2))
for c, fi in enumerate(frames):
    fx = PAD + c * (FW + GAP)
    ax.imshow(mpimg.imread(RAW / f"f{fi}.png"),
              extent=[fx, fx + FW, y0 + PAD, y0 + PAD + FH], aspect="auto", zorder=3)
ax.add_patch(FancyBboxPatch((0.25, y0 + FH + 2 * PAD - 0.42), 4.7, 0.95,
             boxstyle="round,pad=0,rounding_size=0.28", linewidth=0, facecolor=col, zorder=5))
ax.text(0.25 + 2.35, y0 + FH + 2 * PAD + 0.05, "Multiple (M)  多障碍",
        ha="center", va="center", color="white", fontsize=13, fontweight="bold", zorder=6)
ax.annotate("", xy=(total_w - PAD, y0 - 0.05), xytext=(PAD, y0 - 0.05),
            arrowprops=dict(arrowstyle="-|>", color="#8A93A0", lw=2.0), annotation_clip=False)
ax.text(total_w / 2, y0 - 0.5, "初始状态  →  到达目标", ha="center", va="top",
        fontsize=11, color="#6B7785", fontweight="bold")
fig.savefig("/home/hnu-w/DIVO/figure_outputs/fig_m_panel.png", dpi=220, bbox_inches="tight", facecolor="white")
plt.close(fig)

# ---- (2) end-frame chooser ----
ends = [192, 195, 197, 199, 200]
fig2, axes = plt.subplots(1, len(ends), figsize=(len(ends) * 2.2, 2.4))
for ax, fi in zip(axes, ends):
    ax.imshow(mpimg.imread(ENDD / f"e{fi}.png"))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"frame {fi}", fontsize=11, fontweight="bold", color="#c0392b")
fig2.savefig("/home/hnu-w/DIVO/figure_outputs/fig_m41_endframes.png", dpi=180, bbox_inches="tight", facecolor="white")
print("saved fig_m_panel.png and fig_m41_endframes.png")
