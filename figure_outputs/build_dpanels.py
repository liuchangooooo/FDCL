import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

_CJK = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
fm.fontManager.addfont(_CJK)
plt.rcParams["font.sans-serif"] = [fm.FontProperties(fname=_CJK).get_name(), "DejaVu Sans"]

RAW = pathlib.Path("/home/hnu-w/DIVO/figure_outputs/dfinal/raw")
col = "#E07B39"  # D = orange
FW, FH, GAP, PAD = 4.0, 3.0, 0.10, 0.28
BLOCK_FRAME = 1  # 2nd frame = blocking, annotate

for tag in ["u82", "1udf", "sijy"]:
    frames = [f"{tag}_{j}" for j in range(4)]
    NCOL = 4
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
        if c == BLOCK_FRAME:
            # arrow + label pointing to mid-path region of this frame
            cx = fx + FW * 0.52
            cy = PAD + FH * 0.50
            ax.annotate("动态障碍",
                        xy=(cx, cy), xytext=(fx + FW * 0.5, PAD + FH + 0.55),
                        ha="center", va="bottom", fontsize=12, fontweight="bold",
                        color=col,
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=2.4),
                        zorder=6, annotation_clip=False)
    fig.savefig(f"/home/hnu-w/DIVO/figure_outputs/fig_d_{tag}.png",
                dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved fig_d_" + tag + ".png")
