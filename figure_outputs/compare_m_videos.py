"""Contact sheet: 3 M-scene videos x 4 frames, for the user to pick the hardest."""
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

_CJK = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
fm.fontManager.addfont(_CJK)
plt.rcParams["font.sans-serif"] = [fm.FontProperties(fname=_CJK).get_name(), "DejaVu Sans"]

RAW = pathlib.Path("/home/hnu-w/DIVO/figure_outputs/m_compare/raw")
OUT = pathlib.Path("/home/hnu-w/DIVO/figure_outputs/fig_m_compare.png")
VIDS = ["i6exk8zv", "ek9l1kem", "2gw2f9yw"]
NCOL = 4

fig, axes = plt.subplots(len(VIDS), NCOL, figsize=(NCOL * 2.4, len(VIDS) * 1.95))
fig.subplots_adjust(left=0.10, right=0.99, top=0.95, bottom=0.02, wspace=0.04, hspace=0.10)
for r, v in enumerate(VIDS):
    for c in range(NCOL):
        ax = axes[r][c]
        ax.imshow(mpimg.imread(RAW / f"{v}_{c}.png"))
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor("#aaaaaa")
        if c == 0:
            ax.set_ylabel(v, fontsize=11, fontweight="bold", rotation=90, labelpad=8)
        if r == 0:
            ax.set_title(["初始", "t1", "t2", "末"][c], fontsize=10, color="#666")

fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
print("saved", OUT)
