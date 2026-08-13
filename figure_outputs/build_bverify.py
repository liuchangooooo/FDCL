import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

_CJK = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
fm.fontManager.addfont(_CJK)
plt.rcParams["font.sans-serif"] = [fm.FontProperties(fname=_CJK).get_name(), "DejaVu Sans"]

RAW = pathlib.Path("/home/hnu-w/DIVO/figure_outputs/bverify/raw")
OUT = pathlib.Path("/home/hnu-w/DIVO/figure_outputs/fig_bverify.png")
rows = [("43", "ours-best430k 75帧"), ("02", "llm_evolve 135帧"),
        ("04", "llm_static 164帧"), ("27", "evolve-attr 696帧"),
        ("09", "manual 885帧"), ("10", "manual 885帧")]
fig, axes = plt.subplots(len(rows), 4, figsize=(4 * 2.3, len(rows) * 1.85))
fig.subplots_adjust(left=0.16, right=0.99, top=0.96, bottom=0.01, wspace=0.04, hspace=0.12)
for r, (k, lab) in enumerate(rows):
    for c in range(4):
        ax = axes[r][c]
        ax.imshow(mpimg.imread(RAW / f"{k}_{c}.png"))
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor("#bbb")
        if c == 0:
            ax.set_ylabel(f"{k}\n{lab}", fontsize=9.5, rotation=0, labelpad=40, va="center", fontweight="bold")
        if r == 0:
            ax.set_title(["初始", "1/3", "2/3", "末帧"][c], fontsize=10, color="#666")
fig.savefig(OUT, dpi=180, bbox_inches="tight", facecolor="white")
print("saved", OUT)
