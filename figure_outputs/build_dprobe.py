import pathlib, collections, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

RAW = pathlib.Path("/home/hnu-w/DIVO/figure_outputs/dprobe/raw")
OUT = pathlib.Path("/home/hnu-w/DIVO/figure_outputs/fig_dprobe.png")
byv = collections.OrderedDict()
for f in sorted(RAW.glob("*.png")):
    tag, c, fi = f.stem.rsplit("_", 2)
    byv.setdefault(tag, []).append((int(c), int(fi), f))
rows = list(byv.items())
ncol = 8
fig, axes = plt.subplots(len(rows), ncol, figsize=(ncol * 1.7, len(rows) * 1.7))
for r, (tag, items) in enumerate(rows):
    items = sorted(items)
    for c in range(ncol):
        ax = axes[r][c]
        ax.set_xticks([]); ax.set_yticks([])
        _, fi, fp = items[c]
        ax.imshow(mpimg.imread(fp))
        ax.set_title(f"f{fi}", fontsize=9, color="#c0392b")
        for s in ax.spines.values():
            s.set_edgecolor("#bbb")
        if c == 0:
            ax.set_ylabel(tag, fontsize=9, rotation=0, labelpad=34, va="center", fontweight="bold")
fig.subplots_adjust(left=0.10, right=0.99, top=0.93, bottom=0.01, wspace=0.05, hspace=0.2)
fig.savefig(OUT, dpi=170, bbox_inches="tight", facecolor="white")
print("saved", OUT)
