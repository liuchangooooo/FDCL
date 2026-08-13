import pathlib, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

RAW = pathlib.Path("/home/hnu-w/DIVO/figure_outputs/bidx/raw")
OUT = pathlib.Path("/home/hnu-w/DIVO/figure_outputs/fig_bidx.png")
nf = {}
for line in open("/home/hnu-w/DIVO/figure_outputs/bidx/map.txt"):
    m = re.match(r"(\d+) NF=(\d+)", line)
    if m: nf[m.group(1)] = int(m.group(2))

files = sorted(RAW.glob("*.png"))
n = len(files); ncol = 4; nrow = (n + ncol - 1) // ncol
fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2.4, nrow * 2.0))
axes = axes.ravel()
for k, ax in enumerate(axes):
    ax.set_xticks([]); ax.set_yticks([])
    if k < n:
        st = files[k].stem
        ax.imshow(mpimg.imread(files[k]))
        ax.set_title(f"{st}  ({nf.get(st,'?')}帧)", fontsize=11, fontweight="bold", color="#c0392b")
        for s in ax.spines.values():
            s.set_edgecolor("#bbb")
    else:
        ax.axis("off")
fig.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.01, wspace=0.05, hspace=0.18)
fig.savefig(OUT, dpi=180, bbox_inches="tight", facecolor="white")
print("saved", OUT)
