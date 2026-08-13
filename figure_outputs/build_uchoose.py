import pathlib, collections
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

RAW = pathlib.Path("/home/hnu-w/DIVO/figure_outputs/uchoose/raw")
byv = collections.OrderedDict()
for f in sorted(RAW.glob("*.png")):
    tag, idx = f.stem.split("_")
    byv.setdefault(tag, []).append((int(idx), f))

for tag, items in byv.items():
    items = sorted(items)
    ncol = 5; nrow = 2
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2.3, nrow * 2.0))
    axes = axes.ravel()
    for k, ax in enumerate(axes):
        ax.set_xticks([]); ax.set_yticks([])
        if k < len(items):
            idx, fp = items[k]
            ax.imshow(mpimg.imread(fp))
            ax.set_title(f"frame {idx}", fontsize=11, fontweight="bold", color="#c0392b")
            for s in ax.spines.values():
                s.set_edgecolor("#bbb")
        else:
            ax.axis("off")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.01, wspace=0.05, hspace=0.18)
    out = f"/home/hnu-w/DIVO/figure_outputs/fig_uchoose_{tag}.png"
    fig.savefig(out, dpi=175, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", out)
