import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

for fam in ["B", "U", "D"]:
    RAW = pathlib.Path(f"/home/hnu-w/DIVO/figure_outputs/idx_{fam}/raw")
    OUT = pathlib.Path(f"/home/hnu-w/DIVO/figure_outputs/fig_index_{fam}.png")
    files = sorted(RAW.glob("*.png"))
    n = len(files); ncol = 6; nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2.1, nrow * 1.75))
    axes = axes.ravel()
    for k, ax in enumerate(axes):
        ax.set_xticks([]); ax.set_yticks([])
        if k < n:
            ax.imshow(mpimg.imread(files[k]))
            ax.set_title(files[k].stem, fontsize=11, fontweight="bold", color="#c0392b", pad=2)
            for s in ax.spines.values():
                s.set_edgecolor("#bbb")
        else:
            ax.axis("off")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.01, wspace=0.05, hspace=0.18)
    fig.savefig(OUT, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", OUT, "n=", n)
