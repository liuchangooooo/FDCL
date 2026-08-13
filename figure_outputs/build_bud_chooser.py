import pathlib, collections
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

_CJK = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
fm.fontManager.addfont(_CJK)
plt.rcParams["font.sans-serif"] = [fm.FontProperties(fname=_CJK).get_name(), "DejaVu Sans"]

RAW = pathlib.Path("/home/hnu-w/DIVO/figure_outputs/bud/raw")

for fam in ["B", "U", "D"]:
    files = sorted(RAW.glob(f"{fam}__*.png"))
    # group by video name
    byvid = collections.OrderedDict()
    for f in files:
        _, name, idx = f.stem.split("__")
        byvid.setdefault(name, []).append((int(idx), f))
    vids = list(byvid.keys())
    ncol = 6
    nrow = len(vids)
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2.2, nrow * 2.0))
    if nrow == 1:
        axes = axes.reshape(1, -1)
    for r, name in enumerate(vids):
        items = sorted(byvid[name])
        for c in range(ncol):
            ax = axes[r][c]
            ax.set_xticks([]); ax.set_yticks([])
            if c < len(items):
                idx, fpath = items[c]
                ax.imshow(mpimg.imread(fpath))
                ax.set_title(f"f{idx}", fontsize=10, fontweight="bold", color="#c0392b")
                for s in ax.spines.values():
                    s.set_edgecolor("#bbb")
            else:
                ax.axis("off")
            if c == 0:
                ax.set_ylabel(name, fontsize=10, fontweight="bold", rotation=0,
                              labelpad=34, va="center")
    fig.subplots_adjust(left=0.12, right=0.99, top=0.93, bottom=0.01, wspace=0.05, hspace=0.2)
    out = f"/home/hnu-w/DIVO/figure_outputs/fig_chooser_{fam}.png"
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", out)
