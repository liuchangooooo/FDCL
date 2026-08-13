import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

_CJK = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
fm.fontManager.addfont(_CJK)
plt.rcParams["font.sans-serif"] = [fm.FontProperties(fname=_CJK).get_name(), "DejaVu Sans"]

RAW = pathlib.Path("/home/hnu-w/DIVO/figure_outputs/m24/raw")
files = sorted(RAW.glob("*.png"))
n = len(files)
ncol = 5
nrow = (n + ncol - 1) // ncol
fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2.3, nrow * 2.0))
axes = axes.ravel()
for k, ax in enumerate(axes):
    ax.set_xticks([]); ax.set_yticks([])
    if k < n:
        ax.imshow(mpimg.imread(files[k]))
        ax.set_title("frame " + files[k].stem[1:], fontsize=12, fontweight="bold", color="#c0392b")
        for s in ax.spines.values():
            s.set_edgecolor("#bbb")
    else:
        ax.axis("off")
fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.01, wspace=0.05, hspace=0.18)
fig.savefig("/home/hnu-w/DIVO/figure_outputs/fig_m24_chooser.png", dpi=180, bbox_inches="tight", facecolor="white")
print("saved fig_m24_chooser.png")
