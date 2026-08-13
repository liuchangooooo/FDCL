"""7 candidate M scenes x 4 frames (init -> end) for the user to confirm
which one completes the task (block on yellow target at the last frame)."""
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

_CJK = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
fm.fontManager.addfont(_CJK)
plt.rcParams["font.sans-serif"] = [fm.FontProperties(fname=_CJK).get_name(), "DejaVu Sans"]

RAW = pathlib.Path("/home/hnu-w/DIVO/figure_outputs/m_cand/raw")
OUT = pathlib.Path("/home/hnu-w/DIVO/figure_outputs/fig_m_cand.png")
CANDS = ["00", "02", "03", "10", "22", "41", "24"]
METHOD = {"00": "evolve(ours)", "02": "evolve(ours)", "03": "static",
          "10": "manual", "22": "static", "41": "static", "24": "evolve-attr"}
NF = {"00": 77, "02": 68, "03": 80, "10": 149, "22": 201, "41": 201, "24": 538}

fig, axes = plt.subplots(len(CANDS), 4, figsize=(4 * 2.3, len(CANDS) * 1.85))
fig.subplots_adjust(left=0.13, right=0.99, top=0.96, bottom=0.01, wspace=0.04, hspace=0.12)
for r, n in enumerate(CANDS):
    for c in range(4):
        ax = axes[r][c]
        ax.imshow(mpimg.imread(RAW / f"{n}_{c}.png"))
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor("#bbb")
        if c == 0:
            ax.set_ylabel(f"{n}\n{METHOD[n]}\n{NF[n]}帧", fontsize=9.5,
                          fontweight="bold", rotation=0, labelpad=30, va="center")
        if r == 0:
            ax.set_title(["初始", "1/3", "2/3", "末帧"][c], fontsize=10, color="#666")

fig.savefig(OUT, dpi=180, bbox_inches="tight", facecolor="white")
print("saved", OUT)
