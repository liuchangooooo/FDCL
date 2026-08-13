"""Preview-only: tile the three real overlays on white so they can be eyeballed.
This is NOT the teaser; the teaser is assembled by hand in a vector tool."""
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

HERE = pathlib.Path(__file__).resolve().parent
items = [("asset_A_deployed.png", "A: deployed w0 (1 rollout)"),
         ("asset_B_skillfan.png", "B: K=6 skill fan (3 solid / 3 dashed)"),
         ("asset_C_learnability.png", "C: lv = p(1-p), real point p=0.5")]
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, (fn, title) in zip(axes, items):
    ax.imshow(mpimg.imread(HERE / fn))
    ax.set_title(title, fontsize=10)
    ax.axis("off")
fig.patch.set_facecolor("white")
fig.tight_layout()
fig.savefig(HERE / "_contact_sheet.png", dpi=150, facecolor="white")
print("wrote", HERE / "_contact_sheet.png")
