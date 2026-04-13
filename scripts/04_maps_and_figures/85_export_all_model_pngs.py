import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

OUT_DIR = r"Outputs/paper_run_20260118"
os.makedirs(OUT_DIR, exist_ok=True)

# Add/adjust rasters here
RAS = [
    ("model6_xgb",  r"outputs/model6_xgb_prospectivity.tif"),
    ("hdipp",       os.path.join(OUT_DIR, "hdipp_prospectivity.tif")),
    ("model16_p_sup", os.path.join(OUT_DIR, "model16_p_sup_prospectivity.tif")),
    ("model16_p_y_pu_corrected", os.path.join(OUT_DIR, "model16_p_y_pu_corrected_prospectivity.tif")),
    ("model17_continuous", os.path.join(OUT_DIR, "model17_rankblend_CONTINUOUS.tif")),
]

DPI = 300
CMAP = "viridis"   # change if you want: "turbo", "inferno", etc.
VMIN, VMAX = 0.0, 1.0  # same scale for prospectivity maps

def save_png(name, tif_path):
    if not os.path.exists(tif_path):
        print("SKIP (missing):", tif_path)
        return

    with rasterio.open(tif_path) as ds:
        a = ds.read(1, masked=True).astype(np.float32)

    fig, ax = plt.subplots(figsize=(14, 7), constrained_layout=True)
    im = ax.imshow(a, origin="upper", cmap=CMAP, vmin=VMIN, vmax=VMAX)
    ax.set_title(name, fontsize=12)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # Colorbar exactly matched to axis height
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3.5%", pad=0.08)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Prospectivity score")

    out_png = os.path.join(OUT_DIR, f"{name}.png")
    fig.savefig(out_png, dpi=DPI)
    plt.close(fig)
    print("Saved:", out_png)

for name, tifp in RAS:
    save_png(name, tifp)
