import os, glob
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

FINAL_DIR = r"Outputs/paper_run_20260118/FINAL_NAMED"
DPI = 300

# Rainbow/anomaly style: blue(low) -> red(high)
CMAP = "turbo"      # modern rainbow (recommended)
# CMAP = "jet"      # classic rainbow (optional)

# Use fixed 0..1 scale for comparable prospectivity maps
FIXED_01 = True
VMIN, VMAX = 0.0, 1.0

tifs = sorted(glob.glob(os.path.join(FINAL_DIR, "*.tif")))
if not tifs:
    raise SystemExit(f"No .tif files found in: {FINAL_DIR}")

def export_png(tif_path):
    name = os.path.splitext(os.path.basename(tif_path))[0]
    out_png = os.path.join(FINAL_DIR, f"{name}.png")

    with rasterio.open(tif_path) as ds:
        a = ds.read(1, masked=True).astype(np.float32)

    # If fixed scaling but data is outside [0,1], fallback to auto
    vmin, vmax = (VMIN, VMAX)
    if FIXED_01:
        v = a.compressed()
        if v.size and (float(v.min()) < -1e-6 or float(v.max()) > 1.0 + 1e-6):
            vmin, vmax = float(v.min()), float(v.max())
    else:
        v = a.compressed()
        vmin, vmax = (float(v.min()), float(v.max())) if v.size else (0.0, 1.0)

    fig, ax = plt.subplots(figsize=(14, 7), constrained_layout=True)
    im = ax.imshow(a, origin="upper", cmap=CMAP, vmin=vmin, vmax=vmax)

    # Clean, publication-friendly look
    ax.set_title(name, fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])

    # Colorbar height exactly matches the map height
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3.5%", pad=0.08)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Prospectivity score", fontsize=10)

    fig.savefig(out_png, dpi=DPI)
    plt.close(fig)
    print("Saved:", out_png)

for tif in tifs:
    export_png(tif)

print("Done. PNGs are in:", FINAL_DIR)
