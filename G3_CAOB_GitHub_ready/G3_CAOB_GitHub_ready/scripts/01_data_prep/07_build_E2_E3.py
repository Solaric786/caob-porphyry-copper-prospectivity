import os
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
import matplotlib.pyplot as plt

# -----------------------------
# Paths
# -----------------------------
GRID_PATH = r"Outputs/grid1km_CAOB.tif"
DEM_PATH  = r"New Data/G1_DEM_1km_correct.tif"
NEW_DIR   = r"New Data"
OUT_DIR   = r"Outputs"

os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------
# Read 1 km grid + reference CRS
# -----------------------------
with rasterio.open(GRID_PATH) as grd:
    grid_profile = grd.profile
    dst_height   = grd.height
    dst_width    = grd.width
    dst_transform = grd.transform
    print("Target grid (grid1km_CAOB):")
    print("  shape :", dst_height, "x", dst_width)
    print("  bounds:", grd.bounds)
    print("  res   :", grd.res)

with rasterio.open(DEM_PATH) as dem:
    ref_crs = dem.crs  # WGS84 from DEM
print("Reference CRS (DEM):", ref_crs)
print()

# (input_name, output_tif, output_png, colorbar_label)
inputs = [
    (
        "E2_POP_1km.tif",
        "E2_POP_1km_CAOB.tif",
        "E2_POP_1km_CAOB.png",
        "Built-up index",
    ),
    (
        "E3_VIIRS_2018_1km_new.tif",
        "E3_VIIRS_2018_1km_CAOB.tif",
        "E3_VIIRS_2018_1km_CAOB.png",
        "Night-time lights",
    ),
]

for in_name, out_tif_name, out_png_name, cb_label in inputs:
    src_path = os.path.join(NEW_DIR, in_name)
    if not os.path.exists(src_path):
        print(f"WARNING: {src_path} not found, skipping.")
        continue

    with rasterio.open(src_path) as src:
        # use band 1 (these are single-band rasters for modelling)
        src_data      = src.read(1)
        src_transform = src.transform
        src_nodata    = src.nodata
        src_crs       = src.crs

        print(f"Resampling {in_name}:")
        print("  CRS   :", src_crs)
        print("  shape :", src.height, "x", src.width)
        print("  bounds:", src.bounds)

        # allocate destination array on 1 km grid
        dst_data = np.full(
            (dst_height, dst_width),
            src_nodata if src_nodata is not None else 0,
            dtype=src_data.dtype,
        )

        # reproject/resample to grid1km_CAOB
        reproject(
            source=src_data,
            destination=dst_data,
            src_transform=src_transform,
            src_crs=src_crs or ref_crs,   # treat as WGS84 if missing
            dst_transform=dst_transform,
            dst_crs=ref_crs,
            src_nodata=src_nodata,
            dst_nodata=src_nodata,
            resampling=Resampling.bilinear,
        )

    # -------------------------
    # Save aligned GeoTIFF
    # -------------------------
    out_tif = os.path.join(OUT_DIR, out_tif_name)

    # safely remove any existing tif / ovr (ArcMap locks old .ovr otherwise)
    for extra in ["", ".ovr"]:
        try:
            os.remove(out_tif + extra)
        except FileNotFoundError:
            pass

    profile = grid_profile.copy()
    profile.update(
        height=dst_height,
        width=dst_width,
        transform=dst_transform,
        dtype=dst_data.dtype,
        count=1,
        nodata=src_nodata,
        compress="lzw",
        crs=ref_crs,   # write WGS84 into file
    )

    with rasterio.open(out_tif, "w", **profile) as dst:
        dst.write(dst_data, 1)
    print("  saved TIFF:", out_tif)

    # -------------------------
    # PNG preview (with stretched colors)
    # -------------------------
    extent = (
        dst_transform.c,
        dst_transform.c + dst_transform.a * dst_width,
        dst_transform.f + dst_transform.e * dst_height,
        dst_transform.f,
    )

    # ignore NoData in statistics
    data_for_stats = dst_data.astype(float)
    if src_nodata is not None:
        data_for_stats[data_for_stats == src_nodata] = np.nan

    # handle case where everything is NaN
    if np.all(np.isnan(data_for_stats)):
        vmin, vmax = None, None
    else:
        vmin, vmax = np.nanpercentile(data_for_stats, [2, 98])

    plt.figure(figsize=(6, 3))
    im = plt.imshow(
        dst_data,
        origin="upper",
        extent=extent,
        vmin=vmin,
        vmax=vmax,
    )
    plt.colorbar(im, label=cb_label)
    plt.title(out_tif_name.replace(".tif", ""))
    plt.xlabel("Longitude (°)")
    plt.ylabel("Latitude (°)")
    plt.tight_layout()

    out_png = os.path.join(OUT_DIR, out_png_name)
    plt.savefig(out_png, dpi=200)
    plt.close()
    print("  saved PNG :", out_png)
    print()

print("E2–E3 resampling finished.")
