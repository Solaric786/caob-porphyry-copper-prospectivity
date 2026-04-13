import os
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

DATA_DIR = "Data"
NEW_DATA_DIR = "New Data"
OUTPUT_DIR = "Outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

grid_path = os.path.join(OUTPUT_DIR, "grid1km_CAOB.tif")

# --- read grid geometry (target) ---
with rasterio.open(grid_path) as grd:
    dst_height = grd.height
    dst_width = grd.width
    dst_transform = grd.transform
    print("Target grid (grid1km_CAOB):")
    print("  shape:", dst_height, "x", dst_width)
    print("  bounds:", grd.bounds)
    print("  pixel size:", dst_transform.a, -dst_transform.e)

# (src_name, folder, out_name)
files = [
    ("G1_DEM_1km_correct.tif",   NEW_DATA_DIR, "G1_DEM_1km_CAOB.tif"),
    ("G1_SLOPE_1km_correct.tif", NEW_DATA_DIR, "G1_SLOPE_1km_CAOB.tif"),
    ("G1_TRI_1km_new_correct.tif", NEW_DATA_DIR, "G1_TRI_1km_CAOB.tif"),  # <- change
]


for src_name, folder, out_name in files:
    src_path = os.path.join(folder, src_name)
    if not os.path.exists(src_path):
        print(f"WARNING: missing {src_path}, skipping.")
        continue

    with rasterio.open(src_path) as src:
        src_data = src.read(1)
        src_transform = src.transform
        src_nodata = src.nodata

        print(f"\nResampling {src_name}:")
        print("  original shape:", src.height, "x", src.width)
        print("  original bounds:", src.bounds)
        print("  original CRS:", src.crs)

        dst_data = np.full(
            (dst_height, dst_width),
            src_nodata if src_nodata is not None else 0,
            dtype=src_data.dtype,
        )

        reproject(
            source=src_data,
            destination=dst_data,
            src_transform=src_transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=src.crs,
            src_nodata=src_nodata,
            dst_nodata=src_nodata,
            resampling=Resampling.bilinear,
        )

        profile = src.profile
        profile.update({
            "height": dst_height,
            "width": dst_width,
            "transform": dst_transform,
            "nodata": src_nodata,
        })
        profile.pop("crs", None)  # avoid your PROJ error

        out_path = os.path.join(OUTPUT_DIR, out_name)
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(dst_data, 1)

        print("  saved:", out_path)
