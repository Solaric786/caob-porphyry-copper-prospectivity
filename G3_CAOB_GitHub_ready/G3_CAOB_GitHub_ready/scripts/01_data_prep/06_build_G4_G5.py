# 06_build_G4_G5.py
import os
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
import matplotlib.pyplot as plt

OUTPUT_DIR   = "Outputs"
GRID_PATH    = os.path.join(OUTPUT_DIR, "grid1km_CAOB.tif")
DEM_PATH     = os.path.join("New Data", "G1_DEM_1km_correct.tif")

G4_SRC_PATH  = os.path.join("New Data", "G4_ASTER_PHYLLIC_1km_new.tif")
G5_SRC_PATH  = os.path.join("New Data", "G5_L8_iron_1km.tif")

G4_OUT_TIF   = os.path.join(OUTPUT_DIR, "G4_ASTER_PHYLLIC_1km_CAOB.tif")
G5_OUT_TIF   = os.path.join(OUTPUT_DIR, "G5_L8_iron_1km_CAOB.tif")

G4_OUT_PNG   = os.path.join(OUTPUT_DIR, "G4_ASTER_PHYLLIC_1km_CAOB.png")
G5_OUT_PNG   = os.path.join(OUTPUT_DIR, "G5_L8_iron_1km_CAOB.png")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# Read target grid and reference CRS
# -----------------------------
with rasterio.open(GRID_PATH) as grd:
    dst_height = grd.height
    dst_width  = grd.width
    dst_transform = grd.transform
    grid_profile = grd.profile
    grid_bounds  = grd.bounds

print("Target grid (grid1km_CAOB):")
print("  shape :", dst_height, "x", dst_width)
print("  bounds:", grid_bounds)
print("  res   :", (dst_transform.a, -dst_transform.e))

with rasterio.open(DEM_PATH) as dem:
    ref_crs = dem.crs  # WGS84 from DEM
print("Reference CRS (DEM):", ref_crs)

def resample_to_grid(src_path, out_tif_path, title):
    """Resample one raster to the 1 km CAOB grid and save TIFF + PNG."""
    with rasterio.open(src_path) as src:
        src_data = src.read(1)
        src_transform = src.transform
        src_crs = src.crs
        src_nodata = src.nodata

        print(f"\nResampling {os.path.basename(src_path)}:")
        print("  CRS   :", src_crs)
        print("  shape :", src.height, "x", src.width)
        print("  bounds:", src.bounds)

        # destination array
        dst_data = np.full(
            (dst_height, dst_width),
            src_nodata if src_nodata is not None else 0,
            dtype=src_data.dtype
        )

        # reproject/resample onto the grid
        reproject(
            source=src_data,
            destination=dst_data,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=ref_crs,
            src_nodata=src_nodata,
            dst_nodata=src_nodata,
            resampling=Resampling.bilinear,
        )

        # write GeoTIFF with WGS84 CRS
        profile = grid_profile.copy()
        profile.update(
            dtype=dst_data.dtype,
            count=1,
            nodata=src_nodata if src_nodata is not None else 0,
            compress="lzw",
            crs=ref_crs,
            height=dst_height,
            width=dst_width,
            transform=dst_transform,
        )

        with rasterio.open(out_tif_path, "w", **profile) as dst:
            dst.write(dst_data, 1)

        print("  saved TIFF:", out_tif_path)

        # quick PNG preview
        plt.figure(figsize=(6, 3))
        im = plt.imshow(
            dst_data,
            origin="upper",
            extent=(
                dst_transform.c,
                dst_transform.c + dst_transform.a * dst_width,
                dst_transform.f + dst_transform.e * dst_height,
                dst_transform.f,
            )
        )
        plt.colorbar(im, label="Index value")
        plt.title(title)
        plt.xlabel("Longitude (°)")
        plt.ylabel("Latitude (°)")
        plt.tight_layout()
        png_path = G4_OUT_PNG if "G4" in title else G5_OUT_PNG
        plt.savefig(png_path, dpi=200)
        plt.close()
        print("  saved PNG :", png_path)


# Run for G4 and G5
resample_to_grid(G4_SRC_PATH, G4_OUT_TIF, "G4_ASTER_PHYLLIC_1km_CAOB")
resample_to_grid(G5_SRC_PATH, G5_OUT_TIF, "G5_L8_iron_1km_CAOB")

print("\nG4–G5 resampling finished.")
