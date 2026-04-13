import os
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
import matplotlib.pyplot as plt

# -----------------------------
# Paths
# -----------------------------
GRID_PATH   = r"Outputs/grid1km_CAOB.tif"
DEM_PATH    = r"New Data/G1_DEM_1km_correct.tif"
GLIM_PATH   = r"Data/hartmann-moosdorf_2012(1)/glim_wgs84_0point5deg.txt.asc"

OUT_TIF_G2  = r"Outputs/G2_LITHOLOGY8_1km.tif"
OUT_PNG_G2  = r"Outputs/G2_LITHOLOGY8_1km.png"

os.makedirs("Outputs", exist_ok=True)

# -----------------------------
# Read 1 km grid and reference CRS (WGS84) from DEM
# -----------------------------
with rasterio.open(GRID_PATH) as grd:
    dst_height = grd.height
    dst_width  = grd.width
    dst_transform = grd.transform
    grid_profile = grd.profile
    grid_bounds = grd.bounds

print("Grid1km_CAOB:")
print("  shape   :", dst_height, "x", dst_width)
print("  bounds  :", grid_bounds)
print("  pix_deg :", dst_transform.a, -dst_transform.e)

with rasterio.open(DEM_PATH) as dem:
    ref_crs = dem.crs  # WGS84
print("Reference CRS from DEM:", ref_crs)

# -----------------------------
# Read GLiM global raster (0.5°), treat as WGS84
# -----------------------------
with rasterio.open(GLIM_PATH) as src:
    glim_raw       = src.read(1)
    glim_transform = src.transform
    glim_nodata    = src.nodata  # usually -9999
    print("\nGLiM input:")
    print("  CRS    :", src.crs, "(we will treat as WGS84)")
    print("  bounds :", src.bounds)
    print("  res    :", src.res)
    print("  nodata :", glim_nodata)

# -----------------------------
# Reproject / resample GLiM to 1 km grid (nearest neighbour)
# -----------------------------
dst_glim_codes = np.zeros((dst_height, dst_width), dtype=np.int16)

reproject(
    source=glim_raw,
    destination=dst_glim_codes,
    src_transform=glim_transform,
    src_crs=ref_crs,          # treat GLiM as WGS84
    dst_transform=dst_transform,
    dst_crs=ref_crs,
    resampling=Resampling.nearest,
    src_nodata=glim_nodata,
    dst_nodata=0,
)

# -----------------------------
# Reclassify GLiM 16 codes -> 8 lithology groups
# -----------------------------
reclass_map = {
    1: 1,   # su
    2: 2,   # vb
    3: 3,   # ss
    4: 4,   # pb
    5: 5,
    6: 5,
    7: 6,
    8: 6,
    9: 7,
    10: 7,
    11: 8,
    12: 8,
    13: 8,
    14: 8,
    15: 8,
    16: 8,
}

dst_litho = np.zeros_like(dst_glim_codes, dtype=np.uint8)
unique_codes = np.unique(dst_glim_codes)
print("\nUnique GLiM codes on CAOB grid:", unique_codes)

for code in unique_codes:
    if code in reclass_map:
        dst_litho[dst_glim_codes == code] = reclass_map[code]
    elif code == 0:
        # background / nodata already 0
        continue
    else:
        print("WARNING: unmapped GLiM code:", code)

# -----------------------------
# Save G2 as GeoTIFF with CRS = WGS84
# -----------------------------
profile = grid_profile.copy()
profile.update(
    dtype="uint8",
    count=1,
    nodata=0,
    compress="lzw",
    crs=ref_crs,          # write WGS84 into file
)

with rasterio.open(OUT_TIF_G2, "w", **profile) as dst:
    dst.write(dst_litho, 1)

print(f"\nSaved lithology (8 groups) to {OUT_TIF_G2}")

# -----------------------------
# Quick PNG preview
# -----------------------------
plt.figure(figsize=(6, 3))
im = plt.imshow(
    dst_litho,
    origin="upper",
    extent=(
        dst_transform.c,
        dst_transform.c + dst_transform.a * dst_width,
        dst_transform.f + dst_transform.e * dst_height,
        dst_transform.f,
    ),
    interpolation="nearest",
)
plt.colorbar(im, label="Lithology group (1–8)")
plt.title("G2_LITHOLOGY8_1km")
plt.xlabel("Longitude (°)")
plt.ylabel("Latitude (°)")
plt.tight_layout()
plt.savefig(OUT_PNG_G2, dpi=200)
plt.close()
print(f"Saved G2 preview to {OUT_PNG_G2}")
