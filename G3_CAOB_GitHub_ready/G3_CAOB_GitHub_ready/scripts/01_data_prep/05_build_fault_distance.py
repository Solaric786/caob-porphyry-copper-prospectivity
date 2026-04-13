import os
import numpy as np
import rasterio
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt
import geopandas as gpd
import matplotlib.pyplot as plt

GRID_PATH   = r"Outputs/grid1km_CAOB.tif"
DEM_PATH    = r"New Data/G1_DEM_1km_correct.tif"
CAOB_PATH   = r"Outputs/CAOB_window.geojson"
FAULTS_PATH = r"Data/gem-global-active-faults-master/gem-global-active-faults-master/shapefile/gem_active_faults.shp"

OUT_TIF_G3  = r"Outputs/G3_DIST_FAULT_1km.tif"
OUT_PNG_G3  = r"Outputs/G3_DIST_FAULT_1km.png"
OUT_PNG_FAULTS = r"Outputs/faults_CAOB_only.png"

os.makedirs("Outputs", exist_ok=True)

# -----------------------------
# Grid + reference CRS
# -----------------------------
with rasterio.open(GRID_PATH) as grid:
    grid_profile = grid.profile
    dst_height, dst_width = grid.height, grid.width
    dst_transform = grid.transform
    grid_bounds = grid.bounds

print("Grid1km_CAOB:")
print("  shape   :", dst_height, "x", dst_width)
print("  bounds  :", grid_bounds)
print("  pix_deg :", dst_transform.a, -dst_transform.e)

with rasterio.open(DEM_PATH) as dem:
    ref_crs = dem.crs  # WGS84
print("Reference CRS from DEM:", ref_crs)

# approximate km per degree and km per grid cell
km_per_deg = 111.32
cell_size_deg = abs(dst_transform.a)
km_per_cell = km_per_deg * cell_size_deg
print("Approx km per cell:", km_per_cell)

# -----------------------------
# Load CAOB window
# -----------------------------
caob = gpd.read_file(CAOB_PATH)
print("CAOB_window CRS:", caob.crs)

# -----------------------------
# Load faults and set CRS = WGS84
# -----------------------------
faults = gpd.read_file(FAULTS_PATH)
print("Faults CRS before:", faults.crs)

if faults.crs is None:
    faults.set_crs("EPSG:4326", inplace=True)
print("Faults CRS after :", faults.crs)

# -----------------------------
# Clip faults to CAOB window
# -----------------------------
faults_clip = gpd.clip(faults, caob)
print("Number of fault segments in CAOB:", len(faults_clip))
if faults_clip.empty:
    raise RuntimeError("No faults inside CAOB_window – check paths/CRS.")

# --------- NEW: plot ONLY clipped faults in CAOB ---------
fig, ax = plt.subplots(figsize=(6, 4))
faults_clip.plot(ax=ax, linewidth=0.3, color="black")
caob.boundary.plot(ax=ax, edgecolor="red", linewidth=1.0)

# zoom exactly to CAOB box
minx, miny, maxx, maxy = caob.total_bounds
ax.set_xlim(minx, maxx)
ax.set_ylim(miny, maxy)

ax.set_title("GEM active faults inside CAOB window")
ax.set_xlabel("Longitude (°)")
ax.set_ylabel("Latitude (°)")
plt.tight_layout()
plt.savefig(OUT_PNG_FAULTS, dpi=200)
plt.close()
print(f"Saved CAOB faults preview to {OUT_PNG_FAULTS}")
# ----------------------------------------------------------

# -----------------------------
# Rasterize faults on 1 km grid (1 = fault, 0 = no fault)
# -----------------------------
fault_shapes = [
    (geom, 1)
    for geom in faults_clip.geometry
    if geom is not None and not geom.is_empty
]

fault_raster = rasterize(
    fault_shapes,
    out_shape=(dst_height, dst_width),
    transform=dst_transform,
    fill=0,
    all_touched=True,
    dtype="uint8",
)

# -----------------------------
# Distance transform (pixels -> km)
# -----------------------------
no_fault_mask = fault_raster == 0
dist_pixels = distance_transform_edt(no_fault_mask)
dist_km = (dist_pixels * km_per_cell).astype("float32")

# -----------------------------
# Save as GeoTIFF with CRS = WGS84
# -----------------------------
profile = grid_profile.copy()
profile.update(
    dtype="float32",
    count=1,
    nodata=-9999.0,
    compress="lzw",
    crs=ref_crs,
)

with rasterio.open(OUT_TIF_G3, "w", **profile) as dst:
    dst.write(dist_km, 1)

print(f"\nSaved fault-distance raster to {OUT_TIF_G3}")

# -----------------------------
# PNG preview of distance raster
# -----------------------------
plt.figure(figsize=(6, 3))
im = plt.imshow(
    dist_km,
    origin="upper",
    extent=(
        dst_transform.c,
        dst_transform.c + dst_transform.a * dst_width,
        dst_transform.f + dst_transform.e * dst_height,
        dst_transform.f,
    ),
)
plt.colorbar(im, label="Distance to nearest fault (km)")
plt.title("G3_DIST_FAULT_1km (km)")
plt.xlabel("Longitude (°)")
plt.ylabel("Latitude (°)")
plt.tight_layout()
plt.savefig(OUT_PNG_G3, dpi=200)
plt.close()
print(f"Saved G3 preview to {OUT_PNG_G3}")
