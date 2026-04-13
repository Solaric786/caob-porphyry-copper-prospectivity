import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.transform import rowcol
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

# ---------------- Paths ----------------
GRID_PATH  = r"Outputs/grid1km_CAOB.tif"
DEM_PATH   = r"New Data/G1_DEM_1km_correct.tif"
CAOB_PATH  = r"Outputs/CAOB_window.geojson"
MRDS_PATH  = r"New Data/mrds-csv/mrds.csv"   # <-- your file

OUT_TIF_E4 = r"Outputs/E4_KDE_ALLMIN_1km.tif"
OUT_PNG_E4 = r"Outputs/E4_KDE_ALLMIN_1km.png"

os.makedirs("Outputs", exist_ok=True)

# 1) Grid + CRS
with rasterio.open(GRID_PATH) as grd:
    H, W = grd.height, grd.width
    transform = grd.transform
    grid_profile = grd.profile
    bounds = grd.bounds

with rasterio.open(DEM_PATH) as dem:
    ref_crs = dem.crs

print("Grid shape:", H, "x", W)
print("Grid bounds:", bounds)
print("Reference CRS:", ref_crs)

# 2) CAOB window
caob = gpd.read_file(CAOB_PATH).to_crs(ref_crs)

# 3) MRDS → GeoDataFrame, clip to CAOB
df = pd.read_csv(MRDS_PATH, low_memory=False)

# MRDS columns are lowercase: 'longitude', 'latitude'
lon = df["longitude"]
lat = df["latitude"]

mask_valid = lon.notna() & lat.notna()
df_valid = df.loc[mask_valid].copy()

gdf = gpd.GeoDataFrame(
    df_valid,
    geometry=gpd.points_from_xy(df_valid["longitude"], df_valid["latitude"]),
    crs="EPSG:4326"
).to_crs(ref_crs)

gdf_caob = gpd.clip(gdf, caob)
print("MRDS points in CAOB:", len(gdf_caob))

# quick check plot
ax = caob.boundary.plot(figsize=(4, 3), color="red")
gdf_caob.plot(ax=ax, markersize=1, alpha=0.5)
ax.set_title("MRDS points inside CAOB")
plt.tight_layout()
plt.savefig("Outputs/E4_MRDS_points_CAOB.png", dpi=200)
plt.close()

# 4) Bin to 1-km grid
kde_grid = np.zeros((H, W), dtype=np.float32)

xs = gdf_caob.geometry.x.values
ys = gdf_caob.geometry.y.values
rows, cols = rowcol(transform, xs, ys)
inside = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)

for r, c in zip(rows[inside], cols[inside]):
    kde_grid[r, c] += 1.0

print("Total MRDS counts on grid:", kde_grid.sum())

# 5) Gaussian smoothing ≈ KDE
deg_per_cell = abs(transform.a)
km_per_deg = 111.32
km_per_cell = km_per_deg * deg_per_cell
bandwidth_km = 50.0
sigma_cells = bandwidth_km / km_per_cell
print("Gaussian sigma (cells):", sigma_cells)

kde_smooth = gaussian_filter(kde_grid, sigma=sigma_cells)
kde_norm = kde_smooth / kde_smooth.max()

# 6) Save GeoTIFF
profile = grid_profile.copy()
profile.update(
    dtype="float32",
    count=1,
    nodata=-9999.0,
    compress="lzw",
    crs=ref_crs,
)

with rasterio.open(OUT_TIF_E4, "w", **profile) as dst:
    dst.write(kde_norm.astype("float32"), 1)

print("Saved E4 KDE raster:", OUT_TIF_E4)

# 7) PNG preview
plt.figure(figsize=(6, 3))
extent = (
    transform.c,
    transform.c + transform.a * W,
    transform.f + transform.e * H,
    transform.f,
)
im = plt.imshow(kde_norm, origin="upper", extent=extent)
plt.colorbar(im, label="All-mineral KDE (0–1)")
plt.title("E4_KDE_ALLMIN_1km")
plt.xlabel("Longitude (°)")
plt.ylabel("Latitude (°)")
plt.tight_layout()
plt.savefig(OUT_PNG_E4, dpi=200)
plt.close()
print("Saved E4 preview:", OUT_PNG_E4)
