import os
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt

GRID_PATH   = r"Outputs/grid1km_CAOB.tif"
DEM_PATH    = r"New Data/G1_DEM_1km_correct.tif"
CAOB_PATH   = r"Outputs/CAOB_window.geojson"
DEPOSITS_PATH = r"Outputs/deposits_por_Cu.geojson"   # adjust if needed

OUT_TIF_E5  = r"Outputs/E5_DIST_MAJOR_1km.tif"
OUT_PNG_E5  = r"Outputs/E5_DIST_MAJOR_1km.png"

os.makedirs("Outputs", exist_ok=True)

# ---- 1. Grid + CRS ----
with rasterio.open(GRID_PATH) as grd:
    H, W = grd.height, grd.width
    transform = grd.transform
    grid_profile = grd.profile

with rasterio.open(DEM_PATH) as dem:
    ref_crs = dem.crs

deg_per_cell = abs(transform.a)
km_per_deg = 111.32
km_per_cell = km_per_deg * deg_per_cell
print("km per cell:", km_per_cell)

# ---- 2. Load CAOB & deposits ----
caob = gpd.read_file(CAOB_PATH).to_crs(ref_crs)

dep = gpd.read_file(DEPOSITS_PATH)
if dep.crs is None:
    dep.set_crs("EPSG:4326", inplace=True)
dep = dep.to_crs(ref_crs)

# Optionally clip (if not already clipped)
dep_caob = gpd.overlay(dep, caob, how="intersection")
print("Total Cu deposits in CAOB:", len(dep_caob))

# ---- 3. Select "major" deposits ----
# *** IMPORTANT ***: adapt this filter to your actual column names.
# Example: use a column "SizeClass" with values 'major', 'world-class', etc.
if "SizeClass" in dep_caob.columns:
    major = dep_caob[dep_caob["SizeClass"].str.contains("major", case=False, na=False)]
else:
    # fallback: for now use ALL deposits as "major"
    print("WARNING: 'SizeClass' column not found – using ALL deposits for E5.")
    major = dep_caob.copy()

print("Number of major deposits used for E5:", len(major))

# Quick check plot
ax = caob.boundary.plot(figsize=(4, 3), color="black")
major.plot(ax=ax, color="red", markersize=5)
ax.set_title("Major deposits used for E5")
plt.tight_layout()
plt.savefig("Outputs/E5_major_deposits_CAOB.png", dpi=200)
plt.close()

# ---- 4. Rasterize major deposits (1 = deposit, 0 = no deposit) ----
shapes = [(geom, 1) for geom in major.geometry if geom is not None and not geom.is_empty]

depos_raster = rasterize(
    shapes,
    out_shape=(H, W),
    transform=transform,
    fill=0,
    all_touched=True,
    dtype="uint8",
)

# ---- 5. Distance transform (pixels → km) ----
no_dep_mask = depos_raster == 0
dist_pixels = distance_transform_edt(no_dep_mask)
dist_km = (dist_pixels * km_per_cell).astype("float32")

# ---- 6. Save GeoTIFF ----
profile = grid_profile.copy()
profile.update(
    dtype="float32",
    count=1,
    nodata=-9999.0,
    compress="lzw",
    crs=ref_crs,
)

with rasterio.open(OUT_TIF_E5, "w", **profile) as dst:
    dst.write(dist_km, 1)

print("Saved E5 raster:", OUT_TIF_E5)

# ---- 7. PNG preview ----
extent = (transform.c,
          transform.c + transform.a * W,
          transform.f + transform.e * H,
          transform.f)

plt.figure(figsize=(6, 3))
im = plt.imshow(dist_km, origin="upper", extent=extent)
plt.colorbar(im, label="Distance to major deposit (km)")
plt.title("E5_DIST_MAJOR_1km")
plt.xlabel("Longitude (°)")
plt.ylabel("Latitude (°)")
plt.tight_layout()
plt.savefig(OUT_PNG_E5, dpi=200)
plt.close()
print("Saved E5 preview:", OUT_PNG_E5)
