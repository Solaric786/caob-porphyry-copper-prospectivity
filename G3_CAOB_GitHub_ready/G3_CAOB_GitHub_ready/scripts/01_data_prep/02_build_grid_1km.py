import os
import math

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin

OUTPUT_DIR = "Outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

window_path = os.path.join(OUTPUT_DIR, "CAOB_window.geojson")

# 2a. Read CAOB window
gdf_win = gpd.read_file(window_path)
print("CAOB_window CRS:", gdf_win.crs)

if gdf_win.crs.to_string() != "EPSG:4326":
    raise SystemExit("CAOB_window is not in EPSG:4326 – please fix before continuing.")

minx, miny, maxx, maxy = gdf_win.total_bounds
print("CAOB bounds:", (minx, miny, maxx, maxy))

# 2b. Define ~1 km resolution in degrees (about 0.008333°)
res = 0.008333333333333333  # ≈ 1 km at these latitudes

width = math.ceil((maxx - minx) / res)
height = math.ceil((maxy - miny) / res)

print(f"Grid resolution (deg): {res}")
print(f"Grid size (rows x cols): {height} x {width}")

transform = from_origin(minx, maxy, res, res)

profile = {
    "driver": "GTiff",
    "height": height,
    "width": width,
    "count": 1,
    "dtype": "uint8",
    "transform": transform,
    "nodata": 0,
    "crs": None,  # <- workaround: do NOT set EPSG here
}


out_tif = os.path.join(OUTPUT_DIR, "grid1km_CAOB.tif")

# 2c. Create empty 1 km grid raster
with rasterio.open(out_tif, "w", **profile) as dst:
    data = np.zeros((1, height, width), dtype="uint8")
    dst.write(data)

print("Saved 1 km grid to:", out_tif)
