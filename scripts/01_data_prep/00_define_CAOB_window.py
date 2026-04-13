import os
import geopandas as gpd
from shapely.geometry import box

# ==========================
# STEP 0 — Define CAOB window
# ==========================

# 0a. Create output folder
os.makedirs("Outputs", exist_ok=True)

# 0b. Define CAOB bounding box in lon/lat (EPSG:4326)
minx, maxx = 60.0, 120.0  # longitudes
miny, maxy = 37.0, 53.0   # latitudes

caob_poly = box(minx, miny, maxx, maxy)

gdf = gpd.GeoDataFrame(
    {"id": [1]},
    geometry=[caob_poly],
    crs="EPSG:4326"  # IMPORTANT: WGS84 geographic
)

# 0c. Save window
out_path = "Outputs/CAOB_window.geojson"
gdf.to_file(out_path, driver="GeoJSON")
print(f"Saved CAOB window to {out_path}")
print(f"CRS of CAOB_window: {gdf.crs}")
