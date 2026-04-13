import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

DATA_DIR = "Data"
OUTPUT_DIR = "Outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

porphyry_path = os.path.join(DATA_DIR, "Porphyry_datasheet.csv")
window_path = os.path.join(OUTPUT_DIR, "CAOB_window.geojson")

print(f"Reading deposits from: {porphyry_path}")
df = pd.read_csv(porphyry_path, encoding="latin1")


print("Columns in Porphyry_datasheet:")
print(list(df.columns))

# ---- auto-detect latitude / longitude columns ----
cand_lat = [c for c in df.columns if "lat" in c.lower()]
cand_lon = [c for c in df.columns if ("lon" in c.lower()) or ("long" in c.lower())]

if not cand_lat or not cand_lon:
    raise SystemExit(
        "Could not auto-detect lat/lon columns. "
        "Please edit the script and set lat_col / lon_col manually."
    )

lat_col = cand_lat[0]
lon_col = cand_lon[0]
print(f"Using latitude column: {lat_col}")
print(f"Using longitude column: {lon_col}")

# ---- build GeoDataFrame in EPSG:4326 ----
gdf_dep = gpd.GeoDataFrame(
    df.copy(),
    geometry=[Point(xy) for xy in zip(df[lon_col], df[lat_col])],
    crs="EPSG:4326"
)
print("Deposits CRS:", gdf_dep.crs)

# ---- load CAOB window and check CRS ----
gdf_win = gpd.read_file(window_path)
print("CAOB_window CRS:", gdf_win.crs)

# ---- clip to CAOB window ----
gdf_clip = gpd.clip(gdf_dep, gdf_win)
print(f"Total deposits in file: {len(gdf_dep)}")
print(f"Deposits inside CAOB: {len(gdf_clip)}")

# ---- remove duplicates (by coordinates + optional name) ----
gdf_clip["x"] = gdf_clip.geometry.x
gdf_clip["y"] = gdf_clip.geometry.y

name_cols = [c for c in gdf_clip.columns if "name" in c.lower()]
if name_cols:
    name_col = name_cols[0]
    print(f"Using name column for duplicate check: {name_col}")
    subset = ["x", "y", name_col]
else:
    subset = ["x", "y"]

gdf_unique = gdf_clip.drop_duplicates(subset=subset)
print(f"Deposits after removing duplicates: {len(gdf_unique)}")

# ---- save outputs ----
out_geojson = os.path.join(OUTPUT_DIR, "deposits_por_Cu.geojson")
gdf_unique.to_file(out_geojson, driver="GeoJSON")
print("Saved deposits to:", out_geojson)

out_csv = os.path.join(OUTPUT_DIR, "deposits_por_Cu.csv")
df_out = gdf_unique.drop(columns="geometry").copy()
df_out["Longitude"] = gdf_unique.geometry.x
df_out["Latitude"] = gdf_unique.geometry.y
df_out.to_csv(out_csv, index=False)
print("Saved deposits table to:", out_csv)
