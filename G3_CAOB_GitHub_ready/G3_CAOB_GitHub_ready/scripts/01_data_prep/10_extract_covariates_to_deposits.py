import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio

# -------------------------------------------------------------------
# Step 7: Extract covariates to deposits
# -------------------------------------------------------------------

DEPOSITS_PATH = os.path.join("Outputs", "deposits_por_Cu.geojson")
STACK_PATH    = os.path.join("Outputs", "CAOB_covariates_stack.tif")
OUT_CSV_PATH  = os.path.join("Outputs", "deposits_por_Cu_with_covariates.csv")


def main():
    # --- 1. Read deposits and stack ---------------------------------
    if not os.path.exists(DEPOSITS_PATH):
        raise FileNotFoundError(f"Deposits file not found: {DEPOSITS_PATH}")

    if not os.path.exists(STACK_PATH):
        raise FileNotFoundError(f"Stack file not found: {STACK_PATH}")

    print("Reading deposits from:", DEPOSITS_PATH)
    gdf = gpd.read_file(DEPOSITS_PATH)

    print("Number of deposits:", len(gdf))
    print("Deposits CRS:", gdf.crs)

    with rasterio.open(STACK_PATH) as src:
        stack_crs = src.crs
        stack_transform = src.transform
        band_descriptions = src.descriptions
        nodata = src.nodata
        print("Stack CRS:", stack_crs)
        print("Stack transform:", stack_transform)
        print("Number of bands in stack:", src.count)
        print("Band descriptions:", band_descriptions)

    # --- 2. CRS check / reprojection --------------------------------
    if gdf.crs is None:
        print("WARNING: deposits have no CRS defined. "
              "Assuming they are already in the stack CRS.")
        gdf.set_crs(stack_crs, inplace=True)
    elif gdf.crs != stack_crs:
        print("Reprojecting deposits from", gdf.crs, "to", stack_crs)
        gdf = gdf.to_crs(stack_crs)
    else:
        print("Deposits CRS matches stack CRS.")

    # --- 3. Build coordinate list -----------------------------------
    # (x, y) = (lon, lat) in EPSG:4326
    coords = [(geom.x, geom.y) for geom in gdf.geometry]

    # --- 4. Sample all bands at deposit locations -------------------
    print("Sampling covariates at deposit points...")
    with rasterio.open(STACK_PATH) as src:
        samples = np.stack(list(src.sample(coords)), axis=0)  # shape (N_points, N_bands)

    print("Raw sample array shape:", samples.shape)

    # Replace nodata by NaN for convenience
    if nodata is not None:
        samples = samples.astype("float32")
        samples[samples == nodata] = np.nan

    # --- 5. Attach covariates as columns ----------------------------
    band_names = []
    for i, desc in enumerate(band_descriptions, start=1):
        name = desc if desc not in (None, "") else f"band_{i}"
        band_names.append(name)

    for j, name in enumerate(band_names):
        gdf[name] = samples[:, j]

    # Ensure lon/lat columns exist (useful in CSV)
    if "lon" not in gdf.columns:
        gdf["lon"] = gdf.geometry.x
    if "lat" not in gdf.columns:
        gdf["lat"] = gdf.geometry.y

    # --- 6. Save to CSV (drop geometry) -----------------------------
    df_out = pd.DataFrame(gdf.drop(columns="geometry"))
    df_out.to_csv(OUT_CSV_PATH, index=False)
    print("Saved deposits with covariates to:", OUT_CSV_PATH)


if __name__ == "__main__":
    main()
