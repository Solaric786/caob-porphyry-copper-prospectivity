import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio

from shapely.geometry import Point

# ---------------------------------------------------------
# Step 8: Background points + covariates
# ---------------------------------------------------------

CAOB_WINDOW_PATH = os.path.join("Outputs", "CAOB_window.geojson")
STACK_PATH       = os.path.join("Outputs", "CAOB_covariates_stack.tif")

BACKGROUND_GPKG  = os.path.join("Outputs", "background_uniform_1km.gpkg")
OUT_CSV_PATH     = os.path.join("Outputs", "background_points_1km.csv")

# how many background points (you can change this)
N_BACKGROUND = 50000

# if True, regenerate background points even if GPKG exists
OVERWRITE_POINTS = False


def generate_background_points():
    """Generate uniform random background points inside CAOB_window."""
    if not os.path.exists(CAOB_WINDOW_PATH):
        raise FileNotFoundError(f"CAOB window file not found: {CAOB_WINDOW_PATH}")

    print("Reading CAOB_window from:", CAOB_WINDOW_PATH)
    window = gpd.read_file(CAOB_WINDOW_PATH)
    if window.crs is None:
        print("WARNING: CAOB_window has no CRS; setting to EPSG:4326.")
        window.set_crs(epsg=4326, inplace=True)
    elif window.crs.to_epsg() != 4326:
        print("Reprojecting CAOB_window to EPSG:4326.")
        window = window.to_crs(epsg=4326)

    poly = window.geometry.unary_union
    minx, miny, maxx, maxy = poly.bounds
    print("CAOB bounds:", minx, miny, maxx, maxy)

    pts = []
    rng = np.random.default_rng(1234)  # fixed seed for reproducibility
    while len(pts) < N_BACKGROUND:
        xs = rng.uniform(minx, maxx, size=10000)
        ys = rng.uniform(miny, maxy, size=10000)
        for x, y in zip(xs, ys):
            p = Point(x, y)
            if poly.contains(p):
                pts.append(p)
                if len(pts) >= N_BACKGROUND:
                    break

    gdf_bg = gpd.GeoDataFrame(
        {"id": np.arange(1, len(pts) + 1)},
        geometry=pts,
        crs="EPSG:4326",
    )

    print(f"Generated {len(gdf_bg)} background points.")
    os.makedirs(os.path.dirname(BACKGROUND_GPKG), exist_ok=True)
    gdf_bg.to_file(BACKGROUND_GPKG, driver="GPKG")
    print("Saved background points to:", BACKGROUND_GPKG)
    return gdf_bg


def load_or_build_background_points():
    """Load background points if they exist; otherwise generate them."""
    if os.path.exists(BACKGROUND_GPKG) and not OVERWRITE_POINTS:
        print("Loading existing background points from:", BACKGROUND_GPKG)
        gdf_bg = gpd.read_file(BACKGROUND_GPKG)
        print("Number of background points:", len(gdf_bg))
        return gdf_bg
    else:
        if os.path.exists(BACKGROUND_GPKG):
            print("OVERWRITE_POINTS = True, regenerating background points.")
        else:
            print("No background GPKG found; generating new points.")
        return generate_background_points()


def extract_covariates_to_background(gdf_bg):
    """Sample all bands of the stack at background point locations."""
    if not os.path.exists(STACK_PATH):
        raise FileNotFoundError(f"Stack file not found: {STACK_PATH}")

    print("Reading covariate stack from:", STACK_PATH)
    with rasterio.open(STACK_PATH) as src:
        stack_crs = src.crs
        stack_transform = src.transform
        band_descriptions = src.descriptions
        nodata = src.nodata
        n_bands = src.count
        print("Stack CRS:", stack_crs)
        print("Stack transform:", stack_transform)
        print("Number of bands:", n_bands)
        print("Band descriptions:", band_descriptions)

    # CRS check
    if gdf_bg.crs is None:
        print("WARNING: background points have no CRS; assuming stack CRS.")
        gdf_bg.set_crs(stack_crs, inplace=True)
    elif gdf_bg.crs != stack_crs:
        print("Reprojecting background points from", gdf_bg.crs, "to", stack_crs)
        gdf_bg = gdf_bg.to_crs(stack_crs)
    else:
        print("Background CRS matches stack CRS.")

    coords = [(geom.x, geom.y) for geom in gdf_bg.geometry]

    print("Sampling covariates at background points...")
    with rasterio.open(STACK_PATH) as src:
        samples = np.stack(list(src.sample(coords)), axis=0)

    print("Sample array shape:", samples.shape)

    if nodata is not None:
        samples = samples.astype("float32")
        samples[samples == nodata] = np.nan

    band_names = []
    for i, desc in enumerate(band_descriptions, start=1):
        name = desc if desc not in (None, "") else f"band_{i}"
        band_names.append(name)

    for j, name in enumerate(band_names):
        gdf_bg[name] = samples[:, j]

    gdf_bg["lon"] = gdf_bg.geometry.x
    gdf_bg["lat"] = gdf_bg.geometry.y

    df_out = pd.DataFrame(gdf_bg.drop(columns="geometry"))
    df_out.to_csv(OUT_CSV_PATH, index=False)
    print("Saved background points with covariates to:", OUT_CSV_PATH)


def main():
    gdf_bg = load_or_build_background_points()
    extract_covariates_to_background(gdf_bg)


if __name__ == "__main__":
    main()
