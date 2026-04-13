import os
from typing import List, Tuple

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

# -------------------------------------------------------------------
# Step 6: Stack covariates on the common 1 km CAOB grid
# -------------------------------------------------------------------

GRID_PATH = os.path.join("Outputs", "grid1km_CAOB.tif")

# (short_name, path)
COVARIATES: List[Tuple[str, str]] = [
    ("G1_DEM",   os.path.join("New Data", "G1_DEM_1km_correct.tif")),
    ("G1_SLOPE", os.path.join("New Data", "G1_SLOPE_1km_correct.tif")),
    ("G1_TRI",   os.path.join("New Data", "G1_TRI_1km_new_correct.tif")),
    ("G2_LITHOLOGY8", os.path.join("Outputs", "G2_LITHOLOGY8_1km.tif")),
    ("G3_DIST_FAULT", os.path.join("Outputs", "G3_DIST_FAULT_1km.tif")),
    ("G4_ASTER_PHYLLIC",
     os.path.join("Outputs", "G4_ASTER_PHYLLIC_1km_CAOB.tif")),
    ("G5_L8_iron",
     os.path.join("Outputs", "G5_L8_iron_1km_CAOB.tif")),
    ("E1_DIST_ROAD",
     os.path.join("data", "covariates", "E1_DIST_ROAD_1km_PY.tif")),
    ("E2_POP",        os.path.join("Outputs", "E2_POP_1km_CAOB.tif")),
    ("E3_VIIRS_2018",
     os.path.join("Outputs", "E3_VIIRS_2018_1km_CAOB.tif")),
    # The following effort covariates are disabled to avoid leakage
    # ("E4_KDE_ALLMIN",
    #  os.path.join("Outputs", "E4_KDE_ALLMIN_1km.tif")),
    # ("E5_DIST_MAJOR",
    #  os.path.join("Outputs", "E5_DIST_MAJOR_1km.tif")),
]


OUT_STACK_PATH = os.path.join("Outputs", "CAOB_covariates_stack.tif")


def check_alignment() -> None:
    """
    Print CRS / shape / transform of each covariate and check that
    all of them match each other and the grid1km_CAOB raster.
    """
    if not os.path.exists(GRID_PATH):
        raise FileNotFoundError(f"Grid file not found: {GRID_PATH}")

    with rasterio.open(GRID_PATH) as grd:
        grid_width = grd.width
        grid_height = grd.height
        grid_transform = grd.transform
        grid_crs = grd.crs

    ref_crs = None
    ref_transform = None
    ref_width = None
    ref_height = None

    print("=== Checking alignment against grid:", GRID_PATH)

    for name, path in COVARIATES:
        print(f"\n{name}: {path}")
        if not os.path.exists(path):
            print("  MISSING: file does not exist.")
            continue

        with rasterio.open(path) as src:
            print("  CRS        :", src.crs)
            print("  shape      :", src.height, "x", src.width)
            print("  bounds     :", src.bounds)
            print("  pixel size :", src.res)

            if ref_crs is None:
                ref_crs = src.crs
                ref_transform = src.transform
                ref_width = src.width
                ref_height = src.height
                print("  -> set as reference.")
            else:
                same_crs = (src.crs == ref_crs)
                same_transform = (src.transform == ref_transform)
                same_size = (src.width == ref_width and src.height == ref_height)
                print("  same CRS as first      :", same_crs)
                print("  same transform as first:", same_transform)
                print("  same size as first     :", same_size)

            same_grid_size = (src.width == grid_width and src.height == grid_height)
            same_grid_transform = (src.transform == grid_transform)
            print("  matches grid size      :", same_grid_size)
            print("  matches grid transform :", same_grid_transform)
            print("  matches grid CRS       :", src.crs == grid_crs)


def build_stack() -> None:
    """
    Read all covariates, resample to the 1 km CAOB grid when needed,
    and write a multi-band GeoTIFF where each band is one covariate.
    """
    if not os.path.exists(GRID_PATH):
        raise FileNotFoundError(f"Grid file not found: {GRID_PATH}")

    # Read grid metadata
    with rasterio.open(GRID_PATH) as grd:
        grid_profile = grd.profile
        grid_width = grd.width
        grid_height = grd.height
        grid_transform = grd.transform
        grid_crs = grd.crs  # currently None

    # If the grid has no CRS, infer it from the first existing covariate
    if grid_crs is None:
        print("Grid has no CRS; inferring CRS from covariates...")
        inferred_crs = None
        for _, path in COVARIATES:
            if os.path.exists(path):
                with rasterio.open(path) as src:
                    inferred_crs = src.crs
                break
        if inferred_crs is None:
            raise RuntimeError("Could not infer CRS from covariates.")
        grid_crs = inferred_crs
        grid_profile["crs"] = grid_crs
        print("  Inferred grid CRS:", grid_crs)

    # Resampling rule used only when a raster does not already match the grid
    resampling_rule = {
        "G1_DEM": Resampling.bilinear,
        "G1_SLOPE": Resampling.bilinear,
        "G1_TRI": Resampling.bilinear,
        "E1_DIST_ROAD": Resampling.bilinear,
        # all others: nearest (they should already match the grid)
    }

    arrays = []
    band_names = []

    for name, path in COVARIATES:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Covariate missing, cannot stack: {path}")

        with rasterio.open(path) as src:
            needs_resample = not (
                src.crs == grid_crs
                and src.width == grid_width
                and src.height == grid_height
                and src.transform == grid_transform
            )

            if needs_resample:
                print(f"{name}: resampling to grid...")
                dest = np.full((grid_height, grid_width), np.nan, dtype="float32")
                resampling = resampling_rule.get(name, Resampling.nearest)

                reproject(
                    source=rasterio.band(src, 1),
                    destination=dest,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=grid_transform,
                    dst_crs=grid_crs,
                    dst_width=grid_width,
                    dst_height=grid_height,
                    resampling=resampling,
                )
                data = dest
            else:
                print(f"{name}: already aligned with grid.")
                data = src.read(1).astype("float32")

        arrays.append(data)
        band_names.append(name)

    # Stack all bands into one 3D array
    stack_arr = np.stack(arrays, axis=0)

    profile = grid_profile.copy()
    profile.update(
        count=len(band_names),
        dtype="float32",
        nodata=np.nan,
    )

    os.makedirs(os.path.dirname(OUT_STACK_PATH), exist_ok=True)

    with rasterio.open(OUT_STACK_PATH, "w", **profile) as dst:
        dst.write(stack_arr)
        for idx, name in enumerate(band_names, start=1):
            dst.set_band_description(idx, name)

    print("\nSaved covariate stack to:", OUT_STACK_PATH)
    print("Band order:")
    for idx, name in enumerate(band_names, start=1):
        print(f"  {idx}: {name}")



if __name__ == "__main__":
    check_alignment()
    build_stack()
