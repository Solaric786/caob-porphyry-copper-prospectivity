import os
import argparse
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt

# Optional (keep if your rasterio/proj needs it, same as your old script)
os.environ.setdefault("PROJ_LIB", r"D:\ProgramData\Anaconda3\envs\PMRD_Hybird_PPP\Library\share\proj")
os.environ.setdefault("GDAL_DATA", r"D:\ProgramData\Anaconda3\envs\PMRD_Hybird_PPP\Library\share\gdal")

def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def get_lon_lat_cols(df):
    lon_col = pick_col(df, ["lon", "LONGITUDE", "Lon", "Longitude"])
    lat_col = pick_col(df, ["lat", "LATITUDE", "Lat", "Latitude"])
    if lon_col is None or lat_col is None:
        raise ValueError(f"Could not find lon/lat columns. Available: {list(df.columns)}")
    return lon_col, lat_col

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_csv", required=True, help="Outputs/eval_points_with_scores_blocks.csv")
    ap.add_argument("--score_col", required=True, help="e.g., score_model17_rankblend")
    ap.add_argument("--template_tif", required=True, help="CAOB_covariates_stack.tif (for grid/transform)")
    ap.add_argument("--out_tif", required=True)
    ap.add_argument("--out_png", required=True)
    ap.add_argument("--dep_csv", default=None, help="Optional: deposits CSV for black dots overlay")
    ap.add_argument("--cmap", default="turbo")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    df = pd.read_csv(args.eval_csv)
    if args.score_col not in df.columns:
        raise ValueError(f"score_col '{args.score_col}' not in eval_csv. Columns: {list(df.columns)}")

    lon_col, lat_col = get_lon_lat_cols(df)

    lon = pd.to_numeric(df[lon_col], errors="coerce").to_numpy()
    lat = pd.to_numeric(df[lat_col], errors="coerce").to_numpy()
    score = pd.to_numeric(df[args.score_col], errors="coerce").to_numpy()

    ok = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(score)
    lon, lat, score = lon[ok], lat[ok], score[ok]

    with rasterio.open(args.template_tif) as src:
        profile = src.profile.copy()
        transform = src.transform
        width, height = src.width, src.height
        bounds = src.bounds
        crs = src.crs

    Z = np.full((height, width), np.nan, dtype=np.float32)

    # Map lon/lat to pixel indices (assumes template_tif is in lon/lat CRS like your previous plot)
    rows, cols = rasterio.transform.rowcol(transform, lon, lat)

    rows = np.asarray(rows)
    cols = np.asarray(cols)
    inside = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)

    rows, cols, score = rows[inside], cols[inside], score[inside]

    # If multiple points hit same pixel, keep max (prospectivity-style)
    # (fast aggregation)
    lin = rows * width + cols
    order = np.argsort(lin)
    lin, score = lin[order], score[order]

    # groupby max without pandas
    uniq, idx_start = np.unique(lin, return_index=True)
    idx_end = np.r_[idx_start[1:], len(lin)]
    maxvals = np.array([np.nanmax(score[i0:i1]) for i0, i1 in zip(idx_start, idx_end)], dtype=np.float32)

    r = (uniq // width).astype(int)
    c = (uniq % width).astype(int)
    Z[r, c] = maxvals

    # Write GeoTIFF
    out_profile = profile.copy()
    out_profile.update(count=1, dtype="float32", nodata=np.nan, compress="lzw")
    with rasterio.open(args.out_tif, "w", **out_profile) as dst:
        dst.write(Z, 1)

    print("Saved GeoTIFF:", args.out_tif)

    # Optional overlay deposits
    dep_lon = dep_lat = None
    if args.dep_csv and os.path.exists(args.dep_csv):
        dep = pd.read_csv(args.dep_csv)
        dlon = pick_col(dep, ["lon", "LONGITUDE", "Lon", "Longitude"])
        dlat = pick_col(dep, ["lat", "LATITUDE", "Lat", "Latitude"])
        if dlon and dlat:
            dep_lon = pd.to_numeric(dep[dlon], errors="coerce").to_numpy()
            dep_lat = pd.to_numeric(dep[dlat], errors="coerce").to_numpy()

    # PNG
    plt.rcParams["font.family"] = "Times New Roman"
    fig = plt.figure(figsize=(10, 7), dpi=args.dpi)
    ax = plt.gca()

    extent = (bounds.left, bounds.right, bounds.bottom, bounds.top)
    im = ax.imshow(Z, extent=extent, origin="upper", cmap=args.cmap)

    if dep_lon is not None and dep_lat is not None:
        m = np.isfinite(dep_lon) & np.isfinite(dep_lat)
        ax.scatter(dep_lon[m], dep_lat[m], s=6, c="black", alpha=0.7, linewidths=0)

    ax.set_title(f"Prospectivity map — {args.score_col}", fontsize=12)
    ax.set_xlabel("Longitude" if "EPSG:4326" in str(crs) else "X")
    ax.set_ylabel("Latitude" if "EPSG:4326" in str(crs) else "Y")

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Prospectivity score", fontsize=10)

    plt.tight_layout()
    plt.savefig(args.out_png)
    plt.close()
    print("Saved PNG:", args.out_png)

if __name__ == "__main__":
    main()
