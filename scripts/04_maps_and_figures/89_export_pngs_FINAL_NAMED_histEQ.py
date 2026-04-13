# 89_export_pngs_FINAL_NAMED_CONTRAST_v4_with_CuPoints.py
import os, glob
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# -------------------- PATHS --------------------
FINAL_DIR = r"Outputs\paper_run_20260118\FINAL_NAMED"
DEP_CSV   = r"Outputs\deposits_covariates_v2.csv"   # Cu deposits (lon/lat)

# -------------------- OUTPUT STYLE --------------------
DPI = 600                 # 400 or 600
FIGSIZE = (16, 5)
CMAP_NAME = "turbo"

# Make the map less “over-red” (milder contrast than full hist-eq)
DISPLAY_MODE = "stretch_gamma"      # "stretch_gamma" (recommended) or "hist_eq"
CLIP_PCT = (5, 95)                  # was (2,98); tighter -> less saturation
GAMMA = 1.20                        # >1 reduces high-end dominance (less red)

# Copper deposit markers (bigger + black, readable)
DEP_S = 28
DEP_FACE = "black"
DEP_EDGE = "white"                  # keeps black points visible on dark areas
DEP_LW = 0.9
DEP_ALPHA = 0.95

plt.rcParams["font.family"] = "Times New Roman"

# Colormap + make masked/no-data NOT white (avoids white rectangles)
cmap = plt.get_cmap(CMAP_NAME).copy()
bad_color = cmap(0.0)
cmap.set_bad(bad_color)
cmap.set_under(bad_color)

# -------------------- CLEAN TITLES --------------------
TITLE_MAP = {
    "Model02_HDIPP_Baseline_prospectivity": "Model 2 (HDIPP baseline) prospectivity",
    "Model06_XGB_Tuned_Baseline_prospectivity": "Model 6 (tuned XGBoost baseline) prospectivity",
    "Model16_PU_StackMeta_p_sup_prospectivity": "Model 16 (PU stack-meta) — supervised score",
    "Model16_PU_StackMeta_p_y_pu_corrected_prospectivity": "Model 16 (PU stack-meta) — PU-corrected score",
    "Model17_ROWE_prospectivity_CONTINUOUS": "Model 17 (rank-blend ensemble) prospectivity (continuous)",
}
def clean_title(stem: str) -> str:
    return TITLE_MAP.get(stem, stem.replace("_", " "))

# -------------------- HELPERS --------------------
def guess_lonlat_cols(df):
    cols = {c.lower(): c for c in df.columns}
    for lo, la in [("lon","lat"), ("longitude","latitude"), ("x","y"), ("long","lat")]:
        if lo in cols and la in cols:
            return cols[lo], cols[la]
    raise ValueError(f"Cannot find lon/lat columns in deposits CSV. Columns: {list(df.columns)}")

def extent_from_bounds(bounds):
    # rasterio bounds: left, bottom, right, top
    return (bounds.left, bounds.right, bounds.bottom, bounds.top)

def read_raster_masked(ds):
    """
    Robust masked float32 array. Avoids mask broadcast errors.
    """
    a = ds.read(1, masked=True).astype(np.float32)
    data = np.asarray(a.data, dtype=np.float32)
    mask = np.ma.getmaskarray(a).copy()          # ALWAYS 2D
    mask |= ~np.isfinite(data)
    if ds.nodata is not None and np.isfinite(ds.nodata):
        mask |= (data == np.float32(ds.nodata))
    return np.ma.array(data, mask=mask)

def display_normalize(a_masked, mode=DISPLAY_MODE, clip_pct=CLIP_PCT, gamma=GAMMA, bins=2048):
    """
    Returns values in [0,1] for display, with a milder contrast than full hist-eq.
    """
    m = np.ma.getmaskarray(a_masked)
    v = a_masked.compressed()
    v = v[np.isfinite(v)]
    if v.size < 10:
        out = np.zeros_like(a_masked.filled(0.0), dtype=np.float32)
        return np.ma.array(out, mask=m)

    lo, hi = np.percentile(v, clip_pct)
    if (not np.isfinite(lo)) or (not np.isfinite(hi)) or (hi <= lo):
        lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
        if (not np.isfinite(lo)) or (not np.isfinite(hi)) or (hi <= lo):
            out = np.zeros_like(a_masked.filled(0.0), dtype=np.float32)
            out[~m] = 0.5
            return np.ma.array(out, mask=m)

    x = np.clip(a_masked.filled(lo).astype(np.float32), lo, hi)

    if mode == "hist_eq":
        vv = np.clip(v, lo, hi)
        hist, edges = np.histogram(vv, bins=bins, range=(lo, hi))
        cdf = hist.cumsum().astype(np.float64)
        cdf /= (cdf[-1] + 1e-12)
        centers = 0.5 * (edges[:-1] + edges[1:])
        y = np.interp(x, centers, cdf, left=0.0, right=1.0).astype(np.float32)
        y = np.clip(y, 0.0, 1.0) ** gamma       # gamma reduces “too red”
        return np.ma.array(y, mask=m)

    # recommended: percentile stretch + gamma
    y = (x - lo) / (hi - lo + 1e-12)
    y = np.clip(y, 0.0, 1.0) ** gamma
    return np.ma.array(y.astype(np.float32), mask=m)

def filter_points_to_extent(df_pts, lon_col, lat_col, extent):
    west, east, south, north = extent
    d = df_pts[[lon_col, lat_col]].dropna().copy()
    d = d[np.isfinite(d[lon_col]) & np.isfinite(d[lat_col])]
    d = d[(d[lon_col] >= west) & (d[lon_col] <= east) &
          (d[lat_col] >= south) & (d[lat_col] <= north)]
    return d

# -------------------- LOAD DEPOSITS --------------------
dep_df = None
lon_col = lat_col = None
if os.path.exists(DEP_CSV):
    dep_df = pd.read_csv(DEP_CSV)
    lon_col, lat_col = guess_lonlat_cols(dep_df)
else:
    print(f"[WARN] Deposits CSV not found: {DEP_CSV} (maps will export without points)")

# -------------------- MAIN --------------------
tifs = sorted(glob.glob(os.path.join(FINAL_DIR, "*.tif")))
if not tifs:
    raise FileNotFoundError(f"No .tif files found in: {FINAL_DIR}")

for tif in tifs:
    stem = os.path.splitext(os.path.basename(tif))[0]
    out_png = os.path.join(FINAL_DIR, f"{stem}.png")

    with rasterio.open(tif) as ds:
        a = read_raster_masked(ds)
        extent = extent_from_bounds(ds.bounds)

        # Most of your FINAL_NAMED tifs are already lon/lat; keep labels if geographic.
        is_lonlat = bool(ds.crs) and getattr(ds.crs, "is_geographic", False)

    disp = display_normalize(a)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    im = ax.imshow(
        disp, cmap=cmap, vmin=0.0, vmax=1.0,
        origin="upper", extent=extent, interpolation="nearest"
    )
    ax.set_aspect("auto")
    ax.set_facecolor(bad_color)

    ax.set_title(clean_title(stem), fontsize=16)
    ax.set_xlabel("Longitude (°E)" if is_lonlat else "X", fontsize=12)
    ax.set_ylabel("Latitude (°N)" if is_lonlat else "Y", fontsize=12)
    ax.tick_params(labelsize=10)

    # ---- Overlay Cu deposits (bigger black points) ----
    if dep_df is not None and is_lonlat:
        d = filter_points_to_extent(dep_df, lon_col, lat_col, extent)
        ax.scatter(
            d[lon_col].values, d[lat_col].values,
            s=DEP_S, facecolors=DEP_FACE, edgecolors=DEP_EDGE,
            linewidths=DEP_LW, alpha=DEP_ALPHA, zorder=6, label="Cu deposits"
        )
        ax.legend(loc="lower left", frameon=True, fontsize=10)

    # ---- Colorbar ----
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size="3.5%", pad=0.08)
    cb = fig.colorbar(im, cax=cax)
    cb.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cb.set_label("Prospectivity (display-normalized, 0–1)", fontsize=11)
    cb.ax.tick_params(labelsize=10)

    fig.tight_layout()
    fig.savefig(out_png, dpi=DPI)
    plt.close(fig)
    print("Saved:", out_png)
