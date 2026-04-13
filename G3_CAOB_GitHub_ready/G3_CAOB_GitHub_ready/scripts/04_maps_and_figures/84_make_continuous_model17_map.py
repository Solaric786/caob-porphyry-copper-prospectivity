import os, re
import numpy as np
import pandas as pd
import rasterio

try:
    from scipy.ndimage import distance_transform_edt, gaussian_filter
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

OUT_DIR = r"Outputs/paper_run_20260118"
os.makedirs(OUT_DIR, exist_ok=True)

TEMPLATE_TIF = r"outputs/model6_xgb_prospectivity.tif"  # exists, aligned grid
MODEL16_CSV_CAND = [
    os.path.join(OUT_DIR, "cv_preds_model16_pu_stackmeta.csv"),
    r"Outputs/cv_preds_model16_pu_stackmeta.csv",
]
METRICS_CAND = [
    os.path.join(OUT_DIR, "model17_rankblend_metrics.txt"),
    r"Outputs/model17_rankblend_metrics.txt",
]

def first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("Missing all candidates:\n" + "\n".join(paths))

model16_csv = first_existing(MODEL16_CSV_CAND)
metrics_txt = first_existing(METRICS_CAND)

df = pd.read_csv(model16_csv)
need_cols = ["lon","lat","p_sup","p_y_pu_corrected","score_hdipp"]
for c in need_cols:
    if c not in df.columns:
        raise SystemExit(f"Missing column '{c}' in {model16_csv}. Have: {list(df.columns)}")

# weights from model17 metrics (average folds)
w = None
with open(metrics_txt, "r", encoding="utf-8", errors="ignore") as f:
    txt = f.read()
m = re.findall(r"fold\s+\d+:\s+\[([0-9\.\s,]+)\]", txt)
if m:
    W = []
    for s in m:
        nums = [float(x) for x in s.replace(",", " ").split()]
        if len(nums) == 4:
            W.append(nums)
    if W:
        w = np.mean(np.array(W), axis=0)
if w is None:
    # fallback (your observed average)
    w = np.array([0.626, 0.044, 0.230, 0.100], dtype=float)

# point-wise mean/std for standardization (to mimic Model17 blending logic)
mu_sd = {}
for c in ["p_sup","p_y_pu_corrected","score_hdipp"]:
    v = df[c].astype(float).values
    mu = float(np.nanmean(v))
    sd = float(np.nanstd(v))
    if sd <= 1e-12: sd = 1.0
    mu_sd[c] = (mu, sd)

def rasterize_points_to_grid(values, src, lon, lat, nodata=np.nan, reducer="max"):
    H, W = src.height, src.width
    arr = np.full((H, W), np.nan, dtype=np.float32)

    rows, cols = rasterio.transform.rowcol(src.transform, lon, lat)
    rows = np.asarray(rows); cols = np.asarray(cols)
    ok = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)

    rows = rows[ok]; cols = cols[ok]
    v = np.asarray(values, dtype=np.float32)[ok]

    if reducer == "max":
        # keep max if multiple fall into same pixel
        for r, c, val in zip(rows, cols, v):
            if np.isnan(arr[r, c]) or val > arr[r, c]:
                arr[r, c] = val
    else:
        # mean reducer (two-pass)
        s = np.zeros((H, W), dtype=np.float64)
        n = np.zeros((H, W), dtype=np.int32)
        for r, c, val in zip(rows, cols, v):
            s[r, c] += float(val); n[r, c] += 1
        msk = n > 0
        arr[msk] = (s[msk] / n[msk]).astype(np.float32)

    return arr

def fill_missing_nearest(arr):
    m = np.isnan(arr)
    if m.all():
        return arr
    if HAS_SCIPY:
        # nearest-neighbor fill
        _, (ii, jj) = distance_transform_edt(m, return_indices=True)
        filled = arr[ii, jj]
        return filled.astype(np.float32)
    else:
        # fallback: leave sparse (still better than crash)
        return arr

def save_tif(path, src, arr):
    prof = src.profile.copy()
    prof.update(count=1, dtype="float32", nodata=None, compress="lzw")
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr.astype(np.float32), 1)

# load template grid
with rasterio.open(TEMPLATE_TIF) as src:
    lon = df["lon"].astype(float).values
    lat = df["lat"].astype(float).values

    rasters = {}
    for col, outname in [
        ("p_sup", "model16_p_sup_prospectivity.tif"),
        ("p_y_pu_corrected", "model16_p_y_pu_corrected_prospectivity.tif"),
        ("score_hdipp", "hdipp_prospectivity.tif"),
    ]:
        arr = rasterize_points_to_grid(df[col].values, src, lon, lat, reducer="max")
        arr = fill_missing_nearest(arr)

        # optional mild smoothing to remove “blocky” look
        if HAS_SCIPY:
            arr = gaussian_filter(arr, sigma=1.0).astype(np.float32)

        outpath = os.path.join(OUT_DIR, outname)
        save_tif(outpath, src, arr)
        print("Saved:", outpath)
        rasters[col] = arr

    # Standardize using point-wise mu/sd, then blend with Model17 weights
    z_sup = (rasters["p_sup"] - mu_sd["p_sup"][0]) / mu_sd["p_sup"][1]
    z_pu  = (rasters["p_y_pu_corrected"] - mu_sd["p_y_pu_corrected"][0]) / mu_sd["p_y_pu_corrected"][1]
    z_hd  = (rasters["score_hdipp"] - mu_sd["score_hdipp"][0]) / mu_sd["score_hdipp"][1]

    # model6 raster already exists, use it directly
    with rasterio.open(r"outputs/model6_xgb_prospectivity.tif") as m6:
        m6a = m6.read(1).astype(np.float32)
        mu = float(np.nanmean(m6a)); sd = float(np.nanstd(m6a)); 
        if sd <= 1e-12: sd = 1.0
        z_m6 = (m6a - mu) / sd

    # order: (p_sup, p_y_pu_corrected, score_model6, score_hdipp)
    w_sup, w_pu, w_m6, w_hd = [float(x) for x in w]
    blend = w_sup*z_sup + w_pu*z_pu + w_m6*z_m6 + w_hd*z_hd

    # min-max to [0,1]
    bmin = float(np.nanmin(blend)); bmax = float(np.nanmax(blend))
    out01 = (blend - bmin) / (bmax - bmin + 1e-12)

    out_tif = os.path.join(OUT_DIR, "model17_rankblend_CONTINUOUS.tif")
    save_tif(out_tif, src, out01)
    print("Saved:", out_tif)
    print("Weights used (avg folds):", [w_sup, w_pu, w_m6, w_hd])

# also save PNG
import matplotlib.pyplot as plt
import rasterio
with rasterio.open(os.path.join(OUT_DIR, "model17_rankblend_CONTINUOUS.tif")) as ds:
    a = ds.read(1)
plt.figure(figsize=(14,8))
plt.imshow(a, origin="upper")
plt.colorbar(label="Prospectivity score")
plt.title("Prospectivity map — Model17 (continuous blend)")
plt.tight_layout()
png = os.path.join(OUT_DIR, "model17_rankblend_CONTINUOUS.png")
plt.savefig(png, dpi=300)
print("Saved:", png)
