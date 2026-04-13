# 78_model17_rankblend_predict_map.py
# Build Model17 raster by rank-blending 4 component rasters, then plot in the SAME style as 57_model6_xgb_predict_map.py

import os

# (Optional) keep same Windows env pattern as your Model6 map script
# os.environ["PROJ_LIB"]  = r"D:\ProgramData\Anaconda3\envs\PMRD_Hybird_PPP\Library\share\proj"
# os.environ["GDAL_DATA"] = r"D:\ProgramData\Anaconda3\envs\PMRD_Hybird_PPP\Library\share\gdal"

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
import matplotlib.pyplot as plt

# -----------------------------
# PATHS (EDIT THESE)
# -----------------------------
IN_CSV_MODEL16 = r"Outputs/cv_preds_model16_pu_stackmeta.csv"   # has y + score columns used by Model17
DEP_V2         = r"Outputs/deposits_covariates_v2.csv"          # for overlay points

# Component rasters (must be same grid/extent/resolution/CRS)
RASTER_P_SUP   = r"Outputs/model16_p_sup_prospectivity.tif"              # p_sup
RASTER_P_PU    = r"Outputs/model16_p_y_pu_corrected_prospectivity.tif"   # p_y_pu_corrected
RASTER_M6      = r"Outputs/model6_xgb_prospectivity.tif"                 # score_model6
RASTER_HDIPP   = r"Outputs/hdipp_prospectivity.tif"                      # score_hdipp

OUT_DIR = r"Outputs"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_TIF = os.path.join(OUT_DIR, "model17_rankblend_prospectivity.tif")
OUT_PNG = os.path.join(OUT_DIR, "model17_rankblend_prospectivity.png")
OUT_TXT = os.path.join(OUT_DIR, "model17_rankblend_map_weights.txt")

# -----------------------------
# MODEL17 settings (same as 77_...py)
# -----------------------------
SEED = 42
SCORE_COLS = ["p_sup", "p_y_pu_corrected", "score_model6", "score_hdipp"]

W_SUP_GRID = np.linspace(0.55, 0.95, 21)   # keep p_sup dominant
W_M6_GRID  = np.linspace(0.00, 0.25, 6)
W_HD_GRID  = np.linspace(0.00, 0.20, 5)
SUBSAMPLE_NEG = 12000

TILE = 512

# -----------------------------
# Helpers
# -----------------------------
def get_lon_lat(df: pd.DataFrame):
    if "lon" in df.columns and "lat" in df.columns:
        return df["lon"].astype(float).values, df["lat"].astype(float).values
    if "LONGITUDE" in df.columns and "LATITUDE" in df.columns:
        return df["LONGITUDE"].astype(float).values, df["LATITUDE"].astype(float).values
    raise ValueError("No lon/lat columns found")

def topk_recall(y_true, y_score, frac=0.10):
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    n = len(y_true)
    k = max(1, int(frac * n))
    idx = np.argpartition(-y_score, k - 1)[:k]
    denom = max(1, int(y_true.sum()))
    return float(y_true[idx].sum() / denom)

def standardize_train_apply(train_vec, test_vec):
    mu = float(np.mean(train_vec))
    sd = float(np.std(train_vec))
    if sd <= 1e-12:
        sd = 1.0
    return (train_vec - mu) / sd, (test_vec - mu) / sd, mu, sd

def make_train_subset_idx(y_train, rng, n_neg=12000):
    y_train = np.asarray(y_train, dtype=int)
    pos_idx = np.where(y_train == 1)[0]
    neg_idx = np.where(y_train == 0)[0]
    if len(neg_idx) > n_neg:
        neg_idx = rng.choice(neg_idx, size=n_neg, replace=False)
    return np.sort(np.r_[pos_idx, neg_idx])

def check_same_grid(ref, other, other_name="other"):
    ok = True
    msg = []
    if (ref.width != other.width) or (ref.height != other.height):
        ok = False
        msg.append(f"shape ref=({ref.height},{ref.width}) vs {other_name}=({other.height},{other.width})")
    if ref.transform != other.transform:
        ok = False
        msg.append("transform differs")
    if str(ref.crs) != str(other.crs):
        ok = False
        msg.append(f"crs ref={ref.crs} vs {other_name}={other.crs}")
    return ok, "; ".join(msg)

# -----------------------------
# 1) Load Model16 CSV and compute Model17 weights (single global weight set for raster)
# -----------------------------
df = pd.read_csv(IN_CSV_MODEL16)

required = {"y"} | set(SCORE_COLS)
missing = sorted(list(required - set(df.columns)))
if missing:
    raise ValueError(f"Missing required columns in {IN_CSV_MODEL16}: {missing}")

df = df.copy()
df["y"] = df["y"].astype(int)

# Fill NaN scores safely
for c in SCORE_COLS:
    if df[c].isna().any():
        df[c] = df[c].fillna(df[c].median())

y_all = df["y"].values.astype(int)
rng = np.random.default_rng(SEED)

# Standardize scores using ALL data as "train" (since we need one global model for mapping)
Z_list = []
mu_sd = {}
for c in SCORE_COLS:
    z, _, mu, sd = standardize_train_apply(df[c].values.astype(float), df[c].values.astype(float))
    Z_list.append(z)
    mu_sd[c] = (mu, sd)

Z = np.column_stack(Z_list)  # (n, 4)

# Fast subset to tune weights (same idea as Model17)
sub_idx = make_train_subset_idx(y_all, rng, n_neg=SUBSAMPLE_NEG)
Z_sub = Z[sub_idx]
y_sub = y_all[sub_idx]

best_obj = -1e9
best_w = None

for w_sup in W_SUP_GRID:
    for w_m6 in W_M6_GRID:
        for w_hd in W_HD_GRID:
            w_pu = 1.0 - (w_sup + w_m6 + w_hd)
            if w_pu < 0:
                continue
            w = np.array([w_sup, w_pu, w_m6, w_hd], dtype=float)
            s = Z_sub @ w
            t1  = topk_recall(y_sub, s, 0.01)
            t5  = topk_recall(y_sub, s, 0.05)
            t10 = topk_recall(y_sub, s, 0.10)
            obj = t10 + 0.50 * t5 + 0.25 * t1
            if obj > best_obj:
                best_obj = obj
                best_w = w

print("Best global weights (w_sup, w_pu, w_m6, w_hdipp):", best_w, " obj=", best_obj)

with open(OUT_TXT, "w", encoding="utf-8") as f:
    f.write("Model17 raster blend weights (global)\n")
    f.write("Order: (p_sup, p_y_pu_corrected, score_model6, score_hdipp)\n")
    f.write(f"best_w = {best_w.tolist()}\n")
    f.write(f"objective(subset) = {best_obj:.6f}\n")
    f.write("Standardization mu/sd from Outputs/cv_preds_model16_pu_stackmeta.csv:\n")
    for c in SCORE_COLS:
        mu, sd = mu_sd[c]
        f.write(f"  {c}: mu={mu:.8f}, sd={sd:.8f}\n")

# -----------------------------
# 2) Build Model17 GeoTIFF in tiles (two-pass: first find global min/max, then write min-max scaled 0..1)
# -----------------------------
with rasterio.open(RASTER_P_SUP) as src_sup, \
     rasterio.open(RASTER_P_PU) as src_pu, \
     rasterio.open(RASTER_M6) as src_m6, \
     rasterio.open(RASTER_HDIPP) as src_hd:

    # Require same grid to keep your workflow simple/robust
    ok, why = check_same_grid(src_sup, src_pu, "p_y_pu_corrected")
    if not ok: raise ValueError("Raster grid mismatch: " + why)
    ok, why = check_same_grid(src_sup, src_m6, "model6")
    if not ok: raise ValueError("Raster grid mismatch: " + why)
    ok, why = check_same_grid(src_sup, src_hd, "hdipp")
    if not ok: raise ValueError("Raster grid mismatch: " + why)

    profile = src_sup.profile.copy()
    profile.update(count=1, dtype="float32", nodata=np.nan, compress="lzw")

    # --- pass 1: get min/max of blended z-score over raster (ignoring NaNs)
    zmin = np.inf
    zmax = -np.inf

    for row0 in range(0, src_sup.height, TILE):
        for col0 in range(0, src_sup.width, TILE):
            h = min(TILE, src_sup.height - row0)
            w = min(TILE, src_sup.width - col0)
            win = Window(col0, row0, w, h)

            a_sup = src_sup.read(1, window=win).astype(np.float32)
            a_pu  = src_pu.read(1, window=win).astype(np.float32)
            a_m6  = src_m6.read(1, window=win).astype(np.float32)
            a_hd  = src_hd.read(1, window=win).astype(np.float32)

            mask = (~np.isfinite(a_sup)) | (~np.isfinite(a_pu)) | (~np.isfinite(a_m6)) | (~np.isfinite(a_hd))
            if mask.all():
                continue

            mu_sup, sd_sup = mu_sd["p_sup"]
            mu_pu,  sd_pu  = mu_sd["p_y_pu_corrected"]
            mu_m6,  sd_m6  = mu_sd["score_model6"]
            mu_hd,  sd_hd  = mu_sd["score_hdipp"]

            z = (best_w[0] * (a_sup - mu_sup) / sd_sup +
                 best_w[1] * (a_pu  - mu_pu)  / sd_pu  +
                 best_w[2] * (a_m6  - mu_m6)  / sd_m6  +
                 best_w[3] * (a_hd  - mu_hd)  / sd_hd)

            z[mask] = np.nan
            local_min = np.nanmin(z)
            local_max = np.nanmax(z)
            if np.isfinite(local_min): zmin = min(zmin, float(local_min))
            if np.isfinite(local_max): zmax = max(zmax, float(local_max))

    if (not np.isfinite(zmin)) or (not np.isfinite(zmax)) or (zmax <= zmin + 1e-12):
        raise ValueError("Could not compute valid zmin/zmax from rasters (check inputs / NaNs).")

    print("Raster blended z-score range:", zmin, zmax)

    # --- pass 2: write scaled 0..1 raster
    with rasterio.open(OUT_TIF, "w", **profile) as dst:
        for row0 in range(0, src_sup.height, TILE):
            for col0 in range(0, src_sup.width, TILE):
                h = min(TILE, src_sup.height - row0)
                w = min(TILE, src_sup.width - col0)
                win = Window(col0, row0, w, h)

                a_sup = src_sup.read(1, window=win).astype(np.float32)
                a_pu  = src_pu.read(1, window=win).astype(np.float32)
                a_m6  = src_m6.read(1, window=win).astype(np.float32)
                a_hd  = src_hd.read(1, window=win).astype(np.float32)

                mask = (~np.isfinite(a_sup)) | (~np.isfinite(a_pu)) | (~np.isfinite(a_m6)) | (~np.isfinite(a_hd))

                mu_sup, sd_sup = mu_sd["p_sup"]
                mu_pu,  sd_pu  = mu_sd["p_y_pu_corrected"]
                mu_m6,  sd_m6  = mu_sd["score_model6"]
                mu_hd,  sd_hd  = mu_sd["score_hdipp"]

                z = (best_w[0] * (a_sup - mu_sup) / sd_sup +
                     best_w[1] * (a_pu  - mu_pu)  / sd_pu  +
                     best_w[2] * (a_m6  - mu_m6)  / sd_m6  +
                     best_w[3] * (a_hd  - mu_hd)  / sd_hd)

                s01 = (z - zmin) / (zmax - zmin)
                s01 = np.clip(s01, 0.0, 1.0).astype(np.float32)
                s01[mask] = np.nan

                dst.write(s01, 1, window=win)

print(f"Saved GeoTIFF: {OUT_TIF}")

# -----------------------------
# 3) Save PNG with deposits overlay (same style as your Model6 map)
# -----------------------------
dep = pd.read_csv(DEP_V2)
dep_lon, dep_lat = get_lon_lat(dep)

with rasterio.open(OUT_TIF) as src:
    Z = src.read(1)
    extent = (src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top)
    crs = src.crs

plt.rcParams["font.family"] = "Times New Roman"
fig = plt.figure(figsize=(10, 7), dpi=300)
ax = plt.gca()

im = ax.imshow(Z, extent=extent, origin="upper", cmap="turbo")
ax.scatter(dep_lon, dep_lat, s=6, c="black", alpha=0.7, linewidths=0)

ax.set_title("Proposed Model (Model 17: Rank Blend)", fontsize=12)
ax.set_xlabel("Longitude" if "EPSG:4326" in str(crs) else "X")
ax.set_ylabel("Latitude" if "EPSG:4326" in str(crs) else "Y")

cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Prospectivity score", fontsize=10)

plt.tight_layout()
plt.savefig(OUT_PNG)
plt.close()

print(f"Saved PNG   : {OUT_PNG}")
print(f"Saved WEIGHT: {OUT_TXT}")

#%
import glob
print(glob.glob(r"Outputs\*.tif"))
print(glob.glob(r"Outputs\*p_sup*.tif"))
print(glob.glob(r"Outputs\*stage*.tif"))
print(glob.glob(r"Outputs\*model16*.tif"))
