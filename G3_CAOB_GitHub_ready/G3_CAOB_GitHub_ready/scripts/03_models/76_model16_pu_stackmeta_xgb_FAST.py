"""
76_model16_pu_stackmeta_xgb_FAST.py

Model16++ (FAST): Effort-calibrated PU + Supervised + Meta-stacker (OOF)

Works even if Model6 OOF CSV has NO lon/lat:
- If len(Model6)==len(dep)+len(bg): uses row-order alignment (dep then bg)
- Else tries rounded lon/lat merge

Inputs:
  - Outputs/deposits_covariates_v2.csv
  - Outputs/background_covariates_v2.csv
  - Outputs/cv_preds_xgb_model6.csv
  - Outputs/cv_preds_hdipp.csv

Constraints:
  - NEVER use E4_KDE_ALLMIN or E5_DIST_MAJOR
"""

import os
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss

# -------------------- PATHS --------------------
DEP_CSV = r"Outputs/deposits_covariates_v2.csv"
BG_CSV  = r"Outputs/background_covariates_v2.csv"

MODEL6_OOF = r"Outputs/cv_preds_xgb_model6.csv"
HDIPP_OOF  = r"Outputs/cv_preds_hdipp.csv"

OUT_DIR = "Outputs"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_PRED = os.path.join(OUT_DIR, "cv_preds_model16_pu_stackmeta.csv")
OUT_TXT  = os.path.join(OUT_DIR, "model16_pu_stackmeta_metrics.txt")

# -------------------- SETTINGS --------------------
SEED = 42
ROUND_DECIMALS = 6
BLOCK_KM = 50.0

EXCLUDE_EFFORT_FEATURES = {"E4_KDE_ALLMIN", "E5_DIST_MAJOR"}
EFFORT_FEATURES = ["E1_DIST_ROAD", "E2_POP", "E3_VIIRS_2018"]

GEOLOGY_BASE = [
    "G1_DEM", "G1_SLOPE", "G1_TRI",
    "G2_LITHOLOGY8",
    "G3_DIST_FAULT", "G4_ASTER_PHYLLIC", "G5_L8_iron",
]

EPS = 1e-4
WEIGHT_CAP_Q = 0.99
N_JOBS = 4

PROP_XGB_PARAMS = dict(
    n_estimators=500, learning_rate=0.05, max_depth=4, min_child_weight=5,
    subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0, gamma=0.0,
)
PU_XGB_PARAMS = dict(
    n_estimators=700, learning_rate=0.05, max_depth=5, min_child_weight=5,
    subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0, gamma=0.0,
)
SUP_XGB_PARAMS = dict(
    n_estimators=700, learning_rate=0.05, max_depth=5, min_child_weight=5,
    subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0, gamma=0.0,
)
META_XGB_PARAMS = dict(
    n_estimators=500, learning_rate=0.05, max_depth=3, min_child_weight=5,
    subsample=0.9, colsample_bytree=0.9, reg_lambda=2.0, gamma=0.0,
)

# -------------------- HELPERS --------------------
def topk_recall(y_true, y_score, frac=0.10):
    n = len(y_true)
    k = max(1, int(frac * n))
    idx = np.argsort(-y_score)[:k]
    denom = max(1, int(np.sum(y_true)))
    return float(np.sum(y_true[idx]) / denom)

def get_lon_lat(df: pd.DataFrame):
    if "lon" in df.columns and "lat" in df.columns:
        return df["lon"].astype(float).values, df["lat"].astype(float).values
    if "LONGITUDE" in df.columns and "LATITUDE" in df.columns:
        return df["LONGITUDE"].astype(float).values, df["LATITUDE"].astype(float).values
    raise ValueError("No lon/lat columns found")

def make_spatial_blocks(lon, lat, block_km=50.0):
    block_deg = block_km / 111.0
    bx = np.floor(lon / block_deg).astype(int)
    by = np.floor(lat / block_deg).astype(int)
    return bx * 10_000 + by

def make_folds_from_blocks(block_id, n_folds=5, seed=SEED):
    uniq = np.unique(block_id)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    folds = np.zeros_like(block_id, dtype=int)
    for i, b in enumerate(uniq):
        folds[block_id == b] = i % n_folds
    return folds

def median_impute_inplace(X: pd.DataFrame) -> pd.DataFrame:
    for c in X.columns:
        med = float(np.nanmedian(X[c].values)) if np.any(~np.isnan(X[c].values)) else 0.0
        X[c] = X[c].fillna(med)
    return X

def build_design(dep: pd.DataFrame, bg: pd.DataFrame):
    drop_like = {"fid", "id", "index", "lon", "lat", "LONGITUDE", "LATITUDE"}
    shared = [c for c in dep.columns if c in bg.columns]

    feats = []
    for c in shared:
        if c in drop_like:
            continue
        if c in EXCLUDE_EFFORT_FEATURES:
            continue
        if c.lower().startswith("name"):
            continue
        if c == "G2_LITHOLOGY8":
            feats.append(c)
        elif pd.api.types.is_numeric_dtype(dep[c]) or pd.api.types.is_numeric_dtype(bg[c]):
            feats.append(c)

    X_dep = dep[feats].copy()
    X_bg  = bg[feats].copy()

    for c in feats:
        if c == "G2_LITHOLOGY8":
            continue
        X_dep[c] = pd.to_numeric(X_dep[c], errors="coerce")
        X_bg[c]  = pd.to_numeric(X_bg[c], errors="coerce")

    X_all = pd.concat([X_dep, X_bg], ignore_index=True)
    y_all = np.r_[np.ones(len(X_dep), dtype=int), np.zeros(len(X_bg), dtype=int)]

    X_all = median_impute_inplace(X_all)

    if "G2_LITHOLOGY8" in X_all.columns:
        X_all["G2_LITHOLOGY8"] = X_all["G2_LITHOLOGY8"].astype("Int64").astype(str)
        X_all = pd.get_dummies(X_all, columns=["G2_LITHOLOGY8"], drop_first=False)

    dep_lon, dep_lat = get_lon_lat(dep)
    bg_lon, bg_lat   = get_lon_lat(bg)

    lon_all = np.r_[dep_lon, bg_lon]
    lat_all = np.r_[dep_lat, bg_lat]

    return X_all, y_all, lon_all, lat_all

def pick_score_col(df, prefer_contains=None):
    if prefer_contains:
        for c in df.columns:
            if prefer_contains.lower() in c.lower() and "score" in c.lower():
                return c
    for c in ["score_xgb_model6", "score_xgb_model6_hybrid", "score_model6", "score"]:
        if c in df.columns:
            return c
    # fallback: any column starting with score
    cands = [c for c in df.columns if c.lower().startswith("score")]
    if cands:
        return cands[0]
    raise ValueError(f"Could not find a score column. Columns={list(df.columns)}")

def try_merge_by_lonlat(base_df, score_df, score_col):
    if ("lon" not in score_df.columns) or ("lat" not in score_df.columns):
        return None
    s = score_df.copy()
    s["lon_r"] = s["lon"].astype(float).round(ROUND_DECIMALS)
    s["lat_r"] = s["lat"].astype(float).round(ROUND_DECIMALS)
    s = s.groupby(["lon_r", "lat_r"], as_index=False)[score_col].mean()
    out = base_df.merge(s, on=["lon_r", "lat_r"], how="left")
    return out

def fit_xgb_basic(Xtr, ytr, params, sample_weight=None, seed=SEED):
    neg = int((ytr == 0).sum())
    pos = int((ytr == 1).sum())
    spw = max(1.0, neg / max(1, pos))
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=N_JOBS,
        random_state=seed,
        scale_pos_weight=spw,
        **params
    )
    model.fit(Xtr, ytr, sample_weight=sample_weight, verbose=False)
    return model

def summarize(y, score):
    roc = roc_auc_score(y, score)
    ap  = average_precision_score(y, score)
    ll  = log_loss(y, np.clip(score, 1e-6, 1-1e-6))
    t1  = topk_recall(y, score, 0.01)
    t5  = topk_recall(y, score, 0.05)
    t10 = topk_recall(y, score, 0.10)
    return roc, ap, ll, t1, t5, t10

# -------------------- MAIN --------------------
def main():
    dep = pd.read_csv(DEP_CSV)
    bg  = pd.read_csv(BG_CSV)

    X_all_df, y_all, lon_all, lat_all = build_design(dep, bg)

    base = pd.DataFrame({"lon": lon_all, "lat": lat_all, "y": y_all})
    base["lon_r"] = base["lon"].round(ROUND_DECIMALS)
    base["lat_r"] = base["lat"].round(ROUND_DECIMALS)

    # ---- Load Model6 OOF ----
    if not os.path.exists(MODEL6_OOF):
        raise FileNotFoundError(f"Missing {MODEL6_OOF}. Run Model6 first.")
    m6 = pd.read_csv(MODEL6_OOF)
    m6_col = pick_score_col(m6, prefer_contains="model6")

    # Prefer: row-order alignment if lengths match (common in your pipeline)
    if len(m6) == len(base):
        base["score_model6"] = m6[m6_col].astype(float).values
        print("[CONFIRM] Model6 aligned by ROW ORDER (len match). score col:", m6_col)
    else:
        merged = try_merge_by_lonlat(base, m6, m6_col)
        if merged is None:
            raise ValueError(
                "Model6 merge failed: Model6 file has no lon/lat and length != dep+bg.\n"
                f"len(model6)={len(m6)} vs len(data)={len(base)}\n"
                f"Model6 columns: {list(m6.columns)}"
            )
        base = merged
        if base[m6_col].isna().all():
            raise ValueError("Model6 lon/lat merge produced all-NaN. Check coordinate consistency.")
        base["score_model6"] = base[m6_col].astype(float)
        base = base.drop(columns=[m6_col])
        print("[CONFIRM] Model6 merged by lon/lat. score col:", m6_col)

    # ---- Load HDIPP OOF ----
    if not os.path.exists(HDIPP_OOF):
        raise FileNotFoundError(f"Missing {HDIPP_OOF}. Run HDIPP first.")
    h = pd.read_csv(HDIPP_OOF)
    if "score_hdipp" not in h.columns:
        raise ValueError("cv_preds_hdipp.csv must contain score_hdipp")

    merged_h = try_merge_by_lonlat(base, h, "score_hdipp")
    if merged_h is None:
        raise ValueError("HDIPP merge failed: cv_preds_hdipp.csv must contain lon/lat.")
    base = merged_h
    # fill missing
    base["score_hdipp"] = base["score_hdipp"].fillna(base["score_hdipp"].mean())
    base["score_model6"] = base["score_model6"].fillna(base["score_model6"].mean())

    # ---- Folds ----
    block_id = make_spatial_blocks(base["lon"].values, base["lat"].values, block_km=BLOCK_KM)
    folds = make_folds_from_blocks(block_id, n_folds=5, seed=SEED)

    print("N:", len(base), "positives:", int(base["y"].sum()))
    print("[CONFIRM] HDIPP range:", float(base["score_hdipp"].min()), "..", float(base["score_hdipp"].max()))

    # ---- Stage 1: propensity model on E1-E3 ----
    for c in EFFORT_FEATURES:
        if c not in X_all_df.columns:
            raise ValueError(f"Missing effort feature {c}")

    X_prop = X_all_df[EFFORT_FEATURES].values.astype(np.float32)
    s_hat = np.zeros(len(base), dtype=float)

    for f in range(5):
        tr = folds != f
        te = folds == f
        model_prop = fit_xgb_basic(X_prop[tr], y_all[tr], PROP_XGB_PARAMS, sample_weight=None)
        s_hat[te] = model_prop.predict_proba(X_prop[te])[:, 1]

    s_hat = np.clip(s_hat, EPS, 1.0 - EPS)

    # ---- Stage 2: geology + model6 + hdipp ----
    geo_cols = []
    for c in GEOLOGY_BASE:
        if c == "G2_LITHOLOGY8":
            geo_cols.extend([col for col in X_all_df.columns if col.startswith("G2_LITHOLOGY8_")])
        else:
            if c in X_all_df.columns:
                geo_cols.append(c)
    geo_cols = list(dict.fromkeys(geo_cols))
    if not geo_cols:
        raise ValueError("No geology columns found.")

    X_geo_df = X_all_df[geo_cols].copy()
    X_geo_df["score_model6"] = base["score_model6"].values.astype(float)
    X_geo_df["score_hdipp"]  = base["score_hdipp"].values.astype(float)
    X_geo = X_geo_df.values.astype(np.float32)

    # PU weights
    w_base = np.ones(len(base), dtype=float)
    w_base[y_all == 0] = 1.0 / s_hat[y_all == 0]

    p_s_pu = np.zeros(len(base), dtype=float)
    p_y_pu = np.zeros(len(base), dtype=float)
    c_folds = []

    for f in range(5):
        tr = folds != f
        te = folds == f

        wtr = w_base[tr].copy()
        u_tr = (y_all[tr] == 0)
        if u_tr.any():
            cap = float(np.quantile(wtr[u_tr], WEIGHT_CAP_Q))
            wtr[u_tr] = np.minimum(wtr[u_tr], cap)
        wtr = wtr.astype(np.float32)

        model_pu = fit_xgb_basic(X_geo[tr], y_all[tr], PU_XGB_PARAMS, sample_weight=wtr)
        p_te = model_pu.predict_proba(X_geo[te])[:, 1]
        p_s_pu[te] = p_te

        pos_tr = (y_all[tr] == 1)
        if pos_tr.sum() == 0:
            c_hat = 1.0
        else:
            p_pos = model_pu.predict_proba(X_geo[tr][pos_tr])[:, 1]
            c_hat = max(float(np.mean(p_pos)), 1e-3)
        c_folds.append(c_hat)
        p_y_pu[te] = np.clip(p_te / c_hat, 0.0, 1.0)

    # ---- Stage 3: supervised model ----
    p_sup = np.zeros(len(base), dtype=float)
    for f in range(5):
        tr = folds != f
        te = folds == f
        model_sup = fit_xgb_basic(X_geo[tr], y_all[tr], SUP_XGB_PARAMS, sample_weight=None)
        p_sup[te] = model_sup.predict_proba(X_geo[te])[:, 1]

    # ---- Stage 4: meta model ----
    X_meta = np.c_[
        p_y_pu.astype(np.float32),
        p_sup.astype(np.float32),
        base["score_model6"].values.astype(np.float32),
        base["score_hdipp"].values.astype(np.float32),
    ]
    p_meta = np.zeros(len(base), dtype=float)
    for f in range(5):
        tr = folds != f
        te = folds == f
        model_meta = fit_xgb_basic(X_meta[tr], y_all[tr], META_XGB_PARAMS, sample_weight=None)
        p_meta[te] = model_meta.predict_proba(X_meta[te])[:, 1]

    # ---- Metrics ----
    roc_s, ap_s, _, _, _, _ = summarize(y_all, s_hat)
    roc_pu, ap_pu, ll_pu, t1_pu, t5_pu, t10_pu = summarize(y_all, p_y_pu)
    roc_sup, ap_sup, ll_sup, t1_sup, t5_sup, t10_sup = summarize(y_all, p_sup)
    roc_m, ap_m, ll_m, t1_m, t5_m, t10_m = summarize(y_all, p_meta)

    # ---- Save ----
    out = pd.DataFrame({
        "lon": base["lon"].values,
        "lat": base["lat"].values,
        "block_id": block_id.astype(int),
        "fold_id": folds.astype(int),
        "y": y_all.astype(int),
        "score_model6": base["score_model6"].values.astype(float),
        "score_hdipp": base["score_hdipp"].values.astype(float),
        "s_hat_propensity_E123": s_hat.astype(float),
        "p_s_pu": p_s_pu.astype(float),
        "p_y_pu_corrected": p_y_pu.astype(float),
        "p_sup": p_sup.astype(float),
        "p_model16_meta": p_meta.astype(float),
    })
    out.to_csv(OUT_PRED, index=False)

    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("Model16++: Propensity(E1-3) + PU + Supervised + Meta (OOF)\n")
        f.write(f"seed={SEED}, block_km={BLOCK_KM}, round_decimals={ROUND_DECIMALS}\n")
        f.write("Constraints: E4_KDE_ALLMIN and E5_DIST_MAJOR excluded.\n\n")
        f.write(f"[Stage1] s_hat ROC={roc_s:.3f} PR={ap_s:.4f}\n")
        f.write(f"[Stage2] PU  ROC={roc_pu:.3f} PR={ap_pu:.4f} LL={ll_pu:.4f} Top10={t10_pu:.3f}\n")
        f.write(f"[Stage3] SUP ROC={roc_sup:.3f} PR={ap_sup:.4f} LL={ll_sup:.4f} Top10={t10_sup:.3f}\n")
        f.write(f"[Stage4] META ROC={roc_m:.3f} PR={ap_m:.4f} LL={ll_m:.4f} Top10={t10_m:.3f}\n")
        f.write(f"c_hat folds: {c_folds}\n\n")
        f.write(f"Saved preds: {OUT_PRED}\n")
        f.write(f"Saved txt  : {OUT_TXT}\n")

    print("Saved:", OUT_PRED)
    print("Saved:", OUT_TXT)
    print("[META] ROC=%.3f PR=%.4f Top10=%.3f" % (roc_m, ap_m, t10_m))

if __name__ == "__main__":
    main()
