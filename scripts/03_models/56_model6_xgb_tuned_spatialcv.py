import os
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

DEP_CSV = r"Outputs/deposits_covariates_v2.csv"
BG_CSV  = r"Outputs/background_covariates_v2.csv"

OUT_DIR = "Outputs"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_PRED = os.path.join(OUT_DIR, "cv_preds_xgb_model6.csv")
OUT_TXT  = os.path.join(OUT_DIR, "model6_xgb_metrics.txt")

BLOCK_KM = 50.0
RANDOM_TRIALS = 30
SEED = 42

def get_lon_lat(df):
    if "lon" in df.columns and "lat" in df.columns:
        return df["lon"].astype(float).values, df["lat"].astype(float).values
    if "LONGITUDE" in df.columns and "LATITUDE" in df.columns:
        return df["LONGITUDE"].astype(float).values, df["LATITUDE"].astype(float).values
    raise ValueError("No lon/lat columns found")

def make_spatial_blocks(lon, lat, block_km=50.0):
    # Simple block-id using degrees (OK for your CAOB window scale; keeps consistency with your earlier script concept)
    # If you want exact meters, we can switch to pyproj projection.
    block_deg = block_km / 111.0
    bx = np.floor(lon / block_deg).astype(int)
    by = np.floor(lat / block_deg).astype(int)
    block_id = bx * 10_000 + by
    return block_id

def build_xy(dep, bg):
    # Keep all numeric shared columns except obvious id/coords
    drop_like = set(["fid","id","index","lon","lat","LONGITUDE","LATITUDE"])
    shared = [c for c in dep.columns if c in bg.columns]
    feats = []
    for c in shared:
        if c in drop_like:
            continue
        if c.lower().startswith("name"):
            continue
        # keep numeric only
        if pd.api.types.is_numeric_dtype(dep[c]) or pd.api.types.is_numeric_dtype(bg[c]):
            feats.append(c)

    X_dep = dep[feats].apply(pd.to_numeric, errors="coerce")
    X_bg  = bg[feats].apply(pd.to_numeric, errors="coerce")

    X = pd.concat([X_dep, X_bg], ignore_index=True)
    y = np.r_[np.ones(len(X_dep), dtype=int), np.zeros(len(X_bg), dtype=int)]

    # median impute
    for c in X.columns:
        med = float(np.nanmedian(X[c].values))
        X[c] = X[c].fillna(med)

    # one-hot lithology if present
    if "G2_LITHOLOGY8" in X.columns:
        X["G2_LITHOLOGY8"] = X["G2_LITHOLOGY8"].astype(int).astype(str)
        X = pd.get_dummies(X, columns=["G2_LITHOLOGY8"], drop_first=False)

    return X, y, feats, list(X.columns)

def eval_one_params(X, y, block_id, params):
    # group blocks into 5 folds (deterministic)
    uniq = np.unique(block_id)
    rng = np.random.default_rng(SEED)
    rng.shuffle(uniq)
    folds = np.zeros_like(block_id, dtype=int)
    for i, b in enumerate(uniq):
        folds[block_id == b] = i % 5

    fold_scores = []
    fold_ap = []

    oof = np.zeros(len(y), dtype=float)

    for f in range(5):
        tr = folds != f
        te = folds == f

        Xtr, ytr = X[tr], y[tr]
        Xte, yte = X[te], y[te]

        neg = int((ytr == 0).sum())
        pos = int((ytr == 1).sum())
        spw = max(1.0, neg / max(1, pos))

        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=SEED,
            scale_pos_weight=spw,
            **params
        )

        # early stopping
        model.fit(
            Xtr, ytr,
            eval_set=[(Xte, yte)],
            verbose=False
        )

        s = model.predict_proba(Xte)[:, 1]
        oof[te] = s

        if len(np.unique(yte)) > 1:
            fold_scores.append(roc_auc_score(yte, s))
            fold_ap.append(average_precision_score(yte, s))

    return float(np.mean(fold_scores)), float(np.std(fold_scores)), float(np.mean(fold_ap)), float(np.std(fold_ap)), oof, folds

# Load
dep = pd.read_csv(DEP_CSV)
bg  = pd.read_csv(BG_CSV)

dep_lon, dep_lat = get_lon_lat(dep)
bg_lon,  bg_lat  = get_lon_lat(bg)

X, y, raw_feats, Xcols = build_xy(dep, bg)

# spatial blocks from coordinates (combine dep+bg)
lon_all = np.r_[dep_lon, bg_lon]
lat_all = np.r_[dep_lat, bg_lat]
block_id = make_spatial_blocks(lon_all, lat_all, block_km=BLOCK_KM)

Xnp = X.values.astype(np.float32)

# Random search space (fast + effective)
rng = np.random.default_rng(SEED)
best = None

for t in range(RANDOM_TRIALS):
    params = {
        "n_estimators": int(rng.integers(600, 1800)),
        "learning_rate": float(rng.choice([0.03, 0.05, 0.08])),
        "max_depth": int(rng.choice([4, 5, 6])),
        "min_child_weight": float(rng.choice([1, 5, 10])),
        "subsample": float(rng.choice([0.7, 0.85, 1.0])),
        "colsample_bytree": float(rng.choice([0.7, 0.85, 1.0])),
        "reg_lambda": float(rng.choice([0.5, 1.0, 2.0])),
        "gamma": float(rng.choice([0.0, 0.5, 1.0])),
    }

    roc_m, roc_s, ap_m, ap_s, _, _ = eval_one_params(Xnp, y, block_id, params)

    if (best is None) or (roc_m > best["roc_mean"]):
        best = {"params": params, "roc_mean": roc_m, "roc_std": roc_s, "ap_mean": ap_m, "ap_std": ap_s}
        print(f"[best@{t}] ROC={roc_m:.3f}±{roc_s:.3f}  PR={ap_m:.4f}±{ap_s:.4f}")

print("\nBEST PARAMS:", best["params"])
roc_m, roc_s, ap_m, ap_s, oof, folds = eval_one_params(Xnp, y, block_id, best["params"])

# Save preds
out = pd.DataFrame({"fold_id": folds, "y": y, "score_xgb_model6": oof})
out.to_csv(OUT_PRED, index=False)

with open(OUT_TXT, "w", encoding="utf-8") as f:
    f.write(f"Model6 XGBoost tuned, block_km={BLOCK_KM}\n")
    f.write(f"ROC-AUC: {roc_m:.3f} ± {roc_s:.3f}\n")
    f.write(f"PR-AUC : {ap_m:.4f} ± {ap_s:.4f}\n")
    f.write(f"Best params: {best['params']}\n")

print("\nSaved:", OUT_PRED)
print("Saved:", OUT_TXT)
print(f"ROC-AUC: {roc_m:.3f} ± {roc_s:.3f}")
print(f"PR-AUC : {ap_m:.4f} ± {ap_s:.4f}")
