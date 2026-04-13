"""
77_model17_rankblend_from_model16_FAST.py

MODEL 17 (FAST): Fold-wise optimized blend (stacking layer) on top of Model16++ outputs.

Input  : Outputs/cv_preds_model16_pu_stackmeta.csv
Output : Outputs/cv_preds_model17_rankblend.csv
         Outputs/model17_rankblend_metrics.txt

Why this helps:
- p_sup (Stage3) is your best so far.
- A small, leakage-safe blend with p_y_pu_corrected + Model6 + HDIPP often improves Top-K discovery.
- Runs in seconds (no XGBoost training).

Optimization objective (TRAIN folds only):
  obj = Top10 + 0.50*Top5 + 0.25*Top1   (ranking-focused, BED-friendly)

Notes:
- Uses only numpy/pandas/sklearn metrics.
- Standardizes each score using TRAIN mean/std (no leakage).
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score


# -------------------- PATHS --------------------
IN_CSV  = r"Outputs/cv_preds_model16_pu_stackmeta.csv"
OUT_DIR = "Outputs"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_CSV = os.path.join(OUT_DIR, "cv_preds_model17_rankblend.csv")
OUT_TXT = os.path.join(OUT_DIR, "model17_rankblend_metrics.txt")


# -------------------- SETTINGS --------------------
SEED = 42

# Which Model16 columns to blend (must exist in the CSV)
SCORE_COLS = [
    "p_sup",               # Stage3 (best)
    "p_y_pu_corrected",    # Stage2
    "score_model6",        # Model6 tuned
    "score_hdipp"          # HDIPP OOF
]

# Weight search grids (coarse + fast)
# We enforce: w_sup + w_pu + w_m6 + w_hd = 1 and all >= 0
W_SUP_GRID = np.linspace(0.55, 0.95, 21)   # keep p_sup dominant
W_M6_GRID  = np.linspace(0.00, 0.25, 6)
W_HD_GRID  = np.linspace(0.00, 0.20, 5)

# For speed: optimize weights on a subset (all positives + N negatives) of TRAIN
SUBSAMPLE_NEG = 12000


# -------------------- HELPERS --------------------
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
    return (train_vec - mu) / sd, (test_vec - mu) / sd

def make_train_subset_idx(y_train, rng, n_neg=12000):
    y_train = np.asarray(y_train, dtype=int)
    pos_idx = np.where(y_train == 1)[0]
    neg_idx = np.where(y_train == 0)[0]

    if len(neg_idx) > n_neg:
        neg_idx = rng.choice(neg_idx, size=n_neg, replace=False)

    return np.sort(np.r_[pos_idx, neg_idx])


# -------------------- MAIN --------------------
def main():
    df = pd.read_csv(IN_CSV)

    required = {"fold_id", "y"} | set(SCORE_COLS)
    missing = sorted(list(required - set(df.columns)))
    if missing:
        raise ValueError(f"Missing required columns in input CSV: {missing}")

    df = df.copy()
    df["fold_id"] = df["fold_id"].astype(int)
    df["y"] = df["y"].astype(int)

    # Fill any NaN in scores safely (rare)
    for c in SCORE_COLS:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())

    folds = sorted(df["fold_id"].unique().tolist())
    print("Folds:", folds)

    rng = np.random.default_rng(SEED)

    # OOF output
    oof_score = np.zeros(len(df), dtype=float)
    best_weights = {}

    for k in folds:
        tr_mask = (df["fold_id"].values != k)
        te_mask = ~tr_mask

        y_tr = df.loc[tr_mask, "y"].values
        y_te = df.loc[te_mask, "y"].values

        # Standardize each score column using TRAIN only (no leakage)
        Z_tr_list = []
        Z_te_list = []
        for c in SCORE_COLS:
            trv = df.loc[tr_mask, c].values.astype(float)
            tev = df.loc[te_mask, c].values.astype(float)
            ztr, zte = standardize_train_apply(trv, tev)
            Z_tr_list.append(ztr)
            Z_te_list.append(zte)

        Z_tr = np.column_stack(Z_tr_list)  # shape (n_tr, 4)
        Z_te = np.column_stack(Z_te_list)  # shape (n_te, 4)

        # Subset for fast weight tuning (TRAIN only)
        sub_idx = make_train_subset_idx(y_tr, rng, n_neg=SUBSAMPLE_NEG)
        Z_sub = Z_tr[sub_idx]
        y_sub = y_tr[sub_idx]

        best_obj = -1e9
        best_w = None

        # Grid search (fast)
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

        best_weights[k] = best_w

        # Apply best weights to TEST fold
        oof_score[te_mask] = Z_te @ best_w

        print(f"[fold {k}] best_w = {best_w}  obj={best_obj:.4f}")

    # Convert scores to [0,1] for convenience (monotone; keeps ranking)
    # (Only for nicer output; ROC/PR are invariant to monotone transforms for ranking.)
    s = oof_score
    s01 = (s - np.min(s)) / (np.max(s) - np.min(s) + 1e-12)

    y_all = df["y"].values.astype(int)

    roc = roc_auc_score(y_all, s01)
    pr  = average_precision_score(y_all, s01)
    t1  = topk_recall(y_all, s01, 0.01)
    t5  = topk_recall(y_all, s01, 0.05)
    t10 = topk_recall(y_all, s01, 0.10)

    out = df[["lon","lat","block_id","fold_id","y"]].copy() if all(c in df.columns for c in ["lon","lat","block_id"]) else df[["fold_id","y"]].copy()
    out["score_model17_rankblend"] = s01

    # also keep components for traceability
    for c in SCORE_COLS:
        out[c] = df[c].values

    out.to_csv(OUT_CSV, index=False)

    # Write metrics + weights
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("Model17: Fold-wise optimized rank blend on top of Model16++ outputs\n")
        f.write("Inputs blended: " + ", ".join(SCORE_COLS) + "\n")
        f.write(f"SEED={SEED}, SUBSAMPLE_NEG={SUBSAMPLE_NEG}\n")
        f.write("Objective(TRAIN): Top10 + 0.50*Top5 + 0.25*Top1\n\n")
        f.write(f"ROC-AUC: {roc:.3f}\n")
        f.write(f"PR-AUC : {pr:.4f}\n")
        f.write(f"Top-1% recall : {t1:.3f}\n")
        f.write(f"Top-5% recall : {t5:.3f}\n")
        f.write(f"Top-10% recall: {t10:.3f}\n\n")
        f.write("Best weights per fold (w_sup, w_pu, w_m6, w_hdipp):\n")
        for k in sorted(best_weights.keys()):
            f.write(f"  fold {k}: {best_weights[k].tolist()}\n")

    print("Saved:", OUT_CSV)
    print("Saved:", OUT_TXT)
    print(f"[OOF] ROC={roc:.3f}  PR={pr:.4f}  Top10={t10:.3f}")


if __name__ == "__main__":
    main()
