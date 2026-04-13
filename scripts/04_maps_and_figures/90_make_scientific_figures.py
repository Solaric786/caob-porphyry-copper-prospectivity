import os, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

def topk_recall(y_true, y_score, frac):
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    n = len(y_true)
    k = max(1, int(frac * n))
    idx = np.argpartition(-y_score, k - 1)[:k]
    denom = max(1, int(y_true.sum()))
    return float(y_true[idx].sum() / denom)

def safe_score(arr):
    s = np.asarray(arr, dtype=float)
    if np.isnan(s).any():
        s = np.where(np.isnan(s), np.nanmean(s), s)
    return s

def first_existing(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None

def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"None of score columns found. Tried {candidates}. Have {list(df.columns)}")

def fold_metrics(df, ycol, scol, foldcol="fold_id"):
    rows = []
    if foldcol not in df.columns:
        return pd.DataFrame()
    for k, g in df.groupby(foldcol):
        y = g[ycol].values.astype(int)
        if len(np.unique(y)) < 2:
            continue
        s = safe_score(g[scol].values)
        rows.append({
            "fold_id": int(k),
            "roc_auc": float(roc_auc_score(y, s)),
            "pr_auc": float(average_precision_score(y, s)),
            "top1": float(topk_recall(y, s, 0.01)),
            "top5": float(topk_recall(y, s, 0.05)),
            "top10": float(topk_recall(y, s, 0.10)),
        })
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # --- locate your CSVs inside FINAL_NAMED (with fallbacks) ---
    in_dir = args.in_dir

    hdipp_csv = first_existing([
        os.path.join(in_dir, "cv_preds_hdipp.csv"),
        os.path.join(in_dir, "Model02_HDIPP_Baseline_cv_preds.csv"),
        os.path.join("Outputs", "cv_preds_hdipp.csv"),
    ])
    m6_csv = first_existing([
        os.path.join(in_dir, "cv_preds_xgb_model6.csv"),
        os.path.join(in_dir, "Model06_XGB_Tuned_Baseline_cv_preds.csv"),
        os.path.join("Outputs", "cv_preds_xgb_model6.csv"),
    ])
    m16_csv = first_existing([
        os.path.join(in_dir, "cv_preds_model16_pu_stackmeta.csv"),
        os.path.join(in_dir, "Model16_PU_StackMeta_cv_preds.csv"),
        os.path.join("Outputs", "cv_preds_model16_pu_stackmeta.csv"),
    ])
    m17_csv = first_existing([
        os.path.join(in_dir, "cv_preds_model17_rankblend.csv"),
        os.path.join(in_dir, "Model17_RankBlend_cv_preds.csv"),
        os.path.join("Outputs", "cv_preds_model17_rankblend.csv"),
    ])

    for name, p in [("HDIPP", hdipp_csv), ("Model6", m6_csv), ("Model16", m16_csv), ("Model17", m17_csv)]:
        if p is None:
            raise SystemExit(f"Missing {name} CSV in {in_dir} (or Outputs).")

    df_hd = pd.read_csv(hdipp_csv)
    df_m6 = pd.read_csv(m6_csv)
    df_16 = pd.read_csv(m16_csv)
    df_17 = pd.read_csv(m17_csv)

    # columns (robust)
    y_hd  = pick_col(df_hd, ["y", "Y"])
    s_hd  = pick_col(df_hd, ["score_hdipp", "eta_hdipp", "hdipp_score"])

    y_m6  = pick_col(df_m6, ["y", "Y"])
    s_m6  = pick_col(df_m6, ["score_xgb_model6", "score_model6", "p_model6"])

    y_16  = pick_col(df_16, ["y", "Y"])
    s_16a = pick_col(df_16, ["p_y_pu_corrected", "p_pu_corrected", "p_pu"])
    s_16b = pick_col(df_16, ["p_sup"])

    y_17  = pick_col(df_17, ["y", "Y"])
    s_17  = pick_col(df_17, ["score_model17_rankblend", "score_model17", "score_ro_we", "score_rowe"])

    # ORIGINAL model names for legends
    models = [
        ("Model02_HDIPP_Baseline", df_hd, y_hd, s_hd),
        ("Model06_XGB_Tuned_Baseline", df_m6, y_m6, s_m6),
        ("Model16_PU_StackMeta_p_y_pu_corrected", df_16, y_16, s_16a),
        ("Model16_PU_StackMeta_p_sup", df_16, y_16, s_16b),
        ("Model17_ROWE", df_17, y_17, s_17),
    ]

    # Paper style
    plt.rcParams["font.family"] = "Times New Roman"

    # ---- ROC ----
    fig = plt.figure(figsize=(7.6, 6.2))
    ax = plt.gca()
    for name, df, ycol, scol in models:
        y = df[ycol].values.astype(int)
        s = safe_score(df[scol].values)
        fpr, tpr, _ = roc_curve(y, s)
        auc = roc_auc_score(y, s)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0,1],[0,1], linestyle="--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (Out-of-Fold)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "Fig_ROC_OOF.png"), dpi=300)
    plt.close(fig)

    # ---- PR ----
    fig = plt.figure(figsize=(7.6, 6.2))
    ax = plt.gca()
    for name, df, ycol, scol in models:
        y = df[ycol].values.astype(int)
        s = safe_score(df[scol].values)
        prec, rec, _ = precision_recall_curve(y, s)
        apv = average_precision_score(y, s)
        ax.plot(rec, prec, label=f"{name} (AP={apv:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curves (Out-of-Fold)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "Fig_PR_OOF.png"), dpi=300)
    plt.close(fig)

    # ---- Top-K recall (1%..20%) ----
    ks = np.arange(0.01, 0.201, 0.01)
    fig = plt.figure(figsize=(7.6, 6.2))
    ax = plt.gca()
    for name, df, ycol, scol in models:
        y = df[ycol].values.astype(int)
        s = safe_score(df[scol].values)
        vals = [topk_recall(y, s, float(k)) for k in ks]
        ax.plot(ks*100.0, vals, label=name)
    ax.set_xlabel("Top-K area (%)")
    ax.set_ylabel("Recall of known deposits")
    ax.set_title("Top-K Recall Curves (Out-of-Fold)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "Fig_TopKRecall_OOF.png"), dpi=300)
    plt.close(fig)

    # ---- Tables ----
    rows = []
    fold_rows = []
    for name, df, ycol, scol in models:
        y = df[ycol].values.astype(int)
        s = safe_score(df[scol].values)
        rows.append({
            "model": name,
            "roc_auc": float(roc_auc_score(y, s)),
            "pr_auc": float(average_precision_score(y, s)),
            "top1": float(topk_recall(y, s, 0.01)),
            "top5": float(topk_recall(y, s, 0.05)),
            "top10": float(topk_recall(y, s, 0.10)),
            "n": int(len(y)),
            "positives": int(y.sum())
        })
        fm = fold_metrics(df, ycol, scol, "fold_id")
        if len(fm):
            fm["model"] = name
            fold_rows.append(fm)

    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(args.out_dir, "Table_Metrics_OOF.csv"), index=False)

    if fold_rows:
        folds_all = pd.concat(fold_rows, ignore_index=True)
        folds_all.to_csv(os.path.join(args.out_dir, "Table_Metrics_ByFold.csv"), index=False)

    print("Saved to:", args.out_dir)
    print(summary.sort_values("pr_auc", ascending=False).to_string(index=False))

if __name__ == "__main__":
    main()
