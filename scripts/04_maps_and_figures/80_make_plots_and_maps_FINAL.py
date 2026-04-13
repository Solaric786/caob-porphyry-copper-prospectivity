import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve, precision_recall_curve,
    roc_auc_score, average_precision_score
)

# ----------------------------
# CONFIG
# ----------------------------
BASE = Path(".")
OUT_DIR = BASE / "Outputs"
FIG_DIR = OUT_DIR / "Figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Put here the models you want to compare.
# Each CSV must contain lon, lat, y, fold_id (block_id optional), and a score column.
MODEL_SPECS = [
    # ("Model12_CatBoost", OUT_DIR / "cv_preds_model12_catboost_hybrid.csv", "score_model12_catboost"),
    # ("Model15_RecallBlend", OUT_DIR / "cv_preds_model15_recallblend.csv", "score_model15_recallblend"),
    ("Model16_PU_SUP", OUT_DIR / "cv_preds_model16_pu_stackmeta.csv", "p_sup"),
    ("Model17_Final_RankBlend", OUT_DIR / "cv_preds_model17_rankblend.csv", "score_model17_rankblend"),
]

# Map settings (fast + clean)
BIN_DEG = 0.05          # ~5km bins (change to 0.02 if you want finer)
DPI = 350
FONT = "Times New Roman"

# ----------------------------
# METRICS
# ----------------------------
def topk_recall(y_true, y_score, frac):
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    n = len(y_true)
    k = max(1, int(frac * n))
    idx = np.argpartition(-y_score, k - 1)[:k]
    denom = max(1, int(y_true.sum()))
    return float(y_true[idx].sum() / denom)

def reliability_bins(y_true, y_score, n_bins=12):
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    bin_id = np.clip(np.digitize(y_score, edges) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = (bin_id == b)
        if m.sum() == 0:
            continue
        rows.append({
            "bin": b,
            "n": int(m.sum()),
            "pred_mean": float(y_score[m].mean()),
            "obs_rate": float(y_true[m].mean()),
        })
    return pd.DataFrame(rows)

def load_model(name, path, score_col):
    if not path.exists():
        print(f"[SKIP] Missing: {path}")
        return None

    df = pd.read_csv(path)
    need = {"y", score_col}
    miss = sorted(list(need - set(df.columns)))
    if miss:
        raise ValueError(f"{name}: missing columns {miss} in {path}")

    # lon/lat optional for curves; required for maps
    for c in ["lon", "lat", "fold_id"]:
        if c not in df.columns:
            df[c] = np.nan

    # clean
    df = df.copy()
    df["y"] = df["y"].astype(int)
    df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
    df = df[np.isfinite(df[score_col])].copy()

    # scale scores to [0,1] (safe for plotting; ranking is what matters)
    s = df[score_col].values.astype(float)
    s01 = (s - s.min()) / (s.max() - s.min() + 1e-12)
    df["_score01"] = s01
    df["_model"] = name
    df["_score_col"] = score_col
    return df

# ----------------------------
# PLOTS: ROC / PR / TOPK / RELIABILITY
# ----------------------------
def plot_roc(models):
    plt.figure()
    plt.rcParams["font.family"] = FONT
    for df in models:
        y = df["y"].values
        s = df["_score01"].values
        fpr, tpr, _ = roc_curve(y, s)
        auc = roc_auc_score(y, s)
        plt.plot(fpr, tpr, label=f"{df['_model'].iloc[0]} (AUC={auc:.3f})")
    plt.plot([0,1],[0,1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (OOF)")
    plt.legend()
    plt.tight_layout()
    out = FIG_DIR / "fig_ROC_models.png"
    plt.savefig(out, dpi=DPI)
    plt.close()
    print("Saved:", out)

def plot_pr(models):
    plt.figure()
    plt.rcParams["font.family"] = FONT
    for df in models:
        y = df["y"].values
        s = df["_score01"].values
        prec, rec, _ = precision_recall_curve(y, s)
        ap = average_precision_score(y, s)
        plt.plot(rec, prec, label=f"{df['_model'].iloc[0]} (AP={ap:.4f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve (OOF)")
    plt.legend()
    plt.tight_layout()
    out = FIG_DIR / "fig_PR_models.png"
    plt.savefig(out, dpi=DPI)
    plt.close()
    print("Saved:", out)

def plot_topk(models):
    fracs = np.array([0.01, 0.02, 0.05, 0.10, 0.15, 0.20])
    plt.figure()
    plt.rcParams["font.family"] = FONT
    for df in models:
        y = df["y"].values
        s = df["_score01"].values
        vals = [topk_recall(y, s, f) for f in fracs]
        plt.plot(fracs * 100, vals, marker="o", label=df["_model"].iloc[0])
    plt.xlabel("Top area fraction (%)")
    plt.ylabel("Deposit recall within top area")
    plt.title("Top-K Recall (Discovery)")
    plt.legend()
    plt.tight_layout()
    out = FIG_DIR / "fig_TopK_models.png"
    plt.savefig(out, dpi=DPI)
    plt.close()
    print("Saved:", out)

def plot_reliability(models):
    plt.figure()
    plt.rcParams["font.family"] = FONT
    for df in models:
        y = df["y"].values
        s = df["_score01"].values
        rb = reliability_bins(y, s, n_bins=12)
        plt.plot(rb["pred_mean"], rb["obs_rate"], marker="o", label=df["_model"].iloc[0])
    plt.plot([0,1],[0,1], linestyle="--")
    plt.xlabel("Predicted probability (binned mean)")
    plt.ylabel("Observed positive rate")
    plt.title("Reliability / Calibration (OOF)")
    plt.legend()
    plt.tight_layout()
    out = FIG_DIR / "fig_Reliability_models.png"
    plt.savefig(out, dpi=DPI)
    plt.close()
    print("Saved:", out)

# ----------------------------
# MAPS (from 1km background grid)
# ----------------------------
def bin_map(df, score_col="_score01", bin_deg=0.05, agg="mean"):
    # require lon/lat
    if df["lon"].isna().all() or df["lat"].isna().all():
        raise ValueError("Map needs lon/lat in the model CSV.")

    lon = df["lon"].values.astype(float)
    lat = df["lat"].values.astype(float)
    s   = df[score_col].values.astype(float)

    lon_min, lon_max = np.nanmin(lon), np.nanmax(lon)
    lat_min, lat_max = np.nanmin(lat), np.nanmax(lat)

    x_edges = np.arange(lon_min, lon_max + bin_deg, bin_deg)
    y_edges = np.arange(lat_min, lat_max + bin_deg, bin_deg)

    # bin index
    xi = np.clip(np.digitize(lon, x_edges) - 1, 0, len(x_edges) - 2)
    yi = np.clip(np.digitize(lat, y_edges) - 1, 0, len(y_edges) - 2)

    grid = np.full((len(y_edges) - 1, len(x_edges) - 1), np.nan, dtype=float)

    # aggregate per cell
    # use mean by default (stable), or max if you want sharp anomalies
    for yb in range(grid.shape[0]):
        m_y = (yi == yb)
        if not m_y.any():
            continue
        for xb in range(grid.shape[1]):
            m = m_y & (xi == xb)
            if not m.any():
                continue
            if agg == "max":
                grid[yb, xb] = float(np.nanmax(s[m]))
            else:
                grid[yb, xb] = float(np.nanmean(s[m]))

    extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]
    return grid, extent

def plot_map(df, title, out_png, bin_deg=0.05):
    plt.figure(figsize=(10, 6))
    plt.rcParams["font.family"] = FONT

    # Use background grid points for map (y==0), overlay deposits (y==1)
    bg = df[df["y"] == 0].copy()
    dep = df[df["y"] == 1].copy()

    grid, extent = bin_map(bg, score_col="_score01", bin_deg=bin_deg, agg="mean")
    plt.imshow(grid, origin="lower", extent=extent, aspect="auto")
    plt.colorbar(label="Prospectivity score")

    # overlay deposits
    if len(dep) > 0:
        plt.scatter(dep["lon"], dep["lat"], s=10, marker="x", linewidths=0.8)

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=DPI)
    plt.close()
    print("Saved:", out_png)

def plot_topzones(df, title, out_png, frac=0.10):
    """
    Show only the top frac area from BACKGROUND grid as a mask,
    and overlay deposits to visualize how many fall in top zone.
    """
    plt.figure(figsize=(10, 6))
    plt.rcParams["font.family"] = FONT

    bg = df[df["y"] == 0].copy()
    dep = df[df["y"] == 1].copy()

    # threshold by background quantile (area-based)
    thr = float(np.quantile(bg["_score01"].values, 1.0 - frac))
    top_bg = bg[bg["_score01"] >= thr]

    plt.scatter(bg["lon"], bg["lat"], s=1, alpha=0.05)
    plt.scatter(top_bg["lon"], top_bg["lat"], s=3, alpha=0.6)

    if len(dep) > 0:
        plt.scatter(dep["lon"], dep["lat"], s=18, marker="x", linewidths=0.9)

    # compute recall in top zone (area-defined)
    # deposits counted if their score >= threshold (same model)
    dep_in = float((dep["_score01"].values >= thr).mean()) if len(dep) else np.nan

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(f"{title}\nTop {int(frac*100)}% zone (thr={thr:.3f}) | deposit-hit-rate={dep_in:.3f}")
    plt.tight_layout()
    plt.savefig(out_png, dpi=DPI)
    plt.close()
    print("Saved:", out_png)

# ----------------------------
# RUN
# ----------------------------
def main():
    models = []
    rows = []

    for name, path, score_col in MODEL_SPECS:
        df = load_model(name, path, score_col)
        if df is None:
            continue

        y = df["y"].values
        s = df["_score01"].values

        roc = roc_auc_score(y, s)
        ap  = average_precision_score(y, s)
        t1  = topk_recall(y, s, 0.01)
        t5  = topk_recall(y, s, 0.05)
        t10 = topk_recall(y, s, 0.10)

        rows.append({
            "model": name,
            "roc_auc": roc,
            "pr_auc": ap,
            "top1": t1,
            "top5": t5,
            "top10": t10,
            "n": len(df),
            "n_pos": int(df["y"].sum())
        })
        models.append(df)

    if not models:
        raise RuntimeError("No model files found. Check MODEL_SPECS paths + score columns.")

    # Save table
    tab = pd.DataFrame(rows).sort_values(by=["top10","pr_auc","roc_auc"], ascending=False)
    tab_path = OUT_DIR / "table_model_compare_FINAL.csv"
    tab.to_csv(tab_path, index=False)
    print("Saved:", tab_path)
    print(tab)

    # Curves
    plot_roc(models)
    plot_pr(models)
    plot_topk(models)
    plot_reliability(models)

    # Maps for the FINAL model only (Model 17)
    final = None
    for df in models:
        if "Model17" in df["_model"].iloc[0] or "Final" in df["_model"].iloc[0]:
            final = df
            break
    if final is None:
        final = models[0]  # fallback

    plot_map(
        final,
        title=f"Prospectivity Map — {final['_model'].iloc[0]} (binned mean, {BIN_DEG}°)",
        out_png=FIG_DIR / "map_prospectivity_FINAL.png",
        bin_deg=BIN_DEG
    )
    for frac in [0.01, 0.05, 0.10]:
        plot_topzones(
            final,
            title=f"{final['_model'].iloc[0]}",
            out_png=FIG_DIR / f"map_top{int(frac*100)}pct_zone_FINAL.png",
            frac=frac
        )

    print("\nDONE. All figures are in:", FIG_DIR)

if __name__ == "__main__":
    main()
