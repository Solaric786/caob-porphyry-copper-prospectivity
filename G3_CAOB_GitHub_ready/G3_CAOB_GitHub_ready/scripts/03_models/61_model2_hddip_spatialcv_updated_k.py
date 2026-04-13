"""
Spatial block cross-validation for an HD–IPP model (alternating offsets + bias-anchored effort).

Implements the structure in your draft:
  log mu(x) = eta_lambda(x) + eta_pi(x)
  eta_lambda(x) = beta^T z(x)          (geology features, incl. lithology one-hot)
  eta_pi(x)     = alpha^T w(x)         (effort proxies)
with alternating updates:
  - geology step: fit beta with eta_pi as fixed offset
  - effort step:  fit alpha with eta_lambda as fixed offset + anchoring penalty toward r(x)
and recenter eta_pi so mean_{background} eta_pi = 0.

Outputs (in Outputs/cv_preds_hdipp.csv):
  - score_hdipp      = sigmoid(eta_mu)
  - hdipp_eta_lambda = eta_lambda
  - hdipp_eta_pi     = eta_pi
  - hdipp_eta_mu     = eta_lambda + eta_pi
"""

from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(".")

eval_blocks_csv = BASE / "Outputs" / "eval_points_with_scores_blocks.csv"
dep_csv         = BASE / "Outputs" / "deposits_por_Cu_with_covariates.csv"
bg_csv          = BASE / "Outputs" / "background_points_1km.csv"
out_csv         = BASE / "Outputs" / "cv_preds_hdipp.csv"

# -----------------------------
# Feature sets (same naming)
# -----------------------------
GEO_FEATURES = [
    "G1_DEM",
    "G1_SLOPE",
    "G1_TRI",
    "G2_LITHOLOGY8",      # categorical
    "G3_DIST_FAULT",
    "G4_ASTER_PHYLLIC",
    "G5_L8_iron",
]

# E4 is optional now (script will auto-disable anchoring if missing)
EFFORT_FEATURES = [
    "E1_DIST_ROAD",
    "E2_POP",
    "E3_VIIRS_2018",
    "E4_KDE_ALLMIN",
    # "E5_DIST_MAJOR",  # intentionally excluded (leakage)
]

CAT_VAR = "G2_LITHOLOGY8"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _sigmoid(x):
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out

def _standardize_fit(X):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be 2D.")
    if X.shape[1] == 0:
        return np.zeros(0, dtype=float), np.ones(0, dtype=float)
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    mu = np.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
    sd = np.nan_to_num(sd, nan=1.0, posinf=1.0, neginf=1.0)
    return mu, sd

def _standardize_apply(X, mu, sd):
    X = np.asarray(X, dtype=float)
    if X.shape[1] == 0:
        return X
    return (X - mu) / sd

def _one_hot_align(train_series, test_series, prefix):
    tr = train_series.fillna("Unknown").astype("category")
    te = test_series.fillna("Unknown").astype("category")
    X_tr = pd.get_dummies(tr, prefix=prefix)
    X_te = pd.get_dummies(te, prefix=prefix)
    X_te = X_te.reindex(columns=X_tr.columns, fill_value=0)
    return X_tr.to_numpy(dtype=float), X_te.to_numpy(dtype=float), list(X_tr.columns)

def _prep_design_matrices(df_train, df_test, all_features, cont_vars):
    """
    Builds:
      Xg_* : geology design matrix (standardized continuous GEO + one-hot lithology + intercept)
      Xe_* : effort  design matrix (standardized continuous effort features; no intercept)
    """
    df_tr = df_train.copy()
    df_te = df_test.copy()

    # Drop rows with missing continuous covariates
    mask_tr = np.isfinite(df_tr[cont_vars]).all(axis=1)
    mask_te = np.isfinite(df_te[cont_vars]).all(axis=1)

    df_tr = df_tr.loc[mask_tr].copy()
    df_te = df_te.loc[mask_te].copy()

    if len(df_tr) == 0:
        raise ValueError("No training rows left after dropping non-finite values.")

    # --- Geology continuous vars (excluding CAT_VAR)
    geo_cont = [c for c in GEO_FEATURES if c != CAT_VAR]
    Xg_tr_cont = df_tr[geo_cont].to_numpy(dtype=float)
    Xg_te_cont = df_te[geo_cont].to_numpy(dtype=float)

    mu_g, sd_g = _standardize_fit(Xg_tr_cont)
    Xg_tr_cont = _standardize_apply(Xg_tr_cont, mu_g, sd_g)
    Xg_te_cont = _standardize_apply(Xg_te_cont, mu_g, sd_g)

    # Lithology one-hot
    Xg_tr_cat, Xg_te_cat, cat_cols = _one_hot_align(df_tr[CAT_VAR], df_te[CAT_VAR], prefix=CAT_VAR)

    # Intercept ONLY in geology component
    intercept_tr = np.ones((len(df_tr), 1), dtype=float)
    intercept_te = np.ones((len(df_te), 1), dtype=float)

    Xg_tr = np.concatenate([intercept_tr, Xg_tr_cont, Xg_tr_cat], axis=1)
    Xg_te = np.concatenate([intercept_te, Xg_te_cont, Xg_te_cat], axis=1)

    # --- Effort vars (all continuous here)
    if len(EFFORT_FEATURES) > 0:
        Xe_tr = df_tr[EFFORT_FEATURES].to_numpy(dtype=float)
        Xe_te = df_te[EFFORT_FEATURES].to_numpy(dtype=float)
    else:
        Xe_tr = np.zeros((len(df_tr), 0), dtype=float)
        Xe_te = np.zeros((len(df_te), 0), dtype=float)

    mu_e, sd_e = _standardize_fit(Xe_tr)
    Xe_tr = _standardize_apply(Xe_tr, mu_e, sd_e)
    Xe_te = _standardize_apply(Xe_te, mu_e, sd_e)

    y_tr = df_tr["y"].to_numpy(dtype=int)

    idx_te = df_te.index.to_numpy()

    meta = {"mu_g": mu_g, "sd_g": sd_g, "mu_e": mu_e, "sd_e": sd_e, "cat_cols": cat_cols}
    return df_tr, df_te, idx_te, Xg_tr, Xg_te, Xe_tr, Xe_te, y_tr, meta

def _build_r_from_e4(df_tr, eps=1e-12):
    """
    Build r(x) from E4_KDE_ALLMIN on TRAINING fold only.
    If E4 is missing, returns zeros (and anchoring will be disabled via rho_eff=0).
    """
    if "E4_KDE_ALLMIN" not in df_tr.columns:
        return np.zeros(len(df_tr), dtype=float)

    r_raw = df_tr["E4_KDE_ALLMIN"].to_numpy(dtype=float)
    r = np.log(np.clip(r_raw, a_min=eps, a_max=None))

    # center over training background only (y=0)
    bg = (df_tr["y"].to_numpy(dtype=int) == 0)
    if np.any(bg):
        r = r - np.nanmean(r[bg])
    else:
        r = r - np.nanmean(r)

    r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
    return r

def _newton_logistic_offset_ridge(X, y, offset, lam, max_iter=60, tol=1e-6, jitter=1e-8, penalize_intercept=False):
    n, p = X.shape
    b = np.zeros(p, dtype=float)

    for _ in range(max_iter):
        eta = X @ b + offset
        p_hat = _sigmoid(eta)
        w = p_hat * (1.0 - p_hat)

        g = X.T @ (p_hat - y)

        if lam > 0:
            if penalize_intercept:
                g = g + lam * b
            else:
                g[1:] = g[1:] + lam * b[1:]

        Xw = X * w[:, None]
        H = X.T @ Xw

        if lam > 0:
            if penalize_intercept:
                H = H + lam * np.eye(p)
            else:
                H[1:, 1:] = H[1:, 1:] + lam * np.eye(p - 1)

        H = H + jitter * np.eye(p)

        step = np.linalg.solve(H, g)
        b_new = b - step

        if np.max(np.abs(b_new - b)) < tol:
            b = b_new
            break
        b = b_new

    return b

def _newton_effort_step(Xe, y, offset, lam_e, Xe_bg, r_bg, rho, max_iter=60, tol=1e-6, jitter=1e-8):
    n, p = Xe.shape
    if p == 0:
        return np.zeros(0, dtype=float)

    a = np.zeros(p, dtype=float)

    if rho > 0 and Xe_bg is not None and len(r_bg) > 0:
        XtX_bg = Xe_bg.T @ Xe_bg
        Xtr_bg = Xe_bg.T @ r_bg
    else:
        XtX_bg = None
        Xtr_bg = None

    for _ in range(max_iter):
        eta = Xe @ a + offset
        p_hat = _sigmoid(eta)
        w = p_hat * (1.0 - p_hat)

        g = Xe.T @ (p_hat - y)

        if lam_e > 0:
            g = g + lam_e * a

        if rho > 0 and XtX_bg is not None:
            g = g + rho * (XtX_bg @ a - Xtr_bg)

        Xw = Xe * w[:, None]
        H = Xe.T @ Xw

        if lam_e > 0:
            H = H + lam_e * np.eye(p)

        if rho > 0 and XtX_bg is not None:
            H = H + rho * XtX_bg

        H = H + jitter * np.eye(p)

        step = np.linalg.solve(H, g)
        a_new = a - step

        if np.max(np.abs(a_new - a)) < tol:
            a = a_new
            break
        a = a_new

    return a

def fit_hdipp_and_predict(df_train, df_test, all_features, cont_vars, T=20, eps_conv=1e-4, lam_g=1.0, lam_e=1.0, rho=3.0):
    """
    Returns four arrays (full length of df_test):
      score_full       : sigmoid(eta_mu)
      eta_lambda_full  : eta_lambda
      eta_pi_full      : eta_pi (centered using training background mean shift)
      eta_mu_full      : eta_lambda + eta_pi
    """
    df_tr, df_te, idx_te, Xg_tr, Xg_te, Xe_tr, Xe_te, y_tr, _ = _prep_design_matrices(
        df_train, df_test, all_features, cont_vars
    )

    y_tr_full = df_tr["y"].to_numpy(dtype=int)
    bg_mask = (y_tr_full == 0)
    if not np.any(bg_mask):
        raise ValueError("No background rows (y=0) in training fold; cannot fit HD–IPP.")

    # If E4 is absent, anchoring must be disabled
    has_e4 = ("E4_KDE_ALLMIN" in df_tr.columns)
    rho_eff = rho if has_e4 else 0.0

    # r(x) from E4 (or zeros)
    r_tr = _build_r_from_e4(df_tr)

    # init eta_pi^(0)
    eta_pi_tr = r_tr.copy()

    beta = np.zeros(Xg_tr.shape[1], dtype=float)
    alpha = np.zeros(Xe_tr.shape[1], dtype=float)

    last_eta_lam_bg = None
    last_eta_pi_bg = eta_pi_tr[bg_mask].copy()

    mean_bg = 0.0

    for _t in range(T):
        # Geology step
        beta = _newton_logistic_offset_ridge(
            X=Xg_tr, y=y_tr, offset=eta_pi_tr, lam=lam_g,
            max_iter=60, tol=1e-6, penalize_intercept=False
        )
        eta_lam_tr = Xg_tr @ beta

        # Effort step (with optional anchoring)
        Xe_bg = Xe_tr[bg_mask, :]
        r_bg  = r_tr[bg_mask]

        alpha = _newton_effort_step(
            Xe=Xe_tr, y=y_tr, offset=eta_lam_tr, lam_e=lam_e,
            Xe_bg=Xe_bg, r_bg=r_bg, rho=rho_eff,
            max_iter=60, tol=1e-6
        )
        eta_pi_tr = Xe_tr @ alpha if Xe_tr.shape[1] > 0 else np.zeros(len(df_tr), dtype=float)

        # recenter eta_pi on training background
        mean_bg = float(np.mean(eta_pi_tr[bg_mask]))
        eta_pi_tr = eta_pi_tr - mean_bg

        # convergence on background
        eta_lam_bg = eta_lam_tr[bg_mask]
        eta_pi_bg = eta_pi_tr[bg_mask]

        if last_eta_lam_bg is not None:
            d1 = np.max(np.abs(eta_lam_bg - last_eta_lam_bg))
            d2 = np.max(np.abs(eta_pi_bg - last_eta_pi_bg))
            if max(d1, d2) < eps_conv:
                break

        last_eta_lam_bg = eta_lam_bg.copy()
        last_eta_pi_bg = eta_pi_bg.copy()

    # Predict on test
    eta_lam_te = Xg_te @ beta
    eta_pi_te = (Xe_te @ alpha) if Xe_te.shape[1] > 0 else np.zeros(len(df_te), dtype=float)

    # apply SAME centering shift used on training background
    eta_pi_te = eta_pi_te - mean_bg

    eta_mu_te = eta_lam_te + eta_pi_te
    score_te = _sigmoid(eta_mu_te)

    # Map back to full df_test order
    n_test = len(df_test)
    score_full = np.full(n_test, np.nan, dtype=float)
    eta_lambda_full = np.full(n_test, np.nan, dtype=float)
    eta_pi_full = np.full(n_test, np.nan, dtype=float)
    eta_mu_full = np.full(n_test, np.nan, dtype=float)

    score_full[idx_te] = score_te
    eta_lambda_full[idx_te] = eta_lam_te
    eta_pi_full[idx_te] = eta_pi_te
    eta_mu_full[idx_te] = eta_mu_te

    return score_full, eta_lambda_full, eta_pi_full, eta_mu_full


# ---------------------------------------------------------------------
# Main CV loop
# ---------------------------------------------------------------------
print(f"Reading blocks: {eval_blocks_csv}")
df_blocks = pd.read_csv(eval_blocks_csv)[["lon", "lat", "block_id", "fold_id"]]

print(f"Reading deposits: {dep_csv}")
df_dep = pd.read_csv(dep_csv)
df_dep["y"] = 1

print(f"Reading background: {bg_csv}")
df_bg = pd.read_csv(bg_csv)
df_bg["y"] = 0

df_all = pd.concat([df_dep, df_bg], ignore_index=True)

df = df_all.merge(df_blocks, on=["lon", "lat"], how="inner")
print(f"After merge: {len(df)} rows")

df = df[df["fold_id"].notna()].copy()
df["fold_id"] = df["fold_id"].astype(int)

# ---- Handle optional E4 (no crash) ----
if "E4_KDE_ALLMIN" not in df.columns and "E4_KDE_ALLMIN" in EFFORT_FEATURES:
    print("WARNING: Column 'E4_KDE_ALLMIN' not found. Disabling bias-anchoring (rho=0) and removing E4 from effort features.")
    EFFORT_FEATURES = [c for c in EFFORT_FEATURES if c != "E4_KDE_ALLMIN"]

ALL_FEATURES = GEO_FEATURES + EFFORT_FEATURES
CONT_VARS = [c for c in ALL_FEATURES if c != CAT_VAR]

missing = [c for c in ALL_FEATURES if c not in df.columns]
if missing:
    raise ValueError(f"Missing HD–IPP features in data: {missing}")

folds = sorted(df["fold_id"].unique())
print("Folds present:", folds)

records = []

# Hyperparameters
T = 20
eps_conv = 1e-4
lam_g = 1.0   # geology ridge
lam_e = 1.0   # effort ridge
rho   = 3.0   # anchoring strength (auto-set to 0 if E4 missing)

for k in folds:
    print(f"\n=== HD–IPP CV: Fold {k} ===")
    mask_train = df["fold_id"] != k
    mask_test  = df["fold_id"] == k

    df_train = df.loc[mask_train].copy().reset_index(drop=True)
    df_test  = df.loc[mask_test].copy().reset_index(drop=True)

    print(f"  Train n={len(df_train)}, test n={len(df_test)}")

    score_test, eta_lam_test, eta_pi_test, eta_mu_test = fit_hdipp_and_predict(
        df_train, df_test,
        all_features=ALL_FEATURES,
        cont_vars=CONT_VARS,
        T=T, eps_conv=eps_conv,
        lam_g=lam_g, lam_e=lam_e,
        rho=rho,
    )

    tmp = df_test[["lon", "lat", "y", "block_id", "fold_id"]].copy()
    tmp["score_hdipp"] = np.asarray(score_test, dtype=float)

    # (1)–(3) decomposition outputs
    tmp["hdipp_eta_lambda"] = np.asarray(eta_lam_test, dtype=float)  # (1)
    tmp["hdipp_eta_pi"]     = np.asarray(eta_pi_test, dtype=float)   # (2)
    tmp["hdipp_eta_mu"]     = np.asarray(eta_mu_test, dtype=float)   # (3)

    records.append(tmp)

df_out = pd.concat(records, ignore_index=True)
print(f"\nSaving HD–IPP CV predictions to: {out_csv}")
df_out.to_csv(out_csv, index=False)
print("Done.")
