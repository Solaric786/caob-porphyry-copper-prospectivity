"""
27_make_spatial_blocks.py

Adds spatial block IDs and 5-fold CV fold IDs to
  Outputs/eval_points_with_scores.csv

Output:
  Outputs/eval_points_with_scores_blocks.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(".")
in_csv  = BASE / "Outputs" / "eval_points_with_scores.csv"
out_csv = BASE / "Outputs" / "eval_points_with_scores_blocks.csv"

print(f"Reading: {in_csv}")
df = pd.read_csv(in_csv)

# ---------------------------------------------------------------------
# Define spatial blocks on lon/lat (WGS84)
# ---------------------------------------------------------------------
lon = df["lon"].values
lat = df["lat"].values

lon_min, lon_max = lon.min(), lon.max()
lat_min, lat_max = lat.min(), lat.max()

print(f"Lon range: {lon_min:.3f}–{lon_max:.3f}")
print(f"Lat range: {lat_min:.3f}–{lat_max:.3f}")

# Choose a grid of blocks over CAOB window
# here: 10 blocks in longitude × 4 blocks in latitude (40 blocks total)
NX_BLOCKS = 10
NY_BLOCKS = 4

# Normalised coordinates in [0,1)
tx = (lon - lon_min) / (lon_max - lon_min + 1e-9)
ty = (lat - lat_min) / (lat_max - lat_min + 1e-9)

bx = np.clip((tx * NX_BLOCKS).astype(int), 0, NX_BLOCKS - 1)
by = np.clip((ty * NY_BLOCKS).astype(int), 0, NY_BLOCKS - 1)

block_id = by * NX_BLOCKS + bx  # 0,…,NX*NY-1

df["block_id"] = block_id

# ---------------------------------------------------------------------
# Assign 5 CV folds by block (simple checkerboard-style)
# ---------------------------------------------------------------------
K_FOLDS = 5
fold_id = block_id % K_FOLDS
df["fold_id"] = fold_id

print("Block IDs and fold IDs assigned.")
print(df[["lon", "lat", "block_id", "fold_id"]].head())

print(f"Saving with blocks to: {out_csv}")
df.to_csv(out_csv, index=False)

print("Done.")
