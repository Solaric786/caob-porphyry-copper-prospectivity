# G3_CAOB_EffortBias_code

Paper-support code package for the manuscript:

**Bias-Aware Regional Prospectivity Mapping for Porphyry Copper in the Eastern Central Asian Orogenic Belt Under Incomplete Occurrence Data**

This package gathers the scripts most closely aligned with the manuscript workflow:
- study window and deposit preprocessing
- 1 km grid and covariate preparation
- spatial block cross-validation
- model chain for AC-XGB, HDIPP/DAOLI-style baseline, PW-PUStack, and RO-BAB
- map and figure generation

## Repository structure
- `scripts/01_data_prep/` data window, raster/covariate, and point extraction scripts
- `scripts/02_cv_and_tables/` spatial fold construction
- `scripts/03_models/` final model scripts used for reported comparisons
- `scripts/04_maps_and_figures/` final map and figure generation scripts
- `metadata/` saved metrics and blend-weight text files from the final workflow
- `docs/` helper notes for release preparation

## Expected local folders
These scripts were preserved close to the original working version and expect relative folders such as:
- `Data/`
- `New Data/`
- `Outputs/`
- sometimes `Outputs/paper_run_20260118/`

Before running publicly, review the input filenames and paths against the processed data package you plan to upload to Zenodo.

## Minimum processed data to archive with Zenodo
- cleaned porphyry Cu occurrence table
- CAOB study window / mask
- 1 km grid
- sampled background table
- harmonized covariate extracts used for modelling
- spatial fold assignments
- out-of-fold prediction tables for Model 6, Model 2/HDIPP, Model 16, and Model 17
- final map-support rasters or compact derived outputs

## Notes
- The `07_build_E4_allmin_kde.py` and `08_build_E5_dist_major.py` provenance scripts are included, but the final stack script comments indicate those features were disabled in the final paper-support stack to avoid leakage.
- Upstream third-party layers cited in the manuscript should be referenced in the paper and README; they do not all need to be re-hosted if already public.
- Add a license file only after choosing the license you want for the public repo.
