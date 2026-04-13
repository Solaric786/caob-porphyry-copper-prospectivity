# Suggested run order

1. `scripts/01_data_prep/00_define_CAOB_window.py`
2. `scripts/01_data_prep/01_preprocess_deposits.py`
3. `scripts/01_data_prep/02_build_grid_1km.py`
4. `scripts/01_data_prep/03_resample_G1_to_grid.py`
5. `scripts/01_data_prep/04_build_lithology.py`
6. `scripts/01_data_prep/05_build_fault_distance.py`
7. `scripts/01_data_prep/06_build_G4_G5.py`
8. `scripts/01_data_prep/07_build_E2_E3.py`
9. `scripts/01_data_prep/09_stack_covariates.py`
10. `scripts/01_data_prep/10_extract_covariates_to_deposits.py`
11. `scripts/01_data_prep/11_background_points_and_covariates.py`
12. build or combine evaluation table as in your working outputs
13. `scripts/02_cv_and_tables/27_make_spatial_blocks.py`
14. `scripts/03_models/56_model6_xgb_tuned_spatialcv.py`
15. `scripts/03_models/61_model2_hddip_spatialcv_updated_k.py`
16. `scripts/03_models/76_model16_pu_stackmeta_xgb_FAST.py`
17. `scripts/03_models/77_model17_rankblend_from_model16_FAST.py`
18. `scripts/04_maps_and_figures/80_make_plots_and_maps_FINAL.py`
19. `scripts/04_maps_and_figures/81_plot_fine_prospectivity_map.py`
20. `scripts/04_maps_and_figures/82_plot_all_prospectivity_maps.py`
21. `scripts/04_maps_and_figures/84_make_continuous_model17_map.py`
22. `scripts/04_maps_and_figures/90_make_scientific_figures.py`
