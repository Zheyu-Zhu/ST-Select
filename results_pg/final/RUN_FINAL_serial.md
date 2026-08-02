2026-07-28 10:08:02  ======================================================================
2026-07-28 10:08:02  RUN-FINAL [lane=serial] — 2026-07-28 10:08
2026-07-28 10:08:02  ======================================================================
2026-07-28 10:08:02  ### START [AL20-her2] pg_her2_hvg300 methods=20 model=feature_predictor image=False
2026-07-28 13:29:31  ### DONE  [AL20-her2] pg_her2_hvg300 in 201.5 min
      [AL] random=0.2563  best=poisson_disk=0.2605  delta=+0.0042  (n_methods=20)
      top5: poisson_disk=0.2605, cald=0.2605, maxherding=0.2600, ceal=0.2592, margin=0.2591
      full_supervision     per_gene=0.3183 per_slide=0.2568 | moran pred=0.369 fid=0.166
2026-07-28 13:29:31  ### START [UNIe2e-her2] pg_her2_hvg300 methods=1 model=uni_regressor image=True
2026-07-28 17:39:23  ### DONE  [UNIe2e-her2] pg_her2_hvg300 in 249.9 min
      full_supervision     per_gene=0.4429 per_slide=0.3313 | moran pred=0.482 fid=0.116
2026-07-28 17:39:23  ### START [AL20-kidney] pg_kidney_hvg300 methods=20 model=feature_predictor image=False
2026-07-29 11:29:59  ### DONE  [AL20-kidney] pg_kidney_hvg300 in 1070.6 min
      [AL] random=0.2311  best=tod=0.2339  delta=+0.0028  (n_methods=20)
      top5: tod=0.2339, typiclust=0.2328, spatial_stratified=0.2317, similar=0.2313, tissue_aware=0.2311
      full_supervision     per_gene=0.3404 per_slide=0.2274 | moran pred=0.352 fid=0.056
2026-07-29 11:29:59  ### START [UNIe2e-kidney] pg_kidney_hvg300 methods=1 model=uni_regressor image=True
2026-07-29 19:24:50  ### DONE  [UNIe2e-kidney] pg_kidney_hvg300 in 474.8 min
      full_supervision     per_gene=0.4670 per_slide=0.3876 | moran pred=0.403 fid=0.274
2026-07-29 19:24:50  LANE [serial] COMPLETE
2026-07-29 19:24:50  ======================================================================
