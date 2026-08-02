2026-07-26 01:13:12  ======================================================================
2026-07-26 01:13:12  RUN-TONIGHT-PG [lane=cpu] — 2026-07-26 01:13
2026-07-26 01:13:12  ======================================================================
2026-07-26 01:13:12  ### START [AL-bc] pg_bc_hvg300 image_mode=False methods=8 epochs=60
2026-07-26 01:16:56  ======================================================================
2026-07-26 01:16:56  RUN-TONIGHT-PG [lane=cpu] — 2026-07-26 01:16
2026-07-26 01:16:56  ======================================================================
2026-07-26 01:16:56  ### START [AL-bc] pg_bc_hvg300 image_mode=False methods=8 epochs=60
2026-07-26 04:12:53  ======================================================================
2026-07-26 04:12:53  RUN-TONIGHT-PG [lane=cpu] — 2026-07-26 04:12
2026-07-26 04:12:53  ======================================================================
2026-07-26 04:12:53  ### START [AL-bc] pg_bc_hvg300 image_mode=False methods=8 epochs=60
2026-07-26 08:03:01  ### DONE  [AL-bc] pg_bc_hvg300 in 230.1 min
      [AL] random=0.1298 best=badge=0.1321 delta=+0.0023 (per-slide)
      random               per_gene=0.1839 per_slide=0.1298 top100=0.1792
      entropy              per_gene=0.1880 per_slide=0.1319 top100=0.1836
      tod                  per_gene=0.1838 per_slide=0.1299 top100=0.1794
      coreset              per_gene=0.1896 per_slide=0.1318 top100=0.1847
      typiclust            per_gene=0.1855 per_slide=0.1313 top100=0.1805
      badge                per_gene=0.1870 per_slide=0.1321 top100=0.1823
      poisson_disk         per_gene=0.1854 per_slide=0.1315 top100=0.1807
      spatial_stratified   per_gene=0.1823 per_slide=0.1295 top100=0.1776
      full_supervision     per_gene=0.1810 per_slide=0.1262 top100=0.1744
2026-07-26 08:03:01  ### START [AL-kidney] pg_kidney_hvg300 image_mode=False methods=8 epochs=60
2026-07-26 10:52:55  ### DONE  [AL-kidney] pg_kidney_hvg300 in 169.9 min
      [AL] random=0.2311 best=tod=0.2339 delta=+0.0028 (per-slide)
      random               per_gene=0.3429 per_slide=0.2311 top100=0.3956
      entropy              per_gene=0.3404 per_slide=0.2310 top100=0.3934
      tod                  per_gene=0.3405 per_slide=0.2339 top100=0.3929
      coreset              per_gene=0.3392 per_slide=0.2306 top100=0.3926
      typiclust            per_gene=0.3412 per_slide=0.2328 top100=0.3957
      badge                per_gene=0.3410 per_slide=0.2294 top100=0.3963
      poisson_disk         per_gene=0.3430 per_slide=0.2304 top100=0.3974
      spatial_stratified   per_gene=0.3427 per_slide=0.2317 top100=0.3963
      full_supervision     per_gene=0.3404 per_slide=0.2274 top100=0.3930
2026-07-26 10:52:55  ### BUILD pg_her2_norm (densenet121)
2026-07-26 10:54:03  ### BUILD pg_her2_norm OK in 1.1 min
2026-07-26 10:54:03  ### START [target-norm] pg_her2_norm image_mode=False methods=1 epochs=60
2026-07-26 10:56:05  ### DONE  [target-norm] pg_her2_norm in 2.0 min
      full_supervision     per_gene=0.2665 per_slide=0.2261 top100=0.2743
2026-07-26 10:56:05  ### BUILD pg_her2_hvg300_uni (uni)
2026-07-26 11:00:47  ### BUILD pg_her2_hvg300_uni OK in 4.7 min
2026-07-26 11:00:47  ### START [uni-her2] pg_her2_hvg300_uni image_mode=False methods=1 epochs=60
2026-07-26 11:02:34  ### DONE  [uni-her2] pg_her2_hvg300_uni in 1.8 min
      full_supervision     per_gene=0.3765 per_slide=0.3009 top100=0.3905
2026-07-26 11:02:34  LANE [cpu] COMPLETE
2026-07-26 11:02:34  ======================================================================
