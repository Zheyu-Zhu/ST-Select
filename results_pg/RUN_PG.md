2026-07-25 19:29:33  ======================================================================
2026-07-25 19:29:33  RUN-PG: processed_gene phased training — 2026-07-25 19:29
2026-07-25 19:29:33  cuda=True gpu=NVIDIA GeForce RTX 3080
2026-07-25 19:29:33  ======================================================================
2026-07-25 19:29:33  ### BUILD pg_bc_hvg300
2026-07-25 19:31:16  ### BUILD pg_bc_hvg300 OK in 1.7 min
2026-07-25 19:31:16  ### BUILD pg_kidney_hvg300
2026-07-25 19:32:49  ### BUILD pg_kidney_hvg300 OK in 1.5 min
2026-07-25 19:32:49  ### SKIP  [P2-frozen] pg_her2_hvg300 — exists
2026-07-25 19:32:49  ### START [P2-frozen] pg_bc_hvg300  methods=1 image_mode=False epochs=60
2026-07-25 19:36:35  ### DONE  [P2-frozen] pg_bc_hvg300 in 3.8 min
      full_supervision     per_gene=0.1810 per_slide=0.1262 top100=0.1744
2026-07-25 19:36:35  ### START [P2-frozen] pg_kidney_hvg300  methods=1 image_mode=False epochs=60
2026-07-25 19:39:42  ### DONE  [P2-frozen] pg_kidney_hvg300 in 3.1 min
      full_supervision     per_gene=0.3404 per_slide=0.2274 top100=0.3930
2026-07-25 19:39:42  ### START [P3-AL-her2] pg_her2_hvg300  methods=8 image_mode=False epochs=60
2026-07-25 20:31:30  ### DONE  [P3-AL-her2] pg_her2_hvg300 in 51.8 min
      [AL] random=0.2563  best=poisson_disk=0.2605  delta=+0.0042  (per-slide)
      random               per_gene=0.3161 per_slide=0.2563 top100=0.3352
      entropy              per_gene=0.3179 per_slide=0.2570 top100=0.3363
      tod                  per_gene=0.3144 per_slide=0.2558 top100=0.3333
      coreset              per_gene=0.3188 per_slide=0.2580 top100=0.3368
      typiclust            per_gene=0.3170 per_slide=0.2567 top100=0.3353
      badge                per_gene=0.3169 per_slide=0.2571 top100=0.3356
      poisson_disk         per_gene=0.3209 per_slide=0.2605 top100=0.3391
      spatial_stratified   per_gene=0.3163 per_slide=0.2552 top100=0.3348
      full_supervision     per_gene=0.3183 per_slide=0.2568 top100=0.3376
2026-07-25 20:31:30  ### START [P4-e2e-her2] pg_her2_hvg300  methods=1 image_mode=True epochs=40
2026-07-25 22:40:55  ### DONE  [P4-e2e-her2] pg_her2_hvg300 in 129.4 min
      full_supervision     per_gene=0.3928 per_slide=0.3064 top100=0.4025
2026-07-25 22:40:55  ### START [P5-AL-bc] pg_bc_hvg300  methods=8 image_mode=False epochs=60
