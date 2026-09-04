2026-08-02 14:44:27  ======================================================================
2026-08-02 14:44:27  WITHIN-SLIDE ACQUISITION · pg_her2_hvg300 · mode=random
2026-08-02 14:44:27  methods=['random', 'entropy', 'tod', 'coreset', 'typiclust', 'badge', 'poisson_disk', 'spatial_stratified'] seeds=3 ratios=[0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
2026-08-02 14:44:27  train and test spots share slides — NOT comparable to patient-split numbers
2026-08-02 14:44:27  ======================================================================
2026-08-02 14:44:27  --- nearest-neighbour control (no image, no model) ---
2026-08-02 14:44:48        5%=0.1874  10%=0.3065  20%=0.3808  30%=0.4041  50%=0.4397  75%=0.4636  100%=0.4795
2026-08-02 17:11:02  ### DONE pg_her2_hvg300_random in 146.6 min
2026-08-02 17:11:02        entropy              per_slide=0.3373  vs_random=+0.0044
2026-08-02 17:11:02        spatial_stratified   per_slide=0.3343  vs_random=+0.0014
2026-08-02 17:11:02        full_supervision     per_slide=0.3339  vs_random=+0.0011
2026-08-02 17:11:02        poisson_disk         per_slide=0.3335  vs_random=+0.0006
2026-08-02 17:11:02        random               per_slide=0.3329  vs_random=     —
2026-08-02 17:11:02        badge                per_slide=0.3323  vs_random=-0.0005
2026-08-02 17:11:02        typiclust            per_slide=0.3277  vs_random=-0.0051
2026-08-02 17:11:02        coreset              per_slide=0.3276  vs_random=-0.0053
2026-08-02 17:11:02        tod                  per_slide=0.3273  vs_random=-0.0056
2026-08-02 17:11:02        kNN control @100% = 0.4795   model random @100% = 0.3329   model advantage = -0.1467
2026-08-02 17:11:09  ======================================================================
2026-08-02 17:11:09  WITHIN-SLIDE ACQUISITION · pg_her2_hvg300 · mode=spatial
2026-08-02 17:11:09  methods=['random', 'entropy', 'tod', 'coreset', 'typiclust', 'badge', 'poisson_disk', 'spatial_stratified'] seeds=3 ratios=[0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
2026-08-02 17:11:09  train and test spots share slides — NOT comparable to patient-split numbers
2026-08-02 17:11:09  ======================================================================
2026-08-02 17:11:09  --- nearest-neighbour control (no image, no model) ---
2026-08-02 17:11:31        5%=0.1014  10%=0.1847  20%=0.2196  30%=0.2286  50%=0.2384  75%=0.2543  100%=0.2616
2026-08-02 19:36:08  ### DONE pg_her2_hvg300_spatial in 145.0 min
2026-08-02 19:36:08        coreset              per_slide=0.2726  vs_random=+0.0052
2026-08-02 19:36:08        random               per_slide=0.2674  vs_random=     —
2026-08-02 19:36:08        full_supervision     per_slide=0.2669  vs_random=-0.0005
2026-08-02 19:36:08        poisson_disk         per_slide=0.2668  vs_random=-0.0006
2026-08-02 19:36:08        tod                  per_slide=0.2665  vs_random=-0.0009
2026-08-02 19:36:08        typiclust            per_slide=0.2636  vs_random=-0.0038
2026-08-02 19:36:08        spatial_stratified   per_slide=0.2626  vs_random=-0.0049
2026-08-02 19:36:08        badge                per_slide=0.2623  vs_random=-0.0051
2026-08-02 19:36:08        entropy              per_slide=0.2587  vs_random=-0.0087
2026-08-02 19:36:08        kNN control @100% = 0.2616   model random @100% = 0.2674   model advantage = +0.0058
