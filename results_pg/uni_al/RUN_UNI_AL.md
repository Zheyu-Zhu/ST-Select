2026-07-30 00:18:15  ======================================================================
2026-07-30 00:18:15  UNI e2e AL spot-check [DRY] - 2026-07-30 00:18
2026-07-30 00:18:15  dataset=pg_her2_hvg300 methods=['random', 'poisson_disk', 'tod', 'typiclust'] ratios=[0.02, 0.03]
2026-07-30 00:18:15  ======================================================================
2026-07-30 00:33:51  ### DONE pg_her2_hvg300 in 15.6 min
      random           per_slide=0.2848  vs_random=+0.0000  per_gene=0.3000
      typiclust        per_slide=0.2665  vs_random=-0.0183  per_gene=0.2861
      poisson_disk     per_slide=0.2632  vs_random=-0.0216  per_gene=0.2850
      tod              per_slide=0.2452  vs_random=-0.0396  per_gene=0.2674
      -> best non-random: typiclust -0.0183 vs random
2026-07-30 00:33:59  ======================================================================
2026-07-30 00:33:59  UNI e2e AL spot-check - 2026-07-30 00:33
2026-07-30 00:33:59  dataset=pg_her2_hvg300 methods=['random', 'poisson_disk', 'tod', 'typiclust'] ratios=[0.05, 0.3]
2026-07-30 00:33:59  ======================================================================
2026-07-30 06:10:18  ### DONE pg_her2_hvg300 in 336.3 min
      random           per_slide=0.3499  vs_random=+0.0000  per_gene=0.4450
      tod              per_slide=0.3473  vs_random=-0.0027  per_gene=0.4199
      poisson_disk     per_slide=0.3451  vs_random=-0.0048  per_gene=0.4243
      typiclust        per_slide=0.3425  vs_random=-0.0074  per_gene=0.4247
      -> best non-random: tod -0.0027 vs random
2026-07-31 05:54:17  ======================================================================
2026-07-31 05:54:17  UNI e2e AL spot-check - 2026-07-31 05:54
2026-07-31 05:54:17  dataset=pg_her2_hvg300 methods=['random', 'poisson_disk', 'tod', 'typiclust'] ratios=[0.05, 0.3]
2026-07-31 05:54:17  ======================================================================
2026-07-31 11:00:41  ### DONE pg_her2_hvg300 in 306.4 min
      poisson_disk     per_slide=0.3414  vs_random=+0.0031  per_gene=0.4319
      typiclust        per_slide=0.3384  vs_random=+0.0001  per_gene=0.4144
      random           per_slide=0.3383  vs_random=+0.0000  per_gene=0.4351
      tod              per_slide=0.3382  vs_random=-0.0001  per_gene=0.4278
      -> best non-random: poisson_disk +0.0031 vs random
2026-07-31 11:00:49  ======================================================================
2026-07-31 11:00:49  UNI e2e AL spot-check - 2026-07-31 11:00
2026-07-31 11:00:49  dataset=pg_kidney_hvg300 methods=['random', 'poisson_disk', 'tod', 'typiclust'] ratios=[0.05, 0.3]
2026-07-31 11:00:49  ======================================================================
2026-07-31 20:50:01  ### DONE pg_kidney_hvg300 in 589.2 min
      typiclust        per_slide=0.3407  vs_random=+0.0090  per_gene=0.4436
      random           per_slide=0.3318  vs_random=+0.0000  per_gene=0.4394
      poisson_disk     per_slide=0.3122  vs_random=-0.0196  per_gene=0.4311
      tod              per_slide=0.3048  vs_random=-0.0269  per_gene=0.4160
      -> best non-random: typiclust +0.0090 vs random
