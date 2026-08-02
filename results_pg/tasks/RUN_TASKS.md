2026-07-27 00:31:22  ======================================================================
2026-07-27 00:31:22  RUN-TASKS: multi-task evaluation (pitch tasks #4, #5) — 2026-07-27 00:31
2026-07-27 00:31:22  ======================================================================
2026-07-27 00:31:22  ### START [T1-gene-her2] pg_her2_hvg300 methods=1
2026-07-27 00:32:51  ### DONE  [T1-gene-her2] pg_her2_hvg300 in 1.5 min
      full_supervision     per_gene=0.3183 per_slide=0.2568 | moran true=0.328 pred=0.369 fid=0.166
2026-07-27 00:32:51  ### START [T1-gene-her2-uni] pg_her2_hvg300_uni methods=1
2026-07-27 00:34:18  ### DONE  [T1-gene-her2-uni] pg_her2_hvg300_uni in 1.4 min
      full_supervision     per_gene=0.3765 per_slide=0.3009 | moran true=0.328 pred=0.423 fid=0.152
2026-07-27 00:34:18  ### START [T1-gene-bc] pg_bc_hvg300 methods=1
2026-07-27 00:37:32  ### DONE  [T1-gene-bc] pg_bc_hvg300 in 3.2 min
      full_supervision     per_gene=0.1810 per_slide=0.1262 | moran true=0.186 pred=0.287 fid=0.056
2026-07-27 00:37:32  ### START [T1-gene-bc-uni] pg_bc_hvg300_uni methods=1
2026-07-27 00:40:48  ### DONE  [T1-gene-bc-uni] pg_bc_hvg300_uni in 3.3 min
      full_supervision     per_gene=0.2012 per_slide=0.1305 | moran true=0.186 pred=0.314 fid=0.079
2026-07-27 00:40:48  ### START [T1-gene-kidney] pg_kidney_hvg300 methods=1
2026-07-27 00:43:32  ### DONE  [T1-gene-kidney] pg_kidney_hvg300 in 2.7 min
      full_supervision     per_gene=0.3404 per_slide=0.2274 | moran true=0.267 pred=0.352 fid=0.056
2026-07-27 00:43:32  ### START [T1-gene-kidney-uni] pg_kidney_hvg300_uni methods=1
2026-07-27 00:46:16  ### DONE  [T1-gene-kidney-uni] pg_kidney_hvg300_uni in 2.7 min
      full_supervision     per_gene=0.4247 per_slide=0.3259 | moran true=0.267 pred=0.400 fid=0.152
2026-07-27 00:46:16  ### START [T2-ct-her2] pg_her2_celltype methods=1
2026-07-27 00:47:42  ### DONE  [T2-ct-her2] pg_her2_celltype in 1.4 min
      full_supervision     per_gene=0.1801 per_slide=0.0976 | moran true=0.156 pred=0.186 fid=0.336 | cos=0.927 jsd=0.023 pccType=0.084
2026-07-27 00:47:42  ### START [T2-ct-her2-uni] pg_her2_celltype_uni methods=1
2026-07-27 00:49:08  ### DONE  [T2-ct-her2-uni] pg_her2_celltype_uni in 1.4 min
      full_supervision     per_gene=0.1241 per_slide=0.0751 | moran true=0.156 pred=0.240 fid=0.276 | cos=0.931 jsd=0.023 pccType=0.108
2026-07-27 00:49:08  ### START [T2-ct-bc] pg_bc_celltype methods=1
2026-07-27 00:52:19  ### DONE  [T2-ct-bc] pg_bc_celltype in 3.2 min
      full_supervision     per_gene=0.2290 per_slide=0.1852 | moran true=0.170 pred=0.213 fid=0.279 | cos=0.899 jsd=0.030 pccType=0.090
2026-07-27 00:52:19  ### START [T2-ct-bc-uni] pg_bc_celltype_uni methods=1
2026-07-27 00:55:31  ### DONE  [T2-ct-bc-uni] pg_bc_celltype_uni in 3.2 min
      full_supervision     per_gene=0.2922 per_slide=0.2433 | moran true=0.170 pred=0.230 fid=0.273 | cos=0.903 jsd=0.029 pccType=0.090
2026-07-27 00:55:31  ### START [T2-ct-kidney] pg_kidney_celltype methods=1
2026-07-27 00:58:12  ### DONE  [T2-ct-kidney] pg_kidney_celltype in 2.7 min
      full_supervision     per_gene=0.3479 per_slide=0.2445 | moran true=0.244 pred=0.318 fid=0.395 | cos=0.892 jsd=0.032 pccType=0.177
2026-07-27 00:58:12  ### START [T2-ct-kidney-uni] pg_kidney_celltype_uni methods=1
2026-07-27 01:00:53  ### DONE  [T2-ct-kidney-uni] pg_kidney_celltype_uni in 2.7 min
      full_supervision     per_gene=0.3728 per_slide=0.2701 | moran true=0.244 pred=0.400 fid=0.398 | cos=0.904 jsd=0.028 pccType=0.281
2026-07-27 01:00:53  ### START [T3-AL-ct-her2] pg_her2_celltype methods=8
2026-07-27 01:45:34  ### DONE  [T3-AL-ct-her2] pg_her2_celltype in 44.7 min
      [AL] random=0.0815 best=spatial_stratified=0.0841 delta=+0.0026
      random               per_gene=0.1408 per_slide=0.0815 | moran true=0.156 pred=0.170 fid=0.343 | cos=0.928 jsd=0.024 pccType=0.074
      entropy              per_gene=0.1440 per_slide=0.0837 | moran true=0.156 pred=0.171 fid=0.335 | cos=0.928 jsd=0.024 pccType=0.075
      tod                  per_gene=0.1415 per_slide=0.0832 | moran true=0.156 pred=0.174 fid=0.332 | cos=0.928 jsd=0.024 pccType=0.076
      coreset              per_gene=0.1415 per_slide=0.0797 | moran true=0.156 pred=0.174 fid=0.338 | cos=0.928 jsd=0.024 pccType=0.073
      typiclust            per_gene=0.1418 per_slide=0.0823 | moran true=0.156 pred=0.172 fid=0.340 | cos=0.928 jsd=0.024 pccType=0.075
      badge                per_gene=0.1434 per_slide=0.0819 | moran true=0.156 pred=0.172 fid=0.342 | cos=0.928 jsd=0.024 pccType=0.076
      poisson_disk         per_gene=0.1397 per_slide=0.0758 | moran true=0.156 pred=0.170 fid=0.338 | cos=0.928 jsd=0.024 pccType=0.073
      spatial_stratified   per_gene=0.1461 per_slide=0.0841 | moran true=0.156 pred=0.171 fid=0.328 | cos=0.928 jsd=0.024 pccType=0.075
      full_supervision     per_gene=0.1801 per_slide=0.0976 | moran true=0.156 pred=0.186 fid=0.336 | cos=0.927 jsd=0.023 pccType=0.084
2026-07-27 01:45:34  ### START [T3-AL-ct-kidney] pg_kidney_celltype methods=8
2026-07-27 03:41:51  ### DONE  [T3-AL-ct-kidney] pg_kidney_celltype in 116.3 min
      [AL] random=0.2226 best=entropy=0.2298 delta=+0.0072
      random               per_gene=0.3176 per_slide=0.2226 | moran true=0.244 pred=0.308 fid=0.353 | cos=0.890 jsd=0.033 pccType=0.161
      entropy              per_gene=0.3232 per_slide=0.2298 | moran true=0.244 pred=0.315 fid=0.365 | cos=0.890 jsd=0.033 pccType=0.161
      tod                  per_gene=0.3245 per_slide=0.2297 | moran true=0.244 pred=0.312 fid=0.339 | cos=0.889 jsd=0.033 pccType=0.161
      coreset              per_gene=0.3150 per_slide=0.2215 | moran true=0.244 pred=0.307 fid=0.355 | cos=0.890 jsd=0.033 pccType=0.159
      typiclust            per_gene=0.3162 per_slide=0.2235 | moran true=0.244 pred=0.307 fid=0.355 | cos=0.890 jsd=0.033 pccType=0.161
      badge                per_gene=0.3157 per_slide=0.2218 | moran true=0.244 pred=0.305 fid=0.356 | cos=0.890 jsd=0.033 pccType=0.159
      poisson_disk         per_gene=0.3166 per_slide=0.2243 | moran true=0.244 pred=0.310 fid=0.354 | cos=0.890 jsd=0.033 pccType=0.160
      spatial_stratified   per_gene=0.3145 per_slide=0.2204 | moran true=0.244 pred=0.315 fid=0.327 | cos=0.890 jsd=0.033 pccType=0.162
      full_supervision     per_gene=0.3479 per_slide=0.2445 | moran true=0.244 pred=0.318 fid=0.395 | cos=0.892 jsd=0.032 pccType=0.177
2026-07-27 03:41:51  ### START [T3-AL-ct-bc] pg_bc_celltype methods=8
2026-07-27 06:15:55  ### DONE  [T3-AL-ct-bc] pg_bc_celltype in 154.1 min
      [AL] random=0.1860 best=tod=0.1901 delta=+0.0042
      random               per_gene=0.2209 per_slide=0.1860 | moran true=0.170 pred=0.210 fid=0.258 | cos=0.898 jsd=0.030 pccType=0.081
      entropy              per_gene=0.2282 per_slide=0.1848 | moran true=0.170 pred=0.206 fid=0.245 | cos=0.897 jsd=0.031 pccType=0.077
      tod                  per_gene=0.2317 per_slide=0.1901 | moran true=0.170 pred=0.209 fid=0.269 | cos=0.898 jsd=0.030 pccType=0.079
      coreset              per_gene=0.2224 per_slide=0.1861 | moran true=0.170 pred=0.209 fid=0.257 | cos=0.898 jsd=0.031 pccType=0.078
      typiclust            per_gene=0.2184 per_slide=0.1838 | moran true=0.170 pred=0.202 fid=0.253 | cos=0.898 jsd=0.030 pccType=0.076
      badge                per_gene=0.2230 per_slide=0.1801 | moran true=0.170 pred=0.208 fid=0.260 | cos=0.898 jsd=0.030 pccType=0.081
      poisson_disk         per_gene=0.2193 per_slide=0.1821 | moran true=0.170 pred=0.207 fid=0.250 | cos=0.898 jsd=0.030 pccType=0.076
      spatial_stratified   per_gene=0.2189 per_slide=0.1797 | moran true=0.170 pred=0.204 fid=0.272 | cos=0.898 jsd=0.030 pccType=0.077
      full_supervision     per_gene=0.2290 per_slide=0.1852 | moran true=0.170 pred=0.213 fid=0.279 | cos=0.899 jsd=0.030 pccType=0.090
2026-07-27 06:15:55  ALL TASK PHASES COMPLETE
2026-07-27 06:15:55  ======================================================================
