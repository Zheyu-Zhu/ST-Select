2026-07-31 05:42:09  ======================================================================
2026-07-31 05:42:09  BREADTH vs DEPTH — same spot budget, different numbers of patients
2026-07-31 05:42:09  ======================================================================
2026-07-31 05:42:09  ### START pg_her2_hvg300 — 6 training patients, k=[2, 4, 6], fixed budget B=1965 spots, 3 seeds x 4 folds
2026-07-31 05:42:53      fold 0: 9 cells, k2=0.3882  k4=0.3632  k6=0.3891
2026-07-31 05:43:38      fold 1: 9 cells, k2=0.1695  k4=0.1753  k6=0.2011
2026-07-31 05:44:22      fold 2: 9 cells, k2=0.2501  k4=0.2396  k6=0.2942
2026-07-31 05:45:01      fold 3: 9 cells, k2=0.2289  k4=0.2633  k6=0.2843
2026-07-31 05:45:01  ### DONE pg_her2_hvg300 in 2.9 min  (B=1965 spots, fixed)
2026-07-31 05:45:01        k= 2 patients  per-slide=0.2592  per-gene=0.2382  spots=1965  epochs=57  n=12
2026-07-31 05:45:01        k= 4 patients  per-slide=0.2603  per-gene=0.2935  spots=1965  epochs=25  n=12
2026-07-31 05:45:01        k= 6 patients  per-slide=0.2922  per-gene=0.3182  spots=1965  epochs=27  n=12
2026-07-31 05:45:01        -> breadth effect at fixed budget: +0.0330 per-slide going from 2 to 6 patients
2026-07-31 05:45:01  ### START pg_kidney_hvg300 — 16 training patients, k=[2, 4, 8, 16], fixed budget B=621 spots, 3 seeds x 4 folds
2026-07-31 05:46:03      fold 0: 12 cells, k2=0.0995  k4=0.1325  k8=0.1900  k16=0.1703
2026-07-31 05:47:00      fold 1: 12 cells, k2=0.1094  k4=0.0946  k8=0.0857  k16=0.0936
2026-07-31 05:47:50      fold 2: 12 cells, k2=0.1289  k4=0.0892  k8=0.1077  k16=0.0641
2026-07-31 05:48:30      fold 3: 12 cells, k2=0.2204  k4=0.1909  k8=0.1357  k16=0.2149
2026-07-31 05:48:30  ### DONE pg_kidney_hvg300 in 3.5 min  (B=621 spots, fixed)
2026-07-31 05:48:30        k= 2 patients  per-slide=0.1395  per-gene=0.1690  spots=621  epochs=114  n=12
2026-07-31 05:48:30        k= 4 patients  per-slide=0.1268  per-gene=0.1871  spots=621  epochs=49  n=12
2026-07-31 05:48:30        k= 8 patients  per-slide=0.1298  per-gene=0.2110  spots=621  epochs=43  n=12
2026-07-31 05:48:30        k=16 patients  per-slide=0.1357  per-gene=0.2485  spots=621  epochs=42  n=12
2026-07-31 05:48:30        -> breadth effect at fixed budget: -0.0038 per-slide going from 2 to 16 patients
2026-07-31 05:48:31  ### START pg_bc_hvg300 — 17 training patients, k=[2, 4, 8, 16, 17], fixed budget B=1152 spots, 3 seeds x 4 folds
2026-07-31 05:50:03      fold 0: 15 cells, k2=0.0874  k4=0.0846  k8=0.0970  k16=0.1132  k17=0.1144
2026-07-31 05:51:29      fold 1: 15 cells, k2=0.0722  k4=0.1185  k8=0.1037  k16=0.1273  k17=0.1294
2026-07-31 05:52:45      fold 2: 15 cells, k2=0.1141  k4=0.1399  k8=0.1595  k16=0.1589  k17=0.1718
2026-07-31 05:54:11      fold 3: 15 cells, k2=0.1319  k4=0.1571  k8=0.1552  k16=0.1734  k17=0.1713
2026-07-31 05:54:11  ### DONE pg_bc_hvg300 in 5.7 min  (B=1152 spots, fixed)
2026-07-31 05:54:11        k= 2 patients  per-slide=0.1014  per-gene=0.1467  spots=1152  epochs=66  n=12
2026-07-31 05:54:11        k= 4 patients  per-slide=0.1250  per-gene=0.1614  spots=1152  epochs=23  n=12
2026-07-31 05:54:11        k= 8 patients  per-slide=0.1289  per-gene=0.1689  spots=1152  epochs=29  n=12
2026-07-31 05:54:11        k=16 patients  per-slide=0.1432  per-gene=0.1944  spots=1152  epochs=32  n=12
2026-07-31 05:54:11        k=17 patients  per-slide=0.1467  per-gene=0.1861  spots=1152  epochs=31  n=12
2026-07-31 05:54:11        -> breadth effect at fixed budget: +0.0453 per-slide going from 2 to 17 patients
2026-07-31 05:54:11  === breadth-vs-depth complete ===
