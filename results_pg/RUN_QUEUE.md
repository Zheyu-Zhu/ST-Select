2026-07-31 04:48:28  ========================================================================
2026-07-31 04:48:28  QUEUE START — 7 jobs pending, estimated 2606 min = 43.4 h
2026-07-31 04:48:28  ========================================================================
2026-07-31 04:48:28  [1/7] START celltype-budget (t1, est 35 min)
2026-07-31 04:48:28          Does the cell-type task saturate at the same budget as expression? Caches exist; scaled from the 6.6/10.9/13.6 min gene-panel runs.
2026-07-31 05:42:03  [1/7] DONE  celltype-budget in 53.6 min (est 35) rc=0
2026-07-31 05:42:03  [2/7] START breadth-vs-depth (t1, est 120 min)
2026-07-31 05:42:03          Fixed spot budget, varying patient count — the mechanism behind the 5%/50%/75% spread. New experiment; estimate is the least certain here.
2026-07-31 05:54:12  [2/7] DONE  breadth-vs-depth in 12.1 min (est 120) rc=0
2026-07-31 05:54:12  [3/7] START uni-al-her2-rngfix (t2, est 336 min)
2026-07-31 05:54:12          Re-run of E19 after the per-checkpoint reseed fix; removes the initialisation caveat. 336 min measured for the identical job.
2026-07-31 11:00:42  [3/7] DONE  uni-al-her2-rngfix in 306.5 min (est 336) rc=0
2026-07-31 11:00:42  [4/7] START uni-al-kidney (t2, est 640 min)
2026-07-31 11:00:42          Replicates the AL null on a 2nd cohort with the 304M model — currently HER2-only. 336 min x the 1.90 Kidney/HER2 cost ratio.
2026-08-02 00:09:40  ========================================================================
2026-08-02 00:09:40  QUEUE START — 3 jobs pending, estimated 1475 min = 24.6 h
2026-08-02 00:09:40  ========================================================================
2026-08-02 00:09:40  [1/7] SKIP celltype-budget — already produced results_pg/budget/controlled/pg_her2_celltype/results.json
2026-08-02 00:09:40  [2/7] SKIP breadth-vs-depth — already produced results_pg/breadth_depth/pg_bc_hvg300/results.json
2026-08-02 00:09:40  [3/7] SKIP uni-al-her2-rngfix — already produced results_pg/uni_al/rngfix/pg_her2_hvg300/results.json
2026-08-02 00:09:40  [4/7] SKIP uni-al-kidney — already produced results_pg/uni_al/pg_kidney_hvg300/results.json
2026-08-02 00:09:40  [5/7] START uni-budget-her2 (t2, est 750 min)
2026-08-02 00:09:40          The budget curve on the best model, 7 points. Budget fractions sum to 2.9 full-dataset passes x the 250 min measured ceiling.
