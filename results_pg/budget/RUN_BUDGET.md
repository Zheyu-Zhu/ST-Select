2026-07-29 22:32:01  === PILOT: does early stopping restore a monotone budget curve? ===
2026-07-29 22:32:01  ### START [pilot_legacy] pg_her2_hvg300 arm=legacy methods=['random'] seeds=1 epochs=50 val_frac=0.0
2026-07-29 22:36:24  ### DONE  [pilot_legacy] pg_her2_hvg300 in 4.4 min
      random: 5%=0.3027  10%=0.2731  20%=0.2643  30%=0.2654  50%=0.2606  75%=0.2569  100%=0.2607   (n=4 runs/point)
              vs-100%: 5%=116%  10%=105%  20%=101%  30%=102%  50%=100%  75%=99%  100%=100%
              >=95% of full at: 5%
              NOTE peak at 5% (0.3027), not at 100% (0.2607) - curve still non-monotone
2026-07-29 22:36:24  ### START [pilot_controlled] pg_her2_hvg300 arm=controlled methods=['random'] seeds=1 epochs=200 val_frac=0.15
2026-07-29 22:38:36  ### DONE  [pilot_controlled] pg_her2_hvg300 in 2.2 min
      random: 5%=0.3095  10%=0.3198  20%=0.3069  30%=0.2984  50%=0.3123  75%=0.2894  100%=0.2836   (n=4 runs/point)
              vs-100%: 5%=109%  10%=113%  20%=108%  30%=105%  50%=110%  75%=102%  100%=100%
              >=95% of full at: 5%
              NOTE peak at 10% (0.3198), not at 100% (0.2836) - curve still non-monotone
2026-07-29 22:38:36  === PILOT done - compare the two curves above before the full run ===
2026-07-29 22:39:13  --- BEFORE/AFTER  pg_her2_hvg300  method=random ---
2026-07-29 22:39:13        legacy (50 ep, no val)
2026-07-29 22:39:13          curve   : 5%=0.3027  10%=0.2731  20%=0.2643  30%=0.2654  50%=0.2606  75%=0.2569  100%=0.2607
2026-07-29 22:39:13          vs 100% : 5%=116%  10%=105%  20%=101%  30%=102%  50%=100%  75%=99%  100%=100%
2026-07-29 22:39:13          monotone: spearman(budget, pcc) = -0.857   peak at 5%  <-- STILL INVERTED
2026-07-29 22:39:13          per-run budget reaching 95% of its own 100%: 5%, 5%, 5%, 5%   (median 5%)
2026-07-29 22:39:13        controlled (early stop)
2026-07-29 22:39:13          curve   : 5%=0.3095  10%=0.3198  20%=0.3069  30%=0.2984  50%=0.3123  75%=0.2894  100%=0.2836
2026-07-29 22:39:13          vs 100% : 5%=109%  10%=113%  20%=108%  30%=105%  50%=110%  75%=102%  100%=100%
2026-07-29 22:39:13          monotone: spearman(budget, pcc) = -0.714   peak at 10%  <-- STILL INVERTED
2026-07-29 22:39:13          per-run budget reaching 95% of its own 100%: 5%, 5%, 5%, 5%   (median 5%)
2026-07-29 22:41:01  === FULL [controlled] datasets=['pg_her2_hvg300', 'pg_bc_hvg300', 'pg_kidney_hvg300'] methods=['random'] seeds=3 ===
2026-07-29 22:41:01  ### START [controlled] pg_her2_hvg300 arm=controlled methods=['random'] seeds=3 epochs=200 val_frac=0.15
2026-07-29 22:47:36  ### DONE  [controlled] pg_her2_hvg300 in 6.6 min
      random: 5%=0.3067  10%=0.3118  20%=0.3018  30%=0.2946  50%=0.2978  75%=0.2850  100%=0.2825   (n=12 runs/point)
              vs-100%: 5%=109%  10%=110%  20%=107%  30%=104%  50%=105%  75%=101%  100%=100%
              >=95% of full at: 5%
              NOTE peak at 10% (0.3118), not at 100% (0.2825) - curve still non-monotone
2026-07-29 22:47:36  ### START [controlled] pg_bc_hvg300 arm=controlled methods=['random'] seeds=3 epochs=200 val_frac=0.15
2026-07-29 22:58:28  ### DONE  [controlled] pg_bc_hvg300 in 10.9 min
      random: 5%=0.1311  10%=0.1300  20%=0.1297  30%=0.1381  50%=0.1304  75%=0.1427  100%=0.1460   (n=12 runs/point)
              vs-100%: 5%=90%  10%=89%  20%=89%  30%=95%  50%=89%  75%=98%  100%=100%
              >=95% of full at: 75%
2026-07-29 22:58:28  ### START [controlled] pg_kidney_hvg300 arm=controlled methods=['random'] seeds=3 epochs=200 val_frac=0.15
2026-07-29 23:12:06  ### DONE  [controlled] pg_kidney_hvg300 in 13.6 min
      random: 5%=0.1430  10%=0.1586  20%=0.1742  30%=0.1845  50%=0.1914  75%=0.2037  100%=0.2054   (n=12 runs/point)
              vs-100%: 5%=70%  10%=77%  20%=85%  30%=90%  50%=93%  75%=99%  100%=100%
              >=95% of full at: 75%
2026-07-29 23:12:06  === FULL [controlled] complete ===
2026-07-29 23:12:12  === FULL [legacy] datasets=['pg_her2_hvg300', 'pg_bc_hvg300', 'pg_kidney_hvg300'] methods=['random'] seeds=3 ===
2026-07-29 23:12:12  ### START [legacy] pg_her2_hvg300 arm=legacy methods=['random'] seeds=3 epochs=50 val_frac=0.0
2026-07-29 23:25:11  ### DONE  [legacy] pg_her2_hvg300 in 13.0 min
      random: 5%=0.2936  10%=0.2667  20%=0.2588  30%=0.2573  50%=0.2613  75%=0.2585  100%=0.2612   (n=12 runs/point)
              vs-100%: 5%=112%  10%=102%  20%=99%  30%=99%  50%=100%  75%=99%  100%=100%
              >=95% of full at: 5%
              NOTE peak at 5% (0.2936), not at 100% (0.2612) - curve still non-monotone
2026-07-29 23:25:11  ### START [legacy] pg_bc_hvg300 arm=legacy methods=['random'] seeds=3 epochs=50 val_frac=0.0
2026-07-29 23:53:28  ### DONE  [legacy] pg_bc_hvg300 in 28.3 min
      random: 5%=0.1470  10%=0.1402  20%=0.1311  30%=0.1285  50%=0.1289  75%=0.1297  100%=0.1315   (n=12 runs/point)
              vs-100%: 5%=112%  10%=107%  20%=100%  30%=98%  50%=98%  75%=99%  100%=100%
              >=95% of full at: 5%
              NOTE peak at 5% (0.1470), not at 100% (0.1315) - curve still non-monotone
2026-07-29 23:53:28  ### START [legacy] pg_kidney_hvg300 arm=legacy methods=['random'] seeds=3 epochs=50 val_frac=0.0
2026-07-30 00:16:50  ### DONE  [legacy] pg_kidney_hvg300 in 23.4 min
      random: 5%=0.1596  10%=0.1635  20%=0.1950  30%=0.2041  50%=0.2176  75%=0.2273  100%=0.2275   (n=12 runs/point)
              vs-100%: 5%=70%  10%=72%  20%=86%  30%=90%  50%=96%  75%=100%  100%=100%
              >=95% of full at: 50%
2026-07-30 00:16:50  === FULL [legacy] complete ===
2026-07-30 00:16:57  --- BEFORE/AFTER  pg_her2_hvg300  method=random ---
2026-07-30 00:16:57        legacy (50 ep, no val)
2026-07-30 00:16:57          curve   : 5%=0.3027  10%=0.2731  20%=0.2643  30%=0.2654  50%=0.2606  75%=0.2569  100%=0.2607
2026-07-30 00:16:57          vs 100% : 5%=116%  10%=105%  20%=101%  30%=102%  50%=100%  75%=99%  100%=100%
2026-07-30 00:16:57          monotone: spearman(budget, pcc) = -0.857   peak at 5%  <-- STILL INVERTED
2026-07-30 00:16:57          per-run budget reaching 95% of its own 100%: 5%, 5%, 5%, 5%   (median 5%)
2026-07-30 00:16:57        controlled (early stop)
2026-07-30 00:16:57          curve   : 5%=0.3095  10%=0.3198  20%=0.3069  30%=0.2984  50%=0.3123  75%=0.2894  100%=0.2836
2026-07-30 00:16:57          vs 100% : 5%=109%  10%=113%  20%=108%  30%=105%  50%=110%  75%=102%  100%=100%
2026-07-30 00:16:57          monotone: spearman(budget, pcc) = -0.714   peak at 10%  <-- STILL INVERTED
2026-07-30 00:16:57          per-run budget reaching 95% of its own 100%: 5%, 5%, 5%, 5%   (median 5%)
2026-07-30 00:16:57        legacy full
2026-07-30 00:16:57          curve   : 5%=0.2936  10%=0.2667  20%=0.2588  30%=0.2573  50%=0.2613  75%=0.2585  100%=0.2612
2026-07-30 00:16:57          vs 100% : 5%=112%  10%=102%  20%=99%  30%=99%  50%=100%  75%=99%  100%=100%
2026-07-30 00:16:57          monotone: spearman(budget, pcc) = -0.536   peak at 5%  <-- STILL INVERTED
2026-07-30 00:16:57          per-run budget reaching 95% of its own 100%: 5%, 5%, 5%, 5%, 5%, 5%, 5%, 5%, 5%, 5%, 5%, 5%   (median 5%)
2026-07-30 00:16:57          epochs run: 5%=50  10%=50  20%=50  30%=50  50%=50  75%=50  100%=50   (flat = fixed-epoch arm; rising with budget = the confound)
2026-07-30 00:16:57        controlled full
2026-07-30 00:16:57          curve   : 5%=0.3067  10%=0.3118  20%=0.3018  30%=0.2946  50%=0.2978  75%=0.2850  100%=0.2825
2026-07-30 00:16:57          vs 100% : 5%=109%  10%=110%  20%=107%  30%=104%  50%=105%  75%=101%  100%=100%
2026-07-30 00:16:57          monotone: spearman(budget, pcc) = -0.929   peak at 10%  <-- STILL INVERTED
2026-07-30 00:16:57          per-run budget reaching 95% of its own 100%: 5%, 5%, 5%, 5%, 5%, 5%, 5%, 5%, 5%, 5%, 5%, 5%   (median 5%)
2026-07-30 00:16:57          epochs run: 5%=42  10%=28  20%=22  30%=24  50%=19  75%=20  100%=16   (flat = fixed-epoch arm; rising with budget = the confound)
2026-07-30 00:16:57  --- BEFORE/AFTER  pg_bc_hvg300  method=random ---
2026-07-30 00:16:57        legacy full
2026-07-30 00:16:57          curve   : 5%=0.1470  10%=0.1402  20%=0.1311  30%=0.1285  50%=0.1289  75%=0.1297  100%=0.1315
2026-07-30 00:16:57          vs 100% : 5%=112%  10%=107%  20%=100%  30%=98%  50%=98%  75%=99%  100%=100%
2026-07-30 00:16:57          monotone: spearman(budget, pcc) = -0.500   peak at 5%  <-- STILL INVERTED
2026-07-30 00:16:57          per-run budget reaching 95% of its own 100%: 5%, 5%, 5%, 5%, 5%, 5%, 5%, 5%, 5%, 5%, 5%, 5%   (median 5%)
2026-07-30 00:16:57          epochs run: 5%=50  10%=50  20%=50  30%=50  50%=50  75%=50  100%=50   (flat = fixed-epoch arm; rising with budget = the confound)
2026-07-30 00:16:57        controlled full
2026-07-30 00:16:57          curve   : 5%=0.1311  10%=0.1300  20%=0.1297  30%=0.1381  50%=0.1304  75%=0.1427  100%=0.1460
2026-07-30 00:16:57          vs 100% : 5%=90%  10%=89%  20%=89%  30%=95%  50%=89%  75%=98%  100%=100%
2026-07-30 00:16:57          monotone: spearman(budget, pcc) = +0.679   peak at 100%
2026-07-30 00:16:57          per-run budget reaching 95% of its own 100%: 30%, 100%, 75%, 100%, 100%, 5%, 10%, 5%, 5%, 5%, 50%, 75%   (median 50%)
2026-07-30 00:16:57          epochs run: 5%=21  10%=15  20%=13  30%=12  50%=11  75%=11  100%=11   (flat = fixed-epoch arm; rising with budget = the confound)
2026-07-30 00:16:57  --- BEFORE/AFTER  pg_kidney_hvg300  method=random ---
2026-07-30 00:16:57        legacy full
2026-07-30 00:16:57          curve   : 5%=0.1596  10%=0.1635  20%=0.1950  30%=0.2041  50%=0.2176  75%=0.2273  100%=0.2275
2026-07-30 00:16:57          vs 100% : 5%=70%  10%=72%  20%=86%  30%=90%  50%=96%  75%=100%  100%=100%
2026-07-30 00:16:57          monotone: spearman(budget, pcc) = +1.000   peak at 100%
2026-07-30 00:16:57          per-run budget reaching 95% of its own 100%: 75%, 100%, 75%, 30%, 50%, 50%, 75%, 75%, 75%, 75%, 30%, 20%   (median 75%)
2026-07-30 00:16:57          epochs run: 5%=50  10%=50  20%=50  30%=50  50%=50  75%=50  100%=50   (flat = fixed-epoch arm; rising with budget = the confound)
2026-07-30 00:16:57        controlled full
2026-07-30 00:16:57          curve   : 5%=0.1430  10%=0.1586  20%=0.1742  30%=0.1845  50%=0.1914  75%=0.2037  100%=0.2054
2026-07-30 00:16:57          vs 100% : 5%=70%  10%=77%  20%=85%  30%=90%  50%=93%  75%=99%  100%=100%
2026-07-30 00:16:57          monotone: spearman(budget, pcc) = +1.000   peak at 100%
2026-07-30 00:16:57          per-run budget reaching 95% of its own 100%: 20%, 75%, 50%, 30%, 100%, 75%, 30%, 75%, 30%, 100%, 75%, 100%   (median 75%)
2026-07-30 00:16:57          epochs run: 5%=56  10%=40  20%=31  30%=33  50%=26  75%=25  100%=23   (flat = fixed-epoch arm; rising with budget = the confound)
2026-07-30 00:18:08  --- BEFORE/AFTER  pg_her2_hvg300  method=random ---
2026-07-30 00:18:09        legacy (50 ep, no val)
2026-07-30 00:18:09          curve   : 5%=0.3027  10%=0.2731  20%=0.2643  30%=0.2654  50%=0.2606  75%=0.2569  100%=0.2607
2026-07-30 00:18:09          vs 100% : 5%=116%  10%=105%  20%=101%  30%=102%  50%=100%  75%=99%  100%=100%
2026-07-30 00:18:09          monotone: spearman(budget, pcc) = -0.857   peak at 5%  <-- STILL INVERTED
2026-07-30 00:18:09          per-run budget reaching 95% of its own 100%: 5%, 5%, 5%, 5%   (median 5%)
2026-07-30 00:18:09        controlled (early stop)
2026-07-30 00:18:09          curve   : 5%=0.3095  10%=0.3198  20%=0.3069  30%=0.2984  50%=0.3123  75%=0.2894  100%=0.2836
2026-07-30 00:18:09          vs 100% : 5%=109%  10%=113%  20%=108%  30%=105%  50%=110%  75%=102%  100%=100%
2026-07-30 00:18:09          monotone: spearman(budget, pcc) = -0.714   peak at 10%  <-- STILL INVERTED
2026-07-30 00:18:09          per-run budget reaching 95% of its own 100%: 5%, 5%, 5%, 5%   (median 5%)
2026-07-30 00:18:09        legacy full
2026-07-30 00:18:09          curve   : 5%=0.2936  10%=0.2667  20%=0.2588  30%=0.2573  50%=0.2613  75%=0.2585  100%=0.2612
2026-07-30 00:18:09          vs 100% : 5%=112%  10%=102%  20%=99%  30%=99%  50%=100%  75%=99%  100%=100%
2026-07-30 00:18:09          monotone: spearman(budget, pcc) = -0.536   peak at 5%  <-- STILL INVERTED
2026-07-30 00:18:09          per-run budget reaching 95% of its own 100%: 5%, 5%, 5%, 5%, 5%, 5%, 5%, 5%, 5%, 5%, 5%, 5%   (median 5%)
2026-07-30 00:18:09          epochs run: 5%=50  10%=50  20%=50  30%=50  50%=50  75%=50  100%=50   (flat = fixed-epoch arm; rising with budget = the confound)
2026-07-30 00:18:09        controlled full
2026-07-30 00:18:09          curve   : 5%=0.3067  10%=0.3118  20%=0.3018  30%=0.2946  50%=0.2978  75%=0.2850  100%=0.2825
2026-07-30 00:18:09          vs 100% : 5%=109%  10%=110%  20%=107%  30%=104%  50%=105%  75%=101%  100%=100%
2026-07-30 00:18:09          monotone: spearman(budget, pcc) = -0.929   peak at 10%  <-- STILL INVERTED
2026-07-30 00:18:09          per-run budget reaching 95% of its own 100%: 5%, 5%, 5%, 5%, 5%, 5%, 5%, 5%, 5%, 5%, 5%, 5%   (median 5%)
2026-07-30 00:18:09          epochs run: 5%=42  10%=28  20%=22  30%=24  50%=19  75%=20  100%=16   (flat = fixed-epoch arm; rising with budget = the confound)
2026-07-30 00:18:09  --- BEFORE/AFTER  pg_bc_hvg300  method=random ---
2026-07-30 00:18:09        legacy full
2026-07-30 00:18:09          curve   : 5%=0.1470  10%=0.1402  20%=0.1311  30%=0.1285  50%=0.1289  75%=0.1297  100%=0.1315
2026-07-30 00:18:09          vs 100% : 5%=112%  10%=107%  20%=100%  30%=98%  50%=98%  75%=99%  100%=100%
2026-07-30 00:18:09          monotone: spearman(budget, pcc) = -0.500   peak at 5%  <-- STILL INVERTED
2026-07-30 00:18:09          per-run budget reaching 95% of its own 100%: 5%, 5%, 5%, 5%, 5%, 5%, 5%, 5%, 5%, 5%, 5%, 5%   (median 5%)
2026-07-30 00:18:09          epochs run: 5%=50  10%=50  20%=50  30%=50  50%=50  75%=50  100%=50   (flat = fixed-epoch arm; rising with budget = the confound)
2026-07-30 00:18:09        controlled full
2026-07-30 00:18:09          curve   : 5%=0.1311  10%=0.1300  20%=0.1297  30%=0.1381  50%=0.1304  75%=0.1427  100%=0.1460
2026-07-30 00:18:09          vs 100% : 5%=90%  10%=89%  20%=89%  30%=95%  50%=89%  75%=98%  100%=100%
2026-07-30 00:18:09          monotone: spearman(budget, pcc) = +0.679   peak at 100%
2026-07-30 00:18:09          per-run budget reaching 95% of its own 100%: 30%, 100%, 75%, 100%, 100%, 5%, 10%, 5%, 5%, 5%, 50%, 75%   (median 50%)
2026-07-30 00:18:09          epochs run: 5%=21  10%=15  20%=13  30%=12  50%=11  75%=11  100%=11   (flat = fixed-epoch arm; rising with budget = the confound)
2026-07-30 00:18:09  --- BEFORE/AFTER  pg_kidney_hvg300  method=random ---
2026-07-30 00:18:09        legacy full
2026-07-30 00:18:09          curve   : 5%=0.1596  10%=0.1635  20%=0.1950  30%=0.2041  50%=0.2176  75%=0.2273  100%=0.2275
2026-07-30 00:18:09          vs 100% : 5%=70%  10%=72%  20%=86%  30%=90%  50%=96%  75%=100%  100%=100%
2026-07-30 00:18:09          monotone: spearman(budget, pcc) = +1.000   peak at 100%
2026-07-30 00:18:09          per-run budget reaching 95% of its own 100%: 75%, 100%, 75%, 30%, 50%, 50%, 75%, 75%, 75%, 75%, 30%, 20%   (median 75%)
2026-07-30 00:18:09          epochs run: 5%=50  10%=50  20%=50  30%=50  50%=50  75%=50  100%=50   (flat = fixed-epoch arm; rising with budget = the confound)
2026-07-30 00:18:09        controlled full
2026-07-30 00:18:09          curve   : 5%=0.1430  10%=0.1586  20%=0.1742  30%=0.1845  50%=0.1914  75%=0.2037  100%=0.2054
2026-07-30 00:18:09          vs 100% : 5%=70%  10%=77%  20%=85%  30%=90%  50%=93%  75%=99%  100%=100%
2026-07-30 00:18:09          monotone: spearman(budget, pcc) = +1.000   peak at 100%
2026-07-30 00:18:09          per-run budget reaching 95% of its own 100%: 20%, 75%, 50%, 30%, 100%, 75%, 30%, 75%, 30%, 100%, 75%, 100%   (median 75%)
2026-07-30 00:18:09          epochs run: 5%=56  10%=40  20%=31  30%=33  50%=26  75%=25  100%=23   (flat = fixed-epoch arm; rising with budget = the confound)
2026-07-31 04:48:36  === FULL [controlled] datasets=['pg_her2_celltype', 'pg_bc_celltype', 'pg_kidney_celltype'] methods=['random'] seeds=3 ===
2026-07-31 04:48:36  ### START [controlled] pg_her2_celltype arm=controlled methods=['random'] seeds=3 epochs=200 val_frac=0.15
2026-07-31 05:00:28  ### DONE  [controlled] pg_her2_celltype in 11.9 min
      random: 5%=0.0224  10%=0.0381  20%=0.0442  30%=0.0527  50%=0.0669  75%=0.0788  100%=0.0824   (n=12 runs/point)
              vs-100%: 5%=27%  10%=46%  20%=54%  30%=64%  50%=81%  75%=96%  100%=100%
              >=95% of full at: 75%
2026-07-31 05:00:28  ### START [controlled] pg_bc_celltype arm=controlled methods=['random'] seeds=3 epochs=200 val_frac=0.15
2026-07-31 05:20:20  ### DONE  [controlled] pg_bc_celltype in 19.9 min
      random: 5%=0.0307  10%=0.0513  20%=0.0717  30%=0.0906  50%=0.1197  75%=0.1313  100%=0.1588   (n=12 runs/point)
              vs-100%: 5%=19%  10%=32%  20%=45%  30%=57%  50%=75%  75%=83%  100%=100%
              >=95% of full at: 100%
2026-07-31 05:20:20  ### START [controlled] pg_kidney_celltype arm=controlled methods=['random'] seeds=3 epochs=200 val_frac=0.15
2026-07-31 05:42:02  ### DONE  [controlled] pg_kidney_celltype in 21.7 min
      random: 5%=0.0430  10%=0.0825  20%=0.1118  30%=0.1328  50%=0.1566  75%=0.1685  100%=0.1752   (n=12 runs/point)
              vs-100%: 5%=25%  10%=47%  20%=64%  30%=76%  50%=89%  75%=96%  100%=100%
              >=95% of full at: 75%
2026-07-31 05:42:02  === FULL [controlled] complete ===
