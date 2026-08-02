# ST-Select — Run Details & Methodological Audit

> Proactive end-to-end review of the training/evaluation pipeline, plus the exact
> recipe every reported number was produced with. Written 2026-07-25.
> **Bottom line: the two failure modes that would invalidate a benchmark are ruled
> out** (no patient leakage; feature↔label alignment verified by a shuffle control).
> Several non-invalidating methodological wrinkles are documented in §5.

---

## 1. Data pipeline

- **Source:** HEST-1k `patches/*.h5` (224×224 H&E, uint8) + `st/*.h5ad` (counts).
- **HVG selection:** all slides of a dataset concatenated → `normalize_total(1e4)` →
  `log1p` → `highly_variable_genes(n_top=1000, flavor="seurat")`; ranked by
  `dispersions_norm`; **top-300** columns kept. `top-N` (10/50/100/200/300) = the
  first-N of these dispersion-ranked columns (NOT the best-correlated genes).
- **Regression target:**
  - `norm` (default): `log1p(CP10k)` — the normalized value above.
  - `raw`: `log1p(raw counts)` — retains per-spot depth/density; ≈2× higher PCC.
  HVG *selection* is always on normalized data; only the target column differs.
- **image↔expression join:** by spot **barcode**, with a hard error on any
  mismatch (no positional/silent truncation). Verified: `align_expression_to_patches`.
- **Caches** (`feature_cache/*.npz`): frozen-feature caches store `features (N,1024)`;
  image-index caches store `patch_pos` (row into the slide's patch h5) and read
  pixels lazily. Both store `expressions, positions, patient_ids, sample_ids, gene_names`.

## 2. Cross-validation split

- **Patient-level 4-fold** (`PatientKFold`, `seed=42`): all slides of a patient stay
  in one fold. HER2 = 8 patients → 6 train / 2 test per fold; cSCC = 4 → 3/1;
  He-breast = 23 → ~17/6. **Audited empirically: 0 patient overlap, 0 slide overlap
  in every fold.**
- Multi-seed runs (`n_seeds=3` → seeds 42/43/44) **share the same fold structure**;
  the seed varies only the acquisition + training RNG. → error bars reflect
  training/acquisition variance, not fold variance (see §5-D).
- A `--split slide` option exists (leakier, literature-comparable) and reads ~2×
  higher — used only as an ablation.

## 3. Model + training recipe

Common: `AdamW`, `weight_decay=1e-5`, `CosineAnnealingLR(T_max=epochs)`, `MSELoss`,
**bf16 autocast on CUDA** (no GradScaler; no-op on CPU). A fresh optimizer is built
each `train()` call. No validation/early-stopping in benchmark runs (fixed epochs).

| run family | model | backbone | epochs/round · full | batch | lr | bb-lr× |
|---|---|---|---|---|---|---|
| **frozen AL benchmark** (`results_final/`) | `feature_predictor` (MLP) | frozen UNI ViT-L feats, backbone = Identity | 40 · 60 | 128 | 1e-4 | — |
| **frozen ladder / norm_test** | `feature_predictor` | frozen UNI/DenseNet feats | 60 · 60 | 128 | 1e-4 | — |
| **e2e ceiling (raw)** (`ceiling_uni_raw/`) | `uni_regressor` | UNI ViT-L, trainable, grad-ckpt | — · 15 | 16 | 1e-4 | 0.1 |
| **e2e ceiling (cp10k)** (`ceiling/`) | `st_net` | DenseNet-121, trainable | — · 40 | 32 | 1e-4 | 1.0 |
| **Phase 3 trained AL** (`phase3_trained*/`) | `st_net` | DenseNet-121, trainable | 15 · 15 | 32 | 1e-4 | 1.0 |

## 4. Active-learning loop (`ActiveLearningLoop`)

1. **Initial pool** = a random (seeded) set sized to the smallest budget target
   (e.g. 5% of the training fold). Shared across strategies, so all methods start
   identical (this is why the 5% column is identical for every strategy).
2. **Budget targets** are derived from `budget_ratios × n_train`. The pool is grown
   by `budget_per_round` picks via the strategy until it reaches each target; the
   model is (re)trained and evaluated at every target → an AL curve.
3. **Per-checkpoint reset:** `model.apply(_reset_weights_partial)` resets every
   `nn.Linear` with `out_features < 1024` (i.e. the MLP **head**) before retraining.
   - Frozen mode: backbone = Identity, so each budget point retrains the head from
     scratch on the current pool → **clean, independent-budget models**.
   - Image mode: the conv/ViT **backbone is NOT reset** (its Linears are ≥1024-wide
     or Conv), so it **carries over across budget checkpoints** → see §5-A.
4. **Representation handed to strategies:** frozen mode → the cached feature matrix;
   image mode → backbone features extracted once per target with `head_only=True`
   so strategies apply `model.head` (never the image backbone again).
5. **Acquisition post-conditions** (`select_next_batch`): returns exactly `k`
   distinct in-pool indices; dedups / tops-up degenerate strategies.

## 5. Evaluation metrics

Computed on the held-out test fold. All conventions reported:
`pcc_per_gene` (per gene across spots, mean over genes — **primary**),
`pcc_per_gene_by_slide` (per gene **within each slide**, mean over slides — the
PI's convention, removes per-slide batch effects), `pcc_per_spot` (per spot across
genes — inflated), `pcc_topN` / `pcc_topN_by_slide` (top-N ranked genes, pooled /
slide-level), `mse`, `mae`, and — added 2026-07-24 — `ari`/`nmi` (spatial-domain
recovery: KMeans on predicted vs true expression per slide) and `selection_diversity`
(feature/spatial spread of the selected spots). Significance vs `random`: paired
Wilcoxon across the (fold,seed) pairs.

## 6. Audit — verified clean

| check | method | result |
|---|---|---|
| Patient/slide fold leakage | intersect train/test patient & slide sets, all folds | **0 overlap ✓** |
| Feature↔label alignment | shuffle control: train on shuffled labels | real PCC **0.064** vs shuffled **−0.010 ≈ 0** ✓ |
| New metrics don't disturb core PCC | re-run reproduces prior per-gene/per-slide | 0.0517 vs prior 0.0518 ✓ |
| Image-mode AL strategies | all 6 Phase-3 strategies run on real DenseNet+cSCC | no crash ✓ |

## 7. Known methodological wrinkles (non-invalidating; symmetric across strategies)

**A. Image-mode budget curves accumulate backbone training.** Because the backbone
is not reset between budget checkpoints (§4.3), the 30% image-mode model has a
backbone that already saw the 10% + 30% pools. → the budget-CURVE *shape* in image
mode is confounded (not independent-budget models). **Does not affect** the
strategy-vs-random comparison (all strategies share the confound) or the 100%
ceiling. *Fix if we rely on image-mode budget curves:* re-instantiate the model
(reload pretrained backbone + fresh head) at each checkpoint — costs a re-run.

**B. `full_supervision` epoch mismatch in `results_final`.** There `full_epochs=60`
but `epochs_per_round=40`, so the full-supervision "upper bound" trains 20 more
epochs than the AL curve's 100% point and mildly **overfits** — which is why
full_supervision sometimes reads *below* the AL-100% number. It is not a strict
upper bound there. (Phase 3 uses `epochs_per_round == full_epochs = 15`, so it is
clean.) *Fix:* set the two equal for the ceiling comparison.

**C. HVG selection uses all slides (incl. test).** The gene *panel* and dispersion
ranking are computed on the full dataset before the CV split — standard practice
(unsupervised w.r.t. the image→expression task, panel fixed a priori), but a mild
form of information sharing. Per-spot normalization itself has no cross-spot leak.

**D. Multi-seed error bars are training-variance only** (same folds across seeds,
§2). To get fold-variance error bars, vary the fold seed too.

**E. `_reset_weights_partial` is a fragile heuristic** (`out_features < 1024`). It
happens to reset exactly the head and preserve the DenseNet/ViT-L backbones we use,
but a backbone containing a `<1024`-wide `Linear` would be silently partially reset.
*Fix:* reset by explicit module reference (`model.head`) instead of a width test.

## 8. What this means for the results

None of §7 changes the headline: **no AL strategy gives a robust, consistent gain
over random** — the comparison is symmetric under A–E, and the two catastrophic
failure modes (§6) are ruled out. A–E are polish items for a camera-ready benchmark,
not correctness bugs. B is worth fixing before quoting `full_supervision` as an
upper bound; A is worth fixing only if we report image-mode budget *curves* (vs the
100% ceiling we mostly use).
