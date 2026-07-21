# Active Learning for Spatial Transcriptomics Gene-Expression Prediction

A benchmark framework comparing active-learning (AL) acquisition strategies for
predicting per-spot gene expression from H&E histology patches (HEST-1k / ST-Net
task). The design rationale, method catalog, and conventions are documented in
[`AL_for_ST_Prediction_Tutorial.md`](AL_for_ST_Prediction_Tutorial.md).

## Install

```bash
pip install -r requirements.txt
# or, as a package:
pip install -e .
```

## Quick check (no data needed)

The smoke test runs the full benchmark on tiny synthetic data in seconds:

```bash
python tests/test_smoke.py
# or: pytest tests/test_smoke.py -q
```

## List available AL strategies

```bash
python -m src.run_experiment --list-methods
```

29 strategies are registered across 6 families (uncertainty, diversity, hybrid,
medical, spatial, RL).

## Run a benchmark

The benchmark uses the **feature-cache fast path** (tutorial §2.9.4): frozen
patch features are extracted once and a lightweight `FeaturePredictor` MLP is
trained over them. This keeps the predictor and the feature-based AL strategies
(BADGE, CoreSet, TOD, ...) on the same representation.

### 1. Prepare a feature cache

Create `<feature_cache_dir>/<dataset_name>.npz` with arrays:

| array | shape | meaning |
|---|---|---|
| `features` | `(N, D)` | frozen patch features (CONCH / UNI / DINOv2 / DenseNet) |
| `expressions` | `(N, G)` | log1p-normalized top-G HVG expression |
| `positions` | `(N, 2)` | level-0 `(x, y)` spot coordinates |
| `patient_ids` | `(N,)` | patient/donor id per spot (for patient k-fold) |
| `sample_ids` | `(N,)` | slide id per spot (optional) |

Helpers: `src/preprocessing/` (HVG selection, position extraction, cropping) and
`src.models.get_feature_extractor(...)` for frozen backbones.

### 2. Run

```bash
python -m src.run_experiment \
  --dataset her2_breast \
  --methods random badge coreset poisson_disk \
  --n-folds 4 --n-seeds 3 \
  --budget-ratios 0.05 0.10 0.20 0.30 0.50 1.00 \
  --pcc-top-n 10 50 100 200 300 \
  --budget-per-round 50 \
  --epochs 50 --full-epochs 100 \
  --device cpu
```

Each AL method runs one acquisition trajectory per (fold, seed) and is evaluated
as its labeled fraction crosses each `--budget-ratios` checkpoint (an AL curve;
`1.0` = full supervision, the shared upper bound). Outputs
`results/<dataset>/results.json` with:

| block | meaning |
|---|---|
| `final_pcc_per_gene` / `mean` / `std` | per-(fold,seed) PCC at the largest budget |
| `by_ratio` | `{method: {ratio: [pcc per (fold,seed)]}}` — the budget sweep |
| `pcc_topN` | `{method: {N: mean per-gene PCC over top-N ranked genes}}` — the top-N sweep |
| `by_family` | rollup into **Random / Uncertainty / Coverage / Hybrid / Spatial / RL** (+ FullSupervision) |
| `comparisons_vs_random` | Wilcoxon significance vs. the `random` baseline |

Plots: `plot_al_curves`, `plot_budget_comparison`, `plot_family_budget_curves`,
`plot_top_n_pcc`, `plot_gene_pcc_heatmap` in `src.evaluation`.

**Preparing a real cache** (barcode-joined image↔expression, 1000 HVGs, ranked
top-300 columns + `gene_names` persisted):

```bash
python scripts/prepare_hest_cache.py --dataset her2_breast \
  --sample-ids <slide ids...>   # or omit to auto-discover under hest_data/st/
```

## Notes & limitations

- **Supported models for the AL benchmark:** `feature_predictor` (default),
  `st_net`, `histogene`, `hist2st`. Image-input (`thitogene`) and
  retrieval/contrastive models (`bleep`, `mclstexp`, `egn`) are implemented but
  need their own training/inference paths and are not wired into `ALTrainer`.
- **Retrieval-bank leakage:** `BLEEP`/`MclSTExp` `build_retrieval_bank(...)`
  accept `train_indices`/`test_indices` and assert disjointness (tutorial §6).
- **PCC convention:** per-gene is primary; per-spot also reported. See tutorial §4.3.
```
