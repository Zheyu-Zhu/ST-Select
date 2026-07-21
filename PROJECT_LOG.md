# ST-Select — Project Log & Handoff

> **Purpose of this file:** a self-contained record so anyone (or a new AI session)
> can pick up the full context. Read this first. It covers: what the project is,
> what was done, what was found, the exact results, known issues, and how to
> continue.

Last updated: 2026-07 (see git log for exact commits).

---

## 1. What this project is

A **benchmark of active-learning (AL) acquisition strategies** for spot-level
spatial-transcriptomics (ST) gene-expression prediction on **HEST-1k**.

- **Task:** predict per-spot gene expression (top-300 HVGs) from H&E image
  features. All AL methods share **one frozen predictor** (frozen UNI pathology-FM
  features + MLP head) so the *only* variable is the **spot-selection strategy**.
- **Question:** does any AL strategy (Uncertainty / Coverage / Hybrid / Spatial)
  beat **Random** spot selection, under honest evaluation?
- **This is an AL-*selection* benchmark**, distinct from prior model-prediction
  benchmarks (e.g. Wang et al. 2025 NatComm [ref 134] compares prediction
  *models*; the Zhu et al. BiB 2025 survey is a paper *list*, not a benchmark).

Design decisions came from the PI (see the "July 13th" answers / the tutorial
`AL_for_ST_Prediction_Tutorial.md`): spot-level acquisition; family-level
comparison; budget sweep {5,10,20,30,50,100}%; datasets HER2/He-breast/cSCC/
DLPFC/CRC-HD; frozen-and-shared features; 1000 HVG → report top-N per-gene PCC.

---

## 2. Headline finding (robust across everything tried)

**Under honest patient-level evaluation, NO AL strategy beats Random spot
selection — and this holds across 3 datasets, 2 CV splits, 2 backbones, and
every PCC convention.** The real bottleneck is **cross-patient generalization**,
not which spots you pick.

- Budget curves rise monotonically with more data (the loop works), but
  *strategy* barely matters: best-AL − Random ≤ +0.0006 (≈1%, within noise).
- This is a clean, publishable **negative benchmark result**. Suggested framing:
  *"When (if ever) does active learning help image-based ST prediction?"*

Ruled out as artifacts (all checked): feature/label misalignment (shuffle control
→ PCC ~0), metric bug, UNI preprocessing mismatch, weak-backbone excuse (holds
under trainable too), and crashes (all runs completed).

---

## 3. Results (headline = `results_final/`, 3 seeds, per-slide PCC)

**Per-slide PCC** (each slide scored separately then averaged — the PI's
preferred convention, removes per-slide batch effects). random / best-AL:

| Dataset | patient-split | slide-split |
|---|---|---|
| cSCC | 0.025 / badge +0.0002 | 0.046 / entropy +0.0006 |
| HER2 | 0.037 / random +0.0000 | 0.051 / badge +0.0004 |
| He-breast | 0.031 / spatial +0.0001 | 0.037 / poisson +0.0001 |

**PCC depends heavily on convention (same predictions, HER2 100%, random):**

| Convention | patient-split | slide-split |
|---|---|---|
| per-spot (literature-common, inflated) | **0.60** | 0.68 |
| per-gene pooled | 0.067 | 0.141 |
| per-slide (PI's choice) | 0.037 | 0.051 |

→ The "0.4–0.7" in the literature is mostly **per-spot + slide-level split +
end-to-end fine-tuned backbones**. We use the stricter per-gene / patient-split /
**frozen features**, hence lower absolute numbers. Not a bug.

**Frozen vs "trainable" backbone (HER2 patient, per-gene pooled, random):**

| sampling | frozen | trainable-projection |
|---|---|---|
| 10% | 0.041 | 0.046 |
| 100% | 0.067 | 0.074 |

⚠️ **IMPORTANT CAVEAT:** our "trainable" = frozen UNI features + a *learnable
projection MLP* on top (`TrainableFeaturePredictor`). It is **NOT** end-to-end
fine-tuning of the image backbone. The true image backbone (DenseNet/ViT over
raw patches) was never fine-tuned. **This is the #1 open item** — the PI/professor
expects a true end-to-end fine-tune could reach ~0.2, which we have not run.

---

## 4. Data & environment

- **Machine used so far:** Apple M4 Pro, 48GB, **MPS** (Apple GPU). Code is
  device-agnostic — `--device cuda` works on the professor's 3080 (see §6).
- **Datasets present locally** (`hest_data/`, gitignored, ~22GB):
  HER2 (`SPA119–154`, 8 patients), cSCC (`NCBI759–770`, 4 patients),
  He-breast (`SPA51–118`, 23 patients). DLPFC/CRC-HD not yet used.
- **UNI feature caches** (`feature_cache/*.npz`, gitignored): `her2_breast_uni`,
  `cscc_uni`, `he_breast_uni` — 1000 HVG selected, ranked top-300 kept, +gene_names.
- **UNI is gated** on HuggingFace — needs `huggingface-cli login` + access grant.

---

## 5. What changed in the code (vs the original package)

Key edits made during this work (all in git history):

1. **MPS/CUDA support** — `src/utils/reproducibility.py::resolve_device`; replaced
   ~20 cuda-only guards. `--device {auto,mps,cuda,cpu}`.
2. **Bug fixes** (found via review + the trainable path):
   - TOD degeneracy (prev_model == current → zero discrepancy) → snapshot prev round.
   - Frozen acquisition seed (all seeds identical) → `get_strategy(seed=...)`.
   - Dead `full_epochs` → full-supervision now uses it (`--full-epochs`).
   - Silent budget under-fill / duplicate picks → post-condition + top-up in
     `select_next_batch`.
   - Cache image↔expression paired positionally w/ silent truncation → **barcode
     join** in `scripts/prepare_hest_cache.py`.
   - TypiClust PCA `SVD did not converge` crash → robust solver fallback.
   - Device mismatch in 13 AL strategies (batch on strategy device, model
     elsewhere) → `model_device()` helper.
   - BADGE double-projection under dynamic features → fixed loop logic.
3. **New features:**
   - Budget-ratio sweep (`--budget-ratios`), top-N PCC sweep (`--pcc-top-n`).
   - Family rollup (`src/al_methods/families.py`): 8 code families → Random/
     Uncertainty/Coverage/Hybrid/Spatial/RL.
   - `--split {patient,slide}` CV grouping.
   - **per-slide PCC metric** (`pcc_per_gene_grouped`) — the PI's primary metric.
   - `TrainableFeaturePredictor` + dynamic-feature AL scoring (loop
     `dynamic_features`): feature-based AL scores on the *current* representation.
   - Extended `tests/test_smoke.py` (TOD>0, seed varies, k distinct, full_epochs).
   - Dataset registry incl. CRC-HD; PatientKFold n_folds>n_patients guard.

Results dirs: `results_final/` = headline (committed). Others
(`results/`, `results_slide/`, `results_v2/`, `results_lite/`, `results_backbone/`)
were intermediate and are gitignored.

---

## 6. How to reproduce / continue (e.g. on the 3080)

```bash
git clone https://github.com/Zheyu-Zhu/ST-Select.git && cd ST-Select
pip install -r requirements.txt        # uncomment timm / huggingface_hub / matplotlib
huggingface-cli login                  # UNI is gated — needs your HF account + access

# 1) download a dataset (HER2 example) — see README §Datasets for tags/ids
#    HER2 = MahmoodLab/hest, ids SPA119..SPA154 + HEST_v1_3_0.csv metadata
# 2) build the UNI feature cache (barcode-joined, 1000 HVG, top-300, patient map):
python scripts/prepare_hest_cache.py --dataset her2_breast \
  --sample-ids SPA119 SPA120 ... SPA154 \
  --patient-map hest_data/HEST_v1_3_0.csv \
  --feature-extractor uni --device cuda
# 3) run the benchmark (patient-level, per-slide PCC recorded automatically):
python -m src.run_experiment --dataset her2_breast --split patient \
  --methods random entropy tod coreset typiclust badge poisson_disk spatial_stratified \
  --budget-ratios 0.05 0.10 0.20 0.30 0.50 1.00 --pcc-top-n 10 50 100 200 300 \
  --n-folds 4 --n-seeds 3 --device cuda --output-dir ./results_final/patient
```

`smoke test`: `python tests/test_smoke.py` (no data needed).

---

## 7. Open items / TODO (priority order)

1. **[HIGH] True end-to-end backbone fine-tuning.** Current "trainable" only
   trains a projection on frozen UNI features. To answer "can train reach ~0.2",
   run `st_net` (trainable DenseNet over raw image patches). Needs the image-input
   path (not the feature cache); `st_net` + `frozen_backbone=False` exist but the
   benchmark loop currently feeds `STFeatureDataset` only — wiring an image
   dataset is required. Slow (backprop CNN over patches).
2. **[MED] Confirm framing with the professor:** negative-result framing
   ("AL ≈ random; bottleneck is cross-patient generalization"). Also which is the
   primary PCC convention (PI said per-slide; literature uses per-spot).
3. **[MED] More AL methods per family** if a "comprehensive" benchmark is wanted
   (currently 1–2 representatives per family out of ~29 implemented).
4. **[LOW] DLPFC / CRC-HD** datasets (registry ready; not downloaded/run).
5. **[LOW] `full_epochs`** set to 60 (was 100 → mild overfit where full-sup < random).

---

## 8. Pointers

- Design rationale & method catalog: `AL_for_ST_Prediction_Tutorial.md`.
- Cited survey (context, not a benchmark): Zhu et al., "A Comprehensive Survey of
  Computer Vision Methods for Spatial Transcriptomics", Briefings in Bioinformatics 2025.
- Nearest prior AL-for-acquisition work to cite: SOFisher, S²-omics, SCR²-ST.
- Headline numbers: `results_final/{patient,slide}/<dataset>_uni/results.json`.
