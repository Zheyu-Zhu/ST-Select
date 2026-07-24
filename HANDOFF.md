# ST-Select — Handoff for a New Session

> **Read this first.** It is self-contained: purpose, what was run, every phase, every
> number, every model, known traps, and how to continue. You do **not** need to have seen
> this repo before. Companion docs: `PROJECT_LOG.md` (the earlier frozen-feature era) and
> `results_e2e/END_TO_END_REPORT.md` (the end-to-end results report).

Last updated: 2026-07-23 · Repo: `github.com/Zheyu-Zhu/ST-Select` · HEAD: `8ea107b`

---

## 1. TL;DR — the one thing to know

The project predicts **per-spot gene expression from H&E histology patches** (HEST-1k).
The PI's open question was: **can training the image backbone end-to-end reach ~0.2 PCC?**

- The original benchmark said **no** (numbers ~0.02–0.07) and that end-to-end *hurt*.
- **That was largely an artifact of the regression target normalization.** Switching the
  target from `log1p(CP10k)` to `log1p(raw counts)` roughly **doubles** per-gene PCC.
- With **raw target + UNI backbone + end-to-end fine-tuning**, HER2 reaches
  **top-50 per-gene PCC = 0.212 — the ~0.2 target is met**, and end-to-end genuinely helps.

```
HER2 ladder (top-50 per-gene PCC)
  DenseNet · cp10k · frozen      0.079
  UNI      · cp10k · frozen      0.093
  DenseNet · raw   · end-to-end  0.181
  UNI      · raw   · frozen      0.185
  UNI      · raw   · END-TO-END  0.212   <-- clears ~0.2
```
Figure: `results_e2e/her2_ladder.png`

---

## 2. What the project is

Two layers, do not confuse them:

**(a) The original goal — an active-learning (AL) *selection* benchmark.** Which
spot-acquisition strategy (Uncertainty / Coverage / Hybrid / Spatial / RL) beats **Random**,
holding the predictor fixed (frozen UNI features + MLP head)? *Answer from `PROJECT_LOG.md`:
**no strategy beats random**; the bottleneck is cross-patient generalization, not spot choice.*
That work lives in `results_final/` and is unchanged.

**(b) This session's work — the end-to-end / target-normalization investigation.**
Answering the PI's "can end-to-end reach ~0.2?" This added an image-input training path, a
UNI-backbone model, and the normalization finding. Lives in `results_e2e/`.

---

## 3. Environment & prerequisites

| item | value |
|---|---|
| conda env | `st-select` (Python 3.11) — `C:\Users\zheyu\anaconda3\envs\st-select` |
| torch | 2.6.0+cu124 (CUDA works) |
| key deps | scanpy, anndata, h5py, timm 1.0.28, huggingface_hub 1.24, matplotlib |
| GPU | NVIDIA RTX 3080, **10 GB** (the binding constraint) |
| RAM | 16 GB (also a constraint — see the LRU cache note) |

Run things as: `conda run -n st-select python ...`
(`conda.exe` at `C:\Users\zheyu\anaconda3\Scripts\conda.exe`)

**Both HuggingFace resources are GATED** — already authorized for user `Regsteaf`
(token stored via `hf auth login`):
- `MahmoodLab/hest` (the dataset) — needs accepted terms.
- `MahmoodLab/UNI` (the pathology foundation model) — needs accepted terms.

> Note: the CLI is now `hf` (e.g. `hf auth login`); `huggingface-cli` is deprecated.

---

## 4. Data

Downloaded to `hest_data/` (**gitignored, ~17 GB**): only `patches/*.h5` (224×224 H&E
patches) + `st/*.h5ad` (expression). WSIs deliberately skipped.

| dataset | HEST ids | slides | patients | spots |
|---|---|---|---|---|
| cSCC | NCBI759–770 | 12 | 4 | 8,265 |
| HER2 | SPA119–154 | 36 | 8 | 13,612 |
| He-breast | SPA51–118 | 68 | 23 | 30,600 |

Slide→patient map comes from `hest_data/HEST_v1_3_0.csv` (`id`,`patient` columns).

### Caches (`feature_cache/`, gitignored)
| file | what |
|---|---|
| `{cscc,her2_breast,he_breast}_img.npz` | **image index** caches (cp10k target) — no features; store `patch_pos` so images are read lazily for end-to-end training |
| `her2_breast_raw_img.npz` | image index, **raw target** |
| `her2_{dn,uni}_{cp10k,raw}.npz` | **frozen feature** caches (DenseNet / UNI × cp10k / raw) for the controlled normalization test |

---

## 5. Models

| name | what it is | status |
|---|---|---|
| `feature_predictor` | MLP head over **cached frozen features** (backbone = Identity) | default fast path |
| `trainable_feature_predictor` | learnable projection over frozen features (proxy for a moving representation) | works |
| `st_net` | **DenseNet-121** (ImageNet) + head, fine-tuned **end-to-end** over raw patches | works (image mode) |
| `uni_regressor` | **UNI ViT-L/16** (pathology FM) + head, fine-tuned **end-to-end**; gradient checkpointing | **new this session** |
| `histogene`, `hist2st` | implemented, regression-style | wired but unused |
| `thitogene`, `bleep`, `mclstexp`, `egn` | need special inputs / contrastive loops | **not wired** |

**Feature dims:** DenseNet-121 → 1024; UNI ViT-L → 1024 (both match `feature_dim=1024`).

---

## 6. CRITICAL: how to read the numbers

Two axes decide the magnitude far more than the model does. **Always state both.**

### (a) PCC convention
| convention | how | typical HER2 value |
|---|---|---|
| **per-gene** (pooled) | per gene, correlate across all test spots; average over genes | 0.05–0.13 |
| **per-slide** | per-gene PCC computed **within each slide**, then averaged over slides (strictest; the PI's preference — removes per-slide batch effects) | 0.03–0.09 |
| **per-spot** | per spot, correlate across genes; average over spots (**inflated**, common in papers) | 0.38–0.61 |
| **top-N** | per-gene PCC over the N best-ranked HVGs (literature-comparable "best genes") | top-50 up to 0.212 |

Same predictions can read 0.03 or 0.60 depending on convention. Literature's "0.4–0.7" is
mostly **per-spot** — our per-spot (0.60) is in that range.

### (b) Target normalization  ← the big discovery
| target | definition | effect |
|---|---|---|
| `log1p(CP10k)` (`--target norm`, default) | `normalize_total(1e4)` then `log1p` | the original pipeline; **halves** per-gene PCC |
| `log1p(raw)` (`--target raw`) | `log1p(raw counts)`, no library normalization | **~2× higher** per-gene PCC |

**Why:** `log1p(raw)` retains each spot's sequencing-depth / tissue-density signal, which
images predict well. CP10k normalizes it away, leaving only relative expression (harder).
HVG *selection* is always done on normalized data either way — only the target differs.

Also note **split**: everything here is strict **patient-level** 4-fold CV. Slide-level
splits (common in papers) leak and read higher.

---

## 7. Phases and complete results

`results_final/` = n=12 runs (4 folds × 3 seeds). `results_e2e/` = n=4 (4 folds × 1 seed).

### Phase 0 — Frozen UNI baseline (pre-existing, cp10k target) → `results_final/patient/`
| dataset | method | per-gene | per-slide | per-spot |
|---|---|---|---|---|
| cSCC | full_supervision | 0.0226 | 0.0229 | 0.4426 |
| cSCC | random | 0.0245 | 0.0249 | 0.4458 |
| HER2 | full_supervision | 0.0662 | 0.0360 | 0.6031 |
| HER2 | random | 0.0674 | 0.0367 | 0.6049 |
| He-breast | full_supervision | 0.0522 | 0.0307 | 0.6250 |
| He-breast | random | 0.0529 | 0.0309 | 0.6285 |

### Phase 1 — End-to-end ceilings, DenseNet, **cp10k** → `results_e2e/ceiling/`
40 epochs, batch 32, 4-fold. Models saved.
| dataset | per-gene | per-slide | per-spot | vs frozen UNI |
|---|---|---|---|---|
| cSCC | 0.0166 | 0.0173 | 0.3829 | below |
| HER2 | 0.0487 | 0.0268 | 0.5440 | below |
| He-breast | 0.0417 | 0.0241 | 0.5787 | below |

→ Under cp10k, end-to-end DenseNet looked **worse** than frozen UNI everywhere.
*(This is the result that was later explained away — see Phase 3.)*

### Phase 2 — Random budget curves, end-to-end DenseNet, cp10k → `results_e2e/random_curve/`
per-gene PCC at 20% / 50% / 100% of the labeled pool:
| dataset | 20% | 50% | 100% |
|---|---|---|---|
| cSCC | 0.0122 | 0.0134 | 0.0154 |
| HER2 | 0.0362 | 0.0406 | 0.0469 |
| He-breast | 0.0300 | 0.0330 | 0.0391 |

→ Monotonic (more labels help a little), but the ceiling is low. **Label quantity is not
the bottleneck.**

### Phase 3 — Target normalization + backbone (HER2, **frozen**) → `results_e2e/norm_test/`
Controlled: same spots, same genes, only the noted variable changes.
| setting | per-gene | per-slide | per-spot | top-50 |
|---|---|---|---|---|
| DenseNet · cp10k | 0.0518 | 0.0303 | 0.5907 | 0.0785 |
| DenseNet · raw | 0.0962 | 0.0678 | 0.5965 | 0.1508 |
| UNI · cp10k | 0.0666 | 0.0354 | 0.5992 | 0.0927 |
| UNI · raw | 0.1186 | 0.0805 | 0.6098 | 0.1845 |

→ **raw ≈ 1.8–1.9× cp10k** for both backbones; **UNI ≈ 1.25× DenseNet**.
(Sanity check: fresh UNI cp10k 0.0666 reproduces `results_final`'s 0.0662. ✓)

### Phase 4 — End-to-end with **raw** target (HER2) → `results_e2e/ceiling_raw/`, `ceiling_uni_raw/`
| setting | per-gene | per-slide | per-spot | top-10 | **top-50** |
|---|---|---|---|---|---|
| DenseNet · raw · end-to-end (40 ep, batch 32) | 0.1111 | 0.0758 | 0.5803 | 0.1613 | 0.1806 |
| **UNI · raw · END-TO-END** (15 ep, batch 16, bb-LR×0.1) | **0.1280** | **0.0857** | 0.6043 | 0.1918 | **0.2123** |

→ **top-50 = 0.212 clears the ~0.2 target.** And end-to-end > frozen once the target is raw
(UNI: 0.128 vs 0.119; DenseNet: 0.111 vs 0.096).

### Phase 5 — AL strategy comparison end-to-end (P3) — **FAILED, not completed**
Intended: random + entropy + coreset + badge + poisson_disk + spatial_stratified on cSCC.
Crashed after 45 min. See §9 (known issues) — this is the main unfinished item.

---

## 8. Code written/changed this session

| file | change |
|---|---|
| `src/datasets/patch_dataset.py` | **new** `PatchImageDataset` — reads HEST `patches/*.h5` lazily; **LRU-bounded slide cache** (`PATCH_CACHE_GB`, default 5 GB) — without this, He-breast (68 slides) exhausts 16 GB RAM and training crawls |
| `src/models/uni_regressor.py` | **new** `UNIRegressor` — UNI ViT-L backbone (trainable) + MLP head, gradient checkpointing |
| `src/training/trainer.py` | **bf16 AMP** on CUDA (~3–4× faster; ViT-L is impractical without it) and **differential LR** (`backbone_lr_mult`, backbone at ×0.1 of head) |
| `src/preprocessing/hvg_selection.py` | keeps a raw-counts layer; `extract_expression_matrices(..., target="norm"|"raw")` |
| `scripts/prepare_hest_index.py` | **new** — builds the image-index cache; `--target {norm,raw}` |
| `scripts/run_overnight.py` | **new** — prioritized, resumable, failure-isolated driver + logging |
| `src/run_experiment.py` | `--image-mode`, `--save-models-dir`, `--patches-dir`, `--backbone-lr-mult`, `uni_regressor` wiring, `add_full_supervision` flag |
| `src/training/loop.py` | image-mode dataloader for model-prediction AL strategies (entropy) — **unvalidated** |

---

## 9. Known issues / traps (read before running anything)

1. **P3 / image-mode AL strategies are broken.** AL strategies were written for the frozen
   fast path (`model(features)` where the model is an MLP). In image mode the model consumes
   *images*, so:
   - `entropy` got no features and no dataloader → I added a candidate dataloader in
     `loop.py`, **not yet validated**.
   - `badge` calls `model.backbone(features)` on already-extracted 1024-d features →
     **double-backbone crash** with a conv/ViT backbone. **Still unfixed.**
   - `coreset`/`poisson_disk`/`spatial_stratified`/`random` should be fine.
   Fix these + smoke-test each strategy before rerunning P3.
2. **Don't wrap run output in `grep` without `--line-buffered`.** Doing so buffers all
   console output until process exit (I lost live progress on a 2.5 h run this way).
   `results.json` is unaffected — it's written directly to disk.
3. **Avoid `conda run -n env bash -c "..."`** — it crashed conda. Chain `conda run … python`
   calls in the outer shell instead.
4. **Python buffers stdout when piped** — use `python -u` for live logs.
5. **VRAM (10 GB):** DenseNet batch 32 ≈ 7.1 GB (safe); **batch 64 peaked at 9.95 GB — unsafe
   and no faster.** UNI end-to-end batch 16 + grad-checkpointing + bf16 ≈ 8.1 GB (safe).
6. **RAM (16 GB):** keep `PATCH_CACHE_GB=5`. Other apps (games/Steam/browser) competing for
   GPU+RAM slowed one run badly.
7. **Model checkpoints are saved only for `full_supervision`** (and only with
   `--save-models-dir`). UNI checkpoints are ~1.2 GB each — I deliberately did **not** save
   the UNI end-to-end models.
8. **`results_e2e/`, `feature_cache/`, `hest_data/` are gitignored** — the report + 2 figures
   were force-added; data/models/results.json are **not** in the repo.
9. Seeds: `results_e2e` used **1 seed** (n=4) vs `results_final`'s 3 seeds (n=12). Add seeds
   before making strong claims about small differences.

---

## 10. File map

```
HANDOFF.md                      <- this file
PROJECT_LOG.md                  <- the earlier frozen-feature AL benchmark era
README.md, AL_for_ST_Prediction_Tutorial.md
results_final/patient/<ds>_uni/results.json      <- frozen UNI baseline (committed)
results_e2e/                    (gitignored except the report + figures)
  END_TO_END_REPORT.md          <- results report; §0 has the headline revision
  her2_ladder.png               <- the "reaching 0.2" figure
  her2_comparison.png           <- cp10k vs raw figure
  RUN_LOG.md                    <- chronological run log
  ceiling/<ds>_img/             <- Phase 1 (cp10k e2e DenseNet)
  random_curve/<ds>_img/        <- Phase 2
  norm_test/her2_{dn,uni}_{cp10k,raw}/  <- Phase 3
  ceiling_raw/                  <- Phase 4 DenseNet raw e2e
  ceiling_uni_raw/              <- Phase 4 UNI raw e2e  (the 0.212 result)
  models/<ds>/full_supervision_fold{0..3}_seed42.pt
scripts/prepare_hest_cache.py   <- frozen FEATURE cache builder
scripts/prepare_hest_index.py   <- image INDEX cache builder (--target norm|raw)
scripts/run_overnight.py        <- batch driver
src/models/uni_regressor.py     <- UNI end-to-end model
```

---

## 11. Reproduce / continue

```bash
CONDA=/c/Users/zheyu/anaconda3/Scripts/conda.exe

# 1) Build an image-index cache with the RAW target (this is the one that matters)
"$CONDA" run -n st-select python scripts/prepare_hest_index.py \
  --dataset her2_breast_raw --sample-ids SPA119 SPA120 ... SPA154 \
  --patient-map hest_data/HEST_v1_3_0.csv --target raw
#   -> feature_cache/her2_breast_raw_img.npz

# 2) The headline run: end-to-end UNI + raw  (~2.5-3 h on a 3080)
PATCH_CACHE_GB=5 "$CONDA" run -n st-select python -u -m src.run_experiment \
  --dataset her2_breast_raw_img --image-mode --model uni_regressor \
  --methods full_supervision --n-folds 4 --n-seeds 1 --full-epochs 15 \
  --batch-size 16 --lr 1e-4 --backbone-lr-mult 0.1 \
  --pcc-top-n 10 50 100 200 300 --device cuda \
  --output-dir ./results_e2e/ceiling_uni_raw

# 3) Cheap frozen variant (~15 min): build UNI feature cache, then
#    --model feature_predictor  (no --image-mode)
```

---

## 12. Suggested next steps (priority order)

1. **Generalize the finding** — run raw-target + UNI (frozen ≈ minutes, end-to-end ≈ 2–3 h)
   on **cSCC** and **He-breast**. Right now the 0.2 result is **HER2 only**; confirming it on
   the other two is the single most valuable next experiment.
2. **Add seeds** (3 seeds like `results_final`) to the headline configs for error bars.
3. **Decide the reporting convention with the PI** — per-slide (strictest) vs per-gene vs
   top-N, and **which target** (raw vs CP10k) is primary. This choice moves numbers ~2×, so
   it must be stated explicitly in any writeup.
4. **Fix image-mode AL strategies** (badge double-backbone, validate entropy), then rerun the
   P3 strategy comparison — the only way to test "does any AL strategy beat random
   *end-to-end*". Note the frozen benchmark already answered **no**.
5. Optional gains toward >0.25: H&E augmentation (rotate/flip/color), correlation or
   Poisson loss instead of MSE, spatial context from neighboring spots.

---

## 13. The honest caveats to carry into any writeup

- `log1p(raw)` scores higher partly because the model predicts **sequencing depth / tissue
  density**, which is less biologically interesting than relative expression. CP10k is the
  more standard normalization. Neither is "wrong" — but the choice must be disclosed.
- There is a **biological ceiling**: many genes are simply not predictable from morphology.
  top-50 = 0.212 is the *best-predicted* genes; the all-300 mean is 0.128.
- The 0.212 result is **HER2, 1 seed, patient-split**. Treat as a strong signal, not a
  finished claim, until (1) and (2) above are done.
