# End-to-End st_net Benchmark — Comprehensive Report

**Run date:** 2026-07-21 → 2026-07-22 · **Machine:** Windows 11, NVIDIA RTX 3080 (10 GB), 16 GB RAM
**Env:** conda `st-select`, Python 3.11, torch 2.6.0+cu124
**Status:** Phases 1–2 done (cp10k); **UPDATED 2026-07-23** with raw-target + UNI findings — **see §0, which revises the §3 headline.**

> **Tk** = value not yet computed (to come). Everything marked Tk is Phase 3 (strategy
> comparison), which was intentionally **not run** — the run was stopped after He-breast
> random per user request.

---

## 0. HEADLINE UPDATE (2026-07-23) — the regression *target* was the real story

The original headline (§3: "end-to-end doesn't help, far from 0.2") was largely an
**artifact of the target normalization**. Two controlled HER2 follow-ups (same
features / genes / spots; only the noted variable changed) revise it.

**(a) Target `log1p(CP10k)` vs `log1p(raw)` — ~2× on per-gene, for *both* backbones.**
`log1p(raw)` keeps each spot's sequencing-depth / tissue-density signal (which images
predict well); CP10k normalizes it away. HER2 per-gene PCC (all 300):

| backbone / training | cp10k | raw | ratio |
|---|---|---|---|
| DenseNet, frozen | 0.0518 | 0.0962 | 1.86× |
| DenseNet, end-to-end | 0.0487 | 0.1111 | 2.28× |
| UNI, frozen | 0.0666 | 0.1186 | 1.78× |

**(b) Backbone UNI > DenseNet (~1.25×), end-to-end adds more, and the levers stack to
clear ~0.2.** HER2 ladder (top-50 = the literature-comparable "best-predicted genes"
number). The last row is the **definitive result** (UNIRegressor, ViT-L fine-tuned
end-to-end, bf16 AMP + gradient checkpointing, 4-fold × 15 epoch, batch 16, backbone LR ×0.1):

| setting | per-gene (300) | per-slide | top-50 |
|---|---|---|---|
| DenseNet · cp10k · frozen | 0.0518 | 0.0303 | 0.079 |
| DenseNet · raw · end-to-end | 0.1111 | 0.0758 | 0.181 |
| UNI · cp10k · frozen | 0.0666 | 0.0354 | 0.093 |
| UNI · raw · frozen | 0.1186 | 0.0805 | 0.185 |
| **UNI · raw · END-TO-END** | **0.1280** | **0.0857** | **0.212** |

**Revised conclusions (confirmed empirically):**
1. **~0.2 is achieved.** End-to-end UNI + raw gives **top-50 = 0.212** — the PI's ~0.2
   expectation is met. The levers stack exactly as predicted: raw (~2×) → UNI (~1.25×) →
   end-to-end (+8%).
2. **End-to-end *does* help — once the target is raw.** end-to-end UNI raw (0.128) >
   frozen UNI raw (0.119); end-to-end DenseNet raw (0.111) > frozen DenseNet raw (0.096).
   The original "end-to-end < frozen" reflected the cp10k regime *and* comparing a weak
   DenseNet against a strong frozen UNI.
3. The "low" absolute numbers in §3–§8 are the **CP10k regime** — still valid for that
   target, but read them with this correction.

Figure: `results_e2e/her2_comparison.png`. Model: `src/models/uni_regressor.py`
(bf16 AMP + differential LR in `src/training/trainer.py`). Results:
`results_e2e/ceiling_uni_raw/`. Next: extend raw-target + UNI to cSCC / He-breast.

---

## 1. Goal

Answer the PI's open question from `PROJECT_LOG.md`: **does training the image backbone
end-to-end (raw H&E patches → gene expression) reach ~0.2 PCC**, versus the existing
**frozen-feature** baseline (frozen UNI embeddings + MLP head)?

- **Model trained:** `st_net` — ImageNet-pretrained **DenseNet-121**, fine-tuned **end-to-end**
  (backbone + head, `frozen_backbone=False`) over 224×224 patches → 300 top-HVG expression.
- **Baseline compared against:** `results_final/` (frozen **UNI** ViT-L features + MLP).
- **Task/eval identical to the frozen benchmark:** patient-level 4-fold CV, top-300 HVGs,
  three PCC conventions (per-gene pooled / per-slide / per-spot).

## 2. Data (HEST-1k, patches + expression; WSIs skipped, ~16 GB)

| dataset | HEST ids | slides | patients | spots | image cache |
|---|---|---|---|---|---|
| cSCC | NCBI759–770 | 12 | 4 | 8,265 | `feature_cache/cscc_img.npz` |
| HER2 | SPA119–154 | 36 | 8 | 13,612 | `feature_cache/her2_breast_img.npz` |
| He-breast | SPA51–118 | 68 | 23 | 30,600 | `feature_cache/he_breast_img.npz` |

Caches built by `scripts/prepare_hest_index.py` (barcode-joined image↔expression, 1000 HVG →
top-300 kept, patient map from `HEST_v1_3_0.csv`).

## 3. Headline finding

**End-to-end fine-tuning of DenseNet-121 does NOT help and does NOT approach ~0.2.**
Across **all 3 datasets × 3 PCC conventions × 2 supervision settings (full / random)**, the
end-to-end model lands **below** the frozen-UNI baseline — every delta is negative.

## 4. Phase 1 — Ceilings (full-supervision, end-to-end vs frozen)

Trained on the entire training fold (40 epochs, batch 32, 4-fold, 1 seed). Mean over 4 folds.

| dataset | convention | end-to-end (DenseNet) | frozen (UNI) | Δ (e2e − frozen) |
|---|---|---|---|---|
| **cSCC** | per-gene | 0.0166 | 0.0226 | −0.0060 |
| | per-slide | 0.0173 | 0.0229 | −0.0056 |
| | per-spot | 0.3829 | 0.4426 | −0.0597 |
| **HER2** | per-gene | 0.0487 | 0.0662 | −0.0174 |
| | per-slide | 0.0268 | 0.0360 | −0.0092 |
| | per-spot | 0.5440 | 0.6031 | −0.0591 |
| **He-breast** | per-gene | 0.0417 | 0.0522 | −0.0105 |
| | per-slide | 0.0241 | 0.0307 | −0.0066 |
| | per-spot | 0.5787 | 0.6250 | −0.0463 |

## 5. Phase 2 — Random budget curves (end-to-end, does more data help?)

Random acquisition, evaluated at 20% / 50% / 100% of the training pool (25 epochs/round,
4-fold). Per-gene PCC, mean over folds.

| dataset | 20% | 50% | 100% | frozen random @100% |
|---|---|---|---|---|
| **cSCC** | 0.0122 | 0.0134 | 0.0154 | 0.0245 |
| **HER2** | 0.0362 | 0.0406 | 0.0469 | 0.0674 |
| **He-breast** | 0.0300 | 0.0330 | 0.0391 | 0.0529 |

Full-convention random @100% (end-to-end):

| dataset | per-gene | per-slide | per-spot | frozen random (per-gene / per-slide / per-spot) |
|---|---|---|---|---|
| cSCC | 0.0154 | 0.0157 | 0.3866 | 0.0245 / 0.0249 / 0.4458 |
| HER2 | 0.0469 | 0.0258 | 0.5493 | 0.0674 / 0.0367 / 0.6049 |
| He-breast | 0.0391 | 0.0224 | 0.5787 | 0.0529 / 0.0309 / 0.6285 |

**Reading:** every curve rises monotonically with more data (the loop works, more labels help
a little), but even at 100% end-to-end stays **below** frozen — data quantity is not the
bottleneck.

## 6. Phase 3 — AL strategy comparison (does any strategy beat random, end-to-end?)

**Tk — not run** (stopped after Phase 2). Planned: random + entropy + coreset + badge +
poisson_disk + spatial_stratified on cSCC, budgets 0.1/0.3/1.0.

| dataset | method | per-gene | per-slide | per-spot |
|---|---|---|---|---|
| cSCC | random | Tk | Tk | Tk |
| cSCC | entropy | Tk | Tk | Tk |
| cSCC | coreset | Tk | Tk | Tk |
| cSCC | badge | Tk | Tk | Tk |
| cSCC | poisson_disk | Tk | Tk | Tk |
| cSCC | spatial_stratified | Tk | Tk | Tk |

(HER2 / He-breast strategy sweeps also Tk — not scheduled this run.)

*Prior evidence:* the frozen-feature benchmark (`PROJECT_LOG.md`) already found **no AL
strategy beats random**; Phase 3 would test whether that holds end-to-end.

## 7. Interpretation & the critical caveat

- **Consistent, robust negative result:** end-to-end DenseNet < frozen UNI on every dataset,
  convention, and supervision setting. Nowhere near 0.2.
- **CRITICAL CAVEAT (must go to the PI):** this compares **end-to-end DenseNet-121** against
  **frozen UNI**, so it conflates *training regime* with *backbone strength*. UNI is a ViT-L
  pathology **foundation model**; DenseNet-121 is ImageNet-pretrained and far weaker. The honest
  conclusion is: *"an ImageNet CNN, even fine-tuned end-to-end, cannot match frozen features
  from a pathology FM, and does not reach ~0.2."* It does **not** prove "end-to-end training
  can't reach 0.2" in general.
- **The clean next experiment:** fine-tune **UNI itself** end-to-end (ViT-L over raw patches).
  That isolates the training-regime variable. Much heavier (needs the gated UNI weights + more
  VRAM/time), so likely gradient checkpointing / smaller batch / LoRA.
- **PCC convention matters as much as the model:** per-spot (0.38–0.58) ≫ per-slide (0.02–0.03).
  The "0.4–0.7" seen in the literature is largely the per-spot convention; the strict
  per-slide/per-gene numbers here are much lower by construction, not by bug.

## 8. Timing (RTX 3080, batch 32)

| job | duration |
|---|---|
| P1 ceiling cSCC | 62 min |
| P1 ceiling HER2 | 102 min |
| P1 ceiling He-breast | 232 min |
| P2 random cSCC | 66 min |
| P2 random HER2 | 110 min |
| P2 random He-breast | 251 min* |

\* He-breast random ran ~20 min long due to a transient system memory-pressure window (external
apps consumed GPU+RAM mid-run; it survived on the page file and completed cleanly).

## 9. Artifacts on disk

- **Trained models (12):** `results_e2e/models/<dataset>/full_supervision_fold{0..3}_seed42.pt`
  — each is `{state_dict, model_name, n_genes, dataset, fold, seed, image_mode, pretrained}`,
  reloadable standalone.
- **Full metrics:** `results_e2e/ceiling/<dataset>/results.json`,
  `results_e2e/random_curve/<dataset>/results.json` (per-fold, all conventions, by_ratio,
  top-N sweep).
- **Run log:** `results_e2e/RUN_LOG.md` · **machine summary:** `results_e2e/overnight_summary.json`.

## 10. Code changes made for this run

- `src/datasets/patch_dataset.py` — image-input dataset (reads HEST `patches/*.h5`), **LRU-bounded
  slide cache** (`PATCH_CACHE_GB`, default 5 GB) fixing an unbounded-RAM stall on He-breast.
- `src/run_experiment.py` — `--image-mode`, `--save-models-dir` (checkpoints full-sup fold models),
  `--patches-dir`, resume-aware; `add_full_supervision` config flag.
- `scripts/prepare_hest_index.py` — builds the image index cache (barcode join + HVG).
- `scripts/run_overnight.py` — prioritized, resumable, failure-isolated driver + logging.

## 11. How to resume Phase 3 (when you say go)

```bash
# P3: strategy comparison on cSCC, end-to-end (add HER2/He-breast by editing the queue)
PATCH_CACHE_GB=5 conda run -n st-select python -m src.run_experiment \
  --dataset cscc_img --image-mode --model st_net --add-full-supervision? \
  --methods random entropy coreset badge poisson_disk spatial_stratified \
  --n-folds 4 --n-seeds 1 --epochs 20 --budget-ratios 0.1 0.3 1.0 \
  --budget-per-round 512 --initial-budget 256 --batch-size 32 \
  --device cuda --output-dir ./results_e2e/al_sweep
```
(or just re-run `scripts/run_overnight.py` — it skips the completed ceiling + random jobs and
proceeds straight to P3.)
