# ST-Select — Comprehensive Experiment Report

> Master log of **every** experiment: completed, running, and planned. Results not
> yet available are marked **`Tk`** (to come). Durations are wall-clock on one
> **RTX 3080 (10 GB)**; "est." = projected, not yet measured. For the exact training
> recipe and the pipeline audit see [`RUN_DETAILS.md`](RUN_DETAILS.md).
>
> Last updated: 2026-08-02 — E18–E22 complete; only the Kidney UNI ceiling still
> running (`scripts/run_queue.py`, resumable).

**Status legend:** ✅ done · ⏳ running · ⏸ planned · ❌ failed/superseded
**Conventions:** patient-level 4-fold CV · **per-slide PCC is primary** (per gene
within a slide, then averaged over slides — the PI's convention; removes between-slide
batch effects); pooled per-gene shown for reference · top-N = variance/dispersion-ranked
genes, **not** best-predicted genes · target
`norm`=log1p(CP10k) / `raw`=log1p(raw). "best−rand" = best AL strategy minus random.

---

## 0. One-screen summary

| # | experiment | what it answers | status | duration |
|---|---|---|---|---|
| E1 | Frozen AL selection benchmark (8 strat × 3 ds × 2 split × 3 seed) | does any strategy beat random? | ✅ | ~ (pre-session) |
| E2 | Target-normalization ladder (HER2, frozen) | raw vs CP10k, UNI vs DenseNet | ✅ | 4 × 1.7 min |
| E3 | E2E ceilings · cp10k (DenseNet) | can end-to-end reach ~0.2? | ✅ | 62 / 102 / 232 min |
| E4 | E2E ceilings · raw (DenseNet + UNI) | the 0.2 result | ✅ | see §4 |
| E5 | Random budget curves · cp10k (DenseNet e2e) | does more data help? | ✅ | 66 / 110 / 251 min |
| E6 | Frozen-vs-trained AL (cSCC + HER2) | does a trainable backbone change AL? | ✅ | see §6 |
| E7 | Comprehensive 20-strategy sweep on HEST (HER2 + cSCC) | "you only tried 8" | ❌ abandoned | superseded by **E17** on the new data |
| E8 | 3-seed AL · cSCC + He-breast (new metrics) | seeded core on all 3 ds | ❌ dropped | redundant — `results_final` already has n=12 there |
| E9 | Trained-cSCC Phase-3 seeds 43,44 | error bars on trained AL | ❌ dropped | question already answered by the HER2 replication (E6) |
| E10 | Pipeline audit (leakage + alignment) | is the benchmark sound? | ✅ | 2 min |
| E11 | DLPFC / crc-HD datasets | cross-organ generalization | ⏸ | Tk |
| E12 | Diffusion baseline (STEM-style) | 3rd learning paradigm | ⏸ | Tk |
| E13 | VAAL / TA-VAAL / WAAL / CCAL | adversarial AL family | ⏸ | Tk |
| E14 | A/B methodology fixes + reruns | camera-ready cleanliness | ⏸ | Tk |
| **E15** | **Collaborator's `processed_gene` (HER2/BC/Kidney)** | is the null an artifact of an unlearnable task? | ✅ | ~28 h |
| **E16** | Multi-task evaluation (spatial architecture + cell-type composition) | do the pitch's other tasks change the answer? | ✅ | ~6 h |
| **E17** | 20-strategy sweep + UNI fine-tuned end-to-end | "only 8 of 29?" + the best model | ✅ | ~33 h |
| **E18** | Controlled budget re-run (2×2: epochs × sample size) | what does "X% matches 100%" really cost? | ✅ | 96 min |
| **E19** | UNI e2e AL, HER2 + Kidney | does the null survive on the strongest model? | ✅ | 15 h |
| **E20** | **Breadth vs depth** (fixed spots, varying patients) | is the constraint patients or spots? | ✅ | 12 min |
| **E21** | Budget curves for cell-type + spatial tasks | does the budget depend on the task? | ✅ | 54 min |
| **E22** | UNI e2e budget curve + top-N gap closure | is early saturation a data property or a weak-model artifact? | ✅ | 12 h |

**Headline 1 — the AL null (robust across everything run, now including the best model).**
No strategy gives a consistent gain over random: **20 strategies × 2 datasets** on the
learnable data, 8 strategies × 4 tissues, **2 downstream tasks**, frozen DenseNet /
frozen UNI / trained DenseNet, **and UNI ViT-L fine-tuned end to end (304 M params, 2
cohorts, E19)** — **≈59 h of AL training, zero significant winners**. On the 304 M model
the winner rotates between cohorts and the sign flips (`poisson_disk` +0.0031 HER2 /
−0.0196 Kidney; `typiclust` +0.0090 Kidney / +0.0001 HER2), with effects at or below the
0.0018 initialisation-noise floor. The coverage gap flagged in earlier drafts is closed.

**Headline 2 — what does move the numbers.** Measured on identical HER2 slides:
**preprocessing ×8.6** (§10b-1) > **evaluation convention ×1.6–2.3** on learnable data
(×16.7 when the target is near-unlearnable, since per-spot PCC still reads 0.60 with no
real signal) > **model ×1.35** (stable across all three data/preprocessing combinations)
> **acquisition strategy ×1.0**. Best model overall: **UNI end-to-end, Kidney
per-slide 0.388 / per-gene 0.467** (§10d).

**Headline 3 — what to spend a labelling budget on (E18/E20/E21).** The budget needed to
match full supervision is **not a constant**: 5% (HER2) / 50% (Kidney) / 75% (BC) for
expression, and *different again per task* (cell-type 10/30/50%; spatial fidelity still
rising at 75% where expression is flat). What generalises is **how** to spend it — at a
fixed spot budget, spreading labels across more patients beats concentrating them
(+0.0319 HER2 p=0.007; +0.0217 BC p<0.001, 12/12 folds).

**Bottleneck:** cross-patient generalization and preprocessing choices — not spot selection.

---

## 1. E1 — Frozen AL selection benchmark ✅  (`results_final/`)

feature_predictor (MLP) over frozen **UNI** features · 8 strategies · 3 datasets ·
patient + slide split · **n=12** (4 folds × 3 seeds) · budgets 5/10/20/30/50/100%.

**Key result — HER2 patient-split, per-gene PCC (best−random by budget):**

| budget | random | best strategy | best−rand | significance |
|---|---|---|---|---|
| 5% | 0.0311 | (all identical) | +0.0000 | — |
| 10% | 0.0415 | typiclust 0.0439 | **+0.0024** | p=0.016 (fails MC-correction) |
| 30% | 0.0552 | coreset 0.0572 | +0.0020 | p=0.13 |
| 100% | 0.0674 | coreset 0.0671 | −0.0003 | trivial tie (all data) |

Across datasets (per-gene, 100%, random): cSCC 0.0245 · HER2 0.0674 · He-breast 0.0529.
**Verdict:** no robust winner; the only measurable blip is typiclust at 10% (matches
the TypiClust "typicality > uncertainty at low budget" result), gone by 30%.
Duration: pre-session (not re-timed).

---

## 2. E2 — Target-normalization ladder ✅  (`results_slide/norm_test/`)

feature_predictor over frozen features · HER2 · full-supervision · **4 × 1.7 min**.

| setting | per-gene | per-slide | top50-slide | ARI |
|---|---|---|---|---|
| DenseNet · cp10k | 0.0517 | 0.0300 | 0.0564 | 0.042 |
| DenseNet · raw | 0.0961 | 0.0687 | 0.1256 | 0.053 |
| UNI · cp10k | 0.0691 | 0.0366 | 0.0652 | 0.057 |
| UNI · raw | 0.1183 | 0.0816 | **0.1489** | 0.062 |

**Verdict:** raw ≈ 1.8–2.2× cp10k (holds at slide level); UNI ≈ 1.25× DenseNet. Best
frozen slide-level top-50 = 0.149 (the pooled 0.185 was inflated by batch effects).

---

## 3. E3 — End-to-end ceilings · cp10k ✅  (`results_e2e/ceiling/`)

st_net (DenseNet-121, trainable) · 40 epochs · full-supervision.

| dataset | per-gene | per-slide | per-spot | duration |
|---|---|---|---|---|
| cSCC | 0.0166 | 0.0173 | 0.3829 | 62.4 min |
| HER2 | 0.0487 | 0.0268 | 0.5440 | 102.3 min |
| He-breast | 0.0417 | 0.0241 | 0.5787 | 232.1 min |

**Verdict:** under cp10k, end-to-end DenseNet < frozen UNI everywhere — later explained
as the target-normalization artifact (E4).

---

## 4. E4 — End-to-end ceilings · raw ✅  (the 0.2 result)

| setting | per-gene | top-50 (pooled) | per-slide | duration | source |
|---|---|---|---|---|---|
| DenseNet · raw · HER2 | 0.1111 | 0.1806 | 0.0758 | 103.7 min | `ceiling_raw/` |
| **UNI · raw · HER2** | **0.1280** | **0.2123** | 0.0857 | ~170 min (est.) | `ceiling_uni_raw/` |
| UNI · raw · cSCC | 0.0861 | 0.0796 | 0.0865 | 102.6 min | `ceiling_uni_raw/` |
| UNI · raw · He-breast | 0.0638 | 0.0854 | 0.0400 | 379.0 min | `ceiling_uni_raw/` |

**Verdict:** top-50 = 0.212 on HER2 clears ~0.2 (pooled convention). **Does not
generalize** — cSCC/He-breast top out ~0.08. HER2 was simply the most predictable
dataset; the levers amplify existing signal.

---

## 5. E5 — Random budget curves · cp10k ✅  (`results_e2e/random_curve/`)

st_net e2e · random acquisition at 20/50/100%.

| dataset | 20% | 50% | 100% | duration |
|---|---|---|---|---|
| cSCC | 0.0122 | 0.0134 | 0.0154 | 66.3 min |
| HER2 | 0.0362 | 0.0406 | 0.0469 | 109.8 min |
| He-breast | 0.0300 | 0.0330 | 0.0391 | 251.0 min |

**Verdict:** monotonic but low; label quantity is not the bottleneck. *Caveat A
(RUN_DETAILS §7): image-mode curve accumulates backbone training.*

---

## 6. E6 — Frozen-vs-trained AL ✅  (the PI's "does training change AL?")

best AL strategy − random, per-slide PCC:

| regime | dataset | best−rand | duration | source |
|---|---|---|---|---|
| frozen DenseNet | cSCC | +0.0006 | 17.9 min | `frozen_cscc_al/` |
| frozen DenseNet | HER2 | +0.0003 | 36.4 min | `frozen_her2_dn_al/` |
| frozen UNI (3 seed) | HER2 | +0.0006 | 158.4 min | `her2_al/` |
| trained DenseNet e2e | cSCC | −0.0020 | 215.2 min | `phase3_trained/` |
| trained DenseNet e2e | HER2 | +0.0011 | 358.0 min | `phase3_trained_her2/` |

**Verdict:** training the backbone doesn't change the answer — AL stays ≈ random. The
cSCC −0.0020 did **not** replicate on HER2 (+0.0011) → seed/dataset noise, not "AL
hurts." Spatial-domain clustering (ARI ≈ 0.05, NMI ≈ 0.11–0.15) flat across strategies.

---

## 7. E7 — Comprehensive 20-strategy sweep on HEST ❌ abandoned

> Killed mid-flight 2026-07-26 after 291 min (see §12 scheduling lesson) and never
> resumed: **superseded by E17**, which runs the same 20 strategies on the learnable
> data, where the test is far more convincing. No results were written.

feature_predictor · **20 strategies** (all runnable families minus `batchbald`+`cdal`,
which hang at scale) · 1 seed · budgets 10/30/100%.


*Note: the first attempt (22 strategies, HER2, night2) was killed after 291 min when
`batchbald`(20.8 s@N=2000) + `cdal`(6.7 s) blew up; timing every strategy produced the
lean 20-set.*

---

## 8. E8 — 3-seed AL · cSCC + He-breast ❌ dropped

> Deliberately dropped as redundant: `results_final/patient/{cscc,he_breast}_uni`
> already provides n=12 (3 seeds) on both datasets. Only the new slide-level /
> clustering metrics would have been added, which do not change the AL conclusion.

8 core strategies · **3 seeds** · new slide-level + clustering metrics.

---

## 9. E9 — Trained-cSCC Phase-3 seed error bars ❌ dropped

> Dropped: the question it answered (was cSCC's −0.0020 real?) was settled by the
> HER2 trained-backbone replication in E6 (+0.0011), which showed it was noise.

st_net e2e · cSCC · 6 strategies · seeds 43,44 (combine with the existing seed-42 run
for a 3-seed error bar on "trained AL ≈ random").

---

## 10. E10 — Pipeline audit ✅  (`RUN_DETAILS.md §6`)

| check | result |
|---|---|
| patient/slide fold leakage | **0 overlap** ✓ |
| feature↔label alignment (shuffle control) | real 0.064 vs shuffled −0.010 ≈ 0 ✓ |
| new metrics don't disturb core PCC | 0.0517 vs 0.0518 ✓ |
| image-mode AL strategies run | all 6 Phase-3 strategies ✓ |

Duration: ~2 min (CPU).

---

## 10b. E15 — Collaborator's `processed_gene` package ✅ (2026-07-25/26)

A second, independently preprocessed copy of HER2 + BC (= He-breast), **plus Kidney
(23 slides, 22 patients — a new organ)**. Their HVG-300 is variance-ranked and led by
immunoglobulin genes (IGKC, IGHG3, IGLC2), whose plasma-cell infiltrates are visible
in H&E — so the task is far more learnable than our earlier panel (decomposed in
§10b-1 below). Adapter:
`scripts/prepare_processed_gene.py` (one PNG per spot → `PNGPatchDataset`).

**Ceilings** (patient-level 4-fold, target = log1p(HVG-300 raw counts), 1 seed):

| dataset | frozen DenseNet | e2e DenseNet | frozen UNI | e2e duration |
|---|---|---|---|---|
| HER2 | 0.318 / 0.257 | 0.393 / 0.306 | 0.377 / 0.301 | 129 min |
| Kidney | 0.340 / 0.227 | 0.413 / 0.326 | **0.425 / 0.326** | 305 min |
| BC | 0.181 / 0.126 | 0.192 / 0.139 | 0.201 / 0.131 | 386 min |
| *(old HEST HER2)* | *0.052 / 0.030* | *0.049 / 0.027* | *0.069 / 0.037* | — |

(per-gene / per-slide). **Frozen UNI matches or beats end-to-end DenseNet on all three
datasets at ~1/100th the compute** (Kidney 0.425/0.326 from a **2.7 min** frozen run vs
0.413/0.326 from 305 min of fine-tuning): a strong pathology FM used frozen outperforms
fine-tuning a weaker backbone. BC is simply the hard dataset (~0.19) under every
backbone. *(These were the best numbers at the time; **E17's UNI end-to-end later
superseded them** — Kidney 0.467/0.388.)*
Target check: hvg300 (raw counts) **0.318** > `expression_norm` (CP-normalized) **0.267**
per-gene — independently replicating the earlier raw-vs-CP10k finding (§2).

**AL benchmark on the learnable regime** — the decisive test of whether the null was
just an artifact of an unlearnable task:

| dataset | n | random | best @30% | delta | p |
|---|---|---|---|---|---|
| HER2 (3 seeds) | 12 | 0.2573 | tod | +0.0244 | **0.034** |
| BC | 4 | 0.1345 | tod | +0.0066 | 0.625 |
| Kidney | 4 | 0.2039 | coreset | +0.0111 | 0.625 |

The single nominal hit **fails Bonferroni** (α=0.0083 over 6 budgets), **reverses sign
at 50%** (tod −0.017, p=0.012, rank 8/8), and **does not replicate** (Kidney −0.009).
**→ the null holds where the task is genuinely learnable.** Across all 10 AL settings
ever run (§0), exactly 1 is nominally significant — what chance predicts at α=0.05.

Durations: caches ~5 min · frozen ceilings 11 min · e2e 129+386+305 min · AL 52+230+170 min ·
3-seed 193 min · UNI/target ~10 min.

---

### 10b-1. Why the numbers moved — controlled decomposition ⭐

**This is the strongest methodological result in the project** and it is deliberately
kept out of the collaborator-facing report (which is scoped to their data alone), so
this is its only full record.

The HER2 slides are **identical** between the two pipelines — `SPA119–154`, 36 of 36,
verified by slide id — as are BC/He-breast (`SPA51–118`, 68 of 68). Only cSCC (ours)
and Kidney (theirs) are unique. So the comparison below is genuinely controlled: same
images, same patients, same 4-fold split, same frozen DenseNet. One variable changes
per step.

| configuration | per-slide | per-gene | attributable to |
|---|---|---|---|
| our panel + log1p(CP10k) | 0.0300 | 0.0517 | baseline |
| our panel + log1p(**raw**) | 0.0687 | 0.0961 | **target: ×2.29** (per-gene ×1.86) |
| **their** panel + log1p(raw) | **0.2568** | **0.3183** | **gene panel: ×3.74** (per-gene ×3.31) |

2.29 × 3.74 = **8.56**, exactly the end-to-end ratio — the two effects are independent
and multiplicative. ~⅓ target, ~⅔ gene panel. (Pooled per-gene gives ×6.15, i.e. the
strict per-slide metric makes the effect *larger*, because pooling masks part of it
with between-slide variance.)

**Mechanism** — both panels measured as log1p(raw) on the same spots:

| | our panel | their panel |
|---|---|---|
| fraction of all values that are zero | **87.7%** | 29.8% |
| median gene's mean expression | 0.044 | 1.020 (23×) |
| median per-gene std (signal to explain) | 0.183 | 0.853 (4.7×) |
| overlap with the other panel | **11 / 300 genes (4%)** | — |

**Root cause.** We followed the tutorial's `flavor="seurat"` and ranked by
`dispersions_norm` — dispersion normalised *within mean-expression bins*. For sparse
counts, variance/mean² diverges as the mean → 0, so mean-correction systematically
selects the least-detected, noisiest genes. It is the standard scRNA-seq choice and
actively harmful for spot-level ST prediction: our top-300 excluded the
immunoglobulins **and ERBB2, the defining gene of the HER2 cohort**, while their
variance-ranked panel leads with IGKC / IGHG3 / IGLC2 / ERBB2.

**Takeaway: HVG selection decides whether this task is learnable at all** — a
model-independent methodological result. Our earlier "the task is barely learnable"
conclusion was largely an artifact of predicting mostly-zero genes.

---

## 10c. E16 — Multi-task evaluation: pitch Tasks #4 and #5 ✅ (2026-07-27)

Until now the benchmark scored one task with one metric family. The collaborator's
package ships `cell_type_counts` (67/88 deconvolved types) and spot `position`, so
the pitch's remaining evaluation tasks became implementable.

**New metrics** (`src/evaluation/metrics.py`, verified against known-answer cases —
Moran's I gives +1.02 on a spatial gradient, −0.03 on noise, −0.81 on a checkerboard):
- `spatial_autocorrelation_preservation` — Moran's I per gene per slide on true vs
  predicted maps; reports `moran_true`, `moran_pred`, `moran_fidelity` (correlation
  across genes) and `moran_abs_err`.
- `celltype_composition_similarity` — cosine, Jensen-Shannon divergence, per-type PCC,
  with a `valid_mask` for the ~2% of spots containing no cells.

**Task #5 — spatial architecture** (frozen UNI): Moran's I of predictions **exceeds**
the ground truth everywhere (HER2 0.423 vs 0.328 · Kidney 0.400 vs 0.267 · BC 0.314 vs
0.186) — predictions inherit morphology's smoothness and denoise the sparse counts —
but **fidelity is only 0.08–0.15**: the model does not know *which* genes are spatially
organised. PCC cannot reveal this; the same models score 0.3–0.43 per-gene. This
quantifies the "generic tissue-region scorer" failure the review warns about.

**Task #4 — cell-type composition** (per-slide PCC, DenseNet / UNI · per-type PCC):

| dataset | per-slide | per-type PCC | cosine | JSD |
|---|---|---|---|---|
| Kidney (88 types) | 0.245 / 0.270 | **0.281** | 0.90 | 0.028 |
| BC (67 types) | 0.185 / 0.243 | 0.090 | 0.90 | 0.029 |
| HER2 (67 types) | 0.098 / 0.075 | 0.108 | 0.93 | 0.023 |

⚠ **Do not report cosine**: independent random compositions already score 0.756 on a
positive simplex. Per-type PCC is the honest metric. Note dataset difficulty
**reorders between tasks** (BC worst on genes, mid on cell type).

**AL on the cell-type task** — best strategy − random, per-slide, 3 budgets × 3 datasets:
all deltas ≤ +0.036, **all p ≥ 0.125, no significant winner**. The null therefore
survives a change of *downstream task*, not just of preprocessing/tissue/backbone.

Durations: metrics+caches ~10 min · ceilings 30 min · AL 45+116+154 min.

---

## 10d. E17 — Strategy breadth + UNI fine-tuning ✅ (2026-07-28/29)

Two remaining gaps closed. Run **strictly serially** — see the scheduling note in §12.

**(a) 20-strategy sweep** — closes *"you only tried 8 of 29"*, on the learnable data
rather than the near-unlearnable original. 20 strategies spanning all six families;
`batchbald` and `cdal` excluded (O(N²)-with-greedy-loop, 20.8 s and 6.7 s per
acquisition at N=2000; they caused an earlier 5 h hang).

| dataset | n | best @30% | delta | min p | verdict |
|---|---|---|---|---|---|
| HER2 | 4 | tod | +0.0207 | 0.125 | ns |
| Kidney | 4 | probcover | +0.0152 | 0.125 | ns |

Also tested on HER2 whether *any* acquisition beats pure random in aggregate (mean of
all 19 strategies vs random, paired per fold): **+0.003–0.005, p = 0.63–0.88 — not
supported.** On HER2 12–16 of 19 strategies beat random on the mean, but on Kidney only
4–14 do, and the winner rotates (tod / coreset / probcover / coregcn / alfa_mix) — the
signature of noise. A binomial count over strategies reads p≈0.02 but is **invalid**
(strategies share folds, initial pool and overlapping selections), so it is not used.

**(b) UNI fine-tuned end-to-end** — the empty cell in the backbone × training-regime
matrix. Note `frozen_backbone=False`: **all 304.08 M parameters trainable, 0 frozen
tensors** (303.35 M ViT-L + 0.73 M head), backbone at LR ×0.1 (1e-5 vs 1e-4 head),
gradient checkpointing + bf16 to fit 10 GB (9.6/10.2 GB observed).

| dataset | per-slide | per-gene | top-100 (slide) | Moran fidelity | duration |
|---|---|---|---|---|---|
| **Kidney** | **0.388** | **0.467** | 0.406 | **0.274** | 475 min |
| HER2 | 0.331 | 0.443 | 0.346 | 0.116 | 250 min |
| BC | — | — | — | — | not run |

**UNI e2e is the best model on both datasets** and the two levers stack. Kidney is
remarkably fold-stable (per-gene 0.466/0.464/0.477/0.461) whereas HER2 fold 2 remains
an outlier — though that outlier is **model capacity, not a hard patient**: it scored
0.046–0.054 across three seeds with frozen DenseNet and **0.279** with UNI e2e.
Kidney's spatial fidelity (0.274) roughly doubles the frozen models', so spatial
structure is *partly* improvable — but HER2's stays at 0.116 despite a 29% accuracy
gain, so it does not come free with accuracy.

### 10d-1. Complete AL run ledger (every selection experiment, timestamped)

| completed | experiment | strategies | seeds | duration |
|---|---|---|---|---|
| 07-25 20:31 | AL · HER2 (gene) | 8 | 1 | 52 min |
| **07-26 04:12** | **AL · HER2 · 3 seeds** (decisive) | 8 | **3** | 193 min |
| 07-26 08:03 | AL · BC | 8 | 1 | 230 min |
| 07-26 10:52 | AL · Kidney | 8 | 1 | 170 min |
| 07-27 01:45 | AL · HER2 (cell-type task) | 8 | 1 | 45 min |
| 07-27 03:41 | AL · Kidney (cell-type) | 8 | 1 | 116 min |
| 07-27 06:15 | AL · BC (cell-type) | 8 | 1 | 154 min |
| 07-28 13:29 | **AL20 · HER2** | **20** | 1 | 202 min |
| 07-29 11:29 | **AL20 · Kidney** | **20** | 1 | 1071 min |
| 07-30 00:16 | budget 2×2 · 3 ds (E18) | 1 (random) | **3** | 96 min |
| **07-30 06:10** | **UNI e2e AL · HER2 (E19)** — the strongest model | 4 | 1 | 336 min |

**≈ 44 h of AL training. Zero significant winners, on every model class we have.** The
strongest single candidate (tod, HER2 @30%, +0.0244, p=0.034) fails Bonferroni,
reverses to worst-of-20 at 50% budget, and does not replicate on Kidney (−0.009) or
BC (+0.007). On the 304 M-param UNI e2e model all three tested strategies fall *below*
random (E19).

✅ **Coverage gap CLOSED (E19, 2026-07-30).** AL had been tested with
`feature_predictor` (frozen DenseNet/UNI features, 18 runs) and `st_net` (e2e
DenseNet, 2 runs) but never with `uni_regressor`. It now has been: 4 arms × 4 folds
on HER2 at 5%→30%, 336 min. **The null holds on the fully fine-tuned 304 M-param
model — all three strategies land below random** (best −0.0027 per-slide). A full
20-strategy sweep remains infeasible (~25 h per strategy).

---

## 10e. E18 — Controlled budget re-run: what "X% matches 100%" actually costs ⭐ (2026-07-29)

`results_pg/budget/`, `scripts/run_budget.py`. **This entry corrects a headline number.**

**The problem.** Every budget curve before this ran at a fixed `epochs_per_round=50`,
so a larger labeled pool bought both more information *and* more gradient steps. The
HER2 curve came out inverted (5% = 0.2936 > 100% = 0.2597 per-slide), which read as
"labeling less is better" — a sign the axis was confounded, not a finding.

**The fix** is early stopping, not fixed gradient steps (which would just move the
confound onto small pools). The holdout is **group-aware** — whole patients, via
`training.loop.group_aware_val_split` — because the overfitting to stop on is
patient-level: a random row-level holdout shares the training patients' batch effects,
so its loss keeps falling while test PCC decays. Verified 0 patient leakage on all
three cohorts, holdout 11–16%, and the *same* patients held out at every budget so the
stopping criterion does not drift along the curve. `full_supervision` uses the same
split, so the ceiling is not trained by a different recipe than the curve.

**The confound was real** (though see the attribution below — it was not what drove the
old saturation numbers). Epochs actually chosen at 100% budget vs the 50 forced before
(`epochs_run`, new field):

| dataset | epochs wanted @100% | legacy overtraining |
|---|---|---|
| HER2 | 16.1 | **3.1×** |
| BC | 11.4 | **4.4×** |
| Kidney | 22.8 | **2.2×** |

Early stopping never hit the 200 cap, and epochs fall as the budget grows (HER2
41.9 → 16.1), so it is actively curbing large pools. Note that in *gradient steps* the
100% model still takes ~7.7× more than the 5% model (1284 vs 167), so where the curve
stays flat it is not because the big-budget model was undertrained.

**Corrected saturation points.** Smallest budget from which it *and every larger one*
are statistically indistinguishable from 100% — paired Wilcoxon over 12 fold×seed runs,
Bonferroni over 6 comparisons. (The looser "smallest budget that is not significantly
worse" is misleading: on BC it returns 30%, but 50% then fails, so 30% is not a
crossing.)

| dataset | train patients | saturation | 100% per-slide | shape |
|---|---|---|---|---|
| HER2 | 6 | **5%** | 0.2825 | flat after 5% |
| Kidney | 16 | **50%** | 0.2054 | monotone rising |
| BC | 17 | **75%** | 0.1460 | monotone rising |

**What actually caused the old "5% everywhere" — two defects, and a different one per
cohort.** The full 2×2 (fixed-50-epoch vs early-stopped) × (n=4 vs n=12), all cells
measured:

| dataset | legacy n=4 | legacy n=12 | ctrl n=4 | ctrl n=12 | blocker |
|---|---|---|---|---|---|
| HER2 | 5% | 5% | 5% | **5%** | neither — genuinely early |
| Kidney | 5% | 75% | 5% | **50%** | **statistical power** |
| BC | 5% | 5% | 5% | **75%** | **fixed epochs** |

Neither fix alone is sufficient across the board; **both were required**, and each
cohort was broken by a different one. Do not summarise this as a single cause — an
earlier draft of this entry said "power, not epochs" on the strength of the n=4
column alone, which is exactly the column in which nothing is detectable either way.

*Defect 1 — the n=4 significance floor.* At n = 4 the minimum attainable two-sided
Wilcoxon p is **0.125**, so *no* n=4 paired test can reach p < 0.05, let alone
Bonferroni's 0.0083. "Budget X is not significantly worse than 100%" from 1 seed ×
4 folds was guaranteed before any data existed. n ≥ 8 is the first size that clears
Bonferroni/6. This is the same floor as the `min p 0.125` entries in E17's table —
the attainable minimum, not a measurement. It is what hid Kidney's true 75%/50%.

*Defect 2 — differential overtraining.* A fixed epoch count does not penalise every
budget equally; it penalises the large ones hardest, because they converge sooner:

| cohort | overtraining @5% | @100% | legacy curve |
|---|---|---|---|
| BC | 2.4× | **4.4×** | inverted, 0.1470 → 0.1315 |
| HER2 | 1.2× | **3.1×** | inverted, 0.2936 → 0.2612 |
| Kidney | **0.9×** (under) | 2.2× | already rising, 0.1596 → 0.2275 |

That gradient is what tilts the curve down. On BC it inverts it outright, and an
inverted curve can never be fixed by more seeds — low budgets are not *worse*, they
genuinely score higher, so no amount of power flags them. Hence BC stays at 5% even
at legacy n=12. Kidney is the mirror case: at 5% it wanted 56.1 epochs and got 50, so
it was mildly *under*-trained there, the gradient is weakest, and its legacy curve
already rose — leaving power as its only blocker.

**What the epoch fix contributed.** It un-inverted the curve and lifted its level. At
matched n=12: BC went from inverted (5% = 0.1470 > 100% = 0.1315, Spearman −0.50) to
correctly ordered (0.1311 < 0.1460, Spearman **+0.68**) — which is what moved BC's
threshold to 75%. HER2's 100% rose **0.2612 → 0.2825** (+8.2%) with the MSE
degradation removed (0.8024→0.8375 legacy vs flat 0.8227→0.8225). The legacy arm at
n=12 reproduces the originally reported HER2 curve to 3 decimals (5% = 0.2936,
100% = 0.2612 vs the 0.2936/0.2597 first reported), so the A/B is a faithful
before/after.

⚠ **Cost of the controlled arm:** the holdout removes ~11–16% of the training data, so
its absolute ceiling can sit *below* the legacy arm's — Kidney's 100% is 0.2054 vs
0.2311 legacy. Saturation is computed within-arm (against that arm's own 100%), so the
penalty applies at both ends and largely cancels for the ratio; but do not compare
absolute scores across arms.

**For the pitch.** "X% matches 100%" is **not a constant** — 5% vs 75% between two
cohorts, a 33× spread in absolute labels (510 vs 16 740 spots). The optimistic "50%
matches 100%" holds exactly for Kidney, fails for BC, and is far beaten by HER2. It
must be measured per cohort, at n ≥ 8.

HER2's 10% point scores *above* 100% (+0.0293, p = 0.034) but does **not** survive
Bonferroni — reported as indistinguishable, not better.

**Mechanism: one hypothesis measured and refuted.** Cross-section redundancy
(`scripts/measure_redundancy.py`) — cohorts contribute 4.5/3.0/1.0 sections per
patient, so near-replicate sections could make the pool redundant. Measured in feature
space over 5 subsamples: HER2 **0.91** [0.90–0.93], BC **0.92** [0.91–0.93], Kidney
unstable (only 1 of 22 patients has a 2nd section; 0.05–0.41 — quote slides/patient =
1.0 instead, which is exact). HER2 and BC have the same redundancy and saturate 15×
apart, so **redundancy does not predict saturation.** It looked convincing only
because it correlated with the *fixed-epoch* saturation values, which are artifacts.

Training-patient count does track the ordering (6 → 5%, 16 → 50%, 17 → 75%), but n = 3
cohorts and it is confounded with cohort size (10.2k/16.8k/22.3k train spots). Testing
it needs a breadth-vs-depth run: fix the spot budget, vary how many patients supply it.
**Not claimed as a mechanism.**

**Cost:** 96 min total for the whole 2×2 — controlled 6.6 + 10.9 + 13.6 min, legacy
13.0 + 28.3 + 23.4 min (3 datasets × 3 seeds × 4 folds × 7 budget points each). Early
stopping made the controlled arm ~2× *faster* than the fixed-50-epoch arm.

**Reproduce:**
```
python scripts/run_budget.py --pilot                     # HER2 A/B gate, ~7 min
python scripts/run_budget.py --arm controlled --seeds 3  # the honest curves
python scripts/run_budget.py --arm legacy --seeds 3      # the before, at matched n
python scripts/run_budget.py --compare --datasets pg_her2_hvg300 pg_bc_hvg300 pg_kidney_hvg300
python scripts/measure_redundancy.py                     # the refuted hypothesis
```

---

## 10f. E19 — Does the AL null survive on the *strongest* model? ⭐ (2026-07-30)

`results_pg/uni_al/`, `scripts/run_uni_al.py`. **336 min.** This closes the coverage
gap flagged at the end of §10d-1: every AL result until now used frozen features +
an MLP head (18 runs) or e2e DenseNet (2 runs), never `uni_regressor`. The standing
objection was that *"your strategies had nothing to work with — a frozen backbone
gives uncertainty and diversity signals nothing to move against."* This is the one
experiment that could have overturned the paper's main negative result.

**Setup.** UNI ViT-L fine-tuned end to end, **304.08 M params, all trainable, 0 frozen
tensors**, backbone LR ×0.1, bf16 + gradient checkpointing. **HER2 and Kidney**, 4
patient folds each, budgets **5% → 30%** (2 checkpoints). 306 min + 589 min. Strategies pre-registered by *mean* delta across
both pg cohorts, not by rank on HER2 alone (HER2's own top-3 all reverse sign on
Kidney, so picking them here would be selecting on the noise we then test), spanning
three families: `poisson_disk` (spatial coverage), `tod` (uncertainty/temporal),
`typiclust` (diversity/clustering).

| cohort | arm | per-slide @30% | Δ vs random | folds better | p |
|---|---|---|---|---|---|
| **HER2** | poisson_disk | 0.3414 | **+0.0031** | 2/4 | 1.000 |
| | typiclust | 0.3384 | +0.0001 | 2/4 | 1.000 |
| | **random** | **0.3383** | — | — | — |
| | tod | 0.3382 | −0.0001 | 2/4 | 1.000 |
| **Kidney** | typiclust | 0.3407 | **+0.0090** | 2/4 | 0.625 |
| | **random** | **0.3318** | — | — | — |
| | poisson_disk | 0.3122 | −0.0196 | 0/4 | 0.125 |
| | tod | 0.3048 | −0.0269 | 1/4 | 0.250 |

**The null holds on the strongest model, on both cohorts, and the frozen-representation
objection is answered.** With 304 M trainable parameters — a representation that
demonstrably moves — no strategy separates from random on either cohort. Two features
mark this as noise rather than a weak effect:

* **The winner rotates and the sign flips.** `poisson_disk` is best on HER2 (+0.0031)
  but −0.0196 on Kidney; `typiclust` is best on Kidney (+0.0090) but +0.0001 on HER2;
  `tod` is ~0 on HER2 and **worst** on Kidney (−0.0269). Same signature as the
  frozen-feature sweeps (E17), now reproduced on the 304 M model.
* **The effects sit at the measurement floor.** Residual run-to-run variation in the
  initial model is 0.0018 (HER2) / 0.0002 (Kidney), measured at the 5% checkpoint where
  every arm trains on the *identical* seed pool and should therefore score the same.
  HER2's best effect (+0.0031) is barely above its own noise floor. This is a stronger
  statement than any p-value: the differences are unresolvable in principle at this
  scale, not merely unproven.

Read the limit in both directions: at n=4 the significance floor (§13 F) means nothing
here can reach p < 0.05 either way, so the claim is "no effect large enough to matter,
with no consistent sign", not "proven equivalent".

⚠ **The HER2 numbers above are the post-fix re-run.** The original run had an
initialisation asymmetry: arms whose strategy calls `_extract_features()` perturbed the
RNG, so the head re-init drew different weights and those arms began 0.0074 *above*
random at the 5% checkpoint. `loop.run()` now reseeds deterministically on
`(seed, round)` before the head reset, which collapsed that spread from **0.0074 to
0.0018** and removed its correlation with feature extraction. The conclusion did not
change — it sharpened: pre-fix all three arms sat *below* random (−0.0027 to −0.0074),
post-fix they straddle it at ±0.003. Kidney was run after the fix throughout (spread
0.0002). Superseded pre-fix numbers are kept in `results_pg/uni_al/pg_her2_hvg300/`.

**Why the acquisition demonstrably ran** (this design has a silent-failure mode):
a single budget target would have been filled entirely by the random seed pool, so
every arm would report identical numbers — which looks exactly like "no strategy beats
random" while testing nothing. With `[0.05, 0.30]` the strategy picks **83%** of the
final pool (490 → 2943 spots). `training.loop` now raises on the single-target case,
and a `--dry` gate (1 fold, 1 epoch) confirmed all four arms complete a full
acquire/train/eval cycle with *distinct* results before the 5.6 h run was committed.

⚠ **A real asymmetry found in this run, since fixed.** The 5% checkpoint should be
identical for all four arms (same seed → same random pool, no acquisition yet). It was
not. Arms split exactly by whether the strategy triggers `_extract_features()`:

| group | strategies | per-slide @5% |
|---|---|---|
| no feature pass | random, poisson_disk | 0.3417 |
| extracts features | tod, typiclust | **0.3491** |

The extraction pass perturbs the RNG state, so the head re-initialisation that follows
draws different weights — i.e. comparing strategies also compared initialisations.
It does **not** change this conclusion; it strengthens it, because the affected arms
started *above* random and finished *below* it. Measured against each arm's own 5%
baseline: random **+0.0082**, poisson_disk +0.0034, tod −0.0018, typiclust −0.0065 —
same ordering, larger gaps. `loop.run()` now reseeds deterministically on
`(seed, round)` immediately before the head reset, so every arm's checkpoint-*r*
initialisation is identical. **The results above predate that fix.**

**Incidental (not a controlled budget claim).** Random at **5%** (490 spots) scores
per-slide 0.3417 against the measured 100% UNI e2e ceiling of **0.331** — and its own
30% point only adds +0.0082 for 6× the labels. The 5% checkpoint is a single 15-epoch
fine-tune from pretrained weights, the same recipe as the ceiling, so the comparison is
fair on that axis; but this run uses **fixed 15 epochs with no early stopping**, the
exact protocol E18 showed distorts budget curves. Read it as consistent with HER2
saturating at 5% on the strongest model, not as an independent budget result. Note the
30% numbers are additionally warm-started (the backbone carries over from the 5%
checkpoint; only the head resets) so they are not comparable to the from-scratch
ceiling at all — the strategy comparison is unaffected, since every arm shares it.

**Reproduce:**
```
python scripts/run_uni_al.py --dry   # gate: validates the image-mode AL path, ~15 min
python scripts/run_uni_al.py         # 4 arms x 4 folds, ~336 min
```

---

## 10g. E20 — Breadth vs depth: is the constraint patients, or spots? ⭐ (2026-07-31)

`results_pg/breadth_depth/`, `scripts/run_breadth_depth.py`. **12 min**, frozen features.

E18 left the central question open. It measured *how many* labels each cohort needs
(HER2 5%, Kidney 50%, BC 75%) but could not say why, and explicitly refused to claim a
mechanism from n = 3 cohorts. This run answers it by breaking the confound directly:
**hold the spot budget fixed and vary how many patients supply those spots.** Same
number of labels, identical cost to the lab; only the breadth changes.

| cohort | fixed budget | k=4 patients | all patients | Δ | p | folds better |
|---|---|---|---|---|---|---|
| HER2 | 1 965 spots | 0.2603 | 0.2922 (k=6) | **+0.0319** | **0.007** | 10/12 |
| BC | 1 152 spots | 0.1250 | 0.1467 (k=17) | **+0.0217** | **<0.001** | **12/12** |
| Kidney | 621 spots | 0.1268 | 0.1357 (k=16) | +0.0089 | 0.569 | 8/12 |

**Spreading the same labels across more patients wins.** Significant in 2 of 3 cohorts
on the primary metric — both surviving Bonferroni across the three tests (α = 0.0167) —
and positive in all three. On per-gene the effect is positive everywhere and largest
exactly where per-slide is flat (Kidney **+0.0614**), i.e. breadth buys between-slide
calibration on Kidney and within-slide structure on HER2/BC.

This is the actionable rule the budget result was missing: **label fewer spots per
section, across more patients.** It also explains E18's spread without needing a new
variable — HER2 has only 6 training patients, so it exhausts the available breadth
almost immediately, which is exactly what "saturates at 5%" means.

⚠ **The k=2 arm is excluded from the table and must stay excluded.** With only two
patients in the pool, a group-aware holdout cannot leave two patients to train on, so
the split falls back to random rows — early stopping then shares the training patients'
batch effects and runs far longer (57 / 114 / 66 epochs vs 25 / 43 / 31 at higher k).
Its raw numbers *inflate* the breadth effect (HER2 +0.0330, BC +0.0453 measured from
k=2), so the honest comparison starts at k=4 where every arm uses the same scheme.

---

## 10h. E21 — The budget depends on the task, not just the cohort ⭐ (2026-07-31)

`results_pg/budget/controlled/pg_*_celltype/`. **54 min** for the cell-type curves; the
spatial numbers needed no run at all — E18's controlled arm already recorded Moran
fidelity, ARI and NMI at all 7 checkpoints.

**Cell-type composition saturates at a different budget than gene expression:**

| cohort | cell-type composition | gene expression |
|---|---|---|
| HER2 | 10% | 5% |
| Kidney | **30%** | 50% |
| BC | **50%** | 75% |

**Spatial fidelity keeps improving where expression has stopped** (Moran fidelity,
per budget):

| cohort | 5% | 30% | 75% | 100% |
|---|---|---|---|---|
| HER2 | 0.083 | 0.134 | **0.139** | 0.114 |
| BC | 0.063 | 0.078 | 0.082 | **0.086** |
| Kidney | −0.080 | −0.088 | −0.057 | −0.058 |

On HER2 expression PCC is flat from 5% onward while spatial fidelity rises **+67%**
(0.083 → 0.139). So a budget chosen on expression alone under-buys spatial structure.

⚠ **Kidney's spatial fidelity is negative at every budget.** The model's spatial
autocorrelation is *anti*-correlated with the truth there — it assigns spatial
structure to the wrong genes — and more labels only pull it toward zero, never past it.
Kidney is simultaneously the best cohort for accuracy (per-slide 0.388 UNI e2e) and the
worst for spatial fidelity, which is worth stating plainly rather than reporting the
accuracy alone.

**Consequence for the pitch.** "X% of labels is enough" needs two qualifiers, not one:
enough *for which cohort* **and** *for which task*. Across three cohorts and three tasks
the answer ranges from 5% to 75%, and the task can move it as much as the cohort does.

---

## 10i. E22 — The budget answer on the best model, and the top-N gaps closed ✅ (2026-08-02)

`results_pg/uni_budget/`, `results_pg/uni_ceiling/`.

**(a) Budget curve on UNI e2e** (HER2, 7 points, 551 min). The obvious objection to
E18 is that its saturation points reflect a small MLP head that stops learning early,
not the data. It does not survive:

| budget | 5% | 10% | 20% | 30% | 50% | 75% | 100% |
|---|---|---|---|---|---|---|---|
| UNI e2e | 0.3246 | 0.3262 | 0.3352 | **0.3409** | 0.3349 | 0.3352 | 0.3303 |
| frozen DN (n=12) | 0.3067 | 0.3118 | 0.3018 | 0.2946 | 0.2978 | 0.2850 | 0.2825 |

Flat from 20% on, **peaks at 30%**, and 100% is not the best point — the 304 M model
agrees HER2 saturates early. ⚠ Two caveats, both flattering the low end: **n=4** (one
seed; three would have cost ~27 h) so the p=0.125 floor applies and nothing *can* be
flagged worse; and **fixed 15 epochs** at every budget, matching the ceiling's recipe,
which is precisely the confound E18 identified. Treat as corroboration of the curve
*shape*, not an independent budget measurement.

**(b) top-200 recovered** (186 min HER2; Kidney queued). The original UNI ceilings used
cutoffs [10,50,100] and saved no weights, so the missing cell needed a retrain. The
re-run also served as an **unplanned reproducibility check** — identical config, three
days apart:

| | top-10 | top-50 | top-100 | top-200 | top-300 | per-slide | per-gene |
|---|---|---|---|---|---|---|---|
| re-run | 0.2820 | 0.3365 | 0.3458 | **0.3356** | **0.3303** | 0.3303 | 0.4416 |
| original | 0.2801 | 0.3366 | 0.3462 | n/c | n/c | 0.3313 | 0.4429 |

Every overlapping number agrees to **~0.001** (CUDA nondeterminism). That independently
corroborates the 0.0018 initialisation-noise floor quoted in E19, measured a completely
different way — and confirms E19's ±0.003 strategy effects sit at the edge of what this
setup can resolve. Checkpoints are now persisted, so further metrics are a re-eval.

---

## 11. Planned / deferred (not safe to run unattended) ⏸

| # | experiment | why deferred | est. duration |
|---|---|---|---|
| E11 | DLPFC / crc-HD datasets | gated download + unvalidated HEST format | Tk (download + build + runs, ~1 day) |
| E12 | Diffusion baseline (STEM-style) | new model code, needs review | Tk |
| E13 | VAAL / TA-VAAL / WAAL / CCAL | train a net per acquisition round; need tuning | Tk |
| E14 | A/B fixes + targeted reruns | only if a full-sup ceiling / trained budget-curve figure ships (RUN_DETAILS §7) | Tk (B ~1–2 h · A ~Phase-3 scale) |

---

## 12. Compute / timing summary

**Measured wall-clock (RTX 3080):**

| phase | measured total |
|---|---|
| E3 cp10k ceilings (3 ds) | ~397 min |
| E4 raw ceilings (HER2 dn+uni, cSCC, He) | ~755 min |
| E5 random curves (3 ds) | ~427 min |
| E2 norm_test ladder | ~7 min |
| E6 frozen-vs-trained AL (5 runs) | ~786 min |
| night2 killed comprehensive (wasted) | 291 min |
| E15 processed_gene programme | ~28 h |
| E16 multi-task evaluation | ~6 h |
| E17 20-strategy sweeps + UNI e2e | ~33 h |
| E7–E9 HEST marathon | abandoned mid-flight (superseded) |

**Scheduling lesson (cost us 15 h).** The AL sweep *looks* CPU-bound — GPU sits at
37–42% util and 2.8 GB — but it trains its MLP head **on the GPU in ~480 short bursts**.
Running it beside a job that pins the card at 100% (UNI ViT-L) starves those bursts:
**3 h of work took >18 h, a ~6× slowdown.** Profiling at the real pool size showed
acquisition is only 1.9 h, evaluation 0.5 h, training 0.6 h. These job types must run
**strictly serially**. Two mitigations added afterwards: `results_partial.json` is now
written after every method (an 18 h job is no longer all-or-nothing), and long runs are
launched via the env's python directly rather than `conda run`, which was swallowing
child stdout and hiding all progress.

Rules of thumb on this GPU: frozen-MLP full-supervision ≈ 1.7 min/dataset; frozen
8-method AL sweep ≈ 3 min/method/seed; DenseNet e2e ≈ 60 min per (dataset·seed) at
cSCC scale, scaling ~linearly with spot count (HER2 ≈ 1.65×, He-breast ≈ 3.7×); UNI
ViT-L e2e ≈ 1.7× DenseNet. O(N²) strategies (batchbald, cdal) are not scale-safe.

---

## 13. Caveats carried into any writeup

See `RUN_DETAILS.md §7`. In brief: (A) image-mode budget curves accumulate backbone
training; (B) `full_supervision` overfits vs the AL-100% point in `results_final`
(40 vs 60 epochs); (C) HVG panel selected on full data; (D) multi-seed = training
variance only; (E) reset heuristic is fragile. None affects the negative-result
headline (all symmetric across strategies; audit E10 clean).

**(F) ⚠ The n=4 significance floor (added E18).** A two-sided paired Wilcoxon over
n = 4 runs (1 seed × 4 folds) has a **minimum attainable p of 0.125**. Any such test
is incapable of reporting p < 0.05 whatever the data, so every "not significantly
different" from a single-seed run is uninformative — including the `min p 0.125`
entries in E17. Reachability by n: n=6 → 0.031 (clears 0.05, not Bonferroni/6);
**n=8 → 0.0078** (first n that clears Bonferroni/6); n=12 → 0.0005.

This does *not* overturn the AL negative result, which rests on effect sizes and
sign-reversal rather than p-values: the best cross-dataset mean delta is +0.0017
(~0.7% relative) and the per-dataset winners reverse sign (cald +0.0042 HER2 →
−0.0032 Kidney; maxherding +0.0037 → −0.0038). But an *absence* of difference cannot
be claimed from n=4 either — the honest statement is "no strategy shows an effect
large enough to matter", not "proven equivalent". Budget claims **were** affected: at
n=12 Kidney's saturation moves 5% → 50% and BC's 5% → 75%.
