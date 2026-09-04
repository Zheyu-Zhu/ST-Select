# SPIE abstract — rewritten to match measured results

Every number below is traceable to a run in `results_pg/`. Nothing is projected.

---

## Title (option B, recommended)

**Rethinking Cost-Efficient Spatial Transcriptomics: Sampling Geometry Over
Informative Selection**

*Option A (more conservative):* Does Active Spot Selection Reduce Spatial
Transcriptomics Cost? A Systematic Benchmark

The original title — *"Toward Cost-Efficient Spatial Transcriptomics through
Active Spot Selection"* — presupposes that active selection is the mechanism,
which is the one thing the benchmark rules out.

---

## Abstract

Spatial transcriptomics (ST) enables spatially resolved characterization of gene
expression while preserving tissue morphology. However, conventional ST platforms
profile a dense, predefined grid of spots across an entire tissue section,
including morphologically repetitive or biologically uninformative regions. This
information redundancy increases experimental cost and may limit the scalability
of ST studies. We present ActiveST-Bench, a systematic benchmark evaluating
whether active learning can identify a small but informative subset of spatial
spots while preserving the predictive performance obtained from complete spot
measurements. We evaluate 20 acquisition strategies spanning uncertainty,
diversity, coverage, hybrid, spatial, and reinforcement-learning families across
three ST cohorts (HER2-positive breast, invasive breast, and kidney; 70,219 spots,
127 slides, 53 patients), using predictors that range from frozen convolutional
features to a fully fine-tuned 304M-parameter pathology foundation model, and
under both cross-patient and within-slide evaluation. Per-slide Pearson
correlation is the primary metric, with paired significance testing over twelve
fold-seed replicates.

Our central finding is negative and robust: no acquisition strategy consistently
outperforms random sampling. The best-performing method rotates across datasets,
model classes, and evaluation regimes, and its advantage falls within measurement
noise. Redundancy nevertheless exists and is exploitable — random sampling matches
full-supervision performance using 5% of spots in one cohort — but the required
fraction is cohort- and task-dependent, ranging from 5% to 75%. Two results
redirect where the savings actually come from. First, at a fixed spot budget,
distributing measurements across more patients significantly outperforms
concentrating them within fewer (+0.032 PCC, p=0.007). Second, for within-slide
sparsification, a nearest-neighbour spatial interpolator that uses no imaging data
outperforms the learned image model by 0.147 PCC once sampling density exceeds
approximately 10%. Cost-efficient ST design therefore depends on sampling geometry
and cohort breadth rather than on informativeness-driven spot selection.

---

## Where each number comes from

| claim | value | source |
|---|---|---|
| 20 strategies (29 implemented) | — | `src/al_methods`, `results_pg/final/al20/` |
| 70,219 spots / 127 slides / 53 patients | — | `feature_cache/pg_*_hvg300.npz` |
| 304 M params, all trainable | 304.08 M | `src/models/uni_regressor.py`, E17 |
| winner rotates / sign flips | poisson_disk +0.0031 HER2 → −0.0196 Kidney; entropy +0.0044 → −0.0087 between holdout geometries | E19, E23 |
| effects within noise | best cross-dataset mean +0.0017; init noise floor 0.0018 | E17, E19 |
| 5% matches 100% (HER2) | saturation 5% | E18, n=12, Bonferroni |
| range 5–75% | HER2 5%, Kidney 50%, BC 75% | E18 |
| breadth +0.032, p=0.007 | HER2 k=4 → k=6 at fixed 1,965 spots | E20 |
| kNN beats model by 0.147 | 0.4795 vs 0.3329 @100% | E23 (within-slide, interleaved) |
| crossover ≈10% | model +0.126 @5%, −0.066 @20% | E23 |

## Claims deliberately NOT made

- **"[BEST METHOD] consistently outperforms random sampling under low-budget
  conditions"** — the original placeholder sentence. Directly contradicted: across
  three evaluation regimes nothing survives correction, and the ranking inverts
  between geometries.
- **Any single "X% is enough" number.** It is 5 / 50 / 75% by cohort for
  expression, and different again per task (cell-type 10 / 30 / 50%).
- **Equivalence between strategies.** n=4 runs cannot demonstrate equivalence
  either; only the n=12 comparisons are quoted.

## Caveats to carry into the Discussion

1. Within-slide numbers are **not comparable** to cross-patient numbers — the
   model has already seen the slide's morphology and batch effects.
2. The within-slide comparison used frozen DenseNet features. A 304M model is
   ~1.3× stronger, which extrapolates to ~0.43 — still below the 0.4795 kNN
   baseline, but untested.
3. Within-slide is HER2 only so far; the cross-patient results cover all three.
