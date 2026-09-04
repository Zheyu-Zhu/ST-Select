# ActiveST-Bench — how the benchmark works

Reference document for the SPIE methods section. Covers what an acquisition
strategy is required to do, how all 29 are implemented, and exactly what the
benchmark loop does around them.

---

## 1. What "active learning" means here

Spatial transcriptomics profiles a dense grid of spots. Sequencing every spot is
the cost. The question is whether an algorithm can choose *which* spots to
sequence so that a model trained on that subset predicts the unmeasured spots as
well as a model trained on everything.

Formally: given an unlabeled pool of spots `U`, a labeling budget `k`, and
optionally a model trained on what has been labeled so far, produce the subset
`S ⊂ U, |S| = k` that maximises downstream prediction quality. "Labeling" a spot
means paying to sequence it.

The baseline is uniform random selection. Any strategy that cannot beat random is
not worth its implementation complexity, because random costs nothing and has no
failure modes.

---

## 2. The strategy interface

Every strategy subclasses `ActiveLearningStrategy` (`src/al_methods/base.py`) and
implements one method:

```python
def select(self,
           candidate_indices,      # unlabeled pool, local indices
           selected_indices,       # already labeled
           k,                      # how many to pick now
           features=None,          # (N, D) frozen or live backbone features
           positions=None,         # (N, 2) spot coordinates on the slide
           model=None,             # current predictor
           dataloader=None,
           extras=None) -> List[int]
```

Each declares what it needs, and the loop supplies only that:

| flag | meaning |
|---|---|
| `requires_features` | needs a representation of every candidate spot |
| `requires_model` | needs the current predictor (uncertainty, gradients) |
| `requires_positions` | needs spatial coordinates |

`validate_inputs()` raises rather than silently degrading if something required
is missing — a strategy that quietly falls back to random would contaminate the
benchmark with fake nulls.

### Image-mode contract

With an end-to-end model the strategy cannot call `model(x)` on a feature
vector — the backbone expects images. The loop therefore extracts backbone
features once per round and passes `extras={"head_only": True}`; strategies then
apply only the prediction head. Without this the run crashes with a
double-backbone shape error, or worse, silently scores on the wrong tensor.

---

## 3. The 29 strategies

All are implemented; **20 are used in the main sweep**. `batchbald` and `cdal`
are excluded there — both are O(N²) with a greedy inner loop, measured at 20.8 s
and 6.7 s per acquisition at N=2000, which caused a 5 h hang.

| family | strategies | selection signal |
|---|---|---|
| **baseline** (1) | `random` | none — uniform draw |
| **uncertainty** (7) | `entropy`, `margin`, `least_confidence`, `mc_dropout`, `tod`, `batchbald`, `learning_loss` | model's own confidence: predictive entropy, margin between top classes, MC-dropout variance, temporal disagreement between the current and previous model, mutual information, a learned loss-prediction head |
| **diversity / coverage** (6) | `coreset`, `typiclust`, `probcover`, `maxherding`, `cdal`, `badge`* | cover the feature space: k-center greedy, typicality within clusters, ball-cover, kernel herding, contextual diversity |
| **hybrid** (6) | `badge`, `alfa_mix`, `similar`, `ccal`, `vaal`, `waal` | combine uncertainty and diversity: gradient-embedding k-means++ (BADGE), feature interpolation (ALFA-Mix), submodular selection |
| **medical / task-specific** (5) | `cald`, `ceal`, `confidnet`, `coregcn`, `suggestive_annotation` | consistency under augmentation, pseudo-labeling, learned confidence, graph convolution over the pool, representativeness |
| **spatial** (3) | `poisson_disk`, `spatial_stratified`, `tissue_aware` | use spot coordinates: blue-noise sampling for even coverage, grid stratification, tissue-region awareness |
| **RL / adversarial** (4) | `lal_rl`, `vaal`, `ta_vaal`, `waal` | learned or adversarial acquisition policies |

\* `badge` is listed under both hybrid and diversity — it is a gradient-embedding
method that achieves diversity through k-means++ over loss gradients.

Spatial strategies matter here in a way they do not in ordinary image
classification: spots have physical coordinates, and expression is strongly
spatially autocorrelated, so even coverage has a real mechanism.

---

## 4. The benchmark loop

`src/training/loop.py::ActiveLearningLoop.run()`. One run = one strategy × one
model × one dataset × one fold × one seed.

```
1. Split the data (see §5). Test fold is set aside and never touched.
2. Seed the labeled pool with a random draw sized to the first budget target.
   ⚠ This is why a single budget target is refused: the seed pool would fill
     the whole budget, no acquisition would ever run, and every strategy would
     report identical numbers while appearing to confirm the null.
3. For each budget target:
     a. Prepare the representation once (features from the cache, or extracted
        through the current backbone in image mode).
     b. Acquire in chunks of `budget_per_round` until the pool reaches the
        target, calling strategy.select() each time.
     c. Reset the prediction head. Reseed deterministically on (seed, round) —
        without this the head init depends on whether the strategy happened to
        run a feature-extraction pass, which made arms start from measurably
        different models.
     d. Optionally hold out whole patients from the labeled pool for early
        stopping (§6), then train.
     e. Evaluate on the untouched test fold.
4. Report the metric at every budget checkpoint.
```

Training happens **once per budget checkpoint**, not per acquisition chunk.

---

## 5. Splitting

Three modes, answering different questions. They are not comparable to each
other and must never be mixed in one table.

| mode | grouping | question |
|---|---|---|
| `patient` **(default)** | all slides of a patient stay together | can this generalise to a **new patient**? |
| `sample` | by slide | can it generalise to a new **slide**? |
| `within_slide` | inside each slide | can I measure part of a slide and predict the rest? |

**Patient grouping is what prevents leakage.** With 4.5 slides per patient in
HER2, a slide-level split puts 5–7 of 8 patients on both sides — the model has
seen that patient's other sections, same tissue, same batch, same staining.
Measured cost of that leakage: slide-level scores run **1.19–1.83×** higher than
patient-level on identical data and models.

Verified: 0 patient and 0 slide overlap in every fold of every dataset under
`patient` grouping.

---

## 6. Training and evaluation

**Model** — four rungs, all trained by the same `ALTrainer`:

| model | backbone | trainable params |
|---|---|---|
| `feature_predictor` | frozen DenseNet-121 or UNI features + MLP head | ~0.7 M |
| `trainable_feature_predictor` | learnable projection over frozen features | ~1.3 M |
| `st_net` | DenseNet-121 end-to-end | 7 M |
| `uni_regressor` | **UNI ViT-L end-to-end, 0 frozen tensors** | **304.08 M** |

**Early stopping** — a fixed epoch count penalises large budgets harder than
small ones (measured: 2.4× overtraining at 5% budget vs 4.4× at 100% on BC),
which tilts and can invert the whole budget curve. The holdout is **group-aware**
(whole patients pulled from the labeled pool), because the overfitting that
matters is patient-level; a random row-level holdout shares the training
patients' batch effects and never detects it.

**Metric** — per-slide PCC is primary: correlate predicted vs measured expression
for each gene *within* a slide, average over genes, then over slides. Pooling all
test spots instead lets between-slide expression differences inflate the
correlation. Also reported: pooled per-gene, top-N (variance-ranked genes),
cell-type composition, Moran's I spatial fidelity, ARI/NMI clustering.

**Significance** — paired Wilcoxon across (fold, seed) pairs, Bonferroni over
budget comparisons. ⚠ At n=4 the minimum attainable two-sided p is **0.125**, so
single-seed runs cannot produce a significant result whatever the data; n≥8 is
required, and the main comparisons use n=12 (4 folds × 3 seeds).

---

## 7. Controls that changed conclusions

Each of these was added after it turned out to matter:

1. **Single-target guard.** A run with one budget target performs no acquisition
   at all and returns identical numbers for every strategy — indistinguishable
   from a null. The loop now raises.
2. **Deterministic per-checkpoint reseeding.** Strategies needing a
   representation ran a feature pass that perturbed the RNG, so their models
   started +0.0074 above random at a checkpoint where all arms should be
   identical. Fixed; residual spread 0.0018.
3. **Nearest-neighbour control.** For within-slide evaluation, predict each
   unmeasured spot as the mean of its 6 nearest measured neighbours — no image,
   no model. It reaches **0.4795** on HER2, beating the learned model by 0.147.
   Any within-slide result reported without this baseline is uninterpretable.
4. **Shuffle control.** Features vs shuffled labels: real 0.064, shuffled −0.010.
   Confirms feature↔label alignment.

---

## 8. Cost, and why the benchmark is structured as it is

Measured on one RTX 3080 (HER2):

| configuration | wall-clock |
|---|---|
| 20 strategies × 7 budgets × 4 folds, **frozen features** | **201 min** |
| 4 strategies × 2 budgets × 4 folds, **UNI e2e** | **306 min** |
| 20 strategies × 7 budgets × 4 folds, **UNI e2e** (projected) | **≈ 212 h ≈ 8.8 days** |

A 63× gap. This is why strategy **breadth** (20 methods) is measured on frozen
features and model **strength** (304 M params, end-to-end) is verified with a
targeted subset. Neither alone would support the conclusion; together they cover
both axes.

Frozen-feature runs are **CPU-bound** — the O(N²) acquisition dominates and the
GPU sits idle 75% of the time (measured: GPU 10%, CPU 40%). End-to-end runs are
GPU-bound (GPU 87%, 8.5 GB). They must never be run concurrently: doing so once
cost a measured ~6× slowdown, turning 3 h of work into over 18 h.
