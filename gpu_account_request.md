# Request for Cornell Tech computing account — GPU access

**Requester:** Zheyu Zhu
**Faculty sponsors:** Mert R. Sabuncu (Cornell Tech / Weill Cornell Medicine),
Ruining Deng (Weill Cornell Medicine)
**Project:** ActiveST-Bench — benchmarking active spot selection for
cost-efficient spatial transcriptomics
**Related funding:** WCM Radiology AIMI Fellowship; WCM CTSC 2027 Pilot Award;
NIH 1R01HL174863-01A1 and 1U54DK144866-01 (Sabuncu)

---

## 1. What the project is

Spatial transcriptomics (ST) measures gene expression at thousands of spots
across a tissue section while preserving their spatial arrangement. It is the
main experimental route to linking tissue morphology to molecular state, and it
is expensive: one acquisition costs far more than the whole-slide image of the
same section, and cost scales with section area rather than with biological
content, because every spot on the capture grid is sequenced whether or not it
is morphologically distinct from its neighbours.

That observation motivates a question the field has assumed rather than tested:
if an algorithm could decide *which* spots are worth measuring, could the same
biological conclusions be reached from a fraction of them? Active learning (AL)
is the standard tool for spending a limited labelling budget well, and a large
literature reports that it beats uniform random sampling. Whether that transfers
to ST is not obvious — the label is a 300-dimensional continuous expression
vector rather than a class, neighbouring candidates are spatially correlated
rather than independent, and a subset that improves pointwise prediction need
not preserve spatial structure.

We built a controlled benchmark to test it. Everything except the selection rule
is held fixed: within a budget, every arm shares the initial labelled pool, the
acquisition schedule, the morphology-to-expression predictor, the optimiser, and
the number of training epochs, so any difference is attributable to which spots
were chosen. A first paper from this work is in preparation for **SPIE Medical
Imaging 2027**.

## 2. What has already been done, and on what hardware

All work to date has run on a single consumer GPU (NVIDIA RTX 3080, 10 GB),
measured from the run outputs:

| | |
| --- | --- |
| Training runs completed | **412** |
| GPU time consumed | **824 GPU-hours (34 GPU-days)** |
| Wall-clock span | 40 days at an 87% duty cycle |

Covering three public cohorts obtained through the HEST-1k release:

| Cohort | Patients | Slides | Spots | Runs |
| --- | --- | --- | --- | --- |
| HER2-positive breast cancer | 8 | 36 | 13,620 | 84 |
| Cutaneous squamous cell carcinoma | 4 | 12 | 8,671 | 84 |
| Kidney | 22 | 23 | 25,944 | 244 |

Each cohort was swept over five acquisition strategies (uniform random, MC-dropout,
temporal output discrepancy, CoreSet, and a TypiClust-inspired rule), four
labelling budgets (5%, 10%, 30%, 50% of the training pool), and four
patient-level cross-validation folds, plus a full-label reference. The kidney
cohort was additionally repeated under three random seeds to establish that its
result is not a single-draw artefact.

**The finding that motivates the next phase.** On the two cancer cohorts, active
selection is *worse* than random sampling at the budgets where saving money
actually matters, and only marginally better once a third of the section has
been measured. On kidney the result inverts: two of the strategies beat random
sampling in 47 of 48 fold-and-seed comparisons each, at every budget, by margins
consistently larger than the seed-to-seed variation. Whether informed spot selection
helps therefore appears to depend on the structure of the cohort — kidney's 22
patients span a thirteen-fold range in spots per patient, and the winning
strategies measurably redistribute the budget toward under-represented patients.

That is a strong, reproducible, and genuinely surprising result, and it is
exactly the kind of claim that a reviewer will expect to see tested beyond one
tissue type, one predictor architecture, and one imaging platform. Doing so is
what the requested resources are for.

## 3. What the requested GPU access will be used for

Six concrete lines of work, each blocked on compute today.

**(a) Predictor backbones, including pathology foundation models.** Every result
so far uses one convolutional backbone (DenseNet-121, 8M parameters). The
central claim — that acquisition strategy rankings depend on cohort structure —
is only interesting if it survives a change of predictor. The comparison we need
runs ResNet, ConvNeXt, a Vision Transformer, and at least one pathology
foundation-model encoder against the current backbone under identical
acquisition. **This is the hard blocker on local hardware**: the 10 GB card already
runs at about 8 GB with a memory-efficient DenseNet at batch size 80, and
foundation-model encoders in this domain carry 30–40× the parameters. They do not fit at a
usable batch size, at any speed.

**(b) Additional cohorts and platforms.** Three cohorts on one array platform
cannot establish when active selection helps. We plan to add further cancer and
non-cancer tissues, at least one higher-resolution platform where spot geometry
differs, and a cohort outside HEST-1k, then test whether the kidney-style
reversal tracks cohort heterogeneity as the mechanism analysis predicts.

**(c) Repeated seeds across the design.** Repeated seeds currently exist only
for kidney. Extending them to the reference configuration on every cohort is
what converts descriptive differences into claims with an error bar, and it is
the single most-requested item from our internal methods review.

**(d) Additional acquisition baselines.** The current four strategies are a
sample of a much larger literature. Gradient-embedding selection, facility-location
and submodular coverage, Bayesian acquisition, and hybrid uncertainty-diversity
rules all need to be run under the same fixed protocol before the benchmark can
claim to characterise the method space.

**(e) Region-selective acquisition.** The practical payoff of this work is a
region-selective assay: a sequencing protocol that measures contiguous areas
rather than a full grid. Our current results score individual spots, which no
instrument can acquire in isolation. Converting selected spots into physically
realisable capture masks and re-running under that constraint is the experiment
that turns a computational observation into assay-design guidance.

**(f) Target and preprocessing robustness.** The expression targets are
unnormalised log counts, which we have shown carry a substantial sequencing-depth
component. Repeating the headline comparisons under library-size normalisation,
gene-wise standardisation, and biologically selected gene panels establishes
whether the conclusions are a property of the acquisition rule or of the target
definition.

## 4. Scale of the request

Per-run cost is measured, not estimated: 0.9–2.6 GPU-hours per training run on
the current backbone, scaling with cohort size. From those measurements:

| Line of work | Runs | Estimated GPU-hours |
| --- | --- | --- |
| (a) Backbones, incl. foundation models | ~120 | 600–1,000 |
| (b) Additional cohorts and platforms | ~320 | 500–700 |
| (c) Repeated seeds on the reference design | ~240 | 300–400 |
| (d) Additional acquisition baselines | ~150 | 200–300 |
| (e) Region-selective acquisition | ~70 | 100–150 |
| (f) Target and preprocessing robustness | ~160 | 200–250 |
| **Total** | **~1,060** | **≈ 2,000–2,800 GPU-hours** |

The design is deliberately fractional — one experimental axis varied at a time
from a fixed reference configuration — rather than a full crossing of cohorts,
backbones, strategies, budgets, and seeds, which would exceed 80,000 GPU-hours
and is not a serious proposal. Runs are independent single-GPU jobs of 1–6 hours
each, which suits a batch scheduler well and needs no multi-GPU or
high-interconnect allocation. A card with 24 GB or more of memory is the one
hard requirement, and only for line (a).

## 5. Why the local hardware is no longer sufficient

Two independent limits, one of them absolute.

**Memory.** Pathology foundation-model encoders do not fit in 10 GB at a usable
batch size. No amount of patience solves this; line (a) is simply not runnable
locally.

**Throughput and reliability.** At the measured rate, the planned work would
take roughly two and a half years of continuous local compute. It would also be
running on an unmanaged desktop: over the 40-day campaign just completed,
operating-system restarts and sleep events interrupted the job queue three
times, costing about 50 hours of recomputation despite checkpoint-and-resume
logic. A managed cluster with job scheduling removes that failure mode entirely.

## 6. Data handling

All data are **public, de-identified, and already published**: the HER2-positive
breast cancer, cutaneous squamous cell carcinoma, and kidney cohorts are
obtained through the HEST-1k release, which redistributes previously published
spatial transcriptomics datasets. No protected health information, no
institutional clinical data, and no restricted-access material is involved.
Total storage is modest — the complete set of trained outputs for 412 runs is
approximately 23 GB, of which about 3 GB is the analysis-ready subset. Analysis
code is maintained in a private repository and will be released with the
publications.

## 7. Expected output

A conference paper at SPIE Medical Imaging 2027 (in preparation) and a journal
extension covering the multi-cohort, multi-backbone, and region-selective
results described above. Both will acknowledge Cornell Tech computing resources.
