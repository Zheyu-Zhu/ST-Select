# Conference Scope and Journal Extension Plan

## Conference paper: retained scope

The conference paper focuses on one controlled question:

> Under a fixed morphology-to-expression predictor and matched spot budgets, do common active-learning strategies outperform uniform Random spot selection?

The conference version includes:

- Two completed public cohorts: HER2-positive breast cancer and cSCC.
- One shared ST-Net-style DenseNet-121 predictor.
- Random plus four active-learning strategies: MC-dropout, TOD, CoreSet, and TypiClust-inspired.
- A dedicated methodological paragraph for each acquisition strategy, including the implementation adaptations that materially affect interpretation.
- Four fold-wide training budgets: 5%, 10%, 30%, and 50%.
- Retrospective six-round label-masking experiments.
- Patient-level cross-validation.
- Compact but explicit reporting of target construction, patch size and resolution range, acquisition timing, optimizer settings, augmentation family, seed, and fixed-horizon limitation.
- Gene-expression PCC, expression-cluster agreement, and Moran's \(I\) fidelity.
- A dedicated definition paragraph for each evaluation measure and a separate descriptive-reporting paragraph.
- A limited computational-cost comparison.
- Explicit single-seed, fixed-horizon, and incomplete-configuration limitations.

The expanded conference Methods and Data sections are intended to be approximately 1.5 times the earlier compact draft. This added space is used for essential algorithm definitions and evaluation reproducibility, not for journal-scale ablations or additional claims.

## Excluded from the conference paper

The following material should not appear as promised, partial, or TK conference content:

- Kidney or other incomplete datasets.
- The undocumented \(\beta=1\) full-label result.
- Cell-type-composition evaluation that has not been completed.
- Claims about contiguous or customized capture regions.
- Claims that the selected spots are spatially dispersed without measurement.
- Extensive HEST geometry and patch-overlap analysis.
- Full library-size projection and highly variable gene analyses.
- Long discussions of individual genes such as ERBB2 and MDK.
- Comprehensive acquisition-timing and memory profiling.
- Exhaustive implementation edge cases and sensitivity analyses beyond those required to reproduce the reported conference configuration.
- Mechanistic claims about section edges, tissue folds, stain artifacts, or atypical samples without selected-spot analysis.
- Broad claims across predictor architectures or ST platforms.

The main conference paper should state the settings needed to understand the experiment. Full fold assignments, gene lists, sample identifiers, environment files, source code, and additional implementation checks should be placed in the conference repository or supplement rather than omitted.

## Boundary between expanded conference detail and journal novelty

Adding one paragraph per strategy and evaluation measure does not reduce the journal's novelty. These paragraphs document the exact conference experiment. The journal contribution remains the expanded scientific scope:

- New datasets and platforms, not longer descriptions of the original two cohorts.
- New acquisition baselines, not only additional implementation detail for the same four.
- Multiple predictor backbones, not only a more detailed DenseNet-121 description.
- Repeated-seed and convergence-controlled experiments, not only stronger caveat language.
- New biological and region-level evaluation targets, not only expanded PCC, ARI, and Moran's \(I\) definitions.
- New conclusions about generalization across datasets, architectures, and physically feasible acquisition constraints.

## Planned journal extension

The journal paper should provide a substantially broader scientific study rather than simply a longer version of the conference manuscript.

### 1. Additional datasets and platforms

- Kidney after complete validation.
- Additional cancer and noncancer tissues.
- Multiple array-based ST platforms.
- Higher-resolution platforms where spot or bin geometry differs.
- External cohorts not included in HEST-1k.
- Cross-cohort or cross-platform transfer evaluation.

### 2. Additional acquisition baselines

- One-shot Random and one-shot feature-based selection.
- Entropy or variance variants adapted to multivariate regression.
- Bayesian active-learning baselines.
- Gradient-embedding approaches such as BADGE-style selection.
- Facility-location or submodular coverage methods.
- Spatially constrained selection.
- Region-based or connected-component selection.
- Hybrid uncertainty-diversity strategies.
- Cost-aware selection with nonuniform spot or region costs.

### 3. Multiple predictor backbones

- ST-Net-style DenseNet-121.
- ResNet and ConvNeXt backbones.
- Vision Transformer or Swin Transformer predictors.
- Pathology foundation-model encoders.
- Graph- or spatial-context predictors.
- Retrieval- or contrastive-learning predictors.
- Frozen versus fine-tuned encoders.

The journal should test whether acquisition-strategy rankings interact with predictor architecture.

### 4. Stronger experimental reliability

- Multiple model-training seeds.
- Multiple Random acquisition realizations.
- Convergence-matched training.
- Optimizer-step-matched training.
- Nested or repeated patient-level cross-validation where feasible.
- Patient-level confidence intervals.
- Dependence-aware or hierarchical statistical analysis.

### 5. Budget and acquisition ablations

- Full-label \(\beta=1\) reference.
- Additional low and intermediate budgets.
- One-shot versus multi-round selection.
- Number and timing of acquisition rounds.
- Shared versus method-specific warm-up.
- TOD schedule alignment.
- Initial-pool size and composition.

### 6. Target and preprocessing robustness

- Raw counts versus library-size-normalized targets.
- Log normalization and gene-wise standardization.
- Highly variable genes versus alphabetical panels.
- Biologically selected gene panels.
- Different target-panel sizes.
- Stain normalization and color augmentation.
- Resolution- and patch-size sensitivity.

### 7. Expanded biological evaluation

- Cell-type-composition fidelity.
- Pathway or gene-set recovery.
- Differential-expression preservation.
- Spatial-domain agreement using expert or published annotations.
- Marker-gene spatial-pattern recovery.
- Tumor, stroma, immune, and normal-region performance.
- Rare-region and boundary recovery.

### 8. Region-selective acquisition

- Contiguous-region constraints.
- Fixed-shape and variable-shape capture regions.
- Region-level acquisition cost models.
- Conversion of selected spots into feasible capture masks.
- Comparison between retrospective spot selection and physically realizable regions.
- Collaboration with assay developers for prospective feasibility.

### 9. Selection-behavior analysis

- Per-slide and per-patient allocation.
- Spatial dispersion.
- Morphological diversity.
- Selection of tissue edges, folds, background, and staining artifacts.
- Representativeness versus outlier preference.
- Feature-space and physical-space coverage.
- Stability of selected spots across seeds and backbones.

### 10. Computational analysis

- End-to-end training and acquisition time.
- GPU memory and storage requirements.
- Scaling with candidate-pool size, budget, and feature dimension.
- Approximate clustering and nearest-neighbor alternatives.
- Accuracy-cost Pareto analysis.

## Required distinction between publications

The journal manuscript should:

- Cite the conference paper prominently.
- State that it extends the conference benchmark.
- Include a clear list of new datasets, methods, backbones, metrics, analyses, and conclusions.
- Reuse only the minimum text and figures permitted by the target journal and conference copyright policies.
- Avoid presenting unchanged conference results as new journal contributions.
- Ensure that the added experiments materially change or strengthen the scientific conclusions.

## Suggested journal contribution statement

The journal extension will broaden the conference benchmark from two cohorts, four active-selection strategies, and one predictor backbone to a multi-dataset, multi-platform, multi-backbone study of spot- and region-level acquisition. It will add repeated-seed evaluation, convergence-controlled training, target-preprocessing sensitivity, biological fidelity measures, and physically feasible region-selection constraints.
