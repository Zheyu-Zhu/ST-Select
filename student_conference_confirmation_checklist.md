# Student Confirmation and Revision Checklist

This checklist contains only items that require confirmation from the student, access to the code or data, or new experimental output. The expanded conference draft now gives each acquisition strategy and evaluation measure its own paragraph, so every implementation statement below must be checked directly against the code before submission. Editorial revisions are handled separately.

## 1. Conference scope and completed experiments

- [ ] **Remove kidney from the conference paper.** Delete it from the Abstract, Introduction, Methods, tables, Results, Discussion, and Conclusion unless the complete kidney experiment is ready before submission.

- [ ] **Confirm the status of HER2 TypiClust-inspired at 50%.** Complete all four folds or state clearly that the configuration was omitted because of computational cost. Do not describe the HER2 grid as complete if it remains missing.

- [ ] **Remove the \(\beta=1\) result from the conference paper.** It is currently reported only in the Conclusion and has no documented protocol. Reserve the full-label comparison for the journal extension unless its complete method and fold-level results are supplied.

- [ ] **Confirm whether cSCC ARI and Moran's \(I\) fidelity are available.** If they are not, restrict every spatial-metric conclusion to HER2-positive breast cancer.

- [ ] **Provide a completed-results matrix.** For each cohort, method, budget, metric, fold, and seed, mark the configuration as complete, incomplete, failed, or intentionally omitted.

- [ ] **Confirm whether the conference paper will remain single-seed.** If so, retain descriptive language and do not claim that small differences are established.

- [ ] **Provide learning curves for 5%, 10%, 30%, and 50%.** Confirm the statement that the two smaller budgets are not converged and document how the last-ten-epoch change was calculated.

## 2. Dataset and preprocessing information

- [ ] Provide the exact HEST-1k release, commit or version, and download date.

- [ ] Provide all included slide identifiers and their source accessions.

- [ ] Provide the mapping among patients, slides, and the four cross-validation folds.

- [ ] Provide the complete 300-gene list for each conference cohort.

- [ ] Confirm whether counts were read from AnnData X, raw.X, or a named layer.

- [ ] Provide the code or numerical check used to verify that the targets equal \(\log(1+\mathrm{raw})\).

- [ ] Confirm the original extracted patch size, final network input size, interpolation method, RGB conversion, normalization statistics, and image-border handling.

- [ ] Verify the reported image-resolution ranges: \(0.662\)--\(0.946\,\mu\mathrm{m}\)/pixel for HER2-positive breast cancer and \(0.545\)--\(0.554\,\mu\mathrm{m}\)/pixel for cSCC.

- [ ] Confirm whether tissue/background filtering, registration quality control, or patch exclusions were applied, with counts before and after filtering.

- [ ] Confirm public-data licenses, de-identification status, and whether institutional ethics review was required.

## 3. Acquisition protocol

- [ ] Confirm that the exact initial selected spot identifiers are identical across strategies within every fold and budget.

- [ ] Confirm whether acquisition strategies use spatial coordinates, slide identifiers, or patient identifiers. If none does, state that coordinates are used only for spatial evaluation.

- [ ] Provide requested and delivered spot counts for every fold and cohort.

- [ ] Provide per-slide and per-patient selected-spot coverage and identify any slide receiving no labels.

- [ ] Confirm the TOD cache behavior and that TOD controls four of six rounds.

- [ ] Decide whether the unequal TOD schedule will remain a stated limitation or be addressed with a matched run.

- [ ] Decide whether to add a one-shot selection baseline. If it is not added, retain this as a limitation.

## 4. Strategy and predictor implementation

- [ ] **Verify the Random paragraph.** Confirm uniform sampling without replacement over the pooled spot-level candidate set and that Random performs no model-scoring pass.

- [ ] **Verify the MC-dropout paragraph.** Confirm which dropout layers are enabled, that batch-normalization layers remain in evaluation mode, and that variance is aggregated across all 300 outputs.

- [ ] **Verify the TOD paragraph.** Confirm the exact model states compared, cache indexing by spot identifier, \(L_2\) aggregation across outputs, and fallback behavior when no earlier cache exists.

- [ ] **Verify the CoreSet paragraph.** Confirm the feature tensor, dimensionality, normalization status, initial centers, distance metric, and greedy update procedure.

- [ ] **Verify the TypiClust-inspired paragraph.** Confirm the candidate-only clustering pool, cluster-count rule, feature source, typicality definition, and Random first round.

- [ ] Provide the exact MC-dropout variance equation, including the variance denominator, gene aggregation, and tie handling.

- [ ] Justify five MC-dropout passes or provide a sensitivity check.

- [ ] Define TypiClust-inspired behavior for singleton and empty clusters, duplicate or zero-distance features, self-neighbors, and ties.

- [ ] Check whether \(n_{\mathrm{init}}=3\) is stable or increase it.

- [ ] Provide the exact Python, PyTorch, torchvision, CUDA, and scikit-learn versions.

- [ ] Provide the exact DenseNet-121 pretrained-weight identifier.

- [ ] Confirm the regression-head architecture and the feature tensor used by CoreSet and TypiClust-inspired.

- [ ] Confirm the exact augmentation probabilities and operation order.

- [ ] Confirm that the implemented polynomial learning-rate scheduler matches the manuscript equation.

- [ ] Confirm that weight decay applies to all trainable parameters, that the last incomplete batch is retained, and that final-epoch predictions are reported.

- [ ] State whether deterministic CUDA operations were enabled and identify nondeterministic operations.

## 5. Evaluation and statistics

- [ ] **Verify the PCC paragraph.** Confirm per-gene within-slide computation, the \(10^{-8}\) denominator stabilizer, direct averaging without Fisher \(z\)-transformation, and treatment of constant genes.

- [ ] **Verify the ARI paragraph.** Confirm independent measured and predicted clustering, \(k=10\), \(n_{\mathrm{init}}=10\), seed 42, no centering or scaling, no dimensionality reduction, and the \(2k\)-spot exclusion rule.

- [ ] **Verify the Moran's \(I\) paragraph.** Confirm use of array coordinates, six-nearest-neighbor construction, graph symmetrization, binary non-row-standardized weights, minimum evaluable-gene count, and subsequent aggregation.

- [ ] Confirm the complete aggregation order for PCC, ARI, and Moran's \(I\): gene or spot, slide, patient, fold, and cohort.

- [ ] Decide whether patients or slides receive equal weight. Provide a patient-weighted sensitivity result if slide weighting is retained.

- [ ] List slides excluded from ARI because of insufficient spots or clusters.

- [ ] Confirm whether a gene is excluded from Moran's \(I\) when the measured map, predicted map, or either map is constant.

- [ ] Confirm that a common evaluable gene set is used across strategies within a comparison.

- [ ] Provide the exact metric libraries, functions, versions, random states, and spatial-graph construction code.

- [ ] Provide patient- or fold-level paired values for every reported summary.

- [ ] Remove the pooled sign-test and Wilcoxon \(p\)-values from the Results unless a valid dependence-aware analysis replaces them.

- [ ] Do not report a numerical seed-noise range from a single seed.

## 6. Computational cost

- [ ] Verify the conference statement that Random requires no acquisition-scoring pass.

- [ ] Provide timing conditions for each strategy: cohort, fold, budget, round, pool size, quota, and hardware.

- [ ] State how many timing repetitions were performed and report median and dispersion.

- [ ] State whether feature extraction, caching, data transfer, warm-up, and clustering initialization are included.

- [ ] Provide cumulative acquisition time across all rounds.

- [ ] Explain the missing HER2 TypiClust-inspired 50% run and provide its observed or estimated computational bottleneck.

## 7. Results and figures

- [ ] Provide the patient and fold corresponding to qualitative slide SPA143.

- [ ] Confirm that all qualitative panels are held-out final-checkpoint predictions.

- [ ] Explain how SPA143 and COL1A1 were selected; avoid choosing the example after inspecting which method looks best.

- [ ] Provide slide-level PCC and gene-specific correlation or error for the displayed maps.

- [ ] Provide the editable source for the qualitative figure so repeated ground-truth maps can be removed and the blank cell can be labeled.

- [ ] Ensure all final numbers agree across Abstract, Results, tables, figures, Discussion, and Conclusion.

## 8. Factual confirmations

- [ ] Confirm whether the author name is Yaunkai Huo or Yuankai Huo.

- [ ] Verify all funding agencies, award numbers, principal-investigator names, and the WCM CTSC award year.

- [ ] Verify all recent bibliography records against final publisher records.

## Student handoff package

Please return:

1. The completed checklist.
2. A table of completed runs.
3. Fold- and patient-level metric files.
4. Dataset, fold, and gene identifiers.
5. The exact software environment and preprocessing configuration.
6. Learning curves and acquisition-timing logs.
7. Editable source files for the figures.
