# Response to reviewers

Manuscript 14525-21, “Benchmarking active spot selection for cost-efficient spatial transcriptomics”

We thank the reviewers for recognizing the value of a controlled assessment of active spot selection and for identifying limitations in resolution handling, replication, and statistical reporting. We revised the full manuscript while retaining the accepted study's HER2 and cSCC scope. No additional training runs were performed. The revision adds validated analyses of the existing outputs and distinguishes those analyses from experiments that remain necessary.

## Reviewer 1

**Comment: The study has relatively few patients, substantial variability, and largely negative results without a definitive conclusion. Random subsampling may perform comparably to more involved active learning.**

**Response:** We agree that the results are primarily negative and have retained them. The Introduction now identifies the contribution as a test of existing acquisition rules, rather than the development of a new algorithm or proof of sequencing-cost savings. The revised conclusion is deliberately specific: under this two-cohort, fixed-predictor, fixed-horizon protocol, the tested rules provide no consistent low-budget PCC improvement over Random. Three of four active-strategy means are lower at 5%, and all four are lower at 10%, on both cohorts. This is an observed limitation of these implementations, not a universal statement about active learning.

We expanded paired-effect reporting to both cohorts: the PCC table gives individual strategy means and SDs, the paired-summary table gives pooled effects and sign counts, and complete fold-level differences accompany the revision. The cohort dependence of the higher-budget results is explicit: all four HER2 differences are positive at 30%, but three cSCC differences remain negative; at 50%, all four are slightly positive in both cohorts. We corrected the previous generalization of a common 30% crossover. The Results also distinguish higher mean ARI on HER2 from its failure to reproduce on cSCC. The all-four-fold consistency applies to ARI, not to lower PCC on every fold.

The new Discussion subsections “What the Negative Result Establishes” and “Uncertainty and Experimental Controls” explain that thousands of spots do not replace independent patients. Fold SD reflects held-out-patient heterogeneity, not seed robustness. We do not claim equivalence or noninferiority to Random. Broader validation across additional organs, including renal tissue, independent databases, and capture technologies is listed as future work. Selected-spot mechanisms and physically feasible contiguous-region acquisition remain untested in this conference study.

The retrospective experiment hides expression labels while retaining all candidate H&E images. We now state explicitly that the budget counts revealed spots and does not measure reduced image acquisition, reagent expenditure, sequencing reads, or prospective assay costs.

**Locations:** Abstract; Introduction; Methods, “Retrospective Active Spot-Selection Benchmark”; Results, “Paired Comparison With Random”; Discussion and Future Work; Conclusion.

## Reviewer 2

### Comment 1: The study does not account for varying resolutions.

**Response:** We added “Image Resolution and Physical Context” and scale/physical-context columns in the cohort table. H&E pixel scale, physical patch width, and ST capture-grid resolution are now distinguished. All 22,291 referenced training PNGs are 224 × 224 pixels, but a constant pixel size does not harmonize physical context. The actual slide scales imply patch widths of 148.3–212.0 μm on HER2 and 122.2–124.1 μm on cSCC. Twelve sampled training patches across four locally available HER2 whole-slide images were verified pixel-for-pixel against native-resolution crops; local cSCC whole-slide images were unavailable, so its scale audit uses the corresponding slide metadata. HEST's separately resampled HDF5 patches were not the model inputs.

Every paired method comparison uses the same candidate images and scales. That controls scale as an input difference between methods within the comparison, but it does not demonstrate that method rankings are invariant to scale. We have not retrained after harmonizing physical field of view or assessed multiple ST capture resolutions. This concern therefore remains partly unresolved and is explicitly identified as a limitation. Future work will compare common-physical-context extractions and controlled resampling, while separating image resolution, capture geometry, patient, and scanner effects. Descriptive resolution-stratified outputs are included in the validation materials and are not presented as causal evidence.

**Locations:** Data and Experimental Design, “Image Resolution and Physical Context,” Table “Cohort size, H&E scale, and physical context of 224-pixel patches”; Discussion, “Resolution, Targets, and Generality.”

### Comment 2: Experiments, including Random, were run only once and could vary across runs.

**Response:** We clarified that there are four patient-level folds but only one acquisition/training seed per method–budget–fold configuration, including Random. Thus there are 160 budgeted configurations, not 160 independent biological replicates or repeated seeds. We explicitly acknowledge that acquisition randomness, training randomness, their interaction, and CUDA nondeterminism are not separated by the design.

Repeated independent seeds were not completed for these conference cohorts. The revision does not claim to resolve this concern through post-hoc analysis. It now identifies repeated training/acquisition seeds, multiple Random realizations, and repeated patient partitions as necessary validation before ranking small effects. We retain the same training architecture and hyperparameters and make no unperformed-experiment claims.

**Locations:** Data and Experimental Design; Evaluation and Reporting, “Statistical reporting”; Discussion, “Uncertainty and Experimental Controls”; Conclusion.

### Comment 3: Additional statistical testing was not performed.

**Response:** We added an explicitly exploratory sensitivity analysis using the corrected cross-validation paired t statistic described by Bouckaert and Frank (2004). For the four fold-level paired differences, the standard error is √[(1/4 + 1/3)s²], with three degrees of freedom; 1/3 is the nominal held-out/training patient ratio. We apply Holm adjustment over the 32 PCC comparisons across cohorts, budgets, and active rules. The minimum unadjusted p-value is 0.128; all 32 Holm-adjusted values are 1.000. For HER2 CoreSet at 5%, the paired difference is −0.0412 and its approximate corrected pointwise 95% interval is [−0.1041, 0.0216]. Full fold-level differences, approximate intervals, p-values, and analysis scripts accompany this revision.

This is a sensitivity approximation, not calibrated confirmatory inference for four heterogeneous grouped folds. It does not account for unmeasured seed variability and cannot establish equality or noninferiority. We therefore keep effect sizes and fold directions as the primary descriptive analysis, avoid pooling shared-comparator observations as independent samples, and do not add spot-level significance tests. The single-seed limitation remains explicit rather than being obscured by additional p-values.

**Locations:** Evaluation and Reporting, “Statistical reporting”; Results, “Paired Comparison With Random”; Discussion, “Uncertainty and Experimental Controls.”

## Additional corrections identified during validation

1. **Target construction:** Direct comparison with raw HEST counts showed that HER2 uses log1p(raw counts), but cSCC uses spot-wise library scaling. The empirically reconstructed cSCC transformation is log1p(10,000 × count / the spot's total over 13,251 genes common to its 12 slides). It matches all 8,671 × 300 targets within 2.4 × 10⁻⁷. We replaced the incorrect statement that both cohorts were unnormalized and added preprocessing confounding as a limitation of cross-cohort interpretation. The original target-producing script remains unrecovered.
2. **Learning-rate protocol:** Source and saved scalar events show a polynomial indexed by cumulative optimizer updates with denominator 10⁸, not by the budget-specific epoch horizon. We corrected the equation and explained its nearly constant effective rate.
3. **Full-label reference:** The nominal reference releases labels over the same six rounds and can omit up to five spots through integer quotas. An added epoch-50 comparison favors it by +0.0058 on HER2 and +0.0073 on cSCC, but it receives approximately twice as many optimizer updates as the 50% arm. Equal elapsed epochs are not equal optimizer exposure.
4. **Learning-curve claims:** We replaced the unsupported “below 2.5%” high-budget tail-change statement with an explicit formula and audited ranges. These are not convergence guarantees.
5. **Numerical reporting:** We regenerated 40 PCC mean/SD cells from unrounded outputs, correcting 19 HER2 SD entries, and corrected the HER2 30% pooled difference to +0.0055. We corrected the Moran-table boldface and duplicate Discussion heading.
6. **Timings and references:** We separate three later-session TypiClust fold medians (71.1–83.1 s) from an earlier 1,584.3 s outlier without claiming a measured cause. We updated three preprint references to verified proceedings records and added the statistical-method references.

The accepted four-page submission and original archives have been preserved. Kidney results remain reserved for the journal extension.
