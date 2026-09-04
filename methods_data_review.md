# Methods and Data Review: Benchmarking Active Spot Selection for Cost-Efficient Spatial Transcriptomics

## Review scope

This document focuses on problems, missing details, ambiguities, and reproducibility concerns in the **Methods** and **Data** sections of the current manuscript. It excludes unfinished figures and numerical Results.

The authors clarified that:

1. The current figures have not yet been updated.
2. The experimental results are not yet complete.
3. The present study is intended as a retrospective label-subsampling benchmark: it tests whether learning can work with fewer measured spots and, if successful, could motivate future customized capture regions.

This clarification resolves the largest conceptual objection. A retrospective label-subsampling benchmark is valid, but the paper must distinguish the computational experiment from future wet-lab or assay-level implementation.

### Recommended scope statement

> This study retrospectively simulates spot-level label acquisition by masking measurements in fully profiled spatial-transcriptomics datasets. It tests whether selecting a limited number of spot-level expression vectors yields a better morphology-to-expression predictor than selecting the same number uniformly at random. The study does not directly demonstrate reduced wet-lab or sequencing cost; rather, it evaluates the computational premise needed to motivate future customized capture regions or region-selective assays.

# Action checklist organized by the final subsections

## Methods

### 2.1 Benchmark Setting and Problem Formulation

- [ ] State in the opening paragraph that the experiment is a **retrospective label-masking simulation**, not prospective physical acquisition.
- [ ] Define one acquired label as the complete expression vector for one spot and state that the budget counts spots, not genes, reads, patches, or scalar labels.
- [ ] Define all notation, including the candidate pool, selected pool, \(N\), \(G\), budget fraction \(\beta\), selected set, expression vector, and prediction model.
- [ ] List the information available before selection: H&E patches, spatial coordinates, slide/patient identifiers, and no expression values for unselected spots.
- [ ] State whether the candidate pool is constructed globally across the training fold or separately within each slide or patient.
- [ ] Replace claims such as “5% of a section” with “5% of the fold-wide training spot pool” when that is the actual sampling unit.
- [ ] Clarify the intended use case: retrospective training-set construction rather than prospective selection within a new patient.
- [ ] Use “pool-based active spot selection for multivariate regression” when a precise description is required.
- [ ] Separate the evaluated spot-selection task from the future goal of designing contiguous customized capture regions.
- [ ] Avoid claiming that the benchmark directly demonstrates lower wet-lab or sequencing cost.

### 2.2 Multi-round Spot Selection Protocol

- [ ] Define the initial random round, warm-up period, acquisition interval, cutoff rule, number of rounds, and per-round quota.
- [ ] State whether selection at a trigger epoch occurs before or after that epoch's training.
- [ ] Report the exact acquisition epochs and ensure they agree with \(W\), \(I\), and \(C\).
- [ ] Ensure that all strategies receive the same final number of selected spots within each fold and budget.
- [ ] Handle the remainder in \(\beta N\) explicitly rather than silently rounding down at every round.
- [ ] Report nominal and actual selected-spot counts for every fold and budget.
- [ ] State how much of the final pool is controlled by each strategy after the shared random initialization.
- [ ] Explain that TOD has fewer informed acquisition rounds, or modify the protocol to equalize its effective active budget.
- [ ] Clarify that training continues across acquisition rounds and that the model is not reinitialized after each round.
- [ ] Decide whether the estimand is performance at convergence or performance under a fixed training/computational horizon.
- [ ] Report total optimizer steps or processed training examples per budget; epoch counts alone are not comparable across different pool sizes.
- [ ] Consider a one-shot selection baseline if the goal includes measuring the value of iterative acquisition.

### 2.3 Spot Selection Strategies

- [ ] Present Random as the primary reference strategy and cite every non-random baseline.
- [ ] Use consistent names and capitalization throughout the text, figures, and tables: Random, MC-dropout, TOD, CoreSet, and TypiClust.
- [ ] Give a mathematical selection score or unambiguous algorithmic definition for every strategy.
- [ ] Specify the sampling unit, candidate pool, ranking direction, tie handling, and sampling-without-replacement rule.
- [ ] Define the feature layer, feature normalization, and distance metric used by CoreSet and TypiClust.
- [ ] Define the active dropout layer, stochastic-pass count, variance estimator, and batch-normalization behavior for MC-dropout.
- [ ] Define the two checkpoints and cache alignment used by TOD.
- [ ] State how MC-dropout and TOD are adapted to a 300-output regression problem and how gene scale affects their scores.
- [ ] Verify whether the TypiClust implementation matches the published algorithm; otherwise call it “TypiClust-inspired.”
- [ ] Document small-cluster, empty-cluster, duplicate-feature, and nearest-neighbor edge cases for TypiClust.
- [ ] State that coordinates are not used for selection if acquisition is based only on image features or predictions.
- [ ] Use deterministic image preprocessing during candidate scoring.
- [ ] Report both per-round and total computational cost for each strategy.

### 2.4 Gene-expression Prediction Model

- [ ] Describe the predictor as an adaptation of ST-Net if its architecture, targets, or preprocessing differ from the original method.
- [ ] Specify the complete layer sequence, including the DenseNet-121 representation, added linear layer, ReLU, dropout, and output layer.
- [ ] Define the \(224 \times 224\) H&E input patch and the \(G=300\) expression output.
- [ ] Identify the pretrained weights, software implementation, and version.
- [ ] State which layers are trainable and when fine-tuning begins.
- [ ] Define the target representation and any library-size normalization, log transformation, clipping, or gene-wise scaling.
- [ ] Explain how target scale affects MSE training and the uncertainty-based selection scores.
- [ ] Specify the loss, optimizer, momentum, Nesterov setting, weight decay, batch behavior, and learning-rate schedule.
- [ ] Define whether learning-rate time counts epochs or optimizer steps and how the schedule behaves as the labeled pool grows.
- [ ] State whether the final epoch or a separately selected checkpoint is evaluated.
- [ ] Ensure that held-out patients are not used for early stopping, checkpoint selection, or hyperparameter tuning.
- [ ] Explain why a fixed predictor is used across strategies and limit conclusions to that predictor unless additional architectures are tested.

## Data and Experimental Design

### 3.1 Datasets and Preprocessing

- [ ] Provide the exact HEST release, download date, cohort identifiers, slide identifiers, and original-study citations.
- [ ] Name the platform, chemistry, spot diameter, spot pitch, image resolution, and tissue/disease context for every cohort.
- [ ] Report patients, slides, spots, and genes before and after filtering.
- [ ] Define “patient,” “slide,” “section,” “capture area,” “spot,” and “gene” consistently.
- [ ] Provide all spot-, gene-, tissue-, and image-quality filtering rules and state whether they are applied within training folds.
- [ ] Define the exact expression source and preprocessing order, including the AnnData field or layer used.
- [ ] Prevent held-out-patient information from entering filtering, normalization, gene selection, or stain normalization.
- [ ] Justify the target-gene rule and publish the exact 300-gene panel for every cohort.
- [ ] If highly variable genes are discussed, document the method and use them only for post hoc characterization unless selected within the training fold.
- [ ] Define patch extraction, pyramid level, microns per pixel, physical footprint, resizing, interpolation, padding, and tissue/background handling.
- [ ] Describe image-to-expression registration and quality control.
- [ ] Define the coordinate system used for spatial evaluation and distinguish physical patch overlap from morphological redundancy.
- [ ] Provide public-data, ethics, licensing, and accession information.
- [ ] Remove all “TK” dataset values from the submitted manuscript or clearly identify the corresponding cohort as planned rather than completed.

### 3.2 Experimental Protocol and Implementation Details

- [ ] Report all strategy–budget–fold configurations and the resulting number of training runs.
- [ ] Define the budget set and give actual selected-spot counts for every fold.
- [ ] Publish exact patient-level fold assignments and confirm that all sections from a patient remain in the same fold.
- [ ] State whether evaluation slides function as validation data, test data, or both.
- [ ] Use an internal training-patient validation split for convergence or hyperparameter choices when needed.
- [ ] Consolidate epoch counts, batch size, learning rate, augmentation, acquisition parameters, and hardware here rather than repeating them across sections.
- [ ] Define every random seed and what it controls: initialization, data loading, augmentation, sampling, dropout, clustering, and CUDA behavior.
- [ ] State precisely which configurations share model initialization or other random states.
- [ ] Use repeated training seeds and repeated Random acquisitions when effects are small.
- [ ] Report software/library versions and determinism settings.
- [ ] Document training and scoring transforms completely and keep candidate scoring deterministic.
- [ ] State how large slides and patients are weighted during acquisition, training, and evaluation.
- [ ] Report exclusions, failures, incomplete runs, and deviations from the planned protocol.
- [ ] Report hardware, total training time, total selection time, and whether feature-extraction time is included.

### 3.3 Evaluation Metrics

- [ ] Rename “Evaluation targets” to “Evaluation Metrics.”
- [ ] Define the primary metric as mean per-gene within-slide Pearson correlation coefficient and state the full aggregation hierarchy.
- [ ] Define how constant genes and undefined correlations are handled and ensure identical included genes/slides across strategies.
- [ ] State whether correlations are averaged directly or after Fisher \(z\)-transformation.
- [ ] Describe ARI as expression-cluster agreement unless external biological domain annotations are available.
- [ ] Specify expression normalization, dimensionality reduction, cluster count, initialization, and seed used before ARI calculation.
- [ ] Justify \(k=10\) or provide sensitivity analysis across multiple cluster counts.
- [ ] Fully define the Moran's \(I\) graph: coordinate system, neighbors, symmetrization, weights, row normalization, boundaries, and disconnected components.
- [ ] Define how Moran's \(I\) fidelity is calculated and aggregated and how undefined values are handled.
- [ ] State that Moran's \(I\) fidelity can range from −1 to 1.
- [ ] Define acquisition-time measurement as a reproducible efficiency metric, including pool size, hardware, repetitions, and included operations.
- [ ] Mention every reported metric, table, and figure in the main text using consistent `~\ref{}` formatting.

### 3.4 Statistical Analysis

- [ ] Define the patient as the experimental unit and explain the dependence induced by overlapping cross-validation training sets.
- [ ] Predefine the primary matched comparison against Random and the direction of every reported difference.
- [ ] Report means, standard deviations, and uncertainty intervals with a consistent number of decimal places.
- [ ] Do not pool strategy–fold cells as independent observations when they share folds and a common Random baseline.
- [ ] Label sign and Wilcoxon tests as descriptive/exploratory if the dependence problem is not resolved.
- [ ] Prefer per-fold strategy aggregation, hierarchical bootstrap, permutation procedures, or repeated-seed mixed models when feasible.
- [ ] State that analyses are exploratory if no multiplicity adjustment is applied across strategies, budgets, and metrics.
- [ ] Define how incomplete configurations and failed runs are handled before inspecting outcomes.
- [ ] Avoid formal significance claims that are unsupported by four patient-level folds.
- [ ] Release per-fold results, selected-spot identifiers, exact fold assignments, and analysis code.

## Cross-section consistency checks

- [ ] Introduce each abbreviation once and use only the abbreviation afterward.
- [ ] Keep terminology, symbols, model capitalization, budget units, and metric names consistent across text, equations, figures, and tables.
- [ ] Cite every baseline method in both the Methods text and comparison tables.
- [ ] Use LaTeX quotation marks as ``text'' and do not use em dashes.
- [ ] Use nonbreaking spaces in citations and references: `~\cite{}`, `Fig.~\ref{}`, `Table~\ref{}`, and `Eq.~\ref{}`.
- [ ] Use “(Ours)” only for a genuinely proposed method; the benchmarked published selection strategies are not “Ours.”
- [ ] Bold only the best completed values in each table and use consistent decimal precision and units.
- [ ] Mention every figure, table, and equation in the text.
- [ ] Verify every bibliography entry against Google Scholar and the final publisher record; cite the final peer-reviewed version instead of a duplicate preprint when available.
- [ ] Keep the study title and absence of a benchmark name consistent throughout the manuscript.

# Detailed Methods issue inventory

## A. Problem formulation and scope

1. **The retrospective simulation is not stated explicitly enough.**  
   The Problem Setting reads like prospective spot acquisition. State that all expression labels already exist and are masked from the acquisition algorithm.

2. **Define exactly what one “label” means.**  
   One acquired spot apparently reveals its complete \(G=300\)-dimensional expression vector. State that the budget counts spots—not genes, reads, patches, or scalar labels.

3. **State what information is available before acquisition.**  
   Presumably the algorithm has:

   - Every candidate coordinate.
   - Every H&E patch.
   - Slide and patient identities.
   - No expression values for unselected spots.

   This boundary is essential for ruling out leakage.

4. **Separate current evidence from future customized regions.**  
   Current selection operates on individual spots, potentially scattered throughout many slides. It does not select contiguous regions. Customized capture regions should be presented as future translation.

5. **“Percentage of a section” is incorrect.**  
   \(N\) is defined as all spots in a training fold, pooled across multiple patients and slides. Throughout the paper, replace “5% of a section” with “5% of the fold-wide training spot pool.”

6. **Clarify whether acquisition is global or slide-stratified.**  
   If selection is global, a method could concentrate most labels in a few large or unusual slides. Report and preferably constrain:

   - Labels per slide.
   - Labels per patient.
   - Slides receiving zero labels.
   - Whether quotas are proportional to slide size.

7. **Large slides may dominate both training and acquisition.**  
   Uniform spot sampling is not equivalent to uniform slide or patient sampling. Decide which distribution the benchmark is intended to preserve.

8. **The intended deployment population is unclear.**  
   Does the benchmark ask which spots to acquire:

   - Within already-known training slides?
   - Across a new collection of slides?
   - Within every future patient separately?

   The current fold-wide pool corresponds most closely to retrospective dataset construction, not prospective within-patient selection.

9. **The term “active learning” needs qualification.**  
   This is simulated pool-based active learning for multivariate regression. “Retrospective pool-based active spot selection” would be more precise.

10. **The paper does not compare spot selection with region selection.**  
    Therefore it should not imply that disconnected selected spots demonstrate that customized contiguous regions will work. That remains a hypothesis.

## B. Acquisition schedule

11. **The cutoff rule is ambiguous.**  
    With \(C=20\), acquisitions occur at 0, 5, 8, 11, 14, and 17—not 20. Explicitly state that informed acquisition occurs while epoch \(e<C\), rather than “until \(C\).”

12. **Define whether acquisition occurs before or after training at the trigger epoch.**  
    For example, does the epoch-5 query use the model before or after completing epoch 5?

13. **The total budget is systematically rounded down.**  
    The method acquires

    \[
    R\left\lfloor\frac{\beta N}{R}\right\rfloor,
    \]

    not \(\lfloor\beta N\rfloor\). Report the actual count for every fold or distribute the remainder across rounds.

14. **“Identical number of labelled spots” is only approximately tied to \(\beta N\).**  
    It is identical across strategies within a fold, but can be up to \(R-1\) below the nominal budget. Rephrase accordingly.

15. **The first random round dilutes all active strategies.**  
    Active rules control only 5/6 of the final pool. This is disclosed, but the method should also report the initial and per-round counts.

16. **TOD receives a different effective active budget.**  
    TOD controls only four rounds while the other methods control five. This weakens a direct strategy comparison. Consider:

    - Giving TOD an epoch-0 prediction checkpoint.
    - Reporting performance against a baseline with the same number of informed rounds.
    - Calling this an implementation constraint.

17. **The continuous-training protocol should be justified.**  
    The model is not reinitialized between rounds. This makes later selection dependent on earlier optimization history, not only on the accumulated labeled set.

18. **Cross-budget comparisons are confounded by different epoch counts.**  
    Budgets receive 110, 80, 70, and 60 epochs. Consequently, the budget–performance curve reflects both label budget and training schedule.

19. **The adjustment does not equalize optimization.**  
    A 5% pool trained for 110 epochs still processes far fewer labeled examples/gradient updates than a 50% pool trained for 60 epochs. Report total optimizer steps and cumulative training examples per budget.

20. **Low-budget non-convergence is a protocol problem, not only a Results limitation.**  
    Either train to a convergence criterion or define the estimand explicitly as performance under a fixed computational horizon.

21. **No one-shot acquisition baseline is included.**  
    For CoreSet or TypiClust, it would be useful to distinguish benefits of multi-round adaptation from simply selecting the final budget once.

## C. Randomness and experimental control

22. **“Seed 42” is insufficiently defined.**  
    State whether it controls:

    - Model initialization.
    - Data-loader order.
    - Image augmentation.
    - Random sampling.
    - Dropout.
    - k-means.
    - CUDA determinism.

23. **The initialization statement is ambiguous.**  
    “Re-initialised once per fold” could mean one common initialization across all budgets or one per fold–budget combination. State exactly which arms share weights.

24. **Using the same integer seed does not guarantee paired stochastic training.**  
    Different selected pools change data-loader length and random-number consumption. Therefore strategy differences still contain training randomness.

25. **One seed per configuration cannot separate selection effects from seed effects.**  
    Multiple model-training seeds are particularly important when reported effects are approximately 0.005.

26. **The random baseline needs repeated selections.**  
    A single random draw can be unusually good or bad. Random sampling should have several acquisition and training seeds.

27. **Hyperparameter selection is not described.**  
    State whether \(W,I,C,T,p,k,n_{\text{init}}\), epoch counts, and learning rate were chosen before inspecting the reported folds.

## D. Individual acquisition strategies

### Random

28. **Specify the sampling unit and distribution.**  
    Confirm global uniform sampling without replacement across all remaining training spots.

29. **State whether random sampling is stratified.**  
    If not, explain that it preserves the spot-weighted distribution rather than the patient- or slide-weighted distribution.

30. **Specify how ties and residual budget spots are handled.**

### MC-dropout

31. **Provide the exact uncertainty equation.**  
    “Variance averaged over genes” should be formally defined, including whether sample variance uses \(T\) or \(T-1\).

32. **Uncertainty is dominated by gene scale.**  
    Because targets are not gene-wise standardized, high-expression or high-variance genes may dominate the score. Consider standardizing per-gene predictive variance or justify the raw average.

33. **\(T=5\) may give a noisy variance estimate.**  
    Justify it or provide a stability experiment against a larger \(T\).

34. **Identify which dropout layer is active.**  
    The model description contains one dropout layer, but confirm no DenseNet dropout or stochastic layers are also enabled.

35. **Confirm all normalization layers remain fixed.**  
    Saying batch normalization stays in inference mode is good; also clarify whether running statistics were learned only from labeled training spots.

36. **Clarify whether score computation uses raw or transformed predictions.**

### TOD

37. **Define the two model checkpoints precisely.**  
    Specify their epochs and whether both predictions use identical deterministic input preprocessing.

38. **The high-dimensional \(L_2\) score is gene-scale dependent.**  
    As with MC-dropout, large-scale genes can dominate movement.

39. **Clarify whether outputs are cached for every candidate.**

40. **Clarify what happens to newly introduced or removed candidates.**  
    Selected candidates leave the pool; the cache must remain aligned to spot identifiers.

41. **The current implementation is an adaptation to multivariate regression.**  
    The cited TOD work supports task-agnostic use and discusses squared-error regression, but the manuscript should explicitly state how it was adapted to 300-output regression.

### CoreSet

42. **Define the distance metric.**  
    State Euclidean or cosine distance and whether features are \(L_2\)-normalized.

43. **“Trunk features” is ambiguous.**  
    Clarify whether the 1024-dimensional representation is:

    - DenseNet’s output.
    - The added 1024-unit linear layer.
    - Before or after ReLU.
    - Before or after dropout.

44. **Define the initial centers.**  
    Presumably all previously labeled spots serve as centers. State this explicitly.

45. **Clarify whether CoreSet is solved exactly or approximately.**

46. **Feature-space diversity is not spatial diversity.**  
    Later text should not describe CoreSet points as physically “spread out” unless spatial-coordinate dispersion is measured.

47. **Feature normalization can materially change farthest-first selection.**  
    This should be specified and ideally tested.

### TypiClust

48. **The implementation is not the published TypiClust algorithm.**  
    The original method uses self-supervised normalized embeddings and an iterative cluster-prioritization procedure. The current implementation uses task-model features and \(k=\) round quota. Call it “TypiClust-inspired” unless matched to the published method.

49. **The number of clusters differs from the published method.**  
    The original uses \(K=\min(|L|+B,\text{max\_clusters})\), rather than simply \(K=B\). See the [published algorithm](https://proceedings.mlr.press/v162/hacohen22a/hacohen22a.pdf).

50. **Treatment of previously labeled points is unclear.**  
    Are labeled and unlabeled points clustered together, or only candidates?

51. **Small clusters make “20 nearest neighbours” undefined.**  
    Use and document `min(20, cluster_size−1)` or another explicit rule.

52. **State whether the query point itself is excluded from its neighbor set.**

53. **The original method drops very small clusters.**  
    State how clusters containing fewer than five points are handled.

54. **Selecting one point from every cluster may not be possible as described.**  
    Clarify empty clusters, clusters containing only labeled points, and clusters with duplicate/degenerate features.

55. **\(n_{\text{init}}=3\) may be unstable for hundreds of clusters.**  
    Report clustering stability or justify the low setting.

56. **Specify standard `KMeans` versus `MiniBatchKMeans`.**

57. **The stated cluster range 81–558 is incomplete or unclear.**  
    Explain that it applies only to particular HER2 folds/budgets and not all strategies or the nominal 50% budget.

58. **Current-model features make TypiClust model-dependent.**  
    The cited method’s self-supervised representation can operate before any labels. Your version cannot, which explains the random first round and should be discussed as a methodological difference.

## E. Prediction model and training

59. **This is an adapted ST-Net, not necessarily the original ST-Net.**  
    The original work predicts 250 selected genes and uses its own preprocessing. Your architecture and 300 alphabetical genes should be described as “ST-Net-style” or “an adaptation of ST-Net.”

60. **The added 1024-unit head needs a precise equation or layer sequence.**

61. **Specify the pretrained weight source and version.**  
    For example, torchvision ImageNet-1K weights and exact software version.

62. **Clarify whether all DenseNet layers are trainable from epoch 0.**

63. **Learning-rate time \(t\) is undefined.**  
    State whether \(t\) counts epochs or optimizer steps and define \(T_{\max}\).

64. **The learning-rate schedule changes after each acquisition only indirectly.**  
    Explain whether it continues uninterrupted when the pool grows.

65. **Specify whether weight decay applies to biases and normalization parameters.**

66. **Clarify whether SGD uses Nesterov momentum.**

67. **Define batch behavior when the labeled pool is smaller than 80.**  
    State whether the last batch is retained and whether sampling uses replacement.

68. **Larger slides probably contribute more gradient updates.**  
    If spots are sampled uniformly, the model optimizes a spot-weighted objective. Consider slide-balanced batches or state this choice explicitly.

69. **Target scaling for MSE is not described.**  
    If gene values are not standardized, high-variance genes dominate the training loss.

70. **Any gene-wise normalization parameters must be fitted without held-out patients.**  
    Report whether mean/SD statistics are learned from labeled spots, all training spots, or the full cohort.

71. **The augmentation “random quarter turn” is ambiguous.**  
    State the possible angles and probabilities.

72. **Color/stain augmentation or normalization is absent.**  
    This is not necessarily wrong, but stain variability may dominate selection uncertainty and should be acknowledged.

73. **Candidate scoring and evaluation transforms need exact definitions.**  
    Include resizing, interpolation, normalization mean/SD, cropping, and padding.

74. **Training checkpoint selection is unspecified.**  
    Are predictions taken from the final epoch or the best validation checkpoint? The held-out patient fold must not be used for early stopping or model selection.

## F. Evaluation methods

75. **PCC handling of constant genes is missing.**  
    Pearson correlation is undefined when predicted or measured values are constant across a slide.

76. **Specify whether undefined genes are dropped per slide.**  
    If so, strategies may be averaged over different gene sets.

77. **Specify whether correlations are averaged directly or after Fisher \(z\)-transformation.**

78. **Clarify the weighting hierarchy.**  
    Current text suggests gene mean → slide mean → fold mean. State whether every slide is equally weighted and whether patients with more slides receive more weight.

79. **Call the metric “mean per-gene within-slide PCC.”**  
    “Per-slide PCC” alone is ambiguous.

80. **Spatial-domain recovery is actually cluster agreement.**  
    Without biological annotations, ARI compares two data-driven k-means partitions. “Expression-cluster agreement” is safer than “domain recovery.”

81. **Expression preprocessing before k-means is missing.**  
    State whether genes are centered, scaled, normalized, or PCA-reduced.

82. **The choice \(k=10\) is unsupported.**  
    Justify it and preferably test sensitivity to several \(k\) values.

83. **Separate k-means fits require explicit seed handling.**

84. **NMI is said to track ARI but is not shown.**  
    Either provide it in supplementary results or remove the unsupported statement.

85. **The Moran graph construction is incomplete.**  
    Specify:

    - Coordinate system.
    - Euclidean versus grid distance.
    - Directed or symmetric six-nearest-neighbor graph.
    - Binary or distance weights.
    - Row standardization.
    - Treatment of disconnected components and boundaries.

86. **Moran’s \(I\) aggregation is unclear.**  
    Explain whether \(I\) is calculated per gene per slide, how the gene-wise correlation is formed, and how slide/fold values are averaged.

87. **Name the fidelity correlation.**  
    It appears to be Pearson correlation across genes, but this is not stated.

88. **Handle genes with undefined Moran’s \(I\).**

89. **Moran fidelity ranges from −1 to 1, not 0 to 1.**  
    If describing 1 as perfect, also state that negative fidelity is possible.

90. **All metrics should use identical included slides and genes across strategies.**

## G. Statistical analysis and reproducibility

91. **Strategy–fold pairs cannot be pooled as independent observations.**  
    They share folds, training data, and a common random baseline.

92. **The reported sign and Wilcoxon tests are therefore not formal valid tests.**  
    If retained, label them descriptive only. Better options include per-fold strategy aggregation, hierarchical bootstrap, or repeated-seed mixed models.

93. **Four folds are too few for reliable single-cell inference.**

94. **Multiple strategies, budgets, and metrics create multiplicity.**  
    Either correct for it or state that all analyses are exploratory.

95. **Report confidence intervals, not only means and fold SDs.**

96. **CV folds overlap heavily in training patients.**  
    Test patients are disjoint, but fold estimates are not fully independent because training sets overlap.

97. **The experimental unit should be the patient, not the strategy–fold cell.**

98. **Release exact per-fold values and selected spot identifiers.**

99. **Report exclusions and failed/incomplete runs prospectively.**

100. **Record software versions and determinism settings.**

101. **Acquisition cost should include implementation details.**  
     Report hardware, pool size, batch size, repetitions, warm-up exclusion, CPU/GPU use, and whether feature extraction is included.

102. **Cost per round is not total method cost.**  
     Also report total selection time and total training-plus-selection time.

# Detailed Data issue inventory

## A. Dataset identity and provenance

1. **Each cohort needs an exact HEST identifier list.**  
   “HER2,” “cSCC,” and “kidney” are not sufficient to reproduce the cohort.

2. **Specify the HEST release/version and download date.**  
   HEST has continued changing, so results may differ across releases. Cite the exact frozen release used. See the [HEST repository](https://github.com/mahmoodlab/hest).

3. **Provide accession numbers or original download links.**

4. **Name the exact experimental platform for every cohort.**  
   “Barcoded-array platform” is too broad. State the original Spatial Transcriptomics/Visium version, chemistry, spot diameter, and pitch.

5. **The kidney source is not identified.**  
   Add its original study citation, HEST IDs, platform, organ/disease state, and sample counts before including it as a dataset.

6. **“Kidney is TK” is not acceptable in the Methods section.**  
   If the dataset has not been finalized, call it a planned cohort and remove it from the completed Methods counts.

7. **Clarify whether “patients” means unique human donors.**

8. **Clarify the relationship between patients, tissue sections, capture areas, and slides.**

9. **State whether multiple sections come from adjacent levels of the same tissue block.**  
   Adjacent sections can be highly correlated, making patient-level grouping essential.

10. **Provide ethics/data-governance language.**  
    Since the data are public and de-identified, state that no new human-subject data were collected and refer to the original studies’ approvals.

11. **Check the HEST license and conditions for derived-data/code release.**

## B. Counts and filtering

12. **Define what “Spots” means in Table 1.**  
    Is it:

    - All array locations?
    - Tissue-covered spots?
    - Spots remaining after QC?
    - Spots with nonzero expression?
    - Spots with matched image patches?

13. **Define what “Genes” means.**  
    The reported 967 and 392 are presumably retained genes after intersection/filtering, not all measured genes.

14. **Provide the complete filtering pipeline.**  
    Include thresholds for:

    - Minimum genes per spot.
    - Minimum counts per spot.
    - Minimum spots per gene.
    - Mitochondrial content.
    - Tissue coverage.
    - Missing image patches.
    - Duplicate gene symbols.

15. **Report counts before and after every filtering step.**

16. **Explain whether filtering is performed globally or within each training fold.**  
    Expression-dependent filtering using held-out patients can leak test-set information.

17. **State whether every slide has the same retained gene panel.**

18. **Explain how missing genes across slides are handled.**

19. **Provide per-patient and per-slide spot counts.**  
    Ranges for validation folds are not enough.

20. **Large variation in slide spot counts may bias the benchmark.**  
    Report the distribution and explain whether large slides are intentionally weighted more heavily.

## C. Expression preprocessing

21. **The expression representation needs a reproducible definition.**  
    “\(\log(1+x)\) raw counts” should identify the exact AnnData field: `X`, `raw.X`, or a named layer.

22. **The manuscript says HEST documentation disagrees with the stored values.**  
    This is important and needs a reproducible verification procedure, dataset version, and numerical tolerance.

23. **Provide the equation and operation order.**  
    For example:

    \[
    y_{ig}=\log(1+c_{ig})
    \]

    with no library-size normalization, scaling, clipping, or gene-wise standardization.

24. **Verify this preprocessing separately for every cohort.**  
    Do not infer cSCC or kidney preprocessing from HER2.

25. **Library-size confounding is only quantified for HER2.**  
    Repeat the correlation analysis for cSCC and kidney.

26. **Define “total counts.”**  
    It should be calculated from pre-log raw counts across which gene set—the full transcriptome or retained genes?

27. **Unnormalized targets make MSE sensitive to sequencing depth.**  
    This affects both prediction and acquisition scores.

28. **Consider a normalized-target sensitivity analysis.**  
    At minimum compare against library-size normalization followed by `log1p`.

29. **If gene-wise standardization is used internally, document it.**  
    If not, explicitly state that raw gene scales enter MSE, MC-dropout, and TOD.

30. **Any normalization fitted from data must exclude held-out patients.**

## D. Target-gene definition

31. **Alphabetical gene selection is arbitrary and biologically difficult to defend.**  
    It is acceptable for a stress test, but the rationale must be explicit.

32. **Explain why the first 300 genes were chosen rather than all shared genes or training-fold HVGs.**

33. **Confirm that the AnnData gene order is genuinely alphabetical.**  
    Do not assume file order is alphabetical without checking.

34. **Publish the exact 300-gene list for each cohort.**

35. **The target panels differ across cohorts.**  
    Cross-cohort absolute PCC values are therefore not comparable. This is already acknowledged, but should also influence how pooled conclusions are written.

36. **The selection fractions represent different target coverage.**  
    cSCC uses 300 of 392 retained genes, while HER2 uses 300 of 967. These are qualitatively different prediction tasks.

37. **Report low-expression and constant-gene frequencies.**

38. **Some alphabetical genes may have insufficient variation for PCC.**  
    Define an inclusion threshold before evaluation.

39. **The statement that 92 HER2 targets overlap the top-300 HVGs requires a precise HVG method.**  
    Specify normalization, software, flavor, and whether HVGs were computed using the full cohort or training data only.

40. **HVG computation on the full cohort would use held-out expression.**  
    This is acceptable only as post hoc panel characterization, not target selection or model preprocessing.

41. **Gene identifiers need standardization.**  
    State whether targets use gene symbols, Ensembl IDs, or converted identifiers and how duplicates are resolved.

## E. H&E images and patches

42. **Physical patch size is missing.**  
    `224 × 224` pixels is not enough; report microns per pixel and footprint in micrometers.

43. **The non-overlap claim requires physical units.**  
    Compare patch width in micrometers against center-to-center spot pitch for each platform.

44. **Specify the image pyramid level used.**

45. **Explain whether patches are extracted before or after HEST resampling.**

46. **Specify resizing and interpolation.**

47. **Describe behavior near image boundaries.**  
    Are patches padded, cropped, or excluded?

48. **Describe tissue/background filtering.**

49. **State whether spot coordinates refer to patch centers after all registration transformations.**

50. **Report image-to-expression alignment QC.**  
    Misregistration could produce uncertainty that active methods preferentially select.

51. **Scanner and staining information should be included when available.**

52. **Cohort-specific color differences may act as shortcuts.**  
    State whether stain normalization is performed and whether it is fitted without test slides.

53. **Patches from neighboring spots may overlap or be strongly correlated.**  
    Even if they do not physically overlap, morphology may remain highly redundant; distinguish these concepts.

## F. Spatial coordinates and candidate structure

54. **Specify the coordinate system used for Moran’s \(I\).**

55. **Confirm whether coordinates are original array coordinates or image-pixel coordinates.**

56. **Explain how spatial graphs handle irregular tissue boundaries and missing spots.**

57. **The acquisition methods do not appear to use coordinates.**  
    State this explicitly: selection is image-feature/output based, while coordinates are used only for evaluation.

58. **If customized regions are the future goal, add a region-level analysis later.**  
    Useful quantities would include selected-spot contiguity, number of connected components, physical coverage area, and coverage per tissue region.

## G. Cross-validation and leakage

59. **Publish exact fold assignments.**

60. **Confirm that every section from a patient remains in the same fold.**  
    This is stated, but the identifiers should allow verification.

61. **Clarify whether preprocessing occurs before or after splitting.**

62. **Gene filtering, target scaling, stain normalization, and model-selection decisions must not use held-out patient data.**

63. **The same held-out folds appear to serve as validation and final test data.**  
    If hyperparameters were chosen using these results, performance is no longer an unbiased test estimate.

64. **Consider an internal training-patient validation split.**  
    Use it for convergence, epoch selection, and hyperparameter tuning while preserving the held-out patients for final evaluation.

65. **cSCC has only four patients.**  
    Leave-one-patient-out CV is appropriate but highly uncertain and cannot support broad cohort-level claims.

66. **HER2 folds contain two held-out patients, but slide weighting may still become patient-imbalanced.**

67. **Patient identity may be predictable from stain/scanner style.**  
    Patient-level splitting prevents direct leakage but does not eliminate cohort-specific technical effects.

# Highest-priority Methods and Data fixes

Before revising the Results, address these items first:

1. Explicitly frame the study as retrospective label masking.
2. Correct the budget unit from “section” to “fold-wide training pool.”
3. Decide global versus slide-stratified acquisition and report per-slide coverage.
4. Document the exact TypiClust adaptation or rename it.
5. Define expression preprocessing and prevent fold leakage.
6. Specify physical patch size and platform pitch.
7. Fully define PCC, ARI, and Moran’s \(I\).
8. Add repeated training and random-selection seeds.
9. Publish exact cohort IDs, slide IDs, fold assignments, and gene panels.
10. Distinguish current spot selection from future customized-region development.
