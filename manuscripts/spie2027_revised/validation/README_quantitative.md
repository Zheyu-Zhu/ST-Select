Quantitative audit for the revised conference manuscript
======================================================

This audit covers HER2 and cSCC, seed 42: 160 budgeted configurations and eight full-label references. It reads stored CSVs, text logs, and TensorBoard scalar events. It does not train, load model checkpoints, recompute expression/spatial metrics, or alter the external ST-Net tree.

Reproduce from the repository root in PowerShell:

```powershell
& C:\Users\zheyu\anaconda3\python.exe manuscripts\spie2027_revised\validation\audit_quantitative.py
& C:\Users\zheyu\anaconda3\python.exe manuscripts\spie2027_revised\validation\audit_learning_rate.py
```

The first script requires numpy/pandas/scipy; the second uses the standard library. Analysis environment versions in the JSON describe the analysis interpreter, not the training environment. Source hashes preserve provenance. Outputs are written beside the scripts.

How values are aggregated
-------------------------

Within each configuration, the saved slidewise PCC values are averaged equally to produce a fold value. Reported PCC is the mean of four fold values; SD uses `ddof=1`. A paired difference subtracts Random on the same fold and budget. The four active strategies' mean differences and their 16 strategy-fold sign counts are descriptive summaries; they are not 16 independent observations.

`pcc_fold_values.csv`, `pcc_summary.csv`, `paired_pcc_fold_values.csv`, and `paired_pcc_exploratory.csv` contain the authoritative values. `spatial_descriptive_summary.csv` summarizes already-computed ARI/NMI/Moran fidelity without additional hypothesis tests. `original_pcc_precision_audit.csv` checks the preserved original tables: all 40 budgeted PCC means agree at four decimals; **19 HER2 SD cells do not**. Most were rounded to three decimals then padded with a zero. Use `pcc_summary.csv` to replace them.

Pooled PCC descriptions
-----------------------

| Cohort | 5% delta; wins | 10% delta; wins | 30% delta; wins | 50% delta; wins |
|---|---:|---:|---:|---:|
| HER2 | -0.0176; 5/16 | -0.0117; 4/16 | +0.0055; 10/16 | +0.0057; 12/16 |
| cSCC | -0.0089; 7/16 | -0.0087; 4/16 | -0.0016; 8/16 | +0.0028; 13/16 |

The exact HER2 30% mean delta is **0.005537615772078024**, which rounds to **+0.0055**, not +0.0056. This is computed from unrounded fold values; subtracting rounded table means can change the final digit.

Exploratory statistical sensitivity
-----------------------------------

For each of the 32 cohort × budget × active-strategy PCC comparisons, let `d` be the four fold differences and `s²` their sample variance. The implemented correction is `SE = sqrt((1/4 + 1/3) s²)`, `t = mean(d)/SE`, with three degrees of freedom and a two-sided tail. The nominal held-out/training patient ratio is 1/3 in both cohorts. Holm adjustment is over all 32 PCC comparisons together. Approximate 95% intervals in the CSV are pointwise, not simultaneous.

The smallest unadjusted corrected-test p-value is **0.1279724005** (HER2 CoreSet at 5%). **All 32 Holm-adjusted p-values are 1.0.** This is a sensitivity calculation, not calibrated confirmatory inference for this study: one fixed four-fold patient partition and one training seed supply little information about split/optimization variability. Overlapping training sets, heterogeneous grouped patients, and nonlinear PCC also limit the approximation. Non-rejection establishes neither equivalence nor noninferiority. Do not replace the descriptive primary analysis with pooled sign/Wilcoxon tests.

Suggested manuscript text:

> We report paired active-minus-Random differences and fold sign counts descriptively. The four cross-validation training sets overlap, each configuration uses one training seed, and the active strategies share the Random comparator. As an exploratory sensitivity analysis, we applied a corrected cross-validation t-test with standard error sqrt((1/4+1/3)s²), three degrees of freedom, and Holm correction over the 32 PCC comparisons. No comparison was rejected at 0.05. This approximate analysis does not establish equivalence or account for unmeasured seed and partition variability.

Primary methodology sources, read during the audit:

- Nadeau, C., and Bengio, Y. *Inference for the Generalization Error*. Machine Learning 52, 239–281 (2003), DOI [10.1023/A:1024068626366](https://doi.org/10.1023/A:1024068626366). The [earlier primary NIPS paper](https://papers.nips.cc/paper_files/paper/1999/file/7d12b66d3df6af8d429c1a357d8b9e1a-Paper.pdf), Section 3, derives the surrogate variance correction and notes that it can overestimate or underestimate the true variance.
- Bouckaert, R. R., and Frank, E. *Evaluating the Replicability of Significance Tests for Comparing Learning Algorithms*. PAKDD 2004, pp. 3–12, DOI [10.1007/978-3-540-24775-3_3](https://doi.org/10.1007/978-3-540-24775-3_3). [Author PDF](https://ml.cms.waikato.ac.nz/publications/2004/bouckaert-frank.pdf), Sections 3.2–3.3, gives the corrected statistic and CV extension. [University citation record](https://researchcommons.waikato.ac.nz/entities/publication/2cdb0c8d-f868-43fb-81df-605df1de6b56).
- Holm, S. *A Simple Sequentially Rejective Multiple Test Procedure*. Scandinavian Journal of Statistics 6(2), 65–70 (1979), [original publication record](https://www.jstor.org/stable/4615733). No independence assumption among these 32 comparisons is added by the adjustment; valid familywise inference would still require valid underlying p-values.

Equal-epoch reference and actual learning rate
----------------------------------------------

At epoch 50, HER2 Random 50% is **0.2357 ± 0.1362**, versus full-label **0.2416 ± 0.1460**; the paired full-minus-half difference is **+0.0058382217**, positive on **3/4** folds. On cSCC the values are **0.1087 ± 0.0227**, **0.1160 ± 0.0220**, and **+0.0073272313**, positive on **4/4** folds. `epoch50_matched_*` files retain all four paired observations. These are equal-epoch comparisons, not equal optimizer exposure: at epoch 50 the full-label runs have approximately twice as many optimizer steps.

The original manuscript's epoch-horizon learning-rate formula is incorrect. `ST-Net.py` uses `lr = 1e-4 * (1-i/100000000)^0.9`, where `i` is the optimizer-step index. `run_her2_cscc.sh` explicitly supplies 100000000; no other `run_*.sh` supplies a different `max_iterations`. The separate scalar audit verifies the formula against **all 16** Random 50%/full-label runs across both cohorts, and verifies event step counts against the epoch batch totals in their text logs.

HER2 half/full step totals at epoch 50 are respectively `(2536,2536,2865,2578)` / `(5028,5028,5721,5145)`; cSCC totals are `(1804,1353,1804,1715)` / `(3597,2695,3600,3430)`. Learning rates remain within 0.0052% of the initial rate in these reference comparisons. A different 60-vs-50 epoch normalization of the decay schedule is therefore **not** the relevant caveat. `learning_rate_and_steps.csv` records exact scalar values and source paths.

Last-ten-epoch changes
----------------------

Define the reported percentage as `100*(final cohort/fold-mean PCC - PCC ten epochs earlier)/final PCC`, calculated separately for each strategy and budget. These tail changes do not prove convergence.

| Budget | Range across both cohorts and five strategies |
|---|---:|
| 5% | +10.1071788% to +27.8301548% |
| 10% | +4.5883607% to +23.0357834% |
| 30% | -3.2991857% to +3.4728448% |
| 50% | -1.9527335% to +3.9621271% |

Consequently, the original statement that changes are below 2.5% at 30%/50% is false. The maximum absolute change is below 4.0% for these **strategy-level four-fold means**, not necessarily for individual folds. HER2 alone has a +2.5637% TypiClust-inspired 50% change. Full details are in `learning_last10_*`.

Acquisition timing
------------------

At HER2 50%, the ranges of four fold-median logged scoring times are CoreSet **10.7–12.3 s**, MC-dropout **37.1–41.8 s**, and TOD **7.4–8.4 s**. TypiClust-inspired fold medians are **1584.3, 83.1, 80.5, and 71.1 s**. The last three runs have latest TensorBoard start timestamps on August 14, 2026; fold 0 dates to August 9. Thus **71.1–83.1 s** is the range for those three later runs, not all four folds and not a range of all individual rounds. Their round quotas are 817, 930, and 838. The handoff's wording that all clean measurements have the same cluster count of 817 should not be repeated.

Full timing logs and date differences document heterogeneous conditions; no load-monitoring measurements establish their cause. Report the subset and conditions explicitly rather than treating 1584 s as intrinsic scaling cost. `acquisition_timing_rounds.csv` retains logged round/pool/quota conditions and `acquisition_timing_summary.csv` retains medians, ranges, and cumulative scoring time. These totals exclude training and initial Random selection.

The dependency-free scalar reader follows the official [Event protobuf](https://raw.githubusercontent.com/tensorflow/tensorboard/master/tensorboard/compat/proto/event.proto), [Summary protobuf](https://raw.githubusercontent.com/tensorflow/tensorboard/master/tensorboard/compat/proto/summary.proto), and [TFRecord framing](https://www.tensorflow.org/tutorials/load_data/tfrecord). It checks record lengths and consistency with log batch totals, but does not validate CRCs.
